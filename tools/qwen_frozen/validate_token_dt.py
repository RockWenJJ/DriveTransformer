"""Real-data wiring audit; --gpu adds one optimizer step and causal branch checks."""
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root)); os.chdir(root)
    torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0); random.seed(0)
    from mmcv import Config
    from mmcv.models import build_model
    from mmcv.datasets import build_dataset
    from mmcv.parallel import collate
    from mmcv.utils import wrap_fp16_model
    import adzoo.drivetransformer.mmdet3d_plugin
    configs = root/'adzoo/drivetransformer/configs/drivetransformer'
    cfg = Config.fromfile(str(configs/'drivetransformer_qwen08_learntoken_10ep.py'))
    base = Config.fromfile(str(configs/'drivetransformer_res480_no_map_10ep.py'))
    for name in ['data', 'optimizer', 'lr_config', 'runner', 'inference_contract', 'fp16']:
        assert cfg[name] == base[name], name
    assert cfg.data.samples_per_gpu == 20 and cfg.runner.max_iters == 29350
    assert cfg.load_from is None and cfg.resume_from is None
    settings = dict(cfg.model.img_backbone)
    for name in ['type', 'out_channels', 'worker_python', 'timeout', 'token_seed']:
        settings.pop(name)
    settings['memory_fraction'] = 0.16
    (args.output.parent/'worker_validation_settings.json').write_text(json.dumps(settings, indent=2))
    dataset = build_dataset(cfg.data.train)
    proof = dict(passed=False, train_frames=len(dataset), train_routes=len(dataset.routes_names),
        budget=dict(gpus=4, microbatch=20, updates=29350, seed=0),
        baseline_data_optimizer_lr_inference_unchanged=True)
    if not args.gpu:
        camera_count = 0
        for route in dataset.routes_names:
            with open(Path('data/infos')/dataset.infos_dir_name/(route+'.pkl'), 'rb') as stream:
                records = pickle.load(stream)
            for record in records:
                cameras = [k for k in record['sensors'] if 'CAM' in k]
                assert cameras == cfg.qwen_contract.camera_order, (route, cameras)
                camera_count += 1
        proof['camera_order_verified_frames'] = camera_count
        assert camera_count == len(dataset)
    # Sample frame 25 has valid history and future and is not a route boundary.
    sample = dataset[25]
    info = dataset.get_data_by_index(25)
    proof['sample'] = dict(dataset_index=25, scene=info['folder'], frame_idx=int(info['frame_idx']))
    assert tuple(sample['img'].data.shape) == (1,6,3,480,960)
    proof['sample_image_shape'] = list(sample['img'].data.shape)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    state = model.state_dict()
    assert any('traj_branches' in k for k in state)
    assert any('traj_cls_branches' in k for k in state)
    assert not any(k.startswith('pts_bbox_head.map_') for k in state)
    assert tuple(state['img_backbone.q_lang'].shape) == (1,1,1024)
    assert all(not k.startswith('img_backbone.') or k.startswith('img_backbone.visual_projection.') or k == 'img_backbone.q_lang' for k in state)
    model.load_state_dict(state, strict=True)
    proof.update(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        detection_motion_retained=True, learned_q_lang=True, qwen_weights_external=True,
        cpu_state_dict_strict_load=True)
    del state
    if args.gpu:
        torch.cuda.set_per_process_memory_fraction(0.08)
        model.img_backbone.settings['memory_fraction'] = 0.08
        model.cuda(); wrap_fp16_model(model)
        def to_cuda(value):
            if isinstance(value, torch.Tensor): return value.cuda()
            if isinstance(value, dict): return {k:to_cuda(v) for k,v in value.items()}
            if isinstance(value, (list, tuple)): return type(value)(to_cuda(v) for v in value)
            return value
        batch = to_cuda(collate([sample], samples_per_gpu=1))
        model.train()
        from mmcv.optims import build_optimizer
        optimizer = build_optimizer(model, cfg.optimizer)
        token_before = model.img_backbone.q_lang.detach().clone()
        start = time.monotonic()
        try:
            losses = model(batch, return_loss=True)
            loss, values = losses['loss'], losses['log_vars']
            assert all(np.isfinite(v) for v in values.values()), values
            assert 'loss_traj' in values and 'loss_traj_cls' in values
            loss.backward()
            gradients = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    assert bool(torch.isfinite(parameter.grad).all()), name
                    if any(s in name for s in ['visual_projection', 'language_projection', 'language_fusion', 'q_lang']):
                        gradients[name] = float(parameter.grad.float().norm())
            for group in ['visual_projection', 'language_projection', 'language_fusion', 'q_lang']:
                assert any(v > 0 for k,v in gradients.items() if group in k), gradients
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 35))
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            assert not torch.equal(token_before,model.img_backbone.q_lang)
            proof['token_update_max'] = float((token_before-model.img_backbone.q_lang).abs().max())
            proof['token_backward'] = model.img_backbone._client.last_backward_stats
            proof.update(losses=values, grad_norm=grad_norm, adapter_gradients=gradients,
                optimizer_step_seconds=time.monotonic()-start,
                parent_peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                worker=model.img_backbone._client.receipt,
                bridge=model.img_backbone._client.last_stats)
            del losses, loss; gc.collect(); torch.cuda.empty_cache()
            # Isolate language conditioning from the shared visual features and memory.
            model.eval(); head = model.pts_bbox_head
            from drivetransformer_contracts import apply_attention_eval_policy
            apply_attention_eval_policy(model, False)
            with torch.no_grad():
                feats = model.extract_img_feat(batch['img'])[0][:,0]
                language = head._language_context.clone(); head._language_context = None
                data = {k:(v if len(v.shape)==1 else v[:,0]) for k,v in batch.items()
                    if k not in ['img', 'img_metas', 'ego_lcf_feat', 'ego_fut_cmd', 'ego_his_trajs']
                    and hasattr(v, 'shape')}
                timestamp = batch['timestamp']
                if isinstance(timestamp, list): timestamp = np.concatenate(timestamp)
                data['timestamp'] = timestamp if len(timestamp.shape)==1 else timestamp[:,0]
                def predict(hidden):
                    head.reset_memory(); head.set_language_context(hidden)
                    return head(feats, batch['img_metas'], batch['ego_lcf_feat'],
                        batch['ego_fut_cmd'], batch['ego_his_trajs'], **data)
                real = predict(language)
                changed = predict(torch.zeros_like(language))
                for key in ['all_cls_scores','all_bbox_preds','all_traj_preds','all_traj_cls_scores']:
                    assert torch.equal(real[key], changed[key]), key
                for key in ['ego_fut_preds_fix_time','ego_fut_preds_fix_dist']:
                    assert torch.equal(real[key][:-1], changed[key][:-1]), key
                    assert not torch.equal(real[key][-1], changed[key][-1]), key
                proof['language_changes_only_final_planning'] = True
            checkpoint = args.output.parent/'wiring_checkpoint.pth'
            torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                meta=dict(iter=0, qwen_contract=dict(cfg.qwen_contract))), checkpoint)
            model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'], strict=True)
            proof['one_step_checkpoint_strict_load'] = True
        finally:
            model.img_backbone.close()
    proof['passed'] = True
    args.output.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == '__main__':
    main()
