"""Bounded real-data model check. No optimizer step and no checkpoint writes.

Uses smaller images while retaining the complete Base architecture. This is an
integration check, not an accuracy benchmark or full-resolution training test.
"""
import argparse
import copy
import importlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmcv import Config
from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.parallel import DataContainer
from mmcv.parallel.collate import collate
from mmcv.utils import wrap_fp16_model
from drivetransformer_contracts import apply_attention_eval_policy
from tools.ol_eval_tasks_v2 import INPUT_KEYS, to_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-memory-mib', type=int, default=4096)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    free, total = torch.cuda.mem_get_info()
    if free < (args.max_memory_mib + 4096) * 2**20:
        raise RuntimeError('Insufficient free memory for this bounded smoke test')
    torch.cuda.set_per_process_memory_fraction(args.max_memory_mib * 2**20 / total)
    cfg = Config.fromfile(args.config)
    importlib.import_module(os.path.dirname(cfg.plugin_dir).replace('/', '.'))
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    state = torch.load(args.ckpt, map_location='cpu')['state_dict']
    state = {k.replace('model.', '').replace('._orig_mod', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    del state
    wrap_fp16_model(model)
    model.cuda().train()
    train_cfg = copy.deepcopy(cfg.data.train)
    train_cfg.data_aug_conf.update(final_dim=(192, 528), resize_lim=(.32, .345))
    ds = build_dataset(train_cfg)
    batch = to_device(collate([ds[30]], samples_per_gpu=1), 'cuda')
    image_shape = list(batch['img'].shape)
    result = model(batch, return_loss=True)
    assert torch.isfinite(result['loss']), result['log_vars']
    result['loss'].backward()
    gradients = {name: float(param.grad.float().norm()) for name, param in model.named_parameters()
                 if param.grad is not None and ('motion_cls' in name or 'ego' in name)}
    assert gradients and all(np.isfinite(v) for v in gradients.values())
    all_grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in all_grads)
    train_summary = dict(image_shape=image_shape, log_vars=result['log_vars'],
                         parameters_with_gradient=len(all_grads), selected_gradient_norms=gradients)
    del result, batch, all_grads
    model.zero_grad(set_to_none=True)
    model.pts_bbox_head.reset_memory()
    model.eval()
    apply_attention_eval_policy(model)
    # Reuse the actual dataset with its evaluation preparation path. No metrics
    # are claimed on these training frames; this checks shapes and state resets.
    ds.test_mode = True
    def infer(index, scene):
        item = ds[index]
        meta = dict(item['img_metas'].data)
        meta['scene_token'] = scene
        inp = {k: item[k] for k in INPUT_KEYS if k in item}
        inp['img_metas'] = DataContainer(meta, cpu_only=True)
        with torch.no_grad():
            prediction = model(to_device(collate([inp], samples_per_gpu=1), 'cuda'), return_loss=False)[0]
        for key in ['ego_fut_preds_fix_time', 'ego_fut_preds_fix_dist']:
            assert torch.isfinite(prediction[key]).all(), key
        return prediction
    # Same frame and independently reset memory must produce the same plan.
    first = infer(30, 'smoke_first')
    repeated = infer(30, 'smoke_repeat')
    for key in ['ego_fut_preds_fix_time', 'ego_fut_preds_fix_dist']:
        torch.testing.assert_close(first[key], repeated[key], rtol=0, atol=0)
    continued = infer(31, 'smoke_repeat')
    output = dict(config=str(Path(args.config).resolve()), checkpoint=str(Path(args.ckpt).resolve()),
                  scope='complete Base architecture; reduced image resolution; no optimizer update',
                  train=train_summary, eval_deterministic=True, second_frame_memory_passed=True,
                  output_shapes={k: list(continued[k].shape) for k in
                                 ['ego_fut_preds_fix_time', 'ego_fut_preds_fix_dist']},
                  peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
