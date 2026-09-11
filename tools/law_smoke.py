"""Bounded real-data LAW integration check; no optimizer/training job is run.

Run from repository root with an idle CUDA device selected explicitly. Uses the
actual Base architecture and image resolution, random weights, two train samples
(one valid future and one scene tail), and the configured mixed precision path.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmcv.utils import Config, wrap_fp16_model
from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.parallel import collate
import adzoo.drivetransformer.mmdet3d_plugin  # registry
import adzoo.drivetransformer.mmdet3d_plugin.ours.drivetransformer as detector_module
from adzoo.drivetransformer.mmdet3d_plugin.ours.law_hook import LAWLossWeightHook


def to_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {k: to_cuda(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cuda(v) for v in value]
    return value


def seed():
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    np.random.seed(123)


def memory_snapshot(head):
    return {k: v.detach().clone() for k, v in vars(head).items()
            if 'memory' in k and isinstance(v, torch.Tensor)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_law.py')
    parser.add_argument('--output', required=True)
    parser.add_argument('--base-sha', default='39063526f9873800585da480fb49e91c556d7c0c')
    args = parser.parse_args()
    start = time.time()
    torch.set_num_threads(2)
    seed()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train)
    aug = dataset.get_augmentation()
    valid = dataset.prepare_train_data(0, aug)
    tail_index = int(dataset.divide_nums[0]) - 1
    tail = dataset.prepare_train_data(tail_index, aug)
    assert valid['law_future_valid'].data.item()
    assert not tail['law_future_valid'].data.item()
    batch = to_cuda(collate([valid, tail], samples_per_gpu=2))
    assert tuple(batch['img'].shape[:3]) == (2, 1, 6)
    assert tuple(batch['law_future_img'].shape[:2]) == (2, 6)
    cfg.model.pretrained = None  # no checkpoint download or implicit finetuning
    model = build_model(cfg.model).cuda()
    model.init_weights()
    wrap_fp16_model(model)
    model.train()
    model.set_epoch(0)
    model.pts_bbox_head.reset_memory()
    observed = {}
    original_law = model.forward_law

    def checked_law(features, predictions, images, masks):
        memory = memory_snapshot(model.pts_bbox_head)
        modes = {name: m.training for name, m in model.named_modules()}
        buffers = {name: b.clone() for name, b in model.img_backbone.named_buffers()}
        result = original_law(features, predictions, images, masks)
        for key, value in memory.items():
            torch.testing.assert_close(value, getattr(model.pts_bbox_head, key), rtol=0, atol=0)
        assert modes == {name: m.training for name, m in model.named_modules()}
        for name, value in buffers.items():
            torch.testing.assert_close(value, dict(model.img_backbone.named_buffers())[name], rtol=0, atol=0)
        observed.update(loss=result['loss_law'], features=features,
                        trajectory=predictions['ego_fut_preds_fix_time'])
        return result

    model.forward_law = checked_law
    batch['law_future_img'].requires_grad_(True)
    result = model(copy.deepcopy(batch))
    assert torch.isfinite(result['loss'])
    gradients = torch.autograd.grad(observed['loss'],
                                    (observed['features'], observed['trajectory']), retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    result['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.law.parameters())
    # Also verify stop-gradient on the target via the actual future extraction.
    target_input = batch['law_future_img'].detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    isolated = original_law(observed['features'].detach(),
                           dict(ego_fut_preds_fix_time=observed['trajectory'].detach()),
                           target_input, batch['law_future_valid'])
    isolated['loss_law'].backward()
    assert target_input.grad is None
    report = dict(config=args.config, base_sha=args.base_sha,
                  batch_shape=list(batch['img'].shape),
                  future_pairs=[valid['law_future_meta'].data, tail['law_future_meta'].data],
                  law_parameters=sum(p.numel() for p in model.law.parameters()),
                  losses=result['log_vars'],
                  law_gradient_to_fpn=float(gradients[0].abs().sum()),
                  law_gradient_to_waypoints=float(gradients[1].abs().sum()),
                  future_target_stop_gradient=True, future_memory_unchanged=True,
                  target_bn_and_modes_unchanged=True, mixed_precision=True)
    del result, isolated, observed, gradients
    model.forward_law = original_law
    model.zero_grad(set_to_none=True)

    # Original vs modified detector forward_train, LAW off, same tensors/weights.
    source = subprocess.check_output(['git', 'show', args.base_sha +
        ':adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer.py'], text=True)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_train')
    function.decorator_list = []
    namespace = dict(detector_module.__dict__)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<original-forward-train>', 'exec'), namespace)
    old_forward = namespace['forward_train']
    law = model.law
    model.law = None
    baseline_batch = {k: v for k, v in batch.items() if not k.startswith('law_')}
    with torch.no_grad():
        seed()
        model.pts_bbox_head.reset_memory()
        old = old_forward(model, **copy.deepcopy(baseline_batch))
        seed()
        model.pts_bbox_head.reset_memory()
        new = model(copy.deepcopy(baseline_batch))
    assert old['log_vars'] == new['log_vars']
    assert not any(k.startswith('law.') for k in model.state_dict())
    report['law_off_matches_original_training'] = True
    model.law = law

    # The actual inference entry point must never execute the LAW predictor.
    test_sample = dataset.prepare_test_data(0, aug)
    test_batch = to_cuda(collate([test_sample], samples_per_gpu=1))
    def forbid_law(*unused):
        raise AssertionError('Inference called LAW')
    hook = model.law.register_forward_pre_hook(forbid_law)
    model.eval()
    with torch.no_grad():
        seed()
        model.prev_scene_token = None
        model.pts_bbox_head.reset_memory()
        with_law = model(copy.deepcopy(test_batch), return_loss=False)
        model.law = None
        seed()
        model.prev_scene_token = None
        model.pts_bbox_head.reset_memory()
        without_law = model(copy.deepcopy(test_batch), return_loss=False)
    hook.remove()
    for key in with_law[0]:
        if isinstance(with_law[0][key], torch.Tensor):
            torch.testing.assert_close(with_law[0][key], without_law[0][key], rtol=0, atol=0)
    model.law = law
    report['inference_no_future_input_and_identical_outputs'] = True
    # Iteration schedule also works with a wrapper and resumed runner position.
    from types import SimpleNamespace
    schedule = LAWLossWeightHook()
    for iteration, expected in [(0, 0.), (500, 0.1), (1000, 0.2), (7000, 0.2)]:
        schedule.before_train_iter(SimpleNamespace(model=SimpleNamespace(module=model), iter=iteration))
        assert model.law.loss_weight == expected
    report['schedule_and_resume_checked'] = True
    report['cuda_peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024 ** 3
    report['elapsed_seconds'] = time.time() - start
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
