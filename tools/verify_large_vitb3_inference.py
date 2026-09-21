"""Strict-load and real-data full-resolution inference after bounded training."""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmcv import Config
from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.parallel import DataContainer
from mmcv.parallel.collate import collate
from mmcv.utils import wrap_fp16_model
from drivetransformer_contracts import apply_attention_eval_policy
from tools.ol_eval_tasks_v2 import INPUT_KEYS, to_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = Config.fromfile(args.config)
    importlib.import_module(os.path.dirname(cfg.plugin_dir).replace('/', '.'))
    model = build_model(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    original = load_file(cfg.model.img_backbone.checkpoint, device='cpu')
    current = model.img_backbone.dinov3.state_dict()
    assert set(original) == set(current)
    assert all(torch.equal(current[k], original[k]) for k in original), 'Frozen trunk changed'
    optimizer_saved = bool(checkpoint.get('optimizer', {}).get('state'))
    checkpoint_iter = checkpoint['meta']['iter']
    expected_iter = int(Path(args.checkpoint).resolve().stem.split('_')[-1])
    assert checkpoint_iter == expected_iter, (checkpoint_iter, expected_iter)
    del original, current, checkpoint
    ds = build_dataset(cfg.data.val)
    raw = ds.get_data_by_index(30)
    item = ds.get_data_info(30)
    np.testing.assert_allclose(item['ego_lcf_feat'][2:4], raw['ego_accel'][:2], rtol=1e-6)
    np.testing.assert_allclose(item['timestamp'], raw['frame_idx'] * 0.1)
    wrap_fp16_model(model)
    model.cuda().eval()
    apply_attention_eval_policy(model)
    shapes = []
    handle = model.img_backbone.register_forward_hook(lambda m, a, y: shapes.append(list(y[0].shape)))
    def infer(index):
        item = ds[index]
        inp = {k: item[k] for k in INPUT_KEYS if k in item}
        inp['img_metas'] = DataContainer(dict(item['img_metas'].data), cpu_only=True)
        with torch.no_grad():
            result = model(to_device(collate([inp], samples_per_gpu=1), 'cuda'), return_loss=False)[0]
        assert not any('map' in k for k in result), list(result)
        for key in ('ego_fut_preds_fix_time', 'ego_fut_preds_fix_dist'):
            assert torch.isfinite(result[key]).all(), key
        return result
    model.pts_bbox_head.reset_memory()
    first = infer(30)
    model.pts_bbox_head.reset_memory()
    repeat = infer(30)
    for key in ('ego_fut_preds_fix_time', 'ego_fut_preds_fix_dist'):
        torch.testing.assert_close(first[key], repeat[key], rtol=0, atol=0)
    continued = infer(31)
    handle.remove()
    assert all(s == [6, 768, 30, 60] for s in shapes), shapes
    result = dict(strict_checkpoint_load=True, checkpoint_iteration=checkpoint_iter,
        optimizer_state_present=optimizer_saved, frozen_trunk_bitwise_unchanged=True,
        full_resolution_features=shapes, reset_inference_deterministic=True,
        next_frame_memory_finite=True, output_keys=list(continued),
        ego_accel_input=item['ego_lcf_feat'][2:4].tolist(),
        sample_shape=list(np.asarray(raw['ego_accel']).shape),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
