"""One real training-batch forward/backward; no optimizer or checkpoint writes."""
import argparse
import importlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmcv import Config
from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.parallel.collate import collate
from mmcv.utils import wrap_fp16_model
from tools.ol_eval_tasks_v2 import to_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = Config.fromfile(a.config)
    importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))
    ds = build_dataset(cfg.data.train)
    sample = ds[0]
    batch = to_device(collate([sample], samples_per_gpu=1), 'cuda')
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    state = torch.load(a.ckpt, map_location='cpu')['state_dict']
    model.load_state_dict({k.replace('model.', '').replace('._orig_mod', ''): v for k, v in state.items()}, strict=True)
    del state
    # The training runner enables the repository's auto_fp16 wrappers through
    # this function (the local wrapper uses bfloat16 with float32 parameters).
    wrap_fp16_model(model)
    model.cuda().train()
    model.set_epoch(0)
    output = model(batch, return_loss=True)
    output['loss'].backward()
    gradients = {}
    for group, fragment in [('map_reg', 'map_reg_branches'), ('map_cls', 'map_cls_branches'),
                            ('map_position', 'map_ref_embedding'), ('backbone', 'img_backbone')]:
        values = [param.grad for name, param in model.named_parameters() if fragment in name and param.grad is not None]
        assert values and all(torch.isfinite(g).all().item() for g in values), group
        norm = sum(float(g.float().norm().item()) for g in values)
        assert norm > 0., group
        gradients[group] = norm
    all_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    assert all_finite
    assert all(np.isfinite(v) for v in output['log_vars'].values())
    result = dict(status='passed', checkpoint_load='strict', optimizer_updates=0,
                  config=a.config, checkpoint=a.ckpt, seed=0, batch_size=1,
                  precision='bfloat16 autocast, float32 parameters',
                  map_clip_mode=ds.map_clip_mode, map_deduplicate=ds.map_deduplicate,
                  gt_instances=len(sample['map_gt_labels_3d'].data),
                  loss=output['log_vars'], gradient_norm_sum=gradients,
                  all_parameter_gradients_finite=all_finite,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
