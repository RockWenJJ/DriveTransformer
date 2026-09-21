"""Bounded verification through the real distributed training entry point.

Run with torch.distributed.run (4 ranks). All checkpoints belong to the
preparation directory; these are disposable validation weights, not training
initialization. The actual model, resolution, batch and optimizer are retained.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adzoo.drivetransformer import train as training


class AuditHook(training.Fp16OptimizerHook):
    def before_run(self, runner):
        super().before_run(runner)
        self.root = Path(runner.work_dir)
        self.rank = torch.distributed.get_rank()
        model = runner.model.module
        head = model.pts_bbox_head
        assert len(head.decoder_layers) == 12
        assert head.embed_dims == 768 and head.img_stride == 16
        assert head.map_num_query == 0 and not hasattr(head, 'map_memory_embedding')
        bad = [n for n, p in model.named_parameters()
               if any(part.startswith(('map_', 'loss_map_')) for part in n.split('.'))]
        assert not bad, bad
        assert tuple(model.img_backbone.out_indices) == (3, 7, 11)
        assert all(not p.requires_grad for p in model.img_backbone.dinov3.parameters())
        self.shapes = {}
        self.handles = [
            model.img_backbone.adapter[0].register_forward_pre_hook(
                lambda m, a: self.shapes.update(concat=list(a[0].shape))),
            model.img_backbone.register_forward_hook(
                lambda m, a, y: self.shapes.update(feature=list(y[0].shape)))]
        self.parameter_counts = dict(total=sum(p.numel() for p in model.parameters()),
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad))
        self.previous_adapter = model.img_backbone.adapter[0].weight.detach().clone()
        self.start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()

    def after_train_iter(self, runner):
        assert torch.isfinite(runner.outputs['loss']).all()
        assert not any('map' in key for key in runner.outputs['log_vars'])
        super().after_train_iter(runner)
        model = runner.model.module
        gradients = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for _, g in gradients)
        assert all(p.grad is None and not p.requires_grad
                   for p in model.img_backbone.dinov3.parameters())
        adapter = model.img_backbone.adapter[0].weight
        changed = not torch.equal(adapter.detach(), self.previous_adapter)
        assert changed, 'Adapter did not update'
        self.previous_adapter.copy_(adapter.detach())
        assert self.shapes['concat'][1:] == [2304, 30, 60], self.shapes
        assert self.shapes['feature'][1:] == [768, 30, 60], self.shapes
        result = dict(rank=self.rank, iteration=runner.iter+1,
            loss=float(runner.outputs['loss'].detach()), shapes=self.shapes,
            parameter_counts=self.parameter_counts, map_parameters=0,
            frozen_trunk_gradients=0, trainable_parameters_with_gradient=len(gradients),
            adapter_updated=changed, all_gradients_finite=True,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            elapsed_seconds=time.monotonic()-self.start)
        with (self.root / ('audit_rank%d.jsonl' % self.rank)).open('a') as f:
            f.write(json.dumps(result)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--resume-from')
    args = parser.parse_args()
    if not 1 <= args.steps <= 5:
        raise ValueError('Verification must remain bounded to 1..5 iterations')
    training.Fp16OptimizerHook = AuditHook
    cfg = training.Config.fromfile(args.config)
    cfg.runner.max_iters = args.steps
    cfg.checkpoint_config.update(interval=1, max_keep_ckpts=2)
    cfg.log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
    cfg.data.workers_per_gpu = 2
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    smoke_config = work / ('verification_config_rank%s.py' % os.environ.get('LOCAL_RANK', '0'))
    cfg.dump(str(smoke_config))
    sys.argv = ['train.py', str(smoke_config), '--launcher', 'pytorch', '--seed', '0',
        '--work-dir', str(work)]
    if args.resume_from:
        sys.argv.extend(['--resume-from', args.resume_from])
    training.main()


if __name__ == '__main__':
    main()
