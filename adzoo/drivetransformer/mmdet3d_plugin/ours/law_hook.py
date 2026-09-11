"""Iteration-based LAW schedule; runner.iter makes resume deterministic."""
from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class LAWLossWeightHook(Hook):
    def __init__(self, max_weight=0.2, start_iter=0, warmup_iters=1000):
        if max_weight < 0 or start_iter < 0 or warmup_iters < 0:
            raise ValueError('Invalid LAW weight schedule')
        self.max_weight = max_weight
        self.start_iter = start_iter
        self.warmup_iters = warmup_iters

    def before_train_iter(self, runner):
        model = runner.model
        while hasattr(model, 'module'):
            model = model.module
        if getattr(model, 'law', None) is None:
            raise RuntimeError('LAWLossWeightHook requires model.law')
        progress = runner.iter - self.start_iter
        if progress < 0:
            weight = 0.0
        elif self.warmup_iters == 0:
            weight = self.max_weight
        else:
            weight = self.max_weight * min(progress / self.warmup_iters, 1.0)
        model.law.loss_weight = weight
