"""Opt-in continuation of the completed 29350-update Random60 model."""
from mmcv.runner import IterBasedRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.lr_updater import LrUpdaterHook
from mmcv.datasets.samplers.sampler import SAMPLER
from .datasets.random_short_routes import RandomShortRouteBatchSampler


@RUNNERS.register_module()
class Random60ContinuationRunner(IterBasedRunner):
    def resume(self, checkpoint, resume_optimizer=True, map_location='default'):
        if not resume_optimizer:
            raise ValueError('Continuation must restore optimizer state')
        super().resume(checkpoint, resume_optimizer=True, map_location=map_location)
        # This repository saves meta.iter before incrementing the completed step.
        if self._iter != 29349:
            raise ValueError('Expected the original completed 29350-step checkpoint')
        if not self.optimizer.state:
            raise ValueError('Missing optimizer moments')
        self._iter += 1
        self._inner_iter = self._iter
        self.logger.info('Random60 continuation: 29350 completed updates; optimizer restored')


@HOOKS.register_module()
class Random60ContinueConstantLrUpdaterHook(LrUpdaterHook):
    def __init__(self, start_iter=29350, additional_iters=29350, **kwargs):
        super().__init__(**kwargs)
        self.start_iter = start_iter
        self.additional_iters = additional_iters

    def get_lr(self, runner, base_lr):
        if not self.start_iter <= runner.iter < self.start_iter+self.additional_iters:
            raise ValueError('Continuation LR outside its frozen update budget')
        # Original main initial_lr=1e-4. Preserve param-group LR multipliers.
        return base_lr*.1



@SAMPLER.register_module()
class ContinuationRandomShortRouteBatchSampler(RandomShortRouteBatchSampler):
    def __init__(self, *args, skip_batches=29350, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_batches = skip_batches

    def __iter__(self):
        stream = super().__iter__()
        # Advance descriptors only, without reading images or replaying training.
        for _ in range(self.skip_batches):
            next(stream)
        first = next(stream)
        # Temporal caches are not checkpointed. Reset every slot at the boundary.
        yield [sample._replace(start=sample.index) for sample in first]
        yield from stream
