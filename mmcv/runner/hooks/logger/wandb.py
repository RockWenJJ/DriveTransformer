# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import master_only
from ..hook import HOOKS
from .base import LoggerHook

try:
    import wandb
except ImportError:
    wandb = None


@HOOKS.register_module()
class WandbLoggerHook(LoggerHook):
    """Logger used in Visualizations with wandb.

    Args:
        init_kwargs (dict): A dict passed to wandb.init to make a record.
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
        reset_flag (bool): Whether to clear the output buffer after logging.
        commit (bool): Save the metrics dict to the wandb server.
        by_epoch (bool): Whether EpochBasedRunner is used.
        with_step (bool): If True, the step will be logged from
            self.get_iter(runner). Otherwise, step will not be logged.
    """

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 commit=True,
                 by_epoch=True,
                 with_step=True):
        super(WandbLoggerHook, self).__init__(interval, ignore_last,
                                              reset_flag, by_epoch)
        self.import_wandb()
        self.init_wandb_kwargs = init_kwargs
        if self.init_wandb_kwargs is None:
            self.init_wandb_kwargs = {}
        self.commit = commit
        self.with_step = with_step

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')

    @master_only
    def before_run(self, runner):
        super(WandbLoggerHook, self).before_run(runner)
        # this vendored runner may not carry cfg; record it only when available
        cfg = getattr(runner, 'cfg', None)
        if cfg is not None:
            self.init_wandb_kwargs['config'] = cfg._cfg_dict
        self.wandb = wandb.init(**self.init_wandb_kwargs)

    @master_only
    def after_run(self, runner):
        self.wandb.finish()

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            if self.with_step:
                self.wandb.log(
                    tags, step=self.get_iter(runner), commit=self.commit)
            else:
                tags['global_step'] = self.get_iter(runner)
                self.wandb.log(tags, commit=self.commit)
