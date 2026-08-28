from .base import LoggerHook
from .text import TextLoggerHook
from .tensorboard import TensorboardLoggerHook
try:
    from .wandb import WandbLoggerHook
except ImportError:
    pass