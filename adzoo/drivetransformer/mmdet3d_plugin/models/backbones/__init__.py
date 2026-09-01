#from .vovnet import VoVNet
from .eva02 import ViT, SimpleFPN
from .dinov3 import FrozenDINOv3ViT

__all__ = ["ViT", "SimpleFPN", "FrozenDINOv3ViT"]
