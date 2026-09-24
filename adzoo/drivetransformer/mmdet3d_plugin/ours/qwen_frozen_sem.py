"""DT-Qwen08-FrozenSem: immutable Qwen features, trainable DT adapters."""
from pathlib import Path
import sys

import torch
from torch import nn
from mmcv.models.builder import BACKBONES, DETECTORS, HEADS
from .no_map import NoMapDriveTransformer, NoMapHead


@BACKBONES.register_module()
class FrozenQwenBackbone(nn.Module):
    def __init__(self, model_path, worker_python, prompt, revision,
                 out_channels=512, vision_chunk=6, language_chunk=1,
                 memory_fraction=0.20, timeout=600, **kwargs):
        super().__init__()
        if kwargs:
            raise TypeError('Unexpected Qwen settings: %s' % sorted(kwargs))
        self.visual_projection = nn.Conv2d(1024, out_channels, 1)
        self.settings = dict(model_path=model_path, worker_python=worker_python,
            prompt=prompt, revision=revision, vision_chunk=vision_chunk,
            language_chunk=language_chunk, memory_fraction=memory_fraction,
            timeout=timeout)
        self._client = None

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            raise ValueError('Qwen weights are supplied by the pinned worker, not ResNet')

    def forward(self, images):
        if self._client is None:
            # Worker files deliberately have no mmcv/transformers import in this process.
            root = Path(__file__).resolve().parents[4] / 'tools' / 'qwen_frozen'
            sys.path.insert(0, str(root))
            from bridge import FrozenQwenClient
            self._client = FrozenQwenClient(root, images.device, **self.settings)
        visual, language = self._client(images)
        return self.visual_projection(visual), language

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None


@HEADS.register_module()
class QwenFrozenSemHead(NoMapHead):
    def __init__(self, *args, language_dim=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.language_projection = nn.Sequential(nn.LayerNorm(language_dim),
                                                nn.Linear(language_dim, self.embed_dims))
        self.language_fusion = nn.Sequential(nn.LayerNorm(2 * self.embed_dims),
            nn.Linear(2 * self.embed_dims, self.embed_dims), nn.SiLU(),
            nn.Linear(self.embed_dims, self.embed_dims))
        self._language_context = None

    def set_language_context(self, hidden):
        if self._language_context is not None:
            raise RuntimeError('Unconsumed language context; refusing stale-frame fusion')
        if hidden.ndim != 3 or hidden.shape[1:] != (1, 1024) or hidden.requires_grad:
            raise ValueError('Expected detached [batch, 1, 1024] language hidden state')
        self._language_context = hidden

    def forward(self, img_feats, *args, **kwargs):
        if self._language_context is None or len(self._language_context) != len(img_feats):
            raise RuntimeError('Missing or mismatched language context for this frame')
        try:
            return super().forward(img_feats, *args, **kwargs)
        finally:
            self._language_context = None

    def _planning_features(self, features, layer_index):
        if layer_index != len(self.decoder_layers) - 1:
            return features
        language = self.language_projection(self._language_context.to(features.dtype))
        return self.language_fusion(torch.cat((features, language), dim=-1))


@DETECTORS.register_module()
class QwenFrozenSemDriveTransformer(NoMapDriveTransformer):
    def extract_img_feat(self, img, img_metas=None, len_queue=1):
        if len_queue != 1 or img.ndim not in (5, 6):
            raise ValueError('FrozenSem requires one current frame with six cameras')
        if img.ndim == 6:
            if img.shape[1] != 1:
                raise ValueError('Temporal queues must be handled by DT recurrent memory')
            img = img[:, 0]
        batch, cameras, channels, height, width = img.shape
        if cameras != 6 or channels != 3:
            raise ValueError('Camera order: FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT')
        flat = img.reshape(batch * cameras, channels, height, width)
        if self.use_grid_mask:
            flat = self.grid_mask(flat)
        visual, language = self.img_backbone(flat)
        self.pts_bbox_head.set_language_context(language)
        return [visual.reshape(batch, 1, cameras, *visual.shape[1:])]
