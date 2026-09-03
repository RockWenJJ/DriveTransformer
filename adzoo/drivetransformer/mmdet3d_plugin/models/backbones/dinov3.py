import contextlib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from mmcv.models import BACKBONES
from mmcv.models.backbones.base_module import BaseModule


DINOV3_VARIANTS = {
    "vits16": dict(hidden_size=384, intermediate_size=1536, num_heads=6),
    "vitb16": dict(hidden_size=768, intermediate_size=3072, num_heads=12),
}


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class DINOv3Embeddings(nn.Module):
    def __init__(self, hidden_size, patch_size=16, num_register_tokens=4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.mask_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.patch_embeddings = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.register_tokens = nn.Parameter(
            torch.empty(1, num_register_tokens, hidden_size)
        )

    def forward(self, pixel_values):
        patches = self.patch_embeddings(pixel_values)
        patches = patches.flatten(2).transpose(1, 2)
        batch_size = patches.shape[0]
        return torch.cat(
            (
                self.cls_token.expand(batch_size, -1, -1),
                self.register_tokens.expand(batch_size, -1, -1),
                patches,
            ),
            dim=1,
        )


class DINOv3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states, cos, sin):
        batch_size, num_tokens, hidden_size = hidden_states.shape

        def project(layer):
            value = layer(hidden_states)
            return value.view(
                batch_size, num_tokens, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query = project(self.q_proj)
        key = project(self.k_proj)
        value = project(self.v_proj)

        num_patches = cos.shape[-2]
        num_prefix_tokens = num_tokens - num_patches
        query_prefix, query_patches = query.split(
            (num_prefix_tokens, num_patches), dim=-2
        )
        key_prefix, key_patches = key.split(
            (num_prefix_tokens, num_patches), dim=-2
        )
        query_patches = query_patches * cos + _rotate_half(query_patches) * sin
        key_patches = key_patches * cos + _rotate_half(key_patches) * sin
        query = torch.cat((query_prefix, query_patches), dim=-2)
        key = torch.cat((key_prefix, key_patches), dim=-2)

        output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        output = output.transpose(1, 2).reshape(batch_size, num_tokens, hidden_size)
        return self.o_proj(output)


class DINOv3LayerScale(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        return hidden_states * self.lambda1


class DINOv3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        return self.down_proj(F.gelu(self.up_proj(hidden_states)))


class DINOv3Layer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.attention = DINOv3Attention(hidden_size, num_heads)
        self.layer_scale1 = DINOv3LayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = DINOv3MLP(hidden_size, intermediate_size)
        self.layer_scale2 = DINOv3LayerScale(hidden_size)

    def forward(self, hidden_states, cos, sin):
        attention = self.attention(self.norm1(hidden_states), cos, sin)
        hidden_states = hidden_states + self.layer_scale1(attention)
        mlp = self.mlp(self.norm2(hidden_states))
        return hidden_states + self.layer_scale2(mlp)


class DINOv3ViT(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_heads,
        patch_size=16,
        num_layers=12,
        num_register_tokens=4,
        rope_theta=100.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.embeddings = DINOv3Embeddings(
            hidden_size, patch_size, num_register_tokens
        )
        self.layer = nn.ModuleList(
            [
                DINOv3Layer(hidden_size, intermediate_size, num_heads)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, eps=1e-5)

        head_dim = hidden_size // num_heads
        inv_freq = 1.0 / rope_theta ** torch.arange(
            0, 1, 4.0 / head_dim, dtype=torch.float32
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def _rope(self, height, width, dtype, device):
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        coords_h = torch.arange(0.5, patch_h, dtype=torch.float32, device=device)
        coords_w = torch.arange(0.5, patch_w, dtype=torch.float32, device=device)
        coords_h = 2.0 * coords_h / patch_h - 1.0
        coords_w = 2.0 * coords_w / patch_w - 1.0
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        ).flatten(0, 1)
        angles = 2.0 * math.pi * coords[:, :, None] * self.rope_inv_freq
        angles = angles.flatten(1, 2).repeat(1, 2)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        cos, sin = self._rope(
            pixel_values.shape[-2],
            pixel_values.shape[-1],
            hidden_states.dtype,
            hidden_states.device,
        )
        cos = cos[None, None]
        sin = sin[None, None]
        for layer in self.layer:
            hidden_states = layer(hidden_states, cos, sin)
        return self.norm(hidden_states)


@BACKBONES.register_module()
class FrozenDINOv3ViT(BaseModule):
    """Frozen DINOv3 ViT with a trainable stride-32 feature adapter."""

    def __init__(
        self,
        variant,
        checkpoint,
        out_channels=512,
        forward_chunk_size=12,
        amp_dtype="bfloat16",
        freeze=True,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if variant not in DINOV3_VARIANTS:
            raise ValueError(
                "variant must be one of {}, got {!r}".format(
                    sorted(DINOV3_VARIANTS), variant
                )
            )
        if forward_chunk_size < 1:
            raise ValueError("forward_chunk_size must be positive")
        if amp_dtype not in (None, "bfloat16", "float16", "float32"):
            raise ValueError("unsupported amp_dtype: {!r}".format(amp_dtype))

        self.variant = variant
        self.checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
        self._checkpoint_arg = checkpoint
        self.forward_chunk_size = forward_chunk_size
        self.amp_dtype = amp_dtype
        variant_cfg = DINOV3_VARIANTS[variant]
        self.dinov3 = DINOv3ViT(**variant_cfg)
        hidden_size = variant_cfg["hidden_size"]
        self.adapter = nn.Sequential(
            nn.Conv2d(hidden_size, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(32, out_channels),
        )
        self._checkpoint_loaded = False
        self.fp16_enabled = False
        self.freeze_trunk = freeze
        # Load pretrained weights at CONSTRUCTION time: eval-side builders
        # (CARLA agent, log-replay scripts) do build -> load trained ckpt ->
        # forward without ever calling init_weights(), so the pretrained trunk
        # must already be in place before load_checkpoint() overwrites it.
        self._load_pretrained()

    def _freeze_dinov3(self):
        self.dinov3.eval()
        for parameter in self.dinov3.parameters():
            parameter.requires_grad = False

    def init_weights(self):
        if self._checkpoint_loaded:
            return
        self._load_pretrained()

    def _load_pretrained(self):
        # Resolve the safetensors path against the repo root as a fallback, so
        # the CARLA-side agent (cwd = Bench2Drive/leaderboard) can build the
        # model from a config that uses a repo-relative "./ckpts/..." path.
        ckpt = self.checkpoint
        if not os.path.isfile(ckpt):
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
            )
            alt = os.path.join(repo_root, self._checkpoint_arg)
            if os.path.isfile(alt):
                ckpt = alt
            else:
                raise FileNotFoundError(
                    "DINOv3 checkpoint does not exist: {} (also tried {})".format(self.checkpoint, alt)
                )
        state_dict = load_file(ckpt, device="cpu")
        self.dinov3.load_state_dict(state_dict, strict=True)
        for module in self.adapter.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if self.freeze_trunk:
            self._freeze_dinov3()
        self._checkpoint_loaded = True
        self._is_init = True

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_trunk:
            self._freeze_dinov3()
        return self

    def forward(self, pixel_values):
        if not self._checkpoint_loaded:
            # Some eval-side builders never call init_weights(); load lazily
            # instead of raising. Loading a trained checkpoint afterwards
            # still overwrites the pretrained weights as expected.
            self._load_pretrained()
        height, width = pixel_values.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError(
                "input height and width must be divisible by 16, got {}x{}".format(
                    height, width
                )
            )

        out_dtype = pixel_values.dtype
        # The trunk is frozen, so reduced precision only perturbs the features the
        # adapter is trained on: measured against fp32, bf16 keeps per-token cosine
        # similarity at 0.9999 while cutting the ViT-B/16 forward from 3.57s to 0.87s
        # per training step.
        autocast = self.amp_dtype not in (None, "float32") and pixel_values.is_cuda
        run_dtype = getattr(torch, self.amp_dtype) if autocast else out_dtype

        patch_features = []
        trunk_ctx = torch.no_grad() if self.freeze_trunk else contextlib.nullcontext()
        with trunk_ctx:
            for start in range(0, pixel_values.shape[0], self.forward_chunk_size):
                chunk = pixel_values[start : start + self.forward_chunk_size]
                with torch.cuda.amp.autocast(enabled=autocast, dtype=run_dtype):
                    tokens = self.dinov3(chunk.to(run_dtype))
                tokens = tokens[:, 1 + self.dinov3.num_register_tokens :]
                features = tokens.transpose(1, 2).reshape(
                    chunk.shape[0],
                    -1,
                    height // self.dinov3.patch_size,
                    width // self.dinov3.patch_size,
                )
                patch_features.append(features.to(out_dtype))

        features = self.adapter(torch.cat(patch_features, dim=0))
        return (features,)

