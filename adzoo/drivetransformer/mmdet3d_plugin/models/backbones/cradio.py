import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from mmcv.models import BACKBONES
from mmcv.models.backbones.base_module import BaseModule


CRADIO_VARIANTS = {
    "c-radio_v3-b": dict(
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        num_layers=12,
        patch_size=16,
        cpe_max_size=2048,
        num_prefix_tokens=8,
    ),
}

# InputConditioner buffers shipped with the released checkpoints; used only as a
# fallback when a checkpoint stores no conditioner (OpenAI CLIP statistics).
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClsToken(nn.Module):
    def __init__(self, hidden_size, num_prefix_tokens):
        super().__init__()
        self.token = nn.Parameter(torch.empty(num_prefix_tokens, hidden_size))

    def forward(self, patches):
        token = self.token.unsqueeze(0).expand(patches.shape[0], -1, -1)
        return torch.cat((token, patches), dim=1)


class ViTPatchGenerator(nn.Module):
    """RADIO's Cropped Position Embedding patch generator.

    The absolute position embedding is stored at ``cpe_max_size`` resolution and
    resampled per input size. Inference resamples to a square of the larger
    patch-grid side and then crops the top-left corner, which is what makes the
    embedding aspect-preserving; that exact sequence has to be reproduced or the
    features silently drift from the reference implementation.
    """

    def __init__(self, hidden_size, patch_size, cpe_max_size, num_prefix_tokens):
        super().__init__()
        self.patch_size = patch_size
        self.num_rows = cpe_max_size // patch_size
        self.num_cols = cpe_max_size // patch_size
        self.pos_embed = nn.Parameter(
            torch.empty(1, self.num_rows * self.num_cols, hidden_size)
        )
        self.embedder = nn.Linear(3 * patch_size**2, hidden_size, bias=False)
        self.cls_token = ClsToken(hidden_size, num_prefix_tokens)
        self._pos_embed_cache = {}

    def _im_to_patches(self, pixel_values):
        batch, channels, height, width = pixel_values.shape
        patch = self.patch_size
        patches = pixel_values.reshape(
            batch, channels, height // patch, patch, width // patch, patch
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        return patches.reshape(
            batch, (height // patch) * (width // patch), channels * patch * patch
        )

    def _get_pos_embed(self, patch_h, patch_w, dtype):
        key = (patch_h, patch_w, self.pos_embed.device)
        cached = self._pos_embed_cache.get(key)
        if cached is None:
            pos_embed = self.pos_embed.detach().float()
            pos_embed = pos_embed.reshape(1, self.num_rows, self.num_cols, -1)
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            max_dim = max(patch_h, patch_w)
            pos_embed = F.interpolate(
                pos_embed, size=(max_dim, max_dim), mode="bilinear", align_corners=True
            )
            pos_embed = pos_embed[..., :patch_h, :patch_w]
            if pos_embed.shape[-2:] != (patch_h, patch_w):
                pos_embed = F.interpolate(
                    pos_embed,
                    size=(patch_h, patch_w),
                    mode="bilinear",
                    align_corners=True,
                )
            cached = pos_embed.flatten(2).permute(0, 2, 1).contiguous()
            self._pos_embed_cache[key] = cached
        return cached.to(dtype=dtype)

    def forward(self, pixel_values):
        patches = self.embedder(self._im_to_patches(pixel_values))
        patch_h = pixel_values.shape[-2] // self.patch_size
        patch_w = pixel_values.shape[-1] // self.patch_size
        patches = patches + self._get_pos_embed(patch_h, patch_w, patches.dtype)
        return self.cls_token(patches)


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states):
        batch, num_tokens, hidden_size = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(
            batch, num_tokens, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        output = F.scaled_dot_product_attention(query, key, value)
        output = output.transpose(1, 2).reshape(batch, num_tokens, hidden_size)
        return self.proj(output)


class LayerScale(nn.Module):
    # The parameter is called ``grandma`` upstream: HuggingFace Hub rewrites any
    # checkpoint key containing ``gamma``, so RADIO renamed it to survive a round
    # trip through the Hub.
    def __init__(self, hidden_size):
        super().__init__()
        self.grandma = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        return hidden_states * self.grandma


class Mlp(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        return self.fc2(F.gelu(self.fc1(hidden_states)))


class Block(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.ls1 = LayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Mlp(hidden_size, intermediate_size)
        self.ls2 = LayerScale(hidden_size)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.ls1(self.attn(self.norm1(hidden_states)))
        return hidden_states + self.ls2(self.mlp(self.norm2(hidden_states)))


class CRADIOViT(nn.Module):
    """RADIO's timm ViT trunk.

    RADIO is trained with ``model_norm=False``, so the trunk has no final
    LayerNorm and block outputs are returned as-is.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_heads,
        num_layers,
        patch_size,
        cpe_max_size,
        num_prefix_tokens,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_prefix_tokens = num_prefix_tokens
        self.patch_generator = ViTPatchGenerator(
            hidden_size, patch_size, cpe_max_size, num_prefix_tokens
        )
        self.blocks = nn.ModuleList(
            [
                Block(hidden_size, intermediate_size, num_heads)
                for _ in range(num_layers)
            ]
        )

    def forward(self, pixel_values):
        hidden_states = self.patch_generator(pixel_values)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states[:, self.num_prefix_tokens :]


@BACKBONES.register_module()
class FrozenCRADIOViT(BaseModule):
    """Frozen NVIDIA C-RADIO ViT with a trainable stride-32 feature adapter.

    Mirrors :class:`FrozenDINOv3ViT` so the two backbones stay directly
    comparable: same adapter, same stride-32 / ``out_channels`` output contract.
    """

    def __init__(
        self,
        variant,
        checkpoint,
        out_channels=512,
        forward_chunk_size=12,
        pixel_mean=(123.675, 116.28, 103.53),
        pixel_std=(58.395, 57.12, 57.375),
        amp_dtype="bfloat16",
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if variant not in CRADIO_VARIANTS:
            raise ValueError(
                "variant must be one of {}, got {!r}".format(
                    sorted(CRADIO_VARIANTS), variant
                )
            )
        if forward_chunk_size < 1:
            raise ValueError("forward_chunk_size must be positive")
        if amp_dtype not in (None, "bfloat16", "float16", "float32"):
            raise ValueError("unsupported amp_dtype: {!r}".format(amp_dtype))

        self.variant = variant
        self.checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
        self.forward_chunk_size = forward_chunk_size
        self.amp_dtype = amp_dtype
        variant_cfg = CRADIO_VARIANTS[variant]
        self.cradio = CRADIOViT(**variant_cfg)
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

        # The data pipeline hands us ImageNet-normalised pixels while RADIO's
        # InputConditioner expects [0, 1] inputs it normalises itself. Both
        # affine maps are folded into one so the composition is exact.
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor(pixel_std, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer("cond_scale", torch.ones(1, 3, 1, 1))
        self.register_buffer("cond_shift", torch.zeros(1, 3, 1, 1))

        self._checkpoint_loaded = False
        self.fp16_enabled = False
        self._freeze_cradio()

    def _freeze_cradio(self):
        self.cradio.eval()
        for parameter in self.cradio.parameters():
            parameter.requires_grad = False

    def _split_state_dict(self, state_dict):
        anchor = "patch_generator.pos_embed"
        matches = [key for key in state_dict if key.endswith(anchor)]
        if len(matches) != 1:
            raise RuntimeError(
                "expected exactly one {!r} key in {}, found {}".format(
                    anchor, self.checkpoint, len(matches)
                )
            )
        prefix = matches[0][: -len(anchor)]
        root = prefix[: -len("model.")] if prefix.endswith("model.") else ""

        # ``reg_token`` is left over from the timm ViT that RADIO replaces with
        # the CPE patch generator; it is stored but never read.
        ignored = {root + "summary_idxs", prefix + "reg_token"}

        trunk = {}
        conditioner = {}
        unexpected = []
        for key, value in state_dict.items():
            if key in ignored:
                continue
            if key.startswith(prefix):
                name = key[len(prefix) :]
                # Accept checkpoints saved before the gamma -> grandma rename.
                trunk[name.replace(".gamma", ".grandma")] = value
            elif key.startswith(root + "input_conditioner."):
                conditioner[key[len(root + "input_conditioner.") :]] = value
            else:
                unexpected.append(key)
        if unexpected:
            raise RuntimeError(
                "unrecognised keys in {}: {}".format(
                    self.checkpoint, sorted(unexpected)[:10]
                )
            )
        return trunk, conditioner

    def init_weights(self):
        if self._checkpoint_loaded:
            return
        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(
                "C-RADIO checkpoint does not exist: {}".format(self.checkpoint)
            )
        state_dict = load_file(self.checkpoint, device="cpu")
        trunk, conditioner = self._split_state_dict(state_dict)
        self.cradio.load_state_dict(trunk, strict=True)

        norm_mean = conditioner.get("norm_mean")
        norm_std = conditioner.get("norm_std")
        if norm_mean is None or norm_std is None:
            norm_mean = torch.tensor(OPENAI_CLIP_MEAN).view(3, 1, 1)
            norm_std = torch.tensor(OPENAI_CLIP_STD).view(3, 1, 1)
        norm_mean = norm_mean.float().view(1, 3, 1, 1)
        norm_std = norm_std.float().view(1, 3, 1, 1)
        self.cond_scale.copy_(self.pixel_std / (255.0 * norm_std))
        self.cond_shift.copy_((self.pixel_mean / 255.0 - norm_mean) / norm_std)

        for module in self.adapter.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self._freeze_cradio()
        self._checkpoint_loaded = True
        self._is_init = True

    def train(self, mode=True):
        super().train(mode)
        self._freeze_cradio()
        return self

    def forward(self, pixel_values):
        if not self._checkpoint_loaded:
            raise RuntimeError("init_weights() must be called before forward()")
        height, width = pixel_values.shape[-2:]
        patch = self.cradio.patch_size
        if height % patch or width % patch:
            raise ValueError(
                "input height and width must be divisible by {}, got {}x{}".format(
                    patch, height, width
                )
            )

        out_dtype = pixel_values.dtype
        # RADIO is trained under bfloat16 autocast and its dense features contain
        # very high-magnitude outlier tokens, which can overflow float16.
        autocast = self.amp_dtype not in (None, "float32") and pixel_values.is_cuda
        run_dtype = getattr(torch, self.amp_dtype) if autocast else out_dtype

        patch_features = []
        with torch.no_grad():
            for start in range(0, pixel_values.shape[0], self.forward_chunk_size):
                chunk = pixel_values[start : start + self.forward_chunk_size].float()
                chunk = chunk * self.cond_scale + self.cond_shift
                with torch.cuda.amp.autocast(enabled=autocast, dtype=run_dtype):
                    tokens = self.cradio(chunk.to(run_dtype))
                features = tokens.transpose(1, 2).reshape(
                    chunk.shape[0], -1, height // patch, width // patch
                )
                patch_features.append(features.to(out_dtype))

        features = self.adapter(torch.cat(patch_features, dim=0))
        return (features,)
