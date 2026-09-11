"""Training-only latent world model, adapted from BraveGroup/LAW.

The action-aware MLP and TransformerDecoder follow LAW's public implementation.
Fixed camera/grid pooling replaces its learned view queries so that the target
uses the same FPN representation as DriveTransformer, without matching objects.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist


@contextmanager
def temporary_eval(*modules):
    """Disable target BN updates/dropout and restore every child's exact mode."""
    states = {child: child.training for module in modules if module is not None
              for child in module.modules()}
    try:
        for child in states:
            child.training = False
        yield
    finally:
        for child, training in states.items():
            child.training = training


class LAWLatentWorldModel(nn.Module):
    def __init__(self, embed_dims, num_cams=6, pool_size=(2, 4),
                 action_steps=30, num_heads=8, num_layers=2,
                 feedforward_channels=None, dropout=0.0,
                 use_action=True, loss_weight=0.2):
        super().__init__()
        if embed_dims % num_heads or min(pool_size) < 1 or action_steps < 1:
            raise ValueError('Invalid LAW dimensions, pool_size, or action_steps')
        if loss_weight < 0:
            raise ValueError('LAW loss_weight must be nonnegative')
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.pool_size = tuple(pool_size)
        self.action_steps = action_steps
        self.use_action = use_action
        self.loss_weight = float(loss_weight)
        num_tokens = num_cams * self.pool_size[0] * self.pool_size[1]
        self.slot_embedding = nn.Parameter(torch.empty(1, num_tokens, embed_dims))
        nn.init.normal_(self.slot_embedding, std=0.02)
        # Keep the architecture identical in the no-action ablation: zero only
        # the action input. Do not replace predictions with ground-truth poses.
        self.action_aware_encoder = nn.Sequential(
            nn.Linear(embed_dims + 2 * action_steps, embed_dims), nn.ReLU(),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(),
            nn.Linear(embed_dims, embed_dims))
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dims, nhead=num_heads,
            dim_feedforward=feedforward_channels or 4 * embed_dims,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def encode(self, features):
        """[B, camera, C, H, W] -> camera-major, row-major ordered latents.

        Parameter-free normalization controls the scale of MSE while keeping
        supervision on the shared FPN, rather than a free auxiliary projector.
        """
        if features.ndim != 5:
            raise ValueError('LAW expects [B, camera, C, H, W] FPN features')
        b, n, c, h, w = features.shape
        if n != self.num_cams or c != self.embed_dims:
            raise ValueError('LAW camera count / feature width does not match config')
        pooled = F.adaptive_avg_pool2d(features.float().reshape(b * n, c, h, w),
                                      self.pool_size)
        tokens = pooled.flatten(2).transpose(1, 2).reshape(b, -1, c)
        return F.layer_norm(tokens, (c,))

    def select_action(self, trajectories, mode_scores=None):
        """Use the last planning layer's predictions in lidar-frame meters.

        Base DT has one ego mode. For multimodal variants use predicted scores
        per sample; the discrete selection has no gradient, selected waypoints do.
        DT outputs positions, so there is deliberately no cumulative sum here.
        """
        if trajectories.ndim != 4 or trajectories.shape[-2:] != (self.action_steps, 2):
            raise ValueError('LAW action must have shape [B, modes, action_steps, 2]')
        if trajectories.shape[1] == 1:
            action = trajectories[:, 0]
        else:
            if mode_scores is None or mode_scores.numel() != trajectories.shape[0] * trajectories.shape[1]:
                raise ValueError('Multimodal LAW requires one predicted score per ego mode')
            indices = mode_scores.reshape(trajectories.shape[:2]).argmax(dim=1)
            action = trajectories[torch.arange(len(trajectories), device=trajectories.device), indices]
        return action.float() if self.use_action else action.float() * 0.0

    def forward(self, latents, action):
        action = action.flatten(1).unsqueeze(1).expand(-1, latents.shape[1], -1)
        conditioned = self.action_aware_encoder(torch.cat((latents, action), dim=-1))
        conditioned = conditioned + self.slot_embedding
        return self.decoder(conditioned, conditioned)

    def loss(self, predicted, target, valid):
        """Global valid-pair mean under DDP, including ranks with zero pairs.

        DDP averages parameter gradients across ranks. Divide local summed MSE
        by (global valid count / world size) to recover the global pair mean.
        All ranks take the same collectives regardless of their local mask.
        """
        valid = valid.to(device=predicted.device, dtype=torch.bool).reshape(-1)
        if predicted.shape != target.shape or len(valid) != len(predicted):
            raise ValueError('LAW prediction, target, and pair mask shapes disagree')
        pred_valid, target_valid = predicted[valid].float(), target.detach()[valid].float()
        per_pair = (pred_valid - target_valid).square().mean(dim=(1, 2))
        local_sum = per_pair.sum()  # Empty selection remains attached to graph.
        stats = torch.stack((local_sum.detach(), valid.sum().float(),
                             valid.new_tensor(valid.numel(), dtype=torch.float32)))
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats)
            world_size = dist.get_world_size()
        # Clamp global count BEFORE division by world size (not afterwards).
        loss = local_sum * world_size / stats[1].clamp_min(1)
        return dict(loss_law=loss * self.loss_weight,
                    law_mse=stats[0] / stats[1].clamp_min(1),
                    law_valid_fraction=stats[1] / stats[2].clamp_min(1),
                    law_weight=predicted.new_tensor(self.loss_weight))
