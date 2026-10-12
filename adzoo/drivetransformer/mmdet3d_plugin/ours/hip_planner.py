"""HiP-inspired planning for DriveTransformer, with no MMCV dependency.

All geometry inside this module is absolute XY in the current LiDAR frame
(right, forward), in metres. Neither a future-time origin point nor a spatial
origin point is stored in the six-point outputs. Targets are ordered as
``temp5Hz, spat2m, temp2Hz, spat5m``. Planning never changes detection heads.

The original two anchor files retain their mode order. In particular the file
named ``2m`` initializes short straight rays, not points at radii 2,4,...,12m.
Shared mode assignment, cross-group feature fusion and style supervision follow
HiP-AD. Explicit mask handling, a temporal fallback when spatial GT is absent,
single-level image sampling and DT context initialization are hybrid choices.
"""

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


GROUP_NAMES = (
    "temp_5hz", "spat_2m", "temp_2hz", "spat_5m",
    "speed_5hz_stop", "speed_5hz_slow", "speed_5hz_normal",
    "speed_2hz_stop", "speed_2hz_slow", "speed_2hz_normal",
)


def _mlp(in_features, hidden_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, hidden_features), nn.ReLU(),
        nn.LayerNorm(hidden_features), nn.Linear(hidden_features, out_features),
    )


def _load_anchors(anchor_paths):
    if isinstance(anchor_paths, (str, Path)):
        root = Path(anchor_paths)
        paths = [root / "b2d_plan_spat_6x8_2m.npy", root / "b2d_plan_spat_6x8_5m.npy"]
    elif isinstance(anchor_paths, Mapping):
        paths = [anchor_paths.get("2m", anchor_paths.get(("spat", "2m"))),
                 anchor_paths.get("5m", anchor_paths.get(("spat", "5m")))]
    else:
        paths = list(anchor_paths)
    if len(paths) != 2 or any(path is None for path in paths):
        raise ValueError("anchor_paths must provide the real 2m and 5m anchor files")
    anchors = []
    for path in paths:
        array = np.load(str(path), allow_pickle=False)
        if array.shape not in ((6, 8, 6, 2), (48, 6, 2)) or not np.isfinite(array).all():
            raise ValueError("Expected finite official 48 x 6 x 2 anchors: %s" % path)
        anchors.append(torch.from_numpy(array.reshape(48, 6, 2).copy()).float())
    near, far = anchors
    return torch.stack([near, near, far, far, near, near, near, far, far, far])


class _Refinement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.regression = nn.ModuleList([_mlp(dim, dim, 12) for _ in range(6)])
        self.mode_classifier = _mlp(dim, dim, 1)
        self.style_classifier = _mlp(dim, dim, 1)
        for branch in self.regression:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)
        prior_bias = -4.59511985013459  # sigmoid prior of 0.01
        nn.init.constant_(self.mode_classifier[-1].bias, prior_bias)
        nn.init.constant_(self.style_classifier[-1].bias, prior_bias)

    def forward(self, query, position, anchors):
        batch, _, dim = query.shape
        groups = (query + position).reshape(batch, 10, 48, dim)
        aligned = groups[:, :4].sum(dim=1)
        styled = aligned[:, None] + groups[:, 4:7] + groups[:, 7:10]
        offsets = [branch(aligned).reshape(batch, 48, 6, 2)
                   for branch in self.regression[:4]]
        for branch in self.regression[4:]:
            offsets.extend(branch(styled).reshape(batch, 3, 48, 6, 2).unbind(1))
        offsets = torch.stack(offsets, dim=1)
        absolute = anchors.reshape(batch, 10, 48, 6, 2) + offsets
        mode_logits = self.mode_classifier(aligned).squeeze(-1)
        style_logits = self.style_classifier(styled).squeeze(-1).transpose(1, 2)
        return absolute.flatten(1, 2), mode_logits, style_logits


class _ImageAttention(nn.Module):
    """Single-level waypoint sampling with grouped camera/point weights."""
    def __init__(self, dim, num_groups, num_offsets, heights, ground_height,
                 query_chunk_size, checkpoint_sampling):
        super().__init__()
        if dim % num_groups:
            raise ValueError("embed_dims must be divisible by num_groups")
        self.dim, self.num_groups = dim, num_groups
        self.num_offsets = num_offsets
        self.num_points = 6 * len(heights) * num_offsets
        self.query_chunk_size = int(query_chunk_size)
        self.checkpoint_sampling = checkpoint_sampling
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        self.register_buffer("heights", torch.tensor(heights, dtype=torch.float32) + ground_height)
        self.offset = nn.Linear(dim, self.num_points * 2)
        self.camera_encoder = _mlp(12, dim, dim)
        self.weights = nn.Linear(dim, self.num_points * num_groups)
        self.output = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        nn.init.zeros_(self.output.bias)

    def _aggregate(self, feature_map, grid, weights):
        # checkpoint recomputes this intermediate instead of storing every
        # [camera, channel, query, point] sample for 480 planning queries.
        batch, cameras, dim, height, width = feature_map.shape
        queries, points = grid.shape[2:4]
        with torch.autocast(device_type=feature_map.device.type, enabled=False):
            sampled = F.grid_sample(
                feature_map.float().reshape(batch * cameras, dim, height, width),
                grid.float().reshape(batch * cameras, queries, points, 2),
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )
            sampled = sampled.reshape(batch, cameras, self.num_groups,
                                      dim // self.num_groups, queries, points)
            return torch.einsum("bngdqp,bqnpg->bqgd", sampled, weights.float()).flatten(-2)

    def forward(self, query, position, anchors, feature_map, lidar2img, image_hw):
        batch, queries, dim = query.shape
        if feature_map.ndim != 5 or feature_map.shape[0] != batch or feature_map.shape[2] != dim:
            raise ValueError("feature_map must have shape [B, cameras, embed_dims, H, W]")
        cameras = feature_map.shape[1]
        if lidar2img.shape != (batch, cameras, 4, 4) or image_hw.shape != (batch, cameras, 2):
            raise ValueError("Expected lidar2img[B,N,4,4] and padded image_hw[B,N,2] (H,W)")
        if (not bool(torch.isfinite(lidar2img).all()) or not bool(torch.isfinite(image_hw).all())
                or not bool((image_hw > 0).all())):
            raise ValueError("Projection matrices must be finite and padded image dimensions positive")
        offset = self.offset(query).float().reshape(
            batch, queries, 6, len(self.heights), self.num_offsets, 2)
        xy = anchors.float()[:, :, :, None, None, :] + offset
        z = self.heights.to(query.device).view(1, 1, 1, -1, 1, 1).expand(*xy.shape[:-1], 1)
        xyz1 = torch.cat([xy, z, torch.ones_like(z)], dim=-1).flatten(2, 4)
        # Explicit fp32 projection remains stable under outer mixed precision.
        with torch.autocast(device_type=query.device.type, enabled=False):
            projected = torch.einsum("bnij,bqpj->bnqpi", lidar2img.float(), xyz1.float())
            depth = projected[..., 2]
            wh = image_hw.float().flip(-1)[:, :, None, None, :]
            uv = projected[..., :2] / depth.clamp_min(1e-5).unsqueeze(-1) / wh
            valid = ((depth > 1e-5) & torch.isfinite(uv).all(-1)
                     & (uv >= 0).all(-1) & (uv < 1).all(-1))
            grid = torch.where(valid[..., None], uv * 2 - 1, torch.zeros_like(uv))
        camera = self.camera_encoder(lidar2img[..., :3, :].flatten(-2).to(query.dtype))
        logits = self.weights((query + position)[:, :, None, :] + camera[:, None, :, :])
        logits = logits.reshape(batch, queries, cameras * self.num_points, self.num_groups).float()
        mask = valid.permute(0, 2, 1, 3).reshape(batch, queries, cameras * self.num_points, 1)
        logits = logits.masked_fill(~mask, -1e4)
        weights = logits.softmax(dim=2) * mask.to(logits.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        weights = weights.reshape(batch, queries, cameras, self.num_points, self.num_groups)
        chunks = []
        for start in range(0, queries, self.query_chunk_size):
            chunk_grid = grid[:, :, start:start + self.query_chunk_size]
            chunk_weights = weights[:, start:start + self.query_chunk_size]
            if self.training and self.checkpoint_sampling and torch.is_grad_enabled():
                sampled = checkpoint(self._aggregate, feature_map, chunk_grid, chunk_weights,
                                     use_reentrant=False)
            else:
                sampled = self._aggregate(feature_map, chunk_grid, chunk_weights)
            chunks.append(sampled)
        fused = torch.cat(chunks, dim=1).to(query.dtype)
        update = self.output(fused)
        update = update * mask.any(dim=2).to(update.dtype)
        return query + update


class HipPlanner(nn.Module):
    """480 planning queries; each decoder layer can call the three step APIs.

    ``image_hw`` always denotes padded (height, width), not (width, height).
    Image features must have ``embed_dims`` channels. Geometry loss is fp32 and
    normalized by valid supervised XY coordinates; decoder-layer losses add.
    """
    num_groups = 10
    num_modes = 48
    num_points = 6
    num_queries = 480

    def __init__(self, embed_dims, num_layers, anchor_paths, num_attention_groups=8,
                 num_learnable_offsets=3, heights=(0., 0.5, -0.5, 1., -1.),
                 ground_height=-1.84023, position_scale=30., anchor_grad=True,
                 cls_loss_weight=0.5, reg_loss_weight=1.0, query_chunk_size=8,
                 checkpoint_sampling=True):
        super().__init__()
        if num_layers < 1 or position_scale <= 0:
            raise ValueError("num_layers and position_scale must be positive")
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.position_scale = float(position_scale)
        self.cls_loss_weight, self.reg_loss_weight = cls_loss_weight, reg_loss_weight
        self.anchor = nn.Parameter(_load_anchors(anchor_paths).flatten(0, 1), requires_grad=anchor_grad)
        self.position_encoder = _mlp(12, embed_dims, embed_dims)
        self.refinements = nn.ModuleList([_Refinement(embed_dims) for _ in range(num_layers)])
        self.image_attentions = nn.ModuleList([
            _ImageAttention(embed_dims, num_attention_groups, num_learnable_offsets,
                            heights, ground_height, query_chunk_size, checkpoint_sampling)
            for _ in range(num_layers)
        ])

    def init_queries(self, ego_context):
        if ego_context.ndim != 3 or ego_context.shape[1:] != (1, self.embed_dims):
            raise ValueError("ego_context must have shape [B,1,embed_dims]")
        batch = ego_context.shape[0]
        query = ego_context.expand(-1, self.num_queries, -1).clone()
        anchors = self.anchor.to(dtype=ego_context.dtype).unsqueeze(0).expand(batch, -1, -1, -1)
        return query, anchors

    def position(self, anchors):
        return self.position_encoder(anchors.flatten(-2) / self.position_scale)

    def image_attention(self, layer_idx, query, position, anchors, feature_map, lidar2img, image_hw):
        return self.image_attentions[layer_idx](query, position, anchors, feature_map, lidar2img, image_hw)

    def refine(self, layer_idx, query, position, anchors):
        return self.refinements[layer_idx](query, position, anchors)

    @staticmethod
    def _target_valid(targets, masks):
        return masks.bool() & torch.isfinite(targets).all(dim=-1)

    @staticmethod
    def _focal(logits, target, valid):
        # HiP-AD calls mmdet 2.28 FocalLoss without avg_factor, so its default
        # reduction averages BOTH batch and class axes (not sum / num_pos).
        # We retain that class-axis scale and only exclude invalid samples
        # from the batch denominator. For fully valid batches this is exact.
        labels = F.one_hot(target, num_classes=logits.shape[-1]).to(logits.dtype)
        prob = logits.sigmoid()
        pt = prob * labels + (1 - prob) * (1 - labels)
        alpha = 0.25 * labels + 0.75 * (1 - labels)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = loss * alpha * (1 - pt).pow(2) * valid[:, None].to(logits.dtype)
        return loss.sum() / (valid.sum().clamp_min(1) * logits.shape[-1])

    @staticmethod
    def _regression(prediction, target, valid):
        # torch.where, instead of NaN * 0, handles invalid padded coordinates.
        error = torch.where(valid[..., None], prediction - torch.nan_to_num(target),
                            torch.zeros_like(prediction))
        return error.abs().sum() / (valid.sum().clamp_min(1) * 2)

    def assignment(self, prediction, targets, masks):
        """Shared mode target; no 2m GT -> 2Hz fallback -> ignore sample."""
        valid = self._target_valid(targets, masks)
        use_spatial = valid[:, 1].any(dim=-1)
        reference = torch.where(use_spatial[:, None, None], targets[:, 1], targets[:, 2])
        reference_valid = torch.where(use_spatial[:, None], valid[:, 1], valid[:, 2])
        reference_prediction = torch.where(use_spatial[:, None, None, None],
                                           prediction[:, 1], prediction[:, 2])
        error = torch.linalg.vector_norm(reference_prediction - torch.nan_to_num(reference)[:, None], dim=-1)
        distance = torch.where(reference_valid[:, None], error, torch.zeros_like(error)).sum(-1)
        modes = distance.detach().argmin(dim=-1)
        return modes, reference_valid.any(dim=-1)

    @staticmethod
    def style_target(temporal_5hz, mask):
        """Speed from consecutive valid segments, with known valid t=0 origin."""
        valid = mask.bool() & torch.isfinite(temporal_5hz).all(-1)
        points = torch.cat([torch.zeros_like(temporal_5hz[:, :1]),
                            torch.nan_to_num(temporal_5hz)], dim=1)
        endpoint_valid = torch.cat([torch.ones_like(valid[:, :1]), valid], dim=1)
        segment_valid = endpoint_valid[:, 1:] & endpoint_valid[:, :-1]
        distance = torch.linalg.vector_norm(points[:, 1:] - points[:, :-1], dim=-1)
        distance = torch.where(segment_valid, distance, torch.zeros_like(distance))
        speed = distance.sum(-1) / (0.2 * segment_valid.sum(-1).clamp_min(1))
        boundaries = speed.new_tensor([0.4, 3.0])
        # Six float32 segment lengths and their duration can put a mathematically
        # exact 3 m/s sample one ULP below 3. Snap only machine-roundoff-sized
        # differences: < 1.5e-6 m/s at 3, < 2e-7 m/s at 0.4. The measured speed
        # remains untouched; this merely makes the inclusive lower bin edges
        # deterministic, not a new physical tolerance or a wider slow bin.
        classification_speed = speed
        for boundary in boundaries:
            tolerance = 4 * torch.finfo(speed.dtype).eps * boundary.abs()
            classification_speed = torch.where(
                (speed - boundary).abs() <= tolerance, boundary, classification_speed)
        styles = torch.bucketize(classification_speed.contiguous(), boundaries, right=True)
        return styles, segment_valid.any(-1), speed

    def loss(self, predictions, mode_logits, style_logits, targets, masks):
        """Inputs [L,B,10,48,6,2], [L,B,48], [L,B,48,3], [B,4,6,2]."""
        if targets.ndim == 5 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if predictions.ndim != 6 or predictions.shape[2:] != (10, 48, 6, 2):
            raise ValueError("predictions must have shape [L,B,10,48,6,2]")
        if (mode_logits.shape != predictions.shape[:2] + (48,)
                or style_logits.shape != predictions.shape[:2] + (48, 3)):
            raise ValueError("Classification layers must match geometry layers and batch size")
        if targets.shape != (predictions.shape[1], 4, 6, 2) or masks.shape != targets.shape[:-1]:
            raise ValueError("Expected four absolute XY target groups and corresponding masks")
        predictions, mode_logits, style_logits = predictions.float(), mode_logits.float(), style_logits.float()
        targets = targets.float()
        valid = self._target_valid(targets, masks)
        batch_index = torch.arange(targets.shape[0], device=targets.device)
        styles, style_valid, _ = self.style_target(targets[:, 0], valid[:, 0])
        zero = predictions.sum() * 0 + mode_logits.sum() * 0 + style_logits.sum() * 0
        losses = {"loss_hip_plan_cls": zero, "loss_hip_plan_reg": zero,
                  "loss_hip_style_cls": zero, "loss_hip_style_reg": zero}
        for prediction, modes_score, styles_score in zip(predictions, mode_logits, style_logits):
            modes, sample_valid = self.assignment(prediction, targets, valid)
            selected = prediction[batch_index, :, modes]
            losses["loss_hip_plan_cls"] = losses["loss_hip_plan_cls"] + (
                4 * self.cls_loss_weight * self._focal(modes_score, modes, sample_valid))
            for group in range(4):
                mask = valid[:, group] & sample_valid[:, None]
                losses["loss_hip_plan_reg"] = losses["loss_hip_plan_reg"] + (
                    self.reg_loss_weight * self._regression(selected[:, group], targets[:, group], mask))
            active_style = sample_valid & style_valid
            losses["loss_hip_style_cls"] = losses["loss_hip_style_cls"] + (
                2 * self.cls_loss_weight * self._focal(styles_score[batch_index, modes], styles, active_style))
            for group_offset, target_group in ((4, 0), (7, 2)):
                style_prediction = prediction[batch_index, group_offset + styles, modes]
                mask = valid[:, target_group] & active_style[:, None]
                losses["loss_hip_style_reg"] = losses["loss_hip_style_reg"] + (
                    self.reg_loss_weight * self._regression(style_prediction, targets[:, target_group], mask))
        return losses

    def decode(self, prediction, mode_logits, style_logits):
        """Select one common geometric mode, then a style within that mode."""
        if prediction.ndim == 4:
            prediction = prediction.reshape(-1, 10, 48, 6, 2)
        if prediction.ndim != 5 or prediction.shape[1:] != (10, 48, 6, 2):
            raise ValueError("Expected final [B,10,48,6,2] or [B,480,6,2] geometry")
        batch = torch.arange(prediction.shape[0], device=prediction.device)
        modes = mode_logits.argmax(-1)
        styles = style_logits[batch, modes].argmax(-1)
        selected = prediction[batch, :, modes]
        return {
            "plan_temp_5hz": selected[:, 0], "plan_spat_2m": selected[:, 1],
            "plan_temp_2hz": selected[:, 2], "plan_spat_5m": selected[:, 3],
            "plan_speed_5hz": prediction[batch, 4 + styles, modes],
            "plan_speed_2hz": prediction[batch, 7 + styles, modes],
            "mode_index": modes, "style_index": styles,
        }

    @torch.no_grad()
    def diagnostics(self, prediction, mode_logits, style_logits, targets, masks):
        """Detached training monitors; no term here contributes to gradients.

        Mode entropy is Shannon entropy / log(48), in [0,1], using a softmax
        only for monitoring the relative mode logits. Unique fraction counts
        modes selected in this batch / 48, so its maximum also depends on batch
        size. ADE uses only valid spatial-2m samples and points, in metres.
        Mode monitors follow the spatial/temporal-reference validity rule;
        style proportions additionally require a valid 5Hz speed segment.
        """
        if prediction.ndim == 4:
            prediction = prediction.reshape(-1, 10, 48, 6, 2)
        if targets.ndim == 5 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        batch = prediction.shape[0]
        if (prediction.shape != (batch, 10, 48, 6, 2)
                or mode_logits.shape != (batch, 48) or style_logits.shape != (batch, 48, 3)
                or targets.shape != (batch, 4, 6, 2) or masks.shape != (batch, 4, 6)):
            raise ValueError("Invalid planning diagnostics shapes")
        prediction, targets = prediction.float(), targets.float()
        valid = self._target_valid(targets, masks)
        active = valid[:, 1].any(-1) | valid[:, 2].any(-1)
        spatial_active = valid[:, 1].any(-1)
        batch_index = torch.arange(batch, device=prediction.device)
        modes = mode_logits.argmax(-1)
        styles = style_logits[batch_index, modes].argmax(-1)

        def masked_mean(value, mask):
            return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)

        log_prob = F.log_softmax(mode_logits.float(), dim=-1)
        entropy = -(log_prob.exp() * log_prob).sum(-1) / log_prob.new_tensor(48.).log()
        distance = torch.linalg.vector_norm(
            prediction[:, 1] - torch.nan_to_num(targets[:, 1])[:, None], dim=-1)
        distance = torch.where(valid[:, 1, None], distance, torch.zeros_like(distance))
        ade = distance.sum(-1) / valid[:, 1].sum(-1).clamp_min(1)[:, None]
        _, style_valid, _ = self.style_target(targets[:, 0], valid[:, 0])
        style_active = active & style_valid
        return {
            "hip_mode_entropy": masked_mean(entropy, active),
            "hip_mode_unique_fraction": prediction.new_tensor(modes[active].unique().numel() / 48.),
            "hip_selected_ade_2m": masked_mean(ade[batch_index, modes], spatial_active),
            "hip_oracle_ade_2m": masked_mean(ade.min(dim=-1).values, spatial_active),
            "hip_style_stop_fraction": masked_mean((styles == 0).float(), style_active),
            "hip_style_slow_fraction": masked_mean((styles == 1).float(), style_active),
            "hip_style_normal_fraction": masked_mean((styles == 2).float(), style_active),
        }
