"""Ego-only DriveTransformer: no agent/map queries, heads, memories or losses.

Reuse the baseline ego equations, PETR image encoding and planning objectives.
Auxiliary modules built by the shared constructor are removed before optimizer
construction; the forward graph contains only ego self/history/image attention.
"""
import numpy as np
import torch
from torch import nn
from mmcv.models import HEADS
from mmcv.models.builder import DETECTORS
from mmcv.utils import force_fp32
from .drivetransformer import DriveTransformer
from .drivetransformer_head import DriveTransformerlHead
from .utils import pos2posemb, memory_refresh


class EgoDecoderLayer(nn.Module):
    def __init__(self, baseline):
        super().__init__()
        assert tuple(x for x in baseline.operation_order if x != 'norm') == (
            'task_self_attn', 'temporal_cross_attn', 'sensor_cross_attn', 'ffn')
        assert not baseline.cross_task_temporal_attn
        assert not baseline.directed_gated_cross_task_temporal_attn
        self.self_attention = baseline.task_self_attentions[0]
        self.temporal_attention = baseline.temporal_cross_attention_layers[2]
        self.sensor_attention = baseline.sensor_cross_attentions[0]
        self.sensor_query = baseline.cross_w_q[2]
        self.ffn = baseline.ffns[0][2]

    def forward(self, query, pos, image, image_pos, history, history_pos, valid):
        query = self.self_attention(query, query, query, query_pos=pos, key_pos=pos)
        null = history.new_zeros(history.shape[0], 1, history.shape[2])
        memory = torch.cat([null, history], dim=1)
        memory_pos = torch.cat([null, history_pos], dim=1)
        mask = torch.cat([torch.zeros_like(valid[:, :1], dtype=torch.bool),
                          ~valid.bool()], dim=1).transpose(1, 2)
        query = self.temporal_attention(query, memory, memory, query_pos=pos,
                                        key_pos=memory_pos, attn_mask=mask)
        query = self.sensor_attention(self.sensor_query(query + pos), image,
                                      image, identity=query, key_pos=image_pos)
        return self.ffn(query)


@HEADS.register_module()
class PlanningOnlyHead(DriveTransformerlHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.ego_multi_modal and self.fut_ego_fix_dist
        self.refine = self.transformer.decoder.refine
        self.ego_layers = nn.ModuleList([EgoDecoderLayer(x)
                                        for x in self.transformer.decoder.layers])
        self.ego_ref_embedding = self.agent_ref_embedding
        keep = {'ego_layers', 'ego_ref_embedding', 'ego_lcf_encoder',
                'ego_traj_ref_fix_time_embedding', 'ego_traj_ref_fix_dist_embedding',
                'ego_traj_branches_fix_time', 'ego_traj_branches_fix_dist',
                'featurized_pe', 'spatial_alignment', 'time_embedding',
                'img_position_encoder', 'loss_plan_reg_fix_time', 'loss_plan_reg_fix_dist'}
        for name in list(self._modules):
            if name not in keep:
                delattr(self, name)
        for name in ['code_weights', 'map_code_weights']:
            delattr(self, name)
        for name in ['assigner', 'sampler', 'map_assigner', 'map_sampler',
                     'bbox_coder', 'map_bbox_coder']:
            if hasattr(self, name):
                delattr(self, name)
        # No agent or map memory is allocated or propagated.
        self.agent_num_query = self.map_num_query = 0
        self.agent_num_query_sifted = self.map_num_query_sifted = 0
        self.agent_num_propagated = self.map_num_propagated = 0
        self.num_query = 1
        self.reset_memory()

    def init_weights(self):
        for name in ['ego_ref_embedding', 'featurized_pe', 'spatial_alignment',
                     'time_embedding', 'ego_lcf_encoder', 'img_position_encoder']:
            getattr(self, name).apply(self.xavier_uniform_linear)
        self.apply(self.init_ln)

    def reset_memory(self):
        self.ego_memory_embedding = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_prev_exists = None

    @force_fp32()
    def pre_update_memory(self, data):
        with torch.no_grad():
            x = data['prev_exists'].float()
            b = x.size(0)
            if self.ego_memory_embedding is None:
                self.ego_memory_embedding = x.new_zeros(b, self.memory_len_frame, self.embed_dims)
                self.memory_timestamp = np.zeros((b, self.memory_len_frame, 1), dtype=np.float64)
                self.memory_egopose = x.new_zeros(b, self.memory_len_frame, 4, 4)
                self.memory_prev_exists = x.new_zeros(b, self.memory_len_frame, 1)
            else:
                self.memory_timestamp -= data['timestamp'].reshape(b, 1, 1)
                self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
                for name in ['ego_memory_embedding', 'memory_timestamp',
                             'memory_egopose', 'memory_prev_exists']:
                    setattr(self, name, memory_refresh(getattr(self, name)[:, :self.memory_len_frame], x))
            self.memory_egopose += (1-x).view(b, 1, 1, 1) * torch.eye(4, device=x.device)

    @force_fp32()
    def forward(self, img_feats, img_metas, ego_lcf_feat=None, ego_fut_cmd=None,
                ego_his_trajs=None, **data):
        self.pre_update_memory(data)
        b, n, c, h, w = img_feats.shape
        location = self.prepare_location(img_feats, data)
        image = img_feats.permute(0, 1, 3, 4, 2).reshape(b, n*h*w, c)
        image_pos, cone = self.img_3d_position_embedding(data, location.clone(), img_metas)
        image = self.spatial_alignment(image, cone)
        image_pos = self.featurized_pe(image_pos, image)
        query = self.ego_lcf_encoder(torch.cat([
            ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx],
            ego_his_trajs.flatten(-2, -1), ego_fut_cmd.squeeze(1)], dim=-1))
        if query.ndim == 2:
            query = query.unsqueeze(0)  # Preserve the baseline's input contract.
        assert query.shape[:2] == (b, 1), query.shape
        ref = query.new_zeros(b, 1, 3)
        time_ref = self.ego_traj_branches_fix_time[-1](query).view(b, 1, self.fut_ts_ego_fix_time, 2)
        dist_ref = self.ego_traj_branches_fix_dist[-1](query).view(b, 1, self.fut_ts_ego_fix_dist, 1)
        time_outputs, dist_outputs = [time_ref], [dist_ref]
        stamp = torch.as_tensor(self.memory_timestamp, dtype=query.dtype, device=query.device)
        history_pos = self.ego_ref_embedding(pos2posemb(self.memory_egopose[..., :3, 3], self.embed_dims//2))
        history_pos = history_pos + self.time_embedding(pos2posemb(stamp, self.embed_dims).float())
        for i, layer in enumerate(self.ego_layers):
            pos = self.ego_ref_embedding(pos2posemb(ref.detach(), self.embed_dims//2))
            pos = pos + self.ego_traj_ref_fix_time_embedding(pos2posemb(time_ref.detach().flatten(-2), 32))
            pos = pos + self.ego_traj_ref_fix_dist_embedding(pos2posemb(dist_ref.detach().flatten(-2), 32))
            query = layer(query, pos, image, image_pos, self.ego_memory_embedding,
                          history_pos, self.memory_prev_exists)
            t = self.ego_traj_branches_fix_time[i](query+pos).view(b, 1, self.fut_ts_ego_fix_time, 2)
            d = self.ego_traj_branches_fix_dist[i](query+pos).view(b, 1, self.fut_ts_ego_fix_dist, 1)
            time_ref = t + time_ref if self.refine else t
            dist_ref = d + dist_ref if self.refine else d
            time_outputs.append(time_ref); dist_outputs.append(dist_ref)
        with torch.no_grad():
            self.ego_memory_embedding = torch.cat([query.detach(), self.ego_memory_embedding], dim=1)
            pose = torch.eye(4, device=query.device).view(1, 1, 4, 4).expand(b, 1, 4, 4)
            self.memory_egopose = data['ego_pose'].unsqueeze(1) @ torch.cat([pose, self.memory_egopose], dim=1)
            self.memory_timestamp = np.concatenate([np.zeros((b, 1, 1)), self.memory_timestamp], axis=1)
            self.memory_timestamp += data['timestamp'].reshape(b, 1, 1)
            self.memory_prev_exists = torch.cat([data['prev_exists'].float().view(b, 1, 1), self.memory_prev_exists], dim=1)
        return dict(ego_fut_preds_fix_time=torch.stack(time_outputs),
                    ego_fut_preds_fix_dist=torch.stack(dist_outputs), ego_traj_cls_scores=None)

    @force_fp32()
    def loss(self, preds_dicts, ego_fut_gt_fix_time, ego_fut_masks_fix_time,
             ego_fut_gt_fix_dist, ego_fut_masks_fix_dist, ego_fut_classes=None, **unused):
        time_losses, dist_losses, _ = self.loss_planning(
            preds_dicts['ego_fut_preds_fix_time'], preds_dicts['ego_fut_preds_fix_dist'], None,
            ego_fut_gt_fix_time.squeeze(1), ego_fut_masks_fix_time.squeeze(1).squeeze(1),
            ego_fut_gt_fix_dist.squeeze(1), ego_fut_masks_fix_dist.squeeze(1).squeeze(1),
            ego_fut_classes)
        losses = {}
        for i, (t, d) in enumerate(zip(time_losses, dist_losses)):
            losses[f'd{i}.loss_plan_l1_fix_time'] = t
            losses[f'd{i}.loss_plan_l1_fix_dist'] = d
        return losses


@DETECTORS.register_module()
class PlanningOnlyDriveTransformer(DriveTransformer):
    """Same training/image contract, inference returns ego predictions only."""
    @force_fp32()
    def forward_test(self, img_metas, gt_bboxes_3d=None, gt_labels_3d=None,
                     ego_fut_trajs_fix_time=None, img=None, ego_his_trajs=None,
                     ego_fut_cmd=None, ego_lcf_feat=None, **data):
        for key in data:
            if key == 'map_gt_bboxes_3d':
                data[key] = data[key][0]
                continue
            if not isinstance(data[key], list):
                continue
            data[key] = data[key][0]
            if not isinstance(data[key], list):
                data[key] = data[key].reshape(1, -1) if isinstance(data[key], np.ndarray) else data[key].unsqueeze(0)
            else:
                data[key] = data[key][0].unsqueeze(0)
        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = img[0].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = img[0].new_ones(1)
        ego_his_trajs, ego_fut_cmd, ego_lcf_feat = ego_his_trajs[0], ego_fut_cmd[0], ego_lcf_feat[0]
        feats = self.extract_img_feat(img=img.unsqueeze(0), img_metas=img_metas)[self.position_level][:, 0]
        out = self.pts_bbox_head(feats, img_metas, ego_lcf_feat, ego_fut_cmd, ego_his_trajs, **data)
        return [dict(ego_fut_cmd=ego_fut_cmd.cpu(),
                     ego_fut_preds_fix_time=out['ego_fut_preds_fix_time'][-1].cpu(),
                     ego_fut_preds_fix_dist=out['ego_fut_preds_fix_dist'][-1].cpu())]
