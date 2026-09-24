"""DriveTransformer without the map task; retain supervised agents and ego.

Map queries, map memories, map-specific projections/FFNs/heads and map losses
are removed before optimizer construction. Agent/ego equations and losses use
the baseline implementation, including its original motion normalization.
"""
import numpy as np
import torch
from torch import nn
from mmcv.models import HEADS
from mmcv.models.builder import DETECTORS
from mmcv.utils import force_fp32, auto_fp16
from mmcv.core import multi_apply
from mmcv.core.bbox import bbox3d2result
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from .drivetransformer import DriveTransformer
from .drivetransformer_head import DriveTransformerlHead
from .utils import pos2posemb, memory_refresh, transform_reference_points, topk_gather


class AgentEgoDecoderLayer(nn.Module):
    def __init__(self, baseline):
        super().__init__()
        assert tuple(x for x in baseline.operation_order if x != 'norm') == (
            'task_self_attn', 'temporal_cross_attn', 'sensor_cross_attn', 'ffn')
        assert not baseline.cross_task_temporal_attn
        assert not baseline.directed_gated_cross_task_temporal_attn
        self.self_attention = baseline.task_self_attentions[0]
        self.temporal_attentions = nn.ModuleList([baseline.temporal_cross_attention_layers[i] for i in (0, 2)])
        self.sensor_attention = baseline.sensor_cross_attentions[0]
        self.sensor_queries = nn.ModuleList([baseline.cross_w_q[i] for i in (0, 2)])
        self.ffns = nn.ModuleList([baseline.ffns[0][i] for i in (0, 2)])
        self.agent_num_propagated = baseline.agent_num_propagated

    def forward(self, agent, ego, agent_pos, ego_pos, image, image_pos,
                agent_history, agent_history_pos, ego_history, ego_history_pos, valid):
        count = agent.shape[1]
        query = torch.cat([agent, ego], dim=1)
        pos = torch.cat([agent_pos, ego_pos], dim=1)
        query = self.self_attention(query, query, query, query_pos=pos, key_pos=pos)
        queries = [query[:, :count], query[:, count:]]
        positions = [agent_pos, ego_pos]
        for i, (history, history_pos) in enumerate([(agent_history, agent_history_pos), (ego_history, ego_history_pos)]):
            null = history.new_zeros(history.shape[0], 1, history.shape[2])
            memory = torch.cat([null, history], dim=1)
            memory_pos = torch.cat([null, history_pos], dim=1)
            invalid = ~valid.bool().squeeze(-1)
            if i == 0:
                invalid = invalid.repeat_interleave(self.agent_num_propagated, dim=-1)
            invalid = invalid.unsqueeze(1).expand(-1, queries[i].shape[1], -1)
            mask = torch.cat([torch.zeros_like(invalid[..., :1]), invalid], dim=-1)
            queries[i] = self.temporal_attentions[i](queries[i], memory, memory,
                query_pos=positions[i], key_pos=memory_pos, attn_mask=mask)
        identity = torch.cat(queries, dim=1)
        projected = torch.cat([self.sensor_queries[i](queries[i]+positions[i]) for i in range(2)], dim=1)
        query = self.sensor_attention(projected, image, image, identity=identity, key_pos=image_pos)
        return self.ffns[0](query[:, :count]), self.ffns[1](query[:, count:])


@HEADS.register_module()
class NoMapHead(DriveTransformerlHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.ego_multi_modal and self.fut_ego_fix_dist
        self.refine = self.transformer.decoder.refine
        self.decoder_layers = nn.ModuleList([AgentEgoDecoderLayer(x) for x in self.transformer.decoder.layers])
        del self.transformer
        for name in list(self._modules):
            if name.startswith(('map_', 'loss_map_')):
                delattr(self, name)
        for name in ['map_code_weights', 'map_assigner', 'map_sampler', 'map_bbox_coder']:
            if hasattr(self, name):
                delattr(self, name)
        self.map_num_query = self.map_num_query_sifted = self.map_num_propagated = 0
        self.num_query = self.agent_num_query + 1
        self.reset_memory()

    def init_weights(self):
        #self.mode_mlp.apply(self.xavier_uniform_linear)
        self.agent_cls_embedding.apply(self.xavier_uniform_linear)
        self.agent_ref_embedding.apply(self.xavier_uniform_linear)
        self.featurized_pe.apply(self.xavier_uniform_linear)
        self.spatial_alignment.apply(self.xavier_uniform_linear)
        self.time_embedding.apply(self.xavier_uniform_linear)
        self.ego_pose_pe.apply(self.xavier_uniform_linear)
        self.ego_pose_memory.apply(self.xavier_uniform_linear)
        self.ego_lcf_encoder.apply(self.xavier_uniform_linear)
        self.img_position_encoder.apply(self.xavier_uniform_linear)
        self.apply(self.init_ln)
        num_grid_per_dim_agent = int(np.sqrt(self.agent_reference_points.weight.shape[0]))
        xs = torch.linspace(self.pc_range[0], self.pc_range[3], steps=num_grid_per_dim_agent)
        ys = torch.linspace(self.pc_range[1], self.pc_range[4], steps=num_grid_per_dim_agent)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        with torch.no_grad():
            self.agent_reference_points.weight[..., 0] = x.flatten()
            self.agent_reference_points.weight[..., 1] = y.flatten()
            self.agent_reference_points.weight[..., 2] = 0.0
        nn.init.constant_(self.agent_query.weight, 0)


    def reset_memory(self):
        self.agent_memory_embedding = None
        self.ego_memory_embedding = None
        self.agent_memory_reference_point = None
        self.agent_memory_reference_yaw = None
        self.agent_memory_class = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None


    @force_fp32()
    def pre_update_memory(self, data):
        with torch.no_grad():
            x = data['prev_exists'].type(torch.float32)
            B = x.size(0)
            # update the memory at the beginning of forward.
            # memory length differs for different types of queries, but the stored number of history frames is the same
            # agent memory length = frames * propagated agents
            # for ego: memlen = frame * fut_mode (all six modes)
            # for agent ref: memlen = frame * num_propagated * 2 (one for agent itself, one for agent's future position)
            if self.agent_memory_embedding is None: # init the memory with all zeros
                self.agent_memory_embedding = x.new_zeros(B, self.memory_len_frame * self.agent_num_propagated, self.embed_dims)
                self.ego_memory_embedding = x.new_zeros(B, self.memory_len_frame, self.embed_dims)
                self.agent_memory_class = x.new_zeros(B, self.memory_len_frame * self.agent_num_propagated, self.num_classes)
                self.agent_memory_reference_point = x.new_zeros(B, self.memory_len_frame * self.agent_num_propagated, 3)
                self.agent_memory_reference_yaw = x.new_zeros(B, self.memory_len_frame * self.agent_num_propagated, 1)
                self.memory_timestamp = np.zeros((B, self.memory_len_frame, 1),dtype=np.float64)
                self.memory_egopose = x.new_zeros(B, self.memory_len_frame, 4, 4)
                self.memory_velo = x.new_zeros(B, self.memory_len_frame * self.agent_num_propagated, 2)
                self.memory_prev_exists = x.new_zeros(B, self.memory_len_frame, 1)
            else:
                # transpose from global to local
                self.memory_timestamp -= data['timestamp'].reshape(B,1,1)
                self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
                self.agent_memory_reference_point = transform_reference_points(self.agent_memory_reference_point, data['ego_pose_inv'], reverse=False)
                self.agent_memory_reference_yaw -= torch.arctan2(data['ego_pose'][:,1,0],data['ego_pose'][:,0,0]).unsqueeze(-1).unsqueeze(-1)
                # pop with FIFO policy, reset memory if meet a new sequence (prev_exists=False)
                self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len_frame], x)
                self.agent_memory_reference_point = memory_refresh(self.agent_memory_reference_point[:, :self.memory_len_frame * self.agent_num_propagated], x)
                self.agent_memory_embedding = memory_refresh(self.agent_memory_embedding[:, :self.memory_len_frame * self.agent_num_propagated], x)
                self.agent_memory_class = memory_refresh(self.agent_memory_class[:, :self.memory_len_frame * self.agent_num_propagated], x)
                self.agent_memory_reference_yaw = memory_refresh(self.agent_memory_reference_yaw[:, :self.memory_len_frame * self.agent_num_propagated], x)
                self.ego_memory_embedding = memory_refresh(self.ego_memory_embedding[:, :self.memory_len_frame], x)
                self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len_frame], x)
                self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len_frame * self.agent_num_propagated], x)
                self.memory_prev_exists = memory_refresh(self.memory_prev_exists[:, :self.memory_len_frame], x)
            self.memory_egopose  = self.memory_egopose + (1-x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)


    @force_fp32()
    def post_update_memory(
        self,
        data,
        rec_ego_pose,  # [B,1,4,4]
        agent_cls_scores,  # [N_layer,B,N_agent_query,N_type]
        agent_bbox_preds,  # [N_layer,B,N_agent_query,C]
        agent_memory, # [B,N_query,D]
        ego_memory, # [B,N_ego_fut_mode,D]
    ):
        with torch.no_grad():
            # update the memory at the end of forward
            # for agent queries, only top-K confidence scores are kept
            bs = agent_cls_scores.shape[1]
            agent_ref_point = torch.cat(
                [agent_bbox_preds[..., :2], agent_bbox_preds[..., 4:5]], dim=-1)[-1]
            agent_ref_yaw = -torch.arctan2(agent_bbox_preds[-1,..., 6], agent_bbox_preds[-1,..., 7]).unsqueeze(-1)-torch.pi/2

            rec_agent_velo = agent_bbox_preds[..., -2:][-1]
            rec_agent_memory = agent_memory
            rec_agent_score = agent_cls_scores[-1].sigmoid().topk(1, dim=-1).values
            rec_timestamp = np.zeros((rec_agent_score.shape[0],1,1), dtype=np.float64)

            _, agent_topk_indexes = torch.topk(rec_agent_score, self.agent_num_propagated, dim=1)
            rec_agent_score = topk_gather(rec_agent_score, agent_topk_indexes).detach()
            agent_ref_point = topk_gather(agent_ref_point, agent_topk_indexes).detach()
            agent_ref_yaw = topk_gather(agent_ref_yaw, agent_topk_indexes).detach()
            rec_agent_memory = topk_gather(rec_agent_memory, agent_topk_indexes).detach()
            rec_velo = topk_gather(rec_agent_velo, agent_topk_indexes).detach()
            agent_class = topk_gather(agent_cls_scores[-1], agent_topk_indexes).detach()

            # add the feature of current frame to memory
            self.agent_memory_embedding = torch.cat([rec_agent_memory, self.agent_memory_embedding], dim=1)
            self.ego_memory_embedding = torch.cat([ego_memory.detach(), self.ego_memory_embedding], dim=1)
            self.memory_timestamp = np.concatenate([rec_timestamp, self.memory_timestamp], axis=1)
            self.memory_egopose = torch.cat([rec_ego_pose.detach(), self.memory_egopose], dim=1)
            self.agent_memory_reference_point = torch.cat([agent_ref_point, self.agent_memory_reference_point], dim=1)
            self.agent_memory_reference_yaw = torch.cat([agent_ref_yaw, self.agent_memory_reference_yaw], dim=1)
            self.agent_memory_class = torch.cat([agent_class, self.agent_memory_class], dim=1)
            self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
            # transform from local to global
            self.agent_memory_reference_point = transform_reference_points(self.agent_memory_reference_point, data['ego_pose'], reverse=False)
            self.agent_memory_reference_yaw += torch.arctan2(data['ego_pose'][:,1,0],data['ego_pose'][:,0,0]).unsqueeze(-1).unsqueeze(-1)
            self.memory_timestamp += data['timestamp'].reshape(data['timestamp'].shape[0],1,1)
            self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose
            self.memory_prev_exists = torch.cat([data['prev_exists'].type(torch.float32).unsqueeze(-1).unsqueeze(-1), self.memory_prev_exists], dim=1)


    @force_fp32()
    def temporal_alignment(self, agent_query):
        # We have aligned the memory refence points with current frame in pre_update_memory.
        # Now we align the memory features and get position embeddings with memory refence points
        B = agent_query.size(0)
        agent_temp_reference_point = self.agent_memory_reference_point
        ego_temp_reference_point = self.memory_egopose[:,:,0:3,3]
        agent_temp_pos = self.agent_ref_embedding(pos2posemb(agent_temp_reference_point, self.embed_dims//2))
        ego_temp_pos = self.agent_ref_embedding(pos2posemb(ego_temp_reference_point, self.embed_dims//2))
        agent_temp_memory = self.agent_memory_embedding
        rec_ego_pose = torch.eye(4, device=agent_query.device).unsqueeze(0).unsqueeze(0).repeat(B, agent_query.size(1), 1, 1)
        memory_timestamp_tensor = torch.tensor(self.memory_timestamp,dtype=torch.float32,device=agent_query.device)

        # further tuning agent's query & ref
        if self.with_ego_pos:
            memory_ego_motion = []
            # [1, num_frame, dim] -> [1, num_frame*num_propagate_agent, dim] to align with agent pos mem queue
            for t in [memory_timestamp_tensor, self.memory_egopose[..., :3, :].flatten(-2)]:
                tmp_t = t.unsqueeze(2).repeat(1, 1, self.agent_num_propagated, 1).flatten(1, 2)
                memory_ego_motion.append(tmp_t)
            memory_ego_motion.append(self.memory_velo)
            memory_ego_motion = torch.cat(memory_ego_motion, dim=-1).float()
            memory_ego_motion = pos2posemb(memory_ego_motion, 12)
            agent_temp_pos = self.ego_pose_pe(agent_temp_pos, memory_ego_motion)
            agent_temp_memory = self.ego_pose_memory(agent_temp_memory, memory_ego_motion)

        memory_timestamp = memory_timestamp_tensor.unsqueeze(2).repeat(1, 1, self.agent_num_propagated, 1).flatten(1, 2)
        time_emded = self.time_embedding(pos2posemb(memory_timestamp, self.embed_dims).float())
        agent_cls_emded = self.agent_cls_embedding(self.agent_memory_class)
        agent_temp_pos += time_emded # temporal alignment
        agent_temp_pos += agent_cls_emded
        ego_temp_pos += time_emded[:, ::self.agent_num_propagated]

        return agent_temp_memory, agent_temp_pos, ego_temp_pos, rec_ego_pose[:, 0:1]

    @force_fp32()
    def forward(self, img_feats, img_metas, ego_lcf_feat=None, ego_fut_cmd=None,
                ego_his_trajs=None, **data):
        self.pre_update_memory(data)
        bs, cameras, channels, h, w = img_feats.shape
        location = self.prepare_location(img_feats, data)
        image = img_feats.permute(0, 1, 3, 4, 2).reshape(bs, cameras*h*w, channels)
        image_pos, cone = self.img_3d_position_embedding(data, location.clone(), img_metas)
        image = self.spatial_alignment(image, cone)
        image_pos = self.featurized_pe(image_pos, image)
        agent = self.agent_query.weight.to(img_feats.dtype).unsqueeze(0).expand(bs, -1, -1)
        agent_ref = self.agent_reference_points.weight.unsqueeze(0).repeat(bs, 1, 1)
        agent_history, agent_history_pos, ego_history_pos, rec_pose = self.temporal_alignment(agent)
        agent_pos = self.agent_ref_embedding(pos2posemb(agent_ref, self.embed_dims//2))
        agent = self.agent_prep_decoder(agent, image, agent_pos, image_pos).float()
        agent_cls = self.cls_branches[-1](agent)
        agent_box = self.reg_branches[-1](agent + agent_pos)
        agent_box[..., :2] += agent_ref[..., :2]
        agent_box[..., 4:5] += agent_ref[..., 2:3]
        mode = self.mode_embedding.weight.unsqueeze(0).expand(bs, -1, -1)
        agent_mode = agent.unsqueeze(2) + mode.unsqueeze(1)
        traj = self.traj_branches[-1](agent_mode).view(bs, agent.shape[1], self.fut_mode, self.fut_ts, 2)
        traj_cls = self.traj_cls_branches[-1](agent_mode)
        prep_cls, prep_box = agent_cls, agent_box
        ego = self.ego_lcf_encoder(torch.cat([
            ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx],
            ego_his_trajs.flatten(-2, -1), ego_fut_cmd.squeeze(1)], dim=-1))
        if ego.ndim == 2:
            ego = ego.unsqueeze(0)
        assert ego.shape[:2] == (bs, 1)
        ego_ref = ego.new_zeros(bs, 1, 3)
        time_ref = self.ego_traj_branches_fix_time[-1](ego).view(bs, 1, self.fut_ts_ego_fix_time, 2)
        dist_ref = self.ego_traj_branches_fix_dist[-1](ego).view(bs, 1, self.fut_ts_ego_fix_dist, 1)
        agent, ego, decoded = self._decode(agent, ego, agent_box, agent_cls, ego_ref,
            time_ref, dist_ref, mode, image, image_pos, agent_history,
            agent_history_pos, self.ego_memory_embedding, ego_history_pos, self.memory_prev_exists)
        classes, boxes, trajs, traj_classes, times, distances = decoded
        classes.insert(0, agent_cls); boxes.insert(0, agent_box)
        trajs.insert(0, traj); traj_classes.insert(0, traj_cls)
        times.insert(0, time_ref); distances.insert(0, dist_ref)
        self.post_update_memory(data, rec_pose, torch.stack(classes[1:]),
                                torch.stack(boxes[1:]), agent, ego)
        return dict(all_cls_scores=torch.stack(classes), all_bbox_preds=torch.stack(boxes),
                    all_traj_preds=torch.stack(trajs).flatten(-2),
                    all_traj_cls_scores=torch.stack(traj_classes).squeeze(-1),
                    ego_fut_preds_fix_time=torch.stack(times), ego_fut_preds_fix_dist=torch.stack(distances),
                    ego_traj_cls_scores=None, agent_pre_cls_scores=prep_cls, agent_pre_coord_preds=prep_box)

    def _planning_features(self, features, layer_index):
        """Optional planning-only conditioning; baseline is the identity."""
        return features

    @auto_fp16()
    def _decode(self, agent, ego, agent_box, agent_cls, ego_ref, time_ref,
                dist_ref, mode, image, image_pos, agent_history,
                agent_history_pos, ego_history, ego_history_pos, valid):
        # Match the baseline decoder's BF16 cast/autocast boundary.
        bs = agent.shape[0]
        classes, boxes, trajs, traj_classes = [], [], [], []
        times, distances = [], []
        for i, layer in enumerate(self.decoder_layers):
            agent_ref = agent_box[..., [0, 1, 4]]
            agent_pos = self.agent_ref_embedding(pos2posemb(agent_ref.detach(), self.embed_dims//2))
            agent_class_pos = agent_pos + self.agent_cls_embedding(agent_cls.detach())
            ego_pos = self.agent_ref_embedding(pos2posemb(ego_ref.detach(), self.embed_dims//2))
            ego_pos = ego_pos + self.ego_traj_ref_fix_time_embedding(pos2posemb(time_ref.detach().flatten(-2), 32))
            ego_pos = ego_pos + self.ego_traj_ref_fix_dist_embedding(pos2posemb(dist_ref.detach().flatten(-2), 32))
            agent, ego = layer(agent, ego, agent_class_pos, ego_pos, image, image_pos,
                               agent_history, agent_history_pos, ego_history,
                               ego_history_pos, valid)
            agent_cls = self.cls_branches[i](agent)
            box = self.reg_branches[i](agent + agent_pos)
            if self.refine:
                box[..., :2] += agent_box[..., :2]
                box[..., 4:5] += agent_box[..., 4:5]
            agent_box = box
            agent_mode = agent.unsqueeze(2) + mode.unsqueeze(1)
            traj = self.traj_branches[i](agent_mode).view(bs, agent.shape[1], self.fut_mode, self.fut_ts, 2)
            traj_cls = self.traj_cls_branches[i](agent_mode)
            planning = self._planning_features(ego + ego_pos, i)
            t = self.ego_traj_branches_fix_time[i](planning).view(bs, 1, self.fut_ts_ego_fix_time, 2)
            d = self.ego_traj_branches_fix_dist[i](planning).view(bs, 1, self.fut_ts_ego_fix_dist, 1)
            time_ref = t + time_ref if self.refine else t
            dist_ref = d + dist_ref if self.refine else d
            classes.append(agent_cls); boxes.append(agent_box); trajs.append(traj); traj_classes.append(traj_cls)
            times.append(time_ref); distances.append(dist_ref)
        return agent, ego, (classes, boxes, trajs, traj_classes, times, distances)

    @force_fp32()
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             gt_traj_fut_classes,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt_fix_time,
             ego_fut_masks_fix_time,
             ego_fut_gt_fix_dist,
             ego_fut_masks_fix_dist,
             ego_fut_cmd,
             ego_fut_classes,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'


        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_traj_preds = preds_dicts['all_traj_preds']
        all_traj_cls_scores = preds_dicts['all_traj_cls_scores']
        ego_fut_preds_traj_fix_time = preds_dicts['ego_fut_preds_fix_time']
        ego_fut_preds_traj_fix_dist = preds_dicts['ego_fut_preds_fix_dist'] if self.fut_ego_fix_dist else None
        ego_fut_preds_cls = preds_dicts['ego_traj_cls_scores']
        agent_pre_cls_scores = preds_dicts['agent_pre_cls_scores']
        agent_pre_coord_preds = preds_dicts['agent_pre_coord_preds']


        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        if self.pseudo_agent_instance is None:
            self.pseudo_agent_instance = [LiDARInstance3DBoxes(torch.Tensor([[0.0, 0.0, 0.0, 3.0, 1.5, 1.5, 0.0, 0.0, 0.0]]), box_dim=9), torch.zeros(1).long().to(device), torch.zeros(1, 34, device=device)]

        new_gt_bboxes_list = []
        gt_bboxes_mask = torch.ones(len(gt_labels_list), device=device)
        for i, gt_labels in enumerate(gt_labels_list):
            # if there is no gt, mask the corresponding loss, and use the pseudo agent instance
            if len(gt_labels) == 0:
                gt_bboxes_mask[i] = 0
                new_gt_bboxes_list.append(torch.cat(
                    (self.pseudo_agent_instance[0].gravity_center, self.pseudo_agent_instance[0].tensor[:, 3:]),
                    dim=1).to(device))
                gt_labels_list[i] = self.pseudo_agent_instance[1].to(device)
                # no need to adjust gt_traj_fut_classes, cuz it's already padded
                gt_attr_labels[i] = self.pseudo_agent_instance[2].to(device)
            else :
                new_gt_bboxes_list.append(torch.cat(
                    (gt_bboxes_list[i].gravity_center, gt_bboxes_list[i].tensor[:, 3:]), # (x, y, z, x_size, y_size, z_size, yaw, vx, vy)
                    dim=1).to(device))
        all_gt_bboxes_list = [new_gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_traj_fut_classes = [gt_traj_fut_classes for _ in range(num_dec_layers)]
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        all_gt_bboxes_mask = [gt_bboxes_mask for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_traj, losses_traj_cls = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_traj_preds,
            all_traj_cls_scores, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_attr_labels_list, all_gt_traj_fut_classes, all_gt_bboxes_mask, all_gt_bboxes_ignore_list)

        l0_agent_loss_cls_all, l0_aggent_loss_bbox_all = self.loss_single_box_only(agent_pre_cls_scores, agent_pre_coord_preds, all_gt_bboxes_list[0], all_gt_labels_list[0], all_gt_bboxes_mask[0])

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_traj'] = losses_traj[-1]
        loss_dict['loss_traj_cls'] = losses_traj_cls[-1]

        # Planning Loss
        ego_fut_gt_fix_time = ego_fut_gt_fix_time.squeeze(1)
        ego_fut_masks_fix_time = ego_fut_masks_fix_time.squeeze(1).squeeze(1)
        ego_fut_gt_fix_dist = ego_fut_gt_fix_dist.squeeze(1) if ego_fut_gt_fix_dist is not None else None
        ego_fut_masks_fix_dist = ego_fut_masks_fix_dist.squeeze(1).squeeze(1) if ego_fut_masks_fix_dist is not None else None

        batch, num_agent = all_traj_preds[-1].shape[:2]
        loss_plan_input = [ego_fut_preds_traj_fix_time, ego_fut_preds_traj_fix_dist, ego_fut_preds_cls, ego_fut_gt_fix_time, ego_fut_masks_fix_time,
                           ego_fut_gt_fix_dist, ego_fut_masks_fix_dist, ego_fut_classes]

        loss_plan_reg_fix_time, loss_plan_reg_fix_dist, loss_plan_cls = self.loss_planning(*loss_plan_input)
        for i in range(len(loss_plan_reg_fix_time)):
            loss_dict[f"d{i}.loss_plan_l1_fix_time"] = loss_plan_reg_fix_time[i]
            if loss_plan_reg_fix_dist is not None:
                loss_dict[f"d{i}.loss_plan_l1_fix_dist"] = loss_plan_reg_fix_dist[i]
            if self.ego_multi_modal:
                loss_dict[f"d{i}.loss_plan_cls"] = loss_plan_cls[i]

        loss_dict['d0.loss_cls_all'] = l0_agent_loss_cls_all
        loss_dict['d0.loss_bbox_all'] = l0_aggent_loss_bbox_all

        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        num_dec_layer = 0
        for loss_traj_cls_i, loss_traj_i in zip(losses_traj_cls[:-1], losses_traj[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_traj_cls'] = loss_traj_cls_i
            loss_dict[f'd{num_dec_layer}.loss_traj'] = loss_traj_i
            num_dec_layer += 1

        return loss_dict


    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        det_preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(det_preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = det_preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            trajs = preds['trajs']
            trajs = trajs.view(trajs.shape[0], self.fut_mode, -1, 2)
            yaws = bboxes.yaw
            for agent_idx in range(trajs.shape[0]):
                trajs[agent_idx] = self.rotate_agent_trajs_to_ego(bbox_yaw=yaws[agent_idx], trajs=trajs[agent_idx])
            trajs = trajs.flatten(-2, -1)
            ret_list.append([bboxes, scores, labels, trajs])

        return ret_list



@DETECTORS.register_module()
class NoMapDriveTransformer(DriveTransformer):
    """Return detection, motion and ego planning; no fabricated map output."""
    @force_fp32()
    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs_fix_time=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        **data,
    ):

        """Forward inference function for closed-loop evaluation.
        """
        ## Prepare Inputs
        for key in data:
            if key == "map_gt_bboxes_3d":
                data[key] = data[key][0]
                continue
            # in case their is a nested list
            if not isinstance(data[key], list):
                continue
            data[key] = data[key][0]
            if not isinstance(data[key], list):
                if isinstance(data[key], np.ndarray):
                    data[key] = data[key].reshape(1,-1)
                else:
                    data[key] = data[key].unsqueeze(0)
            else:
                data[key] = data[key][0].unsqueeze(0)
        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = img[0].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = img[0].new_ones(1)
        if ego_his_trajs is not None:
            ego_his_trajs = ego_his_trajs[0]
        if ego_fut_cmd is not None:
            ego_fut_cmd = ego_fut_cmd[0]
        if ego_lcf_feat is not None:
            ego_lcf_feat = ego_lcf_feat[0]

        ## Inference
        img_feats = self.extract_img_feat(img=img.unsqueeze(0), img_metas=img_metas)[self.position_level][:, 0]
        outs = self.pts_bbox_head(img_feats, img_metas, ego_lcf_feat, ego_fut_cmd, ego_his_trajs, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
        bbox_results = []
        for i, (bboxes, scores, labels, trajs) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result['trajs_3d'] = trajs.cpu()
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_result['ego_fut_preds_fix_time'] = outs['ego_fut_preds_fix_time'][-1].cpu()
            if 'ego_traj_cls_scores' in outs and outs['ego_traj_cls_scores'] is not None:
                bbox_result['ego_traj_cls_scores'] = outs['ego_traj_cls_scores'][-1].cpu()
            if 'ego_fut_preds_fix_dist' in outs and outs['ego_fut_preds_fix_dist'] is not None:
                bbox_result['ego_fut_preds_fix_dist'] = outs['ego_fut_preds_fix_dist'][-1].cpu()
            bbox_results.append(bbox_result)

        return bbox_results
