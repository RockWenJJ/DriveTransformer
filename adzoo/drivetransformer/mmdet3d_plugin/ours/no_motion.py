"""Detection and ego planning only: no map queries or agent future heads.

This opt-in architecture retains supervised agent queries and agent/ego memory.
Detection velocity regression and ego future planning remain enabled.
"""
import numpy as np
import torch
from mmcv.models import HEADS
from mmcv.models.builder import DETECTORS
from mmcv.utils import force_fp32, auto_fp16
from mmcv.core import multi_apply
from mmcv.core.bbox import bbox3d2result
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from .no_map import NoMapHead, NoMapDriveTransformer
from .utils import pos2posemb


@HEADS.register_module()
class NoMapNoMotionHead(NoMapHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.agent_num_query == self.agent_num_query_sifted
        for name in ('mode_embedding', 'traj_branches', 'traj_cls_branches',
                     'loss_traj', 'loss_traj_cls'):
            delattr(self, name)

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
            time_ref, dist_ref, image, image_pos, agent_history,
            agent_history_pos, self.ego_memory_embedding, ego_history_pos, self.memory_prev_exists)
        classes, boxes, times, distances = decoded
        classes.insert(0, agent_cls); boxes.insert(0, agent_box)
        times.insert(0, time_ref); distances.insert(0, dist_ref)
        self.post_update_memory(data, rec_pose, torch.stack(classes[1:]),
                                torch.stack(boxes[1:]), agent, ego)
        return dict(all_cls_scores=torch.stack(classes), all_bbox_preds=torch.stack(boxes),
                    ego_fut_preds_fix_time=torch.stack(times), ego_fut_preds_fix_dist=torch.stack(distances),
                    ego_traj_cls_scores=None, agent_pre_cls_scores=prep_cls, agent_pre_coord_preds=prep_box)

    @auto_fp16()
    def _decode(self, agent, ego, agent_box, agent_cls, ego_ref, time_ref,
                dist_ref, image, image_pos, agent_history,
                agent_history_pos, ego_history, ego_history_pos, valid):
        # Match the baseline decoder's BF16 cast/autocast boundary.
        bs = agent.shape[0]
        classes, boxes = [], []
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
            t = self.ego_traj_branches_fix_time[i](ego+ego_pos).view(bs, 1, self.fut_ts_ego_fix_time, 2)
            d = self.ego_traj_branches_fix_dist[i](ego+ego_pos).view(bs, 1, self.fut_ts_ego_fix_dist, 1)
            time_ref = t + time_ref if self.refine else t
            dist_ref = d + dist_ref if self.refine else d
            classes.append(agent_cls); boxes.append(agent_box)
            times.append(time_ref); distances.append(dist_ref)
        return agent, ego, (classes, boxes, times, distances)

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
            else :
                new_gt_bboxes_list.append(torch.cat(
                    (gt_bboxes_list[i].gravity_center, gt_bboxes_list[i].tensor[:, 3:]), # (x, y, z, x_size, y_size, z_size, yaw, vx, vy)
                    dim=1).to(device))
        all_gt_bboxes_list = [new_gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        all_gt_bboxes_mask = [gt_bboxes_mask for _ in range(num_dec_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single_box_only, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_bboxes_mask)

        l0_agent_loss_cls_all, l0_aggent_loss_bbox_all = self.loss_single_box_only(agent_pre_cls_scores, agent_pre_coord_preds, all_gt_bboxes_list[0], all_gt_labels_list[0], all_gt_bboxes_mask[0])

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # Planning Loss
        ego_fut_gt_fix_time = ego_fut_gt_fix_time.squeeze(1)
        ego_fut_masks_fix_time = ego_fut_masks_fix_time.squeeze(1).squeeze(1)
        ego_fut_gt_fix_dist = ego_fut_gt_fix_dist.squeeze(1) if ego_fut_gt_fix_dist is not None else None
        ego_fut_masks_fix_dist = ego_fut_masks_fix_dist.squeeze(1).squeeze(1) if ego_fut_masks_fix_dist is not None else None

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

        return loss_dict

    @force_fp32(apply_to=('preds_dicts',))
    def get_bboxes(self, preds_dicts, img_metas):
        result = []
        for i, pred in enumerate(self.bbox_coder.decode(preds_dicts)):
            boxes = pred['bboxes']
            boxes[:, 2] -= boxes[:, 5] * 0.5
            boxes = img_metas[i]['box_type_3d'](boxes, boxes.shape[-1])
            result.append([boxes, pred['scores'], pred['labels']])
        return result


@DETECTORS.register_module()
class NoMapNoMotionDriveTransformer(NoMapDriveTransformer):
    """Public inference omits unavailable map and motion outputs."""
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
        for i, (bboxes, scores, labels) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_result['ego_fut_preds_fix_time'] = outs['ego_fut_preds_fix_time'][-1].cpu()
            if 'ego_traj_cls_scores' in outs and outs['ego_traj_cls_scores'] is not None:
                bbox_result['ego_traj_cls_scores'] = outs['ego_traj_cls_scores'][-1].cpu()
            if 'ego_fut_preds_fix_dist' in outs and outs['ego_fut_preds_fix_dist'] is not None:
                bbox_result['ego_fut_preds_fix_dist'] = outs['ego_fut_preds_fix_dist'][-1].cpu()
            bbox_results.append(bbox_result)

        return bbox_results
