"""No-map integration tests using tiny CPU inputs and reference attention.

The head/decoder/loss code is real. Only the attention backend uses the
repository's explicit softmax capture path, not CUDA/xformers. These tests
establish shape, gradient and scene-reset contracts, not GPU readiness or
driving quality. Run from the repository root with CUDA_VISIBLE_DEVICES=.
"""

import copy
from pathlib import Path
import unittest

import numpy as np
import torch
from mmcv import Config
from mmcv.models import build_head
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
import adzoo.drivetransformer.mmdet3d_plugin  # noqa: F401: register real modules
from adzoo.drivetransformer.mmdet3d_plugin.ours.drivetransformer_layers import capture_attention


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'adzoo/drivetransformer/configs/drivetransformer'


def tiny_head(cfg, hybrid):
    config = copy.deepcopy(cfg.model.pts_bbox_head)
    config.update(agent_num_query=4, agent_num_query_sifted=4,
                  agent_num_propagated=2, memory_len_frame=2)
    config.bbox_coder.max_num = 4
    config.train_cfg = copy.deepcopy(cfg.model.train_cfg.pts)
    decoder = config.transformer.decoder
    decoder.update(num_layers=1, agent_num_query=4)
    decoder.transformerlayers.update(agent_query_num=4, agent_num_propagated=2,
                                      memory_len_frame=2, with_cp=hybrid)
    if not hybrid:
        config.pop('with_mapping', None)  # Exercise the compatibility default.
        config.pop('hip_planner', None)
        config.update(map_num_query=4, map_num_query_sifted=4, map_num_propagated=2)
        decoder.update(map_num_query=4)
        decoder.transformerlayers.update(map_query_num=4, map_num_propagated=2)
    head = build_head(config)
    head.init_weights()
    for module in head.modules():
        if hasattr(module, 'attn_drop'):
            module.attn_drop = 0.
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    return head


def input_data(head, prev_exists=0., time=0., requires_grad=False):
    generator = torch.Generator().manual_seed(5)
    feature = torch.randn(1, 1, head.embed_dims, 2, 2, generator=generator,
                          requires_grad=requires_grad)
    # Camera looks along LiDAR +Y; invertible perspective projection.
    matrix = torch.tensor([[10., 16., 0., 0.], [0., 16., -10., 0.],
                           [0., 1., 0., 0.], [0., 0., 0., 1.]])[None, None]
    data = dict(prev_exists=torch.tensor([prev_exists]), timestamp=np.array([time]),
                ego_pose=torch.eye(4)[None], ego_pose_inv=torch.eye(4)[None],
                lidar2img=matrix, cam_intrinsic=torch.eye(4)[None, None],
                ego_lcf_feat=torch.zeros(1, 1, 1, 9),
                ego_fut_cmd=torch.zeros(1, 1, 1, 140),
                ego_his_trajs=torch.zeros(1, 1, 2, 2))
    return feature, data


class ForbiddenMapGT:
    def __iter__(self):
        raise AssertionError('No-map model attempted to read map GT')

    def __len__(self):
        raise AssertionError('No-map model attempted to read map GT')


class NoMapHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        cls.baseline_config = Config.fromfile(str(CONFIGS / 'drivetransformer_res480_p1_map_datafix_10ep.py'))
        cls.hybrid_config = Config.fromfile(str(CONFIGS / 'drivetransformer_hip_planning_10ep.py'))
        cls.head = tiny_head(cls.hybrid_config, hybrid=True)

    def setUp(self):
        self.head.reset_memory()
        self.head.zero_grad(set_to_none=True)

    def test_baseline_mapping_default_and_detection_motion_structure(self):
        baseline = tiny_head(self.baseline_config, hybrid=False)
        self.assertTrue(baseline.with_mapping)
        for name in ('map_query', 'map_reference_points', 'map_prep_decoder',
                     'map_cls_embedding', 'map_ref_embedding', 'map_cls_branches',
                     'map_reg_branches', 'map_bbox_coder'):
            self.assertIsNotNone(getattr(baseline, name))
            self.assertIsNone(getattr(self.head, name))
        for name in ('agent_prep_decoder', 'agent_ref_embedding', 'agent_cls_embedding',
                     'cls_branches', 'reg_branches', 'traj_branches', 'traj_cls_branches'):
            old = {k: tuple(v.shape) for k, v in getattr(baseline, name).state_dict().items()}
            new = {k: tuple(v.shape) for k, v in getattr(self.head, name).state_dict().items()}
            self.assertEqual(old, new, name)
        self.assertEqual(self.head.map_num_query, 0)
        self.assertEqual(self.head.map_num_propagated, 0)
        for layer in self.head.transformer.decoder.layers:
            self.assertFalse(layer.with_mapping)
            for module in (layer.temporal_cross_attention_layers[1], layer.cross_w_q[1],
                           layer.ffns[0][1]):
                self.assertEqual(sum(p.numel() for p in module.parameters()), 0)

    def test_detection_motion_training_configuration_unchanged(self):
        old = self.baseline_config.model.pts_bbox_head
        new = self.hybrid_config.model.pts_bbox_head
        for key in ('agent_num_query', 'agent_num_query_sifted', 'agent_num_propagated',
                    'agent_prep_decoder', 'fut_ts', 'fut_mode', 'num_classes',
                    'code_size', 'code_weights', 'bbox_coder', 'loss_cls', 'loss_bbox',
                    'loss_traj', 'loss_traj_cls', 'motion_cls_valid_only'):
            self.assertEqual(old.get(key), new.get(key), key)
        self.assertEqual(self.baseline_config.model.train_cfg.pts.assigner,
                         self.hybrid_config.model.train_cfg.pts.assigner)

    def test_reference_forward_backward_without_any_map_labels(self):
        self.head.train()
        feature, data = input_data(self.head, requires_grad=True)
        boxes = LiDARInstance3DBoxes(torch.tensor([[0., 5., 0., 2., 4., 1.5, 0., 0., 0.]]), box_dim=9)
        attrs = torch.zeros(1, 34)
        attrs[:, 12:18] = 1
        # The upstream head's lazy dummy allocation is CUDA-specific. Supply
        # the exact dummy container on CPU; the nonempty GT does not use it.
        self.head.pseudo_agent_instance = [boxes, torch.zeros(1).long(), attrs]
        targets = torch.zeros(1, 1, 4, 6, 2)
        targets[0, 0, :, :, 1] = torch.tensor([2., 4., 6., 8., 10., 12.])
        with capture_attention() as records:
            output = self.head(feature, [{}], **data)
            self.assertEqual(output['map_all_cls_scores'].shape, (2, 1, 0, self.head.map_num_classes))
            self.assertEqual(output['map_all_bbox_preds'].shape, (2, 1, 0, 4))
            self.assertEqual(output['hip_plan_preds'].shape, (1, 1, 10, 48, 6, 2))
            self.assertEqual(self.head.map_memory_embedding.shape[1], 0)
            loss = self.head.loss(
                gt_bboxes_list=[boxes], gt_labels_list=[torch.zeros(1).long()],
                gt_traj_fut_classes=torch.zeros(1, 100).long(),
                map_gt_bboxes_list=ForbiddenMapGT(), map_gt_labels_list=ForbiddenMapGT(),
                preds_dicts=output, ego_fut_gt_fix_time=torch.zeros(1, 1, 30, 2),
                ego_fut_masks_fix_time=torch.ones(1, 1, 30),
                ego_fut_gt_fix_dist=torch.zeros(1, 1, 20, 1),
                ego_fut_masks_fix_dist=torch.ones(1, 1, 20),
                ego_fut_cmd=data['ego_fut_cmd'], ego_fut_classes=torch.zeros(1, 1).long(),
                gt_attr_labels=[attrs], hip_plan_targets=targets,
                hip_plan_masks=torch.ones(1, 1, 4, 6))
            self.assertFalse(any('map' in key for key in loss))
            self.assertTrue({'loss_cls', 'loss_bbox', 'loss_traj', 'loss_traj_cls'} <= set(loss))
            self.assertTrue(all(torch.isfinite(value).all() for value in loss.values()))
            sum(loss.values()).backward()
            self.assertTrue(records)
        self.assertIsNotNone(feature.grad)
        self.assertTrue(torch.isfinite(feature.grad).all())
        for module in (self.head.cls_branches[0], self.head.reg_branches[0],
                       self.head.traj_branches[0], self.head.hip_planner.refinements[0]):
            grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))
        self.assertTrue(torch.isfinite(self.head.hip_planner.anchor.grad).all())

    def test_scene_reset_matches_fresh_memory_and_no_map_decode(self):
        self.head.eval()
        feature, data = input_data(self.head)
        with torch.no_grad(), capture_attention():
            first = self.head(feature, [{}], **data)
            _, follow = input_data(self.head, prev_exists=1., time=.1)
            self.head(feature + .2, [{}], **follow)
            _, reset = input_data(self.head, prev_exists=0., time=.2)
            restarted = self.head(feature, [{}], **reset)
        for key in ('all_cls_scores', 'all_bbox_preds', 'all_traj_preds', 'hip_plan_preds'):
            torch.testing.assert_close(first[key], restarted[key], rtol=1e-5, atol=1e-5)
        decoded = self.head.get_bboxes(restarted, [{'box_type_3d': LiDARInstance3DBoxes}])
        self.assertEqual(len(decoded), 1)
        self.assertEqual(len(decoded[0]), 8)
        self.assertEqual(decoded[0][4:], [None, None, None, None])


if __name__ == '__main__':
    unittest.main()
