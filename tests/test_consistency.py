import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from drivetransformer_contracts import (
    frame_timestamp, acceleration_xy, decode_fixed_distance,
    target_speed_from_time_path, apply_attention_eval_policy,
)
from tools.eval_contracts import score_ordered_match, same_class_match, center_collision_proxy


class GeometryTests(unittest.TestCase):
    def test_physical_time_matches_online_model_cadence(self):
        self.assertAlmostEqual(frame_timestamp(13) - frame_timestamp(12),
                               frame_timestamp(26, .05) - frame_timestamp(24, .05))
        self.assertAlmostEqual(frame_timestamp(13, .5) - frame_timestamp(12, .5), .5)
        with self.assertRaises(ValueError):
            frame_timestamp(0, 0)

    def test_gravity_is_not_horizontal_acceleration(self):
        np.testing.assert_allclose(acceleration_xy([2., -3., 9.81]), [2., -3.])
        np.testing.assert_allclose(acceleration_xy([2., -3., 0.]), [2., -3.])
        np.testing.assert_allclose(acceleration_xy([2., -3., 9.81], 'legacy_z_broadcast'), [9.81, 9.81])
        with self.assertRaises(ValueError):
            acceleration_xy([1., 2.])

    def test_heading_roundtrip(self):
        theta = np.deg2rad([-89, -60, -30, 0, 30, 60, 89])
        pts, diagnostic = decode_fixed_distance(np.sin(theta))
        np.testing.assert_allclose(np.arctan2(pts[:, 1], pts[:, 0]), theta, atol=1e-12)
        self.assertEqual(diagnostic['out_of_range_fraction'], 0.)

    def test_legacy_decoder_reproduces_old_points(self):
        values = np.array([-.8, 0., .5, .95])
        actual, _ = decode_fixed_distance(values, 'legacy_radians')
        expected = np.arange(1, 5)[:, None] * np.stack([np.cos(values), np.sin(values)], axis=-1)
        np.testing.assert_array_equal(actual, expected)

    def test_range_overflow_is_counted_and_finite(self):
        pts, diagnostic = decode_fixed_distance([-1.1, -1., 0., 1., 1.2])
        self.assertTrue(np.isfinite(pts).all())
        self.assertEqual(diagnostic['out_of_range_fraction'], .4)
        with self.assertRaises(ValueError):
            decode_fixed_distance([np.nan])

    def test_speed_and_stationary_path(self):
        times = np.arange(1, 31) * .1
        path = np.stack([times * 4., np.zeros(30)], axis=-1)
        self.assertAlmostEqual(target_speed_from_time_path(path), 4.)
        self.assertEqual(target_speed_from_time_path(np.zeros((30, 2))), 0.)

    def test_pid_speed_window_preserves_existing_controller(self):
        from team_code.pid_controller import DecouplePIDController
        baseline = os.environ.get('BASELINE_REPO')
        if not baseline:
            self.skipTest('BASELINE_REPO required for original-controller comparison')
        spec = importlib.util.spec_from_file_location('baseline_pid', Path(baseline) / 'team_code/pid_controller.py')
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        original, current = old.DecouplePIDController(), DecouplePIDController()
        for v, direction in [(0., 0.), (3., .2), (6., -.4), (1., .8)]:
            times = np.arange(1, 31) * .1
            time_path = np.stack([times * v, times * .1], axis=-1)
            dist_path, _ = decode_fixed_distance(np.repeat(direction, 20), 'legacy_radians')
            np.testing.assert_allclose(current.step(time_path, dist_path, 2.),
                                       original.step(time_path, dist_path, 2.), atol=1e-12)


class EvaluationTests(unittest.TestCase):
    def test_closed_loop_missing_is_not_complete(self):
        from tools.audit_closed_loop import summarize_records
        row = dict(route_id='a', status='Completed', scores=dict(score_composed=80., score_route=100.), infractions={})
        result = summarize_records([row], {'a', 'b'})
        self.assertFalse(result['complete'])
        self.assertEqual(result['missing'], ['b'])
        self.assertEqual(result['provisional_DS'], 80.)
        with self.assertRaises(ValueError):
            summarize_records([row, row], {'a', 'b'})

    def test_high_score_gets_first_match(self):
        pairs = score_ordered_match(np.array([[0., 0.]]), np.array([[.8, 0.], [.1, 0.]]),
                                    np.array([.9, .1]), 1.)
        self.assertEqual(pairs, [(0, 0)])

    def test_motion_does_not_match_wrong_class(self):
        self.assertEqual(same_class_match(np.array([[0., 0.]]), [0], np.array([[0., 0.]]), [1], [.9], 2.), [])

    def test_empty_scene_counts_as_zero_collision(self):
        self.assertEqual(center_collision_proxy([0., 0.], None, None, 0), 0.)

    def test_invalid_future_cannot_cause_collision(self):
        future = np.array([[[0., 0.]], [[10., 0.]]])
        self.assertEqual(center_collision_proxy([0., 0.], future, [[False], [True]], 0), 0.)
        self.assertEqual(center_collision_proxy([0., 0.], future, [[True], [True]], 0), 1.)

    def test_classes_come_from_dataset(self):
        from tools.ol_eval_tasks_v2 import DetAccumulator
        acc = DetAccumulator(['car', 'van'])
        acc.add(np.array([[0., 0.]]), np.array([1]), np.array([[0., 0.]]), np.array([1]), np.array([.9]))
        result = acc.summarize()
        self.assertIn('ap_van@2.0m', result)
        self.assertNotIn('ap_truck@2.0m', result)
        self.assertAlmostEqual(result['ap_van@2.0m'], 1.)


class MotionLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mmcv.models import build_loss
        from adzoo.drivetransformer.mmdet3d_plugin.ours import drivetransformer_head as module
        cls.module = module
        cls.loss_fn = build_loss(dict(type='FocalLoss', use_sigmoid=True, gamma=2., alpha=.5, loss_weight=.2))

    def compute(self, logits, labels, future_mask, frame_mask):
        return self.module.valid_motion_classification(self.loss_fn, logits, torch.zeros(len(logits), dtype=torch.long),
            torch.tensor(labels), torch.tensor(future_mask), torch.tensor(frame_mask), 9)

    def test_background_and_invalid_future_have_zero_gradient(self):
        logits = torch.zeros(4, 6, requires_grad=True)
        loss = self.compute(logits, [0, 9, 0, 0], [[1]*6, [1]*6, [0]*6, [1]*6], [1, 1, 1, 0])
        loss.backward()
        self.assertGreater(logits.grad[0].abs().sum().item(), 0.)
        self.assertEqual(logits.grad[1:].abs().sum().item(), 0.)

    def test_adding_background_does_not_dilute_loss(self):
        one = self.compute(torch.zeros(1, 6), [0], [[1]*6], [1])
        many = self.compute(torch.zeros(900, 6), [0]+[9]*899, [[1]*6]*900, [1]*900)
        torch.testing.assert_close(one, many)

    def test_empty_positive_batch_has_finite_zero_loss(self):
        logits = torch.zeros(8, 6, requires_grad=True)
        loss = self.compute(logits, [9]*8, [[0]*6]*8, [1]*8)
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad.abs().sum().item(), 0.)

    def test_sparse_ddp_normalization_is_not_clamped_to_one(self):
        local = self.compute(torch.zeros(1, 6), [0], [[1]*6], [1])
        with patch.object(self.module, 'reduce_mean', return_value=torch.tensor(.5)):
            distributed = self.compute(torch.zeros(1, 6), [0], [[1]*6], [1])
        torch.testing.assert_close(distributed, 2 * local)


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from adzoo.drivetransformer.mmdet3d_plugin.ours.drivetransformer_layers import AttentionLayer, capture_attention
        cls.layer_type = AttentionLayer
        cls.capture = staticmethod(capture_attention)

    def test_explicit_eval_deterministic_and_train_dropout_active(self):
        torch.manual_seed(13)
        layer = self.layer_type(embed_dims=64, head_dim=32, attn_drop=.5).eval()
        q = torch.randn(2, 8, 64)
        with self.capture():
            a, b = layer(q, q, q), layer(q, q, q)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        layer.train()
        with self.capture():
            a, b = layer(q, q, q), layer(q, q, q)
        self.assertFalse(torch.equal(a, b))

    @unittest.skipUnless(os.environ.get('RUN_CUDA_TESTS') == '1', 'CUDA tests explicitly opt in')
    def test_actual_xformers_eval_capture_backward_and_checkpoint(self):
        self.assertTrue(torch.cuda.is_available())
        torch.manual_seed(17)
        for head_dim in [64, 96]:
            for checkpoint in [False, True]:
                layer = self.layer_type(embed_dims=2*head_dim, head_dim=head_dim,
                                         attn_drop=.1, with_cp=checkpoint).cuda().eval()
                q = torch.randn(1, 8, 2*head_dim, device='cuda', requires_grad=True)
                a, b = layer(q, q, q), layer(q, q, q)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                with self.capture():
                    reference = layer(q, q, q)
                torch.testing.assert_close(a, reference, rtol=2e-4, atol=2e-4)
                apply_attention_eval_policy(layer, True)
                self.assertFalse(torch.equal(layer(q, q, q), layer(q, q, q)))
                layer.train()
                loss = layer(q, q, q).square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(q.grad).all())
                self.assertGreater(q.grad.abs().sum().item(), 0.)
        print('CUDA peak allocated MiB:', torch.cuda.max_memory_allocated() / 2**20)

    @unittest.skipUnless(os.environ.get('RUN_CUDA_TESTS') == '1', 'CUDA tests explicitly opt in')
    def test_actual_xformers_bfloat16(self):
        for head_dim in [64, 96]:
            layer = self.layer_type(embed_dims=2*head_dim, head_dim=head_dim,
                                    attn_drop=.1, with_cp=True).cuda().bfloat16().eval()
            q = torch.randn(1, 8, 2*head_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            a, b = layer(q, q, q), layer(q, q, q)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            with self.capture():
                reference = layer(q, q, q)
            torch.testing.assert_close(a, reference, rtol=.03, atol=.03)
            layer.train()
            layer(q, q, q).float().square().mean().backward()
            self.assertTrue(torch.isfinite(q.grad).all())
            self.assertGreater(q.grad.abs().sum().item(), 0.)


class ConfigurationTests(unittest.TestCase):
    def test_actual_optimizer_parameter_groups(self):
        from mmcv import Config
        from mmcv.optims import build_optimizer
        root = Path(__file__).resolve().parents[1] / 'adzoo/drivetransformer/configs/drivetransformer'
        cfg = Config.fromfile(str(root / 'drivetransformer_base_dinov3_vitb16_ft_consistency.py'))
        model = torch.nn.Module()
        model.img_backbone = torch.nn.Module()
        model.img_backbone.dinov3 = torch.nn.Linear(2, 2)
        model.img_backbone.adapter = torch.nn.Linear(2, 2)
        model.pts_bbox_head = torch.nn.Linear(2, 2)
        optimizer = build_optimizer(model, cfg.optimizer)
        actual = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
        for name, param in model.named_parameters():
            expected = cfg.optimizer.lr * (.1 if '.dinov3.' in name else 1.)
            self.assertAlmostEqual(actual[id(param)], expected)

    def test_corrected_and_legacy_profiles_are_explicit(self):
        from mmcv import Config
        root = Path(__file__).resolve().parents[1] / 'adzoo/drivetransformer/configs/drivetransformer'
        for name, period, accel in [('consistency', .1, 'xy'), ('legacy_replay', .5, 'legacy_z_broadcast')]:
            cfg = Config.fromfile(str(root / ('drivetransformer_base_' + name + '.py')))
            for split in ('train', 'val', 'test'):
                self.assertEqual(cfg.data[split].timestamp_frame_period_s, period)
                self.assertEqual(cfg.data[split].ego_accel_mode, accel)
        cfg = Config.fromfile(str(root / 'drivetransformer_base_dinov3_vitb16_ft_consistency.py'))
        keys = cfg.optimizer.paramwise_cfg.custom_keys
        self.assertEqual(keys['img_backbone.dinov3']['lr_mult'], .1)
        self.assertEqual(keys['img_backbone.adapter']['lr_mult'], 1.)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
