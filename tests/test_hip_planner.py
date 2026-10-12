"""Numerical planner contracts; runnable with CPU PyTorch and NumPy only.

python -m unittest discover -s tests -p test_hip_planner.py -v
"""
import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hip_planner_standalone",
    ROOT / "adzoo/drivetransformer/mmdet3d_plugin/ours/hip_planner.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
HipPlanner = MODULE.HipPlanner
ANCHORS = ROOT / "data/anchors/hip"


class TestHipPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)
        self.planner = HipPlanner(16, 2, ANCHORS, num_attention_groups=4,
                                  query_chunk_size=2)

    def make_targets(self, batch=2):
        targets = torch.zeros(batch, 4, 6, 2)
        for group, step in enumerate([0.2, 2., 0.5, 5.]):
            targets[:, group, :, 1] = torch.arange(1, 7) * step
        return targets, torch.ones(batch, 4, 6, dtype=torch.bool)

    def test_anchor_provenance_shape_and_group_order(self):
        expected = {
            "b2d_plan_spat_6x8_2m.npy": "a6bf63bfb5ea9d2afecc0a94db3d0efac325ceb155b3a10637a98d15d65b640c",
            "b2d_plan_spat_6x8_5m.npy": "94ae8c6335698dadf3a79b52dba38cbad71fd61f9ab5665c9ce2477b6283de13",
        }
        for filename, digest in expected.items():
            self.assertEqual(hashlib.sha256((ANCHORS / filename).read_bytes()).hexdigest(), digest)
        context = torch.randn(2, 1, 16)
        query, anchors = self.planner.init_queries(context)
        self.assertEqual(query.shape, (2, 480, 16))
        self.assertEqual(anchors.shape, (2, 480, 6, 2))
        self.assertTrue(torch.equal(query[:, 47], context[:, 0]))
        groups = anchors[0].reshape(10, 48, 6, 2)
        for group in (1, 4, 5, 6):
            self.assertTrue(torch.equal(groups[0], groups[group]))
        for group in (3, 7, 8, 9):
            self.assertTrue(torch.equal(groups[2], groups[group]))
        near = np.load(ANCHORS / "b2d_plan_spat_6x8_2m.npy").reshape(48, 6, 2)
        far = np.load(ANCHORS / "b2d_plan_spat_6x8_5m.npy").reshape(48, 6, 2)
        np.testing.assert_allclose(near, far[:, :1] * np.arange(1, 7)[None, :, None] / 6,
                                   atol=3e-7)

    def test_refinement_starts_at_real_absolute_anchors(self):
        query, anchors = self.planner.init_queries(torch.randn(2, 1, 16))
        position = self.planner.position(anchors)
        refined, mode, style = self.planner.refine(0, query, position, anchors)
        torch.testing.assert_close(refined, anchors)
        self.assertEqual(mode.shape, (2, 48))
        self.assertEqual(style.shape, (2, 48, 3))

    def test_spatial_assignment_then_temporal_fallback_then_ignore(self):
        targets, mask = self.make_targets(batch=3)
        prediction = torch.full((3, 10, 48, 6, 2), 100.)
        prediction[0, 1, 7] = targets[0, 1]
        prediction[0, 2, 9] = targets[0, 2]
        prediction[1, 2, 13] = targets[1, 2]
        mask[1, 1] = False
        mask[2] = False
        modes, active = self.planner.assignment(prediction, targets, mask)
        self.assertEqual(modes[:2].tolist(), [7, 13])
        self.assertEqual(active.tolist(), [True, True, False])

    def test_style_boundary_origin_and_gap_mask(self):
        speeds = torch.tensor([0., .399, .4, 2.999, 3., 10.])
        positions = torch.zeros(6, 6, 2)
        positions[..., 1] = speeds[:, None] * torch.arange(1, 7)[None] * .2
        style, active, measured = self.planner.style_target(positions, torch.ones(6, 6).bool())
        torch.testing.assert_close(measured, speeds)
        self.assertEqual(style.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertTrue(active.all())
        # Roundoff correction must not change speeds genuinely on either side
        # of a threshold, even just 1e-5 m/s away.
        sides = torch.tensor([.4 - 1e-5, .4, .4 + 1e-5, 3 - 1e-5, 3., 3 + 1e-5])
        positions[..., 1] = sides[:, None] * torch.arange(1, 7)[None] * .2
        side_style, _, _ = self.planner.style_target(positions, torch.ones(6, 6).bool())
        self.assertEqual(side_style.tolist(), [0, 1, 1, 1, 2, 2])
        positions = torch.tensor([[[0., 0.2], [float("nan"), float("nan")],
                                   [0., .6], [0., .8], [0., 1.], [0., 1.2]]])
        mask = torch.tensor([[True, False, True, True, True, True]])
        style, active, measured = self.planner.style_target(positions, mask)
        torch.testing.assert_close(measured, torch.tensor([1.]))
        self.assertEqual(style.item(), 1)
        _, active, measured = self.planner.style_target(positions, torch.zeros_like(mask))
        self.assertFalse(active.item())
        self.assertEqual(measured.item(), 0.)

    def test_decode_selects_common_mode_then_style_per_batch(self):
        pred = torch.arange(2 * 10 * 48 * 6 * 2).reshape(2, 10, 48, 6, 2).float()
        mode = torch.zeros(2, 48)
        mode[0, 4], mode[1, 11] = 5., 6.
        style = torch.zeros(2, 48, 3)
        style[0, 4, 2], style[1, 11, 1] = 7., 8.
        decoded = self.planner.decode(pred, mode, style)
        self.assertEqual(decoded["mode_index"].tolist(), [4, 11])
        self.assertEqual(decoded["style_index"].tolist(), [2, 1])
        torch.testing.assert_close(decoded["plan_spat_2m"][0], pred[0, 1, 4])
        torch.testing.assert_close(decoded["plan_temp_2hz"][1], pred[1, 2, 11])
        torch.testing.assert_close(decoded["plan_speed_5hz"][0], pred[0, 6, 4])
        torch.testing.assert_close(decoded["plan_speed_2hz"][1], pred[1, 8, 11])

    def test_common_mode_permutation_preserves_loss_and_geometry(self):
        targets, mask = self.make_targets()
        pred = torch.randn(2, 2, 10, 48, 6, 2)
        mode = torch.randn(2, 2, 48)
        style = torch.randn(2, 2, 48, 3)
        permutation = torch.randperm(48)
        original = self.planner.loss(pred, mode, style, targets, mask)
        permuted = self.planner.loss(pred[:, :, :, permutation], mode[:, :, permutation],
                                     style[:, :, permutation], targets, mask)
        for name in original:
            torch.testing.assert_close(original[name], permuted[name])
        original_decoded = self.planner.decode(pred[-1], mode[-1], style[-1])
        permuted_decoded = self.planner.decode(pred[-1, :, :, permutation], mode[-1, :, permutation],
                                              style[-1, :, permutation])
        for name in original_decoded:
            if name != "mode_index":
                torch.testing.assert_close(original_decoded[name], permuted_decoded[name])

    def test_all_masked_nan_targets_produce_zero_finite_loss_and_gradients(self):
        targets = torch.full((2, 4, 6, 2), float("nan"))
        mask = torch.zeros(2, 4, 6, dtype=torch.bool)
        pred = torch.randn(2, 2, 10, 48, 6, 2, requires_grad=True)
        mode = torch.randn(2, 2, 48, requires_grad=True)
        style = torch.randn(2, 2, 48, 3, requires_grad=True)
        losses = self.planner.loss(pred, mode, style, targets, mask)
        for value in losses.values():
            self.assertEqual(value.item(), 0.)
        sum(losses.values()).backward()
        for tensor in (pred, mode, style):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertEqual(tensor.grad.abs().sum().item(), 0.)

    def test_focal_scale_matches_official_no_avg_factor_mean(self):
        # Execute the repository's unmodified OpenMMLab CPU focal/reduction
        # functions without importing the MMCV CUDA extension or registry.
        namespace = {"torch": torch, "F": torch.nn.functional}
        sources = [
            ("mmcv/models/losses/utils.py", ("reduce_loss", "weight_reduce_loss")),
            ("mmcv/models/losses/focal_loss.py", ("py_sigmoid_focal_loss",)),
        ]
        for path, functions in sources:
            parsed = ast.parse((ROOT / path).read_text())
            body = [node for node in parsed.body
                    if isinstance(node, ast.FunctionDef) and node.name in functions]
            for node in body:
                node.decorator_list = []
            exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), namespace)
        for classes in (3, 48):
            logits = torch.randn(4, classes, requires_grad=True)
            target = torch.arange(4) % classes
            valid = torch.ones(4, dtype=torch.bool)
            one_hot = torch.nn.functional.one_hot(target, classes)
            reference = namespace["py_sigmoid_focal_loss"](
                logits, one_hot, weight=valid.float(), avg_factor=None)
            actual = self.planner._focal(logits, target, valid)
            torch.testing.assert_close(actual, reference)
            actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
            reference_grad = torch.autograd.grad(reference, logits)[0]
            torch.testing.assert_close(actual_grad, reference_grad)
            # Independent closed form at zero logits catches an accidental
            # x48 or x3 scaling change even if the external helper changes.
            closed_form = math.log(2) * (3 * classes - 2) / (16 * classes)
            result = self.planner._focal(torch.zeros(1, classes), torch.zeros(1).long(),
                                         torch.ones(1).bool())
            self.assertAlmostEqual(result.item(), closed_form, places=6)

    def camera_setup(self, outside=False):
        matrix = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
        matrix[:, :, 2] = 0.
        matrix[:, :, 2, 3] = -1. if outside else 1.
        matrix[:, :, :2, 3] = 4.
        return matrix, torch.full((1, 2, 2), 8.)

    def test_image_projection_mask_normalization_and_checkpoint_backward(self):
        attention = self.planner.image_attentions[0]
        with torch.no_grad():
            attention.offset.weight.zero_()
            attention.offset.bias.zero_()
            attention.output.weight.copy_(torch.eye(16))
            attention.output.bias.zero_()
        query = torch.randn(1, 5, 16, requires_grad=True)
        position = torch.zeros_like(query)
        anchors = torch.zeros(1, 5, 6, 2, requires_grad=True)
        # Camera 0 is visible and constant 2; camera 1 is behind and constant 100.
        feature = torch.full((1, 2, 16, 8, 8), 2., requires_grad=True)
        with torch.no_grad():
            feature[:, 1] = 100.
        matrix, hw = self.camera_setup()
        matrix[:, 1, 2, 3] = -1.
        result = self.planner.image_attention(0, query, position, anchors, feature, matrix, hw)
        torch.testing.assert_close(result - query, torch.full_like(query, 2.))
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(anchors.grad).all())
        self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertEqual(feature.grad[:, 1].abs().sum().item(), 0.)

    def test_all_projected_points_invalid_give_exact_identity_even_with_bias(self):
        query = torch.randn(1, 3, 16, requires_grad=True)
        anchors = torch.zeros(1, 3, 6, 2)
        feature = torch.randn(1, 2, 16, 8, 8)
        matrix, hw = self.camera_setup(outside=True)
        with torch.no_grad():
            self.planner.image_attentions[0].output.bias.fill_(42.)
        result = self.planner.image_attention(0, query, query, anchors, feature, matrix, hw)
        torch.testing.assert_close(result, query, rtol=0, atol=0)
        result.sum().backward()
        torch.testing.assert_close(query.grad, torch.ones_like(query))

    def test_non_square_padding_pixel_centres_and_image_scale(self):
        attention = self.planner.image_attentions[0]
        with torch.no_grad():
            attention.offset.weight.zero_()
            attention.offset.bias.zero_()
            attention.output.weight.copy_(torch.eye(16))
            attention.output.bias.zero_()
        feature = (torch.arange(8)[None] + 10 * torch.arange(4)[:, None]).float()
        feature = feature[None, None, None].expand(1, 1, 16, 4, 8).contiguous()
        query = torch.zeros(1, 1, 16)
        anchors = torch.zeros(1, 1, 6, 2)
        matrix = torch.zeros(1, 1, 4, 4)
        matrix[..., 0, 3], matrix[..., 1, 3], matrix[..., 2, 3] = 2.5, 1.5, 1.
        hw = torch.tensor([[[4., 8.]]])
        result = self.planner.image_attention(0, query, query, anchors, feature, matrix, hw)
        torch.testing.assert_close(result, torch.full_like(result, 12.))
        scaled_matrix = matrix.clone()
        scaled_matrix[..., :2, :] *= 2
        result_scaled = self.planner.image_attention(
            0, query, query, anchors, feature, scaled_matrix, hw * 2)
        torch.testing.assert_close(result_scaled, result)
        matrix[..., 0, 3] = 20.
        result_outside = self.planner.image_attention(0, query, query, anchors, feature, matrix, hw)
        torch.testing.assert_close(result_outside, query, rtol=0, atol=0)

    def test_two_layer_forward_loss_backward_reaches_anchor_and_heads(self):
        context = torch.randn(2, 1, 16, requires_grad=True)
        query, anchors = self.planner.init_queries(context)
        geometry, mode_scores, style_scores = [], [], []
        for layer in range(2):
            position = self.planner.position(anchors)
            anchors, mode, style = self.planner.refine(layer, query, position, anchors)
            geometry.append(anchors.reshape(2, 10, 48, 6, 2))
            mode_scores.append(mode)
            style_scores.append(style)
        targets, masks = self.make_targets()
        losses = self.planner.loss(torch.stack(geometry), torch.stack(mode_scores),
                                   torch.stack(style_scores), targets, masks)
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertGreater(self.planner.anchor.grad.abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(context.grad).all())
        for layer in self.planner.refinements:
            for branch in layer.regression:
                self.assertIsNotNone(branch[-1].weight.grad)
                self.assertTrue(torch.isfinite(branch[-1].weight.grad).all())

    def test_diagnostics_are_detached_masked_and_oracle_bounded(self):
        targets, masks = self.make_targets()
        prediction = torch.full((2, 10, 48, 6, 2), 100.)
        prediction[:, 1, 0] = targets[:, 1]
        prediction[:, 1, 1] = targets[:, 1] + 1.
        prediction.requires_grad_()
        mode = torch.zeros(2, 48)
        mode[0, 1], mode[1, 3] = 3., 4.
        mode.requires_grad_()
        style = torch.zeros(2, 48, 3)
        style[0, 1, 1], style[1, 3, 2] = 5., 6.
        style.requires_grad_()
        metrics = self.planner.diagnostics(prediction, mode, style,
                                           targets[:, None], masks[:, None])
        for name, value in metrics.items():
            self.assertNotIn("loss", name)
            self.assertEqual(value.ndim, 0)
            self.assertTrue(torch.isfinite(value))
            self.assertFalse(value.requires_grad)
        self.assertEqual(metrics["hip_oracle_ade_2m"].item(), 0.)
        self.assertGreater(metrics["hip_selected_ade_2m"].item(), 0.)
        self.assertLessEqual(metrics["hip_oracle_ade_2m"], metrics["hip_selected_ade_2m"])
        self.assertAlmostEqual(metrics["hip_mode_unique_fraction"].item(), 2 / 48)
        self.assertGreater(metrics["hip_mode_entropy"].item(), 0.)
        self.assertLessEqual(metrics["hip_mode_entropy"].item(), 1.)
        self.assertEqual(metrics["hip_style_stop_fraction"].item(), 0.)
        self.assertEqual(metrics["hip_style_slow_fraction"].item(), .5)
        self.assertEqual(metrics["hip_style_normal_fraction"].item(), .5)
        # Removing a sample must exclude its large selected error, not dilute
        # the first sample's valid-point ADE by the full batch size.
        masks[1] = False
        metrics = self.planner.diagnostics(prediction, mode, style, targets, masks)
        self.assertAlmostEqual(metrics["hip_selected_ade_2m"].item(), math.sqrt(2), places=6)
        self.assertAlmostEqual(metrics["hip_mode_unique_fraction"].item(), 1 / 48)
        targets.fill_(float("nan"))
        metrics = self.planner.diagnostics(prediction, mode, style, targets,
                                           torch.zeros_like(masks))
        self.assertTrue(all(value.item() == 0. for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
