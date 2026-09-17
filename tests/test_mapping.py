"""Mapping regressions: real dataset/coder/assignment/loss paths, CPU only."""
import unittest

import numpy as np
import torch
from shapely.geometry import LineString
from shapely.geometry import box
from shapely.ops import unary_union

from adzoo.drivetransformer.mmdet3d_plugin.datasets.bench2drive_drivetransformer_dataset import B2D_DriveTransformer_Dataset
from adzoo.drivetransformer.mmdet3d_plugin.core.bbox.coders.map_nms_free_coder import MapNMSFreeCoder
from adzoo.drivetransformer.mmdet3d_plugin.ours.drivetransformer_head import DriveTransformerlHead
from mmcv.core.bbox.builder import build_assigner, build_sampler
from mmcv.datasets.map_utils.struct import LiDARInstanceLines
from mmcv.models import build_loss


RANGE = [-15., -30., -2., 15., 30., 2.]


def dataset(lines, kinds=None, clip=True, deduplicate=False, triggers=None):
    ds = B2D_DriveTransformer_Dataset.__new__(B2D_DriveTransformer_Dataset)
    ds.point_cloud_range = np.array(RANGE)
    ds.polyline_points_num = 20
    ds.map_clip_mode = 'geometric' if clip else 'legacy'
    ds.map_deduplicate = deduplicate
    ds.map_element_class = dict(Broken=0, Solid=1, SolidSolid=2, Center=3, TrafficLight=4, StopSign=5)
    points = [np.column_stack([p, np.zeros(len(p))]) for p in lines]
    trigger_points = [np.column_stack([p, np.zeros(len(p))]) for p in (triggers or [])]
    ds.map_infos = {'Town': dict(lane_points=points, lane_sample_points=points,
        lane_types=kinds or ['Solid'] * len(points), trigger_volumes_points=trigger_points,
        trigger_volumes_sample_points=[], trigger_volumes_types=['StopSign'] * len(trigger_points))}
    ds.get_data_by_index = lambda index: dict(town_name='Town', sensors={'LIDAR_TOP': {'world2lidar': np.eye(4)}})
    return ds


class MapLabelTests(unittest.TestCase):
    def test_exit_and_reentry_never_create_a_bridge(self):
        ds = dataset([[[-5, 0], [5, 0], [20, 0], [20, 10], [5, 10], [-5, 10]]])
        labels, vectors = ds.get_map_info(0)
        self.assertEqual(labels.tolist(), [1, 1])
        self.assertEqual(len(vectors.instance_list), 2)
        self.assertAlmostEqual(sum(g.length for g in vectors.instance_list), 40.)
        for g in vectors.instance_list:
            self.assertAlmostEqual(g.bounds[1], g.bounds[3])

    def test_sparse_crossing_is_clipped_at_boundary(self):
        labels, vectors = dataset([[[-20, 0], [20, 0]]]).get_map_info(0)
        self.assertEqual(labels.tolist(), [1])
        np.testing.assert_allclose(vectors.instance_list[0].coords, [[-15, 0], [15, 0]])

    def test_internal_line_is_unchanged(self):
        xy = [[-3, 1], [0, 2], [4, 7]]
        _, vectors = dataset([xy]).get_map_info(0)
        np.testing.assert_array_equal(vectors.instance_list[0].coords, xy)

    def test_clipping_matches_full_geometry_for_random_crossings(self):
        from adzoo.drivetransformer.mmdet3d_plugin.datasets.map_utils.geometry import clip_map_line
        rng = np.random.RandomState(17)
        roi = box(-15, -30, 15, 30)
        for _ in range(30):
            xy = rng.uniform(-60, 60, size=(40, 2))
            expected = LineString(xy).intersection(roi)
            actual = unary_union(clip_map_line(xy, roi))
            self.assertLess(expected.hausdorff_distance(actual), 1e-8)
            self.assertAlmostEqual(expected.length, actual.length, places=6)

    def test_legacy_clip_remains_reproducible(self):
        ds = dataset([[[-5, 0], [5, 0], [20, 0], [20, 10], [5, 10], [-5, 10]]], clip=False)
        _, vectors = ds.get_map_info(0)
        self.assertEqual(len(vectors.instance_list), 1)
        np.testing.assert_array_equal(vectors.instance_list[0].coords, [[-5, 0], [5, 0], [5, 10], [-5, 10]])

    def test_exact_duplicate_label_is_only_supervised_once(self):
        xy = [[-4, 0], [0, 0], [4, 0]]
        labels, vectors = dataset([xy, xy[::-1]], deduplicate=True).get_map_info(0)
        self.assertEqual(labels.tolist(), [1])
        self.assertEqual(len(vectors.instance_list), 1)

    def test_nearby_lines_and_different_classes_remain(self):
        xy = [[-4, 0], [4, 0]]
        other = [[-4, .01], [4, .01]]
        labels, _ = dataset([xy, xy, other], ['Solid', 'Broken', 'Solid'], deduplicate=True).get_map_info(0)
        self.assertEqual(labels.tolist(), [1, 0, 1])

    def test_degenerate_line_is_not_a_geometry_target(self):
        labels, _ = dataset([[[2, 2], [2, 2]]]).get_map_info(0)
        self.assertEqual(labels.numel(), 0)

    def test_empty_labels_are_integer_indices(self):
        labels, _ = dataset([]).get_map_info(0)
        self.assertEqual(labels.dtype, torch.long)

    def test_trigger_policy_remains_full_contour(self):
        inner = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
        partial = [[14, -1], [16, -1], [16, 1], [14, 1]]
        labels, vectors = dataset([], triggers=[inner, partial]).get_map_info(0)
        self.assertEqual(labels.tolist(), [5])
        self.assertTrue(vectors.instance_list[0].is_ring)


def coder(**kwargs):
    return MapNMSFreeCoder(pc_range=RANGE,
        post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35], num_classes=6,
        **kwargs)


def predictions(n=3):
    boxes = torch.tensor([[-1., -1., 1., 1.]]).repeat(n, 1)
    pts = torch.zeros(n, 20, 2)
    pts[:, :, 0] = torch.arange(n)[:, None]
    return boxes, pts


class MapDecodeTests(unittest.TestCase):
    def test_threshold_must_allow_empty_output(self):
        boxes, pts = predictions()
        output = coder(max_num=3, score_threshold=.8).decode_single(torch.full((3, 6), -20.), boxes, pts)
        self.assertEqual(output['map_scores'].numel(), 0)

    def test_unique_query_mode_does_not_repeat_geometry_under_two_classes(self):
        boxes, pts = predictions()
        logits = torch.full((3, 6), -20.)
        logits[0, 0], logits[0, 1], logits[1, 2], logits[2, 3] = 5., 4., 3., 2.
        output = coder(max_num=3, unique_queries=True).decode_single(logits, boxes, pts)
        self.assertEqual(output['map_labels'].tolist(), [0, 2, 3])
        self.assertEqual(output['map_pts'][:, 0, 0].tolist(), [0., 1., 2.])

    def test_default_ranking_matches_legacy_flatten_topk(self):
        torch.manual_seed(7)
        boxes, pts = predictions()
        logits = torch.randn(3, 6)
        expected, indices = logits.sigmoid().flatten().topk(3)
        output = coder(max_num=3).decode_single(logits, boxes, pts)
        torch.testing.assert_close(output['map_scores'], expected)
        torch.testing.assert_close(output['map_labels'], indices % 6)
        torch.testing.assert_close(output['map_pts'], pts[indices // 6])

    def test_short_or_empty_input_does_not_crash_topk(self):
        for n in [0, 1]:
            boxes, pts = predictions(n)
            for unique in [False, True]:
                output = coder(max_num=50, unique_queries=unique).decode_single(torch.zeros(n, 6), boxes, pts)
                self.assertEqual(output['map_scores'].numel(), n if unique else n * 6)

    def test_adaptive_threshold_can_reproduce_legacy_fallback(self):
        boxes, pts = predictions()
        output = coder(max_num=3, score_threshold=.8, adaptive_score_threshold=True).decode_single(
            torch.full((3, 6), -20.), boxes, pts)
        self.assertEqual(output['map_scores'].numel(), 3)


def mapping_head(pred_points=20):
    head = DriveTransformerlHead.__new__(DriveTransformerlHead)
    torch.nn.Module.__init__(head)
    head.map_num_classes = head.map_cls_out_channels = 6
    head.map_num_pts_per_vec = pred_points
    head.map_num_pts_per_gt_vec = 20
    head.map_gt_shift_pts_pattern = 'v2'
    head.map_bg_cls_weight = 0.
    head.sync_cls_avg_factor = False
    head.map_dir_interval = 1
    head.pc_range = torch.tensor(RANGE)
    head.loss_map_cls = build_loss(dict(type='FocalLoss', use_sigmoid=True, gamma=2., alpha=.25, loss_weight=2.))
    head.loss_map_pts = build_loss(dict(type='PtsL1Loss', loss_weight=1.))
    head.loss_map_dir = build_loss(dict(type='PtsDirCosLoss', loss_weight=.005))
    head.map_assigner = build_assigner(dict(type='MapHungarianAssigner3D', pc_range=RANGE,
        cls_cost=dict(type='FocalLossCost', weight=2.), reg_cost=dict(type='BBoxL1Cost', weight=0.),
        iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.), pts_cost=dict(type='OrderedPtsL1Cost', weight=1.)))
    head.map_sampler = build_sampler(dict(type='PseudoSampler'), context=head)
    return head


class MapLossTests(unittest.TestCase):
    def test_empty_frame_trains_background_without_fake_map(self):
        head = mapping_head()
        vectors = LiDARInstanceLines([], fixed_num=20, patch_size=(60, 30))
        labels = [torch.empty(0, dtype=torch.long)]
        boxes, shifts = head._prepare_map_gt([vectors], labels, torch.device('cpu'))
        logits = torch.zeros(1, 3, 6, requires_grad=True)
        pts = torch.zeros(1, 3, 20, 2, requires_grad=True)
        pred_boxes = torch.zeros(1, 3, 4)
        cls, point, direction = head.map_loss_single(logits, pred_boxes, pts, boxes, labels, shifts, torch.ones(1))
        (cls + point + direction).backward()
        self.assertGreater(cls.item(), 0.)
        self.assertTrue((logits.grad > 0).all())
        self.assertEqual(point.item(), 0.)
        self.assertEqual(direction.item(), 0.)
        self.assertEqual(pts.grad.abs().sum().item(), 0.)
        self.assertEqual(labels[0].numel(), 0)

    def test_reversed_line_uses_equivalent_point_order(self):
        head = mapping_head()
        vectors = LiDARInstanceLines([LineString([[-5, 0], [5, 0]])], fixed_num=20, patch_size=(60, 30))
        labels = [torch.tensor([1])]
        boxes, shifts = head._prepare_map_gt([vectors], labels, torch.device('cpu'))
        pts = vectors.fixed_num_sampled_points.flip(1)[None].requires_grad_()
        logits = torch.full((1, 1, 6), -8.)
        logits[0, 0, 1] = 8.
        losses = head.map_loss_single(logits, boxes[0][None], pts, boxes, labels, shifts, torch.ones(1))
        self.assertLess(losses[1].item(), 1e-6)
        self.assertLess(losses[2].item(), 1e-6)
        sum(losses).backward()
        self.assertTrue(torch.isfinite(pts.grad).all())

    def test_different_prediction_and_target_point_counts_are_supported(self):
        head = mapping_head(pred_points=10)
        vectors = LiDARInstanceLines([LineString([[-5, 0], [5, 0]])], fixed_num=20, patch_size=(60, 30))
        labels = [torch.tensor([1])]
        boxes, shifts = head._prepare_map_gt([vectors], labels, torch.device('cpu'))
        pts = torch.stack([torch.linspace(-5, 5, 10), torch.zeros(10)], -1)[None, None].requires_grad_()
        losses = head.map_loss_single(torch.zeros(1, 1, 6), boxes[0][None], pts, boxes, labels, shifts, torch.ones(1))
        self.assertLess(losses[1].item(), 1e-6)
        sum(losses).backward()
        self.assertTrue(torch.isfinite(pts.grad).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
