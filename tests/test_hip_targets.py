"""Run with python -m unittest discover -s tests -p test_hip_targets.py.

These geometry tests deliberately need NumPy only, without CUDA/MMCV imports.
"""

import ast
from collections import OrderedDict
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "adzoo/drivetransformer/mmdet3d_plugin/datasets"
SPEC = importlib.util.spec_from_file_location("hip_targets", DATASETS / "hip_targets.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
HipRouteTargetCache = MODULE.HipRouteTargetCache
sample_radial_positions = MODULE.sample_radial_positions
rotate_hip_targets = MODULE.rotate_hip_targets


def poses_from_xy(xy, yaw=0):
    """Construct actual world-to-sensor poses, not trajectory labels."""
    xy = np.asarray(xy)
    lidar2world = np.tile(np.eye(4), (len(xy), 1, 1))
    c, s = np.cos(yaw), np.sin(yaw)
    lidar2world[:, :2, :2] = [[c, -s], [s, c]]
    lidar2world[:, :2, 3] = xy
    return np.linalg.inv(lidar2world)


class HipTargetTests(unittest.TestCase):
    def test_straight_four_group_order_and_absolute_positions(self):
        xy = np.column_stack([np.zeros(61), np.arange(61)])
        cache = HipRouteTargetCache(poses_from_xy(xy), np.arange(61))
        targets, masks = cache.targets(0)
        self.assertEqual(MODULE.HIP_GROUP_NAMES, ('temp5hz', 'spat2m', 'temp2hz', 'spat5m'))
        self.assertEqual(targets.shape, (4, 6, 2))
        self.assertEqual(targets.dtype, np.float32)
        np.testing.assert_array_equal(masks, np.ones((4, 6)))
        np.testing.assert_array_equal(targets[:, :, 0], 0)
        np.testing.assert_allclose(targets[:, :, 1], np.array([
            np.arange(2, 13, 2), np.arange(2, 13, 2),
            np.arange(5, 31, 5), np.arange(5, 31, 5)]))

    def test_curved_spatial_samples_use_radius_not_arc_length(self):
        angle = np.linspace(0, 2.5, 2501)
        radius = 20.
        xy = np.column_stack([radius * (1 - np.cos(angle)), radius * np.sin(angle)])
        cache = HipRouteTargetCache(poses_from_xy(xy), np.arange(len(xy)))
        targets, masks = cache.targets(0)
        self.assertTrue(masks[3].all())
        np.testing.assert_allclose(np.linalg.norm(targets[3], axis=-1), np.arange(5, 31, 5), atol=.02)
        # A 30 m radial displacement on this circle requires >30 m arc length.
        last = targets[3, -1]
        arc_length = radius * np.arctan2(last[1], radius - last[0])
        self.assertGreater(arc_length, 33.)

    def test_stationary_temporal_valid_spatial_unreachable_masked(self):
        xy = np.column_stack([np.zeros(61), np.minimum(np.arange(61), 3)])
        targets, masks = HipRouteTargetCache(poses_from_xy(xy), np.arange(61)).targets(0)
        np.testing.assert_array_equal(masks[0], 1)
        np.testing.assert_array_equal(masks[2], 1)
        np.testing.assert_array_equal(targets[2, :, 1], 3)
        np.testing.assert_array_equal(masks[1], [1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(masks[3], 0)
        np.testing.assert_array_equal(targets[masks == 0], 0)

    def test_scene_boundary_is_not_crossed(self):
        xy = np.column_stack([np.zeros(41), np.arange(41)])
        cache = HipRouteTargetCache(poses_from_xy(xy), np.arange(41), ['A'] * 13 + ['B'] * 28)
        _, masks = cache.targets(0)
        np.testing.assert_array_equal(masks[0], 1)
        np.testing.assert_array_equal(masks[2], [1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(masks[3], [1, 1, 0, 0, 0, 0])
        targets, masks = cache.targets(12)
        np.testing.assert_array_equal(masks, 0)
        np.testing.assert_array_equal(targets, 0)

    def test_missing_frames_use_actual_time_and_invalid_poses_are_masked(self):
        frames = np.array([0, 1, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30])
        xy = np.column_stack([np.zeros(len(frames)), frames])
        poses = poses_from_xy(xy)
        poses[4] = np.nan  # Invalid 0.5 second target.
        poses[7] = 0      # Singular 1.0 second target.
        cache = HipRouteTargetCache(poses, frames)
        targets, masks = cache.targets(0)
        np.testing.assert_array_equal(masks[0], [0, 1, 1, 1, 0, 1])
        np.testing.assert_array_equal(masks[2], [0, 0, 1, 1, 1, 1])
        self.assertTrue(np.isfinite(targets).all())
        np.testing.assert_array_equal(targets[masks == 0], 0)
        np.testing.assert_array_equal(cache.targets(4)[1], 0)

    def test_frame_period_changes_indices_not_target_times(self):
        frames = np.arange(61)
        xy = np.column_stack([frames * .05, np.zeros(61)])
        targets, masks = HipRouteTargetCache(poses_from_xy(xy), frames, frame_period_s=.05).targets(0)
        np.testing.assert_allclose(targets[0, :, 0], np.arange(1, 7) * .2)
        np.testing.assert_allclose(targets[2, :, 0], np.arange(1, 7) * .5)
        np.testing.assert_array_equal(masks[[0, 2]], 1)

    def test_time_reset_and_repeated_scene_name_do_not_bridge_segments(self):
        xy = np.column_stack([np.zeros(8), np.arange(8)])
        cache = HipRouteTargetCache(poses_from_xy(xy), [0, 1, 2, 0, 1, 2, 3, 4])
        self.assertEqual(cache.targets(0)[1][0].sum(), 1)
        cache = HipRouteTargetCache(poses_from_xy(xy), np.arange(8), ['A', 'A', 'B', 'A', 'A', 'A', 'A', 'A'])
        self.assertEqual(cache.targets(0)[1].sum(), 0)

    def test_spatial_global_nearest_then_increasing_index(self):
        targets, masks = sample_radial_positions([[0, 4], [0, 2], [0, 6], [0, 8]], 2)
        # The 4 m closest point precedes the accepted 2 m point; do not replace
        # it by another point later in the route just to make the mask valid.
        np.testing.assert_array_equal(masks, [1, 0, 1, 1, 0, 0])
        np.testing.assert_array_equal(targets[1], 0)

    def test_spatial_tolerance_is_strict_quarter_interval(self):
        _, masks = sample_radial_positions([[0, 2.5], [0, 4.49], [0, 6.5]], 2)
        np.testing.assert_array_equal(masks, [0, 1, 0, 0, 0, 0])

    def test_pose_transform_and_rotation_preserve_masks(self):
        xy = np.column_stack([10 + np.arange(61), np.full(61, 20)])
        targets, masks = HipRouteTargetCache(poses_from_xy(xy, yaw=np.pi / 2), np.arange(61)).targets(0)
        np.testing.assert_allclose(targets[0, :, 0], 0, atol=1e-6)
        np.testing.assert_allclose(targets[0, :, 1], -np.arange(2, 13, 2))
        rotated = rotate_hip_targets(targets, np.pi / 2)
        np.testing.assert_allclose(rotated[0, :, 0], np.arange(2, 13, 2))
        np.testing.assert_allclose(rotate_hip_targets(rotated, -np.pi / 2), targets, atol=1e-6)
        np.testing.assert_array_equal(masks, 1)

    def test_cache_computes_only_once_and_returns_unaliased_arrays(self):
        xy = np.column_stack([np.zeros(61), np.arange(61)])
        cache = HipRouteTargetCache(poses_from_xy(xy), np.arange(61))
        with patch.object(cache, '_compute', wraps=cache._compute) as compute:
            first, mask = cache.targets(0)
            first[:] = -999
            mask[:] = 0
            second, second_mask = cache.targets(0)
            self.assertEqual(compute.call_count, 1)
        self.assertGreater(second.sum(), 0)
        np.testing.assert_array_equal(second_mask, 1)

    def test_dataset_route_selection_and_cache_eviction(self):
        source = ast.parse((DATASETS / 'bench2drive_drivetransformer_dataset.py').read_text())
        cls = next(node for node in source.body if isinstance(node, ast.ClassDef))
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'get_hip_planning_targets')
        namespace = {'np': np, 'HipRouteTargetCache': HipRouteTargetCache}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<dataset>', 'exec'), namespace)
        xy = np.column_stack([np.zeros(14), np.arange(14)])
        infos = [{'folder': route, 'frame_idx': i,
                  'sensors': {'LIDAR_TOP': {'world2lidar': pose}}}
                 for route in ('A', 'B') for i, pose in enumerate(poses_from_xy(xy))]
        ds = SimpleNamespace(use_splited_data=False, _hip_flat_route_bounds=None,
                             _hip_route_cache=OrderedDict(), data_infos=infos,
                             cache_lenth=1, timestamp_frame_period_s=.1)
        get_targets = namespace['get_hip_planning_targets']
        first, first_mask = get_targets(ds, 0)
        self.assertEqual(list(ds._hip_route_cache), [(0, 14)])
        np.testing.assert_array_equal(get_targets(ds, 13)[1], 0)
        second, second_mask = get_targets(ds, 14)
        self.assertEqual(list(ds._hip_route_cache), [(14, 28)])
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_mask, second_mask)
        # The split-data route uses the existing route loader, and yields the
        # identical targets with a route-local rather than global sample index.
        ds.use_splited_data = True
        ds.current_route_start_idx, ds.current_route_end_idx = 14, 28
        ds.current_route_data = infos[14:]
        ds.get_data_by_index = lambda index: infos[index]
        np.testing.assert_array_equal(get_targets(ds, 14)[0], second)

    def test_dataset_pipeline_rejects_inconsistent_nonzero_3d_rotation(self):
        # Exercise the real pipeline method's early guard without importing the
        # CUDA-dependent package. It must fail before it touches image fields.
        source = ast.parse((DATASETS / 'pipelines/image_augmentation.py').read_text())
        cls = next(node for node in source.body if isinstance(node, ast.ClassDef))
        cls.decorator_list = []
        namespace = {}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), '<pipeline>', 'exec'), namespace)
        pipeline = namespace['ResizeCropFlipImage']()
        with self.assertRaisesRegex(ValueError, 'Rotating only planning labels'):
            pipeline({'hip_plan_targets': np.zeros((4, 6, 2)),
                      'aug_config': {'rotate_3d': .1}})


if __name__ == '__main__':
    unittest.main()
