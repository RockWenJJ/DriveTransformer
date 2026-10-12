"""NumPy-only HiP planning labels in the current LIDAR_TOP coordinate frame.

Group order is temporal 5 Hz, spatial 2 m, temporal 2 Hz, spatial 5 m.
Coordinates are absolute [right, forward] positions, not displacement offsets
and not DriveTransformer's fixed-distance sine-heading representation.
"""

import numpy as np


HIP_GROUP_NAMES = ("temp5hz", "spat2m", "temp2hz", "spat5m")


def rotate_hip_targets(targets, angle):
    """Rotate XY labels counterclockwise, preserving shape and zero padding.

    This is a geometry utility, not an enabled dataset augmentation. The DT
    image pipeline does not rotate its other 3D labels/projections; consequently
    HiP target collection rejects nonzero rotate_3d until all those fields are
    transformed consistently.
    """
    targets = np.asarray(targets)
    targets = targets.astype(np.result_type(targets.dtype, np.float32), copy=False)
    if targets.shape[-1] != 2 or not np.isfinite(angle):
        raise ValueError("Expected finite rotation and targets with final XY axis")
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=targets.dtype)
    return targets @ rotation.T


def sample_radial_positions(future_xy, interval, num_points=6):
    """HiP's global radial-nearest search followed by increasing-index gating.

    The search covers the entire remaining scene, not a temporal horizon and
    not accumulated arc length. Each radius independently finds its globally
    closest point; a point earlier than the preceding accepted index is masked,
    rather than searching again only among later points. Invalid labels are
    zero-filled in the absolute-coordinate representation used by this model.
    """
    xy = np.asarray(future_xy, dtype=np.float64).reshape(-1, 2)
    if interval <= 0 or num_points <= 0:
        raise ValueError("interval and num_points must be positive")
    targets = np.zeros((num_points, 2), dtype=np.float32)
    masks = np.zeros(num_points, dtype=np.float32)
    if not len(xy):
        return targets, masks
    radii = np.linalg.norm(xy, axis=-1)
    valid = np.isfinite(xy).all(axis=-1)
    last_index = -1
    for point_index, radius in enumerate(interval * np.arange(1, num_points + 1)):
        errors = np.where(valid, np.abs(radii - radius), np.inf)
        nearest = int(np.argmin(errors))
        if nearest > last_index and errors[nearest] < interval * 0.25:
            targets[point_index] = xy[nearest]
            masks[point_index] = 1
            last_index = nearest
    return targets, masks


class HipRouteTargetCache:
    """Cache poses once per route and memoize labels once per sample.

    Dataset workers retain only a bounded number of these route caches. Pose
    inversion and Python annotation traversal happen at route construction;
    the first query for each frame performs one vectorized remaining-route
    transform. Repeated queries reuse the computed labels without rescanning.
    frame_indices are used for time sampling so dropped frames cannot silently
    change a 0.2/0.5 second target into a different horizon.
    """

    def __init__(self, world2lidar, frame_indices, scene_tokens=None,
                 frame_period_s=0.1):
        poses = np.asarray(world2lidar, dtype=np.float64)
        indices = np.asarray(frame_indices, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError("world2lidar must have shape [N,4,4]")
        if indices.shape != (len(poses),):
            raise ValueError("frame_indices must have shape [N]")
        if not np.isfinite(frame_period_s) or frame_period_s <= 0:
            raise ValueError("frame_period_s must be positive and finite")
        self.poses = poses
        self.frame_indices = indices
        self.frame_period_s = float(frame_period_s)
        self.valid = np.isfinite(poses).all(axis=(1, 2)) & np.isfinite(indices)
        self.world_positions = np.full((len(poses), 4), np.nan, dtype=np.float64)
        # A singular/broken annotation must mask its labels, not poison an
        # otherwise valid batched inverse or the complete training batch.
        for index in np.flatnonzero(self.valid):
            try:
                self.world_positions[index] = np.linalg.inv(poses[index])[:, 3]
            except np.linalg.LinAlgError:
                self.valid[index] = False
        self.valid &= np.isfinite(self.world_positions).all(axis=-1)
        if scene_tokens is None:
            scene_tokens = [None] * len(poses)
        if len(scene_tokens) != len(poses):
            raise ValueError("scene_tokens must have length N")
        # Never cross a scene boundary, including a later recurrence of the
        # same scene name. Non-increasing time starts a new segment as well.
        self.segment_ends = np.empty(len(poses), dtype=np.int64)
        end = len(poses)
        for index in range(len(poses) - 1, -1, -1):
            if index + 1 < len(poses) and (
                scene_tokens[index] != scene_tokens[index + 1]
                or not np.isfinite(indices[index:index + 2]).all()
                or indices[index + 1] <= indices[index]
            ):
                end = index + 1
            self.segment_ends[index] = end
        self._targets = np.zeros((len(poses), 4, 6, 2), dtype=np.float32)
        self._masks = np.zeros((len(poses), 4, 6), dtype=np.float32)
        self._computed = np.zeros(len(poses), dtype=bool)

    @classmethod
    def from_infos(cls, infos, frame_period_s=0.1):
        return cls(
            [item["sensors"]["LIDAR_TOP"]["world2lidar"] for item in infos],
            [item["frame_idx"] for item in infos],
            [item["folder"] for item in infos], frame_period_s,
        )

    def targets(self, index):
        if not 0 <= index < len(self.poses):
            raise IndexError(index)
        if not self._computed[index]:
            if self.valid[index]:
                self._compute(index)
            self._computed[index] = True
        # Augmentations/callers must never mutate cached labels.
        return self._targets[index].copy(), self._masks[index].copy()

    def _compute(self, index):
        end = self.segment_ends[index]
        future = slice(index + 1, end)
        local_xy = (self.world_positions[future] @ self.poses[index].T)[:, :2]
        local_xy[~self.valid[future]] = np.nan
        future_frames = self.frame_indices[future]
        for group, period in ((0, 0.2), (2, 0.5)):
            desired = self.frame_indices[index] + (
                np.arange(1, 7) * period / self.frame_period_s
            )
            # Floating arithmetic can represent 0.6 / 0.1 as 6.000000000001;
            # use the same tolerance for lookup and the exact-time check.
            positions = np.searchsorted(future_frames, desired - 1e-6)
            for point, candidate in enumerate(positions):
                if candidate >= len(future_frames):
                    continue
                if (np.isclose(future_frames[candidate], desired[point],
                               rtol=0, atol=1e-6)
                        and np.isfinite(local_xy[candidate]).all()):
                    self._targets[index, group, point] = local_xy[candidate]
                    self._masks[index, group, point] = 1
        for group, interval in ((1, 2.0), (3, 5.0)):
            target, mask = sample_radial_positions(local_xy, interval)
            self._targets[index, group] = target
            self._masks[index, group] = mask
