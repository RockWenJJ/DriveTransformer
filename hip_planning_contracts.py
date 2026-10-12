"""Explicit native-HiP XY to unchanged DT PID input conversion (NumPy only)."""
import numpy as np


def _path(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (6, 2) or not np.isfinite(value).all():
        raise ValueError('Expected finite six-point absolute XY trajectory')
    return value


def hip_control_paths(temporal, spatial, style, style_index):
    """Return 10 Hz time path and 1 m arclength path in [forward, right].

The six native 5 Hz points span 0.2..1.2 s. Interpolation preserves the
original DT controller's 0.6..1.1 s velocity window. Spatial points are
resampled by arclength; values beyond the predicted path clamp to its endpoint,
never extrapolate an unpredicted turn. PID gains and speed formula are unchanged.
    """
    temporal, spatial, style = map(_path, (temporal, spatial, style))
    style_index = int(style_index)
    if style_index not in (0, 1, 2):
        raise ValueError('Invalid style index')
    # Preserve HiP's consistency safeguard: a style that contradicts the
    # unconditioned temporal prediction falls back to that temporal prediction.
    speed = np.linalg.norm(np.diff(np.vstack([np.zeros(2), temporal]), axis=0), axis=-1).mean() / .2
    classification_speed = speed
    for boundary in (.4, 3.):
        if abs(speed - boundary) <= 4 * np.finfo(np.float32).eps * boundary:
            classification_speed = boundary
    expected_style = int(classification_speed >= .4) + int(classification_speed >= 3.)
    fallback = expected_style != style_index
    chosen = temporal if fallback else style
    native_t = np.arange(7) * .2
    target_t = np.arange(1, 13) * .1
    xy = np.vstack([np.zeros(2), chosen])
    time_xy = np.stack([np.interp(target_t, native_t, xy[:, i]) for i in range(2)], -1)

    xy = np.vstack([np.zeros(2), spatial])
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=-1))]
    keep = np.r_[True, np.diff(distances) > 1e-8]
    # Twenty points retain the unchanged DT controller's nominal 1 m indexing.
    space_xy = np.stack([np.interp(np.arange(1, 21), distances[keep], xy[keep, i])
                         for i in range(2)], -1)
    return time_xy[:, [1, 0]], space_xy[:, [1, 0]], dict(
        style_fallback=bool(fallback), selected_style=style_index,
        temporal_style=expected_style, temporal_average_speed=float(speed),
        spatial_path_length=float(distances[-1]))
