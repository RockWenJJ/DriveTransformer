"""Shared data/inference conventions; no CARLA or model imports.

Fixed-distance checkpoints predict sin(heading), not radians. Their training
targets are sampled by arc length, so the radius below remains a legacy
approximation. It is suitable for the existing heading-only PID, not for
collision checking or a claim of reconstructing the original geometric path.
"""
import numpy as np


def frame_timestamp(frame_index, frame_period_s=0.1):
    if not np.isfinite(frame_period_s) or frame_period_s <= 0:
        raise ValueError("frame_period_s must be positive, finite seconds")
    return float(frame_index) * float(frame_period_s)


def acceleration_xy(acceleration, mode="xy"):
    values = np.asarray(acceleration)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("acceleration must be a finite [ax, ay, az] vector")
    if mode == "xy":
        return values[:2].copy()
    if mode == "legacy_z_broadcast":
        return np.repeat(values[2], 2)
    raise ValueError("unknown acceleration mode: " + str(mode))


def decode_fixed_distance(values, mode="sin_heading", step_m=1.0):
    """Return [forward, right] points and range diagnostics for heading PID.

    sin_heading assumes the forward half-plane; the old scalar representation
    cannot identify the quadrant of a point behind the ego vehicle.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("fixed-distance output must be a finite 1D sequence")
    if not np.isfinite(step_m) or step_m <= 0:
        raise ValueError("step_m must be positive")
    clipped = np.clip(values, -1.0, 1.0)
    diagnostics = {"out_of_range_fraction": float(np.mean(values != clipped))}
    if mode == "sin_heading":
        # Equivalent to sin/cos(arcsin(values)), stable at +/-1.
        directions = np.stack([np.sqrt(np.maximum(1.0 - clipped**2, 0.0)), clipped], axis=-1)
    elif mode == "legacy_radians":
        directions = np.stack([np.cos(values), np.sin(values)], axis=-1)
    else:
        raise ValueError("unknown fixed-distance decoder: " + str(mode))
    return directions * (np.arange(1, len(values) + 1) * step_m)[:, None], diagnostics


def target_speed_from_time_path(path, dt_s=0.1, start_index=5, end_index=10):
    """Preserve the historical 0.6--1.1s window for labels starting at t=0.1s."""
    path = np.asarray(path)
    if dt_s <= 0 or not 0 <= start_index < end_index < len(path):
        raise ValueError("invalid speed sampling interval")
    return float(np.linalg.norm(path[end_index] - path[start_index]) / ((end_index - start_index) * dt_s))


def apply_attention_eval_policy(model, legacy_dropout=False):
    count = 0
    for module in model.modules():
        if hasattr(module, "eval_dropout"):
            module.eval_dropout = bool(legacy_dropout)
            count += 1
    return count
