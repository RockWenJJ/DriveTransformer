"""Geometry-preserving map label preparation, independent of model state."""
import numpy as np
from shapely.geometry import LineString


def clip_map_line(points, roi):
    """Return separate positive-length components; never connect across the ROI.

    Intersect the complete polyline, rather than deleting outside vertices.
    This also retains crossing segments whose endpoints are both outside.
    """
    points = np.asarray(points)
    if len(points) < 2:
        return []
    if not np.isfinite(points).all():
        raise ValueError('Map coordinates must be finite')
    def line_parts(geometry):
        if geometry.geom_type == 'LineString':
            return [geometry] if geometry.length > 1e-6 else []
        if hasattr(geometry, 'geoms'):
            return [line for part in geometry.geoms for line in line_parts(part)]
        return []

    bounds = np.asarray(roi.bounds)
    lower, upper = bounds[:2], bounds[2:]
    # Some source lines contain hundreds of thousands of dense vertices.
    # Only segments whose bounds intersect the ROI can contribute. Keeping
    # BOTH endpoints, in consecutive source runs, preserves true crossings.
    candidate = ((np.maximum(points[:-1], points[1:]) >= lower).all(axis=1)
                 & (np.minimum(points[:-1], points[1:]) <= upper).all(axis=1))
    indices = np.flatnonzero(candidate)
    if not len(indices):
        return []
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    parts = []
    for run in runs:
        vertices = points[run[0]:run[-1] + 2]
        line = LineString(vertices)
        if line.length <= 1e-6:
            continue
        if ((vertices >= lower) & (vertices <= upper)).all():
            parts.append(line)
        else:
            parts.extend(line_parts(line.intersection(roi)))
    return parts


def deduplicate_map_lines(lines, labels):
    """Remove only exactly equal geometry in the same semantic class.

    The bounds are just a candidate index; Shapely topological equality is
    required. Nearby parallel lanes and lines with different labels survive.
    Reversed order and additional collinear vertices represent the same line.
    """
    groups = {}
    unique_lines, unique_labels = [], []
    for line, label in zip(lines, labels):
        key = (label, tuple(np.round(line.bounds, 6)))
        candidates = groups.setdefault(key, [])
        if any(line.equals(other) for other in candidates):
            continue
        candidates.append(line)
        unique_lines.append(line)
        unique_labels.append(label)
    return unique_lines, unique_labels
