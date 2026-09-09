"""Pure NumPy evaluation primitives, intentionally not official nuScenes metrics."""
import numpy as np


def score_ordered_match(gt, pred, scores, threshold):
    gt, pred, scores = np.asarray(gt), np.asarray(pred), np.asarray(scores)
    used = set()
    pairs = []
    for pi in np.argsort(-scores, kind='stable'):
        remaining = [gi for gi in range(len(gt)) if gi not in used]
        if not remaining:
            break
        distance = np.linalg.norm(gt[remaining] - pred[pi], axis=-1)
        j = int(np.argmin(distance))
        if distance[j] <= threshold:
            gi = remaining[j]
            used.add(gi)
            pairs.append((gi, int(pi)))
    return pairs


def same_class_match(gt, gt_labels, pred, pred_labels, scores, threshold):
    pairs = []
    for label in np.unique(gt_labels):
        g = np.flatnonzero(np.asarray(gt_labels) == label)
        p = np.flatnonzero(np.asarray(pred_labels) == label)
        pairs.extend((int(g[gi]), int(p[pi])) for gi, pi in score_ordered_match(
            np.asarray(gt)[g], np.asarray(pred)[p], np.asarray(scores)[p], threshold))
    return pairs


def center_collision_proxy(ego_position, agent_future, valid_mask, horizon_index, threshold=2.0):
    if agent_future is None or not len(agent_future):
        return 0.0  # An empty scene still belongs in the ego-frame denominator.
    valid = np.asarray(valid_mask, dtype=bool)[:, horizon_index]
    centers = np.asarray(agent_future)[valid, horizon_index]
    return float((np.linalg.norm(centers - ego_position, axis=-1) < threshold).any())
