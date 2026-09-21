"""Read-only, deterministic audit of B2D vector-map labels (no images/GPU)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import time

import joblib
import numpy as np
from shapely.geometry import LineString, box

CLASSES = ['Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight', 'StopSign']


def components(g):
    if g.geom_type == 'LineString':
        return [g] if g.length > 1e-6 else []
    if hasattr(g, 'geoms'):
        return [p for part in g.geoms for p in components(part)]
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split', default='val', choices=['train', 'val'])
    p.add_argument('--routes', type=int, default=50)
    p.add_argument('--samples-per-route', type=int, default=3)
    args = p.parse_args()
    info_dir = args.repo / 'data/infos'
    meta_path = info_dir / ('b2d_infos_v1_%s_drivetransformer_meta.pkl' % args.split)
    meta = pickle.loads(meta_path.read_bytes())
    print('metadata', type(meta), list(meta), flush=True)
    map_path = info_dir / 'b2d_map_infos.pkl'
    maps = joblib.load(map_path)
    print('maps loaded', list(maps), flush=True)
    towns = {k: {'lanes': dict(Counter(v['lane_types'])),
                 'triggers': dict(Counter(v['trigger_volumes_types']))} for k, v in maps.items()}
    roi = box(-15, -30, 15, 30)
    rows, examples, duplicate_examples = [], [], []
    routes = meta['routes_names']
    route_indices = np.linspace(0, len(routes) - 1, min(args.routes, len(routes)), dtype=int)
    clips_dir = info_dir / meta['infos_dir_name'].replace('dtr', 'drivetransformer')
    for ri in route_indices:
        clip_path = clips_dir / (routes[ri] + '.pkl')
        clip = pickle.loads(clip_path.read_bytes())
        for fi in np.unique(np.linspace(0, len(clip)-1, args.samples_per_route, dtype=int)):
            ann = clip[fi]
            town = maps[ann['town_name']]
            w2l = np.asarray(ann['sensors']['LIDAR_TOP']['world2lidar'])
            ego = np.linalg.inv(w2l)[:2, 3]
            counts, counts_new = Counter(), Counter()
            gaps, crossing_missed, dropped_triggers, duplicates = 0, 0, 0, 0
            artificial_length, endpoint_loss = 0., 0.
            signatures = set()
            exact_groups = {}
            exact_duplicates = 0
            exact_duplicate_counts = Counter()
            for idx, (points, sample, kind) in enumerate(zip(town['lane_points'], town['lane_sample_points'], town['lane_types'])):
                if kind not in CLASSES or np.min(np.linalg.norm(sample[:, :2] - ego, axis=-1)) >= 50:
                    continue
                xy = (np.asarray(points) @ w2l[:3, :3].T + w2l[:3, 3])[:, :2]
                if len(xy) < 2 or xy[:, 0].max() < -15 or xy[:, 0].min() > 15 or xy[:, 1].max() < -30 or xy[:, 1].min() > 30:
                    continue
                inside = (xy[:, 0] > -15) & (xy[:, 0] < 15) & (xy[:, 1] > -30) & (xy[:, 1] < 30)
                selected = np.flatnonzero(inside)
                old = LineString(xy[inside]) if len(selected) > 1 else None
                true_parts = components(LineString(xy).intersection(roi))
                counts_new[kind] += len(true_parts)
                if old is not None:
                    counts[kind] += 1
                    sampled = np.array([old.interpolate(d).coords[0] for d in np.linspace(0, old.length, 20)])
                    signature = min(tuple(np.round(sampled/.05).astype(int).flatten()), tuple(np.round(sampled[::-1]/.05).astype(int).flatten()))
                    sig = (kind, signature)
                    duplicates += int(sig in signatures)
                    signatures.add(sig)
                    key = (kind, tuple(np.round(old.bounds, 6)))
                    group = exact_groups.setdefault(key, [])
                    equivalent = next((g for g in group if old.equals(g)), None)
                    if equivalent is not None:
                        exact_duplicates += 1
                        exact_duplicate_counts[kind] += 1
                        if len(duplicate_examples) < 8:
                            duplicate_examples.append(dict(route=routes[ri], frame=int(fi), kind=kind,
                                first_xy=list(equivalent.coords), duplicate_xy=list(old.coords),
                                hausdorff_m=old.hausdorff_distance(equivalent), length_m=old.length))
                    else:
                        group.append(old)
                    holes = np.flatnonzero(np.diff(selected) > 1)
                    if len(holes):
                        gaps += 1
                        bridge = sum(np.linalg.norm(xy[selected[j+1]] - xy[selected[j]]) for j in holes)
                        artificial_length += float(bridge)
                        if len(examples) < 12:
                            examples.append(dict(route=routes[ri], frame=int(fi), town=ann['town_name'], lane_index=idx, kind=kind,
                                old_xy=xy[inside].tolist(), clipped_parts=[list(g.coords) for g in true_parts],
                                bridge_length_m=float(bridge), source_xy=xy.tolist()))
                    else:
                        endpoint_loss += max(0., sum(g.length for g in true_parts) - old.length)
                elif true_parts:
                    crossing_missed += 1
            for points, kind in zip(town['trigger_volumes_points'], town['trigger_volumes_types']):
                if kind not in CLASSES:
                    continue
                xy = (np.asarray(points) @ w2l[:3, :3].T + w2l[:3, 3])[:, :2]
                if not len(xy):
                    continue
                inside = (xy[:, 0] > -15) & (xy[:, 0] < 15) & (xy[:, 1] > -30) & (xy[:, 1] < 30)
                if inside.all():
                    counts[kind] += 1
                    counts_new[kind] += 1
                elif len(xy) >= 2 and components(LineString(np.concatenate([xy, xy[:1]])).intersection(roi)):
                    dropped_triggers += 1
            rows.append(dict(route=routes[ri], frame=int(fi), town=ann['town_name'], old_counts=dict(counts),
                clipped_lane_counts_keep_trigger_policy=dict(counts_new), old_total=sum(counts.values()),
                clipped_total=sum(counts_new.values()), bridged_lines=gaps, bridge_length_m=artificial_length,
                short_crossings_missing=crossing_missed, partial_triggers_dropped=dropped_triggers,
                same_class_duplicates_5cm=duplicates, endpoint_length_missing_m=endpoint_loss))
            rows[-1].update(exact_same_class_duplicates=exact_duplicates, exact_duplicate_counts=dict(exact_duplicate_counts),
                deduplicated_old_total=sum(counts.values()) - exact_duplicates)
        print('route', int(ri), 'frames', len(rows), 'bridges', sum(x['bridged_lines'] for x in rows), flush=True)
    totals = [r['old_total'] for r in rows]
    summary = dict(frames=len(rows), routes=len(route_indices), split=args.split, total_instances=sum(totals),
        counts_by_class={c: sum(r['old_counts'].get(c, 0) for r in rows) for c in CLASSES},
        count_percentiles=dict(zip(['min', 'p50', 'p90', 'max'], np.percentile(totals, [0, 50, 90, 100]).tolist())),
        frames_gt_over_50=sum(n > 50 for n in totals), frames_gt_over_100=sum(n > 100 for n in totals),
        frames_no_gt=sum(n == 0 for n in totals), frames_with_bridges=sum(r['bridged_lines'] > 0 for r in rows),
        **{k:sum(r[k] for r in rows) for k in ['bridged_lines', 'bridge_length_m', 'short_crossings_missing',
            'partial_triggers_dropped', 'same_class_duplicates_5cm', 'exact_same_class_duplicates', 'endpoint_length_missing_m']})
    summary['exact_duplicates_by_class'] = {c: sum(r['exact_duplicate_counts'].get(c, 0) for r in rows) for c in CLASSES}
    summary['deduplicated_frames_gt_over_50'] = sum(r['deduplicated_old_total'] > 50 for r in rows)
    summary['deduplicated_frames_gt_over_100'] = sum(r['deduplicated_old_total'] > 100 for r in rows)
    result = dict(summary=summary, towns=towns, rows=rows, examples=examples, duplicate_examples=duplicate_examples,
        provenance=dict(map_path=str(map_path.resolve()), map_size=map_path.stat().st_size,
                        meta_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                        sample='equally spaced routes; first/middle/last frames by default', time=time.time()),
        note='Geometry audit, not map quality evaluation. Trigger policy is measured but unchanged.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
