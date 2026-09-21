"""Exercise actual dataset paths against independent full-line intersections."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys

import joblib
import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import unary_union


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--audits', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--regular-samples-per-split', type=int, default=20)
    a = p.parse_args()
    sys.path.insert(0, str(a.repo))
    from adzoo.drivetransformer.mmdet3d_plugin.datasets.bench2drive_drivetransformer_dataset import B2D_DriveTransformer_Dataset
    from adzoo.drivetransformer.mmdet3d_plugin.datasets.map_utils.geometry import deduplicate_map_lines
    ds = B2D_DriveTransformer_Dataset.__new__(B2D_DriveTransformer_Dataset)
    ds.map_infos = joblib.load(a.repo / 'data/infos/b2d_map_infos.pkl')
    ds.point_cloud_range = np.array([-15., -30., -2., 15., 30., 2.])
    ds.polyline_points_num = 20
    ds.map_element_class = dict(Broken=0, Solid=1, SolidSolid=2, Center=3, TrafficLight=4, StopSign=5)
    roi = box(-15, -30, 15, 30)
    results = []
    for path in a.audits:
        audit = json.loads(path.read_text())
        split = audit['summary']['split']
        prev_route, clip = None, None
        all_rows = audit['rows']
        selected = set(np.linspace(0, len(all_rows)-1, min(a.regular_samples_per_split, len(all_rows)), dtype=int).tolist())
        selected.update(i for i, r in enumerate(all_rows) if r['bridged_lines'] or r['short_crossings_missing'])
        selected.add(max(range(len(all_rows)), key=lambda i: all_rows[i]['old_total']))
        for ri in sorted(selected):
            row = all_rows[ri]
            if row['route'] != prev_route:
                clip = pickle.loads((a.repo / 'data/infos' / ('b2d_infos_v1_%s_drivetransformer' % split) / (row['route']+'.pkl')).read_bytes())
                prev_route = row['route']
            ann = clip[row['frame']]
            ds.get_data_by_index = lambda i: ann
            ds.map_clip_mode, ds.map_deduplicate = 'legacy', False
            labels, old = ds.get_map_info(0)
            assert len(labels) == row['old_total']
            ds.map_deduplicate = True
            de_labels, deduped = ds.get_map_info(0)
            ds.map_clip_mode = 'geometric'
            fixed_labels, fixed = ds.get_map_info(0)
            # Expected geometry from the original complete lane, including
            # boundary intersections; trigger policy intentionally unchanged.
            expected = {c: [] for c in range(6)}
            town = ds.map_infos[ann['town_name']]
            mat = np.asarray(ann['sensors']['LIDAR_TOP']['world2lidar'])
            ego = np.linalg.inv(mat)[:2, 3]
            for pts, sample, kind in zip(town['lane_points'], town['lane_sample_points'], town['lane_types']):
                if kind not in ds.map_element_class or np.min(np.linalg.norm(sample[:, :2]-ego, axis=-1)) >= 50:
                    continue
                if len(pts) < 2:
                    continue
                xy = (mat @ np.concatenate([pts, np.ones((len(pts), 1))], axis=-1).T).T[:, :2]
                expected[ds.map_element_class[kind]].append(LineString(xy).intersection(roi))
            for g, label in zip(old.instance_list, labels.tolist()):
                if label >= 4:
                    expected[label].append(g)
            max_error = 0.
            for c in range(6):
                truth = unary_union(expected[c])
                predicted = unary_union([g for g, label in zip(fixed.instance_list, fixed_labels.tolist()) if label == c])
                if truth.is_empty and predicted.is_empty:
                    continue
                error = truth.hausdorff_distance(predicted)
                assert np.isfinite(error) and error <= 1e-7, (row['route'], row['frame'], c, error)
                assert abs(truth.length - predicted.length) < 1e-6
                max_error = max(max_error, error)
            unique, _ = deduplicate_map_lines(fixed.instance_list, fixed_labels.tolist())
            assert len(unique) == len(fixed.instance_list)
            sampled = fixed.fixed_num_sampled_points.numpy()
            assert np.isfinite(sampled).all() and sampled.shape[1:] == (20, 2)
            result = dict(split=split, route=row['route'], frame=row['frame'], old=len(labels),
                dedup_only=len(de_labels), fixed=len(fixed_labels), max_geometry_error_m=max_error,
                before_class_counts=dict(Counter(labels.tolist())), after_class_counts=dict(Counter(fixed_labels.tolist())))
            results.append(result)
            if len(results) % 25 == 0:
                print('validated', len(results), flush=True)
    summary = {}
    for split in ['train', 'val']:
        rows = [r for r in results if r['split'] == split]
        summary[split] = dict(frames=len(rows), original_instances=sum(r['old'] for r in rows),
            removed_exact_duplicates=sum(r['old']-r['dedup_only'] for r in rows),
            fixed_instances=sum(r['fixed'] for r in rows),
            fixed_frames_over_50=sum(r['fixed'] > 50 for r in rows),
            fixed_frames_over_100=sum(r['fixed'] > 100 for r in rows),
            max_geometry_error_m=max(r['max_geometry_error_m'] for r in rows))
    a.out.write_text(json.dumps(dict(summary=summary, rows=results, status='passed',
        sampling='20 regularly spaced frames per split plus all detected false bridges, missed sparse crossings, and densest frame'), indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
