"""Exercise the real dataset path and sample physical time without GPU jobs."""
import argparse
import importlib
import json
import os
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mmcv import Config
from mmcv.datasets import build_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    cfg = Config.fromfile(args.config)
    importlib.import_module(os.path.dirname(cfg.plugin_dir).replace('/', '.'))
    ds = build_dataset(cfg.data.val)
    index = 30
    corrected = ds.get_data_info(index)
    raw = ds.get_data_by_index(index)
    nxt = ds.get_data_info(index + 1)
    assert raw['folder'] == ds.get_data_by_index(index + 1)['folder']
    np.testing.assert_allclose(corrected['ego_lcf_feat'][2:4], raw['ego_accel'][:2])
    np.testing.assert_allclose(nxt['timestamp'] - corrected['timestamp'], .1)
    # Exercise the complete image/annotation/trajectory pipeline as well.
    item = ds[index]
    assert 'img' in item and 'ego_fut_trajs_fix_dist' in item
    ds.timestamp_frame_period_s, ds.ego_accel_mode = .5, 'legacy_z_broadcast'
    legacy = ds.get_data_info(index)
    np.testing.assert_allclose(legacy['ego_lcf_feat'][2:4], np.repeat(raw['ego_accel'][2], 2))
    np.testing.assert_allclose(legacy['timestamp'], corrected['timestamp'] * 5)
    result = dict(dataset_class=type(ds).__name__, dataset_module=type(ds).__module__,
                  frame_index=int(raw['frame_idx']), corrected_timestamp=corrected['timestamp'],
                  legacy_timestamp=legacy['timestamp'],
                  corrected_accel=corrected['ego_lcf_feat'][2:4].tolist(),
                  legacy_accel=legacy['ego_lcf_feat'][2:4].tolist(),
                  classes=list(ds.CLASSES), pipeline_passed=True, physical_time_samples=[])
    for split in ('train', 'val'):
        meta_path = Path(cfg.data[split].ann_file)
        with meta_path.open('rb') as f:
            meta = pickle.load(f)
        folder = meta_path.parent / meta['infos_dir_name']
        files = sorted(folder.glob('*.pkl'))
        # Spread deterministic samples across the sorted inventory, rather than
        # taking a prefix that may contain only one scenario.
        selected = np.linspace(0, len(files) - 1, min(10, len(files)), dtype=int)
        for file in [files[j] for j in selected]:
            with file.open('rb') as f:
                rows = pickle.load(f)
            inferred = []
            for prev, cur in zip(rows, rows[1:]):
                speed = .5 * (np.linalg.norm(prev['ego_vel'][:2]) + np.linalg.norm(cur['ego_vel'][:2]))
                if speed < 2. or cur['frame_idx'] - prev['frame_idx'] != 1:
                    continue
                delta = np.linalg.inv(cur['sensors']['LIDAR_TOP']['world2lidar'])[:2, 3] - np.linalg.inv(prev['sensors']['LIDAR_TOP']['world2lidar'])[:2, 3]
                inferred.append(float(np.linalg.norm(delta) / speed))
            result['physical_time_samples'].append(dict(split=split, clip=file.stem,
                eligible_pairs=len(inferred), median_delta_s=float(np.median(inferred)) if inferred else None))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
