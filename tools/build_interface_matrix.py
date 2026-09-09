"""Write configs for old-checkpoint CARLA diagnostics; never launch jobs."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.dev24_routes import resolve_routes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True, help='Original R50 or DINO-B frozen config')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--route-manifest', help='Defaults to this checkout\'s frozen dev24_v1 routes')
    args = p.parse_args()
    evaluation = resolve_routes(args.route_manifest)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = [('C0_historical', True, 1., 'legacy_radians', 'xy'),
             ('C1_eval_deterministic', False, 1., 'legacy_radians', 'xy'),
             ('C2_time_compat', False, 5., 'legacy_radians', 'xy'),
             ('C3_heading', False, 1., 'sin_heading', 'xy'),
             ('C4_time_heading', False, 5., 'sin_heading', 'xy'),
             ('C5_accel_compat', False, 1., 'legacy_radians', 'legacy_z_broadcast')]
    manifest = []
    for name, dropout, scale, decoder, accel in cases:
        contract = dict(legacy_eval_dropout=dropout, timestamp_scale=scale,
                        fixed_dist_decoder=decoder, ego_accel_mode=accel)
        path = out / (name + '.py')
        if path.exists():
            raise FileExistsError(path)
        data = {split: dict(timestamp_frame_period_s=0.5, ego_accel_mode='legacy_z_broadcast')
                for split in ('train', 'val', 'test')}
        path.write_text(f'_base_ = [{str(Path(args.base).resolve())!r}]\n'
                        f'consistency_version = {name!r}\n'
                        'model = dict(pts_bbox_head=dict(motion_cls_valid_only=False))\n'
                        f'data = {data!r}\ninference_contract = {contract!r}\n'
                        f'closed_loop_eval = {evaluation!r}\n')
        manifest.append(dict(id=name, config=str(path.resolve()), inference_contract=contract,
                             closed_loop_eval=evaluation))
    (out / 'matrix.json').write_text(json.dumps({
        'purpose': 'Existing-checkpoint closed-loop diagnostics only; dataset replay remains legacy.',
        'note': 'C1-C4 form the 2x2 time/heading matrix. C5 is an isolated old-training acceleration compatibility probe.',
        'closed_loop_eval': evaluation,
        'cases': manifest}, indent=2))


if __name__ == '__main__':
    main()
