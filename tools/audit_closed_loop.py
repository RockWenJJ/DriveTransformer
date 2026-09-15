"""Recompute route statistics from records; do not trust cached global_record."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def summarize_records(records, expected):
    by_id = {}
    for record in records:
        route = record['route_id']
        if route in by_id:
            raise ValueError('Duplicate route record; resolve retry provenance: ' + route)
        by_id[route] = record
    selected = [by_id[key] for key in sorted(expected & set(by_id))]
    def success(r):
        return r['status'] in ('Completed', 'Perfect') and not any(
            values for key, values in r['infractions'].items() if key != 'min_speed_infractions')
    n = len(selected)
    return dict(expected=len(expected), available=n, complete=n == len(expected),
                missing=sorted(expected - set(by_id)), outside_manifest=sorted(set(by_id) - expected),
                provisional_DS=sum(r['scores']['score_composed'] for r in selected) / n if n else None,
                provisional_RC=sum(r['scores']['score_route'] for r in selected) / n if n else None,
                provisional_completion_pct=100 * sum(r['status'] in ('Completed', 'Perfect') for r in selected) / n if n else None,
                provisional_success_pct=100 * sum(success(r) for r in selected) / n if n else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--routes', required=True)
    p.add_argument('--result', action='append', required=True, help='label=/path/results.json')
    p.add_argument('--out', required=True)
    p.add_argument('--require-complete', action='store_true')
    args = p.parse_args()
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(path)
    ids = ['RouteScenario_' + route.get('id') + '_rep0' for route in ET.parse(args.routes).getroot()]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate route IDs in manifest')
    reports = {}
    for item in args.result:
        label, file = item.split('=', 1)
        if label in reports:
            raise ValueError('Duplicate result label: ' + label)
        raw = Path(file).read_bytes()
        report = summarize_records(json.loads(raw)['_checkpoint']['records'], set(ids))
        report.update(source=str(Path(file).resolve()), source_sha256=hashlib.sha256(raw).hexdigest())
        reports[label] = report
    output = dict(route_manifest=str(Path(args.routes).resolve()),
                  route_sha256=hashlib.sha256(Path(args.routes).read_bytes()).hexdigest(),
                  results=reports)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    if args.require_complete and not all(r['complete'] for r in reports.values()):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
