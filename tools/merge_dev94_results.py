"""Merge completed Bench2Drive route shards with canonical statistics."""

import argparse
import json
import os
import xml.etree.ElementTree as ET

from leaderboard.utils.statistics_manager import StatisticsManager


def route_number(route_id):
    return route_id.split('_')[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--routes', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    expected_ids = [
        route.get('id') for route in ET.parse(args.routes).getroot()]
    observed_ids = []
    sensors = None

    for path in args.inputs:
        with open(path) as handle:
            result = json.load(handle)
        checkpoint = result['_checkpoint']
        completed, total = checkpoint['progress']
        records = checkpoint['records']
        if completed != total or len(records) != total:
            raise RuntimeError(
                f'incomplete shard {path}: progress={completed}/{total}, '
                f'records={len(records)}')
        observed_ids.extend(route_number(record['route_id']) for record in records)
        if sensors is None:
            sensors = result.get('sensors', [])

    duplicates = sorted({
        route_id for route_id in observed_ids
        if observed_ids.count(route_id) > 1})
    if duplicates:
        raise RuntimeError(f'duplicate route IDs: {duplicates}')
    if set(observed_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(observed_ids))
        extra = sorted(set(observed_ids) - set(expected_ids))
        raise RuntimeError(f'route mismatch: missing={missing}, extra={extra}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    manager = StatisticsManager(args.out, args.out + '.debug')
    for path in args.inputs:
        manager.add_file_records(path)
    records = manager._results.checkpoint.records
    records_by_id = {route_number(record.route_id): record for record in records}
    manager._results.checkpoint.records = [
        records_by_id[route_id] for route_id in expected_ids]
    for index, record in enumerate(manager._results.checkpoint.records):
        record.index = index
    manager.save_sensors(sensors)
    manager.save_progress(len(observed_ids), len(expected_ids))
    manager.compute_global_statistics()
    manager.validate_and_write_statistics(sensors_initialized=True, crashed=False)
    print(f'merged {len(observed_ids)} routes -> {args.out}')


if __name__ == '__main__':
    main()
