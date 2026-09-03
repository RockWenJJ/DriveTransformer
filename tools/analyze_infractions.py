#!/usr/bin/env python3
"""Decompose Bench2Drive closed-loop infractions across methods (dev94).

Reads results.json records and reports:
  - mean DS / RC / success
  - per-infraction route hit rates (any event of that type)
  - mean event counts per route
  - ability bucket success if scenario_name is present

Usage:
  python tools/analyze_infractions.py \
    --methods base_10ep crosstemp dgrcta lowres camdrop \
    --out ol_eval_results/infraction_comparison.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

DEFAULT_METHODS = {
    "base_100k": "/data5T/wenjj/Projects/Bench2Drive/eval_results/dt_base_dev94/results.json",
    "base_10ep": "/data5T/wenjj/Projects/Bench2Drive/eval_results/dt_base_10ep_dev94_canonical/results.json",
    "crosstemp": "/data5T/wenjj/Projects/Bench2Drive/eval_results/dt_crosstemp_dev94_canonical/results.json",
    "dgrcta": "/data5T/wenjj/Projects/Bench2Drive/eval_results/dgrcta_dev93_partial/results.json",
    "lowres": "/data5T/wenjj/Projects/Bench2Drive/eval_results/dt_lowres_dev94_merged/results.json",
    "camdrop": "/data5T/wenjj/Projects/Bench2Drive/eval_results/dt_camdrop_merged/results.json",
}

# Prefer these if the "canonical" path is missing.
FALLBACKS = {
    "base_100k": [
        "/data5T/wenjj/Projects/Bench2Drive/eval_results/dt_base_merged/results.json",
    ],
    "base_10ep": [
        "/data5T/wenjj/Projects/Bench2Drive/eval_results/dt_base_10ep_dev94/results.json",
    ],
    "crosstemp": [
        "/data5T/wenjj/Projects/Bench2Drive/eval_results/dt_crosstemp_dev94/results.json",
    ],
}

INFRACTION_KEYS = [
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "scenario_timeouts",
    "route_dev",
    "vehicle_blocked",
    "route_timeout",
    "yield_emergency_vehicle_infractions",
]

ABILITY_MAP = {
    "OverTaking": "OT",
    "Merging": "MG",
    "EmergencyBrake": "EB",
    "GiveWay": "GW",
    "TrafficSign": "TS",
}


def resolve_path(method, override=None):
    if override:
        return override
    path = DEFAULT_METHODS.get(method)
    if path and os.path.isfile(path):
        return path
    for alt in FALLBACKS.get(method, []):
        if os.path.isfile(alt):
            return alt
    return path


def load_records(path, dev94_ids=None):
    data = json.load(open(path))
    recs = data.get("records") or data.get("_checkpoint", {}).get("records")
    if recs is None:
        raise KeyError(f"no records in {path}; keys={list(data.keys())[:20]}")
    if dev94_ids is not None:
        ids = set(dev94_ids)
        filtered = [r for r in recs if r.get("route_id") in ids]
        if filtered:
            recs = filtered
    return recs


def summarize(recs):
    n = len(recs)
    ds = [r["scores"]["score_composed"] for r in recs]
    rc = [r["scores"]["score_route"] for r in recs]
    success = sum(
        1 for r in recs
        if r.get("status") in ("Completed", "Perfect")
        and abs(r["scores"]["score_composed"] - 100.0) < 1e-6
    )
    completed = sum(1 for r in recs if r.get("status") in ("Completed", "Perfect"))

    hit = {k: 0 for k in INFRACTION_KEYS}
    counts = {k: 0 for k in INFRACTION_KEYS}
    for r in recs:
        infra = r.get("infractions") or {}
        for k in INFRACTION_KEYS:
            ev = infra.get(k) or []
            if ev:
                hit[k] += 1
                counts[k] += len(ev)

    abilities = defaultdict(lambda: {"n": 0, "ok": 0})
    for r in recs:
        name = r.get("scenario_name") or ""
        bucket = None
        for key, code in ABILITY_MAP.items():
            if key.lower() in name.lower() or key in name:
                bucket = code
                break
        if bucket is None:
            continue
        abilities[bucket]["n"] += 1
        # ability success = completed with no infractions of the penalty types
        infra = r.get("infractions") or {}
        clean = all(not (infra.get(k) or []) for k in INFRACTION_KEYS
                    if k not in ("min_speed_infractions",))
        if r.get("status") in ("Completed", "Perfect") and clean:
            abilities[bucket]["ok"] += 1

    return {
        "n": n,
        "ds_mean": sum(ds) / n if n else None,
        "rc_mean": sum(rc) / n if n else None,
        "completion_pct": 100.0 * completed / n if n else None,
        "success_pct": 100.0 * success / n if n else None,
        "infraction_route_pct": {k: 100.0 * hit[k] / n if n else None for k in INFRACTION_KEYS},
        "infraction_mean_count": {k: counts[k] / n if n else None for k in INFRACTION_KEYS},
        "abilities": {
            k: {"n": v["n"], "success_pct": 100.0 * v["ok"] / v["n"] if v["n"] else None}
            for k, v in abilities.items()
        },
    }


def print_table(results):
    methods = list(results.keys())
    print(f"\n{'metric':28s}" + "".join(f"{m:>12s}" for m in methods))
    for key in ("n", "ds_mean", "rc_mean", "completion_pct", "success_pct"):
        row = f"{key:28s}"
        for m in methods:
            v = results[m]["summary"][key]
            row += f"{v:12.2f}" if isinstance(v, float) else f"{v:12d}"
        print(row)
    print("\nInfraction route hit rate (%)")
    for k in INFRACTION_KEYS:
        row = f"{k:28s}"
        for m in methods:
            v = results[m]["summary"]["infraction_route_pct"][k]
            row += f"{v:12.1f}" if v is not None else f"{'—':>12s}"
        print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+",
                   default=["base_10ep", "crosstemp", "dgrcta", "lowres", "camdrop"])
    p.add_argument("--dev94-ids", default="/data5T/wenjj/tmp_scripts/dev94_ids.json")
    p.add_argument("--path", action="append", default=[],
                   help="override as method=path, repeatable")
    p.add_argument("--out", default="ol_eval_results/infraction_comparison.json")
    args = p.parse_args()

    overrides = {}
    for item in args.path:
        m, path = item.split("=", 1)
        overrides[m] = path

    dev94 = None
    if args.dev94_ids and os.path.isfile(args.dev94_ids):
        raw = json.load(open(args.dev94_ids))
        dev94 = raw if isinstance(raw, list) else raw.get("route_ids") or raw.get("ids")

    results = {}
    for method in args.methods:
        path = resolve_path(method, overrides.get(method))
        if not path or not os.path.isfile(path):
            print(f"[skip] {method}: missing {path}")
            continue
        recs = load_records(path, dev94_ids=dev94)
        # camdrop / base merged may still be 220; filter helps when ids available
        summary = summarize(recs)
        results[method] = {"path": path, "summary": summary}
        print(f"{method}: n={summary['n']} DS={summary['ds_mean']:.2f} "
              f"succ={summary['success_pct']:.1f}%  <- {path}")

    print_table(results)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
