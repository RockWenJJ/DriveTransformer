#!/usr/bin/env python3
"""Dump and visualise DriveTransformer attention for method comparison.

Produces, per sample / method:
  - temporal attention block heatmaps (own / crosstask / directed)
  - sensor cross-attn overlays on the 6 cameras (ego query, last decoder layer)
  - a small JSON summary of mass on agent/map/ego memory blocks
  - optional per-layer plan L2 residuals vs GT

Usage (from repo root, free GPU):
  CUDA_VISIBLE_DEVICES=4 python tools/analyze_attention.py \
    --methods base_10ep crosstemp dgrcta \
    --frames 100 500 1000 --warmup 8 \
    --out-dir ol_eval_results/attn_analysis
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dt_viz_common import (  # noqa: E402
    INPUT_KEYS, build, cameras_of, frame_sequence, pick_frames, to_device,
)
from mmcv.datasets import build_dataset  # noqa: E402
from mmcv.parallel import DataContainer  # noqa: E402
from mmcv.parallel.collate import collate  # noqa: E402

from adzoo.drivetransformer.mmdet3d_plugin.ours.drivetransformer_layers import (  # noqa: E402
    capture_attention,
)

METHODS = {
    "base_10ep": dict(
        config="adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base.py",
        ckpt="adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_base_10ep/iter_24400.pth",
        mode="own",
    ),
    "crosstemp": dict(
        config="adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_crosstemp.py",
        ckpt="adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_base_crosstemp/iter_25000.pth",
        mode="crosstask",
    ),
    "dgrcta": dict(
        config="adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_dgrcta.py",
        ckpt="adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_base_dgrcta/iter_25000.pth",
        mode="directed",
    ),
    "dinov3_vits16": dict(
        config="adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_dinov3_vits16.py",
        ckpt="adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_base_dinov3_vits16/iter_25000.pth",
        mode="own",
    ),
    "dinov3_vitb16": dict(
        config="adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_dinov3_vitb16.py",
        ckpt="adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_base_dinov3_vitb16/iter_25000.pth",
        mode="own",
    ),
}

# Defaults from drivetransformer_base.py
AGENT_PROP = 50
MAP_PROP = 50
MEM_LEN = 10
MAP_PTS = 20
NUM_CAM = 6


def mean_heads(weights):
    """[B,H,Q,K] -> [Q,K] averaged over batch and heads."""
    w = weights.numpy() if torch.is_tensor(weights) else np.asarray(weights)
    return w.mean(axis=(0, 1))


def temporal_block_mass(row, layout, agent_prop=AGENT_PROP, map_prop=MAP_PROP, mem_len=MEM_LEN):
    """Summarise attention mass for one query row over temporal keys.

    layout:
      - ``own_ego``: [null | L ego]
      - ``own_agent``: [null | L * Na]
      - ``own_map``: [null | L * Nm]
      - ``crosstask``: [null | L*Na | L*Nm | L]
    """
    row = np.asarray(row, dtype=np.float64)
    out = {"null": float(row[0]) if len(row) else 0.0}
    if layout == "crosstask":
        a0, a1 = 1, 1 + mem_len * agent_prop
        m0, m1 = a1, a1 + mem_len * map_prop
        e0, e1 = m1, m1 + mem_len
        out.update(
            agent=float(row[a0:a1].sum()),
            map=float(row[m0:m1].sum()),
            ego=float(row[e0:e1].sum()),
        )
        if e1 <= len(row):
            out["ego_per_frame"] = row[e0:e1].tolist()
    elif layout == "own_ego":
        e0, e1 = 1, 1 + mem_len
        out["ego"] = float(row[e0:min(e1, len(row))].sum())
        out["ego_per_frame"] = row[e0:min(e1, len(row))].tolist()
    elif layout == "own_agent":
        a0, a1 = 1, 1 + mem_len * agent_prop
        out["agent"] = float(row[a0:min(a1, len(row))].sum())
    elif layout == "own_map":
        m0, m1 = 1, 1 + mem_len * map_prop
        out["map"] = float(row[m0:min(m1, len(row))].sum())
    return out


def layout_for_label(label):
    if "crosstask" in label:
        return "crosstask"
    if label.endswith("/agent_to_ego") or label.endswith("/own/agent"):
        return "own_agent"
    if label.endswith("/map_to_ego") or label.endswith("/map_to_agent") or label.endswith("/own/map"):
        return "own_map"
    if label.endswith("/own/ego"):
        return "own_ego"
    return "own_ego"


def plot_temporal_heatmap(path, title, matrix, xticks, yticks, vlines=None):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_title(title, fontsize=10)
    if yticks is not None:
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks, fontsize=8)
    if xticks is not None:
        step = max(1, len(xticks) // 16)
        idx = list(range(0, len(xticks), step))
        ax.set_xticks(idx)
        ax.set_xticklabels([xticks[i] for i in idx], fontsize=7, rotation=45, ha="right")
    if vlines:
        for x in vlines:
            ax.axvline(x - 0.5, color="cyan", lw=0.8, alpha=0.7)
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def overlay_sensor_on_cams(cams, heat_per_cam, out_path, title):
    """cams: list of HxWx3 uint8; heat_per_cam: list of hf x wf arrays in [0,1]."""
    n = len(cams)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5))
    axes = axes.ravel()
    for i in range(n):
        ax = axes[i]
        img = cams[i]
        hmap = heat_per_cam[i]
        h, w = img.shape[:2]
        # nearest upsample to camera resolution
        ys = (np.linspace(0, hmap.shape[0] - 1, h)).astype(int)
        xs = (np.linspace(0, hmap.shape[1] - 1, w)).astype(int)
        up = hmap[ys][:, xs]
        ax.imshow(img)
        ax.imshow(up, cmap="jet", alpha=0.45, vmin=0.0, vmax=max(float(up.max()), 1e-6))
        ax.set_title(f"cam{i}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def sensor_to_cams(weights_qk, feat_h, feat_w, num_cam=NUM_CAM):
    """weights_qk: [Q, K] with K = num_cam * feat_h * feat_w. Return list of [H,W]."""
    k = num_cam * feat_h * feat_w
    row = weights_qk.mean(axis=0) if weights_qk.ndim == 2 else weights_qk
    row = row[:k]
    maps = row.reshape(num_cam, feat_h, feat_w)
    # per-cam normalize for visibility
    out = []
    for m in maps:
        m = m - m.min()
        out.append(m / (m.max() + 1e-8))
    return out


def run_with_attn(model, head, seq, token):
    """Warm up memory on seq[:-1], capture attention on the last frame."""
    outs = None
    records = []
    for i, sample in enumerate(seq):
        meta = dict(sample["img_metas"].data)
        meta["scene_token"] = token
        inp = {k: sample[k] for k in INPUT_KEYS if k in sample}
        inp["img_metas"] = DataContainer(meta, cpu_only=True)
        batch = to_device(collate([inp], samples_per_gpu=1), "cuda")
        capture = i == len(seq) - 1
        captured = {}
        orig = head.forward

        def wrapped(*a, _og=orig, **k):
            out = _og(*a, **k)
            captured["outs"] = out
            return out

        head.forward = wrapped
        try:
            if capture:
                with capture_attention(True) as recs, torch.no_grad():
                    model(batch, return_loss=False, rescale=True)
                records = list(recs)
            else:
                with torch.no_grad():
                    model(batch, return_loss=False, rescale=True)
        finally:
            head.forward = orig
        outs = captured["outs"]
    return outs, records


def feat_hw_from_keys(num_keys, num_cam=NUM_CAM):
    tokens = num_keys // num_cam
    # prefer 12x33 (base), then 9x24 (lowres)
    for h, w in ((12, 33), (9, 24), (24, 66), (18, 48)):
        if h * w == tokens:
            return h, w
    # fallback: factorize
    h = int(np.sqrt(tokens))
    while tokens % h:
        h -= 1
    return h, tokens // h


def analyze_records(records, mode, agent_n, map_n_pts, ego_n=1):
    """Pick last-layer temporal (ego) + last sensor attn; build summaries."""
    temporal = [r for r in records if r["label"].startswith("temporal/")]
    sensor = [r for r in records if r["label"].startswith("sensor/")]
    # Re-index sensor calls in capture order: pre-decoder (if any) then dec0..dec5.
    for i, r in enumerate(sensor):
        r["label"] = f"sensor/call{i}" + r["label"][r["label"].find("/"):]
    summary = {"temporal_calls": [r["label"] for r in temporal],
               "sensor_calls": [r["label"] for r in sensor]}

    # Last decoder layer: last occurrence of each temporal label group
    by_label = defaultdict(list)
    for r in temporal:
        by_label[r["label"]].append(r)
    last_temporal = {k: v[-1] for k, v in by_label.items()}

    ego_mass = {}
    heatmaps = {}
    for label, rec in last_temporal.items():
        mat = mean_heads(rec["weights"])  # [Q, K]
        heatmaps[label] = mat
        # Prefer ego-related calls for the summary mass table.
        if not (label.endswith("/ego") or "to_ego" in label):
            continue
        row = mat.mean(axis=0) if mat.shape[0] > 1 else mat[0]
        ego_mass[label] = temporal_block_mass(row, layout_for_label(label))

    last_sensor = sensor[-1] if sensor else None
    sensor_pack = None
    if last_sensor is not None:
        mat = mean_heads(last_sensor["weights"])  # [Q, K]
        # query layout: [agent | map_pts | ego]
        a0, a1 = 0, agent_n
        m0, m1 = a1, a1 + map_n_pts
        e0, e1 = m1, m1 + ego_n
        ego_attn = mat[e0:e1].mean(axis=0)  # [K]
        agent_attn = mat[a0:a1].mean(axis=0)
        map_attn = mat[m0:m1].mean(axis=0)
        feat_h, feat_w = feat_hw_from_keys(last_sensor["num_keys"])
        sensor_pack = dict(
            label=last_sensor["label"],
            feat_h=feat_h,
            feat_w=feat_w,
            ego=ego_attn,
            agent=agent_attn,
            map=map_attn,
            ego_cam_mass=ego_attn.reshape(NUM_CAM, feat_h, feat_w).sum(axis=(1, 2)).tolist(),
        )

    # plan layers if present in outs handled outside
    summary["ego_temporal_mass"] = ego_mass
    return summary, heatmaps, sensor_pack, last_temporal


def plan_layer_l2(outs, sample):
    gt = np.asarray(sample["ego_fut_trajs_fix_time"].data, dtype=np.float64).reshape(-1, 2)
    mask = np.asarray(sample["ego_fut_masks_fix_time"].data).reshape(-1) > 0
    preds = outs["ego_fut_preds_fix_time"].float().cpu().numpy()  # [L, B, mode, T, 2]
    layers = []
    for li in range(preds.shape[0]):
        pred = preds[li, 0, 0]
        errs = []
        for sec, idx in ((1.0, 9), (2.0, 19), (3.0, 29)):
            if idx < len(mask) and mask[idx]:
                errs.append(float(np.linalg.norm(pred[idx] - gt[idx])))
            else:
                errs.append(float("nan"))
        layers.append({"layer": li, "l2_1s": errs[0], "l2_2s": errs[1], "l2_3s": errs[2],
                       "l2_avg": float(np.nanmean(errs))})
    return layers


def plot_plan_layers(path, methods_curves, frame_id):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, layers in methods_curves.items():
        xs = [d["layer"] for d in layers]
        ys = [d["l2_avg"] for d in layers]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("plan L2 avg @1/2/3s (m)")
    ax.set_title(f"Per-layer plan L2 — frame {frame_id}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["base_10ep", "crosstemp", "dgrcta"])
    p.add_argument("--frames", nargs="+", type=int, default=None)
    p.add_argument("--num", type=int, default=4)
    p.add_argument("--stride", type=int, default=200)
    p.add_argument("--scan", type=int, default=4000)
    p.add_argument("--min-gt", type=int, default=3)
    p.add_argument("--max-gt", type=int, default=40)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--out-dir", default="ol_eval_results/attn_analysis")
    p.add_argument("--agent-n", type=int, default=900)
    p.add_argument("--map-n", type=int, default=100)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Build dataset once from the first method's config (same Bench2Drive val).
    from mmcv import Config
    first = METHODS[args.methods[0]]
    cfg = Config.fromfile(first["config"])
    if getattr(cfg, "plugin", False):
        import importlib
        importlib.import_module(os.path.dirname(cfg.plugin_dir).replace("/", "."))
    ds = build_dataset(cfg.data.val)
    frames = args.frames if args.frames else pick_frames(ds, args)
    print(f"frames: {frames}")

    global_summary = {"frames": frames, "methods": {}}
    plan_curves_by_frame = defaultdict(dict)

    for method in args.methods:
        spec = METHODS[method]
        if not os.path.isfile(spec["ckpt"]):
            print(f"[skip] missing ckpt for {method}: {spec['ckpt']}")
            continue
        print(f"\n=== {method} ===")
        _, model = build(spec["config"], spec["ckpt"])
        head = model.pts_bbox_head
        method_dir = os.path.join(args.out_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        method_sum = {}

        for fi in frames:
            seq, token = frame_sequence(ds, int(fi), args.warmup)
            sample = seq[-1]
            outs, records = run_with_attn(model, head, seq, f"{token}#attn")
            print(f"  frame {fi}: {len(records)} attn records")

            summary, heatmaps, sensor_pack, last_temporal = analyze_records(
                records, spec["mode"], args.agent_n, args.map_n * MAP_PTS, ego_n=1)
            layers = plan_layer_l2(outs, sample)
            plan_curves_by_frame[fi][method] = layers
            summary["plan_layers"] = layers

            # temporal plots
            for label, mat in heatmaps.items():
                safe = label.replace("/", "__")
                # average over queries -> 1 x K row image for readability when Q large
                if mat.shape[0] > 8:
                    plot_mat = mat.mean(axis=0, keepdims=True)
                    yticks = ["mean_Q"]
                else:
                    plot_mat = mat
                    yticks = [f"q{i}" for i in range(mat.shape[0])]
                xticks = [str(i) for i in range(plot_mat.shape[1])]
                vlines = None
                if "crosstask" in label:
                    a1 = 1 + MEM_LEN * AGENT_PROP
                    m1 = a1 + MEM_LEN * MAP_PROP
                    vlines = [1, a1, m1]
                    xticks = (
                        ["null"]
                        + [f"A{f}" for f in range(MEM_LEN) for _ in range(AGENT_PROP)]
                        + [f"M{f}" for f in range(MEM_LEN) for _ in range(MAP_PROP)]
                        + [f"E{f}" for f in range(MEM_LEN)]
                    )
                    # downsample labels already handled by step in plot
                plot_temporal_heatmap(
                    os.path.join(method_dir, f"frame{fi:05d}_{safe}.png"),
                    f"{method} | {label} | frame {fi}",
                    plot_mat, xticks, yticks, vlines=vlines,
                )

            # sensor overlays
            cams, names, _ = cameras_of(sample, cfg)
            if sensor_pack is not None and cams:
                for role in ("ego", "agent", "map"):
                    heat = sensor_to_cams(
                        sensor_pack[role][None, :],
                        sensor_pack["feat_h"], sensor_pack["feat_w"])
                    overlay_sensor_on_cams(
                        cams, heat,
                        os.path.join(method_dir, f"frame{fi:05d}_sensor_{role}.png"),
                        f"{method} | sensor attn ({role}) | {sensor_pack['label']} | frame {fi}",
                    )
                summary["sensor_ego_cam_mass"] = sensor_pack["ego_cam_mass"]
                summary["sensor_feat_hw"] = [sensor_pack["feat_h"], sensor_pack["feat_w"]]

            # save raw weights for the last temporal ego call + last sensor
            npz = {}
            for label, rec in last_temporal.items():
                if "ego" in label or "to_ego" in label:
                    npz[label.replace("/", "__")] = mean_heads(rec["weights"])
            if sensor_pack is not None:
                npz["sensor_ego"] = sensor_pack["ego"]
            np.savez_compressed(os.path.join(method_dir, f"frame{fi:05d}_weights.npz"), **npz)
            method_sum[str(fi)] = summary
            print(f"    ego temporal mass: {json.dumps(summary.get('ego_temporal_mass', {}), default=str)[:200]}")

        global_summary["methods"][method] = method_sum
        del model
        torch.cuda.empty_cache()

    for fi, curves in plan_curves_by_frame.items():
        plot_plan_layers(
            os.path.join(args.out_dir, f"frame{fi:05d}_plan_layers.png"),
            curves, fi)

    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(global_summary, f, indent=2)
    print(f"\nsaved summary -> {out_json}")


if __name__ == "__main__":
    main()
