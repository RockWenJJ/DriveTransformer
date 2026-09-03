#!/usr/bin/env python3
"""Side-by-side BEV planning overlays for the same val frames across methods.

Usage:
  CUDA_VISIBLE_DEVICES=4 python tools/analyze_plan_compare.py \
    --methods base_10ep crosstemp dgrcta \
    --frames 100 500 1000 --out-dir ol_eval_results/plan_compare
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dt_viz_common import (  # noqa: E402
    EGO_DT, MAP_STYLE, MAP_SKIP, build, bev_poly, ego_history, ego_size,
    frame_sequence, gt_of, map_of, pick_frames, run_sequence,
)
from mmcv.datasets import build_dataset  # noqa: E402
from mmcv import Config  # noqa: E402

METHODS = {
    "base_10ep": (
        "adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base.py",
        "adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_base_10ep/iter_24400.pth",
    ),
    "crosstemp": (
        "adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_crosstemp.py",
        "adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_base_crosstemp/iter_25000.pth",
    ),
    "dgrcta": (
        "adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_dgrcta.py",
        "adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_base_dgrcta/iter_25000.pth",
    ),
    "dinov3_vits16": (
        "adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_dinov3_vits16.py",
        "adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_base_dinov3_vits16/iter_25000.pth",
    ),
}

COLORS = {
    "gt": "#111111",
    "base_10ep": "#1f77b4",
    "crosstemp": "#d62728",
    "dgrcta": "#2ca02c",
    "dinov3_vits16": "#9467bd",
}


def decode_plan(outs):
    return outs["ego_fut_preds_fix_time"][-1][0, 0].float().cpu().numpy()


def gt_plan(sample):
    traj = np.asarray(sample["ego_fut_trajs_fix_time"].data, dtype=np.float64).reshape(-1, 2)
    mask = np.asarray(sample["ego_fut_masks_fix_time"].data).reshape(-1) > 0
    return traj, mask


def l2_avg(pred, gt, mask):
    errs = []
    for sec in (1.0, 2.0, 3.0):
        k = int(round(sec / EGO_DT)) - 1
        if k < len(mask) and mask[k]:
            errs.append(np.linalg.norm(pred[k] - gt[k]))
    return float(np.mean(errs)) if errs else float("nan")


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
    p.add_argument("--out-dir", default="ol_eval_results/plan_compare")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg0 = Config.fromfile(METHODS[args.methods[0]][0])
    if getattr(cfg0, "plugin", False):
        import importlib
        importlib.import_module(os.path.dirname(cfg0.plugin_dir).replace("/", "."))
    ds = build_dataset(cfg0.data.val)
    frames = args.frames if args.frames else pick_frames(ds, args)
    map_classes = list(getattr(cfg0, "map_classes", MAP_STYLE.keys()))

    models = {}
    for m in args.methods:
        cfg_path, ckpt = METHODS[m]
        if not os.path.isfile(ckpt):
            print(f"[skip] {m}: missing {ckpt}")
            continue
        print(f"loading {m}")
        _, model = build(cfg_path, ckpt)
        models[m] = model

    for fi in frames:
        seq, token = frame_sequence(ds, int(fi), args.warmup)
        sample = seq[-1]
        gt, mask = gt_plan(sample)
        plans = {}
        for m, model in models.items():
            outs = run_sequence(model, model.pts_bbox_head, seq, f"{token}#cmp")
            plans[m] = decode_plan(outs)

        fig, ax = plt.subplots(figsize=(7.5, 9))
        ax.set_aspect("equal")
        ax.set_xlim(-20, 20)
        ax.set_ylim(-8, 42)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x right (m)")
        ax.set_ylabel("y forward (m)")

        map_pts, map_lab = map_of(sample)
        if map_pts is not None and map_lab is not None:
            for pts, lb in zip(map_pts, map_lab):
                name = map_classes[int(lb)] if int(lb) < len(map_classes) else str(lb)
                if name in MAP_SKIP or name not in MAP_STYLE:
                    continue
                color, ls, lw = MAP_STYLE[name]
                ax.plot(pts[:, 0], pts[:, 1], color=color, ls=ls, lw=lw, alpha=0.7)

        _, rows, labels = gt_of(sample)
        for row in rows.numpy():
            gx, gy, _, sx, sy, _, yaw = row[:7]
            ax.add_patch(Polygon(
                bev_poly(gx, gy, sx, sy, yaw), closed=True,
                facecolor="#4c78a8", alpha=0.35, edgecolor="black", lw=0.6))

        wl = ego_size(sample)
        ax.add_patch(Polygon(
            bev_poly(0.0, 0.0, wl[0], wl[1], 0.0), closed=True,
            facecolor="red", edgecolor="darkred", lw=1.2, alpha=0.9))
        hist = ego_history(sample)
        ax.plot(hist[:, 0], hist[:, 1], "k--", lw=1.0, alpha=0.5)

        ax.plot(gt[mask, 0], gt[mask, 1], color=COLORS["gt"], lw=2.8, label="GT")
        legend = [Line2D([0], [0], color=COLORS["gt"], lw=2.8, label="GT")]
        for m, traj in plans.items():
            color = COLORS.get(m, "#333333")
            score = l2_avg(traj, gt, mask)
            ax.plot(traj[:, 0], traj[:, 1], color=color, lw=2.0,
                    label=f"{m} L2={score:.2f}")
            legend.append(Line2D([0], [0], color=color, lw=2.0,
                                 label=f"{m} L2={score:.2f}"))
        ax.legend(handles=legend, fontsize=8, loc="upper right")
        ax.set_title(f"frame {fi} | {token}")
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"frame_{fi:05d}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
