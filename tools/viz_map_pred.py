"""Visualise the online mapping carried by the 100 map queries.

Layout of every output PNG
  top strip   : the 6 camera views with the predicted lane lines drawn on the
                road surface (the polylines are 2-D, so they are placed at the
                road height estimated from the bottom face of the GT vehicles)
  bottom left : BEV of the predicted polylines, coloured and styled by class
  bottom right: BEV of the GT polylines, same styling
  both panels also carry the GT agents (faint) and the ego box

Decoding follows the inference path exactly: MapNMSFreeCoder takes the top-100
(query, class) pairs of the last decoder layer by sigmoid score and returns
map_all_pts_preds untouched -- the polylines are already in metric ego
coordinates, the denormalisation in the coder is commented out.

Usage (from repo root):
  CUDA_VISIBLE_DEVICES=3 python tools/viz_map_pred.py \
      --config adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py \
      --ckpt ckpts/drivetransformer_large.pth \
      --out-dir ol_eval_results/map_viz --num 4 --stride 100 --verify
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dt_viz_common import (  # noqa: E402
    MAP_SKIP, MAP_STYLE, add_common_args, bev_axis, build, cameras_of, draw_ego,
    draw_boxes, draw_camera_strip, draw_map, draw_polyline_on_cameras, ego_size,
    frame_sequence, gt_of, ground_z, make_figure, map_of, pick_frames, run_sequence,
    verify_indices,
)
from mmcv.datasets import build_dataset  # noqa: E402


def decode_map(outs, num_classes, max_num=100, score_thr=0.3):
    """Top-scoring map polylines. Mirrors MapNMSFreeCoder.decode_single."""
    cls = outs["map_all_cls_scores"][-1][0].float().sigmoid().cpu().numpy()
    pts = outs["map_all_pts_preds"][-1][0].float().cpu().numpy()  # [query, n_pts, 2]

    flat = cls.reshape(-1)
    order = np.argsort(-flat)[:max_num]
    out, seen = [], set()
    for k in order:
        score = float(flat[k])
        if score < score_thr:
            break
        qi, label = int(k // num_classes), int(k % num_classes)
        if qi in seen:
            continue
        seen.add(qi)
        out.append(dict(query=qi, score=score, label=label, pts=pts[qi]))
    return out


def chamfer(a, b):
    """Symmetric mean nearest-point distance between two polylines."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return 0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean())


def draw(preds, map_pts, map_labels, gt_rows, gt_labels, cams, cam_names, lidar2img,
         ego_wl, map_classes, pc_range, title, out_path, cam_lines=True,
         label_queries=True):
    cls_cmap = plt.get_cmap("tab10")
    fig, cam_axes, (axP, axG), axLeg = make_figure(cams, [1, 1], title)

    # ---- left panel: predicted polylines ----------------------------------- #
    pred_used, n_pts, k = set(), 0, 0
    for pr in preds:
        name = map_classes[pr["label"]]
        if name in MAP_SKIP or name not in MAP_STYLE:
            continue
        k += 1
        color, ls, lw = MAP_STYLE[name]
        pts = pr["pts"]
        n_pts = len(pts)
        axP.plot(pts[:, 0], pts[:, 1], color=color, ls=ls, lw=lw, alpha=0.9, zorder=3)
        # every regressed point, so their number and spacing can be read off
        axP.plot(pts[:, 0], pts[:, 1], ls="", marker="o", ms=3.2, mfc="white",
                 mec=color, mew=0.9, zorder=4)
        axP.plot(pts[0, 0], pts[0, 1], "s", ms=4.5, color=color, zorder=5)
        if label_queries:
            # staggered, because duplicate queries often start on the same spot
            axP.annotate(f"q{pr['query']} {pr['score']:.2f}", (pts[0, 0], pts[0, 1]),
                         textcoords="offset points", xytext=(4, 4 + 7 * (k % 3)),
                         fontsize=6, color=color, zorder=6)
        pred_used.add(name)
    n_drawn = sum(1 for p in preds if map_classes[p["label"]] not in MAP_SKIP)
    draw_boxes(axP, gt_rows, gt_labels, cls_cmap, alpha=0.25, lw=0.5, heading=False)
    draw_ego(axP, ego_wl)
    bev_axis(axP, pc_range,
             f"predicted map: {n_drawn} polylines of {n_pts} points each\n"
             f"labelled 'q<query id> <score>' at the first point (square)")

    # ---- right panel: GT polylines ----------------------------------------- #
    gt_used = draw_map(axG, map_pts, map_labels, map_classes, alpha=0.9, zorder=3)
    n_gt, n_gt_pts = 0, 0
    if map_pts is not None:
        for pts, lb in zip(map_pts, map_labels):
            name = map_classes[int(lb)]
            if name in MAP_SKIP:
                continue
            n_gt += 1
            n_gt_pts = len(pts)
            color = MAP_STYLE[name][0]
            axG.plot(pts[:, 0], pts[:, 1], ls="", marker="o", ms=3.2, mfc="white",
                     mec=color, mew=0.9, zorder=4)
            axG.plot(pts[0, 0], pts[0, 1], "s", ms=4.5, color=color, zorder=5)
    cls_used, _ = draw_boxes(axG, gt_rows, gt_labels, cls_cmap, alpha=0.25, lw=0.5,
                             heading=False)
    draw_ego(axG, ego_wl)
    bev_axis(axG, pc_range,
             f"ground-truth map: {n_gt} polylines of {n_gt_pts} points each\n"
             f"('{'/'.join(sorted(MAP_SKIP))}' class is never drawn)")

    # ---- legend ------------------------------------------------------------ #
    handles = [Line2D([], [], color=MAP_STYLE[m][0], ls=MAP_STYLE[m][1],
                      lw=MAP_STYLE[m][2], label=f"map: {m}")
               for m in map_classes if m in (pred_used | gt_used)]
    handles += [
        Patch(facecolor="red", label="ego"),
        Patch(facecolor="0.5", alpha=0.25, edgecolor="black", label="GT agents (context)"),
        Line2D([], [], marker="o", ls="", mfc="white", mec="0.4", ms=4,
               label=f"one regressed point ({n_pts} per line)"),
        Line2D([], [], marker="s", ls="", color="0.4", ms=4.5, label="first point"),
    ]
    axLeg.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)

    # ---- camera strip ------------------------------------------------------ #
    if cams:
        draw_camera_strip(cam_axes, cams, cam_names)
        if cam_lines and lidar2img is not None:
            z = ground_z(gt_rows)
            for pr in preds:
                name = map_classes[pr["label"]]
                if name in MAP_SKIP or name not in MAP_STYLE:
                    continue
                color, ls, _ = MAP_STYLE[name]
                draw_polyline_on_cameras(cam_axes, cams, lidar2img, pr["pts"], z,
                                         color, lw=1.6, ls=ls, marker="o", ms=2.2)
            cam_axes[0].text(0.02, 0.03, f"lines drawn at road z = {z:.2f} m",
                             transform=cam_axes[0].transAxes, fontsize=5, color="white")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def verify(cfg, model, ds, args):
    head = model.pts_bbox_head
    map_classes = list(cfg.map_classes)
    print("\n================ map verification ================")
    print(f"map classes: {map_classes} (never drawn: {sorted(MAP_SKIP)})")
    print(f"map queries: {head.map_num_query}, points per polyline: "
          f"{head.map_num_pts_per_vec}")

    matched, control, n_pred, n_gt, road_z = [], [], [], [], []
    rng = np.random.default_rng(0)
    pred_range = [np.inf, -np.inf, np.inf, -np.inf]
    done = 0
    for k, cur in enumerate(verify_indices(ds, args.verify_frames)):
        sample = ds[int(cur)]
        gt_pts, gt_lab = map_of(sample)
        if gt_pts is None or len(gt_pts) == 0:
            continue
        road_z.append(ground_z(gt_of(sample)[1].numpy()))
        seq, token = frame_sequence(ds, int(cur), args.warmup)
        outs = run_sequence(model, head, seq, f"{token}#mapverify{k}")
        preds = decode_map(outs, len(map_classes), score_thr=args.score)
        n_pred.append(len(preds))
        n_gt.append(len(gt_pts))
        for pr in preds:
            pts = pr["pts"]
            pred_range = [min(pred_range[0], pts[:, 0].min()), max(pred_range[1], pts[:, 0].max()),
                          min(pred_range[2], pts[:, 1].min()), max(pred_range[3], pts[:, 1].max())]
            same = [g for g in range(len(gt_pts)) if int(gt_lab[g]) == pr["label"]]
            if not same:
                continue
            matched.append(min(chamfer(pts, gt_pts[g]) for g in same))
            control.append(chamfer(pts, gt_pts[rng.integers(len(gt_pts))]))
        done += 1

    print(f"\npredicted polyline extent: x [{pred_range[0]:.1f}, {pred_range[1]:.1f}] m, "
          f"y [{pred_range[2]:.1f}, {pred_range[3]:.1f}] m")
    print(f"pc_range says x [{cfg.point_cloud_range[0]:g}, {cfg.point_cloud_range[3]:g}], "
          f"y [{cfg.point_cloud_range[1]:g}, {cfg.point_cloud_range[4]:g}] "
          "(a similar extent => the polylines are already metric)")
    print(f"\npolyline counts over {done} frames spread across the val set: predicted "
          f"{np.mean(n_pred):.1f} at score >= {args.score}, GT {np.mean(n_gt):.1f}")
    print(f"road height used for the camera overlay: median {np.median(road_z):.2f} m, "
          f"spread [{np.min(road_z):.2f}, {np.max(road_z):.2f}] m")

    def stat(name, v):
        v = np.asarray(v)
        if not len(v):
            print(f"  {name:<34s} (no samples)")
            return
        print(f"  {name:<34s} n={len(v):4d}  median {np.median(v):.3f} m  "
              f"mean {v.mean():.3f} m  p90 {np.percentile(v, 90):.3f} m")

    print("\nchamfer distance to the ground truth:")
    stat("to the best GT of the same class", matched)
    stat("to a random GT (control)", control)
    print("==================================================\n")


def main():
    p = argparse.ArgumentParser()
    add_common_args(p, "ol_eval_results/map_viz")
    p.add_argument("--score", type=float, default=0.3, help="map score threshold")
    p.add_argument("--no-query-labels", action="store_true",
                   help="do not annotate each polyline with its query id and score")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg, model = build(args.config, args.ckpt)
    head = model.pts_bbox_head
    ds = build_dataset(cfg.data.val)
    print(f"val samples: {len(ds)}")
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]

    if args.verify:
        verify(cfg, model, ds, args)

    rendered = 0
    for i in pick_frames(ds, args):
        seq, token = frame_sequence(ds, i, args.warmup)
        sample = seq[-1]
        _, rows, labels = gt_of(sample)
        outs = run_sequence(model, head, seq, f"{token}#map{i}")
        preds = decode_map(outs, len(cfg.map_classes), score_thr=args.score)
        map_pts, map_labels = map_of(sample)
        cams, cam_names, lidar2img = cameras_of(sample, cfg)
        out_path = os.path.join(args.out_dir, f"frame_{i:05d}.png")
        n_drawn = sum(1 for pr in preds if cfg.map_classes[pr["label"]] not in MAP_SKIP)
        title = (f"{tag}   online map prediction   frame {i}   "
                 f"({n_drawn} polylines at score >= {args.score}, "
                 f"{len(preds) - n_drawn} more of the undrawn '{'/'.join(MAP_SKIP)}' class)")
        draw(preds, map_pts, map_labels, rows.numpy(), labels.numpy(), cams, cam_names,
             lidar2img, ego_size(sample), list(cfg.map_classes),
             list(cfg.point_cloud_range), title, out_path,
             cam_lines=not args.no_cam_boxes,
             label_queries=not args.no_query_labels)
        print(f"[{i}] map preds={len(preds)} -> {out_path}", flush=True)
        rendered += 1
    print(f"done, {rendered} frames saved to {args.out_dir}")


if __name__ == "__main__":
    main()
