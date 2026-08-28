"""Visualise the planning output of the ego query.

Layout of every output PNG
  top strip   : the 6 camera views with the planned path drawn on the road
  bottom left : BEV -- planned trajectory vs the logged GT trajectory, the ego
                box, its recent history, the GT agents and the map
  bottom right: the fixed-distance head, i.e. the bearing profile of the plan
                sampled every metre of path length, prediction vs GT

The model has fut_ego_mode = 1, so planning is single-mode and
ego_traj_cls_scores is None. Two horizons come out of the head:

  ego_fut_preds_fix_time  [layer, B, 1, 30, 2]  absolute ego-frame positions,
                          0.1 s apart, i.e. a 3 s plan. The config sets
                          ego_fut_offset_input=False, so neither the prediction
                          nor the GT is cumulative.
  ego_fut_preds_fix_dist  [layer, B, 1, 20, 1]  x / |(x, y)| of the waypoint
                          reached after each metre of path -- the sine of its
                          bearing off the +y (forward) axis, so 0 is straight
                          ahead, positive is to the right.

Usage (from repo root):
  CUDA_VISIBLE_DEVICES=3 python tools/viz_plan_pred.py \
      --config adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py \
      --ckpt ckpts/drivetransformer_large.pth \
      --out-dir ol_eval_results/plan_viz --num 4 --stride 100 --verify
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
    EGO_DT, MAP_STYLE, add_common_args, bev_axis, build, cameras_of, draw_ego,
    draw_boxes, draw_camera_strip, draw_map, draw_polyline_on_cameras, ego_history,
    ego_size, ego_state, frame_sequence, gt_of, ground_z, make_figure, map_of,
    pick_frames, run_sequence, verify_indices,
)
from mmcv.datasets import build_dataset  # noqa: E402

L2_SECONDS = (1.0, 2.0, 3.0)


def decode_plan(outs):
    """(fix-time waypoints [T, 2], fix-dist bearing profile [D] or None)."""
    traj = outs["ego_fut_preds_fix_time"][-1][0].float().cpu().numpy()  # [mode, T, 2]
    mode = 0
    scores = outs.get("ego_traj_cls_scores")
    if scores is not None and scores[-1] is not None and traj.shape[0] > 1:
        mode = int(np.argmax(scores[-1][0].float().cpu().numpy()))
    dist = outs.get("ego_fut_preds_fix_dist")
    profile = None
    if dist is not None and dist[-1] is not None:
        profile = dist[-1][0].float().cpu().numpy()[mode].reshape(-1)
    return traj[mode], profile


def gt_plan(sample):
    traj = np.asarray(sample["ego_fut_trajs_fix_time"].data, dtype=np.float64).reshape(-1, 2)
    mask = np.asarray(sample["ego_fut_masks_fix_time"].data).reshape(-1) > 0
    profile = np.asarray(sample["ego_fut_trajs_fix_dist"].data, dtype=np.float64).reshape(-1)
    pmask = np.asarray(sample["ego_fut_masks_fix_dist"].data).reshape(-1) > 0
    return traj, mask, profile, pmask


def l2_at(pred, gt, mask, dt=EGO_DT):
    """L2 error at each horizon in L2_SECONDS; nan when the GT is not valid."""
    out = []
    for sec in L2_SECONDS:
        k = int(round(sec / dt)) - 1
        ok = k < len(gt) and k < len(pred) and mask[k]
        out.append(np.linalg.norm(pred[k] - gt[k]) if ok else np.nan)
    return np.array(out)


def draw(pred, profile, gt_traj, gt_mask, gt_profile, gt_pmask, hist, gt_rows, gt_labels,
         map_pts, map_labels, cams, cam_names, lidar2img, ego_wl, state, class_names,
         map_classes, pc_range, title, out_path, cam_path=True):
    cls_cmap = plt.get_cmap("tab10")
    fig, cam_axes, (axB, axD), axLeg = make_figure(cams, [1, 0.42], title)
    speed, cmd_far, cmd_near = state

    # ---- left panel: the plan in BEV --------------------------------------- #
    map_used = draw_map(axB, map_pts, map_labels, map_classes)
    cls_used, dot_used = draw_boxes(axB, gt_rows, gt_labels, cls_cmap, alpha=0.35, lw=0.6)
    draw_ego(axB, ego_wl)

    if len(hist) > 1:
        axB.plot(hist[:, 0], hist[:, 1], color="0.45", lw=1.4, ls=":", zorder=5)
    valid = np.flatnonzero(~gt_mask)
    stop = int(valid[0]) if len(valid) else len(gt_traj)
    gt_path = np.concatenate([np.zeros((1, 2)), gt_traj[:stop]])
    axB.plot(gt_path[:, 0], gt_path[:, 1], color="#1b9e77", lw=2.6, alpha=0.9,
             zorder=6, solid_capstyle="round")
    pred_path = np.concatenate([np.zeros((1, 2)), pred])
    axB.plot(pred_path[:, 0], pred_path[:, 1], color="#d95f02", lw=2.2, ls="--",
             zorder=7, solid_capstyle="round")

    errs = l2_at(pred, gt_traj, gt_mask)
    for sec, err in zip(L2_SECONDS, errs):
        k = int(round(sec / EGO_DT)) - 1
        if k >= len(pred):
            continue
        axB.plot(pred[k, 0], pred[k, 1], "o", ms=5, color="#d95f02", zorder=8)
        if k < stop:
            axB.plot(gt_traj[k, 0], gt_traj[k, 1], "o", ms=5, color="#1b9e77", zorder=8)
        axB.annotate(f"{sec:g}s" + ("" if np.isnan(err) else f"  L2 {err:.2f}m"),
                     (pred[k, 0], pred[k, 1]), textcoords="offset points",
                     xytext=(7, 0), fontsize=7, color="#d95f02", zorder=9)
    shown = "  ".join("nan" if np.isnan(e) else f"{e:.2f}" for e in errs)
    bev_axis(axB, pc_range,
             f"planning: {len(pred) * EGO_DT:g} s ahead   speed {speed:.1f} m/s   "
             f"command near={cmd_near} far={cmd_far}\n"
             f"L2 @ {'/'.join(f'{s:g}' for s in L2_SECONDS)} s = {shown} m")

    # ---- right panel: the fixed-distance bearing profile -------------------- #
    if profile is not None:
        metres = np.arange(1, len(profile) + 1)
        axD.plot(profile, metres, color="#d95f02", lw=2.0, ls="--", marker="o", ms=3,
                 label="predicted")
        if gt_profile is not None:
            gp = np.where(gt_pmask[:len(gt_profile)], gt_profile, np.nan)
            axD.plot(gp[:len(metres)], metres[:len(gp)], color="#1b9e77", lw=2.0,
                     marker="o", ms=3, label="ground truth")
        axD.axvline(0.0, color="0.7", lw=0.8)
        axD.set_ylim(0, len(profile) + 1)
        axD.set_xlim(-1.05, 1.05)
        axD.set_xlabel("x / |(x, y)|   (left  <-  0  ->  right)")
        axD.set_ylabel("path length travelled (m)")
        axD.set_title("fixed-distance head:\nbearing of the waypoint at each metre",
                      fontsize=10)
        axD.grid(alpha=0.3)
        axD.legend(fontsize=8, loc="lower right")
    else:
        axD.axis("off")
        axD.text(0.5, 0.5, "no fixed-distance head", ha="center", va="center")

    # ---- legend ------------------------------------------------------------ #
    handles = [
        Line2D([], [], color="#d95f02", lw=2.2, ls="--", label="planned trajectory"),
        Line2D([], [], color="#1b9e77", lw=2.6, label="logged GT trajectory"),
        Line2D([], [], color="0.45", lw=1.4, ls=":", label="ego history"),
        Patch(facecolor="red", label="ego"),
    ]
    handles += [Patch(facecolor=cls_cmap(c % 10), alpha=0.35, edgecolor="black",
                      label=class_names[c]) for c in sorted(cls_used)]
    handles += [Line2D([], [], color=MAP_STYLE[m][0], ls=MAP_STYLE[m][1],
                       lw=MAP_STYLE[m][2], label=f"map: {m}")
                for m in map_classes if m in map_used]
    axLeg.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)

    # ---- camera strip ------------------------------------------------------ #
    if cams:
        draw_camera_strip(cam_axes, cams, cam_names)
        if cam_path and lidar2img is not None:
            z = ground_z(gt_rows)
            draw_polyline_on_cameras(cam_axes, cams, lidar2img, gt_path, z,
                                     "#1b9e77", lw=2.0)
            draw_polyline_on_cameras(cam_axes, cams, lidar2img, pred_path, z,
                                     "#d95f02", lw=2.0, ls="--")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def verify(cfg, model, ds, args):
    head = model.pts_bbox_head
    print("\n================ planning verification ================")
    print(f"fut_ego_mode={head.fut_ego_mode} (multi-modal: {head.ego_multi_modal}), "
          f"fix_time={head.fut_ts_ego_fix_time} pts @ {EGO_DT} s, "
          f"fix_dist={head.fut_ts_ego_fix_dist} pts @ 1 m")

    # (1) the GT plan must agree with the logged ego speed
    speed_err = []
    for i in verify_indices(ds, 200):
        sample = ds[int(i)]
        gt_traj, gt_mask, _, _ = gt_plan(sample)
        speed, _, _ = ego_state(sample)
        k = int(round(1.0 / EGO_DT)) - 1
        if k < len(gt_mask) and gt_mask[k] and speed > 2.0:
            speed_err.append(abs(np.linalg.norm(gt_traj[k]) - speed * 1.0))
    print(f"\nGT plan vs logged speed ({len(speed_err)} frames above 2 m/s):")
    print(f"  | |traj@1s| - speed*1s |  median {np.median(speed_err):.3f} m  "
          f"p90 {np.percentile(speed_err, 90):.3f} m")

    # (2) the planned trajectory against the log
    errs, ctrl, prof_err, prof_rng = [], [], [], [np.inf, -np.inf]
    prev = None
    for done, cur in enumerate(verify_indices(ds, args.verify_frames)):
        seq, token = frame_sequence(ds, int(cur), args.warmup)
        outs = run_sequence(model, head, seq, f"{token}#planverify{done}")
        pred, profile = decode_plan(outs)
        gt_traj, gt_mask, gt_profile, gt_pmask = gt_plan(seq[-1])
        e = l2_at(pred, gt_traj, gt_mask)
        if not np.isnan(e).all():
            errs.append(e)
        if prev is not None:
            ctrl.append(l2_at(prev, gt_traj, gt_mask))  # another frame's plan on this GT
        prev = pred
        if profile is not None:
            prof_rng = [min(prof_rng[0], profile.min()), max(prof_rng[1], profile.max())]
            ok = gt_pmask[:len(profile)]
            if ok.any():
                prof_err.append(np.abs(profile[ok] - gt_profile[:len(profile)][ok]).mean())

    errs = np.array(errs)
    ctrl = np.array(ctrl) if len(ctrl) else None
    print(f"\nplanned vs logged trajectory ({len(errs)} frames spread over the val set):")
    for j, sec in enumerate(L2_SECONDS):
        col = errs[:, j]
        col = col[~np.isnan(col)]
        line = (f"  L2 @ {sec:g}s   n={len(col):3d}  mean {col.mean():.3f} m  "
                f"median {np.median(col):.3f} m")
        if ctrl is not None:
            c = ctrl[:, j]
            c = c[~np.isnan(c)]
            line += f"   | another frame's plan (control): mean {c.mean():.3f} m"
        print(line)
    print(f"  average over the three horizons: {np.nanmean(errs):.3f} m "
          "(compare with the open-loop numbers in the experiment log)")
    if prof_err:
        print(f"\nfixed-distance bearing profile: range [{prof_rng[0]:.2f}, "
              f"{prof_rng[1]:.2f}] (must stay inside [-1, 1])")
        print(f"  mean |pred - GT| = {np.mean(prof_err):.3f}")
    print("=======================================================\n")


def main():
    p = argparse.ArgumentParser()
    add_common_args(p, "ol_eval_results/plan_viz")
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
        outs = run_sequence(model, head, seq, f"{token}#plan{i}")
        pred, profile = decode_plan(outs)
        gt_traj, gt_mask, gt_profile, gt_pmask = gt_plan(sample)
        map_pts, map_labels = map_of(sample)
        cams, cam_names, lidar2img = cameras_of(sample, cfg)
        out_path = os.path.join(args.out_dir, f"frame_{i:05d}.png")
        errs = l2_at(pred, gt_traj, gt_mask)
        title = (f"{tag}   planning   frame {i}   "
                 f"(avg L2 " + ("nan" if np.isnan(errs).all() else f"{np.nanmean(errs):.2f} m") + ")")
        draw(pred, profile, gt_traj, gt_mask, gt_profile, gt_pmask,
             ego_history(sample), rows.numpy(), labels.numpy(), map_pts, map_labels,
             cams, cam_names, lidar2img, ego_size(sample), ego_state(sample),
             list(cfg.class_names), list(cfg.map_classes),
             list(cfg.point_cloud_range), title, out_path,
             cam_path=not args.no_cam_boxes)
        print(f"[{i}] avg L2 {np.nanmean(errs):.2f} m -> {out_path}", flush=True)
        rendered += 1
    print(f"done, {rendered} frames saved to {args.out_dir}")


if __name__ == "__main__":
    main()
