"""Visualise the motion prediction carried by the 900 agent queries.

Layout of every output PNG
  top strip   : the 6 camera views with the detected boxes and the most likely
                predicted future of each one projected back onto them
  bottom left : BEV of the prediction -- detected boxes, all 6 predicted modes
                per agent (thin) and the most likely mode (thick, with an
                end-of-horizon dot)
  bottom right: BEV of the ground truth -- GT boxes and GT futures
  both panels also carry the map lane lines and the ego box

Decoding follows the inference path exactly: CustomNMSFreeCoder takes the
top-100 (query, class) pairs of the last decoder layer by sigmoid score,
denormalize_bbox turns the 10-code into a box, and
head.rotate_agent_trajs_to_ego rotates the agent-frame trajectory back into the
ego frame. The 6 modes are *semantic* -- the training loss supervises mode k
with the GT trajectory whose class is k -- so all_traj_cls_scores picks the mode
and every mode has the fixed meaning listed in TRAJ_CLASSES.

Usage (from repo root):
  CUDA_VISIBLE_DEVICES=3 python tools/viz_motion_pred.py \
      --config adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py \
      --ckpt ckpts/drivetransformer_large.pth \
      --out-dir ol_eval_results/motion_viz --num 4 --stride 100 --verify
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dt_viz_common import (  # noqa: E402
    AGENT_DT, DOT_SIZE, MAP_STYLE, add_common_args, agent_traj_to_ego, bev_axis,
    bev_poly, build, cameras_of, decode_box_code, draw_box_on_cameras, draw_ego,
    draw_boxes, draw_camera_strip, draw_map, draw_polyline_on_cameras, ego_size,
    frame_sequence, gt_agent_trajs, gt_of, make_figure, map_of, pick_frames,
    run_sequence, verify_indices,
)
from mmcv.datasets import build_dataset  # noqa: E402

# The traj loss supervises mode k with the GT trajectories of class k
# (TrajPreprocess.assign_class), so the modes carry these fixed meanings.
TRAJ_CLASSES = ["straight-right", "straight-left", "right", "left",
                "straight-fast", "straight-slow"]


def decode_detections(outs, num_classes, fut_ts, max_num=100, score_thr=0.3):
    """Top-scoring detections with their multi-modal futures, ego frame.

    Mirrors CustomNMSFreeCoder.decode_single + head.get_bboxes; the only
    addition is dropping duplicates when one query wins with two classes.
    """
    cls = outs["all_cls_scores"][-1][0].float().sigmoid().cpu().numpy()
    box = outs["all_bbox_preds"][-1][0].float().cpu().numpy()
    traj = outs["all_traj_preds"][-1][0].float().cpu().numpy()
    tcls = outs["all_traj_cls_scores"][-1][0].float().cpu().numpy()
    n_mode = traj.shape[1]

    flat = cls.reshape(-1)
    order = np.argsort(-flat)[:max_num]
    dets, seen = [], set()
    for k in order:
        score = float(flat[k])
        if score < score_thr:
            break
        qi, label = int(k // num_classes), int(k % num_classes)
        if qi in seen:
            continue
        seen.add(qi)
        row = decode_box_code(box[qi])
        modes = agent_traj_to_ego(traj[qi].reshape(n_mode, fut_ts, 2), row[6]) + row[:2]
        dets.append(dict(query=qi, score=score, label=label, row=row, modes=modes,
                         mode_scores=tcls[qi], best=int(np.argmax(tcls[qi]))))
    return dets


def draw(dets, gt_rows, gt_labels, gt_trajs, gt_masks, map_pts, map_labels,
         cams, cam_names, lidar2img, ego_wl, class_names, map_classes, pc_range,
         title, out_path, cam_boxes=True):
    cls_cmap = plt.get_cmap("tab10")
    fig, cam_axes, (axP, axG), axLeg = make_figure(cams, [1, 1], title)

    horizon = gt_trajs.shape[1] * AGENT_DT if len(gt_trajs) else 0
    map_used = draw_map(axP, map_pts, map_labels, map_classes)
    draw_map(axG, map_pts, map_labels, map_classes)

    # ---- left panel: predictions ------------------------------------------ #
    pred_classes, pred_dot = set(), False
    for det in dets:
        color = cls_cmap(det["label"] % 10)
        pred_classes.add(det["label"])
        x, y, _, sx, sy, _, yaw = det["row"][:7]
        if max(sx, sy) < DOT_SIZE:
            axP.plot(x, y, "o", ms=7, mfc=color, mec="black", mew=0.8, zorder=3)
            pred_dot = True
        else:
            axP.add_patch(Polygon(bev_poly(x, y, sx, sy, yaw), closed=True, facecolor=color,
                                  alpha=0.35, edgecolor=color, lw=1.4, zorder=2))
        # every mode is anchored at the agent centre, the same way the GT future
        # starts from the GT centre, so the two panels can be read side by side
        for m in range(len(det["modes"])):
            if m == det["best"]:
                continue
            pts = det["modes"][m]
            line = np.concatenate([det["row"][None, :2], pts])
            axP.plot(line[:, 0], line[:, 1], color=color, lw=0.7, alpha=0.45, zorder=3)
            axP.plot(pts[:, 0], pts[:, 1], ls="", marker="o", ms=1.6, color=color,
                     alpha=0.45, zorder=3)
        pts = det["modes"][det["best"]]
        line = np.concatenate([det["row"][None, :2], pts])
        axP.plot(line[:, 0], line[:, 1], color=color, lw=1.8, alpha=0.95, zorder=4)
        # one marker per predicted waypoint, so the sampling is visible
        axP.plot(pts[:, 0], pts[:, 1], ls="", marker="o", ms=3.2, mfc="white",
                 mec=color, mew=0.9, zorder=5)
        axP.plot(pts[-1, 0], pts[-1, 1], "s", ms=4.0, color=color, zorder=5)
        # written along +y so that agents side by side cannot overlap each other
        axP.text(x + max(sx, 1.0) / 2.0 + 0.3, y, f"{det['score']:.2f} "
                 f"{TRAJ_CLASSES[det['best']]}", fontsize=5.5, rotation=90,
                 ha="left", va="center", color=color, zorder=6)
    n_wp = dets[0]["modes"].shape[1] if dets else 0
    draw_ego(axP, ego_wl)
    bev_axis(axP, pc_range,
             f"prediction: {len(dets)} agents, {len(TRAJ_CLASSES)} modes each\n"
             f"thick = most likely mode, {n_wp} waypoints {AGENT_DT:g} s apart "
             f"({horizon:g} s)")

    # ---- right panel: ground truth ----------------------------------------- #
    gt_classes, gt_dot = draw_boxes(axG, gt_rows, gt_labels, cls_cmap)
    for g in range(len(gt_rows)):
        valid = np.flatnonzero(~gt_masks[g])
        stop = int(valid[0]) if len(valid) else gt_trajs.shape[1]
        if stop < 1:
            continue
        pts = np.concatenate([gt_rows[g:g + 1, :2], gt_trajs[g, :stop]])
        color = cls_cmap(int(gt_labels[g]) % 10)
        axG.plot(pts[:, 0], pts[:, 1], color=color, lw=1.8, alpha=0.95, zorder=4)
        axG.plot(pts[1:, 0], pts[1:, 1], ls="", marker="o", ms=3.2, mfc="white",
                 mec=color, mew=0.9, zorder=5)
        axG.plot(pts[-1, 0], pts[-1, 1], "s", ms=4.0, color=color, zorder=5)
    draw_ego(axG, ego_wl)
    bev_axis(axG, pc_range,
             f"ground truth: {len(gt_rows)} agents\n"
             f"futures on the same {gt_trajs.shape[1] if len(gt_trajs) else 0} "
             f"waypoints {AGENT_DT:g} s apart ({horizon:g} s)")

    # ---- legend ------------------------------------------------------------ #
    handles = [Patch(facecolor=cls_cmap(c % 10), alpha=0.5, edgecolor="black",
                     label=class_names[c]) for c in sorted(pred_classes | gt_classes)]
    if pred_dot or gt_dot:
        handles.append(Line2D([], [], marker="o", ls="", mfc="0.5", mec="black",
                              label=f"agent smaller than {DOT_SIZE:g} m"))
    handles += [
        Patch(facecolor="red", label="ego"),
        Line2D([], [], color="0.35", lw=1.8, label="most likely / GT future"),
        Line2D([], [], color="0.35", lw=0.7, alpha=0.5, label="other predicted modes"),
        Line2D([], [], marker="o", ls="", mfc="white", mec="0.35", ms=4,
               label=f"one waypoint ({n_wp} per future)"),
        Line2D([], [], marker="s", ls="", color="0.35", ms=4,
               label=f"end of the {horizon:g} s horizon"),
    ]
    handles += [Line2D([], [], color=MAP_STYLE[m][0], ls=MAP_STYLE[m][1],
                       lw=MAP_STYLE[m][2], label=f"map: {m}")
                for m in map_classes if m in map_used]
    handles.append(Line2D([], [], color="none", label="\nthe 6 modes are fixed:"))
    handles += [Line2D([], [], color="none", label=f"   {m}") for m in TRAJ_CLASSES]
    axLeg.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)

    # ---- camera strip ------------------------------------------------------ #
    if cams:
        draw_camera_strip(cam_axes, cams, cam_names)
        if cam_boxes and lidar2img is not None:
            for det in dets:
                color = cls_cmap(det["label"] % 10)
                draw_box_on_cameras(cam_axes, cams, lidar2img, det["row"], color)
                z = det["row"][2] - det["row"][5] / 2.0  # wheels, i.e. the road
                pts = np.concatenate([det["row"][None, :2], det["modes"][det["best"]]])
                draw_polyline_on_cameras(cam_axes, cams, lidar2img, pts, z, color,
                                         lw=1.4, marker="o", ms=2.2)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def verify(cfg, model, ds, args):
    head = model.pts_bbox_head
    fut_ts = head.fut_ts
    print("\n================ motion verification ================")
    print(f"fut_ts={fut_ts} waypoints, {AGENT_DT} s apart -> {fut_ts * AGENT_DT} s horizon; "
          f"fut_mode={head.fut_mode} ({', '.join(TRAJ_CLASSES)})")

    # (1) GT trajectories must agree with the annotated velocities
    vel_err, step_growth = [], []
    for i in verify_indices(ds, 200):
        sample = ds[int(i)]
        _, rows, _ = gt_of(sample)
        if len(rows) == 0:
            continue
        rows = rows.numpy()
        trajs, masks = gt_agent_trajs(sample, fut_ts)
        moving = np.hypot(rows[:, 7], rows[:, 8]) > 2.0
        for g in np.flatnonzero(moving & masks[:, 0]):
            expected = rows[g, :2] + rows[g, 7:9] * AGENT_DT
            vel_err.append(np.linalg.norm(trajs[g, 0] - expected))
            if masks[g].all():
                d = np.linalg.norm(trajs[g] - rows[g, :2], axis=-1)
                step_growth.append(d[-1] / max(d[0], 1e-6))
    print(f"\nGT future vs annotated velocity ({len(vel_err)} moving agents):")
    print(f"  |traj[0] - (centre + v*{AGENT_DT}s)|  median {np.median(vel_err):.3f} m  "
          f"p90 {np.percentile(vel_err, 90):.3f} m")
    print(f"  |traj[-1]| / |traj[0]| median {np.median(step_growth):.2f} "
          f"(~{fut_ts} => cumulative, ~1 => differential)")

    # (2) predicted futures against the GT of the agent they detect
    cols = ("ade_best", "fde_best", "ade_min", "fde_min", "ade_rand",
            "agent_frame", "moved_pred", "moved_gt")
    acc = {"still": {c: [] for c in cols}, "moving": {c: [] for c in cols}}
    alts = {}
    rng = np.random.default_rng(0)
    n_det = n_matched = n_gt = done = 0
    for k, cur in enumerate(verify_indices(ds, args.verify_frames)):
        _, rows, labels = gt_of(ds[int(cur)])
        if len(rows) < 2:
            continue
        seq, token = frame_sequence(ds, int(cur), args.warmup)
        outs = run_sequence(model, head, seq, f"{token}#mverify{k}")
        dets = decode_detections(outs, len(cfg.class_names), fut_ts,
                                 score_thr=args.score)
        raw = outs["all_traj_preds"][-1][0].float().cpu().numpy()
        gt_rows = rows.numpy()
        gt_trajs, gt_masks = gt_agent_trajs(seq[-1], fut_ts)
        n_det += len(dets)
        n_gt += len(gt_rows)
        for det in dets:
            d = np.linalg.norm(gt_rows[:, :2] - det["row"][:2], axis=1)
            g = int(np.argmin(d))
            if d[g] > 2.0 or int(labels[g]) != det["label"] or not gt_masks[g].all():
                continue
            n_matched += 1
            travel = np.linalg.norm(gt_trajs[g][-1] - gt_rows[g, :2])
            a = acc["moving" if travel > 1.0 else "still"]
            err = np.linalg.norm(det["modes"] - gt_trajs[g][None], axis=-1)  # [mode, T]
            a["ade_best"].append(err[det["best"]].mean())
            a["fde_best"].append(err[det["best"], -1])
            a["ade_min"].append(err.mean(axis=1).min())
            a["fde_min"].append(err[:, -1].min())
            a["ade_rand"].append(np.linalg.norm(
                det["modes"][det["best"]] - gt_trajs[rng.integers(len(gt_rows))],
                axis=-1).mean())
            a["moved_pred"].append(np.linalg.norm(det["modes"][det["best"]][-1] - det["row"][:2]))
            a["moved_gt"].append(travel)
            # purest test: raw network output against the raw training target,
            # both in the agent's own frame, so none of our geometry is involved
            raw_q = raw[det["query"]].reshape(-1, fut_ts, 2)
            tgt = np.asarray(seq[-1]["gt_attr_labels"].data)[g, :fut_ts * 2].reshape(fut_ts, 2)
            a["agent_frame"].append(
                np.linalg.norm(raw_q - tgt[None], axis=-1).mean(axis=1).min())
            for name, pts in _alt_decodings(raw_q, det["row"], gt_rows[g]).items():
                e = np.linalg.norm(pts - gt_trajs[g][None], axis=-1).mean(axis=1)
                alts.setdefault(name, []).append(e.min())
        done += 1

    print(f"\n{done} frames spread over the val set, {n_gt} GT agents, "
          f"{n_det} detections at score >= "
          f"{args.score}, {n_matched} of them matched to a GT with a full future")
    for kind, a in acc.items():
        v = np.asarray(a["ade_best"])
        if not len(v):
            continue
        label = ("agents that stay put (GT travel <= 1 m)" if kind == "still"
                 else "agents on the move (GT travel > 1 m)")
        print(f"\n{label}, n={len(v)}:")
        print(f"  ADE / FDE, most likely mode   {np.median(a['ade_best']):6.3f} / "
              f"{np.median(a['fde_best']):6.3f} m")
        print(f"  ADE / FDE, best of the 6 modes{np.median(a['ade_min']):6.3f} / "
              f"{np.median(a['fde_min']):6.3f} m")
        print(f"  ADE against a random agent    {np.median(a['ade_rand']):6.3f} m "
              "(the control)")
        print(f"  distance travelled over {fut_ts * AGENT_DT:g} s: predicted "
              f"{np.median(a['moved_pred']):.2f} m vs GT {np.median(a['moved_gt']):.2f} m")

    # the rotation we apply is distance preserving, so comparing the raw output
    # with the raw target must give exactly the same error as comparing our
    # decoded waypoints with the decoded GT -- that is the proof it is exact
    every = np.concatenate([acc[k]["agent_frame"] for k in acc])
    print(f"\ndecode check: minADE of the raw output against the raw target, in the "
          f"agent frame, is {np.median(every):.3f} m;")
    print("  the 'rot with GT yaw + GT centre' row below must match it exactly, "
          "since a rotation cannot change a distance.")
    print("\nminADE under alternative decodings:")
    for name, v in sorted(alts.items(), key=lambda kv: np.median(kv[1])):
        print(f"  {name:<34s} median {np.median(v):.3f} m")
    print("=====================================================\n")


def _rot(traj, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.asarray(traj) @ np.array([[c, -s], [s, c]]).T


def _alt_decodings(raw, row, gt_row):
    """Candidate ways of turning the raw trajectory output into ego waypoints."""
    c, yaw = row[:2], row[6]
    cum = np.cumsum(raw, axis=-2)
    return {
        "rot(-yaw-pi) + centre [in use]": _rot(raw, -yaw - np.pi) + c,
        "rot(yaw+pi) + centre": _rot(raw, yaw + np.pi) + c,
        "rot(-yaw-pi/2) + centre": _rot(raw, -yaw - np.pi / 2) + c,
        "rot(pi/2-yaw) + centre": _rot(raw, np.pi / 2 - yaw) + c,
        "no rotation + centre": raw + c,
        "no rotation, absolute": raw,
        "rot(-yaw-pi), absolute": _rot(raw, -yaw - np.pi),
        "cumsum, rot(-yaw-pi) + centre": _rot(cum, -yaw - np.pi) + c,
        "cumsum, no rotation + centre": cum + c,
        "rot with GT yaw + GT centre": _rot(raw, -gt_row[6] - np.pi) + gt_row[:2],
    }


def main():
    p = argparse.ArgumentParser()
    add_common_args(p, "ol_eval_results/motion_viz")
    p.add_argument("--score", type=float, default=0.3, help="detection score threshold")
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
        outs = run_sequence(model, head, seq, f"{token}#motion{i}")
        dets = decode_detections(outs, len(cfg.class_names), head.fut_ts,
                                 score_thr=args.score)
        gt_trajs, gt_masks = gt_agent_trajs(sample, head.fut_ts)
        map_pts, map_labels = map_of(sample)
        cams, cam_names, lidar2img = cameras_of(sample, cfg)
        out_path = os.path.join(args.out_dir, f"frame_{i:05d}.png")
        title = (f"{tag}   agent motion prediction   frame {i}   "
                 f"({len(dets)} detections at score >= {args.score}, {len(rows)} GT)")
        draw(dets, rows.numpy(), labels.numpy(), gt_trajs, gt_masks, map_pts, map_labels,
             cams, cam_names, lidar2img, ego_size(sample), list(cfg.class_names),
             list(cfg.map_classes), list(cfg.point_cloud_range), title, out_path,
             cam_boxes=not args.no_cam_boxes)
        print(f"[{i}] dets={len(dets)} gt={len(rows)} -> {out_path}", flush=True)
        rendered += 1
    print(f"done, {rendered} frames saved to {args.out_dir}")


if __name__ == "__main__":
    main()
