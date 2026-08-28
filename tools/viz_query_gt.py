"""Visualize the correspondence between the 900 agent queries (30x30 BEV grid)
and the GT agents of DriveTransformer.

Layout of every output PNG
  top strip   : the 6 camera views, with the GT boxes (solid) and the matched
                query predictions (dashed) projected back onto them
  bottom left : the 30x30 agent-query grid, one cell per query, anchored at the
                pc_range linspace positions (see head.init_weights); the cells
                matched to a GT are filled and labelled with the GT id
  bottom right: BEV in the ego frame -- GT agents (filled, coloured by class),
                the matched queries' predicted boxes (dashed), ego (red),
                map lane lines; a line connects every matched cell to its GT

The matching replicates the training path exactly: the l0 "pre-sift" assignment
(loss_single_box_only -> _get_target_single_box_only) runs the head's own
assigner + sampler on agent_pre_cls_scores / agent_pre_coord_preds over all 900
queries, so the query ids index the 30x30 grid directly.

Usage (from repo root):
  CUDA_VISIBLE_DEVICES=4 python tools/viz_query_gt.py \
      --config adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py \
      --ckpt ckpts/drivetransformer_large.pth \
      --out-dir ol_eval_results/query_gt_viz --num 4 --stride 100

Add --verify to print a numeric audit of the conventions below instead of
(before) rendering.

Coordinate / box conventions -- all of these are checked by --verify
  * BEV data frame: +x = right, +y = forward.
  * A box row is (x, y, z, x_size, y_size, z_size, yaw, vx, vy) -- the same
    layout the head documents at drivetransformer_head.py:1691. mmdet3d rotates
    the box local axes by *minus* yaw (rotation_3d_in_axis uses the transposed
    matrix), so the x_size edge points along -yaw and the y_size edge -- the
    LONG one, 4.8 m vs 1.9 m for a car -- along pi/2 - yaw. The driving
    direction is the opposite of the y_size axis, i.e. -yaw - pi/2; that is
    exactly where (vx, vy) points, to machine precision, for every moving agent.
  * The predicted 10-code is normalize_bbox()'s output,
    (x, y, log x_size, log y_size, z, log z_size, sin yaw, cos yaw, vx, vy),
    so a prediction is decoded with the very same rule as a GT row.
  * Agent query grid: xs = linspace(pc_range[0], pc_range[3], 30) (x, lateral),
    ys = linspace(pc_range[1], pc_range[4], 30) (y, longitudinal); after
    meshgrid(indexing='xy').flatten() query i sits at (xs[i % 30], ys[i // 30]).
    The script reads the anchors straight out of the head, so it stays correct
    even if a checkpoint ships non-grid reference points.
"""
import argparse
import importlib
import os

import numpy as np
import torch

torch.set_float32_matmul_precision("high")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Polygon

from mmcv import Config
from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.parallel import DataContainer
from mmcv.parallel.collate import collate
from mmcv.utils import wrap_fp16_model

# forward_test chokes on bare bool/int entries, so only these are handed over.
INPUT_KEYS = [
    "img", "ego_his_trajs", "ego_lcf_feat", "ego_fut_cmd",
    "lidar2img", "cam_intrinsic", "timestamp", "ego_pose", "ego_pose_inv", "pad_shape",
]

MAP_STYLE = {  # map class -> (colour, linestyle, linewidth); 'Center' is skipped
    "Broken": ("#2f2f2f", "-", 1.9),
    "Solid": ("#e07b00", "-", 1.2),
    "SolidSolid": ("#e07b00", "--", 1.2),
    "TrafficLight": ("#c9184a", "-", 1.2),
    "StopSign": ("#7b2cbf", "-", 1.2),
}
MAP_SKIP = {"Center"}
EGO_SIZE = (2.1, 4.7)  # (x_size, y_size) fallback when ego_lcf_feat looks odd
DOT_SIZE = 1.5         # agents whose footprint is below this are drawn as dots


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def bev_poly(cx, cy, size_x, size_y, yaw):
    """Footprint (4x2, counter-clockwise) of a box row in the BEV frame.

    mmdet3d applies R(-yaw) to the box local axes, hence the local +x axis
    (length ``size_x``) points along -yaw and the local +y axis (``size_y``)
    along pi/2 - yaw.  Kept in one place so GT and predictions cannot drift.
    """
    ux = np.array([np.cos(yaw), -np.sin(yaw)]) * (size_x / 2.0)
    uy = np.array([np.sin(yaw), np.cos(yaw)]) * (size_y / 2.0)
    c = np.array([cx, cy], dtype=float)
    return np.stack([c - ux - uy, c + ux - uy, c + ux + uy, c - ux + uy])


def heading_vec(yaw):
    """Unit vector the agent drives towards: opposite the local +y axis."""
    return np.array([-np.sin(yaw), -np.cos(yaw)])


def heading_angle(yaw):
    return np.arctan2(-np.cos(yaw), -np.sin(yaw))


def box_corners_3d(row):
    """8x3 corners of (x, y, z_centre, size_x, size_y, size_z, yaw, ...)."""
    cx, cy, cz, sx, sy, sz, yaw = row[:7]
    base = bev_poly(cx, cy, sx, sy, yaw)
    bottom = np.concatenate([base, np.full((4, 1), cz - sz / 2.0)], axis=1)
    top = np.concatenate([base, np.full((4, 1), cz + sz / 2.0)], axis=1)
    return np.concatenate([bottom, top], axis=0)


BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def project(pts3d, lidar2img, width, height, min_depth=0.5):
    """Project 3D points to one camera; None if behind or fully off-frame."""
    hom = np.concatenate([pts3d, np.ones((len(pts3d), 1))], axis=1)
    cam = hom @ np.asarray(lidar2img).T
    if (cam[:, 2] < min_depth).any():
        return None
    uv = cam[:, :2] / cam[:, 2:3]
    if (uv[:, 0].max() < 0 or uv[:, 0].min() > width
            or uv[:, 1].max() < 0 or uv[:, 1].min() > height):
        return None
    return uv


def decode_pred(code):
    """(x, y, z, size_x, size_y, size_z, yaw) from a normalize_bbox 10-code."""
    return np.array([
        code[0], code[1], code[4],
        np.exp(np.clip(code[2], -3.0, 3.0)),
        np.exp(np.clip(code[3], -3.0, 3.0)),
        np.exp(np.clip(code[5], -3.0, 3.0)),
        np.arctan2(code[6], code[7]),
    ])


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# model / data plumbing
# --------------------------------------------------------------------------- #
def to_device(obj, device):
    if torch.is_tensor(obj):
        obj = obj.to(device)
        return obj.float() if obj.dtype == torch.float64 else obj
    if isinstance(obj, list):
        return [to_device(o, device) for o in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def build(cfg_path, ckpt_path):
    cfg = Config.fromfile(cfg_path)
    if getattr(cfg, "plugin", False):
        importlib.import_module(os.path.dirname(cfg.plugin_dir).replace("/", "."))
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    new_state = {k.replace("model.", "").replace("._orig_mod", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"loaded ckpt: {len(new_state)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("  missing e.g.:", list(missing)[:5])
    if unexpected:
        print("  unexpected e.g.:", list(unexpected)[:5])
    wrap_fp16_model(model)
    model.cuda().eval()
    return cfg, model


def run_frame(model, head, sample, token, stash_gt=None):
    """One forward pass; returns the raw head outputs."""
    meta = dict(sample["img_metas"].data)
    meta["scene_token"] = token
    inp = {k: sample[k] for k in INPUT_KEYS if k in sample}
    inp["img_metas"] = DataContainer(meta, cpu_only=True)
    batch = to_device(collate([inp], samples_per_gpu=1), "cuda")

    captured = {}
    orig_forward = head.forward

    def wrapped(*a, _og=orig_forward, **k):
        out = _og(*a, **k)
        captured["outs"] = out
        return out

    head.forward = wrapped
    try:
        with torch.no_grad():
            model(batch, return_loss=False, rescale=True)
    finally:
        head.forward = orig_forward
    outs = captured["outs"]
    if stash_gt is not None:
        outs["_viz_gt_bboxes"], outs["_viz_gt_labels"] = stash_gt
    return outs


def match_900(head, outs):
    """Replicate _get_target_single_box_only on the l0 pre-sift predictions.

    Returns (pos_inds, pos_gt_inds, query_ref[900,2], query_code[900,10]).
    """
    cls_scores = outs["agent_pre_cls_scores"][0].float()
    bbox_preds = outs["agent_pre_coord_preds"][0].float()
    gt_bboxes = outs["_viz_gt_bboxes"].float()
    gt_labels = outs["_viz_gt_labels"].long()
    assign_result = head.assigner.assign(bbox_preds, cls_scores, gt_bboxes, gt_labels, None)
    sampling_result = head.sampler.sample(assign_result, bbox_preds, gt_bboxes)
    return (sampling_result.pos_inds.detach().cpu().numpy(),
            sampling_result.pos_assigned_gt_inds.detach().cpu().numpy(),
            head.agent_reference_points.weight[..., :2].detach().cpu().float().numpy(),
            bbox_preds.detach().cpu().numpy())


def gt_of(sample):
    boxes = sample["gt_bboxes_3d"].data
    rows = torch.cat((boxes.gravity_center, boxes.tensor[:, 3:]), dim=1).float()
    return boxes, rows, sample["gt_labels_3d"].data.long()


def cameras_of(sample, cfg):
    meta = dict(sample["img_metas"].data)
    img = sample["img"].data
    if not torch.is_tensor(img):
        return None, [], None
    ncfg = meta.get("img_norm_cfg") or cfg.img_norm_cfg
    mean, std = np.array(ncfg["mean"]), np.array(ncfg["std"])
    imgs = [np.clip(x * std + mean, 0, 255).astype(np.uint8)
            for x in img.numpy().transpose(0, 2, 3, 1)]
    names = [f.split("/")[-2] for f in meta.get("filename", [])]
    lidar2img = np.asarray(sample["lidar2img"].data, dtype=np.float64)
    return imgs, names, lidar2img


def map_of(sample, index):
    try:
        md = sample["map_gt_bboxes_3d"].data
        return md.fixed_num_sampled_points_torch.numpy(), sample["map_gt_labels_3d"].data.numpy()
    except Exception as exc:  # noqa: BLE001 - map GT is optional for the plot
        print(f"[{index}] map GT unavailable ({exc}), lane lines skipped")
        return None, None


def ego_size(sample):
    """(x_size, y_size) of ego from ego_lcf_feat = (vx,vy,ax,ay,w,length,width,vel,steer)."""
    try:
        feat = np.asarray(sample["ego_lcf_feat"].data).ravel()
        length, width = float(feat[5]), float(feat[6])
        if 1.0 < length < 20.0 and 0.5 < width < 5.0:
            return width, length
    except Exception:  # noqa: BLE001 - fall back to the nominal size
        pass
    return EGO_SIZE


def _prev_exists(sample):
    v = sample["prev_exists"]
    v = v.data if hasattr(v, "data") else v
    return bool(np.asarray(v).ravel()[0])


def frame_sequence(ds, index, warmup):
    """The target sample preceded by up to `warmup` contiguous same-scene ones."""
    target = ds[index]
    token = dict(target["img_metas"].data)["scene_token"]
    seq, j = [], index - 1
    while len(seq) < warmup and j >= 0:
        prev = ds[j]
        if dict(prev["img_metas"].data)["scene_token"] != token:
            break
        seq.append(prev)
        if not _prev_exists(prev):
            break
        j -= 1
    return list(reversed(seq)) + [target], token


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def draw(gt_rows, gt_labels, pos_inds, pos_gt_inds, query_ref, query_code,
         map_pts, map_labels, cams, cam_names, lidar2img, ego_wl,
         class_names, map_classes, pc_range, title, out_path, cam_boxes=True):
    n_gt = len(gt_rows)
    pair_cmap = plt.get_cmap("tab20")     # one colour per matched GT
    cls_cmap = plt.get_cmap("tab10")      # one colour per class
    gt_color = {g: pair_cmap(int(g) % 20) for g in pos_gt_inds}

    x0, y0, x1, y1 = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    n_side = int(round(np.sqrt(len(query_ref))))
    cell_w = (x1 - x0) / (n_side - 1) * 0.45
    cell_h = (y1 - y0) / (n_side - 1) * 0.45
    xlim = (x0 - 2.0, x1 + 2.0)
    ylim = (y0 - 2.0, y1 + 2.0)

    # ---- figure skeleton: camera strip on top, two BEV panels + legend below
    fig_w = 16.0
    cam_aspect = (cams[0].shape[0] / cams[0].shape[1]) if cams else 0.36
    cam_h = fig_w / 6.0 * cam_aspect
    bev_h = 11.0
    fig_h = cam_h + bev_h + 1.1
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=13, y=1 - 0.15 / fig_h)

    top_hi = 1 - 0.60 / fig_h
    top_lo = top_hi - cam_h / fig_h
    gs_bot = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.36],
                      left=0.05, right=0.99, top=top_lo - 0.55 / fig_h,
                      bottom=0.35 / fig_h, wspace=0.16)
    axL = fig.add_subplot(gs_bot[0, 0])
    axR = fig.add_subplot(gs_bot[0, 1])
    axLeg = fig.add_subplot(gs_bot[0, 2])
    axLeg.axis("off")

    # ---- left: the 900 query anchors -------------------------------------- #
    for qx, qy in query_ref:
        axL.add_patch(plt.Rectangle((qx - cell_w, qy - cell_h), 2 * cell_w, 2 * cell_h,
                                    facecolor="0.92", edgecolor="0.75", lw=0.3, zorder=1))
    axL.set_xlim(*xlim)
    axL.set_ylim(*ylim)
    axL.set_aspect("equal")
    axL.set_title(f"agent queries ({len(query_ref)} = {n_side}x{n_side} anchors, "
                  f"x {x0:g}..{x1:g} m, y {y0:g}..{y1:g} m)\nfilled = matched to a GT",
                  fontsize=10)
    axL.set_xlabel("x / right (m)")
    axL.set_ylabel("y / forward (m)")

    # ---- right: map lane lines -------------------------------------------- #
    map_used = set()
    if map_pts is not None and len(map_pts) > 0:
        for pts, lb in zip(map_pts, map_labels):
            name = map_classes[int(lb)]
            if name in MAP_SKIP or name not in MAP_STYLE:
                continue
            color, ls, lw = MAP_STYLE[name]
            axR.plot(pts[:, 0], pts[:, 1], color=color, ls=ls, lw=lw, alpha=0.85, zorder=0.5)
            map_used.add(name)

    # ---- right: every predicted centre, then the GT agents ----------------- #
    axR.scatter(query_code[:, 0], query_code[:, 1], s=2.5, c="lightsteelblue",
                zorder=1, linewidths=0)

    cls_used, dot_used = set(), False
    for g in range(n_gt):
        gx, gy, _, sx, sy, _, yaw = gt_rows[g, :7]
        color = cls_cmap(int(gt_labels[g]) % 10)
        edge = gt_color.get(g)
        cls_used.add(int(gt_labels[g]))
        if max(sx, sy) < DOT_SIZE:  # pedestrian / cone / sign: too small for a box
            axR.plot(gx, gy, "o", ms=8, mfc=color,
                     mec=edge if edge is not None else "black",
                     mew=1.6 if edge is not None else 0.8, zorder=3)
            dot_used = True
            continue
        axR.add_patch(Polygon(bev_poly(gx, gy, sx, sy, yaw), closed=True, facecolor=color,
                              alpha=0.5, edgecolor=edge if edge is not None else "black",
                              lw=1.6 if edge is not None else 0.8, zorder=2))
        nose = heading_vec(yaw) * (sy / 2.0)  # centre -> front face, stays inside the box
        axR.plot([gx, gx + nose[0]], [gy, gy + nose[1]], color="0.1", lw=1.0, zorder=3)

    # ---- right: ego (origin of its own frame, so it faces +y by definition) - #
    axR.add_patch(Polygon(bev_poly(0.0, 0.0, ego_wl[0], ego_wl[1], 0.0), closed=True,
                          facecolor="red", edgecolor="darkred", lw=1.2, alpha=0.9, zorder=5))
    axR.plot([0, 0], [ego_wl[1] / 2.0, ego_wl[1] / 2.0 + 1.0], color="darkred", lw=1.4, zorder=5)
    axR.set_xlim(*xlim)
    axR.set_ylim(*ylim)
    axR.set_aspect("equal")
    axR.set_title("BEV, ego frame (y up = forward, x right)\n"
                  "filled = GT, dashed = matched query prediction", fontsize=10)
    axR.set_xlabel("x / right (m)")
    axR.set_ylabel("y / forward (m)")

    # ---- matched pairs: highlight, connect, draw the predicted box --------- #
    for qi, gi in zip(pos_inds, pos_gt_inds):
        color = gt_color[gi]
        qx, qy = query_ref[qi]
        axL.add_patch(plt.Rectangle((qx - cell_w, qy - cell_h), 2 * cell_w, 2 * cell_h,
                                    facecolor=color, edgecolor="black", lw=0.6, zorder=3))
        axL.text(qx, qy, str(gi), ha="center", va="center", fontsize=5, zorder=4)

        gx, gy = gt_rows[gi, 0], gt_rows[gi, 1]
        axR.text(gx, gy + 1.4, f"{gi}:{class_names[gt_labels[gi]][:5]}", fontsize=6,
                 ha="center", color=color, zorder=6)

        px, py, _, psx, psy, _, pyaw = decode_pred(query_code[qi])
        axR.add_patch(Polygon(bev_poly(px, py, psx, psy, pyaw), closed=True, facecolor="none",
                              edgecolor=color, lw=1.1, ls="--", zorder=4))
        fig.add_artist(ConnectionPatch(xyA=(qx, qy), coordsA=axL.transData, axesA=axL,
                                       xyB=(gx, gy), coordsB=axR.transData, axesB=axR,
                                       color=color, lw=0.7, alpha=0.8, zorder=2))

    # ---- legend ------------------------------------------------------------ #
    handles = [Patch(facecolor=cls_cmap(c % 10), alpha=0.5, edgecolor="black",
                     label=f"GT {class_names[c]}") for c in sorted(cls_used)]
    if dot_used:
        handles.append(Line2D([], [], marker="o", ls="", mfc="0.5", mec="black",
                              label=f"GT smaller than {DOT_SIZE:g} m"))
    handles += [
        Patch(facecolor="red", label="ego"),
        Line2D([], [], color="0.35", ls="--", label="matched query pred box"),
        Line2D([], [], marker="o", color="lightsteelblue", ls="", ms=4,
               label=f"all {len(query_code)} query pred centres"),
        Line2D([], [], color="0.1", lw=1.0, label="GT heading (centre -> front)"),
    ]
    handles += [Line2D([], [], color=MAP_STYLE[m][0], ls=MAP_STYLE[m][1],
                       lw=MAP_STYLE[m][2], label=f"map: {m}")
                for m in map_classes if m in map_used]
    axLeg.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)

    # ---- camera strip, with the same boxes projected back ------------------ #
    if cams:
        gs_top = GridSpec(1, 6, figure=fig, left=0.03, right=0.99,
                          top=top_hi, bottom=top_lo, wspace=0.02)
        pred_rows = {gi: decode_pred(query_code[qi]) for qi, gi in zip(pos_inds, pos_gt_inds)}
        for j, im in enumerate(cams):
            axc = fig.add_subplot(gs_top[0, j])
            axc.imshow(im)
            axc.set_title(cam_names[j] if j < len(cam_names) else f"cam{j}", fontsize=8)
            axc.axis("off")
            if not cam_boxes or lidar2img is None:
                continue
            h, w = im.shape[:2]
            for g in range(n_gt):
                uv = project(box_corners_3d(gt_rows[g]), lidar2img[j], w, h)
                if uv is None:
                    continue
                color = cls_cmap(int(gt_labels[g]) % 10)
                for a, b in BOX_EDGES:
                    axc.plot(uv[[a, b], 0], uv[[a, b], 1], color=color, lw=0.9, alpha=0.95)
            for gi, row in pred_rows.items():
                uv = project(box_corners_3d(row), lidar2img[j], w, h)
                if uv is None:
                    continue
                for a, b in BOX_EDGES:
                    axc.plot(uv[[a, b], 0], uv[[a, b], 1], color=gt_color[gi],
                             lw=0.8, ls="--", alpha=0.9)
            axc.set_xlim(0, w)
            axc.set_ylim(h, 0)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def verify(cfg, model, ds, args):
    """Numeric audit of every convention this script relies on."""
    head = model.pts_bbox_head
    print("\n================ verification ================")
    print(f"class_names ({len(cfg.class_names)}): {cfg.class_names}")
    print(f"map_classes ({len(cfg.map_classes)}): {cfg.map_classes}")
    print(f"pc_range: {list(cfg.point_cloud_range)}")

    ref = head.agent_reference_points.weight[..., :2].detach().cpu().float().numpy()
    n_side = int(round(np.sqrt(len(ref))))
    xs = np.linspace(cfg.point_cloud_range[0], cfg.point_cloud_range[3], n_side)
    ys = np.linspace(cfg.point_cloud_range[1], cfg.point_cloud_range[4], n_side)
    grid = np.stack([np.tile(xs, n_side), np.repeat(ys, n_side)], axis=1)
    print(f"query anchors: {ref.shape}, unique x={len(np.unique(ref[:, 0]))}, "
          f"unique y={len(np.unique(ref[:, 1]))}, "
          f"max |anchor - linspace grid| = {np.abs(ref - grid).max():.4g}")

    # footprint helper vs. the box class' own corners
    worst_poly = 0.0
    dyaw, dlog_sx, dlog_sy, dxy, dz = [], [], [], [], []
    rnd_dxy, rnd_dyaw = [], []
    rng = np.random.default_rng(0)
    done = 0
    idx = 0
    step = max(1, args.stride // 4)
    while done < args.verify_frames and idx < len(ds):
        cur, idx = idx, idx + step
        sample = ds[cur]
        boxes, rows, labels = gt_of(sample)
        if len(rows) < 2:
            continue
        mine = np.stack([bev_poly(r[0], r[1], r[3], r[4], r[6]) for r in rows.numpy()])
        theirs = boxes.corners.numpy()[:, [0, 4, 7, 3], :2]  # bottom face, same winding
        worst_poly = max(worst_poly, float(np.abs(mine - theirs).max()))

        seq, token = frame_sequence(ds, cur, args.warmup)
        gt = (rows.cuda(), labels.cuda())
        for k, s in enumerate(seq):
            outs = run_frame(model, head, s, f"{token}#verify{done}",
                             stash_gt=gt if k == len(seq) - 1 else None)
        pos_inds, pos_gt_inds, _, code = match_900(head, outs)
        gt_np = rows.numpy()
        for qi, gi in zip(pos_inds, pos_gt_inds):
            p = decode_pred(code[qi])
            g = gt_np[gi]
            dxy.append(np.hypot(p[0] - g[0], p[1] - g[1]))
            dz.append(p[2] - g[2])
            dlog_sx.append(np.log(p[3]) - np.log(max(g[3], 1e-3)))
            dlog_sy.append(np.log(p[4]) - np.log(max(g[4], 1e-3)))
            dyaw.append(abs(wrap_pi(p[6] - g[6])))
            r = decode_pred(code[rng.integers(len(code))])
            rnd_dxy.append(np.hypot(r[0] - g[0], r[1] - g[1]))
            rnd_dyaw.append(abs(wrap_pi(r[6] - g[6])))
        done += 1

    print(f"\nbev_poly() vs LiDARInstance3DBoxes.corners: max abs diff = {worst_poly:.3e} m")

    def stat(name, v, unit=""):
        v = np.asarray(v)
        print(f"  {name:<22s} n={len(v):4d}  median {np.median(v):+.3f}{unit}  "
              f"mean {v.mean():+.3f}{unit}  p90 {np.percentile(np.abs(v), 90):.3f}{unit}")

    print(f"\nmatched l0 pre-sift prediction vs its GT ({done} frames):")
    stat("centre distance", dxy, " m")
    stat("z error", dz, " m")
    stat("log x_size error", dlog_sx)
    stat("log y_size error", dlog_sy)
    stat("|yaw error|", dyaw, " rad")
    print("  random query -> same GT (control, decoding-independent baseline):")
    stat("  centre distance", rnd_dxy, " m")
    stat("  |yaw error|", rnd_dyaw, " rad")

    # yaw convention vs. the annotated velocities
    yaw, angv = [], []
    for i in range(0, min(len(ds), 3000), 61):
        _, rows, _ = gt_of(ds[i])
        r = rows.numpy()
        if len(r) == 0:
            continue
        moving = np.hypot(r[:, 7], r[:, 8]) > 2.0
        yaw.append(r[moving, 6])
        angv.append(np.arctan2(r[moving, 8], r[moving, 7]))
    yaw, angv = np.concatenate(yaw), np.concatenate(angv)
    err = np.abs(wrap_pi(angv - heading_angle(yaw)))
    print(f"\nheading convention on {len(yaw)} moving agents: "
          f"|angle(vx,vy) - heading(yaw)| median {np.median(err):.2e} rad, "
          f"max {err.max():.2e} rad  (0 => -yaw-pi/2 is the driving direction)")
    print("==============================================\n")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py")
    p.add_argument("--ckpt", default="ckpts/drivetransformer_large.pth")
    p.add_argument("--out-dir", default="ol_eval_results/query_gt_viz")
    p.add_argument("--num", type=int, default=4, help="number of frames to render")
    p.add_argument("--frames", type=int, nargs="*", default=None,
                   help="explicit dataset indices to render (overrides the scan)")
    p.add_argument("--min-gt", type=int, default=5)
    p.add_argument("--max-gt", type=int, default=40)
    p.add_argument("--scan", type=int, default=400, help="dataset samples to scan for candidates")
    p.add_argument("--stride", type=int, default=100, help="dataset-index jump between rendered frames")
    p.add_argument("--warmup", type=int, default=4,
                   help="preceding same-scene frames replayed to fill the temporal memory")
    p.add_argument("--no-cam-boxes", action="store_true", help="do not project boxes onto the cameras")
    p.add_argument("--verify", action="store_true", help="run the numeric audit first")
    p.add_argument("--verify-frames", type=int, default=12)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg, model = build(args.config, args.ckpt)
    head = model.pts_bbox_head
    ds = build_dataset(cfg.data.val)
    print(f"val samples: {len(ds)}")
    class_names = list(cfg.class_names)
    map_classes = list(cfg.map_classes)
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]

    if args.verify:
        verify(cfg, model, ds, args)

    if args.frames:
        candidates = list(args.frames)
    else:
        candidates = []
        i = 0
        while i < min(args.scan, len(ds)) and len(candidates) < args.num:
            boxes = ds[i]["gt_bboxes_3d"].data
            n = len(boxes.tensor) if boxes is not None else 0
            if args.min_gt <= n <= args.max_gt:
                candidates.append(i)
                i += args.stride  # jump ahead for scene variety
            else:
                i += 1

    rendered = 0
    for i in candidates:
        seq, token = frame_sequence(ds, i, args.warmup)
        sample = seq[-1]
        boxes, rows, labels = gt_of(sample)
        if len(rows) == 0:
            print(f"[{i}] no GT, skipped")
            continue

        gt = (rows.cuda(), labels.cuda())
        for k, s in enumerate(seq):
            outs = run_frame(model, head, s, f"{token}#viz{i}",
                             stash_gt=gt if k == len(seq) - 1 else None)
        pos_inds, pos_gt_inds, query_ref, query_code = match_900(head, outs)

        map_pts, map_labels = map_of(sample, i)
        cams, cam_names, lidar2img = cameras_of(sample, cfg)
        out_path = os.path.join(args.out_dir, f"frame_{i:05d}.png")
        title = (f"{tag}   agent query <-> GT matching   frame {i}   "
                 f"({len(rows)} GT, {len(pos_inds)} matched, {len(seq) - 1} warm-up frames)")
        draw(rows.numpy(), labels.numpy(), pos_inds, pos_gt_inds, query_ref, query_code,
             map_pts, map_labels, cams, cam_names, lidar2img, ego_size(sample),
             class_names, map_classes, list(cfg.point_cloud_range), title, out_path,
             cam_boxes=not args.no_cam_boxes)
        print(f"[{i}] gt={len(rows)} matched={len(pos_inds)} -> {out_path}", flush=True)
        rendered += 1
    print(f"done, {rendered} frames saved to {args.out_dir}")


if __name__ == "__main__":
    main()
