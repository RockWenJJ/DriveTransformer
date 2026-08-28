"""Shared plumbing for the DriveTransformer visualisation scripts.

Everything that encodes a *convention* lives here exactly once, so the four
viewers (query/GT matching, motion, map, planning) cannot drift apart.

Conventions, all of them checked by the --verify mode of the scripts
--------------------------------------------------------------------
BEV frame            +x = right, +y = forward (ego/lidar frame of the frame).

Box row              (x, y, z, x_size, y_size, z_size, yaw, vx, vy), the layout
                     the head documents at drivetransformer_head.py:1691.
                     mmdet3d rotates the box local axes by *minus* yaw
                     (rotation_3d_in_axis uses the transposed matrix), so the
                     x_size edge points along -yaw and the y_size edge -- the
                     long one, 4.8 m vs 1.9 m for a car -- along pi/2 - yaw.
                     The repo agrees: TrajPreprocess turns the stored yaw into a
                     "normal yaw" with -yaw - pi/2, and that is precisely where
                     (vx, vy) points for every moving agent.

Predicted box code   normalize_bbox()'s 10-code
                     (x, y, log x_size, log y_size, z, log z_size,
                      sin yaw, cos yaw, vx, vy).

Agent motion         all_traj_preds is [layer, B, query, mode, fut_ts * 2] in
                     the *agent's own* frame and *cumulative* (TrajPreprocess
                     cumsums the differential GT before it becomes the target).
                     head.rotate_agent_trajs_to_ego rotates it back by
                     -yaw - pi; the result is a displacement from the agent
                     centre. fut_ts = 6 points, 0.5 s apart.

Map                  map_all_pts_preds is [layer, B, query, pts, 2] already in
                     metric ego coordinates -- MapNMSFreeCoder has the
                     denormalisation commented out and returns them as-is.

Planning             ego_fut_preds_fix_time is [layer, B, mode, 30, 2], absolute
                     ego-frame positions 0.1 s apart (the config sets
                     ego_fut_offset_input=False, so the GT is not cumsummed
                     either). fut_ego_mode = 1, so there is a single mode and
                     ego_traj_cls_scores is None.
                     ego_fut_preds_fix_dist is [layer, B, mode, 20, 1] holding
                     x / |(x, y)| -- the sine of the waypoint bearing off the
                     +y axis -- sampled every 1 m of path length.
"""
import importlib
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon

from mmcv import Config
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
AGENT_DT = 0.5         # seconds between agent future waypoints (sample_interval 5 @ 10 Hz)
EGO_DT = 0.1           # seconds between ego fix-time waypoints
CARLA_COMMANDS = ["LEFT", "RIGHT", "STRAIGHT", "LANEFOLLOW",
                  "CHANGELANELEFT", "CHANGELANERIGHT"]

BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def bev_poly(cx, cy, size_x, size_y, yaw):
    """Footprint (4x2, counter-clockwise) of a box row in the BEV frame."""
    ux = np.array([np.cos(yaw), -np.sin(yaw)]) * (size_x / 2.0)
    uy = np.array([np.sin(yaw), np.cos(yaw)]) * (size_y / 2.0)
    c = np.array([cx, cy], dtype=float)
    return np.stack([c - ux - uy, c + ux - uy, c + ux + uy, c - ux + uy])


def heading_vec(yaw):
    """Unit vector the agent drives towards: opposite the local +y axis."""
    return np.array([-np.sin(yaw), -np.cos(yaw)])


def heading_angle(yaw):
    """The 'normal yaw' the repo uses in TrajPreprocess: -yaw - pi/2."""
    return np.arctan2(-np.cos(yaw), -np.sin(yaw))


def box_corners_3d(row):
    """8x3 corners of (x, y, z_centre, size_x, size_y, size_z, yaw, ...)."""
    cx, cy, cz, sx, sy, sz, yaw = row[:7]
    base = bev_poly(cx, cy, sx, sy, yaw)
    bottom = np.concatenate([base, np.full((4, 1), cz - sz / 2.0)], axis=1)
    top = np.concatenate([base, np.full((4, 1), cz + sz / 2.0)], axis=1)
    return np.concatenate([bottom, top], axis=0)


def project(pts3d, lidar2img, width, height, min_depth=0.5, require_all=True):
    """Project 3D points to one camera; None if behind or fully off-frame."""
    pts3d = np.asarray(pts3d, dtype=np.float64)
    hom = np.concatenate([pts3d, np.ones((len(pts3d), 1))], axis=1)
    cam = hom @ np.asarray(lidar2img).T
    front = cam[:, 2] > min_depth
    if require_all and not front.all():
        return None
    if not front.any():
        return None
    uv = np.full((len(pts3d), 2), np.nan)
    uv[front] = cam[front, :2] / cam[front, 2:3]
    vis = uv[front]
    if (np.nanmax(vis[:, 0]) < 0 or np.nanmin(vis[:, 0]) > width
            or np.nanmax(vis[:, 1]) < 0 or np.nanmin(vis[:, 1]) > height):
        return None
    return uv


def decode_box_code(code):
    """(x, y, z, size_x, size_y, size_z, yaw, vx, vy) from a 10-code.

    Same arithmetic as mmcv.core.bbox.util.denormalize_bbox, with the size
    exponentials clipped so a garbage query cannot blow up the axes.
    """
    code = np.asarray(code, dtype=np.float64)
    out = np.empty(code.shape[:-1] + (9,))
    out[..., 0] = code[..., 0]
    out[..., 1] = code[..., 1]
    out[..., 2] = code[..., 4]
    out[..., 3] = np.exp(np.clip(code[..., 2], -3.0, 3.0))
    out[..., 4] = np.exp(np.clip(code[..., 3], -3.0, 3.0))
    out[..., 5] = np.exp(np.clip(code[..., 5], -3.0, 3.0))
    out[..., 6] = np.arctan2(code[..., 6], code[..., 7])
    out[..., 7] = code[..., 8]
    out[..., 8] = code[..., 9]
    return out


def agent_traj_to_ego(trajs, yaw):
    """Agent-self-frame waypoints -> displacement in the ego frame.

    Mirrors head.rotate_agent_trajs_to_ego: the heading is -yaw - pi/2 and the
    trajectory is rotated by heading - pi/2, i.e. by -yaw - pi.
    """
    ang = -yaw - np.pi
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s], [s, c]])
    return np.asarray(trajs) @ rot.T


def gt_agent_trajs(sample, fut_ts=6):
    """GT agent futures -> (ego-frame waypoints [N, T, 2], valid mask [N, T]).

    gt_attr_labels[:, :2T] is already cumulative and in the agent's own frame
    (TrajPreprocess did both), so only the rotation back is needed. The
    waypoints are absolute ego-frame positions, agent centre included.
    """
    attr = np.asarray(sample["gt_attr_labels"].data, dtype=np.float64)
    rows = gt_of(sample)[1].numpy()
    trajs = attr[:, :fut_ts * 2].reshape(-1, fut_ts, 2)
    masks = attr[:, fut_ts * 2:fut_ts * 3] > 0
    out = np.stack([agent_traj_to_ego(trajs[i], rows[i, 6]) + rows[i, :2]
                    for i in range(len(rows))]) if len(rows) else np.zeros((0, fut_ts, 2))
    return out, masks


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


def run_sequence(model, head, seq, token):
    """Replay warm-up frames so the streaming memory is valid, return last outs."""
    outs = None
    for s in seq:
        outs = run_frame(model, head, s, token)
    return outs


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


def map_of(sample):
    try:
        md = sample["map_gt_bboxes_3d"].data
        return md.fixed_num_sampled_points_torch.numpy(), sample["map_gt_labels_3d"].data.numpy()
    except Exception as exc:  # noqa: BLE001 - map GT is optional for the plot
        print(f"  map GT unavailable ({exc})")
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


def ego_state(sample):
    """(speed m/s, far command, near command) for the title line."""
    feat = np.asarray(sample["ego_lcf_feat"].data).ravel()
    speed = float(np.hypot(feat[0], feat[1]))
    cmd = np.asarray(sample["ego_fut_cmd"].data).ravel()
    far = CARLA_COMMANDS[int(np.argmax(cmd[0:6]))] if cmd.size >= 76 else "?"
    near = CARLA_COMMANDS[int(np.argmax(cmd[70:76]))] if cmd.size >= 76 else "?"
    return speed, far, near


def ego_history(sample):
    """Past ego positions in the current ego frame, oldest first (offsets undone)."""
    off = np.asarray(sample["ego_his_trajs"].data, dtype=np.float64).reshape(-1, 2)
    pos = np.zeros((len(off) + 1, 2))
    for j in range(len(off) - 1, -1, -1):
        pos[j] = pos[j + 1] - off[j]
    return pos


def ground_z(gt_rows, default=-1.8, plausible=(-2.6, -1.0)):
    """Road height in the lidar frame, from the bottom face of the GT vehicles.

    Only boxes 1-3 m tall are used, which keeps out the traffic lights and signs
    that hang well above the road, and the result is rejected if it lands
    outside the range a roof-mounted lidar can plausibly sit at.
    """
    if len(gt_rows) == 0:
        return default
    cars = gt_rows[(gt_rows[:, 5] > 1.0) & (gt_rows[:, 5] < 3.0)]
    if len(cars) == 0:
        return default
    z = float(np.median(cars[:, 2] - cars[:, 5] / 2.0))
    return z if plausible[0] <= z <= plausible[1] else default


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


def verify_indices(ds, n):
    """Evenly spaced indices, so an audit covers many scenes instead of one."""
    return np.linspace(0, len(ds) - 1, num=max(n, 1), dtype=int)


def pick_frames(ds, args):
    """Explicit --frames, or scan for frames with a reasonable number of agents."""
    if args.frames:
        return list(args.frames)
    picked, i = [], 0
    while i < min(args.scan, len(ds)) and len(picked) < args.num:
        boxes = ds[i]["gt_bboxes_3d"].data
        n = len(boxes.tensor) if boxes is not None else 0
        if args.min_gt <= n <= args.max_gt:
            picked.append(i)
            i += args.stride  # jump ahead for scene variety
        else:
            i += 1
    return picked


def add_common_args(p, out_dir):
    p.add_argument("--config", default="adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py")
    p.add_argument("--ckpt", default="ckpts/drivetransformer_large.pth")
    p.add_argument("--out-dir", default=out_dir)
    p.add_argument("--num", type=int, default=4, help="number of frames to render")
    p.add_argument("--frames", type=int, nargs="*", default=None,
                   help="explicit dataset indices to render (overrides the scan)")
    p.add_argument("--min-gt", type=int, default=5)
    p.add_argument("--max-gt", type=int, default=40)
    p.add_argument("--scan", type=int, default=400, help="dataset samples to scan for candidates")
    p.add_argument("--stride", type=int, default=100, help="dataset-index jump between rendered frames")
    p.add_argument("--warmup", type=int, default=4,
                   help="preceding same-scene frames replayed to fill the temporal memory")
    p.add_argument("--no-cam-boxes", action="store_true", help="do not draw on the camera strip")
    p.add_argument("--verify", action="store_true", help="run the numeric audit first")
    p.add_argument("--verify-frames", type=int, default=12)
    return p


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def make_figure(cams, widths, title, fig_w=16.0, bev_h=11.0, legend_width=0.36):
    """Camera strip on top, `widths` panels plus a legend column below."""
    cam_aspect = (cams[0].shape[0] / cams[0].shape[1]) if cams else 0.36
    cam_h = fig_w / 6.0 * cam_aspect
    fig_h = cam_h + bev_h + 1.1
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=13, y=1 - 0.15 / fig_h)

    top_hi = 1 - 0.60 / fig_h
    top_lo = top_hi - cam_h / fig_h
    gs_bot = GridSpec(1, len(widths) + 1, figure=fig,
                      width_ratios=list(widths) + [legend_width],
                      left=0.05, right=0.99, top=top_lo - 0.55 / fig_h,
                      bottom=0.35 / fig_h, wspace=0.16)
    axes = [fig.add_subplot(gs_bot[0, i]) for i in range(len(widths))]
    ax_legend = fig.add_subplot(gs_bot[0, len(widths)])
    ax_legend.axis("off")

    cam_axes = []
    if cams:
        gs_top = GridSpec(1, 6, figure=fig, left=0.03, right=0.99,
                          top=top_hi, bottom=top_lo, wspace=0.02)
        for j in range(len(cams)):
            cam_axes.append(fig.add_subplot(gs_top[0, j]))
    return fig, cam_axes, axes, ax_legend


def draw_camera_strip(cam_axes, cams, cam_names):
    for j, (ax, im) in enumerate(zip(cam_axes, cams)):
        ax.imshow(im)
        ax.set_title(cam_names[j] if j < len(cam_names) else f"cam{j}", fontsize=8)
        ax.axis("off")
        ax.set_xlim(0, im.shape[1])
        ax.set_ylim(im.shape[0], 0)


def bev_axis(ax, pc_range, title, margin=2.0):
    ax.set_xlim(pc_range[0] - margin, pc_range[3] + margin)
    ax.set_ylim(pc_range[1] - margin, pc_range[4] + margin)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x / right (m)")
    ax.set_ylabel("y / forward (m)")


def draw_map(ax, map_pts, map_labels, map_classes, alpha=0.85, zorder=0.5):
    """Returns the set of map class names actually drawn."""
    used = set()
    if map_pts is None or len(map_pts) == 0:
        return used
    for pts, lb in zip(map_pts, map_labels):
        name = map_classes[int(lb)]
        if name in MAP_SKIP or name not in MAP_STYLE:
            continue
        color, ls, lw = MAP_STYLE[name]
        ax.plot(pts[:, 0], pts[:, 1], color=color, ls=ls, lw=lw, alpha=alpha, zorder=zorder)
        used.add(name)
    return used


def draw_ego(ax, wl, zorder=5):
    """Ego sits at the origin of its own frame, so it faces +y by definition."""
    ax.add_patch(Polygon(bev_poly(0.0, 0.0, wl[0], wl[1], 0.0), closed=True,
                         facecolor="red", edgecolor="darkred", lw=1.2, alpha=0.9, zorder=zorder))
    ax.plot([0, 0], [wl[1] / 2.0, wl[1] / 2.0 + 1.0], color="darkred", lw=1.4, zorder=zorder)


def draw_boxes(ax, rows, labels, cmap, alpha=0.5, edgecolor="black", lw=0.8,
               zorder=2, heading=True, dots=True):
    """Filled GT-style boxes coloured by class. Returns (classes, any_dot)."""
    used, dot_used = set(), False
    for row, lab in zip(rows, labels):
        gx, gy, _, sx, sy, _, yaw = row[:7]
        color = cmap(int(lab) % 10)
        used.add(int(lab))
        if dots and max(sx, sy) < DOT_SIZE:
            ax.plot(gx, gy, "o", ms=8, mfc=color, mec=edgecolor, mew=0.8, zorder=zorder + 1)
            dot_used = True
            continue
        ax.add_patch(Polygon(bev_poly(gx, gy, sx, sy, yaw), closed=True, facecolor=color,
                             alpha=alpha, edgecolor=edgecolor, lw=lw, zorder=zorder))
        if heading:
            nose = heading_vec(yaw) * (sy / 2.0)  # centre -> front, stays inside the box
            ax.plot([gx, gx + nose[0]], [gy, gy + nose[1]], color="0.1", lw=1.0,
                    zorder=zorder + 1)
    return used, dot_used


def draw_box_on_cameras(cam_axes, cams, lidar2img, row, color, lw=0.9, ls="-", alpha=0.95):
    for j, ax in enumerate(cam_axes):
        h, w = cams[j].shape[:2]
        uv = project(box_corners_3d(row), lidar2img[j], w, h)
        if uv is None:
            continue
        for a, b in BOX_EDGES:
            ax.plot(uv[[a, b], 0], uv[[a, b], 1], color=color, lw=lw, ls=ls, alpha=alpha)


def draw_polyline_on_cameras(cam_axes, cams, lidar2img, pts2d, z, color,
                             lw=1.2, ls="-", alpha=0.9, marker=None, ms=2.0):
    pts3d = np.concatenate([np.asarray(pts2d, dtype=np.float64),
                            np.full((len(pts2d), 1), z)], axis=1)
    for j, ax in enumerate(cam_axes):
        h, w = cams[j].shape[:2]
        uv = project(pts3d, lidar2img[j], w, h, require_all=False)
        if uv is None:
            continue
        ax.plot(uv[:, 0], uv[:, 1], color=color, lw=lw, ls=ls, alpha=alpha)
        if marker is not None:
            ax.plot(uv[:, 0], uv[:, 1], ls="", marker=marker, ms=ms, mfc="white",
                    mec=color, mew=0.5, alpha=alpha)
