"""Versioned custom task proxies; not official nuScenes AP/collision metrics.

v2 reads actual dataset classes, matches detections in score order, counts empty
scenes and agent future masks for center collision, restricts motion matching to
same classes, and diagnoses fixed-distance heading and controller target speed.
Historical numerical results from v1 must not be mixed with this version.

Open-loop per-task capability evaluation for DriveTransformer on Bench2Drive val.

Uses an input whitelist and resets memory at clip boundaries by default.
The legacy-window option retains historical subsequence resets.

Verified field layouts (probed on val samples, see session log):
  - gt_bboxes_3d:        LiDARInstance3DBoxes (N,9) [x,y,z,w,l,h,yaw,vx,vy], ego frame,
                         +x=right / +y=forward, box long axis along yaw
  - gt_labels_3d:        (N,), class ordering read from the dataset
  - gt_attr_labels:      (N,34); cols 0:12 = future traj 6 steps @0.5s in agent frame,
                         stored as R@traj with R=rot(yaw+pi) -> invert with R^T;
                         cols 12:18 = per-step valid flags
  - map_gt_bboxes_3d:    LiDARInstanceLines; .fixed_num_sampled_points (M,20,2) ego BEV
  - map_gt_labels_3d:    (M,) 6 classes [Broken,Solid,SolidSolid,Center,TrafficLight,StopSign]
  - pred (forward_test): boxes_3d (LiDARInstance3DBoxes), scores_3d, labels_3d,
                         trajs_3d (N,fut_mode,6*2) flattened ego frame -> (N,6,6,2),
                         map_pts_3d (K,20,2) ego BEV + map_labels_3d/map_scores_3d,
                         ego_fut_preds_fix_time [1,1,30,2]

Metrics:
  planning : L2@1/2/3s + center-distance collision proxy with valid futures
  detection: custom center-distance AP @ 0.5/1/2/4m averaged -> mAP,
             plus P/R @2m; greedy per-class matching, 41-pt PR interpolation
  motion   : for GT-matched agents (T=2m), pred (6 modes,6,2) vs GT (6,2):
             minADE / minFDE over modes, averaged over valid steps/agents
  mapping  : per-class rasterized IoU of BEV polylines (0.25m/px, thickness 3px
             ~0.75m), averaged over samples & classes

Usage:
  CUDA_VISIBLE_DEVICES=0 python tools/ol_eval_tasks_v2.py --config <cfg> --ckpt <pth> \
      --out ol_eval_results/xxx.json [--limit N]
"""
import argparse
import importlib
import json
import os
import time
import sys
import hashlib
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivetransformer_contracts import apply_attention_eval_policy, decode_fixed_distance, target_speed_from_time_path
from tools.eval_contracts import score_ordered_match, same_class_match, center_collision_proxy

import numpy as np
import torch

torch.set_float32_matmul_precision("high")
from mmcv import Config
from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.parallel import DataContainer
from mmcv.parallel.collate import collate
from mmcv.utils import wrap_fp16_model

INPUT_KEYS = [
    "img", "ego_his_trajs", "ego_lcf_feat", "ego_fut_cmd",
    "lidar2img", "cam_intrinsic", "timestamp", "ego_pose", "ego_pose_inv", "pad_shape",
]

HORIZONS = [("1s", 9, 1), ("2s", 19, 3), ("3s", 29, 5)]  # (name, ego_idx, agent_j)
COLLISION_DIST = 2.0
MAP_CLASSES = ['Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight', 'StopSign']
DET_MATCH_DISTS = [0.5, 1.0, 2.0, 4.0]
MOTION_MATCH_DIST = 2.0
RASTER_RES = 0.25          # m / pixel
RASTER_THICK = 3           # px (~0.75 m)
RASTER_XY = (-16.0, -32.0, 16.0, 32.0)  # xmin,ymin,xmax,ymax


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(8 * 2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def to_device(obj, device):
    if torch.is_tensor(obj):
        obj = obj.to(device)
        if obj.dtype == torch.float64:
            obj = obj.float()
        return obj
    if isinstance(obj, list):
        return [to_device(o, device) for o in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def agent_future_ego(boxes, attr):
    """GT agent futures (6,2) in ego frame from attr cols 0:12 (R@traj, invert R^T)."""
    bt = boxes.tensor.numpy()
    fut = attr[:, :12].reshape(-1, 6, 2)
    if torch.is_tensor(fut):
        fut = fut.numpy()
    yaw = bt[:, 6]
    t = yaw + np.pi
    c, s = np.cos(t), np.sin(t)
    x = c[:, None] * fut[..., 0] + s[:, None] * fut[..., 1]
    y = -s[:, None] * fut[..., 0] + c[:, None] * fut[..., 1]
    return np.stack([x, y], axis=-1) + bt[:, None, :2]   # (N,6,2)


def greedy_match(gt_pts, pred_pts, thresh):
    """Greedy nearest matching. Returns list of (gi, pi) with dist<=thresh."""
    pairs = []
    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return pairs
    d = np.linalg.norm(gt_pts[:, None, :] - pred_pts[None, :, :], axis=-1)
    order = np.argsort(d, axis=None)
    used_g, used_p = set(), set()
    for flat in order:
        gi, pi = divmod(int(flat), d.shape[1])
        if d[gi, pi] > thresh:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi))
    return pairs


class DetAccumulator:
    """Per (class, threshold) detection score/tp lists + gt counts."""

    def __init__(self, classes):
        self.classes = tuple(classes)
        self.data = {c: {t: {"dets": [], "n_gt": 0} for t in DET_MATCH_DISTS}
                     for c in self.classes}

    def add(self, gt_centers, gt_labels, pred_centers, pred_labels, pred_scores):
        for cls_i, cls in enumerate(self.classes):
            g = gt_centers[gt_labels == cls_i] if len(gt_labels) else np.zeros((0, 2))
            p = pred_centers[pred_labels == cls_i] if len(pred_labels) else np.zeros((0, 2))
            s = pred_scores[pred_labels == cls_i] if len(pred_labels) else np.zeros(0)
            for t in DET_MATCH_DISTS:
                slot = self.data[cls][t]
                slot["n_gt"] += len(g)
                pairs = score_ordered_match(g, p, s, t)
                matched_p = {pi for _, pi in pairs}
                for pi in range(len(p)):
                    slot["dets"].append((float(s[pi]), 1.0 if pi in matched_p else 0.0))

    def ap(self, dets, n_gt):
        if n_gt == 0:
            return None
        if not dets:
            return 0.0
        dets = sorted(dets, key=lambda x: -x[0])
        tp = np.array([d[1] for d in dets])
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(1 - tp)
        rec = cum_tp / n_gt
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        # 41-pt interpolation over recall grid
        rec_grid = np.linspace(0.0, 1.0, 41)
        prec_ip = np.zeros_like(rec_grid)
        for i, r in enumerate(rec_grid):
            m = rec >= r
            prec_ip[i] = prec[m].max() if m.any() else 0.0
        return float(prec_ip[1:].mean())  # Custom convention: drop recall=0.

    def summarize(self):
        out, aps = {}, []
        per_t = {t: [] for t in DET_MATCH_DISTS}
        for cls in self.classes:
            for t in DET_MATCH_DISTS:
                slot = self.data[cls][t]
                ap = self.ap(slot["dets"], slot["n_gt"])
                if ap is not None:
                    out[f"ap_{cls}@{t}m"] = round(ap, 4)
                    per_t[t].append(ap)
        for t in DET_MATCH_DISTS:
            if per_t[t]:
                out[f"mAP@{t}m"] = round(float(np.mean(per_t[t])), 4)
                aps.append(np.mean(per_t[t]))
        out["mAP"] = round(float(np.mean(aps)), 4) if aps else None
        # simple P/R at 2 m (pooled over classes)
        tp = fp = ngt = 0
        for cls in self.classes:
            slot = self.data[cls][2.0]
            dets = slot["dets"]
            tp += sum(d[1] for d in dets)
            fp += sum(1 - d[1] for d in dets)
            ngt += slot["n_gt"]
        out["precision@2m"] = round(tp / max(tp + fp, 1), 4) if ngt else None
        out["recall@2m"] = round(tp / max(ngt, 1), 4) if ngt else None
        out["n_gt_agents"] = ngt
        return out


def rasterize(polylines, cls_labels, cls_i, xy=RASTER_XY, res=RASTER_RES, thick=RASTER_THICK):
    """Boolean BEV mask for polylines of one class."""
    import cv2
    x0, y0, x1, y1 = xy
    w, h = int((x1 - x0) / res), int((y1 - y0) / res)
    img = np.zeros((h, w), np.uint8)
    sel = np.where(cls_labels == cls_i)[0]
    if len(sel) == 0:
        return img.astype(bool)
    for k in sel:
        pts = polylines[k]
        px = ((pts[:, 0] - x0) / res).astype(np.int32)
        py = ((y1 - pts[:, 1]) / res).astype(np.int32)  # flip y: +y=forward -> up
        m = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if m.sum() < 2:
            continue
        cv2.polylines(img, [np.stack([px[m], py[m]], 1)], False, 1, thick)
    return img.astype(bool)


def map_iou_per_class(gt_pts, gt_lab, pred_pts, pred_lab):
    out = {}
    for cls_i, cls in enumerate(MAP_CLASSES):
        g = rasterize(gt_pts, gt_lab, cls_i)
        p = rasterize(pred_pts, pred_lab, cls_i)
        if not g.any() and not p.any():
            continue  # class absent in both
        inter = (g & p).sum()
        union = (g | p).sum()
        out[cls] = float(inter / union) if union > 0 else 1.0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stream-mode", choices=["clip", "legacy-window"], default="clip")
    p.add_argument("--map-score-threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow-partial-checkpoint", action="store_true")
    args = p.parse_args()

    if Path(args.out).exists() and not args.overwrite:
        raise FileExistsError("Refusing to overwrite an existing evaluation: " + args.out)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cfg = Config.fromfile(args.config)
    if getattr(cfg, "plugin", False):
        mod = os.path.dirname(cfg.plugin_dir).replace("/", ".")
        importlib.import_module(mod)

    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    ckpt = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    new_state = {}
    for k, v in ckpt.items():
        new_state[k.replace("model.", "").replace("._orig_mod", "")] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"loaded ckpt: {len(new_state)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    if (missing or unexpected) and not args.allow_partial_checkpoint:
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    wrap_fp16_model(model)
    model.cuda()
    model.eval()
    contract = dict(cfg.get("inference_contract", {}))
    apply_attention_eval_policy(model, contract.get("legacy_eval_dropout", False))

    val_cfg = cfg.data.val.copy()
    if args.stream_mode == "clip":
        val_cfg["sub_seq_lenth"] = -1
    ds = build_dataset(val_cfg)
    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    print(f"val samples: {n}")

    l2 = {h: [] for h, _, _ in HORIZONS}
    coll = {h: [] for h, _, _ in HORIZONS}
    det_acc = DetAccumulator(ds.CLASSES)
    fixed_dist_errors = {k: [] for k in (2, 5, 10, 20)}
    speed_errors = []
    decode_out_of_range = []
    motion_valid_gt = 0
    ade_all, fde_all = [], []
    iou_acc = {c: [] for c in MAP_CLASSES}
    window = 0
    prev_token = None
    t0 = time.time()

    for i in range(n):
        d = ds[i]
        gt_traj = d["ego_fut_trajs_fix_time"].astype(np.float64)
        mask = d["ego_fut_masks_fix_time"].astype(np.float64)
        boxes = d["gt_bboxes_3d"].data
        attr = d["gt_attr_labels"].data
        labels = d["gt_labels_3d"].data.numpy() if torch.is_tensor(d["gt_labels_3d"].data) \
            else np.asarray(d["gt_labels_3d"].data)
        map_gt = d["map_gt_bboxes_3d"].data
        map_gt_pts = map_gt.fixed_num_sampled_points.numpy() \
            if torch.is_tensor(map_gt.fixed_num_sampled_points) else np.asarray(map_gt.fixed_num_sampled_points)
        map_gt_lab = d["map_gt_labels_3d"].data.numpy()

        meta = dict(d["img_metas"].data)
        token = meta["scene_token"]
        if prev_token is None or token != prev_token or not d["prev_exists"]:
            window += 1
        prev_token = token
        meta["scene_token"] = f"{token}#w{window}"

        inp = {k: d[k] for k in INPUT_KEYS if k in d}
        inp["img_metas"] = DataContainer(meta, cpu_only=True)
        batch = to_device(collate([inp], samples_per_gpu=1), "cuda")

        with torch.no_grad():
            out = model(batch, return_loss=False, rescale=True)[0]

        # ---------- planning ----------
        pred = out["ego_fut_preds_fix_time"][0, 0].numpy()  # [30,2] ego frame
        gt_centers = boxes.tensor.numpy()[:, :2] if boxes is not None and len(boxes) > 0 \
            else np.zeros((0, 2))
        fut_valid = (attr[:, 12:18].numpy() > 0.5) if attr is not None and len(attr) else None
        agent_fut = agent_future_ego(boxes, attr) if len(gt_centers) else None

        for hname, eidx, aj in HORIZONS:
            if mask[eidx] == 1:
                l2[hname].append(float(np.linalg.norm(pred[eidx] - gt_traj[eidx])))
                coll[hname].append(center_collision_proxy(pred[eidx], agent_fut, fut_valid, aj, COLLISION_DIST))

        # The steering branch is distinct from the speed branch.
        fixed_values = out["ego_fut_preds_fix_dist"][0, 0, :, 0].numpy()
        fixed_gt = np.asarray(d["ego_fut_trajs_fix_dist"]).reshape(-1)
        fixed_mask = np.asarray(d["ego_fut_masks_fix_dist"]).reshape(-1).astype(bool)
        decoded, diagnostic = decode_fixed_distance(fixed_values, contract.get("fixed_dist_decoder", "sin_heading"))
        pred_heading = np.arctan2(decoded[:, 1], decoded[:, 0])
        # Ground truth sin targets have a forward-half-plane ambiguity.
        gt_heading = np.arcsin(np.clip(fixed_gt, -1, 1))
        decode_out_of_range.append(diagnostic["out_of_range_fraction"])
        for distance in fixed_dist_errors:
            j = distance - 1
            if j < len(fixed_gt) and fixed_mask[j]:
                fixed_dist_errors[distance].append(float(abs(pred_heading[j] - gt_heading[j]) * 180 / np.pi))
        if mask[5] and mask[10]:
            speed_errors.append(abs(target_speed_from_time_path(pred) - target_speed_from_time_path(gt_traj)))

        # ---------- detection + motion ----------
        pb = out["boxes_3d"]
        pred_centers = pb.tensor.numpy()[:, :2] if pb is not None and len(pb) > 0 else np.zeros((0, 2))
        pred_labels = out["labels_3d"].numpy() if torch.is_tensor(out["labels_3d"]) else np.asarray(out["labels_3d"])
        pred_scores = out["scores_3d"].numpy() if torch.is_tensor(out["scores_3d"]) else np.asarray(out["scores_3d"])
        det_acc.add(gt_centers, labels, pred_centers, pred_labels, pred_scores)

        motion_valid_gt += int(fut_valid.any(axis=-1).sum()) if fut_valid is not None else 0
        if len(gt_centers) and len(pred_centers):
            for gi, pi in same_class_match(gt_centers, labels, pred_centers, pred_labels, pred_scores, MOTION_MATCH_DIST):
                gt_f = agent_fut[gi]                       # (6,2) ego
                pv = fut_valid[gi] if fut_valid is not None else np.ones(6, bool)
                tr = out["trajs_3d"][pi]                   # (fut_mode*12,)
                tr = tr.view(-1, 6, 2).numpy() if torch.is_tensor(tr) else tr.reshape(-1, 6, 2)
                if pv.sum() < 1:
                    continue
                # trajs_3d is agent-centered (rotate-only, no translation in
                # get_bboxes) -> add the predicted box center for ego frame
                tr = tr + pred_centers[pi]
                mode_ade = np.linalg.norm(tr[:, pv, :] - gt_f[None, pv, :], axis=-1).mean(1)
                mode_fde = np.linalg.norm(tr[:, pv, :] - gt_f[None, pv, :], axis=-1)[:, -1]
                ade_all.append(float(mode_ade.min()))
                fde_all.append(float(mode_fde.min()))

        # ---------- mapping ----------
        mp = out["map_pts_3d"]
        pred_map_pts = mp.numpy() if torch.is_tensor(mp) and mp.dim() == 3 else None
        if pred_map_pts is not None:
            pred_map_lab = out["map_labels_3d"].numpy()
            keep = out["map_scores_3d"].numpy() >= args.map_score_threshold
            pred_map_pts, pred_map_lab = pred_map_pts[keep], pred_map_lab[keep]
            for c, v in map_iou_per_class(map_gt_pts, map_gt_lab,
                                          pred_map_pts, pred_map_lab).items():
                iou_acc[c].append(v)

        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"[{i+1}/{n}] {el:.0f}s det_mAP={det_acc.summarize().get('mAP')} "
                  f"ADE={np.mean(ade_all) if ade_all else float('nan'):.3f} "
                  f"IoU_mean={np.mean([np.mean(v) for v in iou_acc.values() if v]) if any(iou_acc.values()) else 'NA'}",
                  flush=True)

    res = {"config": args.config, "ckpt": args.ckpt, "n_samples": n, "metrics": {},
           "protocol": {"version": "task-proxy-v2", "classes": list(ds.CLASSES),
                        "stream_mode": args.stream_mode, "seed": args.seed,
                        "map_score_threshold": args.map_score_threshold,
                        "inference_contract": contract,
                        "dataset_timestamp_frame_period_s": ds.timestamp_frame_period_s,
                        "dataset_ego_accel_mode": ds.ego_accel_mode,
                        "checkpoint_sha256": file_sha256(args.ckpt),
                        "torch_version": torch.__version__,
                        "contract_sha256": file_sha256(Path(__file__).resolve().parents[1] / 'drivetransformer_contracts.py'),
                        "metric_helpers_sha256": file_sha256(Path(__file__).with_name('eval_contracts.py')),
                        "resolved_config_sha256": hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
                        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "notes": "Custom AP; center-distance collision proxy; fixed-distance heading assumes forward half-plane. Not official nuScenes metrics."}}

    m = res["metrics"]
    l2_means = [np.mean(l2[h]) for h, _, _ in HORIZONS if l2[h]]
    for hname, _, _ in HORIZONS:
        m[f"plan/l2_{hname}"] = float(np.mean(l2[hname])) if l2[hname] else None
        m[f"plan/center_collision_proxy_{hname}"] = float(np.mean(coll[hname])) if coll[hname] else None
    m["plan/l2_avg"] = float(np.mean(l2_means)) if l2_means else None
    coll_means = [np.mean(coll[h]) for h, _, _ in HORIZONS if coll[h]]
    m["plan/center_collision_proxy_avg"] = float(np.mean(coll_means)) if coll_means else None

    for distance, errors in fixed_dist_errors.items():
        m[f"plan/fixdist_heading_mae_deg@{distance}m"] = float(np.mean(errors)) if errors else None
        m[f"plan/fixdist_count@{distance}m"] = len(errors)
    m["plan/target_speed_mae_mps"] = float(np.mean(speed_errors)) if speed_errors else None
    m["plan/fixdist_out_of_range_fraction"] = float(np.mean(decode_out_of_range)) if decode_out_of_range else None
    m["motion/valid_gt"] = motion_valid_gt
    m["motion/matched_coverage"] = len(ade_all) / motion_valid_gt if motion_valid_gt else None
    for h, _, _ in HORIZONS:
        m[f"plan/valid_ego_frames_{h}"] = len(coll[h])
    for k, v in det_acc.summarize().items():
        m[f"det/{k}"] = v
    m["motion/minADE"] = float(np.mean(ade_all)) if ade_all else None
    m["motion/minFDE"] = float(np.mean(fde_all)) if fde_all else None
    m["motion/n_matched"] = len(ade_all)
    for c in MAP_CLASSES:
        if iou_acc[c]:
            m[f"map/iou_{c}"] = float(np.mean(iou_acc[c]))
    ious = [np.mean(v) for v in iou_acc.values() if v]
    m["map/iou_avg"] = float(np.mean(ious)) if ious else None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["metrics"], indent=1))
    print(f"saved -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
