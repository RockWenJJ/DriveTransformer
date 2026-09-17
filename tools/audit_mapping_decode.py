"""Compare map decoders on identical frozen network tensors, without training.

Samples the middle frame of each validation clip and resets memory per frame.
This is a cold-start diagnostic, NOT the complete chronological benchmark.
"""
import argparse
from collections import defaultdict
import importlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmcv import Config
from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.parallel import DataContainer
from mmcv.parallel.collate import collate
from mmcv.utils import wrap_fp16_model
from drivetransformer_contracts import apply_attention_eval_policy
from tools.ol_eval_tasks_v2 import INPUT_KEYS, MAP_CLASSES, to_device, rasterize, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    cfg = Config.fromfile(args.config)
    importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    state = torch.load(args.ckpt, map_location='cpu')['state_dict']
    state = {k.replace('model.', '').replace('._orig_mod', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    del state
    wrap_fp16_model(model)
    model.cuda().eval()
    apply_attention_eval_policy(model, False)
    ds = build_dataset(cfg.data.val)
    assert getattr(ds, 'map_clip_mode', 'legacy') == 'legacy'
    assert not getattr(ds, 'map_deduplicate', False)
    from adzoo.drivetransformer.mmdet3d_plugin.core.bbox.coders.map_nms_free_coder import MapNMSFreeCoder
    decoder_args = dict(cfg.model.pts_bbox_head.map_bbox_coder)
    decoder_args.pop('type')
    modes = [('flat50', False, 50), ('unique50', True, 50), ('unique100', True, 100)]
    decoders = {name: MapNMSFreeCoder(**dict(decoder_args, max_num=k, unique_queries=u, score_threshold=None))
                for name, u, k in modes}
    captured = {}

    def capture(module, inputs, output):
        for name in ['map_all_cls_scores', 'map_all_bbox_preds', 'map_all_pts_preds']:
            # get_bboxes casts to float32 on the GPU. Keep the same device:
            # CPU/GPU top-k may order tied logits differently.
            captured[name] = output[name][-1, 0].detach().float()

    hook = model.pts_bbox_head.register_forward_hook(capture)
    starts = [0] + list(ds.divide_nums[:-1])
    indices = [(int(s) + int(e) - 1) // 2 for s, e in zip(starts, ds.divide_nums)]
    accum = defaultdict(list)
    rows = []
    examples = []
    began = time.time()
    for step, idx in enumerate(indices):
        d = ds[idx]
        vectors = d['map_gt_bboxes_3d'].data
        labels = d['map_gt_labels_3d'].data.numpy()
        gt = vectors.fixed_num_sampled_points.numpy() if len(labels) else np.zeros((0, 20, 2))
        meta = dict(d['img_metas'].data)
        meta['scene_token'] = 'mapping-audit-%d' % idx
        model.pts_bbox_head.reset_memory()
        inputs = {k: d[k] for k in INPUT_KEYS if k in d}
        inputs['img_metas'] = DataContainer(meta, cpu_only=True)
        batch = to_device(collate([inputs], samples_per_gpu=1), 'cuda')
        with torch.no_grad():
            original = model(batch, return_loss=False, rescale=True)[0]
        decoded = {name: dec.decode_single(captured['map_all_cls_scores'], captured['map_all_bbox_preds'],
                                           captured['map_all_pts_preds']) for name, dec in decoders.items()}
        decoded = {name: {key: value.cpu() for key, value in pred.items()} for name, pred in decoded.items()}
        # The reference decoder must reproduce the normal model output exactly.
        for src, dst in [('map_pts', 'map_pts_3d'), ('map_labels', 'map_labels_3d'), ('map_scores', 'map_scores_3d')]:
            torch.testing.assert_close(decoded['flat50'][src], original[dst].cpu(), rtol=0, atol=0)
        np.savez_compressed(args.out / ('frame_%06d.npz' % idx), **{k: v.cpu().numpy() for k, v in captured.items()},
                            gt_pts=gt, gt_labels=labels)
        gt_masks = [rasterize(gt, labels, c) for c in range(6)]
        for name, pred in decoded.items():
            for threshold in [0., .1, .3, .5]:
                keep = pred['map_scores'].numpy() >= threshold
                pts = pred['map_pts'].numpy()[keep]
                lab = pred['map_labels'].numpy()[keep]
                scores = pred['map_scores'].numpy()[keep]
                ious, present, pp, rr = [], [], [], []
                for c, gm in enumerate(gt_masks):
                    pm = rasterize(pts, lab, c)
                    inter = int((gm & pm).sum())
                    union = int((gm | pm).sum())
                    iou = inter / union if union else 1.
                    ious.append(iou)
                    if gm.any():
                        present.append(iou)
                        pp.append(inter / int(pm.sum()) if pm.any() else 0.)
                        rr.append(inter / int(gm.sum()))
                key = '%s@%.1f' % (name, threshold)
                row = dict(index=idx, clip=ds.routes_names[step], mode=key, count=len(pts),
                           macro_iou=float(np.mean(ious)), gt_present_iou=float(np.mean(present)),
                           gt_present_precision=float(np.mean(pp)), gt_present_recall=float(np.mean(rr)),
                           class_iou=ious)
                accum[key].append(row)
                rows.append(row)
                if step == 0 and key in ['flat50@0.0', 'unique100@0.3']:
                    examples.append(dict(mode=key, points=pts.tolist(), labels=lab.tolist(), scores=scores.tolist()))
        print('frame', step+1, '/', len(indices), 'elapsed', round(time.time()-began, 1), flush=True)
    hook.remove()
    summary = {k: {metric: float(np.mean([r[metric] for r in values]))
                  for metric in ['count', 'macro_iou', 'gt_present_iou', 'gt_present_precision', 'gt_present_recall']}
               for k, values in accum.items()}
    result = dict(protocol='50 clip midpoints, independent cold starts, one network forward per frame, shared tensors',
                  seed=0, frames=len(indices), indices=indices, config=str(Path(args.config).resolve()),
                  config_sha256=file_sha256(args.config), checkpoint=str(Path(args.ckpt).resolve()),
                  checkpoint_sha256=file_sha256(args.ckpt), checkpoint_load='strict',
                  default_decoder_exact_match=True, original_gt_unchanged=True,
                  summary=summary, rows=rows, examples=examples,
                  caveat='Threshold sweep is diagnostic; no held-out confirmation, full benchmark, or retraining gain.')
    (args.out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
