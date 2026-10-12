#!/usr/bin/env python3
"""CPU contracts, real-image GPU readiness, and strict checkpoint acceptance.

Readiness is not a driving-quality result. GPU validation uses the unchanged
full-resolution six-camera dataset pipeline, while the separately labelled
small-planner optimization check uses synthetic context and real GT geometry.
No checkpoint produced by these checks is a formal-training initializer.
"""
import argparse
import ast
import copy
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASELINE = ROOT / 'adzoo/drivetransformer/configs/drivetransformer/drivetransformer_res480_p1_map_datafix_10ep.py'
FEATURE_KEYS = {'img', 'img_metas', 'ego_his_trajs', 'ego_fut_cmd', 'ego_lcf_feat',
                'lidar2img', 'cam_intrinsic', 'cam_extrinsic', 'timestamp',
                'ego_pose', 'ego_pose_inv', 'pad_shape', 'prev_exists'}
NEW_TARGET_KEYS = {'hip_plan_targets', 'hip_plan_masks'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_receipt(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def finite_tree(value):
    import torch
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def equal_value(left, right, path=''):
    import numpy as np
    import torch
    if torch.is_tensor(left):
        require(torch.equal(left, right), 'Legacy tensor changed: ' + path)
    elif hasattr(left, 'tensor'):
        equal_value(left.tensor, right.tensor, path)
    elif isinstance(left, np.ndarray):
        require(left.shape == right.shape and left.dtype == right.dtype,
                'Legacy array shape/dtype changed: ' + path)
        same = np.array_equal(left, right, equal_nan=True) if left.dtype.kind in 'fc' else np.array_equal(left, right)
        require(same, 'Legacy array changed: ' + path)
    elif isinstance(left, dict):
        require(set(left) == set(right), 'Legacy keys changed: ' + path)
        for key in left:
            equal_value(left[key], right[key], path + '/' + str(key))
    elif isinstance(left, (list, tuple)):
        require(len(left) == len(right), 'Legacy sequence changed: ' + path)
        for index, (a, b) in enumerate(zip(left, right)):
            equal_value(a, b, path + '/' + str(index))
    else:
        require(left == right or (isinstance(left, float) and math.isnan(left) and math.isnan(right)),
                'Legacy scalar changed: ' + path)


def setup(config):
    import numpy as np
    import torch
    from mmcv import Config
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    cfg = Config.fromfile(str(config))
    importlib.import_module(cfg.get('plugin_dir', 'adzoo/drivetransformer/mmdet3d_plugin/').strip('/').replace('/', '.'))
    return cfg


def build_model(cfg, initialize=True):
    from mmcv.models import build_model as build
    model = build(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    if initialize:
        model.init_weights()
    require(getattr(model.pts_bbox_head, 'hip_planner', None) is not None, 'HiP planner is missing')
    require(getattr(model.pts_bbox_head, 'with_mapping', None) is False, 'Hybrid must disable mapping')
    forbidden = ('map_query', 'map_reference_points', 'map_prep_decoder', 'map_cls_branches', 'map_reg_branches')
    names = list(model.pts_bbox_head.named_parameters())
    for prefix in forbidden:
        require(not any(name.startswith(prefix + '.') or name == prefix for name, _ in names),
                'Mapping parameters remain: ' + prefix)
    return model


def compare_detection_motion(cfg):
    from mmcv import Config
    base = Config.fromfile(str(BASELINE))
    for key in ('img_backbone', 'img_neck', 'pretrained'):
        equal_value(base.model.get(key), cfg.model.get(key), 'model/' + key)
    old, new = base.model.pts_bbox_head, cfg.model.pts_bbox_head
    checked = []
    for key in old:
        if (key.startswith('agent_') or key in ('embed_dims', 'fut_ts', 'fut_mode', 'num_classes',
                'code_size', 'code_weights', 'bbox_coder', 'loss_cls', 'loss_bbox', 'loss_traj',
                'loss_traj_cls', 'motion_cls_valid_only', 'memory_len_frame')):
            equal_value(old[key], new[key], 'head/' + key)
            checked.append(key)
    if 'train_cfg' in old and old.train_cfg:
        equal_value(old.train_cfg['assigner'], new.train_cfg['assigner'], 'head/train_cfg/assigner')
    require(checked, 'No detection/motion configuration checked')
    return {'baseline': str(BASELINE), 'baseline_sha256': sha256(BASELINE), 'unchanged_head_keys': checked,
            'mapping': 'explicitly removed per user request; shared representations can consequently change'}


def check_target_isolation():
    path = ROOT / 'adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer.py'
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'forward_train')
    names = {arg.arg for arg in method.args.args}
    require(NEW_TARGET_KEYS <= names, 'HiP GT must be explicit loss-only arguments, not feature kwargs')
    allowed = set()
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'loss':
            for keyword in node.keywords:
                if keyword.arg in NEW_TARGET_KEYS:
                    allowed.update(id(n) for n in ast.walk(keyword.value) if isinstance(n, ast.Name))
    for node in ast.walk(method):
        if isinstance(node, ast.Name) and node.id in NEW_TARGET_KEYS:
            require(id(node) in allowed, 'Future GT used outside the loss call: ' + node.id)
    return {'ast_loss_only': True, 'feature_allowlist_excludes_future_targets': True}


def run_contract_suites(out):
    patterns = ('test_hip_planner.py', 'test_hip_targets.py', 'test_hip_controller.py',
                'test_hip_training_job.py', 'test_dt_hip_nomap.py')
    results = []
    for pattern in patterns:
        log = out.parent / (out.stem + '.' + pattern + '.log')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT))
        with log.open('x') as stream:
            code = subprocess.call([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests',
                                    '-p', pattern, '-v'], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        results.append({'pattern': pattern, 'returncode': code, 'log': str(log), 'sha256': sha256(log)})
        require(code == 0, 'Contract suite failed: %s; see %s' % (pattern, log))
    return results


def real_dataset(cfg, sample_count=4):
    import numpy as np
    from mmcv.datasets import build_dataset
    from mmcv.parallel import collate
    dataset = build_dataset(cfg.data.train)
    require(dataset.hip_planning_targets is True, 'HiP target generation not enabled')
    require(getattr(dataset, 'with_mapping', None) is False, 'Training dataset must not load map labels')
    indices = sorted(set([0, min(30, len(dataset)-1), len(dataset)//2, len(dataset)-1]))[:sample_count]
    audits = []
    for index in indices:
        dataset.hip_planning_targets = False
        baseline = copy.deepcopy(dataset.get_data_info(index))
        dataset.hip_planning_targets = True
        actual = dataset.get_data_info(index)
        targets, masks = actual['hip_plan_targets'], actual['hip_plan_masks']
        require(targets.shape == (4, 6, 2) and masks.shape == (4, 6), 'Real target shape mismatch')
        require(np.isfinite(targets).all() and np.isfinite(masks).all(), 'Nonfinite real target')
        require(np.isin(masks, [0, 1]).all(), 'Invalid target mask')
        equal_value(baseline, {k: v for k, v in actual.items() if k not in NEW_TARGET_KEYS}, 'sample/%d' % index)
        require(len(actual['img_filename']) == 6 and all(Path(p).is_file() for p in actual['img_filename']),
                'Six real camera files are required')
        audits.append({'index': index, 'scene': actual['scene_token'], 'frame_idx': int(actual['frame_idx']),
                       'valid_targets_per_group': masks.sum(-1).astype(int).tolist(), 'legacy_gt_unchanged': True})
    sample, selected = None, None
    candidates = list(range(min(30, len(dataset)-1), min(94, len(dataset)), 4))
    candidates += np.linspace(0, len(dataset)-1, 32).astype(int).tolist()
    for index in dict.fromkeys(candidates):
        raw = dataset.get_data_info(index)
        if not raw['hip_plan_masks'].astype(bool).all():
            continue
        item = dataset.prepare_train_data(index, dataset.get_augmentation())
        if item is None:
            continue
        batch = collate([item], samples_per_gpu=1)
        if batch['hip_plan_masks'].bool().all():
            sample, selected = batch, index
            break
    require(sample is not None, 'No fully supervised real sample among fixed local and uniform candidate indices')
    require(not any(key.startswith('map_gt') for key in sample), 'Mapping GT remains in training inputs')
    require(tuple(sample['img'].shape) == (1, 1, 6, 3, 480, 960),
            'GPU gate requires true configured 6-camera 480x960 resolution: %s' % (sample['img'].shape,))
    require(bool(sample['img'].float().std() > 0), 'Image pipeline returned constant data')
    info = dataset.get_data_info(selected)
    details = {'size': len(dataset), 'sample_audits': audits, 'full_sample_index': selected,
               'full_sample_scene': info['scene_token'], 'image_shape': list(sample['img'].shape),
               'camera_files': [{'path': p, 'sha256': sha256(p)} for p in info['img_filename']],
               'target_shape': list(sample['hip_plan_targets'].shape),
               'mask_shape': list(sample['hip_plan_masks'].shape), 'mapping_gt_absent': True}
    return dataset, sample, details


def cpu_mode(cfg, args):
    import torch
    suites = run_contract_suites(args.out)
    config_check = compare_detection_motion(cfg)
    isolation = check_target_isolation()
    model = build_model(cfg)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location='cpu')
    model.load_state_dict(restored, strict=True)
    require(finite_tree(restored), 'Nonfinite constructed model')
    parameter_count = sum(p.numel() for p in model.parameters())
    del restored, buffer
    _, sample, dataset = real_dataset(cfg, args.sample_count)
    descent = small_planner_descent(type(model.pts_bbox_head.hip_planner),
                                    sample['hip_plan_targets'], sample['hip_plan_masks'])
    return dict(passed=True, finite=True, strict_load=True, suites=suites, config_check=config_check,
                target_isolation=isolation, real_data=dataset, parameters=parameter_count,
                small_planner_optimization=descent,
                small_planner_optimization=descent,
                complete_attention_forward=('Tiny real head/loss/backward passed using an explicit softmax '
                    'reference backend; full-resolution native CUDA attention awaits the GPU gate'),
                driving_quality_verified=False)


def to_device(value, device):
    import torch
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(to_device(v, device) for v in value)
    if hasattr(value, 'tensor') and hasattr(value, 'to'):
        return value.to(device)
    return value


def gradient_components(planner):
    groups = {'anchor': [planner.anchor], 'position_encoder': list(planner.position_encoder.parameters())}
    for index, module in enumerate(planner.refinements):
        for branch_index, branch in enumerate(module.regression):
            groups['refinement.%d.regression.%d' % (index, branch_index)] = list(branch.parameters())
        for name in ('mode_classifier', 'style_classifier'):
            groups['refinement.%d.%s' % (index, name)] = list(getattr(module, name).parameters())
    for index, module in enumerate(planner.image_attentions):
        for name in ('offset', 'camera_encoder', 'weights', 'output'):
            groups['image_attention.%d.%s' % (index, name)] = list(getattr(module, name).parameters())
    report = {}
    for name, parameters in groups.items():
        gradients = [p.grad for p in parameters if p.requires_grad and p.grad is not None]
        norm = sum(float(g.detach().float().abs().sum()) for g in gradients)
        report[name] = {'finite': finite_tree(gradients), 'l1': norm,
                        'with_gradient': len(gradients), 'parameters': len(parameters)}
    return report


def small_planner_descent(planner_type, targets, masks):
    import torch
    torch.manual_seed(4)
    planner = planner_type(16, 2, ROOT / 'data/anchors/hip', num_attention_groups=4).to(targets.device)
    context = torch.randn(1, 1, 16, device=targets.device)
    optimizer = torch.optim.Adam(planner.parameters(), lr=.003)
    history = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        query, anchors = planner.init_queries(context)
        geometry, modes, styles = [], [], []
        for layer in range(2):
            anchors, mode, style = planner.refine(layer, query, planner.position(anchors), anchors)
            geometry.append(anchors.reshape(1, 10, 48, 6, 2))
            modes.append(mode)
            styles.append(style)
        loss = sum(planner.loss(torch.stack(geometry), torch.stack(modes), torch.stack(styles), targets, masks).values())
        require(bool(torch.isfinite(loss)), 'Small-planner nonfinite objective')
        history.append(float(loss))
        loss.backward()
        optimizer.step()
    require(history[-1] < history[0], 'Small planner did not reduce its fixed-sample objective')
    return {'scope': 'Synthetic context, real sample targets; isolated optimization diagnostic',
            'steps': 30, 'initial_loss': history[0], 'final_loss': history[-1], 'loss_history': history}


def gpu_mode(cfg, args):
    import torch
    from mmcv.datasets import build_dataset
    from mmcv.parallel import collate
    from mmcv.utils import wrap_fp16_model
    require(torch.cuda.is_available(), 'CUDA unavailable; real GPU gate cannot be skipped')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    dataset, sample, audit = real_dataset(cfg, args.sample_count)
    model = build_model(cfg).to(device).train()
    # This repository's historical fp16 name enables BF16 autocast. Match the
    # training hooks rather than silently testing a different FP32 backend.
    wrap_fp16_model(model)
    batch = to_device(sample, device)
    if 'prev_exists' in batch:
        batch['prev_exists'] = torch.zeros_like(batch['prev_exists'])
    observed = []
    def check_head_inputs(module, arguments, keywords):
        require(not NEW_TARGET_KEYS.intersection(keywords), 'Future planning GT leaked into head forward')
        observed.append(sorted(keywords))
    handle = model.pts_bbox_head.register_forward_pre_hook(check_head_inputs, with_kwargs=True)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    losses, gradients = [], None
    for step in range(3):
        model.pts_bbox_head.reset_memory()
        optimizer.zero_grad(set_to_none=True)
        result = model(copy.deepcopy(batch), return_loss=True)
        require(finite_tree(result['log_vars']) and bool(torch.isfinite(result['loss'])), 'Nonfinite real-image training loss')
        require(not any('map' in key and 'loss' in key for key in result['log_vars']), 'Mapping loss remains enabled')
        require(any('loss_hip' in key for key in result['log_vars']), 'HiP loss is absent')
        result['loss'].backward()
        require(all(p.grad is None or finite_tree(p.grad) for p in model.parameters()), 'Nonfinite full-model gradients')
        gradients = gradient_components(model.pts_bbox_head.hip_planner)
        losses.append({k: float(v) for k, v in result['log_vars'].items()})
        if step < 2:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
    handle.remove()
    require(observed, 'Head feature-input isolation hook never executed')
    missing = [name for name, value in gradients.items() if not value['finite'] or value['l1'] <= 0]
    require(not missing, 'New components without finite nonzero gradients after 2 updates: ' + ', '.join(missing))
    model.eval()
    model.pts_bbox_head.reset_memory()
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg['ann_file'] = cfg.data.train.ann_file
    test_dataset = build_dataset(test_cfg)
    index = audit['full_sample_index']
    item = test_dataset.prepare_test_data(index, test_dataset.get_augmentation())
    inference = collate([item], samples_per_gpu=1)
    inference = to_device({key: value for key, value in inference.items() if key in FEATURE_KEYS}, device)
    with torch.no_grad():
        output = model(inference, return_loss=False)[0]
    native = output['hip_planning']
    native_shapes = {}
    for name in ('plan_temp_5hz', 'plan_spat_2m', 'plan_temp_2hz', 'plan_spat_5m',
                 'plan_speed_5hz', 'plan_speed_2hz'):
        require(tuple(native[name].shape[-2:]) == (6, 2) and finite_tree(native[name]), 'Invalid native six-point output: ' + name)
        native_shapes[name] = list(native[name].shape)
    compatible = output['ego_fut_preds_fix_time']
    require(tuple(compatible.shape[-2:]) == (30, 2) and finite_tree(compatible), 'Invalid 30-point compatibility output')
    source = native['plan_temp_2hz'].reshape(-1, 6, 2)
    require(torch.allclose(compatible.reshape(-1, 30, 2)[:, 4::5], source, atol=1e-5),
            'Compatibility output does not interpolate native 2Hz trajectory')
    descent = small_planner_descent(type(model.pts_bbox_head.hip_planner), batch['hip_plan_targets'], batch['hip_plan_masks'])
    return dict(passed=True, finite=True, new_components_grad_ok=True, real_data=audit,
                precision='repo auto_fp16 backend: torch.bfloat16 autocast; fp32 geometry/loss',
                gradient_scaler='not covered by SGD diagnostic; exercised by subsequent real DDP smoke',
                full_model_training_steps=3, diagnostic_updates_discarded=True, losses=losses,
                new_component_gradients=gradients, head_forward_kwargs=observed,
                native_output_shapes=native_shapes, compatibility_shape=list(compatible.shape),
                small_planner_optimization=descent, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                driving_quality_verified=False, batch20_ddp_verified=False)


def audit_training_logs(directory, expected_iterations, logging_interval):
    """Only native TextLogger JSONL files; do not interpret TensorBoard binaries."""
    files = sorted(Path(directory).glob('*.log.json'))
    require(files, 'Native training *.log.json evidence is missing')
    steps, records, sources = set(), 0, []
    for path in files:
        rows = 0
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, TypeError) as exc:
                    raise RuntimeError('Malformed native JSON log %s:%d' % (path, line_number)) from exc
                if not isinstance(row, dict) or row.get('mode') != 'train':
                    continue
                require('iter' in row and 'loss' in row and 'grad_norm' in row,
                        'Training JSON row lacks iter/loss/grad_norm: %s:%d' % (path, line_number))
                step = int(row['iter'])
                require(1 <= step <= expected_iterations, 'Training log iteration exceeds checkpoint budget')
                for key, value in row.items():
                    if 'loss' in key or key == 'grad_norm':
                        require(type(value) in (int, float) and math.isfinite(value),
                                'Nonfinite/nonnumeric %s in %s:%d' % (key, path, line_number))
                steps.add(step)
                records += 1
                rows += 1
        sources.append({'path': str(path.resolve()), 'sha256': sha256(path), 'train_rows': rows})
    require(steps, 'No actual training loss/gradient records found')
    if expected_iterations == 20:
        require(steps == set(range(1, 21)), 'DDP smoke logs must cover every optimizer step 1..20')
    else:
        require(max(steps) >= expected_iterations - max(1, logging_interval),
                'Formal training log ends too far before the final checkpoint')
    return {'finite_losses_and_grad_norm': True, 'training_rows': records,
            'unique_logged_iterations': len(steps), 'min_iteration': min(steps),
            'max_iteration': max(steps), 'smoke_every_step_covered': expected_iterations == 20,
            'sources': sources}


def checkpoint_mode(cfg, args):
    import torch
    require(args.checkpoint is not None and args.expected_iterations is not None, 'Checkpoint and expected iteration are required')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    require('state_dict' in checkpoint and 'optimizer' in checkpoint and checkpoint['optimizer'], 'Checkpoint model/optimizer missing')
    iteration = int(checkpoint['meta']['iter']) + 1
    require(iteration == args.expected_iterations, 'Checkpoint meta.iter + 1 differs from expected budget')
    require(finite_tree(checkpoint['state_dict']) and finite_tree(checkpoint['optimizer']), 'Nonfinite checkpoint model or optimizer')
    require(checkpoint['optimizer'].get('state'), 'Checkpoint optimizer has no initialized state')
    optimizer_steps = []
    for state in checkpoint['optimizer']['state'].values():
        if 'step' in state:
            step = state['step']
            step = float(step.item()) if torch.is_tensor(step) else float(step)
            require(math.isfinite(step) and step.is_integer(), 'Invalid optimizer step counter')
            optimizer_steps.append(int(step))
    require(optimizer_steps and max(optimizer_steps) == args.expected_iterations,
            'Actual optimizer max step differs from the fixed training budget')
    logs = audit_training_logs(args.checkpoint.resolve().parent, args.expected_iterations,
                               int(cfg.get('log_config', {}).get('interval', 20)))
    model = build_model(cfg, initialize=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    # The official training builder supplies parameter groups (backbone LR multiplier,
    # frozen parameters, weight decay overrides); use it when verifying restoration.
    from mmcv.optims import build_optimizer
    optimizer = build_optimizer(model, cfg.optimizer)
    optimizer.load_state_dict(checkpoint['optimizer'])
    require(finite_tree(optimizer.state_dict()), 'Restored optimizer is not finite')
    return dict(passed=True, finite=True, strict_load=True, optimizer_present=True,
                optimizer_finite=True, iteration=iteration, checkpoint=str(args.checkpoint.resolve()),
                checkpoint_sha256=sha256(args.checkpoint), optimizer_restored=True,
                optimizer_step_min=min(optimizer_steps), optimizer_step_max=max(optimizer_steps),
                optimizer_parameters_with_steps=len(optimizer_steps), training_log_audit=logs,
                parameters=sum(p.numel() for p in model.parameters()), driving_quality_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cpu', 'gpu', 'checkpoint'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--expected-iterations', type=int)
    parser.add_argument('--sample-count', type=int, default=4)
    args = parser.parse_args()
    require(1 <= args.sample_count <= 4, 'sample-count must be 1..4')
    args.config, args.out = args.config.resolve(), args.out.resolve()
    require(not args.out.exists(), 'Refusing to overwrite an existing validation receipt')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    started = time.time()
    receipt = {'mode': args.mode, 'config': str(args.config), 'config_sha256': sha256(args.config),
               'started_at': started, 'validator_sha256': sha256(__file__),
               'code_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    try:
        cfg = setup(args.config)
        receipt.update({'cpu': cpu_mode, 'gpu': gpu_mode, 'checkpoint': checkpoint_mode}[args.mode](cfg, args))
    except BaseException as exc:
        receipt.update(passed=False, finite=False, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(finished_at=time.time(), elapsed_seconds=time.time()-started)
        save_receipt(args.out, receipt)
        print(json.dumps({key: receipt.get(key) for key in ('mode', 'passed', 'finite', 'error', 'elapsed_seconds')},
                         ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
