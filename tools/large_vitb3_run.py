"""Explicit start / read-only preflight / status for the prepared experiment.

This script never starts training unless the `start` subcommand is supplied.
"""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large_vitb3_nomap_s16_res480_10ep.py'
DEFAULT_WORK = ROOT / 'work_dirs/large_vitb3_nomap_s16_res480_10ep'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def process_identity(pid):
    try:
        fields = Path('/proc/%s/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def get_config():
    sys.path.insert(0, str(ROOT))
    from mmcv import Config
    return Config.fromfile(str(CONFIG))


def gpu_inventory():
    raw = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    cards = {}
    for line in raw.strip().splitlines():
        index, uuid, used, total, util = [s.strip() for s in line.split(',')]
        cards[index] = dict(uuid=uuid, memory_used_mib=int(used),
            memory_total_mib=int(total), utilization_percent=int(util), pids=[])
    raw = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    by_uuid = {v['uuid']: v for v in cards.values()}
    for line in raw.strip().splitlines():
        uuid, pid = [s.strip() for s in line.split(',')]
        if uuid in by_uuid:
            by_uuid[uuid]['pids'].append(int(pid))
    return cards


def preflight(args):
    cfg = get_config()
    readiness = ROOT / 'preparation/readiness.json'
    if not readiness.exists():
        raise RuntimeError('Preparation signoff is missing')
    ready = json.loads(readiness.read_text())
    if ready.get('status') != 'ready':
        raise RuntimeError('Preparation is not marked ready')
    for rel, expected in ready['source_sha256'].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Source changed since verification: ' + rel)
    for rel in [cfg.data.train.ann_file, cfg.data.val.ann_file,
                cfg.model.img_backbone.checkpoint]:
        if not (ROOT / rel).is_file():
            raise FileNotFoundError(ROOT / rel)
    gpu_ids = args.gpus.split(',')
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError('This configuration requires four distinct GPUs')
    cards = gpu_inventory()
    selected = {i: cards[i] for i in gpu_ids}
    busy = [i for i, c in selected.items() if c['pids'] or c['memory_used_mib'] > 512]
    if busy:
        raise RuntimeError('Selected GPUs are occupied: ' + ','.join(busy))
    if any(c['memory_total_mib'] < 75000 for c in selected.values()):
        raise RuntimeError('Validated for four A100 80GB cards')
    work = Path(args.work_dir).resolve()
    if args.resume_from:
        resume = Path(args.resume_from).resolve()
        if not resume.is_file():
            raise FileNotFoundError(resume)
        if ROOT / 'preparation' in resume.parents:
            raise ValueError('Verification weights must not initialize formal training')
        if resume.parent != work:
            raise ValueError('Resume checkpoint must belong to the selected formal work directory')
        import importlib
        import torch
        from mmcv.models import build_model
        importlib.import_module(os.path.dirname(cfg.plugin_dir).replace('/', '.'))
        checkpoint = torch.load(str(resume), map_location='cpu')
        completed = checkpoint.get('meta', {}).get('iter')
        if (type(completed) is not int or not 0 < completed < cfg.runner.max_iters
                or resume.name != 'iter_%d.pth' % completed):
            raise ValueError('Checkpoint iteration is inconsistent or training is already complete')
        if not checkpoint.get('optimizer', {}).get('state'):
            raise ValueError('Resume requires the saved AdamW state')
        check_model = build_model(cfg.model)
        check_model.load_state_dict(checkpoint['state_dict'], strict=True)
        del checkpoint, check_model
    elif work.exists() and (list(work.glob('*.pth')) or (work / 'run_state.json').exists()):
        raise RuntimeError('Existing run detected: explicitly resume or choose a new work directory')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONPATH=str(ROOT),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    # Preserve any user-selected WANDB_MODE; no login or credential changes.
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=4',
        '--master_port=%d' % args.port, 'adzoo/drivetransformer/train.py',
        str(CONFIG.relative_to(ROOT)), '--launcher', 'pytorch', '--seed', '0',
        '--work-dir', str(work)]
    if args.resume_from:
        cmd += ['--resume-from', str(Path(args.resume_from).resolve())]
    return dict(config=str(CONFIG), work_dir=str(work), command=cmd,
        gpus=selected, training_contract=dict(cfg.training_contract)), env


def status(work):
    work = Path(work).resolve()
    state_path = work / 'run_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'status': 'not_started'}
    result = dict(state)
    result['work_dir'] = str(work)
    if state.get('status') == 'running':
        result['supervisor_alive'] = process_identity(state['supervisor_pid']) == state['supervisor_start_ticks']
        result['trainer_alive'] = process_identity(state['trainer_pid']) == state['trainer_start_ticks']
        if not result['trainer_alive']:
            result['status'] = 'exited_awaiting_final_status' if result['supervisor_alive'] else 'interrupted_or_unrecorded_exit'
    logs = sorted(work.glob('*.log.json'), key=lambda p: p.stat().st_mtime) if work.exists() else []
    records = []
    if logs:
        with logs[-1].open('rb') as f:
            f.seek(max(0, logs[-1].stat().st_size - 1024*1024))
            for line in f.read().decode(errors='replace').splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('mode') == 'train' and 'iter' in row:
                    records.append(row)
        result['log_file'] = str(logs[-1])
        result['log_age_seconds'] = round(time.time() - logs[-1].stat().st_mtime, 1)
    maximum = state.get('training_contract', {}).get('optimizer_updates', 58700)
    if records:
        last = records[-1]
        result.update(iteration=last['iter'], max_iterations=maximum,
            progress_percent=round(100*last['iter']/maximum, 2),
            latest_loss=last.get('loss'), latest_grad_norm=last.get('grad_norm'))
        durations = [x['time'] for x in records[-20:] if isinstance(x.get('time'), (int, float))]
        if durations:
            seconds = sum(durations)/len(durations)
            result.update(seconds_per_iteration=round(seconds, 3),
                eta_hours=round(max(0, maximum-last['iter'])*seconds/3600, 2))
    last_ckpt = work / 'latest.pth'
    result['latest_checkpoint'] = str(last_ckpt.resolve()) if last_ckpt.is_file() else None
    result['evaluation_status'] = 'not_started_by_this_launcher'
    return result


def start(args):
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / '.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared, env = preflight(args)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = work / ('launcher_%s.log' % stamp)
        state = dict(prepared, status='running', started_at=utc_now(),
            supervisor_pid=os.getpid(), supervisor_start_ticks=process_identity(os.getpid()),
            launcher_log=str(log_path), resume_from=args.resume_from)
        with log_path.open('w') as output:
            child = subprocess.Popen(prepared['command'], cwd=ROOT, env=env,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(trainer_pid=child.pid, trainer_start_ticks=process_identity(child.pid))
            atomic_json(work / 'run_state.json', state)
            def forward_signal(signum, frame):
                if child.poll() is None:
                    os.killpg(child.pid, signum)
            signal.signal(signal.SIGTERM, forward_signal)
            signal.signal(signal.SIGINT, forward_signal)
            code = child.wait()
        maximum = prepared['training_contract']['optimizer_updates']
        expected = work / ('iter_%d.pth' % maximum)
        final = status(work)
        completed = code == 0 and expected.is_file() and final.get('iteration') == maximum
        state.update(status='completed' if completed else 'failed', exit_code=code,
            finished_at=utc_now(), final_checkpoint=str(expected) if expected.is_file() else None,
            final_iteration=final.get('iteration'))
        atomic_json(work / 'run_state.json', state)
        print(json.dumps(status(work), indent=2, ensure_ascii=False))
        return 0 if completed else (code or 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'start', 'status'])
    parser.add_argument('--work-dir', default=str(DEFAULT_WORK))
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--port', type=int, default=35531)
    parser.add_argument('--resume-from')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'status':
        print(json.dumps(status(args.work_dir), indent=2, ensure_ascii=False))
    elif args.action == 'check':
        print(json.dumps(preflight(args)[0], indent=2, ensure_ascii=False))
    else:
        return start(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
