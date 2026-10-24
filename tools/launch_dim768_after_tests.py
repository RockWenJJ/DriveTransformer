"""Wait for current evaluations and four idle GPUs, validate DDP, then train."""
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path('/data5T/wenjj/Projects/DriveTransformer')
TMP = pathlib.Path('/data5T/wenjj/tmp_scripts')
RESULTS = pathlib.Path('/data5T/wenjj/Projects/Bench2Drive/eval_results')
CFG = ROOT / 'adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_dim768.py'
PY = '/data5T/wenjj/miniconda3/envs/drivetransformer/bin/python'
WD = ROOT / 'adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_base_dim768_10ep'
VALID = TMP / 'dim768_validation'
STATE = TMP / 'dim768_queue_status.json'


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def status(phase, **kwargs):
    data = dict(phase=phase, time=datetime.datetime.now().isoformat(), **kwargs)
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(STATE)
    print(data, flush=True)


def prerequisites():
    reasons = []
    proof = read(VALID / 'passed.json')
    if proof.get('config_sha256') != hashlib.sha256(CFG.read_bytes()).hexdigest():
        reasons.append('768 config has no matching successful microbatch20 validation')
    if read(TMP / 'orion_openloop_status.json').get('phase') != 'completed':
        reasons.append('ORION open-loop not successfully completed')
    newest = 0
    for i in range(4):
        path = RESULTS / f'dv3bft_94_c{i}/results.json'
        result = read(path).get('_checkpoint', {})
        progress = result.get('progress', [0, 1])
        expected = 24 if i < 2 else 23
        if progress != [expected, expected] or len(result.get('records', [])) != expected:
            reasons.append(f'ViT-B-ft closed-loop chunk {i} incomplete')
        if path.exists():
            newest = max(newest, path.stat().st_mtime)
    merged = RESULTS / 'dt_dinov3_vitb16ft_dev94/results.json'
    records = read(merged).get('_checkpoint', {}).get('records', [])
    if len({r.get('route_id') for r in records}) != 94:
        reasons.append('ViT-B-ft merged results not 94 unique routes')
    ability = merged.parent / 'results_ability.json'
    if not read(ability) or ability.stat().st_mtime < newest:
        reasons.append('ViT-B-ft ability results missing or stale')
    rows = subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid=,args=', '--width', '4000'], text=True)
    markers = ['leaderboard_evaluator.py', 'ability_benchmark.py', 'ability_benchmark_gpu.py',
               'ol_eval_tasks.py', 'ol_eval.py', 'orion_openloop.py', 'orion_queue.py',
               '/bft_finish.sh', '/finish_6routes.sh', '/vitbft_eval.sh', 'CarlaUE4-Linux-Shipping']
    for row in rows.splitlines():
        pid, command = row.strip().split(None, 1)
        if int(pid) == os.getpid():
            continue
        if any(marker in command for marker in markers):
            reasons.append('evaluation/queue still active: pid=' + pid)
        if 'train.py' in command and 'drivetransformer_base_dim768' in command:
            reasons.append('dim768 training already exists: pid=' + pid)
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    free = [int(line.split(',')[0]) for line in lines.splitlines() if int(line.split(',')[1]) < 500]
    if len(free) < 4:
        reasons.append(f'only {len(free)} GPUs idle (<500 MiB)')
    return reasons, free[:4]


def command(gpus, work_dir, smoke=False):
    cmd = [PY, '-m', 'torch.distributed.launch', '--nproc_per_node=4',
           '--master_addr=127.0.0.1', '--master_port=35769',
           'adzoo/drivetransformer/train.py', str(CFG), '--launcher', 'pytorch',
           '--no-validate', '--seed', '0', '--work-dir', str(work_dir)]
    if smoke:
        cmd += ['--cfg-options', 'runner.max_iters=12', 'log_config.interval=1',
                'data.workers_per_gpu=2', 'checkpoint_config.interval=1000',
                'checkpoint_config.save_last=False']
    return cmd


def main():
    lock = (TMP / 'dim768_queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stable = 0
    previous = None
    while True:
        reasons, gpus = prerequisites()
        stable = stable + 1 if not reasons and gpus == previous else (1 if not reasons else 0)
        previous = gpus
        status('waiting', reasons=reasons, candidate_gpus=gpus, stable_checks=stable)
        if stable >= 3:
            break
        time.sleep(60)
    if WD.exists() and any(WD.iterdir()):
        status('blocked', reason='training work directory is nonempty; refusing overwrite')
        return 1
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled')
    ddp_dir = VALID / ('ddp_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    status('ddp_validation', gpus=gpus)
    with (VALID / 'ddp.log').open('w') as log:
        code = subprocess.call(command(gpus, ddp_dir, smoke=True), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if code:
        status('failed_ddp_validation', exit_code=code)
        return code
    reasons, available = prerequisites()
    if reasons or not set(gpus).issubset(available):
        status('blocked', reason='resources or evaluations changed after DDP validation', details=reasons)
        return 1
    env.pop('WANDB_MODE', None)
    env['WANDB_NAME'] = 'dt_base_dim768_10ep_mb20_4gpu'
    status('training', gpus=gpus, microbatch=20, global_batch=80, max_iters=25000, work_dir=str(WD))
    with (TMP / 'dim768_train.log').open('a') as log:
        code = subprocess.call(command(gpus, WD), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    status('completed' if code == 0 else 'training_failed', exit_code=code, work_dir=str(WD))
    return code


if __name__ == '__main__':
    sys.exit(main())
