"""Launch Bench2Drive with the frozen dev24 routes and this checkout's agent.

--dry-run validates configuration/routing without creating an output directory,
loading weights, or starting CARLA. The full evaluator runs only without it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.dev24_routes import resolve_routes


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def build_plan(args):
    from mmcv import Config
    config, checkpoint = Path(args.config).resolve(), Path(args.ckpt).resolve()
    if not config.is_file() or not checkpoint.is_file():
        raise FileNotFoundError('Both config and checkpoint must exist')
    if '+' in str(config) or '+' in str(checkpoint):
        raise ValueError('The agent config+checkpoint format does not support + in paths')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        raise ValueError('run-name must be one directory name')
    cfg = Config.fromfile(str(config))
    profile = dict(cfg.get('closed_loop_eval', {}))
    routes = resolve_routes(profile.get('route_manifest'), args.shard)
    # A bound profile must agree with the canonical 24-route file. Sharding is
    # then resolved explicitly from the same manifest, never from a dev94 file.
    if profile:
        whole = resolve_routes(profile.get('route_manifest'))
        for key in ('subset', 'routes', 'route_count', 'routes_sha256'):
            if profile.get(key) != whole[key]:
                raise ValueError('Stale closed_loop_eval configuration: ' + key)
    benchmark = Path(args.benchmark_root).resolve()
    carla = Path(args.carla_root).resolve()
    leaderboard = benchmark / 'leaderboard'
    evaluator = leaderboard / 'leaderboard/leaderboard_evaluator.py'
    agent = REPO / 'team_code/drivetransformer_b2d_agent.py'
    for path in [evaluator, agent, carla / 'CarlaUE4.sh']:
        if not path.is_file():
            raise FileNotFoundError(path)
    rpc = args.rpc_port if args.rpc_port is not None else 32000 + 20 * args.gpu
    tm = args.tm_port if args.tm_port is not None else 52000 + 20 * args.gpu
    if not (1024 <= rpc < 65535 and 1024 <= tm <= 65535) or tm in (rpc, rpc + 1):
        raise ValueError('RPC/RPC+1 and traffic-manager ports must be distinct usable ports')
    suffix = 'all' if args.shard is None else 'shard%d' % args.shard
    output = benchmark / 'eval_results' / ('dev24_v1_' + args.run_name) / ('seed%d_%s' % (args.seed, suffix))
    env = os.environ.copy()
    # Both forms of imports in the agent must resolve here:
    # team_code.* and DriveTransformer.team_code.*.
    search_paths = [REPO, REPO.parent, benchmark, carla / 'PythonAPI',
                    carla / 'PythonAPI/carla', leaderboard,
                    leaderboard / 'team_code', benchmark / 'scenario_runner']
    env.update(PYTHONPATH=os.pathsep.join(map(str, search_paths)),
               PYTHONHASHSEED=str(args.seed), CUDA_VISIBLE_DEVICES=str(args.gpu),
               CARLA_ROOT=str(carla), SCENARIO_RUNNER_ROOT=str(benchmark / 'scenario_runner'),
               LEADERBOARD_ROOT=str(leaderboard), CHALLENGE_TRACK_CODENAME='SENSORS',
               IS_BENCH2DRIVE='True', PLANNER_TYPE='only_traj', GPU_RANK=str(args.gpu),
               ROUTES=routes['routes'], TEAM_AGENT=str(agent),
               TEAM_CONFIG=str(config)+'+'+str(checkpoint), SAVE_PATH=str(output),
               CHECKPOINT_ENDPOINT=str(output / 'results.json'), DISPLAY='')
    user_dir = Path.home()
    env.setdefault('VK_ICD_FILENAMES', str(user_dir / 'vulkan-local/icd/nvidia_icd.json'))
    env['LD_LIBRARY_PATH'] = str(user_dir / 'vulkan-local/x/usr/lib/x86_64-linux-gnu') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    evaluator_args = [str(evaluator), '--routes='+routes['routes'], '--repetitions=1',
        '--track=SENSORS', '--checkpoint='+str(output / 'results.json'),
        '--debug-checkpoint='+str(output / 'live_results.txt'), '--agent='+str(agent),
        '--agent-config='+env['TEAM_CONFIG'], '--debug=0', '--port='+str(rpc),
        '--traffic-manager-port='+str(tm), '--traffic-manager-seed='+str(args.seed),
        '--gpu-rank='+str(args.gpu)]
    if args.resume:
        evaluator_args.append('--resume=True')
    cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', str(args.seed)] + evaluator_args
    plan = dict(**routes, config=str(config), checkpoint=str(checkpoint),
        checkpoint_size=checkpoint.stat().st_size, checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
        resolved_config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
        inference_contract=dict(cfg.get('inference_contract', {})),
        agent=str(agent), agent_sha256=sha256_file(agent),
        controller=str(REPO / 'team_code/pid_controller.py'),
        controller_sha256=sha256_file(REPO / 'team_code/pid_controller.py'),
        repository=str(REPO), cwd=str(leaderboard), pythonpath=env['PYTHONPATH'],
        seed=args.seed, gpu=args.gpu, rpc_port=rpc, traffic_manager_port=tm,
        output_dir=str(output), command=cmd, evaluator_arguments=evaluator_args,
        dry_run=args.dry_run, resume=args.resume)
    return plan, env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--run-name', required=True, help='For example: base_C1 or dinob_C1')
    p.add_argument('--shard', type=int, choices=[1, 2, 3, 4])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rpc-port', type=int)
    p.add_argument('--tm-port', type=int)
    p.add_argument('--benchmark-root', default='/data5T/wenjj/Projects/Bench2Drive')
    p.add_argument('--carla-root', default='/data5T/wenjj/carla')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.gpu < 0 or not 0 <= args.seed < 2**32:
        p.error('gpu must be nonnegative and seed must be in [0, 2**32)')
    plan, env = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output = Path(plan['output_dir'])
    record = output / 'eval_manifest.json'
    plan['checkpoint_sha256'] = sha256_file(plan['checkpoint'])
    if args.resume:
        previous = json.loads(record.read_text())
        for key in ['routes_sha256', 'config', 'checkpoint_sha256', 'resolved_config_sha256',
                    'seed', 'agent_sha256', 'controller_sha256', 'repository']:
            if previous[key] != plan[key]:
                raise ValueError('Resume protocol differs: ' + key)
    else:
        output.mkdir(parents=True, exist_ok=False)
        record.write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(plan, indent=2), flush=True)
    raise SystemExit(subprocess.call(plan['command'], cwd=plan['cwd'], env=env))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        # Apply the requested seed in the evaluator process before model setup.
        import random
        import runpy
        import numpy as np
        import torch
        seed = int(sys.argv[2])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        sys.argv = sys.argv[3:]
        runpy.run_path(sys.argv[0], run_name='__main__')
    else:
        main()
