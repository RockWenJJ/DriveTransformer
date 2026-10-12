#!/usr/bin/env python3
"""Fail-closed, one-owner DT/HiP training successor (stdlib control plane).

JSON spec: version=1, root, repository, python, code_sha, source_sha256, data_files
{absolute-or-repository-relative-path: sha256}, cpu_receipt, predecessor
{root, service, contract_sha256}, config, smoke_config, train_dir, smoke_dir,
stages {gpu_gate, smoke_train, smoke_checkpoint, train, final_checkpoint}.
Predecessor mode defaults to completed. Explicit mode=user_stopped requires a
preservation_receipt, with both receipt and preserved checkpoint hash-pinned.
Each stage has argv (list; never shell), and gates have receipt. Placeholders:
{root}, {repository}, {python}, {config}, {smoke_config}, {train_dir},
{smoke_dir}, {checkpoint}, {expected_iterations}, {receipt}, {attempt}.
Gate receipts require passed/finite true. gpu_gate additionally requires
new_components_grad_ok true; checkpoint gates require strict_load,
optimizer_present, optimizer_finite true and iteration=20 or 29350.
receipt_keys may map these semantic keys to dot paths in a checker's schema.
Optional resume_checkpoint stage uses identical checkpoint-gate semantics;
its checkpoint and expected_iterations are derived from the saved checkpoint.
CPU receipt has passed=true; its SHA must be pinned in source_sha256.
Data index, anchor, config, this supervisor, validation code, all model source
and input provenance files must be pinned by source_sha256 before deployment.
No batch fallback, predecessor restart, or external process termination.
"""
import argparse
import contextlib
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class GateError(RuntimeError):
    pass


class Interrupted(RuntimeError):
    pass


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


@contextlib.contextmanager
def job_lock(path):
    with Path(path).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GateError('Another supervisor owns this job; no takeover') from exc
        yield


def path_value(value, path):
    for part in path.split('.'):
        value = value[part]
    return value


def verify_receipt(path, kind, expected_iterations=None, keys=None):
    receipt = read_json(path)
    required = {'passed': True}
    if kind != 'cpu':
        required['finite'] = True
    if kind == 'gpu_gate':
        required['new_components_grad_ok'] = True
    if 'checkpoint' in kind:
        required.update(strict_load=True, optimizer_present=True,
                        optimizer_finite=True, iteration=expected_iterations)
    for semantic, expected in required.items():
        try:
            actual = path_value(receipt, (keys or {}).get(semantic, semantic))
        except (KeyError, TypeError) as exc:
            raise GateError('Receipt %s lacks %s' % (path, semantic)) from exc
        if type(actual) is not type(expected) or actual != expected:
            raise GateError('Receipt %s failed %s: %r != %r' % (path, semantic, actual, expected))
    return receipt


def predecessor_status(root, contract_sha):
    """Only the named predecessor is a dependency; unknown states fail closed."""
    root = Path(root)
    job = read_json(root / 'job_state.json')
    phase = job.get('phase')
    if phase in ('failed', 'interrupted', 'needs_attention'):
        raise GateError('Predecessor did not succeed: %s' % phase)
    if phase in ('initializing', 'waiting_for_assigned_gpus', 'training_refiner_online_10_epochs'):
        return False
    if phase != 'completed':
        raise GateError('Unknown predecessor phase: %r' % phase)
    state, result = read_json(root / 'run/state.json'), read_json(root / 'run/result.json')
    if not (state.get('phase') == 'completed' and state.get('complete') is True and
            state.get('epoch') == 10 and result.get('complete') is True and
            result.get('epochs') == 10 and result.get('formal_training') is True):
        raise GateError('Predecessor completion receipts disagree')
    if any(x.get('contract_sha256') != contract_sha for x in (state, result)):
        raise GateError('Predecessor contract changed')
    if any(x.get('raw_samples_seen') != 2347690 for x in (state, result)):
        raise GateError('Predecessor lacks ten full raw-frame passes')
    reports = result.get('epoch_reports', [])
    if len(reports) != 10 or [r.get('epoch') for r in reports] != list(range(1, 11)):
        raise GateError('Predecessor epoch receipts incomplete')
    for report in reports:
        if (report.get('raw_train_records') != 234769 or
                report.get('train_stream', {}).get('complete') is not True or
                report.get('val_stream', {}).get('complete') is not True or
                report.get('val', {}).get('records') != 12806):
            raise GateError('Predecessor train/validation coverage incomplete')
    if not (root / 'run/last.pt').is_file():
        raise GateError('Predecessor final checkpoint missing')
    return True


def user_stopped_predecessor(pred, pinned, repository):
    """Accept explicit stop authorization and preserved weights, never completion."""
    if pred.get('mode') != 'user_stopped':
        raise GateError('Explicit user-stopped predecessor mode is required')
    inventory = {}
    for name, expected in pinned.items():
        path = Path(name)
        if not path.is_absolute():
            path = Path(repository) / path
        inventory[str(path.resolve())] = expected

    def pinned_file(name, label):
        if not isinstance(name, str) or not Path(name).is_absolute():
            raise GateError('Missing absolute %s path' % label)
        path = Path(name).resolve()
        expected = inventory.get(str(path))
        if expected is None:
            raise GateError('Unpinned %s' % label)
        if not path.is_file() or digest(path) != expected:
            raise GateError('Pinned %s changed' % label)
        return path, expected

    receipt_path, receipt_sha = pinned_file(pred.get('preservation_receipt'), 'preservation receipt')
    receipt = read_json(receipt_path)
    for key in ('passed', 'user_authorized_stop', 'strict_load', 'optimizer_present',
                'optimizer_finite', 'finite'):
        if receipt.get(key) is not True:
            raise GateError('Preservation receipt lacks true %s' % key)
    if receipt.get('mode') != 'user_stopped':
        raise GateError('Preservation receipt is not a user-stop authorization')
    if not isinstance(receipt.get('user_request'), str) or not receipt['user_request'].strip():
        raise GateError('Preservation receipt lacks the explicit user request')
    original_root = receipt.get('original_root')
    if (not isinstance(original_root, str) or not Path(original_root).is_absolute() or
            Path(original_root).resolve() != Path(pred['root']).resolve()):
        raise GateError('Preservation receipt predecessor root mismatch')
    if receipt.get('contract_sha256') != pred['contract_sha256']:
        raise GateError('Preserved checkpoint contract mismatch')
    stopped = receipt.get('stopped_service', {})
    if (stopped.get('name') != pred['service'] or stopped.get('ActiveState') != 'inactive' or
            stopped.get('MainPID') != '0'):
        raise GateError('Preservation receipt does not confirm the named service stopped')
    checkpoint, checkpoint_sha = pinned_file(receipt.get('checkpoint_path'), 'preserved checkpoint')
    if receipt.get('checkpoint_sha256') != checkpoint_sha:
        raise GateError('Preserved checkpoint hash disagrees with the receipt')
    return dict(receipt=receipt, receipt_path=str(receipt_path), receipt_sha256=receipt_sha,
                checkpoint_path=str(checkpoint), checkpoint_sha256=checkpoint_sha)


def gpu_occupants(inventory, processes, indexes=(0, 1, 2, 3)):
    ids = {}
    for row in csv.reader(io.StringIO(inventory)):
        if row:
            if len(row) != 2:
                raise GateError('Invalid GPU inventory')
            ids[int(row[0].strip())] = row[1].strip()
    if not set(indexes).issubset(ids):
        raise GateError('Assigned GPUs missing from inventory')
    wanted = {ids[i] for i in indexes}
    occupied = []
    for row in csv.reader(io.StringIO(processes)):
        if row:
            if len(row) < 3 or row[0].strip() not in set(ids.values()):
                raise GateError('Unknown GPU process inventory')
            int(row[1].strip())
            if row[0].strip() in wanted:
                occupied.append([x.strip() for x in row])
    return occupied


def finite_tree(value, torch):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v, torch) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v, torch) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def inspect_checkpoint(path, predecessor_root=None):
    """Runs only in the existing DT Python, with GPUs hidden by the caller."""
    import torch
    checkpoint = torch.load(path, map_location='cpu')
    if not checkpoint.get('optimizer') or not finite_tree(checkpoint['optimizer'], torch):
        raise GateError('Checkpoint lacks a finite optimizer')
    if predecessor_root:
        for name, expected in read_json(Path(predecessor_root) / 'executed_source_sha256.json').items():
            if digest(Path(predecessor_root) / name) != expected:
                raise GateError('Predecessor executed source changed: %s' % name)
        sys.path.insert(0, str(Path(predecessor_root).resolve()))
        from refinement import SingleCandidateRefiner
        model = SingleCandidateRefiner(**checkpoint.get('refiner_config', {}))
        model.load_state_dict(checkpoint['refiner'], strict=True)
        progress = checkpoint['progress']
        if not (progress.get('complete') is True and progress.get('epoch') == 10 and
                progress.get('raw_samples_seen') == 2347690):
            raise GateError('Predecessor checkpoint is not complete')
        weights = checkpoint['refiner']
        iteration = progress['global_step']
    else:
        weights = checkpoint['state_dict']
        iteration = int(checkpoint['meta']['iter']) + 1
    if not weights or not finite_tree(weights, torch):
        raise GateError('Checkpoint weights are not finite')
    return dict(passed=True, finite=True, optimizer_present=True, optimizer_finite=True,
                iteration=iteration, sha256=digest(path),
                contract_sha256=checkpoint.get('contract_sha256'))


class Supervisor:
    def __init__(self, spec):
        self.spec = spec
        self.root = Path(spec['root']).resolve()
        self.repo = Path(spec['repository']).resolve()
        self.python = spec['python']
        self.attempt = time.strftime('%Y%m%dT%H%M%S') + '_%d' % os.getpid()
        self.child = None
        self.state = {'phase': 'initializing', 'stages': {}, 'attempts': [],
                      'spec_sha256': object_digest(spec)}
        self.state_path = self.root / 'job_state.json'
        self.initialized = False

    def emit(self, phase, **extra):
        self.state.update(phase=phase, pid=os.getpid(), updated_at=time.time(), **extra)
        atomic_json(self.state_path, self.state)
        print(json.dumps({'phase': phase, 'pid': os.getpid(), 'updated_at': self.state['updated_at'],
                          **extra}, ensure_ascii=False, allow_nan=False), flush=True)

    def capture(self, argv, cpu=False):
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        if cpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        return subprocess.check_output(argv, cwd=self.repo, env=env, text=True, timeout=120).strip()

    def verify_inputs(self):
        s = self.spec
        if s.get('version') != 1 or s.get('gpus', [0, 1, 2, 3]) != [0, 1, 2, 3]:
            raise GateError('Expected version 1 and GPUs 0-3')
        if s.get('microbatch', 20) != 20 or s.get('max_iters', 29350) != 29350 or s.get('smoke_iters', 20) != 20:
            raise GateError('Fixed 20x4/29350 and 20-step smoke budget changed')
        if self.capture(['git', 'rev-parse', 'HEAD']) != s['code_sha']:
            raise GateError('Code SHA changed')
        if self.capture(['git', 'diff', '--name-only', 'HEAD']):
            raise GateError('Tracked source dirty')
        pinned = s.get('source_sha256', {})
        if not pinned:
            raise GateError('Missing source/config/data hash inventory')
        paths = set()
        for name, expected in pinned.items():
            path = Path(name)
            if not path.is_absolute():
                path = self.repo / path
            paths.add(str(path.resolve()))
            if digest(path) != expected:
                raise GateError('Pinned input changed: %s' % path)
        untracked = self.capture(['git', 'ls-files', '--others', '--exclude-standard', '-z'])
        for name in untracked.split('\0'):
            if not name:
                continue
            path = self.repo / name
            if path.suffix in ('.py', '.so', '.pyd', '.pyc') and str(path.resolve()) not in paths:
                raise GateError('Unpinned untracked importable source: %s' % path)
        for key in ('config', 'smoke_config', 'cpu_receipt'):
            if str(Path(s[key]).resolve()) not in paths:
                raise GateError('Unpinned required input: %s' % key)
        if str(Path(__file__).resolve()) not in paths:
            raise GateError('Supervisor source is not pinned')
        if not s.get('data_files'):
            raise GateError('Missing explicit data provenance files')
        for name in s['data_files']:
            path = Path(name)
            if not path.is_absolute():
                path = self.repo / path
            if str(path.resolve()) not in paths:
                raise GateError('Unpinned data provenance file: %s' % path)
        pred = s.get('predecessor', {})
        if pred.get('mode', 'completed') not in ('completed', 'user_stopped'):
            raise GateError('Unknown predecessor admission mode')
        if pred.get('mode') == 'user_stopped':
            user_stopped_predecessor(pred, pinned, self.repo)
        verify_receipt(s['cpu_receipt'], 'cpu', keys=s.get('cpu_receipt_keys'))
        for key in ('train_dir', 'smoke_dir'):
            path = Path(s[key]).resolve()
            if self.root not in path.parents:
                raise GateError('Work dirs must be isolated under this job root')
        if Path(s['train_dir']).resolve() == Path(s['smoke_dir']).resolve():
            raise GateError('Smoke and formal output dirs must be distinct')
        if set(('gpu_gate', 'smoke_train', 'smoke_checkpoint', 'train', 'final_checkpoint')) - set(s['stages']):
            raise GateError('Missing required stages')

    def initialize(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'logs').mkdir(exist_ok=True)
        if self.state_path.exists():
            previous = read_json(self.state_path)
            if previous.get('spec_sha256') != self.state['spec_sha256']:
                raise GateError('Job spec changed; use a new job directory')
            if previous.get('phase') == 'needs_attention':
                raise GateError('Previous failure requires an explicit audited recovery')
            self.state = previous
        self.state['attempts'].append(dict(id=self.attempt, started_at=time.time()))
        self.verify_inputs()
        self.initialized = True

    def wait_resources(self):
        pred = self.spec['predecessor']
        if pred.get('mode') == 'user_stopped':
            return self.wait_user_stopped_resources()
        if pred.get('mode', 'completed') != 'completed':
            raise GateError('Unknown predecessor admission mode')
        while True:
            success = predecessor_status(pred['root'], pred['contract_sha256'])
            if not success:
                self.emit('waiting_predecessor', predecessor=pred['root'])
            else:
                raw = self.capture(['systemctl', '--user', 'show', pred['service'], '-p', 'ActiveState',
                                    '-p', 'SubState', '-p', 'MainPID', '-p', 'ExecMainStatus'])
                service = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
                if service.get('ActiveState') == 'failed' or service.get('ExecMainStatus') != '0':
                    raise GateError('Predecessor service ended unsuccessfully')
                if service.get('ActiveState') in ('active', 'activating', 'deactivating'):
                    self.emit('waiting_predecessor_exit')
                elif service.get('ActiveState') != 'inactive' or service.get('MainPID') != '0':
                    raise GateError('Unknown predecessor service state')
                else:
                    inventory = self.capture(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'])
                    processes = self.capture(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                              '--format=csv,noheader,nounits'])
                    occupied = gpu_occupants(inventory, processes)
                    if not occupied:
                        checkpoint = Path(pred['root']) / 'run/last.pt'
                        result = json.loads(self.capture([self.python, '-B', str(Path(__file__).resolve()),
                                                         '--inspect-checkpoint', str(checkpoint),
                                                         '--predecessor-root', pred['root']], cpu=True))
                        if result.get('contract_sha256') != pred['contract_sha256']:
                            raise GateError('Predecessor checkpoint contract mismatch')
                        self.state['predecessor_checkpoint'] = result
                        self.emit('resources_ready')
                        return
                    self.emit('waiting_assigned_gpus', occupants=occupied)
            time.sleep(min(60, max(1, self.spec.get('poll_seconds', 30))))

    def wait_user_stopped_resources(self):
        pred = self.spec['predecessor']
        while True:
            preservation = user_stopped_predecessor(pred, self.spec.get('source_sha256', {}), self.repo)
            raw = self.capture(['systemctl', '--user', 'show', pred['service'], '-p', 'ActiveState',
                                '-p', 'SubState', '-p', 'MainPID', '-p', 'ExecMainStatus'])
            service = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
            # An intentional stop may have a nonzero exit status. Admission is
            # instead bound to the explicit authorization and a live empty unit.
            if service.get('ActiveState') != 'inactive' or service.get('MainPID') != '0':
                raise GateError('User-stopped predecessor service is not confirmed inactive')
            inventory = self.capture(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'])
            processes = self.capture(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                      '--format=csv,noheader,nounits'])
            occupied = gpu_occupants(inventory, processes)
            if not occupied:
                self.state['predecessor_preservation'] = preservation
                self.state['predecessor_service'] = service
                self.emit('resources_ready', predecessor_mode='user_stopped')
                return
            self.emit('waiting_assigned_gpus', occupants=occupied, predecessor_mode='user_stopped')
            time.sleep(min(60, max(1, self.spec.get('poll_seconds', 30))))

    def stop_owned(self):
        if self.child is None or self.child.poll() is not None:
            return
        # This Popen instance is owned by this invocation; never kill a saved PID.
        try:
            os.killpg(self.child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.child.wait(timeout=45)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.child.wait(timeout=15)

    def wait_assigned_gpus(self):
        """Recheck after each gate; another task may have arrived during CPU work."""
        while True:
            inventory = self.capture(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'])
            processes = self.capture(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                      '--format=csv,noheader,nounits'])
            occupied = gpu_occupants(inventory, processes)
            if not occupied:
                return
            self.emit('waiting_assigned_gpus', occupants=occupied)
            time.sleep(min(60, max(1, self.spec.get('poll_seconds', 30))))

    def substitutions(self, name, checkpoint=None, expected=None):
        values = dict(self.spec)
        values.update(root=str(self.root), repository=str(self.repo), attempt=self.attempt,
                      checkpoint=str(checkpoint or ''), expected_iterations=expected or 0)
        stage = self.spec['stages'][name]
        values['receipt'] = stage.get('receipt', '').format(**values)
        return values

    def run_stage(self, name, checkpoint=None, expected=None, resume=None):
        stage = self.spec['stages'][name]
        previous = self.state['stages'].get(name)
        if previous and previous.get('status') == 'completed':
            if previous.get('receipt') and digest(previous['receipt']) != previous['receipt_sha256']:
                raise GateError('Accepted gate receipt changed')
            return
        if previous and previous.get('status') in ('running', 'interrupted') and name not in ('train', 'resume_checkpoint'):
            raise GateError('Interrupted smoke/gate is preserved; audited recovery required')
        values = self.substitutions(name, checkpoint, expected)
        argv = [arg.format(**values) for arg in stage['argv']]
        if not argv or argv[0] != self.python:
            raise GateError('All job commands must use the existing DT Python')
        if name in ('train', 'smoke_train'):
            if '--resume-from' in argv or '--load-from' in argv:
                raise GateError('Training argv must not silently inherit weights')
            work_dir = Path(self.spec['train_dir' if name == 'train' else 'smoke_dir'])
            if resume:
                argv += ['--resume-from', str(resume)]
            elif work_dir.exists() and any(work_dir.iterdir()):
                raise GateError('Fresh training output dir is not empty: %s' % work_dir)
        receipt = values['receipt']
        if receipt:
            receipt_path = Path(receipt).resolve()
            if self.root not in receipt_path.parents:
                raise GateError('Gate receipt must remain inside this isolated job')
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
        if receipt and Path(receipt).exists():
            raise GateError('Refusing to overwrite an unaccepted gate receipt: %s' % receipt)
        log = self.root / 'logs' / ('%s_%s.log' % (self.attempt, name))
        env = os.environ.copy()
        env.update(self.spec.get('env', {}))
        env.update(PYTHONPATH=str(self.repo), CUDA_VISIBLE_DEVICES='0,1,2,3',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1')
        if name.endswith('checkpoint'):
            env['CUDA_VISIBLE_DEVICES'] = ''
        entry = dict(status='running', argv=argv, log=str(log), attempt=self.attempt, started_at=time.time())
        self.state['stages'][name] = entry
        self.emit(name)
        with log.open('x') as stream:
            self.child = subprocess.Popen(argv, cwd=self.repo, env=env, stdout=stream,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            entry['pid'] = self.child.pid
            self.emit(name)
            while self.child.poll() is None:
                time.sleep(min(10, self.spec.get('child_poll_seconds', 10)))
            rc = self.child.returncode
        self.child = None
        entry.update(returncode=rc, ended_at=time.time())
        if rc != 0:
            entry['status'] = 'failed'
            raise GateError('%s failed with exit %d; see %s' % (name, rc, log))
        if name not in ('train', 'smoke_train'):
            if not receipt:
                raise GateError('Missing gate receipt path')
            verify_receipt(receipt, name, expected, stage.get('receipt_keys'))
            entry.update(receipt=receipt, receipt_sha256=digest(receipt))
        entry['status'] = 'completed'
        self.emit(name + '_completed')

    def run(self):
        self.initialize()
        if self.state.get('phase') == 'completed':
            return
        self.wait_resources()
        self.verify_inputs()
        self.wait_assigned_gpus()
        self.run_stage('gpu_gate')
        self.wait_assigned_gpus()
        self.run_stage('smoke_train')
        smoke_checkpoint = Path(self.spec['smoke_dir']) / 'iter_20.pth'
        self.run_stage('smoke_checkpoint', smoke_checkpoint, 20)
        self.verify_inputs()
        train = self.state['stages'].get('train', {})
        resume = None
        if train.get('status') in ('running', 'interrupted'):
            resume = Path(self.spec['train_dir']) / 'latest.pth'
            if not resume.is_file() or 'resume_checkpoint' not in self.spec['stages']:
                raise GateError('Interrupted formal training has no verifiable resume checkpoint')
            info = json.loads(self.capture([self.python, '-B', str(Path(__file__).resolve()),
                                           '--inspect-checkpoint', str(resume)], cpu=True))
            if not 0 < info['iteration'] <= 29350:
                raise GateError('Resume checkpoint iteration outside the fixed budget')
            previous = self.state['stages'].pop('resume_checkpoint', None)
            if previous:
                self.state.setdefault('resume_gate_history', []).append(previous)
            self.run_stage('resume_checkpoint', resume, info['iteration'])
            self.state['resume_checkpoint'] = info
            if info['iteration'] == 29350:
                train['status'] = 'completed'
        self.wait_assigned_gpus()
        self.run_stage('train', resume=resume)
        self.run_stage('final_checkpoint', Path(self.spec['train_dir']) / 'iter_29350.pth', 29350)
        self.emit('completed', finished_at=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--inspect-checkpoint', type=Path)
    parser.add_argument('--predecessor-root', type=Path)
    args = parser.parse_args()
    if args.inspect_checkpoint:
        print(json.dumps(inspect_checkpoint(args.inspect_checkpoint, args.predecessor_root)))
        return
    if args.spec is None:
        parser.error('--spec is required')
    spec = read_json(args.spec)
    root = Path(spec['root'])
    root.mkdir(parents=True, exist_ok=True)
    with job_lock(root / 'job.lock'):
        supervisor = Supervisor(spec)
        def interrupted(signum, frame):
            raise Interrupted('signal %d' % signum)
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            supervisor.run()
        except BaseException as exc:
            supervisor.stop_owned()
            phase = supervisor.state.get('phase')
            if phase in supervisor.state['stages']:
                entry = supervisor.state['stages'][phase]
                if entry.get('status') == 'running':
                    entry['status'] = 'interrupted' if isinstance(exc, Interrupted) else 'failed'
            if supervisor.initialized:
                supervisor.emit('interrupted' if isinstance(exc, Interrupted) else 'needs_attention',
                                failed_phase=phase, error=str(exc))
            else:
                # A changed spec must not overwrite the original task's ledger.
                error_path = root / ('initialization_error_%s.json' % supervisor.attempt)
                atomic_json(error_path, dict(phase='needs_attention', error=str(exc),
                                            attempted_spec_sha256=object_digest(spec)))
                print(json.dumps(dict(phase='needs_attention', error=str(exc),
                                      receipt=str(error_path))), flush=True)
            raise


if __name__ == '__main__':
    main()
