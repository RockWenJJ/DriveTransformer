"""No CUDA, torch, SSH, systemd, or real training required."""
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


MODULE = pathlib.Path(__file__).resolve().parents[1] / 'tools/hip_training_job.py'
spec = importlib.util.spec_from_file_location('hip_training_job', MODULE)
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        (self.root / 'run').mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def predecessor(self):
        self.put('job_state.json', {'phase': 'completed'})
        self.put('run/state.json', dict(phase='completed', complete=True, epoch=10,
                                       contract_sha256='contract', raw_samples_seen=2347690))
        self.put('run/result.json', dict(complete=True, epochs=10, formal_training=True,
                                        contract_sha256='contract', raw_samples_seen=2347690,
                                        epoch_reports=[dict(epoch=i, raw_train_records=234769,
                                                            train_stream={'complete': True},
                                                            val_stream={'complete': True},
                                                            val={'records': 12806}) for i in range(1, 11)]))
        (self.root / 'run/last.pt').write_bytes(b'checkpoint is inspected by isolated torch checker')

    def test_predecessor_success(self):
        self.predecessor()
        self.assertTrue(job.predecessor_status(self.root, 'contract'))

    def test_predecessor_training_waits(self):
        self.put('job_state.json', {'phase': 'training_refiner_online_10_epochs'})
        self.assertFalse(job.predecessor_status(self.root, 'contract'))

    def test_predecessor_failure_and_unknown_fail_closed(self):
        for phase in ('failed', 'interrupted', 'bogus'):
            self.put('job_state.json', {'phase': phase})
            with self.assertRaises(job.GateError):
                job.predecessor_status(self.root, 'contract')

    def test_partial_or_wrong_contract_not_success(self):
        self.predecessor()
        with self.assertRaises(job.GateError):
            job.predecessor_status(self.root, 'different')
        self.put('run/state.json', {'phase': 'completed', 'epoch': 9})
        with self.assertRaises(job.GateError):
            job.predecessor_status(self.root, 'contract')

    def test_busy_gpu_low_memory_still_busy(self):
        inventory = '0, GPU-a\n1, GPU-b\n2, GPU-c\n3, GPU-d\n4, GPU-e'
        self.assertEqual(job.gpu_occupants(inventory, 'GPU-a, 123, python'), [['GPU-a', '123', 'python']])
        self.assertEqual(job.gpu_occupants(inventory, 'GPU-e, 456, other'), [])
        with self.assertRaises(job.GateError):
            job.gpu_occupants(inventory, 'GPU-unknown, 456, other')

    def test_duplicate_lock_refused(self):
        with job.job_lock(self.root / 'job.lock'):
            with self.assertRaises(job.GateError):
                with job.job_lock(self.root / 'job.lock'):
                    self.fail('duplicate owner admitted')

    def test_receipt_requires_new_grad_and_finite(self):
        p = self.put('gate.json', {'passed': True, 'finite': True})
        with self.assertRaises(job.GateError):
            job.verify_receipt(p, 'gpu_gate')
        self.put('gate.json', {'passed': True, 'finite': True, 'new_components_grad_ok': True})
        job.verify_receipt(p, 'gpu_gate')
        self.put('gate.json', {'passed': 1, 'finite': True, 'new_components_grad_ok': True})
        with self.assertRaises(job.GateError):
            job.verify_receipt(p, 'gpu_gate')

    def test_checkpoint_iteration_and_optimizer_gate(self):
        receipt = dict(passed=True, finite=True, strict_load=True, optimizer_present=True,
                       optimizer_finite=True, iteration=20)
        p = self.put('ckpt.json', receipt)
        job.verify_receipt(p, 'smoke_checkpoint', 20)
        with self.assertRaises(job.GateError):
            job.verify_receipt(p, 'final_checkpoint', 29350)

    def test_failed_command_never_launches_training(self):
        (self.root / 'logs').mkdir()
        s = dict(root=str(self.root), repository=str(self.root), python=sys.executable,
                 stages={'gpu_gate': {'argv': [sys.executable, '-c', 'raise SystemExit(7)'],
                                      'receipt': str(self.root / 'gate.json')}}, child_poll_seconds=.01)
        supervisor = job.Supervisor(s)
        with self.assertRaises(job.GateError):
            supervisor.run_stage('gpu_gate')
        self.assertEqual(supervisor.state['stages']['gpu_gate']['status'], 'failed')
        self.assertNotIn('train', supervisor.state['stages'])

    def test_formal_training_refuses_smoke_resume_and_nonempty_dir(self):
        s = dict(root=str(self.root), repository=str(self.root), python=sys.executable,
                 train_dir=str(self.root / 'run'), stages={'train': {'argv': [sys.executable, '-c', 'pass']}})
        supervisor = job.Supervisor(s)
        (self.root / 'run/keep.txt').write_text('evidence')
        with self.assertRaises(job.GateError):
            supervisor.run_stage('train')
        self.assertEqual((self.root / 'run/keep.txt').read_text(), 'evidence')
        s['stages']['train']['argv'] += ['--resume-from', 'smoke.pth']
        with self.assertRaises(job.GateError):
            supervisor.run_stage('train')

    def test_successful_gate_is_not_executed_twice(self):
        (self.root / 'logs').mkdir()
        receipt = self.root / 'gpu.json'
        marker = self.root / 'count.txt'
        code = ('import pathlib,json; pathlib.Path(%r).write_text("one invocation"); '
                'pathlib.Path(%r).write_text(json.dumps(dict(passed=True,finite=True,new_components_grad_ok=True)))'
                % (str(marker), str(receipt)))
        s = dict(root=str(self.root), repository=str(self.root), python=sys.executable,
                 stages={'gpu_gate': {'argv': [sys.executable, '-c', code], 'receipt': str(receipt)}},
                 child_poll_seconds=.01)
        supervisor = job.Supervisor(s)
        supervisor.run_stage('gpu_gate')
        with mock.patch.object(job.subprocess, 'Popen', side_effect=AssertionError('duplicate launch')):
            supervisor.run_stage('gpu_gate')
        self.assertEqual(marker.read_text(), 'one invocation')

    def test_changed_accepted_receipt_fails_closed(self):
        receipt = self.put('gpu.json', dict(passed=True, finite=True, new_components_grad_ok=True))
        s = dict(root=str(self.root), repository=str(self.root), python=sys.executable,
                 stages={'gpu_gate': {'argv': [sys.executable, '-c', 'pass'], 'receipt': str(receipt)}})
        supervisor = job.Supervisor(s)
        supervisor.state['stages']['gpu_gate'] = dict(status='completed', receipt=str(receipt),
                                                     receipt_sha256=job.digest(receipt))
        self.put('gpu.json', dict(passed=False))
        with self.assertRaises(job.GateError):
            supervisor.run_stage('gpu_gate')

    def test_unknown_service_state_never_admits_gpu_gate(self):
        self.predecessor()
        s = dict(root=str(self.root), repository=str(self.root), python=sys.executable,
                 predecessor=dict(root=str(self.root), service='example.service', contract_sha256='contract'))
        supervisor = job.Supervisor(s)
        with mock.patch.object(supervisor, 'capture', return_value='ActiveState=unknown\nMainPID=0\nExecMainStatus=0'):
            with self.assertRaises(job.GateError):
                supervisor.wait_resources()
        self.assertEqual(supervisor.state['stages'], {})

    def test_initialization_rejects_new_spec_without_rewriting_original(self):
        previous = dict(spec_sha256='original', phase='waiting_predecessor', stages={}, attempts=[])
        self.put('job_state.json', previous)
        supervisor = job.Supervisor(dict(root=str(self.root), repository=str(self.root), python=sys.executable))
        with self.assertRaises(job.GateError):
            supervisor.initialize()
        self.assertEqual(job.read_json(self.root / 'job_state.json'), previous)

    def input_spec(self):
        config = self.put('config.py', {'config': 1})
        smoke = self.put('smoke.py', {'smoke': 1})
        receipt = self.put('cpu.json', {'passed': True})
        data = self.put('data.json', {'dataset': 1})
        pinned = {str(p): job.digest(p) for p in (config, smoke, receipt, data, MODULE)}
        return dict(version=1, root=str(self.root), repository=str(self.root), python=sys.executable,
                    code_sha='commit', source_sha256=pinned, data_files=[str(data)],
                    config=str(config), smoke_config=str(smoke), cpu_receipt=str(receipt),
                    train_dir=str(self.root / 'train'), smoke_dir=str(self.root / 'smoke'),
                    stages={name: {} for name in ('gpu_gate', 'smoke_train', 'smoke_checkpoint', 'train', 'final_checkpoint')})

    def test_untracked_importable_source_requires_hash_pin(self):
        supervisor = job.Supervisor(self.input_spec())
        (self.root / 'shadow.py').write_text('# unexpected importable source')
        def capture(argv):
            return 'commit' if 'rev-parse' in argv else ('shadow.py\0' if 'ls-files' in argv else '')
        with mock.patch.object(supervisor, 'capture', side_effect=capture):
            with self.assertRaisesRegex(job.GateError, 'Unpinned untracked'):
                supervisor.verify_inputs()

    def test_explicitly_pinned_untracked_runtime_is_accepted(self):
        spec = self.input_spec()
        extension = self.root / 'runtime.so'
        extension.write_bytes(b'extension')
        spec['source_sha256'][str(extension)] = job.digest(extension)
        supervisor = job.Supervisor(spec)
        def capture(argv):
            return 'commit' if 'rev-parse' in argv else ('runtime.so\0' if 'ls-files' in argv else '')
        with mock.patch.object(supervisor, 'capture', side_effect=capture):
            supervisor.verify_inputs()

    def log_validator(self):
        path = MODULE.with_name('validate_dt_hip.py')
        spec = importlib.util.spec_from_file_location('validate_dt_hip_logs', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def native_log(self, steps, loss=.5):
        path = self.root / 'run/20261012.log.json'
        path.write_text(json.dumps({'env_info': 'metadata'}) + '\n' + ''.join(
            json.dumps(dict(mode='train', iter=i, loss=loss, grad_norm=1.2)) + '\n' for i in steps))
        (self.root / 'run/events.out.tfevents').write_bytes(b'not JSON and must not be parsed')
        return path

    def test_smoke_log_requires_all_twenty_finite_steps(self):
        validator = self.log_validator()
        self.native_log(range(1, 21))
        result = validator.audit_training_logs(self.root / 'run', 20, 1)
        self.assertEqual(result['unique_logged_iterations'], 20)
        self.assertTrue(result['finite_losses_and_grad_norm'])

    def test_missing_smoke_step_fails(self):
        validator = self.log_validator()
        self.native_log(range(2, 21))
        with self.assertRaisesRegex(RuntimeError, 'every optimizer step'):
            validator.audit_training_logs(self.root / 'run', 20, 1)

    def test_nonfinite_training_log_fails(self):
        validator = self.log_validator()
        self.native_log(range(1, 21), loss=float('nan'))
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
            validator.audit_training_logs(self.root / 'run', 20, 1)


if __name__ == '__main__':
    unittest.main()
