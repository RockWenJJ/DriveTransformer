"""Monitoring must not start work or misreport a dead run as successful."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.large_vitb3_run import status, process_identity


class RunStatus(unittest.TestCase):
    def test_missing_run_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / 'not-created'
            with patch('subprocess.Popen', side_effect=AssertionError('Must not launch')):
                self.assertEqual(status(work)['status'], 'not_started')
            self.assertFalse(work.exists())

    def test_dead_process_is_not_complete_and_partial_log_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / 'run_state.json').write_text(json.dumps(dict(status='running',
                supervisor_pid=999999999, supervisor_start_ticks='1',
                trainer_pid=999999998, trainer_start_ticks='2',
                training_contract=dict(optimizer_updates=58700))))
            (work / 'sample.log.json').write_text(
                json.dumps(dict(mode='train', iter=100, time=2.0, loss=3.0)) + '\n{"iter":')
            result = status(work)
            self.assertEqual(result['status'], 'interrupted_or_unrecorded_exit')
            self.assertEqual(result['iteration'], 100)
            self.assertEqual(result['latest_loss'], 3.0)
            self.assertAlmostEqual(result['eta_hours'], 32.56, places=2)
            self.assertIsNone(result['latest_checkpoint'])
            self.assertIsNone(process_identity(999999999))


if __name__ == '__main__':
    unittest.main()
