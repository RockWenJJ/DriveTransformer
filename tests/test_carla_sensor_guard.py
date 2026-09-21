"""The opt-in native guard must fail before launching an unverified simulator."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

LAUNCHER = Path(__file__).resolve().parents[1] / 'tools/carla_sensor_guard/CarlaUE4.sh'

class NativeGuardSafety(unittest.TestCase):
    def test_original_installation_must_be_explicit(self):
        env = dict(os.environ)
        env.pop('CARLA_ORIGINAL_ROOT', None)
        run = subprocess.run(['sh', str(LAUNCHER)], env=env, capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('CARLA_ORIGINAL_ROOT', run.stderr)

    def test_unknown_binary_never_reaches_debugger(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'unverified simulator build')
            marker = root / 'debugger-was-called'
            debugger = root / 'debugger'
            debugger.write_text('#!/bin/sh\ntouch "'+str(marker)+'"\n')
            debugger.chmod(0o755)
            env = dict(os.environ, CARLA_ORIGINAL_ROOT=str(root), GDB=str(debugger))
            run = subprocess.run(['sh', str(LAUNCHER)], env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 42)
            self.assertFalse(marker.exists())
            self.assertIn('Refusing unmatched CARLA binary', run.stderr)

if __name__ == '__main__':
    unittest.main()
