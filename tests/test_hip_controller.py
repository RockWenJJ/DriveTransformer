import unittest
import numpy as np
from hip_planning_contracts import hip_control_paths
from drivetransformer_contracts import target_speed_from_time_path
from team_code.pid_controller import DecouplePIDController


class ControllerContracts(unittest.TestCase):
    def test_velocity_window_and_forward_axis(self):
        temporal = np.stack([np.zeros(6), np.arange(1, 7)*.2*5], -1)
        spatial = np.stack([np.zeros(6), np.arange(1, 7)*2.], -1)
        t, s, detail = hip_control_paths(temporal, spatial, temporal, 2)
        self.assertAlmostEqual(target_speed_from_time_path(t), 5.)
        np.testing.assert_allclose(t[:, 1], 0.)
        np.testing.assert_allclose(s[:12, 0], np.arange(1, 13))
        self.assertFalse(detail['style_fallback'])

    def test_stop_and_turn_sign(self):
        zero = np.zeros((6, 2))
        t, s, _ = hip_control_paths(zero, zero, zero, 0)
        steer, throttle, brake = DecouplePIDController().step(t, s, 0.)
        self.assertEqual((throttle, brake), (0., 1.))
        spatial = np.stack([np.arange(1, 7)*.3, np.arange(1, 7)*2.], -1)
        t, s, _ = hip_control_paths(zero, spatial, zero, 0)
        self.assertGreater(DecouplePIDController().step(t, s, 0.)[0], 0.)
        spatial[:, 0] *= -1
        _, s, _ = hip_control_paths(zero, spatial, zero, 0)
        self.assertLess(DecouplePIDController().step(t, s, 0.)[0], 0.)

    def test_style_mismatch_falls_back_without_extrapolation(self):
        temporal = np.stack([np.zeros(6), np.arange(1, 7)*.2], -1)
        t, _, detail = hip_control_paths(temporal, temporal, temporal*10, 2)
        self.assertTrue(detail['style_fallback'])
        self.assertAlmostEqual(target_speed_from_time_path(t), 1.)
        np.testing.assert_allclose(t[-1], [1.2, 0.])


if __name__ == '__main__':
    unittest.main()
