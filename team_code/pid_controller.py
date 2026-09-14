"""Measured-speed PI control with calibrated braking; original steering is preserved."""
import json
from pathlib import Path

import numpy as np
from drivetransformer_contracts import target_speed_from_time_path


class DecouplePIDController:
    def __init__(self, speed_k_p=1., speed_k_i=0., speed_k_d=0., speed_n=20,
                 steer_k_p=1., steer_k_i=0., steer_k_d=0., steer_n=6,
                 **unused):
        # Preserve the agent call signature. Explicit, frozen settings define the
        # new longitudinal units; legacy agent gains do not silently carry over.
        self.config = json.loads(Path(__file__).with_name('controller_settings.json').read_text())
        self.policy_name = self.config['policy_name']
        self.speed_k_p = self.config['kp']
        self.speed_k_i = self.config['ki_per_second']
        self.speed_k_d = 0.
        self.feedforward = self.config.get('feedforward_effort', 0.)
        self.dt = self.config['dt_seconds']
        self.brake_k = self.config['brake_gain']
        self.brake_threshold = self.brake_b = 0.
        self.speed_n, self.steer_n = speed_n, steer_n
        self.steer_k_p, self.steer_k_i, self.steer_k_d = steer_k_p, steer_k_i, steer_k_d
        self.speed_window, self.steer_window = [], []
        self.integral_effort = 0.
        self.braking = False
        self.last_debug = {}

    def state(self):
        return dict(integral_effort=self.integral_effort, braking=self.braking)

    def _plan(self, desired_speed, current_speed):
        error = float(desired_speed-current_speed)
        stopped = abs(current_speed) <= self.config['stopped_speed_epsilon']
        if desired_speed == 0.:
            brake = (1. if stopped else
                     float(np.clip(self.brake_k*self.speed_k_p*abs(current_speed), 0., 1.)))
            return dict(branch='stop_hold' if stopped else 'stop_deceleration',
                        error=error, p_effort=self.speed_k_p*error,
                        integral_before=self.integral_effort, integral_after=0.,
                        feedforward_effort=0., raw_effort=-brake, throttle=0., brake=brake,
                        braking=True, anti_windup=False)

        # Feedforward supplies the measured steady propulsion effort. The PI
        # correction can cancel all of it on overspeed; there is no minimum
        # throttle or target-speed floor. Its total steady effort stays bounded.
        integral = float(np.clip(self.integral_effort+self.speed_k_i*error*self.dt,
                                 -self.feedforward, self.config['max_integral_effort']-self.feedforward))
        p_effort = self.speed_k_p*error
        raw = self.feedforward+p_effort+integral
        anti_windup = raw > 1. and error > 0.
        if anti_windup:
            integral = self.integral_effort
            raw = self.feedforward+p_effort+integral
        if raw >= 0.:
            throttle, brake, braking, branch = float(np.clip(raw,0.,1.)), 0., False, 'pi_drive'
        elif (current_speed < self.config['low_speed_coast_limit'] or
              current_speed-desired_speed < self.config.get('overspeed_brake_margin',0.)):
            # At low speed even tiny nonzero brake commands stopped this CARLA
            # vehicle abruptly in calibration. Let passive deceleration handle
            # a positive speed target; exact-zero requests use the stop branch.
            throttle, brake, braking, branch = 0., 0., False, 'coast_tracking'
        else:
            threshold = self.config['brake_effort_off'] if self.braking else self.config['brake_effort_on']
            braking = raw < -threshold
            throttle = 0.
            brake = float(np.clip(-raw*self.brake_k,0.,1.)) if braking else 0.
            branch = 'pi_brake' if braking else 'coast'
        return dict(branch=branch, error=error, p_effort=p_effort,
                    feedforward_effort=self.feedforward,
                    integral_before=self.integral_effort, integral_after=integral,
                    raw_effort=raw, throttle=throttle, brake=brake,
                    braking=braking, anti_windup=anti_windup)

    def longitudinal_branch(self, desired_speed, current_speed):
        # Diagnostic preview must not advance integral/hysteresis state.
        return self._plan(desired_speed, current_speed)['branch']

    def step(self, traj_fix_time, traj_fix_dist, current_speed):
        desired_speed = target_speed_from_time_path(traj_fix_time)
        control = self._plan(desired_speed,current_speed)
        self.integral_effort = control['integral_after']
        self.braking = control['braking']
        self.last_debug = dict(control, desired_speed=float(desired_speed),
                               current_speed=float(current_speed), dt=self.dt)
        self.speed_window = self.speed_window[-self.speed_n:]+[control['error']]
        throttle, brake = control['throttle'],control['brake']
        # Original steering calculation, including its window and indexing.
        point_idx_for_heading = int(np.clip(current_speed * 0.15, 1, len(traj_fix_dist)-1))
        heading_error = np.arctan2(traj_fix_dist[point_idx_for_heading, 1], traj_fix_dist[point_idx_for_heading, 0])
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        self.steer_window = self.steer_window[-self.steer_n:] + [heading_error]
        steer_integral = np.mean(self.steer_window)
        steer_derivative = self.steer_window[-1] - self.steer_window[-2] if len(self.steer_window) > 1 else 0
        steer = self.steer_k_p * heading_error + self.steer_k_d * steer_derivative + self.steer_k_i * steer_integral
        steer = np.clip(steer, -1.0, 1.0)
        return steer, throttle, brake
