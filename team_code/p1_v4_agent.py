"""Read-only v4 control telemetry; inherited model inputs and controls are unchanged."""
import json
from team_code.drivetransformer_b2d_agent import DriveTransformerAgent


def get_entry_point():
    return 'P1V4Agent'


class P1V4Agent(DriveTransformerAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        assert self.controller.policy_name == 'pi_feedforward_v4'
        assert self.save_path is not None
        (self.save_path / 'controller_settings.json').write_text(
            json.dumps(self.controller.config, indent=2) + '\n')
        self.control_log = (self.save_path / 'controller_v4.jsonl').open('x', buffering=1)

    def run_step(self, input_data, timestamp):
        control = super().run_step(input_data, timestamp)
        # Log every returned control; PI debug updates only on 10 Hz inference.
        self.control_log.write(json.dumps(dict(
            step=self.step, timestamp=float(timestamp), inference=self.step % 2 == 0,
            initial_wait=self.step <= 20, policy=self.controller.policy_name,
            controller=self.controller.last_debug,
            returned=dict(steer=float(control.steer), throttle=float(control.throttle),
                          brake=float(control.brake)),
        ), allow_nan=False) + '\n')
        return control

    def destroy(self):
        if hasattr(self, 'control_log'):
            self.control_log.close()
        super().destroy()
