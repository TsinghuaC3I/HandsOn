from dataclasses import replace
import os

from ..b320_common import B320TaskConfigBase
from ...sim.isaaclab.gen6_audio_mixer_scene import PROMPT, TASK_NAME


class TaskConfig(B320TaskConfigBase):
    task_id = "EVAL-Gen05"
    task_name = TASK_NAME
    task_prompt = PROMPT
    time_limit_seconds = 40
    b320_proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "left_role_pose",
        "right_role_pose",
        "fixture_pose",
        "goal_pose",
        "knob_joint_state",
        "knob_target_state",
        "knob_contact_state",
        "task_validity",
        "sync_task_state",
        "task_terminal_state",
        "task_metrics",
        "table_delta_height",
    ]

    def get_native_environment(self, *args, **kwargs):
        port_base = os.environ.get("HANDSON_GEN6_TELEOP_PORT_BASE", "").strip()
        if port_base:
            port = int(port_base)
            if not 1 <= port <= 65531:
                raise ValueError("GEN6 teleoperation port base must be in 1..65531")
            # Optional endpoint isolation for an independent development UI.
            # The shipped operator input contract and default ports are unchanged.
            self.teleop = replace(
                B320TaskConfigBase.teleop, vive_udp_port=port,
                right_hand_udp_port=port + 2, left_hand_udp_port=port + 4,
            )
        return super().get_native_environment(*args, **kwargs)
