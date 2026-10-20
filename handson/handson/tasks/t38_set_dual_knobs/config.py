from ..b320_common import B320TaskConfigBase


class TaskConfig(B320TaskConfigBase):
    task_id = "38"
    task_name = "t38_set_dual_knobs"
    task_prompt = (
        "Use the left hand on the left knob and the right hand on the right knob, "
        "start both rotations together, set both displayed targets, release both "
        "knobs, and keep the settings stable."
    )
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
