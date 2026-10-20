from ..d_long_horizon_config import DLongHorizonTaskConfigBase


class TaskConfig(DLongHorizonTaskConfigBase):
    task_id = "31"
    task_name = "t31_prepare_coffee"
    task_prompt = (
        "Open either cabinet door, remove the mug, then close both doors. "
        "Place the mug upright on the coffee-machine base with its opening "
        "directly under the circular outlet. Press the bright-green start "
        "button and keep the mug aligned for the full 2-second brew. Place "
        "the filled mug upright and stable on the serving tray, then release "
        "it with both hands."
    )
    time_limit_seconds = 200.0
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "mug_pose",
        "cabinet_pose",
        "coffee_machine_pose",
        "tray_pose",
        "coffee_state",
        "workflow_progress",
        "table_delta_height",
    ]
