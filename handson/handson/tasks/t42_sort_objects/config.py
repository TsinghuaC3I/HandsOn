from ..d_long_horizon_config import DLongHorizonTaskConfigBase


class TaskConfig(DLongHorizonTaskConfigBase):
    task_id = "42"
    task_name = "t42_sort_objects"
    task_prompt = (
        "Find six targets among the clutter and sort boxes, cans, and dishes "
        "into their marked baskets. Transfer the far-right target through both "
        "hands before placing it."
    )
    time_limit_seconds = 210.0
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "target_poses",
        "basket_poses",
        "distractor_poses",
        "classification_progress",
        "table_delta_height",
    ]
