from ..b320_common import B320TaskConfigBase


class TaskConfig(B320TaskConfigBase):
    task_id = "01"
    task_name = "t01_align_blocks_in_midair"
    task_prompt = (
        "Lift both assigned blocks with their assigned hands, then mate their "
        "marked faces in midair and hold the physical contact stable for 0.4 seconds."
    )
    time_limit_seconds = 60
    b320_runtime = "bc021"
    b320_proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "table_delta_height",
        "bc021_block_pose",
        "bc021_marked_face_state",
        "bc021_contact_state",
        "bc021_stage_state",
        "bc021_terminal_state",
        "bc021_hidden_state",
    ]
