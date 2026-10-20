from ..b320_common import B320TaskConfigBase


class TaskConfig(B320TaskConfigBase):
    task_id = "15"
    task_name = "t15_open_drawers"
    task_prompt = (
        "Use each assigned hand to pull the highlighted drawer on its side beyond "
        "60 percent travel; each side independently selects one of three drawers "
        "per episode, and both thresholds must be crossed within 1.2 seconds."
    )
    b320_runtime = "bc017"
    b320_proprio_keys = [
        *B320TaskConfigBase.proprio_keys,
        "bc017_drawer_state",
        "bc017_contact_state",
        "bc017_stage_state",
        "bc017_terminal_state",
        "bc017_hidden_state",
    ]
