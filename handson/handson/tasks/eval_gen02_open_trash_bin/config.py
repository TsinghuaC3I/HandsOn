from ..lw_native_articulation_config import LWNativeArticulationTaskConfigBase


class TaskConfig(LWNativeArticulationTaskConfigBase):
    task_id = "EVAL-Gen02"
    task_name = "eval_gen02_open_trash_bin"
    task_prompt = (
        "The trash bin is fixed after reset at a randomized tabletop pose and yaw. "
        "Open its hinged lid from closed to at least 75 degrees. Success is immediate "
        "when the physical lid reaches that opening. Either hand may be used; "
        "no assigned contact, release, or stable hold is required."
    )
