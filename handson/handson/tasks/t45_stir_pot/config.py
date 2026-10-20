from ..b320_common import B320TaskConfigBase


class TaskConfig(B320TaskConfigBase):
    task_id = "45"
    task_name = "t45_stir_pot"
    task_prompt = (
        "The removable lid is already set beside the open Dutch oven. Hold the "
        "pot fixed with the left hand while the right hand moves the spoon through "
        "six ordered checkpoints and over 650 degrees."
    )
    time_limit_seconds = 40
