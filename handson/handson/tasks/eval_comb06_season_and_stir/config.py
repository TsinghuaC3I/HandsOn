from ..b320_common import B320TaskConfigBase
from ...sim.isaaclab.eval_comb06_season_and_stir_scene import PROMPT, TASK_ID, TASK_NAME


class TaskConfig(B320TaskConfigBase):
    task_id = TASK_ID
    task_name = TASK_NAME
    task_prompt = PROMPT
    time_limit_seconds = 90
    b320_runtime = "xy6"
