from ..b320_common import B320TaskConfigBase


class TaskConfig(B320TaskConfigBase):
    task_id = "10"
    task_name = "t10_lift_carton"
    task_prompt = (
        "Support opposite lower carton sides with both hands, lift the entire "
        "carton clear of the table without either side exceeding 12 N, and "
        "keep it continuously airborne for 3 seconds."
    )
    time_limit_seconds = 22
