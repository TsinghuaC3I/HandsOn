from ..b320_common import B320TaskConfigBase


class TaskConfig(B320TaskConfigBase):
    task_id = "32"
    task_name = "t32_press_soap_pump"
    task_prompt = (
        "Place the dish sponge under the dish-soap bottle, hold the bottle with the "
        "right hand, align its outlet over the sponge, and physically "
        "depress the visible pump head with the right hand."
    )
    time_limit_seconds = 90
