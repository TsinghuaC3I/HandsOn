from ..b320_common import B320TaskConfigBase


class TaskConfig(B320TaskConfigBase):
    task_id = "20"
    task_name = "t20_pass_and_orient_pen"
    task_prompt = (
        "Touch the pen anywhere with the right hand only, then with both hands "
        "at the same time, then with the left hand only, and finish with the "
        "left hand holding the pen tip-up at 90 +/- 2 degrees from horizontal; "
        "after both hands have touched it, do not drop the pen."
    )
    time_limit_seconds = 32
