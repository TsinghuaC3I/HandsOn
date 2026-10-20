from ..b320_common import B320TaskConfigBase


class TaskConfig(B320TaskConfigBase):
    task_id = "05"
    task_name = "t05_fill_cup"
    task_prompt = (
        "Lift the randomized white ceramic mug under the faucet outlet with the "
        "left hand and keep its local +Z axis within 15 degrees, inclusive, of "
        "world +Z. Pull the right lever outward toward the robots along its "
        "source-aligned hinge (with a world -X component) beyond 70 percent of "
        "its 45-degree opening travel; left-right temperature motion does not "
        "count. Keep the faucet open while the visible stream enters the mug "
        "for 15 consecutive control frames (0.5 seconds), then place the "
        "upright mug on the marked counter pad outside the sink and close the "
        "faucet."
    )
    time_limit_seconds = 22
