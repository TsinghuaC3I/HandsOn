from ..b320_common import B320TaskConfigBase


class TaskConfig(B320TaskConfigBase):
    task_id = "30"
    task_name = "t30_pour_seasonings"
    task_prompt = (
        "The salt jar and cyan target are randomized within the left-hand "
        "comfort zone; the pepper jar and orange target are randomized within "
        "the right-hand comfort zone. Use the assigned hands, tilt both caps "
        "downward, and sprinkle their blue water-flow proxies into the matching "
        "targets with a synchronized overlap. Complete before either jar "
        "accumulates more than 4 seconds of pouring or the scene resets."
    )
    time_limit_seconds = 16
