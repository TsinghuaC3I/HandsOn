from ..lw_native_articulation_config import LWNativeArticulationTaskConfigBase


class TaskConfig(LWNativeArticulationTaskConfigBase):
    task_id = "18"
    task_name = "t18_open_micropipette_box"
    task_prompt = (
        "The micropipette box is fixed after reset at a randomized tabletop "
        "pose and yaw. Open its real hinged lid from the closed position to at "
        "least 75 degrees. Success is immediate as soon as the physical lid "
        "joint reaches that opening; either hand may be used, and no assigned "
        "contact, release, or stable-hold condition is required."
    )
