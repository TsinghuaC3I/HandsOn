from ..lw_native_articulation_config import LWNativeArticulationTaskConfigBase


class TaskConfig(LWNativeArticulationTaskConfigBase):
    task_id = "49"
    task_name = "t49_unfold_ruler"
    task_prompt = (
        "Pull both outer ruler segments through their real hinges until all "
        "three segments are straight. The ruler starts at a randomized tabletop "
        "pose and remains free to move; either hand may be used. Success is "
        "immediate when both absolute hinge angles reach 175–180 degrees; "
        "+180 and −180 are equivalent, and no release or stability hold is "
        "required."
    )
