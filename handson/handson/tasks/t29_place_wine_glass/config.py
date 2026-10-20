import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import SingleArmPolicyWrapper
from ..sim_teleop import (
    SingleArmTeleopConfig,
    SingleArmViveHandTeleopWrapper,
)


class TaskConfig(TaskConfigBase):
    task_id = "29"
    task_name = "t29_place_wine_glass"
    task_prompt = (
        'Place the YCB plastic wine glass upright at the center of the YCB '
        'plate, release it, and leave it stable. Only the final placement is '
        'evaluated; no minimum travel or lift history is required.'
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "wine_glass_pose",
        "plate_pose",
        "wine_glass_reset_pose",
        "plate_reset_pose",
        "task_stage",
        "ycb09_event_flags",
        "wine_glass_plate_local",
        "table_delta_height",
    ]
    teleop = SingleArmTeleopConfig()

    def get_native_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=True,
        sim_backend: Literal["isaacsim"] | None = None,
        **kwargs,
    ):
        backend = (
            sim_backend
            or os.environ.get("HANDSON_SIM_BACKEND")
            or "isaacsim"
        ).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                "t29_place_wine_glass supports only Isaac Sim 5.1; "
                f"received sim_backend={sim_backend!r}."
            )

        from ...sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

        # These rates are part of the task contract rather than caller-tunable
        # difficulty parameters.
        kwargs["time_limit"] = 60.0
        kwargs["control_dt"] = 1.0 / 30.0
        kwargs["physics_dt"] = 1.0 / 120.0
        env = HandsOnABCIsaacLabEnv(
            self.task_name,
            render_mode=render_mode,
            randomize=randomize,
            hz=30,
            **kwargs,
        )
        self.attach_prompt_overlay(env)
        if policy_mode:
            env = SingleArmPolicyWrapper(env)
        else:
            env = SingleArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
