import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import (
    BimanualTeleopConfig,
    DualArmViveHandTeleopWrapper,
)


class TaskConfig(TaskConfigBase):
    task_id = "08"
    task_name = "t08_hammer_nail"
    task_prompt = (
        "Pick up the wood block with the left hand, keep the vertical nail inserted "
        "in the block, then use the right hand to pick up the hammer and strike the "
        "nail head so the nail sinks slightly into the block."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "hammer_ori_pose",
        "nail_ori_pose",
        "wood_ori_pose",
        "nail_depth",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        filter_alpha=0.55,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.06,
        filter_max_rot_step_deg=10.0,
    )

    def get_native_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=True,
        sim_backend: Literal["isaacsim"] | None = None,
        **kwargs,
    ):
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend == "isaacsim":
            from ...sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

            env = HandsOnABCIsaacLabEnv(self.task_name, render_mode=render_mode, randomize=randomize, hz=30, time_limit=120.0, **kwargs)
        else:
            raise ValueError(f"Unsupported t08_hammer_nail sim_backend={sim_backend!r}.")
        self.attach_prompt_overlay(env)
        if policy_mode:
            env = DualArmPolicyWrapper(env)
        else:
            env = DualArmViveHandTeleopWrapper(env, self.teleop)
        env = HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)
        return env

    def process_demos(self, demo):
        return demo
