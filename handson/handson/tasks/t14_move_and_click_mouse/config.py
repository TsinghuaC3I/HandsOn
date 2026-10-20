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
    task_id = "14"
    task_name = "t14_move_and_click_mouse"
    task_prompt = (
        "Move the mouse onto the mouse pad without pressing either button, "
        "then click the left mouse button."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "mouse_ori_pose",
        "mousepad_ori_pose",
        "table_delta_height",
    ]
    teleop = SingleArmTeleopConfig(
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

            env = HandsOnABCIsaacLabEnv(self.task_name, render_mode=render_mode, randomize=randomize, hz=30, **kwargs)
        else:
            raise ValueError(f"Unsupported t14_move_and_click_mouse sim_backend={sim_backend!r}.")
        self.attach_prompt_overlay(env)
        if policy_mode:
            env = SingleArmPolicyWrapper(env)
        else:
            env = SingleArmViveHandTeleopWrapper(env, self.teleop)
        env = HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)
        return env

    def process_demos(self, demo):
        return demo
