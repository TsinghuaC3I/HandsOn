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
    task_id = "EVAL-Gen03"
    task_name = "eval_gen03_orient_bottle_upright"
    semantic_task_name = "eval_gen03_orient_bottle_upright"
    task_prompt = (
        "Pick up the yellow mustard bottle and place it upright on the table with its bottom "
        "facing down. Release the bottle and keep it upright for 0.5 seconds."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "target_object_pose",
        "receptacle_pose",
        "object_poses",
        "target_selector",
        "table_delta_height",
    ]
    teleop = SingleArmTeleopConfig(
        filter_alpha=0.55,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.060,
        filter_max_rot_step_deg=10.0,
    )

    def get_native_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=True,
        sim_backend: Literal["mujoco", "isaacsim", "isaacsim_newton"] | None = None,
        **kwargs,
    ):
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend in {"newton", "nvidia_newton", "isaac_newton"}:
            backend = "isaacsim_newton"
        if backend in {"isaacsim", "isaacsim_newton"}:
            from ...sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

            kwargs.setdefault(
                "physics_backend",
                "newton" if backend == "isaacsim_newton" else "physx",
            )
            env = HandsOnABCIsaacLabEnv(self.task_name, render_mode=render_mode, randomize=randomize, hz=30, **kwargs)
        else:
            raise ValueError(f"Unsupported eval_gen03_orient_bottle_upright sim_backend={sim_backend!r}.")
        self.attach_prompt_overlay(env)
        if policy_mode:
            env = SingleArmPolicyWrapper(env)
        else:
            env = SingleArmViveHandTeleopWrapper(env, self.teleop)
        env = HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)
        return env

    def process_demos(self, demo):
        return demo
