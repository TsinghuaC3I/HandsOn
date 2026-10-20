import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "28"
    task_name = "t28_place_tableware"
    task_prompt = (
        "Follow the sampled single-step table-setting instruction. Move only "
        "the named YCB tableware object and leave the plate and distractors in place."
    )
    # Object/goal truth stays in recording-only ycb06_* fields.  The policy
    # receives only robot proprioception, cameras, and the language prompt.
    proprio_keys = ["tcp_pose", "gripper_pose", "table_delta_height"]
    teleop = BimanualTeleopConfig(
        filter_alpha=0.55,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.045,
        filter_max_rot_step_deg=8.0,
        velocity_assist=False,
        max_tracker_delta_m=0.60,
    )

    def get_native_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=True,
        sim_backend: Literal["mujoco", "isaacsim"] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("time_limit", 90.0)
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("control_dt", 1.0 / 30.0)
        kwargs.setdefault("physics_dt", 1.0 / 120.0)
        backend = (
            sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")
        ).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                "YCB06 uses the native Isaac Sim 5.1 collection path and "
                f"requires sim_backend='isaacsim', got {sim_backend!r}."
            )

        from ...sim.isaaclab.ycb06_collection_env import (
            HandsOnYCB06IsaacLabEnv,
        )

        env = HandsOnYCB06IsaacLabEnv(
            render_mode=render_mode,
            randomize=randomize,
            **kwargs,
        )
        self.attach_prompt_overlay(env)
        if policy_mode:
            env = DualArmPolicyWrapper(env)
        else:
            env = DualArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
