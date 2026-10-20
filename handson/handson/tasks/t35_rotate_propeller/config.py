from __future__ import annotations

import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "35"
    task_name = "t35_rotate_propeller"
    task_prompt = (
        "Rotate the YCB airplane propeller. Accumulate 1080 degrees of absolute "
        "angular travel: each 360 degrees counts as one turn, in either direction. "
        "Finger contact and passive coasting both count; three turns complete the task."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "airplane_pose",
        "airplane_linear_velocity",
        "airplane_angular_velocity",
        "propeller_joint_state",
        "contact_state",
        "flick_state",
        "wheel_support_state",
        "airplane_displacement_state",
        "release_stability_state",
        "event_flags",
        "event_latch_state",
        "reset_state",
        "contact_validity",
        "ycb03_replay_history",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        filter_alpha=0.58,
        filter_deadband_m=0.0008,
        filter_rot_deadband_deg=0.4,
        filter_max_step_m=0.045,
        filter_max_rot_step_deg=8.0,
    )

    def get_native_environment(
        self,
        policy_mode: bool = False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize: bool = True,
        sim_backend: Literal["mujoco", "isaacsim"] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("time_limit", 120.0)
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
                f"{self.task_id} uses the Isaac Sim 5.1 / IsaacLab 2.3.2 "
                "collection runtime and requires sim_backend='isaacsim', "
                f"got {sim_backend!r}."
            )

        from ...sim.isaaclab.ycb03_airplane_env import (
            HandsOnYCB03AirplaneIsaacLabEnv,
        )

        env = HandsOnYCB03AirplaneIsaacLabEnv(
            self.task_name,
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
