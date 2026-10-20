from __future__ import annotations

import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper
from ...sim.isaaclab.b320_bimanual_scene import B320_CONTROL_HZ, B320_PHYSICS_HZ


class TaskConfig(TaskConfigBase):
    task_id = "13"
    task_name = "t13_load_cans"
    task_prompt = (
        "Pick up the left and right tomato-soup cans with their assigned hands, "
        "cross-load each can into the opposite-side bin partition, then release both cans "
        "within the synchronization window and leave them stable."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "left_can_pose",
        "right_can_pose",
        "open_bin_pose",
        "partition_centers",
        "bin_load_metrics",
        "bc012_task_state",
        "object_poses",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        pose_scale=3.00,
        max_tracker_delta_m=0.55,
        filter_alpha=0.55,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.050,
        filter_max_rot_step_deg=8.0,
    )

    def get_native_environment(
        self,
        policy_mode: bool = False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize: bool = True,
        sim_backend: Literal["isaacsim"] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("time_limit", 30.0)
        kwargs.setdefault("hz", B320_CONTROL_HZ)
        kwargs.setdefault("control_dt", 1.0 / B320_CONTROL_HZ)
        kwargs.setdefault("physics_dt", 1.0 / B320_PHYSICS_HZ)
        # The task contract records all canonical RGB streams at 480x640
        # (height x width). Keep the wrist override instance-local so loading
        # this task cannot change another task's process-wide camera profile.
        kwargs.setdefault("render_resolution", (480, 640))
        kwargs.setdefault("wrist_camera_output_size", (640, 480))
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                f"{self.task_id} requires Isaac Sim 5.1 PhysX with "
                f"sim_backend='isaacsim', got {sim_backend!r}."
            )

        from ...sim.isaaclab.bc012_env import BC012IsaacLabEnv

        env = BC012IsaacLabEnv(
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
