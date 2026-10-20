from __future__ import annotations

import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "06"
    task_name = "t06_fit_device_cover"
    task_prompt = (
        "At each reset device_body and device_cover receive independently "
        "randomized tabletop positions and yaw directions, and device_body "
        "remains free to move under contact. Use both hands on opposite sides "
        "of device_cover and carry it together above device_body. One hand may "
        "arrive up to two seconds before the other. Keep both side contacts "
        "during the effective transport, align the matching interface, and "
        "seat the cover."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "device_body_pose",
        "device_cover_pose",
        "cover_to_body",
        "alignment_metrics",
        "contact_metrics",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        filter_alpha=0.55,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.055,
        filter_max_rot_step_deg=9.0,
    )

    def get_native_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=True,
        sim_backend: Literal["mujoco", "isaacsim"] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("time_limit", 120.0)
        kwargs.setdefault("hz", 30)
        backend = (
            sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")
        ).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                "G02 uses the migrated SceneSmith USD workbench and PhysX "
                f"contact truth, and requires sim_backend='isaacsim', got {sim_backend!r}."
            )

        from ...sim.isaaclab.t06_fit_device_cover_env import (
            HandsOnG02RepairCloseDeviceIsaacLabEnv,
        )

        env = HandsOnG02RepairCloseDeviceIsaacLabEnv(
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
