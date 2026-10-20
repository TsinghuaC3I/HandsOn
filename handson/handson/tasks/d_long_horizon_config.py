"""Shared configuration for the native-Isaac D06--D19 task family."""

from __future__ import annotations

import os
from typing import Literal

from .config import TaskConfigBase
from .obs_adapters import HandsOnObsAdapter
from .policy_wrappers import DualArmPolicyWrapper
from .sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class DLongHorizonTaskConfigBase(TaskConfigBase):
    """Create a long-horizon D task with the standard dual-arm wrappers."""

    time_limit_seconds = 210.0
    # Keep the ordinary endpoint smoother and per-frame limits explicit, like
    # every other collection task. Tracker fault holds and original-frame
    # recovery are inherited from the shared teleoperation defaults.
    teleop = BimanualTeleopConfig(
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
        sim_backend: Literal["mujoco", "isaacsim"] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("time_limit", float(self.time_limit_seconds))
        kwargs.setdefault("hz", 30)
        backend = (
            sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")
        ).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                f"{self.task_id} uses the native IsaacSim collection path and "
                f"requires sim_backend='isaacsim', got {sim_backend!r}."
            )

        from ..sim.isaaclab.d_long_horizon_env import (
            HandsOnDLongHorizonIsaacLabEnv,
        )

        env = HandsOnDLongHorizonIsaacLabEnv(
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
