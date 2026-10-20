"""Shared collection configuration for user-provided native USD articulations."""

from __future__ import annotations

import math
import os
from typing import Literal

from .config import TaskConfigBase
from .d_long_horizon_config import DLongHorizonTaskConfigBase
from .obs_adapters import HandsOnObsAdapter
from .policy_wrappers import DualArmPolicyWrapper
from .sim_teleop import DualArmViveHandTeleopWrapper


LW_CONTROL_HZ = 30
LW_PHYSICS_HZ = 120
LW_TIME_LIMIT_SECONDS = 90.0
LW_HAND_TYPE = "wuji1"
LW_ARM_TYPE = "rokae"
LW_PROPRIO_KEYS = [
    "tcp_pose",
    "gripper_pose",
    "lw_joint_state",
    "lw_contact_state",
    "lw_task_state",
    "lw_root_pose",
    "lw_role_poses",
    "table_delta_height",
]


def _require_fixed_rate(kwargs: dict, key: str, expected: float) -> None:
    requested = float(kwargs.pop(key, expected))
    if not math.isclose(requested, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            f"LW native-articulation tasks require {key}={expected!r}; "
            f"got {requested!r}."
        )
    kwargs[key] = expected


class LWNativeArticulationTaskConfigBase(TaskConfigBase):
    """Build a fixed-rate Isaac Sim 5.1/PhysX dual-arm collection task."""

    time_limit_seconds = LW_TIME_LIMIT_SECONDS
    proprio_keys = LW_PROPRIO_KEYS
    # Match the responsive D08 collection contract, including its packet-level
    # tracker safeguards and unrestricted steady-state Wuji targets.
    teleop = DLongHorizonTaskConfigBase.teleop

    def get_native_environment(
        self,
        policy_mode: bool = False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize: bool = True,
        sim_backend: Literal["isaacsim"] | None = None,
        **kwargs,
    ):
        backend = (
            sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")
        ).strip().lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                f"{self.task_id} requires the Isaac Sim 5.1 collection runtime; "
                f"sim_backend={sim_backend!r} is unsupported."
            )

        requested_physics = str(kwargs.pop("physics_backend", "physx")).strip().lower()
        if requested_physics != "physx":
            raise ValueError(
                f"{self.task_id} requires PhysX and forbids Newton/Isaac Sim 6; "
                f"physics_backend={requested_physics!r} is unsupported."
            )
        requested_hand = str(kwargs.pop("hand_type", LW_HAND_TYPE)).strip().lower()
        requested_arm = str(kwargs.pop("arm_type", LW_ARM_TYPE)).strip().lower()
        if requested_hand != LW_HAND_TYPE or requested_arm != LW_ARM_TYPE:
            raise ValueError(
                f"{self.task_id} requires arm_type={LW_ARM_TYPE!r} with "
                f"hand_type={LW_HAND_TYPE!r}; got arm_type={requested_arm!r}, "
                f"hand_type={requested_hand!r}."
            )

        _require_fixed_rate(kwargs, "hz", float(LW_CONTROL_HZ))
        _require_fixed_rate(kwargs, "control_dt", 1.0 / LW_CONTROL_HZ)
        _require_fixed_rate(kwargs, "physics_dt", 1.0 / LW_PHYSICS_HZ)
        kwargs.setdefault("time_limit", float(self.time_limit_seconds))
        kwargs["physics_backend"] = "physx"
        kwargs["hand_type"] = LW_HAND_TYPE
        kwargs["arm_type"] = LW_ARM_TYPE

        from ..sim.isaaclab.lw_native_articulation_env import (
            HandsOnLWNativeArticulationIsaacLabEnv,
        )

        env = HandsOnLWNativeArticulationIsaacLabEnv(
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


__all__ = [
    "LW_ARM_TYPE",
    "LW_CONTROL_HZ",
    "LW_HAND_TYPE",
    "LW_PHYSICS_HZ",
    "LW_PROPRIO_KEYS",
    "LW_TIME_LIMIT_SECONDS",
    "LWNativeArticulationTaskConfigBase",
]
