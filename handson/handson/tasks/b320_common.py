"""Shared task configuration for the B320 dual-hand benchmark."""

from __future__ import annotations

import os
from typing import Literal

from .config import TaskConfigBase
from .d_long_horizon_config import DLongHorizonTaskConfigBase
from .obs_adapters import HandsOnObsAdapter
from .policy_wrappers import DualArmPolicyWrapper
from .sim_teleop import DualArmViveHandTeleopWrapper


_RIGID_PROPRIO_KEYS = [
    "tcp_pose",
    "gripper_pose",
    "left_role_pose",
    "right_role_pose",
    "fixture_pose",
    "goal_pose",
    "task_metrics",
    "contact_metrics",
    "stage_state",
    "articulation_q",
    "table_delta_height",
]
_TOWEL_PROPRIO_KEYS = [
    "tcp_pose",
    "gripper_pose",
    "towel_pose",
    "towel_corners",
    "target_bounds",
    "spread_metrics",
    "grasp_state",
    "garment_initial_state",
    "table_delta_height",
]


class B320TaskConfigBase(TaskConfigBase):
    task_id = "B320"
    task_name = ""
    task_prompt = ""
    time_limit_seconds = 30
    towel_task = False
    proprio_keys = _RIGID_PROPRIO_KEYS
    # Keep every B320 task on the same responsive operator-input contract as
    # D08.  The formal collection UI supplies alpha=1 with zero palm
    # deadbands/step limits; the shared object also inherits the common
    # tracker fault hold and unrestricted steady-state Wuji targets.
    teleop = DLongHorizonTaskConfigBase.teleop

    def get_native_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=True,
        sim_backend: Literal["mujoco", "isaacsim"] | None = None,
        **kwargs,
    ):
        backend = (
            sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")
        ).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                f"{self.task_id} requires sim_backend='isaacsim'; got {backend!r}"
            )
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("time_limit", float(self.time_limit_seconds))
        kwargs.setdefault("hand_type", "wuji1")
        kwargs.setdefault("arm_type", "rokae")
        if self.towel_task:
            from ..sim.isaaclab.b320_towel_env import (
                HandsOnB320TowelIsaacLabEnv,
            )

            env = HandsOnB320TowelIsaacLabEnv(
                self.task_name,
                render_mode=render_mode,
                randomize=randomize,
                **kwargs,
            )
            active_proprio_keys = list(_TOWEL_PROPRIO_KEYS)
        else:
            b320_runtime = getattr(self, "b320_runtime", "rigid")
            if b320_runtime == "xy6":
                from ..sim.isaaclab.eval_comb06_season_and_stir_env import (
                    HandsOnXY6IsaacLabEnv as B320Environment,
                )
            elif b320_runtime == "bc017":
                from ..sim.isaaclab.b_eval_setting import setting_for
                selection = setting_for(self.task_name)
                setting = selection['setting_id'] if selection else 'large_six_contact'
                if setting == 'small_upper_d6':
                    from ..sim.isaaclab.b320_bc017_small_d6_env import HandsOnB320BC017IsaacLabEnv as B320Environment
                elif setting == 'large_upper_contact':
                    from ..sim.isaaclab.b320_bc017_large_upper_contact_env import HandsOnB320BC017IsaacLabEnv as B320Environment
                elif setting == 'large_six_contact':
                    from ..sim.isaaclab.b320_bc017_env import HandsOnB320BC017IsaacLabEnv as B320Environment
                else:
                    raise ValueError('Unsupported BC017 setting: ' + setting)
            elif b320_runtime == "bc021":
                from ..sim.isaaclab.b320_bc021_env import (
                    HandsOnB320BC021IsaacLabEnv as B320Environment,
                )
            else:
                from ..sim.isaaclab.b320_bimanual_env import (
                    HandsOnB320BimanualIsaacLabEnv as B320Environment,
                )

            env = B320Environment(
                self.task_name,
                render_mode=render_mode,
                randomize=randomize,
                **kwargs,
            )
            active_proprio_keys = list(
                getattr(self, "b320_proprio_keys", _RIGID_PROPRIO_KEYS)
            )
        self.proprio_keys = active_proprio_keys
        self.attach_prompt_overlay(env)
        if policy_mode:
            env = DualArmPolicyWrapper(env)
        else:
            env = DualArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=active_proprio_keys)

    def process_demos(self, demo):
        return demo


__all__ = ["B320TaskConfigBase"]
