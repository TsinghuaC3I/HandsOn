from __future__ import annotations

import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "EVAL-Flex02"
    task_name = "eval_flex02_fold_towel_twice"
    task_prompt = (
        "Fold the flat towel in half twice, in any directions. The two folds "
        "may be in the same direction or different directions. Release the "
        "folded towel and leave it stable on the table."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "towel_pose",
        "towel_keypoints",
        "towel_aabb",
        "fold_metrics",
        "rotation_metrics",
        "task_phase",
        "fold_event_state",
        "towel_initial_state",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        # The tracker pose is the command.  Keep only a generous finite
        # envelope so this task does not add EMA/deadband/step latency on top
        # of the physical replay clock.
        filter_alpha=1.0,
        filter_deadband_m=0.0,
        filter_rot_deadband_deg=0.0,
        filter_max_step_m=1.0,
        filter_max_rot_step_deg=180.0,
        velocity_assist=False,
        hand_filter_max_step_rad=None,
        hand_takeover_max_step_rad=None,
        hand_takeover_max_control_steps=0,
        reset_rebase_delay_seconds=0.0,
    )

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
        ).strip().lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                f"{self.task_id} requires sim_backend='isaacsim'; "
                f"got {sim_backend!r}."
            )

        requested_physics = (
            str(kwargs.pop("physics_backend", "physx")).strip().lower()
        )
        if requested_physics != "physx":
            raise ValueError(
                f"{self.task_id} requires the Isaac Sim 5.1 PhysX runtime; "
                f"physics_backend={requested_physics!r} is unsupported."
            )
        requested_hand = str(kwargs.pop("hand_type", "wuji1")).strip().lower()
        requested_arm = str(kwargs.pop("arm_type", "rokae")).strip().lower()
        if requested_hand != "wuji1" or requested_arm != "rokae":
            raise ValueError(
                f"{self.task_id} requires two ROKAE arms with Wuji1 hands; "
                f"got arm_type={requested_arm!r}, hand_type={requested_hand!r}."
            )

        kwargs.setdefault("time_limit", 180.0)
        kwargs.setdefault("hz", 30)
        # The operator viewport is only a control aid.  Canonical cameras are
        # rendered by the deferred post-render pass for formal collection.
        kwargs.setdefault("render_resolution", (480, 480))
        kwargs["physics_backend"] = "physx"
        kwargs["hand_type"] = "wuji1"
        kwargs["arm_type"] = "rokae"

        from ...sim.isaaclab.eval_flex02_fold_towel_twice_env import (
            HandsOnFlexibleTowelDoubleFoldRotate90IsaacLabEnv,
        )

        env = HandsOnFlexibleTowelDoubleFoldRotate90IsaacLabEnv(
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
