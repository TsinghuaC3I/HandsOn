from __future__ import annotations

import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class AS019DeferredCameraObsAdapter(HandsOnObsAdapter):
    """Capture AS019 cameras after the inner PhysX reset/step has returned."""

    def __init__(self, env, raw_env, proprio_keys=None):
        self.raw_env = raw_env
        super().__init__(env, proprio_keys=proprio_keys)

    def _inner_without_camera_capture(self, operation, *args, **kwargs):
        capture_images = bool(self.raw_env.image_obs)
        self.raw_env.image_obs = False
        try:
            result = operation(*args, **kwargs)
        finally:
            self.raw_env.image_obs = capture_images
        return result, capture_images

    def _attach_camera_images(self, observation, capture_images):
        if not capture_images:
            return observation
        ego, wrist_left, wrist_right = self.raw_env.render()
        value = dict(observation)
        value["images"] = {
            "random_camera" if self.raw_env.randomize else "ego": ego,
            "wrist_left": wrist_left,
            "wrist_right": wrist_right,
        }
        return value

    def reset(self, **kwargs):
        result, capture_images = self._inner_without_camera_capture(
            self.env.reset, **kwargs
        )
        observation, info = result
        observation = self._attach_camera_images(observation, capture_images)
        return self.observation(observation), info

    def step(self, action):
        result, capture_images = self._inner_without_camera_capture(
            self.env.step, action
        )
        observation, reward, terminated, truncated, info = result
        observation = self._attach_camera_images(observation, capture_images)
        return (
            self.observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )


class TaskConfig(TaskConfigBase):
    task_id = "22"
    task_name = "t22_pass_screwdriver"
    task_prompt = (
        "Touch the screwdriver with the right hand first, then touch it with the left "
        "hand while it is airborne, and let it land in the horizontal green target zone."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "screwdriver_pose",
        "tool_zone_pose",
        "screwdriver_velocity",
        "handover_metrics",
        "handover_stage",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        filter_alpha=0.55,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.05,
        filter_max_rot_step_deg=8.0,
        # The AS019 hand is in contact with a thin cylindrical handle during
        # the handover.  The previous 0.08 rad/frame cap still allowed a
        # malformed MANUS packet to move every finger by 2.4 rad/s; in the
        # failed take-over this changed the left-hand gesture while the
        # handle was loaded and threw the screwdriver.  Keep the live stream
        # continuous at 0.03 rad/frame (0.9 rad/s at 30 Hz) while retaining
        # direct, non-frozen left-hand targets and full operator control.
        hand_filter_max_step_rad=0.03,
        hand_takeover_max_step_rad=0.03,
        hand_takeover_max_control_steps=12,
        # Keep tracker-jump rebases fail-closed so the ordered contact events
        # still consume a coherent arm/hand sample.
        fail_closed_rebase_on_tracker_jump=True,
    )

    @staticmethod
    def _automatic_policy_enabled() -> bool:
        return os.environ.get(
            "HANDSON_AS019_AUTOMATIC_POLICY", ""
        ).strip().lower() in {"1", "true", "yes", "on"}

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
        ).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                "B320-AS019 is an Isaac Sim 5.1 PhysX-only collection task; "
                f"got sim_backend={sim_backend!r}."
            )
        if os.environ.get("HANDSON_ENABLE_ISAACSIM_NEWTON", "0").strip().lower() not in {
            "", "0", "false", "no", "off"
        }:
            raise ValueError("B320-AS019 forbids NVIDIA Newton.")

        from ...sim.isaaclab.as019_screwdriver_handover_env import (
            AS019AutomaticPolicyWrapper,
            AS019HandsOnlyGuidanceEnv,
            AS019ScrewdriverHandoverEnv,
        )

        kwargs.setdefault("hz", 30)
        kwargs.setdefault("physics_dt", 1.0 / 120.0)
        kwargs.setdefault("time_limit", 120.0)
        kwargs.setdefault("physics_backend", "physx")
        environment_type = (
            AS019HandsOnlyGuidanceEnv
            if os.environ.get(
                "HANDSON_TASK_GUIDANCE_RECORDING", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            else AS019ScrewdriverHandoverEnv
        )
        raw_env = environment_type(
            render_mode=render_mode,
            randomize=randomize,
            **kwargs,
        )
        self.attach_prompt_overlay(raw_env)
        automatic = self._automatic_policy_enabled()
        if policy_mode or automatic:
            env = DualArmPolicyWrapper(raw_env)
            if automatic:
                env = AS019AutomaticPolicyWrapper(env, raw_env)
        else:
            env = DualArmViveHandTeleopWrapper(raw_env, self.teleop)
        return AS019DeferredCameraObsAdapter(
            env, raw_env, proprio_keys=type(self).proprio_keys
        )

    def process_demos(self, demo):
        return demo
