import os
from dataclasses import replace
from typing import Literal

from ...sim.isaaclab.ycb11_semantics import YCB11_CONTROL_HZ, YCB11_PHYSICS_HZ
from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "EVAL-Flex01"
    task_name = "eval_flex01_compress_sponge_into_mug"
    task_prompt = (
        "Use the left hand to stabilize the YCB mug. With the right hand, "
        "physically grasp and compress the Sponge 3, insert it through the "
        "mug rim until it is mostly inside, then release both objects and "
        "leave the mug upright and stable."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "mug_pose",
        "mug_velocity",
        "sponge_pose",
        "sponge_keypoints",
        "sponge_aabb",
        "sponge_nodal_velocity_summary",
        "compression_metrics",
        "containment_metrics",
        "contact_metrics",
        "release_metrics",
        "task_phase",
        "task_events",
        "task_initial_state",
        "robot_limit_state",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        filter_alpha=1.0,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        # Match the normal 30 Hz bimanual teleoperation profile.  YCB11 no
        # longer applies a contact-distance speed reducer in the environment.
        filter_max_step_m=0.060,
        filter_max_rot_step_deg=10.0,
        velocity_assist=False,
        hand_filter_max_step_rad=0.08,
        # Refresh the episode origin after initialization. Runtime faults
        # hold the palms until original tracking returns, as in all tasks.
        fail_closed_rebase_on_tracker_jump=True,
        require_stable_pair_on_enable=True,
        refresh_tracker_baseline_on_enable=True,
    )

    def _teleop_config(self) -> BimanualTeleopConfig:
        """Allow isolated validation without changing collection defaults."""

        return replace(
            self.teleop,
            vive_udp_port=int(
                os.environ.get(
                    "HANDSON_YCB11_VIVE_UDP_PORT",
                    self.teleop.vive_udp_port,
                )
            ),
            right_hand_udp_port=int(
                os.environ.get(
                    "HANDSON_YCB11_RIGHT_HAND_UDP_PORT",
                    self.teleop.right_hand_udp_port,
                )
            ),
            left_hand_udp_port=int(
                os.environ.get(
                    "HANDSON_YCB11_LEFT_HAND_UDP_PORT",
                    self.teleop.left_hand_udp_port,
                )
            ),
        )

    def get_native_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=True,
        sim_backend: Literal["mujoco", "isaacsim"] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("time_limit", 240.0)
        kwargs.setdefault("hz", YCB11_CONTROL_HZ)
        kwargs.setdefault("control_dt", 1.0 / YCB11_CONTROL_HZ)
        kwargs.setdefault("physics_dt", 1.0 / YCB11_PHYSICS_HZ)
        backend = (
            sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")
        ).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                "YCB11 uses a native PhysX volume-deformable Sponge 3 and "
                f"requires sim_backend='isaacsim', got {sim_backend!r}."
            )

        from ...sim.isaaclab.ycb11_compress_sponge_env import (
            HandsOnYCB11CompressSpongeIsaacLabEnv,
        )

        env = HandsOnYCB11CompressSpongeIsaacLabEnv(
            self.task_name,
            render_mode=render_mode,
            randomize=randomize,
            **kwargs,
        )
        self.attach_prompt_overlay(env)
        if policy_mode:
            env = DualArmPolicyWrapper(env)
        else:
            env = DualArmViveHandTeleopWrapper(env, self._teleop_config())
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
