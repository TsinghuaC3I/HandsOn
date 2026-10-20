"""Task configuration for bimanual Pillow024-v2 placement in a carton."""

import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "EVAL-Flex03"
    task_name = "eval_flex03_pack_pillow"
    task_prompt = 'Use both hands to squeeze and lift the foam pillow, lower it into the open carton so it rests inside with most of the pillow below the rim, then release both hands and keep it stable for one second. Tilted placement and raised corners are allowed.'
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "pillow_centroid",
        "pillow_aabb",
        "pillow_valid",
        "carton_pose",
        "carton_inner_bounds",
        "task_metrics",
        "stage_state",
        "reset_layout",
        "table_delta_height",
        "foam_grip_state",
    ]
    teleop = BimanualTeleopConfig(
        # The Isaac DLS target limiter already applies a 0.25 blend and a
        # 35 mm site-step cap.  A second slow 0.55/15 mm filter made the
        # live command advance only a couple of millimetres per control frame
        # (the PhysX collector currently runs at about 10--12 Hz), which is
        # perceived as stop-and-go motion during the lift.  Keep a small
        # tracker deadband, but let the task-local contact limiter provide the
        # final anti-poke bound.
        filter_alpha=0.85,
        filter_deadband_m=0.002,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.025,
        filter_max_rot_step_deg=6.0,
        # A glove closure is a live joint target, not a pose sample.  Bound
        # its per-control-frame change so first contact cannot inject an
        # impulsive finger command into the deformable pillow.
        hand_filter_max_step_rad=0.08,
        # A G01 reset can rebuild Isaac while the bridge is still publishing
        # the previous world frame.  Use the bridge's stable two-wrist
        # recovery path so a displaced packet is rebased once instead of
        # being rejected indefinitely (which leaves the pickup looking frozen).
        fail_closed_rebase_on_tracker_jump=True,
        require_stable_pair_on_enable=True,
        refresh_tracker_baseline_on_enable=True,
    )

    def get_native_environment(
        self,
        policy_mode: bool = False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize: bool = True,
        sim_backend: Literal["mujoco", "isaacsim"] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("time_limit", 180.0)
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("control_dt", 1.0 / 30.0)
        backend = (
            sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")
        ).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                "G01-3 uses the PhysX Pillow024-v2 volume deformable and "
                f"requires sim_backend='isaacsim', got {sim_backend!r}."
            )

        from ...sim.isaaclab.g01_deformable_asset_env import (
            HandsOnG01DeformableAssetIsaacLabEnv,
        )

        env = HandsOnG01DeformableAssetIsaacLabEnv(
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
