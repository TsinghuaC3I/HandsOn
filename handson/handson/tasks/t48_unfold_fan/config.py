import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper
from ...sim.isaaclab.p001_contract import (
    P001ScriptedPolicy,
    RENDER_RESOLUTION_HW,
    TASK_PROMPT,
    TIME_LIMIT_SECONDS,
    sample_reset_state,
)


class TaskConfig(TaskConfigBase):
    task_id = "48"
    task_name = "t48_unfold_fan"
    task_prompt = TASK_PROMPT
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "fan_center_pose",
        "fan_left_end_pose",
        "fan_right_end_pose",
        "fan_joint_positions",
        "fan_joint_velocities",
        "fan_target_angle",
        "fan_task_state",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        filter_alpha=0.50,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.035,
        filter_max_rot_step_deg=6.0,
    )

    def get_native_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=True,
        sim_backend: Literal["isaacsim"] | None = None,
        **kwargs,
    ):
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                f"{self.task_id} requires Isaac Sim 5.1 PhysX; "
                f"sim_backend={sim_backend!r} is unsupported."
            )
        requested_physics = str(kwargs.pop("physics_backend", "physx")).lower()
        if requested_physics != "physx":
            raise ValueError(
                f"{self.task_id} forbids Newton/Sim6 and requires physics_backend='physx'."
            )
        kwargs.setdefault("time_limit", TIME_LIMIT_SECONDS)
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("physics_dt", 1.0 / 120.0)
        kwargs.setdefault("render_resolution", RENDER_RESOLUTION_HW)

        from ...sim.isaaclab.p001_collection_env import HandsOnP001IsaacLabEnv

        env = HandsOnP001IsaacLabEnv(
            self.task_name,
            render_mode=render_mode,
            randomize=randomize,
            physics_backend="physx",
            **kwargs,
        )
        self.attach_prompt_overlay(env)
        if policy_mode:
            env = DualArmPolicyWrapper(env)
        else:
            env = DualArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    @staticmethod
    def scripted_policy(seed: int, target_open_deg: float | None = None) -> P001ScriptedPolicy:
        return P001ScriptedPolicy(
            sample_reset_state(seed, target_open_deg=target_open_deg)
        )

    def process_demos(self, demo):
        return demo
