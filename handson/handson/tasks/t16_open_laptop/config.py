import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper
from ...sim.isaaclab.fu015_contract import (
    CONTROL_HZ,
    FU015ScriptedPolicy,
    PHYSICS_HZ,
    RENDER_RESOLUTION_HW,
    TASK_ID,
    TASK_NAME,
    TASK_PROMPT,
    sample_reset_state,
)


class TaskConfig(TaskConfigBase):
    # Keep these literals visible to the repository's static registration
    # audit; the imported constants are asserted equal in CPU tests.
    task_id = "16"
    task_name = "t16_open_laptop"
    task_prompt = TASK_PROMPT
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "laptop_base_pose",
        "laptop_display_pose",
        "hinge_angle",
        "base_displacement",
        "bimanual_role_contacts",
        "fu015_task_state",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        filter_alpha=0.55,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.050,
        filter_max_rot_step_deg=8.0,
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
        kwargs.setdefault("time_limit", 22.0)
        kwargs.setdefault("hz", CONTROL_HZ)
        kwargs.setdefault("control_dt", 1.0 / CONTROL_HZ)
        kwargs.setdefault("physics_dt", 1.0 / PHYSICS_HZ)
        kwargs.setdefault("render_resolution", RENDER_RESOLUTION_HW)

        from ...sim.isaaclab.fu015_collection_env import HandsOnFU015IsaacLabEnv

        env = HandsOnFU015IsaacLabEnv(
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
    def scripted_policy(seed: int) -> FU015ScriptedPolicy:
        return FU015ScriptedPolicy(sample_reset_state(seed))

    def process_demos(self, demo):
        return demo
