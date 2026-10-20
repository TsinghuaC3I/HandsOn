"""The ordinary collection/policy entry point for XY4."""

import os

from ..b320_common import B320TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import DualArmViveHandTeleopWrapper
from ...sim.isaaclab.xy4_contract import TASK_NAME, TASK_PROMPT, TIME_LIMIT_SECONDS


class TaskConfig(B320TaskConfigBase):
    task_id = "EVAL-Comb05"
    benchmark_task_id = "EVAL-Comb05"
    task_name = TASK_NAME
    task_prompt = TASK_PROMPT
    time_limit_seconds = TIME_LIMIT_SECONDS
    # Task event history belongs in recording telemetry, outside policy state.
    proprio_keys = ["tcp_pose", "gripper_pose", "table_delta_height"]

    def get_native_environment(
        self, policy_mode=False, render_mode="human", randomize=True,
        sim_backend=None, **kwargs,
    ):
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend not in {"isaac", "isaacsim", "isaacsim_newton"}:
            raise ValueError("XY4 requires Isaac Sim 5.1 PhysX")
        expected_physics = "newton" if backend == "isaacsim_newton" else "physx"
        physics_backend = kwargs.pop("physics_backend", expected_physics)
        if physics_backend != expected_physics:
            raise ValueError(f"XY4 requires the {expected_physics} backend")
        from ...sim.isaaclab.xy4_env import HandsOnXY4IsaacLabEnv

        env = HandsOnXY4IsaacLabEnv(
            self.task_name, render_mode=render_mode, randomize=randomize,
            physics_backend=physics_backend, **kwargs,
        )
        self.attach_prompt_overlay(env)
        env = (DualArmPolicyWrapper(env) if policy_mode else
               DualArmViveHandTeleopWrapper(env, self.teleop))
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)
