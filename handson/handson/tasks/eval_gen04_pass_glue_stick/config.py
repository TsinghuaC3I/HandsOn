import os
from typing import Literal

from ..config import TaskConfigBase
from .contract import TASK_ID, TASK_NAME, PROMPT
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "EVAL-Gen04"
    task_name = "eval_gen04_pass_glue_stick"
    task_prompt = PROMPT
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "target_pose",
        "box_pose",
        "target_to_box",
        "handoff_metrics",
        "distractor_poses",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        filter_alpha=0.55,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.06,
        filter_max_rot_step_deg=10.0,
    )

    def get_native_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=True,
        sim_backend: Literal["mujoco", "isaacsim", "isaacsim_newton"] | None = None,
        **kwargs,
    ):
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend == "isaac":
            backend = "isaacsim"
        kwargs["physics_backend"] = "newton" if backend == "isaacsim_newton" else "physx"
        if backend in {"isaacsim", "isaacsim_newton"}:
            from ...sim.isaaclab.gen4_env import HandsOnGEN4IsaacLabEnv

            kwargs.setdefault("hz", 30)
            kwargs.setdefault("time_limit", 180.0)
            env = HandsOnGEN4IsaacLabEnv(
                self.task_name, render_mode=render_mode, randomize=randomize, **kwargs
            )
        else:
            raise ValueError(f"GEN4 supports Isaac Sim 5.1 only; received {backend!r}.")
        active_proprio_keys = [
            key
            for key in type(self).proprio_keys
            if key in env.observation_space["state"].spaces
        ]
        self.proprio_keys = active_proprio_keys
        self.attach_prompt_overlay(env)
        if policy_mode:
            env = DualArmPolicyWrapper(env)
        else:
            env = DualArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=active_proprio_keys)

    def process_demos(self, demo):
        return demo
