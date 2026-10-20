import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import SingleArmPolicyWrapper
from ..sim_teleop import SingleArmTeleopConfig, SingleArmViveHandTeleopWrapper
from ...sim.isaaclab.eval_gen06_spray_shoe_scene import PROMPT


class TaskConfig(TaskConfigBase):
    task_id = "EVAL-Gen06"
    task_name = "eval_gen06_spray_shoe"
    task_prompt = PROMPT
    # Preserve the trained C04 policy's field order and dimensions. The
    # plant_ori_pose compatibility alias explicitly denotes the shoe target.
    proprio_keys = ["tcp_pose", "gripper_pose", "spray_ori_pose", "plant_ori_pose", "table_delta_height"]
    teleop = SingleArmTeleopConfig(
        filter_alpha=0.55, filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5, filter_max_step_m=0.06,
        filter_max_rot_step_deg=10.0,
    )

    def get_native_environment(self, policy_mode=False, render_mode: Literal["rgb_array", "human"]="human",
                        randomize=True, sim_backend=None, **kwargs):
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend in {"newton", "nvidia_newton", "isaac_newton"}:
            backend = "isaacsim_newton"
        if backend not in {"isaac", "isaacsim", "isaacsim_newton"}:
            raise ValueError(f"GEN7 supports Isaac Sim 5.1 only; received {backend!r}.")
        from ...sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
        kwargs.setdefault("control_dt", 1.0 / 30.0)
        kwargs.setdefault("physics_dt", 1.0 / 120.0)
        kwargs.setdefault(
            "physics_backend",
            "newton" if backend == "isaacsim_newton" else "physx",
        )
        env = HandsOnABCIsaacLabEnv(self.task_name, render_mode=render_mode, randomize=randomize, hz=30, **kwargs)
        self.attach_prompt_overlay(env)
        env = SingleArmPolicyWrapper(env) if policy_mode else SingleArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
