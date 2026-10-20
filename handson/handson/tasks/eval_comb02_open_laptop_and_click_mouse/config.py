import os

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper
from ...sim.isaaclab.xy2_contract import TASK_PROMPT, TIME_LIMIT_SECONDS


class TaskConfig(TaskConfigBase):
    task_id = "EVAL-Comb02"
    task_name = "eval_comb02_open_laptop_and_click_mouse"
    task_prompt = TASK_PROMPT
    proprio_keys = [
        "tcp_pose", "gripper_pose", "laptop_base_pose", "laptop_display_pose",
        "hinge_angle", "mouse_ori_pose", "mousepad_ori_pose", "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        filter_alpha=0.55, filter_deadband_m=0.001, filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.050, filter_max_rot_step_deg=8.0,
    )

    def get_native_environment(self, policy_mode=False, render_mode="human", randomize=True,
                        sim_backend=None, **kwargs):
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend not in {"isaac", "isaacsim", "isaacsim_newton"}:
            raise ValueError("XY2 requires Isaac Sim 5.1 / IsaacLab 2.3.2 PhysX")
        physics = "newton" if backend == "isaacsim_newton" else "physx"
        if str(kwargs.pop("physics_backend", physics)).lower() != physics:
            raise ValueError(f"{backend} requires {physics}")
        kwargs.setdefault("time_limit", TIME_LIMIT_SECONDS)
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("control_dt", 1.0 / 30.0)
        kwargs.setdefault("physics_dt", 1.0 / 120.0)
        kwargs.setdefault("render_resolution", (480, 640))
        from ...sim.isaaclab.xy2_collection_env import HandsOnXY2IsaacLabEnv

        env = HandsOnXY2IsaacLabEnv(
            self.task_name, render_mode=render_mode, randomize=randomize,
            physics_backend=physics, **kwargs,
        )
        self.attach_prompt_overlay(env)
        env = DualArmPolicyWrapper(env) if policy_mode else DualArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
