import os

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper
from .semantics import PROMPT, TASK_ID, TASK_NAME


class TaskConfig(TaskConfigBase):
    task_id = TASK_ID
    task_name = TASK_NAME
    task_prompt = PROMPT
    proprio_keys = [
        "tcp_pose", "gripper_pose", "target_object_pose", "receptacle_pose",
        "object_poses", "target_selector", "camera_ori_pose", "logo_ori_pose",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        filter_alpha=0.55, filter_deadband_m=0.001, filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.06, filter_max_rot_step_deg=10.0,
    )

    def get_native_environment(self, policy_mode=False, render_mode="human",
                        randomize=True, sim_backend=None, **kwargs):
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend in {"newton", "nvidia_newton", "isaac_newton"}:
            backend = "isaacsim_newton"
        if backend not in {"isaac", "isaacsim", "isaacsim_newton"}:
            raise ValueError(
                f"XY3 uses Isaac Sim collection runtime, got {backend!r}"
            )
        from ...sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

        kwargs.setdefault("time_limit", 240.0)
        kwargs.setdefault(
            "physics_backend",
            "newton" if backend == "isaacsim_newton" else "physx",
        )
        env = HandsOnABCIsaacLabEnv(
            self.task_name, render_mode=render_mode, randomize=randomize,
            hz=30, **kwargs,
        )
        self.attach_prompt_overlay(env)
        raw_env = env
        env.task_overlay_prompt_getter = lambda: (
            f"Pick the {'banana' if raw_env._target_object_name == '011_banana' else raw_env._target_object_name}, "
            "place it in the bucket and release it. Then pick up the camera, "
            "aim at the logo on the bucket and press the shutter."
        )
        env = DualArmPolicyWrapper(env) if policy_mode else DualArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
