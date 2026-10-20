import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "43"
    task_name = "t43_squeeze_scissors"
    task_prompt = (
        "Lift the scissors clear of the table and make three complete cuts in "
        "the air. Try one hand first; the other hand may reposition or assist."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "scissors_pose",
        "scissors_hinge",
        "cut_progress",
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
        sim_backend: Literal["isaacsim"] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("time_limit", 180.0)
        kwargs.setdefault("hz", 30)
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend == "isaacsim":
            from ...sim.isaaclab.d_collection_env import HandsOnDIsaacLabEnv

            env = HandsOnDIsaacLabEnv(
                self.task_name, render_mode=render_mode, randomize=randomize, **kwargs
            )
        else:
            raise ValueError(f"Unsupported D05 sim_backend={sim_backend!r}.")
        self.attach_prompt_overlay(env)
        if policy_mode:
            env = DualArmPolicyWrapper(env)
        else:
            env = DualArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
