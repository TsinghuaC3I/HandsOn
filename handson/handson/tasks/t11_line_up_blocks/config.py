import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import SingleArmPolicyWrapper
from ..sim_teleop import SingleArmTeleopConfig, SingleArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "11"
    task_name = 't11_line_up_blocks'
    task_prompt = 'Use only the set square tool to push three scattered wood blocks onto the baseline and align them side by side.'
    proprio_keys = [
        'tcp_pose',
        'gripper_pose',
        'tool_pose',
        'block_poses',
        'baseline_pose',
        'alignment_error',
        'table_delta_height',
    ]
    teleop = SingleArmTeleopConfig(
        filter_alpha=0.55,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.055,
        filter_max_rot_step_deg=9.0,
    )

    def get_native_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=True,
        sim_backend: Literal["mujoco", "isaacsim"] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("time_limit", 100.0)
        kwargs.setdefault("hz", 30)
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                f"{self.task_id} uses the native IsaacSim collection path and requires "
                f"sim_backend='isaacsim', got {sim_backend!r}."
            )

        from ...sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

        env = HandsOnABCIsaacLabEnv(
            self.task_name,
            render_mode=render_mode,
            randomize=randomize,
            **kwargs,
        )
        self.attach_prompt_overlay(env)
        if policy_mode:
            env = SingleArmPolicyWrapper(env)
        else:
            env = SingleArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
