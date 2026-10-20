from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import DualArmViveHandTeleopWrapper
from ..t21_pass_block.config import TaskConfig as D03TaskConfig
from .contract import TASK_ID, TASK_NAME, PROMPT


class TaskConfig(TaskConfigBase):
    task_id = TASK_ID
    task_name = TASK_NAME
    task_prompt = PROMPT
    proprio_keys = [*D03TaskConfig.proprio_keys, "lid_pose"]
    teleop = D03TaskConfig.teleop

    def get_native_environment(self, policy_mode=False, render_mode: Literal["rgb_array", "human"] = "human",
                        randomize=True, sim_backend=None, **kwargs):
        if sim_backend not in (None, "isaac", "isaacsim", "isaacsim_newton"):
            raise ValueError(f"XY1 supports Isaac Sim 5.1 only: {sim_backend!r}")
        if sim_backend == "isaacsim_newton":
            kwargs["physics_backend"] = "newton"
        from ...sim.isaaclab.d_collection_env import HandsOnDIsaacLabEnv
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("time_limit", 300.0)
        env = HandsOnDIsaacLabEnv(self.task_name, render_mode=render_mode,
                                randomize=randomize, **kwargs)
        missing = set(self.proprio_keys) - set(env.observation_space["state"].spaces)
        if missing:
            raise RuntimeError(f"XY1 missing declared state fields: {sorted(missing)}")
        self.attach_prompt_overlay(env)
        env = DualArmPolicyWrapper(env) if policy_mode else DualArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
