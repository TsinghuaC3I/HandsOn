"""B04 action/observation interface with held-out packaging assets."""
import os

from ..t44_stack_blocks.config import TaskConfig as B04TaskConfig
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import SingleArmPolicyWrapper
from ..sim_teleop import SingleArmViveHandTeleopWrapper
from ...sim.isaaclab.eval_gen07_stack_boxes import TASK_ID, TASK_NAME, PROMPT


class TaskConfig(B04TaskConfig):
    task_id = TASK_ID
    task_name = TASK_NAME
    task_prompt = PROMPT
    proprio_keys = list(B04TaskConfig.proprio_keys)

    def get_native_environment(
        self, policy_mode=False, render_mode="human", randomize=True,
        sim_backend=None, **kwargs,
    ):
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend in {"newton", "nvidia_newton", "isaac_newton"}:
            backend = "isaacsim_newton"
        if backend not in {"isaac", "isaacsim", "isaacsim_newton"}:
            raise ValueError(f"GEN5 requires the Isaac Sim 5.1 backend; received {backend!r}")
        from ...sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

        kwargs.setdefault(
            "physics_backend",
            "newton" if backend == "isaacsim_newton" else "physx",
        )
        env = HandsOnABCIsaacLabEnv(
            self.task_name, render_mode=render_mode, randomize=randomize, hz=30, **kwargs
        )
        self.attach_prompt_overlay(env)
        env = (
            SingleArmPolicyWrapper(env) if policy_mode
            else SingleArmViveHandTeleopWrapper(env, self.teleop)
        )
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)
