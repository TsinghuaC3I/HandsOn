from __future__ import annotations

import os

from ..t04_cover_pitcher.config import TaskConfig as B01TaskConfig
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import SingleArmPolicyWrapper
from ..sim_teleop import SingleArmViveHandTeleopWrapper
from .layout import TASK_NAME


class TaskConfig(B01TaskConfig):
    task_id = "EVAL-Comb01"
    task_name = TASK_NAME
    task_prompt = (
        "First align and seat the pitcher lid, then release it and let it settle. "
        "Next align and seat the clear-box lid, release it and let it settle. "
        "Both lids must remain correctly seated on their own containers."
    )

    def get_native_environment(self, policy_mode=False, render_mode="human", randomize=True, sim_backend=None, **kwargs):
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend not in {"isaac", "isaacsim", "isaacsim_newton"}:
            raise ValueError(f"XY5 requires the Isaac Sim 5.1 collection runtime, got {backend!r}")
        if backend == "isaacsim_newton":
            kwargs["physics_backend"] = "newton"
        from ...sim.isaaclab.eval_comb01_cover_pitcher_and_box_env import XY5PitcherThenBoxLidEnv

        kwargs.setdefault("control_dt", 1.0 / 30.0)
        kwargs.setdefault("physics_dt", 1.0 / 120.0)
        kwargs.setdefault("time_limit", 240.0)
        kwargs.setdefault("hz", 30)
        env = XY5PitcherThenBoxLidEnv(render_mode=render_mode, randomize=randomize, **kwargs)
        self.attach_prompt_overlay(env)
        env = SingleArmPolicyWrapper(env) if policy_mode else SingleArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)
