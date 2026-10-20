import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import (
    BimanualTeleopConfig,
    DualArmViveHandTeleopWrapper,
)


class TaskConfig(TaskConfigBase):
    task_id = "27"
    task_name = "t27_place_largest_fitting_ball"
    task_prompt = (
        "Find the target C, G, or J YCB cup in the shuffled cup row. Choose "
        "the largest of the four balls in the shuffled ball row that can fit "
        "in that cup, place it into the cup using either hand as needed, "
        "release both hands, and leave the ball and cup stable."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "active_cup_pose",
        "active_cup_id",
        "active_cup_velocity",
        "candidate_ball_poses",
        "candidate_ball_ids",
        "candidate_ball_velocities",
        "candidate_valid",
        "task_stage",
        "selected_ball_cup_local",
        "ycb02_event_flags",
        "table_delta_height",
    ]
    # YCB02 deliberately exposes both ROKAE/Wuji arms.  The active cup and
    # candidate balls may be sampled anywhere across the shared tabletop, so
    # a single-arm wrapper would silently discard the left-hand command and
    # make the far-side reset layouts unreachable during collection.
    teleop = BimanualTeleopConfig(
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
        sim_backend: Literal["isaacsim"] | None = None,
        **kwargs,
    ):
        backend = (
            sim_backend
            or os.environ.get("HANDSON_SIM_BACKEND")
            or "isaacsim"
        ).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                "t27_place_largest_fitting_ball supports only Isaac Sim 5.1; "
                f"received sim_backend={sim_backend!r}."
            )

        from ...sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

        kwargs.setdefault("time_limit", 60.0)
        env = HandsOnABCIsaacLabEnv(
            self.task_name,
            render_mode=render_mode,
            randomize=randomize,
            hz=30,
            **kwargs,
        )
        self.attach_prompt_overlay(env)
        if policy_mode:
            env = DualArmPolicyWrapper(env)
        else:
            env = DualArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
