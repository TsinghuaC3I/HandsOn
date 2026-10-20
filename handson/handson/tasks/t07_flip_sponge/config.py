import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "07"
    task_name = "t07_flip_sponge"
    task_prompt = (
        "Lift and level the YCB skillet with the left hand; a brief steady hold "
        "locks the left arm automatically. A tilted or lost hold unlocks for "
        "regripping and leveling. With the right hand, grasp the YCB "
        "spatula, slide its blade under the YCB sponge, lift and flip the sponge "
        "once, then leave the opposite face stable inside the skillet. Do not "
        "touch the sponge directly with either hand."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "skillet_pose",
        "skillet_linear_velocity",
        "skillet_angular_velocity",
        "spatula_pose",
        "spatula_linear_velocity",
        "spatula_angular_velocity",
        "sponge_pose",
        "sponge_linear_velocity",
        "sponge_angular_velocity",
        "sponge_in_pan_position",
        "ycb08_geometry_metrics",
        "ycb08_contact_state",
        "ycb08_left_pan_lock_state",
        "ycb08_event_flags",
        "ycb08_hidden_state",
        "ycb08_sampled_reset_poses",
        "table_delta_height",
    ]
    # Direct mapping until a physical left-hand pan hold is lifted and level;
    # the environment then latches that target and records zero left pose input.
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
        sim_backend: Literal["mujoco", "isaacsim"] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("time_limit", 300.0)
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("control_dt", 1.0 / 30.0)
        kwargs.setdefault("physics_dt", 1.0 / 120.0)
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                "YCB08 uses the qualified Isaac Sim 5.1 collection runtime and "
                f"requires sim_backend='isaacsim', got {sim_backend!r}."
            )

        from ...sim.isaaclab.t07_flip_sponge_env import (
            YCB08SpatulaFlipSpongeIsaacLabEnv,
        )

        env = YCB08SpatulaFlipSpongeIsaacLabEnv(
            self.task_name,
            render_mode=render_mode,
            randomize=randomize,
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
