import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "39"
    task_name = "t39_set_timer"
    task_prompt = (
        "Stabilize the YCB timer with the left hand, program the prompted "
        "minutes and seconds with a right fingertip (MIN/SEC in either order), "
        "press START/STOP last, withdraw both hands, and wait for one countdown "
        "tick while the timer stays flat and stable. A bad press must be cleared "
        "with the physical CLEAR button before continuing."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "timer_pose",
        "timer_target",
        "timer_display",
        "timer_button_positions",
        "timer_logic",
        "timer_button_latches",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        # Keep normal reaches responsive even at the live 15-18 Hz rate.
        # These bounds allow at most 9 mm / 1.8 deg per action; discontinuous
        # tracker frames are rejected independently by the guard below.
        filter_alpha=0.60,
        filter_deadband_m=0.0008,
        filter_rot_deadband_deg=0.4,
        filter_max_step_m=0.015,
        filter_max_rot_step_deg=3.0,
        velocity_assist=False,
        # Cancel stale Cartesian goals at the measured palms, then wait for
        # the original tracker frame.  Never adopt a relocated frame merely
        # because it is stable or a recovery timeout has elapsed.
        tracker_packet_max_age_s=0.25,
        tracker_jump_translation_m=0.060,
        tracker_jump_rotation_deg=20.0,
        fail_closed_rebase_on_tracker_jump=True,
        preserve_tracker_baseline_on_recovery=True,
        require_stable_pair_on_enable=True,
        refresh_tracker_baseline_on_enable=True,
    )

    def get_native_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=True,
        sim_backend: Literal["mujoco", "isaacsim"] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("time_limit", 90.0)
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("control_dt", 1.0 / 30.0)
        kwargs.setdefault("physics_dt", 1.0 / 120.0)
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                f"{self.task_id} uses the Isaac Sim 5.1 collection path and requires "
                f"sim_backend='isaacsim', got {sim_backend!r}."
            )

        from ...sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

        env = HandsOnABCIsaacLabEnv(
            self.task_name,
            render_mode=render_mode,
            randomize=randomize,
            **kwargs,
        )
        env._target_prompt = lambda: (
            "YCB07 | Target "
            f"{env._ycb07_state.target_minutes:02d}:"
            f"{env._ycb07_state.target_seconds:02d} | "
            "Left hand braces; right fingertip presses MIN/SEC then START/STOP; "
            "withdraw both hands for 0.5 s after one countdown tick."
        )
        self.attach_prompt_overlay(env)
        if policy_mode:
            env = DualArmPolicyWrapper(env)
        else:
            env = DualArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
