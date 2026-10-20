import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "34"
    task_name = "t34_roll_ball_through_gate"
    task_prompt = (
        "The tennis ball may start on either side of the centred cracker and "
        "sugar boxes. Use either hand to roll or flick it past the far side "
        "of the boxes: left hand for left-to-right or right hand for "
        "right-to-left. Success is based only on the ball centre passing the "
        "far Y face in the recorded travel direction. "
        "Keep the ball on the table, do not pick up or throw it, and do not "
        "touch either box with the hand or ball; any box contact resets the "
        "attempt. The ball does not need to stop after passing."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "ball_pose",
        "ball_linear_velocity",
        "ball_angular_velocity",
        "left_gate_pose",
        "right_gate_pose",
        "gate_frame_pose",
        "ball_gate_local_position",
        "gate_width",
        "ycb01_start_direction",
        "ycb01_event_flags",
        "ycb01_hidden_state",
        "table_delta_height",
    ]
    # Keep the task direct (there is no grasp or motion assist), while exposing
    # both hands.  The ball starts can now be far from the gate, so the bounded
    # pose filter avoids an unnecessarily abrupt first corrective motion.
    teleop = BimanualTeleopConfig(
        filter_alpha=0.55,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.060,
        filter_max_rot_step_deg=10.0,
    )

    def get_native_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=True,
        sim_backend: Literal["mujoco", "isaacsim"] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("time_limit", 45.0)
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("control_dt", 1.0 / 30.0)
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                "YCB01 uses the qualified Isaac Sim 5.1 collection runtime and "
                f"requires sim_backend='isaacsim', got {sim_backend!r}."
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
            env = DualArmPolicyWrapper(env)
        else:
            env = DualArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
