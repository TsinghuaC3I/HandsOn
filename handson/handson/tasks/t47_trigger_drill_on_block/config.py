import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import SingleArmPolicyWrapper
from ..sim_teleop import SingleArmTeleopConfig, SingleArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "47"
    task_name = "t47_trigger_drill_on_block"
    task_prompt = (
        "Bring the unpowered YCB drill into physical contact with the YCB wood "
        "block, then press the physical trigger to at least 80 percent and "
        "hold for 0.5 seconds while maintaining drill-block contact. Any grasp "
        "and any finger may be used. If already pressed, release the trigger "
        "while in contact before pressing again. Do not drill the block."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "drill_pose",
        "drill_linear_velocity",
        "drill_angular_velocity",
        "wood_block_pose",
        "wood_block_linear_velocity",
        "wood_block_angular_velocity",
        "drill_tip_position",
        "drill_axis_direction",
        "wood_target_position",
        "aim_metrics",
        "trigger_state",
        "contact_state",
        "contact_validity",
        "ycb05_contact_grace",
        "ycb05_invalid_contact_channels",
        "ycb05_stage",
        "ycb05_event_flags",
        "ycb05_semantics_revision",
        "ycb05_timers",
        "ycb05_terminal_reason",
        "ycb05_sampled_reset_poses",
        "ycb05_reference_poses",
        "table_delta_height",
    ]
    # Direct motion and hand commands.  No task-specific grasp or trigger aid.
    teleop = SingleArmTeleopConfig(
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
        sim_backend: Literal["mujoco", "isaacsim"] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("time_limit", 60.0)
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("control_dt", 1.0 / 30.0)
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                "YCB05 uses the qualified Isaac Sim 5.1 collection runtime and "
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
            env = SingleArmPolicyWrapper(env)
        else:
            env = SingleArmViveHandTeleopWrapper(env, self.teleop)
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
