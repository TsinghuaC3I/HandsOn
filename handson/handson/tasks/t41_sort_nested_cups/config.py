import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "41"
    task_name = "t41_sort_nested_cups"
    task_prompt = (
        "Separate the three nested YCB cups, then place them upright in a straight "
        "row from left to right in the main camera: large, medium, small. Release all cups."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "cup_poses",
        "cup_velocities",
        "ycb04_event_state",
        "ycb04_layout_metrics",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        # Fine cup manipulation needs the latest position when the wrist
        # stops. Deadband and jump limits still reject noise/discontinuities.
        filter_alpha=1.0,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.055,
        filter_max_rot_step_deg=9.0,
        max_tracker_delta_m=0.60,
        velocity_assist=False,
    )

    def get_native_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=True,
        sim_backend: Literal["mujoco", "isaacsim"] | None = None,
        **kwargs,
    ):
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                f"{self.task_id} requires the Isaac Sim 5.1 collection runtime; "
                f"got sim_backend={sim_backend!r}."
            )
        from ...sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

        kwargs.setdefault("control_dt", 1.0 / 30.0)
        kwargs.setdefault("physics_dt", 1.0 / 120.0)
        kwargs.setdefault("time_limit", 120.0)
        kwargs.setdefault("hz", 30)
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
