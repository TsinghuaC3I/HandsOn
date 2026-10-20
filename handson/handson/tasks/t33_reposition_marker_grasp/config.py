import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import SingleArmPolicyWrapper
from ..sim_teleop import SingleArmTeleopConfig, SingleArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "33"
    task_name = "t33_reposition_marker_grasp"
    task_prompt = (
        'Pick up the capped YCB marker with the right hand. From the grasp '
        'point recorded on each pickup, move both the thumb and index contact '
        'points at least 2.5 cm along the marker toward its writing tip under '
        'the cap. Success occurs while both fingers hold the lifted marker '
        'there. Dropping it onto the table does not reset the episode; the next '
        'pickup records a new starting grasp point.'
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "marker_pose",
        "palm_pose",
        "marker_velocity",
        "finger_contact_forces",
        "ycb10_contact_validity",
        "ycb10_event_flags",
        "ycb10_progress",
        "ycb10_hidden_state",
        "table_delta_height",
    ]
    teleop = SingleArmTeleopConfig()

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
        backend = (
            sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")
        ).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                "YCB10 is a native Isaac Sim collection task and requires "
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
