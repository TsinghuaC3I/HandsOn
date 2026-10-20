import os
from typing import ClassVar, Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "40"
    task_name = "t40_slide_knife_blade"
    task_prompt = (
        "Pick up the utility knife with the right hand. After pickup, touch the "
        "slider with the left fingertips and move it to the episode target: "
        "bottom to retract the blade or top to extend it."
    )
    proprio_keys: ClassVar[list[str]] = [
        "tcp_pose",
        "gripper_pose",
        "knife_body_pose",
        "blade_pose",
        "blade_joint_state",
        "knife_interaction_state",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        filter_alpha=0.55,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.035,
        filter_max_rot_step_deg=6.0,
    )

    def get_native_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=True,
        sim_backend: Literal["isaacsim"] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("time_limit", 90.0)
        kwargs.setdefault("hz", 30)
        render_resolution = tuple(
            kwargs.pop("render_resolution", (480, 640))
        )
        if render_resolution != (480, 640):
            raise ValueError(
                f"{self.task_id} requires native 640x480 ego frames; "
                f"got render_resolution={render_resolution!r}."
            )
        kwargs["render_resolution"] = render_resolution
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                f"{self.task_id} requires Isaac Sim 5.1 PhysX with "
                f"sim_backend='isaacsim'; got {sim_backend!r}."
            )
        physics_backend = str(kwargs.get("physics_backend", "physx")).strip().lower()
        if physics_backend != "physx":
            raise ValueError(
                f"{self.task_id} supports PhysX only; Newton is forbidden for this task."
            )

        from ...sim.isaaclab.as013_collection_env import HandsOnAS013IsaacLabEnv

        env = HandsOnAS013IsaacLabEnv(
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
