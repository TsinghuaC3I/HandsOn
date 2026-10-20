import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


class TaskConfig(TaskConfigBase):
    task_id = "46"
    task_name = "t46_store_charger"
    task_prompt = (
        "Hold the drawer open with the left hand, place the charger fully "
        "inside with the right hand, release the charger, then release the "
        "left hand and keep the result stable for 0.5 seconds."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "charger_pose",
        "drawer_pose",
        "drawer_joint_state",
        "fu006_task_state",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(
        filter_alpha=0.55,
        filter_deadband_m=0.001,
        filter_rot_deadband_deg=0.5,
        filter_max_step_m=0.050,
        filter_max_rot_step_deg=8.0,
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
        backend = (
            sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")
        ).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError(
                f"{self.task_id} requires the Isaac Sim 5.1 PhysX collection "
                f"path, got sim_backend={sim_backend!r}."
            )
        if os.environ.get("HANDSON_ENABLE_ISAACSIM_NEWTON", "0").strip().lower() not in {
            "",
            "0",
            "false",
            "no",
            "off",
        }:
            raise ValueError(f"{self.task_id} forbids the Newton backend")

        from ...sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

        env = HandsOnABCIsaacLabEnv(
            self.task_name,
            render_mode=render_mode,
            randomize=randomize,
            physics_backend="physx",
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
