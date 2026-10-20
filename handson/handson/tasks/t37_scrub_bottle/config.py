import os
from typing import Literal

from ..config import TaskConfigBase
from ..obs_adapters import HandsOnObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import BimanualTeleopConfig, DualArmViveHandTeleopWrapper


TASK_NAME = "t37_scrub_bottle"
TASK_PROMPT = (
    "Retrieve and stabilize the labeled water bottle with the left hand, "
    "retrieve the cleaning brush with the right hand, scrub the lower, "
    "middle, and upper bottle "
    "bands in order with physical bristle contact, then release both "
    "objects and leave the bottle upright and stable."
)


class TaskConfig(TaskConfigBase):
    # The repository's canonical config identity is the name prefix.  The
    # source benchmark identity remains explicit for UI/evidence manifests.
    task_id = "37"
    benchmark_task_id = "B320-HS005"
    task_name = "t37_scrub_bottle"
    task_prompt = (
        "Retrieve and stabilize the labeled water bottle with the left hand, "
        "retrieve the cleaning brush with the right hand, scrub the lower, "
        "middle, and upper bottle "
        "bands in order with physical bristle contact, then release both "
        "objects and leave the bottle upright and stable."
    )
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "hs005_bottle_pose",
        "hs005_brush_pose",
        "hs005_sink_pose",
        "hs005_body_velocities",
        "hs005_role_contacts",
        "hs005_brush_contact",
        "hs005_coverage_mask",
        "hs005_order_state",
        "hs005_path_length",
        "hs005_release_state",
        "hs005_stability_state",
        "hs005_stage",
        "hs005_reset_parameters",
        "hs005_valid",
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
        sim_backend: Literal["mujoco", "isaacsim"] | None = None,
        **kwargs,
    ):
        backend = (sim_backend or os.environ.get("HANDSON_SIM_BACKEND", "isaacsim")).lower()
        if backend == "isaac":
            backend = "isaacsim"
        if backend != "isaacsim":
            raise ValueError("B320-HS005 requires the Isaac Sim 5.1 PhysX collection backend")
        kwargs.pop("physics_backend", None)
        kwargs.setdefault("time_limit", 45.0)
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("control_dt", 1.0 / 30.0)
        kwargs.setdefault("physics_dt", 1.0 / 240.0)
        from ...sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

        env = HandsOnABCIsaacLabEnv(
            self.task_name,
            render_mode=render_mode,
            randomize=randomize,
            physics_backend="physx",
            **kwargs,
        )
        self.attach_prompt_overlay(env)
        env = (
            DualArmPolicyWrapper(env)
            if policy_mode
            else DualArmViveHandTeleopWrapper(env, self.teleop)
        )
        return HandsOnObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
