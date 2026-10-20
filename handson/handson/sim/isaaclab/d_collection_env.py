"""Native IsaacLab collection environments for HandsOn D01--D05.

The five tasks share the ROKAE + Wuji robot, controller, cameras, reset cache,
and recorder contract from :class:`HandsOnABCIsaacLabEnv`.  This module only
adds D-series scene randomization, training observations, ordered task state
machines, and live success-condition reporting.
"""

from __future__ import annotations
from .b_eval_setting import setting_for as _b_setting_for

import math
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from .abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _C_TOOL_SPREAD_ARM_HOME,
    _YCB01_CLEAR_ARM_HOME,
    _YCB01_CLEAR_LEFT_START_TCP,
    _YCB01_CLEAR_RIGHT_START_TCP,
    _C_TOOL_SPREAD_LEFT_START_TCP,
    _C_TOOL_SPREAD_RIGHT_START_TCP,
    _STACK_ANGULAR_DAMPING,
    _STACK_CONTACT_OFFSET,
    _STACK_LINEAR_DAMPING,
    _STACK_REST_OFFSET,
    _STACK_SLEEP_THRESHOLD,
    _TASK_ROOT,
    _invert_pose_matrix,
    _pose_vector_to_matrix,
    _matrix_to_quat_wxyz,
    _blend_quat_wxyz,
    _quat_wxyz_to_matrix,
    _randomized_physx_pose,
)
from .d_collection_scene import (
    D_TASK_NAMES,
)
from .recording_cameras import constrain_recording_object_xy
from .xy1_composition import XY1CompositionMixin
from ...tasks.eval_comb03_pass_block_and_close_box import contract as xy1


_D01 = "d01_bimanual_nut_bolt_assembly"
_D02 = "t12_load_and_start_microwave"
_D03 = "t21_pass_block"
_D05 = "t43_squeeze_scissors"
_XY1_DISTRACTORS = (
    "d03_flat_screwdriver",
    "d03_phillips_screwdriver",
    "d03_spoon",
    "d03_plastic_bolt",
    "d03_a_marbles",
    "d03_b_marble_1",
    "d03_b_marble_2",
    "d03_hammer",
    "d03_nails",
    "d03_bowl",
)

_D03_TASKS = frozenset({_D03, xy1.TASK_NAME})

_D_CONTACT_OFFSET = 0.006
_D_REST_OFFSET = 0.0
_D_HAND_CONTACT_FORCE = 0.15
_D02_CONTACT_OFFSET = 0.001
_D02_STATIC_FRICTION = 1.5
_D02_DYNAMIC_FRICTION = 1.5
_D02_PLATE_MASS = 0.18
_D02_PLATE_LINEAR_DAMPING = 0.0
_D02_PLATE_ANGULAR_DAMPING = 0.0
_D03_TARGET_PHYSICS_MATERIAL_PATH = (
    "/World/PhysicsMaterials/HandsOnD03TargetContactSafety"
)
_D03_TARGET_STATIC_FRICTION = 4.0
_D03_TARGET_DYNAMIC_FRICTION = 2.5
_D03_TARGET_ENABLE_CCD = False
_D03_TARGET_ENABLE_SPECULATIVE_CCD = True
_D03_TARGET_SOLVER_POSITION_ITERATIONS = 64
_D03_TARGET_SOLVER_VELOCITY_ITERATIONS = 8
# A 20 g block can otherwise receive a large separating impulse when several
# driven fingertips first close around it.  Bound the *physical* contact
# correction and body velocity so the block remains manipulable instead of
# being launched across the workspace; no pose is restored or scripted.
_D03_TARGET_MAX_DEPENETRATION_VELOCITY = 0.15
_D03_TARGET_MAX_LINEAR_VELOCITY = 0.25
_D03_TARGET_MAX_ANGULAR_VELOCITY = 4.0
_D03_HAND_MAX_DEPENETRATION_VELOCITY = 0.15
_D03_HAND_MAX_LINEAR_VELOCITY = 2.0
_D03_HAND_MAX_ANGULAR_VELOCITY = 12.0
_D03_HAND_SOLVER_POSITION_ITERATIONS = 32
_D03_HAND_SOLVER_VELOCITY_ITERATIONS = 4
_D03_TARGET_MASS = 0.02
_D03_RELEASE_DISTANCE = 0.085
# Backwards-compatible name used by downstream task-layout checks.
_D03_B04_PHYSICS_MATERIAL_PATH = _D03_TARGET_PHYSICS_MATERIAL_PATH
# The door-open metrics are retained for diagnostics only; D02 success now
# gates on the door being closed before the green start button is pressed.
_D02_DOOR_OPEN_THRESHOLD = -1.45
_D02_DOOR_LOAD_THRESHOLD = -1.45
# The task-only MJCF drops the robot include that declares radian angles, so
# Isaac initially imports D02's ``-1.6`` lower limit as -1.6 degrees.  Keep
# that tiny limit for the fragile first PhysX reset, then replace it on the
# live articulation with a useful physical span before collection begins.
_D02_LIVE_DOOR_LOWER_LIMIT = -1.55

# The imported plastic nut/bolt meshes do not expose a native helical joint.
# Keep the interaction dynamic by applying a small, equal-and-opposite PhysX
# force that follows a coarse plastic thread, and require the measured motion
# to agree with the same pitch before success can latch.
_D01_THREAD_PITCH = 0.006
_D01_REQUIRED_TWIST_DEG = 90.0
_D01_REQUIRED_AXIAL_TRAVEL = 0.0010
_D01_MAX_HELICAL_ERROR = 0.0025
_D01_RADIAL_STIFFNESS = 32.0
_D01_AXIAL_STIFFNESS = 45.0
_D01_THREAD_DAMPING = 0.12
_D01_MAX_THREAD_FORCE = 0.75
_D01_DISENGAGE_GRACE_SUBSTEPS = 8
_D01_BOLT_HEAD_RADIUS = 0.0085
_D01_BOLT_SHAFT_RADIUS = 0.00445
_D01_BOLT_HEAD_FAR_LOCAL_Z = 0.007
_D01_BOLT_SHAFT_FAR_LOCAL_Z = 0.058
_D01_BOLT_REST_ROLL_RAD = math.pi / 2.0 + math.atan2(
    _D01_BOLT_HEAD_RADIUS - _D01_BOLT_SHAFT_RADIUS,
    _D01_BOLT_SHAFT_FAR_LOCAL_Z - _D01_BOLT_HEAD_FAR_LOCAL_Z,
)
_D01_BOLT_REST_QUAT_XYZW = np.asarray(
    [
        math.sin(0.5 * _D01_BOLT_REST_ROLL_RAD),
        0.0,
        0.0,
        math.cos(0.5 * _D01_BOLT_REST_ROLL_RAD),
    ],
    dtype=np.float64,
)
_D01_BOLT_REST_SUPPORT_LOCAL_MIN_Z = (
    -_D01_BOLT_HEAD_RADIUS * math.sin(_D01_BOLT_REST_ROLL_RAD)
    + _D01_BOLT_HEAD_FAR_LOCAL_Z * math.cos(_D01_BOLT_REST_ROLL_RAD)
)
_D01_NUT_SUPPORT_LOCAL_MIN_Z = -0.004074

_D_TABLE_TOP_Z = 0.89
_D02_PARKED_XY = (-2.0, 0.0)
_D02_PLATE_SUPPORT_LOCAL_MIN_Z = -0.012
_D02_PLATE_FOOD_SUPPORT_LOCAL_Z = 0.012
_D02_FOOD_SUPPORT_CLEARANCE = 0.001
_D02_FOOD_SUPPORT_LOCAL_MIN_Z = {
    # Both tall packages rest on a broad side instead of balancing upright on
    # their narrowest face.
    "003_cracker_box": -0.0299465,
    "008_pudding_box": -0.017804,
}
_D02_FOOD_REST_QUAT_XYZW = np.asarray(
    [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
    dtype=np.float64,
)
_D_TABLE_HEIGHT_BOUNDS = {
    "eval_gen04_pass_glue_stick": (0.0, 0.0),
    _D01: (0.0, 0.05),
    # The microwave and box are fixed/articulated fixtures.  Keep
    # their table height deterministic and randomize the movable objects.
    _D02: (0.0, 0.0),
    _D03: (0.0, 0.0),
    xy1.TASK_NAME: (0.0, 0.0),
    _D05: (0.0, 0.05),
}

_D03_DISTRACTORS = (
    "d03_flat_screwdriver",
    "d03_phillips_screwdriver",
    "d03_spoon",
    "d03_plastic_bolt",
    "d03_a_marbles",
    "d03_b_marble_1",
    "d03_b_marble_2",
    "d03_hammer",
    "d03_nails",
    "d03_bowl",
)
_D01_NUT_CONTACT_BODY = "nut_object"
_D01_BOLT_CONTACT_BODY = "bolt_object"
_D03_CONTACT_BODY = "d03_target_root"
_D_CONTACT_TARGETS = {
    "eval_gen04_pass_glue_stick": ("gen4_glue",),
    # The imported fastener collision shapes belong to fixed child links;
    # their free-joint roots remain the logical pose/replay/force bodies.
    _D01: (_D01_NUT_CONTACT_BODY, _D01_BOLT_CONTACT_BODY),
    _D02: ("d02_start_button_body",),
    # Match B04: the collision geom, free joint, pose, and replay state all
    # belong to one rigid body, so grasp impulses cannot cross a fixed child.
    _D03: (_D03_CONTACT_BODY,),
    xy1.TASK_NAME: (_D03_CONTACT_BODY,),
    _D05: ("d05_scissors_fixed", "d05_scissors_moving"),
}
_D_CONTACT_POSE_ALIASES = {
    _D01_NUT_CONTACT_BODY: "nut_root_body",
    _D01_BOLT_CONTACT_BODY: "bolt_root_body",
}
_D_COLLISION_SAFE_OBJECTS = {
    xy1.TASK_NAME: (xy1.BLOCK, xy1.LID, xy1.BOX, *xy1.DISTRACTORS),
    _D01: (
        "nut_root_body",
        "bolt_root_body",
    ),
    _D02: (
        "d02_plate",
        "003_cracker_box",
        "008_pudding_box",
        "microwave_object",
    ),
    _D03: (
        "d03_target_root",
        "d03_box",
        *_D03_DISTRACTORS,
    ),
    _D05: ("d05_scissors_root_body",),
}

_D_RESET_MAX_ATTEMPTS = 128
_D_RESET_CANDIDATES_PER_OBJECT = 96
_D_RESET_MIN_CLEARANCE_SLACK = 0.005
# Keep the microwave close on X, but offset it slightly away from the left-hand
# door-swing side so the door can open from the shared robot reset.
_D02_MICROWAVE_XY = np.asarray([-0.05, -0.04], dtype=np.float64)
_D02_PLATE_DEFAULT_XY = np.asarray([-0.34, -0.54], dtype=np.float64)
_D02_PLATE_XY_BOUNDS = (
    # Stage the complete 26 cm plate in the reachable right-hand lane.  Its
    # nearest edge remains below y=-0.37, clear of both the microwave plinth
    # (y >= -0.34) and the door's complete physical sweep (y >= -0.225).
    # This 18 x 8 cm region triples the old X span and expands total area.
    np.asarray([-0.42, -0.58], dtype=np.float64),
    np.asarray([-0.24, -0.50], dtype=np.float64),
)
_D02_PLATE_RESET_HAND_MIN_XY_DISTANCE = 0.22
_D03_BOX_DEFAULT_XY = np.asarray([-0.24, -0.58], dtype=np.float64)
_D03_BOX_ROOT_Z = 0.9645
_D03_TARGET_HALF_EXTENT = 0.014925
_D03_BOX_INNER_HALF_EXTENTS = np.asarray(
    [0.138, 0.206], dtype=np.float64
) - _D03_TARGET_HALF_EXTENT - 0.001
_D03_BOX_XY_BOUNDS = (
    _D03_BOX_DEFAULT_XY.copy(),
    _D03_BOX_DEFAULT_XY.copy(),
)
_D03_BOX_RIGHT_HAND_MIN_XY_DISTANCE = 0.20
_D03_OBJECT_Z = {
    "d03_flat_screwdriver": 0.896,
    "d03_phillips_screwdriver": 0.896,
    "d03_spoon": 0.894,
    "d03_plastic_bolt": 0.895,
    "d03_a_marbles": 0.927,
    "d03_b_marble_1": 0.9075,
    "d03_b_marble_2": 0.9075,
    "d03_hammer": 0.904,
    "d03_nails": 0.8922,
    "d03_bowl": 0.916,
}
_D03_OBJECT_XY_RADIUS = {
    "d03_flat_screwdriver": 0.097,
    "d03_phillips_screwdriver": 0.097,
    "d03_spoon": 0.088,
    "d03_plastic_bolt": 0.019,
    "d03_a_marbles": 0.074,
    "d03_b_marble_1": 0.018,
    "d03_b_marble_2": 0.018,
    "d03_hammer": 0.160,
    "d03_nails": 0.011,
    "d03_bowl": 0.080,
}
_D03_DISTRACTOR_PAIR_SLACK = 0.015

_D05_SCISSORS_DEFAULT_XY = np.asarray([-0.28, 0.0], dtype=np.float64)
_D05_SCISSORS_XY_BOUNDS = (
    np.asarray([-0.38, -0.20], dtype=np.float64),
    np.asarray([-0.16, 0.20], dtype=np.float64),
)
_D05_SCISSORS_HAND_MIN_XY_DISTANCE = 0.14

_D05_PHASE_NAMES = (
    "await_open",
    "ready_to_open",
    "ready_to_close",
    "complete",
)

_D_SUCCESS_CONDITION_DISPLAY_LABELS = {
    "xy1_left": "Left hand lifted the block first",
    "xy1_both": "Both hands contacted the block in the middle",
    "xy1_right": "Right hand received after left release",
    "xy1_a": "Handoff and released placement completed",
    "xy1_lid_pickup": "Right hand picked up lid after placement",
    "xy1_containment": "Block remains released and stable inside box",
    "b05_full_coverage": "Lid covers more than 90% of the opening",
    "b05_height_close": "Lid at closing height",
    "b05_top_aligned": "Lid lies flat",
    "b05_lid_stable": "Lid stable",
    "b05_base_stable": "Box stable",
    "b05_base_upright": "Box upright",
    "b05_base_on_table": "Box supported by table",
    "b05_hand_released": "Both hands clear of lid and box",
    "d01_roles": "Each hand stably holds one fastener",
    "d01_align": "Nut center, depth, and axis aligned",
    "d01_twist": "Tightened by at least 90 deg",
    "d01_travel": "Nut travel matches thread pitch",
    "d01_stable": "Assembly remains stable",
    "d02_open": "Microwave door closed",
    "d02_inside": "Plate and food inside microwave cavity",
    "d02_on_plate": "Food remains on plate",
    "d02_button": "Green start button pressed while loaded",
    "d03_left": "Left hand lifted target block first",
    "d03_both": "Both hands contacted block in transfer zone",
    "d03_right": "Right hand received after left release",
    "d03_box": "Block released stably inside right box",
    "d05_air": "Scissors lifted at least 35 mm",
    "d05_handles": "Both handles actively held",
    "d05_cuts": "Three full open-close cycles completed",
    "success_hold": "Hold all conditions stable",
}


def _wrapped_angle(current: float, previous: float) -> float:
    return float((current - previous + math.pi) % (2.0 * math.pi) - math.pi)


class HandsOnDIsaacLabEnv(XY1CompositionMixin, HandsOnABCIsaacLabEnv):
    """Real PhysX D-series task layer over the shared collection runtime."""

    def __new__(cls, task_name: str | None = None, **kwargs: Any):
        if cls is HandsOnDIsaacLabEnv and task_name == "eval_gen04_pass_glue_stick":
            from .gen4_env import HandsOnGEN4IsaacLabEnv
            return object.__new__(HandsOnGEN4IsaacLabEnv)
        return object.__new__(cls)

    def __init__(self, task_name: str, **kwargs: Any):
        self._d03_box_name = xy1.BOX if task_name == xy1.TASK_NAME else "d03_box"
        self._xy1_initialize()
        if task_name == xy1.TASK_NAME:
            if str(kwargs.get("physics_backend", "physx")).strip().lower() != "physx":
                raise ValueError("XY1 collection requires Isaac Sim 5.1 with PhysX.")
            kwargs.setdefault("control_dt", 1.0 / 30.0)
            kwargs.setdefault("physics_dt", 1.0 / 120.0)
            kwargs.setdefault("time_limit", 300.0)
        if task_name not in D_TASK_NAMES:
            supported = ", ".join(sorted(D_TASK_NAMES))
            raise ValueError(
                f"Unsupported IsaacLab D task {task_name!r}; expected one of: {supported}"
            )

        # These fields must exist before ``super`` launches Isaac: setup calls
        # the virtual reset/observation hooks after the first simulation reset.
        self._d02_constructed_state_pending_adoption = bool(
            task_name == _D02 and kwargs.get("launch_isaac", True)
        )
        # The adopted scene already holds the robots at their authored home;
        # avoid a redundant post-step joint-position transaction as well.
        # D03 likewise keeps its stable live reset target: restoring raw USD
        # defaults moves the hands through dense tabletop clutter.
        self.stable_recording_origin_default_robot_restore_enabled = bool(
            task_name not in {_D02, _D03, "eval_gen04_pass_glue_stick"}
        )
        self._d_last_metrics: dict[str, Any] = {}
        self._d_prev_object_poses: dict[str, np.ndarray] = {}
        self._d_hand_contact_views: dict[tuple[str, str], Any] = {}
        self._d_contact_view_errors: set[tuple[str, str]] = set()
        self._d01_thread_force_warning_emitted = False
        self._d01_disengaged_substeps = 0
        self._d01_prev_phase: float | None = None
        self._d01_signed_twist_deg = 0.0
        self._d01_physical_twist_deg = 0.0
        self._d01_thread_entry_depth: float | None = None
        self._d01_min_thread_depth: float | None = None
        self._d01_current_axial_travel = 0.0
        self._d01_helical_error = math.inf
        self._d01_thread_engaged = False
        self._d01_bimanual_turn_seen = False
        self._d01_role_assignment = "unassigned"
        self._d02_active_food = "003_cracker_box"
        self._d02_stage = 0
        self._d02_open_count = 0
        self._d02_load_count = 0
        self._d02_close_count = 0
        self._d02_door_opened = False
        self._d02_payload_loaded = False
        self._d02_door_closed = False
        self._d02_button_armed = False
        self._d02_live_door_limit_configured = False
        self._d03_stage = 0
        self._d03_left_streak = 0
        self._d03_both_streak = 0
        self._d03_right_streak = 0
        self._d03_left_confirmed = False
        self._d03_both_confirmed = False
        self._d03_right_confirmed = False
        self._d03_direct_right_seen = False
        self._d05_cut_count = 0
        self._d05_phase = 0
        self._d05_threshold_streak = 0
        self._d05_inactive_substeps = 0
        self._d05_previous_hinge = 0.68
        super().__init__(task_name=task_name, **kwargs)
        if task_name == _D02 and self._launch_isaac:
            self._configure_d02_live_door_limit()

    def _configure_d02_live_door_limit(self) -> float:
        """Unlock the passive D02 hinge after its safe construction reset.

        Importing the wide range directly makes Isaac Sim 5.1 fail GPU
        narrowphase during the first reset.  Updating the reduced-coordinate
        limit after construction changes only the physical joint constraint;
        it never writes the door pose, velocity, drive target, or trajectory.
        """

        dof_entry = self._task_dof_views.get("microjoint")
        if dof_entry is None or self._torch is None:
            raise RuntimeError(
                "D02 live PhysX scene is missing the microwave door DOF"
            )
        view, dof_index, _authored_position = dof_entry
        limits = view.get_dof_limits().clone()
        limits[:, int(dof_index), 0] = _D02_LIVE_DOOR_LOWER_LIMIT
        limits[:, int(dof_index), 1] = 0.0
        indices = self._torch.arange(
            int(view.count),
            dtype=self._torch.long,
            device="cpu",
        )
        view.set_dof_limits(limits.cpu(), indices)
        self._d02_live_door_limit_configured = True
        print(
            "[HandsOn/IsaacLab] D02 passive microwave door unlocked after "
            f"safe construction reset (lower={_D02_LIVE_DOOR_LOWER_LIMIT:.3f} rad)",
            flush=True,
        )
        return _D02_LIVE_DOOR_LOWER_LIMIT

    def _task_articulation_root_is_writable(self, path: str) -> bool:
        """Protect D02's world-fixed microwave root from tensor writes."""

        if self.task_name != _D02:
            return super()._task_articulation_root_is_writable(path)
        normalized = str(path).rstrip("/")
        return not normalized.endswith(
            "/microwave_object/microwave_object"
        )

    def _task_articulation_dofs_are_writable(self, path: str) -> bool:
        """Protect D02's fixed articulation DOFs from post-start writes."""

        if self.task_name != _D02:
            return super()._task_articulation_dofs_are_writable(path)
        normalized = str(path).rstrip("/")
        return not normalized.endswith(
            "/microwave_object/microwave_object"
        )

    def _task_articulation_root_velocity_is_writable(self, path: str) -> bool:
        """Allow D02's fixed root velocity to enter exact-zero boundaries.

        Root pose writes remain forbidden.  A velocity-only tensor write was
        exercised across following simulation frames and is safe; unlike the
        PhysX sleep transaction, it neither changes the fixed anchor nor
        invalidates the microwave articulation.
        """

        return True

    def _task_articulation_dof_velocity_is_writable(self, path: str) -> bool:
        """Allow exact-zero D02 door/button velocities without changing q."""

        return True

    def _task_detailed_contact_sensor_is_enabled(self, path: str) -> bool:
        """Skip the world-fixed D02 root while retaining door/body reports."""

        if self.task_name != _D02:
            return super()._task_detailed_contact_sensor_is_enabled(path)
        normalized = str(path).rstrip("/")
        return not normalized.endswith(
            "/microwave_object/microwave_object"
        )

    def reset(self, seed: int | None = None, **kwargs: Any):
        """Adopt D02's freshly constructed scene instead of resetting it.

        Isaac Sim 5.1 can invalidate GPU articulation memory when D02's live
        scene is reset in place after construction.  Construction has already
        applied the requested seed, reset all task/controller state, and run
        five successful PhysX steps.  Formal later episodes reconstruct the
        whole PhysX Scene, which sets this one-shot adoption flag again.
        """

        if not self._d02_constructed_state_pending_adoption:
            return super().reset(seed=seed, **kwargs)
        if self._sim is None:
            self._d02_constructed_state_pending_adoption = False
            return super().reset(seed=seed, **kwargs)
        if seed is not None and int(seed) != int(self._seed):
            raise RuntimeError(
                "D02 constructed-scene adoption requires the constructor "
                f"seed {self._seed}, not {int(seed)}; rebuild the PhysX Scene"
            )

        self._d02_constructed_state_pending_adoption = False
        self._manual_reset_requested = False
        self._manual_reset_already_applied = False
        self._manual_reset_viewport_camera_state = None
        self.env_step = 0
        self._success_counter = 0
        self._reset_physics_steps = 0
        self._begin_recording_origin_reset_write_capture()
        self._capture_recording_origin_authored_reset_snapshot(
            require_authored_task_writes=False,
        )
        self._refresh_object_poses()
        info: dict[str, Any] = {
            "succeed": False,
            "constructed_scene_adopted": True,
            "isaac_warnings": tuple(self._isaac_warnings),
        }
        info.update(self._task_info())
        print(
            "[HandsOn/IsaacLab] D02 adopted the fresh constructed PhysX "
            "scene; redundant in-place reset skipped",
            flush=True,
        )
        observation = self._attach_physx_ground_truth(
            self._compute_observation()
        )
        return observation, info

    def _configure_d03_target_stability(self, sim_utils: Any) -> bool:
        """Keep D03's 20 g handoff block inside a bounded contact envelope.

        D03 uses B04's visual/collider but needs stricter velocity and mass
        bounds for two-hand squeezing in GPU PhysX.
        """
        from .b_eval_setting import setting_for
        selection = setting_for(self.task_name)
        if selection is not None:
            from . import b_eval_d03_physics
            return getattr(b_eval_d03_physics, 'configure_' + selection['setting_id'])(self, sim_utils)
        if self._stage is None:
            return False

        target_prim = self._find_rigid_body_prim("d03_target_root")
        if target_prim is None:
            return False
        target_path = str(target_prim.GetPath())
        # B04 first breaks the imported instance before authoring collider
        # material/contact attributes.  D03 must follow that same sequence;
        # binding on the instance root is silently ignored by Isaac Sim 5.1.
        sim_utils.make_uninstanceable(target_path, stage=self._stage)
        sim_utils.update_stage()
        target_prim = self._find_rigid_body_prim("d03_target_root")
        if target_prim is None:
            return False

        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=_D03_TARGET_STATIC_FRICTION,
            dynamic_friction=_D03_TARGET_DYNAMIC_FRICTION,
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="min",
        )
        material_cfg.func(_D03_TARGET_PHYSICS_MATERIAL_PATH, material_cfg)
        sim_utils.bind_physics_material(
            target_path,
            _D03_TARGET_PHYSICS_MATERIAL_PATH,
            stage=self._stage,
            stronger_than_descendants=True,
        )

        from pxr import PhysxSchema, Usd, UsdPhysics

        mass_api_factory = getattr(UsdPhysics, "MassAPI", None)
        if mass_api_factory is not None:
            mass_api = mass_api_factory(target_prim)
            if not mass_api:
                mass_api = mass_api_factory.Apply(target_prim)
            mass_attr = mass_api.GetMassAttr()
            if not mass_attr or not mass_attr.IsValid():
                mass_attr = mass_api.CreateMassAttr()
            mass_attr.Set(_D03_TARGET_MASS)

        collider_count = 0
        for descendant in Usd.PrimRange(target_prim):
            if not descendant.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collision_api = PhysxSchema.PhysxCollisionAPI(descendant)
            if not collision_api:
                collision_api = PhysxSchema.PhysxCollisionAPI.Apply(descendant)
            contact_attr = collision_api.GetContactOffsetAttr()
            if not contact_attr or not contact_attr.IsValid():
                contact_attr = collision_api.CreateContactOffsetAttr()
            contact_attr.Set(_STACK_CONTACT_OFFSET)
            rest_attr = collision_api.GetRestOffsetAttr()
            if not rest_attr or not rest_attr.IsValid():
                rest_attr = collision_api.CreateRestOffsetAttr()
            rest_attr.Set(_STACK_REST_OFFSET)
            collider_count += 1

        rigid_api = PhysxSchema.PhysxRigidBodyAPI(target_prim)
        if not rigid_api:
            rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(target_prim)
        for getter_name, creator_name, value in (
            (
                "GetEnableCCDAttr",
                "CreateEnableCCDAttr",
                _D03_TARGET_ENABLE_CCD,
            ),
            (
                "GetEnableSpeculativeCCDAttr",
                "CreateEnableSpeculativeCCDAttr",
                _D03_TARGET_ENABLE_SPECULATIVE_CCD,
            ),
            (
                "GetSolverPositionIterationCountAttr",
                "CreateSolverPositionIterationCountAttr",
                _D03_TARGET_SOLVER_POSITION_ITERATIONS,
            ),
            (
                "GetSolverVelocityIterationCountAttr",
                "CreateSolverVelocityIterationCountAttr",
                _D03_TARGET_SOLVER_VELOCITY_ITERATIONS,
            ),
            (
                "GetMaxDepenetrationVelocityAttr",
                "CreateMaxDepenetrationVelocityAttr",
                _D03_TARGET_MAX_DEPENETRATION_VELOCITY,
            ),
            (
                "GetMaxLinearVelocityAttr",
                "CreateMaxLinearVelocityAttr",
                _D03_TARGET_MAX_LINEAR_VELOCITY,
            ),
            (
                "GetMaxAngularVelocityAttr",
                "CreateMaxAngularVelocityAttr",
                _D03_TARGET_MAX_ANGULAR_VELOCITY,
            ),
            (
                "GetLinearDampingAttr",
                "CreateLinearDampingAttr",
                _STACK_LINEAR_DAMPING,
            ),
            (
                "GetAngularDampingAttr",
                "CreateAngularDampingAttr",
                _STACK_ANGULAR_DAMPING,
            ),
            (
                "GetSleepThresholdAttr",
                "CreateSleepThresholdAttr",
                _STACK_SLEEP_THRESHOLD,
            ),
        ):
            getter = getattr(rigid_api, getter_name, None)
            creator = getattr(rigid_api, creator_name, None)
            if getter is None:
                continue
            attribute = getter()
            if not attribute or not attribute.IsValid():
                if creator is None:
                    continue
                attribute = creator()
            attribute.Set(value)
        return collider_count > 0

    def _configure_task_collision_safety(
        self, sim_utils: Any
    ) -> tuple[int, int]:
        """Add contact shells and anti-tunneling CCD for delicate D objects.

        Imported MJCF mesh colliders are usually instance proxies.  Make only
        the relevant task roots editable, then tune their real PhysX shapes.
        Robot articulation links stay instanced; speculative CCD and inherited
        hand materials can be authored on the editable link prims themselves.
        """
        object_names = _D_COLLISION_SAFE_OBJECTS.get(self.task_name)
        if self._stage is None or not object_names:
            return 0, 0
        if self.task_name in {_D02, *_D03_TASKS}:
            # These scenes already author their real collision geometry and
            # materials in MJCF.  Uninstancing their converted subtrees and
            # replacing every contact shell immediately before the first
            # full-robot reset is unsafe in Isaac Sim 5.1: it can corrupt the
            # GPU narrowphase table (CUDA 700), while the old 6 mm speculative
            # shell gives visibly premature separating impulses.
            # Preserve the converted collider topology; body-level speculative
            # CCD below does not uninstance or replace shapes.
            # ContactReportAPI metadata supports the ordered success checks.
            from pxr import PhysxSchema

            reporter_count = 0
            for body_name in _D_CONTACT_TARGETS[self.task_name]:
                body_prim = self._find_rigid_body_prim(body_name)
                if body_prim is None:
                    self._add_task_state_warning(
                        "Could not enable native contact reporting for "
                        f"{body_name}"
                    )
                    continue
                if body_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    report_api = PhysxSchema.PhysxContactReportAPI.Get(
                        self._stage, body_prim.GetPath()
                    )
                else:
                    report_api = PhysxSchema.PhysxContactReportAPI.Apply(
                        body_prim
                    )
                report_api.CreateThresholdAttr().Set(0.0)
                reporter_count += 1
            # D02 must keep its imported task and hand rigid-body properties
            # untouched before the first PhysX reset.  Binding materials or
            # applying PhysxRigidBodyAPI opinions to those articulation links
            # corrupts Isaac Sim 5.1's GPU narrowphase table (CUDA 700).  The
            # source MJCF already contains physical collision materials and
            # the explicit door/payload inertias needed for manipulation.
            d03_target_tuned = (
                self._configure_d03_target_stability(sim_utils)
                if self.task_name in _D03_TASKS
                else False
            )
            d03_hand_links_tuned = 0
            if self.task_name in _D03_TASKS and not _b_setting_for(self.task_name):
                traverse = getattr(self._stage, "Traverse", None)
                prims = traverse() if callable(traverse) else ()
                for prim in prims:
                    path_lower = str(prim.GetPath()).lower()
                    if not path_lower.startswith(
                        "/world/handsonrobots/"
                    ):
                        continue
                    if (
                        "palm" not in path_lower
                        and "finger" not in path_lower
                    ):
                        continue
                    rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
                    if not rigid_api:
                        continue
                    for getter_name, creator_name, value in (
                        (
                            "GetMaxDepenetrationVelocityAttr",
                            "CreateMaxDepenetrationVelocityAttr",
                            _D03_HAND_MAX_DEPENETRATION_VELOCITY,
                        ),
                        (
                            "GetMaxLinearVelocityAttr",
                            "CreateMaxLinearVelocityAttr",
                            _D03_HAND_MAX_LINEAR_VELOCITY,
                        ),
                        (
                            "GetMaxAngularVelocityAttr",
                            "CreateMaxAngularVelocityAttr",
                            _D03_HAND_MAX_ANGULAR_VELOCITY,
                        ),
                        (
                            "GetSolverPositionIterationCountAttr",
                            "CreateSolverPositionIterationCountAttr",
                            _D03_HAND_SOLVER_POSITION_ITERATIONS,
                        ),
                        (
                            "GetSolverVelocityIterationCountAttr",
                            "CreateSolverVelocityIterationCountAttr",
                            _D03_HAND_SOLVER_VELOCITY_ITERATIONS,
                        ),
                    ):
                        attribute = getattr(rigid_api, getter_name)()
                        if not attribute or not attribute.IsValid():
                            attribute = getattr(
                                rigid_api, creator_name
                            )()
                        attribute.Set(value)
                    d03_hand_links_tuned += 1
            if (
                reporter_count
                or d03_target_tuned
                or d03_hand_links_tuned
            ):
                sim_utils.update_stage()
            print(
                "[HandsOn/IsaacLab] D collision physics: converted collider "
                f"topology preserved for {self.task_name}, "
                f"contact_reporters={reporter_count}, "
                f"d03_target_stability={d03_target_tuned}, "
                f"d03_hand_links_tuned={d03_hand_links_tuned}",
                flush=True,
            )
            if self.task_name in _D03_TASKS and not d03_target_tuned:
                self._add_task_state_warning(
                    "Could not apply B04 small-block contact physics to D03"
                )
            if self.task_name in _D03_TASKS and not _b_setting_for(self.task_name) and d03_hand_links_tuned == 0:
                self._add_task_state_warning(
                    "Could not apply D03 contact stability to robot hand links"
                )
            return 0, 0

        from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

        task_paths: set[str] = set()
        for name in object_names:
            prim = self._find_prim(name)
            if prim is None or not prim.IsValid():
                self._add_task_state_warning(
                    f"Could not collision-tune D task object: {name}"
                )
                continue
            top_prim = prim
            while top_prim.GetParent().IsValid() and str(
                top_prim.GetParent().GetPath()
            ) != _TASK_ROOT:
                top_prim = top_prim.GetParent()
            top_path = str(top_prim.GetPath())
            if top_path.startswith(f"{_TASK_ROOT}/"):
                task_paths.add(top_path)

        for path in sorted(task_paths):
            sim_utils.make_uninstanceable(path, stage=self._stage)
        if task_paths:
            sim_utils.update_stage()

        material_path = "/World/PhysicsMaterials/HandsOnDContactSafety"
        is_scissors = self.task_name == _D05
        is_microwave = self.task_name == _D02
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=(
                1.55 if is_scissors else
                _D02_STATIC_FRICTION if is_microwave else 1.25
            ),
            dynamic_friction=(
                1.25 if is_scissors else
                _D02_DYNAMIC_FRICTION if is_microwave else 1.0
            ),
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="min",
        )
        material_cfg.func(material_path, material_cfg)

        tuned_bodies = 0
        tuned_colliders = 0
        d02_plate_body_tuned = False

        def tune_rigid_body(prim: Any) -> None:
            nonlocal tuned_bodies
            rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not rigid_api:
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            for getter_name, creator_name, value in (
                ("GetEnableSpeculativeCCDAttr", "CreateEnableSpeculativeCCDAttr", True),
                (
                    "GetSolverPositionIterationCountAttr",
                    "CreateSolverPositionIterationCountAttr",
                    64 if is_scissors else 32,
                ),
                (
                    "GetSolverVelocityIterationCountAttr",
                    "CreateSolverVelocityIterationCountAttr",
                    8 if is_scissors else 4,
                ),
                (
                    "GetMaxDepenetrationVelocityAttr",
                    "CreateMaxDepenetrationVelocityAttr",
                    0.35 if is_scissors else 0.75,
                ),
            ):
                attribute = getattr(rigid_api, getter_name)()
                if not attribute or not attribute.IsValid():
                    attribute = getattr(rigid_api, creator_name)()
                attribute.Set(value)
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                report_api = PhysxSchema.PhysxContactReportAPI.Get(
                    self._stage, prim.GetPath()
                )
            else:
                report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report_api.CreateThresholdAttr().Set(0.0)
            tuned_bodies += 1

        for path in sorted(task_paths):
            task_root = self._stage.GetPrimAtPath(path)
            sim_utils.bind_physics_material(
                path,
                material_path,
                stage=self._stage,
                stronger_than_descendants=True,
            )
            for prim in Usd.PrimRange(task_root):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    tune_rigid_body(prim)
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                collision_api = PhysxSchema.PhysxCollisionAPI(prim)
                if not collision_api:
                    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                contact_attr = collision_api.GetContactOffsetAttr()
                if not contact_attr or not contact_attr.IsValid():
                    contact_attr = collision_api.CreateContactOffsetAttr()
                contact_attr.Set(
                    0.004 if is_scissors else
                    _D02_CONTACT_OFFSET if is_microwave else _D_CONTACT_OFFSET
                )
                rest_attr = collision_api.GetRestOffsetAttr()
                if not rest_attr or not rest_attr.IsValid():
                    rest_attr = collision_api.CreateRestOffsetAttr()
                rest_attr.Set(_D_REST_OFFSET)
                tuned_colliders += 1

        if is_microwave:
            plate_prim = self._find_rigid_body_prim("d02_plate")
            if plate_prim is None:
                self._add_task_state_warning(
                    "Could not tune D02 plate mass and damping"
                )
            else:
                mass_api = UsdPhysics.MassAPI(plate_prim)
                if not mass_api:
                    mass_api = UsdPhysics.MassAPI.Apply(plate_prim)
                mass_attr = mass_api.GetMassAttr()
                if not mass_attr or not mass_attr.IsValid():
                    mass_attr = mass_api.CreateMassAttr()
                mass_attr.Set(_D02_PLATE_MASS)

                plate_rigid_api = PhysxSchema.PhysxRigidBodyAPI(plate_prim)
                if not plate_rigid_api:
                    plate_rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(
                        plate_prim
                    )
                linear_attr = plate_rigid_api.GetLinearDampingAttr()
                if not linear_attr or not linear_attr.IsValid():
                    linear_attr = plate_rigid_api.CreateLinearDampingAttr()
                linear_attr.Set(_D02_PLATE_LINEAR_DAMPING)
                angular_attr = plate_rigid_api.GetAngularDampingAttr()
                if not angular_attr or not angular_attr.IsValid():
                    angular_attr = plate_rigid_api.CreateAngularDampingAttr()
                angular_attr.Set(_D02_PLATE_ANGULAR_DAMPING)
                d02_plate_body_tuned = True

        robot_root = self._stage.GetPrimAtPath("/World/HandsOnRobots")
        d02_hand_links = 0
        hand_material = (
            UsdShade.Material(self._stage.GetPrimAtPath(material_path))
            if is_microwave
            else None
        )
        if robot_root.IsValid():
            for prim in Usd.PrimRange(robot_root):
                path_lower = str(prim.GetPath()).lower()
                if "palm" not in path_lower and "finger" not in path_lower:
                    continue
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    tune_rigid_body(prim)
                    if is_microwave and hand_material is not None:
                        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                        binding_api.Bind(
                            hand_material,
                            bindingStrength=(
                                UsdShade.Tokens.strongerThanDescendants
                            ),
                            materialPurpose="physics",
                        )
                        d02_hand_links += 1
        if is_microwave:
            print(
                "[HandsOn/IsaacLab] D02 plate grasp physics: "
                f"plate_body_tuned={d02_plate_body_tuned}, "
                f"mass={_D02_PLATE_MASS:.3f} kg, "
                f"contact_offset={_D02_CONTACT_OFFSET:.4f} m, "
                f"friction={_D02_STATIC_FRICTION:.1f}, "
                f"hand_links={d02_hand_links}, "
                "hand_colliders=instanced",
                flush=True,
            )
            if d02_hand_links == 0:
                self._add_task_state_warning(
                    "D02 plate grasp physics found no hand rigid links"
                )

        if tuned_colliders == 0:
            self._add_task_state_warning(
                f"{self.task_name} collision safety found no editable task colliders"
            )
        return tuned_bodies, tuned_colliders

    # ------------------------------------------------------------------
    # Spaces, reset state, and native PhysX reset writes
    # ------------------------------------------------------------------

    def _build_observation_space(self) -> spaces.Dict:
        common = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "gripper_pose": spaces.Box(-np.inf, np.inf, shape=(32,), dtype=np.float64),
        }
        task_spaces: dict[str, spaces.Space]
        if self.task_name == _D01:
            task_spaces = {
                "nut_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "bolt_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "nut_to_bolt": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
                "assembly_metrics": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
            }
        elif self.task_name == _D02:
            task_spaces = {
                "plate_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "food_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "microwave_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "microwave_joint_state": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64),
                "workflow_progress": spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float64),
                "food_variant": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float64),
            }
        elif self.task_name in {_D03, xy1.TASK_NAME}:
            task_spaces = {
                "target_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "box_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "target_to_box": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
                "handoff_metrics": spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float64),
                "distractor_poses": spaces.Box(-np.inf, np.inf, shape=(70,), dtype=np.float64),
            }
            if self.task_name == xy1.TASK_NAME:
                task_spaces["lid_pose"] = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64)
        else:
            task_spaces = {
                "scissors_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "scissors_hinge": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64),
                "cut_progress": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float64),
            }
        state_spaces = {
            **common,
            **task_spaces,
            "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
        }
        observation_spaces: dict[str, spaces.Space] = {
            "state": spaces.Dict(state_spaces)
        }
        if self.task_name == xy1.TASK_NAME:
            observation_spaces["xy1_hidden_state"] = spaces.Box(
                -np.inf, np.inf, shape=(len(xy1.HIDDEN_FIELDS),), dtype=np.float64
            )
        if self.image_obs:
            image_space = lambda shape=self._image_shape: spaces.Box(
                0, 255, shape=shape, dtype=np.uint8
            )
            observation_spaces["images"] = spaces.Dict(
                {
                    "random_camera" if self.randomize else "front": image_space(),
                    "wrist_left": image_space((*reversed(self._wrist_recording_camera_output_size), 3)),
                    "wrist_right": image_space((*reversed(self._wrist_recording_camera_output_size), 3)),
                }
            )
        return spaces.Dict(observation_spaces)

    def _apply_raised_hand_home(self, robot_cfg: Any, side: str) -> bool:
        if self.task_name == xy1.TASK_NAME:
            arm_home = _YCB01_CLEAR_ARM_HOME.get(side)
            if arm_home is None:
                return False
            init_state = getattr(robot_cfg, "init_state", None)
            joint_pos = getattr(init_state, "joint_pos", None)
            if not isinstance(joint_pos, dict):
                return False
            for index, value in enumerate(arm_home, start=1):
                joint_pos[f"joint{index}_{side}(_1)?"] = float(value)
            return True
        if self.task_name != _D02:
            return super()._apply_raised_hand_home(robot_cfg, side)
        arm_home = _C_TOOL_SPREAD_ARM_HOME.get(side)
        if arm_home is None:
            return False
        init_state = getattr(robot_cfg, "init_state", None)
        joint_pos = getattr(init_state, "joint_pos", None)
        if not isinstance(joint_pos, dict):
            return False
        for index, value in enumerate(arm_home, start=1):
            joint_pos[f"joint{index}_{side}(_1)?"] = float(value)
        return True

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        if self.task_name == _D02:
            # Reuse C07--C12's established collision-free bimanual start.
            # Both hands stay at working height and spread sideways; this is
            # not a D-task-specific IK pose.
            self._right_tcp_pose = _C_TOOL_SPREAD_RIGHT_START_TCP.copy()
            self._left_tcp_pose = _C_TOOL_SPREAD_LEFT_START_TCP.copy()
        if self.task_name == xy1.TASK_NAME:
            self._right_tcp_pose = _YCB01_CLEAR_RIGHT_START_TCP.copy()
            self._left_tcp_pose = _YCB01_CLEAR_LEFT_START_TCP.copy()
        low, high = _D_TABLE_HEIGHT_BOUNDS[self.task_name]
        self.delta_h[:] = self._random.uniform(low, high) if self.randomize else low
        self._d_last_metrics = {}
        self._d_prev_object_poses = {}

        self._d01_prev_phase = None
        self._d01_signed_twist_deg = 0.0
        self._d01_physical_twist_deg = 0.0
        self._d01_thread_entry_depth = None
        self._d01_min_thread_depth = None
        self._d01_current_axial_travel = 0.0
        self._d01_helical_error = math.inf
        self._d01_thread_engaged = False
        self._d01_disengaged_substeps = 0
        self._d01_bimanual_turn_seen = False
        self._d01_role_assignment = "unassigned"

        self._d02_active_food = str(
            self._random.choice(("003_cracker_box", "008_pudding_box"))
        )
        from .b_eval_setting import setting_for
        selection = setting_for(self.task_name)
        if selection and self.task_name == _D02:
            self._d02_active_food = {"cracker_box": "003_cracker_box", "pudding_box": "008_pudding_box"}[selection["setting_id"]]
        self._d02_stage = 0
        self._d02_open_count = 0
        self._d02_load_count = 0
        self._d02_close_count = 0
        self._d02_door_opened = False
        self._d02_payload_loaded = False
        self._d02_door_closed = False
        self._d02_button_armed = False

        self._d03_stage = 0
        self._d03_left_streak = 0
        self._d03_both_streak = 0
        self._d03_right_streak = 0
        self._d03_left_confirmed = False
        self._d03_both_confirmed = False
        self._d03_right_confirmed = False
        self._d03_direct_right_seen = False

        if self.task_name == xy1.TASK_NAME:
            self._target_object_name = xy1.BLOCK
            self._xy1_reset()

        if self.randomize:
            # Reserved reset draws.  This shared D-series reset contract has
            # always consumed two draws at this point: five candidates sampled
            # without replacement, then a two-way choice.  Preserve the exact
            # stream so the remaining D tasks keep their published layouts.
            self._random.choice(range(5), size=3, replace=False)
            self._random.choice(range(2))

        self._d05_cut_count = 0
        self._d05_phase = 0
        self._d05_threshold_streak = 0
        self._d05_inactive_substeps = 0
        self._d05_previous_hinge = 0.68

    def _capture_task_scene_state(self, task_mjcf) -> None:
        super()._capture_task_scene_state(task_mjcf)
        if self.task_name != _D05 or self._stage is None or not self._robots:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_sim_view is None:
            return
        for name in ("d05_scissors_fixed", "d05_scissors_moving"):
            prim = self._find_prim(name)
            if prim is None or not prim.IsValid():
                continue
            try:
                view = physics_sim_view.create_rigid_body_view(str(prim.GetPath()))
                if int(view.count) > 0:
                    self._task_body_views[name] = view
            except Exception as exc:
                self._add_task_state_warning(
                    f"Could not bind D05 rigid body {name}: {exc}"
                )

    def _init_task_specific_runtime(self) -> None:
        """Create fail-closed PhysX hand/task contact views for D01--D05."""
        self._d_hand_contact_views = {}
        self._d_contact_view_errors = set()
        if not self._robots or self._stage is None:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_sim_view is None:
            self._add_task_state_warning("D-series PhysX contact reporting is unavailable")
            return

        targets = _D_CONTACT_TARGETS[self.task_name]
        hand_paths = {
            side: self._d_hand_body_paths(side) for side in ("left", "right")
        }
        max_contacts = max(
            64, max((len(paths) for paths in hand_paths.values()), default=0) * 8
        )
        for name in targets:
            object_prim = self._find_rigid_body_prim(name)
            if object_prim is None:
                self._add_task_state_warning(
                    f"D-series contact target rigid body is unavailable: {name}"
                )
                continue
            for side, paths in hand_paths.items():
                key = (side, name)
                if not paths:
                    self._d_contact_view_errors.add(key)
                    continue
                try:
                    view = physics_sim_view.create_rigid_contact_view(
                        str(object_prim.GetPath()),
                        filter_patterns=paths,
                        max_contact_data_count=max_contacts,
                    )
                    if (
                        view is not None
                        and int(view.sensor_count) > 0
                        and int(view.filter_count) > 0
                    ):
                        self._d_hand_contact_views[key] = view
                    else:
                        self._d_contact_view_errors.add(key)
                except Exception as exc:
                    self._d_contact_view_errors.add(key)
                    self._add_task_state_warning(
                        f"Could not bind {side}-hand contact for {name}: {exc}"
                    )

        expected = 2 * len(targets)
        if len(self._d_hand_contact_views) != expected:
            missing = sorted(
                f"{side}:{name}"
                for side in ("left", "right")
                for name in targets
                if (side, name) not in self._d_hand_contact_views
            )
            self._add_task_state_warning(
                "D-series contact reporting is incomplete and affected success "
                f"conditions will fail closed: {','.join(missing)}"
            )

    def _d_hand_body_paths(self, side: str) -> list[str]:
        if self._stage is None:
            return []
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        if not root.IsValid():
            return []
        prefix = f"{side}_finger"
        palm = f"{side}_palm_link"
        return [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and (prim.GetName() == palm or prim.GetName().startswith(prefix))
        ]

    def _d_contact_reading(self, side: str, name: str) -> tuple[float, bool]:
        view = self._d_hand_contact_views.get((side, name))
        if view is None:
            return 0.0, False
        return float(self._c04_contact_force(view)), True

    def _d_any_hand_contact_reading(self, name: str) -> tuple[float, bool]:
        readings = [self._d_contact_reading(side, name) for side in ("left", "right")]
        available = all(reading[1] for reading in readings)
        return float(sum(reading[0] for reading in readings)), available

    def _reset_state_maps(self):
        rigid = {
            path: (view, indices, pose, velocity)
            for path, view, indices, pose, velocity in self._task_reset_body_states
        }
        articulations = {
            path: (view, indices, pose, velocity)
            for path, view, indices, pose, velocity, _q, _qd
            in self._task_reset_articulation_states
        }
        return rigid, articulations

    def _write_reset_pose(
        self,
        name: str,
        *,
        xy: np.ndarray | tuple[float, float] | None = None,
        z: float | None = None,
        yaw_degrees: float = 0.0,
        orientation_xyzw: np.ndarray | None = None,
    ) -> bool:
        prim = self._find_rigid_body_prim(name)
        if prim is None:
            self._add_task_state_warning(f"D task body is unavailable: {name}")
            return False
        path = str(prim.GetPath())
        rigid, articulations = self._reset_state_maps()
        state = self._reset_transform_state_for_path(
            path,
            rigid,
            articulations,
        )
        if state is None:
            self._add_task_state_warning(f"D task reset tensor is unavailable: {name}")
            return False
        state_kind, view, indices, root_pose, root_velocity = state
        pose_np = root_pose.detach().cpu().numpy().copy()
        if orientation_xyzw is not None:
            orientation = np.asarray(orientation_xyzw, dtype=np.float64).reshape(4)
            norm = float(np.linalg.norm(orientation))
            if not np.isfinite(norm) or norm <= 1.0e-12:
                raise ValueError(f"D task reset orientation is invalid: {name}")
            pose_np[:, 3:7] = orientation / norm
        xy_offset = np.zeros(2, dtype=np.float64)
        if xy is not None:
            xy_offset = np.asarray(xy, dtype=np.float64).reshape(2) - pose_np[0, :2]
        z_offset = 0.0 if z is None else float(z) - float(pose_np[0, 2])
        randomized = _randomized_physx_pose(
            pose_np,
            xy_offset=xy_offset,
            z_offset=z_offset,
            yaw_degrees=float(yaw_degrees),
        )
        pose_tensor = self._torch.as_tensor(
            randomized, dtype=root_pose.dtype, device=root_pose.device
        )
        if state_kind == "articulation":
            if not self._task_articulation_root_is_writable(path):
                self._add_task_state_warning(
                    f"Protected fixed articulation root was not moved: {path}"
                )
                return False
            view.set_root_transforms(pose_tensor, indices)
            zero_velocity = self._torch.zeros_like(root_velocity)
            view.set_root_velocities(zero_velocity, indices)
            self._record_authored_task_reset_write(
                path,
                transform_xyzw=pose_tensor,
                velocity=zero_velocity,
            )
        else:
            self._write_rigid_reset_state(
                path,
                view,
                indices,
                pose_tensor,
                self._torch.zeros_like(root_velocity),
            )
        return True

    def _set_task_dof(self, name: str, value: float) -> bool:
        return self._set_task_dof_state(name, value, 0.0)

    def _set_task_dof_state(
        self, name: str, position: float, velocity: float
    ) -> bool:
        entry = self._task_dof_views.get(name)
        if entry is None:
            entry = next(
                (entry for dof_name, entry in self._task_dof_views.items() if name in dof_name),
                None,
            )
        if entry is None:
            self._add_task_state_warning(f"D task joint is unavailable: {name}")
            return False
        view, index, _initial = entry
        articulation_path = next(
            (
                path
                for path, candidate_view, *_rest
                in self._task_reset_articulation_states
                if candidate_view is view
            ),
            None,
        )
        if (
            articulation_path is not None
            and not self._task_articulation_dofs_are_writable(
                articulation_path
            )
        ):
            return False
        positions = view.get_dof_positions().clone()
        velocities = view.get_dof_velocities().clone()
        positions[:, int(index)] = float(position)
        velocities[:, int(index)] = float(velocity)
        indices = self._torch.arange(
            int(view.count), dtype=self._torch.long, device=self._isaac_device
        )
        view.set_dof_positions(positions, indices)
        view.set_dof_velocities(velocities, indices)
        if articulation_path is not None:
            self._record_authored_task_reset_write(
                articulation_path,
                dof_position=positions,
                dof_velocity=velocities,
            )
        return True

    def _random_yaw(self, low: float, high: float) -> float:
        return float(self._random.uniform(low, high)) if self.randomize else 0.0

    def _sample_xy_away_from_reset_hands(
        self,
        default_xy: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        minimum_distance: float,
        sides: tuple[str, ...],
        *,
        constrain_to_recording_workspace: bool = True,
    ) -> np.ndarray:
        """Sample in the task workspace without overlapping reset-home links."""
        constrain_xy = (
            constrain_recording_object_xy
            if constrain_to_recording_workspace
            else lambda xy: np.asarray(xy, dtype=np.float64)
        )
        if not self.randomize:
            return constrain_xy(default_xy)
        hand_points = tuple(
            point
            for side in sides
            for point in self._reset_robot_body_points(side)
        )
        best = constrain_xy(default_xy)
        best_clearance = -math.inf
        for _ in range(_D_RESET_MAX_ATTEMPTS):
            candidate = constrain_xy(self._random.uniform(*bounds))
            clearance = min(
                (
                    float(np.linalg.norm(candidate - point[:2]))
                    for point in hand_points
                ),
                default=math.inf,
            )
            if clearance > best_clearance:
                best = candidate.copy()
                best_clearance = clearance
            if clearance >= minimum_distance:
                return candidate
        return best

    def _d03_reset_distractor_layout(
        self,
        target_xy: np.ndarray,
        box_xy: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build a constant-time, envelope-separated clutter layout."""

        del box_xy  # The fixed box is below every partition used here.
        driver_slots = [
            np.asarray([-0.45, 0.52], dtype=np.float64),
            np.asarray([0.08, 0.50], dtype=np.float64),
        ]
        if self.randomize and bool(self._random.randint(0, 2)):
            driver_slots.reverse()
        layout = {
            "d03_hammer": np.asarray([-0.18, 0.62], dtype=np.float64),
            "d03_flat_screwdriver": driver_slots[0],
            "d03_phillips_screwdriver": driver_slots[1],
            "d03_spoon": np.asarray([-0.44, -0.22], dtype=np.float64),
            "d03_bowl": np.asarray([0.08, -0.22], dtype=np.float64),
            "d03_a_marbles": np.asarray([0.08, 0.72], dtype=np.float64),
        }
        small_names = (
            "d03_plastic_bolt",
            "d03_b_marble_1",
            "d03_b_marble_2",
            "d03_nails",
        )
        small_slots = [
            np.asarray(value, dtype=np.float64)
            for value in (
                (-0.43, 0.10),
                (0.08, 0.10),
                (-0.18, 0.34),
                (0.08, 0.32),
                (-0.43, -0.02),
                (0.08, -0.02),
                (-0.18, -0.34),
            )
        ]
        if self.randomize:
            self._random.shuffle(small_slots)
        for name in small_names:
            radius = _D03_OBJECT_XY_RADIUS[name]
            accepted_index = next(
                (
                    index
                    for index, slot in enumerate(small_slots)
                    if float(np.linalg.norm(slot - target_xy))
                    >= radius + 0.065 + _D03_DISTRACTOR_PAIR_SLACK
                    and all(
                        float(np.linalg.norm(slot - other_xy))
                        >= radius
                        + _D03_OBJECT_XY_RADIUS[other_name]
                        + _D03_DISTRACTOR_PAIR_SLACK
                        for other_name, other_xy in layout.items()
                    )
                ),
                None,
            )
            if accepted_index is None:
                raise RuntimeError("D03 分区初始布局与随机目标位置冲突")
            layout[name] = small_slots.pop(accepted_index)

        if self.randomize:
            for name in layout:
                layout[name] = layout[name] + self._random.uniform(
                    -0.004,
                    0.004,
                    size=2,
                )
        return layout

    def _apply_task_pose_randomization(self) -> None:
        if not self._task_reset_body_states and not self._task_reset_articulation_states:
            return
        delta = float(self.delta_h[0])
        self._write_reset_pose("table", z=delta)
        placed: list[str] = []

        if self.task_name == xy1.TASK_NAME:
            # The XY1 composition samples all movable fixtures together.  It
            # must be applied on every reset, otherwise only the initial USD
            # authoring is updated and later episodes reuse stale poses.
            self._xy1_apply_reset()
            placed.extend(self._xy1_sampled_poses)

        elif self.task_name == _D01:
            bolt_xy = np.asarray([-0.34, 0.14], dtype=np.float64)
            nut_xy = np.asarray([-0.36, -0.18], dtype=np.float64)
            if self.randomize:
                bolt_xy = self._random.uniform(
                    [-0.40, 0.04],
                    [-0.22, 0.26],
                )
                nut_xy = self._random.uniform(
                    [-0.40, -0.28],
                    [-0.22, -0.06],
                )
            self._write_reset_pose(
                "bolt_root_body",
                xy=bolt_xy,
                z=(
                    _D_TABLE_TOP_Z
                    - _D01_BOLT_REST_SUPPORT_LOCAL_MIN_Z
                    + delta
                ),
                yaw_degrees=self._random_yaw(-15.0, 15.0),
                orientation_xyzw=_D01_BOLT_REST_QUAT_XYZW,
            )
            self._write_reset_pose(
                "nut_root_body",
                xy=nut_xy,
                z=_D_TABLE_TOP_Z - _D01_NUT_SUPPORT_LOCAL_MIN_Z + delta,
                yaw_degrees=self._random_yaw(-45.0, 45.0),
            )
            placed.extend(("bolt_root_body", "nut_root_body"))

        elif self.task_name == _D02:
            plate_xy = self._sample_xy_away_from_reset_hands(
                _D02_PLATE_DEFAULT_XY,
                _D02_PLATE_XY_BOUNDS,
                _D02_PLATE_RESET_HAND_MIN_XY_DISTANCE,
                ("right", "left"),
                constrain_to_recording_workspace=False,
            )
            plate_yaw = self._random_yaw(-45.0, 45.0)
            self._write_reset_pose(
                "d02_plate",
                xy=plate_xy,
                z=_D_TABLE_TOP_Z - _D02_PLATE_SUPPORT_LOCAL_MIN_Z + delta,
                yaw_degrees=plate_yaw,
            )
            food_xy = plate_xy.copy()
            if self.randomize:
                # Keep the package over the plate's broad support while
                # varying its grasp point across the plate surface.  Its Y
                # offset only moves away from the nearby microwave plinth, so
                # even the long cracker box stays clear at randomized yaw.
                food_xy += self._random.uniform(
                    [-0.050, -0.040],
                    [0.050, 0.000],
                )
            for name, support_local_min_z in _D02_FOOD_SUPPORT_LOCAL_MIN_Z.items():
                if name == self._d02_active_food:
                    self._write_reset_pose(
                        name,
                        xy=food_xy,
                        z=(
                            _D_TABLE_TOP_Z
                            - _D02_PLATE_SUPPORT_LOCAL_MIN_Z
                            + _D02_PLATE_FOOD_SUPPORT_LOCAL_Z
                            - support_local_min_z
                            + _D02_FOOD_SUPPORT_CLEARANCE
                            + delta
                        ),
                        yaw_degrees=self._random_yaw(-70.0, 70.0),
                        orientation_xyzw=_D02_FOOD_REST_QUAT_XYZW,
                    )
                else:
                    self._write_reset_pose(
                        name,
                        xy=_D02_PARKED_XY,
                        z=-support_local_min_z + _D02_FOOD_SUPPORT_CLEARANCE,
                        orientation_xyzw=_D02_FOOD_REST_QUAT_XYZW,
                    )
            # The microwave root, closed door, and released button are all
            # authored before the first PhysX reset.  Sim 5.1 corrupts this
            # world-fixed GPU articulation when either its root or DOF state
            # is rewritten after stepping, so later episodes use the formal
            # fresh-Scene reset path instead of tensor setters.
            placed.extend(("d02_plate", self._d02_active_food, "microwave_object"))

        elif self.task_name == _D03:
            # This is a large fixed receptacle, not a camera-bounded movable
            # prop.  The generic sampler clips centers to y=-0.30 m, placing
            # the 42.8 cm-wide box around the right-hand home.  Keep its
            # authored safe fixture lane at y=-0.58 m instead.
            box_xy = _D03_BOX_DEFAULT_XY.copy()
            target_xy = np.asarray([-0.24, 0.22], dtype=np.float64)
            if self.randomize:
                best_target_xy = target_xy.copy()
                best_distance = -math.inf
                for _ in range(_D_RESET_MAX_ATTEMPTS):
                    candidate = self._random.uniform(
                        [-0.36, 0.10],
                        [-0.16, 0.28],
                    )
                    distance = float(np.linalg.norm(candidate - box_xy))
                    if distance > best_distance:
                        best_target_xy = candidate.copy()
                        best_distance = distance
                    if distance >= 0.24:
                        best_target_xy = candidate
                        break
                target_xy = best_target_xy.astype(np.float64)
            self._write_reset_pose(
                "d03_box",
                xy=box_xy,
                z=_D03_BOX_ROOT_Z + delta,
            )
            self._write_reset_pose(
                "d03_target_root", xy=target_xy, z=0.905,
                yaw_degrees=self._random_yaw(-25.0, 25.0),
            )
            distractor_layout = self._d03_reset_distractor_layout(
                target_xy,
                box_xy,
            )
            for name in _D03_DISTRACTORS:
                self._write_reset_pose(
                    name,
                    xy=distractor_layout[name],
                    z=_D03_OBJECT_Z[name],
                    yaw_degrees=self._random_yaw(-180.0, 180.0),
                )
            placed.extend(("d03_target_root", "d03_box", *_D03_DISTRACTORS))


        else:
            scissors_xy = self._sample_xy_away_from_reset_hands(
                _D05_SCISSORS_DEFAULT_XY,
                _D05_SCISSORS_XY_BOUNDS,
                _D05_SCISSORS_HAND_MIN_XY_DISTANCE,
                ("right", "left"),
            )
            self._write_reset_pose(
                "d05_scissors_root_body",
                xy=scissors_xy,
                z=0.899 + delta,
                yaw_degrees=self._random_yaw(-35.0, 35.0),
            )
            self._set_task_dof("d05_scissors_hinge", 0.68)
            placed.append("d05_scissors_root_body")

        print(
            "[HandsOn/IsaacLab] Randomized D task poses: "
            f"task={self.task_name}, table_delta_h={delta:.4f}m, "
            f"objects={','.join(placed)}",
            flush=True,
        )

    def _prepare_physics_substep_joint_targets(
        self,
        substep_index: int,
        substep_count: int,
    ) -> str:
        return super()._prepare_physics_substep_joint_targets(
            substep_index,
            substep_count,
        )

    def _after_task_physics_substep(self) -> None:
        if self.task_name not in {_D01, _D05}:
            return
        self._refresh_object_poses()
        if self.task_name == _D01:
            self._update_d01_thread_history()
            self._apply_d01_thread_constraint()
        else:
            self._update_d05_cut_state()

    # ------------------------------------------------------------------
    # Shared task-state helpers
    # ------------------------------------------------------------------

    def _fallback_pose(self, name):
        if self.task_name == xy1.TASK_NAME and name in xy1.NOMINAL_POSITIONS:
            pose = self._xy1_sampled_poses.get(name)
            if pose is not None:
                return pose.copy()
            yaw = math.radians(xy1.NOMINAL_YAW_DEG.get(name, 0.0))
            return np.asarray([*xy1.NOMINAL_POSITIONS[name], math.cos(yaw/2), 0., 0., math.sin(yaw/2)])
        return super()._fallback_pose(name)

    def get_recording_schema(self):
        schema = super().get_recording_schema()
        if schema is None or self.task_name != xy1.TASK_NAME:
            return schema
        schema = dict(schema)
        schema["fields"] = {**schema["fields"], "xy1_hidden_state": {
            "shape": [len(xy1.HIDDEN_FIELDS)], "dtype": "float64",
            "source": "xy1_action_aligned_task_state", "layout": list(xy1.HIDDEN_FIELDS),
            "visibility": "hidden_task_logic", "replay_domain": "task_state",
            "physical_replay": True, "replay_restore_supported": False, "replay_atol": 1e-6,
        }}
        schema["task_state"] = {
            "task_name": xy1.TASK_NAME, "field_names": ["xy1_hidden_state"],
            "hidden_field_names": ["xy1_hidden_state"],
            "capture_mode": "action_aligned_current_and_next_observation",
            "replay_policy": "compare_every_step_never_restore",
        }
        return schema



    def _refresh_object_poses(self) -> None:
        super()._refresh_object_poses()
        if self.task_name == _D05:
            for name in ("d05_scissors_fixed", "d05_scissors_moving"):
                self._poses[name] = self._prim_pose(name, self._fallback_pose(name))

    def _d_robot_state(self) -> tuple[np.ndarray, np.ndarray]:
        tcp = np.concatenate([self._right_tcp_pose, self._left_tcp_pose]).astype(
            np.float64
        )
        gripper = np.zeros(32, dtype=np.float64)
        gripper[: min(16, self._right_hand.size)] = self._right_hand[
            : min(16, self._right_hand.size)
        ]
        gripper[16 : 16 + min(16, self._left_hand.size)] = self._left_hand[
            : min(16, self._left_hand.size)
        ]
        return tcp, gripper

    def _d_hand_points(self, side: str) -> tuple[np.ndarray, ...]:
        return self._a06_hand_points(side)

    def _d_hand_distance_to_point(self, side: str, point: np.ndarray) -> float:
        points = self._d_hand_points(side)
        return min(
            (float(np.linalg.norm(p - point)) for p in points),
            default=math.inf,
        )

    def _d_hand_object_contact(
        self, side: str, name: str, threshold: float = 0.060
    ) -> bool:
        force, available = self._d_contact_reading(side, name)
        if available:
            return force >= _D_HAND_CONTACT_FORCE
        # Interface-only unit tests have no PhysX tensor view.  Real Isaac
        # collection must never turn a missing sensor into a successful grasp.
        if self._sim is not None:
            return False
        pose_name = _D_CONTACT_POSE_ALIASES.get(name, name)
        pose = self._poses.get(pose_name, self._fallback_pose(pose_name))
        return self._d_hand_distance_to_point(side, pose[:3]) <= threshold

    def _d_hand_closed(self, side: str) -> bool:
        hand = self._right_hand if side == "right" else self._left_hand
        return bool(hand.size and float(np.max(np.abs(hand))) >= 0.08)

    def _task_dof_velocity(self, name: str) -> float | None:
        entry = self._task_dof_views.get(name)
        if entry is None:
            entry = next(
                (entry for dof_name, entry in self._task_dof_views.items() if name in dof_name),
                None,
            )
        if entry is None:
            return None
        view, index, _initial = entry
        try:
            value = view.get_dof_velocities()[0, int(index)]
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _safe_speed(velocity: np.ndarray | None, angular: bool = False) -> float:
        if velocity is None:
            return math.inf
        part = velocity[3:] if angular else velocity[:3]
        return float(np.linalg.norm(part))

    # ------------------------------------------------------------------
    # D01: bimanual nut/bolt assembly
    # ------------------------------------------------------------------

    def _d01_geometry(self) -> dict[str, float]:
        nut = self._poses.get("nut_root_body", self._fallback_pose("nut_root_body"))
        bolt = self._poses.get("bolt_root_body", self._fallback_pose("bolt_root_body"))
        bolt_rotation = _quat_wxyz_to_matrix(bolt[3:7])
        nut_rotation = _quat_wxyz_to_matrix(nut[3:7])
        local = bolt_rotation.T @ (nut[:3] - bolt[:3])
        relative_rotation = bolt_rotation.T @ nut_rotation
        phase = float(math.atan2(relative_rotation[1, 0], relative_rotation[0, 0]))
        return {
            "radial_error": float(np.linalg.norm(local[:2])),
            "depth": float(local[2]),
            "depth_error": float(abs(local[2] - 0.032)),
            "axis_alignment": float(abs(np.dot(bolt_rotation[:, 2], nut_rotation[:, 2]))),
            "phase": phase,
        }

    def _d01_contact_assignment(self) -> str:
        contacts = {
            "left_nut": self._d_hand_object_contact(
                "left", _D01_NUT_CONTACT_BODY, 0.055
            ),
            "right_nut": self._d_hand_object_contact(
                "right", _D01_NUT_CONTACT_BODY, 0.055
            ),
            "left_bolt": self._d_hand_object_contact(
                "left", _D01_BOLT_CONTACT_BODY, 0.060
            ),
            "right_bolt": self._d_hand_object_contact(
                "right", _D01_BOLT_CONTACT_BODY, 0.060
            ),
        }
        if contacts["left_bolt"] and contacts["right_nut"]:
            return "left_bolt_right_nut"
        if contacts["right_bolt"] and contacts["left_nut"]:
            return "right_bolt_left_nut"
        return "unassigned"

    def _update_d01_thread_history(self) -> None:
        geometry = self._d01_geometry()
        engaged = bool(
            geometry["radial_error"] <= 0.0065
            and 0.010 <= geometry["depth"] <= 0.064
            and geometry["axis_alignment"] >= math.cos(math.radians(18.0))
        )
        if not engaged:
            self._d01_prev_phase = None
            self._d01_disengaged_substeps += 1
            if self._d01_disengaged_substeps >= _D01_DISENGAGE_GRACE_SUBSTEPS:
                self._d01_thread_engaged = False
                self._d01_thread_entry_depth = None
                self._d01_min_thread_depth = None
                self._d01_current_axial_travel = 0.0
                self._d01_helical_error = math.inf
                self._d01_signed_twist_deg = 0.0
                self._d01_physical_twist_deg = 0.0
                self._d01_bimanual_turn_seen = False
                self._d01_role_assignment = "unassigned"
            return
        self._d01_disengaged_substeps = 0
        if not self._d01_thread_engaged:
            self._d01_thread_engaged = True
            self._d01_thread_entry_depth = geometry["depth"]
            self._d01_min_thread_depth = geometry["depth"]
        else:
            self._d01_min_thread_depth = min(
                float(self._d01_min_thread_depth), geometry["depth"]
            )
        phase = geometry["phase"]
        if self._d01_prev_phase is not None:
            increment_deg = math.degrees(_wrapped_angle(phase, self._d01_prev_phase))
            if abs(increment_deg) <= 30.0:
                self._d01_physical_twist_deg += increment_deg
                assignment = self._d01_contact_assignment()
                if assignment != "unassigned":
                    nut_side = "right" if assignment == "left_bolt_right_nut" else "left"
                    bolt_side = "left" if nut_side == "right" else "right"
                    if self._d_hand_closed(nut_side) and self._d_hand_closed(bolt_side):
                        self._d01_role_assignment = assignment
                        self._d01_signed_twist_deg += increment_deg
                        if max(0.0, -self._d01_signed_twist_deg) >= 5.0:
                            self._d01_bimanual_turn_seen = True
        self._d01_prev_phase = phase

    def _apply_d01_thread_constraint(self) -> None:
        """Apply a bounded dynamic helical guide after genuine engagement.

        This does not teleport either body: PhysX receives equal-and-opposite
        forces, so a hand must still hold the bolt and turn the nut.
        """
        if (
            self._sim is None
            or not self._d01_thread_engaged
            or self._d01_thread_entry_depth is None
        ):
            return
        assignment = self._d01_contact_assignment()
        if assignment == "unassigned":
            return

        geometry = self._d01_geometry()
        if not (
            geometry["radial_error"] <= 0.010
            and 0.006 <= geometry["depth"] <= 0.066
            and geometry["axis_alignment"] >= math.cos(math.radians(20.0))
        ):
            return
        nut = self._poses.get("nut_root_body", self._fallback_pose("nut_root_body"))
        bolt = self._poses.get("bolt_root_body", self._fallback_pose("bolt_root_body"))
        bolt_rotation = _quat_wxyz_to_matrix(bolt[3:7])
        axis = bolt_rotation[:, 2]
        local = bolt_rotation.T @ (nut[:3] - bolt[:3])
        radial_world = bolt_rotation[:, :2] @ local[:2]
        twist_progress = max(0.0, -self._d01_signed_twist_deg)
        expected_travel = min(
            0.018, twist_progress / 360.0 * _D01_THREAD_PITCH
        )
        target_depth = float(self._d01_thread_entry_depth) - expected_travel
        axial_error = float(geometry["depth"] - target_depth)

        nut_velocity = self._physics_body_velocity("nut_root_body")
        bolt_velocity = self._physics_body_velocity("bolt_root_body")
        relative_velocity = np.zeros(3, dtype=np.float64)
        if nut_velocity is not None and bolt_velocity is not None:
            relative_velocity = np.asarray(nut_velocity[:3] - bolt_velocity[:3])
        force = (
            -_D01_RADIAL_STIFFNESS * radial_world
            - _D01_AXIAL_STIFFNESS * axial_error * axis
            - _D01_THREAD_DAMPING * relative_velocity
        )
        magnitude = float(np.linalg.norm(force))
        if magnitude > _D01_MAX_THREAD_FORCE:
            force *= _D01_MAX_THREAD_FORCE / magnitude

        try:
            for name, signed_force in (
                ("nut_root_body", force),
                ("bolt_root_body", -force),
            ):
                view = self._task_body_views.get(name)
                if view is None:
                    return
                forces = self._torch.zeros(
                    (int(view.count), 3),
                    dtype=self._torch.float32,
                    device=self._isaac_device,
                )
                forces[0] = self._torch.as_tensor(
                    signed_force,
                    dtype=forces.dtype,
                    device=forces.device,
                )
                indices = self._torch.arange(
                    int(view.count),
                    dtype=self._torch.long,
                    device=self._isaac_device,
                )
                view.apply_forces_and_torques_at_position(
                    forces, None, None, indices, True
                )
        except Exception as exc:
            if not self._d01_thread_force_warning_emitted:
                self._d01_thread_force_warning_emitted = True
                self._add_task_state_warning(
                    f"Could not apply D01 dynamic thread guide: {exc}"
                )

    def _d01_metrics(self) -> dict[str, Any]:
        geometry = self._d01_geometry()
        nut_velocity = self._physics_body_velocity("nut_root_body")
        bolt_velocity = self._physics_body_velocity("bolt_root_body")
        max_linear = max(
            self._safe_speed(nut_velocity), self._safe_speed(bolt_velocity)
        )
        max_angular = max(
            self._safe_speed(nut_velocity, True),
            self._safe_speed(bolt_velocity, True),
        )
        axial_travel = 0.0
        if self._d01_thread_entry_depth is not None and self._d01_min_thread_depth is not None:
            axial_travel = max(
                0.0,
                float(self._d01_thread_entry_depth) - float(self._d01_min_thread_depth),
            )
        twist_progress = max(0.0, -self._d01_signed_twist_deg)
        current_axial_travel = (
            max(0.0, float(self._d01_thread_entry_depth) - geometry["depth"])
            if self._d01_thread_entry_depth is not None
            else 0.0
        )
        expected_axial_travel = twist_progress / 360.0 * _D01_THREAD_PITCH
        helical_error = abs(current_axial_travel - expected_axial_travel)
        self._d01_current_axial_travel = current_axial_travel
        self._d01_helical_error = helical_error
        assignment = self._d01_contact_assignment()
        contact_bodies = {
            "nut": _D01_NUT_CONTACT_BODY,
            "bolt": _D01_BOLT_CONTACT_BODY,
        }
        contact_forces = {
            f"{side}_{name}": self._d_contact_reading(side, contact_body)[0]
            for side in ("left", "right")
            for name, contact_body in contact_bodies.items()
        }
        contact_available = all(
            self._d_contact_reading(side, contact_body)[1]
            for side in ("left", "right")
            for contact_body in contact_bodies.values()
        )
        metrics = {
            **geometry,
            "thread_engaged": bool(self._d01_thread_engaged),
            "twist_progress_deg": float(twist_progress),
            "physical_twist_deg": float(self._d01_physical_twist_deg),
            "axial_travel": float(axial_travel),
            "current_axial_travel": float(current_axial_travel),
            "expected_axial_travel": float(expected_axial_travel),
            "helical_error": float(helical_error),
            "thread_pitch": float(_D01_THREAD_PITCH),
            "role_assignment": self._d01_role_assignment,
            "coordinated_contact_active": assignment != "unassigned",
            "hand_contact_available": bool(contact_available),
            "hand_contact_forces": contact_forces,
            "bimanual_turn_valid": bool(self._d01_bimanual_turn_seen),
            "assembly_stable": bool(max_linear <= 0.05 and max_angular <= 0.80),
            "max_linear_speed": float(max_linear),
            "max_angular_speed": float(max_angular),
        }
        metrics["success_now"] = bool(
            metrics["radial_error"] <= 0.0065
            and metrics["depth_error"] <= 0.014
            and 0.008 <= metrics["depth"] <= 0.055
            and metrics["axis_alignment"] >= math.cos(math.radians(18.0))
            and metrics["thread_engaged"]
            and metrics["twist_progress_deg"] >= _D01_REQUIRED_TWIST_DEG
            and metrics["current_axial_travel"] >= _D01_REQUIRED_AXIAL_TRAVEL
            and metrics["helical_error"] <= _D01_MAX_HELICAL_ERROR
            and metrics["bimanual_turn_valid"]
            and metrics["assembly_stable"]
        )
        return metrics

    # ------------------------------------------------------------------
    # D02: ordered microwave workflow
    # ------------------------------------------------------------------

    def _d02_local_position(self, name: str) -> np.ndarray:
        fixture = self._poses.get("microwave_object", self._fallback_pose("microwave_object"))
        body = self._poses.get(name, self._fallback_pose(name))
        local = _invert_pose_matrix(_pose_vector_to_matrix(fixture)) @ np.asarray(
            [*body[:3], 1.0], dtype=np.float64
        )
        return local[:3]

    def _d02_inside(self, name: str, *, plate: bool) -> bool:
        local = self._d02_local_position(name)
        margin = 0.055 if plate else 0.012
        return bool(
            -0.258 + margin <= local[0] <= 0.156 - margin
            and -0.162 + margin <= local[1] <= 0.222 - margin
            and -0.138 <= local[2] <= 0.178
        )

    def _d02_workflow_metrics(self) -> dict[str, Any]:
        door_reading = self._physics_dof_position("microjoint")
        door_angle = float(door_reading[0]) if door_reading is not None else 0.0
        button_reading = self._physics_dof_position("d02_start_button_joint")
        button_travel = (
            max(0.0, float(button_reading[0] - button_reading[1]))
            if button_reading is not None else 0.0
        )
        plate = self._poses.get("d02_plate", self._fallback_pose("d02_plate"))
        food = self._poses.get(self._d02_active_food, self._fallback_pose(self._d02_active_food))
        plate_matrix = _pose_vector_to_matrix(plate)
        food_local = (_invert_pose_matrix(plate_matrix) @ np.asarray([*food[:3], 1.0]))[:3]
        food_on_plate = bool(np.linalg.norm(food_local[:2]) <= 0.105 and 0.025 <= food_local[2] <= 0.19)
        fixture = self._poses.get("microwave_object", self._fallback_pose("microwave_object"))
        fixture_matrix = _pose_vector_to_matrix(fixture)
        button_local = np.asarray([0.267, -0.2258 + button_travel, -0.072, 1.0])
        button_world = (fixture_matrix @ button_local)[:3]
        button_hand_near = min(
            self._d_hand_distance_to_point("left", button_world),
            self._d_hand_distance_to_point("right", button_world),
        ) <= 0.075
        button_contact_force, button_contact_available = (
            self._d_any_hand_contact_reading("d02_start_button_body")
        )
        if button_contact_available:
            button_hand_contact = button_contact_force >= _D_HAND_CONTACT_FORCE
            button_contact_source = "physx_contact"
        elif self._sim is None:
            button_hand_contact = button_hand_near
            button_contact_source = "interface_distance_fallback"
        else:
            button_hand_contact = False
            button_contact_source = "contact_unavailable"
        plate_inside = self._d02_inside("d02_plate", plate=True)
        food_inside = self._d02_inside(self._d02_active_food, plate=False)
        plate_velocity = self._physics_body_velocity("d02_plate")
        food_velocity = self._physics_body_velocity(self._d02_active_food)
        payload_stable = bool(
            self._safe_speed(plate_velocity) <= 0.045
            and self._safe_speed(plate_velocity, True) <= 0.80
            and self._safe_speed(food_velocity) <= 0.045
            and self._safe_speed(food_velocity, True) <= 0.80
        )
        door_open = door_angle <= _D02_DOOR_OPEN_THRESHOLD
        door_open_for_loading = door_angle <= _D02_DOOR_LOAD_THRESHOLD
        door_closed = abs(door_angle) <= 0.08
        return {
            "door_angle": door_angle,
            "door_open": door_open,
            "door_open_for_loading": door_open_for_loading,
            "door_closed": door_closed,
            "plate_inside": plate_inside,
            "food_inside": food_inside,
            "food_on_plate": food_on_plate,
            "payload_loaded": plate_inside and food_inside,
            "button_ready": bool(
                self._d02_payload_loaded
                and door_closed
                and plate_inside
                and food_inside
                and food_on_plate
            ),
            "payload_stable": payload_stable,
            "button_travel": button_travel,
            "button_hand_near": button_hand_near,
            "button_hand_contact": bool(button_hand_contact),
            "button_contact_force": float(button_contact_force),
            "button_contact_available": bool(button_contact_available),
            "button_contact_source": button_contact_source,
            "button_pressed_by_hand": bool(
                button_travel >= 0.0045 and button_hand_contact
            ),
        }

    def _update_d02_workflow(self) -> bool:
        metrics = self._d02_workflow_metrics()
        self._d02_open_count = self._d02_open_count + 1 if metrics["door_open"] else 0
        if self._d02_open_count >= 3:
            self._d02_door_opened = True
        self._d02_close_count = self._d02_close_count + 1 if metrics["door_closed"] else 0
        self._d02_door_closed = self._d02_close_count >= 3
        if self._d02_stage == 0:
            valid = metrics["payload_loaded"] and metrics["food_on_plate"]
            self._d02_load_count = self._d02_load_count + 1 if valid else 0
            if self._d02_load_count >= 4:
                self._d02_payload_loaded = True
                self._d02_stage = 1
        elif self._d02_stage == 1:
            if self._d02_door_closed and metrics["button_ready"]:
                self._d02_button_armed = (
                    metrics["button_ready"] and metrics["button_travel"] <= 0.0015
                )
                self._d02_stage = 2
        elif self._d02_stage == 2:
            if not metrics["button_ready"]:
                self._d02_button_armed = False
            elif metrics["button_travel"] <= 0.0015:
                self._d02_button_armed = True
            if (
                self._d02_button_armed
                and metrics["button_ready"]
                and metrics["button_pressed_by_hand"]
            ):
                self._d02_stage = 3
        metrics.update(
            {
                "workflow_stage": self._d02_stage,
                "door_opened": self._d02_door_opened,
                "payload_loaded_once": self._d02_payload_loaded,
                "door_reclosed": self._d02_door_closed,
                "button_armed": self._d02_button_armed,
            }
        )
        self._d_last_metrics = metrics
        return bool(
            self._d02_stage >= 3
            and metrics["door_closed"]
            and metrics["plate_inside"]
            and metrics["food_inside"]
            and metrics["food_on_plate"]
        )

    def postprocess_recording_frame(
        self,
        frame: np.ndarray,
        *,
        canonical_key: str,
        source_key: str | None = None,
    ) -> np.ndarray:
        """Hide D02's green task button from saved videos only."""
        del canonical_key, source_key
        if self.task_name != _D02:
            return frame
        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            return frame
        channels = rgb[..., :3].astype(np.int16, copy=False)
        red = channels[..., 0]
        green = channels[..., 1]
        blue = channels[..., 2]
        green_button = (
            (green >= 110)
            & (green >= red + 45)
            & (green >= blue + 35)
            & (red <= 95)
            & (blue <= 120)
        )
        if not np.any(green_button):
            return frame
        output = np.asarray(frame).copy()
        rgb_output = output[..., :3]
        rgb_output[green_button] = np.asarray([58, 62, 58], dtype=output.dtype)
        return output

    # ------------------------------------------------------------------
    # D03: ordered left-to-right handoff
    # ------------------------------------------------------------------

    def _d03_metrics(
        self, *, update: bool, placement_counter: int | None = None
    ) -> dict[str, Any]:
        del placement_counter  # XY1 keeps its placement hold in its own state.
        target = self._poses.get("d03_target_root", self._fallback_pose("d03_target_root"))
        box_name = "Clear_Box" if self.task_name == xy1.TASK_NAME else "d03_box"
        box = self._poses.get(box_name, self._fallback_pose(box_name))
        local = (_invert_pose_matrix(_pose_vector_to_matrix(box)) @ np.asarray([*target[:3], 1.0]))[:3]
        left = self._d_hand_object_contact("left", _D03_CONTACT_BODY, 0.065)
        right = self._d_hand_object_contact("right", _D03_CONTACT_BODY, 0.065)
        lifted = bool(target[2] - self._current_table_top_z() >= 0.065)
        transfer = bool(-0.45 <= target[0] <= 0.12 and abs(target[1]) <= 0.18 and lifted)
        # Derive the acceptance window from the actual box wall half-extents
        # and the analytic target half-size.  The old hard-coded 120/188 mm
        # window rejected physically valid edge-adjacent placements by 2-3 mm.
        if self.task_name == xy1.TASK_NAME:
            inside = bool(self._xy1_block_inside(target, box))
        else:
            inside = bool(
                np.all(np.abs(local[:2]) <= _D03_BOX_INNER_HALF_EXTENTS)
                and -0.058 <= local[2] <= 0.055
            )
        velocity = self._physics_body_velocity("d03_target_root")
        linear = self._safe_speed(velocity)
        angular = self._safe_speed(velocity, True)
        hand_clearance = min(
            self._d_hand_distance_to_point("left", target[:3]),
            self._d_hand_distance_to_point("right", target[:3]),
        )
        # PhysX contact views can retain the last contact impulse for a few
        # frames after the fingers have physically separated.  Once both hands
        # are geometrically clear, treat that stale sensor bit as released;
        # this is still a live pose/contact predicate, not a state write.
        geometric_release = hand_clearance >= _D03_RELEASE_DISTANCE
        released = bool(not (left or right) or geometric_release)
        stable = bool(inside and released and linear <= 0.08 and angular <= 1.8)

        if update:
            if self._d03_stage == 0:
                if right and not left:
                    self._d03_direct_right_seen = True
                valid = left and not right and lifted
                self._d03_left_streak = self._d03_left_streak + 1 if valid else 0
                if self._d03_left_streak >= 2:
                    self._d03_left_confirmed = True
                    self._d03_stage = 1
            elif self._d03_stage == 1:
                valid = left and right and transfer
                self._d03_both_streak = self._d03_both_streak + 1 if valid else 0
                if self._d03_both_streak >= 2:
                    self._d03_both_confirmed = True
                    self._d03_stage = 2
            elif self._d03_stage == 2:
                valid = right and not left and transfer
                self._d03_right_streak = self._d03_right_streak + 1 if valid else 0
                if self._d03_right_streak >= 2:
                    self._d03_right_confirmed = True
                    self._d03_stage = 3
            if self._d03_stage in (1, 2) and not lifted and not (left or right):
                self._d03_stage = 0
                self._d03_left_confirmed = False
                self._d03_both_confirmed = False
                self._d03_right_confirmed = False
            if self._d03_stage == 3 and stable and self._success_counter >= 7:
                self._d03_stage = 4

        metrics = {
            "handoff_stage": self._d03_stage,
            "left_pickup_confirmed": self._d03_left_confirmed,
            "both_hands_confirmed": self._d03_both_confirmed,
            "right_only_confirmed": self._d03_right_confirmed,
            "direct_right_contact_seen": self._d03_direct_right_seen,
            "left_target_contact": left,
            "right_target_contact": right,
            "target_lift_height": float(target[2] - self._current_table_top_z()),
            "target_lifted": lifted,
            "target_in_transfer_zone": transfer,
            "target_inside_box": inside,
            "target_released": released,
            "target_hand_clearance": hand_clearance,
            "target_linear_speed": linear,
            "target_angular_speed": angular,
            "placement_stable": stable,
        }
        self._d_last_metrics = metrics
        return metrics

    def _success_steps_required(self) -> int:
        return super()._success_steps_required()

    # ------------------------------------------------------------------
    # D05: three airborne scissors cycles
    # ------------------------------------------------------------------

    def _d05_hinge(self) -> tuple[float, float]:
        reading = self._physics_dof_position("d05_scissors_hinge")
        angle = float(reading[0]) if reading is not None else self._d05_previous_hinge
        velocity = self._task_dof_velocity("d05_scissors_hinge")
        if velocity is None:
            velocity = (angle - self._d05_previous_hinge) / max(self._physics_dt, 1e-6)
        return angle, float(velocity)

    def _d05_handle_metrics(self) -> dict[str, Any]:
        root = self._poses.get("d05_scissors_root_body", self._fallback_pose("d05_scissors_root_body"))
        fixed = self._poses.get("d05_scissors_fixed", root)
        moving = self._poses.get("d05_scissors_moving", root)
        fixed_point = (_pose_vector_to_matrix(fixed) @ np.asarray([-0.020, -0.054, 0.0, 1.0]))[:3]
        moving_point = (_pose_vector_to_matrix(moving) @ np.asarray([0.021, -0.054, 0.0, 1.0]))[:3]
        all_points = (*self._d_hand_points("left"), *self._d_hand_points("right"))
        fixed_distance = min((float(np.linalg.norm(p - fixed_point)) for p in all_points), default=math.inf)
        moving_distance = min((float(np.linalg.norm(p - moving_point)) for p in all_points), default=math.inf)
        fixed_force, fixed_available = self._d_any_hand_contact_reading(
            "d05_scissors_fixed"
        )
        moving_force, moving_available = self._d_any_hand_contact_reading(
            "d05_scissors_moving"
        )
        contact_available = fixed_available and moving_available
        if contact_available:
            fixed_contact = fixed_force >= _D_HAND_CONTACT_FORCE
            moving_contact = moving_force >= _D_HAND_CONTACT_FORCE
            contact_source = "physx_contact"
        elif self._sim is None:
            fixed_contact = fixed_distance <= 0.055
            moving_contact = moving_distance <= 0.055
            contact_source = "interface_distance_fallback"
        else:
            fixed_contact = False
            moving_contact = False
            contact_source = "contact_unavailable"
        clearance = float(root[2] - self._current_table_top_z())
        return {
            "air_clearance": clearance,
            "fully_lifted": clearance >= 0.035,
            "fixed_handle_contact": bool(fixed_contact),
            "moving_handle_contact": bool(moving_contact),
            "fixed_handle_distance": fixed_distance,
            "moving_handle_distance": moving_distance,
            "fixed_handle_contact_force": float(fixed_force),
            "moving_handle_contact_force": float(moving_force),
            "handle_contact_available": bool(contact_available),
            "handle_contact_source": contact_source,
        }

    def _update_d05_cut_state(self) -> None:
        if self._d05_phase == 3:
            return
        angle, _velocity = self._d05_hinge()
        handle = self._d05_handle_metrics()
        active = bool(
            handle["fully_lifted"]
            and handle["fixed_handle_contact"]
            and handle["moving_handle_contact"]
        )
        if not active:
            self._d05_inactive_substeps += 1
            self._d05_threshold_streak = 0
            if self._d05_inactive_substeps >= 15:
                self._d05_phase = 0
            self._d05_previous_hinge = angle
            return
        self._d05_inactive_substeps = 0
        target_reached = angle >= 0.52 if self._d05_phase in (0, 1) else angle <= 0.20
        self._d05_threshold_streak = self._d05_threshold_streak + 1 if target_reached else 0
        if self._d05_threshold_streak >= 3:
            self._d05_threshold_streak = 0
            if self._d05_phase in (0, 1):
                self._d05_phase = 2
            else:
                self._d05_cut_count += 1
                self._d05_phase = 3 if self._d05_cut_count >= 3 else 1
        self._d05_previous_hinge = angle

    def _d05_metrics(self) -> dict[str, Any]:
        angle, velocity = self._d05_hinge()
        handle = self._d05_handle_metrics()
        metrics = {
            **handle,
            "hinge_angle": angle,
            "hinge_velocity": velocity,
            "effective_handle_contact": bool(
                handle["fixed_handle_contact"] and handle["moving_handle_contact"]
            ),
            "cut_count": self._d05_cut_count,
            "required_cuts": 3,
            "task_phase": _D05_PHASE_NAMES[self._d05_phase],
            "success_now": bool(
                self._d05_cut_count >= 3
                and handle["fully_lifted"]
                and handle["fixed_handle_contact"]
                and handle["moving_handle_contact"]
            ),
        }
        self._d_last_metrics = metrics
        return metrics

    # ------------------------------------------------------------------
    # Success, live right-side condition panel, observations, and replay
    # ------------------------------------------------------------------

    def _compute_success_now(self) -> bool:
        if self.task_name == xy1.TASK_NAME:
            return self._xy1_compute_success()
        if self.task_name == _D01:
            self._d_last_metrics = self._d01_metrics()
            return bool(self._d_last_metrics["success_now"])
        if self.task_name == _D02:
            return self._update_d02_workflow()
        if self.task_name in _D03_TASKS:
            return bool(self._d03_metrics(update=True)["placement_stable"] and self._d03_stage >= 3)
        self._d_last_metrics = self._d05_metrics()
        return bool(self._d_last_metrics["success_now"])

    @staticmethod
    def _success_condition_display_label(row: dict[str, Any]) -> str:
        key = str(row.get("key") or "")
        if key in _D_SUCCESS_CONDITION_DISPLAY_LABELS:
            return _D_SUCCESS_CONDITION_DISPLAY_LABELS[key]
        return key.replace("_", " ") or "Condition"

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        if self.task_name == xy1.TASK_NAME:
            return self._xy1_condition_rows()
        if self.task_name == _D01:
            m = self._d_last_metrics or self._d01_metrics()
            rows = [
                self._condition_row("d01_roles", "Each hand stably holds one fastener", m["coordinated_contact_active"]),
                self._condition_row("d01_align", "Nut center, depth, and axis aligned", m["radial_error"] <= 0.0065 and m["depth_error"] <= 0.014 and m["axis_alignment"] >= math.cos(math.radians(18.0))),
                self._condition_row("d01_twist", "Tightened by at least 90 deg", m["twist_progress_deg"] >= _D01_REQUIRED_TWIST_DEG),
                self._condition_row("d01_travel", "Nut travel matches thread pitch", m["current_axial_travel"] >= _D01_REQUIRED_AXIAL_TRAVEL and m["helical_error"] <= _D01_MAX_HELICAL_ERROR),
                self._condition_row("d01_stable", "Assembly remains stable", m["assembly_stable"]),
            ]
        elif self.task_name == _D02:
            m = self._d_last_metrics or self._d02_workflow_metrics()
            rows = [
                self._condition_row(
                    "d02_open",
                    "Microwave door closed",
                    m["door_closed"],
                ),
                self._condition_row("d02_inside", "Plate and food inside microwave cavity", m["plate_inside"] and m["food_inside"]),
                self._condition_row("d02_on_plate", "Food remains on plate", m["food_on_plate"]),
                self._condition_row("d02_button", "Green start button pressed while loaded", self._d02_stage >= 3),
            ]
        elif self.task_name in _D03_TASKS:
            m = self._d_last_metrics or self._d03_metrics(update=False)
            rows = [
                self._condition_row("d03_left", "Left hand lifted target block first", self._d03_left_confirmed),
                self._condition_row("d03_both", "Both hands contacted block in transfer zone", self._d03_both_confirmed),
                self._condition_row("d03_right", "Right hand received after left release", self._d03_right_confirmed),
                self._condition_row("d03_box", "Block released stably inside right box", m["placement_stable"]),
            ]
        else:
            m = self._d_last_metrics or self._d05_metrics()
            rows = [
                self._condition_row("d05_air", "Scissors lifted at least 35 mm", m["fully_lifted"]),
                self._condition_row("d05_handles", "Both handles actively held", m["effective_handle_contact"]),
                self._condition_row("d05_cuts", f"Three full open-close cycles ({self._d05_cut_count}/3)", self._d05_cut_count >= 3),
            ]
        return self._append_success_hold_row(rows)

    def _compute_observation(self) -> dict[str, Any]:
        tcp, gripper = self._d_robot_state()
        xy1_hidden_state: np.ndarray | None = None
        state: dict[str, np.ndarray] = {
            "tcp_pose": tcp,
            "gripper_pose": gripper,
            "table_delta_height": self.delta_h.astype(np.float64, copy=True),
        }
        if self.task_name == _D01:
            metrics = self._d_last_metrics or self._d01_metrics()
            nut = self._poses.get("nut_root_body", self._fallback_pose("nut_root_body"))
            bolt = self._poses.get("bolt_root_body", self._fallback_pose("bolt_root_body"))
            state.update(
                {
                    "nut_pose": nut.copy(),
                    "bolt_pose": bolt.copy(),
                    "nut_to_bolt": np.asarray(
                        [*(nut[:3] - bolt[:3]), metrics["radial_error"], metrics["depth"], metrics["axis_alignment"]],
                        dtype=np.float64,
                    ),
                    "assembly_metrics": np.asarray(
                        [metrics["radial_error"], metrics["depth_error"], metrics["axis_alignment"], metrics["twist_progress_deg"], metrics["max_linear_speed"], self._success_counter],
                        dtype=np.float64,
                    ),
                }
            )
        elif self.task_name == _D02:
            metrics = self._d_last_metrics or self._d02_workflow_metrics()
            door_reading = self._physics_dof_position("microjoint")
            button_reading = self._physics_dof_position("d02_start_button_joint")
            state.update(
                {
                    "plate_pose": self._poses.get("d02_plate", self._fallback_pose("d02_plate")).copy(),
                    "food_pose": self._poses.get(self._d02_active_food, self._fallback_pose(self._d02_active_food)).copy(),
                    "microwave_pose": self._poses.get("microwave_object", self._fallback_pose("microwave_object")).copy(),
                    "microwave_joint_state": np.asarray(
                        [
                            door_reading[0] if door_reading is not None else 0.0,
                            self._task_dof_velocity("microjoint") or 0.0,
                            button_reading[0] if button_reading is not None else 0.0,
                            self._task_dof_velocity("d02_start_button_joint") or 0.0,
                        ],
                        dtype=np.float64,
                    ),
                    "workflow_progress": np.asarray(
                        [self._d02_stage / 3.0, self._d02_door_closed, self._d02_payload_loaded, self._d02_stage >= 3, self._d02_button_armed, metrics["door_angle"], metrics["button_travel"], metrics["plate_inside"], metrics["food_on_plate"], metrics["food_inside"]],
                        dtype=np.float64,
                    ),
                    "food_variant": np.asarray(
                        [self._d02_active_food == "003_cracker_box", self._d02_active_food == "008_pudding_box"],
                        dtype=np.float64,
                    ),
                }
            )
        elif self.task_name in {_D03, xy1.TASK_NAME}:
            metrics = self._d_last_metrics or self._d03_metrics(update=False)
            target = self._poses.get("d03_target_root", self._fallback_pose("d03_target_root"))
            box_name = "Clear_Box" if self.task_name == xy1.TASK_NAME else "d03_box"
            box = self._poses.get(box_name, self._fallback_pose(box_name))
            state.update(
                {
                    "target_pose": target.copy(),
                    "box_pose": box.copy(),
                    "target_to_box": (target[:3] - box[:3]).astype(np.float64),
                    "handoff_metrics": np.asarray(
                        [self._d03_stage, metrics["left_target_contact"], metrics["right_target_contact"], metrics["target_lift_height"], metrics["target_in_transfer_zone"], self._d03_both_confirmed, self._d03_right_confirmed, metrics["target_inside_box"], metrics["target_linear_speed"], self._success_counter],
                        dtype=np.float64,
                    ),
                    "distractor_poses": np.concatenate(
                        [
                            self._poses.get(name, self._fallback_pose(name)).copy()
                            for name in (_XY1_DISTRACTORS if self.task_name == xy1.TASK_NAME else _D03_DISTRACTORS)
                        ]
                    ),
                }
            )
            if self.task_name == xy1.TASK_NAME:
                state["lid_pose"] = self._poses.get("Box_lid", self._fallback_pose("Box_lid")).copy()
                xy1_hidden_state = self._xy1_hidden_state()
        else:
            metrics = self._d_last_metrics or self._d05_metrics()
            phase = np.zeros(4, dtype=np.float64)
            phase[self._d05_phase] = 1.0
            state.update(
                {
                    "scissors_pose": self._poses.get("d05_scissors_root_body", self._fallback_pose("d05_scissors_root_body")).copy(),
                    "scissors_hinge": np.asarray([metrics["hinge_angle"], metrics["hinge_velocity"], metrics["air_clearance"], metrics["fully_lifted"]], dtype=np.float64),
                    "cut_progress": np.concatenate(
                        [np.asarray([self._d05_cut_count / 3.0, metrics["effective_handle_contact"], self._d05_cut_count >= 3, metrics["fully_lifted"]], dtype=np.float64), phase]
                    ),
                }
            )
        observation: dict[str, Any] = {"state": state}
        if xy1_hidden_state is not None:
            observation["xy1_hidden_state"] = xy1_hidden_state
        if self.image_obs:
            front, wrist_left, wrist_right = self.render()
            observation["images"] = {
                "random_camera" if self.randomize else "front": front,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }
        return observation

    def _task_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "task_id": self.task_name[:3].upper(),
            "task_name": self.task_name,
            "success_counter": int(self._success_counter),
            "success_steps_required": int(self._success_steps_required()),
        }
        info.update(self._d_last_metrics)
        if self.task_name == _D02:
            info["active_food"] = self._d02_active_food
        if self.task_name == _D03:
            info["handoff_stage"] = self._d03_stage
        if self.task_name == xy1.TASK_NAME:
            info["handoff_stage"] = self._d03_stage
            info["xy1_a_complete"] = bool(self._xy1_a_complete)
            info["xy1_lid_ready"] = bool(self._xy1_lid_ready)
            info["xy1_lid_was_moved"] = bool(self._lid_was_moved)
        return info

    def restore_recorded_initial_state(
        self, state_parts: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        delta = np.asarray(
            state_parts.get("table_delta_height", self.delta_h), dtype=np.float64
        ).ravel()
        if delta.size:
            self.delta_h[:] = float(delta[0])
            self._restore_recorded_table_height(float(delta[0]))

        if self.task_name == _D02:
            variant = np.asarray(state_parts.get("food_variant", np.zeros(0))).ravel()
            if variant.size >= 2:
                self._d02_active_food = (
                    "003_cracker_box" if int(np.argmax(variant[:2])) == 0 else "008_pudding_box"
                )
        key_to_name = {
            "nut_pose": "nut_root_body",
            "bolt_pose": "bolt_root_body",
            "plate_pose": "d02_plate",
            "food_pose": self._d02_active_food,
            "target_pose": "d03_target_root",
            "box_pose": "Clear_Box" if self.task_name == xy1.TASK_NAME else "d03_box",
            "scissors_pose": "d05_scissors_root_body",
        }
        restored: set[int] = set()

        def restore_pose(name: str, pose_like: np.ndarray) -> bool:
            pose = np.asarray(pose_like, dtype=np.float64).ravel()
            view = self._task_body_views.get(name)
            if pose.size < 7 or view is None or id(view) in restored:
                return False
            if self._set_task_body_world_pose(view, pose[:7]):
                restored.add(id(view))
                return True
            return False

        for key, name in key_to_name.items():
            restore_pose(name, state_parts.get(key, np.zeros(0)))

        if self.task_name == _D02:
            inactive_food = (
                "008_pudding_box"
                if self._d02_active_food == "003_cracker_box"
                else "003_cracker_box"
            )
            restore_pose(
                inactive_food,
                np.asarray(
                    [
                        *_D02_PARKED_XY,
                        -_D02_FOOD_SUPPORT_LOCAL_MIN_Z[inactive_food],
                        _D02_FOOD_REST_QUAT_XYZW[3],
                        _D02_FOOD_REST_QUAT_XYZW[0],
                        0.0,
                        0.0,
                    ]
                ),
            )
            # ``microwave_joint_state`` remains recorded and compared during
            # physical replay, but is not authored: every D02 replay process
            # constructs the same closed-door/released-button USD state.

        if self.task_name in {_D03, xy1.TASK_NAME}:
            distractor_poses = np.asarray(
                state_parts.get("distractor_poses", np.zeros((0, 7))),
                dtype=np.float64,
            ).reshape(-1, 7)
            names = _XY1_DISTRACTORS if self.task_name == xy1.TASK_NAME else _D03_DISTRACTORS
            for name, pose in zip(names, distractor_poses):
                restore_pose(name, pose)

        if self.task_name == xy1.TASK_NAME:
            restore_pose("Box_lid", state_parts.get("lid_pose", np.zeros(0)))
            hidden = state_parts.get("xy1_hidden_state")
            if hidden is not None:
                self._xy1_restore_hidden_state(hidden)


        if self.task_name == _D05:
            hinge_state = np.asarray(
                state_parts.get("scissors_hinge", np.zeros(0)), dtype=np.float64
            ).ravel()
            if hinge_state.size >= 2:
                self._set_task_dof_state(
                    "d05_scissors_hinge", hinge_state[0], hinge_state[1]
                )

        if self._sim is not None:
            self._sim.forward()
        self._refresh_object_poses()
        return self._compute_observation()
