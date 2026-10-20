"""Native Isaac Sim 5.1 runtime for YCB06 semantic table setting."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from .abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _TABLE_TOP_Z,
    _matrix_to_quat_wxyz,
    _pose_vector_to_matrix,
    _quat_wxyz_to_matrix,
)


YCB06_TASK_NAME = "t28_place_tableware"
YCB06_TARGET_NAMES = ("fork", "knife", "spoon", "mug")
YCB06_DYNAMIC_NAMES = ("plate", *YCB06_TARGET_NAMES)
YCB06_FIXTURE_NAME = "gelatin_box"

YCB06_PROMPTS = {
    "fork": (
        "Pick up the fork and place it to the left of the plate, with the "
        "handle pointing toward you."
    ),
    "knife": (
        "Pick up the knife and place it to the right of the plate, with the "
        "handle pointing toward you and the blade facing the plate."
    ),
    "spoon": (
        "Pick up the spoon and place it on the outer right side of the plate, "
        "with the handle pointing toward you."
    ),
    "mug": (
        "Pick up the mug and place it above and to the right of the plate, "
        "upright with its opening facing up."
    ),
}

# Task-frame +X is operator-right, +Y points away from the operator, and +Z
# points up. The platform's default operator camera is on world -X, looking
# toward +X: screen-right is world -Y. Keep this frame fixed when orbiting
# the viewport, so moving the camera cannot move an episode's goal.
_WORLD_FROM_TASK = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_TASK_FROM_WORLD = _WORLD_FROM_TASK.T

_SLOT_CENTERS_TASK_XY = {
    "fork": np.asarray([-0.19, 0.00], dtype=np.float64),
    "knife": np.asarray([0.19, 0.00], dtype=np.float64),
    "spoon": np.asarray([0.29, 0.00], dtype=np.float64),
    "mug": np.asarray([0.19, 0.18], dtype=np.float64),
}

# Exact original-mesh registration half extents in metres.
_BODY_HALF_EXTENTS = {
    "plate": np.asarray([0.1295725, 0.1298925, 0.0123455]),
    "fork": np.asarray([0.0100350, 0.0985005, 0.0065655]),
    "knife": np.asarray([0.0107330, 0.1080585, 0.0071460]),
    "spoon": np.asarray([0.0213615, 0.0978895, 0.0108470]),
    "mug": np.asarray([0.0397320, 0.0528710, 0.0407645]),
    "gelatin_box": np.asarray([0.0424780, 0.0140000, 0.0365910]),
}
_RESET_CENTER_Z = {
    "plate": _TABLE_TOP_Z + _BODY_HALF_EXTENTS["plate"][2],
    # Measured settled mesh centres plus 0.3 mm unrecorded landing clearance.
    "fork": 0.9134028,
    "knife": 0.9228274,
    "spoon": 0.9152770,
    "mug": _TABLE_TOP_Z + _BODY_HALF_EXTENTS["mug"][2],
}
_UTENSIL_RESET_PITCH_DEG = {
    "fork": -15.702,
    "knife": -1.515,
    "spoon": -13.097,
}
# Four differently rotated real-PhysX support traces agree on these settled
# mesh orientations. Author that equilibrium with 0.3 mm landing clearance,
# including the knife's side tilt, instead of dropping a horizontal knife
# 2.5 mm onto the riser and letting it rock into this pose during every reset.
_UTENSIL_RESET_SIDE_TILT_DEG = {"fork": 0.0, "knife": 1.665, "spoon": -0.165}

_UTENSIL_SLOT_HALF_EXTENTS = {
    name: np.asarray(
        [
            _BODY_HALF_EXTENTS[name][0] + 0.025,
            _BODY_HALF_EXTENTS[name][1] + 0.040,
        ],
        dtype=np.float64,
    )
    for name in ("fork", "knife", "spoon")
}

# Calibration intervals in the unrotated support layout. Randomization uses
# their centres, then transforms the assembly; it never jitters a tool against
# the riser. Plate/mug centres also define the non-randomized preview layout.
_RESET_XY_BOUNDS = {
    "plate": np.asarray([[-0.290, -0.030], [-0.250, 0.030]], dtype=np.float64),
    "fork": np.asarray([[-0.006, 0.027], [0.006, 0.031]], dtype=np.float64),
    # Bias the knife centre 4-6 mm onto the fixed support.  Near the left edge
    # the collision mesh has two valid equilibria and can tip during reset;
    # this measured interval selects the flat, graspable equilibrium.  The
    # knife remains fully dynamic and unrestricted after the episode starts.
    "knife": np.asarray([[0.004, -0.002], [0.006, 0.002]], dtype=np.float64),
    # Keep the spoon's long-axis centre on the supported half of the riser.
    # Negative-x edge samples can rock about 3.9 degrees before choosing the
    # same equilibrium, which violates the strict 3-degree recording origin.
    "spoon": np.asarray([[0.000, -0.041], [0.006, -0.037]], dtype=np.float64),
    "mug": np.asarray([[-0.160, -0.280], [-0.120, -0.220]], dtype=np.float64),
}
_RESET_XY_HALF_EXTENTS = {
    "plate": np.asarray([0.130, 0.130], dtype=np.float64),
    "fork": np.asarray([0.100, 0.011], dtype=np.float64),
    "knife": np.asarray([0.109, 0.012], dtype=np.float64),
    "spoon": np.asarray([0.099, 0.022], dtype=np.float64),
    "mug": np.asarray([0.041, 0.054], dtype=np.float64),
    # Rz(90) Rx(90) maps the gelatin top to world X/Y = 73.2/85.0 mm.
    "gelatin_box": np.asarray([0.036591, 0.042478], dtype=np.float64),
}
_RESET_Z_HALF_EXTENTS = {
    name: float(_RESET_CENTER_Z[name] - _TABLE_TOP_Z)
    for name in YCB06_DYNAMIC_NAMES
}
_FIXTURE_XY = np.asarray([0.075, 0.000], dtype=np.float64)
_FIXTURE_CENTER_Z = 0.904
_FIXTURE_QUAT_XYZW = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
_RESET_CLEARANCE = 0.003
# Wuji's closest reset-home point is a distal finger-link origin.  Its actual
# collision shell stays well inside this origin-to-object clearance; the old
# centre-to-sphere rule inflated the 260 mm plate into a 368 mm sphere and
# rejected every physically clear layout.
_RESET_HAND_ORIGIN_CLEARANCE = 0.040
_RESET_MAX_ATTEMPTS = 128
# Move the calibrated support assembly as one rigid layout. Independent
# millimetre/yaw jitter between a utensil and its riser changes its equilibrium
# and can spend the whole recording-origin budget rocking at a support edge.
# These world-space bands cover the source workspace of both hands.
_RESET_GROUP_XY_BOUNDS = np.asarray([[-0.10, -0.38], [0.07, 0.38]])
_RESET_PLATE_XY_BOUNDS = np.asarray([[-0.33, -0.16], [-0.25, 0.12]])
_RESET_MUG_XY_BOUNDS = np.asarray([[-0.30, -0.44], [0.08, 0.44]])
# The slender original collision meshes need enough contact iterations to
# converge while resting across the table and riser.  This changes numerical
# accuracy, not the recording-origin thresholds or the physical asset.
_SOLVER_POSITION_ITERATIONS = 64
_SOLVER_VELOCITY_ITERATIONS = 16

_LIFT_HEIGHT = 0.020
_WRONG_DISPLACEMENT = 0.060
_WRONG_DEBOUNCE_SECONDS = 0.20
_PLATE_SUCCESS_TRANSLATION = 0.030
_PLATE_SUCCESS_TILT_DEG = 8.0
_PLATE_FAILURE_TRANSLATION = 0.050
_PLATE_FAILURE_TILT_DEG = 15.0
_PLATE_FAILURE_DEBOUNCE_SECONDS = 0.25
_UTENSIL_LATERAL_TOLERANCE = 0.025
_UTENSIL_LONGITUDINAL_TOLERANCE = 0.040
_UTENSIL_ORIENTATION_TOLERANCE_DEG = 20.0
_UTENSIL_MIN_FOOTPRINT_OVERLAP = 0.80
_MUG_XY_TOLERANCE = 0.045
_MUG_OPENING_TOLERANCE_DEG = 15.0
_HAND_CLEARANCE = 0.025
_STABLE_LINEAR_SPEED = 0.020
_STABLE_ANGULAR_SPEED = 0.25
_STABLE_SECONDS = 0.50
_SUPPORT_HEIGHT_TOLERANCE = 0.008
_CONTACT_FORCE_THRESHOLD = 1.0e-6
_CONTACT_RECENT_STEPS = 3

_TABLE_SAFE_X = (-0.55, 0.25)
_TABLE_SAFE_Y = (-0.80, 0.80)

_TERMINAL_REASON_CODE = {
    "": 0,
    "success": 1,
    "wrong_object_manipulated": 2,
    "object_fell_off_table": 3,
    "reference_plate_displaced": 4,
    "robot_safety_violation": 5,
    "timeout": 6,
    "manual_reset": 7,
}


def _quat_wxyz_from_zx_degrees(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    """Quaternion for ``Rz(yaw) @ Rx(pitch)`` in WXYZ order."""
    yaw = math.radians(float(yaw_degrees)) * 0.5
    pitch = math.radians(float(pitch_degrees)) * 0.5
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    quat = np.asarray([cy * cp, cy * sp, sy * sp, sy * cp], dtype=np.float64)
    return quat / max(float(np.linalg.norm(quat)), 1.0e-12)


def _quat_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def _rotation_xy(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.asarray([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])


class HandsOnYCB06IsaacLabEnv(HandsOnABCIsaacLabEnv):
    """YCB06 physical semantics over the shared ABC Isaac 5.1 runtime."""

    fail_fast_recording_origin_geometry = True

    def __init__(self, *args: Any, target_object: str | None = None, **kwargs: Any):
        self._ycb06_fixed_target = (
            str(target_object).strip()
            if target_object in YCB06_TARGET_NAMES
            else None
        )
        self._ycb06_reset_ordinal = 0
        self._ycb06_reset_group_yaw = 0.0
        self._ycb06_reset_fixture_xy = _FIXTURE_XY.copy()
        self._ycb06_reset_mug_yaw = 0.0
        self._ycb06_reset_progress: dict[str, Any] | None = None
        self._ycb06_contact_views: dict[str, Any] = {}
        self._ycb06_initial_poses: dict[str, np.ndarray] = {}
        self._ycb06_plate_reset_pose: np.ndarray | None = None
        self._ycb06_target_contact_seen = False
        self._ycb06_target_contact_recent = 0
        self._ycb06_target_lifted_seen = False
        self._ycb06_placement_seen = False
        self._ycb06_release_seen = False
        self._ycb06_stable_steps = 0
        self._ycb06_wrong_counters = {
            name: 0 for name in YCB06_TARGET_NAMES
        }
        self._ycb06_plate_failure_counter = 0
        self._ycb06_first_terminal_reason = ""
        self._last_ycb06_metrics: dict[str, Any] = {}

        super().__init__(
            YCB06_TASK_NAME,
            *args,
            target_object=target_object,
            **kwargs,
        )
        self._extend_ycb06_observation_space()
        self._init_ycb06_contact_views()
        # The shared setup skips the panel in direct-hand mode. YCB06 still
        # needs its sampled instruction in every interactive control mode.
        self._init_success_condition_overlay()
        self._update_success_condition_overlay()

    def _pre_simulation_context_task_audit(self) -> None:
        super()._pre_simulation_context_task_audit()
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            # Isaac Lab registers its STOP/render callback before the core
            # context constructor stops an already-playing timeline. Drain
            # that startup STOP first, before the callback can wait forever
            # for a scene whose construction has not completed yet.
            timeline.stop()
            self._simulation_app.update()

    def _in_process_scene_rebuild_kwargs(self, seed: int) -> dict[str, Any]:
        kwargs = super()._in_process_scene_rebuild_kwargs(seed)
        # This concrete constructor supplies its own positional task name.
        kwargs.pop("task_name", None)
        # Preserve an explicit target lock; a randomly sampled episode target
        # must not become a permanent lock when the scene is rebuilt.
        kwargs["target_object"] = self._ycb06_fixed_target
        return kwargs

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        """Enable physical hand-contact truth before the first PhysX reset."""
        del sim_utils
        if self._stage is None:
            return

        from pxr import PhysxSchema, Usd, UsdPhysics

        # The MJCF importer wraps free objects in one-link articulations.
        # YCB06 has only independent rigid bodies; retaining those redundant
        # roots gives the thin supported utensils articulation solver/sleep
        # settings and persistent reset jitter.  Keep their original meshes,
        # mass properties and unrestricted rigid-body motion.
        for name in (*YCB06_DYNAMIC_NAMES, YCB06_FIXTURE_NAME):
            body_prim = self._find_rigid_body_prim(name)
            if body_prim is None:
                self._add_task_state_warning(
                    f"Could not configure YCB06 rigid body {name}"
                )
                continue
            if body_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                body_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if body_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                body_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            if name in YCB06_DYNAMIC_NAMES:
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
                rigid_api.CreateSolverPositionIterationCountAttr().Set(
                    _SOLVER_POSITION_ITERATIONS
                )
                rigid_api.CreateSolverVelocityIterationCountAttr().Set(
                    _SOLVER_VELOCITY_ITERATIONS
                )

        fixture_prim = self._find_rigid_body_prim(YCB06_FIXTURE_NAME)
        if fixture_prim is None:
            self._add_task_state_warning(
                "Could not fix the YCB06 gelatin-box fixture"
            )
        else:
            task_root = self._stage.GetPrimAtPath("/World/Task")
            fixture_path = fixture_prim.GetPath()
            for prim in Usd.PrimRange(task_root):
                if not prim.IsActive() or not prim.IsA(UsdPhysics.FixedJoint):
                    continue
                joint = UsdPhysics.Joint(prim)
                targets = set(joint.GetBody0Rel().GetTargets()) | set(
                    joint.GetBody1Rel().GetTargets()
                )
                if fixture_path in targets:
                    prim.SetActive(False)
            rigid_api = UsdPhysics.RigidBodyAPI(fixture_prim)
            if not rigid_api:
                rigid_api = UsdPhysics.RigidBodyAPI.Apply(fixture_prim)
            kinematic_attr = rigid_api.GetKinematicEnabledAttr()
            if not kinematic_attr.IsValid():
                kinematic_attr = rigid_api.CreateKinematicEnabledAttr()
            kinematic_attr.Set(True)

        for name in YCB06_TARGET_NAMES:
            body_prim = self._find_rigid_body_prim(name)
            if body_prim is None:
                self._add_task_state_warning(
                    f"Could not configure YCB06 contact reporting for {name}"
                )
                continue
            if body_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                report_api = PhysxSchema.PhysxContactReportAPI.Get(
                    self._stage,
                    body_prim.GetPath(),
                )
            else:
                report_api = PhysxSchema.PhysxContactReportAPI.Apply(body_prim)
            report_api.CreateThresholdAttr().Set(0.0)

    def _extend_ycb06_observation_space(self) -> None:
        state_spaces = self.observation_space["state"].spaces
        state_spaces["gelatin_box_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(7,), dtype=np.float64
        )
        additions = {
            "ycb06_target_selector": spaces.Box(
                0.0, 1.0, shape=(4,), dtype=np.float64
            ),
            "ycb06_plate_reset_pose": spaces.Box(
                -np.inf, np.inf, shape=(7,), dtype=np.float64
            ),
            "ycb06_slot_pose": spaces.Box(
                -np.inf, np.inf, shape=(7,), dtype=np.float64
            ),
            "ycb06_initial_object_poses": spaces.Box(
                -np.inf, np.inf, shape=(35,), dtype=np.float64
            ),
            "ycb06_fixture_pose": spaces.Box(
                -np.inf, np.inf, shape=(7,), dtype=np.float64
            ),
            "ycb06_event_state": spaces.Box(
                -np.inf, np.inf, shape=(12,), dtype=np.float64
            ),
            "ycb06_terminal_reason_code": spaces.Box(
                0.0,
                float(max(_TERMINAL_REASON_CODE.values())),
                shape=(1,),
                dtype=np.float64,
            ),
            "ycb06_reset_metadata": spaces.Box(
                -np.inf, np.inf, shape=(2,), dtype=np.float64
            ),
        }
        for key, space in additions.items():
            self.observation_space.spaces[key] = space

    def _task_extra_pose_names(self) -> set[str]:
        return super()._task_extra_pose_names() | {YCB06_FIXTURE_NAME}

    def _target_prompt(self) -> str:
        return YCB06_PROMPTS[self._target_object_name]

    def _task_status_text(self) -> str:
        if self._ycb06_reset_progress is not None:
            progress = self._ycb06_reset_progress
            budget = max(1, int(progress.get("physics_step_budget", 1)))
            percent = min(100, int(100 * progress.get("physics_steps", 0) / budget))
            return (
                "YCB06 | RESETTING - waiting for tableware to settle\n"
                f"Attempt {progress.get('attempt', 1)}/{progress.get('attempts_allowed', 3)} | "
                f"Check progress: {percent}%\n"
                "Hands are held until the reset passes. Invalid layouts retry automatically."
            )
        target = self._target_object_name
        metrics = self._last_ycb06_metrics
        slot_x, slot_y = _SLOT_CENTERS_TASK_XY[target] * 100.0
        goal = (
            f"center {abs(slot_x):.0f} cm {'LEFT' if slot_x < 0 else 'RIGHT'}"
            + (f" and {slot_y:.0f} cm FAR" if slot_y else ", same near/far position")
            + " from the plate's initial center."
        )
        orientation = (
            f"Opening UP; tilt <= {_MUG_OPENING_TOLERANCE_DEG:g} deg; "
            f"center error <= {_MUG_XY_TOLERANCE * 100:g} cm."
            if target == "mug"
            else f"Handle TOWARD YOU; angle error <= {_UTENSIL_ORIENTATION_TOLERANCE_DEG:g} deg."
            + (" Knife blade faces the plate." if target == "knife" else "")
        )
        reason = self._ycb06_first_terminal_reason
        if reason and reason != "success":
            next_action = f"Stopped: {reason.replace('_', ' ')}. Press R to reset."
        elif metrics.get("ycb06_success_now"):
            next_action = "SUCCESS: the instructed placement is complete."
        elif not metrics.get("ycb06_target_lifted_seen"):
            next_action = f"Grasp ONLY the {target.upper()} and lift it at least 2 cm."
        elif not metrics.get("ycb06_plate_success_ok"):
            next_action = "Reference plate moved too far; press R to restart."
        elif not metrics.get("ycb06_semantic_pose_ok"):
            next_action = "Adjust the target position and orientation (see rows below)."
        elif not metrics.get("ycb06_valid_table_support"):
            next_action = "Lower the target onto the TABLE, clear of other objects."
        elif not metrics.get("ycb06_target_released"):
            next_action = "Open the hand and move it at least 2.5 cm clear."
        else:
            next_action = "Leave the target and all other tableware still."
        remaining = max(0.0, self._time_limit - self.env_step * self._control_dt)
        return (
            f"YCB06 | MOVE ONLY {target.upper()} | {remaining:.0f} s left\n"
            f"Goal: {goal}\n{orientation}\n"
            "Directions use the default operator view (V restores it).\n"
            "Either hand: grasp -> lift >= 2 cm -> place ON TABLE.\n"
            "Release; BOTH hands >= 2.5 cm clear; keep stable for 0.5 s.\n"
            "Keep the plate in place; leave all other tableware alone.\n"
            f"Next: {next_action}"
        )

    def set_recording_origin_progress(
        self, progress: Mapping[str, Any] | None,
    ) -> None:
        self._ycb06_reset_progress = dict(progress) if progress is not None else None
        self._update_success_condition_overlay()

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        """Show the live task predicates without advancing their history."""
        metrics = self._last_ycb06_metrics
        target = self._target_object_name

        def value(key: str, scale: float = 1.0) -> str:
            number = float(metrics.get(key, math.nan)) * scale
            return f"{number:.1f}" if math.isfinite(number) else "--"

        lateral = float(metrics.get("ycb06_slot_lateral_error", math.inf))
        longitudinal = float(metrics.get("ycb06_slot_longitudinal_error", math.inf))
        goal_x, goal_y = _SLOT_CENTERS_TASK_XY[target]
        # Match the absolute plate-relative distances in the instruction.
        # Previously this row only showed residuals (goal = 0), which looked
        # inconsistent with the 19/29 cm distances above it.
        side = "LEFT" if goal_x < 0 else "RIGHT"
        side_sign = -1.0 if goal_x < 0 else 1.0

        def centimeters(number: float) -> str:
            return f"{number * 100:.1f}" if math.isfinite(number) else "--"

        position = (
            f"From plate: {side} {centimeters(side_sign * (goal_x + lateral))}"
            f" / {abs(goal_x) * 100:g}; FAR {centimeters(goal_y + longitudinal)}"
            f" / {goal_y * 100:g} cm (now / goal). "
        )
        if target == "mug":
            distance = math.hypot(lateral, longitudinal)
            distance_text = f"{distance * 100:.1f}" if math.isfinite(distance) else "--"
            position_label = (
                position + f"Center error {distance_text} <= {_MUG_XY_TOLERANCE * 100:g} cm"
            )
            position_ok = distance <= _MUG_XY_TOLERANCE
            orientation_label = (
                f"Opening UP: tilt error {value('ycb06_mug_opening_error_deg')}"
                f" <= {_MUG_OPENING_TOLERANCE_DEG:g} deg"
            )
            orientation_ok = (
                float(metrics.get("ycb06_mug_opening_error_deg", math.inf))
                <= _MUG_OPENING_TOLERANCE_DEG
            )
            footprint_label = "Mug footprint: center tolerance applies"
            footprint_ok = position_ok
        else:
            position_label = (
                position + f"Tolerance +/-{_UTENSIL_LATERAL_TOLERANCE * 100:g}"
                f" / +/-{_UTENSIL_LONGITUDINAL_TOLERANCE * 100:g} cm"
            )
            position_ok = (
                abs(lateral) <= _UTENSIL_LATERAL_TOLERANCE
                and abs(longitudinal) <= _UTENSIL_LONGITUDINAL_TOLERANCE
            )
            orientation_label = (
                f"Handle TOWARD YOU: angle error {value('ycb06_orientation_error_deg')}"
                f" <= {_UTENSIL_ORIENTATION_TOLERANCE_DEG:g} deg"
            )
            orientation_ok = (
                float(metrics.get("ycb06_orientation_error_deg", math.inf))
                <= _UTENSIL_ORIENTATION_TOLERANCE_DEG
            )
            if target == "knife":
                blade_ok = bool(metrics.get("ycb06_blade_toward_plate"))
                orientation_label += f"; blade toward plate: {'YES' if blade_ok else 'NO'}"
                orientation_ok = orientation_ok and blade_ok
            footprint_label = (
                "Target footprint inside goal "
                f"{value('ycb06_footprint_overlap_ratio', 100)} / >=80%"
            )
            footprint_ok = (
                float(metrics.get("ycb06_footprint_overlap_ratio", 0.0))
                >= _UTENSIL_MIN_FOOTPRINT_OVERLAP
            )
        stable_seconds = min(_STABLE_SECONDS, self._ycb06_stable_steps * self._control_dt)
        others_ok = bool(
            metrics.get("ycb06_non_target_slot_clear")
            and metrics.get("ycb06_all_dynamic_bodies_on_table")
            and not any(self._ycb06_wrong_counters.values())
        )
        return [
            self._condition_row(
                "ycb06_lifted",
                f"ONLY {target.upper()} grasped and lifted >=2 cm",
                bool(metrics.get("ycb06_target_lifted_seen")),
            ),
            self._condition_row("ycb06_position", position_label, position_ok),
            self._condition_row("ycb06_orientation", orientation_label, orientation_ok),
            self._condition_row("ycb06_footprint", footprint_label, footprint_ok),
            self._condition_row(
                "ycb06_support",
                "Target rests on TABLE, clear of plate/box/other tableware",
                bool(metrics.get("ycb06_valid_table_support")),
            ),
            self._condition_row(
                "ycb06_released",
                "BOTH hands released; nearest hand clearance "
                f"{value('ycb06_hand_surface_clearance', 100)} / >=2.5 cm",
                bool(metrics.get("ycb06_target_released")),
            ),
            self._condition_row(
                "ycb06_reference",
                f"Plate moved {value('ycb06_plate_translation', 100)} / 3 cm; "
                f"tilt {value('ycb06_plate_tilt_deg')} / 8 deg",
                bool(metrics.get("ycb06_plate_success_ok")),
            ),
            self._condition_row(
                "ycb06_others",
                "Other tableware untouched; goal clear; nothing fell",
                others_ok,
            ),
            self._condition_row(
                "ycb06_stable",
                "Correct placement + release + stability: "
                f"{stable_seconds:.2f} / 0.50 s",
                bool(metrics.get("ycb06_success_now")),
            ),
        ]

    def _clear_ycb06_episode_state(self) -> None:
        self._ycb06_initial_poses = {}
        self._ycb06_plate_reset_pose = None
        self._ycb06_target_contact_seen = False
        self._ycb06_target_contact_recent = 0
        self._ycb06_target_lifted_seen = False
        self._ycb06_placement_seen = False
        self._ycb06_release_seen = False
        self._ycb06_stable_steps = 0
        self._ycb06_wrong_counters = {
            name: 0 for name in YCB06_TARGET_NAMES
        }
        self._ycb06_plate_failure_counter = 0
        self._ycb06_first_terminal_reason = ""
        self._last_ycb06_metrics = {}

    def reset(self, seed: int | None = None, **kwargs: Any):
        requested = kwargs.get("target_label") or kwargs.get("target_object")
        if requested is not None and str(requested) not in YCB06_TARGET_NAMES:
            raise ValueError(
                f"Unsupported YCB06 target {requested!r}; expected one of "
                f"{YCB06_TARGET_NAMES}."
            )
        self._clear_ycb06_episode_state()
        observation, info = super().reset(seed=seed, **kwargs)

        if requested is not None:
            self._target_object_name = str(requested)
        elif self._ycb06_fixed_target is not None:
            self._target_object_name = self._ycb06_fixed_target
        else:
            index = int(self._random.randint(0, len(YCB06_TARGET_NAMES)))
            self._target_object_name = YCB06_TARGET_NAMES[index]

        self._refresh_object_poses()
        self._ycb06_initial_poses = {
            name: self._poses.get(name, self._fallback_pose(name)).copy()
            for name in YCB06_DYNAMIC_NAMES
        }
        self._ycb06_plate_reset_pose = self._ycb06_initial_poses["plate"].copy()
        self._ycb06_reset_ordinal += 1
        self._last_ycb06_metrics = self._ycb06_metrics(update_history=False)
        self._update_success_condition_overlay()
        info = dict(info)
        info.update(self._task_info())
        observation = self._attach_physx_ground_truth(
            self._compute_observation()
        )
        print(
            f"[YCB06] Reset seed={self._seed}; target={self._target_object_name}; "
            f"selection={'fixed' if requested is not None or self._ycb06_fixed_target is not None else 'random'}; "
            f"prompt={self._target_prompt()}",
            flush=True,
        )
        return observation, info

    def _task_specific_reset_xy_positions(self) -> dict[str, np.ndarray] | None:
        self._ycb06_reset_group_yaw = 0.0
        self._ycb06_reset_fixture_xy = _FIXTURE_XY.copy()
        self._ycb06_reset_mug_yaw = 0.0
        if not self.randomize:
            return {
                **{
                    name: bounds.mean(axis=0)
                    for name, bounds in _RESET_XY_BOUNDS.items()
                },
                YCB06_FIXTURE_NAME: _FIXTURE_XY.copy(),
            }

        hand_points = (
            *self._reset_robot_body_points("right"),
            *self._reset_robot_body_points("left"),
        )
        best_margin = -math.inf
        for _ in range(_RESET_MAX_ATTEMPTS):
            self._ycb06_reset_group_yaw = float(self._random.uniform(-180, 180))
            rotation = _rotation_xy(self._ycb06_reset_group_yaw)
            offset = self._random.uniform(*_RESET_GROUP_XY_BOUNDS)
            candidate = {
                name: rotation @ _RESET_XY_BOUNDS[name].mean(axis=0) + offset
                for name in _UTENSIL_RESET_PITCH_DEG
            }
            self._ycb06_reset_fixture_xy = rotation @ _FIXTURE_XY + offset
            candidate[YCB06_FIXTURE_NAME] = self._ycb06_reset_fixture_xy.copy()
            candidate["plate"] = self._random.uniform(*_RESET_PLATE_XY_BOUNDS)
            candidate["mug"] = self._random.uniform(*_RESET_MUG_XY_BOUNDS)
            self._ycb06_reset_mug_yaw = float(self._random.uniform(-180, 180))
            margin = self._ycb06_reset_layout_margin(candidate, hand_points)
            best_margin = max(best_margin, margin)
            if margin >= 0.0 and self._ycb06_initially_outside_slots(candidate):
                return candidate
        raise RuntimeError(
            "YCB06 could not sample a collision-free reachable reset layout; "
            f"best_margin={best_margin:.6f}"
        )

    @staticmethod
    def _aabb_pair_margin(
        center_a: np.ndarray,
        half_a: np.ndarray,
        center_b: np.ndarray,
        half_b: np.ndarray,
        clearance: float,
    ) -> float:
        gaps = (
            np.abs(np.asarray(center_a) - np.asarray(center_b))
            - np.asarray(half_a)
            - np.asarray(half_b)
            - float(clearance)
        )
        return float(np.max(gaps))

    def _ycb06_reset_layout_margin(
        self,
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...],
    ) -> float:
        margin = math.inf
        names = tuple(name for name in xy_by_name if name in YCB06_DYNAMIC_NAMES)
        group = (*_UTENSIL_RESET_PITCH_DEG, YCB06_FIXTURE_NAME)
        half_by_name = {}
        for name, half in _RESET_XY_HALF_EXTENTS.items():
            angle = (
                self._ycb06_reset_group_yaw if name in group
                else self._ycb06_reset_mug_yaw if name == "mug"
                else 0.0
            )
            half_by_name[name] = np.abs(_rotation_xy(angle)) @ half
        for index, name_a in enumerate(names):
            for name_b in names[index + 1 :]:
                # Their relative geometry is invariant under the shared
                # transform; world AABBs would reject rotated parallel tools.
                if name_a in group and name_b in group:
                    continue
                margin = min(
                    margin,
                    self._aabb_pair_margin(
                        xy_by_name[name_a],
                        half_by_name[name_a],
                        xy_by_name[name_b],
                        half_by_name[name_b],
                        _RESET_CLEARANCE,
                    ),
                )

        # The utensil handles intentionally overlap the fixed gelatin-box
        # footprint in XY.  Plate and mug must remain fully separate from it.
        for name in ("plate", "mug"):
            margin = min(
                margin,
                self._aabb_pair_margin(
                    xy_by_name[name],
                    half_by_name[name],
                    self._ycb06_reset_fixture_xy,
                    half_by_name[YCB06_FIXTURE_NAME],
                    _RESET_CLEARANCE,
                ),
            )

        for name in (*names, YCB06_FIXTURE_NAME):
            xy = xy_by_name.get(name, self._ycb06_reset_fixture_xy)
            half_xy = half_by_name[name]
            margin = min(
                margin,
                float(xy[0] - half_xy[0] - _TABLE_SAFE_X[0]),
                float(_TABLE_SAFE_X[1] - xy[0] - half_xy[0]),
                float(xy[1] - half_xy[1] - _TABLE_SAFE_Y[0]),
                float(_TABLE_SAFE_Y[1] - xy[1] - half_xy[1]),
            )
            center = np.asarray(
                [
                    xy[0],
                    xy[1],
                    _RESET_CENTER_Z.get(name, _FIXTURE_CENTER_Z) + float(self.delta_h[0]),
                ],
                dtype=np.float64,
            )
            half_extents = np.asarray(
                [
                    half_xy[0],
                    half_xy[1],
                    _RESET_Z_HALF_EXTENTS.get(name, 0.014),
                ],
                dtype=np.float64,
            )
            for point in hand_points:
                surface_delta = np.maximum(
                    np.abs(np.asarray(point) - center) - half_extents,
                    0.0,
                )
                margin = min(
                    margin,
                    float(np.linalg.norm(surface_delta))
                    - _RESET_HAND_ORIGIN_CLEARANCE,
                )
        return float(margin)

    def _ycb06_initially_outside_slots(
        self,
        xy_by_name: dict[str, np.ndarray],
    ) -> bool:
        plate = np.asarray(xy_by_name["plate"], dtype=np.float64)
        for name in YCB06_TARGET_NAMES:
            task_xy = _TASK_FROM_WORLD[:2, :2] @ (
                np.asarray(xy_by_name[name], dtype=np.float64) - plate
            )
            # Keep every semantic destination available, including goals
            # belonging to a different target sampled after this layout.
            for slot_name, slot_xy in _SLOT_CENTERS_TASK_XY.items():
                error = task_xy - slot_xy
                if slot_name == "mug":
                    if float(np.linalg.norm(error)) <= _MUG_XY_TOLERANCE + 0.06:
                        return False
                elif (
                    abs(float(error[0])) <= _UTENSIL_LATERAL_TOLERANCE + 0.06
                    and abs(float(error[1])) <= _UTENSIL_LONGITUDINAL_TOLERANCE + 0.10
                ):
                    return False
        return True

    def _customize_task_reset_pose(
        self,
        name: str,
        pose_xyzw: np.ndarray,
        *,
        applied_height: float,
    ) -> np.ndarray:
        result = np.asarray(pose_xyzw, dtype=np.float64).copy().reshape(-1, 7)
        if name == YCB06_FIXTURE_NAME:
            result[:, :2] = self._ycb06_reset_fixture_xy
            result[:, 2] = _FIXTURE_CENTER_Z + float(applied_height)
            result[:, 3:7] = _quat_xyzw(_quat_wxyz_from_zx_degrees(
                90.0 + self._ycb06_reset_group_yaw, 90.0))
            return result
        if name not in YCB06_DYNAMIC_NAMES:
            return result
        result[:, 2] = _RESET_CENTER_Z[name] + float(applied_height)
        if name in {"fork", "knife", "spoon"}:
            yaw = 90.0 + self._ycb06_reset_group_yaw
            rotation = _quat_wxyz_to_matrix(_quat_wxyz_from_zx_degrees(yaw, 0))
            tilt = math.radians(_UTENSIL_RESET_SIDE_TILT_DEG[name])
            rotation = rotation @ np.asarray([
                [math.cos(tilt), 0.0, math.sin(tilt)],
                [0.0, 1.0, 0.0],
                [-math.sin(tilt), 0.0, math.cos(tilt)],
            ]) @ _quat_wxyz_to_matrix(_quat_wxyz_from_zx_degrees(
                0.0, _UTENSIL_RESET_PITCH_DEG[name]))
            result[:, 3:7] = _quat_xyzw(_matrix_to_quat_wxyz(rotation))
        elif name == "mug":
            result[:, 3:7] = _quat_xyzw(_quat_wxyz_from_zx_degrees(
                self._ycb06_reset_mug_yaw, 0.0))
        return result

    def _init_ycb06_contact_views(self) -> None:
        self._ycb06_contact_views = {}
        if not self._robots:
            return
        physics_sim_view = self._robots["right"]._physics_sim_view
        hand_paths = self._ycb04_hand_body_paths()
        if physics_sim_view is None or not hand_paths:
            self._add_task_state_warning("YCB06 hand contact reporting is unavailable")
            return
        try:
            for name in YCB06_TARGET_NAMES:
                prim = self._find_rigid_body_prim(name)
                if prim is None:
                    continue
                view = physics_sim_view.create_rigid_contact_view(
                    str(prim.GetPath()),
                    filter_patterns=hand_paths,
                    max_contact_data_count=max(64, len(hand_paths) * 8),
                )
                if (
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                ):
                    self._ycb06_contact_views[name] = view
        except Exception as exc:
            self._ycb06_contact_views = {}
            self._add_task_state_warning(
                f"Could not initialize YCB06 contact reporting: {exc}"
            )
        if len(self._ycb06_contact_views) != len(YCB06_TARGET_NAMES):
            self._ycb06_contact_views = {}
            self._add_task_state_warning(
                "YCB06 contact reporting did not resolve all four tableware bodies"
            )

    def _ycb06_target_contact(self) -> tuple[float, bool, bool]:
        view = self._ycb06_contact_views.get(self._target_object_name)
        force = self._c04_contact_force(view)
        available = len(self._ycb06_contact_views) == len(YCB06_TARGET_NAMES)
        if available:
            return force, True, force > _CONTACT_FORCE_THRESHOLD
        if self._launch_isaac:
            return 0.0, False, False
        target = self._poses.get(
            self._target_object_name,
            self._fallback_pose(self._target_object_name),
        )
        clearance = self._ycb06_hand_surface_clearance(target)
        return 0.0, False, clearance <= 0.006

    def _ycb06_hand_surface_clearance(self, target_pose: np.ndarray) -> float:
        min_distance = math.inf
        half_extents = _BODY_HALF_EXTENTS[self._target_object_name]
        for point in (
            *self._a06_hand_points("right"), *self._a06_hand_points("left"),
        ):
            min_distance = min(
                min_distance,
                self._point_to_block_surface_distance(
                    point,
                    target_pose,
                    half_extents,
                ),
            )
        return float(min_distance)

    def _task_xy(self, world_xy: np.ndarray) -> np.ndarray:
        if self._ycb06_plate_reset_pose is None:
            return np.zeros(2, dtype=np.float64)
        world_delta = np.asarray(
            [
                float(world_xy[0] - self._ycb06_plate_reset_pose[0]),
                float(world_xy[1] - self._ycb06_plate_reset_pose[1]),
                0.0,
            ],
            dtype=np.float64,
        )
        return (_TASK_FROM_WORLD @ world_delta)[:2]

    def _slot_pose(self, name: str | None = None) -> np.ndarray:
        target_name = name or self._target_object_name
        center_task = _SLOT_CENTERS_TASK_XY[target_name]
        plate = (
            self._ycb06_plate_reset_pose
            if self._ycb06_plate_reset_pose is not None
            else self._poses.get("plate", self._fallback_pose("plate"))
        )
        world_center = np.asarray(plate[:3], dtype=np.float64).copy()
        world_center[:2] += (
            _WORLD_FROM_TASK[:2, :2] @ center_task
        )
        world_center[2] = self._current_table_top_z()
        if target_name == "knife":
            rotation = _WORLD_FROM_TASK @ np.diag([-1.0, 1.0, -1.0])
        else:
            rotation = _WORLD_FROM_TASK.copy()
        return np.concatenate(
            [world_center, _matrix_to_quat_wxyz(rotation)]
        ).astype(np.float64)

    def _utensil_overlap_ratio(
        self,
        name: str,
        pose: np.ndarray,
    ) -> float:
        half_x, half_y = _BODY_HALF_EXTENTS[name][:2]
        corners_local = np.asarray(
            [
                [-half_x, -half_y, 0.0, 1.0],
                [half_x, -half_y, 0.0, 1.0],
                [half_x, half_y, 0.0, 1.0],
                [-half_x, half_y, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        corners_world = (
            _pose_vector_to_matrix(pose) @ corners_local.T
        ).T[:, :3]
        corners_task = np.stack(
            [self._task_xy(point[:2]) for point in corners_world],
            axis=0,
        )
        centered = corners_task - _SLOT_CENTERS_TASK_XY[name]
        clipped = self._clip_polygon_to_axis_aligned_box(
            centered,
            _UTENSIL_SLOT_HALF_EXTENTS[name],
        )
        area = 4.0 * float(half_x) * float(half_y)
        return float(np.clip(self._polygon_area(clipped) / area, 0.0, 1.0))

    def _plate_metrics(self) -> dict[str, float | bool]:
        plate = self._poses.get("plate", self._fallback_pose("plate"))
        reset = self._ycb06_plate_reset_pose
        if reset is None:
            return {
                "translation": math.inf,
                "tilt_deg": math.inf,
                "success_ok": False,
                "failure_now": False,
            }
        translation = float(np.linalg.norm(plate[:3] - reset[:3]))
        local_z = _quat_wxyz_to_matrix(plate[3:7])[:, 2]
        tilt_deg = float(
            np.rad2deg(
                np.arccos(np.clip(float(local_z[2]), -1.0, 1.0))
            )
        )
        return {
            "translation": translation,
            "tilt_deg": tilt_deg,
            "success_ok": bool(
                translation <= _PLATE_SUCCESS_TRANSLATION
                and tilt_deg <= _PLATE_SUCCESS_TILT_DEG
            ),
            "failure_now": bool(
                translation > _PLATE_FAILURE_TRANSLATION
                or tilt_deg > _PLATE_FAILURE_TILT_DEG
            ),
        }

    def _valid_table_support(self, name: str, pose: np.ndarray) -> tuple[bool, float]:
        rotation = _quat_wxyz_to_matrix(pose[3:7])
        vertical_half_extent = float(
            np.dot(np.abs(rotation[2, :]), _BODY_HALF_EXTENTS[name])
        )
        bottom_error = float(
            pose[2] - vertical_half_extent - self._current_table_top_z()
        )
        support_ok = abs(bottom_error) <= _SUPPORT_HEIGHT_TOLERANCE

        plate = self._poses.get("plate", self._fallback_pose("plate"))
        not_on_plate = float(np.linalg.norm(pose[:2] - plate[:2])) > 0.145
        not_on_movable = True
        for other_name in YCB06_TARGET_NAMES:
            if other_name == name:
                continue
            other = self._poses.get(other_name, self._fallback_pose(other_name))
            radius = float(np.linalg.norm(_BODY_HALF_EXTENTS[other_name][:2]))
            if (
                float(np.linalg.norm(pose[:2] - other[:2])) <= radius
                and pose[2] - vertical_half_extent
                > self._current_table_top_z() + 0.015
            ):
                not_on_movable = False
                break
        return bool(support_ok and not_on_plate and not_on_movable), bottom_error

    def _non_target_in_active_slot(self) -> bool:
        slot = _SLOT_CENTERS_TASK_XY[self._target_object_name]
        for name in YCB06_TARGET_NAMES:
            if name == self._target_object_name:
                continue
            pose = self._poses.get(name, self._fallback_pose(name))
            error = self._task_xy(pose[:2]) - slot
            if self._target_object_name == "mug":
                if float(np.linalg.norm(error)) <= _MUG_XY_TOLERANCE:
                    return True
            elif (
                abs(float(error[0])) <= _UTENSIL_LATERAL_TOLERANCE
                and abs(float(error[1])) <= _UTENSIL_LONGITUDINAL_TOLERANCE
            ):
                return True
        return False

    def _body_fell_off_table(self, name: str) -> bool:
        pose = self._poses.get(name, self._fallback_pose(name))
        return bool(
            pose[2] < self._current_table_top_z() - 0.030
            or pose[0] < _TABLE_SAFE_X[0]
            or pose[0] > _TABLE_SAFE_X[1]
            or pose[1] < _TABLE_SAFE_Y[0]
            or pose[1] > _TABLE_SAFE_Y[1]
        )

    def _ycb06_metrics(self, *, update_history: bool) -> dict[str, Any]:
        target_name = self._target_object_name
        target = self._poses.get(target_name, self._fallback_pose(target_name))
        initial = self._ycb06_initial_poses.get(target_name)
        plate_metrics = self._plate_metrics()
        target_task_xy = self._task_xy(target[:2])
        slot_error = target_task_xy - _SLOT_CENTERS_TASK_XY[target_name]
        contact_force, contact_available, contact_now = self._ycb06_target_contact()
        hand_clearance = self._ycb06_hand_surface_clearance(target)

        if update_history:
            if contact_now:
                self._ycb06_target_contact_seen = True
                self._ycb06_target_contact_recent = _CONTACT_RECENT_STEPS
            else:
                self._ycb06_target_contact_recent = max(
                    0,
                    self._ycb06_target_contact_recent - 1,
                )
            if (
                initial is not None
                and self._ycb06_target_contact_seen
                and self._ycb06_target_contact_recent > 0
                and float(target[2] - initial[2]) >= _LIFT_HEIGHT
            ):
                self._ycb06_target_lifted_seen = True

        overlap_ratio = 0.0
        orientation_error_deg = math.inf
        blade_toward_plate = target_name != "knife"
        mug_opening_error_deg = math.inf
        if target_name in {"fork", "knife", "spoon"}:
            rotation = _quat_wxyz_to_matrix(target[3:7])
            long_axis_task = _TASK_FROM_WORLD @ rotation[:, 1]
            orientation_error_deg = float(
                np.rad2deg(
                    np.arccos(
                        np.clip(float(long_axis_task[1]), -1.0, 1.0)
                    )
                )
            )
            overlap_ratio = self._utensil_overlap_ratio(target_name, target)
            if target_name == "knife":
                side_axis_task = _TASK_FROM_WORLD @ rotation[:, 0]
                direction_to_plate = -target_task_xy
                direction_norm = float(np.linalg.norm(direction_to_plate))
                blade_toward_plate = bool(
                    direction_norm > 1.0e-9
                    and float(
                        np.dot(
                            side_axis_task[:2],
                            direction_to_plate / direction_norm,
                        )
                    )
                    > 0.0
                )
            semantic_pose_ok = bool(
                abs(float(slot_error[0])) <= _UTENSIL_LATERAL_TOLERANCE
                and abs(float(slot_error[1])) <= _UTENSIL_LONGITUDINAL_TOLERANCE
                and overlap_ratio >= _UTENSIL_MIN_FOOTPRINT_OVERLAP
                and orientation_error_deg <= _UTENSIL_ORIENTATION_TOLERANCE_DEG
                and blade_toward_plate
            )
        else:
            opening_axis = _quat_wxyz_to_matrix(target[3:7])[:, 2]
            mug_opening_error_deg = float(
                np.rad2deg(
                    np.arccos(np.clip(float(opening_axis[2]), -1.0, 1.0))
                )
            )
            semantic_pose_ok = bool(
                float(np.linalg.norm(slot_error)) <= _MUG_XY_TOLERANCE
                and mug_opening_error_deg <= _MUG_OPENING_TOLERANCE_DEG
            )

        support_ok, bottom_error = self._valid_table_support(target_name, target)
        released = bool(
            (contact_available and contact_force <= _CONTACT_FORCE_THRESHOLD)
            or (not self._launch_isaac and not contact_now)
        ) and hand_clearance >= _HAND_CLEARANCE
        velocity = self._physics_body_velocity(target_name)
        velocity_available = velocity is not None
        linear_speed = (
            float(np.linalg.norm(velocity[:3])) if velocity is not None else math.inf
        )
        angular_speed = (
            float(np.linalg.norm(velocity[3:])) if velocity is not None else math.inf
        )
        if not self._launch_isaac and velocity is None:
            linear_speed = 0.0
            angular_speed = 0.0
        stable_now = bool(
            linear_speed < _STABLE_LINEAR_SPEED
            and angular_speed < _STABLE_ANGULAR_SPEED
        )
        no_non_target_slot = not self._non_target_in_active_slot()
        all_on_table = not any(
            self._body_fell_off_table(name) for name in YCB06_DYNAMIC_NAMES
        )
        placement_now = bool(
            semantic_pose_ok
            and support_ok
            and plate_metrics["success_ok"]
            and no_non_target_slot
            and all_on_table
            and not any(self._ycb06_wrong_counters.values())
        )

        if update_history:
            if self._ycb06_target_lifted_seen and placement_now:
                self._ycb06_placement_seen = True
            if self._ycb06_placement_seen and released:
                self._ycb06_release_seen = True
            if (
                self._ycb06_release_seen
                and placement_now
                and released
                and stable_now
            ):
                self._ycb06_stable_steps += 1
            else:
                self._ycb06_stable_steps = 0

        stable_steps_required = max(
            1,
            int(math.ceil(_STABLE_SECONDS / max(float(self._control_dt), 1.0e-6))),
        )
        success_now = bool(
            self._ycb06_target_lifted_seen
            and self._ycb06_placement_seen
            and self._ycb06_release_seen
            and self._ycb06_stable_steps >= stable_steps_required
            and placement_now
            and released
            and stable_now
        )
        return {
            "ycb06_success_now": success_now,
            "ycb06_target": target_name,
            "ycb06_prompt": self._target_prompt(),
            "ycb06_target_contact_seen": self._ycb06_target_contact_seen,
            "ycb06_target_contact_force": float(contact_force),
            "ycb06_target_contact_available": bool(contact_available),
            "ycb06_target_lifted_seen": self._ycb06_target_lifted_seen,
            "ycb06_placement_seen": self._ycb06_placement_seen,
            "ycb06_release_seen": self._ycb06_release_seen,
            "ycb06_stable_steps": int(self._ycb06_stable_steps),
            "ycb06_stable_steps_required": int(stable_steps_required),
            "ycb06_semantic_pose_ok": bool(semantic_pose_ok),
            "ycb06_valid_table_support": bool(support_ok),
            "ycb06_target_released": bool(released),
            "ycb06_target_stable": bool(stable_now),
            "ycb06_slot_lateral_error": float(slot_error[0]),
            "ycb06_slot_longitudinal_error": float(slot_error[1]),
            "ycb06_footprint_overlap_ratio": float(overlap_ratio),
            "ycb06_orientation_error_deg": float(orientation_error_deg),
            "ycb06_blade_toward_plate": bool(blade_toward_plate),
            "ycb06_mug_opening_error_deg": float(mug_opening_error_deg),
            "ycb06_support_bottom_error": float(bottom_error),
            "ycb06_hand_surface_clearance": float(hand_clearance),
            "ycb06_target_linear_speed": float(linear_speed),
            "ycb06_target_angular_speed": float(angular_speed),
            "ycb06_velocity_available": bool(velocity_available),
            "ycb06_plate_translation": float(plate_metrics["translation"]),
            "ycb06_plate_tilt_deg": float(plate_metrics["tilt_deg"]),
            "ycb06_plate_success_ok": bool(plate_metrics["success_ok"]),
            "ycb06_non_target_slot_clear": bool(no_non_target_slot),
            "ycb06_all_dynamic_bodies_on_table": bool(all_on_table),
        }

    def _compute_success_now(self) -> bool:
        self._last_ycb06_metrics = self._ycb06_metrics(update_history=True)
        return bool(self._last_ycb06_metrics["ycb06_success_now"])

    def _set_first_terminal_reason(self, reason: str) -> None:
        if not self._ycb06_first_terminal_reason:
            self._ycb06_first_terminal_reason = str(reason)

    def _compute_failure(self) -> bool:
        if self._ycb06_first_terminal_reason in {
            "wrong_object_manipulated",
            "object_fell_off_table",
            "reference_plate_displaced",
            "robot_safety_violation",
        }:
            return True

        wrong_steps = max(
            1,
            int(
                math.ceil(
                    _WRONG_DEBOUNCE_SECONDS
                    / max(float(self._control_dt), 1.0e-6)
                )
            ),
        )
        for name in YCB06_TARGET_NAMES:
            if name == self._target_object_name:
                continue
            pose = self._poses.get(name, self._fallback_pose(name))
            initial = self._ycb06_initial_poses.get(name)
            manipulated = bool(
                initial is not None
                and (
                    float(pose[2] - initial[2]) > _LIFT_HEIGHT
                    or float(np.linalg.norm(pose[:3] - initial[:3]))
                    > _WRONG_DISPLACEMENT
                )
            )
            self._ycb06_wrong_counters[name] = (
                self._ycb06_wrong_counters[name] + 1 if manipulated else 0
            )
            if self._ycb06_wrong_counters[name] >= wrong_steps:
                self._set_first_terminal_reason("wrong_object_manipulated")
                return True

        if any(self._body_fell_off_table(name) for name in YCB06_DYNAMIC_NAMES):
            self._set_first_terminal_reason("object_fell_off_table")
            return True

        plate_failure_now = bool(self._plate_metrics()["failure_now"])
        self._ycb06_plate_failure_counter = (
            self._ycb06_plate_failure_counter + 1
            if plate_failure_now
            else 0
        )
        plate_failure_steps = max(
            1,
            int(
                math.ceil(
                    _PLATE_FAILURE_DEBOUNCE_SECONDS
                    / max(float(self._control_dt), 1.0e-6)
                )
            ),
        )
        if self._ycb06_plate_failure_counter >= plate_failure_steps:
            self._set_first_terminal_reason("reference_plate_displaced")
            return True

        if bool(getattr(self, "_robot_safety_violation", False)):
            self._set_first_terminal_reason("robot_safety_violation")
            return True
        return False

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        fixture_pose = self._poses.get(
            YCB06_FIXTURE_NAME,
            self._fallback_pose(YCB06_FIXTURE_NAME),
        ).copy()
        observation["state"]["gelatin_box_pose"] = fixture_pose
        target_index = YCB06_TARGET_NAMES.index(self._target_object_name)
        selector = np.zeros(len(YCB06_TARGET_NAMES), dtype=np.float64)
        selector[target_index] = 1.0
        initial_poses = np.concatenate(
            [
                self._ycb06_initial_poses.get(name, np.zeros(7, dtype=np.float64))
                for name in YCB06_DYNAMIC_NAMES
            ]
        )
        metrics = self._last_ycb06_metrics
        observation.update(
            {
                "ycb06_target_selector": selector,
                "ycb06_plate_reset_pose": (
                    self._ycb06_plate_reset_pose.copy()
                    if self._ycb06_plate_reset_pose is not None
                    else np.zeros(7, dtype=np.float64)
                ),
                "ycb06_slot_pose": self._slot_pose(),
                "ycb06_initial_object_poses": initial_poses,
                "ycb06_fixture_pose": fixture_pose,
                "ycb06_event_state": np.asarray(
                    [
                        float(self._ycb06_target_contact_seen),
                        float(self._ycb06_target_lifted_seen),
                        float(self._ycb06_placement_seen),
                        float(self._ycb06_release_seen),
                        float(self._ycb06_stable_steps),
                        float(metrics.get("ycb06_semantic_pose_ok", False)),
                        float(metrics.get("ycb06_valid_table_support", False)),
                        float(metrics.get("ycb06_target_released", False)),
                        float(metrics.get("ycb06_target_stable", False)),
                        float(any(self._ycb06_wrong_counters.values())),
                        float(self._ycb06_plate_failure_counter),
                        float(metrics.get("ycb06_success_now", False)),
                    ],
                    dtype=np.float64,
                ),
                "ycb06_terminal_reason_code": np.asarray(
                    [
                        _TERMINAL_REASON_CODE.get(
                            self._ycb06_first_terminal_reason,
                            0,
                        )
                    ],
                    dtype=np.float64,
                ),
                "ycb06_reset_metadata": np.asarray(
                    [float(self._seed), float(self._ycb06_reset_ordinal)],
                    dtype=np.float64,
                ),
            }
        )
        return observation

    def _task_info(self) -> dict[str, Any]:
        info = dict(super()._task_info())
        info.update(self._last_ycb06_metrics)
        info.update(
            {
                "task_id": "YCB06",
                "target_name": self._target_object_name,
                "target_id": YCB06_TARGET_NAMES.index(self._target_object_name),
                "target_prompt": self._target_prompt(),
                "target_slot_pose": self._slot_pose().tolist(),
                "first_terminal_reason": self._ycb06_first_terminal_reason,
                "wrong_object_counters": dict(self._ycb06_wrong_counters),
                "reset_seed": int(self._seed),
                "reset_ordinal": int(self._ycb06_reset_ordinal),
            }
        )
        return info

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = super().step(action)
        success = bool(info.get("succeed", False))
        failed = bool(info.get("failed", False))
        manual_reset = bool(info.get("manual_reset", False))
        timed_out = bool(
            not success
            and not failed
            and not manual_reset
            and self.env_step * self._control_dt >= self._time_limit
        )
        if success:
            self._set_first_terminal_reason("success")
        elif failed:
            pass
        elif timed_out:
            self._set_first_terminal_reason("timeout")
            terminated = False
            truncated = True
        elif manual_reset:
            self._set_first_terminal_reason("manual_reset")

        info = dict(info)
        info.update(self._task_info())
        if terminated or truncated:
            # Log the terminal predicates before the recorder replaces this
            # episode. Generic unsuccessful-terminal logs lose the target
            # identity and make incidental tableware motion hard to diagnose.
            summary = {
                "seed": int(self._seed),
                "target": self._target_object_name,
                "reason": self._ycb06_first_terminal_reason,
                "step": int(self.env_step),
                "unmet_conditions": [
                    row["key"] for row in self._success_condition_rows()
                    if not row["ok"]
                ],
                "wrong_object_counters": dict(self._ycb06_wrong_counters),
            }
            print("[YCB06] Terminal " + json.dumps(summary, sort_keys=True), flush=True)
        if isinstance(observation, dict):
            observation["ycb06_terminal_reason_code"] = np.asarray(
                [
                    _TERMINAL_REASON_CODE.get(
                        self._ycb06_first_terminal_reason,
                        0,
                    )
                ],
                dtype=np.float64,
            )
        return observation, reward, bool(terminated), bool(truncated), info

    def restore_recorded_initial_state(
        self,
        state_parts: Mapping[str, np.ndarray],
    ) -> dict[str, Any]:
        observation = super().restore_recorded_initial_state(state_parts)
        selector = np.asarray(
            state_parts.get("ycb06_target_selector", np.zeros(0)),
            dtype=np.float64,
        ).ravel()
        if selector.size == len(YCB06_TARGET_NAMES):
            self._target_object_name = YCB06_TARGET_NAMES[int(np.argmax(selector))]
        plate_reset = np.asarray(
            state_parts.get("ycb06_plate_reset_pose", np.zeros(0)),
            dtype=np.float64,
        ).ravel()
        if plate_reset.size >= 7:
            self._ycb06_plate_reset_pose = plate_reset[:7].copy()
        initial = np.asarray(
            state_parts.get("ycb06_initial_object_poses", np.zeros(0)),
            dtype=np.float64,
        ).ravel()
        if initial.size >= 7 * len(YCB06_DYNAMIC_NAMES):
            poses = initial[: 7 * len(YCB06_DYNAMIC_NAMES)].reshape(-1, 7)
            self._ycb06_initial_poses = {
                name: pose.copy()
                for name, pose in zip(YCB06_DYNAMIC_NAMES, poses)
            }
        event = np.asarray(
            state_parts.get("ycb06_event_state", np.zeros(0)),
            dtype=np.float64,
        ).ravel()
        if event.size >= 5:
            self._ycb06_target_contact_seen = bool(event[0])
            self._ycb06_target_lifted_seen = bool(event[1])
            self._ycb06_placement_seen = bool(event[2])
            self._ycb06_release_seen = bool(event[3])
            self._ycb06_stable_steps = int(round(float(event[4])))
        self._last_ycb06_metrics = self._ycb06_metrics(update_history=False)
        return self._compute_observation()


__all__ = [
    "HandsOnYCB06IsaacLabEnv",
    "YCB06_DYNAMIC_NAMES",
    "YCB06_FIXTURE_NAME",
    "YCB06_PROMPTS",
    "YCB06_TARGET_NAMES",
    "YCB06_TASK_NAME",
]
