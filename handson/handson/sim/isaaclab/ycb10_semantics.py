"""Pure physical semantics for YCB10 marker fingertip gait.

This module intentionally has no Isaac imports.  Live collection, UI metrics,
unit tests, and replay evaluation therefore share one ordered state machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

YCB10_TASK_NAME = "t33_reposition_marker_grasp"
YCB10_MARKER_NAME = "ycb10_marker"

YCB10_MARKER_HALF_LENGTH_M = 0.0605
YCB10_MARKER_RADIUS_M = 0.009772
# The original collision meshes have an octagonal cross-section.  Identity
# rotation balances the body on one vertex, which keeps a real PhysX contact
# manifold microscopically rolling.  Rest one unchanged full 3-D convex-hull
# face on the table instead.  The face spans the marker length and contains the
# COM projection with a positive margin; no support primitive or mesh scaling
# is involved.  Its 0.496-degree longitudinal slope remains far inside the
# task's approved 35-degree inclination limit.
YCB10_MARKER_STABLE_FACE_NORMAL = np.asarray(
    [-0.38264446, 0.00866162, -0.92385507], dtype=np.float64
)
YCB10_MARKER_STABLE_SUPPORT_M = 0.008552397288961086
YCB10_RESET_LOCAL_Y_TILT_DEG = -22.497583091245552
YCB10_RESET_LOCAL_X_TILT_DEG = -0.5371618836495896
YCB10_MARKER_MASS_KG = 0.015799936357279568
YCB10_MARKER_CENTER_OF_MASS_M = np.asarray(
    [2.00735220479312e-07, 0.00593074266394731, -5.15280192568516e-07],
    dtype=np.float64,
)
YCB10_MARKER_DIAGONAL_INERTIA_KGM2 = np.asarray(
    [1.68285700126238e-05, 5.2245746311005e-07, 1.6828584928925e-05],
    dtype=np.float64,
)
YCB10_CAPWARD_LOCAL_AXIS = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)

YCB10_LIFT_HEIGHT_M = 0.015
YCB10_AXIAL_PROGRESS_M = 0.025
YCB10_CONTACT_FORCE_N = 1.0e-4
YCB10_FINAL_SAMPLE_CAPACITY = (
    62  # Reserved v2 history slots keep existing prefix indices stable.
)
YCB10_HIDDEN_STATE_SIZE = 21 + 23 + 32 + 2 * YCB10_FINAL_SAMPLE_CAPACITY + 5

YCB10_WORKSPACE_LOWER = np.asarray([-0.55, -0.80, 0.84], dtype=np.float64)
YCB10_WORKSPACE_UPPER = np.asarray([0.25, 0.80, 1.60], dtype=np.float64)

YCB10_RESET_XY_BOUNDS = np.asarray([[-0.39, -0.30], [-0.17, -0.055]], dtype=np.float64)
YCB10_RESET_XY_FALLBACK = np.asarray([-0.29, -0.14], dtype=np.float64)
YCB10_RESET_YAW_BOUNDS_DEG = (-60.0, 60.0)
YCB10_RESET_SUPPORT_CLEARANCE_M = 0.00075

YCB10_PHASE_CODES = {
    "table_start": 0,
    "lift_and_grasp": 1,
    "fingertip_gait": 2,
    "final_hold": 3,
    "success": 4,
    "failure": 5,
}
YCB10_TERMINAL_REASON_CODES = {
    "": 0,
    "success": 1,
    "table_contact_after_reference": 2,
    "non_finger_support_contact": 3,
    "right_fingertip_contact_lost": 4,
    "workspace_exit_or_drop": 5,
    "irrecoverable_marker_velocity": 6,
    "robot_safety_violation": 7,
    "timeout": 8,
    "invalid_contact_state": 9,
    "invalid_physical_state": 10,
}


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(pose_wxyz: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_wxyz_to_matrix(pose[3:7])
    result[:3, 3] = pose[:3]
    return result


def invert_pose(matrix: np.ndarray) -> np.ndarray:
    source = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = source[:3, :3].T
    result[:3, 3] = -(result[:3, :3] @ source[:3, 3])
    return result


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _axis_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(3)
    second = np.asarray(second, dtype=np.float64).reshape(3)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1.0e-12:
        return math.inf
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def sample_ycb10_reset(
    random_state: Any,
    *,
    randomize: bool,
    table_top_z: float,
) -> np.ndarray:
    """Return a seeded horizontal, source-scale, table-clear marker pose."""
    if randomize:
        xy = random_state.uniform(YCB10_RESET_XY_BOUNDS[0], YCB10_RESET_XY_BOUNDS[1])
        yaw_degrees = float(random_state.uniform(*YCB10_RESET_YAW_BOUNDS_DEG))
    else:
        xy = YCB10_RESET_XY_FALLBACK.copy()
        yaw_degrees = 0.0
    yaw = math.radians(yaw_degrees)
    local_y_tilt = math.radians(YCB10_RESET_LOCAL_Y_TILT_DEG)
    local_x_tilt = math.radians(YCB10_RESET_LOCAL_X_TILT_DEG)
    yaw_cos = math.cos(0.5 * yaw)
    yaw_sin = math.sin(0.5 * yaw)
    y_cos = math.cos(0.5 * local_y_tilt)
    y_sin = math.sin(0.5 * local_y_tilt)
    x_cos = math.cos(0.5 * local_x_tilt)
    x_sin = math.sin(0.5 * local_x_tilt)
    # The fixed base orientation is Ry(local_y_tilt) * Rx(local_x_tilt).
    # It maps the selected original hull-face normal to world down.  A world
    # yaw then randomizes approach direction without changing table support.
    base_w = y_cos * x_cos
    base_x = y_cos * x_sin
    base_y = y_sin * x_cos
    base_z = -y_sin * x_sin
    return np.asarray(
        [
            float(xy[0]),
            float(xy[1]),
            float(
                table_top_z
                + YCB10_MARKER_STABLE_SUPPORT_M
                + YCB10_RESET_SUPPORT_CLEARANCE_M
            ),
            yaw_cos * base_w - yaw_sin * base_z,
            yaw_cos * base_x - yaw_sin * base_y,
            yaw_cos * base_y + yaw_sin * base_x,
            yaw_cos * base_z + yaw_sin * base_w,
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class YCB10PhysicalSample:
    marker_pose: np.ndarray
    palm_pose: np.ndarray
    marker_velocity: np.ndarray
    finger_forces: tuple[float, float, float, float, float]
    table_contact_force: float
    forbidden_contact_force: float
    table_top_z: float
    workspace_ok: bool = True
    robot_safety_ok: bool = True
    contacts_valid: bool = True
    non_distal_contact_force: float = 0.0
    fingertip_positions: np.ndarray | None = None


@dataclass
class YCB10SemanticState:
    phase: str = "table_start"
    elapsed_seconds: float = 0.0
    table_clear_seconds: float = 0.0
    contact_gap_seconds: float = 0.0
    changed_set_candidate_seconds: float = 0.0
    changed_set_candidate_mask: int = 0
    contact_set_change_seen: bool = False
    reference_marker_in_palm: np.ndarray | None = None
    reference_palm_world: np.ndarray | None = None
    reference_contact_mask: int = 0
    final_samples: list[tuple[float, float]] = field(default_factory=list)
    success: bool = False
    failed: bool = False
    first_terminal_reason: str = ""
    last_metrics: dict[str, Any] = field(default_factory=dict)
    support_mask: int = 0
    previous_contact_mask: int = 0
    previous_axial: float = 0.0
    earned_axial: float = 0.0
    transfer_active: bool = False
    transfer_start_axial: float = 0.0
    transfer_support_mask: int = 0
    transfer_released_mask: int = 0
    transfer_joined_mask: int = 0
    transfer_candidate_axial: float = 0.0
    completed_transfers: int = 0
    grip_reference_coordinate: float | None = None
    current_finger_coordinates: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def reset(self) -> None:
        self.grip_reference_coordinate = None
        self.current_finger_coordinates = np.zeros(2)
        self.phase = "table_start"
        self.elapsed_seconds = 0.0
        self.table_clear_seconds = 0.0
        self.contact_gap_seconds = 0.0
        self.changed_set_candidate_seconds = 0.0
        self.changed_set_candidate_mask = 0
        self.contact_set_change_seen = False
        self.reference_marker_in_palm = None
        self.reference_palm_world = None
        self.reference_contact_mask = 0
        self.final_samples.clear()
        self.success = False
        self.failed = False
        self.first_terminal_reason = ""
        self.last_metrics = {}
        self.support_mask = 0
        self.previous_contact_mask = 0
        self.previous_axial = 0.0
        self.earned_axial = 0.0
        self.transfer_active = False
        self.transfer_start_axial = 0.0
        self.transfer_support_mask = 0
        self.transfer_released_mask = 0
        self.transfer_joined_mask = 0
        self.transfer_candidate_axial = 0.0
        self.completed_transfers = 0

    @staticmethod
    def _contact_mask(forces: tuple[float, float, float, float, float]) -> int:
        mask = 0
        for index, force in enumerate(forces):
            if float(force) >= YCB10_CONTACT_FORCE_N:
                mask |= 1 << index
        return mask

    @staticmethod
    def _valid_contact(mask: int) -> bool:
        thumb = bool(mask & 0b00001)
        opposing = bool(mask & 0b11110)
        return bool(thumb and opposing)

    def _set_terminal(self, reason: str, *, success: bool) -> None:
        if self.first_terminal_reason:
            return
        self.first_terminal_reason = str(reason)
        self.success = bool(success)
        self.failed = not bool(success)
        self.phase = "success" if success else "failure"

    def _cancel_transfer(self) -> None:
        self.transfer_active = False
        self.transfer_start_axial = 0.0
        self.transfer_support_mask = 0
        self.transfer_released_mask = 0
        self.transfer_joined_mask = 0
        self.transfer_candidate_axial = 0.0
        self.changed_set_candidate_mask = 0
        self.changed_set_candidate_seconds = 0.0

    def set_timeout(self) -> None:
        """Latch timeout metadata without converting truncation into failure."""
        if self.first_terminal_reason:
            return
        self.first_terminal_reason = "timeout"
        self.success = False
        self.failed = False
        self.last_metrics["ycb10_success_now"] = False
        self.last_metrics["ycb10_failed"] = False
        self.last_metrics["first_terminal_reason"] = "timeout"

    def _kinematic_metrics(
        self,
        marker_matrix: np.ndarray,
        palm_matrix: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        marker_axis_world = marker_matrix[:3, :3] @ YCB10_CAPWARD_LOCAL_AXIS
        inclination = float(
            math.degrees(math.asin(float(np.clip(abs(marker_axis_world[2]), 0.0, 1.0))))
        )
        if self.reference_marker_in_palm is None or self.reference_palm_world is None:
            return 0.0, 0.0, 0.0, 0.0, inclination

        current_marker_in_palm = invert_pose(palm_matrix) @ marker_matrix
        reference = self.reference_marker_in_palm
        reference_axis = reference[:3, :3] @ YCB10_CAPWARD_LOCAL_AXIS
        current_axis = current_marker_in_palm[:3, :3] @ YCB10_CAPWARD_LOCAL_AXIS
        displacement = current_marker_in_palm[:3, 3] - reference[:3, 3]
        axial = float(np.dot(displacement, reference_axis))
        lateral = float(np.linalg.norm(displacement - axial * reference_axis))
        relative_axis_angle = _axis_angle_degrees(reference_axis, current_axis)
        wrist_rotation = rotation_angle_degrees(
            self.reference_palm_world[:3, :3].T @ palm_matrix[:3, :3]
        )
        return axial, lateral, relative_axis_angle, wrist_rotation, inclination

    def update(self, sample: YCB10PhysicalSample, dt: float) -> dict[str, Any]:
        """Measure each new pickup's thumb/index movement on the marker itself."""
        if self.first_terminal_reason:
            return dict(self.last_metrics)
        dt = max(0.0, float(dt))
        self.elapsed_seconds += dt
        marker_pose = np.asarray(sample.marker_pose, dtype=np.float64).reshape(7)
        palm_pose = np.asarray(sample.palm_pose, dtype=np.float64).reshape(7)
        velocity = np.asarray(sample.marker_velocity, dtype=np.float64).reshape(6)
        fingers = np.asarray(sample.fingertip_positions, dtype=np.float64)
        contacts_valid = bool(
            sample.contacts_valid
            and np.all(
                np.isfinite(
                    [
                        *sample.finger_forces,
                        sample.table_contact_force,
                        sample.forbidden_contact_force,
                        sample.non_distal_contact_force,
                    ]
                )
            )
        )
        physical_valid = bool(
            np.all(np.isfinite(marker_pose))
            and np.all(np.isfinite(palm_pose))
            and np.all(np.isfinite(velocity))
            and math.isfinite(sample.table_top_z)
            and np.linalg.norm(marker_pose[3:]) > 1e-12
            and np.linalg.norm(palm_pose[3:]) > 1e-12
        )
        if not contacts_valid or not physical_valid:
            self._set_terminal(
                (
                    "invalid_contact_state"
                    if not contacts_valid
                    else "invalid_physical_state"
                ),
                success=False,
            )
            self.last_metrics.update(
                ycb10_phase=self.phase,
                ycb10_contacts_valid=contacts_valid,
                ycb10_physical_state_valid=physical_valid,
                ycb10_success_now=False,
                ycb10_failed=True,
                first_terminal_reason=self.first_terminal_reason,
                terminal_kind="invalid_physics",
            )
            return dict(self.last_metrics)

        marker = pose_to_matrix(marker_pose)
        palm = pose_to_matrix(palm_pose)
        grasp_points_valid = bool(
            fingers.shape == (2, 3) and np.all(np.isfinite(fingers))
        )
        coordinates = (
            ((fingers - marker[:3, 3]) @ marker[:3, :3]) @ YCB10_CAPWARD_LOCAL_AXIS
            if grasp_points_valid
            else np.zeros(2)
        )
        contact_mask = self._contact_mask(sample.finger_forces)
        pinch = (contact_mask & 0b11) == 0b11
        table_contact = sample.table_contact_force >= YCB10_CONTACT_FORCE_N
        marker_com = marker[:3, 3] + marker[:3, :3] @ YCB10_MARKER_CENTER_OF_MASS_M
        off_table = bool(
            not table_contact
            and marker_com[2] >= sample.table_top_z + YCB10_LIFT_HEIGHT_M
        )
        if table_contact:
            # This resets only the pickup reference, never the environment,
            # action targets, episode clock, recorder, or object state.
            elapsed = self.elapsed_seconds
            self.reset()
            self.elapsed_seconds = elapsed
        self.current_finger_coordinates = coordinates.copy()
        self.table_clear_seconds = (
            self.table_clear_seconds + dt if not table_contact else 0.0
        )
        self.contact_gap_seconds = (
            0.0 if contact_mask else self.contact_gap_seconds + dt
        )
        if (
            not sample.workspace_ok
            or np.any(marker_com < YCB10_WORKSPACE_LOWER)
            or np.any(marker_com > YCB10_WORKSPACE_UPPER)
        ):
            self._set_terminal("workspace_exit_or_drop", success=False)
        elif not sample.robot_safety_ok:
            self._set_terminal("robot_safety_violation", success=False)

        if self.grip_reference_coordinate is None and not self.first_terminal_reason:
            self.phase = "lift_and_grasp" if off_table else "table_start"
            if off_table and pinch and grasp_points_valid:
                self.grip_reference_coordinate = float(np.mean(coordinates))
                self.reference_marker_in_palm = invert_pose(palm) @ marker
                self.reference_palm_world = palm.copy()
                self.reference_contact_mask = contact_mask
                self.phase = "fingertip_gait"
        reference = self.grip_reference_coordinate
        target = 0.0 if reference is None else reference + YCB10_AXIAL_PROGRESS_M
        progress = np.zeros(2) if reference is None else coordinates - reference
        target_reached = bool(
            reference is not None
            and grasp_points_valid
            and np.all(progress + 1e-9 >= YCB10_AXIAL_PROGRESS_M)
        )
        self.earned_axial = max(0.0, float(np.min(progress)))
        self.previous_axial = float(np.min(progress))
        self.contact_set_change_seen = (
            target_reached  # retained legacy observation slot
        )
        if not self.first_terminal_reason and off_table and pinch and target_reached:
            self._set_terminal("success", success=True)
        _, lateral, relative_axis, wrist_rotation, inclination = (
            self._kinematic_metrics(marker, palm)
        )
        self.last_metrics = {
            "ycb10_phase": self.phase,
            "ycb10_reference_latched": reference is not None,
            "ycb10_reference_grip_coordinate_m": (
                0.0 if reference is None else reference
            ),
            "ycb10_target_grip_coordinate_m": target,
            "ycb10_thumb_coordinate_m": float(coordinates[0]),
            "ycb10_index_coordinate_m": float(coordinates[1]),
            "ycb10_thumb_progress_m": float(progress[0]),
            "ycb10_index_progress_m": float(progress[1]),
            "ycb10_target_reached": target_reached,
            "ycb10_marker_off_table": off_table,
            "ycb10_marker_com_height_m": float(marker_com[2]),
            "ycb10_table_clear_seconds": self.table_clear_seconds,
            "ycb10_contact_gap_seconds": self.contact_gap_seconds,
            "ycb10_contact_mask": contact_mask,
            "ycb10_reference_contact_mask": self.reference_contact_mask,
            "ycb10_thumb_contact": bool(contact_mask & 1),
            "ycb10_index_contact": bool(contact_mask & 2),
            "ycb10_opposing_contact_count": int((contact_mask & 0b11110).bit_count()),
            "ycb10_valid_finger_contact": pinch,
            "ycb10_contacts_valid": contacts_valid,
            "ycb10_physical_state_valid": physical_valid,
            "ycb10_grasp_points_valid": grasp_points_valid,
            "ycb10_table_contact": table_contact,
            "ycb10_forbidden_support_contact": sample.forbidden_contact_force
            >= YCB10_CONTACT_FORCE_N,
            "ycb10_non_distal_contact": sample.non_distal_contact_force
            >= YCB10_CONTACT_FORCE_N,
            "ycb10_axial_progress_m": float(np.min(progress)),
            "ycb10_earned_gait_progress_m": self.earned_axial,
            "ycb10_contact_set_change_seen": target_reached,
            "ycb10_lateral_displacement_m": lateral,
            "ycb10_relative_axis_rotation_deg": relative_axis,
            "ycb10_wrist_rotation_deg": wrist_rotation,
            "ycb10_horizontal_inclination_deg": inclination,
            "ycb10_marker_linear_speed_mps": float(np.linalg.norm(velocity[:3])),
            "ycb10_marker_angular_speed_radps": float(np.linalg.norm(velocity[3:])),
            "ycb10_final_hold_seconds": 0.0,
            "ycb10_final_axial_range_m": 0.0,
            "ycb10_success_now": self.success,
            "ycb10_failed": self.failed,
            "first_terminal_reason": self.first_terminal_reason,
        }
        return dict(self.last_metrics)

    def event_flags(self) -> np.ndarray:
        metrics = self.last_metrics
        return np.asarray(
            [
                float(self.reference_marker_in_palm is not None),
                float(self.contact_set_change_seen),
                float(bool(metrics.get("ycb10_marker_off_table", False))),
                float(bool(metrics.get("ycb10_valid_finger_contact", False))),
                float(bool(metrics.get("ycb10_table_contact", False))),
                float(bool(metrics.get("ycb10_forbidden_support_contact", False))),
                float(self.phase == "final_hold"),
                float(self.success),
                float(self.failed),
                float(self.first_terminal_reason == "right_fingertip_contact_lost"),
                float(self.first_terminal_reason == "workspace_exit_or_drop"),
                float(self.first_terminal_reason == "robot_safety_violation"),
            ],
            dtype=np.float64,
        )

    def hidden_vector(self) -> np.ndarray:
        metrics = self.last_metrics
        values = np.asarray(
            [
                YCB10_PHASE_CODES.get(self.phase, -1),
                YCB10_TERMINAL_REASON_CODES.get(self.first_terminal_reason, -1),
                self.elapsed_seconds,
                self.table_clear_seconds,
                self.contact_gap_seconds,
                self.changed_set_candidate_seconds,
                self.reference_contact_mask,
                metrics.get("ycb10_contact_mask", 0),
                metrics.get("ycb10_axial_progress_m", 0.0),
                metrics.get("ycb10_lateral_displacement_m", 0.0),
                metrics.get("ycb10_relative_axis_rotation_deg", 0.0),
                metrics.get("ycb10_wrist_rotation_deg", 0.0),
                metrics.get("ycb10_horizontal_inclination_deg", 0.0),
                metrics.get("ycb10_marker_linear_speed_mps", 0.0),
                metrics.get("ycb10_marker_angular_speed_radps", 0.0),
                metrics.get("ycb10_final_hold_seconds", 0.0),
                metrics.get("ycb10_final_axial_range_m", 0.0),
                float(self.reference_marker_in_palm is not None),
                float(self.contact_set_change_seen),
                float(self.success),
                float(self.failed),
            ],
            dtype=np.float64,
        )
        transfer = np.asarray(
            [
                3,  # Revised pickup-relative physical grasp coordinates.
                self.changed_set_candidate_mask,
                self.support_mask,
                self.previous_contact_mask,
                self.previous_axial,
                self.earned_axial,
                float(self.transfer_active),
                self.transfer_start_axial,
                self.transfer_support_mask,
                self.transfer_released_mask,
                self.transfer_joined_mask,
                self.transfer_candidate_axial,
                self.completed_transfers,
                len(self.final_samples),
                float(metrics.get("ycb10_contacts_valid", False)),
                float(metrics.get("ycb10_physical_state_valid", False)),
                float(metrics.get("ycb10_non_distal_contact", False)),
                float(self.reference_palm_world is not None),
                float(metrics.get("ycb10_marker_off_table", False)),
                float(metrics.get("ycb10_valid_finger_contact", False)),
                float(metrics.get("ycb10_table_contact", False)),
                float(metrics.get("ycb10_forbidden_support_contact", False)),
                float(metrics.get("ycb10_marker_com_height_m", 0.0)),
            ],
            dtype=np.float64,
        )
        references = np.concatenate(
            [
                (np.zeros((4, 4)) if matrix is None else matrix).reshape(-1)
                for matrix in (self.reference_marker_in_palm, self.reference_palm_world)
            ]
        )
        history = np.zeros((YCB10_FINAL_SAMPLE_CAPACITY, 2), dtype=np.float64)
        if self.final_samples:
            history[: len(self.final_samples)] = self.final_samples
        return np.concatenate(
            [
                values,
                transfer,
                references,
                history.reshape(-1),
                [
                    float(self.grip_reference_coordinate is not None),
                    (
                        0.0
                        if self.grip_reference_coordinate is None
                        else self.grip_reference_coordinate
                    ),
                    *self.current_finger_coordinates,
                    float(metrics.get("ycb10_grasp_points_valid", False)),
                ],
            ]
        )

    @classmethod
    def from_hidden_vector(cls, vector: np.ndarray) -> YCB10SemanticState:
        """Restore the complete semantic state without touching physical state."""
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if values.size != YCB10_HIDDEN_STATE_SIZE or not np.all(np.isfinite(values)):
            raise ValueError("Invalid YCB10 semantic-state shape or values")
        if values[21] != 3:
            raise ValueError("Unsupported YCB10 semantic-state schema")
        state = cls()
        state.phase = next(
            name for name, code in YCB10_PHASE_CODES.items() if code == values[0]
        )
        state.first_terminal_reason = next(
            name
            for name, code in YCB10_TERMINAL_REASON_CODES.items()
            if code == values[1]
        )
        (
            state.elapsed_seconds,
            state.table_clear_seconds,
            state.contact_gap_seconds,
            state.changed_set_candidate_seconds,
        ) = values[2:6]
        state.reference_contact_mask = int(values[6])
        metric_names = (
            "ycb10_contact_mask",
            "ycb10_axial_progress_m",
            "ycb10_lateral_displacement_m",
            "ycb10_relative_axis_rotation_deg",
            "ycb10_wrist_rotation_deg",
            "ycb10_horizontal_inclination_deg",
            "ycb10_marker_linear_speed_mps",
            "ycb10_marker_angular_speed_radps",
            "ycb10_final_hold_seconds",
            "ycb10_final_axial_range_m",
        )
        state.last_metrics = dict(zip(metric_names, values[7:17].tolist()))
        state.contact_set_change_seen, state.success, state.failed = map(
            bool, values[18:21]
        )
        state.changed_set_candidate_mask = int(values[22])
        state.support_mask = int(values[23])
        state.previous_contact_mask = int(values[24])
        state.previous_axial, state.earned_axial = values[25:27]
        state.transfer_active = bool(values[27])
        state.transfer_start_axial = float(values[28])
        (
            state.transfer_support_mask,
            state.transfer_released_mask,
            state.transfer_joined_mask,
        ) = map(int, values[29:32])
        state.transfer_candidate_axial = float(values[32])
        state.completed_transfers = int(values[33])
        sample_count = int(values[34])
        if not 0 <= sample_count <= YCB10_FINAL_SAMPLE_CAPACITY:
            raise ValueError("Invalid YCB10 final-hold sample count")
        state.last_metrics.update(
            {
                "ycb10_contacts_valid": bool(values[35]),
                "ycb10_physical_state_valid": bool(values[36]),
                "ycb10_non_distal_contact": bool(values[37]),
                "ycb10_phase": state.phase,
                "ycb10_reference_latched": bool(values[17]),
                "ycb10_reference_contact_mask": state.reference_contact_mask,
                "ycb10_table_clear_seconds": state.table_clear_seconds,
                "ycb10_contact_gap_seconds": state.contact_gap_seconds,
                "ycb10_thumb_contact": bool(int(values[7]) & 1),
                "ycb10_opposing_contact_count": (int(values[7]) & 0b11110).bit_count(),
                "ycb10_contact_set_change_seen": state.contact_set_change_seen,
                "ycb10_earned_gait_progress_m": state.earned_axial,
                "ycb10_transfer_active": state.transfer_active,
                "ycb10_completed_transfers": state.completed_transfers,
                "ycb10_continuous_support_mask": state.transfer_support_mask,
                "ycb10_success_now": state.success,
                "ycb10_failed": state.failed,
                "first_terminal_reason": state.first_terminal_reason,
                "ycb10_marker_off_table": bool(values[39]),
                "ycb10_valid_finger_contact": bool(values[40]),
                "ycb10_table_contact": bool(values[41]),
                "ycb10_forbidden_support_contact": bool(values[42]),
                "ycb10_marker_com_height_m": float(values[43]),
            }
        )
        if state.first_terminal_reason in {
            "invalid_contact_state",
            "invalid_physical_state",
        }:
            state.last_metrics["terminal_kind"] = "invalid_physics"
        if bool(values[17]):
            state.reference_marker_in_palm = values[44:60].reshape(4, 4).copy()
        if bool(values[38]):
            state.reference_palm_world = values[60:76].reshape(4, 4).copy()
        state.final_samples = [
            tuple(row) for row in values[76:-5].reshape(-1, 2)[:sample_count]
        ]
        state.grip_reference_coordinate = float(values[-4]) if values[-5] else None
        state.current_finger_coordinates = values[-3:-1].copy()
        reference = state.grip_reference_coordinate
        progress = (
            np.zeros(2)
            if reference is None
            else state.current_finger_coordinates - reference
        )
        state.last_metrics.update(
            ycb10_reference_grip_coordinate_m=0.0 if reference is None else reference,
            ycb10_target_grip_coordinate_m=(
                0.0 if reference is None else reference + YCB10_AXIAL_PROGRESS_M
            ),
            ycb10_thumb_coordinate_m=float(values[-3]),
            ycb10_index_coordinate_m=float(values[-2]),
            ycb10_grasp_points_valid=bool(values[-1]),
            ycb10_thumb_progress_m=float(progress[0]),
            ycb10_index_progress_m=float(progress[1]),
            ycb10_index_contact=bool(int(values[7]) & 2),
            ycb10_target_reached=bool(
                reference is not None
                and values[-1]
                and np.all(progress + 1e-9 >= YCB10_AXIAL_PROGRESS_M)
            ),
        )
        return state


__all__ = [name for name in globals() if name.startswith("YCB10_")] + [
    "YCB10PhysicalSample",
    "YCB10SemanticState",
    "invert_pose",
    "pose_to_matrix",
    "quaternion_wxyz_to_matrix",
    "rotation_angle_degrees",
    "sample_ycb10_reset",
]
