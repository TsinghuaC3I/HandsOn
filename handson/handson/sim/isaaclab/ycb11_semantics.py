"""Pure geometry and event semantics for YCB11.

This module deliberately has no Isaac/PhysX imports.  The live environment
feeds it physical contacts, rigid-body state, and native deformable nodal
state; tests can therefore audit the task contract without launching Kit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any

import numpy as np


YCB11_TASK_ID = "YCB11"
YCB11_TASK_NAME = "eval_flex01_compress_sponge_into_mug"
YCB11_CONTROL_HZ = 30
# Keep the qualified 120 Hz PhysX step, while allowing the operator stream to
# run at the same 30 Hz cadence as the other bimanual collection tasks.
YCB11_PHYSICS_HZ = 120
# Allow contact errors to resolve faster than the bounded fingers can close.
# The old 0.01 m/s ceiling left overlaps loaded by the position drives.
# Joint speed/drive-error limits independently bound commanded hand motion.
YCB11_HAND_MAX_DEPENETRATION_VELOCITY_MPS = 0.60

# Dimensions after the Sponge 3 glTF unit/axis conversion; YCB mug unchanged. The mug
# origin is the MJCF body origin; its cavity is offset toward +Y by 13.139 mm.
YCB11_SPONGE_SOURCE_SIZE_M = np.asarray(
    [0.07789751, 0.09989101, 0.030000], dtype=np.float64
)
YCB11_MUG_SOURCE_SIZE_M = np.asarray(
    [0.080050, 0.104056, 0.082002], dtype=np.float64
)
YCB11_MUG_CAVITY_CENTER_LOCAL_M = np.asarray(
    [0.0, 0.013139, 0.0], dtype=np.float64
)


class YCB11Stage(IntEnum):
    APPROACH = 0
    MUG_STABILIZED = 1
    SPONGE_GRASPED = 2
    SPONGE_COMPRESSED = 3
    RIM_CROSSED = 4
    PACKED = 5
    RELEASED_STABLE = 6
    SUCCESS = 7


@dataclass(frozen=True)
class YCB11Thresholds:
    """Authoritative, operator-oriented YCB11 thresholds in SI units."""

    mug_contact_hold_s: float = 0.25
    min_sponge_lift_m: float = 0.020
    max_compression_ratio: float = 0.82
    rim_cross_min_inside_fraction: float = 0.15
    packed_inside_fraction: float = 0.85
    packed_below_rim_fraction: float = 0.80
    packed_com_depth_m: float = 0.012
    mug_safe_radius_m: float = 0.030
    mug_safe_bottom_local_z_m: float = -0.0285
    mug_rim_local_z_m: float = 0.040765
    mug_max_success_tilt_deg: float = 15.0
    mug_max_final_displacement_m: float = 0.050
    release_hand_clearance_m: float = 0.0275
    mug_max_stable_linear_speed_mps: float = 0.020
    mug_max_stable_angular_speed_radps: float = 0.20
    sponge_max_stable_rms_speed_mps: float = 0.020
    stable_hold_s: float = 1.0
    failure_mug_tilt_deg: float = 60.0
    failure_mug_tilt_hold_s: float = 0.50
    failure_speed_hold_s: float = 0.25
    failure_invalid_deformation_hold_s: float = 0.20
    failure_max_nodal_speed_mps: float = 6.0
    failure_min_deformation_determinant: float = 0.02
    failure_max_principal_stretch: float = 4.0
    failure_joint_limit_tolerance_rad: float = 0.001
    failure_joint_limit_hold_s: float = 0.25
    failure_joint_limit_hard_excess_rad: float = 0.05


DEFAULT_THRESHOLDS = YCB11Thresholds()


def _finite_points(nodes_world: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes_world, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1:] != (3,) or nodes.shape[0] < 4:
        raise ValueError("YCB11 sponge nodes must have shape (N>=4, 3)")
    if not bool(np.all(np.isfinite(nodes))):
        raise ValueError("YCB11 sponge nodes must be finite")
    return nodes


def quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm < 1e-10:
        raise ValueError("YCB11 mug quaternion is invalid")
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def nodes_in_mug_frame(nodes_world: np.ndarray, mug_pose_wxyz: np.ndarray) -> np.ndarray:
    nodes = _finite_points(nodes_world)
    pose = np.asarray(mug_pose_wxyz, dtype=np.float64).reshape(7)
    if not bool(np.all(np.isfinite(pose))):
        raise ValueError("YCB11 mug pose must be finite")
    rotation = quat_wxyz_to_matrix(pose[3:7])
    return (nodes - pose[:3]) @ rotation


def principal_span(
    nodes_world: np.ndarray, *, rest_nodes: np.ndarray | None = None,
) -> float:
    """Measure the longest span, tracking material axes when rest nodes exist.

    Recomputing PCA after compression can rotate the axes toward a diagonal
    as the sponge becomes square, making a smaller sponge appear larger.
    Corresponding rest nodes let a best-fit rigid rotation remove only motion.
    The task authors the rest pose along the source XYZ axes; subsequent
    randomized yaw and hand rotation do not change those material axes.
    """

    nodes = _finite_points(nodes_world)
    centered = nodes - np.mean(nodes, axis=0)
    if rest_nodes is not None:
        rest = _finite_points(rest_nodes)
        if rest.shape != nodes.shape:
            raise ValueError("YCB11 compression requires corresponding rest nodes")
        reference = rest - np.mean(rest, axis=0)
        u, _values, vt = np.linalg.svd(centered.T @ reference)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vt)
        centered = centered @ (u @ correction @ vt)
        return float(np.max(np.ptp(centered, axis=0)))
    _u, _values, axes_t = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes_t.T
    return float(np.max(np.ptp(projected, axis=0)))


def compression_ratio(
    nodes_world: np.ndarray, rest_principal_span_m: float,
    *, rest_nodes: np.ndarray | None = None,
) -> float:
    rest_span = float(rest_principal_span_m)
    if not math.isfinite(rest_span) or rest_span <= 1e-8:
        raise ValueError("YCB11 rest principal span must be positive and finite")
    return principal_span(nodes_world, rest_nodes=rest_nodes) / rest_span


def mug_tilt_deg(mug_pose_wxyz: np.ndarray) -> float:
    pose = np.asarray(mug_pose_wxyz, dtype=np.float64).reshape(7)
    opening_axis_world = quat_wxyz_to_matrix(pose[3:7])[:, 2]
    alignment = float(np.clip(opening_axis_world[2], -1.0, 1.0))
    return math.degrees(math.acos(alignment))


def containment_metrics(
    nodes_world: np.ndarray,
    mug_pose_wxyz: np.ndarray,
    *,
    thresholds: YCB11Thresholds = DEFAULT_THRESHOLDS,
) -> dict[str, float | bool]:
    """Measure the sponge against an analytic *semantic* mug cavity.

    The cylinder is intentionally smaller than the physical octagonal cavity.
    It is used only for scoring and never authored as collision or visual USD.
    """

    local = nodes_in_mug_frame(nodes_world, mug_pose_wxyz)
    centered_xy = local[:, :2] - YCB11_MUG_CAVITY_CENTER_LOCAL_M[:2]
    radial = np.linalg.norm(centered_xy, axis=1)
    below_rim = local[:, 2] <= thresholds.mug_rim_local_z_m
    above_bottom = local[:, 2] >= thresholds.mug_safe_bottom_local_z_m
    radially_inside = radial <= thresholds.mug_safe_radius_m
    inside = radially_inside & below_rim & above_bottom
    inside_fraction = float(np.mean(inside))
    below_rim_fraction = float(np.mean(below_rim))
    com_local_z = float(np.mean(local[:, 2]))
    com_depth = float(thresholds.mug_rim_local_z_m - com_local_z)
    rim_crossed = bool(
        inside_fraction >= thresholds.rim_cross_min_inside_fraction
        and float(np.min(local[:, 2]))
        <= thresholds.mug_rim_local_z_m - 0.003
    )
    packed = bool(
        inside_fraction >= thresholds.packed_inside_fraction
        and below_rim_fraction >= thresholds.packed_below_rim_fraction
        and com_depth >= thresholds.packed_com_depth_m
    )
    return {
        "inside_fraction": inside_fraction,
        "below_rim_fraction": below_rim_fraction,
        "com_local_z_m": com_local_z,
        "com_depth_below_rim_m": com_depth,
        "min_local_z_m": float(np.min(local[:, 2])),
        "max_local_z_m": float(np.max(local[:, 2])),
        "rim_crossed": rim_crossed,
        "packed": packed,
    }


@dataclass(frozen=True)
class YCB11Signals:
    """Physical facts sampled by the live Isaac environment."""

    left_mug_contact: bool = False
    right_thumb_sponge_contact: bool = False
    right_opposing_sponge_contact: bool = False
    sponge_lift_m: float = 0.0
    compression_ratio: float = math.inf
    rim_crossed: bool = False
    packed: bool = False
    mug_table_contact: bool = False
    mug_tilt_deg: float = math.inf
    mug_displacement_m: float = math.inf
    right_released_sponge: bool = False
    left_released_mug: bool = False
    right_hand_sponge_distance_m: float = 0.0
    left_hand_mug_distance_m: float = 0.0
    right_hand_mug_distance_m: float = 0.0
    left_hand_sponge_distance_m: float = 0.0
    mug_linear_speed_mps: float = math.inf
    mug_angular_speed_radps: float = math.inf
    sponge_rms_speed_mps: float = math.inf
    ejection_detected: bool = False
    mug_out_of_workspace: bool = False
    sponge_out_of_workspace: bool = False
    deformable_state_invalid: bool = False
    max_nodal_speed_mps: float = 0.0
    min_deformation_determinant: float = 1.0
    max_principal_stretch: float = 1.0
    robot_safety_failure: bool = False
    robot_joint_limit_excess_rad: float = 0.0


class YCB11StateMachine:
    """Latched physical event ordering, stability, and failure debounce."""

    def __init__(self, thresholds: YCB11Thresholds = DEFAULT_THRESHOLDS):
        self.thresholds = thresholds
        self.reset()

    def reset(self) -> None:
        self.stage = YCB11Stage.APPROACH
        self.left_contact_s = 0.0
        self.stable_s = 0.0
        self.excessive_tilt_s = 0.0
        self.excessive_speed_s = 0.0
        self.invalid_deformation_s = 0.0
        self.joint_limit_excursion_s = 0.0
        self.failure_reason: str | None = None
        self.event_times_s: dict[str, float] = {}
        self.elapsed_s = 0.0

    @property
    def success(self) -> bool:
        return self.stage == YCB11Stage.SUCCESS

    @property
    def failed(self) -> bool:
        return self.failure_reason is not None

    def _latch(self, stage: YCB11Stage) -> None:
        self.stage = stage
        self.event_times_s[stage.name.lower()] = float(self.elapsed_s)

    def _update_failure(self, signals: YCB11Signals, dt: float) -> None:
        if self.failure_reason is not None:
            return
        if signals.robot_safety_failure:
            self.failure_reason = "robot_safety_failure"
            return
        # Finite contact solver overshoot is recoverable. The recorded 1.2
        # and 1.7 mrad one-frame finger excursions used to reset a grasp.
        excess = signals.robot_joint_limit_excess_rad
        if not math.isfinite(excess) or excess > self.thresholds.failure_joint_limit_hard_excess_rad:
            self.failure_reason = "robot_safety_failure"
            return
        self.joint_limit_excursion_s = (
            self.joint_limit_excursion_s + dt
            if excess > self.thresholds.failure_joint_limit_tolerance_rad else 0.0
        )
        if self.joint_limit_excursion_s + 1e-12 >= self.thresholds.failure_joint_limit_hold_s:
            self.failure_reason = "robot_joint_limit_sustained"
            return
        if signals.deformable_state_invalid:
            self.failure_reason = "invalid_deformable_state"
            return
        if signals.mug_out_of_workspace:
            self.failure_reason = "mug_out_of_workspace"
            return
        if signals.sponge_out_of_workspace:
            self.failure_reason = "sponge_out_of_workspace"
            return
        # Once the sponge has crossed the rim and been packed, losing the
        # required containment is a terminal ejection.  Merely making the
        # stability predicate false would leave the episode in PACKED forever
        # (and eventually report a timeout instead of the physical failure).
        if signals.ejection_detected:
            self.failure_reason = "sponge_ejected"
            return

        self.excessive_tilt_s = (
            self.excessive_tilt_s + dt
            if signals.mug_tilt_deg > self.thresholds.failure_mug_tilt_deg
            else 0.0
        )
        if self.excessive_tilt_s >= self.thresholds.failure_mug_tilt_hold_s:
            self.failure_reason = "mug_tilted_unrecoverably"
            return

        speed_exploded = (
            not math.isfinite(signals.max_nodal_speed_mps)
            or signals.max_nodal_speed_mps
            > self.thresholds.failure_max_nodal_speed_mps
        )
        self.excessive_speed_s = self.excessive_speed_s + dt if speed_exploded else 0.0
        if self.excessive_speed_s >= self.thresholds.failure_speed_hold_s:
            self.failure_reason = "deformable_speed_explosion"
            return

        invalid_deformation = (
            not math.isfinite(signals.min_deformation_determinant)
            or not math.isfinite(signals.max_principal_stretch)
            or signals.min_deformation_determinant
            < self.thresholds.failure_min_deformation_determinant
            or signals.max_principal_stretch
            > self.thresholds.failure_max_principal_stretch
        )
        self.invalid_deformation_s = (
            self.invalid_deformation_s + dt if invalid_deformation else 0.0
        )
        if (
            self.invalid_deformation_s
            >= self.thresholds.failure_invalid_deformation_hold_s
        ):
            self.failure_reason = "invalid_deformable_deformation"

    def update(self, signals: YCB11Signals, dt: float) -> dict[str, Any]:
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("YCB11 state-machine dt must be positive and finite")
        self.elapsed_s += dt
        self._update_failure(signals, dt)
        if self.failed or self.success:
            return self.snapshot()

        self.left_contact_s = (
            self.left_contact_s + dt if signals.left_mug_contact else 0.0
        )
        released = bool(
            signals.right_released_sponge
            and signals.left_released_mug
            and signals.right_hand_sponge_distance_m
            >= self.thresholds.release_hand_clearance_m
            and signals.left_hand_mug_distance_m
            >= self.thresholds.release_hand_clearance_m
            and signals.right_hand_mug_distance_m
            >= self.thresholds.release_hand_clearance_m
            and signals.left_hand_sponge_distance_m
            >= self.thresholds.release_hand_clearance_m
        )
        stable = bool(
            released
            and signals.packed
            and signals.mug_table_contact
            and signals.mug_tilt_deg <= self.thresholds.mug_max_success_tilt_deg
            and signals.mug_displacement_m
            <= self.thresholds.mug_max_final_displacement_m
            and signals.mug_linear_speed_mps
            < self.thresholds.mug_max_stable_linear_speed_mps
            and signals.mug_angular_speed_radps
            < self.thresholds.mug_max_stable_angular_speed_radps
            and signals.sponge_rms_speed_mps
            < self.thresholds.sponge_max_stable_rms_speed_mps
            and not signals.ejection_detected
        )

        mug_held = self.left_contact_s + 1e-12 >= self.thresholds.mug_contact_hold_s
        right_grasp = (
            signals.right_thumb_sponge_contact and signals.right_opposing_sponge_contact
        )
        right_touch = (
            signals.right_thumb_sponge_contact or signals.right_opposing_sponge_contact
        )
        # Advance at most one state per sample.  This prevents a final-pose-only
        # snapshot from fabricating the required history.
        if self.stage == YCB11Stage.APPROACH:
            if (
                self.left_contact_s + 1e-12
                >= self.thresholds.mug_contact_hold_s
            ):
                self._latch(YCB11Stage.MUG_STABILIZED)
        elif self.stage == YCB11Stage.MUG_STABILIZED:
            if (
                mug_held
                and right_grasp
                and signals.sponge_lift_m >= self.thresholds.min_sponge_lift_m
            ):
                self._latch(YCB11Stage.SPONGE_GRASPED)
        elif self.stage == YCB11Stage.SPONGE_GRASPED:
            if (
                mug_held
                and right_grasp
                and signals.compression_ratio <= self.thresholds.max_compression_ratio
            ):
                self._latch(YCB11Stage.SPONGE_COMPRESSED)
        elif self.stage == YCB11Stage.SPONGE_COMPRESSED:
            if mug_held and right_touch and signals.rim_crossed:
                self._latch(YCB11Stage.RIM_CROSSED)
        elif self.stage == YCB11Stage.RIM_CROSSED:
            if mug_held and right_touch and signals.packed:
                self._latch(YCB11Stage.PACKED)
        elif self.stage == YCB11Stage.PACKED:
            if released:
                self.stable_s = dt if stable else 0.0
                self._latch(YCB11Stage.RELEASED_STABLE)
        elif self.stage == YCB11Stage.RELEASED_STABLE:
            self.stable_s = self.stable_s + dt if stable else 0.0
            if self.stable_s + 1e-12 >= self.thresholds.stable_hold_s:
                self._latch(YCB11Stage.SUCCESS)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "stage": int(self.stage),
            "stage_name": self.stage.name.lower(),
            "success": self.success,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "left_contact_s": float(self.left_contact_s),
            "stable_s": float(self.stable_s),
            "joint_limit_excursion_s": float(self.joint_limit_excursion_s),
            "elapsed_s": float(self.elapsed_s),
            "event_times_s": dict(self.event_times_s),
        }


__all__ = [
    "DEFAULT_THRESHOLDS",
    "YCB11_CONTROL_HZ",
    "YCB11_HAND_MAX_DEPENETRATION_VELOCITY_MPS",
    "YCB11_MUG_CAVITY_CENTER_LOCAL_M",
    "YCB11_MUG_SOURCE_SIZE_M",
    "YCB11_SPONGE_SOURCE_SIZE_M",
    "YCB11_PHYSICS_HZ",
    "YCB11_TASK_ID",
    "YCB11_TASK_NAME",
    "YCB11Signals",
    "YCB11Stage",
    "YCB11StateMachine",
    "YCB11Thresholds",
    "compression_ratio",
    "containment_metrics",
    "mug_tilt_deg",
    "nodes_in_mug_frame",
    "principal_span",
    "quat_wxyz_to_matrix",
]
