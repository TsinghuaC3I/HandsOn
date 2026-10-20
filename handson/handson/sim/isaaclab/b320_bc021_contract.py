"""Physical task contract for B320-BC021 two-block face mating.

The module is deliberately independent of Isaac APIs so reset sampling and
success/failure semantics can be exercised before a Kit process is launched.
Final mesh dimensions are supplied by the scene integration boundary rather
than being inferred from object-root distance.

Engineering acceptance thresholds are clean-room calibration data, not task
text.  They remain provisional and formally fail closed until a final asset,
measured geometry, and matching calibration digests are supplied together.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import IntEnum
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np


TASK_NAME = "t01_align_blocks_in_midair"
TASK_ID = "B320-BC021"

CONTACT_KEYS = ("LA", "LB", "RA", "RB", "AB")

# Retained only to decode and compare legacy recordings. New episodes still
# sample reference-pose excursion diagnostics, but this threshold is no longer
# a BC021 success or terminal condition.
MAX_REFERENCE_TRANSLATION_M = 0.020
# Legacy recordings expose a reference pose and excursion counters in their
# hidden state. Keep sampling those values for schema-compatible diagnostics,
# but they are not BC021 success or terminal conditions.
REFERENCE_FLAT_MAX_TILT_DEG = 10.0
RESET_MAX_ATTEMPTS = 64
# BC021 uses a seeded reset sampler so the policy sees meaningful layout
# variation while both blocks remain assigned to comfortable, side-specific
# workspaces.  The 100 mm gap between the comfort radius and the physical arm
# radius prevents a reset from putting either arm at full extension.
RESET_COMMON_X_JITTER_M = 0.025
RESET_INDEPENDENT_X_JITTER_M = 0.020
RESET_Y_JITTER_M = 0.045
RESET_YAW_DEG = 30.0
RESET_ARM_BASE_POSITIONS = {
    "left": (-0.75, 0.065, 1.34),
    "right": (-0.75, -0.065, 1.34),
}
RESET_COMFORT_MIN_RADIUS_M = 0.42
RESET_COMFORT_MAX_RADIUS_M = 0.68

TERMINAL_REASON_CODES = {
    # Reserved for decoding recordings captured before the reference-excursion
    # terminal condition was retired. New episodes never emit this reason.
    "reference_translation_exceeded": 1,
    "sustained_deep_face_penetration": 2,
    "block_dropped_after_airborne": 3,
    "block_left_collection_workspace": 4,
}

# This ordered scalar schema is recorded as current/next task hidden state.
# Optional event frames use -1 rather than 0 so frame zero remains representable.
# ``last_metrics`` is deliberately excluded: it is derived from the public
# physical/contact sample plus the state encoded below.
HIDDEN_OBSERVATION_FIELDS = (
    "stage",
    "assigned_contact_seen",
    "left_airborne_seen",
    "both_airborne_seen",
    "marked_face_contact_seen",
    "reference_pose_valid",
    "reference_pose_x_m",
    "reference_pose_y_m",
    "reference_pose_z_m",
    "reference_pose_qw",
    "reference_pose_qx",
    "reference_pose_qy",
    "reference_pose_qz",
    "max_reference_translation_m",
    "stable_substeps",
    "reference_limit_frames",
    "deep_penetration_frames",
    "drop_frames",
    "workspace_frames",
    "first_assigned_contact_frame",
    "first_left_airborne_frame",
    "first_both_airborne_frame",
    "reference_latched_frame",
    "first_marked_contact_frame",
    "stable_started_frame",
    "first_success_frame",
    "first_terminal_frame",
    "success_latched",
    "terminal_latched",
    "terminal_reason_code",
)
HIDDEN_OBSERVATION_SHAPE = (len(HIDDEN_OBSERVATION_FIELDS),)
HIDDEN_EVENT_FRAME_SENTINEL = -1.0


class BC021Stage(IntEnum):
    RESET = 0
    ASSIGNED_CONTACT = 1
    BOTH_AIRBORNE = 2
    REFERENCE_LATCHED = 3
    MARKED_FACE_CONTACT = 4
    ALIGNED = 5
    FAILED = 6


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quaternion / norm


def _safe_pose(value: Any) -> tuple[np.ndarray, bool]:
    """Return finite calculation data plus a separate fail-closed truth bit."""

    try:
        raw = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        raw = np.zeros(0, dtype=np.float64)
    readable = bool(
        raw.shape == (7,)
        and np.all(np.isfinite(raw))
        and float(np.linalg.norm(raw[3:7])) >= 1.0e-8
    )
    if not readable:
        return np.asarray(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        ), False
    result = raw.copy()
    result[3:7] = _normalize_quaternion(result[3:7])
    return result, True


def _safe_velocity(value: Any) -> tuple[np.ndarray, bool]:
    try:
        raw = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        raw = np.zeros(0, dtype=np.float64)
    readable = bool(raw.shape == (6,) and np.all(np.isfinite(raw)))
    return (
        raw.copy() if readable else np.zeros(6, dtype=np.float64),
        readable,
    )


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = _normalize_quaternion(lhs)
    rw, rx, ry, rz = _normalize_quaternion(rhs)
    return _normalize_quaternion(
        np.asarray(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float64,
        )
    )


def _yaw_quaternion(degrees: float) -> np.ndarray:
    half = math.radians(float(degrees)) * 0.5
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)])


def rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(quaternion_wxyz)
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _angle_degrees(lhs: np.ndarray, rhs: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(lhs) * np.linalg.norm(rhs)),
        1.0e-12,
    )
    cosine = float(np.clip(np.dot(lhs, rhs) / denominator, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


@dataclass(frozen=True)
class BC021Geometry:
    """Geometry measured at the final imported rigid-body frames."""

    left_half_extents_m: tuple[float, float, float]
    right_half_extents_m: tuple[float, float, float]
    left_marked_normal_local: tuple[float, float, float] = (1.0, 0.0, 0.0)
    right_marked_normal_local: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    left_marked_center_local_m: tuple[float, float, float] | None = None
    right_marked_center_local_m: tuple[float, float, float] | None = None
    left_marked_tangent_u_local: tuple[float, float, float] = (0.0, 1.0, 0.0)
    left_marked_tangent_v_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    right_marked_tangent_u_local: tuple[float, float, float] = (0.0, -1.0, 0.0)
    right_marked_tangent_v_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    left_marked_patch_half_extents_m: tuple[float, float] = (0.027, 0.027)
    right_marked_patch_half_extents_m: tuple[float, float] = (0.027, 0.027)
    contact_plane_tolerance_m: float = 0.004
    table_top_z_m: float = 0.89
    provisional: bool = True

    def half_extents(self, side: str) -> np.ndarray:
        values = (
            self.left_half_extents_m
            if side == "left"
            else self.right_half_extents_m
        )
        result = np.asarray(values, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError(f"Invalid BC021 {side} half extents: {values!r}")
        return result

    def marked_normal_local(self, side: str) -> np.ndarray:
        values = (
            self.left_marked_normal_local
            if side == "left"
            else self.right_marked_normal_local
        )
        normal = np.asarray(values, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or norm < 1.0e-12:
            raise ValueError(f"Invalid BC021 {side} marked-face normal")
        return normal / norm

    def marked_center_local(self, side: str) -> np.ndarray:
        authored = (
            self.left_marked_center_local_m
            if side == "left"
            else self.right_marked_center_local_m
        )
        if authored is not None:
            center = np.asarray(authored, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(center)):
                raise ValueError(f"Invalid BC021 {side} marked-face center")
            return center
        normal = self.marked_normal_local(side)
        dominant_axis = int(np.argmax(np.abs(normal)))
        center = np.zeros(3, dtype=np.float64)
        center[dominant_axis] = math.copysign(
            self.half_extents(side)[dominant_axis],
            normal[dominant_axis],
        )
        return center

    def marked_tangents_local(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        raw_u = (
            self.left_marked_tangent_u_local
            if side == "left"
            else self.right_marked_tangent_u_local
        )
        raw_v = (
            self.left_marked_tangent_v_local
            if side == "left"
            else self.right_marked_tangent_v_local
        )
        normal = self.marked_normal_local(side)
        tangent_u = np.asarray(raw_u, dtype=np.float64).reshape(3)
        tangent_v = np.asarray(raw_v, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(tangent_u)) or not np.all(
            np.isfinite(tangent_v)
        ):
            raise ValueError(f"Invalid BC021 {side} marked-face tangents")
        tangent_u_norm = float(np.linalg.norm(tangent_u))
        tangent_v_norm = float(np.linalg.norm(tangent_v))
        if tangent_u_norm < 1.0e-12 or tangent_v_norm < 1.0e-12:
            raise ValueError(f"Degenerate BC021 {side} marked-face tangents")
        tangent_u = tangent_u / tangent_u_norm
        tangent_v = tangent_v / tangent_v_norm
        if (
            abs(float(np.dot(normal, tangent_u))) > 1.0e-6
            or abs(float(np.dot(normal, tangent_v))) > 1.0e-6
            or abs(float(np.dot(tangent_u, tangent_v))) > 1.0e-6
            or float(np.dot(np.cross(tangent_u, tangent_v), normal)) < 0.999999
        ):
            raise ValueError(
                f"BC021 {side} marked-face frame is not right-handed orthonormal"
            )
        return tangent_u, tangent_v

    def marked_patch_half_extents(self, side: str) -> np.ndarray:
        raw = (
            self.left_marked_patch_half_extents_m
            if side == "left"
            else self.right_marked_patch_half_extents_m
        )
        extents = np.asarray(raw, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(extents)) or np.any(extents <= 0.0):
            raise ValueError(f"Invalid BC021 {side} marked patch extents")
        return extents

    def validate(self) -> None:
        """Reject an asset contract whose semantic frame is ambiguous."""

        for side in ("left", "right"):
            self.half_extents(side)
            self.marked_center_local(side)
            self.marked_normal_local(side)
            self.marked_tangents_local(side)
            self.marked_patch_half_extents(side)
        if not math.isfinite(self.contact_plane_tolerance_m) or (
            self.contact_plane_tolerance_m <= 0.0
        ):
            raise ValueError("Invalid BC021 contact-plane tolerance")
        if not math.isfinite(self.table_top_z_m):
            raise ValueError("Invalid BC021 table height")

    def digest_payload(self) -> dict[str, Any]:
        """Return canonical measured geometry semantics for hash binding.

        ``provisional`` is deliberately excluded.  It is an evidence status,
        not geometry: promoting the exact same measured frames to final must
        not change their identity, while the acceptance calibration separately
        requires both final status bits.
        """

        self.validate()
        sides: dict[str, dict[str, Any]] = {}
        for side in ("left", "right"):
            tangent_u, tangent_v = self.marked_tangents_local(side)
            sides[side] = {
                "half_extents_m": self.half_extents(side).tolist(),
                "marked_center_local_m": self.marked_center_local(
                    side
                ).tolist(),
                "marked_normal_local": self.marked_normal_local(side).tolist(),
                "marked_tangent_u_local": tangent_u.tolist(),
                "marked_tangent_v_local": tangent_v.tolist(),
                "marked_patch_half_extents_m": (
                    self.marked_patch_half_extents(side).tolist()
                ),
            }
        return {
            "schema": "handson.bc021.geometry.v1",
            "sides": sides,
            "contact_plane_tolerance_m": float(
                self.contact_plane_tolerance_m
            ),
            "table_top_z_m": float(self.table_top_z_m),
        }

    def digest_sha256(self) -> str:
        encoded = json.dumps(
            self.digest_payload(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


PROVISIONAL_GEOMETRY = BC021Geometry(
    left_half_extents_m=(0.0375, 0.0375, 0.0375),
    right_half_extents_m=(0.0375, 0.0375, 0.0375),
)

# Measured from the selected Old Fashioned Wooden Blocks glTF after the
# Isaac Sim 5.1 conversion and uniform normalization.  Both BC021 bodies use
# the broad square source mesh so the strict reset-origin gate has a stable
# planar support footprint.  The marked faces are the opposing +/-X faces and
# their patches remain inset from the measured visible bounds.
FINAL_GEOMETRY = BC021Geometry(
    left_half_extents_m=(0.0375, 0.03744508789019451, 0.03749999851617847),
    right_half_extents_m=(0.0375, 0.03744508789019451, 0.03749999851617847),
    left_marked_patch_half_extents_m=(0.026, 0.026),
    right_marked_patch_half_extents_m=(0.026, 0.026),
    provisional=False,
)


PROVISIONAL_SCAFFOLD_ASSET_DIGEST_SHA256 = hashlib.sha256(
    b"HandsOn|B320-BC021|procedural-two-75mm-cube-scaffold|v1"
).hexdigest()


@dataclass(frozen=True)
class BC021AcceptanceCalibration:
    """Versioned, hash-bound engineering acceptance envelope.

    ``MAX_REFERENCE_TRANSLATION_M`` is retained outside this class only for
    schema-compatible legacy diagnostics. It is not an active success or
    terminal condition. Every physical-success or terminal-guard threshold in
    the current task is an explicitly provisional or final calibration decision
    with a stated basis and exact asset/geometry identities.
    """

    basis: str
    version: str
    provisional: bool
    final: bool
    asset_digest_sha256: str
    geometry_digest_sha256: str
    minimum_bottom_clearance_m: float
    max_normal_error_deg: float
    max_lateral_error_m: float
    min_axial_gap_m: float
    max_axial_gap_m: float
    max_relative_linear_speed_m_s: float
    max_relative_angular_speed_rad_s: float
    max_block_linear_speed_m_s: float
    max_block_angular_speed_rad_s: float
    physics_hz: int
    stable_hold_seconds: float
    stable_hold_substeps: int
    assigned_hand_contact_min_n: float
    cross_hand_contact_max_n: float
    block_contact_min_n: float
    marked_contact_point_min_n: float
    reference_limit_debounce_frames: int
    deep_penetration_m: float
    deep_penetration_max_normal_error_deg: float
    deep_penetration_max_lateral_error_m: float
    deep_penetration_debounce_frames: int
    drop_clearance_m: float
    drop_debounce_frames: int
    workspace_x_min_m: float
    workspace_x_max_m: float
    workspace_abs_y_max_m: float
    workspace_z_max_m: float
    workspace_debounce_frames: int

    def __post_init__(self) -> None:
        self.validate()

    @staticmethod
    def _validate_sha256(name: str, value: str) -> None:
        valid = bool(
            isinstance(value, str)
            and len(value) == 64
            and value == value.lower()
            and all(character in "0123456789abcdef" for character in value)
        )
        if not valid:
            raise ValueError(f"Invalid BC021 {name} SHA-256 digest")

    def validate(self) -> None:
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise ValueError("BC021 acceptance calibration basis is required")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("BC021 acceptance calibration version is required")
        if type(self.provisional) is not bool or type(self.final) is not bool:
            raise ValueError(
                "BC021 acceptance calibration status flags must be bool"
            )
        if bool(self.provisional) == bool(self.final):
            raise ValueError(
                "BC021 acceptance calibration must be exactly one of "
                "provisional or final"
            )
        self._validate_sha256("asset", self.asset_digest_sha256)
        self._validate_sha256("geometry", self.geometry_digest_sha256)
        if self.final and (
            self.asset_digest_sha256
            == PROVISIONAL_SCAFFOLD_ASSET_DIGEST_SHA256
        ):
            raise ValueError(
                "BC021 final calibration cannot bind the provisional scaffold"
            )

        finite_values = {
            "minimum bottom clearance": self.minimum_bottom_clearance_m,
            "maximum normal error": self.max_normal_error_deg,
            "maximum lateral error": self.max_lateral_error_m,
            "minimum axial gap": self.min_axial_gap_m,
            "maximum axial gap": self.max_axial_gap_m,
            "maximum relative linear speed": (
                self.max_relative_linear_speed_m_s
            ),
            "maximum relative angular speed": (
                self.max_relative_angular_speed_rad_s
            ),
            "maximum block linear speed": self.max_block_linear_speed_m_s,
            "maximum block angular speed": self.max_block_angular_speed_rad_s,
            "stable hold seconds": self.stable_hold_seconds,
            "assigned-hand contact minimum": self.assigned_hand_contact_min_n,
            "cross-hand contact maximum": self.cross_hand_contact_max_n,
            "block-contact minimum": self.block_contact_min_n,
            "marked-contact point minimum": self.marked_contact_point_min_n,
            "deep penetration": self.deep_penetration_m,
            "deep-penetration normal error": (
                self.deep_penetration_max_normal_error_deg
            ),
            "deep-penetration lateral error": (
                self.deep_penetration_max_lateral_error_m
            ),
            "drop clearance": self.drop_clearance_m,
            "workspace x minimum": self.workspace_x_min_m,
            "workspace x maximum": self.workspace_x_max_m,
            "workspace absolute y maximum": self.workspace_abs_y_max_m,
            "workspace z maximum": self.workspace_z_max_m,
        }
        for name, value in finite_values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Invalid BC021 {name}")
        positive_values = {
            "maximum normal error": self.max_normal_error_deg,
            "maximum lateral error": self.max_lateral_error_m,
            "maximum relative linear speed": (
                self.max_relative_linear_speed_m_s
            ),
            "maximum relative angular speed": (
                self.max_relative_angular_speed_rad_s
            ),
            "maximum block linear speed": self.max_block_linear_speed_m_s,
            "maximum block angular speed": self.max_block_angular_speed_rad_s,
            "stable hold seconds": self.stable_hold_seconds,
            "assigned-hand contact minimum": self.assigned_hand_contact_min_n,
            "block-contact minimum": self.block_contact_min_n,
            "marked-contact point minimum": self.marked_contact_point_min_n,
            "deep penetration": self.deep_penetration_m,
            "deep-penetration normal error": (
                self.deep_penetration_max_normal_error_deg
            ),
            "deep-penetration lateral error": (
                self.deep_penetration_max_lateral_error_m
            ),
            "workspace absolute y maximum": self.workspace_abs_y_max_m,
            "workspace z maximum": self.workspace_z_max_m,
        }
        for name, value in positive_values.items():
            if float(value) <= 0.0:
                raise ValueError(f"Invalid BC021 {name}")
        if self.minimum_bottom_clearance_m < 0.0:
            raise ValueError("Invalid BC021 minimum bottom clearance")
        if not 0.0 < self.max_normal_error_deg < 180.0:
            raise ValueError("Invalid BC021 maximum normal error")
        if self.min_axial_gap_m >= self.max_axial_gap_m:
            raise ValueError("Invalid BC021 axial-gap interval")
        if self.cross_hand_contact_max_n < 0.0:
            raise ValueError("Invalid BC021 cross-hand contact maximum")
        if self.marked_contact_point_min_n > self.block_contact_min_n:
            raise ValueError(
                "BC021 marked-point minimum exceeds block-contact minimum"
            )
        if self.workspace_x_min_m >= self.workspace_x_max_m:
            raise ValueError("Invalid BC021 workspace x interval")

        integer_values = {
            "physics Hz": self.physics_hz,
            "stable hold substeps": self.stable_hold_substeps,
            "reference-limit debounce": self.reference_limit_debounce_frames,
            "deep-penetration debounce": (
                self.deep_penetration_debounce_frames
            ),
            "drop debounce": self.drop_debounce_frames,
            "workspace debounce": self.workspace_debounce_frames,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"Invalid BC021 {name}")
        expected_substeps = int(
            round(float(self.physics_hz) * self.stable_hold_seconds)
        )
        if self.stable_hold_substeps != expected_substeps or not math.isclose(
            self.stable_hold_seconds,
            self.stable_hold_substeps / float(self.physics_hz),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "BC021 stable hold seconds/substeps/physics Hz disagree"
            )

    def geometry_is_compatible(self, geometry: BC021Geometry) -> bool:
        if not isinstance(geometry, BC021Geometry):
            return False
        try:
            return self.geometry_digest_sha256 == geometry.digest_sha256()
        except (TypeError, ValueError):
            return False

    def formal_for_geometry(self, geometry: BC021Geometry) -> bool:
        return bool(
            isinstance(geometry, BC021Geometry)
            and self.final
            and not self.provisional
            and not geometry.provisional
            and self.geometry_is_compatible(geometry)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
        }

    def digest_sha256(self) -> str:
        payload = {
            "schema": "handson.bc021.acceptance_calibration.v1",
            **self.as_dict(),
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


PROVISIONAL_ACCEPTANCE_CALIBRATION = BC021AcceptanceCalibration(
    basis=(
        "clean-room derivation from the BC021 task text, 120 Hz physical "
        "contact semantics, and the provisional 75 mm cube scaffold; not "
        "measured from a final selected asset"
    ),
    version="bc021-clean-room-provisional-v1",
    provisional=True,
    final=False,
    asset_digest_sha256=PROVISIONAL_SCAFFOLD_ASSET_DIGEST_SHA256,
    geometry_digest_sha256=PROVISIONAL_GEOMETRY.digest_sha256(),
    minimum_bottom_clearance_m=0.080,
    max_normal_error_deg=5.0,
    max_lateral_error_m=0.005,
    min_axial_gap_m=-0.002,
    max_axial_gap_m=0.003,
    max_relative_linear_speed_m_s=0.040,
    max_relative_angular_speed_rad_s=0.50,
    max_block_linear_speed_m_s=0.055,
    max_block_angular_speed_rad_s=0.60,
    physics_hz=120,
    stable_hold_seconds=0.40,
    stable_hold_substeps=48,
    assigned_hand_contact_min_n=0.10,
    cross_hand_contact_max_n=0.05,
    block_contact_min_n=0.02,
    marked_contact_point_min_n=0.005,
    reference_limit_debounce_frames=2,
    deep_penetration_m=0.010,
    deep_penetration_max_normal_error_deg=15.0,
    deep_penetration_max_lateral_error_m=0.045,
    deep_penetration_debounce_frames=3,
    drop_clearance_m=-0.015,
    drop_debounce_frames=3,
    workspace_x_min_m=-0.62,
    workspace_x_max_m=0.27,
    workspace_abs_y_max_m=0.64,
    workspace_z_max_m=1.60,
    workspace_debounce_frames=3,
)

# Compatibility names remain importable for the current isolated runtime, but
# every value is derived from the explicit calibration above.  Contract logic
# below reads the calibration object directly rather than these aliases.
MIN_BOTTOM_CLEARANCE_M = (
    PROVISIONAL_ACCEPTANCE_CALIBRATION.minimum_bottom_clearance_m
)
MAX_NORMAL_ERROR_DEG = PROVISIONAL_ACCEPTANCE_CALIBRATION.max_normal_error_deg
MAX_LATERAL_ERROR_M = PROVISIONAL_ACCEPTANCE_CALIBRATION.max_lateral_error_m
MIN_AXIAL_GAP_M = PROVISIONAL_ACCEPTANCE_CALIBRATION.min_axial_gap_m
MAX_AXIAL_GAP_M = PROVISIONAL_ACCEPTANCE_CALIBRATION.max_axial_gap_m
MAX_RELATIVE_LINEAR_SPEED_M_S = (
    PROVISIONAL_ACCEPTANCE_CALIBRATION.max_relative_linear_speed_m_s
)
MAX_RELATIVE_ANGULAR_SPEED_RAD_S = (
    PROVISIONAL_ACCEPTANCE_CALIBRATION.max_relative_angular_speed_rad_s
)
MAX_BLOCK_LINEAR_SPEED_M_S = (
    PROVISIONAL_ACCEPTANCE_CALIBRATION.max_block_linear_speed_m_s
)
MAX_BLOCK_ANGULAR_SPEED_RAD_S = (
    PROVISIONAL_ACCEPTANCE_CALIBRATION.max_block_angular_speed_rad_s
)
PHYSICS_HZ = PROVISIONAL_ACCEPTANCE_CALIBRATION.physics_hz
STABLE_HOLD_SECONDS = PROVISIONAL_ACCEPTANCE_CALIBRATION.stable_hold_seconds
STABLE_HOLD_SUBSTEPS = PROVISIONAL_ACCEPTANCE_CALIBRATION.stable_hold_substeps
ASSIGNED_HAND_CONTACT_MIN_N = (
    PROVISIONAL_ACCEPTANCE_CALIBRATION.assigned_hand_contact_min_n
)
CROSS_HAND_CONTACT_MAX_N = (
    PROVISIONAL_ACCEPTANCE_CALIBRATION.cross_hand_contact_max_n
)
BLOCK_CONTACT_MIN_N = PROVISIONAL_ACCEPTANCE_CALIBRATION.block_contact_min_n
REFERENCE_LIMIT_DEBOUNCE_FRAMES = (
    PROVISIONAL_ACCEPTANCE_CALIBRATION.reference_limit_debounce_frames
)
DEEP_PENETRATION_M = PROVISIONAL_ACCEPTANCE_CALIBRATION.deep_penetration_m
DEEP_PENETRATION_MAX_NORMAL_ERROR_DEG = (
    PROVISIONAL_ACCEPTANCE_CALIBRATION.deep_penetration_max_normal_error_deg
)
DEEP_PENETRATION_MAX_LATERAL_ERROR_M = (
    PROVISIONAL_ACCEPTANCE_CALIBRATION.deep_penetration_max_lateral_error_m
)
DEEP_PENETRATION_DEBOUNCE_FRAMES = (
    PROVISIONAL_ACCEPTANCE_CALIBRATION.deep_penetration_debounce_frames
)
DROP_CLEARANCE_M = PROVISIONAL_ACCEPTANCE_CALIBRATION.drop_clearance_m
DROP_DEBOUNCE_FRAMES = PROVISIONAL_ACCEPTANCE_CALIBRATION.drop_debounce_frames
WORKSPACE_DEBOUNCE_FRAMES = (
    PROVISIONAL_ACCEPTANCE_CALIBRATION.workspace_debounce_frames
)


@dataclass(frozen=True)
class BC021ResetSample:
    left_pose_wxyz: np.ndarray
    right_pose_wxyz: np.ndarray
    attempts: int
    fallback_used: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "left_pose_wxyz": self.left_pose_wxyz.tolist(),
            "right_pose_wxyz": self.right_pose_wxyz.tolist(),
            "attempts": int(self.attempts),
            "fallback_used": bool(self.fallback_used),
        }


def _projected_half_extent(
    pose_wxyz: np.ndarray,
    half_extents_m: np.ndarray,
    world_axis: int,
) -> float:
    rotation = rotation_matrix(np.asarray(pose_wxyz, dtype=np.float64)[3:7])
    return float(np.abs(rotation[world_axis, :]) @ half_extents_m)


def _reset_pose_is_valid(
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    geometry: BC021Geometry,
    *,
    left_hand_home: np.ndarray,
    right_hand_home: np.ndarray,
    hand_clearance_points_w: np.ndarray | None = None,
) -> bool:
    table_min = np.asarray([-0.57, -0.64], dtype=np.float64)
    table_max = np.asarray([0.27, 0.64], dtype=np.float64)
    recording_min = np.asarray([-0.55, -0.55], dtype=np.float64)
    recording_max = np.asarray([0.20, 0.55], dtype=np.float64)
    arm_bases = {
        side: np.asarray(position, dtype=np.float64)
        for side, position in RESET_ARM_BASE_POSITIONS.items()
    }
    for side, pose, hand_home in (
        ("left", left_pose, left_hand_home),
        ("right", right_pose, right_hand_home),
    ):
        half = geometry.half_extents(side)
        bottom_z = pose[2] - _projected_half_extent(pose, half, 2)
        if abs(bottom_z - geometry.table_top_z_m) > 0.003:
            return False
        if side == "left" and pose[1] <= 0.06:
            return False
        if side == "right" and pose[1] >= -0.06:
            return False
        footprint = np.asarray(
            [
                _projected_half_extent(pose, half, 0),
                _projected_half_extent(pose, half, 1),
            ]
        )
        if np.any(pose[:2] - footprint < table_min + 0.020):
            return False
        if np.any(pose[:2] + footprint > table_max - 0.020):
            return False
        if np.any(pose[:2] - footprint < recording_min):
            return False
        if np.any(pose[:2] + footprint > recording_max):
            return False
        arm_radius = float(np.linalg.norm(pose[:3] - arm_bases[side]))
        if not (
            RESET_COMFORT_MIN_RADIUS_M
            <= arm_radius
            <= RESET_COMFORT_MAX_RADIUS_M
        ):
            return False
        if np.linalg.norm(pose[:3] - hand_home[:3]) < 0.16:
            return False
        if hand_clearance_points_w is not None:
            points = np.asarray(
                hand_clearance_points_w,
                dtype=np.float64,
            ).reshape(-1, 3)
            if not np.all(np.isfinite(points)):
                return False
            conservative_radius = float(np.linalg.norm(half)) + 0.055
            if len(points) and np.any(
                np.linalg.norm(points - pose[:3], axis=1)
                < conservative_radius
            ):
                return False

    left_radius = float(np.linalg.norm(geometry.half_extents("left")))
    right_radius = float(np.linalg.norm(geometry.half_extents("right")))
    return bool(
        np.linalg.norm(left_pose[:3] - right_pose[:3])
        >= left_radius + right_radius + 0.10
    )


def sample_bc021_reset(
    random_state: Any,
    *,
    left_home_pose_wxyz: np.ndarray,
    right_home_pose_wxyz: np.ndarray,
    geometry: BC021Geometry,
    randomize: bool,
    left_hand_home_wxyz: np.ndarray,
    right_hand_home_wxyz: np.ndarray,
    hand_clearance_points_w: np.ndarray | None = None,
    max_attempts: int = RESET_MAX_ATTEMPTS,
) -> BC021ResetSample:
    """Sample a supported, assigned-side, collision-clear reset pair."""

    left_home = np.asarray(left_home_pose_wxyz, dtype=np.float64).reshape(7)
    right_home = np.asarray(right_home_pose_wxyz, dtype=np.float64).reshape(7)
    left_hand = np.asarray(left_hand_home_wxyz, dtype=np.float64).reshape(7)
    right_hand = np.asarray(right_hand_home_wxyz, dtype=np.float64).reshape(7)
    if not randomize:
        if not _reset_pose_is_valid(
            left_home,
            right_home,
            geometry,
            left_hand_home=left_hand,
            right_hand_home=right_hand,
            hand_clearance_points_w=hand_clearance_points_w,
        ):
            raise RuntimeError(
                "BC021 authored non-random reset is not physically valid"
            )
        return BC021ResetSample(left_home.copy(), right_home.copy(), 0, False)

    for attempt in range(1, max(1, int(max_attempts)) + 1):
        left = left_home.copy()
        right = right_home.copy()
        common_x = float(
            random_state.uniform(
                -RESET_COMMON_X_JITTER_M,
                RESET_COMMON_X_JITTER_M,
            )
        )
        left[0] += common_x + float(
            random_state.uniform(
                -RESET_INDEPENDENT_X_JITTER_M,
                RESET_INDEPENDENT_X_JITTER_M,
            )
        )
        right[0] += common_x + float(
            random_state.uniform(
                -RESET_INDEPENDENT_X_JITTER_M,
                RESET_INDEPENDENT_X_JITTER_M,
            )
        )
        left[1] += float(
            random_state.uniform(-RESET_Y_JITTER_M, RESET_Y_JITTER_M)
        )
        right[1] += float(
            random_state.uniform(-RESET_Y_JITTER_M, RESET_Y_JITTER_M)
        )
        left[3:7] = _quat_multiply(
            _yaw_quaternion(
                float(random_state.uniform(-RESET_YAW_DEG, RESET_YAW_DEG))
            ),
            left_home[3:7],
        )
        right[3:7] = _quat_multiply(
            _yaw_quaternion(
                float(random_state.uniform(-RESET_YAW_DEG, RESET_YAW_DEG))
            ),
            right_home[3:7],
        )
        if _reset_pose_is_valid(
            left,
            right,
            geometry,
            left_hand_home=left_hand,
            right_hand_home=right_hand,
            hand_clearance_points_w=hand_clearance_points_w,
        ):
            return BC021ResetSample(left, right, attempt, False)

    if not _reset_pose_is_valid(
        left_home,
        right_home,
        geometry,
        left_hand_home=left_hand,
        right_hand_home=right_hand,
        hand_clearance_points_w=hand_clearance_points_w,
    ):
        raise RuntimeError("BC021 authored fallback reset is not physically valid")
    return BC021ResetSample(
        left_home.copy(),
        right_home.copy(),
        max(1, int(max_attempts)),
        True,
    )


@dataclass(frozen=True)
class BC021Frame:
    left_pose_wxyz: np.ndarray
    right_pose_wxyz: np.ndarray
    left_velocity: np.ndarray
    right_velocity: np.ndarray
    contact_force_n: Mapping[str, float]
    contact_readable: Mapping[str, bool]
    block_contact_points_w: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )
    block_contact_point_forces_n: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    block_contact_points_readable: bool = False
    transient_marked_contact_seen: bool = False
    pose_truth_readable: bool = True
    velocity_truth_readable: bool = True


def _marked_face_data(
    frame: BC021Frame,
    geometry: BC021Geometry,
    calibration: BC021AcceptanceCalibration,
) -> dict[str, Any]:
    left_pose, _left_pose_finite = _safe_pose(frame.left_pose_wxyz)
    right_pose, _right_pose_finite = _safe_pose(frame.right_pose_wxyz)
    geometry.validate()
    left_rotation = rotation_matrix(left_pose[3:7])
    right_rotation = rotation_matrix(right_pose[3:7])
    left_normal_local = geometry.marked_normal_local("left")
    right_normal_local = geometry.marked_normal_local("right")
    left_normal = left_rotation @ left_normal_local
    right_normal = right_rotation @ right_normal_local
    left_center = left_pose[:3] + left_rotation @ geometry.marked_center_local(
        "left"
    )
    right_center = right_pose[:3] + right_rotation @ geometry.marked_center_local(
        "right"
    )
    center_delta = right_center - left_center
    axial_gap = float(np.dot(center_delta, left_normal))
    lateral = center_delta - axial_gap * left_normal
    normal_error = _angle_degrees(left_normal, -right_normal)

    points = np.asarray(frame.block_contact_points_w, dtype=np.float64).reshape(
        -1, 3
    )
    patch_contact = False
    non_target_contact = False
    point_forces = np.abs(
        np.asarray(
            frame.block_contact_point_forces_n,
            dtype=np.float64,
        ).reshape(-1)
    )
    if (
        frame.block_contact_points_readable
        and len(points)
        and len(point_forces) == len(points)
        and np.all(np.isfinite(points))
        and np.all(np.isfinite(point_forces))
    ):
        left_local = (points - left_pose[:3]) @ left_rotation
        right_local = (points - right_pose[:3]) @ right_rotation
        left_center_local = geometry.marked_center_local("left")
        right_center_local = geometry.marked_center_local("right")
        left_tangent_u, left_tangent_v = geometry.marked_tangents_local(
            "left"
        )
        right_tangent_u, right_tangent_v = geometry.marked_tangents_local(
            "right"
        )
        left_patch_u, left_patch_v = geometry.marked_patch_half_extents(
            "left"
        )
        right_patch_u, right_patch_v = geometry.marked_patch_half_extents(
            "right"
        )
        tolerance = float(geometry.contact_plane_tolerance_m)
        left_delta = left_local - left_center_local
        right_delta = right_local - right_center_local
        left_mask = (
            (np.abs(left_delta @ left_normal_local) <= tolerance)
            & (np.abs(left_delta @ left_tangent_u) <= left_patch_u)
            & (np.abs(left_delta @ left_tangent_v) <= left_patch_v)
        )
        right_mask = (
            (np.abs(right_delta @ right_normal_local) <= tolerance)
            & (np.abs(right_delta @ right_tangent_u) <= right_patch_u)
            & (np.abs(right_delta @ right_tangent_v) <= right_patch_v)
        )
        in_patch = left_mask & right_mask
        # The detailed PhysX contact rows are already physical contact truth.
        # Do not reject a valid resting manifold merely because its solver
        # impulse is zero or below an arbitrary force threshold on one substep.
        patch_contact = bool(np.any(in_patch))
        non_target_contact = bool(np.any(~in_patch))

    return {
        "left_face_center_w": left_center,
        "right_face_center_w": right_center,
        "left_face_normal_w": left_normal,
        "right_face_normal_w": right_normal,
        "normal_error_deg": normal_error,
        "axial_gap_m": axial_gap,
        "lateral_error_m": float(np.linalg.norm(lateral)),
        "marked_patch_contact": patch_contact,
        "non_target_contact_point": non_target_contact,
    }


@dataclass
class BC021State:
    stage: BC021Stage = BC021Stage.RESET
    assigned_contact_seen: bool = False
    left_airborne_seen: bool = False
    both_airborne_seen: bool = False
    reference_pose_wxyz: np.ndarray | None = None
    max_reference_translation_m: float = 0.0
    marked_face_contact_seen: bool = False
    stable_substeps: int = 0
    reference_limit_frames: int = 0
    deep_penetration_frames: int = 0
    drop_frames: int = 0
    workspace_frames: int = 0
    first_assigned_contact_frame: int | None = None
    first_left_airborne_frame: int | None = None
    first_both_airborne_frame: int | None = None
    reference_latched_frame: int | None = None
    first_marked_contact_frame: int | None = None
    stable_started_frame: int | None = None
    first_success_frame: int | None = None
    first_terminal_reason: str | None = None
    first_terminal_frame: int | None = None
    last_metrics: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.stage = BC021Stage.RESET
        self.assigned_contact_seen = False
        self.left_airborne_seen = False
        self.both_airborne_seen = False
        self.reference_pose_wxyz = None
        self.max_reference_translation_m = 0.0
        self.marked_face_contact_seen = False
        self.stable_substeps = 0
        self.reference_limit_frames = 0
        self.deep_penetration_frames = 0
        self.drop_frames = 0
        self.workspace_frames = 0
        self.first_assigned_contact_frame = None
        self.first_left_airborne_frame = None
        self.first_both_airborne_frame = None
        self.reference_latched_frame = None
        self.first_marked_contact_frame = None
        self.stable_started_frame = None
        self.first_success_frame = None
        self.first_terminal_reason = None
        self.first_terminal_frame = None
        self.last_metrics = {}

    def _latch_terminal(self, reason: str, frame_index: int) -> None:
        if (
            self.first_success_frame is None
            and self.first_terminal_reason is None
        ):
            self.first_terminal_reason = str(reason)
            self.first_terminal_frame = int(frame_index)
            self.stage = BC021Stage.FAILED

    def evaluate(
        self,
        frame: BC021Frame,
        geometry: BC021Geometry,
        *,
        calibration: BC021AcceptanceCalibration = (
            PROVISIONAL_ACCEPTANCE_CALIBRATION
        ),
        frame_index: int,
        update_latches: bool,
    ) -> tuple[dict[str, Any], tuple[bool, ...]]:
        if not isinstance(calibration, BC021AcceptanceCalibration):
            raise TypeError("BC021 acceptance calibration is required")
        calibration.validate()
        left_pose, left_pose_finite = _safe_pose(frame.left_pose_wxyz)
        right_pose, right_pose_finite = _safe_pose(frame.right_pose_wxyz)
        left_velocity, left_velocity_finite = _safe_velocity(
            frame.left_velocity
        )
        right_velocity, right_velocity_finite = _safe_velocity(
            frame.right_velocity
        )
        pose_truth_ready = bool(
            frame.pose_truth_readable
            and left_pose_finite
            and right_pose_finite
        )
        velocity_truth_ready = bool(
            frame.velocity_truth_readable
            and left_velocity_finite
            and right_velocity_finite
        )
        face = _marked_face_data(frame, geometry, calibration)
        calibration_geometry_compatible = (
            calibration.geometry_is_compatible(geometry)
        )
        calibration_formal_ready = calibration.formal_for_geometry(geometry)

        left_bottom = (
            left_pose[2]
            - _projected_half_extent(
                left_pose,
                geometry.half_extents("left"),
                2,
            )
            - geometry.table_top_z_m
        )
        right_bottom = (
            right_pose[2]
            - _projected_half_extent(
                right_pose,
                geometry.half_extents("right"),
                2,
            )
            - geometry.table_top_z_m
        )
        minimum_bottom_clearance = float(min(left_bottom, right_bottom))
        left_airborne = bool(
            pose_truth_ready
            and left_bottom >= calibration.minimum_bottom_clearance_m
        )
        both_airborne = bool(
            pose_truth_ready
            and minimum_bottom_clearance
            >= calibration.minimum_bottom_clearance_m
        )
        left_up_axis_world = rotation_matrix(left_pose[3:7]) @ np.asarray(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        left_flatness_error_deg = _angle_degrees(
            left_up_axis_world,
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        )
        left_flat = bool(
            pose_truth_ready
            and left_flatness_error_deg <= REFERENCE_FLAT_MAX_TILT_DEG
        )

        forces = {
            key: float(frame.contact_force_n.get(key, 0.0))
            for key in CONTACT_KEYS
        }
        readable = {
            key: bool(frame.contact_readable.get(key, False))
            and math.isfinite(forces[key])
            for key in CONTACT_KEYS
        }
        point_values = np.zeros((0, 3), dtype=np.float64)
        point_forces = np.zeros(0, dtype=np.float64)
        try:
            point_values = np.asarray(
                frame.block_contact_points_w,
                dtype=np.float64,
            ).reshape(-1, 3)
            point_forces = np.asarray(
                frame.block_contact_point_forces_n,
                dtype=np.float64,
            ).reshape(-1)
            detailed_contact_truth = bool(
                frame.block_contact_points_readable
                and len(point_values) == len(point_forces)
                and np.all(np.isfinite(point_values))
                and np.all(np.isfinite(point_forces))
            )
        except Exception:
            detailed_contact_truth = False
        contact_truth_ready = bool(
            all(readable.values()) and detailed_contact_truth
        )
        left_assigned_current = bool(
            readable["LA"]
            and forces["LA"] >= calibration.assigned_hand_contact_min_n
        )
        right_assigned_current = bool(
            readable["RB"]
            and forces["RB"] >= calibration.assigned_hand_contact_min_n
        )
        assigned_current = bool(
            left_assigned_current and right_assigned_current
        )
        cross_clear = bool(
            readable["LB"]
            and readable["RA"]
            and forces["LB"] < calibration.cross_hand_contact_max_n
            and forces["RA"] < calibration.cross_hand_contact_max_n
        )
        block_contact_current = bool(
            readable["AB"]
            and detailed_contact_truth
            and len(point_values) > 0
        )
        target_face_contact_geometry = bool(
            pose_truth_ready
            and not geometry.provisional
            and face["normal_error_deg"] < calibration.max_normal_error_deg
            and face["lateral_error_m"] < calibration.max_lateral_error_m
            and calibration.min_axial_gap_m
            <= face["axial_gap_m"]
            <= calibration.max_axial_gap_m
        )
        marked_contact_current = bool(
            contact_truth_ready
            and block_contact_current
            and target_face_contact_geometry
        )
        relative_linear_speed = float(
            np.linalg.norm(left_velocity[:3] - right_velocity[:3])
        )
        relative_angular_speed = float(
            np.linalg.norm(left_velocity[3:] - right_velocity[3:])
        )
        motion_stable = bool(
            velocity_truth_ready
            and relative_linear_speed
            <= calibration.max_relative_linear_speed_m_s
            and relative_angular_speed
            <= calibration.max_relative_angular_speed_rad_s
            and max(
                float(np.linalg.norm(left_velocity[:3])),
                float(np.linalg.norm(right_velocity[:3])),
            )
            <= calibration.max_block_linear_speed_m_s
            and max(
                float(np.linalg.norm(left_velocity[3:])),
                float(np.linalg.norm(right_velocity[3:])),
            )
            <= calibration.max_block_angular_speed_rad_s
        )
        outside_workspace_pose = bool(
            pose_truth_ready
            and any(
                pose[0] < calibration.workspace_x_min_m
                or pose[0] > calibration.workspace_x_max_m
                or abs(pose[1]) > calibration.workspace_abs_y_max_m
                or pose[2] > calibration.workspace_z_max_m
                for pose in (left_pose, right_pose)
            )
        )
        outside_workspace = bool(
            self.both_airborne_seen and outside_workspace_pose
        )

        if update_latches:
            if contact_truth_ready and assigned_current and cross_clear:
                self.assigned_contact_seen = True
                self.stage = max(self.stage, BC021Stage.ASSIGNED_CONTACT)
                if self.first_assigned_contact_frame is None:
                    self.first_assigned_contact_frame = int(frame_index)
            if (
                left_airborne
                and contact_truth_ready
                and left_assigned_current
                and cross_clear
            ):
                self.left_airborne_seen = True
                if self.first_left_airborne_frame is None:
                    self.first_left_airborne_frame = int(frame_index)
            if self.assigned_contact_seen and both_airborne:
                self.both_airborne_seen = True
                self.stage = max(self.stage, BC021Stage.BOTH_AIRBORNE)
                if self.first_both_airborne_frame is None:
                    self.first_both_airborne_frame = int(frame_index)
            if (
                left_airborne
                and contact_truth_ready
                and left_assigned_current
                and cross_clear
                and left_flat
                and self.reference_pose_wxyz is None
            ):
                self.reference_pose_wxyz = left_pose.copy()
                self.reference_latched_frame = int(frame_index)
                self.stage = max(self.stage, BC021Stage.REFERENCE_LATCHED)
            if pose_truth_ready and self.reference_pose_wxyz is not None:
                translation = float(
                    np.linalg.norm(
                        left_pose[:3] - self.reference_pose_wxyz[:3]
                    )
                )
                self.max_reference_translation_m = max(
                    self.max_reference_translation_m,
                    translation,
                )
            marked_contact_event = bool(
                self.both_airborne_seen
                and contact_truth_ready
                and (
                    marked_contact_current
                    or frame.transient_marked_contact_seen
                )
            )
            if marked_contact_event:
                self.marked_face_contact_seen = True
                self.stage = max(self.stage, BC021Stage.MARKED_FACE_CONTACT)
                if self.first_marked_contact_frame is None:
                    self.first_marked_contact_frame = int(frame_index)

            reference_exceeded = bool(
                self.reference_pose_wxyz is not None
                and self.max_reference_translation_m
                >= MAX_REFERENCE_TRANSLATION_M
            )
            self.reference_limit_frames = (
                self.reference_limit_frames + 1 if reference_exceeded else 0
            )
            deep_penetration = bool(
                pose_truth_ready
                and contact_truth_ready
                and block_contact_current
                and face["normal_error_deg"]
                <= calibration.deep_penetration_max_normal_error_deg
                and face["lateral_error_m"]
                <= calibration.deep_penetration_max_lateral_error_m
                and face["axial_gap_m"]
                <= -calibration.deep_penetration_m
            )
            self.deep_penetration_frames = (
                self.deep_penetration_frames + 1 if deep_penetration else 0
            )
            dropped = bool(
                pose_truth_ready
                and self.both_airborne_seen
                and minimum_bottom_clearance
                <= calibration.drop_clearance_m
            )
            self.drop_frames = self.drop_frames + 1 if dropped else 0
            # ``both_airborne_seen`` may have latched on this same sample, so
            # arm the workspace counter only after the event updates above.
            outside_workspace = bool(
                self.both_airborne_seen and outside_workspace_pose
            )
            self.workspace_frames = (
                self.workspace_frames + 1 if outside_workspace else 0
            )
            if (
                self.deep_penetration_frames
                >= calibration.deep_penetration_debounce_frames
            ):
                self._latch_terminal(
                    "sustained_deep_face_penetration",
                    frame_index,
                )
            elif self.drop_frames >= calibration.drop_debounce_frames:
                self._latch_terminal(
                    "block_dropped_after_airborne",
                    frame_index,
                )
            elif (
                self.workspace_frames
                >= calibration.workspace_debounce_frames
            ):
                self._latch_terminal(
                    "block_left_collection_workspace",
                    frame_index,
                )

        reference_translation = (
            float(np.linalg.norm(left_pose[:3] - self.reference_pose_wxyz[:3]))
            if self.reference_pose_wxyz is not None
            else 1.0
        )
        roles_ok = bool(
            contact_truth_ready
            and self.assigned_contact_seen
            and assigned_current
            and cross_clear
        )
        geometry_ok = target_face_contact_geometry
        instantaneous_mating_ok = bool(
            roles_ok
            and self.both_airborne_seen
            and both_airborne
            and geometry_ok
            and marked_contact_current
            and self.first_terminal_reason is None
        )
        if update_latches:
            if instantaneous_mating_ok:
                if self.stable_substeps == 0:
                    self.stable_started_frame = int(frame_index)
                self.stable_substeps = min(
                    self.stable_substeps + 1,
                    calibration.stable_hold_substeps,
                )
            else:
                self.stable_substeps = 0
                self.stable_started_frame = None
            if (
                self.stable_substeps >= calibration.stable_hold_substeps
                and self.first_success_frame is None
            ):
                self.first_success_frame = int(frame_index)
        physical_mating_ok = bool(
            instantaneous_mating_ok
            and self.stable_substeps >= calibration.stable_hold_substeps
        )
        success_latched = bool(
            self.first_success_frame is not None
            and calibration_formal_ready
        )
        flags = (
            (True, True, True, True)
            if success_latched
            else (
                roles_ok,
                bool(self.both_airborne_seen and both_airborne),
                geometry_ok,
                physical_mating_ok,
            )
        )
        if update_latches and all(flags):
            self.stage = max(self.stage, BC021Stage.ALIGNED)

        metrics = {
            "left_bottom_clearance_m": float(left_bottom),
            "right_bottom_clearance_m": float(right_bottom),
            "minimum_bottom_clearance_m": minimum_bottom_clearance,
            "normal_error_deg": float(face["normal_error_deg"]),
            "lateral_error_m": float(face["lateral_error_m"]),
            "axial_gap_m": float(face["axial_gap_m"]),
            "reference_translation_m": reference_translation,
            "max_reference_translation_m": float(
                self.max_reference_translation_m
            ),
            "left_flat": left_flat,
            "left_flatness_error_deg": float(left_flatness_error_deg),
            "relative_linear_speed_m_s": relative_linear_speed,
            "relative_angular_speed_rad_s": relative_angular_speed,
            "contact_truth_ready": contact_truth_ready,
            "pose_truth_readable": pose_truth_ready,
            "velocity_truth_readable": velocity_truth_ready,
            "geometry_contract_final": bool(not geometry.provisional),
            "geometry_digest_sha256": geometry.digest_sha256(),
            "acceptance_calibration_basis": calibration.basis,
            "acceptance_calibration_version": calibration.version,
            "acceptance_calibration_provisional": bool(
                calibration.provisional
            ),
            "acceptance_calibration_final": bool(calibration.final),
            "acceptance_calibration_asset_digest_sha256": (
                calibration.asset_digest_sha256
            ),
            "acceptance_calibration_geometry_digest_sha256": (
                calibration.geometry_digest_sha256
            ),
            "acceptance_calibration_digest_sha256": (
                calibration.digest_sha256()
            ),
            "acceptance_calibration_geometry_compatible": bool(
                calibration_geometry_compatible
            ),
            "acceptance_calibration_formal_ready": bool(
                calibration_formal_ready
            ),
            "assigned_contact_current": assigned_current,
            "left_assigned_contact_current": left_assigned_current,
            "right_assigned_contact_current": right_assigned_current,
            "cross_contact_clear": cross_clear,
            "block_contact_current": block_contact_current,
            "marked_face_contact_current": marked_contact_current,
            "marked_patch_contact_point_current": bool(
                face["marked_patch_contact"]
            ),
            "contact_point_outside_marked_patch": bool(
                face["non_target_contact_point"]
            ),
            "target_face_contact_geometry": target_face_contact_geometry,
            "motion_stable": motion_stable,
            "outside_collection_workspace": outside_workspace,
            "stable_substeps": int(self.stable_substeps),
            "stable_substeps_required": calibration.stable_hold_substeps,
            "stable_hold_seconds": self.stable_substeps
            / float(calibration.physics_hz),
            "reference_limit_frames": int(self.reference_limit_frames),
            "deep_penetration_frames": int(self.deep_penetration_frames),
            "drop_frames": int(self.drop_frames),
            "workspace_frames": int(self.workspace_frames),
            "workspace_debounce_frames_required": (
                calibration.workspace_debounce_frames
            ),
            "non_target_block_contact": bool(
                block_contact_current and not target_face_contact_geometry
            ),
            "stage": int(self.stage),
            "reference_latched": self.reference_pose_wxyz is not None,
            "reference_latched_frame": self.reference_latched_frame,
            "first_left_airborne_frame": self.first_left_airborne_frame,
            "first_both_airborne_frame": self.first_both_airborne_frame,
            "first_marked_contact_frame": self.first_marked_contact_frame,
            "first_success_frame": self.first_success_frame,
            "terminal_reason": self.first_terminal_reason,
            "success_now": bool(all(flags)),
            "left_face_center_w": face["left_face_center_w"],
            "right_face_center_w": face["right_face_center_w"],
            "left_face_normal_w": face["left_face_normal_w"],
            "right_face_normal_w": face["right_face_normal_w"],
        }
        self.last_metrics = dict(metrics)
        return metrics, flags

    def stage_observation(
        self,
        calibration: BC021AcceptanceCalibration = (
            PROVISIONAL_ACCEPTANCE_CALIBRATION
        ),
    ) -> np.ndarray:
        """Fixed-shape ordered history for recording and replay comparison."""

        return np.asarray(
            [
                float(self.stage),
                float(self.assigned_contact_seen),
                float(self.left_airborne_seen),
                float(self.both_airborne_seen),
                float(self.reference_pose_wxyz is not None),
                float(self.marked_face_contact_seen),
                float(self.stable_substeps),
                float(
                    self.stable_substeps
                    >= calibration.stable_hold_substeps
                ),
                float(self.max_reference_translation_m),
                float(self.first_terminal_reason is not None),
                float(self.first_success_frame or 0),
                float(self.first_terminal_frame or 0),
            ],
            dtype=np.float64,
        )

    def terminal_observation(self) -> np.ndarray:
        reason = self.first_terminal_reason
        return np.asarray(
            [
                float(reason is not None),
                float(TERMINAL_REASON_CODES.get(reason or "", 0)),
                float(self.first_terminal_frame or 0),
                float(self.stage),
            ],
            dtype=np.float64,
        )

    def hidden_observation(self) -> np.ndarray:
        """Return complete fixed-cardinality state for replay comparison.

        This captures every non-physical latch, counter, reference anchor,
        event frame, and immutable outcome owned by ``BC021State``.  Contact,
        pose, and velocity samples remain in their public physical fields.
        """

        reference_pose, reference_pose_valid = _safe_pose(
            self.reference_pose_wxyz
        )

        def event_frame(value: int | None) -> float:
            return (
                HIDDEN_EVENT_FRAME_SENTINEL
                if value is None
                else float(value)
            )

        observation = np.asarray(
            [
                float(self.stage),
                float(self.assigned_contact_seen),
                float(self.left_airborne_seen),
                float(self.both_airborne_seen),
                float(self.marked_face_contact_seen),
                float(reference_pose_valid),
                *reference_pose.tolist(),
                float(self.max_reference_translation_m),
                float(self.stable_substeps),
                float(self.reference_limit_frames),
                float(self.deep_penetration_frames),
                float(self.drop_frames),
                float(self.workspace_frames),
                event_frame(self.first_assigned_contact_frame),
                event_frame(self.first_left_airborne_frame),
                event_frame(self.first_both_airborne_frame),
                event_frame(self.reference_latched_frame),
                event_frame(self.first_marked_contact_frame),
                event_frame(self.stable_started_frame),
                event_frame(self.first_success_frame),
                event_frame(self.first_terminal_frame),
                float(self.first_success_frame is not None),
                float(self.first_terminal_reason is not None),
                float(
                    TERMINAL_REASON_CODES.get(
                        self.first_terminal_reason or "",
                        0,
                    )
                ),
            ],
            dtype=np.float64,
        )
        if observation.shape != HIDDEN_OBSERVATION_SHAPE:
            raise RuntimeError(
                "BC021 hidden observation schema drifted: "
                f"expected {HIDDEN_OBSERVATION_SHAPE}, got "
                f"{observation.shape}"
            )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("BC021 hidden observation is non-finite")
        return observation


__all__ = [
    "ASSIGNED_HAND_CONTACT_MIN_N",
    "BC021AcceptanceCalibration",
    "BC021Frame",
    "BC021Geometry",
    "BC021ResetSample",
    "BC021Stage",
    "BC021State",
    "CONTACT_KEYS",
    "CROSS_HAND_CONTACT_MAX_N",
    "HIDDEN_EVENT_FRAME_SENTINEL",
    "HIDDEN_OBSERVATION_FIELDS",
    "HIDDEN_OBSERVATION_SHAPE",
    "MAX_REFERENCE_TRANSLATION_M",
    "MIN_BOTTOM_CLEARANCE_M",
    "PHYSICS_HZ",
    "PROVISIONAL_ACCEPTANCE_CALIBRATION",
    "PROVISIONAL_GEOMETRY",
    "FINAL_GEOMETRY",
    "PROVISIONAL_SCAFFOLD_ASSET_DIGEST_SHA256",
    "RESET_COMMON_X_JITTER_M",
    "RESET_INDEPENDENT_X_JITTER_M",
    "RESET_Y_JITTER_M",
    "RESET_YAW_DEG",
    "REFERENCE_FLAT_MAX_TILT_DEG",
    "STABLE_HOLD_SECONDS",
    "STABLE_HOLD_SUBSTEPS",
    "TASK_ID",
    "TASK_NAME",
    "TERMINAL_REASON_CODES",
    "WORKSPACE_DEBOUNCE_FRAMES",
    "rotation_matrix",
    "sample_bc021_reset",
]
