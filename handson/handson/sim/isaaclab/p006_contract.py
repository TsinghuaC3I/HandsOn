"""CPU-safe task contract for P0-06 side-release buckle alignment and press."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import math

import numpy as np


TASK_ID = "P0-06"
TASK_NAME = "p006_side_release_buckle_align_press"
TASK_PROMPT = (
    "Use the right hand to hold the blue female buckle by its rear loop. "
    "Use the left hand to grasp the orange male buckle, align all three "
    "tongues with the opening, and press it straight in until the side teeth "
    "latch. Stop after the aligned press; no pull test is required."
)

ASSET_DIR_RELATIVE_PATH = (
    "taskdev/tasks/p006_side_release_buckle_align_press/assets/side_release_buckle_cc0"
)
SOURCE_STL_SHA256 = "f68abfbfac56f746fef1f411ec68d511d4cb567cfb0f10472cb8e26168d57b23"
FEMALE_ASSET_SHA256 = "ac104be962dc5c83bf43173893e6db4f36be8a5b514e7159030c7958415b56ed"
MALE_ASSET_SHA256 = "04d0e655310a9ba23ca25edc75b1a1d9963eebe779bc286be7c1bf192b2dc065"

FEMALE_ROOT_PATH = "/World/Task/P006FemaleBuckle"
MALE_ROOT_PATH = "/World/Task/P006MaleBuckle"
MALE_TONGUE_PATHS = tuple(
    f"{MALE_ROOT_PATH}/{name}" for name in ("OuterTongueNeg", "OuterTonguePos")
)
# Extra grip is confined to the rear housing/loop, never the insertion rails.
GRIP_STATIC_FRICTION = 1.10
GRIP_DYNAMIC_FRICTION = 0.90
GRIP_COLLIDER_PATHS = (f"{FEMALE_ROOT_PATH}/Collision_02",) + tuple(
    f"{MALE_ROOT_PATH}/Collision_{i:02d}" for i in range(4)
)
TABLE_TOP_Z_M = 0.89
CONTROL_HZ = 30
PHYSICS_HZ = 120
TIMEOUT_STEPS = 720
RENDER_RESOLUTION_HW = (480, 640)

# The CC0 source is enlarged uniformly by the asset converter from 78x38x8 mm
# to a 156x76x16 mm two-piece kit.  Individual halves remain comfortably
# graspable by Wuji Gen1 without becoming oversized for the dual-arm workspace.
NOMINAL_CENTER_X_M = -0.18
NOMINAL_FEMALE_Y_M = -0.115
NOMINAL_MALE_Y_M = 0.135
NOMINAL_ROOT_Z_M = TABLE_TOP_Z_M + 0.002
NOMINAL_YAW_DEG = 90.0
ROOT_X_JITTER_M = 0.012
ROOT_Y_JITTER_M = 0.010
ROOT_YAW_JITTER_DEG = 4.0

# Geometric thresholds are derived from the converted source: the female mouth
# lies 34 mm from its root and the male tongue tip 53.6 mm from its root.
FIRST_CONTACT_SEPARATION_M = 0.0876
MIN_INSERTION_DEPTH_M = 0.018
MAX_FINAL_AXIAL_SEPARATION_M = FIRST_CONTACT_SEPARATION_M - MIN_INSERTION_DEPTH_M
MAX_TRANSVERSE_OFFSET_M = 0.015
MAX_AXIS_ERROR_DEG = 12.0
MIN_BODY_CONTACT_FORCE_N = 0.10
MIN_ROLE_CONTACT_FORCE_N = 0.10
LATCH_DWELL_STEPS = 5
SUCCESS_HOLD_STEPS = 12
MAX_RELATIVE_LINEAR_SPEED_MPS = 0.12
MAX_RELATIVE_ANGULAR_SPEED_RADPS = 1.0
MAX_PENETRATION_M = 0.005


class BucklePhase(IntEnum):
    SEPARATED = 0
    ROLE_GRASPED = 1
    ALIGNED = 2
    PRESSED_LATCHED = 3


@dataclass(frozen=True)
class P006ResetState:
    seed: int
    female_position_m: tuple[float, float, float]
    male_position_m: tuple[float, float, float]
    female_yaw_deg: float
    male_yaw_deg: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sample_reset_state(seed: int) -> P006ResetState:
    rng = np.random.default_rng(int(seed))
    shared_x = NOMINAL_CENTER_X_M + float(
        rng.uniform(-ROOT_X_JITTER_M, ROOT_X_JITTER_M)
    )
    female_x = shared_x + float(rng.uniform(-0.006, 0.006))
    male_x = shared_x + float(rng.uniform(-0.006, 0.006))
    female_y = NOMINAL_FEMALE_Y_M + float(
        rng.uniform(-ROOT_Y_JITTER_M, ROOT_Y_JITTER_M)
    )
    male_y = NOMINAL_MALE_Y_M + float(
        rng.uniform(-ROOT_Y_JITTER_M, ROOT_Y_JITTER_M)
    )
    female_yaw = NOMINAL_YAW_DEG + float(
        rng.uniform(-ROOT_YAW_JITTER_DEG, ROOT_YAW_JITTER_DEG)
    )
    male_yaw = NOMINAL_YAW_DEG + float(
        rng.uniform(-ROOT_YAW_JITTER_DEG, ROOT_YAW_JITTER_DEG)
    )
    return P006ResetState(
        seed=int(seed),
        female_position_m=(female_x, female_y, NOMINAL_ROOT_Z_M),
        male_position_m=(male_x, male_y, NOMINAL_ROOT_Z_M),
        female_yaw_deg=female_yaw,
        male_yaw_deg=male_yaw,
    )


@dataclass(frozen=True)
class P006PhysicalReadback:
    female_position_m: tuple[float, float, float]
    female_quaternion_wxyz: tuple[float, float, float, float]
    male_position_m: tuple[float, float, float]
    male_quaternion_wxyz: tuple[float, float, float, float]
    female_linear_velocity_mps: tuple[float, float, float]
    female_angular_velocity_radps: tuple[float, float, float]
    male_linear_velocity_mps: tuple[float, float, float]
    male_angular_velocity_radps: tuple[float, float, float]
    right_female_contact_force_n: float
    left_male_contact_force_n: float
    buckle_body_contact_force_n: float
    robot_table_contact_force_n: float
    max_penetration_m: float = 0.0


def _unit_quaternion(values) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quaternion / norm


def _x_axis_wxyz(quaternion) -> np.ndarray:
    w, x, y, z = _unit_quaternion(quaternion)
    return np.asarray(
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)],
        dtype=np.float64,
    )


def _all_finite(readback: P006PhysicalReadback) -> bool:
    values = []
    for field_name in (
        "female_position_m",
        "female_quaternion_wxyz",
        "male_position_m",
        "male_quaternion_wxyz",
        "female_linear_velocity_mps",
        "female_angular_velocity_radps",
        "male_linear_velocity_mps",
        "male_angular_velocity_radps",
    ):
        values.extend(getattr(readback, field_name))
    values.extend(
        [
            readback.right_female_contact_force_n,
            readback.left_male_contact_force_n,
            readback.buckle_body_contact_force_n,
            readback.robot_table_contact_force_n,
            readback.max_penetration_m,
        ]
    )
    return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def physical_metrics(readback: P006PhysicalReadback) -> dict[str, float | bool]:
    female_position = np.asarray(readback.female_position_m, dtype=np.float64)
    male_position = np.asarray(readback.male_position_m, dtype=np.float64)
    female_axis = _x_axis_wxyz(readback.female_quaternion_wxyz)
    male_axis = _x_axis_wxyz(readback.male_quaternion_wxyz)
    relative = male_position - female_position
    axial_separation = float(np.dot(relative, female_axis))
    transverse = relative - axial_separation * female_axis
    transverse_offset = float(np.linalg.norm(transverse))
    axis_dot = float(np.clip(np.dot(female_axis, male_axis), -1.0, 1.0))
    axis_error_deg = math.degrees(math.acos(axis_dot))
    insertion_depth = max(0.0, FIRST_CONTACT_SEPARATION_M - axial_separation)
    relative_linear_speed = float(
        np.linalg.norm(
            np.asarray(readback.male_linear_velocity_mps)
            - np.asarray(readback.female_linear_velocity_mps)
        )
    )
    relative_angular_speed = float(
        np.linalg.norm(
            np.asarray(readback.male_angular_velocity_radps)
            - np.asarray(readback.female_angular_velocity_radps)
        )
    )
    return {
        "finite": _all_finite(readback),
        "axial_separation_m": axial_separation,
        "insertion_depth_m": insertion_depth,
        "transverse_offset_m": transverse_offset,
        "axis_error_deg": axis_error_deg,
        "relative_linear_speed_mps": relative_linear_speed,
        "relative_angular_speed_radps": relative_angular_speed,
        "right_female_role_contact": (
            readback.right_female_contact_force_n >= MIN_ROLE_CONTACT_FORCE_N
        ),
        "left_male_role_contact": (
            readback.left_male_contact_force_n >= MIN_ROLE_CONTACT_FORCE_N
        ),
        "body_contact": (
            readback.buckle_body_contact_force_n >= MIN_BODY_CONTACT_FORCE_N
        ),
        "aligned": (
            transverse_offset <= MAX_TRANSVERSE_OFFSET_M
            and axis_error_deg <= MAX_AXIS_ERROR_DEG
        ),
        "deep_enough": (
            insertion_depth >= MIN_INSERTION_DEPTH_M
            and axial_separation <= MAX_FINAL_AXIAL_SEPARATION_M
        ),
    }


class P006LatchStateMachine:
    """Ordered, contact-gated alignment/press state machine with no pull stage."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.phase = BucklePhase.SEPARATED
        self.latch_dwell_steps = 0
        self.success_hold_steps = 0
        self.right_female_contact_seen = False
        self.left_male_contact_seen = False
        self.aligned_seen = False
        self.latched_seen = False

    def update(self, readback: P006PhysicalReadback) -> dict[str, float | bool | int | str]:
        metrics = physical_metrics(readback)
        correct_roles = bool(
            metrics["right_female_role_contact"]
            and metrics["left_male_role_contact"]
        )
        self.right_female_contact_seen |= bool(metrics["right_female_role_contact"])
        self.left_male_contact_seen |= bool(metrics["left_male_role_contact"])
        if correct_roles:
            self.phase = max(self.phase, BucklePhase.ROLE_GRASPED)
        if correct_roles and bool(metrics["aligned"]):
            self.phase = max(self.phase, BucklePhase.ALIGNED)
            self.aligned_seen = True

        latch_now = bool(
            metrics["finite"]
            and correct_roles
            and metrics["aligned"]
            and metrics["deep_enough"]
            and metrics["body_contact"]
            and readback.max_penetration_m <= MAX_PENETRATION_M
        )
        self.latch_dwell_steps = self.latch_dwell_steps + 1 if latch_now else 0
        if self.latch_dwell_steps >= LATCH_DWELL_STEPS:
            self.latched_seen = True
            self.phase = BucklePhase.PRESSED_LATCHED

        stable_now = bool(
            self.latched_seen
            and metrics["finite"]
            and readback.max_penetration_m <= MAX_PENETRATION_M
            and metrics["deep_enough"]
            and metrics["aligned"]
            and metrics["relative_linear_speed_mps"] <= MAX_RELATIVE_LINEAR_SPEED_MPS
            and metrics["relative_angular_speed_radps"] <= MAX_RELATIVE_ANGULAR_SPEED_RADPS
        )
        self.success_hold_steps = self.success_hold_steps + 1 if stable_now else 0
        return {
            **metrics,
            "phase": self.phase.name.lower(),
            "phase_code": int(self.phase),
            "latch_dwell_steps": self.latch_dwell_steps,
            "success_hold_steps": self.success_hold_steps,
            "right_female_contact_seen": self.right_female_contact_seen,
            "left_male_contact_seen": self.left_male_contact_seen,
            "aligned_seen": self.aligned_seen,
            "latched_seen": self.latched_seen,
            "success_now": self.success_hold_steps >= SUCCESS_HOLD_STEPS,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase.name.lower(),
            "phase_code": int(self.phase),
            "latch_dwell_steps": self.latch_dwell_steps,
            "success_hold_steps": self.success_hold_steps,
            "right_female_contact_seen": self.right_female_contact_seen,
            "left_male_contact_seen": self.left_male_contact_seen,
            "aligned_seen": self.aligned_seen,
            "latched_seen": self.latched_seen,
        }


def reset_is_valid(reset: P006ResetState) -> bool:
    female = np.asarray(reset.female_position_m, dtype=np.float64)
    male = np.asarray(reset.male_position_m, dtype=np.float64)
    separation = float(np.linalg.norm(male - female))
    return bool(
        np.isfinite(np.concatenate([female, male])).all()
        and separation >= 0.20
        and female[2] >= TABLE_TOP_Z_M
        and male[2] >= TABLE_TOP_Z_M
        and abs(reset.female_yaw_deg - NOMINAL_YAW_DEG) <= ROOT_YAW_JITTER_DEG
        and abs(reset.male_yaw_deg - NOMINAL_YAW_DEG) <= ROOT_YAW_JITTER_DEG
    )
