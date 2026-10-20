"""CPU-safe physical contract for P0-01 bimanual folding-fan opening.

The task uses the project-owned ``Foldingfan001`` USD without changing its
geometry or articulation topology.  Success is evaluated from the eighteen
active revolute joints, rigid-body motion, and exact side-specific contacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


TASK_ID = "P0-01"
TASK_NAME = "t48_unfold_fan"
TASK_PROMPT = (
    "Use both hands to grasp the two outermost fan ribs and unfold the fan "
    "symmetrically to the episode target angle (60, 90, or 120 degrees). "
    "Keep the fan supported on the table and hold the target steadily."
)

ASSET_RELATIVE_PATH = (
    "sim/envs/usd/t48_unfold_fan/"
    "Foldingfan001/Foldingfan001.usd"
)
ASSET_SHA256 = "f0f3bbdabfc3131a7278b69f6a6216f658effb6f5e5a45e129c5d1eb6a4974f8"
ASSET_FILE_SHA256 = {
    "Foldingfan001.usd": ASSET_SHA256,
    "texture/T_Foldingfan001_BC001.png": (
        "12978ae73732fe3696e05f4922512d56f4d8a03018cbc309f49235fe941c7ee4"
    ),
    "texture/T_Foldingfan001_N001.png": (
        "b2082ebfb2523cd5cb0ece1ebdd70a57697f168183951c48d692ca854e7dff44"
    ),
    "texture/T_Foldingfan001_ORM001.png": (
        "d6fd28dbd8c4329b5e76fb9f41f763d02482aa0c94f6ccb39e6fbccbfdcf81c5"
    ),
}

FAN_ROOT_PATH = "/World/Task/P001Fan"
CENTER_BODY_PATH = f"{FAN_ROOT_PATH}/Foldingfan001"
LEFT_END_BODY_PATH = f"{FAN_ROOT_PATH}/Foldingfan001_Prop001"
RIGHT_END_BODY_PATH = f"{FAN_ROOT_PATH}/Foldingfan001_Prop019"
LEFT_JOINT_NAMES = tuple(f"Prop{index:03d}_joint" for index in range(1, 10))
LOCKED_JOINT_NAME = "Prop010_joint"
RIGHT_JOINT_NAMES = tuple(f"Prop{index:03d}_joint" for index in range(11, 20))
ACTIVE_JOINT_NAMES = LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES

TARGET_OPEN_CHOICES_DEG = (60.0, 90.0, 120.0)
TARGET_TOLERANCE_DEG = 8.0
INITIAL_SPAN_RANGE_DEG = (12.0, 20.0)
SOURCE_JOINT_LIMIT_DEG = 8.0
LEFT_RIGHT_BALANCE_MAX_DEG = 10.0
REQUIRED_NET_EXPANSION_FRACTION = 0.70
REQUIRED_BILATERAL_EXPANSION_FRACTION = 0.30

NOMINAL_ROOT_XY_M = np.asarray([-0.25, 0.0], dtype=np.float64)
ROOT_XY_HALF_RANGE_M = 0.025
NOMINAL_ROOT_YAW_DEG = 90.0
ROOT_YAW_HALF_RANGE_DEG = 5.0
ROOT_Z_M = 0.913
TABLE_TOP_Z_M = 0.890
FRICTION_RANGE = (0.50, 0.80)
MASS_SCALE_RANGE = (0.90, 1.10)

REQUIRED_CONTACT_FORCE_N = 0.10
# Crossed-rib contact remains a recoverable success/progress guard.  Robot-table
# force is diagnostic-only because the current aggregate cannot distinguish a
# benign fingertip/table touch from an unsafe arm collision.
FORBIDDEN_CONTACT_FORCE_N = 0.02
CONTACT_POLICY_VERSION = "p001_table_diagnostic_bilateral30_v2"
P001_TASK_STATE_SCHEMA_VERSION = 2
P001_TASK_STATE_DIM = 23
ROOT_TRANSLATION_MAX_M = 0.025
ROOT_LIFT_MAX_M = 0.015
PLANARITY_MAX_M = 0.030
STABLE_LINEAR_SPEED_MPS = 0.08
STABLE_JOINT_SPEED_RADPS = 0.35
EXPLOSION_LINEAR_SPEED_MPS = 5.0
EXPLOSION_JOINT_SPEED_RADPS = 20.0
SUCCESS_HOLD_STEPS = 15
CONTROL_HZ = 30
PHYSICS_HZ = 120
TIME_LIMIT_SECONDS = 60.0
TIMEOUT_STEPS = int(TIME_LIMIT_SECONDS * CONTROL_HZ)
RENDER_RESOLUTION_HW = (480, 640)

HAND_ACTION_DIM = 20
ACTION_DIM = 14 + 2 * HAND_ACTION_DIM


def p001_action_to_side_major_storage(action: object) -> np.ndarray:
    """Convert native pose-pose-hand-hand order to recorder side-major order."""

    native = np.asarray(action, dtype=np.float64)
    if native.shape != (ACTION_DIM,) or not np.all(np.isfinite(native)):
        raise ValueError(
            f"P0-01 action must be finite with shape ({ACTION_DIM},), "
            f"got {native.shape}"
        )
    right_pose = native[:7]
    left_pose = native[7:14]
    right_hand = native[14 : 14 + HAND_ACTION_DIM]
    left_hand = native[14 + HAND_ACTION_DIM :]
    return np.concatenate([right_pose, right_hand, left_pose, left_hand])
_DOWNWARD_GRASP_QUAT_WXYZ = np.asarray(
    [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], dtype=np.float64
)
_RIGHT_INWARD_GRASP_QUAT_WXYZ = np.asarray(
    [math.sqrt((2.0 + math.sqrt(2.0)) / 8.0),
     -math.sqrt((2.0 - math.sqrt(2.0)) / 8.0),
     math.sqrt((2.0 + math.sqrt(2.0)) / 8.0),
     math.sqrt((2.0 - math.sqrt(2.0)) / 8.0)], dtype=np.float64
)
_LEFT_INWARD_GRASP_QUAT_WXYZ = np.asarray(
    [math.sqrt((2.0 + math.sqrt(2.0)) / 8.0),
     math.sqrt((2.0 - math.sqrt(2.0)) / 8.0),
     math.sqrt((2.0 + math.sqrt(2.0)) / 8.0),
     -math.sqrt((2.0 - math.sqrt(2.0)) / 8.0)], dtype=np.float64
)
RIGHT_HOME = np.asarray(
    [-0.3715, -0.2395, 0.9975, *_DOWNWARD_GRASP_QUAT_WXYZ], dtype=np.float64
)
LEFT_HOME = np.asarray(
    [-0.3715, 0.2395, 0.9975, *_DOWNWARD_GRASP_QUAT_WXYZ], dtype=np.float64
)
WUJI_FAN_GRIP = np.asarray(
    [
        0.42, 0.06, 0.62, 0.52, 0.50,
        0.00, 0.72, 0.62, 0.52, 0.00,
        0.74, 0.64, 0.52, 0.00, 0.72,
        0.62, 0.50, 0.00, 0.70, 0.60,
    ],
    dtype=np.float64,
)

# Controller calibration is expressed relative to the randomized fan root,
# rather than as fixed world coordinates.  The reference is the seed-101
# physical calibration episode used to establish the first 60/90/120-degree
# positive trajectories.  Applying the sampled root SE(2) delta keeps the
# same hand-to-fan geometry for every allowed reset without writing object
# state after reset.
_FEEDBACK_REFERENCE_ROOT_XY_M = np.asarray(
    [-0.24918006861487768, 0.00353337934340699], dtype=np.float64
)
_FEEDBACK_REFERENCE_ROOT_YAW_DEG = 85.28474226478097
_FEEDBACK_REFERENCE_INITIAL_SPAN_DEG = 13.372173249800824
_FEEDBACK_ENDPOINT_RADIUS_M = 0.150
_FEEDBACK_REFERENCE_RIGHT_POSE = np.asarray(
    [
        -0.220355389,
        -0.138409221,
        0.985,
        0.501561588,
        -0.143754787,
        0.842030714,
        0.136948206,
    ],
    dtype=np.float64,
)
# The seed-101 calibration uses the lowest collision-free left-palm command.
# Randomized roots need a bounded workspace-dependent IK clearance adjustment;
# it is applied in ``calibrated_feedback_probe_poses`` below.
_FEEDBACK_REFERENCE_LEFT_START_POSE = np.asarray(
    [
        -0.161676959,
        0.220,
        0.980,
        0.453677554,
        0.364046707,
        0.785792597,
        -0.210182459,
    ],
    dtype=np.float64,
)
_FEEDBACK_REFERENCE_LEFT_END_POSE = np.asarray(
    [
        -0.161676959,
        0.145,
        0.980,
        0.453677554,
        0.364046707,
        0.785792597,
        -0.210182459,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class P001ResetState:
    seed: int
    root_x_m: float
    root_y_m: float
    root_yaw_deg: float
    initial_span_deg: float
    target_open_deg: float
    friction: float
    mass_scale: float

    def __post_init__(self) -> None:
        if float(self.target_open_deg) not in TARGET_OPEN_CHOICES_DEG:
            raise ValueError(
                f"P0-01 target must be one of {TARGET_OPEN_CHOICES_DEG}, "
                f"got {self.target_open_deg!r}"
            )
        if not INITIAL_SPAN_RANGE_DEG[0] <= float(self.initial_span_deg) <= INITIAL_SPAN_RANGE_DEG[1]:
            raise ValueError("P0-01 initial span is outside the frozen reset range")
        if float(self.initial_span_deg) >= float(self.target_open_deg) - TARGET_TOLERANCE_DEG:
            raise ValueError("P0-01 reset may not begin inside the success band")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @property
    def active_joint_positions_deg(self) -> tuple[float, ...]:
        per_joint = float(self.initial_span_deg) / 18.0
        return (per_joint,) * 9 + (-per_joint,) * 9


def sample_reset_state(
    seed: int,
    *,
    target_open_deg: float | None = None,
) -> P001ResetState:
    """Sample one deterministic reset without touching global RNG state."""

    normalized_seed = int(seed) % (2**32 - 1)
    rng = np.random.RandomState(normalized_seed)
    xy = NOMINAL_ROOT_XY_M + rng.uniform(
        -ROOT_XY_HALF_RANGE_M, ROOT_XY_HALF_RANGE_M, size=2
    )
    yaw = NOMINAL_ROOT_YAW_DEG + rng.uniform(
        -ROOT_YAW_HALF_RANGE_DEG, ROOT_YAW_HALF_RANGE_DEG
    )
    initial_span = rng.uniform(*INITIAL_SPAN_RANGE_DEG)
    sampled_target = (
        float(rng.choice(TARGET_OPEN_CHOICES_DEG))
        if target_open_deg is None
        else float(target_open_deg)
    )
    return P001ResetState(
        seed=normalized_seed,
        root_x_m=float(xy[0]),
        root_y_m=float(xy[1]),
        root_yaw_deg=float(yaw),
        initial_span_deg=float(initial_span),
        target_open_deg=sampled_target,
        friction=float(rng.uniform(*FRICTION_RANGE)),
        mass_scale=float(rng.uniform(*MASS_SCALE_RANGE)),
    )


def calibrated_feedback_probe_poses(
    reset_state: P001ResetState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform the physical probe calibration by the sampled fan-root SE(2).

    Returns ``(right, left_start, left_end)`` Cartesian palm poses in world
    coordinates using ``[x, y, z, qw, qx, qy, qz]`` convention.
    """

    delta_yaw = math.radians(
        float(reset_state.root_yaw_deg) - _FEEDBACK_REFERENCE_ROOT_YAW_DEG
    )
    cosine = math.cos(delta_yaw)
    sine = math.sin(delta_yaw)
    rotation_xy = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    destination_origin = np.asarray(
        [reset_state.root_x_m, reset_state.root_y_m], dtype=np.float64
    )
    yaw_quaternion = np.asarray(
        [math.cos(delta_yaw / 2.0), 0.0, 0.0, math.sin(delta_yaw / 2.0)],
        dtype=np.float64,
    )

    def transform(reference_pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(reference_pose, dtype=np.float64).copy()
        pose[:2] = destination_origin + rotation_xy @ (
            pose[:2] - _FEEDBACK_REFERENCE_ROOT_XY_M
        )
        w1, x1, y1, z1 = yaw_quaternion
        w2, x2, y2, z2 = pose[3:]
        pose[3:] = np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
        pose[3:] /= np.linalg.norm(pose[3:])
        return pose

    right_pose = transform(_FEEDBACK_REFERENCE_RIGHT_POSE)
    left_start_pose = transform(_FEEDBACK_REFERENCE_LEFT_START_POSE)
    left_end_pose = transform(_FEEDBACK_REFERENCE_LEFT_END_POSE)
    root_xy_delta_m = float(
        np.linalg.norm(destination_origin - _FEEDBACK_REFERENCE_ROOT_XY_M)
    )
    root_yaw_delta_deg = abs(
        float(reset_state.root_yaw_deg) - _FEEDBACK_REFERENCE_ROOT_YAW_DEG
    )
    initial_span_delta_deg = abs(
        float(reset_state.initial_span_deg)
        - _FEEDBACK_REFERENCE_INITIAL_SPAN_DEG
    )
    # The left ROKAE chain needs extra vertical IK margin only as the fan moves
    # farther forward (negative world X) from the calibration layout.  A
    # direction-agnostic cap made the opposite-X edge layouts too high and
    # caused the fingers to miss Prop001 before touching the table.
    forward_reach_delta_m = max(
        0.0,
        float(_FEEDBACK_REFERENCE_ROOT_XY_M[0] - destination_origin[0]),
    )
    left_ik_clearance_cap_m = 0.006 + min(
        0.002,
        0.10 * forward_reach_delta_m,
    )
    left_ik_clearance_m = min(
        left_ik_clearance_cap_m,
        0.20 * root_xy_delta_m
        + 0.0007 * root_yaw_delta_deg
        + 0.0005 * initial_span_delta_deg,
    )
    left_start_pose[2] += left_ik_clearance_m
    left_end_pose[2] += left_ik_clearance_m
    endpoint_radial_delta_m = _FEEDBACK_ENDPOINT_RADIUS_M * (
        math.sin(math.radians(float(reset_state.initial_span_deg)) / 2.0)
        - math.sin(math.radians(_FEEDBACK_REFERENCE_INITIAL_SPAN_DEG) / 2.0)
    )
    outward_xy = rotation_xy @ np.asarray(
        [0.0, endpoint_radial_delta_m], dtype=np.float64
    )
    # A wider reset places the actual outer ribs farther apart.  Track that
    # geometry so neither hand crosses to the opposite outer rib while moving
    # from the safe approach pose into contact.
    left_end_pose[:2] += outward_xy
    right_pose[:2] -= outward_xy
    return right_pose, left_start_pose, left_end_pose


def span_components_deg(active_joint_angles_rad: Sequence[float]) -> tuple[float, float, float]:
    """Return total, left, and right physical opening from the 18 active DOFs."""

    angles = np.asarray(active_joint_angles_rad, dtype=np.float64).reshape(-1)
    if angles.shape != (18,):
        raise ValueError(f"P0-01 requires 18 active joints, got {angles.shape}")
    left = float(np.sum(np.rad2deg(angles[:9])))
    right = float(np.sum(-np.rad2deg(angles[9:])))
    return left + right, left, right


@dataclass(frozen=True)
class P001PhysicalReadback:
    active_joint_angles_rad: tuple[float, ...]
    active_joint_velocities_radps: tuple[float, ...]
    root_position_m: tuple[float, float, float]
    reset_root_position_m: tuple[float, float, float]
    max_body_linear_speed_mps: float
    planarity_error_m: float
    left_end_contact_force_n: float
    right_end_contact_force_n: float
    left_wrong_end_contact_force_n: float
    right_wrong_end_contact_force_n: float
    robot_table_contact_force_n: float
    positive_span_increase_deg: float
    bilateral_positive_span_increase_deg: float
    all_bodies_inside_safe_zone: bool
    min_body_height_m: float


def _finite_readback(readback: P001PhysicalReadback) -> bool:
    values: list[float] = []
    for key, value in asdict(readback).items():
        if key == "robot_table_contact_force_n":
            # The aggregate robot-table view is retained for diagnostics only.
            # A missing/invalid diagnostic must not fail an otherwise valid
            # physical task state.
            continue
        if isinstance(value, tuple):
            values.extend(float(item) for item in value)
        elif isinstance(value, bool):
            continue
        else:
            values.append(float(value))
    return bool(np.all(np.isfinite(values)))


def evaluate_physical_readback(
    readback: P001PhysicalReadback,
    reset_state: P001ResetState,
) -> dict[str, float | bool]:
    """Evaluate the frozen fail-closed task contract from physical readback."""

    finite = _finite_readback(readback)
    if finite:
        span_deg, left_span_deg, right_span_deg = span_components_deg(
            readback.active_joint_angles_rad
        )
    else:
        span_deg = left_span_deg = right_span_deg = -math.inf
    target_error_deg = abs(span_deg - float(reset_state.target_open_deg))
    balance_error_deg = abs(left_span_deg - right_span_deg)
    net_increase_deg = span_deg - float(reset_state.initial_span_deg)
    required_net_increase_deg = REQUIRED_NET_EXPANSION_FRACTION * (
        float(reset_state.target_open_deg) - float(reset_state.initial_span_deg)
    )
    positive_increase = max(0.0, float(readback.positive_span_increase_deg))
    bilateral_increase = max(
        0.0, float(readback.bilateral_positive_span_increase_deg)
    )
    bilateral_fraction = (
        min(1.0, bilateral_increase / positive_increase)
        if positive_increase > 1.0e-9
        else 0.0
    )
    root = np.asarray(readback.root_position_m, dtype=np.float64)
    reset_root = np.asarray(readback.reset_root_position_m, dtype=np.float64)
    root_displacement_m = (
        float(np.linalg.norm(root[:2] - reset_root[:2])) if finite else math.inf
    )
    root_lift_m = abs(float(root[2] - reset_root[2])) if finite else math.inf
    angles_deg = np.rad2deg(
        np.asarray(readback.active_joint_angles_rad, dtype=np.float64)
    )
    joint_direction_ok = bool(
        finite
        and np.all(angles_deg[:9] >= -0.25)
        and np.all(angles_deg[9:] <= 0.25)
    )
    joint_limits_ok = bool(
        finite and np.all(np.abs(angles_deg) <= SOURCE_JOINT_LIMIT_DEG + 0.10)
    )
    left_contact = readback.left_end_contact_force_n >= REQUIRED_CONTACT_FORCE_N
    right_contact = readback.right_end_contact_force_n >= REQUIRED_CONTACT_FORCE_N
    wrong_end_clear = bool(
        readback.left_wrong_end_contact_force_n <= FORBIDDEN_CONTACT_FORCE_N
        and readback.right_wrong_end_contact_force_n <= FORBIDDEN_CONTACT_FORCE_N
    )
    robot_table_clear = bool(
        math.isfinite(float(readback.robot_table_contact_force_n))
        and readback.robot_table_contact_force_n <= FORBIDDEN_CONTACT_FORCE_N
    )
    root_stable = bool(
        root_displacement_m <= ROOT_TRANSLATION_MAX_M
        and root_lift_m <= ROOT_LIFT_MAX_M
    )
    planarity_ok = readback.planarity_error_m <= PLANARITY_MAX_M
    max_joint_speed = (
        float(np.max(np.abs(readback.active_joint_velocities_radps)))
        if finite
        else math.inf
    )
    velocity_stable = bool(
        readback.max_body_linear_speed_mps <= STABLE_LINEAR_SPEED_MPS
        and max_joint_speed <= STABLE_JOINT_SPEED_RADPS
    )
    safe_zone = bool(readback.all_bodies_inside_safe_zone)
    dropped = bool(finite and readback.min_body_height_m < TABLE_TOP_Z_M - 0.03)
    exploded = bool(
        readback.max_body_linear_speed_mps > EXPLOSION_LINEAR_SPEED_MPS
        or max_joint_speed > EXPLOSION_JOINT_SPEED_RADPS
    )
    gross_root_drift = root_displacement_m > 0.08 or root_lift_m > 0.05
    forbidden_contact = not wrong_end_clear
    failure = bool(
        not finite
        or not joint_direction_ok
        or not joint_limits_ok
        or not safe_zone
        or dropped
        or exploded
        or gross_root_drift
    )
    target_ok = target_error_deg <= TARGET_TOLERANCE_DEG
    balance_ok = balance_error_deg <= LEFT_RIGHT_BALANCE_MAX_DEG
    net_increase_ok = net_increase_deg >= required_net_increase_deg
    bilateral_expansion_ok = (
        bilateral_fraction >= REQUIRED_BILATERAL_EXPANSION_FRACTION
    )
    success_now = bool(
        finite
        and not failure
        and not forbidden_contact
        and target_ok
        and balance_ok
        and net_increase_ok
        and bilateral_expansion_ok
        and left_contact
        and right_contact
        and root_stable
        and planarity_ok
        and velocity_stable
    )
    return {
        "finite": finite,
        "span_deg": span_deg,
        "left_span_deg": left_span_deg,
        "right_span_deg": right_span_deg,
        "target_deg": float(reset_state.target_open_deg),
        "target_error_deg": target_error_deg,
        "target_ok": target_ok,
        "balance_error_deg": balance_error_deg,
        "balance_ok": balance_ok,
        "net_increase_deg": net_increase_deg,
        "required_net_increase_deg": required_net_increase_deg,
        "net_increase_ok": net_increase_ok,
        "bilateral_expansion_fraction": bilateral_fraction,
        "bilateral_expansion_ok": bilateral_expansion_ok,
        "left_end_contact": left_contact,
        "right_end_contact": right_contact,
        "wrong_end_clear": wrong_end_clear,
        "robot_table_clear": robot_table_clear,
        "forbidden_contact": forbidden_contact,
        "root_displacement_m": root_displacement_m,
        "root_lift_m": root_lift_m,
        "root_stable": root_stable,
        "planarity_error_m": float(readback.planarity_error_m),
        "planarity_ok": planarity_ok,
        "max_body_linear_speed_mps": float(readback.max_body_linear_speed_mps),
        "max_joint_speed_radps": max_joint_speed,
        "velocity_stable": velocity_stable,
        "joint_direction_ok": joint_direction_ok,
        "joint_limits_ok": joint_limits_ok,
        "safe_zone": safe_zone,
        "dropped": dropped,
        "exploded": exploded,
        "gross_root_drift": gross_root_drift,
        "failure": failure,
        "success_now": success_now,
    }


def bilateral_contact_credit_allowed(
    *,
    left_end_force_n: float,
    right_end_force_n: float,
    left_wrong_end_force_n: float,
    right_wrong_end_force_n: float,
) -> bool:
    """Return whether this substep earns bilateral endpoint progress credit."""

    values = (
        left_end_force_n,
        right_end_force_n,
        left_wrong_end_force_n,
        right_wrong_end_force_n,
    )
    return bool(
        all(math.isfinite(float(value)) for value in values)
        and float(left_end_force_n) >= REQUIRED_CONTACT_FORCE_N
        and float(right_end_force_n) >= REQUIRED_CONTACT_FORCE_N
        and float(left_wrong_end_force_n) <= FORBIDDEN_CONTACT_FORCE_N
        and float(right_wrong_end_force_n) <= FORBIDDEN_CONTACT_FORCE_N
    )


def hard_failure_reason(metrics: Mapping[str, float | bool]) -> str | None:
    """Return the first non-contact terminal reason in stable priority order."""

    if not bool(metrics.get("finite", False)):
        return "nonfinite_state"
    if not bool(metrics.get("joint_direction_ok", False)):
        return "joint_direction"
    if not bool(metrics.get("joint_limits_ok", False)):
        return "joint_limit"
    if not bool(metrics.get("safe_zone", False)):
        return "safe_zone"
    if bool(metrics.get("dropped", False)):
        return "dropped"
    if bool(metrics.get("exploded", False)):
        return "exploded"
    if bool(metrics.get("gross_root_drift", False)):
        return "gross_root_drift"
    return None


class P001SuccessTracker:
    """Require consecutive stable physical success for exactly 0.5 seconds."""

    def __init__(
        self,
        reset_state: P001ResetState,
        hold_steps: int = SUCCESS_HOLD_STEPS,
    ) -> None:
        if int(hold_steps) <= 0:
            raise ValueError("hold_steps must be positive")
        self.reset_state = reset_state
        self.hold_steps = int(hold_steps)
        self.reset()

    def reset(self) -> None:
        self.consecutive_steps = 0
        self.failure_latched = False
        self.last_metrics: dict[str, float | bool] = {}

    def update(self, readback: P001PhysicalReadback) -> tuple[bool, bool]:
        metrics = evaluate_physical_readback(readback, self.reset_state)
        self.last_metrics = metrics
        self.failure_latched = bool(self.failure_latched or metrics["failure"])
        if metrics["success_now"] and not self.failure_latched:
            self.consecutive_steps += 1
        else:
            self.consecutive_steps = 0
        return self.consecutive_steps >= self.hold_steps, self.failure_latched


def _smoothstep(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _blend_pose(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    weight = _smoothstep(alpha)
    position = start[:3] + weight * (end[:3] - start[:3])
    start_quat = start[3:7] / np.linalg.norm(start[3:7])
    end_quat = end[3:7] / np.linalg.norm(end[3:7])
    if float(np.dot(start_quat, end_quat)) < 0.0:
        end_quat = -end_quat
    quaternion = (1.0 - weight) * start_quat + weight * end_quat
    quaternion /= np.linalg.norm(quaternion)
    return np.concatenate([position, quaternion])


def endpoint_tcp_poses(
    reset_state: P001ResetState,
    span_deg: float,
    *,
    height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return side-on right/left palm poses for the two outer ribs.

    Wuji fingertips project roughly 10 cm forward from each palm.  Keeping the
    palms outside the almost-closed fan avoids hand-hand collision while the
    inward-facing fingers meet the respective outermost rib.
    """

    half_angle = math.radians(float(span_deg) / 2.0)
    yaw = math.radians(float(reset_state.root_yaw_deg - NOMINAL_ROOT_YAW_DEG))
    radius = 0.150
    fingertip_forward_offset = 0.110
    grasp_yaw = math.radians(45.0)
    palm_rearward_offset = fingertip_forward_offset * math.cos(grasp_yaw)
    palm_outward_offset = fingertip_forward_offset * math.sin(grasp_yaw)
    local_left = np.asarray(
        [
            radius * math.cos(half_angle) - palm_rearward_offset,
            radius * math.sin(half_angle) + palm_outward_offset,
        ],
        dtype=np.float64,
    )
    local_right = np.asarray([local_left[0], -local_left[1]], dtype=np.float64)
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    origin = np.asarray([reset_state.root_x_m, reset_state.root_y_m])
    left_xy = origin + rotation @ local_left
    right_xy = origin + rotation @ local_right
    yaw_quat = np.asarray(
        [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
        dtype=np.float64,
    )

    def rotate_grasp(quaternion: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = yaw_quat
        w2, x2, y2, z2 = quaternion
        return np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )

    left_quat = rotate_grasp(_LEFT_INWARD_GRASP_QUAT_WXYZ)
    right_quat = rotate_grasp(_RIGHT_INWARD_GRASP_QUAT_WXYZ)
    # The mirrored ROKAE chain settles the left Wuji fingertips slightly higher
    # for the same Cartesian z command.  Compensate at the palm target so both
    # physical fingertip rows meet the same rib surface.
    left_pose = np.asarray(
        [*left_xy, height_m - 0.010, *left_quat], dtype=np.float64
    )
    right_pose = np.asarray(
        [*right_xy, height_m, *right_quat], dtype=np.float64
    )
    return right_pose, left_pose


class P001ScriptedPolicy:
    """Deterministic two-hand path used only for physical acceptance runs."""

    def __init__(
        self,
        reset_state: P001ResetState,
        *,
        right_home: np.ndarray | None = None,
        left_home: np.ndarray | None = None,
    ) -> None:
        self.reset_state = reset_state
        self.right_home = np.asarray(
            RIGHT_HOME if right_home is None else right_home,
            dtype=np.float64,
        ).reshape(7).copy()
        self.left_home = np.asarray(
            LEFT_HOME if left_home is None else left_home,
            dtype=np.float64,
        ).reshape(7).copy()

    def action(self, step: int) -> np.ndarray:
        step = int(step)
        right_high, left_high = endpoint_tcp_poses(
            self.reset_state, self.reset_state.initial_span_deg, height_m=1.030
        )
        right_grasp, left_grasp = endpoint_tcp_poses(
            self.reset_state, self.reset_state.initial_span_deg, height_m=0.875
        )
        right_lift, _ = endpoint_tcp_poses(
            self.reset_state, self.reset_state.initial_span_deg, height_m=0.887
        )
        _, left_lower_grasp = endpoint_tcp_poses(
            self.reset_state, self.reset_state.initial_span_deg, height_m=0.950
        )
        if step < 90:
            alpha = step / 90.0
            right = _blend_pose(self.right_home, right_high, alpha)
            left = _blend_pose(self.left_home, left_high, alpha)
            right_grip = np.zeros(HAND_ACTION_DIM, dtype=np.float64)
            left_grip = np.zeros(HAND_ACTION_DIM, dtype=np.float64)
        elif step < 120:
            right = right_high.copy()
            left = left_high.copy()
            right_grip = np.zeros(HAND_ACTION_DIM, dtype=np.float64)
            left_grip = np.zeros(HAND_ACTION_DIM, dtype=np.float64)
        elif step < 210:
            alpha = (step - 120) / 90.0
            right = _blend_pose(right_high, right_grasp, alpha)
            left = left_high.copy()
            right_grip = _smoothstep(alpha) * WUJI_FAN_GRIP
            left_grip = np.zeros(HAND_ACTION_DIM, dtype=np.float64)
        elif step < 270:
            alpha = (step - 210) / 60.0
            right = _blend_pose(right_grasp, right_lift, alpha)
            left = left_high.copy()
            right_grip = WUJI_FAN_GRIP.copy()
            left_grip = np.zeros(HAND_ACTION_DIM, dtype=np.float64)
        elif step < 360:
            alpha = (step - 270) / 90.0
            right = right_lift.copy()
            left = _blend_pose(left_high, left_lower_grasp, alpha)
            right_grip = WUJI_FAN_GRIP.copy()
            left_grip = _smoothstep(alpha) * WUJI_FAN_GRIP
        elif step < 420:
            right = right_lift.copy()
            left = left_lower_grasp.copy()
            right_grip = WUJI_FAN_GRIP.copy()
            left_grip = WUJI_FAN_GRIP.copy()
        elif step < 600:
            alpha = (step - 420) / 180.0
            span = self.reset_state.initial_span_deg + _smoothstep(alpha) * (
                self.reset_state.target_open_deg - self.reset_state.initial_span_deg
            )
            right, _ = endpoint_tcp_poses(
                self.reset_state, span, height_m=0.887
            )
            _, left = endpoint_tcp_poses(
                self.reset_state, span, height_m=0.950
            )
            right_grip = WUJI_FAN_GRIP.copy()
            left_grip = WUJI_FAN_GRIP.copy()
        else:
            right, _ = endpoint_tcp_poses(
                self.reset_state, self.reset_state.target_open_deg, height_m=0.887
            )
            _, left = endpoint_tcp_poses(
                self.reset_state, self.reset_state.target_open_deg, height_m=0.950
            )
            right_grip = WUJI_FAN_GRIP.copy()
            left_grip = WUJI_FAN_GRIP.copy()
        return np.concatenate([right, left, right_grip, left_grip])

    def metadata(self) -> Mapping[str, Any]:
        return {
            "policy": "p001_scripted_bimanual_fan_arc_v2",
            "reset": self.reset_state.to_dict(),
            "object_pose_writes_after_reset": 0,
        }


__all__ = [
    "P001PhysicalReadback",
    "P001ResetState",
    "P001ScriptedPolicy",
    "P001SuccessTracker",
    "P001_TASK_STATE_DIM",
    "P001_TASK_STATE_SCHEMA_VERSION",
    "bilateral_contact_credit_allowed",
    "calibrated_feedback_probe_poses",
    "evaluate_physical_readback",
    "hard_failure_reason",
    "sample_reset_state",
    "span_components_deg",
]
