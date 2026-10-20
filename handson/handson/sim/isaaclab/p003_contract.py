"""CPU-safe semantic and success contract for P0-03.

The live environment must populate :class:`P003PhysicalReadback` exclusively
from PhysX joint, pose, velocity, and contact readback.  The contract never
writes object state and an endpoint pose alone cannot satisfy the task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np


TASK_ID = "P0-03"
TASK_NAME = "p003_bimanual_folding_ruler_shape_configuration"
ASSET_RELATIVE_PATH = (
    "taskdev/tasks/p003_bimanual_folding_ruler_shape_configuration/"
    "assets/Ruler035/Ruler035.usd"
)
ASSET_SHA256 = "55f3823f9ae46ba34f5bfcc12332aaa512cf99eb344e2db8bfa92a8825ca0369"

RULER_ROOT_PATH = "/World/Task/P003Ruler"
S0_BODY_PATH = f"{RULER_ROOT_PATH}/Ruler035"
S1_BODY_PATH = f"{RULER_ROOT_PATH}/Ruler035_Prop001"
S2_BODY_PATH = f"{RULER_ROOT_PATH}/Ruler035_Prop002"
JOINT_1_PATH = f"{S1_BODY_PATH}/Prop001_joint"
JOINT_2_PATH = f"{S2_BODY_PATH}/Prop002_joint"

CONTROL_HZ = 30
PHYSICS_HZ = 240
ACTION_DIM = 54
HAND_ACTION_DIM = 20
P003_RUNTIME_POLICY_VERSION = "p003_tabletop_physical_v2"
JOINT_TOLERANCE_DEG = 5.0
DIRECTION_TOLERANCE_DEG = 8.0
REQUIRED_CONTACT_FORCE_N = 0.10
RELEASE_CONTACT_FORCE_N = 0.02
FORBIDDEN_CONTACT_FORCE_N = 0.02
FINAL_HAND_TABLE_CONTACT_FORCE_N = 0.02
# ``LIFT_HEIGHT_M`` remains public for old recording/report readers.  It is no
# longer a progress or success predicate in the tabletop contract.
LIFT_HEIGHT_M = 0.05
TABLETOP_OPERATION_BOTTOM_TOLERANCE_M = 0.020
TABLETOP_FINAL_BOTTOM_TOLERANCE_M = 0.010
TABLETOP_BOTTOM_PENETRATION_TOLERANCE_M = 0.001
TABLETOP_MIN_SUPPORTED_SEGMENTS = 2
PLACEMENT_HEIGHT_TOLERANCE_M = TABLETOP_FINAL_BOTTOM_TOLERANCE_M
PLACEMENT_XY_TOLERANCE_M = 0.05
STABLE_LINEAR_SPEED_MPS = 0.05
STABLE_ANGULAR_SPEED_RADPS = 0.50
SHAPE_HOLD_STEPS = 15
RELEASE_HOLD_STEPS = 30
TIMEOUT_STEPS = 900

# The action values are native Wuji joint targets.  This is the mature
# benchmark power-grip posture.  The index and middle fingers stay fully
# curled clear of the table; the right ring finger uses the same clearance
# curl while the remaining non-thumb fingers retain the validated thin-ruler
# posture around the lightweight articulation.
WUJI_RULER_GRIP_LEFT = np.asarray(
    [
        0.36, 0.06, 0.504, 0.42, 0.68,
        0.00, 0.90, 0.78, 0.70, 0.00,
        0.92, 0.80, 0.25, 0.00, 0.35,
        0.32, 0.384, 0.00, 0.516, 0.444,
    ],
    dtype=np.float64,
)
WUJI_RULER_GRIP_RIGHT = np.asarray(
    [
        0.36, 0.06, 0.504, 0.42, 0.68,
        0.00, 0.90, 0.78, 0.70, 0.00,
        0.92, 0.80, 0.25, 0.00, 0.35,
        0.32, 0.64, 0.00, 0.86, 0.74,
    ],
    dtype=np.float64,
)
WUJI_RULER_SECURE_GRIP_LEFT = np.asarray(
    [
        0.60, 0.10, 0.84, 0.70, 0.68,
        0.00, 0.90, 0.78, 0.70, 0.00,
        0.92, 0.80, 0.25, 0.00, 0.35,
        0.32, 0.384, 0.00, 0.516, 0.444,
    ],
    dtype=np.float64,
)
WUJI_RULER_SECURE_GRIP_RIGHT = WUJI_RULER_GRIP_RIGHT.copy()
WUJI_RULER_SECURE_GRIP_RIGHT[:4] = WUJI_RULER_SECURE_GRIP_LEFT[:4]
WUJI_RULER_SECURE_GRIP_RIGHT[12:16] = np.asarray(
    [0.25, 0.00, 0.35, 0.32], dtype=np.float64
)
# The left ring-finger tip is the physical S2 owner during grasp and shape
# configuration, so it must retain the thin-ruler posture above.  During
# placement that same long fingertip can become the lowest robot point while
# S2 is still centimetres above the table.  Once the shape-hold history is
# complete, curl only that finger into the kinematically measured clearance
# posture.  This changes the hand command, never the ruler state or success
# thresholds.
WUJI_RULER_PLACEMENT_GRIP_LEFT = WUJI_RULER_SECURE_GRIP_LEFT.copy()
WUJI_RULER_PLACEMENT_GRIP_LEFT[12:16] = np.asarray(
    [0.55, 0.00, 0.75, 0.68], dtype=np.float64
)
WUJI_RULER_RIGHT_ROLE_CLEARANCE_GRIP = WUJI_RULER_GRIP_RIGHT.copy()
WUJI_RULER_RIGHT_ROLE_CLEARANCE_GRIP[12:16] = np.asarray(
    [0.46, 0.00, 0.62, 0.55], dtype=np.float64
)
WUJI_RULER_RELEASE_OPEN_LEFT = np.zeros(HAND_ACTION_DIM, dtype=np.float64)
WUJI_RULER_RELEASE_OPEN_RIGHT = np.zeros(HAND_ACTION_DIM, dtype=np.float64)
WUJI_RULER_RELEASE_LEFT = np.zeros(HAND_ACTION_DIM, dtype=np.float64)
WUJI_RULER_RELEASE_LEFT[4:20] = 0.50 * WUJI_RULER_SECURE_GRIP_LEFT[4:20]
WUJI_RULER_RELEASE_RIGHT = np.zeros(HAND_ACTION_DIM, dtype=np.float64)
WUJI_RULER_RELEASE_RIGHT[4:20] = 0.50 * WUJI_RULER_SECURE_GRIP_RIGHT[4:20]

_POLICY_CLEARANCE_END_STEP = 30
_POLICY_APPROACH_END_STEP = 90
_POLICY_DESCEND_END_STEP = 210
_POLICY_GRIP_RAMP_STEPS = 45
_POLICY_SECURE_STEPS = 30
_POLICY_RELEASE_STEPS = 5
_POLICY_HIGH_Z_M = 1.10
_POLICY_WRIST_ABOVE_BODY_M = 0.120
_POLICY_NONEXTENDED_STAGING_OFFSET_M = 0.0250
_POLICY_NONEXTENDED_STAGING_XY_TOLERANCE_M = 0.020
_POLICY_RIGHT_CONTACT_QUATERNION_WXYZ = np.asarray(
    [0.0, 0.9238795325, 0.0, 0.3826834324], dtype=np.float64
)
_POLICY_LEFT_CONTACT_QUATERNION_WXYZ = np.asarray(
    [0.0, 0.9238795325, 0.0, 0.3826834324], dtype=np.float64
)
_POLICY_RIGHT_CONTACT_XY_OFFSET_M = np.asarray(
    [-0.153, 0.051], dtype=np.float64
)
_POLICY_LEFT_CONTACT_XY_OFFSET_M = np.asarray(
    [-0.136, -0.032], dtype=np.float64
)


def _unit_xy(angle_deg: float) -> tuple[float, float]:
    angle = math.radians(angle_deg)
    return (math.cos(angle), math.sin(angle))


@dataclass(frozen=True)
class TargetShape:
    key: str
    title: str
    joint_targets_deg: tuple[float, float]
    segment_directions_task_xy: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    shape_description: str
    prompt: str


def _shape(
    key: str,
    title: str,
    q1_deg: float,
    q2_deg: float,
    description: str,
    instruction: str,
) -> TargetShape:
    # Directions point from the first hinge into S0, from hinge 1 to hinge 2
    # along S1, and from hinge 2 toward the free end of S2.  The S2 mesh is
    # authored toward local -X, hence the final +180 degrees.
    directions = (
        _unit_xy(0.0),
        _unit_xy(q1_deg),
        _unit_xy(q1_deg + q2_deg + 180.0),
    )
    prompt = (
        f"Target shape: {title}. Target joints: q1={q1_deg:+.0f}\N{DEGREE SIGN}, "
        f"q2={q2_deg:+.0f}\N{DEGREE SIGN}. Task-frame direction cue: "
        "\N{LEFTWARDS ARROW} left | right \N{RIGHTWARDS ARROW}. "
        "Keep the folding ruler supported by the table. Use the right hand "
        "on ruler segment S0 and the left hand on segment S2, then push and "
        "rotate the leaves on the tabletop to make the target shape: "
        f"{instruction} Keep the ruler in the marked center zone, release "
        "both hands, leave it still on the table, and lift both hands clear "
        "of the table for the final success frame. Touching the table or the "
        "opposite ruler segment only pauses progress; it does not fail the "
        "episode."
    )
    return TargetShape(
        key=key,
        title=title,
        joint_targets_deg=(q1_deg, q2_deg),
        segment_directions_task_xy=directions,
        shape_description=description,
        prompt=prompt,
    )


TARGET_SHAPES: Mapping[str, TargetShape] = {
    "extended": _shape(
        "extended",
        "Nearly straight extension",
        -170.0,
        170.0,
        "All three ruler leaves form one nearly straight line; both hinges are ten degrees short of their hard limits.",
        "open the first hinge clockwise to -170 degrees and the second hinge counter-clockwise to +170 degrees so all three leaves form one nearly straight line.",
    ),
    "square_u": _shape(
        "square_u",
        "Square-corner stepped U",
        -90.0,
        90.0,
        "The middle leaf is perpendicular to both outer leaves, producing a squared three-sided stepped/U silhouette with two right-angle corners.",
        "make the square-corner target: set the S0-S1 corner to -90 degrees and the S1-S2 corner to +90 degrees, with the middle leaf perpendicular to both outer leaves.",
    ),
    "zigzag_left": _shape(
        "zigzag_left",
        "Left-opening oblique zigzag",
        -60.0,
        120.0,
        "An asymmetric oblique zigzag whose first elbow is shallow and whose second elbow opens toward task-frame left.",
        "make the left-opening oblique zigzag: set the first hinge to -60 degrees and the second hinge to +120 degrees; the second elbow must point toward the target's left arrow.",
    ),
    "zigzag_right": _shape(
        "zigzag_right",
        "Right-opening oblique zigzag",
        -120.0,
        60.0,
        "The mirror-direction oblique zigzag: the first elbow is deep and the second elbow opens toward task-frame right.",
        "make the right-opening oblique zigzag: set the first hinge to -120 degrees and the second hinge to +60 degrees; the second elbow must point toward the target's right arrow.",
    ),
}
TARGET_KEYS = tuple(TARGET_SHAPES)


@dataclass(frozen=True)
class P003ResetState:
    seed: int
    target_key: str
    root_x_m: float
    root_y_m: float
    root_yaw_deg: float
    joint_1_deg: float
    joint_2_deg: float
    friction: float
    mass_scale: float

    @property
    def target(self) -> TargetShape:
        return TARGET_SHAPES[self.target_key]

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def sample_reset_state(seed: int) -> P003ResetState:
    """Return one deterministic, non-successful randomized episode reset."""

    normalized_seed = int(seed) % (2**32 - 1)
    rng = np.random.RandomState(normalized_seed)
    target_key = str(rng.choice(TARGET_KEYS))
    return P003ResetState(
        seed=normalized_seed,
        target_key=target_key,
        root_x_m=float(rng.uniform(-0.19, -0.17)),
        root_y_m=float(rng.uniform(-0.09, -0.07)),
        root_yaw_deg=float(rng.uniform(82.0, 98.0)),
        joint_1_deg=float(rng.uniform(-45.0, -25.0)),
        joint_2_deg=float(rng.uniform(25.0, 45.0)),
        friction=float(rng.uniform(0.65, 0.75)),
        mass_scale=float(rng.uniform(0.95, 1.05)),
    )


@dataclass(frozen=True)
class P003PhysicalReadback:
    joint_angles_deg: tuple[float, float]
    segment_directions_task_xy: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    root_position_m: tuple[float, float, float]
    reset_root_position_m: tuple[float, float, float]
    object_center_position_m: tuple[float, float, float]
    reset_object_center_position_m: tuple[float, float, float]
    table_top_z_m: float
    segment_bottom_z_m: tuple[float, float, float]
    minimum_segment_bottom_z_m: float
    maximum_linear_speed_mps: float
    maximum_angular_speed_radps: float
    left_s0_contact_force_n: float
    right_s2_contact_force_n: float
    left_s2_contact_force_n: float
    right_s0_contact_force_n: float
    left_all_ruler_contact_force_n: float
    right_all_ruler_contact_force_n: float
    robot_table_contact_force_n: float
    hand_table_contact_force_n: float = 0.0
    hand_table_contact_valid: bool = True
    finite_state: bool = True


def _direction_error_deg(actual: tuple[float, float], expected: tuple[float, float]) -> float:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return math.inf
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-9 or b_norm < 1e-9:
        return math.inf
    dot = float(np.clip(np.dot(a / a_norm, b / b_norm), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def evaluate_shape(readback: P003PhysicalReadback, target: TargetShape) -> dict[str, object]:
    joint_errors = tuple(
        abs(float(actual) - float(expected))
        for actual, expected in zip(readback.joint_angles_deg, target.joint_targets_deg)
    )
    direction_errors = tuple(
        _direction_error_deg(actual, expected)
        for actual, expected in zip(
            readback.segment_directions_task_xy,
            target.segment_directions_task_xy,
        )
    )
    return {
        "joint_errors_deg": joint_errors,
        "direction_errors_deg": direction_errors,
        "joint_angles_pass": max(joint_errors) <= JOINT_TOLERANCE_DEG,
        "segment_directions_pass": max(direction_errors) <= DIRECTION_TOLERANCE_DEG,
        "pass": bool(
            readback.finite_state
            and max(joint_errors) <= JOINT_TOLERANCE_DEG
            and max(direction_errors) <= DIRECTION_TOLERANCE_DEG
        ),
    }


def _wrong_role_contacts_clear(
    *,
    right_s2_contact_force_n: float,
    left_s0_contact_force_n: float,
) -> bool:
    return bool(
        math.isfinite(float(right_s2_contact_force_n))
        and math.isfinite(float(left_s0_contact_force_n))
        and float(right_s2_contact_force_n) <= FORBIDDEN_CONTACT_FORCE_N
        and float(left_s0_contact_force_n) <= FORBIDDEN_CONTACT_FORCE_N
    )


def _roles_grasped(readback: P003PhysicalReadback) -> bool:
    return bool(
        readback.finite_state
        and readback.right_s0_contact_force_n >= REQUIRED_CONTACT_FORCE_N
        and readback.left_s2_contact_force_n >= REQUIRED_CONTACT_FORCE_N
        and _wrong_role_contacts_clear(
            right_s2_contact_force_n=readback.right_s2_contact_force_n,
            left_s0_contact_force_n=readback.left_s0_contact_force_n,
        )
    )


def _tabletop_support_metrics(
    readback: P003PhysicalReadback,
) -> dict[str, object]:
    """Return geometry-only table support predicates.

    Per-contact force is deliberately not used here: resting PhysX contacts
    can flicker between adjacent substeps, while collider bottoms remain an
    exact, replayable description of whether the articulated ruler is still
    on the tabletop.  During manipulation one leaf may lift locally; final
    placement requires all three leaves to be close to the surface.
    """

    bottoms = np.asarray(readback.segment_bottom_z_m, dtype=np.float64)
    finite = bool(
        readback.finite_state
        and bottoms.shape == (3,)
        and np.all(np.isfinite(bottoms))
        and math.isfinite(float(readback.table_top_z_m))
    )
    if not finite:
        return {
            "finite": False,
            "supported_segment_count": 0,
            "operation_supported": False,
            "final_supported": False,
            "bottom_offsets_m": (math.inf, math.inf, math.inf),
        }

    offsets = bottoms - float(readback.table_top_z_m)
    penetration_clear = bool(
        float(np.min(offsets)) >= -TABLETOP_BOTTOM_PENETRATION_TOLERANCE_M
    )
    supported_count = int(
        np.count_nonzero(offsets <= TABLETOP_OPERATION_BOTTOM_TOLERANCE_M)
    )
    operation_supported = bool(
        penetration_clear
        and supported_count >= TABLETOP_MIN_SUPPORTED_SEGMENTS
    )
    final_supported = bool(
        penetration_clear
        and np.all(offsets <= TABLETOP_FINAL_BOTTOM_TOLERANCE_M)
    )
    return {
        "finite": True,
        "supported_segment_count": supported_count,
        "operation_supported": operation_supported,
        "final_supported": final_supported,
        "bottom_offsets_m": tuple(float(value) for value in offsets),
    }


def _tabletop_manipulation_supported(readback: P003PhysicalReadback) -> bool:
    return bool(_tabletop_support_metrics(readback)["operation_supported"])


def _hands_clear_of_table(readback: P003PhysicalReadback) -> bool:
    force_n = float(readback.hand_table_contact_force_n)
    return bool(
        readback.hand_table_contact_valid
        and math.isfinite(force_n)
        and force_n <= FINAL_HAND_TABLE_CONTACT_FORCE_N
    )


def _released(readback: P003PhysicalReadback) -> bool:
    return bool(
        readback.left_all_ruler_contact_force_n <= RELEASE_CONTACT_FORCE_N
        and readback.right_all_ruler_contact_force_n <= RELEASE_CONTACT_FORCE_N
    )


def _placed(readback: P003PhysicalReadback) -> bool:
    center_xy = np.asarray(
        readback.object_center_position_m[:2], dtype=np.float64
    )
    reset_xy = np.asarray(
        readback.reset_object_center_position_m[:2], dtype=np.float64
    )
    support = _tabletop_support_metrics(readback)
    return bool(
        support["final_supported"]
        and np.all(np.isfinite(center_xy))
        and np.all(np.isfinite(reset_xy))
        and float(np.linalg.norm(center_xy - reset_xy))
        <= PLACEMENT_XY_TOLERANCE_M
    )


def _nonextended_staging_geometry(readback: P003PhysicalReadback) -> bool:
    center_xy = np.asarray(
        readback.object_center_position_m[:2], dtype=np.float64
    )
    reset_xy = np.asarray(
        readback.reset_object_center_position_m[:2], dtype=np.float64
    )
    return bool(
        min(readback.segment_bottom_z_m)
        >= readback.table_top_z_m - 0.0010
        and max(readback.segment_bottom_z_m)
        <= readback.table_top_z_m + _POLICY_NONEXTENDED_STAGING_OFFSET_M
        and float(np.linalg.norm(center_xy - reset_xy))
        <= _POLICY_NONEXTENDED_STAGING_XY_TOLERANCE_M
    )


def _lifted(readback: P003PhysicalReadback) -> bool:
    return bool(
        readback.minimum_segment_bottom_z_m
        >= readback.table_top_z_m + LIFT_HEIGHT_M
    )


def _stable(readback: P003PhysicalReadback) -> bool:
    return bool(
        readback.finite_state
        and readback.maximum_linear_speed_mps <= STABLE_LINEAR_SPEED_MPS
        and readback.maximum_angular_speed_radps <= STABLE_ANGULAR_SPEED_RADPS
    )


class P003StageTracker:
    """Physical tabletop stage machine enforcing acquisition before release."""

    STAGES = (
        "seek_dual_grasp",
        "tabletop_manipulation",
        "shape_held",
        "released_stable",
        "success",
    )

    def __init__(self, target_key: str) -> None:
        if target_key not in TARGET_SHAPES:
            raise ValueError(f"unknown P003 target: {target_key}")
        self.target = TARGET_SHAPES[target_key]
        self.stage_index = 0
        self.roles_acquired = False
        self.shape_hold_count = 0
        self.release_hold_count = 0

    @property
    def stage(self) -> str:
        return self.STAGES[self.stage_index]

    @property
    def success(self) -> bool:
        return self.stage == "success"

    def update(self, readback: P003PhysicalReadback) -> dict[str, object]:
        shape = evaluate_shape(readback, self.target)
        roles = _roles_grasped(readback)
        wrong_role_contacts_clear = _wrong_role_contacts_clear(
            right_s2_contact_force_n=readback.right_s2_contact_force_n,
            left_s0_contact_force_n=readback.left_s0_contact_force_n,
        )
        support = _tabletop_support_metrics(readback)
        tabletop_supported = bool(support["operation_supported"])
        placed = _placed(readback)
        released = _released(readback)
        stable = _stable(readback)
        hands_clear_of_table = _hands_clear_of_table(readback)

        if self.stage == "seek_dual_grasp" and roles and tabletop_supported:
            self.roles_acquired = True
            self.stage_index = 1
        elif self.stage == "tabletop_manipulation":
            # Wrong-role contact and loss of endpoint ownership are ordinary,
            # recoverable manipulation errors.  They block only this frame's
            # progress and clear the consecutive target-shape hold.
            if roles and tabletop_supported and shape["pass"]:
                self.shape_hold_count = min(
                    SHAPE_HOLD_STEPS,
                    self.shape_hold_count + 1,
                )
                if self.shape_hold_count >= SHAPE_HOLD_STEPS:
                    self.stage_index = 2
            else:
                self.shape_hold_count = 0
        elif self.stage == "shape_held":
            if (
                shape["pass"]
                and tabletop_supported
                and placed
                and released
                and wrong_role_contacts_clear
            ):
                self.stage_index = 3
            elif (
                not shape["pass"]
                or not tabletop_supported
                or not wrong_role_contacts_clear
                or not roles
            ):
                # The shape hold is a recoverable latch, not permission to
                # release from a later invalid pose.  Any degradation before
                # a valid final release returns to manipulation and requires
                # fifteen fresh supported, correct-role frames.
                self.stage_index = 1
                self.shape_hold_count = 0
        elif self.stage == "released_stable":
            if (
                shape["pass"]
                and placed
                and released
                and stable
            ):
                self.release_hold_count = min(
                    RELEASE_HOLD_STEPS,
                    self.release_hold_count + 1,
                )
                if (
                    self.release_hold_count >= RELEASE_HOLD_STEPS
                    and hands_clear_of_table
                ):
                    self.stage_index = 4
            else:
                self.release_hold_count = 0

        return {
            "stage": self.stage,
            "success": self.success,
            "shape": shape,
            "roles_grasped": roles,
            "roles_acquired": self.roles_acquired,
            "wrong_role_contacts_clear": wrong_role_contacts_clear,
            # Legacy key retained for old recording/report readers.  Lifting
            # is never required by the v2 tabletop policy.
            "lifted": False,
            "tabletop_supported": tabletop_supported,
            "tabletop_support": support,
            "placed": placed,
            "released": released,
            "stable": stable,
            "hands_clear_of_table": hands_clear_of_table,
            "shape_hold_count": self.shape_hold_count,
            "release_hold_count": self.release_hold_count,
        }


def _smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _normalized_quaternion_wxyz(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("P0-03 policy received an invalid quaternion")
    return quaternion / norm


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = _normalized_quaternion_wxyz(left)
    rw, rx, ry, rz = _normalized_quaternion_wxyz(right)
    return _normalized_quaternion_wxyz(
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


def _quaternion_inverse_wxyz(value: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalized_quaternion_wxyz(value)
    return np.asarray([w, -x, -y, -z], dtype=np.float64)


def _rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = _normalized_quaternion_wxyz(quaternion)
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    w, x, y, z = q
    qv = np.asarray([x, y, z], dtype=np.float64)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)


def _blend_pose(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    start = np.asarray(start, dtype=np.float64).reshape(7)
    end = np.asarray(end, dtype=np.float64).reshape(7)
    amount = _smoothstep(alpha)
    quaternion_start = _normalized_quaternion_wxyz(start[3:7])
    quaternion_end = _normalized_quaternion_wxyz(end[3:7])
    if float(np.dot(quaternion_start, quaternion_end)) < 0.0:
        quaternion_end = -quaternion_end
    quaternion = (1.0 - amount) * quaternion_start + amount * quaternion_end
    quaternion = _normalized_quaternion_wxyz(quaternion)
    return np.concatenate(
        [(1.0 - amount) * start[:3] + amount * end[:3], quaternion]
    )


def _relative_pose(parent_pose: np.ndarray, child_pose: np.ndarray) -> np.ndarray:
    parent = np.asarray(parent_pose, dtype=np.float64).reshape(7)
    child = np.asarray(child_pose, dtype=np.float64).reshape(7)
    inverse_parent = _quaternion_inverse_wxyz(parent[3:7])
    return np.concatenate(
        [
            _rotate_wxyz(inverse_parent, child[:3] - parent[:3]),
            _quaternion_multiply_wxyz(inverse_parent, child[3:7]),
        ]
    )


def _compose_pose(parent_pose: np.ndarray, relative_pose: np.ndarray) -> np.ndarray:
    parent = np.asarray(parent_pose, dtype=np.float64).reshape(7)
    relative = np.asarray(relative_pose, dtype=np.float64).reshape(7)
    return np.concatenate(
        [
            parent[:3] + _rotate_wxyz(parent[3:7], relative[:3]),
            _quaternion_multiply_wxyz(parent[3:7], relative[3:7]),
        ]
    )


def _pose_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    rotation = np.column_stack(
        [
            _rotate_wxyz(pose[3:7], axis)
            for axis in np.eye(3, dtype=np.float64)
        ]
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pose[:3]
    return matrix


def _quaternion_wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        if diagonal_index == 0:
            scale = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            ) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif diagonal_index == 1:
            scale = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            ) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            ) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    return _normalized_quaternion_wxyz(quaternion)


def _limited_rotation_command(
    command_pose: np.ndarray,
    actual_pose: np.ndarray,
    desired_actual_rotation: np.ndarray,
    *,
    max_angle_deg: float,
    maximum_lead_deg: float | None = None,
) -> np.ndarray:
    current_rotation = _pose_matrix(actual_pose)[:3, :3]
    error_rotation = np.asarray(desired_actual_rotation) @ current_rotation.T
    cosine = float(np.clip((np.trace(error_rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(math.acos(cosine))
    if angle <= 1.0e-9:
        return np.asarray(command_pose[3:7], dtype=np.float64).copy()
    axis = np.asarray(
        [
            error_rotation[2, 1] - error_rotation[1, 2],
            error_rotation[0, 2] - error_rotation[2, 0],
            error_rotation[1, 0] - error_rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-9:
        return np.asarray(command_pose[3:7], dtype=np.float64).copy()
    axis /= axis_norm
    limited_angle = min(angle, math.radians(float(max_angle_deg)))
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    delta_rotation = (
        np.eye(3, dtype=np.float64)
        + math.sin(limited_angle) * skew
        + (1.0 - math.cos(limited_angle)) * (skew @ skew)
    )
    if maximum_lead_deg is None:
        return _quaternion_wxyz_from_rotation(delta_rotation @ current_rotation)

    # Integrate a small feed-forward correction so the IK target can lead a
    # moving grasp, but clamp that lead relative to the measured palm.  This
    # preserves the validated controller's lag compensation without allowing
    # the unbounded command drift that can detach the compliant grasp.
    command_rotation = _pose_matrix(command_pose)[:3, :3]
    candidate_rotation = delta_rotation @ command_rotation
    lead_rotation = candidate_rotation @ current_rotation.T
    lead_cosine = float(
        np.clip((np.trace(lead_rotation) - 1.0) / 2.0, -1.0, 1.0)
    )
    lead_angle = float(math.acos(lead_cosine))
    maximum_lead = math.radians(float(maximum_lead_deg))
    if lead_angle <= maximum_lead:
        return _quaternion_wxyz_from_rotation(candidate_rotation)
    lead_axis = np.asarray(
        [
            lead_rotation[2, 1] - lead_rotation[1, 2],
            lead_rotation[0, 2] - lead_rotation[2, 0],
            lead_rotation[1, 0] - lead_rotation[0, 1],
        ],
        dtype=np.float64,
    )
    lead_axis_norm = float(np.linalg.norm(lead_axis))
    if lead_axis_norm <= 1.0e-9:
        return _quaternion_wxyz_from_rotation(current_rotation)
    lead_axis /= lead_axis_norm
    lead_skew = np.asarray(
        [
            [0.0, -lead_axis[2], lead_axis[1]],
            [lead_axis[2], 0.0, -lead_axis[0]],
            [-lead_axis[1], lead_axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    bounded_lead_rotation = (
        np.eye(3, dtype=np.float64)
        + math.sin(maximum_lead) * lead_skew
        + (1.0 - math.cos(maximum_lead)) * (lead_skew @ lead_skew)
    )
    return _quaternion_wxyz_from_rotation(
        bounded_lead_rotation @ current_rotation
    )


class P003ScriptedPolicy:
    """Table-supported bimanual policy used for physical collection trials.

    The policy approaches the two outer leaves from above, acquires the
    endpoint roles, and then moves both palms only in the table plane.  It
    never commands an airborne shaping phase and does not rely on object
    tethers or moving articulation targets.
    """

    def __init__(
        self,
        reset_state: P003ResetState,
        *,
        right_home_pose: np.ndarray,
        left_home_pose: np.ndarray,
        initial_body_poses: np.ndarray,
    ) -> None:
        self.reset_state = reset_state
        self.right_home_pose = np.asarray(right_home_pose, dtype=np.float64).reshape(7)
        self.left_home_pose = np.asarray(left_home_pose, dtype=np.float64).reshape(7)
        self.initial_body_poses = np.asarray(
            initial_body_poses, dtype=np.float64
        ).reshape(3, 7)
        for pose in (
            self.right_home_pose,
            self.left_home_pose,
            *self.initial_body_poses,
        ):
            _normalized_quaternion_wxyz(pose[3:7])
        self.phase = "clearance"
        self._phase_start_step = 0
        self._right_relative_pose: np.ndarray | None = None
        self._left_relative_pose: np.ndarray | None = None
        self._right_retreat_start: np.ndarray | None = None
        self._left_retreat_start: np.ndarray | None = None
        self._last_action: np.ndarray | None = None
        self._release_open_start_step: int | None = None
        self._release_finished = False

    @staticmethod
    def _role_contacts(readback: P003PhysicalReadback | None) -> bool:
        return bool(readback is not None and _roles_grasped(readback))

    @staticmethod
    def _roles_released(readback: P003PhysicalReadback | None) -> bool:
        return bool(readback is not None and _released(readback))

    def _set_phase(self, phase: str, step: int) -> None:
        if phase != self.phase:
            self.phase = phase
            self._phase_start_step = int(step)

    @property
    def configure_preparing(self) -> bool:
        return False

    @property
    def workspace_preparing(self) -> bool:
        return False

    def _pregrasp_pose(self, side: str, *, high: bool) -> np.ndarray:
        body = self.initial_body_poses[0 if side == "right" else 2]
        home = self.right_home_pose if side == "right" else self.left_home_pose
        pose = home.copy()
        pose[:2] = body[:2] + (
            _POLICY_RIGHT_CONTACT_XY_OFFSET_M
            if side == "right"
            else _POLICY_LEFT_CONTACT_XY_OFFSET_M
        )
        pose[3:7] = (
            _POLICY_RIGHT_CONTACT_QUATERNION_WXYZ
            if side == "right"
            else _POLICY_LEFT_CONTACT_QUATERNION_WXYZ
        )
        pose[2] = (
            max(_POLICY_HIGH_Z_M, float(home[2]))
            if high
            else float(body[2]) + _POLICY_WRIST_ABOVE_BODY_M
        )
        if high:
            pose[0] -= 0.030
        elif side == "left":
            pose[2] = float(body[2]) + 0.085
        return pose

    def _capture_grasp_transforms(
        self,
        body_poses: np.ndarray,
        right_tcp_pose: np.ndarray,
        left_tcp_pose: np.ndarray,
    ) -> None:
        self._right_relative_pose = _relative_pose(body_poses[0], right_tcp_pose)
        self._left_relative_pose = _relative_pose(body_poses[2], left_tcp_pose)

    @staticmethod
    def _yaw_rotated_pose(
        pose: np.ndarray,
        *,
        local_direction: np.ndarray,
        desired_world_xy: np.ndarray,
    ) -> np.ndarray:
        """Rotate one body pose about world Z to align its planar axis."""

        result = np.asarray(pose, dtype=np.float64).reshape(7).copy()
        current_direction = _rotate_wxyz(result[3:7], local_direction)[:2]
        current_norm = float(np.linalg.norm(current_direction))
        desired = np.asarray(desired_world_xy, dtype=np.float64).reshape(2)
        desired_norm = float(np.linalg.norm(desired))
        if current_norm <= 1.0e-9 or desired_norm <= 1.0e-9:
            return result
        current_angle = math.atan2(
            float(current_direction[1]), float(current_direction[0])
        )
        desired_angle = math.atan2(float(desired[1]), float(desired[0]))
        delta = math.atan2(
            math.sin(desired_angle - current_angle),
            math.cos(desired_angle - current_angle),
        )
        yaw = np.asarray(
            [math.cos(0.5 * delta), 0.0, 0.0, math.sin(0.5 * delta)],
            dtype=np.float64,
        )
        result[3:7] = _quaternion_multiply_wxyz(yaw, result[3:7])
        return result

    def _target_tabletop_body_poses(
        self,
        body_poses: np.ndarray,
        readback: P003PhysicalReadback,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct target S0/S2 poses while keeping the ruler on the table."""

        poses = np.asarray(body_poses, dtype=np.float64).reshape(3, 7)
        lengths = np.asarray([0.249598, 0.249607, 0.224654], dtype=np.float64)
        weights = lengths / float(np.sum(lengths))

        s0_world = _rotate_wxyz(
            poses[0, 3:7], np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        )[:2]
        s0_norm = float(np.linalg.norm(s0_world))
        if s0_norm <= 1.0e-9:
            raise ValueError("P0-03 policy received a degenerate S0 direction")
        base_angle = math.atan2(float(s0_world[1]), float(s0_world[0]))
        cosine = math.cos(base_angle)
        sine = math.sin(base_angle)
        task_to_world = np.asarray(
            [[cosine, -sine], [sine, cosine]], dtype=np.float64
        )
        target_directions = np.asarray(
            self.reset_state.target.segment_directions_task_xy,
            dtype=np.float64,
        ) @ task_to_world.T
        target_directions /= np.linalg.norm(
            target_directions, axis=1, keepdims=True
        )

        # Express all segment centers from hinge 1, then choose hinge 1 so the
        # target's length-weighted center remains at the reset center.  Both
        # commanded palms therefore perform bounded in-plane manipulation;
        # no body or hand is lifted to make room for the target.
        relative_centers = np.stack(
            [
                0.5 * lengths[0] * target_directions[0],
                0.5 * lengths[1] * target_directions[1],
                lengths[1] * target_directions[1]
                + 0.5 * lengths[2] * target_directions[2],
            ],
            axis=0,
        )
        desired_center = np.asarray(
            readback.reset_object_center_position_m[:2], dtype=np.float64
        )
        hinge_1_xy = desired_center - np.average(
            relative_centers, axis=0, weights=weights
        )

        desired_s0 = self._yaw_rotated_pose(
            poses[0],
            local_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            desired_world_xy=target_directions[0],
        )
        desired_s2 = self._yaw_rotated_pose(
            poses[2],
            local_direction=np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
            desired_world_xy=target_directions[2],
        )
        desired_s0[:2] = hinge_1_xy + relative_centers[0]
        desired_s2[:2] = hinge_1_xy + relative_centers[2]
        # Preserve measured tabletop height.  Only XY and yaw may change.
        desired_s0[2] = poses[0, 2]
        desired_s2[2] = poses[2, 2]
        return desired_s0, desired_s2

    @staticmethod
    def _bounded_tabletop_palm_command(
        base_command: np.ndarray,
        actual_pose: np.ndarray,
        desired_pose: np.ndarray,
    ) -> np.ndarray:
        """Take one bounded XY/yaw step while holding command height fixed."""

        command = np.asarray(base_command, dtype=np.float64).reshape(7).copy()
        actual = np.asarray(actual_pose, dtype=np.float64).reshape(7)
        desired = np.asarray(desired_pose, dtype=np.float64).reshape(7)
        command[:2] = actual[:2] + np.clip(
            desired[:2] - actual[:2], -0.0015, 0.0015
        )
        command[2] = float(base_command[2])
        command[3:7] = _limited_rotation_command(
            command,
            actual,
            _pose_matrix(desired)[:3, :3],
            max_angle_deg=1.5,
            maximum_lead_deg=4.0,
        )
        return command

    def action(
        self,
        step: int,
        *,
        readback: P003PhysicalReadback | None,
        stage: str,
        body_poses: np.ndarray,
        right_tcp_pose: np.ndarray,
        left_tcp_pose: np.ndarray,
    ) -> np.ndarray:
        """Return one table-supported endpoint manipulation command."""

        step = max(0, int(step))
        body_poses = np.asarray(body_poses, dtype=np.float64).reshape(3, 7)
        right_tcp_pose = np.asarray(right_tcp_pose, dtype=np.float64).reshape(7)
        left_tcp_pose = np.asarray(left_tcp_pose, dtype=np.float64).reshape(7)
        right_high = self._pregrasp_pose("right", high=True)
        left_high = self._pregrasp_pose("left", high=True)
        right_contact = self._pregrasp_pose("right", high=False)
        left_contact = self._pregrasp_pose("left", high=False)
        right_home_high = self.right_home_pose.copy()
        left_home_high = self.left_home_pose.copy()
        right_home_high[2] = max(_POLICY_HIGH_Z_M, float(right_home_high[2]))
        left_home_high[2] = max(_POLICY_HIGH_Z_M, float(left_home_high[2]))

        if self.phase == "clearance" and step >= _POLICY_CLEARANCE_END_STEP:
            self._set_phase("approach", step)
        if self.phase == "approach" and step >= _POLICY_APPROACH_END_STEP:
            self._set_phase("descend", step)
        if self.phase == "descend" and (
            step >= _POLICY_DESCEND_END_STEP or self._role_contacts(readback)
        ):
            self._set_phase("grasp", step)
        if (
            self.phase == "grasp"
            and self._role_contacts(readback)
            and (
                self._right_relative_pose is None
                or self._left_relative_pose is None
            )
        ):
            # Capture the measured palms on the first real dual-role contact.
            # Contact on a millimetre-scale leaf may flicker while the fingers
            # finish closing; the stage history keeps that legitimate capture
            # from being discarded without synthesizing a tether or force.
            self._capture_grasp_transforms(
                body_poses,
                right_tcp_pose,
                left_tcp_pose,
            )
        if (
            self.phase == "grasp"
            and step - self._phase_start_step >= _POLICY_SECURE_STEPS
            and stage != "seek_dual_grasp"
            and self._right_relative_pose is not None
            and self._left_relative_pose is not None
        ):
            self._set_phase("configure", step)
        release_stage_ready = bool(
            stage in {"released_stable", "success"}
            or (
                stage == "shape_held"
                and readback is not None
                and _placed(readback)
            )
        )
        if self.phase == "configure" and release_stage_ready:
            self._release_open_start_step = step
            self._set_phase("release", step)
        if self.phase == "release":
            release_start = (
                self._release_open_start_step
                if self._release_open_start_step is not None
                else step
            )
            release_age = step - int(release_start)
            if (
                (self._roles_released(readback) and release_age >= _POLICY_RELEASE_STEPS)
                or release_age >= 2 * _POLICY_RELEASE_STEPS
            ):
                self._release_finished = True
                self._right_retreat_start = right_tcp_pose.copy()
                self._left_retreat_start = left_tcp_pose.copy()
                self._set_phase("retreat", step)

        right_pose = self.right_home_pose.copy()
        left_pose = self.left_home_pose.copy()
        grip_alpha = 0.0
        release_alpha: float | None = None

        if self.phase == "clearance":
            alpha = step / max(1, _POLICY_CLEARANCE_END_STEP)
            right_pose = _blend_pose(self.right_home_pose, right_home_high, alpha)
            left_pose = _blend_pose(self.left_home_pose, left_home_high, alpha)
        elif self.phase == "approach":
            alpha = (step - _POLICY_CLEARANCE_END_STEP) / max(
                1, _POLICY_APPROACH_END_STEP - _POLICY_CLEARANCE_END_STEP
            )
            right_pose = _blend_pose(right_home_high, right_high, alpha)
            left_pose = _blend_pose(left_home_high, left_high, alpha)
        elif self.phase == "descend":
            alpha = (step - _POLICY_APPROACH_END_STEP) / max(
                1, _POLICY_DESCEND_END_STEP - _POLICY_APPROACH_END_STEP
            )
            right_pose = _blend_pose(right_high, right_contact, alpha)
            left_pose = _blend_pose(left_high, left_contact, alpha)
        elif self.phase == "grasp":
            right_pose = right_contact.copy()
            left_pose = left_contact.copy()
            grip_alpha = np.clip(
                (step - self._phase_start_step) / max(1, _POLICY_GRIP_RAMP_STEPS),
                0.0,
                1.0,
            )
        elif self.phase == "configure":
            if self._last_action is None:
                right_pose = right_tcp_pose.copy()
                left_pose = left_tcp_pose.copy()
            else:
                right_pose = self._last_action[:7].copy()
                left_pose = self._last_action[7:14].copy()
            grip_alpha = 1.0
            if (
                readback is not None
                and self._right_relative_pose is not None
                and self._left_relative_pose is not None
                and _wrong_role_contacts_clear(
                    right_s2_contact_force_n=readback.right_s2_contact_force_n,
                    left_s0_contact_force_n=readback.left_s0_contact_force_n,
                )
                and _tabletop_manipulation_supported(readback)
            ):
                desired_s0, desired_s2 = self._target_tabletop_body_poses(
                    body_poses, readback
                )
                desired_right = _compose_pose(
                    desired_s0, self._right_relative_pose
                )
                desired_left = _compose_pose(
                    desired_s2, self._left_relative_pose
                )
                right_pose = self._bounded_tabletop_palm_command(
                    right_pose, right_tcp_pose, desired_right
                )
                left_pose = self._bounded_tabletop_palm_command(
                    left_pose, left_tcp_pose, desired_left
                )
        elif self.phase == "release":
            if self._last_action is not None:
                right_pose = self._last_action[:7].copy()
                left_pose = self._last_action[7:14].copy()
            else:
                right_pose = right_tcp_pose.copy()
                left_pose = left_tcp_pose.copy()
            release_start = (
                self._release_open_start_step
                if self._release_open_start_step is not None
                else step
            )
            release_age = step - int(release_start)
            release_alpha = _smoothstep(
                release_age / max(1, _POLICY_RELEASE_STEPS)
            )
        else:
            right_pose = (
                self._right_retreat_start.copy()
                if self._right_retreat_start is not None
                else right_tcp_pose.copy()
            )
            left_pose = (
                self._left_retreat_start.copy()
                if self._left_retreat_start is not None
                else left_tcp_pose.copy()
            )
            retreat_height = min(
                0.10, 0.004 + 0.002 * (step - self._phase_start_step)
            )
            right_pose[2] += retreat_height
            left_pose[2] += retreat_height
            release_alpha = 1.0

        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[:7] = right_pose
        action[7:14] = left_pose
        grip_alpha = _smoothstep(grip_alpha)
        right_grip = WUJI_RULER_SECURE_GRIP_RIGHT
        left_grip = WUJI_RULER_SECURE_GRIP_LEFT
        if release_alpha is None:
            action[14 : 14 + HAND_ACTION_DIM] = grip_alpha * right_grip
            action[14 + HAND_ACTION_DIM :] = grip_alpha * left_grip
        else:
            action[14 : 14 + HAND_ACTION_DIM] = (
                (1.0 - release_alpha) * right_grip
                + release_alpha * WUJI_RULER_RELEASE_OPEN_RIGHT
            )
            action[14 + HAND_ACTION_DIM :] = (
                (1.0 - release_alpha) * left_grip
                + release_alpha * WUJI_RULER_RELEASE_OPEN_LEFT
            )
        self._last_action = action.copy()
        return action

    def metadata(self) -> Mapping[str, object]:
        return {
            "policy": "p003_scripted_bimanual_ruler_tabletop_v2",
            "runtime_policy_version": P003_RUNTIME_POLICY_VERSION,
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "reset": self.reset_state.to_dict(),
            "target_key": self.reset_state.target_key,
            "target_joint_angles_deg": list(
                self.reset_state.target.joint_targets_deg
            ),
            "action_dim": ACTION_DIM,
            "left_role": "S2",
            "right_role": "S0",
            "object_control": (
                "raw endpoint contact plus table-supported in-plane palm "
                "motion; no compliant tether, moving joint target, or "
                "object-state write"
            ),
        }
