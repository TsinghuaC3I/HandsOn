"""CPU-safe semantic contract for P0-05.

The live environment must populate :class:`P005PhysicalReadback` from PhysX
pose, joint, velocity, and contact readback.  Unlock progress is valid only
while the left hand physically holds a lifted padlock and the right hand owns
an inserted key. The U-shaped shackle pops up automatically after a clockwise
key turn; the left hand does not re-grasp it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import math

import numpy as np


TASK_ID = "P0-05"
TASK_NAME = "p005_key_insert_turn_unlock_open_shackle"
TASK_PROMPT = (
    "Use the left hand to pick up and hold the padlock body. Use the right "
    "hand to pick up the key, insert it into the keyhole on the side of the "
    "padlock, and turn it clockwise until the shackle pops up."
)

CONTROL_HZ = 30
PHYSICS_HZ = 120
TIMEOUT_STEPS = 1500
RENDER_RESOLUTION_HW = (480, 640)
ACTION_DIM = 54
HAND_ACTION_DIM = 20

KEY_BODY_NAME = "p005_key_root_body"
PADLOCK_BODY_NAME = "p005_padlock_body"
SHACKLE_BODY_NAME = "p005_shackle"
LOCK_CYLINDER_BODY_NAME = "p005_lock_cylinder"
LOCK_CYLINDER_DOF = "p005_lock_cylinder_yaw"
SHACKLE_SLIDE_DOF = "p005_shackle_slide_joint"
SHACKLE_HINGE_DOF = "p005_shackle_hinge_joint"
EXPECTED_DOF_NAMES = frozenset(
    {LOCK_CYLINDER_DOF, SHACKLE_SLIDE_DOF, SHACKLE_HINGE_DOF}
)

ASSET_SCALE = 1.20
TABLE_TOP_Z_M = 0.89
PADLOCK_ROOT_XYZ_M = np.asarray([-0.30, 0.19, 0.932900000], dtype=np.float64)
# The YCB padlock is operated through its local +X side face.  Keep these
# frames shared by the MJCF, live geometry readback, and the acceptance policy
# so the visible keyway and the functional cylinder cannot drift apart again.
PADLOCK_KEYHOLE_LOCAL_M = np.asarray(
    [0.0294, 0.0, -0.00972], dtype=np.float64
)
PADLOCK_KEYWAY_INWARD_LOCAL_M = np.asarray(
    [-1.0, 0.0, 0.0], dtype=np.float64
)
KEY_TIP_LOCAL_M = np.asarray(
    [-0.001164444, -0.030426667, -0.000888889], dtype=np.float64
)
KEY_AXIS_LOCAL_M = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
PADLOCK_ROOT_QUATERNION_WXYZ = np.asarray(
    [1.0, 0.0, 0.0, 0.0], dtype=np.float64
)
# The primary YCB key blade, its existing rear ring, and the padlock use
# task-local 1.2x geometry. The interfering spare blade is not instantiated.
KEY_RING_CENTER_LOCAL_M = np.asarray(
    [-0.0013050, 0.0216840, 0.0000300], dtype=np.float64
)
# Stable support of the single operable key: the source spare blade no longer
# props up the assembly. Keep 0.2 mm above its three-point support plane.
KEY_REST_ROOT_Z_M = 0.8982965499339415
KEY_REST_QUATERNION_WXYZ = np.asarray(
    [
        0.9533049708549274,
        0.050251833174530844,
        -0.2929284952151633,
        -0.05364030664402749,
    ],
    dtype=np.float64,
)
KEY_NOMINAL_XYZ_M = np.asarray(
    [-0.30, -0.20, KEY_REST_ROOT_Z_M], dtype=np.float64
)
KEY_XY_HALF_RANGE_M = np.asarray([0.015, 0.020], dtype=np.float64)
KEY_YAW_HALF_RANGE_DEG = 10.0
# Keep the two free objects in the rear tabletop lane. The dual-ROKAE home
# palms are at x=-0.10, leaving deliberate clearance from the hands while
# keeping both objects inside the verified P0-05 reach envelope.
KEY_XY_BOUNDS_M = np.asarray(
    [[-0.46, -0.34], [-0.28, -0.10]], dtype=np.float64
)
PADLOCK_XY_BOUNDS_M = np.asarray(
    [[-0.46, -0.34], [0.10, 0.28]], dtype=np.float64
)
RESET_MIN_HAND_CLEARANCE_M = 0.22
RESET_MIN_OBJECT_CLEARANCE_M = 0.18
PADLOCK_LIFT_MIN_M = 0.035


def key_reset_quaternion_wxyz(yaw_deg: float) -> np.ndarray:
    """Compose world-Z reset yaw with the key ring's stable support pose."""

    half_yaw = math.radians(float(yaw_deg)) / 2.0
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    w, x, y, z = KEY_REST_QUATERNION_WXYZ
    quaternion = np.asarray(
        [
            cy * w - sy * z,
            cy * x - sy * y,
            cy * y + sy * x,
            cy * z + sy * w,
        ],
        dtype=np.float64,
    )
    return quaternion / np.linalg.norm(quaternion)

KEY_INSERTION_DEPTH_M = 0.012
KEY_INSERTION_RELEASE_DEPTH_M = 0.009
KEY_LATERAL_ERROR_M = 0.00804
KEY_AXIS_ERROR_DEG = 12.0
# Physical cylinder contact remains continuous across small keyway edge
# deflections; task completion still uses the stricter 12-degree gate above.
KEY_COUPLING_AXIS_ERROR_DEG = 15.5
KEY_TURN_TARGET_DEG = 55.0
KEY_TURN_RELEASE_DEG = 45.0
SHACKLE_SLIDE_TARGET_M = 0.01596
# The locked shackle's compliant PhysX drive can move roughly 5 mm when the
# operator first contacts the loose key.  Treat only a sustained, deliberate
# pre-unlock pull as an ordering violation instead of poisoning the episode on
# that recoverable contact transient.
SHACKLE_PREUNLOCK_MAX_M = 0.00888
SHACKLE_PREUNLOCK_HOLD_STEPS = 3
REQUIRED_CONTACT_FORCE_N = 0.05
ROLE_CONTACT_GRACE_STEPS = 3
INSERTION_HOLD_STEPS = 4
TURN_HOLD_STEPS = 4
SUCCESS_HOLD_STEPS = 12
STABLE_SHACKLE_LINEAR_SPEED_MPS = 0.04
STABLE_SHACKLE_ANGULAR_SPEED_RADPS = 0.60
MAX_PENETRATION_M = 0.0037
EXPLOSION_LINEAR_SPEED_MPS = 5.0
# A small loose key can briefly spin faster than 20 rad/s under a perfectly
# ordinary fingertip contact.  Reserve the explosion gate for genuinely
# unstable motion; the live runtime also requires a full control interval of
# continuous failure before terminating the episode.
EXPLOSION_ANGULAR_SPEED_RADPS = 60.0
FAILURE_DWELL_SUBSTEPS = 4
KEY_DROP_MARGIN_M = 0.015

LOCK_CYLINDER_STIFFNESS = 1.2
LOCK_CYLINDER_DAMPING = 0.08
LOCK_CYLINDER_MAX_FORCE = 2.0
LOCKED_SHACKLE_SLIDE_STIFFNESS = 30000.0
LOCKED_SHACKLE_HINGE_STIFFNESS = 1200.0
LOCKED_SHACKLE_DAMPING = 25.0
LOCKED_SHACKLE_MAX_FORCE = 250.0
UNLOCKED_SHACKLE_DAMPING = 8.0
UNLOCKED_SHACKLE_MAX_FORCE = 12.0
UNLOCKED_SHACKLE_SLIDE_STIFFNESS = 800.0
UNLOCKED_SHACKLE_HINGE_STIFFNESS = 40.0
UNLOCKED_SHACKLE_TARGET_M = 0.02136


class P005Phase(IntEnum):
    SEEK_KEY = 0
    KEY_INSERTED = 1
    KEY_TURNED = 2
    SHACKLE_OPEN = 3


@dataclass(frozen=True)
class P005ResetState:
    seed: int
    key_x_m: float
    key_y_m: float
    key_z_m: float
    key_yaw_deg: float
    padlock_x_m: float
    padlock_y_m: float
    padlock_z_m: float
    key_friction: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _sample_reset_xy(
    rng: np.random.Generator,
    bounds: np.ndarray,
    *,
    forbidden_centers: tuple[np.ndarray, ...] = (),
    occupied_centers: tuple[np.ndarray, ...] = (),
    min_clearance_m: float = 0.0,
) -> np.ndarray:
    """Sample a tabletop XY point while respecting hand/object clearances."""

    centers = []
    for center in (*forbidden_centers, *occupied_centers):
        value = np.asarray(center, dtype=np.float64).reshape(-1)[:2]
        if value.shape == (2,) and np.isfinite(value).all():
            centers.append(value)

    best_candidate: np.ndarray | None = None
    best_clearance = -math.inf
    for _ in range(256):
        candidate = rng.uniform(bounds[:, 0], bounds[:, 1])
        clearance = (
            min(
                float(np.linalg.norm(candidate - center))
                for center in centers
            )
            if centers
            else math.inf
        )
        if clearance > best_clearance:
            best_candidate = candidate
            best_clearance = clearance
        if clearance >= float(min_clearance_m):
            return candidate

    # The current P0-05 bounds have ample room, but retain a deterministic
    # best-effort fallback if a future robot home pose makes the rejection
    # region unusually large.
    if best_candidate is None:
        raise RuntimeError("P0-05 reset sampler produced no candidate")
    return best_candidate


def sample_reset_state(
    seed: int,
    *,
    forbidden_centers: tuple[np.ndarray, ...] = (),
) -> P005ResetState:
    """Return a reachable, non-successful deterministic key/padlock reset."""

    normalized_seed = int(seed) % (2**32 - 1)
    rng = np.random.default_rng(normalized_seed)
    padlock_xy = _sample_reset_xy(
        rng,
        PADLOCK_XY_BOUNDS_M,
        forbidden_centers=forbidden_centers,
        min_clearance_m=RESET_MIN_HAND_CLEARANCE_M,
    )
    key_xy = _sample_reset_xy(
        rng,
        KEY_XY_BOUNDS_M,
        forbidden_centers=forbidden_centers,
        occupied_centers=(padlock_xy,),
        min_clearance_m=RESET_MIN_OBJECT_CLEARANCE_M,
    )
    return P005ResetState(
        seed=normalized_seed,
        key_x_m=float(key_xy[0]),
        key_y_m=float(key_xy[1]),
        key_z_m=float(KEY_NOMINAL_XYZ_M[2]),
        key_yaw_deg=float(rng.uniform(-KEY_YAW_HALF_RANGE_DEG, KEY_YAW_HALF_RANGE_DEG)),
        padlock_x_m=float(padlock_xy[0]),
        padlock_y_m=float(padlock_xy[1]),
        padlock_z_m=float(PADLOCK_ROOT_XYZ_M[2]),
        key_friction=float(rng.uniform(0.95, 1.15)),
    )


@dataclass(frozen=True)
class P005PhysicalReadback:
    key_insertion_depth_m: float
    key_lateral_error_m: float
    key_axis_error_deg: float
    lock_cylinder_angle_deg: float
    shackle_slide_m: float
    shackle_hinge_deg: float
    shackle_linear_speed_mps: float
    shackle_angular_speed_radps: float
    key_linear_speed_mps: float
    key_angular_speed_radps: float
    key_bottom_z_m: float
    padlock_lift_m: float
    right_key_contact_force_n: float
    left_padlock_contact_force_n: float
    left_shackle_contact_force_n: float
    right_shackle_contact_force_n: float
    # Backward-compatible observation/replay slot. The live runtime no longer
    # creates a robot/table sensor and this value is not a success/failure gate.
    robot_table_contact_force_n: float
    max_penetration_m: float = 0.0
    finite_state: bool = True


def _all_finite(readback: P005PhysicalReadback) -> bool:
    values = [
        readback.key_insertion_depth_m,
        readback.key_lateral_error_m,
        readback.key_axis_error_deg,
        readback.lock_cylinder_angle_deg,
        readback.shackle_slide_m,
        readback.shackle_hinge_deg,
        readback.shackle_linear_speed_mps,
        readback.shackle_angular_speed_radps,
        readback.key_linear_speed_mps,
        readback.key_angular_speed_radps,
        readback.key_bottom_z_m,
        readback.padlock_lift_m,
        readback.right_key_contact_force_n,
        readback.left_padlock_contact_force_n,
        readback.left_shackle_contact_force_n,
        readback.right_shackle_contact_force_n,
        readback.max_penetration_m,
    ]
    return bool(readback.finite_state and np.isfinite(values).all())


def insertion_geometry_ok(readback: P005PhysicalReadback, *, release: bool = False) -> bool:
    depth_threshold = (
        KEY_INSERTION_RELEASE_DEPTH_M if release else KEY_INSERTION_DEPTH_M
    )
    return bool(
        _all_finite(readback)
        and readback.key_insertion_depth_m >= depth_threshold
        and readback.key_lateral_error_m <= KEY_LATERAL_ERROR_M
        and readback.key_axis_error_deg <= KEY_AXIS_ERROR_DEG
    )


def cylinder_coupling_geometry_ok(readback: P005PhysicalReadback) -> bool:
    """Return whether the inserted key can physically drive the cylinder."""

    return bool(
        _all_finite(readback)
        and readback.key_insertion_depth_m >= KEY_INSERTION_RELEASE_DEPTH_M
        and readback.key_lateral_error_m <= KEY_LATERAL_ERROR_M
        and readback.key_axis_error_deg <= KEY_COUPLING_AXIS_ERROR_DEG
    )


class P005StageTracker:
    """Persistent ordered latches with physical role and geometry gates."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.phase = P005Phase.SEEK_KEY
        self.insertion_steps = 0
        self.turn_steps = 0
        self.success_steps = 0
        self.right_key_contact_grace = 0
        self.left_padlock_contact_grace = 0
        self.right_key_contact_seen = False
        self.left_padlock_contact_seen = False
        self.key_inserted_seen = False
        self.key_turned_seen = False
        self.shackle_open_seen = False
        self.preunlock_shackle_motion = False
        self.preunlock_shackle_motion_steps = 0

    @staticmethod
    def _update_grace(current: int, active: bool) -> int:
        if active:
            return ROLE_CONTACT_GRACE_STEPS
        return max(0, int(current) - 1)

    def update(self, readback: P005PhysicalReadback) -> P005Phase:
        if not _all_finite(readback):
            self.success_steps = 0
            return self.phase

        right_key_now = (
            readback.right_key_contact_force_n >= REQUIRED_CONTACT_FORCE_N
        )
        left_padlock_now = (
            readback.left_padlock_contact_force_n >= REQUIRED_CONTACT_FORCE_N
        )
        self.right_key_contact_grace = self._update_grace(
            self.right_key_contact_grace, right_key_now
        )
        self.left_padlock_contact_grace = self._update_grace(
            self.left_padlock_contact_grace, left_padlock_now
        )
        self.right_key_contact_seen |= right_key_now
        self.left_padlock_contact_seen |= left_padlock_now

        preunlock_motion_now = bool(
            not self.key_turned_seen
            and readback.shackle_slide_m > SHACKLE_PREUNLOCK_MAX_M
        )
        self.preunlock_shackle_motion_steps = (
            self.preunlock_shackle_motion_steps + 1
            if preunlock_motion_now
            else 0
        )
        if (
            self.preunlock_shackle_motion_steps
            >= SHACKLE_PREUNLOCK_HOLD_STEPS
        ):
            self.preunlock_shackle_motion = True

        insertion_now = bool(
            self.right_key_contact_grace > 0
            and self.left_padlock_contact_grace > 0
            and readback.padlock_lift_m >= PADLOCK_LIFT_MIN_M
            and insertion_geometry_ok(readback)
        )
        if not self.key_inserted_seen:
            self.insertion_steps = self.insertion_steps + 1 if insertion_now else 0
            if self.insertion_steps >= INSERTION_HOLD_STEPS:
                self.key_inserted_seen = True
                self.phase = P005Phase.KEY_INSERTED

        inserted_for_turn = bool(
            self.key_inserted_seen
            and self.right_key_contact_grace > 0
            and self.left_padlock_contact_grace > 0
            and readback.padlock_lift_m >= PADLOCK_LIFT_MIN_M
            and insertion_geometry_ok(readback, release=True)
        )
        turn_now = bool(
            inserted_for_turn
            and readback.lock_cylinder_angle_deg >= KEY_TURN_TARGET_DEG
        )
        if self.key_inserted_seen and not self.key_turned_seen:
            self.turn_steps = self.turn_steps + 1 if turn_now else 0
            if self.turn_steps >= TURN_HOLD_STEPS:
                self.key_turned_seen = True
                self.phase = P005Phase.KEY_TURNED

        key_still_unlocked = bool(
            self.key_turned_seen
            and self.right_key_contact_grace > 0
            and self.left_padlock_contact_grace > 0
            and readback.padlock_lift_m >= PADLOCK_LIFT_MIN_M
            and insertion_geometry_ok(readback, release=True)
            and readback.lock_cylinder_angle_deg >= KEY_TURN_RELEASE_DEG
        )
        if (
            key_still_unlocked
            and readback.shackle_slide_m >= SHACKLE_SLIDE_TARGET_M
        ):
            self.shackle_open_seen = True
            self.phase = P005Phase.SHACKLE_OPEN

        success_now = bool(
            self.shackle_open_seen
            and key_still_unlocked
            and abs(readback.shackle_linear_speed_mps)
            <= STABLE_SHACKLE_LINEAR_SPEED_MPS
            and abs(readback.shackle_angular_speed_radps)
            <= STABLE_SHACKLE_ANGULAR_SPEED_RADPS
            and not self.preunlock_shackle_motion
        )
        self.success_steps = self.success_steps + 1 if success_now else 0
        return self.phase

    @property
    def success(self) -> bool:
        return self.success_steps >= SUCCESS_HOLD_STEPS

    @property
    def shackle_unlock_allowed(self) -> bool:
        """Unlock only after insertion has completed and the key was turned."""

        return bool(self.key_inserted_seen and self.key_turned_seen)

    def to_dict(self) -> dict[str, bool | int | str]:
        return {
            "phase": self.phase.name.lower(),
            "phase_code": int(self.phase),
            "insertion_steps": int(self.insertion_steps),
            "turn_steps": int(self.turn_steps),
            "success_steps": int(self.success_steps),
            "right_key_contact_seen": self.right_key_contact_seen,
            "left_padlock_contact_seen": self.left_padlock_contact_seen,
            "key_inserted_seen": self.key_inserted_seen,
            "key_turned_seen": self.key_turned_seen,
            "shackle_open_seen": self.shackle_open_seen,
            "preunlock_shackle_motion": self.preunlock_shackle_motion,
            "preunlock_shackle_motion_steps": int(
                self.preunlock_shackle_motion_steps
            ),
            "success": self.success,
        }


def evaluate_physical_readback(
    readback: P005PhysicalReadback,
    tracker: P005StageTracker,
) -> dict[str, float | bool | int | str]:
    finite = _all_finite(readback)
    inserted = insertion_geometry_ok(readback)
    inserted_release = insertion_geometry_ok(readback, release=True)
    key_turned = readback.lock_cylinder_angle_deg >= KEY_TURN_TARGET_DEG
    key_turn_held = readback.lock_cylinder_angle_deg >= KEY_TURN_RELEASE_DEG
    shackle_open = readback.shackle_slide_m >= SHACKLE_SLIDE_TARGET_M
    shackle_stable = bool(
        abs(readback.shackle_linear_speed_mps) <= STABLE_SHACKLE_LINEAR_SPEED_MPS
        and abs(readback.shackle_angular_speed_radps)
        <= STABLE_SHACKLE_ANGULAR_SPEED_RADPS
    )
    penetration_ok = readback.max_penetration_m <= MAX_PENETRATION_M
    key_not_dropped = readback.key_bottom_z_m >= TABLE_TOP_Z_M - KEY_DROP_MARGIN_M
    exploded = bool(
        abs(readback.key_linear_speed_mps) > EXPLOSION_LINEAR_SPEED_MPS
        or abs(readback.shackle_linear_speed_mps) > EXPLOSION_LINEAR_SPEED_MPS
        or abs(readback.key_angular_speed_radps) > EXPLOSION_ANGULAR_SPEED_RADPS
        or abs(readback.shackle_angular_speed_radps) > EXPLOSION_ANGULAR_SPEED_RADPS
    )
    failed = bool(
        not finite
        or tracker.preunlock_shackle_motion
        or not penetration_ok
        or not key_not_dropped
        or exploded
    )
    return {
        "finite": finite,
        "inserted": inserted,
        "inserted_release_gate": inserted_release,
        "key_turned": key_turned,
        "key_turn_held": key_turn_held,
        "shackle_open": shackle_open,
        "shackle_stable": shackle_stable,
        "penetration_ok": penetration_ok,
        "robot_table_contact_gating_enabled": False,
        "key_not_dropped": key_not_dropped,
        "exploded": exploded,
        "failed": failed,
        "success_now": bool(tracker.success and not failed),
        **tracker.to_dict(),
    }


def angle_between_degrees(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 1e-12:
        return math.inf
    dot = float(np.clip(np.dot(a, b) / norm, -1.0, 1.0))
    return math.degrees(math.acos(dot))
