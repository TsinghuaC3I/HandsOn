"""Pure-Python semantics for D20 ``d20_microwave_hot_dog_plating``.

This module owns the ordered cook/plating state machine and every geometric
predicate for the task.  It imports no Isaac module so the complete event
chain can be unit tested on CPU from synthetic readings.  The IsaacLab runtime
(:mod:`d_long_horizon_env`) only gathers live PhysX readings, forwards them
here, and applies the visual/assist side effects.

Scene layout (world frame, table top z = 0.89):

* ``d20_microwave``: selected Fantom Matter shell facing -X, with a real door
  hinge and independent power/time rotary joints. Timer zero is off.
* ``d20_hot_dog``: selected textured mesh with source-derived collision.
* ``d20_tongs``: free-root articulated tongs with one hinge ``d20_tongs_joint``.
* ``d20_serving_tray``: fixed board that carries the plate and the final
  table setting.
* ``d20_plate``/``d20_fork``/``d20_mustard``/``d20_mug``: free rigid bodies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Any, Mapping

import numpy as np


TASK_NAME = "d20_microwave_hot_dog_plating"

MICROWAVE = "d20_microwave"
MICROWAVE_DOOR = "d20_microwave_door"
POWER_KNOB = "d20_power_knob"
TIMER_KNOB = "d20_timer_knob"
DOOR_JOINT = "d20_door_joint"
POWER_JOINT = "d20_power_joint"
TIMER_JOINT = "d20_timer_joint"
HOT_DOG = "d20_hot_dog"
TONGS = "d20_tongs"
TONGS_MOVING_ARM = "d20_tongs_moving_arm"
TONGS_JOINT = "d20_tongs_joint"
SERVING_TRAY = "d20_serving_tray"
PLATE = "d20_plate"
FORK = "d20_fork"
MUSTARD = "d20_mustard"
MUG = "d20_mug"
TABLEWARE = (PLATE, FORK, MUSTARD, MUG)
FREE_BODIES = (HOT_DOG, TONGS, *TABLEWARE)
RELEASE_CONTACT_BODIES = (*FREE_BODIES, TONGS_MOVING_ARM)
RELEASE_CONTACT_FORCE_EPSILON = 1e-6  # Numerical zero, not a grasp-force threshold.
TASK_BODIES = frozenset(
    {MICROWAVE, HOT_DOG, TONGS, SERVING_TRAY, *TABLEWARE}
)
EXTRA_POSES = frozenset({MICROWAVE_DOOR, POWER_KNOB, TIMER_KNOB, TONGS_MOVING_ARM})

# ---------------------------------------------------------------------------
# Cook-state contract (proposal: cook_progress in [0, 1.5]; done light at 0.90;
# acceptable stop window [0.90, 1.10]; burnt above 1.20).
# ---------------------------------------------------------------------------
COOK_SECONDS = 60.0  # high-power equivalent seconds; no hidden time acceleration
DONE_PROGRESS = 0.90
MAX_OK_PROGRESS = 1.10
BURNT_PROGRESS = 1.20
# Floating point accumulation at exactly 54 s (90% of a 60 s dose) can land
# a few ulps below the mathematical boundary.  This is numerical tolerance,
# not a wider physical cook window.
COOK_PROGRESS_EPSILON = 1e-9
RELEASE_HOLD_SECONDS = 0.5

# Physical controls: positive rotation is clockwise when facing the panel.
DOOR_CLOSED_ABS = 0.08
DOOR_OPEN_FOR_LOADING = -1.00
POWER_ANGLES = np.deg2rad([0, 45, 90, 135, 180])
POWER_LABELS = ("低火", "解冻", "中火", "中高火", "高火")
# Equivalent high-power cooking rates.  The staged three-minute medium-power
# path applies a runtime dose calibration after it sees the accepted timer;
# ordinary/full-workflow medium operation retains this base calibration.
POWER_RATES = np.asarray([0.17, 0.30, 0.55, 0.77, 1.0])
TIMER_MINUTES = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35])
TIMER_ANGLES = np.deg2rad([0, 22, 43, 65, 86, 108, 130, 151, 173, 197, 222, 244, 265, 286, 307])
TIMER_OFF_ANGLE = 1e-5  # numerically zero, less than 2 ms on the one-minute scale
# The imported MJCF knob joints are intentionally very light.  These runtime
# values follow the dual-knob calibration task's passive drive/friction model,
# with a modest increase so a release does not kick the setting away.  Any
# real hand contact pauses the timer; this prevents an operator's adjustment
# from being consumed by the stitched fast-forward wait.
POWER_KNOB_DRIVE_DAMPING = 0.060
TIMER_KNOB_DRIVE_DAMPING = 0.160
KNOB_JOINT_FRICTION = 0.070
KNOB_CONTACT_FORCE = 0.15
# Deprecated compatibility constant; hand contact now pauses on one or more
# links instead of requiring a two-finger pinch.
KNOB_TWO_FINGER_COUNT = 2
POWER_LOCAL = np.asarray([0.170747, -0.14831, 0.087263])
TIMER_LOCAL = np.asarray([0.170747, -0.14831, 0.018769])

# Measured inner surfaces of the selected microwave, x=right/y=back/z=up.
CAVITY_MIN = np.asarray([-0.17490, -0.11442, -0.09653], dtype=np.float64)
CAVITY_MAX = np.asarray([0.09840, 0.12479, 0.10172], dtype=np.float64)
# Allow a few millimetres of placement error while still requiring the full
# source collider to remain inside the microwave opening.
HOT_DOG_CAVITY_MARGIN = 0.005
HOT_DOG_FLOOR_CONTACT_TOLERANCE = 0.001
HAND_CAVITY_MARGIN = 0.010

# Hot dog collider half extents in its body frame (long axis = local x).
HOT_DOG_HALF_EXTENTS = np.asarray([0.090, 0.03939, 0.02717], dtype=np.float64)
HAND_TOUCH_MARGIN = 0.020
HAND_CONTACT_FORCE = 0.15

# Tongs geometry in the tongs root frame (see the MJCF body).
TONGS_OPEN_RAD = 0.32
TONGS_HOLD_RAD = 0.13
TONGS_CLOSED_RAD = 0.0
TONGS_MAX_RAD = 0.50
TONGS_JOINT_MAX_RATE = 3.0  # rad/s
TONGS_CONTROL_OPEN_HAND = 0.40
TONGS_CONTROL_CLOSED_HAND = 1.20
TONGS_GRIP_LOCAL_MIN = np.asarray([-0.150, -0.060, -0.060], dtype=np.float64)
TONGS_GRIP_LOCAL_MAX = np.asarray([0.000, 0.120, 0.070], dtype=np.float64)
TONGS_GRIP_HAND_DISTANCE = 0.075
TONGS_PAYLOAD_LOCAL_MIN = np.asarray([0.085, -0.030, -0.045], dtype=np.float64)
TONGS_PAYLOAD_LOCAL_MAX = np.asarray([0.175, 0.110, 0.045], dtype=np.float64)
TONGS_LATCH_CLOSE_RAD = TONGS_HOLD_RAD + 0.03
TONGS_RELEASE_OPEN_RAD = TONGS_HOLD_RAD + 0.06
TONGS_SPRING_STIFFNESS = 8.0
TONGS_SPRING_DAMPING = 0.40
TONGS_SPRING_MAX_FORCE = 2.0

# Plating / table setting (world XY offsets from the plate centre; the robot
# faces +X, so "left of the plate" is +Y and "upper right" is +X/-Y).
PLATE_CENTER_RADIUS = 0.05
HOT_DOG_PLATE_Z_RANGE = (0.000, 0.090)
FORK_OFFSET = np.asarray([0.0, 0.18], dtype=np.float64)
MUSTARD_OFFSET = np.asarray([-0.08, -0.17], dtype=np.float64)
MUG_OFFSET = np.asarray([0.07, -0.17], dtype=np.float64)
TABLEWARE_TOLERANCE = 0.035
TRAY_HALF_XY = (0.161, 0.260)
TRAY_Z_RANGE = (-0.02, 0.28)
UPRIGHT_THRESHOLD = 0.85
HANDS_CLEAR_DISTANCE = 0.11
STABLE_LINEAR_SPEED = 0.045
STABLE_ANGULAR_SPEED = 0.50
DROP_BELOW_TABLE = 0.12
HAND_IN_CAVITY_DEBOUNCE_STEPS = 3

# Reset layout (world XY) and seeded randomization ranges.
RESET_XY = {
    HOT_DOG: np.asarray([-0.29, -0.29], dtype=np.float64),
    TONGS: np.asarray([-0.37, -0.69], dtype=np.float64),
    PLATE: np.asarray([-0.29, -0.29], dtype=np.float64),
    FORK: np.asarray([-0.29, -0.085], dtype=np.float64),
    MUSTARD: np.asarray([-0.22, -0.48], dtype=np.float64),
    MUG: np.asarray([-0.39, -0.48], dtype=np.float64),
}
RESET_XY_JITTER = {
    HOT_DOG: 0.004,  # relative to the independently randomized plate
    TONGS: 0.008,  # keep the angled tool inside the narrow table-edge strip
    PLATE: 0.015,
    FORK: 0.012,
    MUSTARD: 0.015,
    MUG: 0.010,
}
RESET_YAW_JITTER_DEG = {
    # The plate starts with clearly varied food orientations: longitudinal,
    # diagonal, and transverse.  sample_layout quantizes this body's yaw to
    # 45-degree steps while preserving seeded reproducibility.
    HOT_DOG: 90.0,
    TONGS: 4.0,
    PLATE: 20.0,
    FORK: 7.0,
    MUSTARD: 30.0,
    MUG: 30.0,
}
RESET_YAW_CHOICES_DEG = {
    HOT_DOG: (0.0, 45.0, 90.0, 135.0, 180.0),
}
RESET_BASE_YAW_DEG = {
    HOT_DOG: 90.0,
    TONGS: -22.0,
    PLATE: 0.0,
    FORK: 90.0,
    MUSTARD: 0.0,
    MUG: 0.0,
}
MICROWAVE_XY = np.asarray([0.05, 0.14], dtype=np.float64)
TRAY_XY = np.asarray([-0.29, -0.29], dtype=np.float64)

# Collision names are enumerated from the generated asset manifest and hidden
# with computed USD visibility by d20_selected_assets.configure_visuals.

SUCCESS_CONDITION_LABELS = (
    "1. Load the hot dog, close the door and start heating.",
    "2. Stop heating with cook progress between 90% and 110%.",
    "3. Remove after cooking; place within 5 cm of the plate center.",
    "4. Fork: 18 cm left of the plate; position error at most 3.5 cm.",
    "5. Mustard upright: 17 cm right, 8 cm toward you; error at most 3.5 cm.",
    "6. Mug upright: 17 cm right, 7 cm away from you; error at most 3.5 cm.",
    "7. Lower knob at 0, microwave stopped and door closed.",
    "8. Complete in order: load, heat, stop, remove, then plate.",
    "9. No hand touching hot food or entering a running oven; no drops or burns.",
    "10. Plate and tableware on tray; hands 11 cm clear; objects stable for 0.5 s.",
)


def tableware_position_text(name: str, metrics: Mapping[str, Any], fallback: str) -> str:
    """Live XY correction to the same plate-relative target used by success."""
    delta = metrics.get("d20_tableware_target_delta", {}).get(name)
    if delta is None:
        return fallback
    title = {FORK: "4. Fork", MUSTARD: "5. Mustard upright", MUG: "6. Mug upright"}[name]
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        return f"{title}: target distance unavailable"
    distance = float(np.linalg.norm(delta))
    heading = f"{title}: target error {100 * distance:.1f} cm (allowed {100 * TABLEWARE_TOLERANCE:.1f} cm)"
    if distance <= TABLEWARE_TOLERANCE:
        guide = "Position OK"
        if not metrics.get(f"{name}_ok"):
            guide += "; keep on tray" if name == FORK else "; keep upright and on tray"
    else:
        # The robot faces +X: +Y is left, +X is away from the robot.
        corrections = []
        for amount, positive, negative in ((delta[1], "left", "right"),
                                          (delta[0], "away", "toward")):
            if abs(amount) >= 0.0005:
                corrections.append(f"{positive if amount > 0 else negative} {100 * abs(amount):.1f} cm")
        guide = "Move " + ", ".join(corrections)
    return heading + "\n" + guide


def operator_next_action(metrics: Mapping[str, Any]) -> str:
    """Describe the next existing state-machine requirement; never award progress."""
    m = metrics
    progress = float(m.get("d20_cook_progress", 0.0))
    if m.get("success_now"):
        return "SUCCESS: cooking, plating, table setting and release complete."
    failure = m.get("d20_first_failure_reason")
    if failure:
        reason = {
            "hot_dog_burnt": "The hot dog burnt",
            "hot_dog_or_tableware_dropped": "An object dropped",
            "hand_inside_running_microwave": "A hand entered the running oven",
            "fast_forward_unachievable": "The accelerated heating wait cannot reach the valid cook window",
        }.get(str(failure), "A failure condition occurred")
        return f"{reason}. Reset and retry."
    if m.get("d20_order_violation"):
        return "Plated before cooking was complete. Reset and follow the required order."
    if int(m.get("d20_heated_touch_count", 0)):
        return "A hand touched hot food. Reset and use the tongs to handle the hot dog."
    if not m.get("d20_heated_touch_evidence_available", True):
        return "Hot-food contact detection is unavailable; success cannot be confirmed."
    if progress > MAX_OK_PROGRESS:
        return "Cook progress exceeds 110%; this attempt cannot pass. Turn the lower knob to 0, then reset."
    if m.get("d20_heating_paused"):
        return "Timer paused by hand contact on a knob. Keep the setting steady, then release the knob to resume heating."
    if m.get("d20_running"):
        if not m.get("d20_hot_dog_inside"):
            return "Hot dog not fully inside. Turn the lower knob to 0, then load it fully with the tongs."
        if progress >= 1.0:
            return "Target cook progress reached. Turn the lower knob to 0, then open the door and remove the food."
        return "Heating: wait for the lower knob to reach 0. Opening the door pauses heating. Keep hands outside."
    if not m.get("d20_cook_ok") or not m.get("d20_stop_ok"):
        if not m.get("d20_hot_dog_inside"):
            return "Use the tongs to lift the hot dog from its plate. Open the door and load it fully into the oven."
        if not m.get("d20_door_closed"):
            return "Hot dog loaded: withdraw the tongs and hands, then close the door."
        if int(m.get("d20_power_index", 4)) != 4:
            return "Turn the upper knob to High, at the far right."
        if progress > 0:
            remaining = max(0, math.ceil((1.0 - progress) * COOK_SECONDS))
            return f"More heating needed: set the lower knob to about {remaining} seconds; aim for 100%."
        return "Turn the lower knob clockwise to 1 minute. Heating starts automatically with the door closed."
    if m.get("d20_hot_dog_inside"):
        return "Cooking complete and stopped: open the door, use the tongs to remove the hot dog and return it to the plate."
    if not m.get("d20_removed_after_cook") or not m.get("d20_plated_now"):
        return "Use the tongs to place the hot dog within 5 cm of the plate center, then open the jaws."
    if not m.get("d20_fork_ok"):
        return "Place the fork 18 cm left of the plate, inside the tray."
    if not m.get("d20_mustard_ok"):
        return "Place mustard upright on the tray: 17 cm right of the plate and 8 cm toward you."
    if not m.get("d20_mug_ok"):
        return "Place the mug upright on the tray: 17 cm right of the plate and 7 cm away from you."
    if not m.get("d20_plate_on_tray"):
        return "Place the plate back inside the tray."
    if not m.get("d20_microwave_off_closed"):
        return "Turn the lower knob to 0 and close the door."
    if not m.get("d20_ordered_ok"):
        return "The full load, heat, stop, remove, plate sequence is incomplete. Check the steps above."
    if m.get("d20_contact_release"):
        if not m.get("d20_hand_object_contact_available"):
            return "Hand/object contact detection is unavailable; release cannot be confirmed."
        return "Put down the tongs. Stop touching the food, plate, fork, mustard and mug; keep objects stable for 0.5 s."
    return "Put down the tongs. Keep both hands at least 11 cm from all objects and hold objects stable for 0.5 s."


def operator_status_text(metrics: Mapping[str, Any]) -> str:
    seconds = max(0, math.ceil(float(metrics.get("d20_timer_minutes", 0.0)) * 60 - 1e-6))
    if metrics.get("d20_heating_paused"):
        heating = "Heating paused (hand on knob)"
    else:
        heating = "Heating" if metrics.get("d20_running") else "Stopped"
    power = ("Low", "Defrost", "Medium", "Medium-high", "High")[
        max(0, min(4, int(metrics.get("d20_power_index", 4))))
    ]
    return (
        f"Next: {operator_next_action(metrics)}\n"
        f"{heating} | Power: {power} | Time left: {seconds // 60:02d}:{seconds % 60:02d} (simulation time)\n"
        f"Food detected inside: {'YES' if metrics.get('d20_hot_dog_inside') else 'NO'}"
        f" | Door: {'CLOSED' if metrics.get('d20_door_closed') else 'OPEN'}\n"
        f"Cook progress: {100 * float(metrics.get('d20_cook_progress', 0.0)):.1f}%"
        " (pass: 90%-110%; 100% = 60 s on High)\n"
        f"Reference score: {float(metrics.get('d20_score', 0.0)):.0f}/100. Success requires ALL conditions below."
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in np.asarray(quat, dtype=np.float64))
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_wxyz_to_matrix(pose[3:7])
    matrix[:3, 3] = pose[:3]
    return matrix


def invert_matrix(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ matrix[:3, 3]
    return inverse


def point_in_local_box(
    point_world: np.ndarray,
    frame_pose: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
    *,
    margin: float = 0.0,
) -> bool:
    local = (
        invert_matrix(pose_to_matrix(frame_pose))
        @ np.asarray([*np.asarray(point_world, dtype=np.float64)[:3], 1.0])
    )[:3]
    return bool(
        np.all(local >= np.asarray(box_min) + margin)
        and np.all(local <= np.asarray(box_max) - margin)
    )


def local_point(point_world: np.ndarray, frame_pose: np.ndarray) -> np.ndarray:
    return (
        invert_matrix(pose_to_matrix(frame_pose))
        @ np.asarray([*np.asarray(point_world, dtype=np.float64)[:3], 1.0])
    )[:3]


def upright(pose: np.ndarray, threshold: float = UPRIGHT_THRESHOLD) -> bool:
    rotation = quat_wxyz_to_matrix(np.asarray(pose, dtype=np.float64)[3:7])
    return bool(float(rotation[2, 2]) >= threshold)


def speeds(velocity: np.ndarray | None) -> tuple[float, float]:
    if velocity is None:
        return math.inf, math.inf
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    if velocity.shape != (6,) or not np.all(np.isfinite(velocity)):
        return math.inf, math.inf
    return float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))


def is_stable(velocity: np.ndarray | None) -> bool:
    linear, angular = speeds(velocity)
    return bool(linear <= STABLE_LINEAR_SPEED and angular <= STABLE_ANGULAR_SPEED)


def knob_world_point(microwave_pose: np.ndarray, *, power: bool = False) -> np.ndarray:
    local = POWER_LOCAL if power else TIMER_LOCAL
    return (pose_to_matrix(microwave_pose) @ np.asarray([*local, 1.0]))[:3]


def timer_minutes(angle: float) -> float:
    # Use the same numerical zero as the heating interlock and clockwork.
    # Otherwise an already-off knob can display a spurious second remaining.
    if angle <= TIMER_OFF_ANGLE:
        return 0.0
    return float(np.interp(float(angle), TIMER_ANGLES, TIMER_MINUTES))


def timer_angle(minutes: float) -> float:
    return float(np.interp(float(minutes), TIMER_MINUTES, TIMER_ANGLES))


def timer_return_velocity(angle: float) -> float:
    """Clockwork velocity (rad/s), calibrated to the nonuniform minute scale."""
    if angle <= TIMER_OFF_ANGLE:
        return 0.0
    i = int(np.clip(np.searchsorted(TIMER_ANGLES, angle)-1, 0, len(TIMER_ANGLES)-2))
    return -float((TIMER_ANGLES[i+1]-TIMER_ANGLES[i]) / (60*(TIMER_MINUTES[i+1]-TIMER_MINUTES[i])))


def power_index(angle: float) -> int:
    return int(np.argmin(np.abs(POWER_ANGLES-float(angle))))


def hand_touches_hot_dog(hand_points, hot_dog_pose: np.ndarray) -> bool:
    for point in hand_points:
        if point_in_local_box(
            point,
            hot_dog_pose,
            -HOT_DOG_HALF_EXTENTS,
            HOT_DOG_HALF_EXTENTS,
            margin=-HAND_TOUCH_MARGIN,
        ):
            return True
    return False


def tongs_hand_close_fraction(hand_command: float) -> float:
    denominator = max(TONGS_CONTROL_CLOSED_HAND - TONGS_CONTROL_OPEN_HAND, 1e-6)
    return float(
        np.clip((float(hand_command) - TONGS_CONTROL_OPEN_HAND) / denominator, 0.0, 1.0)
    )


def tongs_hand_target_angle(hand_command: float) -> float:
    fraction = tongs_hand_close_fraction(hand_command)
    return float((1.0 - fraction) * TONGS_OPEN_RAD + fraction * TONGS_CLOSED_RAD)


def hand_grips_tongs(hand_points, tongs_pose: np.ndarray) -> bool:
    for point in hand_points:
        if point_in_local_box(point, tongs_pose, TONGS_GRIP_LOCAL_MIN, TONGS_GRIP_LOCAL_MAX):
            return True
    return False


def hot_dog_between_tongs_pads(hot_dog_pose: np.ndarray, tongs_pose: np.ndarray) -> bool:
    return point_in_local_box(
        np.asarray(hot_dog_pose, dtype=np.float64)[:3],
        tongs_pose,
        TONGS_PAYLOAD_LOCAL_MIN,
        TONGS_PAYLOAD_LOCAL_MAX,
    )


# ---------------------------------------------------------------------------
# Readings and the ordered state machine
# ---------------------------------------------------------------------------

@dataclass
class D20Reading:
    """Everything the state machine needs for one control frame."""

    dt: float
    elapsed_seconds: float
    table_top_z: float
    door_angle: float
    timer_angle: float
    power_angle: float
    tongs_angle: float
    poses: Mapping[str, np.ndarray]
    velocities: Mapping[str, np.ndarray | None]
    hand_points: Mapping[str, tuple[np.ndarray, ...]]
    # side -> (force, available).  ``available`` is False when the PhysX
    # contact channel is missing; that never counts as "no contact".
    hand_hot_dog_contact: Mapping[str, tuple[float, bool]] = field(default_factory=dict)
    # True when a hand is physically contacting a control knob.  Holding the
    # knob pauses the clockwork and heating without registering a stop event.
    timer_two_finger_hold: bool = False
    tongs_latched: bool = False
    tongs_holding_side: str = ""
    # Peak force over all movable task bodies, including both tong links.
    # Both side entries must be present and valid for contact-based release.
    hand_object_contact: Mapping[str, tuple[float, bool]] = field(default_factory=dict)


@dataclass
class D20State:
    running: bool = False
    cook_progress: float = 0.0
    burnt: bool = False
    timer_remaining_seconds: float = 0.0
    drop_frames: int = 0
    removed_early_latched: bool = False
    heating_started_count: int = 0
    loaded_seen: bool = False
    start_seen: bool = False
    light_seen: bool = False
    stop_ok: bool = False
    early_stop_count: int = 0
    removed_after_cook: bool = False
    removed_early_count: int = 0
    plated_seen: bool = False
    heated_touch_count: int = 0
    heated_touch_evidence_available: bool = True
    hand_in_cavity_frames: int = 0
    hand_in_cavity_violation: bool = False
    dropped: bool = False
    release_seconds: float = 0.0
    order_violation: str = ""
    first_failure_reason: str = ""


_STATE_FIELDS = tuple(f.name for f in fields(D20State))
_STATE_STRINGS = {
    "order_violation": ("", "start_before_load", "plated_before_cook_complete"),
    "first_failure_reason": ("", "hot_dog_or_tableware_dropped", "hot_dog_burnt", "hand_inside_running_microwave", "fast_forward_unachievable"),
}
HIDDEN_STATE_SIZE = len(_STATE_FIELDS)


class D20StateMachine:
    """Ordered load -> set timer -> automatic/manual stop -> remove -> plate -> set."""

    def __init__(self, *, contact_release: bool = False) -> None:
        self.contact_release = bool(contact_release)
        self.state = D20State()
        # Runtime-only calibration for the staged medium-power setting.  It is
        # inferred from the saved timer on the first running frame, so the
        # hidden-state schema stays compatible with existing recordings.
        self._medium_target_seconds: float | None = None

    def reset(self) -> None:
        self.state = D20State()
        self._medium_target_seconds = None

    def mark_failure(self, reason: str) -> None:
        """Latch a terminal failure caused outside the per-frame physics update."""
        if reason not in _STATE_STRINGS["first_failure_reason"]:
            raise ValueError(f"Unknown D20 failure reason: {reason}")
        if not self.state.first_failure_reason:
            self.state.first_failure_reason = reason

    def hidden_observation(self) -> np.ndarray:
        return np.asarray([
            _STATE_STRINGS[k].index(getattr(self.state, k)) if k in _STATE_STRINGS else getattr(self.state, k)
            for k in _STATE_FIELDS
        ], dtype=np.float32)

    def restore_hidden_observation(self, values) -> None:
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size != HIDDEN_STATE_SIZE or not np.isfinite(values).all():
            raise ValueError("Invalid D20 hidden-state schema")
        defaults = D20State()
        for k, value in zip(_STATE_FIELDS, values):
            restored = _STATE_STRINGS[k][int(value)] if k in _STATE_STRINGS else type(getattr(defaults, k))(value)
            setattr(self.state, k, restored)
        self._medium_target_seconds = None

    # -- predicates ---------------------------------------------------------

    @staticmethod
    def _hot_dog_inside(reading: D20Reading) -> bool:
        from .d20_selected_assets import hot_dog_collision_vertices

        relative = invert_matrix(pose_to_matrix(reading.poses[MICROWAVE])) @ pose_to_matrix(reading.poses[HOT_DOG])
        local = hot_dog_collision_vertices() @ relative[:3,:3].T + relative[:3,3]
        lower = CAVITY_MIN + HOT_DOG_CAVITY_MARGIN
        # Food rests on the floor/turntable. An oriented bounding box adds
        # nonexistent lower corners; requiring air below them rejects loading.
        lower[2] = CAVITY_MIN[2] - HOT_DOG_FLOOR_CONTACT_TOLERANCE
        return bool(np.all(local >= lower) and np.all(local <= CAVITY_MAX - HOT_DOG_CAVITY_MARGIN))

    @staticmethod
    def _hand_in_cavity(reading: D20Reading) -> bool:
        for points in reading.hand_points.values():
            for point in points:
                if point_in_local_box(
                    point,
                    reading.poses[MICROWAVE],
                    CAVITY_MIN,
                    CAVITY_MAX,
                    margin=-HAND_CAVITY_MARGIN,
                ):
                    return True
        return False

    @staticmethod
    def _heated_hand_touch(reading: D20Reading) -> tuple[bool, bool]:
        """Return (touching, evidence_available) for hand/hot-dog contact."""
        if reading.hand_hot_dog_contact:
            available = all(bool(item[1]) for item in reading.hand_hot_dog_contact.values())
            touching = any(
                bool(item[1]) and float(item[0]) >= HAND_CONTACT_FORCE
                for item in reading.hand_hot_dog_contact.values()
            )
            return touching, available
        # Interface-only path: geometric fallback from the hand points.
        points = [p for pts in reading.hand_points.values() for p in pts]
        return hand_touches_hot_dog(points, reading.poses[HOT_DOG]), bool(points)

    # -- update -------------------------------------------------------------

    def update(self, reading: D20Reading, *, update: bool) -> dict[str, Any]:
        s = self.state
        poses = reading.poses
        door_closed = abs(float(reading.door_angle)) <= DOOR_CLOSED_ABS
        door_open = float(reading.door_angle) <= DOOR_OPEN_FOR_LOADING
        timer_on = float(reading.timer_angle) > TIMER_OFF_ANGLE
        selected_power = power_index(reading.power_angle)
        hot_dog_inside = self._hot_dog_inside(reading)
        hand_in_cavity = self._hand_in_cavity(reading)
        heated = s.cook_progress > 0.0
        touching, touch_evidence = self._heated_hand_touch(reading)

        plate = poses[PLATE]
        hot_dog = poses[HOT_DOG]
        hot_dog_plate_local = local_point(hot_dog[:3], plate)
        plated_now = bool(
            float(np.linalg.norm(hot_dog_plate_local[:2])) <= PLATE_CENTER_RADIUS
            and HOT_DOG_PLATE_Z_RANGE[0] <= float(hot_dog_plate_local[2]) <= HOT_DOG_PLATE_Z_RANGE[1]
        )
        plate_xy = np.asarray(plate[:2], dtype=np.float64)

        target_deltas = {
            name: plate_xy + offset - np.asarray(poses[name][:2], dtype=np.float64)
            for name, offset in ((FORK, FORK_OFFSET), (MUSTARD, MUSTARD_OFFSET), (MUG, MUG_OFFSET))
        }

        def _at(name: str) -> bool:
            return bool(float(np.linalg.norm(target_deltas[name])) <= TABLEWARE_TOLERANCE)

        def _on_tray(name: str) -> bool:
            local = local_point(poses[name][:3], poses[SERVING_TRAY])
            return bool(
                abs(float(local[0])) <= TRAY_HALF_XY[0]
                and abs(float(local[1])) <= TRAY_HALF_XY[1]
                and TRAY_Z_RANGE[0] <= float(local[2]) <= TRAY_Z_RANGE[1]
            )

        fork_ok = _at(FORK) and _on_tray(FORK)
        mustard_ok = _at(MUSTARD) and _on_tray(MUSTARD) and upright(poses[MUSTARD])
        mug_ok = _at(MUG) and _on_tray(MUG) and upright(poses[MUG])
        plate_on_tray = _on_tray(PLATE)
        table_set = bool(fork_ok and mustard_ok and mug_ok and plate_on_tray)

        dropped_now = any(
            float(poses[name][2]) < float(reading.table_top_z) - DROP_BELOW_TABLE
            for name in FREE_BODIES
        )
        stable = all(is_stable(reading.velocities.get(name)) for name in FREE_BODIES)
        points = [p for pts in reading.hand_points.values() for p in pts]
        if points:
            hand_min_distance = min(
                float(np.linalg.norm(np.asarray(point)[:3] - poses[name][:3]))
                for point in points
                for name in FREE_BODIES
            )
            hands_released = hand_min_distance >= HANDS_CLEAR_DISTANCE
            hand_pose_available = True
        else:
            hand_min_distance = 0.0
            hands_released = False
            hand_pose_available = False

        release_contact_available = False
        release_contact_force = 0.0
        if self.contact_release:
            contacts = [reading.hand_object_contact.get(side, (0.0, False))
                        for side in ("left", "right")]
            release_contact_available = all(
                available and math.isfinite(force) and force >= 0.0
                for force, available in contacts)
            if release_contact_available:
                release_contact_force = max(force for force, _ in contacts)
            hands_released = bool(
                release_contact_available
                and release_contact_force <= RELEASE_CONTACT_FORCE_EPSILON
                and not reading.tongs_holding_side and not reading.tongs_latched)

        if update:
            dt = max(float(reading.dt), 0.0)
            s.drop_frames = s.drop_frames + 1 if dropped_now and reading.elapsed_seconds >= 0.5 else 0
            if s.drop_frames >= 6:
                s.dropped = True
                if not s.first_failure_reason:
                    s.first_failure_reason = "hot_dog_or_tableware_dropped"
            if hot_dog_inside:
                s.loaded_seen = True
            was_running = s.running
            s.timer_remaining_seconds = 60.0 * timer_minutes(reading.timer_angle)
            s.running = bool(timer_on and door_closed and not s.burnt)
            if was_running and not s.running:
                self._register_stop()
            if s.running and not was_running:
                s.heating_started_count += 1
                if hot_dog_inside:
                    s.start_seen = True
            if (
                s.running
                and selected_power == 2
                and self._medium_target_seconds is None
                and 150.0 <= s.timer_remaining_seconds <= 210.0
            ):
                self._medium_target_seconds = s.timer_remaining_seconds
            power_rate = float(POWER_RATES[selected_power])
            if self._medium_target_seconds is not None and selected_power == 2:
                # Normalize the accepted 2.5-3.5 minute setting so the full
                # selected timer dose reaches one cook unit at medium power.
                power_rate = COOK_SECONDS / self._medium_target_seconds
            if s.running and not reading.timer_two_finger_hold:
                if hot_dog_inside:
                    s.cook_progress = min(1.5, s.cook_progress + dt * power_rate / COOK_SECONDS)
                    if s.cook_progress >= DONE_PROGRESS:
                        s.light_seen = True
                    if s.cook_progress > BURNT_PROGRESS:
                        s.burnt = True
                        s.running = False
                        if not s.first_failure_reason:
                            s.first_failure_reason = "hot_dog_burnt"
            if s.running and hand_in_cavity:
                s.hand_in_cavity_frames += 1
                if s.hand_in_cavity_frames >= HAND_IN_CAVITY_DEBOUNCE_STEPS:
                    s.hand_in_cavity_violation = True
                    if not s.first_failure_reason:
                        s.first_failure_reason = "hand_inside_running_microwave"
            else:
                s.hand_in_cavity_frames = 0
            if heated:
                if not touch_evidence:
                    s.heated_touch_evidence_available = False
                elif touching:
                    s.heated_touch_count += 1
            cook_ok = (
                (not s.running)
                and DONE_PROGRESS - COOK_PROGRESS_EPSILON <= s.cook_progress
                <= MAX_OK_PROGRESS + COOK_PROGRESS_EPSILON
            )
            if s.loaded_seen and not hot_dog_inside and not s.running:
                if cook_ok and s.stop_ok:
                    s.removed_after_cook = True
                elif s.cook_progress > 0.0 and not cook_ok and not s.removed_after_cook:
                    # Taking the hot dog out early is recoverable: it can be
                    # reloaded and reheated, but the excursion is recorded.
                    if not s.removed_early_latched:
                        s.removed_early_count += 1
                        s.removed_early_latched = True
            if hot_dog_inside:
                s.removed_early_latched = False
            if plated_now and s.removed_after_cook:
                s.plated_seen = True
            elif plated_now and not s.removed_after_cook and s.cook_progress > 0.0:
                s.order_violation = s.order_violation or "plated_before_cook_complete"

        cook_ok = (
            (not s.running)
            and DONE_PROGRESS - COOK_PROGRESS_EPSILON <= s.cook_progress
            <= MAX_OK_PROGRESS + COOK_PROGRESS_EPSILON
        )
        microwave_off_closed = bool((not s.running) and not timer_on and door_closed)
        no_heated_touch = bool(s.heated_touch_count == 0 and s.heated_touch_evidence_available)
        ordered_ok = bool(
            s.loaded_seen
            and s.start_seen
            and s.stop_ok
            and s.removed_after_cook
            and s.plated_seen
            and not s.order_violation
        )
        placement_ready = bool(
            ordered_ok
            and cook_ok
            and plated_now
            and table_set
            and microwave_off_closed
            and no_heated_touch
            and not s.dropped
            and not s.burnt
            and not s.hand_in_cavity_violation
            and stable
        )
        release_evidence_available = release_contact_available if self.contact_release else hand_pose_available
        release_ready = bool(placement_ready and release_evidence_available and hands_released)
        if update:
            s.release_seconds = (
                min(RELEASE_HOLD_SECONDS, s.release_seconds + max(float(reading.dt), 0.0))
                if release_ready
                else 0.0
            )
        release_complete = bool(release_ready and s.release_seconds + 1e-9 >= RELEASE_HOLD_SECONDS)
        success = bool(placement_ready and release_complete)

        light_state = (
            "burnt"
            if s.burnt
            else "done"
            if s.cook_progress >= DONE_PROGRESS
            else "heating"
            if s.running
            else "idle"
        )
        score = self._score(
            ordered_ok=ordered_ok,
            cook_ok=cook_ok,
            plated=plated_now and s.removed_after_cook,
            fork_ok=fork_ok,
            mustard_ok=mustard_ok,
            mug_ok=mug_ok,
            microwave_off_closed=microwave_off_closed,
            no_heated_touch=no_heated_touch,
            dropped=s.dropped,
            elapsed=reading.elapsed_seconds,
        )
        failed = bool(
            s.burnt or s.hand_in_cavity_violation or s.dropped
            or s.first_failure_reason == "fast_forward_unachievable"
        )
        return {
            "d20_door_angle": float(reading.door_angle),
            "d20_door_closed": door_closed,
            "d20_door_open": door_open,
            "d20_timer_angle": float(reading.timer_angle),
            "d20_timer_minutes": timer_minutes(reading.timer_angle),
            "d20_timer_two_finger_hold": bool(reading.timer_two_finger_hold),
            # Kept under the historical observation key for compatibility;
            # semantics are now any hand contact on either control knob.
            "d20_timer_hand_hold": bool(reading.timer_two_finger_hold),
            "d20_heating_paused": bool(s.running and reading.timer_two_finger_hold),
            "d20_power_angle": float(reading.power_angle),
            "d20_power_index": selected_power,
            "d20_power_label": POWER_LABELS[selected_power],
            "d20_power_rate": float(
                COOK_SECONDS / self._medium_target_seconds
                if self._medium_target_seconds is not None and selected_power == 2
                else POWER_RATES[selected_power]
            ),
            "d20_hot_dog_inside": hot_dog_inside,
            "d20_hand_in_cavity": hand_in_cavity,
            "d20_running": s.running,
            "d20_cook_progress": float(s.cook_progress),
            "d20_light_state": light_state,
            "d20_light_on": s.cook_progress >= DONE_PROGRESS,
            "d20_burnt": s.burnt,
            "d20_loaded_seen": s.loaded_seen,
            "d20_start_seen": s.start_seen,
            "d20_heating_started_count": s.heating_started_count,
            "d20_light_seen": s.light_seen,
            "d20_stop_ok": s.stop_ok,
            "d20_early_stop_count": s.early_stop_count,
            "d20_cook_ok": cook_ok,
            "d20_removed_after_cook": s.removed_after_cook,
            "d20_removed_early_count": s.removed_early_count,
            "d20_plated_now": plated_now,
            "d20_plated_seen": s.plated_seen,
            "d20_hot_dog_plate_distance": float(np.linalg.norm(hot_dog_plate_local[:2])),
            "d20_tableware_target_delta": {
                name: tuple(float(value) for value in delta) for name, delta in target_deltas.items()
            },
            "d20_fork_ok": fork_ok,
            "d20_mustard_ok": mustard_ok,
            "d20_mug_ok": mug_ok,
            "d20_plate_on_tray": plate_on_tray,
            "d20_table_set": table_set,
            "d20_microwave_off_closed": microwave_off_closed,
            "d20_heated_touch_count": s.heated_touch_count,
            "d20_heated_touch_evidence_available": s.heated_touch_evidence_available,
            "d20_no_heated_touch": no_heated_touch,
            "d20_hand_in_cavity_violation": s.hand_in_cavity_violation,
            "d20_dropped": s.dropped,
            "d20_order_violation": s.order_violation,
            "d20_ordered_ok": ordered_ok,
            "d20_stable": stable,
            "d20_hand_pose_available": hand_pose_available,
            "d20_hands_released": hands_released,
            "d20_contact_release": self.contact_release,
            "d20_hand_object_contact_available": release_contact_available,
            "d20_hand_object_contact_force": release_contact_force,
            "d20_hand_min_distance": float(hand_min_distance),
            "d20_placement_ready": placement_ready,
            "d20_release_seconds": float(s.release_seconds),
            "d20_release_complete": release_complete,
            "d20_tongs_angle": float(reading.tongs_angle),
            "d20_tongs_latched": bool(reading.tongs_latched),
            "d20_tongs_holding_side": str(reading.tongs_holding_side),
            "d20_score": float(score),
            "d20_failed": failed,
            "d20_first_failure_reason": s.first_failure_reason,
            "success_now": success,
        }

    def _register_stop(self) -> None:
        s = self.state
        if DONE_PROGRESS <= s.cook_progress <= MAX_OK_PROGRESS:
            s.stop_ok = True
        elif s.cook_progress < DONE_PROGRESS:
            s.early_stop_count += 1

    @staticmethod
    def _score(
        *,
        ordered_ok: bool,
        cook_ok: bool,
        plated: bool,
        fork_ok: bool,
        mustard_ok: bool,
        mug_ok: bool,
        microwave_off_closed: bool,
        no_heated_touch: bool,
        dropped: bool,
        elapsed: float,
    ) -> float:
        """Proposal scoring protocol (100 points), reported as diagnostics."""
        score = 0.0
        score += 15.0 if ordered_ok else 0.0
        score += 20.0 if cook_ok else 0.0
        score += 15.0 if plated else 0.0
        score += 20.0 * (int(fork_ok) + int(mustard_ok) + int(mug_ok)) / 3.0
        score += 10.0 if microwave_off_closed else 0.0
        score += 10.0 if no_heated_touch else 0.0
        score += 5.0 if not dropped else 0.0
        score += 5.0 * max(0.0, 1.0 - float(elapsed) / 240.0)
        return float(score)


__all__ = [
    "D20Reading",
    "D20State",
    "D20StateMachine",
    "TASK_NAME",
]
