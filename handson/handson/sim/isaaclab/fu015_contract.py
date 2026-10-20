"""CPU-safe contract and scripted policy for B320-FU015.

The live Isaac 5.1 environment imports this module, but the reset and semantic
tests intentionally require neither Kit nor a GPU.  Success is derived only
from physical joint, rigid-body, velocity, and contact readback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np

from .b320_bimanual_scene import B320_CONTROL_HZ, B320_PHYSICS_HZ


TASK_ID = "B320-FU015"
TASK_NAME = "t16_open_laptop"
TASK_PROMPT = (
    "Use the right hand to lift the initially closed display from its front "
    "edge and open it to the target angle shown in the Success Condition "
    "panel. The laptop base is fixed after each randomized reset."
)

ASSET_RELATIVE_PATH = (
    "sim/envs/usd/t16_open_laptop/"
    "dv_laptop/model_computer_9.usd"
)
ASSET_SHA256 = "31ad6079c08b7f2f80a082ec1d5d67309eb98d4bffb3fc24688c583170847aac"
LAPTOP_ROOT_PATH = "/World/Task/FU015Laptop"
BASE_BODY_PATH = f"{LAPTOP_ROOT_PATH}/E_body_1"
DISPLAY_BODY_PATH = f"{LAPTOP_ROOT_PATH}/E_displayer_5"
TOUCHPAD_COLLIDER_PATH = (
    f"{BASE_BODY_PATH}/E_Group_2/touch_pad/touch_pad"
)
HINGE_PATH = f"{DISPLAY_BODY_PATH}/RevoluteJoint_computer_9_up"

NOMINAL_ROOT_XY_M = np.asarray([-0.08, 0.0], dtype=np.float64)
NOMINAL_ROOT_YAW_DEG = -90.0
ROOT_XY_HALF_RANGE_M = np.asarray([0.08, 0.06], dtype=np.float64)
ROOT_YAW_HALF_RANGE_DEG = 15.0
HINGE_TRAVEL_DEG = 120.0
HINGE_INITIAL_FRACTION_RANGE = (0.0, 0.0)
FRICTION_RANGE = (0.80, 1.10)
MASS_SCALE_RANGE = (1.00, 1.20)

TARGET_OPEN_CHOICES_DEG = (30.0, 60.0, 90.0, 120.0)
# Compatibility default for callers that evaluate an isolated readback without
# a reset sample. The live environment always passes the seeded round target.
TARGET_OPEN_DEG = 90.0
TARGET_TOLERANCE_DEG = 7.0
# Command the hand to the asset's physical hinge limit while the passive drive
# remains centered on the 105-degree task target.  Live randomized validation
# measured about 24.5 mm of compliant/DLS TCP lag near the endpoint; stopping
# the hand command at 105 degrees left a valid seed at only 93.71 degrees.
_RIGHT_COMMAND_OPEN_DEG = HINGE_TRAVEL_DEG
BASE_TRANSLATION_MAX_M = 0.015
BASE_LIFT_MAX_M = 0.010
BASE_FAILURE_TRANSLATION_MAX_M = 0.060
BASE_FAILURE_LIFT_MAX_M = 0.040
REQUIRED_CONTACT_FORCE_N = 0.10
FORBIDDEN_CONTACT_FORCE_N = 0.02
STABLE_LINEAR_SPEED_MPS = 0.05
STABLE_ANGULAR_SPEED_RADPS = 0.50
EXPLOSION_LINEAR_SPEED_MPS = 5.0
EXPLOSION_ANGULAR_SPEED_RADPS = 20.0
SUCCESS_HOLD_STEPS = 15
TIMEOUT_STEPS = 660
CONTROL_HZ = B320_CONTROL_HZ
PHYSICS_HZ = B320_PHYSICS_HZ
# The Isaac collection base interprets render_resolution as (height, width),
# while task.yaml declares the camera resolution in the conventional
# [width, height] order.  Keep this explicit so the 640x480 acceptance gate is
# not silently recorded as a 480x640 portrait stream.
RENDER_RESOLUTION_HW = (480, 640)
BILATERAL_OWNER_HOLD_SUBSTEPS = PHYSICS_HZ // CONTROL_HZ

HAND_ACTION_DIM = 20
ACTION_DIM = 14 + 2 * HAND_ACTION_DIM
_RIGHT_REACHABLE_QUAT_WXYZ = np.asarray(
    [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0],
    dtype=np.float64,
)
RIGHT_HOME = np.asarray(
    # Live PhysX readback after applying the FU015-only +0.15 m right-base
    # offset.  Starting from the old nominal command (-0.10, -0.16, 1.07)
    # made DLS close a roughly 20 cm error in the first few actions.  At the
    # valid 27.18-degree reset this swept the palm/thumb through the display.
    # Begin at the actual reset TCP so the clearance transit is continuous.
    [-0.22150017, -0.23950003, 0.99750000, *_RIGHT_REACHABLE_QUAT_WXYZ],
    dtype=np.float64,
)
LEFT_HOME = np.asarray(
    # Live PhysX readback after applying the FU015-only +0.05 m left-base
    # offset.  The old nominal command was about 25 cm from this reset TCP and
    # used a different wrist orientation, which sent negative-X resets into a
    # display-crossing DLS branch.  Start from the actual fixed-base reset pose.
    [-0.32149994, 0.23950008, 0.99749994, *_RIGHT_REACHABLE_QUAT_WXYZ],
    dtype=np.float64,
)
WUJI_POWER_GRIP = np.asarray(
    [
        0.60, 0.10, 0.84, 0.70, 0.68,
        0.00, 0.90, 0.78, 0.70, 0.00,
        0.92, 0.80, 0.68, 0.00, 0.90,
        0.78, 0.64, 0.00, 0.86, 0.74,
    ],
    dtype=np.float64,
)
# Live PhysX ownership at steps 194--239 showed that the display edge is
# carried by right_finger4_link4, with right_finger5_link4 touching the same
# edge but producing no measurable normal force.  Preserve the verified 25%
# pregrasp during transit/approach, then add only a small distal curl to those
# two fingers after contact.  Moving the proximal chain would also move the
# already calibrated edge-contact point.
_WUJI_RIGHT_EDGE_GRIP_DELTA = np.zeros(HAND_ACTION_DIM, dtype=np.float64)
_WUJI_RIGHT_EDGE_GRIP_DELTA[[14, 15, 18, 19]] = (
    WUJI_POWER_GRIP[[14, 15, 18, 19]] * 0.09
)
_RIGHT_EDGE_GRIP_RAMP_STEPS = 45
# Keep the newly established multi-finger edge contact stationary for one
# second before moving the passive hinge.  Without this dwell, the first arc
# commands arrived while adjacent distal links were still entering/leaving the
# contact manifold.  A rendered recording and physics-only certification could
# then distribute one solver impulse across different (but geometrically
# identical) finger/display pairs even though every rigid state was exact.
_RIGHT_CONTACT_SETTLE_STEPS = CONTROL_HZ
_INITIAL_CLEARANCE_END_STEP = 90
_LEFT_OUTSIDE_END_STEP = 165
_LEFT_ABOVE_CONTACT_END_STEP = 180
_LEFT_PRESS_END_STEP = 195
_LEFT_SETTLE_END_STEP = 225
_RIGHT_CONTACT_STEP = 270
_RIGHT_OPEN_START_STEP = _RIGHT_CONTACT_STEP + _RIGHT_CONTACT_SETTLE_STEPS
_RIGHT_OPEN_STEPS = 180
_RIGHT_OPEN_END_STEP = _RIGHT_OPEN_START_STEP + _RIGHT_OPEN_STEPS
# Exact live-angle tracking keeps the owner contact deterministic at shallow
# angles, but becomes a static geometric brace above about 80 degrees.  Once
# there, let the original time-indexed hinge arc advance the target gradually,
# while bounding its lead so a stalled display cannot lose owner contact.
_RIGHT_LIVE_LEAD_BEGIN_DEG = 78.0
_RIGHT_LIVE_LEAD_MAX_DEG = 12.0
# Keep the tracked wrist close enough to retain the measured display-edge
# manifold.  An 8-degree lead was tested at the shallow 13.53-degree reset and
# pulled every right finger off the display once the scheduled arc got ahead;
# 2 degrees preserves the multi-finger contact while the gated hinge drive
# supplies the torque needed to cross the shallow gravity peak.
_RIGHT_SHALLOW_LIVE_LEAD_MAX_DEG = 2.0
# When the display starts nearly flat, the reachable fixed wrist leaves the
# right attachment only fractions of a millimetre above the table.  Lift the
# task-local TCP smoothly at shallow angles; the offset is gone before the
# ordinary hinge arc and does not alter the audited display-edge geometry.
_RIGHT_SHALLOW_CLEARANCE_END_DEG = 22.0
_RIGHT_SHALLOW_CLEARANCE_MAX_M = 0.008
# At the 12.07-degree seed 15506 contact, raising the TCP target by another
# 2 mm was absorbed by the DLS/contact residual and left attachment_right on
# the table.  Its measured horizontal palm offset is nearly perpendicular to
# the palm-to-finger2 display-contact offset.  Rotate about the latter axis:
# A 3-degree live trial cut the attachment/table force from 15.91 N to 8.28 N
# but did not clear the lower attachment envelope.  Use 6 degrees to double
# that measured angular clearance while remaining below the 0.15-rad audited
# orientation-error limit.  The nominal axis below is the measured seed axis
# transformed back out of its +2.32-degree root-yaw randomization.
_RIGHT_SHALLOW_ATTACHMENT_TILT_DEG = 6.0
_RIGHT_SHALLOW_TILT_AXIS_NOMINAL_XY = np.asarray(
    [-0.9820, -0.1889], dtype=np.float64
)
_RIGHT_SHALLOW_TILT_AXIS_NOMINAL_XY /= np.linalg.norm(
    _RIGHT_SHALLOW_TILT_AXIS_NOMINAL_XY
)
_LEFT_FORCE_RELIEF_BEGIN_N = 2.0
_LEFT_FORCE_RELIEF_GAIN_M_PER_N = 0.0005
_LEFT_FORCE_RELIEF_MAX_M = 0.006
_LEFT_FORCE_RELIEF_STEP_MAX_M = 0.001
_LEFT_FORCE_REENGAGE_STEP_M = 0.0002
# Hold the support fingers inside the base band after first contact.  The old
# 11 mm opening retreat made finger2 flicker, while 6 mm let the live shallow
# reset drift 5.4 mm inward and leave the front-edge proxy.  Use the bounded
# midpoint below for both settling and opening.
_LEFT_CONTACT_RETREAT_M = 0.009
_LEFT_CONTACT_RETREAT_STEP_M = 0.001
# Rotate the contact wrist about the line through the finger2/3 support
# manifold.  This preserves those two base contacts while moving the curled
# finger4/5 links to the far side of the initially shallow display plane.
# The task-frame offset is from the commanded TCP to the midpoint of the live
# seed-15302 finger2/3 base manifold before this rotation.
_LEFT_SUPPORT_TILT_DEG = -12.0
# At the 12.07-degree live reset, the full support tilt put the lower edge of
# left_finger2_link4 on the table and routed the support impulse away from the
# laptop front face.  Relaxing the tilt by 3 degrees raises the palm target by
# about 3.5 mm while the calibrated finger2/3 support anchor remains fixed.
# Fade this correction out by 15 degrees.  Extending it to the right-wrist
# 22-degree envelope was live-tested and made the otherwise passing
# 16.46-degree seed 15502 lose its left owner contact until timeout.
_LEFT_SHALLOW_SUPPORT_UNTILT_DEG = 3.0
_LEFT_SHALLOW_SUPPORT_UNTILT_END_DEG = 15.0
_LEFT_SUPPORT_ANCHOR_OFFSET_M = np.asarray(
    [0.0411, 0.0160, -0.1466], dtype=np.float64
)
_WUJI_LEFT_STABILIZER_GRIP = WUJI_POWER_GRIP.copy()
# Keep the middle three fingers on the base edge while folding finger1 (the
# Wuji thumb) fully toward the outside of the display.  Offline FK aligned to
# the live step-150 palm pose places its whole collision mesh at
# x=-0.186..-0.170 m, at least 20 mm beyond the live display envelope, without
# lowering it toward the table.
_WUJI_LEFT_STABILIZER_GRIP[:4] = np.asarray([1.55, 0.92, 1.50, 1.50])
# Randomized seed 15302 attributed the low lateral approach's contacts to the
# two outer fingers (finger4/5), while live base support remained on finger2/3.
# Keep finger4 at the verified-clear setting and tuck only finger5.  Pushing
# both outer fingers near their limits made live contact drive finger4_joint1
# to a 0.026 rad margin.  This setting instead leaves at least 0.059 rad target
# margin on the changed finger5 joints; the calibrated vertical force-relief
# loop handles the remaining shared contact depth.  Finger2/3 remain unchanged
# for required base contact.
_WUJI_LEFT_STABILIZER_GRIP[12:16] = np.asarray([1.40, 0.00, 1.35, 1.35])
_WUJI_LEFT_STABILIZER_GRIP[16:20] = np.asarray([1.48, 0.00, 1.49, 1.50])
# Seed 15502 showed that relaxing finger4 toward the ordinary power grip
# moved finger4_link4 7.5 mm inward and 4.3 mm down, increasing display load
# from 2.7 N to 48 N.  Apply only a small motion in the measured opposite
# direction during the existing right-contact dwell.  The tightest target is
# finger4_joint1 at 1.46 rad, retaining about 0.099 rad to its upper limit.
_WUJI_LEFT_OPENING_GRIP = _WUJI_LEFT_STABILIZER_GRIP.copy()
_WUJI_LEFT_OPENING_GRIP[12:16] = np.asarray([1.46, 0.00, 1.39, 1.40])
_LEFT_FINGER4_TUCK_STEPS = 20

# The source USD joint anchor is (-0.126304, 0.0848212, 0.0067043) m in the
# asset frame.  After the nominal task-root transform it is the line through
# this XZ point, with its source X axis mapped to world -Y.  Live step-194
# contact at the 18.011-degree reset measured a 195 mm radial point in the
# display's audited 45 mm top-edge band.  With the reachable default wrist,
# the measured palm-to-finger4 offset was
# (+0.159941, -0.019145, -0.004386) m.  Subtracting it from that same physical
# contact and transforming back out of seed 15015 gives the nominal TCP below.
_NOMINAL_HINGE_PIVOT_XZ_M = np.asarray(
    [0.1048212, 0.8977043], dtype=np.float64
)
_RIGHT_DISPLAY_TRACK_RADIUS_M = 0.195
_RIGHT_REFERENCE_HINGE_DEG = 18.011459850775623
_RIGHT_REFERENCE_TCP_XZ_M = np.asarray(
    [-0.24095683571924184, 0.9619179964065552],
    dtype=np.float64,
)
_RIGHT_REFERENCE_TCP_Y_M = -0.14305048965321338


def _display_edge_radial_xz(opening_deg: float) -> np.ndarray:
    """Return the tracked display-edge offset from the hinge in task XZ."""

    angle = math.radians(float(opening_deg))
    return np.asarray(
        [
            -_RIGHT_DISPLAY_TRACK_RADIUS_M * math.cos(angle),
            _RIGHT_DISPLAY_TRACK_RADIUS_M * math.sin(angle),
        ],
        dtype=np.float64,
    )


# Keep the right wrist orientation fixed for the first physical repair.  The
# corresponding hand contact point is therefore a fixed world-space offset
# from the TCP.  Translating the TCP by the tracked material point's circular
# motion preserves that offset without the 35 mm inward cut made by a chord.
_RIGHT_TCP_ARC_CENTER_XZ_M = (
    _RIGHT_REFERENCE_TCP_XZ_M
    - _display_edge_radial_xz(_RIGHT_REFERENCE_HINGE_DEG)
)


@dataclass(frozen=True)
class FU015ResetState:
    """One deterministic reset sample from the authoritative source row."""

    seed: int
    root_x_m: float
    root_y_m: float
    root_yaw_deg: float
    hinge_angle_deg: float
    target_open_deg: float
    friction: float
    mass_scale: float

    @property
    def hinge_fraction(self) -> float:
        return abs(float(self.hinge_angle_deg)) / HINGE_TRAVEL_DEG

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def sample_reset_state(seed: int) -> FU015ResetState:
    """Sample exactly once from a local RNG; global RNG state is untouched."""

    normalized_seed = int(seed) % (2**32 - 1)
    rng = np.random.RandomState(normalized_seed)
    xy = NOMINAL_ROOT_XY_M + rng.uniform(
        -ROOT_XY_HALF_RANGE_M,
        ROOT_XY_HALF_RANGE_M,
        size=2,
    )
    yaw = NOMINAL_ROOT_YAW_DEG + rng.uniform(
        -ROOT_YAW_HALF_RANGE_DEG,
        ROOT_YAW_HALF_RANGE_DEG,
    )
    fraction = rng.uniform(*HINGE_INITIAL_FRACTION_RANGE)
    # The source joint opens in its negative-X direction.
    hinge_angle = -HINGE_TRAVEL_DEG * fraction
    target_open_deg = float(rng.choice(TARGET_OPEN_CHOICES_DEG))
    friction = rng.uniform(*FRICTION_RANGE)
    mass_scale = rng.uniform(*MASS_SCALE_RANGE)
    return FU015ResetState(
        seed=normalized_seed,
        root_x_m=float(xy[0]),
        root_y_m=float(xy[1]),
        root_yaw_deg=float(yaw),
        hinge_angle_deg=float(hinge_angle),
        target_open_deg=target_open_deg,
        friction=float(friction),
        mass_scale=float(mass_scale),
    )


@dataclass(frozen=True)
class FU015PhysicalReadback:
    """Minimal physical truth needed for success and failure evaluation."""

    hinge_angle_rad: float
    base_position_m: tuple[float, float, float]
    reset_base_position_m: tuple[float, float, float]
    base_linear_speed_mps: float
    base_angular_speed_radps: float
    display_linear_speed_mps: float
    display_angular_speed_radps: float
    left_base_contact_force_n: float
    right_display_contact_force_n: float
    left_touchpad_contact_force_n: float
    right_base_contact_force_n: float
    robot_table_contact_force_n: float
    # Control-only total left-hand/base load.  Success still uses the
    # role-filtered ``left_base_contact_force_n`` above, but the scripted hand
    # must stop descending when it physically reaches any part of the base.
    left_any_base_contact_force_n: float = 0.0


def _all_finite(readback: FU015PhysicalReadback) -> bool:
    values: list[float] = []
    for value in asdict(readback).values():
        if isinstance(value, tuple):
            values.extend(float(item) for item in value)
        else:
            values.append(float(value))
    return bool(np.all(np.isfinite(values)))


def evaluate_physical_readback(
    readback: FU015PhysicalReadback,
    target_open_deg: float = TARGET_OPEN_DEG,
) -> dict[str, float | bool]:
    """Evaluate the physical contract without proximity or visual shortcuts."""

    finite = _all_finite(readback)
    base = np.asarray(readback.base_position_m, dtype=np.float64)
    reset_base = np.asarray(readback.reset_base_position_m, dtype=np.float64)
    displacement = float(np.linalg.norm(base - reset_base)) if finite else math.inf
    lift = float(max(0.0, base[2] - reset_base[2])) if finite else math.inf
    hinge_deg = (
        abs(float(np.rad2deg(readback.hinge_angle_rad))) if finite else -math.inf
    )
    target_open_deg = float(target_open_deg)
    hinge_error = abs(hinge_deg - target_open_deg)
    left_base = readback.left_base_contact_force_n >= REQUIRED_CONTACT_FORCE_N
    right_display = (
        readback.right_display_contact_force_n >= REQUIRED_CONTACT_FORCE_N
    )
    left_touchpad_clear = (
        readback.left_touchpad_contact_force_n <= FORBIDDEN_CONTACT_FORCE_N
    )
    right_base_clear = (
        readback.right_base_contact_force_n <= FORBIDDEN_CONTACT_FORCE_N
    )
    robot_table_clear = (
        readback.robot_table_contact_force_n <= FORBIDDEN_CONTACT_FORCE_N
    )
    base_stable = bool(
        displacement <= BASE_TRANSLATION_MAX_M and lift <= BASE_LIFT_MAX_M
    )
    base_failure_limit_exceeded = bool(
        displacement > BASE_FAILURE_TRANSLATION_MAX_M
        or lift > BASE_FAILURE_LIFT_MAX_M
    )
    velocity_stable = bool(
        readback.base_linear_speed_mps <= STABLE_LINEAR_SPEED_MPS
        and readback.base_angular_speed_radps <= STABLE_ANGULAR_SPEED_RADPS
        and readback.display_linear_speed_mps <= STABLE_LINEAR_SPEED_MPS
        and readback.display_angular_speed_radps <= STABLE_ANGULAR_SPEED_RADPS
    )
    # Table contact is expected while an operator braces and opens the laptop.
    # Keep reporting robot_table_clear for diagnostics, but only wrong-hand contact
    # with the laptop itself is task-forbidden and allowed to trigger a reset.
    forbidden_contact = not (left_touchpad_clear and right_base_clear)
    exploded = bool(
        readback.base_linear_speed_mps > EXPLOSION_LINEAR_SPEED_MPS
        or readback.display_linear_speed_mps > EXPLOSION_LINEAR_SPEED_MPS
        or readback.base_angular_speed_radps > EXPLOSION_ANGULAR_SPEED_RADPS
        or readback.display_angular_speed_radps > EXPLOSION_ANGULAR_SPEED_RADPS
    )
    base_fell = bool(finite and base[2] < reset_base[2] - 0.03)
    failure = bool(
        not finite
        or base_failure_limit_exceeded
        or exploded
        or base_fell
    )
    success_now = bool(
        finite
        and not failure
        and hinge_error <= TARGET_TOLERANCE_DEG
        and base_stable
        and right_display
        and velocity_stable
    )
    return {
        "finite": finite,
        "hinge_deg": hinge_deg,
        "target_open_deg": target_open_deg,
        "hinge_error_deg": hinge_error,
        "hinge_target_ok": hinge_error <= TARGET_TOLERANCE_DEG,
        "base_displacement_m": displacement,
        "base_lift_m": lift,
        "base_stable": base_stable,
        "base_failure_limit_exceeded": base_failure_limit_exceeded,
        "left_base_contact": left_base,
        "right_display_contact": right_display,
        "left_touchpad_clear": left_touchpad_clear,
        "right_base_clear": right_base_clear,
        "robot_table_clear": robot_table_clear,
        "velocity_stable": velocity_stable,
        "forbidden_contact": forbidden_contact,
        "exploded": exploded,
        "base_fell": base_fell,
        "failure": failure,
        "success_now": success_now,
    }


class FU015SuccessTracker:
    """Consecutive physical-success gate; all hidden state resets explicitly."""

    def __init__(self, hold_steps: int = SUCCESS_HOLD_STEPS):
        if int(hold_steps) <= 0:
            raise ValueError("hold_steps must be positive")
        self.hold_steps = int(hold_steps)
        self.reset()

    def reset(self) -> None:
        self.consecutive_steps = 0
        self.failure_latched = False
        self.last_metrics: dict[str, float | bool] = {}

    def update(self, readback: FU015PhysicalReadback) -> tuple[bool, bool]:
        metrics = evaluate_physical_readback(readback)
        self.last_metrics = metrics
        self.failure_latched = bool(self.failure_latched or metrics["failure"])
        if metrics["success_now"] and not self.failure_latched:
            self.consecutive_steps += 1
        else:
            self.consecutive_steps = 0
        return self.consecutive_steps >= self.hold_steps, self.failure_latched


class FU015BilateralOwnerGate:
    """Compatibility-named dwell gate for continuous required hand contact."""

    def __init__(
        self,
        hold_substeps: int = BILATERAL_OWNER_HOLD_SUBSTEPS,
    ):
        if int(hold_substeps) <= 0:
            raise ValueError("hold_substeps must be positive")
        self.hold_substeps = int(hold_substeps)
        self.reset()

    def reset(self) -> None:
        self.consecutive_substeps = 0

    @property
    def ready(self) -> bool:
        return self.consecutive_substeps >= self.hold_substeps

    def update(self, required_owner_contact: bool) -> bool:
        if required_owner_contact:
            self.consecutive_substeps += 1
        else:
            self.consecutive_substeps = 0
        return self.ready


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _blend(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return start + _smoothstep(alpha) * (end - start)


def _blend_pose(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """Smoothly blend position and a normalized shortest-path quaternion."""

    start = np.asarray(start, dtype=np.float64).reshape(7)
    end = np.asarray(end, dtype=np.float64).reshape(7).copy()
    if float(np.dot(start[3:7], end[3:7])) < 0.0:
        end[3:7] *= -1.0
    pose = _blend(start, end, alpha)
    norm = float(np.linalg.norm(pose[3:7]))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("FU015 pose blend produced an invalid quaternion")
    pose[3:7] /= norm
    return pose


def _task_pose_for_reset(
    nominal_pose: np.ndarray,
    reset_state: FU015ResetState,
) -> np.ndarray:
    """Move a nominal world pose with the randomized laptop task frame."""

    pose = np.asarray(nominal_pose, dtype=np.float64).copy()
    yaw_delta = math.radians(
        float(reset_state.root_yaw_deg) - NOMINAL_ROOT_YAW_DEG
    )
    cosine = math.cos(yaw_delta)
    sine = math.sin(yaw_delta)
    rotation_xy = np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )
    nominal_offset_xy = pose[:2] - NOMINAL_ROOT_XY_M
    pose[:2] = np.asarray(
        [reset_state.root_x_m, reset_state.root_y_m],
        dtype=np.float64,
    ) + rotation_xy @ nominal_offset_xy

    # Absolute TCP actions use wxyz.  Premultiplying by the task-frame yaw
    # preserves each wrist orientation relative to the randomized laptop.
    half_cosine = math.cos(yaw_delta / 2.0)
    half_sine = math.sin(yaw_delta / 2.0)
    w, x, y, z = pose[3:7]
    pose[3:7] = np.asarray(
        [
            half_cosine * w - half_sine * z,
            half_cosine * x - half_sine * y,
            half_cosine * y + half_sine * x,
            half_cosine * z + half_sine * w,
        ],
        dtype=np.float64,
    )
    return pose


def _tilt_left_pose_about_support(
    pose: np.ndarray,
    reset_state: FU015ResetState,
    alpha: float = 1.0,
) -> np.ndarray:
    """Tilt the left wrist without moving the finger2/3 support line."""

    tilted = np.asarray(pose, dtype=np.float64).reshape(7).copy()
    progress = _smoothstep(float(np.clip(alpha, 0.0, 1.0)))
    shallow_start_deg = HINGE_TRAVEL_DEG * HINGE_INITIAL_FRACTION_RANGE[0]
    shallow_progress = np.clip(
        (
            _LEFT_SHALLOW_SUPPORT_UNTILT_END_DEG
            - abs(float(reset_state.hinge_angle_deg))
        )
        / (_LEFT_SHALLOW_SUPPORT_UNTILT_END_DEG - shallow_start_deg),
        0.0,
        1.0,
    )
    shallow_fraction = _smoothstep(shallow_progress)
    effective_tilt_deg = (
        _LEFT_SUPPORT_TILT_DEG
        + _LEFT_SHALLOW_SUPPORT_UNTILT_DEG * shallow_fraction
    )
    angle = math.radians(effective_tilt_deg) * progress
    if abs(angle) <= 1e-12:
        return tilted

    yaw_delta = math.radians(
        float(reset_state.root_yaw_deg) - NOMINAL_ROOT_YAW_DEG
    )
    cosine = math.cos(yaw_delta)
    sine = math.sin(yaw_delta)
    axis = np.asarray([-sine, cosine, 0.0], dtype=np.float64)
    anchor_offset = np.asarray(
        [
            cosine * _LEFT_SUPPORT_ANCHOR_OFFSET_M[0]
            - sine * _LEFT_SUPPORT_ANCHOR_OFFSET_M[1],
            sine * _LEFT_SUPPORT_ANCHOR_OFFSET_M[0]
            + cosine * _LEFT_SUPPORT_ANCHOR_OFFSET_M[1],
            _LEFT_SUPPORT_ANCHOR_OFFSET_M[2],
        ],
        dtype=np.float64,
    )
    anchor = tilted[:3] + anchor_offset
    relative = tilted[:3] - anchor
    angle_cosine = math.cos(angle)
    angle_sine = math.sin(angle)
    tilted[:3] = anchor + (
        relative * angle_cosine
        + np.cross(axis, relative) * angle_sine
        + axis * float(np.dot(axis, relative)) * (1.0 - angle_cosine)
    )

    half_angle = angle / 2.0
    delta_quat = np.asarray(
        [math.cos(half_angle), *(axis * math.sin(half_angle))],
        dtype=np.float64,
    )
    dw, dx, dy, dz = delta_quat
    w, x, y, z = tilted[3:7]
    tilted[3:7] = np.asarray(
        [
            dw * w - dx * x - dy * y - dz * z,
            dw * x + dx * w + dy * z - dz * y,
            dw * y - dx * z + dy * w + dz * x,
            dw * z + dx * y - dy * x + dz * w,
        ],
        dtype=np.float64,
    )
    tilted[3:7] /= np.linalg.norm(tilted[3:7])
    return tilted


def _right_tcp_pose_for_opening(
    opening_deg: float,
    reset_state: FU015ResetState,
) -> np.ndarray:
    """Track one physical display-edge point around the audited hinge axis."""

    opening = float(
        np.clip(abs(float(opening_deg)), 0.0, _RIGHT_COMMAND_OPEN_DEG)
    )
    xz = _RIGHT_TCP_ARC_CENTER_XZ_M + _display_edge_radial_xz(opening)
    nominal_pose = np.asarray(
        [xz[0], _RIGHT_REFERENCE_TCP_Y_M, xz[1], *_RIGHT_REACHABLE_QUAT_WXYZ],
        dtype=np.float64,
    )
    return _task_pose_for_reset(nominal_pose, reset_state)


def _right_shallow_clearance_lift_m(opening_deg: float) -> float:
    """Return the shallow-angle wrist lift that keeps its attachment off-table."""

    start_deg = HINGE_TRAVEL_DEG * HINGE_INITIAL_FRACTION_RANGE[0]
    progress = np.clip(
        (_RIGHT_SHALLOW_CLEARANCE_END_DEG - abs(float(opening_deg)))
        / (_RIGHT_SHALLOW_CLEARANCE_END_DEG - start_deg),
        0.0,
        1.0,
    )
    return float(_RIGHT_SHALLOW_CLEARANCE_MAX_M * _smoothstep(progress))


def _right_shallow_attachment_tilt_pose(
    pose: np.ndarray,
    opening_deg: float,
    reset_state: FU015ResetState,
) -> np.ndarray:
    """Lift the rear wrist attachment about the live display-contact axis."""

    tilted = np.asarray(pose, dtype=np.float64).reshape(7).copy()
    tilt_fraction = (
        _right_shallow_clearance_lift_m(opening_deg)
        / _RIGHT_SHALLOW_CLEARANCE_MAX_M
    )
    angle = math.radians(_RIGHT_SHALLOW_ATTACHMENT_TILT_DEG) * tilt_fraction
    if abs(angle) <= 1e-12:
        return tilted

    yaw_delta = math.radians(
        float(reset_state.root_yaw_deg) - NOMINAL_ROOT_YAW_DEG
    )
    cosine = math.cos(yaw_delta)
    sine = math.sin(yaw_delta)
    nominal_x, nominal_y = _RIGHT_SHALLOW_TILT_AXIS_NOMINAL_XY
    axis = np.asarray(
        [
            cosine * nominal_x - sine * nominal_y,
            sine * nominal_x + cosine * nominal_y,
            0.0,
        ],
        dtype=np.float64,
    )
    half_angle = angle / 2.0
    dw = math.cos(half_angle)
    dx, dy, dz = axis * math.sin(half_angle)
    w, x, y, z = tilted[3:7]
    tilted[3:7] = np.asarray(
        [
            dw * w - dx * x - dy * y - dz * z,
            dw * x + dx * w + dy * z - dz * y,
            dw * y - dx * z + dy * w + dz * x,
            dw * z + dx * y - dy * x + dz * w,
        ],
        dtype=np.float64,
    )
    tilted[3:7] /= np.linalg.norm(tilted[3:7])
    return tilted


def _right_tcp_hinge_arc_pose(
    reset_state: FU015ResetState,
    alpha: float,
) -> np.ndarray:
    """Follow the randomized hinge angle to this round's target setpoint."""

    start_opening = abs(float(reset_state.hinge_angle_deg))
    opening = start_opening + _smoothstep(alpha) * (
        float(reset_state.target_open_deg) - start_opening
    )
    pose = _right_tcp_pose_for_opening(opening, reset_state)
    pose[2] += _right_shallow_clearance_lift_m(opening)
    return _right_shallow_attachment_tilt_pose(pose, opening, reset_state)


class FU015ScriptedPolicy:
    """Deterministic bimanual absolute-pose policy for live tuning/collection.

    The left hand stages outside the display, presses the base front edge, and
    remains there.  The
    right hand reaches the display top edge and follows the measured hinge arc.
    The environment's contact-gated passive drive can only advance while those
    exact side-specific physical contacts are simultaneously present.
    """

    phase_boundaries = (
        _INITIAL_CLEARANCE_END_STEP,
        _LEFT_OUTSIDE_END_STEP,
        _LEFT_ABOVE_CONTACT_END_STEP,
        _LEFT_PRESS_END_STEP,
        _LEFT_SETTLE_END_STEP,
        _RIGHT_CONTACT_STEP,
        _RIGHT_OPEN_START_STEP,
        _RIGHT_OPEN_END_STEP,
    )

    def __init__(self, reset_state: FU015ResetState):
        self.reset_state = reset_state
        self._last_action: np.ndarray | None = None
        self._left_contact_pose: np.ndarray | None = None
        self._left_force_relief_m = 0.0
        self._left_contact_retreat_m = 0.0

    @staticmethod
    def phase_for_step(step: int) -> int:
        return int(sum(int(step) >= boundary for boundary in FU015ScriptedPolicy.phase_boundaries))

    def action(
        self,
        step: int,
        readback: FU015PhysicalReadback | None = None,
    ) -> np.ndarray:
        step = max(0, int(step))
        # Keep the left wrist at its collision-free home height while moving
        # laterally.  The live 66-filter diagnostic showed that lowering an
        # open hand during this transit swept left_finger3_link4 into the table.
        # Approach the front edge from outside the laptop while the distal
        # links remain clear of both the display and table.  Runtime probes on
        # seed 15301 showed that descending after moving inward loaded the
        # display by 15.1 N before reaching the base.  Descend only at the
        # historical verified-clear outer X, move laterally at z=1.05 m, then
        # make the short historical press onto the supported base top band.
        # The reachable high wrist and side-contact wrist are exactly 180
        # degrees apart, so their quaternion dot product is zero.  Reset-yaw
        # roundoff selected opposite interpolation semicircles for seeds 15301
        # and 15302; the latter drove three arm joints to their limits.  This
        # explicit midpoint fixes the verified semicircle, with a short high
        # dwell after the two 15-step turns and before descent.
        left_side_quat = np.asarray([0.0, 1.0, 0.0, 0.0])
        left_turn_mid_quat = np.asarray(
            [0.5, math.sqrt(0.5), 0.5, 0.0], dtype=np.float64
        )
        left_home_high = LEFT_HOME.copy()
        left_home_high[2] = 1.200
        right_home_high = RIGHT_HOME.copy()
        right_home_high[2] = 1.200
        # The composed laptop pose puts the open display across the old direct
        # home-to-pre segment.  Live exact-contact readback measured a 2.22 N
        # left-hand/display collision at step 29.  First move to the target XY
        # above the display, then descend vertically outside its free edge.
        left_high = np.asarray(
            [-0.245, 0.075, 1.200, *_RIGHT_REACHABLE_QUAT_WXYZ]
        )
        left_turn_mid = np.asarray(
            [-0.245, 0.075, 1.200, *left_turn_mid_quat]
        )
        left_pre = np.asarray([-0.245, 0.075, 1.200, *left_side_quat])
        left_edge_outside = np.asarray(
            [-0.245, 0.075, 1.050, *left_side_quat]
        )
        # Land the distal support fingers a few millimetres inside the base
        # front edge.  At the old x=-0.157 target, seed 15302 missed the base
        # by 1.55 mm and contacted the table; +4 mm preserves the verified
        # touchpad clearance while providing a physical edge-contact margin.
        left_above_contact = np.asarray(
            [-0.153, 0.075, 1.050, *left_side_quat]
        )
        # Begin at the same 6 mm vertical relief that the feedback loop can
        # otherwise reach only after a valid front-edge contact is observed.
        # At shallow randomized hinge angles the lower target overran the
        # front edge before the role-filtered force became visible, loading the
        # base from its interior top surface and tripping the 15 mm displacement
        # limit.  Successful seeds retain several newtons of support after this
        # amount of relief, versus the unchanged 0.1 N role threshold.
        left_press = np.asarray([-0.153, 0.075, 1.034, *left_side_quat])
        # The mirrored distal fingers need a slightly higher transit envelope:
        # live PhysX isolated the old path to right_finger2/3_link4 at step 28.
        # Stay at the verified-clear height until the intentional display
        # approach begins.  Settling at z=1.100 previously produced an early
        # 1.09 N display contact at step 59.
        right_pre = np.asarray(
            [-0.18, -0.105, 1.200, *_RIGHT_REACHABLE_QUAT_WXYZ]
        )
        # With the laptop's composed base pose restored, the direct home-to-pre
        # segment crossed the live display at step 29 (6.24 N).  Move to the
        # pregrasp XY at a high clearance first, then descend vertically while
        # remaining outside the display edge.
        right_high = np.asarray(
            [-0.18, -0.105, 1.200, *_RIGHT_REACHABLE_QUAT_WXYZ]
        )
        # Live PhysX measured the reachable 25%-closed pregrasp's distal-link
        # centers at z=0.965--0.980 m, already level with the sampled
        # 18-degree display edge.  Move that geometry only 20 mm toward the
        # edge and 50 mm toward the lid center; deeper/full closure both misses
        # the edge and pushes the DLS solve into a large lateral residual.
        # Adapt the contact point to each reset's randomized initial hinge
        # angle, then track the same physical point to the hinge limit.  The old
        # implementation blended two plausible endpoints with _blend_pose;
        # its straight chord cut about 35 mm inside the true hinge arc and
        # pulled the base beyond the 15 mm contract limit.
        right_contact = _right_tcp_pose_for_opening(
            self.reset_state.hinge_angle_deg,
            self.reset_state,
        )
        right_contact[2] += _right_shallow_clearance_lift_m(
            self.reset_state.hinge_angle_deg
        )
        right_contact = _right_shallow_attachment_tilt_pose(
            right_contact,
            self.reset_state.hinge_angle_deg,
            self.reset_state,
        )
        right_open = _right_tcp_pose_for_opening(
            self.reset_state.target_open_deg,
            self.reset_state,
        )
        (
            left_high,
            left_turn_mid,
            left_pre,
            left_edge_outside,
            left_above_contact,
            left_press,
            right_high,
            right_pre,
        ) = (
            _task_pose_for_reset(pose, self.reset_state)
            for pose in (
                left_high,
                left_turn_mid,
                left_pre,
                left_edge_outside,
                left_above_contact,
                left_press,
                right_high,
                right_pre,
            )
        )
        left_pre_tilted = _tilt_left_pose_about_support(
            left_pre, self.reset_state
        )
        left_edge_outside_tilted = _tilt_left_pose_about_support(
            left_edge_outside, self.reset_state
        )
        left_above_contact_tilted = _tilt_left_pose_about_support(
            left_above_contact, self.reset_state
        )
        left_press_tilted = _tilt_left_pose_about_support(
            left_press, self.reset_state
        )

        if step < _INITIAL_CLEARANCE_END_STEP:
            # Changing wrist orientation during the long home-to-task
            # translation sent the left DLS solve down a reset-dependent branch
            # (seed 15017 retained an 83 mm residual and hit the display).
            # Raise in place, translate with the reachable source wrist, then
            # reorient in place while the entire hand is above the display.
            if step < 15:
                left_pose = _blend_pose(LEFT_HOME, left_home_high, step / 14.0)
            elif step < 45:
                left_pose = _blend_pose(
                    left_home_high, left_high, (step - 15) / 29.0
                )
            elif step < 60:
                left_pose = _blend_pose(
                    left_high, left_turn_mid, (step - 45) / 14.0
                )
            elif step < 75:
                left_pose = _blend_pose(
                    left_turn_mid, left_pre, (step - 60) / 14.0
                )
            else:
                left_pose = _tilt_left_pose_about_support(
                    left_pre, self.reset_state, (step - 75) / 14.0
                )

            # A direct diagonal home-to-high transit intersects the display at
            # the negative-Y edge of the allowed +/-3 cm task randomization
            # (seed 15304).  Raise at the fixed robot home XY first, then move
            # laterally only above the complete randomized display envelope.
            if step < 15:
                right_pose = _blend_pose(
                    RIGHT_HOME, right_home_high, step / 14.0
                )
            elif step < 45:
                right_pose = _blend_pose(
                    right_home_high, right_high, (step - 15) / 29.0
                )
            else:
                right_pose = right_pre.copy()
            # Close during the high transit.  With the fixed source wrist this
            # retracts the distal chain; a light pre-grasp was measured touching
            # the table even though the palm remained at z=1.079 m.
            right_hand_alpha = 0.25
            left_hand_alpha = 0.25 + 0.75 * _smoothstep(step / 20.0)
        elif step < _LEFT_OUTSIDE_END_STEP:
            alpha = (step - _INITIAL_CLEARANCE_END_STEP) / 74.0
            left_pose = _blend_pose(
                left_pre_tilted, left_edge_outside_tilted, alpha
            )
            right_pose = right_pre.copy()
            right_hand_alpha = 0.25
            left_hand_alpha = 1.0
        elif step < _LEFT_ABOVE_CONTACT_END_STEP:
            alpha = (step - _LEFT_OUTSIDE_END_STEP) / 14.0
            left_pose = _blend_pose(
                left_edge_outside_tilted,
                left_above_contact_tilted,
                alpha,
            )
            right_pose = right_pre.copy()
            right_hand_alpha = 0.25
            left_hand_alpha = 1.0
        elif step < _LEFT_PRESS_END_STEP:
            alpha = (step - _LEFT_ABOVE_CONTACT_END_STEP) / 14.0
            left_pose = _blend_pose(
                left_above_contact_tilted, left_press_tilted, alpha
            )
            right_pose = right_pre.copy()
            right_hand_alpha = 0.25
            left_hand_alpha = 1.0
        elif step < _LEFT_SETTLE_END_STEP:
            left_pose = left_press_tilted.copy()
            right_pose = right_pre.copy()
            right_hand_alpha = 0.25
            left_hand_alpha = 1.0
        elif step < _RIGHT_CONTACT_STEP:
            alpha = (step - _LEFT_SETTLE_END_STEP) / 44.0
            left_pose = left_press_tilted.copy()
            right_pose = _blend_pose(right_pre, right_contact, alpha)
            right_hand_alpha = 0.25
            left_hand_alpha = 1.0
        elif step < _RIGHT_OPEN_START_STEP:
            left_pose = left_press_tilted.copy()
            right_pose = right_contact.copy()
            right_hand_alpha = 0.25
            left_hand_alpha = 1.0
        elif step < _RIGHT_OPEN_END_STEP:
            alpha = (step - _RIGHT_OPEN_START_STEP) / (_RIGHT_OPEN_STEPS - 1)
            left_pose = left_press_tilted.copy()
            right_pose = _right_tcp_hinge_arc_pose(self.reset_state, alpha)
            right_hand_alpha = 0.25
            left_hand_alpha = 1.0
        else:
            left_pose = left_press_tilted.copy()
            right_pose = right_open.copy()
            right_hand_alpha = 0.25
            left_hand_alpha = 1.0

        # The passive hinge drive is deliberately gated by exact bilateral
        # owner contact, so the physical display can lag the time-indexed arc
        # whenever the contact manifold flickers.  Track the measured hinge
        # after opening starts; otherwise the wrist runs ahead of the display,
        # loses contact, and the gated drive freezes permanently.  Preserve
        # the deterministic time-indexed pose when no physical readback is
        # available (for offline policy inspection and tests).
        if (
            step >= _RIGHT_OPEN_START_STEP
            and readback is not None
            and np.isfinite(float(readback.hinge_angle_rad))
        ):
            measured_opening_deg = abs(
                math.degrees(float(readback.hinge_angle_rad))
            )
            opening_progress = np.clip(
                (step - _RIGHT_OPEN_START_STEP) / (_RIGHT_OPEN_STEPS - 1),
                0.0,
                1.0,
            )
            initial_opening_deg = abs(
                float(self.reset_state.hinge_angle_deg)
            )
            scheduled_opening_deg = initial_opening_deg + (
                _smoothstep(opening_progress)
                * (_RIGHT_COMMAND_OPEN_DEG - initial_opening_deg)
            )
            commanded_opening_deg = measured_opening_deg
            if measured_opening_deg < _RIGHT_SHALLOW_CLEARANCE_END_DEG:
                commanded_opening_deg = max(
                    measured_opening_deg,
                    min(
                        scheduled_opening_deg,
                        measured_opening_deg
                        + _RIGHT_SHALLOW_LIVE_LEAD_MAX_DEG,
                    ),
                )
            elif measured_opening_deg >= _RIGHT_LIVE_LEAD_BEGIN_DEG:
                commanded_opening_deg = max(
                    measured_opening_deg,
                    min(
                        scheduled_opening_deg,
                        measured_opening_deg + _RIGHT_LIVE_LEAD_MAX_DEG,
                    ),
                )
            right_pose = _right_tcp_pose_for_opening(
                commanded_opening_deg,
                self.reset_state,
            )
            right_pose[2] += _right_shallow_clearance_lift_m(
                commanded_opening_deg
            )
            right_pose = _right_shallow_attachment_tilt_pose(
                right_pose,
                commanded_opening_deg,
                self.reset_state,
            )

        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[:7] = right_pose
        action[7:14] = left_pose
        right_grip = WUJI_POWER_GRIP * right_hand_alpha
        if step >= _RIGHT_OPEN_START_STEP:
            grip_progress = _smoothstep(
                min(
                    (step - _RIGHT_OPEN_START_STEP)
                    / _RIGHT_EDGE_GRIP_RAMP_STEPS,
                    1.0,
                )
            )
            right_grip = (
                right_grip + _WUJI_RIGHT_EDGE_GRIP_DELTA * grip_progress
            )
        action[14 : 14 + HAND_ACTION_DIM] = right_grip
        left_grip = _WUJI_LEFT_STABILIZER_GRIP * left_hand_alpha
        if step >= _RIGHT_CONTACT_STEP:
            tuck_progress = _smoothstep(
                min(
                    (step - _RIGHT_CONTACT_STEP)
                    / (_LEFT_FINGER4_TUCK_STEPS - 1),
                    1.0,
                )
            )
            left_grip[12:16] = (
                _WUJI_LEFT_STABILIZER_GRIP[12:16]
                + tuck_progress
                * (
                    _WUJI_LEFT_OPENING_GRIP[12:16]
                    - _WUJI_LEFT_STABILIZER_GRIP[12:16]
                )
            )
        action[14 + HAND_ACTION_DIM :] = left_grip
        left_control_contact_force_n = (
            0.0
            if readback is None
            else max(
                0.0,
                float(readback.left_base_contact_force_n),
                float(readback.left_any_base_contact_force_n),
            )
        )
        if self._left_contact_pose is None and readback is not None:
            if (
                left_control_contact_force_n >= REQUIRED_CONTACT_FORCE_N
                and self._last_action is not None
            ):
                self._left_contact_pose = self._last_action[7:14].copy()
        if self._left_contact_pose is not None:
            self._left_contact_retreat_m = min(
                _LEFT_CONTACT_RETREAT_M,
                self._left_contact_retreat_m + _LEFT_CONTACT_RETREAT_STEP_M,
            )
            contact_force_n = left_control_contact_force_n
            if contact_force_n > _LEFT_FORCE_RELIEF_BEGIN_N:
                requested_relief_m = min(
                    _LEFT_FORCE_RELIEF_MAX_M,
                    (contact_force_n - _LEFT_FORCE_RELIEF_BEGIN_N)
                    * _LEFT_FORCE_RELIEF_GAIN_M_PER_N,
                )
                if requested_relief_m >= self._left_force_relief_m:
                    self._left_force_relief_m = min(
                        requested_relief_m,
                        self._left_force_relief_m
                        + _LEFT_FORCE_RELIEF_STEP_MAX_M,
                    )
                else:
                    self._left_force_relief_m = max(
                        requested_relief_m,
                        self._left_force_relief_m
                        - _LEFT_FORCE_REENGAGE_STEP_M,
                    )
            else:
                # Re-engage throughout the entire sub-2 N band, not only
                # after contact has already fallen below the 0.1 N owner
                # threshold.  Holding stale relief in that gap left only
                # 0.26--0.69 N of support on the shallow live reset and let
                # individual physics substeps lose the owner manifold.
                self._left_force_relief_m = max(
                    0.0,
                    self._left_force_relief_m - _LEFT_FORCE_REENGAGE_STEP_M,
                )
            left_hold = self._left_contact_pose.copy()
            yaw_delta = math.radians(
                float(self.reset_state.root_yaw_deg) - NOMINAL_ROOT_YAW_DEG
            )
            left_hold[:2] += np.asarray(
                [
                    -math.cos(yaw_delta) * self._left_contact_retreat_m,
                    -math.sin(yaw_delta) * self._left_contact_retreat_m,
                ],
                dtype=np.float64,
            )
            left_hold[2] += self._left_force_relief_m
            action[7:14] = left_hold
        self._last_action = action.copy()
        return action

    def metadata(self) -> Mapping[str, Any]:
        return {
            "policy": "fu015_scripted_bimanual_hinge_arc_v1",
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "reset": self.reset_state.to_dict(),
            "phase_boundaries": list(self.phase_boundaries),
            "action_dim": ACTION_DIM,
            "left_role": "base_front_edge_avoiding_touchpad",
            "right_role": "display_top_edge_open_hinge",
            "left_contact_control": "outer_descent_low_lateral_press_front_retreat_force_relief",
        }


__all__ = [
    "ACTION_DIM",
    "ASSET_RELATIVE_PATH",
    "ASSET_SHA256",
    "BASE_BODY_PATH",
    "BILATERAL_OWNER_HOLD_SUBSTEPS",
    "DISPLAY_BODY_PATH",
    "FU015BilateralOwnerGate",
    "FU015PhysicalReadback",
    "FU015ResetState",
    "FU015ScriptedPolicy",
    "FU015SuccessTracker",
    "HINGE_PATH",
    "LAPTOP_ROOT_PATH",
    "SUCCESS_HOLD_STEPS",
    "TASK_ID",
    "TASK_NAME",
    "TASK_PROMPT",
    "TIMEOUT_STEPS",
    "TOUCHPAD_COLLIDER_PATH",
    "evaluate_physical_readback",
    "sample_reset_state",
]
