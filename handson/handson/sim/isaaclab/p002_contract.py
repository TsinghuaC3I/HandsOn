"""CPU-safe contract and scripted policy for P0-02 stage one."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import math
from typing import Any, Mapping

import numpy as np


TASK_ID = "P0-02"
TASK_NAME = "t17_open_latched_door"
# Keep this legacy-named scalar because it is already part of recorded state and
# task-info schemas.  Version 4 keeps the v3 role-slot mapping and one-sided
# door-success semantics, but calibrates the Helindu Handle002 latch threshold
# to the physical travel measured during operator collection.  The observation
# key and array shapes remain unchanged.
P002_ROLE_CONTRACT_VERSION = 4
TASK_PROMPT = (
    "Use the right hand to depress the near-side lever handle until the latch "
    "physically unlocks, then keep the handle controlled while the left hand "
    "pushes the door panel open to at least 45 degrees."
)

ASSET_RELATIVE_PATH = (
    "sim/envs/usd/t17_open_latched_door/"
    "zryz_door042_source/Door042_stage1.usd"
)
ASSET_SHA256 = "4367b6b8bcf60b934dfa9763502676f2d1d765688e81e261cfdc0d81745975ff"
SOURCE_ASSET_SHA256 = "f27f7148ec917642d9733f3a54161634a005e03fe1168069eb4f4b0ffc4ff651"
DOOR_ROOT_PATH = "/World/Task/P002Door"
FRAME_BODY_PATH = f"{DOOR_ROOT_PATH}/Door042"
DOOR_BODY_PATH = f"{DOOR_ROOT_PATH}/Door042_Door"
REAR_HANDLE_BODY_PATH = f"{DOOR_ROOT_PATH}/Door042_Handle001"
FRONT_HANDLE_BODY_PATH = f"{DOOR_ROOT_PATH}/Door042_Handle002"
REAR_HANDLE_JOINT_PATH = f"{REAR_HANDLE_BODY_PATH}/Handle001_joint"
FRONT_HANDLE_JOINT_PATH = f"{FRONT_HANDLE_BODY_PATH}/Handle002_joint"
DOOR_JOINT_PATH = f"{DOOR_BODY_PATH}/Door_joint"

# Door042 models both lever return springs explicitly with angular drives.
# Let those springs define the two lever rest poses instead of adding a
# second, gravity-driven offset that moves the authored 0-degree recording
# origin.  The frame and door panel keep normal gravity.
GRAVITY_DISABLED_BODY_PATHS = frozenset(
    {FRONT_HANDLE_BODY_PATH, REAR_HANDLE_BODY_PATH}
)

ASSET_SCALE = 0.60
# Keep the complete door fixture near the platform-standard mirrored arm home.
# The old task-specific arm placement put the palms at x=-0.10 m; the shared
# home reads back at x=-0.3715 m.  Moving the fixture by the same delta keeps
# the designed approach distance while every task target remains in one frame.
NOMINAL_ROOT_XYZ_M = np.asarray([0.01, 0.15, 1.05], dtype=np.float64)
NOMINAL_ROOT_YAW_DEG = 90.0
ROOT_XY_HALF_RANGE_M = 0.02
ROOT_YAW_HALF_RANGE_DEG = 3.0

# Operator logs captured 61.19 N of intended Handle002 contact at 28.17 deg,
# while the former 45 deg threshold left ``press_steps`` at zero and the door
# drive locked.  Use a reachable 25 deg actuation point with a separate 15 deg
# release point so the latch retains a real 10 deg hysteresis band.
HANDLE_PRESS_DEG = 25.0
HANDLE_RELEASE_DEG = 15.0
HANDLE_PRESS_DWELL_STEPS = 5
DOOR_ESCAPE_DEG = 12.0
LOCKED_DOOR_MAX_DEG = 2.5
# Keep the historical ``TARGET_DOOR_DEG`` name because the scripted policy and
# recorded consumers already use it.  For success it is a one-sided minimum,
# not the centre of a tolerance band.
TARGET_DOOR_DEG = 45.0
# Automated acceptance intentionally commands beyond the human success
# minimum so small tracking error does not strand the policy just below 45 deg.
SCRIPTED_DOOR_GOAL_DEG = 50.0
SUCCESS_HOLD_STEPS = 15
REQUIRED_CONTACT_FORCE_N = 0.10
FORBIDDEN_CONTACT_FORCE_N = 0.02
STABLE_DOOR_SPEED_RADPS = 0.50
EXPLOSION_ANGULAR_SPEED_RADPS = 20.0
MAX_PENETRATION_M = 0.005
CONTROL_HZ = 30
PHYSICS_HZ = 120
TIMEOUT_STEPS = 720
RENDER_RESOLUTION_HW = (480, 640)

LOCKED_DOOR_STIFFNESS = 1000.0
LOCKED_DOOR_DAMPING = 30.0
LOCKED_DOOR_MAX_FORCE = 100.0
UNLOCKED_DOOR_STIFFNESS = 0.0
UNLOCKED_DOOR_DAMPING = 0.35
UNLOCKED_DOOR_MAX_FORCE = 0.0
HANDLE_STIFFNESS = 2.5
HANDLE_DAMPING = 0.12
HANDLE_MAX_FORCE = 4.0

# The live Isaac 5.1 probe reports this tensor order.  Runtime still resolves
# names dynamically and fails closed if the three-name set changes.
DOOR_DOF = 0
REAR_HANDLE_DOF = 1
FRONT_HANDLE_DOF = 2
EXPECTED_DOF_NAMES = ("Door_joint", "Handle001_joint", "Handle002_joint")

HAND_ACTION_DIM = 20
ACTION_DIM = 14 + 2 * HAND_ACTION_DIM
_REACHABLE_QUAT_WXYZ = np.asarray(
    [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], dtype=np.float64
)
RIGHT_HANDLE_PALM_NOMINAL_Y_M = -0.04
RIGHT_HOME = np.asarray(
    [-0.3715, -0.2395, 0.9975, *_REACHABLE_QUAT_WXYZ], dtype=np.float64
)
LEFT_HOME = np.asarray(
    [-0.3715, 0.2395, 0.9975, *_REACHABLE_QUAT_WXYZ], dtype=np.float64
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


class LatchPhase(IntEnum):
    LOCKED = 0
    HANDLE_PRESSED = 1
    UNLATCHED = 2
    DOOR_OPEN = 3


@dataclass(frozen=True)
class P002ResetState:
    seed: int
    root_x_m: float
    root_y_m: float
    root_z_m: float
    root_yaw_deg: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def sample_reset_state(seed: int) -> P002ResetState:
    rng = np.random.default_rng(int(seed))
    xy = NOMINAL_ROOT_XYZ_M[:2] + rng.uniform(
        -ROOT_XY_HALF_RANGE_M, ROOT_XY_HALF_RANGE_M, size=2
    )
    yaw = NOMINAL_ROOT_YAW_DEG + float(
        rng.uniform(-ROOT_YAW_HALF_RANGE_DEG, ROOT_YAW_HALF_RANGE_DEG)
    )
    return P002ResetState(
        seed=int(seed),
        root_x_m=float(xy[0]),
        root_y_m=float(xy[1]),
        root_z_m=float(NOMINAL_ROOT_XYZ_M[2]),
        root_yaw_deg=yaw,
    )


@dataclass(frozen=True)
class P002PhysicalReadback:
    front_handle_angle_rad: float
    rear_handle_angle_rad: float
    door_angle_rad: float
    door_angular_speed_radps: float
    left_front_handle_contact_force_n: float
    right_door_contact_force_n: float
    robot_table_contact_force_n: float
    robot_frame_contact_force_n: float
    robot_rear_handle_contact_force_n: float
    max_penetration_m: float = 0.0
    hard_failure_max_penetration_m: float = 0.0

    # Preserve the original dataclass field order and keyword constructor for
    # recorded-data and downstream compatibility.  The first two contact slots
    # now carry the corrected operator roles; canonical accessors prevent new
    # runtime/telemetry code from perpetuating the old, reversed names.
    @property
    def right_front_handle_contact_force_n(self) -> float:
        return float(self.left_front_handle_contact_force_n)

    @property
    def left_door_contact_force_n(self) -> float:
        return float(self.right_door_contact_force_n)


def _all_finite(readback: P002PhysicalReadback) -> bool:
    return bool(
        np.isfinite(
            [
                readback.front_handle_angle_rad,
                readback.rear_handle_angle_rad,
                readback.door_angle_rad,
                readback.door_angular_speed_radps,
                readback.right_front_handle_contact_force_n,
                readback.left_door_contact_force_n,
                readback.robot_rear_handle_contact_force_n,
                readback.hard_failure_max_penetration_m,
            ]
        ).all()
    )


def physical_failure_reasons(
    readback: P002PhysicalReadback,
) -> tuple[str, ...]:
    """Return every hard-failure predicate in deterministic priority order."""

    reasons: list[str] = []
    finite = _all_finite(readback)
    if not finite:
        reasons.append("nonfinite_readback")
    door_speed = float(readback.door_angular_speed_radps)
    if math.isfinite(door_speed) and abs(door_speed) > EXPLOSION_ANGULAR_SPEED_RADPS:
        reasons.append("door_speed_explosion")
    # ``max_penetration_m`` remains the full diagnostic reading.
    # Runtime supplies this separate aggregate from only the contact pairs that
    # are still terminal: the intended right-hand/front-handle pair and the
    # rear handle.  Table, door-panel, and door-frame penetration remain
    # observable without ending an otherwise recoverable operator episode.
    penetration = float(readback.hard_failure_max_penetration_m)
    if math.isfinite(penetration) and penetration > MAX_PENETRATION_M:
        reasons.append("penetration_limit")
    rear_handle_force = float(readback.robot_rear_handle_contact_force_n)
    if (
        math.isfinite(rear_handle_force)
        and rear_handle_force >= FORBIDDEN_CONTACT_FORCE_N
    ):
        reasons.append("forbidden_rear_handle_contact")
    return tuple(reasons)


class P002LatchStateMachine:
    """Causal, hysteretic stage-one latch driven only by physical readback."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.phase = LatchPhase.LOCKED
        self.press_steps = 0
        self.right_handle_contact_seen = False
        self.left_door_contact_after_unlock_seen = False
        self.unlocked_seen = False
        self.door_escape_seen = False

    # Compatibility aliases for existing replay/diagnostic consumers.  New
    # code must use the corrected role names above.
    @property
    def left_handle_contact_seen(self) -> bool:
        return self.right_handle_contact_seen

    @left_handle_contact_seen.setter
    def left_handle_contact_seen(self, value: bool) -> None:
        self.right_handle_contact_seen = bool(value)

    @property
    def right_door_contact_after_unlock_seen(self) -> bool:
        return self.left_door_contact_after_unlock_seen

    @right_door_contact_after_unlock_seen.setter
    def right_door_contact_after_unlock_seen(self, value: bool) -> None:
        self.left_door_contact_after_unlock_seen = bool(value)

    @property
    def locked(self) -> bool:
        return self.phase in {LatchPhase.LOCKED, LatchPhase.HANDLE_PRESSED}

    def update(self, readback: P002PhysicalReadback) -> LatchPhase:
        if not _all_finite(readback):
            return self.phase
        handle_deg = math.degrees(float(readback.front_handle_angle_rad))
        door_deg = math.degrees(float(readback.door_angle_rad))
        right_handle_contact = (
            float(readback.right_front_handle_contact_force_n)
            >= REQUIRED_CONTACT_FORCE_N
        )
        left_door_contact = (
            float(readback.left_door_contact_force_n) >= REQUIRED_CONTACT_FORCE_N
        )
        self.right_handle_contact_seen |= right_handle_contact

        if self.locked:
            if right_handle_contact and handle_deg >= HANDLE_PRESS_DEG:
                self.press_steps += 1
                self.phase = LatchPhase.HANDLE_PRESSED
            else:
                self.press_steps = 0
                self.phase = LatchPhase.LOCKED
            if self.press_steps >= HANDLE_PRESS_DWELL_STEPS:
                self.phase = LatchPhase.UNLATCHED
                self.unlocked_seen = True
        else:
            if left_door_contact:
                self.left_door_contact_after_unlock_seen = True
            if door_deg >= DOOR_ESCAPE_DEG:
                self.door_escape_seen = True
                self.phase = LatchPhase.DOOR_OPEN
            elif handle_deg <= HANDLE_RELEASE_DEG:
                self.phase = LatchPhase.LOCKED
                self.press_steps = 0
        return self.phase

    def to_dict(self) -> dict[str, bool | int | str]:
        return {
            "phase": self.phase.name.lower(),
            "phase_code": int(self.phase),
            "locked": self.locked,
            "press_steps": int(self.press_steps),
            "right_handle_contact_seen": self.right_handle_contact_seen,
            "left_door_contact_after_unlock_seen": (
                self.left_door_contact_after_unlock_seen
            ),
            # Legacy keys retain the same values and payload shape expectations
            # for log parsers written before the role correction.
            "left_handle_contact_seen": self.right_handle_contact_seen,
            "right_door_contact_after_unlock_seen": (
                self.left_door_contact_after_unlock_seen
            ),
            "unlocked_seen": self.unlocked_seen,
            "door_escape_seen": self.door_escape_seen,
        }


def evaluate_physical_readback(
    readback: P002PhysicalReadback,
    latch: P002LatchStateMachine,
) -> dict[str, Any]:
    finite = _all_finite(readback)
    handle_deg = math.degrees(float(readback.front_handle_angle_rad))
    door_deg = math.degrees(float(readback.door_angle_rad))
    forbidden_force = float(readback.robot_rear_handle_contact_force_n)
    minimum_angle_ok = door_deg >= TARGET_DOOR_DEG
    stable = abs(float(readback.door_angular_speed_radps)) <= STABLE_DOOR_SPEED_RADPS
    penetration_ok = (
        float(readback.hard_failure_max_penetration_m) <= MAX_PENETRATION_M
    )
    forbidden_clear = forbidden_force < FORBIDDEN_CONTACT_FORCE_N
    success_now = bool(
        finite
        and latch.unlocked_seen
        and latch.right_handle_contact_seen
        and latch.left_door_contact_after_unlock_seen
        and latch.door_escape_seen
        and minimum_angle_ok
        and stable
        and penetration_ok
        and forbidden_clear
    )
    failure_reasons = physical_failure_reasons(readback)
    failed = bool(failure_reasons)
    return {
        "finite": finite,
        "front_handle_angle_deg": handle_deg,
        "door_angle_deg": door_deg,
        # Compatibility field/observation slot: keep the existing name and
        # shape, but its success meaning is now "door angle is at least 45 deg".
        "target_ok": minimum_angle_ok,
        "door_stable": stable,
        "penetration_ok": penetration_ok,
        "forbidden_clear": forbidden_clear,
        "max_forbidden_force_n": forbidden_force,
        "success_now": success_now,
        "failed": failed,
        "failure_reason": failure_reasons[0] if failure_reasons else "",
        "failure_reasons": failure_reasons,
        **latch.to_dict(),
    }


class P002SuccessTracker:
    def __init__(self, hold_steps: int = SUCCESS_HOLD_STEPS) -> None:
        self.hold_steps = int(hold_steps)
        self.reset()

    def reset(self) -> None:
        self.counter = 0

    def update(self, success_now: bool) -> bool:
        self.counter = self.counter + 1 if bool(success_now) else 0
        return self.counter >= self.hold_steps


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _blend(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    value = np.asarray(start, dtype=np.float64) + _smoothstep(alpha) * (
        np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    )
    if value.shape == (7,):
        quat_norm = float(np.linalg.norm(value[3:7]))
        if quat_norm <= 1e-12:
            raise ValueError("P0-02 pose interpolation produced a zero quaternion")
        value[3:7] /= quat_norm
    return value


def _quat_mul_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    value = np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return value / np.linalg.norm(value)


def _task_pose(nominal: np.ndarray, reset: P002ResetState) -> np.ndarray:
    pose = np.asarray(nominal, dtype=np.float64).copy()
    delta = math.radians(reset.root_yaw_deg - NOMINAL_ROOT_YAW_DEG)
    rotation = np.asarray(
        [[math.cos(delta), -math.sin(delta)], [math.sin(delta), math.cos(delta)]],
        dtype=np.float64,
    )
    pose[:2] = np.asarray([reset.root_x_m, reset.root_y_m]) + rotation @ (
        pose[:2] - NOMINAL_ROOT_XYZ_M[:2]
    )
    yaw_quat = np.asarray(
        [math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)],
        dtype=np.float64,
    )
    pose[3:7] = _quat_mul_wxyz(yaw_quat, pose[3:7])
    return pose


def _door_push_tcp_pose(open_deg: float) -> np.ndarray:
    # Nominal hinge and front-face contact are derived from the scaled source
    # joint anchor and panel thickness, then followed on a true circular arc.
    hinge_xy = NOMINAL_ROOT_XYZ_M[:2] + np.asarray(
        [0.0128, 0.2174], dtype=np.float64
    )
    contact_xy = NOMINAL_ROOT_XYZ_M[:2] + np.asarray(
        [-0.0168, -0.0900], dtype=np.float64
    )
    vector = contact_xy - hinge_xy
    angle = math.radians(float(open_deg))
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    contact = hinge_xy + rotation @ vector
    # Live 20-seed IK calibration places the palm 0.1432 m in front of this
    # valid panel point; the fingers can contact without an overextended arm.
    tcp_xy = contact - rotation @ np.asarray([0.1432, 0.0], dtype=np.float64)
    return np.asarray([tcp_xy[0], tcp_xy[1], 1.20, *_REACHABLE_QUAT_WXYZ])


class P002ScriptedPolicy:
    """Bounded deterministic policy used only for automated acceptance."""

    def __init__(self, reset_state: P002ResetState):
        self.reset_state = reset_state
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float64)

    def action(
        self,
        step: int,
        readback: P002PhysicalReadback | None = None,
        latch: P002LatchStateMachine | None = None,
    ) -> np.ndarray:
        step = int(step)
        # Handle002 sits on the operator-right side.  The former +0.02 m
        # lateral palm target was inherited from the left-hand policy and
        # solved only 13/20 randomized right-arm IK trials.  -0.04 m remains
        # inside the enlarged handle's live contact span and solves 20/20 with
        # at least 0.107 rad of joint-limit margin.
        right_handle_pre = _task_pose(
            np.asarray(
                [
                    float(NOMINAL_ROOT_XYZ_M[0] - 0.19),
                    RIGHT_HANDLE_PALM_NOMINAL_Y_M,
                    1.30,
                    *_REACHABLE_QUAT_WXYZ,
                ]
            ),
            self.reset_state,
        )
        right_handle_press = _task_pose(
            np.asarray(
                [
                    float(NOMINAL_ROOT_XYZ_M[0] - 0.19),
                    RIGHT_HANDLE_PALM_NOMINAL_Y_M,
                    1.15,
                    *_REACHABLE_QUAT_WXYZ,
                ]
            ),
            self.reset_state,
        )
        left_door_pre = _task_pose(
            np.asarray(
                [
                    float(NOMINAL_ROOT_XYZ_M[0] - 0.22),
                    0.12,
                    1.28,
                    *_REACHABLE_QUAT_WXYZ,
                ]
            ),
            self.reset_state,
        )
        left_door_contact = _task_pose(
            _door_push_tcp_pose(0.0), self.reset_state
        )

        if step < 60:
            right_pose = _blend(RIGHT_HOME, right_handle_pre, step / 60.0)
            left_pose = _blend(LEFT_HOME, left_door_pre, step / 60.0)
        elif step < 120:
            right_pose = right_handle_pre
            left_pose = _blend(
                left_door_pre, left_door_contact, (step - 60) / 60.0
            )
        elif step < 190:
            right_pose = _blend(
                right_handle_pre,
                right_handle_press,
                (step - 120) / 70.0,
            )
            left_pose = left_door_contact
        else:
            right_pose = right_handle_press.copy()
            unlocked = bool(latch is not None and not latch.locked)
            if unlocked:
                progress = np.clip((step - 190) / 150.0, 0.0, 1.0)
                desired_open = SCRIPTED_DOOR_GOAL_DEG * _smoothstep(
                    float(progress)
                )
                if readback is not None and np.isfinite(readback.door_angle_rad):
                    live_deg = math.degrees(float(readback.door_angle_rad))
                    desired_open = min(desired_open, live_deg + 5.0)
                left_pose = _task_pose(
                    _door_push_tcp_pose(desired_open), self.reset_state
                )
            else:
                left_pose = left_door_contact

        right_grip = 0.40 * WUJI_POWER_GRIP
        left_grip = 0.35 * WUJI_POWER_GRIP
        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[:7] = right_pose
        action[7:14] = left_pose
        action[14 : 14 + HAND_ACTION_DIM] = right_grip
        action[14 + HAND_ACTION_DIM :] = left_grip
        self.last_action = action
        return action.copy()

    def metadata(self) -> Mapping[str, Any]:
        return {
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "action_dim": ACTION_DIM,
            "reset": self.reset_state.to_dict(),
        }


__all__ = [
    "ACTION_DIM",
    "ASSET_RELATIVE_PATH",
    "ASSET_SCALE",
    "DOOR_BODY_PATH",
    "DOOR_DOF",
    "DOOR_ROOT_PATH",
    "EXPECTED_DOF_NAMES",
    "FRAME_BODY_PATH",
    "FRONT_HANDLE_BODY_PATH",
    "FRONT_HANDLE_DOF",
    "LatchPhase",
    "P002LatchStateMachine",
    "P002PhysicalReadback",
    "P002_ROLE_CONTRACT_VERSION",
    "P002ResetState",
    "P002ScriptedPolicy",
    "P002SuccessTracker",
    "REAR_HANDLE_BODY_PATH",
    "REAR_HANDLE_DOF",
    "RIGHT_HANDLE_PALM_NOMINAL_Y_M",
    "SCRIPTED_DOOR_GOAL_DEG",
    "TASK_ID",
    "TASK_NAME",
    "TASK_PROMPT",
    "evaluate_physical_readback",
    "sample_reset_state",
]
