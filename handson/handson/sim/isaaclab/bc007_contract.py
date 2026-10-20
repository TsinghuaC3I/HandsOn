"""Pure task semantics for B320-BC007.

This module deliberately has no Isaac, Gym, Torch, or USD imports.  The live
adapter turns PhysX measurements into :class:`LongRodFrame` values while the
tests exercise the ordered stage graph and anti-cheat rules deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Sequence

import numpy as np


class LongRodStage(IntEnum):
    APPROACH = 0
    DUAL_GRASP = 1
    LIFTED = 2
    REORIENTING = 3
    TARGET_STABLE = 4


STAGE_NAMES = {
    LongRodStage.APPROACH: "approach",
    LongRodStage.DUAL_GRASP: "dual_grasp",
    LongRodStage.LIFTED: "lifted",
    LongRodStage.REORIENTING: "reorienting",
    LongRodStage.TARGET_STABLE: "target_stable",
}


BC007_REPLAY_ASSIST_VERSION = 1
_ASSIST_OWNER_BIT = 1 << 0
_ASSIST_FRAME_ARMED_BIT = 1 << 1
_ASSIST_RELEASE_PENDING_BIT = 1 << 2


def encode_replay_assist_state(
    *,
    left_owner: bool,
    right_owner: bool,
    left_frame_armed: bool,
    right_frame_armed: bool,
    left_release_pending: bool,
    right_release_pending: bool,
) -> np.ndarray:
    """Encode action-aligned fixed-joint assist state for physical replay."""

    def _flags(owner: bool, frame_armed: bool, release_pending: bool) -> int:
        if release_pending and not owner:
            raise ValueError("BC007 release-pending state requires an active owner")
        return (
            (int(bool(owner)) * _ASSIST_OWNER_BIT)
            | (int(bool(frame_armed)) * _ASSIST_FRAME_ARMED_BIT)
            | (int(bool(release_pending)) * _ASSIST_RELEASE_PENDING_BIT)
        )

    return np.asarray(
        [
            _flags(left_owner, left_frame_armed, left_release_pending),
            _flags(right_owner, right_frame_armed, right_release_pending),
            BC007_REPLAY_ASSIST_VERSION,
        ],
        dtype=np.uint8,
    )


def decode_replay_assist_state(state: Sequence[int]) -> dict[str, dict[str, bool]]:
    """Validate and decode the shared ``uint8[3]`` replay hook payload."""

    array = np.asarray(state)
    if array.shape != (3,) or array.dtype != np.dtype(np.uint8):
        raise ValueError(
            "BC007 physical-replay assist state must have uint8 shape (3,), "
            f"got {array.dtype} {array.shape}"
        )
    if int(array[2]) != BC007_REPLAY_ASSIST_VERSION:
        raise ValueError(
            "BC007 physical-replay assist-state version is unsupported: "
            f"{int(array[2])}"
        )
    known_bits = _ASSIST_OWNER_BIT | _ASSIST_FRAME_ARMED_BIT | _ASSIST_RELEASE_PENDING_BIT
    result: dict[str, dict[str, bool]] = {}
    for side, raw in zip(("left", "right"), array[:2], strict=True):
        flags = int(raw)
        if flags & ~known_bits:
            raise ValueError(f"BC007 {side} replay assist flags are invalid: {flags}")
        owner = bool(flags & _ASSIST_OWNER_BIT)
        release_pending = bool(flags & _ASSIST_RELEASE_PENDING_BIT)
        if release_pending and not owner:
            raise ValueError(
                f"BC007 {side} release-pending state requires an active owner"
            )
        result[side] = {
            "owner": owner,
            "frame_armed": bool(flags & _ASSIST_FRAME_ARMED_BIT),
            "release_pending": release_pending,
        }
    return result


@dataclass(frozen=True)
class LongRodThresholds:
    dual_grasp_frames: int = 3
    role_loss_grace_frames: int = 4
    height_exit_grace_frames: int = 6
    table_contact_grace_frames: int = 2
    severe_tilt_grace_frames: int = 5
    strong_collision_grace_frames: int = 3
    height_band_low_m: float = 1.02
    height_band_high_m: float = 1.14
    rotation_start_deg: float = 12.0
    target_progress_min_deg: float = 75.0
    target_yaw_error_max_deg: float = 8.0
    endpoint_height_difference_max_m: float = 0.03
    severe_endpoint_height_difference_m: float = 0.12
    stable_linear_speed_max_mps: float = 0.08
    stable_angular_speed_max_radps: float = 0.30
    target_position_error_max_m: float = 0.04
    minimum_hand_separation_m: float = 0.25


@dataclass(frozen=True)
class LongRodFrame:
    left_role_owned: bool = False
    right_role_owned: bool = False
    center_height_m: float = 0.0
    yaw_progress_deg: float = 0.0
    yaw_error_deg: float = math.inf
    endpoint_height_difference_m: float = math.inf
    linear_speed_mps: float = math.inf
    angular_speed_radps: float = math.inf
    target_position_error_m: float = math.inf
    hand_separation_m: float = 0.0
    table_contact: bool = False
    strong_collision: bool = False
    required_contact_evidence_available: bool = True
    finite_and_in_workspace: bool = True
    dropped: bool = False


class LongRodEvaluator:
    """Monotonic stage history with bounded contact-loss grace periods."""

    def __init__(self, thresholds: LongRodThresholds | None = None) -> None:
        self.thresholds = thresholds or LongRodThresholds()
        self.reset()

    def reset(self) -> None:
        self.stage = LongRodStage.APPROACH
        self.dual_grasp_streak = 0
        self.role_loss_streak = 0
        self.height_exit_streak = 0
        self.table_contact_streak = 0
        self.severe_tilt_streak = 0
        self.strong_collision_streak = 0
        self.dual_grasp_latched = False
        self.lift_latched = False
        self.rotation_latched = False
        self.failure_reason = ""

    @property
    def failed(self) -> bool:
        return bool(self.failure_reason)

    @property
    def stage_name(self) -> str:
        return STAGE_NAMES[self.stage]

    def _fail(self, reason: str) -> None:
        if not self.failure_reason:
            self.failure_reason = str(reason)

    def update(self, frame: LongRodFrame) -> dict[str, object]:
        if self.failed:
            return self.snapshot(success_now=False)

        t = self.thresholds
        dual_owned = bool(frame.left_role_owned and frame.right_role_owned)
        in_height_band = bool(
            t.height_band_low_m <= frame.center_height_m <= t.height_band_high_m
        )

        if not frame.required_contact_evidence_available:
            self._fail("required_contact_evidence_unavailable")
        elif not frame.finite_and_in_workspace:
            self._fail("nonfinite_or_workspace")
        elif frame.dropped:
            self._fail("rod_dropped")

        self.strong_collision_streak = (
            self.strong_collision_streak + 1 if frame.strong_collision else 0
        )
        if self.strong_collision_streak >= t.strong_collision_grace_frames:
            self._fail("sustained_strong_collision")

        if self.stage >= LongRodStage.LIFTED:
            self.role_loss_streak = 0 if dual_owned else self.role_loss_streak + 1
            self.height_exit_streak = 0 if in_height_band else self.height_exit_streak + 1
            self.table_contact_streak = (
                self.table_contact_streak + 1 if frame.table_contact else 0
            )
            self.severe_tilt_streak = (
                self.severe_tilt_streak + 1
                if frame.endpoint_height_difference_m
                > t.severe_endpoint_height_difference_m
                else 0
            )
            if self.role_loss_streak > t.role_loss_grace_frames:
                self._fail("dual_end_role_lost")
            elif self.height_exit_streak > t.height_exit_grace_frames:
                self._fail("left_height_band")
            elif self.table_contact_streak > t.table_contact_grace_frames:
                self._fail("table_assisted_rotation")
            elif self.severe_tilt_streak > t.severe_tilt_grace_frames:
                self._fail("sustained_excessive_tilt")

        if self.failed:
            return self.snapshot(success_now=False)

        self.dual_grasp_streak = self.dual_grasp_streak + 1 if dual_owned else 0
        if self.stage == LongRodStage.APPROACH:
            if self.dual_grasp_streak >= t.dual_grasp_frames:
                self.dual_grasp_latched = True
                self.stage = LongRodStage.DUAL_GRASP

        if self.stage == LongRodStage.DUAL_GRASP and dual_owned and in_height_band:
            self.lift_latched = True
            self.stage = LongRodStage.LIFTED

        if (
            self.stage == LongRodStage.LIFTED
            and dual_owned
            and in_height_band
            and frame.yaw_progress_deg >= t.rotation_start_deg
        ):
            self.rotation_latched = True
            self.stage = LongRodStage.REORIENTING

        success_now = bool(
            self.stage >= LongRodStage.REORIENTING
            and self.dual_grasp_latched
            and self.lift_latched
            and self.rotation_latched
            and dual_owned
            and in_height_band
            and frame.yaw_progress_deg >= t.target_progress_min_deg
            and abs(frame.yaw_error_deg) <= t.target_yaw_error_max_deg
            and frame.endpoint_height_difference_m
            <= t.endpoint_height_difference_max_m
            and frame.linear_speed_mps <= t.stable_linear_speed_max_mps
            and frame.angular_speed_radps <= t.stable_angular_speed_max_radps
            and frame.target_position_error_m <= t.target_position_error_max_m
            and frame.hand_separation_m >= t.minimum_hand_separation_m
            and not frame.table_contact
            and not frame.strong_collision
        )
        if success_now:
            self.stage = LongRodStage.TARGET_STABLE
        return self.snapshot(success_now=success_now)

    def snapshot(self, *, success_now: bool) -> dict[str, object]:
        return {
            "stage": int(self.stage),
            "stage_name": self.stage_name,
            "dual_grasp_streak": int(self.dual_grasp_streak),
            "role_loss_streak": int(self.role_loss_streak),
            "height_exit_streak": int(self.height_exit_streak),
            "table_contact_streak": int(self.table_contact_streak),
            "severe_tilt_streak": int(self.severe_tilt_streak),
            "strong_collision_streak": int(self.strong_collision_streak),
            "dual_grasp_latched": bool(self.dual_grasp_latched),
            "lift_latched": bool(self.lift_latched),
            "rotation_latched": bool(self.rotation_latched),
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "success_now": bool(success_now),
        }


def quat_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return np.full((3, 3), np.nan, dtype=np.float64)
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rod_axis_world(pose_wxyz: Sequence[float]) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    return quat_wxyz_to_matrix(pose[3:7])[:, 2]


def rod_endpoints(pose_wxyz: Sequence[float], half_length_m: float) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    axis = rod_axis_world(pose)
    center = pose[:3]
    return center - axis * float(half_length_m), center + axis * float(half_length_m)


def rod_axis_yaw_deg(pose_wxyz: Sequence[float]) -> float:
    axis = rod_axis_world(pose_wxyz)
    return math.degrees(math.atan2(float(axis[1]), float(axis[0])))


def wrap_degrees(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def signed_yaw_progress_deg(current_deg: float, initial_deg: float) -> float:
    return wrap_degrees(float(current_deg) - float(initial_deg))


__all__ = [
    "BC007_REPLAY_ASSIST_VERSION",
    "LongRodEvaluator",
    "LongRodFrame",
    "LongRodStage",
    "LongRodThresholds",
    "STAGE_NAMES",
    "decode_replay_assist_state",
    "encode_replay_assist_state",
    "quat_wxyz_to_matrix",
    "rod_axis_world",
    "rod_axis_yaw_deg",
    "rod_endpoints",
    "signed_yaw_progress_deg",
    "wrap_degrees",
]
