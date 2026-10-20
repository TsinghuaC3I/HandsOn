"""Pure stage/failure contract for B320-BC005.

The module intentionally has no Isaac, Torch, Gym, or MuJoCo imports.  Runtime
adapters turn physical measurements into :class:`LoadedTrayFrame` values;
tests can then prove ordered success and anti-cheat behavior deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np


class LoadedTrayStage(IntEnum):
    APPROACH = 0
    DUAL_GRASP = 1
    LIFTED = 2
    GATE_CROSSING = 3
    GATE_CROSSED = 4
    PLACED_STABLE = 5


STAGE_NAMES = {
    LoadedTrayStage.APPROACH: "approach",
    LoadedTrayStage.DUAL_GRASP: "dual_grasp",
    LoadedTrayStage.LIFTED: "lifted",
    LoadedTrayStage.GATE_CROSSING: "gate_crossing",
    LoadedTrayStage.GATE_CROSSED: "gate_crossed",
    LoadedTrayStage.PLACED_STABLE: "placed_stable",
}


@dataclass(frozen=True)
class LoadedTrayGateThresholds:
    dual_grasp_frames: int = 3
    crossing_role_loss_grace_frames: int = 2
    payload_loss_grace_frames: int = 2
    severe_gate_collision_frames: int = 3
    excessive_non_grip_contact_frames: int = 6
    tilt_failure_frames: int = 8
    tray_lift_clearance_m: float = 0.055
    severe_gate_penetration_m: float = 0.020


@dataclass(frozen=True)
class LoadedTrayFrame:
    left_role_owned: bool = False
    right_role_owned: bool = False
    tray_lift_clearance_m: float = 0.0
    tray_overlaps_gate: bool = False
    all_beyond_gate: bool = False
    payloads_on_tray: bool = True
    payload_dropped: bool = False
    final_stable: bool = False
    tray_tilt_deg: float = 0.0
    severe_gate_contact: bool = False
    # Correct rim contact is the intended manipulation interface and remains
    # diagnostic even when its aggregate normal force is high.  Only a
    # separately measured, collider-backed non-grip tray contact can arm the
    # terminal safety debounce below.
    excessive_handle_contact: bool = False
    excessive_non_grip_contact: bool = False
    max_gate_penetration_m: float = 0.0
    finite_and_in_workspace: bool = True


class LoadedTrayGateEvaluator:
    """Monotonic task history with debounced physical failure latches."""

    def __init__(
        self,
        thresholds: LoadedTrayGateThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or LoadedTrayGateThresholds()
        self.reset()

    def reset(self) -> None:
        self.stage = LoadedTrayStage.APPROACH
        self.dual_grasp_streak = 0
        self.crossing_role_loss_streak = 0
        self.payload_loss_streak = 0
        self.severe_gate_collision_streak = 0
        self.excessive_non_grip_contact_streak = 0
        self.tilt_failure_streak = 0
        self.crossing_started = False
        self.crossing_ownership_valid = True
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

    def update(self, frame: LoadedTrayFrame) -> dict[str, object]:
        if self.failed:
            return self.snapshot(success_now=False)

        dual_owned = bool(frame.left_role_owned and frame.right_role_owned)
        if not frame.finite_and_in_workspace:
            self._fail("nonfinite_or_workspace")
        elif frame.payload_dropped:
            self._fail("payload_dropped")
        elif frame.max_gate_penetration_m > self.thresholds.severe_gate_penetration_m:
            self._fail("deep_gate_penetration")

        self.tilt_failure_streak = (
            self.tilt_failure_streak + 1
            if frame.tray_tilt_deg > 28.0
            else 0
        )
        if self.tilt_failure_streak >= self.thresholds.tilt_failure_frames:
            self._fail("sustained_excessive_tilt")

        self.severe_gate_collision_streak = (
            self.severe_gate_collision_streak + 1
            if frame.severe_gate_contact
            else 0
        )
        if (
            self.severe_gate_collision_streak
            >= self.thresholds.severe_gate_collision_frames
        ):
            self._fail("sustained_severe_gate_collision")

        non_grip_guard_armed = self.stage >= LoadedTrayStage.DUAL_GRASP
        self.excessive_non_grip_contact_streak = (
            self.excessive_non_grip_contact_streak + 1
            if non_grip_guard_armed and frame.excessive_non_grip_contact
            else 0
        )
        if (
            self.excessive_non_grip_contact_streak
            >= self.thresholds.excessive_non_grip_contact_frames
        ):
            self._fail("sustained_excessive_non_grip_tray_contact")

        if self.stage >= LoadedTrayStage.LIFTED and not frame.payloads_on_tray:
            self.payload_loss_streak += 1
        else:
            self.payload_loss_streak = 0
        if self.payload_loss_streak > self.thresholds.payload_loss_grace_frames:
            self._fail("payload_left_tray")

        if self.failed:
            return self.snapshot(success_now=False)

        if self.stage < LoadedTrayStage.GATE_CROSSING:
            self.dual_grasp_streak = self.dual_grasp_streak + 1 if dual_owned else 0
            if not dual_owned and self.stage in {
                LoadedTrayStage.DUAL_GRASP,
                LoadedTrayStage.LIFTED,
            }:
                self.stage = LoadedTrayStage.APPROACH
            if (
                self.dual_grasp_streak
                >= self.thresholds.dual_grasp_frames
                and self.stage < LoadedTrayStage.DUAL_GRASP
            ):
                self.stage = LoadedTrayStage.DUAL_GRASP
            if (
                self.stage >= LoadedTrayStage.DUAL_GRASP
                and dual_owned
                and frame.tray_lift_clearance_m
                >= self.thresholds.tray_lift_clearance_m
            ):
                self.stage = LoadedTrayStage.LIFTED
            if (
                self.stage >= LoadedTrayStage.LIFTED
                and dual_owned
                and frame.tray_overlaps_gate
            ):
                self.stage = LoadedTrayStage.GATE_CROSSING
                self.crossing_started = True

        if self.stage == LoadedTrayStage.GATE_CROSSING:
            self.crossing_role_loss_streak = (
                0 if dual_owned else self.crossing_role_loss_streak + 1
            )
            if (
                self.crossing_role_loss_streak
                > self.thresholds.crossing_role_loss_grace_frames
            ):
                self.crossing_ownership_valid = False
                self._fail("crossing_role_lost")
            elif (
                dual_owned
                and frame.all_beyond_gate
                and frame.payloads_on_tray
                and self.crossing_ownership_valid
            ):
                self.stage = LoadedTrayStage.GATE_CROSSED

        success_now = bool(
            not self.failed
            and self.stage >= LoadedTrayStage.GATE_CROSSED
            and self.crossing_started
            and self.crossing_ownership_valid
            and frame.all_beyond_gate
            and frame.payloads_on_tray
            and frame.final_stable
        )
        if success_now:
            self.stage = LoadedTrayStage.PLACED_STABLE
        return self.snapshot(success_now=success_now)

    def snapshot(self, *, success_now: bool) -> dict[str, object]:
        return {
            "stage": int(self.stage),
            "stage_name": self.stage_name,
            "dual_grasp_streak": int(self.dual_grasp_streak),
            "crossing_role_loss_streak": int(self.crossing_role_loss_streak),
            "payload_loss_streak": int(self.payload_loss_streak),
            "severe_gate_collision_streak": int(
                self.severe_gate_collision_streak
            ),
            "excessive_non_grip_contact_streak": int(
                self.excessive_non_grip_contact_streak
            ),
            # Stable legacy key for existing dashboards; it now mirrors the
            # non-grip-only guard and never counts intended rim contact.
            "excessive_handle_contact_streak": int(
                self.excessive_non_grip_contact_streak
            ),
            "non_grip_contact_guard_armed": bool(
                self.stage >= LoadedTrayStage.DUAL_GRASP
            ),
            "tilt_failure_streak": int(self.tilt_failure_streak),
            "crossing_started": bool(self.crossing_started),
            "crossing_ownership_valid": bool(self.crossing_ownership_valid),
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "success_now": bool(success_now),
        }


def quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return np.full((3, 3), np.nan, dtype=np.float64)
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix(pose_wxyz: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_wxyz_to_matrix(pose[3:7])
    matrix[:3, 3] = pose[:3]
    return matrix


def rpy_degrees(pose_wxyz: np.ndarray) -> tuple[float, float, float]:
    rotation = quat_wxyz_to_matrix(np.asarray(pose_wxyz, dtype=np.float64)[3:7])
    if not np.all(np.isfinite(rotation)):
        return math.inf, math.inf, math.inf
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def world_aabb_half_extents(
    pose_wxyz: np.ndarray,
    local_half_extents: np.ndarray,
) -> np.ndarray:
    rotation = quat_wxyz_to_matrix(np.asarray(pose_wxyz, dtype=np.float64)[3:7])
    half = np.asarray(local_half_extents, dtype=np.float64).reshape(3)
    return np.abs(rotation) @ half


def cylinder_axis_extent(
    pose_wxyz: np.ndarray,
    *,
    radius: float,
    half_height: float,
    world_axis: int,
) -> float:
    rotation = quat_wxyz_to_matrix(np.asarray(pose_wxyz, dtype=np.float64)[3:7])
    axis_component = float(np.clip(abs(rotation[int(world_axis), 2]), 0.0, 1.0))
    return float(
        half_height * axis_component
        + radius * math.sqrt(max(0.0, 1.0 - axis_component * axis_component))
    )


__all__ = [
    "LoadedTrayFrame",
    "LoadedTrayGateEvaluator",
    "LoadedTrayGateThresholds",
    "LoadedTrayStage",
    "STAGE_NAMES",
    "cylinder_axis_extent",
    "pose_matrix",
    "quat_wxyz_to_matrix",
    "rpy_degrees",
    "world_aabb_half_extents",
]
