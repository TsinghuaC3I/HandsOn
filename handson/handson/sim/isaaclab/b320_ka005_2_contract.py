"""Pure physical-success contract for B320-KA005-2.

The module deliberately has no Isaac or USD imports.  The live B320 runtime
feeds it named PhysX DOF readings and four hand-to-knob contact-force channels
at 120 Hz; CPU tests can exercise the exact same ordered state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any, Mapping

import numpy as np


TASK_ID = "B320-KA005-2"
TASK_NAME = "t38_set_dual_knobs"
DOF_NAMES = ("left_knob_joint", "right_knob_joint")
CONTACT_KEYS = ("LL", "RR", "LR", "RL")

PHYSICS_HZ = 120
CONTACT_ON_N = 0.15
CONTACT_OFF_N = 0.05
CONTACT_GRACE_S = 0.15
REARM_RELEASE_S = 0.25
REARM_TARGET_DISTANCE_RAD = math.radians(30.0)
MOTION_SPEED_RAD_S = 0.12
MOTION_MIN_SUBSTEPS = 3
MOTION_DISPLACEMENT_RAD = math.radians(5.0)
OVERLAP_S = 0.30
TARGET_ENTER_RAD = math.radians(5.0)
TARGET_EXIT_RAD = math.radians(7.0)
TARGET_SPEED_RAD_S = 0.12
TARGET_HOLD_S = 0.25
RELEASE_HOLD_S = 0.25
STABLE_SPEED_RAD_S = 0.05
STABLE_HOLD_S = 0.50
JOINT_LIMIT_RAD = math.radians(100.0)
JOINT_LIMIT_FAILURE_MARGIN_RAD = math.radians(2.0)

INITIAL_TICKS_DEG = (-80.0, -60.0, -40.0, 0.0, 40.0, 60.0, 80.0)
TARGET_TICKS_DEG = (-80.0, -60.0, -40.0, 40.0, 60.0, 80.0)
MIN_TRAVEL_DEG = 45.0
MAX_TRAVEL_DEG = 120.0

# Repeated binary floating-point additions such as 30 * (1 / 120) can land a
# few ulps below the authored duration.  Keep one comparison tolerance for all
# formal timing windows.  It is many orders of magnitude smaller than one
# physics substep, so N - 1 substeps can never satisfy an N-substep window.
_TIME_COMPARISON_EPSILON_S = 1.0e-9


def _time_at_least(elapsed_s: float, threshold_s: float) -> bool:
    return float(elapsed_s) >= float(threshold_s) - _TIME_COMPARISON_EPSILON_S


def _time_exceeds(elapsed_s: float, threshold_s: float) -> bool:
    return float(elapsed_s) > float(threshold_s) + _TIME_COMPARISON_EPSILON_S


class KnobStage(IntEnum):
    WAIT_GRASP = 0
    WAIT_MOTION = 1
    TURNING = 2
    TARGET_HOLD = 3
    WAIT_RELEASE = 4
    STABILIZE = 5
    COMPLETE = 6


@dataclass(frozen=True)
class DualKnobResetSample:
    initial_rad: tuple[float, float]
    target_rad: tuple[float, float]
    fallback_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "initial_rad": list(self.initial_rad),
            "target_rad": list(self.target_rad),
            "fallback_used": bool(self.fallback_used),
        }


def sample_dual_knob_reset(
    rng: Any,
    *,
    randomize: bool,
) -> DualKnobResetSample:
    """Sample signed tick pairs with bounded, non-trivial travel."""
    from .b_eval_setting import setting_for
    selection = setting_for(TASK_NAME)
    if selection:
        profile = selection['profile']
        return DualKnobResetSample(
            initial_rad=tuple(math.radians(x) for x in profile['initial_degrees']),
            target_rad=tuple(math.radians(x) for x in profile['target_degrees']))

    if not randomize:
        return DualKnobResetSample(
            initial_rad=(0.0, 0.0),
            target_rad=(math.radians(60.0), math.radians(-60.0)),
        )
    left_pairs = tuple(
        (initial, target)
        for initial in INITIAL_TICKS_DEG
        for target in TARGET_TICKS_DEG
        if MIN_TRAVEL_DEG <= target - initial <= MAX_TRAVEL_DEG
        and abs(target) <= 85.0
    )
    right_pairs = tuple(
        (initial, target)
        for initial in INITIAL_TICKS_DEG
        for target in TARGET_TICKS_DEG
        if MIN_TRAVEL_DEG <= initial - target <= MAX_TRAVEL_DEG
        and abs(target) <= 85.0
    )
    if not left_pairs or not right_pairs:
        return DualKnobResetSample(
            initial_rad=(0.0, 0.0),
            target_rad=(math.radians(60.0), math.radians(-60.0)),
            fallback_used=True,
        )
    draw_index = getattr(rng, "integers", None) or getattr(rng, "randint")
    selected = (
        left_pairs[int(draw_index(0, len(left_pairs)))],
        right_pairs[int(draw_index(0, len(right_pairs)))],
    )
    return DualKnobResetSample(
        initial_rad=tuple(math.radians(pair[0]) for pair in selected),
        target_rad=tuple(math.radians(pair[1]) for pair in selected),
    )


class DualKnobSyncState:
    """Ordered, no-assist bimanual rotation/release state machine."""

    def __init__(self) -> None:
        self.reset((0.0, 0.0), (math.radians(60.0), math.radians(-60.0)))

    def reset(
        self,
        initial_rad: tuple[float, float] | np.ndarray,
        target_rad: tuple[float, float] | np.ndarray,
    ) -> None:
        self.initial_q = np.asarray(initial_rad, dtype=np.float64).reshape(2).copy()
        self.target_q = np.asarray(target_rad, dtype=np.float64).reshape(2).copy()
        if not np.all(np.isfinite(self.initial_q)) or not np.all(np.isfinite(self.target_q)):
            raise ValueError("Dual-knob reset angles must be finite")
        if np.any(np.abs(self.initial_q) > JOINT_LIMIT_RAD) or np.any(
            np.abs(self.target_q) > JOINT_LIMIT_RAD
        ):
            raise ValueError("Dual-knob reset angles exceed the authored limits")
        self.stage = KnobStage.WAIT_GRASP
        self.sim_time_s = 0.0
        self.attempt_id = 0
        self.attempt_armed = False
        self.attempt_invalid_reason = ""
        self.failure_reason = ""
        self.success = False
        self.contact_truth_available = False
        self.dof_truth_available = False
        self.contact_active = {key: False for key in CONTACT_KEYS}
        self.contact_force_n = {key: 0.0 for key in CONTACT_KEYS}
        self.assigned_contact_seen = np.zeros(2, dtype=bool)
        self.assigned_contact_time = np.full(2, np.nan, dtype=np.float64)
        self.dual_grasp_seen = False
        self.release_to_rearm_s = 0.0
        self.contact_loss_s = np.zeros(2, dtype=np.float64)
        self.attempt_start_q = self.initial_q.copy()
        self.attempt_direction = np.sign(self.target_q - self.initial_q)
        self.required_travel = np.abs(self.target_q - self.initial_q)
        self.last_q = self.initial_q.copy()
        self.last_qd = np.zeros(2, dtype=np.float64)
        self.motion_run = np.zeros(2, dtype=np.int64)
        self.motion_run_start_time = np.full(2, np.nan, dtype=np.float64)
        self.first_motion_time = np.full(2, np.nan, dtype=np.float64)
        self.overlap_s = 0.0
        self.overlap_progress = np.zeros(2, dtype=np.float64)
        self.contact_progress = np.zeros(2, dtype=np.float64)
        self.wrong_direction_progress = np.zeros(2, dtype=np.float64)
        self.sync_motion_qualified = False
        self.target_entry_time = np.full(2, np.nan, dtype=np.float64)
        self.target_seen = np.zeros(2, dtype=bool)
        self.target_hold_s = 0.0
        self.target_hold_complete = False
        self.release_s = 0.0
        self.release_complete = False
        self.stable_s = 0.0

    @property
    def start_skew_s(self) -> float:
        if not np.all(np.isfinite(self.first_motion_time)):
            return -1.0
        return float(abs(self.first_motion_time[0] - self.first_motion_time[1]))

    @property
    def contact_entry_skew_s(self) -> float:
        if not np.all(np.isfinite(self.assigned_contact_time)):
            return -1.0
        return float(abs(self.assigned_contact_time[0] - self.assigned_contact_time[1]))

    @property
    def target_entry_skew_s(self) -> float:
        if not np.all(np.isfinite(self.target_entry_time)):
            return -1.0
        return float(abs(self.target_entry_time[0] - self.target_entry_time[1]))

    @property
    def errors(self) -> np.ndarray:
        return self.target_q - self.last_q

    @property
    def contact_progress_ratio(self) -> np.ndarray:
        return np.clip(
            self.contact_progress / np.maximum(self.required_travel, 1.0e-9),
            0.0,
            1.0,
        )

    def _invalidate_attempt(self, reason: str) -> None:
        self.attempt_invalid_reason = str(reason)
        self.stage = KnobStage.WAIT_GRASP
        self.attempt_armed = False
        self.release_to_rearm_s = 0.0
        self.assigned_contact_time[:] = np.nan
        self.contact_loss_s[:] = 0.0
        self.motion_run[:] = 0
        self.motion_run_start_time[:] = np.nan
        self.first_motion_time[:] = np.nan
        self.overlap_s = 0.0
        self.overlap_progress[:] = 0.0
        self.contact_progress[:] = 0.0
        self.wrong_direction_progress[:] = 0.0
        self.sync_motion_qualified = False
        self.target_entry_time[:] = np.nan
        self.target_seen[:] = False
        self.target_hold_s = 0.0
        self.target_hold_complete = False
        self.release_s = 0.0
        self.release_complete = False
        self.stable_s = 0.0

    def _start_attempt(self, q: np.ndarray) -> None:
        self.attempt_id += 1
        self.attempt_armed = True
        self.attempt_invalid_reason = ""
        self.attempt_start_q = q.copy()
        delta = self.target_q - q
        self.attempt_direction = np.sign(delta)
        self.required_travel = np.abs(delta)
        self.stage = KnobStage.WAIT_MOTION
        self.release_to_rearm_s = 0.0
        self.contact_loss_s[:] = 0.0
        self.motion_run[:] = 0
        self.motion_run_start_time[:] = np.nan
        self.first_motion_time[:] = np.nan
        self.overlap_s = 0.0
        self.overlap_progress[:] = 0.0
        self.contact_progress[:] = 0.0
        self.wrong_direction_progress[:] = 0.0
        self.sync_motion_qualified = False
        self.target_entry_time[:] = np.nan
        self.target_seen[:] = False
        self.target_hold_s = 0.0
        self.target_hold_complete = False
        self.release_s = 0.0
        self.release_complete = False
        self.stable_s = 0.0
        self.dual_grasp_seen = bool(np.all(self.assigned_contact_seen))

    def _update_contact_hysteresis(
        self,
        forces: Mapping[str, float],
        *,
        available: bool,
    ) -> None:
        self.contact_truth_available = bool(available)
        for key in CONTACT_KEYS:
            force = float(forces.get(key, 0.0))
            if not math.isfinite(force):
                available = False
                force = 0.0
            self.contact_force_n[key] = max(0.0, force)
        if not available:
            self.contact_truth_available = False
            self.contact_active = {key: False for key in CONTACT_KEYS}
            return
        for key in CONTACT_KEYS:
            force = self.contact_force_n[key]
            if self.contact_active[key]:
                self.contact_active[key] = force > CONTACT_OFF_N
            else:
                self.contact_active[key] = force >= CONTACT_ON_N

    def prime_recording_origin_truth(
        self,
        q_rad: tuple[float, float] | np.ndarray,
        qd_rad_s: tuple[float, float] | np.ndarray,
        contact_force_n: Mapping[str, float],
    ) -> None:
        """Publish accepted action-0 truth without advancing task time.

        Recording-origin settling happens before the operator trajectory and
        must not count toward grasp rearming, motion qualification, or any
        success timing window.  The runtime calls this only after resetting the
        state machine around the final accepted knob coordinates.
        """

        if (
            self.stage != KnobStage.WAIT_GRASP
            or self.sim_time_s != 0.0
            or self.attempt_id != 0
            or self.attempt_armed
            or self.success
            or bool(self.failure_reason)
        ):
            raise RuntimeError(
                "Dual-knob origin truth can prime only a fresh reset state."
            )
        q = np.asarray(q_rad, dtype=np.float64).reshape(2)
        qd = np.asarray(qd_rad_s, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
            raise ValueError("Dual-knob recording-origin truth must be finite")
        if set(contact_force_n) != set(CONTACT_KEYS):
            raise ValueError(
                "Dual-knob recording-origin truth requires LL/RR/LR/RL."
            )
        forces = {
            key: float(contact_force_n[key]) for key in CONTACT_KEYS
        }
        if not all(math.isfinite(value) for value in forces.values()):
            raise ValueError(
                "Dual-knob recording-origin contact truth must be finite"
            )
        self.dof_truth_available = True
        self._update_contact_hysteresis(forces, available=True)
        self.last_q = q.copy()
        self.last_qd = qd.copy()

    def update(
        self,
        q_rad: tuple[float, float] | np.ndarray,
        qd_rad_s: tuple[float, float] | np.ndarray,
        contact_force_n: Mapping[str, float],
        *,
        dof_available: bool,
        contact_available: bool,
        dt: float,
    ) -> None:
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"Invalid dual-knob physics dt: {dt}")
        q = np.asarray(q_rad, dtype=np.float64).reshape(2)
        qd = np.asarray(qd_rad_s, dtype=np.float64).reshape(2)
        self.sim_time_s += dt
        self.dof_truth_available = bool(dof_available)
        self._update_contact_hysteresis(
            contact_force_n,
            available=contact_available,
        )
        if self.failure_reason:
            # Failure is a terminal first-cause latch.  The runtime can still
            # receive the remaining solver callbacks in the current 30 Hz
            # control frame, but a later unavailable channel must not replace
            # the physical reason captured by the first failed substep.
            if np.all(np.isfinite(q)) and np.all(np.isfinite(qd)):
                self.last_q = q.copy()
                self.last_qd = qd.copy()
            return
        if not self.dof_truth_available or not np.all(np.isfinite(q)) or not np.all(
            np.isfinite(qd)
        ):
            self.failure_reason = "dof_truth_unavailable"
            return
        if not self.contact_truth_available:
            self.failure_reason = "contact_truth_unavailable"
            return
        if np.any(np.abs(q) > JOINT_LIMIT_RAD + JOINT_LIMIT_FAILURE_MARGIN_RAD):
            self.failure_reason = "joint_limit_exceeded"
            return
        if self.success or self.failure_reason:
            self.last_q = q.copy()
            self.last_qd = qd.copy()
            return

        delta_q = q - self.last_q
        self.last_q = q.copy()
        self.last_qd = qd.copy()
        assigned = np.asarray(
            [self.contact_active["LL"], self.contact_active["RR"]], dtype=bool
        )
        cross = np.asarray(
            [self.contact_active["LR"], self.contact_active["RL"]], dtype=bool
        )
        any_contact = bool(any(self.contact_active.values()))
        self.assigned_contact_seen |= assigned
        if self.stage != KnobStage.WAIT_GRASP:
            for index in range(2):
                if assigned[index] and not math.isfinite(
                    self.assigned_contact_time[index]
                ):
                    self.assigned_contact_time[index] = self.sim_time_s
            self.dual_grasp_seen = bool(np.all(self.assigned_contact_seen))

        if self.stage == KnobStage.WAIT_GRASP:
            rearmed = _time_at_least(self.release_to_rearm_s, REARM_RELEASE_S)
            if any_contact:
                if not rearmed and not bool(
                    np.any(np.isfinite(self.assigned_contact_time))
                ):
                    self.release_to_rearm_s = 0.0
            else:
                self.release_to_rearm_s += dt
                self.assigned_contact_time[:] = np.nan
            far_enough = bool(
                np.all(np.abs(self.target_q - q) >= REARM_TARGET_DISTANCE_RAD)
            )
            if rearmed and far_enough and not bool(np.any(cross)):
                for index in range(2):
                    if assigned[index] and not math.isfinite(
                        self.assigned_contact_time[index]
                    ):
                        self.assigned_contact_time[index] = self.sim_time_s
                if bool(np.any(assigned)):
                    self._start_attempt(q)
            return

        if bool(np.any(cross)):
            self._invalidate_attempt("cross_hand_contact")
            return

        for index in range(2):
            if assigned[index]:
                self.contact_loss_s[index] = 0.0
            else:
                self.contact_loss_s[index] += dt
        directed_delta = self.attempt_direction * delta_q
        for index in range(2):
            if assigned[index]:
                self.contact_progress[index] += max(0.0, float(directed_delta[index]))
                self.wrong_direction_progress[index] += max(
                    0.0, -float(directed_delta[index])
                )
        directed_speed = self.attempt_direction * qd
        moving = assigned & (directed_speed > MOTION_SPEED_RAD_S)
        for index in range(2):
            if moving[index]:
                if self.motion_run[index] == 0:
                    self.motion_run_start_time[index] = self.sim_time_s
                self.motion_run[index] += 1
            else:
                self.motion_run[index] = 0
                self.motion_run_start_time[index] = np.nan
            displacement = self.attempt_direction[index] * (
                q[index] - self.attempt_start_q[index]
            )
            if (
                not math.isfinite(self.first_motion_time[index])
                and self.motion_run[index] >= MOTION_MIN_SUBSTEPS
                and displacement >= MOTION_DISPLACEMENT_RAD
            ):
                # The displacement/run filters qualify intentional motion, but
                # start skew is defined by the onset of that qualifying run,
                # not by the later instant at which it crosses five degrees.
                onset = self.motion_run_start_time[index]
                self.first_motion_time[index] = (
                    float(onset) if math.isfinite(onset) else self.sim_time_s
                )

        finite_starts = np.isfinite(self.first_motion_time)
        if bool(np.any(finite_starts)) and self.stage == KnobStage.WAIT_MOTION:
            self.stage = KnobStage.TURNING

        if bool(np.all(moving)):
            self.overlap_s += dt
            self.overlap_progress += np.maximum(directed_delta, 0.0)
        directed_displacement = self.attempt_direction * (q - self.attempt_start_q)
        self.sync_motion_qualified = bool(
            np.all(finite_starts)
            and np.all(directed_displacement >= MOTION_DISPLACEMENT_RAD)
        )

        errors = np.abs(self.target_q - q)
        target_candidate = (errors <= TARGET_ENTER_RAD) & (
            np.abs(qd) <= TARGET_SPEED_RAD_S
        )
        for index in range(2):
            if (
                target_candidate[index]
                and math.isfinite(self.first_motion_time[index])
                and not self.target_seen[index]
            ):
                self.target_seen[index] = True
                self.target_entry_time[index] = self.sim_time_s
        if bool(np.all(self.target_seen)):
            self.stage = max(self.stage, KnobStage.TARGET_HOLD)

        if self.stage == KnobStage.TARGET_HOLD:
            target_hold_valid = bool(
                np.all(errors <= TARGET_EXIT_RAD)
                and np.all(np.abs(qd) <= TARGET_SPEED_RAD_S)
            )
            if target_hold_valid:
                self.target_hold_s += dt
            else:
                self.target_hold_s = 0.0
                self.target_seen[:] = False
                self.target_entry_time[:] = np.nan
                self.stage = KnobStage.TURNING
            if _time_at_least(self.target_hold_s, TARGET_HOLD_S):
                self.target_hold_complete = True
                self.stage = KnobStage.WAIT_RELEASE
                # Keep ordered timing exact: the substep that completes the
                # target hold must not also count toward the release window.
                return

        if self.stage == KnobStage.WAIT_RELEASE:
            if bool(np.any(errors > TARGET_EXIT_RAD)):
                self._invalidate_attempt("target_lost_before_release")
                return
            self.release_s = self.release_s + dt if not any_contact else 0.0
            if _time_at_least(self.release_s, RELEASE_HOLD_S):
                self.release_complete = True
                self.stage = KnobStage.STABILIZE
                # Likewise, begin the stationary post-release window on the
                # next solver substep instead of double-counting this one.
                return

        if self.stage == KnobStage.STABILIZE:
            if bool(np.any(errors > TARGET_EXIT_RAD)):
                self._invalidate_attempt("target_lost_after_release")
                return
            stable = bool(
                not any_contact and np.all(np.abs(qd) <= STABLE_SPEED_RAD_S)
            )
            self.stable_s = self.stable_s + dt if stable else 0.0
            if _time_at_least(self.stable_s, STABLE_HOLD_S):
                self.success = True
                self.stage = KnobStage.COMPLETE

    def condition_flags(self) -> tuple[bool, ...]:
        return (
            bool(self.dof_truth_available and self.contact_truth_available),
            bool(self.dual_grasp_seen and np.all(self.assigned_contact_seen)),
            bool(self.sync_motion_qualified),
            bool(self.target_hold_complete),
            bool(self.release_complete),
            bool(self.success),
        )

    def metrics(self) -> dict[str, float | int | bool]:
        errors = self.errors
        return {
            "stage": int(self.stage),
            "attempt_id": int(self.attempt_id),
            "attempt_armed": bool(self.attempt_armed),
            "q_left_rad": float(self.last_q[0]),
            "q_right_rad": float(self.last_q[1]),
            "qd_left_rad_s": float(self.last_qd[0]),
            "qd_right_rad_s": float(self.last_qd[1]),
            "target_left_rad": float(self.target_q[0]),
            "target_right_rad": float(self.target_q[1]),
            "error_left_rad": float(errors[0]),
            "error_right_rad": float(errors[1]),
            "contact_entry_skew_s": float(self.contact_entry_skew_s),
            "start_skew_s": float(self.start_skew_s),
            "overlap_s": float(self.overlap_s),
            "contact_progress_left": float(self.contact_progress_ratio[0]),
            "contact_progress_right": float(self.contact_progress_ratio[1]),
            "target_entry_skew_s": float(self.target_entry_skew_s),
            "target_hold_s": float(self.target_hold_s),
            "release_s": float(self.release_s),
            "stable_s": float(self.stable_s),
            "contact_truth_available": bool(self.contact_truth_available),
            "dof_truth_available": bool(self.dof_truth_available),
            "success_now": bool(self.success),
        }

    def joint_observation(self) -> np.ndarray:
        return np.asarray(
            [self.last_q[0], self.last_q[1], self.last_qd[0], self.last_qd[1]],
            dtype=np.float64,
        )

    def target_observation(self) -> np.ndarray:
        errors = self.errors
        return np.asarray(
            [
                self.target_q[0],
                self.target_q[1],
                errors[0],
                errors[1],
                float(self.dof_truth_available),
                float(self.dof_truth_available),
            ],
            dtype=np.float64,
        )

    def contact_observation(self) -> np.ndarray:
        return np.asarray(
            [
                *[float(self.contact_active[key]) for key in CONTACT_KEYS],
                *[self.contact_force_n[key] for key in CONTACT_KEYS],
            ],
            dtype=np.float64,
        )

    def stage_observation(self) -> np.ndarray:
        one_hot = np.zeros(7, dtype=np.float64)
        one_hot[int(self.stage)] = 1.0
        return np.asarray(
            [
                *one_hot,
                float(math.isfinite(self.first_motion_time[0])),
                float(math.isfinite(self.first_motion_time[1])),
                float(self.dual_grasp_seen),
                float(self.contact_progress_ratio[0]),
                float(self.contact_progress_ratio[1]),
                float(self.start_skew_s),
                float(self.overlap_s),
                float(self.target_seen[0]),
                float(self.target_seen[1]),
                float(self.target_entry_skew_s),
                min(1.0, self.target_hold_s / TARGET_HOLD_S),
                min(1.0, self.release_s / RELEASE_HOLD_S),
                min(1.0, self.stable_s / STABLE_HOLD_S),
            ],
            dtype=np.float64,
        )


__all__ = [
    "CONTACT_KEYS",
    "DOF_NAMES",
    "DualKnobResetSample",
    "DualKnobSyncState",
    "KnobStage",
    "PHYSICS_HZ",
    "TASK_ID",
    "TASK_NAME",
    "sample_dual_knob_reset",
]
