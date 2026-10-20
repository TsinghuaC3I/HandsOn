"""Simulator-independent contract for synchronized dual-knob setting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


TARGET_TOLERANCE_DEG = 5.0
LINEAR_RELATION_TOLERANCE_DEG = 3.0
VELOCITY_TOLERANCE_DEG_S = 5.0
DIRECTION_HYSTERESIS_DEG = 2.0
FORBIDDEN_DIRECTION_DEG = 5.0
JOINT_LIMIT_DEG = 100.0
JOINT_LIMIT_MARGIN_DEG = 0.5
CONTACT_THRESHOLD_N = 0.15
CROSS_CONTACT_THRESHOLD_N = 0.15
CROSS_CONTACT_FAILURE_FRAMES = 15
FORBIDDEN_DIRECTION_FAILURE_FRAMES = 15
START_SYNCHRONY_SECONDS = 0.8
ENTRY_SYNCHRONY_SECONDS = 1.0
HOLD_FRAMES = 15


@dataclass(frozen=True)
class DualKnobTarget:
    family: str
    parameter_1: float
    parameter_2: float
    parameter_3: float
    x_target_deg: float
    y_target_deg: float
    label: str

    def __post_init__(self) -> None:
        if self.family not in {"ratio_total", "sum_difference"}:
            raise ValueError(f"Unsupported relation family: {self.family!r}")
        values = (
            self.parameter_1,
            self.parameter_2,
            self.parameter_3,
            self.x_target_deg,
            self.y_target_deg,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Relation target parameters must be finite")
        if not (20.0 <= self.x_target_deg <= 65.0):
            raise ValueError(f"Left target is outside the validated range: {self.x_target_deg}")
        if not (20.0 <= self.y_target_deg <= 65.0):
            raise ValueError(f"Right target is outside the validated range: {self.y_target_deg}")
        if self.family == "ratio_total":
            a, b, total = self.parameter_1, self.parameter_2, self.parameter_3
            if a <= 0.0 or b <= 0.0 or total <= 0.0:
                raise ValueError("Ratio and total parameters must be positive")
            expected_x = total * a / (a + b)
            expected_y = total * b / (a + b)
        else:
            total, difference = self.parameter_1, self.parameter_2
            expected_x = 0.5 * (total + difference)
            expected_y = 0.5 * (total - difference)
        if not math.isclose(self.x_target_deg, expected_x, abs_tol=1e-9):
            raise ValueError("Left target does not match the relation parameters")
        if not math.isclose(self.y_target_deg, expected_y, abs_tol=1e-9):
            raise ValueError("Right target does not match the relation parameters")

    @property
    def family_one_hot(self) -> tuple[float, float]:
        return (1.0, 0.0) if self.family == "ratio_total" else (0.0, 1.0)

    @property
    def parameters(self) -> tuple[float, float, float]:
        return (self.parameter_1, self.parameter_2, self.parameter_3)


def ratio_total_target(a: int, b: int, total_deg: float) -> DualKnobTarget:
    total = float(total_deg)
    x_target = total * float(a) / float(a + b)
    y_target = total * float(b) / float(a + b)
    return DualKnobTarget(
        family="ratio_total",
        parameter_1=float(a),
        parameter_2=float(b),
        parameter_3=total,
        x_target_deg=x_target,
        y_target_deg=y_target,
        label=f"x:y={a}:{b}, x+y={total:g}deg",
    )


def sum_difference_target(total_deg: float, difference_deg: float) -> DualKnobTarget:
    total = float(total_deg)
    difference = float(difference_deg)
    return DualKnobTarget(
        family="sum_difference",
        parameter_1=total,
        parameter_2=difference,
        parameter_3=0.0,
        x_target_deg=0.5 * (total + difference),
        y_target_deg=0.5 * (total - difference),
        label=f"x+y={total:g}deg, x-y={difference:g}deg",
    )


DUAL_KNOB_TARGETS = (
    sum_difference_target(60.0, -10.0),  # KA005 canary: x=25, y=35
    sum_difference_target(90.0, -20.0),  # x=35, y=55
    sum_difference_target(100.0, 20.0),  # x=60, y=40
    ratio_total_target(1, 2, 75.0),       # x=25, y=50
    ratio_total_target(2, 3, 100.0),      # x=40, y=60
    ratio_total_target(3, 2, 100.0),      # x=60, y=40
)


def relation_residuals(
    target: DualKnobTarget,
    x_deg: float,
    y_deg: float,
) -> tuple[float, float, float]:
    """Return target-independent primary/secondary residuals and tolerance."""
    x = float(x_deg)
    y = float(y_deg)
    if target.family == "ratio_total":
        a, b, total = target.parameters
        ratio_cross = b * x - a * y
        total_residual = x + y - total
        ratio_tolerance = max(a, b) * LINEAR_RELATION_TOLERANCE_DEG
        return ratio_cross, total_residual, ratio_tolerance
    total, difference, _unused = target.parameters
    return (
        x + y - total,
        x - y - difference,
        LINEAR_RELATION_TOLERANCE_DEG,
    )


def evaluate_dual_knob_reading(
    *,
    target: DualKnobTarget,
    q_left_rad: float,
    q_right_rad: float,
    qd_left_rad_s: float,
    qd_right_rad_s: float,
    left_assigned_force_n: float,
    right_assigned_force_n: float,
    left_cross_force_n: float,
    right_cross_force_n: float,
) -> dict[str, Any]:
    values = (
        q_left_rad,
        q_right_rad,
        qd_left_rad_s,
        qd_right_rad_s,
        left_assigned_force_n,
        right_assigned_force_n,
        left_cross_force_n,
        right_cross_force_n,
    )
    finite = all(math.isfinite(float(value)) for value in values)
    if not finite:
        return {
            "finite": False,
            "x_deg": math.nan,
            "y_deg": math.nan,
            "left_target_ok": False,
            "right_target_ok": False,
            "relation_ok": False,
            "opposite_directions_ok": False,
            "assigned_contacts_ok": False,
            "cross_contacts_clear": False,
            "low_velocity_ok": False,
            "hard_limit_ok": False,
            "instantaneous_hold_ok": False,
            "relation_primary_residual": math.inf,
            "relation_secondary_residual": math.inf,
            "relation_primary_tolerance": LINEAR_RELATION_TOLERANCE_DEG,
            "relation_secondary_tolerance": LINEAR_RELATION_TOLERANCE_DEG,
        }

    q_left_deg = math.degrees(float(q_left_rad))
    q_right_deg = math.degrees(float(q_right_rad))
    qd_left_deg_s = math.degrees(float(qd_left_rad_s))
    qd_right_deg_s = math.degrees(float(qd_right_rad_s))
    # Joint signs are expressed in the panel articulation frame.  From the
    # operator's front-facing view, positive left motion is counter-clockwise
    # and negative right motion is clockwise.
    x_deg = q_left_deg
    y_deg = -q_right_deg
    primary, secondary, primary_tolerance = relation_residuals(
        target, x_deg, y_deg
    )
    left_target_ok = abs(x_deg - target.x_target_deg) <= TARGET_TOLERANCE_DEG
    right_target_ok = abs(y_deg - target.y_target_deg) <= TARGET_TOLERANCE_DEG
    relation_ok = bool(
        abs(primary) <= primary_tolerance
        and abs(secondary) <= LINEAR_RELATION_TOLERANCE_DEG
    )
    directions_ok = bool(
        x_deg > DIRECTION_HYSTERESIS_DEG
        and y_deg > DIRECTION_HYSTERESIS_DEG
    )
    assigned_contacts_ok = bool(
        left_assigned_force_n >= CONTACT_THRESHOLD_N
        and right_assigned_force_n >= CONTACT_THRESHOLD_N
    )
    cross_contacts_clear = bool(
        left_cross_force_n < CROSS_CONTACT_THRESHOLD_N
        and right_cross_force_n < CROSS_CONTACT_THRESHOLD_N
    )
    low_velocity_ok = bool(
        abs(qd_left_deg_s) <= VELOCITY_TOLERANCE_DEG_S
        and abs(qd_right_deg_s) <= VELOCITY_TOLERANCE_DEG_S
    )
    hard_limit_ok = bool(
        abs(q_left_deg) <= JOINT_LIMIT_DEG + JOINT_LIMIT_MARGIN_DEG
        and abs(q_right_deg) <= JOINT_LIMIT_DEG + JOINT_LIMIT_MARGIN_DEG
    )
    instantaneous_hold_ok = bool(
        left_target_ok
        and right_target_ok
        and relation_ok
        and directions_ok
        and assigned_contacts_ok
        and cross_contacts_clear
        and low_velocity_ok
        and hard_limit_ok
    )
    return {
        "finite": True,
        "q_left_deg": q_left_deg,
        "q_right_deg": q_right_deg,
        "qd_left_deg_s": qd_left_deg_s,
        "qd_right_deg_s": qd_right_deg_s,
        "x_deg": x_deg,
        "y_deg": y_deg,
        "left_target_ok": bool(left_target_ok),
        "right_target_ok": bool(right_target_ok),
        "relation_ok": bool(relation_ok),
        "opposite_directions_ok": bool(directions_ok),
        "assigned_contacts_ok": bool(assigned_contacts_ok),
        "cross_contacts_clear": bool(cross_contacts_clear),
        "low_velocity_ok": bool(low_velocity_ok),
        "hard_limit_ok": bool(hard_limit_ok),
        "instantaneous_hold_ok": bool(instantaneous_hold_ok),
        "relation_primary_residual": float(primary),
        "relation_secondary_residual": float(secondary),
        "relation_primary_tolerance": float(primary_tolerance),
        "relation_secondary_tolerance": float(LINEAR_RELATION_TOLERANCE_DEG),
        "left_assigned_force_n": float(left_assigned_force_n),
        "right_assigned_force_n": float(right_assigned_force_n),
        "left_cross_force_n": float(left_cross_force_n),
        "right_cross_force_n": float(right_cross_force_n),
    }


class DualKnobEpisodeTracker:
    """Latch event timing and require an uninterrupted physical hold."""

    def __init__(self, target: DualKnobTarget, *, control_dt: float = 1.0 / 30.0):
        if not math.isfinite(float(control_dt)) or control_dt <= 0.0:
            raise ValueError("control_dt must be finite and positive")
        self.control_dt = float(control_dt)
        self.reset(target)

    def reset(self, target: DualKnobTarget) -> None:
        self.target = target
        self.left_start_frame: int | None = None
        self.right_start_frame: int | None = None
        self.left_entry_frame: int | None = None
        self.right_entry_frame: int | None = None
        self.cross_contact_frames = 0
        self.forbidden_direction_frames = 0
        self.hold_frames = 0
        self.failed = False
        self.failure_reason = ""
        self.success = False
        self.phase = "reset"
        self.last_frame_index: int | None = None
        self.metrics: dict[str, Any] = {
            "phase": self.phase,
            "hold_frames": 0,
            "success": False,
            "failed": False,
        }

    @staticmethod
    def _event_delta_seconds(
        first: int | None,
        second: int | None,
        control_dt: float,
    ) -> float:
        if first is None or second is None:
            return math.inf
        return abs(first - second) * control_dt

    def advance(self, frame_index: int, **reading: float) -> dict[str, Any]:
        frame = int(frame_index)
        if self.last_frame_index is not None:
            if frame < self.last_frame_index:
                raise ValueError("frame_index must be monotonic")
            if frame == self.last_frame_index:
                return dict(self.metrics)
        self.last_frame_index = frame

        metrics = evaluate_dual_knob_reading(target=self.target, **reading)
        x_deg = float(metrics.get("x_deg", math.nan))
        y_deg = float(metrics.get("y_deg", math.nan))
        if math.isfinite(x_deg) and x_deg > DIRECTION_HYSTERESIS_DEG:
            if self.left_start_frame is None:
                self.left_start_frame = frame
        if math.isfinite(y_deg) and y_deg > DIRECTION_HYSTERESIS_DEG:
            if self.right_start_frame is None:
                self.right_start_frame = frame
        if metrics.get("left_target_ok") and self.left_entry_frame is None:
            self.left_entry_frame = frame
        if metrics.get("right_target_ok") and self.right_entry_frame is None:
            self.right_entry_frame = frame

        start_delta = self._event_delta_seconds(
            self.left_start_frame, self.right_start_frame, self.control_dt
        )
        entry_delta = self._event_delta_seconds(
            self.left_entry_frame, self.right_entry_frame, self.control_dt
        )
        start_sync_ok = bool(start_delta <= START_SYNCHRONY_SECONDS)
        entry_sync_ok = bool(entry_delta <= ENTRY_SYNCHRONY_SECONDS)

        if metrics.get("cross_contacts_clear"):
            self.cross_contact_frames = 0
        else:
            self.cross_contact_frames += 1

        if x_deg < -FORBIDDEN_DIRECTION_DEG or y_deg < -FORBIDDEN_DIRECTION_DEG:
            self.forbidden_direction_frames += 1
        else:
            self.forbidden_direction_frames = 0

        if not metrics.get("finite"):
            self.failed = True
            self.failure_reason = "nonfinite_state"
        elif not metrics.get("hard_limit_ok"):
            self.failed = True
            self.failure_reason = "joint_limit_violation"
        elif self.forbidden_direction_frames >= FORBIDDEN_DIRECTION_FAILURE_FRAMES:
            self.failed = True
            self.failure_reason = "forbidden_rotation_direction"
        elif self.cross_contact_frames >= CROSS_CONTACT_FAILURE_FRAMES:
            self.failed = True
            self.failure_reason = "sustained_cross_hand_contact"

        eligible = bool(
            metrics.get("instantaneous_hold_ok")
            and start_sync_ok
            and entry_sync_ok
            and not self.failed
        )
        self.hold_frames = self.hold_frames + 1 if eligible else 0
        self.success = bool(self.hold_frames >= HOLD_FRAMES and not self.failed)
        if self.failed:
            self.phase = "failed"
        elif self.success:
            self.phase = "success"
        elif self.hold_frames:
            self.phase = "hold"
        elif metrics.get("left_target_ok") or metrics.get("right_target_ok"):
            self.phase = "acquire"
        elif self.left_start_frame is not None or self.right_start_frame is not None:
            self.phase = "rotate"
        elif metrics.get("assigned_contacts_ok"):
            self.phase = "engage"
        else:
            self.phase = "reset"

        metrics.update(
            {
                "target_family": self.target.family,
                "target_label": self.target.label,
                "x_target_deg": float(self.target.x_target_deg),
                "y_target_deg": float(self.target.y_target_deg),
                "left_start_frame": self.left_start_frame,
                "right_start_frame": self.right_start_frame,
                "left_entry_frame": self.left_entry_frame,
                "right_entry_frame": self.right_entry_frame,
                "start_delta_s": float(start_delta),
                "entry_delta_s": float(entry_delta),
                "start_sync_ok": bool(start_sync_ok),
                "entry_sync_ok": bool(entry_sync_ok),
                "cross_contact_frames": int(self.cross_contact_frames),
                "forbidden_direction_frames": int(
                    self.forbidden_direction_frames
                ),
                "hold_frames": int(self.hold_frames),
                "hold_frames_required": int(HOLD_FRAMES),
                "phase": self.phase,
                "success": bool(self.success),
                "failed": bool(self.failed),
                "failure_reason": self.failure_reason,
            }
        )
        self.metrics = metrics
        return dict(metrics)


__all__ = [
    "CONTACT_THRESHOLD_N",
    "CROSS_CONTACT_FAILURE_FRAMES",
    "DUAL_KNOB_TARGETS",
    "DualKnobEpisodeTracker",
    "DualKnobTarget",
    "ENTRY_SYNCHRONY_SECONDS",
    "FORBIDDEN_DIRECTION_FAILURE_FRAMES",
    "HOLD_FRAMES",
    "LINEAR_RELATION_TOLERANCE_DEG",
    "START_SYNCHRONY_SECONDS",
    "TARGET_TOLERANCE_DEG",
    "VELOCITY_TOLERANCE_DEG_S",
    "evaluate_dual_knob_reading",
    "ratio_total_target",
    "relation_residuals",
    "sum_difference_target",
]
