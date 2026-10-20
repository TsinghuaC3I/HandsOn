"""Physical event semantics for YCB03, independent of Isaac APIs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math

MIN_LEFT_HOLD_SECONDS = 0.20
REQUIRED_FORWARD_TRAVEL_RAD = 6.0 * math.pi
MAX_SUCCESS_REVERSE_RAD = math.pi / 2.0
MAX_OPERATION_TRANSLATION_M = 0.05
MAX_OPERATION_LIFT_M = 0.04
MAX_OPERATION_TILT_DEG = 25.0
MAX_FINAL_TRANSLATION_M = 0.03
MAX_FINAL_YAW_DEG = 15.0
MAX_FINAL_PROPELLER_SPEED = 1.0
MAX_FINAL_LINEAR_SPEED = 0.03
MAX_FINAL_ANGULAR_SPEED = 0.30
MAX_FINAL_FOURTH_WHEEL_GAP_M = 0.005
MAX_PHYSICAL_PROPELLER_SPEED = 120.0

TERMINAL_REASON_CODES = {
    "": 0,
    "nonfinite_airplane_or_joint_state": 1,
    "nonphysical_propeller_overspeed": 2,
    "airplane_fell_off_table_or_workspace": 3,
    "airplane_displacement_exceeded": 4,
    "airplane_lift_exceeded": 5,
    "airplane_tilt_exceeded": 6,
    "propeller_reverse_full_turn": 7,
    "left_hand_drove_propeller_flick": 8,
    "invalid_airplane_contact_state": 9,
}


@dataclass(frozen=True)
class YCB03PhysicsSample:
    """One 120 Hz physical-truth sample consumed by the event state machine."""

    propeller_angle_rad: float
    propeller_speed_rad_s: float
    left_fuselage_contact: bool
    right_propeller_contact: bool
    left_propeller_contact: bool
    left_any_airplane_contact: bool
    right_any_airplane_contact: bool
    translation_m: float
    lift_m: float
    roll_deg: float
    pitch_deg: float
    yaw_delta_deg: float
    linear_speed_m_s: float
    angular_speed_rad_s: float
    wheel_support_count: int
    maximum_wheel_gap_m: float
    airplane_off_table_or_workspace: bool = False
    finite: bool = True
    contacts_valid: bool = True


@dataclass
class YCB03EventState:
    """Resettable angle history with reserved legacy slots; never writes physics."""

    previous_propeller_angle_rad: float = 0.0
    unwrapped_signed_travel_rad: float = 0.0
    direction: int = 0
    forward_travel_rad: float = 0.0
    reverse_travel_rad: float = 0.0
    hard_reverse_travel_rad: float = 0.0
    valid_flick_count: int = 0
    left_hold_seconds: float = 0.0
    right_contact_previous: bool = False
    right_separation_armed: bool = True
    active_flick: bool = False
    active_flick_valid: bool = False
    active_flick_positive_travel_rad: float = 0.0
    active_flick_negative_travel_rad: float = 0.0
    active_flick_coast_seconds: float = 0.0
    left_propeller_active: bool = False
    left_propeller_travel_rad: float = 0.0
    lift_violation_seconds: float = 0.0
    tilt_violation_seconds: float = 0.0
    first_terminal_reason: str = ""
    last_sample: YCB03PhysicsSample | None = field(default=None, repr=False)

    @staticmethod
    def wrapped_delta(current: float, previous: float) -> float:
        return math.atan2(
            math.sin(float(current) - float(previous)),
            math.cos(float(current) - float(previous)),
        )

    def _operation_valid(self, sample: YCB03PhysicsSample) -> bool:
        return bool(
            sample.finite
            and sample.contacts_valid
            and sample.translation_m <= MAX_OPERATION_TRANSLATION_M
            and sample.lift_m <= MAX_OPERATION_LIFT_M
            and abs(sample.roll_deg) <= MAX_OPERATION_TILT_DEG
            and abs(sample.pitch_deg) <= MAX_OPERATION_TILT_DEG
            and sample.wheel_support_count >= 3
            and sample.maximum_wheel_gap_m <= MAX_FINAL_FOURTH_WHEEL_GAP_M
        )

    def _record_terminal(self, reason: str) -> None:
        if not self.first_terminal_reason:
            self.first_terminal_reason = str(reason)

    @property
    def absolute_travel_rad(self) -> float:
        """Total angular path length, including passive coast in either direction.

        Keep the existing positive/negative travel slots in the recording
        schema. The former flick/timer slots remain reserved and do not gate
        progress. No contact evidence is needed to measure the joint angle.
        """
        return self.forward_travel_rad + self.reverse_travel_rad

    def update(self, sample: YCB03PhysicsSample, dt: float) -> None:
        """Accumulate the actual relative propeller angle once per substep."""
        self.last_sample = sample
        if self.first_terminal_reason:
            return
        if not sample.finite or not all(math.isfinite(value) for value in (
            sample.propeller_angle_rad, sample.propeller_speed_rad_s,
        )):
            self._record_terminal("nonfinite_airplane_or_joint_state")
            return
        if abs(sample.propeller_speed_rad_s) > MAX_PHYSICAL_PROPELLER_SPEED:
            self._record_terminal("nonphysical_propeller_overspeed")
            return
        delta = self.wrapped_delta(
            sample.propeller_angle_rad, self.previous_propeller_angle_rad,
        )
        self.previous_propeller_angle_rad = float(sample.propeller_angle_rad)
        self.unwrapped_signed_travel_rad += delta
        self.forward_travel_rad += max(0.0, delta)
        self.reverse_travel_rad += max(0.0, -delta)

    def final_ready(self) -> bool:
        return bool(
            self.last_sample is not None
            and not self.first_terminal_reason
            and self.absolute_travel_rad >= REQUIRED_FORWARD_TRAVEL_RAD - 1e-10
        )

    def count_blocker(self) -> str:
        if self.first_terminal_reason:
            return "terminal_failure"
        if self.last_sample is None:
            return "waiting_for_joint_state"
        return "complete" if self.final_ready() else "rotate_propeller"

    def metrics(self) -> dict[str, float | int | bool | str]:
        sample = self.last_sample
        return {
            "ycb03_measured_net_turns": self.unwrapped_signed_travel_rad / (2.0 * math.pi),
            "ycb03_absolute_travel_rad": self.absolute_travel_rad,
            "ycb03_absolute_angle_deg": math.degrees(self.absolute_travel_rad),
            "ycb03_absolute_turns": self.absolute_travel_rad / (2.0 * math.pi),
            "ycb03_completed_turns": math.floor((self.absolute_travel_rad + 1e-10) / (2.0 * math.pi)),
            "ycb03_pending_forward_turns": 0.0,
            "ycb03_count_blocker": self.count_blocker(),
            "ycb03_right_fingertip_contact": bool(sample and sample.right_propeller_contact),
            "ycb03_wheel_support_count": sample.wheel_support_count if sample else 0,
            "ycb03_maximum_wheel_gap_m": sample.maximum_wheel_gap_m if sample else 0.0,
            "ycb03_direction": int(self.direction),
            "ycb03_forward_travel_rad": float(self.forward_travel_rad),
            "ycb03_forward_turns": float(self.forward_travel_rad / (2.0 * math.pi)),
            "ycb03_reverse_travel_rad": float(self.reverse_travel_rad),
            "ycb03_valid_flick_count": int(self.valid_flick_count),
            "ycb03_left_hold_seconds": float(self.left_hold_seconds),
            "ycb03_active_flick": bool(self.active_flick),
            "ycb03_left_hold_ready": bool(
                self.left_hold_seconds >= MIN_LEFT_HOLD_SECONDS
            ),
            "ycb03_operation_valid": bool(
                sample is not None and self._operation_valid(sample)
            ),
            "ycb03_hands_released": bool(
                sample is not None
                and sample.contacts_valid
                and not sample.left_any_airplane_contact
                and not sample.right_any_airplane_contact
            ),
            "ycb03_contacts_valid": bool(sample is not None and sample.contacts_valid),
            "ycb03_final_ready": bool(self.final_ready()),
            "ycb03_terminal_reason": self.first_terminal_reason,
            "ycb03_terminal_reason_code": int(
                TERMINAL_REASON_CODES.get(self.first_terminal_reason, -1)
            ),
        }


# Stable V1 order follows the declared dataclass fields; tests bind every field.
_YCB03_EVENT_FIELDS = tuple(f for f in fields(YCB03EventState) if f.name != "last_sample")
_YCB03_SAMPLE_FIELDS = tuple(fields(YCB03PhysicsSample))
YCB03_HISTORY_SIZE = len(_YCB03_EVENT_FIELDS) + 1 + len(_YCB03_SAMPLE_FIELDS)


def event_history_vector(state: YCB03EventState) -> list[float]:
    values = [float(TERMINAL_REASON_CODES[state.first_terminal_reason])
              if f.name == "first_terminal_reason" else float(getattr(state, f.name))
              for f in _YCB03_EVENT_FIELDS]
    values.append(float(state.last_sample is not None))
    values.extend(float(getattr(state.last_sample, f.name)) if state.last_sample is not None else 0.0
                  for f in _YCB03_SAMPLE_FIELDS)
    return values


def event_state_from_history(values) -> YCB03EventState:
    values = list(values)
    if len(values) != YCB03_HISTORY_SIZE or not all(math.isfinite(float(x)) for x in values):
        raise ValueError("Invalid YCB03 recorded event history")

    def decode(f, value):
        value = float(value)
        if f.name == "first_terminal_reason":
            inverse = {code: reason for reason, code in TERMINAL_REASON_CODES.items()}
            if value not in inverse:
                raise ValueError("Invalid YCB03 recorded terminal reason")
            return inverse[value]
        if f.type in ("bool", bool):
            if value not in (0.0, 1.0):
                raise ValueError("Invalid YCB03 recorded boolean")
            return bool(value)
        if f.type in ("int", int):
            if value != int(value):
                raise ValueError("Invalid YCB03 recorded integer")
            return int(value)
        return value

    offset = len(_YCB03_EVENT_FIELDS)
    present = values[offset]
    if present not in (0.0, 1.0):
        raise ValueError("Invalid YCB03 recorded sample presence")
    state = YCB03EventState(**{f.name: decode(f, value)
        for f, value in zip(_YCB03_EVENT_FIELDS, values[:offset])})
    if present:
        state.last_sample = YCB03PhysicsSample(**{f.name: decode(f, value)
            for f, value in zip(_YCB03_SAMPLE_FIELDS, values[offset + 1:])})
    return state
