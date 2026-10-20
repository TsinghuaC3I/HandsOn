"""Asset-independent physical contract for B320-BC017.

The live B320 environment supplies one :class:`BC017Frame` for every 120 Hz
PhysX substep.  This module intentionally imports neither Isaac nor USD so the
same ordered state machine can be exercised in CPU tests and replay tooling.

The authoritative task brief is preserved verbatim in :data:`TASK_REQUIREMENT`.
A drawer crossing is attributed only from the assigned/wrong-hand contact
truth on the immediately preceding and crossing samples.  Contact elsewhere
in the excursion is deliberately irrelevant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

import numpy as np


TASK_ID = "B320-BC017"
TASK_NAME = "t15_open_drawers"
TASK_TITLE = "B320-BC017 Dual Drawer Synchronized Open"
TASK_REQUIREMENT = (
    "Use the assigned hand on each drawer, pull both beyond 60 percent travel, "
    "and cross the threshold within 1.2 seconds."
)

PHYSICS_HZ = 120
PHYSICS_DT_SECONDS = 1.0 / PHYSICS_HZ
PHYSICS_TIME_TOLERANCE_SECONDS = 1.0e-9
OPEN_THRESHOLD = 0.60
SYNC_WINDOW_SECONDS = 1.2
SYNC_WINDOW_STEPS = 144
STABILITY_STEPS = 2

# Fixed recording order.  LL/RR are the assigned channels.  RL is the wrong
# hand on the left drawer and LR is the wrong hand on the right drawer.
CONTACT_KEYS = ("LL", "LR", "RL", "RR")
ASSIGNED_CONTACT_KEYS = ("LL", "RR")
WRONG_CONTACT_KEYS = ("RL", "LR")

_ZERO_SHA256 = "0" * 64
_HEX_DIGITS = frozenset("0123456789abcdef")


class BC017FailureReason(IntEnum):
    """Stable first-cause codes stored in collection and replay state."""

    NONE = 0
    CALIBRATION_NOT_FORMAL = 1
    PHYSICS_STEP_DISCONTINUITY = 2
    PHYSICS_TIME_DISCONTINUITY = 3
    LEFT_DOF_TRUTH_UNAVAILABLE = 4
    RIGHT_DOF_TRUTH_UNAVAILABLE = 5
    LEFT_DOF_NON_FINITE = 6
    RIGHT_DOF_NON_FINITE = 7
    LL_CONTACT_TRUTH_UNAVAILABLE = 8
    LR_CONTACT_TRUTH_UNAVAILABLE = 9
    RL_CONTACT_TRUTH_UNAVAILABLE = 10
    RR_CONTACT_TRUTH_UNAVAILABLE = 11
    LL_CONTACT_FORCE_INVALID = 12
    LR_CONTACT_FORCE_INVALID = 13
    RL_CONTACT_FORCE_INVALID = 14
    RR_CONTACT_FORCE_INVALID = 15
    FIXTURE_TRUTH_UNAVAILABLE = 16
    FIXTURE_TRUTH_INVALID = 17
    LEFT_TRAVEL_LIMIT_EXCEEDED = 18
    RIGHT_TRAVEL_LIMIT_EXCEEDED = 19
    FIXTURE_TRANSLATION_DRIFT = 20
    FIXTURE_ROTATION_DRIFT = 21
    BC017_LIVE_AUDIT_BINDING_DRIFT = 22


class BC017FailureMetric(IntEnum):
    """Append-only stable codes for every terminal evidence metric."""

    NONE = 0
    STABLE_BEYOND_THRESHOLD_SAMPLES = 1
    CALIBRATION_FORMAL = 2
    PHYSICS_STEP_DELTA = 3
    PHYSICS_TIME_FINITE = 4
    PHYSICS_TIME_DELTA_S = 5
    LEFT_DOF_READABLE = 6
    LEFT_DOF_Q_M_FINITE = 7
    LEFT_DOF_QD_M_S_FINITE = 8
    RIGHT_DOF_READABLE = 9
    RIGHT_DOF_Q_M_FINITE = 10
    RIGHT_DOF_QD_M_S_FINITE = 11
    LL_CONTACT_READABLE = 12
    LL_FORCE_FINITE = 13
    LL_FORCE_N = 14
    LR_CONTACT_READABLE = 15
    LR_FORCE_FINITE = 16
    LR_FORCE_N = 17
    RL_CONTACT_READABLE = 18
    RL_FORCE_FINITE = 19
    RL_FORCE_N = 20
    RR_CONTACT_READABLE = 21
    RR_FORCE_FINITE = 22
    RR_FORCE_N = 23
    FIXTURE_TRUTH_READABLE = 24
    FIXTURE_TRANSLATION_DRIFT_M = 25
    FIXTURE_ROTATION_DRIFT_RAD = 26
    LEFT_DRAWER_PROGRESS = 27
    RIGHT_DRAWER_PROGRESS = 28
    BC017_LIVE_AUDIT_BINDING_VALID = 29

    @property
    def metric_name(self) -> str:
        return _FAILURE_METRIC_NAMES[self]


_FAILURE_METRIC_NAMES: Mapping[BC017FailureMetric, str] = MappingProxyType(
    {
        BC017FailureMetric.NONE: "none",
        BC017FailureMetric.STABLE_BEYOND_THRESHOLD_SAMPLES: (
            "stable_beyond_threshold_samples"
        ),
        BC017FailureMetric.CALIBRATION_FORMAL: "calibration_formal",
        BC017FailureMetric.PHYSICS_STEP_DELTA: "physics_step_delta",
        BC017FailureMetric.PHYSICS_TIME_FINITE: "physics_time_finite",
        BC017FailureMetric.PHYSICS_TIME_DELTA_S: "physics_time_delta_s",
        BC017FailureMetric.LEFT_DOF_READABLE: "left_dof_readable",
        BC017FailureMetric.LEFT_DOF_Q_M_FINITE: "left_dof_q_m_finite",
        BC017FailureMetric.LEFT_DOF_QD_M_S_FINITE: (
            "left_dof_qd_m_s_finite"
        ),
        BC017FailureMetric.RIGHT_DOF_READABLE: "right_dof_readable",
        BC017FailureMetric.RIGHT_DOF_Q_M_FINITE: "right_dof_q_m_finite",
        BC017FailureMetric.RIGHT_DOF_QD_M_S_FINITE: (
            "right_dof_qd_m_s_finite"
        ),
        BC017FailureMetric.LL_CONTACT_READABLE: "ll_contact_readable",
        BC017FailureMetric.LL_FORCE_FINITE: "ll_force_finite",
        BC017FailureMetric.LL_FORCE_N: "ll_force_n",
        BC017FailureMetric.LR_CONTACT_READABLE: "lr_contact_readable",
        BC017FailureMetric.LR_FORCE_FINITE: "lr_force_finite",
        BC017FailureMetric.LR_FORCE_N: "lr_force_n",
        BC017FailureMetric.RL_CONTACT_READABLE: "rl_contact_readable",
        BC017FailureMetric.RL_FORCE_FINITE: "rl_force_finite",
        BC017FailureMetric.RL_FORCE_N: "rl_force_n",
        BC017FailureMetric.RR_CONTACT_READABLE: "rr_contact_readable",
        BC017FailureMetric.RR_FORCE_FINITE: "rr_force_finite",
        BC017FailureMetric.RR_FORCE_N: "rr_force_n",
        BC017FailureMetric.FIXTURE_TRUTH_READABLE: "fixture_truth_readable",
        BC017FailureMetric.FIXTURE_TRANSLATION_DRIFT_M: (
            "fixture_translation_drift_m"
        ),
        BC017FailureMetric.FIXTURE_ROTATION_DRIFT_RAD: (
            "fixture_rotation_drift_rad"
        ),
        BC017FailureMetric.LEFT_DRAWER_PROGRESS: "left_drawer_progress",
        BC017FailureMetric.RIGHT_DRAWER_PROGRESS: "right_drawer_progress",
        BC017FailureMetric.BC017_LIVE_AUDIT_BINDING_VALID: (
            "bc017_live_audit_binding_valid"
        ),
    }
)

_TERMINAL_METRICS_BY_REASON: Mapping[
    BC017FailureReason,
    frozenset[BC017FailureMetric],
] = MappingProxyType(
    {
        BC017FailureReason.NONE: frozenset(
            {BC017FailureMetric.STABLE_BEYOND_THRESHOLD_SAMPLES}
        ),
        BC017FailureReason.CALIBRATION_NOT_FORMAL: frozenset(
            {BC017FailureMetric.CALIBRATION_FORMAL}
        ),
        BC017FailureReason.PHYSICS_STEP_DISCONTINUITY: frozenset(
            {BC017FailureMetric.PHYSICS_STEP_DELTA}
        ),
        BC017FailureReason.PHYSICS_TIME_DISCONTINUITY: frozenset(
            {
                BC017FailureMetric.PHYSICS_TIME_FINITE,
                BC017FailureMetric.PHYSICS_TIME_DELTA_S,
            }
        ),
        BC017FailureReason.LEFT_DOF_TRUTH_UNAVAILABLE: frozenset(
            {BC017FailureMetric.LEFT_DOF_READABLE}
        ),
        BC017FailureReason.RIGHT_DOF_TRUTH_UNAVAILABLE: frozenset(
            {BC017FailureMetric.RIGHT_DOF_READABLE}
        ),
        BC017FailureReason.LEFT_DOF_NON_FINITE: frozenset(
            {
                BC017FailureMetric.LEFT_DOF_Q_M_FINITE,
                BC017FailureMetric.LEFT_DOF_QD_M_S_FINITE,
            }
        ),
        BC017FailureReason.RIGHT_DOF_NON_FINITE: frozenset(
            {
                BC017FailureMetric.RIGHT_DOF_Q_M_FINITE,
                BC017FailureMetric.RIGHT_DOF_QD_M_S_FINITE,
            }
        ),
        BC017FailureReason.LL_CONTACT_TRUTH_UNAVAILABLE: frozenset(
            {BC017FailureMetric.LL_CONTACT_READABLE}
        ),
        BC017FailureReason.LR_CONTACT_TRUTH_UNAVAILABLE: frozenset(
            {BC017FailureMetric.LR_CONTACT_READABLE}
        ),
        BC017FailureReason.RL_CONTACT_TRUTH_UNAVAILABLE: frozenset(
            {BC017FailureMetric.RL_CONTACT_READABLE}
        ),
        BC017FailureReason.RR_CONTACT_TRUTH_UNAVAILABLE: frozenset(
            {BC017FailureMetric.RR_CONTACT_READABLE}
        ),
        BC017FailureReason.LL_CONTACT_FORCE_INVALID: frozenset(
            {
                BC017FailureMetric.LL_FORCE_FINITE,
                BC017FailureMetric.LL_FORCE_N,
            }
        ),
        BC017FailureReason.LR_CONTACT_FORCE_INVALID: frozenset(
            {
                BC017FailureMetric.LR_FORCE_FINITE,
                BC017FailureMetric.LR_FORCE_N,
            }
        ),
        BC017FailureReason.RL_CONTACT_FORCE_INVALID: frozenset(
            {
                BC017FailureMetric.RL_FORCE_FINITE,
                BC017FailureMetric.RL_FORCE_N,
            }
        ),
        BC017FailureReason.RR_CONTACT_FORCE_INVALID: frozenset(
            {
                BC017FailureMetric.RR_FORCE_FINITE,
                BC017FailureMetric.RR_FORCE_N,
            }
        ),
        BC017FailureReason.FIXTURE_TRUTH_UNAVAILABLE: frozenset(
            {BC017FailureMetric.FIXTURE_TRUTH_READABLE}
        ),
        BC017FailureReason.FIXTURE_TRUTH_INVALID: frozenset(
            {
                BC017FailureMetric.FIXTURE_TRANSLATION_DRIFT_M,
                BC017FailureMetric.FIXTURE_ROTATION_DRIFT_RAD,
            }
        ),
        BC017FailureReason.LEFT_TRAVEL_LIMIT_EXCEEDED: frozenset(
            {BC017FailureMetric.LEFT_DRAWER_PROGRESS}
        ),
        BC017FailureReason.RIGHT_TRAVEL_LIMIT_EXCEEDED: frozenset(
            {BC017FailureMetric.RIGHT_DRAWER_PROGRESS}
        ),
        BC017FailureReason.FIXTURE_TRANSLATION_DRIFT: frozenset(
            {BC017FailureMetric.FIXTURE_TRANSLATION_DRIFT_M}
        ),
        BC017FailureReason.FIXTURE_ROTATION_DRIFT: frozenset(
            {BC017FailureMetric.FIXTURE_ROTATION_DRIFT_RAD}
        ),
        BC017FailureReason.BC017_LIVE_AUDIT_BINDING_DRIFT: frozenset(
            {BC017FailureMetric.BC017_LIVE_AUDIT_BINDING_VALID}
        ),
    }
)

if set(_FAILURE_METRIC_NAMES) != set(BC017FailureMetric):
    raise RuntimeError("BC017 failure metric names are not exhaustive")
if set(_TERMINAL_METRICS_BY_REASON) != set(BC017FailureReason):
    raise RuntimeError("BC017 terminal reason-to-metric table is not exhaustive")
if (
    frozenset().union(*_TERMINAL_METRICS_BY_REASON.values())
    != frozenset(BC017FailureMetric) - {BC017FailureMetric.NONE}
):
    raise RuntimeError("BC017 terminal metrics are not exhaustively reachable")


_TRAVEL_FAILURES = frozenset(
    {
        BC017FailureReason.LEFT_TRAVEL_LIMIT_EXCEEDED,
        BC017FailureReason.RIGHT_TRAVEL_LIMIT_EXCEEDED,
    }
)
_FIXTURE_DRIFT_FAILURES = frozenset(
    {
        BC017FailureReason.FIXTURE_TRANSLATION_DRIFT,
        BC017FailureReason.FIXTURE_ROTATION_DRIFT,
    }
)

# Ordered hard-terminal first-cause priority.  The runtime audit guard is
# evaluated before frame truth; within frame truth, this is the authoritative
# tie-break when more than one channel is invalid on one physics sample.
BC017_HARD_FIRST_CAUSE_METRIC_PRIORITY = (
    BC017FailureMetric.BC017_LIVE_AUDIT_BINDING_VALID,
    BC017FailureMetric.CALIBRATION_FORMAL,
    BC017FailureMetric.PHYSICS_STEP_DELTA,
    BC017FailureMetric.PHYSICS_TIME_FINITE,
    BC017FailureMetric.PHYSICS_TIME_DELTA_S,
    BC017FailureMetric.LEFT_DOF_READABLE,
    BC017FailureMetric.LEFT_DOF_Q_M_FINITE,
    BC017FailureMetric.LEFT_DOF_QD_M_S_FINITE,
    BC017FailureMetric.RIGHT_DOF_READABLE,
    BC017FailureMetric.RIGHT_DOF_Q_M_FINITE,
    BC017FailureMetric.RIGHT_DOF_QD_M_S_FINITE,
    BC017FailureMetric.LL_CONTACT_READABLE,
    BC017FailureMetric.LL_FORCE_FINITE,
    BC017FailureMetric.LL_FORCE_N,
    BC017FailureMetric.LR_CONTACT_READABLE,
    BC017FailureMetric.LR_FORCE_FINITE,
    BC017FailureMetric.LR_FORCE_N,
    BC017FailureMetric.RL_CONTACT_READABLE,
    BC017FailureMetric.RL_FORCE_FINITE,
    BC017FailureMetric.RL_FORCE_N,
    BC017FailureMetric.RR_CONTACT_READABLE,
    BC017FailureMetric.RR_FORCE_FINITE,
    BC017FailureMetric.RR_FORCE_N,
    BC017FailureMetric.FIXTURE_TRUTH_READABLE,
    BC017FailureMetric.FIXTURE_TRANSLATION_DRIFT_M,
    BC017FailureMetric.FIXTURE_ROTATION_DRIFT_RAD,
)

if set(BC017_HARD_FIRST_CAUSE_METRIC_PRIORITY) != (
    set(BC017FailureMetric)
    - {
        BC017FailureMetric.NONE,
        BC017FailureMetric.STABLE_BEYOND_THRESHOLD_SAMPLES,
        BC017FailureMetric.LEFT_DRAWER_PROGRESS,
        BC017FailureMetric.RIGHT_DRAWER_PROGRESS,
    }
):
    raise RuntimeError("BC017 hard first-cause priority is not exhaustive")
if len(set(BC017_HARD_FIRST_CAUSE_METRIC_PRIORITY)) != len(
    BC017_HARD_FIRST_CAUSE_METRIC_PRIORITY
):
    raise RuntimeError("BC017 hard first-cause priority contains duplicates")

BC017_DEBOUNCED_FIRST_CAUSE_REASON_PRIORITY = (
    BC017FailureReason.LEFT_TRAVEL_LIMIT_EXCEEDED,
    BC017FailureReason.RIGHT_TRAVEL_LIMIT_EXCEEDED,
    BC017FailureReason.FIXTURE_TRANSLATION_DRIFT,
    BC017FailureReason.FIXTURE_ROTATION_DRIFT,
)
if set(BC017_DEBOUNCED_FIRST_CAUSE_REASON_PRIORITY) != (
    set(_TRAVEL_FAILURES) | set(_FIXTURE_DRIFT_FAILURES)
):
    raise RuntimeError("BC017 debounced first-cause priority is not exhaustive")


def _strict_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not {type(value).__name__}")
    return float(value)


def _strict_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _strict_positive_int(value: object, name: str) -> int:
    result = _strict_nonnegative_int(value, name)
    if result < 1:
        raise ValueError(f"{name} must be at least one")
    return result


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool, not {type(value).__name__}")
    return bool(value)


def _finite_real(value: object, name: str) -> float:
    result = _strict_real(value, name)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalize_sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a hexadecimal SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in _HEX_DIGITS for char in normalized):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal characters")
    return normalized


def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_or_zero(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


@dataclass(frozen=True, slots=True)
class DrawerTravelCalibration:
    """Direction-independent closed/open truth for one prismatic drawer."""

    closed_q_m: float
    open_q_m: float
    formal: bool = False
    source_digest_sha256: str = _ZERO_SHA256

    MIN_SPAN_M = 1.0e-6

    def __post_init__(self) -> None:
        closed = _finite_real(self.closed_q_m, "closed_q_m")
        opened = _finite_real(self.open_q_m, "open_q_m")
        formal = _strict_bool(self.formal, "formal")
        digest = _normalize_sha256(
            self.source_digest_sha256,
            "source_digest_sha256",
        )
        span = opened - closed
        if not math.isfinite(span) or abs(span) < self.MIN_SPAN_M:
            raise ValueError(
                "drawer travel span must be finite and at least "
                f"{self.MIN_SPAN_M:g} m"
            )
        if formal and digest == _ZERO_SHA256:
            raise ValueError("formal drawer travel requires non-provisional evidence")
        object.__setattr__(self, "closed_q_m", closed)
        object.__setattr__(self, "open_q_m", opened)
        object.__setattr__(self, "formal", formal)
        object.__setattr__(self, "source_digest_sha256", digest)

    @property
    def span_m(self) -> float:
        return self.open_q_m - self.closed_q_m

    def normalize(self, q_m: float) -> float:
        return (float(q_m) - self.closed_q_m) / self.span_m

    def threshold_q_m(self, fraction: float = OPEN_THRESHOLD) -> float:
        fraction_value = _finite_real(fraction, "fraction")
        return self.closed_q_m + fraction_value * self.span_m

    def is_strictly_beyond(
        self,
        q_m: float,
        fraction: float = OPEN_THRESHOLD,
    ) -> bool:
        """Use raw q comparison so exactly 60 percent never qualifies."""

        position = float(q_m)
        threshold = self.threshold_q_m(fraction)
        return position > threshold if self.span_m > 0.0 else position < threshold

    def as_dict(self) -> dict[str, object]:
        return {
            "closed_q_m": self.closed_q_m,
            "open_q_m": self.open_q_m,
            "formal": self.formal,
            "source_digest_sha256": self.source_digest_sha256,
        }

    @property
    def contract_digest_sha256(self) -> str:
        return _canonical_digest(self.as_dict())


@dataclass(frozen=True, slots=True)
class BC017Calibration:
    """Formal travel, contact, drift, and debounce calibration contract."""

    left_travel: DrawerTravelCalibration
    right_travel: DrawerTravelCalibration
    contact_on_n: float = 0.15
    contact_off_n: float = 0.05
    travel_tolerance_fraction: float = 0.05
    fixture_translation_limit_m: float = 0.003
    fixture_rotation_limit_rad: float = math.radians(1.0)
    truth_failure_debounce_steps: int = 1
    travel_limit_debounce_steps: int = 2
    fixture_drift_debounce_steps: int = 2
    formal: bool = False
    source_digest_sha256: str = _ZERO_SHA256

    def __post_init__(self) -> None:
        if not isinstance(self.left_travel, DrawerTravelCalibration):
            raise TypeError("left_travel must be DrawerTravelCalibration")
        if not isinstance(self.right_travel, DrawerTravelCalibration):
            raise TypeError("right_travel must be DrawerTravelCalibration")
        contact_on = _finite_real(self.contact_on_n, "contact_on_n")
        contact_off = _finite_real(self.contact_off_n, "contact_off_n")
        tolerance = _finite_real(
            self.travel_tolerance_fraction,
            "travel_tolerance_fraction",
        )
        translation_limit = _finite_real(
            self.fixture_translation_limit_m,
            "fixture_translation_limit_m",
        )
        rotation_limit = _finite_real(
            self.fixture_rotation_limit_rad,
            "fixture_rotation_limit_rad",
        )
        truth_debounce = _strict_positive_int(
            self.truth_failure_debounce_steps,
            "truth_failure_debounce_steps",
        )
        travel_debounce = _strict_positive_int(
            self.travel_limit_debounce_steps,
            "travel_limit_debounce_steps",
        )
        fixture_debounce = _strict_positive_int(
            self.fixture_drift_debounce_steps,
            "fixture_drift_debounce_steps",
        )
        formal = _strict_bool(self.formal, "formal")
        digest = _normalize_sha256(
            self.source_digest_sha256,
            "source_digest_sha256",
        )
        if not 0.0 <= contact_off < contact_on:
            raise ValueError("contact thresholds require 0 <= off < on")
        if not 0.0 < tolerance < 0.5:
            raise ValueError("travel_tolerance_fraction must be in (0, 0.5)")
        if translation_limit <= 0.0 or rotation_limit <= 0.0:
            raise ValueError("fixture drift limits must be positive")
        if truth_debounce != 1:
            raise ValueError("missing or malformed physics truth must fail in one sample")
        if formal and (
            not self.left_travel.formal
            or not self.right_travel.formal
            or digest == _ZERO_SHA256
        ):
            raise ValueError(
                "formal BC017 calibration requires formal drawer travel and evidence"
            )
        object.__setattr__(self, "contact_on_n", contact_on)
        object.__setattr__(self, "contact_off_n", contact_off)
        object.__setattr__(self, "travel_tolerance_fraction", tolerance)
        object.__setattr__(self, "fixture_translation_limit_m", translation_limit)
        object.__setattr__(self, "fixture_rotation_limit_rad", rotation_limit)
        object.__setattr__(self, "truth_failure_debounce_steps", truth_debounce)
        object.__setattr__(self, "travel_limit_debounce_steps", travel_debounce)
        object.__setattr__(self, "fixture_drift_debounce_steps", fixture_debounce)
        object.__setattr__(self, "formal", formal)
        object.__setattr__(self, "source_digest_sha256", digest)

    def debounce_steps_for(self, reason: BC017FailureReason) -> int:
        if reason in _TRAVEL_FAILURES:
            return self.travel_limit_debounce_steps
        if reason in _FIXTURE_DRIFT_FAILURES:
            return self.fixture_drift_debounce_steps
        return self.truth_failure_debounce_steps

    @property
    def declared_failure_debounces(self) -> Mapping[BC017FailureReason, int]:
        return MappingProxyType(
            {
                reason: self.debounce_steps_for(reason)
                for reason in BC017FailureReason
                if reason is not BC017FailureReason.NONE
            }
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "left_travel": self.left_travel.as_dict(),
            "right_travel": self.right_travel.as_dict(),
            "contact_on_n": self.contact_on_n,
            "contact_off_n": self.contact_off_n,
            "travel_tolerance_fraction": self.travel_tolerance_fraction,
            "fixture_translation_limit_m": self.fixture_translation_limit_m,
            "fixture_rotation_limit_rad": self.fixture_rotation_limit_rad,
            "truth_failure_debounce_steps": self.truth_failure_debounce_steps,
            "travel_limit_debounce_steps": self.travel_limit_debounce_steps,
            "fixture_drift_debounce_steps": self.fixture_drift_debounce_steps,
            "formal": self.formal,
            "source_digest_sha256": self.source_digest_sha256,
        }

    @property
    def contract_digest_sha256(self) -> str:
        return _canonical_digest(self.as_dict())


@dataclass(frozen=True, slots=True)
class BC017Frame:
    """One strictly typed 120 Hz physical-truth sample.

    Non-finite real scalars are intentionally accepted by the constructor so
    the state machine can attribute them to the exact truth channel and record
    immutable terminal evidence.  Non-numeric scalars and non-bool readability
    values are rejected at the boundary.
    """

    physics_step: int
    physics_time_s: float
    left_q_m: float
    right_q_m: float
    left_qd_m_s: float
    right_qd_m_s: float
    left_dof_readable: bool
    right_dof_readable: bool
    ll_force_n: float
    lr_force_n: float
    rl_force_n: float
    rr_force_n: float
    ll_readable: bool
    lr_readable: bool
    rl_readable: bool
    rr_readable: bool
    fixture_translation_drift_m: float
    fixture_rotation_drift_rad: float
    fixture_truth_readable: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "physics_step",
            _strict_nonnegative_int(self.physics_step, "physics_step"),
        )
        for name in (
            "physics_time_s",
            "left_q_m",
            "right_q_m",
            "left_qd_m_s",
            "right_qd_m_s",
            "ll_force_n",
            "lr_force_n",
            "rl_force_n",
            "rr_force_n",
            "fixture_translation_drift_m",
            "fixture_rotation_drift_rad",
        ):
            object.__setattr__(self, name, _strict_real(getattr(self, name), name))
        for name in (
            "left_dof_readable",
            "right_dof_readable",
            "ll_readable",
            "lr_readable",
            "rl_readable",
            "rr_readable",
            "fixture_truth_readable",
        ):
            object.__setattr__(self, name, _strict_bool(getattr(self, name), name))

    @property
    def q_m(self) -> tuple[float, float]:
        return (self.left_q_m, self.right_q_m)

    @property
    def qd_m_s(self) -> tuple[float, float]:
        return (self.left_qd_m_s, self.right_qd_m_s)

    @property
    def contact_force_n(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                "LL": self.ll_force_n,
                "LR": self.lr_force_n,
                "RL": self.rl_force_n,
                "RR": self.rr_force_n,
            }
        )

    @property
    def contact_readable(self) -> Mapping[str, bool]:
        return MappingProxyType(
            {
                "LL": self.ll_readable,
                "LR": self.lr_readable,
                "RL": self.rl_readable,
                "RR": self.rr_readable,
            }
        )


@dataclass(frozen=True, slots=True)
class BC017TerminalEvidence:
    """Immutable first terminal outcome and measured comparison."""

    success: bool
    failure_reason: BC017FailureReason
    metric: str
    metric_code: BC017FailureMetric
    observed: float
    limit: float
    physics_step: int
    physics_time_s: float
    debounce_steps: int

    def __post_init__(self) -> None:
        success = _strict_bool(self.success, "success")
        if not isinstance(self.failure_reason, BC017FailureReason):
            raise TypeError("failure_reason must be BC017FailureReason")
        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError("metric must be a non-empty string")
        if not isinstance(self.metric_code, BC017FailureMetric):
            raise TypeError("metric_code must be BC017FailureMetric")
        if self.metric_code is BC017FailureMetric.NONE:
            raise ValueError("terminal evidence requires a non-NONE metric code")
        if self.metric != self.metric_code.metric_name:
            raise ValueError(
                "terminal metric string/code mismatch: "
                f"{self.metric!r} != {self.metric_code.metric_name!r}"
            )
        if self.metric_code not in _TERMINAL_METRICS_BY_REASON[self.failure_reason]:
            raise ValueError(
                "terminal failure reason/metric code mismatch: "
                f"{self.failure_reason.name}/{self.metric_code.name}"
            )
        observed = _finite_real(self.observed, "observed")
        limit = _finite_real(self.limit, "limit")
        step = _strict_nonnegative_int(self.physics_step, "physics_step")
        time_s = _finite_real(self.physics_time_s, "physics_time_s")
        debounce = _strict_positive_int(self.debounce_steps, "debounce_steps")
        if success and self.failure_reason is not BC017FailureReason.NONE:
            raise ValueError("success evidence cannot carry a failure reason")
        if not success and self.failure_reason is BC017FailureReason.NONE:
            raise ValueError("failure evidence requires a failure reason")
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "physics_step", step)
        object.__setattr__(self, "physics_time_s", time_s)
        object.__setattr__(self, "debounce_steps", debounce)

    @property
    def failure(self) -> bool:
        return not self.success

    @property
    def reason_code(self) -> int:
        return int(self.failure_reason)


@dataclass(frozen=True, slots=True)
class BC017Snapshot:
    """Immutable, fixed-cardinality view returned by prime/update/reset."""

    primed: bool
    success: bool
    failure: bool
    terminated: bool
    failure_reason_code: int
    terminal_metric_code: int
    truth_valid: bool
    semantic_valid: bool
    physics_step: int
    physics_time_s: float
    q_m: tuple[float, float]
    qd_m_s: tuple[float, float]
    drawer_progress: tuple[float, float]
    contact_force_n: tuple[float, float, float, float]
    contact_readable: tuple[bool, bool, bool, bool]
    contact_active: tuple[bool, bool, bool, bool]
    beyond_threshold: tuple[bool, bool]
    threshold_event_step: tuple[int, int]
    sync_delta_steps: int
    sync_delta_seconds: float
    stable_steps: int
    fixture_drift: tuple[float, float]
    terminal_evidence: BC017TerminalEvidence | None


PUBLIC_OBSERVATION_GROUPS = (
    (
        "bc017_drawer_state",
        (
            "physics_step",
            "physics_time_s",
            "left_q_m",
            "right_q_m",
            "left_qd_m_s",
            "right_qd_m_s",
            "left_progress",
            "right_progress",
            "left_dof_readable",
            "right_dof_readable",
            "left_beyond_60_percent",
            "right_beyond_60_percent",
        ),
    ),
    (
        "bc017_contact_state",
        (
            "ll_force_n",
            "lr_force_n",
            "rl_force_n",
            "rr_force_n",
            "ll_readable",
            "lr_readable",
            "rl_readable",
            "rr_readable",
            "ll_active",
            "lr_active",
            "rl_active",
            "rr_active",
        ),
    ),
    (
        "bc017_stage_state",
        (
            "fixture_truth_readable",
            "fixture_translation_drift_m",
            "fixture_rotation_drift_rad",
            "left_event_seen",
            "right_event_seen",
            "left_event_step",
            "right_event_step",
            "sync_delta_steps",
            "stable_steps",
            "truth_valid",
            "semantic_valid",
            "calibration_formal",
        ),
    ),
    (
        "bc017_terminal_state",
        (
            "success",
            "failure",
            "terminal_reason_code",
            "terminal_metric_code",
            "terminal_step",
            "terminal_time_s",
            "terminal_observed",
            "terminal_limit",
            "terminal_debounce_steps",
        ),
    ),
)
PUBLIC_OBSERVATION_FIELDS = tuple(
    field
    for _group_key, group_fields in PUBLIC_OBSERVATION_GROUPS
    for field in group_fields
)
PUBLIC_OBSERVATION_SHAPE = (len(PUBLIC_OBSERVATION_FIELDS),)

HIDDEN_OBSERVATION_FIELDS = (
    "primed",
    "last_physics_step",
    "last_physics_time_s",
    "previous_semantics_valid",
    "previous_left_beyond",
    "previous_right_beyond",
    "previous_ll_active",
    "previous_lr_active",
    "previous_rl_active",
    "previous_rr_active",
    "current_ll_active",
    "current_lr_active",
    "current_rl_active",
    "current_rr_active",
    "left_event_step",
    "right_event_step",
    "stable_steps",
    "left_travel_violation_steps",
    "right_travel_violation_steps",
    "fixture_translation_violation_steps",
    "fixture_rotation_violation_steps",
    "truth_valid",
    "semantic_valid",
    "terminal_latched",
    "terminal_success",
    "terminal_reason_code",
    "terminal_metric_code",
    "terminal_step",
    "terminal_time_s",
    "terminal_observed",
    "terminal_limit",
    "terminal_debounce_steps",
    "calibration_formal",
)
HIDDEN_OBSERVATION_SHAPE = (len(HIDDEN_OBSERVATION_FIELDS),)

BC017_OBSERVATION_DTYPE = "float64"
BC017_PROPRIO_FLATTEN_ORDER = tuple(
    key for key, _fields in PUBLIC_OBSERVATION_GROUPS
) + ("bc017_hidden_state",)


def bc017_observation_layout_metadata() -> dict[str, object]:
    """Return the JSON-safe consumer layout used by the observation adapter."""

    return {
        "dtype": BC017_OBSERVATION_DTYPE,
        "public_groups": [
            {
                "key": key,
                "fields": list(fields),
                "shape": [len(fields)],
                "dtype": BC017_OBSERVATION_DTYPE,
            }
            for key, fields in PUBLIC_OBSERVATION_GROUPS
        ],
        "hidden_group": {
            "key": "bc017_hidden_state",
            "fields": list(HIDDEN_OBSERVATION_FIELDS),
            "shape": list(HIDDEN_OBSERVATION_SHAPE),
            "dtype": BC017_OBSERVATION_DTYPE,
        },
        "config_flatten_order": list(BC017_PROPRIO_FLATTEN_ORDER),
        "public_flattened_fields": list(PUBLIC_OBSERVATION_FIELDS),
        "public_flattened_shape": list(PUBLIC_OBSERVATION_SHAPE),
    }

BC017_CONTRACT_SCHEMA_NAME = "handson.b320_bc017.physical_contract"
BC017_CONTRACT_SCHEMA_VERSION = 2
BC017_CONTRACT_SCHEMA_DIGEST_SHA256 = _canonical_digest(
    {
        "schema_name": BC017_CONTRACT_SCHEMA_NAME,
        "schema_version": BC017_CONTRACT_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "observation_layout": bc017_observation_layout_metadata(),
        "failure_reason_codes": {
            reason.name: int(reason) for reason in BC017FailureReason
        },
        "terminal_metric_codes": {
            metric.name: int(metric) for metric in BC017FailureMetric
        },
        "terminal_metric_names": {
            metric.name: metric.metric_name for metric in BC017FailureMetric
        },
        "terminal_metrics_by_reason": {
            reason.name: [
                metric.name
                for metric in sorted(
                    _TERMINAL_METRICS_BY_REASON[reason],
                    key=int,
                )
            ]
            for reason in BC017FailureReason
        },
        "hard_first_cause_metric_priority": [
            metric.name for metric in BC017_HARD_FIRST_CAUSE_METRIC_PRIORITY
        ],
        "debounced_first_cause_reason_priority": [
            reason.name
            for reason in BC017_DEBOUNCED_FIRST_CAUSE_REASON_PRIORITY
        ],
    }
)


@dataclass(frozen=True, slots=True)
class _FailureObservation:
    reason: BC017FailureReason
    metric_code: BC017FailureMetric
    observed: float
    limit: float
    debounce_steps: int

    def __post_init__(self) -> None:
        if self.reason is BC017FailureReason.NONE:
            raise ValueError("failure observation requires a non-NONE reason")
        if self.metric_code not in _TERMINAL_METRICS_BY_REASON[self.reason]:
            raise ValueError(
                "failure observation reason/metric mismatch: "
                f"{self.reason.name}/{self.metric_code.name}"
            )

    @property
    def metric(self) -> str:
        return self.metric_code.metric_name


class BC017State:
    """No-assist synchronized dual-drawer contract state machine."""

    def __init__(self, calibration: BC017Calibration) -> None:
        if not isinstance(calibration, BC017Calibration):
            raise TypeError("calibration must be BC017Calibration")
        self.calibration = calibration
        self.reset()

    def reset(self) -> BC017Snapshot:
        """Start a fresh episode; the next accepted call must be ``prime``."""

        self._primed = False
        self._last_physics_step = -1
        self._last_physics_time_s = 0.0
        self._last_q = (0.0, 0.0)
        self._last_qd = (0.0, 0.0)
        self._last_progress = (0.0, 0.0)
        self._last_force = (0.0, 0.0, 0.0, 0.0)
        self._last_contact_readable = (False, False, False, False)
        self._last_dof_readable = (False, False)
        self._last_fixture_readable = False
        self._last_fixture_drift = (0.0, 0.0)
        self._contact_active = [False, False, False, False]
        self._previous_contact_active = [False, False, False, False]
        self._previous_beyond = [False, False]
        self._last_beyond = (False, False)
        self._previous_semantics_valid = False
        self._truth_valid = False
        self._semantic_valid = False
        self._event_step = [-1, -1]
        self._stable_steps = 0
        self._pending_failure_steps = {
            reason: 0 for reason in (*_TRAVEL_FAILURES, *_FIXTURE_DRIFT_FAILURES)
        }
        self._terminal_evidence: BC017TerminalEvidence | None = None
        return self.snapshot()

    @property
    def success(self) -> bool:
        return bool(
            self._terminal_evidence is not None and self._terminal_evidence.success
        )

    @property
    def failure(self) -> bool:
        return bool(
            self._terminal_evidence is not None and self._terminal_evidence.failure
        )

    @property
    def terminated(self) -> bool:
        return self._terminal_evidence is not None

    @property
    def terminal_evidence(self) -> BC017TerminalEvidence | None:
        return self._terminal_evidence

    @property
    def threshold_event_steps(self) -> tuple[int, int]:
        return tuple(self._event_step)

    @property
    def stable_steps(self) -> int:
        return self._stable_steps

    @property
    def contact_active(self) -> Mapping[str, bool]:
        return MappingProxyType(
            dict(zip(CONTACT_KEYS, map(bool, self._contact_active), strict=True))
        )

    @property
    def calibration_digest_sha256(self) -> str:
        return self.calibration.contract_digest_sha256

    @property
    def failure_debounce_steps(self) -> Mapping[BC017FailureReason, int]:
        return self.calibration.declared_failure_debounces

    def prime(self, frame: BC017Frame) -> BC017Snapshot:
        """Publish action-0 origin truth without advancing task semantics."""

        if not isinstance(frame, BC017Frame):
            raise TypeError("frame must be BC017Frame")
        if self._primed:
            raise RuntimeError("BC017 origin truth can prime only a fresh reset state")
        self._primed = True
        self._ingest_frame(frame)

        failure = self._hard_truth_failure(frame, require_continuity=False)
        if failure is not None:
            self._truth_valid = False
            self._semantic_valid = False
            self._latch_failure(failure, frame)
            return self.snapshot()

        self._truth_valid = True
        self._contact_active[:] = self._next_contact_active(frame)
        progress, beyond = self._progress_and_beyond(frame)
        self._last_progress = progress
        self._last_beyond = beyond
        soft_failure, soft_valid = self._update_debounced_guards(frame, progress)
        if soft_failure is not None:
            self._semantic_valid = False
            self._latch_failure(soft_failure, frame)
            return self.snapshot()

        self._semantic_valid = soft_valid
        self._previous_semantics_valid = soft_valid
        self._previous_beyond[:] = beyond
        self._previous_contact_active[:] = self._contact_active
        if not soft_valid:
            self._invalidate_attempt_progress()
        return self.snapshot()

    def update(self, frame: BC017Frame) -> BC017Snapshot:
        """Consume exactly one continuous 120 Hz physics substep."""

        if not isinstance(frame, BC017Frame):
            raise TypeError("frame must be BC017Frame")
        if self.terminated:
            # The first terminal sample freezes every owned state variable.
            return self.snapshot()
        if not self._primed:
            raise RuntimeError("BC017State.update requires explicit prime after reset")

        failure = self._hard_truth_failure(frame, require_continuity=True)
        self._ingest_frame(frame)
        if failure is not None:
            self._truth_valid = False
            self._semantic_valid = False
            self._stable_steps = 0
            self._latch_failure(failure, frame)
            return self.snapshot()

        self._truth_valid = True
        next_active = self._next_contact_active(frame)
        progress, beyond = self._progress_and_beyond(frame)
        self._last_progress = progress
        self._last_beyond = beyond
        self._contact_active[:] = next_active

        soft_failure, soft_valid = self._update_debounced_guards(frame, progress)
        if soft_failure is not None:
            self._semantic_valid = False
            self._stable_steps = 0
            self._latch_failure(soft_failure, frame)
            return self.snapshot()
        if not soft_valid:
            self._semantic_valid = False
            self._invalidate_attempt_progress()
            self._previous_semantics_valid = False
            self._previous_beyond[:] = beyond
            self._previous_contact_active[:] = next_active
            return self.snapshot()

        prior_valid = self._previous_semantics_valid
        prior_beyond = tuple(self._previous_beyond)
        prior_active = tuple(self._previous_contact_active)

        assigned_indices = (0, 3)  # LL, RR
        wrong_indices = (2, 1)  # RL, LR
        for drawer_index, (assigned_index, wrong_index) in enumerate(
            zip(assigned_indices, wrong_indices, strict=True)
        ):
            if not beyond[drawer_index]:
                # A rollback is the only retry/rearm operation required by the
                # literal task brief; it invalidates only that drawer's event.
                self._event_step[drawer_index] = -1
                continue
            crossed = bool(
                prior_valid
                and not prior_beyond[drawer_index]
                and beyond[drawer_index]
            )
            if not crossed:
                continue
            preceding_contact_ok = bool(
                prior_active[assigned_index] and not prior_active[wrong_index]
            )
            crossing_contact_ok = bool(
                next_active[assigned_index] and not next_active[wrong_index]
            )
            self._event_step[drawer_index] = (
                frame.physics_step
                if preceding_contact_ok and crossing_contact_ok
                else -1
            )

        synchronized = bool(
            self._event_step[0] >= 0
            and self._event_step[1] >= 0
            and abs(self._event_step[0] - self._event_step[1])
            <= SYNC_WINDOW_STEPS
        )
        stable_now = bool(synchronized and all(beyond))
        self._stable_steps = self._stable_steps + 1 if stable_now else 0
        self._semantic_valid = True
        self._previous_semantics_valid = True
        self._previous_beyond[:] = beyond
        self._previous_contact_active[:] = next_active

        if self._stable_steps >= STABILITY_STEPS:
            self._terminal_evidence = BC017TerminalEvidence(
                success=True,
                failure_reason=BC017FailureReason.NONE,
                metric="stable_beyond_threshold_samples",
                metric_code=(
                    BC017FailureMetric.STABLE_BEYOND_THRESHOLD_SAMPLES
                ),
                observed=float(self._stable_steps),
                limit=float(STABILITY_STEPS),
                physics_step=frame.physics_step,
                physics_time_s=frame.physics_time_s,
                debounce_steps=STABILITY_STEPS,
            )
        return self.snapshot()

    def latch_live_audit_binding_drift(
        self,
        *,
        physics_step: int,
        physics_time_s: float,
    ) -> BC017Snapshot:
        """Freeze a typed terminal when the accepted live-audit proof drifts."""

        if self.terminated:
            return self.snapshot()
        if not self._primed:
            raise RuntimeError("BC017 live-audit drift requires an accepted origin")
        step = _strict_nonnegative_int(physics_step, "physics_step")
        time_s = _finite_real(physics_time_s, "physics_time_s")
        if step <= self._last_physics_step:
            raise ValueError("BC017 live-audit terminal step must advance")
        self._last_physics_step = step
        self._last_physics_time_s = time_s
        self._truth_valid = False
        self._semantic_valid = False
        self._stable_steps = 0
        self._latch_failure_at(
            _FailureObservation(
                BC017FailureReason.BC017_LIVE_AUDIT_BINDING_DRIFT,
                BC017FailureMetric.BC017_LIVE_AUDIT_BINDING_VALID,
                0.0,
                1.0,
                self.calibration.truth_failure_debounce_steps,
            ),
            physics_step=step,
            physics_time_s=time_s,
        )
        return self.snapshot()

    def _ingest_frame(self, frame: BC017Frame) -> None:
        self._last_physics_step = frame.physics_step
        self._last_physics_time_s = _finite_or_zero(frame.physics_time_s)
        self._last_q = tuple(_finite_or_zero(value) for value in frame.q_m)
        self._last_qd = tuple(_finite_or_zero(value) for value in frame.qd_m_s)
        self._last_force = tuple(
            _finite_or_zero(frame.contact_force_n[key]) for key in CONTACT_KEYS
        )
        self._last_contact_readable = tuple(
            frame.contact_readable[key] for key in CONTACT_KEYS
        )
        self._last_dof_readable = (
            frame.left_dof_readable,
            frame.right_dof_readable,
        )
        self._last_fixture_readable = frame.fixture_truth_readable
        self._last_fixture_drift = (
            _finite_or_zero(frame.fixture_translation_drift_m),
            _finite_or_zero(frame.fixture_rotation_drift_rad),
        )

    def _hard_truth_failure(
        self,
        frame: BC017Frame,
        *,
        require_continuity: bool,
    ) -> _FailureObservation | None:
        one = self.calibration.truth_failure_debounce_steps
        if require_continuity:
            step_delta = frame.physics_step - self._last_physics_step
            if step_delta != 1:
                return _FailureObservation(
                    BC017FailureReason.PHYSICS_STEP_DISCONTINUITY,
                    BC017FailureMetric.PHYSICS_STEP_DELTA,
                    float(step_delta),
                    1.0,
                    one,
                )
            if not math.isfinite(frame.physics_time_s):
                return _FailureObservation(
                    BC017FailureReason.PHYSICS_TIME_DISCONTINUITY,
                    BC017FailureMetric.PHYSICS_TIME_FINITE,
                    0.0,
                    1.0,
                    one,
                )
            time_delta = frame.physics_time_s - self._last_physics_time_s
            if (
                not math.isfinite(time_delta)
                or abs(time_delta - PHYSICS_DT_SECONDS)
                > PHYSICS_TIME_TOLERANCE_SECONDS
            ):
                return _FailureObservation(
                    BC017FailureReason.PHYSICS_TIME_DISCONTINUITY,
                    BC017FailureMetric.PHYSICS_TIME_DELTA_S,
                    _finite_or_zero(time_delta),
                    PHYSICS_DT_SECONDS,
                    one,
                )
        elif not math.isfinite(frame.physics_time_s):
            return _FailureObservation(
                BC017FailureReason.PHYSICS_TIME_DISCONTINUITY,
                BC017FailureMetric.PHYSICS_TIME_FINITE,
                0.0,
                1.0,
                one,
            )

        for readable, unavailable_reason, readable_metric, q_metrics, q, qd in (
            (
                frame.left_dof_readable,
                BC017FailureReason.LEFT_DOF_TRUTH_UNAVAILABLE,
                BC017FailureMetric.LEFT_DOF_READABLE,
                (
                    BC017FailureMetric.LEFT_DOF_Q_M_FINITE,
                    BC017FailureMetric.LEFT_DOF_QD_M_S_FINITE,
                ),
                frame.left_q_m,
                frame.left_qd_m_s,
            ),
            (
                frame.right_dof_readable,
                BC017FailureReason.RIGHT_DOF_TRUTH_UNAVAILABLE,
                BC017FailureMetric.RIGHT_DOF_READABLE,
                (
                    BC017FailureMetric.RIGHT_DOF_Q_M_FINITE,
                    BC017FailureMetric.RIGHT_DOF_QD_M_S_FINITE,
                ),
                frame.right_q_m,
                frame.right_qd_m_s,
            ),
        ):
            if not readable:
                return _FailureObservation(
                    unavailable_reason,
                    readable_metric,
                    0.0,
                    1.0,
                    one,
                )
            if not math.isfinite(q) or not math.isfinite(qd):
                nonfinite_reason = (
                    BC017FailureReason.LEFT_DOF_NON_FINITE
                    if unavailable_reason
                    is BC017FailureReason.LEFT_DOF_TRUTH_UNAVAILABLE
                    else BC017FailureReason.RIGHT_DOF_NON_FINITE
                )
                return _FailureObservation(
                    nonfinite_reason,
                    q_metrics[0] if not math.isfinite(q) else q_metrics[1],
                    0.0,
                    1.0,
                    one,
                )

        unavailable_reasons = {
            "LL": BC017FailureReason.LL_CONTACT_TRUTH_UNAVAILABLE,
            "LR": BC017FailureReason.LR_CONTACT_TRUTH_UNAVAILABLE,
            "RL": BC017FailureReason.RL_CONTACT_TRUTH_UNAVAILABLE,
            "RR": BC017FailureReason.RR_CONTACT_TRUTH_UNAVAILABLE,
        }
        invalid_reasons = {
            "LL": BC017FailureReason.LL_CONTACT_FORCE_INVALID,
            "LR": BC017FailureReason.LR_CONTACT_FORCE_INVALID,
            "RL": BC017FailureReason.RL_CONTACT_FORCE_INVALID,
            "RR": BC017FailureReason.RR_CONTACT_FORCE_INVALID,
        }
        readable_metrics = {
            "LL": BC017FailureMetric.LL_CONTACT_READABLE,
            "LR": BC017FailureMetric.LR_CONTACT_READABLE,
            "RL": BC017FailureMetric.RL_CONTACT_READABLE,
            "RR": BC017FailureMetric.RR_CONTACT_READABLE,
        }
        finite_metrics = {
            "LL": BC017FailureMetric.LL_FORCE_FINITE,
            "LR": BC017FailureMetric.LR_FORCE_FINITE,
            "RL": BC017FailureMetric.RL_FORCE_FINITE,
            "RR": BC017FailureMetric.RR_FORCE_FINITE,
        }
        force_metrics = {
            "LL": BC017FailureMetric.LL_FORCE_N,
            "LR": BC017FailureMetric.LR_FORCE_N,
            "RL": BC017FailureMetric.RL_FORCE_N,
            "RR": BC017FailureMetric.RR_FORCE_N,
        }
        for key in CONTACT_KEYS:
            if not frame.contact_readable[key]:
                return _FailureObservation(
                    unavailable_reasons[key],
                    readable_metrics[key],
                    0.0,
                    1.0,
                    one,
                )
            force = frame.contact_force_n[key]
            if not math.isfinite(force):
                return _FailureObservation(
                    invalid_reasons[key],
                    finite_metrics[key],
                    0.0,
                    1.0,
                    one,
                )
            if force < 0.0:
                return _FailureObservation(
                    invalid_reasons[key],
                    force_metrics[key],
                    force,
                    0.0,
                    one,
                )

        if not frame.fixture_truth_readable:
            return _FailureObservation(
                BC017FailureReason.FIXTURE_TRUTH_UNAVAILABLE,
                BC017FailureMetric.FIXTURE_TRUTH_READABLE,
                0.0,
                1.0,
                one,
            )
        fixture_values = (
            (
                BC017FailureMetric.FIXTURE_TRANSLATION_DRIFT_M,
                frame.fixture_translation_drift_m,
            ),
            (
                BC017FailureMetric.FIXTURE_ROTATION_DRIFT_RAD,
                frame.fixture_rotation_drift_rad,
            ),
        )
        for metric_code, value in fixture_values:
            if not math.isfinite(value) or value < 0.0:
                return _FailureObservation(
                    BC017FailureReason.FIXTURE_TRUTH_INVALID,
                    metric_code,
                    _finite_or_zero(value),
                    0.0,
                    one,
                )
        return None

    def _next_contact_active(self, frame: BC017Frame) -> list[bool]:
        result: list[bool] = []
        for index, key in enumerate(CONTACT_KEYS):
            force = frame.contact_force_n[key]
            if self._contact_active[index]:
                result.append(force > self.calibration.contact_off_n)
            else:
                result.append(force >= self.calibration.contact_on_n)
        return result

    def _progress_and_beyond(
        self,
        frame: BC017Frame,
    ) -> tuple[tuple[float, float], tuple[bool, bool]]:
        progress = (
            self.calibration.left_travel.normalize(frame.left_q_m),
            self.calibration.right_travel.normalize(frame.right_q_m),
        )
        beyond = (
            self.calibration.left_travel.is_strictly_beyond(frame.left_q_m),
            self.calibration.right_travel.is_strictly_beyond(frame.right_q_m),
        )
        return progress, beyond

    def _update_debounced_guards(
        self,
        frame: BC017Frame,
        progress: tuple[float, float],
    ) -> tuple[_FailureObservation | None, bool]:
        tolerance = self.calibration.travel_tolerance_fraction
        observations = {
            BC017FailureReason.LEFT_TRAVEL_LIMIT_EXCEEDED: progress[0],
            BC017FailureReason.RIGHT_TRAVEL_LIMIT_EXCEEDED: progress[1],
            BC017FailureReason.FIXTURE_TRANSLATION_DRIFT: (
                frame.fixture_translation_drift_m
            ),
            BC017FailureReason.FIXTURE_ROTATION_DRIFT: (
                frame.fixture_rotation_drift_rad
            ),
        }
        violated = {
            BC017FailureReason.LEFT_TRAVEL_LIMIT_EXCEEDED: not (
                -tolerance <= progress[0] <= 1.0 + tolerance
            ),
            BC017FailureReason.RIGHT_TRAVEL_LIMIT_EXCEEDED: not (
                -tolerance <= progress[1] <= 1.0 + tolerance
            ),
            BC017FailureReason.FIXTURE_TRANSLATION_DRIFT: (
                frame.fixture_translation_drift_m
                > self.calibration.fixture_translation_limit_m
            ),
            BC017FailureReason.FIXTURE_ROTATION_DRIFT: (
                frame.fixture_rotation_drift_rad
                > self.calibration.fixture_rotation_limit_rad
            ),
        }
        priority = BC017_DEBOUNCED_FIRST_CAUSE_REASON_PRIORITY
        for reason in priority:
            self._pending_failure_steps[reason] = (
                self._pending_failure_steps[reason] + 1 if violated[reason] else 0
            )

        for reason in priority:
            required = self.calibration.debounce_steps_for(reason)
            if self._pending_failure_steps[reason] < required:
                continue
            observed = observations[reason]
            if reason in _TRAVEL_FAILURES:
                limit = -tolerance if observed < -tolerance else 1.0 + tolerance
                metric_code = (
                    BC017FailureMetric.LEFT_DRAWER_PROGRESS
                    if reason is BC017FailureReason.LEFT_TRAVEL_LIMIT_EXCEEDED
                    else BC017FailureMetric.RIGHT_DRAWER_PROGRESS
                )
            elif reason is BC017FailureReason.FIXTURE_TRANSLATION_DRIFT:
                limit = self.calibration.fixture_translation_limit_m
                metric_code = BC017FailureMetric.FIXTURE_TRANSLATION_DRIFT_M
            else:
                limit = self.calibration.fixture_rotation_limit_rad
                metric_code = BC017FailureMetric.FIXTURE_ROTATION_DRIFT_RAD
            return (
                _FailureObservation(
                    reason,
                    metric_code,
                    float(observed),
                    float(limit),
                    required,
                ),
                False,
            )
        return None, not any(violated.values())

    def _invalidate_attempt_progress(self) -> None:
        self._event_step[:] = (-1, -1)
        self._stable_steps = 0

    def _latch_failure(
        self,
        failure: _FailureObservation,
        frame: BC017Frame,
    ) -> None:
        self._latch_failure_at(
            failure,
            physics_step=frame.physics_step,
            physics_time_s=frame.physics_time_s,
        )

    def _latch_failure_at(
        self,
        failure: _FailureObservation,
        *,
        physics_step: int,
        physics_time_s: float,
    ) -> None:
        if self._terminal_evidence is not None:
            return
        self._terminal_evidence = BC017TerminalEvidence(
            success=False,
            failure_reason=failure.reason,
            metric=failure.metric,
            metric_code=failure.metric_code,
            observed=_finite_or_zero(failure.observed),
            limit=_finite_or_zero(failure.limit),
            physics_step=physics_step,
            physics_time_s=_finite_or_zero(physics_time_s),
            debounce_steps=failure.debounce_steps,
        )

    def snapshot(self) -> BC017Snapshot:
        evidence = self._terminal_evidence
        if self._event_step[0] >= 0 and self._event_step[1] >= 0:
            sync_delta_steps = abs(self._event_step[0] - self._event_step[1])
            sync_delta_seconds = sync_delta_steps * PHYSICS_DT_SECONDS
        else:
            sync_delta_steps = -1
            sync_delta_seconds = -1.0
        return BC017Snapshot(
            primed=self._primed,
            success=self.success,
            failure=self.failure,
            terminated=self.terminated,
            failure_reason_code=(
                evidence.reason_code if evidence is not None else 0
            ),
            terminal_metric_code=(
                int(evidence.metric_code) if evidence is not None else 0
            ),
            truth_valid=self._truth_valid,
            semantic_valid=self._semantic_valid,
            physics_step=self._last_physics_step,
            physics_time_s=self._last_physics_time_s,
            q_m=tuple(self._last_q),
            qd_m_s=tuple(self._last_qd),
            drawer_progress=tuple(self._last_progress),
            contact_force_n=tuple(self._last_force),
            contact_readable=tuple(self._last_contact_readable),
            contact_active=tuple(self._contact_active),
            beyond_threshold=tuple(self._last_beyond),
            threshold_event_step=tuple(self._event_step),
            sync_delta_steps=sync_delta_steps,
            sync_delta_seconds=sync_delta_seconds,
            stable_steps=self._stable_steps,
            fixture_drift=tuple(self._last_fixture_drift),
            terminal_evidence=evidence,
        )

    def public_observation(self) -> np.ndarray:
        """Return the fixed-order finite float64 public task observation."""

        evidence = self._terminal_evidence
        terminal_step = evidence.physics_step if evidence is not None else -1
        terminal_time = evidence.physics_time_s if evidence is not None else -1.0
        terminal_observed = evidence.observed if evidence is not None else 0.0
        terminal_limit = evidence.limit if evidence is not None else 0.0
        terminal_debounce = evidence.debounce_steps if evidence is not None else 0
        reason_code = evidence.reason_code if evidence is not None else 0
        metric_code = int(evidence.metric_code) if evidence is not None else 0
        sync_delta = (
            abs(self._event_step[0] - self._event_step[1])
            if min(self._event_step) >= 0
            else -1
        )
        values = {
            "physics_step": float(self._last_physics_step),
            "physics_time_s": self._last_physics_time_s,
            "left_q_m": self._last_q[0],
            "right_q_m": self._last_q[1],
            "left_qd_m_s": self._last_qd[0],
            "right_qd_m_s": self._last_qd[1],
            "left_progress": self._last_progress[0],
            "right_progress": self._last_progress[1],
            "left_dof_readable": float(self._last_dof_readable[0]),
            "right_dof_readable": float(self._last_dof_readable[1]),
            "left_beyond_60_percent": float(self._last_beyond[0]),
            "right_beyond_60_percent": float(self._last_beyond[1]),
            "ll_force_n": self._last_force[0],
            "lr_force_n": self._last_force[1],
            "rl_force_n": self._last_force[2],
            "rr_force_n": self._last_force[3],
            "ll_readable": float(self._last_contact_readable[0]),
            "lr_readable": float(self._last_contact_readable[1]),
            "rl_readable": float(self._last_contact_readable[2]),
            "rr_readable": float(self._last_contact_readable[3]),
            "ll_active": float(self._contact_active[0]),
            "lr_active": float(self._contact_active[1]),
            "rl_active": float(self._contact_active[2]),
            "rr_active": float(self._contact_active[3]),
            "fixture_truth_readable": float(self._last_fixture_readable),
            "fixture_translation_drift_m": self._last_fixture_drift[0],
            "fixture_rotation_drift_rad": self._last_fixture_drift[1],
            "left_event_seen": float(self._event_step[0] >= 0),
            "right_event_seen": float(self._event_step[1] >= 0),
            "left_event_step": float(self._event_step[0]),
            "right_event_step": float(self._event_step[1]),
            "sync_delta_steps": float(sync_delta),
            "stable_steps": float(self._stable_steps),
            "truth_valid": float(self._truth_valid),
            "semantic_valid": float(self._semantic_valid),
            "calibration_formal": float(self.calibration.formal),
            "success": float(self.success),
            "failure": float(self.failure),
            "terminal_reason_code": float(reason_code),
            "terminal_metric_code": float(metric_code),
            "terminal_step": float(terminal_step),
            "terminal_time_s": float(terminal_time),
            "terminal_observed": float(terminal_observed),
            "terminal_limit": float(terminal_limit),
            "terminal_debounce_steps": float(terminal_debounce),
        }
        if set(values) != set(PUBLIC_OBSERVATION_FIELDS):
            raise RuntimeError("BC017 public observation values are not exhaustive")
        observation = np.asarray(
            [values[field] for field in PUBLIC_OBSERVATION_FIELDS],
            dtype=np.float64,
        )
        self._validate_observation(
            observation,
            PUBLIC_OBSERVATION_SHAPE,
            "public",
        )
        return observation

    def hidden_observation(self) -> np.ndarray:
        """Return all non-physical latches/counters needed for replay."""

        evidence = self._terminal_evidence
        terminal_step = evidence.physics_step if evidence is not None else -1
        terminal_time = evidence.physics_time_s if evidence is not None else -1.0
        terminal_observed = evidence.observed if evidence is not None else 0.0
        terminal_limit = evidence.limit if evidence is not None else 0.0
        terminal_debounce = evidence.debounce_steps if evidence is not None else 0
        reason_code = evidence.reason_code if evidence is not None else 0
        metric_code = int(evidence.metric_code) if evidence is not None else 0
        observation = np.asarray(
            [
                float(self._primed),
                float(self._last_physics_step),
                self._last_physics_time_s,
                float(self._previous_semantics_valid),
                *map(float, self._previous_beyond),
                *map(float, self._previous_contact_active),
                *map(float, self._contact_active),
                float(self._event_step[0]),
                float(self._event_step[1]),
                float(self._stable_steps),
                float(
                    self._pending_failure_steps[
                        BC017FailureReason.LEFT_TRAVEL_LIMIT_EXCEEDED
                    ]
                ),
                float(
                    self._pending_failure_steps[
                        BC017FailureReason.RIGHT_TRAVEL_LIMIT_EXCEEDED
                    ]
                ),
                float(
                    self._pending_failure_steps[
                        BC017FailureReason.FIXTURE_TRANSLATION_DRIFT
                    ]
                ),
                float(
                    self._pending_failure_steps[
                        BC017FailureReason.FIXTURE_ROTATION_DRIFT
                    ]
                ),
                float(self._truth_valid),
                float(self._semantic_valid),
                float(evidence is not None),
                float(self.success),
                float(reason_code),
                float(metric_code),
                float(terminal_step),
                float(terminal_time),
                float(terminal_observed),
                float(terminal_limit),
                float(terminal_debounce),
                float(self.calibration.formal),
            ],
            dtype=np.float64,
        )
        self._validate_observation(
            observation,
            HIDDEN_OBSERVATION_SHAPE,
            "hidden",
        )
        return observation

    @staticmethod
    def _validate_observation(
        observation: np.ndarray,
        expected_shape: tuple[int, ...],
        schema_name: str,
    ) -> None:
        if observation.dtype != np.float64:
            raise RuntimeError(f"BC017 {schema_name} observation must be float64")
        if observation.shape != expected_shape:
            raise RuntimeError(
                f"BC017 {schema_name} observation schema drifted: "
                f"expected {expected_shape}, got {observation.shape}"
            )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError(f"BC017 {schema_name} observation is non-finite")


__all__ = [
    "ASSIGNED_CONTACT_KEYS",
    "BC017Calibration",
    "BC017_CONTRACT_SCHEMA_DIGEST_SHA256",
    "BC017_CONTRACT_SCHEMA_NAME",
    "BC017_CONTRACT_SCHEMA_VERSION",
    "BC017_DEBOUNCED_FIRST_CAUSE_REASON_PRIORITY",
    "BC017_HARD_FIRST_CAUSE_METRIC_PRIORITY",
    "BC017_OBSERVATION_DTYPE",
    "BC017_PROPRIO_FLATTEN_ORDER",
    "BC017FailureMetric",
    "BC017FailureReason",
    "BC017Frame",
    "BC017Snapshot",
    "BC017State",
    "BC017TerminalEvidence",
    "CONTACT_KEYS",
    "DrawerTravelCalibration",
    "HIDDEN_OBSERVATION_FIELDS",
    "HIDDEN_OBSERVATION_SHAPE",
    "OPEN_THRESHOLD",
    "PHYSICS_DT_SECONDS",
    "PHYSICS_HZ",
    "PUBLIC_OBSERVATION_GROUPS",
    "PUBLIC_OBSERVATION_FIELDS",
    "PUBLIC_OBSERVATION_SHAPE",
    "STABILITY_STEPS",
    "SYNC_WINDOW_SECONDS",
    "SYNC_WINDOW_STEPS",
    "TASK_ID",
    "TASK_NAME",
    "TASK_REQUIREMENT",
    "TASK_TITLE",
    "WRONG_CONTACT_KEYS",
    "bc017_observation_layout_metadata",
]
