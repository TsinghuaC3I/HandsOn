from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from handson.sim.isaaclab.b320_bc017_contract import (
    BC017Calibration,
    BC017_CONTRACT_SCHEMA_DIGEST_SHA256,
    BC017_CONTRACT_SCHEMA_NAME,
    BC017_CONTRACT_SCHEMA_VERSION,
    BC017_DEBOUNCED_FIRST_CAUSE_REASON_PRIORITY,
    BC017_HARD_FIRST_CAUSE_METRIC_PRIORITY,
    BC017_OBSERVATION_DTYPE,
    BC017_PROPRIO_FLATTEN_ORDER,
    BC017FailureMetric,
    BC017FailureReason,
    BC017Frame,
    BC017State,
    BC017TerminalEvidence,
    CONTACT_KEYS,
    DRAWER_SLOT_COUNT,
    DRAWER_SLOT_INDICES,
    DrawerTravelCalibration,
    HIDDEN_OBSERVATION_FIELDS,
    HIDDEN_OBSERVATION_SHAPE,
    OPEN_THRESHOLD,
    PHYSICS_DT_SECONDS,
    PHYSICS_HZ,
    PUBLIC_OBSERVATION_FIELDS,
    PUBLIC_OBSERVATION_GROUPS,
    PUBLIC_OBSERVATION_SHAPE,
    STABILITY_STEPS,
    SYNC_WINDOW_SECONDS,
    SYNC_WINDOW_STEPS,
    TASK_ID,
    TASK_NAME,
    TASK_REQUIREMENT,
    bc017_observation_layout_metadata,
)


AUTHORITATIVE_REQUIREMENT = (
    "Use each assigned hand to pull the highlighted drawer on its side beyond "
    "60 percent travel; each side independently selects one of three drawers "
    "per episode, and both thresholds must be crossed within 1.2 seconds."
)


def _calibration(
    *,
    left: DrawerTravelCalibration | None = None,
    right: DrawerTravelCalibration | None = None,
    formal: bool = True,
    **kwargs: object,
) -> BC017Calibration:
    if formal:
        left_travel = left or DrawerTravelCalibration(
            0.0,
            1.0,
            formal=True,
            source_digest_sha256="a" * 64,
        )
        right_travel = right or DrawerTravelCalibration(
            0.0,
            1.0,
            formal=True,
            source_digest_sha256="b" * 64,
        )
        digest = "c" * 64
    else:
        left_travel = left or DrawerTravelCalibration(0.0, 1.0)
        right_travel = right or DrawerTravelCalibration(0.0, 1.0)
        digest = "0" * 64
    return BC017Calibration(
        left_travel=left_travel,
        right_travel=right_travel,
        formal=formal,
        source_digest_sha256=digest,
        **kwargs,
    )


def _frame(
    step: int,
    *,
    time_s: float | None = None,
    left: float = 0.0,
    right: float = 0.0,
    left_qd: float = 0.0,
    right_qd: float = 0.0,
    left_dof_readable: bool = True,
    right_dof_readable: bool = True,
    ll: float = 0.25,
    lr: float = 0.0,
    rl: float = 0.0,
    rr: float = 0.25,
    ll_readable: bool = True,
    lr_readable: bool = True,
    rl_readable: bool = True,
    rr_readable: bool = True,
    fixture_translation: float = 0.0,
    fixture_rotation: float = 0.0,
    fixture_readable: bool = True,
) -> BC017Frame:
    return BC017Frame(
        physics_step=step,
        physics_time_s=(step * PHYSICS_DT_SECONDS if time_s is None else time_s),
        left_q_m=left,
        right_q_m=right,
        left_qd_m_s=left_qd,
        right_qd_m_s=right_qd,
        left_dof_readable=left_dof_readable,
        right_dof_readable=right_dof_readable,
        ll_force_n=ll,
        lr_force_n=lr,
        rl_force_n=rl,
        rr_force_n=rr,
        ll_readable=ll_readable,
        lr_readable=lr_readable,
        rl_readable=rl_readable,
        rr_readable=rr_readable,
        fixture_translation_drift_m=fixture_translation,
        fixture_rotation_drift_rad=fixture_rotation,
        fixture_truth_readable=fixture_readable,
    )


def _state() -> BC017State:
    state = BC017State(_calibration())
    state.prime(_frame(0))
    return state


def _succeed(state: BC017State) -> None:
    first = state.update(_frame(1, left=0.61, right=0.61))
    assert first.stable_steps == 1
    assert first.success is False
    second = state.update(
        _frame(2, left=0.61, right=0.61, ll=0.0, rr=0.0)
    )
    assert second.success is True


def test_identity_and_literal_physics_contract_are_exact() -> None:
    assert TASK_ID == "B320-BC017"
    assert TASK_NAME == "t15_open_drawers"
    assert TASK_REQUIREMENT == AUTHORITATIVE_REQUIREMENT
    assert PHYSICS_HZ == 120
    assert OPEN_THRESHOLD == 0.60
    assert SYNC_WINDOW_SECONDS == 1.2
    assert SYNC_WINDOW_STEPS == 144
    assert STABILITY_STEPS == 2
    assert CONTACT_KEYS == ("LL", "LR", "RL", "RR")
    assert DRAWER_SLOT_COUNT == 3
    assert DRAWER_SLOT_INDICES == (0, 1, 2)


def test_failure_metric_codes_and_contract_schema_are_append_only() -> None:
    assert {reason.name: int(reason) for reason in BC017FailureReason} == {
        "NONE": 0,
        "CALIBRATION_NOT_FORMAL": 1,
        "PHYSICS_STEP_DISCONTINUITY": 2,
        "PHYSICS_TIME_DISCONTINUITY": 3,
        "LEFT_DOF_TRUTH_UNAVAILABLE": 4,
        "RIGHT_DOF_TRUTH_UNAVAILABLE": 5,
        "LEFT_DOF_NON_FINITE": 6,
        "RIGHT_DOF_NON_FINITE": 7,
        "LL_CONTACT_TRUTH_UNAVAILABLE": 8,
        "LR_CONTACT_TRUTH_UNAVAILABLE": 9,
        "RL_CONTACT_TRUTH_UNAVAILABLE": 10,
        "RR_CONTACT_TRUTH_UNAVAILABLE": 11,
        "LL_CONTACT_FORCE_INVALID": 12,
        "LR_CONTACT_FORCE_INVALID": 13,
        "RL_CONTACT_FORCE_INVALID": 14,
        "RR_CONTACT_FORCE_INVALID": 15,
        "FIXTURE_TRUTH_UNAVAILABLE": 16,
        "FIXTURE_TRUTH_INVALID": 17,
        "LEFT_TRAVEL_LIMIT_EXCEEDED": 18,
        "RIGHT_TRAVEL_LIMIT_EXCEEDED": 19,
        "FIXTURE_TRANSLATION_DRIFT": 20,
        "FIXTURE_ROTATION_DRIFT": 21,
        "BC017_LIVE_AUDIT_BINDING_DRIFT": 22,
    }
    assert {metric.name: int(metric) for metric in BC017FailureMetric} == {
        "NONE": 0,
        "STABLE_BEYOND_THRESHOLD_SAMPLES": 1,
        "CALIBRATION_FORMAL": 2,
        "PHYSICS_STEP_DELTA": 3,
        "PHYSICS_TIME_FINITE": 4,
        "PHYSICS_TIME_DELTA_S": 5,
        "LEFT_DOF_READABLE": 6,
        "LEFT_DOF_Q_M_FINITE": 7,
        "LEFT_DOF_QD_M_S_FINITE": 8,
        "RIGHT_DOF_READABLE": 9,
        "RIGHT_DOF_Q_M_FINITE": 10,
        "RIGHT_DOF_QD_M_S_FINITE": 11,
        "LL_CONTACT_READABLE": 12,
        "LL_FORCE_FINITE": 13,
        "LL_FORCE_N": 14,
        "LR_CONTACT_READABLE": 15,
        "LR_FORCE_FINITE": 16,
        "LR_FORCE_N": 17,
        "RL_CONTACT_READABLE": 18,
        "RL_FORCE_FINITE": 19,
        "RL_FORCE_N": 20,
        "RR_CONTACT_READABLE": 21,
        "RR_FORCE_FINITE": 22,
        "RR_FORCE_N": 23,
        "FIXTURE_TRUTH_READABLE": 24,
        "FIXTURE_TRANSLATION_DRIFT_M": 25,
        "FIXTURE_ROTATION_DRIFT_RAD": 26,
        "LEFT_DRAWER_PROGRESS": 27,
        "RIGHT_DRAWER_PROGRESS": 28,
        "BC017_LIVE_AUDIT_BINDING_VALID": 29,
    }
    assert len({metric.metric_name for metric in BC017FailureMetric}) == len(
        BC017FailureMetric
    )
    assert BC017_CONTRACT_SCHEMA_NAME == (
        "handson.b320_bc017.physical_contract"
    )
    assert BC017_CONTRACT_SCHEMA_VERSION == 3
    assert BC017_CONTRACT_SCHEMA_DIGEST_SHA256 == (
        "45bcdc6a0cb9ed2f0a89d839d1ef99e1be3ceecc8ec293a3549543ff36e6ab2c"
    )
    assert PUBLIC_OBSERVATION_SHAPE == (47,)
    assert HIDDEN_OBSERVATION_SHAPE == (35,)
    assert PUBLIC_OBSERVATION_GROUPS[0][1][-2:] == (
        "left_target_slot_index",
        "right_target_slot_index",
    )
    assert HIDDEN_OBSERVATION_FIELDS[-2:] == (
        "left_target_slot_index",
        "right_target_slot_index",
    )
    for fields in (PUBLIC_OBSERVATION_FIELDS, HIDDEN_OBSERVATION_FIELDS):
        reason_index = fields.index("terminal_reason_code")
        assert fields[reason_index + 1] == "terminal_metric_code"


def test_public_output_layout_is_the_digest_bound_config_flatten_schema() -> None:
    layout = bc017_observation_layout_metadata()
    assert layout["dtype"] == BC017_OBSERVATION_DTYPE == "float64"
    assert tuple(group["key"] for group in layout["public_groups"]) == (
        "bc017_drawer_state",
        "bc017_contact_state",
        "bc017_stage_state",
        "bc017_terminal_state",
    )
    assert tuple(layout["config_flatten_order"]) == BC017_PROPRIO_FLATTEN_ORDER
    assert tuple(layout["public_flattened_fields"]) == PUBLIC_OBSERVATION_FIELDS
    assert tuple(layout["public_flattened_shape"]) == PUBLIC_OBSERVATION_SHAPE
    assert PUBLIC_OBSERVATION_FIELDS == tuple(
        field
        for _key, fields in PUBLIC_OBSERVATION_GROUPS
        for field in fields
    )
    assert PUBLIC_OBSERVATION_FIELDS[8] == "left_dof_readable"
    for declared, (key, fields) in zip(
        layout["public_groups"],
        PUBLIC_OBSERVATION_GROUPS,
        strict=True,
    ):
        assert declared == {
            "key": key,
            "fields": list(fields),
            "shape": [len(fields)],
            "dtype": "float64",
        }


def test_travel_calibration_is_direction_independent_and_strict_at_sixty_percent() -> None:
    for travel in (
        DrawerTravelCalibration(
            -0.2,
            0.8,
            formal=True,
            source_digest_sha256="a" * 64,
        ),
        DrawerTravelCalibration(
            0.8,
            -0.2,
            formal=True,
            source_digest_sha256="b" * 64,
        ),
    ):
        exact = travel.threshold_q_m()
        assert travel.normalize(exact) == pytest.approx(0.60)
        assert travel.is_strictly_beyond(exact) is False
        direction = math.inf if travel.span_m > 0.0 else -math.inf
        assert travel.is_strictly_beyond(math.nextafter(exact, direction)) is True


def test_travel_calibration_rejects_nonfinite_span_from_finite_endpoints() -> None:
    with pytest.raises(ValueError, match="span must be finite"):
        DrawerTravelCalibration(-1.0e308, 1.0e308)


def test_calibration_digest_is_deterministic_and_bound_to_content() -> None:
    calibration = _calibration()
    same = _calibration()
    changed = _calibration(contact_on_n=0.16)
    assert calibration.formal is True
    assert calibration.left_travel.formal is True
    assert len(calibration.contract_digest_sha256) == 64
    assert calibration.contract_digest_sha256 == same.contract_digest_sha256
    assert calibration.contract_digest_sha256 != changed.contract_digest_sha256
    assert (
        calibration.left_travel.contract_digest_sha256
        != calibration.right_travel.contract_digest_sha256
    )


def test_formal_calibration_cannot_claim_provisional_evidence() -> None:
    with pytest.raises(ValueError, match="formal drawer travel"):
        DrawerTravelCalibration(0.0, 1.0, formal=True)
    with pytest.raises(ValueError, match="formal BC017 calibration"):
        BC017Calibration(
            left_travel=DrawerTravelCalibration(0.0, 1.0),
            right_travel=DrawerTravelCalibration(0.0, 1.0),
            formal=True,
            source_digest_sha256="c" * 64,
        )


def test_static_task_calibration_can_prime_without_acceptance_evidence() -> None:
    state = BC017State(_calibration(formal=False))
    result = state.prime(_frame(12))
    assert result.failure is False
    assert result.terminated is False
    assert result.terminal_evidence is None


def test_terminal_evidence_validates_metric_string_code_and_reason() -> None:
    common = {
        "success": False,
        "failure_reason": BC017FailureReason.LEFT_DOF_NON_FINITE,
        "metric": "left_dof_q_m_finite",
        "metric_code": BC017FailureMetric.LEFT_DOF_Q_M_FINITE,
        "observed": 0.0,
        "limit": 1.0,
        "physics_step": 7,
        "physics_time_s": 7 * PHYSICS_DT_SECONDS,
        "debounce_steps": 1,
    }
    evidence = BC017TerminalEvidence(**common)
    assert evidence.metric_code is BC017FailureMetric.LEFT_DOF_Q_M_FINITE

    with pytest.raises(ValueError, match="string/code mismatch"):
        BC017TerminalEvidence(
            **{
                **common,
                "metric": "left_dof_qd_m_s_finite",
            }
        )
    with pytest.raises(ValueError, match="reason/metric code mismatch"):
        BC017TerminalEvidence(
            **{
                **common,
                "metric": "right_dof_q_m_finite",
                "metric_code": BC017FailureMetric.RIGHT_DOF_Q_M_FINITE,
            }
        )
    with pytest.raises(ValueError, match="non-NONE metric code"):
        BC017TerminalEvidence(
            **{
                **common,
                "metric": "none",
                "metric_code": BC017FailureMetric.NONE,
            }
        )


def test_frame_boundary_is_strictly_typed_but_preserves_nonfinite_truth_for_attribution() -> None:
    with pytest.raises(TypeError, match="physics_step"):
        replace(_frame(0), physics_step=True)
    with pytest.raises(TypeError, match="left_q_m"):
        replace(_frame(0), left_q_m="0.0")
    with pytest.raises(TypeError, match="ll_readable"):
        replace(_frame(0), ll_readable=1)
    frame = replace(_frame(0), left_q_m=math.nan)
    assert math.isnan(frame.left_q_m)
    with pytest.raises(FrozenInstanceError):
        frame.left_q_m = 0.0  # type: ignore[misc]


def test_reset_and_prime_are_explicit_and_prime_does_not_create_events() -> None:
    state = BC017State(_calibration())
    with pytest.raises(RuntimeError, match="explicit prime"):
        state.update(_frame(1))
    origin = state.prime(_frame(100, time_s=7.0, left=0.61, right=0.61))
    assert origin.primed is True
    assert origin.physics_step == 100
    assert origin.threshold_event_step == (-1, -1)
    assert origin.stable_steps == 0
    with pytest.raises(RuntimeError, match="fresh reset"):
        state.prime(_frame(100, time_s=7.0))
    reset = state.reset()
    assert reset.primed is False
    assert reset.physics_step == -1
    assert reset.threshold_event_step == (-1, -1)


def test_target_slots_are_strict_episode_identity_and_replay_state() -> None:
    state = BC017State(_calibration())
    reset = state.reset(target_slots=(2, 1))
    assert state.target_slots == reset.target_slots == (2, 1)
    with pytest.raises(AttributeError):
        setattr(state, "target_slots", (0, 0))

    public_values = dict(
        zip(PUBLIC_OBSERVATION_FIELDS, state.public_observation(), strict=True)
    )
    hidden_values = dict(
        zip(HIDDEN_OBSERVATION_FIELDS, state.hidden_observation(), strict=True)
    )
    for values in (public_values, hidden_values):
        assert values["left_target_slot_index"] == 2.0
        assert values["right_target_slot_index"] == 1.0

    assert state.prime(_frame(0)).target_slots == (2, 1)
    _succeed(state)
    assert state.snapshot().target_slots == (2, 1)
    # Origin acceptance resets state-machine latches without selecting a new
    # target.  Omitting the argument must therefore preserve episode identity.
    assert state.reset().target_slots == (2, 1)
    with pytest.raises(ValueError, match="target_slots"):
        state.reset(target_slots=(3, 0))
    assert state.target_slots == (2, 1)


@pytest.mark.parametrize(
    ("target_slots", "error"),
    [
        ([0, 1], TypeError),
        ((0,), TypeError),
        ((True, 1), TypeError),
        ((1.0, 1), TypeError),
        ((-1, 1), ValueError),
        ((3, 1), ValueError),
        ((1, 3), ValueError),
    ],
)
def test_target_slots_reject_noncanonical_values(
    target_slots: object,
    error: type[Exception],
) -> None:
    state = BC017State(_calibration())
    with pytest.raises(error, match="target_slots"):
        state.reset(target_slots=target_slots)  # type: ignore[arg-type]


def test_exactly_sixty_percent_is_not_a_crossing() -> None:
    state = _state()
    exact = state.update(_frame(1, left=0.60, right=0.60))
    assert exact.beyond_threshold == (False, False)
    assert exact.threshold_event_step == (-1, -1)
    beyond = state.update(
        _frame(
            2,
            left=math.nextafter(0.60, math.inf),
            right=math.nextafter(0.60, math.inf),
        )
    )
    assert beyond.threshold_event_step == (2, 2)


def test_force_hysteresis_uses_distinct_on_and_off_thresholds() -> None:
    state = BC017State(_calibration())
    origin = state.prime(_frame(0, ll=0.15, rr=0.15))
    assert origin.contact_active == (True, False, False, True)
    retained = state.update(_frame(1, ll=0.0500001, rr=0.0500001))
    assert retained.contact_active == (True, False, False, True)
    released = state.update(_frame(2, ll=0.05, rr=0.05))
    assert released.contact_active == (False, False, False, False)


def test_crossing_requires_assigned_contact_on_preceding_and_crossing_samples() -> None:
    preceding_wrong = BC017State(_calibration())
    preceding_wrong.prime(_frame(0, rl=0.25))
    result = preceding_wrong.update(_frame(1, left=0.61, right=0.61))
    assert result.threshold_event_step == (-1, 1)

    crossing_wrong = _state()
    result = crossing_wrong.update(
        _frame(1, left=0.61, right=0.61, lr=0.25, rl=0.25)
    )
    assert result.threshold_event_step == (-1, -1)

    missing_assigned = BC017State(_calibration())
    missing_assigned.prime(_frame(0, ll=0.0, rr=0.0))
    result = missing_assigned.update(_frame(1, left=0.61, right=0.61))
    assert result.threshold_event_step == (-1, -1)


def test_wrong_hand_contact_earlier_in_excursion_does_not_taint_crossing() -> None:
    state = BC017State(_calibration())
    state.prime(_frame(0, rl=0.25))
    clean_preceding = state.update(_frame(1, left=0.50, right=0.0))
    assert clean_preceding.threshold_event_step == (-1, -1)
    crossing = state.update(_frame(2, left=0.61, right=0.61))
    assert crossing.threshold_event_step == (2, 2)
    assert state.update(_frame(3, left=0.61, right=0.61)).success is True


def test_wrong_hand_contact_after_crossing_does_not_retroactively_taint_event() -> None:
    state = _state()
    crossing = state.update(_frame(1, left=0.61, right=0.61))
    assert crossing.threshold_event_step == (1, 1)
    result = state.update(
        _frame(
            2,
            left=0.61,
            right=0.61,
            ll=0.0,
            lr=0.25,
            rl=0.25,
            rr=0.0,
        )
    )
    assert result.success is True
    assert result.threshold_event_step == (1, 1)


def test_two_samples_are_required_to_deglitch_threshold_crossing() -> None:
    state = _state()
    spike = state.update(_frame(1, left=0.61, right=0.61))
    assert spike.stable_steps == 1
    assert spike.success is False
    rollback = state.update(_frame(2, left=0.60, right=0.60))
    assert rollback.stable_steps == 0
    assert rollback.threshold_event_step == (-1, -1)
    retry = state.update(_frame(3, left=0.61, right=0.61))
    assert retry.stable_steps == 1
    assert state.update(_frame(4, left=0.61, right=0.61)).success is True


def test_exactly_144_physics_steps_is_inclusive() -> None:
    state = _state()
    state.update(_frame(1, left=0.61, right=0.0))
    for step in range(2, 145):
        state.update(_frame(step, left=0.61, right=0.0))
    crossing = state.update(_frame(145, left=0.61, right=0.61))
    assert crossing.threshold_event_step == (1, 145)
    assert crossing.sync_delta_steps == 144
    assert crossing.sync_delta_seconds == pytest.approx(1.2)
    assert crossing.success is False
    assert state.update(_frame(146, left=0.61, right=0.61)).success is True


def test_145_step_miss_requires_rollback_and_retry() -> None:
    state = _state()
    state.update(_frame(1, left=0.61, right=0.0))
    for step in range(2, 146):
        state.update(_frame(step, left=0.61, right=0.0))
    late = state.update(_frame(146, left=0.61, right=0.61))
    assert late.threshold_event_step == (1, 146)
    assert late.sync_delta_steps == 145
    assert late.success is False

    rollback = state.update(_frame(147, left=0.60, right=0.61))
    assert rollback.threshold_event_step == (-1, 146)
    retry = state.update(_frame(148, left=0.61, right=0.61))
    assert retry.threshold_event_step == (148, 146)
    assert retry.stable_steps == 1
    assert state.update(_frame(149, left=0.61, right=0.61)).success is True


@pytest.mark.parametrize(
    ("replacement", "reason", "metric", "metric_code"),
    [
        (
            {"left_dof_readable": False},
            BC017FailureReason.LEFT_DOF_TRUTH_UNAVAILABLE,
            "left_dof_readable",
            BC017FailureMetric.LEFT_DOF_READABLE,
        ),
        (
            {"right_dof_readable": False},
            BC017FailureReason.RIGHT_DOF_TRUTH_UNAVAILABLE,
            "right_dof_readable",
            BC017FailureMetric.RIGHT_DOF_READABLE,
        ),
        (
            {"left_q_m": math.nan},
            BC017FailureReason.LEFT_DOF_NON_FINITE,
            "left_dof_q_m_finite",
            BC017FailureMetric.LEFT_DOF_Q_M_FINITE,
        ),
        (
            {"left_qd_m_s": math.inf},
            BC017FailureReason.LEFT_DOF_NON_FINITE,
            "left_dof_qd_m_s_finite",
            BC017FailureMetric.LEFT_DOF_QD_M_S_FINITE,
        ),
        (
            {"right_q_m": math.nan},
            BC017FailureReason.RIGHT_DOF_NON_FINITE,
            "right_dof_q_m_finite",
            BC017FailureMetric.RIGHT_DOF_Q_M_FINITE,
        ),
        (
            {"right_qd_m_s": math.inf},
            BC017FailureReason.RIGHT_DOF_NON_FINITE,
            "right_dof_qd_m_s_finite",
            BC017FailureMetric.RIGHT_DOF_QD_M_S_FINITE,
        ),
        (
            {"ll_readable": False},
            BC017FailureReason.LL_CONTACT_TRUTH_UNAVAILABLE,
            "ll_contact_readable",
            BC017FailureMetric.LL_CONTACT_READABLE,
        ),
        (
            {"lr_readable": False},
            BC017FailureReason.LR_CONTACT_TRUTH_UNAVAILABLE,
            "lr_contact_readable",
            BC017FailureMetric.LR_CONTACT_READABLE,
        ),
        (
            {"rl_readable": False},
            BC017FailureReason.RL_CONTACT_TRUTH_UNAVAILABLE,
            "rl_contact_readable",
            BC017FailureMetric.RL_CONTACT_READABLE,
        ),
        (
            {"rr_readable": False},
            BC017FailureReason.RR_CONTACT_TRUTH_UNAVAILABLE,
            "rr_contact_readable",
            BC017FailureMetric.RR_CONTACT_READABLE,
        ),
        (
            {"ll_force_n": math.nan},
            BC017FailureReason.LL_CONTACT_FORCE_INVALID,
            "ll_force_finite",
            BC017FailureMetric.LL_FORCE_FINITE,
        ),
        (
            {"ll_force_n": -0.1},
            BC017FailureReason.LL_CONTACT_FORCE_INVALID,
            "ll_force_n",
            BC017FailureMetric.LL_FORCE_N,
        ),
        (
            {"lr_force_n": math.nan},
            BC017FailureReason.LR_CONTACT_FORCE_INVALID,
            "lr_force_finite",
            BC017FailureMetric.LR_FORCE_FINITE,
        ),
        (
            {"lr_force_n": -0.1},
            BC017FailureReason.LR_CONTACT_FORCE_INVALID,
            "lr_force_n",
            BC017FailureMetric.LR_FORCE_N,
        ),
        (
            {"rl_force_n": math.inf},
            BC017FailureReason.RL_CONTACT_FORCE_INVALID,
            "rl_force_finite",
            BC017FailureMetric.RL_FORCE_FINITE,
        ),
        (
            {"rl_force_n": -0.1},
            BC017FailureReason.RL_CONTACT_FORCE_INVALID,
            "rl_force_n",
            BC017FailureMetric.RL_FORCE_N,
        ),
        (
            {"rr_force_n": math.inf},
            BC017FailureReason.RR_CONTACT_FORCE_INVALID,
            "rr_force_finite",
            BC017FailureMetric.RR_FORCE_FINITE,
        ),
        (
            {"rr_force_n": -0.1},
            BC017FailureReason.RR_CONTACT_FORCE_INVALID,
            "rr_force_n",
            BC017FailureMetric.RR_FORCE_N,
        ),
        (
            {"fixture_truth_readable": False},
            BC017FailureReason.FIXTURE_TRUTH_UNAVAILABLE,
            "fixture_truth_readable",
            BC017FailureMetric.FIXTURE_TRUTH_READABLE,
        ),
        (
            {"fixture_translation_drift_m": math.nan},
            BC017FailureReason.FIXTURE_TRUTH_INVALID,
            "fixture_translation_drift_m",
            BC017FailureMetric.FIXTURE_TRANSLATION_DRIFT_M,
        ),
        (
            {"fixture_rotation_drift_rad": math.nan},
            BC017FailureReason.FIXTURE_TRUTH_INVALID,
            "fixture_rotation_drift_rad",
            BC017FailureMetric.FIXTURE_ROTATION_DRIFT_RAD,
        ),
    ],
)
def test_each_physical_truth_channel_has_stable_first_cause_attribution(
    replacement: dict[str, object],
    reason: BC017FailureReason,
    metric: str,
    metric_code: BC017FailureMetric,
) -> None:
    state = _state()
    result = state.update(replace(_frame(1), **replacement))
    assert result.failure is True
    evidence = result.terminal_evidence
    assert evidence is not None
    assert evidence.failure_reason is reason
    assert evidence.metric == metric
    assert evidence.metric_code is metric_code
    assert evidence.physics_step == 1
    assert evidence.debounce_steps == 1


@pytest.mark.parametrize(
    ("replacement", "expected_metric"),
    (
        (
            {"left_q_m": math.nan, "left_qd_m_s": math.inf},
            BC017FailureMetric.LEFT_DOF_Q_M_FINITE,
        ),
        (
            {"left_q_m": math.nan, "right_q_m": math.nan},
            BC017FailureMetric.LEFT_DOF_Q_M_FINITE,
        ),
        (
            {
                "fixture_translation_drift_m": math.nan,
                "fixture_rotation_drift_rad": math.nan,
            },
            BC017FailureMetric.FIXTURE_TRANSLATION_DRIFT_M,
        ),
    ),
)
def test_concurrent_invalid_channels_follow_digest_bound_first_cause_priority(
    replacement: dict[str, object],
    expected_metric: BC017FailureMetric,
) -> None:
    state = _state()
    evidence = state.update(replace(_frame(1), **replacement)).terminal_evidence
    assert evidence is not None
    assert evidence.metric_code is expected_metric


def test_digest_bound_first_cause_priority_orders_concurrent_pairs() -> None:
    position = {
        metric: index
        for index, metric in enumerate(BC017_HARD_FIRST_CAUSE_METRIC_PRIORITY)
    }
    assert position[BC017FailureMetric.LEFT_DOF_Q_M_FINITE] < position[
        BC017FailureMetric.LEFT_DOF_QD_M_S_FINITE
    ]
    assert position[BC017FailureMetric.LEFT_DOF_Q_M_FINITE] < position[
        BC017FailureMetric.RIGHT_DOF_Q_M_FINITE
    ]
    assert position[BC017FailureMetric.FIXTURE_TRANSLATION_DRIFT_M] < position[
        BC017FailureMetric.FIXTURE_ROTATION_DRIFT_RAD
    ]


def test_concurrent_debounced_failures_follow_digest_bound_reason_priority() -> None:
    calibration = _calibration(
        fixture_translation_limit_m=0.002,
        fixture_rotation_limit_rad=0.01,
    )
    state = BC017State(calibration)
    state.prime(_frame(0))
    concurrent = {
        "left": 1.06,
        "fixture_translation": 0.004,
        "fixture_rotation": 0.02,
    }
    assert state.update(_frame(1, **concurrent)).failure is False
    evidence = state.update(_frame(2, **concurrent)).terminal_evidence
    assert evidence is not None
    assert evidence.failure_reason is BC017FailureReason.LEFT_TRAVEL_LIMIT_EXCEEDED
    assert BC017_DEBOUNCED_FIRST_CAUSE_REASON_PRIORITY == (
        BC017FailureReason.LEFT_TRAVEL_LIMIT_EXCEEDED,
        BC017FailureReason.RIGHT_TRAVEL_LIMIT_EXCEEDED,
        BC017FailureReason.FIXTURE_TRANSLATION_DRIFT,
        BC017FailureReason.FIXTURE_ROTATION_DRIFT,
    )


def test_live_audit_binding_drift_is_typed_and_freezes_every_state_view() -> None:
    state = _state()
    failed = state.latch_live_audit_binding_drift(
        physics_step=1,
        physics_time_s=PHYSICS_DT_SECONDS,
    )
    evidence = failed.terminal_evidence
    assert evidence is not None
    assert evidence.failure_reason is (
        BC017FailureReason.BC017_LIVE_AUDIT_BINDING_DRIFT
    )
    assert evidence.metric_code is (
        BC017FailureMetric.BC017_LIVE_AUDIT_BINDING_VALID
    )
    assert (evidence.metric, evidence.observed, evidence.limit) == (
        "bc017_live_audit_binding_valid",
        0.0,
        1.0,
    )
    public = state.public_observation().copy()
    hidden = state.hidden_observation().copy()
    assert state.update(_frame(2, left=0.61, right=0.61)) == failed
    np.testing.assert_array_equal(state.public_observation(), public)
    np.testing.assert_array_equal(state.hidden_observation(), hidden)


def test_physics_step_and_time_must_both_be_continuous_at_120_hz() -> None:
    skipped = _state()
    result = skipped.update(_frame(2))
    evidence = result.terminal_evidence
    assert evidence is not None
    assert evidence.failure_reason is BC017FailureReason.PHYSICS_STEP_DISCONTINUITY
    assert evidence.metric_code is BC017FailureMetric.PHYSICS_STEP_DELTA
    assert (evidence.metric, evidence.observed, evidence.limit) == (
        "physics_step_delta",
        2.0,
        1.0,
    )

    wrong_time = _state()
    result = wrong_time.update(_frame(1, time_s=2.0 * PHYSICS_DT_SECONDS))
    evidence = result.terminal_evidence
    assert evidence is not None
    assert evidence.failure_reason is BC017FailureReason.PHYSICS_TIME_DISCONTINUITY
    assert evidence.metric == "physics_time_delta_s"
    assert evidence.metric_code is BC017FailureMetric.PHYSICS_TIME_DELTA_S
    assert evidence.observed == pytest.approx(2.0 * PHYSICS_DT_SECONDS)
    assert evidence.limit == pytest.approx(PHYSICS_DT_SECONDS)

    nonfinite_time = BC017State(_calibration())
    result = nonfinite_time.prime(_frame(0, time_s=math.nan))
    evidence = result.terminal_evidence
    assert evidence is not None
    assert evidence.failure_reason is BC017FailureReason.PHYSICS_TIME_DISCONTINUITY
    assert evidence.metric == "physics_time_finite"
    assert evidence.metric_code is BC017FailureMetric.PHYSICS_TIME_FINITE


def test_declared_debounces_recover_from_one_sample_travel_glitch_and_then_fail_at_two() -> None:
    state = _state()
    debounces = state.failure_debounce_steps
    assert debounces[BC017FailureReason.LEFT_DOF_TRUTH_UNAVAILABLE] == 1
    assert debounces[BC017FailureReason.LL_CONTACT_TRUTH_UNAVAILABLE] == 1
    assert debounces[BC017FailureReason.LEFT_TRAVEL_LIMIT_EXCEEDED] == 2
    assert debounces[BC017FailureReason.FIXTURE_TRANSLATION_DRIFT] == 2
    with pytest.raises(TypeError):
        debounces[BC017FailureReason.LEFT_TRAVEL_LIMIT_EXCEEDED] = 1  # type: ignore[index]

    transient = state.update(_frame(1, left=1.06))
    assert transient.failure is False
    assert transient.semantic_valid is False
    recovered = state.update(_frame(2, left=0.0))
    assert recovered.failure is False
    assert recovered.semantic_valid is True

    state.update(_frame(3, left=1.06))
    failed = state.update(_frame(4, left=1.06))
    evidence = failed.terminal_evidence
    assert evidence is not None
    assert evidence.failure_reason is BC017FailureReason.LEFT_TRAVEL_LIMIT_EXCEEDED
    assert evidence.metric == "left_drawer_progress"
    assert evidence.metric_code is BC017FailureMetric.LEFT_DRAWER_PROGRESS
    assert evidence.observed == pytest.approx(1.06)
    assert evidence.limit == pytest.approx(1.05)
    assert evidence.physics_step == 4
    assert evidence.debounce_steps == 2

    right = _state()
    right.update(_frame(1, right=-0.06))
    evidence = right.update(_frame(2, right=-0.06)).terminal_evidence
    assert evidence is not None
    assert evidence.failure_reason is BC017FailureReason.RIGHT_TRAVEL_LIMIT_EXCEEDED
    assert evidence.metric == "right_drawer_progress"
    assert evidence.metric_code is BC017FailureMetric.RIGHT_DRAWER_PROGRESS


def test_fixture_drift_has_separate_translation_and_rotation_attribution() -> None:
    calibration = _calibration(
        fixture_translation_limit_m=0.002,
        fixture_rotation_limit_rad=0.01,
    )
    state = BC017State(calibration)
    state.prime(_frame(0))
    assert state.update(_frame(1, fixture_translation=0.003)).failure is False
    failed = state.update(_frame(2, fixture_translation=0.004))
    evidence = failed.terminal_evidence
    assert evidence is not None
    assert evidence.failure_reason is BC017FailureReason.FIXTURE_TRANSLATION_DRIFT
    assert evidence.metric_code is (
        BC017FailureMetric.FIXTURE_TRANSLATION_DRIFT_M
    )
    assert evidence.observed == pytest.approx(0.004)
    assert evidence.limit == pytest.approx(0.002)

    rotation = BC017State(calibration)
    rotation.prime(_frame(0))
    rotation.update(_frame(1, fixture_rotation=0.02))
    failed = rotation.update(_frame(2, fixture_rotation=0.03))
    evidence = failed.terminal_evidence
    assert evidence is not None
    assert evidence.failure_reason is BC017FailureReason.FIXTURE_ROTATION_DRIFT
    assert evidence.metric == "fixture_rotation_drift_rad"
    assert evidence.metric_code is BC017FailureMetric.FIXTURE_ROTATION_DRIFT_RAD


def test_terminal_evidence_and_all_state_are_frozen_after_success() -> None:
    state = _state()
    _succeed(state)
    evidence = state.terminal_evidence
    assert evidence is not None
    assert evidence.metric == "stable_beyond_threshold_samples"
    assert evidence.metric_code is (
        BC017FailureMetric.STABLE_BEYOND_THRESHOLD_SAMPLES
    )
    assert (evidence.observed, evidence.limit, evidence.physics_step) == (
        2.0,
        2.0,
        2,
    )
    with pytest.raises(FrozenInstanceError):
        evidence.observed = 3.0  # type: ignore[misc]

    snapshot = state.snapshot()
    public = state.public_observation().copy()
    hidden = state.hidden_observation().copy()
    after = state.update(
        _frame(
            999,
            time_s=math.nan,
            left=math.nan,
            right=math.nan,
            ll_readable=False,
        )
    )
    assert after == snapshot
    assert state.terminal_evidence is evidence
    np.testing.assert_array_equal(state.public_observation(), public)
    np.testing.assert_array_equal(state.hidden_observation(), hidden)


def test_first_failure_evidence_and_all_state_are_frozen() -> None:
    state = _state()
    failed = state.update(_frame(2))
    evidence = failed.terminal_evidence
    public = state.public_observation().copy()
    hidden = state.hidden_observation().copy()
    assert evidence is not None
    after = state.update(_frame(1, left=0.61, right=0.61))
    assert after.terminal_evidence is evidence
    np.testing.assert_array_equal(state.public_observation(), public)
    np.testing.assert_array_equal(state.hidden_observation(), hidden)


def test_public_and_hidden_observations_are_fixed_order_float64_and_finite() -> None:
    state = BC017State(_calibration())
    for observation, fields, shape in (
        (
            state.public_observation(),
            PUBLIC_OBSERVATION_FIELDS,
            PUBLIC_OBSERVATION_SHAPE,
        ),
        (
            state.hidden_observation(),
            HIDDEN_OBSERVATION_FIELDS,
            HIDDEN_OBSERVATION_SHAPE,
        ),
    ):
        assert shape == (len(fields),)
        assert observation.shape == shape
        assert observation.dtype == np.float64
        assert np.all(np.isfinite(observation))

    state.prime(_frame(0))
    state.update(_frame(1, left_qd=0.01, right_qd=-0.02))
    public = state.public_observation()
    values = dict(zip(PUBLIC_OBSERVATION_FIELDS, public, strict=True))
    assert values["physics_step"] == 1.0
    assert values["left_qd_m_s"] == pytest.approx(0.01)
    assert values["right_qd_m_s"] == pytest.approx(-0.02)
    assert values["ll_active"] == 1.0
    assert values["lr_active"] == 0.0
    assert values["calibration_formal"] == 1.0
    assert values["left_target_slot_index"] == 0.0
    assert values["right_target_slot_index"] == 0.0

    nonfinite = _state()
    nonfinite.update(replace(_frame(1), left_q_m=math.nan))
    failed_public = nonfinite.public_observation()
    failed_hidden = nonfinite.hidden_observation()
    assert np.all(np.isfinite(failed_public))
    assert np.all(np.isfinite(failed_hidden))
    public_values = dict(
        zip(PUBLIC_OBSERVATION_FIELDS, failed_public, strict=True)
    )
    hidden_values = dict(
        zip(HIDDEN_OBSERVATION_FIELDS, failed_hidden, strict=True)
    )
    for values in (public_values, hidden_values):
        assert values["terminal_reason_code"] == float(
            BC017FailureReason.LEFT_DOF_NON_FINITE
        )
        assert values["terminal_metric_code"] == float(
            BC017FailureMetric.LEFT_DOF_Q_M_FINITE
        )
