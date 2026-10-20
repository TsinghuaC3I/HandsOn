"""Behavioral boundaries of the paper's lid-opening and ruler tasks."""
from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from handson.sim.isaaclab.lw_native_articulation_contract import (
    G08_SPEC, G10_SPEC, GEN3_SPEC, LWNativeArticulationState,
    LWTaskStage, sample_reset,
)

TASKS = (G08_SPEC.task_name, G10_SPEC.task_name, GEN3_SPEC.task_name)


def advance(state, degrees, *, velocity=0.0, dof_available=True,
            root_translation=0.0, root_rotation=0.0):
    q = np.radians(degrees)
    return state.update(q, np.full_like(q, velocity), 0.0, 0.0,
                        dof_available, False, root_translation, root_rotation,
                        1.0 / 120.0, False, False)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("seed", (0, 1, 27, 20260823))
def test_reset_is_repeatable_and_does_not_start_successful(task, seed):
    first = sample_reset(task, seed=seed, randomize=True)
    assert first == sample_reset(task, seed=seed, randomize=True)
    state = LWNativeArticulationState(task)
    advance(state, np.degrees(first.initial_q_rad))
    assert not state.success
    assert not state.failure_reason


@pytest.mark.parametrize("task", (G08_SPEC.task_name, GEN3_SPEC.task_name))
@pytest.mark.parametrize("angle,success", [(-74.999, False), (-75.0, True), (-80, True), (-90, True), (0, False)])
def test_lid_target_is_position_only_and_inclusive(task, angle, success):
    state = LWNativeArticulationState(task)
    advance(state, [angle], velocity=0.2)
    assert state.success is success
    if success:
        assert state.stage == LWTaskStage.COMPLETE


@pytest.mark.parametrize("angles,success", [((-174.999, 180), False), ((-180, 174.999), False),
    ((-175, 175), True), ((175, -175), True), ((180, 180), True), ((-180, -180), True), ((-90, 90), False)])
def test_ruler_needs_both_straight_joints_with_either_angle_sign(angles, success):
    state = LWNativeArticulationState(G10_SPEC)
    advance(state, angles, velocity=0.2, root_translation=0.05, root_rotation=30)
    assert state.success is success
    assert not state.failure_reason


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("invalid", ("nan", "missing_dof", "joint_limit", "wrong_shape"))
def test_invalid_physics_readback_cannot_succeed(task, invalid):
    state = LWNativeArticulationState(task)
    angles = list(np.degrees(state.spec.target_q_rad))
    if invalid == "nan": angles[0] = math.nan
    if invalid == "joint_limit": angles[0] = -200.0
    if invalid == "wrong_shape": angles.append(0.0)
    advance(state, angles, dof_available=invalid != "missing_dof")
    assert not state.success
    assert state.failure_reason


@pytest.mark.parametrize("task", (G08_SPEC.task_name, GEN3_SPEC.task_name))
def test_fixed_lid_base_drift_rejects_target_then_allows_recovery(task):
    state = LWNativeArticulationState(task)
    advance(state, [-80], root_translation=0.031)
    assert not state.success
    assert not state.failure_reason
    advance(state, [-80])
    assert state.success


@pytest.mark.parametrize("task", TASKS)
def test_escape_guard_requires_continuous_duration_and_latches_failure(task):
    state = LWNativeArticulationState(task)
    for _ in range(59):
        advance(state, np.degrees(state.spec.authored_q0_rad), root_translation=1.0)
        assert not state.failure_reason
    advance(state, np.degrees(state.spec.authored_q0_rad), root_translation=1.0)
    assert state.failure_reason == "root_translation_escape"
    advance(state, np.degrees(state.spec.target_q_rad))
    assert not state.success
    assert state.failure_reason == "root_translation_escape"


def test_specs_are_immutable_and_transfer_preserves_opening_thresholds():
    with pytest.raises(FrozenInstanceError):
        G08_SPEC.task_id = "changed"
    assert GEN3_SPEC.target_enter_lower_rad == G08_SPEC.target_enter_lower_rad
    assert GEN3_SPEC.target_enter_upper_rad == G08_SPEC.target_enter_upper_rad
    assert GEN3_SPEC.dof_names == ("lid_hinge",)
