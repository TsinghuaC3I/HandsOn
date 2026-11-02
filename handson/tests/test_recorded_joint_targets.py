"""The supplemental motor replay must obey real substeps and fail closed."""
from types import SimpleNamespace

import numpy as np
import pytest

from handson.replay.control import RecordedJointTargetControl


class Native:
    _n_substeps = 4

    def __init__(self, indices=(0, 1, 2, 3), fail=False):
        self.indices, self.fail = indices, fail
        self.targets = np.zeros((2, 27))
        self.history = []
        self._physx_ground_truth_trace = SimpleNamespace(
            apply_robot_joint_targets=self.apply,
            snapshot=lambda: {'physx_robot_joint_target': self.targets.copy()})

    def apply(self, packet, schema):
        mask = packet['physx_robot_joint_target_valid'].astype(bool)
        self.targets[mask] = packet['physx_robot_joint_target'][mask]
        return int(mask.sum())

    def _apply_action(self, action):
        self.action = action

    def step_with_current_joint_targets(self):
        for index in self.indices:
            self._handson_pre_physics_substep_callback(index, self._n_substeps)
            self.history.append(self.targets.copy())
            if self.fail:
                raise RuntimeError('physics failure')
        return 'native result'


def test_motor_targets_apply_at_each_substep_and_leave_no_callback():
    native = Native()
    values = np.arange(216).reshape(4, 2, 27) * .001
    control = RecordedJointTargetControl(native, {})
    assert control.step(np.zeros(52), values, np.ones_like(values)) == 'native result'
    np.testing.assert_array_equal(native.history, values)
    assert control.applied_substeps == 4 and control.max_target_error == 0
    assert native._handson_pre_physics_substep_callback is None


@pytest.mark.parametrize('indices', [(0, 2, 1, 3), (0, 1, 2)])
def test_missing_or_reordered_substeps_are_errors(indices):
    native = Native(indices)
    control = RecordedJointTargetControl(native, {})
    with pytest.raises(ValueError, match='substep'):
        control.step(np.zeros(52), np.zeros((4, 2, 27)), np.ones((4, 2, 27)))
    assert native._handson_pre_physics_substep_callback is None


def test_physics_exception_cannot_leave_a_replay_callback_installed():
    native = Native(fail=True)
    control = RecordedJointTargetControl(native, {})
    with pytest.raises(RuntimeError, match='physics failure'):
        control.step(np.zeros(52), np.zeros((4, 2, 27)), np.ones((4, 2, 27)))
    assert native._handson_pre_physics_substep_callback is None


def test_existing_callback_is_preserved_and_not_run():
    native = Native()
    owner = object()
    native._handson_pre_physics_substep_callback = owner
    with pytest.raises(ValueError, match='already installed'):
        RecordedJointTargetControl(native, {}).step(np.zeros(52), np.zeros((4, 2, 27)), np.ones((4, 2, 27)))
    assert native._handson_pre_physics_substep_callback is owner
    assert not hasattr(native, 'action')
