import numpy as np
import pytest

from handson.replay.control import RecordedPoseControl


class NativeLatch:
    def __init__(self):
        self._right_pose_commanded = False
        self._left_pose_commanded = False
        self.calls = []

    def step(self, command):
        self.calls.append(command)
        for side, value in command.items():
            if not np.allclose(value[:7], 0):
                setattr(self, f'_{side}_pose_commanded', True)
        return 'native result'


def test_hand_only_command_does_not_activate_stored_arm_target():
    env = NativeLatch()
    control = RecordedPoseControl(env, [0, 0])
    action = np.arange(52, dtype=float)/100
    original = action.copy()
    assert control.step(action, [0, 0]) == 'native result'
    assert not env._right_pose_commanded and not env._left_pose_commanded
    assert np.array_equal(control.last_command['right'][:7], np.zeros(7))
    assert np.array_equal(control.last_command['right'][7:], action[6:26])
    assert np.array_equal(action, original)
    control.step(action, [1, 0])
    assert env._right_pose_commanded and not env._left_pose_commanded
    assert np.array_equal(control.last_command['right'][:3], action[:3])


def test_clear_or_foreign_controller_change_rejected_before_step():
    env = NativeLatch()
    control = RecordedPoseControl(env, [0, 0])
    control.step(np.zeros(52), [1, 0])
    with pytest.raises(ValueError, match='cannot clear'):
        control.step(np.zeros(52), [0, 0])
    assert len(env.calls) == 1
    env._left_pose_commanded = True
    with pytest.raises(ValueError, match='differs'):
        control.step(np.zeros(52), [1, 1])
    assert len(env.calls) == 1


@pytest.mark.parametrize('flags', [[0], [0, 1, 0], [0, 2], [0, float('nan')]])
def test_invalid_flags_rejected_before_step(flags):
    env = NativeLatch()
    control = RecordedPoseControl(env, [0, 0])
    with pytest.raises(ValueError, match='binary'):
        control.step(np.zeros(52), flags)
    assert not env.calls
