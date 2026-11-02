"""Reset arm positions from the d41468eb paper/collection source."""
from types import SimpleNamespace

import pytest

from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv


@pytest.mark.parametrize('side,expected', [
    ('right', [-1.5707963, 1.5707963, 2.0207963, 1.5707963, -1.5707963, 0., 0.]),
    ('left', [-0.441741, 0.721729, -0.239304, 1.982058, 0.346778, -0.925025, -0.925025]),
])
def test_wine_task_parks_inactive_left_arm_at_paper_height(side, expected):
    env = SimpleNamespace(task_name='t29_place_wine_glass')
    joints = {'right_finger1_joint1': .123, 'left_finger1_joint1': .456}
    config = SimpleNamespace(init_state=SimpleNamespace(joint_pos=joints))
    assert HandsOnABCIsaacLabEnv._apply_raised_hand_home(env, config, side)
    assert [joints[f'joint{i}_{side}(_1)?'] for i in range(1, 8)] == expected
    assert joints['right_finger1_joint1'] == .123
    assert joints['left_finger1_joint1'] == .456
