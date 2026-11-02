"""Paper idle-arm updates must run only in the source-bound Core routes."""
from types import SimpleNamespace
import numpy as np
import pytest
from handson.sim.isaaclab.bimanual_assembly_env import BimanualAssemblyIsaacLabEnv


@pytest.mark.parametrize('task,backend,expected', [
    ('t11_line_up_blocks', 'physx', ['right', 'left']),
    ('t29_place_wine_glass', 'physx', ['right', 'left']),
    ('t33_reposition_marker_grasp', 'physx', ['right', 'left']),
    ('t34_roll_ball_through_gate', 'physx', ['right', 'left']),
    ('t47_trigger_drill_on_block', 'physx', ['right', 'left']),
    ('t29_place_wine_glass', 'newton', ['right']),
    ('eval_gen03_orient_bottle_upright', 'physx', ['right']),
])
def test_idle_arm_uses_the_source_bound_controller(task, backend, expected):
    torch = pytest.importorskip('torch')
    class Robot:
        def __init__(self):
            self.data = SimpleNamespace(default_joint_pos=torch.zeros((1, 7)))
            self.writes = 0
        def set_joint_position_target(self, target):
            self.writes += 1
    env = BimanualAssemblyIsaacLabEnv(launch_isaac=False, image_obs=False)
    env.task_name = task
    env._spec = SimpleNamespace(bimanual=False)
    env._physics_backend = backend
    env._torch = torch
    env._robots = {side: Robot() for side in ('right', 'left')}
    env._control_robot = lambda side: env._robots[side]
    env._joint_targets = {side: torch.zeros((1, 7)) for side in env._robots}
    env._arm_joint_ids = {side: list(range(7)) for side in env._robots}
    env._hand_joint_ids = {side: [] for side in env._robots}
    env._right_hand = env._left_hand = np.zeros(0)
    env._last_control_log_time = float('inf')
    updates = []
    def idle_target(side, *_):
        updates.append(side)
        return None, 'idle'
    env._compute_arm_joint_target = idle_target
    env._apply_joint_targets()
    assert updates == expected
    assert all(robot.writes == 1 for robot in env._robots.values())
