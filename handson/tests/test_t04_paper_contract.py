"""The paper pitcher contract must not inherit later collection changes."""
import numpy as np

from handson.sim.isaaclab import core_pitcher_env as paper
from handson.tasks import CONFIG_MAPPING
from handson.tasks.registry import TASKS


def make_env():
    return paper.HandsOnABCIsaacLabEnv('t04_cover_pitcher', launch_isaac=False, control_dt=1/30)


def test_public_pitcher_routes_camera_and_trace_with_the_paper_environment():
    wrapped = CONFIG_MAPPING['t04_cover_pitcher']().get_environment(policy_mode=True, launch_isaac=False)
    try:
        assert isinstance(wrapped.unwrapped, paper.HandsOnABCIsaacLabEnv)
        assert wrapped.action_space.shape == (52,)
        assert TASKS['t04_cover_pitcher'].camera_environment_module == 'core_pitcher_env'
        assert TASKS['t04_cover_pitcher'].trace_environment_module == 'core_pitcher_base'
    finally:
        wrapped.close()


def test_paper_release_requires_close_open_near_goal_and_clears_on_regrasp():
    env = make_env()
    hand = [0.0]
    zone = [False]
    changes = []
    env._b01_hand_command_max = lambda: hand[0]
    env._b01_lid_in_release_zone = lambda: zone[0]

    def toggle(enabled):
        changes.append(enabled)
        env._b01_release_assist_active = enabled
        return True

    env._set_b01_release_assist = toggle
    try:
        env._update_b01_release_assist()
        assert changes == []
        hand[0] = paper._B01_CLOSED_HAND_MIN_JOINT
        env._update_b01_release_assist()
        hand[0] = 0.0
        env._update_b01_release_assist()
        assert changes == []
        zone[0] = True
        env._update_b01_release_assist()
        assert changes == [True]
        hand[0] = paper._B01_CLOSED_HAND_MIN_JOINT
        env._update_b01_release_assist()
        assert changes == [True, False]
    finally:
        env.close()


def test_paper_success_requires_continuous_half_second_release_after_seating():
    env = make_env()
    base = np.array([0., 0., 1., 1., 0., 0., 0.])
    lid = base.copy()
    lid[2] += paper._B01_LID_INSERTED_GOAL_LOCAL_Z
    clear = [True]
    env._physics_body_velocity = lambda name: np.zeros(6)
    env._b01_hand_lid_clearance = lambda pose: (0.1 if clear[0] else 0.0, clear[0])
    env._lid_start_pose = lid.copy()
    env._lid_start_pose[0] -= 0.15
    env._b01_hand_lid_clear_steps = 0
    # Later collection code introduced this disqualification. It was absent
    # from the paper's native predicate, so it must not leak into this version.
    env._b01_release_assist_ever_applied = True

    def tick():
        return env._b01_lid_insert_metrics(lid, base, update_hand_clear_hold=True)

    try:
        assert env._b01_hand_lid_clear_steps_required() == 15
        for _ in range(14):
            assert not tick()['b01_success_now']
        clear[0] = False
        assert not tick()['b01_success_now']
        assert env._b01_hand_lid_clear_steps == 0
        clear[0] = True
        for _ in range(14):
            assert not tick()['b01_success_now']
        assert tick()['b01_success_now']
        lid[0] += 0.2
        assert not tick()['b01_success_now']
    finally:
        env.close()
