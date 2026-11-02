"""The two paper reset variants differ in which robot bodies they screen."""
import numpy as np
import pytest

from handson.sim.isaaclab.core_largest_ball_env import CoreLargestBallEnv
from handson.tasks import CONFIG_MAPPING
from handson.tasks.t27_place_largest_fitting_ball import semantics as geometry


def test_factory_defaults_to_multitask_paper_hand_clearance():
    wrapped = CONFIG_MAPPING["t27_place_largest_fitting_ball"]().get_environment(
        policy_mode=True, launch_isaac=False, image_obs=False,
    )
    try:
        assert type(wrapped.unwrapped) is CoreLargestBallEnv
        assert wrapped.unwrapped.reset_clearance_scope == "hand_bodies"
        assert wrapped.action_space.shape == (52,)
    finally:
        wrapped.close()


def test_arm_link_near_a_cup_changes_only_the_full_arm_variant():
    env = CoreLargestBallEnv.__new__(CoreLargestBallEnv)
    env.reset_clearance_scope = "hand_bodies"
    env._ycb02_cup_order = tuple(geometry.CUP_NAMES)
    env._ycb02_candidate_order = tuple(geometry.BALL_NAMES)
    env.delta_h = np.zeros(1)
    env.randomize = False
    env._random = np.random.RandomState(17)
    env._a06_hand_points = lambda side: (np.array([10., 10., 10.]),)
    hand_layout = env._ycb02_reset_xy_positions()
    cup = geometry.CUP_NAMES[0]
    arm_point = np.r_[hand_layout[cup], .89 - geometry.CUP_SUPPORT_LOCAL_MIN_Z_M[cup]]
    env._reset_robot_body_points = lambda side: (*env._a06_hand_points(side), arm_point)
    # An elbow near a cup is absent from the hand-only clearance cloud.
    repeated = env._ycb02_reset_xy_positions()
    assert all(np.array_equal(hand_layout[k], repeated[k]) for k in hand_layout)
    env.reset_clearance_scope = "arm_bodies"
    try:
        arm_layout = env._ycb02_reset_xy_positions()
    except RuntimeError as exc:
        assert "could not place" in str(exc)
    else:
        assert any(not np.array_equal(hand_layout[k], arm_layout[k]) for k in hand_layout)
        assert env._ycb02_reset_layout_diagnostics["clearance_scope"] == "arm_bodies"


def test_unknown_scope_and_other_tasks_fail_before_simulator_start():
    with pytest.raises(ValueError, match="reset_clearance_scope"):
        CoreLargestBallEnv(reset_clearance_scope="automatic")
    with pytest.raises(ValueError, match="only t27"):
        CoreLargestBallEnv("t34_roll_ball_through_gate")
