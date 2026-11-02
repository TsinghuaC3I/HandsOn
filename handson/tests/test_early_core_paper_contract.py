"""Keep the cup and plant tasks on their recorded Core paper contracts."""
import numpy as np
import pytest

from handson.sim.isaaclab import core_pitcher_env as paper
from handson.tasks import CONFIG_MAPPING
from handson.tasks.registry import TASKS


@pytest.mark.parametrize('task', ['t19_orient_cup_upright', 't50_water_plant'])
def test_task_factory_camera_and_trace_share_the_paper_runtime(task):
    env = CONFIG_MAPPING[task]().get_environment(policy_mode=True, launch_isaac=False)
    try:
        assert type(env.unwrapped) is paper.HandsOnABCIsaacLabEnv
        assert env.action_space.shape == (52,)
        assert TASKS[task].camera_environment_module == 'core_pitcher_env'
        assert TASKS[task].trace_environment_module == 'core_pitcher_base'
    finally:
        env.close()


def test_cup_requires_upright_pose_and_half_second_release():
    env = paper.HandsOnABCIsaacLabEnv('t19_orient_cup_upright', launch_isaac=False,
                                    image_obs=False, control_dt=1/30)
    try:
        cup = np.array([-.43, -.06, .921, 1., 0., 0., 0.])
        env._poses = {'a_cup': cup}
        env._right_tcp_pose = np.array([-.10, -.16, 1.07, 1., 0., 0., 0.])
        assert env._success_steps_required() == 15
        for _ in range(14):
            assert not env._compute_success()
        env._right_tcp_pose[:3] = cup[:3]
        assert not env._compute_success()
        assert env._success_counter == 0
        env._right_tcp_pose[:3] = [-.10, -.16, 1.07]
        for _ in range(14):
            assert not env._compute_success()
        assert env._compute_success()
        cup[3:] = [0., 1., 0., 0.]
        assert not env._cup_bottom_down()
    finally:
        env.close()


def test_plant_paper_reset_uses_center_clearance_before_drawing_yaw():
    # This historical layout passes the paper's center-distance gate. A later
    # yaw-dependent footprint gate rejects it; that revision is a different
    # reset distribution and must not silently replace the paper route.
    layout = {'spray': np.array([-.2211, -.0809]),
              'plant': np.array([-.3651, .1322])}
    hand = (np.array([-.10, -.16, 1.0]),)
    margin = paper.HandsOnABCIsaacLabEnv._c04_reset_layout_margin(layout, hand, hand)
    assert margin > .005


def test_plant_off_target_emission_fails_even_after_target_hits():
    env = paper.HandsOnABCIsaacLabEnv('t50_water_plant', launch_isaac=False,
                                    image_obs=False, control_dt=1/30)
    try:
        env._c04_trigger_pulled = True
        env._c04_trigger_cycle_latched = True
        env._c04_water_emitting = True
        env._c04_trigger_contact_force = lambda: .20
        env._c04_grasp_contact_force = lambda: .40
        env._physics_dof_position = lambda name: (.35, 0.)
        env._c04_spray_aim_metrics = lambda: dict(aimed=True, spray_hits_soil_or_plant=True)
        for _ in range(11):
            assert not env._compute_success()
        assert env._compute_success()
        env._c04_spray_aim_metrics = lambda: dict(aimed=False, spray_hits_soil_or_plant=False)
        assert not env._compute_success()
        assert env._compute_failure()
    finally:
        env.close()
