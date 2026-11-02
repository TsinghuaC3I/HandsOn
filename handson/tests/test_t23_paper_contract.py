"""The Core photograph predicate uses measured contact, not proximity."""
import numpy as np

from handson.sim.isaaclab import core_photograph_env as paper
from handson.tasks import CONFIG_MAPPING
from handson.tasks.registry import TASKS


def test_photograph_factory_camera_and_trace_use_paper_runtime():
    wrapped = CONFIG_MAPPING['t23_photograph_logo']().get_environment(policy_mode=True, launch_isaac=False)
    try:
        assert type(wrapped.unwrapped) is paper.HandsOnABCIsaacLabEnv
        assert wrapped.action_space.shape == (52,)
        assert TASKS['t23_photograph_logo'].camera_environment_module == 'core_photograph_env'
        assert TASKS['t23_photograph_logo'].trace_environment_module == 'core_photograph_base'
    finally:
        wrapped.close()


def test_shutter_requires_force_contact_with_three_update_grace():
    env = paper.HandsOnABCIsaacLabEnv('t23_photograph_logo', launch_isaac=False,
                                    image_obs=False, control_dt=1/30)
    try:
        forward = np.array([.03, 0., .12])
        forward /= np.linalg.norm(forward)
        camera = np.r_[np.array([.29, .18, 1.1006]) - .3*forward, [1., 0., 0., 0.]]
        env._poses = {'camera': camera, 'photograph_camera': camera,
                      'table': np.array([0., 0., 0., 1., 0., 0., 0.])}
        travel, force = [.0006], [0.]
        env._physics_dof_position = lambda name: (travel[0], 0.)
        env._c06_right_hand_shutter_contact_force = lambda: force[0]
        # A later collection implementation accepts this proximity fallback.
        # It must have no effect on the restored Core paper predicate.
        env._c06_right_hand_shutter_proximity = lambda: (True, 0.)
        assert not env._compute_success_now()
        assert env._last_c06_success_metrics['c06_camera_aimed']
        assert env._last_c06_success_metrics['c06_shutter_pressed']
        force[0] = .03
        assert env._compute_success_now()
        force[0] = 0.
        assert env._compute_success_now()
        assert env._compute_success_now()
        assert not env._compute_success_now()
        force[0] = .03
        travel[0] = 0.
        assert not env._compute_success_now()
        travel[0] = .0006
        camera[0] += .5
        assert not env._compute_success_now()
    finally:
        env.close()
