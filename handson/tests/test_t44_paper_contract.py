"""Exercise the frozen paper stack through the public task route."""
import numpy as np

from handson.sim.isaaclab import core_stack_env as paper
from handson.tasks import CONFIG_MAPPING
from handson.tasks.registry import TASKS


def test_stack_factory_and_native_dependencies_match_paper_version():
    wrapped = CONFIG_MAPPING['t44_stack_blocks']().get_environment(policy_mode=True, launch_isaac=False)
    try:
        assert isinstance(wrapped.unwrapped, paper.HandsOnABCIsaacLabEnv)
        assert TASKS['t44_stack_blocks'].camera_environment_module == 'core_stack_env'
        assert TASKS['t44_stack_blocks'].trace_environment_module == 'core_pitcher_base'
        assert wrapped.action_space.shape == (52,)
    finally:
        wrapped.close()


def test_supported_stack_requires_contact_release_and_rejects_missing_support():
    # Authored collision half heights, independent of the success implementation.
    heights = {'036_wood_block': .042689, 'BuildingBlock017': .014925,
               'BuildingBlock018': .015, 'BuildingBlock019': .015}
    env = paper.HandsOnABCIsaacLabEnv('t44_stack_blocks', launch_isaac=False, control_dt=1/30)
    force = [0.02]

    class Contact:
        sensor_count = filter_count = 1

        def get_contact_force_matrix(self, dt):
            return np.array([[[force[0], 0., 0.]]])

    try:
        z = .89
        env._poses = {}
        for name, half_height in heights.items():
            z += half_height
            env._poses[name] = np.array([-.30, -.10, z, 1., 0., 0., 0.])
            z += half_height
        env._right_tcp_pose = np.array([.30, .30, 1.20, 1., 0., 0., 0.])
        env._physics_body_velocity = lambda name: np.zeros(6)
        env._sim_dt = 1/120
        env._b04_block_contact_views = {name: Contact() for name in heights}
        assert not env._block_stack_stable()
        assert env._task_info()['stack_geometry_valid']
        force[0] = 0.0
        assert env._block_stack_stable()
        env._poses['BuildingBlock019'][0] += .2
        assert not env._block_stack_stable()
    finally:
        env.close()
