"""Guard source-bound Core reset geometry against later shared-scene changes."""
import hashlib
from types import SimpleNamespace

from handson.sim.isaaclab import core_scrub_bottle_env as paper
from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
from handson.tasks import CONFIG_MAPPING


def test_scrub_factory_preserves_paper_clock_and_default_asset():
    wrapped = CONFIG_MAPPING['t37_scrub_bottle']().get_environment(
        policy_mode=True, launch_isaac=False, image_obs=False)
    try:
        env = wrapped.unwrapped
        assert type(env) is paper.CoreScrubBottleEnv
        assert env._physics_dt == 1/240
        assert env._control_dt == 1/30
        assert wrapped.action_space.shape == (52,)
    finally:
        wrapped.close()
    layer = paper._HS005_ASSET_LAYER
    assert hashlib.sha256(layer.read_bytes()).hexdigest() == (
        '5196ac1c18bdf11d533a7833dfe9dfd3505ca6f4be05302c118f9089c09454c2')
    assert paper._HS005_ASSET_LAYER_BY_SCENE[
        paper.HS005_MODERN_APARTMENT_SINK_SCENE] == layer


def test_soap_keeps_nominal_robot_home_from_source_97697a8e():
    env = SimpleNamespace(task_name='t32_press_soap_pump')
    for side in ('left', 'right'):
        joints = {f'joint{i}_{side}(_1)?': i/10 for i in range(1, 8)}
        before = dict(joints)
        config = SimpleNamespace(init_state=SimpleNamespace(joint_pos=joints))
        assert not HandsOnABCIsaacLabEnv._apply_raised_hand_home(env, config, side)
        assert joints == before
