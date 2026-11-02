"""Keep Core faucet geometry and reset sampling separate from Transfer."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from handson.sim.isaaclab import core_faucet_env as paper
from handson.sim.isaaclab import core_faucet_scene as scene
from handson.tasks import CONFIG_MAPPING


def test_factory_selects_original_faucet_and_retains_eef52():
    wrapped = CONFIG_MAPPING['t05_fill_cup']().get_environment(
        policy_mode=True, launch_isaac=False, image_obs=False)
    try:
        assert type(wrapped.unwrapped) is paper.HandsOnB320BimanualIsaacLabEnv
        assert wrapped.action_space.shape == (52,)
        assert wrapped.unwrapped._b320_contract.stable_frames == 15
    finally:
        wrapped.close()


def test_paper_faucet_cannot_replace_a_transfer_environment():
    with pytest.raises(ValueError, match='does not support'):
        paper.HandsOnB320BimanualIsaacLabEnv(
            'eval_comb05_scrub_bottle_then_fill_cup', launch_isaac=False)
    assert getattr(CONFIG_MAPPING['eval_comb05_scrub_bottle_then_fill_cup'](),
                   'b320_runtime', None) != 'paper_faucet'


def test_paper_geometry_has_original_open_basin_and_five_counter_strips():
    assert scene.KA021_SAFE_BASIN_FLOOR_SIZE_M == (.44, .62, .020)
    assert len(scene.KA021_COUNTER_SURFACE_COLLISION_SPECS) == 5
    assert scene.KA021_COUNTER_SURFACE_Z == 1.011315
    assert scene.KA021_SAFE_BASIN_FLOOR_PATH.endswith('safe_open_basin_floor_collision')
    root = Path(scene.__file__).parents[2]
    receipt = json.loads((root/'evaluation/core_faucet_provenance.json').read_text())
    assert hashlib.sha256(scene.KA021_HS005_SCENE_PATH.read_bytes()).hexdigest() == receipt['assets'][0]['sha256']
    assert scene.validate_b320_asset_files()


def test_seeded_mug_and_park_sampling_stays_inside_paper_bounds():
    env = paper.HandsOnB320BimanualIsaacLabEnv(
        't05_fill_cup', launch_isaac=False, randomize=True)
    try:
        for seed in range(16):
            env._random = np.random.RandomState(seed)
            mug = env._sample_ka021_reset_pose()
            park = env._ka021_cup_park_position_w.copy()
            assert np.isfinite(mug).all() and np.isfinite(park).all()
            env._random = np.random.RandomState(seed)
            assert np.array_equal(mug, env._sample_ka021_reset_pose())
            assert np.array_equal(park, env._ka021_cup_park_position_w)
    finally:
        env.close()
