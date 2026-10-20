"""Integration checks for the public factory, native executor and model boundary."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import gymnasium as gym
import numpy as np
import pytest

from handson.evaluation.history import ImageHistory
from handson.evaluation.policy import PolicyClient
from handson.evaluation import cameras
from handson.interfaces import Policy, RENDER_VIEWS
from handson.policies.profiles import PROFILES, noise_seed
from handson.policies.server import predict_request
from handson.tasks.config import TaskConfigBase
from handson.tasks.language import bank
from handson.tasks.registry import TASKS, ASSETS


class NativeEnv(gym.Env):
    """Measured palms deliberately differ from native observation/command poses."""
    _control_dt, _physics_dt, _n_substeps = 1 / 30, 1 / 120, 4
    _sim = None
    hand_type = 'wuji1'

    def __init__(self):
        self._robots, self._hand_joint_ids = {}, {}
        for i, side in enumerate(('right', 'left')):
            pose = np.zeros((1, 1, 13))
            pose[0, 0, :4] = [i + 1, 0, 0, 1]
            self._robots[side] = SimpleNamespace(
                body_names=[side + '_palm_link'],
                joint_names=[f'{side}_finger{f}_joint{j}' for f in range(1, 6) for j in range(1, 5)],
                data=SimpleNamespace(body_state_w=pose, joint_pos=np.full((1, 20), i * .1)))
            self._hand_joint_ids[side] = list(range(20))
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (54,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(dict(
            state=gym.spaces.Box(-np.inf, np.inf, (120,), dtype=np.float32),
            images=gym.spaces.Dict({k: gym.spaces.Box(0, 255, (2, 3, 3), dtype=np.uint8)
                                   for k in ('ego', 'wrist_left', 'wrist_right')})))
        self.closed = False

    def raw_observation(self):
        return dict(state=np.full(120, 999, np.float32),
                    images={k: np.full((2, 3, 3), i, np.uint8)
                            for i, k in enumerate(('ego', 'wrist_left', 'wrist_right'))})

    def reset(self, **kwargs):
        return self.raw_observation(), {'native_reset': True}

    def step(self, command):
        self.received = command
        return self.raw_observation(), .25, True, False, {'succeed': True}

    def close(self):
        self.closed = True


class Config(TaskConfigBase):
    task_name = 't24_place_ball_in_bowl'

    def get_native_environment(self, **kwargs):
        self.kwargs = kwargs
        self.raw = NativeEnv()
        return gym.Wrapper(self.raw)

    def process_demos(self, demo):
        return demo


def test_public_factory_and_rollout_share_physical_commands():
    config = Config()
    env = config.get_environment(policy_mode=True)
    assert env.action_space.shape == (52,)
    assert config.kwargs['hand_type'] == 'wuji1'
    obs, info = env.reset(seed=8)
    assert env.observation_space.contains(obs)
    assert set(obs) == {'state', 'front', 'wrist_left', 'wrist_right'}
    assert obs['state'][0] == 1 and obs['state'][26] == 2
    assert info == {'native_reset': True}
    action = np.linspace(-.2, .3, 52, dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    assert reward == .25 and terminated and not truncated and info['succeed']
    np.testing.assert_array_equal(observation['state'], obs['state'])
    received = config.raw.received
    assert set(received) == {'right', 'left'}
    np.testing.assert_array_equal(received['right'][:3], action[:3])
    np.testing.assert_array_equal(received['left'][7:], action[32:])
    assert np.isclose(np.linalg.norm(received['right'][3:7]), 1)
    from handson.eef52 import EEF52Control
    second = NativeEnv()
    EEF52Control(second).step(action)
    for side in received:
        np.testing.assert_array_equal(received[side], second.received[side])
    with pytest.raises(ValueError):
        env.step(np.zeros(54))
    env.close()
    assert config.raw.closed


def test_native_collection_entry_and_policy_guards():
    config = Config()
    env = config.get_environment(policy_mode=False)
    assert env.action_space.shape == (54,)
    assert config.kwargs['policy_mode'] is False
    with pytest.raises(ValueError, match='wuji1'):
        config.get_environment(policy_mode=True, hand_type='allegro')
    with pytest.raises(ValueError, match='30 Hz'):
        config.get_environment(policy_mode=True, control_dt=1 / 60)


def images(value):
    return {k: np.full((2, 3, 3), value, np.uint8) for k in RENDER_VIEWS}


def test_interleaved_history_and_transports_do_not_mutate_globals(tmp_path):
    predict, render, opener = PolicyClient.predict, cameras.render_probe, urllib.request._opener
    a, b = ImageHistory(6, 10, 30, 10), ImageHistory(6, 10, 30, 10)
    a.views(images(0), 0)
    b.views(images(100), 0)
    for step in (10, 20, 30):
        a.record(step, images(step))
        b.record(step, images(step + 100))
    np.testing.assert_array_equal(a.views(images(30), 1)['main'][:, 0, 0, 0], [0, 0, 0, 10, 20, 30])
    np.testing.assert_array_equal(b.views(images(130), 1)['main'][:, 0, 0, 0], [100, 100, 100, 110, 120, 130])
    a.views(images(7), 0)  # Reset one history without disturbing the other.
    assert b.views(images(130), 1)['main'][-1, 0, 0, 0] == 130
    for kind in ('http_physical_eef52', 'openpi_websocket'):
        PolicyClient(dict(transport=kind), tmp_path)
    hold = PolicyClient(dict(transport='hold', action_horizon=16), tmp_path)
    assert hold.start()['diagnostic_only']
    assert hold.predict(np.arange(52), images(0), 0, 0).shape == (16, 52)
    assert PolicyClient.predict is predict and cameras.render_probe is render
    assert urllib.request._opener is opener


def test_catalogs_and_all_task_model_requests():
    assert len(TASKS) == 66 and sum(s.split == 'core' for s in TASKS.values()) == 50
    from handson.tasks import CONFIG_MAPPING
    assert set(CONFIG_MAPPING) == set(TASKS) == set(bank()['tasks'])
    assert set(json.loads((ASSETS / 'goals.json').read_text())) == set(TASKS)
    accepted = rejected = 0
    for task in TASKS:
        prompt = bank()['tasks'][task]['goals'][0]['canonical']
        for model, profile in PROFILES.items():
            class FixturePolicy:
                def predict(self, state, images, prompt, seed):
                    return np.zeros((profile.model_horizon, 52), np.float32)
            fixture = FixturePolicy()
            assert isinstance(fixture, Policy)
            shape = (6, 2, 3, 3) if profile.history_frames == 6 else (2, 3, 3)
            packet = io.BytesIO()
            np.savez(packet, state=np.zeros(52, np.float32), prompt=np.array(prompt), task=np.array(task),
                     seed=np.array(8), request=np.array(0),
                     **{k: np.zeros(shape, np.uint8) for k in ('front', 'wrist_left', 'wrist_right')})
            if model == 'dp' and TASKS[task].split == 'transfer':
                with pytest.raises(ValueError, match='no implicit Transfer-16'):
                    predict_request(fixture, profile, model, packet.getvalue())
                rejected += 1
            else:
                result = predict_request(fixture, profile, model, packet.getvalue())
                assert np.shape(result['actions']) == (profile.action_horizon, 52)
                accepted += 1
    assert (accepted, rejected) == (578, 16)


def test_all_original_noise_streams_are_preserved():
    prefixes = dict(pi05='A-pi05-120000', dm05='dm05-86000', hy='hy-125000',
                    lingbot='lingbot-151000', openwam='openwam:a01_fruit_to_bucket',
                    fastwam='fastwam', motus='motus', spatial='spatial-84000', dp='dp:a01_fruit_to_bucket')
    for model, prefix in prefixes.items():
        for seed, request in [(0, 0), (42, 3), (881, 80)]:
            expected = int.from_bytes(hashlib.sha256(f'{prefix}:{seed}:{request}'.encode()).digest()[:8], 'little')
            if model in {'hy', 'dm05', 'lingbot'}:
                expected &= 2**63 - 1
            elif model in {'dp', 'openwam', 'fastwam', 'motus'}:
                expected %= 2**63 - 1
            assert noise_seed(model, seed, request, 't26_place_fruit_in_bucket') == expected
