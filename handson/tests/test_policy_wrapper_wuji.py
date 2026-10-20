from __future__ import annotations

import gymnasium as gym
import numpy as np

from handson.sim.envs.hand_specs import policy_side_to_native_hand_values
from handson.tasks.policy_wrappers import DualArmPolicyWrapper


class _NativeDualWujiEnv(gym.Env):
    hand_type = "wuji1"
    _hand_action_dim = 20

    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(54,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(1,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def test_46d_policy_hands_expand_to_native_dual_wuji_commands() -> None:
    native = _NativeDualWujiEnv()
    native.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(46,), dtype=np.float32)
    wrapper = DualArmPolicyWrapper(native)
    assert wrapper.action_space.shape == (46,)

    action = np.linspace(-0.1, 0.9, 46, dtype=np.float64)
    parsed = wrapper.action(action)

    assert parsed["right"].shape == (27,)
    assert parsed["left"].shape == (27,)
    np.testing.assert_allclose(parsed["right"][:7], action[:7])
    np.testing.assert_allclose(parsed["left"][:7], action[7:14])
    np.testing.assert_allclose(
        parsed["right"][7:],
        policy_side_to_native_hand_values(
            "wuji1", action[14:30], side="right", dual_arm=True
        ),
    )
    np.testing.assert_allclose(
        parsed["left"][7:],
        policy_side_to_native_hand_values(
            "wuji1", action[30:46], side="left", dual_arm=True
        ),
    )


def test_native_54d_wuji_commands_preserve_all_hand_joints():
    wrapper = DualArmPolicyWrapper(_NativeDualWujiEnv())
    assert wrapper.action_space.shape == (54,)
    action = np.linspace(-0.1, 0.9, 54)
    parsed = wrapper.action(action)
    np.testing.assert_array_equal(parsed["right"], np.r_[action[:7], action[14:34]])
    np.testing.assert_array_equal(parsed["left"], np.r_[action[7:14], action[34:54]])
