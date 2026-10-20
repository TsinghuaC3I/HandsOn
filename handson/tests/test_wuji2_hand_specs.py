from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from handson.sim.envs.hand_specs import (
    WUJI_NATIVE_HAND_DIM,
    _wuji_actuator_names,
    _wuji_joint_names,
    _wuji_sensor_names,
    hand_action_dim,
    native_hand_joint_names,
    normalize_hand_type,
    policy_side_to_native_hand_values,
)
from handson.tasks.sim_teleop import (
    _is_rokae_wuji_teleop,
    _is_wuji_hand_env,
)


def test_wuji2_aliases_and_native_dimension() -> None:
    assert normalize_hand_type("wuji2") == "wuji2"
    assert normalize_hand_type("wiji2") == "wuji2"
    assert normalize_hand_type("wuji_hand_2") == "wuji2"
    assert hand_action_dim("wuji2") == WUJI_NATIVE_HAND_DIM == 20

    env = SimpleNamespace(arm_type="rokae", hand_type="wuji2")
    assert _is_wuji_hand_env(env)
    assert _is_rokae_wuji_teleop(env)


def test_wuji2_uses_official_firmware_names_and_finite_mapping() -> None:
    right_joints = _wuji_joint_names("wuji2", "right")
    left_actuators = _wuji_actuator_names("wuji2", "left")
    assert len(right_joints) == len(left_actuators) == 20
    assert right_joints[:4] == [
        "r_thumb_cmc_flex",
        "r_thumb_cmc_abd",
        "r_thumb_mcp",
        "r_thumb_ip",
    ]
    assert native_hand_joint_names("wuji2", "right") == tuple(right_joints)
    assert left_actuators[:5] == ["l_THJ0", "l_THJ1", "l_THJ2", "l_THJ3", "l_FFJ0"]
    assert _wuji_sensor_names("wuji2", "right")[0] == (
        "wuji2_right/r_thumb_cmc_flex_pos"
    )

    native = policy_side_to_native_hand_values(
        "wuji2",
        np.zeros(16, dtype=np.float64),
        side="right",
    )
    assert native.shape == (20,)
    assert np.all(np.isfinite(native))


def test_wuji1_native_joint_order_remains_replay_compatible() -> None:
    for side in ("right", "left"):
        expected = tuple(
            f"{side}_finger{finger}_joint{joint}"
            for finger in range(1, 6)
            for joint in range(1, 5)
        )
        assert native_hand_joint_names("wuji1", side) == expected
