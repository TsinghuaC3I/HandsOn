from __future__ import annotations

import gymnasium as gym
import numpy as np

from handson.data.recording_fields import (
    declared_recording_field_names,
    resolve_environment_recording_schema,
    stack_transition_numeric_observation_fields,
)
from handson.tasks.obs_adapters import HandsOnObsAdapter


class _TelemetryEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "pose": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(2,),
                            dtype=np.float64,
                        )
                    }
                ),
                "c01_contact_wrench": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(8, 12),
                    dtype=np.float32,
                ),
                "c01_contact_valid": gym.spaces.Box(
                    0,
                    1,
                    shape=(8,),
                    dtype=np.uint8,
                ),
            }
        )


class _PlanRecordingEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "pose": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(7,),
                            dtype=np.float64,
                        )
                    }
                ),
                "plan_arm_qpos_raw": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(7,),
                    dtype=np.float64,
                ),
                "plan_arm_qpos": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(7,),
                    dtype=np.float64,
                ),
                "plan_arm_success": gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "plan_arm_error": gym.spaces.Box(
                    0.0,
                    np.inf,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "plan_hand_qpos": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(20,),
                    dtype=np.float64,
                ),
            }
        )

    def get_recording_schema(self) -> dict:
        return {
            "name": "handson.plan_arm",
            "version": 1,
            "fields": {
                "plan_arm_qpos_raw": {
                    "shape": [7],
                    "dtype": "float64",
                    "arm_side": "right",
                },
                "plan_arm_qpos": {
                    "shape": [7],
                    "dtype": "float64",
                    "arm_side": "right",
                },
                "plan_arm_success": {
                    "shape": [1],
                    "dtype": "float64",
                },
                "plan_arm_error": {
                    "shape": [1],
                    "dtype": "float64",
                },
                "plan_hand_qpos": {
                    "shape": [20],
                    "dtype": "float64",
                },
            },
        }


class _E01RecordingEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "pose": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(2,),
                            dtype=np.float64,
                        )
                    }
                ),
                "e01_hand_qpos": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(40,),
                    dtype=np.float64,
                ),
            }
        )

    def get_recording_schema(self) -> dict:
        return {
            "name": "handson.e01_actual_hand_state",
            "version": 1,
            "fields": {
                "e01_hand_qpos": {
                    "shape": [40],
                    "dtype": "float64",
                }
            },
        }


class _PhysxRecordingEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "pose": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(2,),
                            dtype=np.float64,
                        )
                    }
                ),
                "physx_tcp_state": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(2, 13),
                    dtype=np.float64,
                ),
                "physx_tcp_valid": gym.spaces.Box(
                    0,
                    1,
                    shape=(2,),
                    dtype=np.uint8,
                ),
            }
        )


class _BC017RecordingEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "bc017_drawer_state": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(2,),
                            dtype=np.float64,
                        ),
                        "bc017_hidden_state": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(3,),
                            dtype=np.float64,
                        ),
                    }
                ),
                "bc017_drawer_state": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(2,),
                    dtype=np.float64,
                ),
                "bc017_hidden_state": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(3,),
                    dtype=np.float64,
                ),
            }
        )


class _OrderedBoundedStateEnv(gym.Env):
    """Raw state whose declared order differs from Gymnasium's sort order."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(3,),
                            dtype=np.float64,
                        ),
                        "active_obstacle_mask": gym.spaces.Box(
                            0.0,
                            1.0,
                            shape=(2,),
                            dtype=np.float64,
                        ),
                    }
                )
            }
        )


def test_physx_ground_truth_is_forwarded_outside_policy_state() -> None:
    adapter = HandsOnObsAdapter(_PhysxRecordingEnv())
    tcp_state = np.arange(26, dtype=np.float64).reshape(2, 13)
    observation = adapter.observation(
        {
            "state": {"pose": np.asarray([1.0, 2.0])},
            "physx_tcp_state": tcp_state,
            "physx_tcp_valid": np.ones(2, dtype=np.uint8),
            "physx_debug": np.ones((3,), dtype=np.float64),
        }
    )

    assert observation["state"].shape == (2,)
    np.testing.assert_array_equal(observation["physx_tcp_state"], tcp_state)
    assert observation["physx_tcp_valid"].dtype == np.uint8
    assert "physx_debug" not in observation
    assert adapter.observation_space.contains(observation)


def test_bc017_public_and_hidden_task_state_survive_policy_flattening() -> None:
    adapter = HandsOnObsAdapter(
        _BC017RecordingEnv(),
        proprio_keys=("bc017_drawer_state", "bc017_hidden_state"),
    )
    drawer = np.asarray([0.59, 0.61], dtype=np.float64)
    hidden = np.asarray([1.0, 17.0, 1.0], dtype=np.float64)

    observation = adapter.observation(
        {
            "state": {
                "bc017_drawer_state": drawer,
                "bc017_hidden_state": hidden,
            },
            "bc017_drawer_state": drawer.copy(),
            "bc017_hidden_state": hidden.copy(),
            "bc017_debug": np.ones((1,), dtype=np.float64),
        }
    )

    np.testing.assert_array_equal(
        observation["state"], np.concatenate([drawer, hidden])
    )
    np.testing.assert_array_equal(observation["bc017_drawer_state"], drawer)
    np.testing.assert_array_equal(observation["bc017_hidden_state"], hidden)
    assert "bc017_debug" not in observation
    assert adapter.observation_space.contains(observation)


def test_flattened_state_preserves_declared_proprio_order_and_bounds() -> None:
    adapter = HandsOnObsAdapter(
        _OrderedBoundedStateEnv(),
        proprio_keys=("tcp_pose", "active_obstacle_mask"),
    )
    observation = adapter.observation(
        {
            "state": {
                "tcp_pose": np.asarray([-0.1, 0.2, 1.07], dtype=np.float64),
                "active_obstacle_mask": np.asarray([1.0, 0.0], dtype=np.float64),
            }
        }
    )

    np.testing.assert_allclose(
        observation["state"],
        np.asarray([-0.1, 0.2, 1.07, 1.0, 0.0], dtype=np.float64),
    )
    assert adapter.observation_space.contains(observation)


def test_c01_matrix_telemetry_is_declared_and_forwarded_without_casting() -> None:
    adapter = HandsOnObsAdapter(_TelemetryEnv())
    raw = {
        "state": {"pose": np.asarray([1.0, 2.0], dtype=np.float64)},
        "c01_contact_wrench": np.arange(96, dtype=np.float32).reshape(8, 12),
        "c01_contact_valid": np.ones(8, dtype=np.uint8),
        "c01_undeclared_debug": np.ones((2, 2), dtype=np.float64),
    }

    observation = adapter.observation(raw)

    assert observation["c01_contact_wrench"].shape == (8, 12)
    assert observation["c01_contact_wrench"].dtype == np.float32
    assert observation["c01_contact_valid"].dtype == np.uint8
    assert "c01_undeclared_debug" not in observation
    assert adapter.observation_space.contains(observation)


def test_plan_recording_schema_fields_are_declared_and_forwarded_without_casting(
) -> None:
    adapter = HandsOnObsAdapter(_PlanRecordingEnv())
    raw = {
        "state": {"pose": np.arange(7, dtype=np.float64)},
        "plan_arm_qpos_raw": np.arange(7, dtype=np.float64) + 0.25,
        "plan_arm_qpos": np.arange(7, dtype=np.float64) + 0.5,
        "plan_arm_success": np.asarray([1.0], dtype=np.float64),
        "plan_arm_error": np.asarray([0.012], dtype=np.float64),
        "plan_hand_qpos": np.linspace(-0.5, 0.5, 20, dtype=np.float64),
        "plan_undeclared_debug": np.ones((2, 2), dtype=np.float64),
    }

    observation = adapter.observation(raw)
    schema = resolve_environment_recording_schema(adapter)

    assert schema is not None
    assert declared_recording_field_names(schema) == {
        "plan_arm_error",
        "plan_arm_qpos",
        "plan_arm_qpos_raw",
        "plan_arm_success",
        "plan_hand_qpos",
    }
    np.testing.assert_array_equal(
        observation["plan_arm_qpos_raw"],
        raw["plan_arm_qpos_raw"],
    )
    np.testing.assert_array_equal(
        observation["plan_arm_qpos"],
        raw["plan_arm_qpos"],
    )
    assert observation["plan_arm_qpos"].dtype == np.float64
    assert observation["plan_arm_success"].shape == (1,)
    assert observation["plan_arm_error"].shape == (1,)
    assert observation["plan_hand_qpos"].shape == (20,)
    assert observation["plan_hand_qpos"].dtype == np.float64
    assert "plan_undeclared_debug" not in observation
    assert adapter.observation_space.contains(observation)
