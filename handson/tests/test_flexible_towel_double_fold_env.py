from __future__ import annotations

import math

import numpy as np
import pytest


pytest.importorskip("gymnasium")

from handson.runtime.collection_support import (  # noqa: E402
    ISAACSIM_COLLECTION_TASKS,
    ISAACSIM_COLLECTION_UNVERIFIED_TASKS,
    ISAACSIM_FORMAL_COLLECTION_TASKS,
    ISAACSIM_NONSTANDARD_DEFERRED_RENDER_TASKS,
)
from handson.sim.isaaclab.abc_collection_env import (  # noqa: E402
    HandsOnABCIsaacLabEnv,
    supported_native_collection_tasks,
)
from handson.sim.isaaclab.e_collection_env import (  # noqa: E402
    HandsOnEIsaacLabEnv,
)
from handson.sim.isaaclab.flexible_towel_double_fold_contract import (  # noqa: E402
    MATERIAL_NODE_COUNT,
    TASK_ID,
    TASK_NAME,
)
from handson.sim.isaaclab.eval_flex02_fold_towel_twice_env import (  # noqa: E402
    HandsOnFlexibleTowelDoubleFoldRotate90IsaacLabEnv,
)


EXPECTED_STATE_SHAPES = {
    "tcp_pose": (14,),
    "gripper_pose": (32,),
    "towel_pose": (7,),
    "towel_keypoints": (24,),
    "towel_aabb": (6,),
    "fold_metrics": (20,),
    "rotation_metrics": (8,),
    "task_phase": (5,),
    "fold_event_state": (16,),
    "towel_initial_state": (6,),
    "table_delta_height": (1,),
}


def _interface_env(
    *, randomize: bool = False
) -> HandsOnFlexibleTowelDoubleFoldRotate90IsaacLabEnv:
    return HandsOnFlexibleTowelDoubleFoldRotate90IsaacLabEnv(
        TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=randomize,
    )


def test_flexible_runtime_identity_defaults_and_rejections() -> None:
    env = _interface_env()
    try:
        assert issubclass(
            HandsOnFlexibleTowelDoubleFoldRotate90IsaacLabEnv,
            HandsOnEIsaacLabEnv,
        )
        assert issubclass(
            HandsOnFlexibleTowelDoubleFoldRotate90IsaacLabEnv,
            HandsOnABCIsaacLabEnv,
        )
        assert env.task_name == TASK_NAME
        assert env._TASK_ID == TASK_ID
        assert env._physics_backend == "physx"
        assert env.hand_type == "wuji1"
        assert env.arm_type == "rokae"
        assert env.hz == 30
        assert env._physics_dt == pytest.approx(1.0 / 120.0)
        assert env._control_dt == pytest.approx(1.0 / 30.0)
        assert env._n_substeps == 4
        assert env._time_limit == pytest.approx(180.0)
        assert env.action_space.shape == (54,)
    finally:
        env.close()

    for kwargs, match in (
        ({"task_name": "e05_fold_cloth_wipe_table"}, "requires task_name"),
        ({"hand_type": "wuji2"}, "requires hand_type"),
        ({"arm_type": "simplified_post_ik"}, "requires arm_type"),
    ):
        with pytest.raises(ValueError, match=match):
            HandsOnFlexibleTowelDoubleFoldRotate90IsaacLabEnv(
                launch_isaac=False,
                image_obs=False,
                **kwargs,
            )


def test_flexible_runtime_rejects_explicit_timing_drift() -> None:
    for kwargs, match in (
        ({"physics_dt": 1.0 / 60.0}, "physics_dt"),
        ({"control_dt": 0.020}, "control_dt"),
        ({"hz": 60}, "hz"),
    ):
        with pytest.raises(ValueError, match=match):
            HandsOnFlexibleTowelDoubleFoldRotate90IsaacLabEnv(
                TASK_NAME,
                launch_isaac=False,
                image_obs=False,
                **kwargs,
            )

    env = HandsOnFlexibleTowelDoubleFoldRotate90IsaacLabEnv(
        TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        physics_dt=1.0 / 120.0,
        control_dt=1.0 / 30.0,
        hz=30,
    )
    try:
        assert env._physics_dt == pytest.approx(1.0 / 120.0)
        assert env._control_dt == pytest.approx(1.0 / 30.0)
        assert env.hz == 30
        assert env._n_substeps == 4
    finally:
        env.close()


def test_flexible_runtime_rejects_timing_environment_pollution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HANDSON_ISAACLAB_CONTROL_DT", raising=False)
    monkeypatch.delenv("HANDSON_ISAACLAB_CONTROL_HZ", raising=False)

    monkeypatch.setenv("HANDSON_ISAACLAB_CONTROL_DT", "0.02")
    with pytest.raises(ValueError, match="HANDSON_ISAACLAB_CONTROL_DT"):
        _interface_env()

    monkeypatch.delenv("HANDSON_ISAACLAB_CONTROL_DT")
    monkeypatch.setenv("HANDSON_ISAACLAB_CONTROL_HZ", "60")
    with pytest.raises(ValueError, match="HANDSON_ISAACLAB_CONTROL_HZ"):
        _interface_env()

    monkeypatch.delenv("HANDSON_ISAACLAB_CONTROL_HZ")
    env = _interface_env()
    try:
        assert env._physics_dt == pytest.approx(1.0 / 120.0)
        assert env._control_dt == pytest.approx(1.0 / 30.0)
        assert env.hz == 30
    finally:
        env.close()


def test_flexible_observation_shapes_are_fixed_finite_and_contained() -> None:
    env = _interface_env(randomize=True)
    try:
        observation, info = env.reset(seed=17)
        assert tuple(observation) == ("state", "e01_hand_qpos")
        assert tuple(observation["state"]) == tuple(EXPECTED_STATE_SHAPES)
        for key, shape in EXPECTED_STATE_SHAPES.items():
            value = observation["state"][key]
            assert value.shape == shape
            assert value.dtype == np.float64
            assert np.all(np.isfinite(value))
        assert observation["e01_hand_qpos"].shape == (40,)
        assert observation["e01_hand_qpos"].dtype == np.float64
        assert np.all(np.isfinite(observation["e01_hand_qpos"]))
        assert env.observation_space.contains(observation)
        assert info["task_id"] == TASK_ID
        assert info["task_name"] == TASK_NAME
        assert info["towel_node_count"] == MATERIAL_NODE_COUNT
        assert info["collection_verified"] is False
        assert info["physical_replay_certified"] is False
        assert info["succeed"] is False
    finally:
        env.close()


def test_flexible_seeded_reset_is_deterministic_and_bounded() -> None:
    first = _interface_env(randomize=True)
    repeat = _interface_env(randomize=True)
    different = _interface_env(randomize=True)
    try:
        first_observation, _ = first.reset(seed=81)
        repeat_observation, _ = repeat.reset(seed=81)
        different_observation, _ = different.reset(seed=82)

        first_state = first_observation["state"]["towel_initial_state"]
        repeat_state = repeat_observation["state"]["towel_initial_state"]
        different_state = different_observation["state"]["towel_initial_state"]
        np.testing.assert_array_equal(first_state, repeat_state)
        np.testing.assert_array_equal(
            first._e_reset_nodes_np, repeat._e_reset_nodes_np
        )
        assert not np.array_equal(first_state, different_state)

        assert -0.080 <= first_state[0] <= 0.080
        assert -0.120 <= first_state[1] <= 0.120
        assert -math.radians(5.0) <= first_state[2] <= math.radians(5.0)
        assert 0.0 <= first_state[3] <= 0.002
        assert first_state[4] in (0.0, 1.0, 2.0)
        assert first_state[5] == 0.0
        np.testing.assert_array_equal(
            first_observation["state"]["table_delta_height"],
            np.zeros(1, dtype=np.float64),
        )
    finally:
        first.close()
        repeat.close()
        different.close()


def test_flexible_fold_event_state_round_trips_and_reset_clears_it() -> None:
    env = _interface_env(randomize=True)
    try:
        initial_observation, _ = env.reset(seed=33)
        initial_state = initial_observation["state"]["towel_initial_state"].copy()
        env._flexible_sequence.restore(
            {
                "task_id": TASK_ID,
                "task_name": TASK_NAME,
                "stage": "rotated",
                "update_count": 12,
                "flat_latched": True,
                "flat_stable_frames": 3,
                "first_fold_latched": True,
                "first_fold_yaw_rad": 0.10,
                "last_yaw_rad": 0.10 + 0.5 * math.pi,
                "cumulative_rotation_rad": 0.5 * math.pi,
                "rotation_direction": 1,
                "rotation_latched": True,
                "second_fold_latched": False,
                "release_latched": False,
                "release_stable_frames": 0,
                "last_metrics_finite": True,
                "ready": False,
                "success": False,
            }
        )
        recorded_event = env._compute_observation()["state"][
            "fold_event_state"
        ].copy()
        assert recorded_event.shape == (16,)
        assert recorded_event[0] == 2.0
        assert recorded_event[1:4].tolist() == [1.0, 1.0, 1.0]

        reset_observation, _ = env.reset(seed=34)
        reset_event = reset_observation["state"]["fold_event_state"]
        assert reset_event[0] == 0.0
        assert not np.any(reset_event[1:6])
        # The pre-action observation is one validated FSM sample, not an
        # applied control action.  It must be deterministic and never retain
        # counters or latches from the previous episode.
        assert reset_event[10] == 1.0
        repeated_reset, _ = env.reset(seed=34)
        np.testing.assert_array_equal(
            repeated_reset["state"]["fold_event_state"], reset_event
        )

        restored = env.restore_recorded_initial_state(
            {
                "towel_initial_state": initial_state,
                "table_delta_height": np.zeros(1, dtype=np.float64),
                "fold_event_state": recorded_event,
            }
        )
        np.testing.assert_array_equal(
            restored["state"]["fold_event_state"], recorded_event
        )
        np.testing.assert_array_equal(
            restored["state"]["towel_initial_state"], initial_state
        )
        snapshot = env._flexible_sequence.snapshot()
        assert snapshot.stage == "rotated"
        assert snapshot.update_count == 12
        assert snapshot.rotation_latched
        assert not snapshot.second_fold_latched
    finally:
        env.close()


def test_flexible_replay_contract_is_unverified_1681_node_120_30() -> None:
    env = _interface_env()
    try:
        replay = env.get_physical_replay_reset_contract()
        assert replay["flexible_task_id"] == TASK_ID
        assert replay["surface_only"] is True
        assert replay["surface_node_count"] == 1681
        assert replay["physics_hz"] == 120
        assert replay["control_hz"] == 30
        assert replay["physics_dt"] == pytest.approx(1.0 / 120.0)
        assert replay["control_dt"] == pytest.approx(1.0 / 30.0)
        assert replay["control_substeps"] == 4
        assert replay["ordered_hidden_state_field"] == "state/fold_event_state"
        assert replay["replay_mode"] == "diagnostic_state_guided"
        assert replay["physical_replay_certified"] is False
        assert TASK_NAME in ISAACSIM_COLLECTION_TASKS
        assert TASK_NAME not in ISAACSIM_COLLECTION_UNVERIFIED_TASKS
        assert TASK_NAME in ISAACSIM_FORMAL_COLLECTION_TASKS
        assert TASK_NAME in ISAACSIM_NONSTANDARD_DEFERRED_RENDER_TASKS
    finally:
        env.close()


def test_flexible_task_is_available_to_the_internal_abc_spec_registry() -> None:
    assert TASK_NAME in supported_native_collection_tasks()
    env = _interface_env()
    try:
        assert env._spec.task_name == TASK_NAME
        assert env._spec.bimanual is True
        assert env._spec.target_objects == ("flexible_towel",)
        assert env._spec.receptacle == "table"
        assert env._spec.aliases == {"towel_pose": "target"}
        assert env._spec.success_steps == 1
    finally:
        env.close()
