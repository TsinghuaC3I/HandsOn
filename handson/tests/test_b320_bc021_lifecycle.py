from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from handson.data.recording_fields import (
    select_current_observation_for_recording,
    select_next_observation_for_recording,
)
from handson.sim.isaaclab.b320_bc021_contract import (
    BC021Frame,
    BC021Stage,
    BC021State,
    CONTACT_KEYS,
    HIDDEN_OBSERVATION_FIELDS,
    HIDDEN_OBSERVATION_SHAPE,
    PROVISIONAL_GEOMETRY,
    WORKSPACE_DEBOUNCE_FRAMES,
)
from handson.sim.isaaclab.b320_bc021_env import (
    HandsOnB320BC021IsaacLabEnv,
)
from handson.tasks.t01_align_blocks_in_midair.config import TaskConfig
from handson.tasks.obs_adapters import HandsOnObsAdapter


FINAL_TEST_GEOMETRY = replace(PROVISIONAL_GEOMETRY, provisional=False)
HIDDEN_WORKSPACE_INDEX = HIDDEN_OBSERVATION_FIELDS.index("workspace_frames")


def _interface_env() -> HandsOnB320BC021IsaacLabEnv:
    env = HandsOnB320BC021IsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    env.hz = 0
    return env


def _zero_action(env: HandsOnB320BC021IsaacLabEnv) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float64)


def _pose(x: float, y: float, z: float) -> np.ndarray:
    return np.asarray(
        [x, y, z, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )


def _outside_workspace_frame() -> BC021Frame:
    return BC021Frame(
        left_pose_wxyz=_pose(-0.10, 0.0, 1.01),
        right_pose_wxyz=_pose(0.28, 0.0, 1.01),
        left_velocity=np.zeros(6, dtype=np.float64),
        right_velocity=np.zeros(6, dtype=np.float64),
        contact_force_n={key: 0.0 for key in CONTACT_KEYS},
        contact_readable={key: False for key in CONTACT_KEYS},
        block_contact_points_w=np.zeros((0, 3), dtype=np.float64),
        block_contact_point_forces_n=np.zeros(0, dtype=np.float64),
        block_contact_points_readable=False,
        pose_truth_readable=True,
        velocity_truth_readable=True,
    )


def _workspace_pending_state(workspace_frames: int) -> BC021State:
    return BC021State(
        stage=BC021Stage.BOTH_AIRBORNE,
        assigned_contact_seen=True,
        left_airborne_seen=True,
        both_airborne_seen=True,
        workspace_frames=workspace_frames,
        first_assigned_contact_frame=2,
        first_left_airborne_frame=3,
        first_both_airborne_frame=4,
    )


def _dirty_bc021_runtime(env: HandsOnB320BC021IsaacLabEnv) -> None:
    state = env._bc021_state
    state.stage = BC021Stage.FAILED
    state.assigned_contact_seen = True
    state.left_airborne_seen = True
    state.both_airborne_seen = True
    state.reference_pose_wxyz = _pose(-0.10, 0.02, 1.01)
    state.max_reference_translation_m = 0.019
    state.marked_face_contact_seen = True
    state.stable_substeps = 17
    state.reference_limit_frames = 1
    state.deep_penetration_frames = 2
    state.drop_frames = 1
    state.workspace_frames = 2
    state.first_assigned_contact_frame = 2
    state.first_left_airborne_frame = 3
    state.first_both_airborne_frame = 4
    state.reference_latched_frame = 3
    state.first_marked_contact_frame = 5
    state.stable_started_frame = 6
    state.first_terminal_reason = "block_left_collection_workspace"
    state.first_terminal_frame = 9
    state.last_metrics = {"stale": True}

    env._bc021_contact_force_n = {key: 7.0 for key in CONTACT_KEYS}
    env._bc021_contact_readable = {key: True for key in CONTACT_KEYS}
    env._bc021_block_contact_points_w = np.asarray(
        [[-0.05, 0.0, 1.01]],
        dtype=np.float64,
    )
    env._bc021_block_contact_point_forces_n = np.asarray(
        [3.0],
        dtype=np.float64,
    )
    env._bc021_block_contact_points_readable = True
    env._bc021_pose_truth_readable = True
    env._bc021_velocity_truth_readable = True
    env._bc021_physics_frame_index = 23
    env._bc021_recording_origin_accepted = True
    env._bc021_recording_origin_poses = {
        "block_a": _pose(-0.10, 0.02, 1.01),
        "block_b": _pose(-0.10, -0.02, 1.01),
    }
    _payload, digest = env._bc021_pose_set_payload(
        env._bc021_recording_origin_poses
    )
    env._bc021_recording_origin_pose_sha256 = digest


def _assert_bc021_runtime_is_clean(
    env: HandsOnB320BC021IsaacLabEnv,
) -> None:
    np.testing.assert_array_equal(
        env._bc021_state.hidden_observation(),
        BC021State().hidden_observation(),
    )
    # reset() immediately builds its returned observation, which refreshes
    # derived metrics without advancing latches.  The pre-reset cache must not
    # survive that refresh.
    assert "stale" not in env._bc021_state.last_metrics
    assert env._bc021_state.last_metrics["contact_truth_ready"] is False
    assert env._bc021_state.last_metrics["terminal_reason"] is None
    assert env._bc021_contact_force_n == {
        key: 0.0 for key in CONTACT_KEYS
    }
    assert env._bc021_contact_readable == {
        key: False for key in CONTACT_KEYS
    }
    assert env._bc021_block_contact_points_w.shape == (0, 3)
    assert env._bc021_block_contact_point_forces_n.shape == (0,)
    assert env._bc021_block_contact_points_readable is False
    assert env._bc021_pose_truth_readable is False
    assert env._bc021_velocity_truth_readable is False
    assert env._bc021_physics_frame_index == 0
    assert env._bc021_recording_origin_accepted is False
    assert env._bc021_recording_origin_poses == {}
    assert env._bc021_recording_origin_pose_sha256 is None


def test_bc021_hard_failure_terminates_and_keeps_reason_in_step_info() -> None:
    env = _interface_env()
    env._time_limit = 30.0
    env._bc021_state._latch_terminal(
        "block_left_collection_workspace",
        frame_index=7,
    )

    _observation, _reward, terminated, truncated, info = env.step(
        _zero_action(env)
    )

    assert terminated is True
    assert truncated is False
    assert info["succeed"] is False
    assert info["failed"] is True
    assert info["b320_failure_reason"] == "block_left_collection_workspace"
    assert (
        info["bc021_first_terminal_reason"]
        == "block_left_collection_workspace"
    )
    assert info["bc021_first_terminal_frame"] == 7


def test_bc021_pure_time_limit_truncates_without_physical_failure() -> None:
    env = _interface_env()
    env._time_limit = env._control_dt

    _observation, _reward, terminated, truncated, info = env.step(
        _zero_action(env)
    )

    assert terminated is False
    assert truncated is True
    assert info["succeed"] is False
    assert info.get("failed", False) is False
    assert info["time_limit_reached"] is True
    assert info["truncated_reason"] == "time_limit"
    assert info["bc021_first_terminal_reason"] is None
    assert info["bc021_first_terminal_frame"] is None


@pytest.mark.parametrize("reset_kind", ["normal", "manual"])
def test_bc021_reset_paths_clear_latches_counters_and_contact_caches(
    reset_kind: str,
) -> None:
    env = _interface_env()
    _dirty_bc021_runtime(env)

    if reset_kind == "manual":
        env._manual_reset_requested = True
        _observation, _reward, terminated, truncated, step_info = env.step(
            _zero_action(env)
        )
        assert terminated is False
        assert truncated is False
        assert step_info["manual_reset"] is True
        # The recorder owns the second half of the reset transaction.
        assert (
            env._bc021_state.first_terminal_reason
            == "block_left_collection_workspace"
        )

    observation, _reset_info = env.reset(seed=17)

    assert env.observation_space.contains(observation)
    _assert_bc021_runtime_is_clean(env)


def test_bc021_hidden_current_and_next_distinguish_terminal_counter_state() -> None:
    env = _interface_env()
    current_state = _workspace_pending_state(
        WORKSPACE_DEBOUNCE_FRAMES - 2
    )
    next_state = _workspace_pending_state(WORKSPACE_DEBOUNCE_FRAMES - 1)

    env._bc021_state = current_state
    current_hidden = env._bc021_observation_values({})[
        "bc021_hidden_state"
    ]
    env._bc021_state = next_state
    next_hidden = env._bc021_observation_values({})["bc021_hidden_state"]

    assert current_hidden.shape == HIDDEN_OBSERVATION_SHAPE == (30,)
    assert next_hidden.shape == HIDDEN_OBSERVATION_SHAPE
    differing = np.flatnonzero(current_hidden != next_hidden)
    np.testing.assert_array_equal(differing, [HIDDEN_WORKSPACE_INDEX])
    assert current_hidden[HIDDEN_WORKSPACE_INDEX] == (
        WORKSPACE_DEBOUNCE_FRAMES - 2
    )
    assert next_hidden[HIDDEN_WORKSPACE_INDEX] == (
        WORKSPACE_DEBOUNCE_FRAMES - 1
    )

    outside = _outside_workspace_frame()
    current_state.evaluate(
        outside,
        FINAL_TEST_GEOMETRY,
        frame_index=10,
        update_latches=True,
    )
    next_state.evaluate(
        outside,
        FINAL_TEST_GEOMETRY,
        frame_index=10,
        update_latches=True,
    )
    assert current_state.first_terminal_reason is None
    assert (
        next_state.first_terminal_reason
        == "block_left_collection_workspace"
    )


def test_bc021_recorder_selectors_keep_current_and_next_hidden_state() -> None:
    env = _interface_env()
    proprio_keys = list(TaskConfig.b320_proprio_keys)
    assert "bc021_hidden_state" in proprio_keys
    adapter = HandsOnObsAdapter(env, proprio_keys=proprio_keys)

    env._bc021_state = _workspace_pending_state(
        WORKSPACE_DEBOUNCE_FRAMES - 2
    )
    current_raw = env._compute_observation()
    current_adapted = adapter.observation(current_raw)
    current_selected = select_current_observation_for_recording(
        current_adapted,
        required_fields=(),
        include_required_fields=True,
    )

    env._bc021_state = _workspace_pending_state(
        WORKSPACE_DEBOUNCE_FRAMES - 1
    )
    next_raw = env._compute_observation()
    next_adapted = adapter.observation(next_raw)
    next_selected = select_next_observation_for_recording(
        next_adapted,
        required_fields=(),
    )

    hidden_offset = sum(
        int(np.prod(env.observation_space["state"][key].shape))
        for key in proprio_keys[: proprio_keys.index("bc021_hidden_state")]
    )
    hidden_slice = slice(
        hidden_offset,
        hidden_offset + HIDDEN_OBSERVATION_SHAPE[0],
    )
    current_recorded = current_selected["state"][hidden_slice]
    next_recorded = next_selected["state"][hidden_slice]

    np.testing.assert_array_equal(
        current_recorded,
        current_raw["state"]["bc021_hidden_state"],
    )
    np.testing.assert_array_equal(
        next_recorded,
        next_raw["state"]["bc021_hidden_state"],
    )
    assert (
        current_recorded[HIDDEN_WORKSPACE_INDEX]
        == WORKSPACE_DEBOUNCE_FRAMES - 2
    )
    assert (
        next_recorded[HIDDEN_WORKSPACE_INDEX]
        == WORKSPACE_DEBOUNCE_FRAMES - 1
    )
    assert not np.array_equal(current_recorded, next_recorded)
