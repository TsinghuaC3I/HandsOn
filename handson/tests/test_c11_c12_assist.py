from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

from handson.sim.isaaclab.abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _C12_BLOCK_NAMES,
    _C12_PUSH_MAX_ANGULAR_SPEED,
    _C12_SETTLE_HOLD_STEPS,
)


def _matrix(
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    yaw_degrees: float = 0.0,
) -> np.ndarray:
    yaw = math.radians(yaw_degrees)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return result


def _c11_env() -> HandsOnABCIsaacLabEnv:
    return HandsOnABCIsaacLabEnv(
        "c11_xylophone_sequence",
        launch_isaac=False,
        image_obs=False,
    )


def _c12_env() -> HandsOnABCIsaacLabEnv:
    return HandsOnABCIsaacLabEnv(
        "t11_line_up_blocks",
        launch_isaac=False,
        image_obs=False,
    )


def test_new_hand_guide_waits_until_next_control_frame_install() -> None:
    env = _c12_env()
    state = {
        "hand_target_matrix": np.eye(4, dtype=np.float64),
        "hand_target_installed": False,
    }
    env._robot_link_pose_matrix = lambda _side: np.eye(4, dtype=np.float64)

    assert (
        env._assist_hand_tracking_ready(
            "right",
            state,
            position_tolerance=0.015,
            angle_tolerance=np.deg2rad(8.0),
        )
        is False
    )
    assert state["hand_tracking_ready"] is False


def test_changed_hand_target_waits_for_that_exact_command_install() -> None:
    env = _c12_env()
    installed = np.eye(4, dtype=np.float64)
    requested = _matrix((0.001, 0.0, 0.0))
    state = {
        "hand_target_matrix": requested,
        "hand_target_installed": True,
        "hand_target_installed_matrix": installed,
    }
    env._robot_link_pose_matrix = lambda _side: installed.copy()

    assert (
        env._assist_hand_tracking_ready(
            "right",
            state,
            position_tolerance=0.015,
            angle_tolerance=np.deg2rad(8.0),
        )
        is False
    )
    assert state["hand_target_installed"] is False
    assert state["hand_tracking_ready"] is False


def test_handoff_rotation_does_not_swing_translation_offset() -> None:
    raw_anchor = np.eye(4, dtype=np.float64)
    hand_anchor = _matrix((0.20, -0.10, 0.05))
    raw_rotated = _matrix(yaw_degrees=90.0)

    mapped = HandsOnABCIsaacLabEnv._map_raw_hand_handoff(
        raw_rotated,
        raw_anchor,
        hand_anchor,
    )

    np.testing.assert_allclose(
        mapped[:3, 3],
        hand_anchor[:3, 3],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        mapped[:3, :3],
        raw_rotated[:3, :3] @ hand_anchor[:3, :3],
        atol=1e-12,
    )


def _install_c11_hit_scene(
    env: HandsOnABCIsaacLabEnv,
    *,
    head_z: float,
) -> dict[str, np.ndarray]:
    matrices = {
        "c11_mallet": _matrix((0.0, -0.105, head_z)),
        "c11_key_red": _matrix(),
        "c11_key_blue": _matrix((1.0, 0.0, 0.0)),
        "c11_key_yellow": _matrix((2.0, 0.0, 0.0)),
        "c11_key_green": _matrix((3.0, 0.0, 0.0)),
    }
    env._current_body_matrix = lambda name: matrices.get(name)
    env._c11_key_contact_force = lambda _name: 0.0
    return matrices


def _install_guided_c11_strike(
    env: HandsOnABCIsaacLabEnv,
    *,
    palm: np.ndarray,
    raw_z: float,
) -> tuple[dict[str, object], list[np.ndarray]]:
    state: dict[str, object] = {
        "phase": "strike_guided",
        "guide_key": "c11_key_red",
        "hand_to_object": np.eye(4, dtype=np.float64),
        "last_target": np.eye(4, dtype=np.float64),
        "hand_target_matrix": None,
        "hand_target_installed": False,
        "visual_hand": np.zeros(20, dtype=np.float64),
        "guide_raw_start_position": np.zeros(3, dtype=np.float64),
        "guide_object_start": np.eye(4, dtype=np.float64),
    }
    writes: list[np.ndarray] = []
    env._sim_dt = 1.0
    env._c11_target_sequence = ("c11_key_red",)
    env._c11_sequence_progress = 0
    env._c11_mallet_grasp = state
    env._raw_right_hand_command_max = lambda: 1.0
    env._raw_tcp_pose = lambda _side: np.asarray(
        [0.0, 0.0, raw_z, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    env._robot_link_pose_matrix = lambda _side: palm.copy()
    env._current_body_matrix = lambda name: (
        np.eye(4, dtype=np.float64)
        if name in {"c11_mallet", "c11_key_red"}
        else None
    )
    env._c11_xy_guided_target = lambda driven, _key: np.asarray(
        driven,
        dtype=np.float64,
    ).copy()

    def write(
        _name: str,
        target: np.ndarray,
        *,
        previous_target: np.ndarray | None = None,
    ) -> tuple[bool, float, float]:
        del previous_target
        writes.append(np.asarray(target, dtype=np.float64).copy())
        return True, 0.0, 0.0

    env._set_body_rigid_grasp_target = write
    return state, writes


def test_c12_grasp_requires_an_opposing_pinch_on_visible_tool_edge() -> None:
    env = _c12_env()
    tool = np.eye(4, dtype=np.float64)

    long_edge_pinch = {
        "right_finger1_tip": np.asarray([0.066, 0.0135, 0.0]),
        "right_finger2_tip": np.asarray([0.066, -0.0135, 0.0]),
    }
    contact, _error, count = env._c12_tool_grasp_contact(
        tool,
        long_edge_pinch,
    )
    assert contact is True
    assert count == 2

    # The center of the triangular opening is not an authored edge.  Pinching
    # across its thickness must not attach the tool to an invisible volume.
    empty_triangle_pinch = {
        "right_finger1_tip": np.asarray([0.040, 0.040, 0.012]),
        "right_finger2_tip": np.asarray([0.040, 0.040, -0.012]),
    }
    contact, _error, _count = env._c12_tool_grasp_contact(
        tool,
        empty_triangle_pinch,
    )
    assert contact is False


def test_c12_push_requires_contact_correct_side_and_forward_motion() -> None:
    env = _c12_env()
    block = _matrix((0.0, 0.0, 0.92))
    goal = _matrix((0.0, 0.10, 0.92))
    env._c12_settled_blocks = {"c12_block_1", "c12_block_2"}
    env._current_body_matrix = (
        lambda name: block.copy() if name == "c12_block_0" else None
    )
    env._c12_block_goal_matrix = lambda _name, _block: goal.copy()

    current_tool = _matrix((0.0, 0.0, 0.92))
    previous_tool = _matrix((0.0, -0.001, 0.92))
    env._c12_tool_block_contact_metrics = (
        lambda _tool, _pose: (
            0.0,
            np.asarray([0.0, -0.030, 0.92]),
        )
    )
    env._c12_tool_block_contact_force = lambda _name: (1.0, True)

    candidate = env._c12_push_contact_candidate(
        current_tool,
        previous_tool,
    )
    assert candidate is not None
    assert candidate["name"] == "c12_block_0"
    assert candidate["contact_alignment"] == pytest.approx(1.0)

    # Live Isaac must fail closed if contact tensor setup is unavailable.
    env._launch_isaac = True
    env._c12_tool_block_contact_force = lambda _name: (0.0, False)
    assert (
        env._c12_push_contact_candidate(current_tool, previous_tool) is None
    )
    env._launch_isaac = False

    # When a PhysX contact view is available, geometry alone is insufficient.
    env._c12_tool_block_contact_force = lambda _name: (0.0, True)
    assert (
        env._c12_push_contact_candidate(current_tool, previous_tool) is None
    )

    # A contact on the target-facing side would pull rather than push.
    env._c12_tool_block_contact_force = lambda _name: (1.0, True)
    env._c12_tool_block_contact_metrics = (
        lambda _tool, _pose: (
            0.0,
            np.asarray([0.0, 0.030, 0.92]),
        )
    )
    assert (
        env._c12_push_contact_candidate(current_tool, previous_tool) is None
    )

    # Contact from the correct side still needs a positive forward tool step.
    env._c12_tool_block_contact_metrics = (
        lambda _tool, _pose: (
            0.0,
            np.asarray([0.0, -0.030, 0.92]),
        )
    )
    backward_tool = _matrix((0.0, -0.001, 0.92))
    assert (
        env._c12_push_contact_candidate(backward_tool, current_tool) is None
    )


def test_c12_simultaneous_positive_block_contacts_reject_candidate() -> None:
    env = _c12_env()
    env._launch_isaac = True
    blocks = {
        "c12_block_0": _matrix((0.0, 0.0, 0.92)),
        "c12_block_1": _matrix((0.05, 0.0, 0.92)),
    }
    env._c12_settled_blocks = {"c12_block_2"}
    env._current_body_matrix = lambda name: blocks.get(name)
    env._c12_block_goal_matrix = lambda _name, block: _matrix(
        (float(block[0, 3]), 0.10, float(block[2, 3]))
    )
    env._c12_tool_block_contact_force = lambda _name: (1.0, True)

    current_tool = _matrix((0.0, 0.0, 0.92))
    previous_tool = _matrix((0.0, -0.001, 0.92))
    assert (
        env._c12_push_contact_candidate(current_tool, previous_tool) is None
    )


def test_c12_forward_intent_survives_physics_substeps_for_debounce() -> None:
    env = _c12_env()
    tool = np.eye(4, dtype=np.float64)
    block = _matrix((0.0, 0.0, 0.92))
    goal = _matrix((0.0, 0.10, 0.92))
    env._c_assist_control_frame = 7
    env._c12_tool_grasp = {
        "phase": "carry",
        "hand_to_object": np.eye(4, dtype=np.float64),
        "last_target": tool.copy(),
        "previous_carried": tool.copy(),
        "previous_raw_tcp": _matrix((0.0, -0.001, 0.0)),
        "hand_target_matrix": None,
        "hand_target_installed": False,
        "visual_hand": np.zeros(20, dtype=np.float64),
    }
    env._raw_right_hand_command_max = lambda: 1.0
    env._raw_tcp_pose = lambda _side: np.asarray(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    env._robot_link_pose_matrix = lambda _side: tool.copy()
    env._c12_ensure_slot_assignment = lambda: None
    env._current_body_matrix = lambda name: {
        "c12_set_square": tool,
        "c12_block_0": block,
    }.get(name)
    movement_deltas: list[np.ndarray] = []

    def candidate(
        _tool: np.ndarray,
        _previous: np.ndarray,
        *,
        movement_delta: np.ndarray | None = None,
    ) -> dict[str, object]:
        movement_deltas.append(np.asarray(movement_delta).copy())
        return {
            "name": "c12_block_0",
            "block": block.copy(),
            "goal": goal.copy(),
            "direction": np.asarray([0.0, 1.0, 0.0]),
            "distance": 0.10,
            "contact_gap": 0.0,
            "contact_alignment": 1.0,
            "forward_step": 0.001,
        }

    env._c12_push_contact_candidate = candidate
    env._set_body_rigid_grasp_target = (
        lambda *_args, **_kwargs: (True, 0.0, 0.0)
    )

    env._update_c12_tool_assist()
    assert env._c12_active_push is None
    env._update_c12_tool_assist()

    assert env._c12_active_push is not None
    assert len(movement_deltas) == 2
    np.testing.assert_allclose(movement_deltas[0], [0.0, 0.001, 0.0])
    np.testing.assert_allclose(movement_deltas[1], movement_deltas[0])


def _install_active_c12_push(
    env: HandsOnABCIsaacLabEnv,
    *,
    last_raw_y: float,
    tracking_target: np.ndarray | None,
) -> tuple[dict[str, object], dict[str, object], list[tuple[str, np.ndarray]]]:
    start = _matrix((0.0, 0.0, 0.92))
    goal = _matrix((0.0, 0.10, 0.92))
    progress = 0.20
    block_target = env._c07_interpolate_pose(start, goal, progress)
    tool_start = np.eye(4, dtype=np.float64)
    tool_to_block = np.linalg.inv(tool_start) @ start
    tool_target = block_target @ np.linalg.inv(tool_to_block)
    palm = tool_target.copy()
    tool = tool_target.copy()
    installed_target = (
        tool_target.copy()
        if tracking_target is None
        else tracking_target.copy()
    )

    state: dict[str, object] = {
        "phase": "guided_push",
        "hand_to_object": np.eye(4, dtype=np.float64),
        "last_target": tool_target.copy(),
        "previous_carried": palm.copy(),
        "previous_raw_tcp": np.eye(4, dtype=np.float64),
        "hand_target_matrix": installed_target.copy(),
        "hand_target_installed": True,
        "hand_target_installed_matrix": installed_target.copy(),
        "visual_hand": np.zeros(20, dtype=np.float64),
    }
    active: dict[str, object] = {
        "name": "c12_block_0",
        "block_start": start.copy(),
        "goal": goal.copy(),
        "direction": np.asarray([0.0, 1.0, 0.0]),
        "distance": 0.10,
        "tool_to_block": tool_to_block,
        "progress": progress,
        "command_progress": progress,
        "raw_start_position": np.asarray(
            [0.0, last_raw_y, 0.0],
            dtype=np.float64,
        ),
        "last_tool_target": tool_target.copy(),
        "last_block_target": block_target.copy(),
        "stable_steps": 0,
    }
    writes: list[tuple[str, np.ndarray]] = []

    env._sim_dt = 0.01
    env._c12_tool_grasp = state
    env._c12_active_push = active
    env._c12_ensure_slot_assignment = lambda: None
    env._raw_right_hand_command_max = lambda: 1.0
    env._raw_tcp_pose = lambda _side: np.asarray(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    env._robot_link_pose_matrix = lambda _side: palm.copy()
    env._current_body_matrix = lambda name: {
        "c12_set_square": tool,
        "c12_block_0": block_target,
    }.get(name)

    def write(
        name: str,
        target: np.ndarray,
        *,
        previous_target: np.ndarray | None = None,
    ) -> tuple[bool, float, float]:
        del previous_target
        writes.append((name, np.asarray(target, dtype=np.float64).copy()))
        return True, 0.0, 0.0

    env._set_body_rigid_grasp_target = write
    return state, active, writes


def test_c12_stationary_input_does_not_advance_or_move_the_block() -> None:
    env = _c12_env()
    _state, active, writes = _install_active_c12_push(
        env,
        last_raw_y=0.0,
        tracking_target=None,
    )
    before = np.asarray(active["last_block_target"]).copy()

    env._update_c12_tool_assist()

    assert active["progress"] == pytest.approx(0.20)
    block_write = next(target for name, target in writes if name == "c12_block_0")
    np.testing.assert_allclose(block_write, before, atol=1e-12)
    np.testing.assert_allclose(active["last_block_target"], before, atol=1e-12)


def test_c12_guided_hand_motion_is_not_reused_as_raw_push_input() -> None:
    env = _c12_env()
    _state, active, writes = _install_active_c12_push(
        env,
        last_raw_y=0.0,
        tracking_target=None,
    )
    before = np.asarray(active["last_block_target"]).copy()
    # Simulate the physical palm following a prior assist target while the
    # glove/tracker itself remains stationary at the raw origin.
    env._robot_link_pose_matrix = lambda _side: _matrix((0.0, 0.030, 0.0))

    env._update_c12_tool_assist()

    assert active["progress"] == pytest.approx(0.20)
    block_write = next(target for name, target in writes if name == "c12_block_0")
    expected = before.copy()
    expected[1, 3] = 0.030
    np.testing.assert_allclose(block_write, expected, atol=1e-12)


def test_c12_forward_raw_glove_motion_advances_the_guided_push() -> None:
    env = _c12_env()
    _state, active, _writes = _install_active_c12_push(
        env,
        last_raw_y=0.0,
        tracking_target=None,
    )
    env._raw_tcp_pose = lambda _side: np.asarray(
        [0.0, 0.040, 0.0, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    env._update_c12_tool_assist()

    # Raw intent queues the next synchronized hand/tool/block waypoint.  The
    # physical block stays with the current palm until that command is
    # installed and followed.
    assert active["progress"] == pytest.approx(0.20)
    assert active["command_progress"] > 0.20


def test_c12_tracking_lag_freezes_progress_despite_forward_input() -> None:
    env = _c12_env()
    lagged_target = _matrix((0.10, 0.0, 0.0))
    state, active, writes = _install_active_c12_push(
        env,
        last_raw_y=-0.020,
        tracking_target=lagged_target,
    )
    before = np.asarray(active["last_block_target"]).copy()

    env._update_c12_tool_assist()

    assert state["hand_tracking_ready"] is False
    assert active["progress"] == pytest.approx(0.20)
    block_write = next(target for name, target in writes if name == "c12_block_0")
    np.testing.assert_allclose(block_write, before, atol=1e-12)


def test_c12_block_writer_failure_rolls_the_tool_back() -> None:
    env = _c12_env()
    state, active, writes = _install_active_c12_push(
        env,
        last_raw_y=0.0,
        tracking_target=None,
    )
    committed_progress = float(active["progress"])
    last_tool_target = np.asarray(active["last_tool_target"]).copy()
    env._raw_tcp_pose = lambda _side: np.asarray(
        [0.0, 0.040, 0.0, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    def write(
        name: str,
        target: np.ndarray,
        *,
        previous_target: np.ndarray | None = None,
    ) -> tuple[bool, float, float]:
        del previous_target
        writes.append((name, np.asarray(target, dtype=np.float64).copy()))
        if name == "c12_block_0":
            return False, 0.2, 0.3
        return True, 0.0, 0.0

    writes.clear()
    env._robot_link_pose_matrix = lambda _side: _matrix(
        (0.0, 0.030, 0.0)
    )
    env._set_body_rigid_grasp_target = write
    env._update_c12_tool_assist()

    assert [name for name, _target in writes] == [
        "c12_set_square",
        "c12_block_0",
        "c12_set_square",
    ]
    assert not np.allclose(writes[0][1], last_tool_target)
    np.testing.assert_allclose(writes[-1][1], last_tool_target)
    np.testing.assert_allclose(active["last_tool_target"], last_tool_target)
    assert active["progress"] == pytest.approx(committed_progress)
    np.testing.assert_allclose(
        state["hand_target_matrix"],
        np.asarray(state["hand_target_installed_matrix"]),
    )
    assert state["applied"] is False
    assert active["block_applied"] is False


def test_c12_failed_block_write_and_failed_tool_rollback_request_reset() -> None:
    env = _c12_env()
    state, active, writes = _install_active_c12_push(
        env,
        last_raw_y=0.0,
        tracking_target=None,
    )
    env._c12_push_candidate = {"name": "c12_block_0"}
    env._raw_tcp_pose = lambda _side: np.asarray(
        [0.0, 0.040, 0.0, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    env._robot_link_pose_matrix = lambda _side: _matrix(
        (0.0, 0.030, 0.0)
    )
    tool_write_count = 0

    def write(
        name: str,
        target: np.ndarray,
        *,
        previous_target: np.ndarray | None = None,
    ) -> tuple[bool, float, float]:
        nonlocal tool_write_count
        del previous_target
        writes.append((name, np.asarray(target, dtype=np.float64).copy()))
        if name == "c12_block_0":
            return False, 0.2, 0.3
        tool_write_count += 1
        if tool_write_count == 2:
            return False, 0.4, 0.5
        return True, 0.0, 0.0

    writes.clear()
    env._set_body_rigid_grasp_target = write
    env._update_c12_tool_assist()

    assert [name for name, _target in writes] == [
        "c12_set_square",
        "c12_block_0",
        "c12_set_square",
    ]
    assert env._manual_reset_requested is True
    assert env._c12_active_push is None
    assert env._c12_push_candidate is None
    assert state["applied"] is False
    assert active["block_applied"] is False


def _arm_c12_push_for_settle(
    env: HandsOnABCIsaacLabEnv,
    state: dict[str, object],
    active: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    block_goal = np.asarray(active["goal"], dtype=np.float64).copy()
    tool_goal = (
        block_goal
        @ np.linalg.inv(
            np.asarray(active["tool_to_block"], dtype=np.float64)
        )
    )
    active.update(
        {
            "progress": 1.0,
            "command_progress": 1.0,
            "last_tool_target": tool_goal.copy(),
            "last_block_target": block_goal.copy(),
            "stable_steps": _C12_SETTLE_HOLD_STEPS - 1,
        }
    )
    state["last_target"] = tool_goal.copy()
    hand_goal = tool_goal @ np.linalg.inv(
        np.asarray(state["hand_to_object"], dtype=np.float64)
    )
    state["hand_target_matrix"] = hand_goal.copy()
    state["hand_target_installed"] = True
    state["hand_target_installed_matrix"] = hand_goal.copy()
    env._robot_link_pose_matrix = lambda _side: hand_goal.copy()
    return tool_goal, block_goal


def test_c12_articulation_settle_never_uses_kinematic_switch() -> None:
    env = _c12_env()
    state, active, _writes = _install_active_c12_push(
        env,
        last_raw_y=0.0,
        tracking_target=None,
    )
    _tool_goal, block_goal = _arm_c12_push_for_settle(
        env,
        state,
        active,
    )
    env._task_articulation_views = {"c12_block_0": object()}
    kinematic_writes: list[tuple[str, bool]] = []
    env._c12_set_block_kinematic = lambda name, enabled: (
        kinematic_writes.append((name, enabled)) or True
    )

    env._update_c12_tool_assist()

    assert kinematic_writes == []
    assert "c12_block_0" in env._c12_settled_blocks
    np.testing.assert_allclose(
        env._c12_settled_targets["c12_block_0"],
        block_goal,
    )
    assert env._c12_active_push is None


def test_c12_settled_articulation_pin_failure_requests_reset() -> None:
    env = _c12_env()
    settled_target = _matrix((0.0, 0.10, 0.92))
    env._c12_settled_blocks = {"c12_block_0"}
    env._c12_settled_targets = {"c12_block_0": settled_target.copy()}
    env._task_articulation_views = {"c12_block_0": object()}
    writes: list[tuple[str, np.ndarray, np.ndarray | None]] = []

    def fail_pin(
        name: str,
        target: np.ndarray,
        *,
        previous_target: np.ndarray | None = None,
    ) -> tuple[bool, float, float]:
        writes.append(
            (
                name,
                np.asarray(target, dtype=np.float64).copy(),
                None
                if previous_target is None
                else np.asarray(previous_target, dtype=np.float64).copy(),
            )
        )
        return False, 0.1, 0.2

    env._set_body_rigid_grasp_target = fail_pin
    env._robot_link_pose_matrix = lambda _side: pytest.fail(
        "pin failure must stop the assist before reading the hand"
    )

    env._update_c12_tool_assist()

    assert env._manual_reset_requested is True
    assert len(writes) == 1
    assert writes[0][0] == "c12_block_0"
    np.testing.assert_allclose(writes[0][1], settled_target)
    np.testing.assert_allclose(writes[0][2], settled_target)


def test_c12_settle_clears_cached_raw_intent() -> None:
    env = _c12_env()
    state, active, _writes = _install_active_c12_push(
        env,
        last_raw_y=0.0,
        tracking_target=None,
    )
    _arm_c12_push_for_settle(env, state, active)
    env._c_assist_control_frame = 17
    state["raw_intent_frame"] = 17
    state["raw_intent_delta"] = np.asarray(
        [0.0, 0.050, 0.0],
        dtype=np.float64,
    )
    env._c12_set_block_kinematic = lambda _name, _enabled: True

    env._update_c12_tool_assist()

    assert env._c12_active_push is None
    assert state["phase"] == "carry_offset"
    assert state["raw_intent_frame"] == 17
    np.testing.assert_allclose(state["raw_intent_delta"], np.zeros(3))


def test_c12_short_90_degree_push_is_limited_by_angular_speed() -> None:
    env = _c12_env()
    state, active, writes = _install_active_c12_push(
        env,
        last_raw_y=0.0,
        tracking_target=None,
    )
    start = _matrix((0.0, 0.0, 0.92))
    goal = _matrix((0.0, 0.001, 0.92), yaw_degrees=90.0)
    tool_start = (
        start
        @ np.linalg.inv(
            np.asarray(active["tool_to_block"], dtype=np.float64)
        )
    )
    active.update(
        {
            "block_start": start.copy(),
            "goal": goal.copy(),
            "distance": 0.001,
            "progress": 0.0,
            "command_progress": 0.0,
            "last_tool_target": tool_start.copy(),
            "last_block_target": start.copy(),
            "stable_steps": 0,
            "raw_start_position": np.zeros(3, dtype=np.float64),
        }
    )
    state["last_target"] = tool_start.copy()
    env._sim_dt = 0.01
    env._control_dt = 0.01
    hand_start = tool_start @ np.linalg.inv(
        np.asarray(state["hand_to_object"], dtype=np.float64)
    )
    state["hand_target_matrix"] = hand_start.copy()
    state["hand_target_installed"] = True
    state["hand_target_installed_matrix"] = hand_start.copy()
    env._robot_link_pose_matrix = lambda _side: hand_start.copy()
    env._raw_tcp_pose = lambda _side: np.asarray(
        [0.0, 0.001, 0.0, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    writes.clear()

    env._update_c12_tool_assist()

    max_progress = (
        _C12_PUSH_MAX_ANGULAR_SPEED * env._control_dt / (math.pi * 0.5)
    )
    assert (
        0.0
        < float(active["command_progress"])
        <= max_progress + 1e-12
    )
    command_tool = (
        np.asarray(state["hand_target_matrix"], dtype=np.float64)
        @ np.asarray(state["hand_to_object"], dtype=np.float64)
    )
    block_target = command_tool @ np.asarray(
        active["tool_to_block"],
        dtype=np.float64,
    )
    rotation_step = float(
        np.linalg.norm(
            env._rotation_error_vector(
                block_target[:3, :3],
                start[:3, :3],
            )
        )
    )
    assert (
        rotation_step
        <= _C12_PUSH_MAX_ANGULAR_SPEED * env._control_dt + 1e-12
    )


def test_c12_slot_assignment_is_stable_after_blocks_move() -> None:
    env = _c12_env()
    baseline = np.eye(4, dtype=np.float64)
    positions = {
        "c12_block_0": _matrix((-0.20, -0.10, 0.92)),
        "c12_block_1": _matrix((0.00, -0.15, 0.92)),
        "c12_block_2": _matrix((0.20, -0.20, 0.92)),
    }
    env._current_body_matrix = lambda name: (
        baseline if name == "c12_baseline_pad" else positions.get(name)
    )

    env._c12_ensure_slot_assignment()
    initial = dict(env._c12_slot_by_block)
    assert set(initial) == set(_C12_BLOCK_NAMES)
    assert initial == {
        "c12_block_0": 0,
        "c12_block_1": 1,
        "c12_block_2": 2,
    }

    positions["c12_block_0"] = _matrix((0.30, -0.10, 0.92))
    positions["c12_block_2"] = _matrix((-0.30, -0.20, 0.92))
    env._c12_ensure_slot_assignment()

    assert env._c12_slot_by_block == initial
