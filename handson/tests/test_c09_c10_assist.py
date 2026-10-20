from __future__ import annotations

import math

import numpy as np
import pytest

from handson.sim.isaaclab.abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _C09_GRIP_TAB_CONTACT_MARGIN,
    _C09_GRIP_TAB_CENTER_LOCAL,
    _C09_GRIP_TAB_CENTERS_LOCAL,
    _C09_GRIP_TAB_HALF_EXTENTS,
    _C09_GRIP_TAB_NEAR_MARGIN,
    _C09_GRIP_TAB_PINCH_MAX_DISTANCE,
    _C09_GRIP_TAB_PINCH_MIN_DISTANCE,
    _C09_HAND_TRACK_POSITION_TOLERANCE,
    _C09_STACK_CAPTURE_Z_TOLERANCE,
    _C09_STACK_SEPARATION,
    _C09_STAPLE_MARK_LOCAL,
    _C09_STAPLE_ZONE_LOCAL,
    _C10_CABLE_GRASP_BOX_CENTER,
    _C10_CABLE_GRASP_BOX_HALF_EXTENTS,
    _C10_CABLE_PINCH_DISTANCE_RANGE,
    _C10_CABLE_SEATED_RELATIVE,
    _C10_GRASP_CONTACT_MARGIN,
    _C10_GRASP_NEAR_MARGIN,
    _C10_HAND_TRACK_POSITION_TOLERANCE,
    _C10_PLUG_SEATED_RELATIVE,
    _C10_PLUG_GRASP_BOX_CENTER,
    _C10_PLUG_GRASP_BOX_HALF_EXTENTS,
    _C10_PLUG_PINCH_DISTANCE_RANGE,
)


def _matrix(
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    roll_degrees: float = 0.0,
    pitch_degrees: float = 0.0,
    yaw_degrees: float = 0.0,
) -> np.ndarray:
    roll = math.radians(roll_degrees)
    pitch = math.radians(pitch_degrees)
    yaw = math.radians(yaw_degrees)
    roll_rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(roll), -math.sin(roll)],
            [0.0, math.sin(roll), math.cos(roll)],
        ],
        dtype=np.float64,
    )
    pitch_rotation = np.asarray(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    yaw_rotation = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = yaw_rotation @ pitch_rotation @ roll_rotation
    result[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return result


def _box_pinch_tips(
    object_matrix: np.ndarray,
    *,
    side: str,
    center_local: np.ndarray,
    half_extents: np.ndarray,
    axis: int = 0,
) -> dict[str, np.ndarray]:
    thumb = np.asarray(center_local, dtype=np.float64).copy()
    opponent = thumb.copy()
    thumb[axis] += float(half_extents[axis])
    opponent[axis] -= float(half_extents[axis])
    return {
        f"{side}_finger1_tip": (
            np.asarray(object_matrix, dtype=np.float64)
            @ np.asarray([*thumb, 1.0], dtype=np.float64)
        )[:3],
        f"{side}_finger2_tip": (
            np.asarray(object_matrix, dtype=np.float64)
            @ np.asarray([*opponent, 1.0], dtype=np.float64)
        )[:3],
    }


@pytest.mark.parametrize(
    ("center", "half_extents", "pinch_range"),
    [
        (
            _C10_PLUG_GRASP_BOX_CENTER,
            _C10_PLUG_GRASP_BOX_HALF_EXTENTS,
            _C10_PLUG_PINCH_DISTANCE_RANGE,
        ),
        (
            _C10_CABLE_GRASP_BOX_CENTER,
            _C10_CABLE_GRASP_BOX_HALF_EXTENTS,
            _C10_CABLE_PINCH_DISTANCE_RANGE,
        ),
    ],
)
def test_c10_connector_grasp_requires_visible_box_contact(
    center: np.ndarray,
    half_extents: np.ndarray,
    pinch_range: tuple[float, float],
) -> None:
    connector = _matrix((-0.14, 0.08, 0.96), yaw_degrees=-31.0)
    tips = _box_pinch_tips(
        connector,
        side="right",
        center_local=center,
        half_extents=half_extents,
        axis=1,
    )
    contact, _error, count = (
        HandsOnABCIsaacLabEnv._visible_box_pinch_contact(
            connector,
            tips,
            side="right",
            center_local=center,
            half_extents=half_extents,
            contact_margin=_C10_GRASP_CONTACT_MARGIN,
            near_margin=_C10_GRASP_NEAR_MARGIN,
            pinch_distance_range=pinch_range,
        )
    )
    assert contact is True
    assert count == 2

    shifted = {
        name: position + connector[:3, 2] * 0.05
        for name, position in tips.items()
    }
    contact, _error, _count = (
        HandsOnABCIsaacLabEnv._visible_box_pinch_contact(
            connector,
            shifted,
            side="right",
            center_local=center,
            half_extents=half_extents,
            contact_margin=_C10_GRASP_CONTACT_MARGIN,
            near_margin=_C10_GRASP_NEAR_MARGIN,
            pinch_distance_range=pinch_range,
        )
    )
    assert contact is False


def test_c10_grasp_starts_without_snap_and_then_follows_the_hand_rigidly() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t02_connect_charger_and_cable",
        launch_isaac=False,
        image_obs=False,
    )
    palm = _matrix((0.08, -0.02, 1.01), yaw_degrees=8.0)
    plug = _matrix((-0.20, -0.13, 0.94), yaw_degrees=-14.0)
    matrices = {
        "c10_charger_plug": plug.copy(),
        "c10_charging_cable": _matrix((0.50, 0.45, 1.15)),
        "c10_power_strip": _matrix((-0.45, 0.35, 0.90)),
    }
    palm_state = {"right": palm.copy(), "left": None}
    tips = _box_pinch_tips(
        plug,
        side="right",
        center_local=_C10_PLUG_GRASP_BOX_CENTER,
        half_extents=_C10_PLUG_GRASP_BOX_HALF_EXTENTS,
        axis=1,
    )
    writes: list[tuple[str, np.ndarray]] = []

    env._robot_link_pose_matrix = lambda side: palm_state[side]
    env._raw_hand_command_max = lambda side: 1.2 if side == "right" else 0.0
    env._wuji_fingertip_positions = lambda side: (
        tips if side == "right" else {}
    )
    env._current_body_matrix = lambda name: matrices.get(name)
    env._current_body_pose = lambda name: (
        env._pose_matrix_to_vector(matrices[name])
        if name in matrices
        else None
    )
    env._actual_hand_joint_pose = lambda side: (
        env._right_hand.copy()
        if side == "right"
        else env._left_hand.copy()
    )
    env._update_c10_connection_latches = lambda: None

    def write(
        name: str,
        target: np.ndarray,
        **_kwargs,
    ) -> tuple[bool, float, float]:
        matrices[name] = np.asarray(target, dtype=np.float64).copy()
        writes.append((name, matrices[name].copy()))
        return True, 0.0, 0.0

    env._set_body_rigid_grasp_target = write
    env._update_c10_connector_assist()

    state = env._c10_connector_grasps["right"]
    hand_to_object = np.asarray(state["hand_to_object"]).copy()
    np.testing.assert_allclose(writes[-1][1], plug, atol=1e-10)
    np.testing.assert_allclose(
        hand_to_object,
        np.linalg.inv(palm) @ plug,
        atol=1e-10,
    )

    moved_palm = _matrix((0.14, -0.04, 1.04), yaw_degrees=19.0)
    palm_state["right"] = moved_palm
    env._update_c10_connector_assist()

    expected = moved_palm @ hand_to_object
    np.testing.assert_allclose(writes[-1][1], expected, atol=1e-10)
    np.testing.assert_allclose(
        matrices["c10_charger_plug"],
        expected,
        atol=1e-10,
    )


def test_unheld_c10_objects_are_never_magnetized() -> None:
    c10 = HandsOnABCIsaacLabEnv(
        "t02_connect_charger_and_cable",
        launch_isaac=False,
        image_obs=False,
    )
    c10_matrices = {
        "c10_charger_plug": _matrix(),
        "c10_charging_cable": _matrix((0.1, 0.0, 0.0)),
        "c10_power_strip": _matrix((0.0, 0.0, -0.02)),
    }
    c10_writes: list[str] = []
    c10._robot_link_pose_matrix = lambda _side: None
    c10._raw_hand_command_max = lambda _side: 0.0
    c10._current_body_matrix = lambda name: c10_matrices.get(name)
    c10._current_body_pose = lambda name: (
        c10._pose_matrix_to_vector(c10_matrices[name])
        if name in c10_matrices
        else None
    )
    c10._set_body_rigid_grasp_target = (
        lambda name, *_args, **_kwargs: (
            c10_writes.append(name) or (True, 0.0, 0.0)
        )
    )
    c10._update_c10_connection_latches = lambda: None
    c10._update_c10_connector_assist()
    assert c10_writes == []
    assert c10._c10_connector_grasps == {}
    assert c10._c10_assist_metrics == {
        "c10_plug_capture_active": False,
        "c10_cable_capture_active": False,
    }


def test_c10_funnel_alignment_freezes_while_the_commanded_hand_lags() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t02_connect_charger_and_cable",
        launch_isaac=False,
        image_obs=False,
    )
    plug = _matrix()
    matrices = {
        "c10_charger_plug": plug.copy(),
        "c10_charging_cable": _matrix((0.4, 0.4, 0.4)),
        "c10_power_strip": _matrix((0.5, -0.4, -0.2)),
    }
    delayed_hand_target = _matrix(
        (_C10_HAND_TRACK_POSITION_TOLERANCE + 0.05, 0.0, 0.0)
    )
    env._c10_connector_grasps = {
        "right": {
            "name": "c10_charger_plug",
            "phase": "funnel_aligning",
            "hand_to_object": np.eye(4),
            "last_target": plug.copy(),
            "hand_target_matrix": delayed_hand_target,
            "hand_target_installed": True,
            "visual_hand": env._right_hand.copy(),
            "approach_target": _matrix((0.08, 0.0, 0.0)),
            "seated_target": _matrix((0.12, 0.0, 0.0)),
            "stable_steps": 0,
        }
    }
    writes: list[np.ndarray] = []
    env._robot_link_pose_matrix = lambda side: (
        np.eye(4) if side == "right" else None
    )
    env._raw_hand_command_max = lambda side: 1.2 if side == "right" else 0.0
    env._current_body_matrix = lambda name: matrices.get(name)
    env._current_body_pose = lambda name: (
        env._pose_matrix_to_vector(matrices[name])
        if name in matrices
        else None
    )
    env._update_c10_connection_latches = lambda: None
    env._set_body_rigid_grasp_target = (
        lambda _name, target, **_kwargs: (
            writes.append(np.asarray(target).copy()) or (True, 0.0, 0.0)
        )
    )
    env._update_c10_connector_assist()

    state = env._c10_connector_grasps["right"]
    assert state["hand_tracking_ready"] is False
    assert (
        state["hand_tracking_position_error"]
        > _C10_HAND_TRACK_POSITION_TOLERANCE
    )
    np.testing.assert_allclose(writes[-1], plug, atol=1e-10)


def test_c10_guidance_rejects_wrong_connector_direction() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t02_connect_charger_and_cable",
        launch_isaac=False,
        image_obs=False,
    )
    geometry = {
        "plug_lateral_error": 0.0,
        "plug_axis_alignment": -1.0,
        "plug_insertion_depth": 0.0,
        "cable_lateral_error": 0.0,
        "cable_axis_alignment": -1.0,
        "cable_insertion_depth": 0.0,
    }
    assert (
        env._c10_guidance_trigger(
            "c10_charger_plug",
            geometry,
        )
        is False
    )
    env._c10_plug_latched = True
    assert (
        env._c10_guidance_trigger(
            "c10_charging_cable",
            geometry,
        )
        is False
    )


def test_c10_connector_rolled_90_degrees_in_socket_is_not_captured() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t02_connect_charger_and_cable",
        launch_isaac=False,
        image_obs=False,
    )
    strip = np.eye(4, dtype=np.float64)
    plug = strip @ _C10_PLUG_SEATED_RELATIVE
    plug[:3, :3] = (
        plug[:3, :3]
        @ _matrix(roll_degrees=90.0)[:3, :3]
    )
    cable = _matrix((0.50, 0.50, 1.20))
    geometry = env._c10_connection_geometry(
        env._pose_matrix_to_vector(plug),
        env._pose_matrix_to_vector(cable),
        env._pose_matrix_to_vector(strip),
    )

    assert geometry["plug_insertion_depth"] > 0.0
    assert geometry["plug_axis_alignment"] == pytest.approx(1.0)
    assert geometry["plug_secondary_alignment"] == pytest.approx(0.0)
    assert geometry["first_connected_now"] is False
    assert (
        env._c10_guidance_trigger("c10_charger_plug", geometry)
        is False
    )


def test_c10_articulation_latch_never_uses_kinematic_switch() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t02_connect_charger_and_cable",
        launch_isaac=False,
        image_obs=False,
    )
    name = "c10_charger_plug"
    target = _matrix((0.02, -0.01, 0.93))
    rigid_writes: list[tuple[str, np.ndarray]] = []
    kinematic_writes: list[tuple[str, bool]] = []
    env._task_articulation_views = {name: object()}
    env._set_body_rigid_grasp_target = (
        lambda written_name, written_target, **_kwargs: (
            rigid_writes.append(
                (
                    written_name,
                    np.asarray(written_target, dtype=np.float64).copy(),
                )
            )
            or (True, 0.0, 0.0)
        )
    )
    env._set_c10_connector_kinematic = lambda written_name, enabled: (
        kinematic_writes.append((written_name, enabled)) or True
    )

    assert env._commit_c10_connector_latch(name, target) is True
    assert [written_name for written_name, _target in rigid_writes] == [name]
    np.testing.assert_allclose(rigid_writes[0][1], target)
    assert kinematic_writes == []
    assert env._c10_plug_latched is True


def test_c10_latch_preserves_actual_connector_and_hand_transform() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t02_connect_charger_and_cable",
        launch_isaac=False,
        image_obs=False,
    )
    name = "c10_charger_plug"
    strip = _matrix((0.17, -0.08, 0.91), yaw_degrees=24.0)
    canonical = strip @ _C10_PLUG_SEATED_RELATIVE
    actual = canonical @ _matrix(roll_degrees=2.0)
    actual[:3, 3] += strip[:3, 1] * 0.003
    palm = _matrix((0.11, -0.16, 1.02), yaw_degrees=-13.0)
    hand_to_object = np.linalg.inv(palm) @ actual
    matrices = {
        name: actual.copy(),
        "c10_charging_cable": _matrix((0.40, 0.35, 1.15)),
        "c10_power_strip": strip.copy(),
    }
    written: list[np.ndarray] = []
    env._task_articulation_views = {name: object()}
    env._c10_connector_grasps = {
        "right": {
            "name": name,
            "phase": "inserting",
            "hand_to_object": hand_to_object.copy(),
            "last_target": actual.copy(),
            "hand_target_installed": True,
        }
    }
    env._c_assist_raw_right_tcp_pose = env._pose_matrix_to_vector(palm)
    env._current_body_matrix = lambda body_name: matrices.get(body_name)
    env._current_body_pose = lambda body_name: (
        env._pose_matrix_to_vector(matrices[body_name])
        if body_name in matrices
        else None
    )

    def write(
        body_name: str,
        target: np.ndarray,
        **_kwargs,
    ) -> tuple[bool, float, float]:
        target_copy = np.asarray(target, dtype=np.float64).copy()
        matrices[body_name] = target_copy
        written.append(target_copy)
        return True, 0.0, 0.0

    env._set_body_rigid_grasp_target = write

    assert env._commit_c10_connector_latch(name, canonical) is True

    state = env._c10_connector_grasps["right"]
    np.testing.assert_allclose(written[-1], actual, atol=1e-10)
    assert not np.allclose(actual, canonical, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(
        env._c10_plug_latch_relative,
        np.linalg.inv(strip) @ actual,
        atol=1e-10,
    )
    assert state["phase"] == "latched_handoff"
    assert state["hand_target_installed"] is False
    np.testing.assert_allclose(state["hand_target_matrix"], palm, atol=1e-10)

    moved_palm = _matrix((0.14, -0.13, 1.04), yaw_degrees=-5.0)
    env._robot_link_pose_matrix = lambda side: (
        moved_palm if side == "right" else None
    )
    env._raw_hand_command_max = lambda side: (
        1.2 if side == "right" else 0.0
    )
    env._update_c10_connection_latches = lambda: None
    env._update_c10_connector_assist()

    carried = moved_palm @ hand_to_object
    np.testing.assert_allclose(written[-1], carried, atol=1e-10)
    np.testing.assert_allclose(matrices[name], carried, atol=1e-10)
    np.testing.assert_allclose(
        np.linalg.inv(moved_palm) @ matrices[name],
        hand_to_object,
        atol=1e-10,
    )
    assert state["hand_target_installed"] is False
    np.testing.assert_allclose(
        state["hand_target_matrix"],
        moved_palm,
        atol=1e-10,
    )


def test_c10_pending_plug_handoff_blocks_cable_guide_and_stability() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t02_connect_charger_and_cable",
        launch_isaac=False,
        image_obs=False,
    )
    strip = _matrix((0.13, -0.09, 0.90), yaw_degrees=17.0)
    plug = strip @ _C10_PLUG_SEATED_RELATIVE
    cable = plug @ _C10_CABLE_SEATED_RELATIVE
    poses = {
        "c10_charger_plug": env._pose_matrix_to_vector(plug),
        "c10_charging_cable": env._pose_matrix_to_vector(cable),
        "c10_power_strip": env._pose_matrix_to_vector(strip),
    }
    geometry = env._c10_connection_geometry(
        poses["c10_charger_plug"],
        poses["c10_charging_cable"],
        poses["c10_power_strip"],
    )
    state = {
        "name": "c10_charger_plug",
        "phase": "latched_handoff",
        "hand_target_installed": False,
    }
    env._c10_connector_grasps = {"right": state}
    env._c10_plug_latched = True
    env._c10_first_connected_seen = True
    env._c10_latch_stable_steps["c10_charging_cable"] = 2
    env._physics_body_pose = lambda body_name: poses[body_name].copy()
    commits: list[str] = []
    env._commit_c10_connector_latch = (
        lambda body_name, _target: commits.append(body_name) or True
    )

    assert geometry["second_connected_now"] is True
    assert (
        env._c10_guidance_trigger("c10_charging_cable", geometry)
        is False
    )
    env._update_c10_connection_latches()
    assert env._c10_latch_stable_steps["c10_charging_cable"] == 0
    assert commits == []

    state["hand_target_installed"] = True
    assert (
        env._c10_guidance_trigger("c10_charging_cable", geometry)
        is True
    )
    env._update_c10_connection_latches()
    assert env._c10_latch_stable_steps["c10_charging_cable"] == 1
    assert commits == []


@pytest.mark.parametrize("failure_stage", ["pose", "kinematic"])
def test_c10_write_failure_never_commits_a_logical_latch(
    failure_stage: str,
) -> None:
    env = HandsOnABCIsaacLabEnv(
        "t02_connect_charger_and_cable",
        launch_isaac=False,
        image_obs=False,
    )
    view = object()
    latch_target = _matrix((0.02, -0.01, 0.93), yaw_degrees=90.0)
    previous_matrix = latch_target.copy()
    previous_matrix[1, 3] -= 0.001
    original_pose = env._pose_matrix_to_vector(
        previous_matrix
    )
    env._poses["c10_charger_plug"] = original_pose.copy()
    env._stage = object()
    env._task_body_views = {"c10_charger_plug": view}
    env._c10_connector_grasps = {
        "right": {
            "name": "c10_charger_plug",
            "phase": "inserting",
        }
    }
    pose_writes: list[tuple[object, np.ndarray]] = []
    kinematic_writes: list[tuple[str, bool]] = []
    env._set_task_body_world_pose = lambda written_view, written_pose: (
        pose_writes.append(
            (
                written_view,
                np.asarray(written_pose, dtype=np.float64).copy(),
            )
        )
        or failure_stage != "pose"
    )
    env._set_c10_connector_kinematic = lambda name, enabled: (
        kinematic_writes.append((name, enabled))
        or failure_stage != "kinematic"
    )

    committed = env._commit_c10_connector_latch(
        "c10_charger_plug",
        latch_target,
    )

    assert committed is False
    assert env._c10_plug_latched is False
    assert env._c10_first_connected_seen is False
    assert env._c10_plug_latch_relative is None
    assert "right" in env._c10_connector_grasps
    np.testing.assert_allclose(
        env._poses["c10_charger_plug"],
        original_pose,
    )
    assert [written_view for written_view, _pose in pose_writes] == (
        [view] if failure_stage == "pose" else [view, view]
    )
    if failure_stage == "kinematic":
        np.testing.assert_allclose(pose_writes[-1][1], original_pose)
    assert kinematic_writes == (
        []
        if failure_stage == "pose"
        else [
            ("c10_charger_plug", True),
            ("c10_charger_plug", False),
        ]
    )
