from __future__ import annotations

import json
import os
from pathlib import Path
import runpy

import numpy as np

from handson.runtime.collection_support import ISAACSIM_COLLECTION_TASKS
from handson.sim.isaaclab.abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _FU006_CHARGER_RESET_XY_BOUNDS,
    _FU006_DRAWER_CAVITY_MAX,
    _FU006_DRAWER_CAVITY_MIN,
    _FU006_DRAWER_FLOOR_REPAIR_SOURCE_MAX,
    _FU006_DRAWER_FLOOR_REPAIR_SOURCE_MIN,
    _FU006_DRAWER_FLOOR_REPAIR_CONTACT_OFFSET,
    _FU006_DRAWER_FLOOR_REPAIR_REST_OFFSET,
    _FU006_DRAWER_INITIAL_POSITION,
    _FU006_DRAWER_OPEN_MIN,
    _FU006_DRAWER_UPPER_LIMIT,
    _FU006_TASK_NAME,
    supported_abc_collection_tasks,
)
from handson.sim.isaaclab.recording_origin import (
    DEFAULT_STABLE_SECONDS,
    STATIC_POSE_FALLBACK_TASKS,
    stable_recording_origin_kwargs_for_task,
)
from handson.tasks.mappings import CONFIG_MAPPING


def _interface_env(*, randomize: bool = False, seed: int = 7):
    return HandsOnABCIsaacLabEnv(
        _FU006_TASK_NAME,
        seed=seed,
        randomize=randomize,
        image_obs=False,
        launch_isaac=False,
    )


def test_fu006_identity_runtime_and_exact_interface_contract() -> None:
    assert _FU006_TASK_NAME in CONFIG_MAPPING
    assert _FU006_TASK_NAME in ISAACSIM_COLLECTION_TASKS
    assert _FU006_TASK_NAME in CONFIG_MAPPING
    config = CONFIG_MAPPING[_FU006_TASK_NAME]()
    assert _FU006_DRAWER_INITIAL_POSITION == 0.35 * 0.28
    assert _FU006_DRAWER_OPEN_MIN == 0.35 * 0.20
    assert _FU006_DRAWER_UPPER_LIMIT == 0.35 * 0.7760000228881836
    assert config.task_id == "46"
    assert config.proprio_keys == [
        "tcp_pose",
        "gripper_pose",
        "charger_pose",
        "drawer_pose",
        "drawer_joint_state",
        "fu006_task_state",
        "table_delta_height",
    ]
    env = _interface_env()
    try:
        assert env.action_space.shape == (54,)
        assert env.control_dt == 1.0 / 30.0
        assert env._fu006_stability_steps_required() == 15
        assert env.observation_space["state"]["charger_pose"].shape == (7,)
        assert env.observation_space["state"]["drawer_pose"].shape == (7,)
        assert env.observation_space["state"]["drawer_joint_state"].shape == (2,)
        assert env.observation_space["state"]["fu006_task_state"].shape == (14,)
    finally:
        env.close()


def test_fu006_floor_repair_matches_exact_authored_faces_and_cavity() -> None:
    floor_min = 0.35 * _FU006_DRAWER_FLOOR_REPAIR_SOURCE_MIN
    floor_max = 0.35 * _FU006_DRAWER_FLOOR_REPAIR_SOURCE_MAX
    assert np.all(floor_max > floor_min)
    assert floor_min[0] < _FU006_DRAWER_CAVITY_MIN[0]
    assert floor_max[0] > _FU006_DRAWER_CAVITY_MAX[0]
    assert floor_min[2] < _FU006_DRAWER_CAVITY_MIN[2]
    assert floor_max[2] > _FU006_DRAWER_CAVITY_MAX[2]
    assert np.isclose(
        _FU006_DRAWER_CAVITY_MIN[1] - floor_max[1], 0.001, atol=2e-6
    )
    assert 0.0 < _FU006_DRAWER_FLOOR_REPAIR_REST_OFFSET
    assert (
        _FU006_DRAWER_FLOOR_REPAIR_REST_OFFSET
        < _FU006_DRAWER_FLOOR_REPAIR_CONTACT_OFFSET
    )
    assert (
        floor_max[1] + _FU006_DRAWER_FLOOR_REPAIR_REST_OFFSET
        > _FU006_DRAWER_CAVITY_MIN[1]
    )


def test_fu006_official_recorder_requires_strict_static_pose_proof() -> None:
    assert _FU006_TASK_NAME in STATIC_POSE_FALLBACK_TASKS
    profile = stable_recording_origin_kwargs_for_task(_FU006_TASK_NAME)
    assert profile["stability_metric"] == "velocity_or_static_pose"
    assert profile["stability_sample_seconds"] == 0.0
    assert profile["pose_translation_limit"] == 1.0e-6
    assert profile["pose_rotation_limit"] == 1.0e-5
    assert profile.get("stable_seconds", DEFAULT_STABLE_SECONDS) == 0.5
    assert profile["post_settle_sleeping"] is False
    assert profile["stabilization_velocity_clamp"] is False
    assert profile["reset_translation_drift_limit"] is None
    assert profile["contact_penetration_limit"] == 0.0005


def test_fu006_is_physx_only_and_has_four_operator_conditions(monkeypatch) -> None:
    config = CONFIG_MAPPING[_FU006_TASK_NAME]()
    try:
        config.get_environment(
            sim_backend="mujoco", launch_isaac=False, image_obs=False
        )
    except ValueError as exc:
        assert "Isaac Sim 5.1 PhysX" in str(exc)
    else:
        raise AssertionError("FU006 accepted a non-Isaac backend")
    monkeypatch.setenv("HANDSON_ENABLE_ISAACSIM_NEWTON", "1")
    try:
        config.get_environment(
            sim_backend="isaacsim", launch_isaac=False, image_obs=False
        )
    except ValueError as exc:
        assert "forbids the Newton backend" in str(exc)
    else:
        raise AssertionError("FU006 accepted Newton")
    preview_source = (
        Path(__file__).parents[1]
        / "handson"
        / "sim"
        / "isaaclab"
        / "task_preview.py"
    ).read_text(encoding="utf-8")
    assert preview_source.count('"t46_store_charger"') >= 3
    assert "Left hand physically holds the lower drawer open" in preview_source
    assert "fu006_internal_collision_groups" in preview_source
    assert preview_source.count("CreateFilteredGroupsRel().AddTarget") >= 2
    assert "drive.GetStiffnessAttr().Set(0.0)" in preview_source
    assert "drive.GetDampingAttr().Set(1.0)" in preview_source
    assert "drive.GetTargetVelocityAttr().Set(0.0)" in preview_source


def test_fu006_seeded_charger_reset_is_bounded_and_drawer_clear() -> None:
    np.testing.assert_allclose(
        _FU006_CHARGER_RESET_XY_BOUNDS,
        np.asarray([[-0.39, -0.24], [-0.35, -0.20]]),
    )
    first = _interface_env(randomize=True, seed=19)
    second = _interface_env(randomize=True, seed=19)
    try:
        for env in (first, second):
            env._reset_robot_body_points = lambda _side: (
                np.asarray([0.30, 0.45, 1.20], dtype=np.float64),
            )
        xy_first = first._task_specific_reset_xy_positions()["fu006_charger"]
        xy_second = second._task_specific_reset_xy_positions()["fu006_charger"]
        assert np.array_equal(xy_first, xy_second)
        assert np.all(xy_first >= _FU006_CHARGER_RESET_XY_BOUNDS[0])
        assert np.all(xy_first <= _FU006_CHARGER_RESET_XY_BOUNDS[1])
        assert first._task_specific_reset_layout_margin(
            {"fu006_charger": xy_first},
            (np.asarray([0.30, 0.45, 1.20]),),
        ) >= 0.005
    finally:
        first.close()
        second.close()


def test_fu006_requires_hold_then_containment_release_and_stability() -> None:
    env = _interface_env()
    try:
        # Event 00000008, frame 350, from the 2026-08-25 physical collection.
        # Right-hand/charger contact had been absent for 93 consecutive frames
        # and the right palm was 0.359 m from the charger.  Reorder the recorded
        # physical joints into native finger-major order.  This real release
        # proves that an opened glove is not a zero-centred joint vector.
        real_released_command = np.asarray(
            [
                0.95804077, 0.00138718, 0.18655945, -0.06907596,
                0.65827245, -0.19983664, 0.00466947, 0.00394352,
                0.58686477, -0.35443017, 0.05152959, 0.15471867,
                0.54345185, -0.30861399, 0.23832732, 0.19397746,
                0.17301163, -0.14270443, 0.42207518, 0.68976021,
            ],
            dtype=np.float64,
        )
        real_released_actual = np.asarray(
            [
                0.95802128, 0.00132277, 0.18657604, -0.06916231,
                0.65830719, -0.19978034, 0.00464609, 0.00398071,
                0.58689857, -0.35444087, 0.05151172, 0.15471351,
                0.54350114, -0.30859545, 0.23820947, 0.19396950,
                0.17306694, -0.14271715, 0.42206606, 0.68966311,
            ],
            dtype=np.float64,
        )
        assert np.max(np.abs(real_released_command)) > 0.04
        assert np.max(np.abs(real_released_actual)) > 0.06

        env._fu006_left_drawer_contact_view = object()
        env._fu006_right_charger_contact_view = object()
        env._physics_dof_position = lambda _name: (0.28, 0.28)
        env._physics_dof_velocity = lambda _name: 0.0
        env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)
        left_contact_force = [0.10]
        right_contact_force = [0.0]
        env._c04_contact_force = lambda view: (
            left_contact_force[0]
            if view is env._fu006_left_drawer_contact_view
            else right_contact_force[0]
        )
        env._hand_command_max = lambda side: (
            0.20
            if side == "left"
            else float(np.max(np.abs(real_released_command)))
        )
        physical_left = type("PhysicalLeft", (), {})()
        physical_left.data = type("PhysicalLeftData", (), {})()
        physical_left.data.joint_pos = np.zeros((1, 20), dtype=np.float64)
        env._robots["left"] = physical_left
        env._hand_joint_ids["left"] = list(range(20))
        physical_right = type("PhysicalRight", (), {})()
        physical_right.data = type("PhysicalRightData", (), {})()
        physical_right.data.joint_pos = real_released_actual.reshape(1, 20)
        env._robots["right"] = physical_right
        env._hand_joint_ids["right"] = list(range(20))
        right_hand_point = np.asarray(
            [0.25, 0.0, 0.90], dtype=np.float64
        )
        env._a06_hand_points = lambda side: (
            right_hand_point.copy(),
        )
        env._poses["fu006_drawer"] = np.asarray(
            [0.0, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        env._poses["fu006_charger"] = np.asarray(
            [0.25, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        blocked = env._fu006_metrics(update_events=False)
        assert blocked["fu006_containment_margin_x"] < 0.0
        assert blocked["fu006_left_hand_command_max"] == 0.20
        assert blocked["fu006_left_hand_actual_max"] == 0.0
        assert not blocked["fu006_left_hand_closed_actual"]
        assert not blocked["fu006_left_hold_now"]

        physical_left.data.joint_pos.fill(0.20)
        first = env._fu006_metrics(update_events=True)
        assert first["fu006_left_hand_command_max"] == 0.20
        assert first["fu006_left_hand_actual_max"] == 0.20
        assert first["fu006_left_hand_closed_actual"]
        assert first["fu006_left_hold_seen"]
        assert not first["fu006_contained_seen"]
        assert not first["fu006_order_violation"]

        env._poses["fu006_charger"] = np.asarray(
            [
                0.015478585,
                -0.02920268,
                1.115791535,
                np.sqrt(0.5),
                np.sqrt(0.5),
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        right_hand_point[:] = env._poses["fu006_charger"][:3]
        blocked_release = env._fu006_metrics(update_events=True)
        assert blocked_release["fu006_charger_contained"]
        np.testing.assert_allclose(
            [
                blocked_release["fu006_containment_margin_x"],
                blocked_release["fu006_containment_margin_y"],
                blocked_release["fu006_containment_margin_z"],
            ],
            [0.102893625, 0.00840088, 0.112166645],
            atol=1.0e-6,
        )
        assert blocked_release["fu006_right_hand_command_max"] > 0.95
        assert blocked_release["fu006_right_hand_actual_max"] > 0.95
        assert not blocked_release["fu006_right_hand_open_actual"]
        assert not blocked_release["fu006_right_released_now"]
        assert not blocked_release["fu006_release_seen"]

        # Distance alone is insufficient while live contact is still active.
        right_hand_point[:] += np.asarray([0.10, 0.0, 0.0])
        right_contact_force[0] = 2.0e-5
        contact_blocked = env._fu006_metrics(update_events=False)
        assert contact_blocked["fu006_right_hand_clearance"] == 0.10
        assert not contact_blocked["fu006_right_released_now"]

        # Once both independent right-hand signals agree, the real non-zero
        # released pose must no longer block the right release event.  The
        # stability timer must remain at zero while the left hand still holds.
        right_contact_force[0] = 0.0
        for _ in range(env._fu006_stability_steps_required()):
            held = env._fu006_metrics(update_events=True)
        assert held["fu006_right_released_now"]
        assert held["fu006_release_seen"]
        assert not held["fu006_left_released_now"]
        assert held["fu006_stability_steps"] == 0
        assert not held["fu006_success_now"]

        # Only physical left-hand contact loss starts the consecutive 0.5 s
        # stability window.  A closed/non-zero glove pose is diagnostic only.
        left_contact_force[0] = 0.0
        for _ in range(env._fu006_stability_steps_required()):
            final = env._fu006_metrics(update_events=True)
        assert final["fu006_charger_contained"]
        assert final["fu006_right_hand_actual_max"] > 0.95
        assert not final["fu006_right_hand_open_actual"]
        assert final["fu006_right_released_now"]
        assert final["fu006_release_seen"]
        assert final["fu006_left_released_now"]
        assert final["fu006_success_now"]
        assert not final["fu006_failed"]
    finally:
        env.close()


def test_fu006_containment_row_tracks_the_visible_drawer_without_left_hold() -> None:
    env = _interface_env()
    try:
        env._physics_dof_position = lambda _name: (
            _FU006_DRAWER_INITIAL_POSITION,
            _FU006_DRAWER_INITIAL_POSITION,
        )
        env._physics_dof_velocity = lambda _name: 0.0
        env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)
        env._hand_command_max = lambda _side: 0.0
        env._a06_hand_points = lambda _side: ()
        env._poses["fu006_drawer"] = np.asarray(
            [0.0, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        env._poses["fu006_charger"] = np.asarray(
            [
                0.015478585,
                -0.02920268,
                1.115791535,
                np.sqrt(0.5),
                np.sqrt(0.5),
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        metrics = env._fu006_metrics(update_events=False)
        rows = env._fu006_success_condition_rows()

        assert metrics["fu006_charger_contained"]
        assert not metrics["fu006_left_hold_seen"]
        assert not rows[0]["ok"]
        assert rows[1]["key"] == "b320_fu006_charger_contained"
        assert rows[1]["ok"]
    finally:
        env.close()


def test_fu006_release_fails_closed_without_live_hand_geometry() -> None:
    env = _interface_env()
    try:
        env._fu006_right_charger_contact_view = object()
        env._c04_contact_force = lambda _view: 0.0
        env._physics_dof_position = lambda _name: (0.28, 0.28)
        env._physics_dof_velocity = lambda _name: 0.0
        env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)
        env._a06_hand_points = lambda _side: ()
        env._poses["fu006_drawer"] = np.asarray(
            [0.0, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        env._poses["fu006_charger"] = np.asarray(
            [
                0.015478585,
                -0.02920268,
                1.115791535,
                np.sqrt(0.5),
                np.sqrt(0.5),
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        metrics = env._fu006_metrics(update_events=False)

        assert metrics["fu006_charger_contained"]
        assert metrics["fu006_right_hand_clearance"] == 0.0
        assert not metrics["fu006_right_released_now"]
    finally:
        env.close()


def test_fu006_premature_drawer_closure_is_terminal_failure() -> None:
    env = _interface_env()
    try:
        env._fu006_left_hold_seen = True
        env._fu006_left_drawer_contact_view = object()
        env._fu006_right_charger_contact_view = object()
        env._physics_dof_position = lambda _name: (
            _FU006_DRAWER_OPEN_MIN * 0.25,
            0.28,
        )
        env._physics_dof_velocity = lambda _name: 0.0
        env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)
        env._c04_contact_force = lambda _view: 0.0
        env._hand_command_max = lambda _side: 0.0
        env._a06_hand_points = lambda _side: ()
        env._poses["fu006_drawer"] = np.asarray(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        env._poses["fu006_charger"] = np.asarray(
            [0.40, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        metrics = env._fu006_metrics(update_events=True)
        assert metrics["fu006_failed"]
        assert env._compute_failure()
    finally:
        env.close()


def test_fu006_packaged_assets_are_complete() -> None:
    xml_dir = Path(__file__).parents[1] / "handson" / "sim" / "envs" / "xmls"
    root = xml_dir / "assets" / "bench320_fu006"
    assert (root / "dv_drawer_45575" / "45575.usd").stat().st_size > 1024
    assert (root / "dv_drawer_45575" / "configuration").is_dir()
    assert (root / "dexverse_charger" / "model_charger.usd").stat().st_size > 1024
    assert (root / "dexverse_charger" / "resource").is_dir()


def test_fu006_runner_isolates_shared_robot_import_meshes(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "canonical_xmls"
    wrist = (
        source
        / "wrist_fisheye/meshes/wrist_fisheye_flange_assembly.stl"
    )
    adapter = (
        source
        / "wuji_attachment/direct_adapter/meshes/Direct-Adapter-Mount-binary.stl"
    )
    wrist.parent.mkdir(parents=True)
    adapter.parent.mkdir(parents=True)
    wrist.write_bytes(b"canonical-wrist")
    adapter.write_bytes(b"canonical-adapter")
    (source / "table_arena").mkdir()
    robot_xml = (
        '<mujoco><asset><mesh file="wrist_fisheye/meshes/'
        'wrist_fisheye_flange_assembly.stl"/><mesh file="wuji_attachment/'
        'direct_adapter/meshes/Direct-Adapter-Mount-binary.stl"/></asset>'
        "</mujoco>"
    )
    for side in ("left", "right"):
        (source / f"rokae_wuji1_{side}.xml").write_text(
            robot_xml, encoding="utf-8"
        )

    workspace = tmp_path / "workspace"
    monkeypatch.setenv("HANDSON_ISAACLAB_XML_DIR", str(source))
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace))
    runner = runpy.run_path(
        str(Path(__file__).parents[2] / "scripts/fu006_vertical_slice.py")
    )
    paths = runner["_configure_task_private_isaac_runtime"](
        tmp_path / "evidence"
    )

    overlay = Path(paths["private_xml_overlay"])
    assert Path(os.environ["HANDSON_ISAACLAB_XML_DIR"]) == overlay
    assert Path(os.environ["HANDSON_ISAACLAB_CACHE"]) == Path(
        paths["private_cache_root"]
    )
    assert (overlay / "table_arena").is_symlink()
    for side in ("left", "right"):
        xml_path = overlay / f"rokae_wuji1_{side}.xml"
        assert xml_path.is_file() and not xml_path.is_symlink()
        content = xml_path.read_text(encoding="utf-8")
        for mesh in (wrist, adapter):
            alias = overlay / "fu006_robot_mesh_aliases" / side / mesh.name
            assert alias.read_bytes() == mesh.read_bytes()
            assert str(alias.relative_to(overlay)) in content
    assert (
        overlay / "fu006_robot_mesh_aliases/left/wrist_fisheye_flange_assembly.stl"
    ).resolve() != (
        overlay / "fu006_robot_mesh_aliases/right/wrist_fisheye_flange_assembly.stl"
    ).resolve()

    schedule = runner["_AUTOMATIC_PHASE_SCHEDULE"]
    assert sum(frames for _name, frames in schedule) == 2335
    observation = {
        "state": {
            key: np.asarray([index], dtype=np.float64)
            for index, key in enumerate(runner["STATE_KEYS"])
        }
    }
    report_path = runner["_write_collection_failure_report"](
        tmp_path / "episode_candidate",
        runtime={"runtime_family": "isaac_sim_5_1"},
        seed=320006,
        reset_gate={
            "name": "scaled_drawer_q_reset_readback",
            "passed": True,
        },
        actions_generated=256,
        transitions=[{"action": np.zeros(54, dtype=np.float32)}],
        initial_observation=observation,
        waypoints={"left_hold": np.zeros(7, dtype=np.float64)},
        observation=observation,
        final_info={"fu006_failed": np.bool_(False), "succeed": False},
        task_trace=[
            {
                "step": 1,
                "fu006_drawer_open": True,
                "fu006_left_hold_now": False,
                "fu006_left_hold_seen": False,
                "fu006_left_released_now": False,
                "fu006_contained_seen": False,
                "fu006_release_seen": False,
                "fu006_success_now": False,
            }
        ],
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path.name == "episode_candidate_failure_report.json"
    assert not (tmp_path / "episode_candidate").exists()
    assert report["steps_executed"] == 1
    assert report["final_task_info"] == {"fu006_failed": False}
    assert report["physical_gate_progress"]["first_unmet_gate"] == (
        "left_handle_contact"
    )
    assert report["replay_eligible"] is False
    assert report["state_writes_after_action_zero"] is False


def test_fu006_automatic_policy_uses_exact_asset_frame_contacts(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "canonical_xmls"
    source.mkdir()
    (source / "table_arena").mkdir()
    for side in ("left", "right"):
        (source / f"rokae_wuji1_{side}.xml").write_text(
            "<mujoco><asset/></mujoco>", encoding="utf-8"
        )
    monkeypatch.setenv("HANDSON_ISAACLAB_XML_DIR", str(source))
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
    runner = runpy.run_path(
        str(Path(__file__).parents[2] / "scripts/fu006_vertical_slice.py")
    )

    right_home = np.asarray(
        [-0.10, -0.16, 1.07, 0.0, 1.0, 0.0, 0.0], dtype=np.float64
    )
    left_home = np.asarray(
        [-0.10, 0.16, 1.07, 0.0, 1.0, 0.0, 0.0], dtype=np.float64
    )
    measured_right_home = np.asarray(
        [
            -0.40361,
            -0.41339,
            0.99032,
            0.65672,
            0.24221,
            0.69504,
            -0.16423,
        ],
        dtype=np.float64,
    )
    measured_left_home = np.asarray(
        [
            -0.40428,
            0.41313,
            0.99229,
            0.65702,
            -0.24172,
            0.69493,
            0.16423,
        ],
        dtype=np.float64,
    )
    measured_right_home[3:7] /= np.linalg.norm(measured_right_home[3:7])
    measured_left_home[3:7] /= np.linalg.norm(measured_left_home[3:7])
    charger = np.asarray(
        [-0.34, -0.22, 0.90750407, 0.9945219, 0.0, 0.0, -0.1045285],
        dtype=np.float64,
    )
    drawer = np.asarray(
        [-0.13991127, 0.01547070, 1.38044908, 0.5, 0.5, -0.5, -0.5],
        dtype=np.float64,
    )
    observation = {
        "state": {
            "tcp_pose": np.concatenate([right_home, left_home]),
            "charger_pose": charger,
            "drawer_pose": drawer,
            "drawer_joint_state": np.asarray(
                [_FU006_DRAWER_INITIAL_POSITION, 0.0], dtype=np.float64
            ),
        }
    }
    waypoints = runner["_asset_frame_waypoints"](
        observation,
        measured_right_home=measured_right_home,
        measured_left_home=measured_left_home,
    )
    right_offset = runner["_rotate_wxyz"](
        waypoints["right_grasp"][3:7],
        runner["_RIGHT_POWER_MAIN_FINGERTIP_LOCAL"],
    )
    left_offset = runner["_rotate_wxyz"](
        waypoints["left_contact"][3:7],
        runner["_LEFT_STABILIZE_MAIN_FINGERTIP_LOCAL"],
    )
    np.testing.assert_allclose(waypoints["right_home"], measured_right_home)
    np.testing.assert_allclose(waypoints["left_home"], measured_left_home)
    expected_carry_orientation = runner["_multiply_wxyz"](
        runner["_yaw_only_wxyz"](charger[3:7]),
        measured_right_home[3:7],
    )
    np.testing.assert_allclose(
        waypoints["right_grasp"][3:7], expected_carry_orientation
    )
    np.testing.assert_allclose(
        waypoints["left_contact"][3:7], runner["_VERTICAL_FINGERS_WXYZ"]
    )
    np.testing.assert_allclose(
        waypoints["right_approach"][3:7], expected_carry_orientation
    )
    np.testing.assert_allclose(
        waypoints["right_initial_lift"][3:7], expected_carry_orientation
    )
    np.testing.assert_allclose(
        waypoints["right_grasp"][:3] + right_offset,
        charger[:3],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        waypoints["left_contact"][:3] + left_offset,
        runner["_drawer_handle_contact_world"](drawer),
        atol=1e-12,
    )
    interpolated_left = runner["_lerp_pose"](
        measured_left_home,
        waypoints["left_contact"],
        runner["_LEFT_HANDLE_CONTACT_FRACTION"],
    )
    np.testing.assert_allclose(
        waypoints["left_hold"][:3],
        waypoints["left_contact"][:3]
        + runner["_LEFT_HANDLE_HOLD_OFFSET_WORLD"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        waypoints["left_hold"][3:7], interpolated_left[3:7], atol=1e-12
    )
    np.testing.assert_allclose(
        waypoints["left_stage"][[0, 2, 3, 4, 5, 6]],
        waypoints["left_hold"][[0, 2, 3, 4, 5, 6]],
        atol=1e-12,
    )
    assert waypoints["left_stage"][1] == measured_left_home[1]
    np.testing.assert_allclose(
        waypoints["left_pull"][1:7], waypoints["left_hold"][1:7]
    )
    assert waypoints["left_pull"][0] == (
        waypoints["left_hold"][0] - runner["_LEFT_HANDLE_PULL_M"]
    )
    np.testing.assert_allclose(
        runner["_drawer_handle_contact_world"](drawer),
        np.asarray([-0.52169865, -0.03173964, 1.35434231]),
        atol=2e-8,
    )
    assert runner["_DRAWER_ABOVE_CENTER_LOCAL"][1] > (
        runner["_DRAWER_DESIRED_CENTER_LOCAL"][1] + 0.0419
    )
    assert np.isclose(runner["_DRAWER_OUTSIDE_ABOVE_HEIGHT_LOCAL_M"], 0.085)
    assert np.all(
        _FU006_DRAWER_CAVITY_MIN < runner["_DRAWER_DESIRED_CENTER_LOCAL"]
    )
    assert np.all(
        runner["_DRAWER_DESIRED_CENTER_LOCAL"] < _FU006_DRAWER_CAVITY_MAX
    )
    assert waypoints["right_approach"][2] == (
        waypoints["right_grasp"][2] + 0.10
    )
    np.testing.assert_allclose(
        waypoints["right_approach"][[0, 1, 3, 4, 5, 6]],
        waypoints["right_grasp"][[0, 1, 3, 4, 5, 6]],
        atol=1e-12,
    )
    assert waypoints["right_initial_lift"][2] == (
        waypoints["right_grasp"][2] + runner["_RIGHT_INITIAL_LIFT_M"]
    )
    np.testing.assert_allclose(
        waypoints["right_initial_lift"][[0, 1, 3, 4, 5, 6]],
        waypoints["right_grasp"][[0, 1, 3, 4, 5, 6]],
        atol=1e-12,
    )

    post_lift_charger = charger.copy()
    post_lift_charger[2] += runner["_RIGHT_INITIAL_LIFT_M"]
    post_lift_charger[3:7] = np.asarray(
        [0.949, -0.007, -0.238, -0.205], dtype=np.float64
    )
    post_lift_charger[3:7] /= np.linalg.norm(post_lift_charger[3:7])
    post_lift_observation = {
        "state": {
            "charger_pose": post_lift_charger,
            "drawer_pose": drawer,
        }
    }
    measured_post_lift = waypoints["right_initial_lift"].copy()
    flatten = runner["_flatten_grasp_waypoint"](
        post_lift_observation, measured_post_lift
    )
    flat_object_orientation = runner["_flat_carry_orientation_wxyz"](
        post_lift_charger[3:7]
    )
    expected_flat_object_orientation = np.asarray(
        [
            np.cos(0.5 * runner["_FLAT_CARRY_WORLD_YAW_RAD"]),
            0.0,
            0.0,
            np.sin(0.5 * runner["_FLAT_CARRY_WORLD_YAW_RAD"]),
        ],
        dtype=np.float64,
    )
    assert abs(
        float(np.dot(flat_object_orientation, expected_flat_object_orientation))
    ) > 0.999999
    world_adjustment = runner["_multiply_wxyz"](
        flat_object_orientation,
        runner["_inverse_wxyz"](post_lift_charger[3:7]),
    )
    expected_flatten_palm = runner["_multiply_wxyz"](
        world_adjustment, measured_post_lift[3:7]
    )
    expected_flatten_position = post_lift_charger[:3] + runner[
        "_rotate_wxyz"
    ](
        world_adjustment,
        measured_post_lift[:3] - post_lift_charger[:3],
    )
    np.testing.assert_allclose(
        flatten[:3], expected_flatten_position, atol=1e-12
    )
    assert np.isclose(
        np.linalg.norm(flatten[:3] - post_lift_charger[:3]),
        np.linalg.norm(measured_post_lift[:3] - post_lift_charger[:3]),
    )
    assert abs(float(np.dot(flatten[3:7], expected_flatten_palm))) > 0.999999
    flattened_object = runner["_multiply_wxyz"](
        world_adjustment, post_lift_charger[3:7]
    )
    assert abs(float(np.dot(flattened_object, flat_object_orientation))) > 0.999999

    post_flat_observation = {
        "state": {
            "charger_pose": np.concatenate(
                [post_lift_charger[:3], flat_object_orientation]
            ),
            "drawer_pose": drawer,
        }
    }
    lift_carry = runner["_adaptive_carry_waypoints"](
        post_flat_observation, flatten
    )
    cavity_turn = runner["_cavity_turn_grasp_waypoint"](
        post_flat_observation, flatten
    )
    cavity_orientation = runner["_cavity_carry_orientation_wxyz"](
        post_flat_observation["state"]["charger_pose"][3:7]
    )
    cavity_adjustment = runner["_multiply_wxyz"](
        cavity_orientation,
        runner["_inverse_wxyz"](
            post_flat_observation["state"]["charger_pose"][3:7]
        ),
    )
    turned_object = runner["_multiply_wxyz"](
        cavity_adjustment,
        post_flat_observation["state"]["charger_pose"][3:7],
    )
    assert abs(float(np.dot(turned_object, cavity_orientation))) > 0.999999
    carry = runner["_adaptive_carry_waypoints"](
        post_flat_observation, cavity_turn
    )
    drawer_inverse = drawer[3:7].copy()
    drawer_inverse[1:4] *= -1.0
    live_contact = runner["_nominal_fingertip_world"](
        cavity_turn, runner["_RIGHT_POWER_MAIN_FINGERTIP_LOCAL"]
    )
    object_contact_offset_local = runner["_rotate_wxyz"](
        drawer_inverse, post_lift_charger[:3] - live_contact
    )
    charger_local = runner["_rotate_wxyz"](
        drawer_inverse, post_lift_charger[:3] - drawer[:3]
    )
    expected_centers = {
        "right_stage": np.asarray(
            [
                runner["_DRAWER_SAFE_LIFT_CENTER_LOCAL_X"],
                charger_local[1],
                charger_local[2]
                + runner["_DRAWER_SAFE_LIFT_LOCAL_Z_OFFSET_M"],
            ]
        ),
        "right_lift": np.asarray(
            [
                runner["_DRAWER_SAFE_LIFT_CENTER_LOCAL_X"],
                runner["_DRAWER_OUTSIDE_ABOVE_HEIGHT_LOCAL_M"],
                charger_local[2]
                + runner["_DRAWER_SAFE_LIFT_LOCAL_Z_OFFSET_M"],
            ]
        ),
        "drawer_above": runner["_DRAWER_ABOVE_CENTER_LOCAL"],
        "drawer_inside": runner["_DRAWER_DESIRED_CENTER_LOCAL"],
    }
    for name, expected_center in expected_centers.items():
        contact = runner["_nominal_fingertip_world"](
            carry[name], runner["_RIGHT_POWER_MAIN_FINGERTIP_LOCAL"]
        )
        contact_local = runner["_rotate_wxyz"](
            drawer_inverse, contact - drawer[:3]
        )
        np.testing.assert_allclose(
            contact_local + object_contact_offset_local,
            expected_center,
            atol=1e-12,
        )
    assert np.isclose(runner["_DRAWER_SAFE_LIFT_CENTER_LOCAL_X"], 0.200)
    assert np.isclose(runner["_DRAWER_SAFE_LIFT_LOCAL_Z_OFFSET_M"], 0.060)
    assert np.isclose(runner["_CAVITY_CARRY_YAW_OFFSET_RAD"], np.pi / 3.0)
    desired_world = drawer[:3] + runner["_rotate_wxyz"](
        drawer[3:7], runner["_DRAWER_DESIRED_CENTER_LOCAL"]
    )
    horizontal_correction = runner[
        "_closed_loop_grasp_translation_waypoint"
    ](post_flat_observation, cavity_turn, horizontal_only=True)
    full_correction = runner["_closed_loop_grasp_translation_waypoint"](
        post_flat_observation, cavity_turn, horizontal_only=False
    )
    expected_translation = desired_world - post_lift_charger[:3]
    np.testing.assert_allclose(
        horizontal_correction[:3] - cavity_turn[:3],
        np.asarray([*expected_translation[:2], 0.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        full_correction[:3] - cavity_turn[:3], expected_translation, atol=1e-12
    )
    limited_correction = runner[
        "_closed_loop_grasp_translation_waypoint"
    ](
        post_flat_observation,
        cavity_turn,
        horizontal_only=False,
        translation_limit_m=runner["_FINAL_RESIDUAL_CORRECTION_LIMIT_M"],
    )
    np.testing.assert_allclose(
        limited_correction[:3] - cavity_turn[:3],
        np.clip(
            expected_translation,
            -runner["_FINAL_RESIDUAL_CORRECTION_LIMIT_M"],
            runner["_FINAL_RESIDUAL_CORRECTION_LIMIT_M"],
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        runner["_desired_charger_center_world"](post_flat_observation),
        desired_world,
        atol=1e-12,
    )
    for waypoint in (
        *waypoints.values(),
        flatten,
        cavity_turn,
        *lift_carry.values(),
        *carry.values(),
    ):
        assert np.all(np.isfinite(waypoint))
        assert np.all(waypoint[:3] >= np.asarray([-0.75, -0.60, 0.89]))
        assert np.all(waypoint[:3] <= np.asarray([0.35, 0.60, 1.65]))

    prefix = runner["automatic_prefix_actions"](
        observation,
        measured_right_home=measured_right_home,
        measured_left_home=measured_left_home,
    )
    assert len(prefix) == 245
    flatten_actions = list(
        runner["_phase_actions"](
            prefix[-1][:7],
            prefix[-1][7:14],
            flatten,
            prefix[-1][7:14],
            prefix[-1][14:34],
            prefix[-1][34:54],
            prefix[-1][14:34],
            prefix[-1][34:54],
            runner["_phase_frames"]("right_flatten"),
        )
    )
    stage_actions = list(
        runner["_phase_actions"](
            flatten_actions[-1][:7],
            flatten_actions[-1][7:14],
            lift_carry["right_stage"],
            flatten_actions[-1][7:14],
            flatten_actions[-1][14:34],
            flatten_actions[-1][34:54],
            flatten_actions[-1][14:34],
            flatten_actions[-1][34:54],
            runner["_phase_frames"]("right_stage_outside_side"),
        )
    )
    lift_actions = runner["_adaptive_suffix_actions"](
        stage_actions[-1], lift_carry
    )[: runner["_phase_frames"]("right_lift_outside_side")]
    left_approach_actions = runner["_left_approach_actions"](
        lift_actions[-1], waypoints
    )
    handle = runner["_drawer_handle_contact_world"](drawer)
    measured_left_fingertips = {
        "near": handle + np.asarray([0.10, 0.05, -0.08]),
        "far": handle + np.asarray([0.30, 0.20, -0.20]),
    }
    measured_left_palm = left_approach_actions[-1][7:14].copy()
    measured_left_palm[:3] += np.asarray([0.06, 0.04, -0.03])
    corrected_left_hold = runner["_left_contact_correction_waypoint"](
        observation,
        left_approach_actions[-1][7:14],
        measured_left_palm,
        measured_left_fingertips,
    )
    np.testing.assert_allclose(
        corrected_left_hold[:3] - measured_left_palm[:3],
        np.asarray([-0.10, -0.05, 0.08]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        corrected_left_hold[3:7], left_approach_actions[-1][10:14]
    )
    left_correction_actions = list(
        runner["_phase_actions"](
            left_approach_actions[-1][:7],
            left_approach_actions[-1][7:14],
            left_approach_actions[-1][:7],
            corrected_left_hold,
            left_approach_actions[-1][14:34],
            left_approach_actions[-1][34:54],
            left_approach_actions[-1][14:34],
            left_approach_actions[-1][34:54],
            runner["_phase_frames"]("left_contact_correction"),
        )
    )
    refined_fingertips = {
        "near": handle + np.asarray([0.04, 0.03, -0.05]),
        "far": handle + np.asarray([0.20, 0.20, -0.20]),
    }
    refined_measured_palm = corrected_left_hold.copy()
    refined_measured_palm[:3] += np.asarray([0.02, 0.01, -0.02])
    refined_left_hold = runner["_left_contact_correction_waypoint"](
        observation,
        left_correction_actions[-1][7:14],
        refined_measured_palm,
        refined_fingertips,
    )
    np.testing.assert_allclose(
        refined_left_hold[:3] - refined_measured_palm[:3],
        np.asarray([-0.04, -0.03, 0.05]),
        atol=1e-12,
    )
    left_contact_actions = runner["_left_contact_actions"](
        left_correction_actions[-1], refined_left_hold
    )
    transfer_actions = runner["_adaptive_suffix_actions"](
        left_contact_actions[-1], carry, include_lift=False
    )
    actions = [
        *prefix,
        *flatten_actions,
        *stage_actions,
        *lift_actions,
        *left_approach_actions,
        *left_correction_actions,
        *left_contact_actions,
        *transfer_actions,
    ]
    assert len(actions) == 2335
    np.testing.assert_allclose(actions[59][:7], waypoints["right_approach"])
    np.testing.assert_allclose(actions[94][:7], waypoints["right_grasp"])
    np.testing.assert_allclose(actions[154][14:34], runner["_POWER_GRIP"])
    np.testing.assert_allclose(
        actions[244][:7], waypoints["right_initial_lift"]
    )
    np.testing.assert_allclose(actions[364][:7], flatten)
    np.testing.assert_allclose(actions[544][:7], lift_carry["right_stage"])
    np.testing.assert_allclose(actions[844][:7], lift_carry["right_lift"])
    np.testing.assert_allclose(actions[934][7:14], waypoints["left_stage"])
    np.testing.assert_allclose(actions[1024][7:14], waypoints["left_hold"])
    np.testing.assert_allclose(actions[1114][7:14], corrected_left_hold)
    np.testing.assert_allclose(actions[1204][7:14], refined_left_hold)
    np.testing.assert_allclose(actions[1249][34:54], runner["_STABILIZE_GRIP"])
    assert np.isclose(
        actions[1294][7], refined_left_hold[0] - runner["_LEFT_HANDLE_PULL_M"]
    )
    np.testing.assert_allclose(actions[1594][:7], carry["drawer_above"])
    np.testing.assert_allclose(actions[1714][:7], carry["drawer_above"])
    np.testing.assert_allclose(actions[1894][:7], carry["drawer_inside"])
    np.testing.assert_allclose(actions[2014][:7], carry["drawer_inside"])
    np.testing.assert_allclose(
        actions[2014][14:34], runner["_RIGHT_SEAT_GRIP"]
    )
    direct_seat = runner["_seat_actions"](
        actions[1894], carry["drawer_inside"]
    )
    assert len(direct_seat) == runner["_phase_frames"]("seat_in_drawer")
    np.testing.assert_allclose(
        direct_seat[-1][14:34], runner["_RIGHT_SEAT_GRIP"]
    )
    np.testing.assert_allclose(
        direct_seat[-1][34:54], actions[1894][34:54]
    )
    assert np.max(runner["_STABILIZE_GRIP"]) > 0.08
    assert np.max(runner["_STABILIZE_GRIP"]) < 0.30
    assert np.max(runner["_RIGHT_SEAT_GRIP"]) < np.max(
        runner["_POWER_GRIP"]
    )
    assert float(np.dot(actions[0][3:7], measured_right_home[3:7])) > 0.9999
    assert not np.array_equal(
        actions[0][3:7], waypoints["right_approach"][3:7]
    )
    np.testing.assert_allclose(actions[0][10:14], measured_left_home[3:7])
    assert not np.array_equal(actions[0][10:14], waypoints["left_hold"][3:7])
    np.testing.assert_allclose(
        actions[95][14:34], runner["_POWER_GRIP"] / 60.0
    )
    np.testing.assert_allclose(
        actions[1205][34:54], runner["_STABILIZE_GRIP"] / 45.0
    )
    np.testing.assert_allclose(
        actions[2114][14:34], np.zeros(20, dtype=np.float64)
    )
    np.testing.assert_allclose(
        actions[-1][34:54], np.zeros(20, dtype=np.float64)
    )
    assert runner["_automatic_phase_for_step"](1) == "right_approach"
    assert runner["_automatic_phase_for_step"](61) == "right_descend"
    assert runner["_automatic_phase_for_step"](96) == "right_close"
    assert runner["_automatic_phase_for_step"](156) == "right_initial_lift"
    assert runner["_automatic_phase_for_step"](246) == "right_flatten"
    assert runner["_automatic_phase_for_step"](366) == "right_stage_outside_side"
    assert runner["_automatic_phase_for_step"](546) == "right_lift_outside_side"
    assert runner["_automatic_phase_for_step"](846) == "left_stage"
    assert runner["_automatic_phase_for_step"](936) == "left_approach"
    assert runner["_automatic_phase_for_step"](1026) == "left_contact_correction"
    assert runner["_automatic_phase_for_step"](1116) == "left_contact_refine"
    assert runner["_automatic_phase_for_step"](1206) == "left_close"
    assert runner["_automatic_phase_for_step"](1251) == "left_pull"
    assert runner["_automatic_phase_for_step"](1296) == "cross_side_above"
    assert runner["_automatic_phase_for_step"](1596) == "align_above_drawer"
    assert runner["_automatic_phase_for_step"](1716) == "lower_into_drawer"
    assert runner["_automatic_phase_for_step"](1896) == "seat_in_drawer"
    assert runner["_automatic_phase_for_step"](2016) == "right_release"
    assert runner["_automatic_phase_for_step"](2335) == "settle"


def test_fu006_success_artifact_keeps_live_physical_gate_telemetry() -> None:
    runner = runpy.run_path(
        str(Path(__file__).parents[2] / "scripts/fu006_vertical_slice.py")
    )

    class LiveDrawer:
        def __init__(self, q: float) -> None:
            self.q = q

        def _physics_dof_position(
            self, name: str
        ) -> tuple[float, float] | None:
            assert name == "joint_1"
            return self.q, 0.098

        def _physics_dof_velocity(self, name: str) -> float | None:
            assert name == "joint_1"
            return 0.0

    reset_observation = {
        "state": {"drawer_joint_state": np.asarray([0.098, 0.0])}
    }
    reset_gate = runner["_scaled_drawer_reset_gate"](
        LiveDrawer(0.098), reset_observation
    )
    assert reset_gate["name"] == "scaled_drawer_q_reset_readback"
    assert reset_gate["passed"] is True
    try:
        runner["_scaled_drawer_reset_gate"](
            LiveDrawer(0.28), reset_observation
        )
    except RuntimeError as error:
        assert "scaled drawer reset gate failed" in str(error)
    else:
        raise AssertionError("unscaled physical drawer q passed the reset gate")

    trace: list[dict[str, object]] = []
    for step, phase in ((1, "right_approach"), (2, "right_approach")):
        item: dict[str, object] = {"step": step, "phase": phase}
        for field in runner["_PHYSICAL_GATE_TELEMETRY_FIELDS"]:
            item[field] = (
                np.arange(7, dtype=np.float64)
                if field.endswith("palm_pose")
                or field in {"charger_pose", "drawer_pose"}
                else np.arange(3, dtype=np.float64)
                if field == "drawer_handle_contact_world"
                else np.float64(step)
                if field
                in {
                    "fu006_containment_margin",
                    "fu006_containment_margin_x",
                    "fu006_containment_margin_y",
                    "fu006_containment_margin_z",
                }
                else np.bool_(step == 2)
                if field == "fu006_success_now"
                else np.bool_(False)
                if field.startswith("fu006_")
                and not field.endswith("contact_force")
                else np.float64(step)
            )
        trace.append(item)

    for item in trace:
        item["fu006_drawer_open"] = True
        item["fu006_left_hold_now"] = True
        item["fu006_left_hold_seen"] = True
    trace[-1]["fu006_charger_contained"] = True
    trace[-1]["fu006_contained_seen"] = True
    trace[-1]["fu006_right_released_now"] = True
    trace[-1]["fu006_release_seen"] = True
    trace[-1]["fu006_left_hold_now"] = False
    trace[-1]["fu006_left_released_now"] = True
    trace[-1]["fu006_stable_now"] = True

    arrays = runner["_physical_gate_telemetry_arrays"](trace)
    assert arrays["fu006_gate_step"].tolist() == [1, 2]
    assert arrays["fu006_gate_phase_id"].tolist() == [0, 0]
    assert arrays["next_fu006_gate_measured_left_palm_pose"].shape == (2, 7)
    assert arrays["next_fu006_gate_drawer_handle_contact_world"].shape == (2, 3)
    assert arrays["next_fu006_gate_fu006_containment_margin_x"].dtype.kind == "f"
    assert set(arrays) == {
        "fu006_gate_step",
        "fu006_gate_phase_id",
        *{
            f"fu006_gate_{field}"
            for field in runner["_ACTION_GATE_TELEMETRY_FIELDS"]
        },
        *{
            f"next_fu006_gate_{field}"
            for field in runner["_NEXT_GATE_TELEMETRY_FIELDS"]
        },
    }

    def state(q: float) -> dict[str, dict[str, np.ndarray]]:
        return {"state": {"drawer_joint_state": np.asarray([q, 0.0])}}

    transitions = [
        {
            "current": state(0.098),
            "next": state(0.098),
            "action": np.zeros(54),
            "succeed": False,
            "terminated": False,
            "truncated": False,
        },
        {
            "current": state(0.098),
            "next": state(0.098),
            "action": np.zeros(54),
            "succeed": True,
            "terminated": True,
            "truncated": False,
        },
    ]
    alignment = runner["_success_artifact_alignment"](
        transitions, trace, reset_gate
    )
    assert alignment["action_count"] == 2
    assert alignment["terminal_success_step"] == 2
    assert alignment["telemetry_timebase"] == "post_action_next_state"

    progress = runner["_physical_gate_progress"](trace, reset_gate)
    assert progress["first_unmet_gate"] is None
    assert progress["gates"]["left_handle_contact"]["first_step"] == 1
    assert progress["gates"]["charger_contained_after_hold"]["first_step"] == 2
    assert progress["gates"]["left_release_before_stability"]["first_step"] == 2
    assert progress["gates"]["terminal_success"]["first_step"] == 2

    no_contact_trace = [dict(item) for item in trace]
    for item in no_contact_trace:
        for field in (
            "fu006_left_hold_now",
            "fu006_left_hold_seen",
            "fu006_charger_contained",
            "fu006_contained_seen",
            "fu006_right_released_now",
            "fu006_release_seen",
            "fu006_left_released_now",
            "fu006_success_now",
        ):
            item[field] = False
    no_contact = runner["_physical_gate_progress"](
        no_contact_trace, reset_gate
    )
    assert no_contact["first_unmet_gate"] == "left_handle_contact"

    misaligned_trace = [dict(item) for item in trace]
    misaligned_trace[-1]["fu006_success_now"] = False
    try:
        runner["_success_artifact_alignment"](
            transitions, misaligned_trace, reset_gate
        )
    except RuntimeError as error:
        assert "post-action success gate is misaligned" in str(error)
    else:
        raise AssertionError("misaligned FU006 success telemetry was accepted")


def test_fu006_scheduler_launch_contract_uses_literal_canonical_roots(
    capsys,
) -> None:
    script = runpy.run_path(
        str(Path(__file__).parents[2] / "scripts/fu006_scheduler_collection.py")
    )
    scheduler = Path("/controller/taskdev_cluster_all_hp.py")
    base_ref = "8" * 40
    argv = script["build_scheduler_argv"](
        scheduler=scheduler,
        host="hp-2",
        gpu=3,
        base_ref=base_ref,
        launch=True,
    )

    def option(name: str) -> str:
        return argv[argv.index(name) + 1]

    assert option("--repo") == (
        "/mnt/pfs/luotianwei/handson-task-worktrees/"
        "t46_store_charger"
    )
    assert option("--state-root") == (
        "/mnt/pfs/luotianwei/.handson-taskdev-cluster"
    )
    assert option("--run-root") == "/mnt/pfs/luotianwei/handson-taskdev-runs"
    assert option("--worktree-root") == (
        "/mnt/pfs/luotianwei/handson-taskdev-worktrees"
    )
    assert option("--base-ref") == base_ref
    assert option("--max-runtime-seconds") == "1800"
    assert "--launch" in argv
    task_command = argv[argv.index("--") + 1 :]
    assert "WORKSPACE_DIR={run_dir}" in task_command
    assert "TMPDIR={run_dir}/tmp" in task_command
    assert "XDG_CACHE_HOME={run_dir}/xdg_cache" in task_command
    assert "PYTHONPATH={worktree}:{worktree}/handson" in task_command
    shell_command = task_command[-1]
    assert '"{worktree}/scripts/fu006_vertical_slice.py" collect' in shell_command
    assert '--output "{run_dir}/episode_candidate"' in shell_command

    run_id = "bench320-fu006-contract-20260818"
    contract = script["resolved_contract"](run_id)
    assert contract == {
        "state_root": "/mnt/pfs/luotianwei/.handson-taskdev-cluster",
        "run_dir": f"/mnt/pfs/luotianwei/handson-taskdev-runs/{run_id}",
        "worktree": (
            f"/mnt/pfs/luotianwei/handson-taskdev-worktrees/{run_id}"
        ),
        "runner": (
            f"/mnt/pfs/luotianwei/handson-taskdev-worktrees/{run_id}/"
            "scripts/fu006_vertical_slice.py"
        ),
        "episode_candidate": (
            f"/mnt/pfs/luotianwei/handson-taskdev-runs/{run_id}/"
            "episode_candidate"
        ),
    }
    assert "/handson-task-worktrees/handson-taskdev-" not in json.dumps(contract)

    assert (
        script["main"](
            [
                "--scheduler",
                str(scheduler),
                "--host",
                "hp-2",
                "--gpu",
                "3",
                "--base-ref",
                base_ref,
                "--contract-run-id",
                run_id,
            ]
        )
        == 0
    )
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["mode"] == "cpu_only_contract"
    assert dry_run["resolved_contract"] == contract
    assert "--launch" not in dry_run["scheduler_argv"]
