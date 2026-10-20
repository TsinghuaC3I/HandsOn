from __future__ import annotations

import copy
import inspect
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from handson.runtime.collection_support import ISAACSIM_COLLECTION_TASKS
from handson.sim.isaaclab import bc012_env as bc012_runtime
from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
from handson.sim.isaaclab.bc012_env import BC012IsaacLabEnv
from handson.tasks.t13_load_cans.config import (
    TaskConfig,
)
from handson.tasks.t13_load_cans import (
    scene as bc012_scene,
)
from handson.tasks.t13_load_cans.scene import (
    BIN_ANCHOR_XY,
    BIN_NAME,
    BIN_XY_JITTER,
    CAN_ANCHOR_XY,
    CAN_NAMES,
    CAN_XY_JITTER,
    CAN_YAW_DEGREES,
    LEFT_CAN_NAME,
    RIGHT_CAN_NAME,
    STABILITY_STEPS,
    TASK_NAME,
    TEMPLATE_LAYOUT_TRANSLATION_XY,
    evaluate_bin_load,
    joint_local_frame_pair,
    matrix_to_quat_wxyz,
    pose_matrix,
    usd_prim_identifier,
    validate_asset_contract,
)
from handson.tasks.mappings import CONFIG_MAPPING
from taskdev.tasks.bench320_bc012_dual_can_synchronized_bin_load.episode import (
    TASK_ID as EPISODE_TASK_ID,
    build_episode_data,
    validate_episode,
    write_episode,
)
from taskdev.tasks.bench320_bc012_dual_can_synchronized_bin_load import (
    run_vertical_slice as bc012_runner,
)


def _pose(x: float, y: float, z: float) -> np.ndarray:
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _success_metrics(**overrides):
    values = {
        "left_pose": _pose(0.0, -0.065, 0.075),
        "right_pose": _pose(0.0, 0.065, 0.075),
        "bin_pose": _pose(0.0, 0.0, 0.0),
        "can_local_center": (0.0, 0.0, 0.0),
        "left_supported": True,
        "right_supported": True,
        "left_hand_clear": True,
        "right_hand_clear": True,
        "left_hand_contact_force": 0.0,
        "right_hand_contact_force": 0.0,
        "left_release_frame": 100,
        "right_release_frame": 105,
        "control_hz": 30.0,
    }
    values.update(overrides)
    return evaluate_bin_load(**values)


def _interface_env(*, randomize: bool = False) -> BC012IsaacLabEnv:
    return BC012IsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        randomize=randomize,
    )


def _successful_collection_report() -> dict:
    return {
        "pass": True,
        "semantic_success": True,
        "terminal": True,
        "truncated": False,
        "automatic_action_count": 732,
        "trajectory_state_write_count": 0,
        "camera_metrics": {
            "ego": {"pass": True},
            "wrist_left": {"pass": True},
            "wrist_right": {"pass": True},
        },
        "episode": {"pass": True, "success_count": 1},
        "final_info": {
            "bc012_grasp_errors": [],
            "bc012_grasp_mode": "physical_contacts",
            "bc012_grasp_assist_enabled": False,
            "bc012_left_owned": False,
            "bc012_right_owned": False,
            "bc012_left_ready": True,
            "bc012_right_ready": True,
            "bc012_left_supported": True,
            "bc012_right_supported": True,
            "bc012_success_now": True,
            "bc012_success_steps_required": 15,
            "bc012_success_counter": 15,
            "bc012_synchronized": True,
            "bc012_release_delta_frames": 0,
            "bc012_sync_window_frames": 24,
            "bc012_event_history": [
                {
                    "frame": 66,
                    "event": "grasp",
                    "side": "left",
                    "source": "assigned_hand_contact",
                    "lifted": True,
                },
                {
                    "frame": 66,
                    "event": "grasp",
                    "side": "right",
                    "source": "assigned_hand_contact",
                    "lifted": True,
                },
                {
                    "frame": 285,
                    "event": "release",
                    "side": "left",
                    "source": "assigned_hand_contact_loss",
                },
                {
                    "frame": 285,
                    "event": "release",
                    "side": "right",
                    "source": "assigned_hand_contact_loss",
                },
            ],
        },
    }


def _zero_warning_stdout_audit() -> dict:
    return {
        "available": True,
        "truncated": False,
        "disjointed_joint_warning_count": 0,
    }


def test_task_info_serializes_release_frame_delta_as_integer() -> None:
    env = _interface_env()
    env._bc012_last_metrics = _success_metrics()
    info = env._task_info()
    assert info["bc012_release_delta_frames"] == 5
    assert isinstance(info["bc012_release_delta_frames"], int)


def test_identity_registration_and_backend_contract() -> None:
    assert TASK_NAME in CONFIG_MAPPING
    assert CONFIG_MAPPING[TASK_NAME] is TaskConfig
    assert TASK_NAME in ISAACSIM_COLLECTION_TASKS
    config = TaskConfig()
    assert config.task_id == "13"
    assert config.task_name == "t13_load_cans"
    assert config.teleop.pose_scale == 3.00
    assert config.teleop.max_tracker_delta_m == 0.55
    assert "opposite-side bin partition" in config.task_prompt
    with pytest.raises(ValueError, match="requires Isaac Sim 5.1 PhysX"):
        config.get_environment(sim_backend="mujoco")
    with pytest.raises(ValueError, match="requires Isaac Sim 5.1 PhysX"):
        config.get_environment(sim_backend="isaacsim_newton")


def test_bc012_always_defers_camera_rendering(monkeypatch) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "0")

    env = _interface_env()

    assert env._deferred_camera_render is False


def test_exact_assets_and_glb_identifier_regression() -> None:
    assert validate_asset_contract() == []
    assert usd_prim_identifier("00_bin_back") == "Mesh_00_bin_back"
    assert usd_prim_identifier("bin side") == "bin_side"


def test_physical_grasp_mode_authors_no_can_to_palm_joint() -> None:
    spawn_source = inspect.getsource(bc012_scene.spawn_bc012_assets)
    setup_source = inspect.getsource(
        BC012IsaacLabEnv._configure_task_specific_pre_physx
    )
    hand_source = inspect.getsource(BC012IsaacLabEnv._postprocess_hand_action)
    assert "BC012AssistJoints" not in spawn_source
    assert "FilteredPairsAPI" not in spawn_source
    assert '"grasp_mode"] = "physical_contacts"' in setup_source
    assert "bind_bc012_grasp_joint_owners" not in setup_source
    assert "_set_bc012_grasp_joint" not in hand_source
    assert "_set_bc012_owner_contact_filter" not in hand_source
    assert "HAND_CONTACT_FORCE_THRESHOLD" in hand_source
    env = _interface_env()
    env._bc012_scene_manifest = {}
    assert env._configure_task_specific_pre_physx() is False
    assert env._bc012_scene_manifest["grasp_mode"] == "physical_contacts"


def test_legacy_replay_assist_entrypoints_fail_closed() -> None:
    env = _interface_env()
    with pytest.raises(RuntimeError, match="grasp assist was removed"):
        env.begin_bc012_replay_assist_control()
    with pytest.raises(RuntimeError, match="grasp assist was removed"):
        env.install_bc012_replay_assist_state(
            np.zeros(14),
            recorded_can_poses={},
            recorded_palm_states={},
        )


def test_joint_local_frames_reconstruct_one_world_anchor() -> None:
    can_world = pose_matrix(
        [-0.22, 0.16, 0.95, 0.965925826, 0.0, 0.0, 0.258819045]
    )
    palm_world = pose_matrix(
        [-0.21, 0.17, 1.06, 0.707106781, 0.0, 0.707106781, 0.0]
    )
    body0_frame, body1_frame = joint_local_frame_pair(can_world, palm_world)
    np.testing.assert_allclose(
        can_world @ body0_frame,
        palm_world @ body1_frame,
        atol=1.0e-12,
        rtol=0.0,
    )
    for frame in (body0_frame, body1_frame):
        quaternion = matrix_to_quat_wxyz(frame[:3, :3])
        reconstructed = pose_matrix([*frame[:3, 3], *quaternion])
        np.testing.assert_allclose(reconstructed, frame, atol=1.0e-8, rtol=0.0)


def test_reset_performs_no_grasp_joint_writes(
    monkeypatch,
) -> None:
    env = _interface_env(randomize=True)
    events: list[str] = []
    monkeypatch.setattr(env, "_set_bc012_grasp_joint", lambda *_a, **_k: events.append("joint"))
    monkeypatch.setattr(
        HandsOnABCIsaacLabEnv,
        "_reset_scene_state",
        lambda _self: events.append("physical_reset"),
    )

    env._reset_scene_state()

    assert events == ["physical_reset"]


def test_physical_grasp_and_release_are_contact_driven(monkeypatch) -> None:
    env = _interface_env()
    env._bc012_scene_manifest["can_local_center"] = [0.0, 0.0, 0.0]
    env._bc012_scene_manifest["can_source_low"] = [0.0, 0.0, -0.05]
    live = np.eye(4, dtype=np.float64)
    live[2, 3] = (
        bc012_scene.TABLE_TOP_Z
        + bc012_scene.CAN_SUPPORT_CLEARANCE
        + 0.05
        + 0.02
    )
    forces = {LEFT_CAN_NAME: 0.0, RIGHT_CAN_NAME: 0.10}

    monkeypatch.setattr(env, "_robot_link_pose_matrix", lambda _side: live.copy())
    monkeypatch.setattr(env, "_current_body_matrix", lambda _name: live.copy())
    monkeypatch.setattr(
        env,
        "_bc012_contact_force",
        lambda name, role: (forces[name], True) if role == "owner" else (0.0, True),
    )
    monkeypatch.setattr(env, "_a06_hand_points", lambda _side: (live[:3, 3] + 0.20,))
    monkeypatch.setattr(
        env,
        "_set_bc012_grasp_joint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("physical grasp must not write D6 joints")
        ),
    )

    env._postprocess_hand_action()
    env._postprocess_hand_action()

    assert env._bc012_physical_grasp_seen == {"left": False, "right": True}
    assert env._bc012_owner_contact_available == {"left": True, "right": True}
    assert env._bc012_owner_contact_force == {"left": 0.0, "right": 0.10}
    assert env._bc012_grasp_owner == {"left": False, "right": False}
    assert env._bc012_event_history[0]["source"] == "assigned_hand_contact"

    forces[RIGHT_CAN_NAME] = 0.0
    env._postprocess_hand_action()
    env._postprocess_hand_action()

    assert env._bc012_release_frame["right"] == 3
    assert env._bc012_event_history[-1]["source"] == "assigned_hand_contact_loss"


def test_table_contact_is_not_misclassified_as_a_grasp(monkeypatch) -> None:
    env = _interface_env()
    env._bc012_scene_manifest["can_local_center"] = [0.0, 0.0, 0.0]
    env._bc012_scene_manifest["can_source_low"] = [0.0, 0.0, -0.05]
    resting = np.eye(4, dtype=np.float64)
    resting[2, 3] = (
        bc012_scene.TABLE_TOP_Z
        + bc012_scene.CAN_SUPPORT_CLEARANCE
        + 0.05
    )
    monkeypatch.setattr(env, "_robot_link_pose_matrix", lambda _side: resting.copy())
    monkeypatch.setattr(env, "_current_body_matrix", lambda _name: resting.copy())
    monkeypatch.setattr(env, "_bc012_contact_force", lambda _name, _role: (0.10, True))
    env._postprocess_hand_action()
    env._postprocess_hand_action()
    assert env._bc012_physical_grasp_seen == {"left": False, "right": False}
    assert env._bc012_event_history == []


def test_runner_rejects_false_zero_exit_without_semantic_report(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        bc012_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(bc012_runner.__file__),
            "--phase",
            "collect",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert bc012_runner._run_checked_child() == 86


def test_fresh_replay_compares_the_complete_hidden_task_state() -> None:
    replay_source = inspect.getsource(bc012_runner._replay)
    assert '_state(observation)["bc012_task_state"]' in replay_source
    assert 'data["next_state_bc012_task_state"][index]' in replay_source
    assert 'bc012_task_state"][:' not in replay_source
    assert "env.begin_bc012_replay_assist_control()" not in replay_source
    assert "image_obs=True" in replay_source
    assert "env.install_bc012_replay_assist_state(" not in replay_source
    assert '"grasp_assist_enabled": False' in replay_source
    assert "t0_origin = _compare_public_t0_origin(" in replay_source
    assert "t0_restore = trace.restore(t0_snapshot, recording_schema)" in replay_source
    assert 'if t0_origin["matched"]:' in replay_source
    assert replay_source.count("env.reset(seed=seed)") == 2
    assert replay_source.index("env.step(") < replay_source.rindex(
        "env.reset(seed=seed)"
    ) < replay_source.index("trace.restore(t0_snapshot, recording_schema)")
    assert replay_source.index("env._apply_action(raw_action)") < replay_source.index(
        "run_articulated_drive_control_step("
    )


def test_t0_snapshot_uses_only_current_pre_action_physx_fields() -> None:
    data = {
        name: np.asarray([[index, index + 1]], dtype=np.float64)
        for index, name in enumerate(bc012_runner.PHYSX_T0_RESTORE_FIELDS)
    }
    data["next_physx_robot_joint_position"] = np.asarray([[999.0]])
    snapshot = bc012_runner._recorded_t0_snapshot(data)
    assert tuple(snapshot) == bc012_runner.PHYSX_T0_RESTORE_FIELDS
    for index, name in enumerate(bc012_runner.PHYSX_T0_RESTORE_FIELDS):
        np.testing.assert_array_equal(snapshot[name], [index, index + 1])
        assert not np.shares_memory(snapshot[name], data[name])
    assert "next_physx_robot_joint_position" not in snapshot

    incomplete = dict(data)
    del incomplete["physx_robot_joint_velocity"]
    with pytest.raises(RuntimeError, match="missing t0 restore fields"):
        bc012_runner._recorded_t0_snapshot(incomplete)


def test_public_t0_origin_uses_schema_tolerances_and_fails_closed() -> None:
    recorded = {
        name: np.asarray([index], dtype=np.float64)
        for index, name in enumerate(bc012_runner.PHYSX_T0_RESTORE_FIELDS)
    }
    observation = {name: value.copy() for name, value in recorded.items()}
    schema = {
        "fields": {
            name: {"replay_atol": 1.0e-6}
            for name in bc012_runner.PHYSX_T0_RESTORE_FIELDS
        }
    }
    comparison = bc012_runner._compare_public_t0_origin(
        observation,
        recorded,
        schema,
    )
    assert comparison["matched"] is True

    first = bc012_runner.PHYSX_T0_RESTORE_FIELDS[0]
    observation[first] = observation[first] + 2.0e-6
    comparison = bc012_runner._compare_public_t0_origin(
        observation,
        recorded,
        schema,
    )
    assert comparison["matched"] is False
    assert comparison["maximum_errors"][first] == pytest.approx(2.0e-6)

    incomplete = dict(observation)
    del incomplete[first]
    with pytest.raises(RuntimeError, match="public t0 origin is missing"):
        bc012_runner._compare_public_t0_origin(
            incomplete,
            recorded,
            schema,
        )


def test_physical_grasp_state_is_recomputed_instead_of_restored(
    monkeypatch,
) -> None:
    env = _interface_env()
    live = np.eye(4, dtype=np.float64)
    live[2, 3] = 10.0
    env._bc012_scene_manifest["can_local_center"] = [0.0, 0.0, 0.0]
    env._bc012_scene_manifest["can_source_low"] = [0.0, 0.0, -0.05]
    monkeypatch.setattr(env, "_robot_link_pose_matrix", lambda _side: live.copy())
    monkeypatch.setattr(env, "_current_body_matrix", lambda _name: live.copy())
    monkeypatch.setattr(env, "_bc012_contact_force", lambda _name, _role: (0.25, True))
    env._postprocess_hand_action()
    env._postprocess_hand_action()
    assert env._bc012_physical_grasp_seen == {"left": True, "right": True}
    assert env._bc012_grasp_owner == {"left": False, "right": False}


def test_bc012_enables_source_wuji_grip_profile_locally() -> None:
    env = _interface_env()
    assert env._smooth_grasp_soft_hand_enabled() is True


def test_collection_acceptance_is_bounded_and_fail_closed(
    tmp_path,
    monkeypatch,
) -> None:
    report = _successful_collection_report()
    log_path = tmp_path / "stdout.log"
    log_path.write_text("clean child log\n", encoding="utf-8")
    audit = bc012_runner._bounded_stdout_audit(log_path)
    assert audit["available"] is True
    assert audit["truncated"] is False
    assert audit["disjointed_joint_warning_count"] == 0
    assert bc012_runner._collection_acceptance(report, audit)["pass"] is True

    log_path.write_text(
        bc012_runner._DISJOINTED_JOINT_WARNING + "\n",
        encoding="utf-8",
    )
    disjointed = bc012_runner._bounded_stdout_audit(log_path)
    assert disjointed["disjointed_joint_warning_count"] == 1
    assert bc012_runner._collection_acceptance(report, disjointed)["pass"] is False

    monkeypatch.setattr(bc012_runner, "_MAX_ACCEPTANCE_LOG_BYTES", 8)
    log_path.write_bytes(b"012345678")
    bounded = bc012_runner._bounded_stdout_audit(log_path)
    assert bounded["bytes_scanned"] == 8
    assert bounded["truncated"] is True
    assert bc012_runner._collection_acceptance(report, bounded)["pass"] is False

    assisted = copy.deepcopy(report)
    assisted["final_info"]["bc012_grasp_assist_enabled"] = True
    assert bc012_runner._collection_acceptance(assisted, audit)["pass"] is False

    unphysical = copy.deepcopy(report)
    unphysical["final_info"]["bc012_event_history"][0]["source"] = "d6_joint"
    assert bc012_runner._collection_acceptance(unphysical, audit)["pass"] is False

    still_owned = copy.deepcopy(report)
    still_owned["final_info"]["bc012_left_owned"] = True
    assert bc012_runner._collection_acceptance(still_owned, audit)["pass"] is False

    unstable = copy.deepcopy(report)
    unstable["final_info"]["bc012_success_counter"] = 14
    assert bc012_runner._collection_acceptance(unstable, audit)["pass"] is False
    assert bc012_runner._collection_acceptance({}, {})["pass"] is False


def test_failed_collection_acceptance_removes_only_episode_zarr(tmp_path) -> None:
    episode = tmp_path / "episode" / "replay.zarr"
    episode.mkdir(parents=True)
    (episode / "chunk").write_bytes(b"failed")
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    audit = _zero_warning_stdout_audit()
    audit["disjointed_joint_warning_count"] = 1

    report = _successful_collection_report()
    assert bc012_runner._apply_collection_acceptance(tmp_path, report, audit) is False
    assert not episode.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert report["episode_retained"] is False
    assert report["failed_episode_removed"] is True


def test_successful_collection_acceptance_retains_episode_zarr(tmp_path) -> None:
    episode = tmp_path / "episode" / "replay.zarr"
    episode.mkdir(parents=True)
    (episode / "chunk").write_bytes(b"accepted")
    report = _successful_collection_report()

    assert bc012_runner._apply_collection_acceptance(
        tmp_path,
        report,
        _zero_warning_stdout_audit(),
    ) is True
    assert episode.is_dir()
    assert report["episode_retained"] is True
    assert report["failed_episode_removed"] is False


def test_interface_schema_is_dual_arm_and_fixed_shape() -> None:
    env = _interface_env()
    assert env.action_space.shape == (54,)
    state_spaces = env.observation_space.spaces["state"].spaces
    assert state_spaces["left_can_pose"].shape == (7,)
    assert state_spaces["right_can_pose"].shape == (7,)
    assert state_spaces["open_bin_pose"].shape == (7,)
    assert state_spaces["partition_centers"].shape == (2, 3)
    assert state_spaces["bin_load_metrics"].shape == (12,)
    assert state_spaces["bc012_task_state"].shape == (14,)


def test_all_three_camera_streams_are_canonical_480_by_640() -> None:
    env = BC012IsaacLabEnv(
        launch_isaac=False,
        image_obs=True,
        randomize=False,
    )
    image_spaces = env.observation_space.spaces["images"].spaces
    assert set(image_spaces) == {"ego", "wrist_left", "wrist_right"}
    assert all(space.shape == (480, 640, 3) for space in image_spaces.values())


def test_missing_contact_views_use_finite_observation_sentinels() -> None:
    env = _interface_env()
    env._poses = {
        LEFT_CAN_NAME: _pose(-0.43, 0.18, 0.94),
        RIGHT_CAN_NAME: _pose(-0.43, -0.18, 0.94),
        BIN_NAME: _pose(-0.08, 0.0, 0.892),
    }
    env._bc012_scene_manifest["can_local_center"] = [0.0, 0.0, 0.0]
    env._a06_hand_points = lambda _side: ()
    state = env._compute_observation()["state"]["bc012_task_state"]
    assert np.isfinite(state).all()
    assert tuple(state[-2:]) == (-1.0, -1.0)


def test_physical_success_requires_both_assigned_supported_released_cans() -> None:
    metrics = _success_metrics()
    assert metrics["left"]["ready"] is True
    assert metrics["right"]["ready"] is True
    assert metrics["synchronized"] is True
    assert metrics["success_now"] is True


def test_control_rate_matches_declared_thirty_hz_contract() -> None:
    env = _interface_env()
    assert env._control_dt == pytest.approx(1.0 / 30.0)
    assert env._physics_dt == pytest.approx(1.0 / 120.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "left_pose": _pose(0.0, 0.065, 0.075),
            "right_pose": _pose(0.0, -0.065, 0.075),
        },
        {"right_pose": _pose(-0.30, 0.065, 0.075)},
        {"left_pose": _pose(0.0, -0.065, 0.145)},
        {"left_supported": False},
        {"left_owned": True},
        {"left_hand_clear": False},
        {"left_hand_contact_force": 2.0},
        {"left_velocity": (0.08, 0.0, 0.0, 0.0, 0.0, 0.0)},
        {"left_release_frame": 0, "right_release_frame": 30},
        {"left_release_frame": None},
    ],
)
def test_false_positive_and_near_miss_cases_fail(overrides) -> None:
    assert _success_metrics(**overrides)["success_now"] is False


def test_upright_gate_rejects_fallen_can() -> None:
    # 90 degrees about X: can local Z is horizontal.
    fallen = np.asarray(
        [0.0, -0.065, 0.075, math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
        dtype=np.float64,
    )
    metrics = _success_metrics(left_pose=fallen)
    assert metrics["left"]["upright"] is False
    assert metrics["success_now"] is False


def test_stability_window_is_exactly_fifteen_control_frames() -> None:
    env = _interface_env()
    env._compute_success_now = lambda: True
    for _ in range(STABILITY_STEPS - 1):
        assert env._compute_success() is False
    assert env._compute_success() is True
    env._compute_success_now = lambda: False
    assert env._compute_success() is False
    assert env._success_counter == 0


def test_seeded_bounded_reset_is_reproducible() -> None:
    env = _interface_env(randomize=True)
    env._bc012_scene_manifest["can_collision_radius"] = 0.033
    env._reset_robot_body_points = lambda _side: (
        np.asarray([-0.10, 0.36, 1.07]),
        np.asarray([-0.10, -0.36, 1.07]),
    )
    env._random = np.random.default_rng(20260817)
    layout_a, yaw_a = env._bc012_reset_layout()
    env._random = np.random.default_rng(20260817)
    layout_b, yaw_b = env._bc012_reset_layout()
    assert tuple(layout_a) == (*CAN_NAMES, BIN_NAME)
    for name in layout_a:
        np.testing.assert_allclose(layout_a[name], layout_b[name], atol=0.0, rtol=0.0)
        assert yaw_a[name] == yaw_b[name]
    assert abs(yaw_a[LEFT_CAN_NAME]) <= 12.0
    assert abs(yaw_a[RIGHT_CAN_NAME]) <= 12.0
    assert yaw_a[BIN_NAME] == 0.0


def test_reachable_layout_uses_the_shared_bimanual_workspace() -> None:
    np.testing.assert_allclose(
        TEMPLATE_LAYOUT_TRANSLATION_XY,
        [0.0, 0.0],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        CAN_ANCHOR_XY[LEFT_CAN_NAME] - BIN_ANCHOR_XY,
        [-0.03, 0.28],
    )
    np.testing.assert_allclose(
        CAN_ANCHOR_XY[RIGHT_CAN_NAME] - BIN_ANCHOR_XY,
        [-0.03, -0.28],
    )
    assert CAN_XY_JITTER == 0.035
    assert BIN_XY_JITTER == 0.025
    assert CAN_YAW_DEGREES == 12.0


def test_partition_command_is_centered_for_observed_crossed_reach_bias() -> None:
    assert bc012_runtime._PLACEMENT_COMMAND_Y_LOCAL_M == pytest.approx(0.046)
    assert 0.044 < bc012_runtime._PLACEMENT_COMMAND_Y_LOCAL_M < 0.086


def test_nominal_layout_remains_clear_of_live_home_palms() -> None:
    env = _interface_env(randomize=False)
    env._bc012_scene_manifest["can_collision_radius"] = 0.033
    layout, _ = env._bc012_reset_layout()
    live_home_palms_xy = (
        np.asarray([-0.4092, 0.4041]),
        np.asarray([-0.4092, -0.4041]),
    )
    assert env._bc012_layout_margin(layout, live_home_palms_xy) >= 0.005


def test_automatic_trajectory_is_action_only_deterministic_and_complete() -> None:
    env = _interface_env()
    observation = {
        "state": {
            "tcp_pose": np.asarray(
                [
                    -0.10,
                    -0.36,
                    1.07,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    -0.10,
                    0.36,
                    1.07,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            ),
            "right_can_pose": _pose(-0.23, -0.18, 0.89),
            "left_can_pose": _pose(-0.23, 0.18, 0.89),
            "open_bin_pose": _pose(0.12, 0.0, 0.892),
        }
    }
    actions_a = env.automatic_success_actions(observation)
    actions_b = env.automatic_success_actions(observation)
    assert len(actions_a) == 732
    assert all(action.shape == (54,) for action in actions_a)
    np.testing.assert_array_equal(np.stack(actions_a), np.stack(actions_b))
    # The only state-changing input returned by the planner is the canonical
    # raw environment action. It never returns object or robot state snapshots.
    assert np.max(np.abs(np.stack(actions_a)[:, 14:])) > 0.5


def test_automatic_waypoints_start_from_physx_palms_and_live_asset_frames() -> None:
    env = _interface_env()
    right_live = np.eye(4, dtype=np.float64)
    right_live[:3, 3] = [-0.32, -0.34, 1.08]
    left_live = np.eye(4, dtype=np.float64)
    left_live[:3, 3] = [-0.31, 0.33, 1.09]
    env._robot_link_pose_matrix = lambda side: (
        right_live.copy() if side == "right" else left_live.copy()
    )
    right_can = _pose(-0.21, -0.17, 0.949)
    left_can = _pose(-0.24, 0.19, 0.949)
    bin_pose = _pose(0.11, 0.01, 0.892)
    observation = {
        "state": {
            "tcp_pose": np.asarray(
                [
                    -9.0,
                    -9.0,
                    -9.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    9.0,
                    9.0,
                    9.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            ),
            "right_can_pose": right_can,
            "left_can_pose": left_can,
            "open_bin_pose": bin_pose,
        }
    }
    actions = np.stack(env.automatic_success_actions(observation))
    np.testing.assert_allclose(actions[0, :3], right_live[:3, 3])
    np.testing.assert_allclose(actions[0, 7:10], left_live[:3, 3])
    assert env._bc012_automatic_start_source == {
        "left": "physx_palm_link",
        "right": "physx_palm_link",
    }
    right_center = right_can[:3] + np.asarray([0.0, 0.0, 0.05])
    left_center = left_can[:3] + np.asarray([0.0, 0.0, 0.05])
    can_center_in_palm = np.asarray([0.055, 0.0, 0.055])
    right_grasp = right_center - can_center_in_palm
    left_grasp = left_center - can_center_in_palm
    np.testing.assert_allclose(actions[64, :3], right_grasp)
    np.testing.assert_allclose(actions[64, 7:10], left_grasp)
    right_center_local = right_center - bin_pose[:3]
    left_center_local = left_center - bin_pose[:3]
    right_lift = (
        bin_pose[:3]
        + [-0.225, right_center_local[1], 0.34]
        - can_center_in_palm
    )
    left_lift = (
        bin_pose[:3]
        + [-0.225, left_center_local[1], 0.34]
        - can_center_in_palm
    )
    right_cross = bin_pose[:3] + [-0.225, 0.10, 0.34] - can_center_in_palm
    left_cross = bin_pose[:3] + [-0.225, -0.10, 0.34] - can_center_in_palm
    right_staged_over = bin_pose[:3] + [0.10, 0.065, 0.34] - can_center_in_palm
    right_command_goal = bin_pose[:3] + [0.05, 0.046, 0.075]
    left_command_goal = bin_pose[:3] + [-0.05, -0.046, 0.075]
    right_palm_compensation = np.asarray([0.005, -0.07, 0.0])
    left_palm_compensation = np.asarray([0.055, 0.06, 0.0])
    right_over = (
        right_command_goal
        + [0.0, 0.0, 0.22]
        - can_center_in_palm
        + right_palm_compensation
    )
    left_over = (
        left_command_goal
        + [0.0, 0.0, 0.22]
        - can_center_in_palm
        + left_palm_compensation
    )
    right_release = (
        right_command_goal
        + [0.0, 0.0, 0.0]
        - can_center_in_palm
        + right_palm_compensation
    )
    left_release = (
        left_command_goal
        + [0.0, 0.0, 0.0]
        - can_center_in_palm
        + left_palm_compensation
    )
    np.testing.assert_allclose(actions[118, :3], right_lift)
    np.testing.assert_allclose(actions[118, 7:10], left_lift)
    # The right can crosses and moves into the bin interior first, clearing the
    # entire outside wrist lane before the left can crosses above the bin wall.
    np.testing.assert_allclose(actions[168, :3], right_cross)
    np.testing.assert_allclose(actions[168, 7:10], left_lift)
    np.testing.assert_allclose(actions[198, :3], right_staged_over)
    np.testing.assert_allclose(actions[198, 7:10], left_lift)
    np.testing.assert_allclose(actions[248, :3], right_staged_over)
    np.testing.assert_allclose(actions[248, 7:10], left_cross)
    np.testing.assert_allclose(actions[268, :3], right_staged_over)
    np.testing.assert_allclose(actions[268, 7:10], left_cross)
    np.testing.assert_allclose(actions[328, :3], right_staged_over)
    np.testing.assert_allclose(actions[328, 7:10], left_over)
    np.testing.assert_allclose(actions[408, :3], right_staged_over)
    np.testing.assert_allclose(
        actions[408, 7:10],
        left_release,
    )
    np.testing.assert_allclose(actions[488, :3], right_over)
    np.testing.assert_allclose(
        actions[488, 7:10],
        left_release,
    )
    np.testing.assert_allclose(
        actions[568, :3],
        right_release,
    )
    np.testing.assert_allclose(
        actions[568, 7:10],
        left_release,
    )
    assert np.max(np.abs(actions[618, 14:34])) == pytest.approx(0.25)
    assert np.max(np.abs(actions[618, 34:54])) == pytest.approx(0.25)
    np.testing.assert_allclose(actions[621, 14:54], 0.0)


def test_power_grip_waypoint_uses_palm_local_finger_array_offset() -> None:
    palm = np.asarray(
        [0.2, -0.1, 1.1, math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0],
        dtype=np.float64,
    )
    can_center = np.asarray([-0.3, 0.2, 0.95], dtype=np.float64)
    planned = BC012IsaacLabEnv._bc012_palm_pose_for_can_center(palm, can_center)
    relative = pose_matrix(planned)[:3, :3].T @ (can_center - planned[:3])
    np.testing.assert_allclose(relative, [0.055, 0.0, 0.055], atol=1.0e-12)


def test_bc012_always_uses_full_pose_ik() -> None:
    env = _interface_env()
    assert env._differential_ik_command_type("right") == "pose"
    assert env._differential_ik_command_type("left") == "pose"
    for frame in (0, 118, 119, 328, 329, 1000):
        env._bc012_control_frame = frame
        assert not env._use_position_only_differential_ik("right")
        assert not env._use_position_only_differential_ik("left")
        assert not env._use_position_only_multistart_ik("right")
        assert not env._use_position_only_multistart_ik("left")


def _episode_observation(env: BC012IsaacLabEnv):
    observation = env._compute_observation()
    assert set(observation["images"]) == {"ego", "wrist_left", "wrist_right"}
    return observation


def test_bc012_zarr_episode_binds_identity_truth_hidden_state_and_cameras(
    tmp_path,
) -> None:
    env = BC012IsaacLabEnv(
        launch_isaac=False,
        image_obs=True,
        randomize=False,
    )
    observation = _episode_observation(env)
    observation.update(
        {
            "physx_substep_count": np.asarray([4], dtype=np.int64),
            "physx_substep_robot_joint_target": np.zeros(
                (4, 2, 27), dtype=np.float64
            ),
            "physx_substep_robot_joint_target_valid": np.ones(
                (4, 2, 27), dtype=np.uint8
            ),
            "physx_robot_joint_target": np.zeros((2, 27), dtype=np.float64),
            "physx_robot_joint_target_valid": np.ones((2, 27), dtype=np.uint8),
        }
    )
    action = np.zeros(54, dtype=np.float32)
    transitions = [
        {
            "observation": observation,
            "action": action,
            "next_observation": observation,
            "terminal": True,
            "truncated": False,
            "succeed": True,
        }
    ]
    data = build_episode_data(transitions)
    assert data["action"].shape == (1, 54)
    assert data["state_bc012_task_state"].shape == (1, 14)
    assert data["next_state_left_can_pose"].shape == (1, 7)
    assert data["image_ego"].shape == (1, 480, 640, 3)
    zarr_path = tmp_path / "replay.zarr"
    summary = write_episode(
        zarr_path,
        transitions,
        reset_seed=17,
        reset_options={"randomize": True},
        recording_schema={
            "name": "handson.physx_ground_truth",
            "version": 6,
            "physx": {
                "robot_sides": ["right", "left"],
                "robot_joint_names": {
                    "right": [f"right_{index}" for index in range(27)],
                    "left": [f"left_{index}" for index in range(27)],
                },
                "physics_config": {"device": "cuda:0"},
            },
        },
        source_pid=1234,
    )
    assert summary["task_id"] == EPISODE_TASK_ID == "B320-BC012"
    validated = validate_episode(
        zarr_path,
        require_success=True,
        require_physical_replay=True,
    )
    assert validated["success_count"] == 1
    assert validated["recorded_drive_targets"] is True
