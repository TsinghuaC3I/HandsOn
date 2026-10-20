from __future__ import annotations

import json
from pathlib import Path
import types

import numpy as np
import pytest

from handson.runtime.collection_support import ISAACSIM_COLLECTION_TASKS
from handson.sim.isaaclab.abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _HS005_BOTTLE_RESET_XY,
    _HS005_BRUSH_STABLE_RESET_POSE_XYZW,
    _HS005_CLEAR_ARM_HOME,
    _HS005_CLEAR_LEFT_START_TCP,
    _HS005_CLEAR_RIGHT_START_TCP,
    _HS005_RESET_ROOT_Z_BY_SCENE,
    _randomized_physx_pose,
    _sample_hs005_reset_offsets,
)
from handson.sim.isaaclab.simulation_scene_catalog import (
    HS005_APPARTEMENT_KITCHEN_SINK_SCENE,
    HS005_MODERN_APARTMENT_SINK_SCENE,
    HS005_MODERN_KITCHEN_SINK_SCENE,
    HS005_REALISTIC_KITCHEN_SCENE_SINK_SCENE,
)
from handson.tasks.mappings import CONFIG_MAPPING
from taskdev.tasks.bench320_hs005_sink_bottle_stabilized_scrub.automated_policy import (
    BRUSH_PLACE_PALM_LIFT_M,
    BRUSH_BRISTLE_LOCAL,
    HS005AutomatedPolicy,
    LEFT_BOTTLE_NEAR_SURFACE_LOCAL,
    LEFT_PALM_APPROACH_REACH_M,
    LEFT_PALM_CONTACT_REACH_M,
    MAX_COMMAND_ROTATION_STEP_RAD,
    MAX_COMMAND_TRANSLATION_STEP_M,
    MAX_TASK_PALM_TRANSLATION_M,
    OPEN_HAND,
    RIGHT_CARRIED_PALM_T_BRUSH,
    RIGHT_BRUSH_T_PALM,
    SCRUB_BRISTLE_TARGETS_BOTTLE_LOCAL,
    TASK_ROLE_BY_SIDE,
    _world_yaw_rotation,
)
from taskdev.tasks.bench320_hs005_sink_bottle_stabilized_scrub.vertical_slice_io import (
    HIDDEN_KEYS,
    HS005EpisodeWriter,
    load_zarr_frame,
    sha256_tree,
)
from taskdev.tasks.bench320_hs005_sink_bottle_stabilized_scrub.success_classifier_audit import (
    run_audit as run_success_classifier_audit,
)


TASK = "t37_scrub_bottle"
ROOT = Path(__file__).resolve().parents[2]


def _environment(**kwargs):
    return HandsOnABCIsaacLabEnv(
        TASK,
        launch_isaac=False,
        image_obs=False,
        render_mode="rgb_array",
        **kwargs,
    )


def test_hs005_live_success_panel_uses_supported_ascii_text() -> None:
    env = object.__new__(HandsOnABCIsaacLabEnv)
    env.task_name = TASK
    env._last_hs005_success_metrics = {
        "hs005_order_progress": 2,
        "hs005_stability_seconds": 0.25,
    }

    status = env._task_status_text()
    assert status == "Scrub bands 2/3  |  Stable 0.25/0.50 sec"
    status.encode("ascii")

    expected_labels = {
        "hs005_left_stabilized": "Left hand physically stabilized bottle",
        "hs005_right_grasped": "Right hand safely grasped scrub brush handle",
        "hs005_ordered_coverage": (
            "Lower, middle, and upper bands covered in order"
        ),
        "hs005_path_length": "Physical scrub path exceeds 8 cm",
        "hs005_release": "Both hands and objects released",
        "hs005_stable": "Bottle upright and stable for 0.5 sec",
    }
    for key, expected in expected_labels.items():
        label = env._success_condition_display_label(
            {"key": key, "label": "unsupported non-ASCII fallback"}
        )
        assert label == expected
        label.encode("ascii")


def test_hs005_identity_ui_config_mapping_and_collection_tasks() -> None:
    config = CONFIG_MAPPING[TASK]
    assert config.task_name == TASK
    assert config.task_id == "37"
    assert config.benchmark_task_id == "B320-HS005"
    assert TASK in ISAACSIM_COLLECTION_TASKS
    preview = (
        ROOT / "handson/handson/sim/isaaclab/task_preview.py"
    ).read_text(encoding="utf-8")
    assert f'"{TASK}"' in preview
    task_config = (
        ROOT / "configs/multi_task/t37_scrub_bottle.yaml"
    ).read_text(encoding="utf-8")
    assert "- cleaning_brush" in task_config


def test_hs005_live_runners_exercise_dynamics_randomization_and_bound_teardown() -> None:
    task_dir = (
        ROOT / "taskdev/tasks/bench320_hs005_sink_bottle_stabilized_scrub"
    )
    for runner_name in (
        "reachability_probe.py",
        "vertical_slice_runner.py",
        "fresh_replay_runner.py",
    ):
        source = (task_dir / runner_name).read_text(encoding="utf-8")
        assert "randomize_dynamics=True" in source
        assert "kit_cleanup_watchdog_seconds" in source
        assert "evidence_written_before_teardown" in source


def test_hs005_observation_action_space_and_three_camera_contract() -> None:
    env = _environment()
    try:
        assert env.action_space.shape == (54,)
        for key, shape in {
            "hs005_bottle_pose": (7,),
            "hs005_brush_pose": (7,),
            "hs005_body_velocities": (12,),
            "hs005_role_contacts": (6,),
            "hs005_coverage_mask": (3,),
            "hs005_order_state": (4,),
            "hs005_stage": (6,),
            "hs005_reset_parameters": (9,),
        }.items():
            assert env.observation_space["state"][key].shape == shape
        assert env._physics_backend == "physx"
    finally:
        env.close()

    camera_env = HandsOnABCIsaacLabEnv(
        TASK,
        launch_isaac=False,
        image_obs=True,
        randomize=False,
        render_mode="rgb_array",
    )
    try:
        assert set(camera_env.observation_space["images"].spaces) == {
            "ego",
            "wrist_left",
            "wrist_right",
        }
    finally:
        camera_env.close()


def test_hs005_seeded_reset_randomization_is_repeatable_and_bounded() -> None:
    np.testing.assert_array_equal(_HS005_BOTTLE_RESET_XY, [-0.10, 0.11])
    env = _environment(randomize=True, randomize_dynamics=True, seed=17)
    try:
        env._sample_reset_state(clear_manual_request=True)
        first = np.asarray(
            [env._hs005_friction, env._hs005_bottle_mass_scale, env._hs005_brush_mass_scale]
        )
        env._seed = 17
        env._random.seed(17)
        env._sample_reset_state(clear_manual_request=True)
        second = np.asarray(
            [env._hs005_friction, env._hs005_bottle_mass_scale, env._hs005_brush_mass_scale]
        )
        np.testing.assert_array_equal(first, second)
        assert 0.55 <= first[0] <= 0.90
        assert np.all((0.85 <= first[1:]) & (first[1:] <= 1.15))
        env._random.seed(18)
        env._sample_reset_state(clear_manual_request=True)
        third = np.asarray(
            [env._hs005_friction, env._hs005_bottle_mass_scale, env._hs005_brush_mass_scale]
        )
        assert not np.array_equal(first, third)
    finally:
        env.close()


def test_hs005_seeded_reset_has_real_position_variation_for_bottle_and_brush() -> None:
    seeds = tuple(range(32))
    samples = np.asarray(
        [
            _sample_hs005_reset_offsets(
                np.random.RandomState(seed),
                randomize=True,
            )
            for seed in seeds
        ]
    )
    repeats = np.asarray(
        [
            _sample_hs005_reset_offsets(
                np.random.RandomState(seed),
                randomize=True,
            )
            for seed in seeds
        ]
    )
    np.testing.assert_array_equal(samples, repeats)

    # Position variation is mandatory for both manipulated objects; changing
    # only yaw, materials, or dynamics does not satisfy the reset contract.
    assert np.ptp(samples[:, 0]) > 0.030
    assert np.ptp(samples[:, 1]) > 0.030
    assert np.ptp(samples[:, 3]) > 0.020
    assert np.ptp(samples[:, 4]) > 0.020
    assert np.all(np.abs(samples[:, :2]) <= 0.020 + 1.0e-12)
    assert np.all(np.abs(samples[:, 3:5]) <= 0.015 + 1.0e-12)

    fixed = _sample_hs005_reset_offsets(
        np.random.RandomState(17),
        randomize=False,
    )
    np.testing.assert_array_equal(fixed, np.zeros(6, dtype=np.float64))


def test_hs005_brush_reset_randomizes_from_measured_support_equilibrium() -> None:
    base = _HS005_BRUSH_STABLE_RESET_POSE_XYZW.reshape(1, 7)
    randomized = _randomized_physx_pose(
        base,
        xy_offset=np.asarray([0.015, -0.015]),
        z_offset=0.0,
        yaw_degrees=10.0,
    )
    np.testing.assert_allclose(
        randomized[0, :2],
        _HS005_BRUSH_STABLE_RESET_POSE_XYZW[:2] + [0.015, -0.015],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert randomized[0, 2] == _HS005_BRUSH_STABLE_RESET_POSE_XYZW[2]
    assert np.linalg.norm(randomized[0, 3:7]) == pytest.approx(1.0)


def test_hs005_each_scene_resets_on_its_measured_visible_basin_floor() -> None:
    expected = {
        HS005_MODERN_APARTMENT_SINK_SCENE: (0.895, 0.90902124),
        HS005_MODERN_KITCHEN_SINK_SCENE: (0.71930, 0.73332124),
        HS005_REALISTIC_KITCHEN_SCENE_SINK_SCENE: (0.64815, 0.66217124),
        HS005_APPARTEMENT_KITCHEN_SINK_SCENE: (0.75207, 0.76609124),
    }
    assert _HS005_RESET_ROOT_Z_BY_SCENE == expected
    source = (
        ROOT / "handson/handson/sim/isaaclab/abc_collection_env.py"
    ).read_text(encoding="utf-8")
    assert "generic table colliders disabled" in source
    assert "CreateCollisionEnabledAttr().Set(False)" in source


def test_hs005_robot_home_lifts_both_hands_above_every_sink_rim() -> None:
    assert _HS005_CLEAR_RIGHT_START_TCP[2] == pytest.approx(1.15)
    assert _HS005_CLEAR_LEFT_START_TCP[2] == pytest.approx(1.15)
    np.testing.assert_allclose(
        _HS005_CLEAR_ARM_HOME["left"],
        -_HS005_CLEAR_ARM_HOME["right"]
        * np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
        rtol=0.0,
        atol=1.0e-7,
    )
    env = _environment(randomize=False)
    try:
        env._sample_reset_state(clear_manual_request=True)
        np.testing.assert_array_equal(
            env._right_tcp_pose,
            _HS005_CLEAR_RIGHT_START_TCP,
        )
        np.testing.assert_array_equal(
            env._left_tcp_pose,
            _HS005_CLEAR_LEFT_START_TCP,
        )
    finally:
        env.close()


def _install_synthetic_physics(env: HandsOnABCIsaacLabEnv):
    tokens = {name: object() for name in ("left_bottle", "right_brush", "brush_bottle")}
    forces = {id(token): 1.0 for token in tokens.values()}
    poses = {
        "hs005_bottle": np.asarray([0.05, 0.0, 0.924, 1.0, 0.0, 0.0, 0.0]),
        "hs005_brush": np.asarray([-0.29, -0.16, 0.928, 1.0, 0.0, 0.0, 0.0]),
    }
    velocities = {
        "hs005_bottle": np.zeros(6),
        "hs005_brush": np.zeros(6),
    }
    geometry = {"value": (np.zeros(3), 0.05, 0.0, 0)}
    clear = {"value": (False, False)}
    env._hs005_contact_views = tokens
    env._current_body_pose = types.MethodType(lambda self, name: poses.get(name), env)
    env._physics_body_velocity = types.MethodType(lambda self, name: velocities.get(name), env)
    env._c04_contact_force = types.MethodType(lambda self, view: forces.get(id(view), 0.0), env)
    env._hs005_bristle_geometry = types.MethodType(lambda self: geometry["value"], env)
    env._hs005_hand_clear = types.MethodType(lambda self: clear["value"], env)
    env._hs005_brush_grasp_state = types.MethodType(
        lambda self: (True, False, 0.0, 2),
        env,
    )
    return forces, geometry, clear


def test_hs005_safe_brush_grasp_requires_opposing_handle_tips_and_avoids_bristles() -> None:
    brush = np.eye(4, dtype=np.float64)
    safe_tips = {
        "right_finger1_tip": np.asarray([0.065, 0.0215, 0.0]),
        "right_finger2_tip": np.asarray([0.065, -0.0215, 0.0]),
    }
    safe, bristle_touch, error, tip_count = (
        HandsOnABCIsaacLabEnv._hs005_safe_brush_grasp_geometry(
            brush,
            safe_tips,
        )
    )
    assert safe is True
    assert bristle_touch is False
    assert error <= 0.008
    assert tip_count == 2

    # Runtime tip positions are visible-mesh centres.  Their conservative
    # 7.5 mm mesh radius leaves a 0.95 mm surface gap at the observed 8.45 mm
    # centre distance, without changing the original 8 mm near-face gate.
    live_tip_centres = {
        "right_finger1_tip": np.asarray([0.065, 0.02245, 0.0]),
        "right_finger2_tip": np.asarray([0.065, -0.02245, 0.0]),
    }
    live_safe, live_bristle, live_error, live_count = (
        HandsOnABCIsaacLabEnv._hs005_safe_brush_grasp_geometry(
            brush,
            live_tip_centres,
        )
    )
    assert live_safe is True
    assert live_bristle is False
    assert live_error == pytest.approx(0.00095)
    assert live_count == 2

    # The unchanged 12 mm centre admissibility gate still rejects a centre
    # just 0.1 mm beyond it, even though the visible mesh has finite radius.
    too_far = {
        **live_tip_centres,
        "right_finger1_tip": np.asarray([0.065, 0.0261, 0.0]),
    }
    assert HandsOnABCIsaacLabEnv._hs005_safe_brush_grasp_geometry(
        brush,
        too_far,
    )[0] is False

    one_sided = {
        "right_finger1_tip": np.asarray([0.065, 0.0215, 0.0]),
        "right_finger2_tip": np.asarray([0.065, 0.0200, 0.0]),
    }
    assert HandsOnABCIsaacLabEnv._hs005_safe_brush_grasp_geometry(
        brush,
        one_sided,
    )[0] is False

    unsafe_tips = {
        **safe_tips,
        "right_finger3_tip": np.asarray([-0.0555, 0.0215, 0.0]),
    }
    safe, bristle_touch, _error, _tip_count = (
        HandsOnABCIsaacLabEnv._hs005_safe_brush_grasp_geometry(
            brush,
            unsafe_tips,
        )
    )
    assert safe is False
    assert bristle_touch is True


def test_hs005_success_requires_ordered_physical_contact_release_and_stability() -> None:
    env = _environment()
    try:
        forces, geometry, clear = _install_synthetic_physics(env)
        samples = (
            (0.060, 0),
            (0.060, 0),
            (0.125, 1),
            (0.190, 2),
            (0.190, 2),
            (0.190, 2),
        )
        for index, (local_z, band) in enumerate(samples):
            geometry["value"] = (
                np.asarray([0.02 * index, 0.0, 1.0]),
                local_z,
                0.0,
                band,
            )
            env._update_hs005_physical_events()
        assert env._hs005_order_progress == 3
        assert np.all(env._hs005_coverage_mask == 1)
        assert env._hs005_path_length > 0.08
        assert env._hs005_success_now() is False

        for key in list(forces):
            forces[key] = 0.0
        clear["value"] = (True, True)
        required = int(np.ceil(0.5 / env._physics_dt))
        for _ in range(required):
            env._update_hs005_physical_events()
        assert env._hs005_success_now() is True
        info = env._task_info()
        assert info["hs005_success_now"] is True
        assert info["hs005_right_safe_grasp_seen"] is True
        assert info["hs005_release_seen"] is True
        assert info["hs005_stability_seconds"] >= 0.5
    finally:
        env.close()


def test_hs005_wrong_band_order_is_failure_not_false_success() -> None:
    env = _environment()
    try:
        _forces, geometry, _clear = _install_synthetic_physics(env)
        geometry["value"] = (np.zeros(3), 0.095, 0.0, 1)
        env._update_hs005_physical_events()
        assert env._hs005_order_violation is True
        assert env._compute_failure() is True
        assert env._hs005_success_now() is False
    finally:
        env.close()


def test_hs005_success_classifier_has_ten_positive_and_ten_negative_trials() -> None:
    report = run_success_classifier_audit()
    assert report["positive_trial_count"] == 10
    assert report["negative_trial_count"] == 10
    assert report["false_positive_count"] == 0
    assert report["false_negative_count"] == 0
    assert report["pass"] is True


def test_hs005_zero_contact_never_advances_or_succeeds() -> None:
    env = _environment()
    try:
        forces, geometry, clear = _install_synthetic_physics(env)
        for key in list(forces):
            forces[key] = 0.0
        geometry["value"] = (np.asarray([0.20, 0.0, 1.0]), 0.14, 0.0, 2)
        clear["value"] = (True, True)
        for _ in range(int(np.ceil(0.75 / env._physics_dt))):
            env._update_hs005_physical_events()
        assert env._hs005_order_progress == 0
        assert not np.any(env._hs005_coverage_mask)
        assert env._hs005_path_length == 0.0
        assert env._hs005_success_now() is False
        assert env._task_info()["hs005_success_now"] is False
    finally:
        env.close()


def test_hs005_scrub_path_requires_live_monotonic_contact() -> None:
    env = _environment()
    try:
        forces, geometry, _clear = _install_synthetic_physics(env)
        geometry["value"] = (np.zeros(3), 0.05, 0.0, 0)
        env._update_hs005_physical_events()
        assert env._hs005_order_progress == 1
        assert env._hs005_path_length == 0.0

        for key in list(forces):
            forces[key] = 0.0
        geometry["value"] = (np.asarray([0.20, 0.0, 0.0]), 0.05, 0.0, 0)
        env._update_hs005_physical_events()
        assert env._hs005_order_progress == 1
        assert env._hs005_path_length == 0.0

        for key in list(forces):
            forces[key] = 1.0
        geometry["value"] = (np.asarray([0.40, 0.0, 0.0]), 0.05, 0.0, 0)
        env._update_hs005_physical_events()
        assert env._hs005_path_length == 0.0

        geometry["value"] = (np.asarray([0.42, 0.0, 0.0]), 0.05, 0.0, 0)
        env._update_hs005_physical_events()
        geometry["value"] = (np.asarray([0.44, 0.0, 0.0]), 0.095, 0.0, 1)
        env._update_hs005_physical_events()
        assert env._hs005_order_progress == 2
        assert env._hs005_path_length == pytest.approx(0.04)

        geometry["value"] = (np.asarray([0.46, 0.0, 0.0]), 0.05, 0.0, 0)
        env._update_hs005_physical_events()
        assert env._hs005_order_violation is True
        assert env._hs005_path_length == pytest.approx(0.04)
        assert env._compute_failure() is True
        assert env._hs005_success_now() is False
    finally:
        env.close()


def _policy_from_live_reset_truth(
    bottle_mass_scale: float = 1.0,
) -> tuple[HS005AutomatedPolicy, np.ndarray, np.ndarray]:
    # Frame-zero truth from the complete v4 PhysX episode (XYZ+WXYZ).
    bottle = np.asarray(
        [
            0.0381648391,
            -0.00800731964,
            0.922506988,
            0.999759436,
            -0.000000239663137,
            0.000000420126042,
            -0.0219300445,
        ]
    )
    brush = np.asarray(
        [
            -0.26460421,
            -0.108317,
            0.92781425,
            0.99361354,
            -0.07224819,
            -0.00628578,
            0.08644528,
        ]
    )
    command_tcp = np.asarray(
        [
            [-0.10, -0.36, 1.07, 0.0, 1.0, 0.0, 0.0],
            [-0.10, 0.36, 1.07, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    physical_tcp = np.asarray(
        [
            [-0.40918088, -0.40413451, 0.99749988, 0.68928349, 0.15776011, 0.68928349, -0.15775998],
            [-0.40918073, 0.40413457, 0.99749994, 0.68928349, -0.15776013, 0.68928343, 0.15776020],
        ]
    )
    return (
        HS005AutomatedPolicy(
            bottle,
            brush,
            command_tcp,
            physical_tcp,
            np.ones(2, dtype=np.uint8),
            np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.725, bottle_mass_scale, 1.0]
            ),
        ),
        bottle,
        brush,
    )


def _pose_rotation(pose: np.ndarray) -> np.ndarray:
    w, x, y, z = pose[3:7] / np.linalg.norm(pose[3:7])
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _asset_local_point(asset_pose: np.ndarray, local_xyz: list[float]) -> np.ndarray:
    return asset_pose[:3] + _pose_rotation(asset_pose) @ np.asarray(local_xyz)


def _world_to_asset_point(asset_pose: np.ndarray, world_xyz: np.ndarray) -> np.ndarray:
    return _pose_rotation(asset_pose).T @ (np.asarray(world_xyz) - asset_pose[:3])


def _pose_homogeneous(pose: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _pose_rotation(pose)
    matrix[:3, 3] = np.asarray(pose, dtype=np.float64)[:3]
    return matrix


def test_hs005_closed_loop_policy_emits_teleoperation_sized_steps() -> None:
    policy, _bottle, _brush = _policy_from_live_reset_truth()
    report = policy.reachability_report
    physical = np.asarray(
        [
            report["per_side"]["right"]["reset_palm_pose_wxyz"],
            report["per_side"]["left"]["reset_palm_pose_wxyz"],
        ]
    )
    action = policy.action(physical, np.ones(2, dtype=np.uint8))
    commands = np.stack((action[:7], action[7:14])).astype(np.float64)
    assert np.all(
        np.linalg.norm(commands[:, :3] - physical[:, :3], axis=1)
        <= MAX_COMMAND_TRANSLATION_STEP_M + 1.0e-7
    )
    for command, actual in zip(commands, physical, strict=True):
        cosine = abs(float(np.dot(command[3:7], actual[3:7])))
        angle = 2.0 * np.arccos(np.clip(cosine, 0.0, 1.0))
        assert angle <= MAX_COMMAND_ROTATION_STEP_RAD + 1.0e-6
    assert not np.allclose(commands, policy.last_desired_target_palm)


def test_hs005_automatic_policy_emits_only_raw_robot_actions() -> None:
    policy, bottle, brush = _policy_from_live_reset_truth()
    assert policy.total_frames > 300
    assert policy.total_frames <= 1350
    actions = [policy.action() for _ in range(policy.total_frames)]
    assert all(action.shape == (54,) for action in actions)
    assert all(action.dtype == np.float32 for action in actions)

    # The shared IsaacLab controller consumes world-frame physical palm poses
    # directly.  The task-relative targets must therefore round-trip without a
    # reset-time calibration transform.
    bottle_contact = _asset_local_point(
        bottle,
        list(LEFT_BOTTLE_NEAR_SURFACE_LOCAL),
    )

    phase_start: dict[str, int] = {}
    phase_end: dict[str, int] = {}
    frame = 0
    for phase, frames in policy.waypoint_schedule:
        phase_start[phase] = frame
        frame += frames
        phase_end[phase] = frame - 1
    right_approach_pose = policy.command_to_palm(
        "right", actions[phase_start["approach"]][:7]
    )
    left_approach_pose = policy.command_to_palm(
        "left", actions[phase_start["approach"]][7:14]
    )
    right_contact_pose = policy.command_to_palm(
        "right", actions[phase_start["contact"]][:7]
    )
    left_contact_pose = policy.command_to_palm(
        "left", actions[phase_start["contact"]][7:14]
    )
    right_forward = _pose_rotation(right_contact_pose)[:, 2]
    left_forward = _pose_rotation(left_contact_pose)[:, 2]
    assert right_forward[1] > 0.0
    assert left_forward[1] < 0.0
    assert right_approach_pose[1] < 0.0
    assert left_approach_pose[1] > 0.0
    expected_right_contact = _pose_homogeneous(brush) @ RIGHT_BRUSH_T_PALM
    np.testing.assert_allclose(
        _pose_homogeneous(right_contact_pose),
        expected_right_contact,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        _pose_homogeneous(right_approach_pose),
        _pose_homogeneous(right_contact_pose),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        bottle_contact - left_approach_pose[:3],
        LEFT_PALM_APPROACH_REACH_M * left_forward,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        bottle_contact - left_contact_pose[:3],
        LEFT_PALM_CONTACT_REACH_M * left_forward,
        atol=1.0e-6,
    )
    scrub_indices = tuple(
        phase_end[phase]
        for phase in (
            "scrub_lower_a",
            "scrub_lower_b",
            "scrub_middle",
            "scrub_upper",
        )
    )
    scrub_local = np.asarray(
        [
            _world_to_asset_point(
                bottle,
                policy.predicted_bristle_for_command(actions[index][:7]),
            )
            for index in scrub_indices
        ]
    )
    np.testing.assert_allclose(
        scrub_local,
        SCRUB_BRISTLE_TARGETS_BOTTLE_LOCAL,
        atol=1.0e-6,
    )
    # A single mass-selected physical grip is held without changing the left
    # role or palm target anywhere in the complete scrub.
    np.testing.assert_allclose(
        actions[phase_end["scrub_lower_b"]][7:14],
        actions[phase_end["scrub_middle"]][7:14],
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        actions[phase_end["scrub_middle"]][7:14],
        actions[phase_end["scrub_upper"]][7:14],
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        actions[phase_end["scrub_middle"]][34:54],
        actions[phase_end["scrub_upper"]][34:54],
        atol=1.0e-7,
    )
    release_palm = _pose_homogeneous(brush) @ RIGHT_BRUSH_T_PALM
    release_palm[2, 3] += BRUSH_PLACE_PALM_LIFT_M
    release_brush = release_palm @ RIGHT_CARRIED_PALM_T_BRUSH
    expected_release_bristle = (
        release_brush
        @ np.asarray([*BRUSH_BRISTLE_LOCAL, 1.0], dtype=np.float64)
    )[:3]
    np.testing.assert_allclose(
        policy.predicted_bristle_for_command(
            actions[phase_end["place_brush"]][:7]
        ),
        expected_release_bristle,
        atol=1.0e-6,
    )
    # The stabilizer remains physically closed through the final scrub, then
    # gently loosens and retreats while the right hand lifts the brush clear.
    # This leaves the bottle a long passive settling interval before the right
    # hand releases the placed brush.
    left_lift_start = policy.command_to_palm(
        "left", actions[phase_start["lift_from_scrub"]][7:14]
    )
    left_lift_end = policy.command_to_palm(
        "left", actions[phase_end["lift_from_scrub"]][7:14]
    )
    assert np.linalg.norm(left_lift_end[:3] - bottle[:3]) > np.linalg.norm(
        left_lift_start[:3] - bottle[:3]
    )
    assert np.linalg.norm(
        actions[phase_end["lift_from_scrub"]][34:54]
    ) < np.linalg.norm(actions[phase_start["lift_from_scrub"]][34:54])
    left_return_start = policy.command_to_palm(
        "left", actions[phase_start["return_high"]][7:14]
    )
    left_return_end = policy.command_to_palm(
        "left", actions[phase_end["return_high"]][7:14]
    )
    assert np.linalg.norm(left_return_end[:3] - bottle[:3]) > np.linalg.norm(
        left_return_start[:3] - bottle[:3]
    )
    assert np.linalg.norm(
        actions[phase_end["return_high"]][34:54]
    ) < np.linalg.norm(actions[phase_start["return_high"]][34:54])
    np.testing.assert_allclose(
        actions[phase_end["place_brush"]][34:54], OPEN_HAND, atol=1.0e-7
    )
    assert np.linalg.norm(
        actions[phase_end["release"]][14:34]
    ) < np.linalg.norm(actions[phase_start["release"]][14:34])
    np.testing.assert_allclose(
        actions[phase_end["release"]][34:54],
        actions[phase_start["release"]][34:54],
        atol=1.0e-7,
    )
    left_final_retreat_start = policy.command_to_palm(
        "left", actions[phase_start["retreat"]][7:14]
    )
    left_final_retreat_end = policy.command_to_palm(
        "left", actions[phase_end["retreat"]][7:14]
    )
    np.testing.assert_allclose(
        left_final_retreat_end, left_final_retreat_start, atol=1.0e-7
    )
    np.testing.assert_allclose(
        actions[phase_end["retreat"]][34:54],
        actions[phase_start["retreat"]][34:54],
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        scrub_local[:, 2],
        [0.060, 0.060, 0.125, 0.190],
        atol=1.0e-6,
    )
    radial_error = np.abs(np.linalg.norm(scrub_local[:, :2], axis=1) - 0.0347)
    assert np.all(radial_error <= 0.035)
    clipped_path = sum(
        min(float(np.linalg.norm(end - start)), 0.02)
        for start, end in zip(scrub_local[:-1], scrub_local[1:])
    )
    assert clipped_path == pytest.approx(0.06)
    transit_actions = np.asarray(
        [
            actions[index][:3]
            for index in range(
                phase_start["transit_high"],
                phase_end["transit_high"] + 1,
            )
        ]
    )
    assert np.max(np.linalg.norm(np.diff(transit_actions, axis=0), axis=1)) < 0.007
    assert not hasattr(policy, "environment")
    assert not hasattr(policy, "stage")


def test_hs005_bottle_grip_uses_public_mass_not_seed_identity() -> None:
    light_policy, _bottle, _brush = _policy_from_live_reset_truth(0.90)
    heavy_policy, _bottle, _brush = _policy_from_live_reset_truth(1.10)
    assert light_policy.bottle_mass_scale == pytest.approx(0.90)
    assert heavy_policy.bottle_mass_scale == pytest.approx(1.10)
    assert light_policy.left_bottle_work_grip_scale == pytest.approx(0.75)
    assert heavy_policy.left_bottle_work_grip_scale == pytest.approx(0.90)
    assert np.linalg.norm(heavy_policy.action()[34:54]) == pytest.approx(0.0)
    assert np.linalg.norm(light_policy.action()[34:54]) == pytest.approx(0.0)
    for _ in range(170):
        heavy_action = heavy_policy.action()
        light_action = light_policy.action()
    assert np.linalg.norm(heavy_action[34:54]) > np.linalg.norm(light_action[34:54])
    assert heavy_policy.reachability_report["left_bottle_grip_selection"] == (
        "public_reset_bottle_mass_scale"
    )


def test_hs005_task_to_tcp_contract_is_reachable_and_keeps_same_side_roles() -> None:
    policy, _bottle, _brush = _policy_from_live_reset_truth()
    report = policy.reachability_report
    assert report["convention"] == "world_T_command=world_T_target_palm"
    assert report["role_by_side"] == TASK_ROLE_BY_SIDE == {
        "right": "brush",
        "left": "bottle",
    }
    assert report["empirical_world_residual_prewarp"] is False
    assert report["shared_robot_stack_modified"] is False
    assert report["maximum_translation_m"] <= MAX_TASK_PALM_TRANSLATION_M

    first_action = policy.action()
    right_approach = policy.command_to_palm("right", first_action[:7])
    left_approach = policy.command_to_palm("left", first_action[7:14])
    assert right_approach[1] < 0.0
    assert left_approach[1] > 0.0

    source = (
        ROOT
        / "taskdev/tasks/bench320_hs005_sink_bottle_stabilized_scrub/automated_policy.py"
    ).read_text(encoding="utf-8")
    assert "TASK_FRAME_PREWARP" not in source
    assert "world_T_command = world_T_target_palm" in source
    assert "palm_T_command" not in source
    assert tuple(phase for phase, _frames in policy.waypoint_schedule) == (
        "lift_clear",
        "orient_high",
        "stage_high",
        "preshape",
        "approach",
        "contact",
        "grasp_a",
        "grasp_b",
        "pickup_lift",
        "transit_translate",
        "rotate_high",
        "transit_high",
        "descend_lower_clear",
        "scrub_lower_a",
        "scrub_lower_b",
        "scrub_middle",
        "scrub_upper",
        "lift_from_scrub",
        "return_high",
        "place_brush",
        "release",
        "retreat",
        "stability",
    )


def test_hs005_task_to_tcp_contract_fails_closed_without_live_palm_truth() -> None:
    policy, bottle, brush = _policy_from_live_reset_truth()
    report = policy.reachability_report
    command = np.asarray(
        [
            report["per_side"]["right"]["reset_command_pose_wxyz"],
            report["per_side"]["left"]["reset_command_pose_wxyz"],
        ]
    )
    physical = np.asarray(
        [
            report["per_side"]["right"]["reset_palm_pose_wxyz"],
            report["per_side"]["left"]["reset_palm_pose_wxyz"],
        ]
    )
    with pytest.raises(ValueError, match="both live PhysX palms"):
        HS005AutomatedPolicy(
            bottle,
            brush,
            command,
            physical,
            np.asarray([1, 0], dtype=np.uint8),
        )


def test_hs005_terminal_and_truncated_contract_stays_gymnasium_five_tuple() -> None:
    env = _environment(time_limit=1.0 / 30.0)
    try:
        terminated = False
        truncated = False
        info = {}
        for _ in range(3):
            _observation, _reward, terminated, truncated, info = (
                env.step_with_current_joint_targets()
            )
            if terminated or truncated:
                break
        assert terminated is False
        assert truncated is True
        assert info["succeed"] is False
    finally:
        env.close()


def test_hs005_action_aligned_zarr_contains_truth_hidden_state_and_three_cameras(
    tmp_path: Path,
) -> None:
    state = {
        key: np.zeros((3,), dtype=np.float64)
        for key in HIDDEN_KEYS
    }
    state["hs005_valid"] = np.ones((1,), dtype=np.uint8)
    physical = {
        "physx_robot_joint_target": np.zeros((2, 27), dtype=np.float64),
        "physx_substep_count": np.asarray([8], dtype=np.int64),
    }
    observation = {
        "state": state,
        "images": {
            "random_camera": np.full((480, 640, 3), 80, dtype=np.uint8),
            "wrist_left": np.full((240, 320, 3), 90, dtype=np.uint8),
            "wrist_right": np.full((240, 320, 3), 100, dtype=np.uint8),
        },
        **physical,
    }
    writer = HS005EpisodeWriter(tmp_path / "episode", {"seed": 17})
    writer.append(
        observation,
        np.zeros((54,), dtype=np.float32),
        observation,
        0.0,
        False,
        False,
        {"succeed": False},
    )
    manifest = writer.finalize({"collection_pass": False})
    assert manifest["frame_count"] == 1
    assert manifest["zarr_sha256"] == sha256_tree(tmp_path / "episode/replay.zarr")

    import zarr

    root = zarr.open_group(str(tmp_path / "episode/replay.zarr"), mode="r")
    data = root["data"]
    assert data["raw_action"].shape == (1, 54)
    assert set(data["cameras"].array_keys()) == {"main", "wrist_left", "wrist_right"}
    assert set(load_zarr_frame(data["physical/next"], 0)) == set(physical)
    assert set(data["task_hidden/next"].array_keys()) == set(HIDDEN_KEYS)


def test_hs005_runners_are_physx_only_and_fresh_replay_forbids_state_restore() -> None:
    task_dir = ROOT / "taskdev/tasks/bench320_hs005_sink_bottle_stabilized_scrub"
    collection = (task_dir / "vertical_slice_runner.py").read_text(encoding="utf-8")
    replay = (task_dir / "fresh_replay_runner.py").read_text(encoding="utf-8")
    assert 'HANDSON_ENABLE_ISAACSIM_NEWTON"] = "0"' in collection
    assert 'physics_backend="physx"' in collection
    assert "RecordedArticulatedDrivePath" in replay
    assert "run_articulated_drive_control_step" in replay
    assert "restore_recorded_initial_state" not in replay
    assert "restore_recorded_visual_state" not in replay
    assert "state_restore_after_action_zero" in replay
    assert 'manifest.get("collection_pass") is not True' in replay
    assert '"task_complete": rc == 0' in replay
    assert 'getattr(exc, "diagnostics", None)' in collection
    assert '"final_task_info": final_task_info' in collection
    assert 'observation["physx_tcp_state"]' in collection
    assert 'observation["physx_tcp_valid"]' in collection
    assert '"git_commit": commit' in collection
    assert '"seed": int(args.seed)' in collection
    assert "**metadata" in collection
    assert '"task_tcp_contract_sha256"' in collection
    assert "environment._apply_action(raw_action)" in replay
    assert "environment._apply_joint_targets()" not in replay
    assert replay.count("include_nonreplayable_contact_outputs=False") == 2
