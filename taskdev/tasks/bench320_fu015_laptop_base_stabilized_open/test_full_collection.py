from __future__ import annotations

import ast
from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from handson.sim.isaaclab.fu015_contract import (
    ACTION_DIM,
    ASSET_RELATIVE_PATH,
    ASSET_SHA256,
    BASE_LIFT_MAX_M,
    BASE_TRANSLATION_MAX_M,
    BILATERAL_OWNER_HOLD_SUBSTEPS,
    FORBIDDEN_CONTACT_FORCE_N,
    FU015BilateralOwnerGate,
    FU015PhysicalReadback,
    FU015ScriptedPolicy,
    FU015SuccessTracker,
    HINGE_INITIAL_FRACTION_RANGE,
    LEFT_HOME,
    MASS_SCALE_RANGE,
    REQUIRED_CONTACT_FORCE_N,
    ROOT_XY_HALF_RANGE_M,
    ROOT_YAW_HALF_RANGE_DEG,
    SUCCESS_HOLD_STEPS,
    TASK_ID,
    TASK_NAME,
    TASK_PROMPT,
    _RIGHT_REFERENCE_TCP_Y_M,
    _right_tcp_pose_for_opening,
    _tilt_left_pose_about_support,
    evaluate_physical_readback,
    sample_reset_state,
)


TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]


def _successful_readback(**overrides) -> FU015PhysicalReadback:
    values = {
        "hinge_angle_rad": math.radians(-105.0),
        "base_position_m": (0.02, 0.0, 0.891),
        "reset_base_position_m": (0.02, 0.0, 0.891),
        "base_linear_speed_mps": 0.0,
        "base_angular_speed_radps": 0.0,
        "display_linear_speed_mps": 0.0,
        "display_angular_speed_radps": 0.0,
        "left_base_contact_force_n": REQUIRED_CONTACT_FORCE_N + 0.1,
        "right_display_contact_force_n": REQUIRED_CONTACT_FORCE_N + 0.1,
        "left_touchpad_contact_force_n": 0.0,
        "right_base_contact_force_n": 0.0,
        "robot_table_contact_force_n": 0.0,
    }
    values.update(overrides)
    return FU015PhysicalReadback(**values)


def test_authoritative_identity_and_exact_asset_are_unchanged() -> None:
    assert TASK_ID == "B320-FU015"
    assert TASK_NAME == "t16_open_laptop"
    assert "left hand" in TASK_PROMPT.lower()
    assert "touchpad" in TASK_PROMPT.lower()
    assert "right hand" in TASK_PROMPT.lower()
    asset = REPOSITORY_ROOT / ASSET_RELATIVE_PATH
    import hashlib

    assert asset.is_file()
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == ASSET_SHA256


def test_seeded_reset_is_deterministic_and_inside_source_row_ranges() -> None:
    samples = [sample_reset_state(seed) for seed in range(128)]
    assert samples == [sample_reset_state(seed) for seed in range(128)]
    for sample in samples:
        assert abs(sample.root_x_m - 0.02) <= ROOT_XY_HALF_RANGE_M
        assert abs(sample.root_y_m) <= ROOT_XY_HALF_RANGE_M
        assert abs(sample.root_yaw_deg + 90.0) <= ROOT_YAW_HALF_RANGE_DEG
        assert HINGE_INITIAL_FRACTION_RANGE[0] <= sample.hinge_fraction <= HINGE_INITIAL_FRACTION_RANGE[1]
        assert 0.45 <= sample.friction <= 0.85
        assert MASS_SCALE_RANGE[0] <= sample.mass_scale <= MASS_SCALE_RANGE[1]


def test_physical_success_requires_exact_bimanual_roles_and_hold() -> None:
    readback = _successful_readback()
    metrics = evaluate_physical_readback(readback)
    assert metrics["success_now"] is True
    assert metrics["failure"] is False

    tracker = FU015SuccessTracker()
    for _ in range(SUCCESS_HOLD_STEPS - 1):
        assert tracker.update(readback) == (False, False)
    assert tracker.update(readback) == (True, False)
    tracker.reset()
    assert tracker.consecutive_steps == 0
    assert tracker.failure_latched is False


def test_bilateral_owner_gate_requires_full_control_interval_and_resets() -> None:
    gate = FU015BilateralOwnerGate()
    assert BILATERAL_OWNER_HOLD_SUBSTEPS == 4
    for _ in range(BILATERAL_OWNER_HOLD_SUBSTEPS - 1):
        assert gate.update(True) is False
    assert gate.update(False) is False
    assert gate.consecutive_substeps == 0
    for _ in range(BILATERAL_OWNER_HOLD_SUBSTEPS - 1):
        assert gate.update(True) is False
    assert gate.update(True) is True
    assert gate.ready is True
    assert gate.update(False) is False
    assert gate.consecutive_substeps == 0


@pytest.mark.parametrize(
    ("overrides", "expected_failure"),
    [
        ({"hinge_angle_rad": math.radians(-80.0)}, False),
        ({"left_base_contact_force_n": 0.0}, False),
        ({"right_display_contact_force_n": 0.0}, False),
        ({"left_touchpad_contact_force_n": FORBIDDEN_CONTACT_FORCE_N + 0.01}, True),
        ({"right_base_contact_force_n": FORBIDDEN_CONTACT_FORCE_N + 0.01}, True),
        ({"robot_table_contact_force_n": FORBIDDEN_CONTACT_FORCE_N + 0.01}, True),
        ({"base_position_m": (0.02 + BASE_TRANSLATION_MAX_M + 0.001, 0.0, 0.891)}, True),
        ({"base_position_m": (0.02, 0.0, 0.891 + BASE_LIFT_MAX_M + 0.001)}, True),
        ({"display_angular_speed_radps": 1.0}, False),
        ({"hinge_angle_rad": math.nan}, True),
    ],
)
def test_negative_semantics_fail_closed(overrides, expected_failure) -> None:
    metrics = evaluate_physical_readback(_successful_readback(**overrides))
    assert metrics["success_now"] is False
    assert metrics["failure"] is expected_failure


def test_scripted_policy_is_deterministic_bimanual_and_role_preserving() -> None:
    reset = sample_reset_state(73)
    first = FU015ScriptedPolicy(reset)
    second = FU015ScriptedPolicy(reset)
    phase_steps = (0, 59, 74, 89, 164, 179, 194, 224, 269, 449, 480)
    actions = [first.action(step) for step in phase_steps]
    assert all(action.shape == (ACTION_DIM,) for action in actions)
    assert all(np.all(np.isfinite(action)) for action in actions)
    assert all(
        np.array_equal(action, second.action(step))
        for action, step in zip(actions, phase_steps)
    )
    # Once both hands engage, the left stabilizer stays fixed while only the
    # right target follows the opening arc.
    assert np.allclose(first.action(269)[7:14], first.action(415)[7:14])
    assert first.action(375)[0] > first.action(269)[0]
    assert first.action(375)[2] > first.action(269)[2]
    assert all(
        np.isclose(np.linalg.norm(action[tcp_slice][3:7]), 1.0)
        for action in actions
        for tcp_slice in (slice(0, 7), slice(7, 14))
    )
    metadata = first.metadata()
    assert metadata["left_role"] == "base_front_edge_avoiding_touchpad"
    assert metadata["right_role"] == "display_top_edge_open_hinge"
    assert metadata["left_contact_control"] == (
        "outer_descent_low_lateral_press_front_retreat_force_relief"
    )


def test_left_stabilizer_stages_outside_then_presses_front_edge() -> None:
    nominal_reset = replace(
        sample_reset_state(73),
        root_x_m=0.02,
        root_y_m=0.0,
        root_yaw_deg=-90.0,
    )
    policy = FU015ScriptedPolicy(nominal_reset)
    left_home_high = policy.action(14)[7:14]
    left_high = policy.action(44)[7:14]
    left_turn_mid = policy.action(59)[7:14]
    left_pre = policy.action(74)[7:14]
    left_reorientation_dwell = policy.action(89)[7:14]
    left_edge_outside = policy.action(164)[7:14]
    left_above_contact = policy.action(179)[7:14]
    left_press = policy.action(194)[7:14]
    left_stabilizer_grip = policy.action(194)[34:]

    assert left_home_high[0] == pytest.approx(LEFT_HOME[0])
    assert left_home_high[1] == pytest.approx(LEFT_HOME[1])
    assert left_home_high[2] == pytest.approx(1.200)
    assert left_high[0] == pytest.approx(-0.245)
    assert left_high[1] == pytest.approx(0.075)
    assert left_high[2] == pytest.approx(1.200)
    assert np.dot(left_high[3:], left_turn_mid[3:]) > 0.5
    assert np.dot(left_turn_mid[3:], left_pre[3:]) > 0.5
    for reset_seed in (15301, 15302):
        reset_policy = FU015ScriptedPolicy(sample_reset_state(reset_seed))
        reset_turn = [
            reset_policy.action(step)[10:14] for step in (44, 59, 74)
        ]
        assert np.dot(reset_turn[0], reset_turn[1]) > 0.5
        assert np.dot(reset_turn[1], reset_turn[2]) > 0.5
    assert left_pre[0] == pytest.approx(-0.245)
    expected_tilted_pre = _tilt_left_pose_about_support(
        left_pre, nominal_reset
    )
    assert np.allclose(left_reorientation_dwell, expected_tilted_pre)
    assert left_reorientation_dwell[0] < left_pre[0]
    assert left_reorientation_dwell[2] < left_pre[2]
    assert left_pre[1] == pytest.approx(0.075)
    assert left_high[2] == pytest.approx(left_pre[2])
    assert not np.allclose(left_high[3:], left_pre[3:])
    assert left_edge_outside[2] < left_pre[2]
    assert left_above_contact[2] == pytest.approx(left_edge_outside[2])
    assert left_above_contact[0] > left_edge_outside[0]
    assert left_press[2] < left_above_contact[2]
    assert np.allclose(left_stabilizer_grip[:4], [1.55, 0.92, 1.50, 1.50])
    assert np.allclose(left_stabilizer_grip[4:8], [0.68, 0.00, 0.90, 0.78])
    assert np.allclose(left_stabilizer_grip[12:16], [1.40, 0.00, 1.35, 1.35])
    assert np.allclose(left_stabilizer_grip[16:20], [1.48, 0.00, 1.49, 1.50])


def test_right_grasp_stops_at_elevated_display_edge() -> None:
    nominal_reset = replace(
        sample_reset_state(73),
        root_x_m=0.02,
        root_y_m=0.0,
        root_yaw_deg=-90.0,
    )
    policy = FU015ScriptedPolicy(nominal_reset)
    right_high = policy.action(44)[:7]
    right_pre = policy.action(224)[:7]
    right_contact = policy.action(269)[:7]
    expected_contact = _right_tcp_pose_for_opening(
        nominal_reset.hinge_angle_deg,
        nominal_reset,
    )

    assert right_high[0] == pytest.approx(-0.180)
    assert right_high[1] == pytest.approx(-0.105)
    assert right_high[2] == pytest.approx(1.200)
    assert right_pre[2] == pytest.approx(1.200)
    assert np.allclose(right_contact, expected_contact)
    assert right_contact[1] == pytest.approx(_RIGHT_REFERENCE_TCP_Y_M)
    assert 0.940 < right_contact[2] < 1.000
    assert np.array_equal(policy.action(224)[14:34], policy.action(269)[14:34])


def test_scripted_policy_tracks_reset_yaw_without_changing_hand_actions() -> None:
    nominal_reset = replace(
        sample_reset_state(73),
        root_x_m=0.02,
        root_y_m=0.0,
        root_yaw_deg=-90.0,
    )
    yaw_delta_deg = 5.0
    randomized_reset = replace(
        nominal_reset,
        root_x_m=0.041,
        root_y_m=-0.017,
        root_yaw_deg=nominal_reset.root_yaw_deg + yaw_delta_deg,
    )
    nominal_policy = FU015ScriptedPolicy(nominal_reset)
    randomized_policy = FU015ScriptedPolicy(randomized_reset)
    yaw_delta = math.radians(yaw_delta_deg)
    rotation_xy = np.asarray(
        [
            [math.cos(yaw_delta), -math.sin(yaw_delta)],
            [math.sin(yaw_delta), math.cos(yaw_delta)],
        ]
    )
    # Phase endpoints are asset-relative task poses; fixed robot homes at step
    # zero remain fixed.  Hand commands are intentionally reset-invariant.
    assert np.array_equal(
        randomized_policy.action(0)[:14],
        nominal_policy.action(0)[:14],
    )
    for step in (59, 74, 89, 164, 179, 194, 224, 269, 449, 480):
        nominal_action = nominal_policy.action(step)
        randomized_action = randomized_policy.action(step)
        for tcp_slice in (slice(0, 7), slice(7, 14)):
            nominal_pose = nominal_action[tcp_slice]
            randomized_pose = randomized_action[tcp_slice]
            expected_xy = np.asarray(
                [randomized_reset.root_x_m, randomized_reset.root_y_m]
            ) + rotation_xy @ (nominal_pose[:2] - np.asarray([0.02, 0.0]))
            assert np.allclose(randomized_pose[:2], expected_xy, atol=1e-12)
            assert randomized_pose[2] == nominal_pose[2]
            w, x, y, z = nominal_pose[3:7]
            expected_quaternion = np.asarray(
                [
                    math.cos(yaw_delta / 2.0) * w
                    - math.sin(yaw_delta / 2.0) * z,
                    math.cos(yaw_delta / 2.0) * x
                    - math.sin(yaw_delta / 2.0) * y,
                    math.cos(yaw_delta / 2.0) * y
                    + math.sin(yaw_delta / 2.0) * x,
                    math.cos(yaw_delta / 2.0) * z
                    + math.sin(yaw_delta / 2.0) * w,
                ]
            )
            assert abs(
                float(np.dot(randomized_pose[3:7], expected_quaternion))
            ) == pytest.approx(1.0, abs=1e-12)
        assert np.array_equal(randomized_action[14:], nominal_action[14:])


def test_substep_failure_remains_latched_when_contact_clears_before_frame_end() -> None:
    runtime_path = (
        REPOSITORY_ROOT
        / "handson/handson/sim/isaaclab/fu015_collection_env.py"
    )
    tree = ast.parse(runtime_path.read_text())
    runtime_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "HandsOnFU015IsaacLabEnv"
    )
    methods = {
        node.name: ast.unparse(node)
        for node in runtime_class.body
        if isinstance(node, ast.FunctionDef)
    }
    latch = methods["_latch_fu015_failure"]
    after_substep = methods["_after_task_physics_substep"]
    frame_failure = methods["_compute_failure"]

    assert "self._fu015_failure_latched or metrics['failure']" in latch
    assert after_substep.index("self._refresh_fu015_metrics()") < after_substep.index(
        "self._latch_fu015_failure(metrics)"
    )
    assert "not self._fu015_failure_latched" in after_substep
    assert "return self._latch_fu015_failure(metrics)" in frame_failure


def test_task_config_and_ui_register_physx_only_three_camera_collection() -> None:
    from handson.runtime.collection_support import ISAACSIM_COLLECTION_TASKS
    from handson.tasks.mappings import CONFIG_MAPPING

    assert TASK_NAME in CONFIG_MAPPING
    config = CONFIG_MAPPING[TASK_NAME]()
    assert config.task_id == TASK_ID
    assert config.task_name == TASK_NAME
    assert config.proprio_keys[:2] == ["tcp_pose", "gripper_pose"]
    assert {"hinge_angle", "bimanual_role_contacts", "fu015_task_state"} <= set(config.proprio_keys)
    assert TASK_NAME in ISAACSIM_COLLECTION_TASKS

    multi_task = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/multi_task" / f"{TASK_NAME}.yaml").read_text()
    )
    assert multi_task["task_id"] == TASK_ID
    assert multi_task["camera_mapping"] == {
        "base": "ego",
        "wrist_left": "wrist_left",
        "wrist_right": "wrist_right",
    }
    assert multi_task["robot_type"] == "dual_arm"
    assert multi_task["sim_backend"] == "isaacsim"
    assert multi_task["physics_backend"] == "physx"

    ui_source = (REPOSITORY_ROOT / "collection_ui/server.py").read_text()
    assert TASK_NAME in ui_source
    assert f'"{TASK_NAME}": "ipad_2.png"' in ui_source
    assert "B320-FU015 Laptop Base Stabilized Open" in ui_source


def test_interface_only_runtime_exposes_action_and_proprio_contract() -> None:
    from handson.tasks.t16_open_laptop.config import TaskConfig

    env = TaskConfig().get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=True,
        launch_isaac=False,
        image_obs=False,
        seed=19,
        hand_type="wuji1",
        arm_type="rokae",
    )
    try:
        observation, info = env.reset(seed=19)
        assert getattr(env.unwrapped, "_fast_shutdown_on_close", False) is True
        assert env.action_space.shape == (ACTION_DIM,)
        assert observation["state"].shape == (86,)
        assert info["fu015_reset_seed"] == 19
        assert info["fu015_contact_contract_ready"] is False
        assert info["fu015_hinge_drive_stiffness"] == 36.0
        assert info["fu015_hinge_drive_max_force"] == 4.0
        assert info["fu015_right_robot_base_x_offset_m"] == 0.15
        assert info["fu015_left_robot_base_x_offset_m"] == 0.05
    finally:
        env.close()


def test_live_runtime_uses_exact_contact_points_and_no_action_phase_state_write() -> None:
    runtime_path = (
        REPOSITORY_ROOT
        / "handson/handson/sim/isaaclab/fu015_collection_env.py"
    )
    source = runtime_path.read_text()
    ast.parse(source)
    assert "omni.isaac." not in source
    assert "isaaclab_newton" not in source
    assert "create_rigid_contact_view" in source
    assert "get_contact_data" in source
    assert "TOUCHPAD_COLLIDER_PATH" in source
    assert "UsdPhysics.DriveAPI" in source
    assert "set_dof_position_targets" in source
    assert "GetTargetPositionAttr().Set" not in source
    assert "_physx_prefer_tensor_contact_capture = True" in source
    assert "self._fast_shutdown_on_close = True" in source
    assert "_configure_fu015_robot_cfg(robot_cfg, side)" in source
    assert "ComputeAlignedRange()" in source
    assert "ComputeAlignedBox().GetRange()" not in source
    assert "summarize_contact_forces_by_filter" in source
    assert "robot_table_contact_force_by_path_n" in source
    assert '"RayTracedLighting"' in source
    assert "UsdLux.DomeLight" in source
    assert "UsdLux.DistantLight" in source
    after_hook = source.split("def _after_task_physics_substep", 1)[1].split("def _compute_success_now", 1)[0]
    assert "set_root_transforms" not in after_hook
    assert "set_dof_positions" not in after_hook
    assert after_hook.index("owner_sample = bool") < after_hook.index(
        "_fu015_bilateral_owner_gate.update(owner_sample)"
    )
    assert after_hook.index(
        "_fu015_bilateral_owner_gate.update(owner_sample)"
    ) < after_hook.index("_set_hinge_drive_target")
    assert "_freeze_fu015_hinge_drive_at_readback" not in after_hook
    assert "_set_hinge_drive_target" in after_hook
    success_hook = source.split("def _compute_success_now", 1)[1].split(
        "def _compute_failure", 1
    )[0]
    assert "_fu015_bilateral_owner_gate.ready" in success_hook

    abc_runtime_source = (
        REPOSITORY_ROOT
        / "handson/handson/sim/isaaclab/abc_collection_env.py"
    ).read_text()
    robot_setup = abc_runtime_source.split(
        "self._apply_raised_hand_home(robot_cfg, side)", 1
    )[1].split("robot_root_paths.append", 1)[0]
    assert robot_setup.index("self._configure_task_robot_cfg(robot_cfg, side)") < (
        robot_setup.index("Articulation(robot_cfg)")
    )

    recorder_source = (REPOSITORY_ROOT / "scripts/record_demos_zarr.py").read_text()
    assert '"scripted_policy"' in recorder_source
    assert "FU015ScriptedPolicy" in recorder_source
    assert "fu015_policy_seed != reset_seed" in recorder_source
    assert 'readback=getattr(raw_env, "_fu015_last_readback", None)' in recorder_source
