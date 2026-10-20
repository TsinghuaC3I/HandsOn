from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from handson.sim.isaaclab.bc007_contract import (
    BC007_REPLAY_ASSIST_VERSION,
    LongRodEvaluator,
    LongRodFrame,
    LongRodStage,
    decode_replay_assist_state,
    encode_replay_assist_state,
    rod_axis_yaw_deg,
    rod_endpoints,
)
from handson.sim.isaaclab.bc007_env import BC007IsaacLabEnv
from handson.sim.isaaclab.bc007_scene import (
    ROD_HALF_LENGTH_M,
    ROD_RADIUS_M,
    ROD_SLEEP_THRESHOLD,
    ROD_START_CENTER,
    ROD_START_ORIENTATION_WXYZ,
    ROD_USD_SHA256,
    TABLE_TOP_Z,
    initial_orientation_wxyz,
    joint_local_frame_pair,
    sha256_file,
    target_orientation_wxyz,
    validate_asset_contract,
)


def _valid(**overrides) -> LongRodFrame:
    values = dict(
        left_role_owned=True,
        right_role_owned=True,
        center_height_m=1.08,
        yaw_progress_deg=90.0,
        yaw_error_deg=0.0,
        endpoint_height_difference_m=0.0,
        linear_speed_mps=0.0,
        angular_speed_radps=0.0,
        target_position_error_m=0.0,
        hand_separation_m=0.46,
        table_contact=False,
        strong_collision=False,
        finite_and_in_workspace=True,
        dropped=False,
    )
    values.update(overrides)
    return LongRodFrame(**values)


def _advance_to_reorienting(evaluator: LongRodEvaluator) -> None:
    for _ in range(3):
        evaluator.update(_valid(center_height_m=0.94, yaw_progress_deg=0.0, yaw_error_deg=90.0))
    assert evaluator.stage == LongRodStage.DUAL_GRASP
    evaluator.update(_valid(yaw_progress_deg=0.0, yaw_error_deg=90.0))
    assert evaluator.stage == LongRodStage.LIFTED
    evaluator.update(_valid(yaw_progress_deg=15.0, yaw_error_deg=75.0))
    assert evaluator.stage == LongRodStage.REORIENTING


def test_asset_is_exact_vendored_dexverse_file() -> None:
    assert validate_asset_contract() == []
    from handson.sim.isaaclab.bc007_scene import ROD_USD

    assert sha256_file(ROD_USD) == ROD_USD_SHA256


def test_start_and_target_geometry_are_horizontal_and_ninety_degrees_apart() -> None:
    start = np.asarray([*ROD_START_CENTER, *ROD_START_ORIENTATION_WXYZ])
    target = np.asarray([*ROD_START_CENTER, *target_orientation_wxyz(ROD_START_ORIENTATION_WXYZ)])
    start_ends = rod_endpoints(start, ROD_HALF_LENGTH_M)
    target_ends = rod_endpoints(target, ROD_HALF_LENGTH_M)
    assert abs(start_ends[0][2] - start_ends[1][2]) < 1.0e-9
    assert abs(target_ends[0][2] - target_ends[1][2]) < 1.0e-9
    yaw_delta = (rod_axis_yaw_deg(target) - rod_axis_yaw_deg(start) + 180.0) % 360.0 - 180.0
    assert math.isclose(yaw_delta, 90.0, abs_tol=1.0e-9)
    assert math.isclose(
        float(ROD_START_CENTER[2]),
        TABLE_TOP_Z + ROD_RADIUS_M,
        abs_tol=1.0e-12,
    )
    assert ROD_SLEEP_THRESHOLD == 0.0


def test_reset_yaw_jitter_preserves_horizontal_axis() -> None:
    for yaw in (-8.0, -3.0, 0.0, 4.5, 8.0):
        pose = np.asarray([*ROD_START_CENTER, *initial_orientation_wxyz(yaw)])
        ends = rod_endpoints(pose, ROD_HALF_LENGTH_M)
        assert abs(ends[0][2] - ends[1][2]) < 1.0e-9


def test_grasp_joint_frames_are_coincident_without_moving_either_body() -> None:
    rod = np.eye(4, dtype=np.float64)
    rod[:3, 3] = (-0.40, 0.0, 0.9135)
    palm = np.eye(4, dtype=np.float64)
    palm[:3, 3] = (-0.50, 0.23, 0.9535)
    body0_frame, body1_frame = joint_local_frame_pair(rod, palm)
    np.testing.assert_allclose(rod @ body0_frame, palm @ body1_frame, atol=1.0e-12)


def test_ordered_dual_hand_success() -> None:
    evaluator = LongRodEvaluator()
    _advance_to_reorienting(evaluator)
    result = evaluator.update(_valid())
    assert result["success_now"] is True
    assert evaluator.stage == LongRodStage.TARGET_STABLE


def test_terminal_near_misses_do_not_pass() -> None:
    cases = [
        {"left_role_owned": False},
        {"right_role_owned": False},
        {"center_height_m": 1.141},
        {"yaw_progress_deg": 74.9},
        {"yaw_error_deg": 8.1},
        {"endpoint_height_difference_m": 0.031},
        {"linear_speed_mps": 0.081},
        {"angular_speed_radps": 0.301},
        {"target_position_error_m": 0.041},
        {"hand_separation_m": 0.249},
        {"table_contact": True},
        {"strong_collision": True},
    ]
    for case in cases:
        evaluator = LongRodEvaluator()
        _advance_to_reorienting(evaluator)
        assert evaluator.update(_valid(**case))["success_now"] is False


def test_table_assist_role_loss_height_exit_and_collision_fail() -> None:
    scenarios = [
        ("table_contact", True, 3, "table_assisted_rotation"),
        ("left_role_owned", False, 5, "dual_end_role_lost"),
        ("center_height_m", 0.99, 7, "left_height_band"),
        ("strong_collision", True, 3, "sustained_strong_collision"),
    ]
    for field, value, frames, reason in scenarios:
        evaluator = LongRodEvaluator()
        _advance_to_reorienting(evaluator)
        for _ in range(frames):
            evaluator.update(_valid(**{field: value}))
        assert evaluator.failure_reason == reason


def test_drop_and_nonfinite_workspace_fail_directly() -> None:
    for frame, reason in [
        (_valid(dropped=True), "rod_dropped"),
        (_valid(finite_and_in_workspace=False), "nonfinite_or_workspace"),
    ]:
        evaluator = LongRodEvaluator()
        assert evaluator.update(frame)["failure_reason"] == reason


def test_required_contact_evidence_unavailable_fails_closed_immediately() -> None:
    evaluator = LongRodEvaluator()
    result = evaluator.update(
        _valid(required_contact_evidence_available=False)
    )
    assert result["success_now"] is False
    assert result["failed"] is True
    assert result["failure_reason"] == "required_contact_evidence_unavailable"


def test_fixed_shape_replay_assist_state_round_trips() -> None:
    encoded = encode_replay_assist_state(
        left_owner=True,
        right_owner=False,
        left_frame_armed=True,
        right_frame_armed=True,
        left_release_pending=True,
        right_release_pending=False,
    )
    assert encoded.shape == (3,)
    assert encoded.dtype == np.uint8
    assert int(encoded[2]) == BC007_REPLAY_ASSIST_VERSION
    assert decode_replay_assist_state(encoded) == {
        "left": {
            "owner": True,
            "frame_armed": True,
            "release_pending": True,
        },
        "right": {
            "owner": False,
            "frame_armed": True,
            "release_pending": False,
        },
    }


def test_replay_assist_state_rejects_unsafe_or_wrong_shape_values() -> None:
    with pytest.raises(ValueError, match="requires an active owner"):
        encode_replay_assist_state(
            left_owner=False,
            right_owner=False,
            left_frame_armed=False,
            right_frame_armed=False,
            left_release_pending=True,
            right_release_pending=False,
        )
    with pytest.raises(ValueError, match=r"uint8 shape \(3,\)"):
        decode_replay_assist_state(np.zeros(3, dtype=np.int64))
    invalid_version = np.asarray([0, 0, 255], dtype=np.uint8)
    with pytest.raises(ValueError, match="version is unsupported"):
        decode_replay_assist_state(invalid_version)


def test_runtime_wires_assist_clear_into_both_reset_paths() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    runtime_path = (
        repository_root / "handson" / "handson" / "sim" / "isaaclab" / "bc007_env.py"
    )
    module = ast.parse(runtime_path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in ("_sample_reset_state", "restore_recorded_initial_state"):
        calls = [
            node
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_clear_bc007_assist_state"
        ]
        assert calls, f"{name} must clear BC007 fixed-joint assist state"
        assert any(
            keyword.arg == "disable_joints"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for call in calls
            for keyword in call.keywords
        ), f"{name} must physically disable both fixed joints"
    runtime_source = runtime_path.read_text(encoding="utf-8")
    assert "def get_physical_replay_task_controller_state" in runtime_source
    assert "def set_physical_replay_task_controller_state" in runtime_source
    assert '"replay_restore_supported": True' in runtime_source
    assert "pre_action_latched_sampled_with_next_state" in runtime_source
    assert "initial_robot_state_report(self)" in runtime_source


def test_live_smoke_defaults_to_ten_seconds_and_actual_robot_readback() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source = (
        repository_root
        / "taskdev"
        / "tasks"
        / "bench320_bc007_bimanual_long_rod_yaw_reorientation"
        / "run_live_smoke.py"
    ).read_text(encoding="utf-8")
    assert "DEFAULT_CONTROL_STEPS = 300" in source
    assert "required_minimum_physics_steps\": 1200" in source
    assert "initial_robot_state_report(core)" in source
    assert "RobotStabilityAudit.start(core)" in source
    assert "_neutral_hold_action(env, core, initial_robot_state)" in source
    assert '"hand_fields": "absolute_actual_reset_joint_positions"' in source
    assert "core._bc007_per_physics_robot_audit = robot_audit" in source
    assert "sampled_physics_steps == physics_steps" in source
    assert "robot_snapshot(core)" in source


class _ContactView:
    def __init__(self, force_n: float, *, readable: bool = True):
        self.force_n = float(force_n)
        self.readable = bool(readable)

    def get_contact_force_matrix(self, _dt: float) -> np.ndarray:
        if not self.readable:
            raise RuntimeError("contact tensor unavailable")
        return np.asarray([[[self.force_n, 0.0, 0.0]]], dtype=np.float64)


def _stable_origin_environment() -> BC007IsaacLabEnv:
    environment = object.__new__(BC007IsaacLabEnv)
    environment._sim_dt = 1.0 / 120.0
    environment._bc007_contact_errors = set()
    environment._bc007_hand_contact_views = {
        "left": _ContactView(0.0),
        "right": _ContactView(0.0),
    }
    environment._bc007_table_contact_view = _ContactView(0.02)
    environment._physics_body_pose = lambda _name: np.asarray(
        [*ROD_START_CENTER, *ROD_START_ORIENTATION_WXYZ], dtype=np.float64
    )
    environment._physics_body_velocity = lambda _name: np.zeros(
        6, dtype=np.float64
    )
    environment._rod_awake_readback = lambda: (True, True, "")
    return environment


def test_stable_origin_readiness_requires_awake_support_and_contacts() -> None:
    environment = _stable_origin_environment()
    ready = environment.stable_recording_origin_task_readiness()
    assert ready["ready"] is True
    assert ready["required_contact_evidence_available"] is True
    assert ready["table_support_active"] is True
    assert ready["rod_awake"] is True

    environment._rod_awake_readback = lambda: (False, True, "")
    sleeping = environment.stable_recording_origin_task_readiness()
    assert sleeping["ready"] is False
    assert "rod_sleeping" in sleeping["failure_reasons"]

    environment._rod_awake_readback = lambda: (True, True, "")
    environment._bc007_table_contact_view = _ContactView(0.0)
    unsupported = environment.stable_recording_origin_task_readiness()
    assert unsupported["ready"] is False
    assert "table_support_force_below_minimum" in unsupported["failure_reasons"]

    environment._bc007_table_contact_view = _ContactView(0.02, readable=False)
    unreadable = environment.stable_recording_origin_task_readiness()
    assert unreadable["ready"] is False
    assert unreadable["required_contact_evidence_available"] is False
    assert "required_contact_evidence_unavailable" in unreadable["failure_reasons"]


def test_stable_origin_acceptance_rechecks_actual_initial_state() -> None:
    environment = object.__new__(BC007IsaacLabEnv)
    environment.env_step = 0
    environment._live_arm_visibility_guard_uses_stable_origin = False
    environment.stable_recording_origin_task_readiness = lambda: {
        "ready": True,
        "rod_awake": True,
    }
    calls = []

    def actual_state():
        calls.append("actual")
        return {"actual_robot_state": {"left": {}, "right": {}}}

    environment._validate_task_initial_state_before_ready = actual_state
    accepted = environment.accept_stable_recording_origin(
        {
            "task_readiness_required": True,
            "final_task_readiness": {"ready": True},
        }
    )
    assert calls == ["actual"]
    assert accepted["accepted"] is True
    assert accepted["initial_state"]["actual_robot_state"] == {
        "left": {},
        "right": {},
    }


def test_live_robot_audit_samples_once_per_physics_substep() -> None:
    environment = object.__new__(BC007IsaacLabEnv)
    environment.task_name = "bench320_bc007_bimanual_long_rod_yaw_reorientation"

    class Audit:
        samples = 0

        def sample(self, observed_environment) -> None:
            assert observed_environment is environment
            self.samples += 1

    audit = Audit()
    environment._bc007_per_physics_robot_audit = audit
    for _ in range(4):
        environment._after_task_physics_substep()
    assert audit.samples == 4
