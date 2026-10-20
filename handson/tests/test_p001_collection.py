from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from gymnasium import spaces

from handson.sim.isaaclab.p001_contract import (
    ACTION_DIM,
    ASSET_FILE_SHA256,
    ASSET_RELATIVE_PATH,
    CONTACT_POLICY_VERSION,
    CONTROL_HZ,
    EXPLOSION_LINEAR_SPEED_MPS,
    FAN_ROOT_PATH,
    FORBIDDEN_CONTACT_FORCE_N,
    INITIAL_SPAN_RANGE_DEG,
    LEFT_RIGHT_BALANCE_MAX_DEG,
    MASS_SCALE_RANGE,
    NOMINAL_ROOT_XY_M,
    NOMINAL_ROOT_YAW_DEG,
    PLANARITY_MAX_M,
    P001PhysicalReadback,
    P001ScriptedPolicy,
    P001SuccessTracker,
    P001_TASK_STATE_DIM,
    P001_TASK_STATE_SCHEMA_VERSION,
    REQUIRED_BILATERAL_EXPANSION_FRACTION,
    REQUIRED_CONTACT_FORCE_N,
    ROOT_LIFT_MAX_M,
    ROOT_TRANSLATION_MAX_M,
    ROOT_XY_HALF_RANGE_M,
    ROOT_YAW_HALF_RANGE_DEG,
    SOURCE_JOINT_LIMIT_DEG,
    STABLE_JOINT_SPEED_RADPS,
    STABLE_LINEAR_SPEED_MPS,
    TABLE_TOP_Z_M,
    SUCCESS_HOLD_STEPS,
    TARGET_OPEN_CHOICES_DEG,
    TARGET_TOLERANCE_DEG,
    TASK_ID,
    TASK_NAME,
    TIME_LIMIT_SECONDS,
    TIMEOUT_STEPS,
    bilateral_contact_credit_allowed,
    evaluate_physical_readback,
    p001_action_to_side_major_storage,
    sample_reset_state,
    span_components_deg,
)
from handson.sim.isaaclab.p001_collection_env import (
    HandsOnP001IsaacLabEnv,
    _FAN_BODY_PATHS,
    _P001_PHYSICAL_STATE_REPLAY_ATOL,
    _P001_PHYSICAL_STATE_REPLAY_FIELDS,
    _authored_rigid_state_from_transform,
)
from handson.tasks.policy_wrappers import _dual_arm_hand_dim
from handson.sim.isaaclab.recording_origin import (
    STATIC_POSE_FALLBACK_TASKS,
    _contact_penetration_summary,
    _full_coordinate_exact_mismatches,
    stable_recording_origin_kwargs_for_task,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _successful_readback(
    reset_state,
    *,
    target_deg: float | None = None,
    **overrides,
) -> P001PhysicalReadback:
    target = float(reset_state.target_open_deg if target_deg is None else target_deg)
    per_joint_rad = math.radians(target / 18.0)
    values = {
        "active_joint_angles_rad": (per_joint_rad,) * 9 + (-per_joint_rad,) * 9,
        "active_joint_velocities_radps": (0.0,) * 18,
        "root_position_m": (reset_state.root_x_m, reset_state.root_y_m, 0.913),
        "reset_root_position_m": (reset_state.root_x_m, reset_state.root_y_m, 0.913),
        "max_body_linear_speed_mps": 0.0,
        "planarity_error_m": 0.005,
        "left_end_contact_force_n": REQUIRED_CONTACT_FORCE_N + 0.10,
        "right_end_contact_force_n": REQUIRED_CONTACT_FORCE_N + 0.10,
        "left_wrong_end_contact_force_n": 0.0,
        "right_wrong_end_contact_force_n": 0.0,
        "robot_table_contact_force_n": 0.0,
        "positive_span_increase_deg": target - reset_state.initial_span_deg,
        "bilateral_positive_span_increase_deg": target - reset_state.initial_span_deg,
        "all_bodies_inside_safe_zone": True,
        "min_body_height_m": 0.890,
    }
    values.update(overrides)
    return P001PhysicalReadback(**values)


def _initial_readback(reset_state) -> P001PhysicalReadback:
    return P001PhysicalReadback(
        active_joint_angles_rad=tuple(
            math.radians(value) for value in reset_state.active_joint_positions_deg
        ),
        active_joint_velocities_radps=(0.0,) * 18,
        root_position_m=(reset_state.root_x_m, reset_state.root_y_m, 0.913),
        reset_root_position_m=(reset_state.root_x_m, reset_state.root_y_m, 0.913),
        max_body_linear_speed_mps=0.0,
        planarity_error_m=0.005,
        left_end_contact_force_n=0.0,
        right_end_contact_force_n=0.0,
        left_wrong_end_contact_force_n=0.0,
        right_wrong_end_contact_force_n=0.0,
        robot_table_contact_force_n=0.0,
        positive_span_increase_deg=0.0,
        bilateral_positive_span_increase_deg=0.0,
        all_bodies_inside_safe_zone=True,
        min_body_height_m=TABLE_TOP_Z_M,
    )


def test_identity_registration_and_frozen_targets() -> None:
    from handson.runtime.collection_support import ISAACSIM_COLLECTION_TASKS
    from handson.tasks.mappings import CONFIG_MAPPING

    assert TASK_ID == "P0-01"
    assert TASK_NAME == "t48_unfold_fan"
    assert TARGET_OPEN_CHOICES_DEG == (60.0, 90.0, 120.0)
    assert TARGET_TOLERANCE_DEG == 8.0
    assert TASK_NAME in CONFIG_MAPPING
    assert TASK_NAME in ISAACSIM_COLLECTION_TASKS


def test_exact_project_asset_bundle_hashes() -> None:
    asset = REPOSITORY_ROOT / "handson/handson" / ASSET_RELATIVE_PATH
    bundle_root = asset.parent
    assert asset.is_file()
    for relative_path, expected_sha in ASSET_FILE_SHA256.items():
        path = bundle_root / relative_path
        assert path.is_file(), relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha


def test_recording_origin_uses_strict_every_frame_static_pose_fallback() -> None:
    assert TASK_NAME in STATIC_POSE_FALLBACK_TASKS
    contract = stable_recording_origin_kwargs_for_task(TASK_NAME)
    assert contract["stability_metric"] == "velocity_or_static_pose"
    assert contract["stability_sample_seconds"] == 0.0
    assert contract["pose_translation_limit"] == 1.0e-6
    assert contract["pose_rotation_limit"] == 1.0e-5
    assert contract["reset_rotation_drift_limit"] == pytest.approx(
        math.radians(3.0)
    )
    assert contract["stabilization_velocity_clamp"] is False
    assert contract["post_settle_velocity_zeroing"] is True
    assert contract["require_valid_contact_report"] is True
    assert contract["contact_penetration_limit"] == 0.0005


def test_authored_rigid_state_uses_trace_wxyz_layout_and_zero_velocity() -> None:
    state = _authored_rigid_state_from_transform(
        [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]]
    )
    assert np.array_equal(
        state,
        np.asarray(
            [1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
    )


def test_authored_snapshot_accepts_articulation_only_fan_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "handson.sim.isaaclab.abc_collection_env."
        "HandsOnABCIsaacLabEnv._capture_recording_origin_authored_reset_snapshot",
        lambda self, *, require_authored_task_writes=True: None,
    )
    environment = object.__new__(HandsOnP001IsaacLabEnv)
    environment._physx_ground_truth_enabled = True
    environment._recording_origin_authored_reset_snapshot = {}
    environment._physx_ground_truth_trace = SimpleNamespace(
        rigid_bodies=[],
        articulations=[SimpleNamespace(name=FAN_ROOT_PATH)],
    )

    environment._capture_recording_origin_authored_reset_snapshot()


def test_authored_snapshot_rejects_missing_fan_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "handson.sim.isaaclab.abc_collection_env."
        "HandsOnABCIsaacLabEnv._capture_recording_origin_authored_reset_snapshot",
        lambda self, *, require_authored_task_writes=True: None,
    )
    environment = object.__new__(HandsOnP001IsaacLabEnv)
    environment._physx_ground_truth_enabled = True
    environment._recording_origin_authored_reset_snapshot = {}
    environment._physx_ground_truth_trace = SimpleNamespace(
        rigid_bodies=[],
        articulations=[],
    )

    with pytest.raises(RuntimeError, match="missing the fan articulation"):
        environment._capture_recording_origin_authored_reset_snapshot()


def test_core_environment_registers_exact_fan_contact_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Prim:
        def IsValid(self) -> bool:
            return True

        def IsActive(self) -> bool:
            return True

        def HasAPI(self, _api: object) -> bool:
            return True

    class _Stage:
        def GetPrimAtPath(self, path: str) -> _Prim:
            assert path in _FAN_BODY_PATHS
            return _Prim()

    configured_calls: list[tuple[object, tuple[str, ...]]] = []

    def _configure(stage: object, paths: list[str]) -> tuple[str, ...]:
        configured_calls.append((stage, tuple(paths)))
        return tuple(paths)

    monkeypatch.setattr(
        "handson.sim.isaaclab.p001_collection_env."
        "configure_physx_contact_reports_for_recording",
        _configure,
    )
    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(UsdPhysics=SimpleNamespace(RigidBodyAPI=object())),
    )
    environment = object.__new__(HandsOnP001IsaacLabEnv)
    environment._stage = _Stage()

    environment._configure_ground_truth_contact_reports(Path("unused.xml"))

    assert configured_calls == [(environment._stage, _FAN_BODY_PATHS)]
    assert environment._physx_contact_sensor_paths_preconfigured == _FAN_BODY_PATHS
    assert environment._physx_contact_report_paths_preconfigured == _FAN_BODY_PATHS


def test_p001_replay_tolerance_is_local_to_articulated_fan_pose_fields() -> None:
    environment = object.__new__(HandsOnP001IsaacLabEnv)

    overrides = environment._physx_ground_truth_replay_atol_overrides()

    assert set(overrides) == _P001_PHYSICAL_STATE_REPLAY_FIELDS
    assert set(overrides.values()) == {_P001_PHYSICAL_STATE_REPLAY_ATOL}
    assert "physx_robot_joint_position" not in overrides
    assert "physx_contact_signature_pair_metrics" not in overrides


def _synthetic_contact_snapshot(*, metric_delta: float) -> tuple[dict, dict]:
    point = np.asarray(
        [[0.1, 0.2, 0.3, 0.3, 0.4, 0.0, 0.2, 0.7, 0.0, -0.001]],
        dtype=np.float32,
    )
    normal = np.asarray(point[0, 3:6], dtype=np.float64)
    impulse = np.asarray(point[0, 6:9], dtype=np.float64)
    penetration = max(0.0, -float(point[0, 9]))
    normal_impulse = abs(float(np.dot(impulse, normal / np.linalg.norm(normal))))
    recorded_metric = np.nextafter(normal_impulse, math.inf) + metric_delta
    snapshot = {
        "physx_contact_report_valid": np.asarray([1], dtype=np.uint8),
        "physx_contact_capture_source": np.asarray([2], dtype=np.uint8),
        "physx_contact_header_count": np.asarray([1], dtype=np.uint32),
        "physx_contact_point_count": np.asarray([1], dtype=np.uint32),
        "physx_friction_anchor_count": np.asarray([0], dtype=np.uint32),
        "physx_contact_signature_valid": np.asarray([1], dtype=np.uint8),
        "physx_contact_signature_pair_count": np.asarray([1], dtype=np.uint32),
        "physx_contact_header_ids": np.asarray(
            [[1, 2, 1, 2, 0]], dtype=np.uint64
        ),
        "physx_contact_header_meta": np.asarray(
            [[0, 0, 1, 0, 0, 0, 0]], dtype=np.uint32
        ),
        "physx_contact_point_data": point,
        "physx_contact_point_face_indices": np.zeros((1, 2), dtype=np.uint32),
        "physx_contact_point_material_ids": np.zeros((1, 2), dtype=np.uint64),
        "physx_friction_anchor_data": np.zeros((0, 6), dtype=np.float32),
        "physx_contact_signature_pair_path_indices": np.asarray(
            [[1, 1, 2, 2]], dtype=np.uint32
        ),
        "physx_contact_signature_pair_counts": np.asarray(
            [[1, 0]], dtype=np.uint32
        ),
        "physx_contact_signature_pair_metrics": np.asarray(
            [[penetration, recorded_metric, 0.0]], dtype=np.float64
        ),
    }
    schema = {
        "physx": {
            "contact_report": {
                "path_table": [
                    {"id": 1, "path": "/World/A"},
                    {"id": 2, "path": "/World/B"},
                ]
            }
        }
    }
    return snapshot, schema


def test_contact_signature_accepts_one_ulp_metric_rounding() -> None:
    snapshot, schema = _synthetic_contact_snapshot(metric_delta=0.0)
    summary = _contact_penetration_summary(snapshot, schema)
    assert summary["valid"] is True
    assert summary["error"] is None


def test_contact_signature_rejects_material_metric_change() -> None:
    snapshot, schema = _synthetic_contact_snapshot(metric_delta=1.0e-8)
    summary = _contact_penetration_summary(snapshot, schema)
    assert summary["valid"] is False
    assert str(summary["error"]).startswith(
        "canonical_contact_metrics_mismatch:"
    )


def _coordinate_snapshot(x_position: np.float32) -> dict[str, np.ndarray]:
    # Ground-truth storage promotes public PhysX float32 values to float64.
    rigid_state = np.zeros((1, 13), dtype=np.float64)
    rigid_state[0, 0] = x_position
    return {
        "physx_task_rigid_body_state": rigid_state,
        "physx_task_rigid_body_valid": np.asarray([1], dtype=np.uint8),
        "physx_task_articulation_root_state": np.zeros((0, 13), dtype=np.float32),
        "physx_task_articulation_root_valid": np.zeros(0, dtype=np.uint8),
        "physx_robot_root_state": np.zeros((0, 13), dtype=np.float32),
        "physx_robot_root_valid": np.zeros(0, dtype=np.uint8),
        "physx_tcp_state": np.zeros((0, 13), dtype=np.float32),
        "physx_tcp_valid": np.zeros(0, dtype=np.uint8),
        "physx_task_articulation_dof_position": np.zeros(
            (0, 0), dtype=np.float32
        ),
        "physx_task_articulation_dof_valid": np.zeros(
            (0, 0), dtype=np.uint8
        ),
        "physx_robot_joint_position": np.zeros((0, 0), dtype=np.float32),
        "physx_robot_joint_valid": np.zeros((0, 0), dtype=np.uint8),
        "physx_deformable_position": np.zeros((0, 3), dtype=np.float32),
        "physx_deformable_valid": np.zeros(0, dtype=np.uint8),
    }


def test_no_step_coordinate_gate_accepts_one_native_float_ulp() -> None:
    before_x = np.float32(0.25)
    after_x = np.nextafter(before_x, np.float32(math.inf), dtype=np.float32)
    assert _full_coordinate_exact_mismatches(
        _coordinate_snapshot(before_x),
        _coordinate_snapshot(after_x),
    ) == []


def test_no_step_coordinate_gate_accepts_near_zero_unit_scale_rounding() -> None:
    unit_rounding = np.float32(np.finfo(np.float32).eps / 4.0)
    assert _full_coordinate_exact_mismatches(
        _coordinate_snapshot(np.float32(0.0)),
        _coordinate_snapshot(unit_rounding),
    ) == []


def test_no_step_coordinate_gate_rejects_material_pose_change() -> None:
    assert _full_coordinate_exact_mismatches(
        _coordinate_snapshot(np.float32(0.25)),
        _coordinate_snapshot(np.float32(0.251)),
    ) == ["physx_task_rigid_body_state"]


def test_seeded_reset_is_repeatable_and_covers_all_targets() -> None:
    samples = [sample_reset_state(seed) for seed in range(256)]
    assert samples == [sample_reset_state(seed) for seed in range(256)]
    assert {sample.target_open_deg for sample in samples} == set(
        TARGET_OPEN_CHOICES_DEG
    )
    for sample in samples:
        assert abs(sample.root_x_m - NOMINAL_ROOT_XY_M[0]) <= ROOT_XY_HALF_RANGE_M
        assert abs(sample.root_y_m - NOMINAL_ROOT_XY_M[1]) <= ROOT_XY_HALF_RANGE_M
        assert abs(sample.root_yaw_deg - NOMINAL_ROOT_YAW_DEG) <= ROOT_YAW_HALF_RANGE_DEG
        assert INITIAL_SPAN_RANGE_DEG[0] <= sample.initial_span_deg <= INITIAL_SPAN_RANGE_DEG[1]
        assert sample.target_open_deg - sample.initial_span_deg > TARGET_TOLERANCE_DEG
        assert MASS_SCALE_RANGE[0] <= sample.mass_scale <= MASS_SCALE_RANGE[1]
        assert len(sample.active_joint_positions_deg) == 18
        assert all(value > 0.0 for value in sample.active_joint_positions_deg[:9])
        assert all(value < 0.0 for value in sample.active_joint_positions_deg[9:])


@pytest.mark.parametrize("target_deg", TARGET_OPEN_CHOICES_DEG)
def test_each_target_has_a_physical_positive_and_half_second_hold(target_deg: float) -> None:
    reset_state = sample_reset_state(101, target_open_deg=target_deg)
    readback = _successful_readback(reset_state)
    metrics = evaluate_physical_readback(readback, reset_state)
    assert metrics["span_deg"] == pytest.approx(target_deg)
    assert metrics["target_ok"] is True
    assert metrics["success_now"] is True

    tracker = P001SuccessTracker(reset_state)
    for _ in range(SUCCESS_HOLD_STEPS - 1):
        assert tracker.update(readback) == (False, False)
    assert tracker.update(readback) == (True, False)


def test_initial_state_only_passes_baseline_guard_conditions() -> None:
    reset_state = sample_reset_state(101, target_open_deg=90.0)
    metrics = evaluate_physical_readback(_initial_readback(reset_state), reset_state)

    assert metrics["span_deg"] == pytest.approx(reset_state.initial_span_deg)
    assert metrics["target_ok"] is False
    assert metrics["balance_ok"] is True
    assert metrics["left_end_contact"] is False
    assert metrics["right_end_contact"] is False
    assert metrics["net_increase_ok"] is False
    assert metrics["bilateral_expansion_ok"] is False
    assert metrics["root_stable"] is True
    assert metrics["planarity_ok"] is True
    assert metrics["velocity_stable"] is True
    assert metrics["wrong_end_clear"] is True
    assert metrics["robot_table_clear"] is True
    assert metrics["safe_zone"] is True
    assert metrics["success_now"] is False


def test_initial_overlay_is_ascii_descriptive_and_not_successful() -> None:
    reset_state = sample_reset_state(101, target_open_deg=90.0)
    environment = object.__new__(HandsOnP001IsaacLabEnv)
    environment.task_name = TASK_NAME
    environment._p001_reset = reset_state
    environment._p001_last_metrics = evaluate_physical_readback(
        _initial_readback(reset_state), reset_state
    )
    environment._success_counter = 0

    status = environment._task_status_text()
    rows = environment._success_condition_rows()
    display_labels = [
        environment._success_condition_display_label(row) for row in rows
    ]

    assert status == (
        f"Fan {reset_state.initial_span_deg:.1f}/90 deg"
        "  |  L/R diff 0.0 deg  |  Hold 0/15"
    )
    assert status.isascii()
    assert "?" not in status
    assert [row["key"] for row in rows] == [
        "p001_target",
        "p001_balance",
        "p001_contacts",
        "p001_expansion",
        "p001_support",
        "p001_velocity",
        "p001_forbidden",
        "success_hold",
    ]
    assert [row["ok"] for row in rows] == [
        False,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
    ]
    assert all(label.isascii() and "?" not in label for label in display_labels)
    assert all(not label.startswith("p001 ") for label in display_labels)
    assert "30%" in rows[3]["label"]
    assert "30%" in display_labels[3]
    assert "table" not in display_labels[6].lower()


def test_span_is_sum_of_18_active_joint_openings() -> None:
    values = np.deg2rad(np.asarray([4.0] * 9 + [-6.0] * 9))
    total, left, right = span_components_deg(values)
    assert total == pytest.approx(90.0)
    assert left == pytest.approx(36.0)
    assert right == pytest.approx(54.0)


@pytest.mark.parametrize(
    ("kind", "expected_false_metric", "expected_failure"),
    [
        ("missing_right_contact", "right_end_contact", False),
        ("missing_left_contact", "left_end_contact", False),
        ("left_crossed_ownership", "wrong_end_clear", False),
        ("right_crossed_ownership", "wrong_end_clear", False),
        ("wrong_target", "target_ok", False),
        ("imbalance", "balance_ok", False),
        ("insufficient_net_expansion", "net_increase_ok", False),
        ("insufficient_bilateral_fraction", "bilateral_expansion_ok", False),
        ("root_translation", "root_stable", False),
        ("root_lift", "root_stable", False),
        ("nonplanar", "planarity_ok", False),
        ("high_joint_speed", "velocity_stable", False),
        ("high_body_speed", "velocity_stable", False),
        ("joint_wrong_direction", "joint_direction_ok", True),
        ("joint_limit", "joint_limits_ok", True),
        ("drop", "dropped", True),
        ("table_edge", "safe_zone", True),
        ("explosion", "exploded", True),
        ("nan", "finite", True),
    ],
)
def test_required_negative_semantics_fail_closed(
    kind: str,
    expected_false_metric: str,
    expected_failure: bool,
) -> None:
    reset_state = sample_reset_state(2026, target_open_deg=90.0)
    readback = _successful_readback(reset_state)
    values = dict(readback.__dict__)
    if kind == "missing_right_contact":
        values["right_end_contact_force_n"] = 0.0
    elif kind == "missing_left_contact":
        values["left_end_contact_force_n"] = 0.0
    elif kind == "left_crossed_ownership":
        values["left_wrong_end_contact_force_n"] = FORBIDDEN_CONTACT_FORCE_N + 0.01
    elif kind == "right_crossed_ownership":
        values["right_wrong_end_contact_force_n"] = FORBIDDEN_CONTACT_FORCE_N + 0.01
    elif kind == "wrong_target":
        per_joint = math.radians(60.0 / 18.0)
        values["active_joint_angles_rad"] = (per_joint,) * 9 + (-per_joint,) * 9
    elif kind == "imbalance":
        left = math.radians(25.0 / 9.0)
        right = math.radians(65.0 / 9.0)
        values["active_joint_angles_rad"] = (left,) * 9 + (-right,) * 9
    elif kind == "insufficient_net_expansion":
        values["positive_span_increase_deg"] = 5.0
        values["bilateral_positive_span_increase_deg"] = 5.0
        values["active_joint_angles_rad"] = tuple(
            math.radians(value) for value in reset_state.active_joint_positions_deg
        )
    elif kind == "insufficient_bilateral_fraction":
        progress = 90.0 - reset_state.initial_span_deg
        values["positive_span_increase_deg"] = progress
        values["bilateral_positive_span_increase_deg"] = (
            progress * (REQUIRED_BILATERAL_EXPANSION_FRACTION - 0.05)
        )
    elif kind == "root_translation":
        root = list(values["root_position_m"])
        root[0] += ROOT_TRANSLATION_MAX_M + 0.001
        values["root_position_m"] = tuple(root)
    elif kind == "root_lift":
        root = list(values["root_position_m"])
        root[2] += ROOT_LIFT_MAX_M + 0.001
        values["root_position_m"] = tuple(root)
    elif kind == "nonplanar":
        values["planarity_error_m"] = PLANARITY_MAX_M + 0.001
    elif kind == "high_joint_speed":
        values["active_joint_velocities_radps"] = (
            STABLE_JOINT_SPEED_RADPS + 0.05,
        ) * 18
    elif kind == "high_body_speed":
        values["max_body_linear_speed_mps"] = STABLE_LINEAR_SPEED_MPS + 0.01
    elif kind == "joint_wrong_direction":
        angles = list(values["active_joint_angles_rad"])
        angles[0] = math.radians(-1.0)
        values["active_joint_angles_rad"] = tuple(angles)
    elif kind == "joint_limit":
        angles = list(values["active_joint_angles_rad"])
        angles[0] = math.radians(SOURCE_JOINT_LIMIT_DEG + 0.2)
        values["active_joint_angles_rad"] = tuple(angles)
    elif kind == "drop":
        values["min_body_height_m"] = TABLE_TOP_Z_M - 0.031
    elif kind == "table_edge":
        values["all_bodies_inside_safe_zone"] = False
    elif kind == "explosion":
        values["max_body_linear_speed_mps"] = EXPLOSION_LINEAR_SPEED_MPS + 0.1
    elif kind == "nan":
        values["active_joint_angles_rad"] = (math.nan,) + values["active_joint_angles_rad"][1:]
    metrics = evaluate_physical_readback(P001PhysicalReadback(**values), reset_state)
    assert metrics["success_now"] is False
    if expected_false_metric in ("dropped", "exploded"):
        assert metrics[expected_false_metric] is True
    else:
        assert metrics[expected_false_metric] is False
    assert metrics["failure"] is expected_failure


@pytest.mark.parametrize(
    "table_force_n",
    [
        FORBIDDEN_CONTACT_FORCE_N + 0.001,
        20.001,
        100.0,
        math.nan,
    ],
)
def test_robot_table_force_is_diagnostic_only(table_force_n: float) -> None:
    reset_state = sample_reset_state(2026, target_open_deg=90.0)
    readback = _successful_readback(
        reset_state,
        robot_table_contact_force_n=table_force_n,
    )

    metrics = evaluate_physical_readback(readback, reset_state)

    assert metrics["robot_table_clear"] is False
    assert metrics["forbidden_contact"] is False
    assert metrics["success_now"] is True
    assert metrics["failure"] is False


def test_bilateral_expansion_fraction_has_an_inclusive_30_percent_boundary() -> None:
    reset_state = sample_reset_state(2026, target_open_deg=90.0)
    at_boundary = _successful_readback(
        reset_state,
        positive_span_increase_deg=100.0,
        bilateral_positive_span_increase_deg=30.0,
    )
    just_below = replace(
        at_boundary,
        bilateral_positive_span_increase_deg=30.0 - 1.0e-6,
    )

    boundary_metrics = evaluate_physical_readback(at_boundary, reset_state)
    below_metrics = evaluate_physical_readback(just_below, reset_state)

    assert boundary_metrics["bilateral_expansion_fraction"] == pytest.approx(0.30)
    assert boundary_metrics["bilateral_expansion_ok"] is True
    assert boundary_metrics["success_now"] is True
    assert below_metrics["bilateral_expansion_fraction"] < 0.30
    assert below_metrics["bilateral_expansion_ok"] is False
    assert below_metrics["success_now"] is False


def test_imbalance_limit_is_strictly_enforced() -> None:
    reset_state = sample_reset_state(7, target_open_deg=90.0)
    left_total = 45.0 - LEFT_RIGHT_BALANCE_MAX_DEG / 2.0 - 0.1
    right_total = 90.0 - left_total
    left = math.radians(left_total / 9.0)
    right = math.radians(right_total / 9.0)
    readback = _successful_readback(
        reset_state,
        active_joint_angles_rad=(left,) * 9 + (-right,) * 9,
    )
    metrics = evaluate_physical_readback(readback, reset_state)
    assert metrics["target_ok"] is True
    assert metrics["balance_error_deg"] > LEFT_RIGHT_BALANCE_MAX_DEG
    assert metrics["success_now"] is False


@pytest.mark.parametrize(
    "soft_contact",
    [
        {"left_wrong_end_contact_force_n": 100.0},
        {"right_wrong_end_contact_force_n": 100.0},
    ],
)
def test_crossed_contact_resets_hold_and_recovers_in_same_episode(
    soft_contact: dict[str, float],
) -> None:
    reset_state = sample_reset_state(42, target_open_deg=120.0)
    stable = _successful_readback(reset_state)
    contact = replace(stable, **soft_contact)
    tracker = P001SuccessTracker(reset_state)
    for _ in range(SUCCESS_HOLD_STEPS - 1):
        assert tracker.update(stable) == (False, False)
    assert tracker.update(contact) == (False, False)
    assert tracker.consecutive_steps == 0
    for _ in range(SUCCESS_HOLD_STEPS - 1):
        assert tracker.update(stable) == (False, False)
    assert tracker.update(stable) == (True, False)


def test_robot_table_force_does_not_reset_success_hold() -> None:
    reset_state = sample_reset_state(42, target_open_deg=120.0)
    table_contact = _successful_readback(
        reset_state,
        robot_table_contact_force_n=100.0,
    )
    tracker = P001SuccessTracker(reset_state)

    for _ in range(SUCCESS_HOLD_STEPS - 1):
        assert tracker.update(table_contact) == (False, False)
    assert tracker.update(table_contact) == (True, False)


def test_irrecoverable_failure_still_latches() -> None:
    reset_state = sample_reset_state(42, target_open_deg=120.0)
    stable = _successful_readback(reset_state)
    exploded = replace(
        stable,
        max_body_linear_speed_mps=EXPLOSION_LINEAR_SPEED_MPS + 0.1,
    )
    tracker = P001SuccessTracker(reset_state)

    assert tracker.update(exploded) == (False, True)
    assert tracker.update(stable) == (False, True)


def test_crossed_contacts_block_success_and_bilateral_progress_credit() -> None:
    clear = {
        "left_end_force_n": REQUIRED_CONTACT_FORCE_N,
        "right_end_force_n": REQUIRED_CONTACT_FORCE_N,
        "left_wrong_end_force_n": 0.0,
        "right_wrong_end_force_n": 0.0,
    }
    assert bilateral_contact_credit_allowed(**clear) is True
    for field in (
        "left_wrong_end_force_n",
        "right_wrong_end_force_n",
    ):
        blocked = dict(clear)
        blocked[field] = FORBIDDEN_CONTACT_FORCE_N + 0.001
        assert bilateral_contact_credit_allowed(**blocked) is False

    reset_state = sample_reset_state(7, target_open_deg=90.0)
    crossed = _successful_readback(
        reset_state,
        left_wrong_end_contact_force_n=100.0,
    )
    metrics = evaluate_physical_readback(crossed, reset_state)
    assert metrics["forbidden_contact"] is True
    assert metrics["success_now"] is False
    assert metrics["failure"] is False


def _stopgap_substep_environment(
    *, robot_table_force_n: float,
) -> HandsOnP001IsaacLabEnv:
    reset_state = sample_reset_state(11, target_open_deg=90.0)
    environment = object.__new__(HandsOnP001IsaacLabEnv)
    environment.task_name = TASK_NAME
    environment._p001_articulation_view = object()
    environment._p001_substep_count = 0
    environment._p001_previous_span_deg = reset_state.initial_span_deg
    environment._p001_positive_span_increase_deg = 0.0
    environment._p001_bilateral_positive_span_increase_deg = 0.0
    environment._p001_bilateral_frame_start_deg = 0.0
    environment._p001_contact_seen_this_control_frame = False
    environment._p001_robot_table_hard_contact_substeps = 0
    environment._p001_robot_table_hard_failure_latched = False
    environment._p001_failure_latched = False
    environment._p001_failure_reported = True
    environment._p001_failure_reason = ""
    environment._trajectory_action_phase_active = True
    environment._trajectory_action_phase_substep_index = 0
    positions = np.asarray(
        [math.radians(value) for value in reset_state.active_joint_positions_deg],
        dtype=np.float64,
    )
    environment._p001_test_positions = positions
    environment._active_joint_state = lambda: (
        environment._p001_test_positions,
        np.zeros(18),
    )
    environment._p001_test_force_by_key = {
        "left_end": REQUIRED_CONTACT_FORCE_N,
        "right_end": REQUIRED_CONTACT_FORCE_N,
        "left_wrong_end": 0.0,
        "right_wrong_end": 0.0,
        "robot_table": robot_table_force_n,
    }
    environment._bidirectional_contact_force = (
        lambda key: environment._p001_test_force_by_key[key]
    )
    environment._contact_force = (
        lambda key: environment._p001_test_force_by_key[key]
    )
    readback = _initial_readback(reset_state)
    metrics = evaluate_physical_readback(readback, reset_state)
    environment._p001_reset = reset_state
    environment._p001_last_readback = readback
    environment._p001_last_metrics = metrics
    environment._success_counter = 0
    environment._p001_contact_contract_ready = True
    environment._p001_contact_paths = {}
    environment._time_limit = TIME_LIMIT_SECONDS
    environment._refresh_metrics = lambda: metrics
    return environment


@pytest.mark.parametrize("contact_key", ["left_wrong_end", "right_wrong_end"])
def test_actual_substep_crossed_contact_blocks_credit_without_failure(
    contact_key: str,
) -> None:
    environment = _stopgap_substep_environment(robot_table_force_n=0.0)
    environment._p001_test_force_by_key[contact_key] = (
        FORBIDDEN_CONTACT_FORCE_N + 0.001
    )
    increment = math.radians(1.0 / 18.0)
    environment._p001_test_positions = (
        environment._p001_test_positions
        + np.asarray([increment] * 9 + [-increment] * 9)
    )

    environment._after_task_physics_substep()

    assert environment._p001_positive_span_increase_deg == pytest.approx(1.0)
    assert environment._p001_bilateral_positive_span_increase_deg == 0.0
    assert environment._p001_contact_seen_this_control_frame is True
    assert environment._p001_failure_latched is False

    environment._p001_test_force_by_key[contact_key] = 0.0
    environment._trajectory_action_phase_substep_index = 0
    environment._p001_test_positions = (
        environment._p001_test_positions
        + np.asarray([increment] * 9 + [-increment] * 9)
    )
    environment._after_task_physics_substep()

    assert environment._p001_contact_seen_this_control_frame is False
    assert environment._p001_bilateral_positive_span_increase_deg == pytest.approx(
        1.0
    )
    assert environment._p001_failure_latched is False


def test_environment_keeps_table_force_diagnostic_only_during_action() -> None:
    environment = _stopgap_substep_environment(robot_table_force_n=100.0)
    increment = math.radians(1.0 / 18.0)

    for substep in range(16):
        environment._trajectory_action_phase_substep_index = substep % 4
        environment._p001_test_positions = (
            environment._p001_test_positions
            + np.asarray([increment] * 9 + [-increment] * 9)
        )
        environment._after_task_physics_substep()
        assert environment._p001_failure_latched is False

    assert environment._p001_positive_span_increase_deg == pytest.approx(16.0)
    assert environment._p001_bilateral_positive_span_increase_deg == pytest.approx(
        16.0
    )
    assert environment._p001_contact_seen_this_control_frame is False
    assert environment._p001_robot_table_hard_contact_substeps == 0
    assert environment._p001_robot_table_hard_failure_latched is False
    assert environment._p001_failure_reason == ""
    environment._p001_last_readback = replace(
        environment._p001_last_readback,
        robot_table_contact_force_n=100.0,
    )
    info = environment._task_info()
    assert info["p001_current_hard_failure"] is False
    assert info["p001_failure"] is False
    assert info["p001_failure_latched"] is False
    assert info["p001_robot_table_diagnostic_only"] is True
    assert info["p001_robot_table_diagnostic_valid"] is True
    assert info["p001_robot_table_contact_observed"] is True
    assert info["p001_robot_table_hard_contact_force_n"] is None
    assert info["p001_robot_table_hard_contact_substeps"] == 0
    assert info["p001_robot_table_hard_contact_substeps_required"] == 0
    assert info["p001_robot_table_hard_failure_latched"] is False


def test_crossed_contact_rolls_back_bilateral_credit_for_the_whole_frame() -> None:
    environment = _stopgap_substep_environment(robot_table_force_n=0.0)
    increment = math.radians(1.0 / 18.0)

    for substep, wrong_force in enumerate(
        (0.0, FORBIDDEN_CONTACT_FORCE_N + 0.001, 0.0)
    ):
        environment._trajectory_action_phase_substep_index = substep
        environment._p001_test_force_by_key["left_wrong_end"] = wrong_force
        environment._p001_test_positions = (
            environment._p001_test_positions
            + np.asarray([increment] * 9 + [-increment] * 9)
        )
        environment._after_task_physics_substep()

    assert environment._p001_positive_span_increase_deg == pytest.approx(3.0)
    assert environment._p001_bilateral_positive_span_increase_deg == 0.0
    assert environment._p001_contact_seen_this_control_frame is True

    environment._trajectory_action_phase_substep_index = 0
    environment._p001_test_positions = (
        environment._p001_test_positions
        + np.asarray([increment] * 9 + [-increment] * 9)
    )
    environment._after_task_physics_substep()

    assert environment._p001_contact_seen_this_control_frame is False
    assert environment._p001_bilateral_positive_span_increase_deg == pytest.approx(
        1.0
    )


def test_non_action_substeps_do_not_earn_progress_credit() -> None:
    environment = _stopgap_substep_environment(robot_table_force_n=100.0)
    environment._trajectory_action_phase_active = False
    increment = math.radians(1.0 / 18.0)
    environment._p001_test_positions = (
        environment._p001_test_positions
        + np.asarray([increment] * 9 + [-increment] * 9)
    )

    environment._after_task_physics_substep()

    assert environment._p001_positive_span_increase_deg == 0.0
    assert environment._p001_bilateral_positive_span_increase_deg == 0.0
    assert environment._p001_contact_seen_this_control_frame is False
    assert environment._p001_failure_latched is False


def test_transient_crossed_contact_blocks_only_the_current_control_frame() -> None:
    environment = object.__new__(HandsOnP001IsaacLabEnv)
    environment._p001_last_metrics = {}
    environment._p001_failure_latched = False
    environment._p001_failure_reported = True
    environment._p001_failure_reason = ""
    environment._p001_robot_table_hard_failure_latched = False
    metrics = evaluate_physical_readback(
        _successful_readback(sample_reset_state(9, target_open_deg=90.0)),
        sample_reset_state(9, target_open_deg=90.0),
    )
    environment._refresh_metrics = lambda: metrics

    environment._p001_contact_seen_this_control_frame = True
    environment._p001_bilateral_frame_start_deg = 9.0
    assert environment._compute_success_now() is False
    environment._p001_contact_seen_this_control_frame = False
    assert environment._compute_success_now() is True


def test_reset_clears_crossed_contact_and_legacy_table_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "handson.sim.isaaclab.abc_collection_env."
        "HandsOnABCIsaacLabEnv._sample_reset_state",
        lambda self, *, clear_manual_request: None,
    )
    environment = object.__new__(HandsOnP001IsaacLabEnv)
    environment.delta_h = np.ones(1, dtype=np.float64)
    environment._seed = 5
    environment._p001_reset_serial = 0
    environment._p001_forced_target_deg = 90.0
    environment._p001_failure_latched = True
    environment._p001_failure_reported = True
    environment._p001_failure_reason = "gross_root_drift"
    environment._p001_contact_seen_this_control_frame = True
    environment._p001_robot_table_hard_contact_substeps = 11
    environment._p001_robot_table_hard_failure_latched = True

    environment._sample_reset_state(clear_manual_request=True)

    assert environment._p001_failure_latched is False
    assert environment._p001_failure_reported is False
    assert environment._p001_failure_reason == ""
    assert environment._p001_contact_seen_this_control_frame is False
    assert environment._p001_bilateral_frame_start_deg == 0.0
    assert environment._p001_robot_table_hard_contact_substeps == 0
    assert environment._p001_robot_table_hard_failure_latched is False


def test_operator_contract_is_consistent() -> None:
    assert CONTACT_POLICY_VERSION == "p001_table_diagnostic_bilateral30_v2"
    assert P001_TASK_STATE_SCHEMA_VERSION == 2
    assert P001_TASK_STATE_DIM == 23
    assert REQUIRED_BILATERAL_EXPANSION_FRACTION == 0.30
    assert CONTROL_HZ == 30
    assert TIME_LIMIT_SECONDS == 60.0
    assert TIMEOUT_STEPS == 1800
    assert TIMEOUT_STEPS / CONTROL_HZ == TIME_LIMIT_SECONDS
    multitask = yaml.safe_load(
        (
            REPOSITORY_ROOT
            / "configs/multi_task/t48_unfold_fan.yaml"
        ).read_text(encoding="utf-8")
    )
    assert multitask["time_limit_seconds"] == 60
    assert multitask["env_name"] == "t48_unfold_fan"


def test_environment_default_timeout_uses_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _capture_init(self, task_name: str, *args, **kwargs) -> None:
        del self, args
        captured["task_name"] = task_name
        captured.update(kwargs)

    monkeypatch.setattr(
        "handson.sim.isaaclab.abc_collection_env."
        "HandsOnABCIsaacLabEnv.__init__",
        _capture_init,
    )

    HandsOnP001IsaacLabEnv(seed=17)

    assert captured["task_name"] == TASK_NAME
    assert captured["time_limit"] == TIME_LIMIT_SECONDS


def test_task_config_passes_contract_timeout_to_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from handson.tasks.t48_unfold_fan import (
        config as task_config,
    )

    captured: dict[str, object] = {}

    class _Environment:
        def __init__(self, task_name: str, **kwargs: object) -> None:
            captured["task_name"] = task_name
            captured.update(kwargs)

    monkeypatch.setattr(
        "handson.sim.isaaclab.p001_collection_env.HandsOnP001IsaacLabEnv",
        _Environment,
    )
    monkeypatch.setattr(
        task_config,
        "DualArmViveHandTeleopWrapper",
        lambda environment, teleop: environment,
    )
    monkeypatch.setattr(
        task_config,
        "HandsOnObsAdapter",
        lambda environment, proprio_keys: environment,
    )

    task_config.TaskConfig().get_environment(
        render_mode="rgb_array",
        randomize=False,
        sim_backend="isaacsim",
    )

    assert captured["task_name"] == TASK_NAME
    assert captured["time_limit"] == TIME_LIMIT_SECONDS


def test_observation_space_declares_stopgap_task_state_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "handson.sim.isaaclab.abc_collection_env."
        "HandsOnABCIsaacLabEnv._build_observation_space",
        lambda self: spaces.Dict({"state": spaces.Dict({})}),
    )
    environment = object.__new__(HandsOnP001IsaacLabEnv)

    observation_space = environment._build_observation_space()

    assert observation_space["state"]["fan_task_state"].shape == (
        P001_TASK_STATE_DIM,
    )


@pytest.mark.parametrize("target_deg", TARGET_OPEN_CHOICES_DEG)
def test_scripted_policy_is_deterministic_bimanual_and_target_conditioned(target_deg: float) -> None:
    reset_state = sample_reset_state(88, target_open_deg=target_deg)
    first = P001ScriptedPolicy(reset_state)
    second = P001ScriptedPolicy(reset_state)
    steps = (0, 74, 134, 250, 464, 500)
    actions = [first.action(step) for step in steps]
    assert all(action.shape == (ACTION_DIM,) for action in actions)
    assert all(np.all(np.isfinite(action)) for action in actions)
    assert all(np.array_equal(action, second.action(step)) for action, step in zip(actions, steps))
    initial_separation = abs(actions[2][8] - actions[2][1])
    target_separation = abs(actions[-1][8] - actions[-1][1])
    assert target_separation > initial_separation
    assert all(
        np.linalg.norm(action[index + 3 : index + 7]) == pytest.approx(1.0)
        for action in actions
        for index in (0, 7)
    )
    assert first.metadata()["object_pose_writes_after_reset"] == 0


def test_scripted_policy_accepts_measured_tcp_home_and_keeps_unit_quaternions() -> None:
    reset_state = sample_reset_state(19, target_open_deg=90.0)
    right_home = np.asarray(
        [-0.371506, -0.239510, 0.997483, -math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0]
    )
    left_home = np.asarray(
        [-0.371506, 0.239510, 0.997483, math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]
    )
    policy = P001ScriptedPolicy(
        reset_state,
        right_home=right_home,
        left_home=left_home,
    )
    for step in (0, 25, 50, 74, 75):
        action = policy.action(step)
        assert np.linalg.norm(action[3:7]) == pytest.approx(1.0)
        assert np.linalg.norm(action[10:14]) == pytest.approx(1.0)
        assert np.all(np.isfinite(action))


def test_full_bimanual_wuji_action_space_keeps_native_hand_dimension() -> None:
    env = SimpleNamespace(
        action_space=spaces.Box(-np.inf, np.inf, shape=(ACTION_DIM,)),
        hand_type="wuji1",
    )
    assert _dual_arm_hand_dim(env) == 20


def test_native_p001_action_is_stored_in_certifier_side_major_order() -> None:
    right_pose = np.arange(7, dtype=np.float64) + 10.0
    left_pose = np.arange(7, dtype=np.float64) + 20.0
    right_hand = np.arange(20, dtype=np.float64) + 30.0
    left_hand = np.arange(20, dtype=np.float64) + 60.0
    native = np.concatenate([right_pose, left_pose, right_hand, left_hand])

    stored = p001_action_to_side_major_storage(native)

    np.testing.assert_array_equal(
        stored[:27],
        np.concatenate([right_pose, right_hand]),
    )
    np.testing.assert_array_equal(
        stored[27:],
        np.concatenate([left_pose, left_hand]),
    )
