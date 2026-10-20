from __future__ import annotations

import copy
import io
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from taskdev.tasks.bench320_as019_screwdriver_safe_orientation_handover import (
    run_acceptance_validation as acceptance,
)


def _valid_telemetry_record() -> dict[str, object]:
    finger = {
        "query_valid": True,
        "query_error": "",
        "raw_contact_count": 0,
        "accepted_contact_count": 0,
        "accepted_force_n": 0.0,
        "contact_local_x_m": None,
        "radial_yz_m": None,
        "radial_radius_m": None,
        "minimum_separation_m": None,
        "maximum_separation_m": None,
        "maximum_penetration_m": 0.0,
        "penetration_violation": False,
    }
    pose = {
        "position_m": [0.0, 0.0, 0.0],
        "rotation_matrix": np.eye(3).tolist(),
    }
    return {
        "trial_step": 17,
        "env_step": 17,
        "stage": 0,
        "automatic_acquire_gate_required": True,
        "automatic_acquire_gate_open": False,
        "automatic_arrival_gate": {
            "gate": "approach_not_started",
            "open": False,
        },
        "right_semantic_owner": False,
        "active_right_wrench": False,
        "right_palm_tool_local_pose": pose,
        "right_palm_command_tracking": {
            "policy_action_step": 16,
            "commanded_tool_local_pose": pose,
            "translation_residual_tool_local_xyz_m": [0.0, 0.0, 0.0],
            "translation_residual_norm_m": 0.0,
            "rotation_residual_rad": 0.0,
            "command_error": None,
        },
        "right_hand_command_tracking": {
            "policy_action_step": 16,
            "hand_joint_ids": list(range(20)),
            "commanded_20d_rad": [0.0] * 20,
            "measured_20d_rad": [0.0] * 20,
            "measured_minus_commanded_20d_rad": [0.0] * 20,
            "maximum_abs_residual_rad": 0.0,
            "rms_residual_rad": 0.0,
            "command_error": None,
            "measurement_error": None,
            "residual_error": None,
        },
        "right_arm_control": {
            "trace_env_step": 16,
            "policy_action_step": 16,
            "physics_backend": "physx",
            "arm_ik_backend": "haorui",
            "haorui_ik_enabled": True,
            "ik_command_type": "pose",
            "ik_status": "haorui",
            "requested_pose_wxyz": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "current_pose_before_ik_wxyz": [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            "admitted_pose_wxyz": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "pre_joint_limit_target_7d_rad": [0.0] * 7,
            "previous_applied_target_7d_rad": [0.0] * 7,
            "post_joint_limit_target_7d_rad": [0.0] * 7,
            "final_command_target_7d_rad": [0.0] * 7,
            "sim_drive_target_7d_rad": [0.0] * 7,
            "measured_joint_position_7d_rad": [0.0] * 7,
            "measured_joint_velocity_7d_radps": [0.0] * 7,
            "arm_joint_ids": list(range(7)),
            "requested_to_admitted_translation_m": 0.0,
            "requested_to_admitted_rotation_rad": 0.0,
            "admitted_pose_step_translation_m": 0.0,
            "admitted_pose_step_rotation_rad": 0.0,
            "maximum_joint_limiter_correction_rad": 0.0,
            "maximum_applied_joint_step_rad": 0.0,
            "maximum_final_command_step_rad": 0.0,
            "maximum_drive_buffer_delta_rad": 0.0,
            "error": None,
        },
        "contact_query_valid": True,
        "contact_query_error": "",
        "right_wrench_sample": {"right_active": False, "pass": True},
        "table_safety": {"pass": True},
        "right_fingers": {
            f"finger{index}": copy.deepcopy(finger) for index in range(1, 6)
        },
    }


def test_stage0_canary_cli_is_single_seed_and_mutually_exclusive() -> None:
    args = acceptance._parser().parse_args(
        [
            "--output-dir",
            "/tmp/as019-stage0-canary",
            "--stage0-canary-seed",
            "103",
            "--stage0-max-steps",
            "240",
        ]
    )
    assert args.stage0_canary_seed == 103
    assert args.stage0_max_steps == 240
    assert args.positive_max_steps is None
    assert args.close_watchdog_seconds == pytest.approx(10.0)
    assert args.outcomes_only is False
    assert args.camera_stability_only is False

    disabled_watchdog = acceptance._parser().parse_args(
        [
            "--output-dir",
            "/tmp/as019-stage0-canary",
            "--stage0-canary-seed",
            "103",
            "--close-watchdog-seconds",
            "0",
        ]
    )
    assert disabled_watchdog.close_watchdog_seconds == 0.0
    assert disabled_watchdog.stage0_max_steps == 320

    with pytest.raises(SystemExit):
        acceptance._parser().parse_args(
            [
                "--output-dir",
                "/tmp/as019-stage0-canary",
                "--outcomes-only",
                "--stage0-canary-seed",
                "103",
            ]
        )


def test_reset_metrics_are_recovered_from_cached_task_info_only() -> None:
    info = {
        "task_id": "bench320_as019_screwdriver_safe_orientation_handover",
        "as019_table_safety_query_valid": True,
        "as019_tool_table_supported": True,
        "as019_tool_table_max_penetration_m": 0.001,
        "unrelated": "ignored",
    }

    assert acceptance._as019_metrics_from_reset_info(info) == {
        "table_safety_query_valid": True,
        "tool_table_supported": True,
        "tool_table_max_penetration_m": 0.001,
    }
    assert acceptance._as019_metrics_from_reset_info(None) == {}


def test_validation_fingerprint_is_stable_and_scoped(tmp_path: Path) -> None:
    for relative in acceptance._FINGERPRINT_RELATIVE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{relative}\n".encode())

    first = acceptance._validation_content_fingerprint(tmp_path)
    second = acceptance._validation_content_fingerprint(tmp_path)
    assert first["combined_sha256"] == second["combined_sha256"]
    assert set(first["files"]) == set(acceptance._FINGERPRINT_RELATIVE_PATHS)

    changed_path = tmp_path / acceptance._FINGERPRINT_RELATIVE_PATHS[0]
    changed_path.write_bytes(b"changed\n")
    changed = acceptance._validation_content_fingerprint(tmp_path)
    assert changed["combined_sha256"] != first["combined_sha256"]


def test_expected_checkpoint_is_required_and_must_match_exactly() -> None:
    actual = "a" * 64
    fingerprint = {"combined_sha256": actual}
    assert acceptance._expected_checkpoint_gate(fingerprint, {})["reason"] == (
        "missing_expected_checkpoint"
    )
    assert acceptance._expected_checkpoint_gate(
        fingerprint, {"AS019_EXPECTED_CHECKPOINT": "A" * 64}
    )["reason"] == "invalid_expected_checkpoint_format"
    assert acceptance._expected_checkpoint_gate(
        fingerprint, {"AS019_EXPECTED_CHECKPOINT": "b" * 64}
    )["reason"] == "checkpoint_mismatch"
    matched = acceptance._expected_checkpoint_gate(
        fingerprint, {"AS019_EXPECTED_CHECKPOINT": actual}
    )
    assert matched["pass"] is True
    assert matched["reason"] == ""


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_ambient_automatic_policy_truthy_values_are_rejected(value: str) -> None:
    gate = acceptance._ambient_automatic_policy_gate(
        {"HANDSON_AS019_AUTOMATIC_POLICY": value}
    )
    assert gate["automatic_policy_enabled"] is True
    assert gate["pass"] is False


def test_isaaclab_haorui_environment_is_explicit_and_exact() -> None:
    missing = acceptance._required_arm_ik_environment_gate({})
    assert missing["pass"] is False
    assert missing["reason"] == "missing_required_isaaclab_arm_ik_backend"

    dls = acceptance._required_arm_ik_environment_gate(
        {
            "HANDSON_ISAACLAB_ARM_IK_BACKEND": "dls",
            "HANDSON_ISAACLAB_ALLOW_DLS_FALLBACK": "0",
            "HANDSON_AS019_CONTROL_TRACE": "1",
        }
    )
    assert dls["pass"] is False
    assert dls["reason"] == "unexpected_isaaclab_arm_ik_backend"

    matched = acceptance._required_arm_ik_environment_gate(
        {
            "HANDSON_ISAACLAB_ARM_IK_BACKEND": " haorui_new ",
            "HANDSON_ISAACLAB_ALLOW_DLS_FALLBACK": "0",
            "HANDSON_AS019_CONTROL_TRACE": "1",
        }
    )
    assert matched["pass"] is True
    assert matched["normalized_value"] == "haorui_new"

    fallback_enabled = acceptance._required_arm_ik_environment_gate(
        {
            "HANDSON_ISAACLAB_ARM_IK_BACKEND": "haorui_new",
            "HANDSON_ISAACLAB_ALLOW_DLS_FALLBACK": "1",
            "HANDSON_AS019_CONTROL_TRACE": "1",
        }
    )
    assert fallback_enabled["pass"] is False
    assert fallback_enabled["reason"] == "dls_fallback_must_be_disabled"

    trace_disabled = acceptance._required_arm_ik_environment_gate(
        {
            "HANDSON_ISAACLAB_ARM_IK_BACKEND": "haorui_new",
            "HANDSON_ISAACLAB_ALLOW_DLS_FALLBACK": "0",
            "HANDSON_AS019_CONTROL_TRACE": "0",
        }
    )
    assert trace_disabled["pass"] is False
    assert trace_disabled["reason"] == "as019_control_trace_must_be_enabled"


def test_runtime_control_gate_rejects_dls_or_haorui_initialization_failure() -> None:
    raw = SimpleNamespace(
        _physics_backend="physx",
        _arm_ik_backend="haorui",
        _haorui_ik_enabled=True,
        _ik_command_types={"right": "pose", "left": "pose"},
        _as019_control_trace_enabled=True,
    )
    assert acceptance._runtime_control_gate(raw)["pass"] is True

    raw._arm_ik_backend = "dls"
    assert acceptance._runtime_control_gate(raw)["pass"] is False
    raw._arm_ik_backend = "haorui"
    raw._haorui_ik_enabled = False
    assert acceptance._runtime_control_gate(raw)["pass"] is False
    raw._haorui_ik_enabled = True
    raw._ik_command_types["right"] = "position"
    assert acceptance._runtime_control_gate(raw)["pass"] is False
    raw._ik_command_types["right"] = "pose"
    raw._as019_control_trace_enabled = False
    assert acceptance._runtime_control_gate(raw)["pass"] is False


def test_runtime_rate_gate_requires_30hz_120hz_and_four_substeps() -> None:
    raw = SimpleNamespace(
        control_dt=1.0 / 30.0,
        physics_dt=1.0 / 120.0,
        _sim_dt=1.0 / 120.0,
        _n_substeps=4,
    )
    passed = acceptance._runtime_rate_gate(raw)
    assert passed["pass"] is True
    assert passed["control_hz"] == pytest.approx(30.0)
    assert passed["physics_hz"] == pytest.approx(120.0)

    raw._n_substeps = 3
    failed = acceptance._runtime_rate_gate(raw)
    assert failed["pass"] is False
    assert failed["checks"]["four_substeps"] is False


def test_stage0_seed_plan_and_reset_gate_allow_only_requested_reset() -> None:
    assert acceptance._validation_seed_plan(103) == {
        "environment_construction_seed": 103,
        "initial_reset_seed": None,
    }
    assert acceptance._validation_seed_plan(None) == {
        "environment_construction_seed": 9000,
        "initial_reset_seed": 9000,
    }

    class Env:
        reset_seeds: list[int] = []

        def reset(self, *, seed: int):
            self.reset_seeds.append(seed)
            return {}, {}

    env = Env()
    audited: list[int] = []
    acceptance._reset_with_audit(env, 103, audited)
    assert env.reset_seeds == [103]
    assert acceptance._stage0_reset_gate(
        requested_seed=103, explicit_reset_seeds=audited
    )["pass"] is True
    audited.insert(0, 9000)
    assert acceptance._stage0_reset_gate(
        requested_seed=103, explicit_reset_seeds=audited
    )["pass"] is False


def test_positive_budget_loads_task_yaml_or_explicit_cli_override(
    tmp_path: Path,
) -> None:
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(
        "validation:\n  automatic_policy:\n    positive_max_steps: 1200\n",
        encoding="utf-8",
    )
    loaded = acceptance._automatic_positive_max_steps(task_yaml)
    assert loaded["max_steps"] == 1200
    assert loaded["source"] == "task_yaml"
    overridden = acceptance._automatic_positive_max_steps(
        task_yaml, cli_override=1400
    )
    assert overridden == {"max_steps": 1400, "source": "cli", "task_yaml": None}

    task_yaml.write_text("validation: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="automatic_policy.positive_max_steps"):
        acceptance._automatic_positive_max_steps(task_yaml)


def test_zero_activation_is_not_positive_right_wrench_coverage() -> None:
    coverage = {
        "positive": {
            "sampled_steps": 10,
            "right_tail_physical_steps": 0,
            "active_right_wrench_steps": 0,
            "active_and_physical_steps": 0,
            "active_without_physical_steps": 0,
        }
    }
    result = acceptance._right_wrench_gate_result(
        [],
        coverage,
        require_positive_coverage=True,
    )
    assert result["invariant_pass"] is True
    assert result["positive_coverage_pass"] is False
    assert result["pass"] is False

    coverage["positive"]["right_tail_physical_steps"] = 1
    coverage["positive"]["active_right_wrench_steps"] = 1
    coverage["positive"]["active_and_physical_steps"] = 1
    result = acceptance._right_wrench_gate_result(
        [],
        coverage,
        require_positive_coverage=True,
    )
    assert result["positive_coverage_pass"] is True
    assert result["pass"] is True

    coverage["positive"]["sampled_steps"] = 11
    coverage["positive"]["active_right_wrench_steps"] = 2
    coverage["positive"]["active_without_physical_steps"] = 1
    result = acceptance._right_wrench_gate_result(
        [],
        coverage,
        require_positive_coverage=True,
    )
    assert result["positive_coverage_pass"] is False
    assert result["pass"] is False


def _valid_right_wrench_raw() -> SimpleNamespace:
    return SimpleNamespace(
        env_step=18,
        _n_substeps=4,
        _as019_active_wrench_owners=("right",),
        _as019_last_metrics={
            "right_tail_physical": False,
            "metrics_generation": 42,
        },
        _as019_active_wrench_gate_snapshot={
            "active_wrench_owners": ("right",),
            "control_step_before": 17,
            "physics_substep_index": 3,
            "physics_substep_count": 4,
            "metrics_generation": 41,
            "contact_query_valid": True,
            "right_tail_physical": True,
            "right_close_command": True,
            "right_wrench_invariant": True,
            "apply_succeeded": True,
        },
    )


def test_right_wrench_gate_uses_same_generation_snapshot_not_post_step() -> None:
    raw = _valid_right_wrench_raw()
    sample = acceptance._right_wrench_sample_gate(raw)
    assert sample["pass"] is True
    assert sample["right_active"] is True
    assert sample["snapshot_right_tail_physical"] is True
    assert sample["post_step_right_tail_physical"] is False

    raw._as019_active_wrench_gate_snapshot["right_tail_physical"] = False
    raw._as019_last_metrics["right_tail_physical"] = True
    rejected = acceptance._right_wrench_sample_gate(raw)
    assert rejected["pass"] is False
    assert rejected["snapshot_right_tail_physical"] is False
    assert rejected["post_step_right_tail_physical"] is True
    assert "gate:right_tail_physical" in rejected["failure_reasons"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("control_step_before", 16, "lineage:control_step_matches"),
        ("physics_substep_index", 2, "lineage:final_physics_substep"),
        (
            "physics_substep_count",
            3,
            "lineage:physics_substep_count_matches_runtime",
        ),
        ("metrics_generation", True, "lineage:metrics_generation_is_valid"),
        ("contact_query_valid", False, "gate:contact_query_valid"),
        ("right_close_command", False, "gate:right_close_command"),
        ("right_wrench_invariant", False, "gate:right_wrench_invariant"),
        ("apply_succeeded", False, "gate:apply_succeeded"),
    ],
)
def test_right_wrench_gate_rejects_broken_lineage_or_gate(
    field: str, value: object, reason: str
) -> None:
    raw = _valid_right_wrench_raw()
    raw._as019_active_wrench_gate_snapshot[field] = value
    sample = acceptance._right_wrench_sample_gate(raw)
    assert sample["pass"] is False
    assert reason in sample["failure_reasons"]


def test_post_step_generation_is_diagnostic_only() -> None:
    raw = _valid_right_wrench_raw()
    raw._as019_last_metrics["metrics_generation"] = 41

    sample = acceptance._right_wrench_sample_gate(raw)

    assert sample["pass"] is True
    assert sample["lineage_pass"] is True
    assert sample["diagnostic_checks"] == {
        "post_step_generation_after_snapshot": False
    }
    assert sample["failure_reasons"] == []


def test_right_wrench_apply_failure_is_visible_after_current_owner_clears() -> None:
    raw = _valid_right_wrench_raw()
    raw._as019_active_wrench_owners = ()
    raw._as019_active_wrench_gate_snapshot["apply_succeeded"] = False
    sample = acceptance._right_wrench_sample_gate(raw)
    assert sample["right_active"] is True
    assert sample["post_step_active_wrench_owners"] == []
    assert sample["pass"] is False
    assert "gate:apply_succeeded" in sample["failure_reasons"]


@pytest.mark.parametrize(
    ("kwargs", "expected_limit", "expected_reason"),
    [
        (
            {
                "stage0_canary": True,
                "control_steps": 320,
                "maximum_trial_steps": 320,
                "stage0_exit_step": None,
                "confirmation_complete": False,
            },
            True,
            "stage0_acquisition_budget_exhausted",
        ),
        (
            {
                "stage0_canary": False,
                "control_steps": 1200,
                "maximum_trial_steps": 1200,
                "stage0_exit_step": None,
                "confirmation_complete": False,
            },
            True,
            "positive_validation_budget_exhausted",
        ),
        (
            {
                "stage0_canary": True,
                "control_steps": 321,
                "maximum_trial_steps": 320,
                "stage0_exit_step": 320,
                "confirmation_complete": True,
            },
            False,
            "stage0_confirmation_complete",
        ),
    ],
)
def test_validation_trial_terminal_status_records_validator_budget(
    kwargs: dict[str, object], expected_limit: bool, expected_reason: str
) -> None:
    result = acceptance._validation_trial_terminal_status(
        **kwargs,
        success=False,
        terminated=False,
        truncated=False,
        failure_reason="",
    )
    assert result == {
        "time_limit_reached": expected_limit,
        "terminal_reason": expected_reason,
    }


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"success": True}, "success"),
        ({"failure_reason": "unsafe"}, "task_failure:unsafe"),
        ({"terminated": True}, "task_terminated"),
        ({"truncated": True}, "task_truncated"),
    ],
)
def test_validation_trial_terminal_status_does_not_mislabel_task_terminal(
    overrides: dict[str, object], reason: str
) -> None:
    arguments: dict[str, object] = {
        "stage0_canary": False,
        "control_steps": 1200,
        "maximum_trial_steps": 1200,
        "stage0_exit_step": None,
        "confirmation_complete": False,
        "success": False,
        "terminated": False,
        "truncated": False,
        "failure_reason": "",
    }
    arguments.update(overrides)
    result = acceptance._validation_trial_terminal_status(**arguments)
    assert result == {"time_limit_reached": False, "terminal_reason": reason}


def test_table_safety_gate_uses_cached_separation_evidence_without_force() -> None:
    metrics = {
        "table_safety_query_valid": True,
        "table_safety_query_error": "",
        "tool_table_supported": True,
        "tool_table_analytic_gap_m": 0.0002,
        "tool_table_min_separation_m": -0.004,
        "tool_table_penetration_violation": False,
        "tool_table_contact_paths": ("/World/envs/env_0/Table",),
        "arm_table_physical_contact": False,
        "arm_table_min_separation_m": float("inf"),
        "arm_table_contact_paths": (),
        "hand_table_physical_contact": True,
        "hand_table_min_separation_m": -0.003,
        "hand_table_penetration_violation": False,
        "hand_table_contact_paths": ("/World/envs/env_0/RightRobot/palm",),
    }
    sample = acceptance._table_safety_sample_gate(metrics)
    assert sample["pass"] is True
    assert sample["tool_table_supported"] is True
    assert sample["tool_table_max_penetration_m"] == pytest.approx(0.004)
    assert sample["hand_table_max_penetration_m"] == pytest.approx(0.003)
    assert sample["arm_table_max_penetration_m"] == pytest.approx(0.0)
    assert sample["force_gate_n"] is None
    assert sample["force_used_for_pass"] is False
    json.dumps(sample, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("table_safety_query_valid", "table_safety_query_invalid"),
        (
            "tool_table_penetration_violation",
            "tool_table_penetration_violation",
        ),
        (
            "hand_table_penetration_violation",
            "hand_table_penetration_violation",
        ),
    ],
)
def test_table_safety_gate_fails_closed_for_each_runtime_invariant(
    field: str, reason: str
) -> None:
    metrics = {
        "table_safety_query_valid": True,
        "arm_table_physical_contact": False,
        "tool_table_penetration_violation": False,
        "hand_table_penetration_violation": False,
    }
    metrics[field] = False if field == "table_safety_query_valid" else True
    sample = acceptance._table_safety_sample_gate(metrics)
    assert sample["pass"] is False
    assert reason in sample["failure_reasons"]


def test_arm_table_contact_is_diagnostic_only() -> None:
    sample = acceptance._table_safety_sample_gate(
        {
            "table_safety_query_valid": True,
            "arm_table_physical_contact": True,
            "arm_table_min_separation_m": -0.002,
            "tool_table_penetration_violation": False,
            "hand_table_penetration_violation": False,
        }
    )
    assert sample["pass"] is True
    assert sample["arm_table_physical_contact"] is True
    assert sample["arm_table_max_penetration_m"] == pytest.approx(0.002)


def test_reset_clearance_accepts_safe_sleep_without_claiming_support() -> None:
    metrics = {
        "table_safety_query_valid": True,
        "tool_table_supported": False,
        "tool_table_analytic_gap_m": 0.00019995,
        "tool_table_min_separation_m": float("inf"),
        "tool_table_penetration_violation": False,
        "arm_table_physical_contact": False,
        "hand_table_penetration_violation": False,
    }
    gate = acceptance._reset_table_clearance_gate(
        metrics,
        expected_gap_m=0.0002,
        tolerance_m=1.2e-7,
    )
    assert gate["pass"] is True
    assert gate["physical_support_required"] is False
    assert gate["physical_support_observed"] is False
    assert gate["analytic_gap_matches_reset_contract"] is True


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        (
            {"tool_table_analytic_gap_m": 0.0031},
            "reset_table_clearance_mismatch",
        ),
        (
            {"tool_table_analytic_gap_m": -0.0051},
            "reset_table_clearance_mismatch",
        ),
        (
            {"table_safety_query_valid": False},
            "table_safety_query_invalid",
        ),
        (
            {"tool_table_penetration_violation": True},
            "tool_table_penetration_violation",
        ),
        (
            {"hand_table_penetration_violation": True},
            "hand_table_penetration_violation",
        ),
    ],
)
def test_reset_clearance_fails_closed_without_reusing_support_window(
    updates: dict[str, object], reason: str
) -> None:
    metrics = {
        "table_safety_query_valid": True,
        "tool_table_supported": False,
        "tool_table_analytic_gap_m": 0.0002,
        "tool_table_penetration_violation": False,
        "arm_table_physical_contact": False,
        "hand_table_penetration_violation": False,
    }
    metrics.update(updates)
    gate = acceptance._reset_table_clearance_gate(
        metrics,
        expected_gap_m=0.0002,
        tolerance_m=1.2e-7,
    )
    assert gate["pass"] is False
    assert reason in gate["failure_reasons"]


def test_per_finger_rows_report_exact_accepted_geometry_and_penetration() -> None:
    rows = [
        (2.0, np.asarray([-0.100, 0.040, 0.010]), -0.001),
        (1.0, np.asarray([-0.080, 0.010, -0.010]), 0.0),
        (4.0, np.asarray([-0.030, 0.030, 0.000]), 0.0),
        (4.0, np.asarray([-0.090, 0.040, 0.000]), -0.006),
    ]
    class Runtime:
        _CONTACT_FORCE_N = 0.5
        _CONTACT_MAX_SEPARATION_M = 0.0
        _CONTACT_MAX_PENETRATION_M = 0.005
        _RIGHT_TAIL_CONTACT_LOCAL_X_MIN_M = -0.120
        _RIGHT_TAIL_CONTACT_LOCAL_X_MAX_M = -0.060
        _HANDLE_LOCAL_MIN = np.asarray([-0.125, 0.000, -0.020])
        _HANDLE_LOCAL_MAX = np.asarray([-0.050, 0.050, 0.020])
        _HANDLE_LOCAL_CENTER = np.asarray([-0.0875, 0.025, 0.000])

    class Raw:
        _as019_contact_query_valid = True
        _as019_contact_query_error = ""
        _as019_right_per_finger_views = {
            f"finger{index}": index for index in range(1, 6)
        }

        @staticmethod
        def _tool_matrix() -> np.ndarray:
            return np.eye(4)

        @staticmethod
        def _detailed_contact_rows(view, _name):
            return (rows, True) if view == 1 else ([], False)

        @staticmethod
        def _inside_local_envelope(point, low, high) -> bool:
            return bool(np.all(point >= low) and np.all(point <= high))

    record = acceptance._right_finger_contact_telemetry(Raw(), Runtime())[
        "finger1"
    ]

    assert record["raw_contact_count"] == 4
    assert record["accepted_contact_count"] == 2
    assert record["accepted_force_n"] == pytest.approx(3.0)
    assert record["contact_local_x_m"] == pytest.approx(-0.28 / 3.0)
    assert record["radial_yz_m"] == pytest.approx([0.005, 0.01 / 3.0])
    assert record["radial_radius_m"] == pytest.approx(
        math.hypot(0.005, 0.01 / 3.0)
    )
    assert record["minimum_separation_m"] == pytest.approx(-0.001)
    assert record["maximum_separation_m"] == pytest.approx(0.0)
    assert record["maximum_penetration_m"] == pytest.approx(0.006)
    assert record["penetration_violation"] is True


def test_grouped_contact_records_preserve_exact_finger_attribution() -> None:
    class Runtime:
        _CONTACT_FORCE_N = 0.5
        _CONTACT_MAX_SEPARATION_M = 0.0
        _CONTACT_MAX_PENETRATION_M = 0.005
        _RIGHT_TAIL_CONTACT_LOCAL_X_MIN_M = -0.120
        _RIGHT_TAIL_CONTACT_LOCAL_X_MAX_M = -0.060
        _HANDLE_LOCAL_MIN = np.asarray([-0.125, 0.000, -0.020])
        _HANDLE_LOCAL_MAX = np.asarray([-0.050, 0.050, 0.020])
        _HANDLE_LOCAL_CENTER = np.asarray([-0.0875, 0.025, 0.000])

    class Raw:
        _as019_contact_query_valid = True
        _as019_contact_query_error = ""
        _as019_owner_contact_records = {
            "right_fingers": [
                {
                    "filter_path": "/World/envs/env_0/right_finger1_link",
                    "force": 2.0,
                    "point": np.asarray([-0.100, 0.040, 0.010]),
                    "separation": -0.001,
                },
                {
                    "filter_path": "/World/envs/env_0/right_finger3_link",
                    "force": 1.5,
                    "point": np.asarray([-0.080, 0.010, -0.010]),
                    "separation": 0.0,
                },
            ]
        }

        @staticmethod
        def _tool_matrix() -> np.ndarray:
            return np.eye(4)

        @staticmethod
        def _finger_name_from_filter_path(side: str, filter_path: str) -> str | None:
            name = filter_path.rsplit("/", 1)[-1]
            prefix = f"{side}_finger"
            if not name.startswith(prefix):
                return None
            index = name[len(prefix) :].split("_", 1)[0]
            return f"finger{index}" if index in {"1", "2", "3", "4", "5"} else None

        @staticmethod
        def _inside_local_envelope(point, low, high) -> bool:
            return bool(np.all(point >= low) and np.all(point <= high))

    records = acceptance._right_finger_contact_telemetry(Raw(), Runtime())

    assert records["finger1"]["raw_contact_count"] == 1
    assert records["finger1"]["accepted_force_n"] == pytest.approx(2.0)
    np.testing.assert_allclose(
        records["finger1"]["raw_contact_local_xyz_m"],
        [[-0.100, 0.040, 0.010]],
    )
    assert records["finger3"]["raw_contact_count"] == 1
    assert records["finger3"]["accepted_force_n"] == pytest.approx(1.5)
    assert records["finger2"]["raw_contact_count"] == 0
    assert all(record["query_valid"] is True for record in records.values())


def test_diagnostic_contact_query_restores_production_validity_state() -> None:
    class Raw:
        _as019_contact_query_valid = False
        _as019_contact_query_error = "production_failure"

        def _detailed_contact_rows(self, _view, _name):
            self._as019_contact_query_valid = False
            self._as019_contact_query_error = "diagnostic_failure"
            return [], False

    raw = Raw()
    rows, penetration, valid, error = acceptance._observational_detailed_contact_rows(
        raw,
        object(),
        "finger1",
    )
    assert rows == []
    assert penetration is False
    assert valid is False
    assert error == "diagnostic_failure"
    assert raw._as019_contact_query_valid is False
    assert raw._as019_contact_query_error == "production_failure"


def test_snapshot_contains_phase_and_palm_pose_in_live_tool_frame() -> None:
    class Runtime:
        _HANDLE_LOCAL_CENTER = np.asarray([-0.0875, 0.025, 0.000])

    class Raw:
        env_step = 17
        _as019_stage = 0
        _as019_automatic_phase = 2
        _as019_automatic_acquire_gate_required = True
        _as019_automatic_acquire_gate_open = False
        _as019_automatic_arrival_gate = {
            "gate": "approach_not_started",
            "open": False,
        }
        _as019_assist_anchor = {}
        _as019_active_wrench_owners = ()
        _as019_active_wrench_gate_snapshot = {}
        _n_substeps = 4
        _as019_right_per_finger_views = {
            f"finger{index}": index for index in range(1, 6)
        }
        _as019_contact_query_valid = True
        _as019_contact_query_error = ""
        _as019_last_policy_action_step = 16
        _as019_last_metrics = {
            "contact_query_valid": True,
            "contact_query_error": "",
            "right_tail_physical": False,
            "right_thumb_tail_force": 0.0,
            "right_opposing_tail_force": 0.0,
            "right_contact_local_x": float("nan"),
            "right_contact_radial_cosine": float("nan"),
            "right_contact_radial_span_m": 0.0,
            "right_contact_min_separation_m": float("inf"),
            "right_contact_max_separation_m": float("inf"),
            "right_contact_penetration_violation": False,
            "table_safety_query_valid": True,
            "table_safety_query_error": "",
            "tool_table_supported": True,
            "tool_table_analytic_gap_m": 0.0002,
            "tool_table_min_separation_m": 0.0002,
            "tool_table_penetration_violation": False,
            "tool_table_contact_paths": ("/World/envs/env_0/Table",),
            "arm_table_physical_contact": False,
            "arm_table_min_separation_m": float("inf"),
            "arm_table_contact_paths": (),
            "hand_table_physical_contact": False,
            "hand_table_min_separation_m": float("inf"),
            "hand_table_penetration_violation": False,
            "hand_table_contact_paths": (),
        }

        @staticmethod
        def _tool_matrix() -> np.ndarray:
            return np.eye(4)

        @staticmethod
        def _palm_matrix(side: str) -> np.ndarray:
            assert side == "right"
            palm = np.eye(4)
            palm[:3, 3] = [0.10, 0.20, 0.30]
            return palm

        @staticmethod
        def _detailed_contact_rows(_view, _name):
            return [], False

    commanded_hand = np.linspace(0.1, 2.0, 20)
    measured_hand = commanded_hand + 0.01
    action = np.zeros(54, dtype=np.float64)
    action[:7] = [
        0.07,
        0.16,
        0.30,
        math.sqrt(0.5),
        0.0,
        0.0,
        math.sqrt(0.5),
    ]
    action[14:34] = commanded_hand
    joint_ids = list(range(7, 27))
    joint_positions = np.zeros((1, 30), dtype=np.float64)
    joint_positions[0, joint_ids] = measured_hand
    raw = Raw()
    raw._as019_last_policy_action = action
    raw._hand_joint_ids = {"right": joint_ids}
    arm_joint_ids = list(range(7))
    raw._arm_joint_ids = {"right": arm_joint_ids}
    raw._physics_backend = "physx"
    raw._arm_ik_backend = "haorui"
    raw._haorui_ik_enabled = True
    raw._ik_command_types = {"right": "pose"}
    raw._joint_targets = {"right": np.zeros((1, 30), dtype=np.float64)}
    raw._as019_arm_control_trace = {
        "right": {
            "env_step": 16,
            "policy_action_step": 16,
            "status": "haorui",
            "requested_pose_wxyz": action[:7].copy(),
            "current_pose_before_ik_wxyz": action[:7].copy(),
            "admitted_pose_wxyz": action[:7].copy(),
            "pre_joint_limit_target_7d_rad": np.zeros(7),
            "previous_applied_target_7d_rad": np.zeros(7),
            "post_joint_limit_target_7d_rad": np.zeros(7),
        }
    }
    raw._scene = {
        "right_robot": type(
            "Robot",
            (),
            {
                "data": type(
                    "Data",
                    (),
                    {
                        "joint_pos": joint_positions,
                        "joint_vel": np.zeros((1, 30), dtype=np.float64),
                        "joint_pos_target": np.zeros((1, 30), dtype=np.float64),
                    },
                )()
            },
        )()
    }

    record = acceptance._positive_telemetry_snapshot(
        raw, Runtime(), trial_step=17, sample_reason=["periodic_30_steps"]
    )
    assert record["phase"] == 2
    assert record["stage"] == 0
    assert record["right_palm_tool_local_pose"]["position_m"] == pytest.approx(
        [0.10, 0.20, 0.30]
    )
    np.testing.assert_allclose(
        record["right_palm_tool_local_pose"]["rotation_matrix"], np.eye(3)
    )
    palm_tracking = record["right_palm_command_tracking"]
    assert palm_tracking["policy_action_step"] == 16
    assert palm_tracking["command_error"] is None
    assert palm_tracking["commanded_tool_local_pose"]["position_m"] == (
        pytest.approx([0.07, 0.16, 0.30])
    )
    assert palm_tracking["translation_residual_tool_local_xyz_m"] == (
        pytest.approx([0.03, 0.04, 0.0])
    )
    assert palm_tracking["translation_residual_norm_m"] == pytest.approx(0.05)
    assert palm_tracking["rotation_residual_rad"] == pytest.approx(math.pi / 2.0)

    hand_tracking = record["right_hand_command_tracking"]
    assert hand_tracking["policy_action_step"] == 16
    assert hand_tracking["hand_joint_ids"] == joint_ids
    assert hand_tracking["commanded_20d_rad"] == pytest.approx(commanded_hand)
    assert hand_tracking["measured_20d_rad"] == pytest.approx(measured_hand)
    assert hand_tracking["measured_minus_commanded_20d_rad"] == pytest.approx(
        np.full(20, 0.01)
    )
    assert hand_tracking["maximum_abs_residual_rad"] == pytest.approx(0.01)
    assert hand_tracking["rms_residual_rad"] == pytest.approx(0.01)
    assert hand_tracking["command_error"] is None
    assert hand_tracking["measurement_error"] is None
    assert record["right_tail_aggregate"]["contact_local_x_m"] is None
    assert record["right_tail_aggregate"]["minimum_separation_m"] is None
    assert record["validity"]["pass"] is True
    json.dumps(record, allow_nan=False)


def test_snapshot_reports_command_and_measurement_errors_as_json_nulls() -> None:
    class Runtime:
        _HANDLE_LOCAL_CENTER = np.asarray([-0.0875, 0.025, 0.000])

    class Raw:
        env_step = 1
        _as019_stage = 0
        _as019_automatic_phase = 0
        _as019_active_wrench_owners = ()
        _as019_right_per_finger_views = {}
        _as019_last_policy_action_step = 0
        _as019_last_metrics = {
            "right_contact_local_x": float("nan"),
            "right_contact_min_separation_m": float("inf"),
            "right_contact_max_separation_m": float("inf"),
        }
        _as019_last_policy_action = np.zeros(54, dtype=np.float64)
        _hand_joint_ids = {"right": []}
        _scene = None

        @staticmethod
        def _tool_matrix() -> np.ndarray:
            return np.eye(4)

        @staticmethod
        def _palm_matrix(side: str) -> np.ndarray:
            assert side == "right"
            return np.eye(4)

    record = acceptance._positive_telemetry_snapshot(
        Raw(), Runtime(), trial_step=1, sample_reason=["first"]
    )

    palm_tracking = record["right_palm_command_tracking"]
    assert palm_tracking["commanded_tool_local_pose"] is None
    assert palm_tracking["translation_residual_norm_m"] is None
    assert palm_tracking["rotation_residual_rad"] is None
    assert "zero norm" in palm_tracking["command_error"]

    hand_tracking = record["right_hand_command_tracking"]
    assert hand_tracking["commanded_20d_rad"] == pytest.approx(np.zeros(20))
    assert hand_tracking["measured_20d_rad"] is None
    assert hand_tracking["measured_minus_commanded_20d_rad"] is None
    assert hand_tracking["maximum_abs_residual_rad"] is None
    assert "expected 20 right hand_joint_ids" in hand_tracking[
        "measurement_error"
    ]
    assert record["validity"]["pass"] is False
    json.dumps(record, allow_nan=False)


def test_telemetry_gate_rejects_step_hand_id_and_finger_query_errors() -> None:
    valid = _valid_telemetry_record()
    assert acceptance._telemetry_validity_gate(valid)["pass"] is True

    mismatched_step = copy.deepcopy(valid)
    mismatched_step["right_palm_command_tracking"]["policy_action_step"] = 15
    mismatch_gate = acceptance._telemetry_validity_gate(mismatched_step)
    assert mismatch_gate["pass"] is False
    assert "right_palm_command_tracking:policy_step_mismatch" in mismatch_gate[
        "reasons"
    ]

    missing_ids = copy.deepcopy(valid)
    missing_ids["right_hand_command_tracking"]["hand_joint_ids"] = list(range(19))
    ids_gate = acceptance._telemetry_validity_gate(missing_ids)
    assert ids_gate["pass"] is False
    assert "right_hand_command_tracking:invalid_hand_joint_ids" in ids_gate[
        "reasons"
    ]

    query_error = copy.deepcopy(valid)
    query_error["right_fingers"]["finger3"]["query_valid"] = False
    query_error["right_fingers"]["finger3"]["query_error"] = "buffer_saturated"
    query_gate = acceptance._telemetry_validity_gate(query_error)
    assert query_gate["pass"] is False
    assert "right_fingers.finger3:query_invalid" in query_gate["reasons"]
    assert "right_fingers.finger3:query_error" in query_gate["reasons"]

    invalid_wrench = copy.deepcopy(valid)
    invalid_wrench["right_wrench_sample"] = {
        "right_active": True,
        "pass": False,
    }
    wrench_gate = acceptance._telemetry_validity_gate(invalid_wrench)
    assert wrench_gate["pass"] is False
    assert "right_wrench_sample:invalid_active_gate" in wrench_gate["reasons"]

    invalid_table = copy.deepcopy(valid)
    invalid_table["table_safety"] = {"pass": False}
    table_gate = acceptance._telemetry_validity_gate(invalid_table)
    assert table_gate["pass"] is False
    assert "table_safety:unsafe_or_invalid" in table_gate["reasons"]

    premature_owner = copy.deepcopy(valid)
    premature_owner["right_semantic_owner"] = True
    owner_gate = acceptance._telemetry_validity_gate(premature_owner)
    assert owner_gate["pass"] is False
    assert "automatic_arrival_gate:ownership_before_open" in owner_gate[
        "reasons"
    ]

    malformed_arrival = copy.deepcopy(valid)
    malformed_arrival["automatic_arrival_gate"] = {
        "gate": "acquire",
        "open": False,
    }
    arrival_gate = acceptance._telemetry_validity_gate(malformed_arrival)
    assert arrival_gate["pass"] is False
    assert any(
        reason.startswith("automatic_arrival_gate.target_pose_wxyz")
        for reason in arrival_gate["reasons"]
    )

    spoofed_open = copy.deepcopy(valid)
    spoofed_open["automatic_acquire_gate_open"] = True
    spoofed_open["automatic_arrival_gate"]["open"] = True
    spoofed_open_gate = acceptance._telemetry_validity_gate(spoofed_open)
    assert spoofed_open_gate["pass"] is False
    assert "automatic_arrival_gate:gate_not_acquire" in spoofed_open_gate[
        "reasons"
    ]

    proven_open = copy.deepcopy(valid)
    proven_open["automatic_acquire_gate_open"] = True
    proven_open["right_semantic_owner"] = True
    proven_open["automatic_arrival_gate"] = {
        "gate": "acquire",
        "open": True,
        "target_pose_wxyz": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "actual_pose_wxyz": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "position_error_m": 0.0,
        "rotation_error_rad": 0.0,
        "position_tolerance_m": 0.01,
        "rotation_tolerance_rad": np.deg2rad(10.0),
        "within_tolerance": True,
        "streak": 4,
        "required_streak": 4,
        "arrived_step": 16,
        "sample_env_step": 16,
        "start_step": 13,
        "deadline_step": 42,
        "timed_out": False,
    }
    proven_gate = acceptance._telemetry_validity_gate(proven_open)
    assert proven_gate["pass"] is True

    stale_or_future = copy.deepcopy(proven_open)
    stale_or_future["automatic_arrival_gate"]["sample_env_step"] = 18
    future_gate = acceptance._telemetry_validity_gate(stale_or_future)
    assert future_gate["pass"] is False
    assert "automatic_arrival_gate:sample_from_future" in future_gate["reasons"]

    impossible_proof = copy.deepcopy(proven_open)
    impossible_proof["automatic_arrival_gate"].update(
        {
            "position_error_m": 171.0,
            "rotation_error_rad": 3.0,
            "streak": 5,
            "start_step": 100,
            "deadline_step": 10,
        }
    )
    impossible_gate = acceptance._telemetry_validity_gate(impossible_proof)
    assert impossible_gate["pass"] is False
    assert {
        "automatic_arrival_gate:position_outside_tolerance",
        "automatic_arrival_gate:rotation_outside_tolerance",
        "automatic_arrival_gate:streak_not_exact",
        "automatic_arrival_gate:deadline_before_start",
        "automatic_arrival_gate:arrival_outside_deadline",
    }.issubset(impossible_gate["reasons"])

    held_ik = copy.deepcopy(valid)
    held_ik["right_arm_control"]["ik_status"] = "haorui_hold"
    held_gate = acceptance._telemetry_validity_gate(held_ik)
    assert held_gate["pass"] is False
    assert "right_arm_control:ik_not_solved" in held_gate["reasons"]

    oversized_joint_step = copy.deepcopy(valid)
    oversized_joint_step["right_arm_control"][
        "maximum_final_command_step_rad"
    ] = 0.031
    joint_gate = acceptance._telemetry_validity_gate(oversized_joint_step)
    assert joint_gate["pass"] is False
    assert "right_arm_control:joint_step_cap_exceeded" in joint_gate["reasons"]


def test_close_watchdog_is_cancelled_after_normal_close(tmp_path: Path) -> None:
    class Env:
        close_called = False

        def close(self) -> None:
            self.close_called = True

    env = Env()
    summary_path = tmp_path / "summary.json"
    summary: dict[str, object] = {"validation_pass": False}
    exit_codes: list[int] = []

    acceptance._close_environment_with_watchdog(
        env,
        summary_path=summary_path,
        summary=summary,
        planned_exit_code=2,
        timeout_seconds=1.0,
        exit_fn=exit_codes.append,
        log_stream=io.StringIO(),
    )

    assert env.close_called is True
    assert exit_codes == []
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["shutdown_watchdog"] == {
        "close_attempted": True,
        "close_timed_out": False,
        "enabled": True,
        "forced_exit_code": None,
        "normal_close_completed": True,
        "planned_exit_code": 2,
        "timeout_seconds": 1.0,
    }


def test_close_watchdog_timeout_persists_failure_and_uses_forced_exit_code(
    tmp_path: Path,
) -> None:
    timed_out = threading.Event()
    exit_codes: list[int] = []
    log_stream = io.StringIO()

    def fake_exit(code: int) -> None:
        exit_codes.append(code)
        timed_out.set()

    summary_path = tmp_path / "summary.json"
    summary: dict[str, object] = {"validation_pass": True}
    shutdown_state: dict[str, object] = {
        "close_attempted": True,
        "normal_close_completed": False,
        "close_timed_out": False,
        "forced_exit_code": None,
    }
    summary["shutdown_watchdog"] = shutdown_state
    _cancel, timed_out_event, thread = acceptance._start_close_watchdog(
        0.01,
        0,
        summary_path=summary_path,
        summary=summary,
        shutdown_state=shutdown_state,
        exit_fn=fake_exit,
        log_stream=log_stream,
    )

    assert timed_out.wait(timeout=1.0)
    assert timed_out_event is not None and timed_out_event.is_set()
    assert thread is not None
    thread.join(timeout=1.0)
    assert exit_codes == [70]
    timeout_record = log_stream.getvalue()
    assert "AS019_SHUTDOWN_WATCHDOG_TIMEOUT=" in timeout_record
    assert '"planned_exit_code": 0' in timeout_record
    assert '"forced_exit_code": 70' in timeout_record
    assert '"summary_state": "close_timed_out"' in timeout_record
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["validation_pass_before_close_timeout"] is True
    assert persisted["validation_pass"] is False
    assert persisted["shutdown_terminal_reason"] == "close_timeout"
    assert persisted["terminal_reason"] == "close_timeout"
    assert persisted["shutdown_watchdog"]["normal_close_completed"] is False
    assert persisted["shutdown_watchdog"]["close_timed_out"] is True
    assert persisted["shutdown_watchdog"]["forced_exit_code"] == 70


def test_close_watchdog_timeout_cannot_be_overwritten_by_late_close(
    tmp_path: Path,
) -> None:
    forced_exit = threading.Event()
    exit_codes: list[int] = []

    def fake_exit(code: int) -> None:
        exit_codes.append(code)
        forced_exit.set()

    class Env:
        def close(self) -> None:
            assert forced_exit.wait(timeout=1.0)

    summary_path = tmp_path / "summary.json"
    summary: dict[str, object] = {"validation_pass": True}
    acceptance._close_environment_with_watchdog(
        Env(),
        summary_path=summary_path,
        summary=summary,
        planned_exit_code=0,
        timeout_seconds=0.01,
        exit_fn=fake_exit,
        log_stream=io.StringIO(),
    )

    assert exit_codes == [70]
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["validation_pass"] is False
    assert persisted["shutdown_terminal_reason"] == "close_timeout"
    assert persisted["terminal_reason"] == "close_timeout"
    assert persisted["shutdown_watchdog"]["close_timed_out"] is True
    assert persisted["shutdown_watchdog"]["normal_close_completed"] is False


def test_close_exception_propagates_after_summary_is_persisted(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"

    class CloseFailure(RuntimeError):
        pass

    class Env:
        preclose_summary: dict[str, object] | None = None

        def close(self) -> None:
            assert summary_path.exists()
            self.preclose_summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
            raise CloseFailure("close failed")

    env = Env()
    summary: dict[str, object] = {"validation_pass": True}
    with pytest.raises(CloseFailure, match="close failed"):
        acceptance._close_environment_with_watchdog(
            env,
            summary_path=summary_path,
            summary=summary,
            planned_exit_code=2,
            timeout_seconds=None,
        )

    assert env.preclose_summary is not None
    assert env.preclose_summary["shutdown_watchdog"] == {
        "close_attempted": True,
        "close_timed_out": False,
        "enabled": False,
        "forced_exit_code": None,
        "normal_close_completed": False,
        "planned_exit_code": 2,
        "timeout_seconds": None,
    }
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    shutdown = persisted["shutdown_watchdog"]
    assert persisted["validation_pass_before_close_exception"] is True
    assert persisted["validation_pass"] is False
    assert persisted["shutdown_terminal_reason"] == "close_exception"
    assert persisted["terminal_reason"] == "close_exception"
    assert shutdown["normal_close_completed"] is False
    assert "CloseFailure('close failed')" in shutdown["close_exception"]


def test_stage0_exit_requires_exactly_one_later_confirmation_step() -> None:
    exit_step, complete = acceptance._stage0_confirmation_status(
        stage=0,
        control_step=119,
        stage0_exit_step=None,
    )
    assert (exit_step, complete) == (None, False)

    exit_step, complete = acceptance._stage0_confirmation_status(
        stage=1,
        control_step=120,
        stage0_exit_step=exit_step,
    )
    assert (exit_step, complete) == (120, False)

    exit_step, complete = acceptance._stage0_confirmation_status(
        stage=1,
        control_step=121,
        stage0_exit_step=exit_step,
    )
    assert (exit_step, complete) == (120, True)


def test_stage0_runner_requests_no_camera_observations() -> None:
    source = Path(acceptance.__file__).read_text(encoding="utf-8")
    assert "image_obs=args.stage0_canary_seed is None" in source
    assert source.index("content_fingerprint = _validation_content_fingerprint()") < (
        source.index("env = TaskConfig().get_environment(")
    )
    assert source.index("automatic_policy_gate = _ambient_automatic_policy_gate()") < (
        source.index("env = TaskConfig().get_environment(")
    )
    assert source.index("arm_ik_environment_gate = _required_arm_ik_environment_gate()") < (
        source.index("env = TaskConfig().get_environment(")
    )
    assert "seed=environment_construction_seed" in source
    assert "reasons.append(\"stage0_every_step\")" in source
    assert "audit_table_safety(\"positive\")" in source
    assert "reset_table_clearance = _reset_table_clearance_gate(" in source
    assert "initial_table_clearance = _reset_table_clearance_gate(" in source
    assert 'initial_table_clearance["tool_table_supported"]' not in source
    assert '"table_safety": reset_table_clearance' in source
    assert '"initial_table_safety": initial_table_clearance' in source
    assert 'summary["terminal_reason"] = "close_timeout"' in source
    assert 'summary["terminal_reason"] = "close_exception"' in source
    assert "_as019_metrics_from_reset_info(reset_info)" in source
    assert "raw._table_safety_metrics" not in source
    assert "raw._metrics" not in source
    assert source.count("return exit_code") == 1
