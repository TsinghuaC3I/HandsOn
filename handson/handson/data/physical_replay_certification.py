"""Shared integrity contract for fresh-process PhysX replay certification."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from handson.data.recording_origin_profile import (
    is_abc_style_collection_task,
)
from handson.runtime.gpu import (
    is_canonical_cuda_device,
    is_physx_recording_device_allowed,
)
from handson.sim.isaaclab.as019_screwdriver_handover_env import (
    AS019_HIDDEN_STATE_FIELDS,
    TASK_NAME as AS019_TASK_NAME,
)
from handson.sim.isaaclab.b320_bc017_contract import (
    BC017_OBSERVATION_DTYPE,
    BC017_PROPRIO_FLATTEN_ORDER,
    HIDDEN_OBSERVATION_FIELDS,
    PUBLIC_OBSERVATION_GROUPS,
    TASK_NAME as BC017_TASK_NAME,
)

from handson.sim.isaaclab.xy2_contract import (
    TASK_NAME as XY2_TASK_NAME, RECORDING_LAYOUTS as XY2_RECORDING_LAYOUTS,
    RECORDING_ATOLS as XY2_RECORDING_ATOLS,
)


from handson.sim.isaaclab.xy4_contract import (
    TASK_NAME as XY4_TASK_NAME, HIDDEN_FIELDS as XY4_HIDDEN_FIELDS,
)

REPORT_NAME = "physical_replay_certification.json"
REPORT_FORMAT = "handson.physical_replay_certification"
REPORT_VERSION = 4
PHYSICAL_REPLAY_OUTCOME_VERSION = 1
PHYSICAL_REPLAY_VALID_OUTCOME_CODE = "physical_replay_valid_v1"
PHYSICAL_REPLAY_INVALID_OUTCOME_CODE = "physical_replay_invalid_v1"
PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE = "physical_replay_review_required_v1"
PHYSICAL_REPLAY_REVIEW_VALID = "valid"
PHYSICAL_REPLAY_REVIEW_INVALID = "invalid"
PHYSICAL_REPLAY_REVIEW_REQUIRED = "review_required"
CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD = "recorded_contact_signature_read_only_overlay"
CONTACT_SIGNATURE_OVERLAY_FORMAT = "handson.physx_contact_signature_read_only_overlay"
CONTACT_SIGNATURE_OVERLAY_VERSION = 1
CONTACT_SIGNATURE_OVERLAY_ALGORITHM = "rigid_contact_tensor_exact_zero_padding_v1"
CLEAN_COMMIT_SOURCE_IDENTITY = "clean_commit_match"
DIRTY_CONTENT_SOURCE_IDENTITY = "dirty_content_match"
CLEAN_COMMIT_REVIEWER_UPGRADE_SOURCE_IDENTITY = "clean_recording_reviewer_upgrade"
DIRTY_SNAPSHOT_REVIEWER_UPGRADE_SOURCE_IDENTITY = (
    "dirty_recording_exact_snapshot_reviewer_upgrade"
)
PORTABLE_SOURCE_IDENTITIES = frozenset(
    {
        CLEAN_COMMIT_SOURCE_IDENTITY,
        CLEAN_COMMIT_REVIEWER_UPGRADE_SOURCE_IDENTITY,
        DIRTY_SNAPSHOT_REVIEWER_UPGRADE_SOURCE_IDENTITY,
    }
)
SOURCE_REVIEWER_UPGRADE_ALLOWED_PATHS = frozenset(
    {
        "scripts/certify_physx_replay.py",
        "scripts/tests/test_certify_physx_replay.py",
        "scripts/deferred_physical_replay_certification_worker.py",
        "scripts/tests/test_deferred_physical_replay_certification_worker.py",
        "handson/handson/data/physical_replay_certification.py",
        "handson/tests/test_physical_replay_certification.py",
        "handson/handson/data/verified_publication_v3.py",
        "handson/tests/test_verified_publication_v3.py",
        "handson/handson/sim/isaaclab/physx_ground_truth.py",
        "handson/tests/test_physx_ground_truth.py",
    }
)
CERTIFICATION_MODE = "task_success_from_fresh_seeded_physx_origin"
FRESH_SCENE_INITIALIZATION_MODE = (
    "fresh_process_seeded_reset_identical_stabilization_no_extra_t0_restore"
)
INITIALIZATION_MODE = FRESH_SCENE_INITIALIZATION_MODE
RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT = (
    "physx_state_t_plus_raw_command_t_plus_recorded_robot_substep_drive_targets_t_"
    "produces_next_physx_state_t"
)
RECORDED_DRIVE_TARGET_INPUT_MODE = (
    "raw_command_plus_recorded_robot_substep_drive_targets"
)
LEGACY_RAW_ACTION_TRANSITION_CONTRACT = (
    "physx_state_t_plus_action_t_produces_next_physx_state_t"
)
LEGACY_RAW_ACTION_INPUT_MODE = "exact_raw_environment_action_executed"
DETACHED_HAND_ACTION_TRANSITION_CONTRACT = (
    "physx_state_t_plus_raw_command_t_plus_deterministic_kinematic_wuji_"
    "substep_path_t_produces_next_physx_state_t"
)
DETACHED_HAND_ACTION_INPUT_MODE = (
    "raw_command_through_recorded_simplified_wuji_substep_controller"
)
FRESH_PROCESS_CERTIFICATION_MODE = "fresh_process"
FRESH_PROCESS_CERTIFICATION_MODE_WHITELIST = frozenset(
    {FRESH_PROCESS_CERTIFICATION_MODE}
)
INPUT_IDENTITY_ALGORITHM = "sha256_tree_v1"
FRESH_PROCESS_ACCEPTANCE_NAME = "fresh_process_acceptance.json"
FRESH_PROCESS_ACCEPTANCE_FORMAT = "handson.fresh_process_acceptance"
FRESH_PROCESS_ACCEPTANCE_VERSION = 1
DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_FORMAT = (
    "handson.deferred_physical_replay_acceptance"
)
DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_VERSION = 1
DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE = "deferred_post_collection"
PERSISTENT_RECORDER_LIFECYCLE = "persistent_process"
FRESH_RECORDER_LIFECYCLE = "fresh_process_per_attempt"
ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE = "ordered_collection_session"
SESSION_REPLAY_EXECUTION_MODE = "single_process_ordered_episode_replay"
SESSION_EVENT_REPLAY_PASSED_STATUS = "replayed_verified_session_prefix"
SESSION_PUBLICATION_COMMIT_FORMAT = "handson.physical_replay_session_publication_commit"
SESSION_PUBLICATION_COMMIT_VERSION = 1
SESSION_PUBLICATION_COMMIT_DIRECTORY = ".physical_replay_session_commits"
FRESH_PROCESS_REPORT_DIRECTORY = "physical_replay_certification_runs"
DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS = 3
INITIALIZATION_RESTORE_COUNTS_SOURCE = (
    "recorded_and_reexecuted_stabilization_final_boundary_write"
)
_TRAJECTORY_INPUT_MODE_BY_TRANSITION_CONTRACT = {
    RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT: RECORDED_DRIVE_TARGET_INPUT_MODE,
    LEGACY_RAW_ACTION_TRANSITION_CONTRACT: LEGACY_RAW_ACTION_INPUT_MODE,
    DETACHED_HAND_ACTION_TRANSITION_CONTRACT: DETACHED_HAND_ACTION_INPUT_MODE,
}
_RECORDED_DRIVE_TARGET_COUNT_FIELDS = (
    "recorded_drive_target_substeps_expected",
    "recorded_drive_target_substeps_executed",
    "recorded_drive_target_writes_expected",
    "recorded_drive_target_writes_executed",
)
INITIALIZATION_RESTORE_COUNT_FIELDS = frozenset(
    {
        "robot_joints",
        "rigid_bodies",
        "articulation_roots",
        "articulation_dofs",
        "deformable_nodes",
    }
)
ACTIVE_PHYSICS_LAYOUT_FORMAT = "handson.active_physics_layout"
ACTIVE_PHYSICS_LAYOUT_VERSION = 1
PHYSX_GROUND_TRUTH_SCHEMA_NAME = "handson.physx_ground_truth"
PHYSX_GROUND_TRUTH_SCHEMA_VERSION = 6
_BC017_TASK_STATE_FIELD_NAMES = tuple(BC017_PROPRIO_FLATTEN_ORDER)
_BC017_TASK_STATE_LAYOUTS = {
    **{name: tuple(layout) for name, layout in PUBLIC_OBSERVATION_GROUPS},
    "bc017_hidden_state": tuple(HIDDEN_OBSERVATION_FIELDS),
}
_BC017_TASK_STATE_REPLAY_ATOLS = {
    name: (0.0 if name == "bc017_hidden_state" else 1.0e-6)
    for name in _BC017_TASK_STATE_FIELD_NAMES
}
_AS019_TASK_STATE_FIELD_NAMES = ("as019_hidden_state",)
_AS019_TASK_STATE_LAYOUTS = {
    "as019_hidden_state": tuple(AS019_HIDDEN_STATE_FIELDS),
}
_AS019_TASK_STATE_REPLAY_ATOLS = {"as019_hidden_state": 1.0e-6}
_ACTIVE_PHYSICS_LAYOUT_CATEGORIES = (
    "rigid_body",
    "articulation_root",
    "articulation_dof",
    "deformable_node",
)
_CONTACT_SIGNATURE_FIELDS = frozenset(
    {
        "physx_contact_signature_valid",
        "physx_contact_signature_pair_count",
        "physx_contact_signature_pair_path_indices",
        "physx_contact_signature_pair_counts",
        "physx_contact_signature_pair_metrics",
    }
)
_SUBSTEP_CONTACT_SIGNATURE_FIELDS = frozenset(
    name.replace("physx_", "physx_substep_", 1) for name in _CONTACT_SIGNATURE_FIELDS
)
_REQUIRED_CONTACT_SIGNATURE_FIELDS = (
    _CONTACT_SIGNATURE_FIELDS
    | _SUBSTEP_CONTACT_SIGNATURE_FIELDS
    | {"physx_substep_count"}
)
_PERSISTENT_COLLECTION_SEED_SOURCES = frozenset(
    {"explicit_cli_seed", "os_cryptographic_random_31bit"}
)
_PERSISTENT_EPISODE_SEED_DERIVATION = (
    "collection_run_seed_plus_reset_attempt_index_mod_2pow31"
)
_DEFERRED_CANONICAL_INITIALIZATION_HISTORY_FIELDS = (
    "stabilization_velocity_clamp",
    "initialization_per_physics_step_velocity_writes",
    "stabilization_velocity_clamp_scope",
    "physics_dt",
    "timeout_seconds",
    "physics_steps",
    "initial_settle_steps",
    "stabilization_velocity_clamp_steps",
    "stabilization_velocity_clamp_write",
    "stability_metric",
    "task_name",
    "stability_sample_steps",
    "stability_samples",
    "stable_steps_required",
    "stable_steps_observed",
    "gate_task_velocity_stable",
    "gate_task_static_pose_stable",
    "gate_robot_velocity_stable",
    "gate_static_pose_fallback_samples",
    "gate_pose_samples",
    "full_frame_pose_gate",
    "reset_pose_drift_checked",
    "reset_translation_drift_limit",
    "reset_rotation_drift_limit",
    "reset_coordinate_drift_limit",
    "reset_max_translation_drift_m",
    "reset_max_rotation_drift_rad",
    "reset_max_coordinate_drift",
    "reset_state_coverage",
    "active_physics_layout",
    "gate_active_physics_layout",
    "final_active_physics_layout",
    "active_physics_layout_gate_final_exact_match",
    "reset_drift_sample_count",
    "reset_drift_expected_sample_count",
    "reset_drift_all_physics_frames_sampled",
    "authored_reset_snapshot_source",
    "authored_reset_snapshot_serial",
    "require_valid_contact_report",
    "contact_penetration_limit",
    "initialization_contact_summary",
    "gate_contact_summary",
    "initial_geometry_gate_passed",
    "scene_rebuilt_before_reset",
    "scene_generation",
    "final_reset_pose_drift_checked",
    "final_reset_translation_drift_limit",
    "final_reset_rotation_drift_limit",
    "final_reset_coordinate_drift_limit",
    "final_reset_max_translation_drift_m",
    "final_reset_max_rotation_drift_rad",
    "final_reset_max_coordinate_drift",
    "final_reset_state_coverage",
    "gate_final_translation_delta_limit",
    "gate_final_rotation_delta_limit",
    "gate_final_max_task_translation_delta_m",
    "gate_final_max_task_rotation_delta_rad",
    "final_contact_penetration_limit",
    "final_contact_summary",
    "final_boundary_physics_gate_passed",
    "post_settle_velocity_zeroing",
    "post_settle_sleeping",
    "velocity_zeroing_passes",
    "first_state_write",
    "post_settle_state_write",
    "post_settle_sleep",
    "boundary_state_write_mode",
    "contact_warm_start_preserved",
    "final_boundary_state_write",
    "final_boundary_sleep",
    "final_boundary_restore_sync",
    "final_boundary_physics_steps_after_write",
    "default_joint_state_restores",
    "robot_hold_target_source",
    "post_settle_robot_hold_target_source",
    "final_default_joint_state_restore",
    "final_robot_hold_target_source",
    "hold_target_sync",
    "final_hold_target_sync",
    "final_boundary_task_and_joint_velocity_exact_zero",
    "state_writes",
    "trajectory_per_frame_state_writes",
    "per_frame_state_writes",
)
_NATURAL_SETTLE_ADAPTIVE_INITIALIZATION_HISTORY_FIELDS = frozenset(
    {
        "physics_steps",
        "stability_samples",
        "hold_target_sync",
        "reset_drift_sample_count",
        "reset_drift_expected_sample_count",
        "initialization_contact_summary",
    }
)
SLEEPING_ABC_ORIGIN_TASKS = frozenset(
    {
        # BC011 intentionally keeps its cylindrical source collider.  Its pen
        # is velocity-zeroed after falling to the table, then put into PhysX's
        # wakeable sleep state until the first physical hand contact.
        "t20_pass_and_orient_pen",
        # HS005 has the same audited cylindrical-contact failure mode.  Its
        # bottle and brush are allowed to enter wakeable sleep only at the
        # final pre-action boundary, after natural settling and velocity
        # zeroing; physical hand contact wakes them during collection.
        "t37_scrub_bottle",
    }
)
_GATE_CLAMP_ORIGIN_TASKS = frozenset(
    {
        "t20_pass_and_orient_pen",
    }
)

# These values are reductions derived from float32 PhysX public readback.  A
# fresh process can reproduce the authored reset and later replay the recorded
# physical trace while these diagnostic maxima differ within the source-native
# pose tolerances.  Only the derived summaries use this comparison: discrete
# initialization history and the underlying replay state keep their existing
# exact/per-field contracts.
_FRESH_PROCESS_DERIVED_RESET_SCALAR_TOLERANCES = {
    "reset_max_translation_drift_m": 1.0e-5,
    "reset_max_rotation_drift_rad": 1.0e-4,
    "reset_max_coordinate_drift": 1.0e-5,
    "final_reset_max_translation_drift_m": 1.0e-5,
    "final_reset_max_rotation_drift_rad": 1.0e-4,
    "final_reset_max_coordinate_drift": 1.0e-5,
    "gate_final_max_task_translation_delta_m": 1.0e-5,
    "gate_final_max_task_rotation_delta_rad": 1.0e-4,
}
_FRESH_PROCESS_CONTACT_PENETRATION_TOLERANCE_M = 1.0e-6
_FRESH_PROCESS_CONTACT_IMPULSE_TOLERANCE_NS = 1.0e-5


def _fresh_process_contact_summary_matches(
    recorded_value: Any,
    replayed_value: Any,
    *,
    adaptive_sample_counts: bool,
) -> bool:
    """Compare reset-contact topology exactly and physical metrics narrowly."""

    if not isinstance(recorded_value, Mapping) or not isinstance(
        replayed_value,
        Mapping,
    ):
        return False
    recorded = dict(recorded_value)
    replayed = dict(replayed_value)
    if adaptive_sample_counts:
        for adaptive_field in (
            "sample_count",
            "expected_sample_count",
            "deepest_sample_index",
        ):
            recorded.pop(adaptive_field, None)
            replayed.pop(adaptive_field, None)

    def pop_finite_number(value: dict[str, Any], name: str) -> float | None:
        raw = value.pop(name, None)
        if (
            type(raw) not in {int, float}
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
        ):
            return None
        return float(raw)

    recorded_penetration = pop_finite_number(
        recorded,
        "maximum_penetration_depth_m",
    )
    replayed_penetration = pop_finite_number(
        replayed,
        "maximum_penetration_depth_m",
    )
    if (
        recorded_penetration is None
        or replayed_penetration is None
        or abs(recorded_penetration - replayed_penetration)
        > _FRESH_PROCESS_CONTACT_PENETRATION_TOLERANCE_M
    ):
        return False

    recorded_signature = recorded.pop("canonical_contact_signature", None)
    replayed_signature = replayed.pop("canonical_contact_signature", None)
    if recorded_signature is None or replayed_signature is None:
        if recorded_signature is not None or replayed_signature is not None:
            return False
    else:
        if not isinstance(recorded_signature, Mapping) or not isinstance(
            replayed_signature,
            Mapping,
        ):
            return False
        recorded_signature = dict(recorded_signature)
        replayed_signature = dict(replayed_signature)
        recorded_metrics = recorded_signature.pop("pair_metrics", None)
        replayed_metrics = replayed_signature.pop("pair_metrics", None)
        if not _json_values_equal(recorded_signature, replayed_signature):
            return False
        if not isinstance(recorded_metrics, list) or not isinstance(
            replayed_metrics,
            list,
        ) or len(recorded_metrics) != len(replayed_metrics):
            return False
        for recorded_row, replayed_row in zip(
            recorded_metrics,
            replayed_metrics,
            strict=True,
        ):
            if (
                not isinstance(recorded_row, list)
                or not isinstance(replayed_row, list)
                or len(recorded_row) != 3
                or len(replayed_row) != 3
            ):
                return False
            for column, (recorded_metric, replayed_metric) in enumerate(
                zip(recorded_row, replayed_row, strict=True)
            ):
                if (
                    type(recorded_metric) not in {int, float}
                    or type(replayed_metric) not in {int, float}
                    or isinstance(recorded_metric, bool)
                    or isinstance(replayed_metric, bool)
                ):
                    return False
                recorded_float = float(recorded_metric)
                replayed_float = float(replayed_metric)
                if not math.isfinite(recorded_float) or not math.isfinite(
                    replayed_float
                ):
                    return False
                tolerance = (
                    _FRESH_PROCESS_CONTACT_PENETRATION_TOLERANCE_M
                    if column == 0
                    else _FRESH_PROCESS_CONTACT_IMPULSE_TOLERANCE_NS
                )
                if abs(recorded_float - replayed_float) > tolerance:
                    return False

    return _json_values_equal(recorded, replayed)


def _is_abc_collection_task(task_name: str) -> bool:
    normalized = str(task_name).strip().lower()
    return bool(
        normalized[:1] in {"a", "b", "c"}
        or normalized.startswith("ycb")
        or normalized == "eval_gen03_orient_bottle_upright"
    )


def _is_natural_settle_then_zero_history(
    history: Mapping[str, Any],
) -> bool:
    """Return whether initialization follows its approved settle/zero profile."""

    task_name = str(history.get("task_name") or "").strip().lower()
    is_abc = is_abc_style_collection_task(task_name)
    uses_awake_abc_contract = (
        is_abc and task_name not in SLEEPING_ABC_ORIGIN_TASKS
    )
    uses_gate_clamp = task_name in _GATE_CLAMP_ORIGIN_TASKS
    clamp_contract_matches = (
        history.get("stabilization_velocity_clamp") is True
        and history.get("initialization_per_physics_step_velocity_writes")
        is True
        and history.get("stabilization_velocity_clamp_scope") == "gate_only"
        and type(history.get("physics_steps")) is int
        and type(history.get("initial_settle_steps")) is int
        and type(history.get("stabilization_velocity_clamp_steps")) is int
        and history.get("stabilization_velocity_clamp_steps")
        == history.get("physics_steps") - history.get("initial_settle_steps")
        and history.get("stabilization_velocity_clamp_steps") > 0
        and isinstance(
            history.get("stabilization_velocity_clamp_write"),
            Mapping,
        )
        if uses_gate_clamp
        else (
            history.get("stabilization_velocity_clamp") is False
            and history.get("initialization_per_physics_step_velocity_writes")
            is False
            and history.get("stabilization_velocity_clamp_scope") == "none"
            and history.get("stabilization_velocity_clamp_steps") == 0
            and history.get("stabilization_velocity_clamp_write") is None
        )
    )
    sleep_contract_matches = (
        history.get("post_settle_sleeping") is False
        and history.get("post_settle_sleep") is None
        and history.get("final_boundary_sleep") is None
        and history.get("contact_warm_start_preserved") is True
        if uses_awake_abc_contract
        else (
            history.get("post_settle_sleeping") is True
            and history.get("contact_warm_start_preserved") is False
        )
    )

    return bool(
        clamp_contract_matches
        and history.get("post_settle_velocity_zeroing") is True
        and history.get("boundary_state_write_mode") == "velocity_only"
        and sleep_contract_matches
    )


def _natural_settle_adaptive_history_errors(
    history: Mapping[str, Any],
    *,
    label: str,
) -> list[str]:
    """Validate adaptive counters without requiring identical settle time."""

    errors: list[str] = []
    integer_fields = (
        "physics_steps",
        "initial_settle_steps",
        "stability_sample_steps",
        "stability_samples",
        "stable_steps_required",
        "stable_steps_observed",
    )
    values: dict[str, int] = {}
    for field in integer_fields:
        value = history.get(field)
        if type(value) is not int:
            errors.append(f"{label}.{field} must be an integer")
        else:
            values[field] = int(value)
    if len(values) != len(integer_fields):
        return errors

    physics_steps = values["physics_steps"]
    initial_settle_steps = values["initial_settle_steps"]
    sample_steps = values["stability_sample_steps"]
    stability_samples = values["stability_samples"]
    stable_steps_required = values["stable_steps_required"]
    stable_steps_observed = values["stable_steps_observed"]
    gate_steps = physics_steps - initial_settle_steps
    if initial_settle_steps <= 0:
        errors.append(f"{label}.initial_settle_steps must be positive")
    if sample_steps <= 0:
        errors.append(f"{label}.stability_sample_steps must be positive")
    task_name = str(history.get("task_name") or "").strip().lower()
    if is_abc_style_collection_task(task_name) and sample_steps != 1:
        errors.append(f"{label}.stability_sample_steps 对 ABC 必须为 1")
    if stability_samples <= 0:
        errors.append(f"{label}.stability_samples must be positive")
    if stable_steps_required <= 0:
        errors.append(f"{label}.stable_steps_required must be positive")
    if gate_steps <= 0:
        errors.append(f"{label}.physics_steps must include strict-gate steps")
    if (
        sample_steps > 0
        and stability_samples > 0
        and (gate_steps != sample_steps * stability_samples)
    ):
        errors.append(
            f"{label}.physics_steps/stability_samples contradict the "
            "stability sampling interval"
        )
    if stable_steps_observed < stable_steps_required:
        errors.append(
            f"{label}.stable_steps_observed did not cover the required window"
        )
    if gate_steps > 0 and stable_steps_observed > gate_steps:
        errors.append(f"{label}.stable_steps_observed exceeds executed gate steps")

    raw_physics_dt = history.get("physics_dt")
    raw_timeout_seconds = history.get("timeout_seconds")
    if raw_physics_dt is not None or raw_timeout_seconds is not None:
        try:
            if type(raw_physics_dt) is bool or type(raw_timeout_seconds) is bool:
                raise TypeError
            physics_dt = float(raw_physics_dt)
            timeout_seconds = float(raw_timeout_seconds)
        except (TypeError, ValueError):
            physics_dt = timeout_seconds = float("nan")
        if (
            not math.isfinite(physics_dt)
            or physics_dt <= 0.0
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0.0
        ):
            errors.append(f"{label} has invalid physics_dt/timeout_seconds")
        elif gate_steps > math.ceil(timeout_seconds / physics_dt):
            errors.append(f"{label}.physics_steps exceeds its timeout")

    hold_target_sync = history.get("hold_target_sync")
    if not isinstance(hold_target_sync, Mapping):
        errors.append(f"{label}.hold_target_sync must be an object")
    else:
        passes = hold_target_sync.get("passes")
        if type(passes) is not int or passes != physics_steps + 2:
            errors.append(
                f"{label}.hold_target_sync.passes must equal physics_steps + 2"
            )
    return errors


def _active_physics_layout_errors(
    layout: Any,
    *,
    coverage: Mapping[str, Any] | None,
    label: str,
) -> list[str]:
    """Validate the exact active task-physics entity layout and bitmaps."""

    errors: list[str] = []
    expected_fields = {
        "format",
        "version",
        *_ACTIVE_PHYSICS_LAYOUT_CATEGORIES,
    }
    if not isinstance(layout, Mapping) or set(layout) != expected_fields:
        return [f"{label} is missing the exact active physics layout fields"]
    if layout.get("format") != ACTIVE_PHYSICS_LAYOUT_FORMAT:
        errors.append(f"{label}.format is invalid")
    if type(layout.get("version")) is not int or (
        layout.get("version") != ACTIVE_PHYSICS_LAYOUT_VERSION
    ):
        errors.append(f"{label}.version must be {ACTIVE_PHYSICS_LAYOUT_VERSION}")

    active_counts: dict[str, int] = {}
    for category in _ACTIVE_PHYSICS_LAYOUT_CATEGORIES:
        entry = layout.get(category)
        entry_label = f"{label}.{category}"
        if not isinstance(entry, Mapping) or set(entry) != {"names", "valid"}:
            errors.append(f"{entry_label} must contain names and valid")
            continue
        names = entry.get("names")
        valid = entry.get("valid")
        if (
            not isinstance(names, list)
            or any(not isinstance(name, str) or not name for name in names)
            or names != list(dict.fromkeys(names))
        ):
            errors.append(f"{entry_label}.names must be unique nonempty text")
            continue
        if (
            not isinstance(valid, list)
            or len(valid) != len(names)
            or any(type(value) is not int or value not in {0, 1} for value in valid)
        ):
            errors.append(f"{entry_label}.valid must be an exact binary bitmap")
            continue
        active_counts[category] = sum(valid)
        if isinstance(coverage, Mapping) and coverage.get(category) != sum(valid):
            errors.append(
                f"{entry_label}.valid count differs from reset_state_coverage"
            )

    if len(active_counts) == len(_ACTIVE_PHYSICS_LAYOUT_CATEGORIES) and (
        sum(active_counts.values()) <= 0
    ):
        errors.append(f"{label} covers no active task physics entity")
    return errors


def _active_physics_layout_counts(layout: Any) -> dict[str, int] | None:
    if not isinstance(layout, Mapping):
        return None
    counts: dict[str, int] = {}
    for category in _ACTIVE_PHYSICS_LAYOUT_CATEGORIES:
        entry = layout.get(category)
        valid = entry.get("valid") if isinstance(entry, Mapping) else None
        if not isinstance(valid, list) or any(
            type(value) is not int or value not in {0, 1} for value in valid
        ):
            return None
        counts[category] = sum(valid)
    return counts


def _restore_counts_cover_active_layout_errors(
    restore_counts: Any,
    *,
    layout: Any,
    label: str,
) -> list[str]:
    active_counts = _active_physics_layout_counts(layout)
    if not isinstance(restore_counts, Mapping) or active_counts is None:
        return [f"{label} 无法绑定 active physics layout"]
    expected = {
        "rigid_bodies": active_counts["rigid_body"],
        "articulation_roots": active_counts["articulation_root"],
        "articulation_dofs": active_counts["articulation_dof"],
        "deformable_nodes": active_counts["deformable_node"],
    }
    errors = []
    for field, count in expected.items():
        if restore_counts.get(field) != count:
            errors.append(f"{label}.{field} 必须精确覆盖 active layout")
    if type(restore_counts.get("robot_joints")) is not int or (
        restore_counts.get("robot_joints") <= 0
    ):
        errors.append(f"{label}.robot_joints 必须为正整数")
    return errors


def _physical_origin_geometry_history_errors(
    history: Mapping[str, Any],
    *,
    label: str,
) -> list[str]:
    """Fail closed unless reset drift and solved contacts passed strict gates."""

    errors: list[str] = []
    if history.get("reset_pose_drift_checked") is not True:
        errors.append(f"{label}.reset_pose_drift_checked must be true")
    if history.get("require_valid_contact_report") is not True:
        errors.append(f"{label}.require_valid_contact_report must be true")
    if history.get("initial_geometry_gate_passed") is not True:
        errors.append(f"{label}.initial_geometry_gate_passed must be true")

    limits = (
        ("reset_translation_drift_limit", 0.005),
        ("reset_rotation_drift_limit", 0.05235987755982989),
        ("contact_penetration_limit", 0.0005),
    )
    parsed_limits: dict[str, float] = {}
    for field, maximum in limits:
        value = history.get(field)
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            or float(value) > maximum
        ):
            errors.append(f"{label}.{field} must be finite in (0, {maximum}]")
        else:
            parsed_limits[field] = float(value)

    coordinate_limit = history.get("reset_coordinate_drift_limit")
    expected_coordinate_limit = min(
        parsed_limits.get("reset_translation_drift_limit", float("inf")),
        parsed_limits.get("reset_rotation_drift_limit", float("inf")),
    )
    if (
        type(coordinate_limit) not in {int, float}
        or isinstance(coordinate_limit, bool)
        or not math.isfinite(float(coordinate_limit))
        or float(coordinate_limit) <= 0.0
        or float(coordinate_limit) > expected_coordinate_limit
    ):
        errors.append(
            f"{label}.reset_coordinate_drift_limit must be no greater than "
            "both pose limits"
        )

    for field, limit_field in (
        ("reset_max_translation_drift_m", "reset_translation_drift_limit"),
        ("reset_max_rotation_drift_rad", "reset_rotation_drift_limit"),
    ):
        value = history.get(field)
        limit = parsed_limits.get(limit_field)
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or (limit is not None and float(value) > limit)
        ):
            errors.append(f"{label}.{field} exceeds its declared limit")

    coordinate_drift = history.get("reset_max_coordinate_drift")
    if (
        type(coordinate_drift) not in {int, float}
        or isinstance(coordinate_drift, bool)
        or not math.isfinite(float(coordinate_drift))
        or float(coordinate_drift) < 0.0
        or (
            type(coordinate_limit) in {int, float}
            and not isinstance(coordinate_limit, bool)
            and float(coordinate_drift) > float(coordinate_limit)
        )
    ):
        errors.append(f"{label}.reset_max_coordinate_drift exceeds its declared limit")

    expected_coverage_fields = {
        "rigid_body",
        "articulation_root",
        "robot_root",
        "tcp",
        "articulation_dof",
        "robot_joint",
        "hand_command",
        "deformable_node",
    }
    coverage = history.get("reset_state_coverage")
    if not isinstance(coverage, Mapping) or set(coverage) != (expected_coverage_fields):
        errors.append(f"{label}.reset_state_coverage is incomplete")
    elif any(type(value) is not int or value < 0 for value in coverage.values()):
        errors.append(f"{label}.reset_state_coverage counts are invalid")
    else:
        if coverage.get("robot_joint", 0) <= 0:
            errors.append(f"{label}.reset_state_coverage covers no robot state")
        if (
            sum(coverage.get(field, 0) for field in _ACTIVE_PHYSICS_LAYOUT_CATEGORIES)
            <= 0
        ):
            errors.append(f"{label}.reset_state_coverage covers no task physics state")
    errors.extend(
        _active_physics_layout_errors(
            history.get("active_physics_layout"),
            coverage=coverage if isinstance(coverage, Mapping) else None,
            label=f"{label}.active_physics_layout",
        )
    )
    gate_layout = history.get("gate_active_physics_layout")
    final_layout = history.get("final_active_physics_layout")
    errors.extend(
        _active_physics_layout_errors(
            gate_layout,
            coverage=coverage if isinstance(coverage, Mapping) else None,
            label=f"{label}.gate_active_physics_layout",
        )
    )
    errors.extend(
        _active_physics_layout_errors(
            final_layout,
            coverage=coverage if isinstance(coverage, Mapping) else None,
            label=f"{label}.final_active_physics_layout",
        )
    )
    if (
        history.get("active_physics_layout_gate_final_exact_match") is not True
        or not _json_values_equal(history.get("active_physics_layout"), gate_layout)
        or not _json_values_equal(gate_layout, final_layout)
    ):
        errors.append(f"{label} gate/final active physics layouts must exactly match")

    scene_rebuilt = history.get("scene_rebuilt_before_reset")
    scene_generation = history.get("scene_generation")
    if type(scene_rebuilt) is not bool:
        errors.append(f"{label}.scene_rebuilt_before_reset must be boolean")
    if type(scene_generation) is not int or scene_generation < 0:
        errors.append(f"{label}.scene_generation must be a nonnegative integer")
    elif scene_rebuilt is False and scene_generation != 0:
        errors.append(f"{label}.scene_generation must be zero without a scene rebuild")
    elif scene_rebuilt is True and scene_generation <= 0:
        errors.append(
            f"{label}.scene_generation must be positive after a scene rebuild"
        )

    drift_sample_count = history.get("reset_drift_sample_count")
    drift_expected_count = history.get("reset_drift_expected_sample_count")
    physics_steps = history.get("physics_steps")
    if (
        type(drift_sample_count) is not int
        or type(drift_expected_count) is not int
        or type(physics_steps) is not int
        or drift_sample_count != physics_steps + 1
        or drift_expected_count != physics_steps + 1
        or history.get("reset_drift_all_physics_frames_sampled") is not True
    ):
        errors.append(f"{label}.reset drift frame coverage is invalid")

    if (
        history.get("authored_reset_snapshot_source")
        != "environment_after_authored_writes_before_first_physics_step"
    ):
        errors.append(f"{label}.authored_reset_snapshot_source is invalid")
    authored_serial = history.get("authored_reset_snapshot_serial")
    if type(authored_serial) is not int or authored_serial <= 0:
        errors.append(f"{label}.authored_reset_snapshot_serial is invalid")

    initialization_contact = history.get("initialization_contact_summary")
    if not isinstance(initialization_contact, Mapping):
        errors.append(f"{label}.initialization_contact_summary must be an object")
    else:
        sample_count = initialization_contact.get("sample_count")
        expected_sample_count = initialization_contact.get("expected_sample_count")
        sources = initialization_contact.get("capture_sources")
        max_point_count = initialization_contact.get("maximum_contact_point_count")
        penetration = initialization_contact.get("maximum_penetration_depth_m")
        penetration_limit = parsed_limits.get("contact_penetration_limit")
        if initialization_contact.get("valid") is not True:
            errors.append(f"{label}.initialization_contact_summary.valid must be true")
        if (
            type(sample_count) is not int
            or sample_count <= 0
            or type(expected_sample_count) is not int
            or expected_sample_count != sample_count
            or type(history.get("physics_steps")) is not int
            or expected_sample_count != history.get("physics_steps") + 1
            or initialization_contact.get("all_physics_frames_sampled") is not True
        ):
            errors.append(
                f"{label}.initialization_contact_summary frame coverage is invalid"
            )
        if initialization_contact.get("invalid_sample_count") != 0:
            errors.append(f"{label}.initialization_contact_summary has invalid samples")
        if (
            not isinstance(sources, list)
            or not sources
            or any(
                type(source) is not int or source not in {1, 2} for source in sources
            )
        ):
            errors.append(
                f"{label}.initialization_contact_summary.capture_sources is invalid"
            )
        if type(max_point_count) is not int or max_point_count < 0:
            errors.append(
                f"{label}.initialization_contact_summary point count is invalid"
            )
        if (
            type(penetration) not in {int, float}
            or isinstance(penetration, bool)
            or not math.isfinite(float(penetration))
            or float(penetration) < 0.0
            or (
                penetration_limit is not None and float(penetration) > penetration_limit
            )
        ):
            errors.append(
                f"{label}.initialization_contact_summary penetration exceeds its limit"
            )
        if initialization_contact.get("error") is not None:
            errors.append(f"{label}.initialization_contact_summary.error must be null")

    contact = history.get("gate_contact_summary")
    if not isinstance(contact, Mapping):
        errors.append(f"{label}.gate_contact_summary must be an object")
    else:
        if contact.get("valid") is not True:
            errors.append(f"{label}.gate_contact_summary.valid must be true")
        if type(contact.get("capture_source")) is not int or contact.get(
            "capture_source"
        ) not in {1, 2}:
            errors.append(f"{label}.gate_contact_summary.capture_source is invalid")
        point_count = contact.get("contact_point_count")
        if type(point_count) is not int or point_count < 0:
            errors.append(
                f"{label}.gate_contact_summary.contact_point_count is invalid"
            )
        penetration = contact.get("maximum_penetration_depth_m")
        penetration_limit = parsed_limits.get("contact_penetration_limit")
        if (
            type(penetration) not in {int, float}
            or isinstance(penetration, bool)
            or not math.isfinite(float(penetration))
            or float(penetration) < 0.0
            or (
                penetration_limit is not None and float(penetration) > penetration_limit
            )
        ):
            errors.append(f"{label}.gate_contact_summary penetration exceeds its limit")
        if contact.get("error") is not None:
            errors.append(f"{label}.gate_contact_summary.error must be null")
        if (
            isinstance(initialization_contact, Mapping)
            and type(initialization_contact.get("maximum_penetration_depth_m"))
            in {int, float}
            and type(penetration) in {int, float}
            and not isinstance(penetration, bool)
            and float(initialization_contact["maximum_penetration_depth_m"])
            < float(penetration)
        ):
            errors.append(
                f"{label}.initialization_contact_summary must include the "
                "gate penetration sample"
            )

    if history.get("final_reset_pose_drift_checked") is not True:
        errors.append(f"{label}.final_reset_pose_drift_checked must be true")
    final_limits = (
        ("final_reset_translation_drift_limit", 0.005),
        ("final_reset_rotation_drift_limit", 0.05235987755982989),
        ("final_contact_penetration_limit", 0.0005),
    )
    parsed_final_limits: dict[str, float] = {}
    for field, maximum in final_limits:
        value = history.get(field)
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            or float(value) > maximum
        ):
            errors.append(f"{label}.{field} exceeds canonical maximum")
        else:
            parsed_final_limits[field] = float(value)
    final_coordinate_limit = history.get("final_reset_coordinate_drift_limit")
    if (
        type(final_coordinate_limit) not in {int, float}
        or isinstance(final_coordinate_limit, bool)
        or not math.isfinite(float(final_coordinate_limit))
        or float(final_coordinate_limit) <= 0.0
        or float(final_coordinate_limit)
        > min(
            parsed_final_limits.get(
                "final_reset_translation_drift_limit", float("inf")
            ),
            parsed_final_limits.get("final_reset_rotation_drift_limit", float("inf")),
        )
    ):
        errors.append(f"{label}.final_reset_coordinate_drift_limit invalid")
    for field, limit_field in (
        (
            "final_reset_max_translation_drift_m",
            "final_reset_translation_drift_limit",
        ),
        (
            "final_reset_max_rotation_drift_rad",
            "final_reset_rotation_drift_limit",
        ),
        (
            "final_reset_max_coordinate_drift",
            "final_reset_coordinate_drift_limit",
        ),
    ):
        value = history.get(field)
        limit = history.get(limit_field)
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or type(limit) not in {int, float}
            or isinstance(limit, bool)
            or float(value) > float(limit)
        ):
            errors.append(f"{label}.{field} exceeds its final boundary limit")
    final_coverage = history.get("final_reset_state_coverage")
    if not _json_values_equal(final_coverage, coverage):
        errors.append(f"{label}.final_reset_state_coverage must equal reset coverage")
    gate_final_limits = (
        ("gate_final_translation_delta_limit", 0.005),
        ("gate_final_rotation_delta_limit", 0.05235987755982989),
    )
    for field, maximum in gate_final_limits:
        value = history.get(field)
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            or float(value) > maximum
        ):
            errors.append(f"{label}.{field} invalid")
    for field, limit_field in (
        (
            "gate_final_max_task_translation_delta_m",
            "gate_final_translation_delta_limit",
        ),
        (
            "gate_final_max_task_rotation_delta_rad",
            "gate_final_rotation_delta_limit",
        ),
    ):
        value = history.get(field)
        limit = history.get(limit_field)
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or type(limit) not in {int, float}
            or isinstance(limit, bool)
            or float(value) > float(limit)
        ):
            errors.append(f"{label}.{field} exceeds its gate-final limit")
    final_contact = history.get("final_contact_summary")
    if not isinstance(final_contact, Mapping):
        errors.append(f"{label}.final_contact_summary must be an object")
    else:
        final_penetration = final_contact.get("maximum_penetration_depth_m")
        final_limit = parsed_final_limits.get("final_contact_penetration_limit")
        if (
            final_contact.get("valid") is not True
            or type(final_contact.get("capture_source")) is not int
            or final_contact.get("capture_source") not in {1, 2}
            or type(final_contact.get("contact_point_count")) is not int
            or final_contact.get("contact_point_count") < 0
            or type(final_penetration) not in {int, float}
            or isinstance(final_penetration, bool)
            or not math.isfinite(float(final_penetration))
            or float(final_penetration) < 0.0
            or (final_limit is not None and float(final_penetration) > final_limit)
            or final_contact.get("error") is not None
        ):
            errors.append(f"{label}.final_contact_summary invalid")
        if not _json_values_equal(final_contact, contact):
            errors.append(f"{label}.final_contact_summary must equal gate contact")
    if history.get("final_boundary_physics_gate_passed") is not True:
        errors.append(f"{label}.final_boundary_physics_gate_passed must be true")
    errors.extend(
        _restore_counts_cover_active_layout_errors(
            history.get("final_boundary_state_write"),
            layout=history.get("active_physics_layout"),
            label=f"{label}.final_boundary_state_write",
        )
    )
    return errors


def _stable_recording_origin_contract_errors(
    recording: Mapping[str, Any],
) -> list[str]:
    """Validate the complete physical initialization evidence for every mode."""

    label = "recording.stable_recording_origin"
    stable_origin = recording.get("stable_recording_origin")
    if not isinstance(stable_origin, Mapping):
        return [f"{label} 必须为 object"]

    errors: list[str] = []
    for field in _DEFERRED_CANONICAL_INITIALIZATION_HISTORY_FIELDS:
        if field not in stable_origin:
            errors.append(f"{label}.{field} 缺失")

    clamp_enabled = stable_origin.get("stabilization_velocity_clamp")
    per_step_writes = stable_origin.get(
        "initialization_per_physics_step_velocity_writes"
    )
    clamp_scope = stable_origin.get("stabilization_velocity_clamp_scope")
    task_name = str(stable_origin.get("task_name") or "").strip().lower()
    gate_clamp_requested = bool(
        task_name in _GATE_CLAMP_ORIGIN_TASKS
        and clamp_enabled is True
        and per_step_writes is True
        and clamp_scope == "gate_only"
    )
    natural_settle_then_zero = bool(
        clamp_enabled is False and per_step_writes is False and clamp_scope == "none"
    )
    accepted_settle_then_zero = bool(
        natural_settle_then_zero or gate_clamp_requested
    )
    legacy_all_step_clamp_requested = bool(
        clamp_enabled is True
        and per_step_writes is True
        and clamp_scope == "all_initialization_steps"
    )
    if legacy_all_step_clamp_requested:
        errors.append(
            f"{label} 的 all_initialization_steps 每帧速度钳制仅可诊断，"
            "禁止进入 v4 发布认证"
        )
    if not accepted_settle_then_zero:
        errors.append(
            f"{label} 的 stabilization_velocity_clamp / "
            "initialization_per_physics_step_velocity_writes / "
            "stabilization_velocity_clamp_scope 初始化策略无效"
        )

    physics_steps = stable_origin.get("physics_steps")
    initial_settle_steps = stable_origin.get("initial_settle_steps")
    clamp_steps = stable_origin.get("stabilization_velocity_clamp_steps")
    if type(physics_steps) is not int or physics_steps <= 0:
        errors.append(f"{label}.physics_steps 必须为正整数")
    if type(initial_settle_steps) is not int or (
        initial_settle_steps < (1 if accepted_settle_then_zero else 0)
        or (type(physics_steps) is int and initial_settle_steps > physics_steps)
    ):
        errors.append(f"{label}.initial_settle_steps 无效")
    expected_clamp_steps = (
        physics_steps - initial_settle_steps
        if gate_clamp_requested
        and type(physics_steps) is int
        and type(initial_settle_steps) is int
        else 0
    )
    if type(clamp_steps) is not int or (
        type(expected_clamp_steps) is int and clamp_steps != expected_clamp_steps
    ):
        errors.append(f"{label}.stabilization_velocity_clamp_steps 与初始化策略不一致")

    clamp_write = stable_origin.get("stabilization_velocity_clamp_write")
    if accepted_settle_then_zero:
        if natural_settle_then_zero and clamp_write is not None:
            errors.append(f"{label}.stabilization_velocity_clamp_write 必须为 null")
        if gate_clamp_requested and not isinstance(clamp_write, Mapping):
            errors.append(
                f"{label}.stabilization_velocity_clamp_write 必须记录 gate_only 写入"
            )
        if stable_origin.get("post_settle_velocity_zeroing") is not True:
            errors.append(f"{label}.post_settle_velocity_zeroing 必须为 true")
        is_abc = is_abc_style_collection_task(task_name)
        uses_awake_abc_contract = (
            is_abc and task_name not in SLEEPING_ABC_ORIGIN_TASKS
        )
        if stable_origin.get("boundary_state_write_mode") != "velocity_only":
            errors.append(f"{label}.boundary_state_write_mode 必须为 velocity_only")
        if uses_awake_abc_contract:
            if stable_origin.get("post_settle_sleeping") is not False:
                errors.append(f"{label}.post_settle_sleeping 对 ABC 必须为 false")
            if stable_origin.get("post_settle_sleep") is not None:
                errors.append(f"{label}.post_settle_sleep 对 ABC 必须为 null")
            if stable_origin.get("final_boundary_sleep") is not None:
                errors.append(f"{label}.final_boundary_sleep 对 ABC 必须为 null")
            if stable_origin.get("contact_warm_start_preserved") is not True:
                errors.append(
                    f"{label}.contact_warm_start_preserved 对 ABC 必须为 true"
                )
        else:
            if stable_origin.get("post_settle_sleeping") is not True:
                errors.append(f"{label}.post_settle_sleeping 对该任务必须为 true")
            if stable_origin.get("contact_warm_start_preserved") is not False:
                errors.append(
                    f"{label}.contact_warm_start_preserved 对该任务必须为 false"
                )
        errors.extend(
            _natural_settle_adaptive_history_errors(
                stable_origin,
                label=label,
            )
        )
    elif clamp_write is not None:
        errors.append(
            f"{label}.stabilization_velocity_clamp_write 对不可认证历史必须为 null"
        )

    source_rebuild = recording.get("source_scene_rebuild_per_reset")
    source_generation = recording.get("source_scene_generation")
    if type(source_rebuild) is not bool:
        errors.append("recording.source_scene_rebuild_per_reset 必须为布尔值")
    if type(source_generation) is not int or source_generation < 0:
        errors.append("recording.source_scene_generation 必须为非负整数")
    if type(source_rebuild) is bool and type(source_generation) is int:
        ordinal = recording.get("episode_reset_ordinal")
        expected_rebuilt = bool(source_rebuild and type(ordinal) is int and ordinal > 0)
        if stable_origin.get("scene_rebuilt_before_reset") is not expected_rebuilt:
            errors.append(f"{label}.scene_rebuilt_before_reset 与 manifest 不一致")
        if stable_origin.get("scene_generation") != source_generation:
            errors.append(f"{label}.scene_generation 与 manifest 不一致")

    errors.extend(
        _physical_origin_geometry_history_errors(
            stable_origin,
            label=label,
        )
    )
    return errors


def _initialization_history_field_matches(
    field: str,
    recorded_value: Any,
    replayed_value: Any,
    *,
    natural_settle: bool,
) -> bool:
    """Compare fixed initialization calls while allowing adaptive duration."""

    if field in {
        "initialization_contact_summary",
        "gate_contact_summary",
        "final_contact_summary",
    }:
        return _fresh_process_contact_summary_matches(
            recorded_value,
            replayed_value,
            adaptive_sample_counts=(
                natural_settle and field == "initialization_contact_summary"
            ),
        )

    absolute_tolerance = _FRESH_PROCESS_DERIVED_RESET_SCALAR_TOLERANCES.get(field)
    if absolute_tolerance is not None:
        if (
            type(recorded_value) not in {int, float}
            or type(replayed_value) not in {int, float}
            or isinstance(recorded_value, bool)
            or isinstance(replayed_value, bool)
        ):
            return False
        recorded_float = float(recorded_value)
        replayed_float = float(replayed_value)
        if not math.isfinite(recorded_float) or not math.isfinite(replayed_float):
            return False
        tolerance = max(
            absolute_tolerance,
            8.0 * math.ulp(recorded_float),
            8.0 * math.ulp(replayed_float),
        )
        return abs(recorded_float - replayed_float) <= tolerance

    if not natural_settle or (
        field not in _NATURAL_SETTLE_ADAPTIVE_INITIALIZATION_HISTORY_FIELDS
    ):
        return _json_values_equal(recorded_value, replayed_value)
    if field in {
        "physics_steps",
        "stability_samples",
        "reset_drift_sample_count",
        "reset_drift_expected_sample_count",
    }:
        return type(recorded_value) is int and type(replayed_value) is int
    if field == "hold_target_sync":
        if not isinstance(recorded_value, Mapping) or not isinstance(
            replayed_value,
            Mapping,
        ):
            return False
        recorded_fixed = dict(recorded_value)
        replayed_fixed = dict(replayed_value)
        recorded_fixed.pop("passes", None)
        replayed_fixed.pop("passes", None)
        return _json_values_equal(recorded_fixed, replayed_fixed)
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_values_equal(left: Any, right: Any) -> bool:
    """Compare JSON evidence without Python's bool/int equality coercion."""

    try:
        options = {
            "allow_nan": False,
            "ensure_ascii": True,
            "separators": (",", ":"),
            "sort_keys": True,
        }
        return json.dumps(left, **options) == json.dumps(right, **options)
    except (TypeError, ValueError):
        return False


def _canonical_initialization_history_projection(
    history: Mapping[str, Any],
) -> dict[str, Any]:
    """Project recorded reset evidence exactly as the certifier reports it."""

    return {
        field: history[field]
        for field in _DEFERRED_CANONICAL_INITIALIZATION_HISTORY_FIELDS
        if field in history
    }


def _replay_tree_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise OSError(f"Replay tree is missing or invalid: {path}")

    def raise_walk_error(error: OSError) -> None:
        raise error

    files: list[tuple[str, Path]] = []
    for root, directory_names, file_names in os.walk(
        path,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        root_path = Path(root)
        for name in tuple(directory_names):
            child = root_path / name
            if child.is_symlink():
                raise OSError(f"Replay tree contains a symbolic link: {child}")
        directory_names[:] = sorted(directory_names)
        for name in sorted(file_names):
            child = root_path / name
            if child.is_symlink() or not child.is_file():
                raise OSError(f"Replay tree contains a non-regular file: {child}")
            files.append((child.relative_to(path).as_posix(), child))
    files.sort(key=lambda item: item[0])

    digest = hashlib.sha256()
    total_bytes = 0
    for relative_path, file_path in files:
        relative_bytes = relative_path.encode("utf-8")
        size_before = file_path.stat().st_size
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(size_before.to_bytes(8, "big"))
        bytes_read = 0
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                bytes_read += len(chunk)
        if bytes_read != size_before or file_path.stat().st_size != size_before:
            raise OSError(f"Replay file changed while hashing: {file_path}")
        total_bytes += size_before
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "bytes": total_bytes,
    }


def capture_certified_inputs(demo_dir: str | Path) -> dict[str, Any]:
    """Hash every input that determines one replay certification result."""

    demo_path = Path(demo_dir).expanduser().resolve()
    tree = _replay_tree_identity(demo_path / "replay.zarr")
    return {
        "algorithm": INPUT_IDENTITY_ALGORITHM,
        "replay_manifest_sha256": _sha256_file(demo_path / "replay_manifest.json"),
        "recording_schema_sha256": _sha256_file(demo_path / "recording_schema.json"),
        "replay_zarr_sha256": tree["sha256"],
        "replay_zarr_file_count": tree["file_count"],
        "replay_zarr_bytes": tree["bytes"],
    }


def _contains_physical_trace(demo_path: Path) -> bool:
    try:
        collection = json.loads(
            (demo_path / "collection_meta.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        collection = None
    if isinstance(collection, Mapping) and (
        str(collection.get("sim_backend") or "").lower() == "isaacsim"
        or collection.get("engine_code") == "E02"
        or collection.get("is_stable_v2_data") is True
        or collection.get("is_physx_v3_data") is True
    ):
        return True
    if any(
        (demo_path / name).exists()
        for name in (
            "DATASET_PHYSX_V3.json",
            "DATASET_STABLE_V2.json",
            REPORT_NAME,
            FRESH_PROCESS_ACCEPTANCE_NAME,
            FRESH_PROCESS_REPORT_DIRECTORY,
        )
    ):
        return True
    schema_path = demo_path / "recording_schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        schema = None
    if isinstance(schema, Mapping):
        environment_schema = schema.get("environment_schema")
        if isinstance(environment_schema, Mapping) and isinstance(
            environment_schema.get("physx"), Mapping
        ):
            return True
        fields = schema.get("fields")
        if isinstance(fields, Mapping) and any(
            str(name).startswith(("physx_", "next_physx_")) for name in fields
        ):
            return True
    data_path = demo_path / "replay.zarr" / "data"
    try:
        return any(
            child.name.startswith(("physx_", "next_physx_"))
            for child in data_path.iterdir()
        )
    except OSError:
        return False


def _physical_replay_recording_requirement(
    demo_path: Path,
) -> tuple[Mapping[str, Any] | None, dict[str, Any] | None]:
    """Resolve certification requirement without trusting a missing manifest."""

    physical_trace = _contains_physical_trace(demo_path)
    manifest_path = demo_path / "replay_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        if not physical_trace:
            return None, None
        return None, {
            "code": "physical_replay_manifest_invalid",
            "detail": f"PhysX 轨迹缺少合法 replay_manifest.json：{exc}",
        }
    if not isinstance(manifest, Mapping):
        if not physical_trace:
            return None, None
        return None, {
            "code": "physical_replay_manifest_invalid",
            "detail": "PhysX 轨迹的 replay_manifest.json 必须为 object。",
        }
    recording = manifest.get("recording")
    if isinstance(recording, Mapping):
        declared_runs = recording.get("physical_replay_certification_runs_required")
        manifest_physical_hint = bool(
            str(recording.get("sim_backend") or "").lower() == "isaacsim"
            or recording.get("record_physx_ground_truth") is True
            or recording.get("physical_replay_state_recorded") is True
            or recording.get("physical_transition_contract")
            in _TRAJECTORY_INPUT_MODE_BY_TRANSITION_CONTRACT
            or recording.get("trajectory_input_mode")
            in _TRAJECTORY_INPUT_MODE_BY_TRANSITION_CONTRACT.values()
            or type(declared_runs) is int
            or recording.get("fresh_recorder_process_per_episode_required") is True
            or recording.get("fresh_recorder_process_per_attempt_required") is True
            or recording.get("publish_only_after_fresh_certification") is True
        )
        physical_trace = physical_trace or manifest_physical_hint
    if isinstance(recording, Mapping) and (
        recording.get("physical_replay_certification_required") is True
    ):
        return recording, None
    if not physical_trace:
        return None, None
    return None, {
        "code": "physical_replay_certification_required_invalid",
        "detail": (
            "已检测到 PhysX 轨迹，但 manifest 未声明 "
            "physical_replay_certification_required=true，暂不上传。"
        ),
    }


def physical_replay_certification_required(demo_dir: str | Path) -> bool:
    """Return whether an episode is physical or claims a physical contract."""

    recording, blocker = _physical_replay_recording_requirement(Path(demo_dir))
    return recording is not None or blocker is not None


def _required_certification_runs(recording: Mapping[str, Any]) -> int:
    declared = recording.get("physical_replay_certification_runs_required")
    if type(declared) is int and (
        declared >= DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS
    ):
        return declared
    return DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS


def _trajectory_contract_errors(
    recording: Mapping[str, Any],
    report: Mapping[str, Any] | None = None,
    *,
    task_name: str | None = None,
) -> tuple[list[str], str | None, str | None]:
    """Validate and, for reports, bind the declared replay input contract."""

    errors: list[str] = []
    transition_contract = recording.get("physical_transition_contract")
    trajectory_input_mode = recording.get("trajectory_input_mode")
    expected_input_mode = _TRAJECTORY_INPUT_MODE_BY_TRANSITION_CONTRACT.get(
        transition_contract
    )
    if expected_input_mode is None:
        errors.append("recording.physical_transition_contract 不受支持")
    if trajectory_input_mode != expected_input_mode:
        errors.append(
            "recording.trajectory_input_mode 与 physical_transition_contract 不匹配"
        )

    if recording.get("physical_replay_goal") != CERTIFICATION_MODE:
        errors.append(f"recording.physical_replay_goal 必须为 {CERTIFICATION_MODE}")
    errors.extend(_stable_recording_origin_contract_errors(recording))

    certification_process_mode = recording.get("physical_replay_certification_mode")
    if certification_process_mode == FRESH_PROCESS_CERTIFICATION_MODE:
        for field in (
            "fresh_recorder_process_per_episode_required",
            "fresh_recorder_process_per_attempt_required",
            "fresh_scene_replay_required",
            "fresh_certifier_process_per_run_required",
            "publish_only_after_fresh_certification",
        ):
            if recording.get(field) is not True:
                errors.append(f"recording.{field} 必须为 true")
        if recording.get("recorder_lifecycle") != FRESH_RECORDER_LIFECYCLE:
            errors.append(
                "recording.recorder_lifecycle 必须为 fresh_process_per_attempt"
            )
        if recording.get("source_process_reuse_enabled") is not False:
            errors.append("recording.source_process_reuse_enabled 必须显式为 false")
        recorder_pid = recording.get("recorder_pid")
        if type(recorder_pid) is not int or recorder_pid <= 0:
            errors.append("recording.recorder_pid 必须为正整数")
    elif certification_process_mode == DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE:
        for field in (
            "fresh_recorder_process_per_episode_required",
            "fresh_recorder_process_per_attempt_required",
        ):
            if recording.get(field) is not False:
                errors.append(f"recording.{field} 必须显式为 false")
        if recording.get("recorder_lifecycle") != PERSISTENT_RECORDER_LIFECYCLE:
            errors.append("recording.recorder_lifecycle 必须为 persistent_process")
        if recording.get("fresh_scene_replay_required") is not True:
            errors.append("recording.fresh_scene_replay_required 必须为 true")
        if recording.get("publish_only_after_fresh_certification") is not True:
            errors.append(
                "recording.publish_only_after_fresh_certification 必须为 true"
            )
        recorder_pid = recording.get("recorder_pid")
        if type(recorder_pid) is not int or recorder_pid <= 0:
            errors.append("recording.recorder_pid 必须为正整数")
        for field in (
            "source_process_reuse_enabled",
            "fresh_certifier_process_per_run_required",
        ):
            if recording.get(field) is not True:
                errors.append(f"recording.{field} 必须为 true")

        collection_seed = recording.get("collection_run_seed")
        episode_seed = recording.get("episode_reset_seed")
        episode_ordinal = recording.get("episode_reset_ordinal")
        if type(collection_seed) is not int or collection_seed < 0:
            errors.append("recording.collection_run_seed 必须为非负整数")
        if type(episode_seed) is not int or not (0 <= episode_seed < 2**31):
            errors.append("recording.episode_reset_seed 必须是 31-bit 非负整数")
        if type(episode_ordinal) is not int or episode_ordinal < 0:
            errors.append("recording.episode_reset_ordinal 必须为非负整数")
        if (
            recording.get("collection_seed_source")
            not in _PERSISTENT_COLLECTION_SEED_SOURCES
        ):
            errors.append("recording.collection_seed_source 不受支持")
        if (
            recording.get("episode_seed_derivation")
            != _PERSISTENT_EPISODE_SEED_DERIVATION
        ):
            errors.append("recording.episode_seed_derivation 不受支持")
        if (
            type(collection_seed) is int
            and collection_seed >= 0
            and type(episode_seed) is int
            and type(episode_ordinal) is int
            and episode_ordinal >= 0
            and episode_seed != (collection_seed + episode_ordinal) % (2**31)
        ):
            errors.append(
                "recording.episode_reset_seed 必须等于 "
                "(collection_run_seed + episode_reset_ordinal) mod 2**31"
            )

    else:
        errors.append(
            "物理 replay 认证模式必须为 fresh_process 或 deferred_post_collection"
        )
    if (
        trajectory_input_mode == LEGACY_RAW_ACTION_INPUT_MODE
        and task_name != "e01_fold_garment"
    ):
        errors.append("legacy raw-action replay 仅允许任务 e01_fold_garment")

    if report is not None:
        if report.get("physical_transition_contract") != transition_contract:
            errors.append("report.physical_transition_contract 与 manifest 不一致")
        if report.get("trajectory_input_mode") != trajectory_input_mode:
            errors.append("report.trajectory_input_mode 与 manifest 不一致")
        if report.get("certification_mode") != recording.get("physical_replay_goal"):
            errors.append("report.certification_mode 与 manifest goal 不一致")

    return errors, transition_contract, trajectory_input_mode


def _certification_mode_blocker(
    recording: Mapping[str, Any],
    *,
    task_name: str | None = None,
) -> dict[str, Any] | None:
    errors, _, _ = _trajectory_contract_errors(
        recording,
        task_name=task_name,
    )
    if errors:
        return {
            "code": "physical_replay_certification_mode_invalid",
            "detail": "PhysX replay 模式契约无效：" + "；".join(errors),
        }
    return None


def _manifest_task_name(demo_path: Path) -> str | None:
    try:
        manifest = json.loads(
            (demo_path / "replay_manifest.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, Mapping):
        return None
    task = manifest.get("task")
    if not isinstance(task, Mapping):
        return None
    name = task.get("name")
    return name if isinstance(name, str) and name else None


def _parse_process_timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_git_object_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value) is not None
    )


def _ordered_session_manifest_contract(
    demo_path: Path,
    recording: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Validate the sealed journal binding on one deferred demo."""

    errors: list[str] = []
    scope = recording.get("physical_replay_certification_scope")
    if scope != ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE:
        errors.append(
            "recording.physical_replay_certification_scope 必须为 "
            f"{ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE}"
        )
    if recording.get("physical_replay_session_history_pending") is not False:
        errors.append("recording.physical_replay_session_history_pending 必须为 false")

    history_id = recording.get("physical_replay_session_history_id")
    if not isinstance(history_id, str) or not history_id.strip():
        errors.append("recording.physical_replay_session_history_id 必须为非空字符串")
        history_id = ""
    else:
        history_id = history_id.strip()

    history_relative_path = recording.get(
        "physical_replay_session_history_root_relative_path"
    )
    if not isinstance(history_relative_path, str) or not (
        history_relative_path.strip()
    ):
        errors.append(
            "recording.physical_replay_session_history_root_relative_path "
            "必须为非空相对路径"
        )
        history_relative_path = ""
    else:
        history_relative_path = history_relative_path.strip()
        parsed_history_path = Path(history_relative_path)
        if parsed_history_path.is_absolute() or (".." in parsed_history_path.parts):
            errors.append(
                "recording.physical_replay_session_history_root_relative_path "
                "必须位于输出根目录内"
            )

    history_sha256 = recording.get("physical_replay_session_history_sha256")
    if not _is_sha256(history_sha256):
        errors.append(
            "recording.physical_replay_session_history_sha256 必须为小写 SHA-256"
        )

    event_count = recording.get("physical_replay_session_history_event_count")
    if type(event_count) is not int or event_count <= 0:
        errors.append(
            "recording.physical_replay_session_history_event_count 必须为正整数"
        )

    accepted_event_ordinal = recording.get(
        "physical_replay_session_history_accepted_event_ordinal"
    )
    if (
        type(accepted_event_ordinal) is not int
        or accepted_event_ordinal <= 0
        or type(event_count) is not int
        or accepted_event_ordinal >= event_count
    ):
        errors.append(
            "recording.physical_replay_session_history_accepted_event_ordinal "
            "必须指向 history 内的 accepted trajectory"
        )

    certified_through_event_ordinal = recording.get(
        "physical_replay_session_history_certified_through_event_ordinal"
    )
    if (
        type(certified_through_event_ordinal) is not int
        or type(accepted_event_ordinal) is not int
        or certified_through_event_ordinal < accepted_event_ordinal
        or type(event_count) is not int
        or certified_through_event_ordinal >= event_count
    ):
        errors.append(
            "recording.physical_replay_session_history_"
            "certified_through_event_ordinal 无效"
        )

    episode_index = recording.get("episode_index")
    if type(episode_index) is not int or episode_index <= 0:
        errors.append("recording.episode_index 必须为正整数")

    metadata_path = demo_path / "collection_meta.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        metadata = None
        errors.append(f"collection_meta.json 无法读取：{exc}")
    if not isinstance(metadata, Mapping):
        if metadata is not None:
            errors.append("collection_meta.json 必须为 JSON object")
        metadata = {}

    binding_fields = (
        "physical_replay_certification_scope",
        "physical_replay_session_history_pending",
        "physical_replay_session_history_id",
        "physical_replay_session_history_root_relative_path",
        "physical_replay_session_history_sha256",
        "physical_replay_session_history_event_count",
        "physical_replay_session_history_accepted_event_ordinal",
        "physical_replay_session_history_certified_through_event_ordinal",
    )
    for field in binding_fields:
        if metadata.get(field) != recording.get(field):
            errors.append(f"collection_meta.{field} 必须与 recording 一致")
    collection_session_id = metadata.get("collection_session_id")
    if not isinstance(collection_session_id, str) or (
        collection_session_id.strip() != history_id
    ):
        errors.append(
            "collection_meta.collection_session_id 必须与 session history id 一致"
        )

    return (
        {
            "scope": scope,
            "history_id": history_id,
            "history_root_relative_path": history_relative_path,
            "history_sha256": history_sha256,
            "event_count": event_count,
            "accepted_event_ordinal": accepted_event_ordinal,
            "certified_through_event_ordinal": (certified_through_event_ordinal),
            "episode_index": episode_index,
            "episode_reset_ordinal": recording.get("episode_reset_ordinal"),
            "episode_reset_seed": recording.get("episode_reset_seed"),
            "collection_run_seed": recording.get("collection_run_seed"),
            "collection_session_id": collection_session_id,
            "stable_recording_origin": recording.get("stable_recording_origin"),
        },
        errors,
    )


def _journal_event_replay_contract(
    events: list[Any],
    *,
    acceptance: Mapping[str, Any],
    session_contract: Mapping[str, Any],
    history_path: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Derive every replay-prefix counter from the finalized journal."""

    errors: list[str] = []
    accepted_ordinals = session_contract.get("accepted_event_ordinals")
    episode_index = acceptance.get("session_episode_index")
    target_accepted = session_contract.get("accepted_event_ordinal")
    if (
        not isinstance(accepted_ordinals, list)
        or type(episode_index) is not int
        or not (1 <= episode_index <= len(accepted_ordinals))
        or type(target_accepted) is not int
        or accepted_ordinals[episode_index - 1] != target_accepted
    ):
        return None, ["journal accepted event 顺序无法派生 replay counters"]
    target_reset = target_accepted - 1
    start = 0 if episode_index == 1 else accepted_ordinals[episode_index - 2] + 1
    if not (0 <= start <= target_reset < len(events)):
        return None, ["journal replay prefix 边界无效"]

    reset_attempts = 0
    rejected_resets = 0
    discarded_trajectories = 0
    discarded_actions = 0
    discarded_drive_substeps = 0
    cursor = start
    while cursor < target_reset:
        event = events[cursor]
        payload = event.get("payload") if isinstance(event, Mapping) else None
        if (
            not isinstance(event, Mapping)
            or event.get("kind") != "reset_attempt"
            or not isinstance(payload, Mapping)
        ):
            errors.append(f"journal event {cursor} 必须为 reset_attempt")
            break
        reset_attempts += 1
        accepted = payload.get("accepted")
        if accepted is False:
            rejected_resets += 1
            cursor += 1
            continue
        if accepted is not True or cursor + 1 >= target_reset:
            errors.append(
                f"journal event {cursor} accepted reset 缺少 discarded trajectory"
            )
            break
        trajectory_event = events[cursor + 1]
        trajectory_payload = (
            trajectory_event.get("payload")
            if isinstance(trajectory_event, Mapping)
            else None
        )
        if (
            not isinstance(trajectory_event, Mapping)
            or trajectory_event.get("kind") != "trajectory"
            or not isinstance(trajectory_payload, Mapping)
            or trajectory_payload.get("accepted") is not False
            or trajectory_payload.get("reset_ordinal") != payload.get("reset_ordinal")
        ):
            errors.append(f"journal event {cursor + 1} 不是匹配的 discarded trajectory")
            break
        step_count = trajectory_payload.get("step_count")
        if type(step_count) is not int or step_count < 0:
            errors.append(f"journal event {cursor + 1}.step_count 无效")
            break
        discarded_trajectories += 1
        discarded_actions += step_count
        replay_mode = trajectory_payload.get("replay_input_mode")
        if step_count == 0:
            if replay_mode != "none":
                errors.append(f"journal event {cursor + 1} 空轨迹 replay mode 无效")
        elif replay_mode == "recorded_articulated_drive_targets":
            try:
                from handson.data.physx_session_history import (
                    load_discarded_trajectory,
                )

                arrays = load_discarded_trajectory(
                    history_path,
                    cursor + 1,
                    require_finalized=True,
                )
                substeps = arrays.get("next_physx_substep_count")
                if (
                    substeps is None
                    or getattr(substeps, "shape", (0,))[0] != step_count
                ):
                    raise ValueError("next_physx_substep_count shape mismatch")
                discarded_drive_substeps += sum(
                    int(value) for value in substeps.reshape(-1)
                )
            except Exception as exc:
                errors.append(
                    f"journal event {cursor + 1} drive substeps 无法派生：{exc}"
                )
        elif replay_mode != "raw_environment_action":
            errors.append(f"journal event {cursor + 1} replay_input_mode 无效")
        cursor += 2
    if cursor != target_reset:
        errors.append("journal replay prefix 未精确终止于目标 reset")
    if errors:
        return None, errors
    return {
        "events_replayed": target_reset - start,
        "first_event_ordinal": start if start < target_reset else None,
        "last_event_ordinal": target_reset - 1 if start < target_reset else None,
        "target_accepted_reset_event_ordinal": target_reset,
        "target_accepted_trajectory_event_ordinal": target_accepted,
        "reset_attempts_replayed": reset_attempts,
        "rejected_reset_attempts_replayed": rejected_resets,
        "discarded_trajectories_replayed": discarded_trajectories,
        "discarded_actions_replayed": discarded_actions,
        "discarded_drive_substeps_replayed": discarded_drive_substeps,
        "previous_episode_reset_ordinal": (
            None
            if episode_index == 1
            else acceptance.get("session_episode_reset_ordinals", [])[episode_index - 2]
        ),
        "next_episode_reset_ordinal": session_contract.get("episode_reset_ordinal"),
    }, []


def _ordered_session_history_errors(
    demo_path: Path,
    acceptance: Mapping[str, Any],
    session_contract: dict[str, Any],
    *,
    task_name: str | None,
) -> list[str]:
    """Resolve and verify the manifest-bound journal, not only its report."""

    errors: list[str] = []
    relative_text = session_contract.get("history_root_relative_path")
    if not isinstance(relative_text, str) or not relative_text:
        return ["session history root-relative path 无效"]
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        return ["session history root-relative path 逃逸输出目录"]

    raw_bases = (demo_path.parent.parent, demo_path.parent)
    candidates: dict[Path, Path] = {}
    for raw_base in raw_bases:
        try:
            if raw_base.is_symlink():
                raise OSError("output-root candidate is a symlink")
            base = raw_base.resolve(strict=True)
        except OSError:
            continue
        unresolved = base / relative
        cursor = base
        has_symlink = False
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                has_symlink = True
                break
            if not cursor.exists():
                break
        if has_symlink:
            errors.append(f"session history 路径不得包含符号链接：{cursor}")
            continue
        if not unresolved.exists():
            continue
        try:
            resolved = unresolved.resolve(strict=True)
            resolved.relative_to(base)
        except (OSError, ValueError) as exc:
            errors.append(f"session history 路径无法安全解析：{exc}")
            continue
        if not resolved.is_dir():
            errors.append(f"session history 不是目录：{resolved}")
            continue
        candidates[resolved] = base
    if len(candidates) != 1:
        errors.append(
            "session history 必须在 pending/published 输出根目录中唯一解析，"
            f"实际为 {len(candidates)} 个"
        )
        return errors
    history_path = next(iter(candidates))

    try:
        from handson.data.physx_session_history import load_session_history

        loaded = load_session_history(history_path, require_finalized=True)
    except Exception as exc:
        errors.append(f"session history 未通过 finalized 完整性验证：{exc}")
        return errors
    if not isinstance(loaded, Mapping):
        return errors + ["session history loader 未返回 object"]
    session = loaded.get("session")
    events = loaded.get("events")
    finalization = loaded.get("finalization")
    if not isinstance(session, Mapping):
        errors.append("session history header 无效")
        session = {}
    if not isinstance(events, list):
        errors.append("session history events 无效")
        events = []
    if not isinstance(finalization, Mapping):
        errors.append("session history finalization 无效")
        finalization = {}

    header_expected = {
        "session_id": session_contract.get("history_id"),
        "task_name": task_name,
        "collection_run_seed": session_contract.get("collection_run_seed"),
    }
    for field, expected in header_expected.items():
        actual = session.get(field)
        if type(actual) is not type(expected) or actual != expected:
            errors.append(f"session history header.{field} 与 manifest 不一致")

    finalization_expected = {
        **header_expected,
        "event_count": session_contract.get("event_count"),
        "certified_through_event_ordinal": session_contract.get(
            "certified_through_event_ordinal"
        ),
        "sim_time_continuity": "complete",
    }
    for field, expected in finalization_expected.items():
        actual = finalization.get(field)
        if type(actual) is not type(expected) or actual != expected:
            errors.append(f"session history finalization.{field} 与 manifest 不一致")
    if len(events) != session_contract.get("event_count"):
        errors.append("session history 实际 event 数与 manifest 不一致")
    history_identity = finalization.get("history_identity")
    if not isinstance(history_identity, Mapping) or history_identity.get(
        "sha256"
    ) != session_contract.get("history_sha256"):
        errors.append("session history 实际 SHA-256 与 manifest 不一致")

    accepted_rows = finalization.get("accepted_demos")
    demo_names = acceptance.get("session_demo_names")
    reset_ordinals = acceptance.get("session_episode_reset_ordinals")
    if (
        not isinstance(accepted_rows, list)
        or not isinstance(demo_names, list)
        or not isinstance(reset_ordinals, list)
        or len(accepted_rows) != len(demo_names)
        or len(accepted_rows) != len(reset_ordinals)
    ):
        errors.append("session history accepted_demos 未覆盖完整 session")
        return errors

    accepted_event_ordinals: list[int] = []
    for index, (row, demo_name, reset_ordinal) in enumerate(
        zip(accepted_rows, demo_names, reset_ordinals, strict=True)
    ):
        label = f"session history accepted_demos[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"{label} 必须为 object")
            continue
        if row.get("demo_id") != demo_name:
            errors.append(f"{label}.demo_id 与 acceptance 顺序不一致")
        if row.get("reset_ordinal") != reset_ordinal:
            errors.append(f"{label}.reset_ordinal 与 acceptance 不一致")
        event_ordinal = row.get("event_ordinal")
        if (
            type(event_ordinal) is not int
            or event_ordinal <= 0
            or event_ordinal >= len(events)
        ):
            errors.append(f"{label}.event_ordinal 无效")
        else:
            accepted_event_ordinals.append(event_ordinal)
        if type(row.get("step_count")) is not int or row.get("step_count") <= 0:
            errors.append(f"{label}.step_count 必须为正整数")
    if any(
        current <= previous
        for previous, current in zip(
            accepted_event_ordinals,
            accepted_event_ordinals[1:],
        )
    ):
        errors.append("session history accepted event ordinals 必须严格递增")
    if accepted_event_ordinals and accepted_event_ordinals[-1] != (
        session_contract.get("certified_through_event_ordinal")
    ):
        errors.append("session history accepted rows 未覆盖 certified-through")

    episode_index = acceptance.get("session_episode_index")
    if type(episode_index) is int and 0 < episode_index <= len(accepted_rows):
        current_row = accepted_rows[episode_index - 1]
        actual_steps, actual_step_errors = _recorded_trajectory_step_count(demo_path)
        errors.extend(actual_step_errors)
        if (
            isinstance(current_row, Mapping)
            and current_row.get("step_count") != actual_steps
        ):
            errors.append(
                "当前 session accepted trajectory step_count 与 replay.zarr 不一致"
            )
        if isinstance(current_row, Mapping) and current_row.get(
            "event_ordinal"
        ) != session_contract.get("accepted_event_ordinal"):
            errors.append("当前 session history accepted event 与 manifest 不一致")
        accepted_event_ordinal = session_contract.get("accepted_event_ordinal")
        if type(accepted_event_ordinal) is int and 0 < accepted_event_ordinal < len(
            events
        ):
            reset_event = events[accepted_event_ordinal - 1]
            reset_payload = (
                reset_event.get("payload") if isinstance(reset_event, Mapping) else None
            )
            if not isinstance(reset_payload, Mapping):
                errors.append("当前 session history accepted reset payload 无效")
            else:
                session_contract["target_reset_payload"] = dict(reset_payload)
            if isinstance(reset_payload, Mapping) and not _json_values_equal(
                reset_payload.get("stable_origin"),
                session_contract.get("stable_recording_origin"),
            ):
                errors.append(
                    "当前 session history accepted reset stable_origin "
                    "与 recording 不一致"
                )
            if isinstance(reset_payload, Mapping):
                stable_origin = session_contract.get("stable_recording_origin")
                if reset_payload.get("seed") != session_contract.get(
                    "episode_reset_seed"
                ):
                    errors.append("当前 session accepted reset seed 与 manifest 不一致")
                if reset_payload.get("reset_ordinal") != session_contract.get(
                    "episode_reset_ordinal"
                ):
                    errors.append(
                        "当前 session accepted reset ordinal 与 manifest 不一致"
                    )
                if not isinstance(stable_origin, Mapping) or (
                    stable_origin.get("scene_rebuilt_before_reset")
                    is not reset_payload.get("scene_rebuilt_before_reset")
                    or stable_origin.get("scene_generation")
                    != reset_payload.get("scene_generation")
                ):
                    errors.append(
                        "当前 session reset scene rebuild/generation 与 stable origin 不一致"
                    )
    session_contract["accepted_event_ordinals"] = accepted_event_ordinals
    session_contract["resolved_history_path"] = str(history_path)
    expected_event_replay, replay_errors = _journal_event_replay_contract(
        events,
        acceptance=acceptance,
        session_contract=session_contract,
        history_path=history_path,
    )
    errors.extend(replay_errors)
    if expected_event_replay is not None:
        session_contract["expected_event_replay"] = expected_event_replay
    return errors


def _ordered_session_acceptance_contract_errors(
    demo_path: Path,
    recording: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    session_contract: Mapping[str, Any],
) -> list[str]:
    """Bind an acceptance proof to its complete ordered source session."""

    errors: list[str] = []
    if (
        acceptance.get("certification_scope")
        != ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE
    ):
        errors.append(
            "certification_scope 必须为 "
            f"{ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE}"
        )
    if acceptance.get("collection_session_id") != session_contract.get("history_id"):
        errors.append("collection_session_id 必须与 manifest session history id 一致")
    if acceptance.get("collection_run_seed") != recording.get("collection_run_seed"):
        errors.append("collection_run_seed 必须与 recording 一致")

    episode_count = acceptance.get("session_episode_count")
    episode_index = acceptance.get("session_episode_index")
    if type(episode_count) is not int or episode_count <= 0:
        errors.append("session_episode_count 必须为正整数")
    if (
        type(episode_index) is not int
        or episode_index <= 0
        or type(episode_count) is not int
        or episode_index > episode_count
    ):
        errors.append("session_episode_index 必须位于 session 范围内")
    if episode_index != session_contract.get("episode_index"):
        errors.append("session_episode_index 必须与 recording.episode_index 一致")

    reset_ordinals = acceptance.get("session_episode_reset_ordinals")
    if not isinstance(reset_ordinals, list) or (
        type(episode_count) is not int or len(reset_ordinals) != episode_count
    ):
        errors.append("session_episode_reset_ordinals 必须覆盖完整 session")
        reset_ordinals = []
    elif any(type(value) is not int or value < 0 for value in reset_ordinals):
        errors.append("session_episode_reset_ordinals 必须为非负整数列表")
    elif any(
        current <= previous
        for previous, current in zip(reset_ordinals, reset_ordinals[1:])
    ):
        errors.append("session_episode_reset_ordinals 必须严格递增")
    if (
        type(episode_index) is int
        and 0 < episode_index <= len(reset_ordinals)
        and reset_ordinals[episode_index - 1]
        != session_contract.get("episode_reset_ordinal")
    ):
        errors.append("当前 session reset ordinal 必须与 recording 一致")

    demo_names = acceptance.get("session_demo_names")
    if not isinstance(demo_names, list) or (
        type(episode_count) is not int or len(demo_names) != episode_count
    ):
        errors.append("session_demo_names 必须覆盖完整 session")
        demo_names = []
    elif any(
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or name in {".", ".."}
        for name in demo_names
    ):
        errors.append("session_demo_names 必须为安全的单级目录名列表")
    elif len(set(demo_names)) != len(demo_names):
        errors.append("session_demo_names 不得重复")
    if (
        type(episode_index) is int
        and 0 < episode_index <= len(demo_names)
        and demo_names[episode_index - 1] != demo_path.name
    ):
        errors.append("当前 demo 必须位于 session_demo_names 对应位置")

    for field in (
        "fresh_certifier_process_per_session_run",
        "session_publication_all_targets_prevalidated",
        "session_publication_rollback_on_catchable_error",
    ):
        if acceptance.get(field) is not True:
            errors.append(f"{field} 必须为 true")
    return errors


def _ordered_session_report_errors(
    report: Mapping[str, Any],
    *,
    acceptance: Mapping[str, Any],
    session_contract: Mapping[str, Any],
    label: str,
) -> list[str]:
    """Validate report evidence for one accepted event in a shared replay."""

    errors: list[str] = []
    session = report.get("session_replay")
    if not isinstance(session, Mapping):
        return [f"{label}.session_replay 必须为 object"]

    expected_fields = {
        "mode": SESSION_REPLAY_EXECUTION_MODE,
        "status": "passed",
        "shared_environment": True,
        "episode_index": acceptance.get("session_episode_index", 0) - 1
        if type(acceptance.get("session_episode_index")) is int
        else None,
        "episode_count": acceptance.get("session_episode_count"),
        "ordered_episode_reset_ordinals": acceptance.get(
            "session_episode_reset_ordinals"
        ),
        "session_history_id": session_contract.get("history_id"),
        "session_history_sha256": session_contract.get("history_sha256"),
        "session_history_event_count": session_contract.get("event_count"),
        "accepted_reset_event_ordinal": (
            session_contract.get("accepted_event_ordinal") - 1
            if type(session_contract.get("accepted_event_ordinal")) is int
            else None
        ),
        "accepted_event_ordinal": session_contract.get("accepted_event_ordinal"),
        "replayed_through_event_ordinal": session_contract.get(
            "accepted_event_ordinal"
        ),
        "history_certified_through_event_ordinal": session_contract.get(
            "certified_through_event_ordinal"
        ),
    }
    for field, expected in expected_fields.items():
        actual = session.get(field)
        if type(actual) is not type(expected) or actual != expected:
            errors.append(
                f"{label}.session_replay.{field} 必须与 session manifest 一致"
            )
    recorded_stable_origin = session_contract.get("stable_recording_origin")
    expected_replayed_history = (
        _canonical_initialization_history_projection(recorded_stable_origin)
        if isinstance(recorded_stable_origin, Mapping)
        else None
    )
    if not _json_values_equal(
        report.get("replayed_stable_origin_history"),
        expected_replayed_history,
    ):
        errors.append(
            f"{label}.replayed_stable_origin_history 必须与 ordered session "
            "recording 完全一致"
        )

    event_replay = session.get("event_replay")
    if not isinstance(event_replay, Mapping):
        errors.append(f"{label}.session_replay.event_replay 必须为 object")
        return errors
    if event_replay.get("status") != SESSION_EVENT_REPLAY_PASSED_STATUS:
        errors.append(
            f"{label}.session_replay.event_replay.status 必须为 "
            f"{SESSION_EVENT_REPLAY_PASSED_STATUS}"
        )
    journal_expected = session_contract.get("expected_event_replay")
    if not isinstance(journal_expected, Mapping):
        errors.append(f"{label} 缺少 journal-derived event replay contract")
    else:
        for field, expected in journal_expected.items():
            actual = event_replay.get(field)
            if type(actual) is not type(expected) or actual != expected:
                errors.append(
                    f"{label}.session_replay.event_replay.{field} "
                    "必须等于 journal 派生值"
                )
    accepted_event_ordinal = session_contract.get("accepted_event_ordinal")
    target_reset_ordinal = (
        accepted_event_ordinal - 1 if type(accepted_event_ordinal) is int else None
    )
    event_expected = {
        "target_accepted_reset_event_ordinal": target_reset_ordinal,
        "target_accepted_trajectory_event_ordinal": accepted_event_ordinal,
        "previous_episode_reset_ordinal": (
            None
            if acceptance.get("session_episode_index") == 1
            else (
                acceptance.get("session_episode_reset_ordinals", [])[
                    acceptance.get("session_episode_index") - 2
                ]
                if (
                    isinstance(
                        acceptance.get("session_episode_reset_ordinals"),
                        list,
                    )
                    and type(acceptance.get("session_episode_index")) is int
                    and 1
                    < acceptance.get("session_episode_index")
                    <= len(acceptance.get("session_episode_reset_ordinals"))
                )
                else None
            )
        ),
        "next_episode_reset_ordinal": session_contract.get("episode_reset_ordinal"),
    }
    for field, expected in event_expected.items():
        actual = event_replay.get(field)
        if type(actual) is not type(expected) or actual != expected:
            errors.append(f"{label}.session_replay.event_replay.{field} 无效")

    target_reset_payload = session_contract.get("target_reset_payload")
    if not isinstance(target_reset_payload, Mapping):
        errors.append(f"{label} 缺少 journal target reset payload")
    else:
        expected_rebuilt = target_reset_payload.get("scene_rebuilt_before_reset")
        expected_generation = target_reset_payload.get("scene_generation")
        expected_rebuild_report = target_reset_payload.get("scene_rebuild_report")
        target_scene_rebuild = event_replay.get("target_scene_rebuild")
        if expected_rebuilt is True:
            if "target_scene_rebuild" not in event_replay:
                errors.append(
                    f"{label}.session_replay.event_replay.target_scene_rebuild 缺失"
                )
            elif not isinstance(expected_rebuild_report, Mapping):
                errors.append(f"{label} journal scene_rebuild_report 无效")
            elif not _json_values_equal(
                target_scene_rebuild,
                expected_rebuild_report,
            ):
                errors.append(
                    f"{label}.session_replay.event_replay.target_scene_rebuild "
                    "必须与 journal 完全一致"
                )
            elif (
                not isinstance(target_scene_rebuild, Mapping)
                or target_scene_rebuild.get("scene_rebuilt") is not True
                or target_scene_rebuild.get("scene_generation") != expected_generation
                or target_scene_rebuild.get("seed") != target_reset_payload.get("seed")
            ):
                errors.append(
                    f"{label}.session_replay.event_replay.target_scene_rebuild "
                    "未绑定目标 scene generation/seed"
                )
        elif expected_rebuilt is False:
            if expected_generation != 0 or expected_rebuild_report is not None:
                errors.append(f"{label} journal 未重建 scene 的证据矛盾")
            if target_scene_rebuild is not None:
                errors.append(
                    f"{label}.session_replay.event_replay.target_scene_rebuild "
                    "必须为 null"
                )
        else:
            errors.append(f"{label} journal scene_rebuilt_before_reset 无效")

    counter_fields = (
        "events_replayed",
        "reset_attempts_replayed",
        "rejected_reset_attempts_replayed",
        "discarded_trajectories_replayed",
        "discarded_actions_replayed",
        "discarded_drive_substeps_replayed",
    )
    counters: dict[str, int] = {}
    for field in counter_fields:
        value = event_replay.get(field)
        if type(value) is not int or value < 0:
            errors.append(f"{label}.session_replay.event_replay.{field} 必须为非负整数")
        else:
            counters[field] = value
    events_replayed = counters.get("events_replayed")
    first_event = event_replay.get("first_event_ordinal")
    last_event = event_replay.get("last_event_ordinal")
    if events_replayed == 0:
        if first_event is not None or last_event is not None:
            errors.append(f"{label}.session_replay.event_replay 空前缀边界无效")
    elif events_replayed is not None:
        if (
            type(first_event) is not int
            or type(last_event) is not int
            or first_event < 0
            or last_event < first_event
            or last_event - first_event + 1 != events_replayed
            or type(target_reset_ordinal) is not int
            or last_event != target_reset_ordinal - 1
        ):
            errors.append(f"{label}.session_replay.event_replay 前缀边界与事件数不一致")
        if acceptance.get("session_episode_index") == 1 and first_event != 0:
            errors.append(
                f"{label}.session_replay.event_replay 首条 episode 必须从事件 0 重放"
            )

    reset_attempts = counters.get("reset_attempts_replayed")
    rejected_resets = counters.get("rejected_reset_attempts_replayed")
    discarded = counters.get("discarded_trajectories_replayed")
    if (
        reset_attempts is not None
        and rejected_resets is not None
        and discarded is not None
        and reset_attempts != rejected_resets + discarded
    ):
        errors.append(f"{label}.session_replay.event_replay reset/trajectory 计数矛盾")
    if (
        events_replayed is not None
        and reset_attempts is not None
        and discarded is not None
        and events_replayed != reset_attempts + discarded
    ):
        errors.append(f"{label}.session_replay.event_replay 事件总数与重放计数矛盾")
    return errors


def _ordered_session_publication_commit_errors(
    demo_path: Path,
    acceptance: Mapping[str, Any],
    session_contract: Mapping[str, Any],
    *,
    require_publication_commit: bool,
) -> list[str]:
    """Validate the root-level atomic commit marker for a published session."""

    errors: list[str] = []
    commit_id = acceptance.get("session_publication_commit_id")
    if (
        not isinstance(commit_id, str)
        or not commit_id
        or re.fullmatch(r"[A-Za-z0-9._-]+", commit_id) is None
        or commit_id in {".", ".."}
    ):
        errors.append("session_publication_commit_id 无效")
        commit_id = ""
    relative_path = acceptance.get("session_publication_commit_root_relative_path")
    expected_relative_path = (
        Path(SESSION_PUBLICATION_COMMIT_DIRECTORY) / f"{commit_id}.json"
    ).as_posix()
    if relative_path != expected_relative_path:
        errors.append(
            "session_publication_commit_root_relative_path 与 commit id 不一致"
        )
    if not require_publication_commit or errors:
        return errors

    output_root = demo_path.parent
    commit_path = output_root / expected_relative_path
    commit_parent = commit_path.parent
    try:
        resolved_root = output_root.resolve(strict=True)
        if commit_parent.is_symlink() or commit_path.is_symlink():
            raise OSError("publication commit path may not use symlinks")
        resolved_commit = commit_path.resolve(strict=True)
        resolved_commit.relative_to(resolved_root)
        if not resolved_commit.is_file():
            raise OSError("publication commit marker is not a regular file")
        marker = json.loads(resolved_commit.read_text(encoding="utf-8"))
    except (
        FileNotFoundError,
        OSError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return [f"session publication commit marker 无法验证：{exc}"]
    if not isinstance(marker, Mapping):
        return ["session publication commit marker 必须为 JSON object"]

    marker_expected = {
        "format": SESSION_PUBLICATION_COMMIT_FORMAT,
        "version": SESSION_PUBLICATION_COMMIT_VERSION,
        "status": "committed",
        "session_publication_commit_id": commit_id,
        "session_publication_commit_root_relative_path": relative_path,
        "collection_session_id": acceptance.get("collection_session_id"),
        "session_demo_names": acceptance.get("session_demo_names"),
        "session_episode_reset_ordinals": acceptance.get(
            "session_episode_reset_ordinals"
        ),
        "session_history_id": session_contract.get("history_id"),
        "session_history_sha256": session_contract.get("history_sha256"),
        "session_history_event_count": session_contract.get("event_count"),
        "session_history_certified_through_event_ordinal": (
            session_contract.get("certified_through_event_ordinal")
        ),
    }
    for field, expected in marker_expected.items():
        actual = marker.get(field)
        if type(actual) is not type(expected) or actual != expected:
            errors.append(f"session publication marker.{field} 不匹配")

    members = marker.get("members")
    demo_names = acceptance.get("session_demo_names")
    reset_ordinals = acceptance.get("session_episode_reset_ordinals")
    if (
        not isinstance(members, list)
        or not isinstance(demo_names, list)
        or len(members) != len(demo_names)
    ):
        errors.append("session publication marker.members 未覆盖完整 session")
        return errors

    accepted_ordinals: list[int] = []
    current_member: Mapping[str, Any] | None = None
    for index, member in enumerate(members):
        label = f"session publication marker.members[{index}]"
        if not isinstance(member, Mapping):
            errors.append(f"{label} 必须为 object")
            continue
        expected_name = demo_names[index]
        expected_reset = (
            reset_ordinals[index]
            if isinstance(reset_ordinals, list) and index < len(reset_ordinals)
            else None
        )
        if member.get("demo_name") != expected_name:
            errors.append(f"{label}.demo_name 与 session 顺序不一致")
        if member.get("episode_reset_ordinal") != expected_reset:
            errors.append(f"{label}.episode_reset_ordinal 不一致")
        accepted_ordinal = member.get("accepted_event_ordinal")
        if (
            type(accepted_ordinal) is not int
            or accepted_ordinal <= 0
            or type(session_contract.get("event_count")) is not int
            or accepted_ordinal >= session_contract.get("event_count")
        ):
            errors.append(f"{label}.accepted_event_ordinal 无效")
        else:
            accepted_ordinals.append(accepted_ordinal)
        if not _is_sha256(member.get("acceptance_sha256")):
            errors.append(f"{label}.acceptance_sha256 必须为小写 SHA-256")
        if not isinstance(member.get("certified_inputs"), Mapping):
            errors.append(f"{label}.certified_inputs 必须为 object")
        if expected_name == demo_path.name:
            current_member = member

        if (
            not isinstance(expected_name, str)
            or not expected_name
            or Path(expected_name).name != expected_name
            or expected_name in {".", ".."}
        ):
            errors.append(f"{label}.demo_name 不是安全的单级目录名")
            continue
        sibling = resolved_root / expected_name
        sibling_acceptance_path = sibling / FRESH_PROCESS_ACCEPTANCE_NAME
        try:
            if sibling.is_symlink() or not sibling.is_dir():
                raise OSError(
                    "session member is missing, a symlink, or not a directory"
                )
            resolved_sibling = sibling.resolve(strict=True)
            resolved_sibling.relative_to(resolved_root)
            if sibling_acceptance_path.is_symlink():
                raise OSError("session member acceptance is a symlink")
            resolved_acceptance = sibling_acceptance_path.resolve(strict=True)
            resolved_acceptance.relative_to(resolved_sibling)
            if not resolved_acceptance.is_file():
                raise OSError("session member acceptance is not a regular file")
            sibling_acceptance_sha256 = _sha256_file(resolved_acceptance)
            sibling_acceptance = json.loads(
                resolved_acceptance.read_text(encoding="utf-8")
            )
        except (
            FileNotFoundError,
            OSError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            errors.append(f"{label} 对应 sibling acceptance 无法验证：{exc}")
            continue
        if not isinstance(sibling_acceptance, Mapping):
            errors.append(f"{label} 对应 sibling acceptance 必须为 object")
            continue
        if member.get("acceptance_sha256") != sibling_acceptance_sha256:
            errors.append(f"{label}.acceptance_sha256 与 sibling 不一致")
        try:
            sibling_actual_inputs = capture_certified_inputs(resolved_sibling)
        except (OSError, UnicodeError, OverflowError) as exc:
            errors.append(f"{label} 对应 sibling inputs 无法复核：{exc}")
            sibling_actual_inputs = None
        if sibling_actual_inputs is not None and not _json_values_equal(
            member.get("certified_inputs"),
            sibling_actual_inputs,
        ):
            errors.append(f"{label}.certified_inputs 与 sibling 当前输入不一致")
        if sibling_actual_inputs is not None and not _json_values_equal(
            sibling_acceptance.get("certified_inputs"),
            sibling_actual_inputs,
        ):
            errors.append(
                f"{label} 对应 sibling acceptance.certified_inputs 与当前输入不一致"
            )
        sibling_expected = {
            "format": DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_FORMAT,
            "version": DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_VERSION,
            "mode": DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE,
            "status": "passed",
            "certification_scope": (
                ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE
            ),
            "collection_session_id": acceptance.get("collection_session_id"),
            "session_episode_count": len(demo_names),
            "session_episode_index": index + 1,
            "session_episode_reset_ordinals": reset_ordinals,
            "session_demo_names": demo_names,
            "session_publication_commit_id": commit_id,
            "session_publication_commit_root_relative_path": relative_path,
            "certified_inputs": member.get("certified_inputs"),
        }
        for field, expected in sibling_expected.items():
            actual = sibling_acceptance.get(field)
            if type(actual) is not type(expected) or actual != expected:
                errors.append(f"{label} 对应 sibling acceptance.{field} 不匹配")

    if any(
        current <= previous
        for previous, current in zip(accepted_ordinals, accepted_ordinals[1:])
    ):
        errors.append("session publication marker accepted_event_ordinal 必须严格递增")
    if accepted_ordinals != session_contract.get("accepted_event_ordinals"):
        errors.append(
            "session publication marker accepted events 与实际 history 不一致"
        )
    if accepted_ordinals and accepted_ordinals[-1] != session_contract.get(
        "certified_through_event_ordinal"
    ):
        errors.append(
            "session publication marker 未覆盖 history certified-through event"
        )
    if current_member is None:
        errors.append("session publication marker 缺少当前 demo member")
        return errors
    if current_member.get("accepted_event_ordinal") != session_contract.get(
        "accepted_event_ordinal"
    ):
        errors.append("当前 demo member accepted_event_ordinal 与 manifest 不一致")
    acceptance_path = demo_path / FRESH_PROCESS_ACCEPTANCE_NAME
    try:
        acceptance_sha256 = _sha256_file(acceptance_path)
    except OSError as exc:
        errors.append(f"当前 acceptance 无法哈希：{exc}")
    else:
        if current_member.get("acceptance_sha256") != acceptance_sha256:
            errors.append("当前 acceptance 与 session publication marker 哈希不一致")
    if current_member.get("certified_inputs") != acceptance.get("certified_inputs"):
        errors.append("当前 acceptance certified_inputs 与 publication marker 不一致")
    return errors


def _recording_manifest_software(
    demo_path: Path,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, list[str]]:
    """Load the immutable source/runtime identities bound by the manifest."""

    try:
        manifest = json.loads(
            (demo_path / "replay_manifest.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, None, [f"replay_manifest.json 无法读取：{exc}"]
    software = manifest.get("software") if isinstance(manifest, Mapping) else None
    if not isinstance(software, Mapping):
        return None, None, ["manifest.software 必须为 object"]
    recorded_git = software.get("git")
    runtime_identity = software.get("runtime_replay_identity")
    errors: list[str] = []
    if not isinstance(recorded_git, Mapping):
        errors.append("manifest.software.git 必须为 object")
        recorded_git = None
    if not isinstance(runtime_identity, Mapping) or not runtime_identity:
        errors.append("manifest.software.runtime_replay_identity 必须为非空 object")
        runtime_identity = None
    return recorded_git, runtime_identity, errors


def _flatten_mapping_values(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for raw_name, item in value.items():
        name = f"{prefix}.{raw_name}" if prefix else str(raw_name)
        if isinstance(item, Mapping):
            fields.update(_flatten_mapping_values(item, prefix=name))
        else:
            fields[name] = item
    return {name: fields[name] for name in sorted(fields)}


def _canonical_replay_atol(name: str, dtype: str) -> float:
    if dtype.startswith(("bool", "int", "uint")):
        return 0.0
    if name == "physx_sim_time":
        return 1.0e-12
    if "velocity" in name:
        return 1.0e-4 if "deformable" in name else 1.0e-5
    if "effort" in name:
        return 1.0e-4
    if "deformable_position" in name:
        return 1.0e-5
    if "command_" in name or "filter_state" in name:
        return 1.0e-9
    return 1.0e-6


def _canonical_physical_replay_field(name: str) -> bool:
    is_signature = name.startswith("physx_contact_signature_") or (
        name.startswith("physx_substep_contact_signature_")
    )
    is_contact = (
        name.startswith("physx_contact_")
        or name.startswith("physx_friction_anchor_")
        or name.startswith("physx_substep_contact_")
        or name.startswith("physx_substep_friction_anchor_")
    )
    # Match PhysxGroundTruthTrace.schema(): solver-produced impulse and
    # separation aggregates are diagnostics. Discrete contact identities and
    # counts remain replay requirements, alongside the physical state fields.
    is_metric = name in {
        "physx_contact_signature_pair_metrics",
        "physx_substep_contact_signature_pair_metrics",
    }
    return (is_signature and not is_metric) or not is_contact


def _schema_names(value: Any, *, label: str, errors: list[str]) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(name, str) or not name for name in value)
        or value != list(dict.fromkeys(value))
    ):
        errors.append(f"recording schema {label} 必须为非空唯一名称列表")
        return []
    return list(value)


def _recorded_trajectory_step_count(demo_path: Path) -> tuple[int | None, list[str]]:
    """Read the immutable Zarr action-array shape without trusting a report."""

    action_path = demo_path / "replay.zarr" / "data" / "action"
    metadata_paths = (action_path / ".zarray", action_path / "zarr.json")
    metadata: Any = None
    for metadata_path in metadata_paths:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            break
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return None, [f"replay.zarr action metadata 无法读取：{exc}"]
    if not isinstance(metadata, Mapping):
        return None, ["replay.zarr 缺少 action array metadata"]
    shape = metadata.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        return None, ["replay.zarr action shape 必须为非空正整数维度"]
    return int(shape[0]), []


def _recording_schema_report_contract(
    demo_path: Path,
) -> tuple[
    dict[str, float],
    dict[str, Any],
    dict[str, list[str]],
    list[str],
]:
    """Validate canonical PhysX schema and derive immutable report evidence."""

    try:
        schema = json.loads(
            (demo_path / "recording_schema.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, {}, {}, [f"recording_schema.json 无法读取：{exc}"]
    if not isinstance(schema, Mapping):
        return {}, {}, {}, ["recording_schema.json 必须为 object"]
    environment_schema = schema.get("environment_schema")
    if not isinstance(environment_schema, Mapping):
        return {}, {}, {}, ["recording_schema.environment_schema 必须为 object"]
    errors: list[str] = []
    try:
        manifest = json.loads(
            (demo_path / "replay_manifest.json").read_text(encoding="utf-8")
        )
        task_name = str(manifest.get("task", {}).get("name") or "")
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        task_name = ""
    if environment_schema.get("name") != PHYSX_GROUND_TRUTH_SCHEMA_NAME:
        errors.append("recording schema name 不是 PhysX ground truth")
    if environment_schema.get("version") != PHYSX_GROUND_TRUTH_SCHEMA_VERSION:
        errors.append(
            f"recording schema version 必须为 {PHYSX_GROUND_TRUTH_SCHEMA_VERSION}"
        )
    fields = environment_schema.get("fields")
    stored_fields = schema.get("fields")
    if not isinstance(fields, Mapping):
        return {}, {}, {}, errors + ["recording schema fields 必须为 object"]
    if not isinstance(stored_fields, Mapping):
        return {}, {}, {}, errors + ["recording schema 顶层 fields 必须为 object"]

    physx = environment_schema.get("physx")
    if not isinstance(physx, Mapping):
        return {}, {}, {}, errors + ["recording schema physx 必须为 object"]
    robot_sides = _schema_names(
        physx.get("robot_sides"),
        label="physx.robot_sides",
        errors=errors,
    )
    if robot_sides != ["right", "left"]:
        errors.append("recording schema robot_sides 必须为 right/left")
    robot_joint_names = physx.get("robot_joint_names")
    if not isinstance(robot_joint_names, Mapping) or set(robot_joint_names) != set(
        robot_sides
    ):
        errors.append("recording schema robot_joint_names 未精确覆盖双臂")
        robot_joint_names = {}
    joint_rows = {
        side: _schema_names(
            robot_joint_names.get(side),
            label=f"physx.robot_joint_names.{side}",
            errors=errors,
        )
        for side in robot_sides
    }
    max_robot_joints = max((len(row) for row in joint_rows.values()), default=0)
    if max_robot_joints <= 0:
        errors.append("recording schema 未声明机器人关节")

    rigid_names = (
        _schema_names(
            physx.get("rigid_body_names"),
            label="physx.rigid_body_names",
            errors=errors,
        )
        if physx.get("rigid_body_names") != []
        else []
    )
    articulation_names = (
        _schema_names(
            physx.get("articulation_names"),
            label="physx.articulation_names",
            errors=errors,
        )
        if physx.get("articulation_names") != []
        else []
    )
    raw_dof_names = physx.get("articulation_dof_names")
    if not isinstance(raw_dof_names, Mapping) or set(raw_dof_names) != set(
        articulation_names
    ):
        errors.append("recording schema articulation_dof_names 键不完整")
        raw_dof_names = {}
    qualified_dof_names: list[str] = []
    max_articulation_dofs = 0
    for articulation in articulation_names:
        row = (
            _schema_names(
                raw_dof_names.get(articulation),
                label=f"physx.articulation_dof_names.{articulation}",
                errors=errors,
            )
            if raw_dof_names.get(articulation) != []
            else []
        )
        max_articulation_dofs = max(max_articulation_dofs, len(row))
        qualified_dof_names.extend(f"{articulation}:{name}" for name in row)
    if len(qualified_dof_names) != len(set(qualified_dof_names)):
        errors.append("recording schema articulation DOF 名称不唯一")

    deformable_name = physx.get("deformable_name")
    if deformable_name is not None and (
        not isinstance(deformable_name, str) or not deformable_name
    ):
        errors.append("recording schema deformable_name 无效")

    physics_config = physx.get("physics_config")
    if not isinstance(physics_config, Mapping) or not physics_config:
        errors.append("recording schema 缺少非空 PhysX physics_config")
        physics_values: dict[str, Any] = {}
    else:
        physics_values = _flatten_mapping_values(physics_config)
        for name, value in physics_values.items():
            if isinstance(value, float) and not math.isfinite(value):
                errors.append(f"recording schema physics_config.{name} 非有限")
    device = (
        physics_config.get("device") if isinstance(physics_config, Mapping) else None
    )
    cpu_exception = physx.get("cpu_backend_exception")
    device_allowed = is_physx_recording_device_allowed(
        device,
        cpu_exception=cpu_exception,
    )
    requires_cuda_backend = physx.get("requires_cuda_backend")
    if not device_allowed or requires_cuda_backend is not is_canonical_cuda_device(
        device
    ):
        errors.append(
            "recording schema 必须使用 cuda:N，或声明匹配的显式 CPU "
            "exact-asset exception；device 与 requires_cuda_backend 不一致"
        )
    physics_dt = (
        physics_config.get("physics_dt")
        if isinstance(physics_config, Mapping)
        else None
    )
    control_dt = (
        physics_config.get("control_dt")
        if isinstance(physics_config, Mapping)
        else None
    )
    n_substeps = (
        physics_config.get("n_substeps")
        if isinstance(physics_config, Mapping)
        else None
    )
    if (
        type(physics_dt) not in {int, float}
        or isinstance(physics_dt, bool)
        or not math.isfinite(float(physics_dt))
        or float(physics_dt) <= 0.0
        or type(control_dt) not in {int, float}
        or isinstance(control_dt, bool)
        or not math.isfinite(float(control_dt))
        or float(control_dt) <= 0.0
        or type(n_substeps) is not int
        or n_substeps <= 0
        or not math.isclose(
            float(control_dt),
            float(physics_dt) * n_substeps,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        errors.append("recording schema physics_dt/control_dt/n_substeps 无效")
        n_substeps = 0
    for field, expected in (
        ("physics_dt", physics_dt),
        ("control_dt", control_dt),
        ("n_substeps", n_substeps),
    ):
        if not _json_values_equal(physx.get(field), expected):
            errors.append(f"recording schema physx.{field} 与 physics_config 不一致")
    enhanced = (
        physics_config.get("solver", {}).get("enable_enhanced_determinism")
        if isinstance(physics_config, Mapping)
        and isinstance(physics_config.get("solver"), Mapping)
        else None
    )
    if enhanced is not True or physx.get("enhanced_determinism") is not True:
        errors.append("recording schema 必须启用 enhanced determinism")

    contact = physx.get("contact_report")
    path_count = 0
    if not isinstance(contact, Mapping):
        errors.append("recording schema 缺少 canonical contact_report")
    else:
        path_table = contact.get("path_table")
        if not isinstance(path_table, list) or not path_table:
            errors.append("recording schema contact_report.path_table 为空")
        else:
            ids: list[int] = []
            paths: list[str] = []
            for row in path_table:
                if (
                    not isinstance(row, Mapping)
                    or type(row.get("id")) is not int
                    or row.get("id") <= 0
                    or not isinstance(row.get("path"), str)
                    or not row.get("path")
                ):
                    errors.append("recording schema contact path row 无效")
                    continue
                ids.append(row["id"])
                paths.append(row["path"])
            if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
                errors.append("recording schema contact path identity 不唯一")
            path_count = len(paths)
            report_paths = contact.get("report_enabled_task_body_paths")
            sensor_paths = contact.get("tensor_sensor_paths")
            filter_paths = contact.get("tensor_filter_paths")
            covered = []
            if isinstance(report_paths, list):
                covered.extend(report_paths)
            if isinstance(sensor_paths, list) and isinstance(filter_paths, list):
                covered.extend(sensor_paths)
                covered.extend(filter_paths)
            if not covered or any(path not in paths for path in covered):
                errors.append("recording schema canonical contact coverage 无效")

    def shape_of(name: str) -> list[int] | None:
        entry = fields.get(name)
        shape = entry.get("shape") if isinstance(entry, Mapping) else None
        if not isinstance(shape, list) or any(
            type(value) is not int or value < 0 for value in shape
        ):
            return None
        return list(shape)

    header_shape = shape_of("physx_contact_header_ids")
    point_shape = shape_of("physx_contact_point_data")
    friction_shape = shape_of("physx_friction_anchor_data")
    header_capacity = header_shape[0] if header_shape and len(header_shape) == 2 else 0
    point_capacity = point_shape[0] if point_shape and len(point_shape) == 2 else 0
    friction_capacity = (
        friction_shape[0] if friction_shape and len(friction_shape) == 2 else 0
    )
    hand_shape = shape_of("physx_command_hand")
    hand_dim = hand_shape[1] if hand_shape and len(hand_shape) == 2 else 0
    deformable_shape = shape_of("physx_deformable_position")
    deformable_count = (
        deformable_shape[0]
        if deformable_name is not None
        and deformable_shape
        and len(deformable_shape) == 2
        and deformable_shape[1] == 3
        else 0
    )

    expected_specs: dict[str, tuple[list[int], str]] = {
        "physx_robot_joint_position": ([2, max_robot_joints], "float64"),
        "physx_robot_joint_velocity": ([2, max_robot_joints], "float64"),
        "physx_robot_joint_target": ([2, max_robot_joints], "float64"),
        "physx_robot_joint_target_valid": ([2, max_robot_joints], "uint8"),
        "physx_robot_joint_valid": ([2, max_robot_joints], "uint8"),
        "physx_robot_root_state": ([2, 13], "float64"),
        "physx_robot_root_valid": ([2], "uint8"),
        "physx_robot_joint_effort": ([2, max_robot_joints], "float64"),
        "physx_robot_joint_effort_valid": ([2, max_robot_joints], "uint8"),
        "physx_tcp_state": ([2, 13], "float64"),
        "physx_tcp_valid": ([2], "uint8"),
        "physx_sim_time": ([1], "float64"),
        "physx_control_step": ([1], "float64"),
        "physx_command_tcp_pose": ([2, 7], "float64"),
        "physx_command_hand": ([2, hand_dim], "float64"),
        "physx_command_hand_valid": ([2, hand_dim], "uint8"),
        "physx_pose_commanded": ([2], "uint8"),
        "physx_pose_filter_state": ([2, 14], "float64"),
        "physx_pose_filter_valid": ([2], "uint8"),
        "physx_contact_report_valid": ([1], "uint8"),
        "physx_contact_capture_source": ([1], "uint8"),
        "physx_contact_header_count": ([1], "uint32"),
        "physx_contact_header_ids": ([header_capacity, 5], "uint64"),
        "physx_contact_header_meta": ([header_capacity, 7], "uint32"),
        "physx_contact_point_count": ([1], "uint32"),
        "physx_contact_point_data": ([point_capacity, 10], "float32"),
        "physx_contact_point_face_indices": ([point_capacity, 2], "uint32"),
        "physx_contact_point_material_ids": ([point_capacity, 2], "uint64"),
        "physx_friction_anchor_count": ([1], "uint32"),
        "physx_friction_anchor_data": ([friction_capacity, 6], "float32"),
        "physx_contact_signature_valid": ([1], "uint8"),
        "physx_contact_signature_pair_count": ([1], "uint32"),
        "physx_contact_signature_pair_path_indices": ([header_capacity, 4], "uint32"),
        "physx_contact_signature_pair_counts": ([header_capacity, 2], "uint32"),
        "physx_contact_signature_pair_metrics": ([header_capacity, 3], "float64"),
        "physx_substep_count": ([1], "int64"),
    }
    if rigid_names:
        expected_specs.update(
            {
                "physx_task_rigid_body_state": ([len(rigid_names), 13], "float64"),
                "physx_task_rigid_body_valid": ([len(rigid_names)], "uint8"),
            }
        )
    if articulation_names:
        expected_specs.update(
            {
                "physx_task_articulation_root_state": (
                    [len(articulation_names), 13],
                    "float64",
                ),
                "physx_task_articulation_root_valid": (
                    [len(articulation_names)],
                    "uint8",
                ),
            }
        )
    if max_articulation_dofs > 0:
        for name, dtype in (
            ("physx_task_articulation_dof_position", "float64"),
            ("physx_task_articulation_dof_velocity", "float64"),
            ("physx_task_articulation_dof_valid", "uint8"),
        ):
            expected_specs[name] = (
                [len(articulation_names), max_articulation_dofs],
                dtype,
            )
    if deformable_name is not None:
        if deformable_count <= 0:
            errors.append("recording schema deformable node shape 无效")
        expected_specs.update(
            {
                "physx_deformable_position": ([deformable_count, 3], "float64"),
                "physx_deformable_velocity": ([deformable_count, 3], "float64"),
                "physx_deformable_valid": ([1], "uint8"),
            }
        )
    if not (rigid_names or articulation_names or deformable_count > 0):
        errors.append("recording schema 未声明任何 task physics entity")
    required_substep_bases = {
        "physx_robot_joint_position",
        "physx_robot_joint_velocity",
        "physx_robot_joint_valid",
        "physx_robot_joint_target",
        "physx_robot_joint_target_valid",
        "physx_sim_time",
    }
    for base_name in tuple(expected_specs):
        if base_name in {
            "physx_control_step",
            "physx_pose_filter_state",
            "physx_pose_filter_valid",
            "physx_substep_count",
        }:
            continue
        if base_name in fields and base_name.startswith("physx_"):
            substep_name = base_name.replace("physx_", "physx_substep_", 1)
            if substep_name in fields or base_name in required_substep_bases:
                shape, dtype = expected_specs[base_name]
                expected_specs[substep_name] = ([n_substeps, *shape], dtype)

    missing_contact = sorted(_REQUIRED_CONTACT_SIGNATURE_FIELDS - set(fields))
    if missing_contact:
        errors.append(
            "recording schema 缺少 canonical contact signature 字段："
            + ", ".join(missing_contact)
        )
    if header_capacity <= 0 or point_capacity <= 0 or friction_capacity <= 0:
        errors.append("recording schema contact buffer capacity 必须为正")
    if hand_dim <= 0 or path_count <= 0:
        errors.append("recording schema hand/contact canonical dimension 无效")

    atols: dict[str, float] = {}
    for raw_name, entry in fields.items():
        name = str(raw_name)
        if not name.startswith("physx_"):
            continue
        if not isinstance(entry, Mapping):
            errors.append(f"recording schema {name} 必须为 object")
            continue
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        if (
            not isinstance(shape, list)
            or any(type(value) is not int or value < 0 for value in shape)
            or not isinstance(dtype, str)
        ):
            errors.append(f"recording schema {name} shape/dtype 无效")
            continue
        expected_spec = expected_specs.get(name)
        if expected_spec is not None and (
            shape != expected_spec[0] or dtype != expected_spec[1]
        ):
            errors.append(f"recording schema {name} canonical shape/dtype 不匹配")
        expected_physical = _canonical_physical_replay_field(name)
        if entry.get("physical_replay") is not expected_physical:
            errors.append(f"recording schema {name}.physical_replay 无效")
        tolerance = entry.get("replay_atol")
        expected_tolerance = _canonical_replay_atol(name, dtype)
        if (
            type(tolerance) not in {int, float}
            or isinstance(tolerance, bool)
            or not math.isfinite(float(tolerance))
            or float(tolerance) != expected_tolerance
        ):
            errors.append(f"recording schema {name}.replay_atol 非 canonical")
            continue
        if expected_physical:
            atols[name] = float(tolerance)
        current = stored_fields.get(name)
        next_entry = stored_fields.get(f"next_{name}")
        for stored_name, stored_entry in (
            (name, current),
            (f"next_{name}", next_entry),
        ):
            if (
                not isinstance(stored_entry, Mapping)
                or stored_entry.get("shape_per_step") != shape
                or stored_entry.get("dtype") != dtype
                or not _json_values_equal(stored_entry.get("semantics"), entry)
            ):
                errors.append(
                    f"recording schema 顶层 {stored_name} 未精确绑定 environment field"
                )
    task_state_fields = {
        str(name): entry
        for name, entry in fields.items()
        if isinstance(entry, Mapping)
        and (
            entry.get("replay_domain") == "task_state"
            or str(name).startswith("bc017_")
        )
    }
    if task_name == BC017_TASK_NAME:
        expected_task_names = set(_BC017_TASK_STATE_FIELD_NAMES)
        if set(task_state_fields) != expected_task_names:
            errors.append(
                "BC017 recording schema 必须精确包含四组 public "
                "task_state 和一组 hidden task_state"
            )
        expected_envelope = {
            "task_name": BC017_TASK_NAME,
            "field_names": list(_BC017_TASK_STATE_FIELD_NAMES),
            "hidden_field_names": ["bc017_hidden_state"],
            "capture_mode": "action_aligned_current_and_next_observation",
            "replay_policy": "compare_every_step_never_restore",
        }
        if environment_schema.get("task_state") != expected_envelope:
            errors.append("BC017 recording schema task_state envelope 无效")
        for name in _BC017_TASK_STATE_FIELD_NAMES:
            entry = task_state_fields.get(name)
            if not isinstance(entry, Mapping):
                continue
            expected_shape = [len(_BC017_TASK_STATE_LAYOUTS[name])]
            expected_visibility = (
                "hidden_task_logic"
                if name == "bc017_hidden_state"
                else "public_task_state"
            )
            expected_atol = _BC017_TASK_STATE_REPLAY_ATOLS[name]
            if (
                entry.get("shape") != expected_shape
                or entry.get("dtype") != BC017_OBSERVATION_DTYPE
                or entry.get("source") != "bc017_action_aligned_task_state"
                or entry.get("layout") != list(_BC017_TASK_STATE_LAYOUTS[name])
                or entry.get("visibility") != expected_visibility
                or entry.get("replay_domain") != "task_state"
                or entry.get("physical_replay") is not True
                or entry.get("replay_restore_supported") is not False
                or entry.get("replay_atol") != expected_atol
            ):
                errors.append(f"BC017 recording schema {name} 读取/比较契约无效")
                continue
            atols[name] = expected_atol
            for stored_name in (name, f"next_{name}"):
                stored_entry = stored_fields.get(stored_name)
                if (
                    not isinstance(stored_entry, Mapping)
                    or stored_entry.get("shape_per_step") != expected_shape
                    or stored_entry.get("dtype") != BC017_OBSERVATION_DTYPE
                    or not _json_values_equal(
                        stored_entry.get("semantics"), entry
                    )
                ):
                    errors.append(
                        f"BC017 recording schema 顶层 {stored_name} 未绑定 task_state field"
                    )
    elif task_name in {AS019_TASK_NAME, XY2_TASK_NAME}:
        xy2 = task_name == XY2_TASK_NAME
        task_field_names = tuple(XY2_RECORDING_LAYOUTS) if xy2 else _AS019_TASK_STATE_FIELD_NAMES
        task_layouts = XY2_RECORDING_LAYOUTS if xy2 else _AS019_TASK_STATE_LAYOUTS
        task_atols = XY2_RECORDING_ATOLS if xy2 else _AS019_TASK_STATE_REPLAY_ATOLS
        task_source = "xy2_action_aligned_task_state" if xy2 else "as019_action_aligned_task_state"
        task_label = "XY2" if xy2 else "AS019"
        expected_task_names = set(task_field_names)
        if set(task_state_fields) != expected_task_names:
            errors.append(
                f"{task_label} recording schema 必须精确包含声明的 hidden task_state"
            )
        expected_envelope = {
            "task_name": task_name,
            "field_names": list(task_field_names),
            "hidden_field_names": list(task_field_names),
            "capture_mode": "action_aligned_current_and_next_observation",
            "replay_policy": "compare_every_step_never_restore",
        }
        if environment_schema.get("task_state") != expected_envelope:
            errors.append(f"{task_label} recording schema task_state envelope 无效")
        for name in task_field_names:
            entry = task_state_fields.get(name)
            if not isinstance(entry, Mapping):
                continue
            expected_shape = [len(task_layouts[name])]
            expected_atol = task_atols[name]
            if (
                entry.get("shape") != expected_shape
                or entry.get("dtype") != "float64"
                or entry.get("source") != task_source
                or entry.get("layout") != list(task_layouts[name])
                or entry.get("visibility") != "hidden_task_logic"
                or entry.get("replay_domain") != "task_state"
                or entry.get("physical_replay") is not True
                or entry.get("replay_restore_supported") is not False
                or entry.get("replay_atol") != expected_atol
            ):
                errors.append(f"{task_label} recording schema {name} 读取/比较契约无效")
                continue
            atols[name] = expected_atol
            for stored_name in (name, f"next_{name}"):
                stored_entry = stored_fields.get(stored_name)
                if (
                    not isinstance(stored_entry, Mapping)
                    or stored_entry.get("shape_per_step") != expected_shape
                    or stored_entry.get("dtype") != "float64"
                    or not _json_values_equal(
                        stored_entry.get("semantics"), entry
                    )
                ):
                    errors.append(
                        f"{task_label} recording schema 顶层 {stored_name} 未绑定 task_state field"
                    )
    elif task_name == XY4_TASK_NAME:
        expected_task_names = set(("xy4_hidden_state",))
        if set(task_state_fields) != expected_task_names:
            errors.append(
                "XY4 recording schema 必须精确包含一组 hidden task_state"
            )
        expected_envelope = {
            "task_name": XY4_TASK_NAME,
            "field_names": list(("xy4_hidden_state",)),
            "hidden_field_names": ["xy4_hidden_state"],
            "capture_mode": "action_aligned_current_and_next_observation",
            "replay_policy": "compare_every_step_never_restore",
        }
        if environment_schema.get("task_state") != expected_envelope:
            errors.append("XY4 recording schema task_state envelope 无效")
        for name in ("xy4_hidden_state",):
            entry = task_state_fields.get(name)
            if not isinstance(entry, Mapping):
                continue
            expected_shape = [len(XY4_HIDDEN_FIELDS)]
            expected_atol = 1e-6
            if (
                entry.get("shape") != expected_shape
                or entry.get("dtype") != "float64"
                or entry.get("source") != "xy4_action_aligned_task_state"
                or entry.get("layout") != list(XY4_HIDDEN_FIELDS)
                or entry.get("visibility") != "hidden_task_logic"
                or entry.get("replay_domain") != "task_state"
                or entry.get("physical_replay") is not True
                or entry.get("replay_restore_supported") is not False
                or entry.get("replay_atol") != expected_atol
            ):
                errors.append(f"XY4 recording schema {name} 读取/比较契约无效")
                continue
            atols[name] = expected_atol
            for stored_name in (name, f"next_{name}"):
                stored_entry = stored_fields.get(stored_name)
                if (
                    not isinstance(stored_entry, Mapping)
                    or stored_entry.get("shape_per_step") != expected_shape
                    or stored_entry.get("dtype") != "float64"
                    or not _json_values_equal(
                        stored_entry.get("semantics"), entry
                    )
                ):
                    errors.append(
                        f"XY4 recording schema 顶层 {stored_name} 未绑定 task_state field"
                    )
    elif task_name == "eval_comb01_cover_pitcher_and_box":
        from ..tasks.eval_comb01_cover_pitcher_and_box.recording import (
            hidden_field_schema, task_state_envelope, REPLAY_ATOL,
        )
        expected_field = hidden_field_schema()
        if task_state_fields != {"xy5_hidden_state": expected_field}:
            errors.append("XY5 recording schema hidden task_state contract is invalid")
        if environment_schema.get("task_state") != task_state_envelope():
            errors.append("XY5 recording schema task_state envelope is invalid")
        atols["xy5_hidden_state"] = REPLAY_ATOL
        for stored_name in ("xy5_hidden_state", "next_xy5_hidden_state"):
            stored_entry = stored_fields.get(stored_name)
            if (
                not isinstance(stored_entry, Mapping)
                or stored_entry.get("shape_per_step") != expected_field["shape"]
                or stored_entry.get("dtype") != "float64"
                or not _json_values_equal(stored_entry.get("semantics"), expected_field)
            ):
                errors.append(f"XY5 recording schema {stored_name} is not bound to task_state")
    elif task_state_fields:
        errors.append("当前仅 BC017、AS019 和 XY5 允许声明 task_state replay domain")
    for name in expected_specs:
        if name not in fields:
            errors.append(f"recording schema 缺少 mandatory field {name}")
    if not atols:
        errors.append("recording schema 没有 physical_replay=true 字段")
    expected_layout = {
        "rigid_body": rigid_names,
        "articulation_root": articulation_names,
        "articulation_dof": qualified_dof_names,
        "deformable_node": (
            [f"{deformable_name}:{index}" for index in range(deformable_count)]
            if deformable_name is not None
            else []
        ),
    }
    return atols, physics_values, expected_layout, errors


def _schema_bound_origin_errors(
    history: Any,
    *,
    expected_layout: Mapping[str, list[str]],
    physics_config: Mapping[str, Any],
    recording: Mapping[str, Any],
    label: str,
) -> list[str]:
    """Bind stable-origin names, bitmaps and timing to the sealed schema."""

    if not isinstance(history, Mapping):
        return [f"{label} 必须为 object"]
    errors: list[str] = []
    for layout_field in (
        "active_physics_layout",
        "gate_active_physics_layout",
        "final_active_physics_layout",
    ):
        layout = history.get(layout_field)
        if not isinstance(layout, Mapping):
            errors.append(f"{label}.{layout_field} 必须为 object")
            continue
        for category in _ACTIVE_PHYSICS_LAYOUT_CATEGORIES:
            section = layout.get(category)
            expected_names = expected_layout.get(category)
            if (
                not isinstance(section, Mapping)
                or section.get("names") != expected_names
                or section.get("valid") != [1] * len(expected_names or [])
            ):
                errors.append(
                    f"{label}.{layout_field}.{category} "
                    "必须精确匹配 schema names 且全部有效"
                )
    expected_counts = {
        category: len(expected_layout.get(category, []))
        for category in _ACTIVE_PHYSICS_LAYOUT_CATEGORIES
    }
    for coverage_field in (
        "reset_state_coverage",
        "final_reset_state_coverage",
    ):
        coverage = history.get(coverage_field)
        if isinstance(coverage, Mapping):
            for category, expected in expected_counts.items():
                if coverage.get(category) != expected:
                    errors.append(
                        f"{label}.{coverage_field}.{category} 与 schema layout 不一致"
                    )

    schema_physics_dt = physics_config.get("physics_dt")
    if (
        type(schema_physics_dt) not in {int, float}
        or isinstance(schema_physics_dt, bool)
        or history.get("physics_dt") != schema_physics_dt
    ):
        errors.append(f"{label}.physics_dt 必须与 schema physics_config 一致")
    data_fps = recording.get("data_fps")
    control_dt = physics_config.get("control_dt")
    if (
        type(data_fps) not in {int, float}
        or isinstance(data_fps, bool)
        or not math.isfinite(float(data_fps))
        or float(data_fps) <= 0.0
        or type(control_dt) not in {int, float}
        or isinstance(control_dt, bool)
        or not math.isclose(
            float(control_dt),
            1.0 / float(data_fps),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        errors.append("recording.data_fps 必须与 schema control_dt 一致")
    return errors


def _report_execution_evidence_errors(
    report: Mapping[str, Any],
    *,
    expected_atols: Mapping[str, float],
    expected_physics_config: Mapping[str, Any],
    trajectory_input_mode: str | None,
    trajectory_step_count: int | None,
) -> list[str]:
    """Reject reports that only claim success without numeric proof."""

    errors: list[str] = []
    if report.get("final_succeed") is not True:
        errors.append("final_succeed 必须为 true")
    for field in ("first_divergence", "first_outcome_mismatch"):
        if field not in report or report.get(field) is not None:
            errors.append(f"{field} 必须显式为 null")

    field_atols = report.get("field_atols")
    max_errors = report.get("max_errors")
    expected_names = set(expected_atols)
    if not isinstance(field_atols, Mapping) or set(field_atols) != expected_names:
        errors.append("field_atols 必须精确覆盖 recording schema 物理字段")
    if not isinstance(max_errors, Mapping) or set(max_errors) != expected_names:
        errors.append("max_errors 必须精确覆盖 recording schema 物理字段")
    if isinstance(field_atols, Mapping) and isinstance(max_errors, Mapping):
        for name in sorted(expected_names):
            tolerance = field_atols.get(name)
            maximum_error = max_errors.get(name)
            if (
                type(tolerance) not in {int, float}
                or isinstance(tolerance, bool)
                or not math.isfinite(float(tolerance))
                or float(tolerance) < 0.0
                or float(tolerance) != expected_atols[name]
            ):
                errors.append(f"field_atols.{name} 与 recording schema 不一致")
                continue
            if (
                type(maximum_error) not in {int, float}
                or isinstance(maximum_error, bool)
                or not math.isfinite(float(maximum_error))
                or float(maximum_error) < 0.0
                or float(maximum_error) > float(tolerance)
            ):
                errors.append(f"max_errors.{name} 超出 field_atols")

    physics_match = report.get("recorded_physics_configuration_match")
    if trajectory_input_mode in {
        RECORDED_DRIVE_TARGET_INPUT_MODE,
        DETACHED_HAND_ACTION_INPUT_MODE,
    }:
        if not isinstance(physics_match, Mapping):
            errors.append("recorded_physics_configuration_match 必须为 object")
        else:
            checked_fields = physics_match.get("checked_fields")
            checked_count = physics_match.get("checked_field_count")
            expected_fields = sorted(expected_physics_config)
            if physics_match.get("matched") is not True:
                errors.append("recorded physics configuration 未匹配")
            if physics_match.get("unrecorded_static_dynamics_fingerprint") is not True:
                errors.append("recorded physics configuration 指纹声明无效")
            if (
                not isinstance(checked_fields, list)
                or checked_fields != expected_fields
                or checked_fields != sorted(set(checked_fields))
                or type(checked_count) is not int
                or checked_count != len(checked_fields)
                or checked_count <= 0
            ):
                errors.append("recorded physics configuration 字段覆盖不完整")
            if not _json_values_equal(
                physics_match.get("recorded_values"),
                expected_physics_config,
            ):
                errors.append("recorded physics configuration values 与 schema 不一致")
            if not _json_values_equal(
                physics_match.get("runtime_values"),
                expected_physics_config,
            ):
                errors.append(
                    "runtime physics configuration values 与 recording 不一致"
                )
    elif physics_match is not None:
        errors.append("legacy raw-action report 的 physics configuration 必须为 null")
    expected_steps = report.get("steps_expected")
    executed_steps = report.get("steps_executed")
    if (
        trajectory_step_count is None
        or trajectory_step_count <= 0
        or type(expected_steps) is not int
        or type(executed_steps) is not int
        or expected_steps != trajectory_step_count
        or executed_steps != trajectory_step_count
    ):
        errors.append("steps_expected/steps_executed 必须等于实际正数轨迹长度")
    return errors


def _report_identity_errors(
    report: Mapping[str, Any],
    *,
    recorded_git: Mapping[str, Any] | None,
    runtime_identity: Mapping[str, Any] | None,
    source_identity: Any,
) -> list[str]:
    """Cross-bind report source/runtime identity to the recording manifest."""

    errors: list[str] = []
    report_recorded = report.get("recorded_git")
    current = report.get("current_git")
    if recorded_git is None or not _json_values_equal(report_recorded, recorded_git):
        errors.append("report.recorded_git 必须与 manifest.software.git 完全一致")
    if runtime_identity is None or not _json_values_equal(
        report.get("runtime_replay_identity"),
        runtime_identity,
    ):
        errors.append("report.runtime_replay_identity 必须与 manifest 完全一致")

    if source_identity == CLEAN_COMMIT_SOURCE_IDENTITY:
        if (
            not isinstance(report_recorded, Mapping)
            or report_recorded.get("dirty") is not False
            or not _is_git_object_id(report_recorded.get("commit"))
            or report_recorded.get("changed_files") != []
            or not isinstance(current, Mapping)
            or current.get("dirty") is not False
            or current.get("changed_files") != []
            or current.get("commit") != report_recorded.get("commit")
        ):
            errors.append(
                "clean_commit_match 要求 recorded/current 为同一 clean commit"
            )
    elif source_identity == DIRTY_CONTENT_SOURCE_IDENTITY:
        if (
            not isinstance(report_recorded, Mapping)
            or report_recorded.get("dirty") is not True
            or not isinstance(current, Mapping)
            or current.get("dirty") is not True
            or current.get("commit") != report_recorded.get("commit")
            or current.get("changed_files") != report_recorded.get("changed_files")
            or current.get("worktree_content_sha256")
            != report_recorded.get("worktree_content_sha256")
        ):
            errors.append(
                "dirty_content_match 要求 current_git 与 recorded_git 内容一致"
            )
    return errors


def _contact_signature_overlay_errors(
    report: Mapping[str, Any],
) -> list[str]:
    """Validate versioned, deterministic legacy-signature provenance."""

    overlay = report.get(CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD)
    if overlay is None:
        return []
    errors: list[str] = []
    if not isinstance(overlay, Mapping):
        return [f"{CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD} 必须为 object"]
    expected_keys = {
        "format",
        "version",
        "algorithm",
        "source_open_mode",
        "source_files_mutated",
        "segments",
        "segment_count",
        "signature_frames_reconstructed",
        "exact_zero_padding_rows_removed",
        "digest_sha256",
    }
    if set(overlay) != expected_keys:
        errors.append("contact signature overlay 字段集合不精确")
    if overlay.get("format") != CONTACT_SIGNATURE_OVERLAY_FORMAT:
        errors.append("contact signature overlay format 不匹配")
    if overlay.get("version") != CONTACT_SIGNATURE_OVERLAY_VERSION:
        errors.append(
            "contact signature overlay version 必须为 "
            f"{CONTACT_SIGNATURE_OVERLAY_VERSION}"
        )
    if overlay.get("algorithm") != CONTACT_SIGNATURE_OVERLAY_ALGORITHM:
        errors.append("contact signature overlay algorithm 不匹配")
    if overlay.get("source_open_mode") != "read_only":
        errors.append("contact signature overlay 必须只读打开来源")
    if overlay.get("source_files_mutated") is not False:
        errors.append("contact signature overlay 不得修改来源文件")

    raw_segments = overlay.get("segments")
    segments = raw_segments if isinstance(raw_segments, list) else []
    if not segments:
        errors.append("contact signature overlay segments 必须非空")
    if overlay.get("segment_count") != len(segments):
        errors.append("contact signature overlay segment_count 不一致")
    expected_segment_keys = {
        "segment_id",
        "signature_frames_checked",
        "signature_frames_reconstructed",
        "control_frames_reconstructed",
        "substep_frames_reconstructed",
        "exact_zero_padding_rows_removed",
        "coordinates_sha256",
        "raw_input_sha256",
        "canonical_output_sha256",
    }
    segment_ids: list[str] = []
    reconstructed_total = 0
    zero_rows_total = 0
    for index, segment in enumerate(segments):
        label = f"contact signature overlay segment_{index}"
        if not isinstance(segment, Mapping):
            errors.append(f"{label} 必须为 object")
            continue
        if set(segment) != expected_segment_keys:
            errors.append(f"{label} 字段集合不精确")
        segment_id = segment.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id:
            errors.append(f"{label}.segment_id 无效")
        else:
            segment_ids.append(segment_id)
        checked = segment.get("signature_frames_checked")
        reconstructed = segment.get("signature_frames_reconstructed")
        control = segment.get("control_frames_reconstructed")
        substep = segment.get("substep_frames_reconstructed")
        zero_rows = segment.get("exact_zero_padding_rows_removed")
        if type(checked) is not int or checked <= 0:
            errors.append(f"{label}.signature_frames_checked 必须为正整数")
        if type(reconstructed) is not int or reconstructed <= 0:
            errors.append(f"{label}.signature_frames_reconstructed 必须为正整数")
        if type(control) is not int or control < 0:
            errors.append(f"{label}.control_frames_reconstructed 无效")
        if type(substep) is not int or substep < 0:
            errors.append(f"{label}.substep_frames_reconstructed 无效")
        if (
            type(reconstructed) is int
            and type(control) is int
            and type(substep) is int
            and control + substep != reconstructed
        ):
            errors.append(f"{label} control/substep 重建数不一致")
        if (
            type(checked) is int
            and type(reconstructed) is int
            and reconstructed > checked
        ):
            errors.append(f"{label} 重建帧数超过检查帧数")
        if (
            type(zero_rows) is not int
            or zero_rows <= 0
            or (type(reconstructed) is int and zero_rows < reconstructed)
        ):
            errors.append(f"{label} 精确全零 padding 计数无效")
        for digest_name in (
            "coordinates_sha256",
            "raw_input_sha256",
            "canonical_output_sha256",
        ):
            if not _is_sha256(segment.get(digest_name)):
                errors.append(f"{label}.{digest_name} 必须为 SHA-256")
        if type(reconstructed) is int and reconstructed > 0:
            reconstructed_total += reconstructed
        if type(zero_rows) is int and zero_rows > 0:
            zero_rows_total += zero_rows
    if segment_ids != sorted(set(segment_ids)) or len(segment_ids) != len(segments):
        errors.append("contact signature overlay segment_id 必须排序且唯一")
    if overlay.get("signature_frames_reconstructed") != reconstructed_total:
        errors.append("contact signature overlay 重建帧总数不一致")
    if overlay.get("exact_zero_padding_rows_removed") != zero_rows_total:
        errors.append("contact signature overlay 全零 padding 总数不一致")

    digest = overlay.get("digest_sha256")
    if not _is_sha256(digest):
        errors.append("contact signature overlay digest_sha256 必须为 SHA-256")
    else:
        payload = {
            key: overlay[key]
            for key in expected_keys - {"digest_sha256"}
            if key in overlay
        }
        expected_digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if digest != expected_digest:
            errors.append("contact signature overlay digest_sha256 不匹配")

    source_identity = report.get("source_identity")
    reviewer_paths = report.get("source_reviewer_upgrade_paths")
    required_reviewer_paths = {
        "scripts/certify_physx_replay.py",
        "handson/handson/sim/isaaclab/physx_ground_truth.py",
    }
    if (
        report.get("portable") is not True
        or source_identity
        not in {
            CLEAN_COMMIT_REVIEWER_UPGRADE_SOURCE_IDENTITY,
            DIRTY_SNAPSHOT_REVIEWER_UPGRADE_SOURCE_IDENTITY,
        }
        or not isinstance(reviewer_paths, list)
        or not required_reviewer_paths.issubset(set(reviewer_paths))
    ):
        errors.append(
            "contact signature overlay 必须绑定 portable committed reviewer "
            "upgrade 及受控重建实现"
        )
    return errors


def _certification_report_blocker(
    demo_dir: str | Path,
    report: Mapping[str, Any],
    *,
    actual_inputs: Mapping[str, Any] | None = None,
    require_portable: bool,
) -> dict[str, Any] | None:
    """Validate one replay report under an explicit internal policy."""

    demo_path = Path(demo_dir)
    if report.get("status") == "failed":
        reason = str(report.get("reason") or "").strip()
        return {
            "code": "physical_replay_certification_failed",
            "detail": (
                f"连续 PhysX replay 认证未通过：{reason}，暂不上传。"
                if reason
                else "连续 PhysX replay 认证未通过，暂不上传。"
            ),
            "physical_replay_certification": dict(report),
        }

    errors: list[str] = []
    recording, requirement_blocker = _physical_replay_recording_requirement(demo_path)
    if requirement_blocker is not None:
        return requirement_blocker
    if recording is None:
        return {
            "code": "physical_replay_certification_invalid",
            "detail": "认证报告没有可绑定的 PhysX recording manifest 契约。",
            "physical_replay_certification": dict(report),
        }
    task_name = _manifest_task_name(demo_path)
    mode_blocker = _certification_mode_blocker(
        recording,
        task_name=task_name,
    )
    if mode_blocker is not None:
        return {
            **mode_blocker,
            "physical_replay_certification": dict(report),
        }
    contract_errors, _, trajectory_input_mode = _trajectory_contract_errors(
        recording,
        report,
        task_name=task_name,
    )
    errors.extend(contract_errors)
    manifest_recorded_git, runtime_identity, software_errors = (
        _recording_manifest_software(demo_path)
    )
    (
        expected_atols,
        expected_physics_config,
        expected_layout,
        schema_errors,
    ) = _recording_schema_report_contract(demo_path)
    trajectory_step_count, trajectory_step_errors = _recorded_trajectory_step_count(
        demo_path
    )
    errors.extend(software_errors)
    errors.extend(schema_errors)
    errors.extend(trajectory_step_errors)
    errors.extend(
        _schema_bound_origin_errors(
            recording.get("stable_recording_origin"),
            expected_layout=expected_layout,
            physics_config=expected_physics_config,
            recording=recording,
            label="recording.stable_recording_origin",
        )
    )
    if report.get("format") != REPORT_FORMAT:
        errors.append("format 不匹配")
    if type(report.get("version")) is not int or (
        report.get("version") != REPORT_VERSION
    ):
        errors.append(f"version 必须为 {REPORT_VERSION}")
    if report.get("status") != "passed":
        errors.append("status 必须为 passed")
    portable = report.get("portable")
    source_identity = report.get("source_identity")
    if type(portable) is not bool:
        errors.append("portable 必须为布尔值")
    elif portable:
        if source_identity not in PORTABLE_SOURCE_IDENTITIES:
            errors.append("portable=true 时 source_identity 必须是受支持的可迁移身份")
        elif source_identity == CLEAN_COMMIT_SOURCE_IDENTITY:
            if "source_snapshot_commit" in report:
                errors.append("clean commit 身份不能携带 source_snapshot_commit")
            if "source_reviewer_upgrade_paths" in report:
                errors.append("clean commit 身份不能携带 reviewer upgrade 路径")
        elif source_identity == CLEAN_COMMIT_REVIEWER_UPGRADE_SOURCE_IDENTITY:
            snapshot_commit = report.get("source_snapshot_commit")
            reviewer_paths = report.get("source_reviewer_upgrade_paths")
            recorded_git = report.get("recorded_git")
            current_git = report.get("current_git")
            if not _is_git_object_id(snapshot_commit):
                errors.append("source_snapshot_commit 必须是 Git object id")
            if (
                not isinstance(reviewer_paths, list)
                or not reviewer_paths
                or any(not isinstance(path, str) for path in reviewer_paths)
                or reviewer_paths != sorted(set(reviewer_paths))
            ):
                errors.append(
                    "source_reviewer_upgrade_paths 必须是非空、排序、去重的路径列表"
                )
            elif not set(reviewer_paths).issubset(
                SOURCE_REVIEWER_UPGRADE_ALLOWED_PATHS
            ):
                errors.append("source_reviewer_upgrade_paths 超出复核/发布实现白名单")
            if (
                not isinstance(recorded_git, Mapping)
                or recorded_git.get("dirty") is not False
                or not _is_git_object_id(recorded_git.get("commit"))
                or recorded_git.get("changed_files") != []
            ):
                errors.append(
                    "clean reviewer identity 要求 recorded_git 为 clean commit"
                )
            elif recorded_git.get("commit") != snapshot_commit:
                errors.append("source_snapshot_commit 必须等于 clean recording commit")
            if (
                not isinstance(current_git, Mapping)
                or current_git.get("dirty") is not False
                or not _is_git_object_id(current_git.get("commit"))
                or current_git.get("changed_files") != []
            ):
                errors.append("clean reviewer identity 要求 current_git 为 clean HEAD")
            elif current_git.get("commit") == snapshot_commit:
                errors.append("current_git.commit 必须包含 committed reviewer upgrade")
        else:
            snapshot_commit = report.get("source_snapshot_commit")
            reviewer_paths = report.get("source_reviewer_upgrade_paths")
            recorded_git = report.get("recorded_git")
            current_git = report.get("current_git")
            if not _is_git_object_id(snapshot_commit):
                errors.append("source_snapshot_commit 必须是 Git object id")
            if (
                not isinstance(reviewer_paths, list)
                or not reviewer_paths
                or any(not isinstance(path, str) for path in reviewer_paths)
                or reviewer_paths != sorted(set(reviewer_paths))
            ):
                errors.append(
                    "source_reviewer_upgrade_paths 必须是非空、排序、去重的路径列表"
                )
            elif not set(reviewer_paths).issubset(
                SOURCE_REVIEWER_UPGRADE_ALLOWED_PATHS
            ):
                errors.append("source_reviewer_upgrade_paths 超出复核/发布实现白名单")
            if (
                not isinstance(recorded_git, Mapping)
                or recorded_git.get("dirty") is not True
                or not _is_git_object_id(recorded_git.get("commit"))
                or not isinstance(recorded_git.get("changed_files"), list)
                or not recorded_git.get("changed_files")
                or any(
                    not isinstance(path, str) or not path
                    for path in recorded_git.get("changed_files", [])
                )
                or recorded_git.get("changed_files")
                != sorted(set(recorded_git.get("changed_files", [])))
                or not _is_sha256(recorded_git.get("worktree_content_sha256"))
            ):
                errors.append(
                    "snapshot reviewer identity 要求有效的 dirty recorded_git"
                )
            elif recorded_git.get("commit") == snapshot_commit:
                errors.append("source_snapshot_commit 不能等于 dirty recording base")
            if (
                not isinstance(current_git, Mapping)
                or current_git.get("dirty") is not False
                or not _is_git_object_id(current_git.get("commit"))
                or current_git.get("changed_files") != []
            ):
                errors.append(
                    "snapshot reviewer identity 要求 current_git 为 clean HEAD"
                )
            elif current_git.get("commit") == snapshot_commit:
                errors.append("current_git.commit 必须包含 committed reviewer upgrade")
            elif isinstance(recorded_git, Mapping) and current_git.get(
                "commit"
            ) == recorded_git.get("commit"):
                errors.append("current_git.commit 不能回退到 dirty recording base")
    else:
        if require_portable:
            errors.append("portable 必须为 true")
        if source_identity != DIRTY_CONTENT_SOURCE_IDENTITY:
            errors.append(
                "portable=false 时 source_identity 必须为 dirty_content_match"
            )
        if "source_snapshot_commit" in report:
            errors.append("portable=false 不能携带 source_snapshot_commit")
        if "source_reviewer_upgrade_paths" in report:
            errors.append("portable=false 不能携带 reviewer upgrade 路径")
    errors.extend(
        _report_identity_errors(
            report,
            recorded_git=manifest_recorded_git,
            runtime_identity=runtime_identity,
            source_identity=source_identity,
        )
    )
    errors.extend(_contact_signature_overlay_errors(report))
    errors.extend(
        _report_execution_evidence_errors(
            report,
            expected_atols=expected_atols,
            expected_physics_config=expected_physics_config,
            trajectory_input_mode=trajectory_input_mode,
            trajectory_step_count=trajectory_step_count,
        )
    )
    for field in (
        "task_success_replay",
        "trace_match",
        "transition_outcomes_match",
    ):
        if report.get(field) is not True:
            errors.append(f"{field} 必须为 true")
    if report.get("certification_mode") != CERTIFICATION_MODE:
        errors.append("certification_mode 必须是 fresh seeded PhysX 初态连续物理重执行")
    if report.get("initialization_mode") != INITIALIZATION_MODE:
        errors.append(
            "initialization_mode 必须从 fresh scene 执行相同 seeded reset 和稳定化"
        )
    restore_counts = report.get("initialization_restore_counts")
    if not isinstance(restore_counts, Mapping):
        errors.append("缺少 initialization_restore_counts object")
    elif set(restore_counts) != INITIALIZATION_RESTORE_COUNT_FIELDS:
        errors.append("initialization_restore_counts 必须包含完整的恢复计数字段")
    elif any(type(value) is not int or value < 0 for value in restore_counts.values()):
        errors.append("initialization_restore_counts 的值必须为非负整数")
    elif not any(value > 0 for value in restore_counts.values()):
        errors.append("initialization_restore_counts 未恢复任何物理实体")
    errors.extend(
        _restore_counts_cover_active_layout_errors(
            restore_counts,
            layout=recording.get("stable_recording_origin", {}).get(
                "active_physics_layout"
            )
            if isinstance(recording.get("stable_recording_origin"), Mapping)
            else None,
            label="initialization_restore_counts",
        )
    )
    if (
        report.get("initialization_restore_counts_source")
        != INITIALIZATION_RESTORE_COUNTS_SOURCE
    ):
        errors.append(
            "initialization_restore_counts_source 必须来自记录并重执行的"
            "稳定化末端边界写"
        )
    if report.get("initialization_readback_match") is not True:
        errors.append("initialization_readback_match 必须为 true")
    is_deferred_persistent = (
        recording.get("physical_replay_certification_mode")
        == DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE
    )
    if is_deferred_persistent:
        report_source_bindings = {
            "source_recorder_lifecycle": recording.get("recorder_lifecycle"),
            "source_process_reuse_enabled": recording.get(
                "source_process_reuse_enabled"
            ),
            "fresh_source_process_per_attempt": recording.get(
                "fresh_recorder_process_per_attempt_required"
            ),
            "fresh_scene_replay_required": recording.get("fresh_scene_replay_required"),
            "collection_seed_source": recording.get("collection_seed_source"),
            "episode_reset_ordinal": recording.get("episode_reset_ordinal"),
            "episode_seed_derivation": recording.get("episode_seed_derivation"),
            "source_recorder_pid": recording.get("recorder_pid"),
        }
        for field, expected_value in report_source_bindings.items():
            actual_value = report.get(field)
            if type(actual_value) is not type(expected_value) or (
                actual_value != expected_value
            ):
                errors.append(f"report.{field} 必须与 manifest 来源契约一致")
    stable_origin = recording.get("stable_recording_origin")
    recorded_natural_settle = bool(
        isinstance(stable_origin, Mapping)
        and _is_natural_settle_then_zero_history(stable_origin)
    )
    initialization_history_required = isinstance(stable_origin, Mapping)
    if initialization_history_required:
        assert isinstance(stable_origin, Mapping)
        if report.get("initialization_history_match") is not True:
            errors.append("initialization_history_match 必须为 true")
        replayed_history = report.get("replayed_stable_origin_history")
        if not isinstance(replayed_history, Mapping):
            errors.append("缺少 replayed_stable_origin_history object")
        else:
            errors.extend(
                _schema_bound_origin_errors(
                    replayed_history,
                    expected_layout=expected_layout,
                    physics_config=expected_physics_config,
                    recording=recording,
                    label="replayed_stable_origin_history",
                )
            )
            replayed_natural_settle = _is_natural_settle_then_zero_history(
                replayed_history
            )
            if replayed_natural_settle != recorded_natural_settle:
                errors.append("复核初始化策略与 manifest 不一致")
            if recorded_natural_settle:
                errors.extend(
                    _natural_settle_adaptive_history_errors(
                        stable_origin,
                        label="recording.stable_recording_origin",
                    )
                )
                errors.extend(
                    _natural_settle_adaptive_history_errors(
                        replayed_history,
                        label="replayed_stable_origin_history",
                    )
                )
            errors.extend(
                _physical_origin_geometry_history_errors(
                    replayed_history,
                    label="replayed_stable_origin_history",
                )
            )
            for field in _DEFERRED_CANONICAL_INITIALIZATION_HISTORY_FIELDS:
                if field not in stable_origin:
                    errors.append(f"recording.stable_recording_origin.{field} 缺失")
                    continue
                if field not in replayed_history:
                    errors.append(f"replayed_stable_origin_history.{field} 缺失")
                    continue
                if not _initialization_history_field_matches(
                    field,
                    stable_origin[field],
                    replayed_history[field],
                    natural_settle=recorded_natural_settle,
                ):
                    errors.append(
                        f"replayed_stable_origin_history.{field} 与 manifest 不一致"
                    )
    if report.get("post_stabilization_snapshot_restore_performed") is not False:
        errors.append("post_stabilization_snapshot_restore_performed 必须为 false")
    physics_steps_before_action = report.get(
        "physics_steps_between_initialization_readback_and_action_0"
    )
    if type(physics_steps_before_action) is not int or (
        physics_steps_before_action != 0
    ):
        errors.append(
            "physics_steps_between_initialization_readback_and_action_0 必须为 0"
        )
    for field in (
        "state_writes_performed",
        "initialization_state_writes_performed",
    ):
        if report.get(field) is not True:
            errors.append(f"{field} 必须为 true")
    detached_hand_inputs = (
        trajectory_input_mode == DETACHED_HAND_ACTION_INPUT_MODE
    )
    for field in (
        "trajectory_state_writes_performed",
        "state_writes_after_action_start",
    ):
        expected = True if detached_hand_inputs else False
        if report.get(field) is not expected:
            errors.append(f"{field} 必须为 {str(expected).lower()}")

    if trajectory_input_mode == RECORDED_DRIVE_TARGET_INPUT_MODE:
        for field in (
            "raw_commands_installed",
            "recorded_drive_targets_applied",
        ):
            if report.get(field) is not True:
                errors.append(f"{field} 必须为 true")
        for field in (
            "ik_recomputed",
            "raw_environment_step_called",
        ):
            if report.get(field) is not False:
                errors.append(f"{field} 必须为 false")
        count_values: dict[str, int] = {}
        for field in _RECORDED_DRIVE_TARGET_COUNT_FIELDS:
            value = report.get(field)
            if type(value) is not int or value <= 0:
                errors.append(f"{field} 必须为正整数")
            else:
                count_values[field] = value
        for expected_field, executed_field in (
            (
                "recorded_drive_target_substeps_expected",
                "recorded_drive_target_substeps_executed",
            ),
            (
                "recorded_drive_target_writes_expected",
                "recorded_drive_target_writes_executed",
            ),
        ):
            if (
                expected_field in count_values
                and executed_field in count_values
                and count_values[expected_field] != count_values[executed_field]
            ):
                errors.append(f"{executed_field} 必须等于 {expected_field}")
    elif trajectory_input_mode == LEGACY_RAW_ACTION_INPUT_MODE:
        if report.get("raw_environment_step_called") is not True:
            errors.append("raw_environment_step_called 必须为 true")
        if report.get("ik_recomputed") is not True:
            errors.append("ik_recomputed 必须为 true")
        if report.get("recorded_drive_targets_applied") is not False:
            errors.append("recorded_drive_targets_applied 必须为 false")
        for field in _RECORDED_DRIVE_TARGET_COUNT_FIELDS:
            value = report.get(field)
            if type(value) is not int or value != 0:
                errors.append(f"{field} 必须为整数 0")
    elif trajectory_input_mode == DETACHED_HAND_ACTION_INPUT_MODE:
        if report.get("raw_environment_step_called") is not True:
            errors.append("raw_environment_step_called 必须为 true")
        if report.get("ik_recomputed") is not False:
            errors.append("ik_recomputed 必须为 false")
        if report.get("recorded_drive_targets_applied") is not False:
            errors.append("recorded_drive_targets_applied 必须为 false")
        for field in _RECORDED_DRIVE_TARGET_COUNT_FIELDS:
            value = report.get(field)
            if type(value) is not int or value != 0:
                errors.append(f"{field} 必须为整数 0")
    expected = report.get("steps_expected")
    executed = report.get("steps_executed")
    if type(expected) is not int or type(executed) is not int:
        errors.append("steps_expected 和 steps_executed 必须为整数")
    elif expected != executed:
        errors.append(f"执行步数不一致：{executed} != {expected}")

    certified_inputs = report.get("certified_inputs")
    if not isinstance(certified_inputs, Mapping):
        errors.append("缺少 certified_inputs object")
    else:
        if certified_inputs.get("algorithm") != INPUT_IDENTITY_ALGORITHM:
            errors.append(
                f"certified_inputs.algorithm 必须为 {INPUT_IDENTITY_ALGORITHM}"
            )
        for field in (
            "replay_manifest_sha256",
            "recording_schema_sha256",
            "replay_zarr_sha256",
        ):
            value = certified_inputs.get(field)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                errors.append(f"certified_inputs.{field} 必须为 SHA-256")
        for field in ("replay_zarr_file_count", "replay_zarr_bytes"):
            value = certified_inputs.get(field)
            if type(value) is not int or value < 0:
                errors.append(f"certified_inputs.{field} 必须为非负整数")
    if errors:
        return {
            "code": "physical_replay_certification_invalid",
            "detail": "连续 PhysX replay 认证未通过上传门禁：" + "；".join(errors),
            "physical_replay_certification": dict(report),
        }

    if actual_inputs is None:
        try:
            actual_inputs = capture_certified_inputs(demo_path)
        except (OSError, UnicodeError, OverflowError) as exc:
            return {
                "code": "physical_replay_certification_stale",
                "detail": f"无法复核连续 PhysX replay 的认证输入：{exc}",
                "physical_replay_certification": dict(report),
            }
    mismatch = [
        field
        for field in actual_inputs
        if certified_inputs.get(field) != actual_inputs[field]
    ]
    if mismatch:
        return {
            "code": "physical_replay_certification_stale",
            "detail": (
                "连续 PhysX replay 认证与当前输入不一致，需重新认证："
                + "、".join(mismatch)
            ),
            "physical_replay_certification": dict(report),
            "physical_replay_certified_inputs_actual": dict(actual_inputs),
        }
    return None


def certification_report_blocker(
    demo_dir: str | Path,
    report: Mapping[str, Any],
    *,
    actual_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Apply the strict publication/upload gate to one replay report."""

    return _certification_report_blocker(
        demo_dir,
        report,
        actual_inputs=actual_inputs,
        require_portable=True,
    )


def local_physical_replay_report_blocker(
    demo_dir: str | Path,
    report: Mapping[str, Any],
    *,
    actual_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate local physics reproduction without granting portability.

    This explicit API is used only while executing deferred certification.
    A dirty source may prove an exact local physical match, but publication
    and upload still pass through :func:`certification_report_blocker` and
    therefore continue to require ``portable=true``.
    """

    return _certification_report_blocker(
        demo_dir,
        report,
        actual_inputs=actual_inputs,
        require_portable=False,
    )


def certification_upload_blocker(
    demo_dir: str | Path,
    *,
    actual_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate the conventional latest single-run certification report."""

    demo_path = Path(demo_dir)
    recording, requirement_blocker = _physical_replay_recording_requirement(demo_path)
    if requirement_blocker is not None:
        return requirement_blocker
    if recording is None:
        return None

    certification_path = demo_path / REPORT_NAME
    if not certification_path.is_file():
        return {
            "code": "physical_replay_certification_missing",
            "detail": "该轨迹要求连续 PhysX replay 认证，但认证文件尚未生成，暂不上传。",
        }
    try:
        report = json.loads(certification_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "code": "physical_replay_certification_invalid",
            "detail": f"连续 PhysX replay 认证文件无法读取：{exc}",
        }
    if not isinstance(report, Mapping):
        return {
            "code": "physical_replay_certification_invalid",
            "detail": "连续 PhysX replay 认证文件必须是 JSON object，暂不上传。",
        }
    return certification_report_blocker(
        demo_path,
        report,
        actual_inputs=actual_inputs,
    )


def fresh_process_acceptance_upload_blocker(
    demo_dir: str | Path,
    *,
    actual_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate independent archived reports and the publication proof."""

    demo_path = Path(demo_dir)
    recording, requirement_blocker = _physical_replay_recording_requirement(demo_path)
    if requirement_blocker is not None:
        return requirement_blocker
    if recording is None:
        return None
    task_name = _manifest_task_name(demo_path)
    mode_blocker = _certification_mode_blocker(
        recording,
        task_name=task_name,
    )
    if mode_blocker is not None:
        return mode_blocker
    mode = str(recording.get("physical_replay_certification_mode") or "").strip()
    if mode == DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE:
        return deferred_physical_replay_acceptance_upload_blocker(
            demo_path,
            actual_inputs=actual_inputs,
        )
    if mode not in FRESH_PROCESS_CERTIFICATION_MODE_WHITELIST:
        return {
            "code": "physical_replay_certification_mode_invalid",
            "detail": f"fresh-process 门禁不接受认证模式 {mode!r}。",
        }
    _, transition_contract, trajectory_input_mode = _trajectory_contract_errors(
        recording,
        task_name=task_name,
    )
    required_runs = _required_certification_runs(recording)
    acceptance_path = demo_path / FRESH_PROCESS_ACCEPTANCE_NAME
    try:
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return {
            "code": "fresh_process_acceptance_missing",
            "detail": ("该 PhysX 轨迹尚无逐条 fresh-process 多次认证证明，暂不上传。"),
        }
    if not isinstance(acceptance, Mapping):
        return {
            "code": "fresh_process_acceptance_invalid",
            "detail": "fresh-process 多次认证证明必须为 JSON object。",
        }

    errors: list[str] = []
    if acceptance.get("format") != FRESH_PROCESS_ACCEPTANCE_FORMAT:
        errors.append("format 不匹配")
    if acceptance.get("version") != FRESH_PROCESS_ACCEPTANCE_VERSION:
        errors.append(f"version 必须为 {FRESH_PROCESS_ACCEPTANCE_VERSION}")
    if acceptance.get("status") != "passed":
        errors.append("status 必须为 passed")
    if acceptance.get("certification_runs_required") != required_runs:
        errors.append(f"必须要求 {required_runs} 次独立认证")
    if acceptance.get("certification_runs_passed") != required_runs:
        errors.append(f"必须通过 {required_runs} 次独立认证")
    for manifest_field in (
        "fresh_recorder_process_per_episode_required",
        "fresh_recorder_process_per_attempt_required",
        "publish_only_after_fresh_certification",
    ):
        if recording.get(manifest_field) is not True:
            errors.append(f"recording.{manifest_field} 必须为 true")
    for field in (
        "fresh_recorder_process_per_episode",
        "fresh_recorder_process_per_attempt",
        "fresh_certifier_process_per_run",
        "published_only_after_certification",
    ):
        if acceptance.get(field) is not True:
            errors.append(f"{field} 必须为 true")
    if acceptance.get("physical_transition_contract") != transition_contract:
        errors.append("acceptance physical_transition_contract 与 manifest 不一致")
    if acceptance.get("trajectory_input_mode") != trajectory_input_mode:
        errors.append("acceptance trajectory_input_mode 与 manifest 不一致")

    if actual_inputs is None:
        try:
            actual_inputs = capture_certified_inputs(demo_path)
        except (OSError, UnicodeError, OverflowError) as exc:
            errors.append(f"无法复核认证输入：{exc}")
            actual_inputs = None
    if actual_inputs is not None and (
        acceptance.get("certified_inputs") != actual_inputs
    ):
        errors.append("acceptance certified_inputs 与当前轨迹不一致")

    recorder = acceptance.get("recorder")
    recorder_pid = recorder.get("pid") if isinstance(recorder, Mapping) else None
    if type(recorder_pid) is not int or recorder_pid <= 0:
        errors.append("recorder.pid 必须为正整数")
    if acceptance.get("recorder_pid") != recorder_pid:
        errors.append("recorder_pid 必须与 recorder.pid 一致")
    if recorder_pid != recording.get("recorder_pid"):
        errors.append("recorder.pid 必须与 recording.recorder_pid 一致")
    recorder_started = (
        _parse_process_timestamp(recorder.get("started_at"))
        if isinstance(recorder, Mapping)
        else None
    )
    recorder_finished = (
        _parse_process_timestamp(recorder.get("finished_at"))
        if isinstance(recorder, Mapping)
        else None
    )
    recorder_returncode = (
        recorder.get("returncode") if isinstance(recorder, Mapping) else None
    )
    if type(recorder_returncode) is not int or recorder_returncode != 0:
        errors.append("recorder.returncode 必须为整数 0")
    if (
        recorder_started is None
        or recorder_finished is None
        or recorder_finished < recorder_started
    ):
        errors.append("recorder 进程时间无效")

    runs = acceptance.get("certification_runs")
    run_rows = runs if isinstance(runs, list) else []
    if len(run_rows) != required_runs:
        errors.append(f"certification_runs 必须恰好有 {required_runs} 条")
    certifier_pids: list[int] = []
    contact_signature_overlay_digests: list[str | None] = []
    previous_finished = recorder_finished
    reports_root = (demo_path / FRESH_PROCESS_REPORT_DIRECTORY).resolve()
    for expected_index, run in enumerate(run_rows, start=1):
        label = f"run_{expected_index}"
        if not isinstance(run, Mapping):
            errors.append(f"{label} 必须为 object")
            continue
        if run.get("run") != expected_index:
            errors.append(f"{label}.run 序号不连续")
        pid = run.get("pid")
        if type(pid) is not int or pid <= 0:
            errors.append(f"{label}.pid 必须为正整数")
        else:
            certifier_pids.append(pid)
        if run.get("status") != "passed":
            errors.append(f"{label}.status 必须为 passed")
        returncode = run.get("returncode")
        if type(returncode) is not int or returncode != 0:
            errors.append(f"{label}.returncode 必须为整数 0")
        started_at = _parse_process_timestamp(run.get("started_at"))
        finished_at = _parse_process_timestamp(run.get("finished_at"))
        if (
            started_at is None
            or finished_at is None
            or finished_at < started_at
            or (previous_finished is not None and started_at < previous_finished)
        ):
            errors.append(f"{label} 进程时间无效或与上一进程重叠")
        if finished_at is not None:
            previous_finished = finished_at
        report_sha = run.get("report_sha256")
        if (
            not isinstance(report_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", report_sha) is None
        ):
            errors.append(f"{label}.report_sha256 必须为 SHA-256")
            continue
        relative_report = run.get("report")
        if not isinstance(relative_report, str) or not relative_report:
            errors.append(f"{label}.report 路径无效")
            continue
        report_path = demo_path / relative_report
        try:
            resolved_report = report_path.resolve(strict=True)
            resolved_report.relative_to(reports_root)
            resolved_report.relative_to(demo_path.resolve())
        except (OSError, ValueError):
            errors.append(f"{label}.report 必须位于认证归档目录内")
            continue
        if report_path.is_symlink() or not resolved_report.is_file():
            errors.append(f"{label}.report 必须为普通文件")
            continue
        if _sha256_file(resolved_report) != report_sha:
            errors.append(f"{label}.report_sha256 与归档报告不一致")
            continue
        try:
            archived_report = json.loads(resolved_report.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors.append(f"{label}.report 无法读取")
            continue
        if not isinstance(archived_report, Mapping):
            errors.append(f"{label}.report 必须为 JSON object")
            continue
        archived_overlay = archived_report.get(CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD)
        contact_signature_overlay_digests.append(
            archived_overlay.get("digest_sha256")
            if isinstance(archived_overlay, Mapping)
            else None
        )
        if run.get("certified_inputs") != archived_report.get("certified_inputs"):
            errors.append(f"{label}.certified_inputs 与归档报告不一致")
        blocker = certification_report_blocker(
            demo_path,
            archived_report,
            actual_inputs=actual_inputs,
        )
        if blocker is not None:
            errors.append(
                f"{label}.report 未通过单次认证门禁："
                + str(blocker.get("detail") or blocker.get("code"))
            )

    declared_certifier_pids = acceptance.get("certifier_pids")
    if declared_certifier_pids != certifier_pids:
        errors.append("certifier_pids 必须与逐次认证记录一致")
    if len(set(certifier_pids)) != required_runs:
        errors.append(f"必须来自 {required_runs} 个不同 certifier pid")
    if type(recorder_pid) is int and recorder_pid in certifier_pids:
        errors.append("recorder 与 certifier 必须是不同进程")
    if (
        len(contact_signature_overlay_digests) == required_runs
        and len(set(contact_signature_overlay_digests)) != 1
    ):
        errors.append("三次认证的 contact signature overlay digest 必须一致")
    if errors:
        return {
            "code": "fresh_process_acceptance_invalid",
            "detail": "fresh-process 多次认证证明无效：" + "；".join(errors),
            "fresh_process_acceptance": dict(acceptance),
        }
    return None


def deferred_physical_replay_acceptance_upload_blocker(
    demo_dir: str | Path,
    *,
    actual_inputs: Mapping[str, Any] | None = None,
    require_published: bool = True,
    require_publication_commit: bool = True,
) -> dict[str, Any] | None:
    """Validate deferred post-collection replay certification proofs."""

    demo_path = Path(demo_dir)
    recording, requirement_blocker = _physical_replay_recording_requirement(demo_path)
    if requirement_blocker is not None:
        return requirement_blocker
    if recording is None:
        return None
    mode_blocker = _certification_mode_blocker(
        recording,
        task_name=_manifest_task_name(demo_path),
    )
    if mode_blocker is not None:
        return mode_blocker
    mode = recording.get("physical_replay_certification_mode")
    if mode != DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE:
        return {
            "code": "physical_replay_certification_mode_invalid",
            "detail": f"deferred 门禁不接受认证模式 {mode!r}。",
        }
    required_runs = _required_certification_runs(recording)
    acceptance_path = demo_path / FRESH_PROCESS_ACCEPTANCE_NAME
    try:
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return {
            "code": "deferred_physical_replay_acceptance_missing",
            "detail": ("该 PhysX 轨迹尚无采集后统一复核证明，暂不上传。"),
        }
    if not isinstance(acceptance, Mapping):
        return {
            "code": "deferred_physical_replay_acceptance_invalid",
            "detail": "采集后统一复核证明必须为 JSON object。",
        }

    session_contract, session_contract_errors = _ordered_session_manifest_contract(
        demo_path, recording
    )
    errors: list[str] = list(session_contract_errors)
    if acceptance.get("format") != DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_FORMAT:
        errors.append("format 不匹配")
    if acceptance.get("version") != DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_VERSION:
        errors.append(f"version 必须为 {DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_VERSION}")
    if acceptance.get("mode") != DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE:
        errors.append("mode 必须为 deferred_post_collection")
    if acceptance.get("status") != "passed":
        errors.append("status 必须为 passed")
    if acceptance.get("certification_runs_required") != required_runs:
        errors.append(f"必须要求 {required_runs} 次统一复核")
    if acceptance.get("certification_runs_passed") != required_runs:
        errors.append(f"必须通过 {required_runs} 次统一复核")
    expected_published = bool(require_published)
    if (
        acceptance.get("published_after_post_collection_certification")
        is not expected_published
    ):
        errors.append(
            "published_after_post_collection_certification 必须为 "
            f"{str(expected_published).lower()}"
        )
    acceptance_source_bindings = {
        "persistent_source_process_for_collection_batch": recording.get(
            "source_process_reuse_enabled"
        ),
        "fresh_source_process_per_episode": recording.get(
            "fresh_recorder_process_per_episode_required"
        ),
        "fresh_certifier_process_per_run": recording.get(
            "fresh_certifier_process_per_run_required"
        ),
        "recorder_lifecycle": recording.get("recorder_lifecycle"),
        "fresh_scene_replay_required": recording.get("fresh_scene_replay_required"),
        "fresh_certifier_process_per_run_required": recording.get(
            "fresh_certifier_process_per_run_required"
        ),
    }
    for field, expected_value in acceptance_source_bindings.items():
        actual_value = acceptance.get(field)
        if type(actual_value) is not type(expected_value) or (
            actual_value != expected_value
        ):
            errors.append(f"{field} 必须与 recording 来源契约一致")
    errors.extend(
        _ordered_session_acceptance_contract_errors(
            demo_path,
            recording,
            acceptance,
            session_contract,
        )
    )
    errors.extend(
        _ordered_session_history_errors(
            demo_path,
            acceptance,
            session_contract,
            task_name=_manifest_task_name(demo_path),
        )
    )

    manifest_recorder_pid = recording.get("recorder_pid")
    for field in ("source_recorder_pid", "recorder_pid"):
        if type(acceptance.get(field)) is not int or (
            acceptance.get(field) != manifest_recorder_pid
        ):
            errors.append(f"{field} 必须与 recording.recorder_pid 一致")

    # Ordered-session publication is an integrity boundary.  Recompute this
    # member even when a caller supplies a cached identity; the root commit
    # separately recomputes every sibling below.
    try:
        actual_inputs = capture_certified_inputs(demo_path)
    except (OSError, UnicodeError, OverflowError) as exc:
        errors.append(f"无法复核认证输入：{exc}")
        actual_inputs = None
    if actual_inputs is not None and (
        acceptance.get("certified_inputs") != actual_inputs
    ):
        errors.append("certified_inputs 与当前轨迹不一致")

    runs = acceptance.get("certification_runs")
    run_rows = runs if isinstance(runs, list) else []
    if len(run_rows) != required_runs:
        errors.append(f"certification_runs 必须恰好有 {required_runs} 条")
    certifier_pids: list[int] = []
    contact_signature_overlay_digests: list[str | None] = []
    reports_root = (demo_path / FRESH_PROCESS_REPORT_DIRECTORY).resolve()
    for expected_index, run in enumerate(run_rows, start=1):
        label = f"run_{expected_index}"
        if not isinstance(run, Mapping):
            errors.append(f"{label} 必须为 object")
            continue
        if run.get("run") != expected_index:
            errors.append(f"{label}.run 序号不连续")
        pid = run.get("pid")
        if type(pid) is not int or pid <= 0:
            errors.append(f"{label}.pid 必须为正整数")
        else:
            certifier_pids.append(pid)
        if run.get("status") != "passed":
            errors.append(f"{label}.status 必须为 passed")
        if run.get("session_episode_count") != acceptance.get("session_episode_count"):
            errors.append(f"{label}.session_episode_count 必须与 session proof 一致")
        returncode = run.get("returncode")
        if type(returncode) is not int or returncode != 0:
            errors.append(f"{label}.returncode 必须为整数 0")
        started_at = _parse_process_timestamp(run.get("started_at"))
        finished_at = _parse_process_timestamp(run.get("finished_at"))
        if started_at is None or finished_at is None or finished_at < started_at:
            errors.append(f"{label} 进程时间无效")
        report_sha = run.get("report_sha256")
        if (
            not isinstance(report_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", report_sha) is None
        ):
            errors.append(f"{label}.report_sha256 必须为 SHA-256")
            continue
        relative_report = run.get("report")
        if not isinstance(relative_report, str) or not relative_report:
            errors.append(f"{label}.report 路径无效")
            continue
        report_path = demo_path / relative_report
        try:
            resolved_report = report_path.resolve(strict=True)
            resolved_report.relative_to(reports_root)
            resolved_report.relative_to(demo_path.resolve())
        except (OSError, ValueError):
            errors.append(f"{label}.report 必须位于认证归档目录内")
            continue
        if report_path.is_symlink() or not resolved_report.is_file():
            errors.append(f"{label}.report 必须为普通文件")
            continue
        if _sha256_file(resolved_report) != report_sha:
            errors.append(f"{label}.report_sha256 与归档报告不一致")
            continue
        try:
            archived_report = json.loads(resolved_report.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors.append(f"{label}.report 无法读取")
            continue
        if not isinstance(archived_report, Mapping):
            errors.append(f"{label}.report 必须为 JSON object")
            continue
        archived_overlay = archived_report.get(CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD)
        contact_signature_overlay_digests.append(
            archived_overlay.get("digest_sha256")
            if isinstance(archived_overlay, Mapping)
            else None
        )
        if run.get("certified_inputs") != archived_report.get("certified_inputs"):
            errors.append(f"{label}.certified_inputs 与归档报告不一致")
        blocker = certification_report_blocker(
            demo_path,
            archived_report,
            actual_inputs=actual_inputs,
        )
        if blocker is not None:
            errors.append(
                f"{label}.report 未通过单次认证门禁："
                + str(blocker.get("detail") or blocker.get("code"))
            )
        errors.extend(
            _ordered_session_report_errors(
                archived_report,
                acceptance=acceptance,
                session_contract=session_contract,
                label=label,
            )
        )

    declared_certifier_pids = acceptance.get("certifier_pids")
    if declared_certifier_pids != certifier_pids:
        errors.append("certifier_pids 必须与逐次认证记录一致")
    if len(set(certifier_pids)) != required_runs:
        errors.append(f"必须来自 {required_runs} 个不同 certifier pid")
    if type(manifest_recorder_pid) is int and manifest_recorder_pid in certifier_pids:
        errors.append("persistent recorder 与 certifier 必须是不同进程")
    if (
        len(contact_signature_overlay_digests) == required_runs
        and len(set(contact_signature_overlay_digests)) != 1
    ):
        errors.append("三次认证的 contact signature overlay digest 必须一致")
    errors.extend(
        _ordered_session_publication_commit_errors(
            demo_path,
            acceptance,
            session_contract,
            require_publication_commit=require_publication_commit,
        )
    )
    if errors:
        return {
            "code": "deferred_physical_replay_acceptance_invalid",
            "detail": "采集后统一复核证明无效：" + "；".join(errors),
            "deferred_physical_replay_acceptance": dict(acceptance),
        }
    return None


def physical_replay_upload_blocker(
    demo_dir: str | Path,
) -> dict[str, Any] | None:
    """Validate the latest report and fresh acceptance with one tree hash."""

    demo_path = Path(demo_dir)
    # B03/Newton has a deliberately separate evidence model: the source and
    # replay must use different GPU *models*, an exactly matched software/
    # driver identity, Newton's deterministic collision ordering, and the
    # schema's semantic tolerances.  Dispatch before the PhysX manifest parser
    # so neither contract can accidentally weaken the other.
    from handson.data.newton_cross_gpu_certification import (
        upload_blocker as newton_cross_gpu_upload_blocker,
    )

    newton_blocker = newton_cross_gpu_upload_blocker(demo_path)
    if newton_blocker is not None:
        return newton_blocker
    try:
        manifest = json.loads(
            (demo_path / "replay_manifest.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        manifest = None
    if isinstance(manifest, Mapping):
        task = manifest.get("task")
        recording = manifest.get("recording")
        if (
            isinstance(task, Mapping)
            and isinstance(recording, Mapping)
            and task.get("name") == "t09_insert_peg"
            and recording.get("sim_backend") == "isaacsim_newton"
        ):
            # A valid Newton acceptance returned no blocker.  Never continue
            # into the unrelated PhysX validator.
            return None
    if ".physical_replay_pending" in demo_path.resolve().parts:
        return {
            "code": "physical_replay_pending_unpublished",
            "detail": ("该轨迹仍在物理认证 pending 区，未完成原子发布，暂不上传。"),
        }
    recording, requirement_blocker = _physical_replay_recording_requirement(demo_path)
    if requirement_blocker is not None:
        return requirement_blocker
    if recording is None:
        return None
    mode_blocker = _certification_mode_blocker(
        recording,
        task_name=_manifest_task_name(demo_path),
    )
    if mode_blocker is not None:
        return mode_blocker
    if not (demo_path / REPORT_NAME).is_file():
        return certification_upload_blocker(demo_path)
    mode = str(recording.get("physical_replay_certification_mode") or "").strip()
    try:
        actual_inputs = capture_certified_inputs(demo_path)
    except (OSError, UnicodeError, OverflowError) as exc:
        return {
            "code": "physical_replay_certification_stale",
            "detail": f"无法复核连续 PhysX replay 的认证输入：{exc}",
        }
    blocker = certification_upload_blocker(
        demo_path,
        actual_inputs=actual_inputs,
    )
    if blocker is not None:
        return blocker
    if mode == DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE:
        return deferred_physical_replay_acceptance_upload_blocker(
            demo_path,
            actual_inputs=actual_inputs,
        )
    return fresh_process_acceptance_upload_blocker(
        demo_path,
        actual_inputs=actual_inputs,
    )


__all__ = [
    "CERTIFICATION_MODE",
    "CONTACT_SIGNATURE_OVERLAY_ALGORITHM",
    "CONTACT_SIGNATURE_OVERLAY_FORMAT",
    "CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD",
    "CONTACT_SIGNATURE_OVERLAY_VERSION",
    "DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS",
    "DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_FORMAT",
    "DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_VERSION",
    "DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE",
    "DETACHED_HAND_ACTION_INPUT_MODE",
    "DETACHED_HAND_ACTION_TRANSITION_CONTRACT",
    "FRESH_PROCESS_ACCEPTANCE_FORMAT",
    "FRESH_PROCESS_ACCEPTANCE_NAME",
    "FRESH_PROCESS_ACCEPTANCE_VERSION",
    "FRESH_PROCESS_CERTIFICATION_MODE",
    "FRESH_PROCESS_CERTIFICATION_MODE_WHITELIST",
    "FRESH_PROCESS_REPORT_DIRECTORY",
    "FRESH_SCENE_INITIALIZATION_MODE",
    "INPUT_IDENTITY_ALGORITHM",
    "INITIALIZATION_MODE",
    "INITIALIZATION_RESTORE_COUNT_FIELDS",
    "INITIALIZATION_RESTORE_COUNTS_SOURCE",
    "LEGACY_RAW_ACTION_INPUT_MODE",
    "LEGACY_RAW_ACTION_TRANSITION_CONTRACT",
    "ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE",
    "PERSISTENT_RECORDER_LIFECYCLE",
    "PORTABLE_SOURCE_IDENTITIES",
    "PHYSICAL_REPLAY_INVALID_OUTCOME_CODE",
    "PHYSICAL_REPLAY_OUTCOME_VERSION",
    "PHYSICAL_REPLAY_REVIEW_INVALID",
    "PHYSICAL_REPLAY_REVIEW_REQUIRED",
    "PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE",
    "PHYSICAL_REPLAY_REVIEW_VALID",
    "PHYSICAL_REPLAY_VALID_OUTCOME_CODE",
    "RECORDED_DRIVE_TARGET_INPUT_MODE",
    "RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT",
    "REPORT_FORMAT",
    "REPORT_NAME",
    "REPORT_VERSION",
    "SESSION_PUBLICATION_COMMIT_DIRECTORY",
    "SESSION_PUBLICATION_COMMIT_FORMAT",
    "SESSION_PUBLICATION_COMMIT_VERSION",
    "SESSION_REPLAY_EXECUTION_MODE",
    "SLEEPING_ABC_ORIGIN_TASKS",
    "capture_certified_inputs",
    "certification_report_blocker",
    "certification_upload_blocker",
    "deferred_physical_replay_acceptance_upload_blocker",
    "fresh_process_acceptance_upload_blocker",
    "local_physical_replay_report_blocker",
    "physical_replay_certification_required",
    "physical_replay_upload_blocker",
]
