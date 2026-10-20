from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from handson.sim.isaaclab.articulated_replay import (
    RecordedArticulatedDrivePath,
)
from handson.sim.isaaclab.recording_origin import (
    RecordingOriginStabilityError,
    VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS,
    VELOCITY_CLAMP_SCOPE_NONE,
)
from handson.sim.isaaclab.b320_bc017_contract import (
    BC017_PROPRIO_FLATTEN_ORDER,
    HIDDEN_OBSERVATION_FIELDS,
    PUBLIC_OBSERVATION_GROUPS,
    TASK_NAME as BC017_TASK_NAME,
)
from handson.sim.isaaclab.as019_screwdriver_handover_env import (
    AS019_HIDDEN_STATE_FIELDS,
    TASK_NAME as AS019_TASK_NAME,
)
from handson.data.physical_replay_certification import (
    CERTIFICATION_MODE,
    DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE,
    FRESH_PROCESS_CERTIFICATION_MODE,
    INITIALIZATION_MODE,
    INITIALIZATION_RESTORE_COUNTS_SOURCE,
    LEGACY_RAW_ACTION_INPUT_MODE,
    LEGACY_RAW_ACTION_TRANSITION_CONTRACT,
    RECORDED_DRIVE_TARGET_INPUT_MODE,
    RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT,
    _DEFERRED_CANONICAL_INITIALIZATION_HISTORY_FIELDS,
    capture_certified_inputs,
    certification_upload_blocker,
    fresh_process_acceptance_upload_blocker,
    physical_replay_upload_blocker,
)
from handson.runtime.collection_support import (
    BC011_COLLECTION_ANGLE_GUIDANCE_ENV,
)
from scripts.certify_physx_replay import (
    PHYSICS_ONLY_EXECUTION_MODE,
    SESSION_REPLAY_EXECUTION_MODE,
    CertificationError,
    LoadedDemo,
    LoadedSessionReplay,
    PhysicalFieldSpec,
    _canonical_physx_schema,
    _construct_environment,
    _decode_recorded_action,
    _hide_physics_only_app_window,
    _install_recorded_raw_command_without_ik,
    _initialization_field_partitions,
    _resolve_action_storage_layout,
    _install_recorded_task_controller_state,
    _load_platform_session_history,
    _restore_recorded_reset_snapshot,
    _shared_session_field_value,
    _replay_discarded_session_trajectory,
    _rebuild_session_scene_if_recorded,
    _recorded_environment,
    _step_with_recorded_action,
    _validated_replayed_stable_origin_history,
    _validate_runtime_trace_schema,
    _validated_session_history,
    certify_cli_demo_dirs,
    certify_demo,
    certify_demo_session,
    compare_snapshot,
    load_demo,
    main,
    run_certification,
    replay_session_events,
    verify_source_identity,
    write_report_atomic,
)
from scripts.record_demos_zarr import _action_storage_layout_for_action


def test_physical_replay_omits_bc011_collection_angle_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(BC011_COLLECTION_ANGLE_GUIDANCE_ENV, "stale")

    with _recorded_environment(
        {
            BC011_COLLECTION_ANGLE_GUIDANCE_ENV: "1",
            "HANDSON_RECORDED_TEST_VALUE": "present",
        }
    ):
        assert BC011_COLLECTION_ANGLE_GUIDANCE_ENV not in os.environ
        assert os.environ["HANDSON_RECORDED_TEST_VALUE"] == "present"

    assert os.environ[BC011_COLLECTION_ANGLE_GUIDANCE_ENV] == "stale"

_OBSOLETE_EXACT_ROBOT_SUBSTEP_CONTRACT = (
    "physx_state_t_plus_exact_robot_substep_path_t_produces_next_physx_state_t"
)
_EXACT_ROBOT_SUBSTEP_FIELDS = (
    "physx_sim_time",
    "physx_substep_count",
    "physx_substep_sim_time",
    "physx_substep_robot_joint_position",
    "physx_substep_robot_joint_velocity",
    "physx_substep_robot_joint_valid",
    "physx_substep_robot_joint_target",
    "physx_substep_robot_joint_target_valid",
)
_REQUIRED_TEST_CONTACT_SIGNATURE_FIELDS = {
    "physx_contact_signature_valid",
    "physx_contact_signature_pair_count",
    "physx_contact_signature_pair_path_indices",
    "physx_contact_signature_pair_counts",
    "physx_contact_signature_pair_metrics",
    "physx_substep_contact_signature_valid",
    "physx_substep_contact_signature_pair_count",
    "physx_substep_contact_signature_pair_path_indices",
    "physx_substep_contact_signature_pair_counts",
    "physx_substep_contact_signature_pair_metrics",
}


def _test_canonical_replay_atol(name: str, dtype: str) -> float:
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


def _test_physical_replay_field(name: str) -> bool:
    is_signature = name.startswith("physx_contact_signature_") or (
        name.startswith("physx_substep_contact_signature_")
    )
    is_contact = (
        name.startswith("physx_contact_")
        or name.startswith("physx_friction_anchor_")
        or name.startswith("physx_substep_contact_")
        or name.startswith("physx_substep_friction_anchor_")
    )
    return is_signature or not is_contact


def _complete_v6_test_schema(
    *,
    schema_version: int,
    omit_substep_field: str | None,
) -> dict[str, object]:
    """Build the smallest fixture with the same sealed layout as production."""

    specs: dict[str, tuple[list[int], str]] = {
        # Legacy test-only state used by the lightweight fake environments.
        "physx_robot_qpos": ([2], "float64"),
        "physx_robot_joint_position": ([2, 2], "float64"),
        "physx_robot_joint_velocity": ([2, 2], "float64"),
        "physx_robot_joint_target": ([2, 2], "float64"),
        "physx_robot_joint_target_valid": ([2, 2], "uint8"),
        "physx_robot_joint_valid": ([2, 2], "uint8"),
        "physx_robot_root_state": ([2, 13], "float64"),
        "physx_robot_root_valid": ([2], "uint8"),
        "physx_robot_joint_effort": ([2, 2], "float64"),
        "physx_robot_joint_effort_valid": ([2, 2], "uint8"),
        "physx_tcp_state": ([2, 13], "float64"),
        "physx_tcp_valid": ([2], "uint8"),
        "physx_sim_time": ([1], "float64"),
        "physx_control_step": ([1], "float64"),
        "physx_command_tcp_pose": ([2, 7], "float64"),
        "physx_command_hand": ([2, 20], "float64"),
        "physx_command_hand_valid": ([2, 20], "uint8"),
        "physx_pose_commanded": ([2], "uint8"),
        "physx_pose_filter_state": ([2, 14], "float64"),
        "physx_pose_filter_valid": ([2], "uint8"),
        "physx_contact_report_valid": ([1], "uint8"),
        "physx_contact_capture_source": ([1], "uint8"),
        "physx_contact_header_count": ([1], "uint32"),
        "physx_contact_header_ids": ([2, 5], "uint64"),
        "physx_contact_header_meta": ([2, 7], "uint32"),
        "physx_contact_point_count": ([1], "uint32"),
        "physx_contact_point_data": ([4, 10], "float32"),
        "physx_contact_point_face_indices": ([4, 2], "uint32"),
        "physx_contact_point_material_ids": ([4, 2], "uint64"),
        "physx_friction_anchor_count": ([1], "uint32"),
        "physx_friction_anchor_data": ([4, 6], "float32"),
        "physx_contact_signature_valid": ([1], "uint8"),
        "physx_contact_signature_pair_count": ([1], "uint32"),
        "physx_contact_signature_pair_path_indices": ([2, 4], "uint32"),
        "physx_contact_signature_pair_counts": ([2, 2], "uint32"),
        "physx_contact_signature_pair_metrics": ([2, 3], "float64"),
        "physx_substep_count": ([1], "int64"),
        "physx_task_rigid_body_state": ([1, 13], "float64"),
        "physx_task_rigid_body_valid": ([1], "uint8"),
        "physx_substep_robot_joint_position": ([2, 2, 2], "float64"),
        "physx_substep_robot_joint_velocity": ([2, 2, 2], "float64"),
        "physx_substep_robot_joint_valid": ([2, 2, 2], "uint8"),
        "physx_substep_robot_joint_target": ([2, 2, 2], "float64"),
        "physx_substep_robot_joint_target_valid": ([2, 2, 2], "uint8"),
        "physx_substep_sim_time": ([2, 1], "float64"),
        "physx_substep_contact_signature_valid": ([2, 1], "uint8"),
        "physx_substep_contact_signature_pair_count": ([2, 1], "uint32"),
        "physx_substep_contact_signature_pair_path_indices": (
            [2, 2, 4],
            "uint32",
        ),
        "physx_substep_contact_signature_pair_counts": (
            [2, 2, 2],
            "uint32",
        ),
        "physx_substep_contact_signature_pair_metrics": (
            [2, 2, 3],
            "float64",
        ),
    }
    if omit_substep_field is not None:
        specs.pop(omit_substep_field, None)

    environment_fields: dict[str, dict[str, object]] = {}
    stored_fields: dict[str, dict[str, object]] = {
        "action": {"dtype": "float64", "shape_per_step": [3]},
        "terminal": {"dtype": "uint8", "shape_per_step": []},
        "succeed": {"dtype": "uint8", "shape_per_step": []},
    }
    for name, (shape, dtype) in specs.items():
        semantics: dict[str, object] = {
            "shape": shape,
            "dtype": dtype,
            "physical_replay": _test_physical_replay_field(name),
            "replay_atol": _test_canonical_replay_atol(name, dtype),
        }
        if name in _REQUIRED_TEST_CONTACT_SIGNATURE_FIELDS:
            semantics["replay_restore_supported"] = False
        environment_fields[name] = semantics
        stored_entry = {
            "dtype": dtype,
            "shape_per_step": shape,
            "semantics": dict(semantics),
        }
        stored_fields[name] = stored_entry
        stored_fields[f"next_{name}"] = dict(stored_entry)

    physics_config = {
        "device": "cuda:0",
        "physics_dt": 1.0 / 120.0,
        "control_dt": 1.0 / 60.0,
        "n_substeps": 2,
        "solver": {"enable_enhanced_determinism": True},
    }
    return {
        "environment_schema": {
            "name": "handson.physx_ground_truth",
            "version": schema_version,
            "reset_replay_contract": {
                "allow_pre_action_state_overwrite": True,
                "allows_single_recorded_t0_restore": True,
                "forbids_state_writes_after_action_start": True,
                "physics_steps_after_exact_recorded_t0_restore": 0,
            },
            "physx": {
                "bitwise_checkpoint": False,
                "requires_cuda_backend": True,
                "requires_fresh_process_certification": True,
                "certification_forbids_state_writes": False,
                "allows_single_recorded_t0_restore": True,
                "certification_forbids_state_writes_after_action_start": True,
                "robot_sides": ["right", "left"],
                "robot_joint_names": {
                    "right": ["right_joint_0", "right_joint_1"],
                    "left": ["left_joint_0", "left_joint_1"],
                },
                "rigid_body_names": ["/World/Task/body"],
                "articulation_names": [],
                "articulation_dof_names": {},
                "deformable_name": None,
                "contact_report": {
                    "path_table": [
                        {"id": 11, "path": "/World/A"},
                        {"id": 12, "path": "/World/B"},
                        {"id": 13, "path": "/World/C"},
                        {"id": 14, "path": "/World/D"},
                    ],
                    "report_enabled_task_body_paths": ["/World/A"],
                    "tensor_sensor_paths": [],
                    "tensor_filter_paths": [],
                },
                "physics_config": physics_config,
                "physics_dt": physics_config["physics_dt"],
                "control_dt": physics_config["control_dt"],
                "n_substeps": physics_config["n_substeps"],
                "enhanced_determinism": True,
            },
            "fields": environment_fields,
        },
        "fields": stored_fields,
    }


def _fresh_origin_history(*, deformable_nodes: int = 0) -> dict[str, object]:
    rigid_names = ["/World/Task/body"] if deformable_nodes == 0 else []
    deformable_names = [f"surface:{index}" for index in range(deformable_nodes)]
    active_layout = {
        "format": "handson.active_physics_layout",
        "version": 1,
        "rigid_body": {
            "names": rigid_names,
            "valid": [1] * len(rigid_names),
        },
        "articulation_root": {"names": [], "valid": []},
        "articulation_dof": {"names": [], "valid": []},
        "deformable_node": {
            "names": deformable_names,
            "valid": [1] * len(deformable_names),
        },
    }
    coverage = {
        "rigid_body": int(deformable_nodes == 0),
        "articulation_root": 0,
        "robot_root": 2,
        "tcp": 2,
        "articulation_dof": 0,
        "robot_joint": 2,
        "hand_command": 2,
        "deformable_node": deformable_nodes,
    }
    return {
        "stabilization_velocity_clamp": False,
        "initialization_per_physics_step_velocity_writes": False,
        "stabilization_velocity_clamp_scope": VELOCITY_CLAMP_SCOPE_NONE,
        "physics_dt": 1.0 / 120.0,
        "timeout_seconds": 39.0,
        "physics_steps": 180,
        "initial_settle_steps": 120,
        "stabilization_velocity_clamp_steps": 0,
        "stabilization_velocity_clamp_write": None,
        "stability_metric": "velocity",
        "task_name": "t09_insert_peg",
        "stability_sample_steps": 1,
        "stability_samples": 60,
        "stable_steps_required": 60,
        "stable_steps_observed": 60,
        "gate_task_velocity_stable": True,
        "gate_task_static_pose_stable": False,
        "gate_robot_velocity_stable": True,
        "gate_static_pose_fallback_samples": 0,
        "gate_pose_samples": 0,
        "full_frame_pose_gate": False,
        "reset_pose_drift_checked": True,
        "reset_translation_drift_limit": 0.005,
        "reset_rotation_drift_limit": 0.05235987755982989,
        "reset_max_translation_drift_m": 0.0004,
        "reset_max_rotation_drift_rad": 0.001,
        "reset_coordinate_drift_limit": 0.005,
        "reset_max_coordinate_drift": 0.0003,
        "reset_state_coverage": coverage,
        "active_physics_layout": active_layout,
        "gate_active_physics_layout": active_layout,
        "final_active_physics_layout": active_layout,
        "active_physics_layout_gate_final_exact_match": True,
        "reset_drift_sample_count": 181,
        "reset_drift_expected_sample_count": 181,
        "reset_drift_all_physics_frames_sampled": True,
        "authored_reset_snapshot_source": (
            "environment_after_authored_writes_before_first_physics_step"
        ),
        "authored_reset_snapshot_serial": 1,
        "require_valid_contact_report": True,
        "contact_penetration_limit": 0.0005,
        "initialization_contact_summary": {
            "valid": True,
            "sample_count": 181,
            "invalid_sample_count": 0,
            "capture_sources": [2],
            "maximum_contact_point_count": 4,
            "maximum_penetration_depth_m": 0.000001,
            "deepest_sample_index": 2,
            "deepest_contact_point_index": 2,
            "error": None,
            "expected_sample_count": 181,
            "all_physics_frames_sampled": True,
        },
        "gate_contact_summary": {
            "valid": True,
            "capture_source": 2,
            "contact_point_count": 4,
            "maximum_penetration_depth_m": 0.000001,
            "deepest_contact_point_index": 2,
            "error": None,
        },
        "initial_geometry_gate_passed": True,
        "scene_rebuilt_before_reset": False,
        "scene_generation": 0,
        "final_reset_pose_drift_checked": True,
        "final_reset_translation_drift_limit": 0.005,
        "final_reset_rotation_drift_limit": 0.05235987755982989,
        "final_reset_coordinate_drift_limit": 0.005,
        "final_reset_max_translation_drift_m": 0.0004,
        "final_reset_max_rotation_drift_rad": 0.001,
        "final_reset_max_coordinate_drift": 0.0003,
        "final_reset_state_coverage": coverage,
        "gate_final_translation_delta_limit": 0.005,
        "gate_final_rotation_delta_limit": 0.05235987755982989,
        "gate_final_max_task_translation_delta_m": 0.0002,
        "gate_final_max_task_rotation_delta_rad": 0.0005,
        "final_contact_penetration_limit": 0.0005,
        "final_contact_summary": {
            "valid": True,
            "capture_source": 2,
            "contact_point_count": 4,
            "maximum_penetration_depth_m": 0.000001,
            "deepest_contact_point_index": 2,
            "error": None,
        },
        "final_boundary_physics_gate_passed": True,
        "post_settle_velocity_zeroing": True,
        "post_settle_sleeping": False,
        "velocity_zeroing_passes": 3,
        "first_state_write": None,
        "post_settle_state_write": {
            "robot_joints": 2,
            "rigid_bodies": int(deformable_nodes == 0),
            "articulation_roots": 0,
            "articulation_dofs": 0,
            "deformable_nodes": deformable_nodes,
        },
        "post_settle_sleep": None,
        "boundary_state_write_mode": "velocity_only",
        "contact_warm_start_preserved": True,
        "final_boundary_state_write": {
            "robot_joints": 2,
            "rigid_bodies": int(deformable_nodes == 0),
            "articulation_roots": 0,
            "articulation_dofs": 0,
            "deformable_nodes": deformable_nodes,
        },
        "final_boundary_sleep": None,
        "final_boundary_restore_sync": True,
        "final_boundary_physics_steps_after_write": 0,
        "default_joint_state_restores": 0,
        "robot_hold_target_source": "current_joint_state",
        "post_settle_robot_hold_target_source": "current_joint_state",
        "final_default_joint_state_restore": False,
        "final_robot_hold_target_source": "current_joint_state",
        "hold_target_sync": {"passes": 182},
        "final_hold_target_sync": {"passes": 1},
        "final_boundary_task_and_joint_velocity_exact_zero": True,
        "state_writes": True,
        "trajectory_per_frame_state_writes": False,
        "per_frame_state_writes": False,
    }


def _write_demo(
    tmp_path: Path,
    *,
    exact_robot_substeps: bool = False,
    schema_version: int = 6,
    omit_substep_field: str | None = None,
) -> Path:
    demo = tmp_path / "demo"
    demo.mkdir()
    manifest = {
        "task": {"name": "t09_insert_peg"},
        "recording": {
            "data_fps": 60,
            "collection_run_seed": 17,
            "episode_reset_seed": 18,
            "randomize": True,
            "arm_type": "rokae",
            "hand_type": "wuji1",
            "render_mode": "rgb_array",
            "image_obs": True,
            "headless": True,
            "target_label": "",
            "camera_screen_effect": False,
            "ipad_screen_effect": False,
            "action_storage_layout": "single_pose_hand",
            "action_semantics": "exact_raw_environment_action_executed",
            "physical_transition_contract": (
                "physx_state_t_plus_action_t_produces_next_physx_state_t"
            ),
            "trajectory_input_mode": LEGACY_RAW_ACTION_INPUT_MODE,
            "physical_replay_goal": CERTIFICATION_MODE,
            "physical_replay_certification_mode": (FRESH_PROCESS_CERTIFICATION_MODE),
            "physical_replay_state_recorded": True,
            "physical_replay_certification_required": True,
            "source_scene_rebuild_per_reset": False,
            "source_scene_generation": 0,
            "stable_recording_origin": {
                "strategy": "continuous_current_target_all_velocity_settle",
                "stability_metric": "velocity",
                "initial_settle_seconds": 0.0,
                "post_settle_velocity_zeroing": False,
                "linear_velocity_limit": 0.001,
                "angular_velocity_limit": 0.002,
                "joint_velocity_limit": 0.001,
                "stable_seconds_required": 0.5,
                "timeout_seconds": 30.0,
                "state_writes": True,
                "per_frame_state_writes": False,
                "post_settle_sleeping": False,
                "stabilization_velocity_clamp": True,
                "initialization_per_physics_step_velocity_writes": True,
                "stabilization_velocity_clamp_scope": (
                    VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS
                ),
                "physics_steps": 60,
                "initial_settle_steps": 0,
                "stabilization_velocity_clamp_steps": 60,
                "stabilization_velocity_clamp_write": {
                    "rigid_bodies": 1,
                    "articulation_roots": 0,
                    "articulation_dofs": 0,
                    "deformable_nodes": 0,
                },
                "final_boundary_state_write": {
                    "robot_joints": 2,
                    "rigid_bodies": 1,
                    "articulation_roots": 0,
                    "articulation_dofs": 0,
                    "deformable_nodes": 0,
                },
                "final_boundary_restore_sync": True,
                "final_boundary_sleep": None,
                "final_boundary_physics_steps_after_write": 0,
                "final_boundary_task_and_joint_velocity_exact_zero": True,
                "reset_pose_drift_checked": True,
                "reset_translation_drift_limit": 0.005,
                "reset_rotation_drift_limit": 0.05235987755982989,
                "reset_max_translation_drift_m": 0.0004,
                "reset_max_rotation_drift_rad": 0.001,
                "reset_coordinate_drift_limit": 0.005,
                "reset_max_coordinate_drift": 0.0003,
                "reset_state_coverage": {
                    "rigid_body": 1,
                    "articulation_root": 0,
                    "robot_root": 2,
                    "tcp": 2,
                    "articulation_dof": 0,
                    "robot_joint": 2,
                    "hand_command": 2,
                    "deformable_node": 0,
                },
                "active_physics_layout": {
                    "format": "handson.active_physics_layout",
                    "version": 1,
                    "rigid_body": {
                        "names": ["/World/Task/body"],
                        "valid": [1],
                    },
                    "articulation_root": {"names": [], "valid": []},
                    "articulation_dof": {"names": [], "valid": []},
                    "deformable_node": {"names": [], "valid": []},
                },
                "reset_drift_sample_count": 61,
                "reset_drift_expected_sample_count": 61,
                "reset_drift_all_physics_frames_sampled": True,
                "authored_reset_snapshot_source": (
                    "environment_after_authored_writes_before_first_physics_step"
                ),
                "authored_reset_snapshot_serial": 1,
                "require_valid_contact_report": True,
                "contact_penetration_limit": 0.0005,
                "initialization_contact_summary": {
                    "valid": True,
                    "sample_count": 61,
                    "invalid_sample_count": 0,
                    "capture_sources": [2],
                    "maximum_contact_point_count": 4,
                    "maximum_penetration_depth_m": 0.000001,
                    "deepest_sample_index": 2,
                    "deepest_contact_point_index": 2,
                    "error": None,
                    "expected_sample_count": 61,
                    "all_physics_frames_sampled": True,
                },
                "gate_contact_summary": {
                    "valid": True,
                    "capture_source": 2,
                    "contact_point_count": 4,
                    "maximum_penetration_depth_m": 0.000001,
                    "deepest_contact_point_index": 2,
                    "error": None,
                },
                "initial_geometry_gate_passed": True,
                "scene_rebuilt_before_reset": False,
                "scene_generation": 0,
            },
        },
        "environment": {"variables": {"HANDSON_ISAACLAB_ARM_IK_BACKEND": "dls"}},
        "software": {
            "git": {
                "commit": "a" * 40,
                "dirty": False,
                "changed_files": [],
                "diff_binary_sha256": "0" * 64,
                "worktree_content_sha256": "1" * 64,
            },
            "runtime_replay_identity": {
                "platform": "test",
            },
        },
    }
    stable_origin = manifest["recording"]["stable_recording_origin"]
    manifest["recording"]["stable_recording_origin"] = {
        **stable_origin,
        **_fresh_origin_history(),
    }
    runtime_physical_entry = {
        "shape": [2],
        "dtype": "float64",
        "physical_replay": True,
        "replay_atol": 1.0e-8,
    }
    physical_entry = {
        "dtype": "float64",
        "shape_per_step": [2],
        "semantics": dict(runtime_physical_entry),
    }
    schema = {
        "environment_schema": {
            "name": "handson.physx_ground_truth",
            "version": schema_version,
            "reset_replay_contract": {
                "allow_pre_action_state_overwrite": True,
                "allows_single_recorded_t0_restore": True,
                "forbids_state_writes_after_action_start": True,
                "physics_steps_after_exact_recorded_t0_restore": 0,
            },
            "physx": {
                "bitwise_checkpoint": False,
                "requires_fresh_process_certification": True,
                "certification_forbids_state_writes": False,
                "allows_single_recorded_t0_restore": True,
                "certification_forbids_state_writes_after_action_start": True,
                "contact_report": {
                    "path_table": [
                        {"id": 11, "path": "/World/A"},
                        {"id": 12, "path": "/World/B"},
                        {"id": 13, "path": "/World/C"},
                        {"id": 14, "path": "/World/D"},
                    ],
                    "report_enabled_task_body_paths": ["/World/A"],
                    "tensor_sensor_paths": [],
                    "tensor_filter_paths": [],
                },
                "physics_config": {
                    "device": "cuda:0",
                    "physics_dt": 1.0 / 120.0,
                    "n_substeps": 2,
                    "solver": {"enable_enhanced_determinism": True},
                },
            },
            "fields": {
                "physx_robot_qpos": dict(runtime_physical_entry),
            },
        },
        "fields": {
            "action": {"dtype": "float64", "shape_per_step": [3]},
            "terminal": {"dtype": "uint8", "shape_per_step": []},
            "succeed": {"dtype": "uint8", "shape_per_step": []},
            "physx_robot_qpos": physical_entry,
            "next_physx_robot_qpos": physical_entry,
        },
    }
    if exact_robot_substeps:
        manifest["recording"]["physical_transition_contract"] = (
            RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT
        )
        manifest["recording"]["trajectory_input_mode"] = (
            RECORDED_DRIVE_TARGET_INPUT_MODE
        )
        manifest["recording"]["fresh_recorder_process_per_attempt_required"] = True
        manifest["recording"]["fresh_recorder_process_per_episode_required"] = True
        manifest["recording"]["fresh_scene_replay_required"] = True
        manifest["recording"]["fresh_certifier_process_per_run_required"] = True
        manifest["recording"]["publish_only_after_fresh_certification"] = True
        manifest["recording"]["source_process_reuse_enabled"] = False
        manifest["recording"]["recorder_pid"] = 12345
        manifest["recording"]["recorder_lifecycle"] = "fresh_process_per_attempt"
        for name in _EXACT_ROBOT_SUBSTEP_FIELDS:
            if name == omit_substep_field:
                continue
            if name == "physx_substep_count":
                shape = [1]
                dtype = "int64"
            elif name == "physx_sim_time":
                shape = [1]
                dtype = "float64"
            elif name == "physx_substep_sim_time":
                shape = [2, 1]
                dtype = "float64"
            else:
                shape = [2, 2, 3]
                dtype = "float64"
            runtime_entry = {
                "shape": shape,
                "dtype": dtype,
                "physical_replay": True,
                "replay_atol": 1.0e-8,
            }
            entry = {
                "dtype": dtype,
                "shape_per_step": shape,
                "semantics": dict(runtime_entry),
            }
            schema["fields"][name] = entry
            schema["fields"][f"next_{name}"] = dict(entry)
            schema["environment_schema"]["fields"][name] = runtime_entry
    elif "physx_substep_count" not in schema["environment_schema"]["fields"]:
        runtime_entry = {
            "shape": [1],
            "dtype": "int64",
            "physical_replay": True,
            "replay_atol": 0.0,
        }
        entry = {
            "dtype": "int64",
            "shape_per_step": [1],
            "semantics": dict(runtime_entry),
        }
        schema["fields"]["physx_substep_count"] = entry
        schema["fields"]["next_physx_substep_count"] = dict(entry)
        schema["environment_schema"]["fields"]["physx_substep_count"] = runtime_entry

    contact_signature_shapes = {
        "physx_contact_signature_valid": ([1], "uint8"),
        "physx_contact_signature_pair_count": ([1], "uint32"),
        "physx_contact_signature_pair_path_indices": ([2, 4], "uint32"),
        "physx_contact_signature_pair_counts": ([2, 2], "uint32"),
        "physx_contact_signature_pair_metrics": ([2, 3], "float64"),
        "physx_substep_contact_signature_valid": ([2, 1], "uint8"),
        "physx_substep_contact_signature_pair_count": ([2, 1], "uint32"),
        "physx_substep_contact_signature_pair_path_indices": (
            [2, 2, 4],
            "uint32",
        ),
        "physx_substep_contact_signature_pair_counts": (
            [2, 2, 2],
            "uint32",
        ),
        "physx_substep_contact_signature_pair_metrics": (
            [2, 2, 3],
            "float64",
        ),
    }
    for name, (shape, dtype) in contact_signature_shapes.items():
        runtime_entry = {
            "shape": shape,
            "dtype": dtype,
            "physical_replay": True,
            "replay_restore_supported": False,
            "replay_atol": 1.0e-6 if dtype == "float64" else 0.0,
        }
        entry = {
            "dtype": dtype,
            "shape_per_step": shape,
            "semantics": dict(runtime_entry),
        }
        schema["fields"][name] = entry
        schema["fields"][f"next_{name}"] = dict(entry)
        schema["environment_schema"]["fields"][name] = runtime_entry
    schema = _complete_v6_test_schema(
        schema_version=schema_version,
        omit_substep_field=(omit_substep_field if exact_robot_substeps else None),
    )
    (demo / "replay_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (demo / "recording_schema.json").write_text(
        json.dumps(schema),
        encoding="utf-8",
    )
    root = zarr.open_group(str(demo / "replay.zarr"), mode="w")
    data = root.create_group("data")
    data.array("action", np.zeros((2, 3), dtype=np.float64))
    data.array("terminal", np.asarray([0, 1], dtype=np.uint8))
    data.array("succeed", np.asarray([0, 1], dtype=np.uint8))
    data.array(
        "physx_robot_qpos",
        np.asarray([[1.0, 2.0], [1.1, 2.1]], dtype=np.float64),
    )
    data.array(
        "next_physx_robot_qpos",
        np.asarray([[1.1, 2.1], [1.2, 2.2]], dtype=np.float64),
    )
    if not exact_robot_substeps:
        values = np.full((2, 1), 2, dtype=np.int64)
        data.array("physx_substep_count", values)
        data.array("next_physx_substep_count", values.copy())
    signature_values = {
        "physx_contact_signature_valid": np.ones((2, 1), dtype=np.uint8),
        "physx_contact_signature_pair_count": np.ones((2, 1), dtype=np.uint32),
        "physx_contact_signature_pair_path_indices": np.tile(
            np.asarray([[[1, 2, 3, 4], [0, 0, 0, 0]]], dtype=np.uint32),
            (2, 1, 1),
        ),
        "physx_contact_signature_pair_counts": np.tile(
            np.asarray([[[1, 1], [0, 0]]], dtype=np.uint32),
            (2, 1, 1),
        ),
        "physx_contact_signature_pair_metrics": np.tile(
            np.asarray([[[0.001, 0.2, 0.1], [0.0, 0.0, 0.0]]]),
            (2, 1, 1),
        ),
        "physx_substep_contact_signature_valid": np.ones((2, 2, 1), dtype=np.uint8),
        "physx_substep_contact_signature_pair_count": np.ones(
            (2, 2, 1), dtype=np.uint32
        ),
        "physx_substep_contact_signature_pair_path_indices": np.tile(
            np.asarray(
                [[[[1, 2, 3, 4], [0, 0, 0, 0]]] * 2],
                dtype=np.uint32,
            ),
            (2, 1, 1, 1),
        ),
        "physx_substep_contact_signature_pair_counts": np.tile(
            np.asarray([[[[1, 1], [0, 0]]] * 2], dtype=np.uint32),
            (2, 1, 1, 1),
        ),
        "physx_substep_contact_signature_pair_metrics": np.tile(
            np.asarray([[[[0.001, 0.2, 0.1], [0.0, 0.0, 0.0]]] * 2]),
            (2, 1, 1, 1),
        ),
    }
    for name, values in signature_values.items():
        data.array(name, values)
        data.array(f"next_{name}", values.copy())
    if exact_robot_substeps:
        for name in _EXACT_ROBOT_SUBSTEP_FIELDS:
            if name == omit_substep_field:
                continue
            entry = schema["fields"][name]
            shape = (2, *entry["shape_per_step"])
            if name == "physx_substep_count":
                values = np.full(shape, 2, dtype=np.int64)
            else:
                values = np.zeros(shape, dtype=np.dtype(entry["dtype"]))
            data.array(name, values)
            data.array(f"next_{name}", values.copy())
    for name, entry in schema["environment_schema"]["fields"].items():
        shape = (2, *entry["shape"])
        dtype = np.dtype(entry["dtype"])
        for stored_name in (name, f"next_{name}"):
            if stored_name not in data:
                data.array(stored_name, np.zeros(shape, dtype=dtype))
    return demo


def _write_bc017_task_state_demo(tmp_path: Path) -> Path:
    """Add the exact five-group BC017 task-state boundary to a v6 demo."""

    demo = _write_demo(tmp_path)
    manifest_path = demo / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task"]["name"] = BC017_TASK_NAME
    manifest["recording"]["stable_recording_origin"]["task_name"] = (
        BC017_TASK_NAME
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    schema_path = demo / "recording_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    environment = schema["environment_schema"]
    environment["reset_replay_contract"][
        "b320_physical_replay_certified"
    ] = False
    environment["task_state"] = {
        "task_name": BC017_TASK_NAME,
        "field_names": list(BC017_PROPRIO_FLATTEN_ORDER),
        "hidden_field_names": ["bc017_hidden_state"],
        "capture_mode": "action_aligned_current_and_next_observation",
        "replay_policy": "compare_every_step_never_restore",
    }
    layouts = dict(PUBLIC_OBSERVATION_GROUPS)
    layouts["bc017_hidden_state"] = HIDDEN_OBSERVATION_FIELDS
    data = zarr.open_group(str(demo / "replay.zarr"), mode="a")["data"]
    for name in BC017_PROPRIO_FLATTEN_ORDER:
        layout = tuple(layouts[name])
        semantics = {
            "shape": [len(layout)],
            "dtype": "float64",
            "source": "bc017_action_aligned_task_state",
            "layout": list(layout),
            "visibility": (
                "hidden_task_logic"
                if name == "bc017_hidden_state"
                else "public_task_state"
            ),
            "replay_domain": "task_state",
            "physical_replay": True,
            "replay_restore_supported": False,
            "replay_atol": (
                0.0 if name == "bc017_hidden_state" else 1.0e-6
            ),
        }
        environment["fields"][name] = semantics
        stored = {
            "dtype": "float64",
            "shape_per_step": [len(layout)],
            "semantics": dict(semantics),
        }
        schema["fields"][name] = stored
        schema["fields"][f"next_{name}"] = json.loads(json.dumps(stored))

        boundary = np.zeros((3, len(layout)), dtype=np.float64)
        boundary[0] = np.arange(len(layout), dtype=np.float64) * 0.001
        boundary[1] = boundary[0] + 1.0
        boundary[2] = boundary[1] + 1.0
        data.array(name, boundary[:2].copy())
        data.array(f"next_{name}", boundary[1:].copy())
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    return demo


def _write_as019_task_state_demo(tmp_path: Path) -> Path:
    """Add AS019's exact assist/slider hidden boundary to a v6 demo."""

    demo = _write_demo(tmp_path)
    manifest_path = demo / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task"]["name"] = AS019_TASK_NAME
    manifest["recording"]["stable_recording_origin"]["task_name"] = (
        AS019_TASK_NAME
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    schema_path = demo / "recording_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    environment = schema["environment_schema"]
    environment["task_state"] = {
        "task_name": AS019_TASK_NAME,
        "field_names": ["as019_hidden_state"],
        "hidden_field_names": ["as019_hidden_state"],
        "capture_mode": "action_aligned_current_and_next_observation",
        "replay_policy": "compare_every_step_never_restore",
    }
    semantics = {
        "shape": [len(AS019_HIDDEN_STATE_FIELDS)],
        "dtype": "float64",
        "source": "as019_action_aligned_task_state",
        "layout": list(AS019_HIDDEN_STATE_FIELDS),
        "visibility": "hidden_task_logic",
        "replay_domain": "task_state",
        "physical_replay": True,
        "replay_restore_supported": False,
        "replay_atol": 1.0e-6,
    }
    environment["fields"]["as019_hidden_state"] = semantics
    stored = {
        "dtype": "float64",
        "shape_per_step": [len(AS019_HIDDEN_STATE_FIELDS)],
        "semantics": dict(semantics),
    }
    schema["fields"]["as019_hidden_state"] = stored
    schema["fields"]["next_as019_hidden_state"] = json.loads(
        json.dumps(stored)
    )
    boundary = np.zeros(
        (3, len(AS019_HIDDEN_STATE_FIELDS)), dtype=np.float64
    )
    boundary[0] = np.arange(
        len(AS019_HIDDEN_STATE_FIELDS), dtype=np.float64
    ) * 0.001
    boundary[1] = boundary[0] + 1.0
    boundary[2] = boundary[1] + 1.0
    data = zarr.open_group(str(demo / "replay.zarr"), mode="a")["data"]
    data.array("as019_hidden_state", boundary[:2].copy())
    data.array("next_as019_hidden_state", boundary[1:].copy())
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    return demo


def _valid_recorded_drive_target_report(demo: Path) -> dict[str, object]:
    manifest = json.loads((demo / "replay_manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((demo / "recording_schema.json").read_text(encoding="utf-8"))
    physical_fields = {
        name: entry
        for name, entry in schema["environment_schema"]["fields"].items()
        if entry.get("physical_replay") is True
    }

    def flatten_keys(value, prefix=""):
        result = []
        for name, item in value.items():
            key = f"{prefix}.{name}" if prefix else name
            if isinstance(item, dict):
                result.extend(flatten_keys(item, key))
            else:
                result.append(key)
        return sorted(result)

    checked_physics_fields = flatten_keys(
        schema["environment_schema"]["physx"]["physics_config"]
    )

    def flatten_values(value, prefix=""):
        result = {}
        for name, item in value.items():
            key = f"{prefix}.{name}" if prefix else name
            if isinstance(item, dict):
                result.update(flatten_values(item, key))
            else:
                result[key] = item
        return {name: result[name] for name in sorted(result)}

    physics_values = flatten_values(
        schema["environment_schema"]["physx"]["physics_config"]
    )
    origin = manifest["recording"]["stable_recording_origin"]
    history = {
        field: origin[field]
        for field in _DEFERRED_CANONICAL_INITIALIZATION_HISTORY_FIELDS
        if field in origin
    }
    return {
        "status": "passed",
        "reason": "continuous_physx_replay_matched",
        "portable": True,
        "source_identity": "clean_commit_match",
        "recorded_git": manifest["software"]["git"],
        "current_git": manifest["software"]["git"],
        "runtime_replay_identity": manifest["software"]["runtime_replay_identity"],
        "task_success_replay": True,
        "final_succeed": True,
        "trace_match": True,
        "transition_outcomes_match": True,
        "first_divergence": None,
        "first_outcome_mismatch": None,
        "field_atols": {
            name: float(entry["replay_atol"]) for name, entry in physical_fields.items()
        },
        "max_errors": {name: 0.0 for name in physical_fields},
        "recorded_physics_configuration_match": {
            "matched": True,
            "checked_field_count": len(checked_physics_fields),
            "checked_fields": checked_physics_fields,
            "unrecorded_static_dynamics_fingerprint": True,
            "recorded_values": physics_values,
            "runtime_values": physics_values,
        },
        "certification_mode": CERTIFICATION_MODE,
        "physical_transition_contract": (RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT),
        "trajectory_input_mode": RECORDED_DRIVE_TARGET_INPUT_MODE,
        "initialization_mode": INITIALIZATION_MODE,
        "initialization_restore_counts": {
            "robot_joints": 2,
            "rigid_bodies": 1,
            "articulation_roots": 0,
            "articulation_dofs": 0,
            "deformable_nodes": 0,
        },
        "initialization_restore_counts_source": (INITIALIZATION_RESTORE_COUNTS_SOURCE),
        "initialization_readback_match": True,
        "initialization_history_match": True,
        "replayed_stable_origin_history": history,
        "post_stabilization_snapshot_restore_performed": False,
        "physics_steps_between_initialization_readback_and_action_0": 0,
        "raw_commands_installed": True,
        "recorded_drive_targets_applied": True,
        "ik_recomputed": False,
        "raw_environment_step_called": False,
        "recorded_drive_target_substeps_expected": 4,
        "recorded_drive_target_substeps_executed": 4,
        "recorded_drive_target_writes_expected": 8,
        "recorded_drive_target_writes_executed": 8,
        "state_writes_performed": True,
        "initialization_state_writes_performed": True,
        "trajectory_state_writes_performed": False,
        "state_writes_after_action_start": False,
        "steps_expected": 2,
        "steps_executed": 2,
        "certified_inputs": capture_certified_inputs(demo),
    }


def _write_sleeping_abc_demo(tmp_path: Path, *, task_name: str) -> Path:
    demo = _write_demo(tmp_path)
    manifest_path = demo / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task"]["name"] = task_name
    origin = manifest["recording"]["stable_recording_origin"]
    origin["task_name"] = task_name
    sleep_report = {
        "rigid_bodies": 1,
        "body_paths": ["/World/Task/body"],
        "skipped_non_dynamic_or_non_path_names": [],
    }
    origin.update(
        {
            "post_settle_sleeping": True,
            "post_settle_sleep": sleep_report,
            "final_boundary_sleep": sleep_report,
            "contact_warm_start_preserved": False,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return demo


def _write_a06_hybrid_demo(tmp_path: Path) -> Path:
    demo = _write_demo(tmp_path)
    manifest_path = demo / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task"]["name"] = "t19_orient_cup_upright"
    origin = manifest["recording"]["stable_recording_origin"]
    origin.update(
        {
            "stability_metric": "velocity_or_static_pose",
            "task_name": "t19_orient_cup_upright",
            "physics_dt": 1.0 / 120.0,
            "pose_translation_limit": 1.0e-6,
            "pose_rotation_limit": 1.0e-5,
            "stable_steps_required": 60,
            "stable_steps_observed": 60,
            "stability_sample_steps": 1,
            "stability_sample_interval_seconds": 1.0 / 120.0,
            "full_frame_pose_gate": True,
            "stabilization_velocity_clamp": True,
            "initialization_per_physics_step_velocity_writes": True,
            "stabilization_velocity_clamp_scope": (
                VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS
            ),
            "physics_steps": 60,
            "initial_settle_steps": 0,
            "stabilization_velocity_clamp_steps": 60,
            "stabilization_velocity_clamp_write": {
                "rigid_bodies": 1,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 0,
            },
            "gate_task_velocity_stable": False,
            "gate_task_static_pose_stable": True,
            "gate_robot_velocity_stable": True,
            "gate_static_pose_fallback_samples": 60,
            "gate_pose_samples": 60,
            "gate_max_task_linear_velocity": 0.0,
            "gate_max_task_angular_velocity": 0.02,
            "gate_max_robot_linear_velocity": 0.0,
            "gate_max_robot_angular_velocity": 0.0,
            "gate_max_robot_joint_velocity": 0.0,
            "gate_max_task_translation_delta": 0.0,
            "gate_max_task_rotation_delta": 0.0,
            "gate_max_fallback_translation_delta": 0.0,
            "gate_max_fallback_rotation_delta": 0.0,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return demo


def _write_a01_hybrid_demo(tmp_path: Path) -> Path:
    demo = _write_demo(tmp_path)
    manifest_path = demo / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task"]["name"] = "t26_place_fruit_in_bucket"
    origin = manifest["recording"]["stable_recording_origin"]
    origin.update(
        {
            "stability_metric": "velocity_or_static_pose",
            "task_name": "t26_place_fruit_in_bucket",
            "physics_dt": 1.0 / 120.0,
            "pose_translation_limit": 1.0e-6,
            "pose_rotation_limit": 1.0e-5,
            "stable_steps_required": 60,
            "stable_steps_observed": 60,
            "stability_sample_steps": 1,
            "stability_sample_interval_seconds": 1.0 / 120.0,
            "full_frame_pose_gate": True,
            "gate_task_velocity_stable": False,
            "gate_task_static_pose_stable": True,
            "gate_robot_velocity_stable": True,
            "gate_static_pose_fallback_samples": 60,
            "gate_pose_samples": 60,
            # These are the bounded GPU-TGS residuals seen on the analytic
            # banana/pear colliders; only their per-frame poses use fallback.
            "gate_max_task_linear_velocity": 4.5e-4,
            "gate_max_task_angular_velocity": 1.00245e-2,
            "gate_max_robot_linear_velocity": 0.0,
            "gate_max_robot_angular_velocity": 0.0,
            "gate_max_robot_joint_velocity": 4.0e-4,
            "gate_max_task_translation_delta": 7.450580596923828e-9,
            "gate_max_task_rotation_delta": 0.0,
            "gate_max_fallback_translation_delta": 7.450580596923828e-9,
            "gate_max_fallback_rotation_delta": 0.0,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return demo


def test_certification_rejects_randomized_physical_parameters(tmp_path: Path) -> None:
    demo_dir = _write_demo(tmp_path)
    manifest_path = demo_dir / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recording"]["randomize_dynamics"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CertificationError, match="randomized physical parameters"):
        _construct_environment(None, load_demo(demo_dir))


def test_construct_environment_preserves_source_scene_then_disables_step_render(
    tmp_path: Path,
) -> None:
    demo = load_demo(_write_demo(tmp_path))
    captured_kwargs: dict[str, object] = {}

    class RawEnvironment:
        render_mode = demo.render_mode
        image_obs = demo.image_obs
        _headless = demo.headless
        hz = 30

    class Environment:
        unwrapped = RawEnvironment()

    class Config:
        def get_native_environment(self, **kwargs):
            captured_kwargs.update(kwargs)
            return Environment()

    environment = _construct_environment(Config(), demo)

    # Construction still receives the exact source display configuration.
    assert captured_kwargs["render_mode"] == demo.render_mode
    assert captured_kwargs["image_obs"] is demo.image_obs
    assert captured_kwargs["headless"] is demo.headless
    assert captured_kwargs["seed"] == demo.collection_run_seed
    assert captured_kwargs["randomize"] is demo.randomize
    assert captured_kwargs["arm_type"] == demo.arm_type
    assert captured_kwargs["hand_type"] == demo.hand_type
    assert "target_open_deg" not in captured_kwargs
    # The already-created scene remains intact; only per-step image/render and
    # wall-clock pacing are disabled for certification.
    assert environment.unwrapped.render_mode == demo.render_mode
    assert environment.unwrapped._headless is demo.headless
    assert environment.unwrapped.image_obs is False
    assert environment.unwrapped.hz == 0
    assert environment.unwrapped._handson_physics_only_app_window_state == {
        "policy": "hide_after_source_scene_construction",
        "status": "source_headless_no_window",
        "visible": False,
        "hide_call_performed": False,
    }


@pytest.mark.parametrize(
    ("target_label", "target_open_deg"),
    [("60deg", 60.0), ("90deg", 90.0), ("120deg", 120.0)],
)
def test_construct_p001_environment_uses_recorded_target_from_first_scene_build(
    tmp_path: Path,
    target_label: str,
    target_open_deg: float,
) -> None:
    demo = replace(
        load_demo(_write_demo(tmp_path)),
        task_name="t48_unfold_fan",
        target_label=target_label,
    )
    captured_kwargs: dict[str, object] = {}

    class RawEnvironment:
        image_obs = demo.image_obs
        hz = 30

    class Environment:
        unwrapped = RawEnvironment()

    class Config:
        def get_native_environment(self, **kwargs):
            captured_kwargs.update(kwargs)
            return Environment()

    _construct_environment(Config(), demo)

    assert captured_kwargs["target_open_deg"] == target_open_deg


@pytest.mark.parametrize("target_label", ["", "60", "60.0deg", "75deg"])
def test_construct_p001_environment_rejects_noncanonical_recorded_target(
    tmp_path: Path,
    target_label: str,
) -> None:
    demo = replace(
        load_demo(_write_demo(tmp_path)),
        task_name="t48_unfold_fan",
        target_label=target_label,
    )

    with pytest.raises(
        CertificationError,
        match="requires an exact recorded target_label",
    ):
        _construct_environment(None, demo)


def test_construct_environment_hides_window_after_source_scene_and_fast_switches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = replace(
        load_demo(_write_demo(tmp_path)),
        render_mode="human",
        image_obs=True,
        headless=False,
    )
    events: list[str] = []

    class RawEnvironment:
        render_mode = demo.render_mode
        image_obs = demo.image_obs
        _headless = demo.headless
        hz = 30

    class Environment:
        unwrapped = RawEnvironment()

    environment = Environment()

    class Config:
        def get_native_environment(self, **kwargs):
            assert kwargs["render_mode"] == "human"
            assert kwargs["image_obs"] is True
            assert kwargs["headless"] is False
            events.append("source_scene_constructed")
            return environment

    def hide_window(*, source_headless: bool):
        assert source_headless is False
        assert environment.unwrapped.image_obs is False
        assert environment.unwrapped.hz == 0
        events.append("native_window_hidden")
        return {
            "policy": "hide_after_source_scene_construction",
            "status": "hidden",
            "visible": False,
            "hide_call_performed": True,
        }

    monkeypatch.setattr(
        "scripts.certify_physx_replay._hide_physics_only_app_window",
        hide_window,
    )

    constructed = _construct_environment(Config(), demo)

    assert constructed is environment
    assert events == ["source_scene_constructed", "native_window_hidden"]
    assert (
        environment.unwrapped._handson_physics_only_app_window_state["status"]
        == "hidden"
    )


def test_hide_physics_only_app_window_uses_native_windowing_without_kit_update() -> (
    None
):
    native_window = object()
    events: list[tuple[str, object | None]] = []

    class AppWindow:
        def get_window(self):
            events.append(("get_window", None))
            return native_window

    class AppWindowModule:
        @staticmethod
        def get_default_app_window():
            events.append(("get_default_app_window", None))
            return AppWindow()

    class Windowing:
        def hide_window(self, window):
            events.append(("hide_window", window))

        def poll_events(self):
            events.append(("poll_events", None))

    windowing = Windowing()

    class WindowingModule:
        @staticmethod
        def acquire_windowing_interface():
            events.append(("acquire_windowing_interface", None))
            return windowing

    state = _hide_physics_only_app_window(
        source_headless=False,
        appwindow_module=AppWindowModule,
        windowing_module=WindowingModule,
    )

    assert state == {
        "policy": "hide_after_source_scene_construction",
        "status": "hidden",
        "visible": False,
        "hide_call_performed": True,
    }
    assert events == [
        ("get_default_app_window", None),
        ("get_window", None),
        ("acquire_windowing_interface", None),
        ("hide_window", native_window),
        ("poll_events", None),
    ]


def test_hide_physics_only_app_window_minimizes_when_hide_is_unsupported() -> None:
    native_window = object()
    events: list[str] = []

    class AppWindow:
        def get_window(self):
            return native_window

    class AppWindowModule:
        @staticmethod
        def get_default_app_window():
            return AppWindow()

    class Windowing:
        def hide_window(self, window):
            assert window is native_window
            events.append("hide")
            raise RuntimeError("hide is unsupported")

        def minimize_window(self, window):
            assert window is native_window
            events.append("minimize")

        def poll_events(self):
            events.append("poll")

    windowing = Windowing()

    class WindowingModule:
        @staticmethod
        def acquire_windowing_interface():
            return windowing

    state = _hide_physics_only_app_window(
        source_headless=False,
        appwindow_module=AppWindowModule,
        windowing_module=WindowingModule,
    )

    assert state == {
        "policy": "hide_after_source_scene_construction",
        "status": "minimized_fallback",
        "visible": False,
        "hide_call_performed": False,
        "minimize_call_performed": True,
        "hide_error_type": "RuntimeError",
    }
    assert events == ["hide", "minimize", "poll"]


def test_load_demo_reads_strict_transition_contract(tmp_path: Path) -> None:
    demo = load_demo(_write_demo(tmp_path))

    assert demo.task_name == "t09_insert_peg"
    assert demo.collection_run_seed == 17
    assert demo.episode_reset_seed == 18
    assert demo.recorded_runtime_identity == {"platform": "test"}
    assert demo.actions.shape == (2, 3)
    specs = {spec.name: spec for spec in demo.field_specs}
    assert "physx_robot_qpos" in specs
    assert set(_REQUIRED_TEST_CONTACT_SIGNATURE_FIELDS) <= set(specs)
    assert specs["physx_robot_qpos"].replay_atol == 1.0e-6
    assert specs["physx_robot_qpos"].replay_restore_supported is True
    np.testing.assert_allclose(
        demo.next_fields["physx_robot_qpos"][-1],
        [1.2, 2.2],
    )


def test_bc017_loader_includes_five_task_state_groups_without_restore(
    tmp_path: Path,
) -> None:
    demo = load_demo(_write_bc017_task_state_demo(tmp_path))
    specs = {spec.name: spec for spec in demo.field_specs}

    assert set(demo.recorded_task_state_field_schema) == set(
        BC017_PROPRIO_FLATTEN_ORDER
    )
    for name in BC017_PROPRIO_FLATTEN_ORDER:
        spec = specs[name]
        assert spec.replay_domain == "task_state"
        assert spec.replay_restore_supported is False
        np.testing.assert_array_equal(
            demo.current_fields[name][1],
            demo.next_fields[name][0],
        )
    restorable, compared, diagnostic = _initialization_field_partitions(
        demo.field_specs
    )
    assert not set(BC017_PROPRIO_FLATTEN_ORDER).intersection(
        spec.name for spec in restorable
    )
    assert set(BC017_PROPRIO_FLATTEN_ORDER) <= {
        spec.name for spec in compared
    }
    assert "bc017_hidden_state" not in diagnostic


@pytest.mark.parametrize(
    ("field_name", "layout_name"),
    (
        ("bc017_hidden_state", "terminal_latched"),
        ("bc017_stage_state", "left_event_step"),
        ("bc017_drawer_state", "left_beyond_60_percent"),
        ("bc017_terminal_state", "terminal_reason_code"),
        ("bc017_contact_state", "ll_active"),
    ),
)
def test_bc017_loader_rejects_task_state_current_next_tampering(
    tmp_path: Path,
    field_name: str,
    layout_name: str,
) -> None:
    demo_dir = _write_bc017_task_state_demo(tmp_path)
    schema = json.loads(
        (demo_dir / "recording_schema.json").read_text(encoding="utf-8")
    )
    layout = schema["environment_schema"]["fields"][field_name]["layout"]
    data = zarr.open_group(str(demo_dir / "replay.zarr"), mode="a")["data"]
    current = data[field_name]
    current[1, layout.index(layout_name)] += 1.0

    with pytest.raises(
        CertificationError,
        match=f"continuity is broken for {field_name}",
    ):
        load_demo(demo_dir)


@pytest.mark.parametrize(
    ("field_name", "layout_name"),
    (
        ("bc017_hidden_state", "terminal_latched"),
        ("bc017_stage_state", "left_event_step"),
        ("bc017_drawer_state", "left_beyond_60_percent"),
        ("bc017_terminal_state", "terminal_reason_code"),
    ),
)
def test_bc017_certifier_detects_each_task_state_semantic_tamper(
    tmp_path: Path,
    field_name: str,
    layout_name: str,
) -> None:
    demo = load_demo(_write_bc017_task_state_demo(tmp_path))
    recorded = {
        spec.name: np.asarray(demo.next_fields[spec.name][0])
        for spec in demo.field_specs
    }
    actual = {name: np.asarray(value).copy() for name, value in recorded.items()}
    layout = demo.recorded_task_state_field_schema[field_name]["layout"]
    actual[field_name][layout.index(layout_name)] += 1.0

    _errors, divergence = compare_snapshot(
        recorded,
        actual,
        demo.field_specs,
        phase="post_action",
        action_index=0,
        recorded_time_origin=0.0,
        actual_time_origin=0.0,
    )

    assert divergence is not None
    assert divergence["field"] == field_name


def test_bc017_t0_hidden_latch_is_compared_but_never_restored(
    tmp_path: Path,
) -> None:
    demo = load_demo(_write_bc017_task_state_demo(tmp_path))
    restorable, compared, _diagnostic = _initialization_field_partitions(
        demo.field_specs
    )
    recorded = {
        spec.name: np.asarray(demo.current_fields[spec.name][0])
        for spec in compared
    }
    actual = {name: np.asarray(value).copy() for name, value in recorded.items()}
    hidden_layout = demo.recorded_task_state_field_schema[
        "bc017_hidden_state"
    ]["layout"]
    actual["bc017_hidden_state"][hidden_layout.index("terminal_latched")] += 1.0

    _errors, divergence = compare_snapshot(
        recorded,
        actual,
        compared,
        phase="reset",
        action_index=None,
        recorded_time_origin=0.0,
        actual_time_origin=0.0,
    )

    assert divergence is not None
    assert divergence["phase"] == "reset"
    assert divergence["field"] == "bc017_hidden_state"
    assert "bc017_hidden_state" not in {spec.name for spec in restorable}


def test_as019_loader_compares_slider_hidden_state_without_restoring_it(
    tmp_path: Path,
) -> None:
    demo = load_demo(_write_as019_task_state_demo(tmp_path))
    specs = {spec.name: spec for spec in demo.field_specs}
    hidden = specs["as019_hidden_state"]

    assert set(demo.recorded_task_state_field_schema) == {
        "as019_hidden_state"
    }
    assert hidden.replay_domain == "task_state"
    assert hidden.replay_restore_supported is False
    assert hidden.replay_atol == pytest.approx(1.0e-6)
    restorable, compared, diagnostic = _initialization_field_partitions(
        demo.field_specs
    )
    assert "as019_hidden_state" not in {spec.name for spec in restorable}
    assert "as019_hidden_state" in {spec.name for spec in compared}
    assert "as019_hidden_state" not in diagnostic


def test_as019_loader_rejects_slider_hidden_state_continuity_tamper(
    tmp_path: Path,
) -> None:
    demo_dir = _write_as019_task_state_demo(tmp_path)
    data = zarr.open_group(str(demo_dir / "replay.zarr"), mode="a")[
        "data"
    ]
    index = AS019_HIDDEN_STATE_FIELDS.index("slider_orientation_latched")
    data["as019_hidden_state"][1, index] += 1.0

    with pytest.raises(
        CertificationError,
        match="continuity is broken for as019_hidden_state",
    ):
        load_demo(demo_dir)


def test_as019_certifier_detects_slider_owner_or_wrist_lock_tamper(
    tmp_path: Path,
) -> None:
    demo = load_demo(_write_as019_task_state_demo(tmp_path))
    recorded = {
        spec.name: np.asarray(demo.next_fields[spec.name][0])
        for spec in demo.field_specs
    }
    actual = {name: np.asarray(value).copy() for name, value in recorded.items()}
    index = AS019_HIDDEN_STATE_FIELDS.index("slider_left_wrist_lock_qw")
    actual["as019_hidden_state"][index] += 0.01

    _errors, divergence = compare_snapshot(
        recorded,
        actual,
        demo.field_specs,
        phase="post_action",
        action_index=0,
        recorded_time_origin=0.0,
        actual_time_origin=0.0,
    )

    assert divergence is not None
    assert divergence["field"] == "as019_hidden_state"
    assert divergence["index"] == [index]


def test_load_demo_accepts_persistent_source_with_fresh_scene_replay(
    tmp_path: Path,
) -> None:
    demo_dir = _write_demo(tmp_path, exact_robot_substeps=True)
    manifest_path = demo_dir / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recording = manifest["recording"]
    recording["physical_replay_certification_mode"] = "deferred_post_collection"
    recording["fresh_recorder_process_per_attempt_required"] = False
    recording["fresh_recorder_process_per_episode_required"] = False
    recording["recorder_lifecycle"] = "persistent_process"
    recording["source_process_reuse_enabled"] = True
    recording["fresh_scene_replay_required"] = True
    recording["fresh_certifier_process_per_run_required"] = True
    recording["publish_only_after_fresh_certification"] = True
    recording["recorder_pid"] = 12345
    recording["collection_seed_source"] = "os_cryptographic_random_31bit"
    recording["episode_reset_ordinal"] = 1
    recording["episode_seed_derivation"] = (
        "collection_run_seed_plus_reset_attempt_index_mod_2pow31"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    demo = load_demo(demo_dir)

    assert demo.fresh_recorder_process_per_attempt_required is False
    assert demo.fresh_scene_replay_required is True
    assert demo.recorder_lifecycle == "persistent_process"
    assert demo.source_process_reuse_enabled is True
    assert demo.collection_seed_source == "os_cryptographic_random_31bit"
    assert demo.episode_reset_ordinal == 1
    assert demo.source_recorder_pid == 12345


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "episode_reset_seed",
            19,
            "does not match its recorded collection seed",
        ),
        (
            "collection_seed_source",
            "wall_clock_seconds_modulo_seed_range",
            "collection_seed_source is unsupported",
        ),
    ],
)
def test_load_demo_rejects_invalid_persistent_seed_provenance(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    demo_dir = _write_demo(tmp_path, exact_robot_substeps=True)
    manifest_path = demo_dir / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recording = manifest["recording"]
    recording.update(
        {
            "physical_replay_certification_mode": "deferred_post_collection",
            "fresh_recorder_process_per_attempt_required": False,
            "fresh_recorder_process_per_episode_required": False,
            "recorder_lifecycle": "persistent_process",
            "source_process_reuse_enabled": True,
            "fresh_scene_replay_required": True,
            "fresh_certifier_process_per_run_required": True,
            "publish_only_after_fresh_certification": True,
            "recorder_pid": 12345,
            "collection_seed_source": "os_cryptographic_random_31bit",
            "episode_reset_ordinal": 1,
            "episode_seed_derivation": (
                "collection_run_seed_plus_reset_attempt_index_mod_2pow31"
            ),
        }
    )
    recording[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CertificationError, match=message):
        load_demo(demo_dir)


def test_load_demo_preserves_t0_restore_support_flag(
    tmp_path: Path,
) -> None:
    demo_dir = _write_demo(tmp_path)
    schema_path = demo_dir / "recording_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["environment_schema"]["fields"]["physx_robot_qpos"][
        "replay_restore_supported"
    ] = False
    for name in ("physx_robot_qpos", "next_physx_robot_qpos"):
        schema["fields"][name]["semantics"]["replay_restore_supported"] = False
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    demo = load_demo(demo_dir)

    assert (
        next(
            spec for spec in demo.field_specs if spec.name == "physx_robot_qpos"
        ).replay_restore_supported
        is False
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("final_boundary_state_write", None),
        ("final_boundary_restore_sync", False),
        ("final_boundary_physics_steps_after_write", False),
        ("final_boundary_task_and_joint_velocity_exact_zero", False),
        ("post_settle_sleeping", None),
        ("post_settle_sleeping", True),
    ],
)
def test_load_demo_rejects_incomplete_exact_stable_origin(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    demo_dir = _write_demo(tmp_path)
    manifest_path = demo_dir / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recording"]["stable_recording_origin"][field] = invalid_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CertificationError):
        load_demo(demo_dir)


def test_load_demo_rejects_corrupted_current_transition_row(
    tmp_path: Path,
) -> None:
    demo_dir = _write_demo(tmp_path)
    root = zarr.open_group(str(demo_dir / "replay.zarr"), mode="a")
    root["data"]["physx_robot_qpos"][1, 0] = 99.0

    with pytest.raises(CertificationError, match="continuity is broken"):
        load_demo(demo_dir)


def test_load_demo_rejects_field_semantics_not_bound_to_environment_schema(
    tmp_path: Path,
) -> None:
    demo_dir = _write_demo(tmp_path)
    schema_path = demo_dir / "recording_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["fields"]["physx_robot_qpos"]["semantics"]["replay_atol"] = 999.0
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(CertificationError, match="not bound"):
        load_demo(demo_dir)


def test_runtime_schema_owns_replayable_field_set_and_tolerances(
    tmp_path: Path,
) -> None:
    demo_dir = _write_demo(tmp_path)
    schema_path = demo_dir / "recording_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    runtime_fields = json.loads(json.dumps(schema["environment_schema"]["fields"]))
    for name in ("physx_robot_qpos", "next_physx_robot_qpos"):
        schema["fields"][name]["semantics"]["replay_atol"] = 999.0
    schema["environment_schema"]["fields"]["physx_robot_qpos"]["replay_atol"] = 999.0
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    demo = load_demo(demo_dir)

    with pytest.raises(CertificationError, match="tolerances differ"):
        _validate_runtime_trace_schema(
            demo,
            {
                "name": "handson.physx_ground_truth",
                "version": 6,
                "physx": demo.recorded_physx_schema,
                "fields": runtime_fields,
            },
        )


def test_load_demo_rejects_pose_based_recording_origin(tmp_path: Path) -> None:
    demo_dir = _write_demo(tmp_path)
    manifest_path = demo_dir / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recording"]["stable_recording_origin"]["stability_metric"] = "pose"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        CertificationError,
        match="strict velocity gate",
    ):
        load_demo(demo_dir)


def test_load_demo_accepts_bounded_a06_full_frame_pose_fallback(
    tmp_path: Path,
) -> None:
    demo = load_demo(_write_a06_hybrid_demo(tmp_path))

    assert demo.task_name == "t19_orient_cup_upright"
    assert demo.stable_recording_origin["stability_metric"] == "velocity_or_static_pose"


def test_load_demo_accepts_descriptive_task_with_abc_style_origin(
    tmp_path: Path,
) -> None:
    demo_dir = _write_demo(tmp_path)
    manifest_path = demo_dir / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_name = "loaded_tray_through_narrow_gate"
    manifest["task"]["name"] = task_name
    manifest["recording"]["stable_recording_origin"]["task_name"] = task_name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    demo = load_demo(demo_dir)

    assert demo.task_name == task_name
    assert demo.stable_recording_origin["post_settle_sleeping"] is False
    assert demo.stable_recording_origin["contact_warm_start_preserved"] is True


def test_load_demo_accepts_hs005_wakeable_sleep_origin(tmp_path: Path) -> None:
    task_name = "t37_scrub_bottle"
    demo = load_demo(_write_sleeping_abc_demo(tmp_path, task_name=task_name))

    assert demo.task_name == task_name
    assert demo.stable_recording_origin["post_settle_sleeping"] is True
    assert demo.stable_recording_origin["contact_warm_start_preserved"] is False


def test_load_demo_rejects_sleeping_origin_for_unapproved_abc_task(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        CertificationError,
        match="ABC stable origins must preserve the awake contact manifold",
    ):
        load_demo(_write_sleeping_abc_demo(tmp_path, task_name="t09_insert_peg"))


def test_load_demo_accepts_bounded_unclamped_a01_full_frame_pose_gate(
    tmp_path: Path,
) -> None:
    demo = load_demo(_write_a01_hybrid_demo(tmp_path))

    origin = demo.stable_recording_origin
    assert demo.task_name == "t26_place_fruit_in_bucket"
    assert origin["stability_metric"] == "velocity_or_static_pose"
    assert origin["stabilization_velocity_clamp"] is False
    assert origin["initialization_per_physics_step_velocity_writes"] is False
    assert origin["post_settle_sleeping"] is False
    assert origin["full_frame_pose_gate"] is True
    assert origin["stable_steps_observed"] == 60
    assert origin["boundary_state_write_mode"] == "velocity_only"
    assert origin["final_boundary_physics_steps_after_write"] == 0
    assert origin["final_boundary_task_and_joint_velocity_exact_zero"] is True


@pytest.mark.parametrize(
    "task_name",
    [
        "t04_cover_pitcher",
        "t44_stack_blocks",
        "t36_scoop_marbles",
        "t50_water_plant",
        "c07_pitcher_pour_cup",
        "c08_tongs_ice_cubes",
        "t11_line_up_blocks",
    ],
)
def test_load_demo_accepts_other_approved_full_frame_pose_gates(
    tmp_path: Path,
    task_name: str,
) -> None:
    demo = _write_a01_hybrid_demo(tmp_path)
    manifest_path = demo / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task"]["name"] = task_name
    manifest["recording"]["stable_recording_origin"]["task_name"] = task_name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_demo(demo)

    assert loaded.task_name == task_name
    assert (
        loaded.stable_recording_origin["stability_metric"] == "velocity_or_static_pose"
    )


def test_load_demo_rejects_static_pose_gate_for_unapproved_task(
    tmp_path: Path,
) -> None:
    demo = _write_a01_hybrid_demo(tmp_path)
    manifest_path = demo / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task"]["name"] = "b99_unapproved_task"
    manifest["recording"]["stable_recording_origin"]["task_name"] = (
        "b99_unapproved_task"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CertificationError, match="approved task identities"):
        load_demo(demo)


def test_load_demo_rejects_static_pose_gate_for_loaded_tray(
    tmp_path: Path,
) -> None:
    demo = _write_a01_hybrid_demo(tmp_path)
    manifest_path = demo / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_name = "loaded_tray_through_narrow_gate"
    manifest["task"]["name"] = task_name
    manifest["recording"]["stable_recording_origin"]["task_name"] = task_name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CertificationError, match="approved task identities"):
        load_demo(demo)


def test_load_demo_rejects_clamped_a01_static_pose_gate(
    tmp_path: Path,
) -> None:
    demo = _write_a01_hybrid_demo(tmp_path)
    manifest_path = demo / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    origin = manifest["recording"]["stable_recording_origin"]
    origin.update(
        {
            "stabilization_velocity_clamp": True,
            "initialization_per_physics_step_velocity_writes": True,
            "stabilization_velocity_clamp_scope": (
                VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS
            ),
            "stabilization_velocity_clamp_steps": 180,
            "stabilization_velocity_clamp_write": {
                "rigid_bodies": 1,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 0,
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CertificationError, match="requires unclamped physics"):
        load_demo(demo)


def test_load_demo_rejects_a01_pose_gate_above_absolute_limit(
    tmp_path: Path,
) -> None:
    demo = _write_a01_hybrid_demo(tmp_path)
    manifest_path = demo / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recording"]["stable_recording_origin"]["pose_translation_limit"] = 1.1e-6
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CertificationError, match="approved absolute contract"):
        load_demo(demo)


def test_load_demo_accepts_legacy_disabled_velocity_clamp_contract(
    tmp_path: Path,
) -> None:
    demo_dir = _write_demo(tmp_path)
    manifest_path = demo_dir / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    origin = manifest["recording"]["stable_recording_origin"]
    origin["stabilization_velocity_clamp"] = False
    origin["initialization_per_physics_step_velocity_writes"] = False
    for field in (
        "stabilization_velocity_clamp_scope",
        "stabilization_velocity_clamp_steps",
        "stabilization_velocity_clamp_write",
    ):
        origin.pop(field)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_demo(demo_dir)

    assert loaded.stable_recording_origin["stabilization_velocity_clamp"] is False
    assert (
        loaded.stable_recording_origin[
            "initialization_per_physics_step_velocity_writes"
        ]
        is False
    )


def test_load_demo_accepts_historical_explicit_none_without_clamp_flags(
    tmp_path: Path,
) -> None:
    demo_dir = _write_demo(tmp_path)
    manifest_path = demo_dir / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    origin = manifest["recording"]["stable_recording_origin"]
    origin.pop("stabilization_velocity_clamp")
    origin.pop("initialization_per_physics_step_velocity_writes")
    origin["stabilization_velocity_clamp_scope"] = "none"
    origin.pop("stabilization_velocity_clamp_steps")
    origin.pop("stabilization_velocity_clamp_write")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_demo(demo_dir)

    assert (
        loaded.stable_recording_origin["stabilization_velocity_clamp_scope"] == "none"
    )


def test_load_demo_accepts_legacy_gate_only_clamp_without_scope(
    tmp_path: Path,
) -> None:
    demo_dir = _write_demo(tmp_path)
    manifest_path = demo_dir / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    origin = manifest["recording"]["stable_recording_origin"]
    origin["stabilization_velocity_clamp"] = True
    origin["initialization_per_physics_step_velocity_writes"] = True
    origin.pop("stabilization_velocity_clamp_scope")
    origin["initial_settle_seconds"] = 0.1
    origin["physics_steps"] = 72
    origin["initial_settle_steps"] = 12
    origin["stabilization_velocity_clamp_steps"] = 60
    origin["stabilization_velocity_clamp_write"] = {
        "rigid_bodies": 1,
        "articulation_roots": 0,
        "articulation_dofs": 0,
        "deformable_nodes": 0,
    }
    origin["stabilization_velocity_clamp_write"].pop("deformable_nodes")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_demo(demo_dir)

    assert loaded.stable_recording_origin["stabilization_velocity_clamp"] is True
    assert "stabilization_velocity_clamp_scope" not in loaded.stable_recording_origin


@pytest.mark.parametrize(
    ("enabled", "per_step_writes"),
    [(True, False), (False, True)],
)
def test_load_demo_rejects_mismatched_velocity_clamp_flags(
    tmp_path: Path,
    enabled: bool,
    per_step_writes: bool,
) -> None:
    demo_dir = _write_demo(tmp_path)
    manifest_path = demo_dir / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    origin = manifest["recording"]["stable_recording_origin"]
    origin["stabilization_velocity_clamp"] = enabled
    origin["initialization_per_physics_step_velocity_writes"] = per_step_writes
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CertificationError, match="flags must match"):
        load_demo(demo_dir)


@pytest.mark.parametrize("clamp_steps", [59, 61])
def test_load_demo_rejects_velocity_clamp_step_count_outside_scope(
    tmp_path: Path,
    clamp_steps: int,
) -> None:
    demo_dir = _write_demo(tmp_path)
    manifest_path = demo_dir / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    origin = manifest["recording"]["stable_recording_origin"]
    origin.update(
        {
            "stabilization_velocity_clamp": True,
            "initialization_per_physics_step_velocity_writes": True,
            "stabilization_velocity_clamp_scope": (
                VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS
            ),
            "physics_steps": 60,
            "initial_settle_steps": 0,
            "stabilization_velocity_clamp_steps": clamp_steps,
            "stabilization_velocity_clamp_write": {
                "rigid_bodies": 1,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 0,
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CertificationError, match="contradicts its scope"):
        load_demo(demo_dir)


def test_replayed_stable_origin_rejects_different_gate_step_count() -> None:
    recorded = _fresh_origin_history()
    recorded.update(
        {
            "stabilization_velocity_clamp": True,
            "initialization_per_physics_step_velocity_writes": True,
            "stabilization_velocity_clamp_scope": (
                VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS
            ),
            "physics_steps": 60,
            "initial_settle_steps": 0,
            "stabilization_velocity_clamp_steps": 60,
            "stabilization_velocity_clamp_write": {
                "rigid_bodies": 1,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 0,
            },
        }
    )
    recorded["reset_drift_sample_count"] = 61
    recorded["reset_drift_expected_sample_count"] = 61
    recorded["initialization_contact_summary"]["sample_count"] = 61
    recorded["initialization_contact_summary"]["expected_sample_count"] = 61
    actual = json.loads(json.dumps(recorded))
    actual["physics_steps"] = 61
    actual["stabilization_velocity_clamp_steps"] = 61
    actual["reset_drift_sample_count"] = 62
    actual["reset_drift_expected_sample_count"] = 62
    actual["initialization_contact_summary"]["sample_count"] = 62
    actual["initialization_contact_summary"]["expected_sample_count"] = 62

    with pytest.raises(CertificationError, match="at physics_steps"):
        _validated_replayed_stable_origin_history(
            recorded,
            {"physical_recording_origin": actual},
        )


def _natural_settle_history_for_replay_test() -> dict[str, object]:
    return _fresh_origin_history()


def test_replayed_natural_settle_allows_adaptive_duration() -> None:
    recorded = _natural_settle_history_for_replay_test()
    actual = json.loads(json.dumps(recorded))
    actual["physics_steps"] = 204
    actual["stability_samples"] = 84
    actual["hold_target_sync"]["passes"] = 206
    actual["reset_drift_sample_count"] = 205
    actual["reset_drift_expected_sample_count"] = 205
    actual["initialization_contact_summary"]["sample_count"] = 205
    actual["initialization_contact_summary"]["expected_sample_count"] = 205

    matched = _validated_replayed_stable_origin_history(
        recorded,
        {"physical_recording_origin": actual},
    )

    assert matched["physics_steps"] == 204
    assert matched["stability_samples"] == 84
    assert matched["hold_target_sync"]["passes"] == 206


def test_replayed_natural_settle_rejects_inconsistent_adaptive_duration() -> None:
    recorded = _natural_settle_history_for_replay_test()
    actual = json.loads(json.dumps(recorded))
    actual["physics_steps"] = 205
    actual["stability_samples"] = 84
    actual["hold_target_sync"]["passes"] = 207
    actual["reset_drift_sample_count"] = 206
    actual["reset_drift_expected_sample_count"] = 206
    actual["initialization_contact_summary"]["sample_count"] = 206
    actual["initialization_contact_summary"]["expected_sample_count"] = 206

    with pytest.raises(CertificationError, match="sampling interval"):
        _validated_replayed_stable_origin_history(
            recorded,
            {"physical_recording_origin": actual},
        )


def test_replayed_stable_origin_rejects_different_boundary_write_count() -> None:
    recorded = {
        **_fresh_origin_history(),
        "final_boundary_state_write": {
            "robot_joints": 2,
            "rigid_bodies": 1,
            "articulation_roots": 0,
            "articulation_dofs": 0,
            "deformable_nodes": 0,
        },
    }
    actual = json.loads(json.dumps(recorded))
    actual["final_boundary_state_write"]["rigid_bodies"] = 2

    with pytest.raises(CertificationError, match="active layout"):
        _validated_replayed_stable_origin_history(
            recorded,
            {"physical_recording_origin": actual},
        )


def test_replayed_stable_origin_reports_every_canonical_history_field() -> None:
    restore_counts = {
        "robot_joints": 2,
        "rigid_bodies": 1,
        "articulation_roots": 0,
        "articulation_dofs": 0,
        "deformable_nodes": 0,
    }
    sleep_counts = {
        "rigid_bodies": 1,
        "body_paths": ["/World/Task/body"],
        "skipped_non_dynamic_or_non_path_names": [],
    }
    recorded = {
        **_fresh_origin_history(),
        "stability_metric": "velocity",
        "stability_sample_steps": 6,
        "stability_samples": 10,
        "stable_steps_required": 60,
        "stable_steps_observed": 60,
        "gate_task_velocity_stable": True,
        "gate_task_static_pose_stable": False,
        "gate_robot_velocity_stable": True,
        "gate_static_pose_fallback_samples": 0,
        "gate_pose_samples": 0,
        "full_frame_pose_gate": False,
        "post_settle_velocity_zeroing": True,
        "post_settle_sleeping": True,
        "velocity_zeroing_passes": 2,
        "first_state_write": restore_counts,
        "post_settle_state_write": restore_counts,
        "post_settle_sleep": sleep_counts,
        "final_boundary_state_write": restore_counts,
        "final_boundary_sleep": sleep_counts,
        "final_boundary_restore_sync": True,
        "final_boundary_physics_steps_after_write": 0,
        "default_joint_state_restores": 1,
        "robot_hold_target_source": "default_joint_state",
        "post_settle_robot_hold_target_source": ("settled_after_default_joint_state"),
        "final_default_joint_state_restore": True,
        "final_robot_hold_target_source": ("settled_after_default_joint_state"),
        "hold_target_sync": {"passes": 61},
        "final_hold_target_sync": {"passes": 1},
        "final_boundary_task_and_joint_velocity_exact_zero": True,
        "state_writes": True,
        "trajectory_per_frame_state_writes": False,
        "per_frame_state_writes": False,
    }
    actual = json.loads(json.dumps(recorded))

    matched = _validated_replayed_stable_origin_history(
        recorded,
        {"physical_recording_origin": actual},
    )

    assert set(_DEFERRED_CANONICAL_INITIALIZATION_HISTORY_FIELDS) <= set(matched)
    for field in _DEFERRED_CANONICAL_INITIALIZATION_HISTORY_FIELDS:
        assert matched[field] == recorded[field]


def test_replayed_stable_origin_rejects_bool_int_history_mismatch() -> None:
    recorded = {
        **_fresh_origin_history(),
        "state_writes": True,
    }
    actual = json.loads(json.dumps(recorded))
    actual["state_writes"] = 1

    with pytest.raises(CertificationError, match="at state_writes"):
        _validated_replayed_stable_origin_history(
            recorded,
            {"physical_recording_origin": actual},
        )


def test_load_demo_rejects_gate_only_a06_synthetic_velocity_evidence(
    tmp_path: Path,
) -> None:
    demo = _write_a06_hybrid_demo(tmp_path)
    manifest_path = demo / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    origin = manifest["recording"]["stable_recording_origin"]
    origin["stabilization_velocity_clamp_scope"] = "gate_only"
    origin["gate_task_velocity_stable"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CertificationError, match="synthetic zero"):
        load_demo(demo)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("pose_translation_limit", 1.0e-4),
        ("pose_rotation_limit", 1.0e-3),
        ("joint_velocity_limit", 1.0),
        ("stable_seconds_required", 0.1),
        ("pose_translation_limit", True),
        ("stabilization_velocity_clamp", False),
        ("initialization_per_physics_step_velocity_writes", False),
        ("gate_max_task_rotation_delta", True),
        ("gate_max_task_rotation_delta", 2.0e-5),
    ],
)
def test_load_demo_rejects_a06_fallback_outside_absolute_contract(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    demo = _write_a06_hybrid_demo(tmp_path)
    manifest_path = demo / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recording"]["stable_recording_origin"][field] = invalid_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CertificationError):
        load_demo(demo)


def test_load_demo_accepts_complete_v6_robot_substep_contract(
    tmp_path: Path,
) -> None:
    demo = load_demo(_write_demo(tmp_path, exact_robot_substeps=True))

    field_names = {spec.name for spec in demo.field_specs}
    assert set(_EXACT_ROBOT_SUBSTEP_FIELDS) <= field_names
    assert (
        demo.physical_transition_contract == RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT
    )
    assert demo.trajectory_input_mode == RECORDED_DRIVE_TARGET_INPUT_MODE


def test_load_demo_rejects_obsolete_exact_robot_substep_contract(
    tmp_path: Path,
) -> None:
    episode = _write_demo(tmp_path, exact_robot_substeps=True)
    manifest_path = episode / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recording"]["physical_transition_contract"] = (
        _OBSOLETE_EXACT_ROBOT_SUBSTEP_CONTRACT
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CertificationError, match="obsolete"):
        load_demo(episode)


def test_load_demo_rejects_legacy_seed_only_initialization_contract(
    tmp_path: Path,
) -> None:
    demo_dir = _write_demo(tmp_path)
    schema_path = demo_dir / "recording_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    physx = schema["environment_schema"]["physx"]
    physx["certification_forbids_state_writes"] = True
    physx.pop("allows_single_recorded_t0_restore")
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(CertificationError, match="single exact recorded t0"):
        load_demo(demo_dir)


@pytest.mark.parametrize("invalid_value", [False, 0.0])
def test_load_demo_rejects_non_integer_zero_steps_after_t0_restore(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    demo_dir = _write_demo(tmp_path)
    schema_path = demo_dir / "recording_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["environment_schema"]["reset_replay_contract"][
        "physics_steps_after_exact_recorded_t0_restore"
    ] = invalid_value
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(CertificationError, match="zero physics steps"):
        load_demo(demo_dir)


@pytest.mark.parametrize("schema_version", [None, 5])
def test_load_demo_rejects_schema_before_exact_t0_restore_contract(
    tmp_path: Path,
    schema_version: int | None,
) -> None:
    episode = _write_demo(tmp_path, schema_version=schema_version or 6)
    if schema_version is None:
        schema_path = episode / "recording_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema["environment_schema"].pop("version")
        schema_path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(
        CertificationError,
        match="schema version 6",
    ):
        load_demo(episode)


def test_load_demo_rejects_wrong_physx_schema_name(tmp_path: Path) -> None:
    episode = _write_demo(tmp_path)
    schema_path = episode / "recording_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["environment_schema"]["name"] = "forged.schema"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(CertificationError, match="schema version 6"):
        load_demo(episode)


@pytest.mark.parametrize(
    "omitted_field",
    [
        "physx_substep_robot_joint_velocity",
        "physx_sim_time",
        "physx_substep_sim_time",
    ],
)
def test_load_demo_rejects_incomplete_v6_robot_substep_contract(
    tmp_path: Path,
    omitted_field: str,
) -> None:
    episode = _write_demo(
        tmp_path,
        exact_robot_substeps=True,
        omit_substep_field=omitted_field,
    )

    with pytest.raises(
        CertificationError,
        match="missing required schema fields",
    ):
        load_demo(episode)


def test_compare_snapshot_reports_first_declared_divergence() -> None:
    specs = (
        PhysicalFieldSpec(
            "physx_robot_qpos",
            "next_physx_robot_qpos",
            (2,),
            np.dtype("float64"),
            1.0e-4,
        ),
    )
    errors, divergence = compare_snapshot(
        {"physx_robot_qpos": np.asarray([1.0, 2.0])},
        {"physx_robot_qpos": np.asarray([1.0, 2.001])},
        specs,
        phase="post_action",
        action_index=4,
    )

    assert errors["physx_robot_qpos"] > 0.0009
    assert divergence["action_index"] == 4
    assert divergence["field"] == "physx_robot_qpos"
    assert divergence["index"] == [1]


def test_compare_snapshot_uses_relative_physx_time() -> None:
    specs = (
        PhysicalFieldSpec(
            "physx_sim_time",
            "next_physx_sim_time",
            (1,),
            np.dtype("float64"),
            1.0e-10,
        ),
        PhysicalFieldSpec(
            "physx_substep_sim_time",
            "next_physx_substep_sim_time",
            (2, 1),
            np.dtype("float64"),
            1.0e-10,
        ),
    )

    errors, divergence = compare_snapshot(
        {
            "physx_sim_time": np.asarray([70.05]),
            "physx_substep_sim_time": np.asarray([[70.025], [70.05]]),
        },
        {
            "physx_sim_time": np.asarray([1.60]),
            "physx_substep_sim_time": np.asarray([[1.575], [1.60]]),
        },
        specs,
        phase="post_action",
        action_index=0,
        recorded_time_origin=70.0,
        actual_time_origin=1.55,
    )

    assert max(errors.values()) < 1.0e-12
    assert divergence is None


def test_exact_recorded_reset_snapshot_is_restored_and_read_back() -> None:
    events: list[str] = []

    class Trace:
        def restore(self, snapshot, schema, *, sync):
            events.append("restore")
            assert sync is True
            np.testing.assert_array_equal(
                snapshot["physx_robot_qpos"],
                [1.0, 2.0],
            )
            assert schema == {"physx": {"robot_sides": ["right"]}}
            return {
                "robot_joints": 2,
                "rigid_bodies": 1,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 0,
            }

        def put_task_bodies_to_sleep(self):
            events.append("sleep")
            return {"rigid_bodies": 1, "body_paths": ["/World/object"]}

        def begin_substep_capture(self):
            events.append("begin_substeps")

        def snapshot(self):
            events.append("snapshot")
            return {"physx_robot_qpos": np.asarray([1.0, 2.0], dtype=np.float64)}

        def substep_snapshot(self, *, consume):
            events.append("substeps")
            assert consume is True
            return {"physx_substep_count": np.asarray([0], dtype=np.int64)}

    trace = Trace()

    class RawEnvironment:
        def _refresh_object_poses(self):
            events.append("refresh")

        def _compute_observation(self):
            events.append("compute")
            return {
                "physx_robot_qpos": np.asarray([9.0, 9.0]),
                "unrelated": "restored-live-value",
            }

        def _attach_physx_ground_truth(self, observation):
            events.append("attach")
            result = dict(observation)
            result.update(trace.snapshot())
            result.update(trace.substep_snapshot(consume=True))
            return result

    class ObservationWrapper:
        def __init__(self, env):
            self.env = env

        @property
        def unwrapped(self):
            return getattr(self.env, "unwrapped", self.env)

    class InnerObservationWrapper(ObservationWrapper):
        def observation(self, observation):
            events.append("inner_wrapper")
            result = dict(observation)
            result["inner_value"] = "created-by-inner"
            return result

    class OuterObservationWrapper(ObservationWrapper):
        def observation(self, observation):
            events.append("outer_wrapper")
            assert observation["inner_value"] == "created-by-inner"
            result = dict(observation)
            result["outer_value"] = "created-by-outer"
            return result

    environment = OuterObservationWrapper(InnerObservationWrapper(RawEnvironment()))

    restored, counts, sleep = _restore_recorded_reset_snapshot(
        environment,
        trace,
        {"physx_robot_qpos": np.asarray([1.0, 2.0])},
        {"robot_sides": ["right"]},
        {
            "post_settle_sleeping": True,
            "final_boundary_sleep": {"rigid_bodies": 1},
        },
    )

    np.testing.assert_array_equal(restored["physx_robot_qpos"], [1.0, 2.0])
    np.testing.assert_array_equal(restored["physx_substep_count"], [0])
    assert restored["unrelated"] == "restored-live-value"
    assert restored["inner_value"] == "created-by-inner"
    assert restored["outer_value"] == "created-by-outer"
    assert counts == {
        "robot_joints": 2,
        "rigid_bodies": 1,
        "articulation_roots": 0,
        "articulation_dofs": 0,
        "deformable_nodes": 0,
    }
    assert sleep == {"rigid_bodies": 1, "body_paths": ["/World/object"]}
    assert events == [
        "restore",
        "sleep",
        "refresh",
        "begin_substeps",
        "compute",
        "attach",
        "snapshot",
        "substeps",
        "inner_wrapper",
        "outer_wrapper",
        "begin_substeps",
    ]


def test_exact_recorded_reset_snapshot_preserves_awake_origin() -> None:
    class Trace:
        def restore(self, _snapshot, _schema, *, sync):
            assert sync is True
            return {
                "robot_joints": 0,
                "rigid_bodies": 1,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 0,
            }

        def put_task_bodies_to_sleep(self):
            raise AssertionError("awake recording origin must remain awake")

        def begin_substep_capture(self):
            pass

        def snapshot(self):
            return {"physx_value": np.asarray([3.0], dtype=np.float64)}

        def substep_snapshot(self, *, consume):
            assert consume is True
            return {}

    trace = Trace()

    class RawEnvironment:
        def _compute_observation(self):
            return {"physx_value": np.asarray([8.0])}

        def _attach_physx_ground_truth(self, observation):
            result = dict(observation)
            result.update(trace.snapshot())
            result.update(trace.substep_snapshot(consume=True))
            return result

    class Environment:
        unwrapped = RawEnvironment()

    restored, counts, sleep = _restore_recorded_reset_snapshot(
        Environment(),
        trace,
        {"physx_value": np.asarray([3.0])},
        {},
        {"post_settle_sleeping": False},
    )

    np.testing.assert_array_equal(restored["physx_value"], [3.0])
    assert counts == {
        "robot_joints": 0,
        "rigid_bodies": 1,
        "articulation_roots": 0,
        "articulation_dofs": 0,
        "deformable_nodes": 0,
    }
    assert sleep is None


def test_exact_recorded_reset_rejects_entity_count_change() -> None:
    class Trace:
        def restore(self, _snapshot, _schema, *, sync):
            assert sync is True
            return {
                "robot_joints": 53,
                "rigid_bodies": 3,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 0,
            }

    with pytest.raises(CertificationError, match="robot_joints"):
        _restore_recorded_reset_snapshot(
            object(),
            Trace(),
            {},
            {},
            {
                "final_boundary_state_write": {
                    "robot_joints": 54,
                    "rigid_bodies": 3,
                }
            },
        )


def test_exact_recorded_reset_mismatch_fails_before_action_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Trace:
        def __init__(self) -> None:
            self.restored = False

        def schema(self):
            return {"physx": {}, "fields": {}}

        def restore(self, _snapshot, _schema, *, sync):
            assert sync is True
            self.restored = True
            return {
                "robot_joints": 2,
                "rigid_bodies": 0,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 0,
            }

        def begin_substep_capture(self):
            pass

        def snapshot(self):
            # Deliberately fail the strict t0 readback gate.
            return {"physx_robot_qpos": np.asarray([1.0, 2.001], dtype=np.float64)}

        def substep_snapshot(self, *, consume):
            assert consume is True
            return {}

    trace = Trace()

    class RawEnvironment:
        hz = 30
        _physx_ground_truth_trace = trace

        def __init__(self) -> None:
            self.step_calls = 0

        def _compute_observation(self):
            return {}

        def _attach_physx_ground_truth(self, observation):
            result = dict(observation)
            result.update(trace.snapshot())
            return result

        def step(self, _action):
            self.step_calls += 1
            raise AssertionError("action 0 must not execute after t0 mismatch")

    class Environment:
        def __init__(self) -> None:
            self.unwrapped = RawEnvironment()

        def reset(self, **_kwargs):
            return {"physx_robot_qpos": np.asarray([9.0, 9.0], dtype=np.float64)}, {}

        def close(self):
            pass

    environment = Environment()
    identity = {
        "commit": "a" * 40,
        "dirty": False,
        "changed_files": [],
    }
    demo = LoadedDemo(
        demo_dir=Path("/tmp/test-exact-origin"),
        task_name="t09_insert_peg",
        collection_run_seed=1,
        episode_reset_seed=2,
        randomize=True,
        randomize_dynamics=False,
        arm_type="rokae",
        hand_type="wuji1",
        render_mode="rgb_array",
        image_obs=False,
        headless=True,
        target_label="",
        camera_screen_effect=False,
        ipad_screen_effect=False,
        environment_variables={},
        recorded_git=identity,
        recorded_runtime_identity={"runtime": "same"},
        recorded_physx_schema={},
        recorded_physx_field_schema={},
        stable_recording_origin={},
        action_storage_layout="flat_raw_env_action",
        actions=np.zeros((2, 3), dtype=np.float64),
        terminals=np.asarray([0, 1], dtype=np.uint8),
        succeeds=np.asarray([0, 1], dtype=np.uint8),
        field_specs=(
            PhysicalFieldSpec(
                "physx_robot_qpos",
                "next_physx_robot_qpos",
                (2,),
                np.dtype("float64"),
                1.0e-6,
            ),
        ),
        current_fields={
            "physx_robot_qpos": np.asarray([[1.0, 2.0], [1.1, 2.1]], dtype=np.float64)
        },
        next_fields={
            "physx_robot_qpos": np.asarray([[1.1, 2.1], [1.2, 2.2]], dtype=np.float64)
        },
    )

    monkeypatch.setattr(
        "scripts.certify_physx_replay.capture_git_replay_state",
        lambda *_args, **_kwargs: dict(identity),
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay.capture_runtime_replay_identity",
        lambda: {"runtime": "same"},
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._construct_environment",
        lambda *_args, **_kwargs: environment,
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay.stabilize_recording_origin",
        lambda _env, observation, info, **_kwargs: (observation, info),
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._validate_runtime_trace_schema",
        lambda *_args, **_kwargs: None,
    )

    report = run_certification(demo)

    assert trace.restored is True
    assert environment.unwrapped.step_calls == 0
    assert report["status"] == "failed"
    assert report["reason"] == "physical_state_diverged"
    assert report["steps_executed"] == 0
    assert report["initialization_readback_match"] is False
    assert report["initialization_first_divergence"]["phase"] == "reset"


def test_persistent_source_drive_targets_certify_from_adaptive_natural_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physx_schema = {
        "robot_sides": ["right"],
        "robot_joint_names": {"right": ["joint0", "joint1"]},
        "physics_config": {
            "device": "cuda:0",
            "physics_dt": 1.0 / 120.0,
            "n_substeps": 2,
        },
    }
    target_path = np.asarray(
        [[[0.1, 0.2]], [[0.3, 0.4]]],
        dtype=np.float64,
    )
    target_valid = np.ones(target_path.shape, dtype=np.uint8)
    initial = {
        "physx_robot_qpos": np.asarray([1.0, 2.0], dtype=np.float64),
        "physx_substep_count": np.asarray([0], dtype=np.int64),
        "physx_substep_robot_joint_target": np.zeros_like(target_path),
        "physx_substep_robot_joint_target_valid": np.zeros_like(target_valid),
        "physx_robot_joint_target": np.asarray([[1.0, 2.0]], dtype=np.float64),
        "physx_robot_joint_target_valid": np.ones((1, 2), dtype=np.uint8),
    }
    expected = {
        "physx_robot_qpos": np.asarray([1.1, 2.1], dtype=np.float64),
        "physx_substep_count": np.asarray([2], dtype=np.int64),
        "physx_substep_robot_joint_target": target_path,
        "physx_substep_robot_joint_target_valid": target_valid,
        "physx_robot_joint_target": target_path[-1].copy(),
        "physx_robot_joint_target_valid": target_valid[-1].copy(),
    }

    trace_events: list[str] = []
    applied_targets: list[np.ndarray] = []

    class Trace:
        def schema(self):
            return {"physx": physx_schema, "fields": {}}

        def restore(self, *_args, **_kwargs):
            raise AssertionError("new certification must not restore recorded t0")

        def begin_substep_capture(self):
            trace_events.append("begin_substeps")

        def apply_robot_joint_targets(self, snapshot, schema):
            assert schema == {"physx": physx_schema}
            applied_targets.append(
                np.asarray(snapshot["physx_robot_joint_target"]).copy()
            )
            return 2

    trace = Trace()

    class RawEnvironment:
        hz = 30
        _physx_ground_truth_trace = trace

        def __init__(self) -> None:
            self.raw_actions: list[np.ndarray] = []
            self.step_calls = 0
            self.ik_calls = 0

        def _apply_action(self, action):
            self.raw_actions.append(np.asarray(action).copy())

        def _apply_joint_targets(self):
            self.ik_calls += 1
            raise AssertionError("online IK must not run")

        def step(self, _action):
            self.step_calls += 1
            raise AssertionError("raw env.step must not run")

        def step_with_current_joint_targets(self):
            trace.begin_substep_capture()
            for index in range(2):
                self._handson_pre_physics_substep_callback(index, 2)
            return dict(expected), 1.0, True, False, {"succeed": True}

    class Environment:
        def __init__(self) -> None:
            self.unwrapped = RawEnvironment()

        def reset(self, **_kwargs):
            return {"not_the_origin": np.asarray([99.0])}, {}

        def close(self):
            pass

    environment = Environment()
    identity = {
        "commit": "a" * 40,
        "dirty": False,
        "changed_files": [],
    }

    def spec(name: str, shape: tuple[int, ...], dtype: str):
        return PhysicalFieldSpec(
            name,
            f"next_{name}",
            shape,
            np.dtype(dtype),
            1.0e-8,
        )

    field_specs = (
        spec("physx_robot_qpos", (2,), "float64"),
        spec("physx_substep_count", (1,), "int64"),
        spec(
            "physx_substep_robot_joint_target",
            (2, 1, 2),
            "float64",
        ),
        spec(
            "physx_substep_robot_joint_target_valid",
            (2, 1, 2),
            "uint8",
        ),
        spec("physx_robot_joint_target", (1, 2), "float64"),
        spec("physx_robot_joint_target_valid", (1, 2), "uint8"),
    )
    demo = LoadedDemo(
        demo_dir=Path("/tmp/test-drive-target-origin"),
        task_name="t09_insert_peg",
        collection_run_seed=1,
        episode_reset_seed=2,
        randomize=True,
        randomize_dynamics=False,
        arm_type="rokae",
        hand_type="wuji1",
        render_mode="rgb_array",
        image_obs=False,
        headless=True,
        target_label="",
        camera_screen_effect=False,
        ipad_screen_effect=False,
        environment_variables={},
        recorded_git=identity,
        recorded_runtime_identity={"runtime": "same"},
        recorded_physx_schema=physx_schema,
        recorded_physx_field_schema={},
        stable_recording_origin={
            **_natural_settle_history_for_replay_test(),
            "final_boundary_state_write": {
                "robot_joints": 2,
                "rigid_bodies": 1,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 0,
            },
            "final_boundary_sleep": None,
        },
        action_storage_layout="flat_raw_env_action",
        actions=np.asarray([[7.0, 8.0, 9.0]], dtype=np.float64),
        terminals=np.asarray([1], dtype=np.uint8),
        succeeds=np.asarray([1], dtype=np.uint8),
        field_specs=field_specs,
        current_fields={name: np.asarray([value]) for name, value in initial.items()},
        next_fields={name: np.asarray([value]) for name, value in expected.items()},
        physical_transition_contract=(RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT),
        trajectory_input_mode=RECORDED_DRIVE_TARGET_INPUT_MODE,
        fresh_recorder_process_per_attempt_required=False,
        fresh_scene_replay_required=True,
        recorder_lifecycle="persistent_process",
        source_process_reuse_enabled=True,
    )

    monkeypatch.setattr(
        "scripts.certify_physx_replay.capture_git_replay_state",
        lambda *_args, **_kwargs: dict(identity),
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay.capture_runtime_replay_identity",
        lambda: {"runtime": "same"},
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._construct_environment",
        lambda *_args, **_kwargs: environment,
    )

    replayed_origin = json.loads(json.dumps(demo.stable_recording_origin))
    replayed_origin["physics_steps"] = 204
    replayed_origin["stability_samples"] = 84
    replayed_origin["hold_target_sync"]["passes"] = 206
    replayed_origin["reset_drift_sample_count"] = 205
    replayed_origin["reset_drift_expected_sample_count"] = 205
    replayed_origin["initialization_contact_summary"]["sample_count"] = 205
    replayed_origin["initialization_contact_summary"]["expected_sample_count"] = 205
    monkeypatch.setattr(
        "scripts.certify_physx_replay.stabilize_recording_origin",
        lambda _env, _observation, info, **_kwargs: (
            dict(initial),
            {
                **info,
                "physical_recording_origin": replayed_origin,
            },
        ),
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._validate_runtime_trace_schema",
        lambda *_args, **_kwargs: None,
    )

    report = run_certification(demo)

    assert report["status"] == "passed"
    assert report["execution_mode"] == PHYSICS_ONLY_EXECUTION_MODE
    assert report["source_display_config"] == {
        "render_mode": "rgb_array",
        "image_obs": False,
        "headless": True,
    }
    assert report["effective_display_config"] == {
        "render_mode": "rgb_array",
        "image_obs": False,
        "headless": True,
    }
    assert report["app_window_visibility"] == {
        "policy": "hide_after_source_scene_construction",
        "status": "source_headless_no_window",
        "visible": False,
        "hide_call_performed": False,
    }
    assert report["wall_clock_throttle_hz"] == 0
    assert report["source_scene_constructed_before_display_override"] is True
    assert report["initialization_readback_match"] is True
    assert report["trace_match"] is True
    assert report["transition_outcomes_match"] is True
    assert report["replayed_stable_origin_history"]["physics_steps"] == 204
    assert report["post_stabilization_snapshot_restore_performed"] is False
    assert report["raw_commands_installed"] is True
    assert report["ik_recomputed"] is False
    assert report["raw_environment_step_called"] is False
    assert report["recorded_drive_targets_applied"] is True
    assert report["recorded_drive_target_substeps_expected"] == 2
    assert report["recorded_drive_target_substeps_executed"] == 2
    assert report["recorded_drive_target_writes_expected"] == 4
    assert report["recorded_drive_target_writes_executed"] == 4
    physics_match = report["recorded_physics_configuration_match"]
    assert physics_match["recorded_values"] == physics_match["runtime_values"]
    assert physics_match["recorded_values"] == {
        "device": "cuda:0",
        "n_substeps": 2,
        "physics_dt": 1.0 / 120.0,
    }
    assert report["trajectory_state_writes_performed"] is False
    assert report["source_recorder_lifecycle"] == "persistent_process"
    assert report["source_process_reuse_enabled"] is True
    assert "source_process_reused_across_episodes" not in report
    assert report["fresh_source_process_per_attempt"] is False
    assert report["fresh_scene_replay_required"] is True
    assert environment.unwrapped.step_calls == 0
    assert environment.unwrapped.ik_calls == 0
    assert len(environment.unwrapped.raw_actions) == 1
    np.testing.assert_array_equal(
        environment.unwrapped.raw_actions[0],
        [7.0, 8.0, 9.0],
    )
    assert len(applied_targets) == 2
    np.testing.assert_array_equal(applied_targets[0], target_path[0])
    np.testing.assert_array_equal(applied_targets[1], target_path[1])

    diagnostic_origin = json.loads(json.dumps(demo.stable_recording_origin))
    diagnostic_origin.update(
        {
            "stabilization_velocity_clamp": True,
            "initialization_per_physics_step_velocity_writes": True,
            "stabilization_velocity_clamp_scope": (
                VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS
            ),
            "stabilization_velocity_clamp_steps": 180,
            "stabilization_velocity_clamp_write": {
                "rigid_bodies": 1,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 0,
            },
        }
    )
    demo.stable_recording_origin.clear()
    demo.stable_recording_origin.update(diagnostic_origin)
    replayed_origin = json.loads(json.dumps(diagnostic_origin))

    diagnostic_report = run_certification(demo)

    assert diagnostic_report["status"] == "failed"
    assert diagnostic_report["reason"] == "replay_contract_is_diagnostic_only"


def test_e01_legacy_action_certification_uses_fresh_origin_without_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = {"physx_robot_qpos": np.asarray([1.0, 2.0])}
    expected = {"physx_robot_qpos": np.asarray([1.1, 2.1])}

    class Trace:
        def schema(self):
            return {"physx": {}, "fields": {}}

        def restore(self, *_args, **_kwargs):
            raise AssertionError("fresh E01 certification must not restore recorded t0")

        def begin_substep_capture(self):
            pass

    trace = Trace()

    class RawEnvironment:
        hz = 30
        _physx_ground_truth_trace = trace

        def __init__(self) -> None:
            self.actions: list[np.ndarray] = []
            self.ik_calls = 0

        def step(self, action):
            self.actions.append(np.asarray(action).copy())
            self.ik_calls += 1
            return dict(expected), 1.0, True, False, {"succeed": True}

    class Environment:
        def __init__(self) -> None:
            self.unwrapped = RawEnvironment()

        def reset(self, **_kwargs):
            return {"not_the_origin": np.asarray([99.0])}, {}

        def close(self):
            pass

    environment = Environment()
    identity = {
        "commit": "a" * 40,
        "dirty": False,
        "changed_files": [],
    }
    field_spec = PhysicalFieldSpec(
        "physx_robot_qpos",
        "next_physx_robot_qpos",
        (2,),
        np.dtype("float64"),
        1.0e-8,
    )
    demo = LoadedDemo(
        demo_dir=Path("/tmp/test-e01-fresh-legacy-origin"),
        task_name="e01_fold_garment",
        collection_run_seed=1,
        episode_reset_seed=2,
        randomize=True,
        randomize_dynamics=False,
        arm_type="rokae",
        hand_type="wuji1",
        render_mode="rgb_array",
        image_obs=False,
        headless=True,
        target_label="",
        camera_screen_effect=False,
        ipad_screen_effect=False,
        environment_variables={},
        recorded_git=identity,
        recorded_runtime_identity={"runtime": "same"},
        recorded_physx_schema={},
        recorded_physx_field_schema={},
        stable_recording_origin={
            **_fresh_origin_history(deformable_nodes=1),
            "final_boundary_state_write": {
                "robot_joints": 2,
                "rigid_bodies": 0,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 1,
            },
            "final_boundary_sleep": None,
        },
        action_storage_layout="flat_raw_env_action",
        actions=np.asarray([[7.0, 8.0, 9.0]], dtype=np.float64),
        terminals=np.asarray([1], dtype=np.uint8),
        succeeds=np.asarray([1], dtype=np.uint8),
        field_specs=(field_spec,),
        current_fields={"physx_robot_qpos": np.asarray([initial["physx_robot_qpos"]])},
        next_fields={"physx_robot_qpos": np.asarray([expected["physx_robot_qpos"]])},
        physical_transition_contract=LEGACY_RAW_ACTION_TRANSITION_CONTRACT,
        trajectory_input_mode=LEGACY_RAW_ACTION_INPUT_MODE,
        fresh_recorder_process_per_attempt_required=True,
    )

    monkeypatch.setattr(
        "scripts.certify_physx_replay.capture_git_replay_state",
        lambda *_args, **_kwargs: dict(identity),
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay.capture_runtime_replay_identity",
        lambda: {"runtime": "same"},
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._construct_environment",
        lambda *_args, **_kwargs: environment,
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay.stabilize_recording_origin",
        lambda _env, _observation, info, **_kwargs: (
            dict(initial),
            {
                **info,
                "physical_recording_origin": dict(demo.stable_recording_origin),
            },
        ),
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._validate_runtime_trace_schema",
        lambda *_args, **_kwargs: None,
    )

    report = run_certification(demo)

    assert report["status"] == "passed"
    assert report["legacy_raw_action_diagnostic_only"] is False
    assert report["post_stabilization_snapshot_restore_performed"] is False
    assert report["raw_commands_installed"] is False
    assert report["ik_recomputed"] is True
    assert report["raw_environment_step_called"] is True
    assert report["recorded_drive_targets_applied"] is False
    assert environment.unwrapped.ik_calls == 1
    np.testing.assert_array_equal(
        environment.unwrapped.actions[0],
        [7.0, 8.0, 9.0],
    )


def test_physx_schema_canonicalizes_process_local_path_ids() -> None:
    recorded = {
        "contact_report": {
            "path_id_encoding": "temporary",
            "path_table": [
                {"id": 91, "path": "/World/B"},
                {"id": 17, "path": "/World/A"},
            ],
        },
        "state_layout": ["position_xyz"],
    }
    replay = {
        "contact_report": {
            "path_id_encoding": "temporary",
            "path_table": [
                {"id": 888, "path": "/World/A"},
                {"id": 999, "path": "/World/B"},
            ],
        },
        "state_layout": ["position_xyz"],
    }

    assert _canonical_physx_schema(recorded) == _canonical_physx_schema(replay)


def _install_legacy_zero_padding_contact_frame(
    demo_dir: Path,
) -> None:
    schema_path = demo_dir / "recording_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    contact = schema["environment_schema"]["physx"]["contact_report"]
    contact.update(
        {
            "available_in_this_runtime": True,
            "max_headers": 2,
            "max_contact_points": 4,
            "max_friction_anchors": 4,
            "capture_source_encoding": {
                "0": "unavailable",
                "1": "full_contact_report",
                "2": "rigid_contact_tensor",
            },
            "tensor_fallback_available_in_this_runtime": True,
            "tensor_sensor_paths": ["/World/A"],
            "tensor_filter_paths": ["/World/B"],
            "tensor_event_type_sentinel": int(np.uint32(0xFFFFFFFE)),
            "tensor_source_actor_and_collider_ids_are_body_ids": True,
            "tensor_source_face_and_material_ids_available": False,
            "tensor_source_friction_rows_are_points_not_cache_anchors": True,
            "replay_restore_supported": False,
        }
    )
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    data = zarr.open_group(str(demo_dir / "replay.zarr"), mode="a")["data"]
    for name in (
        "physx_contact_signature_valid",
        "physx_contact_signature_pair_count",
        "physx_contact_signature_pair_path_indices",
        "physx_contact_signature_pair_counts",
        "physx_contact_signature_pair_metrics",
    ):
        values = np.asarray(data[name])
        values[0].fill(0)
        data[name][:] = values
    values = np.asarray(data["physx_contact_report_valid"])
    values[0] = 1
    data["physx_contact_report_valid"][:] = values
    values = np.asarray(data["physx_contact_capture_source"])
    values[0] = 2
    data["physx_contact_capture_source"][:] = values
    values = np.asarray(data["physx_contact_header_count"])
    values[0] = 1
    data["physx_contact_header_count"][:] = values
    values = np.asarray(data["physx_contact_header_ids"])
    values[0, 0] = [11, 12, 11, 12, 99]
    data["physx_contact_header_ids"][:] = values
    values = np.asarray(data["physx_contact_header_meta"])
    values[0, 0] = [
        np.uint32(0xFFFFFFFE),
        0,
        2,
        0,
        1,
        np.iinfo(np.uint32).max,
        np.iinfo(np.uint32).max,
    ]
    data["physx_contact_header_meta"][:] = values
    values = np.asarray(data["physx_contact_point_count"])
    values[0] = 2
    data["physx_contact_point_count"][:] = values
    values = np.asarray(data["physx_contact_point_data"])
    values[0, 0] = [1, 2, 3, 0, 0, 1, 0, 0, 2, -0.01]
    values[0, 1] = 0
    data["physx_contact_point_data"][:] = values
    values = np.asarray(data["physx_contact_point_face_indices"])
    values[0, :2] = np.iinfo(np.uint32).max
    data["physx_contact_point_face_indices"][:] = values
    values = np.asarray(data["physx_friction_anchor_count"])
    values[0] = 1
    data["physx_friction_anchor_count"][:] = values
    values = np.asarray(data["physx_friction_anchor_data"])
    values[0, 0] = [4, 5, 6, 0, 0, 3]
    data["physx_friction_anchor_data"][:] = values


def test_load_demo_rebuilds_only_exact_legacy_tensor_zero_padding(
    tmp_path: Path,
) -> None:
    demo_dir = _write_demo(tmp_path)
    _install_legacy_zero_padding_contact_frame(demo_dir)

    first = load_demo(demo_dir)
    second = load_demo(demo_dir)

    segment = first.contact_signature_overlay_segment
    assert segment is not None
    assert segment["signature_frames_reconstructed"] == 1
    assert segment["control_frames_reconstructed"] == 1
    assert segment["substep_frames_reconstructed"] == 0
    assert segment["exact_zero_padding_rows_removed"] == 1
    assert segment == second.contact_signature_overlay_segment
    np.testing.assert_array_equal(
        first.current_fields["physx_contact_signature_valid"][0],
        [1],
    )
    np.testing.assert_array_equal(
        first.current_fields["physx_contact_signature_pair_path_indices"][0, 0],
        [1, 1, 2, 2],
    )
    np.testing.assert_array_equal(
        first.current_fields["physx_contact_signature_pair_counts"][0, 0],
        [1, 1],
    )
    np.testing.assert_allclose(
        first.current_fields["physx_contact_signature_pair_metrics"][0, 0],
        [0.01, 2.0, 3.0],
    )
    stored = zarr.open_group(str(demo_dir / "replay.zarr"), mode="r")["data"]
    np.testing.assert_array_equal(
        np.asarray(stored["physx_contact_signature_valid"])[0],
        [0],
    )


@pytest.mark.parametrize(
    ("field", "mutate", "message"),
    [
        (
            "physx_contact_capture_source",
            lambda values: values.__setitem__((0, 0), 1),
            "not a valid rigid-contact-tensor capture",
        ),
        (
            "physx_contact_point_data",
            lambda values: values.__setitem__((0, 1, 0), 1.0),
            "no exact ten-field zero",
        ),
        (
            "physx_contact_header_meta",
            lambda values: values.__setitem__((0, 0, 1), 1),
            "invalid active slice|coverage is not exact",
        ),
        (
            "physx_contact_header_ids",
            lambda values: values.__setitem__((0, 0, 0), 999),
            "unresolved path id|header 0 is invalid",
        ),
        (
            "physx_contact_signature_pair_metrics",
            lambda values: values.__setitem__((0, 0, 0), 1.0),
            "stored signature is not canonical zero",
        ),
    ],
)
def test_legacy_contact_overlay_fails_closed_for_other_corruption(
    tmp_path: Path,
    field: str,
    mutate,
    message: str,
) -> None:
    demo_dir = _write_demo(tmp_path)
    _install_legacy_zero_padding_contact_frame(demo_dir)
    data = zarr.open_group(str(demo_dir / "replay.zarr"), mode="a")["data"]
    values = np.asarray(data[field])
    mutate(values)
    data[field][:] = values

    with pytest.raises(CertificationError, match=message):
        load_demo(demo_dir)


@pytest.mark.parametrize(
    ("storage_name", "index"),
    [
        ("physx_contact_signature_valid", (0, 0)),
        ("next_physx_contact_signature_valid", (1, 0)),
        ("physx_substep_contact_signature_valid", (0, 0, 0)),
        ("next_physx_substep_contact_signature_valid", (1, 1, 0)),
    ],
)
def test_load_demo_rejects_invalid_canonical_contact_signature(
    tmp_path: Path,
    storage_name: str,
    index: tuple[int, ...],
) -> None:
    demo_dir = _write_demo(tmp_path)
    data = zarr.open_group(str(demo_dir / "replay.zarr"), mode="a")["data"]
    values = np.asarray(data[storage_name])
    values[index] = 0
    data[storage_name][:] = values

    with pytest.raises(
        CertificationError,
        match="no certifiable canonical contact signature",
    ):
        load_demo(demo_dir)


@pytest.mark.parametrize("failure", ["empty_coverage", "missing_path"])
def test_load_demo_rejects_incomplete_contact_path_evidence(
    tmp_path: Path,
    failure: str,
) -> None:
    demo_dir = _write_demo(tmp_path)
    schema_path = demo_dir / "recording_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    contact = schema["environment_schema"]["physx"]["contact_report"]
    if failure == "empty_coverage":
        contact["report_enabled_task_body_paths"] = []
    else:
        contact["path_table"].pop()
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(CertificationError, match="contact (replay|signature)"):
        load_demo(demo_dir)


@pytest.mark.parametrize(
    ("field", "index", "delta"),
    [
        ("physx_contact_signature_pair_path_indices", (0, 0), 1),
        ("physx_contact_signature_pair_metrics", (0, 0), 0.01),
        ("physx_contact_signature_pair_metrics", (0, 1), 0.01),
        ("physx_contact_signature_pair_metrics", (0, 2), 0.01),
    ],
)
def test_compare_snapshot_rejects_contact_topology_and_impulse_changes(
    field: str,
    index: tuple[int, int],
    delta: float,
) -> None:
    if field.endswith("path_indices"):
        dtype = np.dtype("uint32")
        shape = (2, 4)
        recorded_value = np.asarray(
            [[1, 2, 3, 4], [0, 0, 0, 0]],
            dtype=dtype,
        )
        tolerance = 0.0
    else:
        dtype = np.dtype("float64")
        shape = (2, 3)
        recorded_value = np.asarray(
            [[0.001, 0.2, 0.1], [0.0, 0.0, 0.0]],
            dtype=dtype,
        )
        tolerance = 1.0e-6
    actual_value = recorded_value.copy()
    actual_value[index] += delta
    spec = PhysicalFieldSpec(
        field,
        f"next_{field}",
        shape,
        dtype,
        tolerance,
        False,
    )

    _errors, divergence = compare_snapshot(
        {field: recorded_value},
        {field: actual_value},
        (spec,),
        phase="post_action",
        action_index=3,
    )

    assert divergence is not None
    assert divergence["field"] == field
    assert divergence["index"] == list(index)


def test_session_schema_comparison_ignores_process_local_path_ids() -> None:
    first = SimpleNamespace(
        recorded_physx_schema={
            "contact_report": {
                "path_table": [{"id": 17, "path": "/World/A"}],
            },
        },
    )
    second = SimpleNamespace(
        recorded_physx_schema={
            "contact_report": {
                "path_table": [{"id": 999, "path": "/World/A"}],
            },
        },
    )

    assert _shared_session_field_value(
        first, "recorded_physx_schema"
    ) == _shared_session_field_value(second, "recorded_physx_schema")


def test_load_demo_skips_diagnostic_non_replayable_physx_fields(
    tmp_path: Path,
) -> None:
    demo_dir = _write_demo(tmp_path)
    schema_path = demo_dir / "recording_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    runtime_diagnostic_entry = {
        "shape": [1],
        "dtype": "float64",
        "physical_replay": False,
        "replay_restore_supported": False,
    }
    diagnostic_entry = {
        "dtype": "float64",
        "shape_per_step": [1],
        "semantics": dict(runtime_diagnostic_entry),
    }
    schema["fields"]["physx_contact_impulse"] = diagnostic_entry
    schema["fields"]["next_physx_contact_impulse"] = diagnostic_entry
    schema["environment_schema"]["fields"]["physx_contact_impulse"] = (
        runtime_diagnostic_entry
    )
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    root = zarr.open_group(str(demo_dir / "replay.zarr"), mode="a")
    root["data"].array(
        "physx_contact_impulse",
        np.ones((2, 1), dtype=np.float64),
    )
    root["data"].array(
        "next_physx_contact_impulse",
        np.ones((2, 1), dtype=np.float64),
    )

    demo = load_demo(demo_dir)

    assert "physx_contact_impulse" not in {spec.name for spec in demo.field_specs}


def test_step_preserves_complete_right_then_left_action_layout() -> None:
    class RawEnv:
        def __init__(self) -> None:
            self.received = None

        def step(self, action):
            self.received = action
            return {}, 0.0, False, False, {}

    class Wrapped:
        def __init__(self) -> None:
            self.unwrapped = RawEnv()

    env = Wrapped()
    action = np.arange(54, dtype=np.float64)
    _step_with_recorded_action(
        env,
        action,
        storage_layout="right_pose_hand_then_left_pose_hand",
    )

    np.testing.assert_array_equal(env.unwrapped.received["right"], action[:27])
    np.testing.assert_array_equal(env.unwrapped.received["left"], action[27:])


def test_decode_recorded_action_copies_bimanual_layout() -> None:
    action = np.arange(54, dtype=np.float64)

    decoded = _decode_recorded_action(
        action,
        storage_layout="right_pose_hand_then_left_pose_hand",
    )

    assert set(decoded) == {"right", "left"}
    np.testing.assert_array_equal(decoded["right"], action[:27])
    np.testing.assert_array_equal(decoded["left"], action[27:])
    decoded["right"][0] = -1.0
    assert action[0] == 0.0


def test_flat_bimanual_action_layout_is_preserved_before_storage() -> None:
    action = np.arange(54, dtype=np.float64)

    assert (
        _action_storage_layout_for_action(action)
        == "right_pose_left_pose_right_hand_left_hand"
    )
    assert (
        _action_storage_layout_for_action(
            {"right": action[:27], "left": action[27:]}
        )
        == "right_pose_hand_then_left_pose_hand"
    )


def test_resolve_legacy_grouped_bimanual_layout_from_recorded_commands() -> None:
    right_pose = np.asarray([0.1, 0.2, 1.07, 1.0, 0.0, 0.0, 0.0])
    left_pose = np.asarray([-0.1, 0.3, 1.02, 1.0, 0.0, 0.0, 0.0])
    right_hand = np.linspace(0.0, 0.19, 20)
    left_hand = np.linspace(0.2, 0.39, 20)
    action = np.concatenate(
        [right_pose, left_pose, right_hand, left_hand]
    )[None, :]
    next_fields = {
        "physx_command_tcp_pose": np.stack(
            [right_pose, left_pose], axis=0
        )[None, ...],
        "physx_command_hand": np.stack(
            [right_hand, left_hand], axis=0
        )[None, ...],
    }

    resolved = _resolve_action_storage_layout(
        "right_pose_hand_then_left_pose_hand",
        action,
        next_fields,
    )

    assert resolved == "right_pose_left_pose_right_hand_left_hand"
    decoded = _decode_recorded_action(action[0], storage_layout=resolved)
    np.testing.assert_array_equal(decoded, action[0])


def test_resolve_preserves_true_per_side_bimanual_layout() -> None:
    right_pose = np.asarray([0.1, 0.2, 1.07, 1.0, 0.0, 0.0, 0.0])
    left_pose = np.asarray([-0.1, 0.3, 1.02, 1.0, 0.0, 0.0, 0.0])
    right_hand = np.linspace(0.0, 0.19, 20)
    left_hand = np.linspace(0.2, 0.39, 20)
    action = np.concatenate(
        [right_pose, right_hand, left_pose, left_hand]
    )[None, :]
    next_fields = {
        "physx_command_tcp_pose": np.stack(
            [right_pose, left_pose], axis=0
        )[None, ...],
        "physx_command_hand": np.stack(
            [right_hand, left_hand], axis=0
        )[None, ...],
    }

    assert (
        _resolve_action_storage_layout(
            "right_pose_hand_then_left_pose_hand",
            action,
            next_fields,
        )
        == "right_pose_hand_then_left_pose_hand"
    )


def test_install_recorded_raw_command_bypasses_step_and_ik() -> None:
    events: list[str] = []

    class RawEnv:
        def _apply_action(self, action):
            events.append("raw_command")
            np.testing.assert_array_equal(action, [1.0, 2.0, 3.0])

        def _apply_joint_targets(self):
            raise AssertionError("online IK must not run")

        def step(self, _action):
            raise AssertionError("env.step must not run")

    class Environment:
        unwrapped = RawEnv()

    _install_recorded_raw_command_without_ik(
        Environment(),
        np.asarray([1.0, 2.0, 3.0]),
        storage_layout="flat_raw_env_action",
    )

    assert events == ["raw_command"]


def test_install_recorded_task_controller_state_precedes_raw_command() -> None:
    events: list[tuple[str, object]] = []

    class RawEnv:
        def set_physical_replay_task_controller_state(self, state):
            events.append(("controller", np.asarray(state).copy()))

        def _apply_action(self, action):
            events.append(("raw_command", np.asarray(action).copy()))

    class Environment:
        unwrapped = RawEnv()

    recorded_next = {
        "physx_task_controller_state": np.asarray([6, 1, 0], dtype=np.uint8)
    }
    installed = _install_recorded_task_controller_state(
        Environment(),
        recorded_next,
    )
    _install_recorded_raw_command_without_ik(
        Environment(),
        np.asarray([1.0, 2.0, 3.0]),
        storage_layout="flat_raw_env_action",
    )

    assert installed is True
    assert [event[0] for event in events] == ["controller", "raw_command"]
    np.testing.assert_array_equal(events[0][1], [6, 1, 0])


def test_install_recorded_task_controller_state_is_optional_for_other_tasks() -> None:
    assert _install_recorded_task_controller_state(object(), {}) is False


@pytest.mark.parametrize(
    "value",
    (
        np.asarray([6, 1], dtype=np.uint8),
        np.asarray([6, 1, 0], dtype=np.int64),
    ),
)
def test_install_recorded_task_controller_state_rejects_invalid_field(value) -> None:
    class RawEnv:
        def set_physical_replay_task_controller_state(self, _state):
            raise AssertionError("invalid state must not reach the runtime")

    with pytest.raises(CertificationError, match="uint8 shape"):
        _install_recorded_task_controller_state(
            SimpleNamespace(unwrapped=RawEnv()),
            {"physx_task_controller_state": value},
        )


def test_recorded_drive_target_path_rejects_incomplete_joint_targets() -> None:
    schema = {
        "physx": {
            "robot_sides": ["right"],
            "robot_joint_names": {"right": ["joint0", "joint1"]},
        }
    }
    targets = np.asarray([[[0.1, 0.2]], [[0.3, 0.4]]], dtype=np.float64)
    valid = np.ones(targets.shape, dtype=np.uint8)
    valid[1, 0, 1] = 0
    snapshot = {
        "physx_substep_count": np.asarray([2], dtype=np.int64),
        "physx_substep_robot_joint_target": targets,
        "physx_substep_robot_joint_target_valid": valid,
        "physx_robot_joint_target": targets[-1],
        "physx_robot_joint_target_valid": np.ones((1, 2), dtype=np.uint8),
    }

    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        RecordedArticulatedDrivePath(schema).control_step(snapshot)


def test_certification_report_is_bound_to_manifest_schema_and_zarr(
    tmp_path: Path,
) -> None:
    demo = _write_demo(tmp_path, exact_robot_substeps=True)
    report = _valid_recorded_drive_target_report(demo)
    write_report_atomic(demo, report)

    assert certification_upload_blocker(demo) is None
    root = zarr.open_group(str(demo / "replay.zarr"), mode="a")
    root["data"]["action"][0, 0] = 1.0
    blocker = certification_upload_blocker(demo)
    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_stale"


def test_certification_report_rejects_state_writes_during_action_replay(
    tmp_path: Path,
) -> None:
    demo = _write_demo(tmp_path, exact_robot_substeps=True)
    report = _valid_recorded_drive_target_report(demo)
    report["trajectory_state_writes_performed"] = True
    report["state_writes_after_action_start"] = True
    write_report_atomic(demo, report)

    blocker = certification_upload_blocker(demo)

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_invalid"
    assert "trajectory_state_writes_performed" in blocker["detail"]


@pytest.mark.parametrize(
    "missing_field",
    [
        "initialization_mode",
        "initialization_restore_counts",
        "initialization_restore_counts_source",
        "initialization_readback_match",
        "post_stabilization_snapshot_restore_performed",
        "physics_steps_between_initialization_readback_and_action_0",
    ],
)
def test_upload_rejects_report_without_fresh_origin_contract(
    tmp_path: Path,
    missing_field: str,
) -> None:
    demo = _write_demo(tmp_path, exact_robot_substeps=True)
    report = _valid_recorded_drive_target_report(demo)
    report.pop(missing_field)
    write_report_atomic(demo, report)

    blocker = certification_upload_blocker(demo)

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_invalid"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("initialization_restore_counts", {}),
        (
            "initialization_restore_counts",
            {
                "robot_joints": 0,
                "rigid_bodies": 0,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 0,
            },
        ),
        (
            "initialization_restore_counts",
            {
                "robot_joints": True,
                "rigid_bodies": 0,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 0,
            },
        ),
        (
            "physics_steps_between_initialization_readback_and_action_0",
            False,
        ),
        (
            "physics_steps_between_initialization_readback_and_action_0",
            0.0,
        ),
        ("post_stabilization_snapshot_restore_performed", True),
    ],
)
def test_upload_rejects_forged_fresh_origin_evidence(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    demo = _write_demo(tmp_path, exact_robot_substeps=True)
    report = _valid_recorded_drive_target_report(demo)
    report[field] = invalid_value
    write_report_atomic(demo, report)

    blocker = certification_upload_blocker(demo)

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_invalid"


@pytest.mark.parametrize("manifest_state", ["missing", "false", "corrupt"])
def test_physx_trace_cannot_disable_required_certification_via_manifest(
    tmp_path: Path,
    manifest_state: str,
) -> None:
    demo = _write_demo(tmp_path)
    manifest_path = demo / "replay_manifest.json"
    if manifest_state == "missing":
        manifest_path.unlink()
        expected_code = "physical_replay_manifest_invalid"
    elif manifest_state == "corrupt":
        manifest_path.write_text("{broken", encoding="utf-8")
        expected_code = "physical_replay_manifest_invalid"
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["recording"]["physical_replay_certification_required"] = False
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        expected_code = "physical_replay_certification_required_invalid"

    for blocker_function in (
        certification_upload_blocker,
        fresh_process_acceptance_upload_blocker,
        physical_replay_upload_blocker,
    ):
        blocker = blocker_function(demo)
        assert blocker is not None
        assert blocker["code"] == expected_code


def test_legacy_non_physx_episode_without_manifest_remains_uploadable(
    tmp_path: Path,
) -> None:
    demo = tmp_path / "legacy"
    (demo / "replay.zarr" / "data" / "action").mkdir(parents=True)
    (demo / "recording_schema.json").write_text(
        json.dumps({"environment_schema": {}, "fields": {"action": {}}}),
        encoding="utf-8",
    )

    assert certification_upload_blocker(demo) is None
    assert fresh_process_acceptance_upload_blocker(demo) is None


@pytest.mark.parametrize(
    "recording_hint",
    [
        {"sim_backend": "isaacsim"},
        {"record_physx_ground_truth": True},
        {"physical_replay_certification_runs_required": 3},
        {"fresh_recorder_process_per_episode_required": True},
        {"publish_only_after_fresh_certification": True},
    ],
)
def test_manifest_physical_hints_cannot_disable_certification(
    tmp_path: Path,
    recording_hint: dict[str, object],
) -> None:
    demo = tmp_path / "physical_hint"
    (demo / "replay.zarr" / "data" / "action").mkdir(parents=True)
    (demo / "recording_schema.json").write_text("{}", encoding="utf-8")
    (demo / "replay_manifest.json").write_text(
        json.dumps(
            {
                "recording": {
                    **recording_hint,
                    "physical_replay_certification_required": False,
                }
            }
        ),
        encoding="utf-8",
    )

    blocker = certification_upload_blocker(demo)

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_required_invalid"


@pytest.mark.parametrize(
    "anchor_name",
    [
        "physical_replay_certification.json",
        "fresh_process_acceptance.json",
        "physical_replay_certification_runs",
    ],
)
def test_certification_artifact_anchor_requires_valid_manifest(
    tmp_path: Path,
    anchor_name: str,
) -> None:
    demo = tmp_path / "anchored"
    (demo / "replay.zarr" / "data" / "action").mkdir(parents=True)
    (demo / "recording_schema.json").write_text("{}", encoding="utf-8")
    anchor = demo / anchor_name
    if anchor_name.endswith(".json"):
        anchor.write_text("{}", encoding="utf-8")
    else:
        anchor.mkdir()

    blocker = certification_upload_blocker(demo)

    assert blocker is not None
    assert blocker["code"] == "physical_replay_manifest_invalid"


def test_certify_demo_writes_report_before_closing_isaac(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = _write_demo(tmp_path)
    report_path = demo_dir / "physical_replay_certification.json"
    close_observations: list[dict[str, object]] = []

    class Environment:
        def close(self) -> None:
            close_observations.append(
                json.loads(report_path.read_text(encoding="utf-8"))
            )

    def fake_run_certification(demo, *, on_environment_created=None):
        assert on_environment_created is not None
        on_environment_created(Environment())
        return {
            "status": "passed",
            "reason": "continuous_physx_replay_matched",
            "portable": True,
            "source_identity": "clean_commit_match",
            "task_success_replay": True,
            "trace_match": True,
            "transition_outcomes_match": True,
            "certification_mode": CERTIFICATION_MODE,
            "initialization_mode": INITIALIZATION_MODE,
            "initialization_restore_counts": {
                "robot_joints": 2,
                "rigid_bodies": 0,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 0,
            },
            "initialization_readback_match": True,
            "physics_steps_between_restore_and_action_0": 0,
            "state_writes_performed": True,
            "initialization_state_writes_performed": True,
            "trajectory_state_writes_performed": False,
            "state_writes_after_action_start": False,
            "steps_expected": 2,
            "steps_executed": 2,
        }

    monkeypatch.setattr(
        "scripts.certify_physx_replay.run_certification",
        fake_run_certification,
    )

    report = certify_demo(demo_dir)

    assert report["status"] == "passed"
    assert len(close_observations) == 1
    assert close_observations[0]["status"] == "passed"
    assert close_observations[0]["certified_inputs"] == (
        capture_certified_inputs(demo_dir)
    )


def test_certify_demo_writes_failure_before_closing_isaac(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = _write_demo(tmp_path)
    report_path = demo_dir / "physical_replay_certification.json"
    status_seen_during_close: list[str] = []

    class Environment:
        def close(self) -> None:
            status_seen_during_close.append(
                json.loads(report_path.read_text(encoding="utf-8"))["status"]
            )

    def fake_run_certification(demo, *, on_environment_created=None):
        assert on_environment_created is not None
        on_environment_created(Environment())
        raise CertificationError("intentional replay divergence")

    monkeypatch.setattr(
        "scripts.certify_physx_replay.run_certification",
        fake_run_certification,
    )

    report = certify_demo(demo_dir)

    assert report["status"] == "failed"
    assert status_seen_during_close == ["failed"]


def _accepted_session_demo(
    base: LoadedDemo,
    demo_dir: Path,
    *,
    ordinal: int,
) -> LoadedDemo:
    demo_dir.mkdir()
    return replace(
        base,
        demo_dir=demo_dir.resolve(),
        episode_reset_seed=(base.collection_run_seed + ordinal) % (2**31),
        recorder_lifecycle="persistent_process",
        source_process_reuse_enabled=True,
        collection_seed_source="explicit_cli_seed",
        episode_reset_ordinal=ordinal,
        episode_seed_derivation=(
            "collection_run_seed_plus_reset_attempt_index_mod_2pow31"
        ),
        source_recorder_pid=12345,
        session_history_root_relative_path=(
            ".physical_replay_pending/.physx_session_history/test-session"
        ),
        session_history_sha256="a" * 64,
        session_history_event_count=100,
        session_history_accepted_event_ordinal=ordinal * 2 + 1,
    )


def _mock_loaded_session(
    tmp_path: Path,
    demos: tuple[LoadedDemo, ...],
) -> LoadedSessionReplay:
    last_accepted = demos[-1].session_history_accepted_event_ordinal
    events = tuple({} for _ in range(last_accepted + 1))
    return LoadedSessionReplay(
        path=tmp_path / "history",
        session={"session_id": "test-session"},
        events=events,
        finalization={
            "certified_through_event_ordinal": last_accepted,
        },
    )


def test_session_replay_rebuilds_the_recorded_physx_scene_generation() -> None:
    calls: list[int] = []

    class RawEnvironment:
        def hard_reset_physx_scene(self, *, seed: int):
            calls.append(seed)
            return {
                "scene_rebuilt": True,
                "scene_generation": len(calls),
                "seed": seed,
                "process_restarted": False,
            }

    class WrappedEnvironment:
        unwrapped = RawEnvironment()

    result = _rebuild_session_scene_if_recorded(
        WrappedEnvironment(),
        {
            "seed": 1235,
            "scene_rebuilt_before_reset": True,
            "scene_generation": 1,
        },
        event_ordinal=4,
    )

    assert calls == [1235]
    assert result == {
        "scene_rebuilt": True,
        "scene_generation": 1,
        "seed": 1235,
        "process_restarted": False,
    }


def test_session_replay_rejects_generation_without_a_rebuild_marker() -> None:
    with pytest.raises(CertificationError, match="nonzero scene generation"):
        _rebuild_session_scene_if_recorded(
            object(),
            {
                "scene_rebuilt_before_reset": False,
                "scene_generation": 2,
            },
            event_ordinal=4,
        )


def test_validated_session_history_binds_path_hash_and_accepted_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    base = load_demo(_write_demo(fixture_root))
    output_root = tmp_path / "output"
    pending = output_root / ".physical_replay_pending"
    demo_dir = pending / "accepted-demo"
    demo_dir.mkdir(parents=True)
    relative_history = ".physical_replay_pending/.physx_session_history/session-a"
    history_path = output_root / relative_history
    history_path.mkdir(parents=True)
    demo = replace(
        base,
        demo_dir=demo_dir.resolve(),
        recorder_lifecycle="persistent_process",
        source_process_reuse_enabled=True,
        collection_seed_source="explicit_cli_seed",
        episode_reset_ordinal=1,
        episode_seed_derivation=(
            "collection_run_seed_plus_reset_attempt_index_mod_2pow31"
        ),
        source_recorder_pid=12345,
        session_history_root_relative_path=relative_history,
        session_history_sha256="d" * 64,
        session_history_event_count=3,
        session_history_accepted_event_ordinal=2,
    )
    hashes = [character * 64 for character in "abc"]
    origin = dict(demo.stable_recording_origin)
    events = [
        {
            "ordinal": 0,
            "kind": "reset_attempt",
            "previous_event_sha256": None,
            "event_sha256": hashes[0],
            "payload": {
                "reset_ordinal": 0,
                "seed": demo.collection_run_seed,
                "outcome": "rejected_unstable_origin",
                "accepted": False,
            },
        },
        {
            "ordinal": 1,
            "kind": "reset_attempt",
            "previous_event_sha256": hashes[0],
            "event_sha256": hashes[1],
            "payload": {
                "reset_ordinal": 1,
                "seed": demo.episode_reset_seed,
                "outcome": "accepted",
                "accepted": True,
                "stable_origin": origin,
                "physics_steps": origin["physics_steps"],
            },
        },
        {
            "ordinal": 2,
            "kind": "trajectory",
            "previous_event_sha256": hashes[1],
            "event_sha256": hashes[2],
            "payload": {
                "reset_ordinal": 1,
                "accepted": True,
                "storage": "accepted_demo_reference",
                "demo_id": demo.demo_dir.name,
                "step_count": len(demo.actions),
            },
        },
    ]
    loaded = {
        "session": {
            "session_id": "session-a",
            "task_name": demo.task_name,
            "collection_run_seed": demo.collection_run_seed,
        },
        "events": events,
        "finalization": {
            "task_name": demo.task_name,
            "collection_run_seed": demo.collection_run_seed,
            "event_count": 3,
            "event_sha256_chain": hashes,
            "history_identity": {"sha256": "d" * 64},
            "accepted_demos": [
                {
                    "event_ordinal": 2,
                    "reset_ordinal": 1,
                    "demo_id": demo.demo_dir.name,
                    "step_count": len(demo.actions),
                }
            ],
            "certified_through_event_ordinal": 2,
        },
    }
    monkeypatch.setattr(
        "scripts.certify_physx_replay.load_session_history",
        lambda path, require_finalized=True: (
            loaded
            if Path(path) == history_path.resolve() and require_finalized
            else pytest.fail("unexpected history path")
        ),
    )

    session = _validated_session_history([demo])

    assert session.path == history_path.resolve()
    assert session.events[2]["payload"]["demo_id"] == demo.demo_dir.name
    with pytest.raises(CertificationError, match="SHA-256"):
        _validated_session_history([replace(demo, session_history_sha256="e" * 64)])


def test_platform_session_history_includes_event_directories(
    tmp_path: Path,
) -> None:
    history = tmp_path / "history"
    events = history / "events"
    (events / "00000000").mkdir(parents=True)
    (events / "00000001").mkdir()
    (history / "session.json").write_text(
        json.dumps({"session_id": "session-a"}),
        encoding="utf-8",
    )
    (history / "FINALIZED.json").write_text(
        json.dumps({"event_count": 2}),
        encoding="utf-8",
    )
    for ordinal in range(2):
        (events / f"{ordinal:08d}" / "event.json").write_text(
            json.dumps({"ordinal": ordinal}),
            encoding="utf-8",
        )

    loaded = _load_platform_session_history(history)

    assert [event["event_directory"] for event in loaded["events"]] == [
        "events/00000000",
        "events/00000001",
    ]


def test_load_demo_parses_all_or_none_session_history_binding(
    tmp_path: Path,
) -> None:
    demo_dir = _write_demo(tmp_path)
    manifest_path = demo_dir / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recording = manifest["recording"]
    recording.update(
        {
            "physical_replay_session_history_root_relative_path": (
                ".physical_replay_pending/.physx_session_history/session"
            ),
            "physical_replay_session_history_sha256": "f" * 64,
            "physical_replay_session_history_event_count": 4,
            "physical_replay_session_history_accepted_event_ordinal": 3,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_demo(demo_dir)

    assert loaded.session_history_sha256 == "f" * 64
    assert loaded.session_history_event_count == 4
    assert loaded.session_history_accepted_event_ordinal == 3
    del recording["physical_replay_session_history_accepted_event_ordinal"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CertificationError, match="incomplete"):
        load_demo(demo_dir)


def test_single_demo_session_still_uses_verified_shared_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    base = load_demo(_write_demo(fixture_root))
    demo = _accepted_session_demo(base, tmp_path / "accepted", ordinal=0)
    session = _mock_loaded_session(tmp_path, (demo,))
    environment = object()
    closed: list[bool] = []

    class ClosableEnvironment:
        def close(self):
            closed.append(True)

    environment = ClosableEnvironment()
    monkeypatch.setattr(
        "scripts.certify_physx_replay.capture_certified_inputs",
        lambda _path: {"token": "same"},
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay.load_demo",
        lambda _path: demo,
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._validated_session_history",
        lambda demos: session if demos == [demo] else pytest.fail("bad demos"),
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._construct_environment",
        lambda _config, _demo: environment,
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._validate_accepted_demo_session_reset_report",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay.certify_demo",
        lambda *_args, **_kwargs: pytest.fail("single-demo bypass used"),
    )

    def replay(_environment, **kwargs):
        assert kwargs["session_events"] is session
        return {"status": "replayed", "events_replayed": 0}

    monkeypatch.setattr(
        "scripts.certify_physx_replay.run_certification",
        lambda _demo, **kwargs: (
            {"status": "passed", "reason": "matched"}
            if kwargs["environment"] is environment
            else pytest.fail("shared environment not used")
        ),
    )

    reports = certify_demo_session(
        [demo.demo_dir],
        event_replayer=replay,
    )

    assert reports[demo.demo_dir]["status"] == "passed"
    assert closed == [True]


def test_session_replay_sorts_accepted_demos_and_reuses_one_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    base = load_demo(_write_demo(fixture_root))
    first = _accepted_session_demo(base, tmp_path / "first", ordinal=2)
    second = _accepted_session_demo(base, tmp_path / "second", ordinal=7)
    demos_by_path = {
        first.demo_dir: first,
        second.demo_dir: second,
    }
    inputs_by_path = {
        first.demo_dir: {"token": "first"},
        second.demo_dir: {"token": "second"},
    }
    replay_session = _mock_loaded_session(tmp_path, (first, second))
    events: list[tuple[object, ...]] = []

    class Environment:
        def close(self) -> None:
            events.append(
                (
                    "close",
                    (first.demo_dir / "physical_replay_certification.json").is_file(),
                    (second.demo_dir / "physical_replay_certification.json").is_file(),
                )
            )

    shared_environment = Environment()

    monkeypatch.setattr(
        "scripts.certify_physx_replay.capture_certified_inputs",
        lambda path: dict(inputs_by_path[Path(path).resolve()]),
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay.load_demo",
        lambda path: demos_by_path[Path(path).resolve()],
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._validated_session_history",
        lambda demos: replay_session,
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._validate_accepted_demo_session_reset_report",
        lambda *_args: None,
    )

    def construct(_config, demo):
        events.append(("construct", demo.episode_reset_ordinal))
        return shared_environment

    monkeypatch.setattr(
        "scripts.certify_physx_replay._construct_environment",
        construct,
    )

    def replay_events(
        received_environment,
        *,
        previous_demo,
        next_demo,
        session_events,
    ):
        assert received_environment is shared_environment
        assert session_events is replay_session
        events.append(
            (
                "events",
                None if previous_demo is None else previous_demo.episode_reset_ordinal,
                next_demo.episode_reset_ordinal,
            )
        )
        return {"status": "replayed", "events_replayed": 0}

    def certify(demo, *, environment=None, on_environment_created=None):
        assert environment is shared_environment
        assert on_environment_created is None
        events.append(("certify", demo.episode_reset_ordinal))
        return {"status": "passed", "reason": "matched"}

    monkeypatch.setattr(
        "scripts.certify_physx_replay.run_certification",
        certify,
    )

    reports = certify_demo_session(
        [second.demo_dir, first.demo_dir],
        event_replayer=replay_events,
    )

    assert list(reports) == [first.demo_dir, second.demo_dir]
    assert events == [
        ("construct", 2),
        ("events", None, 2),
        ("certify", 2),
        ("events", 2, 7),
        ("certify", 7),
        ("close", True, True),
    ]
    for index, demo in enumerate((first, second)):
        report = reports[demo.demo_dir]
        assert report["status"] == "passed"
        assert report["certified_inputs"] == inputs_by_path[demo.demo_dir]
        assert report["session_replay"] == {
            "mode": SESSION_REPLAY_EXECUTION_MODE,
            "status": "passed",
            "shared_environment": True,
            "episode_index": index,
            "episode_count": 2,
            "ordered_episode_reset_ordinals": [2, 7],
            "session_history_id": "test-session",
            "session_history_sha256": "a" * 64,
            "session_history_event_count": 100,
            "accepted_reset_event_ordinal": (
                demo.session_history_accepted_event_ordinal - 1
            ),
            "accepted_event_ordinal": (demo.session_history_accepted_event_ordinal),
            "replayed_through_event_ordinal": (
                demo.session_history_accepted_event_ordinal
            ),
            "history_certified_through_event_ordinal": (
                second.session_history_accepted_event_ordinal
            ),
            "event_replay": {
                "status": "replayed",
                "events_replayed": 0,
            },
        }


def test_session_replay_skips_every_episode_after_first_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    base = load_demo(_write_demo(fixture_root))
    first = _accepted_session_demo(base, tmp_path / "first", ordinal=1)
    second = _accepted_session_demo(base, tmp_path / "second", ordinal=2)
    demos_by_path = {
        first.demo_dir: first,
        second.demo_dir: second,
    }
    inputs_by_path = {
        first.demo_dir: {"token": "first"},
        second.demo_dir: {"token": "second"},
    }
    replay_session = _mock_loaded_session(tmp_path, (first, second))
    certified_ordinals: list[int] = []
    event_ordinals: list[int] = []
    close_count = 0

    class Environment:
        def close(self) -> None:
            nonlocal close_count
            close_count += 1

    environment = Environment()
    monkeypatch.setattr(
        "scripts.certify_physx_replay.capture_certified_inputs",
        lambda path: dict(inputs_by_path[Path(path).resolve()]),
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay.load_demo",
        lambda path: demos_by_path[Path(path).resolve()],
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._validated_session_history",
        lambda demos: replay_session,
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._validate_accepted_demo_session_reset_report",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._construct_environment",
        lambda _config, _demo: environment,
    )

    def replay_events(_environment, **kwargs):
        event_ordinals.append(kwargs["next_demo"].episode_reset_ordinal)
        return {"status": "replayed", "events_replayed": 0}

    def certify(demo, **_kwargs):
        certified_ordinals.append(demo.episode_reset_ordinal)
        return {
            "status": "failed",
            "reason": "physical_state_diverged",
        }

    monkeypatch.setattr(
        "scripts.certify_physx_replay.run_certification",
        certify,
    )

    reports = certify_demo_session(
        [second.demo_dir, first.demo_dir],
        event_replayer=replay_events,
    )

    assert certified_ordinals == [1]
    assert event_ordinals == [1]
    assert close_count == 1
    assert reports[first.demo_dir]["status"] == "failed"
    skipped = reports[second.demo_dir]
    assert skipped["status"] == "failed"
    assert skipped["error_type"] == "SessionReplaySkipped"
    assert skipped["session_replay"]["status"] == ("skipped_due_to_prior_failure")
    assert skipped["session_replay"]["blocked_by"] == {
        "demo_dir": str(first.demo_dir),
        "episode_reset_ordinal": 1,
    }


def test_session_event_replayer_executes_rejected_and_discarded_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    base = load_demo(_write_demo(fixture_root))
    target = replace(
        _accepted_session_demo(base, tmp_path / "target", ordinal=2),
        session_history_event_count=5,
        session_history_accepted_event_ordinal=4,
    )
    origin = dict(base.stable_recording_origin)
    rejected_diagnostics = {
        "failure_phase": "stability_timeout",
        "physics_steps": 3,
        "stability_samples": 1,
        "hold_target_sync_passes": 4,
    }

    def reset_payload(reset_ordinal: int, *, accepted: bool):
        return {
            "reset_ordinal": reset_ordinal,
            "seed": base.collection_run_seed + reset_ordinal,
            "outcome": ("accepted" if accepted else "rejected_unstable_origin"),
            "accepted": accepted,
            "stable_kwargs": {},
            "stable_origin": origin if accepted else None,
            "stability_diagnostics": (None if accepted else rejected_diagnostics),
            "error": None if accepted else "did not settle",
            "physics_steps": (origin["physics_steps"] if accepted else 3),
        }

    events = (
        {"kind": "reset_attempt", "payload": reset_payload(0, accepted=False)},
        {"kind": "reset_attempt", "payload": reset_payload(1, accepted=True)},
        {
            "kind": "trajectory",
            "payload": {
                "reset_ordinal": 1,
                "accepted": False,
                "step_count": 2,
                "storage": "compact_replay_zarr",
                "replay_input_mode": "recorded_articulated_drive_targets",
            },
        },
        {"kind": "reset_attempt", "payload": reset_payload(2, accepted=True)},
        {
            "kind": "trajectory",
            "payload": {
                "reset_ordinal": 2,
                "accepted": True,
                "step_count": len(target.actions),
                "storage": "accepted_demo_reference",
                "demo_id": target.demo_dir.name,
            },
        },
    )
    session = LoadedSessionReplay(
        path=tmp_path / "history",
        session={"session_id": "test-session"},
        events=events,
        finalization={"certified_through_event_ordinal": 4},
    )
    reset_seeds: list[int] = []

    class Environment:
        def reset(self, **kwargs):
            reset_seeds.append(kwargs["seed"])
            return {}, {}

    stabilization_calls = 0

    def stabilize(_env, observation, _info, **_kwargs):
        nonlocal stabilization_calls
        stabilization_calls += 1
        if stabilization_calls == 1:
            from handson.sim.isaaclab.recording_origin import (
                RecordingOriginStabilityError,
            )

            raise RecordingOriginStabilityError(
                "did not settle",
                diagnostics=rejected_diagnostics,
            )
        return observation, {"physical_recording_origin": dict(origin)}

    replayed_segments: list[int] = []

    def replay_discarded(*_args, **kwargs):
        replayed_segments.append(kwargs["event_ordinal"])
        return {"actions_replayed": 2, "drive_substeps_replayed": 4}

    monkeypatch.setattr(
        "scripts.certify_physx_replay.stabilize_recording_origin",
        stabilize,
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay._replay_discarded_session_trajectory",
        replay_discarded,
    )

    report = replay_session_events(
        Environment(),
        previous_demo=None,
        next_demo=target,
        session_events=session,
    )

    assert reset_seeds == [17, 18]
    assert replayed_segments == [2]
    assert report["status"] == "replayed_verified_session_prefix"
    assert report["events_replayed"] == 3
    assert report["target_accepted_reset_event_ordinal"] == 3
    assert report["discarded_actions_replayed"] == 2
    assert report["discarded_drive_substeps_replayed"] == 4

    with pytest.raises(CertificationError, match="verified PhysX session"):
        replay_session_events(
            object(),
            previous_demo=None,
            next_demo=target,
        )


def test_rejected_reset_requires_the_same_failure_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = load_demo(_write_demo(tmp_path))
    recorded = {
        "failure_phase": "stability_timeout",
        "physics_steps": 180,
        "stability_samples": 60,
        "hold_target_sync_passes": 182,
    }
    payload = {
        "reset_ordinal": 0,
        "seed": demo.collection_run_seed,
        "outcome": "rejected_unstable_origin",
        "accepted": False,
        "stable_kwargs": {
            "reset_translation_drift_limit": 0.005,
            "reset_rotation_drift_limit": 0.05235987755982989,
            "require_valid_contact_report": True,
            "contact_penetration_limit": 0.0005,
        },
        "stable_origin": None,
        "stability_diagnostics": recorded,
        "error": "did not settle",
        "physics_steps": 180,
    }

    class Environment:
        def reset(self, **_kwargs):
            return {}, {}

    def fail_at_other_phase(*_args, **_kwargs):
        from handson.sim.isaaclab.recording_origin import (
            RecordingOriginStabilityError,
        )

        raise RecordingOriginStabilityError(
            "final boundary failed",
            diagnostics={
                **recorded,
                "failure_phase": "final_boundary_readback",
            },
        )

    monkeypatch.setattr(
        "scripts.certify_physx_replay.stabilize_recording_origin",
        fail_at_other_phase,
    )

    with pytest.raises(CertificationError, match="exact stability diagnostics"):
        from scripts.certify_physx_replay import (
            _replay_rejected_session_reset,
        )

        _replay_rejected_session_reset(
            Environment(),
            payload,
            demo=demo,
            event_ordinal=0,
        )


def _geometry_rejection_diagnostics() -> dict[str, object]:
    return {
        "failure_phase": "initial_geometry_gate",
        "geometry_failures": ["contact_penetration_exceeded"],
        "reset_max_translation_drift_m": 0.0002,
        "reset_max_rotation_drift_rad": 0.001,
        "reset_max_coordinate_drift": 0.0001,
        "reset_state_coverage": {
            "rigid_body": 1,
            "articulation_root": 0,
            "robot_root": 2,
            "tcp": 2,
            "articulation_dof": 0,
            "robot_joint": 54,
            "hand_command": 40,
            "deformable_node": 0,
        },
        "reset_drift_sample_count": 4,
        "reset_drift_expected_sample_count": 4,
        "reset_drift_all_physics_frames_sampled": True,
        "reset_pose_drift_peaks": [
            {
                "kind": "rigid_body",
                "index": 0,
                "name": "object",
                "translation_m": 0.0002,
                "rotation_rad": 0.001,
            }
        ],
        "contact_summary": {
            "valid": True,
            "sample_count": 4,
            "invalid_sample_count": 0,
            "capture_sources": [2],
            "maximum_contact_point_count": 2,
            "maximum_penetration_depth_m": 0.001,
            "deepest_sample_index": 1,
            "deepest_contact_point_index": 0,
            "error": None,
            "expected_sample_count": 4,
            "all_physics_frames_sampled": True,
        },
        "gate_contact_summary": {
            "valid": True,
            "capture_source": 2,
            "contact_point_count": 0,
            "maximum_penetration_depth_m": 0.0,
            "deepest_contact_point_index": None,
            "error": None,
        },
        "physics_steps": 3,
        "stability_samples": 1,
        "hold_target_sync_passes": 5,
    }


def test_rejected_geometry_reset_replays_exact_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.certify_physx_replay import _replay_rejected_session_reset

    demo = load_demo(_write_demo(tmp_path))
    recorded = _geometry_rejection_diagnostics()
    payload = {
        "reset_ordinal": 0,
        "seed": demo.collection_run_seed,
        "outcome": "rejected_unstable_origin",
        "accepted": False,
        "stable_kwargs": {
            "reset_translation_drift_limit": 0.005,
            "reset_rotation_drift_limit": 0.05235987755982989,
            "require_valid_contact_report": True,
            "contact_penetration_limit": 0.0005,
        },
        "stable_origin": None,
        "stability_diagnostics": recorded,
        "error": "penetrating reset",
        "physics_steps": 3,
    }

    class Environment:
        def reset(self, **_kwargs):
            return {}, {}

    def reject(*_args, **_kwargs):
        raise RecordingOriginStabilityError(
            "penetrating reset",
            diagnostics=json.loads(json.dumps(recorded)),
        )

    monkeypatch.setattr(
        "scripts.certify_physx_replay.stabilize_recording_origin",
        reject,
    )
    _replay_rejected_session_reset(Environment(), payload, demo=demo, event_ordinal=0)

    changed = json.loads(json.dumps(recorded))
    changed["contact_summary"]["maximum_penetration_depth_m"] = 0.002

    def reject_changed(*_args, **_kwargs):
        raise RecordingOriginStabilityError("penetrating reset", diagnostics=changed)

    monkeypatch.setattr(
        "scripts.certify_physx_replay.stabilize_recording_origin",
        reject_changed,
    )
    with pytest.raises(CertificationError, match="exact stability diagnostics"):
        _replay_rejected_session_reset(
            Environment(), payload, demo=demo, event_ordinal=0
        )


def test_rejected_missing_authored_snapshot_replays_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.certify_physx_replay import _replay_rejected_session_reset

    demo = load_demo(_write_demo(tmp_path))
    diagnostics = {
        "failure_phase": "authored_reset_snapshot",
        "geometry_failures": ["authored_reset_snapshot_missing"],
    }
    payload = {
        "reset_ordinal": 0,
        "seed": demo.collection_run_seed,
        "outcome": "rejected_unstable_origin",
        "accepted": False,
        "stable_kwargs": {},
        "stable_origin": None,
        "stability_diagnostics": diagnostics,
        "error": "missing snapshot",
        "physics_steps": None,
    }

    class Environment:
        def reset(self, **_kwargs):
            return {}, {}

    def reject(*_args, **_kwargs):
        raise RecordingOriginStabilityError(
            "missing snapshot", diagnostics=dict(diagnostics)
        )

    monkeypatch.setattr(
        "scripts.certify_physx_replay.stabilize_recording_origin",
        reject,
    )
    _replay_rejected_session_reset(Environment(), payload, demo=demo, event_ordinal=0)


def test_discarded_trajectory_installs_raw_commands_and_all_drive_substeps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = load_demo(_write_demo(tmp_path))
    arrays = {
        "action": np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=np.float64,
        ),
        "terminal": np.zeros((2,), dtype=np.uint8),
        "succeed": np.zeros((2,), dtype=np.uint8),
        "physx_robot_qpos": np.asarray(
            [[1.0, 2.0], [1.1, 2.1]],
            dtype=np.float64,
        ),
        "next_physx_robot_qpos": np.asarray(
            [[1.1, 2.1], [1.2, 2.2]],
            dtype=np.float64,
        ),
    }
    for spec in demo.field_specs:
        arrays.setdefault(
            spec.name,
            np.asarray(demo.current_fields[spec.name]),
        )
        arrays.setdefault(
            spec.next_name,
            np.asarray(demo.next_fields[spec.name]),
        )
    payload = {
        "step_count": 2,
        "storage": "compact_replay_zarr",
        "replay_input_mode": "recorded_articulated_drive_targets",
        "fields": sorted(arrays),
    }
    installed_actions: list[np.ndarray] = []

    class Trace:
        def schema(self):
            return {
                "name": "handson.physx_ground_truth",
                "version": 6,
                "physx": demo.recorded_physx_schema,
                "fields": demo.recorded_physx_field_schema,
            }

    class RawEnvironment:
        def __init__(self):
            self._physx_ground_truth_trace = Trace()
            self._manual_reset_requested = False

        def _apply_action(self, action):
            installed_actions.append(np.asarray(action).copy())

    class Environment:
        def __init__(self):
            self.unwrapped = RawEnvironment()

    class DriveStep:
        substep_count = 2

    drive_snapshots: list[np.ndarray] = []
    partial_authorizations: list[bool] = []

    class DrivePath:
        def __init__(self, _schema):
            pass

        def control_step(
            self,
            snapshot,
            *,
            allow_partial_terminal_guard=False,
        ):
            drive_snapshots.append(snapshot["physx_robot_qpos"].copy())
            partial_authorizations.append(
                allow_partial_terminal_guard is True
            )
            return DriveStep()

    step_index = 0
    partial_mode = False

    def run_drive(_raw_env, _trace, _schema, drive_step):
        nonlocal step_index
        assert drive_step.substep_count == 2
        observation = {
            spec.name: arrays[spec.next_name][step_index] for spec in demo.field_specs
        }
        step_index += 1
        info = {"succeed": False}
        if partial_mode and step_index == 2:
            info.update(
                {
                    "manual_reset": True,
                    "numerical_guard_event": {
                        "body": "hanoi_disk_large",
                        "reason": "rigid_motion_spike",
                    },
                }
            )
        return (
            (observation, 0.0, False, False, info),
            {"recorded_substep_count": 2},
        )

    import handson.sim.isaaclab.articulated_replay as articulated_replay

    monkeypatch.setattr(
        "scripts.certify_physx_replay.load_discarded_trajectory",
        lambda *_args, **_kwargs: arrays,
    )
    monkeypatch.setattr(
        articulated_replay,
        "RecordedArticulatedDrivePath",
        DrivePath,
    )
    monkeypatch.setattr(
        articulated_replay,
        "run_articulated_drive_control_step",
        run_drive,
    )
    monkeypatch.setattr(
        articulated_replay,
        "validate_recorded_physics_configuration",
        lambda *_args, **_kwargs: {},
    )
    session = LoadedSessionReplay(
        path=tmp_path / "history",
        session={"session_id": "session"},
        events=(),
        finalization={"certified_through_event_ordinal": 0},
    )

    report = _replay_discarded_session_trajectory(
        Environment(),
        session=session,
        event_ordinal=3,
        payload=payload,
        demo=demo,
        stable_observation={
            spec.name: arrays[spec.name][0] for spec in demo.field_specs
        },
    )

    assert report == {
        "actions_replayed": 2,
        "drive_substeps_replayed": 4,
    }
    assert step_index == 2
    assert np.array_equal(installed_actions[0], arrays["action"][0])
    assert np.array_equal(installed_actions[1], arrays["action"][1])
    assert np.array_equal(drive_snapshots[1], arrays["next_physx_robot_qpos"][1])
    assert partial_authorizations == [False, False]

    step_index = 0
    partial_mode = True
    partial_authorizations.clear()
    partial_payload = dict(payload)
    partial_payload["partial_terminal_guard"] = True
    partial_report = _replay_discarded_session_trajectory(
        Environment(),
        session=session,
        event_ordinal=3,
        payload=partial_payload,
        demo=demo,
        stable_observation={
            spec.name: arrays[spec.name][0] for spec in demo.field_specs
        },
    )
    assert partial_report == report
    assert partial_authorizations == [False, True]


def test_main_routes_repeated_demo_dirs_through_one_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = (tmp_path / "first").resolve()
    second = (tmp_path / "second").resolve()
    for path in (first, second):
        path.mkdir()
        (path / "replay_manifest.json").write_text(
            json.dumps(
                {
                    "recording": {
                        "physical_replay_certification_mode": (
                            DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE
                        )
                    }
                }
            ),
            encoding="utf-8",
        )
    observed: list[Path] = []

    def certify(paths):
        observed.extend(Path(path).resolve() for path in paths)
        return {
            first: {"status": "passed", "reason": "matched"},
            second: {"status": "failed", "reason": "diverged"},
        }

    monkeypatch.setattr(
        "scripts.certify_physx_replay.certify_demo_session",
        certify,
    )

    returncode = main(
        [
            "--demo-dir",
            str(second),
            "--demo-dir",
            str(first),
        ]
    )

    assert observed == [second, first]
    assert returncode == 1
    output = capsys.readouterr().out
    assert f"[passed] {first}: matched" in output
    assert f"[failed] {second}: diverged" in output


def test_main_routes_one_fresh_process_demo_without_session_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    demo = (tmp_path / "fresh").resolve()
    demo.mkdir()
    (demo / "replay_manifest.json").write_text(
        json.dumps(
            {
                "recording": {
                    "physical_replay_certification_mode": (
                        FRESH_PROCESS_CERTIFICATION_MODE
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    observed: list[Path] = []

    def certify(path):
        observed.append(Path(path).resolve())
        return {"status": "passed", "reason": "fresh matched"}

    monkeypatch.setattr(
        "scripts.certify_physx_replay.certify_demo",
        certify,
    )
    monkeypatch.setattr(
        "scripts.certify_physx_replay.certify_demo_session",
        lambda _paths: pytest.fail("fresh replay entered session certification"),
    )

    assert main(["--demo-dir", str(demo)]) == 0
    assert observed == [demo]
    assert f"[passed] {demo}: fresh matched" in capsys.readouterr().out


def test_cli_rejects_multiple_fresh_process_demos_in_one_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demos = []
    for name in ("first", "second"):
        demo = (tmp_path / name).resolve()
        demo.mkdir()
        (demo / "replay_manifest.json").write_text(
            json.dumps(
                {
                    "recording": {
                        "physical_replay_certification_mode": (
                            FRESH_PROCESS_CERTIFICATION_MODE
                        )
                    }
                }
            ),
            encoding="utf-8",
        )
        demos.append(demo)

    monkeypatch.setattr(
        "scripts.certify_physx_replay.certify_demo",
        lambda _path: pytest.fail("ambiguous fresh batch was certified"),
    )

    with pytest.raises(
        CertificationError,
        match="exactly one episode",
    ):
        certify_cli_demo_dirs(demos)


def test_source_identity_only_marks_clean_commit_as_portable() -> None:
    clean = {
        "commit": "a" * 40,
        "dirty": False,
        "changed_files": [],
        "worktree_content_sha256": "0" * 64,
    }
    clean_result = verify_source_identity(clean, dict(clean))
    assert clean_result.source_identity == "clean_commit_match"
    assert clean_result.portable is True
    assert clean_result.source_snapshot_commit is None

    dirty = {
        **clean,
        "dirty": True,
        "changed_files": ["local.py"],
        "worktree_content_sha256": "1" * 64,
    }
    dirty_result = verify_source_identity(dirty, dict(dirty))
    assert dirty_result.source_identity == "dirty_content_match"
    assert dirty_result.portable is False
    assert dirty_result.source_snapshot_commit is None


def _git_for_source_identity(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _recorded_single_file_content_sha256(path: str, payload: bytes) -> str:
    identity = [
        {
            "path": path,
            "kind": "file",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    serialized = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _write_source_migration_repo(tmp_path: Path) -> tuple[Path, dict, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_for_source_identity(repo, "init")
    _git_for_source_identity(repo, "config", "user.email", "test@example.com")
    _git_for_source_identity(repo, "config", "user.name", "Test User")

    recorded_path = "collection_ui/server.py"
    source = repo / recorded_path
    source.parent.mkdir(parents=True)
    source.write_bytes(b"base\n")
    _git_for_source_identity(repo, "add", recorded_path)
    _git_for_source_identity(repo, "commit", "-m", "base")
    recorded_commit = _git_for_source_identity(repo, "rev-parse", "HEAD")

    recorded_payload = b"recorded dirty content\n"
    source.write_bytes(recorded_payload)
    recorded = {
        "commit": recorded_commit,
        "dirty": True,
        "changed_files": [recorded_path],
        "worktree_content_sha256": _recorded_single_file_content_sha256(
            recorded_path,
            recorded_payload,
        ),
    }
    _git_for_source_identity(repo, "add", recorded_path)
    _git_for_source_identity(repo, "commit", "-m", "exact snapshot")
    snapshot_commit = _git_for_source_identity(repo, "rev-parse", "HEAD")

    reviewer = repo / "scripts/certify_physx_replay.py"
    reviewer.parent.mkdir(parents=True)
    reviewer.write_text("# reviewer upgrade\n", encoding="utf-8")
    _git_for_source_identity(repo, "add", "scripts/certify_physx_replay.py")
    _git_for_source_identity(repo, "commit", "-m", "reviewer upgrade")
    return repo, recorded, snapshot_commit


def _clean_git_state(repo: Path) -> dict[str, object]:
    return {
        "commit": _git_for_source_identity(repo, "rev-parse", "HEAD"),
        "dirty": False,
        "changed_files": [],
        "worktree_content_sha256": hashlib.sha256(b"[]").hexdigest(),
    }


def test_dirty_recording_exact_snapshot_reviewer_upgrade_is_portable(
    tmp_path: Path,
) -> None:
    repo, recorded, snapshot_commit = _write_source_migration_repo(tmp_path)

    result = verify_source_identity(
        recorded,
        _clean_git_state(repo),
        repo_root=repo,
    )

    assert result.source_identity == ("dirty_recording_exact_snapshot_reviewer_upgrade")
    assert result.portable is True
    assert result.source_snapshot_commit == snapshot_commit
    assert result.reviewer_upgrade_paths == ("scripts/certify_physx_replay.py",)


def test_clean_recording_reviewer_upgrade_is_portable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_for_source_identity(repo, "init")
    _git_for_source_identity(repo, "config", "user.email", "test@example.com")
    _git_for_source_identity(repo, "config", "user.name", "Test User")
    reviewer = repo / "scripts/certify_physx_replay.py"
    reviewer.parent.mkdir(parents=True)
    reviewer.write_text("# recorded reviewer\n", encoding="utf-8")
    _git_for_source_identity(repo, "add", "scripts/certify_physx_replay.py")
    _git_for_source_identity(repo, "commit", "-m", "recorded source")
    recorded_commit = _git_for_source_identity(repo, "rev-parse", "HEAD")
    reviewer.write_text("# fixed reviewer\n", encoding="utf-8")
    _git_for_source_identity(repo, "add", "scripts/certify_physx_replay.py")
    _git_for_source_identity(repo, "commit", "-m", "reviewer fix")
    recorded = {
        "commit": recorded_commit,
        "dirty": False,
        "changed_files": [],
        "worktree_content_sha256": hashlib.sha256(b"[]").hexdigest(),
    }

    result = verify_source_identity(
        recorded,
        _clean_git_state(repo),
        repo_root=repo,
    )

    assert result.source_identity == "clean_recording_reviewer_upgrade"
    assert result.portable is True
    assert result.source_snapshot_commit == recorded_commit
    assert result.reviewer_upgrade_paths == ("scripts/certify_physx_replay.py",)


def test_dirty_recording_snapshot_rejects_content_mismatch(
    tmp_path: Path,
) -> None:
    repo, recorded, _snapshot_commit = _write_source_migration_repo(tmp_path)
    recorded["worktree_content_sha256"] = "f" * 64

    with pytest.raises(CertificationError, match="No ancestor commit exactly"):
        verify_source_identity(
            recorded,
            _clean_git_state(repo),
            repo_root=repo,
        )


def test_dirty_recording_snapshot_rejects_changed_path_mismatch(
    tmp_path: Path,
) -> None:
    repo, recorded, _snapshot_commit = _write_source_migration_repo(tmp_path)
    recorded["changed_files"] = ["other.py"]
    deleted_identity = [{"path": "other.py", "kind": "deleted", "sha256": None}]
    recorded["worktree_content_sha256"] = hashlib.sha256(
        json.dumps(
            deleted_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(CertificationError, match="No ancestor commit exactly"):
        verify_source_identity(
            recorded,
            _clean_git_state(repo),
            repo_root=repo,
        )


def test_dirty_recording_snapshot_rejects_extra_reviewer_path(
    tmp_path: Path,
) -> None:
    repo, recorded, _snapshot_commit = _write_source_migration_repo(tmp_path)
    (repo / "unrelated.py").write_text("unexpected\n", encoding="utf-8")
    _git_for_source_identity(repo, "add", "unrelated.py")
    _git_for_source_identity(repo, "commit", "-m", "unexpected source")

    with pytest.raises(CertificationError, match="outside.*allowlist"):
        verify_source_identity(
            recorded,
            _clean_git_state(repo),
            repo_root=repo,
        )


def test_dirty_recording_snapshot_requires_clean_worktree(tmp_path: Path) -> None:
    repo, recorded, _snapshot_commit = _write_source_migration_repo(tmp_path)
    (repo / "local.tmp").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(CertificationError, match="completely clean"):
        verify_source_identity(
            recorded,
            _clean_git_state(repo),
            repo_root=repo,
        )
