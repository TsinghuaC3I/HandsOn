from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from handson.data.physx_session_history import PhysxSessionHistoryJournal
from handson.data.physical_replay_certification import (
    CERTIFICATION_MODE,
    CONTACT_SIGNATURE_OVERLAY_ALGORITHM,
    CONTACT_SIGNATURE_OVERLAY_FORMAT,
    CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD,
    CONTACT_SIGNATURE_OVERLAY_VERSION,
    DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_FORMAT,
    DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_VERSION,
    DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE,
    FRESH_PROCESS_ACCEPTANCE_FORMAT,
    FRESH_PROCESS_ACCEPTANCE_VERSION,
    FRESH_PROCESS_CERTIFICATION_MODE,
    FRESH_SCENE_INITIALIZATION_MODE,
    INITIALIZATION_RESTORE_COUNTS_SOURCE,
    INPUT_IDENTITY_ALGORITHM,
    LEGACY_RAW_ACTION_INPUT_MODE,
    LEGACY_RAW_ACTION_TRANSITION_CONTRACT,
    ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE,
    PERSISTENT_RECORDER_LIFECYCLE,
    RECORDED_DRIVE_TARGET_INPUT_MODE,
    RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT,
    REPORT_FORMAT,
    REPORT_NAME,
    REPORT_VERSION,
    SESSION_PUBLICATION_COMMIT_DIRECTORY,
    SESSION_PUBLICATION_COMMIT_FORMAT,
    SESSION_PUBLICATION_COMMIT_VERSION,
    SESSION_REPLAY_EXECUTION_MODE,
    capture_certified_inputs,
    certification_report_blocker,
    local_physical_replay_report_blocker,
    certification_upload_blocker,
    deferred_physical_replay_acceptance_upload_blocker,
    fresh_process_acceptance_upload_blocker,
    physical_replay_upload_blocker,
)


_INPUTS = {
    "algorithm": INPUT_IDENTITY_ALGORITHM,
    "replay_manifest_sha256": "1" * 64,
    "recording_schema_sha256": "2" * 64,
    "replay_zarr_sha256": "3" * 64,
    "replay_zarr_file_count": 4,
    "replay_zarr_bytes": 5,
}
_SESSION_ID = "collection-session-test-001"
_SESSION_HISTORY_SHA256 = "a" * 64
_SESSION_HISTORY_EVENT_COUNT = 6
_SESSION_ACCEPTED_EVENT_ORDINAL = 5
_PHYSICAL_FIELD = "physx_task_rigid_body_state"
_PHYSICAL_ATOL = 1.0e-6
_PHYSICS_DT = 0.05
_CONTROL_DT = 0.1
_N_SUBSTEPS = 2
_CLEAN_GIT = {
    "commit": "1" * 40,
    "dirty": False,
    "status": "",
    "changed_files": [],
    "diff_binary_sha256": "0" * 64,
    "worktree_content_sha256": "4" * 64,
}
_DIRTY_GIT = {
    "commit": "1" * 40,
    "dirty": True,
    "status": " M collection_ui/server.py",
    "changed_files": ["collection_ui/server.py"],
    "diff_binary_sha256": "5" * 64,
    "worktree_content_sha256": "4" * 64,
}
_RUNTIME_IDENTITY = {
    "platform": "test-platform",
    "python": "3.11-test",
}


def _recording_schema() -> dict[str, object]:
    raw_contact_prefixes = (
        "physx_contact_",
        "physx_friction_anchor_",
        "physx_substep_contact_",
        "physx_substep_friction_anchor_",
    )

    def physical_replay(name: str) -> bool:
        signature = name.startswith("physx_contact_signature_") or (
            name.startswith("physx_substep_contact_signature_")
        )
        metric = name in {
            "physx_contact_signature_pair_metrics",
            "physx_substep_contact_signature_pair_metrics",
        }
        return (signature and not metric) or not name.startswith(
            raw_contact_prefixes
        )

    def replay_atol(name: str, dtype: str) -> float:
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

    specs: dict[str, tuple[list[int], str]] = {
        "physx_robot_joint_position": ([2, 1], "float64"),
        "physx_robot_joint_velocity": ([2, 1], "float64"),
        "physx_robot_joint_target": ([2, 1], "float64"),
        "physx_robot_joint_target_valid": ([2, 1], "uint8"),
        "physx_robot_joint_valid": ([2, 1], "uint8"),
        "physx_robot_root_state": ([2, 13], "float64"),
        "physx_robot_root_valid": ([2], "uint8"),
        "physx_robot_joint_effort": ([2, 1], "float64"),
        "physx_robot_joint_effort_valid": ([2, 1], "uint8"),
        "physx_tcp_state": ([2, 13], "float64"),
        "physx_tcp_valid": ([2], "uint8"),
        "physx_sim_time": ([1], "float64"),
        "physx_control_step": ([1], "float64"),
        "physx_command_tcp_pose": ([2, 7], "float64"),
        "physx_command_hand": ([2, 2], "float64"),
        "physx_command_hand_valid": ([2, 2], "uint8"),
        "physx_pose_commanded": ([2], "uint8"),
        "physx_pose_filter_state": ([2, 14], "float64"),
        "physx_pose_filter_valid": ([2], "uint8"),
        "physx_contact_report_valid": ([1], "uint8"),
        "physx_contact_capture_source": ([1], "uint8"),
        "physx_contact_header_count": ([1], "uint32"),
        "physx_contact_header_ids": ([4, 5], "uint64"),
        "physx_contact_header_meta": ([4, 7], "uint32"),
        "physx_contact_point_count": ([1], "uint32"),
        "physx_contact_point_data": ([8, 10], "float32"),
        "physx_contact_point_face_indices": ([8, 2], "uint32"),
        "physx_contact_point_material_ids": ([8, 2], "uint64"),
        "physx_friction_anchor_count": ([1], "uint32"),
        "physx_friction_anchor_data": ([8, 6], "float32"),
        "physx_contact_signature_valid": ([1], "uint8"),
        "physx_contact_signature_pair_count": ([1], "uint32"),
        "physx_contact_signature_pair_path_indices": ([4, 4], "uint32"),
        "physx_contact_signature_pair_counts": ([4, 2], "uint32"),
        "physx_contact_signature_pair_metrics": ([4, 3], "float64"),
        "physx_substep_count": ([1], "int64"),
        "physx_task_rigid_body_state": ([3, 13], "float64"),
        "physx_task_rigid_body_valid": ([3], "uint8"),
        "physx_substep_robot_joint_position": ([2, 2, 1], "float64"),
        "physx_substep_robot_joint_velocity": ([2, 2, 1], "float64"),
        "physx_substep_robot_joint_target": ([2, 2, 1], "float64"),
        "physx_substep_robot_joint_target_valid": ([2, 2, 1], "uint8"),
        "physx_substep_robot_joint_valid": ([2, 2, 1], "uint8"),
        "physx_substep_sim_time": ([2, 1], "float64"),
        "physx_substep_contact_signature_valid": ([2, 1], "uint8"),
        "physx_substep_contact_signature_pair_count": ([2, 1], "uint32"),
        "physx_substep_contact_signature_pair_path_indices": (
            [2, 4, 4],
            "uint32",
        ),
        "physx_substep_contact_signature_pair_counts": ([2, 4, 2], "uint32"),
        "physx_substep_contact_signature_pair_metrics": ([2, 4, 3], "float64"),
    }
    fields = {
        name: {
            "shape": shape,
            "dtype": dtype,
            "physical_replay": physical_replay(name),
            "replay_atol": replay_atol(name, dtype),
        }
        for name, (shape, dtype) in specs.items()
    }
    stored_fields: dict[str, object] = {}
    for name, semantics in fields.items():
        stored = {
            "dtype": semantics["dtype"],
            "shape_per_step": semantics["shape"],
            "semantics": dict(semantics),
        }
        stored_fields[name] = stored
        stored_fields[f"next_{name}"] = json.loads(json.dumps(stored))
    return {
        "environment_schema": {
            "name": "handson.physx_ground_truth",
            "version": 6,
            "physx": {
                "robot_sides": ["right", "left"],
                "robot_joint_names": {"right": ["right_joint"], "left": ["left_joint"]},
                "rigid_body_names": ["object", "tray", "table"],
                "articulation_names": [],
                "articulation_dof_names": {},
                "deformable_name": None,
                "requires_cuda_backend": True,
                "physics_config": {
                    "device": "cuda:0",
                    "physics_dt": _PHYSICS_DT,
                    "control_dt": _CONTROL_DT,
                    "n_substeps": _N_SUBSTEPS,
                    "solver": {"enable_enhanced_determinism": True},
                },
                "physics_dt": _PHYSICS_DT,
                "control_dt": _CONTROL_DT,
                "n_substeps": _N_SUBSTEPS,
                "enhanced_determinism": True,
                "contact_report": {
                    "canonical_metric_replay_policy": (
                        "diagnostic_only_solver_output"
                    ),
                    "path_table": [
                        {"id": 1, "path": "/World/object"},
                        {"id": 2, "path": "/World/table"},
                    ],
                    "report_enabled_task_body_paths": ["/World/object"],
                    "tensor_sensor_paths": [],
                    "tensor_filter_paths": [],
                },
            },
            "fields": fields,
        },
        "fields": stored_fields,
    }


def _expected_field_atols() -> dict[str, float]:
    fields = _recording_schema()["environment_schema"]["fields"]
    assert isinstance(fields, dict)
    return {
        name: float(entry["replay_atol"])
        for name, entry in fields.items()
        if isinstance(entry, dict) and entry["physical_replay"] is True
    }


def _flattened_physics_config() -> dict[str, object]:
    return {
        "control_dt": _CONTROL_DT,
        "device": "cuda:0",
        "n_substeps": _N_SUBSTEPS,
        "physics_dt": _PHYSICS_DT,
        "solver.enable_enhanced_determinism": True,
    }


def _recording_contract() -> dict[str, object]:
    return {
        "physical_replay_certification_required": True,
        "physical_replay_certification_runs_required": 3,
        "physical_replay_certification_mode": FRESH_PROCESS_CERTIFICATION_MODE,
        "physical_replay_goal": CERTIFICATION_MODE,
        "physical_transition_contract": RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT,
        "trajectory_input_mode": RECORDED_DRIVE_TARGET_INPUT_MODE,
        "fresh_recorder_process_per_episode_required": True,
        "fresh_recorder_process_per_attempt_required": True,
        "recorder_lifecycle": "fresh_process_per_attempt",
        "recorder_pid": 101,
        "data_fps": 1.0 / _CONTROL_DT,
        "source_process_reuse_enabled": False,
        "source_scene_rebuild_per_reset": False,
        "source_scene_generation": 0,
        "fresh_scene_replay_required": True,
        "fresh_certifier_process_per_run_required": True,
        "publish_only_after_fresh_certification": True,
        "stable_recording_origin": _natural_settle_origin_history(),
    }


def _deferred_recording_contract() -> dict[str, object]:
    recording = _recording_contract()
    recording.update(
        {
            "physical_replay_certification_mode": (
                DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE
            ),
            "fresh_recorder_process_per_episode_required": False,
            "fresh_recorder_process_per_attempt_required": False,
            "recorder_lifecycle": PERSISTENT_RECORDER_LIFECYCLE,
            "recorder_pid": 101,
            "source_process_reuse_enabled": True,
            "source_scene_rebuild_per_reset": False,
            "source_scene_generation": 0,
            "fresh_scene_replay_required": True,
            "fresh_certifier_process_per_run_required": True,
            "publish_only_after_fresh_certification": True,
            "collection_run_seed": 17,
            "episode_reset_seed": 19,
            "collection_seed_source": "explicit_cli_seed",
            "episode_reset_ordinal": 2,
            "episode_seed_derivation": (
                "collection_run_seed_plus_reset_attempt_index_mod_2pow31"
            ),
            "stable_recording_origin": _natural_settle_origin_history(),
            "episode_index": 1,
            "physical_replay_certification_scope": (
                ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE
            ),
            "physical_replay_session_history_pending": False,
            "physical_replay_session_history_id": _SESSION_ID,
            "physical_replay_session_history_root_relative_path": (
                f".physical_replay_pending/.physx_session_history/{_SESSION_ID}"
            ),
            "physical_replay_session_history_sha256": (_SESSION_HISTORY_SHA256),
            "physical_replay_session_history_event_count": (
                _SESSION_HISTORY_EVENT_COUNT
            ),
            "physical_replay_session_history_accepted_event_ordinal": (
                _SESSION_ACCEPTED_EVENT_ORDINAL
            ),
            "physical_replay_session_history_certified_through_event_ordinal": (
                _SESSION_ACCEPTED_EVENT_ORDINAL
            ),
        }
    )
    return recording


def _all_step_origin_history() -> dict[str, object]:
    coverage = {
        "rigid_body": 3,
        "articulation_root": 0,
        "robot_root": 2,
        "tcp": 2,
        "articulation_dof": 0,
        "robot_joint": 54,
        "hand_command": 40,
        "deformable_node": 0,
    }
    active_layout = {
        "format": "handson.active_physics_layout",
        "version": 1,
        "rigid_body": {
            "names": ["object", "tray", "table"],
            "valid": [1, 1, 1],
        },
        "articulation_root": {"names": [], "valid": []},
        "articulation_dof": {"names": [], "valid": []},
        "deformable_node": {"names": [], "valid": []},
    }
    gate_contact = {
        "valid": True,
        "capture_source": 2,
        "contact_point_count": 4,
        "maximum_penetration_depth_m": 0.000001,
        "deepest_contact_point_index": 2,
        "error": None,
    }
    return {
        "stabilization_velocity_clamp": True,
        "initialization_per_physics_step_velocity_writes": True,
        "physics_dt": _PHYSICS_DT,
        "timeout_seconds": 20.0,
        "physics_steps": 180,
        "initial_settle_steps": 120,
        "stabilization_velocity_clamp_scope": "all_initialization_steps",
        "stabilization_velocity_clamp_steps": 180,
        "stabilization_velocity_clamp_write": {
            "rigid_bodies": 3,
            "articulation_roots": 0,
            "articulation_dofs": 0,
            "deformable_nodes": 0,
        },
        "stability_metric": "velocity",
        "task_name": "t25_place_cup_on_plate",
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
        "reset_pose_drift_checked": True,
        "reset_translation_drift_limit": 0.005,
        "reset_rotation_drift_limit": 0.05235987755982989,
        "reset_max_translation_drift_m": 0.0004,
        "reset_max_rotation_drift_rad": 0.001,
        "reset_coordinate_drift_limit": 0.005,
        "reset_max_coordinate_drift": 0.0003,
        "reset_state_coverage": coverage,
        "active_physics_layout": active_layout,
        "gate_active_physics_layout": json.loads(json.dumps(active_layout)),
        "final_active_physics_layout": json.loads(json.dumps(active_layout)),
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
        "gate_contact_summary": gate_contact,
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
        "final_reset_state_coverage": json.loads(json.dumps(coverage)),
        "gate_final_translation_delta_limit": 0.005,
        "gate_final_rotation_delta_limit": 0.05235987755982989,
        "gate_final_max_task_translation_delta_m": 0.0,
        "gate_final_max_task_rotation_delta_rad": 0.0,
        "final_contact_penetration_limit": 0.0005,
        "final_contact_summary": json.loads(json.dumps(gate_contact)),
        "final_boundary_physics_gate_passed": True,
        "post_settle_velocity_zeroing": False,
        "post_settle_sleeping": False,
        "velocity_zeroing_passes": 2,
        "first_state_write": {
            "robot_joints": 54,
            "rigid_bodies": 3,
            "articulation_roots": 0,
            "articulation_dofs": 0,
            "deformable_nodes": 0,
        },
        "post_settle_state_write": None,
        "post_settle_sleep": None,
        "boundary_state_write_mode": "velocity_only",
        "contact_warm_start_preserved": True,
        "final_boundary_state_write": {
            "robot_joints": 54,
            "rigid_bodies": 3,
            "articulation_roots": 0,
            "articulation_dofs": 0,
            "deformable_nodes": 0,
        },
        "final_boundary_sleep": None,
        "final_boundary_restore_sync": True,
        "final_boundary_physics_steps_after_write": 0,
        "default_joint_state_restores": 1,
        "robot_hold_target_source": "default_joint_state",
        "post_settle_robot_hold_target_source": "default_joint_state",
        "final_default_joint_state_restore": False,
        "final_robot_hold_target_source": "accepted_settled_joint_state",
        "hold_target_sync": {"robot_joints": 54, "hand_commands": 44},
        "final_hold_target_sync": {
            "robot_joints": 54,
            "hand_commands": 44,
        },
        "final_boundary_task_and_joint_velocity_exact_zero": True,
        "state_writes": True,
        "trajectory_per_frame_state_writes": False,
        "per_frame_state_writes": False,
    }


def _natural_settle_origin_history() -> dict[str, object]:
    history = _all_step_origin_history()
    history.update(
        {
            "timeout_seconds": 39.0,
            "stabilization_velocity_clamp": False,
            "initialization_per_physics_step_velocity_writes": False,
            "stabilization_velocity_clamp_scope": "none",
            "stabilization_velocity_clamp_steps": 0,
            "stabilization_velocity_clamp_write": None,
            "post_settle_velocity_zeroing": True,
            "post_settle_sleeping": False,
            "stability_sample_steps": 1,
            "stability_samples": 60,
            "velocity_zeroing_passes": 3,
            "hold_target_sync": {
                "robot_joints": 54,
                "hand_commands": 44,
                "passes": 182,
            },
            "post_settle_state_write": {
                "robot_joints": 54,
                "rigid_bodies": 3,
                "articulation_roots": 0,
                "articulation_dofs": 0,
                "deformable_nodes": 0,
            },
            "post_settle_sleep": None,
            "final_boundary_sleep": None,
            "boundary_state_write_mode": "velocity_only",
            "contact_warm_start_preserved": True,
        }
    )
    return history


def _canonical_report_history(
    recorded: dict[str, object],
) -> dict[str, object]:
    """Mirror the canonical initialization evidence emitted by certifier."""

    canonical_fields = _all_step_origin_history().keys()
    return {
        field: json.loads(json.dumps(recorded[field])) for field in canonical_fields
    }


def _write_action_zarr_metadata(demo: Path, *, steps: int = 2) -> None:
    action = demo / "replay.zarr" / "data" / "action"
    action.mkdir(parents=True, exist_ok=True)
    (action / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [steps, 1],
                "chunks": [max(1, steps), 1],
                "dtype": "<f8",
                "compressor": None,
                "fill_value": 0.0,
                "order": "C",
                "filters": None,
            }
        ),
        encoding="utf-8",
    )


def _write_manifest(
    tmp_path: Path,
    *,
    recording: dict[str, object] | None = None,
    task_name: str = "t25_place_cup_on_plate",
    recorded_git: dict[str, object] | None = None,
) -> Path:
    demo = tmp_path / "demo"
    demo.mkdir()
    recording_contract = recording or _recording_contract()
    source_git = recorded_git or _CLEAN_GIT
    (demo / "replay_manifest.json").write_text(
        json.dumps(
            {
                "task": {"name": task_name},
                "recording": recording_contract,
                "software": {
                    "git": source_git,
                    "runtime_replay_identity": _RUNTIME_IDENTITY,
                },
            }
        ),
        encoding="utf-8",
    )
    (demo / "recording_schema.json").write_text(
        json.dumps(_recording_schema()),
        encoding="utf-8",
    )
    _write_action_zarr_metadata(demo)
    if (
        recording_contract.get("physical_replay_certification_mode")
        == DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE
    ):
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
        metadata = {
            "collection_session_id": recording_contract.get(
                "physical_replay_session_history_id"
            ),
            **{field: recording_contract.get(field) for field in binding_fields},
        }
        (demo / "collection_meta.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        journal_path = (
            tmp_path
            / ".physical_replay_pending"
            / ".physx_session_history"
            / _SESSION_ID
        )
        journal = PhysxSessionHistoryJournal.create(
            journal_path,
            session_id=_SESSION_ID,
            task_name=task_name,
            collection_run_seed=17,
        )
        stable_origin = recording_contract.get("stable_recording_origin")
        if not isinstance(stable_origin, dict):
            stable_origin = _all_step_origin_history()
        for ordinal in (0, 1):
            journal.record_reset_attempt(
                17 + ordinal,
                ordinal,
                "accepted",
                {"timeout_seconds": 5.0},
                stable_origin=stable_origin,
                start_sim_time=float(ordinal),
                end_sim_time=float(ordinal + 1),
            )
            journal.record_trajectory(
                {},
                reason="discarded_test_segment",
                reset_ordinal=ordinal,
                start_sim_time=float(ordinal + 1),
                end_sim_time=float(ordinal + 1),
            )
        journal.record_reset_attempt(
            19,
            2,
            "accepted",
            {"timeout_seconds": 5.0},
            stable_origin=stable_origin,
            start_sim_time=2.0,
            end_sim_time=3.0,
        )
        journal.record_accepted_demo(
            {
                "action": [[0.0], [0.0]],
                "physx_sim_time": [3.0, 3.5],
                "next_physx_sim_time": [3.5, 4.0],
            },
            demo_dir=demo,
            reset_ordinal=2,
        )
        finalization = journal.finalize([demo])
        accepted = finalization["accepted_demos"][0]
        binding_update = {
            "physical_replay_session_history_sha256": finalization["history_identity"][
                "sha256"
            ],
            "physical_replay_session_history_event_count": finalization["event_count"],
            "physical_replay_session_history_accepted_event_ordinal": accepted[
                "event_ordinal"
            ],
            "physical_replay_session_history_certified_through_event_ordinal": (
                finalization["certified_through_event_ordinal"]
            ),
        }
        recording_contract.update(binding_update)
        metadata.update(binding_update)
        (demo / "replay_manifest.json").write_text(
            json.dumps(
                {
                    "task": {"name": task_name},
                    "recording": recording_contract,
                    "software": {
                        "git": source_git,
                        "runtime_replay_identity": _RUNTIME_IDENTITY,
                    },
                }
            ),
            encoding="utf-8",
        )
        (demo / "collection_meta.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
    return demo


def _passed_report() -> dict[str, object]:
    expected_atols = _expected_field_atols()
    physics_values = _flattened_physics_config()
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "status": "passed",
        "portable": True,
        "source_identity": "clean_commit_match",
        "recorded_git": dict(_CLEAN_GIT),
        "current_git": dict(_CLEAN_GIT),
        "runtime_replay_identity": dict(_RUNTIME_IDENTITY),
        "task_success_replay": True,
        "trace_match": True,
        "transition_outcomes_match": True,
        "final_succeed": True,
        "first_divergence": None,
        "first_outcome_mismatch": None,
        "max_errors": {name: 0.0 for name in expected_atols},
        "field_atols": expected_atols,
        "certification_mode": CERTIFICATION_MODE,
        "initialization_mode": FRESH_SCENE_INITIALIZATION_MODE,
        "physical_transition_contract": RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT,
        "trajectory_input_mode": RECORDED_DRIVE_TARGET_INPUT_MODE,
        "source_recorder_lifecycle": PERSISTENT_RECORDER_LIFECYCLE,
        "source_process_reuse_enabled": True,
        "fresh_source_process_per_attempt": False,
        "fresh_scene_replay_required": True,
        "collection_seed_source": "explicit_cli_seed",
        "episode_reset_ordinal": 2,
        "episode_seed_derivation": (
            "collection_run_seed_plus_reset_attempt_index_mod_2pow31"
        ),
        "source_recorder_pid": 101,
        "initialization_restore_counts": {
            "robot_joints": 54,
            "rigid_bodies": 3,
            "articulation_roots": 0,
            "articulation_dofs": 0,
            "deformable_nodes": 0,
        },
        "initialization_restore_counts_source": (INITIALIZATION_RESTORE_COUNTS_SOURCE),
        "initialization_readback_match": True,
        "initialization_history_match": True,
        "replayed_stable_origin_history": _natural_settle_origin_history(),
        "post_stabilization_snapshot_restore_performed": False,
        "physics_steps_between_initialization_readback_and_action_0": 0,
        "state_writes_performed": True,
        "initialization_state_writes_performed": True,
        "trajectory_state_writes_performed": False,
        "state_writes_after_action_start": False,
        "raw_commands_installed": True,
        "ik_recomputed": False,
        "recorded_drive_targets_applied": True,
        "recorded_physics_configuration_match": {
            "matched": True,
            "checked_field_count": len(physics_values),
            "checked_fields": sorted(physics_values),
            "recorded_values": dict(physics_values),
            "runtime_values": dict(physics_values),
            "unrecorded_static_dynamics_fingerprint": True,
        },
        "raw_environment_step_called": False,
        "recorded_drive_target_substeps_expected": 8,
        "recorded_drive_target_substeps_executed": 8,
        "recorded_drive_target_writes_expected": 108,
        "recorded_drive_target_writes_executed": 108,
        "steps_expected": 2,
        "steps_executed": 2,
        "certified_inputs": dict(_INPUTS),
    }


def _materialize_certified_inputs(demo: Path) -> dict[str, object]:
    (demo / "recording_schema.json").write_text(
        json.dumps(_recording_schema()),
        encoding="utf-8",
    )
    _write_action_zarr_metadata(demo)
    return capture_certified_inputs(demo)


def _write_deferred_acceptance(
    demo: Path,
    *,
    inputs: dict[str, object] | None = None,
    pids: tuple[int, int, int] = (201, 202, 203),
    recorder_pid: int = 101,
    write_publication_commit: bool = True,
    contact_signature_overlays: tuple[
        dict[str, object] | None,
        dict[str, object] | None,
        dict[str, object] | None,
    ]
    | None = None,
) -> dict[str, object]:
    certified_inputs = dict(inputs or capture_certified_inputs(demo))
    manifest = json.loads((demo / "replay_manifest.json").read_text(encoding="utf-8"))
    recording = manifest["recording"]
    episode_reset_ordinal = recording["episode_reset_ordinal"]
    accepted_event_ordinal = recording[
        "physical_replay_session_history_accepted_event_ordinal"
    ]
    accepted_reset_event_ordinal = accepted_event_ordinal - 1
    event_replay = {
        "status": "replayed_verified_session_prefix",
        "events_replayed": accepted_reset_event_ordinal,
        "first_event_ordinal": (0 if accepted_reset_event_ordinal > 0 else None),
        "last_event_ordinal": (
            accepted_reset_event_ordinal - 1
            if accepted_reset_event_ordinal > 0
            else None
        ),
        "target_accepted_reset_event_ordinal": (accepted_reset_event_ordinal),
        "target_accepted_trajectory_event_ordinal": accepted_event_ordinal,
        "reset_attempts_replayed": 2,
        "rejected_reset_attempts_replayed": 0,
        "discarded_trajectories_replayed": 2,
        "discarded_actions_replayed": 0,
        "discarded_drive_substeps_replayed": 0,
        "previous_episode_reset_ordinal": None,
        "next_episode_reset_ordinal": episode_reset_ordinal,
    }
    reports_dir = demo / "physical_replay_certification_runs"
    reports_dir.mkdir()
    runs: list[dict[str, object]] = []
    for index, pid in enumerate(pids, start=1):
        report = (
            _snapshot_reviewer_report()
            if manifest["software"]["git"].get("dirty") is True
            else _passed_report()
        )
        if contact_signature_overlays is not None:
            overlay = contact_signature_overlays[index - 1]
            if overlay is not None:
                report["source_reviewer_upgrade_paths"] = [
                    "handson/handson/sim/isaaclab/physx_ground_truth.py",
                    "scripts/certify_physx_replay.py",
                ]
                report[CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD] = overlay
        report["certified_inputs"] = dict(certified_inputs)
        report["replayed_stable_origin_history"] = _canonical_report_history(
            recording["stable_recording_origin"]
        )
        report["session_replay"] = {
            "mode": SESSION_REPLAY_EXECUTION_MODE,
            "status": "passed",
            "shared_environment": True,
            "episode_index": 0,
            "episode_count": 1,
            "ordered_episode_reset_ordinals": [episode_reset_ordinal],
            "session_history_id": recording["physical_replay_session_history_id"],
            "session_history_sha256": recording[
                "physical_replay_session_history_sha256"
            ],
            "session_history_event_count": recording[
                "physical_replay_session_history_event_count"
            ],
            "accepted_reset_event_ordinal": accepted_reset_event_ordinal,
            "accepted_event_ordinal": accepted_event_ordinal,
            "replayed_through_event_ordinal": accepted_event_ordinal,
            "history_certified_through_event_ordinal": recording[
                "physical_replay_session_history_certified_through_event_ordinal"
            ],
            "event_replay": dict(event_replay),
        }
        report_bytes = json.dumps(report, sort_keys=True).encode("utf-8")
        report_path = reports_dir / f"run_{index}.json"
        report_path.write_bytes(report_bytes)
        runs.append(
            {
                "run": index,
                "pid": pid,
                "status": "passed",
                "returncode": 0,
                "started_at": f"2026-08-02T01:0{index}:00+00:00",
                "finished_at": f"2026-08-02T01:0{index}:30+00:00",
                "report": (f"physical_replay_certification_runs/run_{index}.json"),
                "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "certified_inputs": dict(certified_inputs),
                "session_episode_count": 1,
            }
        )
    commit_id = "publication-commit-test-001"
    commit_relative_path = f"{SESSION_PUBLICATION_COMMIT_DIRECTORY}/{commit_id}.json"
    acceptance: dict[str, object] = {
        "format": DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_FORMAT,
        "version": DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_VERSION,
        "mode": DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE,
        "status": "passed",
        "certification_runs_required": 3,
        "certification_runs_passed": 3,
        "certifier_pids": list(pids),
        "certification_runs": runs,
        "certified_inputs": dict(certified_inputs),
        "published_after_post_collection_certification": True,
        "persistent_source_process_for_collection_batch": True,
        "fresh_source_process_per_episode": False,
        "fresh_certifier_process_per_run": True,
        "recorder_lifecycle": PERSISTENT_RECORDER_LIFECYCLE,
        "fresh_scene_replay_required": True,
        "fresh_certifier_process_per_run_required": True,
        "source_recorder_pid": recorder_pid,
        "recorder_pid": recorder_pid,
        "certification_scope": (ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE),
        "collection_session_id": recording["physical_replay_session_history_id"],
        "collection_run_seed": recording["collection_run_seed"],
        "session_episode_count": 1,
        "session_episode_index": 1,
        "session_episode_reset_ordinals": [episode_reset_ordinal],
        "session_demo_names": [demo.name],
        "fresh_certifier_process_per_session_run": True,
        "session_publication_all_targets_prevalidated": True,
        "session_publication_rollback_on_catchable_error": True,
        "session_publication_commit_id": commit_id,
        "session_publication_commit_root_relative_path": commit_relative_path,
    }
    acceptance_path = demo / "fresh_process_acceptance.json"
    acceptance_path.write_text(
        json.dumps(acceptance),
        encoding="utf-8",
    )
    if write_publication_commit:
        commit_path = demo.parent / commit_relative_path
        commit_path.parent.mkdir()
        marker = {
            "format": SESSION_PUBLICATION_COMMIT_FORMAT,
            "version": SESSION_PUBLICATION_COMMIT_VERSION,
            "status": "committed",
            "session_publication_commit_id": commit_id,
            "session_publication_commit_root_relative_path": (commit_relative_path),
            "collection_session_id": recording["physical_replay_session_history_id"],
            "session_demo_names": [demo.name],
            "session_episode_reset_ordinals": [episode_reset_ordinal],
            "session_history_id": recording["physical_replay_session_history_id"],
            "session_history_sha256": recording[
                "physical_replay_session_history_sha256"
            ],
            "session_history_event_count": recording[
                "physical_replay_session_history_event_count"
            ],
            "session_history_certified_through_event_ordinal": recording[
                "physical_replay_session_history_certified_through_event_ordinal"
            ],
            "members": [
                {
                    "demo_name": demo.name,
                    "episode_reset_ordinal": episode_reset_ordinal,
                    "accepted_event_ordinal": accepted_event_ordinal,
                    "acceptance_sha256": hashlib.sha256(
                        acceptance_path.read_bytes()
                    ).hexdigest(),
                    "certified_inputs": dict(certified_inputs),
                }
            ],
            "committed_at": "2026-08-02T01:04:00+00:00",
        }
        commit_path.write_text(json.dumps(marker), encoding="utf-8")
    return acceptance


def test_v4_recorded_drive_target_report_passes_bound_contract(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(tmp_path)

    assert (
        certification_report_blocker(
            demo,
            _passed_report(),
            actual_inputs=_INPUTS,
        )
        is None
    )


@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        "cuda",
        "CUDA:0",
        " cuda:0",
        "cuda:0 ",
        "cuda:00",
        "cuda:01",
        "cuda:-1",
        "cuda:²",
    ),
)
def test_recording_schema_rejects_noncanonical_cuda_device(
    tmp_path: Path,
    device: str,
) -> None:
    demo = _write_manifest(tmp_path)
    schema_path = demo / "recording_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["environment_schema"]["physx"]["physics_config"]["device"] = device
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    blocker = certification_report_blocker(
        demo,
        _passed_report(),
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_invalid"
    assert "cuda:N" in blocker["detail"]


def test_local_replay_can_pass_without_weakening_portable_upload_gate(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(tmp_path, recorded_git=dict(_DIRTY_GIT))
    report = _passed_report()
    report["portable"] = False
    report["source_identity"] = "dirty_content_match"
    report["recorded_git"] = dict(_DIRTY_GIT)
    report["current_git"] = dict(_DIRTY_GIT)

    local_blocker = local_physical_replay_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )
    upload_blocker = certification_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )

    assert local_blocker is None
    assert upload_blocker is not None
    assert upload_blocker["code"] == "physical_replay_certification_invalid"
    assert "portable 必须为 true" in upload_blocker["detail"]


def _snapshot_reviewer_report() -> dict[str, object]:
    report = _passed_report()
    report.update(
        {
            "source_identity": ("dirty_recording_exact_snapshot_reviewer_upgrade"),
            "source_snapshot_commit": "2" * 40,
            "source_reviewer_upgrade_paths": ["scripts/certify_physx_replay.py"],
            "recorded_git": dict(_DIRTY_GIT),
            "current_git": {
                **_CLEAN_GIT,
                "commit": "3" * 40,
            },
        }
    )
    return report


def _contact_signature_overlay_proof(seed: str = "a") -> dict[str, object]:
    segment = {
        "segment_id": "accepted_demo:demo",
        "signature_frames_checked": 20,
        "signature_frames_reconstructed": 2,
        "control_frames_reconstructed": 1,
        "substep_frames_reconstructed": 1,
        "exact_zero_padding_rows_removed": 2,
        "coordinates_sha256": seed * 64,
        "raw_input_sha256": "b" * 64,
        "canonical_output_sha256": "c" * 64,
    }
    payload = {
        "format": CONTACT_SIGNATURE_OVERLAY_FORMAT,
        "version": CONTACT_SIGNATURE_OVERLAY_VERSION,
        "algorithm": CONTACT_SIGNATURE_OVERLAY_ALGORITHM,
        "source_open_mode": "read_only",
        "source_files_mutated": False,
        "segments": [segment],
        "segment_count": 1,
        "signature_frames_reconstructed": 2,
        "exact_zero_padding_rows_removed": 2,
    }
    return {
        **payload,
        "digest_sha256": hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _overlay_snapshot_reviewer_report(seed: str = "a") -> dict[str, object]:
    report = _snapshot_reviewer_report()
    report["source_reviewer_upgrade_paths"] = [
        "handson/handson/sim/isaaclab/physx_ground_truth.py",
        "scripts/certify_physx_replay.py",
    ]
    report[CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD] = _contact_signature_overlay_proof(
        seed
    )
    return report


def test_snapshot_reviewer_identity_passes_portable_report_gate(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(tmp_path, recorded_git=dict(_DIRTY_GIT))

    assert (
        certification_report_blocker(
            demo,
            _snapshot_reviewer_report(),
            actual_inputs=_INPUTS,
        )
        is None
    )


def test_contact_signature_overlay_passes_only_with_bound_reviewer_evidence(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(tmp_path, recorded_git=dict(_DIRTY_GIT))

    assert (
        certification_report_blocker(
            demo,
            _overlay_snapshot_reviewer_report(),
            actual_inputs=_INPUTS,
        )
        is None
    )


@pytest.mark.parametrize("failure", ["digest", "reviewer_path", "mutation"])
def test_contact_signature_overlay_gate_fails_closed(
    tmp_path: Path,
    failure: str,
) -> None:
    demo = _write_manifest(tmp_path, recorded_git=dict(_DIRTY_GIT))
    report = _overlay_snapshot_reviewer_report()
    overlay = report[CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD]
    assert isinstance(overlay, dict)
    if failure == "digest":
        overlay["digest_sha256"] = "f" * 64
    elif failure == "reviewer_path":
        report["source_reviewer_upgrade_paths"] = ["scripts/certify_physx_replay.py"]
    else:
        overlay["source_files_mutated"] = True

    blocker = certification_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_invalid"
    assert "contact signature overlay" in blocker["detail"]


def test_clean_reviewer_identity_passes_portable_report_gate(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(tmp_path)
    report = _passed_report()
    report.update(
        {
            "source_identity": "clean_recording_reviewer_upgrade",
            "source_snapshot_commit": "1" * 40,
            "source_reviewer_upgrade_paths": ["scripts/certify_physx_replay.py"],
            "recorded_git": dict(_CLEAN_GIT),
            "current_git": {
                **_CLEAN_GIT,
                "commit": "3" * 40,
            },
        }
    )

    assert (
        certification_report_blocker(
            demo,
            report,
            actual_inputs=_INPUTS,
        )
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_snapshot_commit", None),
        ("source_snapshot_commit", "2" * 39),
        ("source_reviewer_upgrade_paths", []),
        ("source_reviewer_upgrade_paths", ["unrelated.py"]),
        ("recorded_git", {"commit": "1" * 40, "dirty": False}),
        ("current_git", {"commit": "3" * 40, "dirty": True}),
    ],
)
def test_snapshot_reviewer_identity_rejects_invalid_evidence(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    demo = _write_manifest(tmp_path, recorded_git=dict(_DIRTY_GIT))
    report = _snapshot_reviewer_report()
    report[field] = value

    blocker = certification_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_invalid"


@pytest.mark.parametrize(
    ("portable", "source_identity"),
    [
        (None, "dirty_content_match"),
        ("false", "dirty_content_match"),
        (False, "clean_commit_match"),
        (False, "unexpected_identity"),
        (True, "dirty_content_match"),
    ],
)
def test_local_replay_requires_consistent_source_portability_identity(
    tmp_path: Path,
    portable: object,
    source_identity: str,
) -> None:
    demo = _write_manifest(tmp_path)
    report = _passed_report()
    report["portable"] = portable
    report["source_identity"] = source_identity

    blocker = local_physical_replay_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_invalid"


def test_all_step_origin_report_binds_replayed_initialization_history(
    tmp_path: Path,
) -> None:
    recording = _recording_contract()
    recording["stable_recording_origin"] = _all_step_origin_history()
    demo = _write_manifest(tmp_path, recording=recording)
    report = _passed_report()
    report["initialization_history_match"] = True
    report["replayed_stable_origin_history"] = _all_step_origin_history()

    blocker = certification_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_mode_invalid"
    assert "all_initialization_steps" in blocker["detail"]


def test_deferred_natural_settle_then_zero_contract_is_certifiable(
    tmp_path: Path,
) -> None:
    history = _natural_settle_origin_history()
    recording = _deferred_recording_contract()
    recording["stable_recording_origin"] = history
    demo = _write_manifest(tmp_path, recording=recording)
    report = _passed_report()
    replayed_history = json.loads(json.dumps(history))
    replayed_history["physics_steps"] = 204
    replayed_history["stability_samples"] = 84
    replayed_history["hold_target_sync"]["passes"] = 206
    replayed_history["reset_drift_sample_count"] = 205
    replayed_history["reset_drift_expected_sample_count"] = 205
    replayed_history["initialization_contact_summary"]["sample_count"] = 205
    replayed_history["initialization_contact_summary"]["expected_sample_count"] = 205
    report["replayed_stable_origin_history"] = replayed_history

    assert (
        certification_report_blocker(
            demo,
            report,
            actual_inputs=_INPUTS,
        )
        is None
    )


@pytest.mark.parametrize(
    "field",
    ["active_physics_layout", "initialization_contact_summary"],
)
def test_fresh_manifest_requires_complete_geometry_history(
    tmp_path: Path,
    field: str,
) -> None:
    recording = _recording_contract()
    stable_origin = recording["stable_recording_origin"]
    assert isinstance(stable_origin, dict)
    stable_origin.pop(field)
    demo = _write_manifest(tmp_path, recording=recording)

    blocker = certification_report_blocker(
        demo,
        _passed_report(),
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_mode_invalid"
    assert field in blocker["detail"]


def test_fresh_report_requires_exact_active_physics_layout(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(tmp_path)
    report = _passed_report()
    replayed = report["replayed_stable_origin_history"]
    assert isinstance(replayed, dict)
    replayed.pop("active_physics_layout")

    blocker = certification_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert "replayed_stable_origin_history.active_physics_layout" in blocker["detail"]


def test_robot_only_reset_coverage_is_not_certifiable(
    tmp_path: Path,
) -> None:
    recording = _recording_contract()
    stable_origin = recording["stable_recording_origin"]
    assert isinstance(stable_origin, dict)
    coverage = stable_origin["reset_state_coverage"]
    layout = stable_origin["active_physics_layout"]
    assert isinstance(coverage, dict)
    assert isinstance(layout, dict)
    coverage["rigid_body"] = 0
    layout["rigid_body"] = {"names": [], "valid": []}
    demo = _write_manifest(tmp_path, recording=recording)

    blocker = certification_report_blocker(
        demo,
        _passed_report(),
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert "covers no task physics state" in blocker["detail"]


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("authored_reset_snapshot_serial", lambda value: int(value) + 1),
        (
            "gate_contact_summary",
            lambda value: {**value, "contact_point_count": 5},
        ),
        (
            "initialization_contact_summary",
            lambda value: {
                **value,
                "maximum_contact_point_count": 5,
            },
        ),
    ],
)
def test_natural_settle_does_not_ignore_physics_identity_evidence(
    tmp_path: Path,
    field: str,
    mutate: object,
) -> None:
    history = _natural_settle_origin_history()
    recording = _deferred_recording_contract()
    recording["stable_recording_origin"] = history
    demo = _write_manifest(tmp_path, recording=recording)
    report = _passed_report()
    replayed = json.loads(json.dumps(history))
    replayed[field] = mutate(replayed[field])
    report["replayed_stable_origin_history"] = replayed

    blocker = certification_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert f"replayed_stable_origin_history.{field}" in blocker["detail"]


@pytest.mark.parametrize(
    ("mutation", "expected_detail"),
    [
        ("final", "final_succeed"),
        ("divergence", "first_divergence"),
        ("outcome", "first_outcome_mismatch"),
        ("max_error", f"max_errors.{_PHYSICAL_FIELD}"),
        ("field_atol", f"field_atols.{_PHYSICAL_FIELD}"),
        ("runtime", "runtime_replay_identity"),
        ("recorded_git", "report.recorded_git"),
        ("physics_config", "recorded physics configuration"),
    ],
)
def test_upload_report_numeric_and_identity_evidence_fails_closed(
    tmp_path: Path,
    mutation: str,
    expected_detail: str,
) -> None:
    demo = _write_manifest(tmp_path)
    report = _passed_report()
    if mutation == "final":
        report["final_succeed"] = False
    elif mutation == "divergence":
        report["first_divergence"] = {"step": 0}
    elif mutation == "outcome":
        report["first_outcome_mismatch"] = {"step": 0}
    elif mutation == "max_error":
        report["max_errors"][_PHYSICAL_FIELD] = _PHYSICAL_ATOL * 2
    elif mutation == "field_atol":
        report["field_atols"][_PHYSICAL_FIELD] = _PHYSICAL_ATOL * 2
    elif mutation == "runtime":
        report["runtime_replay_identity"] = {"platform": "other"}
    elif mutation == "recorded_git":
        report["recorded_git"] = {**_CLEAN_GIT, "commit": "2" * 40}
    else:
        report["recorded_physics_configuration_match"]["checked_fields"] = []

    blocker = certification_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_invalid"
    assert expected_detail in blocker["detail"]


@pytest.mark.parametrize("mutation", ["flag", "steps", "missing_history"])
def test_all_step_origin_report_history_evidence_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    recording = _recording_contract()
    recording["stable_recording_origin"] = _all_step_origin_history()
    demo = _write_manifest(tmp_path, recording=recording)
    report = _passed_report()
    report["initialization_history_match"] = True
    report["replayed_stable_origin_history"] = _all_step_origin_history()
    if mutation == "flag":
        report["initialization_history_match"] = False
    elif mutation == "steps":
        report["replayed_stable_origin_history"]["physics_steps"] = 181
    else:
        report.pop("replayed_stable_origin_history")

    blocker = certification_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_mode_invalid"


def test_old_v2_report_is_rejected(tmp_path: Path) -> None:
    demo = _write_manifest(tmp_path)
    report = _passed_report()
    report["version"] = 2

    blocker = certification_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_invalid"
    assert "version 必须为 4" in blocker["detail"]


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("raw_commands_installed", False),
        ("ik_recomputed", True),
        ("recorded_drive_targets_applied", False),
        ("raw_environment_step_called", True),
        ("recorded_drive_target_substeps_expected", 0),
        ("recorded_drive_target_substeps_executed", False),
        ("recorded_drive_target_writes_expected", -1),
        ("recorded_drive_target_writes_executed", 0.0),
        ("post_stabilization_snapshot_restore_performed", True),
        ("trajectory_state_writes_performed", True),
    ],
)
def test_recorded_drive_target_evidence_fails_closed(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    demo = _write_manifest(tmp_path)
    report = _passed_report()
    report[field] = invalid_value

    blocker = certification_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_invalid"
    assert field in blocker["detail"]


@pytest.mark.parametrize(
    ("expected_field", "executed_field"),
    [
        (
            "recorded_drive_target_substeps_expected",
            "recorded_drive_target_substeps_executed",
        ),
        (
            "recorded_drive_target_writes_expected",
            "recorded_drive_target_writes_executed",
        ),
    ],
)
def test_recorded_drive_target_counts_must_match(
    tmp_path: Path,
    expected_field: str,
    executed_field: str,
) -> None:
    demo = _write_manifest(tmp_path)
    report = _passed_report()
    report[executed_field] = int(report[expected_field]) - 1

    blocker = certification_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert executed_field in blocker["detail"]


def test_report_transition_and_input_mode_are_bound_to_manifest(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(tmp_path)
    report = _passed_report()
    report["trajectory_input_mode"] = LEGACY_RAW_ACTION_INPUT_MODE

    blocker = certification_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert "report.trajectory_input_mode 与 manifest 不一致" in blocker["detail"]


def test_manifest_rejects_mixed_transition_and_input_modes(
    tmp_path: Path,
) -> None:
    recording = _recording_contract()
    recording["trajectory_input_mode"] = LEGACY_RAW_ACTION_INPUT_MODE
    demo = _write_manifest(tmp_path, recording=recording)

    blocker = certification_report_blocker(
        demo,
        _passed_report(),
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_mode_invalid"


@pytest.mark.parametrize("mode", ["", "fresh-ish", None])
def test_unknown_certification_process_mode_fails_closed(
    tmp_path: Path,
    mode: object,
) -> None:
    recording = _recording_contract()
    recording["physical_replay_certification_mode"] = mode
    demo = _write_manifest(tmp_path, recording=recording)

    blocker = physical_replay_upload_blocker(demo)

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_mode_invalid"


def test_deferred_mode_allows_persistent_recorder_and_fresh_replay(
    tmp_path: Path,
) -> None:
    recording = _deferred_recording_contract()
    demo = _write_manifest(tmp_path, recording=recording)

    assert (
        certification_report_blocker(
            demo,
            _passed_report(),
            actual_inputs=_INPUTS,
        )
        is None
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("fresh_recorder_process_per_episode_required", True),
        ("fresh_recorder_process_per_episode_required", None),
        ("fresh_recorder_process_per_attempt_required", True),
        ("fresh_recorder_process_per_attempt_required", None),
        ("recorder_lifecycle", "fresh_process"),
        ("recorder_lifecycle", None),
        ("recorder_pid", 0),
        ("recorder_pid", True),
        ("recorder_pid", None),
        ("source_process_reuse_enabled", False),
        ("source_process_reuse_enabled", None),
        ("fresh_scene_replay_required", False),
        ("fresh_scene_replay_required", None),
        ("fresh_certifier_process_per_run_required", False),
        ("fresh_certifier_process_per_run_required", None),
        ("publish_only_after_fresh_certification", False),
        ("publish_only_after_fresh_certification", None),
        ("collection_run_seed", -1),
        ("collection_run_seed", True),
        ("collection_run_seed", None),
        ("episode_reset_seed", -1),
        ("episode_reset_seed", 20),
        ("episode_reset_seed", None),
        ("collection_seed_source", "wall_clock_text"),
        ("collection_seed_source", None),
        ("episode_reset_ordinal", -1),
        ("episode_reset_ordinal", True),
        ("episode_reset_ordinal", None),
        ("episode_seed_derivation", "seed_plus_one"),
        ("episode_seed_derivation", None),
        ("stable_recording_origin", []),
        ("stable_recording_origin", None),
    ],
)
def test_deferred_mode_manifest_contract_fails_closed(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    recording = _deferred_recording_contract()
    if invalid_value is None:
        recording.pop(field)
    else:
        recording[field] = invalid_value
    demo = _write_manifest(tmp_path, recording=recording)

    blocker = certification_report_blocker(
        demo,
        _passed_report(),
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_mode_invalid"
    assert field in blocker["detail"]


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("stabilization_velocity_clamp", True),
        ("initialization_per_physics_step_velocity_writes", True),
        ("stabilization_velocity_clamp_scope", "gate_only"),
        ("physics_steps", 0),
        ("initial_settle_steps", 181),
        ("stabilization_velocity_clamp_steps", 179),
        ("reset_pose_drift_checked", False),
        ("reset_translation_drift_limit", 0.006),
        ("reset_rotation_drift_limit", 0.06),
        ("reset_coordinate_drift_limit", 0.006),
        ("reset_max_coordinate_drift", 0.006),
        ("authored_reset_snapshot_serial", 0),
        ("require_valid_contact_report", False),
        ("contact_penetration_limit", 0.001),
        ("reset_max_translation_drift_m", 0.006),
        ("initial_geometry_gate_passed", False),
    ],
)
def test_deferred_manifest_requires_natural_settle_initialization_contract(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    recording = _deferred_recording_contract()
    stable_origin = recording["stable_recording_origin"]
    assert isinstance(stable_origin, dict)
    stable_origin[field] = invalid_value
    demo = _write_manifest(tmp_path, recording=recording)

    blocker = certification_report_blocker(
        demo,
        _passed_report(),
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_mode_invalid"
    assert field in blocker["detail"]


def test_deferred_canonical_initialization_history_requires_every_field(
    tmp_path: Path,
) -> None:
    canonical = _all_step_origin_history()
    for field in canonical:
        case_root = tmp_path / field
        case_root.mkdir()
        recording = _deferred_recording_contract()
        stable_origin = recording["stable_recording_origin"]
        assert isinstance(stable_origin, dict)
        stable_origin.pop(field)
        demo = _write_manifest(case_root, recording=recording)

        blocker = certification_report_blocker(
            demo,
            _passed_report(),
            actual_inputs=_INPUTS,
        )

        assert blocker is not None, field
        assert field in blocker["detail"]


@pytest.mark.parametrize("mutation", ["missing", "different"])
def test_deferred_replayed_initialization_history_binds_every_field(
    tmp_path: Path,
    mutation: str,
) -> None:
    canonical = _canonical_report_history(_all_step_origin_history())
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    for field in canonical:
        report = _passed_report()
        replayed = report["replayed_stable_origin_history"]
        assert isinstance(replayed, dict)
        if mutation == "missing":
            replayed.pop(field)
        else:
            replayed[field] = "changed-by-test"

        blocker = certification_report_blocker(
            demo,
            report,
            actual_inputs=_INPUTS,
        )

        assert blocker is not None, field
        assert blocker["code"] == "physical_replay_certification_invalid"
        assert f"replayed_stable_origin_history.{field}" in blocker["detail"]


@pytest.mark.parametrize(
    "field",
    [
        "source_recorder_lifecycle",
        "source_process_reuse_enabled",
        "fresh_source_process_per_attempt",
        "fresh_scene_replay_required",
        "collection_seed_source",
        "episode_reset_ordinal",
        "episode_seed_derivation",
        "source_recorder_pid",
    ],
)
def test_deferred_report_binds_source_derivation_to_manifest(
    tmp_path: Path,
    field: str,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    report = _passed_report()
    report.pop(field)

    blocker = certification_report_blocker(
        demo,
        report,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_invalid"
    assert f"report.{field}" in blocker["detail"]


def test_recorded_drive_target_mode_requires_fresh_process_per_attempt(
    tmp_path: Path,
) -> None:
    recording = _recording_contract()
    recording["fresh_recorder_process_per_attempt_required"] = False
    demo = _write_manifest(tmp_path, recording=recording)

    blocker = certification_report_blocker(
        demo,
        _passed_report(),
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_mode_invalid"


def test_fresh_v3_legacy_raw_action_is_restricted_to_e01(tmp_path: Path) -> None:
    recording = _recording_contract()
    recording["physical_transition_contract"] = LEGACY_RAW_ACTION_TRANSITION_CONTRACT
    recording["trajectory_input_mode"] = LEGACY_RAW_ACTION_INPUT_MODE
    demo = _write_manifest(
        tmp_path,
        recording=recording,
        task_name="e01_fold_garment",
    )
    report = _passed_report()
    report["physical_transition_contract"] = LEGACY_RAW_ACTION_TRANSITION_CONTRACT
    report["trajectory_input_mode"] = LEGACY_RAW_ACTION_INPUT_MODE
    report["raw_environment_step_called"] = True
    report["ik_recomputed"] = True
    report["recorded_drive_targets_applied"] = False
    report["recorded_physics_configuration_match"] = None
    for field in (
        "recorded_drive_target_substeps_expected",
        "recorded_drive_target_substeps_executed",
        "recorded_drive_target_writes_expected",
        "recorded_drive_target_writes_executed",
    ):
        report[field] = 0

    assert (
        certification_report_blocker(
            demo,
            report,
            actual_inputs=_INPUTS,
        )
        is None
    )


def test_legacy_raw_action_is_rejected_for_non_e01_task(tmp_path: Path) -> None:
    recording = _recording_contract()
    recording["physical_transition_contract"] = LEGACY_RAW_ACTION_TRANSITION_CONTRACT
    recording["trajectory_input_mode"] = LEGACY_RAW_ACTION_INPUT_MODE
    demo = _write_manifest(tmp_path, recording=recording)

    blocker = certification_report_blocker(
        demo,
        _passed_report(),
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_mode_invalid"
    assert "仅允许任务 e01_fold_garment" in blocker["detail"]


def test_deferred_acceptance_requires_three_fresh_certifier_processes(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    _write_deferred_acceptance(demo)

    assert (
        deferred_physical_replay_acceptance_upload_blocker(
            demo,
            actual_inputs=_INPUTS,
        )
        is None
    )


def test_deferred_acceptance_requires_same_contact_overlay_digest_three_times(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
        recorded_git=dict(_DIRTY_GIT),
    )
    first = _contact_signature_overlay_proof("a")
    different = _contact_signature_overlay_proof("d")
    _write_deferred_acceptance(
        demo,
        contact_signature_overlays=(first, first, different),
    )

    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        require_published=True,
    )

    assert blocker is not None
    assert blocker["code"] == "deferred_physical_replay_acceptance_invalid"
    assert "overlay digest 必须一致" in blocker["detail"]


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected_detail"),
    [
        (
            "physical_replay_certification_scope",
            "per_episode",
            "physical_replay_certification_scope",
        ),
        (
            "physical_replay_session_history_pending",
            True,
            "physical_replay_session_history_pending",
        ),
        (
            "physical_replay_session_history_id",
            "different-session",
            "physical_replay_session_history_id",
        ),
        (
            "physical_replay_session_history_sha256",
            "A" * 64,
            "physical_replay_session_history_sha256",
        ),
        (
            "physical_replay_session_history_event_count",
            0,
            "physical_replay_session_history_event_count",
        ),
        (
            "physical_replay_session_history_accepted_event_ordinal",
            0,
            "physical_replay_session_history_accepted_event_ordinal",
        ),
        (
            "physical_replay_session_history_certified_through_event_ordinal",
            4,
            "certified_through_event_ordinal",
        ),
    ],
)
def test_deferred_ordered_session_manifest_binding_fails_closed(
    tmp_path: Path,
    field: str,
    invalid_value: object,
    expected_detail: str,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    _write_deferred_acceptance(demo)
    manifest_path = demo / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recording"][field] = invalid_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "deferred_physical_replay_acceptance_invalid"
    assert expected_detail in blocker["detail"]


def test_deferred_ordered_session_metadata_must_match_manifest(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    _write_deferred_acceptance(demo)
    metadata_path = demo / "collection_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["physical_replay_session_history_sha256"] = "b" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert "collection_meta.physical_replay_session_history_sha256" in blocker["detail"]


@pytest.mark.parametrize(
    ("mutation", "expected_detail"),
    [
        ("missing_session_replay", "run_1.session_replay 必须为 object"),
        ("history_sha", "run_1.session_replay.session_history_sha256"),
        (
            "episode_order",
            "run_1.session_replay.ordered_episode_reset_ordinals",
        ),
        (
            "accepted_event",
            "run_1.session_replay.accepted_event_ordinal",
        ),
        (
            "event_target",
            "target_accepted_trajectory_event_ordinal",
        ),
        ("target_scene_rebuild", "target_scene_rebuild 必须为 null"),
        ("counter_contradiction", "事件总数与重放计数矛盾"),
    ],
)
def test_deferred_archived_report_requires_ordered_session_replay_proof(
    tmp_path: Path,
    mutation: str,
    expected_detail: str,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    acceptance = _write_deferred_acceptance(demo)
    runs = acceptance["certification_runs"]
    assert isinstance(runs, list)
    report_path = demo / str(runs[0]["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if mutation == "missing_session_replay":
        report.pop("session_replay")
    elif mutation == "history_sha":
        report["session_replay"]["session_history_sha256"] = "b" * 64
    elif mutation == "episode_order":
        report["session_replay"]["ordered_episode_reset_ordinals"] = [3]
    elif mutation == "accepted_event":
        report["session_replay"]["accepted_event_ordinal"] = 4
    elif mutation == "event_target":
        report["session_replay"]["event_replay"][
            "target_accepted_trajectory_event_ordinal"
        ] = 4
    elif mutation == "target_scene_rebuild":
        report["session_replay"]["event_replay"]["target_scene_rebuild"] = {
            "scene_rebuilt": True,
            "scene_generation": 2,
            "seed": 19,
        }
    else:
        report["session_replay"]["event_replay"]["discarded_trajectories_replayed"] = 1
    report_bytes = json.dumps(report, sort_keys=True).encode("utf-8")
    report_path.write_bytes(report_bytes)
    runs[0]["report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    (demo / "fresh_process_acceptance.json").write_text(
        json.dumps(acceptance),
        encoding="utf-8",
    )

    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "deferred_physical_replay_acceptance_invalid"
    assert expected_detail in blocker["detail"]


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("certification_scope", "per_episode"),
        ("collection_session_id", "different-session"),
        ("session_episode_count", 2),
        ("session_episode_index", 2),
        ("session_episode_reset_ordinals", [3]),
        ("session_demo_names", ["another-demo"]),
        ("fresh_certifier_process_per_session_run", False),
        ("session_publication_all_targets_prevalidated", False),
        ("session_publication_rollback_on_catchable_error", False),
    ],
)
def test_deferred_acceptance_session_contract_fails_closed(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    acceptance = _write_deferred_acceptance(demo)
    acceptance[field] = invalid_value
    (demo / "fresh_process_acceptance.json").write_text(
        json.dumps(acceptance),
        encoding="utf-8",
    )

    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "deferred_physical_replay_acceptance_invalid"
    assert field in blocker["detail"]


def test_deferred_publication_requires_atomic_session_commit_marker(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    _write_deferred_acceptance(demo, write_publication_commit=False)

    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert "session publication commit marker" in blocker["detail"]
    assert (
        deferred_physical_replay_acceptance_upload_blocker(
            demo,
            actual_inputs=_INPUTS,
            require_publication_commit=False,
        )
        is None
    )


def test_deferred_publication_marker_binds_current_acceptance_hash(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    acceptance = _write_deferred_acceptance(demo)
    acceptance["accepted_at"] = "2026-08-02T02:00:00+00:00"
    (demo / "fresh_process_acceptance.json").write_text(
        json.dumps(acceptance),
        encoding="utf-8",
    )

    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert (
        "当前 acceptance 与 session publication marker 哈希不一致" in blocker["detail"]
    )


def test_deferred_gate_loads_and_verifies_actual_finalized_history(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    _write_deferred_acceptance(demo)
    manifest = json.loads((demo / "replay_manifest.json").read_text(encoding="utf-8"))
    history_path = (
        demo.parent
        / manifest["recording"]["physical_replay_session_history_root_relative_path"]
    )
    event_path = history_path / "events" / "00000000" / "event.json"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["payload"]["seed"] = 999
    event_path.write_text(json.dumps(event), encoding="utf-8")

    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert "session history 未通过 finalized 完整性验证" in blocker["detail"]


def test_deferred_gate_rejects_symlinked_session_history(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    _write_deferred_acceptance(demo)
    manifest = json.loads((demo / "replay_manifest.json").read_text(encoding="utf-8"))
    history_path = (
        demo.parent
        / manifest["recording"]["physical_replay_session_history_root_relative_path"]
    )
    moved_path = history_path.with_name(f"{history_path.name}-real")
    history_path.rename(moved_path)
    history_path.symlink_to(moved_path, target_is_directory=True)

    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert "session history 路径不得包含符号链接" in blocker["detail"]


def test_deferred_marker_verifies_every_sibling_acceptance(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    acceptance = _write_deferred_acceptance(demo)
    missing_name = "missing-session-member"
    acceptance["session_episode_count"] = 2
    acceptance["session_episode_reset_ordinals"] = [2, 3]
    acceptance["session_demo_names"] = [demo.name, missing_name]
    acceptance_path = demo / "fresh_process_acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")

    marker_path = demo.parent / str(
        acceptance["session_publication_commit_root_relative_path"]
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["session_demo_names"] = [demo.name, missing_name]
    marker["session_episode_reset_ordinals"] = [2, 3]
    marker["members"][0]["acceptance_sha256"] = hashlib.sha256(
        acceptance_path.read_bytes()
    ).hexdigest()
    marker["members"].append(
        {
            "demo_name": missing_name,
            "episode_reset_ordinal": 3,
            "accepted_event_ordinal": 5,
            "acceptance_sha256": "b" * 64,
            "certified_inputs": dict(_INPUTS),
        }
    )
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert "sibling acceptance 无法验证" in blocker["detail"]


def test_deferred_marker_recomputes_every_member_input_identity(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    _write_deferred_acceptance(demo)
    (demo / "replay.zarr" / "data" / "action" / "0.0").write_bytes(b"tampered-actions")

    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert "members[0].certified_inputs 与 sibling 当前输入不一致" in blocker["detail"]


def test_ordered_session_report_requires_exact_adaptive_stable_origin(
    tmp_path: Path,
) -> None:
    recording = _deferred_recording_contract()
    recording["stable_recording_origin"] = _natural_settle_origin_history()
    demo = _write_manifest(tmp_path, recording=recording)
    acceptance = _write_deferred_acceptance(demo)
    runs = acceptance["certification_runs"]
    assert isinstance(runs, list)
    report_path = demo / str(runs[0]["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    replayed = report["replayed_stable_origin_history"]
    replayed["physics_steps"] = 204
    replayed["stability_samples"] = 84
    replayed["hold_target_sync"]["passes"] = 206
    replayed["reset_drift_sample_count"] = 205
    replayed["reset_drift_expected_sample_count"] = 205
    replayed["initialization_contact_summary"]["sample_count"] = 205
    replayed["initialization_contact_summary"]["expected_sample_count"] = 205
    report_bytes = json.dumps(report, sort_keys=True).encode("utf-8")
    report_path.write_bytes(report_bytes)
    runs[0]["report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    (demo / "fresh_process_acceptance.json").write_text(
        json.dumps(acceptance),
        encoding="utf-8",
    )

    assert (
        certification_report_blocker(
            demo,
            report,
            actual_inputs=capture_certified_inputs(demo),
        )
        is None
    )
    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert "replayed_stable_origin_history 必须与 ordered session" in blocker["detail"]


def test_ordered_session_accepts_canonical_stable_origin_projection(
    tmp_path: Path,
) -> None:
    recording = _deferred_recording_contract()
    stable_origin = _natural_settle_origin_history()
    stable_origin.update(
        {
            "strategy": "continuous_current_target_all_velocity_settle",
            "physics_dt": _PHYSICS_DT,
            "timeout_seconds": 3.0,
            "collection_run_seed": 17,
            "episode_reset_seed": 19,
            "episode_reset_ordinal": 2,
            "gate_max_task_linear_velocity": 0.00001,
            "final_max_task_linear_velocity": 0.0,
            "final_robot_position_target_error": 0.0,
        }
    )
    # Diagnostic-only fields beside the canonical initialization projection
    # must not change the exact report contract.
    for index in range(96 - len(stable_origin)):
        stable_origin[f"diagnostic_only_{index}"] = index
    recording["stable_recording_origin"] = stable_origin
    demo = _write_manifest(tmp_path, recording=recording)
    acceptance = _write_deferred_acceptance(demo)
    first_report = json.loads(
        (demo / str(acceptance["certification_runs"][0]["report"])).read_text(
            encoding="utf-8"
        )
    )
    replayed = first_report["replayed_stable_origin_history"]
    expected_projection = _canonical_report_history(_all_step_origin_history())

    assert len(stable_origin) == 96
    assert len(replayed) == len(expected_projection)
    assert "strategy" not in replayed
    assert set(replayed) == set(expected_projection)
    assert (
        deferred_physical_replay_acceptance_upload_blocker(
            demo,
            actual_inputs=_INPUTS,
        )
        is None
    )


def test_physical_upload_dispatches_persistent_source_to_deferred_gate(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    inputs = _materialize_certified_inputs(demo)
    report = _passed_report()
    report["certified_inputs"] = dict(inputs)
    (demo / REPORT_NAME).write_text(json.dumps(report), encoding="utf-8")
    _write_deferred_acceptance(demo, inputs=inputs)

    assert physical_replay_upload_blocker(demo) is None


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("persistent_source_process_for_collection_batch", None),
        ("persistent_source_process_for_collection_batch", False),
        ("fresh_source_process_per_episode", None),
        ("fresh_source_process_per_episode", True),
        ("fresh_certifier_process_per_run", None),
        ("fresh_certifier_process_per_run", False),
        ("recorder_lifecycle", None),
        ("recorder_lifecycle", "fresh_process"),
        ("fresh_scene_replay_required", None),
        ("fresh_scene_replay_required", False),
        ("fresh_certifier_process_per_run_required", None),
        ("fresh_certifier_process_per_run_required", False),
        ("source_recorder_pid", None),
        ("source_recorder_pid", 102),
        ("recorder_pid", None),
        ("recorder_pid", 102),
    ],
)
def test_deferred_acceptance_source_and_certifier_contract_fails_closed(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    acceptance = _write_deferred_acceptance(demo)
    if invalid_value is None:
        acceptance.pop(field)
    else:
        acceptance[field] = invalid_value
    (demo / "fresh_process_acceptance.json").write_text(
        json.dumps(acceptance),
        encoding="utf-8",
    )

    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "deferred_physical_replay_acceptance_invalid"
    assert field in blocker["detail"]


def test_deferred_acceptance_rejects_source_pid_reused_by_certifier(
    tmp_path: Path,
) -> None:
    recording = _deferred_recording_contract()
    recording["recorder_pid"] = 201
    demo = _write_manifest(tmp_path, recording=recording)
    _write_deferred_acceptance(demo, recorder_pid=201)

    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "deferred_physical_replay_acceptance_invalid"
    assert "persistent recorder 与 certifier 必须是不同进程" in blocker["detail"]


@pytest.mark.parametrize(
    ("mutation", "expected_detail"),
    [
        ("duplicate_pid", "3 个不同 certifier pid"),
        ("run_failed", "run_2.status 必须为 passed"),
        ("positive_returncode", "run_2.returncode 必须为整数 0"),
        ("report_failed", "run_2.report 未通过单次认证门禁"),
        ("run_input_changed", "run_2.report 未通过单次认证门禁"),
        ("acceptance_input_changed", "certified_inputs 与当前轨迹不一致"),
    ],
)
def test_deferred_acceptance_fails_closed(
    tmp_path: Path,
    mutation: str,
    expected_detail: str,
) -> None:
    demo = _write_manifest(
        tmp_path,
        recording=_deferred_recording_contract(),
    )
    acceptance = _write_deferred_acceptance(demo)
    runs = acceptance["certification_runs"]
    assert isinstance(runs, list)

    if mutation == "duplicate_pid":
        runs[1]["pid"] = runs[0]["pid"]
        acceptance["certifier_pids"] = [run["pid"] for run in runs]
    elif mutation == "run_failed":
        runs[1]["status"] = "failed"
    elif mutation == "positive_returncode":
        runs[1]["returncode"] = 2
    elif mutation in {"report_failed", "run_input_changed"}:
        report_path = demo / str(runs[1]["report"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if mutation == "report_failed":
            report["status"] = "failed"
            report["reason"] = "test failure"
        else:
            changed_inputs = dict(_INPUTS)
            changed_inputs["replay_zarr_sha256"] = "9" * 64
            report["certified_inputs"] = changed_inputs
            runs[1]["certified_inputs"] = changed_inputs
        report_bytes = json.dumps(report, sort_keys=True).encode("utf-8")
        report_path.write_bytes(report_bytes)
        runs[1]["report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    else:
        changed_inputs = dict(_INPUTS)
        changed_inputs["replay_zarr_sha256"] = "9" * 64
        acceptance["certified_inputs"] = changed_inputs

    (demo / "fresh_process_acceptance.json").write_text(
        json.dumps(acceptance),
        encoding="utf-8",
    )
    blocker = deferred_physical_replay_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "deferred_physical_replay_acceptance_invalid"
    assert expected_detail in blocker["detail"]


def test_fresh_acceptance_requires_attempt_lifecycle_and_contract_binding(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(tmp_path)
    (demo / "fresh_process_acceptance.json").write_text(
        json.dumps(
            {
                "format": FRESH_PROCESS_ACCEPTANCE_FORMAT,
                "version": FRESH_PROCESS_ACCEPTANCE_VERSION,
                "status": "passed",
                "certification_runs_required": 3,
                "certification_runs_passed": 3,
                "fresh_recorder_process_per_episode": True,
                "fresh_recorder_process_per_attempt": False,
                "fresh_certifier_process_per_run": True,
                "published_only_after_certification": True,
                "physical_transition_contract": (LEGACY_RAW_ACTION_TRANSITION_CONTRACT),
                "trajectory_input_mode": LEGACY_RAW_ACTION_INPUT_MODE,
                "certified_inputs": dict(_INPUTS),
                "certification_runs": [],
            }
        ),
        encoding="utf-8",
    )

    blocker = fresh_process_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert blocker["code"] == "fresh_process_acceptance_invalid"
    assert "fresh_recorder_process_per_attempt 必须为 true" in blocker["detail"]
    assert "physical_transition_contract 与 manifest 不一致" in blocker["detail"]
    assert "trajectory_input_mode 与 manifest 不一致" in blocker["detail"]


def test_fresh_acceptance_rejects_positive_certifier_returncode(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(tmp_path)
    reports_dir = demo / "physical_replay_certification_runs"
    reports_dir.mkdir()
    runs: list[dict[str, object]] = []
    for index, pid in enumerate((201, 202, 203), start=1):
        report = _passed_report()
        report_bytes = json.dumps(report, sort_keys=True).encode("utf-8")
        report_path = reports_dir / f"run_{index}.json"
        report_path.write_bytes(report_bytes)
        runs.append(
            {
                "run": index,
                "pid": pid,
                "status": "passed",
                "returncode": 2 if index == 2 else 0,
                "started_at": f"2026-08-02T01:0{index}:00+00:00",
                "finished_at": f"2026-08-02T01:0{index}:30+00:00",
                "report": (f"physical_replay_certification_runs/run_{index}.json"),
                "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "certified_inputs": dict(_INPUTS),
            }
        )
    acceptance = {
        "format": FRESH_PROCESS_ACCEPTANCE_FORMAT,
        "version": FRESH_PROCESS_ACCEPTANCE_VERSION,
        "status": "passed",
        "certification_runs_required": 3,
        "certification_runs_passed": 3,
        "fresh_recorder_process_per_episode": True,
        "fresh_recorder_process_per_attempt": True,
        "fresh_certifier_process_per_run": True,
        "published_only_after_certification": True,
        "physical_transition_contract": (RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT),
        "trajectory_input_mode": RECORDED_DRIVE_TARGET_INPUT_MODE,
        "certified_inputs": dict(_INPUTS),
        "recorder": {
            "pid": 101,
            "returncode": 0,
            "started_at": "2026-08-02T01:00:00+00:00",
            "finished_at": "2026-08-02T01:00:30+00:00",
        },
        "recorder_pid": 101,
        "certifier_pids": [201, 202, 203],
        "certification_runs": runs,
    }
    (demo / "fresh_process_acceptance.json").write_text(
        json.dumps(acceptance),
        encoding="utf-8",
    )

    blocker = fresh_process_acceptance_upload_blocker(
        demo,
        actual_inputs=_INPUTS,
    )

    assert blocker is not None
    assert "run_2.returncode 必须为整数 0" in blocker["detail"]


def test_new_physical_mode_hint_cannot_disable_certification(
    tmp_path: Path,
) -> None:
    recording = _recording_contract()
    recording["physical_replay_certification_required"] = False
    demo = _write_manifest(tmp_path, recording=recording)

    blocker = certification_upload_blocker(demo)

    assert blocker is not None
    assert blocker["code"] == "physical_replay_certification_required_invalid"
