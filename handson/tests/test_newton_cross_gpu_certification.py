from __future__ import annotations

import json
from pathlib import Path

from handson.data import newton_cross_gpu_certification as certification
from handson.data.physical_replay_certification import (
    physical_replay_upload_blocker,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _asset_tree(digit: str) -> dict:
    return {
        "format": "test.asset_tree",
        "version": 1,
        "file_count": 1,
        "files": [{"path": "asset.bin", "sha256": digit * 64}],
        "combined_sha256": digit * 64,
    }


def _physics_config() -> dict:
    return {
        "backend": "nvidia_newton_mujoco_warp",
        "physics_dt": 1.0 / 240.0,
        "control_dt": 1.0 / 30.0,
        "n_substeps": 8,
        "solver": {
            "solver": "newton",
            "integrator": "implicitfast",
            "iterations": 100,
        },
    }


def _identity(model: str, arch: str, uuid: str) -> dict:
    return {
        "python": "3.12.12",
        "platform": "Linux-test",
        "device": "cuda:0",
        "software_versions": {
            "isaacsim": "6.0.0",
            "isaaclab": "3.0.0",
            "isaaclab-newton": "0.13.6",
            "newton": "1.4.0",
            "warp-lang": "1.15.0",
            "mujoco": "3.10.0",
            "mujoco-warp": "3.10.0.2",
        },
        "b03_physics_source_sha256": "a" * 64,
        "newton_qd_projection": {
            "format": "handson.newton_qd_projection",
            "version": 1,
            "combined_sha256": "b" * 64,
            "module_sha256": {
                "mujoco_warp._src.forward": "c" * 64,
                "newton._src.solvers.mujoco.solver_mujoco": "d" * 64,
            },
        },
        "b03_generated_robot_assets": {
            "right": {
                "format": "handson.b03_generated_robot_asset",
                "version": 3,
                "generated_mjcf_sha256": "e" * 64,
                "referenced_assets": _asset_tree("1"),
                "loaded_usd_cache": _asset_tree("2"),
            },
            "left": {
                "format": "handson.b03_generated_robot_asset",
                "version": 3,
                "generated_mjcf_sha256": "f" * 64,
                "referenced_assets": _asset_tree("3"),
                "loaded_usd_cache": _asset_tree("4"),
            },
            "task": {
                "format": "handson.b03_loaded_task_asset",
                "version": 2,
                "generated_mjcf_sha256": "8" * 64,
                "canonical_generated_mjcf_sha256": "5" * 64,
                "referenced_assets": _asset_tree("6"),
                "loaded_usd_cache": _asset_tree("7"),
            },
        },
        "gpu_name": model,
        "gpu_arch": arch,
        "gpu_uuid": uuid,
        "gpu_pci_bus_id": "0000:01:00.0",
        "cuda_driver_version": [13, 0],
        "cuda_toolkit_version": [13, 0],
        "nvidia_driver_version": "580.173.02",
        "warp_deterministic_mode": "gpu_to_gpu",
        "solver_deterministic_mode": "gpu_to_gpu",
        "collision_pipeline_deterministic": True,
        "collision_broad_phase": "explicit",
        "cross_gpu_method": certification.NEWTON_CROSS_GPU_METHOD,
        "determinism_profile": "cross_gpu_v1",
        "cross_gpu_ready": True,
    }


def _source(tmp_path: Path) -> tuple[Path, dict, dict]:
    demo = tmp_path / "b03_demo__newton_v2"
    demo.mkdir()
    source_identity = _identity("NVIDIA RTX 4090", "sm_89", "GPU-A")
    replay_identity = _identity("NVIDIA RTX 5090", "sm_120", "GPU-B")
    generation = {
        "dataset_generation": certification.DATASET_GENERATION,
        "training_generation_status": certification.DATASET_TRAINING_STATUS,
        "dataset_directory_suffix": certification.DATASET_DIRECTORY_SUFFIX,
        "dataset_marker_file": certification.DATASET_MARKER_NAME,
        "is_newton_cross_gpu_v2_data": True,
    }
    recording = {
        **generation,
        "sim_backend": "isaacsim_newton",
        "step_count": 12,
        "newton_replay_profile": "cross_gpu_v1",
        "newton_cross_gpu_certification_required": True,
        "newton_cross_gpu_certification_runs_required": 3,
        "newton_cross_gpu_certification_mode": (
            "fresh_process_distinct_gpu_model_v1"
        ),
        "publish_only_after_newton_cross_gpu_certification": True,
        "fresh_scene_replay_required": True,
        "fresh_certifier_process_per_run_required": True,
        "live_direct_hand_control": False,
        "newton_physical_replay_available": True,
        "recorded_drive_target_replay_required": True,
        "physical_transition_contract": (
            "seeded_fresh_newton_reset_plus_per_substep_drive_targets_"
            "cross_gpu_v2"
        ),
        "trajectory_input_mode": (
            "recorded_newton_substep_robot_joint_position_targets"
        ),
        "physical_replay_drive_target_source": (
            "exact_recorded_per_newton_substep_robot_position_targets"
        ),
        "randomize_dynamics": False,
        "action_shape_per_step": [certification.B03_POLICY_ACTION_DIM],
        "action_storage_layout": certification.B03_ACTION_STORAGE_LAYOUT,
    }
    _write_json(
        demo / "replay_manifest.json",
        {
            "task": {"name": "t09_insert_peg"},
            "recording": recording,
        },
    )
    _write_json(
        demo / "collection_meta.json",
        {
            "task_name": "t09_insert_peg",
            "sim_backend": "isaacsim_newton",
            "task_outcome": "success",
            "source_outcome": "success",
            "step_count": 12,
            **generation,
            **{
                name: recording[name]
                for name in (
                    "newton_replay_profile",
                    "newton_cross_gpu_certification_required",
                    "newton_cross_gpu_certification_runs_required",
                    "newton_cross_gpu_certification_mode",
                    "publish_only_after_newton_cross_gpu_certification",
                    "live_direct_hand_control",
                )
            },
        },
    )
    _write_json(
        demo / certification.DATASET_MARKER_NAME,
        {
            "format": "handson.dataset_generation",
            "version": 1,
            **generation,
        },
    )
    determinism = {
        "profile": "cross_gpu_v1",
        "cross_gpu_requested": True,
        "cross_gpu_ready": True,
        "cross_gpu_method": (
            "warp_gpu_to_gpu_plus_semantic_three_fresh_runs_distinct_gpu_model_v2"
        ),
        "warp_deterministic_mode": "gpu_to_gpu",
        "solver_deterministic_mode": "gpu_to_gpu",
        "bitwise_cross_gpu_claim": False,
        "collision_pipeline_deterministic": True,
        "collision_pipeline_deterministic_required": True,
        "collision_broad_phase": "explicit",
    }
    _write_json(
        demo / "recording_schema.json",
        {
            "environment_schema": {
                "name": "handson.b03_newton_ground_truth",
                "version": 3,
                "fields": {
                    "newton_robot_joint_position": {
                        "physical_replay": True,
                        "replay_atol": 0.0002,
                    }
                },
                "newton": {
                    "bitwise_checkpoint": False,
                    "requires_fresh_process_certification": True,
                    "requires_distinct_gpu_model_certification": True,
                    "allows_recorded_t0_state_restore": False,
                    "certification_forbids_state_writes": True,
                    "certification_forbids_state_writes_after_action_start": True,
                    "replay_input": (
                        "exact_recorded_per_newton_substep_robot_position_"
                        "targets"
                    ),
                    "hand_peg_max_solid_penetration_m": 0.001,
                    "cross_gpu_replay_profile": "cross_gpu_v1",
                    "cross_gpu_certification_runs_required": 3,
                    "determinism": determinism,
                    "physics_config": _physics_config(),
                    "source_runtime_identity": source_identity,
                },
            }
        },
    )
    replay_tree = demo / "replay.zarr" / "data"
    replay_tree.mkdir(parents=True)
    (replay_tree / "placeholder").write_bytes(b"immutable-replay")
    return demo, source_identity, replay_identity


def _report(
    demo: Path,
    source_identity: dict,
    replay_identity: dict,
    *,
    index: int,
) -> dict:
    return {
        "format": certification.REPORT_FORMAT,
        "version": certification.REPORT_VERSION,
        "task": "t09_insert_peg",
        "profile": "cross_gpu_v1",
        "drive_input": certification.DRIVE_INPUT_MODE,
        "fresh_process_per_demo": True,
        "state_restore_used": False,
        "same_gpu_smoke": False,
        "certification_eligible": True,
        "comparison_passed": True,
        "task_success_reproduced": True,
        "source_contact_evidence_valid": True,
        "replay_contact_buffers_readable": True,
        "replay_hand_peg_contact_observed": True,
        "replay_max_hand_peg_penetration_m": 0.0002,
        "passed": True,
        "certified_inputs": certification.capture_newton_certified_inputs(demo),
        "source_runtime_identity": source_identity,
        "replay_runtime_identity": replay_identity,
        "semantic_physics_identity_exact_match": True,
        "runtime_physics_config_exact_match": True,
        "replay_runtime_physics_config": _physics_config(),
        "replay_runtime_physics_config_sha256": (
            certification.canonical_physics_config_sha256(_physics_config())
        ),
        "cross_gpu_environment_differences": (
            certification.runtime_environment_differences(
                source_identity,
                replay_identity,
            )
        ),
        "replay_determinism": {
            "profile": "cross_gpu_v1",
            "cross_gpu_requested": True,
            "collision_pipeline_deterministic": True,
            "collision_pipeline_deterministic_required": True,
            "collision_broad_phase": "explicit",
            "cross_gpu_ready": True,
            "cross_gpu_method": (
                "warp_gpu_to_gpu_plus_semantic_three_fresh_runs_distinct_gpu_model_v2"
            ),
            "warp_deterministic_mode": "gpu_to_gpu",
            "solver_deterministic_mode": "gpu_to_gpu",
            "bitwise_cross_gpu_claim": False,
        },
        "steps_requested": 12,
        "steps_replayed": 12,
        "state_writes_after_action_start": 0,
        "state_write_journal": [],
        "scene_lifecycle": {"scene_destroyed": True},
        "max_error_by_metric": {"newton_robot_joint_position": 1e-5},
        "run_id": f"run-{index}",
        "process_id": 5000 + index,
        "started_at": f"2026-08-07T00:00:0{index}+00:00",
        "completed_at": f"2026-08-07T00:01:0{index}+00:00",
    }


def test_three_distinct_model_runs_unlock_the_shared_upload_gate(tmp_path: Path) -> None:
    demo, source_identity, replay_identity = _source(tmp_path)
    replay_identity["platform"] = "Linux-cloud"
    paths = []
    for index in range(1, 4):
        path = demo / certification.RUN_DIRECTORY / f"run_{index:03d}.json"
        _write_json(
            path,
            _report(
                demo,
                source_identity,
                replay_identity,
                index=index,
            ),
        )
        paths.append(path)

    acceptance = certification.build_acceptance(demo, paths)

    assert certification.upload_blocker(demo) is None
    assert physical_replay_upload_blocker(demo) is None
    assert acceptance["semantic_physics_identity_exact_match"] is True
    assert set(acceptance["cross_gpu_environment_differences"]) >= {
        "platform",
        "gpu_name",
        "gpu_arch",
        "gpu_uuid",
    }


def test_changed_qd_projection_remains_a_hard_identity_failure(
    tmp_path: Path,
) -> None:
    demo, source_identity, replay_identity = _source(tmp_path)
    replay_identity["newton_qd_projection"] = {
        **replay_identity["newton_qd_projection"],
        "combined_sha256": "9" * 64,
    }
    report = _report(
        demo,
        source_identity,
        replay_identity,
        index=1,
    )

    errors = certification.report_errors(demo, report)

    assert any("semantic physics identities differ" in error for error in errors)


def test_changed_loaded_usd_content_remains_a_hard_identity_failure(
    tmp_path: Path,
) -> None:
    demo, source_identity, replay_identity = _source(tmp_path)
    replay_identity["b03_generated_robot_assets"]["task"][
        "loaded_usd_cache"
    ] = _asset_tree("8")
    report = _report(demo, source_identity, replay_identity, index=1)

    errors = certification.report_errors(demo, report)

    assert any("semantic physics identities differ" in error for error in errors)


def test_reported_runtime_physics_config_is_recomputed_and_bound(
    tmp_path: Path,
) -> None:
    demo, source_identity, replay_identity = _source(tmp_path)
    report = _report(demo, source_identity, replay_identity, index=1)
    report["replay_runtime_physics_config"]["solver"]["iterations"] = 99
    report["replay_runtime_physics_config_sha256"] = (
        certification.canonical_physics_config_sha256(
            report["replay_runtime_physics_config"]
        )
    )

    errors = certification.report_errors(demo, report)

    assert any("physics config differs from source" in error for error in errors)


def test_replay_identity_error_marker_is_fail_closed(tmp_path: Path) -> None:
    demo, source_identity, replay_identity = _source(tmp_path)
    replay_identity["newton_qd_projection_error"] = "stale overlay"
    report = _report(demo, source_identity, replay_identity, index=1)

    errors = certification.report_errors(demo, report)

    assert any("newton_qd_projection_error" in error for error in errors)


def test_same_gpu_model_never_qualifies_even_with_a_different_uuid(tmp_path: Path) -> None:
    demo, source_identity, _replay_identity = _source(tmp_path)
    same_model = _identity("NVIDIA RTX 4090", "sm_89", "GPU-OTHER")
    report = _report(
        demo,
        source_identity,
        same_model,
        index=1,
    )

    errors = certification.report_errors(demo, report)

    assert any("distinct GPU models" in error for error in errors)


def test_solver_gpu_to_gpu_evidence_is_required_in_identity_and_contract(
    tmp_path: Path,
) -> None:
    demo, source_identity, replay_identity = _source(tmp_path)
    report = _report(
        demo,
        source_identity,
        replay_identity,
        index=1,
    )
    report["replay_runtime_identity"]["solver_deterministic_mode"] = (
        "run_to_run"
    )
    report["replay_determinism"]["solver_deterministic_mode"] = (
        "run_to_run"
    )

    errors = certification.report_errors(demo, report)

    assert any("SolverMuJoCo mode" in error for error in errors)


def test_mutated_archived_report_relocks_upload(tmp_path: Path) -> None:
    demo, source_identity, replay_identity = _source(tmp_path)
    paths = []
    for index in range(1, 4):
        path = demo / certification.RUN_DIRECTORY / f"run_{index:03d}.json"
        _write_json(path, _report(demo, source_identity, replay_identity, index=index))
        paths.append(path)
    certification.build_acceptance(demo, paths)
    changed = json.loads(paths[1].read_text(encoding="utf-8"))
    changed["steps_replayed"] = 11
    _write_json(paths[1], changed)

    blocker = physical_replay_upload_blocker(demo)

    assert blocker is not None
    assert blocker["code"] == "newton_cross_gpu_certification_invalid"


def test_mutated_acceptance_runtime_identity_relocks_upload(tmp_path: Path) -> None:
    demo, source_identity, replay_identity = _source(tmp_path)
    paths = []
    for index in range(1, 4):
        path = demo / certification.RUN_DIRECTORY / f"run_{index:03d}.json"
        _write_json(path, _report(demo, source_identity, replay_identity, index=index))
        paths.append(path)
    certification.build_acceptance(demo, paths)
    acceptance_path = demo / certification.ACCEPTANCE_NAME
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance["replay_runtime_identity"]["gpu_name"] = "forged"
    _write_json(acceptance_path, acceptance)

    blocker = physical_replay_upload_blocker(demo)

    assert blocker is not None
    assert blocker["code"] == "newton_cross_gpu_certification_invalid"


def test_missing_semantic_metric_never_qualifies(tmp_path: Path) -> None:
    demo, source_identity, replay_identity = _source(tmp_path)
    report = _report(demo, source_identity, replay_identity, index=1)
    report["max_error_by_metric"] = {}

    errors = certification.report_errors(demo, report)

    assert any("newton_robot_joint_position" in error for error in errors)


def test_missing_manifest_is_blocked_by_collection_identity_hint(
    tmp_path: Path,
) -> None:
    demo, _source_identity, _replay_identity = _source(tmp_path)
    (demo / "replay_manifest.json").unlink()

    blocker = physical_replay_upload_blocker(demo)

    assert blocker is not None
    assert blocker["code"] == "newton_cross_gpu_source_invalid"


def test_marker_mutation_after_acceptance_relocks_upload(tmp_path: Path) -> None:
    demo, source_identity, replay_identity = _source(tmp_path)
    paths = []
    for index in range(1, 4):
        path = demo / certification.RUN_DIRECTORY / f"run_{index:03d}.json"
        _write_json(path, _report(demo, source_identity, replay_identity, index=index))
        paths.append(path)
    certification.build_acceptance(demo, paths)
    marker_path = demo / certification.DATASET_MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["post_certification_mutation"] = True
    _write_json(marker_path, marker)

    blocker = physical_replay_upload_blocker(demo)

    assert blocker is not None
    assert blocker["code"] == "newton_cross_gpu_certification_invalid"
