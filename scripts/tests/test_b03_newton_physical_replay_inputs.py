from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import zarr


_SCRIPT = Path(__file__).resolve().parents[1] / "replay_b03_newton_physics.py"
_SPEC = importlib.util.spec_from_file_location(
    "replay_b03_newton_physics",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _asset_tree(digit: str) -> dict:
    return {
        "file_count": 1,
        "files": [{"path": "asset.bin", "sha256": digit * 64}],
        "combined_sha256": digit * 64,
    }


def _runtime_identity() -> dict:
    robot_assets = {}
    for side, digit in (("right", "4"), ("left", "5")):
        robot_assets[side] = {
            "format": "handson.b03_generated_robot_asset",
            "version": 3,
            "generated_mjcf_sha256": digit * 64,
            "referenced_assets": _asset_tree("6"),
            "loaded_usd_cache": _asset_tree("7"),
        }
    robot_assets["task"] = {
        "format": "handson.b03_loaded_task_asset",
        "version": 2,
        "generated_mjcf_sha256": "f" * 64,
        "canonical_generated_mjcf_sha256": "8" * 64,
        "referenced_assets": _asset_tree("9"),
        "loaded_usd_cache": _asset_tree("a"),
    }
    return {
        "python": "3.11.0",
        "platform": "Linux-test",
        "software_versions": {"newton": "1.4.0"},
        "b03_physics_source_sha256": "b" * 64,
        "newton_qd_projection": {
            "combined_sha256": "c" * 64,
            "module_sha256": {
                "mujoco_warp._src.forward": "d" * 64,
                "newton._src.solvers.mujoco.solver_mujoco": "e" * 64,
            },
        },
        "b03_generated_robot_assets": robot_assets,
        "gpu_name": "NVIDIA RTX 4090",
        "gpu_arch": "sm_89",
        "cuda_driver_version": [13, 0],
        "cuda_toolkit_version": [13, 0],
        "nvidia_driver_version": "580.173.02",
        "determinism_profile": "cross_gpu_v1",
        "cross_gpu_ready": True,
        "warp_deterministic_mode": "gpu_to_gpu",
        "solver_deterministic_mode": "gpu_to_gpu",
        "collision_pipeline_deterministic": True,
        "collision_broad_phase": "explicit",
        "cross_gpu_method": (
            "warp_gpu_to_gpu_plus_semantic_three_fresh_runs_"
            "distinct_gpu_model_v2"
        ),
    }


def _write_source(
    root: Path,
    *,
    outcome: str = "success",
    metadata_steps: int = 2,
) -> Path:
    root.mkdir()
    generation = {
        "dataset_generation": _MODULE.DATASET_GENERATION,
        "training_generation_status": _MODULE.DATASET_TRAINING_STATUS,
        "dataset_directory_suffix": _MODULE.DATASET_DIRECTORY_SUFFIX,
        "dataset_marker_file": _MODULE.DATASET_MARKER_NAME,
        "is_newton_cross_gpu_v2_data": True,
    }
    execution = {
        "cuda_graph_requested": True,
        "use_cuda_graph": True,
        "cuda_graph_captured": True,
        "cuda_graph_scope": "newton_manager_simulate_full",
        "cuda_graph_capture_mode": "rtx_relaxed_deferred",
        "capture_completed_before_episode_reset": True,
        "newton_manager_num_substeps": 1,
        "newton_manager_decimation": 1,
    }
    solver = {
        "solver": "newton",
        "integrator": "implicitfast",
        "cone": "pyramidal",
        "iterations": 100,
        "line_search_iterations": 50,
        "ccd_iterations": 50,
        "tolerance": 1.0e-8,
        "njmax": 512,
        "nconmax": 512,
        "capacity_policy": "fixed_explicit",
        "mujoco_read_only_sensor_pipeline": "disabled",
        "use_mujoco_cpu": False,
        "use_mujoco_contacts": False,
        "use_newton_actuators": True,
        "deterministic_mode": "gpu_to_gpu",
    }
    physics_config = {
        "backend": "nvidia_newton_mujoco_warp",
        "device": "cuda:0",
        "physics_dt": 1.0 / 120.0,
        "live_physics_dt": 1.0 / 120.0,
        "control_dt": 1.0 / 30.0,
        "n_substeps": 4,
        "solver": solver,
        "execution": execution,
        "contact_model": {},
        "determinism": {},
    }
    reset_contract = {
        "strategy": (
            "fresh_newton_process_seeded_environment_reset_then_exact_"
            "recorded_per_substep_drive_targets"
        ),
        "requires_collection_run_seed": True,
        "requires_episode_reset_seed": True,
        "requires_fresh_process": True,
        "allow_pre_action_state_overwrite": False,
        "allows_single_recorded_t0_restore": False,
        "forbids_state_writes_after_action_start": True,
        "pre_action_observation_index": 0,
        "recorded_idle_actions_before_intervention": True,
        "physics_steps_after_reset_state_write": 1,
        "physics_dt": 1.0 / 120.0,
        "control_dt": 1.0 / 30.0,
        "control_substeps": 4,
        "device": "cuda:0",
        "newton": {
            "solver": "newton",
            "integrator": "implicitfast",
            "iterations": 100,
            "line_search_iterations": 50,
            "ccd_iterations": 50,
            "njmax": 512,
            "nconmax": 512,
            "use_mujoco_contacts": False,
            "use_newton_actuators": True,
            "execution": execution,
            "solver_history_reset": {
                "strategy": "solver_mujoco_internal_buffers_flags_zero",
                "flags": 0,
                "preserves_authored_joint_state": True,
                "performed_on_last_reset": True,
            },
        },
        "episode_reset_seed": 123,
    }
    (root / "replay_manifest.json").write_text(
        json.dumps(
            {
                "task": {"name": "t09_insert_peg"},
                "recording": {
                    **generation,
                    "sim_backend": "isaacsim_newton",
                    "step_count": 2,
                    "newton_cross_gpu_certification_required": True,
                    "newton_cross_gpu_certification_runs_required": 3,
                    "newton_cross_gpu_certification_mode": (
                        "fresh_process_distinct_gpu_model_v1"
                    ),
                    "publish_only_after_newton_cross_gpu_certification": True,
                    "newton_replay_profile": "cross_gpu_v1",
                    "live_direct_hand_control": False,
                    "fresh_scene_replay_required": True,
                    "fresh_certifier_process_per_run_required": True,
                    "physical_transition_contract": (
                        "seeded_fresh_newton_reset_plus_per_substep_drive_"
                        "targets_cross_gpu_v2"
                    ),
                    "trajectory_input_mode": (
                        "recorded_newton_substep_robot_joint_position_targets"
                    ),
                    "physical_replay_drive_target_source": (
                        "exact_recorded_per_newton_substep_robot_position_"
                        "targets"
                    ),
                    "randomize_dynamics": False,
                    "action_shape_per_step": [_MODULE.B03_POLICY_ACTION_DIM],
                    "action_storage_layout": _MODULE.B03_ACTION_STORAGE_LAYOUT,
                    "collection_run_seed": 122,
                    "episode_reset_seed": 123,
                    "reset_replay_contract": reset_contract,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "recording_schema.json").write_text(
        json.dumps(
            {
                "environment_schema": {
                    "name": "handson.b03_newton_ground_truth",
                    "version": 3,
                    "fields": {},
                    "reset_replay_contract": reset_contract,
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
                        "rigid_body_names": [
                            "/World/Task/industreal_round_peg_8mm"
                        ],
                        "cross_gpu_replay_profile": "cross_gpu_v1",
                        "determinism": {
                            "profile": "cross_gpu_v1",
                            "cross_gpu_requested": True,
                            "cross_gpu_ready": True,
                            "cross_gpu_method": (
                                "warp_gpu_to_gpu_plus_semantic_three_fresh_"
                                "runs_distinct_gpu_model_v2"
                            ),
                            "warp_deterministic_mode": "gpu_to_gpu",
                            "solver_deterministic_mode": "gpu_to_gpu",
                            "bitwise_cross_gpu_claim": False,
                            "collision_pipeline_deterministic": True,
                            "collision_pipeline_deterministic_required": True,
                            "collision_broad_phase": "explicit",
                        },
                        "source_runtime_identity": _runtime_identity(),
                        "physics_config": physics_config,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "collection_meta.json").write_text(
        json.dumps(
            {
                "task_name": "t09_insert_peg",
                "sim_backend": "isaacsim_newton",
                "task_outcome": outcome,
                "source_outcome": outcome,
                "step_count": metadata_steps,
                **generation,
                "newton_cross_gpu_certification_required": True,
                "newton_cross_gpu_certification_runs_required": 3,
                "newton_cross_gpu_certification_mode": (
                    "fresh_process_distinct_gpu_model_v1"
                ),
                "publish_only_after_newton_cross_gpu_certification": True,
                "newton_replay_profile": "cross_gpu_v1",
                "live_direct_hand_control": False,
            }
        ),
        encoding="utf-8",
    )
    (root / _MODULE.DATASET_MARKER_NAME).write_text(
        json.dumps(
            {
                "format": "handson.dataset_generation",
                "version": 1,
                **generation,
            }
        ),
        encoding="utf-8",
    )
    data = zarr.open_group(str(root / "replay.zarr"), mode="w").create_group(
        "data"
    )
    arrays = {
        "action": np.zeros((2, 27), dtype=np.float32),
        "succeed": np.asarray([0, 1], dtype=np.uint8),
    }
    for name in _MODULE._COMPARE_FIELDS:
        for prefix in ("", "next_"):
            arrays[f"{prefix}{name}"] = np.zeros(
                (2, 1),
                dtype=(
                    np.uint8
                    if "valid" in name
                    else np.int64
                    if name == "newton_substep_count"
                    else np.float32
                ),
            )
    for prefix in ("", "next_"):
        arrays[f"{prefix}newton_robot_joint_target"] = np.zeros(
            (2, 2, 27),
            dtype=np.float32,
        )
        arrays[f"{prefix}newton_robot_joint_target_valid"] = np.ones(
            (2, 2, 27),
            dtype=np.uint8,
        )
        arrays[f"{prefix}newton_substep_robot_joint_target"] = np.zeros(
            (2, 2, 2, 27),
            dtype=np.float32,
        )
        arrays[
            f"{prefix}newton_substep_robot_joint_target_valid"
        ] = np.ones(
            (2, 2, 2, 27),
            dtype=np.uint8,
        )
        arrays[f"{prefix}newton_substep_count"] = np.full(
            (2, 1),
            2,
            dtype=np.int64,
        )
        arrays[f"{prefix}newton_hand_peg_contact_valid"] = np.ones(
            (2, 1),
            dtype=np.uint8,
        )
        arrays[f"{prefix}newton_hand_peg_contact_count"] = np.asarray(
            [[0], [1]],
            dtype=np.int64,
        )
        arrays[f"{prefix}newton_hand_peg_min_solid_separation"] = np.zeros(
            (2, 1),
            dtype=np.float64,
        )
        arrays[f"{prefix}newton_hand_peg_max_solid_penetration"] = np.zeros(
            (2, 1),
            dtype=np.float64,
        )
        arrays[
            f"{prefix}newton_substep_hand_peg_contact_valid"
        ] = np.ones((2, 2, 1), dtype=np.uint8)
        arrays[
            f"{prefix}newton_substep_hand_peg_contact_count"
        ] = np.asarray([[[0], [0]], [[1], [1]]], dtype=np.int64)
        arrays[
            f"{prefix}newton_substep_hand_peg_min_solid_separation"
        ] = np.zeros((2, 2, 1), dtype=np.float64)
        arrays[
            f"{prefix}newton_substep_hand_peg_max_solid_penetration"
        ] = np.zeros((2, 2, 1), dtype=np.float64)
        rigid_state = np.zeros((2, 1, 13), dtype=np.float64)
        rigid_state[:, 0, 2] = (
            np.asarray([0.94, 0.95])
            if not prefix
            else np.asarray([0.95, 0.96])
        )
        arrays[f"{prefix}newton_task_rigid_body_state"] = rigid_state
        arrays[f"{prefix}newton_task_rigid_body_valid"] = np.ones(
            (2, 1),
            dtype=np.uint8,
        )
    for name, value in arrays.items():
        data.create_dataset(name, data=value)
    return root


def test_physical_replay_accepts_formal_successful_b03_newton_source(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "formal")
    manifest, schema, metadata, data = _MODULE._load_inputs(source)
    assert manifest["recording"]["sim_backend"] == "isaacsim_newton"
    assert schema["environment_schema"]["name"] == (
        "handson.b03_newton_ground_truth"
    )
    assert schema["environment_schema"]["version"] == 3
    assert schema["environment_schema"]["reset_replay_contract"][
        "physics_steps_after_reset_state_write"
    ] == 1
    assert metadata["task_outcome"] == "success"
    assert data["action"].shape[0] == 2


@pytest.mark.parametrize(
    ("outcome", "metadata_steps"),
    [("failed", 2), ("success", 1)],
)
def test_physical_replay_rejects_non_success_or_unsealed_source(
    tmp_path: Path,
    outcome: str,
    metadata_steps: int,
) -> None:
    source = _write_source(
        tmp_path / f"invalid-{outcome}-{metadata_steps}",
        outcome=outcome,
        metadata_steps=metadata_steps,
    )
    with pytest.raises(RuntimeError):
        _MODULE._load_inputs(source)


def test_source_physics_timing_requires_positive_finite_values() -> None:
    assert _MODULE._positive_physics_float(
        {"control_dt": 1.0 / 30.0},
        "control_dt",
    ) == pytest.approx(1.0 / 30.0)
    for invalid in (None, True, 0.0, -0.01, float("nan"), float("inf")):
        with pytest.raises(RuntimeError):
            _MODULE._positive_physics_float(
                {"control_dt": invalid},
                "control_dt",
            )


def test_physical_replay_rejects_missing_graph_execution_contract(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "missing-execution")
    path = source / "recording_schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    del schema["environment_schema"]["newton"]["physics_config"][
        "execution"
    ]
    path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(RuntimeError, match="execution configuration"):
        _MODULE._load_inputs(source)


def test_physical_replay_rejects_uncleared_solver_history(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "uncleared-history")
    schema_path = source / "recording_schema.json"
    manifest_path = source / "replay_manifest.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema["environment_schema"]["reset_replay_contract"]["newton"][
        "solver_history_reset"
    ]["performed_on_last_reset"] = False
    manifest["recording"]["reset_replay_contract"] = schema[
        "environment_schema"
    ]["reset_replay_contract"]
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="history was not cleared"):
        _MODULE._load_inputs(source)


def test_physical_replay_rejects_manifest_reset_contract_mismatch(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "reset-mismatch")
    path = source / "replay_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["recording"]["reset_replay_contract"][
        "physics_steps_after_reset_state_write"
    ] = 0
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="reset replay contracts differ"):
        _MODULE._load_inputs(source)


@pytest.mark.parametrize("invalid_evidence", ["no_contact", "penetration", "no_lift"])
def test_physical_replay_rejects_invalid_grasp_evidence(
    tmp_path: Path,
    invalid_evidence: str,
) -> None:
    source = _write_source(tmp_path / f"invalid-{invalid_evidence}")
    data = zarr.open_group(str(source / "replay.zarr"), mode="r+")["data"]
    if invalid_evidence == "no_contact":
        for name in (
            "newton_hand_peg_contact_count",
            "next_newton_hand_peg_contact_count",
            "newton_substep_hand_peg_contact_count",
            "next_newton_substep_hand_peg_contact_count",
        ):
            data[name][:] = 0
    elif invalid_evidence == "penetration":
        data["next_newton_substep_hand_peg_max_solid_penetration"][
            1,
            0,
            0,
        ] = 0.002
    else:
        for name in (
            "newton_task_rigid_body_state",
            "next_newton_task_rigid_body_state",
        ):
            value = np.asarray(data[name])
            value[:, 0, 2] = 0.94
            data[name][:] = value

    with pytest.raises(RuntimeError):
        _MODULE._load_inputs(source)


def test_cross_gpu_compare_is_quaternion_sign_invariant_and_semantic() -> None:
    expected = np.zeros((1, 13), dtype=np.float64)
    expected[0, 3] = 1.0
    actual = expected.copy()
    actual[0, 3:7] *= -1.0
    specs = {
        "newton_robot_root_state": {
            "replay_metric": "rigid_state_v1",
            "replay_atol": 0.0,
            "replay_tolerance": {
                "position_m": 2.0e-4,
                "quaternion_angle_rad": np.deg2rad(0.2),
                "linear_velocity_m_s": 2.0e-3,
                "angular_velocity_rad_s": 5.0e-3,
            },
        }
    }
    maxima = {}

    assert not _MODULE._compare(
        {"newton_robot_root_state": actual},
        {"newton_robot_root_state": expected},
        specs,
        maxima,
    )
    assert maxima["newton_robot_root_state.quaternion_angle_rad"] == 0.0

    actual[0, 0] = 3.0e-4
    failures = _MODULE._compare(
        {"newton_robot_root_state": actual},
        {"newton_robot_root_state": expected},
        specs,
        {},
    )
    assert any("position_m" in failure for failure in failures)


def test_cross_gpu_compare_ignores_matching_zero_substep_padding() -> None:
    expected = np.zeros((4, 2, 13), dtype=np.float64)
    actual = expected.copy()
    expected[0, :, 3] = 1.0
    actual[0, :, 3] = 1.0

    errors = _MODULE._rigid_state_errors(actual, expected)

    assert errors["quaternion_angle_rad"] == pytest.approx(0.0)


def test_cross_gpu_compare_rejects_one_sided_invalid_quaternion() -> None:
    expected = np.zeros((1, 13), dtype=np.float64)
    actual = expected.copy()
    actual[0, 3] = 1.0

    errors = _MODULE._rigid_state_errors(actual, expected)

    assert np.isinf(errors["quaternion_angle_rad"])


def test_cross_gpu_compare_rejects_a_missing_live_physical_field() -> None:
    failures = _MODULE._compare(
        {},
        {"newton_robot_joint_position": np.zeros((1,), dtype=np.float32)},
        {"newton_robot_joint_position": {"replay_atol": 2.0e-4}},
        {},
    )

    assert any("missing from live snapshot" in failure for failure in failures)


def test_report_writer_creates_the_formal_run_directory(tmp_path: Path) -> None:
    path = tmp_path / "newton_cross_gpu_certification_runs" / "run_001.json"

    _MODULE._write_report(path, {"passed": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"passed": True}
