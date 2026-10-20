#!/usr/bin/env python3
"""Physically replay one B03 Newton demo in a fresh, disposable scene.

This is deliberately a one-demo/one-process command.  It never restores
measured simulator state: after a seeded reset it writes only the exact robot
position-drive targets recorded at each Newton physics substep, compares live
readback with the source, destroys the Newton scene, and then exits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any
import uuid

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "handson"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# Kit queries platform information through a short-lived fork during startup.
# Keep BLAS single-threaded before NumPy/SciPy load so that fork cannot race an
# OpenBLAS worker shutdown in this one-demo certification process.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import scipy  # noqa: F401 -- load OpenBLAS before Kit starts worker threads
import zarr

from handson.data.newton_cross_gpu_certification import (
    B03_ACTION_STORAGE_LAYOUT,
    B03_POLICY_ACTION_DIM,
    DATASET_DIRECTORY_SUFFIX,
    DATASET_GENERATION,
    DATASET_MARKER_NAME,
    DATASET_TRAINING_STATUS,
    DRIVE_INPUT_MODE,
    REPORT_FORMAT,
    REPORT_NAME,
    REPORT_VERSION,
    canonical_physics_config_sha256,
    capture_newton_certified_inputs,
)
from handson.sim.isaaclab.newton_determinism import (
    NEWTON_CROSS_GPU_METHOD,
    NEWTON_CROSS_GPU_PROFILE,
    NEWTON_REPLAY_PROFILE_ENV,
    capture_newton_runtime_identity,
    is_distinct_gpu_model,
    newton_runtime_identity_errors,
    runtime_environment_differences,
    runtime_environment_identity,
    software_identity,
)
from handson.sim.isaaclab.newton_ground_truth import (
    B03_HAND_PEG_MAX_SOLID_PENETRATION_M,
)
from handson.sim.isaaclab.newton_replay import (
    RecordedNewtonDrivePath,
    destroy_newton_scene,
    run_newton_drive_control_step,
)
from handson.sim.isaaclab.replay_visuals import (
    reapply_reference_robot_appearance_for_replay,
)
from handson.tasks.t09_insert_peg.config import TaskConfig


_COMPARE_FIELDS = (
    "newton_robot_joint_position",
    "newton_robot_joint_velocity",
    "newton_robot_joint_target",
    "newton_robot_joint_valid",
    "newton_robot_joint_target_valid",
    "newton_robot_joint_effort",
    "newton_robot_joint_effort_valid",
    "newton_robot_root_state",
    "newton_robot_root_valid",
    "newton_tcp_state",
    "newton_tcp_valid",
    "newton_task_rigid_body_state",
    "newton_task_rigid_body_valid",
    "newton_sim_time",
    "newton_control_step",
    "newton_substep_robot_joint_position",
    "newton_substep_robot_joint_velocity",
    "newton_substep_robot_joint_target",
    "newton_substep_robot_joint_valid",
    "newton_substep_robot_joint_target_valid",
    "newton_substep_robot_joint_effort",
    "newton_substep_robot_joint_effort_valid",
    "newton_substep_sim_time",
    "newton_substep_tcp_state",
    "newton_substep_tcp_valid",
    "newton_substep_task_rigid_body_state",
    "newton_substep_task_rigid_body_valid",
    "newton_substep_count",
)
_CONTACT_FIELDS = (
    "newton_hand_peg_contact_valid",
    "newton_hand_peg_contact_count",
    "newton_hand_peg_min_solid_separation",
    "newton_hand_peg_max_solid_penetration",
    "newton_substep_hand_peg_contact_valid",
    "newton_substep_hand_peg_contact_count",
    "newton_substep_hand_peg_min_solid_separation",
    "newton_substep_hand_peg_max_solid_penetration",
)
_B03_MINIMUM_RECORDED_PEG_LIFT_M = 0.010


def _require_b03_grasp_evidence(
    schema: dict[str, Any],
    data: Any,
    step_count: int,
) -> None:
    """Reject a formally successful episode that did not grasp cleanly."""

    environment = _object(
        schema.get("environment_schema"),
        label="recording_schema.environment_schema",
    )
    newton = _object(
        environment.get("newton"),
        label="environment_schema.newton",
    )
    rigid_names = newton.get("rigid_body_names")
    if not isinstance(rigid_names, list):
        raise RuntimeError("B03 Newton schema has no rigid-body names")
    peg_rows = [
        index
        for index, raw_name in enumerate(rigid_names)
        if "industreal_round_peg_8mm" in str(raw_name).lower()
        and "industreal_tray_insert_round_peg_8mm"
        not in str(raw_name).lower()
    ]
    if len(peg_rows) != 1:
        raise RuntimeError(
            "B03 Newton schema must identify exactly one round peg rigid body"
        )

    for prefix in ("", "next_"):
        valid = np.asarray(data[f"{prefix}newton_hand_peg_contact_valid"])
        if valid.shape != (step_count, 1) or not np.all(valid == 1):
            raise RuntimeError(
                "B03 Newton source has unreadable hand/peg contact frames"
            )
    substep_count = np.asarray(data["next_newton_substep_count"]).reshape(
        step_count,
        -1,
    )[:, 0]
    substep_valid = np.asarray(
        data["next_newton_substep_hand_peg_contact_valid"]
    )
    for index, raw_count in enumerate(substep_count):
        count = int(raw_count)
        if count <= 0 or count > substep_valid.shape[1]:
            raise RuntimeError("B03 Newton source has an invalid substep count")
        if not np.all(substep_valid[index, :count] == 1):
            raise RuntimeError(
                "B03 Newton source has unreadable hand/peg substep contacts"
            )

    contact_samples = [
        np.asarray(data[name])
        for name in (
            "newton_hand_peg_contact_count",
            "next_newton_hand_peg_contact_count",
            "newton_substep_hand_peg_contact_count",
            "next_newton_substep_hand_peg_contact_count",
        )
    ]
    if max(int(np.max(value)) for value in contact_samples) <= 0:
        raise RuntimeError(
            "B03 Newton success episode has no recorded hand/peg contact"
        )
    penetration_samples = [
        np.asarray(data[name], dtype=np.float64)
        for name in (
            "newton_hand_peg_max_solid_penetration",
            "next_newton_hand_peg_max_solid_penetration",
            "newton_substep_hand_peg_max_solid_penetration",
            "next_newton_substep_hand_peg_max_solid_penetration",
        )
    ]
    max_penetration = max(float(np.max(value)) for value in penetration_samples)
    if (
        not np.isfinite(max_penetration)
        or max_penetration > B03_HAND_PEG_MAX_SOLID_PENETRATION_M
    ):
        raise RuntimeError(
            "B03 Newton source exceeds the hand/peg solid-penetration limit: "
            f"{max_penetration:.9g} m > "
            f"{B03_HAND_PEG_MAX_SOLID_PENETRATION_M:.9g} m"
        )

    peg_row = peg_rows[0]
    heights: list[float] = []
    for prefix in ("", "next_"):
        states = np.asarray(data[f"{prefix}newton_task_rigid_body_state"])
        valid = np.asarray(data[f"{prefix}newton_task_rigid_body_valid"])
        if states.ndim != 3 or states.shape[0] != step_count:
            raise RuntimeError("B03 Newton rigid-body trace has invalid shape")
        if peg_row >= states.shape[1] or peg_row >= valid.shape[1]:
            raise RuntimeError("B03 Newton peg row is outside the rigid trace")
        heights.extend(
            float(states[index, peg_row, 2])
            for index in range(step_count)
            if bool(valid[index, peg_row])
        )
    if not heights or not np.all(np.isfinite(heights)):
        raise RuntimeError("B03 Newton source has no finite peg height trace")
    initial_height = float(
        np.asarray(data["newton_task_rigid_body_state"])[0, peg_row, 2]
    )
    peg_lift = max(heights) - initial_height
    if peg_lift < _B03_MINIMUM_RECORDED_PEG_LIFT_M:
        raise RuntimeError(
            "B03 Newton success episode does not prove that the hand lifted "
            f"the peg: lift={peg_lift:.9g} m"
        )


def _object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _load_inputs(
    demo_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    manifest = _object(
        json.loads((demo_dir / "replay_manifest.json").read_text("utf-8")),
        label="replay manifest",
    )
    schema = _object(
        json.loads((demo_dir / "recording_schema.json").read_text("utf-8")),
        label="recording schema",
    )
    collection_meta = _object(
        json.loads((demo_dir / "collection_meta.json").read_text("utf-8")),
        label="collection metadata",
    )
    marker = _object(
        json.loads((demo_dir / DATASET_MARKER_NAME).read_text("utf-8")),
        label="Newton v2 dataset marker",
    )
    task = _object(manifest.get("task"), label="manifest.task")
    recording = _object(
        manifest.get("recording"),
        label="manifest.recording",
    )
    if task.get("name") != "t09_insert_peg":
        raise RuntimeError("Newton physical replay accepts only B03 demos")
    if recording.get("sim_backend") != "isaacsim_newton":
        raise RuntimeError("B03 replay source was not recorded with Newton")
    if collection_meta.get("task_name") != "t09_insert_peg":
        raise RuntimeError("Collection metadata is not a B03 recording")
    if collection_meta.get("sim_backend") != "isaacsim_newton":
        raise RuntimeError("Collection metadata is not a Newton recording")
    if (
        collection_meta.get("task_outcome") != "success"
        or collection_meta.get("source_outcome") != "success"
    ):
        raise RuntimeError(
            "Newton physical replay requires a formally saved successful "
            "B03 collection episode"
        )
    generation_contract = {
        "dataset_generation": DATASET_GENERATION,
        "training_generation_status": DATASET_TRAINING_STATUS,
        "dataset_directory_suffix": DATASET_DIRECTORY_SUFFIX,
        "dataset_marker_file": DATASET_MARKER_NAME,
        "is_newton_cross_gpu_v2_data": True,
    }
    if marker.get("format") != "handson.dataset_generation" or marker.get(
        "version"
    ) != 1:
        raise RuntimeError("B03 Newton dataset marker format/version is invalid")
    for label, payload in (
        (DATASET_MARKER_NAME, marker),
        ("manifest.recording", recording),
        ("collection_meta", collection_meta),
    ):
        for field, expected in generation_contract.items():
            if payload.get(field) != expected:
                raise RuntimeError(f"{label}.{field} must be {expected!r}")
    environment_schema = _object(
        schema.get("environment_schema"),
        label="recording_schema.environment_schema",
    )
    if environment_schema.get("name") != "handson.b03_newton_ground_truth":
        raise RuntimeError("B03 demo has no Newton physical trace")
    if environment_schema.get("version") != 3:
        raise RuntimeError(
            "B03 cross-GPU replay requires Newton ground-truth schema v3"
        )
    newton = _object(
        environment_schema.get("newton"),
        label="environment_schema.newton",
    )
    expected_newton_contract = {
        "bitwise_checkpoint": False,
        "requires_fresh_process_certification": True,
        "requires_distinct_gpu_model_certification": True,
        "allows_recorded_t0_state_restore": False,
        "certification_forbids_state_writes": True,
        "certification_forbids_state_writes_after_action_start": True,
        "replay_input": (
            "exact_recorded_per_newton_substep_robot_position_targets"
        ),
        "hand_peg_max_solid_penetration_m": (
            B03_HAND_PEG_MAX_SOLID_PENETRATION_M
        ),
    }
    for field, expected in expected_newton_contract.items():
        if newton.get(field) != expected:
            raise RuntimeError(
                f"environment_schema.newton.{field} must be {expected!r}"
            )
    determinism = _object(
        newton.get("determinism"),
        label="environment_schema.newton.determinism",
    )
    source_identity = _object(
        newton.get("source_runtime_identity"),
        label="environment_schema.newton.source_runtime_identity",
    )
    if newton.get("cross_gpu_replay_profile") != NEWTON_CROSS_GPU_PROFILE:
        raise RuntimeError("B03 source does not declare cross_gpu_v1")
    if determinism.get("profile") != NEWTON_CROSS_GPU_PROFILE:
        raise RuntimeError("B03 source was not collected with cross_gpu_v1")
    if (
        determinism.get("cross_gpu_ready") is not True
        or determinism.get("cross_gpu_requested") is not True
        or determinism.get("cross_gpu_method") != NEWTON_CROSS_GPU_METHOD
        or determinism.get("warp_deterministic_mode") != "gpu_to_gpu"
        or determinism.get("solver_deterministic_mode") != "gpu_to_gpu"
        or determinism.get("collision_pipeline_deterministic") is not True
        or determinism.get("collision_pipeline_deterministic_required") is not True
        or determinism.get("collision_broad_phase") != "explicit"
        or determinism.get("bitwise_cross_gpu_claim") is not False
    ):
        raise RuntimeError(
            "B03 source lacks the canonical semantic replay and deterministic "
            "Newton collision evidence"
        )
    if (
        source_identity.get("determinism_profile")
        != NEWTON_CROSS_GPU_PROFILE
        or source_identity.get("cross_gpu_ready") is not True
        or source_identity.get("warp_deterministic_mode") != "gpu_to_gpu"
        or source_identity.get("solver_deterministic_mode") != "gpu_to_gpu"
        or source_identity.get("collision_pipeline_deterministic") is not True
        or source_identity.get("collision_broad_phase") != "explicit"
        or source_identity.get("cross_gpu_method") != NEWTON_CROSS_GPU_METHOD
    ):
        raise RuntimeError("B03 source runtime identity is not cross-GPU ready")
    source_identity_errors = newton_runtime_identity_errors(
        source_identity,
        label="source_runtime_identity",
    )
    if source_identity_errors:
        raise RuntimeError(
            "B03 source runtime identity is incomplete: "
            + "; ".join(source_identity_errors)
        )
    required_recording = {
        "newton_cross_gpu_certification_required": True,
        "newton_cross_gpu_certification_runs_required": 3,
        "newton_cross_gpu_certification_mode": (
            "fresh_process_distinct_gpu_model_v1"
        ),
        "publish_only_after_newton_cross_gpu_certification": True,
        "newton_replay_profile": NEWTON_CROSS_GPU_PROFILE,
        "live_direct_hand_control": False,
        "fresh_scene_replay_required": True,
        "fresh_certifier_process_per_run_required": True,
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
        "action_shape_per_step": [B03_POLICY_ACTION_DIM],
        "action_storage_layout": B03_ACTION_STORAGE_LAYOUT,
    }
    for field, expected in required_recording.items():
        if recording.get(field) != expected:
            raise RuntimeError(
                f"manifest.recording.{field} must be {expected!r}"
            )
    _validated_newton_source_contract(schema, recording)
    fields = _object(
        environment_schema.get("fields"),
        label="environment_schema.fields",
    )
    physical_fields = {
        str(name)
        for name, spec in fields.items()
        if isinstance(spec, dict) and spec.get("physical_replay") is True
    }
    data = zarr.open(str(demo_dir / "replay.zarr"), mode="r")["data"]
    required = {
        "action",
        "succeed",
    }
    required.update(
        f"{prefix}{name}"
        for name in (*_COMPARE_FIELDS, *_CONTACT_FIELDS, *physical_fields)
        for prefix in ("", "next_")
    )
    missing = sorted(required - set(data.array_keys()))
    if missing:
        raise RuntimeError(f"B03 Newton demo is missing fields: {missing}")
    action_shape = tuple(int(size) for size in data["action"].shape)
    step_count = action_shape[0] if action_shape else 0
    if len(action_shape) != 2 or action_shape[1] != B03_POLICY_ACTION_DIM:
        raise RuntimeError(
            "B03 Newton right-arm action layout must be "
            f"[T, {B03_POLICY_ACTION_DIM}], got {action_shape}"
        )
    if (
        step_count <= 0
        or collection_meta.get("step_count") != step_count
        or recording.get("step_count") != step_count
    ):
        raise RuntimeError(
            "B03 Newton action count is empty or differs from collection "
            "metadata/manifest"
        )
    succeed = np.asarray(data["succeed"])
    if succeed.shape != (step_count,) or int(succeed[-1]) != 1:
        raise RuntimeError(
            "B03 Newton zarr does not end in a successful task transition"
        )
    _require_b03_grasp_evidence(schema, data, step_count)
    return manifest, schema, collection_meta, data


def _recording_int(recording: dict[str, Any], name: str) -> int:
    value = recording.get(name)
    if type(value) is not int:
        raise RuntimeError(f"manifest.recording.{name} must be an integer")
    return int(value)


def _positive_physics_float(
    physics_config: dict[str, Any],
    name: str,
) -> float:
    value = physics_config.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(
            f"source Newton physics_config.{name} must be numeric"
        )
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise RuntimeError(
            f"source Newton physics_config.{name} must be finite and positive"
        )
    return result


def _validated_newton_source_contract(
    schema: dict[str, Any],
    recording: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the graph/reset boundary before a fresh GPU scene starts."""

    environment = _object(
        schema.get("environment_schema"),
        label="recording_schema.environment_schema",
    )
    newton = _object(
        environment.get("newton"),
        label="environment_schema.newton",
    )
    physics = _object(
        newton.get("physics_config"),
        label="source Newton physics configuration",
    )
    physics_dt = _positive_physics_float(physics, "physics_dt")
    control_dt = _positive_physics_float(physics, "control_dt")
    substeps = physics.get("n_substeps")
    if type(substeps) is not int or substeps <= 0:
        raise RuntimeError(
            "source Newton physics_config.n_substeps must be a positive integer"
        )
    expected_substeps = max(1, int(round(control_dt / physics_dt)))
    if substeps != expected_substeps:
        raise RuntimeError(
            "source Newton control/physics timing is internally inconsistent: "
            f"n_substeps={substeps}, expected={expected_substeps}"
        )

    execution = _object(
        physics.get("execution"),
        label="source Newton execution configuration",
    )
    bool_fields = (
        "cuda_graph_requested",
        "use_cuda_graph",
        "cuda_graph_captured",
        "capture_completed_before_episode_reset",
    )
    for field in bool_fields:
        if type(execution.get(field)) is not bool:
            raise RuntimeError(
                f"source Newton physics_config.execution.{field} must be bool"
            )
    requested = execution["cuda_graph_requested"]
    effective = execution["use_cuda_graph"]
    captured = execution["cuda_graph_captured"]
    captured_before_reset = execution[
        "capture_completed_before_episode_reset"
    ]
    if requested != effective or effective != captured or (
        effective != captured_before_reset
    ):
        raise RuntimeError(
            "source Newton CUDA graph request, capture, effective use, and "
            "pre-reset capture evidence must agree"
        )
    capture_mode = execution.get("cuda_graph_capture_mode")
    scope = execution.get("cuda_graph_scope")
    if effective:
        if capture_mode not in {"rtx_relaxed_deferred", "standard_warp"}:
            raise RuntimeError(
                "source Newton CUDA graph capture mode is invalid: "
                f"{capture_mode!r}"
            )
        if scope not in {
            "newton_manager_simulate_full",
            "newton_manager_simulate_physics_only",
        }:
            raise RuntimeError(
                f"source Newton CUDA graph scope is invalid: {scope!r}"
            )
    elif capture_mode != "disabled" or scope != "disabled":
        raise RuntimeError(
            "disabled source Newton CUDA graph must declare disabled mode/scope"
        )
    for field in (
        "newton_manager_num_substeps",
        "newton_manager_decimation",
    ):
        if type(execution.get(field)) is not int or execution[field] != 1:
            raise RuntimeError(
                f"source Newton physics_config.execution.{field} must be 1"
            )

    reset_contract = _object(
        environment.get("reset_replay_contract"),
        label="environment_schema.reset_replay_contract",
    )
    manifest_reset = _object(
        recording.get("reset_replay_contract"),
        label="manifest.recording.reset_replay_contract",
    )
    if manifest_reset != reset_contract:
        raise RuntimeError(
            "manifest and environment schema reset replay contracts differ"
        )
    expected_reset_fields = {
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
        "control_substeps": substeps,
    }
    for field, expected in expected_reset_fields.items():
        actual = reset_contract.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise RuntimeError(
                "environment_schema.reset_replay_contract."
                f"{field} must be {expected!r}"
            )
    for field, expected in (
        ("physics_dt", physics_dt),
        ("control_dt", control_dt),
    ):
        actual = reset_contract.get(field)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or float(actual) != expected
        ):
            raise RuntimeError(
                "environment_schema.reset_replay_contract."
                f"{field} must equal physics_config.{field}"
            )
    if not str(reset_contract.get("device", "")).startswith("cuda"):
        raise RuntimeError(
            "environment_schema.reset_replay_contract.device must be CUDA"
        )
    reset_seed = reset_contract.get("episode_reset_seed")
    if (
        type(reset_seed) is not int
        or reset_seed != recording.get("episode_reset_seed")
    ):
        raise RuntimeError(
            "schema/manifest episode reset seeds must be the same integer"
        )

    reset_newton = _object(
        reset_contract.get("newton"),
        label="environment_schema.reset_replay_contract.newton",
    )
    reset_execution = _object(
        reset_newton.get("execution"),
        label="reset replay Newton execution configuration",
    )
    if reset_execution != execution:
        raise RuntimeError(
            "reset replay execution differs from Newton physics execution"
        )
    solver = _object(
        physics.get("solver"),
        label="source Newton solver configuration",
    )
    for field in ("solver", "integrator"):
        if not isinstance(solver.get(field), str) or not solver[field]:
            raise RuntimeError(
                f"source Newton physics_config.solver.{field} must be a string"
            )
    for field in (
        "iterations",
        "line_search_iterations",
        "ccd_iterations",
        "njmax",
        "nconmax",
    ):
        if type(solver.get(field)) is not int or solver[field] <= 0:
            raise RuntimeError(
                "source Newton physics_config.solver."
                f"{field} must be a positive integer"
            )
    if type(solver.get("use_mujoco_contacts")) is not bool:
        raise RuntimeError(
            "source Newton physics_config.solver.use_mujoco_contacts must be bool"
        )
    if solver.get("use_newton_actuators") is not True:
        raise RuntimeError(
            "source Newton physics_config.solver.use_newton_actuators must be true"
        )
    expected_reset_solver = {
        name: solver.get(name)
        for name in (
            "solver",
            "integrator",
            "iterations",
            "line_search_iterations",
            "ccd_iterations",
            "njmax",
            "nconmax",
            "use_mujoco_contacts",
            "use_newton_actuators",
        )
    }
    for field, expected in expected_reset_solver.items():
        actual = reset_newton.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise RuntimeError(
                f"reset replay Newton solver field {field!r} differs from "
                "physics_config.solver"
            )
    history = _object(
        reset_newton.get("solver_history_reset"),
        label="reset replay Newton solver history contract",
    )
    expected_history = {
        "strategy": "solver_mujoco_internal_buffers_flags_zero",
        "flags": 0,
        "preserves_authored_joint_state": True,
        "performed_on_last_reset": True,
    }
    if history != expected_history:
        raise RuntimeError(
            "source Newton solver history was not cleared at the reset boundary"
        )
    return physics, execution, reset_contract


def _source_snapshot(data: Any, index: int, *, after: bool) -> dict[str, np.ndarray]:
    prefix = "next_" if after else ""
    result: dict[str, np.ndarray] = {}
    for name in data.array_keys():
        if name.startswith(f"{prefix}newton_"):
            public_name = name[len(prefix) :]
            result[public_name] = np.asarray(data[name][index])
    return result


def _environment_field_specs(schema: dict[str, Any]) -> dict[str, Any]:
    environment = _object(
        schema.get("environment_schema"),
        label="recording_schema.environment_schema",
    )
    return _object(environment.get("fields"), label="environment_schema.fields")


def _maximum_vector_norm(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    if not array.size:
        return 0.0
    return float(np.max(np.linalg.norm(array.reshape(-1, array.shape[-1]), axis=1)))


def _rigid_state_errors(
    actual: np.ndarray,
    expected: np.ndarray,
) -> dict[str, float]:
    """Compare xyz+wxyz+velocity state without quaternion sign ambiguity."""

    if actual.shape[-1] != 13 or expected.shape[-1] != 13:
        raise ValueError("rigid_state_v1 requires a final dimension of 13")
    actual_rows = np.asarray(actual, dtype=np.float64).reshape(-1, 13)
    expected_rows = np.asarray(expected, dtype=np.float64).reshape(-1, 13)
    position = _maximum_vector_norm(actual_rows[:, :3] - expected_rows[:, :3])
    linear_velocity = _maximum_vector_norm(
        actual_rows[:, 7:10] - expected_rows[:, 7:10]
    )
    angular_velocity = _maximum_vector_norm(
        actual_rows[:, 10:13] - expected_rows[:, 10:13]
    )
    actual_quat = actual_rows[:, 3:7]
    expected_quat = expected_rows[:, 3:7]
    actual_norm = np.linalg.norm(actual_quat, axis=1)
    expected_norm = np.linalg.norm(expected_quat, axis=1)
    actual_valid = actual_norm > 1.0e-12
    expected_valid = expected_norm > 1.0e-12
    if np.any(actual_valid != expected_valid):
        quaternion_angle = float("inf")
    elif not np.any(actual_valid):
        # Fixed-size substep buffers are zero-filled before the first action
        # and after a short final control interval.  Matching padding rows do
        # not represent a physical quaternion and must not fail replay.
        quaternion_angle = 0.0
    else:
        dot = np.sum(
            actual_quat[actual_valid] / actual_norm[actual_valid, None]
            * (
                expected_quat[expected_valid]
                / expected_norm[expected_valid, None]
            ),
            axis=1,
        )
        # q and -q encode exactly the same orientation.
        quaternion_angle = float(
            np.max(2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))
        )
    return {
        "position_m": position,
        "quaternion_angle_rad": quaternion_angle,
        "linear_velocity_m_s": linear_velocity,
        "angular_velocity_rad_s": angular_velocity,
    }


def _compare(
    live: dict[str, Any],
    source: dict[str, np.ndarray],
    specs: dict[str, Any],
    maxima: dict[str, float],
) -> list[str]:
    failures: list[str] = []
    compare_fields = tuple(
        dict.fromkeys(
            (
                *(
                    name for name in _COMPARE_FIELDS if name in specs
                ),
                *(
                    str(name)
                    for name, spec in specs.items()
                    if isinstance(spec, dict)
                    and spec.get("physical_replay") is True
                ),
            )
        )
    )
    for name in compare_fields:
        if name not in source:
            failures.append(f"{name}: missing from source snapshot")
            continue
        if name not in live:
            failures.append(f"{name}: missing from live snapshot")
            continue
        expected = np.asarray(source[name])
        actual = np.asarray(live[name])
        if actual.shape != expected.shape:
            failures.append(
                f"{name}: shape live={actual.shape}, source={expected.shape}"
            )
            continue
        spec = specs.get(name, {})
        atol = float(spec.get("replay_atol", 0.0))
        if np.issubdtype(expected.dtype, np.integer):
            error = 0.0 if np.array_equal(actual, expected) else float("inf")
            matches = np.array_equal(actual, expected)
        elif spec.get("replay_metric") == "rigid_state_v1":
            tolerances = spec.get("replay_tolerance")
            if not isinstance(tolerances, dict):
                failures.append(f"{name}: rigid_state_v1 tolerance is missing")
                continue
            if not np.all(np.isfinite(actual)) or not np.all(
                np.isfinite(expected)
            ):
                metric_errors = {
                    str(metric): float("inf") for metric in tolerances
                }
            else:
                metric_errors = _rigid_state_errors(actual, expected)
            matches = True
            for metric, error in metric_errors.items():
                tolerance = float(tolerances.get(metric, -1.0))
                metric_name = f"{name}.{metric}"
                maxima[metric_name] = max(maxima.get(metric_name, 0.0), error)
                if error > tolerance:
                    matches = False
                    failures.append(
                        f"{metric_name}: error={error:.9g}, "
                        f"tolerance={tolerance:.9g}"
                    )
            # Metric components have already been recorded and explained.
            continue
        else:
            if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
                error = float("inf")
                matches = False
            elif expected.size:
                error = float(np.max(np.abs(actual - expected)))
                matches = error <= atol
            else:
                error = 0.0
                matches = True
        maxima[name] = max(maxima.get(name, 0.0), error)
        if not matches:
            failures.append(f"{name}: max_abs_error={error:.9g}, atol={atol:.9g}")
    return failures


def _update_replay_contact_evidence(
    snapshot: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    """Accumulate readable live Newton contacts without comparing ordering."""

    valid = np.asarray(snapshot["newton_hand_peg_contact_valid"]).reshape(-1)
    if valid.size != 1 or not np.all(valid == 1):
        raise RuntimeError("Fresh replay has unreadable hand/peg contacts")
    count = np.asarray(snapshot["newton_hand_peg_contact_count"]).reshape(-1)
    penetration = np.asarray(
        snapshot["newton_hand_peg_max_solid_penetration"],
        dtype=np.float64,
    ).reshape(-1)
    if count.size != 1 or penetration.size != 1:
        raise RuntimeError("Fresh replay contact summary has invalid shape")
    substep_count_value = np.asarray(
        snapshot["newton_substep_count"]
    ).reshape(-1)
    if substep_count_value.size != 1:
        raise RuntimeError("Fresh replay contact substep count is invalid")
    substep_count = int(substep_count_value[0])
    substep_valid = np.asarray(
        snapshot["newton_substep_hand_peg_contact_valid"]
    )
    substep_contacts = np.asarray(
        snapshot["newton_substep_hand_peg_contact_count"]
    )
    substep_penetration = np.asarray(
        snapshot["newton_substep_hand_peg_max_solid_penetration"],
        dtype=np.float64,
    )
    if (
        substep_count <= 0
        or substep_valid.ndim < 1
        or substep_count > substep_valid.shape[0]
        or substep_count > substep_contacts.shape[0]
        or substep_count > substep_penetration.shape[0]
        or not np.all(substep_valid[:substep_count] == 1)
    ):
        raise RuntimeError("Fresh replay has unreadable contact substeps")
    candidates = np.concatenate(
        [penetration, substep_penetration[:substep_count].reshape(-1)]
    )
    if not np.all(np.isfinite(candidates)) or np.any(candidates < 0.0):
        raise RuntimeError("Fresh replay produced invalid contact penetration")
    maximum = float(np.max(candidates)) if candidates.size else 0.0
    evidence["maximum_penetration_m"] = max(
        float(evidence.get("maximum_penetration_m", 0.0)),
        maximum,
    )
    evidence["contact_observed"] = bool(
        evidence.get("contact_observed", False)
        or np.any(count > 0)
        or np.any(substep_contacts[:substep_count] > 0)
    )
    evidence["buffers_readable"] = True


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def replay(
    demo_dir: Path,
    *,
    report_path: Path | None = None,
    same_gpu_smoke: bool = False,
) -> dict[str, Any]:
    manifest, schema, _collection_meta, data = _load_inputs(demo_dir)
    certified_inputs = capture_newton_certified_inputs(demo_dir)
    recording = _object(manifest["recording"], label="manifest.recording")
    source_newton = _object(
        schema["environment_schema"].get("newton"),
        label="source newton schema",
    )
    source_identity = _object(
        source_newton.get("source_runtime_identity"),
        label="source Newton runtime identity",
    )
    (
        source_physics_config,
        source_execution,
        source_reset_contract,
    ) = _validated_newton_source_contract(
        schema,
        recording,
    )
    source_physics_dt = _positive_physics_float(
        source_physics_config,
        "physics_dt",
    )
    source_control_dt = _positive_physics_float(
        source_physics_config,
        "control_dt",
    )
    source_use_cuda_graph = source_execution["use_cuda_graph"]
    source_graph_capture_mode = source_execution["cuda_graph_capture_mode"]
    replay_image_obs = source_graph_capture_mode == "rtx_relaxed_deferred"
    step_count = int(np.asarray(data["action"]).shape[0])
    collection_seed = _recording_int(recording, "collection_run_seed")
    reset_seed = _recording_int(recording, "episode_reset_seed")
    report_path = (
        demo_dir / REPORT_NAME
        if report_path is None
        else report_path.expanduser().resolve()
    )
    os.environ[NEWTON_REPLAY_PROFILE_ENV] = NEWTON_CROSS_GPU_PROFILE
    os.environ["HANDSON_RECORD_NEWTON_GROUND_TRUTH"] = "1"
    os.environ["HANDSON_ISAACLAB_DIRECT_HAND_CONTROL"] = "0"
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
    os.environ["HANDSON_NEWTON_USE_CUDA_GRAPH"] = (
        "1" if source_use_cuda_graph else "0"
    )
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env = None
    started_at = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(
        timespec="microseconds"
    )
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "task": "t09_insert_peg",
        "profile": NEWTON_CROSS_GPU_PROFILE,
        "source_demo": str(demo_dir),
        "certified_inputs": certified_inputs,
        "source_runtime_identity": source_identity,
        "run_id": uuid.uuid4().hex,
        "process_id": int(os.getpid()),
        "started_at": started_at,
        "fresh_process_per_demo": True,
        "state_writes_after_action_start": 0,
        "state_write_journal": [],
        "state_restore_used": False,
        "drive_input": DRIVE_INPUT_MODE,
        "same_gpu_smoke": bool(same_gpu_smoke),
        "certification_eligible": False,
        "source_contact_evidence_valid": True,
        "steps_requested": step_count,
        "passed": False,
    }
    failure: BaseException | None = None
    try:
        env = TaskConfig().get_native_environment(
            sim_backend="isaacsim_newton",
            policy_mode=True,
            render_mode="rgb_array",
            randomize=bool(recording.get("randomize", False)),
            randomize_dynamics=False,
            image_obs=replay_image_obs,
            seed=collection_seed,
            arm_type=str(recording.get("arm_type", "rokae")),
            hand_type=str(recording.get("hand_type", "wuji1")),
            newton_replay_profile=NEWTON_CROSS_GPU_PROFILE,
            newton_use_cuda_graph=source_use_cuda_graph,
            control_dt=source_control_dt,
            physics_dt=source_physics_dt,
        )
        observation, _info = env.reset(seed=reset_seed)
        if replay_image_obs:
            reapply_reference_robot_appearance_for_replay(env)
        raw_env = env.unwrapped
        trace = getattr(raw_env, "_physx_ground_truth_trace", None)
        if trace is None or not getattr(raw_env, "_newton_ground_truth_enabled", False):
            raise RuntimeError("Fresh B03 scene did not enable its Newton trace")
        runtime_schema = _object(
            raw_env.get_recording_schema(),
            label="runtime Newton environment schema",
        )
        runtime_newton = _object(
            runtime_schema.get("newton"),
            label="runtime newton schema",
        )
        source_newton = _object(
            schema["environment_schema"].get("newton"),
            label="source newton schema",
        )
        runtime_physics_config = _object(
            runtime_newton.get("physics_config"),
            label="runtime Newton physics configuration",
        )
        runtime_physics_config_match = (
            runtime_physics_config == source_newton.get("physics_config")
        )
        report.update(
            {
                "replay_runtime_physics_config": dict(runtime_physics_config),
                "replay_runtime_physics_config_sha256": (
                    canonical_physics_config_sha256(runtime_physics_config)
                ),
                "runtime_physics_config_exact_match": (
                    runtime_physics_config_match
                ),
            }
        )
        if not runtime_physics_config_match:
            raise RuntimeError("Fresh Newton physics configuration differs from source")
        runtime_reset_contract = _object(
            runtime_schema.get("reset_replay_contract"),
            label="runtime Newton reset replay contract",
        )
        if runtime_reset_contract != source_reset_contract:
            raise RuntimeError(
                "Fresh Newton reset boundary differs from the source contract"
            )
        replay_determinism = _object(
            runtime_newton.get("determinism"),
            label="runtime Newton determinism",
        )
        replay_identity = capture_newton_runtime_identity(
            str(getattr(raw_env, "_isaac_device", "cuda:0")),
            determinism_contract=replay_determinism,
        )
        replay_identity["b03_generated_robot_assets"] = {
            side: dict(identity)
            for side, identity in dict(
                getattr(raw_env, "_newton_b03_asset_identity", {})
            ).items()
        }
        replay_identity_errors = newton_runtime_identity_errors(
            replay_identity,
            label="replay_runtime_identity",
        )
        if replay_identity_errors:
            report["replay_runtime_identity"] = replay_identity
            raise RuntimeError(
                "Fresh replay runtime identity is incomplete: "
                + "; ".join(replay_identity_errors)
            )
        schema_runtime_identity = _object(
            runtime_newton.get("source_runtime_identity"),
            label="runtime Newton identity",
        )
        if replay_identity != schema_runtime_identity:
            raise RuntimeError(
                "Fresh runtime identity changed while the replay scene started"
            )
        cross_model = is_distinct_gpu_model(source_identity, replay_identity)
        semantic_identity_match = (
            software_identity(source_identity)
            == software_identity(replay_identity)
        )
        environment_differences = runtime_environment_differences(
            source_identity,
            replay_identity,
        )
        report.update(
            {
                "replay_runtime_identity": replay_identity,
                "replay_determinism": replay_determinism,
                "distinct_gpu_model": cross_model,
                "semantic_physics_identity_exact_match": (
                    semantic_identity_match
                ),
                "cross_gpu_environment_differences": (
                    environment_differences
                ),
            }
        )
        if not semantic_identity_match:
            raise RuntimeError(
                "Fresh replay semantic physics identity differs from the "
                "source; use the same qualified code, assets, solver patch, "
                "configuration, and immutable toolchain on both hosts"
            )
        if not cross_model and not same_gpu_smoke:
            raise RuntimeError(
                "Formal B03 certification must run on a different GPU model; "
                "use --same-gpu-smoke only for a non-publishable local check"
            )
        if (
            not cross_model
            and runtime_environment_identity(source_identity)
            != runtime_environment_identity(replay_identity)
        ):
            raise RuntimeError(
                "Same-GPU smoke requires the exact recorded host/card/driver "
                "environment; cross-host differences require a distinct-GPU "
                "semantic certification run"
            )

        path = RecordedNewtonDrivePath(schema)
        specs = _environment_field_specs(schema)
        maxima: dict[str, float] = {}
        failures = _compare(
            dict(observation),
            _source_snapshot(data, 0, after=False),
            specs,
            maxima,
        )
        if failures:
            raise RuntimeError("Fresh reset differs from source: " + "; ".join(failures))

        replay_succeeded = False
        contact_evidence: dict[str, Any] = {
            "buffers_readable": False,
            "contact_observed": False,
            "maximum_penetration_m": 0.0,
        }
        for index in range(step_count):
            after = _source_snapshot(data, index, after=True)
            drive_step = path.control_step(after)
            next_observation, _reward, _terminated, _truncated, _step_info = (
                run_newton_drive_control_step(
                    raw_env,
                    trace,
                    schema,
                    drive_step,
                )
            )
            replay_succeeded = bool(_step_info.get("succeed", False))
            failures = _compare(
                dict(next_observation),
                after,
                specs,
                maxima,
            )
            if failures:
                raise RuntimeError(
                    f"Newton replay diverged at control step {index}: "
                    + "; ".join(failures)
                )
            _update_replay_contact_evidence(
                dict(next_observation),
                contact_evidence,
            )
        if not replay_succeeded:
            raise RuntimeError(
                "Fresh Newton trajectory matched its recorded state but did "
                "not reproduce the B03 task success condition"
            )
        if not contact_evidence["buffers_readable"]:
            raise RuntimeError("Fresh Newton contact buffers were never readable")
        if not contact_evidence["contact_observed"]:
            raise RuntimeError("Fresh Newton replay never reproduced a hand/peg grasp")
        if (
            float(contact_evidence["maximum_penetration_m"])
            > B03_HAND_PEG_MAX_SOLID_PENETRATION_M
        ):
            raise RuntimeError(
                "Fresh Newton replay exceeded the hand/peg penetration bound: "
                f"{contact_evidence['maximum_penetration_m']:.9g} m"
            )
        state_write_count = int(
            getattr(raw_env, "_trajectory_state_write_count", 0)
        )
        state_write_journal = list(
            getattr(raw_env, "_trajectory_state_write_journal", ())
        )
        report["state_writes_after_action_start"] = state_write_count
        report["state_write_journal"] = state_write_journal
        if state_write_count != 0 or state_write_journal:
            raise RuntimeError(
                "Fresh Newton replay performed forbidden simulator-state writes"
            )
        report.update(
            {
                "steps_replayed": step_count,
                "task_success_reproduced": True,
                "comparison_passed": True,
                "max_error_by_metric": maxima,
                "replay_contact_buffers_readable": bool(
                    contact_evidence["buffers_readable"]
                ),
                "replay_hand_peg_contact_observed": bool(
                    contact_evidence["contact_observed"]
                ),
                "replay_max_hand_peg_penetration_m": float(
                    contact_evidence["maximum_penetration_m"]
                ),
                "certification_eligible": bool(
                    cross_model and not same_gpu_smoke
                ),
                "passed": True,
            }
        )
    except BaseException as exc:  # destroy the scene before propagating
        failure = exc
        report.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if env is not None:
            raw_env = getattr(env, "unwrapped", env)
            report["state_writes_after_action_start"] = int(
                getattr(raw_env, "_trajectory_state_write_count", 0)
            )
            report["state_write_journal"] = list(
                getattr(raw_env, "_trajectory_state_write_journal", ())
            )
            try:
                report["scene_lifecycle"] = destroy_newton_scene(env)
            except BaseException as exc:
                report["scene_lifecycle"] = {
                    "scene_destroyed": False,
                    "error": str(exc),
                }
                report["passed"] = False
                if failure is None:
                    failure = exc
        else:
            report["scene_lifecycle"] = {"scene_destroyed": True, "scene_created": False}
        current_inputs = capture_newton_certified_inputs(demo_dir)
        if current_inputs != certified_inputs:
            report["passed"] = False
            report["certified_inputs_after_replay"] = current_inputs
            mutation_error = RuntimeError(
                "B03 Newton replay inputs changed while certification ran"
            )
            if failure is None:
                failure = mutation_error
        if (
            report.get("state_writes_after_action_start") != 0
            or report.get("state_write_journal")
        ):
            report["passed"] = False
            report["certification_eligible"] = False
            if failure is None:
                failure = RuntimeError(
                    "B03 Newton replay state-write audit is non-zero"
                )
        report["completed_at"] = (
            dt.datetime.now(dt.timezone.utc)
            .astimezone()
            .isoformat(timespec="microseconds")
        )
        _write_report(report_path, report)
    if failure is not None:
        raise failure
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo_dir", type=Path)
    parser.add_argument(
        "--report-path",
        type=Path,
        help=(
            "Durable report destination. The three-run certifier supplies "
            "newton_cross_gpu_certification_runs/run_NNN.json."
        ),
    )
    parser.add_argument(
        "--same-gpu-smoke",
        action="store_true",
        help=(
            "Allow the source GPU model for debugging. The resulting report "
            "is explicitly non-certifying and can never unlock upload."
        ),
    )
    args = parser.parse_args()
    demo_dir = args.demo_dir.expanduser().resolve()
    report_path = (
        args.report_path.expanduser().resolve()
        if args.report_path is not None
        else demo_dir / REPORT_NAME
    )
    try:
        report = replay(
            demo_dir,
            report_path=report_path,
            same_gpu_smoke=bool(args.same_gpu_smoke),
        )
    except BaseException as exc:
        print(f"[B03 Newton replay] FAILED: {exc}", file=sys.stderr, flush=True)
        # The scene has already been cleared and its report made durable.
        # Avoid Kit's shutdown hook replacing this failure with exit code 0.
        os._exit(1)
    status = "SMOKE PASSED (NON-CERTIFYING)" if args.same_gpu_smoke else "PASSED"
    print(
        f"[B03 Newton replay] {status}: steps={report['steps_replayed']}, "
        f"scene_destroyed=True, report={report_path}",
        flush=True,
    )
    # One demo owns one process.  The Newton manager and SimulationContext are
    # already clear; process termination releases the remaining Kit shell.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
