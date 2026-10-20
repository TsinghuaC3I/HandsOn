"""Fail-closed B03 Newton cross-GPU replay publication contract.

The source episode is collected on one GPU model.  A qualified worker on a
different GPU model must replay the complete episode three times, one fresh
process and scene per run, using only the recorded per-substep articulation
drive targets.  This module validates and binds those reports to the immutable
replay inputs before the episode may be uploaded for training.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from handson.data.physical_replay_certification import capture_certified_inputs
from handson.sim.isaaclab.newton_determinism import (
    NEWTON_CROSS_GPU_METHOD,
    NEWTON_CROSS_GPU_PROFILE,
    is_distinct_gpu_model,
    newton_runtime_identity_errors,
    runtime_environment_differences,
    software_identity,
)
from handson.sim.isaaclab.newton_ground_truth import (
    B03_HAND_PEG_MAX_SOLID_PENETRATION_M,
)


REPORT_NAME = "newton_cross_gpu_replay.json"
REPORT_FORMAT = "handson.b03_newton_cross_gpu_replay"
REPORT_VERSION = 3
ACCEPTANCE_NAME = "newton_cross_gpu_acceptance.json"
ACCEPTANCE_FORMAT = "handson.b03_newton_cross_gpu_acceptance"
ACCEPTANCE_VERSION = 2
RUN_DIRECTORY = "newton_cross_gpu_certification_runs"
DATASET_MARKER_NAME = "DATASET_NEWTON_CROSS_GPU_V2.json"
DATASET_GENERATION = "nvidia_newton_cross_gpu_v2"
DATASET_TRAINING_STATUS = "blocked_until_cross_gpu_certified"
DATASET_DIRECTORY_SUFFIX = "__newton_v2"
REQUIRED_RUNS = 3
CERTIFICATION_MODE = "fresh_process_distinct_gpu_model_v1"
DRIVE_INPUT_MODE = "recorded_per_newton_substep_position_targets"
B03_POLICY_ACTION_DIM = 27
B03_ACTION_STORAGE_LAYOUT = "single_pose_hand"


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} cannot be read: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_physics_config_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def capture_newton_certified_inputs(demo_dir: str | Path) -> dict[str, Any]:
    """Bind the shared replay inputs plus B03's sealed outcome metadata."""

    demo_path = Path(demo_dir).expanduser().resolve()
    return {
        **capture_certified_inputs(demo_path),
        "collection_meta_sha256": _sha256_file(
            demo_path / "collection_meta.json"
        ),
        "dataset_marker_sha256": _sha256_file(
            demo_path / DATASET_MARKER_NAME
        ),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _has_b03_newton_identity_hint(
    demo_dir: Path,
    manifest: Mapping[str, Any] | None = None,
) -> bool:
    """Recognize a Newton-v2 episode even when its manifest is damaged."""

    if isinstance(manifest, Mapping):
        task = manifest.get("task")
        recording = manifest.get("recording")
        if (
            isinstance(task, Mapping)
            and isinstance(recording, Mapping)
            and task.get("name") == "t09_insert_peg"
            and recording.get("sim_backend") == "isaacsim_newton"
        ):
            return True
    try:
        collection = _read_object(
            demo_dir / "collection_meta.json",
            label="collection_meta.json",
        )
    except ValueError:
        collection = {}
    return bool(
        (
            collection.get("task_name") == "t09_insert_peg"
            and collection.get("sim_backend") == "isaacsim_newton"
        )
        or (demo_dir / DATASET_MARKER_NAME).is_file()
    )


def _source_contract(
    demo_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    manifest = _read_object(
        demo_dir / "replay_manifest.json",
        label="replay_manifest.json",
    )
    schema = _read_object(
        demo_dir / "recording_schema.json",
        label="recording_schema.json",
    )
    collection = _read_object(
        demo_dir / "collection_meta.json",
        label="collection_meta.json",
    )
    marker = _read_object(
        demo_dir / DATASET_MARKER_NAME,
        label=DATASET_MARKER_NAME,
    )
    task = _mapping(manifest.get("task"), label="manifest.task")
    recording = _mapping(
        manifest.get("recording"),
        label="manifest.recording",
    )
    if task.get("name") != "t09_insert_peg":
        raise ValueError("Newton cross-GPU certification accepts only B03")
    if recording.get("sim_backend") != "isaacsim_newton":
        raise ValueError("B03 source is not an isaacsim_newton recording")
    if (
        collection.get("task_name") != "t09_insert_peg"
        or collection.get("sim_backend") != "isaacsim_newton"
        or collection.get("task_outcome") != "success"
        or collection.get("source_outcome") != "success"
    ):
        raise ValueError("B03 collection metadata is not a sealed success")
    if collection.get("step_count") != recording.get("step_count"):
        raise ValueError("B03 collection/manifest step counts differ")
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
        raise ValueError("B03 Newton dataset marker format/version is invalid")
    for label, payload in (
        (DATASET_MARKER_NAME, marker),
        ("manifest.recording", recording),
        ("collection_meta", collection),
    ):
        for field, expected in generation_contract.items():
            if payload.get(field) != expected:
                raise ValueError(f"{label}.{field} must be {expected!r}")
    environment = _mapping(
        schema.get("environment_schema"),
        label="recording_schema.environment_schema",
    )
    if environment.get("name") != "handson.b03_newton_ground_truth":
        raise ValueError("B03 source has no Newton ground-truth schema")
    if environment.get("version") != 3:
        raise ValueError("B03 source must use Newton ground-truth schema v3")
    newton = _mapping(
        environment.get("newton"),
        label="environment_schema.newton",
    )
    _mapping(
        newton.get("physics_config"),
        label="environment_schema.newton.physics_config",
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
            raise ValueError(
                f"environment_schema.newton.{field} must be {expected!r}"
            )
    determinism = _mapping(
        newton.get("determinism"),
        label="environment_schema.newton.determinism",
    )
    source_identity = _mapping(
        newton.get("source_runtime_identity"),
        label="environment_schema.newton.source_runtime_identity",
    )
    if newton.get("cross_gpu_replay_profile") != NEWTON_CROSS_GPU_PROFILE:
        raise ValueError("B03 source does not declare cross_gpu_v1")
    if newton.get("cross_gpu_certification_runs_required") != REQUIRED_RUNS:
        raise ValueError("B03 source does not require exactly three replay runs")
    if determinism.get("profile") != NEWTON_CROSS_GPU_PROFILE:
        raise ValueError("B03 source did not run the cross_gpu_v1 profile")
    if determinism.get("cross_gpu_requested") is not True:
        raise ValueError("B03 source did not request cross-GPU replay")
    if determinism.get("cross_gpu_ready") is not True:
        raise ValueError("B03 source runtime contract was not cross-GPU ready")
    if determinism.get("cross_gpu_method") != NEWTON_CROSS_GPU_METHOD:
        raise ValueError("B03 source has a non-canonical cross-GPU method")
    if determinism.get("warp_deterministic_mode") != "gpu_to_gpu":
        raise ValueError("B03 source Warp mode differs from the canonical profile")
    if determinism.get("solver_deterministic_mode") != "gpu_to_gpu":
        raise ValueError(
            "B03 source SolverMuJoCo mode differs from the canonical profile"
        )
    if determinism.get("bitwise_cross_gpu_claim") is not False:
        raise ValueError("B03 source must not claim bitwise cross-GPU replay")
    if determinism.get("collision_pipeline_deterministic") is not True:
        raise ValueError("B03 source collision ordering was not deterministic")
    if determinism.get("collision_pipeline_deterministic_required") is not True:
        raise ValueError("B03 source did not require deterministic collisions")
    if determinism.get("collision_broad_phase") != "explicit":
        raise ValueError("B03 source did not use the explicit broad phase")
    if source_identity.get("determinism_profile") != NEWTON_CROSS_GPU_PROFILE:
        raise ValueError("B03 source identity has the wrong determinism profile")
    if source_identity.get("cross_gpu_ready") is not True:
        raise ValueError("B03 source identity is not cross-GPU ready")
    if source_identity.get("warp_deterministic_mode") != "gpu_to_gpu":
        raise ValueError("B03 source identity has a non-canonical Warp mode")
    if source_identity.get("solver_deterministic_mode") != "gpu_to_gpu":
        raise ValueError("B03 source identity has a non-canonical solver mode")
    if source_identity.get("collision_pipeline_deterministic") is not True:
        raise ValueError(
            "B03 source identity has a non-deterministic collision pipeline"
        )
    if source_identity.get("collision_broad_phase") != "explicit":
        raise ValueError("B03 source identity has a non-canonical broad phase")
    if source_identity.get("cross_gpu_method") != NEWTON_CROSS_GPU_METHOD:
        raise ValueError("B03 source identity has a non-canonical method")
    source_identity_errors = newton_runtime_identity_errors(
        source_identity,
        label="source_runtime_identity",
    )
    if source_identity_errors:
        raise ValueError(
            "B03 source runtime identity is incomplete: "
            + "; ".join(source_identity_errors)
        )
    for field in (
        "newton_replay_profile",
        "newton_cross_gpu_certification_required",
        "newton_cross_gpu_certification_runs_required",
        "newton_cross_gpu_certification_mode",
        "publish_only_after_newton_cross_gpu_certification",
        "live_direct_hand_control",
    ):
        if collection.get(field) != recording.get(field):
            raise ValueError(
                f"collection_meta.{field} differs from replay_manifest"
            )
    return manifest, schema, recording, source_identity


def _recording_requirement_errors(
    recording: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected = {
        "newton_cross_gpu_certification_required": True,
        "newton_cross_gpu_certification_runs_required": REQUIRED_RUNS,
        "newton_cross_gpu_certification_mode": CERTIFICATION_MODE,
        "publish_only_after_newton_cross_gpu_certification": True,
        "fresh_scene_replay_required": True,
        "fresh_certifier_process_per_run_required": True,
        "newton_replay_profile": NEWTON_CROSS_GPU_PROFILE,
        "live_direct_hand_control": False,
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
    for name, value in expected.items():
        if recording.get(name) != value:
            errors.append(f"manifest.recording.{name} must be {value!r}")
    if recording.get("newton_physical_replay_available") is not True:
        errors.append("manifest does not contain a replayable Newton trace")
    if recording.get("recorded_drive_target_replay_required") is not True:
        errors.append("manifest does not require recorded drive-target replay")
    return errors


def _semantic_metric_tolerances(schema: Mapping[str, Any]) -> dict[str, float]:
    """Return every physical metric that a replay report must prove."""

    environment = _mapping(
        schema.get("environment_schema"),
        label="recording_schema.environment_schema",
    )
    fields = _mapping(
        environment.get("fields"),
        label="environment_schema.fields",
    )
    result: dict[str, float] = {}
    for raw_name, raw_spec in fields.items():
        if (
            not isinstance(raw_spec, Mapping)
            or raw_spec.get("physical_replay") is not True
        ):
            continue
        name = str(raw_name)
        if raw_spec.get("replay_metric") == "rigid_state_v1":
            tolerances = _mapping(
                raw_spec.get("replay_tolerance"),
                label=f"environment_schema.fields.{name}.replay_tolerance",
            )
            for metric, raw_tolerance in tolerances.items():
                try:
                    tolerance = float(raw_tolerance)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{name}.{metric} has an invalid replay tolerance"
                    ) from exc
                if not math.isfinite(tolerance) or tolerance < 0.0:
                    raise ValueError(
                        f"{name}.{metric} has an invalid replay tolerance"
                    )
                result[f"{name}.{metric}"] = tolerance
            continue
        try:
            tolerance = float(raw_spec.get("replay_atol"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} has no valid replay_atol") from exc
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"{name} has no valid replay_atol")
        result[name] = tolerance
    if not result:
        raise ValueError("B03 Newton schema has no physical replay metrics")
    return result


def report_errors(
    demo_dir: str | Path,
    report: Mapping[str, Any],
    *,
    actual_inputs: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return every reason a single fresh-process report is non-certifying."""

    demo_path = Path(demo_dir).expanduser().resolve()
    errors: list[str] = []
    try:
        _manifest, schema, recording, source_identity = _source_contract(
            demo_path
        )
    except ValueError as exc:
        return [str(exc)]
    errors.extend(_recording_requirement_errors(recording))
    expected_inputs = (
        dict(actual_inputs)
        if actual_inputs is not None
        else capture_newton_certified_inputs(demo_path)
    )

    scalar_expectations = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "task": "t09_insert_peg",
        "profile": NEWTON_CROSS_GPU_PROFILE,
        "drive_input": DRIVE_INPUT_MODE,
        "fresh_process_per_demo": True,
        "state_restore_used": False,
        "same_gpu_smoke": False,
        "certification_eligible": True,
        "comparison_passed": True,
        "task_success_reproduced": True,
        "source_contact_evidence_valid": True,
        "replay_contact_buffers_readable": True,
        "replay_hand_peg_contact_observed": True,
        "semantic_physics_identity_exact_match": True,
        "runtime_physics_config_exact_match": True,
        "passed": True,
    }
    for name, value in scalar_expectations.items():
        if report.get(name) != value:
            errors.append(f"report.{name} must be {value!r}")
    if report.get("certified_inputs") != expected_inputs:
        errors.append("report certified-input hashes are stale")
    if report.get("source_runtime_identity") != source_identity:
        errors.append("report source runtime identity differs from the schema")

    environment = _mapping(
        schema.get("environment_schema"),
        label="recording_schema.environment_schema",
    )
    source_newton = _mapping(
        environment.get("newton"),
        label="environment_schema.newton",
    )
    source_physics_config = _mapping(
        source_newton.get("physics_config"),
        label="environment_schema.newton.physics_config",
    )
    replay_physics_config_value = report.get("replay_runtime_physics_config")
    if not isinstance(replay_physics_config_value, Mapping):
        errors.append("report replay runtime physics config is missing")
    else:
        replay_physics_config = replay_physics_config_value
        if replay_physics_config != source_physics_config:
            errors.append("replay runtime physics config differs from source")
        expected_physics_hash = canonical_physics_config_sha256(
            replay_physics_config
        )
        if (
            report.get("replay_runtime_physics_config_sha256")
            != expected_physics_hash
        ):
            errors.append("report replay runtime physics config hash is stale")
        if expected_physics_hash != canonical_physics_config_sha256(
            source_physics_config
        ):
            errors.append("replay runtime physics config hash differs from source")

    replay_identity_value = report.get("replay_runtime_identity")
    if not isinstance(replay_identity_value, Mapping):
        errors.append("report replay runtime identity is missing")
        replay_identity: Mapping[str, Any] = {}
    else:
        replay_identity = replay_identity_value
        errors.extend(
            newton_runtime_identity_errors(
                replay_identity,
                label="replay_runtime_identity",
            )
        )
    if not is_distinct_gpu_model(source_identity, replay_identity):
        errors.append("source and replay must use distinct GPU models")
    expected_environment_differences = runtime_environment_differences(
        source_identity,
        replay_identity,
    )
    if (
        report.get("cross_gpu_environment_differences")
        != expected_environment_differences
    ):
        errors.append("report cross-GPU environment differences are stale")
    if software_identity(source_identity) != software_identity(replay_identity):
        errors.append("source and replay semantic physics identities differ")
    if replay_identity.get("determinism_profile") != NEWTON_CROSS_GPU_PROFILE:
        errors.append("replay identity has the wrong determinism profile")
    if replay_identity.get("cross_gpu_ready") is not True:
        errors.append("replay identity is not cross-GPU ready")
    if replay_identity.get("warp_deterministic_mode") != "gpu_to_gpu":
        errors.append("replay used a non-canonical Warp atomic mode")
    if replay_identity.get("solver_deterministic_mode") != "gpu_to_gpu":
        errors.append("replay used a non-canonical SolverMuJoCo mode")
    if replay_identity.get("collision_pipeline_deterministic") is not True:
        errors.append("replay identity has a non-deterministic collision pipeline")
    if replay_identity.get("collision_broad_phase") != "explicit":
        errors.append("replay identity has a non-canonical broad phase")
    if replay_identity.get("cross_gpu_method") != NEWTON_CROSS_GPU_METHOD:
        errors.append("replay identity has a non-canonical cross-GPU method")

    replay_determinism = report.get("replay_determinism")
    if not isinstance(replay_determinism, Mapping):
        errors.append("report replay determinism contract is missing")
    else:
        if replay_determinism.get("profile") != NEWTON_CROSS_GPU_PROFILE:
            errors.append("replay determinism contract has the wrong profile")
        if replay_determinism.get("cross_gpu_requested") is not True:
            errors.append("replay did not request cross-GPU execution")
        if (
            replay_determinism.get("cross_gpu_method")
            != NEWTON_CROSS_GPU_METHOD
        ):
            errors.append("replay cross-GPU method is non-canonical")
        if replay_determinism.get("collision_pipeline_deterministic") is not True:
            errors.append("replay collision ordering was not deterministic")
        if replay_determinism.get("collision_broad_phase") != "explicit":
            errors.append("replay did not use the explicit broad phase")
        if replay_determinism.get("cross_gpu_ready") is not True:
            errors.append("replay determinism contract is not cross-GPU ready")
        if replay_determinism.get("warp_deterministic_mode") != "gpu_to_gpu":
            errors.append("replay Warp atomic mode is non-canonical")
        if replay_determinism.get("solver_deterministic_mode") != "gpu_to_gpu":
            errors.append("replay SolverMuJoCo mode is non-canonical")
        if replay_determinism.get("bitwise_cross_gpu_claim") is not False:
            errors.append("replay must not claim bitwise cross-GPU equivalence")

    step_count = recording.get("step_count")
    if type(step_count) is not int or step_count <= 0:
        errors.append("manifest recording step count is invalid")
    else:
        if report.get("steps_requested") != step_count:
            errors.append("report requested step count differs from source")
        if report.get("steps_replayed") != step_count:
            errors.append("report did not replay the complete trajectory")

    if report.get("state_writes_after_action_start") != 0:
        errors.append("replay performed simulator-state writes after action 0")
    journal = report.get("state_write_journal")
    if journal not in (None, []):
        errors.append("replay state-write journal is not empty")
    lifecycle = report.get("scene_lifecycle")
    if not isinstance(lifecycle, Mapping) or lifecycle.get("scene_destroyed") is not True:
        errors.append("replay scene was not destroyed")

    max_penetration = report.get("replay_max_hand_peg_penetration_m")
    if (
        type(max_penetration) not in {int, float}
        or isinstance(max_penetration, bool)
        or not math.isfinite(float(max_penetration))
        or float(max_penetration) > B03_HAND_PEG_MAX_SOLID_PENETRATION_M
        or float(max_penetration) < 0.0
    ):
        errors.append("replay hand/peg penetration is invalid or unsafe")
    metrics = report.get("max_error_by_metric")
    if not isinstance(metrics, Mapping):
        errors.append("report has no semantic comparison metrics")
    else:
        try:
            expected_metrics = _semantic_metric_tolerances(schema)
        except ValueError as exc:
            errors.append(str(exc))
            expected_metrics = {}
        for metric, tolerance in expected_metrics.items():
            value = metrics.get(metric)
            if (
                type(value) not in {int, float}
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                or float(value) > tolerance
            ):
                errors.append(
                    f"report metric {metric} is missing or exceeds "
                    f"{tolerance:.9g}"
                )
    if not isinstance(report.get("run_id"), str) or not report.get("run_id"):
        errors.append("report has no unique run_id")
    if type(report.get("process_id")) is not int or report.get("process_id") <= 0:
        errors.append("report has no valid process_id")
    return errors


def build_acceptance(
    demo_dir: str | Path,
    report_paths: Sequence[str | Path],
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Validate three archived reports and optionally seal acceptance JSON."""

    demo_path = Path(demo_dir).expanduser().resolve()
    paths = [Path(path).expanduser().resolve() for path in report_paths]
    if len(paths) != REQUIRED_RUNS:
        raise ValueError(f"exactly {REQUIRED_RUNS} replay reports are required")
    actual_inputs = capture_newton_certified_inputs(demo_path)
    rows: list[dict[str, Any]] = []
    run_ids: list[str] = []
    process_ids: list[int] = []
    replay_identities: list[Mapping[str, Any]] = []
    replay_physics_configs: list[Mapping[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        try:
            relative = path.relative_to(demo_path).as_posix()
        except ValueError as exc:
            raise ValueError(f"report {path} is outside the demo directory") from exc
        expected_relative = f"{RUN_DIRECTORY}/run_{index:03d}.json"
        if relative != expected_relative:
            raise ValueError(
                f"report {index} must be archived as {expected_relative}"
            )
        report = _read_object(path, label=relative)
        errors = report_errors(
            demo_path,
            report,
            actual_inputs=actual_inputs,
        )
        if errors:
            raise ValueError(f"{relative} is invalid: " + "; ".join(errors))
        run_ids.append(str(report["run_id"]))
        process_ids.append(int(report["process_id"]))
        replay_identities.append(
            _mapping(
                report["replay_runtime_identity"],
                label=f"{relative}.replay_runtime_identity",
            )
        )
        replay_physics_configs.append(
            _mapping(
                report["replay_runtime_physics_config"],
                label=f"{relative}.replay_runtime_physics_config",
            )
        )
        rows.append(
            {
                "run": index,
                "path": relative,
                "sha256": _sha256_file(path),
                "run_id": report["run_id"],
                "process_id": report["process_id"],
                "started_at": report.get("started_at"),
                "completed_at": report.get("completed_at"),
            }
        )
    if len(set(run_ids)) != REQUIRED_RUNS:
        raise ValueError("the three reports must have unique run_id values")
    # PID reuse is possible across long periods, so run_id is authoritative;
    # still require the normal sequential-worker case to prove separate live
    # processes rather than three copies produced by one invocation.
    if len(set(process_ids)) != REQUIRED_RUNS:
        raise ValueError("the three reports must come from distinct process IDs")
    first_replay_identity = replay_identities[0]
    if any(identity != first_replay_identity for identity in replay_identities[1:]):
        raise ValueError("the three replay runtime identities must be identical")
    first_replay_physics_config = replay_physics_configs[0]
    if any(
        config != first_replay_physics_config
        for config in replay_physics_configs[1:]
    ):
        raise ValueError("the three replay runtime physics configs must be identical")

    _manifest, source_schema, _recording, source_identity = _source_contract(
        demo_path
    )
    source_environment = _mapping(
        source_schema.get("environment_schema"),
        label="recording_schema.environment_schema",
    )
    source_newton = _mapping(
        source_environment.get("newton"),
        label="environment_schema.newton",
    )
    source_physics_config = _mapping(
        source_newton.get("physics_config"),
        label="environment_schema.newton.physics_config",
    )
    source_physics_hash = canonical_physics_config_sha256(source_physics_config)
    acceptance = {
        "format": ACCEPTANCE_FORMAT,
        "version": ACCEPTANCE_VERSION,
        "status": "accepted",
        "task": "t09_insert_peg",
        "mode": CERTIFICATION_MODE,
        "profile": NEWTON_CROSS_GPU_PROFILE,
        "required_runs": REQUIRED_RUNS,
        "completed_runs": REQUIRED_RUNS,
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "certified_inputs": actual_inputs,
        "source_runtime_identity": dict(source_identity),
        "replay_runtime_identity": dict(first_replay_identity),
        "distinct_gpu_model": True,
        "software_identity_exact_match": True,
        "semantic_physics_identity_exact_match": True,
        "runtime_physics_config_exact_match": True,
        "source_physics_config_sha256": source_physics_hash,
        "replay_runtime_physics_config_sha256": canonical_physics_config_sha256(
            first_replay_physics_config
        ),
        "cross_gpu_environment_differences": (
            runtime_environment_differences(
                source_identity,
                first_replay_identity,
            )
        ),
        "all_runs_fresh_process_and_scene": True,
        "all_runs_complete_and_successful": True,
        "all_runs_zero_trajectory_state_writes": True,
        "certification_runs": rows,
    }
    if write:
        _write_json_atomic(demo_path / ACCEPTANCE_NAME, acceptance)
    return acceptance


def upload_blocker(demo_dir: str | Path) -> dict[str, Any] | None:
    """Block B03/Newton upload unless cross-model acceptance is intact."""

    demo_path = Path(demo_dir).expanduser().resolve()
    try:
        manifest = _read_object(
            demo_path / "replay_manifest.json",
            label="replay_manifest.json",
        )
    except ValueError as exc:
        if _has_b03_newton_identity_hint(demo_path):
            return {
                "code": "newton_cross_gpu_source_invalid",
                "detail": f"B03 Newton replay manifest is invalid: {exc}",
            }
        return None
    task = manifest.get("task")
    recording = manifest.get("recording")
    if not isinstance(task, Mapping) or not isinstance(recording, Mapping):
        if _has_b03_newton_identity_hint(demo_path, manifest):
            return {
                "code": "newton_cross_gpu_source_invalid",
                "detail": "B03 Newton replay manifest has no task/recording contract",
            }
        return None
    if not (
        task.get("name") == "t09_insert_peg"
        and recording.get("sim_backend") == "isaacsim_newton"
    ):
        if _has_b03_newton_identity_hint(demo_path, manifest):
            return {
                "code": "newton_cross_gpu_source_invalid",
                "detail": (
                    "B03 Newton collection metadata and replay manifest disagree"
                ),
            }
        return None
    try:
        _manifest, _schema, source_recording, _source_identity = _source_contract(
            demo_path
        )
    except ValueError as exc:
        return {
            "code": "newton_cross_gpu_source_invalid",
            "detail": f"B03 Newton source is not cross-GPU certifiable: {exc}",
        }
    requirement_errors = _recording_requirement_errors(source_recording)
    if requirement_errors:
        return {
            "code": "newton_cross_gpu_requirement_missing",
            "detail": "B03 Newton publication contract is incomplete: "
            + "; ".join(requirement_errors),
        }
    acceptance_path = demo_path / ACCEPTANCE_NAME
    try:
        acceptance = _read_object(
            acceptance_path,
            label=ACCEPTANCE_NAME,
        )
    except ValueError as exc:
        return {
            "code": "newton_cross_gpu_certification_missing",
            "detail": (
                "B03 Newton requires three fresh-process replays on a distinct "
                f"GPU model before upload: {exc}"
            ),
        }
    try:
        actual_inputs = capture_newton_certified_inputs(demo_path)
    except (OSError, UnicodeError, OverflowError) as exc:
        return {
            "code": "newton_cross_gpu_certification_stale",
            "detail": f"B03 Newton certified inputs cannot be verified: {exc}",
        }
    errors: list[str] = []
    expected_acceptance = {
        "format": ACCEPTANCE_FORMAT,
        "version": ACCEPTANCE_VERSION,
        "status": "accepted",
        "task": "t09_insert_peg",
        "mode": CERTIFICATION_MODE,
        "profile": NEWTON_CROSS_GPU_PROFILE,
        "required_runs": REQUIRED_RUNS,
        "completed_runs": REQUIRED_RUNS,
        "distinct_gpu_model": True,
        "software_identity_exact_match": True,
        "semantic_physics_identity_exact_match": True,
        "runtime_physics_config_exact_match": True,
        "all_runs_fresh_process_and_scene": True,
        "all_runs_complete_and_successful": True,
        "all_runs_zero_trajectory_state_writes": True,
    }
    for name, value in expected_acceptance.items():
        if acceptance.get(name) != value:
            errors.append(f"acceptance.{name} must be {value!r}")
    if acceptance.get("certified_inputs") != actual_inputs:
        errors.append("acceptance certified-input hashes are stale")
    rows = acceptance.get("certification_runs")
    if not isinstance(rows, list) or len(rows) != REQUIRED_RUNS:
        errors.append("acceptance must bind exactly three reports")
        rows = []
    run_ids: list[str] = []
    process_ids: list[int] = []
    report_replay_identities: list[Mapping[str, Any]] = []
    report_replay_physics_configs: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            errors.append(f"acceptance run {index} is not an object")
            continue
        expected_relative = f"{RUN_DIRECTORY}/run_{index:03d}.json"
        if row.get("run") != index or row.get("path") != expected_relative:
            errors.append(f"acceptance run {index} path/order is invalid")
            continue
        report_path = demo_path / expected_relative
        if not report_path.is_file():
            errors.append(f"acceptance run {index} report is missing")
            continue
        if row.get("sha256") != _sha256_file(report_path):
            errors.append(f"acceptance run {index} report hash differs")
            continue
        try:
            report = _read_object(report_path, label=expected_relative)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        report_problems = report_errors(
            demo_path,
            report,
            actual_inputs=actual_inputs,
        )
        errors.extend(f"run {index}: {problem}" for problem in report_problems)
        if row.get("run_id") != report.get("run_id"):
            errors.append(f"acceptance run {index} run_id differs")
        if row.get("process_id") != report.get("process_id"):
            errors.append(f"acceptance run {index} process_id differs")
        if isinstance(report.get("run_id"), str):
            run_ids.append(report["run_id"])
        if type(report.get("process_id")) is int:
            process_ids.append(report["process_id"])
        if isinstance(report.get("replay_runtime_identity"), Mapping):
            report_replay_identities.append(report["replay_runtime_identity"])
        if isinstance(report.get("replay_runtime_physics_config"), Mapping):
            report_replay_physics_configs.append(
                report["replay_runtime_physics_config"]
            )
    if len(set(run_ids)) != REQUIRED_RUNS:
        errors.append("accepted report run IDs are not unique")
    if len(set(process_ids)) != REQUIRED_RUNS:
        errors.append("accepted report process IDs are not unique")
    try:
        _manifest, _schema, _recording, source_identity = _source_contract(
            demo_path
        )
    except ValueError as exc:
        errors.append(str(exc))
        source_identity = {}
    if acceptance.get("source_runtime_identity") != source_identity:
        errors.append("acceptance source runtime identity differs")
    source_environment = _mapping(
        _schema.get("environment_schema"),
        label="recording_schema.environment_schema",
    )
    source_newton = _mapping(
        source_environment.get("newton"),
        label="environment_schema.newton",
    )
    source_physics_config = _mapping(
        source_newton.get("physics_config"),
        label="environment_schema.newton.physics_config",
    )
    source_physics_hash = canonical_physics_config_sha256(source_physics_config)
    if acceptance.get("source_physics_config_sha256") != source_physics_hash:
        errors.append("acceptance source physics config hash differs")
    if len(report_replay_identities) == REQUIRED_RUNS:
        first_replay_identity = report_replay_identities[0]
        if any(
            identity != first_replay_identity
            for identity in report_replay_identities[1:]
        ):
            errors.append("accepted replay runtime identities differ")
        if acceptance.get("replay_runtime_identity") != first_replay_identity:
            errors.append("acceptance replay runtime identity differs")
        expected_environment_differences = runtime_environment_differences(
            source_identity,
            first_replay_identity,
        )
        if (
            acceptance.get("cross_gpu_environment_differences")
            != expected_environment_differences
        ):
            errors.append("acceptance cross-GPU environment differences differ")
    else:
        errors.append("accepted replay runtime identities are incomplete")
    if len(report_replay_physics_configs) == REQUIRED_RUNS:
        first_replay_physics_config = report_replay_physics_configs[0]
        if any(
            config != first_replay_physics_config
            for config in report_replay_physics_configs[1:]
        ):
            errors.append("accepted replay runtime physics configs differ")
        replay_physics_hash = canonical_physics_config_sha256(
            first_replay_physics_config
        )
        if (
            acceptance.get("replay_runtime_physics_config_sha256")
            != replay_physics_hash
        ):
            errors.append("acceptance replay runtime physics config hash differs")
        if replay_physics_hash != source_physics_hash:
            errors.append("accepted replay physics config differs from source")
    else:
        errors.append("accepted replay runtime physics configs are incomplete")
    if errors:
        return {
            "code": "newton_cross_gpu_certification_invalid",
            "detail": "B03 Newton cross-GPU acceptance is invalid: "
            + "; ".join(errors),
        }
    return None


__all__ = [
    "ACCEPTANCE_FORMAT",
    "ACCEPTANCE_NAME",
    "ACCEPTANCE_VERSION",
    "B03_ACTION_STORAGE_LAYOUT",
    "B03_POLICY_ACTION_DIM",
    "CERTIFICATION_MODE",
    "DATASET_DIRECTORY_SUFFIX",
    "DATASET_GENERATION",
    "DATASET_MARKER_NAME",
    "DATASET_TRAINING_STATUS",
    "DRIVE_INPUT_MODE",
    "REPORT_FORMAT",
    "REPORT_NAME",
    "REPORT_VERSION",
    "REQUIRED_RUNS",
    "RUN_DIRECTORY",
    "build_acceptance",
    "canonical_physics_config_sha256",
    "capture_newton_certified_inputs",
    "report_errors",
    "upload_blocker",
]
