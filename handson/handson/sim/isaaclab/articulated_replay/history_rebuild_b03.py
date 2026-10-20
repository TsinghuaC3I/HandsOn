#!/usr/bin/env python3
"""Replay consecutive recorded B03 demos in one PhysX scene and audit all steps.

This is deliberately a narrow experiment for the 2026-08-01 B03 recording.
It does not write a public-state checkpoint into PhysX.  Instead it repeats
the source reset/stabilization and drive-target history so PhysX itself can
recreate as much private contact/solver history as the published recordings
permit.  Public solver readback and contact reports are compared at every
control step; no mismatch is corrected and no snapshot is restored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


# Make the outer ``handson/`` directory importable when this file is executed
# directly rather than with ``python -m``.
_PACKAGE_ROOT = Path(__file__).resolve().parents[4]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


_DRIVE_FIELDS = (
    "next_physx_substep_robot_joint_target",
    "next_physx_substep_robot_joint_target_valid",
    "next_physx_substep_count",
    "next_physx_robot_joint_target",
    "next_physx_robot_joint_target_valid",
)

_CORE_PHYSICAL_FIELDS = frozenset(
    {
        "physx_robot_joint_position",
        "physx_robot_joint_velocity",
        "physx_robot_joint_effort",
        "physx_robot_joint_target",
        "physx_robot_joint_valid",
        "physx_robot_joint_effort_valid",
        "physx_robot_joint_target_valid",
        "physx_robot_root_state",
        "physx_robot_root_valid",
        "physx_tcp_state",
        "physx_tcp_valid",
        "physx_task_rigid_body_state",
        "physx_task_rigid_body_valid",
        "physx_task_articulation_root_state",
        "physx_task_articulation_root_valid",
        "physx_task_articulation_dof_position",
        "physx_task_articulation_dof_velocity",
        "physx_task_articulation_dof_valid",
        "physx_deformable_position",
        "physx_deformable_velocity",
        "physx_deformable_valid",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _manifest_recording(demo_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(demo_dir / "replay_manifest.json")
    recording = manifest.get("recording")
    if not isinstance(recording, dict):
        raise ValueError(f"{demo_dir} has no manifest recording object")
    return manifest, recording


def _apply_recorded_environment(manifest: Mapping[str, Any]) -> None:
    environment = manifest.get("environment")
    variables = environment.get("variables") if isinstance(environment, Mapping) else None
    if isinstance(variables, Mapping):
        for raw_name, value in variables.items():
            name = str(raw_name)
            if name.startswith("HANDSON_") and value is not None:
                os.environ[name] = str(value)
    # The trace is required both for comparisons and for named target writes.
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"
    os.environ["HANDSON_ARM_TYPE"] = "rokae"


def _replay_path(demo_dir: Path) -> Path:
    from handson.runtime.replay_archive import resolve_replay_path

    return resolve_replay_path(demo_dir)


def _load_source(
    demo_dir: Path,
    *,
    load_drive_path: bool,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Load schema, frame 0, drives, audit readback, and task outcomes."""

    from handson.runtime.replay_archive import open_replay_group
    from handson.sim.isaaclab.articulated_replay.trajectory_audit import (
        audit_field_names,
    )

    schema = _read_json(demo_dir / "recording_schema.json")
    audit_names = set(audit_field_names(schema))
    initial: dict[str, np.ndarray] = {}
    drives: dict[str, np.ndarray] = {}
    next_audit: dict[str, np.ndarray] = {}
    outcomes: dict[str, np.ndarray] = {}
    with open_replay_group(_replay_path(demo_dir)) as root:
        data = root["data"]
        for name in data.array_keys():
            if name in audit_names:
                initial[name] = np.asarray(data[name][0])
            if (
                load_drive_path
                and name.startswith("next_physx_")
                and name[len("next_") :] in audit_names
            ):
                next_audit[name[len("next_") :]] = np.asarray(data[name])
        if load_drive_path:
            missing = [name for name in _DRIVE_FIELDS if name not in data]
            if missing:
                raise ValueError(f"{demo_dir} is missing drive fields: {missing}")
            for name in _DRIVE_FIELDS:
                drives[name] = np.asarray(data[name])
            for name in ("succeed", "terminal"):
                if name not in data:
                    raise ValueError(f"{demo_dir} is missing task outcome field {name}")
                outcomes[name] = np.asarray(data[name])
    missing_initial = sorted(audit_names.difference(initial))
    if missing_initial:
        raise ValueError(f"{demo_dir} is missing frame-0 audit fields: {missing_initial}")
    if load_drive_path:
        missing_next = sorted(audit_names.difference(next_audit))
        if missing_next:
            raise ValueError(f"{demo_dir} is missing next audit fields: {missing_next}")
        step_count = int(drives[_DRIVE_FIELDS[0]].shape[0])
        bad_lengths = {
            name: int(values.shape[0])
            for name, values in {**drives, **next_audit, **outcomes}.items()
            if int(values.shape[0]) != step_count
        }
        if bad_lengths:
            raise ValueError(
                f"{demo_dir} has fields inconsistent with {step_count} steps: "
                f"{bad_lengths}"
            )
    return schema, initial, drives, next_audit, outcomes


def _scalar_int(recording: Mapping[str, Any], name: str) -> int:
    value = recording.get(name)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"recording.{name} must be an integer")
    return int(value)


def _equal_with_nan(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    try:
        return np.equal(expected, actual) | (np.isnan(expected) & np.isnan(actual))
    except TypeError:
        return np.equal(expected, actual)


def _field_comparison(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    result: dict[str, Any] = {
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "expected_dtype": str(expected.dtype),
        "actual_dtype": str(actual.dtype),
    }
    if expected.shape != actual.shape:
        result.update({"exact": False, "max_abs_error": None, "reason": "shape"})
        return result
    equal = _equal_with_nan(expected, actual)
    result["exact"] = bool(np.all(equal))
    if np.issubdtype(expected.dtype, np.number) and np.issubdtype(actual.dtype, np.number):
        lhs = expected.astype(np.float64, copy=False)
        rhs = actual.astype(np.float64, copy=False)
        finite = np.isfinite(lhs) & np.isfinite(rhs)
        if np.any(~equal & ~finite):
            max_error = float("inf")
        elif np.any(finite):
            max_error = float(np.max(np.abs(lhs[finite] - rhs[finite])))
        else:
            max_error = 0.0
        result["max_abs_error"] = max_error if np.isfinite(max_error) else None
        if not np.isfinite(max_error):
            result["nonfinite_mismatch"] = True
    else:
        result["max_abs_error"] = None
    return result


def _compare_snapshot(
    expected: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    per_field: dict[str, Any] = {}
    for name in sorted(expected):
        if name not in actual:
            per_field[name] = {
                "exact": False,
                "max_abs_error": None,
                "reason": "missing_runtime_field",
            }
            continue
        per_field[name] = _field_comparison(expected[name], actual[name])

    exact_fields = [name for name, item in per_field.items() if item["exact"]]
    failed_fields = [name for name, item in per_field.items() if not item["exact"]]
    finite_errors = [
        float(item["max_abs_error"])
        for item in per_field.values()
        if item.get("max_abs_error") is not None
    ]
    core_checked = sorted(name for name in per_field if name in _CORE_PHYSICAL_FIELDS)
    core_failed = [name for name in core_checked if not per_field[name]["exact"]]
    core_errors = [
        float(per_field[name]["max_abs_error"])
        for name in core_checked
        if per_field[name].get("max_abs_error") is not None
    ]
    return {
        "all_recorded_public_fields_exact": not failed_fields,
        "core_physical_fields_exact": bool(core_checked) and not core_failed,
        "checked_field_count": len(per_field),
        "exact_field_count": len(exact_fields),
        "failed_fields": failed_fields,
        "core_checked_fields": core_checked,
        "core_failed_fields": core_failed,
        "max_abs_error": max(finite_errors, default=0.0),
        "core_max_abs_error": max(core_errors, default=0.0),
        "per_field": per_field,
    }


def _reset_and_stabilize(env: Any, seed: int, stable_report: Mapping[str, Any]) -> dict[str, Any]:
    from handson.sim.isaaclab.recording_origin import (
        RecordingOriginStabilityError,
        stable_recording_origin_kwargs_from_report,
        stabilize_recording_origin,
    )

    started = time.monotonic()
    observation, info = env.reset(seed=int(seed))
    try:
        _observation, stable_info = stabilize_recording_origin(
            env,
            observation,
            info,
            **stable_recording_origin_kwargs_from_report(stable_report),
        )
    except RecordingOriginStabilityError as exc:
        return {
            "seed": int(seed),
            "stabilized": False,
            "error": str(exc),
            "wall_seconds": time.monotonic() - started,
        }
    origin = stable_info.get("physical_recording_origin", {})
    return {
        "seed": int(seed),
        "stabilized": True,
        "physics_steps": int(origin.get("physics_steps", 0)),
        "final_max_linear_velocity": float(origin.get("final_max_linear_velocity", 0.0)),
        "final_max_angular_velocity": float(origin.get("final_max_angular_velocity", 0.0)),
        "wall_seconds": time.monotonic() - started,
    }


def _physics_step_from_time(value: Any, physics_dt: float) -> tuple[int, float]:
    array = np.asarray(value).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]):
        raise ValueError(f"simulation time is not one finite scalar: {array}")
    raw_step = float(array[0]) / physics_dt
    step = int(round(raw_step))
    return step, float(array[0]) - step * physics_dt


def _failed_stabilization_physics_steps(stable_report: Mapping[str, Any]) -> int:
    physics_dt = float(stable_report["physics_dt"])
    initial_steps = int(stable_report["initial_settle_steps"])
    timeout_steps = max(
        1,
        int(np.ceil(float(stable_report["timeout_seconds"]) / physics_dt)),
    )
    return initial_steps + timeout_steps


def _run_inferred_unpublished_prefix(
    raw_env: Any,
    trace: Any,
    source_demo1_initial: Mapping[str, np.ndarray],
    stable_report: Mapping[str, Any],
    demo1_attempts: list[Mapping[str, Any]],
    *,
    n_substeps: int,
    progress_every: int,
) -> dict[str, Any]:
    """Fill the unpublished prefix duration with a no-command-change hold.

    The source clock reveals the duration, but not the missing teleoperation
    inputs.  Holding the reset drive targets is therefore only a declared
    approximation and never upgrades the exact-history certificate.
    """

    started = time.monotonic()
    physics_dt = float(stable_report["physics_dt"])
    if n_substeps <= 0:
        raise ValueError("recorded n_substeps must be positive")
    source_step, source_rounding_residual = _physics_step_from_time(
        source_demo1_initial["physx_sim_time"],
        physics_dt,
    )
    current_snapshot = trace.snapshot()
    current_step, current_rounding_residual = _physics_step_from_time(
        current_snapshot["physx_sim_time"],
        physics_dt,
    )
    future_reset_steps = 0
    per_attempt_steps: list[dict[str, Any]] = []
    for attempt in demo1_attempts:
        expected_stable = bool(attempt["expected_to_stabilize"])
        stabilization_steps = (
            int(stable_report["physics_steps"])
            if expected_stable
            else _failed_stabilization_physics_steps(stable_report)
        )
        total = 1 + stabilization_steps
        future_reset_steps += total
        per_attempt_steps.append(
            {
                "seed": int(attempt["seed"]),
                "expected_to_stabilize": expected_stable,
                "reset_physics_steps": 1,
                "stabilization_physics_steps": stabilization_steps,
                "total_physics_steps": total,
            }
        )
    hold_physics_steps = source_step - current_step - future_reset_steps
    valid = bool(hold_physics_steps >= 0 and hold_physics_steps % n_substeps == 0)
    report: dict[str, Any] = {
        "unobserved_prefix_mode": "hold_current_targets_for_inferred_duration",
        "is_exact_source_action_replay": False,
        "source_frame0_physics_step": source_step,
        "current_physics_step_after_collection_prefix": current_step,
        "future_demo1_reset_physics_steps": future_reset_steps,
        "future_demo1_reset_breakdown": per_attempt_steps,
        "inferred_hold_physics_steps": hold_physics_steps,
        "n_substeps": n_substeps,
        "inference_valid": valid,
        "source_time_rounding_residual_seconds": source_rounding_residual,
        "current_time_rounding_residual_seconds": current_rounding_residual,
    }
    if not valid:
        report["executed_hold_control_steps"] = 0
        report["reason"] = (
            "inferred missing duration is negative or not divisible by the "
            "recorded control substep count; no duration was guessed"
        )
        report["wall_seconds"] = time.monotonic() - started
        return report

    hold_control_steps = hold_physics_steps // n_substeps
    for control_index in range(hold_control_steps):
        raw_env.step_with_current_joint_targets()
        if progress_every > 0 and (
            (control_index + 1) % progress_every == 0
            or control_index + 1 == hold_control_steps
        ):
            print(
                "[history rebuild] inferred unpublished prefix hold "
                f"{control_index + 1}/{hold_control_steps} control steps",
                flush=True,
            )
    final_step, final_rounding_residual = _physics_step_from_time(
        trace.snapshot()["physx_sim_time"],
        physics_dt,
    )
    expected_final_step = source_step - future_reset_steps
    report.update(
        {
            "executed_hold_control_steps": hold_control_steps,
            "executed_hold_physics_steps": hold_control_steps * n_substeps,
            "expected_physics_step_before_demo1_attempts": expected_final_step,
            "actual_physics_step_before_demo1_attempts": final_step,
            "clock_step_residual": final_step - expected_final_step,
            "final_time_rounding_residual_seconds": final_rounding_residual,
            "wall_seconds": time.monotonic() - started,
        }
    )
    return report


def _drive_snapshot(drives: Mapping[str, np.ndarray], step_index: int) -> dict[str, np.ndarray]:
    return {
        name[len("next_") :]: np.asarray(values[step_index])
        for name, values in drives.items()
    }


def _run_demo_drives(
    raw_env: Any,
    trace: Any,
    schema: Mapping[str, Any],
    drives: Mapping[str, np.ndarray],
    expected_next_audit: Mapping[str, np.ndarray],
    source_outcomes: Mapping[str, np.ndarray],
    *,
    demo_label: str,
    progress_every: int,
    divergence_tolerance: float,
) -> dict[str, Any]:
    from handson.sim.isaaclab.articulated_replay import (
        RecordedArticulatedDrivePath,
        run_articulated_drive_control_step,
    )
    from handson.sim.isaaclab.articulated_replay.trajectory_audit import (
        TrajectoryAudit,
    )

    path = RecordedArticulatedDrivePath(schema)
    step_count = int(drives[_DRIVE_FIELDS[0]].shape[0])
    audit = TrajectoryAudit(
        schema,
        divergence_tolerance,
        runtime_schema=trace.schema(),
    )
    started = time.monotonic()
    total_substeps = 0
    total_target_writes = 0
    terminal_steps: list[int] = []
    terminated_steps: list[int] = []
    truncated_steps: list[int] = []
    succeeded_steps: list[int] = []
    runtime_error: dict[str, Any] | None = None
    for step_index in range(step_count):
        try:
            drive_step = path.control_step(_drive_snapshot(drives, step_index))
            result, drive_report = run_articulated_drive_control_step(
                raw_env,
                trace,
                schema,
                drive_step,
            )
        except BaseException as exc:
            runtime_error = {
                "step_zero_based": step_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            break
        observation, _reward, terminated, truncated, info = result
        if not isinstance(observation, Mapping):
            runtime_error = {
                "step_zero_based": step_index,
                "error_type": "TypeError",
                "error": "runtime observation is not a mapping",
            }
            break
        total_substeps += int(drive_report["recorded_substep_count"])
        total_target_writes += int(drive_report["applied_robot_joint_target_count"])
        if bool(terminated or truncated):
            terminal_steps.append(step_index)
        if bool(terminated):
            terminated_steps.append(step_index)
        if bool(truncated):
            truncated_steps.append(step_index)
        if isinstance(info, Mapping) and bool(info.get("succeed", False)):
            succeeded_steps.append(step_index)

        expected_step = {
            name: np.asarray(values[step_index])
            for name, values in expected_next_audit.items()
        }
        audit.compare_step(expected_step, observation, step_index)
        if progress_every > 0 and (
            (step_index + 1) % progress_every == 0
            or step_index + 1 == step_count
        ):
            print(
                f"[history rebuild] {demo_label} {step_index + 1}/{step_count} "
                f"control steps, {total_substeps} substeps",
                flush=True,
            )
    executed_steps = audit.step_count
    audit_report = audit.report()
    source_succeed = np.asarray(source_outcomes["succeed"]).astype(bool)
    source_terminal = np.asarray(source_outcomes["terminal"]).astype(bool)
    actual_succeed = np.zeros(executed_steps, dtype=bool)
    actual_terminal = np.zeros(executed_steps, dtype=bool)
    if succeeded_steps:
        actual_succeed[np.asarray(succeeded_steps, dtype=np.int64)] = True
    if terminal_steps:
        valid_terminal = [index for index in terminal_steps if index < executed_steps]
        actual_terminal[np.asarray(valid_terminal, dtype=np.int64)] = True
    source_succeed_executed = source_succeed[:executed_steps]
    source_terminal_executed = source_terminal[:executed_steps]
    outcome_mismatch = np.flatnonzero(
        (source_succeed_executed != actual_succeed)
        | (source_terminal_executed != actual_terminal)
    )
    source_task_outcome = {
        "final_succeed": bool(source_succeed[-1]) if source_succeed.size else False,
        "final_terminal": bool(source_terminal[-1]) if source_terminal.size else False,
        "any_succeed": bool(np.any(source_succeed)),
        "terminal_steps_zero_based": np.flatnonzero(source_terminal).astype(int).tolist(),
        "succeeded_steps_zero_based": np.flatnonzero(source_succeed).astype(int).tolist(),
    }
    actual_task_outcome = {
        "final_succeed": bool(actual_succeed[-1]) if actual_succeed.size else False,
        "final_terminal": bool(actual_terminal[-1]) if actual_terminal.size else False,
        "any_succeed": bool(np.any(actual_succeed)),
        "terminal_steps_zero_based": terminal_steps,
        "terminated_steps_zero_based": terminated_steps,
        "truncated_steps_zero_based": truncated_steps,
        "succeeded_steps_zero_based": succeeded_steps,
    }
    task_outcome = {
        "source": source_task_outcome,
        "actual": actual_task_outcome,
        "compared_control_steps": executed_steps,
        "stepwise_match_for_executed_prefix": not bool(outcome_mismatch.size),
        "first_mismatch_step_zero_based": (
            int(outcome_mismatch[0]) if outcome_mismatch.size else None
        ),
        "final_outcome_match": bool(
            executed_steps == step_count
            and source_task_outcome["final_succeed"] == actual_task_outcome["final_succeed"]
            and source_task_outcome["final_terminal"] == actual_task_outcome["final_terminal"]
        ),
    }
    return {
        "source_control_steps": step_count,
        "executed_control_steps": executed_steps,
        "executed_all_control_steps": executed_steps == step_count,
        "physics_substeps": total_substeps,
        "joint_target_assignments": total_target_writes,
        "audit": audit_report,
        "strict_schema_pass": bool(
            executed_steps == step_count and audit_report["strict_schema_pass"]
        ),
        "global_tolerance_pass": bool(
            executed_steps == step_count and audit_report["global_tolerance_pass"]
        ),
        "first_nonexact": audit_report["first_nonexact"],
        "first_divergence": audit_report["first_divergence"],
        "runtime_error": runtime_error,
        "task_outcome": task_outcome,
        "wall_seconds": time.monotonic() - started,
    }


def _reset_attempt_count(recording: Mapping[str, Any]) -> int:
    stable = recording.get("stable_recording_origin")
    if not isinstance(stable, Mapping):
        raise ValueError("recording has no stable_recording_origin object")
    value = stable.get("reset_stabilization_attempt", 1)
    if isinstance(value, bool):
        raise ValueError("reset_stabilization_attempt must be an integer")
    attempt = int(value)
    if attempt <= 0:
        raise ValueError("reset_stabilization_attempt must be positive")
    return attempt


def _validate_source_chain(
    recordings: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(recordings) < 2:
        raise ValueError("history rebuild requires at least two demo directories")
    collections = [_scalar_int(item, "collection_run_seed") for item in recordings]
    if len(set(collections)) != 1:
        raise ValueError(f"demos have different collection seeds: {collections}")
    for index, recording in enumerate(recordings, start=1):
        if recording.get("randomize") is not True:
            raise ValueError(f"demo {index} requires recording.randomize=true")
        if recording.get("randomize_dynamics") is not False:
            raise ValueError(f"demo {index} requires randomize_dynamics=false")

    accepted_seeds = [_scalar_int(item, "episode_reset_seed") for item in recordings]
    attempt_counts = [_reset_attempt_count(item) for item in recordings]
    for index in range(1, len(recordings)):
        expected = accepted_seeds[index - 1] + attempt_counts[index]
        if accepted_seeds[index] != expected:
            raise ValueError(
                "source accepted reset seeds do not match recorded attempt counts: "
                f"demo {index + 1} seed={accepted_seeds[index]}, expected={expected}"
            )
    attempts: list[list[dict[str, Any]]] = []
    for accepted_seed, attempt_count in zip(accepted_seeds, attempt_counts, strict=True):
        first_seed = accepted_seed - attempt_count + 1
        attempts.append(
            [
                {
                    "seed": seed,
                    "expected_to_stabilize": seed == accepted_seed,
                }
                for seed in range(first_seed, accepted_seed + 1)
            ]
        )
    return {
        "collection_seed": collections[0],
        "accepted_seeds": accepted_seeds,
        "reset_attempts": attempts,
    }


def _demo_dirs_from_args(args: argparse.Namespace) -> list[Path]:
    raw = getattr(args, "demo_dirs", None)
    if raw is None:
        # Compatibility for callers that constructed the old two-field
        # Namespace directly instead of going through argparse.
        raw = [args.demo1_dir, args.demo2_dir]
    result = [Path(item).expanduser().resolve() for item in raw]
    if len(result) < 2:
        raise ValueError("provide at least two consecutive demo directories")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not np.isfinite(args.divergence_tolerance) or args.divergence_tolerance < 0.0:
        raise ValueError("--divergence-tolerance must be finite and non-negative")
    demo_dirs = _demo_dirs_from_args(args)
    manifests_and_recordings = [_manifest_recording(path) for path in demo_dirs]
    manifests = [item[0] for item in manifests_and_recordings]
    recordings = [item[1] for item in manifests_and_recordings]
    chain = _validate_source_chain(recordings)
    collection_seed = int(chain["collection_seed"])
    schema1 = _read_json(demo_dirs[0] / "recording_schema.json")

    _apply_recorded_environment(manifests[0])
    # This experiment never has a public-state restore path, even when a
    # caller's shell previously enabled snapshot initialization.
    os.environ.pop("HANDSON_FORCE_PHYSX_SNAPSHOT_INIT", None)

    from handson.sim.isaaclab.articulated_replay import (
        validate_recorded_physics_configuration,
    )
    from handson.tasks.mappings import CONFIG_MAPPING

    recording1 = recordings[0]
    render_mode = str(recording1.get("render_mode", "human"))
    image_obs = bool(recording1.get("image_obs", True))
    headless = bool(recording1.get("headless", render_mode != "human"))
    config = CONFIG_MAPPING["t09_insert_peg"]()
    env = config.get_native_environment(
        # Collection used the teleoperation wrapper.  We retain it for the
        # same construction/reset call path, while driving env.unwrapped below.
        policy_mode=False,
        render_mode=render_mode,
        randomize=True,
        sim_backend="isaacsim",
        randomize_dynamics=False,
        seed=collection_seed,
        arm_type="rokae",
        hand_type="wuji1",
        image_obs=image_obs,
        headless=headless,
    )
    raw_env = env.unwrapped
    if args.no_wall_clock_throttle:
        raw_env.hz = 0
    trace = getattr(raw_env, "_physx_ground_truth_trace", None)
    if trace is None:
        raise RuntimeError("runtime did not initialize a PhysX ground-truth trace")

    report: dict[str, Any] = {
        "format": "handson.b03_same_scene_history_rebuild",
        "version": 3,
        "demo_directories": [str(path) for path in demo_dirs],
        "demo1": str(demo_dirs[0]),
        "demo2": str(demo_dirs[1]),
        "collection_seed": collection_seed,
        "accepted_episode_reset_seeds": chain["accepted_seeds"],
        "randomize": True,
        "same_environment_instance": True,
        "no_public_state_restore": True,
        "no_snapshot_fallback": True,
        "private_contact_state_directly_written": False,
        "contact_output_is_not_private_solver_cache": True,
        "source_wrapper_reproduced": "SingleArmViveHandTeleopWrapper",
        "exact_history_certificate_valid": False,
        "exact_history_certificate_reasons": [
            (
                "the collection performed an accepted reset/stabilization at "
                f"seed {collection_seed}, followed by an unpublished episode; "
                "its action duration and drive history are absent from the "
                "published demo directories"
            )
        ],
        "demos": [
            {
                "index": index,
                "directory": str(path),
                "episode_reset_seed": int(chain["accepted_seeds"][index - 1]),
                "status": "pending",
            }
            for index, path in enumerate(demo_dirs, start=1)
        ],
    }
    try:
        report["physics_configuration"] = validate_recorded_physics_configuration(
            schema1,
            trace.schema(),
        )
        stable_report1 = recording1.get("stable_recording_origin", {})
        prefix_reset = _reset_and_stabilize(env, collection_seed, stable_report1)
        prefix_reset["expected_to_stabilize"] = True
        prefix_reset["matches_source_outcome"] = bool(prefix_reset["stabilized"])
        prefix_reset["followed_by_unpublished_episode_in_source"] = True
        report["known_collection_prefix_reset"] = prefix_reset
        print(
            f"[history rebuild] collection prefix seed {collection_seed}: "
            f"stabilized={prefix_reset['stabilized']} (source expected True)",
            flush=True,
        )
        abort_reason: str | None = None
        prior_divergence = not bool(prefix_reset["matches_source_outcome"])
        if not prefix_reset["stabilized"]:
            abort_reason = (
                f"accepted collection-prefix seed {collection_seed} did not stabilize"
            )
        else:
            _prefix_schema, source_demo1_initial, _empty_drives, _empty_audit, _empty_outcomes = (
                _load_source(demo_dirs[0], load_drive_path=False)
            )
            physx_schema = schema1.get("environment_schema", {}).get("physx", {})
            n_substeps = int(physx_schema.get("n_substeps", 0))
            try:
                unobserved_prefix = _run_inferred_unpublished_prefix(
                    raw_env,
                    trace,
                    source_demo1_initial,
                    stable_report1,
                    chain["reset_attempts"][0],
                    n_substeps=n_substeps,
                    progress_every=args.progress_every,
                )
            except BaseException as exc:
                unobserved_prefix = {
                    "unobserved_prefix_mode": (
                        "hold_current_targets_for_inferred_duration"
                    ),
                    "inference_valid": False,
                    "runtime_error": {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                }
                abort_reason = (
                    "unrecoverable runtime error while executing the inferred "
                    "unpublished-prefix duration"
                )
            report["unobserved_collection_prefix"] = unobserved_prefix
            if not unobserved_prefix.get("inference_valid", False):
                prior_divergence = True
            if unobserved_prefix.get("clock_step_residual", 0) != 0:
                prior_divergence = True

        for demo_offset, (demo_dir, recording) in enumerate(
            zip(demo_dirs, recordings, strict=True)
        ):
            entry = report["demos"][demo_offset]
            entry["downstream_of_prior_divergence"] = bool(prior_divergence)
            entry["downstream_of_unpublished_prefix_history"] = True
            if abort_reason is not None:
                entry["status"] = "skipped"
                entry["skip_reason"] = abort_reason
                continue

            schema, initial, drives, next_audit, source_outcomes = _load_source(
                demo_dir,
                load_drive_path=True,
            )
            entry["physics_configuration"] = validate_recorded_physics_configuration(
                schema,
                trace.schema(),
            )
            stable_report = recording.get("stable_recording_origin", {})
            reset_results: list[dict[str, Any]] = []
            accepted_stabilized = False
            reset_history_matches = True
            for attempt_spec in chain["reset_attempts"][demo_offset]:
                attempt = _reset_and_stabilize(
                    env,
                    int(attempt_spec["seed"]),
                    stable_report,
                )
                expected_stable = bool(attempt_spec["expected_to_stabilize"])
                attempt["expected_to_stabilize"] = expected_stable
                attempt["matches_source_outcome"] = (
                    bool(attempt["stabilized"]) == expected_stable
                )
                reset_history_matches = bool(
                    reset_history_matches and attempt["matches_source_outcome"]
                )
                reset_results.append(attempt)
                print(
                    f"[history rebuild] demo {demo_offset + 1} seed "
                    f"{attempt['seed']}: stabilized={attempt['stabilized']} "
                    f"(source expected {expected_stable})",
                    flush=True,
                )
                if expected_stable:
                    accepted_stabilized = bool(attempt["stabilized"])
            entry["reset_attempts"] = reset_results
            entry["reset_history_matches_source"] = reset_history_matches
            if not reset_history_matches:
                prior_divergence = True
            if not accepted_stabilized:
                entry["status"] = "reset_failed"
                entry["failure_reason"] = (
                    "the accepted source reset did not stabilize; trajectory "
                    "execution is not physically meaningful"
                )
                abort_reason = (
                    f"demo {demo_offset + 1} accepted reset failed; later demos "
                    "were not executed"
                )
                prior_divergence = True
                continue

            from handson.sim.isaaclab.articulated_replay.trajectory_audit import (
                TrajectoryAudit,
            )

            runtime_initial = trace.snapshot()
            # A reset-origin observation has no preceding control interval.
            # Clear any capture left by the inferred prefix hold and reproduce
            # the source frame-0 convention: count=0 and padded zeros.
            trace.begin_substep_capture()
            runtime_initial.update(trace.substep_snapshot(consume=True))
            pre_action_core = _compare_snapshot(initial, runtime_initial)
            pre_action_auditor = TrajectoryAudit(
                schema,
                args.divergence_tolerance,
                runtime_schema=trace.schema(),
            )
            pre_action_auditor.compare_step(initial, runtime_initial, 0)
            pre_action_audit = pre_action_auditor.report()
            pre_action = {
                "core_public_snapshot": pre_action_core,
                "audit": pre_action_audit,
                "strict_schema_pass": pre_action_audit["strict_schema_pass"],
                "solver_state_schema_pass": pre_action_audit[
                    "solver_state_schema_pass"
                ],
                "all_audited_public_outputs_pass": pre_action_audit[
                    "all_audited_public_outputs_pass"
                ],
            }
            entry["pre_action"] = pre_action
            print(
                f"[history rebuild] demo {demo_offset + 1} frame0 core exact="
                f"{pre_action_core['core_physical_fields_exact']}, core max="
                f"{pre_action_core['core_max_abs_error']:.6g}, schema pass="
                f"{pre_action_audit['strict_schema_pass']}",
                flush=True,
            )
            if not pre_action_audit["strict_schema_pass"]:
                prior_divergence = True

            trajectory = _run_demo_drives(
                raw_env,
                trace,
                schema,
                drives,
                next_audit,
                source_outcomes,
                demo_label=f"demo {demo_offset + 1}",
                progress_every=args.progress_every,
                divergence_tolerance=args.divergence_tolerance,
            )
            entry["trajectory"] = trajectory
            entry["task_outcome"] = trajectory["task_outcome"]
            entry["strict_schema_pass"] = bool(
                reset_history_matches
                and pre_action_audit["strict_schema_pass"]
                and trajectory["strict_schema_pass"]
                and trajectory["task_outcome"][
                    "stepwise_match_for_executed_prefix"
                ]
                and trajectory["task_outcome"]["final_outcome_match"]
            )
            entry["global_tolerance_pass"] = bool(
                reset_history_matches
                and trajectory["global_tolerance_pass"]
                and trajectory["task_outcome"][
                    "stepwise_match_for_executed_prefix"
                ]
                and trajectory["task_outcome"]["final_outcome_match"]
            )
            if trajectory["runtime_error"] is not None:
                entry["status"] = "runtime_error"
                entry["failure_reason"] = trajectory["runtime_error"]
                abort_reason = (
                    f"demo {demo_offset + 1} encountered an unrecoverable runtime "
                    "error; later demos were not executed"
                )
            else:
                entry["status"] = "completed"
            if not entry["strict_schema_pass"]:
                prior_divergence = True

        completed_demos = [item for item in report["demos"] if item["status"] == "completed"]
        all_executed = all(
            item["status"] == "completed"
            and item.get("trajectory", {}).get("executed_all_control_steps", False)
            for item in report["demos"]
        )
        trajectory_strict_pass = bool(
            all_executed
            and all(item.get("strict_schema_pass", False) for item in report["demos"])
        )
        report["execution"] = {
            "requested_demo_count": len(demo_dirs),
            "completed_demo_count": len(completed_demos),
            "all_demos_executed": all_executed,
            "abort_reason": abort_reason,
        }
        report["trajectory_strict_pass"] = trajectory_strict_pass
        report["all_audited_public_outputs_pass"] = trajectory_strict_pass
        # Fail closed: even exact public trajectories cannot certify exact
        # private history while the initial unpublished episode is missing.
        report["strict_pass"] = bool(
            trajectory_strict_pass and report["exact_history_certificate_valid"]
        )
        report["conclusion"] = {
            "trajectory_strict_pass": trajectory_strict_pass,
            "exact_history_certificate_valid": False,
            "strict_pass": report["strict_pass"],
            "m_t_lambda_t_directly_observable": False,
            "interpretation": (
                "Every completed trajectory was audited without correction. "
                "The public contact report is compared separately and is not "
                "a serialization or equality proof for PhysX's private "
                "M_t/lambda_t. The missing unpublished prefix episode makes "
                "an exact-history certificate unavailable even if all public "
                "outputs happen to match."
            ),
        }
        # Preserve the most useful v1 aliases for existing report consumers.
        if report["demos"]:
            report["demo1_pre_action_comparison"] = report["demos"][0].get("pre_action")
            report["demo1_drive_replay"] = report["demos"][0].get("trajectory")
        if len(report["demos"]) > 1:
            report["demo2_pre_action_comparison"] = report["demos"][1].get("pre_action")
        # On this Isaac runtime SimulationApp.close() can end the process
        # instead of returning to ``main``.  Persist the completed audit while
        # Kit is still alive; ``main`` safely rewrites it when close() returns.
        report["completed"] = True
        _write_report_atomic(args.report, report)
        return report
    finally:
        try:
            env.close()
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "demo_dirs",
        type=Path,
        nargs="+",
        help="Two or more consecutive demo directories from one collection run.",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--divergence-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--no-wall-clock-throttle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable the 30 Hz sleep; this does not change PhysX stepping.",
    )
    return parser.parse_args()


def _write_report_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def main() -> int:
    args = _parse_args()
    try:
        report = run(args)
    except BaseException as exc:
        failure = {
            "format": "handson.b03_same_scene_history_rebuild",
            "version": 3,
            "completed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_report_atomic(args.report, failure)
        raise
    report["completed"] = True
    output = _write_report_atomic(args.report, report)
    print(f"[history rebuild] report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
