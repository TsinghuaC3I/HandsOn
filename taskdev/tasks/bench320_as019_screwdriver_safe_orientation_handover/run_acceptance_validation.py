#!/usr/bin/env python3
"""Run AS019 reset, camera, stability, outcome, or stage-0 canary gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import traceback
from typing import Callable, TextIO

import numpy as np


_EXPECTED_CHECKPOINT_ENV = "AS019_EXPECTED_CHECKPOINT"
_AUTOMATIC_POLICY_ENV = "HANDSON_AS019_AUTOMATIC_POLICY"
_AUTOMATIC_POLICY_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_ARM_IK_BACKEND_ENV = "HANDSON_ISAACLAB_ARM_IK_BACKEND"
_REQUIRED_ARM_IK_BACKEND = "haorui_new"
_ALLOW_DLS_FALLBACK_ENV = "HANDSON_ISAACLAB_ALLOW_DLS_FALLBACK"
_REQUIRED_ALLOW_DLS_FALLBACK = "0"
_CONTROL_TRACE_ENV = "HANDSON_AS019_CONTROL_TRACE"
_REQUIRED_CONTROL_TRACE = "1"
_CLOSE_WATCHDOG_FORCED_EXIT_CODE = 70
_FINGERPRINT_RELATIVE_PATHS = (
    "handson/handson/sim/isaaclab/as019_screwdriver_handover_env.py",
    "handson/handson/sim/isaaclab/bimanual_assembly_env.py",
    (
        "handson/handson/tasks/"
        "t22_pass_screwdriver/config.py"
    ),
    (
        "taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/"
        "scene_contract.py"
    ),
    (
        "taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/"
        "scene_runtime.py"
    ),
    (
        "taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/"
        "task.yaml"
    ),
    (
        "taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/"
        "assets/polyhaven_flathead_screwdriver/asset_manifest.json"
    ),
    (
        "taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/"
        "assets/polyhaven_flathead_screwdriver/runtime/flathead_screwdriver.usd"
    ),
    (
        "taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/"
        "run_acceptance_validation.py"
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_content_fingerprint(
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Hash the exact code/config/asset inputs used by an AS019 validation."""

    root = (
        Path(__file__).resolve().parents[3]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    file_hashes: dict[str, str] = {}
    combined = hashlib.sha256()
    for relative in sorted(_FINGERPRINT_RELATIVE_PATHS):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"AS019 fingerprint input is missing: {path}")
        file_hash = _sha256_file(path)
        file_hashes[relative] = file_hash
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(bytes.fromhex(file_hash))
        combined.update(b"\n")
    return {
        "algorithm": "sha256(sorted(relative_path + NUL + file_sha256_bytes + LF))",
        "repository_root": str(root),
        "files": file_hashes,
        "combined_sha256": combined.hexdigest(),
    }


def _expected_checkpoint_gate(
    fingerprint: dict[str, object],
    environ: object = os.environ,
) -> dict[str, object]:
    expected = getattr(environ, "get")(_EXPECTED_CHECKPOINT_ENV)
    actual = str(fingerprint["combined_sha256"])
    valid_format = bool(
        isinstance(expected, str)
        and len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected)
    )
    matches = bool(valid_format and expected == actual)
    if expected is None or expected == "":
        reason = "missing_expected_checkpoint"
    elif not valid_format:
        reason = "invalid_expected_checkpoint_format"
    elif not matches:
        reason = "checkpoint_mismatch"
    else:
        reason = ""
    return {
        "pass": matches,
        "environment_variable": _EXPECTED_CHECKPOINT_ENV,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "reason": reason,
    }


def _ambient_automatic_policy_gate(
    environ: object = os.environ,
) -> dict[str, object]:
    value = getattr(environ, "get")(_AUTOMATIC_POLICY_ENV, "")
    normalized = str(value).strip().lower()
    enabled = normalized in _AUTOMATIC_POLICY_TRUE_VALUES
    return {
        "pass": not enabled,
        "environment_variable": _AUTOMATIC_POLICY_ENV,
        "raw_value": value,
        "automatic_policy_enabled": enabled,
    }


def _required_arm_ik_environment_gate(
    environ: object = os.environ,
) -> dict[str, object]:
    """Require the exact IsaacLab IK backend used by AS019 validation."""

    value = getattr(environ, "get")(_ARM_IK_BACKEND_ENV)
    normalized = str(value or "").strip().lower()
    fallback_value = getattr(environ, "get")(_ALLOW_DLS_FALLBACK_ENV)
    fallback_normalized = str(fallback_value or "").strip().lower()
    trace_value = getattr(environ, "get")(_CONTROL_TRACE_ENV)
    trace_normalized = str(trace_value or "").strip().lower()
    backend_matches = normalized == _REQUIRED_ARM_IK_BACKEND
    fallback_matches = fallback_normalized == _REQUIRED_ALLOW_DLS_FALLBACK
    trace_matches = trace_normalized == _REQUIRED_CONTROL_TRACE
    matches = bool(backend_matches and fallback_matches and trace_matches)
    if value is None or normalized == "":
        reason = "missing_required_isaaclab_arm_ik_backend"
    elif not backend_matches:
        reason = "unexpected_isaaclab_arm_ik_backend"
    elif fallback_value is None or fallback_normalized == "":
        reason = "missing_dls_fallback_policy"
    elif not fallback_matches:
        reason = "dls_fallback_must_be_disabled"
    elif trace_value is None or trace_normalized == "":
        reason = "missing_as019_control_trace_policy"
    elif not trace_matches:
        reason = "as019_control_trace_must_be_enabled"
    else:
        reason = ""
    return {
        "pass": matches,
        "environment_variable": _ARM_IK_BACKEND_ENV,
        "required_value": _REQUIRED_ARM_IK_BACKEND,
        "raw_value": value,
        "normalized_value": normalized,
        "fallback_environment_variable": _ALLOW_DLS_FALLBACK_ENV,
        "required_fallback_value": _REQUIRED_ALLOW_DLS_FALLBACK,
        "fallback_raw_value": fallback_value,
        "fallback_normalized_value": fallback_normalized,
        "trace_environment_variable": _CONTROL_TRACE_ENV,
        "required_trace_value": _REQUIRED_CONTROL_TRACE,
        "trace_raw_value": trace_value,
        "trace_normalized_value": trace_normalized,
        "reason": reason,
    }


def _runtime_control_gate(raw: object) -> dict[str, object]:
    """Fail closed unless the constructed runtime retained Haorui pose IK."""

    physics_backend = str(getattr(raw, "_physics_backend", ""))
    arm_ik_backend = str(getattr(raw, "_arm_ik_backend", ""))
    haorui_enabled = getattr(raw, "_haorui_ik_enabled", None) is True
    command_types_raw = getattr(raw, "_ik_command_types", {})
    command_types = (
        {str(key): str(value) for key, value in command_types_raw.items()}
        if isinstance(command_types_raw, dict)
        else {}
    )
    checks = {
        "physx_backend": physics_backend == "physx",
        "haorui_backend": arm_ik_backend == "haorui",
        "haorui_initialized": haorui_enabled,
        "right_pose_command": command_types.get("right") == "pose",
        "left_pose_command": command_types.get("left") == "pose",
        "control_trace_enabled": (
            getattr(raw, "_as019_control_trace_enabled", None) is True
        ),
    }
    return {
        "pass": bool(all(checks.values())),
        "physics_backend": physics_backend,
        "arm_ik_backend": arm_ik_backend,
        "haorui_ik_enabled": haorui_enabled,
        "ik_command_types": command_types,
        "checks": checks,
    }


def _runtime_rate_gate(raw: object) -> dict[str, object]:
    errors: list[str] = []

    def finite_float(attribute: str) -> float | None:
        try:
            value = float(getattr(raw, attribute))
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"{attribute}:{type(exc).__name__}: {exc}")
            return None
        if not math.isfinite(value) or value <= 0.0:
            errors.append(f"{attribute}:expected_finite_positive")
            return None
        return value

    control_dt = finite_float("control_dt")
    physics_dt = finite_float("physics_dt")
    simulation_physics_dt = finite_float("_sim_dt")
    try:
        raw_substeps = getattr(raw, "_n_substeps")
        if isinstance(raw_substeps, bool):
            raise ValueError("boolean substep count")
        substeps = int(raw_substeps)
        if float(raw_substeps) != float(substeps):
            raise ValueError("non-integer substep count")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        errors.append(f"_n_substeps:{type(exc).__name__}: {exc}")
        substeps = None

    checks = {
        "control_hz_30": bool(
            control_dt is not None
            and math.isclose(control_dt, 1.0 / 30.0, rel_tol=1.0e-9, abs_tol=1.0e-12)
        ),
        "physics_hz_120": bool(
            physics_dt is not None
            and math.isclose(physics_dt, 1.0 / 120.0, rel_tol=1.0e-9, abs_tol=1.0e-12)
        ),
        "simulation_physics_hz_120": bool(
            simulation_physics_dt is not None
            and math.isclose(
                simulation_physics_dt,
                1.0 / 120.0,
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            )
        ),
        "four_substeps": substeps == 4,
        "dt_ratio_four": bool(
            control_dt is not None
            and physics_dt is not None
            and math.isclose(
                control_dt / physics_dt,
                4.0,
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            )
        ),
        "configured_and_simulation_physics_dt_match": bool(
            physics_dt is not None
            and simulation_physics_dt is not None
            and math.isclose(
                physics_dt,
                simulation_physics_dt,
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            )
        ),
    }
    return {
        "pass": bool(not errors and all(checks.values())),
        "expected": {"control_hz": 30, "physics_hz": 120, "substeps": 4},
        "control_dt": control_dt,
        "physics_dt": physics_dt,
        "simulation_physics_dt": simulation_physics_dt,
        "control_hz": 1.0 / control_dt if control_dt is not None else None,
        "physics_hz": 1.0 / physics_dt if physics_dt is not None else None,
        "simulation_physics_hz": (
            1.0 / simulation_physics_dt
            if simulation_physics_dt is not None
            else None
        ),
        "substeps": substeps,
        "checks": checks,
        "errors": errors,
    }


def _reset_with_audit(
    env: object,
    seed: int,
    explicit_reset_seeds: list[int],
):
    normalized_seed = int(seed)
    explicit_reset_seeds.append(normalized_seed)
    return env.reset(seed=normalized_seed)


def _as019_metrics_from_reset_info(info: object) -> dict[str, object]:
    """Recover the reset-time cached AS019 metrics without querying PhysX again."""

    if not isinstance(info, dict):
        return {}
    prefix = "as019_"
    return {
        key.removeprefix(prefix): value
        for key, value in info.items()
        if isinstance(key, str) and key.startswith(prefix)
    }


def _validation_seed_plan(stage0_canary_seed: int | None) -> dict[str, int | None]:
    if stage0_canary_seed is None:
        return {"environment_construction_seed": 9000, "initial_reset_seed": 9000}
    requested = int(stage0_canary_seed)
    return {"environment_construction_seed": requested, "initial_reset_seed": None}


def _stage0_reset_gate(
    *, requested_seed: int, explicit_reset_seeds: list[int]
) -> dict[str, object]:
    expected = [int(requested_seed)]
    observed = [int(seed) for seed in explicit_reset_seeds]
    return {
        "pass": observed == expected,
        "expected_explicit_reset_seeds": expected,
        "observed_explicit_reset_seeds": observed,
        "explicit_reset_count": len(observed),
    }


def _automatic_positive_max_steps(
    task_yaml_path: Path,
    *,
    cli_override: int | None = None,
) -> dict[str, object]:
    if cli_override is not None:
        if isinstance(cli_override, bool) or not isinstance(cli_override, int):
            raise ValueError("--positive-max-steps must be a positive integer")
        value = cli_override
        if value <= 0:
            raise ValueError("--positive-max-steps must be positive")
        return {"max_steps": value, "source": "cli", "task_yaml": None}

    import yaml

    path = Path(task_yaml_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        value = payload["validation"]["automatic_policy"]["positive_max_steps"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "task.yaml must define validation.automatic_policy.positive_max_steps"
        ) from exc
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "validation.automatic_policy.positive_max_steps must be a positive integer"
        )
    return {"max_steps": value, "source": "task_yaml", "task_yaml": str(path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--camera-stability-only", action="store_true")
    modes.add_argument("--outcomes-only", action="store_true")
    modes.add_argument(
        "--stage0-canary-seed",
        type=int,
        help=(
            "run one right-tail acquisition seed and stop when stage 0 exits; "
            "this is diagnostic evidence, not full task acceptance"
        ),
    )
    parser.add_argument("--stage0-max-steps", type=int, default=320)
    parser.add_argument("--positive-max-steps", type=int, default=None)
    parser.add_argument(
        "--close-watchdog-seconds",
        type=float,
        default=10.0,
        help="validator-only env.close timeout; set to 0 to disable",
    )
    return parser


def _finite(value) -> bool:
    return bool(np.all(np.isfinite(np.asarray(value))))


def _yaw_from_wxyz(quat: np.ndarray) -> float:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    w, x, y, z = quat
    return float(
        math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def _json_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class _ValidationFinished(Exception):
    """Internal control flow after a requested validation scope completes."""


def _write_summary_atomic(path: Path, summary: dict[str, object]) -> None:
    """Atomically replace a flushed JSON summary in its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        summary, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _start_close_watchdog(
    timeout_seconds: float | None,
    planned_exit_code: int,
    *,
    summary_path: Path,
    summary: dict[str, object],
    shutdown_state: dict[str, object],
    state_lock: object | None = None,
    exit_fn: Callable[[int], None] = os._exit,
    log_stream: TextIO | None = None,
) -> tuple[
    threading.Event | None,
    threading.Event | None,
    threading.Thread | None,
]:
    if timeout_seconds is None:
        return None, None, None
    cancel = threading.Event()
    timed_out = threading.Event()
    stream = sys.stderr if log_stream is None else log_stream
    lock = threading.Lock() if state_lock is None else state_lock

    def watchdog() -> None:
        if cancel.wait(timeout_seconds):
            return
        persistence_error: str | None = None
        with lock:
            shutdown_state["normal_close_completed"] = False
            shutdown_state["close_timed_out"] = True
            shutdown_state["forced_exit_code"] = _CLOSE_WATCHDOG_FORCED_EXIT_CODE
            summary["validation_pass_before_close_timeout"] = summary.get(
                "validation_pass"
            )
            summary["validation_pass"] = False
            summary["shutdown_terminal_reason"] = "close_timeout"
            # Schema-v3 compatibility: older evidence consumers inspect the
            # unqualified shutdown terminal key.  The namespaced field above
            # is authoritative and no longer obscures validation outcome.
            summary["terminal_reason"] = "close_timeout"
            try:
                _write_summary_atomic(summary_path, summary)
            except BaseException as exc:
                persistence_error = f"{type(exc).__name__}: {exc}"
            finally:
                timed_out.set()
        message = {
            "timeout_seconds": timeout_seconds,
            "planned_exit_code": int(planned_exit_code),
            "forced_exit_code": _CLOSE_WATCHDOG_FORCED_EXIT_CODE,
            "summary_path": str(summary_path),
            "summary_state": "close_timed_out",
            "summary_persistence_error": persistence_error,
            "action": "os._exit",
        }
        print(
            "AS019_SHUTDOWN_WATCHDOG_TIMEOUT="
            + json.dumps(message, sort_keys=True),
            file=stream,
            flush=True,
        )
        exit_fn(_CLOSE_WATCHDOG_FORCED_EXIT_CODE)

    thread = threading.Thread(
        target=watchdog,
        name="as019-validation-close-watchdog",
        daemon=True,
    )
    thread.start()
    return cancel, timed_out, thread


def _cancel_close_watchdog(
    cancel: threading.Event | None,
    thread: threading.Thread | None,
) -> None:
    if cancel is not None:
        cancel.set()
    if thread is not None:
        thread.join(timeout=1.0)


def _close_environment_with_watchdog(
    env: object | None,
    *,
    summary_path: Path,
    summary: dict[str, object],
    planned_exit_code: int,
    timeout_seconds: float | None,
    exit_fn: Callable[[int], None] = os._exit,
    log_stream: TextIO | None = None,
) -> None:
    state: dict[str, object] = {
        "enabled": timeout_seconds is not None,
        "timeout_seconds": timeout_seconds,
        "planned_exit_code": int(planned_exit_code),
        "close_attempted": env is not None,
        "normal_close_completed": None if env is None else False,
        "close_timed_out": False,
        "forced_exit_code": None,
    }
    summary["shutdown_watchdog"] = state
    # Evidence must be durable before Kit teardown starts.  A timeout replaces
    # this pending-close state with a durable failure before forcing exit 70.
    _write_summary_atomic(summary_path, summary)
    if env is None:
        return

    state_lock = threading.Lock()
    cancel, timed_out, thread = _start_close_watchdog(
        timeout_seconds,
        planned_exit_code,
        summary_path=summary_path,
        summary=summary,
        shutdown_state=state,
        state_lock=state_lock,
        exit_fn=exit_fn,
        log_stream=log_stream,
    )
    try:
        env.close()
    except BaseException as exc:
        _cancel_close_watchdog(cancel, thread)
        with state_lock:
            if timed_out is None or not timed_out.is_set():
                state["close_exception"] = repr(exc)
                summary["validation_pass_before_close_exception"] = summary.get(
                    "validation_pass"
                )
                summary["validation_pass"] = False
                summary["shutdown_terminal_reason"] = "close_exception"
                summary["terminal_reason"] = "close_exception"
                _write_summary_atomic(summary_path, summary)
        raise
    else:
        _cancel_close_watchdog(cancel, thread)
        with state_lock:
            if timed_out is None or not timed_out.is_set():
                state["normal_close_completed"] = True
                _write_summary_atomic(summary_path, summary)


def _stage0_confirmation_status(
    *, stage: int, control_step: int, stage0_exit_step: int | None
) -> tuple[int | None, bool]:
    """Latch first stage exit and stop only after one confirmation step."""

    if stage0_exit_step is None:
        return (control_step, False) if stage > 0 else (None, False)
    return stage0_exit_step, bool(control_step > stage0_exit_step)


def _strict_snapshot_int(value: object) -> int | None:
    """Return an exact finite integer without accepting booleans."""

    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    return None


def _right_wrench_sample_gate(raw: object) -> dict[str, object]:
    """Audit right-wrench ownership against its exact pre-substep evidence.

    The active wrench and its contact gate are sampled before the final physics
    substep, while ``_as019_last_metrics`` is refreshed afterwards.  Mixing
    those generations can both invent and hide a physical grasp.  The gate
    snapshot is therefore authoritative for wrench coverage; post-step metrics
    are retained only as explicitly named diagnostics and a later-contact
    confirmation signal.
    """

    metrics = dict(getattr(raw, "_as019_last_metrics", {}) or {})
    snapshot = dict(
        getattr(raw, "_as019_active_wrench_gate_snapshot", {}) or {}
    )
    snapshot_owners = tuple(snapshot.get("active_wrench_owners", ()) or ())
    current_owners = tuple(
        getattr(raw, "_as019_active_wrench_owners", ()) or ()
    )
    right_active = "right" in snapshot_owners
    current_env_step = _strict_snapshot_int(getattr(raw, "env_step", None))
    runtime_substeps = _strict_snapshot_int(
        getattr(raw, "_n_substeps", None)
    )
    snapshot_control_step = _strict_snapshot_int(
        snapshot.get("control_step_before")
    )
    snapshot_substep_index = _strict_snapshot_int(
        snapshot.get("physics_substep_index")
    )
    snapshot_substep_count = _strict_snapshot_int(
        snapshot.get("physics_substep_count")
    )
    snapshot_generation = _strict_snapshot_int(
        snapshot.get("metrics_generation")
    )
    post_step_generation = _strict_snapshot_int(
        metrics.get("metrics_generation")
    )
    lineage_checks = {
        "control_step_matches": bool(
            current_env_step is not None
            and snapshot_control_step == current_env_step - 1
        ),
        "final_physics_substep": bool(
            snapshot_substep_count is not None
            and snapshot_substep_count > 0
            and snapshot_substep_index == snapshot_substep_count - 1
        ),
        "physics_substep_count_matches_runtime": bool(
            runtime_substeps is not None
            and runtime_substeps > 0
            and snapshot_substep_count == runtime_substeps
        ),
        "metrics_generation_is_valid": bool(
            snapshot_generation is not None and snapshot_generation >= 0
        ),
    }
    diagnostic_checks = {
        "post_step_generation_after_snapshot": bool(
            snapshot_generation is not None
            and snapshot_generation >= 0
            and post_step_generation is not None
            and post_step_generation > snapshot_generation
        ),
    }
    gate_checks = {
        "contact_query_valid": bool(
            snapshot.get("contact_query_valid", False)
        ),
        "right_tail_physical": bool(
            snapshot.get("right_tail_physical", False)
        ),
        "right_close_command": bool(
            snapshot.get("right_close_command", False)
        ),
        "right_wrench_invariant": bool(
            snapshot.get("right_wrench_invariant", False)
        ),
        "apply_succeeded": bool(snapshot.get("apply_succeeded", False)),
    }
    lineage_pass = bool(all(lineage_checks.values()))
    active_gate_pass = bool(all(gate_checks.values()))
    failure_reasons = [
        f"lineage:{key}"
        for key, passed in lineage_checks.items()
        if right_active and not passed
    ] + [
        f"gate:{key}"
        for key, passed in gate_checks.items()
        if right_active and not passed
    ]
    return {
        "pass": bool(not right_active or (lineage_pass and active_gate_pass)),
        "right_active": right_active,
        "snapshot_active_wrench_owners": list(snapshot_owners),
        "post_step_active_wrench_owners": list(current_owners),
        "snapshot_right_tail_physical": gate_checks["right_tail_physical"],
        "post_step_right_tail_physical": bool(
            metrics.get("right_tail_physical", False)
        ),
        "snapshot_metrics_generation": snapshot_generation,
        "post_step_metrics_generation": post_step_generation,
        "lineage_pass": lineage_pass,
        "lineage_checks": lineage_checks,
        "diagnostic_checks": diagnostic_checks,
        "active_gate_pass": active_gate_pass,
        "gate_checks": gate_checks,
        "failure_reasons": failure_reasons,
        "gate_snapshot": snapshot,
    }


def _validation_trial_terminal_status(
    *,
    stage0_canary: bool,
    control_steps: int,
    maximum_trial_steps: int,
    stage0_exit_step: int | None,
    confirmation_complete: bool,
    success: bool,
    terminated: bool,
    truncated: bool,
    failure_reason: str,
) -> dict[str, object]:
    """Describe why a validator trial stopped, including its own budget."""

    if success:
        reason = "success"
    elif failure_reason:
        reason = f"task_failure:{failure_reason}"
    elif terminated:
        reason = "task_terminated"
    elif truncated:
        reason = "task_truncated"
    elif stage0_canary and confirmation_complete:
        reason = "stage0_confirmation_complete"
    else:
        exhausted = bool(
            control_steps >= maximum_trial_steps
            and (not stage0_canary or stage0_exit_step is None)
        )
        reason = (
            "stage0_acquisition_budget_exhausted"
            if exhausted and stage0_canary
            else "positive_validation_budget_exhausted"
            if exhausted
            else "loop_stopped_without_terminal"
        )
        return {"time_limit_reached": exhausted, "terminal_reason": reason}
    return {"time_limit_reached": False, "terminal_reason": reason}


def _table_safety_sample_gate(metrics: object) -> dict[str, object]:
    """Audit cached table evidence without issuing another PhysX query."""

    values = dict(metrics or {})

    def minimum_and_penetration(key: str) -> tuple[float | None, float]:
        minimum = _json_number(values.get(key))
        return minimum, max(0.0, -minimum) if minimum is not None else 0.0

    tool_minimum, tool_penetration = minimum_and_penetration(
        "tool_table_min_separation_m"
    )
    arm_minimum, arm_penetration = minimum_and_penetration(
        "arm_table_min_separation_m"
    )
    hand_minimum, hand_penetration = minimum_and_penetration(
        "hand_table_min_separation_m"
    )
    query_valid = bool(values.get("table_safety_query_valid", False))
    arm_contact = bool(values.get("arm_table_physical_contact", False))
    tool_violation = bool(
        values.get("tool_table_penetration_violation", False)
    )
    hand_violation = bool(
        values.get("hand_table_penetration_violation", False)
    )
    failure_reasons = []
    if not query_valid:
        failure_reasons.append("table_safety_query_invalid")
    if tool_violation:
        failure_reasons.append("tool_table_penetration_violation")
    if hand_violation:
        failure_reasons.append("hand_table_penetration_violation")
    return {
        "pass": not failure_reasons,
        "failure_reasons": failure_reasons,
        "query_valid": query_valid,
        "query_error": str(values.get("table_safety_query_error", "")),
        "tool_table_supported": bool(
            values.get("tool_table_supported", False)
        ),
        "tool_table_analytic_gap_m": _json_number(
            values.get("tool_table_analytic_gap_m")
        ),
        "tool_table_min_separation_m": tool_minimum,
        "tool_table_max_penetration_m": tool_penetration,
        "tool_table_penetration_violation": tool_violation,
        "tool_table_contact_paths": list(
            values.get("tool_table_contact_paths", ()) or ()
        ),
        "arm_table_physical_contact": arm_contact,
        "arm_table_min_separation_m": arm_minimum,
        "arm_table_max_penetration_m": arm_penetration,
        "arm_table_contact_paths": list(
            values.get("arm_table_contact_paths", ()) or ()
        ),
        "hand_table_physical_contact": bool(
            values.get("hand_table_physical_contact", False)
        ),
        "hand_table_min_separation_m": hand_minimum,
        "hand_table_max_penetration_m": hand_penetration,
        "hand_table_penetration_violation": hand_violation,
        "hand_table_contact_paths": list(
            values.get("hand_table_contact_paths", ()) or ()
        ),
        "force_gate_n": None,
        "force_used_for_pass": False,
    }


def _reset_table_clearance_gate(
    metrics: object,
    *,
    expected_gap_m: float,
    tolerance_m: float,
) -> dict[str, object]:
    """Validate the authored sleeping reset clearance without claiming support.

    The AS019 reset contract deliberately authors the tool 0.2 mm above the
    table and puts it to sleep.  A detailed table-contact row is therefore not
    required at reset.  Physical support remains mandatory for the later
    placement and stability stages inside the task runtime.
    """

    safety = _table_safety_sample_gate(metrics)
    expected = _json_number(expected_gap_m)
    tolerance = _json_number(tolerance_m)
    observed = safety["tool_table_analytic_gap_m"]
    finite_contract = bool(
        expected is not None
        and expected >= 0.0
        and tolerance is not None
        and tolerance >= 0.0
    )
    gap_matches = bool(
        finite_contract
        and observed is not None
        and observed >= 0.0
        and abs(observed - expected) <= tolerance
    )
    failure_reasons = list(safety["failure_reasons"])
    if not finite_contract:
        failure_reasons.append("invalid_reset_clearance_contract")
    elif observed is None:
        failure_reasons.append("missing_reset_table_analytic_gap")
    elif not gap_matches:
        failure_reasons.append("reset_table_clearance_mismatch")
    return {
        **safety,
        "pass": not failure_reasons,
        "failure_reasons": failure_reasons,
        "expected_analytic_gap_m": expected,
        "analytic_gap_tolerance_m": tolerance,
        "analytic_gap_matches_reset_contract": gap_matches,
        "physical_support_required": False,
        "physical_support_observed": bool(safety["tool_table_supported"]),
    }


def _right_wrench_gate_result(
    failures: list[dict[str, object]],
    coverage: dict[str, dict[str, int]],
    *,
    require_positive_coverage: bool,
) -> dict[str, object]:
    positive = coverage["positive"]
    invariant_pass = not failures
    positive_coverage_pass = bool(
        positive["active_right_wrench_steps"] > 0
        and positive["active_without_physical_steps"] == 0
    )
    return {
        "pass": bool(
            invariant_pass
            and (positive_coverage_pass if require_positive_coverage else True)
        ),
        "invariant_pass": invariant_pass,
        "positive_coverage_required": require_positive_coverage,
        "positive_coverage_pass": (
            positive_coverage_pass if require_positive_coverage else None
        ),
        "failure_count": len(failures),
        "failures": failures,
        "coverage_by_context": coverage,
    }


def _observational_detailed_contact_rows(
    raw: object,
    view: object,
    query_name: str,
) -> tuple[list[tuple[float, np.ndarray, float]], bool, bool, str]:
    """Query detailed rows without changing the task's production query state."""

    prior_valid = getattr(raw, "_as019_contact_query_valid")
    prior_error = getattr(raw, "_as019_contact_query_error")
    rows: list[tuple[float, np.ndarray, float]] = []
    penetration_violation = False
    query_valid = False
    query_error = "diagnostic_query_not_run"
    try:
        raw._as019_contact_query_valid = True
        raw._as019_contact_query_error = ""
        rows, penetration_violation = raw._detailed_contact_rows(view, query_name)
        query_valid = bool(raw._as019_contact_query_valid)
        query_error = str(raw._as019_contact_query_error)
    except Exception as exc:
        rows = []
        penetration_violation = False
        query_valid = False
        query_error = f"{query_name}:{type(exc).__name__}: {exc}"
    finally:
        raw._as019_contact_query_valid = prior_valid
        raw._as019_contact_query_error = prior_error
    return rows, penetration_violation, query_valid, query_error


def _right_finger_contact_telemetry(raw: object, runtime: object) -> dict[str, object]:
    tool_inverse = np.linalg.inv(
        np.asarray(raw._tool_matrix(), dtype=np.float64).reshape(4, 4)
    )
    radial_center = np.asarray(runtime._HANDLE_LOCAL_CENTER, dtype=np.float64)[1:3]
    grouped_records = getattr(raw, "_as019_owner_contact_records", {}).get(
        "right_fingers"
    )
    grouped_by_finger: dict[str, list[tuple[float, np.ndarray, float]]] | None = None
    grouped_query_valid = bool(getattr(raw, "_as019_contact_query_valid", False))
    grouped_query_error = str(getattr(raw, "_as019_contact_query_error", ""))
    if grouped_records is not None:
        grouped_by_finger = {
            f"finger{index}": [] for index in range(1, 6)
        }
        for record in grouped_records:
            name = raw._finger_name_from_filter_path(
                "right", str(record.get("filter_path", ""))
            )
            if name not in grouped_by_finger:
                grouped_query_valid = False
                grouped_query_error = "right_fingers:unattributed_filter_path"
                continue
            grouped_by_finger[name].append(
                (
                    float(record["force"]),
                    np.asarray(record["point"], dtype=np.float64),
                    float(record["separation"]),
                )
            )
    result: dict[str, object] = {}
    for index in range(1, 6):
        name = f"finger{index}"
        if grouped_by_finger is not None:
            rows = grouped_by_finger[name]
            penetration_violation = any(
                separation < -runtime._CONTACT_MAX_PENETRATION_M
                for _force, _point, separation in rows
            )
            query_valid = grouped_query_valid
            query_error = grouped_query_error
        else:
            view = raw._as019_right_per_finger_views.get(name)
            if view is None:
                result[name] = {
                    "query_valid": False,
                    "query_error": "view_unavailable",
                }
                continue
            rows, penetration_violation, query_valid, query_error = (
                _observational_detailed_contact_rows(
                    raw,
                    view,
                    f"right_{name}_canary",
                )
            )
        accepted: list[tuple[float, np.ndarray, float]] = []
        raw_local_rows: list[tuple[float, np.ndarray, float]] = []
        maximum_penetration = 0.0
        for force, point, separation in rows:
            maximum_penetration = max(maximum_penetration, max(0.0, -separation))
            local = (
                tool_inverse
                @ np.asarray([*point, 1.0], dtype=np.float64)
            )[:3]
            raw_local_rows.append((float(force), local, float(separation)))
            if (
                separation < -runtime._CONTACT_MAX_PENETRATION_M
                or force < runtime._CONTACT_FORCE_N
                or separation > runtime._CONTACT_MAX_SEPARATION_M
                or not (
                    runtime._RIGHT_TAIL_CONTACT_LOCAL_X_MIN_M
                    <= float(local[0])
                    <= runtime._RIGHT_TAIL_CONTACT_LOCAL_X_MAX_M
                )
                or not raw._inside_local_envelope(
                    local,
                    runtime._HANDLE_LOCAL_MIN,
                    runtime._HANDLE_LOCAL_MAX,
                )
            ):
                continue
            accepted.append((float(force), local, float(separation)))
        accepted_force = sum(row[0] for row in accepted)
        weighted = (
            sum(force * point for force, point, _ in accepted) / accepted_force
            if accepted_force > 0.0
            else None
        )
        separations = [row[2] for row in accepted]
        radial = weighted[1:3] - radial_center if weighted is not None else None
        result[name] = {
            "query_valid": query_valid,
            "query_error": query_error,
            "raw_contact_count": len(rows),
            "raw_contact_local_xyz_m": [
                point.tolist() for _force, point, _separation in raw_local_rows
            ],
            "raw_contact_force_n": [
                force for force, _point, _separation in raw_local_rows
            ],
            "raw_contact_separation_m": [
                separation for _force, _point, separation in raw_local_rows
            ],
            "accepted_contact_count": len(accepted),
            "accepted_force_n": float(accepted_force),
            "contact_local_x_m": (
                float(weighted[0]) if weighted is not None else None
            ),
            "radial_yz_m": radial.tolist() if radial is not None else None,
            "radial_radius_m": (
                float(np.linalg.norm(radial)) if radial is not None else None
            ),
            "minimum_separation_m": min(separations) if separations else None,
            "maximum_separation_m": max(separations) if separations else None,
            "maximum_penetration_m": maximum_penetration,
            "penetration_violation": bool(penetration_violation),
        }
    return result


def _pose_wxyz_matrix(pose: object) -> np.ndarray:
    """Convert one finite position/wxyz command into a rigid transform."""

    value = np.asarray(pose, dtype=np.float64).reshape(-1)
    if value.shape != (7,):
        raise ValueError(f"expected 7D pose, got {value.shape}")
    if not _finite(value):
        raise ValueError("pose contains non-finite values")
    quaternion = value[3:7]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("pose quaternion has zero norm")
    w, x, y, z = quaternion / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
    matrix[:3, 3] = value[:3]
    return matrix


def _matrix_pose_record(matrix: np.ndarray) -> dict[str, object]:
    value = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if not _finite(value):
        raise ValueError("pose matrix contains non-finite values")
    return {
        "position_m": value[:3, 3].tolist(),
        "rotation_matrix": value[:3, :3].tolist(),
    }


def _right_command_tracking_telemetry(
    raw: object,
    *,
    tool_inverse: np.ndarray,
    actual_palm_local: np.ndarray,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compare the last 54D policy command with live right-palm/hand state."""

    try:
        policy_step: int | None = int(raw._as019_last_policy_action_step)
    except (AttributeError, TypeError, ValueError, OverflowError):
        policy_step = None

    action: np.ndarray | None = None
    action_error: str | None = None
    try:
        value = getattr(raw, "_as019_last_policy_action")
        action = np.asarray(value, dtype=np.float64).reshape(-1)
        if action.shape != (54,):
            raise ValueError(f"expected 54D policy action, got {action.shape}")
    except Exception as exc:
        action = None
        action_error = f"{type(exc).__name__}: {exc}"

    palm_tracking: dict[str, object] = {
        "policy_action_step": policy_step,
        "commanded_tool_local_pose": None,
        "translation_residual_tool_local_xyz_m": None,
        "translation_residual_norm_m": None,
        "rotation_residual_rad": None,
        "command_error": action_error,
    }
    if action is not None:
        try:
            commanded_world = _pose_wxyz_matrix(action[:7])
            commanded_local = tool_inverse @ commanded_world
            translation_residual = (
                actual_palm_local[:3, 3] - commanded_local[:3, 3]
            )
            relative_rotation = (
                commanded_local[:3, :3].T @ actual_palm_local[:3, :3]
            )
            translation_residual_norm = float(
                np.linalg.norm(translation_residual)
            )
            cosine = float(
                np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
            )
            rotation_residual = math.acos(cosine)
            if (
                not _finite(translation_residual)
                or not math.isfinite(translation_residual_norm)
                or not math.isfinite(rotation_residual)
            ):
                raise ValueError("palm tracking residual is non-finite")
            palm_tracking.update(
                {
                    "commanded_tool_local_pose": _matrix_pose_record(
                        commanded_local
                    ),
                    "translation_residual_tool_local_xyz_m": (
                        translation_residual.tolist()
                    ),
                    "translation_residual_norm_m": translation_residual_norm,
                    "rotation_residual_rad": float(rotation_residual),
                    "command_error": None,
                }
            )
        except Exception as exc:
            palm_tracking["command_error"] = f"{type(exc).__name__}: {exc}"

    hand_tracking: dict[str, object] = {
        "policy_action_step": policy_step,
        "hand_joint_ids": None,
        "commanded_20d_rad": None,
        "measured_20d_rad": None,
        "measured_minus_commanded_20d_rad": None,
        "maximum_abs_residual_rad": None,
        "rms_residual_rad": None,
        "command_error": action_error,
        "measurement_error": None,
        "residual_error": None,
    }
    commanded_hand: np.ndarray | None = None
    if action is not None:
        try:
            commanded_hand = np.asarray(action[14:34], dtype=np.float64)
            if commanded_hand.shape != (20,):
                raise ValueError(
                    f"expected 20D right-hand command, got {commanded_hand.shape}"
                )
            if not _finite(commanded_hand):
                raise ValueError("right-hand command contains non-finite values")
            hand_tracking["commanded_20d_rad"] = commanded_hand.tolist()
            hand_tracking["command_error"] = None
        except Exception as exc:
            commanded_hand = None
            hand_tracking["command_error"] = f"{type(exc).__name__}: {exc}"

    measured_hand: np.ndarray | None = None
    try:
        joint_ids = [
            int(value)
            for value in raw._hand_joint_ids.get("right", ())
        ]
        if len(joint_ids) != 20:
            raise ValueError(
                f"expected 20 right hand_joint_ids, got {len(joint_ids)}"
            )
        if raw._scene is None:
            raise RuntimeError("right_robot scene is unavailable")
        right_robot = raw._scene["right_robot"]
        values = right_robot.data.joint_pos[0, joint_ids]
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "numpy"):
            values = values.numpy()
        measured_hand = np.asarray(values, dtype=np.float64).reshape(-1)
        if measured_hand.shape != (20,):
            raise ValueError(
                f"expected 20D measured right hand, got {measured_hand.shape}"
            )
        if not _finite(measured_hand):
            raise ValueError("measured right hand contains non-finite values")
        hand_tracking["hand_joint_ids"] = joint_ids
        hand_tracking["measured_20d_rad"] = measured_hand.tolist()
    except Exception as exc:
        measured_hand = None
        hand_tracking["measurement_error"] = f"{type(exc).__name__}: {exc}"

    if commanded_hand is not None and measured_hand is not None:
        residual = measured_hand - commanded_hand
        maximum_abs_residual = float(np.max(np.abs(residual)))
        rms_residual = float(np.sqrt(np.mean(residual * residual)))
        if (
            _finite(residual)
            and math.isfinite(maximum_abs_residual)
            and math.isfinite(rms_residual)
        ):
            hand_tracking.update(
                {
                    "measured_minus_commanded_20d_rad": residual.tolist(),
                    "maximum_abs_residual_rad": maximum_abs_residual,
                    "rms_residual_rad": rms_residual,
                }
            )
        else:
            hand_tracking["residual_error"] = (
                "ValueError: right-hand tracking residual is non-finite"
            )
    return palm_tracking, hand_tracking


def _right_arm_control_telemetry(raw: object) -> dict[str, object]:
    """Report the post-limiter/IK/drive chain for the right seven-axis arm."""

    result: dict[str, object] = {
        "trace_env_step": None,
        "policy_action_step": None,
        "physics_backend": str(getattr(raw, "_physics_backend", "")),
        "arm_ik_backend": str(getattr(raw, "_arm_ik_backend", "")),
        "haorui_ik_enabled": getattr(raw, "_haorui_ik_enabled", None) is True,
        "ik_command_type": None,
        "ik_status": None,
        "requested_pose_wxyz": None,
        "current_pose_before_ik_wxyz": None,
        "admitted_pose_wxyz": None,
        "pre_joint_limit_target_7d_rad": None,
        "previous_applied_target_7d_rad": None,
        "post_joint_limit_target_7d_rad": None,
        "final_command_target_7d_rad": None,
        "sim_drive_target_7d_rad": None,
        "measured_joint_position_7d_rad": None,
        "measured_joint_velocity_7d_radps": None,
        "arm_joint_ids": None,
        "requested_to_admitted_translation_m": None,
        "requested_to_admitted_rotation_rad": None,
        "admitted_pose_step_translation_m": None,
        "admitted_pose_step_rotation_rad": None,
        "maximum_joint_limiter_correction_rad": None,
        "maximum_applied_joint_step_rad": None,
        "maximum_final_command_step_rad": None,
        "maximum_drive_buffer_delta_rad": None,
        "error": None,
    }

    def vector(value: object, size: int, label: str) -> np.ndarray:
        current = value
        if hasattr(current, "detach"):
            current = current.detach()
        if hasattr(current, "cpu"):
            current = current.cpu()
        if hasattr(current, "numpy"):
            current = current.numpy()
        array = np.asarray(current, dtype=np.float64)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        array = array.reshape(-1)
        if array.shape != (size,):
            raise ValueError(f"{label}: expected {size} values, got {array.shape}")
        if not _finite(array):
            raise ValueError(f"{label}: non-finite values")
        return array

    try:
        command_types = getattr(raw, "_ik_command_types")
        result["ik_command_type"] = str(command_types["right"])
        traces = getattr(raw, "_as019_arm_control_trace")
        trace = traces["right"]
        if not isinstance(trace, dict) or not trace:
            raise ValueError("right arm control trace is unavailable")
        result["trace_env_step"] = int(trace["env_step"])
        result["policy_action_step"] = int(trace["policy_action_step"])
        result["ik_status"] = str(trace["status"])

        requested = vector(trace["requested_pose_wxyz"], 7, "requested_pose")
        current = vector(
            trace["current_pose_before_ik_wxyz"],
            7,
            "current_pose_before_ik",
        )
        admitted = vector(trace["admitted_pose_wxyz"], 7, "admitted_pose")
        pre_joint = vector(
            trace["pre_joint_limit_target_7d_rad"],
            7,
            "pre_joint_limit_target",
        )
        previous_joint = vector(
            trace["previous_applied_target_7d_rad"],
            7,
            "previous_applied_target",
        )
        post_joint = vector(
            trace["post_joint_limit_target_7d_rad"],
            7,
            "post_joint_limit_target",
        )
        result["requested_pose_wxyz"] = requested.tolist()
        result["current_pose_before_ik_wxyz"] = current.tolist()
        result["admitted_pose_wxyz"] = admitted.tolist()
        result["pre_joint_limit_target_7d_rad"] = pre_joint.tolist()
        result["previous_applied_target_7d_rad"] = previous_joint.tolist()
        result["post_joint_limit_target_7d_rad"] = post_joint.tolist()

        requested_matrix = _pose_wxyz_matrix(requested)
        current_matrix = _pose_wxyz_matrix(current)
        admitted_matrix = _pose_wxyz_matrix(admitted)
        result["requested_to_admitted_translation_m"] = float(
            np.linalg.norm(admitted_matrix[:3, 3] - requested_matrix[:3, 3])
        )
        requested_rotation_delta = (
            requested_matrix[:3, :3].T @ admitted_matrix[:3, :3]
        )
        result["requested_to_admitted_rotation_rad"] = float(
            math.acos(
                float(
                    np.clip(
                        (np.trace(requested_rotation_delta) - 1.0) * 0.5,
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        result["admitted_pose_step_translation_m"] = float(
            np.linalg.norm(admitted_matrix[:3, 3] - current_matrix[:3, 3])
        )
        admitted_rotation_delta = (
            current_matrix[:3, :3].T @ admitted_matrix[:3, :3]
        )
        result["admitted_pose_step_rotation_rad"] = float(
            math.acos(
                float(
                    np.clip(
                        (np.trace(admitted_rotation_delta) - 1.0) * 0.5,
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        result["maximum_joint_limiter_correction_rad"] = float(
            np.max(np.abs(post_joint - pre_joint))
        )
        result["maximum_applied_joint_step_rad"] = float(
            np.max(np.abs(post_joint - previous_joint))
        )

        joint_ids = [int(index) for index in raw._arm_joint_ids["right"]]
        if len(joint_ids) != 7:
            raise ValueError(f"expected 7 right arm joint ids, got {len(joint_ids)}")
        result["arm_joint_ids"] = joint_ids
        robot = raw._scene["right_robot"]
        final_command = vector(
            raw._joint_targets["right"][:, joint_ids],
            7,
            "final_command_target",
        )
        sim_drive = vector(
            robot.data.joint_pos_target[:, joint_ids],
            7,
            "sim_drive_target",
        )
        measured = vector(
            robot.data.joint_pos[:, joint_ids],
            7,
            "measured_joint_position",
        )
        velocity = vector(
            robot.data.joint_vel[:, joint_ids],
            7,
            "measured_joint_velocity",
        )
        result["final_command_target_7d_rad"] = final_command.tolist()
        result["sim_drive_target_7d_rad"] = sim_drive.tolist()
        result["measured_joint_position_7d_rad"] = measured.tolist()
        result["measured_joint_velocity_7d_radps"] = velocity.tolist()
        result["maximum_drive_buffer_delta_rad"] = float(
            np.max(np.abs(sim_drive - final_command))
        )
        result["maximum_final_command_step_rad"] = float(
            np.max(np.abs(final_command - previous_joint))
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _automatic_arrival_open_proof(
    arrival: object,
    *,
    expected_policy_step: int,
) -> dict[str, object]:
    """Validate that an open automatic gate came from a stable acquire pose."""

    reasons: list[str] = []
    if not isinstance(arrival, dict):
        return {"pass": False, "reasons": ["missing"]}
    if arrival.get("gate") != "acquire":
        reasons.append("gate_not_acquire")
    if arrival.get("open") is not True:
        reasons.append("not_open")
    if arrival.get("within_tolerance") is not True:
        reasons.append("not_within_tolerance")
    if arrival.get("timed_out") is not False:
        reasons.append("timed_out_or_missing")

    arrived_step = _strict_snapshot_int(arrival.get("arrived_step"))
    sample_step = _strict_snapshot_int(arrival.get("sample_env_step"))
    streak = _strict_snapshot_int(arrival.get("streak"))
    required = _strict_snapshot_int(arrival.get("required_streak"))
    start_step = _strict_snapshot_int(arrival.get("start_step"))
    deadline_step = _strict_snapshot_int(arrival.get("deadline_step"))
    position_error = _json_number(arrival.get("position_error_m"))
    position_tolerance = _json_number(arrival.get("position_tolerance_m"))
    rotation_error = _json_number(arrival.get("rotation_error_rad"))
    rotation_tolerance = _json_number(arrival.get("rotation_tolerance_rad"))
    if arrived_step is None:
        reasons.append("arrived_step_invalid")
    if sample_step is None:
        reasons.append("sample_env_step_invalid")
    if streak is None:
        reasons.append("streak_invalid")
    if required is None or required <= 0:
        reasons.append("required_streak_invalid")
    elif streak is not None and streak != required:
        reasons.append("streak_not_exact")
    if start_step is None or deadline_step is None:
        reasons.append("deadline_lineage_invalid")
    elif deadline_step < start_step:
        reasons.append("deadline_before_start")
    if (
        position_error is None
        or position_tolerance is None
        or position_error < 0.0
        or position_tolerance < 0.0
        or position_error > position_tolerance
    ):
        reasons.append("position_outside_tolerance")
    if (
        rotation_error is None
        or rotation_tolerance is None
        or rotation_error < 0.0
        or rotation_tolerance < 0.0
        or rotation_error > rotation_tolerance
    ):
        reasons.append("rotation_outside_tolerance")
    if arrived_step is not None and sample_step is not None:
        if arrived_step != sample_step:
            reasons.append("arrival_sample_mismatch")
        if sample_step > expected_policy_step:
            reasons.append("sample_from_future")
    if (
        arrived_step is not None
        and start_step is not None
        and deadline_step is not None
        and not (start_step <= arrived_step <= deadline_step)
    ):
        reasons.append("arrival_outside_deadline")
    if arrived_step is not None and arrived_step > expected_policy_step:
        reasons.append("arrival_from_future")
    return {"pass": not reasons, "reasons": reasons}


def _telemetry_validity_gate(snapshot: dict[str, object]) -> dict[str, object]:
    reasons: list[str] = []

    def require_finite(
        label: str,
        value: object,
        shape: tuple[int, ...],
    ) -> None:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            reasons.append(f"{label}:not_numeric")
            return
        if array.shape != shape:
            reasons.append(f"{label}:shape_{array.shape}_expected_{shape}")
        elif not _finite(array):
            reasons.append(f"{label}:non_finite")

    try:
        env_step = int(snapshot["env_step"])
        trial_step = int(snapshot["trial_step"])
    except (KeyError, TypeError, ValueError, OverflowError):
        env_step = -1
        trial_step = -1
        reasons.append("step:missing_or_invalid")
    if trial_step != env_step:
        reasons.append("step:trial_env_mismatch")
    expected_policy_step = env_step - 1

    actual_pose = snapshot.get("right_palm_tool_local_pose")
    if not isinstance(actual_pose, dict):
        reasons.append("right_palm_tool_local_pose:missing")
    else:
        require_finite(
            "right_palm_tool_local_pose.position_m",
            actual_pose.get("position_m"),
            (3,),
        )
        require_finite(
            "right_palm_tool_local_pose.rotation_matrix",
            actual_pose.get("rotation_matrix"),
            (3, 3),
        )

    palm = snapshot.get("right_palm_command_tracking")
    if not isinstance(palm, dict):
        reasons.append("right_palm_command_tracking:missing")
    else:
        if palm.get("policy_action_step") != expected_policy_step:
            reasons.append("right_palm_command_tracking:policy_step_mismatch")
        if palm.get("command_error") is not None:
            reasons.append("right_palm_command_tracking:command_error")
        commanded_pose = palm.get("commanded_tool_local_pose")
        if not isinstance(commanded_pose, dict):
            reasons.append("right_palm_command_tracking:commanded_pose_missing")
        else:
            require_finite(
                "right_palm_command_tracking.commanded_position_m",
                commanded_pose.get("position_m"),
                (3,),
            )
            require_finite(
                "right_palm_command_tracking.commanded_rotation_matrix",
                commanded_pose.get("rotation_matrix"),
                (3, 3),
            )
        require_finite(
            "right_palm_command_tracking.translation_residual_xyz_m",
            palm.get("translation_residual_tool_local_xyz_m"),
            (3,),
        )
        require_finite(
            "right_palm_command_tracking.translation_residual_norm_m",
            palm.get("translation_residual_norm_m"),
            (),
        )
        require_finite(
            "right_palm_command_tracking.rotation_residual_rad",
            palm.get("rotation_residual_rad"),
            (),
        )

    hand = snapshot.get("right_hand_command_tracking")
    if not isinstance(hand, dict):
        reasons.append("right_hand_command_tracking:missing")
    else:
        if hand.get("policy_action_step") != expected_policy_step:
            reasons.append("right_hand_command_tracking:policy_step_mismatch")
        for error_name in ("command_error", "measurement_error", "residual_error"):
            if hand.get(error_name) is not None:
                reasons.append(f"right_hand_command_tracking:{error_name}")
        joint_ids = hand.get("hand_joint_ids")
        if not (
            isinstance(joint_ids, list)
            and len(joint_ids) == 20
            and all(
                isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                for value in joint_ids
            )
            and len(set(int(value) for value in joint_ids)) == 20
        ):
            reasons.append("right_hand_command_tracking:invalid_hand_joint_ids")
        for field in (
            "commanded_20d_rad",
            "measured_20d_rad",
            "measured_minus_commanded_20d_rad",
        ):
            require_finite(f"right_hand_command_tracking.{field}", hand.get(field), (20,))
        require_finite(
            "right_hand_command_tracking.maximum_abs_residual_rad",
            hand.get("maximum_abs_residual_rad"),
            (),
        )
        require_finite(
            "right_hand_command_tracking.rms_residual_rad",
            hand.get("rms_residual_rad"),
            (),
        )

    arm = snapshot.get("right_arm_control")
    if not isinstance(arm, dict):
        reasons.append("right_arm_control:missing")
    else:
        if arm.get("error") is not None:
            reasons.append("right_arm_control:error")
        if arm.get("trace_env_step") != expected_policy_step:
            reasons.append("right_arm_control:trace_step_mismatch")
        if arm.get("policy_action_step") != expected_policy_step:
            reasons.append("right_arm_control:policy_step_mismatch")
        if arm.get("physics_backend") != "physx":
            reasons.append("right_arm_control:unexpected_physics_backend")
        if arm.get("arm_ik_backend") != "haorui":
            reasons.append("right_arm_control:unexpected_ik_backend")
        if arm.get("haorui_ik_enabled") is not True:
            reasons.append("right_arm_control:haorui_not_enabled")
        if arm.get("ik_command_type") != "pose":
            reasons.append("right_arm_control:unexpected_command_type")
        if arm.get("ik_status") != "haorui":
            reasons.append("right_arm_control:ik_not_solved")
        joint_ids = arm.get("arm_joint_ids")
        if not (
            isinstance(joint_ids, list)
            and len(joint_ids) == 7
            and all(
                isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                for value in joint_ids
            )
            and len(set(int(value) for value in joint_ids)) == 7
        ):
            reasons.append("right_arm_control:invalid_arm_joint_ids")
        for field in (
            "requested_pose_wxyz",
            "current_pose_before_ik_wxyz",
            "admitted_pose_wxyz",
            "pre_joint_limit_target_7d_rad",
            "previous_applied_target_7d_rad",
            "post_joint_limit_target_7d_rad",
            "final_command_target_7d_rad",
            "sim_drive_target_7d_rad",
            "measured_joint_position_7d_rad",
            "measured_joint_velocity_7d_radps",
        ):
            require_finite(f"right_arm_control.{field}", arm.get(field), (7,))
        for field in (
            "requested_to_admitted_translation_m",
            "requested_to_admitted_rotation_rad",
            "admitted_pose_step_translation_m",
            "admitted_pose_step_rotation_rad",
            "maximum_joint_limiter_correction_rad",
            "maximum_applied_joint_step_rad",
            "maximum_final_command_step_rad",
            "maximum_drive_buffer_delta_rad",
        ):
            require_finite(f"right_arm_control.{field}", arm.get(field), ())
        if (
            _json_number(arm.get("admitted_pose_step_translation_m")) is None
            or float(arm["admitted_pose_step_translation_m"]) > 0.035000001
        ):
            reasons.append("right_arm_control:site_step_cap_exceeded")
        if (
            _json_number(arm.get("admitted_pose_step_rotation_rad")) is None
            or float(arm["admitted_pose_step_rotation_rad"])
            > math.radians(20.0) + 1.0e-9
        ):
            reasons.append("right_arm_control:rotation_step_cap_exceeded")
        if (
            _json_number(arm.get("maximum_final_command_step_rad")) is None
            or float(arm["maximum_final_command_step_rad"]) > 0.0300001
        ):
            reasons.append("right_arm_control:joint_step_cap_exceeded")
        if (
            _json_number(arm.get("maximum_drive_buffer_delta_rad")) is None
            or float(arm["maximum_drive_buffer_delta_rad"]) > 1.0e-6
        ):
            reasons.append("right_arm_control:drive_target_mismatch")

    acquire_required = snapshot.get("automatic_acquire_gate_required")
    acquire_open = snapshot.get("automatic_acquire_gate_open")
    if acquire_required is not True:
        reasons.append("automatic_arrival_gate:not_required")
    if not isinstance(acquire_open, (bool, np.bool_)):
        reasons.append("automatic_arrival_gate:open_not_boolean")
    arrival = snapshot.get("automatic_arrival_gate")
    if not isinstance(arrival, dict):
        reasons.append("automatic_arrival_gate:missing")
    else:
        gate_name = arrival.get("gate")
        if gate_name not in {"approach_not_started", "pick", "acquire"}:
            reasons.append("automatic_arrival_gate:unexpected_gate")
        if arrival.get("open") is not bool(acquire_open):
            reasons.append("automatic_arrival_gate:open_mismatch")
        if gate_name in {"pick", "acquire"}:
            require_finite(
                "automatic_arrival_gate.target_pose_wxyz",
                arrival.get("target_pose_wxyz"),
                (7,),
            )
            require_finite(
                "automatic_arrival_gate.actual_pose_wxyz",
                arrival.get("actual_pose_wxyz"),
                (7,),
            )
            for field in (
                "position_error_m",
                "rotation_error_rad",
                "position_tolerance_m",
                "rotation_tolerance_rad",
            ):
                require_finite(
                    f"automatic_arrival_gate.{field}",
                    arrival.get(field),
                    (),
                )
            for field in (
                "streak",
                "required_streak",
                "sample_env_step",
                "start_step",
                "deadline_step",
            ):
                if _strict_snapshot_int(arrival.get(field)) is None:
                    reasons.append(f"automatic_arrival_gate:{field}_invalid")
    if acquire_open is True:
        open_proof = _automatic_arrival_open_proof(
            arrival,
            expected_policy_step=expected_policy_step,
        )
        reasons.extend(
            f"automatic_arrival_gate:{reason}"
            for reason in open_proof["reasons"]
        )
    if (
        bool(snapshot.get("right_semantic_owner", False))
        or bool(snapshot.get("active_right_wrench", False))
        or int(snapshot.get("stage", 0)) > 0
    ) and acquire_open is not True:
        reasons.append("automatic_arrival_gate:ownership_before_open")

    if snapshot.get("contact_query_valid") is not True:
        reasons.append("aggregate_contact_query:invalid")
    if snapshot.get("contact_query_error") not in (None, ""):
        reasons.append("aggregate_contact_query:error")

    wrench_sample = snapshot.get("right_wrench_sample")
    if not isinstance(wrench_sample, dict):
        reasons.append("right_wrench_sample:missing")
    elif wrench_sample.get("right_active") is True and wrench_sample.get(
        "pass"
    ) is not True:
        reasons.append("right_wrench_sample:invalid_active_gate")

    table_safety = snapshot.get("table_safety")
    if not isinstance(table_safety, dict):
        reasons.append("table_safety:missing")
    elif table_safety.get("pass") is not True:
        reasons.append("table_safety:unsafe_or_invalid")

    fingers = snapshot.get("right_fingers")
    expected_fingers = {f"finger{index}" for index in range(1, 6)}
    if not isinstance(fingers, dict) or set(fingers) != expected_fingers:
        reasons.append("right_fingers:missing_or_extra_ids")
    else:
        for name in sorted(expected_fingers):
            record = fingers[name]
            if not isinstance(record, dict):
                reasons.append(f"right_fingers.{name}:invalid_record")
                continue
            if record.get("query_valid") is not True:
                reasons.append(f"right_fingers.{name}:query_invalid")
            if record.get("query_error") not in (None, ""):
                reasons.append(f"right_fingers.{name}:query_error")
            if record.get("penetration_violation") is not False:
                reasons.append(f"right_fingers.{name}:penetration_violation")
            for count_name in ("raw_contact_count", "accepted_contact_count"):
                count = record.get(count_name)
                if not (
                    isinstance(count, (int, np.integer))
                    and not isinstance(count, bool)
                    and int(count) >= 0
                ):
                    reasons.append(f"right_fingers.{name}:{count_name}_invalid")
            for scalar_name in ("accepted_force_n", "maximum_penetration_m"):
                require_finite(
                    f"right_fingers.{name}.{scalar_name}",
                    record.get(scalar_name),
                    (),
                )
            for optional_scalar_name in (
                "contact_local_x_m",
                "radial_radius_m",
                "minimum_separation_m",
                "maximum_separation_m",
            ):
                value = record.get(optional_scalar_name)
                if value is not None:
                    require_finite(
                        f"right_fingers.{name}.{optional_scalar_name}",
                        value,
                        (),
                    )
            radial_yz = record.get("radial_yz_m")
            if radial_yz is not None:
                require_finite(
                    f"right_fingers.{name}.radial_yz_m",
                    radial_yz,
                    (2,),
                )

    return {
        "pass": not reasons,
        "expected_policy_action_step": expected_policy_step,
        "failure_count": len(reasons),
        "reasons": reasons,
    }


def _positive_telemetry_snapshot(
    raw: object,
    runtime: object,
    *,
    trial_step: int,
    sample_reason: list[str],
) -> dict[str, object]:
    tool_inverse = np.linalg.inv(
        np.asarray(raw._tool_matrix(), dtype=np.float64).reshape(4, 4)
    )
    palm_local = tool_inverse @ np.asarray(
        raw._palm_matrix("right"), dtype=np.float64
    ).reshape(4, 4)
    palm_tracking, hand_tracking = _right_command_tracking_telemetry(
        raw,
        tool_inverse=tool_inverse,
        actual_palm_local=palm_local,
    )
    arm_control = _right_arm_control_telemetry(raw)
    metrics = raw._as019_last_metrics
    snapshot: dict[str, object] = {
        "trial_step": trial_step,
        "env_step": int(raw.env_step),
        "stage": int(raw._as019_stage),
        "phase": int(raw._as019_automatic_phase),
        "automatic_acquire_gate_required": bool(
            getattr(raw, "_as019_automatic_acquire_gate_required", False)
        ),
        "automatic_acquire_gate_open": bool(
            getattr(raw, "_as019_automatic_acquire_gate_open", False)
        ),
        "automatic_arrival_gate": dict(
            getattr(raw, "_as019_automatic_arrival_gate", {})
        ),
        "right_semantic_owner": bool(
            "right" in getattr(raw, "_as019_assist_anchor", {})
        ),
        "sample_reason": sample_reason,
        "right_palm_tool_local_pose": _matrix_pose_record(palm_local),
        "right_palm_command_tracking": palm_tracking,
        "right_hand_command_tracking": hand_tracking,
        "right_arm_control": arm_control,
        "right_tail_physical": bool(metrics.get("right_tail_physical", False)),
        "active_right_wrench": bool(
            "right" in tuple(raw._as019_active_wrench_owners)
        ),
        "contact_query_valid": bool(metrics.get("contact_query_valid", False)),
        "contact_query_error": str(metrics.get("contact_query_error", "")),
        "right_wrench_sample": _right_wrench_sample_gate(raw),
        "table_safety": _table_safety_sample_gate(metrics),
        "right_tail_aggregate": {
            "thumb_force_n": float(metrics.get("right_thumb_tail_force", 0.0)),
            "opposing_force_n": float(
                metrics.get("right_opposing_tail_force", 0.0)
            ),
            "contact_local_x_m": _json_number(
                metrics.get("right_contact_local_x")
            ),
            "radial_cosine": _json_number(
                metrics.get("right_contact_radial_cosine")
            ),
            "radial_span_m": _json_number(
                metrics.get("right_contact_radial_span_m")
            ),
            "minimum_separation_m": _json_number(
                metrics.get("right_contact_min_separation_m")
            ),
            "maximum_separation_m": _json_number(
                metrics.get("right_contact_max_separation_m")
            ),
            "penetration_violation": bool(
                metrics.get("right_contact_penetration_violation", False)
            ),
        },
        "right_fingers": _right_finger_contact_telemetry(raw, runtime),
    }
    snapshot["validity"] = _telemetry_validity_gate(snapshot)
    return snapshot


def main() -> int:
    args = _parser().parse_args()
    if args.stage0_max_steps <= 0:
        raise SystemExit("--stage0-max-steps must be positive")
    if args.positive_max_steps is not None and args.positive_max_steps <= 0:
        raise SystemExit("--positive-max-steps must be positive")
    if (
        not math.isfinite(args.close_watchdog_seconds)
        or args.close_watchdog_seconds < 0.0
    ):
        raise SystemExit("--close-watchdog-seconds must be finite and non-negative")
    close_watchdog_seconds = (
        None
        if args.close_watchdog_seconds == 0.0
        else float(args.close_watchdog_seconds)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "as019_acceptance_validation.json"
    started = time.time()
    env = None
    execution_mode = (
        "camera_stability"
        if args.camera_stability_only
        else "stage0_canary"
        if args.stage0_canary_seed is not None
        else "outcomes"
        if args.outcomes_only
        else "full"
    )
    summary: dict[str, object] = {
        "schema_version": 3,
        "task": "t22_pass_screwdriver",
        "execution_mode": execution_mode,
        "task_complete": False,
        "checkpoint": None,
        "purpose": (
            "single-seed diagnostic of stage-0 physical right-tail acquisition"
            if args.stage0_canary_seed is not None
            else (
                "20 randomized resets, 10 positive and 10 negative trials, "
                "three-camera 30-frame gate, and 10-second stability gate"
            )
        ),
        "image_observations_requested": args.stage0_canary_seed is None,
        "compatibility_aliases": {
            "terminal_reason": "shutdown_terminal_reason",
            "reset_trials[].table_safety": "reset_trials[].table_clearance",
            "positive_trials[].initial_table_safety": (
                "positive_trials[].initial_table_clearance"
            ),
        },
    }
    exit_code = 1
    full_positive_budget: dict[str, object] | None = None
    try:
        automatic_policy_gate = _ambient_automatic_policy_gate()
        summary["ambient_automatic_policy_gate"] = automatic_policy_gate
        if not automatic_policy_gate["pass"]:
            raise RuntimeError(
                f"{_AUTOMATIC_POLICY_ENV} enables an ambient automatic wrapper; "
                "the validator requires exactly one explicitly constructed wrapper"
            )

        arm_ik_environment_gate = _required_arm_ik_environment_gate()
        summary["arm_ik_environment_gate"] = arm_ik_environment_gate
        if not arm_ik_environment_gate["pass"]:
            raise RuntimeError(
                "AS019 validation requires "
                f"{_ARM_IK_BACKEND_ENV}={_REQUIRED_ARM_IK_BACKEND}; "
                f"reason={arm_ik_environment_gate['reason']}"
            )

        content_fingerprint = _validation_content_fingerprint()
        checkpoint_gate = _expected_checkpoint_gate(content_fingerprint)
        summary["content_fingerprint"] = content_fingerprint
        summary["checkpoint_gate"] = checkpoint_gate
        summary["checkpoint"] = checkpoint_gate["actual_sha256"]
        if not checkpoint_gate["pass"]:
            raise RuntimeError(
                f"AS019 validation checkpoint rejected: {checkpoint_gate['reason']}"
            )

        if args.stage0_canary_seed is None and not args.camera_stability_only:
            full_positive_budget = _automatic_positive_max_steps(
                Path(__file__).with_name("task.yaml"),
                cli_override=args.positive_max_steps,
            )
        summary["automatic_positive_budget"] = full_positive_budget

        from handson.tasks.t22_pass_screwdriver.config import (
            TaskConfig,
        )
        from handson.sim.isaaclab import (
            as019_screwdriver_handover_env as as019_runtime,
        )
        from handson.sim.isaaclab.as019_screwdriver_handover_env import (
            AS019AutomaticPolicyWrapper,
            _OUTWARD_COS,
        )
        from taskdev.tasks.bench320_as019_screwdriver_safe_orientation_handover.scene_contract import (
            MARKER_POSITION_M,
            MARKER_ORIENTATION_WXYZ,
            MARKER_RANDOMIZATION_XY_M,
            SCREWDRIVER_FRICTION_RANGE,
            SCREWDRIVER_MASS_KG,
            SCREWDRIVER_MASS_MULTIPLIER_RANGE,
            SCREWDRIVER_ORIENTATION_WXYZ,
            SCREWDRIVER_POSITION_M,
            SCREWDRIVER_RANDOMIZATION_XY_M,
            SCREWDRIVER_RANDOMIZATION_YAW_RAD,
        )

        summary["right_tail_contact_telemetry_contract"] = {
            "force_threshold_n": float(as019_runtime._CONTACT_FORCE_N),
            "maximum_separation_m": float(
                as019_runtime._CONTACT_MAX_SEPARATION_M
            ),
            "maximum_penetration_m": float(
                as019_runtime._CONTACT_MAX_PENETRATION_M
            ),
            "tail_local_x_range_m": [
                float(as019_runtime._RIGHT_TAIL_CONTACT_LOCAL_X_MIN_M),
                float(as019_runtime._RIGHT_TAIL_CONTACT_LOCAL_X_MAX_M),
            ],
        }
        summary["table_safety_contract"] = {
            "support_maximum_separation_m": float(
                as019_runtime._TABLE_SUPPORT_MAX_SEPARATION_M
            ),
            "maximum_penetration_m": float(
                as019_runtime._TABLE_MAX_PENETRATION_M
            ),
            "arm_contact_is_diagnostic_only": True,
            "arm_contact_triggers_failure": False,
            "shallow_hand_table_contact_is_allowed": True,
            "force_gate_n": None,
            "force_used_for_pass": False,
            "evidence_source": "cached_runtime_metrics_only",
        }

        seed_plan = _validation_seed_plan(args.stage0_canary_seed)
        environment_construction_seed = int(
            seed_plan["environment_construction_seed"]
        )
        summary["environment_construction_seed"] = environment_construction_seed
        env = TaskConfig().get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=True,
            randomize_dynamics=True,
            seed=environment_construction_seed,
            launch_isaac=True,
            headless=True,
            image_obs=args.stage0_canary_seed is None,
            render_resolution=(480, 640),
            isaac_device="cuda:0",
        )
        raw = env.unwrapped
        raw._fast_shutdown_on_close = True
        runtime_rate_gate = _runtime_rate_gate(raw)
        summary["runtime_rate_gate"] = runtime_rate_gate
        if not runtime_rate_gate["pass"]:
            raise RuntimeError("AS019 runtime must run at 30 Hz / 120 Hz / 4 substeps")
        runtime_control_gate = _runtime_control_gate(raw)
        summary["runtime_control_gate"] = runtime_control_gate
        if not runtime_control_gate["pass"]:
            raise RuntimeError(
                "AS019 runtime did not retain PhysX + initialized Haorui pose IK"
            )
        preflight_checks = {
            "content_fingerprint_matches_expected": bool(checkpoint_gate["pass"]),
            "ambient_automatic_policy_disabled": bool(
                automatic_policy_gate["pass"]
            ),
            "haorui_new_environment_required": bool(
                arm_ik_environment_gate["pass"]
            ),
            "runtime_rate_30hz_120hz_four_substeps": bool(runtime_rate_gate["pass"]),
            "runtime_physx_haorui_pose_ik": bool(runtime_control_gate["pass"]),
        }
        summary["preflight_checks"] = preflight_checks
        explicit_reset_seeds: list[int] = []
        summary["reset_sequence"] = {
            "explicit_reset_seeds": explicit_reset_seeds,
            "stage0_expected_single_reset": args.stage0_canary_seed is not None,
        }
        wrench_gate_failures: list[dict[str, object]] = []
        observed_contact_separations: list[float] = []
        wrench_coverage = {
            context: {
                "sampled_steps": 0,
                "right_tail_physical_steps": 0,
                "active_right_wrench_steps": 0,
                "active_and_physical_steps": 0,
                "active_without_physical_steps": 0,
            }
            for context in ("camera_stability", "negative", "positive")
        }
        table_safety_failures: list[dict[str, object]] = []
        table_safety_coverage = {
            context: {
                "sampled_steps": 0,
                "query_invalid_steps": 0,
                "tool_supported_steps": 0,
                "arm_contact_steps": 0,
                "tool_penetration_violation_steps": 0,
                "hand_penetration_violation_steps": 0,
                "maximum_tool_penetration_m": 0.0,
                "maximum_arm_penetration_m": 0.0,
                "maximum_hand_penetration_m": 0.0,
            }
            for context in ("camera_stability", "negative", "positive")
        }

        def audit_right_wrench_gate(context: str) -> None:
            metrics = raw._as019_last_metrics
            sample = _right_wrench_sample_gate(raw)
            right_active = bool(sample["right_active"])
            right_physical = bool(
                sample["snapshot_right_tail_physical"]
            )
            coverage = wrench_coverage[context]
            coverage["sampled_steps"] += 1
            coverage["right_tail_physical_steps"] += int(right_physical)
            coverage["active_right_wrench_steps"] += int(right_active)
            coverage["active_and_physical_steps"] += int(
                right_active and right_physical
            )
            coverage["active_without_physical_steps"] += int(
                right_active and not right_physical
            )
            for key in (
                "right_contact_min_separation_m",
                "right_contact_max_separation_m",
            ):
                value = _json_number(metrics.get(key))
                if value is not None:
                    observed_contact_separations.append(value)
            if right_active and not bool(sample["pass"]):
                wrench_gate_failures.append(
                    {
                        "step": int(raw.env_step),
                        "stage": int(raw._as019_stage),
                        "context": context,
                        "sample": sample,
                        "current_min_separation_m": _json_number(
                            metrics.get("right_contact_min_separation_m")
                        ),
                        "current_max_separation_m": _json_number(
                            metrics.get("right_contact_max_separation_m")
                        ),
                    }
                )

        def record_right_wrench_gate(
            *, require_positive_coverage: bool
        ) -> dict[str, object]:
            record = _right_wrench_gate_result(
                wrench_gate_failures,
                wrench_coverage,
                require_positive_coverage=require_positive_coverage,
            )
            record["minimum_accepted_separation_m"] = (
                min(observed_contact_separations)
                if observed_contact_separations
                else None
            )
            record["maximum_accepted_separation_m"] = (
                max(observed_contact_separations)
                if observed_contact_separations
                else None
            )
            summary["right_wrench_gate"] = record
            return record

        def audit_table_safety(context: str) -> dict[str, object]:
            sample = _table_safety_sample_gate(raw._as019_last_metrics)
            coverage = table_safety_coverage[context]
            coverage["sampled_steps"] += 1
            coverage["query_invalid_steps"] += int(
                not bool(sample["query_valid"])
            )
            coverage["tool_supported_steps"] += int(
                bool(sample["tool_table_supported"])
            )
            coverage["arm_contact_steps"] += int(
                bool(sample["arm_table_physical_contact"])
            )
            coverage["tool_penetration_violation_steps"] += int(
                bool(sample["tool_table_penetration_violation"])
            )
            coverage["hand_penetration_violation_steps"] += int(
                bool(sample["hand_table_penetration_violation"])
            )
            for coverage_key, sample_key in (
                ("maximum_tool_penetration_m", "tool_table_max_penetration_m"),
                ("maximum_arm_penetration_m", "arm_table_max_penetration_m"),
                ("maximum_hand_penetration_m", "hand_table_max_penetration_m"),
            ):
                coverage[coverage_key] = max(
                    float(coverage[coverage_key]), float(sample[sample_key])
                )
            if not bool(sample["pass"]):
                table_safety_failures.append(
                    {
                        "step": int(raw.env_step),
                        "stage": int(raw._as019_stage),
                        "context": context,
                        "sample": sample,
                    }
                )
            return sample

        def record_table_safety_gate() -> dict[str, object]:
            record = {
                "pass": not table_safety_failures,
                "failure_count": len(table_safety_failures),
                "failures": table_safety_failures,
                "coverage_by_context": table_safety_coverage,
                "force_gate_n": None,
                "force_used_for_pass": False,
            }
            summary["table_safety_gate"] = record
            return record

        # Cameras must exist when the Isaac scene is constructed.  Keep image
        # observations disabled during physics steps and render the three real
        # sensors explicitly afterward.  This follows the production split
        # between simulation and camera capture and avoids coupling a camera
        # annotator wait into ``env.step``.
        raw.image_obs = False
        if seed_plan["initial_reset_seed"] is not None:
            _observation, _reset_info = _reset_with_audit(
                env,
                int(seed_plan["initial_reset_seed"]),
                explicit_reset_seeds,
            )
        diagnostic_only = bool(
            args.outcomes_only or args.stage0_canary_seed is not None
        )
        camera_frame_count = 0 if diagnostic_only else 30
        stability_step_count = 0 if diagnostic_only else 300
        camera_records: list[dict[str, object]] = []
        zero = np.zeros(env.action_space.shape, dtype=np.float32)
        stability_max_linear = 0.0
        stability_max_angular = 0.0
        stability_nan_count = 0
        stability_exploding_count = 0
        stability_terminated = False
        for frame_index in range(camera_frame_count):
            _observation, _reward, terminated, truncated, _info = env.step(zero)
            audit_right_wrench_gate("camera_stability")
            audit_table_safety("camera_stability")
            stability_terminated |= bool(terminated or truncated)
            rendered = raw.render()
            images = dict(
                zip(("random_camera", "wrist_left", "wrist_right"), rendered)
            )
            for camera_name in ("random_camera", "wrist_left", "wrist_right"):
                frame = np.asarray(images.get(camera_name))
                record = {
                    "frame": frame_index,
                    "camera": camera_name,
                    "shape": list(frame.shape),
                    "mean_rgb": float(frame.mean()) if frame.size else 0.0,
                    "fully_black": bool(not frame.size or np.all(frame == 0)),
                    "finite": _finite(frame),
                }
                camera_records.append(record)
            velocity = np.asarray(raw._tool_velocity(), dtype=np.float64)
            stability_nan_count += int(not _finite(velocity))
            if _finite(velocity):
                linear = float(np.linalg.norm(velocity[:3]))
                angular = float(np.linalg.norm(velocity[3:]))
                stability_max_linear = max(stability_max_linear, linear)
                stability_max_angular = max(stability_max_angular, angular)
                stability_exploding_count += int(linear > 5.0 or angular > 20.0)
        for _index in range(camera_frame_count, stability_step_count):
            _observation, _reward, terminated, truncated, _info = env.step(zero)
            audit_right_wrench_gate("camera_stability")
            audit_table_safety("camera_stability")
            stability_terminated |= bool(terminated or truncated)
            velocity = np.asarray(raw._tool_velocity(), dtype=np.float64)
            stability_nan_count += int(not _finite(velocity))
            if _finite(velocity):
                linear = float(np.linalg.norm(velocity[:3]))
                angular = float(np.linalg.norm(velocity[3:]))
                stability_max_linear = max(stability_max_linear, linear)
                stability_max_angular = max(stability_max_angular, angular)
                stability_exploding_count += int(linear > 5.0 or angular > 20.0)

        camera_pass = bool(
            diagnostic_only
            or (
                len(camera_records) == 90
                and all(record["finite"] for record in camera_records)
                and not any(record["fully_black"] for record in camera_records)
                and min(float(record["mean_rgb"]) for record in camera_records) >= 30.0
                and all(
                    len(record["shape"]) == 3
                    and int(record["shape"][0]) >= 480
                    and int(record["shape"][1]) >= 640
                    and int(record["shape"][2]) == 3
                    for record in camera_records
                )
            )
        )
        stability_pass = bool(
            diagnostic_only
            or (
                not stability_terminated
                and stability_nan_count == 0
                and stability_exploding_count == 0
                and not raw._as019_failure_reason
            )
        )
        summary["camera"] = {
            "pass": None if diagnostic_only else camera_pass,
            "frames_per_camera": camera_frame_count,
            "records": camera_records,
        }
        summary["stability"] = {
            "pass": None if diagnostic_only else stability_pass,
            "sim_seconds": stability_step_count / 30.0,
            "steps": stability_step_count,
            "nan_count": stability_nan_count,
            "exploding_body_count": stability_exploding_count,
            "max_linear_speed_mps": stability_max_linear,
            "max_angular_speed_radps": stability_max_angular,
            "failure_reason": str(raw._as019_failure_reason),
        }
        if args.camera_stability_only:
            wrench_gate = record_right_wrench_gate(
                require_positive_coverage=False
            )
            table_safety_gate = record_table_safety_gate()
            summary["checks"] = {
                **preflight_checks,
                "camera_30_frames_three_views": camera_pass,
                "stability_10_seconds": stability_pass,
                "right_wrench_invariant_zero_violations": bool(
                    wrench_gate["invariant_pass"]
                ),
                "table_safety_zero_violations": bool(
                    table_safety_gate["pass"]
                ),
            }
            summary["validation_pass"] = bool(all(summary["checks"].values()))
            exit_code = 0 if summary["validation_pass"] else 2
            raise _ValidationFinished

        reset_trials: list[dict[str, object]] = []
        negative_trials: list[dict[str, object]] = []
        canonical_tool_xy = np.asarray(SCREWDRIVER_POSITION_M[:2], dtype=np.float64)
        canonical_marker_xy = np.asarray(MARKER_POSITION_M[:2], dtype=np.float64)
        canonical_yaw = _yaw_from_wxyz(
            np.asarray(SCREWDRIVER_ORIENTATION_WXYZ, dtype=np.float64)
        )
        reset_contract_tool_pose = np.asarray(
            (*SCREWDRIVER_POSITION_M, *SCREWDRIVER_ORIENTATION_WXYZ),
            dtype=np.float32,
        ).astype(np.float64)
        reset_contract_marker_pose = np.asarray(
            (*MARKER_POSITION_M, *MARKER_ORIENTATION_WXYZ),
            dtype=np.float32,
        ).astype(np.float64)
        reset_table_expected_gap_m = float(
            as019_runtime._tool_table_gap_m(
                reset_contract_tool_pose,
                reset_contract_marker_pose,
            )
        )
        reset_table_gap_tolerance_m = float(
            abs(np.spacing(np.float32(SCREWDRIVER_POSITION_M[2])))
            + abs(np.spacing(np.float32(MARKER_POSITION_M[2])))
        )
        summary["reset_table_clearance_contract"] = {
            "expected_analytic_gap_m": reset_table_expected_gap_m,
            "analytic_gap_tolerance_m": reset_table_gap_tolerance_m,
            "source": "float32_authored_tool_and_marker_pose",
            "physical_support_required": False,
            "placement_physical_support_requirement_unchanged": True,
        }
        mass_min = SCREWDRIVER_MASS_KG * SCREWDRIVER_MASS_MULTIPLIER_RANGE[0]
        mass_max = SCREWDRIVER_MASS_KG * SCREWDRIVER_MASS_MULTIPLIER_RANGE[1]
        reset_seeds = (
            () if args.stage0_canary_seed is not None else range(9100, 9120)
        )
        for ordinal, seed in enumerate(reset_seeds):
            _observation, info = _reset_with_audit(
                env,
                seed,
                explicit_reset_seeds,
            )
            pose = np.asarray(raw._peg_ori_pose, dtype=np.float64).copy()
            marker = np.asarray(raw._socket_ori_pose, dtype=np.float64).copy()
            metrics = _as019_metrics_from_reset_info(info)
            yaw = _yaw_from_wxyz(pose[3:7])
            relative_yaw = math.atan2(
                math.sin(yaw - canonical_yaw),
                math.cos(yaw - canonical_yaw),
            )
            reset_table_clearance = _reset_table_clearance_gate(
                metrics,
                expected_gap_m=reset_table_expected_gap_m,
                tolerance_m=reset_table_gap_tolerance_m,
            )
            trial_pass = bool(
                _finite(pose)
                and _finite(marker)
                and np.all(
                    np.abs(pose[:2] - canonical_tool_xy)
                    <= SCREWDRIVER_RANDOMIZATION_XY_M + 1.0e-9
                )
                and abs(relative_yaw)
                <= SCREWDRIVER_RANDOMIZATION_YAW_RAD + 1.0e-9
                and np.all(
                    np.abs(marker[:2] - canonical_marker_xy)
                    <= MARKER_RANDOMIZATION_XY_M + 1.0e-9
                )
                and mass_min <= raw._as019_mass_kg <= mass_max
                and SCREWDRIVER_FRICTION_RANGE[0]
                <= raw._as019_friction
                <= SCREWDRIVER_FRICTION_RANGE[1]
                and reset_table_clearance["pass"]
                and float(metrics.get("right_force", 0.0)) < 0.015
                and float(metrics.get("left_force", 0.0)) < 0.015
                and not bool(info.get("succeed", False))
                and not raw._as019_failure_reason
                and raw._as019_event_steps.tolist() == [-1, -1, -1, -1, -1]
            )
            reset_record = {
                "ordinal": ordinal,
                "seed": seed,
                "pass": trial_pass,
                "tool_pose": pose.tolist(),
                "marker_pose": marker.tolist(),
                "yaw_rad": yaw,
                "relative_yaw_rad": relative_yaw,
                "mass_kg": float(raw._as019_mass_kg),
                "friction": float(raw._as019_friction),
                "table_clearance": reset_table_clearance,
                "table_safety": reset_table_clearance,
                "right_force_n": float(metrics.get("right_force", 0.0)),
                "left_force_n": float(metrics.get("left_force", 0.0)),
                "sampling_attempts": int(raw._as019_reset_attempts),
                "failure_reason": str(raw._as019_failure_reason),
            }
            reset_trials.append(reset_record)
            if ordinal < 10:
                false_positive = False
                terminated_early = False
                table_safety_start = dict(table_safety_coverage["negative"])
                for _ in range(10):
                    _observation, _reward, terminated, truncated, step_info = env.step(zero)
                    audit_right_wrench_gate("negative")
                    audit_table_safety("negative")
                    false_positive |= bool(step_info.get("succeed", False))
                    terminated_early |= bool(terminated or truncated)
                table_safety_end = table_safety_coverage["negative"]
                negative_table_safety = {
                    key: int(table_safety_end[key])
                    - int(table_safety_start[key])
                    for key in (
                        "sampled_steps",
                        "query_invalid_steps",
                        "arm_contact_steps",
                        "tool_penetration_violation_steps",
                        "hand_penetration_violation_steps",
                    )
                }
                negative_trials.append(
                    {
                        "seed": seed,
                        "pass": bool(
                            not false_positive
                            and not terminated_early
                            and not raw._as019_success
                            and raw._as019_event_steps.tolist()
                            == [-1, -1, -1, -1, -1]
                            and negative_table_safety["query_invalid_steps"] == 0
                            and negative_table_safety["arm_contact_steps"] == 0
                            and negative_table_safety[
                                "tool_penetration_violation_steps"
                            ]
                            == 0
                            and negative_table_safety[
                                "hand_penetration_violation_steps"
                            ]
                            == 0
                        ),
                        "false_positive": false_positive,
                        "terminated": terminated_early,
                        "event_steps": raw._as019_event_steps.tolist(),
                        "failure_reason": str(raw._as019_failure_reason),
                        "table_safety": negative_table_safety,
                    }
                )

        reset_pass = (
            None
            if args.stage0_canary_seed is not None
            else bool(
                len(reset_trials) == 20
                and all(trial["pass"] for trial in reset_trials)
            )
        )
        negative_pass = (
            None
            if args.stage0_canary_seed is not None
            else bool(
                len(negative_trials) == 10
                and all(trial["pass"] for trial in negative_trials)
            )
        )
        summary["reset"] = {
            "pass": reset_pass,
            "trials": reset_trials,
            "failure_count": sum(not bool(trial["pass"]) for trial in reset_trials),
        }
        summary["negative_trials"] = {
            "pass": negative_pass,
            "trials": negative_trials,
            "false_positive_count": sum(
                bool(trial["false_positive"]) for trial in negative_trials
            ),
        }

        stage0_canary = args.stage0_canary_seed is not None
        positive_seeds = (
            (int(args.stage0_canary_seed),)
            if stage0_canary
            else tuple(range(103, 113))
        )
        if stage0_canary:
            maximum_trial_steps = int(args.stage0_max_steps)
        else:
            if full_positive_budget is None:
                raise RuntimeError("full positive validation budget was not loaded")
            maximum_trial_steps = int(full_positive_budget["max_steps"])
        positive_trials: list[dict[str, object]] = []
        for seed in positive_seeds:
            _observation, reset_info = _reset_with_audit(
                env,
                seed,
                explicit_reset_seeds,
            )
            initial_tool_pose = np.asarray(raw._peg_ori_pose, dtype=np.float64).copy()
            initial_marker_pose = np.asarray(raw._socket_ori_pose, dtype=np.float64).copy()
            initial_table_clearance = _reset_table_clearance_gate(
                _as019_metrics_from_reset_info(reset_info),
                expected_gap_m=reset_table_expected_gap_m,
                tolerance_m=reset_table_gap_tolerance_m,
            )
            policy = AS019AutomaticPolicyWrapper(env.env, raw)
            success = False
            terminated = False
            truncated = False
            min_outward_after_entry = 1.0
            max_tip_palm_force_in_radius = 0.0
            maximum_stage = int(raw._as019_stage)
            phase_counts: dict[str, int] = {}
            stage_counts: dict[str, int] = {}
            right_tail_physical_steps = 0
            telemetry_trace: list[dict[str, object]] = []
            last_signature: tuple[object, ...] | None = None
            last_snapshot_step = 0
            wrench_start = dict(wrench_coverage["positive"])
            table_safety_start = dict(table_safety_coverage["positive"])
            stage0_exit_step: int | None = None
            stage0_confirmation: dict[str, object] = {
                "required": stage0_canary,
                "executed": False,
                "pass": False if stage0_canary else None,
            }
            confirmation_complete = False
            control_steps = 0
            loop_step_count = maximum_trial_steps + int(stage0_canary)
            for _index in range(loop_step_count):
                if (
                    stage0_canary
                    and _index >= maximum_trial_steps
                    and stage0_exit_step is None
                ):
                    break
                action = policy.automatic_action()
                if action.shape != (54,) or not _finite(action):
                    raise RuntimeError(f"invalid AS019 automatic action for seed {seed}")
                _observation, _reward, terminated, truncated, info = env.step(action)
                audit_right_wrench_gate("positive")
                audit_table_safety("positive")
                control_steps = _index + 1
                metrics = raw._as019_last_metrics
                stage = int(raw._as019_stage)
                phase = int(raw._as019_automatic_phase)
                maximum_stage = max(maximum_stage, stage)
                stage_counts[str(stage)] = stage_counts.get(str(stage), 0) + 1
                phase_counts[str(phase)] = phase_counts.get(str(phase), 0) + 1
                right_tail_physical = bool(
                    metrics.get("right_tail_physical", False)
                )
                right_tail_physical_steps += int(right_tail_physical)
                confirmation_step = bool(
                    stage0_canary
                    and stage0_exit_step is not None
                    and control_steps > stage0_exit_step
                )
                signature = (
                    stage,
                    phase,
                    right_tail_physical,
                    "right" in tuple(raw._as019_active_wrench_owners),
                    bool(metrics.get("contact_query_valid", False)),
                    float(metrics.get("right_thumb_tail_force", 0.0)) > 0.0,
                    float(metrics.get("right_opposing_tail_force", 0.0)) > 0.0,
                )
                reasons: list[str] = []
                if stage0_canary:
                    reasons.append("stage0_every_step")
                if _index == 0:
                    reasons.append("first")
                if _index % 30 == 0:
                    reasons.append("periodic_30_steps")
                if signature != last_signature:
                    reasons.append("state_change")
                if confirmation_step:
                    reasons.append("post_stage0_confirmation")
                if reasons:
                    telemetry_trace.append(
                        _positive_telemetry_snapshot(
                            raw,
                            as019_runtime,
                            trial_step=control_steps,
                            sample_reason=reasons,
                        )
                    )
                    last_snapshot_step = control_steps
                last_signature = signature
                if stage > 0:
                    min_outward_after_entry = min(
                        min_outward_after_entry,
                        float(metrics.get("outward_alignment", -1.0)),
                    )
                if float(metrics.get("tip_left_palm_distance", 1.0)) <= 0.05:
                    max_tip_palm_force_in_radius = max(
                        max_tip_palm_force_in_radius,
                        float(metrics.get("left_palm_force", 0.0)),
                    )
                success = bool(info.get("succeed", False))
                stage0_exit_step, observed_confirmation_step = (
                    _stage0_confirmation_status(
                        stage=stage,
                        control_step=control_steps,
                        stage0_exit_step=stage0_exit_step,
                    )
                )
                confirmation_complete = bool(
                    stage0_canary and observed_confirmation_step
                )
                if stage0_canary:
                    if confirmation_complete:
                        wrench_sample = _right_wrench_sample_gate(raw)
                        arrival_proof = _automatic_arrival_open_proof(
                            raw._as019_automatic_arrival_gate,
                            expected_policy_step=int(raw.env_step) - 1,
                        )
                        active_right_wrench = bool(
                            wrench_sample["right_active"]
                        )
                        same_generation_wrench_gate = bool(
                            wrench_sample["pass"]
                        )
                        stage0_confirmation = {
                            "required": True,
                            "executed": True,
                            "control_step": control_steps,
                            "post_step_right_tail_physical": (
                                right_tail_physical
                            ),
                            "active_right_wrench": active_right_wrench,
                            "same_generation_wrench_gate": (
                                same_generation_wrench_gate
                            ),
                            "automatic_acquire_gate_open": bool(
                                raw._as019_automatic_acquire_gate_open
                            ),
                            "automatic_arrival_gate_proof": arrival_proof,
                            "wrench_sample": wrench_sample,
                            "pass": bool(
                                right_tail_physical
                                and active_right_wrench
                                and same_generation_wrench_gate
                                and raw._as019_automatic_acquire_gate_open
                                and arrival_proof["pass"]
                            ),
                        }
                if (
                    success
                    or terminated
                    or truncated
                    or confirmation_complete
                ):
                    break
            if control_steps > 0 and last_snapshot_step != control_steps:
                telemetry_trace.append(
                    _positive_telemetry_snapshot(
                        raw,
                        as019_runtime,
                        trial_step=control_steps,
                        sample_reason=["final"],
                    )
                )
            terminal_status = _validation_trial_terminal_status(
                stage0_canary=stage0_canary,
                control_steps=control_steps,
                maximum_trial_steps=maximum_trial_steps,
                stage0_exit_step=stage0_exit_step,
                confirmation_complete=confirmation_complete,
                success=success,
                terminated=bool(terminated),
                truncated=bool(truncated),
                failure_reason=str(raw._as019_failure_reason),
            )
            events = raw._as019_event_steps.tolist()
            ordered = bool(
                all(value >= 0 for value in events)
                and all(left < right for left, right in zip(events, events[1:]))
            )
            wrench_end = wrench_coverage["positive"]
            trial_wrench_coverage = {
                key: int(wrench_end[key]) - int(wrench_start[key])
                for key in wrench_start
            }
            table_safety_end = table_safety_coverage["positive"]
            trial_table_safety = {
                key: int(table_safety_end[key])
                - int(table_safety_start[key])
                for key in (
                    "sampled_steps",
                    "query_invalid_steps",
                    "tool_supported_steps",
                    "arm_contact_steps",
                    "tool_penetration_violation_steps",
                    "hand_penetration_violation_steps",
                )
            }
            trial_table_safety["pass"] = bool(
                trial_table_safety["query_invalid_steps"] == 0
                and trial_table_safety["arm_contact_steps"] == 0
                and trial_table_safety[
                    "tool_penetration_violation_steps"
                ]
                == 0
                and trial_table_safety[
                    "hand_penetration_violation_steps"
                ]
                == 0
            )
            stage0_acquired = bool(
                events[0] >= 0
                and stage0_exit_step is not None
                and right_tail_physical_steps > 0
            )
            telemetry_failures = [
                {
                    "trial_step": int(snapshot["trial_step"]),
                    "reasons": list(snapshot["validity"]["reasons"]),
                }
                for snapshot in telemetry_trace
                if not bool(snapshot["validity"]["pass"])
            ]
            telemetry_gate = {
                "required": stage0_canary,
                "pass": (
                    bool(
                        control_steps > 0
                        and len(telemetry_trace) == control_steps
                        and not telemetry_failures
                    )
                    if stage0_canary
                    else None
                ),
                "expected_sample_count": control_steps if stage0_canary else None,
                "sample_count": len(telemetry_trace),
                "failure_count": len(telemetry_failures),
                "failures": telemetry_failures,
            }
            stage0_reset_gate = (
                _stage0_reset_gate(
                    requested_seed=int(args.stage0_canary_seed),
                    explicit_reset_seeds=explicit_reset_seeds,
                )
                if stage0_canary
                else None
            )
            if stage0_canary:
                summary["reset_sequence"]["stage0_gate"] = stage0_reset_gate
            if stage0_canary:
                trial_pass = bool(
                    stage0_acquired
                    and stage0_confirmation["pass"]
                    and telemetry_gate["pass"]
                    and stage0_reset_gate["pass"]
                    and trial_wrench_coverage["active_right_wrench_steps"] > 0
                    and trial_wrench_coverage["active_without_physical_steps"] == 0
                    and trial_table_safety["pass"]
                    and initial_table_clearance["pass"]
                    and not raw._as019_failure_reason
                    and not terminated
                    and not truncated
                    and raw._trajectory_state_write_count == 0
                )
            else:
                trial_pass = bool(
                    success
                    and ordered
                    and raw._as019_stage == 5
                    and not raw._as019_failure_reason
                    and min_outward_after_entry >= _OUTWARD_COS
                    and max_tip_palm_force_in_radius < 0.015
                    and not raw._as019_assist_anchor
                    and raw._trajectory_state_write_count == 0
                    and trial_table_safety["pass"]
                    and initial_table_clearance["pass"]
                )
            record = {
                "seed": seed,
                "acceptance_scope": (
                    "stage0_right_tail_acquisition"
                    if stage0_canary
                    else "full_ordered_handover"
                ),
                "pass": trial_pass,
                "stage0_acquired": stage0_acquired,
                "success": success,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                **terminal_status,
                "steps": int(raw.env_step),
                "trial_control_steps": control_steps,
                "maximum_trial_steps": maximum_trial_steps,
                "maximum_stage": maximum_stage,
                "stage_counts": stage_counts,
                "phase_counts": phase_counts,
                "event_steps": events,
                "ordered_events": ordered,
                "failure_reason": str(raw._as019_failure_reason),
                "min_outward_after_entry": min_outward_after_entry,
                "max_tip_palm_force_in_radius_n": max_tip_palm_force_in_radius,
                "trajectory_state_write_count": int(raw._trajectory_state_write_count),
                "initial_tool_pose": initial_tool_pose.tolist(),
                "initial_marker_pose": initial_marker_pose.tolist(),
                "initial_table_clearance": initial_table_clearance,
                "initial_table_safety": initial_table_clearance,
                "mass_kg": float(raw._as019_mass_kg),
                "friction": float(raw._as019_friction),
                "right_tail_physical_steps": right_tail_physical_steps,
                "stage0_exit_control_step": stage0_exit_step,
                "stage0_confirmation": stage0_confirmation,
                "stage0_reset_gate": stage0_reset_gate,
                "right_wrench_coverage": trial_wrench_coverage,
                "table_safety": trial_table_safety,
                "telemetry_gate": telemetry_gate,
                "telemetry_trace_sampling": (
                    "every control step"
                    if stage0_canary
                    else "first/final, every 30 steps, and state changes"
                ),
                "telemetry_trace": telemetry_trace,
            }
            positive_trials.append(record)
            print(
                "AS019_POSITIVE_TRIAL="
                + json.dumps(
                    {
                        "seed": seed,
                        "pass": trial_pass,
                        "stage0_acquired": stage0_acquired,
                        "maximum_stage": maximum_stage,
                        "stage_counts": stage_counts,
                        "phase_counts": phase_counts,
                        "event_steps": events,
                        "failure_reason": str(raw._as019_failure_reason),
                        **terminal_status,
                        "stage0_confirmation": stage0_confirmation,
                        "stage0_reset_gate": stage0_reset_gate,
                        "right_wrench_coverage": trial_wrench_coverage,
                        "table_safety": trial_table_safety,
                        "telemetry_gate": telemetry_gate,
                        "telemetry_sample_count": len(telemetry_trace),
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )

        positive_pass = (
            None
            if stage0_canary
            else bool(
                len(positive_trials) == 10
                and all(trial["pass"] for trial in positive_trials)
            )
        )
        summary["positive_trials"] = {
            "pass": positive_pass,
            "maximum_steps_per_trial": maximum_trial_steps,
            "trials": positive_trials,
            "full_task_outcome_evaluated": not stage0_canary,
            "false_negative_count": (
                sum(not bool(trial["pass"]) for trial in positive_trials)
                if not stage0_canary
                else None
            ),
        }
        wrench_gate = record_right_wrench_gate(
            require_positive_coverage=True
        )
        table_safety_gate = record_table_safety_gate()
        if stage0_canary:
            summary["validation_terminal_reason"] = str(
                positive_trials[0]["terminal_reason"]
            )
            canary_pass = bool(
                positive_trials[0]["pass"]
                and wrench_gate["invariant_pass"]
                and wrench_gate["positive_coverage_pass"]
                and table_safety_gate["pass"]
            )
            summary["stage0_canary"] = {
                "pass": canary_pass,
                "seed": int(args.stage0_canary_seed),
                "max_steps": int(args.stage0_max_steps),
                "reset_gate": positive_trials[0]["stage0_reset_gate"],
                "telemetry_gate": positive_trials[0]["telemetry_gate"],
                "table_safety": positive_trials[0]["table_safety"],
                "full_task_acceptance_evaluated": False,
            }
            summary["checks"] = {
                **preflight_checks,
                "stage0_right_tail_acquired": bool(
                    positive_trials[0]["stage0_acquired"]
                ),
                "stage0_post_exit_confirmation": bool(
                    positive_trials[0]["stage0_confirmation"]["pass"]
                ),
                "stage0_exactly_one_requested_reset": bool(
                    positive_trials[0]["stage0_reset_gate"]["pass"]
                ),
                "stage0_every_step_telemetry_valid": bool(
                    positive_trials[0]["telemetry_gate"]["pass"]
                ),
                "right_wrench_invariant_zero_violations": bool(
                    wrench_gate["invariant_pass"]
                ),
                "right_wrench_positive_coverage_nonzero": bool(
                    wrench_gate["positive_coverage_pass"]
                ),
                "table_safety_zero_violations": bool(
                    table_safety_gate["pass"]
                ),
                "zero_trajectory_state_writes": bool(
                    positive_trials[0]["trajectory_state_write_count"] == 0
                ),
            }
            summary["validation_pass"] = canary_pass
            summary["full_acceptance_pass"] = None
            exit_code = 0 if canary_pass else 2
            raise _ValidationFinished

        outcome_checks = {
            **preflight_checks,
            "reset_20_zero_failures": bool(reset_pass),
            "negative_10_zero_false_positives": bool(negative_pass),
            "positive_10_zero_false_negatives": bool(positive_pass),
            "right_wrench_invariant_zero_violations": bool(
                wrench_gate["invariant_pass"]
            ),
            "right_wrench_positive_coverage_nonzero": bool(
                wrench_gate["positive_coverage_pass"]
            ),
            "table_safety_zero_violations": bool(
                table_safety_gate["pass"]
            ),
        }
        summary["checks"] = (
            outcome_checks
            if args.outcomes_only
            else {
                "camera_30_frames_three_views": camera_pass,
                "stability_10_seconds": stability_pass,
                **outcome_checks,
            }
        )
        summary["validation_pass"] = bool(all(summary["checks"].values()))
        exit_code = 0 if summary["validation_pass"] else 2
    except _ValidationFinished:
        pass
    except BaseException as exc:
        summary["exception"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        exit_code = 1
    finally:
        summary["elapsed_seconds"] = time.time() - started
        _close_environment_with_watchdog(
            env,
            summary_path=summary_path,
            summary=summary,
            planned_exit_code=exit_code,
            timeout_seconds=close_watchdog_seconds,
        )
        print("AS019_ACCEPTANCE_SUMMARY=" + str(summary_path), flush=True)
        print(
            "AS019_ACCEPTANCE_JSON="
            + json.dumps(summary, sort_keys=True, allow_nan=False),
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
