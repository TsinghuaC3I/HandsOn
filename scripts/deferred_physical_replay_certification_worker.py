#!/usr/bin/env python3
"""Batch PhysX replay certification after collection has already stopped."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "handson"
for _entry in (str(PACKAGE_ROOT), str(REPO_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from handson.data.physical_replay_certification import (  # noqa: E402
    CERTIFICATION_MODE,
    CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD,
    DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS,
    FRESH_PROCESS_ACCEPTANCE_NAME,
    FRESH_PROCESS_REPORT_DIRECTORY,
    PHYSICAL_REPLAY_INVALID_OUTCOME_CODE,
    PHYSICAL_REPLAY_OUTCOME_VERSION,
    PHYSICAL_REPLAY_REVIEW_INVALID,
    PHYSICAL_REPLAY_REVIEW_REQUIRED,
    PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE,
    PHYSICAL_REPLAY_REVIEW_VALID,
    PHYSICAL_REPLAY_VALID_OUTCOME_CODE,
    PORTABLE_SOURCE_IDENTITIES,
    PERSISTENT_RECORDER_LIFECYCLE,
    REPORT_NAME,
    REPORT_FORMAT,
    REPORT_VERSION,
    capture_certified_inputs,
    certification_report_blocker,
    deferred_physical_replay_acceptance_upload_blocker,
    local_physical_replay_report_blocker,
)
from handson.runtime.collection_support import (  # noqa: E402
    BC011_COLLECTION_ANGLE_GUIDANCE_ENV,
)


CERTIFIER = REPO_ROOT / "scripts" / "certify_physx_replay.py"
DEFERRED_MODE = "deferred_post_collection"
ACCEPTANCE_FORMAT = "handson.deferred_physical_replay_acceptance"
SESSION_CERTIFICATION_SCOPE = "ordered_collection_session"
SESSION_PUBLICATION_COMMIT_DIRECTORY = ".physical_replay_session_commits"
SESSION_PUBLICATION_COMMIT_FORMAT = "handson.physical_replay_session_publication_commit"
DEFAULT_CERTIFICATION_TIMEOUT_SECONDS = 1800.0
DEFAULT_TERMINATION_GRACE_SECONDS = 5.0
GPU_STARTUP_MAX_ATTEMPTS = 2
PENDING_DIRECTORY_NAME = ".physical_replay_pending"
WORKER_STATE_DIRECTORY_NAME = ".physical_replay_certification_state"
WORKER_STATE_FORMAT = "handson.deferred_physical_replay_worker_state"
PHYSICS_ONLY_EXECUTION_MODE = "physics_only_source_scene_no_step_render_unthrottled"
E01_POSTPROCESS_STATUS_NAME = "e01_postprocess.json"
SOURCE_RECORDER_PID_ENV = "HANDSON_DEFERRED_SOURCE_RECORDER_PID"
BATCH_EXECUTION_LOCK_NAME = ".physical_replay_certification_execution.lock"
BATCH_EXECUTION_LOCK_PATH = (
    REPO_ROOT / "demos" / "collection_sessions" / BATCH_EXECUTION_LOCK_NAME
)
_DETERMINISTIC_ENVIRONMENT_NAMES = {
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "GARMENTLAB_ASSET_ROOT",
    "HAND",
    "MKL_NUM_THREADS",
    "NVIDIA_TF32_OVERRIDE",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
}
# The deferred worker is forked directly by the one long-lived recorder.  Keep
# its original PID before that parent exits and this process is re-parented.
SOURCE_RECORDER_PID = os.getppid()

_EXPLICIT_GPU_CONTEXT_POISON_MARKERS = (
    "fetching gpu narrowphase failed! 700",
    "synchronizing gpu narrowphase failed! 700",
    "cuda error 700:",
    "cudaerrorillegaladdress",
    "physx internal cuda error. simulation cannot continue! error code 700",
    "physx cannot start gpu simulation because of previous cuda errors! "
    "error code 700",
    "cuda/physx backend is poisoned",
    "nvrm: xid",
    "nvidia xid error",
)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _output_scoped_gpu_gate_path(output_root: Path) -> Path:
    """Match the collection UI's cross-worktree, output-scoped GPU gate."""

    runtime_base_text = str(
        os.environ.get("HANDSON_COLLECTION_RUNTIME_DIR") or ""
    ).strip()
    if runtime_base_text:
        runtime_base = Path(runtime_base_text).expanduser()
    else:
        xdg_runtime = str(os.environ.get("XDG_RUNTIME_DIR") or "").strip()
        runtime_base = (
            Path(xdg_runtime or f"/run/user/{os.getuid()}") / "handson-collection"
        )
    output_scope = hashlib.sha256(
        str(output_root.expanduser().resolve()).encode("utf-8")
    ).hexdigest()[:16]
    return runtime_base / output_scope / ".deferred_physical_replay_gpu_launch.lock"


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                dict(payload),
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Durably publish exactly the bytes hashed by the worker."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _prepare_fresh_report_output(
    demo_dir: Path,
    run_index: int,
) -> tuple[Path, Path]:
    """Remove every path that could otherwise masquerade as this run.

    Recovery archives the previous review before launching this worker.  A
    repeated review starts its run numbering at one again, so both the latest
    report and the corresponding immutable-run path must be absent before the
    new certifier process starts.
    """

    report_path = demo_dir / REPORT_NAME
    reports_dir = demo_dir / FRESH_PROCESS_REPORT_DIRECTORY
    if reports_dir.exists() and (
        reports_dir.is_symlink() or not reports_dir.is_dir()
    ):
        raise RuntimeError(
            f"physical replay report directory is not a regular directory: "
            f"{reports_dir}"
        )
    reports_dir.mkdir(parents=True, exist_ok=True)
    proof_path = reports_dir / f"run_{run_index:03d}.json"
    for stale_path in (report_path, proof_path):
        try:
            stale_path.unlink()
        except FileNotFoundError:
            pass
    return report_path, proof_path


def _read_regular_file_bytes(path: Path) -> bytes | None:
    """Read one regular file without ever following a report symlink."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError:
        return None
    finally:
        os.close(descriptor)


def _archive_fresh_report(
    demo_dir: Path,
    report_path: Path,
    proof_path: Path,
) -> tuple[dict[str, Any], str, str]:
    """Archive and hash only bytes created after the pre-launch cleanup."""

    report_bytes = _read_regular_file_bytes(report_path)
    if report_bytes is None:
        return {}, "", ""
    _write_bytes_atomic(proof_path, report_bytes)
    try:
        value = json.loads(report_bytes)
    except (UnicodeError, json.JSONDecodeError):
        value = {}
    report = value if isinstance(value, dict) else {}
    return (
        report,
        proof_path.relative_to(demo_dir).as_posix(),
        hashlib.sha256(report_bytes).hexdigest(),
    )


def _worker_state_path(demo_dir: Path) -> Path:
    """Return the launcher-owned state path for one pending demo."""

    return demo_dir.parent / WORKER_STATE_DIRECTORY_NAME / f"{demo_dir.name}.json"


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _physical_invalid_outcome_evidence(
    run: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return immutable evidence only for a completed physical replay.

    A certifier process failure is not a physical outcome.  In particular,
    schema/preflight errors, environment construction errors, watchdog
    timeouts, and publication failures do not contain a completed trajectory
    and must remain reviewable.  The versioned invalid code is issued only
    when the archived report proves every recorded action was executed and
    records an explicit state/outcome mismatch.
    """

    report_status = run.get("physical_report_status", run.get("status"))
    report_reason = str(run.get("physical_report_reason", run.get("reason") or ""))
    steps_expected = run.get("steps_expected")
    steps_executed = run.get("steps_executed")
    source_identity = run.get("source_identity")
    if (
        report_status != "failed"
        or run.get("timed_out") is True
        or run.get("error_type") not in {None, ""}
        or run.get("report_format") != REPORT_FORMAT
        or run.get("report_version") != REPORT_VERSION
        or run.get("certification_mode") != CERTIFICATION_MODE
        or run.get("execution_mode") != PHYSICS_ONLY_EXECUTION_MODE
        or run.get("portable") is not True
        or source_identity not in PORTABLE_SOURCE_IDENTITIES
        or type(steps_expected) is not int
        or steps_expected <= 0
        or type(steps_executed) is not int
        or steps_executed != steps_expected
        or not _is_sha256(run.get("report_sha256"))
    ):
        return None

    overlay = run.get(CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD)
    if overlay is not None:
        if not isinstance(overlay, Mapping):
            return None
        overlay_digest = overlay.get("digest_sha256")
        if not _is_sha256(overlay_digest):
            return None
        overlay_payload = {
            str(key): value for key, value in overlay.items() if key != "digest_sha256"
        }
        expected_overlay_digest = hashlib.sha256(
            json.dumps(
                overlay_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        reviewer_paths = run.get("source_reviewer_upgrade_paths")
        if (
            overlay_digest != expected_overlay_digest
            or overlay.get("source_open_mode") != "read_only"
            or overlay.get("source_files_mutated") is not False
            or not isinstance(reviewer_paths, list)
            or not {
                "scripts/certify_physx_replay.py",
                "handson/handson/sim/isaaclab/physx_ground_truth.py",
            }.issubset(set(reviewer_paths))
        ):
            return None

    first_divergence = run.get("first_divergence")
    first_outcome_mismatch = run.get("first_outcome_mismatch")
    evidence_kind: str | None = None
    evidence_detail: dict[str, Any] | None = None
    if (
        report_reason == "physical_state_diverged"
        and isinstance(first_divergence, Mapping)
        and type(first_divergence.get("action_index")) is int
        and 0 <= int(first_divergence["action_index"]) < steps_expected
    ):
        evidence_kind = "physical_state_divergence"
        evidence_detail = dict(first_divergence)
    elif (
        report_reason == "transition_outcome_diverged"
        and isinstance(first_outcome_mismatch, Mapping)
        and type(first_outcome_mismatch.get("action_index")) is int
        and 0 <= int(first_outcome_mismatch["action_index"]) < steps_expected
    ):
        evidence_kind = "transition_outcome_divergence"
        evidence_detail = dict(first_outcome_mismatch)
    elif (
        report_reason == "final_info_succeed_is_false"
        and run.get("task_success_replay") is False
        and run.get("final_succeed") is False
        and run.get("trace_match") is True
        and run.get("transition_outcomes_match") is True
    ):
        evidence_kind = "final_task_outcome_failure"
        evidence_detail = {
            "task_success_replay": False,
            "final_succeed": False,
            "trace_match": True,
            "transition_outcomes_match": True,
        }
    if evidence_kind is None or evidence_detail is None:
        return None

    evidence = {
        "format": "handson.physical_replay_invalid_outcome_evidence",
        "version": PHYSICAL_REPLAY_OUTCOME_VERSION,
        "kind": evidence_kind,
        "reason": report_reason,
        "run": run.get("run"),
        "report": run.get("report"),
        "report_sha256": run.get("report_sha256"),
        "steps_expected": steps_expected,
        "steps_executed": steps_executed,
        "detail": evidence_detail,
        "portable": True,
        "source_identity": source_identity,
        "source_snapshot_commit": run.get("source_snapshot_commit"),
        "source_reviewer_upgrade_paths": run.get("source_reviewer_upgrade_paths"),
    }
    if isinstance(overlay, Mapping):
        evidence[CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD] = dict(overlay)
        evidence["contact_signature_overlay_digest_sha256"] = overlay.get(
            "digest_sha256"
        )
    return evidence


def _review_outcome_from_runs(
    runs: list[dict[str, Any]],
    *,
    physically_valid: bool,
) -> dict[str, Any]:
    if physically_valid:
        return {
            "physical_replay_outcome_version": (PHYSICAL_REPLAY_OUTCOME_VERSION),
            "physical_replay_outcome_code": (PHYSICAL_REPLAY_VALID_OUTCOME_CODE),
            "review_status": PHYSICAL_REPLAY_REVIEW_VALID,
        }

    review_required = {
        "physical_replay_outcome_version": (PHYSICAL_REPLAY_OUTCOME_VERSION),
        "physical_replay_outcome_code": (
            PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE
        ),
        "review_status": PHYSICAL_REPLAY_REVIEW_REQUIRED,
    }

    def _run_passed_without_physical_regressions(run: Mapping[str, Any]) -> bool:
        if run.get("status") != "passed":
            return False
        if run.get("task_success_replay") is not True:
            return False
        if run.get("trace_match") is not True:
            return False
        if run.get("transition_outcomes_match") is not True:
            return False
        if run.get("final_succeed") is not True:
            return False
        return True

    if not runs:
        return review_required
    for run in runs:
        evidence = _physical_invalid_outcome_evidence(run)
        if evidence is not None:
            return {
                "physical_replay_outcome_version": (PHYSICAL_REPLAY_OUTCOME_VERSION),
                "physical_replay_outcome_code": (PHYSICAL_REPLAY_INVALID_OUTCOME_CODE),
                "review_status": PHYSICAL_REPLAY_REVIEW_INVALID,
                "outcome_evidence": evidence,
            }
        if not _run_passed_without_physical_regressions(run):
            return review_required
    return review_required


def _result_episode_outcome(
    result: Mapping[str, Any],
    demo_dir: Path,
) -> dict[str, Any]:
    rows = result.get("episode_outcomes")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if row.get("demo_name") == demo_dir.name:
                return dict(row)
    if result.get("status") == "passed":
        return _review_outcome_from_runs([], physically_valid=True)
    return _review_outcome_from_runs([], physically_valid=False)


def _write_worker_terminal_states(
    demo_dirs: list[Path],
    *,
    status: str,
    result: Mapping[str, Any] | None,
    error: str,
) -> None:
    """Durably replace every launcher ``running`` marker with a terminal one.

    The state files live beside the pending directory rather than inside a
    demo.  Their paths therefore remain valid after successful publication
    moves the demo itself into the output root.
    """

    if status not in {"passed", "failed"}:
        raise ValueError(f"invalid terminal worker status: {status!r}")
    completed_at = _now_iso()
    outcome = dict(result) if isinstance(result, Mapping) else {}
    physical_status = str(outcome.get("physical_certification_status") or "unknown")
    publication_status = str(outcome.get("publication_status") or "not_attempted")
    failure_stage = outcome.get("failure_stage")
    if status == "failed" and not failure_stage:
        failure_stage = "worker_exception"
    for demo_dir in demo_dirs:
        state_path = _worker_state_path(demo_dir)
        state = _read_json(state_path)
        episode_outcome = _result_episode_outcome(outcome, demo_dir)
        review_status = str(
            episode_outcome.get("review_status") or PHYSICAL_REPLAY_REVIEW_REQUIRED
        )
        outcome_code = str(
            episode_outcome.get("physical_replay_outcome_code")
            or PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE
        )
        if review_status == PHYSICAL_REPLAY_REVIEW_INVALID:
            persisted_status = "invalid"
            terminal = True
        elif review_status == PHYSICAL_REPLAY_REVIEW_VALID:
            persisted_status = "passed"
            terminal = True
        else:
            persisted_status = PHYSICAL_REPLAY_REVIEW_REQUIRED
            terminal = False
        state.update(
            {
                "format": WORKER_STATE_FORMAT,
                "version": 1,
                "status": persisted_status,
                "pid": None,
                "terminal": terminal,
                "updated_at": completed_at,
                "completed_at": completed_at,
                "worker_exit_code": 0 if status == "passed" else 1,
                "physical_certification_status": physical_status,
                "publication_status": publication_status,
                "failure_stage": failure_stage,
                "physical_replay_outcome_version": (PHYSICAL_REPLAY_OUTCOME_VERSION),
                "physical_replay_outcome_code": outcome_code,
                "review_status": review_status,
            }
        )
        evidence = episode_outcome.get("outcome_evidence")
        if isinstance(evidence, Mapping):
            state["outcome_evidence"] = dict(evidence)
        else:
            state.pop("outcome_evidence", None)
        if outcome.get("collection_session_id"):
            state["collection_session_id"] = outcome["collection_session_id"]
        runs = outcome.get("runs")
        if isinstance(runs, list):
            state["physical_certification_runs_completed"] = len(runs)
            state["physical_certification_runs_passed"] = sum(
                1
                for run in runs
                if isinstance(run, Mapping) and run.get("status") == "passed"
            )
        if error:
            state["error"] = error
        else:
            state.pop("error", None)
        _write_json_atomic(state_path, state)


def _certifier_environment(demo_dir: Path) -> dict[str, str]:
    """Rebuild replay-relevant child state from the recorded manifest."""

    manifest = _read_json(demo_dir / "replay_manifest.json")
    environment = manifest.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    recorded = environment.get("variables")
    if not isinstance(recorded, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in recorded.items()
    ):
        raise RuntimeError(f"invalid recorded environment variables in {demo_dir}")

    child_env = os.environ.copy()
    for name in tuple(child_env):
        if name.startswith("HANDSON_") or name in _DETERMINISTIC_ENVIRONMENT_NAMES:
            child_env.pop(name, None)
    child_env.update(recorded)
    child_env.pop(BC011_COLLECTION_ANGLE_GUIDANCE_ENV, None)
    child_env.pop("HANDSON_FRESH_CERTIFIED_COLLECTION_WORKER", None)
    pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PACKAGE_ROOT), str(REPO_ROOT), pythonpath) if part
    )
    return child_env


def _recording_process_contract(demo_dir: Path) -> tuple[str, bool, int]:
    """Validate optional manifest evidence and resolve the source PID."""

    manifest = _read_json(demo_dir / "replay_manifest.json")
    recording = manifest.get("recording")
    recording = recording if isinstance(recording, Mapping) else {}
    if (
        "recorder_lifecycle" in recording
        and recording.get("recorder_lifecycle") != PERSISTENT_RECORDER_LIFECYCLE
    ):
        raise RuntimeError(
            "recorded recorder_lifecycle is inconsistent with deferred "
            "persistent collection"
        )
    if (
        "fresh_scene_replay_required" in recording
        and recording.get("fresh_scene_replay_required") is not True
    ):
        raise RuntimeError(
            "recorded fresh_scene_replay_required must be true for deferred "
            "certification"
        )
    for field in (
        "source_process_reuse_enabled",
        "fresh_certifier_process_per_run_required",
    ):
        if field in recording and recording.get(field) is not True:
            raise RuntimeError(f"recorded {field} must be true")

    possible_pids: list[Any] = [
        recording.get("recorder_pid"),
        recording.get("source_recorder_pid"),
        manifest.get("source_recorder_pid"),
        manifest.get("recorder_pid"),
    ]
    recorder = manifest.get("recorder")
    if isinstance(recorder, Mapping):
        possible_pids.append(recorder.get("pid"))
    declared_pids = [value for value in possible_pids if value is not None]
    if any(type(value) is not int or value <= 0 for value in declared_pids):
        raise RuntimeError("recorded source recorder pid must be a positive integer")
    if len(set(declared_pids)) > 1:
        raise RuntimeError("recorded source recorder pid fields are inconsistent")
    source_pid = int(declared_pids[0]) if declared_pids else SOURCE_RECORDER_PID
    return PERSISTENT_RECORDER_LIFECYCLE, True, source_pid


def _wait_for_postprocess_inputs(
    demo_dir: Path,
    *,
    timeout_seconds: float,
    poll_seconds: float = 1.0,
) -> None:
    """Wait until any queued E/F replay.zarr mutation has finished."""

    status_path = demo_dir / E01_POSTPROCESS_STATUS_NAME
    if not status_path.is_file():
        return
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        status = str(_read_json(status_path).get("status") or "").lower()
        if status in {"complete", "not_requested"}:
            return
        if status in {"failed", "stopped"}:
            raise RuntimeError(
                f"E/F postprocess ended with status {status!r}: {demo_dir}"
            )
        if status not in {"queued", "running"}:
            raise RuntimeError(
                f"E/F postprocess status is invalid ({status!r}): {demo_dir}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(
                "timed out waiting for E/F postprocess to finish before "
                f"physical certification: {demo_dir}"
            )
        time.sleep(min(float(poll_seconds), remaining))


def _wait_for_launching_recorder_exit(
    *,
    timeout_seconds: float,
    poll_seconds: float = 0.05,
) -> None:
    """Do not initialize a fresh Isaac certifier before the source exits."""

    raw_pid = os.environ.get(SOURCE_RECORDER_PID_ENV, "").strip()
    if not raw_pid:
        return
    try:
        source_pid = int(raw_pid)
    except ValueError as exc:
        raise RuntimeError(f"invalid {SOURCE_RECORDER_PID_ENV}={raw_pid!r}") from exc
    if source_pid <= 1:
        raise RuntimeError(f"invalid {SOURCE_RECORDER_PID_ENV}={source_pid}")

    def source_is_running() -> bool:
        try:
            os.kill(source_pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    deadline = time.monotonic() + float(timeout_seconds)
    while source_is_running():
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(
                "source recorder did not exit before fresh certification"
            )
        time.sleep(min(float(poll_seconds), remaining))


def _communicate_with_timeout(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float,
    termination_grace_seconds: float,
) -> tuple[str, bool]:
    """Collect output and kill the entire child session on wall timeout."""

    def as_text(value: str | bytes | None) -> str:
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return value or ""

    try:
        output, _ = process.communicate(timeout=timeout_seconds)
        return as_text(output), False
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        output, _ = process.communicate(timeout=termination_grace_seconds)
    except subprocess.TimeoutExpired:
        # The process-group leader may already have exited while a descendant
        # keeps the output pipe open, so target the original PGID unconditionally.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            output, _ = process.communicate(timeout=termination_grace_seconds)
        except subprocess.TimeoutExpired as exc:
            # Never let a broken inherited pipe turn the worker watchdog into
            # another unbounded wait after SIGKILL.
            output = exc.output or ""
            if process.stdout is not None:
                process.stdout.close()
            process.wait(timeout=termination_grace_seconds)
    return as_text(output), True


def _explicit_gpu_context_poisoned_before_any_report(
    output: str,
    *,
    timed_out: bool,
    report_paths: Sequence[Path],
) -> bool:
    """Recognize only a report-free, explicit CUDA/PhysX context failure.

    A failed process is safe to discard only before it has emitted even one
    episode report.  This keeps a deterministic physical mismatch, an ordinary
    Python failure, and a timeout terminal for the current review attempt.  A
    retry always launches the same recorded CUDA environment in a new process;
    there is deliberately no CPU fallback.
    """

    if timed_out or any(os.path.lexists(path) for path in report_paths):
        return False
    lowered = str(output or "").lower()
    return any(marker in lowered for marker in _EXPLICIT_GPU_CONTEXT_POISON_MARKERS)


def _run_certifier_with_gpu_startup_retry(
    command: Sequence[str],
    *,
    child_environment: Mapping[str, str],
    report_paths: Sequence[Path],
    timeout_seconds: float,
    termination_grace_seconds: float,
) -> dict[str, Any]:
    """Run one logical proof in a bounded number of fresh GPU processes."""

    discarded_attempts: list[dict[str, Any]] = []
    process: subprocess.Popen[str]
    output = ""
    timed_out = False
    started_at = ""
    finished_at = ""
    gpu_context_poisoned = False
    for launch_attempt in range(1, GPU_STARTUP_MAX_ATTEMPTS + 1):
        # A poisoned CUDA context cannot be repaired in-process.  Clear only
        # latest-report paths before launching the same recorded GPU
        # environment in a brand-new process.  There is no CPU fallback.
        for report_path in report_paths:
            try:
                report_path.unlink()
            except FileNotFoundError:
                pass
        started_at = _now_iso()
        process = subprocess.Popen(
            list(command),
            cwd=str(REPO_ROOT),
            env=dict(child_environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        output, timed_out = _communicate_with_timeout(
            process,
            timeout_seconds=timeout_seconds,
            termination_grace_seconds=termination_grace_seconds,
        )
        if output:
            sys.stdout.write(output)
        finished_at = _now_iso()
        gpu_context_poisoned = _explicit_gpu_context_poisoned_before_any_report(
            output,
            timed_out=timed_out,
            report_paths=report_paths,
        )
        if gpu_context_poisoned and launch_attempt < GPU_STARTUP_MAX_ATTEMPTS:
            discarded_attempts.append(
                {
                    "attempt": launch_attempt,
                    "pid": int(process.pid),
                    "returncode": int(process.returncode),
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "timed_out": False,
                    "classification": "explicit_gpu_context_poisoned_before_report",
                    "output_sha256": hashlib.sha256(
                        output.encode("utf-8", errors="replace")
                    ).hexdigest(),
                }
            )
            print(
                "[Deferred PhysX certification worker] explicit CUDA/PhysX "
                f"startup failure in fresh pid={process.pid}; discarding "
                f"attempt {launch_attempt}/{GPU_STARTUP_MAX_ATTEMPTS} and "
                "launching one clean GPU-only process retry",
                flush=True,
            )
            continue
        break

    return {
        "process": process,
        "output": output,
        "timed_out": timed_out,
        "started_at": started_at,
        "finished_at": finished_at,
        "gpu_context_poisoned": gpu_context_poisoned,
        "fresh_process_launch_attempts": len(discarded_attempts) + 1,
        "discarded_gpu_startup_attempts": discarded_attempts,
    }


def _report_outcome_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the minimum report fields needed to prove an invalid outcome."""

    return {
        "report_format": report.get("format"),
        "report_version": report.get("version"),
        "physical_report_status": report.get("status"),
        "physical_report_reason": report.get("reason"),
        "error_type": report.get("error_type"),
        "certification_mode": report.get("certification_mode"),
        "execution_mode": report.get("execution_mode"),
        "steps_expected": report.get("steps_expected"),
        "steps_executed": report.get("steps_executed"),
        "task_success_replay": report.get("task_success_replay"),
        "final_succeed": report.get("final_succeed"),
        "trace_match": report.get("trace_match"),
        "transition_outcomes_match": report.get("transition_outcomes_match"),
        "first_divergence": report.get("first_divergence"),
        "first_outcome_mismatch": report.get("first_outcome_mismatch"),
        "portable": report.get("portable"),
        "source_identity": report.get("source_identity"),
        "source_snapshot_commit": report.get("source_snapshot_commit"),
        "source_reviewer_upgrade_paths": report.get("source_reviewer_upgrade_paths"),
        CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD: report.get(
            CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD
        ),
    }


def _run_single_certification(
    demo_dir: Path,
    run_index: int,
    *,
    timeout_seconds: float = DEFAULT_CERTIFICATION_TIMEOUT_SECONDS,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
) -> dict[str, Any]:
    _recording_process_contract(demo_dir)
    report_path, proof_path = _prepare_fresh_report_output(demo_dir, run_index)

    launch = _run_certifier_with_gpu_startup_retry(
        [
            sys.executable,
            str(CERTIFIER),
            "--demo-dir",
            str(demo_dir),
        ],
        child_environment=_certifier_environment(demo_dir),
        report_paths=[report_path],
        timeout_seconds=timeout_seconds,
        termination_grace_seconds=termination_grace_seconds,
    )
    process = launch["process"]
    timed_out = bool(launch["timed_out"])
    returncode = process.returncode
    report, report_reference, report_sha256 = _archive_fresh_report(
        demo_dir,
        report_path,
        proof_path,
    )
    result = {
        "run": run_index,
        "pid": int(process.pid),
        "returncode": int(returncode),
        "started_at": launch["started_at"],
        "finished_at": launch["finished_at"],
        "status": str(report.get("status") or "invalid"),
        "reason": str(report.get("reason") or ""),
        "report": report_reference,
        "report_sha256": report_sha256,
        "certified_inputs": report.get("certified_inputs"),
        "timed_out": timed_out,
        "fresh_process_launch_attempts": launch[
            "fresh_process_launch_attempts"
        ],
        "discarded_gpu_startup_attempts": launch[
            "discarded_gpu_startup_attempts"
        ],
        **_report_outcome_projection(report),
    }
    blocker = certification_report_blocker(demo_dir, report)
    invalid_report = not report or report.get("status") != "passed"
    if timed_out or returncode != 0 or invalid_report or blocker is not None:
        result["status"] = "failed"
        if timed_out:
            reason = f"certifier timed out after {timeout_seconds:g}s"
        elif launch["gpu_context_poisoned"]:
            reason = (
                "explicit CUDA/PhysX context-poisoning startup failure "
                f"exhausted {GPU_STARTUP_MAX_ATTEMPTS} GPU-only "
                "fresh-process attempts"
            )
        elif returncode != 0:
            reason = (
                f"certifier terminated by signal {-returncode}"
                if returncode < 0
                else f"certifier exited with status {returncode}"
            )
        elif invalid_report:
            reason = str(report.get("reason") or "certifier produced no passed report")
        else:
            reason = str(blocker.get("detail") or blocker.get("code") or blocker)
        result["reason"] = reason
    return result


def _validated_session_context(
    demo_dirs: list[Path],
    *,
    exp_name: str,
) -> dict[str, Any]:
    """Validate and order one persistent-recorder collection session.

    Accepted episode numbers are contiguous, while reset ordinals may contain
    gaps for rejected attempts.  The latter therefore defines physical-history
    order and the former guards against missing accepted episodes.
    """

    if not demo_dirs:
        raise RuntimeError("deferred session contains no demos")
    rows: list[dict[str, Any]] = []
    for raw_demo_dir in demo_dirs:
        demo_dir = Path(raw_demo_dir).expanduser().resolve()
        manifest = _read_json(demo_dir / "replay_manifest.json")
        recording = manifest.get("recording")
        task = manifest.get("task")
        environment = manifest.get("environment")
        if not isinstance(recording, Mapping):
            raise RuntimeError(f"invalid recording manifest in {demo_dir}")
        if not isinstance(task, Mapping) or task.get("name") != exp_name:
            raise RuntimeError(
                f"session task mismatch in {demo_dir}: expected {exp_name!r}"
            )
        if not isinstance(environment, Mapping) or not isinstance(
            environment.get("variables"), Mapping
        ):
            raise RuntimeError(f"invalid recorded environment variables in {demo_dir}")
        metadata = _read_json(demo_dir / "collection_meta.json")
        session_id = metadata.get("collection_session_id")
        episode_index = recording.get("episode_index")
        reset_ordinal = recording.get("episode_reset_ordinal")
        collection_run_seed = recording.get("collection_run_seed")
        certification_scope = recording.get("physical_replay_certification_scope")
        history_pending = recording.get("physical_replay_session_history_pending")
        history_id = recording.get("physical_replay_session_history_id")
        history_path = recording.get(
            "physical_replay_session_history_root_relative_path"
        )
        history_sha256 = recording.get("physical_replay_session_history_sha256")
        history_event_count = recording.get(
            "physical_replay_session_history_event_count"
        )
        accepted_event_ordinal = recording.get(
            "physical_replay_session_history_accepted_event_ordinal"
        )
        certified_through_event_ordinal = recording.get(
            "physical_replay_session_history_certified_through_event_ordinal"
        )
        if not isinstance(session_id, str) or not session_id.strip():
            raise RuntimeError(f"collection_session_id is missing in {demo_dir}")
        if type(episode_index) is not int or episode_index <= 0:
            raise RuntimeError(f"episode_index is invalid in {demo_dir}")
        if type(reset_ordinal) is not int or reset_ordinal < 0:
            raise RuntimeError(f"episode_reset_ordinal is invalid in {demo_dir}")
        if type(collection_run_seed) is not int or collection_run_seed < 0:
            raise RuntimeError(f"collection_run_seed is invalid in {demo_dir}")
        if certification_scope != SESSION_CERTIFICATION_SCOPE:
            raise RuntimeError(
                f"ordered session certification scope is missing in {demo_dir}"
            )
        if history_pending is not False:
            raise RuntimeError(f"PhysX session history is not sealed in {demo_dir}")
        if not isinstance(history_id, str) or history_id != session_id.strip():
            raise RuntimeError(f"PhysX session history id is invalid in {demo_dir}")
        if not isinstance(history_path, str) or not history_path:
            raise RuntimeError(f"PhysX session history path is invalid in {demo_dir}")
        if not isinstance(history_sha256, str) or len(history_sha256) != 64:
            raise RuntimeError(
                f"PhysX session history SHA-256 is invalid in {demo_dir}"
            )
        try:
            int(history_sha256, 16)
        except ValueError as exc:
            raise RuntimeError(
                f"PhysX session history SHA-256 is invalid in {demo_dir}"
            ) from exc
        if type(history_event_count) is not int or history_event_count <= 0:
            raise RuntimeError(
                f"PhysX session history event count is invalid in {demo_dir}"
            )
        if (
            type(accepted_event_ordinal) is not int
            or accepted_event_ordinal <= 0
            or accepted_event_ordinal >= history_event_count
        ):
            raise RuntimeError(
                f"accepted session event ordinal is invalid in {demo_dir}"
            )
        if (
            type(certified_through_event_ordinal) is not int
            or certified_through_event_ordinal < accepted_event_ordinal
            or certified_through_event_ordinal >= history_event_count
        ):
            raise RuntimeError(
                f"certified session event boundary is invalid in {demo_dir}"
            )
        for key in (
            "physical_replay_certification_scope",
            "physical_replay_session_history_pending",
            "physical_replay_session_history_id",
            "physical_replay_session_history_root_relative_path",
            "physical_replay_session_history_sha256",
            "physical_replay_session_history_event_count",
            "physical_replay_session_history_accepted_event_ordinal",
            "physical_replay_session_history_certified_through_event_ordinal",
        ):
            if metadata.get(key) != recording.get(key):
                raise RuntimeError(
                    f"session history binding differs between manifest and "
                    f"metadata at {key} in {demo_dir}"
                )
        lifecycle, fresh_scene, source_pid = _recording_process_contract(demo_dir)
        rows.append(
            {
                "demo_dir": demo_dir,
                "session_id": session_id.strip(),
                "episode_index": int(episode_index),
                "reset_ordinal": int(reset_ordinal),
                "collection_run_seed": int(collection_run_seed),
                "source_recorder_pid": int(source_pid),
                "recorder_lifecycle": lifecycle,
                "fresh_scene_replay_required": bool(fresh_scene),
                "environment_variables": dict(environment["variables"]),
                "history_id": history_id,
                "history_path": history_path,
                "history_sha256": history_sha256.lower(),
                "history_event_count": int(history_event_count),
                "accepted_event_ordinal": int(accepted_event_ordinal),
                "certified_through_event_ordinal": int(certified_through_event_ordinal),
            }
        )

    for field in (
        "session_id",
        "collection_run_seed",
        "source_recorder_pid",
        "recorder_lifecycle",
        "fresh_scene_replay_required",
        "environment_variables",
        "history_id",
        "history_path",
        "history_sha256",
        "history_event_count",
        "certified_through_event_ordinal",
    ):
        values = [row[field] for row in rows]
        if any(value != values[0] for value in values[1:]):
            raise RuntimeError(f"deferred batch mixes collection sessions at {field}")

    rows.sort(key=lambda row: (row["reset_ordinal"], row["episode_index"]))
    expected_episode_indices = list(range(1, len(rows) + 1))
    actual_episode_indices = [row["episode_index"] for row in rows]
    if actual_episode_indices != expected_episode_indices:
        raise RuntimeError(
            "accepted episode_index values must be contiguous in reset "
            f"order: expected={expected_episode_indices}, "
            f"actual={actual_episode_indices}"
        )
    reset_ordinals = [row["reset_ordinal"] for row in rows]
    if any(
        current <= previous
        for previous, current in zip(reset_ordinals, reset_ordinals[1:])
    ):
        raise RuntimeError(
            "episode_reset_ordinal values must be unique and strictly increasing"
        )

    return {
        "collection_session_id": rows[0]["session_id"],
        "collection_run_seed": rows[0]["collection_run_seed"],
        "source_recorder_pid": rows[0]["source_recorder_pid"],
        "recorder_lifecycle": rows[0]["recorder_lifecycle"],
        "fresh_scene_replay_required": rows[0]["fresh_scene_replay_required"],
        "demo_dirs": [row["demo_dir"] for row in rows],
        "episode_indices": actual_episode_indices,
        "episode_reset_ordinals": reset_ordinals,
        "session_history_id": rows[0]["history_id"],
        "session_history_root_relative_path": rows[0]["history_path"],
        "session_history_sha256": rows[0]["history_sha256"],
        "session_history_event_count": rows[0]["history_event_count"],
        "session_history_accepted_event_ordinals": [
            row["accepted_event_ordinal"] for row in rows
        ],
        "session_history_certified_through_event_ordinal": rows[0][
            "certified_through_event_ordinal"
        ],
    }


def _run_session_certification(
    demo_dirs: list[Path],
    run_index: int,
    *,
    timeout_seconds: float = DEFAULT_CERTIFICATION_TIMEOUT_SECONDS,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
) -> dict[str, Any]:
    """Certify every accepted episode in one fresh subprocess."""

    demos = [Path(path).expanduser().resolve() for path in demo_dirs]
    report_outputs: dict[Path, tuple[Path, Path]] = {}
    for demo_dir in demos:
        _recording_process_contract(demo_dir)
        report_outputs[demo_dir] = _prepare_fresh_report_output(
            demo_dir,
            run_index,
        )

    command = [sys.executable, str(CERTIFIER)]
    for demo_dir in demos:
        command.extend(("--demo-dir", str(demo_dir)))
    launch = _run_certifier_with_gpu_startup_retry(
        command,
        child_environment=_certifier_environment(demos[0]),
        report_paths=[report_output[0] for report_output in report_outputs.values()],
        timeout_seconds=timeout_seconds,
        termination_grace_seconds=termination_grace_seconds,
    )
    process = launch["process"]
    timed_out = bool(launch["timed_out"])
    returncode = int(process.returncode)
    episode_results: list[dict[str, Any]] = []
    for episode_index, demo_dir in enumerate(demos, start=1):
        report_path, proof_path = report_outputs[demo_dir]
        report, report_reference, report_sha256 = _archive_fresh_report(
            demo_dir,
            report_path,
            proof_path,
        )
        episode_result: dict[str, Any] = {
            "episode_index": episode_index,
            "demo_name": demo_dir.name,
            "demo_dir": str(demo_dir),
            "status": str(report.get("status") or "invalid"),
            "reason": str(report.get("reason") or ""),
            "report": report_reference,
            "report_sha256": report_sha256,
            "certified_inputs": report.get("certified_inputs"),
            **_report_outcome_projection(report),
        }
        blocker = local_physical_replay_report_blocker(demo_dir, report)
        if not report or report.get("status") != "passed" or blocker is not None:
            episode_result["status"] = "failed"
            if blocker is not None:
                reason = blocker.get("detail") or blocker.get("code") or blocker
            else:
                reason = report.get("reason") or ("certifier produced no passed report")
            episode_result["reason"] = str(reason)
        episode_results.append(episode_result)

    failed_episode = next(
        (result for result in episode_results if result.get("status") != "passed"),
        None,
    )
    status = "passed"
    reason = ""
    if timed_out:
        status = "failed"
        reason = f"certifier timed out after {timeout_seconds:g}s"
    elif launch["gpu_context_poisoned"]:
        status = "failed"
        reason = (
            "explicit CUDA/PhysX context-poisoning startup failure exhausted "
            f"{GPU_STARTUP_MAX_ATTEMPTS} GPU-only fresh-process attempts"
        )
    elif returncode != 0:
        status = "failed"
        reason = (
            f"certifier terminated by signal {-returncode}"
            if returncode < 0
            else f"certifier exited with status {returncode}"
        )
    elif failed_episode is not None:
        status = "failed"
        reason = (
            f"{failed_episode['demo_name']}: "
            f"{failed_episode.get('reason') or 'certification failed'}"
        )

    return {
        "run": run_index,
        "pid": int(process.pid),
        "returncode": returncode,
        "started_at": launch["started_at"],
        "finished_at": launch["finished_at"],
        "status": status,
        "reason": reason,
        "timed_out": timed_out,
        "fresh_process_launch_attempts": launch[
            "fresh_process_launch_attempts"
        ],
        "discarded_gpu_startup_attempts": launch[
            "discarded_gpu_startup_attempts"
        ],
        "episodes": episode_results,
    }


def _session_publication_commit_descriptor(
    session_context: Mapping[str, Any],
) -> dict[str, str]:
    """Return the immutable marker identity referenced by every acceptance."""

    identity_input = {
        "collection_session_id": session_context["collection_session_id"],
        "session_demo_names": [
            Path(path).name for path in session_context["demo_dirs"]
        ],
        "session_episode_reset_ordinals": list(
            session_context["episode_reset_ordinals"]
        ),
        "session_history_sha256": session_context["session_history_sha256"],
    }
    import hashlib

    digest = hashlib.sha256(
        json.dumps(
            identity_input,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    commit_id = f"session-{digest}"
    return {
        "session_publication_commit_id": commit_id,
        "session_publication_commit_root_relative_path": (
            f"{SESSION_PUBLICATION_COMMIT_DIRECTORY}/{commit_id}.json"
        ),
    }


def _write_acceptance(
    demo_dir: Path,
    *,
    exp_name: str,
    certification_runs_required: int,
    runs: list[dict[str, Any]],
    session_context: Mapping[str, Any] | None = None,
    session_episode_index: int | None = None,
) -> None:
    certified_inputs = capture_certified_inputs(demo_dir)
    recorder_lifecycle, fresh_scene_replay_required, source_recorder_pid = (
        _recording_process_contract(demo_dir)
    )
    passed = sum(1 for run in runs if run.get("status") == "passed")
    certifier_pids = [int(run["pid"]) for run in runs]
    independent_fresh_certifiers = bool(
        len(runs) == DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS
        and passed == DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS
        and len(set(certifier_pids)) == DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS
        and source_recorder_pid not in certifier_pids
    )
    review_outcome = _review_outcome_from_runs(
        runs,
        physically_valid=independent_fresh_certifiers,
    )
    prepared_at = _now_iso()
    payload: dict[str, Any] = {
        "format": ACCEPTANCE_FORMAT,
        "version": 1,
        "mode": DEFERRED_MODE,
        "status": "passed" if independent_fresh_certifiers else "failed",
        "accepted_at": prepared_at,
        "exp_name": exp_name,
        "certification_runs_required": certification_runs_required,
        "certification_runs_passed": passed,
        "certifier_pids": certifier_pids,
        "certification_runs": runs,
        "certified_inputs": certified_inputs,
        # This final proof is prepared while the directory is still hidden.
        # Its same-filesystem rename into the publication root is the only
        # commit point: a crash before rename remains hidden, while a crash
        # after rename exposes an already complete proof with no second write.
        "published_after_post_collection_certification": True,
        "publication_status": "committed_by_atomic_directory_location",
        "publication_commit": "same_filesystem_atomic_directory_rename",
        "publication_proof_prepared_at": prepared_at,
        "published_at": prepared_at,
        "persistent_source_process_for_collection_batch": True,
        "fresh_source_process_per_episode": False,
        "fresh_certifier_process_per_run": True,
        "recorder_lifecycle": recorder_lifecycle,
        "fresh_scene_replay_required": fresh_scene_replay_required,
        "source_recorder_pid": source_recorder_pid,
        "recorder_pid": source_recorder_pid,
        "fresh_certifier_process_per_run_required": True,
        **review_outcome,
    }
    if session_context is not None:
        commit_descriptor = _session_publication_commit_descriptor(session_context)
        payload.update(
            {
                "certification_scope": SESSION_CERTIFICATION_SCOPE,
                "collection_session_id": session_context["collection_session_id"],
                "collection_run_seed": session_context["collection_run_seed"],
                "session_episode_count": len(session_context["demo_dirs"]),
                "session_episode_index": session_episode_index,
                "session_episode_reset_ordinals": list(
                    session_context["episode_reset_ordinals"]
                ),
                "session_demo_names": [
                    Path(path).name for path in session_context["demo_dirs"]
                ],
                "fresh_certifier_process_per_session_run": True,
                "session_publication_all_targets_prevalidated": True,
                "session_publication_rollback_on_catchable_error": True,
                "publication_status": ("committed_by_atomic_session_marker"),
                "publication_commit": (
                    "ordered_directory_renames_then_atomic_session_marker"
                ),
                **commit_descriptor,
            }
        )
    if not independent_fresh_certifiers:
        payload["reason"] = (
            "deferred post-collection certification requires exactly three "
            "successful fresh certifier processes, all distinct from the "
            "persistent source recorder"
        )
    _write_json_atomic(demo_dir / FRESH_PROCESS_ACCEPTANCE_NAME, payload)


def _mark_acceptance_failed(demo_dir: Path, reason: str) -> None:
    acceptance_path = demo_dir / FRESH_PROCESS_ACCEPTANCE_NAME
    acceptance = _read_json(acceptance_path)
    acceptance["status"] = "failed"
    acceptance["reason"] = str(reason)
    acceptance["published_after_post_collection_certification"] = False
    acceptance["publication_status"] = "publication_failed"
    acceptance["published_at"] = None
    if (
        acceptance.get("physical_replay_outcome_code")
        != PHYSICAL_REPLAY_INVALID_OUTCOME_CODE
        or acceptance.get("review_status") != PHYSICAL_REPLAY_REVIEW_INVALID
        or not isinstance(acceptance.get("outcome_evidence"), Mapping)
    ):
        acceptance.update(_review_outcome_from_runs([], physically_valid=False))
        acceptance.pop("outcome_evidence", None)
    _write_json_atomic(acceptance_path, acceptance)


def _pending_publication_target(demo_dir: Path) -> Path:
    demo_dir = demo_dir.resolve()
    if demo_dir.parent.name != PENDING_DIRECTORY_NAME:
        raise RuntimeError(
            "deferred certification source must be a direct child of "
            f"{PENDING_DIRECTORY_NAME}"
        )
    publication_root = demo_dir.parent.parent
    if not publication_root.is_dir():
        raise RuntimeError(
            f"deferred publication root does not exist: {publication_root}"
        )
    if demo_dir.stat().st_dev != publication_root.stat().st_dev:
        raise RuntimeError(
            "pending demo and publication root must share one filesystem"
        )
    return publication_root / demo_dir.name


def _publish_pending_demo(demo_dir: Path) -> Path:
    """Atomically expose one fully certified pending demo without clobbering."""

    demo_dir = demo_dir.resolve()
    target = _pending_publication_target(demo_dir)
    if os.path.lexists(target):
        raise RuntimeError(f"refusing to replace existing published demo {target}")
    # Source and destination have already been proven to share a filesystem;
    # rename therefore exposes the complete directory in one metadata operation.
    os.rename(demo_dir, target)
    return target


def _publish_pending_session(
    demo_dirs: list[Path],
    *,
    session_context: Mapping[str, Any],
) -> list[Path]:
    """Publish a batch and commit it with one final atomic marker.

    Individual directories must be renamed one at a time.  They remain
    upload-ineligible until the final marker is atomically installed after
    every member is present, which makes a crash between renames fail closed.
    """

    demos = [Path(path).expanduser().resolve() for path in demo_dirs]
    targets = [_pending_publication_target(demo) for demo in demos]
    if len(set(targets)) != len(targets):
        raise RuntimeError("deferred session contains duplicate targets")
    existing = [target for target in targets if os.path.lexists(target)]
    if existing:
        raise RuntimeError(
            "refusing to replace existing published session targets: "
            + ", ".join(str(path) for path in existing)
        )

    descriptor = _session_publication_commit_descriptor(session_context)
    publication_root = targets[0].parent
    marker_path = (
        publication_root / descriptor["session_publication_commit_root_relative_path"]
    )
    if marker_path.exists():
        raise RuntimeError(f"refusing to replace existing session commit {marker_path}")

    renamed: list[tuple[Path, Path]] = []
    try:
        for demo, target in zip(demos, targets):
            os.rename(demo, target)
            renamed.append((demo, target))
        # Persist both sides of every rename before installing the commit
        # marker.  The marker is the upload gate's final commit point, so it
        # must never reach durable storage ahead of a session member.
        _fsync_directory(demos[0].parent)
        _fsync_directory(publication_root)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        _fsync_directory(publication_root)
        members: list[dict[str, Any]] = []
        accepted_ordinals = list(
            session_context["session_history_accepted_event_ordinals"]
        )
        reset_ordinals = list(session_context["episode_reset_ordinals"])
        for target, reset_ordinal, accepted_ordinal in zip(
            targets,
            reset_ordinals,
            accepted_ordinals,
            strict=True,
        ):
            acceptance_path = target / FRESH_PROCESS_ACCEPTANCE_NAME
            acceptance = _read_json(acceptance_path)
            if not acceptance or acceptance.get("status") != "passed":
                raise RuntimeError(
                    f"published member has no passed acceptance: {target}"
                )
            members.append(
                {
                    "demo_name": target.name,
                    "episode_reset_ordinal": int(reset_ordinal),
                    "accepted_event_ordinal": int(accepted_ordinal),
                    "acceptance_sha256": _sha256_file(acceptance_path),
                    "certified_inputs": acceptance.get("certified_inputs"),
                }
            )
        _write_json_atomic(
            marker_path,
            {
                "format": SESSION_PUBLICATION_COMMIT_FORMAT,
                "version": 1,
                "status": "committed",
                **descriptor,
                "collection_session_id": session_context["collection_session_id"],
                "session_demo_names": [target.name for target in targets],
                "session_episode_reset_ordinals": reset_ordinals,
                "session_history_id": session_context["session_history_id"],
                "session_history_sha256": session_context["session_history_sha256"],
                "session_history_event_count": session_context[
                    "session_history_event_count"
                ],
                "session_history_certified_through_event_ordinal": (
                    session_context["session_history_certified_through_event_ordinal"]
                ),
                "members": members,
                "committed_at": _now_iso(),
            },
        )
    except BaseException:
        marker_removed = False
        try:
            marker_path.unlink()
            marker_removed = True
        except FileNotFoundError:
            pass
        if marker_removed:
            with contextlib.suppress(OSError):
                _fsync_directory(marker_path.parent)
        rollback_errors: list[str] = []
        for demo, target in reversed(renamed):
            try:
                os.rename(target, demo)
            except OSError as exc:
                rollback_errors.append(f"{target} -> {demo}: {exc}")
        with contextlib.suppress(OSError):
            _fsync_directory(demos[0].parent)
        with contextlib.suppress(OSError):
            _fsync_directory(publication_root)
        if rollback_errors:
            raise RuntimeError(
                "session publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            )
        raise
    return targets


def certify_demo_dir(
    demo_dir: Path,
    *,
    exp_name: str,
    certification_runs_required: int,
    timeout_seconds: float = DEFAULT_CERTIFICATION_TIMEOUT_SECONDS,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
) -> dict[str, Any]:
    if certification_runs_required != DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS:
        raise ValueError(
            "deferred certification requires exactly "
            f"{DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS} fresh runs"
        )
    if timeout_seconds <= 0.0:
        raise ValueError("certification timeout must be positive")
    if termination_grace_seconds <= 0.0:
        raise ValueError("termination grace must be positive")
    _pending_publication_target(demo_dir)
    _wait_for_postprocess_inputs(
        demo_dir,
        timeout_seconds=timeout_seconds,
    )
    runs: list[dict[str, Any]] = []
    source_recorder_pid = _recording_process_contract(demo_dir)[2]
    for run_index in range(1, certification_runs_required + 1):
        result = _run_single_certification(
            demo_dir,
            run_index,
            timeout_seconds=timeout_seconds,
            termination_grace_seconds=termination_grace_seconds,
        )
        prior_pids = {int(run["pid"]) for run in runs}
        if int(result["pid"]) == source_recorder_pid:
            result["status"] = "failed"
            result["reason"] = "certifier pid equals the persistent source recorder pid"
        elif int(result["pid"]) in prior_pids:
            result["status"] = "failed"
            result["reason"] = "certifier pid was reused within the three-run proof"
        runs.append(result)
        if result.get("status") != "passed":
            break
    _write_acceptance(
        demo_dir,
        exp_name=exp_name,
        certification_runs_required=certification_runs_required,
        runs=runs,
    )
    passed = bool(
        len(runs) == DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS
        and all(run.get("status") == "passed" for run in runs)
        and len({int(run["pid"]) for run in runs})
        == DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS
        and _recording_process_contract(demo_dir)[2]
        not in {int(run["pid"]) for run in runs}
    )
    physical_certification_passed = passed
    failure_reason = str(
        next(
            (run.get("reason") for run in runs if run.get("status") != "passed"),
            "deferred post-collection certification failed",
        )
    )
    if passed:
        blocker = deferred_physical_replay_acceptance_upload_blocker(demo_dir)
        if blocker is not None:
            passed = False
            failure_reason = str(
                blocker.get("detail") or blocker.get("code") or blocker
            )
            _mark_acceptance_failed(demo_dir, failure_reason)

    published_demo_dir: Path | None = None
    if passed:
        try:
            published_demo_dir = _publish_pending_demo(demo_dir)
        except (OSError, RuntimeError) as exc:
            passed = False
            failure_reason = f"atomic publication failed: {exc}"
            failure_demo_dir = demo_dir
            if published_demo_dir is not None:
                try:
                    os.rename(published_demo_dir, demo_dir)
                    published_demo_dir = None
                except OSError:
                    # Even if rollback itself fails, the acceptance stays
                    # failed/ unpublished and the upload gate remains closed.
                    failure_demo_dir = published_demo_dir
            _mark_acceptance_failed(failure_demo_dir, failure_reason)

    publication_failure = bool(not passed and physical_certification_passed)
    review_outcome = _review_outcome_from_runs(
        [] if publication_failure else runs,
        physically_valid=passed,
    )
    return {
        "demo_dir": str(published_demo_dir or demo_dir),
        "pending_demo_dir": str(demo_dir),
        "published_demo_dir": (
            str(published_demo_dir) if published_demo_dir is not None else None
        ),
        "status": "passed" if passed else "failed",
        "reason": "" if passed else failure_reason,
        "runs": runs,
        "episode_outcomes": [{"demo_name": demo_dir.name, **review_outcome}],
    }


def _episode_run_projection(
    session_run: Mapping[str, Any],
    *,
    demo_dir: Path,
) -> dict[str, Any]:
    episode_result = next(
        (
            row
            for row in session_run.get("episodes", [])
            if isinstance(row, Mapping) and row.get("demo_name") == demo_dir.name
        ),
        None,
    )
    if not isinstance(episode_result, Mapping):
        episode_result = {
            "status": "failed",
            "reason": "session run omitted this accepted episode",
            "report": "",
            "report_sha256": "",
            "certified_inputs": None,
        }
    session_passed = session_run.get("status") == "passed"
    episode_passed = episode_result.get("status") == "passed"
    reason = ""
    if not session_passed:
        reason = str(session_run.get("reason") or "session certification failed")
    elif not episode_passed:
        reason = str(episode_result.get("reason") or "episode certification failed")
    projection = {
        "run": session_run.get("run"),
        "pid": session_run.get("pid"),
        "returncode": session_run.get("returncode"),
        "started_at": session_run.get("started_at"),
        "finished_at": session_run.get("finished_at"),
        "status": "passed" if session_passed and episode_passed else "failed",
        "reason": reason,
        "report": episode_result.get("report"),
        "report_sha256": episode_result.get("report_sha256"),
        "certified_inputs": episode_result.get("certified_inputs"),
        "timed_out": bool(session_run.get("timed_out")),
        "fresh_process_launch_attempts": session_run.get(
            "fresh_process_launch_attempts"
        ),
        "discarded_gpu_startup_attempts": list(
            session_run.get("discarded_gpu_startup_attempts", [])
        ),
        "session_episode_count": len(session_run.get("episodes", [])),
    }
    for field in (
        "report_format",
        "report_version",
        "physical_report_status",
        "physical_report_reason",
        "error_type",
        "certification_mode",
        "execution_mode",
        "steps_expected",
        "steps_executed",
        "task_success_replay",
        "final_succeed",
        "trace_match",
        "transition_outcomes_match",
        "first_divergence",
        "first_outcome_mismatch",
        "portable",
        "source_identity",
        "source_snapshot_commit",
        "source_reviewer_upgrade_paths",
        CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD,
    ):
        projection[field] = episode_result.get(field)
    return projection


def certify_session(
    demo_dirs: list[Path],
    *,
    exp_name: str,
    certification_runs_required: int,
    timeout_seconds: float = DEFAULT_CERTIFICATION_TIMEOUT_SECONDS,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
) -> dict[str, Any]:
    """Certify and publish one accepted-episode session as a single unit."""

    if certification_runs_required != DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS:
        raise ValueError(
            "deferred certification requires exactly "
            f"{DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS} fresh runs"
        )
    if timeout_seconds <= 0.0:
        raise ValueError("certification timeout must be positive")
    if termination_grace_seconds <= 0.0:
        raise ValueError("termination grace must be positive")

    session = _validated_session_context(demo_dirs, exp_name=exp_name)
    demos = list(session["demo_dirs"])
    # Resolve every destination before starting an expensive proof.  No member
    # may be published if even one target is already occupied.
    targets = [_pending_publication_target(demo) for demo in demos]
    if len(set(targets)) != len(targets):
        raise RuntimeError("deferred session contains duplicate targets")
    existing = [target for target in targets if os.path.lexists(target)]
    if existing:
        raise RuntimeError(
            "refusing to replace existing published session targets: "
            + ", ".join(str(path) for path in existing)
        )
    for demo in demos:
        _wait_for_postprocess_inputs(demo, timeout_seconds=timeout_seconds)

    source_recorder_pid = int(session["source_recorder_pid"])
    session_runs: list[dict[str, Any]] = []
    for run_index in range(1, certification_runs_required + 1):
        result = _run_session_certification(
            demos,
            run_index,
            timeout_seconds=timeout_seconds,
            termination_grace_seconds=termination_grace_seconds,
        )
        prior_pids = {int(run["pid"]) for run in session_runs}
        if int(result["pid"]) == source_recorder_pid:
            result["status"] = "failed"
            result["reason"] = "certifier pid equals the persistent source recorder pid"
        elif int(result["pid"]) in prior_pids:
            result["status"] = "failed"
            result["reason"] = "certifier pid was reused within the three-run proof"
        session_runs.append(result)
        if result.get("status") != "passed":
            break

    certifier_pids = [int(run["pid"]) for run in session_runs]
    physical_certification_passed = bool(
        len(session_runs) == DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS
        and all(run.get("status") == "passed" for run in session_runs)
        and len(set(certifier_pids)) == DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS
        and source_recorder_pid not in certifier_pids
    )
    passed = physical_certification_passed
    failure_stage: str | None = (
        None if physical_certification_passed else "physical_certification"
    )
    publication_status = "not_attempted"
    failure_reason = str(
        next(
            (
                run.get("reason")
                for run in session_runs
                if run.get("status") != "passed"
            ),
            "deferred post-collection session certification failed",
        )
    )

    projected_runs_by_demo: dict[Path, list[dict[str, Any]]] = {}
    for episode_index, demo in enumerate(demos, start=1):
        projected_runs = [
            _episode_run_projection(run, demo_dir=demo) for run in session_runs
        ]
        projected_runs_by_demo[demo] = projected_runs
        _write_acceptance(
            demo,
            exp_name=exp_name,
            certification_runs_required=certification_runs_required,
            runs=projected_runs,
            session_context=session,
            session_episode_index=episode_index,
        )
        if not passed:
            _mark_acceptance_failed(demo, failure_reason)

    if passed:
        for demo in demos:
            blocker = deferred_physical_replay_acceptance_upload_blocker(
                demo,
                require_published=True,
                require_publication_commit=False,
            )
            if blocker is not None:
                passed = False
                failure_stage = "publication_gate"
                publication_status = "blocked"
                failure_reason = str(
                    blocker.get("detail") or blocker.get("code") or blocker
                )
                break
        if not passed:
            for demo in demos:
                _mark_acceptance_failed(demo, failure_reason)

    published: list[Path] = []
    if passed:
        publication_status = "publishing"
        try:
            published = _publish_pending_session(
                demos,
                session_context=session,
            )
            publication_status = "published"
        except (OSError, RuntimeError) as exc:
            passed = False
            failure_stage = "publication_commit"
            publication_status = "blocked"
            failure_reason = f"session publication failed: {exc}"
            for demo, target in zip(demos, targets):
                current = target if target.is_dir() else demo
                if current.is_dir():
                    _mark_acceptance_failed(current, failure_reason)

    if passed:
        for published_demo in published:
            blocker = deferred_physical_replay_acceptance_upload_blocker(published_demo)
            if blocker is not None:
                passed = False
                failure_stage = "post_publication_gate"
                publication_status = "blocked"
                failure_reason = str(
                    blocker.get("detail") or blocker.get("code") or blocker
                )
                break

    episode_outcomes: list[dict[str, Any]] = []
    publication_failure = failure_stage in {
        "publication_gate",
        "publication_commit",
        "post_publication_gate",
    }
    for demo in demos:
        if publication_failure:
            review_outcome = _review_outcome_from_runs([], physically_valid=False)
        else:
            review_outcome = _review_outcome_from_runs(
                projected_runs_by_demo[demo],
                physically_valid=passed,
            )
        episode_outcomes.append({"demo_name": demo.name, **review_outcome})

    return {
        "collection_session_id": session["collection_session_id"],
        "demo_dirs": [str(path) for path in (published or demos)],
        "pending_demo_dirs": [str(path) for path in demos],
        "published_demo_dirs": [str(path) for path in published],
        "status": "passed" if passed else "failed",
        "reason": "" if passed else failure_reason,
        "failure_stage": failure_stage,
        "physical_certification_status": (
            "passed" if physical_certification_passed else "failed"
        ),
        "physical_certification_runs_passed": sum(
            1 for run in session_runs if run.get("status") == "passed"
        ),
        "publication_status": publication_status,
        "runs": session_runs,
        "episode_outcomes": episode_outcomes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument(
        "--certification-runs",
        type=int,
        default=DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS,
    )
    parser.add_argument("--log_path", required=True)
    parser.add_argument(
        "--certification-timeout-seconds",
        "--physical-replay-certification-timeout-seconds",
        "--physical_replay_certification_timeout_seconds",
        type=float,
        default=DEFAULT_CERTIFICATION_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--termination-grace-seconds",
        type=float,
        default=DEFAULT_TERMINATION_GRACE_SECONDS,
    )
    parser.add_argument("demo_dirs", nargs="+")
    args = parser.parse_args(argv)
    if args.certification_runs != DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS:
        parser.error(
            "--certification-runs must be exactly "
            f"{DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS}"
        )
    if args.certification_timeout_seconds <= 0.0:
        parser.error("--certification-timeout-seconds must be positive")
    if args.termination_grace_seconds <= 0.0:
        parser.error("--termination-grace-seconds must be positive")

    demo_dirs = [
        Path(raw_demo_dir).expanduser().resolve() for raw_demo_dir in args.demo_dirs
    ]
    log_path = Path(args.log_path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    execution_lock_handle = None
    output_gpu_gate_handle = None
    terminal_status = "failed"
    terminal_result: Mapping[str, Any] | None = None
    terminal_error = ""
    try:
        _wait_for_launching_recorder_exit(
            timeout_seconds=float(args.certification_timeout_seconds),
        )
        pending_parents = {demo_dir.parent for demo_dir in demo_dirs}
        if len(pending_parents) != 1:
            raise RuntimeError(
                "one deferred certification batch must use one pending root"
            )
        output_root = next(iter(pending_parents)).parent.resolve()
        output_gpu_gate_path = _output_scoped_gpu_gate_path(output_root)
        output_gpu_gate_path.parent.mkdir(parents=True, exist_ok=True)
        output_gpu_gate_handle = output_gpu_gate_path.open("a+", encoding="utf-8")
        # The UI intentionally holds this gate while spawning the recovery
        # CLI and this worker.  Blocking here until that short launch
        # transaction ends closes the cross-worktree race without rejecting
        # the already queued review.
        fcntl.flock(output_gpu_gate_handle.fileno(), fcntl.LOCK_EX)
        # This lock is deliberately repo-scoped rather than pending-root
        # scoped.  Recovery CLIs can be launched manually for unrelated
        # output roots, but PhysX certification must still use the GPU one
        # batch at a time.  Acquire it only after the source recorder exits so
        # an attached worker never monopolizes the lock while collection is
        # still alive.
        execution_lock_path = BATCH_EXECUTION_LOCK_PATH
        execution_lock_path.parent.mkdir(parents=True, exist_ok=True)
        execution_lock_handle = execution_lock_path.open(
            "a+",
            encoding="utf-8",
        )
        fcntl.flock(execution_lock_handle.fileno(), fcntl.LOCK_EX)
        # E/F postprocessing can mutate replay.zarr and can itself use the
        # GPU.  Preflight the entire batch before launching even the first
        # fresh PhysX process, otherwise certification of demo 1 could race
        # postprocessing of demo 2.
        for demo_dir in demo_dirs:
            if not (demo_dir / "replay.zarr").is_dir():
                continue
            _wait_for_postprocess_inputs(
                demo_dir,
                timeout_seconds=float(args.certification_timeout_seconds),
            )
        print(
            "[Deferred PhysX certification worker] starting batch: "
            f"exp_name={args.exp_name}, demos={len(args.demo_dirs)}, "
            f"runs={args.certification_runs}, "
            f"timeout={args.certification_timeout_seconds:g}s",
            flush=True,
        )
        missing = [
            str(demo_dir)
            for demo_dir in demo_dirs
            if not (demo_dir / "replay.zarr").is_dir()
        ]
        if missing:
            raise RuntimeError(
                "Deferred PhysX certification failed; missing replay.zarr "
                "for: " + ", ".join(missing)
            )
        result = certify_session(
            demo_dirs,
            exp_name=args.exp_name,
            certification_runs_required=max(1, int(args.certification_runs)),
            timeout_seconds=float(args.certification_timeout_seconds),
            termination_grace_seconds=float(args.termination_grace_seconds),
        )
        terminal_result = result
        print(
            f"[Deferred PhysX certification worker] {result['status']}: "
            f"session={result['collection_session_id']} "
            f"demos={len(result['demo_dirs'])} "
            "physical_certification="
            f"{result.get('physical_certification_status', 'unknown')} "
            f"publication={result.get('publication_status', 'unknown')} "
            f"failure_stage={result.get('failure_stage') or 'none'}",
            flush=True,
        )
        if result["status"] != "passed":
            raise RuntimeError(
                "Deferred PhysX certification failed for session "
                f"{result['collection_session_id']}: {result['reason']}"
            )
        print(
            "[Deferred PhysX certification worker] batch complete",
            flush=True,
        )
        terminal_status = "passed"
        return 0
    except Exception as exc:  # noqa: BLE001
        terminal_error = f"{type(exc).__name__}: {exc}"
        print(
            "[Deferred PhysX certification worker] failed with exception:\n"
            f"{terminal_error}",
            file=log_handle,
            flush=True,
        )
        raise
    finally:
        active_exception = sys.exc_info()[0] is not None
        state_error: Exception | None = None
        try:
            _write_worker_terminal_states(
                demo_dirs,
                status=terminal_status,
                result=terminal_result,
                error=terminal_error,
            )
        except Exception as exc:  # noqa: BLE001
            state_error = exc
            print(
                "[Deferred PhysX certification worker] failed to write "
                "terminal state:\n"
                f"{type(exc).__name__}: {exc}",
                file=log_handle,
                flush=True,
            )
        if execution_lock_handle is not None:
            fcntl.flock(execution_lock_handle.fileno(), fcntl.LOCK_UN)
            execution_lock_handle.close()
        if output_gpu_gate_handle is not None:
            fcntl.flock(output_gpu_gate_handle.fileno(), fcntl.LOCK_UN)
            output_gpu_gate_handle.close()
        log_handle.close()
        if state_error is not None and not active_exception:
            raise state_error


if __name__ == "__main__":
    raise SystemExit(main())
