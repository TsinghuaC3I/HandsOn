#!/usr/bin/env python3
"""Collect PhysX episodes in fresh processes and publish only certified ones.

The normal recorder owns one long-lived simulator process.  This supervisor
instead launches it for exactly one recording attempt in a private staging
root, waits for that process to exit, and then launches the PhysX certifier in
a different fresh process.  A demo becomes visible in the requested ``out_dir``
only after every requested certification run passes.  Failed attempts remain
in a sibling quarantine tree that the collection UI never scans.

All ordinary ``record_demos_zarr.py`` flags are accepted and forwarded.  The
supervisor consumes only the following additional flags:

``--fresh-certification-runs``
    Number of independent fresh-process certifications required per episode
    (default: 3).
``--fresh-staging-root`` / ``--fresh-quarantine-root``
    Optional roots outside the formal output directory.
``--fresh-status-file``
    Optional JSON status file used by the collection UI to pause/resume the
    currently active child process group as well as this supervisor.
``--fresh-max-attempts``
    Stop after this many recorded attempts; zero means retry until the
    requested number of certified episodes has been published.
``--fresh-termination-grace-seconds``
    Grace period before a stopped child process group is killed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "handson"
for _entry in (str(PACKAGE_ROOT), str(REPO_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from handson.data.physical_replay_certification import (  # noqa: E402
    DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS,
    FRESH_PROCESS_ACCEPTANCE_FORMAT as ACCEPTANCE_FORMAT,
    FRESH_PROCESS_ACCEPTANCE_NAME as ACCEPTANCE_NAME,
    FRESH_PROCESS_ACCEPTANCE_VERSION as ACCEPTANCE_VERSION,
    FRESH_PROCESS_REPORT_DIRECTORY as RUN_REPORT_DIRECTORY,
    LEGACY_RAW_ACTION_TRANSITION_CONTRACT,
    RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT,
    REPORT_NAME,
    capture_certified_inputs,
    certification_report_blocker,
    certification_upload_blocker,
    fresh_process_acceptance_upload_blocker,
)
from handson.runtime.collection_support import (  # noqa: E402
    ISAACSIM_NONSTANDARD_DEFERRED_RENDER_TASKS,
    supports_isaacsim_deferred_camera_render,
)


RECORDER = REPO_ROOT / "scripts" / "record_demos_zarr.py"
CERTIFIER = REPO_ROOT / "scripts" / "certify_physx_replay.py"
DEFERRED_ARM_WORKER = REPO_ROOT / "scripts" / "isaac_deferred_arm_replay_worker.py"
REJECTION_NAME = "fresh_process_rejection.json"
MIN_FRESH_CERTIFICATION_RUNS = DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS
_RETRYABLE_ATTEMPT_EXIT_CODE = getattr(os, "EX_TEMPFAIL", 75)

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


class SupervisorError(RuntimeError):
    """Raised when fail-closed collection cannot safely continue."""


class StopRequested(SupervisorError):
    """Raised after SIGINT or SIGTERM has been relayed to the active child."""

    def __init__(self, signum: int):
        super().__init__(f"stop requested by signal {signum}")
        self.signum = int(signum)


class ChildTimeout(SupervisorError):
    """Raised after a child exceeds its configured wall-clock timeout."""


class ChildStartupTimeout(ChildTimeout):
    """Raised when Isaac never publishes its startup-ready marker."""


@dataclass(frozen=True)
class ProcessResult:
    pid: int
    returncode: int
    started_at: str
    finished_at: str


@dataclass(frozen=True)
class SupervisorConfig:
    out_dir: Path
    staging_root: Path
    quarantine_root: Path
    successes_needed: int
    certification_runs: int
    certification_timeout_seconds: int
    termination_grace_seconds: float
    startup_timeout_seconds: float
    max_attempts: int
    base_seed: int
    recorder_args: tuple[str, ...]
    session_id: str
    status_file: Path | None


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )


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
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_private_root(path: Path, out_dir: Path, *, label: str) -> Path:
    private = path.expanduser().resolve()
    formal = out_dir.expanduser().resolve()
    if (
        private == formal
        or _is_relative_to(private, formal)
        or _is_relative_to(formal, private)
    ):
        raise SupervisorError(
            f"{label} must be outside and disjoint from formal out_dir: "
            f"{private} versus {formal}"
        )
    return private


def _default_private_root(out_dir: Path, suffix: str) -> Path:
    formal = out_dir.expanduser().resolve()
    name = formal.name or "handson_demos"
    # The collection UI defaults to ``<repo>/demos``.  A hidden sibling in
    # that case would still be an untracked Git path: the recorder would bind
    # that temporary startup/quarantine content into software.git, then the
    # certifier would see a different worktree after the marker disappeared
    # and reject every otherwise valid episode.  Keep all private lifecycle
    # data outside the source tree while retaining a sibling fallback for
    # output roots that are already external to the repository.
    if _is_relative_to(formal, REPO_ROOT):
        path_tag = hashlib.sha256(str(formal).encode("utf-8")).hexdigest()[:12]
        return (
            REPO_ROOT.parent
            / ".handson_collection_private"
            / f"{name}-{path_tag}"
            / suffix
        )
    return formal.parent / f".{name}.{suffix}"


def _option_value(
    arguments: Sequence[str],
    name: str,
    default: str,
) -> str:
    prefix = f"--{name}="
    result = default
    for index, token in enumerate(arguments):
        if token.startswith(prefix):
            result = token[len(prefix) :]
        elif token == f"--{name}":
            if index + 1 >= len(arguments) or arguments[index + 1].startswith(
                "--"
            ):
                raise SupervisorError(f"--{name} requires a value")
            result = arguments[index + 1]
    return result


def _bool_option(arguments: Sequence[str], name: str, default: bool) -> bool:
    result = bool(default)
    prefix = f"--{name}="
    for token in arguments:
        if token == f"--{name}":
            result = True
        elif token == f"--no{name}":
            result = False
        elif token.startswith(prefix):
            value = token[len(prefix) :].strip().lower()
            if value not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
                raise SupervisorError(f"--{name} has invalid boolean value {value!r}")
            result = value in {"1", "true", "yes", "on"}
    return result


def _remove_option(
    arguments: Sequence[str],
    name: str,
    *,
    takes_value: bool = True,
) -> list[str]:
    result: list[str] = []
    index = 0
    prefix = f"--{name}="
    while index < len(arguments):
        token = arguments[index]
        if token.startswith(prefix) or (not takes_value and token == f"--no{name}"):
            index += 1
            continue
        if token == f"--{name}":
            index += 1
            if takes_value:
                if index >= len(arguments):
                    raise SupervisorError(f"--{name} requires a value")
                index += 1
            continue
        result.append(token)
        index += 1
    return result


def _replace_option(
    arguments: Sequence[str],
    name: str,
    value: str | int | float | bool,
    *,
    boolean: bool = False,
) -> list[str]:
    cleaned = _remove_option(arguments, name, takes_value=not boolean)
    rendered = str(value).lower() if isinstance(value, bool) else str(value)
    cleaned.append(f"--{name}={rendered}")
    return cleaned


def _consume_supervisor_options(arguments: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    remaining = list(arguments)
    defaults = {
        "fresh-certification-runs": "3",
        "fresh-staging-root": "",
        "fresh-quarantine-root": "",
        "fresh-max-attempts": "0",
        "fresh-termination-grace-seconds": "12",
        "fresh-status-file": "",
    }
    for name, default in defaults.items():
        values[name] = _option_value(remaining, name, default)
        remaining = _remove_option(remaining, name)
    return values, remaining


def _safe_session_id(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip()).strip("._-")
    return safe[:96] or (
        dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-"
        + secrets.token_hex(6)
    )


def build_config(arguments: Sequence[str]) -> SupervisorConfig:
    supervisor, recorder_args = _consume_supervisor_options(arguments)
    out_dir = Path(_option_value(recorder_args, "out_dir", "./")).expanduser().resolve()
    successes_needed = int(_option_value(recorder_args, "successes_needed", "2"))
    if successes_needed <= 0:
        raise SupervisorError("--successes_needed must be positive")
    certification_runs = int(supervisor["fresh-certification-runs"])
    if certification_runs < MIN_FRESH_CERTIFICATION_RUNS:
        raise SupervisorError(
            "--fresh-certification-runs must be at least "
            f"{MIN_FRESH_CERTIFICATION_RUNS}; the fail-closed contract cannot "
            "be weakened from the CLI"
        )
    max_attempts = int(supervisor["fresh-max-attempts"])
    if max_attempts < 0:
        raise SupervisorError("--fresh-max-attempts cannot be negative")
    grace = float(supervisor["fresh-termination-grace-seconds"])
    if grace <= 0.0:
        raise SupervisorError("--fresh-termination-grace-seconds must be positive")
    try:
        startup_timeout = float(
            os.environ.get("HANDSON_ISAAC_STARTUP_TIMEOUT_SECONDS", "60")
        )
    except ValueError as exc:
        raise SupervisorError(
            "HANDSON_ISAAC_STARTUP_TIMEOUT_SECONDS must be numeric"
        ) from exc
    if startup_timeout <= 0.0:
        raise SupervisorError(
            "HANDSON_ISAAC_STARTUP_TIMEOUT_SECONDS must be positive"
        )
    timeout = int(
        _option_value(
            recorder_args,
            "physical_replay_certification_timeout_seconds",
            "1800",
        )
    )
    if timeout <= 0:
        raise SupervisorError(
            "--physical_replay_certification_timeout_seconds must be positive"
        )
    raw_seed = int(_option_value(recorder_args, "seed", "-1"))
    base_seed = raw_seed if raw_seed >= 0 else int(secrets.randbits(31))

    raw_staging = supervisor["fresh-staging-root"].strip()
    raw_quarantine = supervisor["fresh-quarantine-root"].strip()
    staging_root = _validate_private_root(
        Path(raw_staging)
        if raw_staging
        else _default_private_root(out_dir, "fresh_process_staging"),
        out_dir,
        label="fresh staging root",
    )
    quarantine_root = _validate_private_root(
        Path(raw_quarantine)
        if raw_quarantine
        else _default_private_root(out_dir, "physical_replay_quarantine"),
        out_dir,
        label="fresh quarantine root",
    )
    if staging_root == quarantine_root:
        raise SupervisorError("fresh staging and quarantine roots must differ")
    for label, private_root in (
        ("fresh staging root", staging_root),
        ("fresh quarantine root", quarantine_root),
    ):
        if _is_relative_to(private_root, REPO_ROOT):
            raise SupervisorError(
                f"{label} must be outside the source repository so temporary "
                f"episode state cannot change replay source identity: {private_root}"
            )
    if _is_relative_to(out_dir, REPO_ROOT) and not _is_relative_to(
        out_dir,
        REPO_ROOT / "demos",
    ):
        raise SupervisorError(
            "an in-repository formal out_dir must be under the Git-ignored "
            f"{REPO_ROOT / 'demos'} tree; got {out_dir}"
        )

    session_id = _safe_session_id(
        os.environ.get("HANDSON_COLLECTION_SESSION_ID", "")
    )
    raw_status_file = supervisor["fresh-status-file"].strip()
    status_file = (
        Path(raw_status_file).expanduser().resolve()
        if raw_status_file
        else None
    )
    return SupervisorConfig(
        out_dir=out_dir,
        staging_root=staging_root,
        quarantine_root=quarantine_root,
        successes_needed=successes_needed,
        certification_runs=certification_runs,
        certification_timeout_seconds=timeout,
        termination_grace_seconds=grace,
        startup_timeout_seconds=startup_timeout,
        max_attempts=max_attempts,
        base_seed=base_seed,
        recorder_args=tuple(recorder_args),
        session_id=session_id,
        status_file=status_file,
    )


def _kill_process_group(
    process: subprocess.Popen[Any],
    *,
    initial_signal: int,
    grace_seconds: float,
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, initial_signal)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=max(0.1, float(grace_seconds)))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=max(0.1, float(grace_seconds)))
    except subprocess.TimeoutExpired:
        pass


class SignalRelay:
    """Relay termination signals to a child that owns a separate process group."""

    def __init__(
        self,
        grace_seconds: float,
        *,
        status_file: Path | None = None,
    ):
        self.grace_seconds = float(grace_seconds)
        self.status_file = status_file
        self.active: subprocess.Popen[Any] | None = None
        self.requested_signal: int | None = None
        self._previous: dict[int, Any] = {}

    def _status(self, **patch: Any) -> None:
        if self.status_file is None:
            return
        current = _read_json(self.status_file)
        current.update(patch)
        current["updated_at"] = _now_iso()
        _write_json_atomic(self.status_file, current)

    def __enter__(self) -> "SignalRelay":
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        self._status(
            format="handson.fresh_process_supervisor_status",
            version=1,
            supervisor_pid=os.getpid(),
            supervisor_pgid=os.getpgrp(),
            status="running",
            active_child_pid=None,
            active_child_pgid=None,
            active_phase=None,
        )
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stop_active()
        self._status(
            status="stopped" if self.requested_signal else "idle",
            active_child_pid=None,
            active_child_pgid=None,
            active_phase=None,
        )
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)

    def _handle(self, signum: int, _frame: Any) -> None:
        if self.requested_signal is None:
            self.requested_signal = int(signum)
        process = self.active
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    def stop_active(self) -> None:
        process = self.active
        if process is None:
            return
        _kill_process_group(
            process,
            initial_signal=self.requested_signal or signal.SIGTERM,
            grace_seconds=self.grace_seconds,
        )
        self.active = None

    def raise_if_stop_requested(self) -> None:
        if self.requested_signal is not None:
            self.stop_active()
            raise StopRequested(self.requested_signal)

    def run(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: float | None = None,
        phase: str = "child",
        startup_ready_file: Path | None = None,
        startup_status_file: Path | None = None,
        startup_timeout_seconds: float | None = None,
    ) -> ProcessResult:
        self.raise_if_stop_requested()
        started_at = _now_iso()
        process = subprocess.Popen(
            list(command),
            cwd=str(REPO_ROOT),
            env=dict(env),
            start_new_session=True,
        )
        self.active = process
        self._status(
            status="running",
            active_phase=str(phase),
            active_child_pid=int(process.pid),
            active_child_pgid=int(process.pid),
            active_child_started_at=started_at,
        )
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + max(0.1, float(timeout_seconds))
        )
        startup_started_at = time.monotonic()
        startup_stall_deadline = (
            None
            if startup_ready_file is None
            else startup_started_at
            + max(0.1, float(startup_timeout_seconds or 60.0))
        )
        startup_absolute_timeout = max(
            float(startup_timeout_seconds or 60.0) * 10.0,
            float(startup_timeout_seconds or 60.0) + 60.0,
        )
        startup_absolute_deadline = (
            None
            if startup_ready_file is None
            else startup_started_at + startup_absolute_timeout
        )
        startup_status_mtime_ns: int | None = None
        startup_ready = startup_ready_file is None
        try:
            while process.poll() is None:
                if self.requested_signal is not None:
                    self.stop_active()
                    raise StopRequested(self.requested_signal)
                if not startup_ready:
                    startup_ready = bool(startup_ready_file.is_file())
                    if startup_ready:
                        self._status(
                            active_phase=str(phase),
                            startup_ready=True,
                            startup_ready_at=_now_iso(),
                        )
                    else:
                        try:
                            status_mtime_ns = (
                                startup_status_file.stat().st_mtime_ns
                                if startup_status_file is not None
                                else None
                            )
                        except (FileNotFoundError, OSError):
                            status_mtime_ns = None
                        if (
                            status_mtime_ns is not None
                            and status_mtime_ns != startup_status_mtime_ns
                        ):
                            startup_status_mtime_ns = status_mtime_ns
                            startup_stall_deadline = time.monotonic() + max(
                                0.1,
                                float(startup_timeout_seconds or 60.0),
                            )
                            self._status(
                                active_phase=str(phase),
                                startup_progress=_read_json(startup_status_file),
                                startup_progress_at=_now_iso(),
                            )
                    if (
                        not startup_ready
                        and startup_absolute_deadline is not None
                        and time.monotonic() >= startup_absolute_deadline
                    ):
                        self.stop_active()
                        raise ChildStartupTimeout(
                            "Isaac collection did not publish its startup-ready "
                            "marker within the absolute bounded startup window "
                            f"of {startup_absolute_timeout:.0f}s: "
                            f"{' '.join(command)}"
                        )
                    if (
                        not startup_ready
                        and startup_stall_deadline is not None
                        and time.monotonic() >= startup_stall_deadline
                    ):
                        self.stop_active()
                        raise ChildStartupTimeout(
                            "Isaac collection did not publish scene/stabilization "
                            "progress or its startup-ready marker for "
                            f"{startup_timeout_seconds or 60.0}s: "
                            f"{' '.join(command)}"
                        )
                if deadline is not None and time.monotonic() >= deadline:
                    self.stop_active()
                    raise ChildTimeout(
                        f"child process {process.pid} timed out after "
                        f"{timeout_seconds}s: {' '.join(command)}"
                    )
                time.sleep(0.1)
            if not startup_ready and startup_ready_file is not None:
                startup_ready = startup_ready_file.is_file()
            if not startup_ready:
                raise SupervisorError(
                    "Isaac recorder exited before publishing its startup-ready "
                    f"marker: pid={process.pid}, returncode={process.returncode}"
                )
            self.raise_if_stop_requested()
            return ProcessResult(
                pid=int(process.pid),
                returncode=int(process.returncode),
                started_at=started_at,
                finished_at=_now_iso(),
            )
        finally:
            if self.active is process:
                self.active = None
            self._status(
                active_child_pid=None,
                active_child_pgid=None,
                active_phase=None,
            )
            if startup_ready_file is not None:
                try:
                    startup_ready_file.unlink()
                except FileNotFoundError:
                    pass
            if startup_status_file is not None:
                try:
                    startup_status_file.unlink()
                except FileNotFoundError:
                    pass


def _episode_directories(attempt_root: Path) -> list[Path]:
    try:
        children = tuple(attempt_root.iterdir())
    except OSError:
        return []
    return sorted(
        child.resolve()
        for child in children
        if child.is_dir()
        and not child.name.startswith(".")
        and (child / "replay.zarr").is_dir()
    )


def _episode_integrity_error(demo_dir: Path) -> str | None:
    required_files = (
        demo_dir / "replay_manifest.json",
        demo_dir / "recording_schema.json",
        demo_dir / "collection_meta.json",
    )
    invalid: list[str] = []
    for path in required_files:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                invalid.append(path.name)
        except OSError:
            invalid.append(path.name)
    manifest = _read_json(demo_dir / "replay_manifest.json")
    metadata = _read_json(demo_dir / "collection_meta.json")
    recording = manifest.get("recording")
    recording = recording if isinstance(recording, Mapping) else {}
    task = manifest.get("task")
    task_name = task.get("name", "") if isinstance(task, Mapping) else ""
    modern_declaration = any(
        "canonical_camera_postrender_required" in payload
        or "online_camera_capture_scope" in payload
        for payload in (recording, metadata)
    )
    deferred_cameras = supports_isaacsim_deferred_camera_render(
        recording.get("sim_backend", ""), task_name,
    ) and all(
        (
            payload.get("canonical_camera_postrender_required") is True
            and payload.get("online_camera_capture_scope") == "all_canonical_postrender"
        ) if modern_declaration else payload.get("isaac_camera_render_deferred") is True
        for payload in (recording, metadata)
    )
    if not deferred_cameras:
        videos_dir = demo_dir / "videos"
        try:
            videos = tuple(videos_dir.glob("*.mp4")) if videos_dir.is_dir() else ()
            if not videos or any(video.stat().st_size <= 0 for video in videos):
                invalid.append("videos/*.mp4")
        except OSError:
            invalid.append("videos/*.mp4")
    if invalid:
        return "incomplete atomically-published episode artifacts: " + ", ".join(
            invalid
        )
    return None


def _certifier_environment(demo_dir: Path) -> dict[str, str]:
    manifest = _read_json(demo_dir / "replay_manifest.json")
    environment = manifest.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    recorded = environment.get("variables")
    if not isinstance(recorded, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in recorded.items()
    ):
        raise SupervisorError(
            f"invalid recorded environment variables in {demo_dir}"
        )
    child_env = os.environ.copy()
    for name in tuple(child_env):
        if name.startswith("HANDSON_") or name in _DETERMINISTIC_ENVIRONMENT_NAMES:
            child_env.pop(name, None)
    child_env.update(recorded)
    child_env.pop("HANDSON_FRESH_CERTIFIED_COLLECTION_WORKER", None)
    pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(PACKAGE_ROOT), str(REPO_ROOT), pythonpath)
        if part
    )
    return child_env


def _recording_task_name(config: SupervisorConfig) -> str:
    return _option_value(config.recorder_args, "exp_name", "water_plant").strip()


def _deferred_arm_requested(config: SupervisorConfig) -> bool:
    task_name = _recording_task_name(config).lower()
    configured = os.environ.get("HANDSON_ISAACLAB_DEFER_ARM_RENDER", "0")
    return bool(
        (
            task_name[:1] in {"a", "b", "c", "d"}
            or task_name.startswith("ycb")
            or task_name in ISAACSIM_NONSTANDARD_DEFERRED_RENDER_TASKS
        )
        and configured.strip().lower() not in {"0", "false", "no", "off", ""}
    )


def _canonical_camera_postrender_requested(
    config: SupervisorConfig,
) -> bool:
    """Return whether the accepted episode needs the shared camera worker."""

    return supports_isaacsim_deferred_camera_render(
        _option_value(config.recorder_args, "sim_backend", "mujoco"),
        _recording_task_name(config),
    )


def _recorder_environment(
    config: SupervisorConfig,
    *,
    startup_ready_file: Path,
    startup_status_file: Path,
) -> dict[str, str]:
    child_env = os.environ.copy()
    # The fallback contract is one fresh source process per attempt.  A stale
    # service/shell setting for the persistent deferred mode must never turn
    # its child into a hidden pending producer that this supervisor cannot
    # discover or publish.
    child_env.pop("HANDSON_DEFER_PHYSICAL_REPLAY_CERTIFICATION", None)
    # A failed stabilization attempt mutates the same PhysX scene before the
    # next reset.  One attempt per process is required for a genuinely fresh
    # episode origin; an unstable origin therefore rejects this whole attempt.
    child_env["HANDSON_STABLE_ORIGIN_RESET_ATTEMPTS"] = "1"
    child_env["HANDSON_FRESH_CERTIFIED_COLLECTION_WORKER"] = "1"
    child_env["HANDSON_ISAAC_STARTUP_READY_FILE"] = str(startup_ready_file)
    child_env["HANDSON_ISAAC_WORKER_STATUS_FILE"] = str(startup_status_file)
    child_env["HANDSON_ISAAC_STARTUP_TIMEOUT_SECONDS"] = str(
        config.startup_timeout_seconds
    )
    # Never leave a detached worker holding a staging path that will move after
    # certification.  E/F postprocessing finishes synchronously in staging.
    child_env["HANDSON_E01_POSTPROCESS_BACKGROUND"] = "0"
    if (
        _deferred_arm_requested(config)
        or _canonical_camera_postrender_requested(config)
    ):
        # Run the source collector directly.  The ordinary recorder supervisor
        # would launch a background canonical camera worker against staging;
        # this supervisor queues it only after formal publication.
        child_env["HANDSON_ISAAC_DEFERRED_ARM_WORKER"] = "1"
    if _canonical_camera_postrender_requested(config):
        child_env["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "1"
        child_env["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"] = "0"
    return child_env


def _bind_manifest_to_supervisor_contract(
    demo_dir: Path,
    *,
    certification_runs: int,
) -> dict[str, Any]:
    manifest_path = demo_dir / "replay_manifest.json"
    manifest = _read_json(manifest_path)
    recording = manifest.get("recording")
    if not isinstance(recording, dict):
        raise SupervisorError(f"missing recording manifest contract in {demo_dir}")
    if recording.get("physical_replay_certification_required") is not True:
        raise SupervisorError(
            f"recorder did not require physical replay certification: {demo_dir}"
        )
    task = manifest.get("task")
    task_name = (
        str(task.get("name") or "").strip()
        if isinstance(task, Mapping)
        else ""
    )
    transition_contract = recording.get("physical_transition_contract")
    if transition_contract == LEGACY_RAW_ACTION_TRANSITION_CONTRACT:
        if task_name != "e01_fold_garment":
            raise SupervisorError(
                "legacy raw-action physical replay is allowed only for "
                f"task.name=e01_fold_garment, got {task_name!r}: {demo_dir}"
            )
    elif transition_contract != RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT:
        raise SupervisorError(
            "fresh recorder must use the recorded-drive-target physical "
            f"transition contract, got {transition_contract!r}: {demo_dir}"
        )
    episode_index = recording.get("episode_index")
    if type(episode_index) is not int or episode_index != 1:
        raise SupervisorError(
            "fresh recorder must publish episode_index=1, got "
            f"{episode_index!r}: {demo_dir}"
        )
    collection_run_seed = recording.get("collection_run_seed")
    episode_reset_seed = recording.get("episode_reset_seed")
    if (
        type(collection_run_seed) is not int
        or type(episode_reset_seed) is not int
        or episode_reset_seed != collection_run_seed
    ):
        raise SupervisorError(
            "fresh recorder requires integer episode_reset_seed equal to "
            f"collection_run_seed, got {episode_reset_seed!r} versus "
            f"{collection_run_seed!r}: {demo_dir}"
        )
    stable_origin = recording.get("stable_recording_origin")
    if not isinstance(stable_origin, Mapping):
        raise SupervisorError(
            f"fresh recorder is missing stable_recording_origin: {demo_dir}"
        )
    stabilization_attempt = stable_origin.get("reset_stabilization_attempt")
    stabilization_attempts_allowed = stable_origin.get(
        "reset_stabilization_attempts_allowed"
    )
    if (
        type(stabilization_attempt) is not int
        or stabilization_attempt != 1
        or type(stabilization_attempts_allowed) is not int
        or stabilization_attempts_allowed != 1
    ):
        raise SupervisorError(
            "fresh recorder must use stabilization attempt 1/1, got "
            f"{stabilization_attempt!r}/{stabilization_attempts_allowed!r}: "
            f"{demo_dir}"
        )
    recording["physical_replay_certification_runs_required"] = int(
        certification_runs
    )
    recording["fresh_recorder_process_per_episode_required"] = True
    recording["fresh_recorder_process_per_attempt_required"] = True
    recording["publish_only_after_fresh_certification"] = True
    manifest["recording"] = recording
    _write_json_atomic(manifest_path, manifest)
    return recording


def _recorder_command(config: SupervisorConfig, attempt: int, attempt_root: Path) -> list[str]:
    arguments = list(config.recorder_args)
    arguments = _replace_option(arguments, "successes_needed", 1)
    arguments = _replace_option(arguments, "out_dir", attempt_root)
    arguments = _replace_option(
        arguments,
        "certify_physical_replay",
        False,
        boolean=True,
    )
    arguments = _replace_option(
        arguments,
        "record_physx_ground_truth",
        True,
        boolean=True,
    )
    attempt_seed = int((config.base_seed + attempt - 1) % (2**31))
    arguments = _replace_option(arguments, "seed", attempt_seed)
    return [sys.executable, str(RECORDER), *arguments]


def _quarantine_attempt(
    config: SupervisorConfig,
    attempt_root: Path,
    *,
    attempt: int,
    payload: Mapping[str, Any],
) -> Path:
    _write_json_atomic(attempt_root / REJECTION_NAME, payload)
    destination_parent = config.quarantine_root / config.session_id
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination = destination_parent / f"attempt_{attempt:06d}"
    if destination.exists():
        destination = destination_parent / (
            f"attempt_{attempt:06d}-{secrets.token_hex(4)}"
        )
    os.replace(attempt_root, destination)
    return destination


def _launch_deferred_arm_replays(
    config: SupervisorConfig,
    demos: Sequence[Path],
) -> int | None:
    if not demos or not (
        _deferred_arm_requested(config)
        or _canonical_camera_postrender_requested(config)
    ):
        return None
    video_fps = max(
        1,
        int(_option_value(config.recorder_args, "video_fps", "30")),
    )
    task_name = _recording_task_name(config)
    log_dir = config.out_dir / ".deferred_arm_replay_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (
        f"{config.session_id}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    )
    command = [
        sys.executable,
        str(DEFERRED_ARM_WORKER),
        f"--exp_name={task_name}",
        f"--video_fps={video_fps}",
        f"--log_path={log_path}",
        *[str(path.resolve()) for path in demos],
    ]
    queued_at = _now_iso()
    for demo in demos:
        _write_json_atomic(
            demo / "deferred_arm_replay_status.json",
            {
                "status": "queued",
                "queued_at": queued_at,
                "updated_at": queued_at,
                "exp_name": task_name,
                "video_fps": video_fps,
                "command": command,
                "log_path": str(log_path),
                "paths_are_formally_published": True,
            },
        )
    child_env = os.environ.copy()
    child_env.pop("HANDSON_ISAAC_DEFERRED_ARM_WORKER", None)
    pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(PACKAGE_ROOT), str(REPO_ROOT), pythonpath)
        if part
    )
    try:
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=child_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except Exception as exc:
        failed_at = _now_iso()
        for demo in demos:
            _write_json_atomic(
                demo / "deferred_arm_replay_status.json",
                {
                    "status": "failed",
                    "queued_at": queued_at,
                    "updated_at": failed_at,
                    "completed_at": failed_at,
                    "exp_name": task_name,
                    "video_fps": video_fps,
                    "command": command,
                    "log_path": str(log_path),
                    "paths_are_formally_published": True,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        raise SupervisorError(
            f"could not launch deferred arm replay for accepted demos: {exc}"
        ) from exc
    print(
        "[Fresh PhysX supervisor] queued deferred arm replay only after "
        f"publication: pid={process.pid}, demos={len(demos)}, log={log_path}",
        flush=True,
    )
    return int(process.pid)


def _certify_episode(
    config: SupervisorConfig,
    demo_dir: Path,
    relay: SignalRelay,
) -> tuple[bool, list[dict[str, Any]], str]:
    reports_dir = demo_dir / RUN_REPORT_DIRECTORY
    reports_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    child_env = _certifier_environment(demo_dir)
    for run_index in range(1, config.certification_runs + 1):
        report_path = demo_dir / REPORT_NAME
        try:
            report_path.unlink()
        except FileNotFoundError:
            pass
        command = [
            sys.executable,
            str(CERTIFIER),
            "--demo-dir",
            str(demo_dir),
        ]
        try:
            process = relay.run(
                command,
                env=child_env,
                timeout_seconds=config.certification_timeout_seconds,
                phase=f"certification_{run_index}_of_{config.certification_runs}",
            )
        except ChildTimeout as exc:
            return False, results, str(exc)
        report = _read_json(report_path)
        if not report:
            raise SupervisorError(
                f"certifier produced no readable {REPORT_NAME} for {demo_dir}"
            )
        proof_path = reports_dir / f"run_{run_index:03d}.json"
        shutil.copy2(report_path, proof_path)
        result = {
            "run": run_index,
            "pid": process.pid,
            "returncode": process.returncode,
            "started_at": process.started_at,
            "finished_at": process.finished_at,
            "status": str(report.get("status") or "invalid"),
            "reason": str(report.get("reason") or ""),
            "report": proof_path.relative_to(demo_dir).as_posix(),
            "report_sha256": _sha256_file(proof_path),
            "certified_inputs": report.get("certified_inputs"),
        }
        results.append(result)
        if process.returncode < 0:
            return False, results, (
                "certifier_terminated_by_signal_"
                f"{-int(process.returncode)}"
            )
        if report.get("status") != "passed":
            return False, results, (
                str(report.get("reason") or "certification failed")
            )
        blocker = certification_report_blocker(demo_dir, report)
        if blocker is not None:
            raise SupervisorError(
                "certifier claimed passed but the upload contract rejected it: "
                + str(blocker.get("detail") or blocker.get("code") or blocker)
            )
    return True, results, ""


def supervise(config: SupervisorConfig) -> int:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.staging_root.mkdir(parents=True, exist_ok=True)
    config.quarantine_root.mkdir(parents=True, exist_ok=True)
    if config.out_dir.stat().st_dev != config.staging_root.stat().st_dev:
        raise SupervisorError(
            "fresh staging root and formal out_dir must share a filesystem "
            "for atomic publication"
        )
    if config.out_dir.stat().st_dev != config.quarantine_root.stat().st_dev:
        raise SupervisorError(
            "fresh quarantine root and formal out_dir must share a filesystem"
        )

    session_root = config.staging_root / config.session_id
    session_root.mkdir(parents=True, exist_ok=False)
    accepted = 0
    attempt = 0
    published_demos: list[Path] = []
    print(
        "[Fresh PhysX supervisor] fail-closed collection enabled: "
        f"target={config.successes_needed}, certifications_per_episode="
        f"{config.certification_runs}, staging={session_root}, "
        f"quarantine={config.quarantine_root / config.session_id}",
        flush=True,
    )

    try:
        with SignalRelay(
            config.termination_grace_seconds,
            status_file=config.status_file,
        ) as relay:
            while accepted < config.successes_needed:
                attempt += 1
                if config.max_attempts and attempt > config.max_attempts:
                    raise SupervisorError(
                        f"reached fresh collection attempt limit {config.max_attempts} "
                        f"with only {accepted}/{config.successes_needed} certified"
                    )
                attempt_root = session_root / f"attempt_{attempt:06d}"
                attempt_root.mkdir(parents=True, exist_ok=False)
                startup_ready_file = attempt_root / ".isaac_startup_ready.json"
                startup_status_file = attempt_root / ".isaac_startup_status.json"
                command = _recorder_command(config, attempt, attempt_root)
                print(
                    f"[Fresh PhysX supervisor] recording attempt {attempt} in a "
                    f"new process: {' '.join(command)}",
                    flush=True,
                )
                try:
                    recording = relay.run(
                        command,
                        env=_recorder_environment(
                            config,
                            startup_ready_file=startup_ready_file,
                            startup_status_file=startup_status_file,
                        ),
                        phase="recording",
                        startup_ready_file=startup_ready_file,
                        startup_status_file=startup_status_file,
                        startup_timeout_seconds=config.startup_timeout_seconds,
                    )
                except StopRequested:
                    if attempt_root.exists():
                        destination = _quarantine_attempt(
                            config,
                            attempt_root,
                            attempt=attempt,
                            payload={
                                "format": "handson.fresh_process_rejection",
                                "version": 1,
                                "status": "interrupted",
                                "attempt": attempt,
                                "recorded_at": _now_iso(),
                            },
                        )
                        print(
                            f"[Fresh PhysX supervisor] interrupted attempt isolated at "
                            f"{destination}",
                            flush=True,
                        )
                    raise
                except (ChildStartupTimeout, SupervisorError) as exc:
                    destination = _quarantine_attempt(
                        config,
                        attempt_root,
                        attempt=attempt,
                        payload={
                            "format": "handson.fresh_process_rejection",
                            "version": 1,
                            "status": "recorder_startup_failed",
                            "attempt": attempt,
                            "reason": str(exc),
                            "recorded_at": _now_iso(),
                        },
                    )
                    raise SupervisorError(
                        f"Isaac recorder startup failed; attempt quarantined at "
                        f"{destination}: {exc}"
                    ) from exc
                demos = _episode_directories(attempt_root)
                if not demos:
                    rejected_attempt = (
                        recording.returncode == _RETRYABLE_ATTEMPT_EXIT_CODE
                    )
                    destination = _quarantine_attempt(
                        config,
                        attempt_root,
                        attempt=attempt,
                        payload={
                            "format": "handson.fresh_process_rejection",
                            "version": 1,
                            "status": (
                                "recording_attempt_rejected"
                                if rejected_attempt
                                else "no_successful_episode"
                            ),
                            "attempt": attempt,
                            "recorder": recording.__dict__,
                            "episode_count": 0,
                            "retryable": bool(rejected_attempt),
                            "reason": (
                                "fresh recorder rejected this episode attempt "
                                "and exited before reusing its PhysX scene"
                                if rejected_attempt
                                else "recorder exited without a completed episode"
                            ),
                            "recorded_at": _now_iso(),
                        },
                    )
                    print(
                        "[Fresh PhysX supervisor] "
                        + (
                            "recording attempt was rejected"
                            if rejected_attempt
                            else "no completed episode"
                        )
                        + "; retrying "
                        f"in a new process. Quarantine: {destination}",
                        flush=True,
                    )
                    continue
                if len(demos) != 1:
                    destination = _quarantine_attempt(
                        config,
                        attempt_root,
                        attempt=attempt,
                        payload={
                            "format": "handson.fresh_process_rejection",
                            "version": 1,
                            "status": "invalid_episode_count",
                            "attempt": attempt,
                            "recorder": recording.__dict__,
                            "episode_count": len(demos),
                            "recorded_at": _now_iso(),
                        },
                    )
                    raise SupervisorError(
                        "one-episode child produced multiple episodes; "
                        f"quarantine={destination}"
                    )

                demo_dir = demos[0]
                if recording.returncode < 0:
                    destination = _quarantine_attempt(
                        config,
                        attempt_root,
                        attempt=attempt,
                        payload={
                            "format": "handson.fresh_process_rejection",
                            "version": 1,
                            "status": "recorder_terminated_by_signal",
                            "attempt": attempt,
                            "recorder": recording.__dict__,
                            "recorded_at": _now_iso(),
                        },
                    )
                    raise SupervisorError(
                        "recorder was terminated by signal; refusing to trust "
                        f"saved artifacts at {destination}"
                    )
                integrity_error = _episode_integrity_error(demo_dir)
                if integrity_error is not None:
                    destination = _quarantine_attempt(
                        config,
                        attempt_root,
                        attempt=attempt,
                        payload={
                            "format": "handson.fresh_process_rejection",
                            "version": 1,
                            "status": "incomplete_episode_artifacts",
                            "attempt": attempt,
                            "reason": integrity_error,
                            "recorder": recording.__dict__,
                            "recorded_at": _now_iso(),
                        },
                    )
                    raise SupervisorError(
                        f"{integrity_error}; quarantine={destination}"
                    )
                try:
                    recording_contract = _bind_manifest_to_supervisor_contract(
                        demo_dir,
                        certification_runs=config.certification_runs,
                    )
                except (OSError, SupervisorError) as exc:
                    destination = _quarantine_attempt(
                        config,
                        attempt_root,
                        attempt=attempt,
                        payload={
                            "format": "handson.fresh_process_rejection",
                            "version": 1,
                            "status": "invalid_recording_contract",
                            "attempt": attempt,
                            "reason": str(exc),
                            "recorder": recording.__dict__,
                            "recorded_at": _now_iso(),
                        },
                    )
                    raise SupervisorError(
                        f"invalid recording contract; quarantine={destination}: {exc}"
                    ) from exc
                print(
                    f"[Fresh PhysX supervisor] recorder process {recording.pid} "
                    f"exited rc={recording.returncode}; certifying {demo_dir} "
                    "in fresh process(es)",
                    flush=True,
                )
                try:
                    passed, certification_runs, failure_reason = _certify_episode(
                        config,
                        demo_dir,
                        relay,
                    )
                except StopRequested:
                    destination = _quarantine_attempt(
                        config,
                        attempt_root,
                        attempt=attempt,
                        payload={
                            "format": "handson.fresh_process_rejection",
                            "version": 1,
                            "status": "interrupted",
                            "attempt": attempt,
                            "recorder": recording.__dict__,
                            "recorded_at": _now_iso(),
                        },
                    )
                    print(
                        f"[Fresh PhysX supervisor] interrupted attempt isolated at "
                        f"{destination}",
                        flush=True,
                    )
                    raise
                except SupervisorError as exc:
                    destination = _quarantine_attempt(
                        config,
                        attempt_root,
                        attempt=attempt,
                        payload={
                            "format": "handson.fresh_process_rejection",
                            "version": 1,
                            "status": "certifier_error",
                            "attempt": attempt,
                            "reason": str(exc),
                            "recorder": recording.__dict__,
                            "recorded_at": _now_iso(),
                        },
                    )
                    raise SupervisorError(
                        f"certifier infrastructure error; quarantine={destination}: {exc}"
                    ) from exc
                if not passed:
                    destination = _quarantine_attempt(
                        config,
                        attempt_root,
                        attempt=attempt,
                        payload={
                            "format": "handson.fresh_process_rejection",
                            "version": 1,
                            "status": "certification_failed",
                            "attempt": attempt,
                            "reason": failure_reason,
                            "recorder": recording.__dict__,
                            "certification_runs": certification_runs,
                            "recorded_at": _now_iso(),
                        },
                    )
                    print(
                        "[Fresh PhysX supervisor] REJECTED; formal dataset unchanged. "
                        f"Reason: {failure_reason}. Quarantine: {destination}",
                        flush=True,
                    )
                    continue

                try:
                    relay.raise_if_stop_requested()
                    certified_inputs = capture_certified_inputs(demo_dir)
                    acceptance = {
                        "format": ACCEPTANCE_FORMAT,
                        "version": ACCEPTANCE_VERSION,
                        "status": "passed",
                        "policy": "all_fresh_process_certification_runs_must_pass",
                        "accepted_at": _now_iso(),
                        "session_id": config.session_id,
                        "attempt": attempt,
                        "certified_success_index": accepted + 1,
                        "certification_runs_required": config.certification_runs,
                        "certification_runs_passed": len(certification_runs),
                        "recorder_pid": recording.pid,
                        "recorder": recording.__dict__,
                        "certifier_pids": [
                            run["pid"] for run in certification_runs
                        ],
                        "certification_runs": certification_runs,
                        "certified_inputs": certified_inputs,
                        "physical_transition_contract": recording_contract.get(
                            "physical_transition_contract"
                        ),
                        "trajectory_input_mode": recording_contract.get(
                            "trajectory_input_mode"
                        ),
                        "fresh_recorder_process_per_episode": True,
                        "fresh_recorder_process_per_attempt": True,
                        "fresh_certifier_process_per_run": True,
                        "published_only_after_certification": True,
                    }
                    _write_json_atomic(demo_dir / ACCEPTANCE_NAME, acceptance)
                    blocker = certification_upload_blocker(demo_dir)
                    if blocker is None:
                        blocker = fresh_process_acceptance_upload_blocker(
                            demo_dir
                        )
                    if blocker is not None:
                        raise SupervisorError(
                            "final certification became invalid before publication: "
                            + str(
                                blocker.get("detail")
                                or blocker.get("code")
                                or blocker
                            )
                        )
                    destination = config.out_dir / demo_dir.name
                    if destination.exists():
                        raise SupervisorError(
                            f"refusing to replace existing formal episode {destination}"
                        )
                    relay.raise_if_stop_requested()
                    os.replace(demo_dir, destination)
                except StopRequested:
                    destination = _quarantine_attempt(
                        config,
                        attempt_root,
                        attempt=attempt,
                        payload={
                            "format": "handson.fresh_process_rejection",
                            "version": 1,
                            "status": "interrupted_before_publication",
                            "attempt": attempt,
                            "recorder": recording.__dict__,
                            "certification_runs": certification_runs,
                            "recorded_at": _now_iso(),
                        },
                    )
                    print(
                        "[Fresh PhysX supervisor] stop won the publication "
                        f"race; certified attempt isolated at {destination}",
                        flush=True,
                    )
                    raise
                except (OSError, SupervisorError) as exc:
                    if attempt_root.exists():
                        quarantine = _quarantine_attempt(
                            config,
                            attempt_root,
                            attempt=attempt,
                            payload={
                                "format": "handson.fresh_process_rejection",
                                "version": 1,
                                "status": "publication_contract_error",
                                "attempt": attempt,
                                "reason": str(exc),
                                "recorder": recording.__dict__,
                                "certification_runs": certification_runs,
                                "recorded_at": _now_iso(),
                            },
                        )
                        raise SupervisorError(
                            f"publication contract failed; quarantine={quarantine}: {exc}"
                        ) from exc
                    raise
                try:
                    attempt_root.rmdir()
                except OSError:
                    # Non-episode diagnostic logs may remain private in staging.
                    pass
                accepted += 1
                published_demos.append(destination)
                print(
                    f"[Fresh PhysX supervisor] ACCEPTED {accepted}/"
                    f"{config.successes_needed}: {destination}",
                    flush=True,
                )
    except StopRequested:
        # An explicit operator stop must not start new detached GPU work.
        raise
    except (OSError, SupervisorError):
        _launch_deferred_arm_replays(config, published_demos)
        raise

    _launch_deferred_arm_replays(config, published_demos)

    try:
        session_root.rmdir()
    except OSError:
        pass
    print(
        f"[Fresh PhysX supervisor] complete: {accepted}/"
        f"{config.successes_needed} formal episodes passed every fresh replay",
        flush=True,
    )
    return 0


def _should_passthrough(arguments: Sequence[str]) -> bool:
    return bool(
        _option_value(arguments, "sim_backend", "mujoco") != "isaacsim"
        or _bool_option(arguments, "preview_only", False)
        or "--help" in arguments
        or "-h" in arguments
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    _, recorder_arguments = _consume_supervisor_options(arguments)
    if _should_passthrough(recorder_arguments):
        os.execv(
            sys.executable,
            [sys.executable, str(RECORDER), *recorder_arguments],
        )
        raise AssertionError("os.execv unexpectedly returned")
    try:
        return supervise(build_config(arguments))
    except StopRequested as exc:
        print(f"[Fresh PhysX supervisor] stopped by signal {exc.signum}", flush=True)
        return 128 + exc.signum
    except (OSError, ValueError, SupervisorError) as exc:
        print(f"[Fresh PhysX supervisor] FAILED CLOSED: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
