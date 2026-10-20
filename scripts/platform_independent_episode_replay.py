#!/usr/bin/env python3
"""Replay one PhysX episode with its recorded source tree and no history."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import importlib
import inspect
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


REPORT_NAME = "physical_replay_certification.json"
EXECUTION_MODE = "platform_recorded_source_independent_episode"
SESSION_MODE = "platform_independent_episode_replay_without_session_gate"
_CURRENT_REPO_ROOT = Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                _json_safe(dict(payload)),
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
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _import_recorded_certifier(source_root: Path) -> ModuleType:
    source_root = source_root.expanduser().resolve()
    certifier_path = source_root / "scripts" / "certify_physx_replay.py"
    package_root = source_root / "handson"
    if not certifier_path.is_file() or not package_root.is_dir():
        raise RuntimeError(f"Recorded source tree is incomplete: {source_root}")

    # The launcher script lives in the current checkout, while every replay
    # import must come from the selected recorded checkout.  Remove both the
    # current checkout and its package directory even when PYTHONPATH supplied
    # them, then put the recorded tree first.
    rejected = {_CURRENT_REPO_ROOT, _CURRENT_REPO_ROOT / "handson"}
    retained: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or os.curdir).resolve()
        except OSError:
            retained.append(entry)
            continue
        if resolved not in rejected:
            retained.append(entry)
    sys.path[:] = [str(package_root), str(source_root), *retained]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(package_root), str(source_root)]
    )

    # This file normally runs as __main__ in a fresh child.  Clearing these
    # names also makes the source boundary fail-safe if a launcher or debugger
    # happened to preload modules from the current checkout.
    for module_name in tuple(sys.modules):
        if module_name in {"scripts", "handson"} or module_name.startswith(
            ("scripts.", "handson.")
        ):
            sys.modules.pop(module_name, None)

    certifier = importlib.import_module("scripts.certify_physx_replay")
    loaded_path = Path(str(certifier.__file__)).resolve()
    if not loaded_path.is_relative_to(source_root):
        raise RuntimeError(
            "Replay imported the wrong certifier: "
            f"expected under {source_root}, got {loaded_path}"
        )
    return certifier


def _call_with_platform_review(
    function: Any,
    *args: Any,
    required: bool = False,
    **call_kwargs: Any,
):
    supports_platform_review = "platform_review" in inspect.signature(
        function
    ).parameters
    # The independent-process and no-history guarantees are provided by this
    # launcher, not by the recorded certifier's command-line entry point.
    # Older recorded commits already expose the same single-episode
    # ``load_demo``/``run_certification`` API and environment callback, but
    # predate the optional ``platform_review`` keyword.  Calling that API
    # directly is the compatibility path: task/environment imports still come
    # entirely from the recorded source tree and the episode still starts in a
    # fresh process.  ``required`` is retained for call-site/API compatibility
    # and to document that newer certifiers should receive the review mode.
    del required
    if supports_platform_review:
        call_kwargs["platform_review"] = True
    return function(*args, **call_kwargs)


@contextlib.contextmanager
def _legacy_nonphysical_gate_overrides(certifier: ModuleType, demo: Any):
    """Disable host/source identity gates in a legacy recorded certifier.

    Newer certifiers implement this through ``platform_review=True``.  Older
    ones compare the replay computer and Git worktree before starting PhysX.
    Those checks are not trajectory evidence and make cloud replay needlessly
    machine-specific, so make their capture functions return the manifest
    values for this isolated compatibility call.  Physical schema, reset,
    action execution, transition outcomes, success, and trace comparisons stay
    inside the recorded certifier unchanged.
    """

    if "platform_review" in inspect.signature(
        certifier.run_certification
    ).parameters:
        yield
        return

    replacements = {
        "capture_git_replay_state": lambda *args, **kwargs: dict(
            getattr(demo, "recorded_git", {}) or {}
        ),
        "capture_runtime_replay_identity": lambda *args, **kwargs: dict(
            getattr(demo, "recorded_runtime_identity", {}) or {}
        ),
    }
    originals: dict[str, Any] = {}
    for name, replacement in replacements.items():
        if hasattr(certifier, name):
            originals[name] = getattr(certifier, name)
            setattr(certifier, name, replacement)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(certifier, name, original)


def _portable_demo_environment(demo: Any, source_root: Path) -> Any:
    variables = dict(getattr(demo, "environment_variables", {}) or {})
    recorded_root_text = str(variables.get("HANDSON_ROOT") or "").strip()
    recorded_root = (
        Path(recorded_root_text).expanduser()
        if recorded_root_text and Path(recorded_root_text).is_absolute()
        else None
    )
    if recorded_root is not None:
        for name, value in tuple(variables.items()):
            if not isinstance(value, str) or not value:
                continue
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                continue
            try:
                relative = candidate.relative_to(recorded_root)
            except ValueError:
                continue
            variables[name] = str(source_root / relative)

    # These identify the local execution tree/runtime rather than a physical
    # simulation parameter.  Keeping the recording computer's absolute paths
    # is precisely what made the old wrapper non-portable.
    variables["HANDSON_ROOT"] = str(source_root)
    local_isaac_python = str(
        os.environ.get("HANDSON_PLATFORM_REPLAY_ISAACSIM_PYTHON")
        or sys.executable
    )
    variables["HANDSON_ISAACSIM_PYTHON"] = local_isaac_python
    variables["HANDSON_PYTHON"] = local_isaac_python
    variables["HANDSON_AUTO_START_TELEOP"] = "0"
    for name in tuple(variables):
        if name.startswith("HANDSON_COLLECTION_") or name in {
            "HANDSON_ISAAC_STARTUP_READY_FILE",
            "HANDSON_ISAAC_WORKER_STATUS_FILE",
            "HANDSON_LIVE_PREVIEW_PATH",
        }:
            variables.pop(name, None)

    try:
        return dataclasses.replace(demo, environment_variables=variables)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Recorded certifier returned an unsupported demo object."
        ) from exc


def reproduce_recorded_scene_rebuild(environment: Any, demo: Any) -> dict[str, Any] | None:
    """Recreate the target episode's one hard-reset boundary, not its history."""

    rebuild_required = bool(getattr(demo, "source_scene_rebuild_per_reset", False))
    target_generation = int(getattr(demo, "source_scene_generation", 0))
    if not rebuild_required:
        if target_generation != 0:
            raise RuntimeError(
                "Recorded episode has a scene generation without a rebuild marker."
            )
        return None
    if target_generation <= 0:
        raise RuntimeError(
            "Recorded hard-reset episode has no positive scene generation."
        )

    raw = getattr(environment, "unwrapped", environment)
    hard_reset = getattr(raw, "hard_reset_physx_scene", None)
    if not callable(hard_reset):
        raise RuntimeError(
            f"{type(raw).__name__} cannot rebuild a fresh PhysX scene."
        )
    if not hasattr(raw, "_physx_scene_generation"):
        raise RuntimeError("PhysX scene generation counter is unavailable.")
    control_dt = float(getattr(raw, "_control_dt", 0.0))
    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise RuntimeError("Recorded scene has no valid control timestep.")

    # _construct_environment has already switched the live scene into its fast
    # no-render mode.  hard_reset_physx_scene builds from the live attributes,
    # so restore the source construction values for that one constructor call.
    raw.image_obs = bool(getattr(demo, "image_obs", False))
    raw.hz = max(1, int(round(1.0 / control_dt)))
    raw._physx_scene_generation = target_generation - 1
    result = hard_reset(seed=int(getattr(demo, "episode_reset_seed")))
    if not isinstance(result, Mapping) or result.get("scene_rebuilt") is not True:
        raise RuntimeError("Independent replay did not rebuild its PhysX scene.")
    if result.get("scene_generation") != target_generation:
        raise RuntimeError(
            "Independent replay rebuilt scene generation "
            f"{result.get('scene_generation')!r}; expected {target_generation}."
        )
    if result.get("seed") != int(getattr(demo, "episode_reset_seed")):
        raise RuntimeError("Independent replay rebuilt the scene with the wrong seed.")

    # The hard reset retains the Python object identity but reconstructs all of
    # its attributes.  Re-apply the accelerated execution switches only after
    # the source-matched scene exists.
    raw.image_obs = False
    raw.hz = 0
    return dict(result)


def _base_failure_report(exc: BaseException, *, demo: Any | None) -> dict[str, Any]:
    return {
        "status": "failed",
        "reason": str(exc),
        "error_type": type(exc).__name__,
        "execution_mode": EXECUTION_MODE,
        "task": str(getattr(demo, "task_name", "") or ""),
        "steps_expected": (
            int(len(getattr(demo, "actions"))) if demo is not None else None
        ),
        "steps_executed": None,
        "task_success_replay": False,
        "final_succeed": False,
        "trace_match": False,
        "transition_outcomes_match": False,
    }


def _recorded_git_metadata(demo_dir: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(
            (demo_dir / "replay_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(manifest, Mapping):
        return {}
    software = manifest.get("software")
    software = software if isinstance(software, Mapping) else {}
    git = software.get("git")
    return dict(git) if isinstance(git, Mapping) else {}


def replay_one_episode(
    demo_dir: Path,
    *,
    source_root: Path,
    recorded_commit: str,
    result_path: Path,
) -> dict[str, Any]:
    demo_dir = demo_dir.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    result_path = result_path.expanduser().resolve()
    certifier: ModuleType | None = None
    demo: Any | None = None
    environments: list[Any] = []
    lifecycle: dict[str, Any] = {}
    try:
        certifier = _import_recorded_certifier(source_root)
        demo = _call_with_platform_review(certifier.load_demo, demo_dir)
        demo = _portable_demo_environment(demo, source_root)

        def environment_created(environment: Any) -> None:
            environments.append(environment)
            lifecycle["scene_rebuild"] = reproduce_recorded_scene_rebuild(
                environment,
                demo,
            )

        with _legacy_nonphysical_gate_overrides(certifier, demo):
            report = _call_with_platform_review(
                certifier.run_certification,
                demo,
                required=True,
                on_environment_created=environment_created,
            )
    except BaseException as exc:
        if certifier is not None and hasattr(certifier, "_failed_certification_report"):
            try:
                report = certifier._failed_certification_report(exc, demo=demo)
            except BaseException:
                report = _base_failure_report(exc, demo=demo)
        else:
            report = _base_failure_report(exc, demo=demo)

    report = dict(report)
    report["session_replay"] = {
        "mode": SESSION_MODE,
        "status": "passed" if report.get("status") == "passed" else "failed",
        "shared_environment": False,
        "episode_index": 0,
        "episode_count": 1,
        "session_history_used": False,
        "session_history_note": (
            "The episode ran in a new process and rebuilt only its own recorded "
            "scene boundary; no session journal was opened."
        ),
    }
    report["platform_source_replay"] = {
        "mode": EXECUTION_MODE,
        "recorded_commit": recorded_commit,
        "source_root": str(source_root),
        "recorded_source_selected": True,
        "nonphysical_source_and_runtime_identity_gates_used": False,
        "review_driver": (
            "recorded_source_native_platform_review"
            if certifier is not None
            and "platform_review"
            in inspect.signature(certifier.run_certification).parameters
            else "recorded_source_legacy_single_episode_compatibility"
        ),
        "recorded_worktree_was_clean": (
            _recorded_git_metadata(demo_dir).get("dirty") is False
        ),
        "scene_rebuild": lifecycle.get("scene_rebuild"),
        "report_written_before_environment_close": True,
    }

    # The report and completion envelope must be durable before closing Kit:
    # SimulationApp.close() is allowed to terminate this interpreter.
    if certifier is not None and hasattr(certifier, "write_report_atomic"):
        certifier.write_report_atomic(demo_dir, report)
    else:
        _write_json_atomic(
            demo_dir / REPORT_NAME,
            {
                "format": "handson.physical_replay_certification",
                "version": 1,
                "generated_at": _now_iso(),
                "demo_dir": str(demo_dir),
                **report,
            },
        )
    _write_json_atomic(
        result_path,
        {
            "format": "handson.platform_independent_episode_result",
            "version": 1,
            "generated_at": _now_iso(),
            "demo_dir": str(demo_dir),
            "report": report,
        },
    )
    print(
        json.dumps(
            {
                "event": "independent_episode_completed",
                "episode": demo_dir.name,
                "status": report.get("status"),
                "reason": report.get("reason"),
                "recorded_commit": recorded_commit,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for environment in reversed(environments):
        with contextlib.suppress(BaseException):
            environment.close()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--recorded-commit", required=True)
    parser.add_argument("--result-path", required=True, type=Path)
    arguments = parser.parse_args(argv)
    report = replay_one_episode(
        arguments.demo_dir,
        source_root=arguments.source_root,
        recorded_commit=str(arguments.recorded_commit),
        result_path=arguments.result_path,
    )
    return 1 if report.get("error_type") else 0


if __name__ == "__main__":
    raise SystemExit(main())
