#!/usr/bin/env python3
"""Background batch runner for Isaac deferred arm replay publishing."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPO_ROOT / "handson"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from handson.runtime.gpu import configure_isaac_gpu_environment
from handson.sim.isaaclab.runtime_display import (
    configure_isaac_offline_render_environment,
)
from handson.data.deferred_arm_replay_artifacts import (
    capture_deferred_arm_replay_artifacts,
    deferred_arm_replay_artifact_errors,
)
from handson.data.episode_publication_lock import (
    INHERITED_EPISODE_PUBLICATION_LOCK_FD_ENV,
    episode_publication_lock,
)
from handson.data.simplified_post_ik import (
    declares_simplified_post_ik,
    simplified_arm_plan_artifact_errors,
    simplified_arm_plan_errors,
)


STATUS_FILE = "deferred_arm_replay_status.json"
DEFERRED_COLLECTION_ONLY_ENV_KEYS = (
    "HANDSON_ISAACLAB_DIRECT_HAND_CONTROL",
    "HANDSON_ISAACLAB_DEFER_CAMERA_RENDER",
    "HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY",
    "HANDSON_ISAACLAB_DIRECT_VIEWPORT_DIVISOR",
    "HANDSON_ISAACLAB_ARM_TARGET_ALPHA",
    "HANDSON_ISAACLAB_ARM_MAX_SITE_STEP",
    "HANDSON_ISAACLAB_ARM_TARGET_JUMP_DELTA",
    "HANDSON_ISAACLAB_ARM_TARGET_POS_DEADBAND",
    "HANDSON_ISAACLAB_ARM_TARGET_ROT_JUMP_DEG",
    "HANDSON_ISAACLAB_ARM_TARGET_ROT_DEADBAND_DEG",
    "HANDSON_ISAACLAB_ARM_TARGET_JUMP_LIMIT",
    "HANDSON_ISAACLAB_ARM_MAX_JOINT_STEP",
    "HANDSON_ISAACSIM_VIEWPORT_WIDTH",
    "HANDSON_ISAACSIM_VIEWPORT_HEIGHT",
    "HANDSON_ISAACSIM_DLSS_EXEC_MODE",
)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _status_path(demo_dir: Path) -> Path:
    return demo_dir / STATUS_FILE


def _update_status(demo_dir: Path, patch: dict[str, object]) -> None:
    status = _read_json(_status_path(demo_dir))
    status.update(patch)
    status["updated_at"] = _now_iso()
    _write_json_atomic(_status_path(demo_dir), status)


def _clear_artifact_status(demo_dir: Path) -> None:
    status = _read_json(_status_path(demo_dir))
    for key in (
        "artifact_proof",
        "artifact_quality_profile",
        "artifact_quality_evidence_sha256",
    ):
        status.pop(key, None)
    status["artifact_validation_passed"] = False
    status["updated_at"] = _now_iso()
    _write_json_atomic(_status_path(demo_dir), status)


def _render_command(demo_dir: Path, exp_name: str, video_fps: int) -> list[str]:
    return [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "isaac_render_arm_replay.py"),
        f"--arm_replay_demo_dir={demo_dir}",
        f"--exp_name={exp_name}",
        f"--arm_replay_fps={max(1, int(video_fps))}",
    ]


def _simplified_plan_command(demo_dir: Path) -> list[str]:
    configured_python = str(
        os.environ.get("HANDSON_E01_ARM_PLAN_PYTHON")
        or os.environ.get("HANDSON_PYTHON")
        or ""
    ).strip()
    planner_python = sys.executable
    if configured_python:
        candidate = Path(configured_python).expanduser().resolve()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise RuntimeError(
                f"Configured offline IK Python is not executable: {candidate}"
            )
        planner_python = str(candidate)
    return [
        planner_python,
        str(_REPO_ROOT / "scripts" / "simplified_post_ik_plan.py"),
        str(demo_dir),
        "--overwrite",
    ]


def _uses_simplified_post_ik(demo_dir: Path) -> bool:
    manifest = _read_json(demo_dir / "replay_manifest.json")
    recording = manifest.get("recording")
    recording = recording if isinstance(recording, dict) else {}
    collection = _read_json(demo_dir / "collection_meta.json")
    return bool(
        declares_simplified_post_ik(recording)
        or declares_simplified_post_ik(collection)
    )


def _render_env(
    *,
    inherited_episode_lock_fd: int | None = None,
) -> dict[str, str]:
    """Build the renderer env with an inherited held-lock capability.

    ``_run_one`` passes this exact descriptor through ``Popen(pass_fds=...)``
    while retaining the lock for the complete child lifetime.  A PID or a
    generic "lock busy" observation is not an ownership proof.
    """

    env = os.environ.copy()
    env.pop("HANDSON_ISAAC_DEFERRED_ARM_WORKER", None)
    env.pop(INHERITED_EPISODE_PUBLICATION_LOCK_FD_ENV, None)
    if inherited_episode_lock_fd is not None:
        if (
            type(inherited_episode_lock_fd) is not int
            or inherited_episode_lock_fd < 0
        ):
            raise ValueError("inherited episode publication lock fd is invalid")
        env[INHERITED_EPISODE_PUBLICATION_LOCK_FD_ENV] = str(
            inherited_episode_lock_fd
        )
    for key in DEFERRED_COLLECTION_ONLY_ENV_KEYS:
        env.pop(key, None)
    env["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] = "0"
    env["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
    env["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"] = "0"
    pythonpath = env.get("PYTHONPATH", "")
    entries = [str(_PACKAGE_ROOT), str(_REPO_ROOT)]
    entries.extend(part for part in pythonpath.split(os.pathsep) if part)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(entries))
    env = configure_isaac_offline_render_environment(env)
    return configure_isaac_gpu_environment(env, render_mode="offscreen")


def _stop_render_process(process: subprocess.Popen) -> int | None:
    for sig, timeout in (
        (signal.SIGINT, 3.0),
        (signal.SIGTERM, 2.0),
        (signal.SIGKILL, 1.0),
    ):
        if process.poll() is not None:
            return process.returncode
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            return process.poll()
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
    return process.poll()


def _run_one(
    demo_dir: Path,
    *,
    exp_name: str,
    video_fps: int,
    log_path: Path | None,
    force: bool,
) -> int:
    with episode_publication_lock(demo_dir, blocking=True) as publication_lock:
        return _run_one_locked(
            demo_dir,
            exp_name=exp_name,
            video_fps=video_fps,
            log_path=log_path,
            force=force,
            publication_lock_fd=publication_lock.fileno(),
        )


def _run_one_locked(
    demo_dir: Path,
    *,
    exp_name: str,
    video_fps: int,
    log_path: Path | None,
    force: bool,
    publication_lock_fd: int,
) -> int:
    demo_path = demo_dir.expanduser().resolve()
    current = _read_json(_status_path(demo_path))
    uses_simplified_post_ik = _uses_simplified_post_ik(demo_path)
    plan_artifact_errors = (
        [
            *simplified_arm_plan_errors(demo_path / "replay.zarr"),
            *simplified_arm_plan_artifact_errors(
                demo_path,
                current.get("simplified_arm_plan_artifact"),
            ),
        ]
        if uses_simplified_post_ik
        else []
    )
    if (
        not force
        and str(current.get("status") or "").lower() == "complete"
        and not deferred_arm_replay_artifact_errors(
            demo_path,
            current.get("artifact_proof"),
        )
        and not plan_artifact_errors
    ):
        print(f"[Isaac deferred arm replay worker] skip complete: {demo_path}", flush=True)
        return 0

    commands = []
    if uses_simplified_post_ik:
        commands.append(_simplified_plan_command(demo_path))
    commands.append(_render_command(demo_path, exp_name, video_fps))
    _clear_artifact_status(demo_path)
    _update_status(
        demo_path,
        {
            "status": "running",
            "started_at": _now_iso(),
            "worker_pid": os.getpid(),
            "exp_name": exp_name,
            "video_fps": max(1, int(video_fps)),
            "command": commands[-1],
            "commands": commands,
            "simplified_post_ik_planning_required": len(commands) == 2,
            "log_path": str(log_path) if log_path is not None else "",
        },
    )
    print(
        "[Isaac deferred arm replay worker] starting: "
        + " then ".join(" ".join(command) for command in commands),
        flush=True,
    )
    process: subprocess.Popen | None = None
    try:
        returncode = 0
        for command_index, command in enumerate(commands):
            phase = (
                "planning_simplified_arms"
                if len(commands) == 2 and command_index == 0
                else "rendering_canonical_cameras"
            )
            _update_status(
                demo_path,
                {
                    "status": "running",
                    "publication_phase": phase,
                    "active_command": command,
                },
            )
            process = subprocess.Popen(
                command,
                # _run_one holds this same open-file-description lock until
                # all plan/render children exit.
                env=_render_env(
                    inherited_episode_lock_fd=publication_lock_fd,
                ),
                pass_fds=(publication_lock_fd,),
                stderr=subprocess.STDOUT,
            )
            returncode = int(process.wait())
            if returncode != 0:
                break
        artifact_proof: dict[str, object] | None = None
        artifact_error = ""
        if returncode == 0:
            try:
                if uses_simplified_post_ik:
                    rendered_status = _read_json(_status_path(demo_path))
                    plan_errors = [
                        *simplified_arm_plan_errors(
                            demo_path / "replay.zarr"
                        ),
                        *simplified_arm_plan_artifact_errors(
                            demo_path,
                            rendered_status.get(
                                "simplified_arm_plan_artifact"
                            ),
                        ),
                    ]
                    if plan_errors:
                        raise ValueError("; ".join(plan_errors))
                rendered_status = _read_json(_status_path(demo_path))
                rendered_proof = rendered_status.get("artifact_proof")
                if isinstance(rendered_proof, Mapping) and not (
                    deferred_arm_replay_artifact_errors(
                        demo_path,
                        rendered_proof,
                    )
                ):
                    # The renderer already performed the expensive BC021 full
                    # decode while holding the same episode lock.  Reuse that
                    # exact proof instead of decoding all three videos twice.
                    artifact_proof = dict(rendered_proof)
                else:
                    artifact_proof = capture_deferred_arm_replay_artifacts(
                        demo_path,
                        task_name=exp_name,
                    )
            except (OSError, ValueError) as exc:
                artifact_error = f"{type(exc).__name__}: {exc}"
        status = (
            "complete"
            if returncode == 0 and artifact_proof is not None
            else "failed"
        )
        patch: dict[str, object] = {
            "status": status,
            "completed_at": _now_iso(),
            "returncode": returncode,
            "artifact_validation_passed": artifact_proof is not None,
        }
        if artifact_proof is not None:
            patch["artifact_proof"] = artifact_proof
            quality = artifact_proof.get("quality")
            if isinstance(quality, Mapping):
                patch["artifact_quality_profile"] = str(
                    quality.get("profile") or ""
                )
                patch["artifact_quality_evidence_sha256"] = str(
                    quality.get("evidence_sha256") or ""
                )
        if artifact_error:
            patch["error"] = artifact_error
        _update_status(
            demo_path,
            patch,
        )
        print(
            f"[Isaac deferred arm replay worker] {status}: {demo_path} rc={returncode}",
            flush=True,
        )
        return returncode if status == "complete" else 1
    except KeyboardInterrupt:
        returncode = (
            _stop_render_process(process)
            if process is not None
            else -int(signal.SIGINT)
        )
        _update_status(
            demo_path,
            {
                "status": "stopped",
                "completed_at": _now_iso(),
                "returncode": int(returncode or -int(signal.SIGINT)),
                "error": "stopped by operator or collection handoff",
                "operator_stopped": True,
                "attempt_consumed": False,
            },
        )
        print(
            f"[Isaac deferred arm replay worker] stopped: {demo_path} rc={returncode}",
            flush=True,
        )
        raise
    except Exception as exc:  # noqa: BLE001
        _update_status(
            demo_path,
            {
                "status": "failed",
                "completed_at": _now_iso(),
                "returncode": None,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        print(
            "[Isaac deferred arm replay worker] failed with exception:\n"
            + traceback.format_exc(),
            flush=True,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo_dirs", nargs="+", help="Saved demo directories to render.")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--log_path", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    log_path = Path(args.log_path).expanduser().resolve() if args.log_path else None
    print(
        "[Isaac deferred arm replay worker] batch start: "
        f"{len(args.demo_dirs)} demo(s), exp_name={args.exp_name}, log={log_path or ''}",
        flush=True,
    )
    failures = 0
    demo_paths = [Path(demo).expanduser().resolve() for demo in args.demo_dirs]
    try:
        for demo_path in demo_paths:
            if _run_one(
                demo_path,
                exp_name=args.exp_name,
                video_fps=args.video_fps,
                log_path=log_path,
                force=bool(args.force),
            ) != 0:
                failures += 1
    except KeyboardInterrupt:
        stopped_at = _now_iso()
        for demo_path in demo_paths:
            status = _read_json(_status_path(demo_path))
            if (
                str(status.get("status") or "").strip().lower()
                not in {"queued", "running", "stopped"}
                or status.get("worker_pid") not in {None, os.getpid()}
            ):
                continue
            _update_status(
                demo_path,
                {
                    "status": "stopped",
                    "completed_at": stopped_at,
                    "error": "stopped by operator or collection handoff",
                    "operator_stopped": True,
                    "attempt_consumed": False,
                },
            )
        raise
    print(
        f"[Isaac deferred arm replay worker] batch done: failures={failures}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
