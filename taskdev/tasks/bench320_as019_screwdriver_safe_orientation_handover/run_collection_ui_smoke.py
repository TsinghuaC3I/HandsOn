#!/usr/bin/env python3
"""Exercise the production collection-UI start path for AS019."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


TASK_NAME = "t22_pass_screwdriver"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ready-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--successful-episodes",
        type=int,
        default=0,
        help=(
            "After the UI reaches ready, wait for this many automatically "
            "successful recorder episodes; zero keeps the load/start smoke path."
        ),
    )
    parser.add_argument(
        "--completion-timeout-seconds", type=float, default=1800.0
    )
    return parser


def _write(path: Path, summary: dict[str, object]) -> None:
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _completed_demo_dirs(output_dir: Path) -> list[Path]:
    """Return recorder episodes only after replay publication is complete."""

    return sorted(
        path
        for path in (output_dir / "dataset").rglob(f"{TASK_NAME}_demo_*")
        if path.is_dir() and (path / "replay.zarr").is_dir()
    )


def main() -> int:
    args = _parser().parse_args()
    if args.successful_episodes < 0:
        raise ValueError("--successful-episodes must be non-negative")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "as019_collection_ui_smoke.json"
    root = Path(__file__).resolve().parents[3]
    isaac_python = os.environ.get(
        "HANDSON_ISAACSIM_PYTHON",
        "/mnt/pfs/luotianwei/envs/isaacsim-5.1-handson/bin/python",
    )
    os.environ.setdefault("HANDSON_COLLECTION_TELEOP_STACK", "disabled")
    os.environ.setdefault("HANDSON_COLLECTION_REQUIRE_TELEOP_READY", "0")
    os.environ.setdefault("HANDSON_BACKGROUND_V3_UPLOAD", "0")
    os.environ.setdefault("HANDSON_REVIEW_AUTO_UPLOAD", "0")
    os.environ.setdefault("HANDSON_POSTPROCESS_MANUAL_ONLY", "1")
    os.environ.setdefault(
        "HANDSON_COLLECTION_RUNTIME_DIR", str(output_dir / "runtime")
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "task": TASK_NAME,
        "checkpoint": os.environ.get("AS019_EXPECTED_CHECKPOINT", "unknown"),
        "root": str(root),
        "start_response": None,
        "ready": False,
        "stopped": False,
        "collection_target_reached": False,
        "successful_episodes_requested": args.successful_episodes,
    }
    state = None
    exit_code = 1
    try:
        from collection_ui.server import CollectionState

        state = CollectionState(
            root,
            sys.executable,
            output_dir / "dataset",
            isaac_python,
        )
        payload = {
            "task_name": TASK_NAME,
            "sim_backend": "isaacsim",
            "isaac_arm_render_mode": "deferred",
            "arm_type": "rokae",
            "hand_type": "wuji1",
            "collector_id": "as019_platform_smoke",
            "successes_needed": max(1, args.successful_episodes),
            "randomize": True,
            "show_sim_cameras": False,
            "render_mode": "rgb_array",
            "save_depth": False,
            "out_dir": str(output_dir / "dataset"),
        }
        start = state.start_collection(payload)
        summary["start_response"] = start
        if not bool(start.get("success")):
            raise RuntimeError(f"collection UI rejected AS019 start: {start}")
        session = dict(start.get("session") or {})
        command = [str(value) for value in session.get("command") or []]
        expected_arguments = {
            f"--exp_name={TASK_NAME}",
            "--sim_backend=isaacsim",
            "--arm_type=rokae",
            "--hand_type=wuji1",
            f"--successes_needed={max(1, args.successful_episodes)}",
        }
        summary["command_contract"] = {
            "record_demos_zarr": any(
                Path(value).name == "record_demos_zarr.py" for value in command
            ),
            "expected_arguments": sorted(expected_arguments),
            "missing_arguments": sorted(expected_arguments.difference(command)),
            "command": command,
        }
        runtime_status_path = Path(str(session.get("runtime_status_path") or ""))
        log_path = Path(str(session.get("log_path") or ""))
        deadline = time.monotonic() + max(1.0, args.ready_timeout_seconds)
        last_runtime: dict[str, object] = {}
        while time.monotonic() < deadline:
            process = state.collection_process
            if process is None or process.poll() is not None:
                break
            if runtime_status_path.is_file():
                try:
                    value = json.loads(runtime_status_path.read_text(encoding="utf-8"))
                    if isinstance(value, dict):
                        last_runtime = value
                except (OSError, json.JSONDecodeError):
                    pass
            log_text = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.is_file()
                else ""
            )
            if (
                str(last_runtime.get("phase") or "") == "running"
                and f"Environment reset complete for {TASK_NAME}." in log_text
            ):
                summary["ready"] = True
                break
            time.sleep(2.0)
        summary["last_runtime_status"] = last_runtime
        summary["log_path"] = str(log_path)
        summary["log_tail"] = (
            log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            if log_path.is_file()
            else ""
        )
        process = state.collection_process
        summary["returncode_before_stop"] = (
            process.poll() if process is not None else None
        )
        if not bool(summary["ready"]):
            raise RuntimeError("AS019 collection process did not reach ready state")

        if args.successful_episodes:
            completion_deadline = (
                time.monotonic() + max(1.0, args.completion_timeout_seconds)
            )
            while time.monotonic() < completion_deadline:
                completed = _completed_demo_dirs(output_dir)
                if len(completed) >= args.successful_episodes:
                    # A published replay.zarr and its sibling videos are the
                    # authoritative recorder boundary.  Stop through the same
                    # CollectionState path used by the UI once the requested
                    # count is durable; waiting for Kit's natural teardown can
                    # trip its heartbeat watchdog after all data is saved.
                    summary["collection_target_reached"] = True
                    process = state.collection_process
                    if process is not None and process.poll() is None:
                        summary["target_stop_response"] = (
                            state.stop_collection()
                        )
                    stop_deadline = time.monotonic() + 90.0
                    while time.monotonic() < stop_deadline:
                        process = state.collection_process
                        if process is None or process.poll() is not None:
                            summary["stopped"] = True
                            break
                        time.sleep(1.0)
                    break
                process = state.collection_process
                if process is None or process.poll() is not None:
                    summary["stopped"] = True
                    break
                if runtime_status_path.is_file():
                    try:
                        value = json.loads(
                            runtime_status_path.read_text(encoding="utf-8")
                        )
                        if isinstance(value, dict):
                            last_runtime = value
                    except (OSError, json.JSONDecodeError):
                        pass
                time.sleep(2.0)
            process = state.collection_process
            summary["returncode_after_collection"] = (
                process.poll() if process is not None else None
            )
            summary["last_runtime_status"] = last_runtime
            if not bool(summary["stopped"]):
                raise RuntimeError(
                    "AS019 collection process did not finish the requested episodes"
                )
        else:
            stop = state.stop_collection()
            summary["stop_response"] = stop
            stop_deadline = time.monotonic() + 90.0
            while time.monotonic() < stop_deadline:
                process = state.collection_process
                if process is None or process.poll() is not None:
                    summary["stopped"] = True
                    break
                time.sleep(1.0)

        demo_dirs = _completed_demo_dirs(output_dir)
        schema_errors: list[str] = []
        for demo_dir in demo_dirs:
            for relative in (
                "replay.zarr",
                "videos/main.mp4",
                "videos/wrist_left.mp4",
                "videos/wrist_right.mp4",
            ):
                artifact = demo_dir / relative
                if not artifact.exists():
                    schema_errors.append(f"{demo_dir.name}: missing {relative}")
                elif artifact.is_file() and artifact.stat().st_size <= 0:
                    schema_errors.append(f"{demo_dir.name}: empty {relative}")
        summary["collection_artifacts"] = {
            "episode_count": len(demo_dirs),
            "demo_dirs": [str(path) for path in demo_dirs],
            "schema_errors": schema_errors,
        }
        summary["log_tail"] = (
            log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            if log_path.is_file()
            else ""
        )
        process = state.collection_process
        returncode = process.poll() if process is not None else None
        requested_episodes_pass = bool(
            args.successful_episodes == 0
            or (
                summary["collection_target_reached"]
                and len(demo_dirs) >= args.successful_episodes
                and not schema_errors
            )
        )
        contract = dict(summary["command_contract"])
        summary["validation_pass"] = bool(
            summary["ready"]
            and summary["stopped"]
            and contract["record_demos_zarr"]
            and not contract["missing_arguments"]
            and requested_episodes_pass
        )
        exit_code = 0 if summary["validation_pass"] else 2
    except BaseException as exc:
        summary["exception"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        exit_code = 1
    finally:
        if state is not None:
            process = state.collection_process
            if process is not None and process.poll() is None:
                try:
                    state.stop_collection()
                    time.sleep(1.0)
                    process = state.collection_process
                    if process is not None and process.poll() is None:
                        state.stop_collection()
                except BaseException as exc:
                    summary["final_stop_error"] = repr(exc)
        _write(summary_path, summary)
        print(f"AS019_COLLECTION_UI_SMOKE={summary_path}", flush=True)
        print(json.dumps(summary, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
