#!/usr/bin/env python3
"""Exercise the YCB09 collection HTTP surface without changing Git branches.

This is an isolated-worktree acceptance helper.  It uses the production
``CollectionState`` and ``CollectionRequestHandler`` implementations, but
replaces the repository-pin transaction in memory so the smoke run cannot
switch the worktree from its task branch to ``release/collection``.

The helper starts a real deferred Isaac collection process, waits for the
recorder worker to report that it is running, and then stops it through the
same HTTP API used by the collection workbench.  It does not synthesize a
success, write an episode, or alter the task's success semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE_ROOT = ROOT / "handson"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from collection_ui import server as collection_server  # noqa: E402


TASK_NAME = "t29_place_wine_glass"


class _TerminationRequested(RuntimeError):
    """Turn an outer acceptance TERM into orderly collection cleanup."""


def _raise_termination_requested(signum: int, _frame: object) -> None:
    raise _TerminationRequested(f"received signal {signum}")


def _request_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError(f"{path} returned a non-object JSON response")
    return result


def _task_summary(tasks_payload: dict[str, Any]) -> dict[str, Any]:
    items = tasks_payload.get("items")
    if not isinstance(items, list):
        raise RuntimeError("/api/tasks did not return an items list")
    matches = [item for item in items if item.get("name") == TASK_NAME]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {TASK_NAME} task entry, found {len(matches)}"
        )
    task = matches[0]
    if task.get("sim_backends") != ["isaacsim"]:
        raise RuntimeError(f"unexpected YCB09 backends: {task.get('sim_backends')}")
    if task.get("default_arm_render_mode") != "deferred":
        raise RuntimeError("YCB09 is not exposed with deferred arm rendering")
    steps = task.get("steps")
    if not isinstance(steps, list) or not any("25" in str(step) for step in steps):
        raise RuntimeError("YCB09 UI instructions omit the 25 millimeter target")
    return {
        "name": task.get("name"),
        "label": task.get("label"),
        "prompt": task.get("prompt"),
        "sim_backends": task.get("sim_backends"),
        "arm_render_modes": task.get("arm_render_modes"),
        "default_arm_render_mode": task.get("default_arm_render_mode"),
        "steps": steps,
    }


def _collection_summary(payload: dict[str, Any]) -> dict[str, Any]:
    collection = payload.get("collection")
    if not isinstance(collection, dict):
        return {}
    session = collection.get("session")
    if not isinstance(session, dict):
        session = {}
    return {
        "running": bool(collection.get("running")),
        "paused": bool(collection.get("paused")),
        "pid": collection.get("pid"),
        "returncode": collection.get("returncode"),
        "phase": collection.get("phase"),
        "detail": collection.get("detail"),
        "session_id": session.get("session_id"),
        "task_name": session.get("task_name"),
        "sim_backend": session.get("sim_backend"),
        "isaac_arm_render_mode": session.get("isaac_arm_render_mode"),
        "command": session.get("command"),
        "log_path": collection.get("log_path"),
        "runtime_status_path": session.get("runtime_status_path"),
    }


def _wait_for_running(
    base_url: str,
    timeout_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout_s
    history: list[dict[str, Any]] = []
    last_phase = ""
    while time.monotonic() < deadline:
        status = _request_json(base_url, "/api/status", timeout=30.0)
        summary = _collection_summary(status)
        phase = str(summary.get("phase") or "")
        if phase != last_phase:
            history.append({"at_s": time.monotonic(), **summary})
            print(
                f"[YCB09 UI smoke] phase={phase or 'unknown'} "
                f"running={summary.get('running')} detail={summary.get('detail')}",
                flush=True,
            )
            last_phase = phase
        if phase == "running" and summary.get("running"):
            return status, history
        if not summary.get("running"):
            raise RuntimeError(
                "collection process exited before running: "
                f"phase={phase!r} returncode={summary.get('returncode')} "
                f"detail={summary.get('detail')!r}"
            )
        time.sleep(2.0)
    raise TimeoutError(f"YCB09 collection did not reach running in {timeout_s}s")


def _wait_for_stopped(base_url: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last = {}
    while time.monotonic() < deadline:
        last = _request_json(base_url, "/api/status", timeout=30.0)
        collection = _collection_summary(last)
        if not collection.get("running") and collection.get("phase") != "stopping":
            return last
        time.sleep(1.0)
    raise TimeoutError(
        "YCB09 collection did not stop cleanly: "
        f"{_collection_summary(last)}"
    )


def _collection_process_camera_environment(pid: object) -> dict[str, str]:
    try:
        process_id = int(pid)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"collection status has no valid pid: {pid!r}") from exc
    path = Path(f"/proc/{process_id}/environ")
    values = {}
    for item in path.read_bytes().split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        decoded_key = key.decode("utf-8", errors="strict")
        if decoded_key in {
            "HANDSON_ISAACLAB_DEFER_CAMERA_RENDER",
            "HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY",
        }:
            values[decoded_key] = value.decode("utf-8", errors="strict")
    expected = {
        "HANDSON_ISAACLAB_DEFER_CAMERA_RENDER": "1",
        "HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY": "0",
    }
    if values != expected:
        raise RuntimeError(
            f"formal collection camera environment mismatch: {values!r}"
        )
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--stop-timeout", type=float, default=90.0)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    demos_dir = output_dir / "demos"
    runtime_dir = output_dir / "runtime"
    demos_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # All settings are local to this acceptance process and its children.
    os.environ["HANDSON_COLLECTION_SIM_BACKEND"] = "isaacsim"
    os.environ["HANDSON_COLLECTION_ARM_TYPE"] = "rokae"
    os.environ["HANDSON_COLLECTION_HAND_TYPE"] = "wuji1"
    os.environ["HANDSON_COLLECTION_TELEOP_STACK"] = "acceptance_no_device"
    os.environ["HANDSON_COLLECTION_REQUIRE_TELEOP_READY"] = "0"
    os.environ["HANDSON_COLLECTION_CLEAN_STALE_ON_START"] = "0"
    os.environ["HANDSON_RELEASE_AUTO_UPDATE"] = "0"
    os.environ["HANDSON_ISAACSIM_CACHE_WARMUP"] = "0"
    os.environ["HANDSON_AUTO_START_TELEOP"] = "0"
    os.environ["HANDSON_POSTPROCESS_MANUAL_ONLY"] = "1"
    os.environ["HANDSON_COLLECTION_RUNTIME_DIR"] = str(runtime_dir)
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "1"
    os.environ["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"] = "0"

    state = collection_server.CollectionState(
        ROOT,
        sys.executable,
        demos_dir,
        sys.executable,
        "",
    )
    previous_signal_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_signal_handlers:
        signal.signal(signum, _raise_termination_requested)

    # Validation-only: preserve the isolated task branch.  Every other start
    # preflight and the launched collection process remain production code.
    def repository_pin_acceptance_stub(
        action: str,
        *,
        fetch: bool = False,
        idle_only: bool = False,
    ) -> dict[str, Any]:
        del fetch, idle_only
        if action != "pin":
            raise RuntimeError(
                f"acceptance stub permits only the implicit pin, got {action!r}"
            )
        return {
            "success": True,
            "code": "acceptance_branch_preserved",
            "detail": "isolated task branch retained for UI smoke",
            "updated": False,
            "publish_prompt_required": False,
            "can_collect": True,
        }

    state.run_repository_action = repository_pin_acceptance_stub  # type: ignore[method-assign]
    server = collection_server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        collection_server.CollectionRequestHandler,
    )
    server.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    report: dict[str, Any] = {
        "format": "handson.ycb09_collection_ui_smoke",
        "version": 1,
        "task_name": TASK_NAME,
        "root": str(ROOT),
        "python": sys.executable,
        "base_url": base_url,
        "repository_pin": "validation-only in-memory bypass",
        "synthetic_success": False,
        "episode_written": False,
        "passed": False,
    }
    report_path = output_dir / "report.json"

    try:
        health = _request_json(base_url, "/api/health")
        if not health.get("success"):
            raise RuntimeError(f"health endpoint failed: {health}")
        task = _task_summary(_request_json(base_url, "/api/tasks"))
        report["health"] = {
            "success": health.get("success"),
            "service": health.get("service"),
            "root": health.get("root"),
            "server_code_sha256": health.get("server_code_sha256"),
        }
        report["task"] = task

        start = _request_json(
            base_url,
            "/api/collection/start",
            {
                "task_name": TASK_NAME,
                "sim_backend": "isaacsim",
                "isaac_arm_render_mode": "deferred",
                "arm_type": "rokae",
                "hand_type": "wuji1",
                "collector_id": "ycb09_acceptance",
                "successes_needed": 1,
                "randomize": True,
                "show_sim_cameras": False,
                "save_depth": False,
                "render_mode": "rgb_array",
                "out_dir": str(demos_dir),
                "repository_publish_decision": "skip",
            },
            timeout=60.0,
        )
        if not start.get("success"):
            raise RuntimeError(f"collection start endpoint failed: {start}")
        report["start"] = {
            "success": True,
            "pid": start.get("pid"),
            "log_path": start.get("log_path"),
            "session": _collection_summary({"collection": start}),
        }

        running_status, phase_history = _wait_for_running(
            base_url,
            args.startup_timeout,
        )
        running_summary = _collection_summary(running_status)
        if running_summary.get("task_name") != TASK_NAME:
            raise RuntimeError(f"wrong task launched: {running_summary}")
        if running_summary.get("sim_backend") != "isaacsim":
            raise RuntimeError(f"wrong simulator launched: {running_summary}")
        report["phase_history"] = phase_history
        report["running"] = running_summary
        report["camera_environment"] = _collection_process_camera_environment(
            running_summary.get("pid")
        )

        stop = _request_json(
            base_url,
            "/api/collection/stop",
            {},
            timeout=30.0,
        )
        if not stop.get("success") or not stop.get("stop_accepted"):
            raise RuntimeError(f"collection stop endpoint failed: {stop}")
        stopped_status = _wait_for_stopped(base_url, args.stop_timeout)
        report["stop"] = {
            "accepted": True,
            "final": _collection_summary(stopped_status),
        }
        report["passed"] = True
        print("[YCB09 UI smoke] PASS", flush=True)
        return 0
    except BaseException as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[YCB09 UI smoke] FAIL: {report['error']}", flush=True)
        return 1
    finally:
        process = state.collection_process
        if process is not None and process.poll() is None:
            try:
                state.stop_collection()
                deadline = time.monotonic() + args.stop_timeout
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.5)
            except BaseException as cleanup_exc:  # noqa: BLE001
                report["cleanup_error"] = (
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
        for signum, handler in previous_signal_handlers.items():
            signal.signal(signum, handler)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[YCB09 UI smoke] report={report_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
