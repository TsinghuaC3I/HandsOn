#!/usr/bin/env python3
"""Stop a detached evaluation group if any unfinished task stops advancing."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--stall-timeout-seconds", type=int, default=1200)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--log", type=Path, required=True)
    return parser.parse_args()


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def read_progress(run_root: Path, task: str) -> tuple[int, int]:
    path = run_root / task / "eval" / "summary.json"
    if not path.exists():
        return 0, 100
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return int(value.get("completed", 0)), int(value.get("requested", 100))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0, 100


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"time": datetime.now().astimezone().isoformat(), **event}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if os.getpgid(args.launcher_pid) != args.launcher_pid:
        raise RuntimeError(
            "launcher must be a process-group leader (start it with setsid)"
        )
    now = time.monotonic()
    progress = {
        task: read_progress(args.run_root, task)[0] for task in args.tasks
    }
    changed_at = {task: now for task in args.tasks}
    append_event(
        args.log,
        {"event": "watch_started", "progress": progress, "launcher_pid": args.launcher_pid},
    )
    while process_exists(args.launcher_pid):
        time.sleep(args.poll_seconds)
        now = time.monotonic()
        for task in args.tasks:
            completed, requested = read_progress(args.run_root, task)
            if completed != progress[task]:
                progress[task] = completed
                changed_at[task] = now
                append_event(
                    args.log,
                    {"event": "progress", "task": task, "completed": completed, "requested": requested},
                )
            if completed < requested and now - changed_at[task] >= args.stall_timeout_seconds:
                append_event(
                    args.log,
                    {
                        "event": "stalled",
                        "task": task,
                        "completed": completed,
                        "requested": requested,
                        "stall_seconds": int(now - changed_at[task]),
                        "action": "terminate_launcher_process_group",
                    },
                )
                os.killpg(args.launcher_pid, signal.SIGTERM)
                time.sleep(30)
                if process_exists(args.launcher_pid):
                    os.killpg(args.launcher_pid, signal.SIGKILL)
                raise SystemExit(2)
    append_event(args.log, {"event": "launcher_exited", "progress": progress})


if __name__ == "__main__":
    main()
