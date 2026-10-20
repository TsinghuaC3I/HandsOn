#!/usr/bin/env python3
"""Run at least three independent B03 full-history Isaac processes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


_PACKAGE_ROOT = Path(__file__).resolve().parents[5]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from handson.sim.isaaclab.articulated_replay.full_history_trials.reporting import (
    build_aggregate_report,
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _load_worker_report(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "worker report was not created"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"worker report could not be decoded: {type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return None, "worker report root is not a JSON object"
    return value, None


def _exit_description(exit_code: int) -> dict[str, Any]:
    if exit_code == 0:
        return {"kind": "zero_exit", "signal": None}
    if exit_code < 0:
        return {"kind": "signal", "signal": -exit_code}
    return {"kind": "nonzero_exit", "signal": None}


def _trial_command(
    args: argparse.Namespace,
    worker_report_path: Path,
) -> list[str]:
    runner = args.runner.expanduser().resolve()
    command = [
        str(args.isaac_python.expanduser().resolve()),
        str(runner),
        *(str(path.expanduser().resolve()) for path in args.demo_dirs),
        "--report",
        str(worker_report_path),
        "--progress-every",
        str(args.progress_every),
        "--divergence-tolerance",
        repr(float(args.divergence_tolerance)),
    ]
    if args.no_wall_clock_throttle:
        command.append("--no-wall-clock-throttle")
    else:
        command.append("--no-no-wall-clock-throttle")
    return command


def _run_trial(args: argparse.Namespace, trial_index: int, output_dir: Path) -> dict[str, Any]:
    trial_dir = output_dir / f"trial_{trial_index:03d}"
    trial_dir.mkdir(parents=False, exist_ok=False)
    log_path = trial_dir / "isaac.log"
    worker_report_path = trial_dir / "worker_report.json"
    process_report_path = trial_dir / "process_report.json"
    command = _trial_command(args, worker_report_path)
    started_at = _now()
    started = time.monotonic()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    pid: int | None = None
    exit_code: int | None = None
    launch_error: str | None = None
    with log_path.open("wb") as log_stream:
        try:
            process = subprocess.Popen(
                command,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                env=environment,
                cwd=str(_PACKAGE_ROOT),
            )
            pid = process.pid
            exit_code = process.wait()
        except OSError as exc:
            launch_error = f"{type(exc).__name__}: {exc}"
            log_stream.write((launch_error + "\n").encode("utf-8", errors="replace"))
    worker_report, report_error = _load_worker_report(worker_report_path)
    worker_mapping = worker_report if isinstance(worker_report, Mapping) else {}
    execution = worker_mapping.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    exit_details = (
        _exit_description(exit_code)
        if exit_code is not None
        else {"kind": "launch_error", "signal": None}
    )
    trial: dict[str, Any] = {
        "trial_index": trial_index,
        "fresh_process": True,
        "pid": pid,
        "command": command,
        "started_at": started_at,
        "finished_at": _now(),
        "wall_seconds": time.monotonic() - started,
        "exit_code": exit_code,
        **exit_details,
        "launch_error": launch_error,
        "log_path": str(log_path),
        "worker_report_path": str(worker_report_path),
        "process_report_path": str(process_report_path),
        "worker_report": worker_report,
        "worker_report_error": report_error,
        "runner_completed": bool(
            worker_mapping.get("completed") is True
        ),
        "all_demos_executed": bool(
            execution.get("all_demos_executed") is True
            or worker_mapping.get("all_demo_trajectories_compared") is True
        ),
        "strict_full_trajectory_pass": bool(
            exit_code == 0
            and worker_mapping.get("completed") is True
            and (
                worker_mapping.get("trajectory_strict_pass") is True
                or worker_mapping.get("strict_full_trajectory_pass") is True
            )
        ),
        "runner_strict_pass": worker_mapping.get("strict_pass") is True,
        "exact_history_certificate_valid": (
            worker_mapping.get("exact_history_certificate_valid") is True
        ),
        "hard_crash_or_early_exit": bool(
            isinstance(exit_code, int)
            and exit_code != 0
            and worker_report is None
        ),
    }
    _write_json_atomic(process_report_path, trial)
    return trial


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo_dirs", type=Path, nargs=3, metavar="DEMO_DIR")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "history_rebuild_b03.py",
        help="Full-chain per-process runner (defaults to history_rebuild_b03.py).",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--divergence-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--no-wall-clock-throttle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.trials < 3:
        raise ValueError("--trials must be at least 3")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    if not math.isfinite(args.divergence_tolerance) or args.divergence_tolerance < 0:
        raise ValueError("--divergence-tolerance must be finite and non-negative")
    if not args.isaac_python.expanduser().is_file():
        raise FileNotFoundError(args.isaac_python)
    if not args.runner.expanduser().is_file():
        raise FileNotFoundError(args.runner)
    for path in args.demo_dirs:
        if not path.expanduser().is_dir():
            raise FileNotFoundError(path)

    output_dir = args.out_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    aggregate_path = output_dir / "aggregate.json"
    demo_dirs = [str(path.expanduser().resolve()) for path in args.demo_dirs]
    trials: list[dict[str, Any]] = []
    _write_json_atomic(
        aggregate_path,
        build_aggregate_report(
            demo_dirs=demo_dirs,
            requested_trial_count=args.trials,
            trials=trials,
            completed=False,
        ),
    )
    for trial_index in range(1, args.trials + 1):
        print(
            f"[multi trial] starting fresh process {trial_index}/{args.trials}",
            flush=True,
        )
        trial = _run_trial(args, trial_index, output_dir)
        trials.append(trial)
        _write_json_atomic(
            aggregate_path,
            build_aggregate_report(
                demo_dirs=demo_dirs,
                requested_trial_count=args.trials,
                trials=trials,
                completed=False,
            ),
        )
        print(
            f"[multi trial] trial {trial_index} exit={trial['exit_code']}; "
            f"log={trial['log_path']}",
            flush=True,
        )

    aggregate = build_aggregate_report(
        demo_dirs=demo_dirs,
        requested_trial_count=args.trials,
        trials=trials,
        completed=True,
    )
    _write_json_atomic(aggregate_path, aggregate)
    print(f"[multi trial] aggregate: {aggregate_path}", flush=True)
    # Nonzero worker exits are data in the aggregate, not an orchestrator
    # failure: later independent trials must still run after an Isaac crash.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
