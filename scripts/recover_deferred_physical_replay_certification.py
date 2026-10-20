#!/usr/bin/env python3
"""Recover orphaned deferred PhysX certification without retry spinning."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "handson"
for entry in (str(PACKAGE_ROOT), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from scripts import record_demos_zarr as recorder  # noqa: E402


def _is_physx_v3_candidate(demo_dir: Path) -> bool:
    return bool(
        demo_dir.is_dir()
        and demo_dir.name.endswith(recorder.PHYSX_V3_DIRECTORY_SUFFIX)
        and (demo_dir / recorder.PHYSX_V3_MARKER_FILE).is_file()
        and (demo_dir / "replay.zarr").is_dir()
    )


def _discover_tasks(output_root: Path) -> list[str]:
    pending_parent = recorder._physical_replay_pending_parent(output_root)
    if not pending_parent.is_dir():
        return []
    return sorted(
        {
            demo_dir.name.split("_demo_", 1)[0]
            for demo_dir in pending_parent.iterdir()
            if _is_physx_v3_candidate(demo_dir)
            and "_demo_" in demo_dir.name
        }
    )


def recover(
    output_root: str | Path,
    *,
    tasks: list[str] | None = None,
    certification_timeout_seconds: int = 1800,
    allow_review_required_rerun: bool = False,
    manual_review_run_id: str | None = None,
) -> dict[str, object]:
    root = Path(output_root).expanduser().resolve()
    selected_tasks = sorted(set(tasks or _discover_tasks(root)))
    if not allow_review_required_rerun:
        manual_review_run_id = None
    if allow_review_required_rerun and not manual_review_run_id:
        manual_review_run_id = f"manual-{time.time_ns()}"
    task_results: list[dict[str, object]] = []
    for task_name in selected_tasks:
        summary: dict[str, object] = {}
        # Keep stdout as one machine-readable JSON document.  Operational
        # worker launch lines remain visible on stderr for humans/systemd.
        with contextlib.redirect_stdout(sys.stderr):
            recorder._ensure_deferred_physical_replay_certification(
                root,
                exp_name=task_name,
                certification_runs=(
                    recorder.DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS
                ),
                certification_timeout_seconds=(certification_timeout_seconds),
                source_recorder_pid=0,
                skip_live_source_recorders=True,
                recovery_summary=summary,
                allow_review_required_rerun=allow_review_required_rerun,
                manual_review_run_id=manual_review_run_id,
                physx_v3_only=True,
            )
        task_results.append(summary)
        # Do not start one Isaac/PhysX worker per task in the same sweep.  The
        # next poll will pick up the next task after this worker exits.
        if int(summary.get("launched_count") or 0) > 0:
            break
    return {
        "format": "handson.deferred_physical_replay_recovery",
        "version": 1,
        "status": "complete",
        "output_root": str(root),
        "task_count": len(selected_tasks),
        "launched_count": sum(
            int(result.get("launched_count", 0)) for result in task_results
        ),
        "live_worker_skipped_count": sum(
            int(result.get("live_worker_skipped_count", 0)) for result in task_results
        ),
        "live_source_skipped_count": sum(
            int(result.get("live_source_skipped_count", 0)) for result in task_results
        ),
        "recovery_exhausted_count": sum(
            int(result.get("recovery_exhausted_count", 0)) for result in task_results
        ),
        "legacy_review_migrated_count": sum(
            int(result.get("legacy_review_migrated_count", 0))
            for result in task_results
        ),
        "tasks": task_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Task name to recover; repeat for multiple tasks (default: all).",
    )
    parser.add_argument(
        "--certification-timeout-seconds",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--allow-review-required-rerun",
        action="store_true",
        help=(
            "Allow a manual rerun of review_required items even when "
            "auto-recovery has reached the retry limit."
        ),
    )
    parser.add_argument(
        "--manual-review-run-id",
        default=None,
        help="Stable id for one UI-triggered manual PhysX review pass.",
    )
    args = parser.parse_args(argv)
    if args.certification_timeout_seconds <= 0:
        parser.error("--certification-timeout-seconds must be positive")
    try:
        result = recover(
            args.out_root,
            tasks=args.task,
            certification_timeout_seconds=(args.certification_timeout_seconds),
            allow_review_required_rerun=args.allow_review_required_rerun,
            manual_review_run_id=args.manual_review_run_id,
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "format": "handson.deferred_physical_replay_recovery",
            "version": 1,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
