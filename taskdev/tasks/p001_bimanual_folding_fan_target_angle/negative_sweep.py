"""Run the P0-01 live negative matrix sequentially.

The matrix complements the fail-closed semantic contract tests with physical
Isaac trajectories.  It deliberately uses only robot actions exposed by the
normal task runner; it never writes the fan root pose or joint state after
reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


CASES = (
    ("zero", 60, 201),
    ("zero", 90, 202),
    ("zero", 120, 203),
    ("right_pose", 60, 204),
    ("right_pose", 90, 205),
    ("right_pose", 120, 206),
    ("left_pose", 60, 207),
    ("left_pose", 90, 208),
    ("left_pose", 120, 209),
    ("hand_only", 60, 210),
    ("hand_only", 90, 211),
    ("hand_only", 120, 212),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_result(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _expected_contact_absence(mode: str, task_info: dict[str, Any]) -> bool:
    left_contact = bool(task_info.get("p001_left_end_contact", False))
    right_contact = bool(task_info.get("p001_right_end_contact", False))
    if mode == "right_pose":
        return not left_contact
    if mode == "left_pose":
        return not right_contact
    return not left_contact and not right_contact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if not args.launcher.is_file():
        raise FileNotFoundError(args.launcher)

    repo_root = Path(__file__).resolve().parents[3]
    implementation_files = {
        "contract": repo_root
        / "handson/handson/sim/isaaclab/p001_contract.py",
        "environment": repo_root
        / "handson/handson/sim/isaaclab/p001_collection_env.py",
        "runner": Path(__file__).with_name("live_smoke.py"),
    }
    result: dict[str, Any] = {
        "schema": "p001_live_negative_sweep_v1",
        "cases": [
            {"mode": mode, "target_deg": target, "seed": seed}
            for mode, target, seed in CASES
        ],
        "steps_per_trial": args.steps,
        "implementation_sha256": {
            name: _sha256(path) for name, path in implementation_files.items()
        },
        "records": [],
        "passed": False,
    }
    args.run_root.mkdir(parents=True, exist_ok=True)
    _write_result(args.output, result)

    environment = os.environ.copy()
    environment["P001_ARM_IK_BACKEND"] = "dls"
    environment["P001_FORCE_ARM_LIMITS"] = "0"
    if args.cache is not None:
        environment["P001_ISAACLAB_CACHE"] = str(args.cache)

    for mode, target, seed in CASES:
        run_dir = args.run_root / f"{mode}-target{target}-seed{seed}"
        completed = subprocess.run(
            [
                str(args.launcher),
                str(run_dir),
                mode,
                str(args.steps),
                str(target),
                str(seed),
            ],
            env=environment,
            check=False,
        )
        trial_result_path = run_dir / "output/result.json"
        if trial_result_path.is_file():
            trial = json.loads(trial_result_path.read_text(encoding="utf-8"))
            task_info = trial.get("task_info", {})
            record = {
                "mode": mode,
                "target_deg": target,
                "seed": seed,
                "run_dir": str(run_dir),
                "launcher_exit_code": completed.returncode,
                "result_present": True,
                "executed_steps": int(trial.get("executed_steps", -1)),
                "terminal": bool(trial.get("terminal", False)),
                "success": bool(trial.get("success", True)),
                "failure": bool(trial.get("failure", False)),
                "span_deg": float(task_info.get("p001_span_deg", -1.0)),
                "target_error_deg": float(
                    task_info.get("p001_target_error_deg", -1.0)
                ),
                "left_end_contact": bool(
                    task_info.get("p001_left_end_contact", False)
                ),
                "right_end_contact": bool(
                    task_info.get("p001_right_end_contact", False)
                ),
                "target_ok": bool(task_info.get("p001_target_ok", False)),
                "net_increase_ok": bool(
                    task_info.get("p001_net_increase_ok", False)
                ),
                "bilateral_expansion_ok": bool(
                    task_info.get("p001_bilateral_expansion_ok", False)
                ),
                "wrong_end_clear": bool(
                    task_info.get("p001_wrong_end_clear", False)
                ),
                "robot_table_clear": bool(
                    task_info.get("p001_robot_table_clear", False)
                ),
                "object_state_writes_after_reset": int(
                    task_info.get(
                        "p001_object_state_writes_after_action_zero", -1
                    )
                ),
            }
            record["passed"] = bool(
                completed.returncode == 0
                and record["terminal"]
                and record["executed_steps"] == args.steps
                and not record["success"]
                and not record["failure"]
                and record["object_state_writes_after_reset"] == 0
                and _expected_contact_absence(mode, task_info)
            )
        else:
            record = {
                "mode": mode,
                "target_deg": target,
                "seed": seed,
                "run_dir": str(run_dir),
                "launcher_exit_code": completed.returncode,
                "result_present": False,
                "passed": False,
            }
        result["records"].append(record)
        _write_result(args.output, result)
        print(
            json.dumps(
                {
                    key: record.get(key)
                    for key in (
                        "mode",
                        "target_deg",
                        "seed",
                        "passed",
                        "terminal",
                        "success",
                        "failure",
                        "executed_steps",
                        "span_deg",
                        "left_end_contact",
                        "right_end_contact",
                        "object_state_writes_after_reset",
                    )
                },
                sort_keys=True,
            ),
            flush=True,
        )

    records = result["records"]
    result.update(
        {
            "trial_count": len(records),
            "trial_pass_count": sum(bool(record["passed"]) for record in records),
            "false_positive_count": sum(
                bool(record.get("success", False)) for record in records
            ),
            "failure_latched_count": sum(
                bool(record.get("failure", False)) for record in records
            ),
            "timeout_like_count": sum(
                int(record.get("executed_steps", -1)) == args.steps
                and not bool(record.get("failure", False))
                for record in records
            ),
            "post_reset_object_state_write_count": sum(
                int(record.get("object_state_writes_after_reset", -1))
                for record in records
            ),
            "live_categories": [
                "single_hand",
                "insufficient_contact_caused_opening",
                "timeout",
            ],
        }
    )
    result["passed"] = bool(
        len(records) == len(CASES)
        and all(bool(record["passed"]) for record in records)
        and result["false_positive_count"] == 0
        and result["post_reset_object_state_write_count"] == 0
    )
    _write_result(args.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "trial_count",
                    "trial_pass_count",
                    "false_positive_count",
                    "failure_latched_count",
                    "timeout_like_count",
                    "post_reset_object_state_write_count",
                    "passed",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
