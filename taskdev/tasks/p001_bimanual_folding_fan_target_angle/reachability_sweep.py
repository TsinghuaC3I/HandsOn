"""Run the frozen P0-01 full-trajectory reachability matrix sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--targets", type=int, nargs="+", default=(60, 90, 120)
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=tuple(range(101, 111))
    )
    parser.add_argument("--steps", type=int, default=660)
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
        "controller_runner": Path(__file__).with_name("live_smoke.py"),
    }
    result: dict[str, Any] = {
        "schema": "p001_reachability_sweep_v1",
        "targets_deg": list(args.targets),
        "seeds": list(args.seeds),
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

    for target in args.targets:
        for seed in args.seeds:
            run_dir = args.run_root / f"target{target}-seed{seed}"
            completed = subprocess.run(
                [
                    str(args.launcher),
                    str(run_dir),
                    "bimanual_feedback_open",
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
                    "target_deg": target,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "launcher_exit_code": completed.returncode,
                    "result_present": True,
                    "success": bool(trial.get("success", False)),
                    "failure": bool(trial.get("failure", True)),
                    "executed_steps": int(trial.get("executed_steps", -1)),
                    "span_deg": float(task_info.get("p001_span_deg", -1.0)),
                    "target_error_deg": float(
                        task_info.get("p001_target_error_deg", -1.0)
                    ),
                    "balance_error_deg": float(
                        task_info.get("p001_balance_error_deg", -1.0)
                    ),
                    "bilateral_expansion_fraction": float(
                        task_info.get(
                            "p001_bilateral_expansion_fraction", -1.0
                        )
                    ),
                    "left_end_contact": bool(
                        task_info.get("p001_left_end_contact", False)
                    ),
                    "right_end_contact": bool(
                        task_info.get("p001_right_end_contact", False)
                    ),
                    "wrong_end_clear": bool(
                        task_info.get("p001_wrong_end_clear", False)
                    ),
                    "robot_table_clear": bool(
                        task_info.get("p001_robot_table_clear", False)
                    ),
                    "root_displacement_m": float(
                        task_info.get("p001_root_displacement_m", -1.0)
                    ),
                    "root_lift_m": float(
                        task_info.get("p001_root_lift_m", -1.0)
                    ),
                    "planarity_error_m": float(
                        task_info.get("p001_planarity_error_m", -1.0)
                    ),
                    "object_state_writes_after_reset": int(
                        task_info.get(
                            "p001_object_state_writes_after_action_zero", -1
                        )
                    ),
                }
                record["passed"] = bool(
                    completed.returncode == 0
                    and record["success"]
                    and not record["failure"]
                    and record["left_end_contact"]
                    and record["right_end_contact"]
                    and record["wrong_end_clear"]
                    and record["object_state_writes_after_reset"] == 0
                )
            else:
                record = {
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
                            "target_deg",
                            "seed",
                            "passed",
                            "success",
                            "failure",
                            "executed_steps",
                            "span_deg",
                            "target_error_deg",
                            "balance_error_deg",
                            "bilateral_expansion_fraction",
                            "wrong_end_clear",
                            "robot_table_clear",
                        )
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    expected_count = len(args.targets) * len(args.seeds)
    target_counts = {
        str(target): sum(
            bool(record["passed"]) and record["target_deg"] == target
            for record in result["records"]
        )
        for target in args.targets
    }
    result["target_pass_counts"] = target_counts
    result["trial_count"] = len(result["records"])
    result["passed"] = bool(
        len(result["records"]) == expected_count
        and all(record["passed"] for record in result["records"])
    )
    _write_result(args.output, result)
    print(
        json.dumps(
            {
                "trial_count": result["trial_count"],
                "target_pass_counts": target_counts,
                "passed": result["passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
