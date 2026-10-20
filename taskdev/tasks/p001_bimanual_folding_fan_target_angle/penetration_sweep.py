"""Run 15 positive P0-01 trajectories with per-frame penetration monitoring."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any


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
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--steps", type=int, default=660)
    parser.add_argument(
        "--targets", type=int, nargs="+", default=(60, 90, 120)
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(101, 102, 103, 104, 105)
    )
    args = parser.parse_args()
    if not args.launcher.is_file():
        raise FileNotFoundError(args.launcher)

    result: dict[str, Any] = {
        "schema": "p001_penetration_sweep_v1",
        "targets_deg": list(args.targets),
        "seeds": list(args.seeds),
        "steps_per_trial": args.steps,
        "threshold_m": 0.005,
        "records": [],
        "passed": False,
    }
    args.run_root.mkdir(parents=True, exist_ok=True)
    _write_result(args.output, result)
    base_environment = os.environ.copy()
    base_environment["P001_ARM_IK_BACKEND"] = "dls"
    base_environment["P001_FORCE_ARM_LIMITS"] = "0"
    if args.cache is not None:
        base_environment["P001_ISAACLAB_CACHE"] = str(args.cache)

    for target in args.targets:
        for seed in args.seeds:
            run_dir = args.run_root / f"target{target}-seed{seed}"
            certificate_path = run_dir / "output/penetration.json"
            environment = base_environment.copy()
            environment["P001_PENETRATION_OUTPUT"] = str(certificate_path)
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
            trial_path = run_dir / "output/result.json"
            if trial_path.is_file() and certificate_path.is_file():
                trial = json.loads(trial_path.read_text(encoding="utf-8"))
                certificate = json.loads(
                    certificate_path.read_text(encoding="utf-8")
                )
                task_info = trial.get("task_info", {})
                record = {
                    "target_deg": target,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "launcher_exit_code": completed.returncode,
                    "success": bool(trial.get("success", False)),
                    "failure": bool(trial.get("failure", True)),
                    "object_state_writes_after_reset": int(
                        task_info.get(
                            "p001_object_state_writes_after_action_zero", -1
                        )
                    ),
                    "control_steps_sampled": int(
                        certificate.get("control_steps_sampled", -1)
                    ),
                    "active_contact_samples": int(
                        certificate.get("active_contact_samples", -1)
                    ),
                    "maximum_penetration_m": float(
                        certificate.get("maximum_penetration_m", -1.0)
                    ),
                    "sample_error_count": len(
                        certificate.get("sample_errors", [])
                    ),
                    "certificate_passed": bool(
                        certificate.get("passed", False)
                    ),
                }
                record["passed"] = bool(
                    completed.returncode == 0
                    and record["success"]
                    and not record["failure"]
                    and record["object_state_writes_after_reset"] == 0
                    and record["certificate_passed"]
                )
            else:
                record = {
                    "target_deg": target,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "launcher_exit_code": completed.returncode,
                    "result_present": trial_path.is_file(),
                    "certificate_present": certificate_path.is_file(),
                    "passed": False,
                }
            result["records"].append(record)
            _write_result(args.output, result)
            print(json.dumps(record, sort_keys=True), flush=True)

    records = result["records"]
    result["trial_count"] = len(records)
    result["trial_pass_count"] = sum(
        bool(record["passed"]) for record in records
    )
    result["maximum_penetration_m"] = max(
        float(record.get("maximum_penetration_m", -1.0))
        for record in records
    )
    result["active_contact_samples"] = sum(
        int(record.get("active_contact_samples", 0)) for record in records
    )
    result["passed"] = bool(
        len(records) == len(args.targets) * len(args.seeds)
        and all(bool(record["passed"]) for record in records)
    )
    _write_result(args.output, result)
    print(
        json.dumps(
            {
                "trial_count": result["trial_count"],
                "trial_pass_count": result["trial_pass_count"],
                "maximum_penetration_m": result["maximum_penetration_m"],
                "active_contact_samples": result["active_contact_samples"],
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
