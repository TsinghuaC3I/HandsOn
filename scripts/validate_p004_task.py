#!/usr/bin/env python3
"""Validate P0-04 randomized resets and idle physics in the real Isaac runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np


TASK_NAME = "p004_hold_spring_clamp_open_insert_card"
REQUIRED_STATE_SHAPES = {
    "card_pose": (7,),
    "clamp_pose": (7,),
    "task_stage": (1,),
    "clamp_joint_state": (2,),
    "card_insertion_depth": (1,),
    "card_alignment": (1,),
    "handle_hold_state": (5,),
    "card_contact_state": (3,),
    "stage_lineage": (7,),
}
MAX_IDLE_CARD_DRIFT_M = 0.004
MAX_IDLE_CLAMP_DRIFT_M = 0.004


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed-start", type=int, default=2026082900)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--idle-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def state_from(observation: dict[str, Any]) -> dict[str, Any]:
    state = observation.get("state")
    if not isinstance(state, dict):
        raise RuntimeError("P0-04 observation has no state dictionary")
    return state


def validate_state(state: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key, expected_shape in REQUIRED_STATE_SHAPES.items():
        if key not in state:
            failures.append(f"state is missing {key}")
            continue
        value = np.asarray(state[key])
        if value.shape != expected_shape:
            failures.append(f"state {key} shape is {value.shape}, expected {expected_shape}")
        elif not np.all(np.isfinite(value)):
            failures.append(f"state {key} contains non-finite values")
    return failures


def main() -> int:
    args = parse_args()
    if args.seed_count < 20:
        raise ValueError("P0-04 randomized reset gate requires at least 20 seeds")
    if args.idle_steps < 3:
        raise ValueError("P0-04 idle gate requires at least 3 control steps")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "p004_reset_idle_validation.json"

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "handson"))
    from handson.sim.isaaclab.p004_hold_spring_clamp_open_insert_card_env import (
        HandsOnP004IsaacLabEnv,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "task": TASK_NAME,
        "gate": "randomized_reset_and_idle_physics",
        "passed": False,
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "idle_steps": int(args.idle_steps),
        "device_argument": str(args.device),
        "taskdev_physical_gpu": os.environ.get("TASKDEV_PHYSICAL_GPU"),
        "taskdev_run_id": os.environ.get("TASKDEV_RUN_ID"),
        "phase": "starting",
        "seeds": [],
    }
    write_report(report_path, report)
    environment = None
    try:
        environment = HandsOnP004IsaacLabEnv(
            TASK_NAME,
            seed=int(args.seed_start),
            render_mode="rgb_array",
            image_obs=False,
            randomize=True,
            randomize_dynamics=False,
            launch_isaac=True,
            headless=True,
            isaac_device=str(args.device),
            hz=0,
            time_limit=120.0,
        )

        import torch

        gpu_failures: list[str] = []
        require(torch.cuda.is_available(), "torch reports CUDA unavailable", gpu_failures)
        cuda_count = int(torch.cuda.device_count())
        require(cuda_count == 1, f"expected one isolated CUDA GPU, observed {cuda_count}", gpu_failures)
        if cuda_count:
            torch.zeros(1, device="cuda:0")
            torch.cuda.synchronize(0)
        report["gpu"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": cuda_count,
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda_count else None,
            "failures": gpu_failures,
        }

        all_failures = list(gpu_failures)
        card_xy_samples: list[np.ndarray] = []
        clamp_xy_samples: list[np.ndarray] = []
        report["phase"] = "validating"
        for offset in range(int(args.seed_count)):
            seed = int(args.seed_start) + offset
            observation, info = environment.reset(seed=seed)
            state = state_from(observation)
            card_initial = np.asarray(state["card_pose"], dtype=np.float64).copy()
            clamp_initial = np.asarray(state["clamp_pose"], dtype=np.float64).copy()
            card_xy_samples.append(card_initial[:2])
            clamp_xy_samples.append(clamp_initial[:2])
            failures = validate_state(state)
            warnings = [str(item) for item in info.get("isaac_warnings", ())]

            require(bool(info.get("p004_finite")), "non-finite reset state", failures)
            require(bool(info.get("p004_contacts_ready")), "contact views unavailable", failures)
            require(str(info.get("p004_stage_name")) == "RESET_CLOSED", "stage is not RESET_CLOSED", failures)
            require(not bool(info.get("succeed")), "reset is already successful", failures)
            require(not bool(info.get("failed")), "reset is already failed", failures)
            require(info.get("p004_failure_reason") in (None, ""), "reset has a failure reason", failures)
            require(float(info.get("p004_clamp_angle_rad", math.inf)) <= 0.06, "clamp is not closed at reset", failures)
            require(float(info.get("p004_insertion_depth_m", math.inf)) < 0.0, "card begins inserted", failures)
            require(not bool(info.get("p004_left_fixed_contact")), "left hand touches fixed jaw at reset", failures)
            require(not bool(info.get("p004_left_moving_contact")), "left hand touches moving jaw at reset", failures)
            require(not bool(info.get("p004_right_card_contact")), "right hand touches card at reset", failures)
            require(not bool(info.get("p004_card_fixed_jaw_contact")), "card touches fixed jaw at reset", failures)
            require(not bool(info.get("p004_card_moving_jaw_contact")), "card touches moving jaw at reset", failures)
            require(not bool(info.get("p004_forbidden_wedge_contact")), "moving jaw is wedged at reset", failures)
            require(
                not any("unavailable" in warning.lower() or "fallback" in warning.lower() for warning in warnings),
                "Isaac runtime reported unavailable/fallback data",
                failures,
            )

            zero_action = np.zeros(environment.action_space.shape, dtype=np.float32)
            final_observation = observation
            final_info = info
            for _ in range(int(args.idle_steps)):
                final_observation, _reward, terminated, truncated, final_info = environment.step(zero_action)
                require(not terminated, "idle validation terminated", failures)
                require(not truncated, "idle validation truncated", failures)
                if terminated or truncated:
                    break

            final_state = state_from(final_observation)
            failures.extend(validate_state(final_state))
            card_final = np.asarray(final_state["card_pose"], dtype=np.float64)
            clamp_final = np.asarray(final_state["clamp_pose"], dtype=np.float64)
            card_drift = float(np.linalg.norm(card_final[:3] - card_initial[:3]))
            clamp_drift = float(np.linalg.norm(clamp_final[:3] - clamp_initial[:3]))
            require(card_drift <= MAX_IDLE_CARD_DRIFT_M, f"idle card drift {card_drift:.6f} m", failures)
            require(clamp_drift <= MAX_IDLE_CLAMP_DRIFT_M, f"idle clamp drift {clamp_drift:.6f} m", failures)
            require(bool(final_info.get("p004_finite")), "idle state is non-finite", failures)
            require(str(final_info.get("p004_stage_name")) == "RESET_CLOSED", "idle action advanced stage", failures)
            require(not bool(final_info.get("succeed")), "idle action produced success", failures)
            require(not bool(final_info.get("failed")), "idle action produced failure", failures)

            report["seeds"].append(
                {
                    "seed": seed,
                    "passed": not failures,
                    "failures": failures,
                    "warnings": warnings,
                    "card_initial_pose": card_initial,
                    "card_final_pose": card_final,
                    "clamp_initial_pose": clamp_initial,
                    "clamp_final_pose": clamp_final,
                    "card_drift_m": card_drift,
                    "clamp_drift_m": clamp_drift,
                    "reset_metrics": {key: value for key, value in info.items() if str(key).startswith("p004_")},
                    "final_metrics": {key: value for key, value in final_info.items() if str(key).startswith("p004_")},
                }
            )
            all_failures.extend(f"seed {seed}: {failure}" for failure in failures)
            report["failures"] = all_failures
            write_report(report_path, report)
            print(
                f"[P004] seed={seed} passed={not failures} "
                f"card_drift={card_drift:.6f}m clamp_drift={clamp_drift:.6f}m",
                flush=True,
            )

        unique_card_xy = len({tuple(np.round(value, 5)) for value in card_xy_samples})
        unique_clamp_xy = len({tuple(np.round(value, 5)) for value in clamp_xy_samples})
        if unique_card_xy < 10:
            all_failures.append(f"only {unique_card_xy} unique randomized card XY poses")
        if unique_clamp_xy < 10:
            all_failures.append(f"only {unique_clamp_xy} unique randomized clamp XY poses")
        report["randomization"] = {
            "unique_card_xy": unique_card_xy,
            "unique_clamp_xy": unique_clamp_xy,
            "card_xy_span_m": np.ptp(np.asarray(card_xy_samples), axis=0),
            "clamp_xy_span_m": np.ptp(np.asarray(clamp_xy_samples), axis=0),
        }
        report["failures"] = all_failures
        report["passed"] = not all_failures
        report["phase"] = "validation_complete"
        write_report(report_path, report)
    except BaseException as exc:
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["passed"] = False
        report["phase"] = "validation_failed"
        write_report(report_path, report)
    finally:
        if environment is not None:
            try:
                environment._fast_shutdown_on_close = True
                environment.close()
            except Exception as exc:
                report["close_error"] = f"{type(exc).__name__}: {exc}"
                report["passed"] = False
        report["phase"] = "finished"
        write_report(report_path, report)

    print(f"[P004] {'PASS' if report.get('passed') else 'FAIL'} report={report_path}", flush=True)
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
