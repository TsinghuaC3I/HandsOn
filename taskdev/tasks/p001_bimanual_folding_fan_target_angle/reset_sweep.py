"""Single-session reset/layout acceptance sweep for P0-01."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from handson.sim.isaaclab.p001_contract import (
    FRICTION_RANGE,
    INITIAL_SPAN_RANGE_DEG,
    MASS_SCALE_RANGE,
    NOMINAL_ROOT_XY_M,
    NOMINAL_ROOT_YAW_DEG,
    ROOT_XY_HALF_RANGE_M,
    ROOT_YAW_HALF_RANGE_DEG,
    TARGET_OPEN_CHOICES_DEG,
)
from handson.tasks.t48_unfold_fan.config import TaskConfig


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _observation_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_observation_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_observation_finite(item) for item in value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        return True
    return bool(np.all(np.isfinite(array)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts-per-target", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.layouts_per_target <= 0:
        raise ValueError("--layouts-per-target must be positive")

    env = TaskConfig().get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=True,
        sim_backend="isaacsim",
        seed=args.seed_base,
        target_open_deg=60.0,
        image_obs=False,
        show_sim_cameras=False,
    )
    records: list[dict[str, Any]] = []
    try:
        for target_index, target in enumerate(TARGET_OPEN_CHOICES_DEG):
            for layout_index in range(args.layouts_per_target):
                seed = (
                    int(args.seed_base)
                    + 100 * target_index
                    + layout_index
                )
                observation, reset_info = env.reset(
                    seed=seed, target_open_deg=float(target)
                )
                raw_env = env.unwrapped
                reset_state = raw_env._p001_reset
                action = np.zeros(env.action_space.shape, dtype=np.float64)
                _observation, _reward, terminated, truncated, step_info = env.step(
                    action
                )
                reset_dict = reset_state.to_dict()
                checks = {
                    "seed": int(reset_dict["seed"]) == seed,
                    "target": float(reset_dict["target_open_deg"])
                    == float(target),
                    "root_x": abs(
                        float(reset_dict["root_x_m"])
                        - float(NOMINAL_ROOT_XY_M[0])
                    )
                    <= ROOT_XY_HALF_RANGE_M,
                    "root_y": abs(
                        float(reset_dict["root_y_m"])
                        - float(NOMINAL_ROOT_XY_M[1])
                    )
                    <= ROOT_XY_HALF_RANGE_M,
                    "root_yaw": abs(
                        float(reset_dict["root_yaw_deg"])
                        - NOMINAL_ROOT_YAW_DEG
                    )
                    <= ROOT_YAW_HALF_RANGE_DEG,
                    "initial_span": INITIAL_SPAN_RANGE_DEG[0]
                    <= float(reset_dict["initial_span_deg"])
                    <= INITIAL_SPAN_RANGE_DEG[1],
                    "friction": FRICTION_RANGE[0]
                    <= float(reset_dict["friction"])
                    <= FRICTION_RANGE[1],
                    "mass_scale": MASS_SCALE_RANGE[0]
                    <= float(reset_dict["mass_scale"])
                    <= MASS_SCALE_RANGE[1],
                    "reset_observation_finite": _observation_finite(observation),
                    "step_observation_finite": _observation_finite(_observation),
                    "not_terminal": not bool(terminated or truncated),
                    "no_failure": not bool(step_info.get("p001_failure", True)),
                    "no_object_writes": int(
                        step_info.get(
                            "p001_object_state_writes_after_action_zero", -1
                        )
                    )
                    == 0,
                }
                records.append(
                    {
                        "target_deg": float(target),
                        "layout_index": layout_index,
                        "seed": seed,
                        "reset": reset_dict,
                        "reset_info": _jsonable(reset_info),
                        "step_info": _jsonable(step_info),
                        "checks": checks,
                        "passed": all(checks.values()),
                    }
                )
        unique_layouts = {
            (
                record["target_deg"],
                record["reset"]["root_x_m"],
                record["reset"]["root_y_m"],
                record["reset"]["root_yaw_deg"],
                record["reset"]["initial_span_deg"],
            )
            for record in records
        }
        counts = {
            str(int(target)): sum(
                record["target_deg"] == float(target) for record in records
            )
            for target in TARGET_OPEN_CHOICES_DEG
        }
        result = {
            "schema": "p001_reset_sweep_v1",
            "layouts_per_target": int(args.layouts_per_target),
            "records": records,
            "target_counts": counts,
            "unique_layout_count": len(unique_layouts),
            "passed": bool(
                len(records)
                == len(TARGET_OPEN_CHOICES_DEG) * args.layouts_per_target
                and len(unique_layouts) == len(records)
                and all(record["passed"] for record in records)
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps({key: value for key, value in result.items() if key != "records"}, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
