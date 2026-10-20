#!/usr/bin/env python3
"""Bounded full-scene Isaac/PhysX load, reset, and step smoke for P0-06."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

from isaaclab.app import AppLauncher  # noqa: E402


def _readback_is_finite(readback) -> bool:
    values: list[float] = []
    for value in readback.__dict__.values():
        values.extend(np.asarray(value, dtype=np.float64).reshape(-1).tolist())
    return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False
    args.device = str(getattr(args, "device", None) or "cuda:0")
    os.environ["HANDSON_ISAACLAB_ARM_IK_BACKEND"] = "haorui"
    launcher = AppLauncher(args)
    app = launcher.app
    environment = None
    result: dict[str, object] = {
        "schema": "handson.p006.runtime_reset_smoke.v1",
        "passed": False,
    }
    rc = 1
    try:
        from handson.sim.isaaclab.p006_collection_env import (
            HandsOnP006IsaacLabEnv,
        )

        environment = HandsOnP006IsaacLabEnv(
            seed=0,
            control_dt=1.0 / 30.0,
            physics_dt=1.0 / 120.0,
            time_limit=24.0,
            render_mode="rgb_array",
            image_obs=False,
            randomize=True,
            randomize_dynamics=False,
            hand_type="wuji1",
            arm_type="rokae",
            launch_isaac=True,
            headless=True,
            isaac_device=args.device,
            physics_backend="physx",
            existing_simulation_app=app,
            existing_app_launcher=launcher,
        )
        observation, info = environment.reset(seed=0)
        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        step_rows = []
        for index in range(4):
            observation, reward, terminated, truncated, info = environment.step(action)
            readback = environment._physical_readback()
            step_rows.append(
                {
                    "step": index + 1,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "finite": _readback_is_finite(readback),
                    "max_penetration_m": float(readback.max_penetration_m),
                }
            )
        contacts = {
            key: {
                "sensor_count": int(view.sensor_count),
                "filter_count": int(view.filter_count),
            }
            for key, view in environment._p006_contact_views.items()
        }
        body_counts = {
            key: int(view.count) for key, view in environment._p006_body_views.items()
        }
        checks = {
            "dual_robots": sorted(environment._robots) == ["left", "right"],
            "two_buckle_bodies": body_counts == {"female": 1, "male": 1},
            "six_contact_views": len(contacts) == 6
            and all(row["sensor_count"] == 1 for row in contacts.values()),
            "reset_not_success": not bool(info.get("succeed", False)),
            "reset_not_failed": not bool(info.get("failed", False)),
            "four_finite_steps": all(row["finite"] for row in step_rows),
            "penetration_within_gate": all(
                row["max_penetration_m"] <= 0.005 for row in step_rows
            ),
        }
        result.update(
            {
                "passed": all(checks.values()),
                "checks": checks,
                "body_counts": body_counts,
                "contact_views": contacts,
                "reset_sample": environment._p006_reset.to_dict(),
                "observation_state_keys": sorted(observation["state"]),
                "steps": step_rows,
            }
        )
        rc = 0 if result["passed"] else 2
    except Exception as exc:
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if environment is not None:
            try:
                environment.close()
            except Exception:
                traceback.print_exc()
        try:
            app.close()
        except SystemExit:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
