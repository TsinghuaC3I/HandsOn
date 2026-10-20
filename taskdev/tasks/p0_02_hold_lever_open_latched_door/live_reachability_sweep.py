#!/usr/bin/env python3
"""Find comfortable P0-02 palm targets using the production online IK path."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-usd-root", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False
    args.device = str(getattr(args, "device", None) or "cuda:0")
    os.environ["HANDSON_P002_PREBUILT_ROBOT_USD_ROOT"] = str(
        args.robot_usd_root.resolve()
    )
    # Use the repository's exact multi-start solver for this kinematic gate;
    # the collection runtime remains on its standard incremental DLS path.
    os.environ["HANDSON_ISAACLAB_ARM_IK_BACKEND"] = "haorui"
    launcher = AppLauncher(args)
    app = launcher.app
    environment = None
    result: dict[str, object] = {"passed": False}
    return_code = 1
    try:
        from handson.sim.isaaclab.p002_collection_env import HandsOnP002IsaacLabEnv
        from handson.sim.isaaclab.p002_contract import (
            P002ScriptedPolicy,
            sample_reset_state,
        )

        environment = HandsOnP002IsaacLabEnv(
            seed=0, control_dt=1.0 / 30.0, physics_dt=1.0 / 120.0,
            time_limit=30.0, render_mode="rgb_array", image_obs=False,
            randomize=False, randomize_dynamics=False, hand_type="wuji1",
            arm_type="rokae", launch_isaac=True, headless=True,
            isaac_device=args.device, physics_backend="physx",
            existing_simulation_app=app, existing_app_launcher=launcher,
        )
        environment.reset(seed=0)
        rows: list[dict[str, object]] = []
        inventory: dict[str, object] = {}
        for side in ("left", "right"):
            robot = environment._robots[side]
            model = environment._haorui_ik_models[side]
            q_now = environment._arm_reference_q(side, robot)
            base = environment._robot_base_pose_matrix(side, robot)
            inventory[side] = {
                "base_matrix": base.tolist(),
                "reference_q_rad": np.asarray(q_now, dtype=np.float64).tolist(),
                "current_tcp_matrix": (base @ model.fk(q_now)).tolist(),
                "current_ee_matrix": environment._robot_body_pose_matrix(
                    robot, environment._ee_body_indices[side]
                ).tolist(),
            }
        for seed in range(20):
            policy = P002ScriptedPolicy(sample_reset_state(seed))
            role_targets = {
                "right_handle_press": ("right", policy.action(190)[:7]),
                "left_door_closed": ("left", policy.action(120)[7:14]),
            }
            for role, (side, pose) in role_targets.items():
                robot = environment._robots[side]
                limits = np.asarray(
                    robot.data.soft_joint_pos_limits[
                        0, environment._arm_joint_ids[side]
                    ].detach().cpu(),
                    dtype=np.float64,
                )
                q = environment._compute_arm_joint_target_haorui(
                    side, robot, pose
                )
                row: dict[str, object] = {
                    "seed": seed,
                    "role": role,
                    "side": side,
                    "target_pose_wxyz": pose.tolist(),
                    "solved": q is not None,
                }
                if q is not None:
                    q_np = np.asarray(q.detach().cpu(), dtype=np.float64)
                    row["joint_limit_margin_rad"] = float(
                        np.min(
                            np.minimum(
                                q_np - limits[:, 0],
                                limits[:, 1] - q_np,
                            )
                        )
                    )
                rows.append(row)

        role_summaries: list[dict[str, object]] = []
        for role, side in (
            ("right_handle_press", "right"),
            ("left_door_closed", "left"),
        ):
            role_rows = [row for row in rows if row["role"] == role]
            margins = [
                float(row["joint_limit_margin_rad"])
                for row in role_rows
                if row["solved"]
            ]
            role_summaries.append(
                {
                    "role": role,
                    "side": side,
                    "solved_count": sum(bool(row["solved"]) for row in role_rows),
                    "trial_count": len(role_rows),
                    "min_joint_limit_margin_rad": min(margins, default=-1.0),
                }
            )
        robust = [
            row
            for row in role_summaries
            if row["solved_count"] == 20
            and row["min_joint_limit_margin_rad"] >= 0.05
        ]
        result = {
            "passed": len(robust) == 2,
            "inventory": inventory,
            "role_count": len(role_summaries),
            "trial_count": len(rows),
            "role_summaries": role_summaries,
            "robust_roles": robust,
            "trials": rows,
        }
        return_code = 0 if result["passed"] else 2
    except Exception as exc:
        result = {"passed": False, "exception": {
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        }}
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
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
