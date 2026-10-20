#!/usr/bin/env python3
"""Bounded pre-human P0-06 action diagnostic; not acceptance evidence."""

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


TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

from isaaclab.app import AppLauncher  # noqa: E402


WUJI_POWER_GRIP = np.asarray(
    [
        0.60, 0.10, 0.84, 0.70, 0.68,
        0.00, 0.90, 0.78, 0.70, 0.00,
        0.92, 0.80, 0.68, 0.00, 0.90,
        0.78, 0.70, 0.00, 0.88, 0.76,
    ],
    dtype=np.float64,
)
RIGHT_PRECISION_GRIP = WUJI_POWER_GRIP.copy()
RIGHT_PRECISION_GRIP[4:8] = 0.0
RIGHT_PRECISION_GRIP[16:20] = 0.0
LEFT_PRECISION_GRIP = WUJI_POWER_GRIP.copy()
LEFT_PRECISION_GRIP[12:20] = 0.0


def _blend(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    value = (1.0 - fraction) * np.asarray(start, dtype=np.float64) + fraction * np.asarray(
        end, dtype=np.float64
    )
    if value.shape == (7,):
        norm = float(np.linalg.norm(value[3:7]))
        if norm <= 0.0:
            raise ValueError("zero quaternion in P0-06 scripted interpolation")
        value[3:7] /= norm
    return value


def _phase_action(
    step: int,
    right_home: np.ndarray,
    left_home: np.ndarray,
    reset: Any,
) -> tuple[str, np.ndarray]:
    quat = np.asarray([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])
    right_pre = np.asarray(
        [
            reset.female_position_m[0] - 0.115,
            reset.female_position_m[1] - 0.030,
            1.080,
            *quat,
        ]
    )
    right_grasp = right_pre.copy()
    right_grasp[2] = 0.970
    right_lift = right_grasp.copy()
    right_lift[2] = 0.995

    left_pre = np.asarray(
        [
            reset.male_position_m[0] - 0.115,
            reset.male_position_m[1] + 0.015,
            1.070,
            *quat,
        ]
    )
    left_grasp = left_pre.copy()
    left_grasp[2] = 0.960
    left_lift = left_grasp.copy()
    left_lift[2] = 0.985
    left_press = left_lift.copy()
    left_press[0] += reset.female_position_m[0] - reset.male_position_m[0]
    left_press[1] = reset.female_position_m[1] + 0.080

    if step < 60:
        phase = "approach"
        right = _blend(right_home, right_pre, step / 60.0)
        left = _blend(left_home, left_pre, step / 60.0)
        grip = 0.0
    elif step < 105:
        phase = "preshape"
        right = right_pre
        left = left_pre
        grip = 0.65 * ((step - 60) / 45.0)
    elif step < 165:
        phase = "descend"
        right = _blend(right_pre, right_grasp, (step - 105) / 60.0)
        left = _blend(left_pre, left_grasp, (step - 105) / 60.0)
        grip = 0.65
    elif step < 210:
        phase = "close"
        right = right_grasp
        left = left_grasp
        grip = 0.65 + 0.05 * ((step - 165) / 45.0)
    elif step < 270:
        phase = "lift"
        right = _blend(right_grasp, right_lift, (step - 210) / 60.0)
        left = _blend(left_grasp, left_lift, (step - 210) / 60.0)
        grip = 0.70
    elif step < 390:
        phase = "align_press"
        right = right_lift
        left = _blend(left_lift, left_press, (step - 270) / 120.0)
        grip = 0.70
    else:
        phase = "hold"
        right = right_lift
        left = left_press
        grip = 0.70

    action = np.zeros(54, dtype=np.float64)
    action[:7] = right
    action[7:14] = left
    action[14:34] = grip * RIGHT_PRECISION_GRIP
    action[34:54] = grip * LEFT_PRECISION_GRIP
    return phase, action


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _actual_tcp_pose(environment: Any, side: str) -> np.ndarray:
    fallback = (
        environment._right_tcp_pose
        if side == "right"
        else environment._left_tcp_pose
    )
    matrix = environment._native_end_effector_pose_matrix(side, fallback)
    return np.asarray(environment._pose_matrix_to_vector(matrix), dtype=np.float64)


def _hand_body_positions(environment: Any, side: str) -> dict[str, list[float]]:
    robot = environment._robots[side]
    poses = robot.data.body_pose_w[0, :, :3].detach().cpu().numpy()
    return {
        str(name): np.asarray(poses[index], dtype=np.float64).tolist()
        for index, name in enumerate(robot.body_names)
        if "palm" in str(name) or "finger" in str(name)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=480)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False
    args.device = str(getattr(args, "device", None) or "cuda:0")
    os.environ["HANDSON_ISAACLAB_ARM_IK_BACKEND"] = "haorui"
    launcher = AppLauncher(args)
    app = launcher.app
    environment = None
    report: dict[str, Any] = {
        "schema": "handson.p006.pre_human_action_diagnostic.v1",
        "passed": False,
        "action_driven": True,
        "object_pose_writes_after_reset": 0,
    }
    rc = 1
    try:
        from handson.sim.isaaclab.p006_collection_env import (
            HandsOnP006IsaacLabEnv,
        )

        environment = HandsOnP006IsaacLabEnv(
            seed=int(args.seed),
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
        _observation, reset_info = environment.reset(seed=int(args.seed))
        right_home = _actual_tcp_pose(environment, "right")
        left_home = _actual_tcp_pose(environment, "left")
        samples: list[dict[str, Any]] = []
        phase_events: list[dict[str, Any]] = []
        previous_phase = ""
        terminated = False
        truncated = False
        last_info = dict(reset_info)
        peak_forces = {
            "right_female": 0.0,
            "left_male": 0.0,
            "buckle_body": 0.0,
            "robot_table": 0.0,
        }
        for step in range(min(int(args.steps), 720)):
            phase, action = _phase_action(
                step, right_home, left_home, environment._p006_reset
            )
            if phase != previous_phase:
                phase_events.append({"step": step, "phase": phase})
                previous_phase = phase
            _observation, reward, terminated, truncated, last_info = environment.step(
                action
            )
            readback = environment._p006_last_readback or environment._physical_readback()
            for key, value in (
                ("right_female", readback.right_female_contact_force_n),
                ("left_male", readback.left_male_contact_force_n),
                ("buckle_body", readback.buckle_body_contact_force_n),
                ("robot_table", readback.robot_table_contact_force_n),
            ):
                peak_forces[key] = max(peak_forces[key], float(value))
            if step % 10 == 0 or terminated or truncated:
                metrics = dict(environment._p006_last_metrics)
                samples.append(
                    {
                        "step": step,
                        "phase": phase,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "female_position_m": list(readback.female_position_m),
                        "male_position_m": list(readback.male_position_m),
                        "right_female_contact_force_n": float(
                            readback.right_female_contact_force_n
                        ),
                        "left_male_contact_force_n": float(
                            readback.left_male_contact_force_n
                        ),
                        "buckle_body_contact_force_n": float(
                            readback.buckle_body_contact_force_n
                        ),
                        "robot_table_contact_force_n": float(
                            readback.robot_table_contact_force_n
                        ),
                        "max_penetration_m": float(readback.max_penetration_m),
                        "metrics": metrics,
                        "right_tcp_actual_wxyz_m": _actual_tcp_pose(
                            environment, "right"
                        ).tolist(),
                        "left_tcp_actual_wxyz_m": _actual_tcp_pose(
                            environment, "left"
                        ).tolist(),
                        "right_tcp_command_wxyz_m": action[:7].tolist(),
                        "left_tcp_command_wxyz_m": action[7:14].tolist(),
                    }
                )
            if terminated or truncated:
                break

        final_readback = environment._p006_last_readback or environment._physical_readback()
        report.update(
            {
                "passed": bool(
                    last_info.get("succeed", False)
                    and not last_info.get("failed", False)
                    and not truncated
                ),
                "seed": int(args.seed),
                "steps_executed": int(environment.env_step),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "reset": environment._p006_reset.to_dict(),
                "phase_events": phase_events,
                "peak_contact_forces_n": peak_forces,
                "final_readback": final_readback.__dict__,
                "final_metrics": dict(environment._p006_last_metrics),
                "final_hand_body_positions_m": {
                    side: _hand_body_positions(environment, side)
                    for side in ("right", "left")
                },
                "failure_latched": bool(environment._p006_failure_latched),
                "last_info": {
                    key: value
                    for key, value in last_info.items()
                    if key.startswith("p006_") or key in {"succeed", "failed"}
                },
                "samples": samples,
            }
        )
        rc = 0 if report["passed"] else 2
    except Exception as exc:
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8"
        )
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
