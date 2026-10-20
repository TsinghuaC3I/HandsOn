#!/usr/bin/env python3
"""Exercise the live B03 Newton hand slew and strict source-validity gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "handson"
for path in (_REPOSITORY_ROOT, _PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("HANDSON_RECORD_NEWTON_GROUND_TRUTH", "1")


def _hand_layout(schema: dict[str, Any]) -> list[tuple[int, int, str]]:
    newton = schema["newton"]
    rows: list[tuple[int, int, str]] = []
    for side_index, side in enumerate(newton["robot_sides"]):
        for joint_index, name in enumerate(newton["robot_joint_names"][side]):
            if str(name).startswith(f"{side}_finger"):
                rows.append((side_index, joint_index, str(name)))
    return rows


def main() -> int:
    from handson.sim.isaaclab import abc_collection_env
    from handson.data.b03_newton_validity import (
        B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S,
        evaluate_b03_newton_substep_observation,
    )
    from handson.sim.isaaclab.newton_replay import destroy_newton_scene
    from handson.tasks.t09_insert_peg.config import TaskConfig

    physics_hz = float(os.environ.get("HANDSON_SMOKE_PHYSICS_HZ", "120"))
    if physics_hz <= 0.0:
        raise ValueError("HANDSON_SMOKE_PHYSICS_HZ must be positive")
    # Production is fail-closed at the qualified fallback frequency.  This
    # diagnostic intentionally exercises both 120 Hz and 240 Hz without
    # weakening that production default.
    abc_collection_env._B03_NEWTON_ARTICULATED_PHYSICS_HZ = physics_hz

    environment = None
    payload: dict[str, Any] = {"valid": False}
    try:
        environment = TaskConfig().get_native_environment(
            sim_backend="isaacsim_newton",
            policy_mode=True,
            render_mode="rgb_array",
            randomize=False,
            randomize_dynamics=False,
            image_obs=False,
            physics_dt=1.0 / physics_hz,
            seed=20260810,
            newton_replay_profile="cross_gpu_v1",
        )
        _observation, _info = environment.reset(seed=20260810)
        raw = environment.unwrapped
        schema = raw.get_recording_schema()
        right_robot = raw._robots["right"]
        right_hand_actuator_cfg = right_robot.cfg.actuators["hand"]
        payload.update(
            {
                "physics_hz": physics_hz,
                "actuator_runtime_keys": sorted(right_robot.actuators),
                "actuator_cfg_class": type(right_hand_actuator_cfg).__name__,
                "actuator_velocity_limit": float(
                    right_hand_actuator_cfg.velocity_limit
                ),
                "actuator_saturation_effort": float(
                    right_hand_actuator_cfg.saturation_effort
                ),
            }
        )
        hand_layout = _hand_layout(schema)
        newton = schema["newton"]
        shaping = newton["physics_config"]["hand_target_shaping"]
        payload["asset_identity"] = newton["physics_config"][
            "asset_identity"
        ]
        payload["qd_projection_sha256"] = newton["physics_config"][
            "joint_velocity_projection"
        ]["combined_sha256"]
        maximum_target_delta = float(
            shaping["maximum_delta_per_physics_substep_rad"]
        )

        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        if action.shape[0] < 27:
            raise RuntimeError(f"unexpected B03 action shape: {action.shape}")

        target_rows: list[np.ndarray] = []
        internal_snapshots: list[dict[str, Any]] = []
        control_actual_hz: list[float] = []
        control_work_ms: list[float] = []
        maximum_hand_velocity = 0.0
        maximum_penetration = 0.0
        commands = (
            (0.0, 4),
            (1.0, 24),
            (-1.0, 24),
            (0.0, 12),
        )
        control_step = 0
        for command, frame_count in commands:
            action[7:] = command
            for _ in range(frame_count):
                observation, _, terminated, truncated, _info = environment.step(
                    action
                )
                timing = _info.get("control_timing", {})
                try:
                    actual_hz = float(timing["actual_hz"])
                    work_ms = float(timing["work_ms"])
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    if np.isfinite(actual_hz) and actual_hz > 0.0:
                        control_actual_hz.append(actual_hz)
                    if np.isfinite(work_ms) and work_ms >= 0.0:
                        control_work_ms.append(work_ms)
                report = evaluate_b03_newton_substep_observation(
                    observation,
                    schema,
                )
                if not report.get("valid"):
                    payload.update(
                        {
                            "valid": False,
                            "control_step": control_step,
                            "report": report,
                        }
                    )
                    return 1

                maximum_hand_velocity = max(
                    maximum_hand_velocity,
                    float(report["hand_max_abs_joint_velocity_rad_s"]),
                )
                maximum_penetration = max(
                    maximum_penetration,
                    float(report["hand_peg_max_solid_penetration_m"]),
                )
                targets = np.asarray(
                    observation["newton_substep_robot_joint_target"],
                    dtype=np.float64,
                )
                hand_targets = np.stack(
                    [
                        targets[:, side_index, joint_index]
                        for side_index, joint_index, _name in hand_layout
                    ],
                    axis=-1,
                )
                target_rows.append(hand_targets)
                if control_step in {0, 3, 4, 7, 15, 27, 31, 51, 55, 63}:
                    applied = np.asarray(
                        raw._joint_targets["right"].detach().cpu(),
                        dtype=np.float64,
                    )[0, raw._hand_joint_ids["right"]]
                    goal = np.asarray(
                        raw._newton_hand_goal_targets["right"].detach().cpu(),
                        dtype=np.float64,
                    )[0]
                    internal_snapshots.append(
                        {
                            "control_step": control_step,
                            "command": float(command),
                            "applied_min": float(np.min(applied)),
                            "applied_max": float(np.max(applied)),
                            "goal_min": float(np.min(goal)),
                            "goal_max": float(np.max(goal)),
                            "trace_min": float(np.min(hand_targets)),
                            "trace_max": float(np.max(hand_targets)),
                        }
                    )
                control_step += 1
                if terminated or truncated:
                    raise RuntimeError(
                        f"B03 ended during slew stress at control step {control_step}"
                    )

        flattened_targets = np.concatenate(target_rows, axis=0)
        target_delta = np.abs(np.diff(flattened_targets, axis=0))
        maximum_observed_target_delta = float(np.max(target_delta))
        tolerance = 5.0e-7
        slew_valid = (
            maximum_observed_target_delta > maximum_target_delta * 0.5
            and maximum_observed_target_delta
            <= maximum_target_delta + tolerance
        )
        # The first four neutral frames include lazy runtime/ring setup.  Keep
        # them in the physical stress result but exclude them from the steady
        # control-throughput summary.
        steady_actual_hz = np.asarray(control_actual_hz[4:], dtype=np.float64)
        steady_work_ms = np.asarray(control_work_ms[4:], dtype=np.float64)
        payload.update(
            {
                "valid": bool(slew_valid),
                "control_steps": control_step,
                "physics_substeps": control_step * int(newton["n_substeps"]),
                "hand_joint_count": len(hand_layout),
                "slew_rate_rad_s": float(shaping["rate_rad_s"]),
                "slew_enabled": bool(shaping["enabled"]),
                "target_delta_limit_rad": maximum_target_delta,
                "target_delta_observed_rad": maximum_observed_target_delta,
                "hand_velocity_limit_rad_s": (
                    B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S
                ),
                "hand_velocity_observed_rad_s": maximum_hand_velocity,
                "penetration_observed_m": maximum_penetration,
                "steady_control_hz_mean": (
                    float(np.mean(steady_actual_hz))
                    if steady_actual_hz.size
                    else None
                ),
                "steady_control_hz_median": (
                    float(np.median(steady_actual_hz))
                    if steady_actual_hz.size
                    else None
                ),
                "steady_control_work_ms_p95": (
                    float(np.percentile(steady_work_ms, 95.0))
                    if steady_work_ms.size
                    else None
                ),
                "state_writes": int(raw._trajectory_state_write_count),
                "internal_snapshots": internal_snapshots,
                "source_sha256": newton["source_runtime_identity"][
                    "b03_physics_source_sha256"
                ],
            }
        )
        return 0 if slew_valid else 1
    except BaseException as exc:
        payload.update(
            {
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return 1
    finally:
        if environment is not None:
            payload["lifecycle"] = destroy_newton_scene(environment)
        summary_keys = (
            "valid",
            "error",
            "physics_hz",
            "control_step",
            "control_steps",
            "physics_substeps",
            "hand_velocity_limit_rad_s",
            "hand_velocity_observed_rad_s",
            "penetration_observed_m",
            "target_delta_limit_rad",
            "target_delta_observed_rad",
            "state_writes",
            "steady_control_hz_mean",
            "steady_control_hz_median",
            "steady_control_work_ms_p95",
            "source_sha256",
            "qd_projection_sha256",
        )
        summary = {key: payload[key] for key in summary_keys if key in payload}
        report = payload.get("report")
        if isinstance(report, dict):
            summary["invalid_reasons"] = report.get("invalid_reasons")
            summary["hand_velocity_observed_rad_s"] = report.get(
                "hand_max_abs_joint_velocity_rad_s"
            )
            summary["penetration_observed_m"] = report.get(
                "hand_peg_max_solid_penetration_m"
            )
        print(
            "B03_NEWTON_HAND_SLEW_SUMMARY="
            + json.dumps(summary, sort_keys=True),
            flush=True,
        )
        print(
            "B03_NEWTON_HAND_SLEW=" + json.dumps(payload, sort_keys=True),
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
