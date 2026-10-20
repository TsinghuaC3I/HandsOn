#!/usr/bin/env python3
"""Strict 10-second PhysX priority-zero gate for the exact P0-04 child."""

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


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

from isaaclab.app import AppLauncher  # noqa: E402


PHYSICS_HZ = 120
CONTROL_HZ = 30
CARD_NAME = "p004_card"
CLAMP_NAME = "p004_spring_clamp_root_body"
CLAMP_JOINT_NAME = "p004_spring_clamp_moving_jaw_joint"
STAGE_NAME = "p004_card_stage"
PRIORITY_ZERO_LINEAR_SPEED_LIMIT_MPS = 0.001
PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RADPS = 0.002
STATIC_POSE_TRANSLATION_STEP_LIMIT_M = 1.0e-6
STATIC_POSE_ROTATION_STEP_LIMIT_RAD = 1.0e-5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reset-count", type=int, default=20)
    parser.add_argument("--stability-seconds", type=float, default=10.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _rotation_error(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left[:3, :3]).T @ np.asarray(right[:3, :3])
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _support_contact_reading(
    environment: Any,
    label: str,
) -> tuple[float, float, bool]:
    """Return exact-pair force and penetration from one P0-04 contact view."""

    view = environment._p004_contact_views.get(label)
    if view is None:
        return 0.0, math.inf, False
    try:
        force, penetration = environment._p004_contact_reading(label)
        valid = bool(
            np.isfinite(force)
            and np.isfinite(penetration)
            and int(view.sensor_count) == 1
            and int(view.filter_count) == 1
        )
        return force, penetration, valid
    except Exception:
        return 0.0, math.inf, False


def _task_state(environment: Any) -> dict[str, Any]:
    signals = environment._p004_signals()
    card = environment._current_body_matrix(CARD_NAME)
    clamp = environment._current_body_matrix(CLAMP_NAME)
    card_velocity = environment._physics_body_velocity(CARD_NAME)
    clamp_velocity = environment._physics_body_velocity(CLAMP_NAME)
    clamp_joint = environment._physics_dof_position(CLAMP_JOINT_NAME)
    clamp_joint_velocity = environment._physics_dof_velocity(CLAMP_JOINT_NAME)
    metrics = dict(environment._p004_last_metrics)
    authored_reset_pose = dict(environment._authored_reset_pose_report())
    finite = bool(
        card is not None
        and clamp is not None
        and card_velocity is not None
        and clamp_velocity is not None
        and clamp_joint is not None
        and clamp_joint_velocity is not None
        and np.all(np.isfinite(card))
        and np.all(np.isfinite(clamp))
        and np.all(np.isfinite(card_velocity))
        and np.all(np.isfinite(clamp_velocity))
        and np.all(np.isfinite(clamp_joint))
        and np.isfinite(float(clamp_joint_velocity))
        and signals.finite
    )
    return {
        "finite": finite,
        "card": None if card is None else np.asarray(card, dtype=np.float64),
        "clamp": None if clamp is None else np.asarray(clamp, dtype=np.float64),
        "card_velocity": (
            None
            if card_velocity is None
            else np.asarray(card_velocity, dtype=np.float64)
        ),
        "clamp_velocity": (
            None
            if clamp_velocity is None
            else np.asarray(clamp_velocity, dtype=np.float64)
        ),
        "clamp_joint": (
            None
            if clamp_joint is None
            else np.asarray(clamp_joint, dtype=np.float64)
        ),
        "clamp_joint_velocity": (
            None if clamp_joint_velocity is None else float(clamp_joint_velocity)
        ),
        "metrics": metrics,
        "authored_reset_pose": authored_reset_pose,
        "failed": bool(environment._p004_state.failed),
        "success": bool(environment._p004_state.success),
        "stage": int(environment._p004_state.stage),
    }


def main() -> int:
    args = _parser().parse_args()
    args.headless = True
    args.enable_cameras = False
    args.device = str(getattr(args, "device", None) or "cuda:0")
    os.environ["HANDSON_ISAACLAB_ARM_IK_BACKEND"] = "haorui"
    os.environ["HANDSON_ENABLE_ISAACSIM_NEWTON"] = "0"
    launcher = AppLauncher(args)
    app = launcher.app
    environment = None
    result: dict[str, Any] = {
        "schema": "handson.p004.priority_zero_live.v1",
        "passed": False,
        "runtime": {
            "passed": False,
            "requested_device": args.device,
        },
    }
    rc = 1
    try:
        from handson.sim.isaaclab.p004_hold_spring_clamp_open_insert_card_env import (
            HandsOnP004IsaacLabEnv,
        )
        from handson.sim.isaaclab.priority_zero_robot_audit import (
            RobotStabilityAudit,
            initial_robot_state_report,
            priority_zero_runtime_contract,
        )

        environment = HandsOnP004IsaacLabEnv(
            seed=0,
            control_dt=1.0 / CONTROL_HZ,
            physics_dt=1.0 / PHYSICS_HZ,
            time_limit=240.0,
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
        if environment._sim is None:
            raise RuntimeError("P0-04 exact production child has no PhysX context")
        runtime_contract = priority_zero_runtime_contract(
            environment,
            requested_device=args.device,
            physics_hz=PHYSICS_HZ,
            control_hz=CONTROL_HZ,
        )
        result["runtime"] = runtime_contract
        exact_production_child = bool(
            environment.__class__ is HandsOnP004IsaacLabEnv
            and environment.task_name
            == "p004_hold_spring_clamp_open_insert_card"
            and runtime_contract["passed"]
        )

        robot_inventory = {
            side: {
                "arm_joints": len(environment._arm_joint_ids[side]),
                "hand_joints": len(environment._hand_joint_ids[side]),
            }
            for side in ("left", "right")
        }
        contact_inventory = {
            name: {
                "sensor_count": int(view.sensor_count),
                "filter_count": int(view.filter_count),
            }
            for name, view in environment._p004_contact_views.items()
        }
        support_prims = {
            "card": environment._find_rigid_body_prim(CARD_NAME),
            "clamp": environment._find_rigid_body_prim(CLAMP_NAME),
            "moving_jaw": environment._find_rigid_body_prim(
                "p004_spring_clamp_moving_jaw"
            ),
            "stage": environment._find_rigid_body_prim(STAGE_NAME),
            "table": environment._find_rigid_body_prim("table"),
        }
        expected_support_pairs = (
            None
            if any(prim is None for prim in support_prims.values())
            else {
                "card_support": (
                    str(support_prims["card"].GetPath()),
                    (str(support_prims["stage"].GetPath()),),
                ),
                "clamp_support": (
                    str(support_prims["clamp"].GetPath()),
                    (str(support_prims["table"].GetPath()),),
                ),
                "moving_support": (
                    str(support_prims["moving_jaw"].GetPath()),
                    (str(support_prims["table"].GetPath()),),
                ),
            }
        )
        support_pairs_exact = bool(
            expected_support_pairs is not None
            and all(
                environment._p004_contact_paths.get(label) == pair
                and contact_inventory.get(label)
                == {"sensor_count": 1, "filter_count": 1}
                for label, pair in expected_support_pairs.items()
            )
        )
        reset_rows: list[dict[str, Any]] = []
        for seed in range(int(args.reset_count)):
            _observation, info = environment.reset(seed=seed)
            robot_passed, robot_report = initial_robot_state_report(environment)
            task = _task_state(environment)
            (
                card_support_force,
                card_support_penetration,
                card_support_valid,
            ) = _support_contact_reading(environment, "card_support")
            (
                clamp_support_force,
                clamp_support_penetration,
                clamp_support_valid,
            ) = _support_contact_reading(
                environment,
                "clamp_support",
            )
            reset_rows.append(
                {
                    "seed": seed,
                    "finite": task["finite"],
                    "initial_failed": bool(info.get("failed", False) or task["failed"]),
                    "initial_success": bool(
                        info.get("succeed", False) or task["success"]
                    ),
                    "stage": task["stage"],
                    "contacts_ready": bool(
                        task["metrics"].get("contacts_ready", False)
                    ),
                    # P004 workspace bounds are diagnostic telemetry only.  A
                    # reset remains valid when the card/clamp solver briefly
                    # leaves that envelope; physical finiteness, support,
                    # contact, and velocity checks below are the certifying
                    # reset contract.
                    "workspace_ok": bool(
                        task["metrics"].get("workspace_ok", False)
                    ),
                    "card_linear_speed_mps": (
                        float(np.linalg.norm(task["card_velocity"][:3]))
                        if task["card_velocity"] is not None
                        else math.inf
                    ),
                    "card_angular_speed_radps": (
                        float(np.linalg.norm(task["card_velocity"][3:]))
                        if task["card_velocity"] is not None
                        else math.inf
                    ),
                    "clamp_linear_speed_mps": (
                        float(np.linalg.norm(task["clamp_velocity"][:3]))
                        if task["clamp_velocity"] is not None
                        else math.inf
                    ),
                    "clamp_angular_speed_radps": (
                        float(np.linalg.norm(task["clamp_velocity"][3:]))
                        if task["clamp_velocity"] is not None
                        else math.inf
                    ),
                    "clamp_joint_speed_radps": (
                        abs(float(task["clamp_joint_velocity"]))
                        if task["clamp_joint_velocity"] is not None
                        else math.inf
                    ),
                    "clamp_angle_rad": (
                        float(task["clamp_joint"][0])
                        if task["clamp_joint"] is not None
                        else math.inf
                    ),
                    "authored_reset_pose": task["authored_reset_pose"],
                    "card_support_force_n": float(card_support_force),
                    "card_support_penetration_m": float(
                        card_support_penetration
                    ),
                    "card_support_readback_valid": card_support_valid,
                    "clamp_support_force_n": float(clamp_support_force),
                    "clamp_support_penetration_m": float(
                        clamp_support_penetration
                    ),
                    "clamp_support_readback_valid": clamp_support_valid,
                    "robot_initial_state_passed": robot_passed,
                    "robot_initial_state": robot_report,
                }
            )

        environment.reset(seed=0)
        initial = _task_state(environment)
        if not initial["finite"]:
            raise RuntimeError("P0-04 initial task state is non-finite")
        initial_task_velocity_exact_zero_observed = bool(
            np.array_equal(initial["card_velocity"], np.zeros(6))
            and np.array_equal(initial["clamp_velocity"], np.zeros(6))
            and float(initial["clamp_joint_velocity"]) == 0.0
        )
        robot_stability = RobotStabilityAudit.start(environment)
        stability_steps = int(round(float(args.stability_seconds) * PHYSICS_HZ))
        max_card_linear_speed = 0.0
        max_card_angular_speed = 0.0
        max_clamp_linear_speed = 0.0
        max_clamp_angular_speed = 0.0
        max_clamp_joint_speed = 0.0
        max_card_translation_drift_m = 0.0
        max_card_rotation_drift_rad = 0.0
        max_clamp_translation_drift_m = 0.0
        max_clamp_rotation_drift_rad = 0.0
        max_clamp_joint_drift_rad = 0.0
        task_velocity_invalid_samples = 0
        pose_increment_samples = 0
        pose_increment_invalid_samples = 0
        static_pose_covered_steps = 0
        max_pose_step_translation_m = 0.0
        max_pose_step_rotation_rad = 0.0
        max_clamp_joint_step_rad = 0.0
        previous_card_pose = initial["card"].copy()
        previous_clamp_pose = initial["clamp"].copy()
        previous_clamp_joint_position = float(initial["clamp_joint"][0])
        authored_reset_pose_samples = 0
        support_samples = 0
        supported_samples = 0
        support_readback_invalid_samples = 0
        max_support_penetration = 0.0
        contact_readback_invalid_samples = 0
        max_contact_penetration = 0.0
        nonfinite_samples = 0
        stability_samples: list[dict[str, Any]] = []
        for step in range(stability_steps):
            for robot in environment._robots.values():
                robot.write_data_to_sim()
            environment._sim.step(render=False)
            for robot in environment._robots.values():
                robot.update(environment._sim_dt)
            environment._after_task_physics_substep()
            robot_stability.sample(environment)
            current = _task_state(environment)
            authored_reset_pose_samples += int(
                bool(current["authored_reset_pose"].get("passed", False))
            )
            (
                card_support_force,
                card_support_penetration,
                card_support_valid,
            ) = _support_contact_reading(environment, "card_support")
            (
                clamp_support_force,
                clamp_support_penetration,
                clamp_support_valid,
            ) = _support_contact_reading(environment, "clamp_support")
            support_samples += 1
            support_valid = bool(card_support_valid and clamp_support_valid)
            support_readback_invalid_samples += int(not support_valid)
            supported_samples += int(
                support_valid
                and card_support_force >= 0.01
                and clamp_support_force >= 0.01
            )
            max_support_penetration = max(
                max_support_penetration,
                float(card_support_penetration),
                float(clamp_support_penetration),
            )
            frame_contact_penetration = max(
                float(card_support_penetration),
                float(clamp_support_penetration),
            )
            frame_contact_readback_valid = support_valid
            for label in environment._p004_contact_views:
                if label in {"card_support", "clamp_support"}:
                    continue
                try:
                    force, penetration = environment._p004_contact_reading(label)
                except Exception:
                    force, penetration = 0.0, math.inf
                contact_readback_valid = bool(
                    np.isfinite(force) and np.isfinite(penetration)
                )
                frame_contact_readback_valid = bool(
                    frame_contact_readback_valid
                    and contact_readback_valid
                )
                frame_contact_penetration = max(
                    frame_contact_penetration,
                    float(penetration) if contact_readback_valid else math.inf,
                )
            contact_readback_invalid_samples += int(
                not frame_contact_readback_valid
            )
            max_contact_penetration = max(
                max_contact_penetration,
                frame_contact_penetration,
            )
            nonfinite_samples += int(not current["finite"])
            task_velocity_valid = bool(
                current["finite"]
                and np.all(np.isfinite(current["card_velocity"]))
                and np.all(np.isfinite(current["clamp_velocity"]))
                and np.isfinite(float(current["clamp_joint_velocity"]))
            )
            task_velocity_invalid_samples += int(not task_velocity_valid)
            if current["finite"]:
                card_linear_speed = float(
                    np.linalg.norm(current["card_velocity"][:3])
                )
                card_angular_speed = float(
                    np.linalg.norm(current["card_velocity"][3:])
                )
                clamp_linear_speed = float(
                    np.linalg.norm(current["clamp_velocity"][:3])
                )
                clamp_angular_speed = float(
                    np.linalg.norm(current["clamp_velocity"][3:])
                )
                clamp_joint_speed = abs(float(current["clamp_joint_velocity"]))
                max_card_linear_speed = max(
                    max_card_linear_speed,
                    card_linear_speed,
                )
                max_card_angular_speed = max(
                    max_card_angular_speed,
                    card_angular_speed,
                )
                max_clamp_linear_speed = max(
                    max_clamp_linear_speed,
                    clamp_linear_speed,
                )
                max_clamp_angular_speed = max(
                    max_clamp_angular_speed,
                    clamp_angular_speed,
                )
                max_clamp_joint_speed = max(
                    max_clamp_joint_speed,
                    clamp_joint_speed,
                )
                max_card_translation_drift_m = max(
                    max_card_translation_drift_m,
                    float(
                        np.linalg.norm(
                            current["card"][:3, 3] - initial["card"][:3, 3]
                        )
                    ),
                )
                max_card_rotation_drift_rad = max(
                    max_card_rotation_drift_rad,
                    _rotation_error(current["card"], initial["card"]),
                )
                max_clamp_translation_drift_m = max(
                    max_clamp_translation_drift_m,
                    float(
                        np.linalg.norm(
                            current["clamp"][:3, 3]
                            - initial["clamp"][:3, 3]
                        )
                    ),
                )
                max_clamp_rotation_drift_rad = max(
                    max_clamp_rotation_drift_rad,
                    _rotation_error(current["clamp"], initial["clamp"]),
                )
                max_clamp_joint_drift_rad = max(
                    max_clamp_joint_drift_rad,
                    abs(
                        float(
                            current["clamp_joint"][0]
                            - initial["clamp_joint"][0]
                        )
                    ),
                )
            pose_increment_samples += 1
            pose_increment_valid = bool(
                current["finite"]
                and np.all(np.isfinite(previous_card_pose))
                and np.all(np.isfinite(previous_clamp_pose))
                and np.isfinite(previous_clamp_joint_position)
            )
            pose_increment_invalid_samples += int(not pose_increment_valid)
            if pose_increment_valid:
                frame_pose_step_translation_m = max(
                    float(
                        np.linalg.norm(
                            current["card"][:3, 3]
                            - previous_card_pose[:3, 3]
                        )
                    ),
                    float(
                        np.linalg.norm(
                            current["clamp"][:3, 3]
                            - previous_clamp_pose[:3, 3]
                        )
                    ),
                )
                frame_pose_step_rotation_rad = max(
                    _rotation_error(current["card"], previous_card_pose),
                    _rotation_error(current["clamp"], previous_clamp_pose),
                )
                frame_clamp_joint_step_rad = abs(
                    float(current["clamp_joint"][0])
                    - previous_clamp_joint_position
                )
                max_pose_step_translation_m = max(
                    max_pose_step_translation_m,
                    frame_pose_step_translation_m,
                )
                max_pose_step_rotation_rad = max(
                    max_pose_step_rotation_rad,
                    frame_pose_step_rotation_rad,
                )
                max_clamp_joint_step_rad = max(
                    max_clamp_joint_step_rad,
                    frame_clamp_joint_step_rad,
                )
                frame_static_pose = bool(
                    frame_pose_step_translation_m
                    <= STATIC_POSE_TRANSLATION_STEP_LIMIT_M
                    and frame_pose_step_rotation_rad
                    <= STATIC_POSE_ROTATION_STEP_LIMIT_RAD
                    and frame_clamp_joint_step_rad
                    <= STATIC_POSE_TRANSLATION_STEP_LIMIT_M
                )
                previous_card_pose = current["card"].copy()
                previous_clamp_pose = current["clamp"].copy()
                previous_clamp_joint_position = float(
                    current["clamp_joint"][0]
                )
            else:
                frame_pose_step_translation_m = math.inf
                frame_pose_step_rotation_rad = math.inf
                frame_clamp_joint_step_rad = math.inf
                frame_static_pose = False
            static_pose_covered_steps += int(frame_static_pose)
            if step % 12 == 0 or step == stability_steps - 1:
                stability_samples.append(
                    {
                        "physics_step": step + 1,
                        "finite": bool(current["finite"]),
                        "card_linear_speed_mps": (
                            card_linear_speed if current["finite"] else None
                        ),
                        "card_angular_speed_radps": (
                            card_angular_speed if current["finite"] else None
                        ),
                        "clamp_linear_speed_mps": (
                            clamp_linear_speed if current["finite"] else None
                        ),
                        "clamp_angular_speed_radps": (
                            clamp_angular_speed if current["finite"] else None
                        ),
                        "clamp_joint_speed_radps": (
                            clamp_joint_speed if current["finite"] else None
                        ),
                        "card_support_force_n": float(card_support_force),
                        "card_support_penetration_m": float(
                            card_support_penetration
                        ),
                        "clamp_support_force_n": float(clamp_support_force),
                        "clamp_support_penetration_m": float(
                            clamp_support_penetration
                        ),
                        "support_readback_valid": support_valid,
                        "max_contact_penetration_m": float(
                            frame_contact_penetration
                        ),
                        "contact_readback_valid": (
                            frame_contact_readback_valid
                        ),
                        "task_pose_step_translation_m": (
                            frame_pose_step_translation_m
                        ),
                        "task_pose_step_rotation_rad": (
                            frame_pose_step_rotation_rad
                        ),
                        "clamp_joint_step_rad": frame_clamp_joint_step_rad,
                        "static_pose_fallback_frame": frame_static_pose,
                    }
                )

        final = _task_state(environment)
        final_card_translation_drift_m = float(
            np.linalg.norm(final["card"][:3, 3] - initial["card"][:3, 3])
        )
        final_card_rotation_drift_rad = _rotation_error(
            final["card"], initial["card"]
        )
        final_clamp_translation_drift_m = float(
            np.linalg.norm(final["clamp"][:3, 3] - initial["clamp"][:3, 3])
        )
        final_clamp_rotation_drift_rad = _rotation_error(
            final["clamp"], initial["clamp"]
        )
        final_clamp_joint_drift_rad = abs(
            float(final["clamp_joint"][0] - initial["clamp_joint"][0])
        )
        final_task_velocity_exact_zero_observed = bool(
            final["finite"]
            and np.array_equal(final["card_velocity"], np.zeros(6))
            and np.array_equal(final["clamp_velocity"], np.zeros(6))
            and float(final["clamp_joint_velocity"]) == 0.0
        )
        (
            final_card_support_force,
            final_card_support_penetration,
            final_card_support_valid,
        ) = _support_contact_reading(environment, "card_support")
        (
            final_clamp_support_force,
            final_clamp_support_penetration,
            final_clamp_support_valid,
        ) = _support_contact_reading(environment, "clamp_support")
        reset_pass = bool(
            len(reset_rows) == int(args.reset_count)
            and int(args.reset_count) >= 20
            and all(
                row["finite"]
                and row["contacts_ready"]
                and not row["initial_failed"]
                and not row["initial_success"]
                and row["robot_initial_state_passed"]
                and row["card_linear_speed_mps"] <= 0.05
                and row["card_angular_speed_radps"] <= 0.75
                and row["clamp_linear_speed_mps"] <= 0.05
                and row["clamp_angular_speed_radps"] <= 0.75
                and row["clamp_joint_speed_radps"] <= 0.10
                and row["clamp_angle_rad"] >= -0.006
                and row["clamp_angle_rad"] <= 0.020
                and bool(row["authored_reset_pose"].get("passed", False))
                and row["card_support_force_n"] >= 0.01
                and row["clamp_support_force_n"] >= 0.01
                and row["card_support_readback_valid"]
                and row["clamp_support_readback_valid"]
                and row["card_support_penetration_m"] <= 0.0005
                and row["clamp_support_penetration_m"] <= 0.0005
                for row in reset_rows
            )
        )
        ordinary_velocity_pass = bool(
            task_velocity_invalid_samples == 0
            and max(max_card_linear_speed, max_clamp_linear_speed)
            <= PRIORITY_ZERO_LINEAR_SPEED_LIMIT_MPS
            and max(
                max_card_angular_speed,
                max_clamp_angular_speed,
                max_clamp_joint_speed,
            )
            <= PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RADPS
        )
        static_pose_fallback_pass = bool(
            pose_increment_samples == stability_steps
            and pose_increment_invalid_samples == 0
            and static_pose_covered_steps == stability_steps
            and max_pose_step_translation_m
            <= STATIC_POSE_TRANSLATION_STEP_LIMIT_M
            and max_pose_step_rotation_rad
            <= STATIC_POSE_ROTATION_STEP_LIMIT_RAD
            and max_clamp_joint_step_rad
            <= STATIC_POSE_TRANSLATION_STEP_LIMIT_M
        )
        velocity_or_static_pose_pass = bool(
            task_velocity_invalid_samples == 0
            and (ordinary_velocity_pass or static_pose_fallback_pass)
        )
        velocity_branch = (
            "ordinary_velocity"
            if ordinary_velocity_pass
            else "per_frame_static_pose_fallback"
            if static_pose_fallback_pass
            else "failed"
        )
        stability_pass = bool(
            stability_steps >= 10 * PHYSICS_HZ
            and support_samples == stability_steps
            and authored_reset_pose_samples == stability_steps
            and nonfinite_samples == 0
            and support_readback_invalid_samples == 0
            and contact_readback_invalid_samples == 0
            and final["finite"]
            and not final["failed"]
            and not final["success"]
            and task_velocity_invalid_samples == 0
            and pose_increment_samples == stability_steps
            and pose_increment_invalid_samples == 0
            and velocity_or_static_pose_pass
            and max_card_translation_drift_m <= 0.005
            and max_card_rotation_drift_rad <= math.radians(3.0)
            and max_clamp_translation_drift_m <= 0.005
            and max_clamp_rotation_drift_rad <= math.radians(3.0)
            and max_clamp_joint_drift_rad <= math.radians(1.0)
            and max_contact_penetration <= 0.0005
            and supported_samples == support_samples
            and final_card_support_valid
            and final_clamp_support_valid
            and final_card_support_force >= 0.01
            and final_clamp_support_force >= 0.01
            and max(
                max_support_penetration,
                final_card_support_penetration,
                final_clamp_support_penetration,
            )
            <= 0.0005
            and robot_stability.passed(
                physics_steps=stability_steps,
                physics_hz=PHYSICS_HZ,
            )
        )
        # ``moving_support`` was added when the moving jaw became a separately
        # audited dynamic body.  Keep the historical result key for consumers,
        # but validate the named contract instead of rejecting that legitimate
        # ninth view as an exact-count mismatch.
        required_contact_views = {
            "left_fixed",
            "left_moving",
            "right_card",
            "card_fixed",
            "card_moving",
            "moving_stage",
            "card_support",
            "clamp_support",
            "moving_support",
        }
        checks = {
            "exact_production_child": exact_production_child,
            "exact_production_runtime": runtime_contract["passed"],
            "dual_rokae_wuji1": robot_inventory
            == {
                "left": {"arm_joints": 7, "hand_joints": 20},
                "right": {"arm_joints": 7, "hand_joints": 20},
            },
            "eight_required_contact_views": required_contact_views.issubset(
                contact_inventory
            )
            and all(
                contact_inventory[name]["sensor_count"] >= 1
                and contact_inventory[name]["filter_count"] >= 1
                for name in required_contact_views
            )
            and support_pairs_exact,
            "twenty_strict_initial_states": reset_pass,
            "ten_second_zero_action_stability": stability_pass,
        }
        result.update(
            {
                "passed": all(checks.values()),
                "checks": checks,
                "robot_inventory": robot_inventory,
                "contact_inventory": contact_inventory,
                "support_pairs_exact": support_pairs_exact,
                "expected_support_pairs": expected_support_pairs,
                "reset_trials": reset_rows,
                "velocity_or_static_pose": {
                    "passed": velocity_or_static_pose_pass,
                    "selected_branch": velocity_branch,
                    "ordinary_velocity": {
                        "passed": ordinary_velocity_pass,
                        "invalid_sample_count": task_velocity_invalid_samples,
                        "peak_linear_speed_mps": max(
                            max_card_linear_speed,
                            max_clamp_linear_speed,
                        ),
                        "linear_speed_limit_mps": (
                            PRIORITY_ZERO_LINEAR_SPEED_LIMIT_MPS
                        ),
                        "peak_angular_speed_radps": max(
                            max_card_angular_speed,
                            max_clamp_angular_speed,
                            max_clamp_joint_speed,
                        ),
                        "angular_speed_limit_radps": (
                            PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RADPS
                        ),
                    },
                    "per_frame_static_pose_fallback": {
                        "passed": static_pose_fallback_pass,
                        "required_physics_steps": stability_steps,
                        "pose_increment_sample_count": pose_increment_samples,
                        "invalid_pose_increment_sample_count": (
                            pose_increment_invalid_samples
                        ),
                        "covered_physics_steps": static_pose_covered_steps,
                        "max_body_translation_step_m": (
                            max_pose_step_translation_m
                        ),
                        "body_translation_step_limit_m": (
                            STATIC_POSE_TRANSLATION_STEP_LIMIT_M
                        ),
                        "max_body_rotation_step_rad": (
                            max_pose_step_rotation_rad
                        ),
                        "body_rotation_step_limit_rad": (
                            STATIC_POSE_ROTATION_STEP_LIMIT_RAD
                        ),
                        "max_clamp_joint_step_rad": max_clamp_joint_step_rad,
                        "clamp_joint_step_limit_rad": (
                            STATIC_POSE_TRANSLATION_STEP_LIMIT_M
                        ),
                    },
                },
                "standalone_velocity_boundary": {
                    "initial_zero_velocity_write_performed": False,
                    "final_zero_velocity_write_performed": False,
                    "initial_task_velocity_exact_zero_observed": (
                        initial_task_velocity_exact_zero_observed
                    ),
                    "final_task_velocity_exact_zero_observed": (
                        final_task_velocity_exact_zero_observed
                    ),
                    "production_readiness_gap": (
                        "Standalone acceptance performs no initialization or "
                        "final exact-zero velocity boundary write; the production "
                        "recording-origin gate must still verify that boundary."
                    ),
                },
                "stability": {
                    "physics_steps": stability_steps,
                    "physics_hz": PHYSICS_HZ,
                    "simulated_seconds": stability_steps / PHYSICS_HZ,
                    "physics_sample_count": stability_steps,
                    "authored_reset_pose_sample_count": (
                        authored_reset_pose_samples
                    ),
                    "telemetry_sample_count": len(stability_samples),
                    "nonfinite_sample_count": nonfinite_samples,
                    "task_velocity_invalid_samples": (
                        task_velocity_invalid_samples
                    ),
                    "pose_increment_sample_count": pose_increment_samples,
                    "pose_increment_invalid_samples": (
                        pose_increment_invalid_samples
                    ),
                    "static_pose_covered_steps": static_pose_covered_steps,
                    "max_pose_step_translation_m": (
                        max_pose_step_translation_m
                    ),
                    "max_pose_step_rotation_rad": max_pose_step_rotation_rad,
                    "max_clamp_joint_step_rad": max_clamp_joint_step_rad,
                    "max_card_linear_speed_mps": max_card_linear_speed,
                    "max_card_angular_speed_radps": max_card_angular_speed,
                    "max_clamp_linear_speed_mps": max_clamp_linear_speed,
                    "max_clamp_angular_speed_radps": max_clamp_angular_speed,
                    "max_clamp_joint_speed_radps": max_clamp_joint_speed,
                    "max_card_translation_drift_m": (
                        max_card_translation_drift_m
                    ),
                    "max_card_rotation_drift_deg": math.degrees(
                        max_card_rotation_drift_rad
                    ),
                    "max_clamp_translation_drift_m": (
                        max_clamp_translation_drift_m
                    ),
                    "max_clamp_rotation_drift_deg": math.degrees(
                        max_clamp_rotation_drift_rad
                    ),
                    "max_clamp_joint_drift_rad": max_clamp_joint_drift_rad,
                    "final_card_translation_drift_m": (
                        final_card_translation_drift_m
                    ),
                    "final_card_rotation_drift_deg": math.degrees(
                        final_card_rotation_drift_rad
                    ),
                    "final_clamp_translation_drift_m": (
                        final_clamp_translation_drift_m
                    ),
                    "final_clamp_rotation_drift_deg": math.degrees(
                        final_clamp_rotation_drift_rad
                    ),
                    "final_clamp_joint_drift_rad": (
                        final_clamp_joint_drift_rad
                    ),
                    "support_samples": support_samples,
                    "supported_samples": supported_samples,
                    "support_readback_invalid_samples": (
                        support_readback_invalid_samples
                    ),
                    "contact_readback_invalid_samples": (
                        contact_readback_invalid_samples
                    ),
                    "supported_sample_fraction": (
                        supported_samples / max(1, support_samples)
                    ),
                    "final_card_support_force_n": float(
                        final_card_support_force
                    ),
                    "final_clamp_support_force_n": float(
                        final_clamp_support_force
                    ),
                    "max_support_penetration_m": float(
                        max(
                            max_support_penetration,
                            final_card_support_penetration,
                            final_clamp_support_penetration,
                        )
                    ),
                    "max_contact_penetration_m": float(
                        max_contact_penetration
                    ),
                    "robot": robot_stability.to_dict(),
                    "samples": stability_samples,
                },
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
        args.output.write_text(
            json.dumps(
                _json_safe(result),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        # This executable owns its bounded Kit process.  Isaac teardown can
        # either block or terminate with status zero, masking a failed gate.
        # Persist and flush the evidence, then exit with the truthful result.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
