#!/usr/bin/env python3
"""Live PhysX reset, multipose, reachability, and stability gate for P0-06."""

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


PRIORITY_ZERO_MAX_PENETRATION_M = 0.0005
PRIORITY_ZERO_LINEAR_SPEED_LIMIT_MPS = 0.001
PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RADPS = 0.002
STATIC_POSE_TRANSLATION_STEP_LIMIT_M = 1.0e-6
STATIC_POSE_ROTATION_STEP_LIMIT_RAD = 1.0e-5


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _readback_array(readback: Any) -> np.ndarray:
    values: list[float] = []
    for value in readback.__dict__.values():
        values.extend(np.asarray(value, dtype=np.float64).reshape(-1).tolist())
    return np.asarray(values, dtype=np.float64)


def _rotation_error(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left[:3, :3]).T @ np.asarray(right[:3, :3])
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def _quaternion_rotation_error(left_wxyz: Any, right_wxyz: Any) -> float:
    left = np.asarray(left_wxyz, dtype=np.float64).reshape(4)
    right = np.asarray(right_wxyz, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return math.inf
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1.0e-12 or right_norm <= 1.0e-12:
        return math.inf
    cosine_half_angle = float(
        np.clip(abs(np.dot(left / left_norm, right / right_norm)), 0.0, 1.0)
    )
    return 2.0 * math.acos(cosine_half_angle)


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


def _pose_xyzw(position: tuple[float, float, float], yaw_deg: float) -> np.ndarray:
    half = math.radians(float(yaw_deg)) * 0.5
    return np.asarray(
        [*position, 0.0, 0.0, math.sin(half), math.cos(half)],
        dtype=np.float64,
    )


def _set_body_pose(environment: Any, key: str, pose_xyzw: np.ndarray) -> None:
    view = environment._p006_body_views[key]
    current = view.get_transforms().clone()
    current[0] = environment._torch.as_tensor(
        pose_xyzw, dtype=current.dtype, device=current.device
    )
    indices = environment._torch.arange(
        int(view.count), dtype=environment._torch.long, device=current.device
    )
    view.set_transforms(current, indices)
    velocity = environment._torch.zeros_like(view.get_velocities())
    velocity_indices = indices.to(device=velocity.device)
    view.set_velocities(velocity, velocity_indices)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reset-trials", type=int, default=20)
    parser.add_argument("--stability-seconds", type=float, default=10.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False
    args.device = str(getattr(args, "device", None) or "cuda:0")
    os.environ["HANDSON_ISAACLAB_ARM_IK_BACKEND"] = "haorui"
    launcher = AppLauncher(args)
    app = launcher.app
    environment = None
    result: dict[str, Any] = {
        "schema": "handson.p006.live_physics_acceptance.v1",
        "passed": False,
    }
    rc = 1
    try:
        from handson.sim.isaaclab.abc_collection_env import _pose_vector_to_matrix
        from handson.sim.isaaclab.p006_collection_env import (
            HandsOnP006IsaacLabEnv,
        )
        from handson.sim.isaaclab.priority_zero_robot_audit import (
            RobotStabilityAudit,
            initial_robot_state_report,
            priority_zero_runtime_contract,
        )
        from handson.sim.isaaclab.p006_contract import (
            CONTROL_HZ,
            FEMALE_ROOT_PATH,
            MAX_PENETRATION_M,
            MALE_ROOT_PATH,
            PHYSICS_HZ,
            TASK_NAME,
            reset_is_valid,
            sample_reset_state,
        )

        environment = HandsOnP006IsaacLabEnv(
            seed=0,
            control_dt=1.0 / CONTROL_HZ,
            physics_dt=1.0 / PHYSICS_HZ,
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
        sim = environment._sim
        if sim is None:
            raise RuntimeError("P0-06 simulation context is unavailable")
        runtime_contract = priority_zero_runtime_contract(
            environment,
            requested_device=args.device,
            physics_hz=PHYSICS_HZ,
            control_hz=CONTROL_HZ,
        )
        result["runtime"] = runtime_contract
        exact_production_child = bool(
            environment.__class__ is HandsOnP006IsaacLabEnv
            and environment.task_name == TASK_NAME
            and runtime_contract["passed"]
        )

        contact_inventory = {
            key: {
                "sensor_count": int(view.sensor_count),
                "filter_count": int(view.filter_count),
            }
            for key, view in environment._p006_contact_views.items()
        }
        table_prim = environment._find_rigid_body_prim("table")
        expected_female_support_pair = (
            None
            if table_prim is None
            else (FEMALE_ROOT_PATH, (str(table_prim.GetPath()),))
        )
        expected_male_support_pair = (
            None
            if table_prim is None
            else (MALE_ROOT_PATH, (str(table_prim.GetPath()),))
        )
        support_pairs_exact = bool(
            expected_female_support_pair is not None
            and expected_male_support_pair is not None
            and environment._p006_contact_paths.get("female_support")
            == expected_female_support_pair
            and environment._p006_contact_paths.get("male_support")
            == expected_male_support_pair
            and contact_inventory.get("female_support")
            == {"sensor_count": 1, "filter_count": 1}
            and contact_inventory.get("male_support")
            == {"sensor_count": 1, "filter_count": 1}
        )
        robot_inventory = {
            side: {
                "arm_joints": len(environment._arm_joint_ids[side]),
                "hand_joints": len(environment._hand_joint_ids[side]),
            }
            for side in ("left", "right")
        }

        reset_rows: list[dict[str, Any]] = []
        for seed in range(int(args.reset_trials)):
            _observation, info = environment.reset(seed=seed)
            robot_initial_passed, robot_initial_state = (
                initial_robot_state_report(environment)
            )
            readback = environment._physical_readback()
            female_support_force, female_support_penetration = (
                environment._contact_reading("female_support")
            )
            male_support_force, male_support_penetration = (
                environment._contact_reading("male_support")
            )
            metrics = environment._p006_latch.update(readback)
            expected = sample_reset_state(seed)
            authored_reset_pose = environment._authored_reset_pose_report(readback)
            reset_rows.append(
                {
                    "seed": seed,
                    "sample": expected.to_dict(),
                    "sample_valid": reset_is_valid(expected),
                    "finite": bool(np.isfinite(_readback_array(readback)).all()),
                    "initial_success": bool(info.get("succeed", False)),
                    "initial_failed": bool(info.get("failed", False)),
                    "robot_initial_state_passed": robot_initial_passed,
                    "robot_initial_state": robot_initial_state,
                    "max_penetration_m": float(readback.max_penetration_m),
                    "female_position_m": list(readback.female_position_m),
                    "male_position_m": list(readback.male_position_m),
                    "axis_error_deg": float(metrics["axis_error_deg"]),
                    "female_support_force_n": float(female_support_force),
                    "female_support_penetration_m": float(
                        female_support_penetration
                    ),
                    "male_support_force_n": float(male_support_force),
                    "male_support_penetration_m": float(
                        male_support_penetration
                    ),
                    "female_linear_speed_mps": float(
                        np.linalg.norm(readback.female_linear_velocity_mps)
                    ),
                    "female_angular_speed_radps": float(
                        np.linalg.norm(readback.female_angular_velocity_radps)
                    ),
                    "male_linear_speed_mps": float(
                        np.linalg.norm(readback.male_linear_velocity_mps)
                    ),
                    "male_angular_speed_radps": float(
                        np.linalg.norm(readback.male_angular_velocity_radps)
                    ),
                    "root_xy_error_m": max(
                        float(
                            np.linalg.norm(
                                np.asarray(readback.female_position_m[:2])
                                - np.asarray(expected.female_position_m[:2])
                            )
                        ),
                        float(
                            np.linalg.norm(
                                np.asarray(readback.male_position_m[:2])
                                - np.asarray(expected.male_position_m[:2])
                            )
                        ),
                    ),
                    "authored_reset_pose": authored_reset_pose,
                }
            )

        reachability: list[dict[str, Any]] = []
        reachable_quat = np.asarray(
            [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], dtype=np.float64
        )
        for seed in range(int(args.reset_trials)):
            reset = sample_reset_state(seed)
            targets = {
                "right_female_rear_loop": (
                    "right",
                    np.asarray(
                        [
                            reset.female_position_m[0] - 0.115,
                            reset.female_position_m[1] - 0.030,
                            0.970,
                            *reachable_quat,
                        ],
                        dtype=np.float64,
                    ),
                ),
                "left_male_body": (
                    "left",
                    np.asarray(
                        [
                            reset.male_position_m[0] - 0.115,
                            reset.male_position_m[1] + 0.015,
                            0.960,
                            *reachable_quat,
                        ],
                        dtype=np.float64,
                    ),
                ),
            }
            for role, (side, target) in targets.items():
                robot = environment._robots[side]
                q = environment._compute_arm_joint_target_haorui(side, robot, target)
                row: dict[str, Any] = {
                    "seed": seed,
                    "role": role,
                    "side": side,
                    "target_pose_wxyz": target.tolist(),
                    "solved": q is not None,
                }
                if q is not None:
                    q_np = _array(q).reshape(-1)
                    arm_ids = environment._arm_joint_ids[side]
                    limits = _array(robot.data.soft_joint_pos_limits)[0, arm_ids]
                    margin = float(
                        np.min(
                            np.minimum(
                                q_np - limits[:, 0], limits[:, 1] - q_np
                            )
                        )
                    )
                    base_from_world = np.linalg.inv(
                        environment._robot_base_pose_matrix(side, robot)
                    )
                    current_ee_in_base = (
                        base_from_world
                        @ environment._robot_body_pose_matrix(
                            robot, environment._ee_body_indices[side]
                        )
                    )
                    model = environment._haorui_ik_models[side]
                    q_now = environment._arm_reference_q(side, robot)
                    current_tcp_in_base = model.fk(q_now)
                    tcp_to_ee = np.linalg.inv(current_tcp_in_base) @ current_ee_in_base
                    predicted_ee_world = np.linalg.inv(base_from_world) @ (
                        model.fk(q_np) @ tcp_to_ee
                    )
                    target_matrix = _pose_vector_to_matrix(target)
                    row.update(
                        {
                            "joint_position_rad": q_np.tolist(),
                            "joint_limit_margin_rad": margin,
                            "position_error_m": float(
                                np.linalg.norm(
                                    predicted_ee_world[:3, 3]
                                    - target_matrix[:3, 3]
                                )
                            ),
                            "orientation_error_rad": _rotation_error(
                                predicted_ee_world, target_matrix
                            ),
                        }
                    )
                reachability.append(row)

        multipose_specs = (
            ("separated", 0.110, 0.0036, 0.0),
            ("first_contact", 0.0876, 0.0036, 0.0),
            ("insert_18mm", 0.0696, 0.0036, 0.0),
            ("insert_22_6mm", 0.0650, 0.0036, 0.0),
            ("insert_contact_low", 0.0650, 0.0016, 0.0),
            ("insert_yaw_plus_8", 0.0650, 0.0036, 8.0),
            ("insert_yaw_minus_8", 0.0650, 0.0036, -8.0),
        )
        multipose_rows: list[dict[str, Any]] = []
        for name, separation, male_dz, male_yaw_delta in multipose_specs:
            female_pose = _pose_xyzw((0.20, -0.035, 1.20), 90.0)
            male_pose = _pose_xyzw(
                (0.20, -0.035 + separation, 1.20 + male_dz),
                90.0 + male_yaw_delta,
            )
            _set_body_pose(environment, "female", female_pose)
            _set_body_pose(environment, "male", male_pose)
            for robot in environment._robots.values():
                robot.write_data_to_sim()
            sim.step(render=False)
            for robot in environment._robots.values():
                robot.update(environment._sim_dt)
            readback = environment._physical_readback()
            buckle_force, buckle_penetration = environment._contact_reading(
                "buckle_body"
            )
            multipose_rows.append(
                {
                    "name": name,
                    "requested_separation_m": separation,
                    "requested_male_dz_m": male_dz,
                    "requested_yaw_delta_deg": male_yaw_delta,
                    "finite": bool(np.isfinite(_readback_array(readback)).all()),
                    "body_contact_force_n": float(buckle_force),
                    "max_buckle_penetration_m": float(buckle_penetration),
                    "female_position_m": list(readback.female_position_m),
                    "male_position_m": list(readback.male_position_m),
                }
            )

        environment.reset(seed=0)
        robot_stability = RobotStabilityAudit.start(environment)
        initial_stability_readback = environment._physical_readback()
        previous_female_position = np.asarray(
            initial_stability_readback.female_position_m, dtype=np.float64
        )
        previous_male_position = np.asarray(
            initial_stability_readback.male_position_m, dtype=np.float64
        )
        previous_female_quaternion = np.asarray(
            initial_stability_readback.female_quaternion_wxyz,
            dtype=np.float64,
        )
        previous_male_quaternion = np.asarray(
            initial_stability_readback.male_quaternion_wxyz,
            dtype=np.float64,
        )
        stability_steps = int(round(float(args.stability_seconds) * PHYSICS_HZ))
        stability_samples: list[dict[str, Any]] = []
        nan_count = 0
        explosive_count = 0
        max_penetration = 0.0
        max_forbidden_force = 0.0
        peak_linear_speed = 0.0
        peak_angular_speed = 0.0
        support_samples = 0
        supported_samples = 0
        support_nonfinite_samples = 0
        max_support_penetration = 0.0
        pose_invalid_samples = 0
        max_female_translation_drift_m = 0.0
        max_male_translation_drift_m = 0.0
        max_female_rotation_drift_rad = 0.0
        max_male_rotation_drift_rad = 0.0
        velocity_invalid_samples = 0
        pose_increment_samples = 0
        pose_increment_invalid_samples = 0
        static_pose_covered_steps = 0
        max_pose_step_translation_m = 0.0
        max_pose_step_rotation_rad = 0.0
        authored_reset_pose_samples = 0
        for step in range(stability_steps):
            for robot in environment._robots.values():
                robot.write_data_to_sim()
            sim.step(render=False)
            for robot in environment._robots.values():
                robot.update(environment._sim_dt)
            environment._after_task_physics_substep()
            robot_stability.sample(environment)
            readback = environment._physical_readback()
            authored_reset_pose_samples += int(
                bool(
                    environment._authored_reset_pose_report(readback).get(
                        "passed", False
                    )
                )
            )
            values = _readback_array(readback)
            nan_count += int(values.size - np.isfinite(values).sum())
            linear_speed = max(
                float(np.linalg.norm(readback.female_linear_velocity_mps)),
                float(np.linalg.norm(readback.male_linear_velocity_mps)),
            )
            angular_speed = max(
                float(np.linalg.norm(readback.female_angular_velocity_radps)),
                float(np.linalg.norm(readback.male_angular_velocity_radps)),
            )
            velocity_valid = bool(
                np.isfinite(linear_speed) and np.isfinite(angular_speed)
            )
            velocity_invalid_samples += int(not velocity_valid)
            if velocity_valid:
                peak_linear_speed = max(peak_linear_speed, linear_speed)
                peak_angular_speed = max(peak_angular_speed, angular_speed)
            max_penetration = max(
                max_penetration, float(readback.max_penetration_m)
            )
            max_forbidden_force = max(
                max_forbidden_force,
                float(readback.robot_table_contact_force_n),
            )
            female_force, female_penetration = environment._contact_reading(
                "female_support"
            )
            male_force, male_penetration = environment._contact_reading(
                "male_support"
            )
            support_finite = bool(
                np.isfinite(female_force)
                and np.isfinite(female_penetration)
                and np.isfinite(male_force)
                and np.isfinite(male_penetration)
            )
            support_samples += 1
            support_nonfinite_samples += int(not support_finite)
            supported_samples += int(
                support_finite
                and female_force >= 0.01
                and male_force >= 0.01
            )
            max_support_penetration = max(
                max_support_penetration,
                float(female_penetration),
                float(male_penetration),
            )
            if linear_speed > 3.0 or angular_speed > 15.0:
                explosive_count += 1
            female_position = np.asarray(
                readback.female_position_m, dtype=np.float64
            )
            male_position = np.asarray(readback.male_position_m, dtype=np.float64)
            female_quaternion = np.asarray(
                readback.female_quaternion_wxyz, dtype=np.float64
            )
            male_quaternion = np.asarray(
                readback.male_quaternion_wxyz, dtype=np.float64
            )
            pose_valid = bool(
                np.all(np.isfinite(female_position))
                and np.all(np.isfinite(male_position))
                and np.all(np.isfinite(female_quaternion))
                and np.all(np.isfinite(male_quaternion))
                and np.all(np.isfinite(previous_female_position))
                and np.all(np.isfinite(previous_male_position))
                and np.all(np.isfinite(previous_female_quaternion))
                and np.all(np.isfinite(previous_male_quaternion))
                and np.linalg.norm(female_quaternion) > 1.0e-12
                and np.linalg.norm(male_quaternion) > 1.0e-12
                and np.linalg.norm(previous_female_quaternion) > 1.0e-12
                and np.linalg.norm(previous_male_quaternion) > 1.0e-12
            )
            pose_increment_samples += 1
            pose_invalid_samples += int(not pose_valid)
            pose_increment_invalid_samples += int(not pose_valid)
            if pose_valid:
                current_female_translation_drift_m = float(
                    np.linalg.norm(
                        female_position
                        - np.asarray(initial_stability_readback.female_position_m)
                    )
                )
                current_male_translation_drift_m = float(
                    np.linalg.norm(
                        male_position
                        - np.asarray(initial_stability_readback.male_position_m)
                    )
                )
                current_female_rotation_drift_rad = _quaternion_rotation_error(
                    female_quaternion,
                    initial_stability_readback.female_quaternion_wxyz,
                )
                current_male_rotation_drift_rad = _quaternion_rotation_error(
                    male_quaternion,
                    initial_stability_readback.male_quaternion_wxyz,
                )
                max_female_translation_drift_m = max(
                    max_female_translation_drift_m,
                    current_female_translation_drift_m,
                )
                max_male_translation_drift_m = max(
                    max_male_translation_drift_m,
                    current_male_translation_drift_m,
                )
                max_female_rotation_drift_rad = max(
                    max_female_rotation_drift_rad,
                    current_female_rotation_drift_rad,
                )
                max_male_rotation_drift_rad = max(
                    max_male_rotation_drift_rad,
                    current_male_rotation_drift_rad,
                )
                frame_pose_step_translation_m = max(
                    float(
                        np.linalg.norm(
                            female_position - previous_female_position
                        )
                    ),
                    float(np.linalg.norm(male_position - previous_male_position)),
                )
                frame_pose_step_rotation_rad = max(
                    _quaternion_rotation_error(
                        female_quaternion, previous_female_quaternion
                    ),
                    _quaternion_rotation_error(
                        male_quaternion, previous_male_quaternion
                    ),
                )
                max_pose_step_translation_m = max(
                    max_pose_step_translation_m,
                    frame_pose_step_translation_m,
                )
                max_pose_step_rotation_rad = max(
                    max_pose_step_rotation_rad,
                    frame_pose_step_rotation_rad,
                )
            else:
                current_female_translation_drift_m = math.inf
                current_male_translation_drift_m = math.inf
                current_female_rotation_drift_rad = math.inf
                current_male_rotation_drift_rad = math.inf
                frame_pose_step_translation_m = math.inf
                frame_pose_step_rotation_rad = math.inf
            frame_static_pose = bool(
                pose_valid
                and frame_pose_step_translation_m
                <= STATIC_POSE_TRANSLATION_STEP_LIMIT_M
                and frame_pose_step_rotation_rad
                <= STATIC_POSE_ROTATION_STEP_LIMIT_RAD
            )
            static_pose_covered_steps += int(frame_static_pose)
            previous_female_position = female_position.copy()
            previous_male_position = male_position.copy()
            previous_female_quaternion = female_quaternion.copy()
            previous_male_quaternion = male_quaternion.copy()
            if step % 12 == 0 or step == stability_steps - 1:
                stability_samples.append(
                    {
                        "physics_step": step + 1,
                        "finite": bool(np.isfinite(values).all()),
                        "linear_speed_mps": linear_speed,
                        "angular_speed_radps": angular_speed,
                        "max_penetration_m": float(readback.max_penetration_m),
                        "robot_table_contact_force_n": float(
                            readback.robot_table_contact_force_n
                        ),
                        "female_support_force_n": float(female_force),
                        "female_support_penetration_m": float(
                            female_penetration
                        ),
                        "male_support_force_n": float(male_force),
                        "male_support_penetration_m": float(male_penetration),
                        "female_translation_drift_m": (
                            current_female_translation_drift_m
                        ),
                        "male_translation_drift_m": (
                            current_male_translation_drift_m
                        ),
                        "female_rotation_drift_deg": math.degrees(
                            current_female_rotation_drift_rad
                        ),
                        "male_rotation_drift_deg": math.degrees(
                            current_male_rotation_drift_rad
                        ),
                        "pose_step_translation_m": (
                            frame_pose_step_translation_m
                        ),
                        "pose_step_rotation_rad": frame_pose_step_rotation_rad,
                        "static_pose_fallback_frame": frame_static_pose,
                    }
                )

        ordinary_velocity_pass = bool(
            velocity_invalid_samples == 0
            and peak_linear_speed <= PRIORITY_ZERO_LINEAR_SPEED_LIMIT_MPS
            and peak_angular_speed <= PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RADPS
        )
        static_pose_fallback_pass = bool(
            pose_increment_samples == stability_steps
            and pose_increment_invalid_samples == 0
            and static_pose_covered_steps == stability_steps
            and max_pose_step_translation_m
            <= STATIC_POSE_TRANSLATION_STEP_LIMIT_M
            and max_pose_step_rotation_rad
            <= STATIC_POSE_ROTATION_STEP_LIMIT_RAD
        )
        velocity_or_static_pose_pass = bool(
            velocity_invalid_samples == 0
            and (ordinary_velocity_pass or static_pose_fallback_pass)
        )
        velocity_branch = (
            "ordinary_velocity"
            if ordinary_velocity_pass
            else "per_frame_static_pose_fallback"
            if velocity_or_static_pose_pass and static_pose_fallback_pass
            else "failed"
        )
        final_stability_readback = environment._physical_readback()
        final_female_translation_drift_m = float(
            np.linalg.norm(
                np.asarray(final_stability_readback.female_position_m)
                - np.asarray(initial_stability_readback.female_position_m)
            )
        )
        final_male_translation_drift_m = float(
            np.linalg.norm(
                np.asarray(final_stability_readback.male_position_m)
                - np.asarray(initial_stability_readback.male_position_m)
            )
        )
        final_female_rotation_drift_rad = _quaternion_rotation_error(
            final_stability_readback.female_quaternion_wxyz,
            initial_stability_readback.female_quaternion_wxyz,
        )
        final_male_rotation_drift_rad = _quaternion_rotation_error(
            final_stability_readback.male_quaternion_wxyz,
            initial_stability_readback.male_quaternion_wxyz,
        )
        final_linear_speed = max(
            float(
                np.linalg.norm(
                    final_stability_readback.female_linear_velocity_mps
                )
            ),
            float(
                np.linalg.norm(final_stability_readback.male_linear_velocity_mps)
            ),
        )
        final_angular_speed = max(
            float(
                np.linalg.norm(
                    final_stability_readback.female_angular_velocity_radps
                )
            ),
            float(
                np.linalg.norm(
                    final_stability_readback.male_angular_velocity_radps
                )
            ),
        )
        initial_task_velocity_exact_zero_observed = bool(
            all(
                value == 0.0
                for value in (
                    *initial_stability_readback.female_linear_velocity_mps,
                    *initial_stability_readback.female_angular_velocity_radps,
                    *initial_stability_readback.male_linear_velocity_mps,
                    *initial_stability_readback.male_angular_velocity_radps,
                )
            )
        )
        final_task_velocity_exact_zero_observed = bool(
            all(
                value == 0.0
                for value in (
                    *final_stability_readback.female_linear_velocity_mps,
                    *final_stability_readback.female_angular_velocity_radps,
                    *final_stability_readback.male_linear_velocity_mps,
                    *final_stability_readback.male_angular_velocity_radps,
                )
            )
        )
        final_female_support_force, final_female_support_penetration = (
            environment._contact_reading("female_support")
        )
        final_male_support_force, final_male_support_penetration = (
            environment._contact_reading("male_support")
        )

        reset_pass = bool(
            len(reset_rows) == int(args.reset_trials)
            and int(args.reset_trials) >= 20
            and all(
                row["sample_valid"]
                and row["finite"]
                and not row["initial_success"]
                and not row["initial_failed"]
                and row["robot_initial_state_passed"]
                and row["max_penetration_m"]
                <= PRIORITY_ZERO_MAX_PENETRATION_M
                and row["root_xy_error_m"] <= 0.003
                and row["axis_error_deg"] <= 8.1
                and bool(row["authored_reset_pose"].get("passed", False))
                and row["female_support_force_n"] >= 0.01
                and row["male_support_force_n"] >= 0.01
                and row["female_support_penetration_m"] <= 0.0005
                and row["male_support_penetration_m"] <= 0.0005
                and row["female_linear_speed_mps"] <= 0.05
                and row["female_angular_speed_radps"] <= 0.75
                and row["male_linear_speed_mps"] <= 0.05
                and row["male_angular_speed_radps"] <= 0.75
                for row in reset_rows
            )
        )
        reachability_pass = bool(
            len(reachability) == 2 * int(args.reset_trials)
            and int(args.reset_trials) >= 20
            and all(
                row["solved"]
                and row.get("joint_limit_margin_rad", -math.inf) >= 0.05
                and row.get("position_error_m", math.inf) <= 0.01
                and row.get("orientation_error_rad", math.inf) <= 0.15
                for row in reachability
            )
        )
        multipose_pass = bool(
            len(multipose_rows) == len(multipose_specs)
            and all(
                row["finite"]
                and row["max_buckle_penetration_m"] <= MAX_PENETRATION_M
                for row in multipose_rows
            )
        )
        stability_pass = bool(
            stability_steps >= 10 * PHYSICS_HZ
            and nan_count == 0
            and explosive_count == 0
            and max_penetration <= PRIORITY_ZERO_MAX_PENETRATION_M
            and max_forbidden_force < 5.0
            and velocity_invalid_samples == 0
            and pose_increment_samples == stability_steps
            and authored_reset_pose_samples == stability_steps
            and pose_increment_invalid_samples == 0
            and velocity_or_static_pose_pass
            and pose_invalid_samples == 0
            and max_female_translation_drift_m <= 0.005
            and max_male_translation_drift_m <= 0.005
            and max_female_rotation_drift_rad <= math.radians(3.0)
            and max_male_rotation_drift_rad <= math.radians(3.0)
            and support_samples == stability_steps
            and support_nonfinite_samples == 0
            and supported_samples == support_samples
            and final_female_support_force >= 0.01
            and final_male_support_force >= 0.01
            and max(
                max_support_penetration,
                final_female_support_penetration,
                final_male_support_penetration,
            )
            <= 0.0005
            and robot_stability.passed(
                physics_steps=stability_steps,
                physics_hz=PHYSICS_HZ,
            )
            and not environment._p006_failure_latched
        )
        checks = {
            "exact_production_child": exact_production_child,
            "exact_production_runtime": runtime_contract["passed"],
            "dual_rokae_wuji": robot_inventory
            == {
                "left": {"arm_joints": 7, "hand_joints": 20},
                "right": {"arm_joints": 7, "hand_joints": 20},
            },
            "exact_contact_views": len(contact_inventory) == 6
            and all(row["sensor_count"] == 1 for row in contact_inventory.values())
            and support_pairs_exact,
            "twenty_resets": reset_pass,
            "forty_role_targets_reachable": reachability_pass,
            "seven_multipose_geometry_cases": multipose_pass,
            "velocity_or_static_pose": velocity_or_static_pose_pass,
            "ten_second_stability": stability_pass,
        }
        result.update(
            {
                "passed": all(checks.values()),
                "checks": checks,
                "robot_inventory": robot_inventory,
                "contact_inventory": contact_inventory,
                "support_pairs_exact": support_pairs_exact,
                "expected_support_pairs": {
                    "female_support": expected_female_support_pair,
                    "male_support": expected_male_support_pair,
                },
                "reset_trials": reset_rows,
                "reachability": reachability,
                "multipose": multipose_rows,
                "velocity_or_static_pose": {
                    "passed": velocity_or_static_pose_pass,
                    "selected_branch": velocity_branch,
                    "ordinary_velocity": {
                        "passed": ordinary_velocity_pass,
                        "invalid_sample_count": velocity_invalid_samples,
                        "peak_linear_speed_mps": peak_linear_speed,
                        "linear_speed_limit_mps": (
                            PRIORITY_ZERO_LINEAR_SPEED_LIMIT_MPS
                        ),
                        "peak_angular_speed_radps": peak_angular_speed,
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
                        "max_body_rotation_step_rad": max_pose_step_rotation_rad,
                        "body_rotation_step_limit_rad": (
                            STATIC_POSE_ROTATION_STEP_LIMIT_RAD
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
                    "nan_count": nan_count,
                    "explosive_count": explosive_count,
                    "peak_linear_speed_mps": peak_linear_speed,
                    "peak_angular_speed_radps": peak_angular_speed,
                    "velocity_invalid_samples": velocity_invalid_samples,
                    "final_linear_speed_mps": final_linear_speed,
                    "final_angular_speed_radps": final_angular_speed,
                    "pose_increment_sample_count": pose_increment_samples,
                    "pose_increment_invalid_samples": (
                        pose_increment_invalid_samples
                    ),
                    "static_pose_covered_steps": static_pose_covered_steps,
                    "max_pose_step_translation_m": (
                        max_pose_step_translation_m
                    ),
                    "max_pose_step_rotation_rad": max_pose_step_rotation_rad,
                    "pose_invalid_samples": pose_invalid_samples,
                    "max_female_translation_drift_m": (
                        max_female_translation_drift_m
                    ),
                    "max_male_translation_drift_m": (
                        max_male_translation_drift_m
                    ),
                    "max_female_rotation_drift_deg": math.degrees(
                        max_female_rotation_drift_rad
                    ),
                    "max_male_rotation_drift_deg": math.degrees(
                        max_male_rotation_drift_rad
                    ),
                    "final_female_translation_drift_m": (
                        final_female_translation_drift_m
                    ),
                    "final_male_translation_drift_m": (
                        final_male_translation_drift_m
                    ),
                    "final_female_rotation_drift_deg": math.degrees(
                        final_female_rotation_drift_rad
                    ),
                    "final_male_rotation_drift_deg": math.degrees(
                        final_male_rotation_drift_rad
                    ),
                    "support_samples": support_samples,
                    "authored_reset_pose_samples": authored_reset_pose_samples,
                    "supported_samples": supported_samples,
                    "support_nonfinite_samples": support_nonfinite_samples,
                    "telemetry_sample_count": len(stability_samples),
                    "supported_sample_fraction": (
                        supported_samples / max(1, support_samples)
                    ),
                    "final_female_support_force_n": float(
                        final_female_support_force
                    ),
                    "final_male_support_force_n": float(
                        final_male_support_force
                    ),
                    "max_support_penetration_m": float(
                        max(
                            max_support_penetration,
                            final_female_support_penetration,
                            final_male_support_penetration,
                        )
                    ),
                    "max_penetration_m": max_penetration,
                    "max_penetration_limit_m": (
                        PRIORITY_ZERO_MAX_PENETRATION_M
                    ),
                    "max_forbidden_force_n": max_forbidden_force,
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
            json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # The validator owns this bounded Kit process.  Avoid teardown paths
        # that can hang or replace a failed gate with process status zero.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
