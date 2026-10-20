#!/usr/bin/env python3
"""Bounded live Isaac Sim 5.1 / PhysX acceptance for P0-05."""

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reset-count", type=int, default=20)
    parser.add_argument("--stability-seconds", type=float, default=10.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


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


def main() -> int:
    args = _parser().parse_args()
    args.headless = True
    args.enable_cameras = False
    args.device = str(getattr(args, "device", None) or "cuda:0")
    os.environ["HANDSON_ISAACLAB_ARM_IK_BACKEND"] = "haorui"
    launcher = AppLauncher(args)
    app = launcher.app
    environment = None
    result: dict[str, Any] = {
        "schema": "handson.p005.live_physics_acceptance.v1",
        "passed": False,
        "device": args.device,
    }
    rc = 1
    try:
        import torch

        from handson.sim.isaaclab.p005_collection_env import (
            HandsOnP005IsaacLabEnv,
        )
        from handson.sim.isaaclab.priority_zero_robot_audit import (
            RobotStabilityAudit,
            initial_robot_state_report,
            priority_zero_runtime_contract,
        )
        from handson.sim.isaaclab.p005_contract import (
            CONTROL_HZ,
            EXPECTED_DOF_NAMES,
            KEY_BODY_NAME,
            LOCK_CYLINDER_DOF,
            PADLOCK_BODY_NAME,
            PHYSICS_HZ,
            SHACKLE_BODY_NAME,
            SHACKLE_HINGE_DOF,
            SHACKLE_PREUNLOCK_MAX_M,
            SHACKLE_SLIDE_DOF,
            TASK_NAME,
        )

        environment = HandsOnP005IsaacLabEnv(
            seed=0,
            control_dt=1.0 / CONTROL_HZ,
            physics_dt=1.0 / PHYSICS_HZ,
            time_limit=50.0,
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
            raise RuntimeError("P0-05 complete scene did not expose PhysX runtime")
        runtime_contract = priority_zero_runtime_contract(
            environment,
            requested_device=args.device,
            physics_hz=PHYSICS_HZ,
            control_hz=CONTROL_HZ,
        )
        result["runtime"] = runtime_contract
        exact_production_child = bool(
            environment.__class__ is HandsOnP005IsaacLabEnv
            and environment.task_name == TASK_NAME
            and runtime_contract["passed"]
        )

        robot_inventory = {
            "names": sorted(environment._robots),
            "arm_joint_counts": {
                side: len(environment._arm_joint_ids[side])
                for side in ("left", "right")
            },
            "hand_joint_counts": {
                side: len(environment._hand_joint_ids[side])
                for side in ("left", "right")
            },
        }
        contact_inventory = {
            key: {
                "sensor_count": int(sensor.sensor_count),
                "filter_count": int(sensor.filter_count),
            }
            for key, sensor in environment._p005_contact_views.items()
        }
        key_prim = environment._find_rigid_body_prim(KEY_BODY_NAME)
        table_prim = environment._find_rigid_body_prim("table")
        expected_key_support_pair = (
            None
            if key_prim is None or table_prim is None
            else (
                str(key_prim.GetPath()),
                (str(table_prim.GetPath()),),
            )
        )
        key_support_pair_exact = bool(
            expected_key_support_pair is not None
            and environment._p005_contact_paths.get("key_support")
            == expected_key_support_pair
            and contact_inventory.get("key_support")
            == {"sensor_count": 1, "filter_count": 1}
        )
        task_body_names = (KEY_BODY_NAME, PADLOCK_BODY_NAME, SHACKLE_BODY_NAME)

        reset_trials: list[dict[str, Any]] = []
        zero_action = np.zeros(environment.action_space.shape, dtype=np.float32)
        for seed in range(int(args.reset_count)):
            _obs, reset_info = environment.reset(seed=seed)
            robot_initial_passed, robot_initial_state = (
                initial_robot_state_report(environment)
            )
            key_support_force, key_support_penetration = (
                environment._contact_reading("key_support")
            )
            raw_readback = environment._physical_readback()
            raw_numeric = np.asarray(
                [float(value) for value in raw_readback.__dict__.values()],
                dtype=np.float64,
            )
            raw_task_body_velocities = {
                name: environment._body_velocity(name)
                for name in task_body_names
            }
            raw_task_dof_velocities = {
                name: float(environment._dof_reading(name)[1])
                for name in EXPECTED_DOF_NAMES
            }
            raw_task_dof_positions = {
                name: float(environment._dof_reading(name)[0])
                for name in EXPECTED_DOF_NAMES
            }
            raw_task_velocity_finite = bool(
                all(
                    velocity.shape == (6,) and np.all(np.isfinite(velocity))
                    for velocity in raw_task_body_velocities.values()
                )
                and np.isfinite(list(raw_task_dof_velocities.values())).all()
            )
            raw_task_position_finite = bool(
                np.isfinite(list(raw_task_dof_positions.values())).all()
            )
            authored_reset_pose = environment._authored_reset_pose_report()
            raw_task_linear_speed_mps = (
                max(
                    *(
                        float(np.linalg.norm(velocity[:3]))
                        for velocity in raw_task_body_velocities.values()
                    ),
                    abs(raw_task_dof_velocities[SHACKLE_SLIDE_DOF]),
                )
                if raw_task_velocity_finite
                else math.inf
            )
            raw_task_angular_speed_radps = (
                max(
                    *(
                        float(np.linalg.norm(velocity[3:]))
                        for velocity in raw_task_body_velocities.values()
                    ),
                    abs(raw_task_dof_velocities[LOCK_CYLINDER_DOF]),
                    abs(raw_task_dof_velocities[SHACKLE_HINGE_DOF]),
                )
                if raw_task_velocity_finite
                else math.inf
            )
            _obs, _reward, terminated, truncated, step_info = environment.step(
                zero_action
            )
            post_zero_readback = environment._physical_readback()
            post_zero_numeric = np.asarray(
                [float(value) for value in post_zero_readback.__dict__.values()],
                dtype=np.float64,
            )
            post_zero_task_body_velocities = {
                name: environment._body_velocity(name)
                for name in task_body_names
            }
            post_zero_task_dof_velocities = {
                name: float(environment._dof_reading(name)[1])
                for name in EXPECTED_DOF_NAMES
            }
            post_zero_task_velocity_finite = bool(
                all(
                    velocity.shape == (6,) and np.all(np.isfinite(velocity))
                    for velocity in post_zero_task_body_velocities.values()
                )
                and np.isfinite(
                    list(post_zero_task_dof_velocities.values())
                ).all()
            )
            post_zero_task_linear_speed_mps = (
                max(
                    *(
                        float(np.linalg.norm(velocity[:3]))
                        for velocity in post_zero_task_body_velocities.values()
                    ),
                    abs(post_zero_task_dof_velocities[SHACKLE_SLIDE_DOF]),
                )
                if post_zero_task_velocity_finite
                else math.inf
            )
            post_zero_task_angular_speed_radps = (
                max(
                    *(
                        float(np.linalg.norm(velocity[3:]))
                        for velocity in post_zero_task_body_velocities.values()
                    ),
                    abs(post_zero_task_dof_velocities[LOCK_CYLINDER_DOF]),
                    abs(post_zero_task_dof_velocities[SHACKLE_HINGE_DOF]),
                )
                if post_zero_task_velocity_finite
                else math.inf
            )
            reset_trials.append(
                {
                    "seed": seed,
                    "sample": environment._p005_reset.to_dict(),
                    "finite": bool(
                        raw_readback.finite_state
                        and np.isfinite(raw_numeric).all()
                        and raw_task_velocity_finite
                        and raw_task_position_finite
                    ),
                    "post_zero_action_finite": bool(
                        post_zero_readback.finite_state
                        and np.isfinite(post_zero_numeric).all()
                        and post_zero_task_velocity_finite
                    ),
                    "failed": bool(step_info.get("failed", False)),
                    "initial_success": bool(
                        reset_info.get("succeed", False)
                        or step_info.get("succeed", False)
                        or terminated
                    ),
                    "truncated": bool(truncated),
                    "robot_initial_state_passed": robot_initial_passed,
                    "robot_initial_state": robot_initial_state,
                    "max_penetration_m": float(raw_readback.max_penetration_m),
                    "post_zero_action_max_penetration_m": float(
                        post_zero_readback.max_penetration_m
                    ),
                    "key_bottom_z_m": float(raw_readback.key_bottom_z_m),
                    "shackle_slide_m": raw_task_dof_positions[SHACKLE_SLIDE_DOF],
                    "lock_cylinder_angle_deg": math.degrees(
                        raw_task_dof_positions[LOCK_CYLINDER_DOF]
                    ),
                    "shackle_hinge_deg": math.degrees(
                        raw_task_dof_positions[SHACKLE_HINGE_DOF]
                    ),
                    "key_linear_speed_mps": float(
                        raw_readback.key_linear_speed_mps
                    ),
                    "key_angular_speed_radps": float(
                        raw_readback.key_angular_speed_radps
                    ),
                    "shackle_linear_speed_mps": float(
                        raw_readback.shackle_linear_speed_mps
                    ),
                    "shackle_angular_speed_radps": float(
                        raw_readback.shackle_angular_speed_radps
                    ),
                    "raw_task_body_velocity_linear_angular": {
                        name: velocity.tolist()
                        for name, velocity in raw_task_body_velocities.items()
                    },
                    "raw_task_dof_velocity": raw_task_dof_velocities,
                    "raw_task_dof_position": raw_task_dof_positions,
                    "authored_reset_pose": authored_reset_pose,
                    "raw_task_linear_speed_mps": raw_task_linear_speed_mps,
                    "raw_task_angular_speed_radps": (
                        raw_task_angular_speed_radps
                    ),
                    "post_zero_task_linear_speed_mps": (
                        post_zero_task_linear_speed_mps
                    ),
                    "post_zero_task_angular_speed_radps": (
                        post_zero_task_angular_speed_radps
                    ),
                    "key_support_force_n": float(key_support_force),
                    "key_support_penetration_m": float(key_support_penetration),
                }
            )

        # IK feasibility uses the real loaded robot, current TCP orientations,
        # and three task-critical workspace locations for each seeded layout.
        reachability: list[dict[str, Any]] = []
        environment.reset(seed=0)
        # These are measured palm poses from the passing ordinary-54D HP-14
        # trajectory, not guessed task endpoints.  Solving them from reset
        # checks that the exact deployed IK model still covers each critical
        # stage with usable joint-limit margin.
        workspace_targets = {
            "right_key_bow": (
                "right",
                np.asarray([
                    -0.421663, -0.244688, 0.959759,
                    0.706391, 0.005393, 0.707801, 0.000384,
                ]),
            ),
            "right_keyway": (
                "right",
                np.asarray([
                    -0.317836, -0.183358, 1.088023,
                    0.326051, -0.066125, 0.720447, 0.608501,
                ]),
            ),
            "left_shackle": (
                "left",
                np.asarray([
                    -0.446085, 0.075134, 1.194710,
                    0.664468, -0.033663, 0.743707, -0.065191,
                ]),
            ),
        }
        for role, (side, target) in workspace_targets.items():
            robot = environment._robots[side]
            solution = environment._compute_arm_joint_target_haorui(
                side, robot, target
            )
            row: dict[str, Any] = {
                "role": role,
                "side": side,
                "target_pose_wxyz": target.tolist(),
                "solved": solution is not None,
            }
            if solution is not None:
                q = _array(solution).reshape(-1)
                arm_ids = environment._arm_joint_ids[side]
                limits = _array(robot.data.soft_joint_pos_limits)[0, arm_ids]
                row["joint_limit_margin_rad"] = float(
                    np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q))
                )
                row["joint_position_rad"] = q.tolist()
            reachability.append(row)

        # Exercise the same state-part restore hook used by physical replay.
        # Perturb both the free key and every articulated task DOF, then demand
        # that the recorded frame-0 state restores all of them.
        initial_observation, _initial_info = environment.reset(seed=7)
        initial_state = {
            key: np.asarray(value).copy()
            for key, value in initial_observation["state"].items()
        }
        key_view = environment._p005_body_views[KEY_BODY_NAME]
        perturbed_key = key_view.get_transforms().clone()
        perturbed_key[:, 0] += 0.050
        key_indices = torch.arange(
            int(key_view.count),
            dtype=torch.long,
            device=perturbed_key.device,
        )
        key_view.set_transforms(perturbed_key, key_indices)
        for view, names in environment._grouped_dofs().values():
            positions = view.get_dof_positions().clone()
            for name, value in (
                ("p005_lock_cylinder_yaw", 0.35),
                ("p005_shackle_slide_joint", 0.020),
                ("p005_shackle_hinge_joint", 0.20),
            ):
                index = names.get(name)
                if index is not None:
                    positions[:, index] = value
            indices = torch.arange(
                int(view.count), dtype=torch.long, device=positions.device
            )
            view.set_dof_positions(positions, indices)
        restored = environment.restore_recorded_initial_state(initial_state)
        restored_key = np.asarray(restored["state"]["p005_key_pose"])
        restored_joint = np.asarray(restored["state"]["p005_joint_state"])
        initial_key = np.asarray(initial_state["p005_key_pose"])
        initial_joint = np.asarray(initial_state["p005_joint_state"])
        initial_state_restore = {
            "key_pose_max_abs_error": float(
                np.max(np.abs(restored_key - initial_key))
            ),
            "task_dof_max_abs_error": float(
                np.max(np.abs(restored_joint[:3] - initial_joint[:3]))
            ),
            "shackle_relocked": bool(environment._p005_shackle_locked),
        }

        # Apply a deliberate pre-unlock pull. A real locked mechanism must
        # remain below the frozen 3 mm early-motion failure threshold.
        environment.reset(seed=0)
        max_locked_slide = 0.0
        for view, names in environment._grouped_dofs().values():
            force = torch.zeros_like(view.get_dof_positions())
            indices = torch.arange(
                int(view.count), dtype=torch.long, device=force.device
            )
            slide_index = names.get(SHACKLE_SLIDE_DOF)
            hinge_index = names.get(SHACKLE_HINGE_DOF)
            if slide_index is not None:
                force[:, slide_index] = 25.0
            if hinge_index is not None:
                force[:, hinge_index] = 4.0
            for _ in range(PHYSICS_HZ):
                view.set_dof_actuation_forces(force, indices)
                for robot in environment._robots.values():
                    robot.write_data_to_sim()
                sim.step(render=False)
                for robot in environment._robots.values():
                    robot.update(environment._sim_dt)
                max_locked_slide = max(
                    max_locked_slide,
                    float(environment._dof_reading(SHACKLE_SLIDE_DOF)[0]),
                )
            force.zero_()
            view.set_dof_actuation_forces(force, indices)

        environment.reset(seed=0)
        robot_stability = RobotStabilityAudit.start(environment)
        initial_task_body_poses = {
            name: environment._body_pose_wxyz(name) for name in task_body_names
        }
        previous_task_body_poses = {
            name: pose.copy() for name, pose in initial_task_body_poses.items()
        }
        initial_task_body_velocities = {
            name: environment._body_velocity(name) for name in task_body_names
        }
        initial_task_dof_positions = {
            name: float(environment._dof_reading(name)[0])
            for name in EXPECTED_DOF_NAMES
        }
        initial_task_dof_velocities = {
            name: float(environment._dof_reading(name)[1])
            for name in EXPECTED_DOF_NAMES
        }
        previous_task_dof_positions = dict(initial_task_dof_positions)
        stability_steps = int(round(float(args.stability_seconds) * PHYSICS_HZ))
        max_penetration = 0.0
        peak_key_speed = 0.0
        peak_key_angular_speed = 0.0
        peak_shackle_speed = 0.0
        peak_shackle_angular_speed = 0.0
        supported_samples = 0
        support_samples = 0
        support_nonfinite_samples = 0
        max_support_penetration = 0.0
        nan_count = 0
        key_pose_invalid_samples = 0
        max_key_translation_drift_m = 0.0
        max_key_rotation_drift_rad = 0.0
        max_task_body_translation_drift_m = {
            name: 0.0 for name in task_body_names
        }
        max_task_body_rotation_drift_rad = {
            name: 0.0 for name in task_body_names
        }
        max_lock_cylinder_drift_deg = 0.0
        max_shackle_slide_drift_m = 0.0
        max_shackle_hinge_drift_deg = 0.0
        task_velocity_invalid_samples = 0
        peak_task_linear_speed_mps = 0.0
        peak_task_angular_speed_radps = 0.0
        task_pose_increment_samples = 0
        task_pose_increment_invalid_samples = 0
        static_pose_covered_steps = 0
        max_task_pose_step_translation_m = 0.0
        max_task_pose_step_rotation_rad = 0.0
        max_task_dof_step_translation_m = 0.0
        max_task_dof_step_rotation_rad = 0.0
        authored_reset_pose_samples = 0
        stability_samples: list[dict[str, Any]] = []
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
                bool(environment._authored_reset_pose_report().get("passed", False))
            )
            values = np.asarray(
                [float(value) for value in readback.__dict__.values()],
                dtype=np.float64,
            )
            nan_count += int(values.size - np.isfinite(values).sum())
            max_penetration = max(
                max_penetration, float(readback.max_penetration_m)
            )
            peak_key_speed = max(
                peak_key_speed, float(readback.key_linear_speed_mps)
            )
            peak_key_angular_speed = max(
                peak_key_angular_speed,
                float(readback.key_angular_speed_radps),
            )
            peak_shackle_speed = max(
                peak_shackle_speed, float(readback.shackle_linear_speed_mps)
            )
            peak_shackle_angular_speed = max(
                peak_shackle_angular_speed,
                float(readback.shackle_angular_speed_radps),
            )
            task_body_velocities = {
                name: environment._body_velocity(name)
                for name in task_body_names
            }
            current_task_dof_readings = {
                name: environment._dof_reading(name)
                for name in EXPECTED_DOF_NAMES
            }
            current_task_dof_positions = {
                name: float(reading[0])
                for name, reading in current_task_dof_readings.items()
            }
            current_task_dof_velocities = {
                name: float(reading[1])
                for name, reading in current_task_dof_readings.items()
            }
            task_velocity_valid = bool(
                all(
                    velocity.shape == (6,) and np.all(np.isfinite(velocity))
                    for velocity in task_body_velocities.values()
                )
                and np.isfinite(
                    list(current_task_dof_velocities.values())
                ).all()
            )
            task_velocity_invalid_samples += int(not task_velocity_valid)
            if task_velocity_valid:
                current_task_slide_speed_mps = abs(
                    current_task_dof_velocities[SHACKLE_SLIDE_DOF]
                )
                current_task_rotational_dof_speed_radps = max(
                    abs(current_task_dof_velocities[LOCK_CYLINDER_DOF]),
                    abs(current_task_dof_velocities[SHACKLE_HINGE_DOF]),
                )
                current_task_linear_speed_mps = max(
                    *(
                        float(np.linalg.norm(velocity[:3]))
                        for velocity in task_body_velocities.values()
                    ),
                    current_task_slide_speed_mps,
                )
                current_task_angular_speed_radps = max(
                    *(
                        float(np.linalg.norm(velocity[3:]))
                        for velocity in task_body_velocities.values()
                    ),
                    current_task_rotational_dof_speed_radps,
                )
                peak_task_linear_speed_mps = max(
                    peak_task_linear_speed_mps,
                    current_task_linear_speed_mps,
                )
                peak_task_angular_speed_radps = max(
                    peak_task_angular_speed_radps,
                    current_task_angular_speed_radps,
                )
            else:
                current_task_linear_speed_mps = math.inf
                current_task_angular_speed_radps = math.inf
                current_task_slide_speed_mps = math.inf
                current_task_rotational_dof_speed_radps = math.inf
            support_force, support_penetration = environment._contact_reading(
                "key_support"
            )
            support_finite = bool(
                np.isfinite(support_force)
                and np.isfinite(support_penetration)
            )
            support_samples += 1
            support_nonfinite_samples += int(not support_finite)
            supported_samples += int(
                support_finite
                and support_force >= 0.01
            )
            max_support_penetration = max(
                max_support_penetration, float(support_penetration)
            )
            current_task_body_poses = {
                name: environment._body_pose_wxyz(name)
                for name in task_body_names
            }
            task_pose_increment_samples += 1
            task_pose_valid = all(
                current.shape == (7,)
                and previous.shape == (7,)
                and initial.shape == (7,)
                and np.all(np.isfinite(current))
                and np.all(np.isfinite(previous))
                and np.all(np.isfinite(initial))
                and np.linalg.norm(current[3:7]) > 1.0e-12
                and np.linalg.norm(previous[3:7]) > 1.0e-12
                and np.linalg.norm(initial[3:7]) > 1.0e-12
                for current, previous, initial in (
                    (
                        current_task_body_poses[name],
                        previous_task_body_poses[name],
                        initial_task_body_poses[name],
                    )
                    for name in task_body_names
                )
            )
            key_pose_invalid_samples += int(
                not bool(
                    current_task_body_poses[KEY_BODY_NAME].shape == (7,)
                    and np.all(
                        np.isfinite(current_task_body_poses[KEY_BODY_NAME])
                    )
                    and np.linalg.norm(
                        current_task_body_poses[KEY_BODY_NAME][3:7]
                    )
                    > 1.0e-12
                )
            )
            task_pose_increment_invalid_samples += int(not task_pose_valid)
            current_task_body_translation_drift_m: dict[str, float] = {}
            current_task_body_rotation_drift_rad: dict[str, float] = {}
            frame_pose_step_translation_m = math.inf
            frame_pose_step_rotation_rad = math.inf
            if task_pose_valid:
                frame_pose_step_translation_m = 0.0
                frame_pose_step_rotation_rad = 0.0
                for name in task_body_names:
                    current_pose = current_task_body_poses[name]
                    initial_pose = initial_task_body_poses[name]
                    previous_pose = previous_task_body_poses[name]
                    translation_drift = float(
                        np.linalg.norm(current_pose[:3] - initial_pose[:3])
                    )
                    rotation_drift = _quaternion_rotation_error(
                        current_pose[3:7], initial_pose[3:7]
                    )
                    translation_step = float(
                        np.linalg.norm(current_pose[:3] - previous_pose[:3])
                    )
                    rotation_step = _quaternion_rotation_error(
                        current_pose[3:7], previous_pose[3:7]
                    )
                    current_task_body_translation_drift_m[name] = (
                        translation_drift
                    )
                    current_task_body_rotation_drift_rad[name] = rotation_drift
                    max_task_body_translation_drift_m[name] = max(
                        max_task_body_translation_drift_m[name],
                        translation_drift,
                    )
                    max_task_body_rotation_drift_rad[name] = max(
                        max_task_body_rotation_drift_rad[name],
                        rotation_drift,
                    )
                    frame_pose_step_translation_m = max(
                        frame_pose_step_translation_m,
                        translation_step,
                    )
                    frame_pose_step_rotation_rad = max(
                        frame_pose_step_rotation_rad,
                        rotation_step,
                    )
                max_task_pose_step_translation_m = max(
                    max_task_pose_step_translation_m,
                    frame_pose_step_translation_m,
                )
                max_task_pose_step_rotation_rad = max(
                    max_task_pose_step_rotation_rad,
                    frame_pose_step_rotation_rad,
                )
            else:
                current_task_body_translation_drift_m = {
                    name: math.inf for name in task_body_names
                }
                current_task_body_rotation_drift_rad = {
                    name: math.inf for name in task_body_names
                }
            current_key_translation_drift_m = (
                current_task_body_translation_drift_m[KEY_BODY_NAME]
            )
            current_key_rotation_drift_rad = (
                current_task_body_rotation_drift_rad[KEY_BODY_NAME]
            )
            max_key_translation_drift_m = max_task_body_translation_drift_m[
                KEY_BODY_NAME
            ]
            max_key_rotation_drift_rad = max_task_body_rotation_drift_rad[
                KEY_BODY_NAME
            ]
            task_dof_valid = bool(
                np.isfinite(list(current_task_dof_positions.values())).all()
                and np.isfinite(list(previous_task_dof_positions.values())).all()
                and np.isfinite(list(initial_task_dof_positions.values())).all()
            )
            frame_dof_step_translation_m = math.inf
            frame_dof_step_rotation_rad = math.inf
            if task_dof_valid:
                current_lock_cylinder_drift_deg = math.degrees(
                    abs(
                        current_task_dof_positions[LOCK_CYLINDER_DOF]
                        - initial_task_dof_positions[LOCK_CYLINDER_DOF]
                    )
                )
                current_shackle_slide_drift_m = abs(
                    current_task_dof_positions[SHACKLE_SLIDE_DOF]
                    - initial_task_dof_positions[SHACKLE_SLIDE_DOF]
                )
                current_shackle_hinge_drift_deg = math.degrees(
                    abs(
                        current_task_dof_positions[SHACKLE_HINGE_DOF]
                        - initial_task_dof_positions[SHACKLE_HINGE_DOF]
                    )
                )
                max_lock_cylinder_drift_deg = max(
                    max_lock_cylinder_drift_deg,
                    current_lock_cylinder_drift_deg,
                )
                max_shackle_slide_drift_m = max(
                    max_shackle_slide_drift_m,
                    current_shackle_slide_drift_m,
                )
                max_shackle_hinge_drift_deg = max(
                    max_shackle_hinge_drift_deg,
                    current_shackle_hinge_drift_deg,
                )
                frame_dof_step_translation_m = abs(
                    current_task_dof_positions[SHACKLE_SLIDE_DOF]
                    - previous_task_dof_positions[SHACKLE_SLIDE_DOF]
                )
                frame_dof_step_rotation_rad = max(
                    abs(
                        current_task_dof_positions[LOCK_CYLINDER_DOF]
                        - previous_task_dof_positions[LOCK_CYLINDER_DOF]
                    ),
                    abs(
                        current_task_dof_positions[SHACKLE_HINGE_DOF]
                        - previous_task_dof_positions[SHACKLE_HINGE_DOF]
                    ),
                )
                max_task_dof_step_translation_m = max(
                    max_task_dof_step_translation_m,
                    frame_dof_step_translation_m,
                )
                max_task_dof_step_rotation_rad = max(
                    max_task_dof_step_rotation_rad,
                    frame_dof_step_rotation_rad,
                )
            else:
                current_lock_cylinder_drift_deg = math.inf
                current_shackle_slide_drift_m = math.inf
                current_shackle_hinge_drift_deg = math.inf
            frame_static_pose = bool(
                task_pose_valid
                and task_dof_valid
                and frame_pose_step_translation_m
                <= STATIC_POSE_TRANSLATION_STEP_LIMIT_M
                and frame_pose_step_rotation_rad
                <= STATIC_POSE_ROTATION_STEP_LIMIT_RAD
                and frame_dof_step_translation_m
                <= STATIC_POSE_TRANSLATION_STEP_LIMIT_M
                and frame_dof_step_rotation_rad
                <= STATIC_POSE_ROTATION_STEP_LIMIT_RAD
            )
            static_pose_covered_steps += int(frame_static_pose)
            previous_task_body_poses = {
                name: pose.copy()
                for name, pose in current_task_body_poses.items()
            }
            previous_task_dof_positions = current_task_dof_positions
            if step % 12 == 0 or step == stability_steps - 1:
                stability_samples.append(
                    {
                        "physics_step": step + 1,
                        "finite": bool(np.isfinite(values).all()),
                        "key_linear_speed_mps": float(
                            readback.key_linear_speed_mps
                        ),
                        "key_angular_speed_radps": float(
                            readback.key_angular_speed_radps
                        ),
                        "shackle_linear_speed_mps": float(
                            readback.shackle_linear_speed_mps
                        ),
                        "shackle_angular_speed_radps": float(
                            readback.shackle_angular_speed_radps
                        ),
                        "task_peak_linear_speed_mps": (
                            current_task_linear_speed_mps
                        ),
                        "task_peak_angular_speed_radps": (
                            current_task_angular_speed_radps
                        ),
                        "task_slide_speed_mps": current_task_slide_speed_mps,
                        "task_rotational_dof_speed_radps": (
                            current_task_rotational_dof_speed_radps
                        ),
                        "max_penetration_m": float(
                            readback.max_penetration_m
                        ),
                        "key_support_force_n": float(support_force),
                        "key_support_penetration_m": float(
                            support_penetration
                        ),
                        "key_translation_drift_m": (
                            current_key_translation_drift_m
                        ),
                        "key_rotation_drift_deg": math.degrees(
                            current_key_rotation_drift_rad
                        ),
                        "lock_cylinder_drift_deg": (
                            current_lock_cylinder_drift_deg
                        ),
                        "shackle_slide_drift_m": current_shackle_slide_drift_m,
                        "shackle_hinge_drift_deg": (
                            current_shackle_hinge_drift_deg
                        ),
                        "task_body_translation_drift_m": (
                            current_task_body_translation_drift_m
                        ),
                        "task_body_rotation_drift_deg": {
                            name: math.degrees(value)
                            for name, value in (
                                current_task_body_rotation_drift_rad.items()
                            )
                        },
                        "task_pose_step_translation_m": (
                            frame_pose_step_translation_m
                        ),
                        "task_pose_step_rotation_rad": frame_pose_step_rotation_rad,
                        "task_dof_step_translation_m": (
                            frame_dof_step_translation_m
                        ),
                        "task_dof_step_rotation_rad": frame_dof_step_rotation_rad,
                        "static_pose_fallback_frame": frame_static_pose,
                    }
                )

        ordinary_velocity_pass = bool(
            task_velocity_invalid_samples == 0
            and peak_task_linear_speed_mps
            <= PRIORITY_ZERO_LINEAR_SPEED_LIMIT_MPS
            and peak_task_angular_speed_radps
            <= PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RADPS
        )
        static_pose_fallback_pass = bool(
            task_pose_increment_samples == stability_steps
            and task_pose_increment_invalid_samples == 0
            and static_pose_covered_steps == stability_steps
            and max_task_pose_step_translation_m
            <= STATIC_POSE_TRANSLATION_STEP_LIMIT_M
            and max_task_pose_step_rotation_rad
            <= STATIC_POSE_ROTATION_STEP_LIMIT_RAD
            and max_task_dof_step_translation_m
            <= STATIC_POSE_TRANSLATION_STEP_LIMIT_M
            and max_task_dof_step_rotation_rad
            <= STATIC_POSE_ROTATION_STEP_LIMIT_RAD
        )
        velocity_or_static_pose_pass = bool(
            task_velocity_invalid_samples == 0
            and (ordinary_velocity_pass or static_pose_fallback_pass)
        )
        velocity_branch = (
            "ordinary_velocity"
            if ordinary_velocity_pass
            else "per_frame_static_pose_fallback"
            if velocity_or_static_pose_pass and static_pose_fallback_pass
            else "failed"
        )
        final_readback = environment._physical_readback()
        final_key_pose = environment._body_pose_wxyz(KEY_BODY_NAME)
        final_task_body_velocities = {
            name: environment._body_velocity(name) for name in task_body_names
        }
        final_task_dof_velocities = {
            name: float(environment._dof_reading(name)[1])
            for name in EXPECTED_DOF_NAMES
        }
        initial_task_velocity_exact_zero_observed = bool(
            all(np.array_equal(velocity, np.zeros(6)) for velocity in (
                initial_task_body_velocities.values()
            ))
            and all(value == 0.0 for value in initial_task_dof_velocities.values())
        )
        final_task_velocity_exact_zero_observed = bool(
            all(np.array_equal(velocity, np.zeros(6)) for velocity in (
                final_task_body_velocities.values()
            ))
            and all(value == 0.0 for value in final_task_dof_velocities.values())
        )
        final_support_force, final_support_penetration = (
            environment._contact_reading("key_support")
        )
        final_key_translation_drift_m = float(
            np.linalg.norm(
                final_key_pose[:3]
                - initial_task_body_poses[KEY_BODY_NAME][:3]
            )
        )
        final_key_rotation_drift_rad = _quaternion_rotation_error(
            final_key_pose[3:7], initial_task_body_poses[KEY_BODY_NAME][3:7]
        )

        reset_pass = bool(
            len(reset_trials) == int(args.reset_count)
            and int(args.reset_count) >= 20
            and all(
                row["finite"]
                and row["post_zero_action_finite"]
                and not row["failed"]
                and not row["initial_success"]
                and not row["truncated"]
                and row["robot_initial_state_passed"]
                and row["max_penetration_m"]
                <= PRIORITY_ZERO_MAX_PENETRATION_M
                and row["post_zero_action_max_penetration_m"]
                <= PRIORITY_ZERO_MAX_PENETRATION_M
                and row["key_linear_speed_mps"] <= 0.05
                and row["key_angular_speed_radps"] <= 0.75
                and row["shackle_linear_speed_mps"] <= 0.05
                and row["shackle_angular_speed_radps"] <= 0.75
                and abs(row["lock_cylinder_angle_deg"]) <= 1.0
                and abs(row["shackle_slide_m"]) <= 0.001
                and abs(row["shackle_hinge_deg"]) <= 1.0
                and bool(row["authored_reset_pose"].get("passed", False))
                and row["raw_task_linear_speed_mps"] <= 0.05
                and row["raw_task_angular_speed_radps"] <= 0.75
                and row["post_zero_task_linear_speed_mps"] <= 0.05
                and row["post_zero_task_angular_speed_radps"] <= 0.75
                and row["key_support_force_n"] >= 0.01
                and row["key_support_penetration_m"] <= 0.0005
                for row in reset_trials
            )
        )
        reachability_pass = bool(
            len(reachability) == 3
            and all(
                row["solved"]
                and row.get("joint_limit_margin_rad", -math.inf) >= 0.03
                for row in reachability
            )
        )
        locked_pass = max_locked_slide <= SHACKLE_PREUNLOCK_MAX_M
        stability_pass = bool(
            stability_steps >= 10 * PHYSICS_HZ
            and nan_count == 0
            and max_penetration <= PRIORITY_ZERO_MAX_PENETRATION_M
            and task_velocity_invalid_samples == 0
            and task_pose_increment_samples == stability_steps
            and authored_reset_pose_samples == stability_steps
            and task_pose_increment_invalid_samples == 0
            and velocity_or_static_pose_pass
            and key_pose_invalid_samples == 0
            and all(
                drift <= 0.005
                for drift in max_task_body_translation_drift_m.values()
            )
            and all(
                drift <= math.radians(3.0)
                for drift in max_task_body_rotation_drift_rad.values()
            )
            and max_lock_cylinder_drift_deg <= 3.0
            and max_shackle_slide_drift_m <= 0.005
            and max_shackle_hinge_drift_deg <= 3.0
            and support_samples == stability_steps
            and support_nonfinite_samples == 0
            and supported_samples == support_samples
            and final_support_force >= 0.01
            and max(max_support_penetration, final_support_penetration) <= 0.0005
            and robot_stability.passed(
                physics_steps=stability_steps,
                physics_hz=PHYSICS_HZ,
            )
            and not environment._p005_failure_latched
        )
        checks = {
            "complete_scene_initialized": exact_production_child,
            "exact_production_runtime": runtime_contract["passed"],
            "dual_rokae_wuji": bool(
                robot_inventory["names"] == ["left", "right"]
                and all(v == 7 for v in robot_inventory["arm_joint_counts"].values())
                and all(v == 20 for v in robot_inventory["hand_joint_counts"].values())
            ),
            "three_dof_lock_topology": set(environment._p005_dof_views)
            == set(EXPECTED_DOF_NAMES),
            "five_exact_contact_views": len(contact_inventory) == 5
            and all(row["sensor_count"] == 1 for row in contact_inventory.values())
            and key_support_pair_exact,
            "twenty_reset_and_zero_action_trials": reset_pass,
            "critical_workspace_reachable": reachability_pass,
            "locked_shackle_force_probe": locked_pass,
            "recorded_initial_state_restore": bool(
                initial_state_restore["key_pose_max_abs_error"] <= 2e-3
                and initial_state_restore["task_dof_max_abs_error"] <= 1e-4
                and initial_state_restore["shackle_relocked"]
            ),
            "velocity_or_static_pose": velocity_or_static_pose_pass,
            "ten_second_stability": stability_pass,
        }
        result.update(
            {
                "passed": all(checks.values()),
                "checks": checks,
                "robot": robot_inventory,
                "dof_names": sorted(environment._p005_dof_views),
                "contact_views": contact_inventory,
                "key_support_pair_exact": key_support_pair_exact,
                "expected_key_support_pair": expected_key_support_pair,
                "reset_trials": reset_trials,
                "reachability_trials": reachability,
                "locked_force_probe": {
                    "slide_force_n": 25.0,
                    "hinge_force_nm": 4.0,
                    "steps": PHYSICS_HZ,
                    "max_shackle_slide_m": max_locked_slide,
                    "threshold_m": SHACKLE_PREUNLOCK_MAX_M,
                },
                "recorded_initial_state_restore": initial_state_restore,
                "velocity_or_static_pose": {
                    "passed": velocity_or_static_pose_pass,
                    "selected_branch": velocity_branch,
                    "ordinary_velocity": {
                        "passed": ordinary_velocity_pass,
                        "invalid_sample_count": task_velocity_invalid_samples,
                        "peak_linear_speed_mps": peak_task_linear_speed_mps,
                        "linear_speed_limit_mps": (
                            PRIORITY_ZERO_LINEAR_SPEED_LIMIT_MPS
                        ),
                        "peak_angular_speed_radps": (
                            peak_task_angular_speed_radps
                        ),
                        "angular_speed_limit_radps": (
                            PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RADPS
                        ),
                    },
                    "per_frame_static_pose_fallback": {
                        "passed": static_pose_fallback_pass,
                        "required_physics_steps": stability_steps,
                        "pose_increment_sample_count": (
                            task_pose_increment_samples
                        ),
                        "invalid_pose_increment_sample_count": (
                            task_pose_increment_invalid_samples
                        ),
                        "covered_physics_steps": static_pose_covered_steps,
                        "max_body_translation_step_m": (
                            max_task_pose_step_translation_m
                        ),
                        "body_translation_step_limit_m": (
                            STATIC_POSE_TRANSLATION_STEP_LIMIT_M
                        ),
                        "max_body_rotation_step_rad": (
                            max_task_pose_step_rotation_rad
                        ),
                        "body_rotation_step_limit_rad": (
                            STATIC_POSE_ROTATION_STEP_LIMIT_RAD
                        ),
                        "max_dof_translation_step_m": (
                            max_task_dof_step_translation_m
                        ),
                        "max_dof_rotation_step_rad": (
                            max_task_dof_step_rotation_rad
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
                    "final_task_body_velocity_linear_angular": {
                        name: velocity.tolist()
                        for name, velocity in final_task_body_velocities.items()
                    },
                    "final_task_dof_velocity": final_task_dof_velocities,
                    "production_readiness_gap": (
                        "Standalone acceptance performs no initialization or "
                        "final exact-zero velocity boundary write; the production "
                        "recording-origin gate must still verify that boundary."
                    ),
                },
                "stability": {
                    "steps": stability_steps,
                    "physics_hz": PHYSICS_HZ,
                    "simulated_seconds": stability_steps / PHYSICS_HZ,
                    "nan_count": nan_count,
                    "max_penetration_m": max_penetration,
                    "max_penetration_limit_m": (
                        PRIORITY_ZERO_MAX_PENETRATION_M
                    ),
                    "peak_key_linear_speed_mps": peak_key_speed,
                    "peak_key_angular_speed_radps": peak_key_angular_speed,
                    "peak_shackle_linear_speed_mps": peak_shackle_speed,
                    "peak_shackle_angular_speed_radps": (
                        peak_shackle_angular_speed
                    ),
                    "final_key_linear_speed_mps": float(
                        final_readback.key_linear_speed_mps
                    ),
                    "final_key_angular_speed_radps": float(
                        final_readback.key_angular_speed_radps
                    ),
                    "final_shackle_linear_speed_mps": float(
                        final_readback.shackle_linear_speed_mps
                    ),
                    "final_shackle_angular_speed_radps": float(
                        final_readback.shackle_angular_speed_radps
                    ),
                    "key_pose_invalid_samples": key_pose_invalid_samples,
                    "max_key_translation_drift_m": (
                        max_key_translation_drift_m
                    ),
                    "max_key_rotation_drift_deg": math.degrees(
                        max_key_rotation_drift_rad
                    ),
                    "max_task_body_translation_drift_m": (
                        max_task_body_translation_drift_m
                    ),
                    "max_task_body_rotation_drift_deg": {
                        name: math.degrees(value)
                        for name, value in max_task_body_rotation_drift_rad.items()
                    },
                    "max_lock_cylinder_drift_deg": (
                        max_lock_cylinder_drift_deg
                    ),
                    "max_shackle_slide_drift_m": max_shackle_slide_drift_m,
                    "max_shackle_hinge_drift_deg": max_shackle_hinge_drift_deg,
                    "final_key_translation_drift_m": (
                        final_key_translation_drift_m
                    ),
                    "final_key_rotation_drift_deg": math.degrees(
                        final_key_rotation_drift_rad
                    ),
                    "support_samples": support_samples,
                    "authored_reset_pose_samples": authored_reset_pose_samples,
                    "supported_samples": supported_samples,
                    "support_nonfinite_samples": support_nonfinite_samples,
                    "supported_sample_fraction": (
                        supported_samples / max(1, support_samples)
                    ),
                    "final_key_support_force_n": float(final_support_force),
                    "max_key_support_penetration_m": float(
                        max(max_support_penetration, final_support_penetration)
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
        rc = 1
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
        # The validator owns this bounded Kit process.  Avoid teardown paths
        # that can hang or replace a failed gate with process status zero.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
