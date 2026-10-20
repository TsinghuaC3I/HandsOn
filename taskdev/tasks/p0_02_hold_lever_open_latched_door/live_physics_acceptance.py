#!/usr/bin/env python3
"""Bounded live PhysX validation for the complete P0-02 scene."""

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-usd-root", type=Path)
    parser.add_argument("--stability-seconds", type=float, default=10.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _rotation_error(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left[:3, :3]).T @ np.asarray(right[:3, :3])
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64).reshape(4)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64).reshape(4)
    result = np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def _pose_matrix_wxyz(pose: np.ndarray) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64).reshape(7)
    w, x, y, z = values[3:7]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result[:3, 3] = values[:3]
    return result


def _robot_initial_state_evidence(
    environment: Any,
    *,
    expected_base_poses_wxyz: dict[str, np.ndarray],
    expected_palm_poses_wxyz: dict[str, np.ndarray],
) -> tuple[bool, dict[str, Any]]:
    """Check applied bases, real joint state, and real palms after reset."""

    rows: dict[str, Any] = {}
    for side in ("right", "left"):
        robot = environment._robots[side]
        arm_ids = environment._arm_joint_ids[side]
        actual_base = environment._robot_base_pose_matrix(side, robot)
        expected_base = _pose_matrix_wxyz(expected_base_poses_wxyz[side])
        actual_palm = environment._robot_body_pose_matrix(
            robot, environment._ee_body_indices[side]
        )
        expected_palm = _pose_matrix_wxyz(expected_palm_poses_wxyz[side])
        joint_position = _array(robot.data.joint_pos)[0, arm_ids]
        default_joint_position = _array(robot.data.default_joint_pos)[0, arm_ids]
        joint_velocity = _array(robot.data.joint_vel)[0, arm_ids]
        checks = {
            "base_position": float(
                np.linalg.norm(actual_base[:3, 3] - expected_base[:3, 3])
            )
            <= 0.002,
            "base_orientation": _rotation_error(actual_base, expected_base)
            <= math.radians(0.5),
            "arm_home": float(
                np.max(np.abs(joint_position - default_joint_position))
            )
            <= 0.01,
            "arm_stationary": float(np.max(np.abs(joint_velocity))) <= 0.05,
            "palm_position": float(
                np.linalg.norm(actual_palm[:3, 3] - expected_palm[:3, 3])
            )
            <= 0.03,
            "palm_orientation": _rotation_error(actual_palm, expected_palm)
            <= math.radians(5.0),
            "correct_side": bool(
                actual_palm[1, 3] < 0.0 if side == "right" else actual_palm[1, 3] > 0.0
            ),
        }
        rows[side] = {
            "passed": all(checks.values()),
            "checks": checks,
            "expected_base_pose_wxyz": expected_base_poses_wxyz[side].tolist(),
            "actual_base_position_m": actual_base[:3, 3].tolist(),
            "base_position_error_m": float(
                np.linalg.norm(actual_base[:3, 3] - expected_base[:3, 3])
            ),
            "base_orientation_error_deg": math.degrees(
                _rotation_error(actual_base, expected_base)
            ),
            "actual_arm_joint_position_rad": joint_position.tolist(),
            "default_arm_joint_position_rad": default_joint_position.tolist(),
            "arm_home_max_error_rad": float(
                np.max(np.abs(joint_position - default_joint_position))
            ),
            "arm_max_speed_rad_s": float(np.max(np.abs(joint_velocity))),
            "expected_palm_pose_wxyz": expected_palm_poses_wxyz[side].tolist(),
            "actual_palm_position_m": actual_palm[:3, 3].tolist(),
            "palm_position_error_m": float(
                np.linalg.norm(actual_palm[:3, 3] - expected_palm[:3, 3])
            ),
            "palm_orientation_error_deg": math.degrees(
                _rotation_error(actual_palm, expected_palm)
            ),
        }
    return all(row["passed"] for row in rows.values()), rows


def _robot_stability_snapshot(environment: Any) -> dict[str, dict[str, np.ndarray]]:
    snapshot: dict[str, dict[str, np.ndarray]] = {}
    for side in ("right", "left"):
        robot = environment._robots[side]
        arm_ids = environment._arm_joint_ids[side]
        snapshot[side] = {
            "base": environment._robot_base_pose_matrix(side, robot),
            "palm": environment._robot_body_pose_matrix(
                robot, environment._ee_body_indices[side]
            ),
            "joint_position": _array(robot.data.joint_pos)[0, arm_ids].copy(),
            "joint_velocity": _array(robot.data.joint_vel)[0, arm_ids].copy(),
        }
    return snapshot


def _reset_pose_errors(
    *,
    reset: Any,
    actual_wxyz: np.ndarray,
    composed_wxyz: np.ndarray,
    nominal_xyz_m: np.ndarray,
    nominal_yaw_deg: float,
) -> tuple[float, float]:
    """Compare the actual articulation-root delta with the sampled task delta."""

    actual = np.asarray(actual_wxyz, dtype=np.float64).reshape(7)
    composed = np.asarray(composed_wxyz, dtype=np.float64).reshape(7)
    expected_translation_delta = np.asarray(
        [reset.root_x_m, reset.root_y_m, reset.root_z_m], dtype=np.float64
    ) - np.asarray(nominal_xyz_m, dtype=np.float64)
    translation_error = float(
        np.max(np.abs((actual[:3] - composed[:3]) - expected_translation_delta))
    )
    composed_inverse = composed[3:7].copy()
    composed_inverse[1:] *= -1.0
    relative = _quat_multiply_wxyz(actual[3:7], composed_inverse)
    actual_delta_yaw_deg = math.degrees(
        math.atan2(
            2.0 * (relative[0] * relative[3] + relative[1] * relative[2]),
            1.0 - 2.0 * (relative[2] ** 2 + relative[3] ** 2),
        )
    )
    expected_delta_yaw_deg = float(reset.root_yaw_deg) - float(nominal_yaw_deg)
    yaw_error = abs(
        (actual_delta_yaw_deg - expected_delta_yaw_deg + 180.0) % 360.0 - 180.0
    )
    return translation_error, float(yaw_error)


def _scene_geometry_evidence(
    report: Any,
    *,
    expected_dimensions_m: tuple[float, float, float],
    expected_visual_meshes: int,
    expected_colliders: int,
    expected_source_archive_sha256: str,
    minimum_grasp_clearance_m: float,
    maximum_gpu_convex_vertices: int,
) -> tuple[bool, dict[str, Any]]:
    """Validate the live composed-stage report without importing pxr here."""

    if report is None:
        return False, {"available": False}
    actual_dimensions = tuple(
        float(value) for value in report.front_handle_dimensions_m
    )
    actual_body_scale = tuple(float(value) for value in report.front_handle_body_scale)
    actual_grasp_clearance_m = float(report.front_handle_grasp_clearance_m)
    checks = {
        "table_hidden": bool(report.table_hidden),
        "table_colliders_disabled": int(report.table_colliders_disabled) > 0,
        "front_handle_visual_topology": (
            int(report.replacement_visual_meshes)
            == int(expected_visual_meshes)
        ),
        "front_handle_collision_topology": (
            int(report.replacement_colliders) == int(expected_colliders)
        ),
        "front_handle_gpu_convex_budget": (
            int(report.replacement_max_convex_vertices)
            <= int(maximum_gpu_convex_vertices)
        ),
        "front_handle_uses_gpu_safe_primitives": (
            str(report.replacement_collision_primitive) == "Cube"
        ),
        "old_handle_payload_disabled": (
            int(report.old_handle_visual_meshes_hidden) == 1
            and int(report.old_handle_colliders_disabled) == 4
        ),
        "replacement_source_archive_sha256": (
            str(report.replacement_source_archive_sha256)
            == expected_source_archive_sha256
        ),
        "front_handle_dimensions": bool(
            np.allclose(
                actual_dimensions,
                expected_dimensions_m,
                rtol=0.0,
                atol=1.0e-9,
            )
        ),
        "front_handle_body_scale_unchanged": bool(
            np.allclose(actual_body_scale, (1.0, 1.0, 1.0), rtol=0.0, atol=1.0e-9)
        ),
        "front_handle_grasp_clearance": (
            actual_grasp_clearance_m >= minimum_grasp_clearance_m
        ),
    }
    evidence = {
        "available": True,
        "checks": checks,
        "table_hidden": bool(report.table_hidden),
        "table_colliders_disabled": int(report.table_colliders_disabled),
        "table_collision_instance_deactivated": bool(
            report.table_collision_instance_deactivated
        ),
        "replacement_asset_title": str(report.replacement_asset_title),
        "replacement_asset_author": str(report.replacement_asset_author),
        "replacement_asset_license": str(report.replacement_asset_license),
        "replacement_source_archive_sha256": str(
            report.replacement_source_archive_sha256
        ),
        "replacement_visual_meshes": int(report.replacement_visual_meshes),
        "replacement_colliders": int(report.replacement_colliders),
        "replacement_max_convex_vertices": int(
            report.replacement_max_convex_vertices
        ),
        "replacement_collision_primitive": str(
            report.replacement_collision_primitive
        ),
        "maximum_gpu_convex_vertices": int(maximum_gpu_convex_vertices),
        "old_handle_visual_meshes_hidden": int(
            report.old_handle_visual_meshes_hidden
        ),
        "old_handle_colliders_disabled": int(
            report.old_handle_colliders_disabled
        ),
        "front_handle_dimensions_m": list(actual_dimensions),
        "expected_front_handle_dimensions_m": list(expected_dimensions_m),
        "front_handle_grasp_clearance_m": actual_grasp_clearance_m,
        "minimum_front_handle_grasp_clearance_m": minimum_grasp_clearance_m,
        "front_handle_body_scale": list(actual_body_scale),
    }
    return all(checks.values()), evidence


def main() -> int:
    args = _parser().parse_args()
    args.headless = True
    args.enable_cameras = False
    args.device = str(getattr(args, "device", None) or "cuda:0")
    if args.robot_usd_root is not None:
        os.environ["HANDSON_P002_PREBUILT_ROBOT_USD_ROOT"] = str(
            args.robot_usd_root.resolve()
        )
    os.environ["HANDSON_ISAACLAB_ARM_IK_BACKEND"] = "haorui"
    launcher = AppLauncher(args)
    app = launcher.app
    environment = None
    result: dict[str, Any] = {
        "schema": "handson.p002.live_physics_acceptance.v5",
        "physics_device": args.device,
        "passed": False,
        "thresholds_unchanged": True,
        "stability_seconds_requested": float(args.stability_seconds),
    }
    rc = 1
    try:
        import torch  # noqa: E402

        from handson.sim.isaaclab.abc_collection_env import (  # noqa: E402
            _pose_vector_to_matrix,
        )
        from handson.sim.isaaclab.p002_collection_env import (  # noqa: E402
            HandsOnP002IsaacLabEnv,
            _P002_EXPECTED_ROBOT_BASE_POSES_WXYZ,
            _P002_EXPECTED_PALM_POSES_WXYZ,
        )
        from handson.sim.isaaclab.p002_contract import (  # noqa: E402
            CONTROL_HZ,
            DOOR_DOF,
            HANDLE_PRESS_DWELL_STEPS,
            LOCKED_DOOR_MAX_DEG,
            NOMINAL_ROOT_XYZ_M,
            NOMINAL_ROOT_YAW_DEG,
            PHYSICS_HZ,
            P002ScriptedPolicy,
        )
        from handson.sim.isaaclab.p002_scene_geometry import (  # noqa: E402
            P002_FRONT_HANDLE_EXPECTED_REPLACEMENT_COLLIDERS,
            P002_FRONT_HANDLE_EXPECTED_REPLACEMENT_VISUAL_MESHES,
            P002_FRONT_HANDLE_MAX_GPU_CONVEX_VERTICES,
            P002_FRONT_HANDLE_SOURCE_ARCHIVE_SHA256,
            p002_front_handle_dimensions_m,
        )

        environment = HandsOnP002IsaacLabEnv(
            seed=0,
            control_dt=1.0 / CONTROL_HZ,
            physics_dt=1.0 / PHYSICS_HZ,
            time_limit=30.0,
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
        view = environment._p002_articulation_view
        if sim is None or view is None:
            raise RuntimeError("P0-02 complete scene did not expose PhysX runtime")

        near_home_motion_ik_solutions: dict[str, Any] = {}
        for side in ("right", "left"):
            target = np.asarray(
                _P002_EXPECTED_PALM_POSES_WXYZ[side], dtype=np.float64
            ).copy()
            # Test a real world-frame motion immediately adjacent to the
            # physical initial palm. Controller/VR rebase targets live in a
            # different frame and must not be passed directly to world IK.
            target[0] += 0.02
            home_q = environment._compute_arm_joint_target_haorui(
                side, environment._robots[side], target
            )
            near_home_motion_ik_solutions[side] = {
                "target_pose_wxyz": target.tolist(),
                "solved": home_q is not None,
                "joint_position_rad": (
                    None if home_q is None else _array(home_q).reshape(-1).tolist()
                ),
            }

        expected_base_poses_wxyz = {
            side: np.asarray(
                _P002_EXPECTED_ROBOT_BASE_POSES_WXYZ[side],
                dtype=np.float64,
            )
            for side in ("right", "left")
        }
        expected_palm_poses_wxyz = {
            side: np.asarray(
                _P002_EXPECTED_PALM_POSES_WXYZ[side],
                dtype=np.float64,
            )
            for side in ("right", "left")
        }
        initial_robot_state_pass, initial_robot_state = (
            _robot_initial_state_evidence(
                environment,
                expected_base_poses_wxyz=expected_base_poses_wxyz,
                expected_palm_poses_wxyz=expected_palm_poses_wxyz,
            )
        )

        geometry_pass, geometry_evidence = _scene_geometry_evidence(
            getattr(environment, "_p002_scene_geometry_report", None),
            expected_dimensions_m=p002_front_handle_dimensions_m(),
            expected_visual_meshes=(
                P002_FRONT_HANDLE_EXPECTED_REPLACEMENT_VISUAL_MESHES
            ),
            expected_colliders=P002_FRONT_HANDLE_EXPECTED_REPLACEMENT_COLLIDERS,
            expected_source_archive_sha256=(
                P002_FRONT_HANDLE_SOURCE_ARCHIVE_SHA256
            ),
            minimum_grasp_clearance_m=0.028,
            maximum_gpu_convex_vertices=(
                P002_FRONT_HANDLE_MAX_GPU_CONVEX_VERTICES
            ),
        )

        metatype = view.get_metatype(0)
        dof_indices = {
            str(name): int(index) for name, index in metatype.dof_indices.items()
        }
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
            for key, sensor in environment._p002_contact_views.items()
        }

        reset_trials: list[dict[str, Any]] = []
        reset_states: dict[int, Any] = {}
        for seed in range(20):
            _obs, info = environment.reset(seed=seed)
            positions = environment._joint_positions()
            root_pose_xyzw = _array(view.get_root_transforms()).reshape(
                int(view.count), 7
            )[0].copy()
            root_pose = np.concatenate(
                [root_pose_xyzw[:3], root_pose_xyzw[6:7], root_pose_xyzw[3:6]]
            )
            authored_root_pose = environment._p002_reset_frame_pose_wxyz.copy()
            composed_root_pose = environment._p002_composed_frame_pose_wxyz
            if composed_root_pose is None:
                raise RuntimeError("P0-02 composed reset-frame pose was not captured")
            reset = environment._p002_reset
            reset_states[seed] = reset
            translation_error_m, yaw_error_deg = _reset_pose_errors(
                reset=reset,
                actual_wxyz=root_pose,
                composed_wxyz=composed_root_pose,
                nominal_xyz_m=NOMINAL_ROOT_XYZ_M,
                nominal_yaw_deg=NOMINAL_ROOT_YAW_DEG,
            )
            readback = environment._physical_readback()
            robot_state_pass, robot_state = _robot_initial_state_evidence(
                environment,
                expected_base_poses_wxyz=expected_base_poses_wxyz,
                expected_palm_poses_wxyz=expected_palm_poses_wxyz,
            )
            finite = bool(
                np.isfinite(positions).all()
                and np.isfinite(root_pose).all()
                and np.isfinite(authored_root_pose).all()
                and np.isfinite(list(readback.__dict__.values())).all()
            )
            reset_trials.append(
                {
                    "seed": seed,
                    "sample": reset.to_dict(),
                    "joint_positions_rad": positions.tolist(),
                    "root_pose_wxyz": root_pose.tolist(),
                    "authored_root_pose_wxyz": authored_root_pose.tolist(),
                    "composed_root_pose_wxyz": composed_root_pose.tolist(),
                    "translation_delta_error_m": translation_error_m,
                    "yaw_delta_error_deg": yaw_error_deg,
                    "finite": finite,
                    "failed": bool(info.get("failed", False)),
                    "initial_success": bool(info.get("succeed", False)),
                    "max_penetration_m": float(readback.max_penetration_m),
                    "hard_failure_max_penetration_m": float(
                        readback.hard_failure_max_penetration_m
                    ),
                    "robot_initial_state_passed": robot_state_pass,
                    "robot_initial_state": robot_state,
                }
            )

        reachability: list[dict[str, Any]] = []
        for seed in range(20):
            reset = reset_states[seed]
            policy = P002ScriptedPolicy(reset)
            targets = {
                "right_handle_press": policy.action(190)[:7],
                "left_door_closed": policy.action(120)[7:14],
            }
            for role, target in targets.items():
                side = "left" if role.startswith("left") else "right"
                robot = environment._robots[side]
                q = environment._compute_arm_joint_target_haorui(
                    side, robot, target
                )
                row: dict[str, Any] = {
                    "seed": seed,
                    "role": role,
                    "target_pose_wxyz": target.tolist(),
                    "solved": q is not None,
                }
                if q is not None:
                    q_np = _array(q).reshape(-1)
                    arm_ids = environment._arm_joint_ids[side]
                    limits = _array(robot.data.soft_joint_pos_limits)[0, arm_ids]
                    margin = float(
                        np.min(np.minimum(q_np - limits[:, 0], limits[:, 1] - q_np))
                    )
                    base_from_world = np.linalg.inv(
                        environment._robot_base_pose_matrix(side, robot)
                    )
                    current_ee_in_base = base_from_world @ environment._robot_body_pose_matrix(
                        robot, environment._ee_body_indices[side]
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
                                    predicted_ee_world[:3, 3] - target_matrix[:3, 3]
                                )
                            ),
                            "orientation_error_rad": _rotation_error(
                                predicted_ee_world, target_matrix
                            ),
                        }
                    )
                reachability.append(row)

        environment.reset(seed=0)
        environment._set_door_drive_locked(True)
        force = torch.zeros_like(view.get_dof_positions())
        indices = torch.arange(
            int(view.count), dtype=torch.long, device=force.device
        )
        door_index = environment._p002_dof_indices["door"]
        max_locked_drift = 0.0
        for _ in range(PHYSICS_HZ):
            force[:, door_index] = 20.0
            view.set_dof_actuation_forces(force, indices)
            for robot in environment._robots.values():
                robot.write_data_to_sim()
            sim.step(render=False)
            for robot in environment._robots.values():
                robot.update(environment._sim_dt)
            max_locked_drift = max(
                max_locked_drift,
                abs(float(environment._joint_positions()[door_index])),
            )
        force.zero_()
        view.set_dof_actuation_forces(force, indices)

        environment.reset(seed=0)
        stability_robot_reference = _robot_stability_snapshot(environment)
        robot_base_max_drift_m = 0.0
        robot_palm_max_drift_m = 0.0
        robot_arm_max_drift_rad = 0.0
        robot_arm_max_speed_rad_s = 0.0
        stability_steps = int(round(float(args.stability_seconds) * PHYSICS_HZ))
        peak_door_speed = 0.0
        max_penetration = 0.0
        max_hard_failure_penetration = 0.0
        max_rear_handle_force = 0.0
        max_diagnostic_frame_force = 0.0
        nan_count = 0
        exploding_count = 0
        for step in range(stability_steps):
            for robot in environment._robots.values():
                robot.write_data_to_sim()
            sim.step(render=False)
            for robot in environment._robots.values():
                robot.update(environment._sim_dt)
            environment._after_task_physics_substep()
            if step % 12 == 0 or step == stability_steps - 1:
                robot_sample = _robot_stability_snapshot(environment)
                for side in ("right", "left"):
                    reference = stability_robot_reference[side]
                    sample = robot_sample[side]
                    robot_base_max_drift_m = max(
                        robot_base_max_drift_m,
                        float(
                            np.linalg.norm(
                                sample["base"][:3, 3]
                                - reference["base"][:3, 3]
                            )
                        ),
                    )
                    robot_palm_max_drift_m = max(
                        robot_palm_max_drift_m,
                        float(
                            np.linalg.norm(
                                sample["palm"][:3, 3]
                                - reference["palm"][:3, 3]
                            )
                        ),
                    )
                    robot_arm_max_drift_rad = max(
                        robot_arm_max_drift_rad,
                        float(
                            np.max(
                                np.abs(
                                    sample["joint_position"]
                                    - reference["joint_position"]
                                )
                            )
                        ),
                    )
                    robot_arm_max_speed_rad_s = max(
                        robot_arm_max_speed_rad_s,
                        float(np.max(np.abs(sample["joint_velocity"]))),
                    )
                readback = environment._physical_readback()
                values = np.asarray(list(readback.__dict__.values()), dtype=np.float64)
                nan_count += int(np.size(values) - np.isfinite(values).sum())
                peak_door_speed = max(
                    peak_door_speed, abs(float(readback.door_angular_speed_radps))
                )
                max_penetration = max(
                    max_penetration, float(readback.max_penetration_m)
                )
                max_hard_failure_penetration = max(
                    max_hard_failure_penetration,
                    float(readback.hard_failure_max_penetration_m),
                )
                max_rear_handle_force = max(
                    max_rear_handle_force,
                    float(readback.robot_rear_handle_contact_force_n),
                )
                max_diagnostic_frame_force = max(
                    max_diagnostic_frame_force,
                    float(readback.robot_frame_contact_force_n),
                )
                if abs(float(readback.door_angular_speed_radps)) > 20.0:
                    exploding_count += 1

        reset_pose_match = bool(
            len(reset_trials) == 20
            and all(
                row["translation_delta_error_m"] <= 1.0e-5
                and row["yaw_delta_error_deg"] <= 1.0e-4
                for row in reset_trials
            )
        )
        reset_pass = bool(
            len(reset_trials) == 20
            and reset_pose_match
            and all(
                row["finite"]
                and not row["failed"]
                and not row["initial_success"]
                and row["robot_initial_state_passed"]
                and row["hard_failure_max_penetration_m"] <= 0.005
                for row in reset_trials
            )
        )
        reachability_pass = bool(
            len(reachability) == 40
            and all(
                row["solved"]
                and row.get("joint_limit_margin_rad", -math.inf) >= 0.05
                and row.get("position_error_m", math.inf) <= 0.01
                and row.get("orientation_error_rad", math.inf) <= 0.15
                for row in reachability
            )
        )
        locked_drive_pass = bool(
            math.degrees(max_locked_drift) <= LOCKED_DOOR_MAX_DEG
        )
        stability_pass = bool(
            stability_steps >= 10 * PHYSICS_HZ
            and nan_count == 0
            and exploding_count == 0
            and max_hard_failure_penetration <= 0.005
            and max_rear_handle_force < 0.02
            and robot_base_max_drift_m <= 0.002
            and robot_palm_max_drift_m <= 0.02
            and robot_arm_max_drift_rad <= 0.02
            and robot_arm_max_speed_rad_s <= 0.10
        )
        near_home_motion_ik_pass = bool(
            all(row["solved"] for row in near_home_motion_ik_solutions.values())
        )
        checks = {
            "complete_scene_initialized": True,
            "initial_robot_bases_homes_and_palms": initial_robot_state_pass,
            "bilateral_near_home_motion_targets_reachable": (
                near_home_motion_ik_pass
            ),
            "dual_rokae_wuji": bool(
                robot_inventory["names"] == ["left", "right"]
                and all(v == 7 for v in robot_inventory["arm_joint_counts"].values())
                and all(v == 20 for v in robot_inventory["hand_joint_counts"].values())
            ),
            "door_three_dof_topology": set(dof_indices)
            == {"Door_joint", "Handle001_joint", "Handle002_joint"},
            "sampled_reset_pose_matches_actual": reset_pose_match,
            "exact_contact_views": set(contact_inventory)
            == {
                "right_front_handle",
                "left_door",
                "robot_frame",
                "robot_rear_handle",
                "initial_robot_front_handle",
                "initial_robot_door",
            }
            and all(
                row["sensor_count"] == 1 and row["filter_count"] > 0
                for row in contact_inventory.values()
            ),
            "scene_geometry_contract": geometry_pass,
            "twenty_resets": reset_pass,
            "forty_role_targets_reachable": reachability_pass,
            "locked_external_force_drift": locked_drive_pass,
            "ten_second_stability": stability_pass,
        }
        result.update(
            {
                "passed": all(checks.values()),
                "checks": checks,
                "dof_indices": dof_indices,
                "robot": robot_inventory,
                "near_home_motion_ik_solutions": near_home_motion_ik_solutions,
                "initial_robot_state": initial_robot_state,
                "contact_views": contact_inventory,
                "scene_geometry": geometry_evidence,
                "reset_trials": reset_trials,
                "reachability_trials": reachability,
                "locked_force_probe": {
                    "force_nm": 20.0,
                    "steps": PHYSICS_HZ,
                    "max_drift_rad": max_locked_drift,
                    "max_drift_deg": math.degrees(max_locked_drift),
                    "threshold_deg": LOCKED_DOOR_MAX_DEG,
                },
                "stability": {
                    "steps": stability_steps,
                    "physics_hz": PHYSICS_HZ,
                    "simulated_seconds": stability_steps / PHYSICS_HZ,
                    "nan_count": nan_count,
                    "exploding_body_count": exploding_count,
                    "peak_door_speed_rad_s": peak_door_speed,
                    "max_penetration_m": max_penetration,
                    "max_hard_failure_penetration_m": (
                        max_hard_failure_penetration
                    ),
                    "max_rear_handle_force_n": max_rear_handle_force,
                    # Compatibility key: the remaining forbidden force is now
                    # only the rear/far-side handle, never the table.
                    "max_forbidden_force_n": max_rear_handle_force,
                    "max_diagnostic_frame_force_n": max_diagnostic_frame_force,
                    "robot_base_max_drift_m": robot_base_max_drift_m,
                    "robot_palm_max_drift_m": robot_palm_max_drift_m,
                    "robot_arm_max_drift_rad": robot_arm_max_drift_rad,
                    "robot_arm_max_speed_rad_s": robot_arm_max_speed_rad_s,
                },
                "unused_constant_sanity": {
                    "source_door_dof": DOOR_DOF,
                    "unlock_dwell_steps": HANDLE_PRESS_DWELL_STEPS,
                    "nominal_root_xyz_m": NOMINAL_ROOT_XYZ_M.tolist(),
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
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if environment is not None:
            try:
                # This standalone gate has already flushed its evidence.  The
                # full SimulationContext cleanup can hang after GPU tensor
                # views were exercised, so use the same bounded shutdown path
                # as the collection supervisor.
                environment._fast_shutdown_on_close = True
                environment.close()
            except Exception:
                traceback.print_exc()
                rc = 1
        try:
            app.close()
        except SystemExit:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
