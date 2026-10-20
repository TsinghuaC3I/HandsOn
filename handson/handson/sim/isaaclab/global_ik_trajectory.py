"""Pure-Isaac continuous Cartesian replay.

The dataset contract contains absolute wrist poses, hand commands, and timing.
This module plans a continuous Cartesian command curve for the whole episode
and samples it at every PhysX substep.  Isaac Lab's live Jacobian/DLS
controller then computes joint targets from the current simulated state.

No recorded or generated arm joint trajectory is used or persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, RotationSpline


@dataclass(frozen=True)
class GlobalIkPlan:
    substep_action: np.ndarray
    report: Mapping[str, Any]


def _active_pose_slices(env, actions: np.ndarray) -> dict[str, slice]:
    action_dim = int(actions.shape[1])
    hand_dim = int(env._hand_action_dim)
    if action_dim >= 14 + 2 * hand_dim:
        result = {"right": slice(0, 7)}
        if not np.allclose(actions[:, 7:14], 0.0):
            result["left"] = slice(7, 14)
        return result
    if action_dim >= 7 + hand_dim:
        return {"right": slice(0, 7)}
    raise ValueError(
        f"Pure-Isaac replay cannot decode action shape {actions.shape}"
    )


def _continuous_pose_knots(values: np.ndarray, side: str) -> np.ndarray:
    poses = np.asarray(values, dtype=np.float64).copy()
    valid = (
        np.all(np.isfinite(poses), axis=1)
        & (np.linalg.norm(poses[:, 3:7], axis=1) > 1.0e-8)
    )
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        raise ValueError(f"Pure-Isaac replay has no valid {side} wrist poses")

    first = int(valid_indices[0])
    poses[:first] = poses[first]
    previous = poses[first].copy()
    previous[3:7] /= np.linalg.norm(previous[3:7])
    poses[first] = previous
    for frame in range(first + 1, poses.shape[0]):
        if not valid[frame]:
            poses[frame] = previous
            continue
        pose = poses[frame]
        pose[3:7] /= np.linalg.norm(pose[3:7])
        if float(np.dot(previous[3:7], pose[3:7])) < 0.0:
            pose[3:7] *= -1.0
        poses[frame] = pose
        previous = pose.copy()
    return poses


def _sample_times(
    frame_count: int,
    substep_count: int,
    control_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    knot_times = np.arange(frame_count, dtype=np.float64) * control_dt
    sample_times = np.empty(
        (frame_count, substep_count),
        dtype=np.float64,
    )
    for frame in range(frame_count):
        for substep in range(substep_count):
            sample_times[frame, substep] = max(
                0.0,
                (
                    frame
                    - 1.0
                    + float(substep + 1) / float(substep_count)
                )
                * control_dt,
            )
    return knot_times, sample_times


def _interpolate_actions(
    env,
    actions: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    frame_count, action_dim = actions.shape
    substep_count = max(1, int(env._n_substeps))
    control_dt = float(env._control_dt)
    physics_dt = float(env._physics_dt)
    knot_times, sample_times = _sample_times(
        frame_count,
        substep_count,
        control_dt,
    )
    flat_sample_times = sample_times.reshape(-1)

    if frame_count == 1:
        dense = np.repeat(
            actions[:, None, :],
            substep_count,
            axis=1,
        )
    else:
        scalar_curve = PchipInterpolator(
            knot_times,
            actions,
            axis=0,
            extrapolate=False,
        )
        dense = scalar_curve(flat_sample_times).reshape(
            frame_count,
            substep_count,
            action_dim,
        )

    side_reports: dict[str, Any] = {}
    pose_slices = _active_pose_slices(env, actions)
    for side, pose_slice in pose_slices.items():
        poses = _continuous_pose_knots(actions[:, pose_slice], side)
        if frame_count == 1:
            positions = np.repeat(
                poses[:, None, :3],
                substep_count,
                axis=1,
            )
            quaternions = np.repeat(
                poses[:, None, 3:7],
                substep_count,
                axis=1,
            )
        else:
            position_curve = PchipInterpolator(
                knot_times,
                poses[:, :3],
                axis=0,
                extrapolate=False,
            )
            positions = position_curve(flat_sample_times).reshape(
                frame_count,
                substep_count,
                3,
            )
            rotations = Rotation.from_quat(
                poses[:, [4, 5, 6, 3]]
            )
            rotation_curve = RotationSpline(knot_times, rotations)
            xyzw = rotation_curve(flat_sample_times).as_quat().reshape(
                frame_count,
                substep_count,
                4,
            )
            quaternions = xyzw[:, :, [3, 0, 1, 2]]

        dense[:, :, pose_slice.start : pose_slice.start + 3] = positions
        dense[:, :, pose_slice.start + 3 : pose_slice.stop] = quaternions

        flat_positions = positions.reshape(-1, 3)
        linear_speed = (
            np.linalg.norm(np.diff(flat_positions, axis=0), axis=1)
            / max(physics_dt, 1.0e-8)
            if flat_positions.shape[0] > 1
            else np.zeros(1, dtype=np.float64)
        )
        flat_quaternions = quaternions.reshape(-1, 4)
        if flat_quaternions.shape[0] > 1:
            dots = np.abs(
                np.sum(
                    flat_quaternions[:-1] * flat_quaternions[1:],
                    axis=1,
                )
            )
            angles = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
            angular_speed = angles / max(physics_dt, 1.0e-8)
        else:
            angular_speed = np.zeros(1, dtype=np.float64)
        side_reports[side] = {
            "max_linear_speed_m_s": float(np.max(linear_speed)),
            "max_angular_speed_rad_s": float(np.max(angular_speed)),
        }

    max_linear_speed = float(
        os.environ.get(
            "HANDSON_GLOBAL_CARTESIAN_MAX_LINEAR_SPEED_M_S",
            "1.0",
        )
    )
    max_angular_speed = float(
        os.environ.get(
            "HANDSON_GLOBAL_CARTESIAN_MAX_ANGULAR_SPEED_RAD_S",
            "4.0",
        )
    )
    finite = bool(np.all(np.isfinite(dense)))
    speed_valid = all(
        report["max_linear_speed_m_s"] <= max_linear_speed
        and report["max_angular_speed_rad_s"] <= max_angular_speed
        for report in side_reports.values()
    )
    return dense, {
        "mode": "pure_isaac_global_cartesian_substep_dls",
        "frame_count": int(frame_count),
        "physics_substeps_per_control": int(substep_count),
        "control_dt_s": control_dt,
        "physics_dt_s": physics_dt,
        "uses_recorded_joint_state": False,
        "stores_generated_ik": False,
        "ik_backend": "isaaclab_dls",
        "target_physical_replay_success_rate": 0.90,
        "plan_ready_for_physical_replay": bool(finite and speed_valid),
        "limits": {
            "max_linear_speed_m_s": max_linear_speed,
            "max_angular_speed_rad_s": max_angular_speed,
        },
        "sides": side_reports,
    }


def _regularize_global_values(values, *, strength, max_deviation):
    """Solve one whole-path smoothness objective inside a tracking tube."""

    from scipy.sparse import diags, eye
    from scipy.sparse.linalg import spsolve

    source = np.asarray(values, dtype=np.float64)
    if source.shape[0] < 3:
        return source.copy()
    row_count = source.shape[0] - 2
    second_difference = diags(
        (
            np.ones(row_count, dtype=np.float64),
            -2.0 * np.ones(row_count, dtype=np.float64),
            np.ones(row_count, dtype=np.float64),
        ),
        (0, 1, 2),
        shape=(row_count, source.shape[0]),
        format="csc",
    )
    system = (
        eye(source.shape[0], format="csc")
        + float(strength)
        * (second_difference.T @ second_difference)
    )
    optimized = np.column_stack(
        [
            spsolve(system, source[:, dimension])
            for dimension in range(source.shape[1])
        ]
    )
    optimized = source + np.clip(
        optimized - source,
        -float(max_deviation),
        float(max_deviation),
    )
    optimized[0] = source[0]
    optimized[-1] = source[-1]
    return optimized


def _regularize_global_tcp_path(poses, *, active_side_count):
    """Regularize the complete recorded TCP path without moving contact far."""

    from scipy.spatial.transform import Rotation

    result = np.asarray(poses, dtype=np.float64).copy()
    flat = result.reshape(-1, result.shape[2], result.shape[3])
    for side in range(active_side_count):
        flat[:, side, :3] = _regularize_global_values(
            flat[:, side, :3],
            strength=0.08,
            max_deviation=2.5e-4,
        )

        source_quaternion = flat[:, side, 3:7].copy()
        norms = np.linalg.norm(source_quaternion, axis=1, keepdims=True)
        source_quaternion /= np.maximum(norms, 1.0e-12)
        for index in range(1, source_quaternion.shape[0]):
            if np.dot(
                source_quaternion[index - 1],
                source_quaternion[index],
            ) < 0.0:
                source_quaternion[index] *= -1.0

        rotation_vectors = Rotation.from_quat(
            source_quaternion[:, [1, 2, 3, 0]]
        ).as_rotvec()
        rotation_vectors = np.unwrap(rotation_vectors, axis=0)
        smooth_vectors = _regularize_global_values(
            rotation_vectors,
            strength=0.08,
            max_deviation=1.0e-3,
        )
        candidate = Rotation.from_rotvec(smooth_vectors).as_quat()
        candidate = candidate[:, [3, 0, 1, 2]]
        signs = np.sign(
            np.sum(source_quaternion * candidate, axis=1, keepdims=True)
        )
        signs[signs == 0.0] = 1.0
        candidate *= signs
        angular_delta = 2.0 * np.arccos(
            np.clip(
                np.sum(source_quaternion * candidate, axis=1),
                -1.0,
                1.0,
            )
        )
        blend = np.minimum(
            1.0,
            1.0e-3 / np.maximum(angular_delta, 1.0e-12),
        )[:, None]
        smooth_quaternion = (
            (1.0 - blend) * source_quaternion + blend * candidate
        )
        smooth_quaternion /= np.maximum(
            np.linalg.norm(
                smooth_quaternion,
                axis=1,
                keepdims=True,
            ),
            1.0e-12,
        )
        smooth_quaternion[0] = source_quaternion[0]
        smooth_quaternion[-1] = source_quaternion[-1]
        flat[:, side, 3:7] = smooth_quaternion
    return result


def _tcp_path_metrics(poses, *, active_side_count, physics_dt):
    flat = np.asarray(poses, dtype=np.float64).reshape(
        -1,
        poses.shape[2],
        poses.shape[3],
    )
    dt = max(float(physics_dt), 1.0e-9)
    peak_linear_speed = 0.0
    peak_angular_speed = 0.0
    for side in range(active_side_count):
        position_delta = np.diff(flat[:, side, :3], axis=0)
        if position_delta.size:
            peak_linear_speed = max(
                peak_linear_speed,
                float(
                    np.max(np.linalg.norm(position_delta, axis=1)) / dt
                ),
            )
        quaternion = flat[:, side, 3:7].copy()
        quaternion /= np.maximum(
            np.linalg.norm(quaternion, axis=1, keepdims=True),
            1.0e-12,
        )
        if quaternion.shape[0] > 1:
            quaternion_dot = np.abs(
                np.sum(quaternion[:-1] * quaternion[1:], axis=1)
            )
            angular_delta = 2.0 * np.arccos(
                np.clip(quaternion_dot, -1.0, 1.0)
            )
            peak_angular_speed = max(
                peak_angular_speed,
                float(np.max(angular_delta) / dt),
            )
    return {
        "peak_linear_speed_m_s": peak_linear_speed,
        "peak_angular_speed_rad_s": peak_angular_speed,
    }


def _recorded_physx_substep_actions(env, actions, replay, *, limits):
    fields = getattr(replay, "fields", None)
    if fields is None or not callable(getattr(fields, "get", None)):
        return None
    tcp_state_value = fields.get("next_physx_substep_tcp_state")
    tcp_valid_value = fields.get("next_physx_substep_tcp_valid")
    hand_value = fields.get("next_physx_substep_command_hand")
    hand_valid_value = fields.get(
        "next_physx_substep_command_hand_valid"
    )
    count_value = fields.get("next_physx_substep_count")
    if (
        tcp_state_value is None
        or tcp_valid_value is None
        or hand_value is None
        or hand_valid_value is None
        or count_value is None
    ):
        return None

    tcp_states = np.asarray(tcp_state_value, dtype=np.float64)
    tcp_valid = np.asarray(tcp_valid_value, dtype=bool)
    hands = np.asarray(hand_value, dtype=np.float64)
    hand_valid = np.asarray(hand_valid_value, dtype=bool)
    counts = np.asarray(count_value).reshape(len(count_value), -1)
    frame_count, action_dim = actions.shape
    expected_substeps = max(1, int(getattr(env, "_n_substeps", 1)))
    if (
        tcp_states.ndim != 4
        or tcp_valid.ndim != 3
        or hands.ndim != 4
        or hand_valid.ndim != 4
        or tcp_states.shape[0] < frame_count
        or tcp_valid.shape[0] < frame_count
        or hands.shape[0] < frame_count
        or hand_valid.shape[0] < frame_count
        or counts.shape[0] < frame_count
        or tcp_states.shape[1] != expected_substeps
        or tcp_valid.shape[1] != expected_substeps
        or hands.shape[1] != expected_substeps
        or hand_valid.shape[1] != expected_substeps
        or tcp_states.shape[2] < 1
        or hands.shape[2] < 1
        or tcp_states.shape[3] < 7
        or np.any(counts[:frame_count, 0] < expected_substeps)
    ):
        return None

    tcp_states = tcp_states[:frame_count]
    tcp_valid = tcp_valid[:frame_count]
    hands = hands[:frame_count]
    hand_valid = hand_valid[:frame_count]
    if (
        not np.all(np.isfinite(tcp_states[..., :7]))
        or not np.all(np.isfinite(hands))
    ):
        return None

    hand_dim = int(hands.shape[-1])
    if action_dim >= 14 + 2 * hand_dim and tcp_states.shape[2] >= 2:
        active_side_count = 2
    elif action_dim >= 7 + hand_dim:
        active_side_count = 1
    else:
        return None
    if (
        not np.all(tcp_valid[:, :, :active_side_count])
        or not np.all(hand_valid[:, :, :active_side_count])
    ):
        return None

    poses = _regularize_global_tcp_path(
        tcp_states[..., :7],
        active_side_count=active_side_count,
    )
    dense = np.repeat(actions[:, None, :], expected_substeps, axis=1)
    if active_side_count == 2:
        for side in range(2):
            pose_slice = slice(side * 7, (side + 1) * 7)
            hand_start = 14 + side * hand_dim
            hand_slice = slice(hand_start, hand_start + hand_dim)
            dense[:, :, pose_slice] = poses[:, :, side]
            dense[:, :, hand_slice] = hands[:, :, side]
    else:
        dense[:, :, :7] = poses[:, :, 0]
        dense[:, :, 7 : 7 + hand_dim] = hands[:, :, 0]

    physics_dt = float(
        getattr(
            env,
            "_physics_dt",
            getattr(env, "_sim_dt", 1.0 / 120.0),
        )
    )
    metrics = _tcp_path_metrics(
        poses,
        active_side_count=active_side_count,
        physics_dt=physics_dt,
    )
    linear_limit = max(
        1.0,
        float(limits.get("max_linear_speed_m_s", 1.0)),
    )
    angular_limit = max(
        12.0,
        float(limits.get("max_angular_speed_rad_s", 12.0)),
    )
    speed_valid = bool(
        metrics["peak_linear_speed_m_s"] <= linear_limit * 1.05
        and metrics["peak_angular_speed_rad_s"] <= angular_limit * 1.05
    )
    initial_robot_state = all(
        fields.get(name) is not None
        for name in (
            "physx_robot_joint_position",
            "physx_robot_joint_velocity",
            "physx_robot_joint_target",
            "physx_command_tcp_pose",
            "physx_command_hand",
        )
    )
    initial_object_state = any(
        fields.get(name) is not None
        for name in (
            "physx_task_rigid_body_state",
            "physx_task_articulation_root_state",
            "physx_deformable_position",
        )
    )
    contract_complete = bool(
        initial_robot_state
        and initial_object_state
        and speed_valid
    )
    return dense, {
        **metrics,
        "actual_tcp_substep_coverage": 1.0,
        "hand_command_substep_coverage": 1.0,
        "initial_robot_and_controller_state_available": bool(
            initial_robot_state
        ),
        "initial_object_state_available": bool(initial_object_state),
        "speed_limits_satisfied": speed_valid,
        "actual_tcp_linear_speed_limit_m_s": linear_limit,
        "actual_tcp_angular_speed_limit_rad_s": angular_limit,
        "global_position_regularization_max_deviation_m": 2.5e-4,
        "global_orientation_regularization_max_deviation_rad": 1.0e-3,
        "contact_state_written_during_replay": False,
        "meets_theoretical_90pct_prerequisites": contract_complete,
    }


def build_global_ik_plan(env, actions, trace, replay) -> GlobalIkPlan:
    del trace
    action_array = np.asarray(actions, dtype=np.float64)
    if action_array.ndim != 2 or action_array.shape[0] == 0:
        raise ValueError(
            f"Pure-Isaac replay expected non-empty 2D actions, got "
            f"{action_array.shape}"
        )
    substep_action, report = _interpolate_actions(env, action_array)
    recorded_plan = _recorded_physx_substep_actions(
        env,
        action_array,
        replay,
        limits=report.get("limits", {}),
    )
    if recorded_plan is not None:
        substep_action, recorded_report = recorded_plan
        report = dict(report)
        report.update(
            {
                "mode": (
                    "pure_isaac_actual_tcp_substep_global_"
                    "cartesian_dls"
                ),
                "trajectory_source": (
                    "replay.zarr/next_physx_substep_tcp_state"
                ),
                "uses_recorded_physics_substeps": True,
                "uses_recorded_joint_trajectory": False,
                "global_path_optimization": (
                    "whole_trajectory_second_difference_regularization"
                ),
                "theoretical_physical_success_rate_target": 0.90,
                **recorded_report,
            }
        )
        report["plan_ready_for_physical_replay"] = bool(
            recorded_report[
                "meets_theoretical_90pct_prerequisites"
            ]
        )
    else:
        report = dict(report)
        report.update(
            {
                "trajectory_source": "control_rate_pchip_fallback",
                "uses_recorded_physics_substeps": False,
            }
        )
    return GlobalIkPlan(
        substep_action=substep_action,
        report=report,
    )


def _install_isaac_substep_dls(env) -> None:
    if getattr(env, "_global_isaac_substep_dls_installed", False):
        return

    def apply_substep_target(
        substep_index: int,
        substep_count: int,
    ) -> None:
        commands = getattr(env, "_global_isaac_substep_actions", None)
        if commands is None:
            return
        index = min(
            max(int(substep_index), 0),
            min(int(substep_count), int(commands.shape[0])) - 1,
        )
        env._apply_action(commands[index])
        env._apply_joint_targets()

    env._handson_pre_physics_substep_callback = apply_substep_target
    # Do not shadow a task-owned post-substep observer.  BC004 qualifies its
    # continuous airborne hold from physical truth at 120 Hz; action replay
    # must change only the pre-substep command writer and leave that observer
    # active so live recording and replay evaluate the same 360 samples.
    env._global_isaac_substep_dls_installed = True


def apply_global_ik_plan_step(
    env,
    action,
    plan: GlobalIkPlan,
    frame: int,
) -> int:
    del action
    if frame < 0 or frame >= plan.substep_action.shape[0]:
        raise IndexError(
            f"Pure-Isaac replay frame {frame} is outside "
            f"{plan.substep_action.shape[0]} planned frames"
        )
    env._global_isaac_substep_actions = plan.substep_action[frame]
    _install_isaac_substep_dls(env)
    return int(plan.substep_action.shape[2])
