#!/usr/bin/env python3
"""Run P0-05 with only ordinary 54-D bimanual teleoperation actions."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import sys
import traceback

import numpy as np


TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

from isaaclab.app import AppLauncher  # noqa: E402
from handson.sim.isaaclab.p005_contract import (  # noqa: E402
    KEY_AXIS_LOCAL_M,
    KEY_TIP_LOCAL_M,
    KEY_TURN_RELEASE_DEG,
    PADLOCK_KEYHOLE_LOCAL_M,
    PADLOCK_KEYWAY_INWARD_LOCAL_M,
    TASK_NAME,
    insertion_geometry_ok,
)


POWER_GRIP = np.asarray(
    [
        0.60, 0.10, 0.84, 0.70, 0.68,
        0.00, 0.90, 0.78, 0.70, 0.00,
        0.92, 0.80, 0.68, 0.00, 0.90,
        0.78, 0.64, 0.00, 0.86, 0.74,
    ],
    dtype=np.float64,
)
PINCH_GRIP = POWER_GRIP.copy()
PINCH_GRIP[8:] = 0.0
RIGHT_HOME = np.asarray([-0.10, -0.16, 1.07, 0.0, 1.0, 0.0, 0.0])
LEFT_HOME = np.asarray([-0.10, 0.16, 1.07, 0.0, 1.0, 0.0, 0.0])
GRASP_QUAT = np.asarray([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])

# Historical seed-0 grasp calibration; re-calibrate for the single-operable-key
# support pose before treating this policy as current acceptance evidence.
# No task body is written after reset: these
# are solely palm pose targets sent through the normal online IK controller.
KEY_PRE = np.asarray([-0.421, -0.245, 1.029, *GRASP_QUAT])
KEY_GRASP = np.asarray([-0.421, -0.245, 0.959, *GRASP_QUAT])
KEY_MICRO_LIFT = np.asarray([-0.421, -0.245, 0.974, *GRASP_QUAT])
KEY_LIFT = np.asarray([-0.421, -0.245, 1.050, *GRASP_QUAT])
KEY_TRANSIT = np.asarray(
    [-0.350, -0.200, 1.180, 0.2499498400, 0.0812565800, 0.6958873400, 0.6683287100]
)
KEY_APPROACH_HIGH = np.asarray(
    [-0.19026, -0.10300, 1.16000, 0.2499498400, 0.0812565800, 0.6958873400, 0.6683287100]
)
KEY_APPROACH = np.asarray(
    [-0.19026, -0.10300, 1.09500, 0.2499498400, 0.0812565800, 0.6958873400, 0.6683287100]
)
KEY_INSERT = np.asarray(
    [-0.20926, -0.10300, 1.09500, 0.2499498400, 0.0812565800, 0.6958873400, 0.6683287100]
)
KEY_TURN = np.asarray(
    [-0.20926, -0.10300, 1.09500, 0.2513149900, -0.0360710300, 0.7968644300, 0.5482214300]
)
KEY_ALIGN = np.asarray(
    [-0.209260, -0.103000, 1.095000, 0.25192426, 0.01287569, 0.75833690, 0.60107698]
)
KEY_SEATED = np.asarray(
    [-0.213378, -0.110364, 1.095779, 0.25192426, 0.01287569, 0.75833690, 0.60107698]
)
KEY_TURN_AXIS = np.asarray(
    [-0.258553, -0.046793, 1.190733, 0.44305587, -0.26561536, 0.82875764, 0.21519933]
)

# The left target uses the mirrored Wuji thumb/index pinch offset and lifts
# along the padlock's physical shackle slide axis after the unlock latch.  The
# 1.300 m waypoint keeps every open finger above the closed shackle during the
# early, pre-unlock lateral transit.
SHACKLE_PRE = np.asarray([-0.503, -0.052, 1.300, *GRASP_QUAT])
SHACKLE_GRASP = np.asarray([-0.503, -0.052, 1.179, *GRASP_QUAT])
SHACKLE_LIFT = np.asarray([-0.503, -0.052, 1.219, *GRASP_QUAT])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--turn-grip-scale", type=float, default=0.88)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _smooth(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _blend(start: np.ndarray, end: np.ndarray, value: float) -> np.ndarray:
    result = np.asarray(start) + _smooth(value) * (np.asarray(end) - np.asarray(start))
    result[3:7] /= np.linalg.norm(result[3:7])
    return result


def _segment(step: int, begin: int, end: int, start: np.ndarray, target: np.ndarray):
    return _blend(start, target, (step - begin) / float(end - begin))


def _slerp_wxyz(start: np.ndarray, target: np.ndarray, value: float) -> np.ndarray:
    value = _smooth(value)
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    start /= np.linalg.norm(start)
    target /= np.linalg.norm(target)
    cosine = float(np.dot(start, target))
    if cosine < 0.0:
        target = -target
        cosine = -cosine
    if cosine > 0.9995:
        result = start + value * (target - start)
        return result / np.linalg.norm(result)
    theta = math.acos(float(np.clip(cosine, -1.0, 1.0)))
    sine = math.sin(theta)
    return (
        math.sin((1.0 - value) * theta) / sine * start
        + math.sin(value * theta) / sine * target
    )


def _pose_interpolate(start: np.ndarray, target: np.ndarray, value: float) -> np.ndarray:
    blend = _smooth(value)
    return np.concatenate(
        [
            np.asarray(start[:3]) + blend * (np.asarray(target[:3]) - np.asarray(start[:3])),
            _slerp_wxyz(start[3:7], target[3:7], value),
        ]
    )


def _rotation_from_quat_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
             (matrix[0, 2] - matrix[2, 0]) / scale,
             (matrix[1, 0] - matrix[0, 1]) / scale]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [(matrix[0, 2] - matrix[2, 0]) / scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                [(matrix[1, 0] - matrix[0, 1]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale]
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return a world-frame Rodrigues rotation about ``axis``."""

    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]],
         [axis[2], 0.0, -axis[0]],
         [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    identity = np.eye(3, dtype=np.float64)
    return (
        identity
        + math.sin(angle_rad) * cross
        + (1.0 - math.cos(angle_rad)) * (cross @ cross)
    )


def _rotation_aligning_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the shortest rotation that maps ``source`` onto ``target``."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= 1e-10:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        basis = np.zeros(3, dtype=np.float64)
        basis[int(np.argmin(np.abs(source)))] = 1.0
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis)
        return _axis_angle_rotation(axis, math.pi)
    skew = np.asarray(
        [[0.0, -cross[2], cross[1]],
         [cross[2], 0.0, -cross[0]],
         [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * (
        (1.0 - cosine) / (sine * sine)
    )


def _axis_angle_from_rotation(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a unit axis and angle for a small/medium rotation matrix."""

    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1e-9:
        return np.asarray([1.0, 0.0, 0.0]), 0.0
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return np.asarray([1.0, 0.0, 0.0]), angle
    return axis / norm, angle


def _palm_target_for_key(environment, desired_key_pose: np.ndarray) -> np.ndarray:
    """Map a desired key pose to a palm pose from live grasp readback."""

    key_pose = environment._body_pose_wxyz("p005_key_root_body")
    key_rotation = _rotation_from_quat_wxyz(key_pose[3:7])
    palm = environment._robot_body_pose_matrix(
        environment._robots["right"], environment._ee_body_indices["right"]
    )
    palm_rotation = np.asarray(palm[:3, :3], dtype=np.float64)
    palm_position = np.asarray(palm[:3, 3], dtype=np.float64)
    relative_rotation = palm_rotation.T @ key_rotation
    relative_position = palm_rotation.T @ (key_pose[:3] - palm_position)
    desired_key_rotation = _rotation_from_quat_wxyz(desired_key_pose[3:7])
    desired_palm_rotation = desired_key_rotation @ relative_rotation.T
    desired_palm_position = (
        desired_key_pose[:3] - desired_palm_rotation @ relative_position
    )
    return np.concatenate(
        [desired_palm_position, _quat_wxyz_from_rotation(desired_palm_rotation)]
    )


def _palm_target_from_grasp(
    desired_key_pose: np.ndarray,
    relative_rotation: np.ndarray,
    relative_position: np.ndarray,
) -> np.ndarray:
    desired_key_rotation = _rotation_from_quat_wxyz(desired_key_pose[3:7])
    desired_palm_rotation = desired_key_rotation @ relative_rotation.T
    desired_palm_position = (
        desired_key_pose[:3] - desired_palm_rotation @ relative_position
    )
    return np.concatenate(
        [desired_palm_position, _quat_wxyz_from_rotation(desired_palm_rotation)]
    )


def _action(step: int, environment, policy_state: dict) -> np.ndarray:
    right_grip = np.zeros(20)
    if step < 70:
        right = _segment(step, 0, 70, RIGHT_HOME, KEY_PRE)
    elif step < 130:
        right = _segment(step, 70, 130, KEY_PRE, KEY_GRASP)
    elif step < 170:
        right = KEY_GRASP
        right_grip = ((step - 130) / 40.0) * 0.86 * PINCH_GRIP
    elif step < 190:
        right = _segment(step, 170, 190, KEY_GRASP, KEY_MICRO_LIFT)
        right_grip = 0.86 * PINCH_GRIP
    elif step < 310:
        progress = _smooth((step - 190) / 120.0)
        right = _segment(step, 190, 310, KEY_MICRO_LIFT, KEY_LIFT)
        right_grip = (0.86 + 0.01 * progress) * PINCH_GRIP
    elif step < 340:
        right = KEY_LIFT
        right_grip = 0.87 * PINCH_GRIP
    elif step < 460:
        right = _segment(step, 340, 460, KEY_LIFT, KEY_TRANSIT)
        right_grip = 0.87 * PINCH_GRIP
    elif step < 600:
        right = _segment(step, 460, 600, KEY_TRANSIT, KEY_APPROACH_HIGH)
        right_grip = 0.87 * PINCH_GRIP
    elif step < 690:
        right = _segment(step, 600, 690, KEY_APPROACH_HIGH, KEY_APPROACH)
        right_grip = 0.87 * PINCH_GRIP
    elif step < 760:
        right = _segment(step, 690, 760, KEY_APPROACH, KEY_INSERT)
        right_grip = 0.87 * PINCH_GRIP
    elif step < 850:
        right = KEY_INSERT
        right_grip = 0.87 * PINCH_GRIP
    else:
        # First align the key while its tip is 12 mm outside the physical
        # guide, then translate it down the guide axis.  This avoids the
        # non-physical sweep through a solid keyway that the early diagnostic
        # trajectory used when the guide was visual-only.
        if "align_start_key_pose" not in policy_state:
            key_pose = environment._body_pose_wxyz("p005_key_root_body")
            padlock_pose = environment._body_pose_wxyz("p005_padlock_body")
            padlock_rotation = _rotation_from_quat_wxyz(padlock_pose[3:7])
            palm = environment._robot_body_pose_matrix(
                environment._robots["right"],
                environment._ee_body_indices["right"],
            )
            palm_rotation = np.asarray(palm[:3, :3], dtype=np.float64)
            palm_position = np.asarray(palm[:3, 3], dtype=np.float64)
            key_rotation = _rotation_from_quat_wxyz(key_pose[3:7])
            relative_rotation = palm_rotation.T @ key_rotation
            relative_position = palm_rotation.T @ (key_pose[:3] - palm_position)
            inward = padlock_rotation @ PADLOCK_KEYWAY_INWARD_LOCAL_M
            keyhole = padlock_pose[:3] + padlock_rotation @ PADLOCK_KEYHOLE_LOCAL_M
            key_axis = key_rotation @ KEY_AXIS_LOCAL_M
            target_key_rotation = (
                _rotation_aligning_vectors(key_axis, inward) @ key_rotation
            )
            # Fix the roll reference after aligning the shaft. This makes
            # zero turn equal to the padlock's local +Z reference while
            # preserving the side-face insertion axis.
            target_key_rotation = (
                _rotation_aligning_vectors(
                    target_key_rotation @ np.asarray([0.0, 0.0, 1.0]),
                    padlock_rotation @ np.asarray([0.0, 0.0, 1.0]),
                )
                @ target_key_rotation
            )
            target_key_quaternion = _quat_wxyz_from_rotation(target_key_rotation)
            key_tip_offset = target_key_rotation @ KEY_TIP_LOCAL_M
            outside_root = keyhole - 0.030 * inward - key_tip_offset
            seated_root = keyhole + 0.015 * inward - key_tip_offset
            policy_state.update(
                {
                    "align_start_key_pose": np.asarray(key_pose, dtype=np.float64),
                    "outside_key_pose": np.concatenate(
                        [outside_root, target_key_quaternion]
                    ),
                    "seated_key_pose": np.concatenate(
                        [seated_root, target_key_quaternion]
                    ),
                    "turn_axis": inward,
                    "align_relative_rotation": relative_rotation,
                    "align_relative_position": relative_position,
                }
            )
        relative_rotation = policy_state["align_relative_rotation"]
        relative_position = policy_state["align_relative_position"]
        if step < 930:
            desired_key_pose = _pose_interpolate(
                policy_state["align_start_key_pose"],
                policy_state["outside_key_pose"],
                (step - 850) / 80.0,
            )
        elif step < 1010:
            desired_key_pose = policy_state["outside_key_pose"]
        elif step < 1140:
            desired_key_pose = _pose_interpolate(
                policy_state["outside_key_pose"],
                policy_state["seated_key_pose"],
                (step - 1010) / 130.0,
            )
        elif step < 1160:
            desired_key_pose = policy_state["seated_key_pose"]
        else:
            tracker = getattr(environment, "_p005_tracker", None)
            key_turned_seen = bool(
                tracker is not None and tracker.key_turned_seen
            )
            turn_candidate_seen = bool(
                tracker is not None and tracker.turn_steps > 0
            )
            readback = environment._physical_readback()
            release_pose_seen = bool(
                insertion_geometry_ok(readback, release=True)
                and readback.lock_cylinder_angle_deg >= KEY_TURN_RELEASE_DEG
            )
            if (
                (release_pose_seen or turn_candidate_seen or key_turned_seen)
                and "unlock_hold_key_pose" not in policy_state
            ):
                live_key_pose = environment._body_pose_wxyz("p005_key_root_body")
                live_palm = environment._robot_body_pose_matrix(
                    environment._robots["right"],
                    environment._ee_body_indices["right"],
                )
                live_palm_rotation = np.asarray(
                    live_palm[:3, :3], dtype=np.float64
                )
                live_key_rotation = _rotation_from_quat_wxyz(
                    live_key_pose[3:7]
                )
                policy_state["unlock_hold_key_pose"] = np.asarray(
                    live_key_pose, dtype=np.float64
                )
                policy_state["unlock_hold_palm_pose"] = np.concatenate(
                    [
                        np.asarray(live_palm[:3, 3], dtype=np.float64),
                        _quat_wxyz_from_rotation(live_palm_rotation),
                    ]
                )
                policy_state["unlock_hold_relative_rotation"] = (
                    live_palm_rotation.T @ live_key_rotation
                )
                policy_state["unlock_hold_relative_position"] = (
                    live_palm_rotation.T
                    @ (
                        live_key_pose[:3]
                        - np.asarray(live_palm[:3, 3], dtype=np.float64)
                    )
                )
                policy_state["key_position_feedback"] = np.zeros(3)
            if "unlock_hold_key_pose" in policy_state:
                desired_key_pose = policy_state["unlock_hold_key_pose"]
                relative_rotation = policy_state[
                    "unlock_hold_relative_rotation"
                ]
                relative_position = policy_state[
                    "unlock_hold_relative_position"
                ]
            elif "turn_key_pose" not in policy_state:
                live_key_pose = environment._body_pose_wxyz("p005_key_root_body")
                live_palm = environment._robot_body_pose_matrix(
                    environment._robots["right"],
                    environment._ee_body_indices["right"],
                )
                live_palm_rotation = np.asarray(
                    live_palm[:3, :3], dtype=np.float64
                )
                live_palm_position = np.asarray(
                    live_palm[:3, 3], dtype=np.float64
                )
                live_key_rotation = _rotation_from_quat_wxyz(live_key_pose[3:7])
                policy_state["turn_axis"] = np.asarray(
                    policy_state["turn_axis"], dtype=np.float64
                )
                policy_state["turn_key_pose"] = np.asarray(
                    live_key_pose, dtype=np.float64
                )
                policy_state["turn_relative_rotation"] = (
                    live_palm_rotation.T @ live_key_rotation
                )
                policy_state["turn_relative_position"] = live_palm_rotation.T @ (
                    live_key_pose[:3] - live_palm_position
                )
            if "unlock_hold_key_pose" not in policy_state:
                seated_pose = policy_state["turn_key_pose"]
                progress = _smooth((step - 1160) / 160.0)
                turn_rotation = _axis_angle_rotation(
                    policy_state["turn_axis"], math.radians(80.0) * progress
                ) @ _rotation_from_quat_wxyz(seated_pose[3:7])
                desired_key_pose = np.concatenate(
                    [seated_pose[:3], _quat_wxyz_from_rotation(turn_rotation)]
                )
                relative_rotation = policy_state["turn_relative_rotation"]
                relative_position = policy_state["turn_relative_position"]
        commanded_key_pose = np.asarray(desired_key_pose, dtype=np.float64).copy()
        if step >= 1160:
            live_key_pose = environment._body_pose_wxyz("p005_key_root_body")
            desired_rotation = _rotation_from_quat_wxyz(desired_key_pose[3:7])
            live_rotation = _rotation_from_quat_wxyz(live_key_pose[3:7])
            error_axis, error_angle = _axis_angle_from_rotation(
                desired_rotation @ live_rotation.T
            )
            policy_state["max_turn_orientation_error_deg"] = max(
                float(policy_state.get("max_turn_orientation_error_deg", 0.0)),
                math.degrees(error_angle),
            )
            correction_angle = min(error_angle, math.radians(20.0))
            commanded_rotation = (
                _axis_angle_rotation(error_axis, correction_angle)
                @ desired_rotation
            )
            commanded_key_pose[3:7] = _quat_wxyz_from_rotation(
                commanded_rotation
            )
        right = _palm_target_from_grasp(
            commanded_key_pose, relative_rotation, relative_position
        )
        if step >= 930:
            # Slowly integrate only the measured key translation error while
            # the tip is held outside the guide.  The accumulated offset also
            # follows the key through insertion and turn; the frozen grasp
            # orientation avoids rotational positive feedback.
            live_key_pose = environment._body_pose_wxyz("p005_key_root_body")
            error = np.asarray(desired_key_pose[:3] - live_key_pose[:3])
            error_norm = float(np.linalg.norm(error))
            if error_norm > 0.006:
                error *= 0.006 / error_norm
            feedback = np.asarray(
                policy_state.get("key_position_feedback", np.zeros(3)),
                dtype=np.float64,
            )
            feedback += 0.08 * error
            feedback_norm = float(np.linalg.norm(feedback))
            if feedback_norm > 0.020:
                feedback *= 0.020 / feedback_norm
            policy_state["key_position_feedback"] = feedback
            right[:3] += feedback
        if "unlock_hold_palm_pose" in policy_state:
            right = np.asarray(
                policy_state["unlock_hold_palm_pose"], dtype=np.float64
            ).copy()
            policy_state["key_position_feedback"] = np.zeros(3)
        right_grip = (
            0.87 + 0.01 * _smooth((step - 850) / 80.0)
        ) * PINCH_GRIP

    if step >= 1140:
        turn_grip_scale = float(policy_state.get("turn_grip_scale", 0.88))
        grip_scale = 0.88 + (turn_grip_scale - 0.88) * _smooth(
            (step - 1140) / 20.0
        )
        right_grip = grip_scale * PINCH_GRIP

    left_grip = np.zeros(20)
    left_anchor = np.asarray(policy_state["left_anchor_pose"], dtype=np.float64)
    left_safe_up = left_anchor.copy()
    left_safe_up[2] = SHACKLE_PRE[2]
    if step < 1000:
        left = left_anchor
    elif step < 1080:
        left = _segment(step, 1000, 1080, left_anchor, left_safe_up)
    elif step < 1240:
        left = _segment(step, 1080, 1240, left_safe_up, SHACKLE_PRE)
    elif step < 1320:
        left = SHACKLE_PRE
    elif step < 1380:
        if "left_pre_actual_pose" not in policy_state:
            live_left_palm = environment._robot_body_pose_matrix(
                environment._robots["left"],
                environment._ee_body_indices["left"],
            )
            policy_state["left_pre_actual_pose"] = np.concatenate(
                [
                    np.asarray(live_left_palm[:3, 3], dtype=np.float64),
                    _quat_wxyz_from_rotation(live_left_palm[:3, :3]),
                ]
            )
        left = _segment(step, 1320, 1380, SHACKLE_PRE, SHACKLE_GRASP)
    elif step < 1420:
        left = SHACKLE_GRASP
        left_grip = ((step - 1380) / 40.0) * 0.82 * PINCH_GRIP
    elif step < 1470:
        if "left_lift_start_pose" not in policy_state:
            live_left_palm = environment._robot_body_pose_matrix(
                environment._robots["left"],
                environment._ee_body_indices["left"],
            )
            lift_start = np.concatenate(
                [
                    np.asarray(live_left_palm[:3, 3], dtype=np.float64),
                    _quat_wxyz_from_rotation(live_left_palm[:3, :3]),
                ]
            )
            # Retrace the already-executed, feasible descent in reverse.  A
            # pure +Z request from the contact pose lies outside the left-arm
            # IK manifold, while this target supplies 25+ mm of upward travel
            # together with the small inward retreat the arm requires.
            lift_end = np.asarray(
                policy_state["left_pre_actual_pose"], dtype=np.float64
            ).copy()
            policy_state["left_lift_start_pose"] = lift_start
            policy_state["left_lift_end_pose"] = lift_end
        left = _pose_interpolate(
            policy_state["left_lift_start_pose"],
            policy_state["left_lift_end_pose"],
            (step - 1420) / 50.0,
        )
        left_grip = 0.82 * PINCH_GRIP
    else:
        left = np.asarray(policy_state["left_lift_end_pose"], dtype=np.float64)
        left_grip = 0.82 * PINCH_GRIP

    action = np.zeros(54, dtype=np.float64)
    action[:7] = right
    action[7:14] = left
    action[14:34] = right_grip
    action[34:54] = left_grip
    return action


def _contact_force_by_filter(environment, key: str) -> dict[str, float]:
    view = environment._p005_contact_views.get(key)
    path_entry = environment._p005_contact_paths.get(key)
    if view is None or path_entry is None:
        return {}
    try:
        normal_forces, _points, _normals, _separations, counts, starts = (
            view.get_contact_data(environment._sim_dt)
        )
        forces = environment._view_numpy(normal_forces, (-1,))
        counts = environment._view_numpy(
            counts, (int(view.sensor_count), int(view.filter_count))
        ).astype(np.int64)
        starts = environment._view_numpy(starts, counts.shape).astype(np.int64)
        result = {}
        for index, path in enumerate(path_entry[1]):
            count = int(counts[0, index])
            if count <= 0:
                continue
            start = int(starts[0, index])
            force = float(np.sum(np.abs(forces[start:start + count])))
            if force > 1e-4:
                result[str(path).rsplit("/", 1)[-1]] = force
        return result
    except Exception:
        return {}


def _sample(environment, step: int, info: dict) -> dict:
    readback = environment._physical_readback()
    result = {
        "step": int(step),
        "key_pose_wxyz": environment._body_pose_wxyz("p005_key_root_body").tolist(),
        "shackle_pose_wxyz": environment._body_pose_wxyz("p005_shackle").tolist(),
        "right_tcp_target_wxyz": np.asarray(environment._right_tcp_pose).tolist(),
        "left_tcp_target_wxyz": np.asarray(environment._left_tcp_pose).tolist(),
        "readback": {
            "key_insertion_depth_m": readback.key_insertion_depth_m,
            "key_lateral_error_m": readback.key_lateral_error_m,
            "key_axis_error_deg": readback.key_axis_error_deg,
            "lock_cylinder_angle_deg": readback.lock_cylinder_angle_deg,
            "key_physical_turn_deg": info.get("key_physical_turn_deg"),
            "shackle_slide_m": readback.shackle_slide_m,
            "shackle_hinge_deg": readback.shackle_hinge_deg,
            "right_key_contact_force_n": readback.right_key_contact_force_n,
            "left_padlock_contact_force_n": readback.left_padlock_contact_force_n,
            "padlock_lift_m": readback.padlock_lift_m,
            "key_linear_speed_mps": readback.key_linear_speed_mps,
            "key_angular_speed_radps": readback.key_angular_speed_radps,
            "shackle_linear_speed_mps": readback.shackle_linear_speed_mps,
            "shackle_angular_speed_radps": readback.shackle_angular_speed_radps,
            "max_penetration_m": readback.max_penetration_m,
        },
        "metrics": {
            key: info.get(key)
            for key in (
                "phase", "key_inserted_seen", "key_turned_seen",
                "left_padlock_contact_seen", "shackle_open_seen",
                "success_steps", "success", "failed", "exploded",
                "penetration_ok", "robot_table_contact_gating_enabled",
                "key_not_dropped",
                "preunlock_shackle_motion",
            )
        },
        "right_key_contact_force_by_body_n": _contact_force_by_filter(
            environment, "right_key"
        ),
    }
    if step >= 930 or step in {130, 180, 280, 340, 480, 600, 680, 740, 800, 900}:
        for side in ("right", "left"):
            palm = environment._robot_body_pose_matrix(
                environment._robots[side], environment._ee_body_indices[side]
            )
            result[f"{side}_palm_actual_xyz_m"] = palm[:3, 3].tolist()
            result[f"{side}_palm_actual_quat_wxyz"] = _quat_wxyz_from_rotation(
                palm[:3, :3]
            ).tolist()
            result[f"{side}_fingertips_xyz_m"] = {
                name: position.tolist()
                for name, position in environment._wuji_fingertip_positions(side).items()
            }
    return result


def main() -> int:
    args = _parser().parse_args()
    args.headless = True
    args.enable_cameras = False
    args.device = str(getattr(args, "device", None) or "cuda:0")
    os.environ["HANDSON_ISAACLAB_ARM_IK_BACKEND"] = "haorui"
    # Match the normal collection recorder, which requires this gate for every
    # Isaac episode.  Without the flag stabilize_recording_origin() is a
    # deliberate no-op and cannot provide platform-equivalent frame-0 proof.
    os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"
    launcher = AppLauncher(args)
    app = launcher.app
    environment = None
    result = {"schema": "handson.p005.live_end_to_end_acceptance.v1", "passed": False}
    rc = 1
    try:
        from handson.sim.isaaclab.p005_collection_env import HandsOnP005IsaacLabEnv
        from handson.sim.isaaclab.recording_origin import (
            stabilize_recording_origin,
            stable_recording_origin_kwargs_for_task,
        )

        environment = HandsOnP005IsaacLabEnv(
            seed=0,
            time_limit=50.0,
            render_mode="rgb_array",
            image_obs=False,
            randomize=False,
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
        observation, reset_info = environment.reset(seed=0)
        observation, reset_info = stabilize_recording_origin(
            environment,
            observation,
            reset_info,
            **stable_recording_origin_kwargs_for_task(TASK_NAME),
        )
        raw_origin_report = reset_info.get("physical_recording_origin")
        if not isinstance(raw_origin_report, Mapping):
            raise RuntimeError(
                "P0-05 platform-equivalent acceptance requires a successful "
                "physical_recording_origin report"
            )
        origin_report = dict(raw_origin_report)
        runtime_timing = {
            "control_dt_s": float(environment._control_dt),
            "physics_dt_s": float(environment._sim_dt),
            "time_limit_s": float(environment._time_limit),
            "action_step_limit": int(
                round(environment._time_limit / environment._control_dt)
            ),
        }
        samples = []
        left_palm = environment._robot_body_pose_matrix(
            environment._robots["left"],
            environment._ee_body_indices["left"],
        )
        policy_state = {
            "turn_grip_scale": float(args.turn_grip_scale),
            "left_anchor_pose": np.concatenate(
                [
                    np.asarray(left_palm[:3, 3], dtype=np.float64),
                    _quat_wxyz_from_rotation(left_palm[:3, :3]),
                ]
            ),
        }
        peak_right_key = 0.0
        peak_left_padlock = 0.0
        final_info = {}
        final_terminated = False
        final_truncated = False
        for step in range(int(args.steps)):
            _obs, _reward, terminated, truncated, info = environment.step(
                _action(step, environment, policy_state)
            )
            final_info = dict(info)
            final_terminated = bool(terminated)
            final_truncated = bool(truncated)
            readback = environment._physical_readback()
            peak_right_key = max(peak_right_key, readback.right_key_contact_force_n)
            peak_left_padlock = max(peak_left_padlock, readback.left_padlock_contact_force_n)
            if step % 10 == 0 or step == int(args.steps) - 1 or terminated or truncated:
                samples.append(_sample(environment, step, info))
            if terminated or truncated:
                break
        passed = bool(final_info.get("success", False) and not final_info.get("failed", True))
        result.update(
            {
                "passed": passed,
                "executed_steps": int(samples[-1]["step"] + 1),
                "peak_right_key_contact_force_n": peak_right_key,
                "peak_left_padlock_contact_force_n": peak_left_padlock,
                "turn_grip_scale": float(args.turn_grip_scale),
                "alignment_feedback_xyz_m": np.asarray(
                    policy_state.get("key_position_feedback", np.zeros(3)),
                    dtype=np.float64,
                ).tolist(),
                "max_turn_orientation_error_deg": float(
                    policy_state.get("max_turn_orientation_error_deg", 0.0)
                ),
                "runtime_timing": runtime_timing,
                "recording_origin": {
                    key: origin_report.get(key)
                    for key in (
                        "physics_steps",
                        "stability_samples",
                        "post_settle_sleeping",
                        "gate_task_velocity_stable",
                        "gate_task_static_pose_stable",
                        "gate_robot_velocity_stable",
                        "final_max_linear_velocity",
                        "final_max_angular_velocity",
                        "final_max_robot_joint_velocity",
                    )
                },
                "terminated": final_terminated,
                "truncated": final_truncated,
                "failure_snapshot": final_info.get("p005_failure_snapshot"),
                "final_metrics": {
                    key: final_info.get(key)
                    for key in (
                        "phase", "key_inserted_seen", "key_turned_seen",
                        "left_padlock_contact_seen", "shackle_open_seen",
                        "success_steps", "success", "failed", "preunlock_shackle_motion",
                    )
                },
                "samples": samples,
            }
        )
        rc = 0 if passed else 2
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
