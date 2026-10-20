#!/usr/bin/env python3
"""Attempt B320-BC005 using only public bimanual robot actions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from handson.sim.isaaclab.abc_collection_env import (
    _C_TOOL_WUJI_POWER_GRIP,
    _matrix_to_quat_wxyz,
)
from handson.sim.isaaclab.loaded_tray_gate_env import (
    HandsOnLoadedTrayGateIsaacLabEnv,
)
from handson.sim.isaaclab.loaded_tray_gate_scene import (
    GATE_PLANE_X,
    LEFT_GRIP_LOCAL_POSITION,
    RIGHT_GRIP_LOCAL_POSITION,
    TRAY_SCRIPTED_PLACED_CENTER_X,
    TRAY_START_POSITION,
    tray_local_position_to_world,
)


PALM_ABOVE_HANDLE_M = 0.060
_VIDEO_CAMERA_KEYS = {
    "front": "main",
    "random_camera": "main",
    "wrist_left": "wrist_left",
    "wrist_right": "wrist_right",
}
_EVIDENCE_INFO_KEYS = (
    "task_stage",
    "task_stage_index",
    "left_role_owned",
    "right_role_owned",
    "left_tray_contact_force_n",
    "right_tray_contact_force_n",
    "left_handle_distance_m",
    "right_handle_distance_m",
    "payloads_on_tray",
    "tray_lift_clearance_m",
    "tray_min_x",
    "tray_max_x",
    "target_error_m",
    "tray_tilt_deg",
    "tray_linear_speed_mps",
    "tray_angular_speed_radps",
    "all_beyond_gate",
    "final_stable",
    "success_counter",
    "success_steps_required",
    "gate_contact_force_n",
    "task_gate_contact_force_n",
    "hand_gate_contact_force_n",
    "left_hand_gate_contact_force_n",
    "right_hand_gate_contact_force_n",
    "max_gate_penetration_m",
    "succeed",
    "failed",
    "termination_reason",
    "failure_reason",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class _EvidenceRecorder:
    """Write three synchronized MP4 streams plus one state row per frame."""

    def __init__(self, video_dir: Path | None, video_fps: int):
        self.enabled = video_dir is not None
        self.video_fps = int(video_fps)
        self.frame_count = 0
        self._closed = False
        self._writers: dict[str, Any] = {}
        self.video_dir = None if video_dir is None else video_dir.resolve()
        self.state_path: Path | None = None
        self._state_stream = None
        if not self.enabled:
            return
        if self.video_fps <= 0:
            raise ValueError("--video-fps must be positive")
        assert self.video_dir is not None
        self.video_dir.mkdir(parents=True, exist_ok=True)
        paths = [
            self.video_dir / "main.mp4",
            self.video_dir / "wrist_left.mp4",
            self.video_dir / "wrist_right.mp4",
            self.video_dir / "frame_state.jsonl",
        ]
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite recording evidence: " + ", ".join(existing)
            )
        import imageio.v2 as imageio

        for key in ("main", "wrist_left", "wrist_right"):
            self._writers[key] = imageio.get_writer(
                self.video_dir / f"{key}.mp4",
                fps=self.video_fps,
            )
        self.state_path = self.video_dir / "frame_state.jsonl"
        self._state_stream = self.state_path.open("x", encoding="utf-8")

    def append(
        self,
        observation: dict[str, Any],
        phase: str,
        env_frame: int,
        info: dict[str, Any],
        env: HandsOnLoadedTrayGateIsaacLabEnv,
    ) -> None:
        if not self.enabled:
            return
        images = observation.get("images", {})
        encoded: dict[str, np.ndarray] = {}
        for source_key, canonical_key in _VIDEO_CAMERA_KEYS.items():
            if source_key not in images:
                continue
            encoded[canonical_key] = np.asarray(images[source_key])
        missing = sorted(set(self._writers) - set(encoded))
        if missing:
            raise RuntimeError(f"Recording observation lacks cameras: {missing}")
        for key, writer in self._writers.items():
            frame = encoded[key]
            if frame.ndim != 3 or frame.shape[-1] < 3:
                raise RuntimeError(f"Invalid {key} frame shape {frame.shape}")
            writer.append_data(np.asarray(frame[..., :3], dtype=np.uint8))

        right_actual, left_actual = env.get_end_effector_pose_matrix()
        state_row = {
            "video_frame_index": self.frame_count,
            "env_frame": int(env_frame),
            "control_time_s": float(env_frame) * float(env._control_dt),
            "phase": str(phase),
            "tray_pose_wxyz": np.asarray(
                env._poses["loaded_tray"], dtype=np.float64
            ).reshape(7).tolist(),
            "can_pose_wxyz": np.asarray(
                env._poses["loaded_tray_can"], dtype=np.float64
            ).reshape(7).tolist(),
            "cup_pose_wxyz": np.asarray(
                env._poses["loaded_tray_cup"], dtype=np.float64
            ).reshape(7).tolist(),
            "right_palm_position": np.asarray(
                right_actual[:3, 3], dtype=np.float64
            ).tolist(),
            "left_palm_position": np.asarray(
                left_actual[:3, 3], dtype=np.float64
            ).tolist(),
            "right_command_pose_wxyz": np.asarray(
                env._right_tcp_pose, dtype=np.float64
            ).reshape(7).tolist(),
            "left_command_pose_wxyz": np.asarray(
                env._left_tcp_pose, dtype=np.float64
            ).reshape(7).tolist(),
            "task_info": {
                key: _json_value(info[key])
                for key in _EVIDENCE_INFO_KEYS
                if key in info
            },
        }
        assert self._state_stream is not None
        self._state_stream.write(json.dumps(state_row, sort_keys=True) + "\n")
        self.frame_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for writer in self._writers.values():
            try:
                writer.close()
            except BaseException as exc:
                errors.append(exc)
        if self._state_stream is not None:
            try:
                self._state_stream.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"Failed to finalize recording evidence: {errors[0]}")

    def summary(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        assert self.video_dir is not None and self.state_path is not None
        files = {}
        for name in ("main.mp4", "wrist_left.mp4", "wrist_right.mp4"):
            path = self.video_dir / name
            files[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
            }
        files[self.state_path.name] = {
            "path": str(self.state_path),
            "bytes": self.state_path.stat().st_size,
        }
        return {
            "enabled": True,
            "fps": self.video_fps,
            "frames": self.frame_count,
            "files": files,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--isaac-device", default=None)
    parser.add_argument("--approach-height", type=float, default=0.060)
    parser.add_argument("--grip-scale", type=float, default=1.0)
    parser.add_argument("--left-command-bias-x", type=float, default=0.0)
    parser.add_argument("--left-command-bias-y", type=float, default=0.0)
    parser.add_argument("--left-command-bias-z", type=float, default=0.0)
    parser.add_argument("--direct-reset-grasp-only", action="store_true")
    parser.add_argument("--direct-reset-transport", action="store_true")
    parser.add_argument("--direct-approach-frames", type=int, default=30)
    parser.add_argument("--direct-inward-m", type=float, default=0.0)
    parser.add_argument("--direct-delta-x", type=float, default=0.0)
    parser.add_argument("--direct-delta-z", type=float, default=0.0)
    parser.add_argument("--transport-inward-m", type=float, default=0.045)
    parser.add_argument("--transport-grasp-slide-x-m", type=float, default=0.0)
    parser.add_argument("--transport-lift-m", type=float, default=0.075)
    parser.add_argument("--transport-lift-frames", type=int, default=120)
    parser.add_argument("--transport-crossing-hand-x-offset-m", type=float, default=0.0)
    parser.add_argument("--transport-crossing-inward-m", type=float, default=0.0)
    parser.add_argument("--transport-crossing-frames", type=int, default=120)
    parser.add_argument("--transport-crossing-settle-frames", type=int, default=0)
    parser.add_argument("--transport-wrist-roll-frames", type=int, default=0)
    parser.add_argument("--transport-wrist-roll-before-crossing", action="store_true")
    parser.add_argument(
        "--transport-final-center-x",
        type=float,
        default=float(TRAY_SCRIPTED_PLACED_CENTER_X),
    )
    parser.add_argument("--transport-final-hand-x-offset-m", type=float, default=0.0)
    parser.add_argument("--transport-clear-frames", type=int, default=55)
    parser.add_argument("--transport-clear-waypoints", type=int, default=1)
    parser.add_argument("--transport-clear-right-local-roll-deg", type=float, default=0.0)
    parser.add_argument("--transport-clear-left-local-roll-deg", type=float, default=0.0)
    parser.add_argument("--transport-final-right-local-roll-deg", type=float, default=None)
    parser.add_argument("--transport-final-left-local-roll-deg", type=float, default=None)
    parser.add_argument("--transport-final-roll-frames", type=int, default=0)
    parser.add_argument("--transport-release-left-before-clear", action="store_true")
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--result-json", type=Path, default=None)
    return parser.parse_args()


def _pose(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(position, dtype=np.float64).reshape(3),
            np.asarray(quat_wxyz, dtype=np.float64).reshape(4),
        ]
    )


def _normalized_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("quaternion must be finite and nonzero")
    return quat / norm


def _multiply_quat_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = _normalized_quat_wxyz(left)
    rw, rx, ry, rz = _normalized_quat_wxyz(right)
    return _normalized_quat_wxyz(
        np.asarray(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float64,
        )
    )


def _with_local_roll(quat_wxyz: np.ndarray, roll_degrees: float) -> np.ndarray:
    half_angle = math.radians(float(roll_degrees)) * 0.5
    local_roll = np.asarray(
        [math.cos(half_angle), math.sin(half_angle), 0.0, 0.0],
        dtype=np.float64,
    )
    return _multiply_quat_wxyz(quat_wxyz, local_roll)


def _slerp_quat_wxyz(
    start_wxyz: np.ndarray,
    target_wxyz: np.ndarray,
    alpha: float,
) -> np.ndarray:
    start = _normalized_quat_wxyz(start_wxyz)
    target = _normalized_quat_wxyz(target_wxyz)
    dot = float(np.dot(start, target))
    if dot < 0.0:
        target = -target
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalized_quat_wxyz((1.0 - alpha) * start + alpha * target)
    angle = math.acos(dot)
    scale = math.sin(angle)
    return _normalized_quat_wxyz(
        math.sin((1.0 - alpha) * angle) / scale * start
        + math.sin(alpha * angle) / scale * target
    )


def _public_action(
    env: HandsOnLoadedTrayGateIsaacLabEnv,
    right_pose: np.ndarray,
    left_pose: np.ndarray,
    right_hand: np.ndarray,
    left_hand: np.ndarray,
) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[:7] = right_pose
    action[7:14] = left_pose
    action[14 : 14 + env._hand_action_dim] = right_hand
    action[14 + env._hand_action_dim :] = left_hand
    return action


def _emit_result(
    result: dict[str, Any],
    result_json: Path | None,
    recorder: _EvidenceRecorder | None = None,
) -> None:
    if recorder is not None:
        recorder.close()
        result["recording"] = recorder.summary()
    result_text = json.dumps(result, indent=2, sort_keys=True)
    if result_json is not None:
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(result_text + "\n", encoding="utf-8")
    print(result_text, flush=True)


def _hand_body_points(
    env: HandsOnLoadedTrayGateIsaacLabEnv,
    side: str,
) -> dict[str, list[float]]:
    robot = env._robots.get(side)
    if robot is None:
        return {}
    points: dict[str, list[float]] = {}
    try:
        body_names = list(robot.body_names)
        body_poses = robot.data.body_pose_w[0]
        for index, name in enumerate(body_names):
            normalized = str(name).lower()
            if "palm" not in normalized and "finger" not in normalized:
                continue
            point = body_poses[int(index), :3]
            if hasattr(point, "detach"):
                point = point.detach().cpu().numpy()
            point_np = np.asarray(point, dtype=np.float64).reshape(3)
            if np.all(np.isfinite(point_np)):
                points[str(name)] = point_np.tolist()
    except Exception:
        return {}
    return points


def _rpy_degrees(quat_wxyz: np.ndarray) -> list[float]:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw]).tolist()


def _arm_joint_state(
    env: HandsOnLoadedTrayGateIsaacLabEnv,
    side: str,
) -> dict[str, Any]:
    try:
        robot = env._robots[side]
        joint_ids = env._arm_joint_ids[side]
        actual = robot.data.joint_pos[0, joint_ids]
        target = env._joint_targets[side][0, joint_ids]
        limits = robot.data.soft_joint_pos_limits[0, joint_ids]
        if hasattr(actual, "detach"):
            actual = actual.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            limits = limits.detach().cpu().numpy()
        actual_np = np.asarray(actual, dtype=np.float64).reshape(-1)
        target_np = np.asarray(target, dtype=np.float64).reshape(-1)
        limits_np = np.asarray(limits, dtype=np.float64).reshape(-1, 2)
        margin = np.minimum(
            actual_np - limits_np[:, 0],
            limits_np[:, 1] - actual_np,
        )
        return {
            "actual": actual_np.tolist(),
            "target": target_np.tolist(),
            "minimum_soft_limit_margin_rad": float(np.min(margin)),
        }
    except Exception as exc:
        return {"diagnostic_error": str(exc)}


def _snapshot(
    env: HandsOnLoadedTrayGateIsaacLabEnv,
    phase: str,
    frame: int,
    info: dict[str, Any],
) -> dict[str, Any]:
    right_actual, left_actual = env.get_end_effector_pose_matrix()
    tray_pose = np.asarray(env._poses["loaded_tray"], dtype=np.float64).reshape(7)
    snapshot = {
        "phase": phase,
        "frame": frame,
        "stage": str(info["task_stage"]),
        "left_role_owned": bool(info["left_role_owned"]),
        "right_role_owned": bool(info["right_role_owned"]),
        "left_contact_force_n": float(info["left_tray_contact_force_n"]),
        "right_contact_force_n": float(info["right_tray_contact_force_n"]),
        "left_handle_distance_m": float(info["left_handle_distance_m"]),
        "right_handle_distance_m": float(info["right_handle_distance_m"]),
        "tray_lift_clearance_m": float(info["tray_lift_clearance_m"]),
        "tray_min_x": float(info["tray_min_x"]),
        "tray_max_x": float(info["tray_max_x"]),
        "target_error_m": float(info["target_error_m"]),
        "tray_tilt_deg": float(info["tray_tilt_deg"]),
        "payloads_on_tray": bool(info["payloads_on_tray"]),
        "tilt_stabilizer_active": bool(
            info.get("tilt_stabilizer_active", False)
        ),
        "tilt_stabilizer_torque_nm": float(
            info.get("tilt_stabilizer_torque_nm", 0.0)
        ),
        "tilt_stabilizer_max_torque_nm": float(
            info.get("tilt_stabilizer_max_torque_nm", 0.0)
        ),
        "transport_stabilizer_active": bool(
            info.get("transport_stabilizer_active", False)
        ),
        "transport_stabilizer_force_n": float(
            info.get("transport_stabilizer_force_n", 0.0)
        ),
        "transport_stabilizer_max_force_n": float(
            info.get("transport_stabilizer_max_force_n", 0.0)
        ),
        "crossing_nullspace_active": bool(
            info.get("crossing_nullspace_active", False)
        ),
        "crossing_nullspace_max_step_rad": float(
            info.get("crossing_nullspace_max_step_rad", 0.0)
        ),
        "all_beyond_gate": bool(info["all_beyond_gate"]),
        "max_gate_penetration_m": float(info["max_gate_penetration_m"]),
        "gate_contact_force_n": float(info["gate_contact_force_n"]),
        "task_gate_contact_force_n": float(
            info.get("task_gate_contact_force_n", info["gate_contact_force_n"])
        ),
        "hand_gate_contact_force_n": float(
            info.get("hand_gate_contact_force_n", 0.0)
        ),
        "left_hand_gate_contact_force_n": float(
            info.get("left_hand_gate_contact_force_n", 0.0)
        ),
        "right_hand_gate_contact_force_n": float(
            info.get("right_hand_gate_contact_force_n", 0.0)
        ),
        "final_stable": bool(info["final_stable"]),
        "success_counter": int(info["success_counter"]),
        "succeed": bool(info["succeed"]),
        "failed": bool(info.get("failed", False)),
        "termination_reason": str(info.get("termination_reason", "")),
        "tray_pose_wxyz": tray_pose.tolist(),
        "tray_rpy_deg": _rpy_degrees(tray_pose[3:7]),
        "can_pose_wxyz": np.asarray(
            env._poses["loaded_tray_can"], dtype=np.float64
        ).reshape(7).tolist(),
        "cup_pose_wxyz": np.asarray(
            env._poses["loaded_tray_cup"], dtype=np.float64
        ).reshape(7).tolist(),
        "right_arm_joint_state": _arm_joint_state(env, "right"),
        "left_arm_joint_state": _arm_joint_state(env, "left"),
        "right_hand_command": np.asarray(
            env._right_hand, dtype=np.float64
        ).tolist(),
        "left_hand_command": np.asarray(
            env._left_hand, dtype=np.float64
        ).tolist(),
        "right_hand_actual": np.asarray(
            env._actual_hand_joint_pose("right"), dtype=np.float64
        ).tolist(),
        "left_hand_actual": np.asarray(
            env._actual_hand_joint_pose("left"), dtype=np.float64
        ).tolist(),
        "right_hand_body_points": _hand_body_points(env, "right"),
        "left_hand_body_points": _hand_body_points(env, "left"),
    }
    for side, actual, command in (
        ("right", right_actual, env._right_tcp_pose),
        ("left", left_actual, env._left_tcp_pose),
    ):
        actual_position = np.asarray(actual[:3, 3], dtype=np.float64)
        command_position = np.asarray(command[:3], dtype=np.float64)
        snapshot[f"{side}_palm_position"] = actual_position.tolist()
        snapshot[f"{side}_command_position"] = command_position.tolist()
        snapshot[f"{side}_command_quat_wxyz"] = np.asarray(
            command[3:7], dtype=np.float64
        ).tolist()
        snapshot[f"{side}_palm_rotation_matrix"] = np.asarray(
            actual[:3, :3], dtype=np.float64
        ).reshape(-1).tolist()
        snapshot[f"{side}_command_position_error_m"] = float(
            np.linalg.norm(actual_position - command_position)
        )
    return snapshot


def main() -> int:
    args = _parse_args()
    recorder = _EvidenceRecorder(args.video_dir, args.video_fps)
    env = HandsOnLoadedTrayGateIsaacLabEnv(
        launch_isaac=True,
        headless=True,
        image_obs=recorder.enabled,
        randomize=False,
        hz=0,
        time_limit=45.0,
        isaac_device=args.isaac_device,
    )
    trace: list[dict[str, Any]] = []
    failure = ""
    succeeded = False
    last_info: dict[str, Any] = {}
    frame = 0

    open_hand = np.zeros(env._hand_action_dim, dtype=np.float64)
    grip_hand = np.clip(
        np.asarray(_C_TOOL_WUJI_POWER_GRIP, dtype=np.float64)
        * float(args.grip_scale),
        -1.0,
        1.0,
    )

    right_handle = np.asarray(
        tray_local_position_to_world(
            TRAY_START_POSITION, RIGHT_GRIP_LOCAL_POSITION
        ),
        dtype=np.float64,
    )
    left_handle = np.asarray(
        tray_local_position_to_world(
            TRAY_START_POSITION, LEFT_GRIP_LOCAL_POSITION
        ),
        dtype=np.float64,
    )
    grasp_z = float(left_handle[2]) + float(args.approach_height)
    left_command_bias = np.asarray(
        [
            args.left_command_bias_x,
            args.left_command_bias_y,
            args.left_command_bias_z,
        ],
        dtype=np.float64,
    )
    def run_segment(
        phase: str,
        right_target: np.ndarray,
        left_target: np.ndarray,
        right_hand_target: np.ndarray,
        left_hand_target: np.ndarray,
        frames: int,
        *,
        success_allowed: bool = False,
    ) -> None:
        nonlocal frame, failure, succeeded, last_info
        right_start = env._right_tcp_pose.copy()
        left_start = env._left_tcp_pose.copy()
        right_hand_start = env._right_hand.copy()
        left_hand_start = env._left_hand.copy()
        for local_frame in range(1, frames + 1):
            alpha = float(local_frame) / float(frames)
            right_command = (1.0 - alpha) * right_start + alpha * right_target
            left_command = (1.0 - alpha) * left_start + alpha * left_target
            right_command[3:7] = _slerp_quat_wxyz(
                right_start[3:7], right_target[3:7], alpha
            )
            left_command[3:7] = _slerp_quat_wxyz(
                left_start[3:7], left_target[3:7], alpha
            )
            right_hand_command = (
                (1.0 - alpha) * right_hand_start + alpha * right_hand_target
            )
            left_hand_command = (
                (1.0 - alpha) * left_hand_start + alpha * left_hand_target
            )
            _observation, _reward, terminated, truncated, info = env.step(
                _public_action(
                    env,
                    right_command,
                    left_command,
                    right_hand_command,
                    left_hand_command,
                )
            )
            frame += 1
            last_info = info
            recorder.append(_observation, phase, frame, info, env)
            if local_frame == frames or local_frame % 10 == 0 or terminated:
                trace.append(_snapshot(env, phase, frame, info))
            if info.get("failed", False):
                failure = str(info.get("termination_reason") or "task_failure")
                return
            if truncated:
                failure = "truncated"
                return
            if info["succeed"]:
                succeeded = True
                if not success_allowed:
                    failure = f"unexpected_success_during_{phase}"
                return
            if terminated:
                failure = str(info.get("termination_reason") or "terminated")
                return

    try:
        _observation, reset_info = env.reset(seed=args.seed)
        last_info = reset_info
        recorder.append(_observation, "reset", frame, reset_info, env)
        trace.append(_snapshot(env, "reset", frame, reset_info))

        if args.direct_reset_grasp_only or args.direct_reset_transport:
            right_actual, left_actual = env.get_end_effector_pose_matrix()
            direct_offset_right = np.asarray(
                [args.direct_delta_x, args.direct_inward_m, args.direct_delta_z],
                dtype=np.float64,
            )
            direct_offset_left = np.asarray(
                [args.direct_delta_x, -args.direct_inward_m, args.direct_delta_z],
                dtype=np.float64,
            )
            right_direct_target = _pose(
                np.asarray(right_actual[:3, 3], dtype=np.float64)
                + direct_offset_right,
                _matrix_to_quat_wxyz(right_actual[:3, :3]),
            )
            left_direct_target = _pose(
                np.asarray(left_actual[:3, 3], dtype=np.float64)
                + direct_offset_left,
                _matrix_to_quat_wxyz(left_actual[:3, :3]),
            )
            approach_frames = max(0, int(args.direct_approach_frames))
            has_direct_offset = bool(
                np.linalg.norm(direct_offset_right) > 0.0
                or np.linalg.norm(direct_offset_left) > 0.0
            )
            if has_direct_offset:
                for local_frame in range(1, approach_frames + 1):
                    alpha = float(local_frame) / float(approach_frames)
                    right_command = _pose(
                        (1.0 - alpha) * right_actual[:3, 3]
                        + alpha * right_direct_target[:3],
                        right_direct_target[3:7],
                    )
                    left_command = _pose(
                        (1.0 - alpha) * left_actual[:3, 3]
                        + alpha * left_direct_target[:3],
                        left_direct_target[3:7],
                    )
                    _observation, _reward, terminated, truncated, info = env.step(
                        _public_action(
                            env,
                            right_command,
                            left_command,
                            open_hand,
                            open_hand,
                        )
                    )
                    frame += 1
                    last_info = info
                    recorder.append(
                        _observation,
                        "direct_reset_approach",
                        frame,
                        info,
                        env,
                    )
                    if (
                        local_frame % 10 == 0
                        or local_frame == approach_frames
                        or terminated
                    ):
                        trace.append(
                            _snapshot(env, "direct_reset_approach", frame, info)
                        )
                    if terminated or truncated:
                        break
            else:
                right_direct_target = np.zeros(7, dtype=np.float64)
                left_direct_target = np.zeros(7, dtype=np.float64)
            for local_frame in range(1, 86):
                alpha = min(1.0, float(local_frame) / 60.0)
                hand_command = alpha * grip_hand
                _observation, _reward, terminated, truncated, info = env.step(
                    _public_action(
                        env,
                        right_direct_target,
                        left_direct_target,
                        hand_command,
                        hand_command,
                    )
                )
                frame += 1
                last_info = info
                recorder.append(
                    _observation,
                    "direct_reset_grasp",
                    frame,
                    info,
                    env,
                )
                if local_frame % 10 == 0 or local_frame == 85 or terminated:
                    trace.append(_snapshot(env, "direct_reset_grasp", frame, info))
                if terminated or truncated:
                    break
            grasp_passed = bool(
                last_info["left_role_owned"] and last_info["right_role_owned"]
            )
            if args.direct_reset_grasp_only:
                result = {
                    "case": "public_action_direct_reset_grasp",
                    "seed": args.seed,
                    "status": "PASS" if grasp_passed else "FAIL",
                    "probe_passed": grasp_passed,
                    "task_succeeded": bool(last_info["succeed"]),
                    "failure": (
                        "" if grasp_passed else "dual_role_ownership_not_established"
                    ),
                    "frames": frame,
                    "final": _snapshot(env, "final", frame, last_info),
                    "trace": trace,
                }
                _emit_result(result, args.result_json, recorder)
                return 0 if grasp_passed else 1

            if last_info.get("failed", False):
                failure = str(
                    last_info.get("termination_reason") or "task_failure_during_grasp"
                )
            elif not grasp_passed:
                failure = "dual_role_ownership_not_established"

            inward = max(0.0, float(args.transport_inward_m))
            lifted_right = right_direct_target.copy()
            lifted_left = left_direct_target.copy()
            lifted_right[0] += float(args.transport_grasp_slide_x_m)
            lifted_left[0] += float(args.transport_grasp_slide_x_m)
            lifted_right[1] += inward
            lifted_left[1] -= inward
            lifted_right[2] += float(args.transport_lift_m)
            lifted_left[2] += float(args.transport_lift_m)
            if not failure:
                run_segment(
                    "direct_lift_and_tuck",
                    lifted_right,
                    lifted_left,
                    grip_hand,
                    grip_hand,
                    max(1, int(args.transport_lift_frames)),
                )

            wrist_roll_frames = max(0, int(args.transport_wrist_roll_frames))
            roll_before_crossing = bool(args.transport_wrist_roll_before_crossing)
            rolled_lifted_right = lifted_right.copy()
            rolled_lifted_left = lifted_left.copy()
            if roll_before_crossing:
                rolled_lifted_right[3:7] = _with_local_roll(
                    lifted_right[3:7],
                    float(args.transport_clear_right_local_roll_deg),
                )
                rolled_lifted_left[3:7] = _with_local_roll(
                    lifted_left[3:7],
                    float(args.transport_clear_left_local_roll_deg),
                )
                if not failure and wrist_roll_frames > 0:
                    run_segment(
                        "direct_rotate_wrists_before_crossing",
                        rolled_lifted_right,
                        rolled_lifted_left,
                        grip_hand,
                        grip_hand,
                        wrist_roll_frames,
                    )

            crossing_right = rolled_lifted_right.copy()
            crossing_left = rolled_lifted_left.copy()
            crossing_right[0] = float(GATE_PLANE_X) + float(
                args.transport_crossing_hand_x_offset_m
            )
            crossing_left[0] = float(GATE_PLANE_X) + float(
                args.transport_crossing_hand_x_offset_m
            )
            crossing_inward = max(0.0, float(args.transport_crossing_inward_m))
            crossing_right[1] += crossing_inward
            crossing_left[1] -= crossing_inward
            if not failure:
                run_segment(
                    "direct_cross_gate",
                    crossing_right,
                    crossing_left,
                    grip_hand,
                    grip_hand,
                    max(1, int(args.transport_crossing_frames)),
                )
            if not failure and int(args.transport_crossing_settle_frames) > 0:
                run_segment(
                    "direct_crossing_settle",
                    crossing_right,
                    crossing_left,
                    grip_hand,
                    grip_hand,
                    int(args.transport_crossing_settle_frames),
                )

            rolled_right = crossing_right.copy()
            rolled_left = crossing_left.copy()
            if not roll_before_crossing:
                rolled_right[3:7] = _with_local_roll(
                    crossing_right[3:7],
                    float(args.transport_clear_right_local_roll_deg),
                )
                rolled_left[3:7] = _with_local_roll(
                    crossing_left[3:7],
                    float(args.transport_clear_left_local_roll_deg),
                )
                if not failure and wrist_roll_frames > 0:
                    run_segment(
                        "direct_rotate_wrists",
                        rolled_right,
                        rolled_left,
                        grip_hand,
                        grip_hand,
                        wrist_roll_frames,
                    )

            placed_right = rolled_right.copy()
            placed_left = rolled_left.copy()
            if args.transport_final_right_local_roll_deg is not None:
                placed_right[3:7] = _with_local_roll(
                    lifted_right[3:7],
                    float(args.transport_final_right_local_roll_deg),
                )
            if args.transport_final_left_local_roll_deg is not None:
                placed_left[3:7] = _with_local_roll(
                    lifted_left[3:7],
                    float(args.transport_final_left_local_roll_deg),
                )
            final_roll_frames = max(0, int(args.transport_final_roll_frames))
            if not failure and final_roll_frames > 0:
                final_rolled_right = placed_right.copy()
                final_rolled_left = placed_left.copy()
                final_rolled_right[0] = rolled_right[0]
                final_rolled_left[0] = rolled_left[0]
                run_segment(
                    "direct_rotate_wrists_before_clear",
                    final_rolled_right,
                    final_rolled_left,
                    grip_hand,
                    grip_hand,
                    final_roll_frames,
                )
            placed_right[0] = float(args.transport_final_center_x) + float(
                args.transport_final_hand_x_offset_m
            )
            placed_left[0] = float(args.transport_final_center_x) + float(
                args.transport_final_hand_x_offset_m
            )
            clear_frames = max(1, int(args.transport_clear_frames))
            clear_waypoints = max(1, int(args.transport_clear_waypoints))
            clear_left_hand = (
                open_hand
                if args.transport_release_left_before_clear
                else grip_hand
            )
            for waypoint_index in range(1, clear_waypoints + 1):
                if failure or succeeded:
                    break
                waypoint_alpha = float(waypoint_index) / float(clear_waypoints)
                waypoint_right = placed_right.copy()
                waypoint_left = placed_left.copy()
                waypoint_right[0] = (
                    (1.0 - waypoint_alpha) * rolled_right[0]
                    + waypoint_alpha * placed_right[0]
                )
                waypoint_left[0] = (
                    (1.0 - waypoint_alpha) * rolled_left[0]
                    + waypoint_alpha * placed_left[0]
                )
                waypoint_frames = clear_frames // clear_waypoints
                if waypoint_index <= clear_frames % clear_waypoints:
                    waypoint_frames += 1
                run_segment(
                    (
                        "direct_clear_gate"
                        if clear_waypoints == 1
                        else f"direct_clear_gate_{waypoint_index}_of_{clear_waypoints}"
                    ),
                    waypoint_right,
                    waypoint_left,
                    grip_hand,
                    clear_left_hand,
                    max(1, waypoint_frames),
                    success_allowed=True,
                )

            lowered_right = placed_right.copy()
            lowered_left = placed_left.copy()
            lowered_right[2] = right_direct_target[2]
            lowered_left[2] = left_direct_target[2]
            ready_to_lower = bool(last_info.get("all_beyond_gate", False))
            if not failure and ready_to_lower:
                run_segment(
                    "direct_lower_over_target",
                    lowered_right,
                    lowered_left,
                    grip_hand,
                    clear_left_hand,
                    40,
                    success_allowed=True,
                )
            elif not failure:
                failure = "loaded_tray_not_fully_beyond_gate"
            if not failure and not succeeded:
                run_segment(
                    "direct_stable_hold",
                    lowered_right,
                    lowered_left,
                    grip_hand,
                    clear_left_hand,
                    60,
                    success_allowed=True,
                )
            if not succeeded and not failure:
                failure = "success_not_reached"

            result = {
                "case": "public_action_direct_reset_transport",
                "seed": args.seed,
                "grip_scale": args.grip_scale,
                "transport_inward_m": inward,
                "transport_lift_m": float(args.transport_lift_m),
                "transport_final_center_x": float(args.transport_final_center_x),
                "transport_crossing_settle_frames": int(
                    args.transport_crossing_settle_frames
                ),
                "transport_wrist_roll_frames": wrist_roll_frames,
                "transport_wrist_roll_before_crossing": roll_before_crossing,
                "transport_final_roll_frames": final_roll_frames,
                "transport_release_left_before_clear": bool(
                    args.transport_release_left_before_clear
                ),
                "transport_clear_waypoints": clear_waypoints,
                "transport_clear_local_roll_deg": {
                    "right": float(args.transport_clear_right_local_roll_deg),
                    "left": float(args.transport_clear_left_local_roll_deg),
                },
                "transport_final_local_roll_deg": {
                    "right": (
                        None
                        if args.transport_final_right_local_roll_deg is None
                        else float(args.transport_final_right_local_roll_deg)
                    ),
                    "left": (
                        None
                        if args.transport_final_left_local_roll_deg is None
                        else float(args.transport_final_left_local_roll_deg)
                    ),
                },
                "status": "PASS" if succeeded and not failure else "FAIL",
                "succeeded": succeeded,
                "failure": failure,
                "frames": frame,
                "final": _snapshot(env, "final", frame, last_info),
                "trace": trace,
            }
            _emit_result(result, args.result_json, recorder)
            return 0 if succeeded and not failure else 1

        right_actual, left_actual = env.get_end_effector_pose_matrix()
        right_reset_quat = _matrix_to_quat_wxyz(right_actual[:3, :3])
        left_reset_quat = _matrix_to_quat_wxyz(left_actual[:3, :3])
        right_hold = _pose(right_actual[:3, 3], right_reset_quat)
        left_hold = _pose(left_actual[:3, 3], left_reset_quat)
        right_grasp = _pose(
            np.asarray([right_handle[0], right_handle[1], grasp_z]),
            right_reset_quat,
        )
        left_grasp = _pose(
            np.asarray([left_handle[0], left_handle[1], grasp_z])
            + left_command_bias,
            left_reset_quat,
        )

        run_segment(
            "hold_reset_pose",
            right_hold,
            left_hold,
            open_hand,
            open_hand,
            12,
        )
        pregrasp_z = grasp_z + 0.050
        if not failure:
            run_segment(
                "move_above_handles",
                _pose(
                    np.asarray([right_handle[0], right_handle[1], pregrasp_z]),
                    right_reset_quat,
                ),
                _pose(
                    np.asarray([left_handle[0], left_handle[1], pregrasp_z])
                    + left_command_bias,
                    left_reset_quat,
                ),
                open_hand,
                open_hand,
                45,
            )
        if not failure:
            run_segment(
                "descend_open",
                right_grasp,
                left_grasp,
                open_hand,
                open_hand,
                30,
            )
        if not failure:
            run_segment(
                "close_both_hands",
                right_grasp,
                left_grasp,
                grip_hand,
                grip_hand,
                35,
            )
        if not failure:
            run_segment(
                "grasp_settle",
                right_grasp,
                left_grasp,
                grip_hand,
                grip_hand,
                20,
            )
        if not failure and not (
            last_info["left_role_owned"] and last_info["right_role_owned"]
        ):
            failure = "dual_role_ownership_not_established"

        lifted_right = right_grasp.copy()
        lifted_left = left_grasp.copy()
        lifted_right[2] += 0.075
        lifted_left[2] += 0.075
        if not failure:
            run_segment(
                "lift_loaded_tray",
                lifted_right,
                lifted_left,
                grip_hand,
                grip_hand,
                35,
            )

        crossing_right = lifted_right.copy()
        crossing_left = lifted_left.copy()
        crossing_right[0] = float(GATE_PLANE_X)
        crossing_left[0] = float(GATE_PLANE_X)
        if not failure:
            run_segment(
                "cross_gate_plane",
                crossing_right,
                crossing_left,
                grip_hand,
                grip_hand,
                45,
            )

        placed_right = crossing_right.copy()
        placed_left = crossing_left.copy()
        placed_right[0] = float(TRAY_SCRIPTED_PLACED_CENTER_X)
        placed_left[0] = float(TRAY_SCRIPTED_PLACED_CENTER_X)
        if not failure:
            run_segment(
                "clear_gate",
                placed_right,
                placed_left,
                grip_hand,
                grip_hand,
                45,
            )

        lowered_right = placed_right.copy()
        lowered_left = placed_left.copy()
        lowered_right[2] = grasp_z
        lowered_left[2] = grasp_z
        if not failure:
            run_segment(
                "lower_over_target",
                lowered_right,
                lowered_left,
                grip_hand,
                grip_hand,
                35,
            )
        if not failure:
            run_segment(
                "stable_hold",
                lowered_right,
                lowered_left,
                grip_hand,
                grip_hand,
                60,
                success_allowed=True,
            )
        if not succeeded and not failure:
            failure = "success_not_reached"

        result = {
            "case": "public_action_scripted_positive",
            "seed": args.seed,
            "approach_height_m": args.approach_height,
            "grip_scale": args.grip_scale,
            "left_command_bias_m": left_command_bias.tolist(),
            "status": "PASS" if succeeded and not failure else "FAIL",
            "succeeded": succeeded,
            "failure": failure,
            "frames": frame,
            "final": _snapshot(env, "final", frame, last_info),
            "trace": trace,
        }
        _emit_result(result, args.result_json, recorder)
    finally:
        recorder.close()
        env.close()
    return 0 if succeeded and not failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
