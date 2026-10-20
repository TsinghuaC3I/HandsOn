#!/usr/bin/env python3
"""Inspect and validate B320-BC007 using only normal robot actions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("HANDSON_ENABLE_ISAACSIM_NEWTON", "0")

from handson.sim.isaaclab.bc007_scene import TASK_ID, TASK_NAME  # noqa: E402
from handson.tasks.mappings import CONFIG_MAPPING  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--grasp-probe", action="store_true")
    parser.add_argument(
        "--grasp-palm-offset",
        type=float,
        nargs=3,
        default=(-0.10, 0.0, 0.015),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--grip-scale", type=float, default=0.25)
    parser.add_argument("--probe-lift-m", type=float, default=0.16)
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _unwrap_core(env: Any) -> Any:
    current = env
    for _ in range(12):
        if getattr(current, "task_name", None) == TASK_NAME:
            return current
        child = getattr(current, "env", None)
        if child is None or child is current:
            break
        current = child
    raise RuntimeError("Could not unwrap the BC007 native environment")


def _body_pose_records(core: Any, side: str) -> list[dict[str, Any]]:
    robot = core._robots[side]
    poses = robot.data.body_pose_w[0]
    records = []
    for index, name in enumerate(robot.body_names):
        normalized = str(name).lower()
        if "palm" not in normalized and "finger" not in normalized:
            continue
        pose = poses[index, :7]
        if hasattr(pose, "detach"):
            pose = pose.detach().cpu().numpy()
        records.append({"name": str(name), "pose_isaac": np.asarray(pose)})
    return records


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _joint_limit_snapshot(core: Any) -> dict[str, Any]:
    """Measure controlled-joint margins from the live IsaacLab articulation."""

    result: dict[str, Any] = {"sides": {}}
    all_arm_margins: list[float] = []
    all_hand_margins: list[float] = []
    for side in ("left", "right"):
        robot = core._robots[side]
        positions = _numpy(robot.data.joint_pos[0])
        limits = _numpy(robot.data.soft_joint_pos_limits[0])
        side_result: dict[str, Any] = {}
        for group, ids in (
            ("arm", list(core._arm_joint_ids[side])),
            ("hand", list(core._hand_joint_ids[side])),
        ):
            group_positions = positions[ids]
            group_limits = limits[ids]
            margins = np.minimum(
                group_positions - group_limits[:, 0],
                group_limits[:, 1] - group_positions,
            )
            if not np.all(np.isfinite(margins)):
                raise RuntimeError(f"Non-finite {side} {group} joint-limit margin")
            values = [float(item) for item in margins]
            minimum_index = int(np.argmin(margins))
            side_result[f"{group}_minimum_margin_rad"] = values[minimum_index]
            side_result[f"{group}_worst_joint"] = {
                "joint_index": int(ids[minimum_index]),
                "joint_name": str(robot.joint_names[int(ids[minimum_index])]),
                "margin_rad": values[minimum_index],
                "position_rad": float(group_positions[minimum_index]),
                "lower_limit_rad": float(group_limits[minimum_index, 0]),
                "upper_limit_rad": float(group_limits[minimum_index, 1]),
            }
            side_result[f"{group}_margins_rad"] = values
            if group == "arm":
                all_arm_margins.extend(values)
            else:
                all_hand_margins.extend(values)
        result["sides"][side] = side_result
    result["arm_minimum_margin_rad"] = min(all_arm_margins)
    result["hand_minimum_margin_rad"] = min(all_hand_margins)
    result["controlled_minimum_margin_rad"] = min(
        result["arm_minimum_margin_rad"], result["hand_minimum_margin_rad"]
    )
    for group in ("arm", "hand"):
        candidates = [
            (result["sides"][side][f"{group}_minimum_margin_rad"], side)
            for side in ("left", "right")
        ]
        _, worst_side = min(candidates)
        result[f"{group}_worst_joint"] = {
            "side": worst_side,
            **result["sides"][worst_side][f"{group}_worst_joint"],
        }
    controlled_group = min(
        ("arm", "hand"),
        key=lambda group: result[f"{group}_minimum_margin_rad"],
    )
    result["controlled_worst_joint"] = {
        "group": controlled_group,
        **result[f"{controlled_group}_worst_joint"],
    }
    return result


def _tag_joint_limit_snapshot(snapshot: dict[str, Any], action_index: int) -> None:
    for key in ("arm_worst_joint", "hand_worst_joint", "controlled_worst_joint"):
        snapshot[key]["action_index"] = int(action_index)
    for side in ("left", "right"):
        for group in ("arm", "hand"):
            snapshot["sides"][side][f"{group}_worst_joint"]["action_index"] = int(
                action_index
            )


def _tcp_tracking_errors(core: Any, action: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side, offset in (("right", 0), ("left", 7)):
        measured_matrix = core._robot_link_pose_matrix(side)
        if measured_matrix is None:
            raise RuntimeError(f"Missing measured {side} palm pose")
        measured = core._pose_matrix_to_vector(measured_matrix)
        commanded = np.asarray(action[offset : offset + 7], dtype=np.float64)
        q_commanded = commanded[3:7] / max(
            float(np.linalg.norm(commanded[3:7])), 1.0e-12
        )
        q_measured = measured[3:7] / max(
            float(np.linalg.norm(measured[3:7])), 1.0e-12
        )
        result[side] = {
            "position_error_m": float(np.linalg.norm(measured[:3] - commanded[:3])),
            "orientation_error_rad": float(
                2.0 * np.arccos(np.clip(abs(np.dot(q_measured, q_commanded)), 0.0, 1.0))
            ),
        }
    return result


def _diagnostic(core: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
    state = observation["state"]
    metrics = core._bc007_metrics(update=False)
    result: dict[str, Any] = {
        "commanded_tcp_pose": np.asarray(state["tcp_pose"]),
        "rod_pose": np.asarray(state["rod_pose"]),
        "rod_endpoints": np.asarray(state["rod_endpoints"]).reshape(2, 3),
        "metrics": metrics,
        "sides": {},
    }
    for side in ("left", "right"):
        palm = core._robot_link_pose_matrix(side)
        result["sides"][side] = {
            "palm_matrix": palm,
            "hand_points": core._a06_hand_points(side),
            "body_poses": _body_pose_records(core, side),
        }
    return result


def _core_action(
    right_pose: np.ndarray,
    left_pose: np.ndarray,
    right_hand: np.ndarray,
    left_hand: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [right_pose, left_pose, right_hand, left_hand]
    ).astype(np.float32)


def _interpolate_pose(start: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    result = np.asarray(start, dtype=np.float64).copy()
    result[:3] = (1.0 - fraction) * start[:3] + fraction * target[:3]
    result[3:7] = target[3:7]
    return result


def _grasp_probe_actions(
    core: Any,
    observation: Mapping[str, Any],
    palm_offset: np.ndarray,
    grip_scale: float,
    lift_m: float,
) -> list[np.ndarray]:
    endpoints = np.asarray(observation["state"]["rod_endpoints"], dtype=np.float64).reshape(2, 3)
    starts: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        matrix = core._robot_link_pose_matrix(side)
        if matrix is None:
            raise RuntimeError(f"Missing measured {side} palm pose")
        starts[side] = core._pose_matrix_to_vector(matrix)

    grasps = {
        "left": starts["left"].copy(),
        "right": starts["right"].copy(),
    }
    grasps["left"][:3] = endpoints[0] + palm_offset
    grasps["right"][:3] = endpoints[1] + palm_offset
    above = {
        side: pose.copy()
        for side, pose in grasps.items()
    }
    above["left"][2] += 0.080
    above["right"][2] += 0.080
    open_hand = np.zeros(20, dtype=np.float64)
    closed_hand = (
        core._tool_power_grip_pose(open_hand)
        * float(np.clip(grip_scale, 0.0, 1.0))
    )
    lifted = {side: pose.copy() for side, pose in grasps.items()}
    lifted["left"][2] += max(0.0, float(lift_m))
    lifted["right"][2] += max(0.0, float(lift_m))
    actions: list[np.ndarray] = []

    def hold(
        right_pose: np.ndarray,
        left_pose: np.ndarray,
        right_hand: np.ndarray,
        left_hand: np.ndarray,
        frames: int,
    ) -> None:
        action = _core_action(right_pose, left_pose, right_hand, left_hand)
        actions.extend(action.copy() for _ in range(frames))

    def move(
        right_from: np.ndarray,
        right_to: np.ndarray,
        left_from: np.ndarray,
        left_to: np.ndarray,
        right_hand_from: np.ndarray,
        right_hand_to: np.ndarray,
        left_hand_from: np.ndarray,
        left_hand_to: np.ndarray,
        frames: int,
    ) -> None:
        for index in range(1, frames + 1):
            fraction = index / frames
            actions.append(
                _core_action(
                    _interpolate_pose(right_from, right_to, fraction),
                    _interpolate_pose(left_from, left_to, fraction),
                    (1.0 - fraction) * right_hand_from + fraction * right_hand_to,
                    (1.0 - fraction) * left_hand_from + fraction * left_hand_to,
                )
            )

    hold(starts["right"], starts["left"], open_hand, open_hand, 10)
    move(
        starts["right"], above["right"], starts["left"], above["left"],
        open_hand, open_hand, open_hand, open_hand, 30,
    )
    move(
        above["right"], grasps["right"], above["left"], grasps["left"],
        open_hand, open_hand, open_hand, open_hand, 25,
    )
    move(
        grasps["right"], grasps["right"], grasps["left"], grasps["left"],
        open_hand, closed_hand, open_hand, closed_hand, 12,
    )
    hold(grasps["right"], grasps["left"], closed_hand, closed_hand, 2)
    move(
        grasps["right"], lifted["right"], grasps["left"], lifted["left"],
        closed_hand, closed_hand, closed_hand, closed_hand, 35,
    )
    hold(lifted["right"], lifted["left"], closed_hand, closed_hand, 30)
    return actions


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "action_validation_summary.json"
    started = time.time()
    summary: dict[str, Any] = {
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "host": os.environ.get("TASKDEV_HOST", ""),
        "observed_hostname": socket.gethostname(),
        "physical_gpu": os.environ.get("TASKDEV_PHYSICAL_GPU", ""),
        "device": args.device,
        "seed": args.seed,
        "diagnostic_only": bool(args.diagnostic_only),
        "grasp_probe": bool(args.grasp_probe),
        "grasp_palm_offset": list(args.grasp_palm_offset),
        "grip_scale": float(args.grip_scale),
        "probe_lift_m": float(args.probe_lift_m),
        "pass": False,
    }
    env = None
    try:
        config = CONFIG_MAPPING[TASK_NAME]()
        env = config.get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=False,
            sim_backend="isaacsim",
            seed=args.seed,
            image_obs=False,
            randomize_dynamics=False,
            hand_type="wuji1",
            arm_type="rokae",
            launch_isaac=True,
            headless=True,
            isaac_device=args.device,
            physics_backend="physx",
        )
        core = _unwrap_core(env)
        core._fast_shutdown_on_close = True
        _, reset_info = env.reset(seed=args.seed)
        observation = core._compute_observation()
        summary["wrapper_action_shape"] = list(env.action_space.shape)
        summary["core_action_shape"] = list(core.action_space.shape)
        summary["reset_info"] = reset_info
        summary["initial"] = _diagnostic(core, observation)
        if args.diagnostic_only:
            summary["pass"] = True
        else:
            actions = (
                _grasp_probe_actions(
                    core,
                    observation,
                    np.asarray(args.grasp_palm_offset, dtype=np.float64),
                    args.grip_scale,
                    args.probe_lift_m,
                )
                if args.grasp_probe
                else core.automatic_success_actions(observation)
            )
            terminal = False
            truncated = False
            final_info: Mapping[str, Any] = {}
            peak_write_count = int(core._trajectory_state_write_count)
            minimum_joint_margins = _joint_limit_snapshot(core)
            _tag_joint_limit_snapshot(minimum_joint_margins, -1)
            trace: list[dict[str, Any]] = []
            for index, action in enumerate(actions):
                _, _, terminal, truncated, final_info = core.step(action)
                current_joint_margins = _joint_limit_snapshot(core)
                _tag_joint_limit_snapshot(current_joint_margins, index)
                for key in (
                    "arm_minimum_margin_rad",
                    "hand_minimum_margin_rad",
                    "controlled_minimum_margin_rad",
                ):
                    if current_joint_margins[key] < minimum_joint_margins[key]:
                        minimum_joint_margins[key] = current_joint_margins[key]
                        minimum_joint_margins[
                            key.replace("minimum_margin_rad", "worst_joint")
                        ] = current_joint_margins[
                            key.replace("minimum_margin_rad", "worst_joint")
                        ]
                for side in ("left", "right"):
                    for group in ("arm", "hand"):
                        key = f"{group}_minimum_margin_rad"
                        if (
                            current_joint_margins["sides"][side][key]
                            < minimum_joint_margins["sides"][side][key]
                        ):
                            minimum_joint_margins["sides"][side][key] = (
                                current_joint_margins["sides"][side][key]
                            )
                            minimum_joint_margins["sides"][side][
                                f"{group}_worst_joint"
                            ] = current_joint_margins["sides"][side][
                                f"{group}_worst_joint"
                            ]
                peak_write_count = max(
                    peak_write_count, int(core._trajectory_state_write_count)
                )
                if index % 10 == 0 or terminal or truncated:
                    metrics = core._bc007_last_metrics
                    trace.append(
                        {
                            "action_index": index,
                            "stage": metrics.get("stage_name", ""),
                            "rod_pose": np.asarray(core._poses["bc007_long_rod"]),
                            "left_role_owned": metrics.get("left_role_owned", False),
                            "right_role_owned": metrics.get("right_role_owned", False),
                            "left_grasp_owner": metrics.get("left_grasp_owner", False),
                            "right_grasp_owner": metrics.get("right_grasp_owner", False),
                            "left_owner_continuity_ok": metrics.get(
                                "left_owner_continuity_ok", False
                            ),
                            "right_owner_continuity_ok": metrics.get(
                                "right_owner_continuity_ok", False
                            ),
                            "left_contact_force_n": metrics.get("left_contact_force_n", 0.0),
                            "right_contact_force_n": metrics.get("right_contact_force_n", 0.0),
                            "yaw_progress_deg": metrics.get("yaw_progress_deg", 0.0),
                            "target_position_error_m": metrics.get("target_position_error_m", 0.0),
                            "table_contact_force_n": metrics.get("table_contact_force_n", 0.0),
                            "center_height_m": metrics.get("center_height_m", 0.0),
                            "left_role_distance_m": metrics.get("left_role_distance_m", 0.0),
                            "right_role_distance_m": metrics.get("right_role_distance_m", 0.0),
                            "failure_reason": metrics.get("failure_reason", ""),
                            "succeed": bool(final_info.get("succeed", False)),
                            "terminal": bool(terminal),
                        }
                    )
                if terminal or truncated:
                    break
            final_observation = core._compute_observation()
            final_metrics = core._bc007_last_metrics or core._bc007_metrics(update=False)
            summary.update(
                {
                    "action_count_planned": len(actions),
                    "action_count_executed": index + 1,
                    "trace": trace,
                    "final": _diagnostic(core, final_observation),
                    "final_info": final_info,
                    "terminated": bool(terminal),
                    "truncated": bool(truncated),
                    "trajectory_state_write_count": peak_write_count,
                    "grasp_events": list(core._bc007_grasp_events),
                    "grasp_errors": list(core._bc007_grasp_errors),
                    "joint_frame_alignment_error_m": dict(
                        core._bc007_joint_frame_alignment_error_m
                    ),
                    "joint_limit_margins": minimum_joint_margins,
                    "final_tcp_command_tracking": _tcp_tracking_errors(core, action),
                    "reachability": {
                        "target_fraction": 1.0 if final_info.get("succeed", False) else 0.0,
                        "min_joint_limit_margin_rad": minimum_joint_margins[
                            "arm_minimum_margin_rad"
                        ],
                        "max_assigned_grasp_position_error_m": max(
                            float(final_metrics.get("left_owner_position_error_m", 1.0e9)),
                            float(final_metrics.get("right_owner_position_error_m", 1.0e9)),
                        ),
                        "max_assigned_grasp_orientation_error_rad": max(
                            float(final_metrics.get("left_owner_angle_error_rad", 1.0e9)),
                            float(final_metrics.get("right_owner_angle_error_rad", 1.0e9)),
                        ),
                    },
                }
            )
            if args.grasp_probe:
                summary["pass"] = bool(
                    final_metrics.get("dual_grasp_latched", False)
                    and (
                        args.probe_lift_m <= 0.0
                        or final_metrics.get("lift_latched", False)
                    )
                    and final_metrics.get("left_role_owned", False)
                    and final_metrics.get("right_role_owned", False)
                    and not final_metrics.get("failed", False)
                    and not truncated
                    and peak_write_count == 0
                )
            else:
                summary["pass"] = bool(
                    final_info.get("succeed", False)
                    and terminal
                    and not truncated
                    and not final_metrics.get("failed", False)
                    and peak_write_count == 0
                )
        summary["elapsed_seconds"] = time.time() - started
    except Exception as exc:
        summary.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - started,
            }
        )
    finally:
        _write_json(summary_path, summary)
        print(json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if summary.get("pass") else 1)
    return 0 if summary.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
