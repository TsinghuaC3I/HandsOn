#!/usr/bin/env python3
"""Summarize action targets and actual PhysX paths in one recorded episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr


SIDES = ("right", "left")


def _replay_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.name == "replay.zarr":
        return path
    return path / "replay.zarr"


def _schema(root: zarr.Group, replay_path: Path) -> dict[str, Any]:
    value = root.attrs.get("recording_schema")
    if isinstance(value, dict):
        return value
    sidecar = replay_path.parent / "recording_schema.json"
    if sidecar.is_file():
        with sidecar.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if isinstance(document, dict):
            return document
    return {}


def _state_path_metrics(
    states: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    positions = np.asarray(states[..., :3], dtype=np.float64)
    linear_velocity = np.asarray(states[..., 7:10], dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    finite = np.all(np.isfinite(positions), axis=-1)
    usable = validity & finite
    adjacent = usable[:-1] & usable[1:]
    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    speeds = np.linalg.norm(linear_velocity, axis=-1)
    finite_speeds = usable & np.isfinite(speeds)
    return {
        "valid_frames": int(np.count_nonzero(usable)),
        "path_length_m": float(np.sum(deltas[adjacent])),
        "rms_linear_speed_m_s": (
            float(np.sqrt(np.mean(np.square(speeds[finite_speeds]))))
            if np.any(finite_speeds)
            else float("nan")
        ),
        "max_linear_speed_m_s": (
            float(np.max(speeds[finite_speeds]))
            if np.any(finite_speeds)
            else float("nan")
        ),
    }


def summarize_physx_episode(
    replay_path: str | Path,
) -> dict[str, Any]:
    replay = _replay_path(replay_path)
    if not replay.is_dir():
        raise FileNotFoundError(f"Replay not found: {replay}")
    root = zarr.open(str(replay), mode="r")
    data = root["data"]
    required = {
        "physx_robot_joint_position",
        "next_physx_robot_joint_position",
        "next_physx_robot_joint_target",
        "next_physx_robot_joint_target_valid",
        "physx_robot_joint_valid",
        "next_physx_robot_joint_valid",
        "physx_tcp_state",
        "next_physx_tcp_state",
        "physx_tcp_valid",
        "next_physx_tcp_valid",
        "physx_sim_time",
        "next_physx_sim_time",
    }
    missing = sorted(name for name in required if name not in data)
    if missing:
        raise ValueError(
            "This episode does not contain a complete PhysX ground-truth trace. "
            f"Missing data fields: {missing}"
        )

    current_time = np.asarray(data["physx_sim_time"][:], dtype=np.float64)[:, 0]
    next_time = np.asarray(data["next_physx_sim_time"][:], dtype=np.float64)[:, 0]
    actual = np.asarray(
        data["next_physx_robot_joint_position"][:],
        dtype=np.float64,
    )
    target = np.asarray(
        data["next_physx_robot_joint_target"][:],
        dtype=np.float64,
    )
    joint_valid = np.asarray(
        data["next_physx_robot_joint_valid"][:],
        dtype=bool,
    )
    target_valid = np.asarray(
        data["next_physx_robot_joint_target_valid"][:],
        dtype=bool,
    )
    if actual.shape != target.shape or actual.shape != joint_valid.shape:
        raise ValueError(
            "Robot joint trace shapes disagree: "
            f"actual={actual.shape}, target={target.shape}, "
            f"valid={joint_valid.shape}."
        )

    result: dict[str, Any] = {
        "replay": str(replay),
        "control_transitions": int(actual.shape[0]),
        "duration_s": (
            float(next_time[-1] - current_time[0])
            if current_time.size and next_time.size
            else 0.0
        ),
        "robot": {},
    }
    schema = _schema(root, replay)
    environment_schema = schema.get("environment_schema", {})
    physx_schema = (
        environment_schema.get("physx", {})
        if isinstance(environment_schema, dict)
        else {}
    )
    joint_names_by_side = (
        physx_schema.get("robot_joint_names", {})
        if isinstance(physx_schema, dict)
        else {}
    )

    current_tcp = np.asarray(data["physx_tcp_state"][:], dtype=np.float64)
    next_tcp = np.asarray(data["next_physx_tcp_state"][:], dtype=np.float64)
    current_tcp_valid = np.asarray(data["physx_tcp_valid"][:], dtype=bool)
    next_tcp_valid = np.asarray(data["next_physx_tcp_valid"][:], dtype=bool)
    tcp_states = np.concatenate([current_tcp[:1], next_tcp], axis=0)
    tcp_valid = np.concatenate([current_tcp_valid[:1], next_tcp_valid], axis=0)

    for side_index, side in enumerate(SIDES):
        mask = (
            joint_valid[:, side_index]
            & target_valid[:, side_index]
            & np.isfinite(actual[:, side_index])
            & np.isfinite(target[:, side_index])
        )
        errors = actual[:, side_index] - target[:, side_index]
        selected = errors[mask]
        names = joint_names_by_side.get(side, [])
        result["robot"][side] = {
            "joint_names": list(names),
            "joint_tracking_samples": int(selected.size),
            "joint_target_actual_rmse": (
                float(np.sqrt(np.mean(np.square(selected))))
                if selected.size
                else float("nan")
            ),
            "joint_target_actual_max_abs": (
                float(np.max(np.abs(selected)))
                if selected.size
                else float("nan")
            ),
            "tcp": _state_path_metrics(
                tcp_states[:, side_index],
                tcp_valid[:, side_index],
            ),
        }

    if (
        "physx_task_rigid_body_state" in data
        and "next_physx_task_rigid_body_state" in data
        and "physx_task_rigid_body_valid" in data
        and "next_physx_task_rigid_body_valid" in data
    ):
        current = np.asarray(data["physx_task_rigid_body_state"][:])
        following = np.asarray(data["next_physx_task_rigid_body_state"][:])
        current_valid = np.asarray(data["physx_task_rigid_body_valid"][:])
        following_valid = np.asarray(data["next_physx_task_rigid_body_valid"][:])
        states = np.concatenate([current[:1], following], axis=0)
        validity = np.concatenate([current_valid[:1], following_valid], axis=0)
        names = physx_schema.get("rigid_body_names", [])
        result["rigid_bodies"] = {
            str(names[index]) if index < len(names) else f"body_{index}":
            _state_path_metrics(states[:, index], validity[:, index])
            for index in range(states.shape[1])
        }

    if "next_physx_deformable_position" in data:
        current_nodes = np.asarray(
            data["physx_deformable_position"][:1],
            dtype=np.float64,
        )
        next_nodes = np.asarray(
            data["next_physx_deformable_position"][:],
            dtype=np.float64,
        )
        nodes = np.concatenate([current_nodes, next_nodes], axis=0)
        if (
            "physx_deformable_valid" in data
            and "next_physx_deformable_valid" in data
        ):
            current_valid = np.asarray(
                data["physx_deformable_valid"][:1],
                dtype=bool,
            )[:, 0]
            next_valid = np.asarray(
                data["next_physx_deformable_valid"][:],
                dtype=bool,
            )[:, 0]
            valid = np.concatenate([current_valid, next_valid])
        else:
            valid = np.ones(nodes.shape[0], dtype=bool)
        valid &= np.all(np.isfinite(nodes), axis=(1, 2))
        adjacent = valid[:-1] & valid[1:]
        displacement = np.linalg.norm(np.diff(nodes, axis=0), axis=-1)
        node_path = np.sum(displacement[adjacent], axis=0)
        result["deformable"] = {
            "node_count": int(nodes.shape[1]),
            "frames": int(nodes.shape[0]),
            "valid_frames": int(np.count_nonzero(valid)),
            "mean_node_path_length_m": (
                float(np.mean(node_path))
                if node_path.size
                else 0.0
            ),
            "max_node_path_length_m": (
                float(np.max(node_path))
                if node_path.size
                else 0.0
            ),
        }
    return result


def _human_report(summary: dict[str, Any]) -> str:
    lines = [
        f"Replay: {summary['replay']}",
        (
            f"Transitions: {summary['control_transitions']}, "
            f"duration: {summary['duration_s']:.6f} s"
        ),
    ]
    for side in SIDES:
        robot = summary["robot"][side]
        tcp = robot["tcp"]
        lines.append(
            f"{side}: joint target/actual RMSE="
            f"{robot['joint_target_actual_rmse']:.6g}, max="
            f"{robot['joint_target_actual_max_abs']:.6g}; "
            f"TCP path={tcp['path_length_m']:.6f} m, "
            f"valid={tcp['valid_frames']}"
        )
    for name, metrics in summary.get("rigid_bodies", {}).items():
        lines.append(
            f"body {name}: path={metrics['path_length_m']:.6f} m, "
            f"valid={metrics['valid_frames']}"
        )
    deformable = summary.get("deformable")
    if deformable:
        lines.append(
            "deformable: "
            f"{deformable['node_count']} nodes, "
            f"mean node path={deformable['mean_node_path_length_m']:.6f} m"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare commanded joint targets with the actual PhysX path."
    )
    parser.add_argument("episode", help="Demo directory or replay.zarr path")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    args = parser.parse_args()
    summary = summarize_physx_episode(args.episode)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    else:
        print(_human_report(summary))


if __name__ == "__main__":
    main()
