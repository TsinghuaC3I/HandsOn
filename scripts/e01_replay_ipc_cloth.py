#!/usr/bin/env python3
"""Replay a successful E01 hand trajectory through high-quality IPC cloth.

The live collector deliberately favors latency.  This pass runs after live
collection, rebuilds the downloaded GarmentLab TCLC_Top578
Cotton_40s_Poplin garment in Genesis, lets it settle, and then replays the
protected palm poses with all 20 measured Wuji joints per hand. Legacy episodes
without the action-aligned readback field explicitly fall back to the stored
commands. Every output is namespaced with ``e01_ipc_replay_`` so the live
observation and action arrays remain unchanged.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import zarr


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPO_ROOT / "handson"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

_PREFIX = "e01_ipc_replay"
_HAND_DIM = 20
_ACTION_DIM = 2 * (7 + _HAND_DIM)
_TABLE_TOP_Z = 0.89


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(
        timespec="seconds"
    )


def _resolve_replay_paths(paths: list[Path]) -> list[Path]:
    replay_paths: list[Path] = []
    for supplied in paths:
        path = supplied.expanduser().resolve()
        if path.name == "replay.zarr" and path.is_dir():
            replay_paths.append(path)
        elif (path / "replay.zarr").is_dir():
            replay_paths.append(path / "replay.zarr")
        elif path.is_dir():
            replay_paths.extend(sorted(path.glob("**/replay.zarr")))
    return sorted(dict.fromkeys(replay_paths))


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _remove_existing_outputs(data: zarr.Group, *, overwrite: bool) -> None:
    names = [name for name in data.keys() if name.startswith(f"{_PREFIX}_")]
    attrs = [name for name in data.attrs.keys() if name.startswith(f"{_PREFIX}_")]
    if (names or attrs) and not overwrite:
        existing = names[0] if names else attrs[0]
        raise ValueError(
            f"{existing!r} already exists; pass --overwrite to replace IPC replay outputs."
        )
    for name in names:
        del data[name]
    for name in attrs:
        del data.attrs[name]


def _load_protected_trajectory(
    data: zarr.Group,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    """Return protected TCPs plus action-aligned complete Wuji joint states.

    ``next_state`` is preferred because it is the protected observation after
    the same transition stored in ``action``.  It is still an environment state
    source, never the unguarded action TCP.  Likewise,
    ``next_e01_hand_qpos[t]`` is the complete actual joint readback produced by
    ``action[t]``.  Older episodes without that field fall back to the complete
    action command and report that limitation explicitly in metadata.
    """
    if "state" not in data or "action" not in data:
        raise ValueError("E01 IPC replay requires data/state and data/action.")

    actions = np.asarray(data["action"][:], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != _ACTION_DIM:
        raise ValueError(
            "E01 action must use the recorder's right27+left27 layout; "
            f"expected (T, {_ACTION_DIM}), got {actions.shape}."
        )

    state_name = "next_state" if "next_state" in data else "state"
    states = np.asarray(data[state_name][:], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] < 14:
        raise ValueError(
            f"data/{state_name} must start with protected right/left TCP poses; "
            f"got {states.shape}."
        )
    if states.shape[0] != actions.shape[0]:
        raise ValueError(
            f"data/{state_name} has {states.shape[0]} frames but action has "
            f"{actions.shape[0]}."
        )

    tcp = np.asarray(states[:, :14], dtype=np.float64)
    if "next_e01_hand_qpos" in data:
        hand_source_name = "next_e01_hand_qpos"
        hand_source_kind = "actual"
        hands = np.asarray(data[hand_source_name][:], dtype=np.float64)
        if hands.shape != (actions.shape[0], 2 * _HAND_DIM):
            raise ValueError(
                f"data/{hand_source_name} must have action-aligned shape "
                f"({actions.shape[0]}, {2 * _HAND_DIM}), got {hands.shape}."
            )
    else:
        hand_source_name = "action"
        hand_source_kind = "command_fallback"
        # Recorder dict flattening is right(TCP7+hand20), left(TCP7+hand20).
        hands = np.concatenate(
            [actions[:, 7:27], actions[:, 34:54]], axis=1
        ).astype(np.float64, copy=False)
    timestamps = (
        np.asarray(data["timestamp"][:], dtype=np.float64).reshape(-1)
        if "timestamp" in data
        else np.arange(actions.shape[0], dtype=np.float64) / 30.0
    )
    if timestamps.shape != (actions.shape[0],):
        raise ValueError(
            f"data/timestamp must have shape ({actions.shape[0]},), got "
            f"{timestamps.shape}."
        )
    if not np.all(np.isfinite(tcp)):
        raise ValueError(f"data/{state_name} protected TCP trajectory is non-finite.")
    if not np.all(np.isfinite(hands)):
        raise ValueError(
            f"data/{hand_source_name} contains a non-finite Wuji hand "
            f"{hand_source_kind}."
        )
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("data/timestamp contains non-finite values.")
    return (
        tcp,
        hands,
        timestamps,
        state_name,
        hand_source_name,
        hand_source_kind,
    )


def _create_array(
    data: zarr.Group,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype | str,
    chunks: tuple[int, ...] | None = None,
) -> Any:
    if chunks is None:
        chunks = (min(1024, max(1, shape[0])), *shape[1:])
    return data.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
    )


def _quat_angle(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a /= max(float(np.linalg.norm(a)), 1e-12)
    b /= max(float(np.linalg.norm(b)), 1e-12)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))


def _contact_link_names(env: Any) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for side in ("right", "left"):
        robot = env._robots[side]
        coupled = set(getattr(robot.material, "coup_links", ()))
        result[side] = sorted(str(name) for name in coupled)
    return result


def _settle_scene(env: Any, settle_steps: int) -> None:
    if settle_steps <= 0:
        return
    for _ in range(settle_steps):
        env._scene.step()
    env._refresh_state(reset_nodal_velocity=True)
    env._base_nodes = env._nodes.copy()
    env._reset_nodes = env._nodes.copy()
    env._configure_tracking(env._nodes)
    env._last_metrics = env._garment_metrics()
    env.env_step = 0
    env._success_counter = 0


def _run_replay(
    replay_path: Path,
    *,
    overwrite: bool,
    settle_steps: int,
    physics_dt: float,
    newton_max_iterations: int,
    store_full_nodes: bool,
    table_penetration_tolerance: float,
    max_node_step: float,
    max_tcp_position_error: float,
    max_tcp_orientation_error: float,
    env_factory: Any | None = None,
) -> dict[str, Any]:
    replay_path = replay_path.expanduser().resolve()
    demo_dir = replay_path.parent
    status_path = demo_dir / f"{_PREFIX}.json"
    root = zarr.open(str(replay_path), mode="a")
    data = root["data"]
    _remove_existing_outputs(data, overwrite=overwrite)
    (
        tcp,
        hands,
        timestamps,
        tcp_state_name,
        hand_source_name,
        hand_source_kind,
    ) = _load_protected_trajectory(data)
    frame_count = int(tcp.shape[0])

    started_at = _utc_now()
    document: dict[str, Any] = {
        "format": "handson.e01_ipc_replay",
        "version": 2,
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "replay_path": str(replay_path),
        "completed_frames": 0,
        "frame_count": frame_count,
        "quality_valid": False,
        "failure_reason": "",
        "trajectory_sources": {
            "tcp": f"data/{tcp_state_name}[:,:14] (protected environment state)",
            "hand": (
                "data/next_e01_hand_qpos (action-aligned actual PhysX joint "
                "readback; right20+left20)"
                if hand_source_kind == "actual"
                else "data/action right[7:27],left[34:54] (legacy fallback "
                "command; actual joint state unavailable)"
            ),
            "hand_kind": hand_source_kind,
            "hand_alignment": (
                "action[t] -> next observation[t]"
                if hand_source_kind == "actual"
                else "data/action[t] command (no joint readback available)"
            ),
            "hand_state_limitation": (
                "The 40D actual joint readback is recording-only; the 102D "
                "policy state remains unchanged."
                if hand_source_kind == "actual"
                else "Legacy episode has no next_e01_hand_qpos; IPC replay is "
                "driven by command targets rather than actual simulated joints."
            ),
        },
        "solver": {
            "backend": "Genesis（旧 PBD 诊断 / IPC 离线）",
            "physics_dt": float(physics_dt),
            "newton_max_iterations": int(newton_max_iterations),
            "settle_steps": int(settle_steps),
            "realtime_sleep": False,
            "rendering": False,
            "full_hand_collision_required": True,
        },
    }
    _atomic_write_json(status_path, document)
    data.attrs[f"{_PREFIX}_status"] = "running"
    data.attrs[f"{_PREFIX}_started_at"] = started_at
    data.attrs[f"{_PREFIX}_hand_source"] = hand_source_name
    data.attrs[f"{_PREFIX}_hand_source_kind"] = hand_source_kind
    data.attrs[f"{_PREFIX}_hand_source_is_actual"] = bool(
        hand_source_kind == "actual"
    )

    # These options are read while the Genesis scene is constructed.  Keep
    # them local to this subprocess so the low-latency live configuration is
    # unaffected.
    os.environ["HANDSON_GENESIS_IPC_NEWTON_MAX_ITERATIONS"] = str(
        int(newton_max_iterations)
    )
    os.environ.setdefault("HANDSON_GENESIS_IPC_NEWTON_TOLERANCE", "0.02")
    os.environ.setdefault("HANDSON_GENESIS_IPC_TRANSLATION_TOLERANCE", "0.05")
    os.environ.setdefault("HANDSON_GENESIS_IPC_D_HAT", "0.001")
    document["solver"].update(
        {
            "newton_tolerance": float(
                os.environ["HANDSON_GENESIS_IPC_NEWTON_TOLERANCE"]
            ),
            "newton_translation_tolerance": float(
                os.environ["HANDSON_GENESIS_IPC_TRANSLATION_TOLERANCE"]
            ),
            "contact_d_hat": float(os.environ["HANDSON_GENESIS_IPC_D_HAT"]),
            "contact_resistance": float(
                os.environ.get("HANDSON_GENESIS_IPC_CONTACT_RESISTANCE", "1e8")
            ),
            "two_way_coupling": os.environ.get(
                "HANDSON_GENESIS_IPC_TWO_WAY_COUPLING", "0"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            "cloth": {
                "youngs_modulus": float(
                    os.environ.get("HANDSON_GENESIS_E01_YOUNGS_MODULUS", "1e4")
                ),
                "poisson_ratio": float(
                    os.environ.get("HANDSON_GENESIS_E01_POISSON_RATIO", "0.30")
                ),
                "density": float(
                    os.environ.get("HANDSON_GENESIS_E01_DENSITY", "300")
                ),
                "thickness": float(
                    os.environ.get("HANDSON_GENESIS_E01_THICKNESS", "0.0002")
                ),
                "bending_stiffness": float(
                    os.environ.get(
                        "HANDSON_GENESIS_E01_BENDING_STIFFNESS", "10"
                    )
                ),
                "friction": float(
                    os.environ.get("HANDSON_GENESIS_E01_CLOTH_FRICTION", "0.7")
                ),
            },
            "quality_thresholds": {
                "table_penetration_tolerance": float(
                    table_penetration_tolerance
                ),
                "max_node_step": float(max_node_step),
                "max_tcp_position_error": float(max_tcp_position_error),
                "max_tcp_orientation_error": float(max_tcp_orientation_error),
            },
        }
    )

    env: Any | None = None
    arrays: dict[str, Any] = {}
    completed_frames = 0
    try:
        if env_factory is None:
            from handson.sim.genesis.e01_fold_garment_env import (
                HandsOnE01GenesisEnv,
            )

            env_factory = HandsOnE01GenesisEnv
        env = env_factory(
            render_mode="rgb_array",
            randomize=False,
            image_obs=False,
            show_native_viewer=False,
            hz=30,
            physics_dt=physics_dt,
        )
        env.reset()
        # ``control_dt`` and substep count were fixed at construction.  This
        # only bypasses the wall-clock sleep at the end of Env.step().
        env.hz = 0

        contact_links = _contact_link_names(env)
        for side, names in contact_links.items():
            expected = {f"{side}_palm_link"} | {
                f"{side}_finger{finger}_link{link}"
                for finger in range(1, 6)
                for link in range(2, 5)
            }
            missing = sorted(expected.difference(names))
            if missing:
                raise RuntimeError(
                    f"{side} IPC hand is missing full palm/finger collision "
                    f"links: {missing}."
                )
        document["solver"]["contact_links"] = contact_links
        document["solver"]["garment_asset"] = str(env._asset_path)

        _settle_scene(env, settle_steps)
        node_count = int(env._nodes.shape[0])
        keypoint_indices = np.asarray(env._keypoint_indices, dtype=np.int64)
        keypoint_count = int(keypoint_indices.size)
        document["solver"]["garment_node_count"] = node_count
        document["solver"]["keypoint_indices"] = keypoint_indices.tolist()

        arrays["tcp_target"] = _create_array(
            data,
            f"{_PREFIX}_tcp_target",
            shape=(frame_count, 14),
            dtype="f8",
        )
        arrays["hand_command"] = _create_array(
            data,
            f"{_PREFIX}_hand_command",
            shape=(frame_count, 40),
            dtype="f8",
        )
        arrays["hand_qpos"] = _create_array(
            data,
            f"{_PREFIX}_hand_qpos",
            shape=(frame_count, 40),
            dtype="f4",
        )
        arrays["keypoints"] = _create_array(
            data,
            f"{_PREFIX}_cloth_keypoints",
            shape=(frame_count, keypoint_count, 3),
            dtype="f4",
            chunks=(min(64, max(1, frame_count)), keypoint_count, 3),
        )
        if store_full_nodes:
            arrays["nodes"] = _create_array(
                data,
                f"{_PREFIX}_cloth_nodes",
                shape=(frame_count, node_count, 3),
                dtype="f4",
                chunks=(1, node_count, 3),
            )
        for key in (
            "finite_frame",
            "table_valid_frame",
            "motion_valid_frame",
            "tcp_valid_frame",
            "valid_frame",
        ):
            arrays[key] = _create_array(
                data,
                f"{_PREFIX}_{key}",
                shape=(frame_count, 1),
                dtype="u1",
            )
        for key in (
            "table_penetration",
            "max_node_step",
            "tcp_position_error",
            "tcp_orientation_error",
            "step_seconds",
        ):
            width = 2 if key.startswith("tcp_") else 1
            arrays[key] = _create_array(
                data,
                f"{_PREFIX}_{key}",
                shape=(frame_count, width),
                dtype="f4",
            )
        arrays["timestamp"] = _create_array(
            data,
            f"{_PREFIX}_timestamp",
            shape=(frame_count,),
            dtype="f8",
        )
        arrays["tcp_target"][:] = tcp
        arrays["hand_command"][:] = hands
        arrays["timestamp"][:] = timestamps

        previous_nodes = np.asarray(env._nodes, dtype=np.float64).copy()
        for frame_index in range(frame_count):
            right_action = np.concatenate(
                [tcp[frame_index, :7], hands[frame_index, :20]]
            )
            left_action = np.concatenate(
                [tcp[frame_index, 7:14], hands[frame_index, 20:40]]
            )
            start = time.perf_counter()
            env.step({"right": right_action, "left": left_action})
            elapsed = time.perf_counter() - start

            nodes = np.asarray(env._nodes, dtype=np.float64)
            if nodes.shape != (node_count, 3):
                raise RuntimeError(
                    f"Cloth topology changed at frame {frame_index}: "
                    f"expected {(node_count, 3)}, got {nodes.shape}."
                )
            finite = bool(np.all(np.isfinite(nodes)))
            penetration = (
                max(0.0, _TABLE_TOP_Z - float(np.min(nodes[:, 2])))
                if finite
                else float("inf")
            )
            node_step = (
                float(np.max(np.linalg.norm(nodes - previous_nodes, axis=1)))
                if finite
                else float("inf")
            )
            actual_tcp = np.concatenate(
                [env._right_tcp_pose, env._left_tcp_pose]
            ).astype(np.float64)
            tcp_position_error = np.asarray(
                [
                    np.linalg.norm(actual_tcp[:3] - tcp[frame_index, :3]),
                    np.linalg.norm(actual_tcp[7:10] - tcp[frame_index, 7:10]),
                ],
                dtype=np.float64,
            )
            tcp_orientation_error = np.asarray(
                [
                    _quat_angle(actual_tcp[3:7], tcp[frame_index, 3:7]),
                    _quat_angle(actual_tcp[10:14], tcp[frame_index, 10:14]),
                ],
                dtype=np.float64,
            )
            table_valid = bool(penetration <= table_penetration_tolerance)
            motion_valid = bool(node_step <= max_node_step)
            tcp_valid = bool(
                np.all(tcp_position_error <= max_tcp_position_error)
                and np.all(tcp_orientation_error <= max_tcp_orientation_error)
            )
            valid = bool(finite and table_valid and motion_valid and tcp_valid)

            arrays["keypoints"][frame_index] = nodes[keypoint_indices].astype(
                np.float32
            )
            if "nodes" in arrays:
                arrays["nodes"][frame_index] = nodes.astype(np.float32)
            arrays["hand_qpos"][frame_index] = np.concatenate(
                [env._right_hand, env._left_hand]
            ).astype(np.float32)
            arrays["finite_frame"][frame_index, 0] = finite
            arrays["table_valid_frame"][frame_index, 0] = table_valid
            arrays["motion_valid_frame"][frame_index, 0] = motion_valid
            arrays["tcp_valid_frame"][frame_index, 0] = tcp_valid
            arrays["valid_frame"][frame_index, 0] = valid
            arrays["table_penetration"][frame_index, 0] = penetration
            arrays["max_node_step"][frame_index, 0] = node_step
            arrays["tcp_position_error"][frame_index] = tcp_position_error
            arrays["tcp_orientation_error"][frame_index] = tcp_orientation_error
            arrays["step_seconds"][frame_index, 0] = elapsed
            previous_nodes = nodes.copy()
            completed_frames = frame_index + 1

        valid_frames = np.asarray(arrays["valid_frame"][:], dtype=np.uint8)[:, 0]
        quality_valid = bool(frame_count > 0 and np.all(valid_frames != 0))
        if frame_count == 0:
            raise RuntimeError("E01 IPC replay cannot process an empty episode.")
        if not quality_valid:
            bad = np.flatnonzero(valid_frames == 0)
            preview = ", ".join(str(int(index)) for index in bad[:12])
            suffix = "..." if bad.size > 12 else ""
            raise RuntimeError(
                f"IPC replay quality checks failed on {bad.size}/{frame_count} "
                f"frames ({preview}{suffix})."
            )

        document["status"] = "complete"
        document["quality_valid"] = True
        document["completed_frames"] = completed_frames
        document["completed_at"] = _utc_now()
        data.attrs[f"{_PREFIX}_status"] = "complete"
        data.attrs[f"{_PREFIX}_quality_valid"] = True
        data.attrs[f"{_PREFIX}_completed_frames"] = completed_frames
        data.attrs[f"{_PREFIX}_completed_at"] = document["completed_at"]
        data.attrs[f"{_PREFIX}_tcp_source"] = tcp_state_name
        data.attrs[f"{_PREFIX}_hand_source"] = hand_source_name
        data.attrs[f"{_PREFIX}_hand_source_kind"] = hand_source_kind
        data.attrs[f"{_PREFIX}_hand_source_is_actual"] = bool(
            hand_source_kind == "actual"
        )
        data.attrs[f"{_PREFIX}_physics_dt"] = float(physics_dt)
        data.attrs[f"{_PREFIX}_newton_max_iterations"] = int(
            newton_max_iterations
        )
        data.attrs[f"{_PREFIX}_settle_steps"] = int(settle_steps)
        data.attrs[f"{_PREFIX}_store_full_nodes"] = bool(store_full_nodes)
        data.attrs[f"{_PREFIX}_garment_asset"] = str(env._asset_path)
        data.attrs[f"{_PREFIX}_contact_links"] = contact_links
        _atomic_write_json(status_path, document)
        return document
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        document["status"] = "failed"
        document["quality_valid"] = False
        document["completed_frames"] = int(completed_frames)
        document["completed_at"] = _utc_now()
        document["failure_reason"] = reason
        data.attrs[f"{_PREFIX}_status"] = "failed"
        data.attrs[f"{_PREFIX}_quality_valid"] = False
        data.attrs[f"{_PREFIX}_completed_frames"] = int(completed_frames)
        data.attrs[f"{_PREFIX}_failure_reason"] = reason
        data.attrs[f"{_PREFIX}_completed_at"] = document["completed_at"]
        _atomic_write_json(status_path, document)
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay E01 protected hand trajectories through an offline "
            "high-quality Genesis/libuipc cloth scene."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Demo dir, replay.zarr, or a directory containing E01 demos.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--newton-max-iterations", type=int, default=8)
    parser.add_argument(
        "--keypoints-only",
        action="store_true",
        help="Do not store the full (T,N,3) cloth node trajectory.",
    )
    parser.add_argument(
        "--table-penetration-tolerance", type=float, default=0.004
    )
    parser.add_argument("--max-node-step", type=float, default=0.05)
    parser.add_argument("--max-tcp-position-error", type=float, default=0.003)
    parser.add_argument("--max-tcp-orientation-error", type=float, default=0.03)
    args = parser.parse_args()

    replay_paths = _resolve_replay_paths(args.paths)
    if not replay_paths:
        raise SystemExit("No replay.zarr paths found.")
    if args.settle_steps < 0:
        raise SystemExit("--settle-steps must be non-negative.")
    if args.physics_dt <= 0.0:
        raise SystemExit("--physics-dt must be positive.")
    if args.newton_max_iterations < 1:
        raise SystemExit("--newton-max-iterations must be positive.")
    if args.max_node_step <= 0.0:
        raise SystemExit("--max-node-step must be positive.")

    for replay_path in replay_paths:
        result = _run_replay(
            replay_path,
            overwrite=args.overwrite,
            settle_steps=args.settle_steps,
            physics_dt=args.physics_dt,
            newton_max_iterations=args.newton_max_iterations,
            store_full_nodes=not args.keypoints_only,
            table_penetration_tolerance=args.table_penetration_tolerance,
            max_node_step=args.max_node_step,
            max_tcp_position_error=args.max_tcp_position_error,
            max_tcp_orientation_error=args.max_tcp_orientation_error,
        )
        print(
            f"[E01 IPC replay] {replay_path}: {result['status']} "
            f"({result['completed_frames']}/{result['frame_count']} frames)",
            flush=True,
        )


if __name__ == "__main__":
    main()
