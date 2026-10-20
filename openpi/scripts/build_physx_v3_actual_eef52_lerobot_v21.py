#!/usr/bin/env python3
"""Convert certified PhysX-v3 episodes to an actual-state EEF52 dataset.

Each row is action-aligned as follows:

* observation.state = actual state before the recorded control action
* action = actual state after the recorded control action

Both vectors use the same right-then-left 52-D layout.  Each arm contributes
``xyz(3) + rotation_vector(3) + actual_hand_joint_position(20)``.  The source
is the PhysX readback, never the legacy command-valued ``data/state`` field.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HANDSON_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "handson"
if str(HANDSON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(HANDSON_PACKAGE_ROOT))

from handson.policy_joint_order import (  # noqa: E402
    ACTUAL_EEF52_POLICY_SCHEMA,
    PHYSX_ARTICULATION_JOINT_ORDER_SCHEMA,
    ROBOT_SIDE_ORDER,
    WUJI_HAND_CONTROL_ORDER_SCHEMA,
    reorder_articulation_hands_to_wuji_control,
    wuji_hand_order_contract,
)


DEFAULT_RAW_ROOT = Path(
    "/mnt/pfs/luotianwei/handson/datasets/new_cloud_physx_v3_raw_20260805"
)
DEFAULT_MANIFEST = DEFAULT_RAW_ROOT / "_manifests/new_cloud_physx_v3_sync.json"
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/pfs/luotianwei/handson/datasets/lerobot_v21/"
    "physx_v3_actual_eef52"
)
DEFAULT_ASSETS_DIR = Path(
    "/mnt/pfs/luotianwei/handson-h200-bench-repo/openpi/assets/"
    "physx_v3_actual_eef52_wuji1/local_repo"
)
ISAAC_SITE_PACKAGES = Path(
    "/mnt/pfs/luotianwei/envs/isaacsim-5.1-handson/lib/python3.11/site-packages"
)

SIDE_COUNT = 2
ARM_DOF = 7
HAND_DOF = 20
SIDE_DIM = 26
STATE_DIM = 52
ACTION_DIM = 52
FPS = 30

TASK_PROMPTS = {
    "t04_cover_pitcher": "Place the lid on the pitcher.",
    "t09_insert_peg": "Insert the peg into the socket.",
    "t44_stack_blocks": "Stack the blocks.",
    "t03_cover_box": "Place the lid on the clear box.",
    "t36_scoop_marbles": "Scoop the marbles with the spoon.",
    "t50_water_plant": "Water the plant.",
    "t23_photograph_logo": (
        "Pick up the camera with both hands and take a photograph."
    ),
    "t14_move_and_click_mouse": "Move the mouse and click it.",
    "t43_squeeze_scissors": (
        "Lift the scissors clear of the table and make three complete cuts in the air. "
        "Try one hand first; the other hand may reposition or assist."
    ),
    "t21_pass_block": (
        "Hand off the object between the hands and place it into the box."
    ),
}

TASK_TARGET_LABELS = {
    "t26_place_fruit_in_bucket": {"apple", "011_banana", "orange", "pear"},
    "t25_place_cup_on_plate": {"025_mug"},
    "t24_place_ball_in_bowl": {"tennis_ball", "golf_ball", "mini_soccer_ball"},
    "t19_orient_cup_upright": {"a_cup"},
}


def task_prompt(task: str, target_label: Any) -> str:
    """Return the episode-specific policy instruction.

    The legacy task ids are not a sufficient semantic label: A01 and A03
    contain multiple targets, while A06 is still named ``orient_rubiks_cube``
    even though the PhysX-v3 scene and success predicate operate on a cup.
    Never silently fall back to a prompt derived from the task id.
    """

    target = "" if target_label is None else str(target_label).strip()
    expected = TASK_TARGET_LABELS.get(task)
    if expected is not None and target not in expected:
        raise ValueError(
            f"Unsupported target_label={target!r} for task={task!r}; "
            f"expected one of {sorted(expected)}"
        )
    if task == "t26_place_fruit_in_bucket":
        display = {"011_banana": "banana"}.get(target, target)
        return f"Pick the {display} and place it into the bucket."
    if task == "t25_place_cup_on_plate":
        return "Pick the mug and place it into the tray."
    if task == "t24_place_ball_in_bowl":
        display = {
            "mini_soccer_ball": "mini soccer ball",
        }.get(target, target.replace("_", " "))
        return f"Pick the {display} and place it into the bowl."
    if task == "t19_orient_cup_upright":
        return "Pick up the cup and place it upright on the table."
    try:
        return TASK_PROMPTS[task]
    except KeyError as exc:
        raise ValueError(f"No policy prompt contract for task={task!r}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--repo-id", default="handson/physx_v3_actual_eef52")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-source-success-without-certification",
        action="store_true",
        help=(
            "Include PhysX-v3 episodes whose collection metadata marks both "
            "source_outcome and task_outcome as success even when physical "
            "replay certification is missing or failed."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        )


def quat_wxyz_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    """Convert arbitrary-shaped normalized quaternions to principal rotvecs."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected (..., 4) quaternion, got {quaternion.shape}")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 1e-12):
        raise ValueError("Quaternion contains non-finite or zero-norm values")
    quaternion = quaternion / norm
    # q and -q encode the same rotation.  Choosing w >= 0 gives a principal
    # rotation angle in [0, pi], matching scipy Rotation.as_rotvec semantics.
    quaternion = np.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    w = np.clip(quaternion[..., 0], -1.0, 1.0)
    vector = quaternion[..., 1:4]
    vector_norm = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, w)
    scale = np.zeros_like(angle)
    nonzero = vector_norm > 1e-12
    scale[nonzero] = angle[nonzero] / vector_norm[nonzero]
    return vector * scale[..., None]


def load_robot_joint_names(recording_schema_path: Path) -> dict[str, list[str]]:
    """Read the authoritative articulation axis from an episode schema."""

    recording_schema = load_json(recording_schema_path)
    environment_schema = recording_schema.get("environment_schema")
    if not isinstance(environment_schema, dict):
        raise ValueError("recording_schema.json has no environment_schema")
    physx = environment_schema.get("physx")
    if not isinstance(physx, dict):
        raise ValueError("recording schema has no environment_schema.physx")
    names = physx.get("robot_joint_names")
    if not isinstance(names, dict):
        raise ValueError("recording schema has no PhysX robot_joint_names")
    result = {}
    for side in ROBOT_SIDE_ORDER:
        side_names = names.get(side)
        if not isinstance(side_names, list):
            raise ValueError(f"recording schema has no {side} robot joint names")
        result[side] = [str(name) for name in side_names]
    declared_order = physx.get("robot_joint_order_schema")
    if declared_order not in (None, PHYSX_ARTICULATION_JOINT_ORDER_SCHEMA):
        raise ValueError(
            f"unsupported PhysX robot joint order: {declared_order!r}"
        )
    return result


def policy_schema(order_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "handson.policy_schema",
        "version": 1,
        "schema_id": ACTUAL_EEF52_POLICY_SCHEMA,
        "state_schema": "actual_before_action_right26_left26",
        "action_schema": "actual_after_action_right26_left26",
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "side_order": list(ROBOT_SIDE_ORDER),
        "per_side_layout": ["tcp_xyz3", "tcp_rotvec3", "hand_joint20"],
        "hand_joint_order": order_contract,
    }


def actual_eef52(
    tcp_state: np.ndarray,
    joint_position: np.ndarray,
    robot_joint_names: dict[str, list[str]],
) -> np.ndarray:
    """Build [right26, left26] from live simulator readback arrays."""
    tcp_state = np.asarray(tcp_state, dtype=np.float64)
    joint_position = np.asarray(joint_position, dtype=np.float64)
    if tcp_state.ndim != 3 or tcp_state.shape[1:] != (SIDE_COUNT, 13):
        raise ValueError(f"Unexpected TCP state shape: {tcp_state.shape}")
    if joint_position.ndim != 3 or joint_position.shape[1:] != (
        SIDE_COUNT,
        ARM_DOF + HAND_DOF,
    ):
        raise ValueError(f"Unexpected joint state shape: {joint_position.shape}")
    pose = np.concatenate(
        [tcp_state[..., :3], quat_wxyz_to_rotvec(tcp_state[..., 3:7])],
        axis=-1,
    )
    hand = reorder_articulation_hands_to_wuji_control(
        joint_position,
        robot_joint_names,
    )
    per_side = np.concatenate([pose, hand], axis=-1)
    result = per_side.reshape(per_side.shape[0], STATE_DIM)
    if not np.all(np.isfinite(result)):
        raise ValueError("Actual EEF52 contains non-finite values")
    return result.astype(np.float32)


def feature_stats(array: np.ndarray) -> dict[str, Any]:
    return {
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def video_feature(height: int, width: int) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def dataset_info(
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    schema: dict[str, Any],
) -> dict[str, Any]:
    return {
        "codebase_version": "v2.1",
        "robot_type": "rokae_wuji1_physx_v3_actual_eef52",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * 3,
        "total_chunks": math.ceil(total_episodes / 1000),
        "chunks_size": 1000,
        "fps": FPS,
        "handson_policy_schema": schema,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.front": video_feature(640, 640),
            "observation.images.wrist_left": video_feature(480, 848),
            "observation.images.wrist_right": video_feature(480, 848),
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": ["action"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def make_table(
    state: np.ndarray,
    action: np.ndarray,
    episode_index: int,
    global_start: int,
    task_index: int,
) -> pa.Table:
    length = state.shape[0]
    schema = pa.schema(
        [
            pa.field("observation.state", pa.list_(pa.float32(), STATE_DIM)),
            pa.field("action", pa.list_(pa.float32(), ACTION_DIM)),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
    )
    arrays = [
        pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1)), STATE_DIM),
        pa.FixedSizeListArray.from_arrays(pa.array(action.reshape(-1)), ACTION_DIM),
        pa.array(np.arange(length, dtype=np.float32) / np.float32(FPS)),
        pa.array(np.arange(length, dtype=np.int64)),
        pa.array(np.full(length, episode_index, dtype=np.int64)),
        pa.array(np.arange(global_start, global_start + length, dtype=np.int64)),
        pa.array(np.full(length, task_index, dtype=np.int64)),
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def require_all_valid(group: Any, keys: tuple[str, ...], length: int) -> None:
    for key in keys:
        values = np.asarray(group[f"data/{key}"][:length])
        if values.size == 0 or not np.all(values == 1):
            raise ValueError(f"{key} contains invalid readback entries")


def preflight_task_prompts(
    raw_root: Path,
    rows: list[dict[str, Any]],
) -> Counter[str]:
    """Validate every source episode's semantic label before writing output."""

    prompt_counts: Counter[str] = Counter()
    errors: list[str] = []
    for row in rows:
        episode_dir = raw_root.joinpath(*row["prefix"].split("/"))
        meta_path = episode_dir / "collection_meta.json"
        try:
            meta = load_json(meta_path)
            prompt = task_prompt(str(row["task"]), meta.get("target_label"))
            prompt_counts[prompt] += 1
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{episode_dir}: {exc}")
    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(
            f"Prompt preflight failed for {len(errors)} episode(s):\n{preview}"
        )
    return prompt_counts


def main() -> None:
    args = parse_args()
    if ISAAC_SITE_PACKAGES.exists():
        sys.path.append(str(ISAAC_SITE_PACKAGES))
    import zarr  # noqa: PLC0415

    from openpi.shared import normalize  # noqa: PLC0415

    sync_manifest = load_json(args.manifest)
    rows = sorted(sync_manifest["episodes"], key=lambda row: row["prefix"])
    if args.limit > 0:
        rows = rows[: args.limit]
    prompt_counts = preflight_task_prompts(args.raw_root, rows)

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.output_root}")
        shutil.rmtree(args.output_root)
    if args.assets_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Assets already exist: {args.assets_dir}")
        shutil.rmtree(args.assets_dir)
    args.output_root.mkdir(parents=True)

    task_to_index: dict[str, int] = {}
    task_counts: Counter[str] = Counter()
    converted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    state_running = normalize.RunningStats()
    action_running = normalize.RunningStats()
    global_index = 0
    canonical_order_contract: dict[str, Any] | None = None

    for row in rows:
        episode_dir = args.raw_root.joinpath(*row["prefix"].split("/"))
        task = str(row["task"])
        cert_path = episode_dir / "physical_replay_certification.json"
        marker_path = episode_dir / "DATASET_PHYSX_V3.json"
        collection_meta_path = episode_dir / "collection_meta.json"
        recording_schema_path = episode_dir / "recording_schema.json"
        zarr_path = episode_dir / "replay.zarr"
        video_paths = {
            "observation.images.front": episode_dir / "videos/main.mp4",
            "observation.images.wrist_left": episode_dir / "videos/wrist_left.mp4",
            "observation.images.wrist_right": episode_dir / "videos/wrist_right.mp4",
        }
        required = [
            marker_path,
            collection_meta_path,
            recording_schema_path,
            zarr_path,
            *video_paths.values(),
        ]
        if not args.include_source_success_without_certification:
            required.insert(0, cert_path)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            skipped.append(
                {
                    "episode": str(episode_dir),
                    "reason": "missing_files",
                    "files": missing,
                }
            )
            continue
        marker = load_json(marker_path)
        collection_meta = load_json(collection_meta_path)
        prompt = task_prompt(task, collection_meta.get("target_label"))
        cert = load_json(cert_path) if cert_path.exists() else {}
        certified = (
            cert.get("status") == "passed" and cert.get("final_succeed") is True
        )
        source_success = (
            collection_meta.get("source_outcome") == "success"
            and collection_meta.get("task_outcome") == "success"
        )
        eligible = certified or (
            args.include_source_success_without_certification and source_success
        )
        if marker.get("is_physx_v3_data") is not True or not eligible:
            skipped.append(
                {
                    "episode": str(episode_dir),
                    "reason": "not_eligible_successful_physx_v3",
                    "cert_status": cert.get("status"),
                    "final_succeed": cert.get("final_succeed"),
                    "source_outcome": collection_meta.get("source_outcome"),
                    "task_outcome": collection_meta.get("task_outcome"),
                }
            )
            continue

        group = zarr.open_group(str(zarr_path), mode="r")
        keys = (
            "physx_tcp_state",
            "physx_robot_joint_position",
            "next_physx_tcp_state",
            "next_physx_robot_joint_position",
        )
        lengths = [int(group[f"data/{key}"].shape[0]) for key in keys]
        length = min(lengths)
        if length < 2 or len(set(lengths)) != 1:
            skipped.append(
                {
                    "episode": str(episode_dir),
                    "reason": "invalid_or_misaligned_length",
                    "lengths": dict(zip(keys, lengths, strict=True)),
                }
            )
            continue
        try:
            robot_joint_names = load_robot_joint_names(recording_schema_path)
            episode_order_contract = wuji_hand_order_contract(robot_joint_names)
            if canonical_order_contract is None:
                canonical_order_contract = episode_order_contract
            elif episode_order_contract != canonical_order_contract:
                raise ValueError(
                    "episode robot joint ordering differs from earlier episodes"
                )
            require_all_valid(
                group,
                (
                    "physx_tcp_valid",
                    "physx_robot_joint_valid",
                    "next_physx_tcp_valid",
                    "next_physx_robot_joint_valid",
                ),
                length,
            )
            state = actual_eef52(
                group["data/physx_tcp_state"][:length],
                group["data/physx_robot_joint_position"][:length],
                robot_joint_names,
            )
            action = actual_eef52(
                group["data/next_physx_tcp_state"][:length],
                group["data/next_physx_robot_joint_position"][:length],
                robot_joint_names,
            )
        except (KeyError, ValueError) as exc:
            skipped.append(
                {
                    "episode": str(episode_dir),
                    "reason": "invalid_actual_state",
                    "detail": str(exc),
                }
            )
            continue

        if prompt not in task_to_index:
            task_to_index[prompt] = len(task_to_index)
        task_index = task_to_index[prompt]
        episode_index = len(converted)
        chunk = episode_index // 1000
        parquet_path = (
            args.output_root
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            make_table(state, action, episode_index, global_index, task_index),
            parquet_path,
            compression="snappy",
        )

        for video_key, source in video_paths.items():
            destination = (
                args.output_root
                / "videos"
                / f"chunk-{chunk:03d}"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(source.resolve(), destination)

        append_jsonl(
            args.output_root / "meta/episodes.jsonl",
            {"episode_index": episode_index, "tasks": [prompt], "length": length},
        )
        append_jsonl(
            args.output_root / "meta/episodes_stats.jsonl",
            {
                "episode_index": episode_index,
                "stats": {
                    "observation.state": feature_stats(state),
                    "action": feature_stats(action),
                },
            },
        )
        state_running.update(state)
        action_running.update(action)
        task_counts[task] += 1
        converted.append(
            {
                "episode_index": episode_index,
                "source": str(episode_dir),
                "task": task,
                "target_label": collection_meta.get("target_label"),
                "prompt": prompt,
                "frames": length,
                "physical_replay_certification_status": cert.get("status"),
                "physical_replay_final_succeed": cert.get("final_succeed"),
                "source_outcome": collection_meta.get("source_outcome"),
                "task_outcome": collection_meta.get("task_outcome"),
            }
        )
        global_index += length
        if len(converted) % 25 == 0:
            print(
                f"converted={len(converted)}/{len(rows)} frames={global_index}",
                flush=True,
            )

    for prompt, task_index in sorted(task_to_index.items(), key=lambda item: item[1]):
        append_jsonl(
            args.output_root / "meta/tasks.jsonl",
            {"task_index": task_index, "task": prompt},
        )

    if canonical_order_contract is None:
        raise RuntimeError("No episode supplied a valid joint-order contract")
    canonical_policy_schema = policy_schema(canonical_order_contract)
    write_json(
        args.output_root / "meta/info.json",
        dataset_info(
            len(converted),
            global_index,
            len(task_to_index),
            canonical_policy_schema,
        ),
    )
    write_json(
        args.output_root / "meta/handson_policy_schema.json",
        canonical_policy_schema,
    )
    write_json(
        args.output_root / "conversion_manifest.json",
        {
            "format": "handson.policy_dataset_conversion",
            "version": 1,
            "raw_root": str(args.raw_root),
            "source_manifest": str(args.manifest),
            "dataset_root": str(args.output_root),
            "repo_id": args.repo_id,
            "source_contract": (
                "PhysX-v3 episodes with collection source_outcome=success and "
                "task_outcome=success; physical replay certification is recorded "
                "but not required"
                if args.include_source_success_without_certification
                else "certified PhysX-v3 episodes with status=passed and "
                "final_succeed=true"
            ),
            "include_source_success_without_certification": (
                args.include_source_success_without_certification
            ),
            "state_schema": (
                "actual_before_action: [right_xyz3,right_rotvec3,right_hand_joint20,"
                "left_xyz3,left_rotvec3,left_hand_joint20]"
            ),
            "action_schema": (
                "actual_after_action: [right_xyz3,right_rotvec3,right_hand_joint20,"
                "left_xyz3,left_rotvec3,left_hand_joint20]"
            ),
            "tcp_source": "physx_tcp_state/next_physx_tcp_state palm_link body_state_w",
            "hand_source": (
                "physx_robot_joint_position/next_physx_robot_joint_position "
                "reordered from articulation order into Wuji control hand20 order"
            ),
            "policy_schema": canonical_policy_schema,
            "joint_order_conversion": canonical_order_contract,
            "quaternion_source_order": "wxyz",
            "rotation_output": "principal rotation vector in radians",
            "side_order": "right_then_left",
            "legacy_data_state_used": False,
            "legacy_command_fields_used": False,
            "candidate_count": len(rows),
            "converted_count": len(converted),
            "skipped_count": len(skipped),
            "total_frames": global_index,
            "task_counts": dict(sorted(task_counts.items())),
            "prompt_counts": dict(sorted(prompt_counts.items())),
            "skipped": skipped,
            "episodes": converted,
        },
    )
    normalize.save(
        args.assets_dir,
        {
            "state": state_running.get_statistics(),
            "actions": action_running.get_statistics(),
        },
    )
    write_json(
        args.assets_dir / "handson_policy_schema.json",
        canonical_policy_schema,
    )
    summary = {
        "candidate_count": len(rows),
        "converted_count": len(converted),
        "skipped_count": len(skipped),
        "total_frames": global_index,
        "task_counts": dict(sorted(task_counts.items())),
        "prompt_counts": dict(sorted(prompt_counts.items())),
        "dataset_root": str(args.output_root),
        "assets_dir": str(args.assets_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if skipped:
        raise SystemExit(
            f"Conversion skipped {len(skipped)} episodes; inspect manifest"
        )


if __name__ == "__main__":
    main()
