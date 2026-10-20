#!/usr/bin/env python3
"""Convert certified PhysX-v3 episodes to real-policy-aligned actual Joint54.

For every row, observation.state is the measured joint state before action[t]
and action is the measured state after action[t].  Arms cross the same
ROKAE-radian to Tianji-degree boundary as the real robot.  Hands are gathered
by recorded joint name into Wuji controller order.
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
    ACTUAL_JOINT54_POLICY_SCHEMA,
    ROBOT_SIDE_ORDER,
    reorder_articulation_hands_to_wuji_control,
    wuji_hand_order_contract,
)

import build_physx_v3_actual_eef52_lerobot_v21 as common  # noqa: E402


DEFAULT_RAW_ROOT = common.DEFAULT_RAW_ROOT
DEFAULT_MANIFEST = common.DEFAULT_MANIFEST
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/pfs/luotianwei/handson/datasets/lerobot_v21/"
    "physx_v3_actual_joint54"
)
DEFAULT_ASSETS_DIR = Path(
    "/mnt/pfs/luotianwei/handson-h200-bench-repo/openpi/assets/"
    "physx_v3_actual_joint54_wuji1/local_repo"
)
ISAAC_SITE_PACKAGES = common.ISAAC_SITE_PACKAGES
JOINT54_DIM = 54
FPS = 30
ROKAE_TIANJI_SIGN = np.asarray(
    [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--repo-id", default="handson/physx_v3_actual_joint54")
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


def validate_arm_joint_names(names: dict[str, list[str]]) -> None:
    for side in ROBOT_SIDE_ORDER:
        expected = [f"joint{index}_{side}" for index in range(1, 8)]
        if names[side][:7] != expected:
            raise ValueError(
                f"{side} arm articulation order differs: "
                f"got {names[side][:7]}, expected {expected}"
            )


def actual_joint54(
    joint_position: np.ndarray,
    names: dict[str, list[str]],
) -> np.ndarray:
    values = np.asarray(joint_position, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (2, 27):
        raise ValueError(f"Unexpected joint state shape: {values.shape}")
    validate_arm_joint_names(names)
    hands = reorder_articulation_hands_to_wuji_control(values, names)
    arms_deg = np.rad2deg(values[..., :7]) * ROKAE_TIANJI_SIGN
    pieces = []
    for side in ("left", "right"):
        side_index = ROBOT_SIDE_ORDER.index(side)
        pieces.extend((arms_deg[:, side_index], hands[:, side_index]))
    result = np.concatenate(pieces, axis=-1)
    if result.shape != (values.shape[0], JOINT54_DIM):
        raise ValueError(f"Unexpected Joint54 result shape: {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("Actual Joint54 contains non-finite values")
    return result.astype(np.float32)


def policy_schema(order_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "handson.policy_schema",
        "version": 1,
        "schema_id": ACTUAL_JOINT54_POLICY_SCHEMA,
        "state_schema": "actual_before_action_left27_right27",
        "action_schema": "actual_after_action_left27_right27",
        "state_dim": JOINT54_DIM,
        "action_dim": JOINT54_DIM,
        "side_order": ["left", "right"],
        "per_side_layout": ["arm_joint7_tianji_deg", "hand_joint20_rad"],
        "arm_source": "PhysX ROKAE radians",
        "arm_policy_transform": {
            "units": "degrees",
            "rokae_tianji_sign": ROKAE_TIANJI_SIGN.tolist(),
        },
        "hand_joint_order": order_contract,
    }


def dataset_info(
    episodes: int,
    frames: int,
    tasks: int,
    schema: dict[str, Any],
) -> dict[str, Any]:
    return {
        "codebase_version": "v2.1",
        "robot_type": "rokae_wuji1_physx_v3_actual_joint54",
        "total_episodes": episodes,
        "total_frames": frames,
        "total_tasks": tasks,
        "total_videos": episodes * 3,
        "total_chunks": math.ceil(episodes / 1000),
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{episodes}"},
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "handson_policy_schema": schema,
        "features": {
            "observation.images.front": common.video_feature(640, 640),
            "observation.images.wrist_left": common.video_feature(480, 848),
            "observation.images.wrist_right": common.video_feature(480, 848),
            "observation.state": {
                "dtype": "float32",
                "shape": [JOINT54_DIM],
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": [JOINT54_DIM],
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
    return pa.table(
        {
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(state.reshape(-1)), JOINT54_DIM
            ),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(action.reshape(-1)), JOINT54_DIM
            ),
            "timestamp": pa.array(
                np.arange(length, dtype=np.float32) / np.float32(FPS)
            ),
            "frame_index": pa.array(np.arange(length, dtype=np.int64)),
            "episode_index": pa.array(
                np.full(length, episode_index, dtype=np.int64)
            ),
            "index": pa.array(
                np.arange(global_start, global_start + length, dtype=np.int64)
            ),
            "task_index": pa.array(np.full(length, task_index, dtype=np.int64)),
        }
    )


def main() -> None:
    args = parse_args()
    if ISAAC_SITE_PACKAGES.exists():
        sys.path.append(str(ISAAC_SITE_PACKAGES))
    import zarr  # noqa: PLC0415

    from openpi.shared import normalize  # noqa: PLC0415

    rows = sorted(
        common.load_json(args.manifest)["episodes"],
        key=lambda row: row["prefix"],
    )
    if args.limit > 0:
        rows = rows[: args.limit]
    prompt_counts = common.preflight_task_prompts(args.raw_root, rows)

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
        required = {
            "cert": episode_dir / "physical_replay_certification.json",
            "marker": episode_dir / "DATASET_PHYSX_V3.json",
            "collection_meta": episode_dir / "collection_meta.json",
            "schema": episode_dir / "recording_schema.json",
            "zarr": episode_dir / "replay.zarr",
            "front": episode_dir / "videos/main.mp4",
            "wrist_left": episode_dir / "videos/wrist_left.mp4",
            "wrist_right": episode_dir / "videos/wrist_right.mp4",
        }
        missing = [
            str(path)
            for key, path in required.items()
            if not path.exists()
            and not (
                args.include_source_success_without_certification and key == "cert"
            )
        ]
        if missing:
            skipped.append({"source": str(episode_dir), "missing": missing})
            continue
        marker = common.load_json(required["marker"])
        collection_meta = common.load_json(required["collection_meta"])
        prompt = common.task_prompt(task, collection_meta.get("target_label"))
        cert = common.load_json(required["cert"]) if required["cert"].exists() else {}
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
                    "source": str(episode_dir),
                    "reason": "not_eligible_successful_physx_v3",
                    "cert_status": cert.get("status"),
                    "final_succeed": cert.get("final_succeed"),
                    "source_outcome": collection_meta.get("source_outcome"),
                    "task_outcome": collection_meta.get("task_outcome"),
                }
            )
            continue
        group = zarr.open_group(str(required["zarr"]), mode="r")
        keys = ("physx_robot_joint_position", "next_physx_robot_joint_position")
        lengths = [int(group[f"data/{key}"].shape[0]) for key in keys]
        length = min(lengths)
        try:
            if length < 2 or len(set(lengths)) != 1:
                raise ValueError(f"misaligned lengths: {lengths}")
            common.require_all_valid(
                group,
                ("physx_robot_joint_valid", "next_physx_robot_joint_valid"),
                length,
            )
            names = common.load_robot_joint_names(required["schema"])
            order_contract = wuji_hand_order_contract(names)
            if canonical_order_contract is None:
                canonical_order_contract = order_contract
            elif order_contract != canonical_order_contract:
                raise ValueError("joint ordering differs from earlier episodes")
            state = actual_joint54(
                group["data/physx_robot_joint_position"][:length], names
            )
            action = actual_joint54(
                group["data/next_physx_robot_joint_position"][:length], names
            )
        except (KeyError, ValueError) as exc:
            skipped.append(
                {"source": str(episode_dir), "reason": "invalid", "detail": str(exc)}
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
        videos = {
            "observation.images.front": required["front"],
            "observation.images.wrist_left": required["wrist_left"],
            "observation.images.wrist_right": required["wrist_right"],
        }
        for video_key, source in videos.items():
            destination = (
                args.output_root
                / "videos"
                / f"chunk-{chunk:03d}"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(source.resolve(), destination)
        common.append_jsonl(
            args.output_root / "meta/episodes.jsonl",
            {"episode_index": episode_index, "tasks": [prompt], "length": length},
        )
        common.append_jsonl(
            args.output_root / "meta/episodes_stats.jsonl",
            {
                "episode_index": episode_index,
                "stats": {
                    "observation.state": common.feature_stats(state),
                    "action": common.feature_stats(action),
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
            print(f"converted={len(converted)}/{len(rows)} frames={global_index}")

    if canonical_order_contract is None:
        raise RuntimeError("No episode supplied a valid joint-order contract")
    schema = policy_schema(canonical_order_contract)
    for prompt, task_index in sorted(task_to_index.items(), key=lambda item: item[1]):
        common.append_jsonl(
            args.output_root / "meta/tasks.jsonl",
            {"task_index": task_index, "task": prompt},
        )
    common.write_json(
        args.output_root / "meta/info.json",
        dataset_info(len(converted), global_index, len(task_to_index), schema),
    )
    common.write_json(args.output_root / "meta/handson_policy_schema.json", schema)
    common.write_json(
        args.output_root / "conversion_manifest.json",
        {
            "format": "handson.policy_dataset_conversion",
            "version": 1,
            "representation": "actual_joint54",
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
            "policy_schema": schema,
            "joint_order_conversion": canonical_order_contract,
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
    common.write_json(args.assets_dir / "handson_policy_schema.json", schema)
    print(
        json.dumps(
            {
                "converted_count": len(converted),
                "skipped_count": len(skipped),
                "total_frames": global_index,
                "task_counts": dict(sorted(task_counts.items())),
                "prompt_counts": dict(sorted(prompt_counts.items())),
                "dataset_root": str(args.output_root),
                "assets_dir": str(args.assets_dir),
            },
            indent=2,
        )
    )
    if skipped:
        raise SystemExit(f"Conversion skipped {len(skipped)} episodes")


if __name__ == "__main__":
    main()
