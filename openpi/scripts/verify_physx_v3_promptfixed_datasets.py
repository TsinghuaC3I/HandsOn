#!/usr/bin/env python3
"""Verify paired EEF52/Joint54 PhysX-v3 datasets after prompt repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_physx_v3_actual_eef52_lerobot_v21 as converter  # noqa: E402


VIDEO_KEYS = (
    "observation.images.front",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eef-root", type=Path, required=True)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=7395)
    parser.add_argument("--check-all-files", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_root(root: Path, expected: int) -> tuple[dict, list[dict], list[dict]]:
    manifest = load_json(root / "conversion_manifest.json")
    info = load_json(root / "meta/info.json")
    episodes = load_jsonl(root / "meta/episodes.jsonl")
    tasks = load_jsonl(root / "meta/tasks.jsonl")
    assert manifest["candidate_count"] == expected
    assert manifest["converted_count"] == expected
    assert manifest["skipped_count"] == 0
    assert len(manifest["episodes"]) == expected
    assert info["total_episodes"] == expected
    assert len(episodes) == expected
    assert info["total_frames"] == manifest["total_frames"]
    task_by_index = {row["task_index"]: row["task"] for row in tasks}
    assert len(task_by_index) == len(tasks) == info["total_tasks"]
    assert set(task_by_index.values()) == {
        row["prompt"] for row in manifest["episodes"]
    }
    for index, (manifest_row, episode_row) in enumerate(
        zip(manifest["episodes"], episodes, strict=True)
    ):
        assert manifest_row["episode_index"] == index
        assert episode_row["episode_index"] == index
        assert episode_row["tasks"] == [manifest_row["prompt"]]
        assert episode_row["length"] == manifest_row["frames"]
        expected_prompt = converter.task_prompt(
            manifest_row["task"], manifest_row.get("target_label")
        )
        assert manifest_row["prompt"] == expected_prompt
    return manifest, episodes, tasks


def validate_files(root: Path, manifest: dict, state_dim: int) -> None:
    total_rows = 0
    for row in manifest["episodes"]:
        episode_index = int(row["episode_index"])
        chunk = episode_index // 1000
        parquet_path = (
            root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        )
        parquet = pq.ParquetFile(parquet_path)
        assert parquet.metadata.num_rows == row["frames"]
        schema = parquet.schema_arrow
        assert schema.field("observation.state").type.list_size == state_dim
        assert schema.field("action").type.list_size == state_dim
        total_rows += parquet.metadata.num_rows
        for key in VIDEO_KEYS:
            video = (
                root
                / "videos"
                / f"chunk-{chunk:03d}"
                / key
                / f"episode_{episode_index:06d}.mp4"
            )
            assert video.exists(), video
    assert total_rows == manifest["total_frames"]


def main() -> None:
    args = parse_args()
    eef_manifest, _, _ = validate_root(args.eef_root, args.expected_episodes)
    joint_manifest, _, _ = validate_root(args.joint_root, args.expected_episodes)
    assert eef_manifest["total_frames"] == joint_manifest["total_frames"]
    for eef, joint in zip(
        eef_manifest["episodes"], joint_manifest["episodes"], strict=True
    ):
        for key in ("episode_index", "source", "task", "target_label", "prompt", "frames"):
            assert eef[key] == joint[key], (key, eef, joint)
    if args.check_all_files:
        validate_files(args.eef_root, eef_manifest, 52)
        validate_files(args.joint_root, joint_manifest, 54)
    print(
        json.dumps(
            {
                "status": "passed",
                "episodes": args.expected_episodes,
                "frames": eef_manifest["total_frames"],
                "tasks": len(set(row["task"] for row in eef_manifest["episodes"])),
                "prompts": len(set(row["prompt"] for row in eef_manifest["episodes"])),
                "checked_all_parquets_and_videos": args.check_all_files,
                "eef_root": str(args.eef_root),
                "joint_root": str(args.joint_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
