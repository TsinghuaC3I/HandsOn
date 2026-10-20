#!/usr/bin/env python3
"""Certify and mark a legacy actual-EEF52 dataset's hand permutation.

The marker is written only after every Parquet hand value is compared with the
name-resolved PhysX readback for both state and action.  This is the migration
path for datasets created before the versioned policy marker existed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq


HANDSON_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "handson"
ISAAC_SITE_PACKAGES = Path(
    "/mnt/pfs/luotianwei/envs/isaacsim-5.1-handson/lib/python3.11/site-packages"
)
for package_path in (HANDSON_PACKAGE_ROOT, ISAAC_SITE_PACKAGES):
    if package_path.exists() and str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))

import zarr  # noqa: E402

from handson.policy_joint_order import (  # noqa: E402
    ACTUAL_EEF52_POLICY_SCHEMA,
    ROBOT_SIDE_ORDER,
    reorder_articulation_hands_to_wuji_control,
    wuji_hand_order_contract,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def robot_joint_names(episode: Path) -> dict[str, list[str]]:
    schema = load_json(episode / "recording_schema.json")
    try:
        names = schema["environment_schema"]["physx"]["robot_joint_names"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing robot_joint_names in {episode}") from exc
    result = {}
    for side in ROBOT_SIDE_ORDER:
        values = names.get(side) if isinstance(names, dict) else None
        if not isinstance(values, list):
            raise ValueError(f"Missing {side} robot_joint_names in {episode}")
        result[side] = [str(value) for value in values]
    return result


def fixed_list_values(path: Path, key: str, width: int) -> np.ndarray:
    column = pq.read_table(path, columns=[key])[key].combine_chunks()
    values = column.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(len(column), width)


def policy_schema(order_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "handson.policy_schema",
        "version": 1,
        "schema_id": ACTUAL_EEF52_POLICY_SCHEMA,
        "state_schema": "actual_before_action_right26_left26",
        "action_schema": "actual_after_action_right26_left26",
        "state_dim": 52,
        "action_dim": 52,
        "side_order": list(ROBOT_SIDE_ORDER),
        "per_side_layout": ["tcp_xyz3", "tcp_rotvec3", "hand_joint20"],
        "hand_joint_order": order_contract,
    }


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Refusing to replace a different marker: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    manifest = load_json(args.dataset / "conversion_manifest.json")
    info = load_json(args.dataset / "meta/info.json")
    if info["features"]["observation.state"]["shape"] != [52]:
        raise ValueError("Dataset state is not 52-D")
    if info["features"]["action"]["shape"] != [52]:
        raise ValueError("Dataset action is not 52-D")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Conversion manifest has no episodes")

    reference_contract: dict[str, Any] | None = None
    value_digest = hashlib.sha256()
    total_frames = 0
    for number, row in enumerate(episodes, start=1):
        episode_index = int(row["episode_index"])
        episode = Path(row["source"])
        names = robot_joint_names(episode)
        contract = wuji_hand_order_contract(names)
        if reference_contract is None:
            reference_contract = contract
        elif contract != reference_contract:
            raise ValueError(f"Joint-order contract changed at {episode}")

        parquet_path = (
            args.dataset
            / "data"
            / f"chunk-{episode_index // 1000:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        state = fixed_list_values(parquet_path, "observation.state", 52)
        action = fixed_list_values(parquet_path, "action", 52)
        group = zarr.open_group(str(episode / "replay.zarr"), mode="r")
        state_hand = reorder_articulation_hands_to_wuji_control(
            group["data/physx_robot_joint_position"][:], names
        ).astype(np.float32)
        action_hand = reorder_articulation_hands_to_wuji_control(
            group["data/next_physx_robot_joint_position"][:], names
        ).astype(np.float32)
        if (
            state.shape[0] != state_hand.shape[0]
            or action.shape[0] != action_hand.shape[0]
        ):
            raise ValueError(f"Frame-count mismatch for {episode}")
        actual_state_hand = np.stack((state[:, 6:26], state[:, 32:52]), axis=1)
        actual_action_hand = np.stack((action[:, 6:26], action[:, 32:52]), axis=1)
        if not np.array_equal(actual_state_hand, state_hand):
            raise ValueError(f"State hand permutation/value mismatch for {episode}")
        if not np.array_equal(actual_action_hand, action_hand):
            raise ValueError(f"Action hand permutation/value mismatch for {episode}")
        value_digest.update(actual_state_hand.tobytes(order="C"))
        value_digest.update(actual_action_hand.tobytes(order="C"))
        total_frames += state.shape[0]
        if number % 25 == 0 or number == len(episodes):
            print(
                f"verified={number}/{len(episodes)} frames={total_frames}",
                flush=True,
            )

    if reference_contract is None:
        raise RuntimeError("No joint-order contract was verified")
    legacy_ids = manifest.get("hand_control_joint_ids")
    expected_ids = reference_contract["source_indices_in_target_order"]["right"]
    if legacy_ids is not None and legacy_ids != expected_ids:
        raise ValueError("Legacy manifest hand_control_joint_ids disagree with names")

    marker = policy_schema(reference_contract)
    marker["certification"] = {
        "format": "handson.policy_schema.numeric_certification",
        "version": 1,
        "method": "all_frames_parquet_vs_name_resolved_physx_readback",
        "episodes_verified": len(episodes),
        "frames_verified": total_frames,
        "hand_values_sha256": value_digest.hexdigest(),
        "evidence": str(args.evidence) if args.evidence is not None else None,
    }
    print(json.dumps(marker, ensure_ascii=False, indent=2))
    if not args.write:
        print("AUDIT_ONLY: pass --write to install the marker")
        return
    write_new_json(args.dataset / "meta/handson_policy_schema.json", marker)
    write_new_json(args.assets_dir / "handson_policy_schema.json", marker)
    print("MARKERS_WRITTEN")


if __name__ == "__main__":
    main()
