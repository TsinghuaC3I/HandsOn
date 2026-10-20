"""Contract helpers for detached-hand collection with offline ROKAE IK."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


SIMPLIFIED_POST_IK_PROFILE = "rokae_simplified_post_ik"
SIMPLIFIED_POST_IK_REPLAY_SCOPE = (
    "physx_task_and_detached_wuji_hands_plus_validated_offline_rokae_ik"
)
SIMPLIFIED_POST_IK_TCP_SOURCE = "physx_tcp_state_with_physx_tcp_valid"
SIMPLIFIED_ARM_PLAN_PREFIX = "simplified_arm_plan"
SIMPLIFIED_ARM_PLAN_FILENAME = "simplified_arm_plan.npz"


def declares_simplified_post_ik(values: Mapping[str, Any]) -> bool:
    """Read either the new profile field or its requested-arm alias."""

    candidates = (
        values.get("arm_control_profile"),
        values.get("requested_arm_type"),
        values.get("arm_type"),
    )
    return any(
        str(value or "").strip().lower().replace("-", "_")
        == SIMPLIFIED_POST_IK_PROFILE
        for value in candidates
    )


def simplified_post_ik_contract_errors(
    recording: Mapping[str, Any],
    collection_meta: Mapping[str, Any],
) -> list[str]:
    """Validate the immutable/mutable mirrored collection declaration."""

    manifest_declared = declares_simplified_post_ik(recording)
    metadata_declared = declares_simplified_post_ik(collection_meta)
    if manifest_declared != metadata_declared:
        return [
            "collection_meta simplified-post-IK declaration differs from "
            "the replay manifest"
        ]
    if not manifest_declared:
        return []
    errors: list[str] = []
    for label, values in (
        ("replay_manifest.recording", recording),
        ("collection_meta", collection_meta),
    ):
        if values.get("arm_control_profile") != SIMPLIFIED_POST_IK_PROFILE:
            errors.append(f"{label}.arm_control_profile is invalid")
        if values.get("simplified_post_ik") is not True:
            errors.append(f"{label}.simplified_post_ik must be true")
        if (
            values.get("simplified_post_ik_replay_scope")
            != SIMPLIFIED_POST_IK_REPLAY_SCOPE
        ):
            errors.append(f"{label}.simplified_post_ik_replay_scope is invalid")
        if values.get("arm_ik_deferred") is not True:
            errors.append(f"{label}.arm_ik_deferred must be true")
        if (
            values.get("simplified_post_ik_tcp_source")
            != SIMPLIFIED_POST_IK_TCP_SOURCE
        ):
            errors.append(f"{label}.simplified_post_ik_tcp_source is invalid")
    return errors


def load_validated_simplified_arm_plan(
    replay_path: Path,
    *,
    expected_frames: int | None = None,
) -> np.ndarray:
    """Load a plan only when every IK/FK/collision postcheck is valid."""

    import json

    from handson.runtime.replay_archive import replay_tree_identity

    replay = Path(replay_path)
    plan_path = (
        replay
        if replay.name == SIMPLIFIED_ARM_PLAN_FILENAME
        else replay.parent / SIMPLIFIED_ARM_PLAN_FILENAME
    )
    replay = replay.parent / "replay.zarr" if replay == plan_path else replay
    if not replay.is_dir():
        raise ValueError(f"{replay}: source replay.zarr is missing")
    try:
        archive = np.load(plan_path, allow_pickle=False)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(f"{plan_path}: cannot read simplified arm plan: {exc}") from exc
    with archive as data:
        if "contract_json" not in data.files:
            raise ValueError(f"{plan_path}: missing contract_json")
        try:
            contract = json.loads(str(np.asarray(data["contract_json"]).item()))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{plan_path}: invalid plan contract: {exc}") from exc
        if not isinstance(contract, dict):
            raise ValueError(f"{plan_path}: plan contract must be an object")
        if (
            contract.get("format") != "handson.simplified_post_ik_plan"
            or contract.get("version") != 1
        ):
            raise ValueError(f"{plan_path}: unsupported plan contract")
        if contract.get("source_replay_tree_identity") != replay_tree_identity(replay):
            raise ValueError(f"{plan_path}: plan is stale or belongs to another replay")
        attrs = contract.get("attrs")
        if (
            not isinstance(attrs, dict)
            or attrs.get("simplified_arm_plan_tcp_pose_dataset")
            != "physx_tcp_state"
            or attrs.get("simplified_arm_plan_tcp_valid_dataset")
            != "physx_tcp_valid"
        ):
            raise ValueError(
                f"{plan_path}: plan is not derived from certified PhysX TCP "
                "readback"
            )
        required = (
            "qpos",
            "valid_frame",
            "fk_valid_frame",
            "collision_valid_frame",
        )
        arrays: dict[str, np.ndarray] = {}
        for suffix in required:
            name = f"{SIMPLIFIED_ARM_PLAN_PREFIX}_{suffix}"
            if name not in data.files:
                raise ValueError(f"{plan_path}: missing {name}")
            arrays[suffix] = np.asarray(data[name])
    qpos = np.asarray(arrays["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 14 or not np.all(np.isfinite(qpos)):
        raise ValueError(
            f"{plan_path}: simplified arm qpos must be finite (frames, 14), got "
            f"{qpos.shape}"
        )
    if contract.get("frame_count") != len(qpos):
        raise ValueError(
            f"{plan_path}: contract frame_count does not match qpos "
            f"({contract.get('frame_count')!r} != {len(qpos)})"
        )
    if expected_frames is not None and len(qpos) != int(expected_frames):
        raise ValueError(
            f"{plan_path}: simplified arm plan frame mismatch "
            f"{len(qpos)} != {int(expected_frames)}"
        )
    for suffix in required[1:]:
        values = np.asarray(arrays[suffix]).reshape(-1)
        if len(values) != len(qpos) or not np.all(values == 1):
            invalid = int(np.count_nonzero(values != 1))
            raise ValueError(
                f"{plan_path}: {SIMPLIFIED_ARM_PLAN_PREFIX}_{suffix} has "
                f"{invalid} invalid frame(s)"
            )
    return qpos


def simplified_arm_plan_errors(
    replay_path: Path,
    *,
    expected_frames: int | None = None,
) -> list[str]:
    try:
        load_validated_simplified_arm_plan(
            replay_path,
            expected_frames=expected_frames,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return [str(exc)]
    return []


def simplified_arm_plan_artifact_identity(demo_dir: Path) -> dict[str, Any]:
    path = Path(demo_dir) / SIMPLIFIED_ARM_PLAN_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path}: simplified arm plan is missing")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if size <= 0:
        raise ValueError(f"{path}: simplified arm plan is empty")
    return {
        "path": SIMPLIFIED_ARM_PLAN_FILENAME,
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def simplified_arm_plan_artifact_errors(
    demo_dir: Path,
    proof: Any,
) -> list[str]:
    if not isinstance(proof, Mapping):
        return ["simplified arm plan artifact proof is missing"]
    try:
        actual = simplified_arm_plan_artifact_identity(demo_dir)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    return (
        []
        if dict(proof) == actual
        else ["simplified arm plan changed after canonical postrender"]
    )


__all__ = [
    "SIMPLIFIED_ARM_PLAN_PREFIX",
    "SIMPLIFIED_ARM_PLAN_FILENAME",
    "SIMPLIFIED_POST_IK_PROFILE",
    "SIMPLIFIED_POST_IK_REPLAY_SCOPE",
    "SIMPLIFIED_POST_IK_TCP_SOURCE",
    "declares_simplified_post_ik",
    "load_validated_simplified_arm_plan",
    "simplified_arm_plan_artifact_errors",
    "simplified_arm_plan_artifact_identity",
    "simplified_arm_plan_errors",
    "simplified_post_ik_contract_errors",
]
