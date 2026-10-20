from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from handson.data.simplified_post_ik import (
    SIMPLIFIED_POST_IK_PROFILE,
    SIMPLIFIED_POST_IK_REPLAY_SCOPE,
    SIMPLIFIED_POST_IK_TCP_SOURCE,
    declares_simplified_post_ik,
    load_validated_simplified_arm_plan,
    simplified_arm_plan_artifact_errors,
    simplified_arm_plan_artifact_identity,
    simplified_post_ik_contract_errors,
)
from handson.runtime.replay_archive import replay_tree_identity
from handson.sim.envs.hand_specs import (
    is_simplified_post_ik_arm_type,
    normalize_arm_type,
)
from handson.sim.isaaclab.e01_detached_hand_scene import (
    prepare_e01_detached_wuji_hand_mjcf,
)
from handson.sim.isaaclab.e01_detached_hand_control import (
    NumPyWujiPrimitiveCollisionGuard,
)


def _declaration() -> dict[str, object]:
    return {
        "arm_type": SIMPLIFIED_POST_IK_PROFILE,
        "requested_arm_type": SIMPLIFIED_POST_IK_PROFILE,
        "arm_control_profile": SIMPLIFIED_POST_IK_PROFILE,
        "simplified_post_ik": True,
        "simplified_post_ik_replay_scope": SIMPLIFIED_POST_IK_REPLAY_SCOPE,
        "simplified_post_ik_tcp_source": SIMPLIFIED_POST_IK_TCP_SOURCE,
        "arm_ik_deferred": True,
    }


def _write_plan(demo: Path, *, frames: int = 3) -> None:
    replay = demo / "replay.zarr"
    prefix = "simplified_arm_plan_"
    contract = {
        "format": "handson.simplified_post_ik_plan",
        "version": 1,
        "frame_count": frames,
        "source_replay_tree_identity": replay_tree_identity(replay),
        "attrs": {
            "simplified_arm_plan_tcp_pose_dataset": "physx_tcp_state",
            "simplified_arm_plan_tcp_valid_dataset": "physx_tcp_valid",
        },
    }
    np.savez_compressed(
        demo / "simplified_arm_plan.npz",
        contract_json=np.asarray(json.dumps(contract)),
        **{
            prefix + "qpos": np.zeros((frames, 14), dtype=np.float64),
            prefix + "valid_frame": np.ones((frames, 1), dtype=np.uint8),
            prefix + "fk_valid_frame": np.ones((frames, 1), dtype=np.uint8),
            prefix + "collision_valid_frame": np.ones(
                (frames, 1), dtype=np.uint8
            ),
        },
    )


def _replace_plan_contract(demo: Path, patch: dict[str, object]) -> None:
    path = demo / "simplified_arm_plan.npz"
    with np.load(path, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    contract = json.loads(str(payload["contract_json"].item()))
    contract.update(patch)
    payload["contract_json"] = np.asarray(json.dumps(contract))
    np.savez_compressed(path, **payload)


def test_simplified_arm_type_keeps_rokae_asset_family() -> None:
    assert is_simplified_post_ik_arm_type(SIMPLIFIED_POST_IK_PROFILE)
    assert normalize_arm_type(SIMPLIFIED_POST_IK_PROFILE) == "rokae"


def test_simplified_contract_must_be_mirrored() -> None:
    declaration = _declaration()
    assert declares_simplified_post_ik(declaration)
    assert not simplified_post_ik_contract_errors(declaration, declaration)
    assert simplified_post_ik_contract_errors(declaration, {})


def test_external_plan_is_bound_to_unchanged_replay_tree(tmp_path: Path) -> None:
    replay = tmp_path / "replay.zarr"
    replay.mkdir()
    (replay / "source").write_bytes(b"source-v1")
    _write_plan(tmp_path)

    qpos = load_validated_simplified_arm_plan(replay, expected_frames=3)
    assert qpos.shape == (3, 14)
    proof = simplified_arm_plan_artifact_identity(tmp_path)
    assert not simplified_arm_plan_artifact_errors(tmp_path, proof)

    (replay / "source").write_bytes(b"source-v2")
    with pytest.raises(ValueError, match="stale"):
        load_validated_simplified_arm_plan(replay)


def test_postrender_proof_detects_plan_replacement(tmp_path: Path) -> None:
    replay = tmp_path / "replay.zarr"
    replay.mkdir()
    (replay / "source").write_bytes(b"source")
    _write_plan(tmp_path)
    proof = simplified_arm_plan_artifact_identity(tmp_path)
    with (tmp_path / "simplified_arm_plan.npz").open("ab") as stream:
        stream.write(b"changed")
    assert simplified_arm_plan_artifact_errors(tmp_path, proof)


def test_plan_contract_frame_count_must_match_payload(tmp_path: Path) -> None:
    replay = tmp_path / "replay.zarr"
    replay.mkdir()
    (replay / "source").write_bytes(b"source")
    _write_plan(tmp_path, frames=3)
    _replace_plan_contract(tmp_path, {"frame_count": 2})
    with pytest.raises(ValueError, match="contract frame_count"):
        load_validated_simplified_arm_plan(replay)


def test_detached_hands_keep_complete_collision_topology(
    tmp_path: Path,
) -> None:
    for side in ("right", "left"):
        path = prepare_e01_detached_wuji_hand_mjcf(
            side,
            cache_dir=tmp_path / side,
            ensure_proximal_finger_collisions=True,
        )
        root = ET.parse(path).getroot()
        collision_names = [
            str(geom.attrib.get("name") or "")
            for geom in root.iter("geom")
            if geom.attrib.get("contype", "1") != "0"
            or geom.attrib.get("conaffinity", "1") != "0"
        ]
        rigid_body_names = [
            str(body.attrib.get("name") or "")
            for body in root.iter("body")
            if body.attrib.get("name") == f"{side}_palm_link"
            or str(body.attrib.get("name") or "").startswith(
                f"{side}_finger"
            )
        ]
        assert len(collision_names) == 26
        assert len(rigid_body_names) == 21
        assert sum(name.endswith("link1_col2") for name in collision_names) == 5


def test_table_guard_matches_added_proximal_colliders() -> None:
    for side in ("right", "left"):
        legacy = NumPyWujiPrimitiveCollisionGuard(side)
        complete = NumPyWujiPrimitiveCollisionGuard(
            side,
            ensure_proximal_finger_collisions=True,
        )
        assert len(legacy._geoms) == 21
        assert len(complete._geoms) == 26
