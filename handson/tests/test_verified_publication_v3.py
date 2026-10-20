from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from handson.data.deferred_arm_replay_artifacts import (
    CANONICAL_ARM_VIDEO_PATHS,
    capture_deferred_arm_replay_artifacts,
)
from handson.data.physical_replay_certification import capture_certified_inputs
from handson.data.recording_fields import wall_sampling_contract
from handson.data.verified_publication_v3 import (
    OFFLINE_ARM_ARTIFACT_PROOF_FORMAT,
    OFFLINE_ARM_ARTIFACT_PROOF_VERSION,
    OFFLINE_ARM_VIDEO_PATHS,
    PUBLICATION_COMMIT_FILENAME,
    VERIFIED_EPISODE_FILENAME,
    VERIFIED_PUBLICATION_CONTRACT,
    VERIFIED_PUBLICATION_SCHEMA_VERSION,
    VERIFIED_PUBLICATION_SUFFIX,
    VerifiedPublicationV3Error,
    build_episode_verified_v3,
    build_publication_commit_v3,
    canonical_json_bytes,
    publication_commit_v3_errors,
    publication_id_for_episode,
    validate_publication_commit_v3,
    validate_verified_episode_v3,
    verified_episode_v3_errors,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _verified_demo(tmp_path: Path) -> Path:
    demo = tmp_path / "t26_place_fruit_in_bucket_demo_1_2026-08-02_18-00-00_000001"
    demo.mkdir()
    (demo / "replay.zarr").mkdir()
    (demo / "replay.zarr" / ".zgroup").write_bytes(b'{"zarr_format":2}')

    step_count = 4
    metadata = {
        "task_name": "t26_place_fruit_in_bucket",
        "collector_id": "collector_001",
        "collection_collector_id": "redundant_collector",
        "collection_session_id": "session-001",
        "collected_at": "2026-08-02T18:00:05+08:00",
        "duration_seconds": 0.133333,
        "step_count": step_count,
        "data_fps": 30.0,
        "video_fps": 30,
        "target_label": "apple",
        "arm_type": "rokae",
        "hand_type": "wuji1",
        "sim_backend": "isaacsim",
        "engine_code": "E02",
        "task_outcome": "success",
        "physical_replay_certification_required": True,
        "isaac_arm_render_deferred": True,
        "recording_camera_profile": {
            "version": "camera_pose_v6",
            "wrist": {"large_redundant_matrix": list(range(100))},
        },
        # These private values must never be projected into the envelope.
        "command": ["/home/user/isaacsim/python.sh", "--secret"],
        "log_path": "/home/user/private/render.log",
        "endpoint": "ks3.example.invalid",
        "environment": {"HANDSON_PRIVATE": "/home/user/private"},
    }
    _write_json(demo / "collection_meta.json", metadata)

    manifest = {
        "format": "handson.replay_manifest",
        "version": 1,
        "task": {"name": "t26_place_fruit_in_bucket"},
        "recording": {
            "step_count": step_count,
            "recording_camera_profile": {"version": "camera_pose_v6"},
            "physical_replay_certification_mode": "deferred_post_collection",
            "physical_replay_certification_required": True,
            "isaac_arm_render_deferred": True,
        },
        "environment": {
            "variables": {
                "HANDSON_ROOT": "/home/user/handson-main",
                "HANDSON_KS3_ENDPOINT": "https://ks3.example.invalid",
            }
        },
    }
    _write_json(demo / "replay_manifest.json", manifest)
    _write_json(
        demo / "recording_schema.json",
        {"format": "test.recording_schema", "fields": {"action": [27]}},
    )
    certified_inputs = capture_certified_inputs(demo)

    _write_json(
        demo / "physical_replay_certification.json",
        {
            "format": "handson.physical_replay_certification",
            "version": 3,
            "status": "passed",
            "portable": True,
            "source_identity": "clean_commit_match",
            "certification_mode": "task_success_from_fresh_seeded_physx_origin",
            "task_success_replay": True,
            "trace_match": True,
            "transition_outcomes_match": True,
            "certified_inputs": certified_inputs,
            "current_git": {
                "root": "/home/user/handson-main",
                "large_diagnostic": list(range(100)),
            },
            "environment": {"PRIVATE": "/home/user/private"},
        },
    )
    _write_json(
        demo / "fresh_process_acceptance.json",
        {
            "format": "handson.deferred_physical_replay_acceptance",
            "version": 1,
            "status": "passed",
            "accepted_at": "2026-08-02T18:05:00+08:00",
            "certification_scope": "ordered_collection_session",
            "certification_runs_required": 3,
            "certification_runs_passed": 3,
            "published_after_post_collection_certification": True,
            "certified_inputs": certified_inputs,
            "certification_runs": [
                {
                    "pid": index + 100,
                    "command": ["/home/user/isaacsim/python.sh"],
                    "log_path": f"/home/user/private/run-{index}.log",
                }
                for index in range(3)
            ],
        },
    )

    for index, relative_path in enumerate(CANONICAL_ARM_VIDEO_PATHS, start=1):
        path = demo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([index]) * (20 + index))
    artifact_proof = capture_deferred_arm_replay_artifacts(demo)
    _write_json(
        demo / "deferred_arm_replay_status.json",
        {
            "status": "complete",
            "completed_at": "2026-08-02T18:06:00+08:00",
            "updated_at": "2026-08-02T18:06:00+08:00",
            "video_fps": 30,
            "artifact_validation_passed": True,
            "artifact_proof": artifact_proof,
            "command": [
                "/home/user/isaacsim/python.sh",
                *[f"--large-argument-{index}" for index in range(225)],
            ],
            "log_path": "/home/user/private/deferred-render.log",
            "endpoint": "https://ks3.example.invalid",
            "environment": {"PATH": "/home/user/private/bin"},
        },
    )

    proof_by_path = {row["path"]: row for row in artifact_proof["files"]}
    _write_json(
        demo / "wrist_rerender_meta.json",
        {
            "format": "handson.camera_rerender",
            "version": 6,
            "status": "replaced_locally",
            "method": "isaaclab_continuous_physics_camera_pose_v6",
            "camera_scope": "main_and_wrists",
            "task_sites_hidden": True,
            "mocap_target_helpers_hidden": True,
            "wrist_camera_models_hidden": True,
            "quality_gate": {
                "continuous_physics": True,
                "per_frame_state_restore": False,
                "success_in_tail": True,
                "video_validation": "passed",
                "frame_count": step_count,
            },
            "videos": [
                {
                    "path": relative_path,
                    "bytes": proof_by_path[relative_path]["bytes"],
                    "new_sha256": proof_by_path[relative_path]["sha256"],
                    "frame_count": step_count,
                    "backup_path": f"/home/user/private/{Path(relative_path).name}",
                }
                for relative_path in CANONICAL_ARM_VIDEO_PATHS
            ],
            "stage_dir": "/home/user/private/v6-stage",
            "source_replay_zarr": str(demo / "replay.zarr"),
            "command": ["/home/user/private/repair.py"],
            "log_path": "/home/user/private/v6.log",
            "endpoint": "https://ks3.example.invalid",
        },
    )
    _write_json(
        demo / "camera_repair_v6_review.json",
        {
            "format": "handson.camera_repair_v6_review",
            "version": 1,
            "status": "replaced_locally",
            "tag": "高疑似异常",
            "updated_at": "2026-08-02T18:07:00+08:00",
            "worker_result": {
                "episode": str(demo),
                "command": ["/home/user/private/worker.py"],
            },
            "log_path": "/home/user/private/review.log",
        },
    )
    metadata["quality_review"] = {
        "status": "first_reviewed",
        "final_outcome": "valid",
        "result_code": "S01",
        "is_success": True,
        "reviewer_id": "reviewer-001",
        "reviewed_at": "2026-08-02T18:08:00+08:00",
        "canonical_video_sha256": {
            relative_path: proof_by_path[relative_path]["sha256"]
            for relative_path in CANONICAL_ARM_VIDEO_PATHS
        },
    }
    _write_json(demo / "collection_meta.json", metadata)
    return demo


def test_verified_v3_rejects_an_unqualified_real_sampling_rate(
    tmp_path: Path,
) -> None:
    demo = _verified_demo(tmp_path)
    contract = wall_sampling_contract(
        np.arange(4, dtype=np.float64) / 20.0,
        target_hz=30.0,
    )
    metadata = json.loads(
        (demo / "collection_meta.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (demo / "replay_manifest.json").read_text(encoding="utf-8")
    )
    metadata["wall_clock_sampling"] = contract
    manifest["recording"]["wall_clock_sampling"] = contract
    _write_json(demo / "collection_meta.json", metadata)
    _write_json(demo / "replay_manifest.json", manifest)

    with pytest.raises(VerifiedPublicationV3Error, match="not production-qualified"):
        build_episode_verified_v3(demo)


def test_publication_id_is_idempotent_and_uses_only_the_basename() -> None:
    source = Path("/tmp/source/episode_001")
    expected = "episode_001" + VERIFIED_PUBLICATION_SUFFIX
    assert publication_id_for_episode(source) == expected
    assert publication_id_for_episode(Path("/tmp/other") / expected) == expected


def test_build_episode_verified_v3_is_compact_and_does_not_leak_private_fields(
    tmp_path: Path,
) -> None:
    demo = _verified_demo(tmp_path)
    manifest_before = (demo / "replay_manifest.json").read_bytes()
    metadata_before = (demo / "collection_meta.json").read_bytes()

    envelope = build_episode_verified_v3(demo)

    assert envelope["contract"] == VERIFIED_PUBLICATION_CONTRACT
    assert envelope["schema_version"] == VERIFIED_PUBLICATION_SCHEMA_VERSION
    assert envelope["publication_id"] == demo.name + VERIFIED_PUBLICATION_SUFFIX
    assert envelope["source_episode_id"] == demo.name
    assert envelope["collector_id"] == "collector_001"
    assert envelope["physical_replay"] == {
        "status": "passed",
        "portable": True,
        "source_identity": "clean_commit_match",
        "certification_mode": "task_success_from_fresh_seeded_physx_origin",
        "certification_scope": "ordered_collection_session",
        "runs_required": 3,
        "runs_passed": 3,
        "accepted_at": "2026-08-02T18:05:00+08:00",
    }
    assert envelope["render"]["v6"]["status"] == "rerendered"
    assert len(envelope["render"]["files"]) == 3
    assert validate_verified_episode_v3(envelope) is None

    serialized = canonical_json_bytes(envelope).decode("utf-8")
    for forbidden in (
        "command",
        "log_path",
        "endpoint",
        "environment",
        "/home/user",
        "large_redundant_matrix",
        "large_diagnostic",
    ):
        assert forbidden not in serialized
    assert (demo / "replay_manifest.json").read_bytes() == manifest_before
    assert (demo / "collection_meta.json").read_bytes() == metadata_before

    with pytest.raises(VerifiedPublicationV3Error, match="publication_id"):
        build_episode_verified_v3(demo, publication_id="wrong__verified_v3")


def test_verified_v3_preserves_dirty_snapshot_reviewer_identity(
    tmp_path: Path,
) -> None:
    demo = _verified_demo(tmp_path)
    report_path = demo / "physical_replay_certification.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    snapshot_commit = "2" * 40
    report.update(
        {
            "source_identity": (
                "dirty_recording_exact_snapshot_reviewer_upgrade"
            ),
            "source_snapshot_commit": snapshot_commit,
            "source_reviewer_upgrade_paths": [
                "scripts/certify_physx_replay.py"
            ],
            "recorded_git": {
                "commit": "1" * 40,
                "dirty": True,
                "changed_files": ["collection_ui/server.py"],
                "worktree_content_sha256": "4" * 64,
            },
            "current_git": {
                "commit": "3" * 40,
                "dirty": False,
                "changed_files": [],
            },
        }
    )
    _write_json(report_path, report)

    envelope = build_episode_verified_v3(demo)

    assert envelope["physical_replay"]["source_identity"] == (
        "dirty_recording_exact_snapshot_reviewer_upgrade"
    )
    assert envelope["physical_replay"]["source_snapshot_commit"] == (
        snapshot_commit
    )
    assert validate_verified_episode_v3(envelope) is None

    missing_snapshot = copy.deepcopy(envelope)
    del missing_snapshot["physical_replay"]["source_snapshot_commit"]
    assert any(
        "v3 schema" in error
        for error in verified_episode_v3_errors(missing_snapshot)
    )


def test_verified_v3_rejects_snapshot_identity_with_extra_reviewer_path(
    tmp_path: Path,
) -> None:
    demo = _verified_demo(tmp_path)
    report_path = demo / "physical_replay_certification.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "source_identity": (
                "dirty_recording_exact_snapshot_reviewer_upgrade"
            ),
            "source_snapshot_commit": "2" * 40,
            "source_reviewer_upgrade_paths": ["unrelated.py"],
            "recorded_git": {
                "commit": "1" * 40,
                "dirty": True,
                "changed_files": ["collection_ui/server.py"],
                "worktree_content_sha256": "4" * 64,
            },
            "current_git": {
                "commit": "3" * 40,
                "dirty": False,
                "changed_files": [],
            },
        }
    )
    _write_json(report_path, report)

    with pytest.raises(VerifiedPublicationV3Error, match="reviewer upgrade"):
        build_episode_verified_v3(demo)


def test_quality_review_changes_do_not_change_immutable_envelope(
    tmp_path: Path,
) -> None:
    demo = _verified_demo(tmp_path)
    before = build_episode_verified_v3(demo)
    metadata = json.loads(
        (demo / "collection_meta.json").read_text(encoding="utf-8")
    )
    metadata["quality_review"]["reviewer_id"] = "reviewer-002"
    _write_json(demo / "collection_meta.json", metadata)

    after = build_episode_verified_v3(demo)

    assert after == before


@pytest.mark.parametrize(
    ("outcome", "is_success", "result_code"),
    (
        ("invalid", False, "F00"),
        ("suspect", False, ""),
        ("bug", False, ""),
    ),
)
def test_explicit_negative_quality_review_overrides_success_metadata(
    tmp_path: Path,
    outcome: str,
    is_success: bool,
    result_code: str,
) -> None:
    demo = _verified_demo(tmp_path)
    metadata = json.loads(
        (demo / "collection_meta.json").read_text(encoding="utf-8")
    )
    metadata["quality_review"].update(
        {
            "final_outcome": outcome,
            "is_success": is_success,
            "result_code": result_code,
        }
    )
    _write_json(demo / "collection_meta.json", metadata)

    with pytest.raises(VerifiedPublicationV3Error, match="quality review"):
        build_episode_verified_v3(demo)


def test_camera_repair_requires_current_hash_bound_human_review(
    tmp_path: Path,
) -> None:
    demo = _verified_demo(tmp_path)
    metadata_path = demo / "collection_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    metadata["quality_review"]["reviewer_id"] = "auto_v6"
    _write_json(metadata_path, metadata)
    with pytest.raises(VerifiedPublicationV3Error, match="human reviewer"):
        build_episode_verified_v3(demo)

    metadata["quality_review"]["reviewer_id"] = "reviewer-001"
    metadata["quality_review"]["reviewed_at"] = (
        "2026-08-02T18:06:59+08:00"
    )
    _write_json(metadata_path, metadata)
    with pytest.raises(VerifiedPublicationV3Error, match="predates"):
        build_episode_verified_v3(demo)

    metadata["quality_review"]["reviewed_at"] = (
        "2026-08-02T18:08:00+08:00"
    )
    metadata["quality_review"]["canonical_video_sha256"][
        "videos/main.mp4"
    ] = "0" * 64
    _write_json(metadata_path, metadata)
    with pytest.raises(VerifiedPublicationV3Error, match="hash is stale"):
        build_episode_verified_v3(demo)


def test_verified_episode_validator_rejects_forbidden_keys_and_absolute_paths(
    tmp_path: Path,
) -> None:
    envelope = build_episode_verified_v3(_verified_demo(tmp_path))
    leaked = copy.deepcopy(envelope)
    leaked["render"]["v6"]["command"] = ["secret"]
    leaked["render"]["v6"]["camera_profile"] = "/home/user/private/profile"

    errors = verified_episode_v3_errors(leaked)
    assert any("forbidden" in error for error in errors)
    assert any("absolute path" in error for error in errors)


def test_publication_commit_is_pure_compact_and_binds_canonical_envelope(
    tmp_path: Path,
) -> None:
    envelope = build_episode_verified_v3(_verified_demo(tmp_path))
    archive_sha = hashlib.sha256(b"archive").hexdigest()
    required_rows = [
        *envelope["render"]["files"],
    ]
    commit = build_publication_commit_v3(
        envelope,
        [
            {
                "path": "replay.zarr.zip",
                "bytes": 7,
                "sha256": archive_sha,
                "key": "/private/cloud/key",
                "endpoint": "https://ks3.example.invalid",
                "etag": "private",
            },
            *required_rows,
            {
                "path": VERIFIED_EPISODE_FILENAME,
                "bytes": 1,
                "sha256": "0" * 64,
            },
        ],
        committed_at="2026-08-02T18:07:00+08:00",
    )

    assert all(set(row) == {"path", "bytes", "sha256"} for row in commit["files"])
    assert [row["path"] for row in commit["files"]] == sorted(
        row["path"] for row in commit["files"]
    )
    assert PUBLICATION_COMMIT_FILENAME not in {
        row["path"] for row in commit["files"]
    }
    envelope_bytes = canonical_json_bytes(envelope)
    envelope_row = next(
        row for row in commit["files"] if row["path"] == VERIFIED_EPISODE_FILENAME
    )
    assert envelope_row == {
        "path": VERIFIED_EPISODE_FILENAME,
        "bytes": len(envelope_bytes),
        "sha256": hashlib.sha256(envelope_bytes).hexdigest(),
    }
    assert validate_publication_commit_v3(commit, envelope=envelope) is None
    serialized = canonical_json_bytes(commit).decode("utf-8")
    assert "endpoint" not in serialized
    assert "/private/cloud/key" not in serialized

    tampered = copy.deepcopy(commit)
    envelope_index = next(
        index
        for index, row in enumerate(tampered["files"])
        if row["path"] == VERIFIED_EPISODE_FILENAME
    )
    tampered["files"][envelope_index]["sha256"] = "f" * 64
    errors = publication_commit_v3_errors(tampered, envelope=envelope)
    assert any("envelope hash" in error for error in errors)


def test_builder_rejects_stale_certified_inputs(tmp_path: Path) -> None:
    demo = _verified_demo(tmp_path)
    (demo / "replay.zarr" / ".zgroup").write_bytes(b"changed")
    with pytest.raises(VerifiedPublicationV3Error, match="stale"):
        build_episode_verified_v3(demo)


def test_camera_postrender_declaration_uses_strict_three_video_proof(
    tmp_path: Path,
) -> None:
    demo = _verified_demo(tmp_path)
    metadata = json.loads(
        (demo / "collection_meta.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (demo / "replay_manifest.json").read_text(encoding="utf-8")
    )
    for owner in (metadata, manifest["recording"]):
        owner["physical_replay_certification_required"] = False
        owner["isaac_arm_render_deferred"] = False
        owner["canonical_camera_postrender_required"] = True
        owner["online_camera_capture_scope"] = "all_canonical_postrender"
    _write_json(demo / "collection_meta.json", metadata)
    _write_json(demo / "replay_manifest.json", manifest)

    envelope = build_episode_verified_v3(demo)
    assert envelope["render"]["status"] == "complete"
    assert len(envelope["render"]["files"]) == 3

    status = json.loads(
        (demo / "deferred_arm_replay_status.json").read_text(
            encoding="utf-8"
        )
    )
    status.pop("artifact_proof")
    _write_json(demo / "deferred_arm_replay_status.json", status)
    with pytest.raises(
        VerifiedPublicationV3Error,
        match="artifact proof",
    ):
        build_episode_verified_v3(demo)


def test_nonphysical_source_recording_does_not_require_deferred_proofs(
    tmp_path: Path,
) -> None:
    demo = tmp_path / "legacy_task_demo_7_2026-07-07_14-04-38_061043"
    action_dir = demo / "replay.zarr" / "data" / "action"
    action_dir.mkdir(parents=True)
    _write_json(
        action_dir / ".zarray",
        {"zarr_format": 2, "shape": [12, 4], "chunks": [12, 4]},
    )
    (action_dir / "0.0").write_bytes(b"trajectory")
    _write_json(
        demo / "collection_meta.json",
        {
            "data_code": demo.name,
            "engine_code": "E01",
            "sim_backend": "mujoco",
            "quality_review": {"is_success": True},
            "endpoint": "https://must-not-leak.invalid",
        },
    )
    videos = demo / "videos"
    videos.mkdir()
    (videos / "front.mp4").write_bytes(b"front-source-video")
    (videos / "wrist.mp4").write_bytes(b"wrist-source-video")
    (videos / "front_depth.mp4").write_bytes(b"depth-sensor-stream")

    envelope = build_episode_verified_v3(demo)

    assert envelope["task_name"] == "legacy_task"
    assert envelope["step_count"] == 12
    assert envelope["physical_replay"] == {"status": "not_required"}
    assert envelope["render"]["status"] == "source_recording"
    assert envelope["render"]["v6"] == {
        "status": "source_recording",
        "camera_profile": "",
    }
    assert [row["path"] for row in envelope["render"]["files"]] == [
        "videos/front.mp4",
        "videos/wrist.mp4",
    ]
    assert envelope["evidence"] == []
    assert validate_verified_episode_v3(envelope) is None
    assert "endpoint" not in canonical_json_bytes(envelope).decode("utf-8")


def test_offline_arm_render_uses_its_own_strict_artifact_contract(
    tmp_path: Path,
) -> None:
    demo = tmp_path / "t08_hammer_nail_demo_1_2026-08-02_18-10-00_000001"
    (demo / "replay.zarr").mkdir(parents=True)
    (demo / "replay.zarr" / ".zgroup").write_bytes(b'{"zarr_format":2}')
    declarations = {
        "offline_arm_render_deferred": True,
        "physical_replay_certification_required": False,
        "isaac_arm_render_deferred": False,
    }
    _write_json(
        demo / "collection_meta.json",
        {
            "task_name": "t08_hammer_nail",
            "collected_at": "2026-08-02T18:10:00+08:00",
            "step_count": 3,
            "data_fps": 30.0,
            "video_fps": 30,
            "sim_backend": "mujoco",
            "engine_code": "E01",
            "task_outcome": "success",
            **declarations,
        },
    )
    _write_json(
        demo / "replay_manifest.json",
        {
            "task": {"name": "t08_hammer_nail"},
            "recording": {"step_count": 3, **declarations},
        },
    )
    rows = []
    for index, relative_path in enumerate(OFFLINE_ARM_VIDEO_PATHS, start=1):
        payload = bytes([index]) * (12 + index)
        path = demo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        rows.append(
            {
                "path": relative_path,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    _write_json(
        demo / "offline_arm_render_status.json",
        {
            "status": "complete",
            "video_fps": 30,
            "completed_at": "2026-08-02T18:11:00+08:00",
            "files": rows,
        },
    )

    envelope = build_episode_verified_v3(demo)

    assert envelope["render"]["artifact_proof_format"] == (
        OFFLINE_ARM_ARTIFACT_PROOF_FORMAT
    )
    assert envelope["render"]["artifact_proof_version"] == (
        OFFLINE_ARM_ARTIFACT_PROOF_VERSION
    )
    assert [row["path"] for row in envelope["render"]["files"]] == sorted(
        OFFLINE_ARM_VIDEO_PATHS
    )
    assert validate_verified_episode_v3(envelope) is None


def test_declared_deferred_contract_remains_fail_closed(tmp_path: Path) -> None:
    demo = _verified_demo(tmp_path)
    (demo / "deferred_arm_replay_status.json").unlink()
    with pytest.raises(VerifiedPublicationV3Error, match="required evidence"):
        build_episode_verified_v3(demo)
