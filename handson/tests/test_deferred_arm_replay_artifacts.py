from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from handson.data.deferred_arm_replay_artifacts import (
    B320_BC021_TASK_NAME,
    BC021_QUALITY_PROFILE,
    CANONICAL_ARM_VIDEO_PATHS,
    capture_deferred_arm_replay_artifacts,
    deferred_arm_replay_artifact_errors,
)


def _write_videos(demo_dir: Path) -> None:
    for index, relative_path in enumerate(CANONICAL_ARM_VIDEO_PATHS, start=1):
        path = demo_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([index]) * (index + 2))


def _encode_h264(path: Path, frames: list[np.ndarray], *, fps: int = 30) -> None:
    av = pytest.importorskip("av")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _rich_frames(
    camera_index: int,
    *,
    count: int = 6,
    height: int = 64,
    width: int = 96,
) -> list[np.ndarray]:
    yy, xx = np.indices((height, width), dtype=np.uint16)
    frames = []
    for frame_index in range(count):
        frame = np.stack(
            [
                (3 * xx + 5 * yy + 29 * camera_index + 11 * frame_index) % 256,
                (7 * xx + 2 * yy + 53 * camera_index + 17 * frame_index) % 256,
                (2 * xx + 9 * yy + 71 * camera_index + 23 * frame_index) % 256,
            ],
            axis=-1,
        ).astype(np.uint8)
        x0 = (7 * frame_index + 13 * camera_index) % (width - 16)
        frame[18:34, x0 : x0 + 16] = np.asarray(
            [240 - 30 * camera_index, 40 + 50 * camera_index, 120],
            dtype=np.uint8,
        )
        frames.append(frame)
    return frames


def _camera_pose_v6_profile(*, width: int, height: int) -> dict[str, object]:
    rectified_k = [
        [70.0, 0.0, (width - 1.0) * 0.5],
        [0.0, 70.0, (height - 1.0) * 0.5],
        [0.0, 0.0, 1.0],
    ]
    return {
        "version": "camera_pose_v6",
        "head": {
            "projection": "pinhole",
            "eye_xyz_m": [-0.675, 0.0, 1.5885],
            "quat_wxyz_opengl": [1.0, 0.0, 0.0, 0.0],
            "vertical_fov_deg": 80.0,
        },
        "wrist": {
            "output_projection": "pinhole",
            "output_resolution": [width, height],
            "horizontal_fov_deg": 100.0,
            "rendered_frames_are_already_rectified": True,
            "K_new": rectified_k,
            "mount_pose_xyz_quat_wxyz_by_side": {
                "left": [0.01, 0.02, 0.03, 1.0, 0.0, 0.0, 0.0],
                "right": [-0.01, 0.02, 0.03, 1.0, 0.0, 0.0, 0.0],
            },
        },
    }


def _write_bc021_episode(
    demo_dir: Path,
    frames: dict[str, list[np.ndarray]],
    *,
    metadata_frame_count: int | None = None,
    metadata_fps: int = 30,
    encoded_fps: int = 30,
) -> None:
    frame_count = metadata_frame_count or len(frames["main"])
    wrist_height, wrist_width = frames["wrist_left"][0].shape[:2]
    camera_profile = _camera_pose_v6_profile(
        width=wrist_width,
        height=wrist_height,
    )
    demo_dir.mkdir(parents=True, exist_ok=True)
    (demo_dir / "collection_meta.json").write_text(
        json.dumps(
            {
                "task_name": B320_BC021_TASK_NAME,
                "step_count": frame_count,
                "video_fps": metadata_fps,
                "recording_camera_profile": camera_profile,
                "canonical_camera_postrender_required": True,
                "online_camera_capture_scope": "all_canonical_postrender",
            }
        ),
        encoding="utf-8",
    )
    (demo_dir / "replay_manifest.json").write_text(
        json.dumps(
            {
                "task": {"name": B320_BC021_TASK_NAME},
                "recording": {
                    "step_count": frame_count,
                    "video_fps": metadata_fps,
                    "recording_camera_profile": camera_profile,
                    "canonical_camera_postrender_required": True,
                    "online_camera_capture_scope": "all_canonical_postrender",
                },
            }
        ),
        encoding="utf-8",
    )
    action_metadata = demo_dir / "replay.zarr" / "data" / "action" / ".zarray"
    action_metadata.parent.mkdir(parents=True, exist_ok=True)
    action_metadata.write_text(
        json.dumps({"shape": [frame_count, 54], "dtype": "<f8"}),
        encoding="utf-8",
    )

    aggregate: dict[str, object] = {
        "schema": "handson_video_camera_info.v1",
        "camera_info": {},
    }
    videos_dir = demo_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    for camera_key in ("main", "wrist_left", "wrist_right"):
        camera_frames = frames[camera_key]
        height, width = camera_frames[0].shape[:2]
        sidecar = {
            "schema": "handson_video_camera_info.v1",
            "camera_key": camera_key,
            "video_file": f"{camera_key}.mp4",
            "frame": {"width": width, "height": height},
            "recording_camera_profile_version": "camera_pose_v6",
        }
        if camera_key == "main":
            sidecar["sim_pinhole_intrinsics"] = {
                "image_width": width,
                "image_height": height,
                "fovy_deg": camera_profile["head"]["vertical_fov_deg"],
            }
        else:
            wrist_profile = camera_profile["wrist"]
            sidecar["saved_projection"] = {
                "model": wrist_profile["output_projection"],
                "image_width": width,
                "image_height": height,
                "horizontal_fov_deg": wrist_profile["horizontal_fov_deg"],
                "K": wrist_profile["K_new"],
                "D": [0.0, 0.0, 0.0, 0.0],
                "already_rectified": True,
                "must_not_rectify_again": True,
            }
        (videos_dir / f"{camera_key}_camera_info.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
        aggregate["camera_info"][camera_key] = sidecar
        _encode_h264(
            videos_dir / f"{camera_key}.mp4",
            camera_frames,
            fps=encoded_fps,
        )
    (videos_dir / "camera_info.json").write_text(
        json.dumps(aggregate), encoding="utf-8"
    )


def _valid_bc021_frames() -> dict[str, list[np.ndarray]]:
    return {
        camera_key: _rich_frames(index)
        for index, camera_key in enumerate(
            ("main", "wrist_left", "wrist_right")
        )
    }


def test_arm_replay_artifact_proof_detects_missing_or_changed_video(
    tmp_path: Path,
) -> None:
    _write_videos(tmp_path)
    proof = capture_deferred_arm_replay_artifacts(tmp_path)
    assert "quality" not in proof
    assert deferred_arm_replay_artifact_errors(tmp_path, proof) == []

    (tmp_path / "videos" / "wrist_left.mp4").write_bytes(b"changed")
    errors = deferred_arm_replay_artifact_errors(tmp_path, proof)
    assert any("wrist_left.mp4" in error for error in errors)


def test_arm_replay_artifact_proof_requires_all_three_nonempty_files(
    tmp_path: Path,
) -> None:
    _write_videos(tmp_path)
    (tmp_path / "videos" / "main.mp4").unlink()
    try:
        capture_deferred_arm_replay_artifacts(tmp_path)
    except OSError as exc:
        assert "main.mp4" in str(exc)
    else:
        raise AssertionError("missing canonical video unexpectedly passed")


def test_bc021_quality_proof_fully_decodes_and_binds_episode_sidecars(
    tmp_path: Path,
) -> None:
    _write_bc021_episode(tmp_path, _valid_bc021_frames())

    proof = capture_deferred_arm_replay_artifacts(
        tmp_path,
        task_name=B320_BC021_TASK_NAME,
    )

    assert deferred_arm_replay_artifact_errors(tmp_path, proof) == []
    quality = proof["quality"]
    assert quality["profile"] == BC021_QUALITY_PROFILE
    assert quality["passed"] is True
    assert quality["expected"]["frame_count"] == 6
    assert quality["expected"]["fps"] == 30
    assert quality["expected"]["recording_camera_profile_version"] == (
        "camera_pose_v6"
    )
    assert len(quality["expected"]["recording_camera_profile_sha256"]) == 64
    assert len(quality["videos"]) == 3
    assert len(quality["cross_camera"]) == 3
    assert all(row["checks"]["all_frames_readable"] for row in quality["videos"])
    assert all(row["checks"]["not_static_repeat"] for row in quality["videos"])
    assert all(row["decoded_rgb_sha256"] for row in quality["videos"])

    legacy = copy.deepcopy(proof)
    legacy.pop("quality")
    assert any(
        "quality proof is missing" in error
        for error in deferred_arm_replay_artifact_errors(tmp_path, legacy)
    )


def test_bc021_renderer_complete_status_binds_quality_evidence(
    tmp_path: Path,
) -> None:
    _write_bc021_episode(tmp_path, _valid_bc021_frames())
    program = (
        "from pathlib import Path; import sys; "
        "from scripts.isaac_render_arm_replay import _mark_arm_render_complete; "
        f"_mark_arm_render_complete(Path(sys.argv[1]), {B320_BC021_TASK_NAME!r})"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    status = json.loads(
        (tmp_path / "deferred_arm_replay_status.json").read_text(
            encoding="utf-8"
        )
    )
    quality = status["artifact_proof"]["quality"]
    assert status["status"] == "complete"
    assert status["artifact_validation_passed"] is True
    assert status["artifact_quality_profile"] == BC021_QUALITY_PROFILE
    assert status["artifact_quality_evidence_sha256"] == quality["evidence_sha256"]
    assert deferred_arm_replay_artifact_errors(
        tmp_path, status["artifact_proof"]
    ) == []


def test_bc021_quality_proof_rejects_tampered_evidence_or_sidecar(
    tmp_path: Path,
) -> None:
    _write_bc021_episode(tmp_path, _valid_bc021_frames())
    proof = capture_deferred_arm_replay_artifacts(tmp_path)

    tampered = copy.deepcopy(proof)
    tampered["quality"]["videos"][0]["checks"]["all_frames_readable"] = False
    errors = deferred_arm_replay_artifact_errors(tmp_path, tampered)
    assert any("evidence hash" in error for error in errors)
    assert any("quality checks" in error for error in errors)

    sidecar = tmp_path / "videos" / "main_camera_info.json"
    sidecar.write_text(sidecar.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    errors = deferred_arm_replay_artifact_errors(tmp_path, proof)
    assert any("input identities changed" in error for error in errors)


def test_bc021_quality_gate_requires_camera_pose_v6_sidecar_binding(
    tmp_path: Path,
) -> None:
    _write_bc021_episode(tmp_path, _valid_bc021_frames())
    sidecar_path = tmp_path / "videos" / "wrist_left_camera_info.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["recording_camera_profile_version"] = "legacy_camera_pose"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="not bound to camera_pose_v6"):
        capture_deferred_arm_replay_artifacts(tmp_path)


def test_bc021_quality_gate_rejects_one_frame_episode(
    tmp_path: Path,
) -> None:
    frames = {
        camera_key: _rich_frames(index, count=1)
        for index, camera_key in enumerate(
            ("main", "wrist_left", "wrist_right")
        )
    }
    _write_bc021_episode(tmp_path, frames)

    with pytest.raises(ValueError, match="at least two frames"):
        capture_deferred_arm_replay_artifacts(tmp_path)


def test_bc021_quality_gate_rejects_known_two_color_fallback(
    tmp_path: Path,
) -> None:
    frames = _valid_bc021_frames()
    fallback = np.full((64, 96, 3), (58, 66, 70), dtype=np.uint8)
    fallback[64 // 3 : 2 * 64 // 3, 96 // 3 : 2 * 96 // 3] = (
        154,
        154,
        145,
    )
    frames["main"] = [fallback.copy() for _ in range(6)]
    _write_bc021_episode(tmp_path, frames)

    with pytest.raises(ValueError, match="known two-color Isaac fallback"):
        capture_deferred_arm_replay_artifacts(tmp_path)


def test_bc021_quality_gate_rejects_low_information_and_static_streams(
    tmp_path: Path,
) -> None:
    low_information = _valid_bc021_frames()
    poor_frames = []
    for frame_index in range(6):
        frame = np.full((64, 96, 3), (20, 30, 40), dtype=np.uint8)
        x0 = 5 + frame_index * 4
        frame[20:32, x0 : x0 + 10] = (200, 50, 80)
        poor_frames.append(frame)
    low_information["main"] = poor_frames
    low_dir = tmp_path / "low"
    _write_bc021_episode(low_dir, low_information)
    with pytest.raises(ValueError, match="low-information-frame gate"):
        capture_deferred_arm_replay_artifacts(low_dir)

    static = _valid_bc021_frames()
    fixed = _rich_frames(0, count=1)[0]
    static["main"] = [fixed.copy() for _ in range(6)]
    static_dir = tmp_path / "static"
    _write_bc021_episode(static_dir, static)
    with pytest.raises(ValueError, match="static repeated stream"):
        capture_deferred_arm_replay_artifacts(static_dir)


def test_bc021_quality_gate_rejects_duplicate_cameras(
    tmp_path: Path,
) -> None:
    shared = _rich_frames(0)
    frames = {
        camera_key: [frame.copy() for frame in shared]
        for camera_key in ("main", "wrist_left", "wrist_right")
    }
    _write_bc021_episode(tmp_path, frames)

    with pytest.raises(ValueError, match="duplicated or nearly identical"):
        capture_deferred_arm_replay_artifacts(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("frame_count", "decoded 6 frames; episode requires 7"),
        ("fps", "does not match episode fps=25"),
        ("garbage", "could not be completely decoded"),
        ("extra_mp4", "exactly the three canonical MP4s"),
    ),
)
def test_bc021_quality_gate_rejects_decode_contract_mismatches(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    kwargs: dict[str, int] = {}
    if mutation == "frame_count":
        kwargs["metadata_frame_count"] = 7
    elif mutation == "fps":
        kwargs["metadata_fps"] = 25
    _write_bc021_episode(tmp_path, _valid_bc021_frames(), **kwargs)
    if mutation == "garbage":
        (tmp_path / "videos" / "main.mp4").write_bytes(b"not-an-mp4")
    elif mutation == "extra_mp4":
        (tmp_path / "videos" / "extra.mp4").write_bytes(b"extra")

    with pytest.raises(ValueError, match=re.escape(message)):
        capture_deferred_arm_replay_artifacts(tmp_path)
