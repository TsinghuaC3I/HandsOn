"""Integrity and visual-quality proof for deferred canonical arm videos.

Most existing tasks retain the v1 size/SHA contract.  B320-BC021 is the first
task whose formal post-render path additionally requires a fail-closed, fully
decoded three-camera quality proof.  Keeping that profile task-scoped avoids
silently claiming that older episodes have passed checks which did not exist
when they were rendered.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ARTIFACT_PROOF_FORMAT = "handson.deferred_arm_replay_artifacts"
ARTIFACT_PROOF_VERSION = 1
CANONICAL_ARM_VIDEO_PATHS = (
    "videos/main.mp4",
    "videos/wrist_left.mp4",
    "videos/wrist_right.mp4",
)

B320_BC021_TASK_NAME = "t01_align_blocks_in_midair"
BC021_QUALITY_FORMAT = "handson.deferred_arm_replay_video_quality"
BC021_QUALITY_VERSION = 1
BC021_QUALITY_PROFILE = "b320_bc021_formal_three_camera_v1"
BC021_CAMERA_PROFILE_VERSION = "camera_pose_v6"
BC021_QUALITY_THRESHOLDS: dict[str, float | int] = {
    "minimum_frame_count": 2,
    "fps_absolute_tolerance": 0.01,
    "minimum_coarse_rgb_entropy_bits": 1.4,
    "maximum_dominant_coarse_rgb_fraction": 0.90,
    "minimum_luma_std": 12.0,
    "minimum_luma_p95_minus_p05": 32.0,
    "minimum_horizontal_edge_fraction": 0.002,
    "minimum_vertical_edge_fraction": 0.002,
    "maximum_black_fraction": 0.95,
    "minimum_motion_thumbnail_mad": 0.5,
    "minimum_cross_camera_thumbnail_mad": 1.0,
}

_BC021_CAMERA_KEYS = ("main", "wrist_left", "wrist_right")
_BC021_FALLBACK_COLORS = {
    "main": ((58, 66, 70), (154, 154, 145)),
    "wrist_left": ((64, 74, 92), (154, 154, 145)),
    "wrist_right": ((92, 72, 62), (154, 154, 145)),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"BC021 formal video quality requires {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"BC021 {label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"BC021 {label} must be a JSON object: {path}")
    return payload


def _metadata_task_names(root: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    collection_path = root / "collection_meta.json"
    if collection_path.is_file() and not collection_path.is_symlink():
        try:
            collection = json.loads(collection_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            collection = None
        if isinstance(collection, Mapping):
            value = collection.get("task_name")
            if isinstance(value, str) and value.strip():
                names["collection_meta.task_name"] = value.strip()

    manifest_path = root / "replay_manifest.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, Mapping):
            task = manifest.get("task")
            if isinstance(task, Mapping):
                value = task.get("name")
                if isinstance(value, str) and value.strip():
                    names["replay_manifest.task.name"] = value.strip()
    return names


def requires_deferred_arm_replay_visual_quality(
    demo_dir: str | Path,
    *,
    task_name: str | None = None,
) -> bool:
    """Return whether this episode is in the task-scoped formal quality gate."""

    names = set(_metadata_task_names(Path(demo_dir).expanduser().resolve()).values())
    if isinstance(task_name, str) and task_name.strip():
        names.add(task_name.strip())
    return B320_BC021_TASK_NAME in names


def _file_identity(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"BC021 quality input is missing: {relative_path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"BC021 quality input is empty: {relative_path}")
    return {
        "path": relative_path,
        "bytes": int(size),
        "sha256": _sha256_file(path),
    }


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"BC021 {label} must be a positive integer")
    return int(value)


def _finite_number(value: Any, *, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"BC021 {label} must be a finite number")
    return float(value)


def _finite_vector(value: Any, *, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"BC021 {label} must have length {length}")
    return [
        _finite_number(component, label=f"{label}[{index}]")
        for index, component in enumerate(value)
    ]


def _validate_camera_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("version") != BC021_CAMERA_PROFILE_VERSION:
        raise ValueError(
            "BC021 formal render requires recording camera profile "
            f"{BC021_CAMERA_PROFILE_VERSION!r}"
        )
    head = profile.get("head")
    wrist = profile.get("wrist")
    if not isinstance(head, Mapping) or not isinstance(wrist, Mapping):
        raise ValueError("BC021 camera_pose_v6 head/wrist mappings are missing")
    if head.get("projection") != "pinhole":
        raise ValueError("BC021 camera_pose_v6 head projection must be pinhole")
    _finite_vector(head.get("eye_xyz_m"), length=3, label="head.eye_xyz_m")
    quaternion = _finite_vector(
        head.get("quat_wxyz_opengl"),
        length=4,
        label="head.quat_wxyz_opengl",
    )
    quaternion_norm = math.sqrt(sum(component * component for component in quaternion))
    if abs(quaternion_norm - 1.0) > 1e-3:
        raise ValueError("BC021 head camera quaternion is not normalized")
    head_fovy = _finite_number(
        head.get("vertical_fov_deg"), label="head.vertical_fov_deg"
    )
    if not 1.0 < head_fovy < 179.0:
        raise ValueError("BC021 head vertical FOV is invalid")

    if wrist.get("output_projection") != "pinhole":
        raise ValueError("BC021 wrist output projection must be pinhole")
    if wrist.get("rendered_frames_are_already_rectified") is not True:
        raise ValueError("BC021 wrist frames must be declared already rectified")
    output_resolution = wrist.get("output_resolution")
    if (
        not isinstance(output_resolution, list)
        or len(output_resolution) != 2
        or any(type(value) is not int or value <= 0 for value in output_resolution)
    ):
        raise ValueError("BC021 wrist output resolution is invalid")
    wrist_hfov = _finite_number(
        wrist.get("horizontal_fov_deg"), label="wrist.horizontal_fov_deg"
    )
    if not 1.0 < wrist_hfov < 179.0:
        raise ValueError("BC021 wrist horizontal FOV is invalid")
    if not isinstance(wrist.get("K_new"), list):
        raise ValueError("BC021 wrist rectified intrinsics K_new are missing")
    mount_poses = wrist.get("mount_pose_xyz_quat_wxyz_by_side")
    if not isinstance(mount_poses, Mapping) or set(mount_poses) != {"left", "right"}:
        raise ValueError("BC021 wrist mount-pose mapping is incomplete")
    for side in ("left", "right"):
        pose = _finite_vector(
            mount_poses.get(side), length=7, label=f"wrist.mount_pose.{side}"
        )
        quaternion_norm = math.sqrt(sum(component * component for component in pose[3:]))
        if abs(quaternion_norm - 1.0) > 1e-3:
            raise ValueError(f"BC021 {side} wrist camera quaternion is not normalized")


def _episode_and_sidecar_contract(
    root: Path,
    *,
    declared_task_name: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    collection = _read_json_object(
        root / "collection_meta.json", label="collection metadata"
    )
    manifest = _read_json_object(
        root / "replay_manifest.json", label="replay manifest"
    )
    task = manifest.get("task")
    recording = manifest.get("recording")
    if not isinstance(task, Mapping) or not isinstance(recording, Mapping):
        raise ValueError(
            "BC021 replay manifest must contain task and recording objects"
        )
    collection_camera_profile = collection.get("recording_camera_profile")
    recording_camera_profile = recording.get("recording_camera_profile")
    if not isinstance(collection_camera_profile, Mapping) or not isinstance(
        recording_camera_profile, Mapping
    ):
        raise ValueError(
            "BC021 collection/replay recording_camera_profile is missing"
        )
    if collection_camera_profile != recording_camera_profile:
        raise ValueError(
            "BC021 collection/replay recording_camera_profile values differ"
        )
    _validate_camera_profile(recording_camera_profile)
    for owner_label, owner in (
        ("collection_meta", collection),
        ("replay_manifest.recording", recording),
    ):
        if owner.get("canonical_camera_postrender_required") is not True:
            raise ValueError(
                f"BC021 {owner_label} must require canonical camera post-render"
            )
        if owner.get("online_camera_capture_scope") != "all_canonical_postrender":
            raise ValueError(
                f"BC021 {owner_label} must declare all_canonical_postrender"
            )
    task_names = {
        str(collection.get("task_name") or "").strip(),
        str(task.get("name") or "").strip(),
        str(declared_task_name or "").strip(),
    }
    task_names.discard("")
    if task_names != {B320_BC021_TASK_NAME}:
        raise ValueError(
            "BC021 formal render task identity is missing or inconsistent: "
            f"{sorted(task_names)!r}"
        )

    collection_steps = _positive_int(
        collection.get("step_count"), label="collection_meta.step_count"
    )
    manifest_steps = _positive_int(
        recording.get("step_count"), label="recording.step_count"
    )
    collection_fps = _positive_int(
        collection.get("video_fps"), label="collection_meta.video_fps"
    )
    manifest_fps = _positive_int(
        recording.get("video_fps"), label="recording.video_fps"
    )
    if collection_steps != manifest_steps:
        raise ValueError(
            "BC021 collection/replay step counts differ: "
            f"{collection_steps} != {manifest_steps}"
        )
    if collection_steps < int(BC021_QUALITY_THRESHOLDS["minimum_frame_count"]):
        raise ValueError(
            f"BC021 formal render needs at least two frames, got {collection_steps}"
        )
    if collection_fps != manifest_fps:
        raise ValueError(
            "BC021 collection/replay video fps differ: "
            f"{collection_fps} != {manifest_fps}"
        )

    action_metadata_relative = ""
    action_metadata: dict[str, Any] | None = None
    for relative_path in (
        "replay.zarr/data/action/.zarray",
        "replay.zarr/data/action/zarr.json",
    ):
        candidate = root / relative_path
        if candidate.is_file() and not candidate.is_symlink():
            action_metadata_relative = relative_path
            action_metadata = _read_json_object(
                candidate, label="replay action-array metadata"
            )
            break
    if action_metadata is None:
        raise ValueError(
            "BC021 replay.zarr action-array metadata is missing"
        )
    action_shape = action_metadata.get("shape")
    if (
        not isinstance(action_shape, list)
        or not action_shape
        or type(action_shape[0]) is not int
        or action_shape[0] <= 0
    ):
        raise ValueError("BC021 replay action-array shape is invalid")
    if int(action_shape[0]) != collection_steps:
        raise ValueError(
            "BC021 replay action count differs from episode step count: "
            f"{action_shape[0]} != {collection_steps}"
        )

    videos_dir = root / "videos"
    mp4_entries = list(videos_dir.glob("*.mp4"))
    actual_mp4 = {path.relative_to(root).as_posix() for path in mp4_entries}
    if actual_mp4 != set(CANONICAL_ARM_VIDEO_PATHS) or any(
        path.is_symlink() or not path.is_file() for path in mp4_entries
    ):
        raise ValueError(
            "BC021 formal render must contain exactly the three canonical MP4s; "
            f"got {sorted(actual_mp4)!r}"
        )

    aggregate = _read_json_object(
        videos_dir / "camera_info.json", label="aggregate camera sidecar"
    )
    if aggregate.get("schema") != "handson_video_camera_info.v1":
        raise ValueError("BC021 aggregate camera sidecar schema is invalid")
    aggregate_cameras = aggregate.get("camera_info")
    if not isinstance(aggregate_cameras, Mapping) or set(aggregate_cameras) != set(
        _BC021_CAMERA_KEYS
    ):
        raise ValueError(
            "BC021 aggregate camera sidecar must cover exactly main/wrist_left/"
            "wrist_right"
        )

    cameras: dict[str, dict[str, Any]] = {}
    sidecar_paths: list[str] = []
    for camera_key in _BC021_CAMERA_KEYS:
        relative_path = f"videos/{camera_key}_camera_info.json"
        sidecar = _read_json_object(
            root / relative_path, label=f"{camera_key} camera sidecar"
        )
        frame = sidecar.get("frame")
        if not isinstance(frame, Mapping):
            raise ValueError(f"BC021 {camera_key} sidecar frame is missing")
        width = _positive_int(frame.get("width"), label=f"{camera_key}.frame.width")
        height = _positive_int(
            frame.get("height"), label=f"{camera_key}.frame.height"
        )
        if sidecar.get("camera_key") != camera_key:
            raise ValueError(f"BC021 {camera_key} sidecar camera_key is invalid")
        if sidecar.get("schema") != "handson_video_camera_info.v1":
            raise ValueError(f"BC021 {camera_key} sidecar schema is invalid")
        if (
            sidecar.get("recording_camera_profile_version")
            != BC021_CAMERA_PROFILE_VERSION
        ):
            raise ValueError(
                f"BC021 {camera_key} sidecar is not bound to camera_pose_v6"
            )
        if sidecar.get("video_file") != f"{camera_key}.mp4":
            raise ValueError(f"BC021 {camera_key} sidecar video_file is invalid")
        aggregate_row = aggregate_cameras.get(camera_key)
        if not isinstance(aggregate_row, Mapping):
            raise ValueError(f"BC021 aggregate sidecar is missing {camera_key}")
        if aggregate_row != sidecar:
            raise ValueError(
                f"BC021 aggregate/per-camera sidecars disagree for {camera_key}"
            )
        if camera_key == "main":
            pinhole = sidecar.get("sim_pinhole_intrinsics")
            if (
                not isinstance(pinhole, Mapping)
                or pinhole.get("image_width") != width
                or pinhole.get("image_height") != height
                or _finite_number(
                    pinhole.get("fovy_deg"),
                    label="main.sim_pinhole_intrinsics.fovy_deg",
                )
                != _finite_number(
                    recording_camera_profile["head"].get("vertical_fov_deg"),
                    label="head.vertical_fov_deg",
                )
            ):
                raise ValueError(
                    "BC021 main sidecar does not match the camera_pose_v6 head profile"
                )
        else:
            saved_projection = sidecar.get("saved_projection")
            wrist_profile = recording_camera_profile["wrist"]
            if (
                not isinstance(saved_projection, Mapping)
                or saved_projection.get("model")
                != wrist_profile.get("output_projection")
                or saved_projection.get("image_width") != width
                or saved_projection.get("image_height") != height
                or saved_projection.get("horizontal_fov_deg")
                != wrist_profile.get("horizontal_fov_deg")
                or saved_projection.get("K") != wrist_profile.get("K_new")
                or saved_projection.get("D") != [0.0, 0.0, 0.0, 0.0]
                or saved_projection.get("already_rectified") is not True
                or saved_projection.get("must_not_rectify_again") is not True
            ):
                raise ValueError(
                    f"BC021 {camera_key} sidecar does not match the "
                    "camera_pose_v6 wrist projection"
                )
            if wrist_profile.get("output_resolution") != [width, height]:
                raise ValueError(
                    f"BC021 {camera_key} shape differs from camera_pose_v6"
                )
        cameras[camera_key] = {
            "path": f"videos/{camera_key}.mp4",
            "width": width,
            "height": height,
            "sidecar_path": relative_path,
        }
        sidecar_paths.append(relative_path)

    input_paths = [
        "collection_meta.json",
        "replay_manifest.json",
        action_metadata_relative,
        "videos/camera_info.json",
        *sidecar_paths,
    ]
    inputs = [_file_identity(root, relative_path) for relative_path in input_paths]
    expected = {
        "task_name": B320_BC021_TASK_NAME,
        "frame_count": collection_steps,
        "fps": collection_fps,
        "action_array_shape": list(action_shape),
        "action_array_metadata_path": action_metadata_relative,
        "recording_camera_profile_version": BC021_CAMERA_PROFILE_VERSION,
        "recording_camera_profile_sha256": _canonical_json_sha256(
            recording_camera_profile
        ),
        "canonical_camera_postrender_required": True,
        "online_camera_capture_scope": "all_canonical_postrender",
        "cameras": cameras,
    }
    return expected, inputs


def _thumbnail_rgb(frame: Any) -> Any:
    import numpy as np

    height, width = frame.shape[:2]
    rows = np.linspace(0, height - 1, min(48, height), dtype=np.int64)
    columns = np.linspace(0, width - 1, min(64, width), dtype=np.int64)
    return np.ascontiguousarray(frame[rows][:, columns, :3], dtype=np.uint8)


def _frame_metrics(thumbnail: Any) -> dict[str, float]:
    import numpy as np

    rgb = np.asarray(thumbnail, dtype=np.uint8)
    quantized = rgb // 16
    codes = (
        quantized[..., 0].astype(np.int32) * 256
        + quantized[..., 1].astype(np.int32) * 16
        + quantized[..., 2].astype(np.int32)
    )
    counts = np.bincount(codes.reshape(-1), minlength=4096)
    counts = counts[counts > 0].astype(np.float64)
    probabilities = counts / float(counts.sum())
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    luma = (
        0.2126 * rgb[..., 0].astype(np.float64)
        + 0.7152 * rgb[..., 1].astype(np.float64)
        + 0.0722 * rgb[..., 2].astype(np.float64)
    )
    horizontal = np.abs(np.diff(luma, axis=1))
    vertical = np.abs(np.diff(luma, axis=0))
    return {
        "coarse_rgb_entropy_bits": entropy,
        "dominant_coarse_rgb_fraction": float(probabilities.max()),
        "luma_std": float(np.std(luma)),
        "luma_p95_minus_p05": float(
            np.percentile(luma, 95) - np.percentile(luma, 5)
        ),
        "horizontal_edge_fraction": float(np.mean(horizontal > 4.0)),
        "vertical_edge_fraction": float(np.mean(vertical > 4.0)),
        "black_fraction": float(np.mean(np.all(rgb < 5, axis=-1))),
    }


def _known_fallback_frame(frame: Any, camera_key: str) -> bool:
    """Recognize the Isaac two-color fallback after ordinary H.264 loss."""

    import numpy as np

    rgb = np.asarray(frame, dtype=np.uint8)
    height, width = rgb.shape[:2]
    if height < 12 or width < 12:
        return False
    margin = max(2, min(height, width) // 64)
    y0, y1 = height // 3 + margin, 2 * height // 3 - margin
    x0, x1 = width // 3 + margin, 2 * width // 3 - margin
    stride_y = max(1, height // 96)
    stride_x = max(1, width // 128)
    center = rgb[y0:y1:stride_y, x0:x1:stride_x, :3].reshape(-1, 3)
    outside_parts = (
        rgb[: max(1, height // 3 - margin) : stride_y, ::stride_x, :3],
        rgb[min(height, 2 * height // 3 + margin) :: stride_y, ::stride_x, :3],
        rgb[y0:y1:stride_y, : max(1, width // 3 - margin) : stride_x, :3],
        rgb[y0:y1:stride_y, min(width, 2 * width // 3 + margin) :: stride_x, :3],
    )
    outside = np.concatenate(
        [part.reshape(-1, 3) for part in outside_parts if part.size], axis=0
    )
    if not center.size or not outside.size:
        return False
    center = center.astype(np.float32)
    outside = outside.astype(np.float32)
    base_color, center_color = _BC021_FALLBACK_COLORS[camera_key]
    center_median = np.median(center, axis=0)
    outside_median = np.median(outside, axis=0)
    center_residual = np.percentile(
        np.abs(center - center_median).reshape(-1), 95
    )
    outside_residual = np.percentile(
        np.abs(outside - outside_median).reshape(-1), 95
    )
    return bool(
        np.max(np.abs(center_median - np.asarray(center_color))) <= 12.0
        and np.max(np.abs(outside_median - np.asarray(base_color))) <= 12.0
        and center_residual <= 8.0
        and outside_residual <= 8.0
    )


def _decode_video_quality(
    path: Path,
    *,
    relative_path: str,
    camera_key: str,
    expected_frames: int,
    expected_fps: int,
    expected_width: int,
    expected_height: int,
) -> tuple[dict[str, Any], dict[int, Any]]:
    try:
        import av
    except ImportError as exc:
        raise ValueError(
            "BC021 formal camera validation requires PyAV in the renderer runtime"
        ) from exc
    import numpy as np

    selected_indices = set(
        int(index)
        for index in np.linspace(
            0, expected_frames - 1, min(9, expected_frames), dtype=np.int64
        )
    )
    selected_thumbnails: dict[int, Any] = {}
    decoded_digest = hashlib.sha256()
    thumbnail_digest = hashlib.sha256()
    thumbnail_hashes: set[str] = set()
    first_thumbnail: Any | None = None
    max_delta_from_first = 0.0
    metrics_per_frame: list[dict[str, float]] = []
    fallback_frames: list[int] = []
    decoded_frames = 0
    codec_name = ""
    decoded_fps = 0.0

    try:
        with av.open(str(path), mode="r") as container:
            streams = [stream for stream in container.streams if stream.type == "video"]
            if len(streams) != 1:
                raise ValueError(
                    f"{relative_path} must contain exactly one video stream, got "
                    f"{len(streams)}"
                )
            stream = streams[0]
            codec_name = str(stream.codec_context.name or "").lower()
            if codec_name != "h264":
                raise ValueError(
                    f"{relative_path} codec must be H.264, got {codec_name!r}"
                )
            rate = stream.average_rate or stream.base_rate or stream.guessed_rate
            decoded_fps = float(rate) if rate is not None else 0.0
            tolerance = float(BC021_QUALITY_THRESHOLDS["fps_absolute_tolerance"])
            if (
                not math.isfinite(decoded_fps)
                or abs(decoded_fps - expected_fps) > tolerance
            ):
                raise ValueError(
                    f"{relative_path} fps={decoded_fps!r} does not match "
                    f"episode fps={expected_fps}"
                )

            for frame_index, decoded in enumerate(container.decode(video=stream.index)):
                rgb = np.asarray(decoded.to_ndarray(format="rgb24"), dtype=np.uint8)
                if rgb.shape != (expected_height, expected_width, 3):
                    raise ValueError(
                        f"{relative_path} frame {frame_index} has shape={rgb.shape}; "
                        "expected sidecar shape="
                        f"{(expected_height, expected_width, 3)}"
                    )
                decoded_digest.update(
                    struct.pack(">QII", frame_index, expected_height, expected_width)
                )
                decoded_digest.update(rgb.tobytes(order="C"))
                thumbnail = _thumbnail_rgb(rgb)
                thumbnail_bytes = thumbnail.tobytes(order="C")
                thumbnail_hash = hashlib.sha256(thumbnail_bytes).hexdigest()
                thumbnail_hashes.add(thumbnail_hash)
                thumbnail_digest.update(struct.pack(">Q", frame_index))
                thumbnail_digest.update(thumbnail_bytes)
                if first_thumbnail is None:
                    first_thumbnail = thumbnail.astype(np.float32)
                else:
                    max_delta_from_first = max(
                        max_delta_from_first,
                        float(
                            np.mean(
                                np.abs(
                                    thumbnail.astype(np.float32) - first_thumbnail
                                )
                            )
                        ),
                    )
                frame_metrics = _frame_metrics(thumbnail)
                metrics_per_frame.append(frame_metrics)
                if _known_fallback_frame(rgb, camera_key):
                    fallback_frames.append(frame_index)
                if frame_index in selected_indices:
                    selected_thumbnails[frame_index] = thumbnail
                decoded_frames += 1
    except av.error.FFmpegError as exc:
        raise ValueError(
            f"{relative_path} could not be completely decoded: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    except ValueError:
        raise
    except Exception as exc:  # codecs expose several backend-specific errors
        raise ValueError(
            f"{relative_path} could not be completely decoded: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if decoded_frames != expected_frames:
        raise ValueError(
            f"{relative_path} decoded {decoded_frames} frames; "
            f"episode requires {expected_frames}"
        )
    if set(selected_thumbnails) != selected_indices:
        raise ValueError(f"{relative_path} did not decode every selected evidence frame")
    if fallback_frames:
        raise ValueError(
            f"{relative_path} contains the known two-color Isaac fallback at "
            f"frames {fallback_frames[:16]}"
        )

    summaries = {
        "minimum_coarse_rgb_entropy_bits": min(
            row["coarse_rgb_entropy_bits"] for row in metrics_per_frame
        ),
        "maximum_dominant_coarse_rgb_fraction": max(
            row["dominant_coarse_rgb_fraction"] for row in metrics_per_frame
        ),
        "minimum_luma_std": min(row["luma_std"] for row in metrics_per_frame),
        "minimum_luma_p95_minus_p05": min(
            row["luma_p95_minus_p05"] for row in metrics_per_frame
        ),
        "minimum_horizontal_edge_fraction": min(
            row["horizontal_edge_fraction"] for row in metrics_per_frame
        ),
        "minimum_vertical_edge_fraction": min(
            row["vertical_edge_fraction"] for row in metrics_per_frame
        ),
        "maximum_black_fraction": max(
            row["black_fraction"] for row in metrics_per_frame
        ),
    }
    checks = {
        "all_frames_readable": bool(
            summaries["minimum_coarse_rgb_entropy_bits"]
            >= float(
                BC021_QUALITY_THRESHOLDS["minimum_coarse_rgb_entropy_bits"]
            )
            and summaries["maximum_dominant_coarse_rgb_fraction"]
            < float(
                BC021_QUALITY_THRESHOLDS[
                    "maximum_dominant_coarse_rgb_fraction"
                ]
            )
            and summaries["minimum_luma_std"]
            >= float(BC021_QUALITY_THRESHOLDS["minimum_luma_std"])
            and summaries["minimum_luma_p95_minus_p05"]
            >= float(
                BC021_QUALITY_THRESHOLDS["minimum_luma_p95_minus_p05"]
            )
            and summaries["minimum_horizontal_edge_fraction"]
            >= float(
                BC021_QUALITY_THRESHOLDS["minimum_horizontal_edge_fraction"]
            )
            and summaries["minimum_vertical_edge_fraction"]
            >= float(
                BC021_QUALITY_THRESHOLDS["minimum_vertical_edge_fraction"]
            )
            and summaries["maximum_black_fraction"]
            < float(BC021_QUALITY_THRESHOLDS["maximum_black_fraction"])
        ),
        "not_static_repeat": bool(
            max_delta_from_first
            >= float(
                BC021_QUALITY_THRESHOLDS["minimum_motion_thumbnail_mad"]
            )
        ),
        "known_fallback_absent": not fallback_frames,
    }
    if not checks["all_frames_readable"]:
        raise ValueError(
            f"{relative_path} failed the BC021 low-information-frame gate: "
            f"{summaries}"
        )
    if not checks["not_static_repeat"]:
        raise ValueError(
            f"{relative_path} is a static repeated stream: "
            f"max thumbnail MAD={max_delta_from_first:.6f}"
        )

    video_size = path.stat().st_size
    report = {
        "path": relative_path,
        "bytes": int(video_size),
        "file_sha256": _sha256_file(path),
        "codec": codec_name,
        "fps": round(decoded_fps, 9),
        "width": expected_width,
        "height": expected_height,
        "frame_count": decoded_frames,
        "decoded_rgb_sha256": decoded_digest.hexdigest(),
        "decoded_thumbnail_sha256": thumbnail_digest.hexdigest(),
        "distinct_thumbnail_sha256_count": len(thumbnail_hashes),
        "maximum_thumbnail_mad_from_first": round(max_delta_from_first, 9),
        "known_fallback_frame_count": len(fallback_frames),
        "selected_frame_indices": sorted(selected_indices),
        "metrics": {key: round(value, 9) for key, value in summaries.items()},
        "checks": checks,
    }
    return report, selected_thumbnails


def _capture_bc021_quality(
    root: Path,
    *,
    declared_task_name: str | None,
    file_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected, inputs = _episode_and_sidecar_contract(
        root, declared_task_name=declared_task_name
    )
    video_reports: list[dict[str, Any]] = []
    thumbnails: dict[str, dict[int, Any]] = {}
    for camera_key in _BC021_CAMERA_KEYS:
        camera = expected["cameras"][camera_key]
        report, selected = _decode_video_quality(
            root / camera["path"],
            relative_path=camera["path"],
            camera_key=camera_key,
            expected_frames=int(expected["frame_count"]),
            expected_fps=int(expected["fps"]),
            expected_width=int(camera["width"]),
            expected_height=int(camera["height"]),
        )
        video_reports.append(report)
        thumbnails[camera_key] = selected

    file_by_path = {row["path"]: row for row in file_rows}
    for report in video_reports:
        identity = file_by_path.get(report["path"])
        if identity is None or (
            report["bytes"] != identity["bytes"]
            or report["file_sha256"] != identity["sha256"]
        ):
            raise ValueError(
                f"BC021 decoded evidence is not bound to {report['path']}"
            )

    import numpy as np

    cross_camera: list[dict[str, Any]] = []
    for left_index, left_key in enumerate(_BC021_CAMERA_KEYS):
        for right_key in _BC021_CAMERA_KEYS[left_index + 1 :]:
            frame_indices = sorted(
                set(thumbnails[left_key]) & set(thumbnails[right_key])
            )
            deltas = [
                float(
                    np.mean(
                        np.abs(
                            thumbnails[left_key][frame_index].astype(np.float32)
                            - thumbnails[right_key][frame_index].astype(np.float32)
                        )
                    )
                )
                for frame_index in frame_indices
            ]
            mean_delta = float(np.mean(deltas)) if deltas else 0.0
            passed = bool(
                mean_delta
                >= float(
                    BC021_QUALITY_THRESHOLDS[
                        "minimum_cross_camera_thumbnail_mad"
                    ]
                )
            )
            row = {
                "left": left_key,
                "right": right_key,
                "sample_count": len(deltas),
                "mean_thumbnail_mad": round(mean_delta, 9),
                "minimum_thumbnail_mad": round(min(deltas) if deltas else 0.0, 9),
                "passed": passed,
            }
            cross_camera.append(row)
            if not passed:
                raise ValueError(
                    "BC021 canonical camera streams are duplicated or nearly "
                    f"identical: {left_key}/{right_key}: {row}"
                )

    quality: dict[str, Any] = {
        "format": BC021_QUALITY_FORMAT,
        "version": BC021_QUALITY_VERSION,
        "profile": BC021_QUALITY_PROFILE,
        "task_name": B320_BC021_TASK_NAME,
        "passed": True,
        "thresholds": dict(BC021_QUALITY_THRESHOLDS),
        "expected": expected,
        "inputs": inputs,
        "videos": video_reports,
        "cross_camera": cross_camera,
    }
    quality["evidence_sha256"] = _canonical_json_sha256(quality)
    return quality


def capture_deferred_arm_replay_artifacts(
    demo_dir: str | Path,
    *,
    task_name: str | None = None,
) -> dict[str, Any]:
    """Capture exact evidence for the three canonical MP4 files.

    B320-BC021 additionally performs the complete task-scoped visual gate.
    A failure raises before a proof can be published as ``complete``.
    """

    root = Path(demo_dir).expanduser().resolve()
    files: list[dict[str, Any]] = []
    for relative_path in CANONICAL_ARM_VIDEO_PATHS:
        path = root / relative_path
        if path.is_symlink() or not path.is_file():
            raise OSError(f"canonical arm replay video is missing: {relative_path}")
        size = path.stat().st_size
        if size <= 0:
            raise OSError(f"canonical arm replay video is empty: {relative_path}")
        files.append(
            {
                "path": relative_path,
                "bytes": int(size),
                "sha256": _sha256_file(path),
            }
        )
    proof: dict[str, Any] = {
        "format": ARTIFACT_PROOF_FORMAT,
        "version": ARTIFACT_PROOF_VERSION,
        "files": files,
    }
    if requires_deferred_arm_replay_visual_quality(root, task_name=task_name):
        proof["quality"] = _capture_bc021_quality(
            root,
            declared_task_name=task_name,
            file_rows=files,
        )
    return proof


def _proof_file_rows(proof: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = proof.get("files")
    rows = rows if isinstance(rows, list) else []
    return {
        str(row.get("path") or ""): row
        for row in rows
        if isinstance(row, Mapping)
    }


def _bc021_quality_proof_errors(
    root: Path,
    proof: Mapping[str, Any],
) -> list[str]:
    quality = proof.get("quality")
    if not isinstance(quality, Mapping):
        return ["BC021 formal three-camera quality proof is missing"]
    errors: list[str] = []
    expected_keys = {
        "format",
        "version",
        "profile",
        "task_name",
        "passed",
        "thresholds",
        "expected",
        "inputs",
        "videos",
        "cross_camera",
        "evidence_sha256",
    }
    if set(quality) != expected_keys:
        errors.append("BC021 quality proof fields are invalid")
    if quality.get("format") != BC021_QUALITY_FORMAT:
        errors.append("BC021 quality proof format is invalid")
    if quality.get("version") != BC021_QUALITY_VERSION:
        errors.append("BC021 quality proof version is invalid")
    if quality.get("profile") != BC021_QUALITY_PROFILE:
        errors.append("BC021 quality proof profile is invalid")
    if quality.get("task_name") != B320_BC021_TASK_NAME:
        errors.append("BC021 quality proof task identity is invalid")
    if quality.get("passed") is not True:
        errors.append("BC021 quality proof did not pass")
    if quality.get("thresholds") != BC021_QUALITY_THRESHOLDS:
        errors.append("BC021 quality proof thresholds are invalid")
    evidence_sha = quality.get("evidence_sha256")
    digest_payload = dict(quality)
    digest_payload.pop("evidence_sha256", None)
    if (
        not isinstance(evidence_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", evidence_sha) is None
        or _canonical_json_sha256(digest_payload) != evidence_sha
    ):
        errors.append("BC021 quality evidence hash is invalid")

    try:
        current_expected, current_inputs = _episode_and_sidecar_contract(
            root, declared_task_name=B320_BC021_TASK_NAME
        )
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        current_expected, current_inputs = None, None
    if current_expected is not None and quality.get("expected") != current_expected:
        errors.append("BC021 quality expected episode/sidecar contract changed")
    if current_inputs is not None and quality.get("inputs") != current_inputs:
        errors.append("BC021 quality input identities changed")

    def finite_metric(owner: Mapping[str, Any], key: str) -> float | None:
        value = owner.get(key)
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            return None
        return float(value)

    proof_files = _proof_file_rows(proof)
    videos = quality.get("videos")
    videos = videos if isinstance(videos, list) else []
    if len(videos) != 3:
        errors.append("BC021 quality proof must contain exactly three video reports")
    video_by_path = {
        str(row.get("path") or ""): row
        for row in videos
        if isinstance(row, Mapping)
    }
    if set(video_by_path) != set(CANONICAL_ARM_VIDEO_PATHS):
        errors.append("BC021 quality video reports do not cover canonical cameras")
    for relative_path in CANONICAL_ARM_VIDEO_PATHS:
        row = video_by_path.get(relative_path)
        identity = proof_files.get(relative_path)
        if not isinstance(row, Mapping) or not isinstance(identity, Mapping):
            continue
        if (
            row.get("bytes") != identity.get("bytes")
            or row.get("file_sha256") != identity.get("sha256")
        ):
            errors.append(
                f"BC021 decoded quality is not bound to {relative_path}"
            )
        checks = row.get("checks")
        if (
            not isinstance(checks, Mapping)
            or set(checks)
            != {
                "all_frames_readable",
                "not_static_repeat",
                "known_fallback_absent",
            }
            or any(value is not True for value in checks.values())
        ):
            errors.append(f"BC021 quality checks are invalid for {relative_path}")
        if row.get("codec") != "h264":
            errors.append(f"BC021 quality codec is invalid for {relative_path}")
        if row.get("known_fallback_frame_count") != 0:
            errors.append(f"BC021 fallback frames are present in {relative_path}")
        camera_key = Path(relative_path).stem
        expected_camera = (
            current_expected.get("cameras", {}).get(camera_key)
            if isinstance(current_expected, Mapping)
            and isinstance(current_expected.get("cameras"), Mapping)
            else None
        )
        if isinstance(expected_camera, Mapping):
            if row.get("frame_count") != current_expected.get("frame_count"):
                errors.append(
                    f"BC021 decoded frame count is invalid for {relative_path}"
                )
            if row.get("width") != expected_camera.get("width") or row.get(
                "height"
            ) != expected_camera.get("height"):
                errors.append(
                    f"BC021 decoded shape is invalid for {relative_path}"
                )
            fps = finite_metric(row, "fps")
            if fps is None or abs(fps - float(current_expected["fps"])) > float(
                BC021_QUALITY_THRESHOLDS["fps_absolute_tolerance"]
            ):
                errors.append(
                    f"BC021 decoded fps is invalid for {relative_path}"
                )
        distinct_count = row.get("distinct_thumbnail_sha256_count")
        motion = finite_metric(row, "maximum_thumbnail_mad_from_first")
        if type(distinct_count) is not int or distinct_count < 2:
            errors.append(
                f"BC021 decoded thumbnails are static for {relative_path}"
            )
        if motion is None or motion < float(
            BC021_QUALITY_THRESHOLDS["minimum_motion_thumbnail_mad"]
        ):
            errors.append(
                f"BC021 decoded motion metric is invalid for {relative_path}"
            )
        selected_indices = row.get("selected_frame_indices")
        if (
            not isinstance(selected_indices, list)
            or not selected_indices
            or any(type(index) is not int or index < 0 for index in selected_indices)
            or selected_indices != sorted(set(selected_indices))
            or (
                isinstance(current_expected, Mapping)
                and (
                    selected_indices[0] != 0
                    or selected_indices[-1]
                    != int(current_expected["frame_count"]) - 1
                )
            )
        ):
            errors.append(
                f"BC021 selected frame evidence is invalid for {relative_path}"
            )
        metrics = row.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        minimum_metrics = {
            "minimum_coarse_rgb_entropy_bits": (
                "minimum_coarse_rgb_entropy_bits"
            ),
            "minimum_luma_std": "minimum_luma_std",
            "minimum_luma_p95_minus_p05": "minimum_luma_p95_minus_p05",
            "minimum_horizontal_edge_fraction": (
                "minimum_horizontal_edge_fraction"
            ),
            "minimum_vertical_edge_fraction": "minimum_vertical_edge_fraction",
        }
        for metric_name, threshold_name in minimum_metrics.items():
            metric_value = finite_metric(metrics, metric_name)
            if metric_value is None or metric_value < float(
                BC021_QUALITY_THRESHOLDS[threshold_name]
            ):
                errors.append(
                    f"BC021 {metric_name} is invalid for {relative_path}"
                )
        maximum_metrics = {
            "maximum_dominant_coarse_rgb_fraction": (
                "maximum_dominant_coarse_rgb_fraction"
            ),
            "maximum_black_fraction": "maximum_black_fraction",
        }
        for metric_name, threshold_name in maximum_metrics.items():
            metric_value = finite_metric(metrics, metric_name)
            if metric_value is None or metric_value >= float(
                BC021_QUALITY_THRESHOLDS[threshold_name]
            ):
                errors.append(
                    f"BC021 {metric_name} is invalid for {relative_path}"
                )
        for digest_key in ("decoded_rgb_sha256", "decoded_thumbnail_sha256"):
            digest = row.get(digest_key)
            if not isinstance(digest, str) or re.fullmatch(
                r"[0-9a-f]{64}", digest
            ) is None:
                errors.append(
                    f"BC021 {digest_key} is invalid for {relative_path}"
                )

    cross_camera = quality.get("cross_camera")
    cross_camera = cross_camera if isinstance(cross_camera, list) else []
    expected_pairs = {
        ("main", "wrist_left"),
        ("main", "wrist_right"),
        ("wrist_left", "wrist_right"),
    }
    actual_pairs = {
        (str(row.get("left") or ""), str(row.get("right") or ""))
        for row in cross_camera
        if isinstance(row, Mapping)
    }
    if actual_pairs != expected_pairs or len(cross_camera) != 3:
        errors.append("BC021 cross-camera quality proof is incomplete")
    elif any(
        not isinstance(row, Mapping) or row.get("passed") is not True
        for row in cross_camera
    ):
        errors.append("BC021 duplicate-camera rejection did not pass")
    else:
        minimum_cross_delta = float(
            BC021_QUALITY_THRESHOLDS["minimum_cross_camera_thumbnail_mad"]
        )
        for row in cross_camera:
            assert isinstance(row, Mapping)
            sample_count = row.get("sample_count")
            mean_delta = finite_metric(row, "mean_thumbnail_mad")
            if type(sample_count) is not int or sample_count <= 0:
                errors.append("BC021 cross-camera sample count is invalid")
            if mean_delta is None or mean_delta < minimum_cross_delta:
                errors.append("BC021 cross-camera separation metric is invalid")
    return errors


def deferred_arm_replay_artifact_errors(
    demo_dir: str | Path,
    proof: Mapping[str, Any] | None,
) -> list[str]:
    """Return fail-closed proof/current-file mismatches."""

    if not isinstance(proof, Mapping):
        return ["arm replay artifact proof is missing"]
    errors: list[str] = []
    if proof.get("format") != ARTIFACT_PROOF_FORMAT:
        errors.append("arm replay artifact proof format is invalid")
    if proof.get("version") != ARTIFACT_PROOF_VERSION:
        errors.append("arm replay artifact proof version is invalid")
    rows = proof.get("files")
    rows = rows if isinstance(rows, list) else []
    by_path = {
        str(row.get("path") or ""): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if set(by_path) != set(CANONICAL_ARM_VIDEO_PATHS) or len(rows) != len(
        CANONICAL_ARM_VIDEO_PATHS
    ):
        errors.append("arm replay artifact proof must cover exactly three videos")
    root = Path(demo_dir).expanduser().resolve()
    for relative_path in CANONICAL_ARM_VIDEO_PATHS:
        row = by_path.get(relative_path)
        if not isinstance(row, Mapping):
            errors.append(f"arm replay artifact proof is missing {relative_path}")
            continue
        size = row.get("bytes")
        expected_sha = row.get("sha256")
        if type(size) is not int or size <= 0:
            errors.append(f"arm replay artifact size is invalid for {relative_path}")
            continue
        if not isinstance(expected_sha, str) or re.fullmatch(
            r"[0-9a-f]{64}", expected_sha
        ) is None:
            errors.append(f"arm replay artifact hash is invalid for {relative_path}")
            continue
        path = root / relative_path
        try:
            if path.is_symlink() or not path.is_file():
                errors.append(f"arm replay artifact is missing: {relative_path}")
            elif path.stat().st_size != size:
                errors.append(f"arm replay artifact size changed: {relative_path}")
            elif _sha256_file(path) != expected_sha:
                errors.append(f"arm replay artifact hash changed: {relative_path}")
        except OSError as exc:
            errors.append(f"arm replay artifact cannot be read: {relative_path}: {exc}")
    if requires_deferred_arm_replay_visual_quality(root):
        errors.extend(_bc021_quality_proof_errors(root, proof))
    elif isinstance(proof.get("quality"), Mapping) and (
        proof["quality"].get("profile") == BC021_QUALITY_PROFILE
    ):
        errors.append("BC021 quality proof is attached to a non-BC021 episode")
    return errors


__all__ = [
    "ARTIFACT_PROOF_FORMAT",
    "ARTIFACT_PROOF_VERSION",
    "B320_BC021_TASK_NAME",
    "BC021_CAMERA_PROFILE_VERSION",
    "BC021_QUALITY_FORMAT",
    "BC021_QUALITY_PROFILE",
    "BC021_QUALITY_THRESHOLDS",
    "BC021_QUALITY_VERSION",
    "CANONICAL_ARM_VIDEO_PATHS",
    "capture_deferred_arm_replay_artifacts",
    "deferred_arm_replay_artifact_errors",
    "requires_deferred_arm_replay_visual_quality",
]
