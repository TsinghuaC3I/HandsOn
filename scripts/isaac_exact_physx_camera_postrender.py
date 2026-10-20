"""Render canonical cameras from recorded PhysX states without physics steps.

The existing caller retains its episode lock, completion record and replacement
invalidation callback. This visual reconstruction is separate from independent
physical replay and cannot certify task success.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np
from handson.data.camera_info import write_video_camera_info
from handson.data.canonical_videos import (
    CANONICAL_CAMERA_KEYS, observation_image, postprocess_canonical_image,
    select_canonical_camera_sources_for_trajectory,
)
from handson.data.video_writer import Mp4VideoWriter
from handson.tasks.mappings import CONFIG_MAPPING
from scripts.isaac_arm_replay_publish import publish_arm_video_directory
from scripts.replay_demos_zarr import _load_episode_actions, _load_physx_replay_trajectory


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    frame = np.asarray(image)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise ValueError(f"Expected an RGB frame, got shape={frame.shape}")
    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if frame.size and float(np.nanmax(frame)) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


_VISUAL_PHYSX_FIELDS = {
    "physx_robot_joint_position": "physx_robot_joint_valid",
    "physx_robot_root_state": "physx_robot_root_valid",
    "physx_task_rigid_body_state": "physx_task_rigid_body_valid",
    "physx_task_articulation_root_state": (
        "physx_task_articulation_root_valid"
    ),
    "physx_task_articulation_dof_position": (
        "physx_task_articulation_dof_valid"
    ),
    "physx_deformable_position": "physx_deformable_valid",
}


_VISUAL_PHYSX_RESTORE_ATOL = 1.0e-4


def _recorded_render_settings(
    demo_dir: Path,
) -> tuple[dict[str, object], bool, int, int, str, str]:
    manifest = _read_json(demo_dir / "replay_manifest.json")
    recording = manifest.get("recording")
    if not isinstance(recording, dict):
        raise RuntimeError("Replay manifest has no recording contract.")
    backend = str(recording.get("sim_backend") or "").strip().lower()
    if backend != "isaacsim":
        raise RuntimeError(
            "Exact PhysX camera post-render requires sim_backend=isaacsim; "
            f"recorded {backend!r}."
        )

    def required_nonnegative_int(name: str) -> int:
        value = recording.get(name)
        if isinstance(value, bool):
            raise RuntimeError(f"recording.{name} must be an integer.")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Exact camera post-render requires recording.{name}."
            ) from exc
        if result < 0:
            raise RuntimeError(f"recording.{name} must be non-negative.")
        return result

    collection_seed = required_nonnegative_int("collection_run_seed")
    episode_seed = required_nonnegative_int("episode_reset_seed")
    arm_type = str(recording.get("arm_type") or "").strip().lower()
    hand_type = str(recording.get("hand_type") or "").strip().lower()
    if arm_type != "rokae" or hand_type not in {"wuji1", "wuji2"}:
        raise RuntimeError(
            "Exact camera post-render requires recorded ROKAE/Wuji identity; "
            f"recorded arm={arm_type!r}, hand={hand_type!r}."
        )
    return (
        recording,
        bool(recording.get("randomize", False)),
        collection_seed,
        episode_seed,
        arm_type,
        hand_type,
    )


def _visual_physx_restore_errors(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for field, valid_field in _VISUAL_PHYSX_FIELDS.items():
        if field not in expected:
            continue
        if field not in actual or valid_field not in expected:
            errors[field] = float("inf")
            continue
        expected_values = np.asarray(expected[field], dtype=np.float64)
        actual_values = np.asarray(actual[field], dtype=np.float64)
        if expected_values.shape != actual_values.shape:
            errors[field] = float("inf")
            continue
        valid = np.asarray(expected[valid_field], dtype=bool)
        while valid.ndim < expected_values.ndim:
            valid = np.expand_dims(valid, axis=-1)
        try:
            valid = np.broadcast_to(valid, expected_values.shape)
        except ValueError:
            errors[field] = float("inf")
            continue
        if not np.any(valid):
            continue
        difference = np.abs(expected_values - actual_values)
        errors[field] = (
            float(np.max(difference[valid]))
            if np.all(np.isfinite(difference[valid]))
            else float("inf")
        )
    return errors


def _write_exact_physx_videos(
    demo_dir: Path,
    env: object,
    physx_replay: object,
    *,
    recording: Mapping[str, object],
    fps: int,
    intrinsics_yaml: str,
    before_replace,
) -> tuple[Path, dict[str, object]]:
    raw_env = env.unwrapped
    trace = getattr(raw_env, "_physx_ground_truth_trace", None)
    if trace is None:
        raise RuntimeError(
            "Exact camera post-render requires the PhysX ground-truth trace "
            "to be configured before sim.reset()."
        )
    trace.validate_restore_schema(physx_replay.schema)
    required_camera_objects = {
        "head_camera",
        "wrist_left_camera",
        "wrist_right_camera",
    }
    live_camera_objects = set(getattr(raw_env, "_cameras", {}))
    missing_camera_objects = sorted(
        required_camera_objects - live_camera_objects
    )
    if missing_camera_objects:
        raise RuntimeError(
            "Exact camera post-render did not construct live cameras: "
            f"{missing_camera_objects}."
        )

    temporary = Path(tempfile.mkdtemp(prefix=".videos_physx-", dir=str(demo_dir)))
    writers: dict[str, Mp4VideoWriter] = {}
    stopped_writers: set[str] = set()
    first_frames: dict[str, np.ndarray] = {}
    expected_shapes: dict[str, tuple[int, int, int]] = {}
    camera_sources: dict[str, str] = {}
    maximum_restore_errors: dict[str, float] = {}
    first_restore_counts: dict[str, int] = {}
    warning_start = len(getattr(raw_env, "_isaac_warnings", ()))
    try:
        for step_index in range(int(physx_replay.step_count)):
            snapshot = physx_replay.snapshot(
                step_index,
                after_action=False,
            )
            restore_counts = trace.restore(snapshot, physx_replay.schema)
            if step_index == 0:
                first_restore_counts = {
                    str(name): int(value)
                    for name, value in restore_counts.items()
                }
            errors = _visual_physx_restore_errors(snapshot, trace.snapshot())
            for name, value in errors.items():
                maximum_restore_errors[name] = max(
                    maximum_restore_errors.get(name, 0.0),
                    float(value),
                )
            failed_fields = {
                name: value
                for name, value in errors.items()
                if not np.isfinite(value)
                or value > _VISUAL_PHYSX_RESTORE_ATOL
            }
            if failed_fields:
                raise RuntimeError(
                    "Exact PhysX camera state readback exceeded tolerance at "
                    f"frame {step_index}: {failed_fields}."
                )

            raw_observation = raw_env._compute_observation()
            observation = env.observation(raw_observation)
            if not camera_sources:
                camera_sources = select_canonical_camera_sources_for_trajectory(
                    [{"observations": observation}]
                )
                missing = [
                    key
                    for key in CANONICAL_CAMERA_KEYS
                    if key not in camera_sources
                ]
                if missing:
                    raise RuntimeError(
                        "Exact camera post-render is missing canonical cameras: "
                        f"{missing}."
                    )

            for camera_key in CANONICAL_CAMERA_KEYS:
                source_key = camera_sources[camera_key]
                image = observation_image(observation, source_key)
                if image is None:
                    raise RuntimeError(
                        f"Exact camera frame is missing source {source_key!r}."
                    )
                frame = postprocess_canonical_image(
                    _safe_rgb(image),
                    camera_key,
                    source_key,
                )
                if not frame.size or float(np.max(np.std(frame.astype(np.float32), axis=(0, 1)))) <= 1.0:
                    raise RuntimeError(
                        f"Exact camera {camera_key} produced a constant frame "
                        f"at index {step_index}."
                    )
                if camera_key not in writers:
                    first_frames[camera_key] = frame.copy()
                    expected_shapes[camera_key] = tuple(frame.shape)
                    writer = Mp4VideoWriter.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=21,
                        # Post-render already owns the selected GPU for RTX;
                        # NVENC may be unavailable in shared GPU namespaces.
                        # Use deterministic software encoding so a successful
                        # visual replay never depends on encoder permissions.
                        encoder="cpu",
                        thread_type="FRAME",
                        thread_count=2,
                    )
                    writer.start(str(temporary / f"{camera_key}.mp4"))
                    writers[camera_key] = writer
                elif tuple(frame.shape) != expected_shapes[camera_key]:
                    raise RuntimeError(
                        f"Exact camera {camera_key} changed shape from "
                        f"{expected_shapes[camera_key]} to {frame.shape}."
                    )
                writers[camera_key].write_frame(frame)

            warnings = tuple(getattr(raw_env, "_isaac_warnings", ()))[
                warning_start:
            ]
            fallback_warnings = [
                str(message)
                for message in warnings
                if "fallback frame" in str(message).lower()
                or "rgb unavailable" in str(message).lower()
            ]
            if fallback_warnings:
                raise RuntimeError(
                    "Exact camera post-render used a fallback image: "
                    + "; ".join(fallback_warnings)
                )
            if step_index % 100 == 0 or step_index + 1 == physx_replay.step_count:
                print(
                    "[Exact PhysX camera post-render] "
                    f"frame {step_index + 1}/{physx_replay.step_count}",
                    flush=True,
                )

        for camera_key, writer in writers.items():
            writer.stop()
            stopped_writers.add(camera_key)
        video_quality = [
            _validate_encoded_video(
                temporary / f"{camera_key}.mp4",
                expected_frames=int(physx_replay.step_count),
                expected_shape=expected_shapes[camera_key],
            )
            for camera_key in CANONICAL_CAMERA_KEYS
        ]
        camera_profile = recording.get("recording_camera_profile")
        write_video_camera_info(
            videos_dir=temporary,
            camera_keys=list(CANONICAL_CAMERA_KEYS),
            first_frames=first_frames,
            intrinsics_yaml=intrinsics_yaml,
            camera_views={},
            camera_profile=(
                dict(camera_profile)
                if isinstance(camera_profile, Mapping)
                else {}
            ),
        )
        reconstruction = {
            "format": "handson.exact_physx_camera_postrender",
            "version": 1,
            "state_phase": "recorded_pre_action_physx_state_per_control_frame",
            "physics_steps_per_frame": 0,
            "actions_reexecuted": False,
            "state_guidance_or_tracking_forces": False,
            "frame_count": int(physx_replay.step_count),
            "restore_tolerance": _VISUAL_PHYSX_RESTORE_ATOL,
            "maximum_restore_errors": maximum_restore_errors,
            "first_restore_counts": first_restore_counts,
            "camera_scope": list(CANONICAL_CAMERA_KEYS),
        }
        _write_json_atomic(
            temporary / "camera_reconstruction.json",
            reconstruction,
        )
        output = publish_arm_video_directory(
            demo_dir,
            temporary,
            before_replace=before_replace,
        )
        return output, {
            "status": "passed",
            "expected_frame_count": int(physx_replay.step_count),
            "camera_scope": list(CANONICAL_CAMERA_KEYS),
            "videos": video_quality,
            "state_reconstruction": reconstruction,
        }
    except BaseException:
        for camera_key, writer in writers.items():
            if camera_key in stopped_writers:
                continue
            try:
                writer.stop()
            except Exception:
                pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_encoded_video(
    path: Path,
    *,
    expected_frames: int,
    expected_shape: tuple[int, int, int],
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Post-render video cannot be decoded: {path}")
    frame_count = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frame_count += 1
            if tuple(bgr.shape) != expected_shape:
                raise RuntimeError(
                    f"Post-render video {path.name} decoded shape {bgr.shape}, "
                    f"expected {expected_shape}."
                )
            if not bgr.size or float(np.max(np.std(bgr.astype(np.float32), axis=(0, 1)))) <= 1.0:
                raise RuntimeError(
                    f"Post-render video {path.name} contains a constant frame "
                    f"at index {frame_count - 1}."
                )
    finally:
        capture.release()
    if frame_count != expected_frames:
        raise RuntimeError(
            f"Post-render video {path.name} has {frame_count} decoded frames, "
            f"expected {expected_frames}."
        )
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Post-render video is missing or empty: {path}")
    return {
        "path": path.name,
        "frame_count": frame_count,
        "height": expected_shape[0],
        "width": expected_shape[1],
        "channels": expected_shape[2],
        "all_frames_nonconstant": True,
        "decode_validation": "passed",
    }


def render_exact_physx_episode(
    demo_dir: Path,
    task_name: str,
    *,
    fps: int,
    intrinsics_yaml: str,
    before_replace,
) -> tuple[Path, dict[str, object]]:
    replay_path = demo_dir / "replay.zarr"
    if not replay_path.is_dir():
        raise FileNotFoundError(f"Missing replay.zarr: {replay_path}")

    # The source collector inherits camera deferral.  This fresh process
    # constructs all three cameras, but does not re-execute actions or advance
    # PhysX: each image is rendered from the exact recorded pre-action state.
    os.environ["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] = "0"
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
    os.environ["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"] = "0"
    os.environ["HANDSON_ISAACLAB_PREVIEW_LIVE_CAMERA"] = "1"
    # The exact camera lane restores recorded PhysX snapshots.  Contact-report
    # APIs and tensor views must therefore be authored while the fresh scene is
    # being built; constructing a trace after reset would invalidate Isaac Lab
    # tensor views and is deliberately rejected by PhysxGroundTruthTrace.
    os.environ["HANDSON_ISAACLAB_REQUIRE_REAL_RECORDING_CAMERAS"] = "1"
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    (
        recording,
        recorded_randomize,
        collection_seed,
        episode_seed,
        arm_type,
        hand_type,
    ) = _recorded_render_settings(demo_dir)
    actions = _load_episode_actions(replay_path)
    physx_replay = _load_physx_replay_trajectory(
        replay_path,
        demo_dir,
        expected_steps=len(actions),
    )

    config = CONFIG_MAPPING[task_name]()
    env = config.get_native_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=recorded_randomize,
        randomize_dynamics=False,
        seed=collection_seed,
        sim_backend="isaacsim",
        arm_type=arm_type,
        hand_type=hand_type,
        image_obs=True,
        headless=True,
        launch_isaac=True,
        isaac_device=os.environ.get("HANDSON_ISAACLAB_DEVICE", "cuda:0"),
    )
    env.reset(seed=episode_seed)
    output, camera_quality = _write_exact_physx_videos(
        demo_dir,
        env,
        physx_replay,
        recording=recording,
        fps=max(1, int(fps)),
        intrinsics_yaml=intrinsics_yaml,
        before_replace=before_replace,
    )
    return output, camera_quality
