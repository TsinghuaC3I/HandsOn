#!/usr/bin/env python3
"""Postrender a saved Isaac episode into complete canonical camera videos."""

from __future__ import annotations

from collections.abc import Mapping
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPO_ROOT / "handson"
for path in (_REPO_ROOT, _PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
from absl import app, flags

from handson.data.camera_info import (
    DEFAULT_FISHEYE_INTRINSICS_PATH,
    write_video_camera_info,
)
from handson.data.canonical_videos import (
    CANONICAL_CAMERA_KEYS,
    observation_image,
    postprocess_canonical_image,
    select_canonical_camera_sources_for_trajectory,
)
from handson.data.episode_publication_lock import (
    episode_publication_scope,
)
from handson.data.deferred_arm_replay_artifacts import (
    capture_deferred_arm_replay_artifacts,
)
from handson.data.video_writer import Mp4VideoWriter
from handson.data.simplified_post_ik import (
    declares_simplified_post_ik,
    load_validated_simplified_arm_plan,
    simplified_arm_plan_artifact_identity,
    simplified_post_ik_contract_errors,
)
from handson.runtime.collection_support import (
    BC011_COLLECTION_ANGLE_GUIDANCE_ENV,
)
from handson.tasks.mappings import CONFIG_MAPPING
from scripts.isaac_arm_replay_publish import publish_arm_video_directory
from scripts.replay_demos_zarr import (
    _load_episode_actions,
    _load_episode_initial_state,
    _load_episode_states,
    _load_physx_replay_trajectory,
    _load_ycb02_recording_state_extras,
    _load_xy6_recording_state_extras,
    _recorded_action_storage_layout,
    _replay_single_demo,
)


FLAGS = flags.FLAGS
_FU015_TASK_NAME = "t16_open_laptop"
_FU015_KITCHEN_PREVIEW_ENV = "HANDSON_FU015_KITCHEN_PREVIEW"
_AS019_TASK_NAME = "t22_pass_screwdriver"
flags.DEFINE_string("arm_replay_demo_dir", None, "Saved demo directory to render.")
flags.DEFINE_integer("arm_replay_fps", 30, "Output video frame rate.")
flags.DEFINE_string(
    "arm_replay_fisheye_intrinsics_yaml",
    str(DEFAULT_FISHEYE_INTRINSICS_PATH),
    "Wrist fisheye calibration used by canonical postprocessing.",
)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )


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


def _recorded_replay_settings(demo_dir: Path) -> tuple[bool, int, int | None]:
    """Recover the source episode's visual-randomization seed contract."""
    manifest = _read_json(demo_dir / "replay_manifest.json")
    recording = manifest.get("recording")
    if not isinstance(recording, dict):
        recording = {}
    collection = _read_json(demo_dir / "collection_meta.json")
    randomize = bool(
        recording.get("randomize")
        if "randomize" in recording
        else collection.get("randomize", False)
    )

    def integer(name: str) -> int | None:
        value = recording.get(name)
        if isinstance(value, bool):
            return None
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    collection_seed = integer("collection_run_seed")
    episode_seed = integer("episode_reset_seed")
    if randomize and episode_seed is None:
        raise RuntimeError(
            "Randomized arm replay requires recording.episode_reset_seed in "
            f"{demo_dir / 'replay_manifest.json'}"
        )
    env_seed = collection_seed if collection_seed is not None else episode_seed
    return randomize, int(env_seed or 0), episode_seed


def _restore_recorded_fu015_kitchen_preview(
    demo_dir: Path,
    task_name: str,
) -> str | None:
    """Restore the shared opt-in kitchen before constructing any replay env."""

    manifest_path = demo_dir / "replay_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Camera replay requires a valid manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, Mapping) or not manifest:
        raise RuntimeError(
            "Camera replay manifest must be a non-empty mapping: "
            f"{manifest_path}"
        )
    environment = manifest.get("environment", {})
    if not isinstance(environment, Mapping):
        raise RuntimeError(
            "Replay manifest environment must be a mapping"
        )
    variables = environment.get("variables", {})
    if not isinstance(variables, Mapping):
        raise RuntimeError(
            "Replay manifest environment.variables must be a mapping"
        )
    value = variables.get(_FU015_KITCHEN_PREVIEW_ENV, "0")
    if type(value) is not str or value not in {"0", "1"}:
        raise RuntimeError(
            "Replay manifest has invalid "
            f"{_FU015_KITCHEN_PREVIEW_ENV}={value!r}; expected '0' or '1'"
        )
    os.environ[_FU015_KITCHEN_PREVIEW_ENV] = value
    return value


def _update_arm_replay_status(
    demo_dir: Path,
    patch: dict[str, object],
) -> None:
    path = demo_dir / "deferred_arm_replay_status.json"
    status = _read_json(path)
    if str(patch.get("status") or "").strip().lower() in {
        "running",
        "complete",
    }:
        status.pop("error", None)
    status.update(patch)
    status["updated_at"] = _now_iso()
    _write_json_atomic(path, status)


def _mark_arm_render_started(demo_dir: Path) -> None:
    path = demo_dir / "deferred_arm_replay_status.json"
    status = _read_json(path)
    for key in (
        "artifact_proof",
        "artifact_quality_profile",
        "artifact_quality_evidence_sha256",
    ):
        status.pop(key, None)
    status.update(
        {
            "status": "running",
            "terminal": False,
            "publication_phase": "rendering",
            "artifact_validation_passed": False,
            "started_at": _now_iso(),
            "renderer_pid": os.getpid(),
            "updated_at": _now_iso(),
        }
    )
    _write_json_atomic(path, status)


def _prepare_arm_video_replacement(demo_dir: Path) -> None:
    status_path = demo_dir / "deferred_arm_replay_status.json"
    status = _read_json(status_path)
    status.pop("artifact_proof", None)
    status.pop("artifact_quality_profile", None)
    status.pop("artifact_quality_evidence_sha256", None)
    status.pop("error", None)
    status.update(
        {
            "status": "running",
            "terminal": False,
            "publication_phase": "replacing",
            "artifact_validation_passed": False,
            "updated_at": _now_iso(),
            "renderer_pid": os.getpid(),
        }
    )
    _write_json_atomic(status_path, status)

    cloud_path = demo_dir / "cloud_upload_meta.json"
    cloud = _read_json(cloud_path)
    if cloud:
        cloud.update(
            {
                "status": "stale_local_artifacts",
                "cloud_ack": "stale",
                "needs_reupload": True,
                "local_artifacts_changed": True,
                "local_artifacts_changed_at": _now_iso(),
                "local_artifacts_change_source": "isaac_arm_replay",
            }
        )
        cloud.pop("cloud_ack_at", None)
        _write_json_atomic(cloud_path, cloud)


def _mark_arm_render_complete(
    demo_dir: Path,
    task_name: str | None = None,
) -> None:
    manifest = _read_json(demo_dir / "replay_manifest.json")
    recording = manifest.get("recording")
    recording = recording if isinstance(recording, dict) else {}
    collection = _read_json(demo_dir / "collection_meta.json")
    simplified = bool(
        declares_simplified_post_ik(recording)
        or declares_simplified_post_ik(collection)
    )
    artifact_proof = capture_deferred_arm_replay_artifacts(
        demo_dir,
        task_name=task_name,
    )
    quality = artifact_proof.get("quality")
    quality = quality if isinstance(quality, Mapping) else {}
    _update_arm_replay_status(
        demo_dir,
        {
            "status": "complete",
            "terminal": False,
            "publication_phase": "complete",
            "artifact_validation_passed": True,
            "artifact_proof": artifact_proof,
            **(
                {
                    "artifact_quality_profile": str(quality["profile"]),
                    "artifact_quality_evidence_sha256": str(
                        quality["evidence_sha256"]
                    ),
                }
                if quality
                else {}
            ),
            **(
                {
                    "simplified_arm_plan_artifact": (
                        simplified_arm_plan_artifact_identity(demo_dir)
                    )
                }
                if simplified
                else {}
            ),
            "completed_at": _now_iso(),
            "renderer_pid": os.getpid(),
        },
    )


def _mark_arm_render_failed(demo_dir: Path, exc: BaseException) -> None:
    _update_arm_replay_status(
        demo_dir,
        {
            "status": "failed",
            "terminal": False,
            "publication_phase": "failed",
            "artifact_validation_passed": False,
            "completed_at": _now_iso(),
            "renderer_pid": os.getpid(),
            "error": f"{type(exc).__name__}: {exc}",
        },
    )


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


def _recorded_replay_backend(demo_dir: Path) -> str:
    manifest = _read_json(demo_dir / "replay_manifest.json")
    recording = manifest.get("recording")
    recording = recording if isinstance(recording, dict) else {}
    collection = _read_json(demo_dir / "collection_meta.json")
    backend = str(
        recording.get("sim_backend")
        or manifest.get("sim_backend")
        or collection.get("sim_backend")
        or ""
    ).strip().lower()
    return {
        "isaac": "isaacsim",
        "newton": "isaacsim_newton",
        "nvidia_newton": "isaacsim_newton",
        "isaac_newton": "isaacsim_newton",
    }.get(backend, backend)


def _recorded_camera_profile(demo_dir: Path) -> dict[str, object]:
    manifest = _read_json(demo_dir / "replay_manifest.json")
    recording = manifest.get("recording")
    recording = recording if isinstance(recording, dict) else {}
    profile = recording.get("recording_camera_profile")
    if not isinstance(profile, dict):
        profile = _read_json(demo_dir / "collection_meta.json").get(
            "recording_camera_profile"
        )
    return dict(profile) if isinstance(profile, dict) else {}


def _configure_canonical_camera_postrender(
    *,
    source_backend: str,
    simplified_post_ik: bool = False,
    exact_physx_visual_state: bool = False,
) -> str:
    """Select visual-only replay without changing physical replay semantics."""
    os.environ.pop(BC011_COLLECTION_ANGLE_GUIDANCE_ENV, None)
    os.environ["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] = "0"
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
    os.environ["HANDSON_ISAACLAB_DIRECT_HAND_CONTROL"] = "0"
    os.environ.pop("HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY", None)
    # Global IK consumes a PhysX-v2 substep trace.  Canonical camera
    # postrendering never claims or requires that physical-replay trace.
    os.environ["HANDSON_GLOBAL_IK_PLAN"] = "0"

    exact_physx_visual_state = bool(
        exact_physx_visual_state
        and source_backend == "isaacsim"
        and not simplified_post_ik
    )
    kinematic_visual_reconstruction = bool(
        source_backend == "isaacsim_newton" or simplified_post_ik
    )
    FLAGS.sim_backend = "isaacsim"
    if simplified_post_ik:
        # The output scene contains full canonical ROKAE arms.  The source's
        # simplified type is a live-control profile, not a separate XML.
        os.environ["HANDSON_ARM_TYPE"] = "rokae"
        os.environ["HANDSON_HAND_TYPE"] = "wuji1"
    # AS019's canonical videos are a visual reconstruction from the complete
    # action-aligned PhysX snapshots.  It must not silently fall back to raw
    # 30 Hz action replay: the contact-gated orientation guide has hidden
    # state, and action replay alone cannot certify identical camera frames.
    # This state-restore path is visual-only and never counts as the task's
    # independent physical-replay proof.
    if exact_physx_visual_state:
        os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    FLAGS.restore_state = not exact_physx_visual_state
    FLAGS.state_guided = bool(
        not exact_physx_visual_state
        and not kinematic_visual_reconstruction
    )
    FLAGS.kinematic_visual_replay = kinematic_visual_reconstruction
    FLAGS.save_depth = False
    FLAGS.extend_steps = 0
    return (
        "physx_state_restore_visual"
        if exact_physx_visual_state
        else (
            "kinematic_visual"
            if kinematic_visual_reconstruction
            else "state_guided"
        )
    )


def _write_arm_videos(
    demo_dir: Path,
    trajectory: list[dict],
    *,
    fps: int,
) -> Path:
    camera_sources = select_canonical_camera_sources_for_trajectory(trajectory)
    missing = [key for key in CANONICAL_CAMERA_KEYS if key not in camera_sources]
    if missing:
        raise RuntimeError(f"Arm replay is missing canonical cameras: {missing}")

    temporary = Path(tempfile.mkdtemp(prefix=".videos_arm-", dir=str(demo_dir)))
    first_frames: dict[str, np.ndarray] = {}
    try:
        for camera_key in CANONICAL_CAMERA_KEYS:
            source_key = camera_sources[camera_key]
            output_path = temporary / f"{camera_key}.mp4"
            writer = Mp4VideoWriter.create_h264(
                fps=fps,
                codec="h264",
                input_pix_fmt="rgb24",
                crf=21,
                thread_type="FRAME",
                thread_count=2,
            )
            writer.start(str(output_path))
            try:
                for transition in trajectory:
                    image = observation_image(
                        transition["observations"],
                        source_key,
                    )
                    if image is None:
                        raise RuntimeError(
                            f"Arm replay frame is missing camera {source_key!r}"
                        )
                    frame = postprocess_canonical_image(
                        _safe_rgb(image),
                        camera_key,
                        source_key,
                    )
                    if float(frame.reshape(-1, 3).std(axis=0).max()) <= 1.0:
                        raise RuntimeError(f"Constant recording camera frame: {camera_key}")
                    first_frames.setdefault(camera_key, frame)
                    writer.write_frame(frame)
            finally:
                writer.stop()

        from repair_and_upload_wrist_cameras import inspect_video
        for camera_key in CANONICAL_CAMERA_KEYS:
            inspected = inspect_video(temporary / f"{camera_key}.mp4")
            frame = first_frames[camera_key]
            if (inspected.has_issue or inspected.frame_count != len(trajectory)
                    or (inspected.height, inspected.width) != frame.shape[:2]):
                raise RuntimeError(f"Invalid encoded recording camera {camera_key}: {inspected}")
        write_video_camera_info(
            videos_dir=temporary,
            camera_keys=CANONICAL_CAMERA_KEYS,
            first_frames=first_frames,
            intrinsics_yaml=FLAGS.arm_replay_fisheye_intrinsics_yaml,
            camera_views={},
            camera_profile=_recorded_camera_profile(demo_dir),
        )
        return publish_arm_video_directory(
            demo_dir,
            temporary,
            before_replace=lambda: _prepare_arm_video_replacement(
                demo_dir
            ),
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _render_and_publish(demo_dir: Path, task_name: str) -> Path:
    replay_path = demo_dir / "replay.zarr"
    if not replay_path.is_dir():
        raise FileNotFoundError(f"Missing replay.zarr: {replay_path}")

    if task_name in {"t35_rotate_propeller", "t47_trigger_drill_on_block", "t29_place_wine_glass", "t33_reposition_marker_grasp"}:
        from scripts.isaac_exact_physx_camera_postrender import render_exact_physx_episode

        output, quality = render_exact_physx_episode(
            demo_dir,
            task_name,
            fps=max(1, int(FLAGS.arm_replay_fps)),
            intrinsics_yaml=FLAGS.arm_replay_fisheye_intrinsics_yaml,
            before_replace=lambda: _prepare_arm_video_replacement(demo_dir),
        )
        _update_arm_replay_status(demo_dir, {"camera_postrender_quality": quality})
        return output

    # Collection-only render flags are captured in the source manifest.  This
    # replay must always construct the complete robots and all three cameras.
    _restore_recorded_fu015_kitchen_preview(demo_dir, task_name)
    source_backend = _recorded_replay_backend(demo_dir)
    manifest = _read_json(demo_dir / "replay_manifest.json")
    recording = manifest.get("recording")
    recording = recording if isinstance(recording, dict) else {}
    action_storage_layout = _recorded_action_storage_layout(
        recording, task_id=task_name,
    )
    collection = _read_json(demo_dir / "collection_meta.json")
    simplified_post_ik = bool(
        declares_simplified_post_ik(recording)
        or declares_simplified_post_ik(collection)
    )
    if simplified_post_ik:
        contract_errors = simplified_post_ik_contract_errors(
            recording,
            collection,
        )
        if contract_errors:
            raise RuntimeError(
                "Invalid simplified-post-IK declaration: "
                + "; ".join(contract_errors)
            )
    # XY2 needs the recorded mouse-button DOFs as well as the laptop hinge.
    # The existing full PhysX visual path restores those only for post-render;
    # formal physical certification continues to compare without restoration.
    exact_physx_visual_state = task_name in {
        _AS019_TASK_NAME, "eval_comb02_open_laptop_and_click_mouse",
    }
    _configure_canonical_camera_postrender(
        source_backend=source_backend,
        simplified_post_ik=simplified_post_ik,
        exact_physx_visual_state=exact_physx_visual_state,
    )
    recorded_randomize, env_seed, reset_seed = _recorded_replay_settings(
        demo_dir
    )
    FLAGS.randomize = recorded_randomize
    actions = _load_episode_actions(replay_path)
    initial_state = _load_episode_initial_state(replay_path)
    states = _load_episode_states(replay_path)
    if initial_state is None or states is None:
        raise RuntimeError(
            "Canonical camera postrender requires per-frame recorded state"
        )
    # YCB02 keeps the complete eight-body layout and event identity in
    # recording-only arrays.  Load those arrays here as well as in the main
    # replay CLI; otherwise canonical arm-camera postrender would silently
    # construct a fresh randomized scene and render the arm path over the
    # wrong cup/ball arrangement.
    recording_state_extras = _load_ycb02_recording_state_extras(
        replay_path,
        expected_steps=(
            int(actions.shape[0])
            if task_name == "t27_place_largest_fitting_ball"
            else None
        ),
    )
    if task_name == "eval_comb06_season_and_stir":
        recording_state_extras.update(_load_xy6_recording_state_extras(
            replay_path, expected_steps=int(actions.shape[0]),
        ))
    physx_visual_replay = (
        _load_physx_replay_trajectory(
            replay_path,
            demo_dir,
            expected_steps=actions.shape[0],
        )
        if exact_physx_visual_state
        else None
    )
    visual_arm_plan_qpos = (
        load_validated_simplified_arm_plan(
            replay_path,
            expected_frames=len(states),
        )
        if simplified_post_ik
        else None
    )

    config = CONFIG_MAPPING[task_name]()
    (
        _succeed,
        trajectory,
        _camera_metadata,
        _camera_sources,
        _runtime_recording_schema,
    ) = _replay_single_demo(
        actions,
        initial_state,
        states,
        task_name,
        config,
        env_seed=env_seed,
        reset_seed=reset_seed,
        desc=f"{demo_dir.name}: canonical cameras",
        sim_backend="isaacsim",
        physx_replay=physx_visual_replay,
        close_env=False,
        visual_arm_plan_qpos=visual_arm_plan_qpos,
        arm_type_override="rokae" if simplified_post_ik else None,
        hand_type_override="wuji1" if simplified_post_ik else None,
        recording_state_extras=recording_state_extras,
        action_storage_layout=action_storage_layout,
    )
    if len(trajectory) != len(actions):
        raise RuntimeError(
            f"Canonical camera postrender frame mismatch: actions={len(actions)}, "
            f"frames={len(trajectory)}"
        )
    output = _write_arm_videos(
        demo_dir,
        trajectory,
        fps=max(1, int(FLAGS.arm_replay_fps)),
    )
    return output


def main(_argv: list[str]) -> None:
    if not FLAGS.arm_replay_demo_dir:
        raise ValueError("--arm_replay_demo_dir is required")
    demo_dir = Path(FLAGS.arm_replay_demo_dir).expanduser().resolve()
    task_name = str(FLAGS.exp_name).strip()
    if not task_name:
        raise ValueError("--exp_name is required")

    # Manual/direct invocation owns this lock for the complete replay and
    # publish operation.  A deferred worker passes the already-locked file
    # descriptor through Popen(pass_fds=...); episode_publication_scope
    # validates that exact inode/flock capability before avoiding a second
    # blocking acquisition in this child.
    with episode_publication_scope(demo_dir):
        _mark_arm_render_started(demo_dir)
        try:
            output = _render_and_publish(demo_dir, task_name)
            _mark_arm_render_complete(demo_dir, task_name)
        except BaseException as exc:
            _mark_arm_render_failed(demo_dir, exc)
            raise
    print(f"[Isaac arm replay] published {output}", flush=True)
    # SimulationContext.stop() can block after RTX capture. This process is
    # dedicated to one replay, so terminate only after videos are published.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    app.run(main)
