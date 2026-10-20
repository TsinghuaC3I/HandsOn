#!/usr/bin/env python3
"""Repair IsaacLab wrist videos and replace their KS3 objects safely.

The default mode only scans and reports episodes. Rendering, local replacement,
and cloud upload each require an explicit flag. Cloud metadata is committed last,
after every required KS3 PUT returned 2xx with a verifiable ETag.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "handson"
for _import_root in (REPO_ROOT, PACKAGE_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from handson.data.deferred_arm_replay_artifacts import (
    capture_deferred_arm_replay_artifacts,
)
from handson.data.episode_publication_lock import episode_publication_lock
from handson.sim.isaaclab.replay_visuals import (
    reapply_reference_robot_appearance_for_replay,
)

DEFAULT_DEMOS_ROOT = Path("/home/user/handson_demos")
DEFAULT_STAGING_ROOT = Path("/home/user/handson_wrist_rerenders")
DEFAULT_BACKUP_ROOT = Path("/home/user/handson_wrist_rerender_backups")
DEFAULT_ISAAC_PYTHON = Path("/home/user/.local/share/mamba/envs/isaacsim/bin/python")
WRIST_VIDEO_PATHS = ("videos/wrist_left.mp4", "videos/wrist_right.mp4")
RERENDER_META_NAME = "wrist_rerender_meta.json"
UPLOAD_FAILURE_NAME = "wrist_rerender_upload_failure.json"
RERENDER_SOURCE = "wrist_rerender_direct_ks3"
RERENDER_VERSION = 1
LOCAL_TZ = dt.timezone(dt.timedelta(hours=8))
ABC_TASK_NAME_PATTERN = re.compile(r"^[abc]\d{2}_")


@dataclass(frozen=True)
class Episode:
    path: Path
    task: str
    collected_at: str
    collector_id: str
    video_fps: int
    expected_frames: int
    apply_fisheye_distortion: bool
    fisheye_intrinsics_yaml: str


@dataclass(frozen=True)
class VideoInspection:
    path: str
    exists: bool
    readable: bool
    frame_count: int
    width: int
    height: int
    obstruction_frames: tuple[int, ...]
    error: str = ""

    @property
    def has_issue(self) -> bool:
        return (
            not self.exists
            or not self.readable
            or self.frame_count <= 0
            or bool(self.obstruction_frames)
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return dt.datetime.now(tz=LOCAL_TZ).isoformat(timespec="seconds")


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{int(value)}B" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{size}B"


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _parse_time(value: str, *, end_of_day: bool = False) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    date_only = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", text))
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if date_only and end_of_day:
        parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.astimezone(dt.timezone.utc)


def _episode_time(episode: Episode) -> dt.datetime | None:
    try:
        return _parse_time(episode.collected_at)
    except ValueError:
        return None


def _recording_settings(demo_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    collection = _read_json(demo_dir / "collection_meta.json")
    manifest = _read_json(demo_dir / "replay_manifest.json")
    recording = manifest.get("recording")
    if not isinstance(recording, dict):
        recording = {}
    return collection, recording


def _episode_from_dir(demo_dir: Path) -> Episode | None:
    collection, recording = _recording_settings(demo_dir)
    backend = str(collection.get("sim_backend") or recording.get("sim_backend") or "").lower()
    engine_code = str(collection.get("engine_code") or "").upper()
    if backend not in {"isaac", "isaacsim", "isaacsim_newton"} and engine_code not in {
        "E02",
        "E04",
    }:
        return None

    manifest = _read_json(demo_dir / "replay_manifest.json")
    task_block = manifest.get("task") if isinstance(manifest.get("task"), dict) else {}
    task = str(collection.get("task_name") or task_block.get("name") or "").strip()
    if not task or ABC_TASK_NAME_PATTERN.match(task.lower()) is None:
        return None

    try:
        video_fps = int(collection.get("video_fps") or recording.get("video_fps") or 30)
        expected_frames = int(collection.get("step_count") or recording.get("step_count") or 0)
    except (TypeError, ValueError):
        return None
    if video_fps <= 0 or expected_frames <= 0:
        return None

    return Episode(
        path=demo_dir.resolve(),
        task=task,
        collected_at=str(collection.get("collected_at") or ""),
        collector_id=str(
            collection.get("collector_id")
            or collection.get("collection_collector_id")
            or ""
        ).strip(),
        video_fps=video_fps,
        expected_frames=expected_frames,
        apply_fisheye_distortion=bool(recording.get("apply_fisheye_distortion", False)),
        fisheye_intrinsics_yaml=str(recording.get("fisheye_intrinsics_yaml") or ""),
    )


def discover_episodes(root: Path) -> list[Episode]:
    episodes = []
    for replay_path in root.rglob("replay.zarr"):
        if not replay_path.is_dir():
            continue
        episode = _episode_from_dir(replay_path.parent)
        if episode is not None:
            episodes.append(episode)
    return sorted(episodes, key=lambda item: (item.collected_at, item.path.name))


def _frame_is_housing_obstruction(frame: np.ndarray) -> bool:
    """Recognize the almost uniform gray camera-housing frames from the old mount."""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        return True
    sample = image[::8, ::8, :3].astype(np.uint8, copy=False)
    quantized = (sample // 16).reshape(-1, 3)
    if quantized.size == 0:
        return True
    _, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant_ratio = float(counts.max()) / float(len(quantized))
    return dominant_ratio >= 0.98 and float(sample.std()) <= 12.0


def inspect_video(
    path: Path,
    *,
    scan_stride: int = 1,
    stop_on_obstruction: bool = False,
) -> VideoInspection:
    if not path.is_file():
        return VideoInspection(str(path), False, False, 0, 0, 0, (), "missing")
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required; run this script with the HandsOn or Isaac Python environment"
        ) from exc

    frame_count = 0
    width = 0
    height = 0
    obstruction_frames: list[int] = []
    try:
        with av.open(str(path), mode="r") as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                frame_count += 1
                width = int(frame.width)
                height = int(frame.height)
                if frame_index % max(1, int(scan_stride)):
                    continue
                if _frame_is_housing_obstruction(frame.to_ndarray(format="rgb24")):
                    obstruction_frames.append(frame_index)
                    if stop_on_obstruction:
                        break
    except Exception as exc:  # noqa: BLE001 - codecs expose several exception types.
        return VideoInspection(
            str(path), True, False, frame_count, width, height, tuple(obstruction_frames), str(exc)
        )
    return VideoInspection(
        str(path), True, frame_count > 0, frame_count, width, height, tuple(obstruction_frames)
    )


def _filter_episodes(episodes: list[Episode], args: argparse.Namespace) -> list[Episode]:
    tasks = {value.strip().lower() for value in args.task if value.strip()}
    names = {Path(value).name for value in args.episode if value.strip()}
    from_time = _parse_time(args.from_time) if args.from_time else None
    to_time = _parse_time(args.to_time, end_of_day=True) if args.to_time else None
    selected = []
    for episode in episodes:
        if tasks and episode.task.lower() not in tasks:
            continue
        if names and episode.path.name not in names:
            continue
        if args.collector and episode.collector_id != args.collector:
            continue
        episode_time = _episode_time(episode)
        if from_time and (episode_time is None or episode_time < from_time):
            continue
        if to_time and (episode_time is None or episode_time > to_time):
            continue
        selected.append(episode)
    return selected


def _scan_candidates(
    episodes: list[Episode], args: argparse.Namespace
) -> tuple[list[Episode], dict[str, list[VideoInspection]]]:
    candidates = []
    inspections: dict[str, list[VideoInspection]] = {}
    for index, episode in enumerate(episodes, start=1):
        if args.all:
            inspections[str(episode.path)] = []
            candidates.append(episode)
            print(f"[select {index}/{len(episodes)}] {episode.path.name}")
            if args.max_episodes and len(candidates) >= args.max_episodes:
                break
            continue
        reports = [
            inspect_video(
                episode.path / rel_path,
                scan_stride=args.scan_stride,
                stop_on_obstruction=True,
            )
            for rel_path in WRIST_VIDEO_PATHS
        ]
        inspections[str(episode.path)] = reports
        has_issue = any(report.has_issue for report in reports)
        repaired_before = (episode.path / RERENDER_META_NAME).is_file()
        if has_issue or (args.upload_cloud and repaired_before):
            candidates.append(episode)
        state = "NEEDS_REPAIR" if has_issue else "ok"
        print(f"[scan {index}/{len(episodes)}] {state:12s} {episode.path.name}")
        if args.max_episodes and len(candidates) >= args.max_episodes:
            break
    return candidates, inspections


def _active_recorders() -> list[tuple[int, str]]:
    active = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if "record_demos_zarr.py" in command:
            active.append((int(entry.name), command.strip()))
    return sorted(active)


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    if array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and np.nanmax(array) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got {array.shape}")
    return np.ascontiguousarray(array)


def _initial_state_layout(raw_env, config) -> tuple[tuple[str, int], ...]:
    state = raw_env._compute_observation()["state"]
    return tuple(
        (key, np.asarray(state[key]).ravel().size)
        for key in config.proprio_keys
    )


def _split_state(
    config, state_vec: np.ndarray, layout: tuple[tuple[str, int], ...]
) -> dict[str, np.ndarray]:
    parts = {}
    offset = 0
    for key, size in layout:
        parts[key] = np.asarray(state_vec[offset : offset + size], dtype=np.float64)
        offset += size
    if offset != np.asarray(state_vec).size:
        raise ValueError(
            f"recorded state has {np.asarray(state_vec).size} values, "
            f"but {config.task_name} consumes {offset}"
        )
    return parts


def _step_with_recorded_action(env, action_flat: np.ndarray):
    action_flat = np.asarray(action_flat, dtype=np.float64)
    if action_flat.ndim == 1 and action_flat.shape[0] in (46, 54):
        side_dim = action_flat.shape[0] // 2
        raw_action: Any = {
            "right": action_flat[:side_dim],
            "left": action_flat[side_dim:],
        }
    else:
        raw_action = action_flat
    raw_obs, reward, done, truncated, info = env.unwrapped.step(raw_action)
    return env.observation(raw_obs), reward, done, truncated, info


def _save_worker_results(path: Path, results: list[dict[str, Any]]) -> None:
    _write_json_atomic(path, {"version": RERENDER_VERSION, "results": results})


def _staged_result_is_resumable(item: dict[str, Any], result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict) or not result.get("success"):
        return False
    if int(result.get("frame_count") or 0) != int(item.get("expected_frames") or 0):
        return False
    stage_dir = Path(result.get("stage_dir") or item.get("stage_dir") or "")
    return all(
        (stage_dir / rel_path).is_file() and (stage_dir / rel_path).stat().st_size > 0
        for rel_path in WRIST_VIDEO_PATHS
    )


def _observation_after_state_restore(env, config, state, layout):
    """Restore task state without rendering, then render exactly once."""
    raw_env = env.unwrapped
    restore = getattr(raw_env, "restore_recorded_initial_state", None)
    if restore is None:
        raise RuntimeError("Isaac environment does not support recorded-state restoration")
    image_obs = bool(raw_env.image_obs)
    raw_env.image_obs = False
    try:
        restore(_split_state(config, state, layout))
    finally:
        raw_env.image_obs = image_obs
    return env.observation(raw_env._compute_observation())


def _worker_rerender_episode(env, config, item: dict[str, Any]) -> dict[str, Any]:
    import zarr

    from handson.data.canonical_videos import (
        observation_image,
        postprocess_canonical_image,
        select_canonical_camera_sources,
    )
    from handson.data.video_writer import Mp4VideoWriter

    episode_path = Path(item["episode"])
    stage_dir = Path(item["stage_dir"])
    zarr_root = zarr.open(str(episode_path / "replay.zarr"), mode="r")
    actions = np.asarray(zarr_root["data"]["action"])
    if "state" not in zarr_root["data"]:
        raise RuntimeError("replay.zarr has no state array; exact state-guided rerender is impossible")
    states = np.asarray(zarr_root["data"]["state"])
    if len(actions) != len(states):
        raise RuntimeError(f"action/state length mismatch: {len(actions)} != {len(states)}")
    expected_frames = int(item["expected_frames"])
    if len(actions) != expected_frames:
        raise RuntimeError(f"manifest/action length mismatch: {expected_frames} != {len(actions)}")

    if bool(item.get("apply_fisheye_distortion")):
        os.environ["HANDSON_APPLY_FISHEYE_DISTORTION"] = "1"
    else:
        os.environ["HANDSON_APPLY_FISHEYE_DISTORTION"] = "0"
    intrinsics = str(item.get("fisheye_intrinsics_yaml") or "")
    if intrinsics:
        os.environ["HANDSON_FISHEYE_INTRINSICS_YAML"] = intrinsics

    raw_env = env.unwrapped
    image_obs = bool(raw_env.image_obs)
    raw_env.image_obs = False
    try:
        env.reset()
    finally:
        raw_env.image_obs = image_obs
    reapply_reference_robot_appearance_for_replay(env)
    layout = _initial_state_layout(raw_env, config)
    observation = _observation_after_state_restore(env, config, states[0], layout)
    camera_sources = select_canonical_camera_sources(observation, env=env)
    missing = [key for key in ("wrist_left", "wrist_right") if key not in camera_sources]
    if missing:
        raise RuntimeError(f"missing wrist camera source(s): {missing}")

    videos_dir = stage_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    temp_paths = {}
    final_paths = {}
    single_wrist_camera = not bool(getattr(raw_env._spec, "bimanual", False))
    encoded_keys = ("wrist_right",) if single_wrist_camera else ("wrist_left", "wrist_right")
    try:
        for key in encoded_keys:
            final_path = videos_dir / f"{key}.mp4"
            temp_path = videos_dir / f".{key}.{os.getpid()}.tmp.mp4"
            writer = Mp4VideoWriter.create_h264(
                fps=int(item["video_fps"]),
                codec="h264",
                input_pix_fmt="rgb24",
                crf=int(item.get("video_crf") or 21),
                thread_type="FRAME",
                thread_count=2,
            )
            writer.start(str(temp_path))
            writers[key] = writer
            temp_paths[key] = temp_path
            final_paths[key] = final_path

        for step_index, action in enumerate(actions):
            for key, writer in writers.items():
                source = camera_sources[key]
                image = observation_image(observation, source)
                if image is None:
                    raise RuntimeError(f"camera source {source!r} disappeared at frame {step_index}")
                frame = postprocess_canonical_image(_safe_rgb(image), key, source)
                writer.write_frame(_safe_rgb(frame))

            image_obs = bool(raw_env.image_obs)
            raw_env.image_obs = False
            try:
                _step_with_recorded_action(env, action)
            finally:
                raw_env.image_obs = image_obs
            source_index = min(step_index + 1, len(states) - 1)
            observation = _observation_after_state_restore(
                env, config, states[source_index], layout
            )

        for writer in writers.values():
            writer.stop()
        for key, temp_path in temp_paths.items():
            os.replace(temp_path, final_paths[key])
        if single_wrist_camera:
            left_path = videos_dir / "wrist_left.mp4"
            left_temp = videos_dir / f".wrist_left.{os.getpid()}.tmp.mp4"
            shutil.copy2(final_paths["wrist_right"], left_temp)
            os.replace(left_temp, left_path)
    except Exception:
        for writer in writers.values():
            try:
                writer.abort()
            except Exception:
                pass
        for temp_path in temp_paths.values():
            temp_path.unlink(missing_ok=True)
        raise

    return {
        "success": True,
        "episode": str(episode_path),
        "stage_dir": str(stage_dir),
        "task": str(item["task"]),
        "frame_count": len(actions),
        "camera_sources": camera_sources,
    }


def _run_worker(manifest_path: Path, results_path: Path) -> int:
    os.environ.setdefault("HANDSON_ISAACSIM_OFFLINE_RENDER_PASSES", "16")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from handson.tasks.mappings import CONFIG_MAPPING

    manifest = _read_json(manifest_path)
    items = manifest.get("episodes")
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"worker manifest has no episodes: {manifest_path}")
    tasks = {str(item.get("task") or "") for item in items if isinstance(item, dict)}
    if len(tasks) != 1 or "" in tasks:
        raise RuntimeError(f"one worker may handle exactly one task, got {sorted(tasks)}")
    task = next(iter(tasks))
    previous_payload = _read_json(results_path)
    previous_results = previous_payload.get("results")
    if not isinstance(previous_results, list):
        previous_results = []
    result_by_episode = {
        str(result.get("episode") or ""): result
        for result in previous_results
        if isinstance(result, dict) and str(result.get("episode") or "")
    }
    pending_items = [
        item
        for item in items
        if not _staged_result_is_resumable(
            item,
            result_by_episode.get(str(item.get("episode") or "")),
        )
    ]
    if not pending_items:
        print(f"[worker resume] all {len(items)} episode(s) already staged", flush=True)
        return 0

    config = CONFIG_MAPPING[task]()
    os.environ["HANDSON_ARM_TYPE"] = "rokae"
    env = config.get_native_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=False,
        randomize_dynamics=False,
        seed=int(manifest.get("seed") or 0),
        sim_backend="isaacsim",
        arm_type="rokae",
        hand_type="wuji1",
    )
    try:
        for index, item in enumerate(items, start=1):
            episode_key = str(item.get("episode") or "")
            existing = result_by_episode.get(episode_key)
            if _staged_result_is_resumable(item, existing):
                print(f"[worker resume {index}/{len(items)}] {Path(episode_key).name}", flush=True)
                continue
            print(f"[worker {index}/{len(items)}] {Path(item['episode']).name}", flush=True)
            try:
                result = _worker_rerender_episode(env, config, item)
            except Exception as exc:  # noqa: BLE001 - preserve batch progress.
                result = {
                    "success": False,
                    "episode": str(item.get("episode") or ""),
                    "stage_dir": str(item.get("stage_dir") or ""),
                    "task": task,
                    "error": str(exc),
                }
                print(f"[worker failed] {result['episode']}: {exc}", file=sys.stderr, flush=True)
            result_by_episode[episode_key] = result
            ordered_results = [
                result_by_episode[str(manifest_item.get("episode") or "")]
                for manifest_item in items
                if str(manifest_item.get("episode") or "") in result_by_episode
            ]
            _save_worker_results(results_path, ordered_results)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return 0 if all(
        result_by_episode.get(str(item.get("episode") or ""), {}).get("success")
        for item in items
    ) else 1


def _worker_item(episode: Episode, stage_dir: Path) -> dict[str, Any]:
    _, recording = _recording_settings(episode.path)
    return {
        "episode": str(episode.path),
        "stage_dir": str(stage_dir),
        "task": episode.task,
        "video_fps": episode.video_fps,
        "video_crf": int(recording.get("video_spool_crf") or 21),
        "expected_frames": episode.expected_frames,
        "apply_fisheye_distortion": episode.apply_fisheye_distortion,
        "fisheye_intrinsics_yaml": episode.fisheye_intrinsics_yaml,
    }


def _launch_workers(
    episodes: list[Episode], args: argparse.Namespace, batch_dir: Path
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Episode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.task, []).append(episode)
    jobs = []
    for task, task_episodes in grouped.items():
        task_dir = batch_dir / task
        manifest_path = task_dir / "worker_manifest.json"
        results_path = task_dir / "worker_results.json"
        payload = {
            "version": RERENDER_VERSION,
            "batch_id": batch_dir.name,
            "seed": args.seed,
            "episodes": [
                _worker_item(episode, task_dir / episode.path.name)
                for episode in task_episodes
            ],
        }
        _write_json_atomic(manifest_path, payload)
        command = [
            str(Path(args.isaac_python).expanduser()),
            str(Path(__file__).resolve()),
            "--worker-manifest",
            str(manifest_path),
            "--worker-results",
            str(results_path),
        ]
        jobs.append(
            {
                "task": task,
                "episode_count": len(task_episodes),
                "results_path": results_path,
                "command": command,
            }
        )

    all_results: dict[str, dict[str, Any]] = {}
    pending = list(jobs)
    active: list[dict[str, Any]] = []
    try:
        while pending or active:
            while pending and len(active) < args.parallel_tasks:
                job = pending.pop(0)
                print(
                    f"[render] task={job['task']} episodes={job['episode_count']}",
                    flush=True,
                )
                job["process"] = subprocess.Popen(job["command"], cwd=REPO_ROOT)
                active.append(job)

            completed = []
            for job in active:
                process = job["process"]
                returncode = process.poll()
                if returncode is None:
                    continue
                worker_payload = _read_json(job["results_path"])
                worker_results = worker_payload.get("results")
                if not isinstance(worker_results, list):
                    worker_results = []
                for item in worker_results:
                    if isinstance(item, dict):
                        all_results[str(Path(item.get("episode") or "").resolve())] = item
                if returncode != 0:
                    print(
                        f"[render warning] worker for {job['task']} exited with {returncode}",
                        flush=True,
                    )
                completed.append(job)
            for job in completed:
                active.remove(job)
            if active and not completed:
                time.sleep(0.5)
    except BaseException:
        for job in active:
            process = job["process"]
            if process.poll() is None:
                process.terminate()
        for job in active:
            try:
                job["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                job["process"].kill()
        raise
    return all_results


def _validate_staged_videos(episode: Episode, stage_dir: Path) -> dict[str, VideoInspection]:
    reports = {}
    for rel_path in WRIST_VIDEO_PATHS:
        staged = stage_dir / rel_path
        report = inspect_video(staged, scan_stride=1, stop_on_obstruction=False)
        if report.has_issue:
            raise RuntimeError(f"staged video failed validation: {rel_path}: {asdict(report)}")
        if report.frame_count != episode.expected_frames:
            raise RuntimeError(
                f"staged frame count mismatch for {rel_path}: "
                f"{report.frame_count} != {episode.expected_frames}"
            )
        original = inspect_video(
            episode.path / rel_path, scan_stride=max(1, episode.expected_frames), stop_on_obstruction=True
        )
        if original.readable and (report.width, report.height) != (original.width, original.height):
            raise RuntimeError(
                f"staged dimensions changed for {rel_path}: "
                f"{report.width}x{report.height} != {original.width}x{original.height}"
            )
        reports[rel_path] = report
    return reports


def _apply_local_replacement(
    episode: Episode,
    stage_dir: Path,
    backup_dir: Path,
    batch_id: str,
    worker_result: dict[str, Any],
) -> dict[str, Any]:
    with episode_publication_lock(episode.path, blocking=False):
        return _apply_local_replacement_locked(
            episode,
            stage_dir,
            backup_dir,
            batch_id,
            worker_result,
        )


def _apply_local_replacement_locked(
    episode: Episode,
    stage_dir: Path,
    backup_dir: Path,
    batch_id: str,
    worker_result: dict[str, Any],
) -> dict[str, Any]:
    reports = _validate_staged_videos(episode, stage_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    records = []
    installed: list[tuple[Path, Path | None]] = []
    prepared: list[tuple[Path, Path, Path | None]] = []
    _write_json_atomic(
        episode.path / RERENDER_META_NAME,
        {
            "format": "handson.wrist_rerender",
            "version": RERENDER_VERSION,
            "status": "replacing",
            "batch_id": batch_id,
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
            "episode": episode.path.name,
            "task": episode.task,
            "detail": (
                "wrist videos are being replaced; cloud publication must "
                "remain blocked until completed metadata is committed"
            ),
        },
    )
    try:
        for rel_path in WRIST_VIDEO_PATHS:
            source = episode.path / rel_path
            staged = stage_dir / rel_path
            backup = backup_dir / rel_path
            old_sha256 = _sha256_file(source) if source.is_file() else ""
            backup_path: Path | None = None
            if source.is_file():
                backup.parent.mkdir(parents=True, exist_ok=True)
                if backup.exists() and _sha256_file(backup) != old_sha256:
                    backup = backup.with_name(
                        f"{backup.stem}.{dt.datetime.now().strftime('%H%M%S%f')}{backup.suffix}"
                    )
                if not backup.exists():
                    shutil.copy2(source, backup)
                backup_path = backup
            temp = source.with_name(f".{source.name}.{batch_id}.tmp")
            temp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, temp)
            prepared.append((temp, source, backup_path))
            records.append(
                {
                    "path": rel_path,
                    "old_sha256": old_sha256,
                    "new_sha256": _sha256_file(staged),
                    "bytes": staged.stat().st_size,
                    "frame_count": reports[rel_path].frame_count,
                    "width": reports[rel_path].width,
                    "height": reports[rel_path].height,
                    "backup_path": str(backup_path) if backup_path else "",
                }
            )
        cloud_meta_path = episode.path / "cloud_upload_meta.json"
        cloud_meta = _read_json(cloud_meta_path)
        if cloud_meta:
            cloud_meta.update(
                {
                    "status": "stale_local_artifacts",
                    "cloud_ack": "stale",
                    "local_artifacts_changed_at": _now_iso(),
                    "local_artifacts_change_source": (
                        "legacy_wrist_rerender"
                    ),
                }
            )
            cloud_meta.pop("cloud_ack_at", None)
            _write_json_atomic(cloud_meta_path, cloud_meta)
        for temp, destination, backup in prepared:
            os.replace(temp, destination)
            installed.append((destination, backup))
    except Exception:
        for destination, backup in reversed(installed):
            if backup is None:
                destination.unlink(missing_ok=True)
            else:
                shutil.copy2(backup, destination)
        for temp, _, _ in prepared:
            temp.unlink(missing_ok=True)
        raise

    meta = {
        "format": "handson.wrist_rerender",
        "version": RERENDER_VERSION,
        "status": "replaced_locally",
        "batch_id": batch_id,
        "updated_at": _now_iso(),
        "episode": episode.path.name,
        "task": episode.task,
        "method": "isaaclab_state_guided_exact",
        "camera_fix": "latest_wrist_mount_pose_before_explicit_render",
        "software_revision": _git_revision(),
        "source_replay_zarr": str(episode.path / "replay.zarr"),
        "stage_dir": str(stage_dir),
        "camera_sources": worker_result.get("camera_sources") or {},
        "videos": records,
    }
    _write_json_atomic(episode.path / RERENDER_META_NAME, meta)
    arm_status_path = episode.path / "deferred_arm_replay_status.json"
    arm_status = _read_json(arm_status_path)
    if str(arm_status.get("status") or "").strip().lower() == "complete":
        arm_status.update(
            {
                "artifact_proof": capture_deferred_arm_replay_artifacts(
                    episode.path
                ),
                "artifact_validation_passed": True,
                "artifact_proof_refreshed_at": _now_iso(),
                "artifact_proof_refreshed_by": "legacy_wrist_rerender",
                "updated_at": _now_iso(),
            }
        )
        _write_json_atomic(arm_status_path, arm_status)
    return meta


def _verified_rerender_meta(episode: Episode) -> dict[str, Any]:
    meta = _read_json(episode.path / RERENDER_META_NAME)
    videos = meta.get("videos")
    if meta.get("status") != "replaced_locally" or not isinstance(videos, list):
        raise RuntimeError(f"{episode.path.name} has no completed local wrist replacement")
    by_path = {str(item.get("path")): item for item in videos if isinstance(item, dict)}
    for rel_path in WRIST_VIDEO_PATHS:
        local_path = episode.path / rel_path
        item = by_path.get(rel_path)
        if item is None or not local_path.is_file():
            raise RuntimeError(f"rerender metadata is missing {rel_path}")
        if _sha256_file(local_path) != str(item.get("new_sha256") or ""):
            raise RuntimeError(f"local {rel_path} changed after rerender")
    return meta


def _join_remote_key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part and part.strip("/"))


def _cloud_remote_prefix(
    episode: Episode, demos_root: Path, old_meta: dict[str, Any], uploader
) -> str:
    old_prefix = str(old_meta.get("remote_prefix") or "").strip("/")
    if old_prefix:
        return old_prefix
    try:
        relative = episode.path.relative_to(demos_root.resolve()).as_posix()
    except ValueError:
        relative = episode.path.name
    return _join_remote_key(uploader.prefix, relative)


def _verified_put(uploader, local_path: Path, remote_key: str) -> dict[str, Any]:
    result = uploader.put_file(local_path, remote_key)
    status = int(result.get("status") or 0)
    etag = str(result.get("etag") or "").strip().strip('"').lower()
    if not result.get("success") or not (200 <= status < 300):
        raise RuntimeError(
            f"{remote_key}: HTTP {result.get('status')} {str(result.get('detail') or '')[:500]}"
        )
    if not etag:
        raise RuntimeError(f"{remote_key}: KS3 returned 2xx without ETag")
    local_md5 = _md5_file(local_path)
    if re.fullmatch(r"[0-9a-f]{32}", etag) and etag != local_md5:
        raise RuntimeError(f"{remote_key}: ETag does not match local MD5")
    print(f"[cloud OK] HTTP {status} ETag={etag} {remote_key}")
    return {
        "path": local_path,
        "key": remote_key,
        "bytes": int(result.get("bytes") or local_path.stat().st_size),
        "etag": str(result.get("etag") or ""),
        "sha256": _sha256_file(local_path),
    }


def _upsert_file_entry(entries: list[dict[str, Any]], uploaded: dict[str, Any], rel_path: str) -> None:
    new_entry = {
        "path": rel_path,
        "key": uploaded["key"],
        "bytes": uploaded["bytes"],
        "etag": uploaded["etag"],
        "sha256": uploaded["sha256"],
    }
    for index, item in enumerate(entries):
        if str(item.get("path") or "") == rel_path:
            entries[index] = new_entry
            return
    entries.append(new_entry)


def _cloud_files_for_full_upload(episode: Episode) -> list[Path]:
    excluded = {"cloud_upload_meta.json", UPLOAD_FAILURE_NAME}
    return [
        path
        for path in sorted(episode.path.rglob("*"))
        if path.is_file() and path.name not in excluded and ".partial" not in path.name
    ]


def _upload_cloud_replacement(
    episode: Episode,
    demos_root: Path,
    batch_id: str,
    *,
    force_full: bool = False,
) -> dict[str, Any]:
    with episode_publication_lock(episode.path, blocking=False):
        return _upload_cloud_replacement_locked(
            episode,
            demos_root,
            batch_id,
            force_full=force_full,
        )


def _upload_cloud_replacement_locked(
    episode: Episode,
    demos_root: Path,
    batch_id: str,
    *,
    force_full: bool = False,
) -> dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from collection_ui.server import (
        KS3_UPLOAD_PROTOCOL,
        Ks3Uploader,
        cloud_upload_confirmed,
        postprocess_upload_blocker,
    )

    rerender_meta = _verified_rerender_meta(episode)
    blocker = postprocess_upload_blocker(episode.path)
    if blocker is not None:
        raise RuntimeError(
            "Cloud replacement is blocked by the collection upload gate: "
            f"{blocker.get('detail') or blocker.get('code') or blocker}"
        )
    cloud_meta_path = episode.path / "cloud_upload_meta.json"
    old_meta = _read_json(cloud_meta_path)
    uploader = Ks3Uploader.from_env()
    if not uploader.configured():
        raise RuntimeError(f"KS3 config missing: {', '.join(uploader.missing_config())}")
    old_bucket = str(old_meta.get("bucket") or "").strip()
    if old_bucket and old_bucket != uploader.bucket:
        raise RuntimeError(
            f"old bucket {old_bucket!r} differs from configured bucket {uploader.bucket!r}"
        )
    remote_prefix = _cloud_remote_prefix(episode, demos_root, old_meta, uploader)
    old_confirmed = cloud_upload_confirmed(old_meta)
    full_upload = force_full or not old_confirmed
    started = time.time()

    if full_upload:
        entries: list[dict[str, Any]] = []
        local_files = _cloud_files_for_full_upload(episode)
    else:
        entries = [dict(item) for item in old_meta.get("files", []) if isinstance(item, dict)]
        local_files = [episode.path / rel_path for rel_path in WRIST_VIDEO_PATHS]
        local_files.append(episode.path / RERENDER_META_NAME)

    for local_path in local_files:
        rel_path = local_path.relative_to(episode.path).as_posix()
        uploaded = _verified_put(
            uploader, local_path, _join_remote_key(remote_prefix, rel_path)
        )
        _upsert_file_entry(entries, uploaded, rel_path)

    entries.sort(key=lambda item: str(item.get("path") or ""))
    total_bytes = sum(int(item.get("bytes") or 0) for item in entries)
    meta = dict(old_meta) if old_confirmed and not full_upload else {}
    meta.pop("metadata_upload", None)
    meta.pop("cloud_ack_at", None)
    meta.update(
        {
            "status": "uploaded",
            "uploaded_at": _now_iso(),
            "bucket": uploader.bucket,
            "endpoint": uploader.endpoint_host,
            "bucket_domain": uploader.host,
            "region": uploader.region,
            "remote_prefix": remote_prefix,
            "remote_uri": f"ks3://{uploader.bucket}/{remote_prefix}/",
            "file_count": len(entries) + 1,
            "uploaded_files": len(entries) + 1,
            "bytes": total_bytes,
            "bytes_human": _human_bytes(total_bytes),
            "duration_s": round(time.time() - started, 2),
            "files": entries,
            "source": RERENDER_SOURCE,
            "upload_protocol": KS3_UPLOAD_PROTOCOL,
            "upload_verified_by": "ks3_put_object_2xx",
            "cloud_ack": "ok",
            "cloud_ack_source": "ks3_put_object_2xx",
            "cloud_ack_required": True,
            "wrist_rerender": {
                "batch_id": batch_id,
                "updated_at": _now_iso(),
                "mode": "full_episode" if full_upload else "same_key_wrist_overwrite",
                "method": rerender_meta.get("method"),
                "videos": rerender_meta.get("videos"),
            },
        }
    )
    if isinstance(old_meta.get("review_confirmation"), dict):
        meta["review_confirmation"] = old_meta["review_confirmation"]

    meta_key = _join_remote_key(remote_prefix, cloud_meta_path.name)
    meta_result = uploader.put_json(meta, meta_key)
    meta_status = int(meta_result.get("status") or 0)
    meta_etag = str(meta_result.get("etag") or "").strip()
    if not meta_result.get("success") or not (200 <= meta_status < 300) or not meta_etag:
        raise RuntimeError(
            f"metadata commit failed: HTTP {meta_result.get('status')} "
            f"{str(meta_result.get('detail') or '')[:500]}"
        )
    print(f"[cloud OK] HTTP {meta_status} ETag={meta_etag} {meta_key}")
    meta["metadata_upload"] = {
        "status": meta_status,
        "key": meta_key,
        "etag": meta_etag,
        "bytes": int(meta_result.get("bytes") or 0),
    }
    meta["cloud_ack_at"] = _now_iso()
    _write_json_atomic(cloud_meta_path, meta)
    (episode.path / UPLOAD_FAILURE_NAME).unlink(missing_ok=True)
    return meta


def _record_upload_failure(episode: Episode, batch_id: str, exc: Exception) -> None:
    _write_json_atomic(
        episode.path / UPLOAD_FAILURE_NAME,
        {
            "status": "failed",
            "batch_id": batch_id,
            "failed_at": _now_iso(),
            "episode": episode.path.name,
            "error": str(exc),
            "retry_safe": True,
            "cloud_upload_meta_unchanged": True,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos-root", default=str(DEFAULT_DEMOS_ROOT))
    parser.add_argument("--staging-root", default=str(DEFAULT_STAGING_ROOT))
    parser.add_argument("--backup-root", default=str(DEFAULT_BACKUP_ROOT))
    parser.add_argument("--isaac-python", default=str(DEFAULT_ISAAC_PYTHON))
    parser.add_argument("--task", action="append", default=[], help="Exact task name; repeatable")
    parser.add_argument("--episode", action="append", default=[], help="Episode directory name; repeatable")
    parser.add_argument("--collector", default="")
    parser.add_argument("--from-time", default="", help="ISO time or YYYY-MM-DD, Asia/Shanghai if naive")
    parser.add_argument("--to-time", default="", help="ISO time or YYYY-MM-DD, Asia/Shanghai if naive")
    parser.add_argument("--scan-stride", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="Process all selected new-engine episodes")
    parser.add_argument("--rerender", action="store_true", help="Render corrected wrist videos to staging")
    parser.add_argument("--apply-local", action="store_true", help="Back up and replace local wrist videos")
    parser.add_argument("--upload-cloud", action="store_true", help="PUT corrected videos to KS3")
    parser.add_argument(
        "--full-cloud-reupload",
        action="store_true",
        help="Upload every episode file even when the old cloud marker is trusted",
    )
    parser.add_argument("--yes", action="store_true", help="Confirm local replacement/cloud upload")
    parser.add_argument("--allow-concurrent-isaac", action="store_true")
    parser.add_argument(
        "--parallel-tasks",
        type=int,
        default=1,
        help="Number of task-specific Isaac workers to run concurrently",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--worker-manifest", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-results", default="", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.worker_manifest:
        if not args.worker_results:
            raise SystemExit("--worker-results is required in worker mode")
        return _run_worker(Path(args.worker_manifest), Path(args.worker_results))
    if args.scan_stride <= 0:
        raise SystemExit("--scan-stride must be positive")
    if args.parallel_tasks <= 0:
        raise SystemExit("--parallel-tasks must be positive")
    if args.apply_local and not args.rerender:
        raise SystemExit("--apply-local requires --rerender")
    if args.rerender and args.upload_cloud and not args.apply_local:
        raise SystemExit("--rerender --upload-cloud also requires --apply-local")
    if (args.apply_local or args.upload_cloud) and not args.yes:
        raise SystemExit("local replacement/cloud upload requires --yes")

    demos_root = Path(args.demos_root).expanduser().resolve()
    staging_root = Path(args.staging_root).expanduser().resolve()
    backup_root = Path(args.backup_root).expanduser().resolve()
    if staging_root == demos_root or demos_root in staging_root.parents:
        raise SystemExit("--staging-root must be outside --demos-root")
    if backup_root == demos_root or demos_root in backup_root.parents:
        raise SystemExit("--backup-root must be outside --demos-root")
    episodes = _filter_episodes(discover_episodes(demos_root), args)
    if not episodes:
        print("No matching IsaacLab episodes found.")
        return 0
    candidates, inspections = _scan_candidates(episodes, args)
    if not candidates:
        print(f"Scanned {len(episodes)} episode(s); no wrist-camera issue found.")
        return 0
    print(f"Selected {len(candidates)} of {len(episodes)} scanned episode(s).")

    batch_id = args.batch_id.strip() or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_dir = staging_root / batch_id
    backup_batch_dir = backup_root / batch_id
    _write_json_atomic(
        batch_dir / "scan_manifest.json",
        {
            "version": RERENDER_VERSION,
            "batch_id": batch_id,
            "created_at": _now_iso(),
            "demos_root": str(demos_root),
            "episodes": [asdict(item) | {"path": str(item.path)} for item in candidates],
            "inspections": {
                key: [asdict(report) for report in reports]
                for key, reports in inspections.items()
                if any(Path(key) == episode.path for episode in candidates)
            },
        },
    )

    worker_results: dict[str, dict[str, Any]] = {}
    if args.rerender:
        recorders = _active_recorders()
        if recorders and not args.allow_concurrent_isaac:
            details = "\n".join(f"  PID {pid}: {command}" for pid, command in recorders)
            raise SystemExit(
                "Active collection detected; stop it before rerendering or pass "
                f"--allow-concurrent-isaac explicitly:\n{details}"
            )
        worker_results = _launch_workers(candidates, args, batch_dir)

    applied: list[Episode] = []
    failures = 0
    if args.apply_local:
        for episode in candidates:
            result = worker_results.get(str(episode.path))
            if not result or not result.get("success"):
                failures += 1
                print(f"[skip local] rerender failed or missing: {episode.path.name}")
                continue
            stage_dir = Path(result["stage_dir"])
            try:
                _apply_local_replacement(
                    episode,
                    stage_dir,
                    backup_batch_dir / episode.task / episode.path.name,
                    batch_id,
                    result,
                )
            except Exception as exc:  # noqa: BLE001 - continue independent episodes.
                failures += 1
                print(f"[local failed] {episode.path.name}: {exc}")
                continue
            applied.append(episode)
            print(f"[local OK] {episode.path.name}")

    cloud_targets: list[Episode] = []
    if args.upload_cloud:
        cloud_targets = applied if args.rerender else candidates
        for episode in cloud_targets:
            try:
                _upload_cloud_replacement(
                    episode,
                    demos_root,
                    batch_id,
                    force_full=args.full_cloud_reupload,
                )
            except Exception as exc:  # noqa: BLE001 - preserve retry information.
                failures += 1
                _record_upload_failure(episode, batch_id, exc)
                print(f"[cloud failed] {episode.path.name}: {exc}")
                continue
            print(f"[cloud committed] {episode.path.name}")

    if not args.rerender and not args.upload_cloud:
        print(f"Dry scan only. Manifest: {batch_dir / 'scan_manifest.json'}")
    elif args.rerender and not args.apply_local:
        print(f"Rendered files remain in staging: {batch_dir}")
    print(
        f"Done: selected={len(candidates)} local_applied={len(applied)} "
        f"cloud_attempted={len(cloud_targets)} failures={failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
