#!/usr/bin/env python3
"""Fresh-process recording/replay and three-camera smoke test for YCB04."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import threading
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "handson"))

from isaaclab.app import AppLauncher

from handson.sim.isaaclab.runtime_display import (
    enforce_isaac_viewport_only,
    isaac_kit_args,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("record", "replay"), required=True)
parser.add_argument("--episode_dir", type=Path, required=True)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--seed", type=int, default=17)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not any(
    argument == "--device" or argument.startswith("--device=")
    for argument in sys.argv[1:]
):
    args_cli.device = "cuda:0"
args_cli.enable_cameras = True
args_cli.kit_args = isaac_kit_args(
    args_cli.kit_args,
    headless=args_cli.headless,
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
enforce_isaac_viewport_only(simulation_app, headless=args_cli.headless)


import zarr  # noqa: E402

from handson.data.episode_store import ZarrEpisodeStore  # noqa: E402
from handson.sim.isaaclab.abc_collection_env import (  # noqa: E402
    HandsOnABCIsaacLabEnv,
)


_TASK_NAME = "t41_sort_nested_cups"
_CAMERAS = ("random_camera", "wrist_left", "wrist_right")
_SENSORS = {
    "random_camera": ("head_camera", "ego"),
    "wrist_left": ("wrist_left_camera", "wrist_left"),
    "wrist_right": ("wrist_right_camera", "wrist_right"),
}


def _camera_frames(observation: dict[str, object]) -> dict[str, np.ndarray]:
    images = observation.get("images")
    if not isinstance(images, dict):
        raise RuntimeError("YCB04 observation is missing the images mapping")
    if set(images) != set(_CAMERAS):
        raise RuntimeError(
            f"YCB04 expected exactly {_CAMERAS}, got {tuple(images)}"
        )
    frames: dict[str, np.ndarray] = {}
    for key in _CAMERAS:
        frame = np.asarray(images.get(key))
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise RuntimeError(
                f"invalid {key} frame: shape={frame.shape} dtype={frame.dtype}"
            )
        if int(np.ptp(frame)) <= 0:
            raise RuntimeError(f"{key} frame is constant")
        frames[key] = frame.copy()
    return frames


def _assert_live_camera_frames(
    env: HandsOnABCIsaacLabEnv,
    frames: dict[str, np.ndarray],
) -> None:
    for key, (sensor_name, fallback_key) in _SENSORS.items():
        if sensor_name not in env._cameras:
            raise RuntimeError(f"YCB04 did not create {sensor_name}")
        frame = frames[key]
        fallback = env._fallback_camera_frame(
            fallback_key,
            height=frame.shape[0],
            width=frame.shape[1],
        )
        if np.array_equal(frame, fallback):
            raise RuntimeError(f"YCB04 {sensor_name} returned a fallback frame")


def _write_initial_state(root: zarr.Group, state: dict[str, object]) -> None:
    group = root.require_group("initial_state")
    for key, value in state.items():
        array = np.asarray(value)
        if array.dtype == object:
            continue
        group.array(
            key,
            data=np.ascontiguousarray(array),
            compressor=None,
            overwrite=True,
        )


def _read_initial_state(root: zarr.Group) -> dict[str, np.ndarray]:
    group = root["initial_state"]
    return {key: np.asarray(group[key]) for key in group.array_keys()}


def _quaternion_angle(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(4)
    second = np.asarray(second, dtype=np.float64).reshape(4)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1.0e-12:
        return math.inf
    cosine = float(np.clip(abs(np.dot(first, second)) / denominator, 0.0, 1.0))
    return float(2.0 * math.acos(cosine))


def _make_environment() -> HandsOnABCIsaacLabEnv:
    return HandsOnABCIsaacLabEnv(
        _TASK_NAME,
        seed=args_cli.seed,
        control_dt=1.0 / 30.0,
        physics_dt=1.0 / 120.0,
        time_limit=120.0,
        render_mode="rgb_array",
        image_obs=True,
        randomize=True,
        headless=args_cli.headless,
        isaac_device=args_cli.device,
        existing_simulation_app=simulation_app,
        existing_app_launcher=app_launcher,
    )


def _record(env: HandsOnABCIsaacLabEnv, episode_dir: Path) -> dict[str, object]:
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive")
    episode_dir.mkdir(parents=True, exist_ok=True)
    replay_path = episode_dir / "replay.zarr"
    if replay_path.exists():
        raise FileExistsError(f"refusing to overwrite {replay_path}")

    observation, _reset_info = env.reset(seed=args_cli.seed)
    initial_state = {
        key: np.asarray(value).copy()
        for key, value in observation["state"].items()
    }
    initial_frames = _camera_frames(observation)
    _assert_live_camera_frames(env, initial_frames)
    actions: list[np.ndarray] = []
    cup_poses: list[np.ndarray] = []
    cup_velocities: list[np.ndarray] = []
    successes: list[bool] = []
    frames: dict[str, list[np.ndarray]] = {
        key: [] for key in _CAMERAS
    }
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    for _ in range(args_cli.steps):
        observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            raise RuntimeError(
                "YCB04 ended during recording smoke: "
                f"terminated={terminated} truncated={truncated} info={info}"
            )
        current_frames = _camera_frames(observation)
        _assert_live_camera_frames(env, current_frames)
        actions.append(action.copy())
        cup_poses.append(np.asarray(observation["state"]["cup_poses"]).copy())
        cup_velocities.append(
            np.asarray(observation["state"]["cup_velocities"]).copy()
        )
        successes.append(bool(info.get("succeed", False)))
        for key in _CAMERAS:
            frames[key].append(current_frames[key])

    store = zarr.DirectoryStore(str(replay_path))
    episode_store = ZarrEpisodeStore.create_empty(storage=store)
    episode_store.root.attrs.update(
        {
            "task_name": _TASK_NAME,
            "task_id": "YCB04",
            "runtime": "Isaac Sim 5.1 / IsaacLab 2.3.2",
            "physics_hz": 120,
            "control_hz": 30,
            "smoke_only": True,
            "seed": int(args_cli.seed),
            "initial_camera_shapes": {
                key: list(frame.shape) for key, frame in initial_frames.items()
            },
        }
    )
    _write_initial_state(episode_store.root, initial_state)
    episode_store.append_episode(
        {
            "action": np.stack(actions),
            "cup_poses": np.stack(cup_poses),
            "cup_velocities": np.stack(cup_velocities),
            "succeed": np.asarray(successes, dtype=np.uint8),
            "main_rgb": np.stack(frames["random_camera"]),
            "wrist_left_rgb": np.stack(frames["wrist_left"]),
            "wrist_right_rgb": np.stack(frames["wrist_right"]),
        },
        compressors="disk",
    )
    result = {
        "mode": "record",
        "pass": True,
        "episode_dir": str(episode_dir),
        "steps": len(actions),
        "camera_shapes": {
            key: list(values[0].shape) for key, values in frames.items()
        },
        "reset_initial_nested_valid": bool(env._ycb04_initial_nested_valid),
    }
    (episode_dir / "record_smoke.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _replay(env: HandsOnABCIsaacLabEnv, episode_dir: Path) -> dict[str, object]:
    replay_path = episode_dir / "replay.zarr"
    root = zarr.open(str(replay_path), mode="r")
    if str(root.attrs.get("task_name")) != _TASK_NAME:
        raise RuntimeError("replay.zarr is not a YCB04 episode")
    actions = np.asarray(root["data/action"])
    expected_poses = np.asarray(root["data/cup_poses"])
    expected_success = np.asarray(root["data/succeed"], dtype=np.uint8)
    initial_state = _read_initial_state(root)

    env.reset(seed=int(root.attrs["seed"]))
    env.restore_recorded_initial_state(initial_state)
    max_position_error = 0.0
    max_orientation_error = 0.0
    replay_success: list[bool] = []
    final_frames: dict[str, np.ndarray] = {}
    for step_index, action in enumerate(actions):
        observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            raise RuntimeError(
                "YCB04 ended during replay smoke: "
                f"step={step_index} terminated={terminated} "
                f"truncated={truncated} info={info}"
            )
        replay_poses = np.asarray(observation["state"]["cup_poses"])
        max_position_error = max(
            max_position_error,
            float(np.max(np.linalg.norm(replay_poses[:, :3] - expected_poses[step_index, :, :3], axis=1))),
        )
        max_orientation_error = max(
            max_orientation_error,
            max(
                _quaternion_angle(replay_poses[index, 3:7], expected_poses[step_index, index, 3:7])
                for index in range(3)
            ),
        )
        replay_success.append(bool(info.get("succeed", False)))
        final_frames = _camera_frames(observation)
        _assert_live_camera_frames(env, final_frames)

    outcome_match = np.array_equal(
        np.asarray(replay_success, dtype=np.uint8),
        expected_success,
    )
    passed = bool(
        max_position_error <= 0.003
        and max_orientation_error <= math.radians(3.0)
        and outcome_match
        and set(final_frames) == set(_CAMERAS)
    )
    result = {
        "mode": "replay",
        "pass": passed,
        "episode_dir": str(episode_dir),
        "steps": int(actions.shape[0]),
        "max_cup_position_error_m": max_position_error,
        "max_cup_orientation_error_rad": max_orientation_error,
        "outcome_match": outcome_match,
        "camera_shapes": {
            key: list(frame.shape) for key, frame in final_frames.items()
        },
    }
    (episode_dir / "replay_smoke.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(f"YCB04 fresh-process replay mismatch: {result}")
    return result


def main() -> int:
    env = _make_environment()
    completed = False
    try:
        episode_dir = args_cli.episode_dir.expanduser().resolve()
        result = (
            _record(env, episode_dir)
            if args_cli.mode == "record"
            else _replay(env, episode_dir)
        )
        print(f"YCB04_COLLECTION_SMOKE={json.dumps(result, sort_keys=True)}", flush=True)
        completed = True
        return 0
    finally:
        close_guard = threading.Timer(
            5.0,
            lambda: os._exit(0 if completed else 1),
        )
        close_guard.daemon = True
        close_guard.start()
        try:
            env.close()
        finally:
            close_guard.cancel()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        shutdown_guard = threading.Timer(5.0, lambda: os._exit(exit_code))
        shutdown_guard.daemon = True
        shutdown_guard.start()
        try:
            simulation_app.close(wait_for_replicator=False)
        finally:
            shutdown_guard.cancel()
    raise SystemExit(exit_code)
