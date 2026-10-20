#!/usr/bin/env python3
"""Drive the real collection UI through one BC005 validation episode.

This is a platform canary, not an environment shortcut: it starts
``collection_ui/server.py``, selects the task through its HTTP API, and sends
the same tracker/hand UDP packets consumed by the normal ROKAE+Wuji teleop
wrapper.  The trajectory is synchronized to the recorder's live-preview
updates so slow headless rendering does not compress it into too few control
steps.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "handson"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from handson.sim.isaaclab.loaded_tray_gate_scene import (
    GATE_PLANE_X,
    LEFT_GRIP_LOCAL_POSITION,
    RIGHT_GRIP_LOCAL_POSITION,
    TRAY_SCRIPTED_PLACED_CENTER_X,
    TRAY_START_POSITION,
    tray_local_position_to_world,
)


TASK_NAME = "loaded_tray_through_narrow_gate"
TASK_ID = "B320-BC005"
TASK_PROMPT = (
    "Use the left hand on the left long side rim and the right hand on the "
    "right long side rim. Lift the loaded tray narrow-face first, carry it "
    "and both payloads through "
    "the narrow wooden frame, then hold it level and stable over the green target."
)

RESET_RIGHT_POSITION = np.asarray(
    [-0.3715001940727234, -0.239500030875206, 0.9975000023841858]
)
RESET_LEFT_POSITION = np.asarray(
    [-0.37150004506111145, 0.23950007557868958, 0.997499942779541]
)
RESET_RIGHT_ROTATION = np.asarray(
    [
        [-8.429368580920027e-08, 8.390895764625672e-08, 0.9999999999999929],
        [3.8507004024527997e-07, 0.9999999999999224, -8.39089251872778e-08],
        [-0.9999999999999223, 3.850700331722847e-07, -8.429371822771259e-08],
    ],
    dtype=np.float64,
)
RESET_LEFT_ROTATION = np.asarray(
    [
        [8.42936873635125e-08, -2.9103790648348637e-07, 0.9999999999999539],
        [1.0135988174921598e-07, 0.9999999999999525, 2.9103789793948774e-07],
        [-0.9999999999999912, 1.0135985721655378e-07, 8.429371689544496e-08],
    ],
    dtype=np.float64,
)

# Public-action waypoints use the same +90-degree tray transform as runtime.
# Both hands stay side by side on the long rims while the short face leads.
PALM_ABOVE_GRIP_M = 0.060
LIFT_HEIGHT_M = 0.070


def _palm_world_target(
    tray_center_x: float,
    tray_lift_m: float,
    local_grip: tuple[float, float, float],
) -> np.ndarray:
    tray_position = (
        float(tray_center_x),
        float(TRAY_START_POSITION[1]),
        float(TRAY_START_POSITION[2]) + float(tray_lift_m),
    )
    target = np.asarray(
        tray_local_position_to_world(tray_position, local_grip),
        dtype=np.float64,
    )
    target[2] += PALM_ABOVE_GRIP_M
    return target


APPROACH_RIGHT = _palm_world_target(
    TRAY_START_POSITION[0], 0.0, RIGHT_GRIP_LOCAL_POSITION
) - RESET_RIGHT_POSITION
APPROACH_LEFT = _palm_world_target(
    TRAY_START_POSITION[0], 0.0, LEFT_GRIP_LOCAL_POSITION
) - RESET_LEFT_POSITION
LIFTED_RIGHT = _palm_world_target(
    TRAY_START_POSITION[0], LIFT_HEIGHT_M, RIGHT_GRIP_LOCAL_POSITION
) - RESET_RIGHT_POSITION
LIFTED_LEFT = _palm_world_target(
    TRAY_START_POSITION[0], LIFT_HEIGHT_M, LEFT_GRIP_LOCAL_POSITION
) - RESET_LEFT_POSITION

FINAL_TRAY_CENTER_X = TRAY_SCRIPTED_PLACED_CENTER_X
CROSSING_RIGHT = _palm_world_target(
    GATE_PLANE_X, LIFT_HEIGHT_M, RIGHT_GRIP_LOCAL_POSITION
) - RESET_RIGHT_POSITION
CROSSING_LEFT = _palm_world_target(
    GATE_PLANE_X, LIFT_HEIGHT_M, LEFT_GRIP_LOCAL_POSITION
) - RESET_LEFT_POSITION
FINAL_RIGHT = _palm_world_target(
    FINAL_TRAY_CENTER_X, LIFT_HEIGHT_M, RIGHT_GRIP_LOCAL_POSITION
) - RESET_RIGHT_POSITION
FINAL_LEFT = _palm_world_target(
    FINAL_TRAY_CENTER_X, LIFT_HEIGHT_M, LEFT_GRIP_LOCAL_POSITION
) - RESET_LEFT_POSITION

POWER_GRIP = 0.8 * np.asarray(
    [
        0.60, 0.10, 0.84, 0.70,
        0.68, 0.00, 0.90, 0.78,
        0.70, 0.00, 0.92, 0.80,
        0.68, 0.00, 0.90, 0.78,
        0.64, 0.00, 0.86, 0.74,
    ],
    dtype=np.float64,
)
OPEN_HAND = np.zeros(20, dtype=np.float64)
FINALIZATION_GRACE_SECONDS = 90.0
EXPECTED_COMPLETION_PREVIEW_ERROR = (
    "TimeoutError: live preview stopped updating for 45 seconds"
)


@dataclass(frozen=True)
class Keyframe:
    name: str
    frames: int
    right_delta: np.ndarray
    left_delta: np.ndarray
    right_roll_deg: float
    left_roll_deg: float
    hand: np.ndarray


KEYFRAMES = (
    Keyframe("approach", 60, APPROACH_RIGHT, APPROACH_LEFT, 0.0, 0.0, OPEN_HAND),
    Keyframe("grasp", 85, APPROACH_RIGHT, APPROACH_LEFT, 0.0, 0.0, POWER_GRIP),
    Keyframe("lift", 90, LIFTED_RIGHT, LIFTED_LEFT, 0.0, 0.0, POWER_GRIP),
    Keyframe("cross_gate", 180, CROSSING_RIGHT, CROSSING_LEFT, 0.0, 0.0, POWER_GRIP),
    Keyframe("crossing_settle", 80, CROSSING_RIGHT, CROSSING_LEFT, 0.0, 0.0, POWER_GRIP),
    Keyframe("clear_to_target", 450, FINAL_RIGHT, FINAL_LEFT, 0.0, 0.0, POWER_GRIP),
)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _rotation_x(degrees: float) -> np.ndarray:
    angle = np.deg2rad(float(degrees))
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _tracker_rotation(base_rotation: np.ndarray, local_roll_deg: float) -> np.ndarray:
    # tracker_start is identity.  The bridge mapping computes
    # target_R = tracker_now_R @ base_R, so this yields base_R @ Rx @ base_R.T.
    return base_rotation @ _rotation_x(local_roll_deg) @ base_rotation.T


def _pose_packet(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    packet = np.empty((3, 4), dtype=np.float64)
    packet[:, :3] = rotation
    packet[:, 3] = translation
    return packet


class UdpPump:
    def __init__(self, host: str = "127.0.0.1") -> None:
        self.host = host
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._right_rotation = np.eye(3, dtype=np.float64)
        self._left_rotation = np.eye(3, dtype=np.float64)
        self._right_translation = np.zeros(3, dtype=np.float64)
        self._left_translation = np.zeros(3, dtype=np.float64)
        self._right_hand = OPEN_HAND.copy()
        self._left_hand = OPEN_HAND.copy()
        self._thread = threading.Thread(target=self._run, name="loaded-tray-udp-pump", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_targets(
        self,
        right_translation: np.ndarray,
        left_translation: np.ndarray,
        right_roll_deg: float,
        left_roll_deg: float,
        right_hand: np.ndarray,
        left_hand: np.ndarray,
    ) -> None:
        with self._lock:
            self._right_translation = np.asarray(right_translation, dtype=np.float64).copy()
            self._left_translation = np.asarray(left_translation, dtype=np.float64).copy()
            self._right_rotation = _tracker_rotation(RESET_RIGHT_ROTATION, right_roll_deg)
            self._left_rotation = _tracker_rotation(RESET_LEFT_ROTATION, left_roll_deg)
            self._right_hand = np.asarray(right_hand, dtype=np.float64).copy()
            self._left_hand = np.asarray(left_hand, dtype=np.float64).copy()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        tracker_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        right_hand_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        left_hand_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            while not self._stop.is_set():
                with self._lock:
                    tracker = np.concatenate(
                        [
                            _pose_packet(self._right_rotation, self._right_translation).reshape(-1),
                            _pose_packet(self._left_rotation, self._left_translation).reshape(-1),
                        ]
                    )
                    right_hand = self._right_hand.copy()
                    left_hand = self._left_hand.copy()
                tracker_socket.sendto(tracker.tobytes(), (self.host, 5012))
                right_hand_socket.sendto(right_hand.tobytes(), (self.host, 5014))
                left_hand_socket.sendto(left_hand.tobytes(), (self.host, 5016))
                self._stop.wait(0.02)
        finally:
            tracker_socket.close()
            right_hand_socket.close()
            left_hand_socket.close()


def _interpolate_keyframes(step: int) -> tuple[str, float, Keyframe, Keyframe]:
    previous = Keyframe("baseline", 0, np.zeros(3), np.zeros(3), 0.0, 0.0, OPEN_HAND)
    elapsed = 0
    for current in KEYFRAMES:
        if step <= elapsed + current.frames:
            alpha = float(np.clip((step - elapsed) / max(1, current.frames), 0.0, 1.0))
            return current.name, alpha, previous, current
        elapsed += current.frames
        previous = current
    return KEYFRAMES[-1].name, 1.0, KEYFRAMES[-1], KEYFRAMES[-1]


def _lerp(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * np.asarray(start) + alpha * np.asarray(end)


def _wait_for_text(path: Path, needle: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            if needle in path.read_text(encoding="utf-8", errors="replace"):
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {needle!r} in {path}")


def _preview_fingerprint(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_ino, stat.st_mtime_ns, stat.st_size


def _run_step_synchronized_trajectory(
    pump: UdpPump,
    *,
    collection_log: Path,
    preview_path: Path,
    trace_path: Path,
    stop_event: threading.Event,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    _wait_for_text(collection_log, "Teleoperation enabled.", deadline)
    previous_fingerprint = _preview_fingerprint(preview_path)
    preview_updates = 0
    last_phase = ""
    last_update_at = time.monotonic()
    while time.monotonic() < deadline and not stop_event.is_set():
        fingerprint = _preview_fingerprint(preview_path)
        if fingerprint is None or fingerprint == previous_fingerprint:
            if time.monotonic() - last_update_at > 45.0:
                raise TimeoutError("live preview stopped updating for 45 seconds")
            time.sleep(0.01)
            continue
        previous_fingerprint = fingerprint
        last_update_at = time.monotonic()
        preview_updates += 1
        estimated_step = preview_updates * 3
        phase, alpha, previous, current = _interpolate_keyframes(estimated_step)
        right_delta = _lerp(previous.right_delta, current.right_delta, alpha)
        left_delta = _lerp(previous.left_delta, current.left_delta, alpha)
        right_roll = (1.0 - alpha) * previous.right_roll_deg + alpha * current.right_roll_deg
        left_roll = (1.0 - alpha) * previous.left_roll_deg + alpha * current.left_roll_deg
        hand = _lerp(previous.hand, current.hand, alpha)
        pump.set_targets(right_delta, left_delta, right_roll, left_roll, hand, hand)
        if phase != last_phase or preview_updates % 10 == 0:
            _append_jsonl(
                trace_path,
                {
                    "alpha": alpha,
                    "estimated_step": estimated_step,
                    "left_delta": left_delta.tolist(),
                    "left_roll_deg": left_roll,
                    "phase": phase,
                    "preview_updates": preview_updates,
                    "right_delta": right_delta.tolist(),
                    "right_roll_deg": right_roll,
                    "time_monotonic": time.monotonic(),
                },
            )
            last_phase = phase
    if not stop_event.is_set():
        raise TimeoutError("trajectory did not finish before its deadline")


def _request_json(
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(base_url + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object from {path}, got {type(value).__name__}")
    return value


def _wait_for_health(base_url: str, deadline: float) -> dict[str, Any]:
    last_error = ""
    while time.monotonic() < deadline:
        try:
            health = _request_json(base_url, "/api/health", timeout=2.0)
            if health:
                return health
        except (OSError, urllib.error.URLError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise TimeoutError(f"collection UI did not become healthy: {last_error}")


def _pick_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _collection_running(status: dict[str, Any]) -> bool:
    collection = status.get("collection")
    return bool(isinstance(collection, dict) and collection.get("running"))


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _pending_episode_items(out_dir: Path) -> list[dict[str, Any]]:
    """Expose fully sealed platform saves that await physical replay review."""

    pending_root = out_dir / ".physical_replay_pending"
    if not pending_root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for episode_dir in sorted(pending_root.iterdir()):
        if (
            not episode_dir.is_dir()
            or not episode_dir.name.startswith(f"{TASK_NAME}_demo_")
            or not (episode_dir / "replay.zarr").is_dir()
            or not (episode_dir / "DATASET_PHYSX_V3.json").is_file()
        ):
            continue
        manifest = _read_json_object(episode_dir / "replay_manifest.json")
        task = manifest.get("task")
        task = task if isinstance(task, dict) else {}
        if task.get("name") != TASK_NAME:
            continue
        items.append(
            {
                "episode_id": episode_dir.name,
                "path": str(episode_dir.resolve()),
                "publication_state": "physical_replay_pending",
                "task_name": TASK_NAME,
            }
        )
    return items


def _saved_episode_items(
    response: dict[str, Any], out_dir: Path
) -> list[dict[str, Any]]:
    api_items = response.get("items")
    items = list(api_items) if isinstance(api_items, list) else []
    seen = {
        str(item.get("episode_id") or item.get("path") or "")
        for item in items
        if isinstance(item, dict)
    }
    for item in _pending_episode_items(out_dir):
        identity = str(item.get("episode_id") or item.get("path") or "")
        if identity not in seen:
            items.append(item)
            seen.add(identity)
    return items


def _is_expected_completion_preview_error(message: str) -> bool:
    return message == EXPECTED_COMPLETION_PREVIEW_ERROR


def _terminate_server(process: subprocess.Popen[Any], base_url: str) -> None:
    if process.poll() is not None:
        return
    try:
        status = _request_json(base_url, "/api/status", timeout=2.0)
        if _collection_running(status):
            _request_json(base_url, "/api/collection/stop", payload={}, timeout=5.0)
            deadline = time.monotonic() + 45.0
            while time.monotonic() < deadline:
                if not _collection_running(_request_json(base_url, "/api/status", timeout=2.0)):
                    break
                time.sleep(0.5)
    except Exception:
        pass
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--isaacsim-python", required=True)
    parser.add_argument("--collector-id", default="codex_g6_canary")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    evidence_dir = args.evidence_dir.resolve()
    out_dir = args.out_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_path = evidence_dir / "platform_live_preview.png"
    server_log_path = evidence_dir / "platform_server.log"
    trace_path = evidence_dir / "udp_trajectory.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    port = _pick_loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    canary_environment = {
        "HANDSON_BACKGROUND_V3_UPLOAD": "0",
        "HANDSON_COLLECTION_CLEAN_STALE_ON_START": "0",
        "HANDSON_COLLECTION_LIVE_PREVIEW_PATH": str(preview_path),
        "HANDSON_COLLECTION_REQUIRE_TELEOP_READY": "0",
        "HANDSON_COLLECTION_TELEOP_STACK": "disabled",
        "HANDSON_IDLE_MAINTENANCE": "0",
        "HANDSON_ISAACSIM_CACHE_WARMUP": "0",
        "HANDSON_REVIEW_AUTO_UPLOAD": "0",
    }
    environment.update(canary_environment)
    command = [
        args.python,
        str(root / "collection_ui" / "server.py"),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--root", str(root),
        "--python", args.python,
        "--isaacsim-python", args.isaacsim_python,
        "--out-dir", str(out_dir),
    ]
    _json_dump(
        evidence_dir / "platform_invocation.json",
        {"command": command, "environment_overrides": canary_environment},
    )

    pump = UdpPump()
    pump.start()
    stop_trajectory = threading.Event()
    trajectory_error: list[str] = []
    trajectory_thread: threading.Thread | None = None
    server_log = server_log_path.open("wb")
    server = subprocess.Popen(
        command,
        cwd=str(root),
        env=environment,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + args.timeout_seconds
        health = _wait_for_health(base_url, min(deadline, time.monotonic() + 60.0))
        _json_dump(evidence_dir / "api_health.json", health)
        tasks = _request_json(base_url, "/api/tasks")
        _json_dump(evidence_dir / "api_tasks.json", tasks)
        task_items = tasks.get("items") if isinstance(tasks.get("items"), list) else []
        matches = [item for item in task_items if item.get("name") == TASK_NAME]
        if len(matches) != 1:
            raise RuntimeError(f"platform task list contained {len(matches)} {TASK_NAME!r} entries")
        if matches[0].get("prompt") != TASK_PROMPT:
            raise RuntimeError("platform prompt does not match the task package prompt")

        payload = {
            "arm_type": "rokae",
            "collection_note": "G6 isolated HP platform canary; G5 trajectory via normal UDP teleop",
            "collector_id": args.collector_id,
            "hand_type": "wuji1",
            "isaac_arm_render_mode": "live",
            "out_dir": str(out_dir),
            "randomize": False,
            "render_mode": "rgb_array",
            "robot_type": "ROKAE+Wuji",
            "save_depth": False,
            "show_sim_cameras": False,
            "sim_backend": "isaacsim",
            "successes_needed": 1,
            "task_name": TASK_NAME,
            "task_prompt": TASK_PROMPT,
        }
        _json_dump(evidence_dir / "api_start_payload.json", payload)
        started = _request_json(base_url, "/api/collection/start", payload=payload, timeout=30.0)
        _json_dump(evidence_dir / "api_start_response.json", started)
        if not started.get("success"):
            raise RuntimeError(f"platform rejected collection start: {started}")
        session = started.get("session") if isinstance(started.get("session"), dict) else {}
        if session.get("task_name") != TASK_NAME or session.get("sim_backend") != "isaacsim":
            raise RuntimeError(f"platform started the wrong task/backend: {session}")
        if session.get("isaac_arm_render_mode") != "live":
            raise RuntimeError(f"platform did not honor live rendering: {session}")
        collection_log = Path(str(started.get("log_path") or session.get("log_path") or ""))
        if not collection_log.is_absolute():
            raise RuntimeError("platform start response did not provide an absolute collection log")

        def trajectory_worker() -> None:
            try:
                _run_step_synchronized_trajectory(
                    pump,
                    collection_log=collection_log,
                    preview_path=preview_path,
                    trace_path=trace_path,
                    stop_event=stop_trajectory,
                    timeout_s=max(1.0, deadline - time.monotonic()),
                )
            except Exception as exc:  # noqa: BLE001
                trajectory_error.append(f"{type(exc).__name__}: {exc}")

        trajectory_thread = threading.Thread(
            target=trajectory_worker,
            name="loaded-tray-step-synchronized-trajectory",
            daemon=True,
        )
        trajectory_thread.start()

        last_episode_response: dict[str, Any] = {"success": True, "items": []}
        collection_was_running = False
        completion_preview_error_at: float | None = None
        saved_items: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            status = _request_json(base_url, "/api/status", timeout=5.0)
            _append_jsonl(evidence_dir / "api_status.jsonl", status)
            collection_was_running = collection_was_running or _collection_running(status)
            query = urllib.parse.urlencode({"out_dir": str(out_dir)})
            last_episode_response = _request_json(base_url, f"/api/episodes?{query}", timeout=10.0)
            saved_items = _saved_episode_items(last_episode_response, out_dir)
            if saved_items and collection_was_running and not _collection_running(status):
                break
            if trajectory_error:
                if not _is_expected_completion_preview_error(trajectory_error[0]):
                    raise RuntimeError(trajectory_error[0])
                if completion_preview_error_at is None:
                    completion_preview_error_at = time.monotonic()
                if (
                    _collection_running(status)
                    and time.monotonic() - completion_preview_error_at
                    > FINALIZATION_GRACE_SECONDS
                ):
                    raise TimeoutError(
                        "collection stayed active for 90 seconds after live preview stopped"
                    )
            time.sleep(1.0)
        else:
            raise TimeoutError("platform did not finish and save an episode before the deadline")

        stop_trajectory.set()
        if trajectory_thread is not None:
            trajectory_thread.join(timeout=5.0)
        _json_dump(evidence_dir / "api_episodes.json", last_episode_response)
        _json_dump(evidence_dir / "saved_episodes.json", {"items": saved_items})
        if len(saved_items) != 1:
            raise RuntimeError(f"expected exactly one saved episode, got {saved_items}")
        episode = saved_items[0]
        episode_task = str(episode.get("task_name") or episode.get("task") or "")
        if episode_task and episode_task != TASK_NAME:
            raise RuntimeError(f"saved episode reports wrong task: {episode_task}")
        if trajectory_error and not _is_expected_completion_preview_error(
            trajectory_error[0]
        ):
            raise RuntimeError(trajectory_error[0])
        _json_dump(
            evidence_dir / "canary_result.json",
            {
                "episode": episode,
                "out_dir": str(out_dir),
                "platform_session_id": session.get("session_id"),
                "status": "PASS",
                "task_id": TASK_ID,
                "task_name": TASK_NAME,
                "trajectory_warning": trajectory_error[0] if trajectory_error else "",
            },
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        _json_dump(
            evidence_dir / "canary_result.json",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "status": "FAIL",
                "task_id": TASK_ID,
                "task_name": TASK_NAME,
                "trajectory_error": trajectory_error,
            },
        )
        raise
    finally:
        stop_trajectory.set()
        if trajectory_thread is not None:
            trajectory_thread.join(timeout=5.0)
        pump.stop()
        _terminate_server(server, base_url)
        server_log.close()


if __name__ == "__main__":
    sys.exit(main())
