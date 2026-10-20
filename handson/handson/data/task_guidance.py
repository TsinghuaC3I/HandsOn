"""Local, task-scoped hand-route guidance for online arm IK.

Raw guidance episodes deliberately live outside the Git checkout.  Complete
guides are eligible after their physical two-hand route and the existing
simplified-post-IK planner have certified every arm frame.  Hands-only route
guides do not need a task-success terminal transition; when their exact plan
is not fully certifiable, a collision-safe prefix can be published as a
coarse soft hint.
"""

from __future__ import annotations

from collections.abc import Mapping
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from handson.data.simplified_post_ik import (
    load_validated_simplified_arm_plan,
)


GUIDANCE_FORMAT = "handson.task_guidance"
GUIDANCE_VERSION = 1
GUIDANCE_MARKER_NAME = "TASK_GUIDANCE.json"
GUIDANCE_PROFILE_FORMAT = "handson.task_guidance_profile"
GUIDANCE_PROFILE_VERSION = 1
GUIDANCE_PROFILE_NAME = "task_guidance_profile.npz"
GUIDANCE_REMOTE_CACHE_NAME = "task_guidance_remote_profile.npz"
GUIDANCE_PROFILE_POINTS = 256
# A route guide is intentionally a soft hint.  Its arm path is truncated at
# the last collision-safe frame and then resampled onto the same compact
# progress grid as exact guides, while the runtime adds lookahead and an
# endpoint pull.  Keeping one grid lets exact and coarse guides coexist in a
# single shared profile.
GUIDANCE_EXACT_PATH_MODE = "exact_monotonic"
GUIDANCE_COARSE_PATH_MODE = "coarse_endpoint_bias_v1"
GUIDANCE_MAX_SHARED_PROFILE_BYTES = 100 * 1024 * 1024
GUIDANCE_ROOT_ENV = "HANDSON_GUIDANCE_ROOT"
GUIDANCE_ENABLED_ENV = "HANDSON_TASK_GUIDANCE_ENABLED"
GUIDANCE_RECORDING_ENV = "HANDSON_TASK_GUIDANCE_RECORDING"

_SAFE_TASK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
_SAFE_GUIDE_ID = re.compile(r"^[0-9a-f]{20}$")
_PHYSX_PENDING_DIRECTORY = ".physical_replay_pending"
_PHYSX_HISTORY_DIRECTORY = ".physx_session_history"
_PHYSX_SESSION_FORMAT = "handson.physx_session_history"
_PHYSX_SESSION_NAME = "session.json"
_PHYSX_EVENTS_DIRECTORY = "events"
_PHYSX_EVENT_NAME = "event.json"
_PHYSX_FINALIZATION_NAME = "FINALIZED.json"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _repository_root() -> Path:
    # .../repo/handson/handson/data/task_guidance.py -> .../repo
    return Path(__file__).resolve().parents[3]


def default_guidance_root(repository_root: Path | None = None) -> Path:
    """Return the machine-local guide root; never default inside Git."""

    configured = str(os.environ.get(GUIDANCE_ROOT_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    repo = Path(repository_root or _repository_root()).expanduser().resolve()
    return (repo.parent / "handson-local-state" / "task_guidance").resolve()


def task_guidance_dir(task_name: str, root: Path | None = None) -> Path:
    task = str(task_name).strip()
    if not _SAFE_TASK.fullmatch(task):
        raise ValueError(f"Invalid guidance task name: {task_name!r}")
    base = Path(root or default_guidance_root()).expanduser().resolve()
    return base / task


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _episode_arrays(demo_dir: Path) -> dict[str, np.ndarray]:
    import zarr

    replay_path = Path(demo_dir) / "replay.zarr"
    if not replay_path.is_dir():
        raise ValueError(f"{replay_path}: replay.zarr is missing")
    root = zarr.open(str(replay_path), mode="r")
    if "data" not in root:
        raise ValueError(f"{replay_path}: data group is missing")
    data = root["data"]
    required = ("terminal", "succeed", "physx_tcp_state", "physx_tcp_valid")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"{replay_path}: missing guidance fields {missing}")
    return {name: np.asarray(data[name][:]) for name in required}


def _validate_successful_full_episode(demo_dir: Path) -> tuple[np.ndarray, int]:
    arrays = _episode_arrays(demo_dir)
    terminal = np.asarray(arrays["terminal"]).reshape(-1)
    succeed = np.asarray(arrays["succeed"]).reshape(-1)
    tcp = np.asarray(arrays["physx_tcp_state"], dtype=np.float64)
    tcp_valid = np.asarray(arrays["physx_tcp_valid"])
    frame_count = int(tcp.shape[0]) if tcp.ndim >= 1 else 0
    if frame_count < 2:
        raise ValueError("A full guidance demonstration needs at least two frames")
    if terminal.shape != (frame_count,) or succeed.shape != (frame_count,):
        raise ValueError("Guidance terminal/succeed arrays do not match TCP frames")
    if not (bool(terminal[-1]) and bool(succeed[-1])):
        raise ValueError(
            "Guidance was not accepted: the task's final transition is not successful"
        )
    if np.any(terminal[:-1]) or np.any(succeed[:-1]):
        raise ValueError("Guidance contains an earlier terminal boundary")
    if tcp.ndim != 3 or tcp.shape[1] != 2 or tcp.shape[2] < 7:
        raise ValueError(
            "physx_tcp_state must have shape (frames, 2, >=7) for both hands"
        )
    if not np.all(np.isfinite(tcp[:, :, :7])):
        raise ValueError("Guidance TCP poses contain NaN or Inf")
    if tcp_valid.shape != (frame_count, 2) or not np.all(tcp_valid == 1):
        raise ValueError("Both physical TCP readbacks must be valid on every frame")
    return tcp[:, :, :7], frame_count


def _validate_guidance_route_episode(demo_dir: Path) -> tuple[np.ndarray, int]:
    """Validate a hands-only route without requiring task completion.

    A route guide is intentionally not a benchmark episode: the operator may
    stop at any point after demonstrating the desired hand motion.  We still
    require synchronized two-hand physical palm readbacks on every frame so
    the offline arm IK planner never receives a command-frame approximation.
    """

    arrays = _episode_arrays(demo_dir)
    terminal = np.asarray(arrays["terminal"]).reshape(-1)
    succeed = np.asarray(arrays["succeed"]).reshape(-1)
    tcp = np.asarray(arrays["physx_tcp_state"], dtype=np.float64)
    tcp_valid = np.asarray(arrays["physx_tcp_valid"])
    frame_count = int(tcp.shape[0]) if tcp.ndim >= 1 else 0
    if frame_count < 2:
        raise ValueError("A guidance route needs at least two frames")
    if terminal.shape != (frame_count,) or succeed.shape != (frame_count,):
        raise ValueError("Guidance terminal/succeed arrays do not match TCP frames")
    if tcp.ndim != 3 or tcp.shape[1] != 2 or tcp.shape[2] < 7:
        raise ValueError(
            "physx_tcp_state must have shape (frames, 2, >=7) for both hands"
        )
    if not np.all(np.isfinite(tcp[:, :, :7])):
        raise ValueError("Guidance TCP poses contain NaN or Inf")
    if tcp_valid.shape != (frame_count, 2) or not np.all(tcp_valid == 1):
        raise ValueError("Both physical TCP readbacks must be valid on every frame")
    return tcp[:, :, :7], frame_count


def register_successful_guidance_demo(
    demo_dir: Path,
    *,
    task_name: str,
    collector_id: str,
    session_id: str,
) -> Path:
    """Register only a complete task-success episode as a guide candidate."""

    demo = Path(demo_dir).expanduser().resolve()
    _, frame_count = _validate_successful_full_episode(demo)
    replay = demo / "replay.zarr"
    guide_id = hashlib.sha256(
        f"{task_name}\0{session_id}\0{demo.name}".encode("utf-8")
    ).hexdigest()[:20]
    marker = demo / GUIDANCE_MARKER_NAME
    _write_json_atomic(
        marker,
        {
            "format": GUIDANCE_FORMAT,
            "version": GUIDANCE_VERSION,
            "guide_id": guide_id,
            "task_name": str(task_name),
            "collector_id": str(collector_id),
            "session_id": str(session_id),
            "full_process": True,
            "task_success": True,
            "frame_count": frame_count,
            "source_replay": replay.name,
            "planner_status": "pending",
            "active": False,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        },
    )
    return marker


def register_guidance_route_demo(
    demo_dir: Path,
    *,
    task_name: str,
    collector_id: str,
    session_id: str,
) -> Path:
    """Register a hands-only route stopped by the operator.

    This marker deliberately records ``full_process=false`` and
    ``task_success=false``.  The compiler first attempts the full offline
    IK/FK/collision certificate; when that certificate is unavailable, a
    collision-safe prefix may still be published as a coarse soft hint.
    """

    demo = Path(demo_dir).expanduser().resolve()
    _, frame_count = _validate_guidance_route_episode(demo)
    replay = demo / "replay.zarr"
    guide_id = hashlib.sha256(
        f"{task_name}\0{session_id}\0{demo.name}".encode("utf-8")
    ).hexdigest()[:20]
    marker = demo / GUIDANCE_MARKER_NAME
    _write_json_atomic(
        marker,
        {
            "format": GUIDANCE_FORMAT,
            "version": GUIDANCE_VERSION,
            "guide_id": guide_id,
            "task_name": str(task_name),
            "collector_id": str(collector_id),
            "session_id": str(session_id),
            "full_process": False,
            "task_success": False,
            "route_only": True,
            "guidance_type": "hand_route",
            "frame_count": frame_count,
            "source_replay": replay.name,
            "planner_status": "pending",
            "active": False,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        },
    )
    return marker


def _is_guidance_route_marker(marker: Mapping[str, Any], task_name: str) -> bool:
    return bool(
        marker.get("format") == GUIDANCE_FORMAT
        and marker.get("version") == GUIDANCE_VERSION
        and marker.get("task_name") == task_name
        and marker.get("route_only") is True
        and marker.get("guidance_type") == "hand_route"
        and marker.get("full_process") is False
        and marker.get("task_success") is False
    )


def mark_guidance_planner_result(
    demo_dir: Path,
    *,
    success: bool,
    detail: str = "",
) -> dict[str, Any]:
    marker_path = Path(demo_dir) / GUIDANCE_MARKER_NAME
    marker = _load_json(marker_path)
    marker.update(
        {
            "planner_status": "passed" if success else "failed",
            "active": bool(success),
            "planner_detail": str(detail).strip(),
            "updated_at": _now_iso(),
        }
    )
    _write_json_atomic(marker_path, marker)
    return marker


def mark_guidance_hint_result(
    demo_dir: Path,
    *,
    detail: str = "",
    safe_frame_count: int | None = None,
) -> dict[str, Any]:
    """Mark a route as usable as a coarse, non-certified IK preference."""

    marker_path = Path(demo_dir) / GUIDANCE_MARKER_NAME
    marker = _load_json(marker_path)
    marker.update(
        {
            "planner_status": "hint_ready",
            "active": True,
            "guidance_path_mode": GUIDANCE_COARSE_PATH_MODE,
            "planner_detail": str(detail).strip(),
            "updated_at": _now_iso(),
        }
    )
    if safe_frame_count is not None:
        marker["hint_safe_frame_count"] = int(safe_frame_count)
    _write_json_atomic(marker_path, marker)
    return marker


def _candidate_demo_dirs(task_dir: Path) -> list[Path]:
    if not Path(task_dir).is_dir():
        return []
    return sorted(
        (path.parent for path in Path(task_dir).rglob(GUIDANCE_MARKER_NAME)),
        key=lambda path: str(path),
    )


def _normalized_quaternions(values: np.ndarray) -> np.ndarray:
    quats = np.asarray(values, dtype=np.float64).copy()
    for index in range(1, len(quats)):
        if float(np.dot(quats[index - 1], quats[index])) < 0.0:
            quats[index] *= -1.0
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    if np.any(norms <= 1.0e-9):
        raise ValueError("Guidance contains a zero-length quaternion")
    return quats / norms


def _resample(values: np.ndarray, points: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or len(source) < 2:
        raise ValueError(f"Expected a 2D trajectory with >=2 rows, got {source.shape}")
    old = np.linspace(0.0, 1.0, len(source), dtype=np.float64)
    new = np.linspace(0.0, 1.0, int(points), dtype=np.float64)
    result = np.empty((len(new), source.shape[1]), dtype=np.float64)
    for column in range(source.shape[1]):
        result[:, column] = np.interp(new, old, source[:, column])
    return result


def _resample_tcp(tcp: np.ndarray, points: int) -> np.ndarray:
    result = np.empty((int(points), 2, 7), dtype=np.float64)
    for side in range(2):
        result[:, side, :3] = _resample(tcp[:, side, :3], points)
        result[:, side, 3:7] = _normalized_quaternions(
            _resample(_normalized_quaternions(tcp[:, side, 3:7]), points)
        )
    return result


def _load_coarse_route_hint(
    demo_dir: Path,
    *,
    expected_frames: int,
    points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a collision-safe prefix of a C01 plan as a soft route hint.

    The exact simplified-post-IK loader intentionally rejects one bad frame.
    A hand-route guide has a different contract: it may stop before the
    collision checker can certify its tail.  We therefore keep only the
    longest safe prefix beginning at the recorded initial pose.  The live
    hand target remains the hard IK objective; these arrays only provide a
    monotonic path and a preferred endpoint.
    """

    import zarr

    replay = Path(demo_dir) / "replay.zarr"
    root = zarr.open(str(replay), mode="r")
    if "data" not in root:
        raise ValueError(f"{replay}: data group is missing")
    data = root["data"]
    names = {
        "tcp": "physx_tcp_state",
        "tcp_valid": "physx_tcp_valid",
        "qpos": "c01_arm_plan_qpos",
        "valid": "c01_arm_plan_valid_frame",
        "fk_valid": "c01_arm_plan_fk_valid_frame",
        "collision_valid": "c01_arm_plan_collision_valid_frame",
    }
    missing = [name for name in names.values() if name not in data]
    if missing:
        raise ValueError(f"{replay}: coarse route fields are missing {missing}")
    tcp = np.asarray(data[names["tcp"]][:], dtype=np.float64)
    tcp_valid = np.asarray(data[names["tcp_valid"]][:]).reshape(-1, 2)
    qpos = np.asarray(data[names["qpos"]][:], dtype=np.float64)
    valid = np.asarray(data[names["valid"]][:]).reshape(-1)
    fk_valid = np.asarray(data[names["fk_valid"]][:]).reshape(-1)
    collision_valid = np.asarray(data[names["collision_valid"]][:]).reshape(-1)
    frame_count = int(expected_frames)
    if (
        tcp.ndim != 3
        or tcp.shape[0] != frame_count
        or tcp.shape[1] != 2
        or tcp.shape[2] < 7
        or tcp_valid.shape != (frame_count, 2)
        or qpos.shape != (frame_count, 14)
        or any(
            len(values) != frame_count
            for values in (valid, fk_valid, collision_valid)
        )
    ):
        raise ValueError(
            f"{replay}: coarse route arrays do not match {frame_count} frames"
        )
    finite = (
        np.all(np.isfinite(tcp[:, :, :7]), axis=(1, 2))
        & np.all(np.isfinite(qpos), axis=1)
    )
    safe = (
        finite
        & np.all(tcp_valid == 1, axis=1)
        & (valid == 1)
        & (fk_valid == 1)
        & (collision_valid == 1)
    )
    prefix_length = 0
    while prefix_length < frame_count and bool(safe[prefix_length]):
        prefix_length += 1
    if prefix_length < 2:
        raise ValueError(
            f"{replay}: fewer than two collision-safe frames at route start"
        )
    tcp_prefix = tcp[:prefix_length, :, :7]
    qpos_prefix = qpos[:prefix_length]
    return (
        _resample_tcp(tcp_prefix, points),
        _resample(qpos_prefix, points),
        {
            "path_mode": GUIDANCE_COARSE_PATH_MODE,
            "hint_source": "c01_arm_plan_collision_safe_prefix",
            "safe_frame_count": prefix_length,
            "source_frame_count": frame_count,
        },
    )


def _read_profile_path(path: Path, task_name: str) -> dict[str, Any] | None:
    try:
        archive = np.load(Path(path), allow_pickle=False)
    except (FileNotFoundError, OSError, ValueError):
        return None
    with archive as values:
        try:
            contract = json.loads(str(np.asarray(values["contract_json"]).item()))
            guide_ids = np.asarray(values["guide_ids"]).astype(str)
            tcp = np.asarray(values["tcp_pose"], dtype=np.float64)
            qpos = np.asarray(values["arm_qpos"], dtype=np.float64)
            if "path_modes" in values.files:
                path_modes = np.asarray(values["path_modes"]).astype(str)
            else:
                # Profiles produced before soft route hints existed are exact
                # monotonic guides by definition.
                path_modes = np.full(
                    len(guide_ids), GUIDANCE_EXACT_PATH_MODE, dtype="<U32"
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(contract, dict) or contract.get("format") != GUIDANCE_PROFILE_FORMAT:
        return None
    if contract.get("version") != GUIDANCE_PROFILE_VERSION or contract.get("task_name") != task_name:
        return None
    if (
        tcp.ndim != 4
        or tcp.shape[2:] != (2, 7)
        or qpos.shape != (tcp.shape[0], tcp.shape[1], 14)
        or len(guide_ids) != tcp.shape[0]
        or path_modes.shape != (tcp.shape[0],)
        or any(
            str(mode)
            not in {GUIDANCE_EXACT_PATH_MODE, GUIDANCE_COARSE_PATH_MODE}
            for mode in path_modes
        )
        or not np.all(np.isfinite(tcp))
        or not np.all(np.isfinite(qpos))
    ):
        return None
    return {
        "contract": contract,
        "guide_ids": guide_ids,
        "path_modes": path_modes,
        "tcp_pose": tcp,
        "arm_qpos": qpos,
        "path": Path(path),
    }


def _write_profile_path(
    path: Path,
    *,
    task_name: str,
    guide_ids: list[str],
    tcp_pose: np.ndarray,
    arm_qpos: np.ndarray,
    selection: str = "nearest_pose_with_monotonic_progress",
    path_modes: list[str] | np.ndarray | None = None,
) -> Path:
    if not guide_ids:
        raise ValueError("Cannot publish an empty guidance profile")
    tcp_array = np.asarray(tcp_pose, dtype=np.float32)
    q_array = np.asarray(arm_qpos, dtype=np.float32)
    if (
        tcp_array.ndim != 4
        or tcp_array.shape[2:] != (2, 7)
        or q_array.shape != (tcp_array.shape[0], tcp_array.shape[1], 14)
        or len(guide_ids) != tcp_array.shape[0]
    ):
        raise ValueError("Guidance profile arrays have incompatible shapes")
    if path_modes is None:
        mode_array = np.full(
            len(guide_ids), GUIDANCE_EXACT_PATH_MODE, dtype="<U32"
        )
    else:
        mode_array = np.asarray(path_modes).astype(str)
        if mode_array.shape != (len(guide_ids),):
            raise ValueError("Guidance profile path_modes do not match guide_ids")
        if any(
            str(mode) not in {GUIDANCE_EXACT_PATH_MODE, GUIDANCE_COARSE_PATH_MODE}
            for mode in mode_array
        ):
            raise ValueError("Guidance profile contains an unsupported path mode")
    unique_modes = sorted({str(mode) for mode in mode_array})
    contract = {
        "format": GUIDANCE_PROFILE_FORMAT,
        "version": GUIDANCE_PROFILE_VERSION,
        "task_name": task_name,
        "guide_count": len(guide_ids),
        "points_per_guide": int(tcp_array.shape[1]),
        "side_order": ["right", "left"],
        "selection": selection,
        "path_mode": unique_modes[0] if len(unique_modes) == 1 else "mixed",
        "guide_path_modes": [str(mode) for mode in mode_array],
        "created_at": _now_iso(),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                contract_json=np.asarray(
                    json.dumps(contract, ensure_ascii=False, sort_keys=True)
                ),
                guide_ids=np.asarray(
                    guide_ids, dtype=f"<U{max(map(len, guide_ids))}"
                ),
                path_modes=mode_array,
                tcp_pose=tcp_array,
                arm_qpos=q_array,
            )
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size > GUIDANCE_MAX_SHARED_PROFILE_BYTES:
            raise ValueError(
                "Compressed task guidance exceeds the 100 MB sharing limit"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def merge_task_guidance_profiles(
    task_name: str,
    source_paths: list[Path],
    destination: Path,
) -> dict[str, Any]:
    """Merge portable compact profiles by immutable guide id."""

    merged_tcp: list[np.ndarray] = []
    merged_qpos: list[np.ndarray] = []
    guide_ids: list[str] = []
    path_modes: list[str] = []
    seen: set[str] = set()
    points: int | None = None
    for source_path in source_paths:
        profile = _read_profile_path(Path(source_path), task_name)
        if profile is None:
            raise ValueError(f"Invalid guidance profile: {source_path}")
        tcp = np.asarray(profile["tcp_pose"], dtype=np.float64)
        qpos = np.asarray(profile["arm_qpos"], dtype=np.float64)
        if points is None:
            points = int(tcp.shape[1])
        if int(tcp.shape[1]) != points:
            raise ValueError("Guidance profiles use different progress grids")
        profile_modes = np.asarray(
            profile.get("path_modes"), dtype=str
        )
        for index, raw_id in enumerate(profile["guide_ids"]):
            guide_id = str(raw_id)
            if guide_id in seen:
                continue
            seen.add(guide_id)
            guide_ids.append(guide_id)
            path_modes.append(str(profile_modes[index]))
            merged_tcp.append(tcp[index])
            merged_qpos.append(qpos[index])
    _write_profile_path(
        destination,
        task_name=task_name,
        guide_ids=guide_ids,
        tcp_pose=np.asarray(merged_tcp),
        arm_qpos=np.asarray(merged_qpos),
        path_modes=path_modes,
    )
    return {
        "success": True,
        "task_name": task_name,
        "active_guides": len(guide_ids),
        "hint_guides": int(
            sum(mode == GUIDANCE_COARSE_PATH_MODE for mode in path_modes)
        ),
        "profile_path": str(Path(destination)),
        "profile_bytes": Path(destination).stat().st_size,
    }


def compile_task_guidance_profile(
    task_name: str,
    *,
    root: Path | None = None,
    points: int = GUIDANCE_PROFILE_POINTS,
) -> dict[str, Any]:
    """Compile all valid guides without averaging distinct motion strategies."""

    task_dir = task_guidance_dir(task_name, root)
    task_dir.mkdir(parents=True, exist_ok=True)
    tcp_guides: list[np.ndarray] = []
    q_guides: list[np.ndarray] = []
    guide_ids: list[str] = []
    path_modes: list[str] = []
    seen_guide_ids: set[str] = set()
    rejected: list[dict[str, str]] = []
    remote_cache = _read_profile_path(
        task_dir / GUIDANCE_REMOTE_CACHE_NAME,
        task_name,
    )
    if remote_cache is not None:
        for index, raw_id in enumerate(remote_cache["guide_ids"]):
            guide_id = str(raw_id)
            if guide_id in seen_guide_ids:
                continue
            seen_guide_ids.add(guide_id)
            guide_ids.append(guide_id)
            path_modes.append(str(remote_cache["path_modes"][index]))
            tcp_guides.append(
                _resample_tcp(
                    np.asarray(remote_cache["tcp_pose"])[index], points
                )
            )
            q_guides.append(
                _resample(np.asarray(remote_cache["arm_qpos"])[index], points)
            )
    for demo in _candidate_demo_dirs(task_dir):
        try:
            marker = _load_json(demo / GUIDANCE_MARKER_NAME)
            is_full = bool(
                marker.get("format") == GUIDANCE_FORMAT
                and marker.get("version") == GUIDANCE_VERSION
                and marker.get("task_name") == task_name
                and marker.get("full_process") is True
                and marker.get("task_success") is True
            )
            is_route = _is_guidance_route_marker(marker, task_name)
            if not (is_full or is_route):
                raise ValueError("guidance marker is incompatible with this task")
            tcp, frame_count = (
                _validate_successful_full_episode(demo)
                if is_full
                else _validate_guidance_route_episode(demo)
            )
            path_mode = GUIDANCE_EXACT_PATH_MODE
            tcp_path: np.ndarray
            q_path: np.ndarray
            try:
                qpos = load_validated_simplified_arm_plan(
                    demo / "replay.zarr", expected_frames=frame_count
                )
                tcp_path = _resample_tcp(tcp, points)
                q_path = _resample(qpos, points)
            except (OSError, KeyError, TypeError, ValueError) as exact_exc:
                if not is_route:
                    raise
                tcp_path, q_path, hint_meta = _load_coarse_route_hint(
                    demo,
                    expected_frames=frame_count,
                    points=points,
                )
                path_mode = GUIDANCE_COARSE_PATH_MODE
                mark_guidance_hint_result(
                    demo,
                    detail=(
                        "全局 IK 全帧证书不可用，已截取碰撞安全前缀作为粗略路径： "
                        f"{exact_exc}"
                    ),
                    safe_frame_count=int(hint_meta["safe_frame_count"]),
                )
            guide_id = str(marker.get("guide_id") or demo.name)
            if guide_id not in seen_guide_ids:
                seen_guide_ids.add(guide_id)
                tcp_guides.append(tcp_path)
                q_guides.append(q_path)
                guide_ids.append(guide_id)
                path_modes.append(path_mode)
            if (
                path_mode == GUIDANCE_EXACT_PATH_MODE
                and (
                    marker.get("planner_status") != "passed"
                    or marker.get("active") is not True
                )
            ):
                mark_guidance_planner_result(demo, success=True)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            rejected.append({"path": str(demo), "detail": str(exc)})

    profile_path = task_dir / GUIDANCE_PROFILE_NAME
    if not guide_ids:
        profile_path.unlink(missing_ok=True)
        return {
            "success": False,
            "task_name": task_name,
            "active_guides": 0,
            "rejected": rejected,
            "profile_path": str(profile_path),
        }

    _write_profile_path(
        profile_path,
        task_name=task_name,
        guide_ids=guide_ids,
        tcp_pose=np.asarray(tcp_guides),
        arm_qpos=np.asarray(q_guides),
        path_modes=path_modes,
    )
    return {
        "success": True,
        "task_name": task_name,
        "active_guides": len(guide_ids),
        "rejected": rejected,
        "profile_path": str(profile_path),
        "profile_bytes": profile_path.stat().st_size,
        "hint_guides": int(
            sum(mode == GUIDANCE_COARSE_PATH_MODE for mode in path_modes)
        ),
    }


def load_task_guidance_profile(
    task_name: str,
    *,
    root: Path | None = None,
) -> dict[str, Any] | None:
    enabled = str(os.environ.get(GUIDANCE_ENABLED_ENV, "1")).strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    path = task_guidance_dir(task_name, root) / GUIDANCE_PROFILE_NAME
    return _read_profile_path(path, task_name)


def _safe_guidance_demo_directory(demo: Path, task_dir: Path) -> bool:
    """Accept only a real final demo directory in this task's local root."""

    demo = Path(demo)
    task_dir = Path(task_dir)
    pending = task_dir / _PHYSX_PENDING_DIRECTORY
    if task_dir.is_symlink() or demo.is_symlink() or not demo.is_dir():
        return False
    allowed_parents = (task_dir, pending)
    for parent in allowed_parents:
        if parent.is_symlink() or demo.parent != parent:
            continue
        try:
            demo.resolve().relative_to(task_dir.resolve())
        except (OSError, ValueError):
            continue
        return True
    return False


def _process_is_live_recorder(pid: object) -> bool:
    if type(pid) is not int or pid <= 1:
        return False
    process_dir = Path("/proc") / str(pid)
    if not process_dir.exists():
        return False
    try:
        command = (process_dir / "cmdline").read_bytes().split(b"\0")
    except OSError:
        # A process that still exists but cannot be inspected is never safe to
        # classify as an abandoned recording.
        return True
    return any(
        Path(part.decode(errors="replace")).name == "record_demos_zarr.py"
        for part in command
        if part
    )


def _history_accepted_demo_ids(history: Path) -> set[str] | None:
    """Return accepted demo ids, or ``None`` when the journal is malformed."""

    accepted: set[str] = set()
    finalization_path = history / _PHYSX_FINALIZATION_NAME
    if finalization_path.is_file():
        try:
            finalization = _load_json(finalization_path)
            rows = finalization.get("accepted_demos", [])
            if not isinstance(rows, list):
                return None
            for row in rows:
                if not isinstance(row, Mapping):
                    return None
                demo_id = row.get("demo_id")
                if not isinstance(demo_id, str) or Path(demo_id).name != demo_id:
                    return None
                accepted.add(demo_id)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return None

    events_root = history / _PHYSX_EVENTS_DIRECTORY
    if events_root.is_symlink():
        return None
    if not events_root.is_dir():
        return accepted
    for event_dir in sorted(events_root.iterdir()):
        if event_dir.is_symlink() or not event_dir.is_dir():
            return None
        event_path = event_dir / _PHYSX_EVENT_NAME
        try:
            event = _load_json(event_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return None
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            return None
        demo_id = payload.get("demo_id")
        if payload.get("accepted") is True:
            if not isinstance(demo_id, str) or Path(demo_id).name != demo_id:
                return None
            accepted.add(demo_id)
        elif isinstance(demo_id, str) and demo_id:
            # A non-empty demo binding with an unaccepted flag is internally
            # inconsistent. Keep it rather than risking deletion.
            return None
    return accepted


def _safe_history_directory(history: Path, task_dir: Path) -> bool:
    history_root = (
        task_dir
        / _PHYSX_PENDING_DIRECTORY
        / _PHYSX_HISTORY_DIRECTORY
    )
    if (
        task_dir.is_symlink()
        or history_root.is_symlink()
        or history.is_symlink()
        or not history.is_dir()
        or history.parent != history_root
    ):
        return False
    try:
        history.resolve().relative_to(task_dir.resolve())
    except (OSError, ValueError):
        return False
    return True


def _failed_guidance_records(
    task_name: str,
    *,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Return verified local failures, retaining paths only for deletion."""

    task_dir = task_guidance_dir(task_name, root)
    if not task_dir.is_dir() or task_dir.is_symlink():
        return []
    records: list[dict[str, Any]] = []
    for demo in _candidate_demo_dirs(task_dir):
        if not _safe_guidance_demo_directory(demo, task_dir):
            continue
        try:
            marker = _load_json(demo / GUIDANCE_MARKER_NAME)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            continue
        guide_id = str(marker.get("guide_id") or "")
        if (
            marker.get("format") != GUIDANCE_FORMAT
            or marker.get("version") != GUIDANCE_VERSION
            or marker.get("task_name") != task_name
            or not (
                (
                    marker.get("full_process") is True
                    and marker.get("task_success") is True
                )
                or _is_guidance_route_marker(marker, task_name)
            )
            or marker.get("planner_status") != "failed"
            or marker.get("active") is True
            or not _SAFE_GUIDE_ID.fullmatch(guide_id)
        ):
            continue
        records.append(
            {
                "failure_id": f"guide:{guide_id}",
                "kind": "planner_failed",
                "guide_id": guide_id,
                "created_at": str(marker.get("created_at") or ""),
                "detail": str(
                    marker.get("planner_detail")
                    or "手部路线的全局 IK / 碰撞检查未通过。"
                )[:600],
                "_path": demo,
            }
        )

    history_root = (
        task_dir
        / _PHYSX_PENDING_DIRECTORY
        / _PHYSX_HISTORY_DIRECTORY
    )
    if history_root.is_dir() and not history_root.is_symlink():
        for history in sorted(history_root.iterdir()):
            if not _safe_history_directory(history, task_dir):
                continue
            try:
                session = _load_json(history / _PHYSX_SESSION_NAME)
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                continue
            if (
                session.get("format") != _PHYSX_SESSION_FORMAT
                or session.get("task_name") != task_name
            ):
                continue
            metadata = session.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            if _process_is_live_recorder(metadata.get("recorder_pid")):
                continue
            accepted_demo_ids = _history_accepted_demo_ids(history)
            if accepted_demo_ids is None or accepted_demo_ids:
                continue
            session_id = str(session.get("session_id") or "").strip()
            if not session_id:
                continue
            opaque_id = hashlib.sha256(
                f"{task_name}\0{session_id}\0{history.name}".encode("utf-8")
            ).hexdigest()[:20]
            latest_detail = "录制在形成完整成功演示前中止。"
            events_root = history / _PHYSX_EVENTS_DIRECTORY
            event_paths = (
                sorted(events_root.glob(f"*/{_PHYSX_EVENT_NAME}"))
                if events_root.is_dir() and not events_root.is_symlink()
                else []
            )
            if event_paths:
                try:
                    latest = _load_json(event_paths[-1])
                    payload = latest.get("payload")
                    if isinstance(payload, Mapping):
                        outcome = str(
                            payload.get("outcome")
                            or payload.get("reason")
                            or ""
                        ).strip()
                        error = str(payload.get("error") or "").strip()
                        latest_detail = "：".join(
                            value for value in (outcome, error) if value
                        )[:600] or latest_detail
                except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                    continue
            records.append(
                {
                    "failure_id": f"attempt:{opaque_id}",
                    "kind": "recording_attempt",
                    "session_id": session_id,
                    "created_at": str(session.get("created_at") or ""),
                    "detail": latest_detail,
                    "_path": history,
                }
            )
    return records


def failed_guidance_items(
    task_name: str,
    *,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """List deletable failures without exposing any filesystem path."""

    return [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in _failed_guidance_records(task_name, root=root)
    ]


def delete_failed_task_guidance(
    task_name: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Atomically move every verified local failure to a recoverable trash."""

    task_dir = task_guidance_dir(task_name, root)
    records = _failed_guidance_records(task_name, root=root)
    if not records:
        return {
            "success": True,
            "task_name": task_name,
            "deleted_failures": 0,
            "deleted_artifacts": 0,
            "recoverable": True,
        }

    targets: list[Path] = [Path(record["_path"]) for record in records]
    # A failed planner guide has an accepted session-history binding. Remove
    # that journal too when it is exclusively bound to the failed demo.
    history_root = (
        task_dir
        / _PHYSX_PENDING_DIRECTORY
        / _PHYSX_HISTORY_DIRECTORY
    )
    failed_demo_names = {
        path.name
        for record, path in zip(records, targets, strict=True)
        if record["kind"] == "planner_failed"
    }
    if history_root.is_dir() and not history_root.is_symlink():
        for history in sorted(history_root.iterdir()):
            if not _safe_history_directory(history, task_dir):
                continue
            accepted = _history_accepted_demo_ids(history)
            if accepted and accepted <= failed_demo_names:
                targets.append(history)

    base = task_dir.parent
    trash_base = base / ".trash"
    trash_task = trash_base / task_name
    if trash_base.is_symlink() or trash_task.is_symlink():
        raise ValueError("Guidance trash path must not be a symbolic link")
    trash_task.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    for index, target in enumerate(dict.fromkeys(targets)):
        if target.is_symlink() or not target.is_dir():
            raise ValueError("Guidance failure changed before deletion")
        try:
            target.resolve().relative_to(task_dir.resolve())
        except (OSError, ValueError) as exc:
            raise ValueError("Guidance failure escaped its task directory") from exc
        destination = trash_task / f"{stamp}-{index:03d}-{target.name}"
        os.replace(target, destination)
        moved.append(destination)
    return {
        "success": True,
        "task_name": task_name,
        "deleted_failures": len(records),
        "deleted_artifacts": len(moved),
        "deleted_failure_ids": [record["failure_id"] for record in records],
        "recoverable": True,
    }


def guidance_summary(task_name: str, *, root: Path | None = None) -> dict[str, Any]:
    task_dir = task_guidance_dir(task_name, root)
    candidates = _candidate_demo_dirs(task_dir)
    complete = 0
    planned = 0
    hints = 0
    failed = 0
    pending = 0
    for demo in candidates:
        try:
            marker = _load_json(demo / GUIDANCE_MARKER_NAME)
            if marker.get("task_name") != task_name:
                continue
            if (
                marker.get("full_process") is True
                and marker.get("task_success") is True
            ) or _is_guidance_route_marker(marker, task_name):
                complete += 1
            status = str(marker.get("planner_status") or "pending")
            if status == "passed" and marker.get("active") is True:
                planned += 1
            elif status == "hint_ready" and marker.get("active") is True:
                planned += 1
                hints += 1
            elif status == "failed":
                failed += 1
            else:
                pending += 1
        except (OSError, ValueError, json.JSONDecodeError):
            failed += 1
    profile = load_task_guidance_profile(task_name, root=root)
    shared_active = int(len(profile["guide_ids"])) if profile is not None else 0
    failure_items = failed_guidance_items(task_name, root=root)
    failed_attempts = sum(
        item.get("kind") == "recording_attempt" for item in failure_items
    )
    return {
        "success": True,
        "task_name": task_name,
        "complete_guides": complete,
        "active_guides": shared_active,
        "local_active_guides": planned,
        "hint_guides": hints,
        "failed_guides": failed,
        "failed_attempts": failed_attempts,
        "pending_guides": pending,
        "deletable_failures": len(failure_items),
        "failure_items": failure_items,
        "profile_ready": profile is not None,
        "root": str(task_dir),
        "storage": "machine_local_not_git",
    }
