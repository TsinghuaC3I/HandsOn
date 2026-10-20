"""Crash-safe journal for replaying one persistent PhysX collection session.

The ordinary demo writer intentionally publishes only successful trajectories.
That is insufficient when several episodes share one PhysX process: rejected
reset attempts and discarded trajectories can change private solver history
before a later successful demo.  This module stores that missing input history
without storing camera frames.

Each event is committed as one directory.  A discarded trajectory's compact
``replay.zarr`` and its ``event.json`` therefore become visible in the same
atomic rename.  Event identities form a SHA-256 chain, and ``finalize`` binds
the immutable session header plus every event file into one tree identity.

The module does not know how to drive an environment.  A certifier can use
``load_session_history`` and ``load_discarded_trajectory`` to execute the
recorded reset/drive sequence in one fresh environment.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zarr

from handson.data.episode_store import ZarrEpisodeStore
from handson.data.recording_fields import (
    declared_recording_field_names,
    stack_transition_numeric_observation_fields,
    write_recording_schema,
)


SESSION_FORMAT = "handson.physx_session_history"
SESSION_VERSION = 1
EVENT_FORMAT = "handson.physx_session_history_event"
EVENT_VERSION = 1
FINALIZATION_FORMAT = "handson.physx_session_history_finalization"
FINALIZATION_VERSION = 1
TREE_IDENTITY_ALGORITHM = "sha256-path-size-content-v1"

SESSION_HEADER_NAME = "session.json"
EVENTS_DIRECTORY_NAME = "events"
EVENT_DOCUMENT_NAME = "event.json"
DISCARDED_REPLAY_NAME = "replay.zarr"
DISCARDED_SCHEMA_NAME = "recording_schema.json"
FINALIZATION_NAME = "FINALIZED.json"
LOCK_NAME = ".journal.lock"

_EVENT_DIRECTORY_WIDTH = 8
_BASE_TRAJECTORY_FIELDS = frozenset({"action", "terminal", "succeed"})
_RECORDED_DRIVE_FIELDS = frozenset(
    {
        "next_physx_substep_count",
        "next_physx_substep_robot_joint_target",
        "next_physx_substep_robot_joint_target_valid",
        "next_physx_robot_joint_target",
        "next_physx_robot_joint_target_valid",
    }
)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        "PhysX session history values must be JSON-compatible; got "
        f"{type(value).__name__}."
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some filesystems do not support directory fsync.  The atomic rename
        # still prevents readers from observing a partially written event.
        pass
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                _json_safe(payload),
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _regular_files(root: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []

    def raise_walk_error(error: OSError) -> None:
        raise error

    for current, directory_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        current_path = Path(current)
        for name in tuple(directory_names):
            child = current_path / name
            if child.is_symlink():
                raise ValueError(f"Session history contains a symlink: {child}")
        directory_names[:] = sorted(directory_names)
        for name in sorted(file_names):
            child = current_path / name
            if child.is_symlink() or not child.is_file():
                raise ValueError(
                    f"Session history contains a non-regular file: {child}"
                )
            files.append((child.relative_to(root).as_posix(), child))
    files.sort(key=lambda item: item[0])
    return files


def _identity_for_files(
    root: Path,
    files: Iterable[tuple[str, Path]],
) -> dict[str, Any]:
    del root  # Relative paths are supplied explicitly by the caller.
    ordered = sorted(files, key=lambda item: item[0])
    digest = hashlib.sha256()
    total_bytes = 0
    for relative_path, file_path in ordered:
        relative_bytes = relative_path.encode("utf-8")
        size_before = file_path.stat().st_size
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(size_before.to_bytes(8, "big"))
        bytes_read = 0
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                bytes_read += len(chunk)
        if bytes_read != size_before or file_path.stat().st_size != size_before:
            raise OSError(f"Session history changed while hashing: {file_path}")
        total_bytes += size_before
    return {
        "algorithm": TREE_IDENTITY_ALGORITHM,
        "sha256": digest.hexdigest(),
        "file_count": len(ordered),
        "bytes": total_bytes,
    }


def _tree_identity(path: Path) -> dict[str, Any]:
    return _identity_for_files(path, _regular_files(path))


def _scalar_sim_time(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        return None
    result = float(array.reshape(-1)[0])
    return result if math.isfinite(result) else None


def _trajectory_sim_time_bounds(
    episode_data: Mapping[str, np.ndarray],
) -> tuple[float | None, float | None]:
    start_values = episode_data.get("physx_sim_time")
    end_values = episode_data.get("next_physx_sim_time")
    start = None
    end = None
    if start_values is not None and np.asarray(start_values).shape[0] > 0:
        start = _scalar_sim_time(np.asarray(start_values)[0])
    if end_values is not None and np.asarray(end_values).shape[0] > 0:
        end = _scalar_sim_time(np.asarray(end_values)[-1])
    return start, end


def _accepted_trajectory_summary(
    trajectory: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[int, float | None, float | None]:
    """Read only length/time boundaries; accepted data already exists on disk."""

    if isinstance(trajectory, Mapping):
        action = trajectory.get("action", trajectory.get("actions"))
        if action is None:
            return 0, None, None
        step_count = int(np.asarray(action).shape[0])
        return step_count, *_trajectory_sim_time_bounds(
            {str(name): np.asarray(value) for name, value in trajectory.items()}
        )
    steps = list(trajectory)
    if not steps:
        return 0, None, None
    first_observation = steps[0].get("observations", {})
    last_observation = steps[-1].get("next_observations", {})
    start = (
        _scalar_sim_time(first_observation.get("physx_sim_time"))
        if isinstance(first_observation, Mapping)
        else None
    )
    end = (
        _scalar_sim_time(last_observation.get("physx_sim_time"))
        if isinstance(last_observation, Mapping)
        else None
    )
    return len(steps), start, end


def _episode_data_from_transitions(
    trajectory: Sequence[Mapping[str, Any]],
    *,
    environment_schema: Mapping[str, Any] | None,
    required_fields: Iterable[str] | None,
    allow_partial_terminal_guard: bool = False,
) -> dict[str, np.ndarray]:
    if not trajectory:
        return {}
    actions: list[np.ndarray] = []
    terminals: list[bool] = []
    succeeds: list[bool] = []
    for index, transition in enumerate(trajectory):
        if not isinstance(transition, Mapping):
            raise TypeError(f"Trajectory step {index} must be a mapping.")
        action = transition.get("actions", transition.get("action"))
        if action is None:
            raise ValueError(f"Trajectory step {index} has no action.")
        action_array = np.asarray(action)
        if action_array.dtype == object or action_array.ndim == 0:
            raise ValueError(
                f"Trajectory action {index} must be a fixed-shape array."
            )
        actions.append(action_array)
        terminals.append(
            bool(
                transition.get(
                    "dones",
                    transition.get("terminal", False),
                )
            )
        )
        info = transition.get("infos", transition.get("info", {}))
        succeeds.append(
            bool(isinstance(info, Mapping) and info.get("succeed", False))
        )
    try:
        action_values = np.stack(actions, axis=0)
    except ValueError as exc:
        raise ValueError("Trajectory action shape changed within the segment.") from exc

    required = (
        declared_recording_field_names(environment_schema)
        if required_fields is None
        else {str(name) for name in required_fields}
    )
    episode_data = {
        "action": action_values,
        "terminal": np.asarray(terminals, dtype=np.uint8),
        "succeed": np.asarray(succeeds, dtype=np.uint8),
    }
    episode_data.update(
        stack_transition_numeric_observation_fields(
            trajectory,
            required_fields=required,
            environment_schema=environment_schema,
            allow_partial_terminal_guard=allow_partial_terminal_guard,
        )
    )
    return episode_data


def _normalize_episode_data(
    trajectory: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    environment_schema: Mapping[str, Any] | None,
    required_fields: Iterable[str] | None,
    allow_partial_terminal_guard: bool = False,
) -> dict[str, np.ndarray]:
    if isinstance(trajectory, Mapping):
        if allow_partial_terminal_guard:
            raise ValueError(
                "Partial terminal guard telemetry requires transition records "
                "with explicit reset/guard info."
            )
        normalized = {
            str(name): np.asarray(value)
            for name, value in trajectory.items()
        }
        if "terminal" not in normalized and "dones" in normalized:
            normalized["terminal"] = np.asarray(normalized.pop("dones"))
        return normalized
    if isinstance(trajectory, (str, bytes, bytearray)):
        raise TypeError("Trajectory must be an array mapping or step sequence.")
    return _episode_data_from_transitions(
        list(trajectory),
        environment_schema=environment_schema,
        required_fields=required_fields,
        allow_partial_terminal_guard=allow_partial_terminal_guard,
    )


def compact_physx_episode_data(
    episode_data: Mapping[str, Any],
    *,
    environment_schema: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Return replay inputs, PhysX and declared task state without images."""

    fields = (environment_schema or {}).get("fields", {})
    task_names = {
        str(name)
        for name, spec in fields.items()
        if isinstance(spec, Mapping) and spec.get("replay_domain") == "task_state"
    }
    task_fields = task_names | {f"next_{name}" for name in task_names}
    missing_task = sorted(task_fields - set(episode_data))
    if missing_task:
        raise ValueError(
            "Discarded PhysX trajectory is missing declared task state: "
            + ", ".join(missing_task)
        )

    selected = {
        str(name): np.ascontiguousarray(np.asarray(value))
        for name, value in episode_data.items()
        if str(name) in _BASE_TRAJECTORY_FIELDS
        or str(name) in task_fields
        or str(name).startswith("physx_")
        or str(name).startswith("next_physx_")
    }
    if not selected:
        return {}
    missing_base = sorted(_BASE_TRAJECTORY_FIELDS - set(selected))
    if missing_base:
        raise ValueError(
            "Discarded PhysX trajectory is missing required field(s): "
            + ", ".join(missing_base)
        )
    lengths: set[int] = set()
    for name, array in selected.items():
        if array.ndim == 0:
            raise ValueError(f"Discarded trajectory field {name!r} has no time axis.")
        if array.dtype == object:
            raise TypeError(
                f"Discarded trajectory field {name!r} has object dtype."
            )
        lengths.add(int(array.shape[0]))
    if len(lengths) != 1:
        raise ValueError(
            "Discarded trajectory fields have different step counts: "
            f"{sorted(lengths)}"
        )
    step_count = next(iter(lengths))
    if step_count <= 0:
        return {}

    present_drive_fields = _RECORDED_DRIVE_FIELDS & set(selected)
    if present_drive_fields and present_drive_fields != _RECORDED_DRIVE_FIELDS:
        missing_drive = sorted(_RECORDED_DRIVE_FIELDS - present_drive_fields)
        raise ValueError(
            "Discarded trajectory has an incomplete articulated-drive path: "
            + ", ".join(missing_drive)
        )
    if not any(
        name.startswith(("physx_", "next_physx_")) for name in selected
    ):
        raise ValueError("Discarded trajectory contains no PhysX state fields.")
    return selected


def _write_compact_segment(
    event_directory: Path,
    episode_data: Mapping[str, np.ndarray],
    *,
    environment_schema: Mapping[str, Any] | None,
) -> None:
    replay_path = event_directory / DISCARDED_REPLAY_NAME
    store = ZarrEpisodeStore.create_empty(
        storage=zarr.DirectoryStore(str(replay_path))
    )
    store.append_episode(dict(episode_data), compressors="disk")
    write_recording_schema(
        event_directory,
        store.root,
        episode_data,
        environment_schema=environment_schema,
    )


class PhysxSessionHistoryJournal:
    """Append-only session journal with atomic event-directory commits."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.header_path = self.path / SESSION_HEADER_NAME
        self.events_path = self.path / EVENTS_DIRECTORY_NAME
        self.finalization_path = self.path / FINALIZATION_NAME
        self.lock_path = self.path / LOCK_NAME
        self._header = self._load_header()

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        session_id: str,
        task_name: str,
        collection_run_seed: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PhysxSessionHistoryJournal":
        session_id = str(session_id).strip()
        task_name = str(task_name).strip()
        if not session_id:
            raise ValueError("session_id must not be empty.")
        if not task_name:
            raise ValueError("task_name must not be empty.")
        if type(collection_run_seed) is not int or collection_run_seed < 0:
            raise ValueError("collection_run_seed must be a nonnegative integer.")

        root = Path(path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        lock_path = root / LOCK_NAME
        with _exclusive_lock(lock_path):
            header_path = root / SESSION_HEADER_NAME
            if header_path.exists():
                raise FileExistsError(f"Session history already exists: {root}")
            unexpected = [
                child
                for child in root.iterdir()
                if child.name != LOCK_NAME
            ]
            if unexpected:
                raise FileExistsError(
                    f"Session history directory is not empty: {root}"
                )
            (root / EVENTS_DIRECTORY_NAME).mkdir()
            _atomic_write_json(
                header_path,
                {
                    "format": SESSION_FORMAT,
                    "version": SESSION_VERSION,
                    "session_id": session_id,
                    "task_name": task_name,
                    "collection_run_seed": collection_run_seed,
                    "created_at": _now_iso(),
                    "metadata": _json_safe(metadata or {}),
                },
            )
            _fsync_directory(root)
        return cls(root)

    @classmethod
    def open(cls, path: str | Path) -> "PhysxSessionHistoryJournal":
        return cls(path)

    @property
    def session_id(self) -> str:
        return str(self._header["session_id"])

    @property
    def task_name(self) -> str:
        return str(self._header["task_name"])

    @property
    def collection_run_seed(self) -> int:
        return int(self._header["collection_run_seed"])

    def _load_header(self) -> dict[str, Any]:
        header = _read_json_object(self.header_path)
        if header.get("format") != SESSION_FORMAT:
            raise ValueError("PhysX session header format is invalid.")
        if header.get("version") != SESSION_VERSION:
            raise ValueError("PhysX session header version is invalid.")
        if not isinstance(header.get("session_id"), str) or not header[
            "session_id"
        ]:
            raise ValueError("PhysX session header has no session_id.")
        if not isinstance(header.get("task_name"), str) or not header[
            "task_name"
        ]:
            raise ValueError("PhysX session header has no task_name.")
        seed = header.get("collection_run_seed")
        if type(seed) is not int or seed < 0:
            raise ValueError("PhysX session header has an invalid collection seed.")
        if not isinstance(header.get("metadata"), Mapping):
            raise ValueError("PhysX session header metadata must be an object.")
        return header

    def _event_directories(self) -> list[Path]:
        if not self.events_path.is_dir():
            raise ValueError(f"Session event directory is missing: {self.events_path}")
        result: list[Path] = []
        removed_temporary = False
        for child in sorted(self.events_path.iterdir(), key=lambda item: item.name):
            prefix, separator, suffix = child.name.partition(".tmp-")
            suffix_parts = suffix.split("-", 1) if separator else []
            owned_temporary = bool(
                child.is_dir()
                and separator
                and prefix.startswith(".")
                and len(prefix) == _EVENT_DIRECTORY_WIDTH + 1
                and prefix[1:].isdigit()
                and len(suffix_parts) == 2
                and suffix_parts[0].isdigit()
                and len(suffix_parts[1]) == 32
                and all(
                    character in "0123456789abcdef"
                    for character in suffix_parts[1]
                )
            )
            if owned_temporary:
                # Every caller scans while holding the journal lock.  Such a
                # directory can only be residue from a writer that died
                # before its atomic rename; it is not a committed event.
                shutil.rmtree(child)
                removed_temporary = True
                continue
            if not child.is_dir() or len(child.name) != _EVENT_DIRECTORY_WIDTH or (
                not child.name.isdigit()
            ):
                raise ValueError(f"Unexpected session event entry: {child}")
            result.append(child)
        if removed_temporary:
            _fsync_directory(self.events_path)
        return result

    def _scan_events(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        events: list[dict[str, Any]] = []
        previous_hash: str | None = None
        next_reset_ordinal = 0
        open_reset_ordinal: int | None = None
        accepted_demos: list[dict[str, Any]] = []
        accepted_demo_ids: set[str] = set()
        last_end_sim_time: float | None = None
        continuity_observations = 0
        continuity_complete = True
        hard_scene_rebuilds = bool(
            self._header.get("metadata", {}).get(
                "in_process_physx_scene_rebuild_per_reset",
                False,
            )
        )

        for expected_ordinal, event_dir in enumerate(self._event_directories()):
            if event_dir.name != f"{expected_ordinal:0{_EVENT_DIRECTORY_WIDTH}d}":
                raise ValueError(
                    "PhysX session event ordinals are not contiguous at "
                    f"{event_dir}."
                )
            event = _read_json_object(event_dir / EVENT_DOCUMENT_NAME)
            if event.get("format") != EVENT_FORMAT or event.get(
                "version"
            ) != EVENT_VERSION:
                raise ValueError(f"Invalid PhysX session event: {event_dir}")
            if event.get("session_id") != self.session_id:
                raise ValueError(f"Session id changed in event {event_dir}.")
            if event.get("ordinal") != expected_ordinal:
                raise ValueError(f"Event ordinal changed in {event_dir}.")
            if event.get("previous_event_sha256") != previous_hash:
                raise ValueError(f"Event hash chain is broken at {event_dir}.")
            payload = event.get("payload")
            if not isinstance(payload, Mapping):
                raise ValueError(f"Event payload must be an object: {event_dir}")

            identity = _tree_identity(event_dir)
            kind = event.get("kind")
            if kind == "reset_attempt":
                reset_ordinal = payload.get("reset_ordinal")
                if reset_ordinal != next_reset_ordinal:
                    raise ValueError(
                        "Reset attempt ordinals are not contiguous: "
                        f"expected {next_reset_ordinal}, got {reset_ordinal}."
                    )
                if open_reset_ordinal is not None:
                    raise ValueError(
                        "An accepted reset was followed by another reset before "
                        "its trajectory was journaled."
                    )
                next_reset_ordinal += 1
                scene_rebuilt = payload.get(
                    "scene_rebuilt_before_reset",
                    False,
                )
                scene_generation = payload.get("scene_generation", 0)
                expected_rebuild = bool(
                    hard_scene_rebuilds and int(reset_ordinal) > 0
                )
                expected_generation = (
                    int(reset_ordinal) if hard_scene_rebuilds else 0
                )
                if type(scene_rebuilt) is not bool or (
                    scene_rebuilt is not expected_rebuild
                ):
                    raise ValueError(
                        "PhysX reset event has an invalid scene-rebuild marker: "
                        f"reset={reset_ordinal}, expected={expected_rebuild}, "
                        f"actual={scene_rebuilt!r}."
                    )
                if type(scene_generation) is not int or (
                    scene_generation != expected_generation
                ):
                    raise ValueError(
                        "PhysX reset event has an invalid scene generation: "
                        f"reset={reset_ordinal}, expected={expected_generation}, "
                        f"actual={scene_generation!r}."
                    )
                if payload.get("outcome") == "accepted":
                    open_reset_ordinal = int(reset_ordinal)
            elif kind == "trajectory":
                reset_ordinal = payload.get("reset_ordinal")
                if reset_ordinal != open_reset_ordinal:
                    raise ValueError(
                        "Trajectory does not follow its accepted reset: "
                        f"reset_ordinal={reset_ordinal}, open={open_reset_ordinal}."
                    )
                step_count = payload.get("step_count")
                if type(step_count) is not int or step_count < 0:
                    raise ValueError(f"Invalid trajectory length in {event_dir}.")
                storage = payload.get("storage")
                if payload.get("accepted") is True:
                    if storage != "accepted_demo_reference" or step_count <= 0:
                        raise ValueError(
                            f"Accepted trajectory reference is invalid: {event_dir}"
                        )
                    demo_id = payload.get("demo_id")
                    if not isinstance(demo_id, str) or not demo_id:
                        raise ValueError(f"Accepted event has no demo_id: {event_dir}")
                    if demo_id in accepted_demo_ids:
                        raise ValueError(f"Duplicate accepted demo id: {demo_id}")
                    accepted_demo_ids.add(demo_id)
                    accepted_demos.append(
                        {
                            "event_ordinal": expected_ordinal,
                            "reset_ordinal": int(reset_ordinal),
                            "demo_id": demo_id,
                            "step_count": step_count,
                        }
                    )
                    if (event_dir / DISCARDED_REPLAY_NAME).exists():
                        raise ValueError(
                            "Accepted demo references must not duplicate replay.zarr."
                        )
                else:
                    expected_storage = (
                        "none" if step_count == 0 else "compact_replay_zarr"
                    )
                    if storage != expected_storage:
                        raise ValueError(
                            f"Discarded trajectory storage is invalid: {event_dir}"
                        )
                    replay_path = event_dir / DISCARDED_REPLAY_NAME
                    schema_path = event_dir / DISCARDED_SCHEMA_NAME
                    if step_count > 0 and (
                        not replay_path.is_dir() or not schema_path.is_file()
                    ):
                        raise ValueError(
                            f"Discarded trajectory artifacts are missing: {event_dir}"
                        )
                    if step_count == 0 and (
                        replay_path.exists() or schema_path.exists()
                    ):
                        raise ValueError(
                            f"Empty trajectory has unexpected artifacts: {event_dir}"
                        )
                open_reset_ordinal = None
            else:
                raise ValueError(f"Unsupported PhysX session event kind: {kind!r}")

            start_time = payload.get("start_sim_time")
            end_time = payload.get("end_sim_time")
            if kind == "reset_attempt" and payload.get(
                "scene_rebuilt_before_reset",
                False,
            ):
                # A new PxScene has its own simulation clock.  The explicit,
                # hash-bound rebuild marker accounts for the discontinuity;
                # no old-scene simulation can occur after this boundary.
                last_end_sim_time = None
            if start_time is None or end_time is None:
                continuity_complete = False
                # Do not compare a later known boundary across an event whose
                # own simulated-time span was not recorded.  The final proof
                # remains explicitly ``unproven`` instead of inventing a gap.
                last_end_sim_time = None
            else:
                if (
                    type(start_time) is bool
                    or type(end_time) is bool
                    or not math.isfinite(float(start_time))
                    or not math.isfinite(float(end_time))
                    or float(end_time) < float(start_time)
                ):
                    raise ValueError(f"Invalid simulation time range in {event_dir}.")
                if last_end_sim_time is not None and not math.isclose(
                    float(start_time),
                    last_end_sim_time,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-9,
                ):
                    raise ValueError(
                        "Unjournaled PhysX time exists between events: "
                        f"previous_end={last_end_sim_time}, "
                        f"next_start={start_time}."
                    )
                last_end_sim_time = float(end_time)
                continuity_observations += 1

            enriched = dict(event)
            enriched["event_sha256"] = identity["sha256"]
            enriched["event_file_count"] = identity["file_count"]
            enriched["event_bytes"] = identity["bytes"]
            enriched["event_directory"] = event_dir.relative_to(
                self.path
            ).as_posix()
            events.append(enriched)
            previous_hash = str(identity["sha256"])

        return events, {
            "next_reset_ordinal": next_reset_ordinal,
            "open_reset_ordinal": open_reset_ordinal,
            "last_event_sha256": previous_hash,
            "accepted_demos": accepted_demos,
            "sim_time_continuity": (
                "complete"
                if events and continuity_complete
                else "unproven"
            ),
            "sim_time_observation_count": continuity_observations,
        }

    def _assert_appendable(self) -> None:
        if self.finalization_path.exists():
            raise RuntimeError("PhysX session history is already finalized.")

    @staticmethod
    def _validate_sim_time_range(
        start_sim_time: float | None,
        end_sim_time: float | None,
    ) -> tuple[float | None, float | None]:
        if (start_sim_time is None) != (end_sim_time is None):
            raise ValueError(
                "start_sim_time and end_sim_time must be supplied together."
            )
        if start_sim_time is None:
            return None, None
        start = float(start_sim_time)
        end = float(end_sim_time)
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Simulation time range must be finite and ordered.")
        return start, end

    def _validate_new_event(
        self,
        kind: str,
        payload: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> None:
        if kind == "reset_attempt":
            if state["open_reset_ordinal"] is not None:
                raise ValueError(
                    "Record the accepted reset's trajectory before the next reset."
                )
            if payload.get("reset_ordinal") != state["next_reset_ordinal"]:
                raise ValueError(
                    "reset ordinal must be the next contiguous value "
                    f"{state['next_reset_ordinal']}."
                )
            return
        if kind == "trajectory":
            if payload.get("reset_ordinal") != state["open_reset_ordinal"]:
                raise ValueError(
                    "Trajectory must immediately follow its accepted reset."
                )
            return
        raise ValueError(f"Unsupported event kind {kind!r}.")

    def _append_event(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        artifact_writer: Callable[[Path], None] | None = None,
    ) -> dict[str, Any]:
        with _exclusive_lock(self.lock_path):
            self._assert_appendable()
            events, state = self._scan_events()
            self._validate_new_event(kind, payload, state)
            ordinal = len(events)
            target = self.events_path / (
                f"{ordinal:0{_EVENT_DIRECTORY_WIDTH}d}"
            )
            temporary = self.events_path / (
                f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
            )
            temporary.mkdir()
            try:
                if artifact_writer is not None:
                    artifact_writer(temporary)
                _atomic_write_json(
                    temporary / EVENT_DOCUMENT_NAME,
                    {
                        "format": EVENT_FORMAT,
                        "version": EVENT_VERSION,
                        "session_id": self.session_id,
                        "ordinal": ordinal,
                        "kind": kind,
                        "previous_event_sha256": state[
                            "last_event_sha256"
                        ],
                        "recorded_at": _now_iso(),
                        "payload": _json_safe(payload),
                    },
                )
                _fsync_directory(temporary)
                os.rename(temporary, target)
                _fsync_directory(self.events_path)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            committed = _read_json_object(target / EVENT_DOCUMENT_NAME)
            identity = _tree_identity(target)
            committed["event_sha256"] = identity["sha256"]
            committed["event_directory"] = target.relative_to(
                self.path
            ).as_posix()
            return committed

    def record_reset_attempt(
        self,
        seed: int,
        ordinal: int,
        outcome: str,
        stable_kwargs: Mapping[str, Any],
        *,
        scene_rebuilt_before_reset: bool = False,
        scene_generation: int = 0,
        scene_rebuild_report: Mapping[str, Any] | None = None,
        stable_origin: Mapping[str, Any] | None = None,
        stability_diagnostics: Mapping[str, Any] | None = None,
        error: BaseException | str | None = None,
        physics_steps: int | None = None,
        start_sim_time: float | None = None,
        end_sim_time: float | None = None,
    ) -> dict[str, Any]:
        """Append one reset/stabilization attempt in source call order."""

        if type(seed) is not int or seed < 0:
            raise ValueError("reset seed must be a nonnegative integer.")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("reset ordinal must be a nonnegative integer.")
        outcome = str(outcome).strip()
        if not outcome:
            raise ValueError("reset outcome must not be empty.")
        if not isinstance(stable_kwargs, Mapping):
            raise TypeError("stable_kwargs must be a mapping.")
        if type(scene_rebuilt_before_reset) is not bool:
            raise TypeError("scene_rebuilt_before_reset must be boolean.")
        if type(scene_generation) is not int or scene_generation < 0:
            raise ValueError("scene_generation must be a nonnegative integer.")
        if scene_rebuild_report is not None and not isinstance(
            scene_rebuild_report,
            Mapping,
        ):
            raise TypeError("scene_rebuild_report must be a mapping or None.")
        if stable_origin is not None and not isinstance(stable_origin, Mapping):
            raise TypeError("stable_origin must be a mapping or None.")
        if stability_diagnostics is not None and not isinstance(
            stability_diagnostics,
            Mapping,
        ):
            raise TypeError("stability_diagnostics must be a mapping or None.")
        if outcome == "accepted" and stable_origin is None:
            raise ValueError("An accepted reset must include stable_origin.")
        if physics_steps is None and stable_origin is not None:
            raw_steps = stable_origin.get("physics_steps")
            if type(raw_steps) is int:
                physics_steps = raw_steps
        if physics_steps is not None and (
            type(physics_steps) is not int or physics_steps < 0
        ):
            raise ValueError("physics_steps must be a nonnegative integer or None.")
        start, end = self._validate_sim_time_range(
            start_sim_time,
            end_sim_time,
        )
        return self._append_event(
            "reset_attempt",
            {
                "reset_ordinal": ordinal,
                "seed": seed,
                "outcome": outcome,
                "accepted": outcome == "accepted",
                "scene_rebuilt_before_reset": scene_rebuilt_before_reset,
                "scene_generation": scene_generation,
                "scene_rebuild_report": _json_safe(scene_rebuild_report),
                "stable_kwargs": _json_safe(stable_kwargs),
                "stable_origin": _json_safe(stable_origin),
                "stability_diagnostics": _json_safe(
                    stability_diagnostics
                ),
                "error": None if error is None else str(error),
                "physics_steps": physics_steps,
                "start_sim_time": start,
                "end_sim_time": end,
            },
        )

    def record_trajectory(
        self,
        trajectory: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        reason: str,
        reset_ordinal: int,
        accepted_demo_dir: str | Path | None = None,
        environment_schema: Mapping[str, Any] | None = None,
        required_fields: Iterable[str] | None = None,
        start_sim_time: float | None = None,
        end_sim_time: float | None = None,
        allow_partial_terminal_guard: bool = False,
    ) -> dict[str, Any]:
        """Append an accepted demo reference or a compact discarded segment."""

        reason = str(reason).strip()
        if not reason:
            raise ValueError("trajectory reason must not be empty.")
        if type(reset_ordinal) is not int or reset_ordinal < 0:
            raise ValueError("reset_ordinal must be a nonnegative integer.")
        if accepted_demo_dir is not None:
            if allow_partial_terminal_guard:
                raise ValueError(
                    "Accepted trajectories cannot contain a partial terminal "
                    "numerical-guard interval."
                )
            step_count, inferred_start, inferred_end = (
                _accepted_trajectory_summary(trajectory)
            )
            demo_dir = Path(accepted_demo_dir).expanduser().resolve()
            if not demo_dir.is_dir() or not (
                demo_dir / DISCARDED_REPLAY_NAME
            ).is_dir():
                raise FileNotFoundError(
                    f"Accepted demo has no replay.zarr: {demo_dir}"
                )
            if step_count <= 0:
                raise ValueError("Accepted demo trajectory must not be empty.")
            start, end = self._validate_sim_time_range(
                start_sim_time,
                end_sim_time,
            )
            if start is None and end is None:
                start, end = inferred_start, inferred_end
            return self._append_event(
                "trajectory",
                {
                    "reset_ordinal": reset_ordinal,
                    "reason": reason,
                    "accepted": True,
                    "demo_id": demo_dir.name,
                    "step_count": step_count,
                    "storage": "accepted_demo_reference",
                    "start_sim_time": start,
                    "end_sim_time": end,
                },
            )

        episode_data = _normalize_episode_data(
            trajectory,
            environment_schema=environment_schema,
            required_fields=required_fields,
            allow_partial_terminal_guard=allow_partial_terminal_guard,
        )
        action = episode_data.get("action")
        step_count = 0 if action is None else int(np.asarray(action).shape[0])

        if step_count == 0:
            start, end = self._validate_sim_time_range(
                start_sim_time,
                end_sim_time,
            )
            return self._append_event(
                "trajectory",
                {
                    "reset_ordinal": reset_ordinal,
                    "reason": reason,
                    "accepted": False,
                    "partial_terminal_guard": bool(
                        allow_partial_terminal_guard
                    ),
                    "demo_id": None,
                    "step_count": 0,
                    "storage": "none",
                    "replay_input_mode": "none",
                    "fields": [],
                    "start_sim_time": start,
                    "end_sim_time": end,
                },
            )

        compact = compact_physx_episode_data(
            episode_data, environment_schema=environment_schema
        )
        if not compact:
            raise ValueError("Nonempty discarded trajectory produced no replay data.")
        compact_steps = int(compact["action"].shape[0])
        if compact_steps != step_count:
            raise ValueError("Compact trajectory step count changed unexpectedly.")
        present_drive_fields = _RECORDED_DRIVE_FIELDS & set(compact)
        replay_input_mode = (
            "recorded_articulated_drive_targets"
            if present_drive_fields == _RECORDED_DRIVE_FIELDS
            else "raw_environment_action"
        )
        start, end = self._validate_sim_time_range(
            start_sim_time,
            end_sim_time,
        )
        if start is None and end is None:
            start, end = _trajectory_sim_time_bounds(compact)

        def write_artifacts(event_directory: Path) -> None:
            _write_compact_segment(
                event_directory,
                compact,
                environment_schema=environment_schema,
            )

        return self._append_event(
            "trajectory",
            {
                "reset_ordinal": reset_ordinal,
                "reason": reason,
                "accepted": False,
                "partial_terminal_guard": bool(
                    allow_partial_terminal_guard
                ),
                "demo_id": None,
                "step_count": compact_steps,
                "storage": "compact_replay_zarr",
                "replay_input_mode": replay_input_mode,
                "fields": sorted(compact),
                "start_sim_time": start,
                "end_sim_time": end,
            },
            artifact_writer=write_artifacts,
        )

    def record_accepted_demo(
        self,
        trajectory: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        demo_dir: str | Path,
        reset_ordinal: int,
        reason: str = "successful_terminal",
        environment_schema: Mapping[str, Any] | None = None,
        required_fields: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Convenience wrapper for an accepted trajectory reference."""

        return self.record_trajectory(
            trajectory,
            reason=reason,
            reset_ordinal=reset_ordinal,
            accepted_demo_dir=demo_dir,
            environment_schema=environment_schema,
            required_fields=required_fields,
        )

    def _history_identity(self) -> dict[str, Any]:
        files: list[tuple[str, Path]] = [
            (SESSION_HEADER_NAME, self.header_path)
        ]
        for event_dir in self._event_directories():
            prefix = event_dir.relative_to(self.path).as_posix()
            files.extend(
                (f"{prefix}/{relative}", file_path)
                for relative, file_path in _regular_files(event_dir)
            )
        return _identity_for_files(self.path, files)

    @staticmethod
    def _uncertified_tail(
        events: Sequence[Mapping[str, Any]],
        state: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        open_reset_ordinal = state.get("open_reset_ordinal")
        if open_reset_ordinal is not None:
            return {
                "kind": "accepted_reset_without_trajectory",
                "reset_ordinal": open_reset_ordinal,
                "event_ordinal": len(events) - 1,
            }
        accepted = state.get("accepted_demos")
        if not isinstance(accepted, list) or not accepted:
            return None
        certified_through = accepted[-1]["event_ordinal"]
        if certified_through >= len(events) - 1:
            return None
        first = int(certified_through) + 1
        last = len(events) - 1
        return {
            "kind": "events_after_certified_prefix",
            "first_event_ordinal": first,
            "last_event_ordinal": last,
            "event_count": last - first + 1,
        }

    def _verify_finalization(
        self,
        finalization: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
        state: Mapping[str, Any],
    ) -> None:
        if finalization.get("format") != FINALIZATION_FORMAT or finalization.get(
            "version"
        ) != FINALIZATION_VERSION:
            raise ValueError("PhysX session finalization format is invalid.")
        if finalization.get("session_id") != self.session_id:
            raise ValueError("PhysX session finalization id does not match.")
        if finalization.get("event_count") != len(events):
            raise ValueError("PhysX session finalization event count changed.")
        event_hashes = [event["event_sha256"] for event in events]
        if finalization.get("event_sha256_chain") != event_hashes:
            raise ValueError("PhysX session finalization event hashes changed.")
        if finalization.get("accepted_demos") != state["accepted_demos"]:
            raise ValueError("PhysX session accepted demo order changed.")
        if finalization.get("uncertified_tail") != self._uncertified_tail(
            events,
            state,
        ):
            raise ValueError("PhysX session uncertified tail changed.")
        if state["accepted_demos"] and (
            state["sim_time_continuity"] != "complete"
            or finalization.get("sim_time_continuity") != "complete"
        ):
            raise ValueError(
                "Finalized PhysX session does not prove complete simulated-"
                "time continuity."
            )
        identity = self._history_identity()
        recorded_identity = finalization.get("history_identity")
        if recorded_identity != identity:
            raise ValueError("PhysX session history identity no longer matches.")

    def finalize(
        self,
        accepted_demos: Iterable[str | Path] | None = None,
    ) -> dict[str, Any]:
        """Seal the complete replay prefix and return its immutable identity."""

        expected_demo_ids = (
            None
            if accepted_demos is None
            else [Path(value).name for value in accepted_demos]
        )
        with _exclusive_lock(self.lock_path):
            events, state = self._scan_events()
            actual_demo_ids = [
                item["demo_id"] for item in state["accepted_demos"]
            ]
            if expected_demo_ids is not None and expected_demo_ids != actual_demo_ids:
                raise ValueError(
                    "Explicit accepted demo order does not match journal events: "
                    f"expected={expected_demo_ids}, actual={actual_demo_ids}."
                )
            if self.finalization_path.exists():
                finalization = _read_json_object(self.finalization_path)
                self._verify_finalization(finalization, events, state)
                return finalization

            if state["accepted_demos"] and (
                state["sim_time_continuity"] != "complete"
            ):
                raise ValueError(
                    "A certifiable PhysX session must account for every "
                    "simulation-time boundary; journal continuity is "
                    f"{state['sim_time_continuity']!r}."
                )

            identity = self._history_identity()
            certified_through_event_ordinal = (
                state["accepted_demos"][-1]["event_ordinal"]
                if state["accepted_demos"]
                else None
            )
            uncertified_tail = self._uncertified_tail(events, state)
            finalization = {
                "format": FINALIZATION_FORMAT,
                "version": FINALIZATION_VERSION,
                "session_id": self.session_id,
                "task_name": self.task_name,
                "collection_run_seed": self.collection_run_seed,
                "finalized_at": _now_iso(),
                "event_count": len(events),
                "event_sha256_chain": [
                    event["event_sha256"] for event in events
                ],
                "accepted_demos": state["accepted_demos"],
                "certified_through_event_ordinal": (
                    certified_through_event_ordinal
                ),
                "uncertified_tail": uncertified_tail,
                "sim_time_continuity": state["sim_time_continuity"],
                "sim_time_observation_count": state[
                    "sim_time_observation_count"
                ],
                "history_identity": identity,
            }
            _atomic_write_json(self.finalization_path, finalization)
            return finalization


def load_session_history(
    path: str | Path,
    *,
    require_finalized: bool = True,
) -> dict[str, Any]:
    """Load and verify an ordered session chain for a replay certifier."""

    journal = PhysxSessionHistoryJournal.open(path)
    with _exclusive_lock(journal.lock_path):
        events, state = journal._scan_events()
        finalization = None
        if journal.finalization_path.exists():
            finalization = _read_json_object(journal.finalization_path)
            journal._verify_finalization(finalization, events, state)
        elif require_finalized:
            raise ValueError("PhysX session history has not been finalized.")
    return {
        "session": dict(journal._header),
        "events": events,
        "finalization": finalization,
    }


def load_discarded_trajectory(
    path: str | Path,
    event_ordinal: int,
    *,
    require_finalized: bool = True,
) -> dict[str, np.ndarray]:
    """Load one compact discarded segment using ordinary replay field names."""

    if type(event_ordinal) is not int or event_ordinal < 0:
        raise ValueError("event_ordinal must be a nonnegative integer.")
    document = load_session_history(path, require_finalized=require_finalized)
    events = document["events"]
    if event_ordinal >= len(events):
        raise IndexError(f"Session event {event_ordinal} does not exist.")
    event = events[event_ordinal]
    payload = event["payload"]
    if event.get("kind") != "trajectory" or payload.get("accepted") is True:
        raise ValueError(f"Session event {event_ordinal} is not discarded.")
    if payload.get("step_count") == 0:
        return {}
    event_dir = Path(path).expanduser().resolve() / event["event_directory"]
    replay = zarr.open_group(
        store=zarr.DirectoryStore(str(event_dir / DISCARDED_REPLAY_NAME)),
        mode="r",
    )
    data = replay["data"]
    return {
        str(name): np.asarray(data[name])
        for name in sorted(data.array_keys())
    }


__all__ = (
    "DISCARDED_REPLAY_NAME",
    "DISCARDED_SCHEMA_NAME",
    "EVENTS_DIRECTORY_NAME",
    "EVENT_DOCUMENT_NAME",
    "EVENT_FORMAT",
    "EVENT_VERSION",
    "FINALIZATION_FORMAT",
    "FINALIZATION_NAME",
    "FINALIZATION_VERSION",
    "PhysxSessionHistoryJournal",
    "SESSION_FORMAT",
    "SESSION_HEADER_NAME",
    "SESSION_VERSION",
    "TREE_IDENTITY_ALGORITHM",
    "compact_physx_episode_data",
    "load_discarded_trajectory",
    "load_session_history",
)
