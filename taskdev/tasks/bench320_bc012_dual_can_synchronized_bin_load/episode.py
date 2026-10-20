"""BC012 time-major Zarr episode and replay validation contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from handson.data.episode_store import ZarrEpisodeStore


TASK_ID = "B320-BC012"
TASK_NAME = "t13_load_cans"
SCHEMA_VERSION = "handson.bc012.physx_episode.v1"
STATE_FIELDS = (
    "tcp_pose",
    "gripper_pose",
    "left_can_pose",
    "right_can_pose",
    "open_bin_pose",
    "partition_centers",
    "bin_load_metrics",
    "bc012_task_state",
    "object_poses",
    "table_delta_height",
)
CAMERA_FIELDS = {
    "image_ego": ("ego", "random_camera"),
    "image_wrist_left": ("wrist_left",),
    "image_wrist_right": ("wrist_right",),
}
REQUIRED_FIELDS = {
    "action",
    "terminal",
    "truncated",
    "succeed",
    *(f"state_{name}" for name in STATE_FIELDS),
    *(f"next_state_{name}" for name in STATE_FIELDS),
    *CAMERA_FIELDS,
}
PHYSX_DRIVE_FIELDS = {
    "physx_substep_count",
    "physx_substep_robot_joint_target",
    "physx_substep_robot_joint_target_valid",
    "physx_robot_joint_target",
    "physx_robot_joint_target_valid",
}


def _numeric_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object or not (
        np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_
    ):
        raise TypeError(f"{name} must be a fixed-shape numeric array")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(array)


def _state(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    state = observation.get("state")
    if not isinstance(state, Mapping):
        raise TypeError("BC012 observation must contain a mapping at 'state'")
    missing = [name for name in STATE_FIELDS if name not in state]
    if missing:
        raise KeyError(f"BC012 observation is missing state fields: {missing}")
    return state


def _camera(
    observation: Mapping[str, Any],
    aliases: Sequence[str],
) -> np.ndarray:
    images = observation.get("images")
    if not isinstance(images, Mapping):
        raise TypeError("BC012 observation must contain the three RGB streams")
    value = next((images[name] for name in aliases if name in images), None)
    if value is None:
        raise KeyError(f"BC012 camera aliases are absent: {tuple(aliases)}")
    frame = _numeric_array("camera", value).astype(np.uint8, copy=False)
    if frame.shape != (480, 640, 3):
        raise ValueError(f"BC012 camera must be 480x640 RGB, got {frame.shape}")
    return frame


def build_episode_data(
    transitions: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Convert complete action transitions into fixed-shape time-major arrays."""

    if not transitions:
        raise ValueError("A BC012 episode must contain at least one transition")
    columns: dict[str, list[np.ndarray]] = {name: [] for name in REQUIRED_FIELDS}
    physical_fields: set[str] | None = None
    for index, transition in enumerate(transitions):
        observation = transition.get("observation")
        next_observation = transition.get("next_observation")
        if not isinstance(observation, Mapping) or not isinstance(
            next_observation, Mapping
        ):
            raise TypeError(f"transition {index} lacks current/next observations")
        state = _state(observation)
        next_state = _state(next_observation)
        columns["action"].append(
            _numeric_array("action", transition.get("action")).astype(
                np.float32, copy=False
            )
        )
        if columns["action"][-1].shape != (54,):
            raise ValueError("BC012 raw environment actions must have shape (54,)")
        for name in STATE_FIELDS:
            columns[f"state_{name}"].append(_numeric_array(name, state[name]))
            columns[f"next_state_{name}"].append(
                _numeric_array(f"next_{name}", next_state[name])
            )
        for output_name, aliases in CAMERA_FIELDS.items():
            columns[output_name].append(_camera(observation, aliases))
        for name in ("terminal", "truncated", "succeed"):
            columns[name].append(
                np.asarray(bool(transition.get(name, False)), dtype=np.uint8)
            )

        current_physical = {
            str(name)
            for name, value in observation.items()
            if str(name).startswith("physx_")
            and np.asarray(value).dtype != object
        }
        next_physical = {
            str(name)
            for name, value in next_observation.items()
            if str(name).startswith("physx_")
            and np.asarray(value).dtype != object
        }
        common = current_physical & next_physical
        if physical_fields is None:
            physical_fields = common
            for name in sorted(common):
                columns[name] = []
                columns[f"next_{name}"] = []
        elif common != physical_fields:
            raise ValueError("PhysX ground-truth fields changed inside the episode")
        for name in sorted(physical_fields or ()):
            columns[name].append(_numeric_array(name, observation[name]))
            columns[f"next_{name}"].append(
                _numeric_array(f"next_{name}", next_observation[name])
            )

    result: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        if not values:
            raise ValueError(f"BC012 episode field {name!r} is empty")
        try:
            result[name] = np.stack(values, axis=0)
        except ValueError as exc:
            raise ValueError(f"BC012 episode field {name!r} changed shape") from exc
    return result


def write_episode(
    zarr_path: Path,
    transitions: Sequence[Mapping[str, Any]],
    *,
    reset_seed: int,
    reset_options: Mapping[str, Any],
    recording_schema: Mapping[str, Any] | None = None,
    source_pid: int | None = None,
) -> dict[str, Any]:
    """Write one immutable action-only validation episode."""

    zarr_path = Path(zarr_path)
    if zarr_path.exists():
        raise FileExistsError(f"Refusing to overwrite BC012 episode: {zarr_path}")
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_episode_data(transitions)
    store = ZarrEpisodeStore.create_empty(
        storage=zarr.DirectoryStore(str(zarr_path))
    )
    attrs = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "runtime_task_name": TASK_NAME,
        "reset_seed": int(reset_seed),
        "reset_options_json": json.dumps(
            dict(reset_options), sort_keys=True, separators=(",", ":")
        ),
        "source_pid": int(os.getpid() if source_pid is None else source_pid),
        "physics_backend": "physx",
        "isaac_sim_version": "5.1",
        "action_contract": "raw_dual_arm_54_action_only",
        "fresh_process_replay_required": True,
        "state_writes_after_action_zero_forbidden": True,
        "camera_shape_hwc": [480, 640, 3],
        "recorded_drive_targets": bool(recording_schema is not None),
    }
    if recording_schema is not None:
        schema = dict(recording_schema)
        if schema.get("name") != "handson.physx_ground_truth":
            raise ValueError("BC012 requires the canonical PhysX recording schema")
        attrs["recording_schema_json"] = json.dumps(
            schema, sort_keys=True, separators=(",", ":")
        )
    for name, value in attrs.items():
        store.root.attrs[name] = value
        store.data.attrs[name] = value
    store.append_episode(data, compressors="disk")
    return validate_episode(zarr_path, require_success=False)


def validate_episode(
    zarr_path: Path,
    *,
    require_success: bool = True,
    require_physical_replay: bool = False,
) -> dict[str, Any]:
    """Fail closed on identity, schema, finiteness, cameras, and terminal truth."""

    root = zarr.open_group(str(Path(zarr_path)), mode="r")
    data = root["data"]
    fields = set(data.array_keys())
    missing = sorted(REQUIRED_FIELDS - fields)
    errors: list[str] = []
    if missing:
        errors.append(f"missing fields: {missing}")
    if root.attrs.get("task_id") != TASK_ID:
        errors.append("task_id is not B320-BC012")
    if root.attrs.get("runtime_task_name") != TASK_NAME:
        errors.append("runtime task name is not canonical")
    if require_physical_replay:
        missing_drive = sorted(
            {
                *PHYSX_DRIVE_FIELDS,
                *(f"next_{name}" for name in PHYSX_DRIVE_FIELDS),
            }
            - fields
        )
        if missing_drive:
            errors.append(f"missing recorded drive-target fields: {missing_drive}")
        raw_schema = root.attrs.get("recording_schema_json")
        try:
            recording_schema = json.loads(raw_schema)
        except (TypeError, json.JSONDecodeError):
            recording_schema = None
        if not isinstance(recording_schema, dict) or recording_schema.get(
            "name"
        ) != "handson.physx_ground_truth":
            errors.append("canonical PhysX recording schema is missing")
    action_shape = tuple(data["action"].shape) if "action" in data else ()
    if len(action_shape) != 2 or action_shape[1:] != (54,):
        errors.append(f"action shape is not (T, 54): {action_shape}")
    length = int(action_shape[0]) if action_shape else 0
    for name in CAMERA_FIELDS:
        if name in data and tuple(data[name].shape) != (length, 480, 640, 3):
            errors.append(f"{name} shape is {tuple(data[name].shape)}")
    for name in fields:
        array = data[name]
        if array.shape and int(array.shape[0]) != length:
            errors.append(f"{name} length differs from action")
        if np.issubdtype(array.dtype, np.floating):
            for start in range(0, length, 32):
                if not np.isfinite(array[start : start + 32]).all():
                    errors.append(f"{name} contains non-finite values")
                    break
    success_count = (
        int(np.count_nonzero(data["succeed"][:])) if "succeed" in data else 0
    )
    if require_success and success_count < 1:
        errors.append("episode contains no physical success transition")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "schema_version": root.attrs.get("schema_version"),
        "task_id": root.attrs.get("task_id"),
        "runtime_task_name": root.attrs.get("runtime_task_name"),
        "steps": length,
        "fields": sorted(fields),
        "physx_field_count": sum(name.startswith("physx_") for name in fields),
        "recorded_drive_targets": bool(root.attrs.get("recorded_drive_targets")),
        "success_count": success_count,
        "camera_shapes": {
            name: list(data[name].shape) for name in CAMERA_FIELDS if name in data
        },
        "pass": True,
    }


__all__ = [
    "CAMERA_FIELDS",
    "PHYSX_DRIVE_FIELDS",
    "REQUIRED_FIELDS",
    "SCHEMA_VERSION",
    "STATE_FIELDS",
    "TASK_ID",
    "TASK_NAME",
    "build_episode_data",
    "validate_episode",
    "write_episode",
]
