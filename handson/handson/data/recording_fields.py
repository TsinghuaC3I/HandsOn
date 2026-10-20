"""Helpers for storing fixed-shape numeric observation telemetry.

The collection scripts intentionally keep camera frames out of ``replay.zarr``.
At the same time, force/contact telemetry is commonly a matrix (contact slots by
features), not a one-dimensional "low dim" vector.  This module defines the
small-array boundary used by both recorders and writes a machine-readable schema
next to, and into, every replay that exposes such fields.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import zarr


MAX_RECORDED_OBSERVATION_NDIM = 3
# YCB04 records four physics substeps × 2048 contact slots × 10 values
# (81920 elements). Keep the numeric telemetry ceiling above that fixed
# payload so rejecting/resetting an episode can seal its physical history.
# Camera-shaped tensors are filtered separately.
MAX_RECORDED_OBSERVATION_ELEMENTS = 131072

WALL_SAMPLING_CONTRACT = "handson.wall_sampling.v1"
RECORDER_MONOTONIC_TIME_KEY = "recorder_monotonic_time"
MIN_WALL_SAMPLING_RATE_RATIO = 0.85
MAX_WALL_SAMPLING_RATE_RATIO = 1.15

_SCHEMA_METHOD_NAMES = (
    "get_force_recording_schema",
    "get_recording_schema",
)
_SCHEMA_ATTRIBUTE_NAMES = (
    "force_recording_schema",
    "recording_schema",
)
_SCHEMA_METADATA_KEYS = frozenset(
    {
        "$schema",
        "description",
        "format",
        "name",
        "version",
    }
)

_REPLAY_ENVIRONMENT_EXACT_NAMES = frozenset(
    {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "HANDSON_ARM_TYPE",
        "GARMENTLAB_ASSET_ROOT",
        "HAND",
        "MKL_NUM_THREADS",
        "NVIDIA_TF32_OVERRIDE",
        "OMP_NUM_THREADS",
        "PYTHONHASHSEED",
    }
)
_REPLAY_ENVIRONMENT_SENSITIVE_TOKENS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
)
_REPLAY_RUNTIME_PACKAGES = (
    "gymnasium",
    "isaaclab",
    "isaacsim",
    "numpy",
    "torch",
    "zarr",
)
_NVIDIA_DRIVER_VERSION_PATH = Path("/proc/driver/nvidia/version")


def _looks_like_image(array: np.ndarray) -> bool:
    """Return whether a small rank-three array still looks like an image."""
    return bool(
        array.ndim == 3
        and array.shape[-1] in (1, 3, 4)
        and array.shape[-3] >= 8
        and array.shape[-2] >= 8
    )


def _as_recordable_numeric_array(
    value: Any,
    *,
    allow_image_shaped: bool = False,
    allow_empty: bool = False,
) -> np.ndarray | None:
    if not isinstance(value, np.ndarray):
        return None
    array = np.asarray(value)
    if not 1 <= array.ndim <= MAX_RECORDED_OBSERVATION_NDIM:
        return None
    if array.size > MAX_RECORDED_OBSERVATION_ELEMENTS:
        return None
    if array.size == 0 and not allow_empty:
        return None
    if array.dtype.kind not in "biuf":
        return None
    if _looks_like_image(array) and not allow_image_shaped:
        return None
    return array


def _declared_field_problem(
    key: str,
    array: np.ndarray,
    field_spec: Mapping[str, Any] | None,
) -> str | None:
    """Return a schema mismatch description for one per-step field value."""
    if not isinstance(field_spec, Mapping):
        return None

    declared_shape = field_spec.get("shape", field_spec.get("shape_per_step"))
    if declared_shape is not None:
        try:
            expected_shape = tuple(int(size) for size in declared_shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Recording schema field {key!r} has invalid shape "
                f"{declared_shape!r}."
            ) from exc
        if array.shape != expected_shape:
            return (
                f"has shape {array.shape}, but the environment schema declares "
                f"{expected_shape}"
            )

    declared_dtype = field_spec.get("dtype")
    if declared_dtype is not None:
        try:
            expected_dtype = np.dtype(declared_dtype)
        except TypeError as exc:
            raise ValueError(
                f"Recording schema field {key!r} has invalid dtype "
                f"{declared_dtype!r}."
            ) from exc
        if array.dtype != expected_dtype:
            return (
                f"has dtype {array.dtype}, but the environment schema declares "
                f"{expected_dtype}"
            )
    return None


def select_next_observation_for_recording(
    observation: Mapping[str, Any],
    *,
    required_fields: Iterable[str] = (),
) -> dict[str, Any]:
    """Select only transition targets needed by the replay writer.

    RGB observations remain on the current side of each transition for video
    output.  Keeping only state plus schema-declared fields on the next side
    avoids duplicating every camera frame in the in-memory trajectory.
    Values are intentionally not copied here; the recorder deep-copies the
    compact transition once after selection.
    """
    if not isinstance(observation, Mapping):
        raise TypeError("next_observation must be a mapping.")

    required = {str(key) for key in required_fields}
    missing = sorted(key for key in required if key not in observation)
    if missing:
        raise ValueError(
            "Required recording field(s) missing from next_observation: "
            f"{missing}."
        )

    selected_keys = {"state", "low_dim"} | required
    return {
        key: observation[key]
        for key in selected_keys
        if key in observation
    }


def select_current_observation_for_recording(
    observation: Mapping[str, Any],
    *,
    required_fields: Iterable[str] = (),
    include_required_fields: bool,
) -> dict[str, Any]:
    """Keep current images/state while avoiding duplicate force snapshots.

    Except for the initial observation, ``force[t]`` is exactly the preceding
    transition's stored ``next_force[t-1]``.  Omitting declared telemetry from
    later current observations nearly halves trajectory RAM; the stacker
    reconstructs the standard current/next arrays before writing Zarr.
    """
    if not isinstance(observation, Mapping):
        raise TypeError("current_observation must be a mapping.")
    required = {str(key) for key in required_fields}
    if include_required_fields:
        missing = sorted(key for key in required if key not in observation)
        if missing:
            raise ValueError(
                "Required recording field(s) missing from initial observation: "
                f"{missing}."
            )
        return dict(observation)
    return {
        str(key): value
        for key, value in observation.items()
        if str(key) not in required
    }


def transition_timestamps(
    trajectory: Iterable[Mapping[str, Any]],
    *,
    fallback_hz: float,
) -> np.ndarray:
    """Return real per-transition info timestamps, or a fixed-rate fallback."""
    steps = list(trajectory)
    if fallback_hz <= 0:
        raise ValueError(f"fallback_hz must be positive, got {fallback_hz}.")

    values: list[float] = []
    for step in steps:
        info = step.get("infos") or step.get("info") or {}
        if not isinstance(info, Mapping):
            values = []
            break
        raw_value = info.get("timestamp", info.get("time"))
        if raw_value is None:
            values = []
            break
        array = np.asarray(raw_value)
        if array.size != 1 or array.dtype.kind not in "iuf":
            values = []
            break
        value = float(array.reshape(-1)[0])
        if not np.isfinite(value):
            values = []
            break
        values.append(value)

    if len(values) == len(steps):
        return np.asarray(values, dtype=np.float64)
    return np.arange(len(steps), dtype=np.float64) / float(fallback_hz)


def transition_monotonic_timestamps(
    trajectory: Iterable[Mapping[str, Any]],
    *,
    key: str = RECORDER_MONOTONIC_TIME_KEY,
) -> np.ndarray | None:
    """Return a zero-based recorder wall clock, or ``None`` for legacy data.

    Once any transition declares the clock, every transition must provide one
    finite, strictly increasing scalar.  A partial or malformed clock is data
    corruption and must not silently fall back to an advertised fixed rate.
    """

    steps = list(trajectory)
    raw_values = [step.get(key) for step in steps]
    present = [value is not None for value in raw_values]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            f"Transition wall clock {key!r} is missing from part of the trajectory."
        )

    values: list[float] = []
    for index, raw_value in enumerate(raw_values):
        array = np.asarray(raw_value)
        if array.size != 1 or array.dtype.kind not in "iuf":
            raise ValueError(
                f"Transition wall clock {key!r} at index {index} is not numeric."
            )
        value = float(array.reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError(
                f"Transition wall clock {key!r} at index {index} is not finite."
            )
        values.append(value)

    timestamps = np.asarray(values, dtype=np.float64)
    if timestamps.size >= 2 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(
            f"Transition wall clock {key!r} must be strictly increasing."
        )
    if timestamps.size:
        timestamps = timestamps - timestamps[0]
    return timestamps


def effective_sampling_hz(
    timestamps: np.ndarray,
    *,
    fallback_hz: float,
) -> float:
    """Return the median sampling frequency of a monotonic time axis."""

    if fallback_hz <= 0.0 or not np.isfinite(fallback_hz):
        raise ValueError(f"fallback_hz must be finite and positive, got {fallback_hz}.")
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return float(fallback_hz)
    intervals = np.diff(values)
    if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0.0):
        raise ValueError("Recorded timestamps must be finite and strictly increasing.")
    return float(1.0 / np.median(intervals))


def wall_sampling_contract(
    timestamps: np.ndarray,
    *,
    target_hz: float,
) -> dict[str, Any]:
    """Summarize and qualify the real recorder cadence for publication."""

    if target_hz <= 0.0 or not np.isfinite(target_hz):
        raise ValueError(f"target_hz must be finite and positive, got {target_hz}.")
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    lower_hz = float(target_hz) * MIN_WALL_SAMPLING_RATE_RATIO
    upper_hz = float(target_hz) * MAX_WALL_SAMPLING_RATE_RATIO
    summary: dict[str, Any] = {
        "contract": WALL_SAMPLING_CONTRACT,
        "clock": "time.monotonic",
        "sample_count": int(values.size),
        "interval_count": max(0, int(values.size) - 1),
        "target_hz": float(target_hz),
        "minimum_qualified_hz": lower_hz,
        "maximum_qualified_hz": upper_hz,
        "qualified": False,
    }
    if values.size < 2:
        summary["reason"] = "insufficient_samples"
        return summary
    intervals = np.diff(values)
    if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0.0):
        summary["reason"] = "non_monotonic_clock"
        return summary

    elapsed = float(values[-1] - values[0])
    mean_hz = float((values.size - 1) / elapsed)
    median_interval = float(np.median(intervals))
    median_hz = float(1.0 / median_interval)
    p95_interval = float(np.percentile(intervals, 95.0))
    summary.update(
        {
            "elapsed_seconds": elapsed,
            "mean_hz": mean_hz,
            "median_hz": median_hz,
            "median_interval_seconds": median_interval,
            "p95_interval_seconds": p95_interval,
            "maximum_interval_seconds": float(np.max(intervals)),
        }
    )
    qualified = bool(
        lower_hz <= mean_hz <= upper_hz
        and lower_hz <= median_hz <= upper_hz
        and p95_interval <= 1.0 / lower_hz
    )
    summary["qualified"] = qualified
    summary["reason"] = "qualified" if qualified else "cadence_out_of_range"
    return summary


def wall_sampling_contract_errors(payload: Any) -> list[str]:
    """Return fail-closed errors for a recorded wall-sampling declaration."""

    if not isinstance(payload, Mapping):
        return ["wall sampling contract must be an object"]
    errors: list[str] = []
    if payload.get("contract") != WALL_SAMPLING_CONTRACT:
        errors.append("wall sampling contract identifier is invalid")
    if payload.get("clock") != "time.monotonic":
        errors.append("wall sampling clock must be time.monotonic")
    for field in ("sample_count", "interval_count"):
        value = payload.get(field)
        if type(value) is not int or value < 0:
            errors.append(f"wall sampling {field} must be a non-negative integer")
    for field in (
        "target_hz",
        "minimum_qualified_hz",
        "maximum_qualified_hz",
    ):
        value = payload.get(field)
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) <= 0.0
        ):
            errors.append(f"wall sampling {field} must be finite and positive")
    sample_count = payload.get("sample_count")
    interval_count = payload.get("interval_count")
    if type(sample_count) is int and type(interval_count) is int:
        if interval_count != max(0, sample_count - 1):
            errors.append("wall sampling interval_count must equal sample_count - 1")
        if sample_count < 2:
            errors.append("wall sampling needs at least two samples")
    for field in (
        "elapsed_seconds",
        "mean_hz",
        "median_hz",
        "median_interval_seconds",
        "p95_interval_seconds",
        "maximum_interval_seconds",
    ):
        value = payload.get(field)
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) <= 0.0
        ):
            errors.append(f"wall sampling {field} must be finite and positive")
    target_hz = payload.get("target_hz")
    minimum_hz = payload.get("minimum_qualified_hz")
    maximum_hz = payload.get("maximum_qualified_hz")
    if all(type(value) in {int, float} for value in (target_hz, minimum_hz, maximum_hz)):
        if not np.isclose(
            float(minimum_hz),
            float(target_hz) * MIN_WALL_SAMPLING_RATE_RATIO,
            rtol=0.0,
            atol=1e-9,
        ):
            errors.append("wall sampling minimum qualification bound is invalid")
        if not np.isclose(
            float(maximum_hz),
            float(target_hz) * MAX_WALL_SAMPLING_RATE_RATIO,
            rtol=0.0,
            atol=1e-9,
        ):
            errors.append("wall sampling maximum qualification bound is invalid")
    if payload.get("qualified") is not True:
        errors.append("wall sampling cadence is not production-qualified")
    return errors


def stack_numeric_observation_fields(
    trajectory: Iterable[Mapping[str, Any]],
    *,
    reserved_keys: Iterable[str] = ("state", "low_dim"),
    required_fields: Iterable[str] = (),
    field_specs: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Stack consistent small numeric observation fields along a time axis.

    Fields supplied on every step with one fixed per-step shape are returned as
    ``(T, ...)`` arrays. Camera-like tensors, strings, objects, and oversized
    arrays are excluded. A field declared by the environment schema is strict:
    missing, nonnumeric, or changing values raise instead of silently producing
    an incomplete training signal. Undeclared inconsistent fields are skipped
    with a warning so legacy tasks remain recordable.
    """
    steps = list(trajectory)
    if not steps:
        return {}

    observations = [step.get("observations", {}) for step in steps]
    if not all(isinstance(obs, Mapping) for obs in observations):
        raise TypeError("Every trajectory step must contain a mapping observation.")

    reserved = set(reserved_keys)
    required = set(required_fields) - reserved
    observed_keys: set[str] = set()
    for obs in observations:
        observed_keys.update(str(key) for key in obs)

    result: dict[str, np.ndarray] = {}
    for key in sorted(observed_keys | required):
        if key in reserved:
            continue

        values = []
        problem = None
        for step_idx, obs in enumerate(observations):
            if key not in obs:
                problem = f"is missing at trajectory step {step_idx}"
                break
            array = _as_recordable_numeric_array(
                obs[key],
                allow_image_shaped=key in required,
                allow_empty=key in required,
            )
            if array is None:
                problem = (
                    f"is not a supported fixed-shape numeric array at trajectory "
                    f"step {step_idx}"
                )
                break
            schema_problem = _declared_field_problem(
                key,
                array,
                (field_specs or {}).get(key),
            )
            if schema_problem is not None:
                problem = f"{schema_problem} at trajectory step {step_idx}"
                break
            values.append(array)

        if problem is None:
            expected_shape = values[0].shape
            mismatch = next(
                (
                    (idx, value.shape)
                    for idx, value in enumerate(values[1:], start=1)
                    if value.shape != expected_shape
                ),
                None,
            )
            if mismatch is not None:
                problem = (
                    f"changes shape from {expected_shape} to {mismatch[1]} at "
                    f"trajectory step {mismatch[0]}"
                )

        if problem is not None:
            if key in required:
                raise ValueError(f"Required recording field {key!r} {problem}.")
            # Do not warn for ordinary images/objects that were never candidates.
            if values or _as_recordable_numeric_array(
                observations[0].get(key),
                allow_image_shaped=key in required,
                allow_empty=key in required,
            ) is not None:
                warnings.warn(
                    f"Skipping inconsistent observation field {key!r}: {problem}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue

        result[key] = np.ascontiguousarray(np.stack(values, axis=0))

    return result


def _validate_substep_completeness(
    fields: Mapping[str, np.ndarray],
    *,
    n_substeps: int,
) -> None:
    """Reject force frames that omit any expected MuJoCo physics substep."""
    if n_substeps <= 0:
        raise ValueError(
            f"Recording schema declares invalid mujoco.n_substeps={n_substeps}."
        )

    for prefix, allow_initial_reset in (("", True), ("next_", False)):
        count_name = f"{prefix}c01_force_substep_count"
        mask_name = f"{prefix}c01_force_substep_mask"
        if count_name not in fields and mask_name not in fields:
            continue
        if count_name not in fields or mask_name not in fields:
            raise ValueError(
                f"Force telemetry requires both {count_name!r} and {mask_name!r}."
            )

        counts_array = np.asarray(fields[count_name])
        masks = np.asarray(fields[mask_name])
        if counts_array.ndim != 2 or counts_array.shape[1:] != (1,):
            raise ValueError(
                f"Force telemetry {count_name!r} must have stacked shape (T, 1), "
                f"got {counts_array.shape}."
            )
        if masks.ndim != 2 or masks.shape[1:] != (n_substeps,):
            raise ValueError(
                f"Force telemetry {mask_name!r} must have stacked shape "
                f"(T, {n_substeps}), got {masks.shape}."
            )
        if masks.shape[0] != counts_array.shape[0]:
            raise ValueError(
                f"Force telemetry {count_name!r} and {mask_name!r} have "
                "different time lengths."
            )
        if not np.all((masks == 0) | (masks == 1)):
            raise ValueError(f"Force telemetry {mask_name!r} must be binary.")

        counts = counts_array[:, 0].astype(np.int64, copy=False)
        mask_counts = masks.astype(np.int64, copy=False).sum(axis=1)
        mismatch = np.flatnonzero(mask_counts != counts)
        if mismatch.size:
            index = int(mismatch[0])
            raise ValueError(
                f"Force telemetry {mask_name!r} sum {int(mask_counts[index])} "
                f"does not match {count_name!r}={int(counts[index])} at "
                f"trajectory step {index}."
            )

        for step_idx, count in enumerate(counts):
            is_initial_reset = (
                allow_initial_reset
                and step_idx == 0
                and int(count) == 0
                and not np.any(masks[step_idx])
            )
            if is_initial_reset:
                continue
            if int(count) != n_substeps or not np.all(masks[step_idx] == 1):
                raise ValueError(
                    f"Incomplete force substep telemetry in {prefix or 'current_'}"
                    f"observation at trajectory step {step_idx}: expected "
                    f"count={n_substeps} and a contiguous all-one mask."
                )


_PHYSX_V4_SUBSTEP_FIELDS = frozenset(
    {
        "physx_substep_count",
        "physx_substep_robot_joint_position",
        "physx_substep_robot_joint_velocity",
        "physx_substep_robot_joint_target",
        "physx_substep_robot_joint_valid",
        "physx_substep_robot_joint_target_valid",
    }
)


def _validate_physx_v4_substep_trajectory(
    steps: list[Mapping[str, Any]],
    *,
    environment_schema: Mapping[str, Any] | None,
    allow_partial_terminal_guard: bool = False,
) -> None:
    """Reject PhysX recordings that cannot supply an exact robot substep path."""
    if not environment_schema:
        return

    declared = _declared_fields(environment_schema)
    declared_names = {str(name) for name in declared}
    has_physx_fields = any(name.startswith("physx_") for name in declared_names)
    declared_contract = _PHYSX_V4_SUBSTEP_FIELDS & declared_names

    raw_version = environment_schema.get(
        "version",
        environment_schema.get("schema_version"),
    )
    try:
        schema_version = int(raw_version) if raw_version is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"PhysX recording schema has invalid version {raw_version!r}."
        ) from exc

    # Legacy partial schemas in downstream tests and non-PhysX environments do
    # not opt into this contract.  A versioned PhysX schema does: old collectors
    # must fail closed instead of silently producing another unreplayable episode.
    if not declared_contract:
        if has_physx_fields and schema_version is not None:
            if schema_version < 4:
                raise ValueError(
                    "PhysX physical replay recording requires schema v4 or newer; "
                    f"got v{schema_version}."
                )
            missing = sorted(_PHYSX_V4_SUBSTEP_FIELDS)
            raise ValueError(
                "PhysX v4 recording schema is missing required robot substep "
                f"field(s): {missing}."
            )
        return

    if schema_version is None or schema_version < 4:
        raise ValueError(
            "PhysX robot substep fields require a versioned v4-or-newer "
            f"recording schema; got {raw_version!r}."
        )

    missing_schema_fields = sorted(
        _PHYSX_V4_SUBSTEP_FIELDS - declared_names
    )
    if missing_schema_fields:
        raise ValueError(
            "PhysX v4 recording schema is missing required robot substep "
            f"field(s): {missing_schema_fields}."
        )

    field_shapes: dict[str, tuple[int, ...]] = {}
    for name in _PHYSX_V4_SUBSTEP_FIELDS:
        shape = declared[name].get("shape")
        if shape is None:
            raise ValueError(
                f"PhysX v4 recording schema field {name!r} is missing shape."
            )
        try:
            field_shapes[name] = tuple(int(size) for size in shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"PhysX v4 recording schema field {name!r} has invalid "
                f"shape {shape!r}."
            ) from exc

    path_shape = field_shapes["physx_substep_robot_joint_position"]
    if len(path_shape) != 3 or path_shape[0] <= 0:
        raise ValueError(
            "PhysX v4 robot substep fields must have schema shape "
            f"(S, robots, joints) with S > 0; got {path_shape}."
        )
    n_substeps = path_shape[0]
    for name in (
        "physx_substep_robot_joint_velocity",
        "physx_substep_robot_joint_target",
        "physx_substep_robot_joint_valid",
        "physx_substep_robot_joint_target_valid",
    ):
        if field_shapes[name] != path_shape:
            raise ValueError(
                f"PhysX v4 recording schema field {name!r} has shape "
                f"{field_shapes[name]}, expected {path_shape}."
            )
    if field_shapes["physx_substep_count"] != (1,):
        raise ValueError(
            "PhysX v4 recording schema field 'physx_substep_count' must "
            f"have shape (1,), got {field_shapes['physx_substep_count']}."
        )

    def observation_field(
        step_index: int,
        side: str,
        name: str,
    ) -> Any:
        observation = steps[step_index].get(side)
        if not isinstance(observation, Mapping):
            raise ValueError(
                f"PhysX v4 recording requires mapping {side} at trajectory "
                f"step {step_index}."
            )
        if name in observation:
            return observation[name]
        if side == "observations" and step_index > 0:
            previous_next = steps[step_index - 1].get("next_observations")
            if isinstance(previous_next, Mapping) and name in previous_next:
                return previous_next[name]
        raise ValueError(
            f"PhysX v4 recording field {name!r} is missing from {side} "
            f"at trajectory step {step_index}."
        )

    for side in ("observations", "next_observations"):
        for step_index in range(len(steps)):
            arrays: dict[str, np.ndarray] = {}
            for name in _PHYSX_V4_SUBSTEP_FIELDS:
                array = np.asarray(
                    observation_field(step_index, side, name)
                )
                expected_shape = field_shapes[name]
                if array.shape != expected_shape:
                    raise ValueError(
                        f"PhysX v4 recording field {name!r} in {side} at "
                        f"trajectory step {step_index} has shape {array.shape}, "
                        f"expected {expected_shape}."
                    )
                arrays[name] = array

            count_array = arrays["physx_substep_count"]
            if not np.issubdtype(count_array.dtype, np.integer):
                raise ValueError(
                    "PhysX v4 substep count must have integer dtype in "
                    f"{side} at trajectory step {step_index}."
                )
            count = int(count_array[0])
            initial_reset = (
                side == "observations" and step_index == 0 and count == 0
            )
            step_info = steps[step_index].get(
                "infos",
                steps[step_index].get("info", {}),
            )
            partial_terminal_guard = bool(
                allow_partial_terminal_guard
                and side == "next_observations"
                and step_index == len(steps) - 1
                and 0 < count < n_substeps
                and isinstance(step_info, Mapping)
                and step_info.get("manual_reset") is True
                and isinstance(
                    step_info.get("numerical_guard_event"),
                    Mapping,
                )
            )
            if (
                not initial_reset
                and not partial_terminal_guard
                and count != n_substeps
            ):
                raise ValueError(
                    "Incomplete PhysX v4 robot substep telemetry in "
                    f"{side} at trajectory step {step_index}: expected "
                    f"count={n_substeps}, got {count}."
                )

            joint_valid = arrays[
                "physx_substep_robot_joint_valid"
            ]
            target_valid = arrays[
                "physx_substep_robot_joint_target_valid"
            ]
            for name, valid in (
                ("physx_substep_robot_joint_valid", joint_valid),
                ("physx_substep_robot_joint_target_valid", target_valid),
            ):
                if not np.all((valid == 0) | (valid == 1)):
                    raise ValueError(
                        f"PhysX v4 validity field {name!r} must be binary in "
                        f"{side} at trajectory step {step_index}."
                    )

            if initial_reset:
                if np.any(joint_valid) or np.any(target_valid):
                    raise ValueError(
                        "PhysX v4 initial reset substep validity masks must be "
                        "all zero when count=0."
                    )
                continue

            joint_mask = joint_valid.astype(bool, copy=False)
            target_mask = target_valid.astype(bool, copy=False)
            active_slice = slice(0, count)
            padding_slice = slice(count, n_substeps)
            active_joint_mask = joint_mask[active_slice]
            active_target_mask = target_mask[active_slice]
            if partial_terminal_guard:
                if np.any(joint_mask[padding_slice]) or np.any(
                    target_mask[padding_slice]
                ):
                    raise ValueError(
                        "Partial terminal PhysX guard telemetry has non-zero "
                        f"validity padding in {side} at trajectory step "
                        f"{step_index}."
                    )
                for name, field in declared.items():
                    if not str(name).startswith("physx_substep_"):
                        continue
                    raw_shape = field.get("shape")
                    try:
                        shape = tuple(int(size) for size in raw_shape)
                    except (TypeError, ValueError):
                        continue
                    if not shape or shape[0] != n_substeps:
                        continue
                    padded = np.asarray(
                        observation_field(step_index, side, str(name))
                    )
                    if padded.shape != shape:
                        raise ValueError(
                            f"PhysX v4 recording field {name!r} in {side} at "
                            f"trajectory step {step_index} has shape "
                            f"{padded.shape}, expected {shape}."
                        )
                    if np.any(padded[padding_slice] != 0):
                        raise ValueError(
                            "Partial terminal PhysX guard telemetry has "
                            f"non-zero padding in field {name!r} at "
                            f"trajectory step {step_index}."
                        )
            if not np.all(np.any(active_joint_mask, axis=(1, 2))):
                raise ValueError(
                    "PhysX v4 robot_joint_valid does not cover at least one "
                    f"joint in every substep of {side} at trajectory step "
                    f"{step_index}."
                )
            if not np.array_equal(active_joint_mask, active_target_mask):
                raise ValueError(
                    "PhysX v4 robot_joint_target_valid must exactly cover "
                    "robot_joint_valid in "
                    f"{side} at trajectory step {step_index}."
                )
            if not np.all(active_joint_mask == active_joint_mask[0]):
                raise ValueError(
                    "PhysX v4 robot_joint_valid changes within a control step "
                    f"in {side} at trajectory step {step_index}."
                )

            for name in (
                "physx_substep_robot_joint_position",
                "physx_substep_robot_joint_velocity",
            ):
                if not np.all(
                    np.isfinite(arrays[name][active_slice][active_joint_mask])
                ):
                    raise ValueError(
                        f"PhysX v4 recording field {name!r} contains non-finite "
                        f"valid values in {side} at trajectory step {step_index}."
                    )
            target = arrays["physx_substep_robot_joint_target"]
            if not np.all(
                np.isfinite(target[active_slice][active_target_mask])
            ):
                raise ValueError(
                    "PhysX v4 recording field "
                    "'physx_substep_robot_joint_target' contains non-finite "
                    f"valid values in {side} at trajectory step {step_index}."
                )


def stack_transition_numeric_observation_fields(
    trajectory: Iterable[Mapping[str, Any]],
    *,
    required_fields: Iterable[str] = (),
    environment_schema: Mapping[str, Any] | None = None,
    allow_partial_terminal_guard: bool = False,
) -> dict[str, np.ndarray]:
    """Stack observation telemetry with explicit action-to-next alignment.

    Current fields keep their observation names.  Schema-declared fields from
    ``next_observations`` are stored with a ``next_`` prefix, making it
    unambiguous that ``action[t]`` produced ``next_<field>[t]``.  Requiring the
    next observation prevents a force-aware recorder from silently falling
    back to the one-frame-shifted legacy layout.
    """
    steps = list(trajectory)
    _validate_physx_v4_substep_trajectory(
        steps,
        environment_schema=environment_schema,
        allow_partial_terminal_guard=allow_partial_terminal_guard,
    )
    declared = _declared_fields(environment_schema)
    required = set(required_fields) | {str(key) for key in declared}
    has_next_observations = all(
        isinstance(step.get("next_observations"), Mapping) for step in steps
    )
    if required and not has_next_observations:
        raise ValueError(
            "Force-recording schema requires next_observations so action_t can "
            "be paired with force/state at t+1."
        )

    # Recorders keep the reset/current force snapshot once, then rely on the
    # preceding transition's next snapshot.  Reconstruct a virtual full
    # current observation sequence without copying the large numeric arrays.
    current_steps: list[dict[str, Any]] = []
    for step_idx, step in enumerate(steps):
        observation = step.get("observations", {})
        if not isinstance(observation, Mapping):
            raise TypeError(
                "Every trajectory step must contain a mapping observation."
            )
        reconstructed = dict(observation)
        if required and step_idx > 0:
            previous_next = steps[step_idx - 1]["next_observations"]
            for field_name in required:
                if field_name not in reconstructed and field_name in previous_next:
                    reconstructed[field_name] = previous_next[field_name]
        current_steps.append({"observations": reconstructed})

    result = stack_numeric_observation_fields(
        current_steps,
        required_fields=required,
        field_specs=declared,
    )
    if not required:
        return result

    next_trajectory = [
        {"observations": step["next_observations"]}
        for step in steps
    ]
    next_fields = stack_numeric_observation_fields(
        next_trajectory,
        required_fields=required,
        field_specs=declared,
    )
    result.update(
        {
            f"next_{name}": value
            for name, value in next_fields.items()
            if name in required
        }
    )
    if environment_schema is not None:
        mujoco_schema = environment_schema.get("mujoco")
        if isinstance(mujoco_schema, Mapping) and "n_substeps" in mujoco_schema:
            try:
                n_substeps = int(mujoco_schema["n_substeps"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Recording schema mujoco.n_substeps must be an integer."
                ) from exc
            _validate_substep_completeness(result, n_substeps=n_substeps)
    for overflow_name in (
        "c01_force_contact_overflow",
        "c01_force_substep_overflow",
        "next_c01_force_contact_overflow",
        "next_c01_force_substep_overflow",
    ):
        overflow = result.get(overflow_name)
        if overflow is not None and np.any(np.asarray(overflow) > 0):
            raise ValueError(
                f"Force telemetry {overflow_name!r} is nonzero; refusing to save "
                "an incomplete contact/substep record."
            )
    return result


def _iter_environment_chain(env: Any):
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        try:
            current = getattr(current, "env", None)
        except Exception:
            current = None


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
        "Recording schema values must be JSON-compatible; "
        f"got {type(value).__name__}."
    )


def resolve_environment_recording_schema(env: Any) -> dict[str, Any] | None:
    """Read an optional recording schema from an environment/wrapper chain."""
    if env is None:
        return None

    for candidate in _iter_environment_chain(env):
        for name in _SCHEMA_METHOD_NAMES:
            provider = getattr(candidate, name, None)
            if callable(provider):
                schema = provider()
                if schema is None:
                    continue
                if not isinstance(schema, Mapping):
                    raise TypeError(f"{name}() must return a mapping or None.")
                return _json_safe(schema)
        for name in _SCHEMA_ATTRIBUTE_NAMES:
            schema = getattr(candidate, name, None)
            if schema is None:
                continue
            if not isinstance(schema, Mapping):
                raise TypeError(f"{name} must be a mapping or None.")
            return _json_safe(schema)
    return None


def declared_recording_field_names(
    environment_schema: Mapping[str, Any] | None,
) -> set[str]:
    """Extract observation field names from either supported schema layout."""
    if not environment_schema:
        return set()
    fields = environment_schema.get("fields")
    if isinstance(fields, Mapping):
        return {str(key) for key in fields}
    return {
        str(key)
        for key, value in environment_schema.items()
        if key not in _SCHEMA_METADATA_KEYS and isinstance(value, Mapping)
    }


def _declared_fields(
    environment_schema: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not environment_schema:
        return {}
    fields = environment_schema.get("fields")
    if isinstance(fields, Mapping):
        return fields
    return {
        key: value
        for key, value in environment_schema.items()
        if key not in _SCHEMA_METADATA_KEYS and isinstance(value, Mapping)
    }


def build_recording_schema(
    episode_data: Mapping[str, np.ndarray],
    *,
    environment_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the self-describing schema stored with a replay episode."""
    declared = _declared_fields(environment_schema)
    fields: dict[str, Any] = {}
    for name, value in episode_data.items():
        array = np.asarray(value)
        if array.ndim == 0:
            raise ValueError(f"Recording field {name!r} must have a time axis.")
        entry: dict[str, Any] = {
            "dtype": str(array.dtype),
            "shape_per_step": list(array.shape[1:]),
            "time_axis": 0,
        }
        semantic_name = name[5:] if name.startswith("next_") else name
        if semantic_name in declared:
            per_step = array[0] if array.shape[0] else np.empty(
                array.shape[1:], dtype=array.dtype
            )
            problem = _declared_field_problem(
                semantic_name,
                per_step,
                declared[semantic_name],
            )
            if problem is not None:
                raise ValueError(
                    f"Recorded field {name!r} conflicts with the environment "
                    f"schema: {problem}."
                )
        if name == "action":
            entry["source"] = "action"
        elif name == "action_rotvec":
            entry["source"] = "derived_action"
        elif name == "terminal":
            entry["source"] = "transition_done"
        elif name == "succeed":
            entry["source"] = "transition_info"
        elif name in ("timestamp", "next_timestamp"):
            entry["source"] = "recorder"
            entry["units"] = "s"
        elif name.startswith("next_"):
            entry["source"] = "next_observation"
        else:
            entry["source"] = "observation"
        if semantic_name in declared:
            entry["semantics"] = _json_safe(declared[semantic_name])
        fields[str(name)] = entry

    document: dict[str, Any] = {
        "format": "handson.replay.recording_schema",
        "version": 1,
        "fields": fields,
    }
    if environment_schema is not None:
        document["environment_schema"] = _json_safe(environment_schema)
    return document


def write_recording_schema(
    demo_dir: str | Path,
    zarr_root: zarr.Group,
    episode_data: Mapping[str, np.ndarray],
    *,
    environment_schema: Mapping[str, Any] | None = None,
) -> Path:
    """Persist schema as a JSON sidecar plus Zarr group/dataset attributes."""
    document = build_recording_schema(
        episode_data,
        environment_schema=environment_schema,
    )
    demo_path = Path(demo_dir)
    schema_path = demo_path / "recording_schema.json"
    temporary_path = demo_path / ".recording_schema.json.tmp"
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary_path, schema_path)

    zarr_root.attrs["recording_schema"] = document
    zarr_root.attrs["recording_schema_path"] = "../recording_schema.json"
    data_group = zarr_root.get("data")
    if data_group is not None:
        for field_name, metadata in document["fields"].items():
            if field_name in data_group:
                data_group[field_name].attrs.update(metadata)
    return schema_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write one JSON document without exposing a partially written file."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(
                _json_safe(document),
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _telemetry_baseline_model_bytes(env: Any) -> bytes | None:
    """Read Plan1's immutable schema-time MJB snapshot through wrappers."""
    for candidate in _iter_environment_chain(env):
        try:
            telemetry = getattr(candidate, "_c01_force_telemetry", None)
            provider = getattr(telemetry, "compiled_model_bytes", None)
        except Exception:
            continue
        if not callable(provider):
            continue
        payload = provider()
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(
                "_c01_force_telemetry.compiled_model_bytes() must return bytes."
            )
        return bytes(payload)
    return None


def _replay_environment_variables(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return replay-relevant environment values without sensitive entries."""
    source = os.environ if environ is None else environ
    return {
        str(name): str(source[name])
        for name in sorted(source)
        if (
            not any(
                token in name.upper()
                for token in _REPLAY_ENVIRONMENT_SENSITIVE_TOKENS
            )
            and (
                name in _REPLAY_ENVIRONMENT_EXACT_NAMES
                or name.startswith("HANDSON_")
            )
        )
    }


def capture_runtime_replay_identity() -> dict[str, Any]:
    """Return public host/runtime versions that can change PhysX replay."""

    packages: dict[str, str | None] = {}
    for name in _REPLAY_RUNTIME_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    driver: dict[str, str | None]
    try:
        payload = _NVIDIA_DRIVER_VERSION_PATH.read_bytes()
    except OSError:
        driver = {"first_line": None, "content_sha256": None}
    else:
        lines = payload.decode("utf-8", errors="replace").splitlines()
        driver = {
            "first_line": lines[0] if lines else "",
            "content_sha256": hashlib.sha256(payload).hexdigest(),
        }

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": packages,
        "nvidia_driver": driver,
    }


def capture_git_replay_state(
    repo_root: Path,
    *,
    exclude_path: Path | None = None,
) -> dict[str, Any]:
    """Capture revision identity and a digest, never the full source diff."""

    def run(*arguments: str) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
        return result.stdout

    try:
        top_level = Path(
            run("rev-parse", "--show-toplevel").decode("utf-8").strip()
        )
        commit = run("rev-parse", "--verify", "HEAD").decode("ascii").strip()
        porcelain = run(
            "status", "--porcelain=v1", "--untracked-files=all"
        ).decode("utf-8", errors="replace")
        binary_diff = run("diff", "--binary", "HEAD")
    except (OSError, subprocess.SubprocessError):
        from handson.runtime.deployed_source_state import (
            verify_deployed_source_state,
        )

        deployed_state = verify_deployed_source_state(repo_root)
        if deployed_state is not None:
            deployed_state = dict(deployed_state)
            deployed_state.pop("ancestors", None)
            return deployed_state
        return {
            "commit": None,
            "dirty": None,
            "status": "unavailable",
            "changed_files": [],
            "diff_binary_sha256": None,
            "worktree_content_sha256": None,
            "reason": "Git revision state is unavailable.",
        }
    excluded_relative = None
    if exclude_path is not None:
        try:
            excluded_relative = exclude_path.resolve().relative_to(
                top_level.resolve()
            ).as_posix()
        except ValueError:
            pass

    def is_recording_output(path: str) -> bool:
        if not excluded_relative:
            return False
        return any(
            candidate == excluded_relative
            or candidate.startswith(f"{excluded_relative}/")
            for candidate in path.split(" -> ")
        )

    changed_files = sorted(
        {
            line[3:]
            for line in porcelain.splitlines()
            if len(line) >= 4 and line[3:] and not is_recording_output(line[3:])
        }
    )
    content_identity: list[dict[str, str | None]] = []
    for relative_path in changed_files:
        candidate_name = relative_path.split(" -> ")[-1]
        candidate = top_level / candidate_name
        if candidate.is_symlink():
            kind = "symlink"
            payload = os.readlink(candidate).encode(
                "utf-8",
                errors="surrogateescape",
            )
            content_sha256 = hashlib.sha256(payload).hexdigest()
        elif candidate.is_file():
            kind = "file"
            content_sha256 = _sha256_file(candidate)
        else:
            kind = "deleted"
            content_sha256 = None
        content_identity.append(
            {
                "path": relative_path,
                "kind": kind,
                "sha256": content_sha256,
            }
        )
    worktree_bytes = json.dumps(
        content_identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    dirty = bool(changed_files)
    return {
        "commit": commit,
        "dirty": dirty,
        "status": "dirty" if dirty else "clean",
        "changed_files": changed_files,
        "diff_binary_sha256": hashlib.sha256(binary_diff).hexdigest(),
        "worktree_content_sha256": hashlib.sha256(worktree_bytes).hexdigest(),
    }


_git_replay_state = capture_git_replay_state


def _declared_compiled_model_sha256(
    environment_schema: Mapping[str, Any] | None,
) -> str | None:
    if not environment_schema:
        return None
    mujoco_schema = environment_schema.get("mujoco")
    if not isinstance(mujoco_schema, Mapping):
        return None
    value = mujoco_schema.get("compiled_model_sha256")
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(
            "Environment schema mujoco.compiled_model_sha256 must be a "
            "64-character hexadecimal SHA256 digest."
        )
    return normalized


def write_replay_assets(
    demo_dir: str | Path,
    env: Any,
    *,
    task_name: str,
    recording_parameters: Mapping[str, Any],
    environment_schema: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Save the exact MuJoCo model and a self-contained replay manifest.

    The function first uses Plan1 telemetry's immutable schema-time MJB when it
    is available, then serializes the current wrapper-chain ``mujoco.MjModel``
    as a final snapshot.  A changed runtime model is saved separately instead
    of being mistaken for schema corruption.  Non-MuJoCo environments remain
    recordable and get an explicit skipped-model reason.
    """
    demo_path = Path(demo_dir)
    demo_path.mkdir(parents=True, exist_ok=True)
    schema_path = demo_path / "recording_schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(
            f"Recording schema must be written before replay assets: {schema_path}"
        )

    expected_model_sha256 = _declared_compiled_model_sha256(environment_schema)
    baseline_bytes = _telemetry_baseline_model_bytes(env)
    baseline_source = (
        "force_telemetry_schema_baseline" if baseline_bytes is not None else None
    )
    final_bytes = None
    missing_model_reason = None
    mujoco_module = None
    try:
        import mujoco as mujoco_module  # type: ignore[no-redef]
    except ImportError:
        missing_model_reason = "The mujoco Python package is unavailable."
    else:
        model = None
        for candidate in _iter_environment_chain(env):
            try:
                candidate_model = getattr(candidate, "model", None)
            except Exception:
                continue
            if isinstance(candidate_model, mujoco_module.MjModel):
                model = candidate_model
                break

        if model is None:
            missing_model_reason = (
                "The environment was not provided."
                if env is None
                else "No mujoco.MjModel was found in the environment wrapper chain."
            )
        else:
            final_buffer = np.empty(
                mujoco_module.mj_sizeModel(model),
                dtype=np.uint8,
            )
            mujoco_module.mj_saveModel(model, buffer=final_buffer)
            final_bytes = final_buffer.tobytes()

    if baseline_bytes is None and final_bytes is not None:
        baseline_bytes = final_bytes
        baseline_source = "current_model_fallback"

    model_asset: dict[str, Any]
    final_model_asset: dict[str, Any]
    if baseline_bytes is None:
        model_asset = {
            "status": "skipped",
            "role": "schema_baseline",
            "path": None,
            "sha256": None,
            "source": None,
            "reason": missing_model_reason or "No compiled model snapshot is available.",
            "environment_schema_sha256": expected_model_sha256,
            "matches_environment_schema": None,
        }
        final_model_asset = {
            "status": "skipped",
            "role": "runtime_final",
            "path": None,
            "sha256": None,
            "reason": missing_model_reason or "No final MuJoCo model is available.",
        }
    else:
        model_sha256 = hashlib.sha256(baseline_bytes).hexdigest()
        model_path = demo_path / "model.mjb"
        _atomic_write_bytes(model_path, baseline_bytes)
        hash_matches = (
            None
            if expected_model_sha256 is None
            else model_sha256 == expected_model_sha256
        )
        model_asset = {
            "status": "saved",
            "role": "schema_baseline",
            "path": model_path.relative_to(demo_path).as_posix(),
            "sha256": model_sha256,
            "source": baseline_source,
            "environment_schema_sha256": expected_model_sha256,
            "matches_environment_schema": hash_matches,
        }
        if final_bytes is None:
            final_model_asset = {
                "status": "skipped",
                "role": "runtime_final",
                "path": None,
                "sha256": None,
                "reason": missing_model_reason or "No final MuJoCo model is available.",
            }
        else:
            final_sha256 = hashlib.sha256(final_bytes).hexdigest()
            if final_sha256 == model_sha256:
                final_model_asset = {
                    "status": "same_as_baseline",
                    "role": "runtime_final",
                    "path": model_path.relative_to(demo_path).as_posix(),
                    "sha256": final_sha256,
                }
            else:
                final_model_path = demo_path / "final_model.mjb"
                _atomic_write_bytes(final_model_path, final_bytes)
                final_model_asset = {
                    "status": "saved",
                    "role": "runtime_final",
                    "path": final_model_path.relative_to(demo_path).as_posix(),
                    "sha256": final_sha256,
                }

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    git_state = capture_git_replay_state(
        Path(repo_root),
        exclude_path=demo_path,
    )
    mujoco_versions: dict[str, Any]
    if mujoco_module is None:
        mujoco_versions = {
            "python_package": None,
            "runtime": None,
        }
    else:
        mujoco_versions = {
            "python_package": str(getattr(mujoco_module, "__version__", "unknown")),
            "runtime": str(mujoco_module.mj_versionString()),
        }

    manifest = {
        "format": "handson.replay_manifest",
        "version": 1,
        "task": {
            "name": str(task_name),
        },
        "recording": _json_safe(recording_parameters),
        "environment": {
            "variables": _replay_environment_variables(environ),
        },
        "software": {
            "git": git_state,
            "mujoco": mujoco_versions,
            "numpy": str(np.__version__),
            "runtime_replay_identity": capture_runtime_replay_identity(),
        },
        "assets": {
            "recording_schema": {
                "path": schema_path.relative_to(demo_path).as_posix(),
                "sha256": _sha256_file(schema_path),
            },
            "compiled_model": model_asset,
            "final_compiled_model": final_model_asset,
        },
    }
    manifest_path = demo_path / "replay_manifest.json"
    _atomic_write_json(manifest_path, manifest)

    if (
        model_asset.get("source") == "force_telemetry_schema_baseline"
        and model_asset.get("matches_environment_schema") is False
    ):
        raise ValueError(
            "Saved model.mjb SHA256 does not match the compiled model SHA256 "
            "declared by the environment recording schema."
        )
    return manifest_path
