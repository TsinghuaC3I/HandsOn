"""Canonical camera stream selection for saved HandsOn videos."""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np

from handson.data.camera_info import (
    DEFAULT_FISHEYE_INTRINSICS_PATH,
    fisheye_distort_rectilinear_depth,
    fisheye_distort_rectilinear_frame,
    fisheye_mujoco_fovy_from_intrinsics,
)


CANONICAL_CAMERA_KEYS = ("main", "wrist_left", "wrist_right")
ONLINE_CAMERA_SCOPE_ALL_LIVE = "all_canonical_live"
ONLINE_CAMERA_SCOPE_ALL_POSTRENDER = "all_canonical_postrender"
ONLINE_CAMERA_SCOPE_MAIN_LIVE_POSTRENDER = (
    "main_live_all_canonical_postrender"
)
# Episodes written by the earlier detached-arm experiment used this scope.
# Keep recognizing it so their placeholder wrist videos remain fail-closed.
LEGACY_ONLINE_CAMERA_SCOPE_MAIN_ONLY_WRISTS_DEFERRED = (
    "main_only_wrists_deferred"
)
ONLINE_CAMERA_POSTRENDER_SCOPES = frozenset(
    {
        ONLINE_CAMERA_SCOPE_ALL_POSTRENDER,
        ONLINE_CAMERA_SCOPE_MAIN_LIVE_POSTRENDER,
        LEGACY_ONLINE_CAMERA_SCOPE_MAIN_ONLY_WRISTS_DEFERRED,
    }
)
FISHEYE_POSTPROCESS_CAMERA_KEYS = {"wrist_left", "wrist_right", "wrist"}

DEFAULT_CAMERA_SOURCE_CANDIDATES = {
    "main": ("main", "head_main", "front", "ego", "random_camera"),
    "wrist_left": (
        "wrist_left",
        "left_wrist",
        "handcam_rgb_left",
        "wrist_fisheye_left",
        "wrist",
    ),
    "wrist_right": (
        "wrist_right",
        "right_wrist",
        "handcam_rgb_right",
        "wrist_fisheye_right",
        "wrist",
    ),
}


def _unwrap_env(env):
    if env is None:
        return None
    return getattr(env, "unwrapped", env)


def _env_video_sources(env) -> dict[str, str]:
    raw_env = _unwrap_env(env)
    if raw_env is None:
        return {}

    for attr in (
        "video_camera_sources",
        "canonical_video_sources",
        "canonical_camera_sources",
    ):
        value = getattr(raw_env, attr, None)
        if not isinstance(value, Mapping):
            continue
        return {
            str(key): str(source)
            for key, source in value.items()
            if key in CANONICAL_CAMERA_KEYS and isinstance(source, str)
        }
    return {}


def _is_rgb_image(value: Any) -> bool:
    if value is None:
        return False
    arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.ndim >= 3 and arr.shape[-1] in (3, 4)


def observation_image(observation: dict | None, key: str):
    """Return image `key` from flat or nested observation dicts."""
    if not isinstance(observation, dict):
        return None

    value = observation.get(key)
    if _is_rgb_image(value):
        return value

    images = observation.get("images")
    if isinstance(images, dict):
        value = images.get(key)
        if _is_rgb_image(value):
            return value
    return None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@lru_cache(maxsize=8)
def _postprocess_fovy_from_intrinsics(intrinsics_path: str) -> float | None:
    return fisheye_mujoco_fovy_from_intrinsics(intrinsics_path)


def postprocess_canonical_image(
    image,
    canonical_key: str,
    source_key: str | None = None,
):
    """Apply canonical output projection while leaving the source pose untouched."""
    if canonical_key not in FISHEYE_POSTPROCESS_CAMERA_KEYS:
        return image
    if not _env_flag("HANDSON_APPLY_FISHEYE_DISTORTION", False):
        return image

    intrinsics_path = os.environ.get(
        "HANDSON_FISHEYE_INTRINSICS_YAML",
        str(DEFAULT_FISHEYE_INTRINSICS_PATH),
    )
    fovy_raw = os.environ.get("HANDSON_FISHEYE_POSTPROCESS_FOVY")
    try:
        fovy_deg = float(fovy_raw) if fovy_raw else None
    except ValueError:
        fovy_deg = None
    if fovy_deg is None:
        try:
            fovy_deg = _postprocess_fovy_from_intrinsics(intrinsics_path)
        except Exception:
            fovy_deg = None

    try:
        return fisheye_distort_rectilinear_frame(
            np.asarray(image),
            fovy_deg=fovy_deg,
            intrinsics_path=intrinsics_path,
        )
    except Exception as exc:
        print(
            "[Warning] fisheye postprocess skipped for "
            f"{canonical_key} ({source_key or 'unknown source'}): {exc}"
        )
        return image


def postprocess_canonical_depth(
    depth,
    canonical_key: str,
    source_key: str | None = None,
):
    """Apply the same optional wrist projection used by canonical RGB."""
    if canonical_key not in FISHEYE_POSTPROCESS_CAMERA_KEYS:
        return depth
    if not _env_flag("HANDSON_APPLY_FISHEYE_DISTORTION", False):
        return depth

    intrinsics_path = os.environ.get(
        "HANDSON_FISHEYE_INTRINSICS_YAML",
        str(DEFAULT_FISHEYE_INTRINSICS_PATH),
    )
    fovy_raw = os.environ.get("HANDSON_FISHEYE_POSTPROCESS_FOVY")
    try:
        fovy_deg = float(fovy_raw) if fovy_raw else None
    except ValueError:
        fovy_deg = None
    if fovy_deg is None:
        try:
            fovy_deg = _postprocess_fovy_from_intrinsics(intrinsics_path)
        except Exception:
            fovy_deg = None

    try:
        return fisheye_distort_rectilinear_depth(
            np.asarray(depth),
            fovy_deg=fovy_deg,
            intrinsics_path=intrinsics_path,
        )
    except Exception as exc:
        print(
            "[Warning] fisheye depth postprocess skipped for "
            f"{canonical_key} ({source_key or 'unknown source'}): {exc}"
        )
        return depth


def select_canonical_camera_sources(
    observation: dict | None,
    env=None,
) -> dict[str, str]:
    """Map canonical output keys to source observation image keys.

    Environments can override the defaults with `video_camera_sources`, for
    example {"wrist_left": "ego_left"} when a legacy observation key now
    carries the physical left wrist camera stream.
    """
    selected: dict[str, str] = {}

    for canonical_key, source_key in _env_video_sources(env).items():
        if observation_image(observation, source_key) is not None:
            selected[canonical_key] = source_key

    for canonical_key in CANONICAL_CAMERA_KEYS:
        if canonical_key in selected:
            continue
        for source_key in DEFAULT_CAMERA_SOURCE_CANDIDATES[canonical_key]:
            if observation_image(observation, source_key) is not None:
                selected[canonical_key] = source_key
                break

    return selected


def select_canonical_camera_sources_for_trajectory(
    trajectory,
    env=None,
) -> dict[str, str]:
    """Find stable canonical camera sources from the first usable frames."""
    selected: dict[str, str] = {}
    for step in trajectory:
        obs = step.get("observations", {}) if isinstance(step, dict) else {}
        step_sources = select_canonical_camera_sources(obs, env=env)
        for canonical_key in CANONICAL_CAMERA_KEYS:
            if canonical_key not in selected and canonical_key in step_sources:
                selected[canonical_key] = step_sources[canonical_key]
        if all(key in selected for key in CANONICAL_CAMERA_KEYS):
            break
    return selected
