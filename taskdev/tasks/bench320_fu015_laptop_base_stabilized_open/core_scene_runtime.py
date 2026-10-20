"""CPU-safe serialization and frame metrics for the FU015 live runner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .core_scene_spec import HEAD_SHAPE_HWC


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def require_recursive_finite(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite numeric evidence at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            require_recursive_finite(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_recursive_finite(item, f"{path}[{index}]")
        return
    raise TypeError(f"unsupported evidence value at {path}: {type(value).__name__}")


def decoded_rgb_metrics(frame: np.ndarray) -> dict[str, Any]:
    value = np.asarray(frame)
    if value.shape != HEAD_SHAPE_HWC or value.dtype != np.uint8:
        raise ValueError(
            f"head frame must be uint8 {HEAD_SHAPE_HWC}, got {value.shape}/{value.dtype}"
        )
    rgb = value.astype(np.float64)
    luminance = rgb.mean(axis=2)
    return {
        "shape_hwc": list(value.shape),
        "dtype": str(value.dtype),
        "mean_rgb": float(rgb.mean()),
        "min_rgb": int(value.min()),
        "max_rgb": int(value.max()),
        "pixel_variance": float(luminance.var()),
        "black_fraction": float(np.mean(luminance < 5.0)),
    }


def head_content_pass(metrics: Mapping[str, Any], frame_bytes: int) -> bool:
    return bool(
        metrics.get("shape_hwc") == list(HEAD_SHAPE_HWC)
        and metrics.get("dtype") == "uint8"
        and float(metrics.get("mean_rgb", 0.0)) > 30.0
        and float(metrics.get("pixel_variance", 0.0)) > 15.0
        and float(metrics.get("black_fraction", 1.0)) < 0.25
        and int(frame_bytes) >= 150_000
    )


def pose_drift_m(first_xyzw: np.ndarray, last_xyzw: np.ndarray) -> float:
    first = np.asarray(first_xyzw, dtype=np.float64).reshape(7)
    last = np.asarray(last_xyzw, dtype=np.float64).reshape(7)
    return float(np.linalg.norm(last[:3] - first[:3]))


def quaternion_angle_rad(first_xyzw: np.ndarray, last_xyzw: np.ndarray) -> float:
    first = np.asarray(first_xyzw, dtype=np.float64).reshape(7)[3:]
    last = np.asarray(last_xyzw, dtype=np.float64).reshape(7)[3:]
    first /= max(float(np.linalg.norm(first)), 1.0e-12)
    last /= max(float(np.linalg.norm(last)), 1.0e-12)
    cosine = min(1.0, abs(float(np.dot(first, last))))
    return float(2.0 * math.acos(cosine))


def aggregate_checks(checks: Mapping[str, bool]) -> tuple[bool, list[str]]:
    failures = sorted(name for name, passed in checks.items() if passed is not True)
    return not failures, failures
