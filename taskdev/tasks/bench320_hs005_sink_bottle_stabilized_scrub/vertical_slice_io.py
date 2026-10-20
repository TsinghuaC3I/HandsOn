"""Action-aligned Zarr I/O and hash helpers for the HS005 vertical slice."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_NAME = "handson.b320_hs005.collection_vertical_slice"
SCHEMA_VERSION = 1
HIDDEN_KEYS = (
    "hs005_role_contacts",
    "hs005_brush_contact",
    "hs005_coverage_mask",
    "hs005_order_state",
    "hs005_path_length",
    "hs005_release_state",
    "hs005_stability_state",
    "hs005_stage",
    "hs005_reset_parameters",
    "hs005_valid",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    """Convert NumPy-rich runtime evidence into strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _zarr_group(group: Any, path: str) -> Any:
    current = group
    for part in path.split("/"):
        current = current.require_group(part)
    return current


def _append(group: Any, name: str, value: Any, *, compressor: Any = None) -> None:
    array = np.asarray(value)
    if array.dtype == np.dtype("O"):
        raise TypeError(f"Object arrays are forbidden in HS005 recording: {name}")
    if name not in group:
        chunks = (1, *array.shape)
        group.create_dataset(
            name,
            shape=(0, *array.shape),
            chunks=chunks,
            dtype=array.dtype,
            compressor=compressor,
        )
    dataset = group[name]
    if tuple(dataset.shape[1:]) != tuple(array.shape) or dataset.dtype != array.dtype:
        raise ValueError(
            f"HS005 recording field changed shape/dtype: {name}, "
            f"expected={dataset.shape[1:]}/{dataset.dtype}, actual={array.shape}/{array.dtype}"
        )
    dataset.append(array.reshape((1, *array.shape)), axis=0)


class HS005EpisodeWriter:
    def __init__(self, episode_dir: Path, metadata: Mapping[str, Any]):
        import zarr
        from numcodecs import Blosc

        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.zarr_path = self.episode_dir / "replay.zarr"
        self.root = zarr.open_group(str(self.zarr_path), mode="w")
        self.root.attrs.update(
            {
                "schema": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                **dict(metadata),
            }
        )
        self.data = self.root.require_group("data")
        self.compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        self.frame_count = 0

    def append(
        self,
        current: Mapping[str, Any],
        action: np.ndarray,
        next_observation: Mapping[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> None:
        _append(self.data, "raw_action", np.asarray(action, dtype=np.float32))
        _append(self.data, "reward", np.asarray(reward, dtype=np.float32))
        _append(self.data, "terminated", np.asarray(terminated, dtype=np.uint8))
        _append(self.data, "truncated", np.asarray(truncated, dtype=np.uint8))
        _append(
            self.data,
            "succeed",
            np.asarray(bool(info.get("succeed", False)), dtype=np.uint8),
        )
        self._append_observation("current", current)
        self._append_observation("next", next_observation)
        self._append_cameras(current)
        self.frame_count += 1

    def _append_observation(self, phase: str, observation: Mapping[str, Any]) -> None:
        state = observation.get("state")
        if not isinstance(state, Mapping):
            raise TypeError("HS005 raw observation must expose a named state mapping")
        state_group = _zarr_group(self.data, f"observations/{phase}/state")
        hidden_group = _zarr_group(self.data, f"task_hidden/{phase}")
        for key, value in sorted(state.items()):
            _append(state_group, str(key), value, compressor=self.compressor)
            if key in HIDDEN_KEYS:
                _append(hidden_group, str(key), value, compressor=self.compressor)
        physical_group = _zarr_group(self.data, f"physical/{phase}")
        for key, value in sorted(observation.items()):
            if str(key).startswith("physx_"):
                _append(physical_group, str(key), value, compressor=self.compressor)

    def _append_cameras(self, observation: Mapping[str, Any]) -> None:
        images = observation.get("images")
        if not isinstance(images, Mapping):
            raise TypeError("HS005 recording requires all three image streams")
        main = images.get("random_camera", images.get("ego"))
        mapping = {
            "main": main,
            "wrist_left": images.get("wrist_left"),
            "wrist_right": images.get("wrist_right"),
        }
        cameras = _zarr_group(self.data, "cameras")
        for key, value in mapping.items():
            if value is None:
                raise KeyError(f"HS005 recording camera is missing: {key}")
            frame = np.asarray(value)
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError(f"Invalid HS005 {key} camera frame: {frame.shape}/{frame.dtype}")
            _append(cameras, key, frame, compressor=self.compressor)

    def finalize(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        safe_metadata = jsonable(metadata)
        self.root.attrs.update(dict(safe_metadata))
        result = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "frame_count": int(self.frame_count),
            "zarr_path": str(self.zarr_path),
            "zarr_sha256": sha256_tree(self.zarr_path),
        }
        atomic_json(self.episode_dir / "episode_manifest.json", {**result, **dict(safe_metadata)})
        return result


def load_zarr_frame(group: Any, index: int) -> dict[str, np.ndarray]:
    """Read one named frame from a flat Zarr group without materializing the episode."""

    result: dict[str, np.ndarray] = {}
    for name in sorted(group.array_keys()):
        result[str(name)] = np.asarray(group[name][index])
    return result
