#!/usr/bin/env python3
"""Run a remote MuJoCo preview while publishing the head-camera image."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable


HEAD_CAMERA_KEYS = (
    "main",
    "head_main",
    "front",
    "random_camera",
    "ego",
)
FALLBACK_CAMERA_KEYS = (
    "wrist",
    "wrist_left",
    "wrist_right",
    "ego_left",
    "ego_right",
)


def head_preview_frame(
    observation: Any,
    normalize_frame: Callable[[Any], Any],
) -> Any | None:
    """Select a valid head view before considering any wrist camera."""

    if not isinstance(observation, dict):
        return None
    images = observation.get("images")
    mappings = [images] if isinstance(images, dict) else []
    mappings.append(observation)
    for key in (*HEAD_CAMERA_KEYS, *FALLBACK_CAMERA_KEYS):
        for mapping in mappings:
            candidate = mapping.get(key)
            if candidate is None:
                continue
            frame = normalize_frame(candidate)
            if frame is not None:
                return frame
    return None


def load_recorder(path: Path):
    spec = importlib.util.spec_from_file_location(
        "handson_remote_record_demos_zarr",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load recorder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def keep_supported_flags(argv: list[str], recorder: Any) -> list[str]:
    """Drop flags introduced after the selected stable recorder revision."""

    supported = set(recorder.FLAGS)
    kept = [argv[0]]
    for argument in argv[1:]:
        if not argument.startswith("--"):
            kept.append(argument)
            continue
        name = argument[2:].split("=", 1)[0]
        if name.startswith("no") and name[2:] in supported:
            kept.append(argument)
        elif name in supported:
            kept.append(argument)
    return kept


def main() -> None:
    raw_script = os.environ.get("HANDSON_REMOTE_RECORD_SCRIPT", "").strip()
    if not raw_script:
        raise RuntimeError("HANDSON_REMOTE_RECORD_SCRIPT is required")
    script = Path(raw_script).expanduser().resolve()
    if not script.is_file():
        raise RuntimeError(f"remote recorder does not exist: {script}")

    recorder = load_recorder(script)
    recorder._preview_frame_from_observation = lambda observation: head_preview_frame(
        observation,
        recorder._normalize_frame,
    )
    sys.argv = keep_supported_flags(sys.argv, recorder)
    recorder.app.run(recorder.main)


if __name__ == "__main__":
    main()
