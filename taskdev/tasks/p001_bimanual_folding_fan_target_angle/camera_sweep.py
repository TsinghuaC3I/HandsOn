"""Capture and certify the three P0-01 recording-camera streams."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any, Mapping

import numpy as np
from PIL import Image

from handson.tasks.t48_unfold_fan.config import TaskConfig


EXPECTED_SHAPES = {
    "head": (480, 640, 3),
    "wrist_left": (480, 848, 3),
    "wrist_right": (480, 848, 3),
}
SENSOR_NAMES = {
    "head": "head_camera",
    "wrist_left": "wrist_left_camera",
    "wrist_right": "wrist_right_camera",
}
MIN_MEAN_RGB = 30.0


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _rgb_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"expected HxWxC RGB array, got {array.shape}")
    rgb = array[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        if rgb.size and float(np.max(rgb)) <= 1.5:
            rgb = rgb * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _camera_arrays(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    nested_images = observation.get("images")
    images = nested_images if isinstance(nested_images, Mapping) else observation
    head_key = "ego" if "ego" in images else "random_camera"
    required = (head_key, "wrist_left", "wrist_right")
    missing = [key for key in required if key not in images]
    if missing:
        raise KeyError(f"missing P0-01 camera streams: {missing}")
    return {
        "head": _rgb_array(images[head_key]),
        "wrist_left": _rgb_array(images["wrist_left"]),
        "wrist_right": _rgb_array(images["wrist_right"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument(
        "--target", type=float, choices=(60.0, 90.0, 120.0), default=90.0
    )
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 30:
        raise ValueError("P0-01 camera acceptance requires at least 30 frames")

    env = TaskConfig().get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=True,
        sim_backend="isaacsim",
        seed=args.seed,
        target_open_deg=args.target,
        image_obs=True,
        show_sim_cameras=False,
    )
    records: dict[str, list[dict[str, Any]]] = {
        name: [] for name in EXPECTED_SHAPES
    }
    step_infos: list[dict[str, Any]] = []
    args.image_dir.mkdir(parents=True, exist_ok=True)
    try:
        observation, reset_info = env.reset(
            seed=args.seed, target_open_deg=args.target
        )
        raw_env = env.unwrapped
        for frame_index in range(args.frames):
            if frame_index:
                action = np.zeros(env.action_space.shape, dtype=np.float64)
                observation, _reward, terminated, truncated, info = env.step(action)
                step_infos.append(_jsonable(dict(info)))
                if terminated or truncated:
                    raise RuntimeError(
                        f"camera sweep terminated at frame {frame_index}"
                    )
            arrays = _camera_arrays(observation)
            for name, rgb in arrays.items():
                destination = args.image_dir / f"{name}_{frame_index:04d}.png"
                Image.fromarray(rgb).save(destination)
                records[name].append(
                    {
                        "frame": frame_index,
                        "shape": list(rgb.shape),
                        "dtype": str(rgb.dtype),
                        "finite": bool(np.all(np.isfinite(rgb))),
                        "mean_rgb": float(np.mean(rgb)),
                        "min_rgb": int(np.min(rgb)),
                        "max_rgb": int(np.max(rgb)),
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                        "file": destination.name,
                    }
                )

        camera_objects = getattr(raw_env, "_cameras", {})
        aliases = dict(getattr(raw_env, "_camera_aliases", {}))
        warnings = [str(value) for value in getattr(raw_env, "_isaac_warnings", [])]
        fallback_warnings = [
            warning
            for warning in warnings
            if "camera" in warning.lower()
            and (
                "fallback" in warning.lower()
                or "unavailable" in warning.lower()
            )
        ]
        per_camera: dict[str, Any] = {}
        for name, expected_shape in EXPECTED_SHAPES.items():
            camera_records = records[name]
            black_frames = [
                record["frame"]
                for record in camera_records
                if float(record["mean_rgb"]) < MIN_MEAN_RGB
            ]
            invalid_shape_frames = [
                record["frame"]
                for record in camera_records
                if tuple(record["shape"]) != expected_shape
            ]
            nonfinite_frames = [
                record["frame"]
                for record in camera_records
                if not bool(record["finite"])
            ]
            sensor_name = SENSOR_NAMES[name]
            per_camera[name] = {
                "sensor_name": sensor_name,
                "real_sensor_present": sensor_name in camera_objects,
                "aliased": sensor_name in aliases,
                "expected_shape": list(expected_shape),
                "frame_count": len(camera_records),
                "black_frames": black_frames,
                "invalid_shape_frames": invalid_shape_frames,
                "nonfinite_frames": nonfinite_frames,
                "min_mean_rgb": min(
                    float(record["mean_rgb"]) for record in camera_records
                ),
                "max_mean_rgb": max(
                    float(record["mean_rgb"]) for record in camera_records
                ),
                "records": camera_records,
                "passed": bool(
                    sensor_name in camera_objects
                    and sensor_name not in aliases
                    and len(camera_records) >= 30
                    and not black_frames
                    and not invalid_shape_frames
                    and not nonfinite_frames
                ),
            }

        no_object_writes = all(
            int(info.get("p001_object_state_writes_after_action_zero", -1)) == 0
            for info in step_infos
        )
        result = {
            "schema": "p001_camera_sweep_v1",
            "seed": args.seed,
            "target_deg": args.target,
            "requested_frames": args.frames,
            "reset_info": _jsonable(reset_info),
            "camera_sensor_keys": sorted(str(key) for key in camera_objects),
            "camera_aliases": aliases,
            "fallback_warnings": fallback_warnings,
            "no_object_writes_after_reset": no_object_writes,
            "cameras": per_camera,
            "passed": bool(
                all(value["passed"] for value in per_camera.values())
                and not fallback_warnings
                and no_object_writes
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "schema": result["schema"],
                    "passed": result["passed"],
                    "fallback_warning_count": len(fallback_warnings),
                    "no_object_writes_after_reset": no_object_writes,
                    "cameras": {
                        name: {
                            key: value[key]
                            for key in (
                                "frame_count",
                                "min_mean_rgb",
                                "max_mean_rgb",
                                "real_sensor_present",
                                "aliased",
                                "passed",
                            )
                        }
                        for name, value in per_camera.items()
                    },
                },
                sort_keys=True,
            )
        )
        if not result["passed"]:
            raise SystemExit(1)
    except BaseException as exc:
        error_result = {
            "schema": "p001_camera_sweep_v1",
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "captured_frame_counts": {
                name: len(camera_records)
                for name, camera_records in records.items()
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(error_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("[P001_CAMERA_SWEEP_EXCEPTION]", flush=True)
        traceback.print_exc()
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
