#!/usr/bin/env python3
"""Capture real robot head and wrist camera preview videos.

This script is intentionally separate from the MuJoCo/demo renderers. It opens
the physical UVC devices configured for the robot:

  - /dev/stereo_camera       head stereo camera, side-by-side frame
  - /dev/cam_left_wrist      left wrist fisheye
  - /dev/cam_right_wrist     right wrist fisheye

Outputs are H.264 MP4 files encoded with yuv420p so they can be opened by VS
Code, browsers, and most desktop players.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "teleoperation/manus/vendor/wuji-hand-teleop/src/camera/config/camera_config.yaml"
)
DEFAULT_OUT_ROOT = REPO_ROOT / "demo_outputs/real_robot_camera_views"


@dataclass
class CameraSpec:
    key: str
    device: str
    width: int
    height: int
    fps: float
    pixel_format: str
    tf: dict[str, Any]
    raw_config: dict[str, Any]


class H264Writer:
    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self._proc = self._start_ffmpeg()

    def _start_ffmpeg(self) -> subprocess.Popen:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.3f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"{self.path.name}: expected {self.width}x{self.height}, "
                f"got {frame_bgr.shape[1]}x{frame_bgr.shape[0]}"
            )
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self._proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        self.frame_count += 1

    def close(self) -> int:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        return self._proc.wait()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def normalize_fourcc(pixel_format: str) -> str | None:
    fmt = (pixel_format or "").lower()
    if "mjpg" in fmt or "mjpeg" in fmt:
        return "MJPG"
    if "yuyv" in fmt or "yuy2" in fmt:
        return "YUYV"
    return None


def read_v4l2_serial(device: str) -> str | None:
    try:
        real = Path(device).resolve()
        video_name = real.name
        sys_node = Path("/sys/class/video4linux") / video_name
        if not sys_node.exists():
            return None
        node = (sys_node / "device").resolve()
        while str(node) != "/":
            serial_path = node / "serial"
            if serial_path.exists():
                serial = serial_path.read_text(encoding="utf-8").strip()
                return serial or None
            node = node.parent
    except OSError:
        return None
    return None


def find_intrinsics(cam: CameraSpec) -> dict[str, Any]:
    intrinsics_dir = cam.raw_config.get("intrinsics_dir")
    if not intrinsics_dir:
        return {}

    serial = cam.raw_config.get("serial_number") or read_v4l2_serial(cam.device)
    info: dict[str, Any] = {"intrinsics_dir": intrinsics_dir, "serial": serial}
    if not serial:
        return info

    path = Path(intrinsics_dir) / f"{serial}.yaml"
    info["intrinsics_file"] = str(path)
    if path.exists():
        try:
            info["intrinsics"] = load_yaml(path)
        except Exception as exc:  # pragma: no cover - diagnostics only
            info["intrinsics_error"] = str(exc)
    else:
        info["intrinsics_missing"] = True
    return info


def build_specs(config: dict[str, Any], args: argparse.Namespace) -> dict[str, CameraSpec]:
    cameras = config.get("cameras", {})
    specs: dict[str, CameraSpec] = {}
    overrides = {
        "head": args.head_device,
        "left_wrist": args.left_wrist_device,
        "right_wrist": args.right_wrist_device,
    }

    for key in ("head", "left_wrist", "right_wrist"):
        cam = cameras.get(key, {})
        if not cam.get("enabled", True):
            continue
        res = cam.get("resolution", {})
        device = overrides[key] or cam.get("video_device", "")
        specs[key] = CameraSpec(
            key=key,
            device=device,
            width=int(res.get("width", 640)),
            height=int(res.get("height", 480)),
            fps=float(res.get("fps", 30)),
            pixel_format=str(cam.get("pixel_format", "")),
            tf=cam.get("tf", {}),
            raw_config=cam,
        )

    return specs


def scale_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / float(frame.shape[1])
    width = max_width
    height = int(round(frame.shape[0] * scale))
    if width % 2:
        width -= 1
    if height % 2:
        height -= 1
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def even_sized(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return frame[: h - (h % 2), : w - (w % 2)]


def open_capture(cam: CameraSpec) -> tuple[cv2.VideoCapture | None, str | None]:
    if not cam.device:
        return None, "empty device path"
    if not Path(cam.device).exists():
        return None, "device path does not exist"

    cap = cv2.VideoCapture(cam.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, "cv2.VideoCapture failed to open"

    fourcc = normalize_fourcc(cam.pixel_format)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
    cap.set(cv2.CAP_PROP_FPS, cam.fps)

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None, "opened device but could not read first frame"

    return cap, None


def make_writers(
    out_dir: Path,
    cam_key: str,
    first_frame: np.ndarray,
    fps: float,
    max_width: int,
    split_head: bool,
) -> tuple[dict[str, H264Writer], dict[str, tuple[int, int]]]:
    writers: dict[str, H264Writer] = {}
    output_shapes: dict[str, tuple[int, int]] = {}

    def add_writer(name: str, frame: np.ndarray) -> None:
        scaled = even_sized(scale_frame(frame, max_width))
        h, w = scaled.shape[:2]
        writers[name] = H264Writer(out_dir / f"{name}.mp4", w, h, fps)
        output_shapes[name] = (w, h)

    if cam_key == "head":
        add_writer("head_stereo", first_frame)
        if split_head:
            half = first_frame.shape[1] // 2
            add_writer("head_left", first_frame[:, :half])
            add_writer("head_right", first_frame[:, half:])
    else:
        add_writer(cam_key, first_frame)

    return writers, output_shapes


def write_frames(
    writers: dict[str, H264Writer],
    cam_key: str,
    frame: np.ndarray,
    max_width: int,
    split_head: bool,
) -> None:
    def write_one(name: str, src: np.ndarray) -> None:
        writers[name].write(even_sized(scale_frame(src, max_width)))

    if cam_key == "head":
        write_one("head_stereo", frame)
        if split_head:
            half = frame.shape[1] // 2
            write_one("head_left", frame[:, :half])
            write_one("head_right", frame[:, half:])
    else:
        write_one(cam_key, frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record real robot head/wrist camera MP4 previews."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--views", default="head,left_wrist,right_wrist")
    parser.add_argument("--max_width", type=int, default=1280)
    parser.add_argument("--output_fps", type=float, default=0.0)
    parser.add_argument("--no_head_split", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--head_device", default="")
    parser.add_argument("--left_wrist_device", default="")
    parser.add_argument("--right_wrist_device", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    specs = build_specs(config, args)
    requested = [v.strip() for v in args.views.split(",") if v.strip()]
    specs = {key: specs[key] for key in requested if key in specs}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    info: dict[str, Any] = {
        "config": str(args.config),
        "out_dir": str(out_dir),
        "duration": args.duration,
        "max_width": args.max_width,
        "views": {},
    }

    missing = []
    for key, cam in specs.items():
        exists = bool(cam.device and Path(cam.device).exists())
        info["views"][key] = {
            "device": cam.device,
            "device_exists": exists,
            "configured_resolution": [cam.width, cam.height],
            "configured_fps": cam.fps,
            "pixel_format": cam.pixel_format,
            "tf": cam.tf,
            "intrinsics": find_intrinsics(cam),
        }
        if not exists:
            missing.append(f"{key}: {cam.device}")

    if args.dry_run:
        info["dry_run"] = True
        (out_dir / "capture_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    captures: dict[str, cv2.VideoCapture] = {}
    writers_by_camera: dict[str, dict[str, H264Writer]] = {}

    try:
        for key, cam in specs.items():
            cap, error = open_capture(cam)
            info["views"][key]["open_error"] = error
            if cap is None:
                continue

            ok, first = cap.read()
            if not ok or first is None:
                cap.release()
                info["views"][key]["open_error"] = "could not read warmup frame"
                continue

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or cam.fps)
            output_fps = args.output_fps if args.output_fps > 0 else min(actual_fps, cam.fps)
            output_fps = output_fps if output_fps > 0 else 30.0

            writers, output_shapes = make_writers(
                out_dir=out_dir,
                cam_key=key,
                first_frame=first,
                fps=output_fps,
                max_width=args.max_width,
                split_head=not args.no_head_split,
            )
            captures[key] = cap
            writers_by_camera[key] = writers
            info["views"][key].update(
                {
                    "actual_resolution": [actual_w, actual_h],
                    "actual_fps": actual_fps,
                    "output_fps": output_fps,
                    "outputs": {
                        name: {
                            "path": str(out_dir / f"{name}.mp4"),
                            "resolution": [shape[0], shape[1]],
                        }
                        for name, shape in output_shapes.items()
                    },
                }
            )
            write_frames(writers, key, first, args.max_width, not args.no_head_split)

        if not captures:
            info["error"] = "No requested real camera devices opened."
            (out_dir / "capture_info.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("No real camera devices opened.", file=sys.stderr)
            if missing:
                print("Missing device paths:", file=sys.stderr)
                for item in missing:
                    print(f"  {item}", file=sys.stderr)
            print(f"Wrote diagnostics: {out_dir / 'capture_info.json'}", file=sys.stderr)
            return 2

        deadline = time.monotonic() + max(0.0, args.duration)
        while time.monotonic() < deadline:
            for key, cap in captures.items():
                ok, frame = cap.read()
                if not ok or frame is None:
                    info["views"][key]["read_error"] = "read failed during capture"
                    continue
                write_frames(
                    writers_by_camera[key],
                    key,
                    frame,
                    args.max_width,
                    not args.no_head_split,
                )

    finally:
        for cap in captures.values():
            cap.release()
        for key, writers in writers_by_camera.items():
            for name, writer in writers.items():
                ret = writer.close()
                info["views"][key].setdefault("writer_returncodes", {})[name] = ret
                info["views"][key].setdefault("frame_counts", {})[name] = writer.frame_count
        (out_dir / "capture_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"Wrote real camera videos to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
