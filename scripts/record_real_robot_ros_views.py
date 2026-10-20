#!/usr/bin/env python3
"""Record real robot camera ROS topics to playable H.264 MP4 files.

Use this when the camera ROS nodes are already running. It subscribes to the
true robot camera topics instead of opening /dev/video devices directly:

  - /stereo/left/compressed
  - /stereo/right/compressed
  - /cam_left_wrist/color/image_raw
  - /cam_right_wrist/color/image_raw
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO_ROOT / "demo_outputs/real_robot_ros_views"

DEFAULT_TOPICS = {
    "head_left": {
        "topic": "/stereo/left/compressed",
        "type": "compressed",
        "frame_id": "stereo_left",
    },
    "head_right": {
        "topic": "/stereo/right/compressed",
        "type": "compressed",
        "frame_id": "stereo_right",
    },
    "left_wrist": {
        "topic": "/cam_left_wrist/color/image_raw",
        "type": "raw",
        "frame_id": "cam_left_wrist_link",
    },
    "right_wrist": {
        "topic": "/cam_right_wrist/color/image_raw",
        "type": "raw",
        "frame_id": "cam_right_wrist_link",
    },
}


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


class ViewRecorder:
    def __init__(
        self,
        name: str,
        topic: str,
        msg_type: str,
        out_dir: Path,
        output_fps: float,
        max_width: int,
    ) -> None:
        self.name = name
        self.topic = topic
        self.msg_type = msg_type
        self.out_dir = out_dir
        self.output_fps = output_fps
        self.max_width = max_width
        self.writer: H264Writer | None = None
        self.first_frame_time: float | None = None
        self.last_frame_time: float | None = None
        self.messages = 0
        self.decode_errors = 0
        self.output_resolution: list[int] | None = None

    def add_frame(self, frame_bgr: np.ndarray) -> None:
        frame_bgr = even_sized(scale_frame(frame_bgr, self.max_width))
        if self.writer is None:
            h, w = frame_bgr.shape[:2]
            self.output_resolution = [w, h]
            self.writer = H264Writer(self.out_dir / f"{self.name}.mp4", w, h, self.output_fps)
            cv2.imwrite(str(self.out_dir / f"{self.name}_first_frame.jpg"), frame_bgr)
            self.first_frame_time = time.time()
        self.writer.write(frame_bgr)
        self.last_frame_time = time.time()
        self.messages += 1

    def close(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "topic": self.topic,
            "type": self.msg_type,
            "messages": self.messages,
            "decode_errors": self.decode_errors,
            "first_frame_time": self.first_frame_time,
            "last_frame_time": self.last_frame_time,
        }
        if self.writer is not None:
            ret = self.writer.close()
            info.update(
                {
                    "video": str(self.writer.path),
                    "first_frame_jpg": str(self.out_dir / f"{self.name}_first_frame.jpg"),
                    "resolution": self.output_resolution,
                    "output_fps": self.output_fps,
                    "writer_returncode": ret,
                    "frames_written": self.writer.frame_count,
                }
            )
        return info


class RosViewRecorderNode(Node):
    def __init__(self, recorders: dict[str, ViewRecorder]) -> None:
        super().__init__("real_robot_view_recorder")
        self.bridge = CvBridge()
        self.recorders = recorders
        self._subs = []
        for name, recorder in recorders.items():
            if recorder.msg_type == "compressed":
                sub = self.create_subscription(
                    CompressedImage,
                    recorder.topic,
                    lambda msg, view=name: self._on_compressed(view, msg),
                    10,
                )
            else:
                sub = self.create_subscription(
                    Image,
                    recorder.topic,
                    lambda msg, view=name: self._on_raw(view, msg),
                    10,
                )
            self._subs.append(sub)

    def _on_compressed(self, view: str, msg: CompressedImage) -> None:
        recorder = self.recorders[view]
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("cv2.imdecode returned None")
            recorder.add_frame(frame)
        except Exception as exc:  # pragma: no cover - diagnostics only
            recorder.decode_errors += 1
            self.get_logger().warning(f"{view}: failed to decode compressed image: {exc}")

    def _on_raw(self, view: str, msg: Image) -> None:
        recorder = self.recorders[view]
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            recorder.add_frame(frame)
        except Exception as exc:  # pragma: no cover - diagnostics only
            recorder.decode_errors += 1
            self.get_logger().warning(f"{view}: failed to convert raw image: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record real robot camera ROS topics to MP4 previews."
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--views", default="head_left,head_right,left_wrist,right_wrist")
    parser.add_argument("--max_width", type=int, default=1280)
    parser.add_argument("--output_fps", type=float, default=30.0)
    parser.add_argument("--head_left_topic", default=DEFAULT_TOPICS["head_left"]["topic"])
    parser.add_argument("--head_right_topic", default=DEFAULT_TOPICS["head_right"]["topic"])
    parser.add_argument("--left_wrist_topic", default=DEFAULT_TOPICS["left_wrist"]["topic"])
    parser.add_argument("--right_wrist_topic", default=DEFAULT_TOPICS["right_wrist"]["topic"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    topics = {
        "head_left": {**DEFAULT_TOPICS["head_left"], "topic": args.head_left_topic},
        "head_right": {**DEFAULT_TOPICS["head_right"], "topic": args.head_right_topic},
        "left_wrist": {**DEFAULT_TOPICS["left_wrist"], "topic": args.left_wrist_topic},
        "right_wrist": {**DEFAULT_TOPICS["right_wrist"], "topic": args.right_wrist_topic},
    }
    requested = [v.strip() for v in args.views.split(",") if v.strip()]
    topics = {key: topics[key] for key in requested if key in topics}

    rclpy.init()
    recorders = {
        name: ViewRecorder(
            name=name,
            topic=str(spec["topic"]),
            msg_type=str(spec["type"]),
            out_dir=out_dir,
            output_fps=args.output_fps,
            max_width=args.max_width,
        )
        for name, spec in topics.items()
    }
    node = RosViewRecorderNode(recorders)
    observed_topics = {name: types for name, types in node.get_topic_names_and_types()}
    publisher_counts = {
        name: len(node.get_publishers_info_by_topic(recorder.topic))
        for name, recorder in recorders.items()
    }

    info: dict[str, Any] = {
        "out_dir": str(out_dir),
        "duration": args.duration,
        "max_width": args.max_width,
        "output_fps": args.output_fps,
        "topic_names_and_types_after_subscription": observed_topics,
        "views": {
            name: {
                "topic": rec.topic,
                "type": rec.msg_type,
                "publisher_count_at_start": publisher_counts[name],
            }
            for name, rec in recorders.items()
        },
    }

    deadline = time.monotonic() + max(0.0, args.duration)
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        for name, recorder in recorders.items():
            info["views"][name].update(recorder.close())
        (out_dir / "capture_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    total_frames = sum(v.get("frames_written", 0) for v in info["views"].values())
    if total_frames == 0:
        if all(v.get("publisher_count_at_start", 0) == 0 for v in info["views"].values()):
            print("No camera ROS publishers were found for the requested topics.", flush=True)
        else:
            print("No camera ROS frames were received.", flush=True)
        print(f"Wrote diagnostics: {out_dir / 'capture_info.json'}", flush=True)
        return 2

    print(f"Wrote real robot ROS camera videos to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
