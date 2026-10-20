#!/usr/bin/env python3
"""Render E01/F01 IPC cloth replay nodes into an inspectable MP4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import zarr

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "handson"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from handson.data.video_writer import Mp4VideoWriter


_PREFIX = "e01_ipc_replay"


def _resolve_replay_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "replay.zarr" and path.is_dir():
        return path
    replay_path = path / "replay.zarr"
    if replay_path.is_dir():
        return replay_path
    raise FileNotFoundError(f"Could not find replay.zarr under {path}.")


def _node_dataset(data: zarr.Group) -> zarr.Array:
    name = f"{_PREFIX}_cloth_nodes"
    if name not in data:
        raise KeyError(
            f"data/{name} is missing. Re-run scripts/e01_replay_ipc_cloth.py "
            "without --keypoints-only before rendering a cloth replay video."
        )
    nodes = data[name]
    if len(nodes.shape) != 3 or tuple(nodes.shape[2:]) != (3,):
        raise RuntimeError(f"Expected data/{name} shape (T,N,3), got {nodes.shape}.")
    return nodes


def _valid_frames(data: zarr.Group, frame_count: int) -> np.ndarray:
    name = f"{_PREFIX}_valid_frame"
    if name not in data:
        return np.ones(frame_count, dtype=bool)
    values = np.asarray(data[name][:frame_count], dtype=bool).reshape(-1)
    if values.shape[0] != frame_count:
        return np.ones(frame_count, dtype=bool)
    return values


def _bounds(nodes: np.ndarray) -> dict[str, tuple[float, float]]:
    flat = np.asarray(nodes, dtype=np.float64).reshape(-1, 3)
    result = {}
    for axis, name in enumerate(("x", "y", "z")):
        lo = float(np.nanpercentile(flat[:, axis], 0.5))
        hi = float(np.nanpercentile(flat[:, axis], 99.5))
        span = max(hi - lo, 1e-6)
        pad = max(0.02, span * 0.06) if name != "z" else max(0.004, span * 0.20)
        result[name] = (lo - pad, hi + pad)
    return result


def _height_colours(z: np.ndarray, z_bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = z_bounds
    t = np.clip((np.asarray(z, dtype=np.float64) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    colours = np.zeros((t.size, 3), dtype=np.uint8)
    colours[:, 0] = np.asarray(35 + 215 * t, dtype=np.uint8)
    colours[:, 1] = np.asarray(90 + 140 * (1.0 - np.abs(t - 0.55) * 1.8).clip(0.0, 1.0), dtype=np.uint8)
    colours[:, 2] = np.asarray(235 - 180 * t, dtype=np.uint8)
    return colours


def _map_to_rect(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    rect: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    left, top, width, height = rect
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    px = left + (np.asarray(x) - x0) / max(x1 - x0, 1e-6) * width
    py = top + height - (np.asarray(y) - y0) / max(y1 - y0, 1e-6) * height
    return np.rint(px).astype(np.int32), np.rint(py).astype(np.int32)


def _draw_points(
    frame: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    colours: np.ndarray,
    *,
    radius: int = 2,
) -> None:
    height, width = frame.shape[:2]
    for x, y, colour in zip(px, py, colours):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(
                frame,
                (int(x), int(y)),
                radius,
                (int(colour[0]), int(colour[1]), int(colour[2])),
                -1,
                lineType=cv2.LINE_AA,
            )


def _render_frame(
    nodes: np.ndarray,
    *,
    frame_index: int,
    frame_count: int,
    bounds: dict[str, tuple[float, float]],
    valid: bool,
    width: int,
    height: int,
) -> np.ndarray:
    frame = np.full((height, width, 3), 24, dtype=np.uint8)
    midpoint = width // 2
    top_rect = (36, 72, midpoint - 72, height - 128)
    side_rect = (midpoint + 36, 72, midpoint - 72, height - 128)
    cv2.rectangle(frame, (24, 54), (midpoint - 24, height - 36), (68, 68, 68), 1)
    cv2.rectangle(frame, (midpoint + 24, 54), (width - 24, height - 36), (68, 68, 68), 1)

    order = np.argsort(nodes[:, 2])
    ordered = np.asarray(nodes, dtype=np.float64)[order]
    colours = _height_colours(ordered[:, 2], bounds["z"])
    px, py = _map_to_rect(
        ordered[:, 0],
        ordered[:, 1],
        x_bounds=bounds["x"],
        y_bounds=bounds["y"],
        rect=top_rect,
    )
    _draw_points(frame, px, py, colours, radius=2)

    px, py = _map_to_rect(
        ordered[:, 0],
        ordered[:, 2],
        x_bounds=bounds["x"],
        y_bounds=bounds["z"],
        rect=side_rect,
    )
    _draw_points(frame, px, py, colours, radius=2)

    status = "valid" if valid else "invalid"
    status_colour = (90, 230, 120) if valid else (250, 90, 70)
    cv2.putText(frame, "IPC cloth replay", (36, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(frame, f"frame {frame_index + 1}/{frame_count}", (280, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, status, (width - 140, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_colour, 2, cv2.LINE_AA)
    cv2.putText(frame, "top view", (36, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, "side height view", (midpoint + 36, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, "blue=low  yellow=high", (36, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 190, 190), 1, cv2.LINE_AA)
    return frame


def _open_writer(path: Path, fps: int) -> Mp4VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Mp4VideoWriter.create_h264(
        fps=fps,
        codec="h264",
        input_pix_fmt="rgb24",
        crf=21,
        thread_type="FRAME",
        thread_count=2,
    )
    writer.start(str(path))
    return writer


def render_ipc_replay(
    path: Path,
    *,
    out_path: Path | None = None,
    fps: int = 30,
    max_frames: int = 0,
    width: int = 1280,
    height: int = 720,
) -> Path:
    replay_path = _resolve_replay_path(path)
    data = zarr.open(str(replay_path), mode="r")["data"]
    nodes_dataset = _node_dataset(data)
    frame_count = int(nodes_dataset.shape[0])
    if max_frames > 0:
        frame_count = min(frame_count, int(max_frames))
    if frame_count <= 0:
        raise RuntimeError("IPC replay has no frames to render.")
    nodes = np.asarray(nodes_dataset[:frame_count], dtype=np.float32)
    if not np.all(np.isfinite(nodes)):
        raise RuntimeError("IPC replay cloth nodes contain non-finite values.")
    valid = _valid_frames(data, frame_count)
    render_bounds = _bounds(nodes)
    output = out_path or (replay_path.parent / "videos" / "ipc_replay_cloth.mp4")
    writer = _open_writer(output, max(1, int(fps)))
    try:
        for index in range(frame_count):
            frame = _render_frame(
                nodes[index],
                frame_index=index,
                frame_count=frame_count,
                bounds=render_bounds,
                valid=bool(valid[index]),
                width=max(640, int(width)),
                height=max(360, int(height)),
            )
            writer.write_frame(frame)
            if (index + 1) % 100 == 0 or index + 1 == frame_count:
                print(f"[E01 IPC render] {index + 1}/{frame_count}", flush=True)
    finally:
        writer.stop()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render E01/F01 IPC cloth replay nodes to videos/ipc_replay_cloth.mp4."
    )
    parser.add_argument("path", type=Path, help="Demo dir or replay.zarr path.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    out_path = render_ipc_replay(
        args.path,
        out_path=args.out,
        fps=args.fps,
        max_frames=args.max_frames,
        width=args.width,
        height=args.height,
    )
    print(f"[E01 IPC render] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
