"""Simple H.264/MP4 writing utilities for HandsOn recordings."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os

import av
import numpy as np


@lru_cache(maxsize=None)
def _encoder_available(codec_name: str) -> bool:
    try:
        av.codec.Codec(codec_name, "w")
    except Exception:  # noqa: BLE001 - PyAV raises version-specific codec errors.
        return False
    return True


@dataclass
class Mp4VideoWriter:
    fps: int
    codec: str
    input_pix_fmt: str
    output_pix_fmt: str = "yuv420p"
    codec_options: dict[str, str] = field(default_factory=dict)
    fallback_codec: str | None = None
    fallback_codec_options: dict[str, str] = field(default_factory=dict)
    thread_type: str | None = None
    thread_count: int | None = None

    def __post_init__(self):
        self._container = None
        self._stream = None
        self._shape = None

    @classmethod
    def create_h264(
        cls,
        fps,
        codec="h264",
        input_pix_fmt="rgb24",
        output_pix_fmt="yuv420p",
        crf=18,
        profile="high",
        thread_type=None,
        thread_count=None,
        encoder=None,
        video_gpu=None,
        **_,
    ):
        requested_encoder = str(
            encoder or os.environ.get("HANDSON_VIDEO_ENCODER") or "auto"
        ).strip().lower()
        gpu_id = str(video_gpu or os.environ.get("HANDSON_VIDEO_GPU") or "0")
        fallback_codec = None
        fallback_codec_options: dict[str, str] = {}

        if codec == "h264" and requested_encoder in {"auto", "gpu", "cuda", "nvenc"}:
            if _encoder_available("h264_nvenc"):
                codec = "h264_nvenc"
                codec_options = {
                    "preset": os.environ.get("HANDSON_NVENC_PRESET", "p4"),
                    "cq": str(crf),
                    "gpu": gpu_id,
                }
                fallback_codec = "h264"
                fallback_codec_options = {
                    "crf": str(crf),
                    "profile": profile,
                }
            else:
                codec_options = {
                    "crf": str(crf),
                    "profile": profile,
                }
        else:
            codec_options = {
                "crf": str(crf),
                "profile": profile,
            }
        return cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            output_pix_fmt=output_pix_fmt,
            codec_options=codec_options,
            fallback_codec=fallback_codec,
            fallback_codec_options=fallback_codec_options,
            thread_type=thread_type,
            thread_count=thread_count,
        )

    def start(self, file_path):
        self.stop()
        self._container = av.open(file_path, mode="w")
        try:
            self._stream = self._container.add_stream(self.codec, rate=self.fps)
        except Exception:  # noqa: BLE001 - fall back if the runtime codec probe lied.
            if not self.fallback_codec:
                self._container.close()
                self._container = None
                raise
            self.codec = self.fallback_codec
            self.codec_options = dict(self.fallback_codec_options)
            self._stream = self._container.add_stream(self.codec, rate=self.fps)
        self._stream.pix_fmt = self.output_pix_fmt
        if self.thread_type is not None:
            self._stream.thread_type = self.thread_type
        if self.thread_count is not None:
            self._stream.thread_count = self.thread_count
        if self.codec_options:
            self._stream.options = dict(self.codec_options)
        self._shape = None

    def write_frame(self, image: np.ndarray):
        if self._stream is None or self._container is None:
            raise RuntimeError("start() must be called before write_frame().")

        frame_array = np.asarray(image)
        if frame_array.ndim != 3 or frame_array.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {frame_array.shape}.")
        if frame_array.dtype != np.uint8:
            raise TypeError(f"Expected uint8 image, got dtype {frame_array.dtype}.")

        frame_array = np.ascontiguousarray(frame_array)
        if self._shape is None:
            self._shape = frame_array.shape
            height, width, _ = frame_array.shape
            self._stream.width = width
            self._stream.height = height
        elif frame_array.shape != self._shape:
            raise ValueError(f"Frame shape changed from {self._shape} to {frame_array.shape}.")

        video_frame = av.VideoFrame.from_ndarray(frame_array, format=self.input_pix_fmt)
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)

    def stop(self):
        if self._stream is not None and self._container is not None:
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()
        self._container = None
        self._stream = None
        self._shape = None

    def abort(self):
        """Discard an unfinished output without flushing delayed frames."""
        container = self._container
        self._container = None
        self._stream = None
        self._shape = None
        if container is not None:
            container.close()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
