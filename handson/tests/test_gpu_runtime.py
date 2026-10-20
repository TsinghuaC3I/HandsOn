from __future__ import annotations

import sys
from types import SimpleNamespace

from handson.data import video_writer
from handson.runtime import gpu


def test_gpu_environment_prefers_nvidia_glfw_on_desktop(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "_nvidia_runtime_available", lambda: True)

    env = gpu.configure_gpu_environment({"DISPLAY": ":1"}, render_mode="offscreen")

    assert env["MUJOCO_GL"] == "glfw"
    assert env["HANDSON_GPU_RENDERING"] == "1"
    assert env["HANDSON_PHYSICS_BACKEND"] == "mujoco_cpu"
    assert env["__GLX_VENDOR_LIBRARY_NAME"] == "nvidia"
    assert env["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert env["__EGL_VENDOR_LIBRARY_FILENAMES"].endswith("nvidia_egl_vendor.json")


def test_gpu_environment_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "_nvidia_runtime_available", lambda: True)

    env = gpu.configure_gpu_environment(
        {"DISPLAY": ":1", "HANDSON_GPU_MODE": "off"},
        render_mode="human",
    )

    assert env["MUJOCO_GL"] == "glfw"
    assert env["HANDSON_GPU_RENDERING"] == "0"
    assert "__GLX_VENDOR_LIBRARY_NAME" not in env
    assert env["HANDSON_VIDEO_ENCODER"] == "cpu"


def test_isaac_gpu_environment_uses_explicit_device_without_cuda_mask(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gpu, "_nvidia_runtime_available", lambda: True)

    env = gpu.configure_isaac_gpu_environment(
        {
            "DISPLAY": ":1",
            "CUDA_VISIBLE_DEVICES": "0",
            "HANDSON_GPU_DEVICE": "1",
        },
        render_mode="human",
    )

    assert env["HANDSON_ISAACLAB_DEVICE"] == "cuda:1"
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_isaac_gpu_environment_can_preserve_required_container_mask(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gpu, "_nvidia_runtime_available", lambda: True)

    env = gpu.configure_isaac_gpu_environment(
        {
            "CUDA_VISIBLE_DEVICES": "GPU-container-device",
            gpu.ISAAC_PRESERVE_CUDA_VISIBLE_DEVICES_ENV: "1",
        }
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-container-device"


def test_gpu_status_probes_nvenc_only_once_across_ui_polls(monkeypatch) -> None:
    codec_calls: list[tuple[str, str]] = []

    def codec(name: str, mode: str) -> object:
        codec_calls.append((name, mode))
        return object()

    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(codec=SimpleNamespace(Codec=codec)),
    )
    monkeypatch.setattr(gpu, "_nvidia_runtime_available", lambda: True)
    gpu._video_encoder_available.cache_clear()
    try:
        first = gpu.gpu_status({})
        second = gpu.gpu_status({})
    finally:
        gpu._video_encoder_available.cache_clear()

    assert first["nvenc_available"] is True
    assert second["nvenc_available"] is True
    assert codec_calls == [("h264_nvenc", "w")]


def test_h264_writer_uses_nvenc_when_available(monkeypatch) -> None:
    monkeypatch.setenv("HANDSON_VIDEO_ENCODER", "auto")
    monkeypatch.setattr(video_writer, "_encoder_available", lambda name: name == "h264_nvenc")

    writer = video_writer.Mp4VideoWriter.create_h264(fps=30, codec="h264", crf=21)

    assert writer.codec == "h264_nvenc"
    assert writer.fallback_codec == "h264"
    assert writer.codec_options["cq"] == "21"


def test_h264_writer_falls_back_when_nvenc_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("HANDSON_VIDEO_ENCODER", "nvenc")
    monkeypatch.setattr(video_writer, "_encoder_available", lambda name: False)

    writer = video_writer.Mp4VideoWriter.create_h264(fps=30, codec="h264", crf=21)

    assert writer.codec == "h264"
    assert writer.codec_options["crf"] == "21"
    assert writer.fallback_codec is None
