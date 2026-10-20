"""GPU runtime configuration for MuJoCo rendering and video output."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, MutableMapping


NVIDIA_OFFLOAD_ENV = {
    "__NV_PRIME_RENDER_OFFLOAD": "1",
    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    "__VK_LAYER_NV_optimus": "NVIDIA_only",
}
NVIDIA_EGL_VENDOR_JSON = Path(__file__).with_name("nvidia_egl_vendor.json")
ISAAC_PRESERVE_CUDA_VISIBLE_DEVICES_ENV = (
    "HANDSON_ISAAC_PRESERVE_CUDA_VISIBLE_DEVICES"
)
P003_CPU_PHYSX_GROUND_TRUTH_EXCEPTION = (
    "p003_exact_ruler035_cpu_physx_v1"
)
PIZZA_CPU_PHYSX_GROUND_TRUTH_EXCEPTION = (
    "user_pizza_operator_cpu_v1"
)


def is_canonical_cuda_device(device: Any) -> bool:
    """Accept only the exact ASCII ``cuda:N`` spelling used by replay."""

    if not isinstance(device, str) or not device.startswith("cuda:"):
        return False
    index = device[len("cuda:") :]
    return bool(
        index == "0"
        or (
            index
            and index[0] in "123456789"
            and all(character in "0123456789" for character in index)
        )
    )


def is_physx_recording_device_allowed(
    device: Any,
    *,
    cpu_exception: Any = "",
) -> bool:
    """Allow CUDA and explicitly schema-bound CPU operator-test contracts."""

    return bool(
        is_canonical_cuda_device(device)
        or (
            device == "cpu"
            and cpu_exception
            in {
                P003_CPU_PHYSX_GROUND_TRUTH_EXCEPTION,
                PIZZA_CPU_PHYSX_GROUND_TRUTH_EXCEPTION,
            }
        )
    )


def _env_enabled(env: MutableMapping[str, str], name: str, default: bool = True) -> bool:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _nvidia_runtime_available() -> bool:
    return any(
        path.exists()
        for path in (
            Path("/proc/driver/nvidia/version"),
            Path("/lib/x86_64-linux-gnu/libGLX_nvidia.so.0"),
            Path("/lib/x86_64-linux-gnu/libEGL_nvidia.so.0"),
        )
    )


@lru_cache(maxsize=None)
def _video_encoder_available(codec_name: str) -> bool:
    """Probe each codec once instead of reopening FFmpeg on every UI poll."""

    try:
        import av

        av.codec.Codec(codec_name, "w")
    except Exception:  # noqa: BLE001 - optional PyAV codec availability probe.
        return False
    return True


def _default_mujoco_gl(env: MutableMapping[str, str], *, render_mode: str) -> str:
    explicit = (env.get("HANDSON_MUJOCO_GL") or env.get("MUJOCO_GL") or "").strip().lower()
    if explicit:
        return explicit
    if render_mode == "human":
        return "glfw"
    if env.get("DISPLAY"):
        return "glfw"
    return "egl"


def configure_gpu_environment(
    env: MutableMapping[str, str] | None = None,
    *,
    render_mode: str = "auto",
) -> dict[str, str]:
    """Return an environment configured for the best local GPU render path.

    MuJoCo physics still runs on CPU. This function only configures OpenGL/EGL
    rendering and optional encoder selection for child processes.
    """

    configured = dict(os.environ if env is None else env)
    gpu_mode = (configured.get("HANDSON_GPU_MODE") or "auto").strip().lower()
    gpu_allowed = gpu_mode not in {"0", "false", "no", "off", "cpu"}
    nvidia_available = _nvidia_runtime_available()
    gpu_enabled = gpu_allowed and nvidia_available

    configured["MUJOCO_GL"] = _default_mujoco_gl(configured, render_mode=render_mode)
    configured["HANDSON_GPU_RENDERING"] = "1" if gpu_enabled else "0"
    configured.setdefault("HANDSON_PHYSICS_BACKEND", "mujoco_cpu")

    if gpu_enabled:
        for key, value in NVIDIA_OFFLOAD_ENV.items():
            configured.setdefault(key, value)
        configured.setdefault("MUJOCO_EGL_DEVICE_ID", configured.get("HANDSON_GPU_DEVICE", "0"))
        if NVIDIA_EGL_VENDOR_JSON.exists():
            configured.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(NVIDIA_EGL_VENDOR_JSON))
        configured.setdefault("CUDA_VISIBLE_DEVICES", configured.get("HANDSON_GPU_DEVICE", "0"))
    else:
        for key in NVIDIA_OFFLOAD_ENV:
            configured.pop(key, None)
        configured.setdefault("HANDSON_VIDEO_ENCODER", "cpu")

    configured.setdefault("HANDSON_VIDEO_ENCODER", "auto")
    configured.setdefault("HANDSON_VIDEO_GPU", configured.get("HANDSON_GPU_DEVICE", "0"))
    return configured


def configure_isaac_gpu_environment(
    env: MutableMapping[str, str] | None = None,
    *,
    render_mode: str = "auto",
) -> dict[str, str]:
    """Configure an Isaac process without remapping CUDA device ordinals.

    Isaac/Omniverse enumerates the Vulkan and CUDA devices independently and
    warns that ``CUDA_VISIBLE_DEVICES`` can make those ordinals disagree.  Use
    Isaac Lab's explicit device setting instead.  Containers that require a
    CUDA visibility mask can opt back in with
    ``HANDSON_ISAAC_PRESERVE_CUDA_VISIBLE_DEVICES=1``.
    """

    source = dict(os.environ if env is None else env)
    configured = configure_gpu_environment(source, render_mode=render_mode)
    gpu_device = str(source.get("HANDSON_GPU_DEVICE", "0")).strip() or "0"
    configured.setdefault("HANDSON_ISAACLAB_DEVICE", f"cuda:{gpu_device}")
    if not _env_enabled(
        source,
        ISAAC_PRESERVE_CUDA_VISIBLE_DEVICES_ENV,
        default=False,
    ):
        configured.pop("CUDA_VISIBLE_DEVICES", None)
    return configured


def gpu_status(env: MutableMapping[str, str] | None = None) -> dict[str, object]:
    current = dict(os.environ if env is None else env)
    configured = configure_gpu_environment(current)
    return {
        "mode": configured.get("HANDSON_GPU_MODE", "auto"),
        "nvidia_available": _nvidia_runtime_available(),
        "rendering_enabled": configured.get("HANDSON_GPU_RENDERING") == "1",
        "mujoco_gl": configured.get("MUJOCO_GL", ""),
        "egl_device_id": configured.get("MUJOCO_EGL_DEVICE_ID", ""),
        "egl_vendor_file": configured.get("__EGL_VENDOR_LIBRARY_FILENAMES", ""),
        "nvidia_offload": all(configured.get(key) == value for key, value in NVIDIA_OFFLOAD_ENV.items()),
        "physics_backend": configured.get("HANDSON_PHYSICS_BACKEND", "mujoco_cpu"),
        "video_encoder": configured.get("HANDSON_VIDEO_ENCODER", "auto"),
        "nvenc_available": _video_encoder_available("h264_nvenc"),
        "video_gpu": configured.get("HANDSON_VIDEO_GPU", "0"),
    }
