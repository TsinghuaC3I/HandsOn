"""Shared Isaac Sim desktop-display settings for collection and preview."""

from __future__ import annotations

from collections.abc import Mapping
import math
import os


_TRUE_VALUES = {"1", "true", "yes", "on"}
_DEFAULT_PATH_TRACING_SPP_PER_ITERATION = 4
_DEFAULT_PATH_TRACING_TOTAL_SPP = 64
_DEFAULT_PATH_TRACING_MAX_BOUNCES = 8
_DEFAULT_PATH_TRACING_MAX_SPECULAR_BOUNCES = 8
_DEFAULT_PATH_TRACING_MAX_VOLUME_BOUNCES = 4
_DEFAULT_STUDIO_CAMERA_EXPOSURE_EV = -0.35
_DEFAULT_LAB3_VIEWPORT_WIDTH = 1440
_DEFAULT_LAB3_VIEWPORT_HEIGHT = 900
_DEFAULT_DLSS_EXEC_MODE = 2
_LIVE_RENDER_ENV_KEYS = (
    "HANDSON_ISAACSIM_VIEWPORT_WIDTH",
    "HANDSON_ISAACSIM_VIEWPORT_HEIGHT",
    "HANDSON_ISAACSIM_REALTIME_AA",
    "HANDSON_ISAACSIM_DLSS_EXEC_MODE",
    "HANDSON_ISAACSIM_DISABLE_FABRIC_RENDER_TRANSFORMS",
)
_OFFLINE_RENDER_ENV = {
    "HANDSON_ISAACSIM_ULTRA_RENDERING": "1",
    "HANDSON_ISAACSIM_PATH_TRACING_SPP": "4",
    "HANDSON_ISAACSIM_PATH_TRACING_TOTAL_SPP": "16",
    "HANDSON_ISAACSIM_PATH_TRACING_MAX_BOUNCES": "6",
    "HANDSON_ISAACSIM_PATH_TRACING_MAX_SPECULAR_BOUNCES": "6",
    "HANDSON_ISAACSIM_PATH_TRACING_MAX_VOLUME_BOUNCES": "2",
    "HANDSON_ISAACSIM_ACCURATE_CAMERA_VISIBILITY": "1",
}
_VIEWPORT_ONLY_SETTINGS = {
    "/app/window/hideUi": True,
    "/persistent/app/viewport/noPadding": True,
}
_HIDE_VIEWPORT_HELPERS_SETTING = "/persistent/app/viewport/displayOptions"
_HIDE_VIEWPORT_HELPERS_ENV = "HANDSON_ISAACSIM_HIDE_VIEWPORT_HELPERS"
_VIEWPORT_ONLY_KIT_ARGS = tuple(
    f"--{setting}={'true' if value else 'false'}"
    for setting, value in _VIEWPORT_ONLY_SETTINGS.items()
)
_TELEMETRY_DISABLED_KIT_ARGS = (
    "--/telemetry/enableAnonymousData=false",
    "--/telemetry/enableAnonymousAppName=false",
    "--/privacy/usage=false",
    "--/privacy/performance=false",
    "--/privacy/personalization=false",
)
_STABLE_RENDERING_KIT_ARGS = (
    # A preview/collection process owns one explicitly selected GPU. Letting
    # Kit's renderer activate every enumerated GPU reserves VRAM cluster-wide
    # and can collide with unrelated jobs even though PhysX uses one device.
    "--/renderer/multiGpu/enabled=false",
    # Use DLSS Quality for the interactive viewport and camera render
    # products.  It reconstructs a cleaner native-resolution image than FXAA
    # without increasing the direct-light samples per pixel.
    "--/rtx/post/aa/op=3",
    "--/rtx-transient/post/aa/limitedOps=false",
    # Keep direct lighting on the deterministic LTC path.  Isaac Sim 6 can
    # otherwise enable one-sample stochastic lighting for the viewport and
    # every recording render product.  Those samples show up as moving
    # salt-and-pepper noise before DLSS reconstruction, and each camera also
    # pays the sampled-lighting cost.  Do not force the
    # sampled path back on from the transient renderer settings.
    "--/rtx/directLighting/sampledLighting/enabled=false",
    "--/rtx-transient/directLighting/sampledLighting/enforced=false",
    # Isaac Lab 3 defaults Dome Lights to full IBL sampling.  In real-time RTX
    # that produces persistent salt-and-pepper sampling artifacts, even when
    # direct-light samples-per-pixel is raised.  Limited IBL is artifact-free
    # and leaves the scene's Distant Light as the stable direct-light source.
    "--/rtx/domeLight/upperLowerStrategy=3",
    # Auto exposure is global to the renderer.  Head and wrist render products
    # otherwise compete for the same histogram and make one another pulse.
    "--/rtx/post/histogram/enabled=false",
)
_FABRIC_RENDER_COMPATIBILITY_KIT_ARGS = (
    # Isaac Sim 6's RTX Hydra warns that this direct transform path is
    # incompatible with its derived geometry-streaming path.  Physics Fabric
    # remains enabled; publishing transforms through USD is scoped to the
    # opted-in live task profile.
    "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false",
    "--/physics/updateToUsd=true",
)


def isaac_realtime_antialiasing(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return the bounded real-time AA profile used for live collection."""

    source = os.environ if environ is None else environ
    value = str(source.get("HANDSON_ISAACSIM_REALTIME_AA", "dlss"))
    value = value.strip().lower().replace("-", "_")
    if value in {"fxaa", "fast_approximate"}:
        return "fxaa"
    if value in {"taa", "temporal"}:
        return "taa"
    return "dlss"


def _stable_antialiasing_kit_args(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    profile = isaac_realtime_antialiasing(environ=environ)
    if profile == "fxaa":
        # Isaac Lab's rendering experience enables NGX independently from the
        # selected post-AA operation.  Without this startup switch Kit eagerly
        # creates a DLSS context even though the viewport uses FXAA.
        return ("--/ngx/enabled=false", "--/rtx/post/aa/op=2")
    if profile == "taa":
        return ("--/ngx/enabled=false", "--/rtx/post/aa/op=1")
    return (
        "--/rtx/post/aa/op=3",
        "--/rtx/post/dlss/execMode="
        f"{isaac_dlss_exec_mode(environ=environ)}",
    )


def isaac_dlss_exec_mode(
    environ: Mapping[str, str] | None = None,
) -> int:
    """Return the bounded DLSS profile for a real-time Isaac process."""

    source = os.environ if environ is None else environ
    return _int_env(
        source,
        "HANDSON_ISAACSIM_DLSS_EXEC_MODE",
        _DEFAULT_DLSS_EXEC_MODE,
        minimum=0,
        maximum=4,
    )


def configure_isaac_offline_render_environment(
    environ: Mapping[str, str],
) -> dict[str, str]:
    """Replace live viewport tuning with a bounded high-quality replay profile."""

    configured = {str(key): str(value) for key, value in environ.items()}
    for key in _LIVE_RENDER_ENV_KEYS:
        configured.pop(key, None)
    configured.update(_OFFLINE_RENDER_ENV)
    # The default stays compatible with existing collection. An explicit
    # offline quality selection survives the worker's live-profile cleanup.
    # Reuse the bounded parser rather than accepting arbitrary renderer flags.
    for key in _OFFLINE_RENDER_ENV:
        if key.startswith("HANDSON_ISAACSIM_PATH_TRACING_") and key in environ:
            configured[key] = str(environ[key])
    profile = isaac_path_tracing_settings(environ=configured)
    for key, value in (
        ("SPP", profile["spp"]),
        ("TOTAL_SPP", profile["total_spp"]),
        ("MAX_BOUNCES", profile["max_bounces"]),
        ("MAX_SPECULAR_BOUNCES", profile["max_specular_bounces"]),
        ("MAX_VOLUME_BOUNCES", profile["max_volume_bounces"]),
    ):
        configured["HANDSON_ISAACSIM_PATH_TRACING_" + key] = str(value)
    return configured


def _int_env(
    source: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(str(source.get(name, default)).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def isaac_path_tracing_settings(
    environ: Mapping[str, str] | None = None,
) -> dict[str, int | float | bool | str]:
    """Return the bounded offline Path Tracing profile for this process."""

    source = os.environ if environ is None else environ
    spp = _int_env(
        source,
        "HANDSON_ISAACSIM_PATH_TRACING_SPP",
        _DEFAULT_PATH_TRACING_SPP_PER_ITERATION,
        minimum=1,
        maximum=64,
    )
    total_spp = _int_env(
        source,
        "HANDSON_ISAACSIM_PATH_TRACING_TOTAL_SPP",
        _DEFAULT_PATH_TRACING_TOTAL_SPP,
        minimum=spp,
        maximum=4096,
    )
    return {
        "renderer": "PathTracing",
        "spp": spp,
        "total_spp": total_spp,
        "max_bounces": _int_env(
            source,
            "HANDSON_ISAACSIM_PATH_TRACING_MAX_BOUNCES",
            _DEFAULT_PATH_TRACING_MAX_BOUNCES,
            minimum=1,
            maximum=64,
        ),
        "max_specular_bounces": _int_env(
            source,
            "HANDSON_ISAACSIM_PATH_TRACING_MAX_SPECULAR_BOUNCES",
            _DEFAULT_PATH_TRACING_MAX_SPECULAR_BOUNCES,
            minimum=1,
            maximum=64,
        ),
        "max_volume_bounces": _int_env(
            source,
            "HANDSON_ISAACSIM_PATH_TRACING_MAX_VOLUME_BOUNCES",
            _DEFAULT_PATH_TRACING_MAX_VOLUME_BOUNCES,
            minimum=0,
            maximum=64,
        ),
        "denoiser": True,
        # State replay jumps to a new pose every saved frame.  A temporal
        # denoiser can reuse invalid history and make those sharp poses smear.
        "temporal_denoiser": False,
        # Delivery footage favors a fully denoised static accumulation.  At
        # 4K/64 spp, retaining even ten percent of the raw image makes bright
        # path-tracing fireflies visible on the neutral robot shell.
        "denoiser_blend": 0.0,
    }


def isaac_path_tracing_capture_passes(
    environ: Mapping[str, str] | None = None,
) -> int:
    """Return renders needed to reach total SPP without one risky GPU launch."""

    profile = isaac_path_tracing_settings(environ=environ)
    return int(math.ceil(int(profile["total_spp"]) / int(profile["spp"])))


def _ultra_rendering_kit_args(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Build startup fallbacks; SimulationApp config remains authoritative."""

    profile = isaac_path_tracing_settings(environ=environ)
    return (
        "--/rtx/rendermode=PathTracing",
        "--/rtx/post/aa/op=0",
        f"--/rtx/pathtracing/spp={profile['spp']}",
        f"--/rtx/pathtracing/totalSpp={profile['total_spp']}",
        f"--/rtx/pathtracing/clampSpp={profile['total_spp']}",
        f"--/rtx/pathtracing/maxBounces={profile['max_bounces']}",
        "--/rtx/pathtracing/maxSpecularAndTransmissionBounces="
        f"{profile['max_specular_bounces']}",
        f"--/rtx/pathtracing/maxVolumeBounces={profile['max_volume_bounces']}",
        "--/rtx/pathtracing/optixDenoiser/enabled=true",
        "--/rtx/pathtracing/optixDenoiser/temporalMode/enabled=false",
        f"--/rtx/pathtracing/optixDenoiser/blendFactor={profile['denoiser_blend']}",
        # Box sampling keeps 4K edges crisp; multiple PT samples provide the
        # antialiasing, so DLAA's temporal history is deliberately bypassed.
        "--/rtx/pathtracing/aa/op=0",
        "--/rtx/pathtracing/aa/filterRadius=0.5",
        "--/rtx/domeLight/upperLowerStrategy=0",
        "--/rtx/post/histogram/enabled=false",
        "--/rtx/post/tonemap/op=6",
    )


def isaac_viewport_only_enabled(
    *,
    headless: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether a desktop Isaac session should show only its viewport."""
    if headless:
        return False
    source = os.environ if environ is None else environ
    value = source.get("HANDSON_ISAACSIM_VIEWPORT_ONLY")
    if value is None:
        return True
    return value.strip().lower() in _TRUE_VALUES


def isaac_hide_viewport_helpers_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether non-scene viewport gizmos should be hidden.

    Isaac's legacy display-options bitmask draws camera/light/target helpers
    over the operator viewport.  Those helpers are useful while authoring a
    scene but are misleading during collection, where they look like extra
    transparent assets.  Keep this opt-in so other task previews retain their
    normal authoring aids.
    """

    source = os.environ if environ is None else environ
    value = source.get(_HIDE_VIEWPORT_HELPERS_ENV)
    return value is not None and value.strip().lower() in _TRUE_VALUES


def isaac_telemetry_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether Isaac's optional anonymous telemetry should remain enabled."""
    source = os.environ if environ is None else environ
    value = source.get("HANDSON_ISAACSIM_TELEMETRY")
    if value is None:
        return False
    return value.strip().lower() in _TRUE_VALUES


def isaac_stable_rendering_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether deterministic real-time rendering should be enforced."""
    source = os.environ if environ is None else environ
    value = source.get("HANDSON_ISAACSIM_STABLE_RENDERING")
    if value is None:
        return True
    return value.strip().lower() in _TRUE_VALUES


def isaac_disable_fabric_render_transforms_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether RTX should use scene-delegate transform updates.

    Isaac Sim 6 derives geometry-streaming activation after command-line
    settings are loaded.  Its rendering experience also enables direct
    Fabric transform reads; the two paths together are explicitly unsupported
    by RTX Hydra and can leave articulated links visually stale.  B320 opts
    into the compatible scene-delegate path without changing physics Fabric
    or other tasks.
    """

    source = os.environ if environ is None else environ
    value = source.get("HANDSON_ISAACSIM_DISABLE_FABRIC_RENDER_TRANSFORMS")
    if value is None:
        return False
    return value.strip().lower() in _TRUE_VALUES


def isaac_ultra_rendering_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether maximum-quality offline RTX rendering is requested."""
    source = os.environ if environ is None else environ
    value = source.get("HANDSON_ISAACSIM_ULTRA_RENDERING")
    if value is None:
        return False
    return value.strip().lower() in _TRUE_VALUES


def isaac_studio_rendering_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the B03 PhysX studio-light/material profile is requested."""

    source = os.environ if environ is None else environ
    value = source.get("HANDSON_ISAACSIM_STUDIO_RENDERING")
    if value is None:
        return False
    return value.strip().lower() in _TRUE_VALUES


def isaac_studio_camera_exposure(
    environ: Mapping[str, str] | None = None,
) -> float:
    """Return the per-camera exposure offset in EV stops for studio output."""

    source = os.environ if environ is None else environ
    try:
        value = float(
            str(
                source.get(
                    "HANDSON_ISAACSIM_STUDIO_CAMERA_EXPOSURE_EV",
                    _DEFAULT_STUDIO_CAMERA_EXPOSURE_EV,
                )
            ).strip()
        )
    except (TypeError, ValueError):
        return _DEFAULT_STUDIO_CAMERA_EXPOSURE_EV
    return max(-10.0, min(10.0, value))


def isaac_rendering_mode(
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Return the Isaac Lab preset explicitly selected for this process."""
    return "quality" if isaac_ultra_rendering_enabled(environ=environ) else None


def configure_isaac_renderer_launcher(
    launcher_args: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Put Path Tracing in SimulationApp's authoritative launch config.

    Kit command-line RTX settings are applied before ``SimulationApp`` resets
    the renderer from this config.  Supplying these fields is therefore what
    prevents an ultra run from silently falling back to RaytracedLighting.
    """

    if not isaac_ultra_rendering_enabled(environ=environ):
        return False
    profile = isaac_path_tracing_settings(environ=environ)
    setattr(launcher_args, "renderer", profile["renderer"])
    setattr(launcher_args, "anti_aliasing", 0)
    setattr(launcher_args, "samples_per_pixel_per_frame", profile["spp"])
    setattr(launcher_args, "denoiser", profile["denoiser"])
    setattr(launcher_args, "max_bounces", profile["max_bounces"])
    setattr(
        launcher_args,
        "max_specular_transmission_bounces",
        profile["max_specular_bounces"],
    )
    setattr(
        launcher_args,
        "max_volume_bounces",
        profile["max_volume_bounces"],
    )
    return True


def enforce_isaac_live_render_compatibility(
    simulation_app: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Re-apply and verify the opted-in dynamic-geometry renderer profile.

    ``SimulationApp`` overwrites the command-line AA operation while it builds
    the renderer.  The pre-launch flags prevent the incompatible RTX Hydra
    context from being created; this post-launch gate restores the requested
    AA operation and verifies all three effective settings before scene setup.
    """

    del simulation_app  # The running app establishes the carb settings owner.
    if not isaac_disable_fabric_render_transforms_enabled(environ=environ):
        return False

    import carb

    profile = isaac_realtime_antialiasing(environ=environ)
    aa_op = {"taa": 1, "fxaa": 2, "dlss": 3}[profile]
    settings = carb.settings.get_settings()
    expected = {
        "/rtx/hydra/readTransformsFromFabricInRenderDelegate": False,
        "/physics/updateToUsd": True,
        "/rtx/post/aa/op": aa_op,
    }
    for setting, value in expected.items():
        settings.set(setting, value)

    actual_fabric = bool(
        settings.get("/rtx/hydra/readTransformsFromFabricInRenderDelegate")
    )
    actual_update_to_usd = bool(settings.get("/physics/updateToUsd"))
    actual_aa_op = int(settings.get_as_int("/rtx/post/aa/op"))
    if actual_fabric or not actual_update_to_usd or actual_aa_op != aa_op:
        raise RuntimeError(
            "Isaac live renderer did not retain the dynamic-geometry safety "
            "profile: "
            f"fabric_render_transforms={actual_fabric}, "
            f"physics_update_to_usd={actual_update_to_usd}, "
            f"aa_op={actual_aa_op} (expected {aa_op})"
        )
    print(
        "[HandsOn/IsaacLab] Live dynamic-geometry renderer: "
        f"AA={profile.upper()}, Fabric render transforms=off, "
        "physics.updateToUsd=on",
        flush=True,
    )
    return True


def enforce_isaac_ultra_rendering(
    simulation_app: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Re-apply and verify the complete offline PT profile after Kit startup."""

    if not isaac_ultra_rendering_enabled(environ=environ):
        return False

    import carb

    profile = isaac_path_tracing_settings(environ=environ)
    reset_render_settings = getattr(simulation_app, "reset_render_settings", None)
    if callable(reset_render_settings):
        reset_render_settings()
    settings = carb.settings.get_settings()
    values = {
        "/rtx/rendermode": "PathTracing",
        "/rtx/post/aa/op": 0,
        "/rtx/pathtracing/spp": int(profile["spp"]),
        "/rtx/pathtracing/totalSpp": int(profile["total_spp"]),
        "/rtx/pathtracing/clampSpp": int(profile["total_spp"]),
        "/rtx/pathtracing/maxBounces": int(profile["max_bounces"]),
        "/rtx/pathtracing/maxSpecularAndTransmissionBounces": int(
            profile["max_specular_bounces"]
        ),
        "/rtx/pathtracing/maxVolumeBounces": int(profile["max_volume_bounces"]),
        "/rtx/pathtracing/optixDenoiser/enabled": True,
        "/rtx/pathtracing/optixDenoiser/temporalMode/enabled": False,
        "/rtx/pathtracing/optixDenoiser/blendFactor": float(
            profile["denoiser_blend"]
        ),
        "/rtx/pathtracing/aa/op": 0,
        "/rtx/pathtracing/aa/filterRadius": 0.5,
        "/rtx/domeLight/upperLowerStrategy": 0,
        "/rtx/post/histogram/enabled": False,
        "/rtx/post/tonemap/op": 6,
    }
    for setting, value in values.items():
        settings.set(setting, value)
    if isaac_studio_rendering_enabled(environ=environ):
        # Keep the HDRI available for reflections and indirect lighting while
        # presenting a quiet neutral slate background instead of either the
        # photo-studio panorama or a clipped-black void.
        settings.set("/rtx/background/source/type", 2)
        settings.set_float_array(
            "/rtx/background/source/color",
            [0.025, 0.035, 0.055],
        )

    actual_mode = str(settings.get("/rtx/rendermode") or "")
    actual_spp = int(settings.get_as_int("/rtx/pathtracing/spp"))
    actual_total_spp = int(settings.get_as_int("/rtx/pathtracing/totalSpp"))
    if (
        actual_mode != "PathTracing"
        or actual_spp != int(profile["spp"])
        or actual_total_spp != int(profile["total_spp"])
    ):
        raise RuntimeError(
            "Isaac ultra renderer did not retain the requested Path Tracing "
            f"profile: mode={actual_mode!r}, spp={actual_spp}, "
            f"total_spp={actual_total_spp}"
        )
    print(
        "[HandsOn/IsaacLab] Offline renderer: PathTracing "
        f"{actual_spp} spp/iteration, {actual_total_spp} total spp, "
        f"{profile['max_bounces']} bounces, OptiX denoiser "
        f"blend={profile['denoiser_blend']:.2f} "
        "(temporal history disabled)",
        flush=True,
    )
    return True


def isaac_app_renderer_resolution(
    width: int,
    height: int,
    *,
    headless: bool = False,
    isaaclab_v3: bool = False,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Return the Kit app-renderer size without changing recording cameras.

    Isaac Sim 6's desktop window defaults to 1440x900.  Launching its app
    renderer at a recording camera's 640x640 resolution makes Kit stretch the
    viewport and look pixelated.  Only Lab 3 desktop sessions use the larger
    app renderer; headless recording products retain their canonical sizes.
    """

    source = os.environ if environ is None else environ
    if headless or not isaaclab_v3:
        return int(width), int(height)

    def _dimension(name: str, default: int) -> int:
        raw = str(source.get(name, "")).strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        return value if value > 0 else default

    return (
        _dimension("HANDSON_ISAACSIM_VIEWPORT_WIDTH", _DEFAULT_LAB3_VIEWPORT_WIDTH),
        _dimension("HANDSON_ISAACSIM_VIEWPORT_HEIGHT", _DEFAULT_LAB3_VIEWPORT_HEIGHT),
    )


def isaac_kit_args(
    existing: str | None = None,
    *,
    headless: bool = False,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Append HandsOn's low-overhead display and telemetry settings for Kit."""
    arguments = str(existing or "").split()
    required_arguments: tuple[str, ...] = ()
    if not isaac_telemetry_enabled(environ=environ):
        required_arguments += _TELEMETRY_DISABLED_KIT_ARGS
    if isaac_ultra_rendering_enabled(environ=environ):
        required_arguments += _ultra_rendering_kit_args(environ=environ)
    elif isaac_stable_rendering_enabled(environ=environ):
        required_arguments += _stable_antialiasing_kit_args(
            environ=environ
        ) + _STABLE_RENDERING_KIT_ARGS
    if isaac_disable_fabric_render_transforms_enabled(environ=environ):
        required_arguments += _FABRIC_RENDER_COMPATIBILITY_KIT_ARGS
    source = os.environ if environ is None else environ
    portable_root = str(source.get("HANDSON_ISAACSIM_PORTABLE_ROOT", "")).strip()
    if portable_root and not any(
        argument == "--portable-root" or argument.startswith("--portable-root=")
        for argument in arguments
    ):
        # Keep an isolated development collection and its fresh camera worker
        # out of the installed runtime's shared config/cache/log directories.
        arguments.extend(("--portable-root", portable_root))
    renderer_gpu = str(
        source.get("HANDSON_ISAACSIM_RENDERER_GPU", "")
    ).strip()
    if renderer_gpu:
        if not renderer_gpu.isdigit():
            raise ValueError(
                "HANDSON_ISAACSIM_RENDERER_GPU must be a non-negative integer, "
                f"got {renderer_gpu!r}"
            )
        # CUDA_VISIBLE_DEVICES remaps the selected physical CUDA device to
        # logical cuda:0. Vulkan keeps physical ordinals, so Kit must receive
        # the audited physical renderer index explicitly.
        required_arguments += (
            "--/renderer/multiGpu/autoEnable=false",
            "--/renderer/multiGpu/enabled=false",
            "--/renderer/multiGpu/maxGpuCount=1",
            f"--/renderer/activeGpu={renderer_gpu}",
        )
    if isaac_viewport_only_enabled(headless=headless, environ=environ):
        required_arguments += _VIEWPORT_ONLY_KIT_ARGS
    if not headless and isaac_hide_viewport_helpers_enabled(environ=environ):
        required_arguments += (
            f"--{_HIDE_VIEWPORT_HELPERS_SETTING}=0",
        )
    for required in required_arguments:
        setting = required.split("=", 1)[0]
        if not any(argument == setting or argument.startswith(f"{setting}=") for argument in arguments):
            arguments.append(required)
    return " ".join(arguments)


def enforce_isaac_viewport_only(
    simulation_app: object,
    *,
    headless: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Re-apply viewport-only settings after Kit startup and report whether enabled."""
    if not isaac_viewport_only_enabled(headless=headless, environ=environ):
        return False

    import carb

    settings = carb.settings.get_settings()
    for setting, value in _VIEWPORT_ONLY_SETTINGS.items():
        settings.set(setting, value)
    if isaac_hide_viewport_helpers_enabled(environ=environ):
        settings.set(_HIDE_VIEWPORT_HELPERS_SETTING, 0)
    update = getattr(simulation_app, "update", None)
    if callable(update):
        update()
    print(
        "[HandsOn/IsaacLab] Desktop display: viewport-only "
        "(set HANDSON_ISAACSIM_VIEWPORT_ONLY=0 to restore the full Isaac UI)",
        flush=True,
    )
    return True
