from __future__ import annotations

import sys
from types import SimpleNamespace

from handson.sim.isaaclab.runtime_display import (
    configure_isaac_offline_render_environment,
    configure_isaac_renderer_launcher,
    enforce_isaac_live_render_compatibility,
    isaac_disable_fabric_render_transforms_enabled,
    isaac_app_renderer_resolution,
    isaac_kit_args,
    isaac_dlss_exec_mode,
    isaac_path_tracing_capture_passes,
    isaac_path_tracing_settings,
    isaac_realtime_antialiasing,
    isaac_rendering_mode,
    isaac_stable_rendering_enabled,
    isaac_studio_camera_exposure,
    isaac_studio_rendering_enabled,
    isaac_telemetry_enabled,
    isaac_ultra_rendering_enabled,
    isaac_viewport_only_enabled,
)


def test_lab3_desktop_renderer_matches_default_isaac_window() -> None:
    assert isaac_app_renderer_resolution(
        640,
        640,
        isaaclab_v3=True,
        environ={},
    ) == (1440, 900)


def test_recording_resolution_is_preserved_outside_lab3_desktop() -> None:
    assert isaac_app_renderer_resolution(
        640,
        640,
        headless=True,
        isaaclab_v3=True,
        environ={},
    ) == (640, 640)
    assert isaac_app_renderer_resolution(
        640,
        640,
        isaaclab_v3=False,
        environ={},
    ) == (640, 640)


def test_lab3_desktop_renderer_size_can_be_overridden() -> None:
    assert isaac_app_renderer_resolution(
        640,
        640,
        isaaclab_v3=True,
        environ={
            "HANDSON_ISAACSIM_VIEWPORT_WIDTH": "1920",
            "HANDSON_ISAACSIM_VIEWPORT_HEIGHT": "1080",
        },
    ) == (1920, 1080)


def test_viewport_only_is_default_for_desktop_sessions() -> None:
    assert isaac_viewport_only_enabled(environ={})
    arguments = isaac_kit_args(environ={}).split()
    assert "--/app/window/hideUi=true" in arguments
    assert "--/persistent/app/viewport/noPadding=true" in arguments


def test_viewport_only_is_not_added_to_headless_sessions_but_telemetry_stays_disabled() -> None:
    assert not isaac_viewport_only_enabled(headless=True, environ={})
    arguments = isaac_kit_args(
        "--/renderer/multiGpu/enabled=false",
        headless=True,
        environ={},
    ).split()
    assert "--/renderer/multiGpu/enabled=false" in arguments
    assert "--/telemetry/enableAnonymousData=false" in arguments
    assert not any("/app/window/hideUi" in argument for argument in arguments)


def test_viewport_only_can_be_disabled_for_editor_debugging() -> None:
    environ = {
        "HANDSON_ISAACSIM_VIEWPORT_ONLY": "0",
        "HANDSON_ISAACSIM_TELEMETRY": "1",
        "HANDSON_ISAACSIM_STABLE_RENDERING": "0",
    }

    assert not isaac_viewport_only_enabled(environ=environ)
    assert isaac_kit_args(environ=environ) == ""


def test_stable_recording_render_is_default_and_can_be_disabled() -> None:
    assert isaac_stable_rendering_enabled(environ={})
    arguments = isaac_kit_args(environ={}).split()
    assert "--/rtx/post/aa/op=3" in arguments
    assert "--/rtx/post/dlss/execMode=2" in arguments
    assert "--/ngx/enabled=false" not in arguments
    assert "--/rtx/directLighting/sampledLighting/enabled=false" in arguments
    assert (
        "--/rtx-transient/directLighting/sampledLighting/enforced=false"
        in arguments
    )
    assert "--/rtx/domeLight/upperLowerStrategy=3" in arguments
    assert not any("samplesPerPixel" in argument for argument in arguments)
    assert "--/rtx/post/histogram/enabled=false" in arguments

    environ = {"HANDSON_ISAACSIM_STABLE_RENDERING": "0"}
    assert not isaac_stable_rendering_enabled(environ=environ)
    assert not any("/rtx/post/aa/op" in item for item in isaac_kit_args(environ=environ).split())


def test_live_collection_can_select_balanced_dlss() -> None:
    environ = {"HANDSON_ISAACSIM_DLSS_EXEC_MODE": "1"}

    assert isaac_dlss_exec_mode(environ=environ) == 1
    arguments = isaac_kit_args(environ=environ).split()
    assert "--/rtx/post/dlss/execMode=1" in arguments
    assert "--/rtx/post/dlss/execMode=2" not in arguments


def test_live_collection_can_avoid_unavailable_dlss_with_fxaa() -> None:
    environ = {"HANDSON_ISAACSIM_REALTIME_AA": "fxaa"}

    assert isaac_realtime_antialiasing(environ=environ) == "fxaa"
    arguments = isaac_kit_args(environ=environ).split()
    assert "--/ngx/enabled=false" in arguments
    assert "--/rtx/post/aa/op=2" in arguments
    assert not any("/rtx/post/dlss/" in item for item in arguments)


def test_live_collection_disables_ngx_for_taa_without_overriding_explicit_kit_args() -> None:
    environ = {"HANDSON_ISAACSIM_REALTIME_AA": "taa"}

    arguments = isaac_kit_args(environ=environ).split()
    assert "--/ngx/enabled=false" in arguments
    assert "--/rtx/post/aa/op=1" in arguments

    explicit = isaac_kit_args(
        "--/ngx/enabled=true --/UJITSO/geometry=true",
        environ=environ,
    ).split()
    assert "--/ngx/enabled=true" in explicit
    assert "--/ngx/enabled=false" not in explicit
    assert "--/UJITSO/geometry=true" in explicit
    assert "--/UJITSO/geometry=false" not in explicit


def test_dynamic_geometry_safety_uses_scene_delegate_transforms_only_when_opted_in() -> None:
    fabric_setting = "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false"
    usd_setting = "--/physics/updateToUsd=true"

    assert not isaac_disable_fabric_render_transforms_enabled(environ={})
    assert fabric_setting not in isaac_kit_args(environ={}).split()
    assert usd_setting not in isaac_kit_args(environ={}).split()

    environ = {"HANDSON_ISAACSIM_DISABLE_FABRIC_RENDER_TRANSFORMS": "1"}
    assert isaac_disable_fabric_render_transforms_enabled(environ=environ)
    assert fabric_setting in isaac_kit_args(environ=environ).split()
    assert usd_setting in isaac_kit_args(environ=environ).split()


def test_dynamic_geometry_safety_reapplies_and_verifies_fxaa_after_app_launch(
    monkeypatch,
) -> None:
    class FakeSettings:
        def __init__(self) -> None:
            self.values = {
                "/rtx/hydra/readTransformsFromFabricInRenderDelegate": True,
                "/physics/updateToUsd": False,
                "/rtx/post/aa/op": 1,
            }

        def set(self, setting: str, value: object) -> None:
            self.values[setting] = value

        def get(self, setting: str) -> object:
            return self.values.get(setting)

        def get_as_int(self, setting: str) -> int:
            return int(self.values[setting])

    settings = FakeSettings()
    fake_carb = SimpleNamespace(
        settings=SimpleNamespace(get_settings=lambda: settings)
    )
    monkeypatch.setitem(sys.modules, "carb", fake_carb)
    environ = {
        "HANDSON_ISAACSIM_DISABLE_FABRIC_RENDER_TRANSFORMS": "1",
        "HANDSON_ISAACSIM_REALTIME_AA": "fxaa",
    }

    assert enforce_isaac_live_render_compatibility(
        SimpleNamespace(),
        environ=environ,
    )
    assert settings.values == {
        "/rtx/hydra/readTransformsFromFabricInRenderDelegate": False,
        "/physics/updateToUsd": True,
        "/rtx/post/aa/op": 2,
    }


def test_offline_profile_drops_live_viewport_tuning() -> None:
    configured = configure_isaac_offline_render_environment(
        {
            "KEEP": "yes",
            "HANDSON_ISAACSIM_VIEWPORT_WIDTH": "960",
            "HANDSON_ISAACSIM_VIEWPORT_HEIGHT": "600",
            "HANDSON_ISAACSIM_REALTIME_AA": "fxaa",
            "HANDSON_ISAACSIM_DLSS_EXEC_MODE": "1",
            "HANDSON_ISAACSIM_DISABLE_FABRIC_RENDER_TRANSFORMS": "1",
        }
    )

    assert configured["KEEP"] == "yes"
    assert "HANDSON_ISAACSIM_VIEWPORT_WIDTH" not in configured
    assert "HANDSON_ISAACSIM_VIEWPORT_HEIGHT" not in configured
    assert "HANDSON_ISAACSIM_REALTIME_AA" not in configured
    assert "HANDSON_ISAACSIM_DLSS_EXEC_MODE" not in configured
    assert "HANDSON_ISAACSIM_DISABLE_FABRIC_RENDER_TRANSFORMS" not in configured
    assert configured["HANDSON_ISAACSIM_ULTRA_RENDERING"] == "1"
    assert configured["HANDSON_ISAACSIM_PATH_TRACING_SPP"] == "4"
    assert configured["HANDSON_ISAACSIM_PATH_TRACING_TOTAL_SPP"] == "16"
    assert configured["HANDSON_ISAACSIM_PATH_TRACING_MAX_BOUNCES"] == "6"
    assert configured["HANDSON_ISAACSIM_ACCURATE_CAMERA_VISIBILITY"] == "1"


def test_ultra_offline_render_uses_true_path_tracing() -> None:
    environ = {"HANDSON_ISAACSIM_ULTRA_RENDERING": "1"}

    assert isaac_ultra_rendering_enabled(environ=environ)
    assert isaac_rendering_mode(environ=environ) == "quality"
    arguments = isaac_kit_args(environ=environ).split()
    assert "--/rtx/rendermode=PathTracing" in arguments
    assert "--/rtx/post/aa/op=0" in arguments
    assert "--/rtx/pathtracing/spp=4" in arguments
    assert "--/rtx/pathtracing/totalSpp=64" in arguments
    assert "--/rtx/pathtracing/clampSpp=64" in arguments
    assert "--/rtx/pathtracing/maxBounces=8" in arguments
    assert "--/rtx/pathtracing/optixDenoiser/enabled=true" in arguments
    assert "--/rtx/pathtracing/optixDenoiser/blendFactor=0.0" in arguments
    assert (
        "--/rtx/pathtracing/optixDenoiser/temporalMode/enabled=false"
        in arguments
    )
    assert "--/rtx/domeLight/upperLowerStrategy=0" in arguments
    assert "--/rtx/directLighting/sampledLighting/enabled=false" not in arguments


def test_ultra_launcher_and_capture_passes_share_bounded_pt_profile() -> None:
    environ = {
        "HANDSON_ISAACSIM_ULTRA_RENDERING": "1",
        "HANDSON_ISAACSIM_PATH_TRACING_SPP": "3",
        "HANDSON_ISAACSIM_PATH_TRACING_TOTAL_SPP": "10",
        "HANDSON_ISAACSIM_PATH_TRACING_MAX_BOUNCES": "12",
    }
    launcher_args = SimpleNamespace()

    assert configure_isaac_renderer_launcher(
        launcher_args,
        environ=environ,
    )
    assert launcher_args.renderer == "PathTracing"
    assert launcher_args.samples_per_pixel_per_frame == 3
    assert launcher_args.max_bounces == 12
    assert launcher_args.denoiser is True
    assert isaac_path_tracing_capture_passes(environ=environ) == 4
    assert isaac_path_tracing_settings(environ=environ)["total_spp"] == 10


def test_studio_profile_and_camera_exposure_are_explicit_and_bounded() -> None:
    assert not isaac_studio_rendering_enabled(environ={})
    assert isaac_studio_rendering_enabled(
        environ={"HANDSON_ISAACSIM_STUDIO_RENDERING": "1"}
    )
    assert isaac_studio_camera_exposure(environ={}) == -0.35
    assert isaac_studio_camera_exposure(
        environ={"HANDSON_ISAACSIM_STUDIO_CAMERA_EXPOSURE_EV": "20"}
    ) == 10.0


def test_default_rendering_mode_is_left_to_isaac_lab() -> None:
    assert not isaac_ultra_rendering_enabled(environ={})
    assert isaac_rendering_mode(environ={}) is None


def test_telemetry_is_disabled_by_default_and_can_be_enabled() -> None:
    assert not isaac_telemetry_enabled(environ={})
    assert "--/privacy/usage=false" in isaac_kit_args(environ={}).split()

    environ = {"HANDSON_ISAACSIM_TELEMETRY": "1"}
    assert isaac_telemetry_enabled(environ=environ)
    assert not any("/telemetry/" in argument for argument in isaac_kit_args(environ=environ).split())


def test_existing_kit_arguments_and_settings_are_preserved() -> None:
    existing = (
        "--/renderer/multiGpu/enabled=false --/app/window/hideUi=false "
        "--/rtx/post/aa/op=4"
    )

    arguments = isaac_kit_args(existing, environ={}).split()
    assert arguments[0:3] == [
        "--/renderer/multiGpu/enabled=false",
        "--/app/window/hideUi=false",
        "--/rtx/post/aa/op=4",
    ]
    assert "--/persistent/app/viewport/noPadding=true" in arguments
    assert "--/rtx/post/aa/op=3" not in arguments
