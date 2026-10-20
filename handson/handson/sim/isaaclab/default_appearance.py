"""The shared HandsOn rendering contract for previews, environments and replay.

Robot references, cables and the room are installed before the first physics
reset. Subsequent renders update measured cable poses only. Missing assets or
incompatible embodiments fail explicitly; there is no lower-quality fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .render_assets import validate_assets

PROFILE = "handson"
_robots: dict[str, dict] = {}
_previews: dict[str, object] = {}
FILM_ISO = {"head_camera": 260.0, "wrist_left_camera": 690.0, "wrist_right_camera": 590.0}


def _key(stage) -> str:
    return stage.GetRootLayer().identifier


def _check_removed_switches() -> None:
    for name in ("HANDSON_ISAACSIM_ROBOT_APPEARANCE_PROFILE", "HANDSON_ISAACSIM_TASK_VISUAL_PROFILE"):
        if os.environ.get(name, PROFILE).strip().lower() != PROFILE:
            raise ValueError(f"{name} no longer selects an alternative appearance. Remove it to use HandsOn rendering.")
    if os.environ.get("HANDSON_ISAACSIM_BLENDER_VISUAL_BUNDLE"):
        raise ValueError("Custom robot bundle overrides are not supported by the default rendering contract")


def physics_material_signature(stage) -> str:
    from pxr import Usd, UsdPhysics, UsdShade
    rows = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
            values = {} if not mat else {a.GetName(): str(a.Get()) for a in mat.GetPrim().GetAttributes()
                                        if a.GetName().startswith(("physics:", "physx"))}
            rows.append((str(prim.GetPath()), values))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def verify_robot(stage, report: dict) -> None:
    from pxr import Sdf
    root = validate_assets() / "robot"
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {item["link_name"]: Sdf.Reference(str(root / item["asset"]), item["prim"])
                for item in manifest["links"]}
    for path in report["attached"]:
        prim = stage.GetPrimAtPath(path)
        references = prim.GetMetadata("references") if prim else None
        link = prim.GetParent().GetParent().GetName() if prim else ""
        if references is None or references.GetAddedOrExplicitItems() != [expected.get(link)]:
            raise RuntimeError(f"Default robot visual reference changed: {path}")
    if len(report["attached"]) != len(report["link_names"]):
        raise RuntimeError("Incomplete default robot visual bundle")


def apply_robot(stage, roots) -> dict:
    _check_removed_switches()
    key = _key(stage)
    if key in _robots:
        verify_robot(stage, _robots[key])
        return _robots[key]
    from .blender_visual_layer import apply_blender_visual_bundle
    root = validate_assets()
    before = physics_material_signature(stage)
    report = apply_blender_visual_bundle(stage, root / "robot", tuple(roots))
    if before != physics_material_signature(stage):
        raise RuntimeError("Robot appearance changed an effective physics material")
    _robots[key] = report
    verify_robot(stage, report)
    return report


def configure_exposure(sensor: str = "head_camera") -> None:
    import carb.settings
    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/post/histogram/enabled", False)
    settings.set_bool("/rtx/post/tonemap/autoExposure/enabled", False)
    for key, value in (("cameraShutter", 62.5), ("fNumber", 5.0), ("filmIso", FILM_ISO[sensor])):
        settings.set_float("/rtx/post/tonemap/" + key, value)


def hide_helpers(stage) -> list[str]:
    from pxr import Usd, UsdGeom, UsdPhysics
    hidden = []
    for prim in stage.Traverse():
        if "fov_helper" not in prim.GetName() and "fov_ray" not in prim.GetName():
            continue
        children = list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
        if any(p.IsA(UsdGeom.Camera) or p.HasAPI(UsdPhysics.CollisionAPI)
               or p.HasAPI(UsdPhysics.RigidBodyAPI) or p.IsA(UsdPhysics.Joint) for p in children):
            continue
        if UsdGeom.Imageable(prim):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(str(prim.GetPath()))
    return hidden


def install_environment(environment) -> dict:
    """Run once after native scene construction, before its first physics reset."""
    from . import room_appearance, station_appearance
    from .blender_visual_layer import physics_signature
    from .wrist_cable_visuals import apply_configured_wrist_cables

    if getattr(environment, "default_appearance_report", None) is not None:
        verify_environment(environment)
        return environment.default_appearance_report
    if getattr(environment, "hand_type", "wuji1") != "wuji1":
        raise ValueError("The packaged default rendering assets require Wuji1 hands")
    root = validate_assets()
    stage = environment._stage
    before = physics_signature(stage), physics_material_signature(stage)
    # The paper adapter restores the collection-time visual mask before
    # replacing robot surfaces. Retain the native defer/control flags.
    restored_visuals = 0
    if getattr(environment, "_deferred_robot_body_visibility", None):
        restored_visuals = environment._set_deferred_robot_body_hidden(False)
    robot = apply_robot(stage, environment._reference_robot_visual_roots)
    environment.blender_visual_bundle_report = robot
    if all(f"link7_{side}" in robot["link_names"] for side in ("left", "right")):
        cables = apply_configured_wrist_cables(environment, root / "cables.json")
    else:
        cables = {"applied": False, "reason": "Native detached hands have no forearm cable attachment"}
    profile = {"mode": "randomized", "seed": int(getattr(environment, "_seed", 0)),
               "task": str(getattr(environment, "task_name", "")), "spp": 32}
    room = room_appearance.apply(stage, getattr(environment, "_cameras", {}), profile, root / "materials",
                                 table_required=environment.task_name != "t17_open_latched_door")
    body = station_appearance.apply(stage, root / "station")
    hidden = hide_helpers(stage)
    configure_exposure()
    after = physics_signature(stage), physics_material_signature(stage)
    if before != after:
        raise RuntimeError("Default appearance changed the physical scene or effective contact material")
    report = {"profile": PROFILE, "robot": robot, "room": room, "station": body,
              "cables": {k: v for k, v in cables.items() if k not in {"stage", "indices"}}, "hidden_helpers": hidden, "physics_before": before, "physics_after": after,
              "applied_before_physics": True, "restored_native_visuals": restored_visuals,
              "assets_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}
    environment._accurate_camera_visibility = True
    environment._path_tracing_capture = True
    environment.default_appearance_report = report
    environment._wrist_obstruction_visibility = None
    environment._deferred_robot_body_visibility = None
    metadata = getattr(environment, "recording_camera_profile", None)
    if isinstance(metadata, dict):
        metadata["appearance"] = {"profile": PROFILE, "assets_manifest_sha256": report["assets_manifest_sha256"],
                                  "robot_links": robot["links"], "render_mode": "PathTracing", "samples_per_pixel": 32}
    print(f"[HandsOn] Default rendering ready: {robot['links']} robot links, PBR room, station and wrist cables", flush=True)
    return report


def refresh_environment(environment) -> None:
    """Refresh episode surfaces without replacing any robot reference."""
    from . import room_appearance
    report = getattr(environment, "default_appearance_report", None)
    if report is None:
        return
    from .blender_visual_layer import physics_signature
    before = physics_signature(environment._stage), physics_material_signature(environment._stage)
    root = validate_assets()
    profile = dict(report["room"]["profile"], seed=int(getattr(environment, "_seed", 0)))
    # Resolve from the seed rather than carrying the previous episode's resolved values.
    profile = {k: profile[k] for k in ("mode", "seed", "task", "spp")}
    report["room"] = room_appearance.apply(environment._stage, getattr(environment, "_cameras", {}), profile, root / "materials",
                                 table_required=environment.task_name != "t17_open_latched_door")
    configure_exposure()
    after = physics_signature(environment._stage), physics_material_signature(environment._stage)
    if before != after:
        raise RuntimeError("Episode appearance refresh changed the physical scene")
    report["last_refresh_physics_unchanged"] = True
    verify_robot(environment._stage, report["robot"])


def verify_environment(environment) -> None:
    from . import room_appearance, station_appearance
    report = environment.default_appearance_report
    verify_robot(environment._stage, report["robot"])
    room_appearance.verify(environment._stage, report["room"])
    station_appearance.verify(environment._stage, report["station"])
    cameras = getattr(environment, "_cameras", {})
    if not cameras:
        cameras = getattr(getattr(environment, "_scene", None), "sensors", {})
    if "head_camera" in cameras:
        from .calibrated_cameras import verify
        report["camera_verification"] = verify(cameras)


def install_preview(stage, task_name: str, seed: int = 0) -> dict:
    env = SimpleNamespace(_stage=stage, task_name=task_name, _seed=seed, hand_type="wuji1",
                          _reference_robot_visual_roots=("/World/Task",), _cameras={})
    result = install_environment(env)
    _previews[_key(stage)] = env
    return result


def update_preview(stage) -> None:
    from .wrist_cable_visuals import update_wrist_cables
    env = _previews.get(_key(stage))
    if env is not None:
        update_wrist_cables(env)


def verify_preview(stage) -> dict:
    env = _previews[_key(stage)]
    verify_environment(env)
    return env.default_appearance_report
