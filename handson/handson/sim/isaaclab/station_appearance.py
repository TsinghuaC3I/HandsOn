"""Stationary CAD appearance only. Never author cameras or physics."""
import hashlib
import json
from pathlib import Path

ROOT = "/World/HandsOnAppearance/Station"
LEGACY = ("rokae_center_body_visual", "rokae_center_head_fixed_visual")
CENTER_RIG_SUFFIXES = {
    LEGACY[0]: "/rokae_center/rokae_center/visuals",
    LEGACY[1]: "/rokae_center/rokae_center_head/visuals",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera_state(stage):
    from pxr import UsdGeom
    result = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            result[str(prim.GetPath())] = dict(
                attributes={a.GetName(): str(a.Get()) for a in prim.GetAttributes()},
                world_matrix=str(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)),
                relationships={r.GetName(): [str(p) for p in r.GetTargets()]
                               for r in prim.GetRelationships()})
    return result


def apply(stage, assets):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    assets = Path(assets)
    metadata = json.loads((assets / "BODY_ASSET.json").read_text())
    asset = assets / "station_body.usdc"
    assert sha(asset) == metadata["usd_sha256"]
    assert metadata["appearance_only"] and not metadata["cameras_authored"]
    assert not metadata["physics_authored"]
    before = camera_state(stage)
    if stage.GetPrimAtPath(ROOT):
        stage.RemovePrim(ROOT)
    targets = {}
    for prim in stage.Traverse():
        if prim.GetName() == "CenterRig":
            for name, suffix in CENTER_RIG_SUFFIXES.items():
                path = str(prim.GetPath()) + suffix
                if stage.GetPrimAtPath(path):
                    targets.setdefault(name, set()).add(path)
        if prim.GetName() == "visuals":
            path = str(prim.GetPath())
            if "/rokae_center_head_fixed_frame/" in path:
                targets.setdefault(LEGACY[1], set()).add(path)
            elif "/rokae_center_components/" in path:
                targets.setdefault(LEGACY[0], set()).add(path)
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        matching = next((name for name in LEGACY if name in str(prim.GetPath()).split("/")), None)
        if not matching:
            continue
        owner = prim
        while owner.IsInstanceProxy():
            owner = owner.GetParent()
        # Select only the visual subtree, never a head frame with its camera.
        if matching not in str(owner.GetPath()).split("/"):
            continue
        targets.setdefault(matching, set()).add(str(owner.GetPath()))
    if not targets:
        native_center_roots = [p for p in stage.Traverse()
                               if p.GetName() in ("CenterRig", "rokae_center_components",
                                                  "rokae_center_head_fixed_frame")]
        assert not native_center_roots, "An existing center body has an unrecognized visual layout"
        return dict(profile=metadata["version"], applied=False,
                    reason="Native scene has no active central body; preserve task embodiment",
                    asset_sha256=metadata["usd_sha256"], appearance_only=True,
                    cameras_exactly_unchanged=True, physics_authored=False)
    assert set(targets) == set(LEGACY), ("Missing legacy body visuals", targets)
    hidden = []
    for name, paths in targets.items():
        roots = [p for p in sorted(paths, key=len) if not any(p.startswith(q + "/") for q in paths)]
        for path in roots:
            prim = stage.GetPrimAtPath(path)
            descendants = list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
            assert not any(p.IsA(UsdGeom.Camera) for p in descendants)
            assert not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in descendants)
            assert not any(any("Physics" in api or "Physx" in api for api in p.GetAppliedSchemas())
                           for p in descendants)
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                # Imported MJCF instances can ignore visibility in RTX sensors.
                # Only this verified render-only subtree is deactivated.
                prim.SetActive(False)
            hidden.append(path)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        prim = stage.DefinePrim(ROOT, "Xform")
        prim.GetReferences().AddReference(str(asset), "/StationBody")
        prim.SetCustomDataByKey("station_asset_sha256", metadata["usd_sha256"])
    meshes = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROOT), Usd.TraverseInstanceProxies()):
        assert not prim.IsA(UsdGeom.Camera)
        assert not any("Physics" in s or "Physx" in s for s in prim.GetAppliedSchemas())
        assert not prim.GetTypeName().endswith("Light")
        if prim.IsA(UsdGeom.Mesh):
            meshes += 1
    assert meshes >= len(metadata["instances"]), (meshes, len(metadata["instances"]))
    after = camera_state(stage)
    assert before == after, "CAD appearance insertion changed a camera"
    result = dict(profile=metadata["version"], applied=True, asset_sha256=metadata["usd_sha256"],
                  asset_manifest_sha256=sha(assets / "BODY_ASSET.json"),
                  root=ROOT, legacy_visuals_hidden=hidden, meshes=meshes,
                  parts=len(metadata["parts"]), instances=len(metadata["instances"]),
                  bounds_m=metadata["bounds_m"], appearance_only=True,
                  camera_state_before=before, camera_state_after=after,
                  cameras_exactly_unchanged=True, physics_authored=False,
                  legacy_visuals_deactivated=True,
                  source_assembly_sha256=metadata["source_assembly_sha256"])
    verify(stage, result)
    return result


def verify(stage, receipt):
    from pxr import UsdGeom
    if receipt.get("applied") is False:
        assert not stage.GetPrimAtPath(ROOT)
        return dict(applied=False, native_body_absence_preserved=True)
    assert stage.GetPrimAtPath(ROOT).GetCustomDataByKey(
        "station_asset_sha256") == receipt["asset_sha256"]
    for path in receipt["legacy_visuals_hidden"]:
        prim = stage.GetPrimAtPath(path)
        assert prim and not prim.IsActive()
    return dict(asset_sha256=receipt["asset_sha256"], legacy_visuals_hidden=True,
                stationary_geometry=True)
