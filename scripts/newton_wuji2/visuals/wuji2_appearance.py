"""Attach a checked, visual-only Wuji2 USD bundle to native articulated links."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


_BUNDLE = Path(__file__).resolve().parents[5] / "assets" / os.environ.get("WUJI2_VISUAL_BUNDLE", "wuji2_photo_glove_link_v14")


def _physics_signature(stage):
    from pxr import Usd, UsdPhysics

    rows = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        schemas = [s for s in prim.GetAppliedSchemas() if "Physics" in s or "Physx" in s]
        attrs = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                 if a.GetName().startswith(("physics:", "physx"))}
        rels = {r.GetName(): [str(p) for p in r.GetTargets()] for r in prim.GetRelationships()
                if r.GetName().startswith(("physics:", "physx"))
                or r.GetName() == "material:binding:physics"}
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            attrs.update({a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                          if a.GetName() in ("points", "faceVertexCounts", "faceVertexIndices",
                                             "radius", "size", "height", "axis")
                          or a.GetName().startswith("xformOp")})
        if schemas or attrs or rels:
            rows.append((str(prim.GetPath()), schemas, attrs, rels))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def _physics_material_signature(stage):
    from pxr import Usd, UsdPhysics, UsdShade

    rows = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
            materialPurpose="physics"
        )
        values = {} if not mat else {a.GetName(): str(a.Get()) for a in mat.GetPrim().GetAttributes()
                                     if a.GetName().startswith(("physics:", "physx"))}
        rows.append((str(prim.GetPath()), values))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def _verified_manifest():
    from pxr import Usd, UsdUtils

    manifest_path = _BUNDLE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("hardware") != "rokae_wuji2" or not manifest.get("visual_only"):
        raise RuntimeError("Expected Wuji2 visual-only bundle")
    if manifest.get("meters_per_unit") != 1 or manifest.get("up_axis") != "Z":
        raise RuntimeError("Unexpected Wuji2 asset units or axis")
    for item in manifest["links"]:
        asset = (_BUNDLE / item["asset"]).resolve()
        if _BUNDLE.resolve() not in asset.parents:
            raise RuntimeError(f"Asset escapes Wuji2 bundle: {asset}")
        if hashlib.sha256(asset.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Wuji2 asset hash mismatch: {asset}")
        layers, dependencies, unresolved = UsdUtils.ComputeAllDependencies(str(asset))
        if unresolved:
            raise RuntimeError(f"Unresolved Wuji2 asset dependencies: {unresolved}")
        visual = Usd.Stage.Open(str(asset))
        if visual is None or not visual.GetPrimAtPath(item["prim"]):
            raise RuntimeError(f"Invalid Wuji2 link asset: {asset}")
        for prim in visual.Traverse():
            if any("Physics" in s or "Physx" in s for s in prim.GetAppliedSchemas()):
                raise RuntimeError(f"Wuji2 visual asset contains physics: {asset}")
    for relative, expected in manifest["dependencies"].items():
        path = (_BUNDLE / relative).resolve()
        if _BUNDLE.resolve() not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Wuji2 texture hash mismatch: {path}")
    return manifest, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def apply_wuji2_hands(stage, robot_root_paths):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    # The task template is imported first with legacy placeholder robots.
    # The actual recorded Wuji2 articulations are spawned subsequently under
    # DexJoCoRobots. Never attach Wuji2 geometry to those Wuji1 placeholders.
    if tuple(robot_root_paths) == ("/World/Task",):
        return {"profile": "deferred_until_native_wuji2_spawn", "link_count": 0,
                "visual_meshes": [], "tip_plates": []}

    manifest, manifest_hash = _verified_manifest()
    expected = {item["link_name"] for item in manifest["links"]}
    links = {}
    for root_path in robot_root_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise RuntimeError(f"Missing Wuji2 robot root: {root_path}")
        for prim in Usd.PrimRange(root):
            name = prim.GetName()
            if name not in expected or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if name in links:
                raise RuntimeError(f"Duplicate Wuji2 link: {name}")
            links[name] = prim
    if set(links) != expected:
        raise RuntimeError(f"Incomplete Wuji2 robot: {sorted(expected - set(links))}")

    before = (_physics_signature(stage), _physics_material_signature(stage))
    attached, hidden, tip_markers_hidden = [], [], []
    for item in manifest["links"]:
        link = links[item["link_name"]]
        visuals_path = link.GetPath().AppendChild("visuals")
        visuals = stage.GetPrimAtPath(visuals_path)
        if not visuals.IsValid():
            visuals = UsdGeom.Xform.Define(stage, visuals_path).GetPrim()
        if visuals.IsInstance():
            visuals.SetInstanceable(False)
            visuals = stage.GetPrimAtPath(visuals_path)
        binding = UsdShade.MaterialBindingAPI(visuals)
        for purpose in (UsdShade.Tokens.allPurpose, UsdShade.Tokens.preview, UsdShade.Tokens.full):
            relation = binding.GetDirectBindingRel(purpose)
            if relation:
                relation.SetTargets([])
        for child in visuals.GetAllChildren():
            if child.GetName() != "BlenderVisuals" and child.IsA(UsdGeom.Imageable):
                for descendant in Usd.PrimRange(child, Usd.TraverseInstanceProxies()):
                    if any('Physics' in s or 'Physx' in s for s in descendant.GetAppliedSchemas()):
                        raise RuntimeError(f"Refusing to retire physical subtree: {child.GetPath()}")
                UsdGeom.Imageable(child).MakeInvisible()
                # Fabric/RTX can retain native geometry after a visibility-only
                # edit. Retire the replaced pure visual subtree before reset.
                child.SetActive(False)
                hidden.append(str(child.GetPath()))
        reference_appearance = link.GetChild("ReferenceAppearance")
        if reference_appearance.IsValid():
            UsdGeom.Imageable(reference_appearance).MakeInvisible()
            hidden.append(str(reference_appearance.GetPath()))
        path = visuals_path.AppendChild("BlenderVisuals")
        visual = UsdGeom.Xform.Define(stage, path)
        visual.GetPrim().GetReferences().SetReferences([
            Sdf.Reference(str(_BUNDLE / item["asset"]), item["prim"])
        ])
        visual.MakeMatrixXform().Set(UsdGeom.Xformable(visuals).GetLocalTransformation().GetInverse())
        visual.GetPrim().SetInstanceable(True)
        for part in Usd.PrimRange(visual.GetPrim(), Usd.TraverseInstanceProxies()):
            if part.IsA(UsdGeom.Mesh):
                mat, _ = UsdShade.MaterialBindingAPI(part).ComputeBoundMaterial()
                if not mat or not mat.GetPath().HasPrefix(path):
                    raise RuntimeError(f"Wuji2 material overridden: {part.GetPath()}")
        attached.append(str(path))
    tip_site_candidates = []
    for root_path in robot_root_paths:
        for marker in Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()):
            path = str(marker.GetPath())
            if "/sites/" not in path or "finger" not in marker.GetName():
                continue
            tip_site_candidates.append(path)
            if (marker.GetName().endswith("_tip") and not marker.IsInstanceProxy()
                    and marker.IsA(UsdGeom.Imageable)):
                UsdGeom.Imageable(marker).MakeInvisible()
                tip_markers_hidden.append(path)
    after = (_physics_signature(stage), _physics_material_signature(stage))
    if before != after:
        raise RuntimeError("Wuji2 visual bundle changed physics or contact materials")
    if tip_markers_hidden and len(tip_markers_hidden) != 10:
        raise RuntimeError(f"Expected 10 visual fingertip markers when present, found {len(tip_markers_hidden)}")
    report = {
        "profile": manifest["profile"], "link_count": len(attached),
        "robot_roots": list(robot_root_paths),
        "visual_meshes": attached, "tip_plates": [],
        "attached": attached, "hidden": hidden,
        "tip_markers_hidden": tip_markers_hidden,
        "tip_site_candidates": tip_site_candidates,
        "grip_blocks": manifest["total_grip_blocks"],
        "manifest_sha256": manifest_hash,
        "physics_before": before[0], "physics_after": after[0],
        "physics_material_before": before[1], "physics_material_after": after[1],
    }
    output = os.environ.get("DEXJOCO_WUJI2_VISUAL_REPORT")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    return report
