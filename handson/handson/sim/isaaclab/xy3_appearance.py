"""XY3-only object shading and duplicate render-surface cleanup.

The source camera contains coincident material shells (often opposite winding).
Only completely coincident, non-colliding meshes on the same rigid body can be
hidden. Original mesh data, all physical bindings, and the CASIA decal remain
unchanged. Apply in a fresh process before physics initialization.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import os
from pathlib import Path

import numpy as np

APPEARANCE_ENV = "HANDSON_XY3_OBJECT_APPEARANCE"
APPEARANCE_PROFILE = "source_pbr_v1"
TASK_NAME = "eval_comb04_place_fruit_and_photograph_logo"
FRUIT_ROUGHNESS = {"apple": .36, "orange": .72, "pear": .60, "011_banana": .65}
DOWNLOADED_BUCKET_ENV = "HANDSON_XY3_DOWNLOADED_BUCKET"
DOWNLOADED_BUCKET_USD = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "envs", "usd",
                 "xy3_plastic_bucket", "downloaded_plastic_bucket.usda")
)


def physics_material_bindings(stage):
    """Include effective physics-purpose bindings, not just physics attributes."""
    from pxr import Usd, UsdPhysics, UsdShade
    result = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
            material, binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
            result[str(prim.GetPath())] = [str(material.GetPath()) if material else "",
                                          str(binding.GetPath()) if binding else ""]
    return result


def _source_surface(prim):
    from pxr import Usd, UsdShade
    recorded = prim.GetCustomDataByKey("handson:xy3SourceColor")
    if recorded is not None:
        return (tuple(float(v) for v in recorded),
                prim.GetCustomDataByKey("handson:xy3SourceAlbedo") or None,
                prim.GetCustomDataByKey("handson:xy3SourceMaterial") or "")
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    color, texture = (.12, .12, .12), None
    if material:
        for child in Usd.PrimRange(material.GetPrim(), Usd.TraverseInstanceProxies()):
            if not child.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(child)
            for name in ("diffuse_color_constant", "diffuseColor"):
                value = shader.GetInput(name).Get()
                if value is not None:
                    color = tuple(float(v) for v in value)
            value = shader.GetInput("diffuse_texture").Get()
            if value:
                texture = value.resolvedPath or value.path
            diffuse = shader.GetInput("diffuseColor")
            source = diffuse.GetConnectedSource() if diffuse else None
            if source:
                value = UsdShade.Shader(source[0].GetPrim()).GetInput("file").Get()
                if value:
                    texture = value.resolvedPath or value.path
    return color, texture, str(material.GetPath()) if material else ""


def _camera_finish(color):
    r, g, b = color
    if b > r * 1.15 and g > r * 1.15:
        return "LensGlass", (.012, .035, .042), .10, 0., .8
    if r > g * 2 and r > b * 1.5:
        return "RedAccent", (.28, .009, .018), .42, 0., 0.
    if r > b * 2 and g > b * 2:
        return "BrassAccent", (.48, .30, .045), .28, .7, 0.
    if max(color) > .7:
        return "SatinMetalBody", (.48, .49, .50), .36, .72, 0.
    if max(color) > .4:
        return "BrushedControls", (.24, .25, .27), .32, .72, 0.
    if max(color) > .2:
        return "RubberGrip", (.018, .021, .024), .68, 0., 0.
    return "BlackHousing", (.009, .011, .014), .56, 0., 0.


def _material(stage, name, color, roughness, metallic=0., clearcoat=0., texture=None):
    from pxr import Gf, Sdf, UsdShade
    path = "/World/Looks/XY3Appearance/" + name
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, path + "/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    for key, value in (("roughness", roughness), ("metallic", metallic),
                       ("clearcoat", clearcoat), ("ior", 1.5), ("opacity", 1.)):
        surface.CreateInput(key, Sdf.ValueTypeNames.Float).Set(value)
    if texture:
        uv = UsdShade.Shader.Define(stage, path + "/UV")
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(stage, path + "/Albedo")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surface.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    return material


def _editable_visual(stage, path):
    from pxr import UsdPhysics
    prim = stage.GetPrimAtPath(path)
    while prim.IsInstanceProxy():
        root = prim
        while root.IsInstanceProxy():
            root = root.GetParent()
        if "visuals" not in str(root.GetPath()).split("/") or root.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Refusing to deinstance a physical subtree: " + path)
        root.SetInstanceable(False)
        prim = stage.GetPrimAtPath(path)
    return prim


def _strong_visual_bindings(prim):
    """Find render-only ancestor bindings that would mask a leaf material.

    Never unbind a physical prim or a group containing physical descendants:
    their all-purpose material can be the fallback physics-purpose material.
    """
    from pxr import Usd, UsdPhysics, UsdShade
    bindings = []
    parent = prim.GetParent()
    while parent and not parent.IsPseudoRoot():
        relation = UsdShade.MaterialBindingAPI(parent).GetDirectBindingRel()
        if (relation and relation.GetTargets()
                and UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(relation)
                == UsdShade.Tokens.strongerThanDescendants):
            if "visuals" not in str(parent.GetPath()).split("/") or any(
                p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI)
                for p in Usd.PrimRange(parent, Usd.TraverseInstanceProxies())
            ):
                return None
            bindings.append(str(parent.GetPath()))
        parent = parent.GetParent()
    return bindings


def _mesh_key(prim):
    """Canonical polygon coordinates in the unchanged owner frame, at 1 nm."""
    from pxr import UsdGeom, UsdPhysics
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        return None
    owner = prim.GetParent()
    while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
        owner = owner.GetParent()
    if not owner:
        return None
    cache = UsdGeom.XformCache()
    matrix = np.asarray(cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(owner).GetInverse())
    points = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    if (counts.ndim != 1 or indices.ndim != 1 or not len(counts)
            or np.any(counts < 3) or counts.sum() != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(points))):
        return None
    faces = np.split(indices, np.cumsum(counts)[:-1])
    polygons, volume = [], 0.
    center = points.mean(0)
    for face in faces:
        xyz = points[face]
        polygons.append(tuple(sorted(tuple(v) for v in np.round(xyz, 9))))
        for i in range(1, len(face) - 1):
            a, b, c = xyz[[0, i, i + 1]] - center
            volume += float(a @ np.cross(b, c)) / 6
    if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        volume = -volume
    digest = hashlib.sha256(repr(sorted(polygons)).encode()).hexdigest()
    return str(owner.GetPath()), digest, volume, len(faces)


def apply_xy3_object_appearance(stage, *, body_paths, profile=APPEARANCE_PROFILE):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    from .blender_visual_layer import physics_signature
    report = {"profile": profile, "visual_only": True, "bound_paths": [],
              "duplicate_surfaces": [], "source_materials": {}, "warnings": [],
              "render_binding_purpose": "allPurpose", "cleared_visual_binding_paths": [],
              "downloaded_bucket_source": None,
              # XY3 intentionally composes the A01/YCB scene. This is the
              # physical contract used for A/B training comparisons.
              "physical_contract_source": (
                  "arena_arm_hand_ycb_tasks_rokae_wuji1.xml -> bucket.xml + YCB"
              )}
    if profile in {"", "source"}:
        return report
    if profile != APPEARANCE_PROFILE:
        report["warnings"].append("Unknown XY3 appearance profile; kept source visuals")
        return report
    before, bindings = physics_signature(stage), physics_material_bindings(stage)
    candidates, groups = {}, defaultdict(list)
    for role, path in body_paths.items():
        root = stage.GetPrimAtPath(path)
        if not root:
            report["warnings"].append("Missing visual role: " + role)
            continue
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if (not prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI)
                    or "/visuals/" not in path or "logo" in path.lower()
                    or UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible):
                continue
            color, texture, material = _source_surface(prim)
            candidates[path] = (role, color, texture)
            report["source_materials"][path] = material
            if role in {"camera", "bucket"}:
                key = _mesh_key(prim)
                if key:
                    groups[key[:2]].append((path, key[2], key[3], sum(color)))
    hidden = set()
    for (_, digest), rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        # Outward closed shell wins; coincident same-winding/open surfaces use
        # their darker source finish deterministically instead of depth fighting.
        # Coincident faces have equal absolute volume up to floating-point
        # roundoff. Compare its sign rather than allowing roundoff to choose a
        # different material for same-winding duplicates.
        tolerance = max(1e-18, max(abs(row[1]) for row in rows) * 1e-8)
        chosen = sorted(rows, key=lambda row: (
            -(1 if row[1] > tolerance else -1 if row[1] < -tolerance else 0),
            row[3], row[0],
        ))[0]
        # Two opposite single-sided render shells previously covered both
        # viewing directions; preserve that coverage with one render surface.
        UsdGeom.Mesh(_editable_visual(stage, chosen[0])).CreateDoubleSidedAttr(True)
        for path, _, _, _ in rows:
            if path == chosen[0]:
                continue
            UsdGeom.Imageable(_editable_visual(stage, path)).MakeInvisible()
            hidden.add(path)
        report["duplicate_surfaces"].append({
            "kept": chosen[0], "hidden": sorted(row[0] for row in rows if row[0] != chosen[0]),
            "canonical_faces_sha256": digest, "face_count": chosen[2],
            "signed_volumes_m3": {row[0]: row[1] for row in rows},
            "selection": "outward winding, then darker source finish; retained mesh is double-sided",
        })
    for path, (role, color, texture) in candidates.items():
        if path in hidden:
            continue
        if role in FRUIT_ROUGHNESS:
            if not texture or not Path(texture).is_file():
                report["warnings"].append("Kept fruit source material: unresolved albedo at " + path)
                continue
            spec = ("Fruit_" + role, (1., 1., 1.), FRUIT_ROUGHNESS[role], 0., 0.)
        elif role == "bucket":
            spec = ("RedMoldedBucket", (.55, .012, .018), .48, 0., 0.) if color[0] > .2 else (
                "BucketHandleBlack", (.008, .010, .012), .62, 0., 0.)
            texture = None
        else:
            spec, texture = _camera_finish(color), None
        prim = _editable_visual(stage, path)
        visual_bindings = _strong_visual_bindings(prim)
        if visual_bindings is None:
            report["warnings"].append("Kept source shading beneath physical strong binding: " + path)
            continue
        for parent_path in visual_bindings:
            UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(parent_path)).UnbindDirectBinding()
            report["cleared_visual_binding_paths"].append(parent_path)
        material = _material(stage, *spec, texture=texture)
        # RTX 5.1 uses the all-purpose binding. These leaves are render-only;
        # no body/collider bindings, source materials or physics purpose change.
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        actual, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not actual or actual.GetPath() != material.GetPath():
            raise RuntimeError("XY3 render material did not resolve at " + path)
        prim.SetCustomDataByKey("handson:xy3SourceColor", Gf.Vec3d(*color))
        prim.SetCustomDataByKey("handson:xy3SourceAlbedo", candidates[path][2] or "")
        prim.SetCustomDataByKey("handson:xy3SourceMaterial", report["source_materials"][path])
        report["bound_paths"].append(path)
    # The downloaded bucket is deliberately an appearance overlay.  Keep the
    # original bucket body, collision boxes, handle joint and physical
    # material authoritative so reset and task success remain comparable with
    # the existing XY3/YCB baseline.
    if (os.environ.get(DOWNLOADED_BUCKET_ENV, "0").strip() == "1"
            and "bucket" in body_paths and Path(DOWNLOADED_BUCKET_USD).is_file()):
        bucket_root = stage.GetPrimAtPath(body_paths["bucket"])
        # MJCF conversion puts the moving bucket and articulated handle under
        # a static container. Keep that container as the source-visual scope,
        # but attach the replacement mesh to the actual bucket rigid body.
        bucket_body = bucket_root if bucket_root.HasAPI(UsdPhysics.RigidBodyAPI) else next(
            (prim for prim in Usd.PrimRange(bucket_root)
             if prim.GetName() == "bucket" and prim.HasAPI(UsdPhysics.RigidBodyAPI)),
            None,
        )
        if bucket_body is None:
            raise ValueError("XY3 downloaded bucket requires its source rigid body")
        overlay_path = str(bucket_body.GetPath()) + "/DownloadedPlasticBucket"
        existing_overlay = stage.GetPrimAtPath(overlay_path)
        had_reference = bool(
            existing_overlay and existing_overlay.HasAuthoredReferences()
        )
        overlay = UsdGeom.Xform.Define(stage, overlay_path)
        # bucket.xml centers the shell at X=-.01, Z=-.005; its support plane
        # is local Y=-.08. The downloaded asset starts 2 mm above Z=0.
        translate = next((op for op in overlay.GetOrderedXformOps()
                          if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        if translate is None:
            translate = overlay.AddTranslateOp()
        translate.Set(Gf.Vec3d(-.01, -.08, -.005))
        # The downloaded bucket is authored upright (+Z up), while the source
        # bucket body is authored on its side with local +Y as the vertical
        # axis.  Rotate only the render overlay so its bottom lands on the
        # table and leave the source rigid body/collider orientation intact.
        rotate_x = next(
            (
                op for op in overlay.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeRotateX
            ),
            None,
        )
        if rotate_x is None:
            rotate_x = overlay.AddRotateXOp()
        rotate_x.Set(-90.0)
        overlay.SetXformOpOrder([translate, rotate_x])
        if had_reference:
            report["downloaded_bucket_source"] = {
                "asset": DOWNLOADED_BUCKET_USD, "visual_only": True,
                "already_present": True, "local_rotation_deg": -90.0,
            }
        else:
            hidden = []
            for prim in Usd.PrimRange(bucket_root, Usd.TraverseInstanceProxies()):
                path = str(prim.GetPath())
                if (prim.IsA(UsdGeom.Mesh) and "/visuals/" in path
                        and "logo" not in path.lower()
                        and not prim.HasAPI(UsdPhysics.CollisionAPI)
                        and UsdGeom.Imageable(prim).ComputeVisibility()
                        != UsdGeom.Tokens.invisible):
                    UsdGeom.Imageable(_editable_visual(stage, path)).MakeInvisible()
                    hidden.append(path)
            overlay.GetPrim().GetReferences().AddReference(DOWNLOADED_BUCKET_USD,
                                                            "/PlasticBucket")
            report["downloaded_bucket_source"] = {
                "asset": DOWNLOADED_BUCKET_USD,
                "title": "Plastic Bucket", "author": "MaX3Dd",
                "source": "https://sketchfab.com/3d-models/plastic-bucket-006d0f237b4141ac93960fa51d6f0634",
                "license": "CC BY 4.0", "visual_only": True,
                "collision_body_unchanged": True,
                "local_rotation_deg": -90.0,
                "physical_contract_source": report["physical_contract_source"],
                "hidden_source_visuals": hidden,
            }
    after = physics_signature(stage)
    if before != after or physics_material_bindings(stage) != bindings:
        raise RuntimeError("XY3 appearance changed physical configuration or material purpose")
    report.update(physics_before=before, physics_after=after,
                  physics_material_bindings_preserved=True)
    return report


def apply_configured_xy3_appearance(environment):
    selected = os.environ.get(APPEARANCE_ENV, "source").strip().lower()
    if environment.task_name != TASK_NAME or selected in {"", "source"}:
        return None
    paths = {}
    for role in ("camera", "bucket", *FRUIT_ROUGHNESS):
        prim = environment._find_rigid_body_prim("link_19" if role == "camera" else role)
        if prim:
            paths[role] = str(prim.GetParent().GetPath()) if role in {"camera", "bucket"} else str(prim.GetPath())
    report = apply_xy3_object_appearance(environment._stage, body_paths=paths, profile=selected)
    environment.xy3_appearance_report = report
    return report
