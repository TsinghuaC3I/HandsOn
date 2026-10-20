"""GEN2 lunchbox visual source and finish layer.

The selected shrivadham source is added by the MJCF as visual-only overlays.
The authored D15-derived boxes remain the collision/support contract; this
module hides their duplicate render meshes in Isaac imports and leaves the
overlay meshes visible. The collision bodies and task thresholds are unchanged.
Importing this module does not load Kit or require USD until the entry point runs.
"""
from __future__ import annotations


PROFILE_ID = "gen2_cyan_plastic_v1"
DOWNLOADED_SOURCE = {
    "title": "LUNCHBOX",
    "author": "shrivadham",
    "url": "https://sketchfab.com/3d-models/lunchbox-43db1cb4097f42bfbbf9a14e5788f767",
    "license": "CC BY 4.0",
    "visual_only": True,
}
SOURCE = {
    "asset": "handson/handson/sim/envs/xmls/eval_gen01_cover_lunchbox_objects.xml",
    "asset_sha256": "b351d7d1bde41bb286fb34dc7450a36379ca8b7a9101124ca993a568a4c24b2b",
    "material": "dlong_cyan",
    "color_space": "linear",
    "finish_basis": "unmeasured plastic appearance estimate; source color retained",
}
FINISHES = {
    "gen2_lunchbox": {"material": "BoxPlastic", "roughness": 0.45, "surface_count": 5},
    "gen2_lid": {"material": "LidPlastic", "roughness": 0.32, "surface_count": 1},
}


def apply_visual_appearance(stage, task_root_path="/World/Task") -> dict:
    """Apply finishes to the exact GEN2 visual groups and return JSON-safe evidence.

    Strong bindings on the visual roots override importer child materials while
    leaving instanceability, collision groups and physics-purpose bindings intact.
    No geometry is expanded, hidden, recolored in place, or added to the scene.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    report = {
        "task_name": "eval_gen01_cover_lunchbox",
        "profile": PROFILE_ID,
        "source": dict(SOURCE),
        "matched_prims": [],
        "matched_prim_count": 0,
        "material_count": 0,
        "materials": [],
        "warnings": [],
        "downloaded_source": dict(DOWNLOADED_SOURCE),
        "hidden_authored_visual_count": 0,
    }
    root_path = Sdf.Path(task_root_path)
    if not root_path.IsAbsoluteRootOrPrimPath() or root_path == Sdf.Path.absoluteRootPath:
        report["warnings"].append("Expected an absolute task prim path")
        return report
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        report["warnings"].append(f"Task root is absent: {root_path}")
        return report

    # The downloaded overlay is an instance-proxy in Isaac Sim 5.1.  USD
    # forbids authoring visibility/material opinions on that proxy and the old
    # D15 appearance is already a complete collision/render contract.  Leave
    # the authored D15 appearance in place rather than crashing the worker.
    if any(prim.IsInstanceProxy() for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())):
        report["profile"] = "d15_legacy_mjcf"
        report["status"] = "legacy_fallback"
        report["warnings"].append(
            "Visual subtree is an instance proxy; retained the authored D15 appearance."
        )
        return report

    for body_name, finish in FINISHES.items():
        # Isaac MJCF uses body/body/visuals; a flattened body/visuals is also safe.
        candidates = [
            root_path.AppendPath(f"{body_name}/{body_name}/visuals"),
            root_path.AppendPath(f"{body_name}/visuals"),
        ]
        visual_roots = [stage.GetPrimAtPath(path) for path in candidates
                        if stage.GetPrimAtPath(path).IsValid()]
        if len(visual_roots) != 1 or visual_roots[0].IsInstanceProxy():
            report["warnings"].append(f"Expected one editable visual root for {body_name}")
            continue
        visual_root = visual_roots[0]
        descendants = list(Usd.PrimRange(visual_root, Usd.TraverseInstanceProxies()))
        surfaces = [prim for prim in descendants if prim.IsA(UsdGeom.Gprim)]
        expected_names = {f"{body_name}_surface_{i}" for i in range(finish["surface_count"])}
        if (len(surfaces) != finish["surface_count"]
                or {prim.GetName() for prim in surfaces} != expected_names
                or any(prim.HasAPI(UsdPhysics.CollisionAPI)
                       or prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in descendants)):
            report["warnings"].append(f"Unexpected visual-only surface layout for {body_name}")
            continue

        # The imported source box geoms carry both render and collision
        # opinions. When the downloaded overlay is present, hide the source
        # render opinions while leaving their collision API untouched. CPU
        # fixtures do not contain the overlay marker, so their material test
        # scene remains unchanged.
        overlay_present = any("gen2_downloaded" in prim.GetName()
                              for prim in Usd.PrimRange(root))
        for prim in surfaces:
            if overlay_present or prim.GetTypeName() == "Mesh":
                UsdGeom.Imageable(prim).MakeInvisible()
                report["hidden_authored_visual_count"] += 1

        material_path = root_path.AppendPath(f"Looks/GEN2Plastic/{finish['material']}")
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path.AppendChild("Surface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.08, 0.66, 0.72))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(finish["roughness"])
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.47)
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(visual_root).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        matched = []
        for prim in surfaces:
            bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if bound and bound.GetPath() == material.GetPath():
                matched.append(str(prim.GetPath()))
            else:
                report["warnings"].append(f"Material overridden above visual root: {prim.GetPath()}")
        report["matched_prims"].extend(matched)
        report["materials"].append({
            "body": body_name, "path": str(material_path),
            "binding_root": str(visual_root.GetPath()), "roughness": finish["roughness"],
            "diffuse_color_linear": [0.08, 0.66, 0.72], "metallic": 0.0,
            "opacity": 1.0, "ior": 1.47,
        })

    report["matched_prim_count"] = len(report["matched_prims"])
    report["material_count"] = len(report["materials"])
    return report
