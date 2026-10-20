"""Restore the downloaded cardboard material on the four existing visual meshes.

Only render materials and visual-root bindings change. Source artwork, UVs,
normals, vertices, transforms, collider data and physics bindings are untouched.
The shared cold-load layer owns the source/real_v12 switch and robot appearance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROFILE_ID = "gen5_source_cardboard_pbr_v1"
ASSET_DIR = Path(__file__).resolve().parents[2] / "sim/envs/xmls/eval_gen07_stack_boxes"
SOURCE_DIR = ASSET_DIR / "source/cardboard_boxes"
BODY_NAMES = ("gen5_package_base", "gen5_package_small_1", "gen5_package_small_2", "gen5_package_small_3")
SOURCE_HASHES = {
    "scene.gltf": "6e08cf4049b3f04851330385de6712087bd186048479c31a4133759c9c9effbf",
    "textures/CardboardBoxes_baseColor.png": "73e2c892beb1c48cba8614ac9cd4645f3a66048dcfc40f503a67bd151fc81c56",
    "textures/CardboardBoxes_metallicRoughness.png": "214e1ddac242954547f82773e09d972ba5c11b6aefeb86ec7e27d46eb25ad846",
    "textures/CardboardBoxes_normal.png": "29caf713490c25eab6d2ca4b1921e19409ac271ca0947cf72e8f382e4224caeb",
}


def apply_visual_appearance(stage, task_root_path="/World/Task") -> dict:
    """Bind one source-derived UsdPreviewSurface to the four visual roots."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    report = {"task_name": "eval_gen07_stack_boxes", "profile": PROFILE_ID,
        "status": "not_applied", "material_count": 0, "matched_prim_count": 0,
        "matched_prims": [], "warnings": [], "source_hashes": dict(SOURCE_HASHES),
        "source_material": "CardboardBoxes", "uv_changed": False,
        "geometry_changed": False, "physics_changed": False}
    for relative, expected in SOURCE_HASHES.items():
        path = SOURCE_DIR / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            report["warnings"].append("Missing or changed source dependency: " + relative)
    task_path = Sdf.Path(task_root_path)
    if not task_path.IsAbsoluteRootOrPrimPath() or task_path == Sdf.Path.absoluteRootPath:
        report["warnings"].append("Expected an absolute task prim path")
        return report
    targets = []
    for name in BODY_NAMES:
        candidates = [stage.GetPrimAtPath(task_path.AppendPath(s))
            for s in (f"{name}/{name}/visuals", f"{name}/visuals")]
        roots = [p for p in candidates if p.IsValid()]
        if len(roots) != 1 or roots[0].IsInstanceProxy():
            report["warnings"].append("Expected one editable visual root: " + name)
            continue
        descendants = list(Usd.PrimRange(roots[0], Usd.TraverseInstanceProxies()))
        meshes = [p for p in descendants if p.IsA(UsdGeom.Mesh)]
        if len(meshes) != 1 or any(p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI) for p in descendants):
            report["warnings"].append("Unexpected visual-only mesh layout: " + name)
            continue
        uv = UsdGeom.PrimvarsAPI(meshes[0]).GetPrimvar("st")
        if not uv or uv.Get() is None or len(uv.Get()) == 0:
            report["warnings"].append("Source UVs are absent: " + name)
            continue
        targets.append((roots[0], meshes[0]))
    # Do not leave a partially replaced product if imported topology differs.
    if report["warnings"] or len(targets) != 4:
        return report

    gltf = json.loads((SOURCE_DIR / "scene.gltf").read_text())
    source = next(m for m in gltf["materials"] if m["name"] == "CardboardBoxes")
    pbr = source["pbrMetallicRoughness"]
    color = pbr.get("baseColorFactor", [1., 1., 1., 1.])
    metallic, roughness = pbr.get("metallicFactor", 1.), pbr.get("roughnessFactor", 1.)
    normal_scale = source["normalTexture"].get("scale", 1.)
    material_path = task_path.AppendPath("Looks/GEN5SourceCardboard")
    material = UsdShade.Material.Define(stage, material_path)
    surface = UsdShade.Shader.Define(stage, material_path.AppendChild("Surface"))
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(color[3]))
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    reader = UsdShade.Shader.Define(stage, material_path.AppendChild("SourceUV"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    channels = []

    def texture(name, entry, color_space, scale, bias):
        assert entry.get("texCoord", 0) == 0 and not entry.get("extensions"), "Unexpected source UV transform"
        spec = gltf["textures"][entry["index"]]
        uri = gltf["images"][spec["source"]]["uri"]
        assert uri in SOURCE_HASHES, "Texture outside verified source set"
        sampler = gltf.get("samplers", [{}])[spec.get("sampler", 0)]
        node = UsdShade.Shader.Define(stage, material_path.AppendChild(name))
        node.CreateIdAttr("UsdUVTexture")
        node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(SOURCE_DIR / uri)))
        node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        node.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        for axis in ("S", "T"):
            wrap = {10497: "repeat", 33071: "clamp", 33648: "mirror"}[sampler.get("wrap" + axis, 10497)]
            node.CreateInput("wrap" + axis, Sdf.ValueTypeNames.Token).Set(wrap)
        for output, kind in (("rgb", Sdf.ValueTypeNames.Float3), ("g", Sdf.ValueTypeNames.Float), ("b", Sdf.ValueTypeNames.Float)):
            node.CreateOutput(output, kind)
        channels.append({"node": str(node.GetPath()), "source": uri, "sha256": SOURCE_HASHES[uri],
            "color_space": color_space, "scale": list(scale), "bias": list(bias),
            "wrap_s": node.GetInput("wrapS").Get(), "wrap_t": node.GetInput("wrapT").Get()})
        return node

    albedo = texture("BaseColor", pbr["baseColorTexture"], "sRGB", color, [0., 0., 0., 0.])
    mr = texture("MetallicRoughness", pbr["metallicRoughnessTexture"], "raw", [1., roughness, metallic, 1.], [0., 0., 0., 0.])
    # Trimesh's glTF->OBJ conversion retains U and replaces V with 1-V.
    # Keep the established st/image sampling (and printed artwork) intact;
    # negate the tangent normal's green component for the reversed bitangent.
    normal = texture("Normal", source["normalTexture"], "raw", [2*normal_scale, -2*normal_scale, 2., 1.], [-normal_scale, normal_scale, -1., 0.])
    for key, kind, node, output in (
        ("diffuseColor", Sdf.ValueTypeNames.Color3f, albedo, "rgb"),
        ("roughness", Sdf.ValueTypeNames.Float, mr, "g"),
        ("metallic", Sdf.ValueTypeNames.Float, mr, "b"),
        ("normal", Sdf.ValueTypeNames.Normal3f, normal, "rgb"),
    ):
        surface.CreateInput(key, kind).ConnectToSource(node.ConnectableAPI(), output)
    for root, mesh in targets:
        UsdShade.MaterialBindingAPI.Apply(root).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        bound, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        if not bound or bound.GetPath() != material_path:
            report["warnings"].append("Source material binding is overridden: " + str(mesh.GetPath()))
        else:
            report["matched_prims"].append(str(mesh.GetPath()))
    report.update(status="applied" if not report["warnings"] else "binding_error",
        matched_prim_count=len(report["matched_prims"]), material_count=1,
        material=str(material_path), texture_channels=channels,
        source_factors={"base_color": color, "metallic": metallic, "roughness": roughness, "normal_scale": normal_scale},
        source_note="glTF G=roughness, B=metallic; metallicFactor=0 is retained. Original st=(source_u,1-source_v) stays unchanged; normal RGB is decoded with green sign compensation for the reversed UV bitangent. Runtime-generated tangents are not a claim of exact original glTF tangent interpolation.")
    return report
