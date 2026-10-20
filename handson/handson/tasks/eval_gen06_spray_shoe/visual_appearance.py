"""Restore GEN7 source PBR through visual material bindings only.

The common task appearance dispatcher owns source/real_v12 selection. The
original bottle label stays in its source atlas; no geometry split, invented
label mask, wetness or successful-spray visual is introduced here.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path


ASSETS = Path(__file__).resolve().parents[2] / "sim/envs/usd/eval_gen06_spray_shoe/appearance_source"


@lru_cache(maxsize=1)
def source_materials():
    return json.loads((ASSETS / "materials.json").read_text())


def usd_uv_transform(texture_info):
    """Conjugate the glTF transform by its already-exported V flip."""
    transform = texture_info.get("extensions", {}).get("KHR_texture_transform", {})
    if int(transform.get("texCoord", texture_info.get("texCoord", 0))) != 0:
        raise ValueError("GEN7 source texture requires an unexported UV set")
    sx, sy = transform.get("scale", (1., 1.))
    ox, oy = transform.get("offset", (0., 0.))
    angle = float(transform.get("rotation", 0.))
    return {"scale": [sx, sy], "rotation_degrees": -math.degrees(angle),
            "translation": [ox - math.sin(angle) * sy, 1. - oy - math.cos(angle) * sy]}


def make_material(stage, path, specification):
    from pxr import Gf, Sdf, UsdShade
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    source = specification["material"]
    spec_gloss = source.get("extensions", {}).get("KHR_materials_pbrSpecularGlossiness")
    pbr = source.get("pbrMetallicRoughness", {})
    if spec_gloss is not None:
        color = spec_gloss.get("diffuseFactor", [1., 1., 1., 1.])
        roughness = 1. - float(spec_gloss.get("glossinessFactor", 1.))
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*spec_gloss.get("specularFactor", [1., 1., 1.])))
    else:
        color = pbr.get("baseColorFactor", [1., 1., 1., 1.])
        roughness = float(pbr.get("roughnessFactor", 1.))
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(pbr.get("metallicFactor", 1.)))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color[:3]))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(color[3]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    textures = []
    for role, texture in specification["textures"].items():
        info = texture["source_texture_info"]
        transform = usd_uv_transform(info)
        reader = UsdShade.Shader.Define(stage, path + "/" + role + "UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        uv = UsdShade.Shader.Define(stage, path + "/" + role + "Transform")
        uv.CreateIdAttr("UsdTransform2d")
        uv.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        uv.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*transform["scale"]))
        uv.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(transform["rotation_degrees"])
        uv.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*transform["translation"]))
        uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(stage, path + "/" + role)
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(ASSETS / texture["file"])))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
        wraps = {10497: "repeat", 33071: "clamp", 33648: "mirror"}
        for axis in ("S", "T"):
            tex.CreateInput("wrap" + axis, Sdf.ValueTypeNames.Token).Set(wraps[texture["sampler"].get("wrap" + axis, 10497)])
        normal = role == "normalTexture"
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if normal else "sRGB")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        if normal:
            strength = float(info.get("scale", 1.))
            # OBJ st already flips glTF V, so its generated bitangent is
            # reversed. Source TANGENT.w and triangle UV derivatives verify
            # this sign independently; invert decoded Y, not the image/UVs.
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2 * strength, -2 * strength, 2., 1.))
            tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-strength, strength, -1., 0.))
            shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        elif role in ("baseColorTexture", "diffuseTexture"):
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*color))
            shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
        else:
            raise ValueError(f"Unexamined GEN7 source texture role: {role}")
        textures.append({"role": role, "file": texture["file"], "sha256": texture["sha256"], "usd_uv_transform": transform,
                         "normal_y_sign": -1 if normal else None})
    return material, {"path": path, "source_material": source["name"], "roughness": roughness,
                      "workflow": "specular_glossiness" if spec_gloss is not None else "metallic_roughness", "textures": textures}


def apply_visual_appearance(stage, task_root_path="/World/Task") -> dict:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    report = {"task_name": "eval_gen06_spray_shoe", "profile": "gen7_source_pbr_v1",
              "matched_prims": [], "matched_prim_count": 0, "material_count": 0,
              "materials": [], "warnings": [], "geometry_modified": False,
              "label_scope": "Printed bottle label retained in the original Spray_Body atlas",
              "manifest_sha256": hashlib.sha256((ASSETS / "materials.json").read_bytes()).hexdigest()}
    root = stage.GetPrimAtPath(task_root_path)
    if not root:
        report["warnings"].append(f"Task root is absent: {task_root_path}")
        return report
    specifications = source_materials()["materials"]
    for name, specification in specifications.items():
        meshes = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                  if p.IsA(UsdGeom.Mesh) and p.GetName() == name and "/visuals/" in str(p.GetPath())]
        if len(meshes) != 1:
            report["warnings"].append(f"Expected one source visual mesh for {name}, found {len(meshes)}")
            continue
        mesh = meshes[0]
        if mesh.HasAPI(UsdPhysics.CollisionAPI) or mesh.HasAPI(UsdPhysics.RigidBodyAPI):
            report["warnings"].append(f"Refused physical mesh binding: {mesh.GetPath()}")
            continue
        holder = mesh
        while holder.IsInstanceProxy():
            holder = holder.GetParent()
        if "/visuals/" not in str(holder.GetPath()):
            report["warnings"].append(f"No editable per-mesh visual binding for {name}")
            continue
        if specification["textures"] and not UsdGeom.PrimvarsAPI(mesh).FindPrimvarWithInheritance("st"):
            report["warnings"].append(f"Source visual UVs are absent: {mesh.GetPath()}")
            continue
        # Only weaken render-purpose ancestors inside the visual subtree.
        # Their original bindings and every physics-purpose binding remain.
        parent = holder.GetParent()
        while parent and "/visuals" in str(parent.GetPath()):
            rel = UsdShade.MaterialBindingAPI(parent).GetDirectBindingRel()
            if rel and rel.HasAuthoredTargets():
                UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(rel, UsdShade.Tokens.weakerThanDescendants)
            parent = parent.GetParent()
        path = str(Sdf.Path(task_root_path).AppendPath("Looks/GEN7SourcePBR/" + name))
        material, record = make_material(stage, path, specification)
        UsdShade.MaterialBindingAPI.Apply(holder).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        bound, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        if not bound or bound.GetPath() != material.GetPath():
            report["warnings"].append(f"Material overridden above visual subtree: {mesh.GetPath()}")
        record["binding_root"] = str(holder.GetPath())
        report["materials"].append(record)
        report["matched_prims"].append(str(mesh.GetPath()))
    report["matched_prim_count"] = len(report["matched_prims"])
    report["material_count"] = len(report["materials"])
    return report
