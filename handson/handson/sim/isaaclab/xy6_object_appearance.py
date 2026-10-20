"""Opt-in XY6 render materials, applied once during scene construction.

The adjacent source GLB declares a neutral, non-metallic utensil material.
Its USD converter retained displayColor but dropped the PBR inputs. Restore
those source inputs and decode the pot's existing tangent-space normal maps
without changing any mesh, transform, source texture, or physics schema.
Use a fresh stage/process when comparing this profile with the source baseline.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath
from typing import Mapping
import warnings


APPEARANCE_ENV = "HANDSON_XY6_OBJECT_APPEARANCE"
APPEARANCE_PROFILE = "neutral_v1"
# dexverse_authored_spoon.glb: materials[0].pbrMetallicRoughness.
SPOON_BASE_COLOR = (0.46000000834465027, 0.5, 0.5199999809265137)
SPOON_METALLIC = 0.0
SPOON_ROUGHNESS = 0.14000000059604645


def _repair_pot_texture_inputs(stage, body_paths: Mapping[str, str]) -> list[dict]:
    from pxr import Gf, Sdf, Usd, UsdShade

    repairs = []
    for name in ("dutch_oven", "dutch_oven_lid"):
        path = body_paths.get(name, "")
        root = stage.GetPrimAtPath(path) if path else None
        if not root or not root.IsValid():
            continue
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdShade.Shader):
                continue
            surface = UsdShade.Shader(prim)
            if surface.GetIdAttr().Get() != "UsdPreviewSurface":
                continue
            for input_name, filename, values in (
                ("normal", "pot_enamel_01_nor_gl_4k.jpg", {
                    # UsdPreviewSurface expects a signed tangent-space normal.
                    # https://openusd.org/release/spec_usdpreviewsurface.html
                    "sourceColorSpace": (Sdf.ValueTypeNames.Token, "raw"),
                    "scale": (Sdf.ValueTypeNames.Float4, Gf.Vec4f(2, 2, 2, 1)),
                    "bias": (Sdf.ValueTypeNames.Float4, Gf.Vec4f(-1, -1, -1, 0)),
                }),
                ("diffuseColor", "pot_enamel_01_diff_4k.jpg", {
                    "sourceColorSpace": (Sdf.ValueTypeNames.Token, "sRGB"),
                }),
            ):
                source = surface.GetInput(input_name).GetConnectedSource()
                if source is None:
                    continue
                texture = UsdShade.Shader(source[0].GetPrim())
                if (not str(texture.GetPath()).startswith(path + "/")
                        or texture.GetIdAttr().Get() != "UsdUVTexture"):
                    continue
                file_value = texture.GetInput("file").Get()
                if file_value is None or PurePosixPath(file_value.path).name != filename:
                    continue
                changed = []
                for key, (value_type, value) in values.items():
                    shader_input = texture.GetInput(key)
                    if shader_input.Get() != value:
                        texture.CreateInput(key, value_type).Set(value)
                        changed.append(key)
                if changed:
                    repairs.append({"shader_path": str(texture.GetPath()), "inputs": changed})
    return repairs


def apply_xy6_object_appearance(
    stage,
    *,
    body_paths: Mapping[str, str],
    profile: str | None = None,
) -> dict:
    """Restore spoon PBR and pot normal decoding; preserve the jars' identity."""
    selected = str(
        os.environ.get(APPEARANCE_ENV, "source") if profile is None else profile
    ).strip().lower()
    report = {
        "profile": selected,
        "visual_only": True,
        "spoon_material_source": "dexverse_authored_spoon.glb",
        "spoon_mesh_paths": [],
        "pot_source_textures_preserved": True,
        "pot_shader_repairs": [],
        "salt_pepper_materials_preserved": True,
    }
    if selected in {"", "source"}:
        report["profile"] = "source"
        return report
    if selected != APPEARANCE_PROFILE:
        report["warning"] = f"Unknown XY6 object appearance profile: {selected!r}"
        warnings.warn(report["warning"], RuntimeWarning, stacklevel=2)
        return report

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    spoon_path = body_paths.get("spoon", "")
    root = stage.GetPrimAtPath(spoon_path) if spoon_path else None
    meshes = (
        [prim for prim in Usd.PrimRange(root)
         if prim.IsA(UsdGeom.Mesh) and not prim.IsInstanceProxy()
         and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible]
        if root and root.IsValid() else []
    )
    if not meshes:
        report["warning"] = "XY6 spoon has no editable visible render mesh"
        warnings.warn(report["warning"], RuntimeWarning, stacklevel=2)
        return report

    material = UsdShade.Material.Define(stage, f"{spoon_path}/Looks/XY6NeutralUtensil")
    shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*SPOON_BASE_COLOR))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(SPOON_METALLIC)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(SPOON_ROUGHNESS)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for mesh in meshes:
        # The source physics-purpose material binding is separate and retained.
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        report["spoon_mesh_paths"].append(str(mesh.GetPath()))
    report["pot_shader_repairs"] = _repair_pot_texture_inputs(stage, body_paths)
    return report
