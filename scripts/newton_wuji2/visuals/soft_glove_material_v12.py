"""Add the pinned Kit native cloth response to the isolated Wuji2 v12 visuals.

CPU USD authoring only. This deliberately keeps the universal preview shader,
the existing textures, every material binding, and all geometry unchanged.
The caller owns manifest/dependency hash regeneration and RTX validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdShade


def mdl_source():
    """Locate the installed Kit shader without a machine-specific path."""
    import importlib.util
    import os
    if os.environ.get("HANDSON_OMNISURFACE_MDL"):
        return Path(os.environ["HANDSON_OMNISURFACE_MDL"]).resolve(strict=True)
    spec = importlib.util.find_spec("isaacsim")
    for root in (spec.submodule_search_locations or []) if spec else []:
        candidate = Path(root) / "kit/mdl/core/Base/OmniSurface.mdl"
        if candidate.is_file():
            return candidate
    raise RuntimeError("Set HANDSON_OMNISURFACE_MDL to Isaac Sim 5.1 Kit's OmniSurface.mdl")


FABRIC_NAMES = frozenset(("FineBlackKnit", "CoarseOpenKnit", "BlendedPalmKnit"))
MDL_SHADER_NAME = "SoftKnitMDL"
REVISION = "wuji2_soft_fabric_omnisurface_v12"

# Exact input names were checked in the pinned OmniSurface.mdl and its base.
# The retro-reflection lobe calls df::sheen_bsdf. No generic unsupported
# UsdPreviewSurface sheen attributes are authored.
FLOAT_INPUTS = {
    "diffuse_reflection_weight": 1.0,
    "diffuse_reflection_roughness": 0.55,
    "metalness": 0.0,
    "specular_reflection_weight": 0.18,
    "specular_reflection_roughness": 0.82,
    "specular_reflection_ior": 1.45,
    "specular_retro_reflection_weight": 0.28,
    "specular_retro_reflection_roughness": 0.72,
    "geometry_normal_strength": 1.0,
    "geometry_normal_roughness_strength": 1.0,
    "coat_weight": 0.0,
    "emission_weight": 0.0,
    "uvw_s_scale": 1.0,
    "uvw_t_scale": 1.0,
}
COLOR_INPUTS = {
    # Native OmniSurface uses the texture instead of this fallback when valid.
    "diffuse_reflection_color": (0.026, 0.026, 0.026),
    "specular_reflection_color": (1.0, 1.0, 1.0),
    "specular_retro_reflection_color": (0.19, 0.20, 0.21),
}
BOOL_INPUTS = {
    "enable_diffuse_transmission": False,
    "enable_specular_transmission": False,
    "geometry_normal_image_flip_r_channel": False,
    "geometry_normal_image_flip_g_channel": False,
    "geometry_normal_image_flip_b_channel": False,
    "uvw_use_uv_coords": False,
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _texture_path(bundle, path):
    path = Path(path)
    if not path.is_absolute():
        path = bundle / "textures" / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(bundle):
        raise ValueError(f"Texture escapes isolated bundle: {path}")
    return path


def _verify_interface():
    """Catch wrong deployment/runtime before writing asset files."""
    source = mdl_source().read_text()
    required = set(FLOAT_INPUTS) | set(COLOR_INPUTS) | set(BOOL_INPUTS) | {
        "diffuse_reflection_color_image", "geometry_normal_image",
        "uvw_uv_set", "uvw_s_wrap", "uvw_t_wrap",
    }
    absent = sorted(name for name in required if name not in source)
    if absent or "export material OmniSurface(" not in source:
        raise RuntimeError(f"Pinned OmniSurface interface differs: {absent}")


def attach_fabric_mdl(stage, material, bundle, color_file, normal_file, *, wrap="repeat"):
    """Attach the same MDL response to an existing fabric material, without saving.

    `material` is a UsdShade.Material or absolute material prim path. Texture
    filenames may be relative to bundle/textures or absolute within the bundle.
    Callers creating dynamic joint sleeves can keep their preview fallback and
    then call this function. Native UV set zero uses the mesh's authored st.
    """
    _verify_interface()
    bundle = Path(bundle).resolve(strict=True)
    color = _texture_path(bundle, color_file)
    normal = _texture_path(bundle, normal_file)
    if wrap not in ("repeat", "clamp"):
        raise ValueError(f"Unsupported existing texture wrap: {wrap}")
    if not isinstance(material, UsdShade.Material):
        material = UsdShade.Material.Get(stage, str(material))
    if not material:
        raise ValueError("Existing fabric material is required")
    shader = UsdShade.Shader.Define(
        stage, material.GetPath().AppendChild(MDL_SHADER_NAME)
    )
    shader.CreateImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(str(mdl_source())), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")
    for name, value in FLOAT_INPUTS.items():
        shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
    for name, value in COLOR_INPUTS.items():
        shader.CreateInput(name, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*value))
    for name, value in BOOL_INPUTS.items():
        shader.CreateInput(name, Sdf.ValueTypeNames.Bool).Set(value)
    shader.CreateInput("uvw_uv_set", Sdf.ValueTypeNames.Int).Set(0)
    # Native OmniImage enum: wrap_periodic=0, wrap_clamp=2.
    for axis in ("s", "t"):
        shader.CreateInput(f"uvw_{axis}_wrap", Sdf.ValueTypeNames.Int).Set(
            0 if wrap == "repeat" else 2
        )
    for name, path, color_space in (
        ("diffuse_reflection_color_image", color, "sRGB"),
        ("geometry_normal_image", normal, "raw"),
    ):
        inp = shader.CreateInput(name, Sdf.ValueTypeNames.Asset)
        inp.Set(Sdf.AssetPath(str(path)))
        inp.GetAttr().SetColorSpace(color_space)
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return {
        "material": str(material.GetPath()),
        "mdl_shader": str(shader.GetPath()),
        "color": str(color.relative_to(bundle)),
        "normal": str(normal.relative_to(bundle)),
        "color_sha256": _sha(color),
        "normal_sha256": _sha(normal),
        "wrap": wrap,
    }


def _fabric_materials(stage):
    return [UsdShade.Material(p) for p in stage.Traverse()
            if p.IsA(UsdShade.Material) and p.GetName() in FABRIC_NAMES]


def _preserved_digest(stage):
    """Hash every composed property except our new shader and MDL output.

    Includes mesh points, topology, transforms, UVs, material bindings, all
    nonfabric materials, existing preview fabric shaders, and stage metadata.
    """
    layer = stage.Flatten()
    edits = Sdf.BatchNamespaceEdit()
    for material in _fabric_materials(stage):
        for path in (
            material.GetPath().AppendChild(MDL_SHADER_NAME),
            material.GetPath().AppendProperty("outputs:mdl:surface"),
        ):
            if layer.GetObjectAtPath(path):
                edits.Add(path, Sdf.Path.emptyPath)
    if not layer.Apply(edits):
        raise RuntimeError("Could not make preservation audit layer")
    return hashlib.sha256(layer.ExportToString().encode()).hexdigest()


def _preview_textures(stage, material, asset, bundle):
    paths = []
    wraps = []
    for kind in ("Color", "Normal"):
        node = UsdShade.Shader.Get(stage, material.GetPath().AppendChild(kind))
        if not node:
            raise RuntimeError(f"Missing existing {kind} texture: {material.GetPath()}")
        value = node.GetInput("file").Get()
        path = Path(value.path)
        path = path if path.is_absolute() else asset.parent / path
        paths.append(_texture_path(bundle, path))
        for axis in ("S", "T"):
            wraps.append(node.GetInput("wrap" + axis).Get())
    if len(set(wraps)) != 1:
        raise RuntimeError(f"Mixed wrapping needs explicit mapping: {wraps}")
    return (*paths, wraps[0])


def apply_to_bundle(bundle_path):
    """Add fabric MDL context only to assets/links in the named v12 bundle.

    Does not update manifest hashes. Returns a JSON-safe provenance report.
    The parent integrates this after its geometry edits and before hashing.
    """
    requested = Path(bundle_path).absolute()
    bundle = requested.resolve(strict=True)
    if bundle.name != "wuji2_photo_glove_link_v12" or requested.is_symlink():
        raise ValueError("Only a separate wuji2_photo_glove_link_v12 bundle may be edited")
    _verify_interface()
    assets = sorted((bundle / "links").glob("*.usd*"))
    if not assets:
        raise ValueError("No visual link assets found")
    for asset in assets:
        if asset.is_symlink() or not asset.resolve().is_relative_to(bundle):
            raise ValueError(f"Shared/symlink asset refused: {asset}")
        if asset.stat().st_nlink > 1:
            raise ValueError(f"Hard-linked asset refused: {asset}")
    report = {
        "revision": REVISION,
        "bundle": str(bundle),
        "mdl_source": str(mdl_source()),
        "mdl_source_sha256": _sha(mdl_source()),
        "float_inputs": FLOAT_INPUTS,
        "color_inputs": COLOR_INPUTS,
        "bool_inputs": BOOL_INPUTS,
        "asset_count": len(assets),
        "textures_modified": False,
        "geometry_modified_by_this_helper": False,
        "gpu_appearance_verified": False,
        "links": [],
    }
    for asset in assets:
        stage = Usd.Stage.Open(str(asset))
        before = _preserved_digest(stage)
        item = {"asset": str(asset.relative_to(bundle)), "materials": []}
        for material in _fabric_materials(stage):
            color, normal, wrap = _preview_textures(stage, material, asset, bundle)
            item["materials"].append(attach_fabric_mdl(
                stage, material, bundle, color, normal, wrap=wrap
            ))
        after = _preserved_digest(stage)
        if before != after:
            raise AssertionError(f"Material-only preservation failed before save: {asset}")
        stage.GetRootLayer().Save()
        # Reload confirms serialization, not only in-memory authoring.
        stage.GetRootLayer().Reload()
        if _preserved_digest(stage) != before:
            raise AssertionError(f"Material-only preservation failed after reload: {asset}")
        for material in _fabric_materials(stage):
            source = material.ComputeSurfaceSource("mdl")
            if not source or not source[0] or source[0].GetPath().name != MDL_SHADER_NAME:
                raise AssertionError(f"MDL surface resolution failed: {material.GetPath()}")
        item.update(preserved_content_sha256=before, asset_sha256=_sha(asset))
        report["links"].append(item)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    args = parser.parse_args()
    print(json.dumps(apply_to_bundle(args.bundle), indent=2))
