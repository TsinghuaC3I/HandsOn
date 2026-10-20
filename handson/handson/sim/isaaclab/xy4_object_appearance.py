"""Opt-in surface-only overrides for XY4's existing source assets."""

import hashlib
import json
import os


def physics_material_bindings(stage):
    """Snapshot explicit physics bindings and each collider's resolved material."""
    from pxr import UsdPhysics, UsdShade
    explicit = {}
    resolved = {}
    for prim in stage.Traverse():
        for relation in prim.GetRelationships():
            name = relation.GetName()
            if name.startswith("material:binding") and "physics" in name.split(":"):
                explicit[str(relation.GetPath())] = {
                    "targets": [str(path) for path in relation.GetTargets()],
                    "strength": str(UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(relation)),
                }
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            material, relation = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            resolved[str(prim.GetPath())] = {
                "material": str(material.GetPath()) if material else None,
                "binding": str(relation.GetPath()) if relation else None,
            }
    return {"explicit": explicit, "resolved_colliders": resolved}


def apply_xy4_object_appearance(stage, *, kitchen_path, faucet_paths, mug_path,
                                flow_path, environ=None):
    environ = os.environ if environ is None else environ
    profile = environ.get("HANDSON_XY4_OBJECT_APPEARANCE", "source").strip()
    if profile == "source":
        return None
    if profile != "source_pbr_v1":
        raise ValueError("Unknown XY4 appearance profile: " + profile)
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    from .blender_visual_layer import physics_signature

    before = physics_signature(stage)
    bindings = physics_material_bindings(stage)
    edits = []

    def set_input(shader, name, value, kind):
        item = shader.GetInput(name)
        if item and item.HasConnectedSource():
            raise RuntimeError("Refusing to replace a source texture: " + str(item.GetAttr().GetPath()))
        item = shader.CreateInput(name, kind)
        item.Set(value)
        edits.append(str(item.GetAttr().GetPath()))

    def shader_at(path):
        shader = UsdShade.Shader(stage.GetPrimAtPath(path))
        if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
            raise RuntimeError("XY4 source shader unavailable: " + path)
        return shader

    # Keep the sink metallic and the faucet black. Different roughness makes
    # their silhouettes legible under the shared neutral fill.
    sink = shader_at(kitchen_path + "/Looks/StainlessSink/PBR")
    set_input(sink, "diffuseColor", Gf.Vec3f(.34, .37, .39), Sdf.ValueTypeNames.Color3f)
    set_input(sink, "roughness", .34, Sdf.ValueTypeNames.Float)
    for root in faucet_paths:
        prim = stage.GetPrimAtPath(root)
        if not prim:
            raise RuntimeError("XY4 faucet visual unavailable: " + root)
        for candidate in Usd.PrimRange(prim):
            if candidate.IsA(UsdShade.Shader) and "/Material_001/" in str(candidate.GetPath()):
                shader = shader_at(str(candidate.GetPath()))
                set_input(shader, "diffuseColor", Gf.Vec3f(.06, .06, .06), Sdf.ValueTypeNames.Color3f)
                set_input(shader, "roughness", .30, Sdf.ValueTypeNames.Float)
                set_input(shader, "metallic", .75, Sdf.ValueTypeNames.Float)

    bottle = shader_at(kitchen_path + "/Bottle/Visual/Asset/Looks/Plastic/Plastic")
    set_input(bottle, "roughness", .26, Sdf.ValueTypeNames.Float)
    brush = shader_at(kitchen_path + "/Brush/Visual/Asset/Looks/Material_026/Material_026")
    set_input(brush, "roughness", .32, Sdf.ValueTypeNames.Float)

    # Preserve the source label/color and ORM channel connections. These
    # glTF-derived normal PNGs need unsigned [0,1] -> signed [-1,1] decoding.
    for root in (kitchen_path + "/Bottle", kitchen_path + "/Brush", mug_path):
        prim = stage.GetPrimAtPath(root)
        if not prim:
            raise RuntimeError("XY4 source visual unavailable: " + root)
        for candidate in Usd.PrimRange(prim):
            if not candidate.IsA(UsdShade.Shader):
                continue
            texture = UsdShade.Shader(candidate)
            if texture.GetIdAttr().Get() != "UsdUVTexture":
                continue
            name = candidate.GetName().lower()
            if name in {"diffusetex", "emmissivetex", "emissivetex"}:
                set_input(texture, "sourceColorSpace", "sRGB", Sdf.ValueTypeNames.Token)
            elif name == "normaltex":
                set_input(texture, "sourceColorSpace", "raw", Sdf.ValueTypeNames.Token)
                set_input(texture, "scale", Gf.Vec4f(2, 2, 2, 1), Sdf.ValueTypeNames.Float4)
                set_input(texture, "bias", Gf.Vec4f(-1, -1, -1, 0), Sdf.ValueTypeNames.Float4)

    # The unchanged cylinder still follows the existing faucet/cup state.
    # No new liquid geometry, fill state, transform or visibility is authored.
    flow = stage.GetPrimAtPath(flow_path)
    if not flow:
        raise RuntimeError("XY4 stream prim unavailable: " + flow_path)
    # IsaacLab CylinderCfg authors an Xform root and binds the visual material
    # to root/geometry/mesh. Resolve from rendered geometry, not its parent.
    materials = {}
    for candidate in Usd.PrimRange(flow, Usd.TraverseInstanceProxies()):
        if candidate.IsA(UsdGeom.Gprim):
            material, _ = UsdShade.MaterialBindingAPI(candidate).ComputeBoundMaterial()
            if material:
                materials[str(material.GetPath())] = material
    if len(materials) != 1:
        raise RuntimeError("XY4 stream must have one bound visual material: " + flow_path)
    material = next(iter(materials.values()))
    water, _, _ = material.ComputeSurfaceSource()
    if not water or water.GetIdAttr().Get() != "UsdPreviewSurface":
        raise RuntimeError("XY4 stream must retain its PreviewSurface shader")
    for name, value, kind in (
        ("diffuseColor", Gf.Vec3f(.68, .74, .76), Sdf.ValueTypeNames.Color3f),
        ("emissiveColor", Gf.Vec3f(0), Sdf.ValueTypeNames.Color3f),
        ("opacity", .42, Sdf.ValueTypeNames.Float),
        ("roughness", .10, Sdf.ValueTypeNames.Float),
        ("ior", 1.333, Sdf.ValueTypeNames.Float),
    ):
        set_input(water, name, value, kind)

    # PreviewSurface opacity is a cutout/coverage fallback in native RTX, not
    # water transmission. Select the already supported OmniGlass MDL context
    # on this same material, retaining the original universal surface output.
    # The existing USD cylinder is closed, so it uses a volume boundary.
    transmission = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/WaterTransmission")
    transmission.CreateImplementationSourceAttr().Set("sourceAsset")
    transmission.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    transmission.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    for name, value, kind in (
        ("glass_color", Gf.Vec3f(.999), Sdf.ValueTypeNames.Color3f),
        ("glass_ior", 1.333, Sdf.ValueTypeNames.Float),
        ("frosting_roughness", .02, Sdf.ValueTypeNames.Float),
        ("thin_walled", False, Sdf.ValueTypeNames.Bool),
        ("depth", 1.0, Sdf.ValueTypeNames.Float),
        ("enable_opacity", False, Sdf.ValueTypeNames.Bool),
    ):
        set_input(transmission, name, value, kind)
    transmission.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for output in ("surface", "volume", "displacement"):
        material.CreateOutput("mdl:" + output, Sdf.ValueTypeNames.Token).ConnectToSource(
            transmission.ConnectableAPI(), "out")

    after = physics_signature(stage)
    if before != after or bindings != physics_material_bindings(stage):
        raise RuntimeError("XY4 appearance changed physics or physics-purpose bindings")
    return {"profile": profile, "shader_inputs": edits,
            "physics_before": before, "physics_after": after,
            "physics_material_bindings_sha256": hashlib.sha256(
                json.dumps(bindings, sort_keys=True).encode()).hexdigest(),
            "water_state_model": "unchanged_existing_faucet_stream_cylinder",
            "water_render_context": "mdl", "water_material_model": "OmniGlass.mdl",
            "water_ior": 1.333, "water_roughness": .02}
