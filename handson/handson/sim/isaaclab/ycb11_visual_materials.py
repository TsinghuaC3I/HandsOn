"""YCB11-only ceramic shading; keeps source geometry and all physics intact."""
from pathlib import Path


def bind_ycb11_mug_glaze(stage, texture_path: Path) -> tuple[str, ...]:
    """Bind a source-albedo glazed dielectric only on YCB11 mug visuals."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    targets = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if "/ycb11_mug/" not in path or "/visuals/" not in path:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        target = prim
        while target.IsInstanceProxy():
            target = target.GetParent()
        # Binding at an instance root is allowed only within this visual branch.
        if "/visuals/" in str(target.GetPath()) or str(target.GetPath()).endswith("/visuals"):
            targets.append(target)
    if not targets:
        return ()

    material_path = "/World/Looks/YCB11MugGlaze"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    for name, value in (("roughness", 0.22), ("metallic", 0.0), ("ior", 1.5), ("opacity", 1.0)):
        shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
    reader = UsdShade.Shader.Define(stage, material_path + "/PrimvarReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, material_path + "/BaseColor")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(texture_path).resolve())))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    bound = []
    for prim in targets:
        path = str(prim.GetPath())
        if path in bound:
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        bound.append(path)
    return tuple(bound)
