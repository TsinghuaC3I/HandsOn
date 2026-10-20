"""Photo-referenced XY2 refinement on the existing task's visual prims."""
from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Any, Mapping

XY2_VISUAL_PILOT_ENV = 'HANDSON_XY2_VISUAL_PILOT'
XY2_VISUAL_PILOT_PROFILE = 'xy2_photo1017_laptop_photo1018_black_weave'
_MATERIAL_PATH = '/World/Looks/XY2VisualPilot/BlackWovenMousepad'


def xy2_visual_pilot_enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    return source.get(XY2_VISUAL_PILOT_ENV, '0').strip().lower() in {'1', 'true', 'yes'}


def _woven_material(stage):
    from pxr import Gf, Sdf, UsdShade
    assets = Path(__file__).resolve().parents[4] / 'assets/xy2_appearance'
    material = UsdShade.Material.Define(stage, _MATERIAL_PATH)
    surface = UsdShade.Shader.Define(stage, _MATERIAL_PATH + '/Surface')
    surface.CreateIdAttr('UsdPreviewSurface')
    surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.016, .017, .018))
    surface.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.74)
    surface.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
    surface.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(1.46)
    surface.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(1.)
    st = UsdShade.Shader.Define(stage, _MATERIAL_PATH + '/UV')
    st.CreateIdAttr('UsdPrimvarReader_float2')
    st.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    st.CreateOutput('result', Sdf.ValueTypeNames.Float2)
    for name, file, cs, input_name, output, value_type in [
        ('Albedo', 'black_weave_basecolor.png', 'sRGB', 'diffuseColor', 'rgb', Sdf.ValueTypeNames.Color3f),
        ('Roughness', 'black_weave_roughness.png', 'raw', 'roughness', 'r', Sdf.ValueTypeNames.Float),
        ('Normal', 'black_weave_normal.png', 'raw', 'normal', 'rgb', Sdf.ValueTypeNames.Normal3f),
    ]:
        path = assets / file
        if not path.is_file():
            raise FileNotFoundError('XY2 woven map missing: ' + str(path))
        tex = UsdShade.Shader.Define(stage, _MATERIAL_PATH + '/' + name)
        tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(path)))
        tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set(cs)
        tex.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
        tex.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
        tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), 'result')
        tex.CreateOutput(output, Sdf.ValueTypeNames.Float if output == 'r' else Sdf.ValueTypeNames.Float3)
        if name == 'Normal':
            tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            tex.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        surface.CreateInput(input_name, value_type).ConnectToSource(tex.ConnectableAPI(), output)
    surface.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), 'surface')
    return material


def _smooth_pad_mesh(stage, path, radius, height):
    """Replace the existing Z cylinder with a bevel mesh of the same bounds."""
    from pxr import Gf, Sdf, UsdGeom, Vt
    half, bevel, segments = height / 2, min(.00045, height / 4), 256
    rings = [(radius - bevel, half, 0, 1),
             (radius - bevel * .293, half - bevel * .293, .7071068, .7071068),
             (radius, half - bevel, 1, 0), (radius, -half + bevel, 1, 0),
             (radius - bevel * .293, -half + bevel * .293, .7071068, -.7071068),
             (radius - bevel, -half, 0, -1)]
    points, normals, uv = [(0, 0, half)], [(0, 0, 1)], [(.5, .5)]
    for r, z, nr, nz in rings:
        for i in range(segments):
            a = 2 * math.pi * i / segments
            x, y = r * math.cos(a), r * math.sin(a)
            points.append((x, y, z)); normals.append((nr * math.cos(a), nr * math.sin(a), nz))
            uv.append((x / (2 * radius) + .5, y / (2 * radius) + .5))
    bottom = len(points)
    points.append((0, 0, -half)); normals.append((0, 0, -1)); uv.append((.5, .5))
    faces, face_uv = [], []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((0, 1 + i, 1 + j))
        face_uv.extend([uv[index] for index in faces[-1]])
        for ring in range(len(rings) - 1):
            a, b = 1 + ring * segments, 1 + (ring + 1) * segments
            faces.append((a + i, b + i, b + j, a + j))
            # Unwrap the bevel by physical arc length; a planar UV would
            # collapse vertical rim quads and invalidate tangent normals.
            u0, u1 = math.pi * i / segments, math.pi * (i + 1) / segments
            v0 = (half - rings[ring][1]) / (2 * radius)
            v1 = (half - rings[ring + 1][1]) / (2 * radius)
            face_uv.extend([(u0, v0), (u0, v1), (u1, v1), (u1, v0)])
        last = 1 + (len(rings) - 1) * segments
        faces.append((bottom, last + j, last + i))
        face_uv.extend([uv[index] for index in faces[-1]])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr([(-radius, -radius, -half), (radius, radius, half)])
    # Bake the physical fabric scale into UVs for native RTX compatibility.
    # Sixteen repeats of sixteen yarns gives 1.17 mm spacing across this pad.
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set([(u * 16, v * 16) for u, v in face_uv])
    st.BlockIndices()
    mesh.GetPrim().CreateAttribute('userProperties:xy2PadRadius', Sdf.ValueTypeNames.Double, custom=True).Set(radius)
    mesh.GetPrim().CreateAttribute('userProperties:xy2PadHeight', Sdf.ValueTypeNames.Double, custom=True).Set(height)
    return mesh.GetPrim()


def apply_xy2_visual_pilot(stage: Any, mousepad: Any, *, base_visual_path=None, display_visual_path=None) -> dict[str, Any]:
    """Refine XY2 visuals, preserving body transforms and all physical schemas.

    IMG_1018 informs yarn/roughness, not bag geometry. The imported Z cylinder
    has a stale X-axis extent: use its radius/height and fix only visual extent.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    from .blender_visual_layer import physics_signature
    before = physics_signature(stage)
    report = {'profile': XY2_VISUAL_PILOT_PROFILE, 'bound_paths': [], 'rebuilt_paths': []}
    if mousepad is None or not mousepad.IsValid():
        report['warning'] = 'Source mousepad was not found; no appearance change'
        return report
    candidates = []
    for prim in Usd.PrimRange(mousepad, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim) or prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if '/collisions/' in str(prim.GetPath()) or UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        radius = height = None
        if prim.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(prim)
            if cylinder.GetAxisAttr().Get() != UsdGeom.Tokens.z:
                raise ValueError('XY2 pad expected its source Z-axis cylinder')
            radius, height = float(cylinder.GetRadiusAttr().Get()), float(cylinder.GetHeightAttr().Get())
        elif prim.GetAttribute('userProperties:xy2PadRadius'):
            radius = float(prim.GetAttribute('userProperties:xy2PadRadius').Get())
            height = float(prim.GetAttribute('userProperties:xy2PadHeight').Get())
        candidates.append((str(prim.GetPath()), radius, height))
    if not candidates:
        report['warning'] = 'No visible source pad geometry; no appearance change'
        return report
    material = _woven_material(stage)
    for path, radius, height in candidates:
        prim = stage.GetPrimAtPath(path)
        # Override only this instance's visual prim, never its shared prototype.
        while prim.IsInstanceProxy():
            instance = prim
            while instance.IsInstanceProxy():
                instance = instance.GetParent()
            if 'visuals' not in str(instance.GetPath()).split('/'):
                raise RuntimeError('Refusing to deinstance a non-visual pad body')
            instance.SetInstanceable(False)
            prim = stage.GetPrimAtPath(path)
        if radius is not None:
            prim = _smooth_pad_mesh(stage, path, radius, height)
            report['rebuilt_paths'].append(path)
            report['pad_bounds_m'] = [[-radius, -radius, -height / 2], [radius, radius, height / 2]]
        ancestor = prim.GetParent()
        while ancestor.IsValid() and ancestor != mousepad:
            if ancestor.HasAPI(UsdShade.MaterialBindingAPI):
                UsdShade.MaterialBindingAPI(ancestor).UnbindAllBindings()
            ancestor = ancestor.GetParent()
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        actual, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if actual.GetPath() != material.GetPath():
            raise RuntimeError('Old source material still overrides the black pad')
        report['bound_paths'].append(path)
    report['material_path'] = str(material.GetPath())
    if base_visual_path and display_visual_path:
        from .xy2_laptop_visual import rebuild_xy2_laptop_visual
        report['laptop'] = rebuild_xy2_laptop_visual(stage, base_visual_path=base_visual_path, display_visual_path=display_visual_path)
    after = physics_signature(stage)
    if after != before:
        raise RuntimeError('XY2 photo refinement changed physical configuration')
    report['physics_before'], report['physics_after'] = before, after
    return report
