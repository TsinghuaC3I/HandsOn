"""Render-only finishes for the original Clear_Box and Box_lid assets."""

import hashlib
import json
from pathlib import Path


def physics_material_signature(stage, body_paths):
    """Include physical-purpose resolution, which the physics hash omits."""
    from pxr import Usd, UsdPhysics, UsdShade
    rows = []
    for path in sorted(set(body_paths)):
        root = stage.GetPrimAtPath(path)
        if not root:
            continue
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
            bindings = []
            ancestor = prim
            while ancestor and ancestor.GetPath().HasPrefix(root.GetPath()):
                bindings.extend((str(ancestor.GetPath()), rel.GetName(), list(map(str, rel.GetTargets())))
                                for rel in ancestor.GetRelationships()
                                if rel.GetName().startswith('material:binding'))
                ancestor = ancestor.GetParent()
            attrs = ({a.GetName(): str(a.Get()) for a in material.GetPrim().GetAttributes()
                      if a.GetName().startswith(('physics:', 'physx'))} if material else {})
            rows.append((str(prim.GetPath()), str(material.GetPath()) if material else None, bindings, attrs))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def render_meshes(stage, body_path, visual_name):
    """Expand only the named render group; never a rigid body or collider."""
    from pxr import Usd, UsdGeom, UsdPhysics
    root = stage.GetPrimAtPath(body_path)
    if not root:
        raise ValueError('Missing container body: ' + body_path)
    marker = '/visuals/' + visual_name + '/'
    def selected():
        return [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                if p.IsA(UsdGeom.Mesh) and marker in str(p.GetPath())
                and UsdGeom.Imageable(p).ComputeVisibility() != 'invisible']
    targets = set()
    for prim in selected():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('Container render mesh unexpectedly has collision: ' + str(prim.GetPath()))
        while prim.IsInstanceProxy():
            prim = prim.GetParent()
        if prim.IsInstance():
            if '/visuals/' not in str(prim.GetPath()) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('Refusing to expand a physical container root')
            targets.add(str(prim.GetPath()))
    for path in sorted(targets):
        stage.GetPrimAtPath(path).SetInstanceable(False)
    meshes = selected()
    if not meshes:
        raise ValueError('Missing original visible mesh group: ' + visual_name)
    return meshes


def source_color_texture(mesh):
    """Read the existing importer or PreviewSurface albedo texture and UV name."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
    if not material:
        raise ValueError('Missing source material: ' + str(mesh.GetPath()))
    candidates = []
    for prim in Usd.PrimRange(material.GetPrim()):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        direct = shader.GetInput('diffuse_texture')
        if direct and direct.Get():
            candidates.append(direct)
        if shader.GetIdAttr().Get() == 'UsdUVTexture':
            texture = shader.GetInput('file')
            if texture and texture.Get():
                candidates.append(texture)
    if not candidates:
        raise ValueError('No source albedo texture on ' + str(mesh.GetPath()))
    texture = candidates[0]
    asset = texture.Get()
    resolved = asset.resolvedPath
    if not resolved:
        stack = texture.GetAttr().GetPropertyStack()
        resolved = Sdf.ComputeAssetPathRelativeToLayer(stack[0].layer, asset.path) if stack else asset.path
    if not Path(resolved).is_file():
        raise ValueError('Unresolved original color texture: ' + str(resolved))
    uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st')
    if not uv or not uv.HasValue():
        raise ValueError('Original container mesh has no st UV: ' + str(mesh.GetPath()))
    return str(Path(resolved).resolve()), 'st'


def textured_plastic(stage, path, texture_path, uv_name, *, roughness):
    from pxr import Gf, Sdf, UsdShade
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    for key, value in (('roughness', roughness), ('metallic', 0.), ('ior', 1.49),
                       ('clearcoat', 0.), ('opacity', 1.)):
        shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(value)
    reader = UsdShade.Shader.Define(stage, path + '/UV')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set(uv_name)
    reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
    texture = UsdShade.Shader.Define(stage, path + '/Albedo')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
    texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1))
    texture.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0))
    texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
    texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), 'rgb')
    shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return material


def _clear_polymer(stage, path):
    # Glass and fractional surface opacity disappear in the collection
    # viewport. Use a light matte polymer surface so the real opening, rim
    # and walls remain visible; keep the source geometry and physics intact.
    from pxr import Gf, Sdf, UsdShade
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.62, .78, .86))
    for key, value in (('opacity', 1.), ('roughness', .42), ('metallic', 0.),
                       ('ior', 1.49), ('clearcoat', .15), ('opacityThreshold', 0.)):
        shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(value)
    shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return material


def bind_render_material(stage, meshes, material):
    from pxr import UsdShade
    for mesh in meshes:
        # The actual MJCF importer binds strongly on the common visuals root.
        # Retain that material as the fallback for all other visual children,
        # but let this selected child's authored material take precedence.
        visual_root = stage.GetPrimAtPath(str(mesh.GetPath()).split('/visuals/', 1)[0] + '/visuals')
        binding = UsdShade.MaterialBindingAPI(visual_root).GetDirectBinding()
        relation = binding.GetBindingRel()
        if relation and UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(relation) == 'strongerThanDescendants':
            UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(relation, 'weakerThanDescendants')
        group = mesh.GetParent()
        if '/visuals/' not in str(group.GetPath()):
            raise ValueError('Expected a source render group')
        UsdShade.MaterialBindingAPI.Apply(group).Bind(material, bindingStrength='strongerThanDescendants')
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material, bindingStrength='strongerThanDescendants')
        bound, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        if bound.GetPath() != material.GetPath():
            raise RuntimeError('Container appearance binding did not take effect')


def apply_clear_box_materials(stage, body_paths):
    """Use visible matte polymer for the box; retain the lid's exact source albedo."""
    from .blender_visual_layer import physics_signature
    paths = [body_paths[name] for name in ('Clear_Box', 'Box_lid')]
    before = physics_signature(stage)
    bindings = physics_material_signature(stage, paths)
    box = render_meshes(stage, body_paths['Clear_Box'], 'Clear_Box_vis')
    lid = render_meshes(stage, body_paths['Box_lid'], 'Box_lid_vis')
    texture, uv = source_color_texture(lid[0])
    polymer = _clear_polymer(stage, body_paths['Clear_Box'] + '/Looks/XYClearPolymer')
    lid_material = textured_plastic(stage, body_paths['Box_lid'] + '/Looks/XYSourceLid',
                                    texture, uv, roughness=.32)
    bind_render_material(stage, box, polymer)
    bind_render_material(stage, lid, lid_material)
    after = physics_signature(stage)
    after_bindings = physics_material_signature(stage, paths)
    if before != after or bindings != after_bindings:
        raise RuntimeError('Container finish changed the physical contract')
    return {'profile': 'visible_polymer_source_lid_v2', 'lid_source_texture': texture,
            'box_color': (.62, .78, .86),
            'box_ior': 1.49, 'box_roughness': .42, 'box_opacity': 1., 'lid_roughness': .32,
            'bound_meshes': [str(p.GetPath()) for p in (*box, *lid)],
            'physics_before': before, 'physics_after': after,
            'physics_material_before': bindings, 'physics_material_after': after_bindings}
