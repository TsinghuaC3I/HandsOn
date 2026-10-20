"""XY5 source finishes and the selected metallic lunchbox overlay.

The pitcher geometry and UVs remain unchanged. The selected XY5 lunchbox is a
render-only source overlay; interior shading is a material subset of the
existing pitcher faces, not an added liner or collision mesh.
"""
import os

from .xy_container_appearance import (
    apply_clear_box_materials, bind_render_material, physics_material_signature,
    render_meshes, source_color_texture, textured_plastic,
)


PROFILE = 'source_blue_visible_box_v5'
# The source B05 Clear_Box and Box_lid share a 292 x 429 mm footprint and a
# 149 mm Clear_Box height.  The downloaded visual is 202 x 188.104 x 102 mm;
# scale only this render overlay so its visible footprint matches the lid while
# the authoritative source collision bodies remain unchanged.
_SOURCE_BOX_SIZE_M = (0.292, 0.429, 0.149)
_DOWNLOADED_BOX_SIZE_M = (0.202, 0.1881043017, 0.1020000209)
_DOWNLOADED_BOX_CENTER_Z_M = (0.0040000002 + 0.1060000211) / 2.0
_DOWNLOADED_BOX_SCALE = tuple(
    source / downloaded
    for source, downloaded in zip(_SOURCE_BOX_SIZE_M, _DOWNLOADED_BOX_SIZE_M)
)
_DOWNLOADED_BOX_TRANSLATE_M = (
    0.0, 0.0, -_DOWNLOADED_BOX_SCALE[2] * _DOWNLOADED_BOX_CENTER_Z_M
)
DOWNLOADED_BOX_USD = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'envs', 'usd',
                 'eval_gen07_stack_boxes', 'downloaded_plastic_box.usd')
)


def _pitcher_interior_faces(mesh):
    """Select inward-facing cavity walls and floor on the source upright mesh.

    The source has a real inner floor and tapered walls. Infer the cavity axis
    from that upward-facing floor, then exclude the handle outside the body's
    tapered radial envelope. This leaves the original lip/spout finish intact.
    """
    import numpy as np
    from pxr import UsdGeom
    geometry = UsdGeom.Mesh(mesh)
    points = np.asarray(geometry.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(geometry.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.asarray(geometry.GetFaceVertexIndicesAttr().Get(), dtype=int)
    faces = np.split(indices, np.cumsum(counts)[:-1])
    vertices = [points[f] for f in faces]
    centers = np.asarray([v.mean(axis=0) for v in vertices])
    normals = np.asarray([np.cross(v[1] - v[0], v[2] - v[0]) for v in vertices])
    areas = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(areas[:, None], 1e-12)
    if geometry.GetOrientationAttr().Get() == 'leftHanded':
        normals *= -1
    bottom = float(points[:, 2].min())
    floor = (normals[:, 2] > .9) & (centers[:, 2] < bottom + .012)
    if not floor.any():
        raise ValueError('Source pitcher cavity floor was not found')
    axis = np.average(centers[floor, :2], weights=areas[floor], axis=0)
    floor_z = float(np.average(centers[floor, 2], weights=areas[floor]))
    floor_vertices = np.concatenate([v for v, selected in zip(vertices, floor) if selected])
    floor_radius = np.linalg.norm(floor_vertices[:, :2] - axis, axis=1).max()
    top_radius = np.abs(points[:, 0] - axis[0]).max()
    height = float(points[:, 2].max()) - floor_z
    radial = centers[:, :2] - axis
    radius = np.linalg.norm(radial, axis=1)
    radial_normal = (normals[:, :2] * radial).sum(axis=1) / np.maximum(radius, 1e-12)
    envelope = floor_radius + (top_radius - floor_radius) * np.clip((centers[:, 2] - floor_z) / height, 0, 1)
    wall = ((radial_normal < -.65) & (radius <= envelope + .001)
            & (centers[:, 2] >= floor_z - .001))
    selected = np.flatnonzero(floor | wall).tolist()
    if not wall.any() or len(selected) >= len(faces):
        raise ValueError('Source pitcher inner/outer face separation is invalid')
    return selected


def apply_xy5_object_appearance(stage, body_paths):
    from pxr import UsdGeom, UsdShade
    from .blender_visual_layer import physics_signature

    before = physics_signature(stage)
    physical_materials = physics_material_signature(stage, list(body_paths.values()))
    report = apply_clear_box_materials(stage, body_paths)
    # Use the downloaded tjgatt lunchbox as the XY5 box appearance while keeping
    # the existing Clear_Box/Box_lid collision bodies and articulation.  The
    # downloaded asset is a single closed mesh, so it is deliberately visual
    # only; the lid semantics remain owned by the existing Box_lid body.
    if os.environ.get('HANDSON_XY5_DOWNLOADED_BOX', '0').strip() == '1':
        clear_path = body_paths['Clear_Box']
        clear_visuals = render_meshes(stage, clear_path, 'Clear_Box_vis')
        for mesh in clear_visuals:
            UsdGeom.Imageable(mesh).MakeInvisible()
        downloaded = UsdGeom.Xform.Define(stage, clear_path + '/DownloadedPlasticBox')
        translate_op = next(
            (
                op for op in downloaded.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ),
            None,
        )
        if translate_op is None:
            translate_op = downloaded.AddTranslateOp()
        translate_op.Set(_DOWNLOADED_BOX_TRANSLATE_M)
        scale_op = next(
            (
                op for op in downloaded.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeScale
            ),
            None,
        )
        if scale_op is None:
            scale_op = downloaded.AddScaleOp()
        scale_op.Set(_DOWNLOADED_BOX_SCALE)
        downloaded.GetPrim().GetReferences().AddReference(DOWNLOADED_BOX_USD, '/PlasticBox')
        report['downloaded_source'] = {
            'asset': DOWNLOADED_BOX_USD,
            'title': 'Lunchbox',
            'author': 'tjgatt',
            'source': 'https://sketchfab.com/3d-models/lunchbox-dc89faa39e2f4f1096630597e4ea6ccb',
            'license': 'CC BY 4.0',
            'metallic': 0.82,
            'roughness': 0.28,
            'translate_m': _DOWNLOADED_BOX_TRANSLATE_M,
            'scale': _DOWNLOADED_BOX_SCALE,
            'visual_only': True,
            'collision_body_unchanged': True,
        }
    pitchers = []
    for name, visual_name in (('019_pitcher_base', 'pitcher_base_vis'), ('pitcher_lid', 'pitcher_lid_vis')):
        path = body_paths[name]
        meshes = render_meshes(stage, path, visual_name)
        texture, uv = source_color_texture(meshes[0])
        exterior = textured_plastic(stage, path + '/Looks/XY5SourceBlue', texture, uv, roughness=.28)
        bind_render_material(stage, meshes, exterior)
        entry = {'body': name, 'source_texture': texture, 'outer_roughness': .28,
                 'meshes': [str(mesh.GetPath()) for mesh in meshes]}
        if name == '019_pitcher_base':
            interior = textured_plastic(stage, path + '/Looks/XY5SourceBlueInterior', texture, uv, roughness=.40)
            entry['inner_roughness'] = .40
            entry['interior_faces'] = {}
            for mesh in meshes:
                selected = _pitcher_interior_faces(mesh)
                # A strong whole-mesh binding would defeat the subset material.
                for target in (mesh.GetParent(), mesh):
                    relation = UsdShade.MaterialBindingAPI(target).GetDirectBinding().GetBindingRel()
                    UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(relation, 'weakerThanDescendants')
                subset = UsdGeom.Subset.CreateGeomSubset(
                    UsdGeom.Imageable(mesh), 'XY5Interior', 'face', selected,
                    familyName='materialBind', familyType='nonOverlapping')
                UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(interior)
                bound, _ = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()
                if bound.GetPath() != interior.GetPath():
                    raise RuntimeError('Pitcher interior material did not take effect')
                entry['interior_faces'][str(mesh.GetPath())] = len(selected)
        pitchers.append(entry)
    after = physics_signature(stage)
    after_materials = physics_material_signature(stage, list(body_paths.values()))
    if before != after or physical_materials != after_materials:
        raise RuntimeError('XY5 finish changed the physical contract')
    return {'profile': PROFILE, 'visual_only': True, 'containers': report, 'pitchers': pitchers,
            'physics_before': before, 'physics_after': after,
            'physics_material_before': physical_materials, 'physics_material_after': after_materials}


def configured_xy5_visual_metadata(environment):
    report = getattr(environment, 'xy5_object_appearance_report', None)
    if not report:
        return {}
    return {'xy5_object_appearance_profile': report['profile'],
            'xy5_visual_lighting_profile': getattr(environment, 'xy_lighting_appearance_report', {}).get('profile', 'source')}


def apply_configured_xy5_object_appearance(environment):
    if (environment.task_name != 'eval_comb01_cover_pitcher_and_box'
            or os.environ.get('HANDSON_XY5_VISUAL_PILOT', '0').strip() != '1'):
        return None
    body_paths = {}
    for name in ('019_pitcher_base', 'pitcher_lid', 'Clear_Box', 'Box_lid'):
        body = environment._find_rigid_body_prim(name)
        if not body:
            raise ValueError('Missing XY5 source body: ' + name)
        body_paths[name] = str(body.GetPath())
    report = apply_xy5_object_appearance(environment._stage, body_paths)
    environment.xy5_object_appearance_report = report
    environment.recording_camera_profile['xy5_appearance'] = configured_xy5_visual_metadata(environment)
    print('[HandsOn/XY5] Native object appearance: ' + str(report), flush=True)
    return report
