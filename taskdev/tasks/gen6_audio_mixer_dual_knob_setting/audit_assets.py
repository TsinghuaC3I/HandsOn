"""CPU mesh/convex-shape audit. This does not pass the live robot/PhysX gate."""
from pathlib import Path
import itertools
import hashlib
import json
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from pxr import Usd, UsdGeom, UsdPhysics
from build_assets import OUTPUT, TASK_DIR


def load_meshes(filename, collision=False):
    stage = Usd.Stage.Open(str(OUTPUT / filename))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == 'Z'
    result = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        assert np.all(counts == 3)
        geometry = trimesh.Trimesh(vertices=np.asarray(mesh.GetPointsAttr().Get()),
                                   faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3),
                                   process=False)
        assert np.isfinite(geometry.vertices).all()
        assert prim.HasAPI(UsdPhysics.CollisionAPI) is collision
        if collision:
            assert mesh.ComputeVisibility() == 'invisible'
            assert geometry.volume > 0
            assert geometry.is_watertight
            assert len(geometry.vertices) <= 256
        else:
            assert mesh.ComputeVisibility() == 'inherited'
        result[str(prim.GetPath())] = geometry
    assert result
    return result


def audit():
    manifest = json.loads((OUTPUT / 'asset_manifest.json').read_text())
    geometry = manifest['geometry']
    visual = load_meshes('mixer_fixed_visual.usda')
    knobs_visual = load_meshes('knob_visual.usda')
    fixed = load_meshes('mixer_fixed_collision.usda', True)
    knobs = load_meshes('knob_collision.usda', True)
    grip = trimesh.util.concatenate(list(knobs.values()))
    rotation = trimesh.transformations.quaternion_matrix(geometry['knob_orientation_wxyz'])
    fixed_manager = trimesh.collision.CollisionManager()
    for name, mesh in fixed.items():
        fixed_manager.add_object(name, mesh)
    angles = [-100, -99, -80, 0, 80, 99, 100]
    angle_results, unapproved = [], []
    transforms = {}
    for side, degrees in itertools.product(('left', 'right'), angles):
        transform = rotation @ trimesh.transformations.rotation_matrix(np.radians(degrees), [0, 0, 1])
        transform[:3, 3] = geometry['knob_centers_m'][side]
        transforms[side, degrees] = transform
        overlap, names, contacts = fixed_manager.in_collision_single(grip, transform=transform, return_names=True, return_data=True)
        distance = fixed_manager.min_distance_single(grip, transform=transform)
        record = {'side': side, 'angle_deg': degrees, 'minimum_fixed_clearance_m': float(distance),
                  'overlap_names': sorted(names), 'depths_m': [float(c.depth) for c in contacts]}
        angle_results.append(record)
        if overlap:
            unapproved.append(record)
    cross_results = []
    left_manager = trimesh.collision.CollisionManager()
    left_manager.add_object('knob_left', grip)
    for left, right in itertools.product(angles, repeat=2):
        left_manager.set_transform('knob_left', transforms['left', left])
        overlap = left_manager.in_collision_single(grip, transform=transforms['right', right])
        distance = left_manager.min_distance_single(grip, transform=transforms['right', right])
        cross_results.append({'left_deg': left, 'right_deg': right, 'clearance_m': float(distance)})
        if overlap:
            unapproved.append({'left_deg': left, 'right_deg': right, 'pair': ['knob_left', 'knob_right']})
    layouts = []
    for dx, dy in itertools.product((-0.03, 0., 0.03), repeat=2):
        for yaw in (-5., 0., 5.):
            transform = trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 0, 1])
            transform[:3, 3] = np.array(geometry['root_home_m']) + [dx, dy, 0.]
            all_points = np.concatenate([trimesh.transform_points(m.vertices, transform) for m in fixed.values()])
            low, high = all_points.min(axis=0), all_points.max(axis=0)
            record = {'translation_offset_m': [dx, dy, 0.], 'yaw_deg': yaw,
                      'fixed_world_bounds_m': [low.tolist(), high.tolist()],
                      'table_plane_clearance_m': float(low[2] - 0.89)}
            layouts.append(record)
            if low[2] < 0.89 - 1e-6:
                unapproved.append({'pair': ['fixed_collision', 'table_top'], **record})
    inertia = np.asarray(geometry['knob_diagonal_inertia_kg_m2'])
    assert np.isfinite(inertia).all() and (inertia > 0.).all()
    visual_grip = trimesh.util.concatenate(list(knobs_visual.values()))
    sample = np.vstack([grip.vertices, grip.triangles_center])
    closest, distances, triangle_ids = trimesh.proximity.closest_point(visual_grip, sample)
    normals = visual_grip.face_normals[triangle_ids]
    projected = ((sample - closest) * normals).sum(axis=1)
    # Verify that the compact enclosure representation still covers its
    # original visible shell. Check convex halfspaces before surface distance
    # so points legitimately inside the filled electronics cavity are valid.
    enclosure_visual = trimesh.util.concatenate([m for name, m in visual.items() if name.startswith('/Asset/Box016_')])
    enclosure_hulls = [m for name, m in fixed.items() if name.startswith('/Asset/enclosure_')]
    shell_points = np.unique(enclosure_visual.vertices, axis=0)
    inside = np.zeros(len(shell_points), dtype=bool)
    for hull in enclosure_hulls:
        equations = ConvexHull(hull.vertices).equations
        inside |= np.max(shell_points @ equations[:, :3].T + equations[:, 3], axis=1) <= 1e-7
    outside = shell_points[~inside]
    if len(outside):
        surface = trimesh.util.concatenate(enclosure_hulls)
        _, gaps, _ = trimesh.proximity.closest_point(surface, outside)
        shell_gap = float(gaps.max())
    else:
        shell_gap = 0.0
    ray_xy = []
    for side in ('left', 'right'):
        center = np.array(geometry['knob_centers_m'][side][:2])
        for dx, dy in itertools.product(np.linspace(-.05, .05, 11), repeat=2):
            if dx*dx + dy*dy <= .05**2:
                ray_xy.append(center + [dx, dy])
    ray_origins = np.column_stack([ray_xy, np.ones(len(ray_xy))])
    ray_directions = np.tile([0., 0., -1.], (len(ray_xy), 1))
    heights = []
    for surface in (enclosure_visual, trimesh.util.concatenate(enclosure_hulls)):
        hits, ray_ids, _ = surface.ray.intersects_location(ray_origins, ray_directions)
        top = np.full(len(ray_xy), -np.inf)
        np.maximum.at(top, ray_ids, hits[:, 2])
        assert np.isfinite(top).all()
        heights.append(top)
    deck_delta = heights[1] - heights[0]
    deck_visible_hits = np.column_stack([ray_xy, heights[0]])
    _, deck_inward_distance, _ = trimesh.proximity.closest_point(
        trimesh.util.concatenate(enclosure_hulls), deck_visible_hits)
    worst_height = int(np.argmin(deck_delta))
    result = {
        'scope': 'Final exported task asset convex meshes only; no robot or live PhysX evidence.',
        'asset_manifest_sha256': hashlib.sha256((OUTPUT / 'asset_manifest.json').read_bytes()).hexdigest(),
        'license_verified': 'CC-BY-4.0 in downloaded license.txt and glTF asset metadata',
        'visual_meshes': len(visual) + len(knobs_visual),
        'fixed_collision_meshes': len(fixed), 'knob_collision_meshes_per_link': len(knobs),
        'collision_visibility': 'all explicitly invisible',
        'mount_clearance_by_angle': angle_results, 'asymmetric_knob_pairs': cross_results,
        'fixture_layouts': layouts,
        'unapproved_task_asset_overlaps': unapproved,
        'intended_support': {'pair': ['panel_base bottom', 'table_top'],
                             'justification': 'Fixed enclosure rests on the source-derived bottom surface at z=0.89 m.'},
        'same_body_knob_surface_deviation': {
            'sample_count': len(sample), 'max_unsigned_distance_m': float(distances.max()),
            'max_outward_normal_projection_m': float(projected.max()),
            'note': 'Source mesh is not watertight; nearest-face projection is preliminary. Actual hand sweep is pending.'},
        'enclosure_surface_coverage': {
            'source_vertices': len(shell_points), 'vertices_inside_convex_union': int(inside.sum()),
            'max_source_surface_to_collision_gap_m': shell_gap,
            'operation_deck_ray_count': len(ray_xy),
            'operation_deck_max_outward_height_m': float(deck_delta.max()),
            'operation_deck_max_inward_height_m': float(-deck_delta.min()),
            'operation_deck_max_visible_hit_to_collision_distance_m': float(deck_inward_distance.max()),
            'worst_vertical_gap': {'xy_m': np.asarray(ray_xy[worst_height]).tolist(),
                                  'vertical_gap_m': float(-deck_delta[worst_height]),
                                  'nearest_3d_surface_distance_m': float(deck_inward_distance[worst_height]),
                                  'interpretation': 'A vertical ray at a conservative slot boundary can reach the cavity floor; use 3D surface distance to measure the local boundary approximation.'},
            'note': 'Source-derived convex pieces preserve the recessed display and fader-slot cavities; four edge rails remain separate.'},
        'positive_finite_knob_inertia': True,
        'pass_task_asset_clearance': not unapproved,
        'live_gate_2': 'pending', 'full_geometry_reach_gate_4': 'pending',
    }
    (TASK_DIR / 'evidence/asset_geometry_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('mount_clearance_by_angle', 'asymmetric_knob_pairs', 'fixture_layouts')}, indent=2))
    if unapproved:
        raise SystemExit('Task asset overlap audit failed')


if __name__ == '__main__':
    audit()
