"""Derive convex enclosure pieces while preserving source fader-slot cavities.

Only collision geometry is clipped; all visible source vertices are unchanged.
Conservative shared slot bounds avoid narrow numerical collision fragments.
"""
import numpy as np
import shapely
import trimesh
from shapely.geometry import Point, Polygon


def clip(mesh, normal, offset):
    """Keep the convex portion satisfying dot(vertex, normal) >= offset."""
    normal = np.asarray(normal)
    vertices = mesh.vertices
    distances = vertices @ normal - offset
    if distances.min() >= -1e-9:
        return mesh
    if distances.max() <= 1e-9:
        return None
    edges = mesh.edges_unique
    crossing = edges[(distances[edges[:, 0]] * distances[edges[:, 1]]) < -1e-18]
    first, second = distances[crossing[:, 0]], distances[crossing[:, 1]]
    intersections = vertices[crossing[:, 0]] + (
        vertices[crossing[:, 1]] - vertices[crossing[:, 0]]
    ) * (first / (first - second))[:, None]
    points = np.vstack([vertices[distances >= -1e-9], intersections])
    # Micrometer quantization removes numerical slivers before float32 USD export.
    points = np.unique(np.round(points, 6), axis=0)
    if len(points) < 4:
        return None
    result = trimesh.convex.convex_hull(points)
    return result if result.volume > 1e-12 else None


def split_pocket(mesh, bounds, floor_normal, floor_offset):
    """Retain the solid outside a source pocket and below its measured floor."""
    xmin, ymin, xmax, ymax = bounds
    if (mesh.bounds[1, 0] <= xmin or mesh.bounds[0, 0] >= xmax
            or mesh.bounds[1, 1] <= ymin or mesh.bounds[0, 1] >= ymax):
        return [mesh]
    retained = []
    remaining = mesh
    planes = [([1, 0, 0], xmin), ([-1, 0, 0], -xmax),
              ([0, 1, 0], ymin), ([0, -1, 0], -ymax)]
    for normal, offset in planes:
        outside = clip(remaining, -np.asarray(normal), -offset)
        if outside is not None:
            retained.append(outside)
        remaining = clip(remaining, normal, offset)
        if remaining is None:
            return retained
    bottom = clip(remaining, floor_normal, floor_offset)
    if bottom is not None:
        retained.append(bottom)
    return retained


def derive_panel_colliders(mesh, limited_hull):
    sample_xy = np.array([[0, .135], [0, -.135], [.05, 0], [-.05, 0]])
    points = []
    for x, y in sample_xy:
        hits, _, _ = mesh.ray.intersects_location([[x, y, 1]], [[0, 0, -1]])
        points.append([x, y, hits[:, 2].max()])
    points = np.asarray(points)
    design = np.column_stack([points[:, :2], np.ones(4)])
    coefficients = np.linalg.lstsq(design, points[:, 2], rcond=None)[0]
    assert np.max(np.abs(points[:, 2] - design @ coefficients)) < .0002

    triangles = mesh.triangles
    projected = np.column_stack([
        triangles[:, :, :2].reshape(-1, 2), np.ones(len(triangles) * 3)
    ]).reshape(-1, 3, 3)
    deviation = triangles[:, :, 2] - projected @ coefficients
    on_deck = ((mesh.face_normals[:, 2] > .9)
               & (np.max(abs(deviation), axis=1) < .00005))
    deck_polygon = shapely.union_all(
        [Polygon(triangle[:, :2]) for triangle in triangles[on_deck]],
        grid_size=.000001,
    )
    holes = [
        Polygon(hole)
        for polygon in getattr(deck_polygon, 'geoms', [deck_polygon])
        for hole in polygon.interiors if Polygon(hole).area > 1e-7
    ]
    records = []
    for hole in holes:
        bounds = list(hole.bounds)
        xmin, ymin, xmax, ymax = bounds
        xy = np.array([
            [x, y] for x in np.linspace(xmin + .00015, xmax - .00015, 7)
            for y in np.linspace(ymin + .00015, ymax - .00015, 7)
            if hole.contains(Point(x, y))
        ])
        hits, ray_ids, _ = mesh.ray.intersects_location(
            np.column_stack([xy, np.ones(len(xy))]),
            np.tile([0, 0, -1], (len(xy), 1)),
        )
        top = np.full(len(xy), -np.inf)
        np.maximum.at(top, ray_ids, hits[:, 2])
        assert np.isfinite(top).all()
        delta = top - np.column_stack([xy, np.ones(len(xy))]) @ coefficients
        if np.min(delta) > -.01:
            # The shallow display recess follows the original deck inclination.
            normal = [*coefficients[:2], -1.]
            offset = -coefficients[2] - float(delta.min()) + .00001
        else:
            # Deeper fader slots remain open down to the measured cavity floor.
            normal = [0, 0, -1.]
            offset = -float(top.min()) + .00001
        records.append({'bounds': bounds, 'floor_normal': normal,
                        'floor_offset': offset, 'delta_min': float(delta.min())})

    # Both measured banks have nearly equal longitudinal slot bounds. Use a
    # conservative common span to avoid submillimeter fragments between them.
    for center in [-.11, .06]:
        slots = [record for record in records if abs(
            (record['bounds'][0] + record['bounds'][2]) / 2 - center) < .02]
        if slots:
            xmin = min(record['bounds'][0] for record in slots)
            xmax = max(record['bounds'][2] for record in slots)
            for record in slots:
                record['bounds'][0], record['bounds'][2] = xmin, xmax

    vertices = mesh.vertices
    deck_z = np.column_stack([vertices[:, :2], np.ones(len(vertices))]) @ coefficients
    core = limited_hull(trimesh.Trimesh(
        vertices=vertices[vertices[:, 2] <= deck_z + .00005], faces=[], process=False))
    parts = [core]
    for record in records:
        parts = [piece for old in parts for piece in split_pocket(
            old, record['bounds'], record['floor_normal'], record['floor_offset'])]
    rails = [vertices[:, 1] >= .19, vertices[:, 1] <= -.19,
             vertices[:, 0] >= .11, vertices[:, 0] <= -.16]
    parts += [limited_hull(trimesh.Trimesh(
        vertices=vertices[mask], faces=[], process=False)) for mask in rails]
    return parts, records
