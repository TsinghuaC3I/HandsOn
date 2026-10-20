"""Native continuous-rope assets and randomized knot templates for E03."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .surface_deformable_rope import (
    SurfaceDeformableRopeCfg,
    author_surface_deformable_rope,
)


E03_TASK_NAME = "e03_untie_random_rope_knot"
E03_SCENE_XML = "arena_arm_hand_e03_untie_random_rope_knot_rokae_wuji1.xml"
E03_ROPE_ROOT_PATH = "/World/Task/E03Rope"
E03_ROPE_SKIN_MESH_PATH = f"{E03_ROPE_ROOT_PATH}/skinMesh"
E03_ROPE_SIM_MESH_PATH = f"{E03_ROPE_ROOT_PATH}/simMesh"
E03_ROPE_MATERIAL_PATH = "/World/PhysicsMaterials/E03BraidedRope"

E03_SUCCESS_CONDITION_LABELS = (
    "The sampled knot core has been completely opened",
    "The rope is extended with both free ends separated",
    "The rope remains collision-safe and inside the reachable workspace",
    "The untied rope is released and stable for 15 frames",
)

E03_KNOT_NAMES = (
    "left_overhand",
    "right_overhand",
    "figure_eight",
)

E03_ROPE_DIAMETER = 0.010
E03_ROPE_RADIUS = 0.5 * E03_ROPE_DIAMETER
E03_ROPE_SEGMENT_COUNT = 64
E03_ROPE_SEGMENT_SPACING = 0.014
E03_ROPE_CENTERLINE_LENGTH = E03_ROPE_SEGMENT_COUNT * E03_ROPE_SEGMENT_SPACING
E03_ROPE_RADIAL_VERTEX_COUNT = 8
E03_ROPE_RING_COUNT = E03_ROPE_SEGMENT_COUNT + 1
E03_ROPE_NODE_COUNT = E03_ROPE_RING_COUNT * E03_ROPE_RADIAL_VERTEX_COUNT
E03_ROPE_TRIANGLE_COUNT = (
    2 * E03_ROPE_SEGMENT_COUNT * E03_ROPE_RADIAL_VERTEX_COUNT
)

E03_ROPE_LINEAR_DENSITY = 0.060
E03_ROPE_TOTAL_MASS = E03_ROPE_LINEAR_DENSITY * E03_ROPE_CENTERLINE_LENGTH
E03_ROPE_SURFACE_THICKNESS = 0.5 * E03_ROPE_RADIUS
E03_ROPE_MATERIAL_DENSITY = E03_ROPE_TOTAL_MASS / (
    math.pi * E03_ROPE_RADIUS**2 * E03_ROPE_CENTERLINE_LENGTH
)
E03_ROPE_STATIC_FRICTION = 1.15
E03_ROPE_DYNAMIC_FRICTION = 0.95
E03_ROPE_REST_OFFSET = 0.0005
E03_ROPE_CONTACT_OFFSET = 0.0030
E03_ROPE_STRETCH_STIFFNESS = 5.0e4
E03_ROPE_SHEAR_STIFFNESS = 5.0e3
E03_ROPE_BEND_STIFFNESS = 16.0
E03_ROPE_YOUNGS_MODULUS = 5.0e6
E03_ROPE_POISSONS_RATIO = 0.35
E03_ROPE_ELASTICITY_DAMPING = 0.12
E03_ROPE_BEND_DAMPING = 0.08
E03_ROPE_SOLVER_POSITION_ITERATIONS = 64

E03_TABLE_TOP_Z = 0.89
E03_ROPE_BASE_CENTER_XY = np.asarray([-0.16, 0.0], dtype=np.float64)


E03_ROPE_CFG = SurfaceDeformableRopeCfg(
    root_path=E03_ROPE_ROOT_PATH,
    skin_mesh_path=E03_ROPE_SKIN_MESH_PATH,
    simulation_mesh_path=E03_ROPE_SIM_MESH_PATH,
    material_path=E03_ROPE_MATERIAL_PATH,
    diameter=E03_ROPE_DIAMETER,
    centerline_length=E03_ROPE_CENTERLINE_LENGTH,
    centerline_segment_count=E03_ROPE_SEGMENT_COUNT,
    radial_vertex_count=E03_ROPE_RADIAL_VERTEX_COUNT,
    material_density=E03_ROPE_MATERIAL_DENSITY,
    surface_thickness=E03_ROPE_SURFACE_THICKNESS,
    static_friction=E03_ROPE_STATIC_FRICTION,
    dynamic_friction=E03_ROPE_DYNAMIC_FRICTION,
    rest_offset=E03_ROPE_REST_OFFSET,
    contact_offset=E03_ROPE_CONTACT_OFFSET,
    stretch_stiffness=E03_ROPE_STRETCH_STIFFNESS,
    shear_stiffness=E03_ROPE_SHEAR_STIFFNESS,
    bend_stiffness=E03_ROPE_BEND_STIFFNESS,
    youngs_modulus=E03_ROPE_YOUNGS_MODULUS,
    poissons_ratio=E03_ROPE_POISSONS_RATIO,
    elasticity_damping=E03_ROPE_ELASTICITY_DAMPING,
    bend_damping=E03_ROPE_BEND_DAMPING,
    solver_position_iterations=E03_ROPE_SOLVER_POSITION_ITERATIONS,
)


def _resample_polyline(points: np.ndarray, sample_count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    if cumulative[-1] <= 1e-9:
        raise ValueError("Cannot resample a zero-length rope centerline")
    targets = np.linspace(0.0, cumulative[-1], sample_count)
    result = np.empty((sample_count, 3), dtype=np.float64)
    for axis in range(3):
        result[:, axis] = np.interp(targets, cumulative, points[:, axis])
    return result


def _dense_open_knot(knot_index: int, tightness: float) -> np.ndarray:
    """Return an equal-length open trefoil/figure-eight embedding with tails."""
    knot_index = int(knot_index) % len(E03_KNOT_NAMES)
    tightness = float(np.clip(tightness, 0.88, 1.12))
    if knot_index in (0, 1):
        parameter = np.linspace(0.18, 2.0 * math.pi - 0.18, 1600)
        xy_scale = 0.022 * tightness
        z_scale = 0.009 * tightness
        core = np.column_stack(
            [
                xy_scale * (np.sin(parameter) + 2.0 * np.sin(2.0 * parameter)),
                xy_scale * (np.cos(parameter) - 2.0 * np.cos(2.0 * parameter)),
                -z_scale * np.sin(3.0 * parameter),
            ]
        )
        if knot_index == 1:
            core[:, 2] *= -1.0
    else:
        parameter = np.linspace(0.24, 2.0 * math.pi - 0.24, 1600)
        xy_scale = 0.0165 * tightness
        z_scale = 0.009 * tightness
        core = np.column_stack(
            [
                xy_scale
                * (2.0 + np.cos(2.0 * parameter))
                * np.cos(3.0 * parameter),
                xy_scale
                * (2.0 + np.cos(2.0 * parameter))
                * np.sin(3.0 * parameter),
                z_scale * np.sin(4.0 * parameter),
            ]
        )

    core_length = float(np.sum(np.linalg.norm(np.diff(core, axis=0), axis=1)))
    tail_length = 0.5 * (E03_ROPE_CENTERLINE_LENGTH - core_length)
    if tail_length < 0.10:
        raise RuntimeError(
            f"E03 knot template leaves tails too short: {tail_length:.4f}m"
        )
    core_center = np.mean(core, axis=0)
    start_outward = core[0] - core_center
    end_outward = core[-1] - core_center
    # Route each free tail radially away from the knot core in its endpoint's
    # tabletop plane.  Following the parametric tangent would send the tails
    # back through the core and seed a non-physical tube penetration.
    start_outward[2] = 0.0
    end_outward[2] = 0.0
    start_outward /= np.linalg.norm(start_outward)
    end_outward /= np.linalg.norm(end_outward)
    tail_samples = 320
    first_tail = (
        core[0]
        + np.linspace(tail_length, 0.0, tail_samples)[:, None] * start_outward
    )
    second_tail = (
        core[-1]
        + np.linspace(0.0, tail_length, tail_samples)[:, None] * end_outward
    )
    return np.concatenate([first_tail[:-1], core, second_tail[1:]], axis=0)


def e03_knot_centerline_vertices(
    knot_index: int,
    *,
    tightness: float = 1.0,
) -> np.ndarray:
    """Return one tabletop random-knot template in stable vertex order."""
    vertices = _resample_polyline(
        _dense_open_knot(knot_index, tightness),
        E03_ROPE_RING_COUNT,
    )
    vertices[:, :2] += E03_ROPE_BASE_CENTER_XY - np.mean(vertices[:, :2], axis=0)
    vertices[:, 2] += (
        E03_TABLE_TOP_Z
        + E03_ROPE_RADIUS
        + 0.003
        - float(np.min(vertices[:, 2]))
    )
    return vertices


def e03_knot_centerline_centers(
    knot_index: int,
    *,
    tightness: float = 1.0,
) -> np.ndarray:
    vertices = e03_knot_centerline_vertices(knot_index, tightness=tightness)
    return 0.5 * (vertices[:-1] + vertices[1:])


def e03_knot_surface_mesh(
    knot_index: int,
    *,
    tightness: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centerline = e03_knot_centerline_vertices(knot_index, tightness=tightness)
    points = np.empty((E03_ROPE_NODE_COUNT, 3), dtype=np.float64)
    previous_normal: np.ndarray | None = None
    for ring_index, center in enumerate(centerline):
        if ring_index == 0:
            tangent = centerline[1] - centerline[0]
        elif ring_index == E03_ROPE_RING_COUNT - 1:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[ring_index + 1] - centerline[ring_index - 1]
        tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
        reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(tangent, reference))) > 0.92:
            reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        normal = np.cross(tangent, reference)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        if previous_normal is not None and float(np.dot(normal, previous_normal)) < 0.0:
            normal *= -1.0
        binormal = np.cross(tangent, normal)
        binormal /= max(float(np.linalg.norm(binormal)), 1e-12)
        previous_normal = normal
        for radial_index in range(E03_ROPE_RADIAL_VERTEX_COUNT):
            angle = 2.0 * math.pi * radial_index / E03_ROPE_RADIAL_VERTEX_COUNT
            offset = E03_ROPE_RADIUS * (
                math.cos(angle) * normal + math.sin(angle) * binormal
            )
            point_index = ring_index * E03_ROPE_RADIAL_VERTEX_COUNT + radial_index
            points[point_index] = center + offset

    triangles = np.empty((E03_ROPE_TRIANGLE_COUNT, 3), dtype=np.int32)
    triangle_index = 0
    for segment_index in range(E03_ROPE_SEGMENT_COUNT):
        first_ring = segment_index * E03_ROPE_RADIAL_VERTEX_COUNT
        second_ring = (segment_index + 1) * E03_ROPE_RADIAL_VERTEX_COUNT
        for radial_index in range(E03_ROPE_RADIAL_VERTEX_COUNT):
            next_radial = (radial_index + 1) % E03_ROPE_RADIAL_VERTEX_COUNT
            a = first_ring + radial_index
            b = second_ring + radial_index
            c = second_ring + next_radial
            d = first_ring + next_radial
            triangles[triangle_index] = (a, b, c)
            triangles[triangle_index + 1] = (a, c, d)
            triangle_index += 2
    face_counts = np.full(E03_ROPE_TRIANGLE_COUNT, 3, dtype=np.int32)
    return points, face_counts, triangles.reshape(-1)


def _e03_display_colors() -> np.ndarray:
    colors = np.empty((E03_ROPE_NODE_COUNT, 3), dtype=np.float32)
    for ring_index in range(E03_ROPE_RING_COUNT):
        if ring_index < 6:
            color = (0.10, 0.32, 0.90)
        elif ring_index >= E03_ROPE_RING_COUNT - 6:
            color = (0.92, 0.18, 0.10)
        else:
            shade = 0.04 if ring_index % 2 else 0.0
            color = (0.70 + shade, 0.42 + shade, 0.16)
        start = ring_index * E03_ROPE_RADIAL_VERTEX_COUNT
        colors[start : start + E03_ROPE_RADIAL_VERTEX_COUNT] = color
    return colors


def spawn_e03_surface_deformable_rope(stage: Any) -> Path:
    points, face_counts, face_indices = e03_knot_surface_mesh(0)
    return author_surface_deformable_rope(
        stage,
        E03_ROPE_CFG,
        points=points,
        face_vertex_counts=face_counts,
        face_vertex_indices=face_indices,
        display_colors=_e03_display_colors(),
        metadata={
            "task": "random_knot_untying",
            "knotTemplateCount": len(E03_KNOT_NAMES),
        },
    )


__all__ = [
    "E03_KNOT_NAMES",
    "E03_ROPE_CENTERLINE_LENGTH",
    "E03_ROPE_CONTACT_OFFSET",
    "E03_ROPE_DIAMETER",
    "E03_ROPE_NODE_COUNT",
    "E03_ROPE_RADIAL_VERTEX_COUNT",
    "E03_ROPE_RADIUS",
    "E03_ROPE_REST_OFFSET",
    "E03_ROPE_RING_COUNT",
    "E03_ROPE_ROOT_PATH",
    "E03_ROPE_SEGMENT_COUNT",
    "E03_ROPE_TRIANGLE_COUNT",
    "E03_SCENE_XML",
    "E03_SUCCESS_CONDITION_LABELS",
    "E03_TABLE_TOP_Z",
    "E03_TASK_NAME",
    "e03_knot_centerline_centers",
    "e03_knot_centerline_vertices",
    "e03_knot_surface_mesh",
    "spawn_e03_surface_deformable_rope",
]
