"""Closed-loop string geometry and the recorded Cat's Cradle state graph."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from .surface_deformable_rope import (
    SurfaceDeformableRopeCfg,
    author_surface_deformable_rope,
)


E04_TASK_NAME = "e04_cats_cradle_cycle"
E04_SCENE_XML = "arena_arm_hand_e04_cats_cradle_cycle_rokae_wuji1.xml"
E04_STRING_ROOT_PATH = "/World/Task/E04StringLoop"
E04_STRING_SKIN_MESH_PATH = f"{E04_STRING_ROOT_PATH}/skinMesh"
E04_STRING_SIM_MESH_PATH = f"{E04_STRING_ROOT_PATH}/simMesh"
E04_STRING_MATERIAL_PATH = "/World/PhysicsMaterials/E04CatsCradleString"


@dataclass(frozen=True)
class CatsCradleFigureStep:
    index: int
    key: str
    label: str
    operation: str
    terminal: bool = False


@dataclass(frozen=True)
class CatsCradleTransition:
    source: str
    target: str
    operation: str


# The named figures and take-over instructions are recorded explicitly rather
# than hidden in the success predicate.  Clock is the traditional terminating
# branch.  The random collection task samples only E04_CYCLE_TRANSITIONS.
E04_RECORDED_FIGURE_STEPS = (
    CatsCradleFigureStep(
        0,
        "cradle",
        "Cat's Cradle",
        "Form the opening loop across both hands, then let each middle finger pick up the opposite palm string.",
    ),
    CatsCradleFigureStep(
        1,
        "soldiers_bed",
        "Soldier's Bed",
        "Grasp the crossed strings from above, pick up the near and far straight strings from below, then lift the figure off.",
    ),
    CatsCradleFigureStep(
        2,
        "candles",
        "Candles",
        "Enter the narrow loops from above, grasp the crossed strings, then pick up both straight strings from below.",
    ),
    CatsCradleFigureStep(
        3,
        "manger",
        "Manger",
        "Use the little fingers to take the far thumb and near forefinger strings, then pick up both straight strings from below.",
    ),
    CatsCradleFigureStep(
        4,
        "diamonds",
        "Diamonds",
        "Grasp the two crossed strings from below, pass over the side strings, and lift the near and far strings from above.",
    ),
    CatsCradleFigureStep(
        5,
        "cats_eye",
        "Cat's Eye",
        "Repeat the Candles take-over: enter the narrow loops, take the crossings, and lift the straight strings.",
    ),
    CatsCradleFigureStep(
        6,
        "fish_in_a_dish",
        "Fish in a Dish",
        "Grasp the crossed strings from above, pick up the diagonal strings, then spread and lift the loop free.",
    ),
    CatsCradleFigureStep(
        7,
        "clock",
        "Clock",
        "Separate the center strings with the little fingers, take the crossings and diagonals, then lift vertically.",
        terminal=True,
    ),
)

E04_CYCLE_TRANSITIONS = (
    CatsCradleTransition(
        "cradle",
        "soldiers_bed",
        E04_RECORDED_FIGURE_STEPS[1].operation,
    ),
    CatsCradleTransition(
        "soldiers_bed",
        "candles",
        E04_RECORDED_FIGURE_STEPS[2].operation,
    ),
    CatsCradleTransition(
        "candles",
        "manger",
        E04_RECORDED_FIGURE_STEPS[3].operation,
    ),
    CatsCradleTransition(
        "manger",
        "diamonds",
        E04_RECORDED_FIGURE_STEPS[4].operation,
    ),
    CatsCradleTransition(
        "diamonds",
        "cats_eye",
        E04_RECORDED_FIGURE_STEPS[5].operation,
    ),
    CatsCradleTransition(
        "cats_eye",
        "fish_in_a_dish",
        E04_RECORDED_FIGURE_STEPS[6].operation,
    ),
    CatsCradleTransition(
        "fish_in_a_dish",
        "soldiers_bed",
        "Take the crossings from above, move down between the center strings, spread both hands, and lift off into the inverted Soldier's Bed.",
    ),
)

E04_TERMINAL_TRANSITION = CatsCradleTransition(
    "fish_in_a_dish",
    "clock",
    E04_RECORDED_FIGURE_STEPS[7].operation,
)

E04_FIGURE_KEYS = tuple(step.key for step in E04_RECORDED_FIGURE_STEPS)
E04_FIGURE_LABELS = tuple(step.label for step in E04_RECORDED_FIGURE_STEPS)
E04_FIGURE_INDEX = {key: index for index, key in enumerate(E04_FIGURE_KEYS)}

E04_SUCCESS_CONDITION_LABELS = (
    "The sampled source figure has been changed",
    "The requested next Cat's Cradle figure is matched",
    "Both hands support the closed loop inside the reachable workspace",
    "The requested figure remains stable for 15 frames",
)

E04_STRING_DIAMETER = 0.004
E04_STRING_RADIUS = 0.5 * E04_STRING_DIAMETER
E04_STRING_SEGMENT_COUNT = 96
E04_STRING_RING_COUNT = E04_STRING_SEGMENT_COUNT
E04_STRING_CENTERLINE_LENGTH = 1.20
E04_STRING_RADIAL_VERTEX_COUNT = 6
E04_STRING_NODE_COUNT = E04_STRING_RING_COUNT * E04_STRING_RADIAL_VERTEX_COUNT
E04_STRING_TRIANGLE_COUNT = (
    2 * E04_STRING_SEGMENT_COUNT * E04_STRING_RADIAL_VERTEX_COUNT
)
E04_STRING_LINEAR_DENSITY = 0.020
E04_STRING_TOTAL_MASS = E04_STRING_LINEAR_DENSITY * E04_STRING_CENTERLINE_LENGTH
E04_STRING_SURFACE_THICKNESS = 0.5 * E04_STRING_RADIUS
E04_STRING_MATERIAL_DENSITY = E04_STRING_TOTAL_MASS / (
    math.pi * E04_STRING_RADIUS**2 * E04_STRING_CENTERLINE_LENGTH
)
E04_STRING_STATIC_FRICTION = 1.05
E04_STRING_DYNAMIC_FRICTION = 0.86
E04_STRING_REST_OFFSET = 0.0002
E04_STRING_CONTACT_OFFSET = 0.0015
E04_STRING_STRETCH_STIFFNESS = 3.0e4
E04_STRING_SHEAR_STIFFNESS = 3.0e3
E04_STRING_BEND_STIFFNESS = 5.0
E04_STRING_YOUNGS_MODULUS = 3.0e6
E04_STRING_POISSONS_RATIO = 0.35
E04_STRING_ELASTICITY_DAMPING = 0.15
E04_STRING_BEND_DAMPING = 0.10
E04_TABLE_TOP_Z = 0.89
E04_STRING_BASE_CENTER_XY = np.asarray([-0.16, 0.0], dtype=np.float64)


E04_STRING_CFG = SurfaceDeformableRopeCfg(
    root_path=E04_STRING_ROOT_PATH,
    skin_mesh_path=E04_STRING_SKIN_MESH_PATH,
    simulation_mesh_path=E04_STRING_SIM_MESH_PATH,
    material_path=E04_STRING_MATERIAL_PATH,
    diameter=E04_STRING_DIAMETER,
    centerline_length=E04_STRING_CENTERLINE_LENGTH,
    centerline_segment_count=E04_STRING_SEGMENT_COUNT,
    radial_vertex_count=E04_STRING_RADIAL_VERTEX_COUNT,
    material_density=E04_STRING_MATERIAL_DENSITY,
    surface_thickness=E04_STRING_SURFACE_THICKNESS,
    static_friction=E04_STRING_STATIC_FRICTION,
    dynamic_friction=E04_STRING_DYNAMIC_FRICTION,
    rest_offset=E04_STRING_REST_OFFSET,
    contact_offset=E04_STRING_CONTACT_OFFSET,
    stretch_stiffness=E04_STRING_STRETCH_STIFFNESS,
    shear_stiffness=E04_STRING_SHEAR_STIFFNESS,
    bend_stiffness=E04_STRING_BEND_STIFFNESS,
    youngs_modulus=E04_STRING_YOUNGS_MODULUS,
    poissons_ratio=E04_STRING_POISSONS_RATIO,
    elasticity_damping=E04_STRING_ELASTICITY_DAMPING,
    bend_damping=E04_STRING_BEND_DAMPING,
    solver_position_iterations=64,
)


def _resample_closed_curve(points: np.ndarray, sample_count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    closed = np.concatenate([points, points[:1]], axis=0)
    distances = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    targets = np.linspace(0.0, cumulative[-1], sample_count, endpoint=False)
    return np.column_stack(
        [np.interp(targets, cumulative, closed[:, axis]) for axis in range(3)]
    )


def _raw_figure_curve(figure_index: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, 4096, endpoint=False)
    index = int(figure_index) % len(E04_RECORDED_FIGURE_STEPS)
    if index == 0:  # Cat's Cradle
        x = np.sin(2 * t) + 0.14 * np.sin(6 * t)
        y = 0.78 * np.sin(3 * t)
        z = 0.11 * np.cos(5 * t + 0.2)
    elif index == 1:  # Soldier's Bed
        x = 0.92 * np.sin(3 * t)
        y = 0.84 * np.sin(2 * t) + 0.12 * np.sin(6 * t)
        z = 0.11 * np.sin(5 * t + 0.4)
    elif index == 2:  # Candles
        x = 0.62 * np.sin(t) + 0.36 * np.sin(3 * t)
        y = np.sin(2 * t)
        z = 0.12 * np.cos(3 * t + 0.3)
    elif index == 3:  # Manger (inverted Cradle)
        x = np.sin(2 * t) - 0.18 * np.sin(6 * t)
        y = -0.80 * np.sin(3 * t)
        z = -0.11 * np.cos(5 * t - 0.2)
    elif index == 4:  # Diamonds
        x = 0.88 * np.sin(4 * t)
        y = 0.85 * np.sin(3 * t)
        # Fourteen projected crossings need a larger layer separation than
        # the simpler figures so the 4 mm tube never starts interpenetrating.
        z = 0.40 * np.cos(7 * t + 0.25)
    elif index == 5:  # Cat's Eye
        x = np.sin(2 * t)
        y = 0.72 * np.sin(t) + 0.28 * np.sin(5 * t)
        z = 0.12 * np.sin(3 * t + 0.5)
    elif index == 6:  # Fish in a Dish
        x = 0.95 * np.sin(t) + 0.22 * np.sin(3 * t)
        y = 0.72 * np.sin(2 * t) + 0.18 * np.sin(4 * t)
        z = 0.12 * np.cos(5 * t - 0.35)
    else:  # Clock, terminal branch
        x = 0.65 * np.sin(t) + 0.25 * np.sin(5 * t)
        y = np.sin(2 * t)
        z = 0.12 * np.cos(3 * t - 0.25)
    return np.column_stack([x, y, z])


def e04_figure_centerline(figure_index: int) -> np.ndarray:
    """Return one collision-layered, equal-arclength closed string figure."""
    raw = _raw_figure_curve(figure_index)
    closed = np.concatenate([raw, raw[:1]], axis=0)
    raw_length = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
    raw *= E04_STRING_CENTERLINE_LENGTH / raw_length
    points = _resample_closed_curve(raw, E04_STRING_RING_COUNT)
    points[:, :2] += E04_STRING_BASE_CENTER_XY - np.mean(points[:, :2], axis=0)
    points[:, 2] += (
        E04_TABLE_TOP_Z
        + E04_STRING_RADIUS
        + 0.004
        - float(np.min(points[:, 2]))
    )
    return points


def e04_string_surface_mesh(
    figure_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centerline = e04_figure_centerline(figure_index)
    points = np.empty((E04_STRING_NODE_COUNT, 3), dtype=np.float64)
    previous_normal: np.ndarray | None = None
    for ring_index, center in enumerate(centerline):
        previous_center = centerline[(ring_index - 1) % E04_STRING_RING_COUNT]
        next_center = centerline[(ring_index + 1) % E04_STRING_RING_COUNT]
        tangent = next_center - previous_center
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
        for radial_index in range(E04_STRING_RADIAL_VERTEX_COUNT):
            angle = 2.0 * math.pi * radial_index / E04_STRING_RADIAL_VERTEX_COUNT
            offset = E04_STRING_RADIUS * (
                math.cos(angle) * normal + math.sin(angle) * binormal
            )
            point_index = ring_index * E04_STRING_RADIAL_VERTEX_COUNT + radial_index
            points[point_index] = center + offset

    triangles = np.empty((E04_STRING_TRIANGLE_COUNT, 3), dtype=np.int32)
    triangle_index = 0
    for segment_index in range(E04_STRING_SEGMENT_COUNT):
        first_ring = segment_index * E04_STRING_RADIAL_VERTEX_COUNT
        second_ring = (
            (segment_index + 1) % E04_STRING_RING_COUNT
        ) * E04_STRING_RADIAL_VERTEX_COUNT
        for radial_index in range(E04_STRING_RADIAL_VERTEX_COUNT):
            next_radial = (radial_index + 1) % E04_STRING_RADIAL_VERTEX_COUNT
            a = first_ring + radial_index
            b = second_ring + radial_index
            c = second_ring + next_radial
            d = first_ring + next_radial
            triangles[triangle_index] = (a, b, c)
            triangles[triangle_index + 1] = (a, c, d)
            triangle_index += 2
    face_counts = np.full(E04_STRING_TRIANGLE_COUNT, 3, dtype=np.int32)
    return points, face_counts, triangles.reshape(-1)


def _e04_display_colors() -> np.ndarray:
    colors = np.empty((E04_STRING_NODE_COUNT, 3), dtype=np.float32)
    palette = ((0.14, 0.48, 0.92), (0.95, 0.35, 0.16), (0.92, 0.78, 0.14))
    for ring_index in range(E04_STRING_RING_COUNT):
        color = palette[(3 * ring_index // E04_STRING_RING_COUNT) % len(palette)]
        start = ring_index * E04_STRING_RADIAL_VERTEX_COUNT
        colors[start : start + E04_STRING_RADIAL_VERTEX_COUNT] = color
    return colors


def spawn_e04_surface_deformable_string(stage: Any) -> Path:
    points, face_counts, face_indices = e04_string_surface_mesh(0)
    return author_surface_deformable_rope(
        stage,
        E04_STRING_CFG,
        points=points,
        face_vertex_counts=face_counts,
        face_vertex_indices=face_indices,
        display_colors=_e04_display_colors(),
        metadata={
            "task": "cats_cradle_cycle",
            "closedLoop": True,
            "recordedFigureCount": len(E04_RECORDED_FIGURE_STEPS),
            "randomCycleTransitionCount": len(E04_CYCLE_TRANSITIONS),
        },
    )


__all__ = [
    "CatsCradleFigureStep",
    "CatsCradleTransition",
    "E04_CYCLE_TRANSITIONS",
    "E04_FIGURE_INDEX",
    "E04_FIGURE_KEYS",
    "E04_FIGURE_LABELS",
    "E04_RECORDED_FIGURE_STEPS",
    "E04_SCENE_XML",
    "E04_STRING_CENTERLINE_LENGTH",
    "E04_STRING_CONTACT_OFFSET",
    "E04_STRING_DIAMETER",
    "E04_STRING_NODE_COUNT",
    "E04_STRING_RADIAL_VERTEX_COUNT",
    "E04_STRING_RADIUS",
    "E04_STRING_REST_OFFSET",
    "E04_STRING_RING_COUNT",
    "E04_STRING_ROOT_PATH",
    "E04_STRING_SEGMENT_COUNT",
    "E04_STRING_TRIANGLE_COUNT",
    "E04_SUCCESS_CONDITION_LABELS",
    "E04_TABLE_TOP_Z",
    "E04_TASK_NAME",
    "E04_TERMINAL_TRANSITION",
    "e04_figure_centerline",
    "e04_string_surface_mesh",
    "spawn_e04_surface_deformable_string",
]
