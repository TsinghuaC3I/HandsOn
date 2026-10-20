"""Shared analytic collision contract for the four HS005 kitchen scenes.

The downloaded kitchen meshes remain visual assets.  Every scene layer owns
the same collision topology: one basin floor, four full-height basin walls,
and four countertop slabs that leave the measured sink opening uncovered.
Keeping this contract in the installed simulator package makes it available to
every task that references the scene instead of relying on taskdev at runtime.
"""

from __future__ import annotations


MODERN_APARTMENT_SCENE = "hs005_modern_apartment_sink"
MODERN_KITCHEN_SCENE = "hs005_modern_kitchen_sink"
REALISTIC_KITCHEN_SCENE = "hs005_realistic_kitchen_scene_sink"
APPARTEMENT_KITCHEN_SCENE = "hs005_appartement_kitchen_sink"

SCENE_COLLISION_ROOT_NAME = "SceneCollision"
SCENE_COLLISION_POLICY = "full_height_basin_and_open_countertop_ring_v1"
SCENE_COLLISION_CONTACT_OFFSET = 0.003
HAND_COLLISION_CONTACT_OFFSET = 0.003

COLLISION_GEOMETRY_BY_SCENE = {
    MODERN_APARTMENT_SCENE: {
        "basin_bounds": (-0.340, 0.040, -0.200, 0.200),
        "floor_top_z": 0.895,
        "floor_box": ((-0.150, 0.000, 0.885), (0.420, 0.720, 0.020)),
        "rim_top_z": 1.01367905,
        "countertop_z": 1.011315,
    },
    MODERN_KITCHEN_SCENE: {
        "basin_bounds": (-0.360, 0.060, -0.360, 0.360),
        "floor_top_z": 0.71930,
        "floor_box": ((-0.150, 0.000, 0.70930), (0.420, 0.720, 0.020)),
        "rim_top_z": 0.99375,
        "countertop_z": 0.97580,
    },
    REALISTIC_KITCHEN_SCENE: {
        "basin_bounds": (-0.300, 0.000, -0.380, 0.380),
        "floor_top_z": 0.64815,
        "floor_box": ((-0.150, 0.000, 0.63815), (0.280, 0.720, 0.020)),
        "rim_top_z": 0.80429,
        "countertop_z": 0.80429,
    },
    APPARTEMENT_KITCHEN_SCENE: {
        "basin_bounds": (-0.320, 0.080, -0.275, 0.275),
        "floor_top_z": 0.75207,
        "floor_box": ((-0.120, 0.000, 0.74207), (0.400, 0.550, 0.020)),
        "rim_top_z": 0.90000,
        "countertop_z": 0.90000,
    },
}

_WALL_THICKNESS = 0.030
_COUNTERTOP_THICKNESS = 0.050
_COUNTERTOP_X_MARGIN = 0.320
_COUNTERTOP_Y_MARGIN = 0.280


CollisionBox = tuple[
    str,
    tuple[float, float, float],
    tuple[float, float, float],
]


def collision_repair_boxes(scene_id: str) -> tuple[CollisionBox, ...]:
    """Return root-local basin walls and an open countertop collision ring."""

    scene_id = scene_id or MODERN_APARTMENT_SCENE
    try:
        geometry = COLLISION_GEOMETRY_BY_SCENE[scene_id]
    except KeyError as exc:
        raise ValueError(f"unsupported HS005 kitchen scene: {scene_id!r}") from exc

    x_min, x_max, y_min, y_max = geometry["basin_bounds"]
    floor_top_z = float(geometry["floor_top_z"])
    rim_top_z = float(geometry["rim_top_z"])
    countertop_z = float(geometry["countertop_z"])
    wall_height = rim_top_z - floor_top_z
    if wall_height <= 0.0:
        raise ValueError(f"invalid HS005 basin wall height: {geometry}")
    wall_center_z = floor_top_z + 0.5 * wall_height

    outer_x_min = x_min - _COUNTERTOP_X_MARGIN
    outer_x_max = x_max + _COUNTERTOP_X_MARGIN
    outer_y_min = y_min - _COUNTERTOP_Y_MARGIN
    outer_y_max = y_max + _COUNTERTOP_Y_MARGIN
    counter_center_z = countertop_z - 0.5 * _COUNTERTOP_THICKNESS

    return (
        (
            "front_wall",
            (
                x_min - 0.5 * _WALL_THICKNESS,
                0.5 * (y_min + y_max),
                wall_center_z,
            ),
            (
                _WALL_THICKNESS,
                y_max - y_min + 2.0 * _WALL_THICKNESS,
                wall_height,
            ),
        ),
        (
            "back_wall",
            (
                x_max + 0.5 * _WALL_THICKNESS,
                0.5 * (y_min + y_max),
                wall_center_z,
            ),
            (
                _WALL_THICKNESS,
                y_max - y_min + 2.0 * _WALL_THICKNESS,
                wall_height,
            ),
        ),
        (
            "left_wall",
            (
                0.5 * (x_min + x_max),
                y_min - 0.5 * _WALL_THICKNESS,
                wall_center_z,
            ),
            (x_max - x_min, _WALL_THICKNESS, wall_height),
        ),
        (
            "right_wall",
            (
                0.5 * (x_min + x_max),
                y_max + 0.5 * _WALL_THICKNESS,
                wall_center_z,
            ),
            (x_max - x_min, _WALL_THICKNESS, wall_height),
        ),
        (
            "front_countertop",
            (
                0.5 * (outer_x_min + x_min),
                0.5 * (outer_y_min + outer_y_max),
                counter_center_z,
            ),
            (
                x_min - outer_x_min,
                outer_y_max - outer_y_min,
                _COUNTERTOP_THICKNESS,
            ),
        ),
        (
            "back_countertop",
            (
                0.5 * (x_max + outer_x_max),
                0.5 * (outer_y_min + outer_y_max),
                counter_center_z,
            ),
            (
                outer_x_max - x_max,
                outer_y_max - outer_y_min,
                _COUNTERTOP_THICKNESS,
            ),
        ),
        (
            "left_countertop",
            (
                0.5 * (x_min + x_max),
                0.5 * (outer_y_min + y_min),
                counter_center_z,
            ),
            (
                x_max - x_min,
                y_min - outer_y_min,
                _COUNTERTOP_THICKNESS,
            ),
        ),
        (
            "right_countertop",
            (
                0.5 * (x_min + x_max),
                0.5 * (y_max + outer_y_max),
                counter_center_z,
            ),
            (
                x_max - x_min,
                outer_y_max - y_max,
                _COUNTERTOP_THICKNESS,
            ),
        ),
    )


def complete_scene_collision_boxes(scene_id: str) -> tuple[CollisionBox, ...]:
    """Return the floor plus every wall and countertop collider for a scene."""

    scene_id = scene_id or MODERN_APARTMENT_SCENE
    geometry = COLLISION_GEOMETRY_BY_SCENE[scene_id]
    floor_center, floor_size = geometry["floor_box"]
    return (("floor", floor_center, floor_size), *collision_repair_boxes(scene_id))


__all__ = [
    "APPARTEMENT_KITCHEN_SCENE",
    "COLLISION_GEOMETRY_BY_SCENE",
    "HAND_COLLISION_CONTACT_OFFSET",
    "MODERN_APARTMENT_SCENE",
    "MODERN_KITCHEN_SCENE",
    "REALISTIC_KITCHEN_SCENE",
    "SCENE_COLLISION_CONTACT_OFFSET",
    "SCENE_COLLISION_POLICY",
    "SCENE_COLLISION_ROOT_NAME",
    "collision_repair_boxes",
    "complete_scene_collision_boxes",
]
