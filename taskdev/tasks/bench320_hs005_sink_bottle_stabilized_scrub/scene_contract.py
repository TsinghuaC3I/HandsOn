"""CPU-safe geometry and release contract for B320-HS005 core scene."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable


TASK_ID = "bench320_hs005_sink_bottle_stabilized_scrub"
SOURCE_INDEX = 267
SOURCE_ROW = 268
PHYSICS_HZ = 240
SIM_SECONDS = 0.5
WARMUP_STEPS = 120
ZERO_ACTION_STEPS = int(PHYSICS_HZ * SIM_SECONDS)
HEAD_HEIGHT = 480
HEAD_WIDTH = 640

BOTTLE_SOURCE_BOUNDS = (
    (-0.7011311317, 0.0006743568, -0.7011313027),
    (0.7011311317, 4.9963894217, 0.7011312934),
)
BOTTLE_VISUAL_SCALE = 0.05
SINK_SOURCE_BOUNDS = (
    (-3.05192320, 0.71348094, -1.22463718),
    (-2.10111115, 0.83215999, -0.75113154),
)
BRUSH_SOURCE_BOUNDS = (
    (-0.0936167649, -0.0140212401, -0.0140212318),
    (0.1907145379, 0.0140212382, 0.0140212374),
)
BRUSH_VISUAL_SCALE = 1.0
DISH_SOAP_SOURCE_BOUNDS = (
    (-1.2147976690, 0.6083819151, -1.0452475433),
    (1.2143529654, 7.2391861772, 2.4101088245),
)
DISH_SOAP_VISUAL_SCALE = 0.035
SPONGE_SOURCE_BOUNDS = (
    (-1.2276641354, 0.0000139111, -2.7725961561),
    (1.2276641354, 0.9816778611, 2.7725961087),
)
SPONGE_VISUAL_SCALE = 0.018

# Modern Apartment is authored Y-up.  The task transform maps source
# (x, y, z) to task-local (-z, -x, y), placing the counter front toward the
# robot at negative world X.  XY translation centres the actual basin rather
# than the adjacent drainboard at (-0.15, 0.0).
KITCHEN_ORIENT_QUAT_REAL_XYZ = (0.5, 0.5, -0.5, -0.5)
KITCHEN_PLAN_TRANSLATION_XY = (-1.135, -2.800)
SINK_BASIN_SOURCE_FLOOR_Y = 0.71348094
SINK_BASIN_SAFE_PLAN_BOUNDS = ((-0.340, -0.200), (0.040, 0.200))
# Measured by a vertical ray against the downloaded Plane_007 counter mesh at
# the soap/sponge placement strip.  This is 0.121315 m above the native task
# table and must not be confused with the lower bootstrap table surface.
KITCHEN_COUNTER_TOP_OFFSET_M = 0.121315
# The raised bilateral reset palms are centred near x=-0.3715 m. Keep tall
# static context props on the rear counter strip at x=+0.16 m so their complete
# colliders cannot enter either palm/finger envelope during the first PhysX
# frame. The two props remain separated laterally for clear support.
CONTEXT_PROP_REAR_X = 0.16
CONTEXT_PROP_DISH_SOAP_Y = 0.24
CONTEXT_PROP_SPONGE_Y = -0.18


@dataclass(frozen=True)
class Aabb:
    low: tuple[float, float, float]
    high: tuple[float, float, float]

    @property
    def extent(self) -> tuple[float, float, float]:
        return tuple(b - a for a, b in zip(self.low, self.high, strict=True))

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple((a + b) * 0.5 for a, b in zip(self.low, self.high, strict=True))

    def translated(self, offset: Iterable[float]) -> "Aabb":
        value = tuple(float(item) for item in offset)
        if len(value) != 3:
            raise ValueError("translation must have three values")
        return Aabb(
            tuple(a + b for a, b in zip(self.low, value, strict=True)),
            tuple(a + b for a, b in zip(self.high, value, strict=True)),
        )

    def scaled(self, scale: float) -> "Aabb":
        value = float(scale)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("AABB scale must be positive and finite")
        return Aabb(
            tuple(value * item for item in self.low),
            tuple(value * item for item in self.high),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rotate_x_90_bounds(bounds: tuple[tuple[float, ...], tuple[float, ...]]) -> Aabb:
    """Map source (x, y, z) to world-local (x, -z, y)."""

    low, high = bounds
    points = [
        (x, -z, y)
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ]
    return Aabb(
        tuple(min(point[axis] for point in points) for axis in range(3)),
        tuple(max(point[axis] for point in points) for axis in range(3)),
    )


def modern_apartment_bounds(
    bounds: tuple[tuple[float, ...], tuple[float, ...]],
    translation: Iterable[float],
) -> Aabb:
    """Map Modern Apartment source bounds through (-z, -x, y) + translation."""

    offset = tuple(float(value) for value in translation)
    if len(offset) != 3:
        raise ValueError("translation must have three values")
    low, high = bounds
    points = [
        (-z + offset[0], -x + offset[1], y + offset[2])
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ]
    return Aabb(
        tuple(min(point[axis] for point in points) for axis in range(3)),
        tuple(max(point[axis] for point in points) for axis in range(3)),
    )


def centered_box(center: Iterable[float], size: Iterable[float]) -> Aabb:
    center_values = tuple(float(value) for value in center)
    size_values = tuple(float(value) for value in size)
    if len(center_values) != 3 or len(size_values) != 3:
        raise ValueError("box center and size must each have three values")
    if not all(value > 0.0 and math.isfinite(value) for value in size_values):
        raise ValueError("box size must be positive and finite")
    half = tuple(value * 0.5 for value in size_values)
    return Aabb(
        tuple(a - b for a, b in zip(center_values, half, strict=True)),
        tuple(a + b for a, b in zip(center_values, half, strict=True)),
    )


def overlap_depth(first: Aabb, second: Aabb) -> float:
    axis_overlap = tuple(
        min(a_high, b_high) - max(a_low, b_low)
        for a_low, a_high, b_low, b_high in zip(
            first.low, first.high, second.low, second.high, strict=True
        )
    )
    if all(value > 0.0 for value in axis_overlap):
        return min(axis_overlap)
    return 0.0


def placement_contract(table_top_z: float) -> dict[str, object]:
    if not math.isfinite(table_top_z):
        raise ValueError("table_top_z must be finite")
    sink_origin = (-0.15, 0.0, table_top_z + 0.005)
    kitchen_translation = (
        KITCHEN_PLAN_TRANSLATION_XY[0],
        KITCHEN_PLAN_TRANSLATION_XY[1],
        sink_origin[2] - SINK_BASIN_SOURCE_FLOOR_Y,
    )
    # Both dynamic objects begin on the real shallow basin floor.  They occupy
    # separated lanes inside the same cavity and retain positive clearance
    # under the full seeded XY randomization envelope.
    # The upright bottle collider begins at its root. Author it directly on
    # the basin floor; the former 3 mm air gap produced an exact 3 mm origin
    # drift even though the bottle otherwise stayed upright.
    bottle_origin = (-0.10, 0.11, sink_origin[2])
    # The horizontal brush is supported by its 14.02124 mm source/collider
    # radius. Do not add the former 3 mm air gap: it made the brush fall and
    # roll during the strict recording-origin gate.
    brush_origin = (-0.191, -0.125, sink_origin[2] + 0.01402124)
    kitchen_counter_top_z = table_top_z + KITCHEN_COUNTER_TOP_OFFSET_M
    dish_soap_origin = (
        CONTEXT_PROP_REAR_X,
        CONTEXT_PROP_DISH_SOAP_Y,
        kitchen_counter_top_z
        + 0.003
        - 0.6083819151 * DISH_SOAP_VISUAL_SCALE,
    )
    # Keep the low sponge on the same measured rear counter strip, separated
    # laterally from the soap and from both raised reset hands.
    sponge_origin = (
        CONTEXT_PROP_REAR_X,
        CONTEXT_PROP_SPONGE_Y,
        kitchen_counter_top_z + 0.003,
    )
    sink_visual = modern_apartment_bounds(SINK_SOURCE_BOUNDS, kitchen_translation)
    bottle_visual = (
        rotate_x_90_bounds(BOTTLE_SOURCE_BOUNDS)
        .scaled(BOTTLE_VISUAL_SCALE)
        .translated(bottle_origin)
    )
    brush_visual = (
        Aabb(*BRUSH_SOURCE_BOUNDS)
        .scaled(BRUSH_VISUAL_SCALE)
        .translated(brush_origin)
    )
    dish_soap_visual = (
        rotate_x_90_bounds(DISH_SOAP_SOURCE_BOUNDS)
        .scaled(DISH_SOAP_VISUAL_SCALE)
        .translated(dish_soap_origin)
    )
    sponge_visual = (
        rotate_x_90_bounds(SPONGE_SOURCE_BOUNDS)
        .scaled(SPONGE_VISUAL_SCALE)
        .translated(sponge_origin)
    )
    basin_safe = Aabb(
        (
            SINK_BASIN_SAFE_PLAN_BOUNDS[0][0],
            SINK_BASIN_SAFE_PLAN_BOUNDS[0][1],
            sink_origin[2],
        ),
        (
            SINK_BASIN_SAFE_PLAN_BOUNDS[1][0],
            SINK_BASIN_SAFE_PLAN_BOUNDS[1][1],
            sink_visual.high[2],
        ),
    )
    bottle_collider = centered_box(
        (bottle_origin[0], bottle_origin[1], bottle_origin[2] + 0.125),
        (0.0694, 0.0694, 0.250),
    )
    inside_basin_margins = {
        "bottle_front": bottle_collider.low[0] - basin_safe.low[0],
        "bottle_back": basin_safe.high[0] - bottle_collider.high[0],
        "bottle_right": bottle_collider.low[1] - basin_safe.low[1],
        "bottle_left": basin_safe.high[1] - bottle_collider.high[1],
        "brush_front": brush_visual.low[0] - basin_safe.low[0],
        "brush_back": basin_safe.high[0] - brush_visual.high[0],
        "brush_right": brush_visual.low[1] - basin_safe.low[1],
        "brush_left": basin_safe.high[1] - brush_visual.high[1],
    }
    return {
        "table_top_z": table_top_z,
        "kitchen_counter_top_z": kitchen_counter_top_z,
        "sink_origin": sink_origin,
        "kitchen_translation": kitchen_translation,
        "kitchen_orient_quat_real_xyz": KITCHEN_ORIENT_QUAT_REAL_XYZ,
        "bottle_origin": bottle_origin,
        "brush_origin": brush_origin,
        "dish_soap_origin": dish_soap_origin,
        "sponge_origin": sponge_origin,
        "basin_safe": basin_safe,
        "sink_visual": sink_visual,
        "bottle_visual": bottle_visual,
        "brush_visual": brush_visual,
        "dish_soap_visual": dish_soap_visual,
        "sponge_visual": sponge_visual,
        "bottle_collider": bottle_collider,
        "support_gaps_m": {
            "sink_to_table": sink_visual.low[2] - table_top_z,
            "bottle_to_basin": bottle_collider.low[2] - sink_origin[2],
            "brush_to_basin": brush_visual.low[2] - sink_origin[2],
            "dish_soap_to_counter": (
                dish_soap_visual.low[2] - kitchen_counter_top_z
            ),
            "sponge_to_counter": sponge_visual.low[2] - kitchen_counter_top_z,
        },
        "inside_basin_margins_m": inside_basin_margins,
        "forbidden_overlap_m": {
            "bottle_brush": overlap_depth(bottle_collider, brush_visual),
            "bottle_dish_soap": overlap_depth(bottle_collider, dish_soap_visual),
            "bottle_sponge": overlap_depth(bottle_collider, sponge_visual),
            "brush_dish_soap": overlap_depth(brush_visual, dish_soap_visual),
            "brush_sponge": overlap_depth(brush_visual, sponge_visual),
            "dish_soap_sponge": overlap_depth(dish_soap_visual, sponge_visual),
        },
    }


def initial_layout_pass(contract: dict[str, object]) -> bool:
    gaps = contract["support_gaps_m"]
    margins = contract["inside_basin_margins_m"]
    overlaps = contract["forbidden_overlap_m"]
    assert isinstance(gaps, dict) and isinstance(margins, dict) and isinstance(overlaps, dict)
    return bool(
        0.0 <= float(gaps["bottle_to_basin"]) <= 0.0001
        and 0.0 <= float(gaps["brush_to_basin"]) <= 0.0001
        and all(
            0.001 <= float(value) <= 0.006
            for name, value in gaps.items()
            if name not in {"bottle_to_basin", "brush_to_basin"}
        )
        and all(float(value) >= 0.04 - 1.0e-9 for value in margins.values())
        and all(float(value) == 0.0 for value in overlaps.values())
    )


def zero_action_stability_pass(
    *,
    bottle_displacement_m: float,
    brush_displacement_m: float,
    bottle_linear_speed_m_s: float,
    brush_linear_speed_m_s: float,
    bottle_angular_speed_rad_s: float,
    brush_angular_speed_rad_s: float,
    simulated_seconds: float,
) -> bool:
    values = (
        bottle_displacement_m,
        brush_displacement_m,
        bottle_linear_speed_m_s,
        brush_linear_speed_m_s,
        bottle_angular_speed_rad_s,
        brush_angular_speed_rad_s,
        simulated_seconds,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        return False
    return bool(
        simulated_seconds >= SIM_SECONDS
        and bottle_displacement_m <= 0.012
        and brush_displacement_m <= 0.012
        and bottle_linear_speed_m_s <= 0.02
        and brush_linear_speed_m_s <= 0.02
        and bottle_angular_speed_rad_s <= 0.15
        and brush_angular_speed_rad_s <= 0.15
    )


def release_scope() -> dict[str, bool]:
    return {
        "core_scene_ready": False,
        "scrub_path_validated": False,
        "grasp_validated": False,
        "action_tuning_performed": False,
        "task_complete": False,
    }
