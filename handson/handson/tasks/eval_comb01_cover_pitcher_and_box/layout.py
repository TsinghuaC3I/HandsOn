"""Joint layout of the unchanged B01/B05 assets (metres, world frame).

This sampler checks conservative footprints, not final PhysX intersections.
The latter and complete arm/hand reach sweeps are separate acceptance gates.
"""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np

TASK_NAME = "eval_comb01_cover_pitcher_and_box"
BODY_NAMES = ("pitcher_lid", "019_pitcher_base", "Box_lid", "Clear_Box")
# The box stays in B05's far lane; its unchanged lid rests across the near
# rim so the underside can be pinched. The pitcher pair uses the inner lane.
# Source units, scale, mass, colliders, and success predicates are unchanged.
CENTERS = {
    "pitcher_lid": (-0.050, -0.135),
    "019_pitcher_base": (-0.035, 0.160),
    "Box_lid": (-0.355, -0.128),
    "Clear_Box": (-0.355, 0.265),
}
XY_BOUNDS = {
    name: np.asarray([np.asarray(xy) - jitter, np.asarray(xy) + jitter])
    for name, xy in CENTERS.items()
    for jitter in [np.asarray([0.009 if name in {"Box_lid", "Clear_Box"} else 0.012, 0.012])]
}
# The staged lid follows the box yaw instead of sampling independently.
XY_BOUNDS["Box_lid"] = np.asarray([[-0.402, -0.151], [-0.308, -0.105]])
TABLE_XY_BOUNDS = np.asarray([[-0.55, -0.80], [0.25, 0.80]])
TABLE_EDGE_MARGIN = 0.020
MAX_ATTEMPTS = 64


def source_parameters():
    # Lazy import avoids a cycle with ABC's task-spec registration; all values
    # are the source tasks' authoritative constants, not copied tuning knobs.
    from ...sim.isaaclab import abc_collection_env as source

    return source


def support_offset(name: str) -> float:
    source = source_parameters()
    if name in source._B01_SUPPORT_LOCAL_MIN_Z:
        return -source._B01_SUPPORT_LOCAL_MIN_Z[name]
    return -source._B05_SUPPORT_LOCAL_MIN_Z[name] + source._B05_RESET_SUPPORT_CLEARANCE


def half_extents(name: str) -> np.ndarray:
    source = source_parameters()
    if name in source._B01_OBJECT_FOOTPRINT_RADII:
        return np.full(2, source._B01_OBJECT_FOOTPRINT_RADII[name])
    return source._B05_RESET_FOOTPRINT_HALF_EXTENTS[name].copy()


def layout_margin(poses: dict[str, np.ndarray]) -> float:
    """Signed conservative table/pair clearance; positive means separated."""
    source = source_parameters()
    margins = []
    for name in BODY_NAMES:
        xy = np.asarray(poses[name])[:2]
        extents = half_extents(name)
        margins.extend(xy - extents - TABLE_XY_BOUNDS[0] - TABLE_EDGE_MARGIN)
        margins.extend(TABLE_XY_BOUNDS[1] - xy - extents - TABLE_EDGE_MARGIN)
    for first, second in combinations(BODY_NAMES, 2):
        if {first, second} == {"Box_lid", "Clear_Box"}:
            # The box rim intentionally supports the staged lid.
            continue
        separation = np.abs(poses[first][:2] - poses[second][:2])
        clearance = max(source._B01_OBJECT_CLEARANCE_MARGIN, source._B05_OBJECT_CLEARANCE_MARGIN)
        margins.append(float(np.max(separation - half_extents(first) - half_extents(second))) - clearance)
    return float(min(margins))


def propped_lid_pose(box_pose: np.ndarray, table_top_z: float) -> np.ndarray:
    """Rest the unchanged lid on the table and the box's near rim.

    The source lid is 429 mm long and the box is 149 mm tall. Its lower
    edge stays on the table while the opposite edge rests on the existing
    rim, exposing the underside for a physical pinch without a grasp proxy.
    """
    from scipy.spatial.transform import Rotation
    half_length, half_thickness = 0.2145, 0.010
    box_height = 0.149
    overhang = 0.025
    low, high = 0., math.pi / 4
    for _ in range(50):
        tilt = (low + high) / 2
        if 2 * half_length * math.sin(tilt) - overhang * math.tan(tilt) < box_height:
            low = tilt
        else:
            high = tilt
    tilt = (low + high) / 2
    yaw = Rotation.from_quat(box_pose[3:], scalar_first=True)
    rotation = yaw * Rotation.from_euler("x", tilt)
    forward = half_length * math.cos(tilt) + half_thickness * math.sin(tilt)
    position = box_pose[:3].copy()
    position[:2] += yaw.apply([0., -0.2145 + overhang - forward, 0.])[:2]
    position[2] = (table_top_z + half_length * math.sin(tilt)
                   + half_thickness * math.cos(tilt)
                   + source_parameters()._B05_RESET_SUPPORT_CLEARANCE)
    return np.r_[position, rotation.as_quat(scalar_first=True)]


def sample_layout(rng, *, randomize: bool, table_top_z: float) -> dict[str, np.ndarray]:
    source = source_parameters()
    for _ in range(MAX_ATTEMPTS):
        poses = {}
        for name in BODY_NAMES:
            if name == "Box_lid":
                continue
            xy = rng.uniform(*XY_BOUNDS[name]) if randomize else np.asarray(CENTERS[name])
            yaw_bounds = source._B01_YAW_BOUNDS_DEG if name in source._B01_SUPPORT_LOCAL_MIN_Z else source._B05_YAW_BOUNDS_DEG
            yaw = math.radians(float(rng.uniform(*yaw_bounds))) if randomize else 0.0
            poses[name] = np.asarray([
                *xy, table_top_z + support_offset(name),
                math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2),
            ], dtype=np.float64)
        poses["Box_lid"] = propped_lid_pose(poses["Clear_Box"], table_top_z)
        if layout_margin(poses) >= 0.0:
            return poses
        if not randomize:
            break
    raise RuntimeError("XY5 cannot sample a supported, separated four-object layout")


def apply_authored_layout(root) -> None:
    """Use the same supported layout before the first PhysX reset."""
    source = source_parameters()
    poses = sample_layout(np.random.default_rng(0), randomize=False, table_top_z=source._TABLE_TOP_Z)
    for name, pose in poses.items():
        body = root.find(f".//body[@name='{name}']")
        if body is None:
            raise ValueError(f"XY5 source asset missing: {name}")
        body.set("pos", " ".join(f"{v:.10f}" for v in pose[:3]))
        body.set("quat", " ".join(f"{v:.10f}" for v in pose[3:]))
