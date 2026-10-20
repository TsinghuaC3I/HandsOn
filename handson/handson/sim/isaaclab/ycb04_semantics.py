"""Pure geometry and acceptance semantics for YCB04 cup denesting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CupGeometry:
    name: str
    radius_x: float
    radius_y: float
    half_height: float


LARGE_CUP = CupGeometry("ycb04_large_cup", 0.0325, 0.0325, 0.032)
MEDIUM_CUP = CupGeometry("ycb04_medium_cup", 0.0300, 0.0300, 0.031)
SMALL_CUP = CupGeometry("ycb04_small_cup", 0.027499, 0.027499, 0.0299985)
CUPS_LARGE_TO_SMALL = (LARGE_CUP, MEDIUM_CUP, SMALL_CUP)

UPRIGHT_MIN_ALIGNMENT = float(math.cos(math.radians(12.0)))
TABLE_BOTTOM_TOLERANCE_M = 0.005
PAIR_SURFACE_CLEARANCE_M = 0.010
NEIGHBOR_CENTER_MAX_M = 0.160
ROW_DEPTH_SPREAD_MAX_M = 0.030
PALM_SURFACE_CLEARANCE_M = 0.030
LINEAR_SPEED_MAX_MPS = 0.020
ANGULAR_SPEED_MAX_RADPS = 0.300
PAIR_SEPARATION_MARGIN_M = 0.002
INITIAL_NEST_RADIAL_TOLERANCE_M = 0.001
INITIAL_NEST_VERTICAL_OVERLAP_M = 0.003


def quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = np.asarray([w, x, y, z], dtype=np.float64) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def observable_pose_velocity(
    previous_pose: np.ndarray,
    current_pose: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Return the finite-difference twist observable at the control rate.

    YCB04 uses this same pose-derived quantity in simulation and on the real
    robot.  It avoids treating solver-iteration contact impulses, which need
    not produce observable object motion, as a final-state velocity.
    """
    previous = np.asarray(previous_pose, dtype=np.float64).reshape(7)
    current = np.asarray(current_pose, dtype=np.float64).reshape(7)
    sample_dt = max(float(dt), 1.0e-9)
    linear = (current[:3] - previous[:3]) / sample_dt

    previous_quat = previous[3:7]
    current_quat = current[3:7]
    previous_norm = float(np.linalg.norm(previous_quat))
    current_norm = float(np.linalg.norm(current_quat))
    if previous_norm <= 1.0e-12 or current_norm <= 1.0e-12:
        angular = np.zeros(3, dtype=np.float64)
    else:
        previous_quat = previous_quat / previous_norm
        current_quat = current_quat / current_norm
        pw, px, py, pz = previous_quat
        cw, cx, cy, cz = current_quat
        # q_delta = q_current * conjugate(q_previous), in world axes.
        delta = np.asarray(
            [
                cw * pw + cx * px + cy * py + cz * pz,
                -cw * px + cx * pw - cy * pz + cz * py,
                -cw * py + cx * pz + cy * pw - cz * px,
                -cw * pz - cx * py + cy * px + cz * pw,
            ],
            dtype=np.float64,
        )
        if delta[0] < 0.0:
            delta *= -1.0
        vector_norm = float(np.linalg.norm(delta[1:]))
        if vector_norm <= 1.0e-12:
            angular = np.zeros(3, dtype=np.float64)
        else:
            angle = 2.0 * math.atan2(vector_norm, max(float(delta[0]), 0.0))
            angular = delta[1:] * (angle / (vector_norm * sample_dt))
    return np.concatenate([linear, angular]).astype(np.float64)


def upright_alignment(pose: np.ndarray) -> float:
    rotation = quat_wxyz_to_matrix(np.asarray(pose, dtype=np.float64)[3:7])
    return float(rotation[2, 2])


def vertical_half_extent(pose: np.ndarray, geometry: CupGeometry) -> float:
    rotation = quat_wxyz_to_matrix(np.asarray(pose, dtype=np.float64)[3:7])
    return float(
        abs(rotation[2, 0]) * geometry.radius_x
        + abs(rotation[2, 1]) * geometry.radius_y
        + abs(rotation[2, 2]) * geometry.half_height
    )


def world_bottom(pose: np.ndarray, geometry: CupGeometry) -> float:
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    return float(pose[2] - vertical_half_extent(pose, geometry))


def world_top(pose: np.ndarray, geometry: CupGeometry) -> float:
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    return float(pose[2] + vertical_half_extent(pose, geometry))


def pair_surface_clearance_xy(
    first_pose: np.ndarray,
    first: CupGeometry,
    second_pose: np.ndarray,
    second: CupGeometry,
) -> float:
    center_distance = float(
        np.linalg.norm(
            np.asarray(first_pose, dtype=np.float64)[:2]
            - np.asarray(second_pose, dtype=np.float64)[:2]
        )
    )
    return center_distance - max(first.radius_x, first.radius_y) - max(
        second.radius_x, second.radius_y
    )


def pair_is_nested(
    inner_pose: np.ndarray,
    inner: CupGeometry,
    outer_pose: np.ndarray,
    outer: CupGeometry,
) -> bool:
    inner_pose = np.asarray(inner_pose, dtype=np.float64).reshape(7)
    outer_pose = np.asarray(outer_pose, dtype=np.float64).reshape(7)
    radial_distance = float(np.linalg.norm(inner_pose[:2] - outer_pose[:2]))
    radial_allowance = (
        min(outer.radius_x, outer.radius_y)
        - max(inner.radius_x, inner.radius_y)
        + INITIAL_NEST_RADIAL_TOLERANCE_M
    )
    vertical_overlap = min(
        world_top(inner_pose, inner), world_top(outer_pose, outer)
    ) - max(world_bottom(inner_pose, inner), world_bottom(outer_pose, outer))
    return bool(
        upright_alignment(inner_pose) >= UPRIGHT_MIN_ALIGNMENT
        and upright_alignment(outer_pose) >= UPRIGHT_MIN_ALIGNMENT
        and radial_distance <= radial_allowance
        and vertical_overlap >= INITIAL_NEST_VERTICAL_OVERLAP_M
        and inner_pose[2] > world_bottom(outer_pose, outer)
    )


def pair_is_fully_separated(
    inner_pose: np.ndarray,
    inner: CupGeometry,
    outer_pose: np.ndarray,
    outer: CupGeometry,
) -> bool:
    inner_pose = np.asarray(inner_pose, dtype=np.float64).reshape(7)
    outer_pose = np.asarray(outer_pose, dtype=np.float64).reshape(7)
    vertical_gap = max(
        world_bottom(inner_pose, inner) - world_top(outer_pose, outer),
        world_bottom(outer_pose, outer) - world_top(inner_pose, inner),
    )
    radial_clearance = pair_surface_clearance_xy(
        inner_pose, inner, outer_pose, outer
    )
    return bool(
        vertical_gap >= PAIR_SEPARATION_MARGIN_M
        or radial_clearance >= PAIR_SEPARATION_MARGIN_M
    )


def head_camera_coordinates(
    poses: tuple[np.ndarray, ...],
    camera_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera_pose = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)
    centers = np.stack(
        [np.asarray(pose, dtype=np.float64).reshape(7)[:3] for pose in poses],
        axis=0,
    )
    camera_points = (
        camera_pose[:3, :3].T @ (centers - camera_pose[:3, 3]).T
    ).T
    depths = -camera_points[:, 2]
    horizontal = camera_points[:, 0] / np.maximum(depths, 1.0e-9)
    return horizontal.astype(np.float64), depths.astype(np.float64)


def point_to_upright_cup_envelope_distance(
    point: np.ndarray,
    pose: np.ndarray,
    geometry: CupGeometry,
) -> float:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    radial = float(np.linalg.norm(point[:2] - pose[:2]))
    radial_outside = max(0.0, radial - max(geometry.radius_x, geometry.radius_y))
    vertical_outside = max(0.0, abs(float(point[2] - pose[2])) - geometry.half_height)
    return float(math.hypot(radial_outside, vertical_outside))


def final_layout_metrics(
    poses: dict[str, np.ndarray],
    velocities: dict[str, np.ndarray],
    *,
    table_top_z: float,
    camera_pose: np.ndarray,
    palm_points: tuple[np.ndarray, ...],
    hand_contact_force: float,
    contact_available: bool,
) -> dict[str, object]:
    ordered_poses = tuple(poses[cup.name] for cup in CUPS_LARGE_TO_SMALL)
    bottoms = np.asarray(
        [world_bottom(pose, cup) for pose, cup in zip(ordered_poses, CUPS_LARGE_TO_SMALL)],
        dtype=np.float64,
    )
    alignments = np.asarray(
        [upright_alignment(pose) for pose in ordered_poses], dtype=np.float64
    )
    horizontal, depths = head_camera_coordinates(ordered_poses, camera_pose)
    clearances = np.asarray(
        [
            pair_surface_clearance_xy(ordered_poses[i], CUPS_LARGE_TO_SMALL[i], ordered_poses[j], CUPS_LARGE_TO_SMALL[j])
            for i, j in ((0, 1), (1, 2), (0, 2))
        ],
        dtype=np.float64,
    )
    neighbor_distances = np.asarray(
        [
            np.linalg.norm(ordered_poses[0][:3] - ordered_poses[1][:3]),
            np.linalg.norm(ordered_poses[1][:3] - ordered_poses[2][:3]),
        ],
        dtype=np.float64,
    )
    linear_speeds = np.asarray(
        [np.linalg.norm(velocities[cup.name][:3]) for cup in CUPS_LARGE_TO_SMALL],
        dtype=np.float64,
    )
    angular_speeds = np.asarray(
        [np.linalg.norm(velocities[cup.name][3:]) for cup in CUPS_LARGE_TO_SMALL],
        dtype=np.float64,
    )
    palm_clearance = min(
        (
            point_to_upright_cup_envelope_distance(point, pose, cup)
            for point in palm_points
            for pose, cup in zip(ordered_poses, CUPS_LARGE_TO_SMALL)
        ),
        default=math.inf,
    )
    on_table = bool(np.all(np.abs(bottoms - float(table_top_z)) <= TABLE_BOTTOM_TOLERANCE_M))
    upright = bool(np.all(alignments >= UPRIGHT_MIN_ALIGNMENT))
    separated = bool(np.all(clearances >= PAIR_SURFACE_CLEARANCE_M))
    ordered = bool(horizontal[0] < horizontal[1] < horizontal[2])
    neighbor_spacing = bool(np.all(neighbor_distances <= NEIGHBOR_CENTER_MAX_M))
    row_depth = bool(float(np.ptp(depths)) <= ROW_DEPTH_SPREAD_MAX_M)
    stable = bool(
        np.all(linear_speeds < LINEAR_SPEED_MAX_MPS)
        and np.all(angular_speeds < ANGULAR_SPEED_MAX_RADPS)
    )
    released_now = bool(
        contact_available
        and hand_contact_force <= 0.05
        and palm_clearance >= PALM_SURFACE_CLEARANCE_M
    )
    return {
        "on_table": on_table,
        "upright": upright,
        "separated": separated,
        "ordered": ordered,
        "neighbor_spacing": neighbor_spacing,
        "row_depth": row_depth,
        "stable": stable,
        "released_now": released_now,
        "bottom_errors_m": (bottoms - float(table_top_z)).tolist(),
        "upright_alignments": alignments.tolist(),
        "surface_clearances_m": clearances.tolist(),
        "neighbor_distances_m": neighbor_distances.tolist(),
        "head_horizontal": horizontal.tolist(),
        "head_depths_m": depths.tolist(),
        "row_depth_spread_m": float(np.ptp(depths)),
        "linear_speeds_mps": linear_speeds.tolist(),
        "angular_speeds_radps": angular_speeds.tolist(),
        "palm_surface_clearance_m": float(palm_clearance),
        "hand_contact_force_n": float(hand_contact_force),
        "contact_available": bool(contact_available),
    }
