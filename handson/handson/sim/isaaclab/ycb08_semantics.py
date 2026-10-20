"""Pure reset geometry and physical event semantics for YCB08.

The module contains no Isaac imports.  The 120 Hz live environment, replay,
and unit tests all use this one state machine so success cannot be inferred
from visual proximity or from a final pose alone.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


YCB08_TASK_ID = "YCB08"
YCB08_TASK_NAME = "t07_flip_sponge"
YCB08_SKILLET_NAME = "ycb08_skillet"
YCB08_SPATULA_NAME = "ycb08_spatula"
YCB08_SPONGE_NAME = "ycb08_sponge"
YCB08_BODY_NAMES = (
    YCB08_SKILLET_NAME,
    YCB08_SPATULA_NAME,
    YCB08_SPONGE_NAME,
)

# Bounds and semantic frames measured from the tracked, unscaled YCB meshes.
YCB08_SKILLET_LOCAL_MIN_M = np.asarray(
    [-0.1373685, -0.2234180, -0.0455615], dtype=np.float64
)
YCB08_SKILLET_LOCAL_MAX_M = np.asarray(
    [0.1373685, 0.2234180, 0.0455615], dtype=np.float64
)
YCB08_PAN_CENTER_LOCAL_XY_M = np.asarray([0.0010605, 0.0872140], dtype=np.float64)
YCB08_PAN_FLOOR_TOP_LOCAL_Z_M = -0.0385085
YCB08_PAN_RIM_TOP_LOCAL_Z_M = 0.0201675
YCB08_PAN_INNER_RADIUS_M = 0.121495
YCB08_PAN_RIM_CLEARANCE_M = 0.004
YCB08_SPATULA_LOCAL_MIN_M = np.asarray(
    [-0.0412390, -0.1746665, -0.0179495], dtype=np.float64
)
YCB08_SPATULA_LOCAL_MAX_M = np.asarray(
    [0.0412390, 0.1746665, 0.0179495], dtype=np.float64
)
YCB08_SPATULA_BLADE_X_RANGE_M = (-0.042, 0.042)
YCB08_SPATULA_BLADE_Y_RANGE_M = (0.060, 0.175)
YCB08_SPATULA_BLADE_REFERENCE_LOCAL_M = np.asarray(
    [0.0, 0.120, 0.0], dtype=np.float64
)
# The source-density estimate was only 51.5 g, which made a fingertip contact
# capable of rolling the thin, inverted reset pose onto its opposite face.
# Use a normal 100 g kitchen-turner mass and scale the authored inertia by the
# same factor.  The collision-volume centroid lowers the centre of mass by
# about 2.1 mm in the ready pose and removes the artificial origin-centred
# balance without changing the mesh or the blade's insertion geometry.
YCB08_SPATULA_MASS_KG = 0.100
YCB08_SPATULA_CENTER_OF_MASS_LOCAL_M = np.asarray(
    [-0.00150178, -0.02658348, 0.00209358], dtype=np.float64
)
YCB08_SPATULA_DIAGONAL_INERTIA_KG_M2 = np.asarray(
    [0.00103107, 0.00006757, 0.00107767], dtype=np.float64
)
YCB08_SPONGE_HALF_EXTENTS_M = np.asarray(
    [0.0360000, 0.0570000, 0.0070000], dtype=np.float64
)

# Use broad, arm-specific tabletop regions rather than centimetre-scale
# jitter. World -Y is the right-hand half and +Y is the left-hand half. The
# sampler still rejects off-table and mutually overlapping oriented bounds.
YCB08_BOWL_CENTER_XY_BOUNDS_M = np.asarray(
    [[-0.40, 0.06], [-0.18, 0.43]], dtype=np.float64
)
YCB08_SPATULA_CENTER_XY_BOUNDS_M = np.asarray(
    [[-0.40, -0.60], [-0.18, -0.20]], dtype=np.float64
)
YCB08_PAN_YAW_RANGE_DEG = (-180.0, 180.0)
YCB08_SPATULA_YAW_RANGE_DEG = (-180.0, 180.0)
# Keep the flat sponge 30 mm toward the pan-local far side of the bowl. This
# rotates with every handle heading and preserves about 94 mm of open runway
# between the approach-side inner rim and the sponge edge. At a natural
# 25-degree tip-down wrist angle, the unchanged inverted blade can then touch
# the pan floor while its curved shoulder clears the 60 mm rim.
YCB08_SPONGE_INSERTION_RUNWAY_OFFSET_LOCAL_XY_M = np.asarray(
    [0.0, -0.030], dtype=np.float64
)
YCB08_SPONGE_LOCAL_JITTER_XY_M = np.asarray([0.010, 0.008], dtype=np.float64)
YCB08_SPONGE_YAW_JITTER_DEG = 8.0
# The tracked spatula is a strongly curved turner.  In its source orientation,
# its +Y leading edge sits about 32 mm above its lowest collision point, so it
# hits the side/top of the 14 mm sponge before a human can wedge underneath.
# Turn it over about local Y and apply the measured 0.1616-degree support pitch.
# Two original-mesh support points then bracket the centre of mass (local
# y=-36.97 mm and +172.57 mm), while the leading edge is only 0.93 mm above
# the table.  No geometry, scale, fixture, or runtime assist is changed.
YCB08_SPATULA_TABLE_PITCH_DEG = 0.1616
YCB08_SPATULA_TABLE_SUPPORT_LOCAL_Z_M = -0.01746270716254606
# The source-facing side has a different stable table pitch/support height.
# At 2.2 degrees the unchanged collision hull has support on both sides of the
# measured centre of mass, so this is a physical reset rather than a fixture.
YCB08_SPATULA_FACE_UP_PITCH_DEG = 2.2
YCB08_SPATULA_FACE_UP_SUPPORT_LOCAL_Z_M = -0.015555785618101534
YCB08_SPATULA_FACE_UP_PROBABILITY = 0.50
YCB08_RESET_SUPPORT_CLEARANCE_M = 0.0005
YCB08_RESET_OBJECT_CLEARANCE_M = 0.025
YCB08_LAYOUT_ATTEMPTS = 256
# Physical tabletop bounds, inset 20 mm from the 0.8 x 1.6 m support box.
YCB08_TABLE_SUPPORT_XY_BOUNDS_M = np.asarray(
    [[-0.53, -0.78], [0.23, 0.78]], dtype=np.float64
)
YCB08_CAMERA_XY_BOUNDS = np.asarray(
    [[-0.42, -0.64], [-0.14, 0.47]], dtype=np.float64
)

YCB08_SETTLE_SECONDS = 0.25
YCB08_TOOL_MOVE_DISTANCE_M = 0.020
YCB08_MIN_BLADE_OVERLAP = 0.20
YCB08_MIN_SPONGE_LIFT_M = 0.020
YCB08_FLIP_CROSS_ANGLE_DEG = 90.0
YCB08_FINAL_FLIP_ANGLE_DEG = 155.0
YCB08_FLIP_ATTEMPT_RESET_ANGLE_DEG = 30.0
YCB08_MAX_PAN_TILT_DURING_FLIP_DEG = 25.0
YCB08_MAX_FINAL_PAN_TILT_DEG = 12.0
YCB08_MIN_CONTAINMENT_FRACTION = 0.80
YCB08_MAX_FINAL_LINEAR_SPEED_MPS = 0.030
YCB08_MAX_FINAL_ANGULAR_SPEED_RADPS = 0.35
YCB08_TOOL_RELEASE_SECONDS = 0.50
YCB08_FINAL_STABILITY_SECONDS = 1.00
YCB08_DIRECT_HAND_DEBOUNCE_SECONDS = 0.10
YCB08_OUTSIDE_SUPPORT_DEBOUNCE_SECONDS = 0.25
YCB08_PAN_OVERTURN_DEBOUNCE_SECONDS = 0.10
YCB08_PAN_OVERTURN_ANGLE_DEG = 75.0
YCB08_MAX_LINEAR_SPEED_MPS = 3.0
YCB08_MAX_ANGULAR_SPEED_RADPS = 25.0
YCB08_EXCESSIVE_VELOCITY_DEBOUNCE_SECONDS = 0.15
YCB08_CONTACT_FORCE_FLOOR_N = 1.0e-4
# The pan stabilizer is deliberately gated by physical task state.  It only
# latches after the left hand is touching a lifted, nearly level, settled pan;
# proximity or a controller button alone can never activate it.
YCB08_LEFT_PAN_LOCK_MIN_LIFT_M = 0.020
YCB08_LEFT_PAN_LOCK_MAX_TILT_DEG = 10.0
YCB08_LEFT_PAN_LOCK_MAX_LINEAR_SPEED_MPS = 0.12
YCB08_LEFT_PAN_LOCK_MAX_ANGULAR_SPEED_RADPS = 0.80
YCB08_LEFT_PAN_LOCK_DEBOUNCE_SECONDS = 0.20
# Release a disturbed hold so the operator can regrip and level the pan.
# The 10/12-degree hysteresis and debounce reject brief contact chatter.
YCB08_LEFT_PAN_UNLOCK_DEBOUNCE_SECONDS = 0.15
YCB08_WORKSPACE_XY_BOUNDS = np.asarray(
    [[-0.55, -0.72], [0.18, 0.72]], dtype=np.float64
)
YCB08_WORKSPACE_Z_RANGE_M = (0.80, 1.65)

YCB08_EVENT_FLAG_NAMES = (
    "tool_grasp_seen",
    "tool_moved_seen",
    "tool_contact_seen",
    "tool_under_seen",
    "lift_seen",
    "crossed_90_seen",
    "valid_flip_seen",
    "containment_ok",
    "pan_support_ok",
    "tool_released",
    "stable_final",
    "direct_hand_violation",
    "outside_support_violation",
    "pan_overturn_violation",
    "failure",
    "success",
)
YCB08_FINAL_CHECK_NAMES = (
    "history_complete",
    "opposite_face_now",
    "containment_ok",
    "pan_support_ok",
    "pan_level_now",
    "sponge_linear_speed_ok",
    "sponge_angular_speed_ok",
    "tool_released",
)
YCB08_FAILURE_CODES = {
    None: 0,
    "non_finite_state": 1,
    "task_object_exited_workspace": 2,
    "excessive_object_velocity": 3,
    "direct_hand_touched_sponge": 4,
    "sponge_supported_outside_pan": 5,
    "skillet_overturned": 6,
}


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_quaternion_wxyz(yaw_radians: float) -> np.ndarray:
    return np.asarray(
        [math.cos(0.5 * yaw_radians), 0.0, 0.0, math.sin(0.5 * yaw_radians)],
        dtype=np.float64,
    )


def yaw_ready_spatula_quaternion_wxyz(
    yaw_radians: float, pitch_radians: float
) -> np.ndarray:
    """Return ``Rz(yaw) * Rx(pitch) * Ry(pi)`` in wxyz order."""

    yaw_cos = math.cos(0.5 * yaw_radians)
    yaw_sin = math.sin(0.5 * yaw_radians)
    pitch_cos = math.cos(0.5 * pitch_radians)
    pitch_sin = math.sin(0.5 * pitch_radians)
    return np.asarray(
        [
            -yaw_sin * pitch_sin,
            -yaw_sin * pitch_cos,
            yaw_cos * pitch_cos,
            yaw_cos * pitch_sin,
        ],
        dtype=np.float64,
    )


def yaw_face_up_spatula_quaternion_wxyz(
    yaw_radians: float, pitch_radians: float
) -> np.ndarray:
    """Return the source-facing ``Rz(yaw) * Rx(pitch)`` pose in wxyz."""

    yaw_cos = math.cos(0.5 * yaw_radians)
    yaw_sin = math.sin(0.5 * yaw_radians)
    pitch_cos = math.cos(0.5 * pitch_radians)
    pitch_sin = math.sin(0.5 * pitch_radians)
    return np.asarray(
        [
            yaw_cos * pitch_cos,
            yaw_cos * pitch_sin,
            yaw_sin * pitch_sin,
            yaw_sin * pitch_cos,
        ],
        dtype=np.float64,
    )


def ycb08_spatula_is_face_up(spatula_pose: np.ndarray) -> bool:
    """Return whether the source-facing (+Z) side currently points upward."""
    pose = np.asarray(spatula_pose, dtype=np.float64).reshape(7)
    return bool(quaternion_wxyz_to_matrix(pose[3:7])[2, 2] > 0.0)


def transform_point(pose_wxyz: np.ndarray, local_point: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    return pose[:3] + quaternion_wxyz_to_matrix(pose[3:7]) @ np.asarray(
        local_point, dtype=np.float64
    ).reshape(3)


def inverse_transform_point(pose_wxyz: np.ndarray, world_point: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    return quaternion_wxyz_to_matrix(pose[3:7]).T @ (
        np.asarray(world_point, dtype=np.float64).reshape(3) - pose[:3]
    )


def _sample_value(
    random_state: Any,
    lower: np.ndarray | float,
    upper: np.ndarray | float,
    *,
    randomize: bool,
) -> np.ndarray | float:
    if randomize:
        return random_state.uniform(lower, upper)
    return 0.5 * (np.asarray(lower) + np.asarray(upper))


def _oriented_box_xy_corners(
    pose_wxyz: np.ndarray,
    local_min: np.ndarray,
    local_max: np.ndarray,
) -> np.ndarray:
    """Project all eight oriented 3-D bounds corners onto the tabletop."""
    local_min = np.asarray(local_min, dtype=np.float64).reshape(3)
    local_max = np.asarray(local_max, dtype=np.float64).reshape(3)
    corners = np.asarray(
        [
            [x, y, z]
            for x in (local_min[0], local_max[0])
            for y in (local_min[1], local_max[1])
            for z in (local_min[2], local_max[2])
        ],
        dtype=np.float64,
    )
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    world = pose[:3] + corners @ quaternion_wxyz_to_matrix(pose[3:7]).T
    return world[:, :2]


def _convex_xy_bounds_overlap(
    first: np.ndarray,
    second: np.ndarray,
    *,
    clearance_m: float,
) -> bool:
    """Conservatively test two oriented-box projections with SAT axes."""
    first_hull = np.asarray(first, dtype=np.float64).reshape(-1, 2)
    second_hull = np.asarray(second, dtype=np.float64).reshape(-1, 2)
    axes: list[np.ndarray] = []
    for points in (first_hull, second_hull):
        # Eight projected 3-D corners contain duplicate XY points. Their
        # covariance eigenvectors are the two rectangle axes for every yaw.
        covariance = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        del eigenvalues
        axes.extend(eigenvectors.T)
    clearance = max(0.0, float(clearance_m))
    for axis in axes:
        axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
        first_projection = first_hull @ axis
        second_projection = second_hull @ axis
        if (
            float(np.max(first_projection)) + clearance
            <= float(np.min(second_projection))
            or float(np.max(second_projection)) + clearance
            <= float(np.min(first_projection))
        ):
            return False
    return True


def ycb08_layout_is_valid(layout: dict[str, np.ndarray]) -> bool:
    pan = np.asarray(layout[YCB08_SKILLET_NAME], dtype=np.float64).reshape(7)
    spatula = np.asarray(layout[YCB08_SPATULA_NAME], dtype=np.float64).reshape(7)
    sponge = np.asarray(layout[YCB08_SPONGE_NAME], dtype=np.float64).reshape(7)
    bowl_center = transform_point(
        pan,
        np.asarray(
            [*YCB08_PAN_CENTER_LOCAL_XY_M, YCB08_PAN_FLOOR_TOP_LOCAL_Z_M],
            dtype=np.float64,
        ),
    )
    lower, upper = YCB08_CAMERA_XY_BOUNDS
    for point in (bowl_center, spatula[:3], sponge[:3]):
        if np.any(point[:2] < lower) or np.any(point[:2] > upper):
            return False
    if (
        np.any(bowl_center[:2] < YCB08_BOWL_CENTER_XY_BOUNDS_M[0])
        or np.any(bowl_center[:2] > YCB08_BOWL_CENTER_XY_BOUNDS_M[1])
        or np.any(spatula[:2] < YCB08_SPATULA_CENTER_XY_BOUNDS_M[0])
        or np.any(spatula[:2] > YCB08_SPATULA_CENTER_XY_BOUNDS_M[1])
    ):
        return False
    sponge_local = inverse_transform_point(pan, sponge[:3])
    radial = float(
        np.linalg.norm(sponge_local[:2] - YCB08_PAN_CENTER_LOCAL_XY_M)
    )
    if radial + float(np.linalg.norm(YCB08_SPONGE_HALF_EXTENTS_M[:2])) >= (
        YCB08_PAN_INNER_RADIUS_M - YCB08_PAN_RIM_CLEARANCE_M
    ):
        return False
    pan_corners = _oriented_box_xy_corners(
        pan,
        YCB08_SKILLET_LOCAL_MIN_M,
        YCB08_SKILLET_LOCAL_MAX_M,
    )
    spatula_corners = _oriented_box_xy_corners(
        spatula,
        YCB08_SPATULA_LOCAL_MIN_M,
        YCB08_SPATULA_LOCAL_MAX_M,
    )
    table_lower, table_upper = YCB08_TABLE_SUPPORT_XY_BOUNDS_M
    if any(
        np.any(corners < table_lower) or np.any(corners > table_upper)
        for corners in (pan_corners, spatula_corners)
    ):
        return False
    return not _convex_xy_bounds_overlap(
        pan_corners,
        spatula_corners,
        clearance_m=YCB08_RESET_OBJECT_CLEARANCE_M,
    )


def sample_ycb08_layout(
    random_state: Any,
    *,
    randomize: bool,
    table_top_z: float,
) -> dict[str, np.ndarray]:
    """Sample broad left/right tabletop poses with physical rejection."""
    for _ in range(YCB08_LAYOUT_ATTEMPTS):
        bowl_center = np.asarray(
            _sample_value(
                random_state,
                YCB08_BOWL_CENTER_XY_BOUNDS_M[0],
                YCB08_BOWL_CENTER_XY_BOUNDS_M[1],
                randomize=randomize,
            ),
            dtype=np.float64,
        )
        pan_yaw = math.radians(
            float(
                random_state.uniform(*YCB08_PAN_YAW_RANGE_DEG)
                if randomize
                else 180.0
            )
        )
        pan_rotation_xy = np.asarray(
            [
                [math.cos(pan_yaw), -math.sin(pan_yaw)],
                [math.sin(pan_yaw), math.cos(pan_yaw)],
            ],
            dtype=np.float64,
        )
        pan_xy = bowl_center - pan_rotation_xy @ YCB08_PAN_CENTER_LOCAL_XY_M
        pan_z = (
            float(table_top_z)
            - float(YCB08_SKILLET_LOCAL_MIN_M[2])
            + YCB08_RESET_SUPPORT_CLEARANCE_M
        )
        pan_pose = np.concatenate(
            [pan_xy, [pan_z], yaw_quaternion_wxyz(pan_yaw)]
        )

        sponge_local_jitter = np.asarray(
            _sample_value(
                random_state,
                -YCB08_SPONGE_LOCAL_JITTER_XY_M,
                YCB08_SPONGE_LOCAL_JITTER_XY_M,
                randomize=randomize,
            ),
            dtype=np.float64,
        )
        sponge_local_xy = (
            YCB08_PAN_CENTER_LOCAL_XY_M
            + YCB08_SPONGE_INSERTION_RUNWAY_OFFSET_LOCAL_XY_M
            + sponge_local_jitter
        )
        sponge_xy = pan_xy + pan_rotation_xy @ sponge_local_xy
        sponge_yaw = pan_yaw + math.radians(
            float(
                _sample_value(
                    random_state,
                    -YCB08_SPONGE_YAW_JITTER_DEG,
                    YCB08_SPONGE_YAW_JITTER_DEG,
                    randomize=randomize,
                )
            )
        )
        sponge_z = (
            pan_z
            + YCB08_PAN_FLOOR_TOP_LOCAL_Z_M
            + YCB08_SPONGE_HALF_EXTENTS_M[2]
            + YCB08_RESET_SUPPORT_CLEARANCE_M
        )
        sponge_pose = np.concatenate(
            [sponge_xy, [sponge_z], yaw_quaternion_wxyz(sponge_yaw)]
        )

        spatula_xy = np.asarray(
            _sample_value(
                random_state,
                YCB08_SPATULA_CENTER_XY_BOUNDS_M[0],
                YCB08_SPATULA_CENTER_XY_BOUNDS_M[1],
                randomize=randomize,
            ),
            dtype=np.float64,
        )
        spatula_yaw = math.radians(
            float(
                random_state.uniform(*YCB08_SPATULA_YAW_RANGE_DEG)
                if randomize
                else 0.0
            )
        )
        spatula_face_up = bool(
            randomize
            and float(random_state.uniform(0.0, 1.0))
            < YCB08_SPATULA_FACE_UP_PROBABILITY
        )
        spatula_pitch = math.radians(
            YCB08_SPATULA_FACE_UP_PITCH_DEG
            if spatula_face_up
            else YCB08_SPATULA_TABLE_PITCH_DEG
        )
        spatula_support_z = (
            YCB08_SPATULA_FACE_UP_SUPPORT_LOCAL_Z_M
            if spatula_face_up
            else YCB08_SPATULA_TABLE_SUPPORT_LOCAL_Z_M
        )
        spatula_z = (
            float(table_top_z)
            - spatula_support_z
            + YCB08_RESET_SUPPORT_CLEARANCE_M
        )
        spatula_quaternion = (
            yaw_face_up_spatula_quaternion_wxyz(
                spatula_yaw, spatula_pitch
            )
            if spatula_face_up
            else yaw_ready_spatula_quaternion_wxyz(
                spatula_yaw, spatula_pitch
            )
        )
        spatula_pose = np.concatenate(
            [
                spatula_xy,
                [spatula_z],
                spatula_quaternion,
            ]
        )
        layout = {
            YCB08_SKILLET_NAME: pan_pose,
            YCB08_SPATULA_NAME: spatula_pose,
            YCB08_SPONGE_NAME: sponge_pose,
        }
        if ycb08_layout_is_valid(layout):
            return {name: pose.astype(np.float64, copy=True) for name, pose in layout.items()}
    raise RuntimeError("YCB08 could not sample a collision-clear reachable reset layout")


def ycb08_flip_angle_degrees(
    sponge_pose: np.ndarray,
    reset_face_normal_world: np.ndarray,
) -> float:
    current_normal = quaternion_wxyz_to_matrix(
        np.asarray(sponge_pose, dtype=np.float64).reshape(7)[3:7]
    )[:, 2]
    initial = np.asarray(reset_face_normal_world, dtype=np.float64).reshape(3)
    initial /= max(float(np.linalg.norm(initial)), 1.0e-12)
    cosine = float(np.clip(np.dot(current_normal, initial), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def ycb08_pan_tilt_degrees(pan_pose: np.ndarray) -> float:
    normal = quaternion_wxyz_to_matrix(
        np.asarray(pan_pose, dtype=np.float64).reshape(7)[3:7]
    )[:, 2]
    cosine = float(np.clip(normal[2], -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def initial_ycb08_left_pan_lock_state() -> dict[str, Any]:
    return {
        "active": False,
        "ready_s": 0.0,
        "unstable_s": 0.0,
    }


def ycb08_left_pan_lock_ready(sample: Mapping[str, Any]) -> bool:
    """Return whether a physical pan hold is ready to be latched."""

    return bool(
        sample.get("finite", True)
        and sample.get("left_pan_contact_available", False)
        and sample.get("left_hand_pan_contact", False)
        and float(sample.get("pan_lift_m", 0.0))
        >= YCB08_LEFT_PAN_LOCK_MIN_LIFT_M
        and float(sample.get("pan_tilt_deg", math.inf))
        <= YCB08_LEFT_PAN_LOCK_MAX_TILT_DEG
        and float(sample.get("pan_linear_speed_mps", math.inf))
        <= YCB08_LEFT_PAN_LOCK_MAX_LINEAR_SPEED_MPS
        and float(sample.get("pan_angular_speed_radps", math.inf))
        <= YCB08_LEFT_PAN_LOCK_MAX_ANGULAR_SPEED_RADPS
    )


def update_ycb08_left_pan_lock_state(
    state: dict[str, Any],
    sample: Mapping[str, Any],
    dt: float,
) -> dict[str, Any]:
    """Hold a stable grasp, releasing it when physical recovery is needed."""

    if bool(state.get("active", False)):
        hold_ok = bool(
            sample.get("finite", True)
            and sample.get("left_pan_contact_available", False)
            and sample.get("left_hand_pan_contact", False)
            and float(sample.get("pan_lift_m", 0.0))
            >= 0.5 * YCB08_LEFT_PAN_LOCK_MIN_LIFT_M
            and float(sample.get("pan_tilt_deg", math.inf))
            <= YCB08_MAX_FINAL_PAN_TILT_DEG
        )
        state["unstable_s"] = (
            0.0 if hold_ok else float(state.get("unstable_s", 0.0)) + max(0.0, float(dt))
        )
        if state["unstable_s"] + 1.0e-12 >= YCB08_LEFT_PAN_UNLOCK_DEBOUNCE_SECONDS:
            state.update(active=False, ready_s=0.0, unstable_s=0.0)
        return state
    state["unstable_s"] = 0.0
    if ycb08_left_pan_lock_ready(sample):
        state["ready_s"] = float(state.get("ready_s", 0.0)) + max(
            0.0, float(dt)
        )
    else:
        state["ready_s"] = 0.0
    if (
        float(state["ready_s"]) + 1.0e-12
        >= YCB08_LEFT_PAN_LOCK_DEBOUNCE_SECONDS
    ):
        state["active"] = True
    return state


def ycb08_locked_teleop_action(
    action: Mapping[str, np.ndarray],
    frozen_left_hand: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return the exact action used for a latched, replayable left-arm hold."""

    conditioned = {
        key: np.asarray(value, dtype=np.float64).reshape(-1).copy()
        for key, value in action.items()
    }
    if "left" not in conditioned or conditioned["left"].size < 7:
        raise ValueError("YCB08 left-pan hold requires a 7D left pose action.")
    left = conditioned["left"]
    hand_size = left.size - 7
    frozen = (
        np.asarray(frozen_left_hand, dtype=np.float64).reshape(-1).copy()
        if frozen_left_hand is not None
        else left[7:].copy()
    )
    if frozen.size != hand_size:
        frozen = left[7:].copy()
    left[:7] = 0.0
    left[7:] = frozen
    return conditioned, frozen


def _sponge_sample_points_world(sponge_pose: np.ndarray, grid_size: int = 5) -> np.ndarray:
    pose = np.asarray(sponge_pose, dtype=np.float64).reshape(7)
    rotation = quaternion_wxyz_to_matrix(pose[3:7])
    xs = np.linspace(-YCB08_SPONGE_HALF_EXTENTS_M[0], YCB08_SPONGE_HALF_EXTENTS_M[0], grid_size)
    ys = np.linspace(-YCB08_SPONGE_HALF_EXTENTS_M[1], YCB08_SPONGE_HALF_EXTENTS_M[1], grid_size)
    local = np.asarray([(x, y, 0.0) for x in xs for y in ys], dtype=np.float64)
    return pose[:3] + local @ rotation.T


def ycb08_pan_containment_metrics(
    sponge_pose: np.ndarray,
    pan_pose: np.ndarray,
) -> dict[str, float | bool | np.ndarray]:
    pan = np.asarray(pan_pose, dtype=np.float64).reshape(7)
    pan_rotation = quaternion_wxyz_to_matrix(pan[3:7])
    points_world = _sponge_sample_points_world(sponge_pose)
    points_local = (points_world - pan[:3]) @ pan_rotation
    radii = np.linalg.norm(
        points_local[:, :2] - YCB08_PAN_CENTER_LOCAL_XY_M.reshape(1, 2), axis=1
    )
    inside = radii <= YCB08_PAN_INNER_RADIUS_M
    fraction = float(np.mean(inside))
    rim_clear = bool(
        float(np.max(radii))
        <= YCB08_PAN_INNER_RADIUS_M - YCB08_PAN_RIM_CLEARANCE_M
    )
    sponge_local = inverse_transform_point(pan, np.asarray(sponge_pose)[:3])
    floor_gap = float(
        sponge_local[2]
        - YCB08_SPONGE_HALF_EXTENTS_M[2]
        - YCB08_PAN_FLOOR_TOP_LOCAL_Z_M
    )
    return {
        "sponge_in_pan_position": sponge_local,
        "containment_fraction": fraction,
        "max_footprint_radius_m": float(np.max(radii)),
        "rim_clear": rim_clear,
        "com_below_rim": bool(sponge_local[2] < YCB08_PAN_RIM_TOP_LOCAL_Z_M),
        "floor_gap_m": floor_gap,
    }


def ycb08_blade_overlap_metrics(
    sponge_pose: np.ndarray,
    spatula_pose: np.ndarray,
) -> dict[str, float | bool | np.ndarray]:
    spatula = np.asarray(spatula_pose, dtype=np.float64).reshape(7)
    spatula_rotation = quaternion_wxyz_to_matrix(spatula[3:7])
    points_world = _sponge_sample_points_world(sponge_pose)
    points_local = (points_world - spatula[:3]) @ spatula_rotation
    inside = (
        (points_local[:, 0] >= YCB08_SPATULA_BLADE_X_RANGE_M[0])
        & (points_local[:, 0] <= YCB08_SPATULA_BLADE_X_RANGE_M[1])
        & (points_local[:, 1] >= YCB08_SPATULA_BLADE_Y_RANGE_M[0])
        & (points_local[:, 1] <= YCB08_SPATULA_BLADE_Y_RANGE_M[1])
    )
    blade_reference = transform_point(spatula, YCB08_SPATULA_BLADE_REFERENCE_LOCAL_M)
    sponge_center = np.asarray(sponge_pose, dtype=np.float64).reshape(7)[:3]
    return {
        "blade_overlap_fraction": float(np.mean(inside)),
        "blade_reference_world": blade_reference,
        "blade_below_sponge": bool(blade_reference[2] <= sponge_center[2] - 0.002),
    }


def initial_ycb08_state() -> dict[str, Any]:
    return {
        "settle_s": 0.0,
        "tool_grasp_seen": False,
        "tool_moved_seen": False,
        "tool_contact_seen": False,
        "tool_under_seen": False,
        "lift_seen": False,
        "flip_attempt_active": False,
        "flip_attempt_crossed_90": False,
        "flip_attempt_max_pan_tilt_deg": 0.0,
        "crossed_90_seen": False,
        "valid_flip_seen": False,
        "max_flip_angle_deg": 0.0,
        "direct_hand_contact_s": 0.0,
        "outside_support_s": 0.0,
        "pan_overturn_s": 0.0,
        "excessive_velocity_s": 0.0,
        "tool_release_s": 0.0,
        "final_stability_s": 0.0,
        "direct_hand_violation": False,
        "outside_support_violation": False,
        "pan_overturn_violation": False,
        "success": False,
        "failure_reason": None,
    }


def _fail_ycb08(state: dict[str, Any], reason: str) -> None:
    if state.get("failure_reason") is None:
        state["failure_reason"] = str(reason)
    state["success"] = False


def ycb08_final_checks(
    state: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, bool]:
    """Return every live gate that must stay true during the final hold."""
    history_complete = all(
        bool(state.get(key, False))
        for key in (
            "valid_flip_seen",
            "tool_grasp_seen",
            "tool_moved_seen",
            "tool_contact_seen",
            "tool_under_seen",
            "lift_seen",
        )
    )
    return {
        "history_complete": history_complete,
        "opposite_face_now": (
            float(sample.get("flip_angle_deg", 0.0))
            >= YCB08_FINAL_FLIP_ANGLE_DEG
        ),
        "containment_ok": (
            float(sample.get("containment_fraction", 0.0))
            >= YCB08_MIN_CONTAINMENT_FRACTION
        ),
        "pan_support_ok": bool(sample.get("pan_support_ok", False)),
        "pan_level_now": (
            float(sample.get("pan_tilt_deg", math.inf))
            <= YCB08_MAX_FINAL_PAN_TILT_DEG
        ),
        "sponge_linear_speed_ok": (
            float(sample.get("sponge_linear_speed_mps", math.inf))
            <= YCB08_MAX_FINAL_LINEAR_SPEED_MPS
        ),
        "sponge_angular_speed_ok": (
            float(sample.get("sponge_angular_speed_radps", math.inf))
            <= YCB08_MAX_FINAL_ANGULAR_SPEED_RADPS
        ),
        "tool_released": (
            float(state.get("tool_release_s", 0.0)) + 1.0e-12
            >= YCB08_TOOL_RELEASE_SECONDS
        ),
    }


def update_ycb08_state(
    state: dict[str, Any],
    sample: dict[str, Any],
    dt: float,
) -> dict[str, Any]:
    """Advance the fail-closed YCB08 event state by one physics substep."""
    dt = max(0.0, float(dt))
    if state.get("failure_reason") is not None or bool(state.get("success")):
        return state

    finite = bool(sample.get("finite", True))
    workspace_ok = bool(sample.get("workspace_ok", True))
    max_linear_speed = float(sample.get("max_linear_speed_mps", 0.0))
    max_angular_speed = float(sample.get("max_angular_speed_radps", 0.0))
    if not finite:
        _fail_ycb08(state, "non_finite_state")
        return state
    if not workspace_ok:
        _fail_ycb08(state, "task_object_exited_workspace")
        return state
    excessive_velocity = bool(
        max_linear_speed > YCB08_MAX_LINEAR_SPEED_MPS
        or max_angular_speed > YCB08_MAX_ANGULAR_SPEED_RADPS
    )
    state["excessive_velocity_s"] = (
        float(state.get("excessive_velocity_s", 0.0)) + dt
        if excessive_velocity
        else 0.0
    )
    if (
        state["excessive_velocity_s"] + 1.0e-12
        >= YCB08_EXCESSIVE_VELOCITY_DEBOUNCE_SECONDS
    ):
        _fail_ycb08(state, "excessive_object_velocity")
        return state

    state["settle_s"] = float(state.get("settle_s", 0.0)) + dt
    if state["settle_s"] < YCB08_SETTLE_SECONDS:
        return state

    direct_hand_contact = bool(sample.get("direct_hand_sponge_contact", False))
    state["direct_hand_contact_s"] = (
        float(state.get("direct_hand_contact_s", 0.0)) + dt
        if direct_hand_contact
        else 0.0
    )
    if state["direct_hand_contact_s"] + 1.0e-12 >= YCB08_DIRECT_HAND_DEBOUNCE_SECONDS:
        state["direct_hand_violation"] = True
        _fail_ycb08(state, "direct_hand_touched_sponge")
        return state

    outside_support = bool(
        sample.get("sponge_table_contact", False)
        and float(sample.get("containment_fraction", 0.0))
        < YCB08_MIN_CONTAINMENT_FRACTION
    )
    state["outside_support_s"] = (
        float(state.get("outside_support_s", 0.0)) + dt if outside_support else 0.0
    )
    if state["outside_support_s"] + 1.0e-12 >= YCB08_OUTSIDE_SUPPORT_DEBOUNCE_SECONDS:
        state["outside_support_violation"] = True
        _fail_ycb08(state, "sponge_supported_outside_pan")
        return state

    pan_tilt = float(sample.get("pan_tilt_deg", 0.0))
    state["pan_overturn_s"] = (
        float(state.get("pan_overturn_s", 0.0)) + dt
        if pan_tilt > YCB08_PAN_OVERTURN_ANGLE_DEG
        else 0.0
    )
    if state["pan_overturn_s"] + 1.0e-12 >= YCB08_PAN_OVERTURN_DEBOUNCE_SECONDS:
        state["pan_overturn_violation"] = True
        _fail_ycb08(state, "skillet_overturned")
        return state

    if bool(sample.get("spatula_right_hand_contact", False)):
        state["tool_grasp_seen"] = True
    if (
        bool(state.get("tool_grasp_seen"))
        and float(sample.get("spatula_move_distance_m", 0.0))
        >= YCB08_TOOL_MOVE_DISTANCE_M
    ):
        state["tool_moved_seen"] = True

    tool_contact = bool(sample.get("spatula_sponge_contact", False))
    if tool_contact and bool(state.get("tool_moved_seen")):
        state["tool_contact_seen"] = True
    under_now = bool(
        tool_contact
        and float(sample.get("blade_overlap_fraction", 0.0))
        >= YCB08_MIN_BLADE_OVERLAP
        and bool(sample.get("blade_below_sponge", False))
    )
    if under_now and bool(state.get("tool_contact_seen")):
        state["tool_under_seen"] = True
        state["flip_attempt_active"] = True
    if (
        bool(state.get("tool_under_seen"))
        and float(sample.get("sponge_lift_m", 0.0)) >= YCB08_MIN_SPONGE_LIFT_M
    ):
        state["lift_seen"] = True

    flip_angle = float(sample.get("flip_angle_deg", 0.0))
    state["max_flip_angle_deg"] = max(
        float(state.get("max_flip_angle_deg", 0.0)), flip_angle
    )
    if (
        bool(state.get("flip_attempt_active"))
        and bool(state.get("flip_attempt_crossed_90"))
        and flip_angle <= YCB08_FLIP_ATTEMPT_RESET_ANGLE_DEG
    ):
        # A returned, incomplete excursion is recoverable.  Start a fresh
        # continuous excursion without demanding another near-flip contact:
        # the operator may have established physical blade use earlier.
        state["flip_attempt_crossed_90"] = False
        state["flip_attempt_max_pan_tilt_deg"] = pan_tilt
    if bool(state.get("flip_attempt_active")):
        state["flip_attempt_max_pan_tilt_deg"] = max(
            float(state.get("flip_attempt_max_pan_tilt_deg", 0.0)), pan_tilt
        )
        if flip_angle >= YCB08_FLIP_CROSS_ANGLE_DEG:
            state["crossed_90_seen"] = True
            state["flip_attempt_crossed_90"] = True
        if (
            bool(state.get("flip_attempt_crossed_90"))
            and flip_angle >= YCB08_FINAL_FLIP_ANGLE_DEG
            and float(state.get("flip_attempt_max_pan_tilt_deg", 0.0))
            <= YCB08_MAX_PAN_TILT_DURING_FLIP_DEG
            and bool(state.get("lift_seen"))
        ):
            state["valid_flip_seen"] = True

    state["tool_release_s"] = (
        float(state.get("tool_release_s", 0.0)) + dt if not tool_contact else 0.0
    )
    pan_support_ok = bool(
        sample.get(
            "pan_support_ok",
            sample.get("pan_sponge_contact", False)
            and sample.get("rim_clear", False)
            and sample.get("com_below_rim", False)
            and not sample.get("sponge_table_contact", False)
            and not tool_contact
            and not direct_hand_contact,
        )
    )
    final_checks = ycb08_final_checks(
        state,
        {**sample, "pan_support_ok": pan_support_ok},
    )
    final_now = all(final_checks[name] for name in YCB08_FINAL_CHECK_NAMES)
    state["final_stability_s"] = (
        float(state.get("final_stability_s", 0.0)) + dt if final_now else 0.0
    )
    if state["final_stability_s"] + 1.0e-12 >= YCB08_FINAL_STABILITY_SECONDS:
        state["success"] = True
    return state


def ycb08_event_flags(state: dict[str, Any], metrics: dict[str, Any]) -> np.ndarray:
    values = {
        "tool_grasp_seen": state.get("tool_grasp_seen", False),
        "tool_moved_seen": state.get("tool_moved_seen", False),
        "tool_contact_seen": state.get("tool_contact_seen", False),
        "tool_under_seen": state.get("tool_under_seen", False),
        "lift_seen": state.get("lift_seen", False),
        "crossed_90_seen": state.get("crossed_90_seen", False),
        "valid_flip_seen": state.get("valid_flip_seen", False),
        "containment_ok": float(metrics.get("containment_fraction", 0.0))
        >= YCB08_MIN_CONTAINMENT_FRACTION,
        "pan_support_ok": metrics.get("pan_support_ok", False),
        "tool_released": float(state.get("tool_release_s", 0.0))
        >= YCB08_TOOL_RELEASE_SECONDS,
        "stable_final": float(state.get("final_stability_s", 0.0))
        >= YCB08_FINAL_STABILITY_SECONDS,
        "direct_hand_violation": state.get("direct_hand_violation", False),
        "outside_support_violation": state.get("outside_support_violation", False),
        "pan_overturn_violation": state.get("pan_overturn_violation", False),
        "failure": state.get("failure_reason") is not None,
        "success": state.get("success", False),
    }
    return np.asarray([float(bool(values[name])) for name in YCB08_EVENT_FLAG_NAMES], dtype=np.float64)


def ycb08_hidden_state(state: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            state.get("settle_s", 0.0),
            state.get("direct_hand_contact_s", 0.0),
            state.get("outside_support_s", 0.0),
            state.get("pan_overturn_s", 0.0),
            state.get("tool_release_s", 0.0),
            state.get("final_stability_s", 0.0),
            state.get("flip_attempt_max_pan_tilt_deg", 0.0),
            state.get("max_flip_angle_deg", 0.0),
            float(bool(state.get("flip_attempt_active", False))),
            float(bool(state.get("flip_attempt_crossed_90", False))),
            float(YCB08_FAILURE_CODES.get(state.get("failure_reason"), -1)),
            state.get("excessive_velocity_s", 0.0),
        ],
        dtype=np.float64,
    )


__all__ = [name for name in globals() if name.startswith("YCB08_")] + [
    "initial_ycb08_left_pan_lock_state",
    "initial_ycb08_state",
    "inverse_transform_point",
    "quaternion_wxyz_to_matrix",
    "sample_ycb08_layout",
    "transform_point",
    "update_ycb08_state",
    "update_ycb08_left_pan_lock_state",
    "yaw_face_up_spatula_quaternion_wxyz",
    "yaw_ready_spatula_quaternion_wxyz",
    "yaw_quaternion_wxyz",
    "ycb08_blade_overlap_metrics",
    "ycb08_event_flags",
    "ycb08_flip_angle_degrees",
    "ycb08_hidden_state",
    "ycb08_layout_is_valid",
    "ycb08_left_pan_lock_ready",
    "ycb08_locked_teleop_action",
    "ycb08_pan_containment_metrics",
    "ycb08_pan_tilt_degrees",
    "ycb08_spatula_is_face_up",
]
