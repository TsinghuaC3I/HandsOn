"""Native IsaacLab environment for B320-BC005 Loaded Tray Through Narrow Gate."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from .abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _randomized_physx_pose,
)
from .bimanual_assembly_env import _end_effector_link_jacobian
from .loaded_tray_gate_contract import (
    LoadedTrayFrame,
    LoadedTrayGateEvaluator,
    LoadedTrayStage,
    cylinder_axis_extent,
    pose_matrix,
    quat_wxyz_to_matrix,
    rpy_degrees,
    world_aabb_half_extents,
)
from .loaded_tray_gate_scene import (
    GATE_BASE_Z,
    GATE_CLEAR_WIDTH,
    GATE_COLLIDER_BOXES,
    GATE_OUTER_WIDTH,
    GATE_PLANE_X,
    GATE_POST_HALF_X,
    GATE_VISUAL_Y_SCALE,
    LEFT_GRIP_LOCAL_POSITION,
    PAYLOAD_HALF_HEIGHTS,
    PAYLOAD_HORIZONTAL_HALF_EXTENTS,
    PAYLOAD_HORIZONTAL_RADII,
    PAYLOAD_NAMES,
    PAYLOAD_START_LOCAL_XY,
    PAYLOAD_START_POSITIONS,
    RIGHT_GRIP_LOCAL_POSITION,
    SIDE_RIM_GRIP_LOCAL_Y_RANGE,
    SIDE_RIM_GRIP_LOCAL_Z_RANGE,
    TABLE_TOP_Z,
    TARGET_CENTER_TOLERANCE,
    TARGET_CENTER_XY,
    TARGET_HALF_EXTENTS,
    TASK_ID,
    TASK_NAME,
    TRAY_DECK_TOP_LOCAL_Z,
    TRAY_GATE_CLEARANCE_M,
    TRAY_HALF_EXTENTS,
    TRAY_RESET_GATE_GAP_M,
    TRAY_START_POSITION,
    TRAY_START_YAW_DEG,
    TRAY_USABLE_HALF_EXTENTS,
)


_HAND_CONTACT_FORCE_N = 0.15
_HANDLE_EXCESSIVE_CONTACT_FORCE_N = 150.0
_HANDLE_REGION_DISTANCE_M = 0.090
_HANDLE_WRONG_REGION_MARGIN_M = 0.050
_HANDLE_CONTACT_POINT_DISTANCE_M = 0.030
_GRIP_CONTACT_LOCAL_X_MARGIN_M = 0.002
_GRIP_CONTACT_LOCAL_Y_MARGIN_M = 0.008
_GRIP_CONTACT_LOCAL_Z_MARGIN_M = 0.0005
_GATE_SEVERE_CONTACT_FORCE_N = 25.0
_PAYLOAD_GATE_CLEARANCE_M = 0.005
_PAYLOAD_DECK_BOTTOM_OFFSET_MIN_M = -0.010
_PAYLOAD_DECK_BOTTOM_OFFSET_MAX_M = 0.020
_PAYLOAD_SUPPORT_FORCE_N = 1.0e-4
_FINAL_ROLL_PITCH_DEG = 12.0
_FINAL_YAW_DEG = 15.0
_FINAL_TRAY_LINEAR_SPEED_MPS = 0.080
_FINAL_TRAY_ANGULAR_SPEED_RADPS = 0.60
_FINAL_PAYLOAD_LINEAR_SPEED_MPS = 0.10
_PAYLOAD_LOCAL_FOOTPRINT_TOLERANCE_M = 0.006
_PAYLOAD_RESET_XY_JITTER_M = 0.005
_RESET_LAYOUT_MAX_ATTEMPTS = 64
_RESET_GATE_X_JITTER_M = 0.008
_RESET_GATE_Y_JITTER_M = 0.030
_RESET_GATE_YAW_JITTER_DEG = 4.0
_RESET_TRAY_LATERAL_JITTER_M = 0.018
_RESET_TRAY_YAW_JITTER_DEG = 4.0
_RESET_TRAY_EXTRA_GAP_M = 0.010
_RESET_GATE_CLEARANCE_M = 0.010
_RESET_PAYLOAD_PAIR_CLEARANCE_M = 0.008
_RESET_TABLE_X_BOUNDS = (-0.700, 0.250)
_RESET_TABLE_Y_BOUNDS = (-0.700, 0.700)
_RESET_CAMERA_X_BOUNDS = (-0.720, 0.280)
_RESET_CAMERA_Y_BOUNDS = (-0.520, 0.520)
_RESET_GRIP_X_BOUNDS = (-0.620, -0.250)
_RESET_GRIP_ABS_Y_MAX_M = 0.350
_PAYLOAD_UPRIGHT_COS = math.cos(math.radians(35.0))
_TILT_STABILIZER_PROPORTIONAL_NM_PER_RAD = 4.00
_TILT_STABILIZER_DAMPING_NM_PER_RADPS = 0.20
_TILT_STABILIZER_MAX_TORQUE_NM = 1.20
_TILT_STABILIZER_MIN_DUAL_GRASP_FRAMES = 3
_TRANSPORT_STABILIZER_PROPORTIONAL_N_PER_M = 120.0
_TRANSPORT_STABILIZER_DAMPING_N_PER_MPS = 2.0
_TRANSPORT_STABILIZER_MAX_FORCE_N = 8.0
_TRANSPORT_STABILIZER_MIN_LIFT_M = 0.020
_TRANSPORT_TRACKING_FULL_FORCE_ERROR_M = 0.060
_TRANSPORT_TRACKING_ZERO_FORCE_ERROR_M = 0.100
_CROSSING_NULLSPACE_GAIN = 0.20
_CROSSING_NULLSPACE_MAX_STEP_RAD = 0.030
_WORKSPACE_LOWER = np.asarray([-0.65, -0.75, 0.78], dtype=np.float64)
_WORKSPACE_UPPER = np.asarray([0.25, 0.75, 1.55], dtype=np.float64)
_AUTHORED_PAYLOAD_LOCAL_XY = {
    name: np.asarray(PAYLOAD_START_LOCAL_XY[name], dtype=np.float64)
    for name in PAYLOAD_NAMES
}
_AUTHORED_PAYLOAD_Z = {
    name: float(PAYLOAD_START_POSITIONS[name][2]) for name in PAYLOAD_NAMES
}
_BC005_PRODUCTION_ASSET_DIR = (
    Path(__file__).resolve().parents[1]
    / "envs"
    / "usd"
    / "loaded_tray_gate_assets"
)
_BC005_TRAY_ASSET_PATH = (
    _BC005_PRODUCTION_ASSET_DIR / "bc005_selected_wooden_tray.usd"
)
_BC005_FRAME_ASSET_PATH = (
    _BC005_PRODUCTION_ASSET_DIR / "bc005_selected_wooden_frame.usd"
)
_BC005_TRAY_REFERENCE_CHILD = "BC005ProductionTray"
_BC005_FRAME_REFERENCE_CHILD = "BC005ProductionWoodenFrame"
_BC005_EXPECTED_TRAY_VISUAL_MESH_COUNT = 1
_BC005_EXPECTED_TRAY_COLLISION_BOX_COUNT = 5
_BC005_EXPECTED_TRAY_COLLISION_MESH_COUNT = 12
_BC005_EXPECTED_TRAY_COLLISION_API_COUNT = 17
_BC005_EXPECTED_TRAY_PHYSICS_MATERIAL_COUNT = 2
_BC005_EXPECTED_FRAME_VISUAL_MESH_COUNT = 1
_BC005_EXPECTED_FRAME_COLLISION_BOX_COUNT = 3
_BC005_EXPECTED_TRAY_PHYSICS_MATERIAL_PROPERTIES = {
    "Body": (1.10, 1.00, 0.0),
    "Handle": (1.35, 1.20, 0.0),
}


def _wrap_degrees(angle_degrees: float) -> float:
    return (float(angle_degrees) + 180.0) % 360.0 - 180.0


def _planar_axes(yaw_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    """Return forward/lateral unit axes for one world-frame planar yaw."""
    yaw = math.radians(float(yaw_degrees))
    forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    lateral = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    return forward, lateral


def _yaw_quaternion_xyzw(yaw_degrees: float) -> np.ndarray:
    half = 0.5 * math.radians(float(yaw_degrees))
    return np.asarray([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float64)


def _yaw_quaternion_wxyz(yaw_degrees: float) -> np.ndarray:
    xyzw = _yaw_quaternion_xyzw(yaw_degrees)
    return xyzw[[3, 0, 1, 2]]


def _box_half_extent_along_planar_axis(
    half_extents_xy: tuple[float, float] | np.ndarray,
    yaw_degrees: float,
    axis: np.ndarray,
) -> float:
    object_forward, object_lateral = _planar_axes(yaw_degrees)
    half = np.asarray(half_extents_xy, dtype=np.float64).reshape(2)
    unit_axis = np.asarray(axis, dtype=np.float64).reshape(2)
    return float(
        abs(float(np.dot(unit_axis, object_forward))) * half[0]
        + abs(float(np.dot(unit_axis, object_lateral))) * half[1]
    )


def _oriented_box_half_extents_in_frame(
    pose_wxyz: np.ndarray,
    local_half_extents: np.ndarray,
    frame_pose_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return box centre and conservative half extents in another frame."""
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    frame_pose = np.asarray(frame_pose_wxyz, dtype=np.float64).reshape(7)
    object_rotation = quat_wxyz_to_matrix(pose[3:7])
    frame_rotation = quat_wxyz_to_matrix(frame_pose[3:7])
    if not (
        np.all(np.isfinite(object_rotation))
        and np.all(np.isfinite(frame_rotation))
    ):
        return (
            np.full(3, np.nan, dtype=np.float64),
            np.full(3, np.inf, dtype=np.float64),
        )
    centre = frame_rotation.T @ (pose[:3] - frame_pose[:3])
    half = (
        np.abs(frame_rotation.T @ object_rotation)
        @ np.asarray(local_half_extents, dtype=np.float64).reshape(3)
    )
    return centre, half


def _cylinder_half_extents_in_frame(
    pose_wxyz: np.ndarray,
    *,
    radius: float,
    half_height: float,
    frame_pose_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a conservative cylinder AABB in an arbitrary gate frame."""
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    frame_pose = np.asarray(frame_pose_wxyz, dtype=np.float64).reshape(7)
    object_rotation = quat_wxyz_to_matrix(pose[3:7])
    frame_rotation = quat_wxyz_to_matrix(frame_pose[3:7])
    if not (
        np.all(np.isfinite(object_rotation))
        and np.all(np.isfinite(frame_rotation))
    ):
        return (
            np.full(3, np.nan, dtype=np.float64),
            np.full(3, np.inf, dtype=np.float64),
        )
    centre = frame_rotation.T @ (pose[:3] - frame_pose[:3])
    cylinder_axis = object_rotation[:, 2]
    half = np.empty(3, dtype=np.float64)
    for index in range(3):
        component = float(
            np.clip(abs(np.dot(frame_rotation[:, index], cylinder_axis)), 0.0, 1.0)
        )
        half[index] = (
            float(half_height) * component
            + float(radius) * math.sqrt(max(0.0, 1.0 - component * component))
        )
    return centre, half


def _bc005_fallback_reset_layout() -> dict[str, Any]:
    gate_xy = np.asarray([GATE_PLANE_X, 0.0], dtype=np.float64)
    gate_forward, _gate_lateral = _planar_axes(0.0)
    tray_yaw_deg = TRAY_START_YAW_DEG
    tray_forward_extent = _box_half_extent_along_planar_axis(
        TRAY_HALF_EXTENTS[:2], tray_yaw_deg, gate_forward
    )
    tray_xy = gate_xy - gate_forward * (
        GATE_POST_HALF_X + TRAY_RESET_GATE_GAP_M + tray_forward_extent
    )
    return {
        "gate_xy": gate_xy,
        "gate_yaw_deg": 0.0,
        "tray_xy": tray_xy,
        "tray_yaw_deg": tray_yaw_deg,
        "goal_xy": np.asarray(TARGET_CENTER_XY, dtype=np.float64),
        "goal_yaw_deg": 0.0,
        "payload_local_xy": {
            name: value.copy() for name, value in _AUTHORED_PAYLOAD_LOCAL_XY.items()
        },
        "attempts": 0,
        "fallback": True,
    }


def _bc005_reset_layout_valid(layout: Mapping[str, Any]) -> bool:
    """Validate support, reach, separation, path, and camera reset bounds."""
    try:
        gate_xy = np.asarray(layout["gate_xy"], dtype=np.float64).reshape(2)
        tray_xy = np.asarray(layout["tray_xy"], dtype=np.float64).reshape(2)
        goal_xy = np.asarray(layout["goal_xy"], dtype=np.float64).reshape(2)
        gate_yaw_deg = float(layout["gate_yaw_deg"])
        tray_yaw_deg = float(layout["tray_yaw_deg"])
        payload_local_xy = {
            name: np.asarray(layout["payload_local_xy"][name], dtype=np.float64).reshape(2)
            for name in PAYLOAD_NAMES
        }
    except (KeyError, TypeError, ValueError):
        return False
    scalars = np.concatenate(
        [gate_xy, tray_xy, goal_xy, [gate_yaw_deg, tray_yaw_deg]]
    )
    if not np.all(np.isfinite(scalars)):
        return False

    gate_forward, gate_lateral = _planar_axes(gate_yaw_deg)
    relative_tray = tray_xy - gate_xy
    tray_forward_extent = _box_half_extent_along_planar_axis(
        TRAY_HALF_EXTENTS[:2], tray_yaw_deg, gate_forward
    )
    tray_lateral_extent = _box_half_extent_along_planar_axis(
        TRAY_HALF_EXTENTS[:2], tray_yaw_deg, gate_lateral
    )
    reset_gap = float(
        -GATE_POST_HALF_X
        - (np.dot(relative_tray, gate_forward) + tray_forward_extent)
    )
    if not (
        TRAY_RESET_GATE_GAP_M - 1.0e-9
        <= reset_gap
        <= TRAY_RESET_GATE_GAP_M + _RESET_TRAY_EXTRA_GAP_M + 1.0e-9
    ):
        return False
    if (
        abs(float(np.dot(relative_tray, gate_lateral)))
        + tray_lateral_extent
        + _RESET_GATE_CLEARANCE_M
        > 0.5 * GATE_CLEAR_WIDTH
    ):
        return False

    # Complete tray footprint must stay on the tabletop and in the head-camera
    # envelope.  The two grip points are also checked against the audited
    # bimanual reach band, not just the tray centre.
    world_x_extent = _box_half_extent_along_planar_axis(
        TRAY_HALF_EXTENTS[:2], tray_yaw_deg, np.asarray([1.0, 0.0])
    )
    world_y_extent = _box_half_extent_along_planar_axis(
        TRAY_HALF_EXTENTS[:2], tray_yaw_deg, np.asarray([0.0, 1.0])
    )
    if not (
        _RESET_TABLE_X_BOUNDS[0] <= tray_xy[0] - world_x_extent
        and tray_xy[0] + world_x_extent <= _RESET_TABLE_X_BOUNDS[1]
        and _RESET_TABLE_Y_BOUNDS[0] <= tray_xy[1] - world_y_extent
        and tray_xy[1] + world_y_extent <= _RESET_TABLE_Y_BOUNDS[1]
    ):
        return False
    tray_forward, tray_lateral = _planar_axes(tray_yaw_deg)
    grip_points = (
        tray_xy
        + tray_forward * LEFT_GRIP_LOCAL_POSITION[0]
        + tray_lateral * LEFT_GRIP_LOCAL_POSITION[1],
        tray_xy
        + tray_forward * RIGHT_GRIP_LOCAL_POSITION[0]
        + tray_lateral * RIGHT_GRIP_LOCAL_POSITION[1],
    )
    if any(
        not (
            _RESET_GRIP_X_BOUNDS[0] <= point[0] <= _RESET_GRIP_X_BOUNDS[1]
            and abs(float(point[1])) <= _RESET_GRIP_ABS_Y_MAX_M
        )
        for point in grip_points
    ):
        return False

    # Both payloads must remain inside the source-derived usable deck and
    # retain a real footprint clearance from each other.
    for name in PAYLOAD_NAMES:
        local = payload_local_xy[name]
        half = np.asarray(PAYLOAD_HORIZONTAL_HALF_EXTENTS[name], dtype=np.float64)
        if np.any(
            np.abs(local) + half
            > np.asarray(TRAY_USABLE_HALF_EXTENTS, dtype=np.float64) - 0.002
        ):
            return False
    payload_spacing = float(
        np.linalg.norm(
            payload_local_xy[PAYLOAD_NAMES[0]]
            - payload_local_xy[PAYLOAD_NAMES[1]]
        )
    )
    if payload_spacing < (
        PAYLOAD_HORIZONTAL_RADII[PAYLOAD_NAMES[0]]
        + PAYLOAD_HORIZONTAL_RADII[PAYLOAD_NAMES[1]]
        + _RESET_PAYLOAD_PAIR_CLEARANCE_M
    ):
        return False

    # The whole narrowed fixture and goal marker stay on support and within the
    # front camera's audited reset envelope.  Use the deeper lintel X span and
    # full outer Y span under yaw.
    max_frame_half_x = max(
        abs(float(spec["center"][0])) + float(spec["half_extents"][0])
        for spec in GATE_COLLIDER_BOXES.values()
    )
    frame_half_y = 0.5 * GATE_OUTER_WIDTH
    frame_world_x_extent = _box_half_extent_along_planar_axis(
        (max_frame_half_x, frame_half_y), gate_yaw_deg, np.asarray([1.0, 0.0])
    )
    frame_world_y_extent = _box_half_extent_along_planar_axis(
        (max_frame_half_x, frame_half_y), gate_yaw_deg, np.asarray([0.0, 1.0])
    )
    goal_world_x_extent = _box_half_extent_along_planar_axis(
        TARGET_HALF_EXTENTS, gate_yaw_deg, np.asarray([1.0, 0.0])
    )
    goal_world_y_extent = _box_half_extent_along_planar_axis(
        TARGET_HALF_EXTENTS, gate_yaw_deg, np.asarray([0.0, 1.0])
    )
    if not (
        _RESET_TABLE_X_BOUNDS[0] <= gate_xy[0] - frame_world_x_extent
        and gate_xy[0] + frame_world_x_extent <= _RESET_TABLE_X_BOUNDS[1]
        and _RESET_TABLE_Y_BOUNDS[0] <= gate_xy[1] - frame_world_y_extent
        and gate_xy[1] + frame_world_y_extent <= _RESET_TABLE_Y_BOUNDS[1]
        and _RESET_TABLE_X_BOUNDS[0] <= goal_xy[0] - goal_world_x_extent
        and goal_xy[0] + goal_world_x_extent <= _RESET_TABLE_X_BOUNDS[1]
        and _RESET_TABLE_Y_BOUNDS[0] <= goal_xy[1] - goal_world_y_extent
        and goal_xy[1] + goal_world_y_extent <= _RESET_TABLE_Y_BOUNDS[1]
    ):
        return False
    camera_points = np.stack([tray_xy, gate_xy, goal_xy, *grip_points])
    if not (
        np.all(camera_points[:, 0] >= _RESET_CAMERA_X_BOUNDS[0])
        and np.all(camera_points[:, 0] <= _RESET_CAMERA_X_BOUNDS[1])
        and np.all(camera_points[:, 1] >= _RESET_CAMERA_Y_BOUNDS[0])
        and np.all(camera_points[:, 1] <= _RESET_CAMERA_Y_BOUNDS[1])
    ):
        return False
    return True


def _sample_bc005_reset_layout(random_state: Any) -> dict[str, Any]:
    """Jointly sample tray/gate poses with bounded deterministic fallback."""
    goal_forward_offset = GATE_POST_HALF_X + 0.010 + TARGET_HALF_EXTENTS[0]
    for attempt in range(1, _RESET_LAYOUT_MAX_ATTEMPTS + 1):
        gate_xy = np.asarray(
            [
                GATE_PLANE_X
                + float(random_state.uniform(-_RESET_GATE_X_JITTER_M, _RESET_GATE_X_JITTER_M)),
                float(random_state.uniform(-_RESET_GATE_Y_JITTER_M, _RESET_GATE_Y_JITTER_M)),
            ],
            dtype=np.float64,
        )
        gate_yaw_deg = float(
            random_state.uniform(-_RESET_GATE_YAW_JITTER_DEG, _RESET_GATE_YAW_JITTER_DEG)
        )
        relative_tray_yaw_deg = TRAY_START_YAW_DEG + float(
            random_state.uniform(-_RESET_TRAY_YAW_JITTER_DEG, _RESET_TRAY_YAW_JITTER_DEG)
        )
        tray_yaw_deg = gate_yaw_deg + relative_tray_yaw_deg
        gate_forward, gate_lateral = _planar_axes(gate_yaw_deg)
        tray_forward_extent = _box_half_extent_along_planar_axis(
            TRAY_HALF_EXTENTS[:2], tray_yaw_deg, gate_forward
        )
        reset_gap = TRAY_RESET_GATE_GAP_M + float(
            random_state.uniform(0.0, _RESET_TRAY_EXTRA_GAP_M)
        )
        tray_lateral_offset = float(
            random_state.uniform(
                -_RESET_TRAY_LATERAL_JITTER_M,
                _RESET_TRAY_LATERAL_JITTER_M,
            )
        )
        tray_xy = (
            gate_xy
            - gate_forward
            * (GATE_POST_HALF_X + reset_gap + tray_forward_extent)
            + gate_lateral * tray_lateral_offset
        )
        goal_xy = gate_xy + gate_forward * goal_forward_offset
        layout = {
            "gate_xy": gate_xy,
            "gate_yaw_deg": gate_yaw_deg,
            "tray_xy": tray_xy,
            "tray_yaw_deg": tray_yaw_deg,
            "goal_xy": goal_xy,
            "goal_yaw_deg": gate_yaw_deg,
            "payload_local_xy": {
                name: _AUTHORED_PAYLOAD_LOCAL_XY[name]
                + random_state.uniform(
                    -_PAYLOAD_RESET_XY_JITTER_M,
                    _PAYLOAD_RESET_XY_JITTER_M,
                    size=2,
                )
                for name in PAYLOAD_NAMES
            },
            "attempts": attempt,
            "fallback": False,
        }
        if _bc005_reset_layout_valid(layout):
            return layout
    fallback = _bc005_fallback_reset_layout()
    if not _bc005_reset_layout_valid(fallback):
        raise RuntimeError("B320-BC005 deterministic fallback layout is invalid")
    fallback["attempts"] = _RESET_LAYOUT_MAX_ATTEMPTS
    return fallback


def _point_to_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    """Return Euclidean distance to a finite line segment."""
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    segment = end - start
    squared_length = float(np.dot(segment, segment))
    if squared_length <= np.finfo(np.float64).eps:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, segment) / squared_length)
    closest = start + np.clip(fraction, 0.0, 1.0) * segment
    return float(np.linalg.norm(point - closest))


def _payload_bottom_on_deck(bottom_local_z: float) -> bool:
    """Return whether a payload bottom stays within the deck support band."""
    offset = float(bottom_local_z) - TRAY_DECK_TOP_LOCAL_Z
    return bool(
        math.isfinite(offset)
        and _PAYLOAD_DECK_BOTTOM_OFFSET_MIN_M
        <= offset
        <= _PAYLOAD_DECK_BOTTOM_OFFSET_MAX_M
    )


def _payload_retained_on_tray(
    *,
    geometrically_on_tray: bool,
    support_available: bool,
    support_force_n: float,
) -> bool:
    """Require readable tray support in addition to the geometric envelope."""
    return bool(
        geometrically_on_tray
        and support_available
        and float(support_force_n) >= _PAYLOAD_SUPPORT_FORCE_N
    )


def _grip_contact_point_in_role_region(
    point_local: np.ndarray,
    grip_local_x: float,
) -> bool:
    """Reject tray contact points outside one selected long-side rim grip."""
    point = np.asarray(point_local, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(point)):
        return False
    grip_start = np.asarray(
        (
            grip_local_x,
            SIDE_RIM_GRIP_LOCAL_Y_RANGE[0],
            LEFT_GRIP_LOCAL_POSITION[2],
        ),
        dtype=np.float64,
    )
    grip_end = grip_start.copy()
    grip_end[1] = SIDE_RIM_GRIP_LOCAL_Y_RANGE[1]
    return bool(
        abs(float(point[0]) - float(grip_local_x))
        <= _GRIP_CONTACT_LOCAL_X_MARGIN_M
        and SIDE_RIM_GRIP_LOCAL_Y_RANGE[0] - _GRIP_CONTACT_LOCAL_Y_MARGIN_M
        <= float(point[1])
        <= SIDE_RIM_GRIP_LOCAL_Y_RANGE[1] + _GRIP_CONTACT_LOCAL_Y_MARGIN_M
        and SIDE_RIM_GRIP_LOCAL_Z_RANGE[0] - _GRIP_CONTACT_LOCAL_Z_MARGIN_M
        <= float(point[2])
        <= SIDE_RIM_GRIP_LOCAL_Z_RANGE[1] + _GRIP_CONTACT_LOCAL_Z_MARGIN_M
        and _point_to_segment_distance(point, grip_start, grip_end)
        <= _HANDLE_CONTACT_POINT_DISTANCE_M
    )


def _handle_role_owned(
    *,
    contact_available: bool,
    contact_force_n: float,
    hand_closed: bool,
    handle_distance_m: float,
    wrong_handle_distance_m: float,
) -> bool:
    """Combine filtered physical contact with finite-handle proximity."""
    return bool(
        contact_available
        and float(contact_force_n) >= _HAND_CONTACT_FORCE_N
        and hand_closed
        and float(handle_distance_m) <= _HANDLE_REGION_DISTANCE_M
        and float(handle_distance_m) + _HANDLE_WRONG_REGION_MARGIN_M
        < float(wrong_handle_distance_m)
    )


def _contact_tensor_to_numpy(value: Any) -> np.ndarray:
    """Copy Torch/Warp contact data into NumPy without retaining a device view."""
    if hasattr(value, "torch"):
        value = value.torch
    elif hasattr(value, "warp"):
        value = value.warp
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value).copy()


def _bounded_tilt_stabilizing_torque(
    pose_wxyz: np.ndarray,
    velocity: np.ndarray | None,
) -> np.ndarray:
    """Return a capped world-frame roll/pitch torque with no yaw component."""
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    rotation = quat_wxyz_to_matrix(pose[3:7])
    if not np.all(np.isfinite(rotation)):
        return np.zeros(3, dtype=np.float64)
    tray_normal = rotation[:, 2]
    upright_error = np.cross(tray_normal, np.asarray([0.0, 0.0, 1.0]))
    angular_velocity = np.zeros(3, dtype=np.float64)
    if velocity is not None:
        candidate = np.asarray(velocity, dtype=np.float64).reshape(6)[3:].copy()
        if np.all(np.isfinite(candidate)):
            angular_velocity = candidate
    angular_velocity[2] = 0.0
    torque = (
        _TILT_STABILIZER_PROPORTIONAL_NM_PER_RAD * upright_error
        - _TILT_STABILIZER_DAMPING_NM_PER_RADPS * angular_velocity
    )
    torque[2] = 0.0
    magnitude = float(np.linalg.norm(torque))
    if magnitude > _TILT_STABILIZER_MAX_TORQUE_NM:
        torque *= _TILT_STABILIZER_MAX_TORQUE_NM / magnitude
    return torque


def _transport_tracking_scale(max_position_error_m: float) -> float:
    """Fade transport force out before a lagging arm can build high contact."""
    error = float(max_position_error_m)
    if not math.isfinite(error):
        return 0.0
    span = (
        _TRANSPORT_TRACKING_ZERO_FORCE_ERROR_M
        - _TRANSPORT_TRACKING_FULL_FORCE_ERROR_M
    )
    return float(
        np.clip(
            (_TRANSPORT_TRACKING_ZERO_FORCE_ERROR_M - error) / span,
            0.0,
            1.0,
        )
    )


def _bounded_transport_stabilizing_force(
    tray_x: float,
    command_midpoint_x: float,
    grasp_offset_x: float,
    velocity: np.ndarray | None,
) -> np.ndarray:
    """Return a capped world-x force that follows the latched dual-hand grip."""
    desired_x = float(command_midpoint_x) + float(grasp_offset_x)
    linear_velocity_x = 0.0
    if velocity is not None:
        candidate = np.asarray(velocity, dtype=np.float64).reshape(6)
        if np.all(np.isfinite(candidate)):
            linear_velocity_x = float(candidate[0])
    force_x = (
        _TRANSPORT_STABILIZER_PROPORTIONAL_N_PER_M
        * (desired_x - float(tray_x))
        - _TRANSPORT_STABILIZER_DAMPING_N_PER_MPS * linear_velocity_x
    )
    force_x = float(
        np.clip(
            force_x,
            -_TRANSPORT_STABILIZER_MAX_FORCE_N,
            _TRANSPORT_STABILIZER_MAX_FORCE_N,
        )
    )
    return np.asarray([force_x, 0.0, 0.0], dtype=np.float64)


def _bounded_transport_stabilizing_force_along_axis(
    tray_position: np.ndarray,
    command_midpoint: np.ndarray,
    grasp_offset_m: float,
    velocity: np.ndarray | None,
    forward_axis_xy: np.ndarray,
) -> np.ndarray:
    """Rotate the bounded transport assist into the sampled gate axis."""
    axis_xy = np.asarray(forward_axis_xy, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(axis_xy))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    axis_xy = axis_xy / norm
    tray_coordinate = float(
        np.dot(np.asarray(tray_position, dtype=np.float64).reshape(3)[:2], axis_xy)
    )
    command_coordinate = float(
        np.dot(np.asarray(command_midpoint, dtype=np.float64).reshape(3)[:2], axis_xy)
    )
    projected_velocity = None
    if velocity is not None:
        candidate = np.asarray(velocity, dtype=np.float64).reshape(6)
        projected_velocity = np.zeros(6, dtype=np.float64)
        projected_velocity[0] = float(np.dot(candidate[:2], axis_xy))
    scalar = _bounded_transport_stabilizing_force(
        tray_coordinate,
        command_coordinate,
        grasp_offset_m,
        projected_velocity,
    )[0]
    return np.asarray(
        [axis_xy[0] * scalar, axis_xy[1] * scalar, 0.0],
        dtype=np.float64,
    )


def _bounded_nullspace_posture_delta(
    torch_module: Any,
    jacobian: Any,
    current: Any,
    reference: Any,
) -> Any:
    """Return a capped first-order posture correction in Jacobian nullspace."""
    pseudoinverse = torch_module.linalg.pinv(jacobian)
    identity = torch_module.eye(
        int(jacobian.shape[-1]),
        dtype=jacobian.dtype,
        device=jacobian.device,
    ).unsqueeze(0)
    nullspace = identity - torch_module.bmm(pseudoinverse, jacobian)
    posture_error = (reference - current).unsqueeze(-1)
    correction = _CROSSING_NULLSPACE_GAIN * torch_module.bmm(
        nullspace,
        posture_error,
    ).squeeze(-1)
    correction_norm = torch_module.linalg.vector_norm(
        correction,
        dim=1,
        keepdim=True,
    )
    return correction * torch_module.clamp(
        _CROSSING_NULLSPACE_MAX_STEP_RAD
        / torch_module.clamp(correction_norm, min=1.0e-9),
        max=1.0,
    )


class HandsOnLoadedTrayGateIsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Bimanual tray transport with ordered, fail-closed physical evidence."""

    def __init__(self, task_name: str = TASK_NAME, **kwargs: Any):
        if task_name != TASK_NAME:
            raise ValueError(f"Expected task_name={TASK_NAME!r}, got {task_name!r}")
        self._ltg_evaluator = LoadedTrayGateEvaluator()
        self._ltg_last_metrics: dict[str, Any] = {}
        self._ltg_hand_contact_views: dict[str, Any] = {}
        self._ltg_payload_support_views: dict[str, Any] = {}
        self._ltg_hand_gate_contact_views: dict[str, Any] = {}
        self._ltg_gate_contact_views: dict[str, Any] = {}
        self._ltg_contact_errors: set[str] = set()
        self._ltg_frame_collider_paths: tuple[str, ...] = ()
        self._ltg_reset_tray_xy = np.asarray(TRAY_START_POSITION[:2], dtype=np.float64)
        self._ltg_reset_tray_yaw_deg = TRAY_START_YAW_DEG
        self._ltg_reset_gate_xy = np.asarray([GATE_PLANE_X, 0.0], dtype=np.float64)
        self._ltg_reset_gate_yaw_deg = 0.0
        self._ltg_reset_goal_xy = np.asarray(TARGET_CENTER_XY, dtype=np.float64)
        self._ltg_reset_goal_yaw_deg = 0.0
        self._ltg_reset_layout_attempts = 0
        self._ltg_reset_layout_fallback = True
        self._ltg_reset_payload_local_xy = {
            name: value.copy() for name, value in _AUTHORED_PAYLOAD_LOCAL_XY.items()
        }
        self._ltg_tilt_stabilizer_active = False
        self._ltg_tilt_stabilizer_torque = np.zeros(3, dtype=np.float64)
        self._ltg_transport_stabilizer_active = False
        self._ltg_transport_stabilizer_force = np.zeros(3, dtype=np.float64)
        self._ltg_transport_tracking_error_m = 0.0
        self._ltg_transport_tracking_scale = 0.0
        self._ltg_transport_grasp_offset_x: float | None = None
        self._ltg_crossing_nullspace_reference: dict[str, Any] = {}
        self._ltg_crossing_nullspace_active = {"right": False, "left": False}
        self._ltg_crossing_nullspace_step_rad = {"right": 0.0, "left": 0.0}
        self._ltg_tilt_stabilizer_warning_emitted = False
        # This native Isaac task runs in a process-owned collection worker.
        # Isaac's full teardown can block after contact tensor creation; the
        # established worker-safe path skips that redundant global cleanup.
        self._fast_shutdown_on_close = True
        super().__init__(task_name=task_name, **kwargs)

    def _task_extra_pose_names(self) -> set[str]:
        extras = set(super()._task_extra_pose_names())
        extras.add("loaded_tray_gate")
        return extras

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        """Replace BC005 converter visuals with production USD references."""
        # Preserve every shared ABC task hook before specializing BC005.  This
        # method runs after the MJCF scene is composed and before the first
        # PhysX reset, so the original rigid bodies remain the sole owners of
        # mass, reset state, contact reporting, and task semantics.
        super()._spawn_task_specific_assets(sim_utils)
        if self._physics_backend != "physx":
            raise RuntimeError("B320-BC005 production assets require PhysX")
        if self._stage is None:
            raise RuntimeError(
                "B320-BC005 stage is unavailable before production asset composition"
            )

        missing_assets = [
            path
            for path in (_BC005_TRAY_ASSET_PATH, _BC005_FRAME_ASSET_PATH)
            if not path.is_file()
        ]
        if missing_assets:
            raise FileNotFoundError(
                "B320-BC005 production USD asset is missing: "
                + ", ".join(str(path) for path in missing_assets)
            )

        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        tray_body = self._find_rigid_body_prim("loaded_tray")
        frame_body = self._find_rigid_body_prim("loaded_tray_gate")
        if tray_body is None or frame_body is None:
            raise RuntimeError(
                "B320-BC005 converter rigid bodies loaded_tray and "
                "loaded_tray_gate must exist before production composition"
            )
        body_paths = {
            "tray": str(tray_body.GetPath()),
            "frame": str(frame_body.GetPath()),
        }

        # Converter bodies are normally instanceable.  Expand only these two
        # task bodies so their obsolete visual/collision scopes can be
        # deactivated without touching reset joints or semantic sites.
        for body_path in body_paths.values():
            sim_utils.make_uninstanceable(body_path, stage=self._stage)
        tray_body = self._stage.GetPrimAtPath(body_paths["tray"])
        frame_body = self._stage.GetPrimAtPath(body_paths["frame"])
        for role, body in (("tray", tray_body), ("frame", frame_body)):
            if (
                not body.IsValid()
                or not body.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                raise RuntimeError(
                    f"B320-BC005 {role} rigid body became invalid while "
                    "expanding converter instances"
                )

        def converter_scope(
            body: Any,
            scope_name: str,
            role: str,
            *,
            required: bool = True,
        ) -> Any | None:
            matches = [
                prim
                for prim in Usd.PrimRange(body)
                if prim != body and str(prim.GetName()) == scope_name
            ]
            if len(matches) > 1 or (required and not matches):
                raise RuntimeError(
                    f"B320-BC005 {role} converter must expose "
                    f"{'exactly one' if required else 'at most one'} "
                    f"{scope_name!r} scope; got {len(matches)}"
                )
            return matches[0] if matches else None

        tray_visuals = converter_scope(tray_body, "visuals", "tray")
        tray_collisions = converter_scope(tray_body, "collisions", "tray")
        # The current MJCF already authors the analytic frame with alpha zero.
        # Depending on converter version that produces either an inactive
        # visual scope or no visual scope at all; both are valid predecessors.
        frame_visuals = converter_scope(
            frame_body, "visuals", "frame", required=False
        )
        frame_collisions = converter_scope(frame_body, "collisions", "frame")
        if (
            tray_visuals is None
            or tray_collisions is None
            or frame_collisions is None
        ):
            raise RuntimeError(
                "B320-BC005 required converter scopes resolved to null"
            )
        tray_placeholder_colliders = sum(
            prim.HasAPI(UsdPhysics.CollisionAPI)
            for prim in Usd.PrimRange(tray_collisions)
        )
        frame_retained_colliders = sum(
            prim.HasAPI(UsdPhysics.CollisionAPI)
            for prim in Usd.PrimRange(frame_collisions)
        )
        if tray_placeholder_colliders <= 0 or frame_retained_colliders <= 0:
            raise RuntimeError(
                "B320-BC005 converter collision topology is incomplete before "
                "production composition"
            )

        # The selected tray owns both appearance and its source-derived
        # compound collision (five deck/rim boxes plus twelve convex handle
        # prisms).  The selected wooden frame asset itself stays render-only;
        # its shallow converter collision is replaced below by three invisible
        # full-depth boxes aligned to the scaled visible posts and lintel.
        disabled_placeholder_scopes = [
            scope
            for scope in (
                tray_visuals,
                tray_collisions,
                frame_visuals,
                frame_collisions,
            )
            if scope is not None
        ]
        for scope in disabled_placeholder_scopes:
            scope.SetActive(False)
            if scope.IsActive():
                raise RuntimeError(
                    "B320-BC005 could not deactivate converter placeholder "
                    f"scope {scope.GetPath()}"
                )

        tray_reference_path = (
            f"{body_paths['tray']}/{_BC005_TRAY_REFERENCE_CHILD}"
        )
        frame_reference_path = (
            f"{body_paths['frame']}/{_BC005_FRAME_REFERENCE_CHILD}"
        )
        tray_reference = UsdGeom.Xform.Define(
            self._stage, tray_reference_path
        ).GetPrim()
        frame_reference = UsdGeom.Xform.Define(
            self._stage, frame_reference_path
        ).GetPrim()
        for reference, asset_path in (
            (tray_reference, _BC005_TRAY_ASSET_PATH),
            (frame_reference, _BC005_FRAME_ASSET_PATH),
        ):
            references = reference.GetReferences()
            references.ClearReferences()
            if not references.AddReference(str(asset_path.resolve())):
                raise RuntimeError(
                    f"B320-BC005 could not reference production USD {asset_path}"
                )

        frame_reference_xform = UsdGeom.Xformable(frame_reference)
        scale_op = frame_reference_xform.AddScaleOp(
            UsdGeom.XformOp.PrecisionDouble,
            "bc005NarrowOpening",
        )
        scale_op.Set(Gf.Vec3d(1.0, float(GATE_VISUAL_Y_SCALE), 1.0))

        frame_collision_scope_path = (
            f"{body_paths['frame']}/BC005PhysicalFrameCollision"
        )
        frame_collision_scope = UsdGeom.Scope.Define(
            self._stage,
            frame_collision_scope_path,
        ).GetPrim()
        authored_frame_colliders: list[Any] = []
        for member_name, spec in GATE_COLLIDER_BOXES.items():
            cube = UsdGeom.Cube.Define(
                self._stage,
                f"{frame_collision_scope_path}/{member_name}",
            )
            cube.CreateSizeAttr(2.0)
            cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            cube.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            xformable = UsdGeom.Xformable(cube.GetPrim())
            xformable.AddTranslateOp().Set(
                Gf.Vec3d(*[float(value) for value in spec["center"]])
            )
            xformable.AddScaleOp().Set(
                Gf.Vec3d(*[float(value) for value in spec["half_extents"]])
            )
            collider = cube.GetPrim()
            UsdPhysics.CollisionAPI.Apply(collider)
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(collider)
            physx_collision.CreateContactOffsetAttr().Set(0.004)
            physx_collision.CreateRestOffsetAttr().Set(0.0)
            collider.SetCustomDataByKey("benchmarkTaskId", TASK_ID)
            collider.SetCustomDataByKey(
                "bc005CollisionPurpose",
                "visible_member_full_depth_obstacle",
            )
            collider.SetCustomDataByKey("visibleMember", member_name)
            authored_frame_colliders.append(collider)
        if len(authored_frame_colliders) != _BC005_EXPECTED_FRAME_COLLISION_BOX_COUNT:
            raise RuntimeError(
                "B320-BC005 must author one full-depth collider per visible "
                "frame member"
            )
        if any(
            UsdGeom.Imageable(collider).ComputeVisibility()
            != UsdGeom.Tokens.invisible
            for collider in authored_frame_colliders
        ):
            raise RuntimeError(
                "B320-BC005 collision-only frame members must remain invisible"
            )
        self._ltg_frame_collider_paths = tuple(
            str(collider.GetPath()) for collider in authored_frame_colliders
        )
        frame_collision_scope.SetCustomDataByKey(
            "bc005VisibleMemberColliderParity",
            True,
        )

        def forbidden_physics_apis(root: Any) -> list[str]:
            return [
                str(prim.GetPath())
                for prim in Usd.PrimRange(root)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                or prim.HasAPI(UsdPhysics.MassAPI)
            ]

        nested_physics = {
            "tray": forbidden_physics_apis(tray_reference),
            "frame": forbidden_physics_apis(frame_reference),
        }
        if nested_physics["tray"] or nested_physics["frame"]:
            raise RuntimeError(
                "B320-BC005 referenced assets must not author nested rigid, "
                f"articulation, or mass APIs: {nested_physics}"
            )

        tray_prims = list(Usd.PrimRange(tray_reference))
        tray_meshes = [prim for prim in tray_prims if prim.IsA(UsdGeom.Mesh)]
        tray_visual_meshes = [
            prim
            for prim in tray_meshes
            if not prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        tray_collision_meshes = [
            prim
            for prim in tray_meshes
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        tray_collision_boxes = [
            prim
            for prim in tray_prims
            if prim.IsA(UsdGeom.Cube)
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        tray_collision_prims = [
            prim
            for prim in tray_prims
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        tray_mesh_collision_prims = [
            prim
            for prim in tray_collision_meshes
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        ]
        unexpected_collision_types = [
            str(prim.GetPath())
            for prim in tray_collision_prims
            if not prim.IsA(UsdGeom.Mesh) and not prim.IsA(UsdGeom.Cube)
        ]
        tray_physics_material_prims = [
            prim
            for prim in tray_prims
            if prim.HasAPI(UsdPhysics.MaterialAPI)
        ]
        physics_material_paths = {
            str(prim.GetName()): str(prim.GetPath())
            for prim in tray_physics_material_prims
        }
        invalid_physics_material_properties: dict[str, tuple[float, ...]] = {}
        for material_prim in tray_physics_material_prims:
            material_api = UsdPhysics.MaterialAPI(material_prim)
            values = (
                float(material_api.GetStaticFrictionAttr().Get()),
                float(material_api.GetDynamicFrictionAttr().Get()),
                float(material_api.GetRestitutionAttr().Get()),
            )
            expected_values = (
                _BC005_EXPECTED_TRAY_PHYSICS_MATERIAL_PROPERTIES.get(
                    str(material_prim.GetName())
                )
            )
            if expected_values is None or not np.allclose(
                values, expected_values, atol=1.0e-7, rtol=0.0
            ):
                invalid_physics_material_properties[
                    str(material_prim.GetPath())
                ] = values
        invalid_physics_material_bindings: dict[str, dict[str, Any]] = {}
        for collider in tray_collision_prims:
            expected_material_name = (
                "Body" if collider.IsA(UsdGeom.Cube) else "Handle"
            )
            expected_material_path = physics_material_paths.get(
                expected_material_name, ""
            )
            binding_relationships = sorted(
                str(relationship.GetName())
                for relationship in collider.GetRelationships()
                if str(relationship.GetName()).startswith("material:binding")
            )
            relationship = collider.GetRelationship(
                "material:binding:physics"
            )
            targets = (
                [str(path) for path in relationship.GetTargets()]
                if relationship.IsValid()
                else []
            )
            resolved_material, resolved_relationship = (
                UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(
                    "physics"
                )
            )
            resolved_path = (
                str(resolved_material.GetPath()) if resolved_material else ""
            )
            resolved_relationship_name = (
                str(resolved_relationship.GetName())
                if resolved_relationship
                else ""
            )
            if (
                binding_relationships != ["material:binding:physics"]
                or targets != [expected_material_path]
                or resolved_path != expected_material_path
                or resolved_relationship_name != "material:binding:physics"
            ):
                invalid_physics_material_bindings[str(collider.GetPath())] = {
                    "relationships": binding_relationships,
                    "targets": targets,
                    "resolved_path": resolved_path,
                    "resolved_relationship": resolved_relationship_name,
                    "expected_path": expected_material_path,
                }
        if (
            len(tray_visual_meshes)
            != _BC005_EXPECTED_TRAY_VISUAL_MESH_COUNT
            or len(tray_collision_boxes)
            != _BC005_EXPECTED_TRAY_COLLISION_BOX_COUNT
            or len(tray_collision_meshes)
            != _BC005_EXPECTED_TRAY_COLLISION_MESH_COUNT
            or len(tray_mesh_collision_prims)
            != _BC005_EXPECTED_TRAY_COLLISION_MESH_COUNT
            or len(tray_collision_prims)
            != _BC005_EXPECTED_TRAY_COLLISION_API_COUNT
            or len(tray_physics_material_prims)
            != _BC005_EXPECTED_TRAY_PHYSICS_MATERIAL_COUNT
            or set(physics_material_paths)
            != set(_BC005_EXPECTED_TRAY_PHYSICS_MATERIAL_PROPERTIES)
            or unexpected_collision_types
            or invalid_physics_material_properties
            or invalid_physics_material_bindings
        ):
            raise RuntimeError(
                "B320-BC005 selected wooden tray must expose its audited "
                "17-piece source-derived compound collision; "
                f"visual_meshes={len(tray_visual_meshes)}, "
                f"collision_boxes={len(tray_collision_boxes)}, "
                f"collision_meshes={len(tray_collision_meshes)}, "
                f"mesh_collision_apis={len(tray_mesh_collision_prims)}, "
                f"collision_apis={len(tray_collision_prims)}, "
                f"physics_materials={len(tray_physics_material_prims)}, "
                f"unexpected_collision_types={unexpected_collision_types}, "
                "invalid_physics_material_properties="
                f"{invalid_physics_material_properties}, "
                "invalid_physics_material_bindings="
                f"{invalid_physics_material_bindings}"
            )

        frame_prims = list(Usd.PrimRange(frame_reference))
        frame_mesh_count = sum(
            prim.IsA(UsdGeom.Mesh) for prim in frame_prims
        )
        frame_collision_count = sum(
            prim.HasAPI(UsdPhysics.CollisionAPI) for prim in frame_prims
        )
        if (
            frame_mesh_count != _BC005_EXPECTED_FRAME_VISUAL_MESH_COUNT
            or frame_collision_count != 0
        ):
            raise RuntimeError(
                "B320-BC005 selected wooden frame must preserve its one-mesh "
                "source visual and remain render-only; "
                f"meshes={frame_mesh_count}, collisions={frame_collision_count}"
            )

        tray_reference.SetCustomDataByKey("benchmarkTaskId", TASK_ID)
        tray_reference.SetCustomDataByKey("runtimeAssetRole", "loaded_tray")
        tray_reference.SetCustomDataByKey(
            "sourceDerivedCollisionMeshCount",
            len(tray_collision_meshes),
        )
        tray_reference.SetCustomDataByKey(
            "sourceDerivedCollisionBoxCount",
            len(tray_collision_boxes),
        )
        tray_reference.SetCustomDataByKey(
            "sourceDerivedCollisionPieceCount",
            len(tray_collision_prims),
        )
        tray_reference.SetCustomDataByKey(
            "boundPhysicsMaterialCount",
            len(tray_physics_material_prims),
        )
        tray_reference.SetCustomDataByKey(
            "sourceUsd", str(_BC005_TRAY_ASSET_PATH.resolve())
        )
        frame_reference.SetCustomDataByKey("benchmarkTaskId", TASK_ID)
        frame_reference.SetCustomDataByKey(
            "runtimeAssetRole", "narrow_wooden_frame_visual"
        )
        frame_reference.SetCustomDataByKey("physicsMode", "render-only")
        frame_reference.SetCustomDataByKey(
            "bc005VisibleOpeningScaleY", float(GATE_VISUAL_Y_SCALE)
        )
        frame_reference.SetCustomDataByKey(
            "sourceUsd", str(_BC005_FRAME_ASSET_PATH.resolve())
        )
        tray_body.SetCustomDataByKey(
            "bc005ConverterPlaceholdersDisabled", True
        )
        frame_body.SetCustomDataByKey(
            "bc005ConverterVisualsDisabled", True
        )
        frame_body.SetCustomDataByKey(
            "bc005FullDepthFrameColliderCount",
            len(authored_frame_colliders),
        )

        print(
            "[HandsOn/BC005] production assets composed: "
            f"tray_reference={tray_reference_path}, "
            f"tray_visual_meshes={len(tray_visual_meshes)}, "
            f"tray_collision_boxes={len(tray_collision_boxes)}, "
            f"tray_collision_meshes={len(tray_collision_meshes)}, "
            f"tray_collision_apis={len(tray_collision_prims)}, "
            f"tray_physics_materials={len(tray_physics_material_prims)}, "
            f"tray_placeholder_colliders_disabled={tray_placeholder_colliders}, "
            f"frame_reference={frame_reference_path}, "
            f"frame_render_meshes={frame_mesh_count}, "
            f"frame_asset_colliders={frame_collision_count}, "
            f"frame_converter_colliders_disabled={frame_retained_colliders}, "
            f"frame_full_depth_colliders={len(authored_frame_colliders)}, "
            f"frame_opening_scale_y={GATE_VISUAL_Y_SCALE:.6f}",
            flush=True,
        )

    def _configure_task_specific_pre_physx(self) -> bool:
        """Finalize the resettable, collision-backed narrow frame.

        Gate/goal fixed joints are replaced with kinematic roots so their
        sampled transforms can be replayed exactly.  BC005 uses the shared
        mirrored bimanual robot home and keeps frame/robot collision enabled;
        tray and both payloads likewise collide with all three full-depth
        members and use CCD to prevent fast tunnelling.
        """

        super_configured = bool(super()._configure_task_specific_pre_physx())
        if self._stage is None:
            raise RuntimeError("B320-BC005 frame collision setup requires a USD stage")
        from pxr import PhysxSchema, Usd, UsdPhysics

        task_root = self._stage.GetPrimAtPath("/World/Task")
        fixture_fixed_joints_disabled = 0
        fixture_kinematic_count = 0
        for fixture_name in ("loaded_tray_gate", "loaded_tray_goal"):
            fixture = self._find_rigid_body_prim(fixture_name)
            if fixture is None:
                raise RuntimeError(
                    f"B320-BC005 cannot resolve resettable fixture {fixture_name}"
                )
            fixture_path = fixture.GetPath()
            for prim in Usd.PrimRange(task_root):
                if not prim.IsActive() or not prim.IsA(UsdPhysics.FixedJoint):
                    continue
                joint = UsdPhysics.Joint(prim)
                targets = set(joint.GetBody0Rel().GetTargets()) | set(
                    joint.GetBody1Rel().GetTargets()
                )
                if fixture_path in targets:
                    prim.SetActive(False)
                    fixture_fixed_joints_disabled += 1
            rigid_api = UsdPhysics.RigidBodyAPI(fixture)
            if not rigid_api:
                rigid_api = UsdPhysics.RigidBodyAPI.Apply(fixture)
            kinematic = rigid_api.GetKinematicEnabledAttr()
            if not kinematic.IsValid():
                kinematic = rigid_api.CreateKinematicEnabledAttr()
            kinematic.Set(True)
            fixture.SetCustomDataByKey("bc005ResettableKinematicFixture", True)
            fixture_kinematic_count += 1

        ccd_task_bodies = 0
        for body_name in ("loaded_tray", *PAYLOAD_NAMES):
            body = self._find_rigid_body_prim(body_name)
            if body is None:
                raise RuntimeError(
                    f"B320-BC005 cannot enable CCD for {body_name}"
                )
            physx_body = PhysxSchema.PhysxRigidBodyAPI(body)
            if not physx_body:
                physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
            ccd = physx_body.GetEnableCCDAttr()
            if not ccd.IsValid():
                ccd = physx_body.CreateEnableCCDAttr()
            ccd.Set(True)
            body.SetCustomDataByKey("bc005GateTraversalCCD", True)
            ccd_task_bodies += 1

        frame_colliders = [
            self._stage.GetPrimAtPath(path)
            for path in self._ltg_frame_collider_paths
        ]
        if len(frame_colliders) != 3 or any(
            not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI)
            for prim in frame_colliders
        ):
            raise RuntimeError(
                "B320-BC005 expected exactly three physical frame colliders "
                f"before first PhysX; got {self._ltg_frame_collider_paths}"
            )

        for collider in frame_colliders:
            collider.SetCustomDataByKey(
                "bc005CollisionPurpose",
                "robot_and_ccd_task_object_obstacle",
            )
        print(
            "[HandsOn/BC005] frame collision contract: "
            f"fixture_fixed_joints_disabled={fixture_fixed_joints_disabled}, "
            f"fixture_kinematic_count={fixture_kinematic_count}, "
            f"ccd_task_bodies={ccd_task_bodies}, "
            "tray_payload_collisions=enabled, "
            "robot_collisions=enabled, "
            "robot_pairs_filtered=0",
            flush=True,
        )
        return bool(
            super_configured
            or fixture_fixed_joints_disabled
            or fixture_kinematic_count
            or ccd_task_bodies
        )

    def _compute_arm_joint_target(
        self,
        side: str,
        robot: Any,
        pose: np.ndarray,
        commanded: bool,
    ):
        arm_target, status = super()._compute_arm_joint_target(
            side,
            robot,
            pose,
            commanded,
        )
        self._ltg_crossing_nullspace_active[side] = False
        self._ltg_crossing_nullspace_step_rad[side] = 0.0
        metrics = self._ltg_last_metrics
        enabled = bool(
            arm_target is not None
            and commanded
            and np.all(np.isfinite(np.asarray(pose, dtype=np.float64)))
            and self._ltg_evaluator.stage >= LoadedTrayStage.GATE_CROSSING
            and metrics.get("right_role_owned", False)
            and metrics.get("left_role_owned", False)
            and metrics.get("payloads_on_tray", False)
        )
        if not enabled:
            return arm_target, status

        try:
            arm_joint_ids = self._arm_joint_ids[side]
            ee_body_idx = self._ee_body_indices[side]
            ee_jacobi_idx = self._ee_jacobi_indices[side]
            jacobian = _end_effector_link_jacobian(
                robot,
                ee_body_idx if self._isaaclab_uses_xyzw else ee_jacobi_idx,
                arm_joint_ids,
                lab3=self._isaaclab_uses_xyzw,
            )
            current = self._tensor_view(robot.data.joint_pos)[:, arm_joint_ids]
            reference = self._ltg_crossing_nullspace_reference.get(side)
            if reference is None:
                reference = arm_target.detach().clone()
                self._ltg_crossing_nullspace_reference[side] = reference

            correction = _bounded_nullspace_posture_delta(
                self._torch,
                jacobian,
                current,
                reference,
            )
            self._ltg_crossing_nullspace_active[side] = True
            self._ltg_crossing_nullspace_step_rad[side] = float(
                self._torch.linalg.vector_norm(correction).item()
            )
            return arm_target + correction, f"{status}+loaded_tray_nullspace"
        except Exception as exc:
            message = f"{side} loaded-tray nullspace posture hold failed: {exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
            return arm_target, status

    def _fallback_pose(self, name: str) -> np.ndarray:
        yaw_half = math.radians(TRAY_START_YAW_DEG) * 0.5
        tray_orientation_wxyz = (
            math.cos(yaw_half),
            0.0,
            0.0,
            math.sin(yaw_half),
        )
        authored = {
            "loaded_tray": (*TRAY_START_POSITION, *tray_orientation_wxyz),
            "loaded_tray_can": (*PAYLOAD_START_POSITIONS["loaded_tray_can"], *tray_orientation_wxyz),
            "loaded_tray_cup": (*PAYLOAD_START_POSITIONS["loaded_tray_cup"], *tray_orientation_wxyz),
            "loaded_tray_gate": (GATE_PLANE_X, 0.0, GATE_BASE_Z, 1.0, 0.0, 0.0, 0.0),
            "loaded_tray_goal": (TARGET_CENTER_XY[0], TARGET_CENTER_XY[1], 0.8915, 1.0, 0.0, 0.0, 0.0),
        }
        if name in authored:
            return np.asarray(authored[name], dtype=np.float64)
        return super()._fallback_pose(name)

    def _build_observation_space(self) -> spaces.Dict:
        state_spaces: dict[str, spaces.Space] = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "gripper_pose": spaces.Box(-np.inf, np.inf, shape=(32,), dtype=np.float64),
            "tray_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "payload_poses": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "gate_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "goal_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "handle_positions": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
            "task_progress": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float64),
            "tray_metrics": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float64),
            "gate_metrics": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
            "payload_metrics": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float64),
            "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
        }
        observation_spaces: dict[str, spaces.Space] = {"state": spaces.Dict(state_spaces)}
        if self.image_obs:
            image_space = lambda shape=self._image_shape: spaces.Box(0, 255, shape=shape, dtype=np.uint8)
            observation_spaces["images"] = spaces.Dict(
                {
                    "random_camera" if self.randomize else "front": image_space(),
                    "wrist_left": image_space((*reversed(self._wrist_recording_camera_output_size), 3)),
                    "wrist_right": image_space((*reversed(self._wrist_recording_camera_output_size), 3)),
                }
            )
        return spaces.Dict(observation_spaces)

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self._ltg_evaluator.reset()
        self._ltg_last_metrics = {}
        self._ltg_tilt_stabilizer_active = False
        self._ltg_tilt_stabilizer_torque.fill(0.0)
        self._ltg_transport_stabilizer_active = False
        self._ltg_transport_stabilizer_force.fill(0.0)
        self._ltg_transport_tracking_error_m = 0.0
        self._ltg_transport_tracking_scale = 0.0
        self._ltg_transport_grasp_offset_x = None
        self._ltg_crossing_nullspace_reference = {}
        self._ltg_crossing_nullspace_active = {"right": False, "left": False}
        self._ltg_crossing_nullspace_step_rad = {"right": 0.0, "left": 0.0}
        layout = (
            _sample_bc005_reset_layout(self._random)
            if self.randomize
            else _bc005_fallback_reset_layout()
        )
        if not _bc005_reset_layout_valid(layout):
            raise RuntimeError("B320-BC005 sampled reset layout failed validation")
        self._ltg_reset_tray_xy = np.asarray(
            layout["tray_xy"], dtype=np.float64
        ).copy()
        self._ltg_reset_tray_yaw_deg = float(layout["tray_yaw_deg"])
        self._ltg_reset_gate_xy = np.asarray(
            layout["gate_xy"], dtype=np.float64
        ).copy()
        self._ltg_reset_gate_yaw_deg = float(layout["gate_yaw_deg"])
        self._ltg_reset_goal_xy = np.asarray(
            layout["goal_xy"], dtype=np.float64
        ).copy()
        self._ltg_reset_goal_yaw_deg = float(layout["goal_yaw_deg"])
        self._ltg_reset_payload_local_xy = {
            name: np.asarray(layout["payload_local_xy"][name], dtype=np.float64).copy()
            for name in PAYLOAD_NAMES
        }
        self._ltg_reset_layout_attempts = int(layout["attempts"])
        self._ltg_reset_layout_fallback = bool(layout["fallback"])

        # Interface-only tests have no tensor writers, so mirror the same
        # sampled physical transforms into the authoritative pose dictionary.
        tray_rotation = np.asarray(
            [
                [math.cos(math.radians(self._ltg_reset_tray_yaw_deg)), -math.sin(math.radians(self._ltg_reset_tray_yaw_deg))],
                [math.sin(math.radians(self._ltg_reset_tray_yaw_deg)), math.cos(math.radians(self._ltg_reset_tray_yaw_deg))],
            ],
            dtype=np.float64,
        )
        tray_quat_wxyz = _yaw_quaternion_wxyz(self._ltg_reset_tray_yaw_deg)
        gate_quat_wxyz = _yaw_quaternion_wxyz(self._ltg_reset_gate_yaw_deg)
        goal_quat_wxyz = _yaw_quaternion_wxyz(self._ltg_reset_goal_yaw_deg)
        self._poses["loaded_tray"] = np.concatenate(
            [self._ltg_reset_tray_xy, [TRAY_START_POSITION[2]], tray_quat_wxyz]
        )
        for name in PAYLOAD_NAMES:
            payload_xy = (
                self._ltg_reset_tray_xy
                + tray_rotation @ self._ltg_reset_payload_local_xy[name]
            )
            self._poses[name] = np.concatenate(
                [payload_xy, [_AUTHORED_PAYLOAD_Z[name]], tray_quat_wxyz]
            )
        self._poses["loaded_tray_gate"] = np.concatenate(
            [self._ltg_reset_gate_xy, [GATE_BASE_Z], gate_quat_wxyz]
        )
        self._poses["loaded_tray_goal"] = np.concatenate(
            [self._ltg_reset_goal_xy, [0.8915], goal_quat_wxyz]
        )

    def _reset_state_maps(self):
        rigid = {
            path: (view, indices, pose, velocity)
            for path, view, indices, pose, velocity in self._task_reset_body_states
        }
        articulations = {
            path: (view, indices, pose, velocity)
            for path, view, indices, pose, velocity, _q, _qd
            in self._task_reset_articulation_states
        }
        return rigid, articulations

    def _write_reset_pose(
        self,
        name: str,
        *,
        xy: np.ndarray,
        z: float,
        orientation_xyzw: np.ndarray,
    ) -> bool:
        prim = self._find_rigid_body_prim(name)
        if prim is None:
            self._add_task_state_warning(f"Loaded-tray reset body unavailable: {name}")
            return False
        path = str(prim.GetPath())
        rigid, articulations = self._reset_state_maps()
        state = self._reset_transform_state_for_path(path, rigid, articulations)
        if state is None:
            self._add_task_state_warning(f"Loaded-tray reset tensor unavailable: {name}")
            return False
        state_kind, view, indices, root_pose, root_velocity = state
        pose_np = root_pose.detach().cpu().numpy().copy()
        orientation = np.asarray(orientation_xyzw, dtype=np.float64).reshape(4)
        orientation /= max(float(np.linalg.norm(orientation)), 1.0e-12)
        pose_np[:, 3:7] = orientation
        randomized = _randomized_physx_pose(
            pose_np,
            xy_offset=np.asarray(xy, dtype=np.float64).reshape(2) - pose_np[0, :2],
            z_offset=float(z) - float(pose_np[0, 2]),
            yaw_degrees=0.0,
        )
        pose_tensor = self._torch.as_tensor(
            randomized, dtype=root_pose.dtype, device=root_pose.device
        )
        zero_velocity = self._torch.zeros_like(root_velocity)
        if state_kind == "articulation":
            view.set_root_transforms(pose_tensor, indices)
            view.set_root_velocities(zero_velocity, indices)
            self._record_authored_task_reset_write(
                path, transform_xyzw=pose_tensor, velocity=zero_velocity
            )
        else:
            self._write_rigid_reset_state(
                path, view, indices, pose_tensor, zero_velocity
            )
        return True

    def _apply_task_pose_randomization(self) -> None:
        if not self._task_reset_body_states and not self._task_reset_articulation_states:
            return
        yaw = math.radians(self._ltg_reset_tray_yaw_deg)
        orientation_xyzw = _yaw_quaternion_xyzw(self._ltg_reset_tray_yaw_deg)
        rotation_xy = np.asarray(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
            dtype=np.float64,
        )
        writes = [self._write_reset_pose(
            "loaded_tray",
            xy=self._ltg_reset_tray_xy,
            z=float(TRAY_START_POSITION[2]),
            orientation_xyzw=orientation_xyzw,
        )]
        for name in PAYLOAD_NAMES:
            payload_xy = (
                self._ltg_reset_tray_xy
                + rotation_xy @ self._ltg_reset_payload_local_xy[name]
            )
            writes.append(self._write_reset_pose(
                name,
                xy=payload_xy,
                z=_AUTHORED_PAYLOAD_Z[name],
                orientation_xyzw=orientation_xyzw,
            ))
        writes.append(
            self._write_reset_pose(
                "loaded_tray_gate",
                xy=self._ltg_reset_gate_xy,
                z=GATE_BASE_Z,
                orientation_xyzw=_yaw_quaternion_xyzw(
                    self._ltg_reset_gate_yaw_deg
                ),
            )
        )
        writes.append(
            self._write_reset_pose(
                "loaded_tray_goal",
                xy=self._ltg_reset_goal_xy,
                z=0.8915,
                orientation_xyzw=_yaw_quaternion_xyzw(
                    self._ltg_reset_goal_yaw_deg
                ),
            )
        )
        if not all(writes):
            raise RuntimeError(
                "B320-BC005 could not write every sampled tray/gate/goal reset transform"
            )
        print(
            "[HandsOn/IsaacLab] Loaded-tray sampled reset: "
            f"tray_xy={self._ltg_reset_tray_xy.tolist()}, "
            f"tray_yaw_deg={self._ltg_reset_tray_yaw_deg:.3f}, "
            f"gate_xy={self._ltg_reset_gate_xy.tolist()}, "
            f"gate_yaw_deg={self._ltg_reset_gate_yaw_deg:.3f}, "
            f"goal_xy={self._ltg_reset_goal_xy.tolist()}, "
            f"attempts={self._ltg_reset_layout_attempts}, "
            f"fallback={self._ltg_reset_layout_fallback}",
            flush=True,
        )

    def _hand_body_paths(self, side: str) -> list[str]:
        if self._stage is None:
            return []
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        if not root.IsValid():
            return []
        prefix = f"{side}_finger"
        palm = f"{side}_palm_link"
        return [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and (prim.GetName() == palm or prim.GetName().startswith(prefix))
        ]

    def _configure_task_specific_contact_reporting(self) -> int:
        if self._stage is None:
            return 0
        from pxr import PhysxSchema

        configured = 0
        for name in ("loaded_tray", *PAYLOAD_NAMES, "loaded_tray_gate"):
            body_prim = self._find_rigid_body_prim(name)
            if body_prim is None:
                self._add_task_state_warning(
                    f"Could not configure loaded-tray contact reporting for {name}"
                )
                continue
            if body_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                report_api = PhysxSchema.PhysxContactReportAPI.Get(
                    self._stage, body_prim.GetPath()
                )
            else:
                report_api = PhysxSchema.PhysxContactReportAPI.Apply(body_prim)
            report_api.CreateThresholdAttr().Set(0.0)
            configured += 1
        return configured

    def _init_task_specific_runtime(self) -> None:
        self._ltg_hand_contact_views = {}
        self._ltg_payload_support_views = {}
        self._ltg_hand_gate_contact_views = {}
        self._ltg_gate_contact_views = {}
        self._ltg_contact_errors = set()
        if not self._robots or self._stage is None:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_sim_view is None:
            self._add_task_state_warning(
                "Loaded-tray PhysX contact reporting is unavailable"
            )
            return
        tray_prim = self._find_rigid_body_prim("loaded_tray")
        gate_prim = self._find_rigid_body_prim("loaded_tray_gate")
        if tray_prim is None or gate_prim is None:
            self._add_task_state_warning(
                "Loaded-tray contact target rigid bodies are unavailable"
            )
            return
        tray_path = str(tray_prim.GetPath())
        gate_path = str(gate_prim.GetPath())
        for side in ("left", "right"):
            hand_paths = self._hand_body_paths(side)
            key = f"hand:{side}"
            if not hand_paths:
                self._ltg_contact_errors.add(key)
                continue
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    tray_path,
                    filter_patterns=hand_paths,
                    max_contact_data_count=max(64, len(hand_paths) * 8),
                )
                if (
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                ):
                    self._ltg_hand_contact_views[side] = view
                else:
                    self._ltg_contact_errors.add(key)
            except Exception as exc:
                self._ltg_contact_errors.add(key)
                self._add_task_state_warning(
                    f"Could not bind {side}-hand/tray contact: {exc}"
                )
            key = f"gate:hand:{side}"
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    gate_path,
                    filter_patterns=hand_paths,
                    max_contact_data_count=max(64, len(hand_paths) * 8),
                )
                if view is not None and int(view.sensor_count) > 0 and int(view.filter_count) > 0:
                    self._ltg_hand_gate_contact_views[side] = view
                else:
                    self._ltg_contact_errors.add(key)
            except Exception as exc:
                self._ltg_contact_errors.add(key)
                self._add_task_state_warning(
                    f"Could not bind {side}-hand/gate contact: {exc}"
                )
        for name in PAYLOAD_NAMES:
            payload_prim = self._find_rigid_body_prim(name)
            key = f"support:{name}"
            if payload_prim is None:
                self._ltg_contact_errors.add(key)
                continue
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    str(payload_prim.GetPath()),
                    filter_patterns=[tray_path],
                    max_contact_data_count=64,
                )
                if (
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                ):
                    self._ltg_payload_support_views[name] = view
                else:
                    self._ltg_contact_errors.add(key)
            except Exception as exc:
                self._ltg_contact_errors.add(key)
                self._add_task_state_warning(
                    f"Could not bind {name}/tray support contact: {exc}"
                )
        for name in ("loaded_tray", *PAYLOAD_NAMES):
            body_prim = self._find_rigid_body_prim(name)
            key = f"gate:{name}"
            if body_prim is None:
                self._ltg_contact_errors.add(key)
                continue
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    str(body_prim.GetPath()),
                    filter_patterns=[gate_path],
                    max_contact_data_count=64,
                )
                if view is not None and int(view.sensor_count) > 0 and int(view.filter_count) > 0:
                    self._ltg_gate_contact_views[name] = view
                else:
                    self._ltg_contact_errors.add(key)
            except Exception as exc:
                self._ltg_contact_errors.add(key)
                self._add_task_state_warning(
                    f"Could not bind {name}/gate contact: {exc}"
                )
        if self._ltg_contact_errors:
            self._add_task_state_warning(
                "Loaded-tray contact reporting is incomplete; affected checks "
                "fail closed or remain unavailable: "
                + ",".join(sorted(self._ltg_contact_errors))
            )

    def _contact_reading(self, view: Any | None) -> tuple[float, bool]:
        if view is None:
            return 0.0, False
        try:
            values = _contact_tensor_to_numpy(
                view.get_contact_force_matrix(self._sim_dt)
            ).astype(np.float64)
            if not np.all(np.isfinite(values)):
                return 0.0, False
            force = (
                float(np.linalg.norm(values, axis=-1).sum())
                if values.size
                else 0.0
            )
            return force, True
        except Exception:
            return 0.0, False

    def _tray_contact_region_reading(
        self,
        view: Any | None,
        tray_matrix: np.ndarray,
        grip_local_x: float,
    ) -> tuple[float, float, bool]:
        """Partition tray contact manifolds into selected-grip and other force."""
        if view is None:
            return 0.0, 0.0, False
        try:
            data = view.get_contact_data(self._sim_dt)
            if not isinstance(data, tuple) or len(data) != 6:
                return 0.0, 0.0, False
            normal_forces, points, normals, separations, counts, starts = data
            forces = np.abs(
                _contact_tensor_to_numpy(normal_forces)
                .astype(np.float64)
                .reshape(-1)
            )
            points_np = (
                _contact_tensor_to_numpy(points)
                .astype(np.float64)
                .reshape(-1, 3)
            )
            normals_np = (
                _contact_tensor_to_numpy(normals)
                .astype(np.float64)
                .reshape(-1, 3)
            )
            separations_np = (
                _contact_tensor_to_numpy(separations)
                .astype(np.float64)
                .reshape(-1)
            )
            raw_counts = _contact_tensor_to_numpy(counts).reshape(-1)
            raw_starts = _contact_tensor_to_numpy(starts).reshape(-1)
            if (
                not np.all(np.isfinite(raw_counts))
                or not np.all(np.isfinite(raw_starts))
                or not np.all(raw_counts == np.floor(raw_counts))
                or not np.all(raw_starts == np.floor(raw_starts))
            ):
                return 0.0, 0.0, False
            counts_np = raw_counts.astype(np.int64, copy=False)
            starts_np = raw_starts.astype(np.int64, copy=False)
            expected_pairs = int(getattr(view, "sensor_count", 0)) * int(
                getattr(view, "filter_count", 0)
            )
            capacity = int(getattr(view, "max_contact_data_count", 0))
            if (
                expected_pairs <= 0
                or counts_np.size != expected_pairs
                or starts_np.size != expected_pairs
                or np.any(counts_np < 0)
                or np.any(starts_np < 0)
                or capacity <= 0
                or int(np.sum(counts_np)) >= capacity
            ):
                return 0.0, 0.0, False
            world_from_tray = np.asarray(tray_matrix, dtype=np.float64).reshape(4, 4)
            if not np.all(np.isfinite(world_from_tray)):
                return 0.0, 0.0, False
            tray_rotation = world_from_tray[:3, :3]
            tray_translation = world_from_tray[:3, 3]
            grip_force_n = 0.0
            non_grip_force_n = 0.0
            for count, start in zip(counts_np, starts_np, strict=True):
                count_int = int(count)
                start_int = int(start)
                if count_int <= 0:
                    continue
                stop = start_int + count_int
                if (
                    start_int < 0
                    or stop > len(points_np)
                    or stop > len(forces)
                    or stop > len(normals_np)
                    or stop > len(separations_np)
                ):
                    return 0.0, 0.0, False
                for contact_index in range(start_int, stop):
                    point_world = points_np[contact_index]
                    force_n = float(forces[contact_index])
                    if not (
                        np.all(np.isfinite(point_world))
                        and np.all(np.isfinite(normals_np[contact_index]))
                        and math.isfinite(force_n)
                        and math.isfinite(float(separations_np[contact_index]))
                    ):
                        return 0.0, 0.0, False
                    point_local = tray_rotation.T @ (
                        point_world - tray_translation
                    )
                    if _grip_contact_point_in_role_region(
                        point_local,
                        grip_local_x,
                    ):
                        grip_force_n += force_n
                    else:
                        non_grip_force_n += force_n
            return grip_force_n, non_grip_force_n, True
        except Exception:
            return 0.0, 0.0, False

    def _handle_contact_reading(
        self,
        view: Any | None,
        tray_matrix: np.ndarray,
        grip_local_x: float,
    ) -> tuple[float, bool]:
        """Return only contact force inside one selected side-rim grip."""
        grip_force_n, _non_grip_force_n, available = (
            self._tray_contact_region_reading(
                view,
                tray_matrix,
                grip_local_x,
            )
        )
        return grip_force_n, available

    def _hand_closed(self, side: str) -> bool:
        hand = self._left_hand if side == "left" else self._right_hand
        return bool(hand.size and float(np.max(np.abs(hand))) >= 0.08)

    def _hand_distance_to_point(self, side: str, point: np.ndarray) -> float:
        return min(
            (
                float(np.linalg.norm(hand_point - point))
                for hand_point in self._a06_hand_points(side)
            ),
            default=math.inf,
        )

    def _hand_distance_to_segment(
        self,
        side: str,
        start: np.ndarray,
        end: np.ndarray,
    ) -> float:
        return min(
            (
                _point_to_segment_distance(hand_point, start, end)
                for hand_point in self._a06_hand_points(side)
            ),
            default=math.inf,
        )

    @staticmethod
    def _velocity_norms(velocity: np.ndarray | None) -> tuple[float, float, bool]:
        if velocity is None:
            return 0.0, 0.0, False
        value = np.asarray(velocity, dtype=np.float64).reshape(6)
        return (
            float(np.linalg.norm(value[:3])),
            float(np.linalg.norm(value[3:])),
            bool(np.all(np.isfinite(value))),
        )

    @staticmethod
    def _aabb_penetration_with_gate(
        position: np.ndarray,
        half_extents: np.ndarray,
        *,
        gate_x: float,
        gate_base_z: float,
    ) -> float:
        lower = np.asarray(position, dtype=np.float64) - np.asarray(
            half_extents, dtype=np.float64
        )
        upper = np.asarray(position, dtype=np.float64) + np.asarray(
            half_extents, dtype=np.float64
        )
        maximum_penetration = 0.0
        for spec in GATE_COLLIDER_BOXES.values():
            local_center = np.asarray(spec["center"], dtype=np.float64)
            collider_center = np.asarray(
                [
                    gate_x + local_center[0],
                    local_center[1],
                    gate_base_z + local_center[2],
                ],
                dtype=np.float64,
            )
            collider_half = np.asarray(spec["half_extents"], dtype=np.float64)
            collider_lower = collider_center - collider_half
            collider_upper = collider_center + collider_half
            separating_distance = np.minimum(
                upper - collider_lower,
                collider_upper - lower,
            )
            if bool(np.all(separating_distance > 0.0)):
                maximum_penetration = max(
                    maximum_penetration,
                    float(np.min(separating_distance)),
                )
        return maximum_penetration

    @classmethod
    def _oriented_box_penetration_with_gate(
        cls,
        pose_wxyz: np.ndarray,
        local_half_extents: np.ndarray,
        gate_pose_wxyz: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Measure conservative penetration in the sampled gate frame."""
        centre, half = _oriented_box_half_extents_in_frame(
            pose_wxyz,
            local_half_extents,
            gate_pose_wxyz,
        )
        gate_pose = np.asarray(gate_pose_wxyz, dtype=np.float64).reshape(7)
        world_z_centre = float(gate_pose[2] + centre[2])
        penetration = cls._aabb_penetration_with_gate(
            np.asarray([centre[0], centre[1], world_z_centre], dtype=np.float64),
            half,
            gate_x=0.0,
            gate_base_z=float(gate_pose[2]),
        )
        return penetration, centre, half

    @classmethod
    def _cylinder_penetration_with_gate(
        cls,
        pose_wxyz: np.ndarray,
        *,
        radius: float,
        half_height: float,
        gate_pose_wxyz: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        centre, half = _cylinder_half_extents_in_frame(
            pose_wxyz,
            radius=radius,
            half_height=half_height,
            frame_pose_wxyz=gate_pose_wxyz,
        )
        gate_pose = np.asarray(gate_pose_wxyz, dtype=np.float64).reshape(7)
        penetration = cls._aabb_penetration_with_gate(
            np.asarray(
                [centre[0], centre[1], gate_pose[2] + centre[2]],
                dtype=np.float64,
            ),
            half,
            gate_x=0.0,
            gate_base_z=float(gate_pose[2]),
        )
        return penetration, centre, half

    def _refresh_object_poses(self) -> None:
        if not self._launch_isaac:
            for name in (
                "loaded_tray",
                *PAYLOAD_NAMES,
                "loaded_tray_gate",
                "loaded_tray_goal",
                "table",
            ):
                self._poses.setdefault(name, self._fallback_pose(name))
            return
        super()._refresh_object_poses()

    def _loaded_tray_metrics(self, *, update: bool) -> dict[str, Any]:
        tray = self._poses.get("loaded_tray", self._fallback_pose("loaded_tray"))
        payloads = {
            name: self._poses.get(name, self._fallback_pose(name))
            for name in PAYLOAD_NAMES
        }
        gate = self._poses.get(
            "loaded_tray_gate", self._fallback_pose("loaded_tray_gate")
        )
        goal = self._poses.get(
            "loaded_tray_goal", self._fallback_pose("loaded_tray_goal")
        )
        tray_matrix = pose_matrix(tray)
        tray_rotation = tray_matrix[:3, :3]
        left_handle = (
            tray_matrix @ np.asarray([*LEFT_GRIP_LOCAL_POSITION, 1.0])
        )[:3]
        right_handle = (
            tray_matrix @ np.asarray([*RIGHT_GRIP_LOCAL_POSITION, 1.0])
        )[:3]
        grip_y_start, grip_y_end = SIDE_RIM_GRIP_LOCAL_Y_RANGE
        left_rail_start = (
            tray_matrix
            @ np.asarray(
                [
                    LEFT_GRIP_LOCAL_POSITION[0],
                    grip_y_start,
                    LEFT_GRIP_LOCAL_POSITION[2],
                    1.0,
                ]
            )
        )[:3]
        left_rail_end = (
            tray_matrix
            @ np.asarray(
                [
                    LEFT_GRIP_LOCAL_POSITION[0],
                    grip_y_end,
                    LEFT_GRIP_LOCAL_POSITION[2],
                    1.0,
                ]
            )
        )[:3]
        right_rail_start = (
            tray_matrix
            @ np.asarray(
                [
                    RIGHT_GRIP_LOCAL_POSITION[0],
                    grip_y_start,
                    RIGHT_GRIP_LOCAL_POSITION[2],
                    1.0,
                ]
            )
        )[:3]
        right_rail_end = (
            tray_matrix
            @ np.asarray(
                [
                    RIGHT_GRIP_LOCAL_POSITION[0],
                    grip_y_end,
                    RIGHT_GRIP_LOCAL_POSITION[2],
                    1.0,
                ]
            )
        )[:3]

        left_distance = self._hand_distance_to_segment(
            "left", left_rail_start, left_rail_end
        )
        right_distance = self._hand_distance_to_segment(
            "right", right_rail_start, right_rail_end
        )
        left_wrong_distance = self._hand_distance_to_segment(
            "left", right_rail_start, right_rail_end
        )
        right_wrong_distance = self._hand_distance_to_segment(
            "right", left_rail_start, left_rail_end
        )
        left_tray_force, left_tray_contact_available = self._contact_reading(
            self._ltg_hand_contact_views.get("left")
        )
        right_tray_force, right_tray_contact_available = self._contact_reading(
            self._ltg_hand_contact_views.get("right")
        )
        (
            left_handle_force,
            left_non_grip_contact_force,
            left_handle_contact_available,
        ) = self._tray_contact_region_reading(
            self._ltg_hand_contact_views.get("left"),
            tray_matrix,
            LEFT_GRIP_LOCAL_POSITION[0],
        )
        (
            right_handle_force,
            right_non_grip_contact_force,
            right_handle_contact_available,
        ) = self._tray_contact_region_reading(
            self._ltg_hand_contact_views.get("right"),
            tray_matrix,
            RIGHT_GRIP_LOCAL_POSITION[0],
        )
        left_closed = self._hand_closed("left")
        right_closed = self._hand_closed("right")
        left_role_owned = _handle_role_owned(
            contact_available=left_handle_contact_available,
            contact_force_n=left_handle_force,
            hand_closed=left_closed,
            handle_distance_m=left_distance,
            wrong_handle_distance_m=left_wrong_distance,
        )
        right_role_owned = _handle_role_owned(
            contact_available=right_handle_contact_available,
            contact_force_n=right_handle_force,
            hand_closed=right_closed,
            handle_distance_m=right_distance,
            wrong_handle_distance_m=right_wrong_distance,
        )

        tray_half = world_aabb_half_extents(tray, np.asarray(TRAY_HALF_EXTENTS))
        tray_lower = tray[:3] - tray_half
        tray_upper = tray[:3] + tray_half
        (
            tray_gate_penetration,
            tray_gate_centre,
            tray_gate_half,
        ) = self._oriented_box_penetration_with_gate(
            tray,
            np.asarray(TRAY_HALF_EXTENTS, dtype=np.float64),
            gate,
        )
        gate_near = -GATE_POST_HALF_X
        gate_far = GATE_POST_HALF_X
        tray_overlaps_gate = bool(
            float(tray_gate_centre[0] + tray_gate_half[0]) >= gate_near
            and float(tray_gate_centre[0] - tray_gate_half[0]) <= gate_far
        )
        tray_beyond_gate = bool(
            float(tray_gate_centre[0] - tray_gate_half[0])
            >= gate_far + TRAY_GATE_CLEARANCE_M - 1.0e-4
        )

        payload_local: dict[str, np.ndarray] = {}
        payload_deck_gap_m: dict[str, float] = {}
        payload_geometrically_on_tray: dict[str, bool] = {}
        payload_support_force_n: dict[str, float] = {}
        payload_support_available: dict[str, bool] = {}
        payload_supported_by_tray: dict[str, bool] = {}
        payload_on_tray: dict[str, bool] = {}
        payload_beyond_gate: dict[str, bool] = {}
        payload_linear_speeds: dict[str, float] = {}
        payload_velocity_available: dict[str, bool] = {}
        max_gate_penetration = float(tray_gate_penetration)
        payload_dropped = False
        finite_and_in_workspace = bool(np.all(np.isfinite(tray)))
        for name, pose in payloads.items():
            local = tray_rotation.T @ (pose[:3] - tray[:3])
            payload_local[name] = local
            radius = PAYLOAD_HORIZONTAL_RADII[name]
            half_height = PAYLOAD_HALF_HEIGHTS[name]
            payload_rotation = quat_wxyz_to_matrix(pose[3:7])
            upright = bool(
                np.all(np.isfinite(payload_rotation))
                and float(np.dot(payload_rotation[:, 2], tray_rotation[:, 2]))
                >= _PAYLOAD_UPRIGHT_COS
            )
            bottom_local = float(local[2] - half_height)
            deck_gap_m = bottom_local - TRAY_DECK_TOP_LOCAL_Z
            geometrically_on_tray = bool(
                abs(float(local[0])) + radius
                <= TRAY_USABLE_HALF_EXTENTS[0] + _PAYLOAD_LOCAL_FOOTPRINT_TOLERANCE_M
                and abs(float(local[1])) + radius
                <= TRAY_USABLE_HALF_EXTENTS[1] + _PAYLOAD_LOCAL_FOOTPRINT_TOLERANCE_M
                and _payload_bottom_on_deck(bottom_local)
                and upright
            )
            support_force_n, support_available = self._contact_reading(
                self._ltg_payload_support_views.get(name)
            )
            supported_by_tray = bool(
                support_available
                and support_force_n >= _PAYLOAD_SUPPORT_FORCE_N
            )
            payload_deck_gap_m[name] = deck_gap_m
            payload_geometrically_on_tray[name] = geometrically_on_tray
            payload_support_force_n[name] = support_force_n
            payload_support_available[name] = support_available
            payload_supported_by_tray[name] = supported_by_tray
            payload_on_tray[name] = _payload_retained_on_tray(
                geometrically_on_tray=geometrically_on_tray,
                support_available=support_available,
                support_force_n=support_force_n,
            )
            (
                payload_gate_penetration,
                payload_gate_centre,
                payload_gate_half,
            ) = self._cylinder_penetration_with_gate(
                pose,
                radius=radius,
                half_height=half_height,
                gate_pose_wxyz=gate,
            )
            payload_beyond_gate[name] = bool(
                float(payload_gate_centre[0] - payload_gate_half[0])
                >= gate_far + _PAYLOAD_GATE_CLEARANCE_M - 1.0e-4
            )
            z_extent = cylinder_axis_extent(
                pose,
                radius=radius,
                half_height=half_height,
                world_axis=2,
            )
            payload_dropped = payload_dropped or bool(
                float(pose[2]) - z_extent < self._current_table_top_z() - 0.020
            )
            velocity = self._physics_body_velocity(name)
            linear_speed, _angular_speed, velocity_finite = self._velocity_norms(velocity)
            payload_linear_speeds[name] = linear_speed
            payload_velocity_available[name] = velocity is not None
            payload_valid = bool(
                np.all(np.isfinite(pose))
                and np.all(pose[:3] >= _WORKSPACE_LOWER)
                and np.all(pose[:3] <= _WORKSPACE_UPPER)
            )
            if velocity is not None:
                payload_valid = payload_valid and velocity_finite
            finite_and_in_workspace = finite_and_in_workspace and payload_valid
            max_gate_penetration = max(
                max_gate_penetration,
                float(payload_gate_penetration),
            )

        tray_velocity = self._physics_body_velocity("loaded_tray")
        tray_linear_speed, tray_angular_speed, tray_velocity_finite = self._velocity_norms(
            tray_velocity
        )
        if tray_velocity is not None:
            finite_and_in_workspace = finite_and_in_workspace and tray_velocity_finite
        finite_and_in_workspace = finite_and_in_workspace and bool(
            np.all(tray[:3] >= _WORKSPACE_LOWER)
            and np.all(tray[:3] <= _WORKSPACE_UPPER)
        )
        roll_deg, pitch_deg, yaw_deg = rpy_degrees(tray)
        _gate_roll_deg, _gate_pitch_deg, gate_yaw_deg = rpy_degrees(gate)
        yaw_error_deg = _wrap_degrees(
            float(yaw_deg) - (float(gate_yaw_deg) + TRAY_START_YAW_DEG)
        )
        tray_tilt_deg = max(abs(roll_deg), abs(pitch_deg))
        tray_lift_clearance = float(
            tray[2] - TRAY_HALF_EXTENTS[2] - self._current_table_top_z()
        )
        target_error = float(np.linalg.norm(tray[:2] - goal[:2]))
        payloads_retained = bool(all(payload_on_tray.values()))
        all_payloads_beyond = bool(all(payload_beyond_gate.values()))
        all_beyond_gate = bool(tray_beyond_gate and all_payloads_beyond)

        task_gate_contact_readings = [
            self._contact_reading(self._ltg_gate_contact_views.get(name))
            for name in ("loaded_tray", *PAYLOAD_NAMES)
        ]
        hand_gate_contact_readings = {
            side: self._contact_reading(
                self._ltg_hand_gate_contact_views.get(side)
            )
            for side in ("left", "right")
        }
        gate_contact_readings = [
            *task_gate_contact_readings,
            *hand_gate_contact_readings.values(),
        ]
        task_gate_contact_force = float(
            sum(value for value, _available in task_gate_contact_readings)
        )
        hand_gate_contact_force = float(
            sum(value for value, _available in hand_gate_contact_readings.values())
        )
        gate_contact_force = float(
            sum(value for value, _available in gate_contact_readings)
        )
        gate_contact_available = bool(
            gate_contact_readings
            and all(available for _value, available in gate_contact_readings)
        )
        hand_gate_contact_available = bool(
            hand_gate_contact_readings
            and all(
                available
                for _value, available in hand_gate_contact_readings.values()
            )
        )
        # Correct side-rim forces are expected transport contact and must not
        # reset an otherwise valid grasp.  The terminal guard uses only the
        # collider-backed contact manifolds outside the selected grip region;
        # it is stage-armed in
        # LoadedTrayGateEvaluator after dual grasp has been established.
        excessive_handle_contact = bool(
            (
                left_handle_contact_available
                and left_handle_force >= _HANDLE_EXCESSIVE_CONTACT_FORCE_N
            )
            or (
                right_handle_contact_available
                and right_handle_force >= _HANDLE_EXCESSIVE_CONTACT_FORCE_N
            )
        )
        excessive_non_grip_contact = bool(
            left_non_grip_contact_force >= _HANDLE_EXCESSIVE_CONTACT_FORCE_N
            or right_non_grip_contact_force >= _HANDLE_EXCESSIVE_CONTACT_FORCE_N
        )
        final_stable = bool(
            all_beyond_gate
            and payloads_retained
            and target_error <= TARGET_CENTER_TOLERANCE
            and abs(roll_deg) <= _FINAL_ROLL_PITCH_DEG
            and abs(pitch_deg) <= _FINAL_ROLL_PITCH_DEG
            and abs(yaw_error_deg) <= _FINAL_YAW_DEG
            and tray_linear_speed <= _FINAL_TRAY_LINEAR_SPEED_MPS
            and tray_angular_speed <= _FINAL_TRAY_ANGULAR_SPEED_RADPS
            and max(payload_linear_speeds.values(), default=0.0)
            <= _FINAL_PAYLOAD_LINEAR_SPEED_MPS
        )
        frame = LoadedTrayFrame(
            left_role_owned=left_role_owned,
            right_role_owned=right_role_owned,
            tray_lift_clearance_m=tray_lift_clearance,
            tray_overlaps_gate=tray_overlaps_gate,
            all_beyond_gate=all_beyond_gate,
            payloads_on_tray=payloads_retained,
            payload_dropped=payload_dropped,
            final_stable=final_stable,
            tray_tilt_deg=tray_tilt_deg,
            severe_gate_contact=bool(
                gate_contact_available
                and gate_contact_force >= _GATE_SEVERE_CONTACT_FORCE_N
            ),
            excessive_handle_contact=excessive_handle_contact,
            excessive_non_grip_contact=excessive_non_grip_contact,
            max_gate_penetration_m=max_gate_penetration,
            finite_and_in_workspace=finite_and_in_workspace,
        )
        history = (
            self._ltg_evaluator.update(frame)
            if update
            else self._ltg_evaluator.snapshot(success_now=False)
        )
        metrics: dict[str, Any] = {
            **history,
            "left_role_owned": left_role_owned,
            "right_role_owned": right_role_owned,
            "left_hand_closed": left_closed,
            "right_hand_closed": right_closed,
            "left_handle_distance_m": left_distance,
            "right_handle_distance_m": right_distance,
            "left_wrong_handle_distance_m": left_wrong_distance,
            "right_wrong_handle_distance_m": right_wrong_distance,
            "left_tray_contact_force_n": left_tray_force,
            "right_tray_contact_force_n": right_tray_force,
            "left_tray_contact_available": left_tray_contact_available,
            "right_tray_contact_available": right_tray_contact_available,
            "left_handle_contact_force_n": left_handle_force,
            "right_handle_contact_force_n": right_handle_force,
            "left_handle_contact_available": left_handle_contact_available,
            "right_handle_contact_available": right_handle_contact_available,
            "max_handle_contact_force_n": max(
                left_handle_force,
                right_handle_force,
            ),
            "max_tray_contact_force_n": max(
                left_tray_force,
                right_tray_force,
            ),
            "excessive_handle_contact": excessive_handle_contact,
            "left_non_grip_contact_force_n": left_non_grip_contact_force,
            "right_non_grip_contact_force_n": right_non_grip_contact_force,
            "max_non_grip_contact_force_n": max(
                left_non_grip_contact_force,
                right_non_grip_contact_force,
            ),
            "excessive_non_grip_contact": excessive_non_grip_contact,
            "left_handle_position": left_handle,
            "right_handle_position": right_handle,
            "tray_lift_clearance_m": tray_lift_clearance,
            "tray_roll_deg": roll_deg,
            "tray_pitch_deg": pitch_deg,
            "tray_yaw_deg": yaw_deg,
            "tray_yaw_error_deg": yaw_error_deg,
            "tray_tilt_deg": tray_tilt_deg,
            "tray_linear_speed_mps": tray_linear_speed,
            "tray_angular_speed_radps": tray_angular_speed,
            "tray_velocity_available": tray_velocity is not None,
            "target_error_m": target_error,
            "tray_overlaps_gate": tray_overlaps_gate,
            "tray_beyond_gate": tray_beyond_gate,
            "all_payloads_beyond_gate": all_payloads_beyond,
            "all_beyond_gate": all_beyond_gate,
            "payloads_on_tray": payloads_retained,
            "payload_dropped": payload_dropped,
            "payload_on_tray": payload_on_tray,
            "payload_deck_gap_m": payload_deck_gap_m,
            "payload_geometrically_on_tray": payload_geometrically_on_tray,
            "payload_support_force_n": payload_support_force_n,
            "payload_support_available": payload_support_available,
            "payload_supported_by_tray": payload_supported_by_tray,
            "payload_beyond_gate": payload_beyond_gate,
            "payload_local": payload_local,
            "payload_linear_speeds_mps": payload_linear_speeds,
            "payload_velocity_available": payload_velocity_available,
            "gate_near_x": gate_near,
            "gate_far_x": gate_far,
            # These four legacy keys now intentionally describe the sampled
            # gate's local forward axis.  This keeps crossing semantics exact
            # when the fixture yaw is randomized.
            "gate_frame_coordinate_metrics": True,
            "tray_min_x": float(tray_gate_centre[0] - tray_gate_half[0]),
            "tray_max_x": float(tray_gate_centre[0] + tray_gate_half[0]),
            "tray_world_min_x": float(tray_lower[0]),
            "tray_world_max_x": float(tray_upper[0]),
            "gate_yaw_deg": float(gate_yaw_deg),
            "gate_contact_force_n": gate_contact_force,
            "gate_contact_available": gate_contact_available,
            "task_gate_contact_force_n": task_gate_contact_force,
            "hand_gate_contact_force_n": hand_gate_contact_force,
            "hand_gate_contact_available": hand_gate_contact_available,
            "left_hand_gate_contact_force_n": hand_gate_contact_readings["left"][
                0
            ],
            "right_hand_gate_contact_force_n": hand_gate_contact_readings["right"][
                0
            ],
            "max_gate_penetration_m": max_gate_penetration,
            "finite_and_in_workspace": finite_and_in_workspace,
            "final_stable": final_stable,
        }
        if update:
            self._ltg_last_metrics = metrics
        return metrics

    def _apply_loaded_tray_tilt_stabilizer(self) -> None:
        self._ltg_tilt_stabilizer_active = False
        self._ltg_tilt_stabilizer_torque.fill(0.0)
        self._ltg_transport_stabilizer_active = False
        self._ltg_transport_stabilizer_force.fill(0.0)
        if self._sim is None or self._torch is None or self._ltg_evaluator.failed:
            return
        self._refresh_object_poses()
        metrics = self._loaded_tray_metrics(update=False)
        if not (
            self._ltg_evaluator.dual_grasp_streak
            >= _TILT_STABILIZER_MIN_DUAL_GRASP_FRAMES
            and metrics["left_role_owned"]
            and metrics["right_role_owned"]
            and metrics["payloads_on_tray"]
        ):
            return
        tray_pose = self._poses.get("loaded_tray")
        tray_view = self._task_body_views.get("loaded_tray")
        if tray_pose is None or tray_view is None:
            return
        torque = _bounded_tilt_stabilizing_torque(
            tray_pose,
            self._physics_body_velocity("loaded_tray"),
        )
        command_midpoint = 0.5 * (
            np.asarray(self._right_tcp_pose[:3], dtype=np.float64)
            + np.asarray(self._left_tcp_pose[:3], dtype=np.float64)
        )
        gate_pose = self._poses.get(
            "loaded_tray_gate",
            self._fallback_pose("loaded_tray_gate"),
        )
        _gate_roll, _gate_pitch, gate_yaw_deg = rpy_degrees(gate_pose)
        gate_forward_xy, _gate_lateral_xy = _planar_axes(gate_yaw_deg)
        if self._ltg_transport_grasp_offset_x is None:
            self._ltg_transport_grasp_offset_x = (
                float(np.dot(tray_pose[:2] - command_midpoint[:2], gate_forward_xy))
            )
        tray_velocity = self._physics_body_velocity("loaded_tray")
        force = np.zeros(3, dtype=np.float64)
        if float(metrics["tray_lift_clearance_m"]) >= _TRANSPORT_STABILIZER_MIN_LIFT_M:
            force = _bounded_transport_stabilizing_force_along_axis(
                tray_pose[:3],
                command_midpoint,
                self._ltg_transport_grasp_offset_x,
                tray_velocity,
                gate_forward_xy,
            )
            palm_errors = []
            for side, command in (
                ("right", self._right_tcp_pose),
                ("left", self._left_tcp_pose),
            ):
                palm = self._robot_link_pose_matrix(side)
                if palm is None:
                    palm_errors.append(math.inf)
                    continue
                palm_errors.append(
                    float(
                        np.linalg.norm(
                            np.asarray(palm[:3, 3], dtype=np.float64)
                            - np.asarray(command[:3], dtype=np.float64)
                        )
                    )
                )
            tracking_error = max(palm_errors, default=math.inf)
            tracking_scale = _transport_tracking_scale(tracking_error)
            force *= tracking_scale
            self._ltg_transport_tracking_error_m = float(
                tracking_error
                if math.isfinite(tracking_error)
                else _TRANSPORT_TRACKING_ZERO_FORCE_ERROR_M
            )
            self._ltg_transport_tracking_scale = float(tracking_scale)
        try:
            forces = self._torch.zeros(
                (int(tray_view.count), 3),
                dtype=self._torch.float32,
                device=self._isaac_device,
            )
            forces[0] = self._torch.as_tensor(
                force,
                dtype=forces.dtype,
                device=forces.device,
            )
            torques = self._torch.zeros(
                (int(tray_view.count), 3),
                dtype=self._torch.float32,
                device=self._isaac_device,
            )
            torques[0] = self._torch.as_tensor(
                torque,
                dtype=torques.dtype,
                device=torques.device,
            )
            indices = self._torch.arange(
                int(tray_view.count),
                dtype=self._torch.long,
                device=self._isaac_device,
            )
            tray_view.apply_forces_and_torques_at_position(
                forces, torques, None, indices, True
            )
            self._ltg_tilt_stabilizer_active = True
            self._ltg_tilt_stabilizer_torque = torque
            self._ltg_transport_stabilizer_active = bool(
                np.linalg.norm(force) > 0.0
            )
            self._ltg_transport_stabilizer_force = force
        except Exception as exc:
            if not self._ltg_tilt_stabilizer_warning_emitted:
                self._ltg_tilt_stabilizer_warning_emitted = True
                self._add_task_state_warning(
                    f"Could not apply loaded-tray tilt stabilizer: {exc}"
                )

    def _after_task_physics_substep(self) -> None:
        super()._after_task_physics_substep()
        self._apply_loaded_tray_tilt_stabilizer()

    def _compute_success_now(self) -> bool:
        return bool(self._loaded_tray_metrics(update=True)["success_now"])

    def _compute_failure(self) -> bool:
        return bool(self._ltg_evaluator.failed)

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._ltg_last_metrics or self._loaded_tray_metrics(update=False)
        stage = int(metrics["stage"])
        rows = [
            self._condition_row(
                "ltg_dual_roles",
                "Left/right hands own their corresponding bridge handles",
                bool(metrics["left_role_owned"] and metrics["right_role_owned"]),
            ),
            self._condition_row(
                "ltg_lifted",
                "Loaded tray was lifted at least 55 mm",
                stage >= int(LoadedTrayStage.LIFTED),
            ),
            self._condition_row(
                "ltg_crossing_proof",
                "Dual-hand ownership stayed valid through the narrow frame",
                bool(
                    metrics["crossing_started"]
                    and metrics["crossing_ownership_valid"]
                    and stage >= int(LoadedTrayStage.GATE_CROSSED)
                ),
            ),
            self._condition_row(
                "ltg_tray_beyond",
                "The complete tray cleared the far frame face",
                bool(metrics["tray_beyond_gate"]),
            ),
            self._condition_row(
                "ltg_payloads_beyond",
                "Both independent payloads cleared the frame",
                bool(metrics["all_payloads_beyond_gate"]),
            ),
            self._condition_row(
                "ltg_payloads_retained",
                "Both payloads remain upright on the tray",
                bool(metrics["payloads_on_tray"]),
            ),
            self._condition_row(
                "ltg_final_stable",
                "Loaded tray is level and stable over the green target",
                bool(metrics["final_stable"]),
            ),
            self._condition_row(
                "ltg_no_failure",
                "No drop, deep collision, role loss, or excessive tilt",
                not bool(metrics["failed"]),
            ),
        ]
        return self._append_success_hold_row(rows)

    def _robot_state(self) -> tuple[np.ndarray, np.ndarray]:
        tcp = np.concatenate([self._right_tcp_pose, self._left_tcp_pose]).astype(
            np.float64
        )
        gripper = np.zeros(32, dtype=np.float64)
        gripper[: min(16, self._right_hand.size)] = self._right_hand[
            : min(16, self._right_hand.size)
        ]
        gripper[16 : 16 + min(16, self._left_hand.size)] = self._left_hand[
            : min(16, self._left_hand.size)
        ]
        return tcp, gripper

    def _compute_observation(self) -> dict[str, Any]:
        metrics = self._ltg_last_metrics or self._loaded_tray_metrics(update=False)
        tcp, gripper = self._robot_state()
        tray = self._poses.get("loaded_tray", self._fallback_pose("loaded_tray"))
        payloads = [
            self._poses.get(name, self._fallback_pose(name)) for name in PAYLOAD_NAMES
        ]
        gate = self._poses.get(
            "loaded_tray_gate", self._fallback_pose("loaded_tray_gate")
        )
        goal = self._poses.get(
            "loaded_tray_goal", self._fallback_pose("loaded_tray_goal")
        )
        required = max(1, self._success_steps_required())
        timeout = bool(self.env_step * self._control_dt >= self._time_limit)
        task_progress = np.asarray(
            [
                float(metrics["stage"]) / float(LoadedTrayStage.PLACED_STABLE),
                min(1.0, float(metrics["dual_grasp_streak"]) / 3.0),
                min(1.0, float(metrics["crossing_role_loss_streak"]) / 3.0),
                float(bool(metrics["crossing_started"])),
                float(bool(metrics["crossing_ownership_valid"])),
                float(bool(metrics["tray_beyond_gate"])),
                float(bool(metrics["all_payloads_beyond_gate"])),
                float(bool(metrics["payloads_on_tray"])),
                float(bool(metrics["final_stable"])),
                float(bool(metrics["failed"])),
                min(1.0, float(self._success_counter) / float(required)),
                float(timeout),
            ],
            dtype=np.float64,
        )
        tray_metrics = np.asarray(
            [
                metrics["tray_lift_clearance_m"],
                metrics["target_error_m"],
                metrics["tray_roll_deg"],
                metrics["tray_pitch_deg"],
                metrics["tray_yaw_deg"],
                metrics["tray_linear_speed_mps"],
                metrics["tray_angular_speed_radps"],
                max(metrics["payload_linear_speeds_mps"].values(), default=0.0),
                metrics["left_handle_distance_m"],
                metrics["right_handle_distance_m"],
                metrics["left_handle_contact_force_n"],
                metrics["right_handle_contact_force_n"],
            ],
            dtype=np.float64,
        )
        gate_metrics = np.asarray(
            [
                metrics["tray_min_x"],
                metrics["tray_max_x"],
                metrics["gate_near_x"],
                metrics["gate_far_x"],
                metrics["gate_contact_force_n"],
                metrics["max_gate_penetration_m"],
            ],
            dtype=np.float64,
        )
        payload_metrics = np.concatenate(
            [
                np.asarray(
                    [
                        *metrics["payload_local"][name],
                        float(metrics["payload_on_tray"][name]),
                        float(metrics["payload_beyond_gate"][name]),
                        metrics["payload_linear_speeds_mps"][name],
                    ],
                    dtype=np.float64,
                )
                for name in PAYLOAD_NAMES
            ]
        )
        observation: dict[str, Any] = {
            "state": {
                "tcp_pose": tcp,
                "gripper_pose": gripper,
                "tray_pose": tray.copy(),
                "payload_poses": np.concatenate(payloads).astype(np.float64),
                "gate_pose": gate.copy(),
                "goal_pose": goal.copy(),
                "handle_positions": np.concatenate(
                    [metrics["left_handle_position"], metrics["right_handle_position"]]
                ).astype(np.float64),
                "task_progress": task_progress,
                "tray_metrics": tray_metrics,
                "gate_metrics": gate_metrics,
                "payload_metrics": payload_metrics,
                "table_delta_height": self.delta_h.astype(np.float64, copy=True),
            }
        }
        if self.image_obs:
            front, wrist_left, wrist_right = self.render()
            observation["images"] = {
                "random_camera" if self.randomize else "front": front,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }
        return observation

    def _task_info(self) -> dict[str, Any]:
        metrics = self._ltg_last_metrics or self._loaded_tray_metrics(update=False)
        timeout = bool(
            self.env_step * self._control_dt >= self._time_limit
            and self._success_counter < self._success_steps_required()
        )
        if metrics["failed"]:
            termination_reason = str(metrics["failure_reason"])
        elif timeout:
            termination_reason = "timeout"
        elif self._success_counter >= self._success_steps_required():
            termination_reason = "success"
        else:
            termination_reason = ""
        info: dict[str, Any] = {
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "task_stage": metrics["stage_name"],
            "task_stage_index": int(metrics["stage"]),
            "success_counter": int(self._success_counter),
            "success_steps_required": int(self._success_steps_required()),
            "timeout": timeout,
            "termination_reason": termination_reason,
            "failure_reason": str(metrics["failure_reason"]),
        }
        for key, value in metrics.items():
            if isinstance(value, (bool, np.bool_)):
                info[key] = bool(value)
            elif isinstance(value, (int, np.integer)):
                info[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                info[key] = float(value)
            elif isinstance(value, str):
                info[key] = value
        info["payload_on_tray"] = dict(metrics["payload_on_tray"])
        info["payload_deck_gap_m"] = dict(metrics["payload_deck_gap_m"])
        info["payload_geometrically_on_tray"] = dict(
            metrics["payload_geometrically_on_tray"]
        )
        info["payload_support_force_n"] = dict(
            metrics["payload_support_force_n"]
        )
        info["payload_support_available"] = dict(
            metrics["payload_support_available"]
        )
        info["payload_supported_by_tray"] = dict(
            metrics["payload_supported_by_tray"]
        )
        info["payload_beyond_gate"] = dict(metrics["payload_beyond_gate"])
        info["tilt_stabilizer_active"] = bool(
            self._ltg_tilt_stabilizer_active
        )
        info["tilt_stabilizer_torque_nm"] = float(
            np.linalg.norm(self._ltg_tilt_stabilizer_torque)
        )
        info["tilt_stabilizer_max_torque_nm"] = float(
            _TILT_STABILIZER_MAX_TORQUE_NM
        )
        info["transport_stabilizer_active"] = bool(
            self._ltg_transport_stabilizer_active
        )
        info["transport_stabilizer_force_n"] = float(
            np.linalg.norm(self._ltg_transport_stabilizer_force)
        )
        info["transport_stabilizer_max_force_n"] = float(
            _TRANSPORT_STABILIZER_MAX_FORCE_N
        )
        info["excessive_handle_contact_force_threshold_n"] = float(
            _HANDLE_EXCESSIVE_CONTACT_FORCE_N
        )
        info["reset_tray_x_m"] = float(self._ltg_reset_tray_xy[0])
        info["reset_tray_y_m"] = float(self._ltg_reset_tray_xy[1])
        info["reset_tray_yaw_deg"] = float(self._ltg_reset_tray_yaw_deg)
        info["reset_gate_x_m"] = float(self._ltg_reset_gate_xy[0])
        info["reset_gate_y_m"] = float(self._ltg_reset_gate_xy[1])
        info["reset_gate_yaw_deg"] = float(self._ltg_reset_gate_yaw_deg)
        info["reset_goal_x_m"] = float(self._ltg_reset_goal_xy[0])
        info["reset_goal_y_m"] = float(self._ltg_reset_goal_xy[1])
        info["reset_goal_yaw_deg"] = float(self._ltg_reset_goal_yaw_deg)
        info["reset_layout_attempts"] = int(self._ltg_reset_layout_attempts)
        info["reset_layout_fallback"] = bool(self._ltg_reset_layout_fallback)
        info["gate_clear_width_m"] = float(GATE_CLEAR_WIDTH)
        info["gate_full_depth_collider_count"] = int(
            _BC005_EXPECTED_FRAME_COLLISION_BOX_COUNT
        )
        info["reset_layout"] = {
            "tray_xy_m": self._ltg_reset_tray_xy.astype(float).tolist(),
            "tray_yaw_deg": float(self._ltg_reset_tray_yaw_deg),
            "gate_xy_m": self._ltg_reset_gate_xy.astype(float).tolist(),
            "gate_yaw_deg": float(self._ltg_reset_gate_yaw_deg),
            "goal_xy_m": self._ltg_reset_goal_xy.astype(float).tolist(),
            "goal_yaw_deg": float(self._ltg_reset_goal_yaw_deg),
            "payload_local_xy_m": {
                name: self._ltg_reset_payload_local_xy[name].astype(float).tolist()
                for name in PAYLOAD_NAMES
            },
            "attempts": int(self._ltg_reset_layout_attempts),
            "fallback": bool(self._ltg_reset_layout_fallback),
        }
        info["transport_tracking_error_m"] = float(
            self._ltg_transport_tracking_error_m
        )
        info["transport_tracking_scale"] = float(
            self._ltg_transport_tracking_scale
        )
        info["crossing_nullspace_active"] = bool(
            all(self._ltg_crossing_nullspace_active.values())
        )
        info["crossing_nullspace_max_step_rad"] = float(
            max(self._ltg_crossing_nullspace_step_rad.values(), default=0.0)
        )
        return info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if terminated and not info.get("succeed") and not info.get("failed"):
            info["timeout"] = True
            info["termination_reason"] = "timeout"
        return observation, reward, terminated, truncated, info

    def restore_recorded_initial_state(
        self, state_parts: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        super().restore_recorded_initial_state(state_parts)
        for key, name in (
            ("tray_pose", "loaded_tray"),
            ("gate_pose", "loaded_tray_gate"),
            ("goal_pose", "loaded_tray_goal"),
        ):
            pose = np.asarray(
                state_parts.get(key, np.zeros(0)), dtype=np.float64
            ).ravel()
            view = self._task_body_views.get(name)
            if pose.size == 7 and view is not None:
                self._set_task_body_world_pose(view, pose)
        payload_poses = np.asarray(
            state_parts.get("payload_poses", np.zeros(0)), dtype=np.float64
        ).ravel()
        for index, name in enumerate(PAYLOAD_NAMES):
            pose = payload_poses[index * 7 : (index + 1) * 7]
            view = self._task_body_views.get(name)
            if pose.size == 7 and view is not None:
                self._set_task_body_world_pose(view, pose)
        tray_pose = np.asarray(
            state_parts.get("tray_pose", np.zeros(0)), dtype=np.float64
        ).ravel()
        gate_pose = np.asarray(
            state_parts.get("gate_pose", np.zeros(0)), dtype=np.float64
        ).ravel()
        goal_pose = np.asarray(
            state_parts.get("goal_pose", np.zeros(0)), dtype=np.float64
        ).ravel()
        if tray_pose.size == 7:
            self._ltg_reset_tray_xy = tray_pose[:2].copy()
            self._ltg_reset_tray_yaw_deg = float(rpy_degrees(tray_pose)[2])
            tray_rotation = quat_wxyz_to_matrix(tray_pose[3:7])[:2, :2]
            if payload_poses.size == 7 * len(PAYLOAD_NAMES) and np.all(
                np.isfinite(tray_rotation)
            ):
                for index, name in enumerate(PAYLOAD_NAMES):
                    payload_pose = payload_poses[7 * index : 7 * (index + 1)]
                    self._ltg_reset_payload_local_xy[name] = (
                        tray_rotation.T @ (payload_pose[:2] - tray_pose[:2])
                    )
        if gate_pose.size == 7:
            self._ltg_reset_gate_xy = gate_pose[:2].copy()
            self._ltg_reset_gate_yaw_deg = float(rpy_degrees(gate_pose)[2])
        if goal_pose.size == 7:
            self._ltg_reset_goal_xy = goal_pose[:2].copy()
            self._ltg_reset_goal_yaw_deg = float(rpy_degrees(goal_pose)[2])
        if self._sim is not None:
            self._sim.forward()
        for robot in self._robots.values():
            robot.update(0.0)
        self._refresh_object_poses()
        self._ltg_evaluator.reset()
        self._ltg_last_metrics = {}
        return self._compute_observation()


__all__ = ["HandsOnLoadedTrayGateIsaacLabEnv"]
