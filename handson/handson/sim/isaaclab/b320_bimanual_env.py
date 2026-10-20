"""Native IsaacLab runtime and formal predicates for rigid B320 tasks."""

from __future__ import annotations

import math
import os
from typing import Any

from gymnasium import spaces
import numpy as np

from handson.runtime.collection_support import (
    BC011_COLLECTION_ANGLE_GUIDANCE_ENV,
)

from .abc_collection_env import HandsOnABCIsaacLabEnv
from .b320_ka005_2_contract import (
    CONTACT_KEYS as KA005_2_CONTACT_KEYS,
    DOF_NAMES as KA005_2_DOF_NAMES,
    DualKnobResetSample,
    DualKnobSyncState,
    KnobStage,
    sample_dual_knob_reset,
)
from .b320_bimanual_scene import (
    B320_ASSET_ROOT_PATH,
    BC011_PEN_TIP_AXIS_LOCAL,
    BC011_PEN_VARIANTS,
    BC014_FLOW_PROXY_PATHS,
    BC014_GOAL_CENTERS_XY,
    BC014_GOAL_MARKER_PATHS,
    BC014_GOAL_ZONE_SIZE_M,
    BC014_JAR_TABLE_CLEARANCE_M,
    BC014_JAR_TARGET_HEIGHT_M,
    B320_BC014_TASK_NAME,
    B320_BC017_TASK_NAME,
    B320_CONTROL_HZ,
    B320_KA020_TASK_NAME,
    B320_KA021_TASK_NAME,
    B320_FAUCET_TASK_NAMES,
    XY4_TASK_NAME,
    B320_PHYSICS_HZ,
    B320_KA005_2_TARGET_MARKER_PATHS,
    B320_KA005_2_TASK_NAME,
    GEN6_TASK_NAME,
    B320_DUAL_KNOB_TASK_NAMES,
    B320_RIGID_TASK_NAMES,
    B320_TABLE_PATH,
    B320_TABLE_TOP_Z,
    B320_TASK_SPECS,
    FU001_DOOR_SOURCE_PIVOTS,
    FU001_ROOT_HOME,
    FU001_RUNTIME_ORIENTATION_WXYZ,
    FU001_RUNTIME_SCALE,
    KA021_ASSET_BUNDLE_KEY,
    KA021_FAUCET_ASSET_BUNDLE_KEY,
    KA021_FAUCET_BASE_POSITION_W,
    KA021_CUP_OPENING_RADIUS_M,
    KA021_CUP_PARK_POSITION_W,
    KA021_CUP_PARK_MARKER_PATH,
    KA021_CUP_PARK_TOLERANCE_M,
    KA021_CUP_PARK_Z_TOLERANCE_M,
    KA021_CUP_RIM_HEIGHT_M,
    KA021_CUP_UP_AXIS_LOCAL,
    KA021_CUP_UPRIGHT_TOLERANCE_DEG,
    KA021_FAUCET_GRIP_COLLIDER_SIZE_M,
    KA021_FAUCET_HANDLE_PIVOT_W,
    KA021_FAUCET_INITIAL_OPEN_DIRECTION_W,
    KA021_FAUCET_JOINT_AXIS_WORLD_W,
    KA021_FAUCET_JOINT_FRAME_ORIENTATION_WXYZ,
    KA021_FAUCET_OPEN_AXIS,
    KA021_FAUCET_OPEN_LIMIT_DEG,
    KA021_FAUCET_OPEN_TRAVEL_RAD,
    KA021_FAUCET_OUTWARD_DIRECTION_W,
    KA021_FAUCET_RESET_POSITION_RAD,
    KA021_FAUCET_SOURCE_CREATOR,
    KA021_FAUCET_SOURCE_LICENSE,
    KA021_FAUCET_SOURCE_SHA256,
    KA021_FAUCET_SOURCE_URL,
    KA021_HAND_CONTACT_OFFSET,
    KA021_FLOW_PROXY_PATH,
    KA021_FLOW_VISIBLE_FRACTION,
    KA021_MAX_OUTLET_RIM_GAP_M,
    KA021_MIN_OUTLET_RIM_GAP_M,
    KA021_OUTLET_POSITION_W,
    KA021_SAFE_BASIN_FLOOR_CENTER_W,
    KA021_SAFE_BASIN_FLOOR_PATH,
    KA021_SAFE_BASIN_FLOOR_SIZE_M,
    KA021_SHARED_SCENE_COLLISION_POLICY,
    KA021_SHARED_SCENE_COLLISION_ROOT_PATH,
    KA021_SHARED_SCENE_COLLISION_SPECS,
    KA021_VISIBLE_BASIN_OPENING_BOUNDS_XY,
    KA021_CUP_PARK_SUPPORT_PATH,
    KA021_CUP_PARK_SUPPORT_SIZE_M,
    KA021_COUNTER_SURFACE_COLLISION_PATH,
    KA021_COUNTER_SURFACE_COLLISION_SPECS,
    KA021_COUNTER_SURFACE_Z,
    KA021_MUG_HANDLE_COLLIDER_RADIUS_M,
    KA021_MUG_HANDLE_CENTERLINE_SOURCE_M,
    KA021_SUCCESS_OPEN_FRACTION,
    KA020_FLOW_PROXY_PATH,
    KA020_PUMP_HEAD_CENTER_LOCAL_M,
    KA020_PUMP_OUTLET_OFFSET_M,
    KA020_PUMP_TRAVEL_M,
    KA020_SPONGE_PLAN_HALF_EXTENTS_M,
    KA020_SPONGE_THICKNESS_M,
    B320SpawnResult,
    _fu001_runtime_point,
    _set_xform,
    bc011_tip_up_goal_pose_wxyz,
    spawn_b320_task_assets,
)
from .recording_cameras import WRIST_RECORDING_CAMERA_OUTPUT_SIZE
from .realistic_kitchen_scene import (
    RealisticKitchenReport,
    compose_realistic_kitchen_visual,
    realistic_kitchen_metadata,
)


_HAND_CONTACT_N = 0.10
_RELEASE_CONTACT_N = 0.05
_BC011_PICKUP_LIFT_M = 0.010
_BC011_TIP_UP_TOLERANCE_DEG = 2.0
_BC011_TABLE_CONTACT_MAX_N = 0.02
_BC011_DROP_DEBOUNCE_SUBSTEPS = 2
_DRAWER_TRAVEL_M = 0.36 * 0.55
_HINGE_TRAVEL_RAD = math.pi / 2.0
_SOAP_TRAVEL_M = KA020_PUMP_TRAVEL_M
_BC014_KITCHEN_ROOT = "/World/VisualEnvironment/BC014Kitchen"
# BC014 jointly randomizes each shaker and its matching target inside the
# assigned hand's central reach band. These bounds leave at least about 70 mm
# of radial margin inside the physical arm workspace and keep every 100 mm
# target square wholly on its assigned side of the table.
_BC014_RESET_MAX_ATTEMPTS = 128
_BC014_RESET_YAW_DEG = 20.0
_BC014_JAR_HALF_EXTENT_M = 0.029
_BC014_MIN_START_SEPARATION_M = 0.085
_BC014_MIN_JAR_GOAL_SEPARATION_M = 0.120
_BC014_COMFORT_CENTER_BOUNDS_XY = {
    "salt_jar": ((-0.42, 0.11), (-0.21, 0.30)),
    "pepper_jar": ((-0.42, -0.30), (-0.21, -0.11)),
}
_BC014_CANONICAL_ARM_BASE_POSITIONS = {
    "salt_jar": (-0.75, 0.065, 1.34),
    "pepper_jar": (-0.75, -0.065, 1.34),
}
_BC014_COMFORT_REACH_Z_M = 0.970
_BC014_COMFORT_MAX_RADIUS_M = 0.710
# KA012 samples the pot as one seeded reset variable instead of applying the
# generic per-body jitter.  The rectangle is the useful intersection between
# the left-hand handle workspace and the right-hand stirring workspace.  A
# 30 mm radial margin is retained inside the arm's 0.78 m hard workspace.
_KA012_TASK_NAME = "t45_stir_pot"
_KA012_RESET_MAX_ATTEMPTS = 128
_KA012_RESET_YAW_DEG = 8.0
_KA012_POT_CENTER_BOUNDS_XY = ((-0.22, 0.04), (-0.17, 0.12))
_KA012_CANONICAL_ARM_BASE_POSITIONS = {
    "left": (-0.75, 0.065, 1.34),
    "right": (-0.75, -0.065, 1.34),
}
_KA012_COMFORT_MAX_RADIUS_M = 0.750
# Offsets are expressed in the scaled pot asset's source frame.  Source +Y
# becomes world +Z under the authored +90 degree X rotation.
_KA012_LEFT_HANDLE_LOCAL_OFFSET_M = (-0.090, 0.055, 0.0)
_KA012_STIR_CENTER_LOCAL_OFFSET_M = (0.0, 0.110, 0.0)
_KA012_STIR_PATH_RADIUS_M = 0.060
_KA012_POT_FOOTPRINT_RADIUS_M = 0.103
# KA020 samples both task bodies as one seeded, supported, side-assigned
# layout.  The selected dispenser's nozzle reach is included in the pair
# clearance, while the interaction points retain margin inside the 0.78 m arm
# workspace used by the shared ROKAE controller.
_KA020_RESET_MAX_ATTEMPTS = 64
_KA020_RESET_YAW_DEG = 12.0
_KA020_SPONGE_CENTER_BOUNDS_XY = ((-0.24, 0.14), (-0.18, 0.23))
_KA020_BOTTLE_CENTER_BOUNDS_XY = ((-0.24, -0.23), (-0.18, -0.14))
_KA020_CANONICAL_ARM_BASE_POSITIONS = {
    "sponge": (-0.75, 0.065, 1.34),
    "soap_bottle": (-0.75, -0.065, 1.34),
}
_KA020_INTERACTION_HEIGHT_OFFSET_M = {
    "sponge": 0.030,
    "soap_bottle": 0.130,
}
_KA020_COMFORT_MAX_RADIUS_M = 0.740
_KA020_MIN_BODY_SEPARATION_M = 0.240
_KA020_NOZZLE_MIN_HEIGHT_M = 0.020
_KA020_NOZZLE_MAX_HEIGHT_M = 0.080
_KA020_SPONGE_EDGE_MARGIN_M = 0.003
_KA020_PUMP_CONTACT_N = 0.05
# KA021 samples the mug and the outside-sink placement pad as one seeded
# episode layout.  The mug bounds stay inside both the visible deep basin
# opening and its replacement floor; the ribbed drainboard on the opening's
# negative-Y side is explicitly excluded even though the hidden floor extends
# underneath it.  The pad bounds stay on the reviewed left-counter polygon
# and therefore on the thin counter strip colliders authored for KA021.  The
# reset envelope
# includes the complete visual mesh, then retains another 20 mm from the
# visible basin edge so no randomized yaw can visually park part of the mug
# outside the sink.
_KA021_RESET_MAX_ATTEMPTS = 128
_KA021_RESET_YAW_DEG = 25.0
_KA021_CUP_RESET_CENTER_BOUNDS_XY = ((-0.25, -0.13), (-0.04, 0.06))
_KA021_CUP_PARK_CENTER_BOUNDS_XY = ((-0.450, -0.48), (-0.440, -0.20))
_KA021_CUP_PARK_MIN_SEPARATION_M = 0.13
# A vertex sweep of the hash-locked mug USD reports a 73.379 mm maximum
# visual radius; round upward to 74 mm.  This also contains the 67.5 mm
# handle-collider reach and the 40.5 mm mug-body radius.
_KA021_CUP_FOOTPRINT_RADIUS_M = 0.074
_KA021_CUP_RESET_EDGE_CLEARANCE_M = 0.020
_KA021_BASIN_FLOOR_TOP_Z_M = float(
    KA021_SAFE_BASIN_FLOOR_CENTER_W[2]
    + 0.5 * KA021_SAFE_BASIN_FLOOR_SIZE_M[2]
)
# These are the two triangles of the measured HS005 left countertop top
# polygon in world XY.  Requiring all four marker corners to lie in their
# union prevents a random marker from drifting across the sink opening/rim.
_KA021_LEFT_COUNTER_TOP_TRIANGLES_XY = (
    (
        (-0.4776436, 0.2469145),
        (-0.3717142, -0.6631171),
        (-0.3706233, 0.2468897),
    ),
    (
        (-0.3717142, -0.6631171),
        (-0.4776436, 0.2469145),
        (-0.5308915, -0.6296456),
    ),
)
# A shaker emits the D09-style water proxy only once its local cap axis points
# at least 10 degrees below horizontal.  The projected ray must reach the
# unchanged tabletop target plane within a physically reachable arm span.
_BC014_FLOW_TILT_THRESHOLD_DEG = 100.0
_BC014_FLOW_MAX_LENGTH_M = 0.55
_BC014_FLOW_TARGET_Z_M = B320_TABLE_TOP_Z + 0.012
_BC014_FLOW_REQUIRED_SECONDS = 0.60
_BC014_FLOW_OVERLAP_REQUIRED_SECONDS = 0.40
_BC014_FLOW_MAX_POUR_SECONDS = 4.0

# The vendored dv_carton mesh has local bounds x=[-0.199, 0.203],
# y=[-0.234, 0.236], z=[0.000, 0.285] metres.  BC004 assigns the left hand
# to the +Y lower side and the right hand to the -Y lower side.  Keep the
# acceptance zones inset from the exact mesh boundary so small contact-offset
# changes do not alter the task, while still rejecting the broad front/back,
# top, and same-side false positives.
_CARTON_SIDE_ZONE_MIN_ABS_Y_M = 0.115
_CARTON_LOWER_ZONE_MAX_Z_M = 0.145
_CARTON_ZONE_MAX_ABS_X_M = 0.225
_CARTON_ZONE_MIN_Z_M = -0.035
_CARTON_LOCAL_MIN_M = np.asarray([-0.199, -0.234, 0.000], dtype=np.float64)
_CARTON_LOCAL_MAX_M = np.asarray([0.203, 0.236, 0.285], dtype=np.float64)
_CARTON_AIRBORNE_CLEARANCE_M = 0.010
_CARTON_AIRBORNE_HOLD_SECONDS = 3.0
_CARTON_AIRBORNE_REQUIRED_SUBSTEPS = int(
    math.ceil(_CARTON_AIRBORNE_HOLD_SECONDS * B320_PHYSICS_HZ)
)
_CARTON_TABLE_CONTACT_MAX_N = 0.02
_CARTON_DROP_DISTANCE_M = 0.050
_CARTON_FORCE_LIMIT_N = 12.0
# BC004 saturates every hand-to-carton PhysX force sample at 10 N before the
# value enters control, peak tracking, success/failure evaluation, or recording.
_CARTON_RECORDED_FORCE_MAX_N = 10.0
# Each hand owns an independent Cartesian velocity barrier.  A conservative
# contact-stiffness floor converts the requested inward displacement into a
# one-control-frame force prediction; recent positive dF/dt supplies a second
# prediction.  If either says the hand would cross 10 N, only that hand's
# compressive horizontal velocity component is rejected.  The benchmark's
# formal failure ceiling remains 12 N.
_CARTON_FORCE_VELOCITY_GUARD_N = 10.0
_CARTON_FORCE_GUARD_MIN_STIFFNESS_N_PER_M = 10_000.0
_CARTON_FORCE_GUARD_RATE_MEMORY_SECONDS = 0.10
# A target is physically verified only while its measured response still has
# two newtons of headroom below the command guard.  Above this point, retain the
# most recent verified physical position so delayed position-control tracking
# cannot silently turn an old command into additional compression.
_CARTON_FORCE_GUARD_VERIFY_MAX_N = 8.0
_CARTON_BILATERAL_LOCK_FORCE_N = 10.0
_CARTON_FORCE_HOLD_MIN_N = 9.75
_CARTON_FORCE_HOLD_MAX_INWARD_STEP_M = 0.00025
# BC004 is position controlled.  At the command barrier, rebase the drive at
# the measured mechanism state and add one small outward unloading step.
# Merely ignoring the next tracker packet leaves old position error in the
# servo and was observed to produce 233--303 N table jams.
_CARTON_FORCE_RELIEF_STEP_M = 0.0005
_CARTON_QUATERNION_NORM_TOLERANCE = 1.0e-3

# Observation values must remain finite even while a task predicate is not
# yet measurable.  The old ``math.inf`` sentinels were accepted by the broad
# ``Box(-inf, inf)`` spaces, but they make the live observation contract
# unusable for recorders/models that require finite tensors.  These sentinels
# are deliberately just outside the corresponding success thresholds and the
# predicate flags still carry the authoritative validity information.
_UNAVAILABLE_ARTICULATION_DELTA_S = 2.0
_UNAVAILABLE_REFERENCE_MOTION_M = 1.0
_INVALID_CONTACT_FORCE_N = 2.0 * _CARTON_FORCE_LIMIT_N

_KA005_2_FAILURE_REASON_CODES = {
    "dof_truth_unavailable": 1,
    "contact_truth_unavailable": 2,
    "joint_limit_exceeded": 3,
    "panel_base_drift": 4,
}
_KA005_2_OTHER_FAILURE_REASON_CODE = 5
_STRICT_30_120_TIMING_TASKS = {
    "eval_comb06_season_and_stir",
    "t10_lift_carton",
    B320_BC017_TASK_NAME,
    "t01_align_blocks_in_midair",
    B320_KA020_TASK_NAME,
    *B320_DUAL_KNOB_TASK_NAMES,
}


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quaternion / norm


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = _normalize_quaternion(lhs)
    rw, rx, ry, rz = _normalize_quaternion(rhs)
    return _normalize_quaternion(
        np.asarray(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float64,
        )
    )


def _rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(quaternion_wxyz)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_world(pose_wxyz: np.ndarray, local_axis: tuple[float, float, float]) -> np.ndarray:
    axis = _rotation_matrix(pose_wxyz[3:7]) @ np.asarray(local_axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    return axis / max(norm, 1.0e-12)


def _quaternion_from_z_axis(direction: np.ndarray) -> np.ndarray:
    """Return a WXYZ rotation that maps local +Z onto ``direction``."""

    target = np.asarray(direction, dtype=np.float64).reshape(3)
    target /= max(float(np.linalg.norm(target)), 1.0e-12)
    dot = float(np.clip(target[2], -1.0, 1.0))
    if dot <= -1.0 + 1.0e-9:
        return np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    cross = np.asarray([-target[1], target[0], 0.0], dtype=np.float64)
    return _normalize_quaternion(
        np.asarray([1.0 + dot, *cross], dtype=np.float64)
    )


def _bc014_flow_projection(
    pose_wxyz: np.ndarray,
    goal_center_xy: tuple[float, float],
) -> dict[str, Any]:
    """Project one downward-facing shaker cap onto its assigned target."""

    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    cap_axis = _axis_world(pose, (0.0, 0.0, 1.0))
    cap_position = pose[:3] + cap_axis * BC014_JAR_TARGET_HEIGHT_M
    tilt_deg = _angle_degrees(cap_axis, np.asarray([0.0, 0.0, 1.0]))
    downward_enough = bool(tilt_deg >= _BC014_FLOW_TILT_THRESHOLD_DEG)
    impact = cap_position.copy()
    length_m = 0.0
    ray_valid = False
    if downward_enough and cap_axis[2] < -1.0e-6:
        length_m = float(
            (_BC014_FLOW_TARGET_Z_M - cap_position[2]) / cap_axis[2]
        )
        ray_valid = bool(
            0.010 <= length_m <= _BC014_FLOW_MAX_LENGTH_M
            and cap_position[2] > _BC014_FLOW_TARGET_Z_M
        )
        if ray_valid:
            impact = cap_position + length_m * cap_axis

    goal = np.asarray(goal_center_xy, dtype=np.float64)
    zone_offset = impact[:2] - goal
    zone_half_extent = 0.5 * np.asarray(
        BC014_GOAL_ZONE_SIZE_M[:2],
        dtype=np.float64,
    )
    zone_margin = float(np.min(zone_half_extent - np.abs(zone_offset)))
    return {
        "cap_axis": cap_axis,
        "cap_position": cap_position,
        "impact_position": impact,
        "stream_length_m": length_m if ray_valid else 0.0,
        "tilt_deg": tilt_deg,
        "ray_valid": ray_valid,
        "zone_offset_xy_m": zone_offset,
        "zone_margin_m": zone_margin,
        "projects_into_zone": bool(
            ray_valid and np.all(np.abs(zone_offset) <= zone_half_extent)
        ),
    }


def _ka021_flow_projection(
    cup_pose_wxyz: np.ndarray,
    faucet_open_fraction: float,
) -> dict[str, Any]:
    """Return the faucet stream/cup-opening relation from physical state."""

    cup = np.asarray(cup_pose_wxyz, dtype=np.float64).reshape(7)
    outlet = np.asarray(KA021_OUTLET_POSITION_W, dtype=np.float64)
    rim_center = cup[:3] + _rotation_matrix(cup[3:7]) @ np.asarray(
        [0.0, 0.0, KA021_CUP_RIM_HEIGHT_M],
        dtype=np.float64,
    )
    opening_xy_error_m = float(np.linalg.norm(rim_center[:2] - outlet[:2]))
    cup_rim_z_m = float(rim_center[2])
    outlet_rim_gap_m = float(outlet[2] - cup_rim_z_m)
    cup_up_axis = _axis_world(cup, KA021_CUP_UP_AXIS_LOCAL)
    cup_tilt_deg = _angle_degrees(
        cup_up_axis,
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    )
    upright = bool(cup_tilt_deg <= KA021_CUP_UPRIGHT_TOLERANCE_DEG + 1.0e-6)
    opening_aligned = bool(opening_xy_error_m <= KA021_CUP_OPENING_RADIUS_M)
    height_valid = bool(
        KA021_MIN_OUTLET_RIM_GAP_M
        <= outlet_rim_gap_m
        <= KA021_MAX_OUTLET_RIM_GAP_M
    )
    emitting = bool(faucet_open_fraction >= KA021_FLOW_VISIBLE_FRACTION)
    hits_cup = bool(emitting and opening_aligned and upright and height_valid)
    impact = outlet.copy()
    impact[2] = (
        cup_rim_z_m
        if hits_cup
        else B320_TABLE_TOP_Z + 0.006
    )
    return {
        "outlet_position": outlet,
        "cup_rim_position": rim_center,
        "impact_position": impact,
        "opening_xy_error_m": opening_xy_error_m,
        "cup_rim_z_m": cup_rim_z_m,
        "outlet_rim_gap_m": outlet_rim_gap_m,
        "cup_tilt_deg": cup_tilt_deg,
        "cup_upright": upright,
        "opening_aligned": opening_aligned,
        "height_valid": height_valid,
        "emitting": emitting,
        "hits_cup": hits_cup,
    }


def _ka021_cup_reset_fits_basin(center_xy: np.ndarray) -> bool:
    """Return whether the complete mug reset stays in the visible basin."""

    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(center)):
        return False
    basin_center = np.asarray(
        KA021_SAFE_BASIN_FLOOR_CENTER_W[:2],
        dtype=np.float64,
    )
    basin_half_extent = 0.5 * np.asarray(
        KA021_SAFE_BASIN_FLOOR_SIZE_M[:2],
        dtype=np.float64,
    )
    reset_envelope_radius = (
        _KA021_CUP_FOOTPRINT_RADIUS_M
        + _KA021_CUP_RESET_EDGE_CLEARANCE_M
    )
    visible_low, visible_high = (
        np.asarray(values, dtype=np.float64)
        for values in KA021_VISIBLE_BASIN_OPENING_BOUNDS_XY
    )
    return bool(
        np.all(center - reset_envelope_radius >= visible_low)
        and np.all(center + reset_envelope_radius <= visible_high)
        and np.all(
            center - reset_envelope_radius
            >= basin_center - basin_half_extent
        )
        and np.all(
            center + reset_envelope_radius
            <= basin_center + basin_half_extent
        )
    )


def _ka021_cup_reset_pose_fits_basin(pose_wxyz: np.ndarray) -> bool:
    """Return whether one complete KA021 reset pose is supported in-basin."""

    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    return bool(
        np.all(np.isfinite(pose))
        and _ka021_cup_reset_fits_basin(pose[:2])
        and abs(float(pose[2]) - _KA021_BASIN_FLOOR_TOP_Z_M) <= 1.0e-6
    )


def _ka021_xy_in_triangle(
    point_xy: np.ndarray,
    triangle_xy: tuple[tuple[float, float], ...],
) -> bool:
    """Return whether one world-XY point lies inside a counter triangle."""

    point = np.asarray(point_xy, dtype=np.float64).reshape(2)
    a, b, c = (np.asarray(vertex, dtype=np.float64) for vertex in triangle_xy)
    edge_a = c - a
    edge_b = b - a
    offset = point - a
    denominator = float(edge_a[0] * edge_b[1] - edge_b[0] * edge_a[1])
    if abs(denominator) <= 1.0e-12:
        return False
    u = float(
        (offset[0] * edge_b[1] - edge_b[0] * offset[1]) / denominator
    )
    v = float(
        (edge_a[0] * offset[1] - offset[0] * edge_a[1]) / denominator
    )
    tolerance = 1.0e-8
    return bool(
        u >= -tolerance
        and v >= -tolerance
        and u + v <= 1.0 + tolerance
    )


def _ka021_marker_fits_left_counter(center_xy: np.ndarray) -> bool:
    """Check all four marker corners against the visible left counter."""

    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    half_extent = 0.5 * np.asarray(KA021_CUP_PARK_SUPPORT_SIZE_M[:2])
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            corner = center + np.asarray(
                [x_sign * half_extent[0], y_sign * half_extent[1]],
                dtype=np.float64,
            )
            if not any(
                _ka021_xy_in_triangle(corner, triangle)
                for triangle in _KA021_LEFT_COUNTER_TOP_TRIANGLES_XY
            ):
                return False
    return True


def _ka021_cup_parking_projection(
    cup_pose_wxyz: np.ndarray,
    target_position_w: np.ndarray | tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Measure the post-fill mug placement on the outside-sink counter pad."""

    cup = np.asarray(cup_pose_wxyz, dtype=np.float64).reshape(7)
    target = np.asarray(
        KA021_CUP_PARK_POSITION_W
        if target_position_w is None
        else target_position_w,
        dtype=np.float64,
    ).reshape(3)
    xy_error = float(np.linalg.norm(cup[:2] - target[:2]))
    z_error = float(abs(cup[2] - target[2]))
    cup_up_axis = _axis_world(cup, KA021_CUP_UP_AXIS_LOCAL)
    tilt_deg = _angle_degrees(
        cup_up_axis,
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    )
    return {
        "cup_park_xy_error_m": xy_error,
        "cup_park_z_error_m": z_error,
        "cup_park_tilt_deg": float(tilt_deg),
        "cup_parked": bool(
            xy_error <= KA021_CUP_PARK_TOLERANCE_M
            and z_error <= KA021_CUP_PARK_Z_TOLERANCE_M
            and tilt_deg <= KA021_CUP_UPRIGHT_TOLERANCE_DEG + 1.0e-6
        ),
    }


def _angle_degrees(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left = np.asarray(lhs, dtype=np.float64)
    right = np.asarray(rhs, dtype=np.float64)
    denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1.0e-12)
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _quaternion_error_degrees(lhs_wxyz: np.ndarray, rhs_wxyz: np.ndarray) -> float:
    lhs = _normalize_quaternion(lhs_wxyz)
    rhs = _normalize_quaternion(rhs_wxyz)
    cosine_half = float(np.clip(abs(np.dot(lhs, rhs)), 0.0, 1.0))
    return float(math.degrees(2.0 * math.acos(cosine_half)))


def _yaw_quaternion(degrees: float) -> np.ndarray:
    half = math.radians(float(degrees)) * 0.5
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _finite_pose(value: Any, fallback: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape == (7,) and np.all(np.isfinite(pose)):
        return pose.copy()
    return np.asarray(fallback, dtype=np.float64).reshape(7).copy()


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """Copy an Isaac tensor/proxy into NumPy across Torch and Warp frontends."""

    if hasattr(value, "torch"):
        value = value.torch
    elif hasattr(value, "warp"):
        value = value.warp
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value).copy()


def _array_like(reference: Any, values: Any, torch_module: Any) -> Any:
    """Create a native write buffer for the active PhysX tensor frontend."""

    if hasattr(reference, "torch"):
        reference = reference.torch
    elif hasattr(reference, "warp"):
        reference = reference.warp
    payload = np.asarray(values, dtype=np.float32).reshape(reference.shape)
    if hasattr(reference, "detach"):
        return torch_module.as_tensor(
            payload,
            dtype=reference.dtype,
            device=reference.device,
        )
    import warp as wp

    return wp.array(payload, dtype=reference.dtype, device=reference.device)


def _indices_like(reference: Any, count: int, torch_module: Any) -> Any:
    """Create indices accepted by either the Torch or Warp PhysX frontend."""

    if hasattr(reference, "torch"):
        reference = reference.torch
    elif hasattr(reference, "warp"):
        reference = reference.warp
    if hasattr(reference, "detach"):
        return torch_module.arange(
            int(count),
            dtype=torch_module.long,
            device=reference.device,
        )
    import warp as wp

    return wp.array(
        np.arange(int(count), dtype=np.int32),
        dtype=wp.int32,
        device=reference.device,
    )


class HandsOnB320BimanualIsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Rigid/articulated B320 tasks with fail-closed physical truth."""

    def _task_specific_isaac_kit_args(self) -> str | None:
        args = super()._task_specific_isaac_kit_args()
        portable_root = os.environ.get("HANDSON_GEN6_KIT_PORTABLE_ROOT", "").strip()
        if self.task_name == GEN6_TASK_NAME and portable_root:
            # Kit requires a separate argument for portable-root, not '='.
            return f"{args or ''} --portable --portable-root {portable_root}"
        return args

    # BC017 is deliberately unavailable through this generic runtime.  Its
    # dedicated subclass opts in only so the shared robot/control plumbing can
    # initialize; all BC017 assets, truth channels, and semantics remain owned
    # by that subclass.
    _bc017_dedicated_runtime = False

    def physical_replay_post_substep_observer_is_safe(
        self,
        callback: Any,
    ) -> bool:
        """Authorize only this class's read-only physical-truth observer."""

        return bool(
            getattr(callback, "__self__", None) is self
            and getattr(callback, "__func__", None)
            is HandsOnB320BimanualIsaacLabEnv._handson_post_physics_substep_callback
        )

    def __init__(self, task_name: str, **kwargs: Any):
        if task_name not in B320_RIGID_TASK_NAMES:
            raise ValueError(f"Rigid B320 runtime does not support {task_name!r}")
        if task_name == "eval_comb06_season_and_stir" and not getattr(self, "_xy6_dedicated_runtime", False):
            raise ValueError("XY6 requires HandsOnXY6IsaacLabEnv for its phase/contact lifecycle")
        if task_name == B320_BC017_TASK_NAME and not bool(
            getattr(self, "_bc017_dedicated_runtime", False)
        ):
            raise RuntimeError(
                "B320-BC017 generic runtime is fail-closed; use "
                "HandsOnB320BC017IsaacLabEnv"
            )
        requested_hand = str(kwargs.get("hand_type", "wuji1")).strip().lower()
        requested_arm = str(kwargs.get("arm_type", "rokae")).strip().lower()
        if requested_hand not in {"wuji1", "wuji", "wuji_hand1", "wuji-hand1"}:
            raise ValueError("B320 benchmark tasks require hand_type='wuji1'")
        if requested_arm != "rokae":
            raise ValueError("B320 benchmark tasks require arm_type='rokae'")

        previous_bc011_pen_variant_index = kwargs.pop(
            "bc011_previous_pen_variant_index",
            None,
        )
        self._bc011_collection_angle_guidance = bool(
            task_name == "t20_pass_and_orient_pen"
            and os.environ.get(
                BC011_COLLECTION_ANGLE_GUIDANCE_ENV,
                "",
            ).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self._b320_contract = B320_TASK_SPECS[task_name]
        self._b320_spawn_result = B320SpawnResult()
        self._bc014_kitchen_report: RealisticKitchenReport | None = None
        self._bc014_reset_poses: dict[str, np.ndarray] = {}
        self._ka012_reset_pose: np.ndarray | None = None
        self._ka012_reset_attempts = 0
        self._ka020_reset_poses: dict[str, np.ndarray] = {}
        self._ka020_reset_attempts = 0
        self._ka020_pump_contact_view: Any | None = None
        self._ka020_pump_contact_force_n = 0.0
        self._ka020_pump_contact_readable = False
        self._ka020_flow_visible = False
        self._ka020_flow_xform_ops: tuple[Any, ...] | None = None
        self._ka020_flow_visual_warning_emitted = False
        self._ka020_last_flow_sample: dict[str, Any] = {}
        self._bc014_goal_centers_xy = {
            role: np.asarray(center, dtype=np.float64)
            for role, center in BC014_GOAL_CENTERS_XY.items()
        }
        self._bc014_reset_attempts = 0
        self._bc014_flow_hit_seconds = {"left": 0.0, "right": 0.0}
        self._bc014_flow_pour_seconds = {"left": 0.0, "right": 0.0}
        self._bc014_flow_overlap_seconds = 0.0
        self._bc014_flow_hit_latched = {"left": False, "right": False}
        self._bc014_flow_overlap_latched = False
        self._bc014_flow_visible = {
            "salt_jar": False,
            "pepper_jar": False,
        }
        self._bc014_flow_xform_ops: dict[str, tuple[Any, ...]] = {}
        self._bc014_flow_visual_warning_emitted = False
        self._bc014_goal_visual_warning_emitted = False
        self._ka021_flow_visible = False
        self._ka021_flow_xform_ops: tuple[Any, ...] | None = None
        self._ka021_flow_visual_warning_emitted = False
        self._ka021_last_flow_sample: dict[str, Any] = {}
        self._ka021_fill_hold_frames = 0
        self._ka021_fill_last_update_step: int | None = None
        self._ka021_water_filled = False
        self._ka021_reset_pose: np.ndarray | None = None
        self._ka021_cup_park_position_w = np.asarray(
            KA021_CUP_PARK_POSITION_W,
            dtype=np.float64,
        )
        self._ka021_reset_attempts = 0
        self._ka021_park_visual_warning_emitted = False
        self._ka005_2_progress_frame: Any | None = None
        self._ka005_2_progress_labels: dict[str, Any] = {}
        self._ka005_2_progress_cards: dict[str, Any] = {}
        self._bc011_previous_pen_variant_index = (
            None
            if previous_bc011_pen_variant_index is None
            else int(previous_bc011_pen_variant_index)
        )
        self._bc011_pen_variant_index: int | None = None
        self._bc011_pen_variant: str | None = None
        self._ka005_2_state = DualKnobSyncState()
        self._ka005_2_reset_sample = DualKnobResetSample(
            initial_rad=(0.0, 0.0),
            target_rad=(math.radians(60.0), math.radians(-60.0)),
        )
        self._ka005_2_contact_views: dict[str, Any] = {}
        self._ka005_2_contact_force_n = {
            key: 0.0 for key in KA005_2_CONTACT_KEYS
        }
        self._ka005_2_contact_readable = {
            key: False for key in KA005_2_CONTACT_KEYS
        }
        self._ka005_2_recording_origin_accepted = False
        self._ka005_2_recording_origin_q_rad = np.zeros(2, dtype=np.float64)
        self._ka005_2_recording_origin_poses: dict[str, np.ndarray] = {}
        self._b320_contact_views: dict[str, Any] = {}
        self._b320_carton_table_contact_view: Any | None = None
        self._b320_pen_table_contact_view: Any | None = None
        self._b320_carton_table_contact_available = False
        self._b320_carton_table_contact_readable = False
        self._b320_carton_table_contact_force = 0.0
        self._b320_pen_table_contact_available = False
        self._b320_pen_table_contact_readable = False
        self._b320_pen_table_contact_force = 0.0
        self._b320_contact_available = {"left": False, "right": False}
        self._b320_contact_force = {"left": 0.0, "right": 0.0}
        self._b320_contact_peak = {"left": 0.0, "right": 0.0}
        self._b320_contact_seen = {"left": False, "right": False}
        self._b320_contact_readable = {"left": False, "right": False}
        self._b320_carton_contact_detail_available = {
            "left": False,
            "right": False,
        }
        self._b320_carton_zone_valid = {"left": False, "right": False}
        self._b320_carton_zone_force = {"left": 0.0, "right": 0.0}
        self._b320_carton_transport_started = False
        self._b320_carton_transport_contact_lost = False
        self._b320_carton_airborne_substeps = 0
        self._b320_carton_airborne_now = False
        self._b320_carton_force_guard_targets: dict[str, np.ndarray] = {}
        self._b320_carton_force_guard_hand_targets: dict[str, np.ndarray] = {}
        self._b320_carton_prelock_lift_clamped = {
            "left": False,
            "right": False,
        }
        self._b320_carton_prelock_lift_clamp_count = {
            "left": 0,
            "right": 0,
        }
        self._b320_carton_force_guard_active = False
        self._b320_carton_force_guard_active_by_side = {
            "left": False,
            "right": False,
        }
        self._b320_carton_force_guard_clamped = {
            "left": False,
            "right": False,
        }
        self._b320_carton_force_guard_predicted_n = {
            "left": 0.0,
            "right": 0.0,
        }
        self._b320_carton_force_guard_inward_speed_m_s = {
            "left": 0.0,
            "right": 0.0,
        }
        self._b320_carton_force_guard_clamp_count = {
            "left": 0,
            "right": 0,
        }
        self._b320_carton_contact_outward = {
            "left": np.zeros(3, dtype=np.float64),
            "right": np.zeros(3, dtype=np.float64),
        }
        self._b320_carton_contact_direction_available = {
            "left": False,
            "right": False,
        }
        self._b320_carton_contact_lever_local = {
            "left": np.zeros(3, dtype=np.float64),
            "right": np.zeros(3, dtype=np.float64),
        }
        self._b320_carton_contact_lever_available = {
            "left": False,
            "right": False,
        }
        self._b320_contact_force_rise_rate_n_s = {
            "left": 0.0,
            "right": 0.0,
        }
        self._b320_carton_force_guard_verified_position = {
            "left": np.zeros(3, dtype=np.float64),
            "right": np.zeros(3, dtype=np.float64),
        }
        self._b320_carton_force_guard_verified_valid = {
            "left": False,
            "right": False,
        }
        self._b320_carton_force_guard_pending_inward_m = {
            "left": 0.0,
            "right": 0.0,
        }
        self._b320_carton_bilateral_pressure_reached = {
            "left": False,
            "right": False,
        }
        self._b320_carton_bilateral_lock_active = False
        self._b320_carton_bilateral_lock_relative_position = np.zeros(
            3,
            dtype=np.float64,
        )
        self._b320_carton_bilateral_lock_latch_step: int | None = None
        self._b320_carton_bilateral_lock_common_delta = np.zeros(
            3,
            dtype=np.float64,
        )
        self._b320_carton_bilateral_lock_discarded_delta = {
            "left": np.zeros(3, dtype=np.float64),
            "right": np.zeros(3, dtype=np.float64),
        }
        self._b320_carton_bilateral_lock_orientation = {
            "left": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "right": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        }
        self._b320_carton_bilateral_lock_hand_targets: dict[
            str, np.ndarray
        ] = {}
        self._b320_carton_bilateral_lock_force_rejected_axes = np.zeros(
            3,
            dtype=bool,
        )
        self._b320_carton_bilateral_lock_force_clamp_count = 0
        self._b320_carton_bilateral_lift_bypass_count = 0
        self._b320_carton_force_hold_active = {
            "left": False,
            "right": False,
        }
        self._b320_carton_force_hold_inward_step_m = {
            "left": 0.0,
            "right": 0.0,
        }
        self._b320_carton_force_hold_correction_count = {
            "left": 0,
            "right": 0,
        }
        self._b320_carton_force_hold_lift_block_count = 0
        self._b320_carton_force_relief_active = {
            "left": False,
            "right": False,
        }
        self._b320_carton_force_relief_count = {
            "left": 0,
            "right": 0,
        }
        self._b320_carton_bilateral_lock_clamp_count = 0
        self._b320_failure_reason: str | None = None
        self._b320_release_frame: dict[str, int | None] = {"left": None, "right": None}
        self._b320_threshold_frame: dict[str, int | None] = {"left": None, "right": None}
        self._b320_reset_poses: dict[str, np.ndarray] = {}
        self._b320_last_metrics: dict[str, Any] = {}
        self._b320_condition_flags: tuple[bool, ...] = (False,) * len(
            self._b320_contract.success_labels
        )
        self._b320_handover_right_first = False
        self._b320_handover_overlap = False
        self._b320_handover_left_only = False
        self._b320_handover_completed = False
        self._bc011_pen_airborne_latched = False
        self._bc011_pen_ground_contact_substeps = 0
        self._bc011_pen_drop_detected = False
        self._b320_left_contact_lost = False
        self._b320_stir_previous_angle: float | None = None
        self._b320_stir_accumulated_rad = 0.0
        self._b320_stir_sectors: set[int] = set()
        self._b320_stir_anchor_pose: np.ndarray | None = None
        self._b320_stir_anchor_frame: int | None = None
        if task_name in _STRICT_30_120_TIMING_TASKS:
            for key, expected in (
                ("control_dt", 1.0 / B320_CONTROL_HZ),
                ("physics_dt", 1.0 / B320_PHYSICS_HZ),
            ):
                if key in kwargs and not math.isclose(
                    float(kwargs[key]),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError(
                        f"{task_name} requires the 30 Hz control / "
                        f"120 Hz physics contract; got {key}={kwargs[key]!r}"
                    )
        kwargs.setdefault("control_dt", 1.0 / B320_CONTROL_HZ)
        kwargs.setdefault("physics_dt", 1.0 / B320_PHYSICS_HZ)
        kwargs.setdefault("hz", B320_CONTROL_HZ)
        kwargs.setdefault("time_limit", float(self._b320_contract.time_limit_seconds))
        kwargs["hand_type"] = "wuji1"
        kwargs["arm_type"] = "rokae"
        # FU001 and KA020 selected-asset contracts require a live calibrated
        # three-camera preview. Opt in before the ABC constructor creates
        # sensors; collection can still force the shared deferred path with
        # HANDSON_ISAACLAB_DEFER_CAMERA_RENDER=1 for diagnostics.
        self._force_live_camera_render = bool(
            task_name
            in {
                "b320_synchronized_double_cabinet_open",
                B320_KA020_TASK_NAME,
            }
        )
        super().__init__(task_name=task_name, **kwargs)
        if task_name == GEN6_TASK_NAME:
            # The completed GEN6 certifier hung in SimulationContext.stop().
            # Use the existing recorder/RTX worker teardown path after output
            # is flushed; this does not alter the live physics lifecycle.
            self._fast_shutdown_on_close = True
        if self._bc011_pen_variant_index is None:
            self._select_bc011_pen_variant_for_scene()
        if task_name in _STRICT_30_120_TIMING_TASKS and (
            not math.isclose(
                float(self._control_dt),
                1.0 / B320_CONTROL_HZ,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                float(self._sim_dt),
                1.0 / B320_PHYSICS_HZ,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or int(self._n_substeps) != 4
        ):
            raise RuntimeError(
                f"{task_name} did not realize its required 30/120 Hz, "
                "four-substep timing contract"
            )
        if self._launch_isaac and task_name == B320_BC014_TASK_NAME:
            # Keep the 168 MB visual closure out of the first PhysX/CUDA
            # context allocation.  This render-only layer is safe to compose
            # after the physical scene and tensor views are fully initialized.
            self._spawn_bc014_kitchen_visual()
        if not self._launch_isaac:
            self._populate_interface_poses()

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        if self._stage is None:
            raise RuntimeError("B320 stage is unavailable before asset spawn")
        self._select_bc011_pen_variant_for_scene()
        self._b320_spawn_result = spawn_b320_task_assets(
            self._stage,
            sim_utils,
            self.task_name,
            bc011_pen_variant_index=self._bc011_pen_variant_index,
        )

    def _spawn_bc014_kitchen_visual(self) -> None:
        if self._stage is None:
            raise RuntimeError(
                "BC014 stage is unavailable before kitchen composition"
            )
        floor = self._find_prim("floor")
        if floor is None or not floor.IsValid():
            raise RuntimeError(
                "BC014 kitchen composition cannot find the canonical floor"
            )
        self._bc014_kitchen_report = compose_realistic_kitchen_visual(
            self._stage,
            root_path=_BC014_KITCHEN_ROOT,
            table_prim_path=B320_TABLE_PATH,
            floor_prim_path=str(floor.GetPath()),
            transform_op_name="bc014Kitchen",
            log_label="BC014",
        )

    def _set_bc014_flow_visual(
        self,
        role: str,
        sample: dict[str, Any] | None,
    ) -> bool:
        """Place or hide one render-only water-flow cylinder."""

        if self.task_name not in {B320_BC014_TASK_NAME, "eval_comb06_season_and_stir"} or self._stage is None:
            return False
        from pxr import Gf, UsdGeom

        path = BC014_FLOW_PROXY_PATHS[role]
        prim = self._stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid():
            if self._launch_isaac and not self._bc014_flow_visual_warning_emitted:
                self._add_task_state_warning(
                    f"BC014 water-flow proxy is missing: {path}"
                )
                self._bc014_flow_visual_warning_emitted = True
            return False

        visible = bool(sample is not None and sample.get("emitting"))
        imageable = UsdGeom.Imageable(prim)
        if not visible:
            if self._bc014_flow_visible.get(role, False):
                imageable.MakeInvisible()
            self._bc014_flow_visible[role] = False
            return True

        assert sample is not None
        start = np.asarray(sample["cap_position"], dtype=np.float64)
        end = np.asarray(sample["impact_position"], dtype=np.float64)
        segment = end - start
        length = float(np.linalg.norm(segment))
        if not math.isfinite(length) or length <= 1.0e-6:
            imageable.MakeInvisible()
            self._bc014_flow_visible[role] = False
            return False
        midpoint = 0.5 * (start + end)
        orientation = _quaternion_from_z_axis(segment)
        ops = self._bc014_flow_xform_ops.get(role)
        if ops is None:
            # The primitive was initially translated below the table. Remove
            # those authored ops once, then retain the three live ops so the
            # 30 Hz visual update only changes values instead of properties.
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            for attribute in tuple(prim.GetAttributes()):
                if attribute.GetName().startswith("xformOp:"):
                    prim.RemoveProperty(attribute.GetName())
            _set_xform(
                prim,
                translation=tuple(float(value) for value in midpoint),
                orientation_wxyz=tuple(float(value) for value in orientation),
                scale=(1.0, 1.0, length),
            )
            ops = tuple(UsdGeom.Xformable(prim).GetOrderedXformOps())
            self._bc014_flow_xform_ops[role] = ops
        else:
            translate_op, orient_op, scale_op = ops
            translate_op.Set(Gf.Vec3d(*map(float, midpoint)))
            quaternion: Any = Gf.Quatd(
                float(orientation[0]),
                Gf.Vec3d(*map(float, orientation[1:])),
            )
            if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                quaternion = Gf.Quatf(
                    float(orientation[0]),
                    Gf.Vec3f(*map(float, orientation[1:])),
                )
            orient_op.Set(quaternion)
            scale_op.Set(Gf.Vec3d(1.0, 1.0, length))
        imageable.MakeVisible()
        self._bc014_flow_visible[role] = True
        return True

    def _set_ka020_flow_visual(self, sample: dict[str, Any] | None) -> bool:
        """Place or hide the render-only soap stream from physical truth."""

        if self.task_name != B320_KA020_TASK_NAME or self._stage is None:
            return False
        from pxr import Gf, UsdGeom

        prim = self._stage.GetPrimAtPath(KA020_FLOW_PROXY_PATH)
        if prim is None or not prim.IsValid():
            if self._launch_isaac and not self._ka020_flow_visual_warning_emitted:
                self._add_task_state_warning(
                    f"KA020 soap-flow proxy is missing: {KA020_FLOW_PROXY_PATH}"
                )
                self._ka020_flow_visual_warning_emitted = True
            return False

        visible = bool(sample is not None and sample.get("emitting"))
        imageable = UsdGeom.Imageable(prim)
        if not visible:
            if self._ka020_flow_visible:
                imageable.MakeInvisible()
            self._ka020_flow_visible = False
            return True

        assert sample is not None
        start = np.asarray(sample["outlet_position"], dtype=np.float64)
        end = np.asarray(sample["impact_position"], dtype=np.float64)
        segment = end - start
        length = float(np.linalg.norm(segment))
        if not math.isfinite(length) or length <= 1.0e-6:
            imageable.MakeInvisible()
            self._ka020_flow_visible = False
            return False
        midpoint = 0.5 * (start + end)
        orientation = _quaternion_from_z_axis(segment)
        ops = self._ka020_flow_xform_ops
        if ops is None:
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            for attribute in tuple(prim.GetAttributes()):
                if attribute.GetName().startswith("xformOp:"):
                    prim.RemoveProperty(attribute.GetName())
            _set_xform(
                prim,
                translation=tuple(float(value) for value in midpoint),
                orientation_wxyz=tuple(float(value) for value in orientation),
                scale=(1.0, 1.0, length),
            )
            self._ka020_flow_xform_ops = tuple(
                UsdGeom.Xformable(prim).GetOrderedXformOps()
            )
        else:
            translate_op, orient_op, scale_op = ops
            translate_op.Set(Gf.Vec3d(*map(float, midpoint)))
            quaternion: Any = Gf.Quatd(
                float(orientation[0]),
                Gf.Vec3d(*map(float, orientation[1:])),
            )
            if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                quaternion = Gf.Quatf(
                    float(orientation[0]),
                    Gf.Vec3f(*map(float, orientation[1:])),
                )
            orient_op.Set(quaternion)
            scale_op.Set(Gf.Vec3d(1.0, 1.0, length))
        imageable.MakeVisible()
        self._ka020_flow_visible = True
        return True

    def _set_ka021_flow_visual(self, sample: dict[str, Any] | None) -> bool:
        """Place or hide the KA021 water stream from its current joint state."""

        if self.task_name not in B320_FAUCET_TASK_NAMES or self._stage is None:
            return False
        from pxr import Gf, UsdGeom

        prim = self._stage.GetPrimAtPath(KA021_FLOW_PROXY_PATH)
        if prim is None or not prim.IsValid():
            if self._launch_isaac and not self._ka021_flow_visual_warning_emitted:
                self._add_task_state_warning(
                    f"KA021 faucet water-flow proxy is missing: {KA021_FLOW_PROXY_PATH}"
                )
                self._ka021_flow_visual_warning_emitted = True
            return False

        visible = bool(sample is not None and sample.get("emitting"))
        imageable = UsdGeom.Imageable(prim)
        if not visible:
            if self._ka021_flow_visible:
                imageable.MakeInvisible()
            self._ka021_flow_visible = False
            return True

        assert sample is not None
        start = np.asarray(sample["outlet_position"], dtype=np.float64)
        end = np.asarray(sample["impact_position"], dtype=np.float64)
        segment = end - start
        length = float(np.linalg.norm(segment))
        if not math.isfinite(length) or length <= 1.0e-6:
            imageable.MakeInvisible()
            self._ka021_flow_visible = False
            return False
        midpoint = 0.5 * (start + end)
        orientation = _quaternion_from_z_axis(segment)
        ops = self._ka021_flow_xform_ops
        if ops is None:
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            for attribute in tuple(prim.GetAttributes()):
                if attribute.GetName().startswith("xformOp:"):
                    prim.RemoveProperty(attribute.GetName())
            _set_xform(
                prim,
                translation=tuple(float(value) for value in midpoint),
                orientation_wxyz=tuple(float(value) for value in orientation),
                scale=(1.0, 1.0, length),
            )
            self._ka021_flow_xform_ops = tuple(
                UsdGeom.Xformable(prim).GetOrderedXformOps()
            )
        else:
            translate_op, orient_op, scale_op = ops
            translate_op.Set(Gf.Vec3d(*map(float, midpoint)))
            quaternion: Any = Gf.Quatd(
                float(orientation[0]),
                Gf.Vec3d(*map(float, orientation[1:])),
            )
            if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                quaternion = Gf.Quatf(
                    float(orientation[0]),
                    Gf.Vec3f(*map(float, orientation[1:])),
                )
            orient_op.Set(quaternion)
            scale_op.Set(Gf.Vec3d(1.0, 1.0, length))
        imageable.MakeVisible()
        self._ka021_flow_visible = True
        return True

    def _set_bc014_goal_marker_visuals(self) -> bool:
        """Move both highlighted marker fills and outlines for this reset."""

        if self.task_name != B320_BC014_TASK_NAME or self._stage is None:
            return False
        from pxr import Gf, UsdGeom

        moved = 0
        border_width = min(
            0.008,
            0.16 * min(BC014_GOAL_ZONE_SIZE_M[0:2]),
        )
        border_height = max(0.004, BC014_GOAL_ZONE_SIZE_M[2])
        for role, path in BC014_GOAL_MARKER_PATHS.items():
            center = self._bc014_goal_centers_xy[role]
            marker_z = B320_TABLE_TOP_Z + 0.003
            half_x, half_y = (
                0.5 * BC014_GOAL_ZONE_SIZE_M[0],
                0.5 * BC014_GOAL_ZONE_SIZE_M[1],
            )
            border_z = marker_z + 0.5 * (
                BC014_GOAL_ZONE_SIZE_M[2] + border_height
            )
            translations = {
                path: (float(center[0]), float(center[1]), marker_z),
                f"{path}_highlight_front": (
                    float(center[0]),
                    float(center[1] + half_y - 0.5 * border_width),
                    border_z,
                ),
                f"{path}_highlight_rear": (
                    float(center[0]),
                    float(center[1] - half_y + 0.5 * border_width),
                    border_z,
                ),
                f"{path}_highlight_left": (
                    float(center[0] - half_x + 0.5 * border_width),
                    float(center[1]),
                    border_z,
                ),
                f"{path}_highlight_right": (
                    float(center[0] + half_x - 0.5 * border_width),
                    float(center[1]),
                    border_z,
                ),
            }
            for prim_path, translation in translations.items():
                prim = self._stage.GetPrimAtPath(prim_path)
                if prim is None or not prim.IsValid():
                    if (
                        self._launch_isaac
                        and not self._bc014_goal_visual_warning_emitted
                    ):
                        self._add_task_state_warning(
                            "BC014 highlighted goal marker is missing: "
                            f"{prim_path}"
                        )
                        self._bc014_goal_visual_warning_emitted = True
                    continue
                xform = UsdGeom.Xformable(prim)
                translate_op = next(
                    (
                        op
                        for op in xform.GetOrderedXformOps()
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                    ),
                    None,
                )
                if translate_op is None:
                    translate_op = xform.AddTranslateOp()
                value: Any = Gf.Vec3d(*translation)
                if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                    value = Gf.Vec3f(*translation)
                translate_op.Set(value)
                moved += 1
            marker_prim = self._stage.GetPrimAtPath(path)
            if marker_prim is not None and marker_prim.IsValid():
                marker_prim.SetCustomDataByKey(
                    "handson:goalCenterXY",
                    tuple(float(component) for component in center),
                )
        return moved == 5 * len(BC014_GOAL_MARKER_PATHS)

    def _set_ka021_park_target_visuals(self) -> bool:
        """Move the KA021 counter pad, fill, and outline to the reset target."""

        if self.task_name not in B320_FAUCET_TASK_NAMES or self._stage is None:
            return False
        from pxr import Gf, UsdGeom

        target = np.asarray(
            self._ka021_cup_park_position_w,
            dtype=np.float64,
        ).reshape(3)
        marker_z = float(target[2] + 0.004)
        size = np.asarray(KA021_CUP_PARK_SUPPORT_SIZE_M, dtype=np.float64)
        marker_thickness = 0.006
        border_width = min(0.008, 0.16 * min(float(size[0]), float(size[1])))
        border_height = max(0.004, marker_thickness)
        border_z = marker_z + 0.5 * (marker_thickness + border_height)
        translations = {
            KA021_CUP_PARK_SUPPORT_PATH: (
                float(target[0]),
                float(target[1]),
                float(target[2] - 0.010),
            ),
            KA021_CUP_PARK_MARKER_PATH: (
                float(target[0]),
                float(target[1]),
                marker_z,
            ),
            f"{KA021_CUP_PARK_MARKER_PATH}_highlight_front": (
                float(target[0]),
                float(target[1] + 0.5 * (size[1] - border_width)),
                border_z,
            ),
            f"{KA021_CUP_PARK_MARKER_PATH}_highlight_rear": (
                float(target[0]),
                float(target[1] - 0.5 * (size[1] - border_width)),
                border_z,
            ),
            f"{KA021_CUP_PARK_MARKER_PATH}_highlight_left": (
                float(target[0] - 0.5 * (size[0] - border_width)),
                float(target[1]),
                border_z,
            ),
            f"{KA021_CUP_PARK_MARKER_PATH}_highlight_right": (
                float(target[0] + 0.5 * (size[0] - border_width)),
                float(target[1]),
                border_z,
            ),
        }
        moved = 0
        for prim_path, translation in translations.items():
            prim = self._stage.GetPrimAtPath(prim_path)
            if prim is None or not prim.IsValid():
                continue
            xform = UsdGeom.Xformable(prim)
            translate_op = next(
                (
                    op
                    for op in xform.GetOrderedXformOps()
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                ),
                None,
            )
            if translate_op is None:
                translate_op = xform.AddTranslateOp()
            value: Any = Gf.Vec3d(*translation)
            if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                value = Gf.Vec3f(*translation)
            translate_op.Set(value)
            moved += 1

        marker_prim = self._stage.GetPrimAtPath(KA021_CUP_PARK_MARKER_PATH)
        if marker_prim is not None and marker_prim.IsValid():
            marker_prim.SetCustomDataByKey(
                "handson:goalCenterXY",
                (float(target[0]), float(target[1])),
            )
            marker_prim.SetCustomDataByKey(
                "handson:cupParkPositionW",
                tuple(float(component) for component in target),
            )
        return moved == len(translations)

    def _update_bc014_flow_visuals(
        self,
        samples: dict[str, dict[str, Any]],
    ) -> None:
        if self.task_name not in {B320_BC014_TASK_NAME, "eval_comb06_season_and_stir"}:
            return
        try:
            for role in ("salt_jar", "pepper_jar"):
                self._set_bc014_flow_visual(role, samples.get(role))
        except Exception as exc:
            if not self._bc014_flow_visual_warning_emitted:
                self._add_task_state_warning(
                    "Could not update BC014 water-flow proxy: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._bc014_flow_visual_warning_emitted = True

    def _in_process_scene_rebuild_kwargs(self, seed: int) -> dict[str, Any]:
        kwargs = super()._in_process_scene_rebuild_kwargs(seed)
        if self.task_name == "t20_pass_and_orient_pen":
            kwargs["bc011_previous_pen_variant_index"] = (
                self._bc011_pen_variant_index
            )
        return kwargs

    def _configure_randomizable_table_physics(self) -> tuple[int, bool]:
        """B320 owns a fixed worktable and intentionally has no height jitter."""

        return 0, False

    def _task_articulation_root_is_writable(self, path: str) -> bool:
        """Fixed fixture roots stay world-anchored; their DOFs remain writable."""

        return not str(path).endswith("/world_fixed_joint")

    def _task_articulation_root_velocity_is_writable(self, path: str) -> bool:
        if (
            self.task_name == "eval_gen05_set_mixer_knobs"
            and str(path) == f"{B320_ASSET_ROOT_PATH}/panel_base/world_fixed_joint"
        ):
            # The fixed root pose stays protected. Its readable velocity is
            # still an active truth field and participates in the standard
            # velocity-only recording boundary, as for the fixed D02 fixture.
            return True
        return super()._task_articulation_root_velocity_is_writable(path)

    def _find_rigid_body_prim(self, name: str):
        if self._stage is not None:
            path = self._b320_spawn_result.body_paths.get(name)
            if path:
                prim = self._stage.GetPrimAtPath(path)
                if prim is not None and prim.IsValid():
                    return prim
        return super()._find_rigid_body_prim(name)

    def _task_extra_pose_names(self) -> set[str]:
        return set(super()._task_extra_pose_names()) | set(
            self._b320_spawn_result.body_paths
        )

    def _configure_task_collision_safety(self, sim_utils: Any) -> tuple[int, int]:
        bodies, colliders = super()._configure_task_collision_safety(sim_utils)
        if self._stage is None:
            return bodies, colliders
        from pxr import PhysxSchema, Usd, UsdPhysics

        # The source meshes are contact-critical and can arrive as instanced
        # USD payloads. Uninstance the two affected benchmark objects before
        # authoring body/collider attributes so PhysX does not share a stale
        # cache between reset environments.
        stability_names = {
            "t10_lift_carton": ("carton",),
            "t20_pass_and_orient_pen": ("pen",),
            B320_KA021_TASK_NAME: ("ceramic_mug",),
            XY4_TASK_NAME: ("ceramic_mug",),
        }.get(self.task_name, ())
        stability_paths = [
            self._b320_spawn_result.body_paths[name]
            for name in stability_names
            if name in self._b320_spawn_result.body_paths
        ]
        scene_hand_root_paths = (
            tuple(
                f"/World/HandsOnRobots/{side}Robot"
                for side in ("Right", "Left")
            )
            if self.task_name in B320_FAUCET_TASK_NAMES
            else ()
        )
        for path in (*stability_paths, *scene_hand_root_paths):
            sim_utils.make_uninstanceable(path, stage=self._stage)
        if stability_paths or scene_hand_root_paths:
            sim_utils.update_stage()

        configured = 0
        for path in dict.fromkeys(self._b320_spawn_result.body_paths.values()):
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            report = (
                PhysxSchema.PhysxContactReportAPI.Get(self._stage, prim.GetPath())
                if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                else PhysxSchema.PhysxContactReportAPI.Apply(prim)
            )
            report.CreateThresholdAttr().Set(0.0)
            configured += 1

        tuned_bodies = 0
        tuned_colliders = 0
        for name in stability_names:
            path = self._b320_spawn_result.body_paths.get(name)
            if not path:
                continue
            prim = self._stage.GetPrimAtPath(path)
            if prim is None or not prim.IsValid():
                continue
            rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not rigid_api:
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            object_spec = next(
                (
                    candidate
                    for candidate in self._b320_contract.object_specs
                    if candidate.name == name
                ),
                None,
            )
            # BC004 has one authoritative object specification shared by the
            # initial spawn and this post-compose verification pass. KA021's
            # mug must likewise retain its high damping while CCD is enabled.
            # Preserve the legacy pen tuning for BC011.
            use_object_spec = self.task_name in {
                "t10_lift_carton",
                B320_KA021_TASK_NAME,
                XY4_TASK_NAME,
            }
            linear_damping = (
                float(object_spec.linear_damping)
                if use_object_spec and object_spec is not None
                else 0.40
            )
            angular_damping = (
                float(object_spec.angular_damping)
                if use_object_spec and object_spec is not None
                else 2.00
            )
            sleep_threshold = (
                float(object_spec.sleep_threshold)
                if use_object_spec and object_spec is not None
                else 0.010
            )
            contact_offset = (
                float(object_spec.contact_offset)
                if use_object_spec and object_spec is not None
                else 0.001
            )
            for getter_name, creator_name, value in (
                ("GetEnableCCDAttr", "CreateEnableCCDAttr", True),
                (
                    "GetSolverPositionIterationCountAttr",
                    "CreateSolverPositionIterationCountAttr",
                    32,
                ),
                (
                    "GetSolverVelocityIterationCountAttr",
                    "CreateSolverVelocityIterationCountAttr",
                    4,
                ),
                (
                    "GetMaxDepenetrationVelocityAttr",
                    "CreateMaxDepenetrationVelocityAttr",
                    0.8,
                ),
                ("GetLinearDampingAttr", "CreateLinearDampingAttr", linear_damping),
                ("GetAngularDampingAttr", "CreateAngularDampingAttr", angular_damping),
                ("GetSleepThresholdAttr", "CreateSleepThresholdAttr", sleep_threshold),
            ):
                attr = getattr(rigid_api, getter_name)()
                if not attr or not attr.IsValid():
                    attr = getattr(rigid_api, creator_name)()
                attr.Set(value)
            tuned_bodies += 1
            for collider in Usd.PrimRange(prim):
                if not collider.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                collision_api = PhysxSchema.PhysxCollisionAPI(collider)
                if not collision_api:
                    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collider)
                contact_attr = collision_api.GetContactOffsetAttr()
                if not contact_attr or not contact_attr.IsValid():
                    contact_attr = collision_api.CreateContactOffsetAttr()
                contact_attr.Set(contact_offset)
                rest_attr = collision_api.GetRestOffsetAttr()
                if not rest_attr or not rest_attr.IsValid():
                    rest_attr = collision_api.CreateRestOffsetAttr()
                rest_attr.Set(0.0)
                tuned_colliders += 1
        if stability_names and tuned_bodies < len(stability_names):
            self._add_task_state_warning(
                "B320 reset stability tuning is incomplete: "
                f"bodies={tuned_bodies}/{len(stability_names)}"
            )

        hand_bodies = 0
        hand_colliders = 0
        for root_path in scene_hand_root_paths:
            root = self._stage.GetPrimAtPath(root_path)
            if not root.IsValid():
                continue
            for prim in Usd.PrimRange(root):
                normalized_path = str(prim.GetPath()).lower()
                if (
                    "palm" not in normalized_path
                    and "finger" not in normalized_path
                ):
                    continue
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
                    if not rigid_api:
                        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    speculative = rigid_api.GetEnableSpeculativeCCDAttr()
                    if not speculative or not speculative.IsValid():
                        speculative = rigid_api.CreateEnableSpeculativeCCDAttr()
                    speculative.Set(True)
                    hand_bodies += 1
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                shell = PhysxSchema.PhysxCollisionAPI(prim)
                if not shell:
                    shell = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                shell.CreateContactOffsetAttr().Set(
                    KA021_HAND_CONTACT_OFFSET
                )
                shell.CreateRestOffsetAttr().Set(0.0)
                hand_colliders += 1

        if self.task_name in B320_FAUCET_TASK_NAMES and (
            tuned_bodies != 1
            or tuned_colliders == 0
            or hand_bodies == 0
            or hand_colliders == 0
        ):
            raise RuntimeError(
                "KA021 shared-scene collision safety is incomplete: "
                f"mug_bodies={tuned_bodies}/1, "
                f"mug_colliders={tuned_colliders}, "
                f"hand_bodies={hand_bodies}, hand_colliders={hand_colliders}"
            )
        return (
            bodies + configured + hand_bodies,
            colliders + tuned_colliders + hand_colliders,
        )

    def _hand_body_paths(self, side: str) -> list[str]:
        if self._stage is None:
            return []
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        if root is None or not root.IsValid():
            return []
        prefixes = (f"{side}_finger", f"{side}_palm")
        return [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and prim.GetName().startswith(prefixes)
        ]

    def _role_body_path(self, role: str) -> str | None:
        return self._b320_spawn_result.body_paths.get(role)

    def _init_task_specific_runtime(self) -> None:
        self._b320_contact_views = {}
        self._ka005_2_contact_views = {}
        self._ka020_pump_contact_view = None
        self._b320_carton_table_contact_view = None
        self._b320_pen_table_contact_view = None
        if not self._robots:
            return
        physics_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_view is None:
            self._add_task_state_warning("B320 PhysX contact view is unavailable")
            return
        for side, role in (
            ("left", self._b320_contract.left_role),
            ("right", self._b320_contract.right_role),
        ):
            sensor = self._role_body_path(role)
            filters = self._hand_body_paths(side)
            if not sensor or not filters:
                self._add_task_state_warning(
                    f"B320 {side}/{role} contact binding is incomplete"
                )
                continue
            try:
                view = physics_view.create_rigid_contact_view(
                    sensor,
                    filter_patterns=filters,
                    max_contact_data_count=max(96, len(filters) * 12),
                )
                if (
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                ):
                    self._b320_contact_views[side] = view
            except Exception as exc:
                self._add_task_state_warning(
                    f"Could not initialize B320 {side}/{role} contact view: {exc}"
                )
        if self.task_name == B320_KA020_TASK_NAME:
            sensor = self._role_body_path("soap_dispense_pad")
            filters = self._hand_body_paths("right")
            if sensor and filters:
                try:
                    view = physics_view.create_rigid_contact_view(
                        sensor,
                        filter_patterns=filters,
                        max_contact_data_count=max(96, len(filters) * 12),
                    )
                    if (
                        view is not None
                        and int(view.sensor_count) > 0
                        and int(view.filter_count) > 0
                    ):
                        self._ka020_pump_contact_view = view
                except Exception as exc:
                    self._add_task_state_warning(
                        "Could not initialize B320-KA020 right-hand/pump-head "
                        f"contact view: {exc}"
                    )
            if self._ka020_pump_contact_view is None:
                self._add_task_state_warning(
                    "B320-KA020 pump-head contact truth is unavailable; "
                    "success fails closed"
                )
        if self.task_name in B320_DUAL_KNOB_TASK_NAMES:
            if "left" in self._b320_contact_views:
                self._ka005_2_contact_views["LL"] = self._b320_contact_views[
                    "left"
                ]
            if "right" in self._b320_contact_views:
                self._ka005_2_contact_views["RR"] = self._b320_contact_views[
                    "right"
                ]
            for key, role, hand_side in (
                ("LR", "knob_right", "left"),
                ("RL", "knob_left", "right"),
            ):
                sensor = self._role_body_path(role)
                filters = self._hand_body_paths(hand_side)
                if not sensor or not filters:
                    continue
                try:
                    view = physics_view.create_rigid_contact_view(
                        sensor,
                        filter_patterns=filters,
                        max_contact_data_count=max(96, len(filters) * 12),
                    )
                    if (
                        view is not None
                        and int(view.sensor_count) > 0
                        and int(view.filter_count) > 0
                    ):
                        self._ka005_2_contact_views[key] = view
                except Exception as exc:
                    self._add_task_state_warning(
                        f"Could not initialize B320-KA005-2 {key} contact view: {exc}"
                    )
            missing_pairs = sorted(
                set(KA005_2_CONTACT_KEYS) - set(self._ka005_2_contact_views)
            )
            if missing_pairs:
                self._add_task_state_warning(
                    "B320-KA005-2 four-pair contact truth is incomplete; "
                    "success fails closed: " + ",".join(missing_pairs)
                )
        if self.task_name == "t10_lift_carton":
            carton_path = self._role_body_path("carton")
            if carton_path:
                try:
                    table_view = physics_view.create_rigid_contact_view(
                        carton_path,
                        filter_patterns=[B320_TABLE_PATH],
                        max_contact_data_count=64,
                    )
                    if (
                        table_view is not None
                        and int(table_view.sensor_count) > 0
                        and int(table_view.filter_count) > 0
                    ):
                        self._b320_carton_table_contact_view = table_view
                except Exception as exc:
                    self._add_task_state_warning(
                        "Could not initialize BC004 carton-table contact view: "
                        f"{exc}"
                    )
            if self._b320_carton_table_contact_view is None:
                self._add_task_state_warning(
                    "BC004 carton-table contact reporting is unavailable; "
                    "airborne success fails closed"
                )
        elif self.task_name == "t20_pass_and_orient_pen":
            pen_path = self._role_body_path("pen")
            if pen_path:
                try:
                    table_view = physics_view.create_rigid_contact_view(
                        pen_path,
                        filter_patterns=[B320_TABLE_PATH],
                        max_contact_data_count=64,
                    )
                    if (
                        table_view is not None
                        and int(table_view.sensor_count) > 0
                        and int(table_view.filter_count) > 0
                    ):
                        self._b320_pen_table_contact_view = table_view
                except Exception as exc:
                    self._add_task_state_warning(
                        "Could not initialize BC011 pen-table contact view: "
                        f"{exc}"
                    )
            if self._b320_pen_table_contact_view is None:
                self._add_task_state_warning(
                    "BC011 pen-table contact reporting is unavailable; "
                    "left-hand-only success fails closed"
                )
        missing = sorted({"left", "right"} - set(self._b320_contact_views))
        if missing:
            self._add_task_state_warning(
                "B320 contact reporting is incomplete; success fails closed: "
                + ",".join(missing)
            )

    def _fallback_b320_pose(self, name: str) -> np.ndarray:
        for object_spec in self._b320_contract.object_specs:
            if object_spec.name == name:
                return np.asarray(
                    [*object_spec.home_translation, *object_spec.home_orientation_wxyz],
                    dtype=np.float64,
                )
        if self.task_name == GEN6_TASK_NAME:
            from .gen6_audio_mixer_scene import asset_geometry

            geometry = asset_geometry()
            root_home = geometry["root_home_m"]

            if name == "panel_base":
                return np.asarray((*root_home, 1.0, 0.0, 0.0, 0.0))
            if name in {"knob_left", "knob_right"}:
                center = geometry["knob_centers_m"][name.removeprefix("knob_")]
                position = tuple(root_home[i] + center[i] for i in range(3))
                return np.asarray((*position, *geometry["knob_orientation_wxyz"]))
        fixture_positions = {
            "fixture_base": self._b320_contract.goal_pose_wxyz,
            # Local procedural BC017 test fixture pose.  The formal external
            # drawer bundle uses the same logical roles once integrated.
            "left_drawer": (-0.31, 0.15, B320_TABLE_TOP_Z + 0.085, 1.0, 0.0, 0.0, 0.0),
            "right_drawer": (-0.31, -0.15, B320_TABLE_TOP_Z + 0.085, 1.0, 0.0, 0.0, 0.0),
            "cabinet_door_left": (-0.284, 0.368, 1.577, 0.5, 0.5, -0.5, -0.5),
            "cabinet_door_right": (-0.283, -0.375, 1.577, 0.5, 0.5, -0.5, -0.5),
            "faucet_base": (-0.06, 0.0, B320_TABLE_TOP_Z, 0.0, 0.0, 0.0, 1.0),
            "faucet_handle": (
                -0.06,
                -0.085,
                B320_TABLE_TOP_Z + 0.140,
                1.0,
                0.0,
                0.0,
                0.0,
            ),
            "soap_dispense_pad": (
                -0.21 + KA020_PUMP_HEAD_CENTER_LOCAL_M[0],
                -0.18 + KA020_PUMP_HEAD_CENTER_LOCAL_M[1],
                B320_TASK_SPECS[B320_KA020_TASK_NAME].object_specs[1].home_translation[2]
                + KA020_PUMP_HEAD_CENTER_LOCAL_M[2],
                1.0,
                0.0,
                0.0,
                0.0,
            ),
            "panel_base": (-0.14, 0.0, 0.915, 1.0, 0.0, 0.0, 0.0),
            "knob_left": (-0.14, 0.135, 0.940, 1.0, 0.0, 0.0, 0.0),
            "knob_right": (-0.14, -0.135, 0.940, 1.0, 0.0, 0.0, 0.0),
            "table": (-0.15, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        }
        if self.task_name == "b320_synchronized_double_cabinet_open":
            # Keep the interface-only path on the same authored frame as the
            # canonical Poly Haven spawn.  These values are also the fallback
            # used when a recorder asks for role poses before a native tensor
            # readback is available; retaining the retired 44781 coordinates
            # here would make the transport contract disagree with the live
            # scene by more than 0.3 m.
            fixture_positions["fixture_base"] = (
                *FU001_ROOT_HOME,
                1.0,
                0.0,
                0.0,
                0.0,
            )
            for logical, source_pivot in FU001_DOOR_SOURCE_PIVOTS.items():
                runtime_pivot = _fu001_runtime_point(
                    source_pivot,
                    scale=FU001_RUNTIME_SCALE,
                )
                world_pivot = tuple(
                    float(FU001_ROOT_HOME[index] + runtime_pivot[index])
                    for index in range(3)
                )
                fixture_positions[logical] = (
                    *world_pivot,
                    *FU001_RUNTIME_ORIENTATION_WXYZ,
                )
        if self.task_name in B320_FAUCET_TASK_NAMES:
            fixture_positions.update(
                {
                    "faucet_base": (
                        *KA021_FAUCET_BASE_POSITION_W,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ),
                    "faucet_handle": (
                        *KA021_FAUCET_HANDLE_PIVOT_W,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ),
                }
            )
        return np.asarray(
            fixture_positions.get(name, self._b320_contract.goal_pose_wxyz),
            dtype=np.float64,
        )

    def _populate_interface_poses(self) -> None:
        names = set(self._spec.target_objects) | {self._spec.receptacle, "table"}
        names.update(self._b320_spawn_result.body_paths)
        for name in names:
            self._poses.setdefault(name, self._fallback_b320_pose(name))

    def _set_ka005_2_target_markers(self) -> None:
        if self.task_name == GEN6_TASK_NAME:
            # GEN6 displays current/target angles in the inherited HUD. Do not
            # generate new visible marker meshes on the downloaded mixer.
            return
        if self.task_name not in B320_DUAL_KNOB_TASK_NAMES or self._stage is None:
            return
        from pxr import Gf, UsdGeom

        centers = {"left": (0.0, 0.135), "right": (0.0, -0.135)}
        for index, side in enumerate(("left", "right")):
            path = B320_KA005_2_TARGET_MARKER_PATHS[side]
            prim = self._stage.GetPrimAtPath(path)
            if prim is None or not prim.IsValid():
                if self._launch_isaac:
                    raise RuntimeError(
                        f"B320-KA005-2 target marker is missing: {path}"
                    )
                continue
            angle = float(self._ka005_2_reset_sample.target_rad[index])
            center_x, center_y = centers[side]
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(
                Gf.Vec3d(
                    center_x,
                    center_y,
                    0.025,
                )
            )
            xform.AddOrientOp().Set(
                Gf.Quatf(
                    math.cos(angle * 0.5),
                    Gf.Vec3f(0.0, 0.0, math.sin(angle * 0.5)),
                )
            )
            prim.SetCustomDataByKey("handson:targetAngleRad", angle)

    def _select_bc011_pen_variant_for_scene(self) -> None:
        """Choose the one BC011 asset authored into this fresh PhysX scene."""

        if self.task_name != "t20_pass_and_orient_pen":
            return
        count = len(BC011_PEN_VARIANTS)
        if count < 2:
            raise RuntimeError("BC011 requires at least two pen variants")
        previous = self._bc011_previous_pen_variant_index
        if not self.randomize:
            index = 0
        elif previous is None:
            index = int(self._random.randint(count))
        else:
            candidates = [candidate for candidate in range(count) if candidate != previous]
            index = candidates[int(self._random.randint(len(candidates)))]

        from .b_eval_setting import setting_for
        selection = setting_for(self.task_name)
        if selection:
            asset = {"black_pen_negative_x": "cc_by_black_pen_high_quality/pen.usda", "the_pen_positive_x": "cc_by_the_pen/pen.usda"}[selection["setting_id"]]
            index = list(BC011_PEN_VARIANTS).index(asset)
        selected = BC011_PEN_VARIANTS[index]
        self._bc011_pen_variant_index = index
        self._bc011_pen_variant = selected
        print(
            "[HandsOn/IsaacLab] BC011 scene pen variant: "
            f"index={index}, asset={selected}, previous={previous}",
            flush=True,
        )

    def _sample_bc014_reset_poses(self) -> dict[str, np.ndarray]:
        """Sample a seeded, side-assigned BC014 jar-and-goal layout.

        The generic B320 pose jitter is applied only after the Isaac task body
        views exist, which made interface resets and recording-origin metadata
        appear deterministic. BC014 owns its complete joint sample here so
        live Isaac, goal visuals, headless checks, and replay metadata all use
        exactly the same jar and target positions.
        """

        if self.task_name != B320_BC014_TASK_NAME:
            return {}
        authored = {
            spec.name: np.asarray(
                [*spec.home_translation, *spec.home_orientation_wxyz],
                dtype=np.float64,
            )
            for spec in self._b320_contract.object_specs
        }
        if not self.randomize:
            self._bc014_reset_attempts = 0
            self._bc014_goal_centers_xy = {
                role: np.asarray(center, dtype=np.float64)
                for role, center in BC014_GOAL_CENTERS_XY.items()
            }
            return {name: pose.copy() for name, pose in authored.items()}

        table_min = np.asarray([-0.57, -0.64], dtype=np.float64) + 0.020
        table_max = np.asarray([0.27, 0.64], dtype=np.float64) - 0.020
        for attempt in range(1, _BC014_RESET_MAX_ATTEMPTS + 1):
            sampled: dict[str, np.ndarray] = {}
            goals: dict[str, np.ndarray] = {}
            for name in ("salt_jar", "pepper_jar"):
                pose = authored[name].copy()
                low, high = (
                    np.asarray(values, dtype=np.float64)
                    for values in _BC014_COMFORT_CENTER_BOUNDS_XY[name]
                )
                pose[:2] = self._random.uniform(low, high)
                goals[name] = np.asarray(
                    self._random.uniform(low, high),
                    dtype=np.float64,
                )
                pose[3:7] = _quat_multiply(
                    _yaw_quaternion(
                        float(
                            self._random.uniform(
                                -_BC014_RESET_YAW_DEG,
                                _BC014_RESET_YAW_DEG,
                            )
                        )
                    ),
                    authored[name][3:7],
                )
                sampled[name] = pose

            valid = True
            for name, pose in sampled.items():
                if not np.all(np.isfinite(pose)):
                    valid = False
                    break
                footprint = np.asarray(
                    [_BC014_JAR_HALF_EXTENT_M, _BC014_JAR_HALF_EXTENT_M],
                    dtype=np.float64,
                )
                if np.any(pose[:2] - footprint < table_min) or np.any(
                    pose[:2] + footprint > table_max
                ):
                    valid = False
                    break
                if abs(
                    pose[2]
                    - (B320_TABLE_TOP_Z + BC014_JAR_TABLE_CLEARANCE_M)
                ) > 1.0e-6:
                    valid = False
                    break
                goal = goals[name]
                goal_half_extent = 0.5 * np.asarray(
                    BC014_GOAL_ZONE_SIZE_M[:2],
                    dtype=np.float64,
                )
                if np.any(goal - goal_half_extent < table_min) or np.any(
                    goal + goal_half_extent > table_max
                ):
                    valid = False
                    break
                if (
                    float(np.linalg.norm(pose[:2] - goal))
                    < _BC014_MIN_JAR_GOAL_SEPARATION_M
                ):
                    valid = False
                    break
                base = np.asarray(
                    _BC014_CANONICAL_ARM_BASE_POSITIONS[name],
                    dtype=np.float64,
                )
                for point_xy in (pose[:2], goal):
                    reach_point = np.asarray(
                        [*point_xy, _BC014_COMFORT_REACH_Z_M],
                        dtype=np.float64,
                    )
                    if (
                        float(np.linalg.norm(reach_point - base))
                        > _BC014_COMFORT_MAX_RADIUS_M
                    ):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                separation = float(
                    np.linalg.norm(
                        sampled["salt_jar"][:2]
                        - sampled["pepper_jar"][:2]
                    )
                )
                valid = separation >= _BC014_MIN_START_SEPARATION_M
            if valid:
                self._bc014_reset_attempts = attempt
                self._bc014_goal_centers_xy = {
                    role: center.copy() for role, center in goals.items()
                }
                return sampled

        raise RuntimeError(
            "BC014 could not sample a valid randomized reset after "
            f"{_BC014_RESET_MAX_ATTEMPTS} attempts"
        )

    @staticmethod
    def _ka012_pot_reach_metrics(pose_wxyz: np.ndarray) -> dict[str, Any]:
        """Measure the complete KA012 bimanual reach contract for one pot pose."""

        pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
        rotation = _rotation_matrix(pose[3:7])
        handle = pose[:3] + rotation @ np.asarray(
            _KA012_LEFT_HANDLE_LOCAL_OFFSET_M,
            dtype=np.float64,
        )
        stir_center = pose[:3] + rotation @ np.asarray(
            _KA012_STIR_CENTER_LOCAL_OFFSET_M,
            dtype=np.float64,
        )
        left_base = np.asarray(
            _KA012_CANONICAL_ARM_BASE_POSITIONS["left"],
            dtype=np.float64,
        )
        right_base = np.asarray(
            _KA012_CANONICAL_ARM_BASE_POSITIONS["right"],
            dtype=np.float64,
        )
        right_horizontal_center_distance = float(
            np.linalg.norm(stir_center[:2] - right_base[:2])
        )
        right_max_stir_reach = math.hypot(
            right_horizontal_center_distance + _KA012_STIR_PATH_RADIUS_M,
            float(stir_center[2] - right_base[2]),
        )
        return {
            "left_handle_position_m": handle.tolist(),
            "left_handle_reach_m": float(np.linalg.norm(handle - left_base)),
            "right_stir_center_position_m": stir_center.tolist(),
            "right_max_stir_reach_m": right_max_stir_reach,
        }

    @classmethod
    def _ka012_pot_reset_is_reachable(cls, pose_wxyz: np.ndarray) -> bool:
        pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(pose)):
            return False
        if not math.isclose(
            float(np.linalg.norm(pose[3:7])),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            return False
        if not math.isclose(
            float(pose[2]),
            B320_TABLE_TOP_Z + 0.001,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            return False

        table_min = np.asarray([-0.57, -0.64], dtype=np.float64) + 0.020
        table_max = np.asarray([0.27, 0.64], dtype=np.float64) - 0.020
        footprint = np.full(2, _KA012_POT_FOOTPRINT_RADIUS_M, dtype=np.float64)
        if np.any(pose[:2] - footprint < table_min) or np.any(
            pose[:2] + footprint > table_max
        ):
            return False

        metrics = cls._ka012_pot_reach_metrics(pose)
        return bool(
            metrics["left_handle_reach_m"] <= _KA012_COMFORT_MAX_RADIUS_M
            and metrics["right_max_stir_reach_m"]
            <= _KA012_COMFORT_MAX_RADIUS_M
        )

    def _sample_ka012_reset_pose(self) -> np.ndarray | None:
        """Sample one seeded pot pose whose full two-hand action is reachable."""

        if self.task_name != _KA012_TASK_NAME:
            return None
        authored = self._fallback_b320_pose("dutch_oven")
        if not self.randomize:
            self._ka012_reset_attempts = 0
            if not self._ka012_pot_reset_is_reachable(authored):
                raise RuntimeError("KA012 authored pot reset is outside arm reach")
            return authored

        low, high = (
            np.asarray(value, dtype=np.float64)
            for value in _KA012_POT_CENTER_BOUNDS_XY
        )
        for attempt in range(1, _KA012_RESET_MAX_ATTEMPTS + 1):
            pose = authored.copy()
            pose[:2] = self._random.uniform(low, high)
            pose[3:7] = _quat_multiply(
                _yaw_quaternion(
                    float(
                        self._random.uniform(
                            -_KA012_RESET_YAW_DEG,
                            _KA012_RESET_YAW_DEG,
                        )
                    )
                ),
                authored[3:7],
            )
            if self._ka012_pot_reset_is_reachable(pose):
                self._ka012_reset_attempts = attempt
                return pose

        raise RuntimeError(
            "KA012 could not sample a reachable randomized pot reset after "
            f"{_KA012_RESET_MAX_ATTEMPTS} attempts"
        )

    @classmethod
    def _ka020_reset_layout_is_valid(
        cls,
        poses: dict[str, np.ndarray],
        authored: dict[str, np.ndarray],
    ) -> bool:
        """Validate support, table bounds, pair clearance, and arm reach."""

        if set(poses) != {"sponge", "soap_bottle"}:
            return False
        table_min = np.asarray([-0.57, -0.64], dtype=np.float64) + 0.020
        table_max = np.asarray([0.27, 0.64], dtype=np.float64) - 0.020
        footprint_radius = {"sponge": 0.055, "soap_bottle": 0.095}
        for name, pose in poses.items():
            pose = np.asarray(pose, dtype=np.float64).reshape(7)
            if not np.all(np.isfinite(pose)):
                return False
            if not math.isclose(
                float(np.linalg.norm(pose[3:7])),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                return False
            if not math.isclose(
                float(pose[2]),
                float(authored[name][2]),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                return False
            footprint = np.full(2, footprint_radius[name], dtype=np.float64)
            if np.any(pose[:2] - footprint < table_min) or np.any(
                pose[:2] + footprint > table_max
            ):
                return False
            interaction = pose[:3].copy()
            interaction[2] += _KA020_INTERACTION_HEIGHT_OFFSET_M[name]
            base = np.asarray(
                _KA020_CANONICAL_ARM_BASE_POSITIONS[name],
                dtype=np.float64,
            )
            if (
                float(np.linalg.norm(interaction - base))
                > _KA020_COMFORT_MAX_RADIUS_M
            ):
                return False
        return bool(
            float(
                np.linalg.norm(
                    poses["sponge"][:2] - poses["soap_bottle"][:2]
                )
            )
            >= _KA020_MIN_BODY_SEPARATION_M
        )

    def _sample_ka020_reset_poses(self) -> dict[str, np.ndarray]:
        """Sample one seeded, side-assigned layout for the two KA020 bodies."""

        if self.task_name != B320_KA020_TASK_NAME:
            return {}
        authored = {
            spec.name: np.asarray(
                [*spec.home_translation, *spec.home_orientation_wxyz],
                dtype=np.float64,
            )
            for spec in self._b320_contract.object_specs
        }
        if not self.randomize:
            self._ka020_reset_attempts = 0
            if not self._ka020_reset_layout_is_valid(authored, authored):
                raise RuntimeError("KA020 authored reset violates its layout contract")
            return {name: pose.copy() for name, pose in authored.items()}

        bounds = {
            "sponge": _KA020_SPONGE_CENTER_BOUNDS_XY,
            "soap_bottle": _KA020_BOTTLE_CENTER_BOUNDS_XY,
        }
        for attempt in range(1, _KA020_RESET_MAX_ATTEMPTS + 1):
            sampled: dict[str, np.ndarray] = {}
            for name, pose in authored.items():
                candidate = pose.copy()
                low, high = (
                    np.asarray(values, dtype=np.float64)
                    for values in bounds[name]
                )
                candidate[:2] = self._random.uniform(low, high)
                candidate[3:7] = _quat_multiply(
                    _yaw_quaternion(
                        float(
                            self._random.uniform(
                                -_KA020_RESET_YAW_DEG,
                                _KA020_RESET_YAW_DEG,
                            )
                        )
                    ),
                    pose[3:7],
                )
                sampled[name] = candidate
            if self._ka020_reset_layout_is_valid(sampled, authored):
                self._ka020_reset_attempts = attempt
                return sampled

        raise RuntimeError(
            "KA020 could not sample a valid randomized reset after "
            f"{_KA020_RESET_MAX_ATTEMPTS} attempts"
        )

    def _sample_ka021_reset_pose(self) -> np.ndarray | None:
        """Sample a seeded mug reset and matching outside-sink target pad."""

        if self.task_name not in B320_FAUCET_TASK_NAMES:
            return None
        authored = self._fallback_b320_pose("ceramic_mug")
        authored_target = np.asarray(KA021_CUP_PARK_POSITION_W, dtype=np.float64)
        if not _ka021_cup_reset_pose_fits_basin(authored):
            raise RuntimeError(
                "KA021 authored mug reset is not supported inside the sink basin"
            )
        if not self.randomize:
            self._ka021_reset_attempts = 0
            self._ka021_cup_park_position_w = authored_target.copy()
            return authored

        cup_low, cup_high = (
            np.asarray(values, dtype=np.float64)
            for values in _KA021_CUP_RESET_CENTER_BOUNDS_XY
        )
        park_low, park_high = (
            np.asarray(values, dtype=np.float64)
            for values in _KA021_CUP_PARK_CENTER_BOUNDS_XY
        )
        for attempt in range(1, _KA021_RESET_MAX_ATTEMPTS + 1):
            cup_xy = np.asarray(
                self._random.uniform(cup_low, cup_high),
                dtype=np.float64,
            )
            park_xy = np.asarray(
                self._random.uniform(park_low, park_high),
                dtype=np.float64,
            )
            if not _ka021_marker_fits_left_counter(park_xy):
                continue
            if float(np.linalg.norm(cup_xy - park_xy)) < _KA021_CUP_PARK_MIN_SEPARATION_M:
                continue

            pose = authored.copy()
            pose[:2] = cup_xy
            pose[3:7] = _quat_multiply(
                _yaw_quaternion(
                    float(
                        self._random.uniform(
                            -_KA021_RESET_YAW_DEG,
                            _KA021_RESET_YAW_DEG,
                        )
                    )
                ),
                authored[3:7],
            )
            if not _ka021_cup_reset_pose_fits_basin(pose):
                continue

            self._ka021_reset_attempts = attempt
            self._ka021_cup_park_position_w = np.asarray(
                [park_xy[0], park_xy[1], KA021_COUNTER_SURFACE_Z],
                dtype=np.float64,
            )
            return pose

        raise RuntimeError(
            "KA021 could not sample a valid mug/counter reset after "
            f"{_KA021_RESET_MAX_ATTEMPTS} attempts"
        )

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self.delta_h[:] = 0.0
        self._b320_contact_force = {"left": 0.0, "right": 0.0}
        self._b320_contact_peak = {"left": 0.0, "right": 0.0}
        self._b320_contact_seen = {"left": False, "right": False}
        self._b320_contact_available = {
            side: side in self._b320_contact_views for side in ("left", "right")
        }
        self._b320_contact_readable = {"left": False, "right": False}
        self._b320_carton_contact_detail_available = {
            "left": False,
            "right": False,
        }
        self._b320_carton_zone_valid = {"left": False, "right": False}
        self._b320_carton_zone_force = {"left": 0.0, "right": 0.0}
        self._b320_carton_table_contact_available = (
            self._b320_carton_table_contact_view is not None
        )
        self._b320_carton_table_contact_readable = False
        self._b320_carton_table_contact_force = 0.0
        self._b320_pen_table_contact_available = (
            self._b320_pen_table_contact_view is not None
        )
        self._b320_pen_table_contact_readable = False
        self._b320_pen_table_contact_force = 0.0
        self._b320_carton_transport_started = False
        self._b320_carton_transport_contact_lost = False
        self._b320_carton_airborne_substeps = 0
        self._b320_carton_airborne_now = False
        self._b320_carton_force_guard_targets = {
            "right": self._right_tcp_pose.copy(),
            "left": self._left_tcp_pose.copy(),
        }
        self._b320_carton_force_guard_hand_targets = {
            "right": self._right_hand.copy(),
            "left": self._left_hand.copy(),
        }
        self._b320_carton_prelock_lift_clamped = {
            "left": False,
            "right": False,
        }
        self._b320_carton_prelock_lift_clamp_count = {
            "left": 0,
            "right": 0,
        }
        self._b320_carton_force_guard_active = False
        self._b320_carton_force_guard_active_by_side = {
            "left": False,
            "right": False,
        }
        self._b320_carton_force_guard_clamped = {
            "left": False,
            "right": False,
        }
        self._b320_carton_force_guard_predicted_n = {
            "left": 0.0,
            "right": 0.0,
        }
        self._b320_carton_force_guard_inward_speed_m_s = {
            "left": 0.0,
            "right": 0.0,
        }
        self._b320_carton_force_guard_clamp_count = {
            "left": 0,
            "right": 0,
        }
        self._b320_carton_contact_outward = {
            "left": np.zeros(3, dtype=np.float64),
            "right": np.zeros(3, dtype=np.float64),
        }
        self._b320_carton_contact_direction_available = {
            "left": False,
            "right": False,
        }
        self._b320_carton_contact_lever_local = {
            "left": np.zeros(3, dtype=np.float64),
            "right": np.zeros(3, dtype=np.float64),
        }
        self._b320_carton_contact_lever_available = {
            "left": False,
            "right": False,
        }
        self._b320_contact_force_rise_rate_n_s = {
            "left": 0.0,
            "right": 0.0,
        }
        self._b320_carton_force_guard_verified_position = {
            "left": np.zeros(3, dtype=np.float64),
            "right": np.zeros(3, dtype=np.float64),
        }
        self._b320_carton_force_guard_verified_valid = {
            "left": False,
            "right": False,
        }
        self._b320_carton_force_guard_pending_inward_m = {
            "left": 0.0,
            "right": 0.0,
        }
        self._b320_carton_bilateral_pressure_reached = {
            "left": False,
            "right": False,
        }
        self._b320_carton_bilateral_lock_active = False
        self._b320_carton_bilateral_lock_relative_position = np.zeros(
            3,
            dtype=np.float64,
        )
        self._b320_carton_bilateral_lock_latch_step = None
        self._b320_carton_bilateral_lock_common_delta = np.zeros(
            3,
            dtype=np.float64,
        )
        self._b320_carton_bilateral_lock_discarded_delta = {
            "left": np.zeros(3, dtype=np.float64),
            "right": np.zeros(3, dtype=np.float64),
        }
        self._b320_carton_bilateral_lock_orientation = {
            "left": self._left_tcp_pose[3:7].copy(),
            "right": self._right_tcp_pose[3:7].copy(),
        }
        self._b320_carton_bilateral_lock_hand_targets = {
            "left": self._left_hand.copy(),
            "right": self._right_hand.copy(),
        }
        self._b320_carton_bilateral_lock_force_rejected_axes = np.zeros(
            3,
            dtype=bool,
        )
        self._b320_carton_bilateral_lock_force_clamp_count = 0
        self._b320_carton_bilateral_lift_bypass_count = 0
        self._b320_carton_force_hold_active = {
            "left": False,
            "right": False,
        }
        self._b320_carton_force_hold_inward_step_m = {
            "left": 0.0,
            "right": 0.0,
        }
        self._b320_carton_force_hold_correction_count = {
            "left": 0,
            "right": 0,
        }
        self._b320_carton_force_hold_lift_block_count = 0
        self._b320_carton_force_relief_active = {
            "left": False,
            "right": False,
        }
        self._b320_carton_force_relief_count = {
            "left": 0,
            "right": 0,
        }
        self._b320_carton_bilateral_lock_clamp_count = 0
        self._b320_failure_reason = None
        self._b320_release_frame = {"left": None, "right": None}
        self._b320_threshold_frame = {"left": None, "right": None}
        self._b320_reset_poses = {}
        self._bc014_reset_poses = self._sample_bc014_reset_poses()
        self._ka012_reset_pose = self._sample_ka012_reset_pose()
        self._ka020_reset_poses = self._sample_ka020_reset_poses()
        self._ka021_reset_pose = self._sample_ka021_reset_pose()
        if self.task_name == B320_BC014_TASK_NAME:
            try:
                self._set_bc014_goal_marker_visuals()
            except Exception as exc:
                if not self._bc014_goal_visual_warning_emitted:
                    self._add_task_state_warning(
                        "Could not move BC014 highlighted goal markers: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    self._bc014_goal_visual_warning_emitted = True
        self._bc014_flow_hit_seconds = {"left": 0.0, "right": 0.0}
        self._bc014_flow_pour_seconds = {"left": 0.0, "right": 0.0}
        self._bc014_flow_overlap_seconds = 0.0
        self._bc014_flow_hit_latched = {"left": False, "right": False}
        self._bc014_flow_overlap_latched = False
        self._bc014_flow_xform_ops.clear()
        if self.task_name == B320_BC014_TASK_NAME:
            self._set_bc014_flow_visual("salt_jar", None)
            self._set_bc014_flow_visual("pepper_jar", None)
        self._ka020_pump_contact_force_n = 0.0
        self._ka020_pump_contact_readable = False
        self._ka020_flow_xform_ops = None
        self._ka020_last_flow_sample = {}
        if self.task_name == B320_KA020_TASK_NAME:
            self._set_ka020_flow_visual(None)
        self._ka021_flow_xform_ops = None
        self._ka021_last_flow_sample = {}
        self._ka021_fill_hold_frames = 0
        self._ka021_fill_last_update_step = None
        self._ka021_water_filled = False
        if self.task_name in B320_FAUCET_TASK_NAMES:
            self._set_ka021_flow_visual(None)
        self._b320_last_metrics = {}
        self._b320_condition_flags = (False,) * len(self._b320_contract.success_labels)
        self._b320_handover_right_first = False
        self._b320_handover_overlap = False
        self._b320_handover_left_only = False
        self._b320_handover_completed = False
        self._bc011_pen_airborne_latched = False
        self._bc011_pen_ground_contact_substeps = 0
        self._bc011_pen_drop_detected = False
        self._b320_left_contact_lost = False
        self._b320_stir_previous_angle = None
        self._b320_stir_accumulated_rad = 0.0
        self._b320_stir_sectors.clear()
        self._b320_stir_anchor_pose = None
        self._b320_stir_anchor_frame = None
        if self.task_name in B320_DUAL_KNOB_TASK_NAMES:
            self._ka005_2_reset_sample = sample_dual_knob_reset(
                self._random,
                randomize=bool(self.randomize),
            )
            self._ka005_2_state.reset(
                self._ka005_2_reset_sample.initial_rad,
                self._ka005_2_reset_sample.target_rad,
            )
            self._ka005_2_contact_force_n = {
                key: 0.0 for key in KA005_2_CONTACT_KEYS
            }
            self._ka005_2_contact_readable = {
                key: False for key in KA005_2_CONTACT_KEYS
            }
            self._ka005_2_recording_origin_accepted = False
            self._ka005_2_recording_origin_q_rad = np.zeros(
                2,
                dtype=np.float64,
            )
            self._ka005_2_recording_origin_poses = {}
            self._set_ka005_2_target_markers()
        if not self._launch_isaac:
            # The no-Isaac interface path has no PhysX reset callback to
            # refresh object poses.  Restore the declared fallback poses
            # explicitly, then capture them as the transport/origin
            # baseline.  Otherwise BC004 inherited a stale carton pose while
            # its baseline remained empty and failed on the very first zero
            # action as ``carton_transport_support_lost``.
            self._populate_interface_poses()
            interface_names = (
                set(self._spec.target_objects)
                | {self._spec.receptacle, "table"}
                | set(self._b320_spawn_result.body_paths)
            )
            for name in interface_names:
                reset_pose = self._bc014_reset_poses.get(name)
                if name in self._ka020_reset_poses:
                    reset_pose = self._ka020_reset_poses[name]
                elif (
                    name == "soap_dispense_pad"
                    and "soap_bottle" in self._ka020_reset_poses
                ):
                    bottle_pose = self._ka020_reset_poses["soap_bottle"]
                    reset_pose = np.concatenate(
                        [
                            bottle_pose[:3]
                            + _rotation_matrix(bottle_pose[3:7])
                            @ np.asarray(
                                KA020_PUMP_HEAD_CENTER_LOCAL_M,
                                dtype=np.float64,
                            ),
                            bottle_pose[3:7],
                        ]
                    )
                if name == "dutch_oven" and self._ka012_reset_pose is not None:
                    reset_pose = self._ka012_reset_pose
                if name == "ceramic_mug" and self._ka021_reset_pose is not None:
                    reset_pose = self._ka021_reset_pose
                self._poses[name] = (
                    reset_pose
                    if reset_pose is not None
                    else self._fallback_b320_pose(name)
                ).copy()
            self._b320_reset_poses = {
                name: self._poses[name].copy()
                for name in interface_names
                if name in self._poses
            }

    def _randomized_pose(
        self,
        pose: np.ndarray,
        *,
        fixture: bool,
        object_name: str | None = None,
    ) -> np.ndarray:
        result = np.asarray(pose, dtype=np.float64).reshape(7).copy()
        if not self.randomize:
            return result
        if (
            self.task_name == B320_BC014_TASK_NAME
            and object_name in {"salt_jar", "pepper_jar"}
            and object_name in self._bc014_reset_poses
        ):
            return self._bc014_reset_poses[object_name].copy()
        if (
            self.task_name == B320_KA020_TASK_NAME
            and object_name in self._ka020_reset_poses
        ):
            return self._ka020_reset_poses[str(object_name)].copy()
        if (
            self.task_name == B320_KA020_TASK_NAME
            and object_name == "soap_dispense_pad"
            and "soap_bottle" in self._ka020_reset_poses
        ):
            bottle_pose = self._ka020_reset_poses["soap_bottle"]
            return np.concatenate(
                [
                    bottle_pose[:3]
                    + _rotation_matrix(bottle_pose[3:7])
                    @ np.asarray(
                        KA020_PUMP_HEAD_CENTER_LOCAL_M,
                        dtype=np.float64,
                    ),
                    bottle_pose[3:7],
                ]
            )
        if (
            self.task_name == _KA012_TASK_NAME
            and object_name == "dutch_oven"
            and self._ka012_reset_pose is not None
        ):
            return self._ka012_reset_pose.copy()
        if (
            self.task_name == "t10_lift_carton"
            and object_name == "carton"
        ):
            # D08 randomizes small rigid items without changing their authored
            # orientation.  Keep the same bounded-reset principle and narrower
            # lateral range so the 0.47 m carton remains clear of both spread
            # palms and fully supported by the D08-sized tabletop.
            result[0] += float(self._random.uniform(-0.025, 0.025))
            result[1] += float(self._random.uniform(-0.020, 0.020))
            return result
        if self.task_name in B320_FAUCET_TASK_NAMES and fixture:
            # KA021's faucet is authored against the fixed HS005 sink frame.
            # Its outlet, basin floor, and hand-clearance evidence all use
            # that calibrated frame, so a randomized fixture pose would make
            # the metric proxy disagree with the visible/physical faucet.
            return result
        jitter = 0.030 if fixture else 0.035
        yaw_limit = 5.0 if fixture else 12.0
        result[:2] += self._random.uniform(-jitter, jitter, size=2)
        yaw = _yaw_quaternion(float(self._random.uniform(-yaw_limit, yaw_limit)))
        result[3:7] = _quat_multiply(yaw, result[3:7])
        return result

    def _capture_task_scene_state(self, task_mjcf: Any) -> None:
        """Keep reset snapshots native to Isaac's selected tensor frontend.

        The active Isaac Sim 5.1 backend creates these low-level PhysX views
        with the Warp frontend even while IsaacLab robot buffers use Torch.  The inherited
        collector intentionally normalizes its snapshots to Torch, which is
        incompatible with Warp view setters.  Re-snapshot only B320 views in
        their native frontend so reset and replay remain fail-closed.
        """

        super()._capture_task_scene_state(task_mjcf)
        if not self._launch_isaac or self._torch is None:
            return

        articulation_indices: dict[int, Any] = {}
        native_articulations: list[tuple[Any, ...]] = []
        for path, view, _indices, *_state in self._task_reset_articulation_states:
            root_pose_ref = view.get_root_transforms()
            root_velocity_ref = view.get_root_velocities()
            joint_pos_ref = view.get_dof_positions()
            joint_vel_ref = view.get_dof_velocities()
            indices = _indices_like(root_pose_ref, int(view.count), self._torch)
            articulation_indices[id(view)] = indices
            native_articulations.append(
                (
                    path,
                    view,
                    indices,
                    _array_like(root_pose_ref, _tensor_to_numpy(root_pose_ref), self._torch),
                    _array_like(
                        root_velocity_ref,
                        _tensor_to_numpy(root_velocity_ref),
                        self._torch,
                    ),
                    _array_like(joint_pos_ref, _tensor_to_numpy(joint_pos_ref), self._torch),
                    _array_like(joint_vel_ref, _tensor_to_numpy(joint_vel_ref), self._torch),
                )
            )
        self._task_reset_articulation_states[:] = native_articulations

        native_bodies: list[tuple[Any, ...]] = []
        for path, view, _indices, *_state in self._task_reset_body_states:
            transform_ref = view.get_transforms()
            velocity_ref = view.get_velocities()
            indices = _indices_like(transform_ref, int(view.count), self._torch)
            native_bodies.append(
                (
                    path,
                    view,
                    indices,
                    _array_like(transform_ref, _tensor_to_numpy(transform_ref), self._torch),
                    _array_like(velocity_ref, _tensor_to_numpy(velocity_ref), self._torch),
                )
            )
        native_articulation_views: dict[str, tuple[Any, Any]] = {}
        for name, (view, _indices) in self._task_articulation_views.items():
            indices = articulation_indices.get(id(view))
            if indices is None:
                indices = _indices_like(
                    view.get_root_transforms(),
                    int(view.count),
                    self._torch,
                )
            native_articulation_views[name] = (view, indices)

        root_views = {
            path: (view, indices)
            for path, view, indices, *_state in native_articulations
        }
        fixture_view = next(
            (
                root_views[path]
                for path in self._b320_spawn_result.articulation_root_paths
                if path in root_views
            ),
            None,
        )
        if fixture_view is not None:
            if self.task_name in {
                "b320_synchronized_double_cabinet_open",
                B320_BC017_TASK_NAME,
                *B320_DUAL_KNOB_TASK_NAMES,
            }:
                articulation_roles = set(self._b320_spawn_result.body_paths)
            elif self.task_name in {B320_KA021_TASK_NAME, XY4_TASK_NAME}:
                articulation_roles = {
                    "faucet_fixture",
                    "faucet_base",
                    "faucet_handle",
                }
            elif self.task_name == B320_KA020_TASK_NAME:
                articulation_roles = {"soap_bottle", "soap_dispense_pad"}
            else:
                articulation_roles = set()
            for name in articulation_roles:
                if name in self._b320_spawn_result.body_paths:
                    native_articulation_views[name] = fixture_view

        if self.task_name in {
            # FU001's cabinet doors are non-root links of the shared fixed
            # articulation.  They must be restored only through the reduced
            # coordinate view; registering them as rigid-body reset entries
            # makes PhysX reject the transform/velocity writes and can poison
            # the broadphase on the next seeded reset.
            "b320_synchronized_double_cabinet_open",
            B320_BC017_TASK_NAME,
            *B320_DUAL_KNOB_TASK_NAMES,
            B320_KA021_TASK_NAME,
            XY4_TASK_NAME,
        } and fixture_view is not None:
            # Each dedicated fixture and its two moving links form one
            # reduced-coordinate articulation. Reset them exclusively through
            # the articulation root/DOF view. Low-level rigid-body setters
            # reject non-root articulation links.
            fixture_body_paths = {
                path
                for name, path in self._b320_spawn_result.body_paths.items()
                if self.task_name not in B320_FAUCET_TASK_NAMES
                or name
                in {"faucet_fixture", "faucet_base", "faucet_handle"}
            }
            native_bodies = [
                entry
                for entry in native_bodies
                if str(entry[0]) not in fixture_body_paths
            ]

        # A stage can contain an articulation and independent rigid bodies at
        # the same time.  The shared classifier intentionally prefers the
        # articulation root, which would otherwise omit bodies such as the
        # KA020 sponge from reset truth.  Capture every remaining B320 body in
        # its native frontend exactly once.
        captured_body_views = {id(entry[1]) for entry in native_bodies}
        for name, view in self._task_body_views.items():
            if name in native_articulation_views or id(view) in captured_body_views:
                continue
            transform_ref = view.get_transforms()
            velocity_ref = view.get_velocities()
            indices = _indices_like(transform_ref, int(view.count), self._torch)
            native_bodies.append(
                (
                    self._b320_spawn_result.body_paths.get(name, name),
                    view,
                    indices,
                    _array_like(
                        transform_ref,
                        _tensor_to_numpy(transform_ref),
                        self._torch,
                    ),
                    _array_like(
                        velocity_ref,
                        _tensor_to_numpy(velocity_ref),
                        self._torch,
                    ),
                )
            )
            captured_body_views.add(id(view))
        self._task_reset_body_states[:] = native_bodies
        self._task_articulation_views = native_articulation_views
        self._task_dof_views = {
            name: (
                view,
                int(index),
                float(_tensor_to_numpy(view.get_dof_positions())[0, int(index)]),
            )
            for name, (view, index, _initial) in self._task_dof_views.items()
        }

    def _set_articulation_reset(
        self,
        view: Any,
        indices: Any,
        *,
        fixture: bool,
        object_name: str | None = None,
    ) -> None:
        transform_ref = view.get_root_transforms()
        velocity_ref = view.get_root_velocities()
        transforms = next(
            (
                _tensor_to_numpy(reset_transform)
                for (
                    _path,
                    candidate,
                    _indices,
                    reset_transform,
                    _reset_velocity,
                    _reset_position,
                    _reset_dof_velocity,
                ) in self._task_reset_articulation_states
                if candidate is view
            ),
            _tensor_to_numpy(transform_ref),
        )
        velocities = _tensor_to_numpy(velocity_ref)
        transform_np = transforms[0]
        pose = np.asarray(
            [
                *transform_np[:3],
                transform_np[6],
                transform_np[3],
                transform_np[4],
                transform_np[5],
            ],
            dtype=np.float64,
        )
        pose = self._randomized_pose(
            pose,
            fixture=fixture,
            object_name=object_name,
        )
        transforms[:, :3] = pose[:3]
        transforms[:, 3:7] = [pose[4], pose[5], pose[6], pose[3]]
        if not fixture:
            view.set_root_velocities(
                _array_like(
                    velocity_ref,
                    np.zeros_like(velocities),
                    self._torch,
                ),
                indices,
            )
            view.set_root_transforms(
                _array_like(transform_ref, transforms, self._torch),
                indices,
            )

        position_ref = view.get_dof_positions()
        positions = _tensor_to_numpy(position_ref)
        if int(positions.shape[-1]) == 0:
            return
        dof_indices = view.get_metatype(0).dof_indices
        fraction = (
            float(self._random.uniform(0.10, 0.30)) if self.randomize else 0.20
        )
        target_values = {
            "target_joint_revolute": _HINGE_TRAVEL_RAD * fraction,
            "joint_11": -_HINGE_TRAVEL_RAD * fraction,
            "RevoluteJoint_faucet_3_right": KA021_FAUCET_RESET_POSITION_RAD,
            "soap_dispense_joint": 0.0,
        }
        if self.task_name in B320_DUAL_KNOB_TASK_NAMES:
            target_values.update(
                {
                    "left_knob_joint": float(
                        self._ka005_2_reset_sample.initial_rad[0]
                    ),
                    "right_knob_joint": float(
                        self._ka005_2_reset_sample.initial_rad[1]
                    ),
                }
            )
        changed = False
        for name, value in target_values.items():
            if name not in dof_indices:
                continue
            positions[:, int(dof_indices[name])] = float(value)
            changed = True
        if changed:
            velocity_ref = view.get_dof_velocities()
            zero_dof_velocities = np.zeros_like(
                _tensor_to_numpy(velocity_ref)
            )
            position_payload = _array_like(position_ref, positions, self._torch)
            view.set_dof_velocities(
                _array_like(
                    velocity_ref,
                    zero_dof_velocities,
                    self._torch,
                ),
                indices,
            )
            view.set_dof_positions(
                position_payload,
                indices,
            )
            reset_path = next(
                (
                    str(path)
                    for path, candidate, *_state in (
                        self._task_reset_articulation_states
                    )
                    if candidate is view
                ),
                None,
            )
            if reset_path is not None:
                # The live tensor readback can still contain the prior
                # episode until the first solver step.  Journal the exact
                # sampled DOF boundary so recording-origin drift is measured
                # from the authored reset rather than USD's zero defaults.
                self._record_authored_task_reset_write(
                    reset_path,
                    dof_position=positions,
                    dof_velocity=zero_dof_velocities,
                )
            # The authored B320 fixture drives are holding drives.  Writing a
            # randomized reset position while leaving their target at USD's
            # default zero makes a furniture link drift back during recording
            # origin stabilization. Keep the drive
            # target aligned with the reset state.  A real Isaac fixture must
            # expose this API; silently continuing would recreate the old
            # origin-drift failure.  Interface-only mocks may omit it.
            if fixture:
                target_setter = getattr(view, "set_dof_position_targets", None)
                if not callable(target_setter):
                    if self._launch_isaac:
                        raise RuntimeError(
                            "B320 fixture articulation is missing "
                            "set_dof_position_targets; refusing an unheld "
                            "reset that can drift through the origin gate."
                        )
                else:
                    target_setter(position_payload, indices)
                    target_getter = getattr(
                        view, "get_dof_position_targets", None
                    )
                    if self._launch_isaac and not callable(target_getter):
                        raise RuntimeError(
                            "B320 fixture articulation is missing "
                            "get_dof_position_targets; refusing to certify "
                            "a reset without drive-target readback."
                        )
                    if self._launch_isaac:
                        try:
                            actual_targets = _tensor_to_numpy(
                                target_getter()
                            )
                            expected_targets = _tensor_to_numpy(
                                position_payload
                            )
                            if actual_targets.shape != expected_targets.shape:
                                raise RuntimeError(
                                    "fixture drive-target readback shape "
                                    f"mismatch: {actual_targets.shape} vs "
                                    f"{expected_targets.shape}"
                                )
                            if not np.allclose(
                                actual_targets,
                                expected_targets,
                                atol=1.0e-5,
                                rtol=0.0,
                            ):
                                raise RuntimeError(
                                    "fixture drive-target readback does not "
                                    "match the reset positions"
                                )
                        except RuntimeError:
                            raise
                        except Exception as exc:
                            raise RuntimeError(
                                "B320 fixture drive-target readback failed"
                            ) from exc

    def _set_task_body_world_pose(self, view: Any, pose_wxyz: np.ndarray) -> bool:
        """Write a rigid B320 pose through the view's native frontend."""

        self._forbid_physics_only_task_state_write(
            "task-body root pose/velocity writes"
        )
        if self._torch is None:
            return False
        transform_ref = view.get_transforms()
        velocity_ref = view.get_velocities()
        original_transforms = _tensor_to_numpy(transform_ref)
        original_velocities = _tensor_to_numpy(velocity_ref)
        transforms = original_transforms.copy()
        zero_velocities = np.zeros_like(original_velocities)
        pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
        transforms[:, :3] = pose[:3]
        transforms[:, 3:7] = [pose[4], pose[5], pose[6], pose[3]]
        indices = _indices_like(transform_ref, int(view.count), self._torch)
        try:
            view.set_velocities(
                _array_like(velocity_ref, zero_velocities, self._torch),
                indices,
            )
            view.set_transforms(
                _array_like(transform_ref, transforms, self._torch),
                indices,
            )
            subject = next(
                (
                    name
                    for name, candidate in self._task_body_views.items()
                    if candidate is view
                ),
                None,
            )
            reset_path = next(
                (
                    str(path)
                    for path, candidate, *_state in self._task_reset_body_states
                    if candidate is view
                ),
                None,
            )
            if reset_path is None and subject is not None:
                reset_path = self._b320_spawn_result.body_paths.get(subject)
            if reset_path is not None:
                # Record the exact NumPy setter payloads.  B320's low-level
                # PhysX views can use the Warp frontend, whose arrays are not
                # accepted by the shared synchronous reset-journal converter.
                self._record_authored_task_reset_write(
                    reset_path,
                    transform_xyzw=transforms,
                    velocity=zero_velocities,
                )
            self._record_trajectory_state_write(
                "_set_task_body_world_pose",
                subject=subject,
            )
            return True
        except Exception as exc:
            try:
                view.set_velocities(
                    _array_like(velocity_ref, original_velocities, self._torch),
                    indices,
                )
                view.set_transforms(
                    _array_like(transform_ref, original_transforms, self._torch),
                    indices,
                )
            except Exception:
                pass
            self._add_task_state_warning(
                f"Could not restore recorded B320 object pose: {exc}"
            )
            return False

    def _restore_recorded_table_height(self, delta_height: float) -> bool:
        table_prim = self._find_rigid_body_prim("table")
        if table_prim is None or self._torch is None:
            return False
        table_path = str(table_prim.GetPath())
        for path, view, indices, root_pose, root_velocity in self._task_reset_body_states:
            if path != table_path:
                continue
            pose_values = _tensor_to_numpy(root_pose)
            pose_values[:, 2] += float(delta_height)
            velocity_values = np.zeros_like(_tensor_to_numpy(root_velocity))
            self._write_rigid_reset_state(
                path,
                view,
                indices,
                _array_like(root_pose, pose_values, self._torch),
                _array_like(root_velocity, velocity_values, self._torch),
            )
            return True
        return False

    def _apply_task_pose_randomization(self) -> None:
        if self.task_name in B320_FAUCET_TASK_NAMES:
            try:
                moved = self._set_ka021_park_target_visuals()
                if (
                    self._launch_isaac
                    and not moved
                    and not self._ka021_park_visual_warning_emitted
                ):
                    self._add_task_state_warning(
                        "KA021 randomized counter-pad marker/support is incomplete"
                    )
                    self._ka021_park_visual_warning_emitted = True
            except Exception as exc:
                if not self._ka021_park_visual_warning_emitted:
                    self._add_task_state_warning(
                        "Could not move KA021 randomized counter-pad marker: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    self._ka021_park_visual_warning_emitted = True
        processed_articulations: set[int] = set()
        logical_names = list(self._b320_spawn_result.body_paths)
        for name in logical_names:
            articulation = self._task_articulation_views.get(name)
            if articulation is None or id(articulation[0]) in processed_articulations:
                continue
            view, indices = articulation
            processed_articulations.add(id(view))
            self._set_articulation_reset(
                view,
                indices,
                fixture=name
                in {
                    "fixture",
                    "fixture_base",
                    "faucet_fixture",
                    "faucet_base",
                    "panel_base",
                },
                object_name=name,
            )

        for name in logical_names:
            if name in self._task_articulation_views:
                continue
            view = self._task_body_views.get(name)
            reset_transform = next(
                (
                    _tensor_to_numpy(transform)[0].reshape(7)
                    for (
                        _path,
                        candidate,
                        _indices,
                        transform,
                        _velocity,
                    ) in self._task_reset_body_states
                    if candidate is view
                ),
                None,
            )
            pose = (
                np.concatenate(
                    [
                        reset_transform[:3],
                        reset_transform[6:7],
                        reset_transform[3:6],
                    ]
                ).astype(np.float64)
                if reset_transform is not None
                else self._physics_body_pose(name)
            )
            if view is None or pose is None:
                continue
            if (
                self.task_name in B320_FAUCET_TASK_NAMES
                and name == "ceramic_mug"
                and self._ka021_reset_pose is not None
            ):
                randomized_pose = self._ka021_reset_pose.copy()
            else:
                randomized_pose = self._randomized_pose(
                    pose,
                    fixture=False,
                    object_name=name,
                )
            self._set_task_body_world_pose(
                view,
                randomized_pose,
            )

    def _refresh_object_poses(self) -> None:
        interface_names = (
            set(self._spec.target_objects)
            | {self._spec.receptacle, "table"}
            | set(self._b320_spawn_result.body_paths)
        )
        if not self._launch_isaac:
            # There is no native prim/tensor readback in the interface-only
            # contract.  Keep the poses authored by _sample_reset_state (and
            # any explicit test/operator write) instead of replacing them on
            # every step with the shared ABC hash-based fallback.  The latter
            # made BC004's reset carton differ from its origin baseline and
            # triggered a false transport failure on action zero.
            for name in interface_names:
                self._poses.setdefault(name, self._fallback_b320_pose(name))
            if not self._b320_reset_poses:
                self._b320_reset_poses = {
                    name: self._poses[name].copy()
                    for name in interface_names
                    if name in self._poses
                }
            return
        super()._refresh_object_poses()
        for name in self._b320_spawn_result.body_paths:
            self._poses[name] = self._prim_pose(name, self._fallback_b320_pose(name))
        for name in self._task_extra_pose_names():
            self._poses.setdefault(name, self._fallback_b320_pose(name))
        if not self._b320_reset_poses:
            self._b320_reset_poses = {
                name: self._poses[name].copy()
                for name in self._b320_spawn_result.body_paths
                if name in self._poses
            }

    def _pose(self, name: str) -> np.ndarray:
        return _finite_pose(
            self._poses.get(name, self._fallback_b320_pose(name)),
            self._fallback_b320_pose(name),
        )

    def _velocity(self, name: str) -> np.ndarray:
        value = self._physics_body_velocity(name)
        if value is None:
            return np.zeros(6, dtype=np.float64)
        value = np.asarray(value, dtype=np.float64).reshape(6)
        return value if np.all(np.isfinite(value)) else np.zeros(6, dtype=np.float64)

    def _velocity_with_validity(self, name: str) -> tuple[np.ndarray, bool]:
        if not self._launch_isaac:
            value = np.asarray(self._velocity(name), dtype=np.float64).reshape(6)
            return value, bool(np.all(np.isfinite(value)))
        value = self._physics_body_velocity(name)
        if value is None:
            return np.zeros(6, dtype=np.float64), False
        value = np.asarray(value, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(value)):
            return np.zeros(6, dtype=np.float64), False
        return value, True

    def _physics_body_pose(self, name: str) -> np.ndarray | None:
        view = self._task_body_views.get(name)
        if view is None:
            return None
        try:
            transform = _tensor_to_numpy(view.get_transforms())[0].reshape(7)
            return np.concatenate(
                [transform[:3], transform[6:7], transform[3:6]]
            ).astype(np.float64)
        except Exception:
            return None

    def _physics_body_velocity(self, name: str) -> np.ndarray | None:
        view = self._task_body_views.get(name)
        if view is None:
            return None
        try:
            return _tensor_to_numpy(view.get_velocities())[0].reshape(6).astype(np.float64)
        except Exception:
            return None

    def _physics_dof_position(self, name: str) -> tuple[float, float] | None:
        entry = self._task_dof_views.get(name)
        if entry is None:
            entry = next(
                (
                    value
                    for dof_name, value in self._task_dof_views.items()
                    if name in dof_name
                ),
                None,
            )
        if entry is None:
            return None
        view, index, initial = entry
        try:
            value = _tensor_to_numpy(view.get_dof_positions())[0, int(index)]
            return float(value), float(initial)
        except Exception:
            return None

    @staticmethod
    def _read_b320_contact_force_sample(
        view: Any | None,
        dt: float,
    ) -> tuple[float, bool]:
        if view is None:
            return 0.0, False
        try:
            forces = _tensor_to_numpy(view.get_contact_force_matrix(dt)).astype(
                np.float64
            )
            if forces.size == 0 or not np.all(np.isfinite(forces)):
                return _INVALID_CONTACT_FORCE_N, False
            return float(np.linalg.norm(forces, axis=-1).sum()), True
        except Exception:
            return _INVALID_CONTACT_FORCE_N, False

    @staticmethod
    def _read_b320_contact_force(view: Any | None, dt: float) -> float:
        """Compatibility helper for tests and diagnostics that need one value."""

        return HandsOnB320BimanualIsaacLabEnv._read_b320_contact_force_sample(
            view,
            dt,
        )[0]

    @staticmethod
    def _read_b320_contact_points(
        view: Any | None,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return only valid detailed-contact rows from one filtered view."""

        getter = getattr(view, "get_contact_data", None)
        if not callable(getter):
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                False,
            )
        try:
            data = getter(dt)
            if not isinstance(data, (tuple, list)) or len(data) < 6:
                return (
                    np.zeros((0, 3), dtype=np.float64),
                    np.zeros(0, dtype=np.float64),
                    False,
                )
            normal_forces = _tensor_to_numpy(data[0]).astype(np.float64).reshape(-1)
            points = _tensor_to_numpy(data[1]).astype(np.float64).reshape(-1, 3)
            counts = _tensor_to_numpy(data[4]).astype(np.int64).reshape(-1)
            starts = _tensor_to_numpy(data[5]).astype(np.int64).reshape(-1)
            if (
                counts.shape != starts.shape
                or np.any(counts < 0)
                or np.any(starts < 0)
            ):
                return (
                    np.zeros((0, 3), dtype=np.float64),
                    np.zeros(0, dtype=np.float64),
                    False,
                )
            point_rows: list[np.ndarray] = []
            force_rows: list[np.ndarray] = []
            for start, count in zip(starts, counts, strict=True):
                stop = int(start) + int(count)
                if stop > len(points) or stop > len(normal_forces):
                    return (
                        np.zeros((0, 3), dtype=np.float64),
                        np.zeros(0, dtype=np.float64),
                        False,
                    )
                if count:
                    point_rows.append(points[int(start) : stop])
                    force_rows.append(normal_forces[int(start) : stop])
            valid_points = (
                np.concatenate(point_rows, axis=0)
                if point_rows
                else np.zeros((0, 3), dtype=np.float64)
            )
            valid_forces = (
                np.concatenate(force_rows, axis=0)
                if force_rows
                else np.zeros(0, dtype=np.float64)
            )
            if not np.all(np.isfinite(valid_points)) or not np.all(
                np.isfinite(valid_forces)
            ):
                return (
                    np.zeros((0, 3), dtype=np.float64),
                    np.zeros(0, dtype=np.float64),
                    False,
                )
            return valid_points, valid_forces, True
        except Exception:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                False,
            )

    def _contact_sample_readable(self, side: str) -> bool:
        """Whether this hand's latest force sample is trustworthy."""
        force = self._b320_contact_force[side]
        return bool(
            self._b320_contact_available[side]
            and self._b320_contact_readable[side]
            and np.isfinite(force)
        )

    def _contact_at_least(self, side: str, threshold: float) -> bool:
        return bool(
            self._contact_sample_readable(side)
            and self._b320_contact_force[side] >= float(threshold)
        )

    def _bc011_contacting(self, side: str) -> bool:
        """Treat any positive readable whole-pen contact as BC011 contact."""

        return bool(
            self._contact_sample_readable(side)
            and self._b320_contact_force[side] > 0.0
        )

    def _bc011_pen_lift_height(self) -> float:
        pose = self._pose("pen")
        reset_pose = self._b320_reset_poses.get(
            "pen",
            self._fallback_b320_pose("pen"),
        )
        return float(pose[2] - reset_pose[2])

    def _update_bc011_drop_latches(self) -> None:
        """Latch post-overlap pen-table contact as a terminal BC011 drop."""

        if (
            self.task_name != "t20_pass_and_orient_pen"
            or self._bc011_pen_drop_detected
        ):
            return
        table_clear = bool(
            self._b320_pen_table_contact_available
            and self._b320_pen_table_contact_readable
            and np.isfinite(self._b320_pen_table_contact_force)
            and self._b320_pen_table_contact_force <= 0.0
        )
        if (
            self._b320_handover_overlap
            and self._bc011_pen_lift_height() >= _BC011_PICKUP_LIFT_M
            and table_clear
        ):
            self._bc011_pen_airborne_latched = True

        ground_contact = bool(
            self._b320_pen_table_contact_available
            and self._b320_pen_table_contact_readable
            and np.isfinite(self._b320_pen_table_contact_force)
            and self._b320_pen_table_contact_force > 0.0
        )
        if self._bc011_pen_airborne_latched and ground_contact:
            self._bc011_pen_ground_contact_substeps += 1
        else:
            self._bc011_pen_ground_contact_substeps = 0
        if (
            self._bc011_pen_ground_contact_substeps
            >= _BC011_DROP_DEBOUNCE_SUBSTEPS
        ):
            self._bc011_pen_drop_detected = True

    def _carton_contact_zone_matches(
        self,
        side: str,
        world_points: np.ndarray,
        normal_forces: np.ndarray,
    ) -> tuple[bool, float]:
        points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
        forces = np.asarray(normal_forces, dtype=np.float64).reshape(-1)
        if not len(points) or len(points) != len(forces):
            return False, 0.0
        pose = self._physics_body_pose("carton")
        if pose is None:
            pose = self._pose("carton")
        local = (points - pose[:3]) @ _rotation_matrix(pose[3:7])
        expected_side = (
            local[:, 1] >= _CARTON_SIDE_ZONE_MIN_ABS_Y_M
            if side == "left"
            else local[:, 1] <= -_CARTON_SIDE_ZONE_MIN_ABS_Y_M
        )
        lower_edge = np.logical_and(
            local[:, 2] >= _CARTON_ZONE_MIN_Z_M,
            local[:, 2] <= _CARTON_LOWER_ZONE_MAX_Z_M,
        )
        longitudinal = np.abs(local[:, 0]) <= _CARTON_ZONE_MAX_ABS_X_M
        zone_mask = expected_side & lower_edge & longitudinal
        zone_force = float(np.abs(forces[zone_mask]).sum())
        return zone_force >= _HAND_CONTACT_N, zone_force

    def _carton_bottom_clearance(self) -> tuple[float, bool, float]:
        """Return the lowest transformed carton corner above the table."""

        physics_pose = self._physics_body_pose("carton")
        pose_readable = physics_pose is not None
        pose = physics_pose if pose_readable else self._pose("carton")
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(pose)):
            return -1.0, False, 180.0
        quaternion_norm = float(np.linalg.norm(pose[3:7]))
        if not math.isclose(
            quaternion_norm,
            1.0,
            rel_tol=0.0,
            abs_tol=_CARTON_QUATERNION_NORM_TOLERANCE,
        ):
            return -1.0, False, 180.0
        corners = np.asarray(
            [
                (x, y, z)
                for x in (_CARTON_LOCAL_MIN_M[0], _CARTON_LOCAL_MAX_M[0])
                for y in (_CARTON_LOCAL_MIN_M[1], _CARTON_LOCAL_MAX_M[1])
                for z in (_CARTON_LOCAL_MIN_M[2], _CARTON_LOCAL_MAX_M[2])
            ],
            dtype=np.float64,
        )
        world_corners = corners @ _rotation_matrix(pose[3:7]).T + pose[:3]
        clearance = float(np.min(world_corners[:, 2]) - B320_TABLE_TOP_Z)
        tilt = _angle_degrees(
            _axis_world(pose, (0.0, 0.0, 1.0)),
            (0.0, 0.0, 1.0),
        )
        # Interface-only mode has no native PhysX view, but its injected pose
        # remains useful for deterministic predicate tests.  Live success must
        # always come from the physical body readback.
        return clearance, bool(pose_readable or not self._launch_isaac), tilt

    def _carton_support_valid(self) -> bool:
        return bool(
            all(self._b320_contact_available.values())
            and all(self._b320_contact_readable.values())
            and all(self._b320_carton_contact_detail_available.values())
            and all(self._b320_carton_zone_valid.values())
            and self._b320_contact_force["left"] >= _HAND_CONTACT_N
            and self._b320_contact_force["right"] >= _HAND_CONTACT_N
        )

    def _carton_force_velocity_guard_ready(self, side: str) -> bool:
        """Whether this hand has enough physical truth for velocity gating."""

        return bool(
            self.task_name == "t10_lift_carton"
            and self._contact_sample_readable(side)
            and (
                self._b320_contact_force[side] >= _HAND_CONTACT_N
                or self._b320_carton_contact_direction_available[side]
            )
        )

    def _carton_recorded_force_n(self, value: float) -> float:
        """Apply BC004's internal and recorded side-force saturation."""

        force = float(value)
        if (
            self.task_name != "t10_lift_carton"
            or not np.isfinite(force)
        ):
            return force
        return min(force, _CARTON_RECORDED_FORCE_MAX_N)

    def _carton_fallback_outward_direction(
        self,
        side: str,
        carton_pose: np.ndarray,
    ) -> np.ndarray:
        side_axis = _rotation_matrix(carton_pose[3:7])[:, 1]
        return side_axis * (1.0 if side == "left" else -1.0)

    def _carton_contact_outward_direction(
        self,
        side: str,
        points: np.ndarray,
        normal_forces: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Infer the horizontal outward normal from the actual contact patch."""

        contact_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        forces = np.abs(
            np.asarray(normal_forces, dtype=np.float64).reshape(-1)
        )
        if not len(contact_points) or len(contact_points) != len(forces):
            return np.zeros(3, dtype=np.float64), False
        carton_pose = self._physics_body_pose("carton")
        if carton_pose is None:
            carton_pose = self._pose("carton")
        carton_pose = np.asarray(carton_pose, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(carton_pose)):
            return np.zeros(3, dtype=np.float64), False

        radial = contact_points - carton_pose[:3]
        radial[:, 2] = 0.0
        lengths = np.linalg.norm(radial, axis=1)
        valid = lengths > 1.0e-6
        if not np.any(valid):
            return np.zeros(3, dtype=np.float64), False
        unit = radial[valid] / lengths[valid, None]
        weights = forces[valid]
        if float(weights.sum()) <= 1.0e-9:
            weights = np.ones_like(weights)
        direction = np.sum(unit * weights[:, None], axis=0)
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-6:
            direction = self._carton_fallback_outward_direction(
                side,
                carton_pose,
            )
            norm = float(np.linalg.norm(direction))
        return direction / max(norm, 1.0e-12), True

    def _carton_physical_palm_position(self, side: str) -> np.ndarray | None:
        pose = self._carton_physical_palm_pose(side)
        if pose is None:
            return None
        return pose[:3].copy()

    def _carton_physical_palm_pose(self, side: str) -> np.ndarray | None:
        matrix = self._robot_link_pose_matrix(side)
        if matrix is None:
            return None
        pose = self._pose_matrix_to_vector(matrix)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            return None
        return pose.copy()

    def _carton_actual_hand_target(self, side: str) -> np.ndarray:
        commanded = self._right_hand if side == "right" else self._left_hand
        actual = np.asarray(
            self._actual_hand_joint_pose(side),
            dtype=np.float64,
        ).reshape(-1)
        if actual.shape != commanded.shape or not np.all(np.isfinite(actual)):
            return commanded.copy()
        return actual.copy()

    def _carton_current_bilateral_pressure_ready(self) -> bool:
        """Whether both hands currently read the saturated 10 N grasp force."""

        return bool(
            self.task_name == "t10_lift_carton"
            and all(
                self._contact_at_least(
                    side,
                    _CARTON_BILATERAL_LOCK_FORCE_N,
                )
                for side in ("left", "right")
            )
        )

    def _apply_carton_prelock_lift_guard(self) -> bool:
        """Reject upward wrist motion until both hands currently hold 10 N.

        X/Y translation, rotation, and every finger target remain untouched so
        the operator can establish the grasp.  Once both force samples reach
        10 N together, the bilateral lock owns all subsequent motion.
        """

        self._b320_carton_prelock_lift_clamped = {
            "left": False,
            "right": False,
        }
        if (
            self.task_name != "t10_lift_carton"
            or self._b320_carton_bilateral_lock_active
            or self._carton_current_bilateral_pressure_ready()
        ):
            return False
        previous = self._b320_carton_force_guard_targets
        if set(previous) != {"left", "right"}:
            return False

        targets = {
            "right": self._right_tcp_pose.copy(),
            "left": self._left_tcp_pose.copy(),
        }
        for side in ("left", "right"):
            if targets[side][2] <= previous[side][2] + 1.0e-12:
                continue
            targets[side][2] = previous[side][2]
            self._b320_carton_prelock_lift_clamped[side] = True
            self._b320_carton_prelock_lift_clamp_count[side] += 1

        self._right_tcp_pose = targets["right"]
        self._left_tcp_pose = targets["left"]
        return any(self._b320_carton_prelock_lift_clamped.values())

    def _apply_carton_force_velocity_guard(
        self,
        *,
        lookahead_seconds: float | None = None,
    ) -> bool:
        """Drop only a velocity component that predicts >10 N compression.

        Each hand is evaluated independently.  The contact patch chooses the
        direction, so a finger touching a front/back face is not incorrectly
        treated as a conventional left/right side contact.  While force remains
        below 8 N, the physical palm position certifies how much of an earlier
        position request has actually executed safely.  If force later predicts
        more than 10 N, the still-pending inward part is discarded even when the
        latest input frame itself is stationary.  Rotation is evaluated at the
        measured contact patch as well.  When measured force is already rising
        toward 10 N, new orientation and finger requests are also ignored because
        their individual contact-normal contribution is not observable from the
        operator command.  Tangential translation, lift, and the other hand stay
        independent until the bilateral grasp locks.
        """

        targets = {
            "right": self._right_tcp_pose.copy(),
            "left": self._left_tcp_pose.copy(),
        }
        previous = self._b320_carton_force_guard_targets
        if set(previous) != {"left", "right"}:
            previous = {side: pose.copy() for side, pose in targets.items()}
        hands = {
            "right": self._right_hand.copy(),
            "left": self._left_hand.copy(),
        }
        previous_hands = self._b320_carton_force_guard_hand_targets
        if set(previous_hands) != {"left", "right"}:
            previous_hands = {
                side: hand.copy() for side, hand in hands.items()
            }

        self._b320_carton_force_guard_active_by_side = {
            side: self._carton_force_velocity_guard_ready(side)
            for side in ("left", "right")
        }
        self._b320_carton_force_guard_active = any(
            self._b320_carton_force_guard_active_by_side.values()
        )
        self._b320_carton_force_guard_clamped = {
            "left": False,
            "right": False,
        }
        self._b320_carton_force_relief_active = {
            "left": False,
            "right": False,
        }
        self._b320_carton_force_guard_predicted_n = {
            "left": 0.0,
            "right": 0.0,
        }
        self._b320_carton_force_guard_inward_speed_m_s = {
            "left": 0.0,
            "right": 0.0,
        }
        self._b320_carton_force_guard_pending_inward_m = {
            "left": 0.0,
            "right": 0.0,
        }
        if not self._b320_carton_force_guard_active:
            return False

        carton_pose = self._physics_body_pose("carton")
        if carton_pose is None:
            carton_pose = self._pose("carton")
        carton_pose = np.asarray(carton_pose, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(carton_pose)):
            return False
        lookahead_seconds = max(
            float(
                self._control_dt
                if lookahead_seconds is None
                else lookahead_seconds
            ),
            float(self._sim_dt),
        )
        for side in ("left", "right"):
            if not self._b320_carton_force_guard_active_by_side[side]:
                continue
            if self._b320_carton_contact_direction_available[side]:
                outward = self._b320_carton_contact_outward[side].copy()
            else:
                outward = self._carton_fallback_outward_direction(
                    side,
                    carton_pose,
                )
            requested = targets[side]
            accepted = requested.copy()
            outward_delta = float(
                np.dot(requested[:3] - previous[side][:3], outward)
            )
            inward_displacement = max(0.0, -outward_delta)
            rotation_inward_displacement = 0.0
            if self._b320_carton_contact_lever_available[side]:
                lever_local = self._b320_carton_contact_lever_local[side]
                contact_rotation_delta = (
                    _rotation_matrix(requested[3:7])
                    - _rotation_matrix(previous[side][3:7])
                ) @ lever_local
                rotation_inward_displacement = max(
                    0.0,
                    -float(np.dot(contact_rotation_delta, outward)),
                )
            verified_position = self._b320_carton_force_guard_verified_position[
                side
            ]
            verified_valid = bool(
                self._b320_carton_force_guard_verified_valid[side]
            )
            pending_inward = 0.0
            if verified_valid:
                pending_inward = max(
                    0.0,
                    -float(
                        np.dot(
                            requested[:3] - verified_position,
                            outward,
                        )
                    ),
                )
            guarded_translation_inward = max(
                inward_displacement,
                pending_inward,
            )
            guarded_inward = (
                guarded_translation_inward
                + rotation_inward_displacement
            )
            inward_speed = guarded_inward / lookahead_seconds
            force_n = float(self._b320_contact_force[side])
            displacement_prediction = (
                guarded_inward
                * _CARTON_FORCE_GUARD_MIN_STIFFNESS_N_PER_M
            )
            rate_prediction = (
                max(
                    0.0,
                    float(self._b320_contact_force_rise_rate_n_s[side]),
                )
                * lookahead_seconds
            )
            predicted_force = force_n + max(
                displacement_prediction,
                rate_prediction,
            )
            side_clamped = False
            force_response_risk = bool(
                force_n > _CARTON_FORCE_VELOCITY_GUARD_N
                or force_n + rate_prediction
                > _CARTON_FORCE_VELOCITY_GUARD_N
            )
            if force_response_risk:
                # Position control keeps executing an old target even when all
                # new tracker deltas are ignored.  Cancel that latent error at
                # the measured palm/hand state, then create a small outward-only
                # unloading target.  Repeating this at 120 Hz is bounded and
                # stops as soon as the sampled response drops below 10 N.
                physical_pose = self._carton_physical_palm_pose(side)
                accepted = (
                    physical_pose
                    if physical_pose is not None
                    else previous[side].copy()
                )
                accepted[:3] += outward * _CARTON_FORCE_RELIEF_STEP_M
                hands[side] = self._carton_actual_hand_target(side)
                self._b320_carton_force_relief_active[side] = True
                self._b320_carton_force_relief_count[side] += 1
                side_clamped = True
            if (
                not force_response_risk
                and guarded_translation_inward > 1.0e-9
                and predicted_force > _CARTON_FORCE_VELOCITY_GUARD_N
            ):
                # Discard the complete unverified compressive component.  The
                # verified position was captured while the measured response was
                # <=8 N; it is not a newly sampled stop target authored here.
                reference_position = (
                    verified_position
                    if verified_valid
                    else previous[side][:3]
                )
                rejected_component = float(
                    np.dot(reference_position - accepted[:3], outward)
                )
                accepted[:3] += outward * rejected_component
                side_clamped = True
            posture_risk = bool(
                force_n + rate_prediction
                > _CARTON_FORCE_VELOCITY_GUARD_N
            )
            requested_quaternion = requested[3:7]
            previous_quaternion = previous[side][3:7]
            requested_norm = float(np.linalg.norm(requested_quaternion))
            previous_norm = float(np.linalg.norm(previous_quaternion))
            orientation_changed = bool(
                requested_norm > 1.0e-12
                and previous_norm > 1.0e-12
                and abs(
                    float(
                        np.dot(
                            requested_quaternion / requested_norm,
                            previous_quaternion / previous_norm,
                        )
                    )
                )
                < 1.0 - 1.0e-10
            )
            if (
                not force_response_risk
                and orientation_changed
                and predicted_force > _CARTON_FORCE_VELOCITY_GUARD_N
                and (
                    rotation_inward_displacement > 1.0e-9
                    or posture_risk
                )
            ):
                accepted[3:7] = previous_quaternion
                side_clamped = True
            if (
                not force_response_risk
                and posture_risk
                and not np.allclose(
                    hands[side],
                    previous_hands[side],
                    rtol=0.0,
                    atol=1.0e-10,
                )
            ):
                hands[side] = previous_hands[side].copy()
                side_clamped = True
            if side_clamped:
                self._b320_carton_force_guard_clamped[side] = True
                self._b320_carton_force_guard_clamp_count[side] += 1
            self._b320_carton_force_guard_predicted_n[side] = predicted_force
            self._b320_carton_force_guard_inward_speed_m_s[side] = inward_speed
            self._b320_carton_force_guard_pending_inward_m[side] = (
                pending_inward
            )
            targets[side] = accepted

        self._right_tcp_pose = targets["right"]
        self._left_tcp_pose = targets["left"]
        self._right_hand = hands["right"]
        self._left_hand = hands["left"]
        return any(self._b320_carton_force_guard_clamped.values())

    def _carton_bilateral_lock_ready(self) -> bool:
        """Whether both sides currently hold the 10 N lock point."""

        if self.task_name != "t10_lift_carton":
            return False
        for side in ("left", "right"):
            if (
                self._contact_sample_readable(side)
                and self._b320_contact_force[side]
                >= _CARTON_BILATERAL_LOCK_FORCE_N
            ):
                self._b320_carton_bilateral_pressure_reached[side] = True
        return self._carton_current_bilateral_pressure_ready()

    @staticmethod
    def _carton_common_translation(
        right_delta: np.ndarray,
        left_delta: np.ndarray,
    ) -> np.ndarray:
        """Keep the shared signed amount on each Cartesian translation axis."""

        right = np.asarray(right_delta, dtype=np.float64).reshape(3)
        left = np.asarray(left_delta, dtype=np.float64).reshape(3)
        same_direction = right * left > 0.0
        return np.where(
            same_direction,
            np.sign(right) * np.minimum(np.abs(right), np.abs(left)),
            0.0,
        )

    def _apply_carton_bilateral_common_motion_guard(
        self,
        *,
        lookahead_seconds: float | None = None,
    ) -> bool:
        """Turn a 10 N bilateral grasp into one rigid, force-aware command.

        The lock remains latched for the episode.  On the latch transition, any
        still-unexecuted target backlog is discarded and the physical palm pair
        becomes the locked command anchor.  Later frames retain only the shared
        signed translation on each axis.  Both palm orientations and both hand
        postures remain at their latch values, so rotation or finger motion cannot
        silently shrink the actual contact-point separation.

        The locked separation is force-corrected rather than merely frozen.  If
        either readable side drops below 9.75 N (or its sample becomes
        unreadable), shared upward Z pauses and only the weak side receives a
        small inward correction toward 10 N.  The corrected separation becomes
        the next command anchor, so lift resumes from the pressure-holding pose
        once both sides recover.

        A shared axis is otherwise rejected for both hands when its normal
        component, or a measured force rise while the carton is still supported
        by the table, predicts more than 10 N.  A genuinely over-10 N measured
        sample rebases both drives to their physical state and unloads the loaded
        side.  A sampled value held at exactly 10 N does not unload merely
        because its recent rise rate is positive; preserving the corrected
        locked separation keeps the grasp force.
        """

        if self.task_name != "t10_lift_carton":
            return False
        previous = self._b320_carton_force_guard_targets
        if set(previous) != {"left", "right"}:
            return False

        requested = {
            "right": self._right_tcp_pose.copy(),
            "left": self._left_tcp_pose.copy(),
        }
        requested_hands = {
            "right": self._right_hand.copy(),
            "left": self._left_hand.copy(),
        }
        requested_before_lock = {
            side: pose.copy() for side, pose in requested.items()
        }
        hands_before_lock = {
            side: hand.copy() for side, hand in requested_hands.items()
        }
        if not self._b320_carton_bilateral_lock_active:
            if not self._carton_bilateral_lock_ready():
                return False
            self._b320_carton_bilateral_lock_active = True
            self._b320_carton_bilateral_lock_latch_step = int(self.env_step)
            physical = {
                side: self._carton_physical_palm_pose(side)
                for side in ("left", "right")
            }
            actual_hands = {
                side: self._carton_actual_hand_target(side)
                for side in ("left", "right")
            }
            self._b320_carton_bilateral_lock_orientation = {
                side: (
                    physical[side][3:7].copy()
                    if physical[side] is not None
                    else previous[side][3:7].copy()
                )
                for side in ("left", "right")
            }
            self._b320_carton_bilateral_lock_hand_targets = {
                side: actual_hands[side].copy()
                for side in ("left", "right")
            }
            if all(pose is not None for pose in physical.values()):
                anchors = {
                    side: physical[side].copy()
                    for side in ("left", "right")
                }
                self._b320_carton_force_guard_targets = {
                    side: anchors[side].copy() for side in ("left", "right")
                }
                self._right_tcp_pose = anchors["right"]
                self._left_tcp_pose = anchors["left"]
                self._right_hand = (
                    self._b320_carton_bilateral_lock_hand_targets["right"]
                    .copy()
                )
                self._left_hand = (
                    self._b320_carton_bilateral_lock_hand_targets["left"]
                    .copy()
                )
                self._b320_carton_bilateral_lock_relative_position = (
                    anchors["left"][:3] - anchors["right"][:3]
                ).copy()
                self._b320_carton_bilateral_lock_common_delta = np.zeros(
                    3,
                    dtype=np.float64,
                )
                self._b320_carton_bilateral_lock_discarded_delta = {
                    side: (
                        requested_before_lock[side][:3] - anchors[side][:3]
                    )
                    for side in ("left", "right")
                }
                position_clamped = any(
                    float(np.linalg.norm(value)) > 1.0e-12
                    for value in (
                        self._b320_carton_bilateral_lock_discarded_delta.values()
                    )
                )
                posture_clamped = any(
                    not np.allclose(
                        requested_before_lock[side][3:7],
                        anchors[side][3:7],
                        rtol=0.0,
                        atol=1.0e-10,
                    )
                    or not np.allclose(
                        hands_before_lock[side],
                        self._b320_carton_bilateral_lock_hand_targets[side],
                        rtol=0.0,
                        atol=1.0e-10,
                    )
                    for side in ("left", "right")
                )
                clamped = bool(position_clamped or posture_clamped)
                if clamped:
                    self._b320_carton_bilateral_lock_clamp_count += 1
                return clamped
            self._b320_carton_bilateral_lock_relative_position = (
                previous["left"][:3] - previous["right"][:3]
            ).copy()

        lookahead_seconds = max(
            float(
                self._control_dt
                if lookahead_seconds is None
                else lookahead_seconds
            ),
            float(self._sim_dt),
        )
        self._b320_carton_force_relief_active = {
            "left": False,
            "right": False,
        }
        relief_sides = {
            side: bool(
                self._contact_sample_readable(side)
                and float(self._b320_contact_force[side])
                > _CARTON_FORCE_VELOCITY_GUARD_N
            )
            for side in ("left", "right")
        }
        pre_relief_deltas = {
            side: requested[side][:3] - previous[side][:3]
            for side in ("left", "right")
        }
        pre_relief_common = self._carton_common_translation(
            pre_relief_deltas["right"],
            pre_relief_deltas["left"],
        )
        shared_lift_requested = bool(pre_relief_common[2] > 1.0e-12)
        self._b320_carton_force_hold_active = {
            side: bool(
                not self._contact_sample_readable(side)
                or float(self._b320_contact_force[side])
                < _CARTON_FORCE_HOLD_MIN_N
            )
            for side in ("left", "right")
        }
        self._b320_carton_force_hold_inward_step_m = {
            "left": 0.0,
            "right": 0.0,
        }
        if any(self._b320_carton_force_hold_active.values()):
            carton_pose = self._physics_body_pose("carton")
            if carton_pose is None:
                carton_pose = self._pose("carton")
            carton_pose = np.asarray(
                carton_pose,
                dtype=np.float64,
            ).reshape(7)
            hold_common = pre_relief_common.copy()
            if hold_common[2] > 1.0e-12:
                hold_common[2] = 0.0
                self._b320_carton_force_hold_lift_block_count += 1

            hold_targets: dict[str, np.ndarray] = {}
            for side in ("left", "right"):
                target = previous[side].copy()
                target[:3] += hold_common
                target[3:7] = (
                    self._b320_carton_bilateral_lock_orientation[side]
                )
                if self._b320_carton_force_hold_active[side]:
                    outward = (
                        self._b320_carton_contact_outward[side].copy()
                        if self._b320_carton_contact_direction_available[side]
                        else self._carton_fallback_outward_direction(
                            side,
                            carton_pose,
                        )
                    )
                    outward_norm = float(np.linalg.norm(outward))
                    if outward_norm > 1.0e-12:
                        outward /= outward_norm
                    measured_force_n = float(
                        self._b320_contact_force[side]
                    )
                    if (
                        not self._contact_sample_readable(side)
                        or not np.isfinite(measured_force_n)
                    ):
                        measured_force_n = 0.0
                    force_error_n = max(
                        0.0,
                        _CARTON_BILATERAL_LOCK_FORCE_N
                        - measured_force_n,
                    )
                    inward_step_m = min(
                        _CARTON_FORCE_HOLD_MAX_INWARD_STEP_M,
                        force_error_n
                        / _CARTON_FORCE_GUARD_MIN_STIFFNESS_N_PER_M,
                    )
                    target[:3] -= outward * inward_step_m
                    self._b320_carton_force_hold_inward_step_m[side] = (
                        inward_step_m
                    )
                    self._b320_carton_force_hold_correction_count[side] += 1
                hold_targets[side] = target

            self._right_tcp_pose = hold_targets["right"]
            self._left_tcp_pose = hold_targets["left"]
            self._right_hand = (
                self._b320_carton_bilateral_lock_hand_targets["right"].copy()
            )
            self._left_hand = (
                self._b320_carton_bilateral_lock_hand_targets["left"].copy()
            )
            self._b320_carton_force_guard_targets = {
                side: hold_targets[side].copy()
                for side in ("left", "right")
            }
            self._b320_carton_bilateral_lock_relative_position = (
                hold_targets["left"][:3] - hold_targets["right"][:3]
            ).copy()
            self._b320_carton_bilateral_lock_common_delta = (
                hold_common.copy()
            )
            self._b320_carton_bilateral_lock_discarded_delta = {
                side: (
                    requested_before_lock[side][:3]
                    - hold_targets[side][:3]
                )
                for side in ("left", "right")
            }
            self._b320_carton_bilateral_lock_force_rejected_axes = (
                np.asarray([False, False, shared_lift_requested], dtype=bool)
            )
            self._b320_carton_bilateral_lock_clamp_count += 1
            return True

        lift_response_risk = {
            side: bool(
                self._contact_sample_readable(side)
                and float(self._b320_contact_force[side])
                + max(
                    0.0,
                    float(self._b320_contact_force_rise_rate_n_s[side]),
                )
                * lookahead_seconds
                > _CARTON_FORCE_VELOCITY_GUARD_N
            )
            for side in ("left", "right")
        }
        if shared_lift_requested and any(lift_response_risk.values()):
            self._b320_carton_bilateral_lift_bypass_count += 1
        if any(relief_sides.values()) and not shared_lift_requested:
            carton_pose = self._physics_body_pose("carton")
            if carton_pose is None:
                carton_pose = self._pose("carton")
            carton_pose = np.asarray(
                carton_pose,
                dtype=np.float64,
            ).reshape(7)
            relief_targets: dict[str, np.ndarray] = {}
            relief_hands: dict[str, np.ndarray] = {}
            for side in ("left", "right"):
                physical_pose = self._carton_physical_palm_pose(side)
                anchor = (
                    physical_pose
                    if physical_pose is not None
                    else previous[side].copy()
                )
                if relief_sides[side]:
                    outward = (
                        self._b320_carton_contact_outward[side]
                        if self._b320_carton_contact_direction_available[side]
                        else self._carton_fallback_outward_direction(
                            side,
                            carton_pose,
                        )
                    )
                    anchor[:3] += outward * _CARTON_FORCE_RELIEF_STEP_M
                    self._b320_carton_force_relief_active[side] = True
                    self._b320_carton_force_relief_count[side] += 1
                relief_targets[side] = anchor.copy()
                relief_hands[side] = self._carton_actual_hand_target(side)

            self._right_tcp_pose = relief_targets["right"]
            self._left_tcp_pose = relief_targets["left"]
            self._right_hand = relief_hands["right"]
            self._left_hand = relief_hands["left"]
            self._b320_carton_force_guard_targets = {
                side: relief_targets[side].copy()
                for side in ("left", "right")
            }
            self._b320_carton_bilateral_lock_orientation = {
                side: relief_targets[side][3:7].copy()
                for side in ("left", "right")
            }
            self._b320_carton_bilateral_lock_hand_targets = {
                side: relief_hands[side].copy()
                for side in ("left", "right")
            }
            self._b320_carton_bilateral_lock_relative_position = (
                relief_targets["left"][:3]
                - relief_targets["right"][:3]
            ).copy()
            self._b320_carton_bilateral_lock_common_delta = np.zeros(
                3,
                dtype=np.float64,
            )
            self._b320_carton_bilateral_lock_discarded_delta = {
                side: (
                    requested_before_lock[side][:3]
                    - relief_targets[side][:3]
                )
                for side in ("left", "right")
            }
            self._b320_carton_bilateral_lock_force_rejected_axes = np.zeros(
                3,
                dtype=bool,
            )
            self._b320_carton_bilateral_lock_force_clamp_count += 1
            self._b320_carton_bilateral_lock_clamp_count += 1
            return True

        if set(self._b320_carton_bilateral_lock_hand_targets) != {
            "left",
            "right",
        }:
            self._b320_carton_bilateral_lock_hand_targets = {
                side: requested_hands[side].copy()
                for side in ("left", "right")
            }
        for side in ("left", "right"):
            requested[side][3:7] = (
                self._b320_carton_bilateral_lock_orientation[side]
            )
            requested_hands[side] = (
                self._b320_carton_bilateral_lock_hand_targets[side].copy()
            )
        deltas = {
            side: requested[side][:3] - previous[side][:3]
            for side in ("left", "right")
        }
        common = self._carton_common_translation(
            deltas["right"],
            deltas["left"],
        )
        force_rejected_axes = np.zeros(3, dtype=bool)
        if np.any(np.abs(common) > 1.0e-12):
            carton_pose = self._physics_body_pose("carton")
            if carton_pose is None:
                carton_pose = self._pose("carton")
            carton_pose = np.asarray(
                carton_pose,
                dtype=np.float64,
            ).reshape(7)
            carton_airborne = bool(self._b320_carton_airborne_now)
            for side in ("left", "right"):
                if not self._contact_sample_readable(side):
                    continue
                if self._b320_carton_contact_direction_available[side]:
                    outward = self._b320_carton_contact_outward[side]
                else:
                    outward = self._carton_fallback_outward_direction(
                        side,
                        carton_pose,
                    )
                force_n = float(self._b320_contact_force[side])
                rate_prediction = (
                    max(
                        0.0,
                        float(
                            self._b320_contact_force_rise_rate_n_s[side]
                        ),
                    )
                    * lookahead_seconds
                )
                response_risk = bool(
                    force_n + rate_prediction
                    > _CARTON_FORCE_VELOCITY_GUARD_N
                )
                for axis in range(3):
                    if abs(float(common[axis])) <= 1.0e-12:
                        continue
                    inward_displacement = (
                        0.0
                        if carton_airborne
                        else max(
                            0.0,
                            -float(common[axis] * outward[axis]),
                        )
                    )
                    predicted_force = force_n + max(
                        rate_prediction,
                        inward_displacement
                        * _CARTON_FORCE_GUARD_MIN_STIFFNESS_N_PER_M,
                    )
                    # A positive Z request lifts the table-supported carton and
                    # is the one shared direction that cannot squeeze its side
                    # walls.  Horizontal/downward motion is held while a sampled
                    # response is already heading past 10 N.
                    response_sensitive_axis = axis < 2 or common[axis] < 0.0
                    if (
                        predicted_force
                        > _CARTON_FORCE_VELOCITY_GUARD_N
                        and (
                            inward_displacement > 1.0e-9
                            or (
                                response_risk
                                and response_sensitive_axis
                            )
                        )
                    ):
                        force_rejected_axes[axis] = True
        if np.any(force_rejected_axes):
            common[force_rejected_axes] = 0.0
            self._b320_carton_bilateral_lock_force_clamp_count += 1
        self._b320_carton_bilateral_lock_force_rejected_axes = (
            force_rejected_axes.copy()
        )
        discarded = {
            side: deltas[side] - common for side in ("left", "right")
        }
        position_clamped = any(
            float(np.linalg.norm(discarded[side])) > 1.0e-12
            for side in ("left", "right")
        )
        posture_clamped = any(
            not np.allclose(
                requested_before_lock[side][3:7],
                requested[side][3:7],
                rtol=0.0,
                atol=1.0e-10,
            )
            or not np.allclose(
                hands_before_lock[side],
                requested_hands[side],
                rtol=0.0,
                atol=1.0e-10,
            )
            for side in ("left", "right")
        )
        clamped = bool(position_clamped or posture_clamped)
        for side in ("left", "right"):
            requested[side][:3] = previous[side][:3] + common

        self._right_tcp_pose = requested["right"]
        self._left_tcp_pose = requested["left"]
        self._right_hand = requested_hands["right"]
        self._left_hand = requested_hands["left"]
        self._b320_carton_bilateral_lock_common_delta = common.copy()
        self._b320_carton_bilateral_lock_discarded_delta = {
            side: discarded[side].copy() for side in ("left", "right")
        }
        if clamped:
            self._b320_carton_bilateral_lock_clamp_count += 1
        return clamped

    def _apply_action(self, action) -> None:
        if self.task_name == "t10_lift_carton":
            # Snapshot the command that survived the previous control frame.
            # The incoming frame remains provisional through all 120 Hz physics
            # substeps, so a late force rise can still discard only this frame's
            # newly requested compressive component.
            self._b320_carton_force_guard_targets = {
                "right": self._right_tcp_pose.copy(),
                "left": self._left_tcp_pose.copy(),
            }
            self._b320_carton_force_guard_hand_targets = {
                "right": self._right_hand.copy(),
                "left": self._left_hand.copy(),
            }
        super()._apply_action(action)

    def _apply_operator_action_extras(self, action) -> None:
        super()._apply_operator_action_extras(action)
        if self.task_name == "t10_lift_carton":
            if self._b320_carton_bilateral_lock_active:
                self._apply_carton_bilateral_common_motion_guard()
            else:
                self._apply_carton_prelock_lift_guard()
                self._apply_carton_bilateral_common_motion_guard()

    def _prepare_physics_substep_joint_targets(
        self,
        substep_index: int,
        substep_count: int,
    ) -> str:
        route = super()._prepare_physics_substep_joint_targets(
            substep_index,
            substep_count,
        )
        if self.task_name != "t10_lift_carton":
            return route
        if callable(
            getattr(self, "_handson_pre_physics_substep_callback", None)
        ):
            # Recorded-drive replay and exact live trajectory callbacks own
            # their solver inputs.  Do not alter them after the parent route.
            return route
        prelock_lift_clamped = False
        if not self._b320_carton_bilateral_lock_active:
            prelock_lift_clamped = self._apply_carton_prelock_lift_guard()
        bilateral_clamped = self._apply_carton_bilateral_common_motion_guard(
            lookahead_seconds=float(self._sim_dt),
        )
        clamped = bool(prelock_lift_clamped or bilateral_clamped)
        if clamped and self._launch_isaac:
            # Re-solve immediately so a pre-lock lift request or a bilateral
            # force clamp cannot leave stale IK targets active for the rest of
            # the 30 Hz operator frame.
            self._apply_joint_targets()
            guard_routes = []
            if prelock_lift_clamped:
                guard_routes.append("bc004_prelock_lift_guard")
            if bilateral_clamped:
                guard_routes.append("bc004_bilateral_common_motion_guard")
            return f"{route}+{'+'.join(guard_routes)}"
        return route

    def _carton_instantaneous_state(self) -> dict[str, Any]:
        clearance, pose_readable, tilt = self._carton_bottom_clearance()
        table_readable = bool(
            self._b320_carton_table_contact_available
            and self._b320_carton_table_contact_readable
            and np.isfinite(self._b320_carton_table_contact_force)
        )
        geometry_clear = clearance >= _CARTON_AIRBORNE_CLEARANCE_M
        table_clear = bool(
            table_readable
            and self._b320_carton_table_contact_force
            <= _CARTON_TABLE_CONTACT_MAX_N
        )
        airborne = bool(pose_readable and geometry_clear and table_clear)
        force_ok = bool(
            self._b320_contact_peak["left"] <= _CARTON_FORCE_LIMIT_N
            and self._b320_contact_peak["right"] <= _CARTON_FORCE_LIMIT_N
        )
        support = self._carton_support_valid()
        return {
            "bottom_clearance_m": clearance,
            "pose_readable": pose_readable,
            "table_contact_readable": table_readable,
            "table_contact_force_n": float(
                self._b320_carton_table_contact_force
            ),
            "geometry_clear": geometry_clear,
            "table_clear": table_clear,
            "airborne": airborne,
            "support": support,
            "force_ok": force_ok,
            "tilt_deg": tilt,
            "hold_valid": bool(
                airborne
                and support
                and force_ok
            ),
        }

    def _update_carton_transport_latches(self) -> None:
        if self.task_name != "t10_lift_carton":
            return
        state = self._carton_instantaneous_state()
        self._b320_carton_airborne_now = bool(state["airborne"])
        if self._b320_carton_airborne_now:
            self._b320_carton_transport_started = True
        if not self._b320_carton_transport_started:
            return
        if not state["support"]:
            # Historical diagnostic only.  A bad frame resets the substep
            # timer below, but the operator may recover and start a fresh
            # three-second hold instead of terminating on one transient gap.
            self._b320_carton_transport_contact_lost = True

    def _read_contacts(self, *, update_latches: bool) -> None:
        for side in ("left", "right"):
            view = self._b320_contact_views.get(side)
            available = view is not None
            previous_force = float(self._b320_contact_force[side])
            previous_readable = bool(self._b320_contact_readable[side])
            force, readable = self._read_b320_contact_force_sample(
                view,
                self._sim_dt,
            )
            if readable:
                force = self._carton_recorded_force_n(force)
            self._b320_contact_available[side] = available
            self._b320_contact_readable[side] = readable
            self._b320_contact_force[side] = force
            if readable and np.isfinite(force):
                self._b320_contact_peak[side] = max(
                    self._b320_contact_peak[side], force
                )
            if self.task_name == "t10_lift_carton":
                decay = math.exp(
                    -float(self._sim_dt)
                    / _CARTON_FORCE_GUARD_RATE_MEMORY_SECONDS
                )
                retained_rate = (
                    float(self._b320_contact_force_rise_rate_n_s[side])
                    * decay
                )
                raw_rise_rate = 0.0
                if readable and previous_readable:
                    raw_rise_rate = max(
                        0.0,
                        (float(force) - previous_force)
                        / max(float(self._sim_dt), 1.0e-9),
                    )
                self._b320_contact_force_rise_rate_n_s[side] = max(
                    retained_rate,
                    raw_rise_rate,
                )
                points, point_forces, detail_available = self._read_b320_contact_points(
                    view,
                    self._sim_dt,
                )
                self._b320_carton_contact_detail_available[side] = detail_available
                outward, direction_available = (
                    self._carton_contact_outward_direction(
                        side,
                        points,
                        point_forces,
                    )
                )
                self._b320_carton_contact_outward[side] = outward
                self._b320_carton_contact_direction_available[side] = bool(
                    detail_available and direction_available
                )
                self._b320_carton_contact_lever_available[side] = False
                contact_points = np.asarray(
                    points,
                    dtype=np.float64,
                ).reshape(-1, 3)
                contact_weights = np.abs(
                    np.asarray(point_forces, dtype=np.float64).reshape(-1)
                )
                palm_matrix = self._robot_link_pose_matrix(side)
                if (
                    detail_available
                    and direction_available
                    and palm_matrix is not None
                    and len(contact_points) > 0
                    and len(contact_points) == len(contact_weights)
                ):
                    weight_sum = float(contact_weights.sum())
                    if weight_sum <= 1.0e-9:
                        contact_weights = np.ones_like(contact_weights)
                        weight_sum = float(contact_weights.sum())
                    centroid = np.sum(
                        contact_points * contact_weights[:, None],
                        axis=0,
                    ) / weight_sum
                    palm_matrix = np.asarray(
                        palm_matrix,
                        dtype=np.float64,
                    ).reshape(4, 4)
                    lever_local = palm_matrix[:3, :3].T @ (
                        centroid - palm_matrix[:3, 3]
                    )
                    if np.all(np.isfinite(lever_local)):
                        self._b320_carton_contact_lever_local[side] = (
                            lever_local
                        )
                        self._b320_carton_contact_lever_available[side] = True
                if readable and float(force) >= _HAND_CONTACT_N:
                    physical_position = self._carton_physical_palm_position(
                        side
                    )
                    if (
                        physical_position is not None
                        and (
                            not self._b320_carton_force_guard_verified_valid[
                                side
                            ]
                            or float(force)
                            <= _CARTON_FORCE_GUARD_VERIFY_MAX_N
                        )
                    ):
                        self._b320_carton_force_guard_verified_position[
                            side
                        ] = physical_position
                        self._b320_carton_force_guard_verified_valid[
                            side
                        ] = True
                    if float(force) >= _CARTON_BILATERAL_LOCK_FORCE_N:
                        self._b320_carton_bilateral_pressure_reached[
                            side
                        ] = True
                if (
                    readable
                    and float(force) < _RELEASE_CONTACT_N
                    and not direction_available
                ):
                    self._b320_contact_force_rise_rate_n_s[side] = 0.0
                if readable and float(force) < _RELEASE_CONTACT_N:
                    self._b320_carton_force_guard_verified_valid[side] = False
                zone_valid, zone_force = self._carton_contact_zone_matches(
                    side,
                    points,
                    point_forces,
                )
                self._b320_carton_zone_force[side] = (
                    self._carton_recorded_force_n(zone_force)
                )
                self._b320_carton_zone_valid[side] = bool(
                    detail_available
                    and readable
                    and force >= _HAND_CONTACT_N
                    and zone_valid
                )
        if self.task_name == "t10_lift_carton":
            table_view = self._b320_carton_table_contact_view
            table_force, table_readable = self._read_b320_contact_force_sample(
                table_view,
                self._sim_dt,
            )
            self._b320_carton_table_contact_available = table_view is not None
            self._b320_carton_table_contact_readable = table_readable
            self._b320_carton_table_contact_force = float(table_force)
        elif self.task_name == "t20_pass_and_orient_pen":
            table_view = self._b320_pen_table_contact_view
            table_force, table_readable = self._read_b320_contact_force_sample(
                table_view,
                self._sim_dt,
            )
            self._b320_pen_table_contact_available = table_view is not None
            self._b320_pen_table_contact_readable = table_readable
            self._b320_pen_table_contact_force = float(table_force)
        if not update_latches:
            return
        previous_seen = dict(self._b320_contact_seen)
        for side in ("left", "right"):
            contacting = (
                self._bc011_contacting(side)
                if self.task_name == "t20_pass_and_orient_pen"
                else self._contact_at_least(side, _HAND_CONTACT_N)
            )
            if contacting:
                self._b320_contact_seen[side] = True
            released = (
                not self._bc011_contacting(side)
                if self.task_name == "t20_pass_and_orient_pen"
                else self._b320_contact_force[side] < _RELEASE_CONTACT_N
            )
            if (
                self._b320_contact_seen[side]
                and self._contact_sample_readable(side)
                and released
                and self._b320_release_frame[side] is None
            ):
                self._b320_release_frame[side] = int(self.env_step)

        if self.task_name == "t20_pass_and_orient_pen":
            if (
                not previous_seen["left"]
                and self._contact_sample_readable("left")
                and not self._bc011_contacting("left")
                and self._bc011_contacting("right")
            ):
                self._b320_handover_right_first = True
            if (
                self._b320_handover_right_first
                and self._bc011_contacting("left")
                and self._bc011_contacting("right")
            ):
                self._b320_handover_overlap = True
            if (
                self._b320_handover_overlap
                and self._bc011_contacting("left")
                and self._contact_sample_readable("right")
                and not self._bc011_contacting("right")
            ):
                self._b320_handover_left_only = True

        if self.task_name == "t45_stir_pot":
            if (
                previous_seen["left"]
                and self._contact_sample_readable("left")
                and self._b320_contact_force["left"] < _RELEASE_CONTACT_N
            ):
                self._b320_left_contact_lost = True

        self._update_carton_transport_latches()

    def _ka005_2_physics_dof_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        q = np.zeros(2, dtype=np.float64)
        qd = np.zeros(2, dtype=np.float64)
        for index, name in enumerate(KA005_2_DOF_NAMES):
            entry = self._task_dof_views.get(name)
            if entry is None:
                entry = next(
                    (
                        value
                        for dof_name, value in self._task_dof_views.items()
                        if name in dof_name
                    ),
                    None,
                )
            if entry is None:
                return q, qd, False
            view, dof_index, _initial = entry
            try:
                q[index] = float(
                    _tensor_to_numpy(view.get_dof_positions())[0, int(dof_index)]
                )
                qd[index] = float(
                    _tensor_to_numpy(view.get_dof_velocities())[0, int(dof_index)]
                )
            except Exception:
                return q, qd, False
        return q, qd, bool(np.all(np.isfinite(q)) and np.all(np.isfinite(qd)))

    def _read_ka005_2_physics_truth(
        self,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float], bool, bool]:
        """Read the two DOFs and all four filtered contacts without latching."""

        q, qd, dof_available = self._ka005_2_physics_dof_state()
        all_contacts_available = True
        contact_forces: dict[str, float] = {}
        for key in KA005_2_CONTACT_KEYS:
            view = self._ka005_2_contact_views.get(key)
            force, readable = self._read_b320_contact_force_sample(
                view,
                self._sim_dt,
            )
            contact_forces[key] = float(force)
            self._ka005_2_contact_force_n[key] = float(force)
            self._ka005_2_contact_readable[key] = bool(readable)
            all_contacts_available = bool(
                all_contacts_available and view is not None and readable
            )
        return (
            q,
            qd,
            contact_forces,
            bool(dof_available),
            bool(all_contacts_available),
        )

    def stable_recording_origin_task_readiness(
        self,
    ) -> dict[str, Any] | None:
        """Require continuous KA005-2 DOF/contact truth before action zero."""

        if self.task_name not in B320_DUAL_KNOB_TASK_NAMES:
            return None
        q, qd, _forces, dof_available, contact_available = (
            self._read_ka005_2_physics_truth()
        )
        return {
            "kind": "ka005_2_dof_and_four_contact_truth",
            "ready": bool(dof_available and contact_available),
            "dof_truth_available": bool(dof_available),
            "contact_truth_available": bool(contact_available),
            "dof_names": list(KA005_2_DOF_NAMES),
            "q_rad": [float(value) for value in q],
            "qd_rad_s": [float(value) for value in qd],
            "contact_keys": list(KA005_2_CONTACT_KEYS),
            "contact_view_present": {
                key: key in self._ka005_2_contact_views
                for key in KA005_2_CONTACT_KEYS
            },
            "contact_readable": dict(self._ka005_2_contact_readable),
        }

    def accept_stable_recording_origin(
        self,
        report: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Rebase KA005-2 Python state to the accepted physical boundary."""

        if self.task_name not in B320_DUAL_KNOB_TASK_NAMES:
            return None
        from .recording_origin import RecordingOriginStabilityError

        final_readiness = report.get("final_task_readiness")
        if (
            report.get("task_readiness_required") is not True
            or not isinstance(final_readiness, dict)
            or final_readiness.get("ready") is not True
        ):
            raise RecordingOriginStabilityError(
                "B320-KA005-2 reached the origin boundary without its "
                "continuous two-DOF/four-contact truth proof.",
                diagnostics={
                    "failure_phase": "ka005_2_origin_acceptance",
                    "task_readiness": final_readiness,
                },
            )
        if int(self.env_step) != 0:
            raise RecordingOriginStabilityError(
                "B320-KA005-2 origin acceptance ran after action zero.",
                diagnostics={
                    "failure_phase": "ka005_2_origin_acceptance",
                    "env_step": int(self.env_step),
                },
            )

        q, qd, forces, dof_available, contact_available = (
            self._read_ka005_2_physics_truth()
        )
        if (
            not dof_available
            or not contact_available
            or not np.array_equal(qd, np.zeros(2, dtype=np.float64))
        ):
            raise RecordingOriginStabilityError(
                "B320-KA005-2 final truth changed after the exact-zero "
                "recording boundary.",
                diagnostics={
                    "failure_phase": "ka005_2_origin_acceptance",
                    "dof_truth_available": bool(dof_available),
                    "contact_truth_available": bool(contact_available),
                    "qd_rad_s": [float(value) for value in qd],
                    "contact_readable": dict(
                        self._ka005_2_contact_readable
                    ),
                },
            )

        accepted_poses: dict[str, np.ndarray] = {}
        for name in ("panel_base", "knob_left", "knob_right"):
            pose = self._physics_body_pose(name)
            if pose is None:
                raise RecordingOriginStabilityError(
                    "B320-KA005-2 final fixture pose truth is unavailable: "
                    f"{name}.",
                    diagnostics={
                        "failure_phase": "ka005_2_origin_acceptance",
                        "missing_pose": name,
                    },
                )
            pose = np.asarray(pose, dtype=np.float64).reshape(-1)
            if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                raise RecordingOriginStabilityError(
                    "B320-KA005-2 final fixture pose truth is invalid: "
                    f"{name}.",
                    diagnostics={
                        "failure_phase": "ka005_2_origin_acceptance",
                        "invalid_pose": name,
                    },
                )
            accepted_poses[name] = pose.copy()

        self._b320_reset_poses.update(
            {name: pose.copy() for name, pose in accepted_poses.items()}
        )
        self._poses.update(
            {name: pose.copy() for name, pose in accepted_poses.items()}
        )
        self._ka005_2_state.reset(
            q,
            self._ka005_2_reset_sample.target_rad,
        )
        self._ka005_2_state.prime_recording_origin_truth(q, qd, forces)
        self._ka005_2_recording_origin_accepted = True
        self._ka005_2_recording_origin_q_rad = q.copy()
        self._ka005_2_recording_origin_poses = {
            name: pose.copy() for name, pose in accepted_poses.items()
        }
        self._b320_failure_reason = None
        return {
            "accepted": True,
            "kind": "ka005_2_stable_physical_boundary_rebase",
            "physics_state_writes": False,
            "recording_origin_rad": [float(value) for value in q],
            "baseline_pose_names": list(accepted_poses),
            "dof_truth_available": True,
            "contact_truth_available": True,
        }

    def _update_ka005_2_physics_substep(self) -> None:
        if not self._launch_isaac:
            return
        q, qd, _forces, dof_available, all_contacts_available = (
            self._read_ka005_2_physics_truth()
        )
        for side, key in (("left", "LL"), ("right", "RR")):
            force = self._ka005_2_contact_force_n[key]
            readable = self._ka005_2_contact_readable[key]
            self._b320_contact_available[side] = key in self._ka005_2_contact_views
            self._b320_contact_readable[side] = bool(readable)
            self._b320_contact_force[side] = float(force)
            if readable:
                self._b320_contact_peak[side] = max(
                    self._b320_contact_peak[side],
                    float(force),
                )
            if readable and force >= _HAND_CONTACT_N:
                self._b320_contact_seen[side] = True
        self._ka005_2_state.update(
            q,
            qd,
            self._ka005_2_contact_force_n,
            dof_available=dof_available,
            contact_available=all_contacts_available,
            dt=self._sim_dt,
        )

    def _handson_post_physics_substep_callback(
        self,
        _substep_index: int,
        _substep_count: int,
    ) -> None:
        """Latch contact-critical B320 state at the 120 Hz solver rate."""

        if self.task_name in {
            "t10_lift_carton",
            "t20_pass_and_orient_pen",
        }:
            self._read_contacts(update_latches=True)
        if self.task_name == "t20_pass_and_orient_pen":
            self._update_bc011_drop_latches()
        if self.task_name in B320_DUAL_KNOB_TASK_NAMES:
            self._update_ka005_2_physics_substep()
        if self.task_name == "t10_lift_carton":
            state = self._carton_instantaneous_state()
            self._b320_carton_airborne_now = bool(state["airborne"])
            if state["hold_valid"]:
                self._b320_carton_airborne_substeps = min(
                    self._b320_carton_airborne_substeps + 1,
                    _CARTON_AIRBORNE_REQUIRED_SUBSTEPS,
                )
            else:
                self._b320_carton_airborne_substeps = 0

    @staticmethod
    def _ka020_pump_fraction_from_poses(
        bottle: np.ndarray | None,
        pump: np.ndarray | None,
    ) -> tuple[float, bool]:
        """Read maximal-coordinate pump travel from its two rigid bodies."""

        if bottle is None or pump is None:
            return 0.0, False
        bottle = np.asarray(bottle, dtype=np.float64).reshape(7)
        pump = np.asarray(pump, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(np.concatenate([bottle, pump]))):
            return 0.0, False
        bottle_rotation = _rotation_matrix(bottle[3:7])
        rest_center = bottle[:3] + bottle_rotation @ np.asarray(
            KA020_PUMP_HEAD_CENTER_LOCAL_M,
            dtype=np.float64,
        )
        displacement_local = bottle_rotation.T @ (pump[:3] - rest_center)
        fraction = -float(displacement_local[2]) / max(
            _SOAP_TRAVEL_M,
            1.0e-9,
        )
        return float(np.clip(fraction, 0.0, 1.0)), True

    def _dof_fraction(self, name: str) -> tuple[float, bool]:
        reading = self._physics_dof_position(name)
        if (
            reading is None
            and name == "soap_dispense_joint"
            and self._launch_isaac
        ):
            return self._ka020_pump_fraction_from_poses(
                self._physics_body_pose("soap_bottle"),
                self._physics_body_pose("soap_dispense_pad"),
            )
        if reading is None:
            return 0.0, False
        value = float(reading[0])
        if not math.isfinite(value):
            return 0.0, False
        if name == "soap_dispense_joint":
            travel = _SOAP_TRAVEL_M
        elif name == "RevoluteJoint_faucet_3_right":
            signed_fraction = -value / max(KA021_FAUCET_OPEN_TRAVEL_RAD, 1.0e-9)
            return float(np.clip(signed_fraction, 0.0, 1.0)), True
        else:
            travel = _HINGE_TRAVEL_RAD
        return float(np.clip(abs(value) / max(travel, 1.0e-9), 0.0, 1.0)), True

    def _update_threshold_frames(self, left: float, right: float, threshold: float) -> None:
        for side, value in (("left", left), ("right", right)):
            if value >= threshold and self._b320_threshold_frame[side] is None:
                self._b320_threshold_frame[side] = int(self.env_step)

    def _carton_metrics(self) -> tuple[dict[str, Any], tuple[bool, ...]]:
        state = self._carton_instantaneous_state()
        left_peak = self._carton_recorded_force_n(
            self._b320_contact_peak["left"]
        )
        right_peak = self._carton_recorded_force_n(
            self._b320_contact_peak["right"]
        )
        hold_seconds = min(
            _CARTON_AIRBORNE_HOLD_SECONDS,
            self._b320_carton_airborne_substeps / float(B320_PHYSICS_HZ),
        )
        hold_complete = bool(
            self._b320_carton_airborne_substeps
            >= _CARTON_AIRBORNE_REQUIRED_SUBSTEPS
        )
        flags = (
            bool(state["pose_readable"] and state["table_contact_readable"]),
            bool(state["airborne"]),
            bool(state["support"]),
            bool(state["force_ok"]),
            hold_complete,
        )
        return {
            "bottom_clearance_m": float(state["bottom_clearance_m"]),
            "airborne_clearance_required_m": _CARTON_AIRBORNE_CLEARANCE_M,
            "airborne_now": bool(state["airborne"]),
            "table_contact_readable": bool(state["table_contact_readable"]),
            "table_contact_force_n": float(state["table_contact_force_n"]),
            "airborne_hold_substeps": int(self._b320_carton_airborne_substeps),
            "airborne_hold_substeps_required": _CARTON_AIRBORNE_REQUIRED_SUBSTEPS,
            "airborne_hold_seconds": hold_seconds,
            "airborne_hold_seconds_required": _CARTON_AIRBORNE_HOLD_SECONDS,
            "tilt_deg": float(state["tilt_deg"]),
            "left_side_force_peak_n": left_peak,
            "right_side_force_peak_n": right_peak,
            "side_force_peak_n": max(left_peak, right_peak),
            "left_lower_side_zone": bool(self._b320_carton_zone_valid["left"]),
            "right_lower_side_zone": bool(self._b320_carton_zone_valid["right"]),
            "left_lower_side_force_n": float(
                self._carton_recorded_force_n(
                    self._b320_carton_zone_force["left"]
                )
            ),
            "right_lower_side_force_n": float(
                self._carton_recorded_force_n(
                    self._b320_carton_zone_force["right"]
                )
            ),
            "prelock_lift_clamped_left": bool(
                self._b320_carton_prelock_lift_clamped["left"]
            ),
            "prelock_lift_clamped_right": bool(
                self._b320_carton_prelock_lift_clamped["right"]
            ),
            "prelock_lift_clamp_count_left": int(
                self._b320_carton_prelock_lift_clamp_count["left"]
            ),
            "prelock_lift_clamp_count_right": int(
                self._b320_carton_prelock_lift_clamp_count["right"]
            ),
            "force_velocity_guard_active": bool(
                self._b320_carton_force_guard_active
            ),
            "force_velocity_guard_active_left": bool(
                self._b320_carton_force_guard_active_by_side["left"]
            ),
            "force_velocity_guard_active_right": bool(
                self._b320_carton_force_guard_active_by_side["right"]
            ),
            "force_velocity_guard_clamped_left": bool(
                self._b320_carton_force_guard_clamped["left"]
            ),
            "force_velocity_guard_clamped_right": bool(
                self._b320_carton_force_guard_clamped["right"]
            ),
            "force_velocity_guard_threshold_n": (
                _CARTON_FORCE_VELOCITY_GUARD_N
            ),
            "force_velocity_guard_predicted_left_n": float(
                self._carton_recorded_force_n(
                    self._b320_carton_force_guard_predicted_n["left"]
                )
            ),
            "force_velocity_guard_predicted_right_n": float(
                self._carton_recorded_force_n(
                    self._b320_carton_force_guard_predicted_n["right"]
                )
            ),
            "force_velocity_guard_inward_left_m_s": float(
                self._b320_carton_force_guard_inward_speed_m_s["left"]
            ),
            "force_velocity_guard_inward_right_m_s": float(
                self._b320_carton_force_guard_inward_speed_m_s["right"]
            ),
            "force_velocity_guard_clamp_count_left": int(
                self._b320_carton_force_guard_clamp_count["left"]
            ),
            "force_velocity_guard_clamp_count_right": int(
                self._b320_carton_force_guard_clamp_count["right"]
            ),
            "force_rise_rate_left_n_s": float(
                self._b320_contact_force_rise_rate_n_s["left"]
            ),
            "force_rise_rate_right_n_s": float(
                self._b320_contact_force_rise_rate_n_s["right"]
            ),
            "force_guard_verified_left": bool(
                self._b320_carton_force_guard_verified_valid["left"]
            ),
            "force_guard_verified_right": bool(
                self._b320_carton_force_guard_verified_valid["right"]
            ),
            "force_guard_pending_inward_left_m": float(
                self._b320_carton_force_guard_pending_inward_m["left"]
            ),
            "force_guard_pending_inward_right_m": float(
                self._b320_carton_force_guard_pending_inward_m["right"]
            ),
            "force_relief_active_left": bool(
                self._b320_carton_force_relief_active["left"]
            ),
            "force_relief_active_right": bool(
                self._b320_carton_force_relief_active["right"]
            ),
            "force_relief_count_left": int(
                self._b320_carton_force_relief_count["left"]
            ),
            "force_relief_count_right": int(
                self._b320_carton_force_relief_count["right"]
            ),
            "force_relief_step_m": _CARTON_FORCE_RELIEF_STEP_M,
            "bilateral_pressure_reached_left": bool(
                self._b320_carton_bilateral_pressure_reached["left"]
            ),
            "bilateral_pressure_reached_right": bool(
                self._b320_carton_bilateral_pressure_reached["right"]
            ),
            "bilateral_position_lock_active": bool(
                self._b320_carton_bilateral_lock_active
            ),
            "bilateral_position_lock_force_n": (
                _CARTON_BILATERAL_LOCK_FORCE_N
            ),
            "bilateral_position_lock_latch_step": (
                -1
                if self._b320_carton_bilateral_lock_latch_step is None
                else int(self._b320_carton_bilateral_lock_latch_step)
            ),
            "bilateral_common_translation_x_m": float(
                self._b320_carton_bilateral_lock_common_delta[0]
            ),
            "bilateral_common_translation_y_m": float(
                self._b320_carton_bilateral_lock_common_delta[1]
            ),
            "bilateral_common_translation_z_m": float(
                self._b320_carton_bilateral_lock_common_delta[2]
            ),
            "bilateral_discarded_left_m": float(
                np.linalg.norm(
                    self._b320_carton_bilateral_lock_discarded_delta["left"]
                )
            ),
            "bilateral_discarded_right_m": float(
                np.linalg.norm(
                    self._b320_carton_bilateral_lock_discarded_delta["right"]
                )
            ),
            "bilateral_relative_position_error_m": float(
                np.linalg.norm(
                    (self._left_tcp_pose[:3] - self._right_tcp_pose[:3])
                    - self._b320_carton_bilateral_lock_relative_position
                )
                if self._b320_carton_bilateral_lock_active
                else 0.0
            ),
            "bilateral_position_lock_clamp_count": int(
                self._b320_carton_bilateral_lock_clamp_count
            ),
            "bilateral_force_rejected_x": bool(
                self._b320_carton_bilateral_lock_force_rejected_axes[0]
            ),
            "bilateral_force_rejected_y": bool(
                self._b320_carton_bilateral_lock_force_rejected_axes[1]
            ),
            "bilateral_force_rejected_z": bool(
                self._b320_carton_bilateral_lock_force_rejected_axes[2]
            ),
            "bilateral_force_clamp_count": int(
                self._b320_carton_bilateral_lock_force_clamp_count
            ),
            "bilateral_shared_lift_bypass_count": int(
                self._b320_carton_bilateral_lift_bypass_count
            ),
            "bilateral_force_hold_target_n": (
                _CARTON_BILATERAL_LOCK_FORCE_N
            ),
            "bilateral_force_hold_min_n": _CARTON_FORCE_HOLD_MIN_N,
            "bilateral_force_hold_active_left": bool(
                self._b320_carton_force_hold_active["left"]
            ),
            "bilateral_force_hold_active_right": bool(
                self._b320_carton_force_hold_active["right"]
            ),
            "bilateral_force_hold_inward_step_left_m": float(
                self._b320_carton_force_hold_inward_step_m["left"]
            ),
            "bilateral_force_hold_inward_step_right_m": float(
                self._b320_carton_force_hold_inward_step_m["right"]
            ),
            "bilateral_force_hold_correction_count_left": int(
                self._b320_carton_force_hold_correction_count["left"]
            ),
            "bilateral_force_hold_correction_count_right": int(
                self._b320_carton_force_hold_correction_count["right"]
            ),
            "bilateral_force_hold_lift_block_count": int(
                self._b320_carton_force_hold_lift_block_count
            ),
            "transport_started": bool(self._b320_carton_transport_started),
            "transport_support_lost": bool(
                self._b320_carton_transport_contact_lost
            ),
        }, flags

    def _handover_metrics(self) -> tuple[dict[str, Any], tuple[bool, ...]]:
        pose = self._pose("pen")
        velocity, velocity_readable = self._velocity_with_validity("pen")
        tip_axis_local = BC011_PEN_TIP_AXIS_LOCAL.get(self._bc011_pen_variant)
        tip_axis_readable = tip_axis_local is not None
        tip_up_error = (
            _angle_degrees(
                _axis_world(pose, tip_axis_local),
                (0.0, 0.0, 1.0),
            )
            if tip_axis_readable
            else 180.0
        )
        lift_height = self._bc011_pen_lift_height()
        table_clear = bool(
            self._b320_pen_table_contact_available
            and self._b320_pen_table_contact_readable
            and np.isfinite(self._b320_pen_table_contact_force)
            and self._b320_pen_table_contact_force <= _BC011_TABLE_CONTACT_MAX_N
        )
        left_holding_tip_up = bool(
            self._b320_handover_left_only
            and self._bc011_contacting("left")
            and self._contact_sample_readable("right")
            and not self._bc011_contacting("right")
            and lift_height >= _BC011_PICKUP_LIFT_M
            and table_clear
            and tip_axis_readable
            and tip_up_error <= _BC011_TIP_UP_TOLERANCE_DEG
        )
        flags = (
            self._b320_handover_right_first
            and self._contact_sample_readable("left")
            and self._contact_sample_readable("right"),
            self._b320_handover_overlap
            and self._contact_sample_readable("left")
            and self._contact_sample_readable("right"),
            self._b320_handover_left_only
            and self._contact_sample_readable("left")
            and self._contact_sample_readable("right"),
            left_holding_tip_up,
        )
        return {
            "tip_up_error_deg": tip_up_error,
            # Elevation is measured from horizontal so the requested upright
            # pen pose reads +90 degrees in the collection guidance panel.
            "tip_elevation_deg": 90.0 - tip_up_error,
            "tip_axis_local": tuple(tip_axis_local or (0.0, 0.0, 0.0)),
            "tip_axis_readable": tip_axis_readable,
            "lift_height_m": lift_height,
            "velocity_readable": velocity_readable,
            "table_contact_readable": self._b320_pen_table_contact_readable,
            "table_contact_force_n": self._b320_pen_table_contact_force,
            "linear_speed_mps": float(np.linalg.norm(velocity[:3])),
            "angular_speed_radps": float(np.linalg.norm(velocity[3:])),
        }, flags

    def _seasoning_projection(self, role: str) -> dict[str, Any]:
        return _bc014_flow_projection(
            self._pose(role),
            tuple(float(value) for value in self._bc014_goal_centers_xy[role]),
        )

    def _seasoning_metrics(
        self,
        *,
        update: bool = False,
    ) -> tuple[dict[str, Any], tuple[bool, ...]]:
        """Evaluate BC014 from the two cap-directed proxy streams.

        The physical jars still establish hand assignment and cap pose.  Salt
        and pepper particles are deliberately not simulated: a downward ray
        from each cap is both the success signal and the transform source for
        its render-only D09-style water-flow cylinder.
        """

        role_by_side = {"left": "salt_jar", "right": "pepper_jar"}
        samples: dict[str, dict[str, Any]] = {}
        for side, role in role_by_side.items():
            sample = self._seasoning_projection(role)
            contact_now = bool(
                self._contact_sample_readable(side)
                and self._contact_at_least(side, _HAND_CONTACT_N)
            )
            sample["contact_now"] = contact_now
            sample["emitting"] = bool(contact_now and sample["ray_valid"])
            sample["hits_target"] = bool(
                contact_now and sample["projects_into_zone"]
            )
            samples[role] = sample

        if update:
            dt = float(self._control_dt)
            epsilon = 1.0e-9
            for side, role in role_by_side.items():
                if samples[role]["emitting"]:
                    self._bc014_flow_pour_seconds[side] += dt
                if samples[role]["hits_target"]:
                    self._bc014_flow_hit_seconds[side] += dt
                if (
                    self._bc014_flow_hit_seconds[side] + epsilon
                    >= _BC014_FLOW_REQUIRED_SECONDS
                ):
                    self._bc014_flow_hit_latched[side] = True

            both_hit_now = bool(
                all(samples[role]["hits_target"] for role in role_by_side.values())
            )
            if both_hit_now:
                self._bc014_flow_overlap_seconds += dt
            elif not self._bc014_flow_overlap_latched:
                # Synchronization is a continuous overlap window. Individual
                # seasoning time remains cumulative so a brief hand correction
                # does not erase already delivered material.
                self._bc014_flow_overlap_seconds = 0.0
            if (
                self._bc014_flow_overlap_seconds + epsilon
                >= _BC014_FLOW_OVERLAP_REQUIRED_SECONDS
            ):
                self._bc014_flow_overlap_latched = True
            self._update_bc014_flow_visuals(samples)

        assigned_contacts = bool(
            all(
                self._b320_contact_seen[side]
                and self._contact_sample_readable(side)
                for side in role_by_side
            )
        )
        salt = samples["salt_jar"]
        pepper = samples["pepper_jar"]
        flags = (
            assigned_contacts,
            bool(self._bc014_flow_hit_latched["left"]),
            bool(self._bc014_flow_hit_latched["right"]),
            bool(self._bc014_flow_overlap_latched),
        )
        return {
            "assigned_hand_contacts": assigned_contacts,
            "salt_stream_emitting": bool(salt["emitting"]),
            "pepper_stream_emitting": bool(pepper["emitting"]),
            "salt_stream_hits_target": bool(salt["hits_target"]),
            "pepper_stream_hits_target": bool(pepper["hits_target"]),
            "both_streams_hit_now": bool(
                salt["hits_target"] and pepper["hits_target"]
            ),
            "salt_cap_tilt_deg": float(salt["tilt_deg"]),
            "pepper_cap_tilt_deg": float(pepper["tilt_deg"]),
            "salt_impact_x_m": float(salt["impact_position"][0]),
            "salt_impact_y_m": float(salt["impact_position"][1]),
            "pepper_impact_x_m": float(pepper["impact_position"][0]),
            "pepper_impact_y_m": float(pepper["impact_position"][1]),
            "salt_zone_margin_m": float(salt["zone_margin_m"]),
            "pepper_zone_margin_m": float(pepper["zone_margin_m"]),
            "salt_hit_seconds": float(self._bc014_flow_hit_seconds["left"]),
            "pepper_hit_seconds": float(self._bc014_flow_hit_seconds["right"]),
            "overlap_hit_seconds": float(self._bc014_flow_overlap_seconds),
            "salt_pour_seconds": float(
                self._bc014_flow_pour_seconds["left"]
            ),
            "pepper_pour_seconds": float(
                self._bc014_flow_pour_seconds["right"]
            ),
            "maximum_pour_seconds": _BC014_FLOW_MAX_POUR_SECONDS,
            "salt_goal_x_m": float(self._bc014_goal_centers_xy["salt_jar"][0]),
            "salt_goal_y_m": float(self._bc014_goal_centers_xy["salt_jar"][1]),
            "pepper_goal_x_m": float(
                self._bc014_goal_centers_xy["pepper_jar"][0]
            ),
            "pepper_goal_y_m": float(
                self._bc014_goal_centers_xy["pepper_jar"][1]
            ),
            "salt_hit_latched": bool(self._bc014_flow_hit_latched["left"]),
            "pepper_hit_latched": bool(self._bc014_flow_hit_latched["right"]),
            "overlap_hit_latched": bool(self._bc014_flow_overlap_latched),
        }, flags

    def _double_cabinet_metrics(self) -> tuple[dict[str, Any], tuple[bool, ...]]:
        names = ("target_joint_revolute", "joint_11")
        left, left_available = self._dof_fraction(names[0])
        right, right_available = self._dof_fraction(names[1])
        threshold = 0.80
        seconds = 1.0
        self._update_threshold_frames(left, right, threshold)
        crossing_ok = bool(
            all(value is not None for value in self._b320_threshold_frame.values())
            and abs(int(self._b320_threshold_frame["left"]) - int(self._b320_threshold_frame["right"]))
            <= round(seconds * B320_CONTROL_HZ)
        )
        release_frames = self._b320_release_frame
        threshold_frames = self._b320_threshold_frame
        release_complete = bool(
            all(value is not None for value in release_frames.values())
            and all(value is not None for value in threshold_frames.values())
        )
        release_after_threshold = bool(
            release_complete
            and all(
                int(release_frames[side]) >= int(threshold_frames[side])
                for side in ("left", "right")
            )
        )
        # Keep the observation/info contract finite before both release events
        # exist.  The sentinel is deliberately larger than the one-second
        # acceptance window, so it cannot accidentally satisfy ``release_ok``;
        # callers can distinguish it from a measured delta via the release-frame
        # fields and ``release_after_threshold`` flag.
        release_delta_s = (
            abs(
                int(release_frames["left"])
                - int(release_frames["right"])
            )
            / B320_CONTROL_HZ
            if release_complete
            else _UNAVAILABLE_ARTICULATION_DELTA_S
        )
        release_ok = bool(
            release_after_threshold and release_delta_s <= seconds
        )
        contacts = bool(
            all(self._b320_contact_available.values())
            and all(self._b320_contact_readable.values())
            and all(self._b320_contact_seen.values())
        )
        flags = (
            contacts,
            left >= threshold and left_available,
            right >= threshold and right_available,
            crossing_ok,
            release_ok,
            left >= threshold and right >= threshold,
        )
        return {
            "left_fraction": left,
            "right_fraction": right,
            "threshold_delta_s": (
                abs(int(self._b320_threshold_frame["left"]) - int(self._b320_threshold_frame["right"]))
                / B320_CONTROL_HZ
                if all(value is not None for value in self._b320_threshold_frame.values())
                else _UNAVAILABLE_ARTICULATION_DELTA_S
            ),
            "left_release_frame": release_frames["left"],
            "right_release_frame": release_frames["right"],
            "release_delta_s": release_delta_s,
            "release_after_threshold": release_after_threshold,
            "release_ok": release_ok,
        }, flags

    def _dual_knob_metrics(self) -> tuple[dict[str, Any], tuple[bool, ...]]:
        return self._ka005_2_state.metrics(), self._ka005_2_state.condition_flags()

    def _ka005_2_base_stability(self) -> tuple[float, float, bool]:
        if self._launch_isaac:
            # Do not route formal live validity through ``_pose``: that helper
            # intentionally substitutes a declared fallback for unreadable or
            # non-finite state, which could make missing PhysX truth look like
            # a perfectly stable fixed base.
            live_base = self._physics_body_pose("panel_base")
            reset_base = self._b320_reset_poses.get("panel_base")
            if live_base is None or reset_base is None:
                return 1.0, 180.0, False
            base = np.asarray(live_base, dtype=np.float64).reshape(-1)
            reset_base = np.asarray(reset_base, dtype=np.float64).reshape(-1)
            if base.shape != (7,) or reset_base.shape != (7,):
                return 1.0, 180.0, False
        else:
            # Interface-only tests and tooling intentionally expose injectable
            # poses because there is no native PhysX view in that mode.
            base = np.asarray(
                self._poses.get(
                    "panel_base",
                    self._fallback_b320_pose("panel_base"),
                ),
                dtype=np.float64,
            ).reshape(-1)
            reset_base = np.asarray(
                self._b320_reset_poses.get(
                    "panel_base",
                    self._fallback_b320_pose("panel_base"),
                ),
                dtype=np.float64,
            ).reshape(-1)
            if base.shape != (7,) or reset_base.shape != (7,):
                return 1.0, 180.0, False
        if not np.all(np.isfinite(base)) or not np.all(np.isfinite(reset_base)):
            return 1.0, 180.0, False
        translation_m = float(np.linalg.norm(base[:3] - reset_base[:3]))
        rotation_deg = _quaternion_error_degrees(base[3:7], reset_base[3:7])
        return translation_m, rotation_deg, bool(
            translation_m <= 0.002 and rotation_deg <= 1.0
        )

    def _ka005_2_terminal_failure_reason(self) -> str | None:
        """Return the episode's first physical failure as a stable latch."""

        if self._b320_failure_reason:
            return self._b320_failure_reason
        reason = self._ka005_2_state.failure_reason or None
        if reason is not None:
            return reason
        _base_translation_m, _base_rotation_deg, base_stable = (
            self._ka005_2_base_stability()
        )
        return None if base_stable else "panel_base_drift"

    def _ka005_2_terminal_observation(self) -> np.ndarray:
        reason = self._ka005_2_terminal_failure_reason()
        reason_code = (
            0
            if reason is None
            else _KA005_2_FAILURE_REASON_CODES.get(
                reason,
                _KA005_2_OTHER_FAILURE_REASON_CODE,
            )
        )
        return np.asarray(
            [
                float(self._ka005_2_state.success),
                float(reason is not None),
                float(reason_code),
            ],
            dtype=np.float64,
        )

    def _stir_metrics(
        self,
        *,
        update: bool = True,
    ) -> tuple[dict[str, Any], tuple[bool, ...]]:
        oven = self._pose("dutch_oven")
        spoon = self._pose("spoon")
        reset = self._b320_reset_poses.get("dutch_oven", oven)
        spoon_head = spoon[:3] + _rotation_matrix(spoon[3:7]) @ np.asarray([0.0, 0.0, -0.10])
        relative = spoon_head - oven[:3]
        radius = float(np.linalg.norm(relative[:2]))
        angle = float(math.atan2(relative[1], relative[0]))
        valid_circle = bool(0.025 <= radius <= 0.12 and abs(relative[2]) <= 0.16)
        both_hands_actively_own_roles = all(
            self._contact_at_least(side, _HAND_CONTACT_N)
            for side in ("left", "right")
        )
        if (
            update
            and self._b320_stir_anchor_pose is None
            and valid_circle
            and both_hands_actively_own_roles
        ):
            # Grasping a heavy dynamic pot can nudge it before the operation
            # begins.  Anchor the 25 mm stability contract at the first
            # physically valid stirring sample so approach/grasp setup does
            # not invalidate an otherwise stable stirring trajectory.
            self._b320_stir_anchor_pose = oven.copy()
            self._b320_stir_anchor_frame = int(self.env_step)
        drift_reference = (
            self._b320_stir_anchor_pose
            if self._b320_stir_anchor_pose is not None
            else reset
        )
        drift = float(np.linalg.norm(oven[:3] - drift_reference[:3]))
        reset_drift = float(np.linalg.norm(oven[:3] - reset[:3]))
        if valid_circle and all(
            self._b320_contact_seen[side]
            and self._contact_sample_readable(side)
            for side in ("left", "right")
        ):
            if self._b320_stir_previous_angle is not None:
                delta = (angle - self._b320_stir_previous_angle + math.pi) % (2.0 * math.pi) - math.pi
                if abs(delta) <= math.radians(35.0):
                    self._b320_stir_accumulated_rad += delta
            self._b320_stir_previous_angle = angle
            self._b320_stir_sectors.add(int(((angle + math.pi) / (2.0 * math.pi)) * 6.0) % 6)
        accumulated_deg = abs(math.degrees(self._b320_stir_accumulated_rad))
        left_continuous = bool(
            self._contact_sample_readable("left")
            and self._b320_contact_seen["left"]
            and not self._b320_left_contact_lost
            and self._contact_at_least("left", _HAND_CONTACT_N)
        )
        flags = (
            left_continuous,
            self._contact_sample_readable("right")
            and self._b320_contact_seen["right"],
            len(self._b320_stir_sectors) >= 6,
            accumulated_deg > 650.0,
            self._b320_stir_anchor_pose is not None and drift < 0.025,
        )
        return {
            "pot_drift_m": drift,
            "spoon_radius_m": radius,
            "stir_accumulated_deg": accumulated_deg,
            "stir_checkpoint_count": len(self._b320_stir_sectors),
            "pot_reset_drift_m": reset_drift,
            "stir_anchor_set": self._b320_stir_anchor_pose is not None,
            "stir_anchor_frame": (
                -1
                if self._b320_stir_anchor_frame is None
                else int(self._b320_stir_anchor_frame)
            ),
            "stir_anchor_x_m": float(
                0.0
                if self._b320_stir_anchor_pose is None
                else self._b320_stir_anchor_pose[0]
            ),
            "stir_anchor_y_m": float(
                0.0
                if self._b320_stir_anchor_pose is None
                else self._b320_stir_anchor_pose[1]
            ),
            "stir_anchor_z_m": float(
                0.0
                if self._b320_stir_anchor_pose is None
                else self._b320_stir_anchor_pose[2]
            ),
        }, flags

    def _soap_metrics(self) -> tuple[dict[str, Any], tuple[bool, ...]]:
        sponge = self._pose("sponge")
        pump = self._pose("soap_dispense_pad")
        sponge_velocity, sponge_velocity_readable = self._velocity_with_validity(
            "sponge"
        )
        sponge_linear_speed = float(np.linalg.norm(sponge_velocity[:3]))
        sponge_angular_speed = float(np.linalg.norm(sponge_velocity[3:]))

        pump_rotation = _rotation_matrix(pump[3:7])
        nozzle = pump[:3] + pump_rotation @ np.asarray(
            KA020_PUMP_OUTLET_OFFSET_M,
            dtype=np.float64,
        )
        sponge_rotation = _rotation_matrix(sponge[3:7])
        nozzle_delta_local = sponge_rotation.T @ (nozzle - sponge[:3])
        projection = np.asarray(
            [nozzle_delta_local[0], nozzle_delta_local[2]],
            dtype=np.float64,
        )
        usable_half_extents = np.asarray(
            KA020_SPONGE_PLAN_HALF_EXTENTS_M,
            dtype=np.float64,
        ) - _KA020_SPONGE_EDGE_MARGIN_M
        projection_inside = bool(
            np.all(np.abs(projection) <= usable_half_extents)
        )
        height = float(nozzle_delta_local[1] - KA020_SPONGE_THICKNESS_M)
        dispense, dispense_available = self._dof_fraction("soap_dispense_joint")
        if self._launch_isaac:
            pump_force, pump_readable = self._read_b320_contact_force_sample(
                self._ka020_pump_contact_view,
                self._sim_dt,
            )
            self._ka020_pump_contact_force_n = float(pump_force)
            self._ka020_pump_contact_readable = bool(pump_readable)
        pump_contact = bool(
            self._ka020_pump_contact_readable
            and math.isfinite(self._ka020_pump_contact_force_n)
            and self._ka020_pump_contact_force_n >= _KA020_PUMP_CONTACT_N
        )
        height_valid = bool(
            _KA020_NOZZLE_MIN_HEIGHT_M
            <= height
            <= _KA020_NOZZLE_MAX_HEIGHT_M
        )
        dispensed = bool(dispense_available and dispense >= 0.75)
        impact_local = np.asarray(
            [
                float(np.clip(projection[0], -usable_half_extents[0], usable_half_extents[0])),
                KA020_SPONGE_THICKNESS_M,
                float(np.clip(projection[1], -usable_half_extents[1], usable_half_extents[1])),
            ],
            dtype=np.float64,
        )
        impact = sponge[:3] + sponge_rotation @ impact_local
        flow_sample = {
            "emitting": bool(dispensed and projection_inside and height_valid),
            "outlet_position": nozzle.tolist(),
            "impact_position": impact.tolist(),
        }
        self._ka020_last_flow_sample = flow_sample
        try:
            self._set_ka020_flow_visual(flow_sample)
        except Exception as exc:
            if not self._ka020_flow_visual_warning_emitted:
                self._add_task_state_warning(
                    "Could not update KA020 soap-flow visual: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._ka020_flow_visual_warning_emitted = True
        flags = (
            self._contact_at_least("right", _HAND_CONTACT_N),
            pump_contact,
            projection_inside,
            height_valid,
            dispensed,
        )
        return {
            "sponge_linear_speed_m_s": sponge_linear_speed,
            "sponge_angular_speed_rad_s": sponge_angular_speed,
            "sponge_velocity_readable": sponge_velocity_readable,
            "nozzle_projection_error_m": float(np.linalg.norm(projection)),
            "nozzle_height_m": height,
            "pump_contact_force_n": float(self._ka020_pump_contact_force_n),
            "pump_contact_readable": bool(self._ka020_pump_contact_readable),
            "dispense_fraction": dispense,
            "soap_flow_visible": bool(flow_sample["emitting"]),
        }, flags

    def _faucet_metrics(
        self,
        *,
        update: bool = True,
    ) -> tuple[dict[str, Any], tuple[bool, ...]]:
        cup = self._pose("ceramic_mug")
        joint_name = "RevoluteJoint_faucet_3_right"
        fraction, available = self._dof_fraction(joint_name)
        joint_reading = self._physics_dof_position(joint_name)
        joint_position_rad = (
            float(joint_reading[0])
            if joint_reading is not None
            and math.isfinite(float(joint_reading[0]))
            else 0.0
        )
        flow = _ka021_flow_projection(cup, fraction if available else 0.0)
        self._ka021_last_flow_sample = flow
        try:
            self._set_ka021_flow_visual(flow)
        except Exception as exc:
            if not self._ka021_flow_visual_warning_emitted:
                self._add_task_state_warning(
                    "Could not update KA021 faucet water-flow proxy: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._ka021_flow_visual_warning_emitted = True

        left_live = bool(
            self._contact_sample_readable("left")
            and self._contact_at_least("left", _HAND_CONTACT_N)
        )
        right_live = bool(
            self._contact_sample_readable("right")
            and self._contact_at_least("right", _HAND_CONTACT_N)
        )
        joint_open = bool(
            available and fraction >= KA021_SUCCESS_OPEN_FRACTION
        )
        simultaneous = bool(
            left_live
            and right_live
            and flow["opening_aligned"]
            and flow["cup_upright"]
            and flow["height_valid"]
            and joint_open
            and flow["hits_cup"]
        )
        fill_valid = simultaneous
        if self._launch_isaac:
            # ``_metrics(update=False)`` is also called by the UI and
            # observation builder.  Count a fill frame only once per control
            # step so a single physical frame cannot be double-counted.
            if update and self._ka021_fill_last_update_step != self.env_step:
                if fill_valid:
                    self._ka021_fill_hold_frames = min(
                        self._ka021_fill_hold_frames + 1,
                        self._success_steps_required(),
                    )
                else:
                    self._ka021_fill_hold_frames = 0
                if (
                    self._ka021_fill_hold_frames
                    >= self._success_steps_required()
                ):
                    self._ka021_water_filled = True
                self._ka021_fill_last_update_step = int(self.env_step)
            park = _ka021_cup_parking_projection(
                cup,
                self._ka021_cup_park_position_w,
            )
            faucet_closed = bool(available and fraction <= 0.05)
            post_fill_complete = bool(
                self._ka021_water_filled
                and park["cup_parked"]
                and faucet_closed
            )
            if self._ka021_water_filled:
                # The fill latch is deliberately independent of the terminal
                # placement actions.  Once the open-faucet stream has been
                # held for 0.5 s, moving the mug away does not erase that
                # history; parking and closing are reported as two separate
                # terminal conditions below.
                flags = (
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    bool(park["cup_parked"]),
                    bool(faucet_closed),
                )
            else:
                flags = (
                    left_live,
                    right_live,
                    bool(flow["opening_aligned"]),
                    bool(flow["cup_upright"]),
                    bool(flow["height_valid"]),
                    False,
                    False,
                    False,
                )
        else:
            # Kinematic/unit-test mode has no physical stream clock or sink
            # support body, but still advances the same 30 Hz fill latch so
            # the 0.5-second open-faucet requirement is not bypassed.
            park = _ka021_cup_parking_projection(
                cup,
                self._ka021_cup_park_position_w,
            )
            faucet_closed = bool(available and fraction <= 0.05)
            if not self._ka021_water_filled and update:
                if simultaneous:
                    self._ka021_fill_hold_frames = min(
                        self._ka021_fill_hold_frames + 1,
                        self._success_steps_required(),
                    )
                else:
                    self._ka021_fill_hold_frames = 0
                if (
                    self._ka021_fill_hold_frames
                    >= self._success_steps_required()
                ):
                    self._ka021_water_filled = True
            post_fill_complete = bool(
                self._ka021_water_filled
                and park["cup_parked"]
                and faucet_closed
            )
            if self._ka021_water_filled:
                # Match the real-runtime latch semantics: once the open-
                # faucet stream has been valid for the required 0.5 s, the
                # six fill-phase conditions stay satisfied while the mug is
                # moved to the outside-sink pad and the faucet is closed.
                flags = (
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    bool(park["cup_parked"]),
                    bool(faucet_closed),
                )
            else:
                flags = (
                    left_live,
                    right_live,
                    bool(flow["opening_aligned"]),
                    bool(flow["cup_upright"]),
                    bool(flow["height_valid"]),
                    bool(self._ka021_water_filled),
                    bool(park["cup_parked"]),
                    bool(faucet_closed),
                )
        return {
            "cup_outlet_xy_error_m": float(flow["opening_xy_error_m"]),
            "cup_rim_position_w": [
                float(value) for value in flow["cup_rim_position"]
            ],
            "cup_tilt_deg": float(flow["cup_tilt_deg"]),
            "cup_rim_z_m": float(flow["cup_rim_z_m"]),
            "outlet_rim_gap_m": float(flow["outlet_rim_gap_m"]),
            "faucet_open_fraction": fraction,
            "faucet_joint_readable": bool(available),
            "faucet_joint_position_rad": joint_position_rad,
            "faucet_joint_position_deg": math.degrees(joint_position_rad),
            "water_emitting": bool(flow["emitting"]),
            "water_hits_cup": bool(flow["hits_cup"]),
            "ka021_fill_phase_valid": bool(fill_valid),
            "ka021_fill_phase_complete": bool(self._ka021_water_filled),
            "ka021_fill_hold_frames": int(self._ka021_fill_hold_frames),
            "ka021_water_filled": bool(self._ka021_water_filled),
            "cup_park_xy_error_m": float(park["cup_park_xy_error_m"]),
            "cup_park_z_error_m": float(park["cup_park_z_error_m"]),
            "cup_park_tilt_deg": float(park["cup_park_tilt_deg"]),
            "cup_parked": bool(park["cup_parked"]),
            "ka021_cup_park_condition": bool(park["cup_parked"]),
            "faucet_closed": faucet_closed,
            "ka021_faucet_close_condition": bool(faucet_closed),
            "post_fill_complete": post_fill_complete,
        }, flags

    def _metrics(self, *, update: bool) -> dict[str, Any]:
        if update:
            self._read_contacts(update_latches=True)
        task = self.task_name
        if task == "t10_lift_carton":
            values, flags = self._carton_metrics()
        elif task == "t20_pass_and_orient_pen":
            values, flags = self._handover_metrics()
        elif task == "t30_pour_seasonings":
            values, flags = self._seasoning_metrics(update=update)
        elif task == "t01_align_blocks_in_midair":
            raise RuntimeError(
                "B320-BC021 must use HandsOnB320BC021IsaacLabEnv"
            )
        elif task == B320_BC017_TASK_NAME:
            raise RuntimeError(
                "B320-BC017 must use HandsOnB320BC017IsaacLabEnv"
            )
        elif task == "t45_stir_pot":
            values, flags = self._stir_metrics(update=update)
        elif task == B320_KA020_TASK_NAME:
            values, flags = self._soap_metrics()
        elif task == "t05_fill_cup":
            values, flags = self._faucet_metrics(update=update)
        elif task == "b320_synchronized_double_cabinet_open":
            values, flags = self._double_cabinet_metrics()
        elif task in B320_DUAL_KNOB_TASK_NAMES:
            values, flags = self._dual_knob_metrics()
        else:
            raise RuntimeError(f"Unhandled rigid B320 task {task}")
        self._b320_condition_flags = tuple(bool(value) for value in flags)
        complete_now = bool(all(self._b320_condition_flags))
        if (
            task == "t20_pass_and_orient_pen"
            and update
            and complete_now
        ):
            self._b320_handover_completed = True
        values = dict(values)
        values["success_now"] = bool(
            complete_now
            or (
                task == "t20_pass_and_orient_pen"
                and self._b320_handover_completed
            )
        )
        self._b320_last_metrics = values
        return values

    def _build_observation_space(self) -> spaces.Dict:
        state_spaces = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "gripper_pose": spaces.Box(-np.inf, np.inf, shape=(32,), dtype=np.float64),
            "left_role_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "right_role_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "fixture_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "goal_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "task_metrics": spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float64),
            "contact_metrics": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float64),
            "stage_state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float64),
            "articulation_q": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64),
            "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
        }
        if self.task_name in B320_DUAL_KNOB_TASK_NAMES:
            state_spaces.update(
                {
                    "knob_joint_state": spaces.Box(
                        -np.inf, np.inf, shape=(4,), dtype=np.float64
                    ),
                    "knob_target_state": spaces.Box(
                        -np.inf, np.inf, shape=(6,), dtype=np.float64
                    ),
                    "knob_contact_state": spaces.Box(
                        -np.inf, np.inf, shape=(8,), dtype=np.float64
                    ),
                    "task_validity": spaces.Box(
                        0.0, 1.0, shape=(4,), dtype=np.float64
                    ),
                    "sync_task_state": spaces.Box(
                        -np.inf, np.inf, shape=(20,), dtype=np.float64
                    ),
                    "task_terminal_state": spaces.Box(
                        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
                        np.asarray([1.0, 1.0, 5.0], dtype=np.float64),
                        dtype=np.float64,
                    ),
                }
            )
        result: dict[str, spaces.Space] = {"state": spaces.Dict(state_spaces)}
        if self.image_obs:
            head = spaces.Box(0, 255, shape=self._image_shape, dtype=np.uint8)
            wrist_width, wrist_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
            wrist = spaces.Box(
                0,
                255,
                shape=(wrist_height, wrist_width, 3),
                dtype=np.uint8,
            )
            result["images"] = spaces.Dict(
                {
                    "random_camera" if self.randomize else "ego": head,
                    "wrist_left": wrist,
                    "wrist_right": wrist,
                }
            )
        return spaces.Dict(result)

    def _articulation_vector(self) -> np.ndarray:
        if self.task_name in B320_DUAL_KNOB_TASK_NAMES:
            return self._ka005_2_state.joint_observation()
        dof_names = self._b320_contract.dof_names
        values = []
        for name in dof_names:
            fraction, available = self._dof_fraction(name)
            values.append(fraction if available else 0.0)
        return np.pad(np.asarray(values[:4], dtype=np.float64), (0, max(0, 4 - len(values))))

    def _compute_observation(self) -> dict[str, Any]:
        self._populate_interface_poses()
        metrics = self._metrics(update=False)
        metric_values = [
            float(value)
            for value in metrics.values()
            if isinstance(value, (bool, int, float, np.bool_, np.number))
        ]
        task_metrics = np.zeros(24, dtype=np.float64)
        task_metrics[: min(len(metric_values), 24)] = metric_values[:24]
        tcp = np.concatenate([self._right_tcp_pose, self._left_tcp_pose]).astype(np.float64)
        gripper = np.zeros(32, dtype=np.float64)
        gripper[: min(16, self._right_hand.size)] = self._right_hand[:16]
        gripper[16 : 16 + min(16, self._left_hand.size)] = self._left_hand[:16]
        if self.task_name in B320_DUAL_KNOB_TASK_NAMES:
            sync = self._ka005_2_state
            stage_state = np.asarray(
                [
                    float(sync.stage),
                    float(sync.attempt_id),
                    float(sync.attempt_armed),
                    float(sync.sync_motion_qualified),
                    float(sync.target_hold_complete),
                    float(sync.release_complete),
                    float(sync.success),
                    float(self.env_step),
                ],
                dtype=np.float64,
            )
        elif self.task_name == "t10_lift_carton":
            stage_state = np.asarray(
                [
                    float(self._b320_carton_transport_started),
                    float(self._b320_carton_airborne_now),
                    float(self._b320_carton_table_contact_readable),
                    float(self._b320_carton_zone_valid["left"]),
                    float(self._b320_carton_zone_valid["right"]),
                    float(
                        any(
                            peak > _CARTON_FORCE_LIMIT_N
                            for peak in self._b320_contact_peak.values()
                        )
                    ),
                    float(self._b320_carton_airborne_substeps),
                    float(self.env_step),
                ],
                dtype=np.float64,
            )
        elif self.task_name == "t20_pass_and_orient_pen":
            stage_state = np.asarray(
                [
                    float(self._b320_handover_right_first),
                    float(self._b320_handover_overlap),
                    float(self._b320_handover_left_only),
                    float(self._bc011_pen_airborne_latched),
                    float(self._bc011_pen_ground_contact_substeps),
                    float(self._bc011_pen_drop_detected),
                    float(self._b320_handover_completed),
                    float(self.env_step),
                ],
                dtype=np.float64,
            )
        elif self.task_name == B320_BC014_TASK_NAME:
            stage_state = np.asarray(
                [
                    float(self._bc014_flow_hit_seconds["left"]),
                    float(self._bc014_flow_hit_seconds["right"]),
                    float(self._bc014_flow_overlap_seconds),
                    float(self._bc014_flow_pour_seconds["left"]),
                    float(self._bc014_flow_pour_seconds["right"]),
                    float(
                        any(
                            seconds > _BC014_FLOW_MAX_POUR_SECONDS
                            for seconds in self._bc014_flow_pour_seconds.values()
                        )
                    ),
                    float(self._success_counter),
                    float(self.env_step),
                ],
                dtype=np.float64,
            )
        elif self.task_name == _KA012_TASK_NAME:
            anchor_position = (
                np.zeros(3, dtype=np.float64)
                if self._b320_stir_anchor_pose is None
                else self._b320_stir_anchor_pose[:3]
            )
            stage_state = np.asarray(
                [
                    *anchor_position,
                    float(self._b320_left_contact_lost),
                    float(len(self._b320_stir_sectors)),
                    float(self._b320_stir_accumulated_rad),
                    float(self._success_counter),
                    float(self.env_step),
                ],
                dtype=np.float64,
            )
        elif self.task_name in B320_FAUCET_TASK_NAMES:
            ka021_metrics = self._b320_last_metrics
            stage_state = np.asarray(
                [
                    float(ka021_metrics.get("ka021_fill_hold_frames", 0)),
                    float(ka021_metrics.get("ka021_water_filled", False)),
                    float(ka021_metrics.get("cup_parked", False)),
                    float(ka021_metrics.get("faucet_closed", False)),
                    float(ka021_metrics.get("post_fill_complete", False)),
                    float(self._success_counter),
                    float(all(self._b320_condition_flags)),
                    float(self.env_step),
                ],
                dtype=np.float64,
            )
        else:
            stage_state = np.asarray(
                [
                    float(self._b320_handover_right_first),
                    float(self._b320_handover_overlap),
                    float(self._b320_handover_left_only),
                    float(
                        self._b320_handover_completed
                        if self.task_name == "t20_pass_and_orient_pen"
                        else self._b320_left_contact_lost
                    ),
                    float(len(self._b320_stir_sectors)),
                    self._b320_stir_accumulated_rad,
                    float(self._success_counter),
                    float(self.env_step),
                ],
                dtype=np.float64,
            )
        recorded_contact_force = {
            side: self._carton_recorded_force_n(
                self._b320_contact_force[side]
            )
            for side in ("left", "right")
        }
        recorded_contact_peak = {
            side: self._carton_recorded_force_n(
                self._b320_contact_peak[side]
            )
            for side in ("left", "right")
        }
        goal_pose = self._b320_contract.goal_pose_wxyz
        if (
            self.task_name == "t20_pass_and_orient_pen"
            and self._bc011_pen_variant in BC011_PEN_TIP_AXIS_LOCAL
        ):
            goal_pose = bc011_tip_up_goal_pose_wxyz(self._bc011_pen_variant)
        elif self.task_name == B320_BC014_TASK_NAME:
            goal_pose = np.asarray(goal_pose, dtype=np.float64).copy()
            goal_pose[:2] = np.mean(
                np.stack(tuple(self._bc014_goal_centers_xy.values())),
                axis=0,
            )
        state = {
            "tcp_pose": tcp,
            "gripper_pose": gripper,
            "left_role_pose": self._pose(self._b320_contract.left_role),
            "right_role_pose": self._pose(self._b320_contract.right_role),
            "fixture_pose": self._pose(self._spec.receptacle),
            "goal_pose": np.asarray(goal_pose, dtype=np.float64),
            "task_metrics": task_metrics,
            "contact_metrics": np.asarray(
                [
                    float(self._b320_contact_available["left"]),
                    float(self._b320_contact_available["right"]),
                    recorded_contact_force["left"],
                    recorded_contact_force["right"],
                    recorded_contact_peak["left"],
                    recorded_contact_peak["right"],
                    float(self._b320_contact_seen["left"]),
                    float(self._b320_contact_seen["right"]),
                    float(self._b320_release_frame["left"] is not None),
                    float(self._b320_release_frame["right"] is not None),
                    float(all(self._b320_contact_available.values())),
                    float(all(self._b320_condition_flags)),
                ],
                dtype=np.float64,
            ),
            "stage_state": stage_state,
            "articulation_q": self._articulation_vector(),
            "table_delta_height": self.delta_h.astype(np.float64, copy=True),
        }
        if self.task_name in B320_DUAL_KNOB_TASK_NAMES:
            _base_translation_m, _base_rotation_deg, base_stable = (
                self._ka005_2_base_stability()
            )
            state.update(
                {
                    "knob_joint_state": self._ka005_2_state.joint_observation(),
                    "knob_target_state": self._ka005_2_state.target_observation(),
                    "knob_contact_state": self._ka005_2_state.contact_observation(),
                    "task_validity": np.asarray(
                        [
                            float(self._ka005_2_state.dof_truth_available),
                            float(self._ka005_2_state.dof_truth_available),
                            float(self._ka005_2_state.contact_truth_available),
                            float(base_stable),
                        ],
                        dtype=np.float64,
                    ),
                    "sync_task_state": self._ka005_2_state.stage_observation(),
                    "task_terminal_state": (
                        self._ka005_2_terminal_observation()
                    ),
                }
            )
        observation: dict[str, Any] = {"state": state}
        if self.image_obs:
            front, wrist_left, wrist_right = self.render()
            observation["images"] = {
                "random_camera" if self.randomize else "ego": front,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }
        return observation

    def _compute_success_now(self) -> bool:
        return bool(self._metrics(update=True)["success_now"])

    def _compute_failure(self) -> bool:
        if self.task_name == B320_BC014_TASK_NAME:
            goal_completed = bool(
                all(self._bc014_flow_hit_latched.values())
                and self._bc014_flow_overlap_latched
            )
            timed_out_sides = tuple(
                side
                for side, seconds in self._bc014_flow_pour_seconds.items()
                if seconds > _BC014_FLOW_MAX_POUR_SECONDS + 1.0e-9
            )
            reason: str | None = None
            if not goal_completed and timed_out_sides:
                timeout_role = (
                    "both"
                    if len(timed_out_sides) == 2
                    else "salt"
                    if timed_out_sides[0] == "left"
                    else "pepper"
                )
                reason = f"bc014_{timeout_role}_pour_timeout"
            self._b320_failure_reason = reason
            if reason is not None:
                self._success_counter = 0
                return True
            return bool(super()._compute_failure())
        if self.task_name in B320_DUAL_KNOB_TASK_NAMES:
            reason = self._ka005_2_terminal_failure_reason()
            self._b320_failure_reason = reason
            if reason is not None:
                self._success_counter = 0
                return True
            return bool(super()._compute_failure())
        if self.task_name == "t20_pass_and_orient_pen":
            if not self._launch_isaac:
                self._update_bc011_drop_latches()
            reason = (
                "bc011_pen_dropped"
                if self._bc011_pen_drop_detected
                else None
            )
            self._b320_failure_reason = reason
            if reason is not None:
                self._success_counter = 0
                return True
            return bool(super()._compute_failure())
        if self.task_name != "t10_lift_carton":
            return bool(super()._compute_failure())

        self._update_carton_transport_latches()
        pose = self._pose("carton")
        reset_pose = self._b320_reset_poses.get(
            "carton",
            self._fallback_b320_pose("carton"),
        )
        reason: str | None = None
        if any(
            peak > _CARTON_FORCE_LIMIT_N
            for peak in self._b320_contact_peak.values()
        ):
            reason = "carton_side_force_exceeded"
        elif pose[2] < reset_pose[2] - _CARTON_DROP_DISTANCE_M:
            reason = "carton_dropped"

        self._b320_failure_reason = reason
        if reason is not None:
            self._success_counter = 0
            return True
        return False

    def step_with_current_joint_targets(self, *, start_time: float | None = None):
        """Keep B320 time limits as truncation, separate from physical failure."""

        observation, reward, terminated, truncated, info = (
            super().step_with_current_joint_targets(start_time=start_time)
        )
        timed_out = bool(
            self.env_step * self._control_dt >= self._time_limit
            and not info.get("succeed", False)
            and not info.get("failed", False)
            and not info.get("manual_reset", False)
        )
        if timed_out:
            terminated = False
            truncated = True
            info = dict(info)
            info["time_limit_reached"] = True
            info["truncated_reason"] = "time_limit"
        return observation, reward, terminated, truncated, info

    def _success_steps_required(self) -> int:
        return max(1, int(self._b320_contract.stable_frames))

    def _ka005_2_progress_values(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = self._ka005_2_state
        travel_rad = np.abs(state.target_q - state.initial_q)
        direction = np.sign(state.target_q - state.initial_q)
        progress_rad = np.maximum(
            direction * (state.last_q - state.initial_q),
            0.0,
        )
        reached = (
            np.abs(state.target_q - state.last_q) <= math.radians(5.0)
        ) & bool(state.dof_truth_available)
        return np.degrees(progress_rad), np.degrees(travel_rad), reached

    @staticmethod
    def _ka005_2_progress_card_style(side: str, reached: bool) -> dict[str, Any]:
        active_border = 0xFF42DFF5 if side == "left" else 0xFFFFB64A
        return {
            "background_color": 0xD0181C22,
            "border_color": active_border if reached else 0x70FFFFFF,
            "border_width": 5 if reached else 2,
            "border_radius": 12,
        }

    def _init_ka005_2_progress_overlay(self) -> None:
        if (
            self.task_name not in B320_DUAL_KNOB_TASK_NAMES
            or self._headless
            or not self._launch_isaac
            or self._ka005_2_progress_frame is not None
        ):
            return
        try:
            import omni.ui as ui
            from omni.kit.viewport.utility import get_active_viewport_window

            viewport_window = get_active_viewport_window()
            if viewport_window is None:
                return
            frame = viewport_window.get_frame(
                "handson.ka005_2_knob_progress_overlay"
            )
            clear = getattr(frame, "clear", None)
            if callable(clear):
                clear()
            frame.visible = True
            self._ka005_2_progress_frame = frame
            self._ka005_2_progress_labels.clear()
            self._ka005_2_progress_cards.clear()
            with frame:
                with ui.VStack():
                    ui.Spacer()
                    with ui.HStack(height=112):
                        ui.Spacer()
                        titles = (
                            (("left", "LEFT CCW: CURRENT / TARGET"),
                             ("right", "RIGHT CW: CURRENT / TARGET"))
                            if self.task_name == GEN6_TASK_NAME
                            else (("left", "LEFT  CCW"), ("right", "RIGHT  CW"))
                        )
                        for index, (side, title) in enumerate(titles):
                            if index:
                                ui.Spacer(width=70)
                            with ui.ZStack(width=250, height=104):
                                card = ui.Rectangle(
                                    style=self._ka005_2_progress_card_style(
                                        side,
                                        False,
                                    )
                                )
                                with ui.VStack(
                                    spacing=2,
                                    style={"margin": 10},
                                ):
                                    ui.Label(
                                        title,
                                        height=24,
                                        alignment=ui.Alignment.CENTER,
                                        style={
                                            "font_size": 18,
                                            "color": 0xFFD7DEE8,
                                        },
                                    )
                                    value = ui.Label(
                                        "0 / 0 deg",
                                        height=52,
                                        alignment=ui.Alignment.CENTER,
                                        style={
                                            "font_size": 34,
                                            "color": 0xFFFFFFFF,
                                        },
                                    )
                                self._ka005_2_progress_cards[side] = card
                                self._ka005_2_progress_labels[side] = value
                        ui.Spacer()
                    ui.Spacer(height=90)
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not create KA005-2 knob-progress overlay: {exc}"
            )

    def _update_ka005_2_progress_overlay(self) -> None:
        if self.task_name not in B320_DUAL_KNOB_TASK_NAMES:
            return
        self._init_ka005_2_progress_overlay()
        if self._ka005_2_progress_frame is None:
            return
        try:
            self._ka005_2_progress_frame.visible = True
            progress_deg, travel_deg, reached = self._ka005_2_progress_values()
            for index, side in enumerate(("left", "right")):
                label = self._ka005_2_progress_labels.get(side)
                if label is not None:
                    if self.task_name == GEN6_TASK_NAME:
                        current = math.degrees(self._ka005_2_state.last_q[index])
                        target = math.degrees(self._ka005_2_state.target_q[index])
                        label.text = f"{current:+.0f} / {target:+.0f} deg"
                    else:
                        label.text = (
                            f"{progress_deg[index]:.0f} / "
                            f"{travel_deg[index]:.0f} deg"
                        )
                card = self._ka005_2_progress_cards.get(side)
                if card is not None:
                    card.style = self._ka005_2_progress_card_style(
                        side,
                        bool(reached[index]),
                    )
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not update KA005-2 knob-progress overlay: {exc}"
            )

    def _init_success_condition_overlay(self) -> None:
        super()._init_success_condition_overlay()
        self._init_ka005_2_progress_overlay()

    def _update_success_condition_overlay(self) -> None:
        super()._update_success_condition_overlay()
        self._update_ka005_2_progress_overlay()

    def _task_status_text(self) -> str:
        if not self._bc011_collection_angle_guidance:
            return super()._task_status_text()
        self._metrics(update=False)
        metrics = self._b320_last_metrics
        if not bool(metrics.get("tip_axis_readable", False)):
            return "笔尖仰角 --  |  目标 90° ± 2°"
        elevation = float(metrics.get("tip_elevation_deg", -90.0))
        return f"笔尖仰角 {elevation:.1f}°  |  目标 90° ± 2°"

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        self._metrics(update=False)
        rows = [
            self._condition_row(
                f"{self._b320_contract.task_id.lower().replace('-', '_')}_{index + 1}",
                label,
                self._b320_condition_flags[index],
            )
            for index, label in enumerate(self._b320_contract.success_labels)
        ]
        if self.task_name == "t10_lift_carton":
            return rows
        rows.append(
            self._condition_row(
                "b320_success_hold",
                (
                    "Success latches on the first frame when all conditions are met"
                    if self._b320_contract.stable_frames == 1
                    else (
                        "All conditions hold for "
                        f"{self._b320_contract.stable_frames / B320_CONTROL_HZ:.1f} "
                        "seconds"
                    )
                ),
                self._success_counter >= self._success_steps_required(),
            )
        )
        return rows

    def _task_info(self) -> dict[str, Any]:
        base = super()._task_info()
        info = dict(base) if isinstance(base, dict) else {}
        recorded_contact_force = {
            side: self._carton_recorded_force_n(
                self._b320_contact_force[side]
            )
            for side in ("left", "right")
        }
        recorded_contact_peak = {
            side: self._carton_recorded_force_n(
                self._b320_contact_peak[side]
            )
            for side in ("left", "right")
        }
        info.update(
            {
                "task_id": self._b320_contract.task_id,
                "task_name": self.task_name,
                "b320_metrics": dict(self._metrics(update=False)),
                "b320_contact_available": dict(self._b320_contact_available),
                "b320_contact_readable": dict(self._b320_contact_readable),
                "b320_contact_force_n": recorded_contact_force,
                "b320_contact_peak_n": recorded_contact_peak,
                "b320_carton_contact_detail_available": dict(
                    self._b320_carton_contact_detail_available
                ),
                "b320_carton_zone_valid": dict(self._b320_carton_zone_valid),
                "b320_carton_zone_force_n": {
                    side: self._carton_recorded_force_n(
                        self._b320_carton_zone_force[side]
                    )
                    for side in ("left", "right")
                },
                "b320_carton_table_contact_available": bool(
                    self._b320_carton_table_contact_available
                ),
                "b320_carton_table_contact_readable": bool(
                    self._b320_carton_table_contact_readable
                ),
                "b320_carton_table_contact_force_n": float(
                    self._b320_carton_table_contact_force
                ),
                "b320_pen_table_contact_available": bool(
                    self._b320_pen_table_contact_available
                ),
                "b320_pen_table_contact_readable": bool(
                    self._b320_pen_table_contact_readable
                ),
                "b320_pen_table_contact_force_n": float(
                    self._b320_pen_table_contact_force
                ),
                "b320_carton_airborne_now": bool(
                    self._b320_carton_airborne_now
                ),
                "b320_carton_airborne_substeps": int(
                    self._b320_carton_airborne_substeps
                ),
                "b320_carton_airborne_substeps_required": (
                    _CARTON_AIRBORNE_REQUIRED_SUBSTEPS
                ),
                "b320_carton_airborne_hold_seconds": min(
                    _CARTON_AIRBORNE_HOLD_SECONDS,
                    self._b320_carton_airborne_substeps
                    / float(B320_PHYSICS_HZ),
                ),
                "b320_carton_transport_started": bool(
                    self._b320_carton_transport_started
                ),
                "b320_carton_transport_support_lost": bool(
                    self._b320_carton_transport_contact_lost
                ),
                "b320_failure_reason": self._b320_failure_reason,
                "b320_success_counter": int(self._success_counter),
                "b320_success_frames_required": self._success_steps_required(),
                "b320_handover_completed": bool(
                    self._b320_handover_completed
                ),
                "b320_physical_replay_certified": False,
            }
        )
        if self.task_name == "t20_pass_and_orient_pen":
            info["bc011_pen_variant"] = self._bc011_pen_variant
            info["bc011_pen_tip_axis_local"] = list(
                BC011_PEN_TIP_AXIS_LOCAL.get(
                    self._bc011_pen_variant,
                    (0.0, 0.0, 0.0),
                )
            )
            info["bc011_pen_airborne_latched"] = bool(
                self._bc011_pen_airborne_latched
            )
            info["bc011_pen_ground_contact_substeps"] = int(
                self._bc011_pen_ground_contact_substeps
            )
            info["bc011_pen_drop_detected"] = bool(
                self._bc011_pen_drop_detected
            )
        if self.task_name in B320_DUAL_KNOB_TASK_NAMES:
            base_translation_m, base_rotation_deg, _base_stable = (
                self._ka005_2_base_stability()
            )
            info.update(
                {
                    "ka005_2_reset": self._ka005_2_reset_sample.as_dict(),
                    "ka005_2_stage": int(self._ka005_2_state.stage),
                    "ka005_2_attempt_id": int(self._ka005_2_state.attempt_id),
                    "ka005_2_attempt_armed": bool(
                        self._ka005_2_state.attempt_armed
                    ),
                    "ka005_2_attempt_invalid_reason": (
                        self._ka005_2_state.attempt_invalid_reason or None
                    ),
                    "ka005_2_failure_reason": (
                        self._ka005_2_terminal_failure_reason()
                    ),
                    "ka005_2_contact_force_n": dict(
                        self._ka005_2_contact_force_n
                    ),
                    "ka005_2_contact_readable": dict(
                        self._ka005_2_contact_readable
                    ),
                    "ka005_2_target_rad": list(
                        self._ka005_2_reset_sample.target_rad
                    ),
                    "ka005_2_success_latched": bool(
                        self._ka005_2_state.success
                    ),
                    "ka005_2_recording_origin_accepted": bool(
                        self._ka005_2_recording_origin_accepted
                    ),
                    "ka005_2_recording_origin_rad": [
                        float(value)
                        for value in self._ka005_2_recording_origin_q_rad
                    ],
                    "ka005_2_panel_base_translation_drift_m": base_translation_m,
                    "ka005_2_panel_base_rotation_drift_deg": base_rotation_deg,
                }
            )
        if self.task_name == B320_BC014_TASK_NAME:
            info["bc014_reset"] = self._bc014_layout_metadata()
            info["bc014_goal_centers_xy"] = (
                self._bc014_goal_centers_metadata()
            )
            info["bc014_flow_proxy"] = self._bc014_flow_proxy_metadata()
            info["bc014_realistic_kitchen"] = (
                self._bc014_realistic_kitchen_metadata()
            )
        if self.task_name == _KA012_TASK_NAME:
            info["ka012_reset"] = self._ka012_reset_metadata()
            info["ka012_stir_state"] = self._ka012_stir_state_metadata()
        if self.task_name in B320_FAUCET_TASK_NAMES:
            info["ka021_faucet"] = self._ka021_faucet_metadata()
        return info

    def _ka021_faucet_metadata(self) -> dict[str, Any]:
        return {
            "asset_bundle": KA021_ASSET_BUNDLE_KEY,
            "asset_manifest": f"{KA021_ASSET_BUNDLE_KEY}/asset_manifest.json",
            "faucet_asset_bundle": KA021_FAUCET_ASSET_BUNDLE_KEY,
            "faucet_asset_manifest": (
                f"{KA021_FAUCET_ASSET_BUNDLE_KEY}/asset_manifest.json"
            ),
            "faucet_source_creator": KA021_FAUCET_SOURCE_CREATOR,
            "faucet_source_url": KA021_FAUCET_SOURCE_URL,
            "faucet_source_sha256": KA021_FAUCET_SOURCE_SHA256,
            "source_licenses": ["CC BY 4.0", KA021_FAUCET_SOURCE_LICENSE],
            "shared_scene": "B320-HS005",
            "reuses_hs005_realistic_kitchen": True,
            "source_open_sink_collision_enabled": True,
            "shared_scene_collision_policy": (
                KA021_SHARED_SCENE_COLLISION_POLICY
            ),
            "shared_scene_collision_root": (
                KA021_SHARED_SCENE_COLLISION_ROOT_PATH
            ),
            "shared_scene_collision_specs": [
                {
                    "name": name,
                    "size_m": list(size),
                    "center_w": list(center),
                }
                for name, size, center in KA021_SHARED_SCENE_COLLISION_SPECS
            ],
            "safe_open_basin_floor_collision": KA021_SAFE_BASIN_FLOOR_PATH,
            "safe_open_basin_floor_size_m": list(KA021_SAFE_BASIN_FLOOR_SIZE_M),
            "safe_open_basin_floor_center_w": list(
                KA021_SAFE_BASIN_FLOOR_CENTER_W
            ),
            "counter_surface_collision_root": (
                KA021_COUNTER_SURFACE_COLLISION_PATH
            ),
            "counter_surface_collision_specs": [
                {
                    "name": name,
                    "size_m": list(size),
                    "center_w": list(center),
                    "top_z_m": KA021_COUNTER_SURFACE_Z,
                }
                for name, size, center in KA021_COUNTER_SURFACE_COLLISION_SPECS
            ],
            "cup_park_support_collision": KA021_CUP_PARK_SUPPORT_PATH,
            "cup_park_support_size_m": list(KA021_CUP_PARK_SUPPORT_SIZE_M),
            "cup_park_support_center_w": [
                float(self._ka021_cup_park_position_w[0]),
                float(self._ka021_cup_park_position_w[1]),
                float(self._ka021_cup_park_position_w[2] - 0.010),
            ],
            "cup_park_marker": KA021_CUP_PARK_MARKER_PATH,
            "cup_park_region": "outside_sink_left_counter",
            "cup_park_surface_z_m": KA021_COUNTER_SURFACE_Z,
            "cup_park_position_w": [
                float(value) for value in self._ka021_cup_park_position_w
            ],
            "cup_park_tolerance_m": KA021_CUP_PARK_TOLERANCE_M,
            "cup_park_z_tolerance_m": KA021_CUP_PARK_Z_TOLERANCE_M,
            "mug_open_cavity_collision": True,
            "mug_open_cavity_collision_model": (
                "four_wall_segments_plus_handle_capsules_and_foot"
            ),
            "mug_handle_collision_radius_m": KA021_MUG_HANDLE_COLLIDER_RADIUS_M,
            "mug_handle_collision_centerline_source_m": [
                [list(start), list(end)]
                for start, end in KA021_MUG_HANDLE_CENTERLINE_SOURCE_M
            ],
            "source_faucet_articulated": False,
            "runtime_lever_authored": True,
            "runtime_handle_visual_uses_source_geometry": True,
            "water_proxy_path": KA021_FLOW_PROXY_PATH,
            "water_visual_model": "render_only_cylinder",
            "particle_model": False,
            "outlet_position_w": list(KA021_OUTLET_POSITION_W),
            "cup_opening_radius_m": KA021_CUP_OPENING_RADIUS_M,
            "cup_rim_height_m": KA021_CUP_RIM_HEIGHT_M,
            "cup_up_axis_local": list(KA021_CUP_UP_AXIS_LOCAL),
            "cup_home_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "cup_asset_root_up_axis": "Z",
            "cup_asset_root_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "cup_upright_tolerance_deg": KA021_CUP_UPRIGHT_TOLERANCE_DEG,
            "cup_upright_boundary_inclusive": True,
            "outlet_rim_gap_range_m": [
                KA021_MIN_OUTLET_RIM_GAP_M,
                KA021_MAX_OUTLET_RIM_GAP_M,
            ],
            "flow_visible_fraction": KA021_FLOW_VISIBLE_FRACTION,
            "success_open_fraction": KA021_SUCCESS_OPEN_FRACTION,
            "runtime_lever_open_axis": KA021_FAUCET_OPEN_AXIS,
            "runtime_joint_frame": "yawed_source_y_axis_with_sign_flip",
            "runtime_joint_frame_orientation_wxyz": list(
                KA021_FAUCET_JOINT_FRAME_ORIENTATION_WXYZ
            ),
            "runtime_joint_axis_world_w": list(KA021_FAUCET_JOINT_AXIS_WORLD_W),
            "runtime_initial_open_direction_world_w": list(
                KA021_FAUCET_INITIAL_OPEN_DIRECTION_W
            ),
            "runtime_lever_limits_deg": [KA021_FAUCET_OPEN_LIMIT_DEG, 0.0],
            "runtime_lever_reset_position_rad": KA021_FAUCET_RESET_POSITION_RAD,
            "runtime_lever_open_travel_rad": KA021_FAUCET_OPEN_TRAVEL_RAD,
            "runtime_lever_open_travel_deg": abs(KA021_FAUCET_OPEN_LIMIT_DEG),
            "runtime_lever_outward_direction_w": list(
                KA021_FAUCET_OUTWARD_DIRECTION_W
            ),
            "runtime_lever_fraction_formula": "clip(-q_rad / radians(45), 0, 1)",
            "runtime_handle_pivot_w": list(KA021_FAUCET_HANDLE_PIVOT_W),
            "runtime_base_position_w": list(KA021_FAUCET_BASE_POSITION_W),
            "runtime_split_method": "authored_object_boundary",
            "runtime_handle_grip_collider_size_m": list(
                KA021_FAUCET_GRIP_COLLIDER_SIZE_M
            ),
            "lateral_left_right_motion_role": "temperature_only_not_scored",
            "success_hold_seconds": (
                self._b320_contract.stable_frames / B320_CONTROL_HZ
            ),
            "fill_phase_hold_seconds": (
                self._b320_contract.stable_frames / B320_CONTROL_HZ
            ),
            "fill_requires_faucet_open": True,
            "fill_hold_frames": self._b320_contract.stable_frames,
            "success_condition_count": len(self._b320_contract.success_labels),
            "post_fill_requires_cup_park": True,
            "post_fill_requires_faucet_closed": True,
            "randomized_reset_enabled": bool(self.randomize),
            "reset_sample_attempts": int(self._ka021_reset_attempts),
            "cup_reset_pose_wxyz": (
                None
                if self._ka021_reset_pose is None
                else [float(value) for value in self._ka021_reset_pose]
            ),
            "cup_reset_xy_bounds_m": [
                list(values) for values in _KA021_CUP_RESET_CENTER_BOUNDS_XY
            ],
            "cup_reset_complete_visual_footprint_radius_m": (
                _KA021_CUP_FOOTPRINT_RADIUS_M
            ),
            "cup_reset_basin_edge_clearance_m": (
                _KA021_CUP_RESET_EDGE_CLEARANCE_M
            ),
            "cup_reset_basin_floor_top_z_m": _KA021_BASIN_FLOOR_TOP_Z_M,
            "cup_reset_requires_complete_mug_inside_basin": True,
            "cup_reset_visible_basin_opening_bounds_xy_m": [
                list(values)
                for values in KA021_VISIBLE_BASIN_OPENING_BOUNDS_XY
            ],
            "cup_reset_excludes_drainboard": True,
            "cup_park_xy_bounds_m": [
                list(values) for values in _KA021_CUP_PARK_CENTER_BOUNDS_XY
            ],
            "cup_reset_xy_jitter_m": (
                0.0
                if not self.randomize or self._ka021_reset_pose is None
                else float(
                    np.max(
                        np.abs(
                            self._ka021_reset_pose[:2]
                            - np.asarray(
                                self._fallback_b320_pose("ceramic_mug")[:2],
                                dtype=np.float64,
                            )
                        )
                    )
                )
            ),
            "cup_reset_yaw_degrees": _KA021_RESET_YAW_DEG if self.randomize else 0.0,
        }

    def _bc014_flow_proxy_metadata(self) -> dict[str, Any]:
        return {
            "paths": dict(BC014_FLOW_PROXY_PATHS),
            "visual_model": "d09_water_stream_cylinder",
            "particle_model": False,
            "cap_axis_local": [0.0, 0.0, 1.0],
            "tilt_threshold_deg": _BC014_FLOW_TILT_THRESHOLD_DEG,
            "maximum_length_m": _BC014_FLOW_MAX_LENGTH_M,
            "target_plane_z_m": _BC014_FLOW_TARGET_Z_M,
            "required_hit_seconds": _BC014_FLOW_REQUIRED_SECONDS,
            "required_overlap_seconds": _BC014_FLOW_OVERLAP_REQUIRED_SECONDS,
            "maximum_pour_seconds_before_reset": (
                _BC014_FLOW_MAX_POUR_SECONDS
            ),
            "pour_timeout_accumulation": "cumulative_per_jar",
            "pour_timeout_until": "both_target_and_overlap_latches_complete",
            "individual_hit_accumulation": "cumulative",
            "overlap_hit_accumulation": "continuous_until_latched",
            "jar_placement_required": False,
            "jar_release_required": False,
        }

    def _bc014_goal_centers_metadata(self) -> dict[str, list[float]]:
        return {
            role: [float(value) for value in center]
            for role, center in self._bc014_goal_centers_xy.items()
        }

    def _bc014_layout_metadata(self) -> dict[str, Any]:
        return {
            "poses_wxyz": {
                name: pose.tolist()
                for name, pose in sorted(self._bc014_reset_poses.items())
            },
            "goal_centers_xy": self._bc014_goal_centers_metadata(),
            "goal_marker_paths": dict(BC014_GOAL_MARKER_PATHS),
            "attempts": int(self._bc014_reset_attempts),
            "randomize": bool(self.randomize),
            "comfortable_center_bounds_xy": {
                role: {
                    "minimum": list(bounds[0]),
                    "maximum": list(bounds[1]),
                }
                for role, bounds in _BC014_COMFORT_CENTER_BOUNDS_XY.items()
            },
            "assigned_hands": {
                "salt_jar": "left",
                "pepper_jar": "right",
            },
            "yaw_jitter_deg": _BC014_RESET_YAW_DEG,
            "goal_zone_size_m": list(BC014_GOAL_ZONE_SIZE_M),
            "minimum_jar_goal_separation_m": (
                _BC014_MIN_JAR_GOAL_SEPARATION_M
            ),
            "comfort_reach_z_m": _BC014_COMFORT_REACH_Z_M,
            "comfort_max_radius_m": _BC014_COMFORT_MAX_RADIUS_M,
            "spacing_constraint": False,
        }

    def _ka012_reset_metadata(self) -> dict[str, Any]:
        pose = self._ka012_reset_pose
        reach = (
            self._ka012_pot_reach_metrics(pose)
            if pose is not None
            else {}
        )
        return {
            "pot_pose_wxyz": None if pose is None else pose.tolist(),
            "attempts": int(self._ka012_reset_attempts),
            "randomize": bool(self.randomize),
            "center_bounds_xy": {
                "minimum": list(_KA012_POT_CENTER_BOUNDS_XY[0]),
                "maximum": list(_KA012_POT_CENTER_BOUNDS_XY[1]),
            },
            "yaw_jitter_deg": _KA012_RESET_YAW_DEG,
            "comfort_max_radius_m": _KA012_COMFORT_MAX_RADIUS_M,
            "left_role": "near_handle",
            "right_role": "complete_stir_path",
            "reachable": bool(
                pose is not None and self._ka012_pot_reset_is_reachable(pose)
            ),
            **reach,
        }

    def _ka012_stir_state_metadata(self) -> dict[str, Any]:
        return {
            "anchor_pose_wxyz": (
                None
                if self._b320_stir_anchor_pose is None
                else self._b320_stir_anchor_pose.tolist()
            ),
            "anchor_frame": self._b320_stir_anchor_frame,
            "anchor_event": "first_valid_stir_sample_with_both_role_contacts",
            "translation_limit_m": 0.025,
            "left_contact_lost": bool(self._b320_left_contact_lost),
            "checkpoint_count": len(self._b320_stir_sectors),
            "accumulated_angle_rad": float(self._b320_stir_accumulated_rad),
        }

    def _bc014_realistic_kitchen_metadata(self) -> dict[str, Any]:
        if self._bc014_kitchen_report is not None:
            return self._bc014_kitchen_report.replay_metadata()
        return realistic_kitchen_metadata(root_path=_BC014_KITCHEN_ROOT)

    def get_physical_replay_reset_contract(self) -> dict[str, Any]:
        contract = dict(super().get_physical_replay_reset_contract())
        contract.update(
            {
                "b320_task_id": self._b320_contract.task_id,
                "b320_formal_body_paths": dict(self._b320_spawn_result.body_paths),
                "b320_requires_fresh_process_replay": True,
                "b320_physical_replay_certified": False,
            }
        )
        if self.task_name == "t20_pass_and_orient_pen":
            contract.update(
                {
                    "bc011_pen_variant": self._bc011_pen_variant,
                    "bc011_pen_variant_index": self._bc011_pen_variant_index,
                    "bc011_pen_variants": list(BC011_PEN_VARIANTS),
                    "bc011_pen_tip_axis_local": list(
                        BC011_PEN_TIP_AXIS_LOCAL.get(
                            self._bc011_pen_variant,
                            (0.0, 0.0, 0.0),
                        )
                    ),
                }
            )
        if self.task_name == B320_BC014_TASK_NAME:
            goal_spacing = float(
                np.linalg.norm(
                    self._bc014_goal_centers_xy["salt_jar"]
                    - self._bc014_goal_centers_xy["pepper_jar"]
                )
            )
            contract.update(
                {
                    "bc014_goal_centers_xy": (
                        self._bc014_goal_centers_metadata()
                    ),
                    "bc014_goal_spacing_m": goal_spacing,
                    "bc014_goal_zone_size_m": list(BC014_GOAL_ZONE_SIZE_M),
                    "bc014_spacing_constraint": False,
                    "bc014_flow_proxy": self._bc014_flow_proxy_metadata(),
                    "bc014_reset": self._bc014_layout_metadata(),
                    "bc014_realistic_kitchen": (
                        self._bc014_realistic_kitchen_metadata()
                    ),
                }
            )
        if self.task_name == _KA012_TASK_NAME:
            contract["ka012_reset"] = self._ka012_reset_metadata()
            contract["ka012_stir_semantics"] = {
                "translation_reference": "stir_anchor_pose",
                "anchor_event": (
                    "first_valid_stir_sample_with_both_role_contacts"
                ),
                "translation_limit_m": 0.025,
                "reset_anchor_pose": None,
            }
        if self.task_name in B320_DUAL_KNOB_TASK_NAMES:
            contract.update(
                {
                    "ka005_2_dof_names": list(KA005_2_DOF_NAMES),
                    "ka005_2_dof_units": ["radian", "radian"],
                    "ka005_2_initial_rad": list(
                        self._ka005_2_reset_sample.initial_rad
                    ),
                    "ka005_2_target_rad": list(
                        self._ka005_2_reset_sample.target_rad
                    ),
                    "ka005_2_reset_fallback_used": bool(
                        self._ka005_2_reset_sample.fallback_used
                    ),
                    "ka005_2_recording_origin_accepted": bool(
                        self._ka005_2_recording_origin_accepted
                    ),
                    "ka005_2_recording_origin_rad": [
                        float(value)
                        for value in self._ka005_2_recording_origin_q_rad
                    ],
                    "ka005_2_state_machine": "ordered_contact_motion_target_release_stable_v1",
                }
            )
        return contract

    def close(self) -> None:
        frame = self._ka005_2_progress_frame
        self._ka005_2_progress_frame = None
        self._ka005_2_progress_labels.clear()
        self._ka005_2_progress_cards.clear()
        if frame is not None:
            try:
                frame.visible = False
                clear = getattr(frame, "clear", None)
                if callable(clear):
                    clear()
            except Exception:
                pass
        super().close()


__all__ = ["HandsOnB320BimanualIsaacLabEnv"]
