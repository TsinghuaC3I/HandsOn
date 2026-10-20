"""Native IsaacLab environment for G01 deformable-asset inspection."""

from __future__ import annotations

import math
import hashlib
from typing import Any, Literal, Mapping

from gymnasium import spaces
import numpy as np
from scipy.spatial.distance import cdist

from .abc_collection_env import (
    _D06_LOWERED_ARM_HOME,
    _G01_5_LOWERED_ARM_HOME,
    _matrix_to_quat_wxyz,
    _pose_vector_to_matrix,
    _quat_wxyz_to_matrix,
    _randomized_physx_pose,
    HandsOnABCIsaacLabEnv,
)
from .g01_deformable_asset_scene import (
    G01_2_PILLOW_SIM_POINT_COUNT,
    G01_2_PILLOW_SIM_MESH_PATH,
    G01_2_TASK_NAME,
    G01_3_ASSET_SPECS,
    G01_3_CARTON_COLLIDER_PATHS,
    G01_3_CARTON_INNER_LOCAL_MAX,
    G01_3_CARTON_INNER_LOCAL_MIN,
    G01_3_CARTON_LOCAL_MAX,
    G01_3_CARTON_LOCAL_MIN,
    G01_3_CARTON_OPEN_FACE_COUNT,
    G01_3_CARTON_PATH,
    G01_3_CARTON_TRANSLATION,
    G01_3_PILLOW_UNIFORM_SCALE,
    G01_3_TASK_NAME,
    G01_CARTON_TASK_NAMES,
    G01_COMPARISON_PROFILES,
    G01_COMPARISON_NODE_COUNTS,
    G01_5_PILLOW_HALF_HEIGHT_M,
    G01_5_TASK_NAME,
    G01_2_PILLOW_MATERIAL_PATH,
    G01_PILLOW_BODY_PATH,
    G01_CABLE_RIGID_BODY_PATHS,
    G01_PHYSICS_HZ,
    G01_TABLE_TOP_Z,
    G01_TASK_NAME,
    G01_TASK_NAMES,
    G01_TASK_IDS,
    G01_TOWEL_NODE_COUNT,
    G01_TOWEL_SURFACE_MESH_PATH,
    g01_deformable_view_specs,
    g01_asset_specs,
    spawn_g01_deformable_assets,
)
from .recording_cameras import WRIST_RECORDING_CAMERA_OUTPUT_SIZE


_G01_BUTTON_CONTACT_FORCE_THRESHOLD = 0.15
_G01_BUTTON_POSITION = np.asarray([-0.46, -0.68, 0.909], dtype=np.float64)
_G01_2_DEFORMABLE_NODE_COUNTS = {
    "Pillow024": G01_2_PILLOW_SIM_POINT_COUNT,
    "Towl011": G01_TOWEL_NODE_COUNT,
}
_G01_2_VALID_WORLD_MIN = np.asarray([-2.0, -2.0, 0.50], dtype=np.float64)
_G01_2_VALID_WORLD_MAX = np.asarray([2.0, 2.0, 3.00], dtype=np.float64)
_G01_2_MAX_DEFORMABLE_SPAN_METERS = 2.0
_G01_2_MAX_NODAL_SPEED_MPS = 20.0
_G01_BUTTON_ARTICULATION_PATH_SUFFIX = "/g01_button_base/g01_button_base"
_G01_BUTTON_DOF_NAME = "g01_success_button_joint"
_G01_CURRENT_PILLOW_TASK_NAMES = frozenset(
    {G01_2_TASK_NAME, *G01_CARTON_TASK_NAMES}
)

_G01_3_CONTACT_LOCAL_RADIUS_M = 0.085
_G01_3_CONTACT_DISTANCE_M = 0.065
_G01_3_CONTACT_MIN_LOCAL_NODES = 4
_G01_3_CONTACT_MIN_DISPLACEMENT_M = 0.00015
_G01_3_CONTACT_MIN_DISPLACEMENT_EXCESS_M = 0.00008
_G01_3_CONTACT_MIN_SPEED_MPS = 0.015
_G01_3_CONTACT_MIN_SPEED_EXCESS_MPS = 0.005
# G01-5 grip is enabled only after the hand has made a real, localized
# indentation.  This prevents a grazing touch from turning the pillow into a
# sticky rigid object while making a deliberate squeeze liftable.
_G01_5_MIN_INDENTATION_M = 0.006
_G01_5_GRIP_STATIC_FRICTION = 1.80
_G01_5_GRIP_DYNAMIC_FRICTION = 1.45
_G01_5_BASE_STATIC_FRICTION = 0.75
_G01_5_BASE_DYNAMIC_FRICTION = 0.65
_G01_3_LIFT_CLEARANCE_M = 0.030
_G01_3_LIFT_DEBOUNCE_SUBSTEPS = 3
_G01_3_CONTAINMENT_FRACTION = 0.98
_G01_3_CONTAINMENT_TOLERANCE_M = 0.006
_G01_3_RIM_CLEARANCE_M = 0.008
# A foam corner may remain raised; the bulk must be below the rim and the
# pillow must reach the floor. Horizontal containment still excludes a pillow
# hanging over the outside wall, and floor support excludes a rim bridge.
_G01_5_CONTAINMENT_FRACTION = 0.90
_G01_5_FLOOR_SUPPORT_GAP_M = 0.025
_G01_3_RELEASE_CLEARANCE_M = 0.090
_G01_3_HAND_BODY_COVER_RADIUS_M = 0.055
# Do not let a tracker excursion drive a palm/finger body deep through the
# pillow.  The physical hand derivative also carries a 10 mm collection
# shell, so the software guard includes a smaller body-origin envelope.  A few
# millimetres of surface compliance remains valid during a grasp; the guard
# only rejects a swept envelope that is below the live pillow top while its
# horizontal projection is still inside the pillow.
_G01_3_HAND_POKE_HORIZONTAL_MARGIN_M = 0.012
_G01_3_HAND_POKE_MAX_DEPTH_M = 0.025
_G01_3_HAND_POKE_ENVELOPE_RADIUS_M = 0.020
_G01_3_RELEASE_DEBOUNCE_SUBSTEPS = 6
_G01_3_STABLE_RMS_SPEED_MPS = 0.025
_G01_3_STABLE_MAX_SPEED_MPS = 0.080
_G01_3_STABLE_OUTLIER_FRACTION = 0.01
_G01_3_STABLE_SECONDS = 1.0
_G01_3_INITIAL_SETTLE_CONSECUTIVE_SUBSTEPS = 12
_G01_3_INITIAL_SETTLE_TIMEOUT_SECONDS = 5.0
_G01_3_STAGE_WAITING_CONTACT = 0
_G01_3_STAGE_BOTH_CONTACTED = 1
_G01_3_STAGE_LIFTED = 2
_G01_3_STAGE_CONTAINED = 3
_G01_3_STAGE_RELEASED = 4
_G01_3_STAGE_COMPLETE = 5
_G01_3_STAGE_COUNT = 6

# G01-3's native-size pillow and footprint-scaled carton nearly fill the
# tabletop in X, so reset randomization is deliberately narrow and audited as
# a complete layout.  A 180-degree heading flip changes each object's
# direction without changing its footprint; the extra six-degree perturbation
# prevents the reset set from collapsing to two visually identical headings.
_G01_3_RESET_LAYOUT_CANONICAL = np.asarray(
    [-0.43, 0.0, 0.0, 0.10, 0.0, 0.0], dtype=np.float64
)
_G01_3_PILLOW_XY_JITTER_M = np.asarray([0.015, 0.050], dtype=np.float64)
_G01_3_CARTON_XY_JITTER_M = np.asarray([0.015, 0.050], dtype=np.float64)
# The shorter flex03 pillow permits much wider independent depth sampling.
# X remains constrained by the full carton/pillow footprints on the table.
_G01_5_PILLOW_XY_JITTER_M = np.asarray([0.045, 0.220], dtype=np.float64)
_G01_5_CARTON_XY_JITTER_M = np.asarray([0.010, 0.160], dtype=np.float64)
_G01_3_YAW_JITTER_DEG = 6.0
_G01_3_RESET_MAX_ATTEMPTS = 256
_G01_3_RESET_MIN_MARGIN_M = 0.005
_G01_3_TABLE_XY_MIN = np.asarray([-0.67, -0.80], dtype=np.float64)
_G01_3_TABLE_XY_MAX = np.asarray([0.37, 0.80], dtype=np.float64)
_G01_3_TABLE_EDGE_MARGIN_M = 0.015
# Keep every composed object inside the conservative portion of the fixed
# head-camera/table view.  Wrist-camera visibility necessarily follows the
# operator's hands, so the reset sampler can only guarantee the static main
# view before action zero.
_G01_3_MAIN_CAMERA_XY_MIN = np.asarray([-0.66, -0.62], dtype=np.float64)
_G01_3_MAIN_CAMERA_XY_MAX = np.asarray([0.36, 0.62], dtype=np.float64)
_G01_3_OBJECT_GAP_M = 0.025
_G01_3_ARM_BASES = {
    "right": np.asarray([-0.75, -0.065, 1.34], dtype=np.float64),
    "left": np.asarray([-0.75, 0.065, 1.34], dtype=np.float64),
}
_G01_3_ARM_REACH_MIN_M = 0.20
_G01_3_ARM_REACH_MAX_M = 0.72


def _g01_3_yaw_quaternion_wxyz(yaw_degrees: float) -> np.ndarray:
    yaw = math.radians(float(yaw_degrees))
    return np.asarray(
        [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
        dtype=np.float64,
    )


def _g01_3_world_aabb(
    local_min: np.ndarray | tuple[float, ...],
    local_max: np.ndarray | tuple[float, ...],
    translation: np.ndarray,
    yaw_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an exact world AABB for a yawed task-local box."""

    lower = np.asarray(local_min, dtype=np.float64).reshape(3)
    upper = np.asarray(local_max, dtype=np.float64).reshape(3)
    corners = np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )
    rotation = _quat_wxyz_to_matrix(
        _g01_3_yaw_quaternion_wxyz(yaw_degrees)
    )
    world = corners @ rotation.T + np.asarray(
        translation, dtype=np.float64
    ).reshape(1, 3)
    return np.min(world, axis=0), np.max(world, axis=0)


def _g01_3_reach_targets(
    layout: np.ndarray,
) -> dict[str, tuple[str, np.ndarray]]:
    """Return the two pickup and two release targets audited at reset."""

    values = np.asarray(layout, dtype=np.float64).reshape(6)
    pillow_xy = values[:2]
    # A 180-degree object heading does not swap the physical right/left hand
    # lanes.  Audit approach points against the small residual yaw while the
    # full heading is still applied to the actual object reset.
    pillow_lane_yaw = (float(values[2]) + 90.0) % 180.0 - 90.0
    carton_lane_yaw = (float(values[5]) + 90.0) % 180.0 - 90.0
    pillow_rotation = _quat_wxyz_to_matrix(
        _g01_3_yaw_quaternion_wxyz(pillow_lane_yaw)
    )[:2, :2]
    carton_xy = values[3:5]
    carton_rotation = _quat_wxyz_to_matrix(
        _g01_3_yaw_quaternion_wxyz(carton_lane_yaw)
    )[:2, :2]

    targets: dict[str, tuple[str, np.ndarray]] = {}
    for side, sign in (("right", -1.0), ("left", 1.0)):
        pickup_xy = pillow_xy + pillow_rotation @ np.asarray(
            [0.0, sign * 0.30], dtype=np.float64
        )
        release_xy = carton_xy + carton_rotation @ np.asarray(
            [-0.18, sign * 0.20], dtype=np.float64
        )
        targets[f"pillow_{side}_approach"] = (
            side,
            np.asarray([pickup_xy[0], pickup_xy[1], 1.14]),
        )
        targets[f"carton_{side}_release"] = (
            side,
            np.asarray([release_xy[0], release_xy[1], 1.28]),
        )
    return targets


def _g01_3_reset_layout_margin(
    layout: np.ndarray, task_name: str = G01_3_TASK_NAME,
) -> float:
    """Return the smallest support, separation, or arm-reach slack in metres."""

    values = np.asarray(layout, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        return -math.inf

    pillow_spec = (
        g01_asset_specs(task_name)[0] if task_name == G01_5_TASK_NAME
        else G01_3_ASSET_SPECS[0]
    )
    pillow_min, pillow_max = _g01_3_world_aabb(
        np.asarray(pillow_spec.local_bounds_min) * pillow_spec.uniform_scale,
        np.asarray(pillow_spec.local_bounds_max) * pillow_spec.uniform_scale,
        np.asarray([values[0], values[1], pillow_spec.translation[2]]),
        pillow_spec.rotation_z_degrees + values[2],
    )
    carton_min, carton_max = _g01_3_world_aabb(
        G01_3_CARTON_LOCAL_MIN,
        G01_3_CARTON_LOCAL_MAX,
        np.asarray([values[3], values[4], G01_3_CARTON_TRANSLATION[2]]),
        values[5],
    )

    table_min = _G01_3_TABLE_XY_MIN + _G01_3_TABLE_EDGE_MARGIN_M
    table_max = _G01_3_TABLE_XY_MAX - _G01_3_TABLE_EDGE_MARGIN_M
    margins = [
        *list(pillow_min[:2] - table_min),
        *list(table_max - pillow_max[:2]),
        *list(carton_min[:2] - table_min),
        *list(table_max - carton_max[:2]),
        *list(pillow_min[:2] - _G01_3_MAIN_CAMERA_XY_MIN),
        *list(_G01_3_MAIN_CAMERA_XY_MAX - pillow_max[:2]),
        *list(carton_min[:2] - _G01_3_MAIN_CAMERA_XY_MIN),
        *list(_G01_3_MAIN_CAMERA_XY_MAX - carton_max[:2]),
    ]
    separation = np.maximum(
        np.maximum(pillow_min[:2] - carton_max[:2], 0.0),
        carton_min[:2] - pillow_max[:2],
    )
    margins.append(float(np.linalg.norm(separation)) - _G01_3_OBJECT_GAP_M)

    if task_name == G01_5_TASK_NAME:
        # Native readback at the 80 mm lower open-hand home. Bound all 21
        # palm/finger origins per hand, then include the same 55 mm cover as
        # the live audit. Reject carton poses too close to the lowered hands.
        for sign in (-1.0, 1.0):
            hand_min = np.asarray([-0.3740, sign * 0.1833 - 0.0577, 1.1757])
            hand_max = np.asarray([-0.2125, sign * 0.1833 + 0.0577, 1.2336])
            gap = np.maximum(np.maximum(hand_min - carton_max, 0.0),
                             carton_min - hand_max)
            # With the shared 5 mm minimum slack this requires at least
            # 15 mm clearance outside the complete hand cover.
            margins.append(float(np.linalg.norm(gap))
                           - _G01_3_HAND_BODY_COVER_RADIUS_M - 0.010)

    for side, target in _g01_3_reach_targets(values).values():
        radius = float(np.linalg.norm(target - _G01_3_ARM_BASES[side]))
        margins.extend(
            [
                radius - _G01_3_ARM_REACH_MIN_M,
                _G01_3_ARM_REACH_MAX_M - radius,
            ]
        )
    return float(min(margins))


def _g01_3_expected_carton_pose_wxyz(layout: np.ndarray) -> np.ndarray:
    values = np.asarray(layout, dtype=np.float64).reshape(6)
    return np.concatenate(
        [
            np.asarray(
                [values[3], values[4], G01_3_CARTON_TRANSLATION[2]],
                dtype=np.float64,
            ),
            _g01_3_yaw_quaternion_wxyz(values[5]),
        ]
    )


def _g01_3_backtrack_poke_target_pose(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    hand_points: tuple[np.ndarray, ...],
    pillow_lower: np.ndarray,
    pillow_upper: np.ndarray,
    *,
    horizontal_margin_m: float = _G01_3_HAND_POKE_HORIZONTAL_MARGIN_M,
    max_depth_m: float = _G01_3_HAND_POKE_MAX_DEPTH_M,
    envelope_radius_m: float = _G01_3_HAND_POKE_ENVELOPE_RADIUS_M,
) -> tuple[np.ndarray, bool]:
    """Backtrack a palm target before its current hand envelope pokes deep.

    ``hand_points`` are current world-space palm/finger body origins.  Their
    offsets from the current palm are carried to the candidate target and
    checked against the live pillow AABB.  The body origins are expanded by a
    conservative radius to cover the gap between a sampled rigid-link origin
    and its visible/collision geometry.  If the candidate is unsafe, a binary
    search returns the furthest safe translation along the requested path.
    When the live hand is already in the guarded core, outward motion is still
    allowed so recovery cannot deadlock at a previously bad frame.
    """

    current = np.asarray(current_pose, dtype=np.float64).reshape(7)
    target = np.asarray(target_pose, dtype=np.float64).reshape(7)
    points = np.asarray(hand_points, dtype=np.float64).reshape(-1, 3)
    lower = np.asarray(pillow_lower, dtype=np.float64).reshape(3)
    upper = np.asarray(pillow_upper, dtype=np.float64).reshape(3)
    if (
        points.size == 0
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(upper <= lower)
    ):
        return target.copy(), False

    current_matrix = _pose_vector_to_matrix(current)
    target_matrix = _pose_vector_to_matrix(target)
    local_offsets = (
        (current_matrix[:3, :3].T @ (points - current_matrix[:3, 3]).T)
        .T
    )
    envelope = float(max(envelope_radius_m, 0.0))
    horizontal_margin = float(max(horizontal_margin_m, 0.0)) + envelope
    horizontal_lower = lower[:2] - horizontal_margin
    horizontal_upper = upper[:2] + horizontal_margin
    # A body origin above this plane can still have its inflated collision
    # shell below the core threshold.  Expanding the z test makes the guard
    # symmetric with the collection-only primitive collision inflation.
    depth_limit = float(upper[2] - max(float(max_depth_m), 0.0) + envelope)

    def predicted_points(fraction: float) -> np.ndarray:
        position = current_matrix[:3, 3] + float(fraction) * (
            target_matrix[:3, 3] - current_matrix[:3, 3]
        )
        return (
            target_matrix[:3, :3] @ local_offsets.T
        ).T + position

    def penetration(fraction: float) -> float:
        predicted = predicted_points(fraction)
        horizontal_clearance = np.minimum.reduce(
            [
                predicted[:, 0] - horizontal_lower[0],
                horizontal_upper[0] - predicted[:, 0],
                predicted[:, 1] - horizontal_lower[1],
                horizontal_upper[1] - predicted[:, 1],
            ]
        )
        deep = depth_limit - predicted[:, 2]
        inside_xy = horizontal_clearance >= 0.0
        if not np.any(inside_xy):
            return 0.0
        return float(np.max(np.where(inside_xy, deep, 0.0)))

    def unsafe(fraction: float) -> bool:
        predicted = predicted_points(fraction)
        inside_xy = np.all(
            (predicted[:, :2] >= horizontal_lower)
            & (predicted[:, :2] <= horizontal_upper),
            axis=1,
        )
        deep = predicted[:, 2] <= depth_limit
        return bool(np.any(inside_xy & deep))

    if not unsafe(1.0):
        return target.copy(), False
    if unsafe(0.0):
        # If the live hand is already in the guarded core, do not deadlock it:
        # an explicitly outward target is safe to issue even while it remains
        # inside the expanded envelope.  Only inward/lateral-deeper commands
        # are held at the current physical pose.
        current_penetration = penetration(0.0)
        target_penetration = penetration(1.0)
        if target_penetration + 1.0e-6 < current_penetration:
            return target.copy(), False
        return current.copy(), True

    lo, hi = 0.0, 1.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if unsafe(mid):
            hi = mid
        else:
            lo = mid
    clipped = target.copy()
    clipped[:3] = current[:3] + lo * (target[:3] - current[:3])
    return clipped, True


def _detect_g01_3_deformable_contact_response(
    nodes: np.ndarray,
    previous_nodes: np.ndarray,
    velocities: np.ndarray,
    hand_points: tuple[np.ndarray, ...],
) -> tuple[bool, dict[str, float | int | bool | str]]:
    """Detect localized physical nodal response near one real hand envelope."""

    points = np.asarray(nodes, dtype=np.float64)
    previous = np.asarray(previous_nodes, dtype=np.float64)
    speed_vectors = np.asarray(velocities, dtype=np.float64)
    hands = np.asarray(hand_points, dtype=np.float64).reshape(-1, 3)
    diagnostics: dict[str, float | int | bool | str] = {
        "response": False,
        "reason": "invalid_readback",
        "local_node_count": 0,
        "minimum_distance_m": math.inf,
        "local_max_displacement_m": 0.0,
        "local_max_speed_mps": 0.0,
        "background_displacement_m": 0.0,
        "background_speed_mps": 0.0,
        "displacement_excess_m": 0.0,
        "speed_excess_mps": 0.0,
    }
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or previous.shape != points.shape
        or speed_vectors.shape != points.shape
        or points.shape[0] < _G01_3_CONTACT_MIN_LOCAL_NODES
        or hands.size == 0
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(previous))
        or not np.all(np.isfinite(speed_vectors))
        or not np.all(np.isfinite(hands))
    ):
        return False, diagnostics

    distances = np.sqrt(np.min(cdist(points, hands, metric="sqeuclidean"), axis=1))
    minimum_distance = float(np.min(distances))
    local_indices = np.flatnonzero(
        distances <= _G01_3_CONTACT_LOCAL_RADIUS_M
    )
    diagnostics["minimum_distance_m"] = minimum_distance
    diagnostics["local_node_count"] = int(local_indices.size)
    if (
        minimum_distance > _G01_3_CONTACT_DISTANCE_M
        or local_indices.size < _G01_3_CONTACT_MIN_LOCAL_NODES
    ):
        diagnostics["reason"] = "hand_outside_contact_band"
        return False, diagnostics

    displacement = np.linalg.norm(points - previous, axis=1)
    speeds = np.linalg.norm(speed_vectors, axis=1)
    background_mask = np.ones(points.shape[0], dtype=bool)
    background_mask[local_indices] = False
    if int(np.count_nonzero(background_mask)) < _G01_3_CONTACT_MIN_LOCAL_NODES:
        background_mask[:] = True
    background_displacement = float(
        np.percentile(displacement[background_mask], 75.0)
    )
    background_speed = float(np.percentile(speeds[background_mask], 75.0))
    local_max_displacement = float(np.max(displacement[local_indices]))
    local_max_speed = float(np.max(speeds[local_indices]))
    displacement_excess = local_max_displacement - background_displacement
    speed_excess = local_max_speed - background_speed
    response = bool(
        (
            local_max_displacement >= _G01_3_CONTACT_MIN_DISPLACEMENT_M
            and displacement_excess
            >= _G01_3_CONTACT_MIN_DISPLACEMENT_EXCESS_M
        )
        or (
            local_max_speed >= _G01_3_CONTACT_MIN_SPEED_MPS
            and speed_excess >= _G01_3_CONTACT_MIN_SPEED_EXCESS_MPS
        )
    )
    diagnostics.update(
        {
            "response": response,
            "reason": "localized_nodal_response" if response else "background_only_motion",
            "local_max_displacement_m": local_max_displacement,
            "local_max_speed_mps": local_max_speed,
            "background_displacement_m": background_displacement,
            "background_speed_mps": background_speed,
            "displacement_excess_m": displacement_excess,
            "speed_excess_mps": speed_excess,
        }
    )
    return response, diagnostics


def _g01_5_hand_indentations(nodes, reference_nodes, hand_points) -> np.ndarray:
    """Estimate inward deformation near each hand, excluding rigid motion.

    Align the settled reference using nodes outside both hand regions. Require
    four nearby nodes to indent, so one noisy node cannot engage the grip.
    This is a deformation estimate in metres, not a force measurement.
    """
    result = np.zeros(2, dtype=np.float64)
    points = np.asarray(nodes, dtype=np.float64)
    reference = np.asarray(reference_nodes, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1:] != (3,)
            or reference.shape != points.shape
            or not np.isfinite(points).all() or not np.isfinite(reference).all()):
        return result
    masks = []
    for hands in hand_points:
        hands = np.asarray(hands, dtype=np.float64).reshape(-1, 3)
        mask = np.zeros(len(points), dtype=bool)
        if len(hands) and np.isfinite(hands).all():
            distances = np.sqrt(np.min(cdist(points, hands, metric="sqeuclidean"), axis=1))
            if distances.min() <= _G01_3_CONTACT_DISTANCE_M:
                mask = distances <= _G01_3_CONTACT_LOCAL_RADIUS_M
        masks.append(mask)
    background = ~np.logical_or.reduce(masks)
    if np.count_nonzero(background) < 4:
        return result
    source_center = reference[background].mean(axis=0)
    target_center = points[background].mean(axis=0)
    source = reference[background] - source_center
    target = points[background] - target_center
    u, singular, vt = np.linalg.svd(source.T @ target)
    if singular[-1] <= 1e-12:
        return result
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    aligned = (reference - source_center) @ rotation + target_center
    outward = aligned - aligned.mean(axis=0)
    outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-12)
    inward = np.maximum(0.0, -np.einsum("ij,ij->i", points - aligned, outward))
    for side, mask in enumerate(masks):
        if np.count_nonzero(mask) >= 4:
            result[side] = float(np.partition(inward[mask], -4)[-4])
    return result


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _clone_tensor(value: Any) -> Any:
    if hasattr(value, "torch"):
        value = value.torch
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    return value


class HandsOnG01DeformableAssetIsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Display three supplier assets and finish on a real hand/button touch."""

    _TASK_NAME = G01_TASK_NAME
    # G01 is an operator inspection scene, and its supplier deformables are
    # explicitly outside formal replay truth. Enter control immediately after
    # reset instead of repeatedly rejecting the freely moving cable.
    stable_recording_origin_gate_enabled = False

    def __init__(self, task_name: str = G01_TASK_NAME, **kwargs: Any):
        if task_name not in G01_TASK_NAMES:
            raise ValueError(
                f"G01 environment requires one of {G01_TASK_NAMES!r}, "
                f"got {task_name!r}"
            )
        self._g01_task_name = task_name
        self._g01_comparison_tets: np.ndarray | None = None
        self._g01_comparison_surface_ids: np.ndarray | None = None
        self._g01_asset_specs = g01_asset_specs(task_name)
        self._g01_pillow_asset_sha256 = (
            hashlib.sha256(self._g01_asset_specs[0].source_path.read_bytes()).hexdigest()
            if task_name in G01_COMPARISON_PROFILES else None
        )
        self._g01_asset_reference_paths: tuple[str, ...] = ()
        self._g01_button_contact_view: Any | None = None
        self._g01_contact_monitor_available = False
        self._g01_button_touching = False
        self._g01_button_contact_force = 0.0
        self._g01_soft_views: dict[str, tuple[Any, Any, Any, Any]] = {}
        self._g01_cable_body_aliases: tuple[str, ...] = ()
        self._g01_cable_observation_view: Any | None = None
        self._g01_geometry_sampled = False
        self._g01_runtime_asset_valid = np.zeros(
            len(self._g01_asset_specs), dtype=np.float64
        )
        self._g01_3_reset_layout = _G01_3_RESET_LAYOUT_CANONICAL.copy()
        self._g01_last_asset_centroids, self._g01_last_asset_aabbs = (
            self._fallback_asset_geometry()
        )
        self._g01_3_contact_latched = {"left": False, "right": False}
        self._g01_3_contact_response_diagnostics: dict[
            str, dict[str, float | int | bool | str]
        ] = {"left": {}, "right": {}}
        self._g01_3_previous_nodes: np.ndarray | None = None
        self._g01_3_reference_nodes: np.ndarray | None = None
        self._g01_5_grip_friction_active = False
        self._g01_5_max_indentation_m = 0.0
        self._g01_5_hand_indentations_m = np.zeros(2, dtype=np.float64)
        self._g01_3_lift_substeps = 0
        self._g01_3_lift_latched = False
        self._g01_3_contained_latched = False
        self._g01_3_release_substeps = 0
        self._g01_3_release_latched = False
        self._g01_3_stable_substeps = 0
        self._g01_3_stage = _G01_3_STAGE_WAITING_CONTACT
        self._g01_3_initial_state_report: dict[str, Any] = {}
        self._g01_3_last_metrics: dict[str, Any] = {}
        self._g01_3_reset_ordinal = 0
        # Unlike the inspection variants, G01-3 is an interaction task and
        # must retain the shared stable-origin gate.
        self.stable_recording_origin_gate_enabled = bool(
            task_name in G01_CARTON_TASK_NAMES
        )
        if task_name in _G01_CURRENT_PILLOW_TASK_NAMES:
            # Current-schema volume deformables can block indefinitely in
            # SimulationContext.stop() after tensor/contact views are active.
            # Collection runs in a process-owned worker, so use the base
            # runtime's established skip-cleanup shutdown path.
            self._fast_shutdown_on_close = True
        physics_hz = 120 if task_name in G01_COMPARISON_PROFILES else G01_PHYSICS_HZ
        kwargs.setdefault("physics_dt", 1.0 / physics_hz)
        super().__init__(task_name=task_name, **kwargs)

    def _configure_task_robot_cfg(self, robot_cfg: Any, side: str) -> Any:
        """Use a finite, load-bearing Wuji drive for the G01-3 pillow lift."""

        robot_cfg = super()._configure_task_robot_cfg(robot_cfg, side)
        if (
            self._g01_task_name not in G01_CARTON_TASK_NAMES
            or self._physics_backend != "physx"
        ):
            return robot_cfg
        hand_actuator = getattr(robot_cfg, "actuators", {}).get("hand")
        if hand_actuator is None:
            return robot_cfg
        # The default PhysX profile is 6 N / Kp=8 / Kd=.12.  It is adequate
        # for an unloaded inspection hand but stalls and slips under the
        # pillow's gravity.  Keep the stronger profile task-local and cap the
        # finger speed so a glove packet cannot become a piercing impulse.
        hand_actuator.effort_limit_sim = 15.0
        hand_actuator.stiffness = 20.0
        hand_actuator.damping = 0.80
        hand_actuator.velocity_limit_sim = 2.0
        return robot_cfg

    def _a06_hand_points(self, side: str) -> tuple[np.ndarray, ...]:
        """Read each comparison hand in one GPU transfer per physics step."""
        if self._g01_task_name in G01_COMPARISON_PROFILES:
            robot = self._robots.get(side)
            if robot is not None:
                poses = _to_numpy(robot.data.body_pose_w)[0]
                points = tuple(
                    np.asarray(poses[index, :3], dtype=np.float64).copy()
                    for index, name in enumerate(robot.body_names)
                    if ("palm" in str(name).lower() or "finger" in str(name).lower())
                    and np.all(np.isfinite(poses[index, :3]))
                )
                if points:
                    return points
        return super()._a06_hand_points(side)

    def _arm_pose_target_max_offset_m(self) -> float:
        if self._g01_task_name == G01_5_TASK_NAME:
            # The density-corrected latex pillow weighs 2.81 kg. The 35 mm
            # measured-palm offset starves its loaded grasp of drive preload.
            # Keep the shared joint slew, jump and workspace limits; only the
            # G01-5 contact servo may build up to 100 mm of position error.
            return 0.10
        return super()._arm_pose_target_max_offset_m()

    def _limit_arm_pose_target(
        self, side: str, robot, pose: np.ndarray
    ) -> np.ndarray:
        """Limit G01-3 arm targets before the hand enters the pillow core."""

        limited = super()._limit_arm_pose_target(side, robot, pose)
        if (
            self._g01_task_name != G01_3_TASK_NAME
            or not self._g01_soft_views
        ):
            return limited
        try:
            nodes, _velocities = self._g01_3_pillow_readback()
            current_matrix = self._robot_body_pose_matrix(
                robot, self._ee_body_indices[side]
            )
            current_pose = np.concatenate(
                [
                    current_matrix[:3, 3],
                    _matrix_to_quat_wxyz(current_matrix[:3, :3]),
                ]
            )
            hand_points = tuple(self._a06_hand_points(side))
            guarded, clipped = _g01_3_backtrack_poke_target_pose(
                current_pose,
                limited,
                hand_points,
                np.min(nodes, axis=0),
                np.max(nodes, axis=0),
            )
            if clipped and not np.allclose(
                guarded, limited, atol=1.0e-7, rtol=0.0
            ):
                message = (
                    f"G01-3 {side} arm target backtracked before deep "
                    "Pillow024 penetration"
                )
                if message not in self._isaac_warnings:
                    self._isaac_warnings.append(message)
                    print(f"[HandsOn/IsaacLab] {message}", flush=True)
            return guarded
        except Exception as exc:
            message = f"G01-3 {side} pillow target guard unavailable: {exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab][WARN] {message}", flush=True)
            # A malformed live readback must never be used to drive a hand;
            # hold the already-read, collision-safe target for this frame.
            return limited

    def _task_specific_isaac_kit_args(self) -> str | None:
        # G01 now uses the same current PhysX deformable pipeline as E01: a
        # volume-deformable pillow plus a foldable surface-deformable towel.
        return "--/persistent/physics/enableDeformableBeta=true"

    def _task_articulation_root_is_writable(self, path: str) -> bool:
        """Keep the world-fixed button base anchored while resetting its DOF."""

        normalized = str(path).rstrip("/")
        if normalized.endswith(_G01_BUTTON_ARTICULATION_PATH_SUFFIX):
            return False
        return super()._task_articulation_root_is_writable(path)

    def _task_reset_write_is_required(
        self,
        path: str,
        *,
        state_kind: Literal["articulation", "rigid_body"],
    ) -> bool:
        """Require every task-owned dynamic/kinematic reset write."""

        if self._g01_task_name in G01_CARTON_TASK_NAMES:
            return bool(
                state_kind == "rigid_body"
                and str(path).rstrip("/").endswith("/g01_3_carton")
            )
        if self._g01_task_name == G01_2_TASK_NAME:
            normalized = str(path).rstrip("/")
            if (
                state_kind == "rigid_body"
                and normalized in G01_CABLE_RIGID_BODY_PATHS
            ):
                return True
            if state_kind == "articulation" and normalized.endswith(
                _G01_BUTTON_ARTICULATION_PATH_SUFFIX
            ):
                return True
        return super()._task_reset_write_is_required(
            path,
            state_kind=state_kind,
        )

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        del sim_utils
        if self._stage is None:
            raise RuntimeError("G01 stage is unavailable before asset spawn")
        self._g01_asset_reference_paths = spawn_g01_deformable_assets(
            self._stage,
            self._g01_task_name,
        )
        if self._g01_task_name in G01_COMPARISON_PROFILES:
            mesh = self._stage.GetPrimAtPath(G01_2_PILLOW_SIM_MESH_PATH)
            self._g01_comparison_tets = np.asarray(
                mesh.GetAttribute("tetVertexIndices").Get(), dtype=np.int64
            )
            collider = self._stage.GetPrimAtPath(G01_PILLOW_BODY_PATH + "/collision_mesh")
            self._g01_comparison_surface_ids = np.unique(np.asarray(
                collider.GetAttribute("surfaceFaceVertexIndices").Get(), dtype=np.int64
            ))
        print(
            "[HandsOn/IsaacLab] G01 supplier USD references authored: "
            + ", ".join(self._g01_asset_reference_paths),
            flush=True,
        )

    def _configure_task_collision_safety(
        self, sim_utils: Any
    ) -> tuple[int, int]:
        bodies, colliders = super()._configure_task_collision_safety(sim_utils)
        if self._stage is None:
            return bodies, colliders
        if self._g01_task_name in G01_COMPARISON_PROFILES:
            # Keep the native complete-hand geometry and small contact margins.
            # The CCD flag is backend-dependent (Direct GPU rigid CCD is not
            # supported); contact robustness must be checked in the live solver.
            from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

            material_path = "/World/PhysicsMaterials/G01ComparisonCottonGrip"
            cfg = sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.8, restitution=0.0,
                friction_combine_mode="average", restitution_combine_mode="min",
            )
            cfg.func(material_path, cfg)
            material = UsdShade.Material(self._stage.GetPrimAtPath(material_path))
            hand_bodies, hand_colliders = 0, 0
            for side in ("Right", "Left"):
                path = f"/World/HandsOnRobots/{side}Robot"
                sim_utils.make_uninstanceable(path, stage=self._stage)
                root = self._stage.GetPrimAtPath(path)
                if not root.IsValid():
                    continue
                for prim in Usd.PrimRange(root):
                    name = str(prim.GetPath()).lower()
                    if "palm" not in name and "finger" not in name:
                        continue
                    is_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    is_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
                    if is_body:
                        api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                        api.CreateEnableSpeculativeCCDAttr().Set(True)
                        hand_bodies += 1
                    if is_collider:
                        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                        api.CreateContactOffsetAttr().Set(0.004)
                        api.CreateRestOffsetAttr().Set(0.0005)
                        hand_colliders += 1
                    if is_body or is_collider:
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                            materialPurpose="physics",
                        )
            self._g01_comparison_hand_collision_inventory = (hand_bodies, hand_colliders)
            if (hand_bodies, hand_colliders) != (42, 52):
                raise RuntimeError(
                    f"Incomplete comparison pillow hands: {hand_bodies} bodies, {hand_colliders} colliders"
                )
            return bodies + hand_bodies, colliders + hand_colliders
        if self._g01_task_name in G01_CARTON_TASK_NAMES:
            return bodies, colliders

        from pxr import PhysxSchema

        button_prim = self._find_rigid_body_prim("g01_success_button")
        if button_prim is None:
            self._add_task_state_warning(
                "Could not configure G01 green-button contact reporting"
            )
            return bodies, colliders
        if button_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            report_api = PhysxSchema.PhysxContactReportAPI.Get(
                self._stage,
                button_prim.GetPath(),
            )
        else:
            report_api = PhysxSchema.PhysxContactReportAPI.Apply(button_prim)
        report_api.CreateThresholdAttr().Set(0.0)
        return bodies + 1, colliders

    def _set_g01_5_grip_friction(self, active: bool) -> None:
        """Switch the foam pillow's deformable friction after indentation.

        Write the existing deformable material; do not attach or move nodes.
        Live PhysX grasp validation is separate from USD attribute readback.
        """
        if self._g01_task_name != G01_5_TASK_NAME:
            return
        if self._stage is None:
            self._g01_5_grip_friction_active = bool(active)
            return
        path = G01_2_PILLOW_MATERIAL_PATH
        prim = self._stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid():
            raise RuntimeError(f"G01-5 pillow material is unavailable: {path}")
        static_friction = (
            _G01_5_GRIP_STATIC_FRICTION
            if active else _G01_5_BASE_STATIC_FRICTION
        )
        dynamic_friction = (
            _G01_5_GRIP_DYNAMIC_FRICTION
            if active else _G01_5_BASE_DYNAMIC_FRICTION
        )
        for name, value in (
            ("omniphysics:staticFriction", static_friction),
            ("omniphysics:dynamicFriction", dynamic_friction),
        ):
            attr = prim.GetAttribute(name)
            if not attr.IsValid():
                raise RuntimeError(f"G01-5 pillow material lacks {name}")
            attr.Set(float(value))
        self._g01_5_grip_friction_active = bool(active)

    def _update_g01_5_grip_friction(self, nodes, metrics) -> None:
        # Independent of success-contact latches: a shallow first touch must
        # not prevent a later squeeze, and unloading must restore friction.
        self._g01_5_hand_indentations_m = _g01_5_hand_indentations(
            nodes, self._g01_3_reference_nodes,
            (metrics["left_hand_points"], metrics["right_hand_points"]),
        )
        self._g01_5_max_indentation_m = float(
            np.max(self._g01_5_hand_indentations_m)
        )
        active = self._g01_5_max_indentation_m >= _G01_5_MIN_INDENTATION_M
        if active != self._g01_5_grip_friction_active:
            self._set_g01_5_grip_friction(active)

    def _init_task_specific_runtime(self) -> None:
        if self._g01_task_name not in G01_CARTON_TASK_NAMES:
            self._init_g01_button_contact_view()
        self._init_g01_deformable_views()

    def _init_g01_button_contact_view(self) -> None:
        self._g01_button_contact_view = None
        self._g01_contact_monitor_available = False
        if not self._robots:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        hand_paths = self._c02_hand_body_paths()
        button_prim = self._find_rigid_body_prim("g01_success_button")
        if physics_sim_view is None or not hand_paths or button_prim is None:
            self._add_task_state_warning(
                "G01 physical hand/button contact reporting is unavailable"
            )
            return
        try:
            view = physics_sim_view.create_rigid_contact_view(
                str(button_prim.GetPath()),
                filter_patterns=hand_paths,
                max_contact_data_count=max(64, len(hand_paths) * 8),
            )
            if (
                view is not None
                and int(view.sensor_count) > 0
                and int(view.filter_count) > 0
            ):
                self._g01_button_contact_view = view
                self._g01_contact_monitor_available = True
            else:
                self._add_task_state_warning(
                    "G01 green-button contact view has no sensors or filters"
                )
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not initialize G01 green-button contact view: {exc}"
            )

    @staticmethod
    def _soft_get_positions(view: Any) -> Any:
        for name in (
            "get_sim_nodal_positions",
            "get_simulation_nodal_positions",
        ):
            method = getattr(view, name, None)
            if callable(method):
                return method()
        raise AttributeError("Volume-deformable position readback is unavailable")

    @staticmethod
    def _soft_get_velocities(view: Any) -> Any:
        for name in (
            "get_sim_nodal_velocities",
            "get_simulation_nodal_velocities",
        ):
            method = getattr(view, name, None)
            if callable(method):
                return method()
        raise AttributeError("Volume-deformable velocity readback is unavailable")

    @staticmethod
    def _soft_set(view: Any, names: tuple[str, ...], value: Any, indices: Any) -> None:
        for name in names:
            method = getattr(view, name, None)
            if not callable(method):
                continue
            try:
                method(value, indices=indices)
            except TypeError:
                method(value, indices)
            return
        raise AttributeError(f"Volume-deformable write is unavailable: {names}")

    def _init_g01_deformable_views(self) -> None:
        self._g01_soft_views = {}
        view_specs = g01_deformable_view_specs(self._g01_task_name)
        if not self._robots:
            if self._g01_task_name in _G01_CURRENT_PILLOW_TASK_NAMES:
                raise RuntimeError(
                    f"{self._g01_task_name} requires live robots before "
                    "deformable view setup"
                )
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_sim_view is None:
            if self._g01_task_name in _G01_CURRENT_PILLOW_TASK_NAMES:
                raise RuntimeError(
                    f"{self._g01_task_name} deformable simulation view is unavailable"
                )
            self._add_task_state_warning(
                "G01 deformable simulation view is unavailable"
            )
            return

        for name, (deformable_kind, prim_path) in view_specs.items():
            try:
                factory = getattr(
                    physics_sim_view,
                    f"create_{deformable_kind}_deformable_body_view",
                    None,
                )
                if not callable(factory):
                    raise RuntimeError(
                        f"missing_{deformable_kind}_deformable_tensor_factory"
                    )
                view = factory(prim_path)
                if view is None or int(view.count) != 1:
                    raise RuntimeError(
                        f"view_count={0 if view is None else int(view.count)}"
                    )
                positions = self._soft_get_positions(view)
                velocities = self._soft_get_velocities(view)
                nodes = _to_numpy(positions)
                if nodes.ndim != 3 or nodes.shape[0] != 1 or nodes.shape[-1] != 3:
                    raise RuntimeError(f"nodal_shape={nodes.shape}")
                nodal_velocities = _to_numpy(velocities)
                if nodal_velocities.shape != nodes.shape:
                    raise RuntimeError(
                        f"nodal_velocity_shape={nodal_velocities.shape}"
                    )
                if not np.all(np.isfinite(nodes)) or not np.all(
                    np.isfinite(nodal_velocities)
                ):
                    raise RuntimeError("non_finite_nodal_state")
                indices = self._torch.arange(
                    int(view.count),
                    dtype=self._torch.long,
                    device=self._isaac_device,
                )
                initial_positions = _clone_tensor(positions)
                initial_velocities = _clone_tensor(velocities)
                if self._g01_task_name in _G01_CURRENT_PILLOW_TASK_NAMES:
                    # SimulationContext.reset() instantiates/cooks the current
                    # deformable hierarchy before this view exists and may
                    # advance it by several gravity steps.  Authoritative
                    # G01-2 reset truth therefore comes from the composed
                    # bind/rest points, transformed into the world-space
                    # tensor convention, never from that post-cook snapshot.
                    initial_positions, initial_velocities = (
                        self._g01_2_rest_nodal_state(
                            name,
                            positions,
                        )
                    )
                    self._soft_set(
                        view,
                        (
                            "set_sim_nodal_positions",
                            "set_simulation_nodal_positions",
                        ),
                        initial_positions,
                        indices,
                    )
                    self._soft_set(
                        view,
                        (
                            "set_sim_nodal_velocities",
                            "set_simulation_nodal_velocities",
                        ),
                        initial_velocities,
                        indices,
                    )
                self._g01_soft_views[name] = (
                    view,
                    indices,
                    initial_positions,
                    initial_velocities,
                )
            except Exception as exc:
                if self._g01_task_name in _G01_CURRENT_PILLOW_TASK_NAMES:
                    raise RuntimeError(
                        f"Could not initialize required {self._g01_task_name} "
                        f"{deformable_kind} deformable {name}"
                    ) from exc
                self._add_task_state_warning(
                    f"Could not bind G01 {deformable_kind} deformable "
                    f"{name}: {exc}"
                )
        if self._g01_task_name in _G01_CURRENT_PILLOW_TASK_NAMES and set(
            self._g01_soft_views
        ) != set(view_specs):
            raise RuntimeError(
                f"{self._g01_task_name} deformable view inventory changed"
            )
        print(
            "[HandsOn/IsaacLab] G01 deformable views ready: "
            f"{','.join(sorted(self._g01_soft_views)) or 'none'}",
            flush=True,
        )

    def _g01_2_rest_nodal_state(
        self,
        name: str,
        position_template: Any,
    ) -> tuple[Any, Any]:
        """Build exact world-space bind/rest tensors for a G01-2 reset."""

        if self._stage is None:
            raise RuntimeError("G01-2 stage is unavailable for rest-state reset")
        if name == "Pillow024":
            prim_path = G01_2_PILLOW_SIM_MESH_PATH
            point_attribute_name = "omniphysics:restShapePoints"
        elif name == "Towl011":
            prim_path = G01_TOWEL_SURFACE_MESH_PATH
            point_attribute_name = "omniphysics:restShapePoints"
        else:
            raise RuntimeError(f"Unsupported G01-2 deformable {name!r}")

        from pxr import Gf, UsdGeom

        prim = self._stage.GetPrimAtPath(prim_path)
        if prim is None or not prim.IsValid():
            raise RuntimeError(
                f"G01-2 reset source prim did not compose: {prim_path}"
            )
        attribute = prim.GetAttribute(point_attribute_name)
        local_points = attribute.Get() if attribute.IsValid() else None
        if local_points is None or len(local_points) == 0:
            raise RuntimeError(
                f"G01-2 reset points are missing: {prim_path}.{point_attribute_name}"
            )
        if attribute.GetNumTimeSamples() != 0:
            raise RuntimeError(
                f"G01-2 reset points must be a default value: {prim_path}"
            )
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        world_points = np.asarray(
            [
                tuple(
                    transform.Transform(
                        Gf.Vec3d(
                            float(point[0]),
                            float(point[1]),
                            float(point[2]),
                        )
                    )
                )
                for point in local_points
            ],
            dtype=np.float64,
        )

        template = position_template
        if hasattr(template, "torch"):
            template = template.torch
        if tuple(template.shape) != (1, len(local_points), 3):
            raise RuntimeError(
                f"G01-2 {name} tensor shape {tuple(template.shape)} does not "
                f"match {len(local_points)} reset points"
            )
        initial_positions = self._torch.as_tensor(
            world_points,
            dtype=template.dtype,
            device=template.device,
        ).unsqueeze(0)
        initial_velocities = self._torch.zeros_like(initial_positions)
        if not bool(self._torch.isfinite(initial_positions).all().item()):
            raise RuntimeError(f"G01-2 {name} reset points are not finite")
        return initial_positions, initial_velocities

    def _capture_task_scene_state(self, task_mjcf: Any) -> None:
        super()._capture_task_scene_state(task_mjcf)
        self._require_g01_2_button_reset_view()
        self._capture_g01_cable_rigid_state()

    def _require_g01_2_button_reset_view(self) -> None:
        """Fail closed unless the physical button DOF participates in reset."""

        if self._g01_task_name != G01_2_TASK_NAME:
            return
        entry = self._task_dof_views.get(_G01_BUTTON_DOF_NAME)
        if entry is None:
            raise RuntimeError(
                "G01-2 success-button DOF reset view is missing"
            )
        button_view, dof_index, _reset_value = entry
        matches = [
            joint_positions
            for (
                path,
                view,
                _indices,
                _root_pose,
                _root_velocity,
                joint_positions,
                _joint_velocities,
            ) in self._task_reset_articulation_states
            if view is button_view
            and str(path).rstrip("/").endswith(
                _G01_BUTTON_ARTICULATION_PATH_SUFFIX
            )
        ]
        if len(matches) != 1 or not 0 <= int(dof_index) < int(
            matches[0].shape[-1]
        ):
            raise RuntimeError(
                "G01-2 success-button articulation reset state is incomplete"
            )

    def _capture_g01_cable_rigid_state(self) -> None:
        self._g01_cable_body_aliases = ()
        self._g01_cable_observation_view = None
        if self._g01_task_name in G01_CARTON_TASK_NAMES:
            return
        if not self._robots or self._stage is None:
            if self._g01_task_name == G01_2_TASK_NAME:
                raise RuntimeError(
                    "G01-2 requires a live stage and robots for cable reset"
                )
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_sim_view is None:
            if self._g01_task_name == G01_2_TASK_NAME:
                raise RuntimeError(
                    "G01-2 cable simulation view is unavailable"
                )
            return
        existing_paths = {
            path for path, *_rest in self._task_reset_body_states
        }
        aliases: list[str] = []
        for index, path in enumerate(G01_CABLE_RIGID_BODY_PATHS):
            prim = self._stage.GetPrimAtPath(path)
            if prim is None or not prim.IsValid():
                if self._g01_task_name == G01_2_TASK_NAME:
                    raise RuntimeError(
                        f"G01-2 cable rigid body is missing: {path}"
                    )
                self._add_task_state_warning(
                    f"G01 cable rigid body is missing: {path}"
                )
                continue
            alias = f"g01_powercable_body_{index:02d}"
            try:
                view = physics_sim_view.create_rigid_body_view(path)
                if view is None or int(view.count) != 1:
                    raise RuntimeError(
                        f"view_count={0 if view is None else int(view.count)}"
                    )
                indices = self._torch.arange(
                    int(view.count),
                    dtype=self._torch.long,
                    device=self._isaac_device,
                )
                if path not in existing_paths:
                    self._task_reset_body_states.append(
                        (
                            path,
                            view,
                            indices,
                            _clone_tensor(view.get_transforms()),
                            _clone_tensor(view.get_velocities()),
                        )
                    )
                    existing_paths.add(path)
                self._task_body_views[alias] = view
                aliases.append(alias)
            except Exception as exc:
                if self._g01_task_name == G01_2_TASK_NAME:
                    raise RuntimeError(
                        f"Could not snapshot required G01-2 cable body {path}"
                    ) from exc
                self._add_task_state_warning(
                    f"Could not snapshot G01 cable body {path}: {exc}"
                )
        self._g01_cable_body_aliases = tuple(aliases)
        if (
            self._g01_task_name == G01_2_TASK_NAME
            and len(aliases) != len(G01_CABLE_RIGID_BODY_PATHS)
        ):
            raise RuntimeError(
                "G01-2 requires reset views for every Powercable001 body"
            )
        try:
            observation_view = physics_sim_view.create_rigid_body_view(
                [
                    G01_CABLE_RIGID_BODY_PATHS[0].rsplit("_", 1)[0] + "_*",
                    *G01_CABLE_RIGID_BODY_PATHS[-2:],
                ]
            )
            if (
                observation_view is None
                or int(observation_view.count)
                != len(G01_CABLE_RIGID_BODY_PATHS)
            ):
                raise RuntimeError(
                    "view_count="
                    f"{0 if observation_view is None else int(observation_view.count)}"
                )
            self._g01_cable_observation_view = observation_view
        except Exception as exc:
            if self._g01_task_name == G01_2_TASK_NAME:
                raise RuntimeError(
                    "Could not build required grouped G01-2 cable view"
                ) from exc
            self._add_task_state_warning(
                f"Could not build grouped G01 cable observation view: {exc}"
            )
        print(
            "[HandsOn/IsaacLab] G01 cable reset views ready: "
            f"{len(aliases)}/{len(G01_CABLE_RIGID_BODY_PATHS)}, "
            "observation_view="
            f"{'grouped' if self._g01_cable_observation_view is not None else 'fallback'}",
            flush=True,
        )

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        # The external USD placement wrappers are authored against the fixed
        # 0.89 m support plane, so table-height randomization is intentionally
        # disabled for this inspection task.
        self.delta_h[:] = 0.0
        self._g01_button_touching = False
        self._g01_button_contact_force = 0.0
        self._g01_geometry_sampled = False
        if self._g01_task_name in G01_CARTON_TASK_NAMES:
            self._g01_3_reset_ordinal += 1
            self._g01_3_reset_layout = self._sample_g01_3_reset_layout()
            self._g01_3_contact_latched = {"left": False, "right": False}
            self._g01_3_contact_response_diagnostics = {
                "left": {},
                "right": {},
            }
            self._g01_3_previous_nodes = None
            self._g01_3_reference_nodes = None
            self._g01_5_grip_friction_active = False
            self._g01_5_max_indentation_m = 0.0
            self._g01_5_hand_indentations_m = np.zeros(2, dtype=np.float64)
            self._set_g01_5_grip_friction(False)
            self._g01_3_lift_substeps = 0
            self._g01_3_lift_latched = False
            self._g01_3_contained_latched = False
            self._g01_3_release_substeps = 0
            self._g01_3_release_latched = False
            self._g01_3_stable_substeps = 0
            self._g01_3_stage = _G01_3_STAGE_WAITING_CONTACT
            self._g01_3_initial_state_report = {}
            self._g01_3_last_metrics = {}

    def _sample_g01_3_reset_layout(self) -> np.ndarray:
        if not self.randomize:
            return _G01_3_RESET_LAYOUT_CANONICAL.copy()
        pillow_jitter = (
            _G01_5_PILLOW_XY_JITTER_M if self._g01_task_name == G01_5_TASK_NAME
            else _G01_3_PILLOW_XY_JITTER_M
        )
        carton_jitter = (
            _G01_5_CARTON_XY_JITTER_M if self._g01_task_name == G01_5_TASK_NAME
            else _G01_3_CARTON_XY_JITTER_M
        )
        # The carton nearly fills table X beside the lowered hands; sample
        # its feasible near-axis headings directly to avoid excessive retries.
        carton_yaw_jitter = (
            0.5 if self._g01_task_name == G01_5_TASK_NAME
            else _G01_3_YAW_JITTER_DEG
        )
        max_attempts = (
            2048 if self._g01_task_name == G01_5_TASK_NAME
            else _G01_3_RESET_MAX_ATTEMPTS
        )
        for _ in range(max_attempts):
            pillow_xy = _G01_3_RESET_LAYOUT_CANONICAL[:2] + self._random.uniform(
                -pillow_jitter, pillow_jitter,
            )
            carton_xy = _G01_3_RESET_LAYOUT_CANONICAL[3:5] + self._random.uniform(
                -carton_jitter, carton_jitter,
            )
            pillow_yaw = 180.0 * int(self._random.randint(0, 2)) + float(
                self._random.uniform(-_G01_3_YAW_JITTER_DEG, _G01_3_YAW_JITTER_DEG)
            )
            carton_yaw = 180.0 * int(self._random.randint(0, 2)) + float(
                self._random.uniform(-carton_yaw_jitter, carton_yaw_jitter)
            )
            candidate = np.asarray(
                [
                    pillow_xy[0],
                    pillow_xy[1],
                    pillow_yaw,
                    carton_xy[0],
                    carton_xy[1],
                    carton_yaw,
                ],
                dtype=np.float64,
            )
            if (
                _g01_3_reset_layout_margin(candidate, self._g01_task_name)
                >= _G01_3_RESET_MIN_MARGIN_M
            ):
                return candidate
        raise RuntimeError(
            "Could not sample a supported, separated, dual-arm-reachable "
            f"{self._g01_task_name} reset layout in {max_attempts} attempts"
        )

    def _apply_task_pose_randomization(self) -> None:
        """Apply the audited G01-3 carton reset pose before action zero."""

        if self._g01_task_name not in G01_CARTON_TASK_NAMES or not self._robots:
            return None
        matches = [
            entry
            for entry in self._task_reset_body_states
            if str(entry[0]).rstrip("/").endswith("/g01_3_carton")
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "G01-3 reset requires exactly one writable BC004 carton root, "
                f"found {len(matches)}"
            )
        path, view, indices, root_pose, root_velocity = matches[0]
        root_pose_np = np.asarray(_to_numpy(root_pose), dtype=np.float64).reshape(
            -1, 7
        )
        desired_xy = self._g01_3_reset_layout[3:5]
        randomized_pose = _randomized_physx_pose(
            root_pose_np,
            xy_offset=desired_xy - root_pose_np[0, :2],
            z_offset=0.0,
            yaw_degrees=float(self._g01_3_reset_layout[5]),
        )
        pose_tensor = self._torch.as_tensor(
            randomized_pose,
            dtype=root_pose.dtype,
            device=root_pose.device,
        )
        self._write_rigid_reset_state(
            path,
            view,
            indices,
            pose_tensor,
            self._torch.zeros_like(root_velocity),
        )
        return None

    def _reset_scene_state(self) -> None:
        super()._reset_scene_state()
        if not self._robots:
            return
        if self._g01_task_name == G01_5_TASK_NAME and not self._direct_hand_control:
            # Native IK targets use the palm frame, not the source controller
            # home frame. Holding the reset command must leave both arms still.
            for side, robot in self._robots.items():
                robot.update(0.0)
                matrix = self._native_end_effector_pose_matrix(
                    side, getattr(self, f"_{side}_tcp_pose"),
                )
                pose = self._pose_matrix_to_vector(matrix)
                if not np.isfinite(pose).all():
                    raise RuntimeError(f"G01-5 reset returned a nonfinite {side} palm pose")
                setattr(self, f"_{side}_tcp_pose", pose.copy())
        for name, (view, indices, positions, velocities) in (
            self._g01_soft_views.items()
        ):
            try:
                reset_positions = positions
                reset_velocities = velocities
                if (
                    self._g01_task_name in G01_CARTON_TASK_NAMES
                    and name == "Pillow024"
                ):
                    nodes = np.asarray(
                        _to_numpy(positions), dtype=np.float64
                    ).copy()
                    # Every pillow variant has its own authored placement and
                    # TetGen node distribution. Rotate about that rest-state
                    # centroid, then place it at the sampled centroid. Reusing
                    # G01-3's origin leaves the CC0 pillow 28.6 mm off target.
                    canonical_xy = np.mean(nodes[0, :, :2], axis=0)
                    yaw = math.radians(float(self._g01_3_reset_layout[2]))
                    rotation = np.asarray(
                        [
                            [math.cos(yaw), -math.sin(yaw)],
                            [math.sin(yaw), math.cos(yaw)],
                        ],
                        dtype=np.float64,
                    )
                    nodes[..., :2] = (
                        (nodes[..., :2] - canonical_xy) @ rotation.T
                        + self._g01_3_reset_layout[:2]
                    )
                    template = (
                        positions.torch
                        if hasattr(positions, "torch")
                        else positions
                    )
                    reset_positions = self._torch.as_tensor(
                        nodes,
                        dtype=template.dtype,
                        device=template.device,
                    )
                    reset_velocities = self._torch.zeros_like(reset_positions)
                self._soft_set(
                    view,
                    (
                        "set_sim_nodal_positions",
                        "set_simulation_nodal_positions",
                    ),
                    reset_positions,
                    indices,
                )
                self._soft_set(
                    view,
                    (
                        "set_sim_nodal_velocities",
                        "set_simulation_nodal_velocities",
                    ),
                    reset_velocities,
                    indices,
                )
                if self._g01_task_name == G01_5_TASK_NAME and name == "Pillow024":
                    self._record_authored_deformable_reset_write(
                        "g01_pillow", position=reset_positions[0], velocity=reset_velocities[0],
                    )
            except Exception as exc:
                if self._g01_task_name in _G01_CURRENT_PILLOW_TASK_NAMES:
                    raise RuntimeError(
                        f"Could not restore required {self._g01_task_name} "
                        f"deformable {name}"
                    ) from exc
                self._add_task_state_warning(
                    f"Could not restore G01 deformable {name}: {exc}"
                )

    def _fallback_asset_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        centroids: list[np.ndarray] = []
        bounds: list[np.ndarray] = []
        for spec in self._g01_asset_specs:
            lower = np.asarray(spec.local_bounds_min, dtype=np.float64)
            upper = np.asarray(spec.local_bounds_max, dtype=np.float64)
            corners = np.asarray(
                [
                    [x, y, z]
                    for x in (lower[0], upper[0])
                    for y in (lower[1], upper[1])
                    for z in (lower[2], upper[2])
                ],
                dtype=np.float64,
            )
            corners *= float(spec.uniform_scale)
            yaw_delta = (
                float(self._g01_3_reset_layout[2])
                if self._g01_task_name in G01_CARTON_TASK_NAMES
                and spec.name == "Pillow024"
                else 0.0
            )
            angle = math.radians(spec.rotation_z_degrees + yaw_delta)
            rotation = np.asarray(
                [
                    [math.cos(angle), -math.sin(angle), 0.0],
                    [math.sin(angle), math.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            translation = np.asarray(spec.translation, dtype=np.float64)
            if (
                self._g01_task_name in G01_CARTON_TASK_NAMES
                and spec.name == "Pillow024"
            ):
                translation = translation.copy()
                translation[:2] = self._g01_3_reset_layout[:2]
            world = corners @ rotation.T + translation
            world_min = np.min(world, axis=0)
            world_max = np.max(world, axis=0)
            centroids.append(0.5 * (world_min + world_max))
            bounds.append(np.concatenate([world_min, world_max]))
        return np.asarray(centroids), np.asarray(bounds)

    def _asset_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centroids, bounds = self._fallback_asset_geometry()
        valid = (
            np.ones(len(self._g01_asset_specs), dtype=np.float64)
            if not self._robots
            else np.zeros(len(self._g01_asset_specs), dtype=np.float64)
        )
        name_to_index = {
            spec.name: index for index, spec in enumerate(self._g01_asset_specs)
        }
        for name, (view, _indices, _positions, _velocities) in (
            self._g01_soft_views.items()
        ):
            try:
                nodes = _to_numpy(self._soft_get_positions(view))[0]
                velocities = _to_numpy(self._soft_get_velocities(view))[0]
                if nodes.ndim != 2 or nodes.shape[1:] != (3,):
                    continue
                if velocities.shape != nodes.shape:
                    continue
                if not np.all(np.isfinite(nodes)) or not np.all(
                    np.isfinite(velocities)
                ):
                    continue
                geometry_nodes = self._g01_pillow_contact_surface(nodes) if name == "Pillow024" else nodes
                lower = np.min(geometry_nodes, axis=0)
                upper = np.max(geometry_nodes, axis=0)
                if self._g01_task_name in _G01_CURRENT_PILLOW_TASK_NAMES:
                    expected_count = (
                        self._pillow_node_count() if name == "Pillow024"
                        else _G01_2_DEFORMABLE_NODE_COUNTS.get(name)
                    )
                    speeds = np.linalg.norm(velocities, axis=1)
                    if expected_count is None or nodes.shape != (
                        expected_count,
                        3,
                    ):
                        continue
                    if np.any(lower < _G01_2_VALID_WORLD_MIN) or np.any(
                        upper > _G01_2_VALID_WORLD_MAX
                    ):
                        continue
                    if np.any(
                        upper - lower > _G01_2_MAX_DEFORMABLE_SPAN_METERS
                    ):
                        continue
                    if (
                        speeds.size == 0
                        or float(np.max(speeds)) > _G01_2_MAX_NODAL_SPEED_MPS
                    ):
                        continue
                index = name_to_index[name]
                centroids[index] = np.mean(nodes, axis=0)
                bounds[index] = np.concatenate([lower, upper])
                valid[index] = 1.0
            except Exception:
                continue

        cable_points: list[np.ndarray] = []
        grouped_view = self._g01_cable_observation_view
        if grouped_view is not None:
            try:
                transforms = np.asarray(
                    _to_numpy(grouped_view.get_transforms()),
                    dtype=np.float64,
                )
                if transforms.ndim == 2 and transforms.shape[1] >= 3:
                    cable_points.extend(
                        point.copy()
                        for point in transforms[:, :3]
                        if np.all(np.isfinite(point))
                    )
            except Exception:
                cable_points = []
        if len(cable_points) != len(G01_CABLE_RIGID_BODY_PATHS):
            cable_points = []
            for alias in self._g01_cable_body_aliases:
                view = self._task_body_views.get(alias)
                if view is None:
                    continue
                try:
                    transforms = _to_numpy(view.get_transforms())
                    point = np.asarray(transforms[0, :3], dtype=np.float64)
                    if np.all(np.isfinite(point)):
                        cable_points.append(point)
                except Exception:
                    continue
        if cable_points:
            points = np.asarray(cable_points, dtype=np.float64)
            index = name_to_index["Powercable001"]
            lower = np.min(points, axis=0) - 0.02
            upper = np.max(points, axis=0) + 0.02
            centroids[index] = np.mean(points, axis=0)
            bounds[index] = np.concatenate([lower, upper])
            cable_valid = len(cable_points) == len(G01_CABLE_RIGID_BODY_PATHS)
            if self._g01_task_name == G01_2_TASK_NAME:
                cable_valid = bool(
                    cable_valid
                    and np.all(lower >= _G01_2_VALID_WORLD_MIN)
                    and np.all(upper <= _G01_2_VALID_WORLD_MAX)
                )
            valid[index] = float(cable_valid)
        self._g01_last_asset_centroids = centroids
        self._g01_last_asset_aabbs = bounds
        self._g01_runtime_asset_valid = valid
        self._g01_geometry_sampled = True
        return centroids, bounds, valid

    def _pillow_node_count(self) -> int:
        return (
            G01_COMPARISON_NODE_COUNTS[self._g01_task_name]
            if self._g01_task_name in G01_COMPARISON_PROFILES
            else G01_2_PILLOW_SIM_POINT_COUNT
        )

    def _g01_3_pillow_readback(self) -> tuple[np.ndarray, np.ndarray]:
        entry = self._g01_soft_views.get("Pillow024")
        if entry is None:
            raise RuntimeError("G01-3 Pillow024-v2 deformable view is unavailable")
        view = entry[0]
        nodes = np.asarray(
            _to_numpy(self._soft_get_positions(view))[0], dtype=np.float64
        )
        velocities = np.asarray(
            _to_numpy(self._soft_get_velocities(view))[0], dtype=np.float64
        )
        if (
            nodes.shape != (self._pillow_node_count(), 3)
            or velocities.shape != nodes.shape
            or not np.all(np.isfinite(nodes))
            or not np.all(np.isfinite(velocities))
        ):
            raise RuntimeError(
                "G01-3 Pillow024-v2 physical readback is invalid: "
                f"nodes={nodes.shape}, velocities={velocities.shape}"
            )
        return nodes, velocities

    def _g01_3_carton_inner_world_bounds(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        carton_pose = self._physics_body_pose("g01_3_carton")
        if carton_pose is None:
            if self._robots:
                raise RuntimeError(
                    "G01-3 live BC004 carton pose readback is unavailable"
                )
            carton_pose = _g01_3_expected_carton_pose_wxyz(
                self._g01_3_reset_layout
            )
        carton_pose = np.asarray(carton_pose, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(carton_pose)):
            raise RuntimeError("G01-3 carton pose readback is non-finite")
        lower, upper = _g01_3_world_aabb(
            G01_3_CARTON_INNER_LOCAL_MIN,
            G01_3_CARTON_INNER_LOCAL_MAX,
            carton_pose[:3],
            math.degrees(
                math.atan2(
                    _quat_wxyz_to_matrix(carton_pose[3:7])[1, 0],
                    _quat_wxyz_to_matrix(carton_pose[3:7])[0, 0],
                )
            ),
        )
        return lower, upper, carton_pose

    def _g01_pillow_contact_surface(self, simulation_nodes: np.ndarray) -> np.ndarray:
        """Geometry predicates use the actual collision skin, not solver cells."""
        if self._g01_comparison_surface_ids is None:
            return simulation_nodes
        entry = self._g01_soft_views.get("Pillow024")
        if entry is None:
            return simulation_nodes
        points = np.asarray(_to_numpy(entry[0].get_collision_nodal_positions())[0])
        if not np.isfinite(points).all():
            raise RuntimeError("Comparison pillow collision skin is non-finite")
        return points[self._g01_comparison_surface_ids]

    @staticmethod
    def _g01_3_min_hand_clearance(
        nodes: np.ndarray,
        hand_points: tuple[np.ndarray, ...],
    ) -> float:
        if not hand_points:
            return -math.inf
        hands = np.asarray(hand_points, dtype=np.float64).reshape(-1, 3)
        if hands.size == 0 or not np.all(np.isfinite(hands)):
            return -math.inf
        center_distance = math.sqrt(float(np.min(cdist(
            np.asarray(nodes, dtype=np.float64), hands, metric="sqeuclidean"
        ))))
        # Body origins alone overstate clearance.  Subtract a conservative
        # cover radius for every palm/finger body so the complete physical
        # hand envelope, rather than a TCP-only proxy, must clear the pillow.
        return center_distance - _G01_3_HAND_BODY_COVER_RADIUS_M

    def _g01_3_measure_state(
        self,
        nodes: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if nodes is None or velocities is None:
            if "Pillow024" not in self._g01_soft_views and not self._robots:
                _centroids, bounds = self._fallback_asset_geometry()
                lower, upper = bounds[0, :3], bounds[0, 3:]
                nodes = np.asarray(
                    [
                        [x, y, z]
                        for x in (lower[0], upper[0])
                        for y in (lower[1], upper[1])
                        for z in (lower[2], upper[2])
                    ],
                    dtype=np.float64,
                )
                velocities = np.zeros_like(nodes)
            else:
                nodes, velocities = self._g01_3_pillow_readback()
        nodes = np.asarray(nodes, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        geometry_nodes = self._g01_pillow_contact_surface(nodes)
        lower = np.min(geometry_nodes, axis=0)
        upper = np.max(geometry_nodes, axis=0)
        centroid = np.mean(nodes, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        carton_lower, carton_upper, carton_pose = (
            self._g01_3_carton_inner_world_bounds()
        )
        carton_rotation = _quat_wxyz_to_matrix(carton_pose[3:7])
        carton_local_nodes = (geometry_nodes - carton_pose[:3]) @ carton_rotation
        containment_lower = np.asarray(
            G01_3_CARTON_INNER_LOCAL_MIN, dtype=np.float64
        ) - _G01_3_CONTAINMENT_TOLERANCE_M
        containment_upper = np.asarray(
            G01_3_CARTON_INNER_LOCAL_MAX, dtype=np.float64
        ) + _G01_3_CONTAINMENT_TOLERANCE_M
        containment_upper[2] = (
            G01_3_CARTON_INNER_LOCAL_MAX[2] - _G01_3_RIM_CLEARANCE_M
        )
        if self._g01_task_name == G01_5_TASK_NAME:
            containment_upper[2] = G01_3_CARTON_INNER_LOCAL_MAX[2]
        contained_mask = np.all(
            (carton_local_nodes >= containment_lower)
            & (carton_local_nodes <= containment_upper),
            axis=1,
        )
        containment_fraction = float(np.mean(contained_mask))
        contained_now = containment_fraction >= _G01_3_CONTAINMENT_FRACTION
        if self._g01_task_name == G01_5_TASK_NAME:
            # Raised corners may extend past the opening, but the portion
            # below the rim must remain inside the actual walls. Applying
            # the footprint gate to every node contradicted that task rule.
            below_rim = carton_local_nodes[:, 2] <= containment_upper[2]
            horizontal_inside = np.all(
                (carton_local_nodes[:, :2] >= containment_lower[:2])
                & (carton_local_nodes[:, :2] <= containment_upper[:2]), axis=1,
            )
            horizontal_fraction = (
                float(np.mean(horizontal_inside[below_rim]))
                if np.any(below_rim) else 0.0
            )
            floor_height = G01_3_CARTON_INNER_LOCAL_MIN[2]
            contained_now = bool(
                containment_fraction >= _G01_5_CONTAINMENT_FRACTION
                and horizontal_fraction >= _G01_3_CONTAINMENT_FRACTION
                and np.min(carton_local_nodes[:, 2]) >= containment_lower[2]
                and np.quantile(carton_local_nodes[:, 2], 0.05)
                <= floor_height + _G01_5_FLOOR_SUPPORT_GAP_M
            )
        table_top = float(self._current_table_top_z())
        left_points = tuple(self._a06_hand_points("left"))
        right_points = tuple(self._a06_hand_points("right"))
        left_clearance = self._g01_3_min_hand_clearance(geometry_nodes, left_points)
        right_clearance = self._g01_3_min_hand_clearance(geometry_nodes, right_points)
        inverted_tets = 0
        if self._g01_comparison_tets is not None:
            tetrahedra = nodes[self._g01_comparison_tets]
            edges = tetrahedra[:, 1:] - tetrahedra[:, :1]
            signed_six_volumes = np.einsum(
                "ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2])
            )
            inverted_tets = int(np.count_nonzero(signed_six_volumes <= 0.0))
        return {
            "nodes": nodes,
            "velocities": velocities,
            "pillow_centroid": centroid,
            "pillow_aabb": np.concatenate([lower, upper]),
            "pillow_min_z_m": float(lower[2]),
            "pillow_max_z_m": float(upper[2]),
            "pillow_inverted_tet_count": inverted_tets,
            "pillow_lift_clearance_m": float(lower[2] - table_top),
            "pillow_rms_speed_mps": float(np.sqrt(np.mean(speeds * speeds))),
            "pillow_max_speed_mps": float(np.max(speeds)),
            "pillow_speed_p99_mps": float(np.quantile(speeds, 0.99)),
            "pillow_speed_p999_mps": float(np.quantile(speeds, 0.999)),
            "pillow_nodes_above_max_speed": int(
                np.count_nonzero(speeds > _G01_3_STABLE_MAX_SPEED_MPS)
            ),
            "pillow_speed_outlier_fraction": float(
                np.mean(speeds > _G01_3_STABLE_MAX_SPEED_MPS)
            ),
            "carton_pose": carton_pose,
            "carton_inner_bounds": np.concatenate(
                [carton_lower, carton_upper]
            ),
            "containment_fraction": containment_fraction,
            "contained_now": bool(contained_now),
            "left_hand_clearance_m": float(left_clearance),
            "right_hand_clearance_m": float(right_clearance),
            "left_hand_points": left_points,
            "right_hand_points": right_points,
            "both_released_now": bool(
                left_clearance >= _G01_3_RELEASE_CLEARANCE_M
                and right_clearance >= _G01_3_RELEASE_CLEARANCE_M
            ),
            "stable_now": bool(
                speeds.size > 0
                and inverted_tets == 0
                and float(np.sqrt(np.mean(speeds * speeds)))
                <= _G01_3_STABLE_RMS_SPEED_MPS
                and float(np.quantile(speeds, 0.99))
                <= _G01_3_STABLE_MAX_SPEED_MPS
                and float(
                    np.mean(speeds > _G01_3_STABLE_MAX_SPEED_MPS)
                )
                <= _G01_3_STABLE_OUTLIER_FRACTION
            ),
            "table_top_z_m": table_top,
        }

    def _after_task_physics_substep(self) -> None:
        super()._after_task_physics_substep()
        if self._g01_task_name not in G01_CARTON_TASK_NAMES:
            return
        nodes, velocities = self._g01_3_pillow_readback()
        metrics = self._g01_3_measure_state(nodes, velocities)
        previous = self._g01_3_previous_nodes
        if self._g01_task_name == G01_5_TASK_NAME and previous is not None:
            # PhysX damping can make reported velocities small before the
            # shape stops moving. Released success must also see still nodes.
            measured_speeds = np.linalg.norm(nodes - previous, axis=1) / self._sim_dt
            metrics["stable_now"] = bool(
                metrics["stable_now"]
                and np.sqrt(np.mean(measured_speeds ** 2)) <= _G01_3_STABLE_RMS_SPEED_MPS
                and np.quantile(measured_speeds, 0.99) <= _G01_3_STABLE_MAX_SPEED_MPS
            )
        on_table_contact_band = bool(
            metrics["pillow_min_z_m"]
            <= metrics["table_top_z_m"] + 0.020
        )
        if previous is not None and on_table_contact_band:
            for side in ("left", "right"):
                if self._g01_3_contact_latched[side]:
                    continue
                response, diagnostics = _detect_g01_3_deformable_contact_response(
                    nodes,
                    previous,
                    velocities,
                    metrics[f"{side}_hand_points"],
                )
                self._g01_3_contact_response_diagnostics[side] = diagnostics
                if response:
                    self._g01_3_contact_latched[side] = True
        self._g01_3_previous_nodes = nodes.copy()
        if self._g01_task_name == G01_5_TASK_NAME:
            self._update_g01_5_grip_friction(nodes, metrics)
            # A slow squeeze can stay below every per-step response threshold
            # while accumulating a real indentation. Use the same four-node,
            # rigid-motion-invariant physical witness as the foam grip. The
            # second hand may establish its grasp after the pillow leaves the
            # table; proximity alone still cannot advance either contact latch.
            for side, indentation in zip(
                ("left", "right"), self._g01_5_hand_indentations_m
            ):
                if (
                    not self._g01_3_contact_latched[side]
                    and indentation >= _G01_5_MIN_INDENTATION_M
                ):
                    self._g01_3_contact_latched[side] = True
                    self._g01_3_contact_response_diagnostics[side] = {
                        "response": True,
                        "reason": "localized_physical_indentation",
                        "indentation_m": float(indentation),
                    }

        both_contacted = all(self._g01_3_contact_latched.values())
        if both_contacted:
            self._g01_3_stage = max(
                self._g01_3_stage, _G01_3_STAGE_BOTH_CONTACTED
            )
        if both_contacted and not self._g01_3_lift_latched:
            if metrics["pillow_lift_clearance_m"] >= _G01_3_LIFT_CLEARANCE_M:
                self._g01_3_lift_substeps += 1
            else:
                self._g01_3_lift_substeps = 0
            if self._g01_3_lift_substeps >= _G01_3_LIFT_DEBOUNCE_SUBSTEPS:
                self._g01_3_lift_latched = True
                self._g01_3_stage = _G01_3_STAGE_LIFTED

        if self._g01_3_lift_latched and metrics["contained_now"]:
            self._g01_3_contained_latched = True
            self._g01_3_stage = max(
                self._g01_3_stage, _G01_3_STAGE_CONTAINED
            )
        release_was_latched = self._g01_3_release_latched
        if self._g01_3_contained_latched and not self._g01_3_release_latched:
            if metrics["contained_now"] and metrics["both_released_now"]:
                self._g01_3_release_substeps += 1
            else:
                self._g01_3_release_substeps = 0
            if self._g01_3_release_substeps >= _G01_3_RELEASE_DEBOUNCE_SUBSTEPS:
                self._g01_3_release_latched = True
                self._g01_3_stage = _G01_3_STAGE_RELEASED

        stable_required = max(
            1,
            int(
                math.ceil(
                    _G01_3_STABLE_SECONDS / max(float(self._sim_dt), 1.0e-6)
                )
            ),
        )
        if release_was_latched and self._g01_3_stage != _G01_3_STAGE_COMPLETE:
            if (
                metrics["contained_now"] and metrics["stable_now"]
                and (
                    self._g01_task_name != G01_5_TASK_NAME
                    or metrics["both_released_now"]
                )
            ):
                self._g01_3_stable_substeps += 1
            else:
                self._g01_3_stable_substeps = 0
            if self._g01_3_stable_substeps >= stable_required:
                self._g01_3_stable_substeps = stable_required
                self._g01_3_stage = _G01_3_STAGE_COMPLETE
        metrics.update(
            {
                "left_contact_latched": bool(
                    self._g01_3_contact_latched["left"]
                ),
                "right_contact_latched": bool(
                    self._g01_3_contact_latched["right"]
                ),
                "lift_latched": bool(self._g01_3_lift_latched),
                "contained_latched": bool(self._g01_3_contained_latched),
                "release_latched": bool(self._g01_3_release_latched),
                "stable_substeps": int(self._g01_3_stable_substeps),
                "stable_substeps_required": int(stable_required),
                "stage": int(self._g01_3_stage),
                "success": bool(
                    self._g01_3_stage == _G01_3_STAGE_COMPLETE
                ),
            }
        )
        self._g01_3_last_metrics = metrics

    def _read_button_contact(self) -> tuple[float, bool]:
        view = self._g01_button_contact_view
        available = bool(
            self._g01_contact_monitor_available and view is not None
        )
        force = self._c04_contact_force(view) if available else 0.0
        return float(force), available

    def _settle_g01_3_initial_state_or_raise(self) -> dict[str, Any]:
        """Advance only natural physics until the supported pillow is still."""

        if self._sim is None:
            raise RuntimeError("G01-3 initial settling requires a live simulator")
        required = _G01_3_INITIAL_SETTLE_CONSECUTIVE_SUBSTEPS
        check_nodal_drift = self._g01_task_name == G01_5_TASK_NAME
        if check_nodal_drift:
            required = max(required, 1 + int(math.ceil(0.5 / float(self._sim_dt))))
        timeout_seconds = (
            10.0 if check_nodal_drift else _G01_3_INITIAL_SETTLE_TIMEOUT_SECONDS
        )
        maximum_steps = max(
            required,
            int(
                math.ceil(
                    timeout_seconds
                    / max(float(self._sim_dt), 1.0e-6)
                )
            ),
        )
        consecutive = 0
        window_nodes = None
        window_drift = 0.0
        last_metrics: dict[str, Any] = {}
        for step_index in range(maximum_steps + 1):
            nodes, velocities = self._g01_3_pillow_readback()
            last_metrics = self._g01_3_measure_state(nodes, velocities)
            support_gap = float(
                last_metrics["pillow_min_z_m"]
                - last_metrics["table_top_z_m"]
            )
            stable = bool(
                -0.006 <= support_gap <= 0.025
                and last_metrics.get("pillow_inverted_tet_count", 0) == 0
                and last_metrics["pillow_rms_speed_mps"]
                <= _G01_3_STABLE_RMS_SPEED_MPS
                and last_metrics["pillow_speed_p99_mps"]
                <= _G01_3_STABLE_MAX_SPEED_MPS
                and last_metrics["pillow_speed_outlier_fraction"]
                <= _G01_3_STABLE_OUTLIER_FRACTION
            )
            if check_nodal_drift:
                if window_nodes is None:
                    window_nodes = nodes.copy()
                window_drift = float(
                    np.max(np.linalg.norm(nodes - window_nodes, axis=1))
                )
                # Passive damping affects velocity after integration. Require
                # the actual shape to stop settling before anchoring frame 0
                # or the indentation reference, even when velocities are low.
                stable = bool(
                    stable and window_drift <= 0.0005
                    and last_metrics["pillow_max_speed_mps"] <= 0.001
                )
                if not stable:
                    window_nodes = None
            consecutive = consecutive + 1 if stable else 0
            if consecutive >= required:
                self._g01_3_previous_nodes = nodes.copy()
                # Start indentation accounting from the settled support
                # state, so table contact/settling cannot pre-arm G01-5 grip.
                self._g01_3_reference_nodes = nodes.copy()
                return {
                    "natural_physics_substeps": int(step_index),
                    "consecutive_stable_substeps": int(consecutive),
                    "timeout_substeps": int(maximum_steps),
                    "nodal_window_drift_m": window_drift if check_nodal_drift else None,
                }
            if step_index < maximum_steps:
                # This is ordinary gravity/contact/robot-drive integration.
                # No pillow or carton state is written during the settling
                # interval, and the ordered success stage cannot advance while
                # the real hand envelopes remain outside the pickup lane.
                self._step_isaaclab()
        raise RuntimeError(
            f"{G01_TASK_IDS[self._g01_task_name]} pillow did not naturally settle before collection-ready: "
            f"rms={last_metrics.get('pillow_rms_speed_mps', math.inf):.6f}m/s, "
            f"max={last_metrics.get('pillow_max_speed_mps', math.inf):.6f}m/s, "
            f"p99={last_metrics.get('pillow_speed_p99_mps', math.inf):.6f}m/s, "
            f"p999={last_metrics.get('pillow_speed_p999_mps', math.inf):.6f}m/s, "
            f"nodes_above_max={last_metrics.get('pillow_nodes_above_max_speed', -1)}, "
            f"inverted_tets={last_metrics.get('pillow_inverted_tet_count', -1)}, "
            f"support_gap={support_gap:.6f}m, "
            f"nodal_window_drift={window_drift:.6f}m, "
            f"timeout={timeout_seconds:.1f}s"
        )

    def _validate_task_initial_state_before_ready(self) -> dict[str, Any] | None:
        parent = super()._validate_task_initial_state_before_ready()
        if self._g01_task_name not in G01_CARTON_TASK_NAMES:
            return dict(parent) if isinstance(parent, dict) else parent
        if not self._robots:
            raise RuntimeError("G01-3 initial-state audit requires both live robots")

        errors: list[str] = []
        layout_margin = _g01_3_reset_layout_margin(
            self._g01_3_reset_layout, self._g01_task_name
        )
        if layout_margin < _G01_3_RESET_MIN_MARGIN_M:
            errors.append(
                "sampled reset layout lost its support/separation/reach margin: "
                f"{layout_margin:.6f}m"
            )
        settle_report = self._settle_g01_3_initial_state_or_raise()
        nodes, velocities = self._g01_3_pillow_readback()
        metrics = self._g01_3_measure_state(nodes, velocities)
        if metrics.get("pillow_inverted_tet_count", 0):
            errors.append("pillow has inverted tetrahedra at the initial state")
        if metrics["pillow_rms_speed_mps"] > _G01_3_STABLE_RMS_SPEED_MPS:
            errors.append(
                "pillow startup RMS speed exceeds 0.025m/s: "
                f"{metrics['pillow_rms_speed_mps']:.6f}"
            )
        if metrics["pillow_speed_p99_mps"] > _G01_3_STABLE_MAX_SPEED_MPS:
            errors.append(
                "pillow startup P99 speed exceeds 0.080m/s: "
                f"{metrics['pillow_speed_p99_mps']:.6f}"
            )
        if (
            metrics["pillow_speed_outlier_fraction"]
            > _G01_3_STABLE_OUTLIER_FRACTION
        ):
            errors.append(
                "pillow startup speed-outlier fraction exceeds 1%: "
                f"{metrics['pillow_speed_outlier_fraction']:.6f}"
            )
        expected_centroid_xy = self._g01_3_reset_layout[:2]
        if float(
            np.linalg.norm(metrics["pillow_centroid"][:2] - expected_centroid_xy)
        ) > 0.025:
            errors.append("pillow startup centroid drifted from its support lane")
        support_gap = float(
            metrics["pillow_min_z_m"] - metrics["table_top_z_m"]
        )
        if not -0.006 <= support_gap <= 0.025:
            errors.append(
                f"pillow support gap is outside [-6,25]mm: {support_gap:.6f}m"
            )
        carton_pose = np.asarray(metrics["carton_pose"], dtype=np.float64)
        expected_carton_pose = _g01_3_expected_carton_pose_wxyz(
            self._g01_3_reset_layout
        )
        if float(
            np.linalg.norm(carton_pose[:3] - expected_carton_pose[:3])
        ) > 0.002:
            errors.append("BC004 carton moved from its sampled table pose")
        carton_quat = carton_pose[3:7] / max(
            float(np.linalg.norm(carton_pose[3:7])), 1.0e-12
        )
        expected_carton_quat = expected_carton_pose[3:7]
        carton_orientation_error = 2.0 * math.acos(
            min(1.0, abs(float(np.dot(carton_quat, expected_carton_quat))))
        )
        if carton_orientation_error > 1.0e-3:
            errors.append("BC004 carton rotated away from its sampled heading")
        carton_velocity = self._physics_body_velocity("g01_3_carton")
        if (
            carton_velocity is None
            or not np.all(np.isfinite(carton_velocity))
            or float(np.max(np.abs(carton_velocity))) > 0.01
        ):
            errors.append("BC004 kinematic carton velocity is invalid or nonzero")
        if metrics["left_hand_clearance_m"] <= 0.015:
            errors.append("left complete hand envelope overlaps Pillow024-v2")
        if metrics["right_hand_clearance_m"] <= 0.015:
            errors.append("right complete hand envelope overlaps Pillow024-v2")

        carton_outer_lower = np.asarray(
            G01_3_CARTON_LOCAL_MIN, dtype=np.float64
        )
        carton_outer_upper = np.asarray(
            G01_3_CARTON_LOCAL_MAX, dtype=np.float64
        )
        carton_rotation = _quat_wxyz_to_matrix(carton_pose[3:7])
        carton_hand_clearance: dict[str, float] = {}
        for side in ("right", "left"):
            hand_points = np.asarray(
                metrics[f"{side}_hand_points"], dtype=np.float64
            ).reshape(-1, 3)
            if hand_points.size == 0:
                clearance = -math.inf
            else:
                local_hand_points = (
                    hand_points - carton_pose[:3]
                ) @ carton_rotation
                outside = np.maximum(
                    np.maximum(carton_outer_lower - local_hand_points, 0.0),
                    local_hand_points - carton_outer_upper,
                )
                clearance = float(
                    np.min(np.linalg.norm(outside, axis=1))
                    - _G01_3_HAND_BODY_COVER_RADIUS_M
                )
            carton_hand_clearance[side] = clearance
            if clearance <= 0.015:
                errors.append(
                    f"{side} complete hand envelope overlaps the open carton"
                )

        robot_report: dict[str, Any] = {}
        for side in ("right", "left"):
            robot = self._robots.get(side)
            if robot is None:
                errors.append(f"{side} BC004-matched robot is unavailable")
                continue
            arm_ids = self._arm_joint_ids.get(side, [])
            hand_ids = self._hand_joint_ids.get(side, [])
            actual_q = _to_numpy(robot.data.joint_pos)[0]
            actual_qd = _to_numpy(robot.data.joint_vel)[0]
            default_q = _to_numpy(robot.data.default_joint_pos)[0]
            arm_q = np.asarray(actual_q[arm_ids], dtype=np.float64)
            arm_qd = np.asarray(actual_qd[arm_ids], dtype=np.float64)
            hand_q = np.asarray(actual_q[hand_ids], dtype=np.float64)
            hand_qd = np.asarray(actual_qd[hand_ids], dtype=np.float64)
            expected_hand = np.asarray(default_q[hand_ids], dtype=np.float64)
            expected_arm = (
                _G01_5_LOWERED_ARM_HOME if self._g01_task_name == G01_5_TASK_NAME
                else _D06_LOWERED_ARM_HOME
            )[side]
            arm_error = (
                math.inf
                if arm_q.shape != expected_arm.shape
                else float(np.max(np.abs(arm_q - expected_arm)))
            )
            arm_speed = (
                math.inf if arm_qd.size == 0 else float(np.max(np.abs(arm_qd)))
            )
            hand_home_error = (
                math.inf
                if hand_q.shape != expected_hand.shape or hand_q.size == 0
                else float(np.max(np.abs(hand_q - expected_hand)))
            )
            hand_speed = (
                math.inf if hand_qd.size == 0 else float(np.max(np.abs(hand_qd)))
            )
            if arm_error > 0.010:
                errors.append(
                    f"{side} arm does not match the lower symmetric C-tool home: "
                    f"error={arm_error:.6f}rad"
                )
            if arm_speed > 0.050:
                errors.append(f"{side} arm is moving at collection-ready")
            if hand_home_error > 0.010:
                errors.append(
                    f"{side} Wuji hand does not match the source-defined "
                    f"open home: error={hand_home_error:.6f}rad"
                )
            if hand_speed > 0.100:
                errors.append(f"{side} Wuji hand is moving at collection-ready")
            robot_report[side] = {
                "arm_home_max_error_rad": arm_error,
                "arm_max_speed_rad_s": arm_speed,
                "hand_home_max_error_rad": hand_home_error,
                "hand_max_speed_rad_s": hand_speed,
                "hand_envelope_point_count": len(
                    metrics[f"{side}_hand_points"]
                ),
                "carton_clearance_m": carton_hand_clearance[side],
            }

        if self._stage is None:
            errors.append("G01-3 USD stage is unavailable")
        else:
            from pxr import UsdGeom, UsdPhysics

            for collider_path in G01_3_CARTON_COLLIDER_PATHS:
                collider_prim = self._stage.GetPrimAtPath(collider_path)
                if (
                    collider_prim is None
                    or not collider_prim.IsValid()
                    or not (
                        collider_prim.IsA(UsdGeom.Mesh)
                        or collider_prim.IsA(UsdGeom.Cube)
                    )
                    or not collider_prim.HasAPI(UsdPhysics.CollisionAPI)
                    or not bool(
                        UsdPhysics.CollisionAPI(collider_prim)
                        .GetCollisionEnabledAttr()
                        .Get()
                    )
                ):
                    errors.append(f"open-carton collider is invalid: {collider_path}")
                else:
                    approximation = (
                        UsdPhysics.MeshCollisionAPI(collider_prim)
                        .GetApproximationAttr()
                        .Get()
                        if collider_prim.IsA(UsdGeom.Mesh) else None
                    )
                    if (
                        (
                            not collider_prim.IsA(UsdGeom.Cube)
                            and approximation != "convexHull"
                        )
                        or collider_prim.GetCustomDataByKey(
                            "handson:collisionSource"
                        ) != "bc004_source_panel_extrusion"
                        or UsdGeom.Imageable(collider_prim).ComputeVisibility()
                        != UsdGeom.Tokens.invisible
                    ):
                        errors.append(
                            "open-carton contact panel is not a hidden source extrusion"
                        )
            open_mesh = self._stage.GetPrimAtPath(
                G01_3_CARTON_PATH + "/E_carton_1/P_126c3c58f54ffd7c"
            )
            if (
                open_mesh is None
                or not open_mesh.IsValid()
                or open_mesh.GetCustomDataByKey(
                    "handson:removedSourceTopFaceCount"
                ) != 14
                or open_mesh.GetCustomDataByKey(
                    "handson:openRenderFaceCount"
                ) != G01_3_CARTON_OPEN_FACE_COUNT
            ):
                errors.append("BC004 open render topology audit failed")

        report = {
            "valid": not errors,
            "pillow_node_count": int(nodes.shape[0]),
            "pillow_support_gap_m": support_gap,
            "pillow_rms_speed_mps": float(metrics["pillow_rms_speed_mps"]),
            "pillow_max_speed_mps": float(metrics["pillow_max_speed_mps"]),
            "pillow_speed_p99_mps": float(metrics["pillow_speed_p99_mps"]),
            "pillow_speed_outlier_fraction": float(
                metrics["pillow_speed_outlier_fraction"]
            ),
            "pillow_uniform_scale": float(G01_3_PILLOW_UNIFORM_SCALE),
            "carton_collider_count": len(G01_3_CARTON_COLLIDER_PATHS),
            "natural_settle": settle_report,
            "reset_layout": self._g01_3_reset_layout.tolist(),
            "reset_layout_margin_m": layout_margin,
            "reach_target_distances_m": {
                name: float(np.linalg.norm(target - _G01_3_ARM_BASES[side]))
                for name, (side, target) in _g01_3_reach_targets(
                    self._g01_3_reset_layout
                ).items()
            },
            "carton_orientation_error_rad": carton_orientation_error,
            "robot_home_source_task": (
                "g01_5_d06_home_lowered_80mm"
                if self._g01_task_name == G01_5_TASK_NAME
                else "shared_d06_lower_symmetric_c_tool_home"
            ),
            "robots": robot_report,
            "errors": tuple(errors),
        }
        self._g01_3_initial_state_report = report
        if errors:
            raise RuntimeError(
                "G01-3 initial-state audit failed before collection-ready: "
                + "; ".join(errors)
            )
        payload = dict(parent) if isinstance(parent, dict) else {}
        payload["g01_3"] = report
        print(
            f"[HandsOn/IsaacLab] {G01_TASK_IDS[self._g01_task_name]} initial state valid: "
            f"nodes={nodes.shape[0]}, support_gap={support_gap:.6f}m, "
            "BC004_home=True, hand_envelopes_clear=True, carton_open=True",
            flush=True,
        )
        return payload

    def stable_recording_origin_task_readiness(
        self,
    ) -> Mapping[str, Any] | None:
        """Expose a write-free deformable readiness gate for G01-3."""

        if self._g01_task_name not in G01_CARTON_TASK_NAMES:
            return None
        if "Pillow024" not in self._g01_soft_views or not self._robots:
            return {
                "ready": False,
                "reason": "live_pillow_or_robot_views_unavailable",
            }
        nodes, velocities = self._g01_3_pillow_readback()
        metrics = self._g01_3_measure_state(nodes, velocities)
        support_gap = float(
            metrics["pillow_min_z_m"] - metrics["table_top_z_m"]
        )
        carton_pose = np.asarray(metrics["carton_pose"], dtype=np.float64)
        expected_carton_pose = _g01_3_expected_carton_pose_wxyz(
            self._g01_3_reset_layout
        )
        carton_pose_error = float(
            np.linalg.norm(carton_pose[:3] - expected_carton_pose[:3])
        )
        carton_quat = carton_pose[3:7] / max(
            float(np.linalg.norm(carton_pose[3:7])), 1.0e-12
        )
        carton_orientation_error = 2.0 * math.acos(
            min(
                1.0,
                abs(float(np.dot(carton_quat, expected_carton_pose[3:7]))),
            )
        )
        pillow_xy_error = float(
            np.linalg.norm(
                metrics["pillow_centroid"][:2] - self._g01_3_reset_layout[:2]
            )
        )
        finite = bool(
            nodes.shape == (self._pillow_node_count(), 3)
            and velocities.shape == nodes.shape
            and np.all(np.isfinite(nodes))
            and np.all(np.isfinite(velocities))
        )
        ready = bool(
            finite
            and metrics.get("pillow_inverted_tet_count", 0) == 0
            and -0.006 <= support_gap <= 0.025
            and metrics["pillow_rms_speed_mps"]
            <= _G01_3_STABLE_RMS_SPEED_MPS
            and metrics["pillow_speed_p99_mps"]
            <= _G01_3_STABLE_MAX_SPEED_MPS
            and metrics["pillow_speed_outlier_fraction"]
            <= _G01_3_STABLE_OUTLIER_FRACTION
            and pillow_xy_error <= 0.025
            and carton_pose_error <= 0.002
            and carton_orientation_error <= 1.0e-3
            and _g01_3_reset_layout_margin(
                self._g01_3_reset_layout, self._g01_task_name
            )
            >= _G01_3_RESET_MIN_MARGIN_M
            and self._g01_3_stage == _G01_3_STAGE_WAITING_CONTACT
            and bool(self._g01_3_initial_state_report.get("valid"))
        )
        return {
            "ready": ready,
            "kind": "g01_3_deformable_initial_state",
            "finite": finite,
            "pillow_node_count": int(nodes.shape[0]),
            "pillow_inverted_tet_count": metrics.get("pillow_inverted_tet_count", 0),
            "pillow_support_gap_m": support_gap,
            "pillow_rms_speed_mps": float(
                metrics["pillow_rms_speed_mps"]
            ),
            "pillow_max_speed_mps": float(
                metrics["pillow_max_speed_mps"]
            ),
            "pillow_speed_p99_mps": float(
                metrics["pillow_speed_p99_mps"]
            ),
            "pillow_speed_outlier_fraction": float(
                metrics["pillow_speed_outlier_fraction"]
            ),
            "carton_pose_error_m": carton_pose_error,
            "carton_orientation_error_rad": carton_orientation_error,
            "pillow_xy_error_m": pillow_xy_error,
            "reset_layout": self._g01_3_reset_layout.tolist(),
            "reset_layout_margin_m": _g01_3_reset_layout_margin(
                self._g01_3_reset_layout, self._g01_task_name
            ),
            "stage": int(self._g01_3_stage),
            "initial_state_audit_valid": bool(
                self._g01_3_initial_state_report.get("valid")
            ),
        }

    def reset(self, seed: int | None = None, **kwargs: Any):
        """Re-audit the actual G01-3 physical state on every episode reset."""

        result = super().reset(seed=seed, **kwargs)
        if (
            self._g01_task_name not in G01_CARTON_TASK_NAMES
            or not self._robots
            or "Pillow024" not in self._g01_soft_views
        ):
            return result
        self._validate_task_initial_state_before_ready()
        self._refresh_object_poses()
        self._sync_b320_tcp_command_poses_to_physical_state()
        _old_observation, old_info = result
        info = dict(old_info)
        info.update(self._task_info())
        observation = self._attach_physx_ground_truth(
            self._compute_observation()
        )
        return observation, info

    def accept_stable_recording_origin(
        self,
        report: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Recheck the write-free G01-3 boundary accepted for action zero."""

        parent_acceptance = super().accept_stable_recording_origin(report)
        if self._g01_task_name not in G01_CARTON_TASK_NAMES:
            return parent_acceptance
        readiness = self.stable_recording_origin_task_readiness()
        if readiness is None or readiness.get("ready") is not True:
            raise RuntimeError(
                "G01-3 stable recording origin failed deformable readiness: "
                f"{readiness}"
            )
        payload = (
            dict(parent_acceptance)
            if isinstance(parent_acceptance, Mapping)
            else {}
        )
        payload.update(
            {
                "accepted": True,
                "kind": "g01_3_actual_robot_deformable_and_carton_boundary",
                "task_readiness": dict(readiness),
                "initial_state": dict(self._g01_3_initial_state_report),
            }
        )
        return payload

    def _build_observation_space(self) -> spaces.Dict:
        if self._g01_task_name in G01_CARTON_TASK_NAMES:
            state_spaces = {
                "tcp_pose": spaces.Box(
                    -np.inf, np.inf, shape=(14,), dtype=np.float64
                ),
                "gripper_pose": spaces.Box(
                    -np.inf, np.inf, shape=(32,), dtype=np.float64
                ),
                "pillow_centroid": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float64
                ),
                "pillow_aabb": spaces.Box(
                    -np.inf, np.inf, shape=(6,), dtype=np.float64
                ),
                "pillow_valid": spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float64
                ),
                "carton_pose": spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype=np.float64
                ),
                "carton_inner_bounds": spaces.Box(
                    -np.inf, np.inf, shape=(6,), dtype=np.float64
                ),
                "task_metrics": spaces.Box(
                    -np.inf, np.inf, shape=(16,), dtype=np.float64
                ),
                "stage_state": spaces.Box(
                    0.0, 1.0, shape=(8,), dtype=np.float64
                ),
                "reset_layout": spaces.Box(
                    -np.inf, np.inf, shape=(6,), dtype=np.float64
                ),
                "table_delta_height": spaces.Box(
                    -np.inf, np.inf, shape=(1,), dtype=np.float64
                ),
            }
            if self._g01_task_name == G01_5_TASK_NAME:
                state_spaces["foam_grip_state"] = spaces.Box(
                    0.0, np.inf, shape=(3,), dtype=np.float64
                )
            observation_spaces: dict[str, spaces.Space] = {
                "state": spaces.Dict(state_spaces)
            }
            if self.image_obs:
                head_image_space = lambda: spaces.Box(
                    0, 255, shape=self._image_shape, dtype=np.uint8
                )
                wrist_width, wrist_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
                wrist_shape = (wrist_height, wrist_width, 3)
                wrist_image_space = lambda: spaces.Box(
                    0, 255, shape=wrist_shape, dtype=np.uint8
                )
                observation_spaces["images"] = spaces.Dict(
                    {
                        "random_camera" if self.randomize else "front": (
                            head_image_space()
                        ),
                        "wrist_left": wrist_image_space(),
                        "wrist_right": wrist_image_space(),
                    }
                )
            return spaces.Dict(observation_spaces)
        state_spaces = {
            "tcp_pose": spaces.Box(
                -np.inf, np.inf, shape=(14,), dtype=np.float64
            ),
            "gripper_pose": spaces.Box(
                -np.inf, np.inf, shape=(32,), dtype=np.float64
            ),
            "asset_centroids": spaces.Box(
                -np.inf, np.inf, shape=(9,), dtype=np.float64
            ),
            "asset_aabbs": spaces.Box(
                -np.inf, np.inf, shape=(18,), dtype=np.float64
            ),
            "asset_valid": spaces.Box(
                0.0, 1.0, shape=(3,), dtype=np.float64
            ),
            "button_pose": spaces.Box(
                -np.inf, np.inf, shape=(7,), dtype=np.float64
            ),
            "button_contact": spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float64
            ),
            "table_delta_height": spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float64
            ),
        }
        observation_spaces: dict[str, spaces.Space] = {
            "state": spaces.Dict(state_spaces)
        }
        if self.image_obs:
            head_image_space = lambda: spaces.Box(
                0, 255, shape=self._image_shape, dtype=np.uint8
            )
            wrist_width, wrist_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
            wrist_image_shape = (wrist_height, wrist_width, 3)
            wrist_image_space = lambda: spaces.Box(
                0, 255, shape=wrist_image_shape, dtype=np.uint8
            )
            observation_spaces["images"] = spaces.Dict(
                {
                    "random_camera" if self.randomize else "front": (
                        head_image_space()
                    ),
                    "wrist_left": wrist_image_space(),
                    "wrist_right": wrist_image_space(),
                }
            )
        return spaces.Dict(observation_spaces)

    def _robot_state(self) -> tuple[np.ndarray, np.ndarray]:
        tcp = np.concatenate(
            [self._right_tcp_pose, self._left_tcp_pose]
        ).astype(np.float64)
        gripper = np.zeros(32, dtype=np.float64)
        gripper[: min(16, self._right_hand.size)] = self._right_hand[:16]
        gripper[16 : 16 + min(16, self._left_hand.size)] = self._left_hand[:16]
        return tcp, gripper

    def _button_pose(self) -> np.ndarray:
        fallback = np.concatenate(
            [_G01_BUTTON_POSITION, np.asarray([1.0, 0.0, 0.0, 0.0])]
        )
        return np.asarray(
            self._poses.get("g01_success_button", fallback),
            dtype=np.float64,
        ).reshape(7)

    def _compute_observation(self) -> dict[str, Any]:
        tcp, gripper = self._robot_state()
        if self._g01_task_name in G01_CARTON_TASK_NAMES:
            metrics = self._g01_3_measure_state()
            if self._g01_3_last_metrics:
                metrics = {**metrics, **self._g01_3_last_metrics}
            stage_state = np.zeros(8, dtype=np.float64)
            stage_index = int(metrics.get("stage", self._g01_3_stage))
            stage_state[min(max(stage_index, 0), _G01_3_STAGE_COUNT - 1)] = 1.0
            stable_required = max(
                1, int(metrics.get("stable_substeps_required", G01_PHYSICS_HZ))
            )
            stage_state[6] = min(
                1.0,
                float(metrics.get("stable_substeps", 0)) / stable_required,
            )
            stage_state[7] = float(
                stage_index == _G01_3_STAGE_COMPLETE
            )
            task_metrics = np.asarray(
                [
                    float(stage_index),
                    float(self._g01_3_contact_latched["left"]),
                    float(self._g01_3_contact_latched["right"]),
                    float(metrics["pillow_lift_clearance_m"]),
                    float(self._g01_3_lift_latched),
                    float(metrics["containment_fraction"]),
                    float(metrics["contained_now"]),
                    float(metrics["left_hand_clearance_m"]),
                    float(metrics["right_hand_clearance_m"]),
                    float(self._g01_3_release_latched),
                    float(metrics["pillow_rms_speed_mps"]),
                    float(metrics["pillow_max_speed_mps"]),
                    float(metrics.get("stable_substeps", 0)),
                    float(stable_required),
                    float(stage_index == _G01_3_STAGE_COMPLETE),
                    1.0,
                ],
                dtype=np.float64,
            )
            state = {
                "tcp_pose": tcp,
                "gripper_pose": gripper,
                "pillow_centroid": np.asarray(
                    metrics["pillow_centroid"], dtype=np.float64
                ),
                "pillow_aabb": np.asarray(
                    metrics["pillow_aabb"], dtype=np.float64
                ),
                "pillow_valid": np.ones(1, dtype=np.float64),
                "carton_pose": np.asarray(
                    metrics["carton_pose"], dtype=np.float64
                ),
                "carton_inner_bounds": np.asarray(
                    metrics["carton_inner_bounds"], dtype=np.float64
                ),
                "task_metrics": task_metrics,
                "stage_state": stage_state,
                "reset_layout": self._g01_3_reset_layout.astype(
                    np.float64, copy=True
                ),
                "table_delta_height": self.delta_h.astype(
                    np.float64, copy=True
                ),
            }
            if self._g01_task_name == G01_5_TASK_NAME:
                state["foam_grip_state"] = np.asarray([
                    *self._g01_5_hand_indentations_m,
                    float(self._g01_5_grip_friction_active),
                ], dtype=np.float64)
            observation: dict[str, Any] = {"state": state}
            if self.image_obs:
                front, wrist_left, wrist_right = self.render()
                observation["images"] = {
                    "random_camera" if self.randomize else "front": front,
                    "wrist_left": wrist_left,
                    "wrist_right": wrist_right,
                }
            return observation
        centroids, bounds, valid = self._asset_geometry()
        state = {
            "tcp_pose": tcp,
            "gripper_pose": gripper,
            "asset_centroids": centroids.reshape(-1).astype(np.float64),
            "asset_aabbs": bounds.reshape(-1).astype(np.float64),
            "asset_valid": valid.astype(np.float64),
            "button_pose": self._button_pose(),
            "button_contact": np.asarray(
                [
                    float(self._g01_contact_monitor_available),
                    float(self._g01_button_touching),
                    float(self._g01_button_contact_force),
                    self._success_counter
                    / max(1, self._success_steps_required()),
                ],
                dtype=np.float64,
            ),
            "table_delta_height": self.delta_h.astype(
                np.float64, copy=True
            ),
        }
        observation: dict[str, Any] = {"state": state}
        if self.image_obs:
            front, wrist_left, wrist_right = self.render()
            observation["images"] = {
                "random_camera" if self.randomize else "front": front,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }
        return observation

    def _compute_success_now(self) -> bool:
        if self._g01_task_name in G01_CARTON_TASK_NAMES:
            return bool(self._g01_3_stage == _G01_3_STAGE_COMPLETE)
        force, available = self._read_button_contact()
        self._g01_contact_monitor_available = available
        self._g01_button_contact_force = force
        self._g01_button_touching = bool(
            available and force >= _G01_BUTTON_CONTACT_FORCE_THRESHOLD
        )
        if self._g01_task_name != G01_2_TASK_NAME:
            return self._g01_button_touching
        _centroids, _bounds, valid = self._asset_geometry()
        return bool(self._g01_button_touching and np.all(valid > 0.5))

    def _success_steps_required(self) -> int:
        if self._g01_task_name in G01_CARTON_TASK_NAMES:
            # The high-frequency task state machine already enforces the full
            # one-second continuous physical stability window at its physics rate.
            return 1
        return super()._success_steps_required()

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        if self._g01_task_name in G01_CARTON_TASK_NAMES:
            metrics = self._g01_3_last_metrics
            stable_required = max(
                1, int(metrics.get("stable_substeps_required", G01_PHYSICS_HZ))
            )
            rows = [
                self._condition_row(
                    "g01_3_both_hands_contacted",
                    (
                        "Both assigned physical hands contacted the foam pillow"
                        if self._g01_task_name == G01_5_TASK_NAME
                        else "Both assigned physical hands contacted Pillow024-v2 on the table"
                    ),
                    all(self._g01_3_contact_latched.values()),
                ),
                self._condition_row(
                    "g01_3_pillow_lifted",
                    "The pillow was lifted at least 30 mm clear of the tabletop",
                    self._g01_3_lift_latched,
                ),
                self._condition_row(
                    "g01_3_pillow_contained",
                    (
                        "The pillow rests inside the carton with at least 90% below the rim; raised corners are allowed"
                        if self._g01_task_name == G01_5_TASK_NAME
                        else "At least 98% of pillow nodes entered the open BC004 carton"
                    ),
                    self._g01_3_contained_latched,
                ),
                self._condition_row(
                    "g01_3_both_hands_released",
                    "Both complete hand envelopes released at least 90 mm from the pillow",
                    self._g01_3_release_latched,
                ),
                self._condition_row(
                    "g01_3_pillow_stable",
                    "The released pillow remains contained and stable for one second "
                    f"({int(metrics.get('stable_substeps', 0))}/{stable_required} "
                    "physics steps)",
                    self._g01_3_stage == _G01_3_STAGE_COMPLETE,
                ),
            ]
            if self._g01_task_name in G01_COMPARISON_PROFILES:
                profile = G01_COMPARISON_PROFILES[self._g01_task_name]
                for row in rows:
                    row["label"] = row["label"].replace("Pillow024-v2", f"the {profile} pillow")
            return self._append_success_hold_row(rows)
        _centroids, _bounds, valid = self._asset_geometry()
        rows = [
            self._condition_row(
                "g01_assets_loaded",
                "Pillow024, Powercable001, and Towl011 are loaded",
                bool(np.all(valid > 0.5)),
            ),
            self._condition_row(
                "g01_contact_monitor",
                "Physical hand/button contact monitor is active",
                self._g01_contact_monitor_available,
            ),
            self._condition_row(
                "g01_green_button_touched",
                "Either Wuji hand touches the green lower-right button",
                self._g01_button_touching,
            ),
        ]
        return self._append_success_hold_row(rows)

    def _task_info(self) -> dict[str, Any]:
        if self._g01_task_name in G01_CARTON_TASK_NAMES:
            if not self._g01_3_last_metrics and self._g01_soft_views:
                self._g01_3_last_metrics = self._g01_3_measure_state()
            metrics = {
                key: value.tolist()
                if isinstance(value, np.ndarray)
                else value.item()
                if isinstance(value, np.generic)
                else value
                for key, value in self._g01_3_last_metrics.items()
                if key
                not in {
                    "nodes",
                    "velocities",
                    "left_hand_points",
                    "right_hand_points",
                }
            }
            info = super()._task_info()
            info.update(
                {
                    "task_id": G01_TASK_IDS[self._g01_task_name],
                    "task_name": self._g01_task_name,
                    "g01_pillow_revision": (
                        G01_COMPARISON_PROFILES[self._g01_task_name] + ":sewn-pillow-v2"
                        if self._g01_task_name in G01_COMPARISON_PROFILES else "Pillow024_v2"
                    ),
                    "g01_pillow_asset_sha256": self._g01_pillow_asset_sha256,
                    "g01_3_pillow_uniform_scale": float(
                        G01_3_PILLOW_UNIFORM_SCALE
                    ),
                    "g01_5_pillow_half_height_m": float(
                        G01_5_PILLOW_HALF_HEIGHT_M
                    ),
                    "g01_3_carton_source_task": "B320-BC004",
                    "g01_3_carton_open_collider_count": len(
                        G01_3_CARTON_COLLIDER_PATHS
                    ),
                    "g01_3_stage": int(self._g01_3_stage),
                    "g01_3_reset_seed": int(getattr(self, "_seed", 0)),
                    "g01_3_reset_ordinal": int(self._g01_3_reset_ordinal),
                    "g01_3_reset_layout": self._g01_3_reset_layout.tolist(),
                    "g01_3_reset_layout_frame": "world_xy_yaw_degrees",
                    "g01_3_reset_layout_margin_m": _g01_3_reset_layout_margin(
                        self._g01_3_reset_layout, self._g01_task_name
                    ),
                    "g01_3_metrics": metrics,
                    "g01_3_contact_response_diagnostics": {
                        side: dict(values)
                        for side, values in (
                            self._g01_3_contact_response_diagnostics.items()
                        )
                    },
                    "g01_5_min_indentation_m": float(_G01_5_MIN_INDENTATION_M),
                    "g01_5_max_indentation_m": float(self._g01_5_max_indentation_m),
                    "g01_5_grip_friction_active": bool(
                        self._g01_5_grip_friction_active
                    ),
                    "g01_5_grip_static_friction": float(
                        _G01_5_GRIP_STATIC_FRICTION
                    ),
                    "g01_5_grip_dynamic_friction": float(
                        _G01_5_GRIP_DYNAMIC_FRICTION
                    ),
                    "g01_3_initial_state_report": dict(
                        self._g01_3_initial_state_report
                    ),
                    "g01_3_success_assist_enabled": False,
                    "g01_3_contact_assist_enabled": False,
                    "g01_3_auto_failure_reset_enabled": False,
                    "g01_volume_deformable_physical_replay_certified": False,
                }
            )
            return info
        if not self._g01_geometry_sampled:
            self._asset_geometry()
        info = super()._task_info()
        info.update(
            {
                "task_id": (
                    "G01-2"
                    if self._g01_task_name == G01_2_TASK_NAME
                    else "G01"
                ),
                "task_name": self._g01_task_name,
                "g01_asset_names": tuple(
                    spec.name for spec in self._g01_asset_specs
                ),
                "g01_pillow_revision": (
                    "Pillow024_v2"
                    if self._g01_task_name == G01_2_TASK_NAME
                    else "Pillow024_legacy_migrated"
                ),
                "g01_asset_valid": tuple(
                    bool(value > 0.5)
                    for value in self._g01_runtime_asset_valid
                ),
                "g01_button_contact_available": bool(
                    self._g01_contact_monitor_available
                ),
                "g01_button_contact_force": float(
                    self._g01_button_contact_force
                ),
                "g01_button_touched": bool(self._g01_button_touching),
                "g01_button_contact_force_threshold": float(
                    _G01_BUTTON_CONTACT_FORCE_THRESHOLD
                ),
                "g01_multi_deformable_physical_replay_certified": False,
            }
        )
        return info

    def get_physical_replay_reset_contract(self) -> dict[str, Any]:
        contract = dict(super().get_physical_replay_reset_contract())
        pillow_trace = self._g01_task_name == G01_5_TASK_NAME
        contract["g01_deformables_in_formal_truth"] = pillow_trace
        contract["g01_volume_deformables_in_formal_truth"] = pillow_trace
        contract["g01_multi_deformable_replay_certified"] = False
        contract["g01_3_bimanual_pillow_carton_replay_certified"] = False
        contract["g01_3_success_assist_enabled"] = False
        contract["g01_3_contact_assist_enabled"] = False
        contract["g01_3_reset_layout_frame"] = "world_xy_yaw_degrees"
        contract["g01_task_variant"] = self._g01_task_name
        contract["g01_pillow_revision"] = (
            "Pillow024_v2"
            if self._g01_task_name in _G01_CURRENT_PILLOW_TASK_NAMES
            else "Pillow024_legacy_migrated"
        )
        if self._g01_task_name in G01_COMPARISON_PROFILES:
            contract["g01_pillow_revision"] = G01_COMPARISON_PROFILES[self._g01_task_name] + ":sewn-pillow-v2"
            contract["g01_pillow_asset_sha256"] = self._g01_pillow_asset_sha256
        if self._g01_task_name == G01_5_TASK_NAME:
            contract["g01_5_grip_indentation_threshold_m"] = _G01_5_MIN_INDENTATION_M
            contract["g01_5_grip_reference_nodes_world_m"] = (
                self._g01_3_reference_nodes.tolist()
                if self._g01_3_reference_nodes is not None else None
            )
        return contract


__all__ = [
    "HandsOnG01DeformableAssetIsaacLabEnv",
    "_G01_BUTTON_CONTACT_FORCE_THRESHOLD",
    "_G01_5_MIN_INDENTATION_M",
    "_G01_5_GRIP_STATIC_FRICTION",
    "_G01_5_GRIP_DYNAMIC_FRICTION",
    "_detect_g01_3_deformable_contact_response",
]
