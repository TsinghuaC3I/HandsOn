"""Pure layout and event geometry for YCB01.

The functions in this module deliberately have no Isaac imports.  This keeps
the direction-sensitive crossing contract testable without launching Kit and
gives live collection and replay one source of geometry truth.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


YCB01_TASK_NAME = "t34_roll_ball_through_gate"
YCB01_BALL_NAME = "tennis_ball"
YCB01_LEFT_GATE_NAME = "004_sugar_box"
YCB01_RIGHT_GATE_NAME = "003_cracker_box"
YCB01_GATE_BODY_NAMES = (YCB01_LEFT_GATE_NAME, YCB01_RIGHT_GATE_NAME)

YCB01_BALL_RADIUS_M = 0.0330205
YCB01_GATE_WIDTH_M = 0.130
YCB01_GATE_CLEARANCE_MARGIN_M = 0.005
YCB01_START_SIDE_MARGIN_M = 0.010
# The ball centre must stay inside this lateral half-width while it crosses
# either gate boundary.  Keeping the value derived from the authored opening
# avoids duplicating a subtly different clearance calculation in the runtime.
YCB01_VALID_OPENING_HALF_WIDTH_M = max(
    0.0,
    0.5 * YCB01_GATE_WIDTH_M
    - YCB01_BALL_RADIUS_M
    - YCB01_GATE_CLEARANCE_MARGIN_M,
)
# The gate frame's Y axis is the direction of travel.  A centre crossing of
# ``y == 0`` only says that the ball reached the middle of the opening.  The
# success event below deliberately compares the ball centre directly with the
# far carton face: there is no extra radius or safety-margin requirement. Keep
# this zero-valued name as a compatibility alias for audit/telemetry callers.
YCB01_FAR_SIDE_MARGIN_M = 0.0
YCB01_LIFT_CLEARANCE_M = 0.015
YCB01_LIFT_DEBOUNCE_S = 0.10
YCB01_GATE_TRANSLATION_LIMIT_M = 0.030
YCB01_GATE_TILT_LIMIT_DEG = 20.0
YCB01_GATE_DEBOUNCE_S = 0.15
YCB01_HAND_CONTACT_FORCE_N = 1.0e-4
# A gate contact is a hard rule in the collection task.  Keep the force
# threshold above solver denormals while the geometry backstop below catches
# a zero-force contact reported during a very fast CCD step.
YCB01_GATE_CONTACT_FORCE_N = 1.0e-6
# Robot-link origins are not guaranteed to coincide with the outermost hand
# collision shell.  This radius is used only when a PhysX tensor view is not
# available, and is deliberately conservative (it is not a task assist).
YCB01_HAND_CONTACT_RADIUS_M = 0.014

# Source mesh half extents.  Assets are never scaled.
YCB01_SUGAR_HALF_EXTENTS_M = np.asarray([0.0445, 0.0190, 0.0875], dtype=np.float64)
YCB01_CRACKER_HALF_EXTENTS_M = np.asarray(
    [0.079906, 0.0299465, 0.1063595], dtype=np.float64
)
# Both boxes share the sampled gate-frame yaw.  In that frame the farther
# rear face is the largest source half-depth.  The sphere centre only needs to
# reach that far-face Y coordinate before ``full_passed`` can latch.
YCB01_GATE_FAR_FACE_HALF_DEPTH_M = max(
    float(YCB01_SUGAR_HALF_EXTENTS_M[1]),
    float(YCB01_CRACKER_HALF_EXTENTS_M[1]),
)
YCB01_FULL_PASS_LOCAL_Y_THRESHOLD_M = YCB01_GATE_FAR_FACE_HALF_DEPTH_M

# Keep the two boxes at the authored centre of the table.  Only the ball is
# randomized, so operators always see the same gate while learning to handle
# left/right and near/far ball approaches.
YCB01_GATE_CENTER_XY = np.asarray([-0.275, 0.055], dtype=np.float64)
YCB01_GATE_CENTER_JITTER_XY = np.zeros(2, dtype=np.float64)
YCB01_GATE_YAW_LIMIT_DEG = 0.0
# Leave a small edge margin for the sphere/boxes, but otherwise cover the
# complete physical tabletop.  This is intentionally wider than the old
# calibrated camera rectangle, which confined all starts to one narrow lane.
YCB01_TABLE_XY_BOUNDS_M = np.asarray(
    [[-0.52, -0.74], [0.22, 0.74]], dtype=np.float64
)
# Keep the historical public name for downstream reset/audit consumers.  It
# now denotes the broad tabletop camera envelope used by YCB01.
YCB01_CAMERA_XY_BOUNDS = YCB01_TABLE_XY_BOUNDS_M.copy()
# Broad inner approximation of the calibrated head-camera footprint.  The
# rectangular bounds above are the cheap gate; this ellipse screens corners
# that would otherwise fall outside the camera while retaining both tabletop
# lanes.
YCB01_CAMERA_ELLIPSE_CENTER_XY = np.asarray([-0.20, 0.0], dtype=np.float64)
YCB01_CAMERA_ELLIPSE_RADII_M = np.asarray([0.55, 1.00], dtype=np.float64)
YCB01_CAMERA_CLEARANCE_M = 0.015
# Sample the ball across the full tabletop width and on either side of the
# fixed gate.  The world/table and camera checks clip only the physical edges,
# so accepted seeds can appear on the image-left or image-right approach side.
YCB01_BALL_LOCAL_X_LIMIT_M = 0.550
YCB01_BALL_LOCAL_Y_RANGE_M = (-0.700, 0.700)
YCB01_BALL_LOCAL_Y_MAGNITUDE_RANGE_M = (0.080, 0.700)
YCB01_LAYOUT_ATTEMPTS = 64


def _yaw_rotation(yaw_radians: float) -> np.ndarray:
    cosine = math.cos(yaw_radians)
    sine = math.sin(yaw_radians)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def _yaw_quaternion_wxyz(yaw_radians: float) -> np.ndarray:
    return np.asarray(
        [math.cos(0.5 * yaw_radians), 0.0, 0.0, math.sin(0.5 * yaw_radians)],
        dtype=np.float64,
    )


def ycb01_layout_is_valid(layout: dict[str, np.ndarray]) -> bool:
    """Check the complete reset footprint against table/camera bounds."""
    sugar = np.asarray(layout[YCB01_LEFT_GATE_NAME], dtype=np.float64)
    cracker = np.asarray(layout[YCB01_RIGHT_GATE_NAME], dtype=np.float64)
    ball = np.asarray(layout[YCB01_BALL_NAME], dtype=np.float64)
    yaw = 2.0 * math.atan2(float(layout["gate_frame"][6]), float(layout["gate_frame"][3]))
    rotation_abs = np.abs(_yaw_rotation(yaw))
    footprints = (
        (sugar[:2], rotation_abs @ YCB01_SUGAR_HALF_EXTENTS_M[:2]),
        (cracker[:2], rotation_abs @ YCB01_CRACKER_HALF_EXTENTS_M[:2]),
        (ball[:2], np.full(2, YCB01_BALL_RADIUS_M, dtype=np.float64)),
    )
    camera_lower, camera_upper = YCB01_CAMERA_XY_BOUNDS
    table_lower = np.asarray([-0.55, -0.80], dtype=np.float64)
    table_upper = np.asarray([0.25, 0.80], dtype=np.float64)
    if any(
        np.any(center < camera_lower)
        or np.any(center > camera_upper)
        or np.any(center - half < table_lower)
        or np.any(center + half > table_upper)
        for center, half in footprints
    ):
        return False
    # Keep every accepted body in the broad head-camera ellipse.  Shrinking
    # the radii by the footprint and image reserve screens the full object,
    # rather than merely its centre, at arbitrary sampled yaw.
    camera_center = YCB01_CAMERA_ELLIPSE_CENTER_XY
    for center, half in footprints:
        effective = YCB01_CAMERA_ELLIPSE_RADII_M - (
            float(np.max(half)) + YCB01_CAMERA_CLEARANCE_M
        )
        if np.any(effective <= 0.0):
            return False
        if float(np.linalg.norm((center - camera_center) / effective)) > 1.0:
            return False
    local_ball = world_to_ycb01_gate(ball[:3], layout["gate_frame"])
    if abs(float(local_ball[1])) < (
        YCB01_BALL_RADIUS_M + YCB01_START_SIDE_MARGIN_M
    ):
        return False
    # Reject an actual sphere/box overlap as well as a bad start-side sample.
    # This matters for wide lateral starts that sit beside a carton corner;
    # checking only the gate-frame Y coordinate would allow those seeds to
    # begin with an immediate physical contact/reset.
    return not any(
        ycb01_sphere_box_contact(
            ball[:3],
            YCB01_BALL_RADIUS_M,
            layout[name],
            half_extents,
            extra_margin=YCB01_GATE_CLEARANCE_MARGIN_M,
        )
        for name, half_extents in (
            (YCB01_LEFT_GATE_NAME, YCB01_SUGAR_HALF_EXTENTS_M),
            (YCB01_RIGHT_GATE_NAME, YCB01_CRACKER_HALF_EXTENTS_M),
        )
    )


def sample_ycb01_layout(
    random_state: Any,
    *,
    randomize: bool,
    table_top_z: float,
) -> dict[str, np.ndarray]:
    """Sample one seeded, collision-clear gate-and-ball reset layout."""
    for _ in range(YCB01_LAYOUT_ATTEMPTS):
        if randomize:
            gate_center = YCB01_GATE_CENTER_XY.copy()
            yaw_degrees = 0.0
            ball_local_x = float(
                random_state.uniform(-YCB01_BALL_LOCAL_X_LIMIT_M, YCB01_BALL_LOCAL_X_LIMIT_M)
            )
            ball_local_y = float(
                random_state.uniform(*YCB01_BALL_LOCAL_Y_MAGNITUDE_RANGE_M)
            )
            if float(random_state.uniform()) < 0.5:
                ball_local_y *= -1.0
        else:
            gate_center = YCB01_GATE_CENTER_XY.copy()
            yaw_degrees = 0.0
            ball_local_x = 0.0
            # Keep the non-randomized preview/reset pose near the historical
            # centre start; the broad envelope is for seeded collection only.
            ball_local_y = -0.190

        yaw = math.radians(yaw_degrees)
        rotation = _yaw_rotation(yaw)
        quaternion = _yaw_quaternion_wxyz(yaw)
        sugar_local = np.asarray(
            [-(0.5 * YCB01_GATE_WIDTH_M + YCB01_SUGAR_HALF_EXTENTS_M[0]), 0.0],
            dtype=np.float64,
        )
        cracker_local = np.asarray(
            [0.5 * YCB01_GATE_WIDTH_M + YCB01_CRACKER_HALF_EXTENTS_M[0], 0.0],
            dtype=np.float64,
        )
        ball_local = np.asarray([ball_local_x, ball_local_y], dtype=np.float64)
        layout = {
            YCB01_LEFT_GATE_NAME: np.concatenate(
                [
                    gate_center + rotation @ sugar_local,
                    [table_top_z + YCB01_SUGAR_HALF_EXTENTS_M[2]],
                    quaternion,
                ]
            ),
            YCB01_RIGHT_GATE_NAME: np.concatenate(
                [
                    gate_center + rotation @ cracker_local,
                    [table_top_z + YCB01_CRACKER_HALF_EXTENTS_M[2]],
                    quaternion,
                ]
            ),
            YCB01_BALL_NAME: np.concatenate(
                [
                    gate_center + rotation @ ball_local,
                    [table_top_z + YCB01_BALL_RADIUS_M],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ),
            "gate_frame": np.concatenate(
                [gate_center, [table_top_z], quaternion]
            ),
        }
        if ycb01_layout_is_valid(layout):
            return {name: np.asarray(pose, dtype=np.float64) for name, pose in layout.items()}

    raise RuntimeError("YCB01 could not sample a complete collision-clear reset layout")


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def world_to_ycb01_gate(world_point: np.ndarray, gate_frame_pose: np.ndarray) -> np.ndarray:
    frame = np.asarray(gate_frame_pose, dtype=np.float64).reshape(7)
    rotation = quaternion_wxyz_to_matrix(frame[3:7])
    return rotation.T @ (np.asarray(world_point, dtype=np.float64).reshape(3) - frame[:3])


def ycb01_crossing_metrics(
    previous_world: np.ndarray | None,
    current_world: np.ndarray,
    gate_frame_pose: np.ndarray,
    *,
    direction_sign: float = 1.0,
) -> dict[str, Any]:
    """Evaluate centre and far-face Y crossings with interpolation.

    ``center_crossed_now`` records the direction-sensitive gate-plane event;
    ``full_pass_crossed_now`` records the later far-face Y event.  Keeping
    both events in one metric lets the runtime distinguish a centre-only
    passage while still certifying a single high-speed segment that crosses
    both boundaries.
    ``direction_sign`` selects which side is the start side: ``+1`` means
    local ``-Y`` to ``+Y`` (the historical direction), while ``-1`` mirrors
    the same contract for local ``+Y`` to ``-Y`` starts.
    """
    direction = -1.0 if float(direction_sign) < 0.0 else 1.0
    current_local = world_to_ycb01_gate(current_world, gate_frame_pose)
    current_progress = direction * float(current_local[1])
    full_pass_threshold = float(YCB01_FULL_PASS_LOCAL_Y_THRESHOLD_M)
    metrics: dict[str, Any] = {
        "current_local": current_local,
        "direction_sign": direction,
        "center_crossed_now": False,
        "inside_opening_now": False,
        "intersection_local": np.full(3, np.nan, dtype=np.float64),
        "full_pass_threshold": full_pass_threshold,
        "full_passed_now": bool(current_progress >= full_pass_threshold),
        "full_pass_crossed_now": False,
        "full_pass_inside_opening_now": False,
        "full_pass_intersection_local": np.full(3, np.nan, dtype=np.float64),
    }
    if previous_world is None:
        return metrics
    previous_local = world_to_ycb01_gate(previous_world, gate_frame_pose)
    previous_progress = direction * float(previous_local[1])
    denominator = float(current_progress - previous_progress)
    if denominator <= 1.0e-12:
        return metrics
    if previous_progress < 0.0 <= current_progress:
        alpha = float(np.clip(-previous_progress / denominator, 0.0, 1.0))
        intersection = previous_local + alpha * (current_local - previous_local)
        metrics.update(
            {
                "center_crossed_now": True,
                "inside_opening_now": (
                    abs(float(intersection[0]))
                    <= YCB01_VALID_OPENING_HALF_WIDTH_M
                ),
                "intersection_local": intersection,
            }
        )
    # Detect a complete pass even when a high-speed step jumps from the start
    # side to beyond the rear face in one sample.  This is deliberately a
    # separate event from the centre-plane crossing above.
    full_pass_crossed_now = bool(
        previous_progress < full_pass_threshold
        and current_progress >= full_pass_threshold
    )
    metrics["full_pass_crossed_now"] = full_pass_crossed_now
    if full_pass_crossed_now:
        alpha = float(
            np.clip(
                (full_pass_threshold - previous_progress) / denominator,
                0.0,
                1.0,
            )
        )
        far_intersection = previous_local + alpha * (current_local - previous_local)
        metrics["full_pass_intersection_local"] = far_intersection
        metrics["full_pass_inside_opening_now"] = bool(
            abs(float(far_intersection[0])) <= YCB01_VALID_OPENING_HALF_WIDTH_M
        )
    elif current_progress >= full_pass_threshold:
        # This branch is useful when a caller starts evaluating after the
        # threshold was crossed.  The normal runtime latches on the explicit
        # crossing branch above, but retaining a conservative current-position
        # check keeps the metric meaningful for replay/diagnostic consumers.
        metrics["full_pass_inside_opening_now"] = bool(
            abs(float(current_local[0])) <= YCB01_VALID_OPENING_HALF_WIDTH_M
        )
    return metrics


def ycb01_gate_pose_violation(
    current_pose: np.ndarray,
    reset_pose: np.ndarray,
) -> tuple[bool, float, float]:
    current = np.asarray(current_pose, dtype=np.float64).reshape(7)
    reset = np.asarray(reset_pose, dtype=np.float64).reshape(7)
    translation = float(np.linalg.norm(current[:3] - reset[:3]))
    current_up = quaternion_wxyz_to_matrix(current[3:7])[:, 2]
    reset_up = quaternion_wxyz_to_matrix(reset[3:7])[:, 2]
    cosine = float(np.clip(np.dot(current_up, reset_up), -1.0, 1.0))
    tilt_degrees = float(math.degrees(math.acos(cosine)))
    return (
        translation > YCB01_GATE_TRANSLATION_LIMIT_M
        or tilt_degrees > YCB01_GATE_TILT_LIMIT_DEG,
        translation,
        tilt_degrees,
    )


def ycb01_sphere_box_contact(
    sphere_center_world: np.ndarray,
    sphere_radius: float,
    box_pose: np.ndarray,
    box_half_extents: np.ndarray,
    *,
    extra_margin: float = 0.0,
) -> bool:
    """Return whether a sphere touches an oriented YCB box.

    This is a deterministic geometric backstop for PhysX contact tensors. It
    is intentionally one-sided (sphere against the measured box envelope),
    so it can detect a contact even when a high-speed CCD impact has no
    non-zero force in the sampled tensor yet.
    """

    center = np.asarray(sphere_center_world, dtype=np.float64).reshape(3)
    pose = np.asarray(box_pose, dtype=np.float64).reshape(7)
    half = np.asarray(box_half_extents, dtype=np.float64).reshape(3)
    if (
        not np.all(np.isfinite(center))
        or not np.all(np.isfinite(pose))
        or not np.all(np.isfinite(half))
        or np.any(half < 0.0)
    ):
        return False
    rotation = quaternion_wxyz_to_matrix(pose[3:7])
    local = rotation.T @ (center - pose[:3])
    outside = np.maximum(np.abs(local) - half, 0.0)
    clearance = max(0.0, float(sphere_radius) + float(extra_margin))
    return bool(float(np.linalg.norm(outside)) <= clearance)


__all__ = [name for name in globals() if name.startswith("YCB01_")] + [
    "quaternion_wxyz_to_matrix",
    "sample_ycb01_layout",
    "world_to_ycb01_gate",
    "ycb01_crossing_metrics",
    "ycb01_gate_pose_violation",
    "ycb01_sphere_box_contact",
    "ycb01_layout_is_valid",
]
