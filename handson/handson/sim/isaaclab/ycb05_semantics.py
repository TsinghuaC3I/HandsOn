"""Pure reset geometry and ordered success semantics for YCB05.

This module has no Isaac imports.  Live collection, unit tests, and replay use
the same source-scale geometry and the same ordered state machine.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


YCB05_TASK_NAME = "t47_trigger_drill_on_block"
YCB05_DRILL_NAME = "ycb05_power_drill"
YCB05_TRIGGER_BODY_NAME = "power_drill_Trigger"
YCB05_TRIGGER_JOINT_NAME = "power_drill_Trigger_joint"
YCB05_WOOD_NAME = "ycb05_wood_block"

# Exact union bounds of the source collision meshes.  No YCB05 asset is scaled.
YCB05_DRILL_LOCAL_MIN_M = np.asarray(
    [-0.029029, -0.091565, -0.093721], dtype=np.float64
)
YCB05_DRILL_LOCAL_MAX_M = np.asarray(
    [0.029029, 0.091565, 0.093721], dtype=np.float64
)
# The source-scale 13-hull drill has one stable table-supported attitude rather
# than resting exactly on its scan-frame AABB minimum.  This pose and support
# height are the converged Isaac Sim 5.1 public PhysX readback; yaw is composed
# around world Z below so randomization preserves the same physical support.
YCB05_DRILL_SUPPORT_QUATERNION_WXYZ = np.asarray(
    [
        0.9997169375419617,
        0.009390732273459435,
        0.021856704726815224,
        -1.6212725384434634e-08,
    ],
    dtype=np.float64,
)
YCB05_DRILL_SUPPORT_ROOT_HEIGHT_M = 0.0938064312934875
YCB05_WOOD_HALF_EXTENTS_M = np.asarray(
    [0.0433705, 0.100369, 0.042689], dtype=np.float64
)
YCB05_TRIGGER_RANGE_M = 0.004

# The source drill points along local +Y.  The local point is the centre of
# the front-most chuck/nose plane, inferred from the unscaled source mesh.
YCB05_DRILL_TIP_LOCAL_M = np.asarray([0.0, 0.091565, 0.060], dtype=np.float64)
YCB05_DRILL_AXIS_LOCAL = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
# Aim at the square near end of the long YCB wood block.
YCB05_WOOD_TARGET_LOCAL_M = np.asarray(
    [0.0, -YCB05_WOOD_HALF_EXTENTS_M[1], 0.0], dtype=np.float64
)

# Handle-region bounds cover only the source handle colliders (7--13), not the
# barrel or battery.  They are semantic geometry and never render or collide.
YCB05_HANDLE_LOCAL_MIN_M = np.asarray(
    [-0.017632, -0.077965, -0.060380], dtype=np.float64
)
YCB05_HANDLE_LOCAL_MAX_M = np.asarray(
    [0.017497, 0.021276, 0.047833], dtype=np.float64
)
YCB05_HANDLE_PROXIMITY_MARGIN_M = 0.025

YCB05_DRILL_DEFAULT_XY = np.asarray([-0.275, -0.235], dtype=np.float64)
# The old non-random pose intersects the reset right-thumb envelope. Keep the
# historical randomized centre unchanged; fixed layouts clear it by +40 mm X.
YCB05_DRILL_FIXED_XY = np.asarray([-0.235, -0.235], dtype=np.float64)
YCB05_WOOD_DEFAULT_XY = np.asarray([-0.275, 0.145], dtype=np.float64)
YCB05_DRILL_XY_JITTER = np.asarray([0.075, 0.065], dtype=np.float64)
YCB05_WOOD_XY_JITTER = np.asarray([0.075, 0.065], dtype=np.float64)
YCB05_DRILL_YAW_LIMIT_DEG = 20.0
YCB05_WOOD_YAW_LIMIT_DEG = 20.0
YCB05_LAYOUT_ATTEMPTS = 64
YCB05_MIN_ROOT_SEPARATION_M = 0.30
YCB05_CAMERA_XY_BOUNDS = np.asarray(
    [[-0.40, -0.36], [-0.13, 0.29]], dtype=np.float64
)

YCB05_SETTLE_SECONDS = 0.25
YCB05_LIFT_CLEARANCE_M = 0.030
YCB05_GRASP_DWELL_SECONDS = 0.30
YCB05_AIM_DWELL_SECONDS = 0.30
YCB05_PRESS_DWELL_SECONDS = 0.50
YCB05_RELEASE_DWELL_SECONDS = 0.20
YCB05_TARGET_CONTACT_DEBOUNCE_SECONDS = 0.05
YCB05_EARLY_PRESS_DEBOUNCE_SECONDS = 0.15
YCB05_PRESS_AIM_LOSS_DEBOUNCE_SECONDS = 0.15

YCB05_AIM_MIN_DISTANCE_M = 0.08
YCB05_AIM_MAX_DISTANCE_M = 0.20
YCB05_AIM_MAX_ANGLE_DEG = 12.0
YCB05_AIM_MAX_RAY_ERROR_M = 0.025
YCB05_PRESS_MAX_ANGLE_DEG = 15.0
YCB05_PRESS_MAX_RAY_ERROR_M = 0.030
YCB05_TRIGGER_LIGHT_FRACTION = 0.30
YCB05_TRIGGER_EARLY_HARD_FRACTION = 0.70
YCB05_TRIGGER_PRESS_FRACTION = 0.80
YCB05_TRIGGER_RELEASE_FRACTION = 0.20
YCB05_WOOD_TRANSLATION_LIMIT_M = 0.010
YCB05_CONTACT_FORCE_FLOOR_N = 1.0e-3
YCB05_GRASP_CONTACT_FORCE_N = 1.0e-3
YCB05_MAX_GRASP_LINEAR_SPEED_MPS = 0.35
YCB05_MAX_GRASP_ANGULAR_SPEED_RADPS = 3.0
YCB05_WORKSPACE_XY_BOUNDS = np.asarray(
    [[-0.55, -0.65], [0.18, 0.65]], dtype=np.float64
)
YCB05_WORKSPACE_Z_RANGE_M = (0.82, 1.65)

YCB05_SEMANTICS_REVISION = 2
YCB05_STAGE_CONTACT = 1
YCB05_REQUIRED_CONTACT_CHANNELS = ("wood_drill", "trigger_index", "trigger_wrong")

YCB05_STAGE_SETTLING = 0
YCB05_STAGE_GRASP = 1
YCB05_STAGE_AIM = 2
YCB05_STAGE_PRESS = 3
YCB05_STAGE_RELEASE = 4
YCB05_STAGE_SUCCESS = 5
YCB05_STAGE_FAILURE = -1
YCB05_STAGE_NAMES = {
    YCB05_STAGE_SETTLING: "settling",
    YCB05_STAGE_CONTACT: "contact_wood",
    YCB05_STAGE_AIM: "aim",
    YCB05_STAGE_PRESS: "press",
    YCB05_STAGE_RELEASE: "release",
    YCB05_STAGE_SUCCESS: "success",
    YCB05_STAGE_FAILURE: "failure",
}
YCB05_CONTACT_CHANNELS = (
    "trigger_index", "trigger_wrong", "handle_thumb", "handle_middle",
    "handle_ring", "handle_pinky", "wood_drill", "wood_hand",
    "drill_table", "trigger_table",
)

YCB05_FAILURE_REASON_CODES = {
    "invalid_contact_state": 10,
    "non_finite_or_broken_joint": 1,
    "drill_exited_workspace": 2,
    "wood_moved_over_10mm": 3,
    "hand_or_drill_contacted_wood": 4,
    "lifted_drill_hit_table": 5,
    "lifted_drill_dropped": 6,
    "substantial_trigger_press_before_valid_aim": 7,
    "non_index_or_palm_pressed_trigger_without_grasp": 8,
    "aim_lost_during_trigger_press": 9,
}
YCB05_FAILURE_CODE_REASONS = {
    code: reason for reason, code in YCB05_FAILURE_REASON_CODES.items()
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


def _yaw_quaternion_wxyz(yaw_radians: float) -> np.ndarray:
    return np.asarray(
        [math.cos(0.5 * yaw_radians), 0.0, 0.0, math.sin(0.5 * yaw_radians)],
        dtype=np.float64,
    )


def _yaw_then_support_quaternion_wxyz(yaw_radians: float) -> np.ndarray:
    """Compose world-Z yaw with the drill's source-scale support attitude."""

    yaw = _yaw_quaternion_wxyz(yaw_radians)
    support = YCB05_DRILL_SUPPORT_QUATERNION_WXYZ
    yw, yx, yy, yz = yaw
    sw, sx, sy, sz = support
    quaternion = np.asarray(
        [
            yw * sw - yx * sx - yy * sy - yz * sz,
            yw * sx + yx * sw + yy * sz - yz * sy,
            yw * sy - yx * sz + yy * sw + yz * sx,
            yw * sz + yx * sy - yy * sx + yz * sw,
        ],
        dtype=np.float64,
    )
    return quaternion / np.linalg.norm(quaternion)


def pose_transform_point(pose_wxyz: np.ndarray, local_point: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    return pose[:3] + quaternion_wxyz_to_matrix(pose[3:7]) @ np.asarray(
        local_point, dtype=np.float64
    ).reshape(3)


def ycb05_layout_is_valid(layout: dict[str, np.ndarray]) -> bool:
    drill = np.asarray(layout[YCB05_DRILL_NAME], dtype=np.float64).reshape(7)
    wood = np.asarray(layout[YCB05_WOOD_NAME], dtype=np.float64).reshape(7)
    lower, upper = YCB05_CAMERA_XY_BOUNDS
    if np.any(drill[:2] < lower) or np.any(drill[:2] > upper):
        return False
    if np.any(wood[:2] < lower) or np.any(wood[:2] > upper):
        return False
    if float(np.linalg.norm(drill[:2] - wood[:2])) < YCB05_MIN_ROOT_SEPARATION_M:
        return False
    metrics = ycb05_aim_metrics(drill, wood)
    # A reset need not satisfy task aim, but it must put the intended target in
    # front of the drill and close enough to be reached without a large sweep.
    return bool(
        metrics["projection_m"] > 0.10
        and metrics["tip_distance_m"] < 0.35
        and metrics["aim_angle_deg"] < 45.0
    )


def sample_ycb05_layout(
    random_state: Any,
    *,
    randomize: bool,
    table_top_z: float,
) -> dict[str, np.ndarray]:
    """Sample a central, reachable, source-scale two-object layout."""
    for _ in range(YCB05_LAYOUT_ATTEMPTS):
        if randomize:
            drill_xy = YCB05_DRILL_DEFAULT_XY + random_state.uniform(
                -YCB05_DRILL_XY_JITTER, YCB05_DRILL_XY_JITTER
            )
            wood_xy = YCB05_WOOD_DEFAULT_XY + random_state.uniform(
                -YCB05_WOOD_XY_JITTER, YCB05_WOOD_XY_JITTER
            )
            drill_yaw = math.radians(
                float(random_state.uniform(-YCB05_DRILL_YAW_LIMIT_DEG, YCB05_DRILL_YAW_LIMIT_DEG))
            )
            wood_yaw = math.radians(
                float(random_state.uniform(-YCB05_WOOD_YAW_LIMIT_DEG, YCB05_WOOD_YAW_LIMIT_DEG))
            )
        else:
            drill_xy = YCB05_DRILL_FIXED_XY.copy()
            wood_xy = YCB05_WOOD_DEFAULT_XY.copy()
            drill_yaw = 0.0
            wood_yaw = 0.0

        layout = {
            YCB05_DRILL_NAME: np.concatenate(
                [
                    drill_xy,
                    [table_top_z + YCB05_DRILL_SUPPORT_ROOT_HEIGHT_M],
                    _yaw_then_support_quaternion_wxyz(drill_yaw),
                ]
            ),
            YCB05_WOOD_NAME: np.concatenate(
                [
                    wood_xy,
                    [table_top_z + float(YCB05_WOOD_HALF_EXTENTS_M[2])],
                    _yaw_quaternion_wxyz(wood_yaw),
                ]
            ),
        }
        if ycb05_layout_is_valid(layout):
            return {name: np.asarray(value, dtype=np.float64) for name, value in layout.items()}
    raise RuntimeError("YCB05 could not sample a collision-clear reachable reset layout")


def ycb05_aim_metrics(
    drill_pose_wxyz: np.ndarray,
    wood_pose_wxyz: np.ndarray,
) -> dict[str, Any]:
    drill_pose = np.asarray(drill_pose_wxyz, dtype=np.float64).reshape(7)
    wood_pose = np.asarray(wood_pose_wxyz, dtype=np.float64).reshape(7)
    drill_rotation = quaternion_wxyz_to_matrix(drill_pose[3:7])
    tip = drill_pose[:3] + drill_rotation @ YCB05_DRILL_TIP_LOCAL_M
    axis = drill_rotation @ YCB05_DRILL_AXIS_LOCAL
    axis_norm = float(np.linalg.norm(axis))
    axis = axis / max(axis_norm, 1.0e-12)
    target = pose_transform_point(wood_pose, YCB05_WOOD_TARGET_LOCAL_M)
    delta = target - tip
    distance = float(np.linalg.norm(delta))
    desired = delta / max(distance, 1.0e-12)
    cosine = float(np.clip(np.dot(axis, desired), -1.0, 1.0))
    angle_degrees = float(math.degrees(math.acos(cosine)))
    projection = float(np.dot(delta, axis))
    closest = tip + projection * axis
    ray_error = float(np.linalg.norm(target - closest))
    finite = bool(
        np.all(np.isfinite(drill_pose))
        and np.all(np.isfinite(wood_pose))
        and np.all(np.isfinite(tip))
        and np.all(np.isfinite(axis))
        and np.all(np.isfinite(target))
    )
    return {
        "tip_world": tip,
        "axis_world": axis,
        "target_world": target,
        "tip_distance_m": distance,
        "projection_m": projection,
        "aim_angle_deg": angle_degrees,
        "ray_error_m": ray_error,
        "finite": finite,
        "strict_aim": bool(
            finite
            and YCB05_AIM_MIN_DISTANCE_M <= distance <= YCB05_AIM_MAX_DISTANCE_M
            and projection > 0.0
            and angle_degrees <= YCB05_AIM_MAX_ANGLE_DEG
            and ray_error <= YCB05_AIM_MAX_RAY_ERROR_M
        ),
        "press_aim": bool(
            finite
            and YCB05_AIM_MIN_DISTANCE_M <= distance <= YCB05_AIM_MAX_DISTANCE_M
            and projection > 0.0
            and angle_degrees <= YCB05_PRESS_MAX_ANGLE_DEG
            and ray_error <= YCB05_PRESS_MAX_RAY_ERROR_M
        ),
    }


def ycb05_point_in_handle_region(
    drill_pose_wxyz: np.ndarray,
    point_world: np.ndarray,
    *,
    margin_m: float = YCB05_HANDLE_PROXIMITY_MARGIN_M,
) -> bool:
    pose = np.asarray(drill_pose_wxyz, dtype=np.float64).reshape(7)
    rotation = quaternion_wxyz_to_matrix(pose[3:7])
    local = rotation.T @ (np.asarray(point_world, dtype=np.float64).reshape(3) - pose[:3])
    return bool(
        np.all(local >= YCB05_HANDLE_LOCAL_MIN_M - float(margin_m))
        and np.all(local <= YCB05_HANDLE_LOCAL_MAX_M + float(margin_m))
    )


def initial_ycb05_state() -> dict[str, Any]:
    return {
        "stage": YCB05_STAGE_SETTLING,
        "settle_s": 0.0,
        "grasp_s": 0.0,
        "aim_s": 0.0,
        "press_s": 0.0,
        "release_s": 0.0,
        "target_contact_s": 0.0,
        "early_press_s": 0.0,
        "press_aim_loss_s": 0.0,
        "grasp_seen": False,
        "lift_seen": False,
        "aim_dwell_seen": False,
        "trigger_contact_seen": False,
        "valid_press_seen": False,
        "trigger_release_seen": False,
        "wrong_order_press_seen": False,
        "target_contact_seen": False,
        "drop_seen": False,
        "success": False,
        "failure_reason": None,
        "invalid_contact_channels": [],
    }


def _fail_ycb05(state: dict[str, Any], reason: str) -> None:
    if state.get("failure_reason") is None:
        state["failure_reason"] = str(reason)
    state["stage"] = YCB05_STAGE_FAILURE
    state["success"] = False


def update_ycb05_state(
    state: dict[str, Any],
    sample: dict[str, Any],
    dt: float,
) -> dict[str, Any]:
    """Revision 2: physical drill/wood contact, then a real trigger press.

    Grasp, lift, aim, support-finger identity and final trigger release are
    diagnostic only. An early press is recoverable: release while touching
    the wood to arm a new press. Contact loss cancels the dwell and arming.
    Stage encodes arming, so all state remains in recorded stage/timer/flags.
    """
    dt = float(dt)
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("YCB05 dt must be finite and nonnegative")
    if int(state.get("stage", YCB05_STAGE_SETTLING)) in {
        YCB05_STAGE_SUCCESS, YCB05_STAGE_FAILURE,
    }:
        return state
    if not bool(sample.get("contacts_valid", True)):
        state["invalid_contact_channels"] = list(sample.get("invalid_contact_channels", ()))
        _fail_ycb05(state, "invalid_contact_state")
        return state
    trigger_fraction = float(sample.get("trigger_fraction", 0.0))
    if not bool(sample.get("finite", True)) or not math.isfinite(trigger_fraction):
        _fail_ycb05(state, "non_finite_or_broken_joint")
        return state
    if not bool(sample.get("workspace_ok", True)):
        _fail_ycb05(state, "drill_exited_workspace")
        return state

    # Never substitute hand/wood contact or geometric proximity for this.
    wood_contact = bool(sample.get("wood_drill_contact", False))
    trigger_contact = bool(sample.get("index_trigger_contact", False)
                           or sample.get("wrong_trigger_contact", False))
    stage = int(state.get("stage", YCB05_STAGE_SETTLING))
    if stage == YCB05_STAGE_SETTLING:
        state["settle_s"] = float(state.get("settle_s", 0.0)) + dt
        if state["settle_s"] >= YCB05_SETTLE_SECONDS:
            state["stage"] = YCB05_STAGE_CONTACT
        return state

    state["target_contact_s"] = (float(state.get("target_contact_s", 0.0)) + dt
                                 if wood_contact else 0.0)
    if not wood_contact:
        state["stage"] = YCB05_STAGE_CONTACT
        state["press_s"] = 0.0
        return state
    if state["target_contact_s"] >= YCB05_TARGET_CONTACT_DEBOUNCE_SECONDS:
        state["target_contact_seen"] = True
    if stage != YCB05_STAGE_PRESS:
        if trigger_fraction >= YCB05_TRIGGER_PRESS_FRACTION:
            state["wrong_order_press_seen"] = True  # diagnostic, not failure
        if (state["target_contact_s"] >= YCB05_TARGET_CONTACT_DEBOUNCE_SECONDS
                and trigger_fraction <= YCB05_TRIGGER_RELEASE_FRACTION):
            state["stage"] = YCB05_STAGE_PRESS
        return state

    valid_press = trigger_contact and trigger_fraction >= YCB05_TRIGGER_PRESS_FRACTION
    state["trigger_contact_seen"] |= trigger_contact
    state["press_s"] = float(state.get("press_s", 0.0)) + dt if valid_press else 0.0
    if state["press_s"] >= YCB05_PRESS_DWELL_SECONDS:
        state["valid_press_seen"] = True
        state["success"] = True
        state["stage"] = YCB05_STAGE_SUCCESS
    return state


def ycb05_event_flags(state: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(bool(state.get("grasp_seen"))),
            float(bool(state.get("lift_seen"))),
            float(bool(state.get("aim_dwell_seen"))),
            float(bool(state.get("trigger_contact_seen"))),
            float(bool(state.get("valid_press_seen"))),
            float(bool(state.get("trigger_release_seen"))),
            float(bool(state.get("wrong_order_press_seen"))),
            float(bool(state.get("target_contact_seen"))),
            float(bool(state.get("drop_seen"))),
            float(bool(state.get("success"))),
            float(state.get("failure_reason") is not None),
        ],
        dtype=np.float64,
    )


def ycb05_timer_vector(state: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(state.get("settle_s", 0.0)),
            float(state.get("grasp_s", 0.0)),
            float(state.get("aim_s", 0.0)),
            float(state.get("press_s", 0.0)),
            float(state.get("release_s", 0.0)),
            float(state.get("target_contact_s", 0.0)),
            float(state.get("early_press_s", 0.0)),
            float(state.get("press_aim_loss_s", 0.0)),
        ],
        dtype=np.float64,
    )


def ycb05_terminal_reason_code(state: dict[str, Any]) -> np.ndarray:
    """Encode the immutable physical failure reason in a fixed replay field."""
    reason = state.get("failure_reason")
    return np.asarray(
        [float(YCB05_FAILURE_REASON_CODES.get(str(reason), 0))],
        dtype=np.float64,
    )


__all__ = [name for name in globals() if name.startswith("YCB05_")] + [
    "initial_ycb05_state",
    "pose_transform_point",
    "quaternion_wxyz_to_matrix",
    "sample_ycb05_layout",
    "update_ycb05_state",
    "ycb05_aim_metrics",
    "ycb05_event_flags",
    "ycb05_layout_is_valid",
    "ycb05_point_in_handle_region",
    "ycb05_timer_vector",
    "ycb05_terminal_reason_code",
]
