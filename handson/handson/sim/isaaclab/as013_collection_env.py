from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from .abc_collection_env import (
    _TABLE_TOP_Z,
    _TASK_ROOT,
    HandsOnABCIsaacLabEnv,
    _invert_pose_matrix,
    _matrix_to_quat_wxyz,
    _pose_vector_to_matrix,
    _randomized_physx_pose,
    _torch_view,
)

TASK_NAME = "t40_slide_knife_blade"
TASK_CODE = "B320-AS013"
# AS013 role schema v5 is a handheld bimanual task.  At least two distinct
# physical right-hand rigid bodies grasp and lift the housing; no individual
# palm contact is mandatory.  After pickup, any physical left fingertip may
# touch the button/slider.  Each reset then asks for either the fully retracted
# (bottom/inside) or fully extended (top/outside) endpoint.  Touch and pickup
# are episode latches; current contact and a terminal stability hold are not
# success requirements.
# Historical v1--v4 fields remain importable aliases for sealed diagnostics;
# canonical runtime fields always state their physical owner.
ROLE_SCHEMA_VERSION = 5
ORDERED_CONTACT_SCHEMA_VERSION = 2
SLIDER_HAND_SIDE = "left"
STABILIZER_HAND_SIDE = "right"
INACTIVE_HAND_SIDE = ""
RIGHT_THUMB_DISTAL_RIGID_BODY_NAME = "right_finger1_link4"
RIGHT_THUMB_TIP_COLLIDER_NAME = "right_finger1_tip_link_col2"
LEFT_SLIDER_FINGERTIP_RIGID_BODY_NAMES = tuple(
    f"left_finger{finger_number}_link4" for finger_number in range(1, 6)
)
_AS013_CONTACT_VIEW_NAMES = frozenset(
    {
        "right_grip_housing",
        "right_palm_housing",
        "right_non_thumb_housing",
        "left_slider_moving",
        "right_unsafe_moving",
        "left_housing",
        "left_moving",
        "housing_table",
        "moving_table",
    }
)
_AS013_CONTACT_VIEW_FILTER_COUNTS = {
    "right_grip_housing": 21,
    "right_palm_housing": 1,
    "right_non_thumb_housing": 16,
    "left_slider_moving": 5,
    "right_unsafe_moving": 20,
    "left_housing": 21,
    "left_moving": 21,
    "housing_table": 1,
    "moving_table": 1,
}
SCRIPTED_DIAGNOSTIC_ENV = "HANDSON_AS013_SCRIPTED_DIAGNOSTICS"
SCRIPTED_LEFT_HEIGHT_ENV = "HANDSON_AS013_SCRIPTED_LEFT_HEIGHT_M"
SCRIPTED_LEFT_COUNTER_SLIDE_ENV = (
    "HANDSON_AS013_SCRIPTED_LEFT_COUNTER_SLIDE_M"
)
SCRIPTED_LEFT_HAND_PRELOAD_ENV = (
    "HANDSON_AS013_SCRIPTED_LEFT_HAND_PRELOAD_FRACTION"
)
SCRIPTED_LEFT_SAFE_FINGER_PRELOAD_ENV = (
    "HANDSON_AS013_SCRIPTED_LEFT_SAFE_FINGER_PRELOAD_FRACTION"
)
SCRIPTED_LEFT_UNSAFE_FINGER_RETRACTION_ENV = (
    "HANDSON_AS013_SCRIPTED_LEFT_UNSAFE_FINGER_RETRACTION_FRACTION"
)
SCRIPTED_LEFT_UNSAFE_FINGER_DISTAL_ONLY_ENV = (
    "HANDSON_AS013_SCRIPTED_LEFT_UNSAFE_FINGER_DISTAL_ONLY"
)
SCRIPTED_LEFT_APPROACH_GRIP_ENV = (
    "HANDSON_AS013_SCRIPTED_LEFT_APPROACH_GRIP_FRACTION"
)
SCRIPTED_RIGHT_HEIGHT_ENV = "HANDSON_AS013_SCRIPTED_RIGHT_HEIGHT_M"
SCRIPTED_RIGHT_FINGER_EXTENSION_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_FINGER_EXTENSION_FRACTION"
)
SCRIPTED_RIGHT_FINGER_ABDUCTION_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_FINGER_ABDUCTION_RAD"
)
SCRIPTED_RIGHT_FINGER_NUMBER_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_FINGER_NUMBER"
)
SCRIPTED_RIGHT_VERTICAL_GAIN_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_VERTICAL_GAIN"
)
SCRIPTED_RIGHT_LATERAL_GAIN_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_LATERAL_GAIN"
)
SCRIPTED_RIGHT_PALM_YAW_ENV = "HANDSON_AS013_SCRIPTED_RIGHT_PALM_YAW_DEG"
SCRIPTED_RIGHT_PALM_ROLL_ENV = "HANDSON_AS013_SCRIPTED_RIGHT_PALM_ROLL_DEG"
SCRIPTED_RIGHT_ALIGNMENT_REFERENCE_BLEND_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_ALIGNMENT_REFERENCE_BLEND"
)
SCRIPTED_RIGHT_FALLBACK_TARGET_ADAPT_FRAME_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_FALLBACK_TARGET_ADAPT_FRAME"
)
SCRIPTED_RIGHT_FALLBACK_ALIGNMENT_LOCK_FRAME_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_FALLBACK_ALIGNMENT_LOCK_FRAME"
)
SCRIPTED_RIGHT_SLIDE_MAX_LEAD_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_MAX_LEAD_M"
)
SCRIPTED_RIGHT_SLIDE_PALM_TRAVEL_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_PALM_TRAVEL_M"
)
SCRIPTED_RIGHT_SLIDE_NORMAL_PRELOAD_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_NORMAL_PRELOAD_M"
)
SCRIPTED_RIGHT_SLIDE_LATE_NORMAL_PRELOAD_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_LATE_NORMAL_PRELOAD_M"
)
SCRIPTED_RIGHT_SURFACE_SLOW_RADIUS_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SURFACE_SLOW_RADIUS_M"
)
SCRIPTED_RIGHT_SLIDE_CROSS_EDGE_MAX_CORRECTION_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_CROSS_EDGE_MAX_CORRECTION_M"
)
SCRIPTED_RIGHT_SLIDE_CROSS_EDGE_SERVO_SCALE_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_CROSS_EDGE_SERVO_SCALE"
)
SCRIPTED_RIGHT_SLIDE_YAW_RELIEF_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_YAW_RELIEF_DEG"
)
SCRIPTED_RIGHT_SLIDE_PITCH_RELIEF_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_PITCH_RELIEF_DEG"
)
SCRIPTED_RIGHT_SLIDE_ROLL_RELIEF_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_ROLL_RELIEF_DEG"
)
SCRIPTED_RIGHT_SLIDE_FINGER_EXTENSION_DELTA_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_FINGER_EXTENSION_DELTA"
)
SCRIPTED_RIGHT_SLIDE_LATE_FINGER_EXTENSION_DELTA_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_LATE_FINGER_EXTENSION_DELTA"
)
SCRIPTED_RIGHT_SLIDE_FINGER_ABDUCTION_DELTA_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_FINGER_ABDUCTION_DELTA_RAD"
)
SCRIPTED_RIGHT_SLIDE_CONTROL_FRAMES_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_CONTROL_FRAMES"
)
SCRIPTED_RIGHT_SLIDE_REANCHOR_PROGRESS_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_REANCHOR_PROGRESS"
)
SCRIPTED_RIGHT_SLIDE_REANCHOR_EXTRA_LEAD_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_REANCHOR_EXTRA_LEAD_M"
)
SCRIPTED_RIGHT_SLIDE_SERVO_CORRECTION_SCALE_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_SERVO_CORRECTION_SCALE"
)
SCRIPTED_RIGHT_SLIDE_LATE_SERVO_CORRECTION_SCALE_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_LATE_SERVO_CORRECTION_SCALE"
)
CAUSAL_FINGERTIP_DRIVE_ENV = "HANDSON_AS013_CAUSAL_FINGERTIP_DRIVE"
BLADE_DRIVE_STIFFNESS_SCALE_ENV = (
    "HANDSON_AS013_BLADE_DRIVE_STIFFNESS_SCALE"
)
BLADE_DRIVE_DAMPING_SCALE_ENV = "HANDSON_AS013_BLADE_DRIVE_DAMPING_SCALE"
BLADE_JOINT_FRICTION_SCALE_ENV = "HANDSON_AS013_BLADE_JOINT_FRICTION_SCALE"
BLADE_CONTACT_FRICTION_ENV = "HANDSON_AS013_BLADE_CONTACT_FRICTION"
RIGHT_IK_DLS_LAMBDA_ENV = "HANDSON_AS013_RIGHT_IK_DLS_LAMBDA"
SCRIPTED_RIGHT_RECONTACT_GRACE_FRAMES_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_RECONTACT_GRACE_FRAMES"
)
SCRIPTED_RIGHT_INITIAL_CONTACT_DROPOUT_GRACE_FRAMES_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_INITIAL_CONTACT_DROPOUT_GRACE_FRAMES"
)
SCRIPTED_LATE_LEFT_STABILITY_GRACE_START_PROGRESS_ENV = (
    "HANDSON_AS013_SCRIPTED_LATE_LEFT_STABILITY_GRACE_START_PROGRESS"
)
SCRIPTED_RIGHT_APPROACH_TIMEOUT_FRAME_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_APPROACH_TIMEOUT_FRAME"
)
SCRIPTED_RELEASE_CONTROL_FRAMES_ENV = (
    "HANDSON_AS013_SCRIPTED_RELEASE_CONTROL_FRAMES"
)
SCRIPTED_POST_SUCCESS_HOLD_CONTROL_FRAMES_ENV = (
    "HANDSON_AS013_SCRIPTED_POST_SUCCESS_HOLD_CONTROL_FRAMES"
)
SCRIPTED_RELEASE_DWELL_CONTROL_FRAMES_ENV = (
    "HANDSON_AS013_SCRIPTED_RELEASE_DWELL_CONTROL_FRAMES"
)
SCRIPTED_POST_SUCCESS_OVERTRAVEL_ENV = (
    "HANDSON_AS013_SCRIPTED_POST_SUCCESS_OVERTRAVEL_M"
)
SCRIPTED_RIGHT_RECOVERED_FINGER_EXTENSION_DELTA_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_FINGER_EXTENSION_DELTA"
)
SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_EXTENSION_DELTA_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_EXTENSION_DELTA"
)
SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS"
)
SCRIPTED_RIGHT_RECOVERED_LATE_PROXIMAL_EXTENSION_SCALE_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_PROXIMAL_EXTENSION_SCALE"
)
SCRIPTED_RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE"
)
SCRIPTED_RIGHT_RECOVERED_LATE_DISTAL_TIP_EXTENSION_SCALE_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_DISTAL_TIP_EXTENSION_SCALE"
)
SCRIPTED_RIGHT_RECOVERED_LATE_IN_PLANE_PRELOAD_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_IN_PLANE_PRELOAD_M"
)
SCRIPTED_RIGHT_RECOVERED_LATE_NORMAL_PRELOAD_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_NORMAL_PRELOAD_M"
)
SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_DELTA_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_DELTA_RAD"
)
SCRIPTED_RIGHT_LATE_CONTINUOUS_HAND_BLEND_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_LATE_CONTINUOUS_HAND_BLEND"
)
SCRIPTED_RIGHT_FINGER_REENTRY_CONTROL_FRAMES_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_FINGER_REENTRY_CONTROL_FRAMES"
)
SCRIPTED_RIGHT_RECONTACT_SERVO_STEP_ENV = (
    "HANDSON_AS013_SCRIPTED_RIGHT_RECONTACT_SERVO_STEP_M"
)
JOINT_NAME = "PrismaticJoint_utility_knife_3_middle"
JOINT_AXIS_BODY_LOCAL = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
# The selected replacement is authored directly in metres.  Keep the legacy
# values below as an explicit opt-in compatibility path; they are not applied
# to the default collection asset.
ASSET_VARIANT = "utility_knife_ahtusik"
ASSET_SCALE = 1.0
ASSET_REST_ROLL_DEG = 0.0
LEGACY_ASSET_SCALE = 1.3
LEGACY_ASSET_REST_ROLL_DEG = -1.9415
# The sealed d1afd9f physical NON_PASS showed that a 3 mm unsupported
# start can fall/rock before the quiet fallback, creating cumulative reset
# drift and a transitional invalid contact.  Start the composed collision
# bound on the table; the unchanged shared gate still accumulates every
# later frame against the authored reset and requires valid contacts.
TARGET_TABLE_CLEARANCE_M = 0.0
RESET_XY_HALF_RANGE_M = 0.05
# The knife stays flat on the table, but its in-plane direction is fully
# randomized.  Pitch/roll randomization would make a thin tool intersect or
# fall from the support plane at reset.
RESET_YAW_HALF_RANGE_DEG = 180.0
RESET_HAND_CLEARANCE_M = 0.075
# Target-aware start bands keep every sampled episode meaningfully away from
# its requested endpoint while still randomizing the slider position.
RESET_RETRACT_PROGRESS_RANGE = (0.55, 0.85)
RESET_EXTEND_PROGRESS_RANGE = (0.15, 0.45)
RESET_PROGRESS_RANGE = (
    RESET_EXTEND_PROGRESS_RANGE[0],
    RESET_RETRACT_PROGRESS_RANGE[1],
)
TARGET_ENDPOINT_RETRACTED = "retracted_bottom"
TARGET_ENDPOINT_EXTENDED = "extended_top"
TARGET_DIRECTION_RETRACT = -1
TARGET_DIRECTION_EXTEND = 1
RETRACT_SUCCESS_PROGRESS_MAX = 0.05
EXTEND_SUCCESS_PROGRESS_MIN = 0.95
# Retained for replay consumers that interpret the former one-way threshold.
SUCCESS_PROGRESS = 0.80
COMPLETION_FORWARD_PROGRESS_EPS = 1.0e-5
CONTACT_FORCE_THRESHOLD_N = 0.01
# PhysX contacts are sampled at 120 Hz.  A single 0.01 N row can be a contact
# manifold transition, so terminal contact predicates need one complete 30 Hz
# control interval rather than latching on one physics substep.
CONTACT_EVENT_DEBOUNCE_SUBSTEPS = 4
UNSAFE_BLADE_CONTACT_CONSECUTIVE_SUBSTEPS = (
    CONTACT_EVENT_DEBOUNCE_SUBSTEPS
)
ORDER_VIOLATION_CONSECUTIVE_SUBSTEPS = CONTACT_EVENT_DEBOUNCE_SUBSTEPS
PRESTABILIZATION_PROGRESS_CONSECUTIVE_SUBSTEPS = (
    CONTACT_EVENT_DEBOUNCE_SUBSTEPS
)
PICKUP_LIFT_HEIGHT_M = 0.020
PICKUP_CONSECUTIVE_SUBSTEPS = CONTACT_EVENT_DEBOUNCE_SUBSTEPS
LEFT_HAND_PARTICIPATION_CONSECUTIVE_SUBSTEPS = (
    CONTACT_EVENT_DEBOUNCE_SUBSTEPS
)
DROP_CONSECUTIVE_SUBSTEPS = CONTACT_EVENT_DEBOUNCE_SUBSTEPS
DROP_RETURN_LIFT_HEIGHT_M = 0.005
DROP_DESCENT_FROM_PEAK_M = 0.050
DROP_FALLING_VERTICAL_SPEED_MPS = -0.020
DROP_TABLE_IMPACT_FORCE_THRESHOLD_N = 0.10
# Table-contact event fields remain in the metrics ABI for replay diagnostics.
# Runtime drop truth also requires physical return/descent evidence so a local
# table touch does not by itself terminate a recoverable handheld attempt.
DROP_TABLE_CONTACT_FORCE_THRESHOLD_N = CONTACT_FORCE_THRESHOLD_N
DROP_TABLE_CONTACT_CONSECUTIVE_SUBSTEPS = DROP_CONSECUTIVE_SUBSTEPS
SLIDER_CONTACT_DROPOUT_GRACE_SUBSTEPS = 2
RIGHT_GRIP_REQUIRED_CONTACT_OWNERS = 2
HANDHELD_WORKSPACE_X_RANGE_M = (-0.65, 0.30)
HANDHELD_WORKSPACE_Y_RANGE_M = (-0.60, 0.60)
HANDHELD_WORKSPACE_Z_RANGE_M = (_TABLE_TOP_Z - 0.05, _TABLE_TOP_Z + 0.55)
BLADE_JOINT_DRIVE_STIFFNESS = 80.0
# High viscous damping suppresses reorientation-induced slider motion.  Joint
# friction supplies the static holding force that damping alone cannot provide
# at zero velocity, so a vertically held knife does not creep under gravity.
BLADE_JOINT_DRIVE_DAMPING = 25.0
BLADE_JOINT_DRIVE_MAX_FORCE = 12.0
BLADE_JOINT_FRICTION = 3.0
BLADE_JOINT_DYNAMIC_FRICTION = 2.4
# Rewriting an unchanged articulation target wakes the AS013 knife during the
# recorder's otherwise write-free stable-origin gate.  During action, let the
# measured position move this far before refreshing the passive target.  With
# the authored 80 N/m stiffness, the maximum deadband restoring force is 8 mN.
BLADE_DRIVE_TARGET_WRITE_DEADBAND_M = 1.0e-4
# Wuji's downward-facing palm stays about 0.11 m above the fingertip work
# plane.  Targeting the palm itself only 20--25 mm above the knife asks IK to
# drive the palm through the hand, object, and table; the live controller then
# stalls far from the requested XY point and the fingers can sweep the blade
# before the housing is stabilized.
LEFT_PALM_WORK_HEIGHT_M = 0.065
RIGHT_PALM_WORK_HEIGHT_M = 0.080
RIGHT_FINGER_EXTENSION_FRACTION = 0.67
RIGHT_FINGER_ABDUCTION_RAD = -0.20
RIGHT_FINGER_NUMBER = 1
# Curl every non-control finger into the palm so its open fingertip cannot
# become the first link to hit the blade during the selected-finger descent.
RIGHT_NONCONTROL_GRIP_FRACTION = 1.0
RIGHT_VERTICAL_GAIN = 2.5
RIGHT_LATERAL_GAIN = 2.0
RIGHT_PALM_YAW_DEG = 25.0
RIGHT_PALM_ROLL_DEG = 15.0
LEFT_HOLD_HAND_PRELOAD_FRACTION = 0.0
LEFT_HOLD_HAND_PRELOAD_CONTROL_FRAMES = 30
LEFT_HOLD_CONTACT_PRELOAD_M = 0.001
LEFT_SLIDE_HOLD_CONTACT_PRELOAD_M = 0.002
LEFT_RECOVERY_CONTACT_PRELOAD_M = 0.0005
LEFT_FIXED_ANCHOR_RECOVERY_STEP_M = 0.001
# Keep the fallback anchor corrective rather than letting its integral offset
# become the dominant contact load.  A 10 mm cap winds up against the rigid
# housing after right-hand contact and translates it faster than the task's
# stability limit; the measured 4 mm trial did not restore physical contact.
LEFT_FIXED_ANCHOR_MAX_OFFSET_M = 0.003
LEFT_RECOVERY_ORIENTATION_MIN_FORCE_N = 1.0
LEFT_RECOVERY_READY_CONTROL_FRAMES = 15
RIGHT_FINGER_REENTRY_CONTROL_FRAMES = 30
LEFT_APPROACH_GRIP_FRACTION = 0.55
LEFT_FALLBACK_APPROACH_GRIP_FRACTION = 0.45
LEFT_FALLBACK_GRIP_ACTIVATION_FRAME = 120
LEFT_FALLBACK_GRIP_DISTANCE_M = 0.017
# The reachable approach is referenced to finger2, but live contact rows show
# that finger5 simultaneously owns a housing contact at that pose.  Anchor the
# hold to finger5 and curl only finger2 before the blade starts to move.
LEFT_SAFE_STABILIZER_FINGER_NUMBER = 5
LEFT_SAFE_FINGER_PRELOAD_FRACTION = 0.20
LEFT_UNSAFE_BLADE_FINGER_NUMBER = 2
LEFT_UNSAFE_FINGER_RETRACTION_CONTROL_FRAMES = 45
LEFT_UNSAFE_FINGER_RETRACTION_START_FRAME = 60
LEFT_UNSAFE_FINGER_RETRACTION_FRACTION = 1.0
LEFT_UNSAFE_FINGER_DISTAL_ONLY = True
RIGHT_SLIDER_TOUCH_GRIP_FRACTION = 0.0
# Defaults for the replacement housing are deliberately small and are
# overwritten from the checked-in geometry metadata below.  Keep the former
# contact-trace values as named legacy constants for reproducibility only.
LEGACY_LEFT_HOUSING_SURFACE_TARGET_LOCAL_M = np.asarray(
    [-0.050, 0.032, 0.039],
    dtype=np.float64,
)
LEFT_HOUSING_SURFACE_TARGET_LOCAL_M = np.asarray(
    [-0.018, 0.0, 0.001],
    dtype=np.float64,
)
LEFT_HOUSING_CONTACT_FINGER_NUMBER = 2
LEGACY_LEFT_HOUSING_FALLBACK_FEATURE_TARGET_LOCAL_M = np.asarray(
    [-0.0364, 0.1073, 0.0308],
    dtype=np.float64,
)
LEFT_HOUSING_FALLBACK_FEATURE_TARGET_LOCAL_M = np.asarray(
    [-0.018, 0.0, 0.001],
    dtype=np.float64,
)
LEFT_HOUSING_SURFACE_TARGET_LOCAL_X_RANGE_M = (-0.023, -0.010)
LEFT_HOUSING_SURFACE_PROJECTION_START_FRAME = 180
LEFT_HOUSING_FALLBACK_PRELOAD_START_FRAME = 240
LEFT_HOUSING_FALLBACK_READY_DISTANCE_M = 0.004
LEFT_HOUSING_FALLBACK_INWARD_PRELOAD_M = 0.001
LEFT_HOUSING_FALLBACK_PRELOAD_INTERVAL_FRAMES = 60
LEFT_HOUSING_FALLBACK_MAX_PRELOAD_STEPS = 20
LEGACY_LEFT_HOUSING_CONTACT_TRACE_TARGET_LOCAL_YZ_M = (0.1073, 0.0308)
LEFT_HOUSING_CONTACT_TRACE_TARGET_LOCAL_YZ_M = (0.0, 0.001)
LEFT_HOUSING_FALLBACK_INSIDE_TARGET_LOCAL_M = np.asarray(
    [-0.012, 0.0, 0.001],
    dtype=np.float64,
)
RIGHT_FINGERTIP_TARGET_CLEARANCE_M = 0.008
RIGHT_FINGERTIP_TARGET_INSET_M = 0.002
# The replacement target is the midpoint of the authored button region in the
# E_link_blade_2 frame.  Keep the historical point named for audit/replay
# comparisons; the metadata loader below selects the fresh value.
LEGACY_RIGHT_FINGERTIP_TARGET_LOCAL_XY_M = np.asarray(
    [-0.00575, -0.02114], dtype=np.float64
)
RIGHT_FINGERTIP_TARGET_LOCAL_XY_M = np.asarray(
    [0.0, 0.0], dtype=np.float64
)
# Historical physical-right palm orientation from the pre-v2 diagnostic.  It
# is not a calibration for the new physical-left slider hand and is used only
# when the legacy yaw/roll or reference-blend overrides are explicitly set.
RIGHT_CERTIFIED_ALIGNMENT_QUAT_WXYZ = np.asarray(
    [-0.03130, 0.93598, 0.14714, 0.31829], dtype=np.float64
)
RIGHT_ALIGNMENT_REFERENCE_BLEND_DEFAULT = 0.60
# The certified reference above belongs to the historical right-hand slider
# policy.  The v2 left-hand slider starts from its live reachable orientation;
# operators can still opt into that uncalibrated cross-hand initial value with
# the existing environment override for comparison.
SLIDER_ALIGNMENT_REFERENCE_BLEND_DEFAULT = 0.0
# Match the proven recontact clearance so planar alignment cannot sweep the
# fingertip through the housing before it is above the authored button XY.
RIGHT_PLANAR_ENTRY_HEIGHT_M = 0.040
RIGHT_VERTICAL_ENTRY_TOL_M = 0.001
RIGHT_PLANAR_ALIGNMENT_TOL_M = 0.002
RIGHT_RECONTACT_HEIGHT_M = 0.040
RIGHT_RECONTACT_CLEAR_HEIGHT_M = 0.028
RIGHT_RECONTACT_WAIT_HEIGHT_M = 0.030
RIGHT_RECONTACT_CONTACT_PRELOAD_M = 0.003
# Cover the measured 0.18 mm residual fingertip tracking offset after a
# late-slide dropout while remaining below the 0.5 mm bounded preload used by
# the task's late-contact configuration.
RIGHT_RECONTACT_NORMAL_PRELOAD_M = 0.0004
RIGHT_RECOVERED_CONTACT_NORMAL_PRELOAD_M = 0.0002
RIGHT_RECOVERED_CAPTURE_IN_PLANE_PRELOAD_M = 0.0002
RIGHT_RECOVERED_CAPTURE_CROSS_EDGE_PRELOAD_M = 0.0
RIGHT_RECOVERED_CONTACT_IN_PLANE_PRELOAD_M = 0.001
RIGHT_RECOVERED_LATE_IN_PLANE_PRELOAD_CONFIG_MAX_M = 0.003
RIGHT_RECOVERED_LATE_NORMAL_PRELOAD_CONFIG_MAX_M = 0.0005
RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_CONFIG_ABS_MAX_RAD = 0.05
RIGHT_RECOVERED_FINGER_EXTENSION_DELTA = 0.12
RIGHT_RECOVERED_FINGER_EXTENSION_CONFIG_MAX = 0.20
RIGHT_RECOVERED_FINGER_EXTENSION_DELTA_DEFAULT = 0.16
RIGHT_RECOVERED_LATE_FINGER_EXTENSION_DELTA_DEFAULT = 0.024
RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS = 0.70
RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS_DEFAULT = 0.75
RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE_DEFAULT = 2.23
RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_DELTA_RAD_DEFAULT = -0.0125
RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS_RANGE = (0.65, 0.79)
RIGHT_RECOVERED_LATE_FINGER_EXTENSION_CONFIG_ABS_MAX = 0.05
RIGHT_RECOVERED_LATE_PROXIMAL_EXTENSION_SCALE_RANGE = (0.0, 2.0)
RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE_RANGE = (0.0, 4.0)
RIGHT_RECOVERED_LATE_DISTAL_TIP_EXTENSION_SCALE_RANGE = (0.0, 2.0)
RIGHT_RECOVERED_HAND_BLEND_CONTROL_FRAMES = 10
RIGHT_LATE_CONTINUOUS_HAND_BLEND_CONTROL_FRAMES = 30
RIGHT_RECONTACT_CONTACT_GRACE_CONTROL_FRAMES = 30
LEFT_STABLE_CONTACT_CONTROL_GRACE_FRAMES = 15
LATE_LEFT_STABILITY_GRACE_START_PROGRESS_RANGE = (0.75, SUCCESS_PROGRESS)
LATE_LEFT_STABILITY_GRACE_START_PROGRESS_DEFAULT = 0.75
RIGHT_RECONTACT_ADVECTION_FRACTION = 1.0
RIGHT_RECOVERED_TARGET_BLEND_CONTROL_FRAMES = 10
RIGHT_RECOVERED_TARGET_MAX_BLEND = 1.0
RIGHT_CONTACT_PRELOAD_SERVO_STEP_M = 0.006
RIGHT_RECONTACT_LIFT_SERVO_STEP_M = 0.005
RIGHT_SLIDE_PALM_TRAVEL_M = 0.055
RIGHT_SLIDE_PALM_TRAVEL_DEFAULT_M = 0.107
RIGHT_SLIDE_PALM_TRAVEL_CONFIG_MAX_M = 0.120
RIGHT_SLIDE_NORMAL_PRELOAD_M = 0.0012
RIGHT_SLIDE_LATE_NORMAL_PRELOAD_M = 0.00056
RIGHT_SLIDE_MAX_LEAD_M = 0.012
RIGHT_SLIDE_MAX_LEAD_DEFAULT_M = 0.016
RIGHT_SLIDE_CROSS_EDGE_MAX_CORRECTION_M = 0.006
RIGHT_SLIDE_CROSS_EDGE_CONFIG_MAX_M = 0.012
RIGHT_SLIDE_CROSS_EDGE_BLEND_CONTROL_FRAMES = 30
RIGHT_IK_DLS_LAMBDA_DEFAULT = 0.02
RIGHT_SLIDE_ROLL_RELIEF_DEFAULT_DEG = 10.0
RIGHT_SLIDE_ROLL_RELIEF_START_PROGRESS = 0.55
RIGHT_SLIDE_ROLL_RELIEF_END_PROGRESS = 0.70
# A small negative increment keeps the control fingertip just inside the
# authored button width without sacrificing the tangential reach needed near
# the right-arm workspace limit.  The full diagnostic increment (-0.18)
# cleared the cross-edge miss but shortened the physical slide.
RIGHT_SLIDE_FINGER_EXTENSION_DELTA_DEFAULT = -0.03
RIGHT_SLIDE_FINGER_ABDUCTION_DELTA_RAD_DEFAULT = 0.01
RIGHT_SLIDE_CONTROL_FRAMES = 360
RIGHT_SLIDE_CONTROL_FRAMES_DEFAULT = 630
RIGHT_SLIDE_CONTROL_FRAMES_CONFIG_MAX = 720
RIGHT_SLIDE_REANCHOR_EXTRA_LEAD_RAMP_CONTROL_FRAMES = 60
LEFT_COUNTER_SLIDE_CONTROL_FRAMES = 120
RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES = 1
LEFT_CARTESIAN_SERVO_STEP_M = 0.005
LEFT_HOLD_RECOVERY_STEP_M = 0.005
RIGHT_CARTESIAN_SERVO_STEP_M = 0.015
RIGHT_SURFACE_APPROACH_SERVO_STEP_M = 0.001
RIGHT_SURFACE_SLOW_RADIUS_M = 0.010
STABLE_LINEAR_SPEED_MPS = 0.025
STABLE_ANGULAR_SPEED_RADPS = 0.35
# Once the final hold has been physically valid for at least one control
# frame, tolerate this many consecutive transient speed/contact dropouts.  A
# fourth bad frame resets the shared success counter. Grace frames preserve but
# do not advance that counter. Completion prerequisites and hard failures remain
# fail-closed and never consume this grace.
FINAL_HOLD_DROPOUT_GRACE_CONTROL_FRAMES = 3
# One complete 30 Hz control interval at the task's 120 Hz physics rate.
LEFT_STABILIZATION_CONSECUTIVE_SUBSTEPS = 4
RIGHT_SLIDER_CONTACT_CONSECUTIVE_SUBSTEPS = (
    LEFT_STABILIZATION_CONSECUTIVE_SUBSTEPS
)
RIGHT_APPROACH_TIMEOUT_FRAME = 1600
RIGHT_APPROACH_TIMEOUT_FRAME_DEFAULT = 1765
RIGHT_RELEASE_CONTROL_FRAMES = 60
RIGHT_RELEASE_CONTROL_FRAMES_MIN = 15
RIGHT_RELEASE_CONTROL_FRAMES_DEFAULT = 15
RIGHT_RELEASE_STABLE_RESERVE_CONTROL_FRAMES = 20
RIGHT_INITIAL_CONTACT_DROPOUT_GRACE_CONTROL_FRAMES_DEFAULT = 15
RIGHT_POST_SUCCESS_HOLD_CONTROL_FRAMES = 5
RIGHT_POST_SUCCESS_HOLD_CONTROL_FRAMES_MAX = 5
RIGHT_RELEASE_DWELL_CONTROL_FRAMES_DEFAULT = 3
RIGHT_RELEASE_DWELL_CONTROL_FRAMES_MAX = 3
RIGHT_POST_SUCCESS_OVERTRAVEL_DEFAULT_M = 0.0015
RIGHT_POST_SUCCESS_OVERTRAVEL_MAX_M = 0.0015
SLIDE_PROGRESS_DELTA = 0.08
SLIDE_PROGRESS_CONSECUTIVE_SUBSTEPS = (
    RIGHT_SLIDER_CONTACT_CONSECUTIVE_SUBSTEPS
)
# These defaults are replaced by the selected asset's checked-in geometry
# metadata below.  Keeping finite defaults lets lightweight/static imports
# work before an asset checkout is populated; the spawn path still verifies
# the fresh provenance and records the values it used.
SLIDER_BUTTON_LOCAL_MIN_M = np.asarray(
    [-0.0075, -0.0197, -0.0006], dtype=np.float64
)
SLIDER_BUTTON_LOCAL_MAX_M = np.asarray(
    [0.0075, 0.0197, 0.0006], dtype=np.float64
)
# Contact points are reported on colliding surfaces rather than at the
# fingertip/slider centerline.  Keep the exact five-owner filter, but allow a
# small geometric margin at the authored button boundary.
SLIDER_CONTACT_REGION_TOLERANCE_M = 0.0025
# The selected asset exposes one source-derived convex housing collider.  That
# collider contains the two thin side faces used by the right-hand grasp, so
# its physics material is the narrowest reliable place to raise grip traction
# without changing the hand, slider, joint resistance, or success semantics.
HOUSING_CONTACT_STATIC_FRICTION_DEFAULT = 2.5
HOUSING_CONTACT_DYNAMIC_FRICTION_DEFAULT = 2.3
HOUSING_CONTACT_STATIC_FRICTION_RANGE = (2.30, 2.70)
BLADE_CONTACT_FRICTION_DEFAULT = 2.0
BLADE_CONTACT_FRICTION_RANGE = (0.40, 3.00)
RESET_LINEAR_DAMPING = 0.40
RESET_ANGULAR_DAMPING = 2.00
RESET_SLEEP_THRESHOLD = 0.020
RESET_SOLVER_POSITION_ITERATIONS = 16
RESET_SOLVER_VELOCITY_ITERATIONS = 4
HOUSING_COLLISION_APPROXIMATION = "convexHull"
HOUSING_CONTACT_OFFSET_M = 0.001
HOUSING_REST_OFFSET_M = 0.0
MAX_HOUSING_XY_DISPLACEMENT_M = 0.06
KNIFE_ROOT_PATH = f"{_TASK_ROOT}/AS013UtilityKnife"
LEGACY_ASSET_PATH = (
    Path(__file__).resolve().parent
    / "usd"
    / TASK_NAME
    / "utility_knife002"
    / "model_utility_knife_3.usd"
)
ASSET_PATH = (
    Path(__file__).resolve().parent
    / "usd"
    / TASK_NAME
    / ASSET_VARIANT
    / f"{ASSET_VARIANT}.usda"
)
ASSET_METADATA_PATH = ASSET_PATH.with_name(
    f"{ASSET_VARIANT}_metadata.json"
)


def _metadata_key_token(value: object) -> str:
    """Normalize metadata keys for tolerant builder-schema lookup."""

    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _walk_asset_metadata(value: object):
    """Yield ``(key, value)`` pairs recursively from a JSON payload."""

    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_asset_metadata(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_asset_metadata(child)


def _metadata_lookup(payload: object, *keys: str) -> object | None:
    wanted = {_metadata_key_token(key) for key in keys}
    for key, value in _walk_asset_metadata(payload):
        if _metadata_key_token(key) in wanted:
            return value
    return None


def _metadata_vector(
    payload: object,
    keys: tuple[str, ...],
    default: np.ndarray,
) -> np.ndarray:
    """Read a finite 3-vector from flat or nested asset metadata."""

    value = _metadata_lookup(payload, *keys)
    if isinstance(value, dict):
        # Builders commonly encode vectors as ``{x, y, z}`` or ``{value: []}`.
        if all(axis in value for axis in ("x", "y", "z")):
            value = [value["x"], value["y"], value["z"]]
        else:
            for child_key in ("value", "vector", "xyz", "meters", "m"):
                if child_key in value:
                    value = value[child_key]
                    break
    try:
        result = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return np.asarray(default, dtype=np.float64).reshape(3).copy()
    if not np.isfinite(result).all():
        return np.asarray(default, dtype=np.float64).reshape(3).copy()
    return result


def _metadata_array(
    payload: object,
    keys: tuple[str, ...],
    default: np.ndarray,
    *,
    length: int,
) -> np.ndarray:
    """Read a finite 2/3-vector while accepting ``{x,y}`` or list forms."""

    value = _metadata_lookup(payload, *keys)
    if isinstance(value, dict):
        axis_names = ("x", "y") if length == 2 else ("x", "y", "z")
        if all(axis in value for axis in axis_names):
            value = [value[axis] for axis in axis_names]
        else:
            for child_key in ("value", "vector", "xy", "xyz", "meters", "m"):
                if child_key in value:
                    value = value[child_key]
                    break
    try:
        result = np.asarray(value, dtype=np.float64).reshape(length)
    except (TypeError, ValueError):
        return np.asarray(default, dtype=np.float64).reshape(length).copy()
    if not np.isfinite(result).all():
        return np.asarray(default, dtype=np.float64).reshape(length).copy()
    return result


def _metadata_bounds(
    payload: object,
    keys: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a local AABB encoded as ``min/max`` or ``lower/upper``."""

    block = _metadata_lookup(payload, *keys)
    if not isinstance(block, dict):
        return None
    min_value = None
    max_value = None
    for name in ("min", "min_m", "minimum", "minimum_m", "lower", "lo"):
        if name in block:
            min_value = block[name]
            break
    for name in ("max", "max_m", "maximum", "maximum_m", "upper", "hi"):
        if name in block:
            max_value = block[name]
            break
    if min_value is None or max_value is None:
        return None
    try:
        lower = np.asarray(min_value, dtype=np.float64).reshape(3)
        upper = np.asarray(max_value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        return None
    if np.any(upper <= lower):
        return None
    return lower, upper


def _metadata_category_bounds(
    payload: object,
    category: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Union per-visual bounds for one semantic category."""

    if not isinstance(payload, dict):
        return None
    visuals = payload.get("visuals")
    if not isinstance(visuals, list):
        return None
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    for item in visuals:
        if (
            not isinstance(item, dict)
            or str(item.get("category", "")).lower() != category.lower()
        ):
            continue
        bounds = item.get("bounds")
        if not isinstance(bounds, dict):
            continue
        parsed = _metadata_bounds({"bounds": bounds}, ("bounds",))
        if parsed is not None:
            lowers.append(parsed[0])
            uppers.append(parsed[1])
    if not lowers:
        return None
    return np.min(np.stack(lowers), axis=0), np.max(np.stack(uppers), axis=0)


def _load_selected_asset_metadata() -> dict[str, Any]:
    """Load local geometry metadata without making static imports fragile."""

    candidates = [ASSET_METADATA_PATH]
    # The task-dev copy is intentionally a fallback only.  Runtime deployments
    # mirror metadata beside the USD; source checkout paths are not required
    # for a deployed collection worker.
    try:
        candidates.append(
            Path(__file__).resolve().parents[4]
            / "taskdev"
            / "tasks"
            / TASK_NAME
            / "assets"
            / ASSET_VARIANT
            / "runtime"
            / ASSET_METADATA_PATH.name
        )
    except IndexError:
        pass
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


_ASSET_METADATA = _load_selected_asset_metadata()
_ASSET_METADATA_LOADED = bool(_ASSET_METADATA)

# Geometry values are sourced from the generated asset's semantic-region
# record.  The aliases tolerate flat and nested metadata while retaining the
# same canonical body-1 frame used by contact sampling below.
SLIDER_BUTTON_LOCAL_MIN_M = _metadata_vector(
    _ASSET_METADATA,
    (
        "slider_button_local_min_m",
        "button_local_min_m",
        "button_bounds_local_min_m",
        "semantic_button_local_min_m",
        "min_local_m",
    ),
    SLIDER_BUTTON_LOCAL_MIN_M,
)
SLIDER_BUTTON_LOCAL_MAX_M = _metadata_vector(
    _ASSET_METADATA,
    (
        "slider_button_local_max_m",
        "button_local_max_m",
        "button_bounds_local_max_m",
        "semantic_button_local_max_m",
        "max_local_m",
    ),
    SLIDER_BUTTON_LOCAL_MAX_M,
)
# The builder records the semantic region as
# ``button_region.local_bounds_m.{min_m,max_m}``; consume that nested record
# explicitly before falling back to the finite import defaults above.
_button_bounds = _metadata_bounds(
    _ASSET_METADATA,
    ("button_region_local_bounds_m", "local_bounds_m"),
)
if _button_bounds is not None:
    SLIDER_BUTTON_LOCAL_MIN_M = _button_bounds[0]
    SLIDER_BUTTON_LOCAL_MAX_M = _button_bounds[1]
# A malformed/inverted metadata record must not silently turn all contacts
# into slider contacts.  Keep the finite defaults and let the spawn audit
# report the provenance/geometry error instead.
if np.any(SLIDER_BUTTON_LOCAL_MAX_M <= SLIDER_BUTTON_LOCAL_MIN_M):
    SLIDER_BUTTON_LOCAL_MIN_M = np.asarray(
        [-0.0075, -0.0197, -0.0006], dtype=np.float64
    )
    SLIDER_BUTTON_LOCAL_MAX_M = np.asarray(
        [0.0075, 0.0197, 0.0006], dtype=np.float64
    )

_button_center_default = 0.5 * (
    SLIDER_BUTTON_LOCAL_MIN_M + SLIDER_BUTTON_LOCAL_MAX_M
)
# ``button_region.center_m`` is expressed in the housing/body-0 frame.  The
# contact controller's target is intentionally blade/body-1 local, so use the
# local-bounds midpoint as its fallback rather than mixing the two frames.
_button_center_body0 = _metadata_vector(
    _ASSET_METADATA,
    ("button_region_center_m", "center_m"),
    _button_center_default,
)
BUTTON_REGION_CENTER_BODY0_M = _button_center_body0.copy()
BUTTON_REGION_CENTER_LOCAL_M = _button_center_default.copy()
RIGHT_FINGERTIP_TARGET_LOCAL_XY_M = _metadata_array(
    _ASSET_METADATA,
    (
        "right_fingertip_target_local_xy_m",
        "fingertip_target_local_xy_m",
        "button_target_local_xy_m",
        "slider_target_local_xy_m",
    ),
    _button_center_default[:2],
    length=2,
)
# Canonical v2 name.  The historical RIGHT_* constant remains importable for
# sealed diagnostics, but both values describe the physical left slider hand.
LEFT_SLIDER_FINGERTIP_TARGET_LOCAL_XY_M = (
    RIGHT_FINGERTIP_TARGET_LOCAL_XY_M.copy()
)

# The v2 physical right-hand stabilizer must remain on the replacement housing
# rather than the historical 40-mm-high target.  The LEFT_* constants below
# retain their sealed pre-v2 names. Prefer explicit semantic targets; otherwise
# derive a conservative side/upper-surface point from the authored housing
# bounds.  This keeps the target in the same body frame and avoids inventing a
# visible proxy or a nonuniform geometry scale.
_housing_bounds = _metadata_category_bounds(_ASSET_METADATA, "housing")
if _housing_bounds is None:
    _housing_bounds = _metadata_bounds(
        _ASSET_METADATA,
        (
            "housing_bounds_m",
            "housing_bounds_local_m",
            "body_bounds_local_m",
            "housing_aabb_local_m",
            "bounds_m",
        ),
    )
if _housing_bounds is not None:
    _housing_min, _housing_max = _housing_bounds
    _housing_extent = _housing_max - _housing_min
    _housing_derived_surface = np.asarray(
        [
            _housing_min[0] + 0.20 * _housing_extent[0],
            0.50 * (_housing_min[1] + _housing_max[1]),
            _housing_max[2],
        ],
        dtype=np.float64,
    )
    _housing_derived_inside = np.asarray(
        [
            _housing_min[0] + 0.35 * _housing_extent[0],
            _housing_derived_surface[1],
            _housing_derived_surface[2],
        ],
        dtype=np.float64,
    )
    _housing_derived_x_range = (
        float(_housing_min[0] + 0.10 * _housing_extent[0]),
        float(_housing_min[0] + 0.45 * _housing_extent[0]),
    )
    _housing_derived_trace = (
        float(0.50 * (_housing_min[1] + _housing_max[1])),
        float(_housing_max[2]),
    )
else:
    _housing_derived_surface = np.asarray(
        [-0.018, 0.0, 0.001], dtype=np.float64
    )
    _housing_derived_inside = np.asarray(
        [-0.012, 0.0, 0.001], dtype=np.float64
    )
    _housing_derived_x_range = (-0.023, -0.010)
    _housing_derived_trace = (0.0, 0.001)

LEFT_HOUSING_SURFACE_TARGET_LOCAL_M = _metadata_vector(
    _ASSET_METADATA,
    (
        "left_housing_surface_target_local_m",
        "housing_surface_target_local_m",
        "housing_contact_target_local_m",
    ),
    _housing_derived_surface,
)
_housing_x_range = _metadata_array(
    _ASSET_METADATA,
    (
        "left_housing_surface_target_local_x_range_m",
        "housing_surface_target_local_x_range_m",
        "housing_contact_local_x_range_m",
    ),
    np.asarray(_housing_derived_x_range, dtype=np.float64),
    length=2,
)
LEFT_HOUSING_SURFACE_TARGET_LOCAL_X_RANGE_M = (
    float(_housing_x_range[0]),
    float(_housing_x_range[1]),
)
_housing_trace = _metadata_array(
    _ASSET_METADATA,
    (
        "left_housing_contact_trace_target_local_yz_m",
        "housing_contact_trace_target_local_yz_m",
        "housing_trace_target_local_yz_m",
    ),
    np.asarray(_housing_derived_trace, dtype=np.float64),
    length=2,
)
LEFT_HOUSING_CONTACT_TRACE_TARGET_LOCAL_YZ_M = (
    float(_housing_trace[0]),
    float(_housing_trace[1]),
)
LEFT_HOUSING_FALLBACK_FEATURE_TARGET_LOCAL_M = _metadata_vector(
    _ASSET_METADATA,
    (
        "left_housing_fallback_feature_target_local_m",
        "housing_fallback_feature_target_local_m",
    ),
    LEFT_HOUSING_SURFACE_TARGET_LOCAL_M,
)
LEFT_HOUSING_FALLBACK_INSIDE_TARGET_LOCAL_M = _metadata_vector(
    _ASSET_METADATA,
    (
        "left_housing_fallback_inside_target_local_m",
        "housing_fallback_inside_target_local_m",
    ),
    _housing_derived_inside,
)

# Both arms approach from the robot-facing negative-X side; their bases are
# separated along world Y.  Swapping the physical stabilizer therefore must
# not mirror the contact to the far positive-X face.  Reuse the audited near
# surface unless future metadata provides an explicitly live-certified right
# target.
_right_housing_surface_default = LEFT_HOUSING_SURFACE_TARGET_LOCAL_M.copy()
RIGHT_HOUSING_SURFACE_TARGET_LOCAL_M = _metadata_vector(
    _ASSET_METADATA,
    (
        "right_housing_surface_target_local_m",
        "right_housing_contact_target_local_m",
    ),
    _right_housing_surface_default,
)
_right_housing_x_range_default = (
    LEFT_HOUSING_SURFACE_TARGET_LOCAL_X_RANGE_M
)
_right_housing_x_range = _metadata_array(
    _ASSET_METADATA,
    (
        "right_housing_surface_target_local_x_range_m",
        "right_housing_contact_local_x_range_m",
    ),
    np.asarray(_right_housing_x_range_default, dtype=np.float64),
    length=2,
)
RIGHT_HOUSING_SURFACE_TARGET_LOCAL_X_RANGE_M = (
    float(_right_housing_x_range[0]),
    float(_right_housing_x_range[1]),
)
RIGHT_HOUSING_CONTACT_TRACE_TARGET_LOCAL_YZ_M = tuple(
    float(value) for value in LEFT_HOUSING_CONTACT_TRACE_TARGET_LOCAL_YZ_M
)
RIGHT_HOUSING_FALLBACK_FEATURE_TARGET_LOCAL_M = _metadata_vector(
    _ASSET_METADATA,
    (
        "right_housing_fallback_feature_target_local_m",
        "right_housing_feature_target_local_m",
    ),
    LEFT_HOUSING_FALLBACK_FEATURE_TARGET_LOCAL_M.copy(),
)
RIGHT_HOUSING_FALLBACK_INSIDE_TARGET_LOCAL_M = _metadata_vector(
    _ASSET_METADATA,
    (
        "right_housing_fallback_inside_target_local_m",
        "right_housing_inside_target_local_m",
    ),
    LEFT_HOUSING_FALLBACK_INSIDE_TARGET_LOCAL_M.copy(),
)

ASSET_SOURCE_ARCHIVE_SHA256 = (
    "85fe8c49f43383aa15e5ea9803dfca52fec196d3dc00424ea8eaaca326aa648b"
)
ASSET_METADATA_SHA256 = (
    "f67a448283852e90a4349947a0e342245f04a65f3533eaa78621947218f725f6"
)
# Bind the exact reviewable USDA as well as its sidecar.  Provenance strings
# alone are insufficient: a same-named replacement with unchanged metadata
# must fail closed before it can enter a fresh PhysX scene.
ASSET_USDA_SHA256 = (
    "332f1b717677caf73ccbff6fd48debd3c862fb729960b373909692d12f68dfa0"
)
ASSET_SOURCE_URL_TOKEN = "083c2d52b3d04eedb5350cce6e2d1b92"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_provenance_text(source_stage: Any) -> str:
    """Return root custom-data provenance as a stable lowercase string."""

    try:
        root = source_stage.GetDefaultPrim()
        custom_data = root.GetCustomData()
    except Exception:
        return ""
    try:
        return json.dumps(custom_data, sort_keys=True, default=str).lower()
    except (TypeError, ValueError):
        return str(custom_data).lower()


def _is_fresh_selected_asset(source_stage: Any) -> bool:
    """Identify the downloaded replacement without weakening legacy assets."""

    path_parts = {part.lower() for part in ASSET_PATH.parts}
    if ASSET_VARIANT.lower() not in path_parts:
        return False
    provenance = _asset_provenance_text(source_stage)
    # The generated USD carries both source identity tokens.  Require those
    # tokens *and* a matching sidecar (when one is available), so an arbitrary
    # same-named USD cannot bypass the historical SDF guard merely by being
    # placed under the replacement directory.
    metadata_archive = _metadata_lookup(
        _ASSET_METADATA,
        "archive_sha256",
        "source_archive_sha256",
    )
    # A fresh replacement is not trustworthy without its sidecar geometry
    # record.  Falling back to import-time dimensions would let a deployed
    # USD silently use stale contact targets, so missing metadata fails closed.
    metadata_matches = bool(
        _ASSET_METADATA_LOADED
        and str(metadata_archive).lower() == ASSET_SOURCE_ARCHIVE_SHA256
        and ASSET_METADATA_PATH.is_file()
        and _sha256_file(ASSET_METADATA_PATH) == ASSET_METADATA_SHA256
    )
    usda_matches = bool(
        ASSET_PATH.is_file()
        and _sha256_file(ASSET_PATH) == ASSET_USDA_SHA256
    )
    provenance_matches = (
        ASSET_SOURCE_URL_TOKEN in provenance
        and ASSET_SOURCE_ARCHIVE_SHA256 in provenance
        and "ahtusik" in provenance
    )
    return bool(provenance_matches and metadata_matches and usda_matches)


def _body_relative_axis_coordinate(
    point_world: np.ndarray,
    body_pose_world: np.ndarray,
    axis_body_local: np.ndarray,
) -> float:
    """Project a live world point onto one body-local unit axis."""

    point = np.asarray(point_world, dtype=np.float64).reshape(3)
    body_pose = np.asarray(body_pose_world, dtype=np.float64).reshape(7)
    axis = np.asarray(axis_body_local, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if (
        not np.isfinite(point).all()
        or not np.isfinite(body_pose).all()
        or not np.isfinite(axis).all()
        or axis_norm <= 1.0e-12
    ):
        return math.nan
    point_body = (
        _invert_pose_matrix(_pose_vector_to_matrix(body_pose))
        @ np.append(point, 1.0)
    )[:3]
    return float(np.dot(point_body, axis / axis_norm))


def _configured_blade_contact_friction(
    configured: str | None,
    base_friction: float,
) -> float:
    """Return a finite, bounded friction for blade-side colliders."""

    if configured is None or not str(configured).strip():
        value = float(base_friction)
    else:
        try:
            value = float(configured)
        except ValueError as exc:
            raise RuntimeError(
                "AS013 blade contact friction must be numeric"
            ) from exc
    lower, upper = BLADE_CONTACT_FRICTION_RANGE
    if not np.isfinite(value) or not lower <= value <= upper:
        raise RuntimeError(
            "AS013 blade contact friction must be finite and in "
            f"[{lower}, {upper}]"
        )
    return value


def _causal_fingertip_drive_target(
    *,
    contact_joint_position: float,
    measured_joint_position: float,
    contact_fingertip_axis_coordinate: float,
    current_fingertip_axis_coordinate: float,
    contact_command_axis_coordinate: float,
    current_command_axis_coordinate: float,
    previous_forward_displacement_m: float,
    joint_limits: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Return a drive target bounded by both live finger and command motion.

    The passive component may match measured q to remove the authored spring's
    resistance.  Any target advance beyond measured q is capped by the
    greatest displacement jointly observed from the measured fingertip and
    the hand-control target.  Retaining that already reached displacement lets
    the blade remain where the hand physically brought it during release.
    """

    values = np.asarray(
        [
            contact_joint_position,
            measured_joint_position,
            contact_fingertip_axis_coordinate,
            current_fingertip_axis_coordinate,
            contact_command_axis_coordinate,
            current_command_axis_coordinate,
            previous_forward_displacement_m,
            *joint_limits,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        return math.nan, math.nan, math.nan, math.nan
    lower, upper = (float(joint_limits[0]), float(joint_limits[1]))
    if upper <= lower:
        return math.nan, math.nan, math.nan, math.nan
    fingertip_forward = max(
        0.0,
        float(
            current_fingertip_axis_coordinate
            - contact_fingertip_axis_coordinate
        ),
    )
    command_forward = max(
        0.0,
        float(current_command_axis_coordinate - contact_command_axis_coordinate),
    )
    common_forward = max(
        max(0.0, float(previous_forward_displacement_m)),
        min(fingertip_forward, command_forward),
    )
    causal_target = float(contact_joint_position) + common_forward
    # Matching the measured joint position is the original passive latch: it
    # cannot advance the blade.  The causal branch may exceed that position
    # only by displacement already jointly observed from finger and command.
    target = float(
        np.clip(max(float(measured_joint_position), causal_target), lower, upper)
    )
    return target, common_forward, fingertip_forward, command_forward


def _bounded_right_servo_correction(
    fingertip_error_world: np.ndarray,
    *,
    lateral_gain: float,
    vertical_gain: float,
    max_step_m: float = RIGHT_CARTESIAN_SERVO_STEP_M,
) -> np.ndarray:
    """Return one small closed-loop fingertip correction in world space."""

    correction = np.asarray(
        fingertip_error_world,
        dtype=np.float64,
    ).reshape(3).copy()
    correction[1] *= float(lateral_gain)
    correction[2] *= float(vertical_gain)
    correction_norm = float(np.linalg.norm(correction))
    if correction_norm > max_step_m:
        correction *= max_step_m / correction_norm
    return correction


def _contact_filter_counts_by_path(
    counts: np.ndarray,
    sensor_count: int,
    filter_paths: list[str] | tuple[str, ...],
) -> dict[str, int]:
    """Map rigid-contact tensor filter columns back to their body paths."""

    paths = tuple(str(path) for path in filter_paths)
    sensors = int(sensor_count)
    values = np.asarray(counts, dtype=np.uint32).reshape(-1)
    if sensors < 1 or not paths or values.size != sensors * len(paths):
        return {}
    matrix = values.reshape(sensors, len(paths))
    totals = np.sum(matrix, axis=0, dtype=np.uint64)
    return {path: int(total) for path, total in zip(paths, totals)}


def _wuji_hand_body_role(body_name: str, side: str) -> str | None:
    """Classify one actual Wuji rigid-body name for AS013 contacts.

    Wuji finger1 is the thumb.  Its distal mesh and named tip collider are
    owned by ``<side>_finger1_link4``; there is no independent tip rigid body
    in the Isaac articulation.  ``*_tip_link`` remains accepted for older
    exports, but the current runtime requires and filters link4.
    """

    normalized_side = str(side).strip().lower()
    normalized = str(body_name).strip().lower()
    if normalized == f"{normalized_side}_palm_link":
        return "palm"
    if normalized.startswith(f"{normalized_side}_finger1_"):
        if normalized in {
            f"{normalized_side}_finger1_link4",
            f"{normalized_side}_finger1_tip_link",
        }:
            return "thumb_distal_tip"
        return "thumb_proximal"
    for finger_number in range(2, 6):
        if normalized.startswith(
            f"{normalized_side}_finger{finger_number}_"
        ):
            return "non_thumb"
    return None


def _expected_wuji_hand_body_names(
    side: str,
    roles: set[str] | frozenset[str],
) -> frozenset[str]:
    """Return the exact current Wuji1 rigid-owner set for AS013 roles."""

    normalized_side = str(side).strip().lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError(f"unsupported Wuji hand side: {side!r}")
    selected_roles = frozenset(str(role) for role in roles)
    supported_roles = frozenset(
        {"palm", "thumb_proximal", "thumb_distal_tip", "non_thumb"}
    )
    unknown_roles = selected_roles - supported_roles
    if unknown_roles:
        raise ValueError(f"unsupported Wuji hand roles: {sorted(unknown_roles)}")
    names: set[str] = set()
    if "palm" in selected_roles:
        names.add(f"{normalized_side}_palm_link")
    if "thumb_proximal" in selected_roles:
        names.update(
            f"{normalized_side}_finger1_link{link_number}"
            for link_number in range(1, 4)
        )
    if "thumb_distal_tip" in selected_roles:
        names.add(f"{normalized_side}_finger1_link4")
    if "non_thumb" in selected_roles:
        names.update(
            f"{normalized_side}_finger{finger_number}_link{link_number}"
            for finger_number in range(2, 6)
            for link_number in range(1, 5)
        )
    return frozenset(names)


def _left_stabilizer_finger_from_contact_counts(
    contact_counts: dict[str, int],
    fallback_finger_number: int = LEFT_SAFE_STABILIZER_FINGER_NUMBER,
    *,
    hand_side: str = STABILIZER_HAND_SIDE,
    excluded_finger_number: int | None = None,
) -> int:
    """Choose the live stabilizer finger from physical-side contact rows.

    The helper name is retained for sealed pre-v2 diagnostics.  Role schema
    v2 defaults to the physical right stabilizer; callers replaying the legacy
    left-hand role can pass ``hand_side="left"`` and its historical exclusion
    explicitly.
    """

    normalized_side = str(hand_side).strip().lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError(f"unsupported stabilizer hand side: {hand_side!r}")
    totals: dict[int, int] = {}
    for path, raw_count in contact_counts.items():
        name = str(path).rsplit("/", 1)[-1].lower()
        prefix = f"{normalized_side}_finger"
        if not name.startswith(prefix):
            continue
        number_text = name[len(prefix) :].split("_", 1)[0]
        if not number_text.isdigit():
            continue
        finger_number = int(number_text)
        count = max(0, int(raw_count))
        if (
            not 1 <= finger_number <= 5
            or finger_number == excluded_finger_number
            or count <= 0
        ):
            continue
        totals[finger_number] = totals.get(finger_number, 0) + count
    if not totals:
        return int(fallback_finger_number)
    return max(
        totals,
        key=lambda finger_number: (
            totals[finger_number],
            finger_number == int(fallback_finger_number),
        ),
    )


def _left_hold_preload_correction(
    fingertip_error_world: np.ndarray,
    contact_standoff_m: float,
    preload_m: float = LEFT_HOLD_CONTACT_PRELOAD_M,
) -> np.ndarray:
    """Maintain a small bounded closed-loop housing contact preload."""

    error = np.asarray(fingertip_error_world, dtype=np.float64).reshape(3)
    distance = float(np.linalg.norm(error))
    travel = min(
        max(0.0, float(preload_m)),
        max(0.0, distance - float(contact_standoff_m)),
    )
    if distance <= 1.0e-12 or travel <= 0.0:
        return np.zeros(3, dtype=np.float64)
    return error * (travel / distance)


def _left_hold_effective_standoff(
    contact_standoff_m: float,
    preload_m: float,
) -> float:
    """Shorten the frozen contact standoff when slide support is active."""

    additional_preload = max(
        0.0,
        float(preload_m) - LEFT_HOLD_CONTACT_PRELOAD_M,
    )
    return max(0.0, float(contact_standoff_m) - additional_preload)


def _select_left_hold_fingertip(
    fingertips: list[np.ndarray],
    surface_target_world: np.ndarray,
    locked_index: int | None,
) -> tuple[np.ndarray | None, int | None]:
    """Keep using the proven housing-contact fingertip once it is locked."""

    if not fingertips:
        return None, None
    if locked_index is not None and 0 <= int(locked_index) < len(fingertips):
        index = int(locked_index)
    else:
        target = np.asarray(surface_target_world, dtype=np.float64).reshape(3)
        index = int(
            np.argmin(
                [
                    float(np.linalg.norm(np.asarray(point) - target))
                    for point in fingertips
                ]
            )
        )
    return np.asarray(fingertips[index], dtype=np.float64).reshape(3), index


def _left_recovery_preload_target(
    contact_reference_world: np.ndarray,
    housing_surface_target_world: np.ndarray,
) -> np.ndarray:
    """Move the proven contact feature into the housing surface."""

    reference = np.asarray(
        contact_reference_world,
        dtype=np.float64,
    ).reshape(3)
    surface = np.asarray(
        housing_surface_target_world,
        dtype=np.float64,
    ).reshape(3)
    inward = surface - reference
    distance = float(np.linalg.norm(inward))
    if distance <= 1.0e-12:
        return reference.copy()
    return reference + inward * (LEFT_RECOVERY_CONTACT_PRELOAD_M / distance)


def _left_fixed_anchor_recovery_correction(
    target_world: np.ndarray,
    current_world: np.ndarray,
    max_step_m: float = LEFT_FIXED_ANCHOR_RECOVERY_STEP_M,
) -> np.ndarray:
    """Return a bounded correction toward a fixed world-space contact point."""

    target = np.asarray(target_world, dtype=np.float64).reshape(3)
    current = np.asarray(current_world, dtype=np.float64).reshape(3)
    correction = target - current
    norm = float(np.linalg.norm(correction))
    step = max(0.0, float(max_step_m))
    if norm > step and norm > 1.0e-12:
        correction *= step / norm
    return correction


def _advance_left_fixed_anchor_offset(
    current_offset_world: np.ndarray,
    target_world: np.ndarray,
    current_world: np.ndarray,
    max_step_m: float = LEFT_FIXED_ANCHOR_RECOVERY_STEP_M,
    max_offset_m: float = LEFT_FIXED_ANCHOR_MAX_OFFSET_M,
) -> np.ndarray:
    """Integrate anchor error around the latched palm pose with a hard cap."""

    offset = np.asarray(current_offset_world, dtype=np.float64).reshape(3).copy()
    offset += _left_fixed_anchor_recovery_correction(
        target_world,
        current_world,
        max_step_m=max_step_m,
    )
    norm = float(np.linalg.norm(offset))
    limit = max(0.0, float(max_offset_m))
    if norm > limit and norm > 1.0e-12:
        offset *= limit / norm
    return offset


def _left_fixed_anchor_slide_preload_target(
    anchor_world: np.ndarray,
    housing_surface_world: np.ndarray,
) -> np.ndarray:
    """Increase the fixed anchor once to the established slide preload."""

    anchor = np.asarray(anchor_world, dtype=np.float64).reshape(3)
    surface = np.asarray(housing_surface_world, dtype=np.float64).reshape(3)
    inward = surface - anchor
    distance = float(np.linalg.norm(inward))
    extra_preload_m = max(
        0.0,
        LEFT_SLIDE_HOLD_CONTACT_PRELOAD_M - LEFT_RECOVERY_CONTACT_PRELOAD_M,
    )
    if distance <= 1.0e-12:
        return anchor.copy()
    return anchor + inward * (extra_preload_m / distance)


def _staged_right_fingertip_target_local(
    current_local: np.ndarray,
    final_target_local: np.ndarray,
    previous_stage: str = "vertical_entry",
) -> tuple[np.ndarray, str]:
    """Stage vertical, planar, then contact without regressing a stage."""

    current = np.asarray(current_local, dtype=np.float64).reshape(3)
    final_target = np.asarray(final_target_local, dtype=np.float64).reshape(3)
    staged = final_target.copy()
    if current[2] > (
        RIGHT_PLANAR_ENTRY_HEIGHT_M + RIGHT_VERTICAL_ENTRY_TOL_M
    ):
        staged[:2] = current[:2]
        staged[2] = RIGHT_PLANAR_ENTRY_HEIGHT_M
        candidate_stage = "vertical_entry"
    elif float(np.linalg.norm(current[:2] - final_target[:2])) > (
        RIGHT_PLANAR_ALIGNMENT_TOL_M
    ):
        staged[2] = RIGHT_PLANAR_ENTRY_HEIGHT_M
        candidate_stage = "planar_alignment"
    else:
        candidate_stage = "contact_descent"

    stage_order = (
        "vertical_entry",
        "planar_alignment",
        "contact_descent",
    )
    stage = stage_order[
        max(stage_order.index(previous_stage), stage_order.index(candidate_stage))
    ]
    if stage == "planar_alignment":
        staged = final_target.copy()
        staged[2] = RIGHT_PLANAR_ENTRY_HEIGHT_M
    elif stage == "contact_descent":
        staged = final_target.copy()
    return staged, stage


def _fixed_right_fingertip_target_local(
    slider_center_local: np.ndarray,
) -> np.ndarray:
    """Return one root-randomization-invariant point on the slider button."""

    target = np.asarray(slider_center_local, dtype=np.float64).reshape(3).copy()
    target_lower = SLIDER_BUTTON_LOCAL_MIN_M.copy()
    target_upper = SLIDER_BUTTON_LOCAL_MAX_M.copy()
    target_lower[:2] += RIGHT_FINGERTIP_TARGET_INSET_M
    target_upper[:2] -= RIGHT_FINGERTIP_TARGET_INSET_M
    target[:2] = np.clip(
        RIGHT_FINGERTIP_TARGET_LOCAL_XY_M,
        target_lower[:2],
        target_upper[:2],
    )
    target[2] += RIGHT_FINGERTIP_TARGET_CLEARANCE_M
    return target


def _adapt_fallback_right_target_cross_edge(
    current_fingertip_local: np.ndarray,
    fixed_target_local: np.ndarray,
) -> np.ndarray:
    """Choose the nearest authored button cross-section once reach is known."""

    current = np.asarray(
        current_fingertip_local,
        dtype=np.float64,
    ).reshape(3)
    target = np.asarray(
        fixed_target_local,
        dtype=np.float64,
    ).reshape(3).copy()
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        return target
    lower = float(
        SLIDER_BUTTON_LOCAL_MIN_M[0] + RIGHT_FINGERTIP_TARGET_INSET_M
    )
    upper = float(
        SLIDER_BUTTON_LOCAL_MAX_M[0] - RIGHT_FINGERTIP_TARGET_INSET_M
    )
    target[0] = float(np.clip(current[0], lower, upper))
    return target


def _yaw_rotated_world_rotation(
    reference_rotation: np.ndarray,
    reset_yaw_degrees: float,
) -> np.ndarray:
    """Apply the randomized object yaw to a nominal world orientation."""

    reference = np.asarray(reference_rotation, dtype=np.float64).reshape(3, 3)
    yaw = math.radians(float(reset_yaw_degrees))
    yaw_rotation = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return yaw_rotation @ reference


def _certified_right_alignment_world_rotation(
    reset_yaw_degrees: float,
) -> np.ndarray:
    """Rotate the certified nominal palm orientation with randomized yaw."""

    reference_pose = np.concatenate(
        [
            np.zeros(3, dtype=np.float64),
            RIGHT_CERTIFIED_ALIGNMENT_QUAT_WXYZ,
        ]
    )
    reference_rotation = _pose_vector_to_matrix(reference_pose)[:3, :3]
    return _yaw_rotated_world_rotation(
        reference_rotation,
        reset_yaw_degrees,
    )


def _blended_right_alignment_world_rotation(
    actual_rotation: np.ndarray,
    reset_yaw_degrees: float,
    reference_blend: float,
) -> np.ndarray:
    """Regularize a reachable palm rotation toward the certified reference."""

    actual = np.asarray(actual_rotation, dtype=np.float64).reshape(3, 3)
    blend = float(np.clip(float(reference_blend), 0.0, 1.0))
    if blend <= 0.0:
        return actual.copy()
    reference = _certified_right_alignment_world_rotation(
        reset_yaw_degrees
    )
    actual_quat = _matrix_to_quat_wxyz(actual)
    reference_quat = _matrix_to_quat_wxyz(reference)
    if float(np.dot(actual_quat, reference_quat)) < 0.0:
        reference_quat = -reference_quat
    blended_quat = (
        (1.0 - blend) * actual_quat + blend * reference_quat
    )
    norm = float(np.linalg.norm(blended_quat))
    if not np.isfinite(blended_quat).all() or norm <= 1.0e-12:
        return actual.copy()
    blended_pose = np.concatenate(
        [np.zeros(3, dtype=np.float64), blended_quat / norm]
    )
    return _pose_vector_to_matrix(blended_pose)[:3, :3]


def _right_servo_max_step(
    stage: str,
    current_local: np.ndarray,
    final_target_local: np.ndarray,
    surface_slow_radius_m: float = RIGHT_SURFACE_SLOW_RADIUS_M,
) -> float:
    """Approach quickly in free space, then enter the contact region gently."""

    if stage == "vertical_entry":
        return RIGHT_CARTESIAN_SERVO_STEP_M
    if stage == "recontact_lift":
        return RIGHT_RECONTACT_LIFT_SERVO_STEP_M
    if stage == "recontact_wait":
        return RIGHT_SURFACE_APPROACH_SERVO_STEP_M
    if stage == "contact_descent":
        return RIGHT_SURFACE_APPROACH_SERVO_STEP_M
    planar_distance = float(
        np.linalg.norm(
            np.asarray(current_local, dtype=np.float64).reshape(3)[:2]
            - np.asarray(final_target_local, dtype=np.float64).reshape(3)[:2]
        )
    )
    return min(
        RIGHT_CARTESIAN_SERVO_STEP_M,
        max(
            RIGHT_SURFACE_APPROACH_SERVO_STEP_M,
            planar_distance - max(0.0, float(surface_slow_radius_m)),
        ),
    )


def _right_servo_state_max_step(
    *,
    recontact_active: bool,
    preload_settling_active: bool,
    slide_servo_active: bool,
    stage: str,
    current_local: np.ndarray,
    final_target_local: np.ndarray,
    surface_slow_radius_m: float = RIGHT_SURFACE_SLOW_RADIUS_M,
    recontact_step_m: float = RIGHT_SURFACE_APPROACH_SERVO_STEP_M,
) -> float:
    """Use the bounded contact step while causally recovering lost contact."""

    if recontact_active:
        return float(recontact_step_m)
    if preload_settling_active or slide_servo_active:
        return RIGHT_CONTACT_PRELOAD_SERVO_STEP_M
    return _right_servo_max_step(
        stage,
        current_local,
        final_target_local,
        surface_slow_radius_m=surface_slow_radius_m,
    )


def _right_contact_pose_ready(
    *,
    ordered_contact_seen: bool,
    current_left_stable_contact: bool,
    current_slider_contact_force: float,
    servo_stage: str,
) -> bool:
    """Latch the slide pose only with simultaneous ordered contact."""

    return bool(
        ordered_contact_seen
        and current_left_stable_contact
        and current_slider_contact_force > CONTACT_FORCE_THRESHOLD_N
        and servo_stage
        in {"planar_alignment", "contact_descent", "finger_reentry"}
    )


def _right_recontact_needed(
    ordered_contact_seen: bool,
    current_slider_contact_force: float,
    current_left_stable_contact: bool = True,
) -> bool:
    """Resume closed-loop contact search whenever a seen contact is lost."""

    return bool(
        ordered_contact_seen
        and (
            float(current_slider_contact_force) <= CONTACT_FORCE_THRESHOLD_N
            or not current_left_stable_contact
        )
    )


def _right_finger_withdrawal_needed(
    ordered_contact_seen: bool,
    current_left_stable_contact: bool,
    current_slider_contact_force: float = 0.0,
    recovery_ready_frames: int = 0,
) -> bool:
    """Withdraw the touching finger while the housing hold is absent."""

    if not ordered_contact_seen:
        return False
    if (
        current_left_stable_contact
        and float(current_slider_contact_force) > CONTACT_FORCE_THRESHOLD_N
    ):
        return False
    return bool(
        not current_left_stable_contact
        or int(recovery_ready_frames) < LEFT_RECOVERY_READY_CONTROL_FRAMES
    )


def _advance_left_recovery_ready_frames(
    current: int,
    current_left_stable_contact: bool,
    current_slider_contact_force: float,
    reentry_control_frames: int = RIGHT_FINGER_REENTRY_CONTROL_FRAMES,
) -> int:
    """Require a settled stabilizer hold before gradually restoring the finger."""

    if (
        not current_left_stable_contact
        or float(current_slider_contact_force) > CONTACT_FORCE_THRESHOLD_N
    ):
        return 0
    return min(
        LEFT_RECOVERY_READY_CONTROL_FRAMES
        + max(1, int(reentry_control_frames)),
        max(0, int(current)) + 1,
    )


def _effective_left_hand_preload_fraction(
    frame: int,
    start_frame: int | None,
    target_fraction: float,
    control_frames: int = LEFT_HOLD_HAND_PRELOAD_CONTROL_FRAMES,
) -> float:
    """Ramp hand preload only after the stable-contact latch is observed."""

    if start_frame is None:
        return 0.0
    return float(target_fraction) * min(
        1.0,
        max(0, int(frame) - int(start_frame))
        / float(max(1, int(control_frames))),
    )


def _left_fallback_grip_should_activate(
    frame: int,
    stabilized_seen: bool,
    already_active: bool,
    configured_override: bool,
    fingertip_target_distance_m: float | None = None,
) -> bool:
    """Latch a reachable fallback grip only after the nominal grip misses."""

    if already_active:
        return True
    if configured_override or stabilized_seen:
        return False
    distance = (
        float(fingertip_target_distance_m)
        if fingertip_target_distance_m is not None
        else math.nan
    )
    geometry_missed = bool(
        int(frame) >= 60
        and np.isfinite(distance)
        and distance > LEFT_FALLBACK_GRIP_DISTANCE_M
    )
    return bool(
        geometry_missed
        or int(frame) >= LEFT_FALLBACK_GRIP_ACTIVATION_FRAME
    )


def _left_hold_pose_should_refresh(
    physical_contact_now: bool,
    stable_contact_now: bool,
    stabilized_seen: bool,
    hold_pose_available: bool,
) -> bool:
    """Latch the palm pose once stable housing contact has been certified."""

    contact_ready = bool(
        physical_contact_now or stable_contact_now or stabilized_seen
    )
    return bool(
        contact_ready
        and (not stabilized_seen or not hold_pose_available)
    )


def _left_recovery_servo_should_apply(
    hold_pose_available: bool,
    physical_contact_now: bool,
    servo_pose_available: bool,
    stabilized_seen: bool,
    fixed_world_anchor_available: bool = False,
) -> bool:
    """Recover a lost hold without servoing through live physical contact."""

    return bool(
        hold_pose_available
        and servo_pose_available
        and not physical_contact_now
        and (not stabilized_seen or fixed_world_anchor_available)
    )


def _right_finger_reentry_fraction(
    ordered_contact_seen: bool,
    current_left_stable_contact: bool,
    current_slider_contact_force: float,
    recovery_ready_frames: int,
    reentry_control_frames: int = RIGHT_FINGER_REENTRY_CONTROL_FRAMES,
) -> float:
    """Return a bounded neutral-to-contact blend after recovery hysteresis."""

    if not ordered_contact_seen:
        return 1.0
    if (
        current_left_stable_contact
        and float(current_slider_contact_force) > CONTACT_FORCE_THRESHOLD_N
    ):
        return 1.0
    ready = int(recovery_ready_frames)
    if ready < LEFT_RECOVERY_READY_CONTROL_FRAMES:
        return 0.0
    return float(
        np.clip(
            (ready - LEFT_RECOVERY_READY_CONTROL_FRAMES)
            / float(max(1, int(reentry_control_frames))),
            0.0,
            1.0,
        )
    )


def _project_left_housing_surface_target_local(
    base_target_local: np.ndarray,
    fingertip_body_local: np.ndarray | None,
    *,
    x_range_m: tuple[float, float] = (
        LEFT_HOUSING_SURFACE_TARGET_LOCAL_X_RANGE_M
    ),
    trace_yz_m: tuple[float, float] = (
        LEFT_HOUSING_CONTACT_TRACE_TARGET_LOCAL_YZ_M
    ),
) -> np.ndarray:
    """Project onto a reachable point on a selected housing-side trace."""

    target = np.asarray(base_target_local, dtype=np.float64).reshape(3).copy()
    if fingertip_body_local is None:
        return target
    fingertip = np.asarray(fingertip_body_local, dtype=np.float64).reshape(3)
    if not np.isfinite(fingertip).all():
        return target
    target[0] = float(
        np.clip(
            fingertip[0],
            *x_range_m,
        )
    )
    target[1:] = trace_yz_m
    return target


def _preload_left_housing_surface_target_local(
    contact_target_local: np.ndarray,
    inside_target_local: np.ndarray,
) -> np.ndarray:
    """Advance once from the reachable contact section toward the housing."""

    contact = np.asarray(contact_target_local, dtype=np.float64).reshape(3)
    inside = np.asarray(inside_target_local, dtype=np.float64).reshape(3)
    inward = inside - contact
    distance = float(np.linalg.norm(inward))
    if not np.isfinite(distance) or distance <= 1.0e-12:
        return contact.copy()
    step_m = min(LEFT_HOUSING_FALLBACK_INWARD_PRELOAD_M, distance)
    return contact + (step_m / distance) * inward


def _advect_right_recontact_target_local(
    recontact_target_local: np.ndarray,
    joint_axis_blade_local: np.ndarray,
    progress_delta: float,
    joint_travel_m: float,
) -> np.ndarray:
    """Keep recontact on the reachable world locus as the blade advances."""

    target = np.asarray(
        recontact_target_local,
        dtype=np.float64,
    ).reshape(3).copy()
    axis = np.asarray(joint_axis_blade_local, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis).all() or norm <= 1.0e-12:
        return target
    distance = max(0.0, float(progress_delta)) * max(
        0.0,
        float(joint_travel_m),
    )
    target -= distance * axis / norm
    return target


def _preload_lost_right_contact_normal(
    target_local: np.ndarray,
    recontact_lifted: bool,
) -> np.ndarray:
    """Add blade-normal preload only after a proven contact is lost."""

    target = np.asarray(target_local, dtype=np.float64).reshape(3).copy()
    if recontact_lifted:
        target[2] -= RIGHT_RECONTACT_NORMAL_PRELOAD_M
    return target


def _preload_recovered_right_contact_normal(
    target_local: np.ndarray,
) -> np.ndarray:
    """Apply a small normal hold only after contact is physically restored."""

    target = np.asarray(target_local, dtype=np.float64).reshape(3).copy()
    target[2] -= RIGHT_RECOVERED_CONTACT_NORMAL_PRELOAD_M
    return target


def _preload_recovered_capture_in_plane(
    target_local: np.ndarray,
    slider_target_local: np.ndarray,
) -> np.ndarray:
    """Nudge a restored physical contact away from the slider edge."""

    target = np.asarray(target_local, dtype=np.float64).reshape(3).copy()
    slider = np.asarray(slider_target_local, dtype=np.float64).reshape(3)
    inward_xy = slider[:2] - target[:2]
    distance = float(np.linalg.norm(inward_xy))
    if distance > 1.0e-12:
        target[:2] += (
            RIGHT_RECOVERED_CAPTURE_IN_PLANE_PRELOAD_M
            * inward_xy
            / distance
        )
    return target


def _preload_recovered_capture_cross_edge(
    target_local: np.ndarray,
    slider_target_local: np.ndarray,
) -> np.ndarray:
    """Move a restored fingertip center inward across the button width."""

    target = np.asarray(target_local, dtype=np.float64).reshape(3).copy()
    slider = np.asarray(slider_target_local, dtype=np.float64).reshape(3)
    delta = float(slider[0] - target[0])
    target[0] += float(
        np.clip(
            delta,
            -RIGHT_RECOVERED_CAPTURE_CROSS_EDGE_PRELOAD_M,
            RIGHT_RECOVERED_CAPTURE_CROSS_EDGE_PRELOAD_M,
        )
    )
    return target


def _preload_recovered_right_contact_in_plane(
    target_local: np.ndarray,
    slider_target_local: np.ndarray,
    recovered_contact_active: bool,
    preload_m: float = RIGHT_RECOVERED_CONTACT_IN_PLANE_PRELOAD_M,
) -> np.ndarray:
    """Move a confirmed recovered contact slightly toward button center."""

    target = np.asarray(target_local, dtype=np.float64).reshape(3).copy()
    if not recovered_contact_active:
        return target
    slider = np.asarray(slider_target_local, dtype=np.float64).reshape(3)
    inward_xy = slider[:2] - target[:2]
    distance = float(np.linalg.norm(inward_xy))
    if distance > 1.0e-12:
        target[:2] += (
            max(0.0, float(preload_m))
            * inward_xy
            / distance
        )
    return target


def _advance_recovered_right_hand_blend(
    current: float,
    recontact_lifted: bool,
    grace_active: bool,
) -> float:
    """Ramp the recovered fingertip posture without a whole-hand step."""

    if not recontact_lifted or not grace_active:
        return 0.0
    return float(
        np.clip(
            float(current)
            + 1.0 / float(RIGHT_RECOVERED_HAND_BLEND_CONTROL_FRAMES),
            0.0,
            1.0,
        )
    )


def _advance_late_continuous_right_hand_blend(
    current: float,
    maximum_blend: float,
    blade_progress: float,
    start_progress: float,
    physical_contact_ready: bool,
) -> float:
    """Ramp a physical fingertip posture only during a proven late slide."""

    maximum = float(np.clip(float(maximum_blend), 0.0, 1.0))
    if maximum <= 0.0:
        return 0.0
    bounded_current = float(np.clip(float(current), 0.0, maximum))
    if (
        float(blade_progress) < float(start_progress)
        or not physical_contact_ready
    ):
        return bounded_current
    return float(
        min(
            maximum,
            bounded_current
            + maximum
            / float(RIGHT_LATE_CONTINUOUS_HAND_BLEND_CONTROL_FRAMES),
        )
    )


def _blend_recovered_fingertip_hand(
    contact_hand: np.ndarray,
    recovered_hand: np.ndarray,
    finger_index: int,
    fraction: float,
) -> np.ndarray:
    """Change only the selected finger while preserving the contact hand."""

    base = np.asarray(contact_hand, dtype=np.float64).reshape(20)
    recovered = np.asarray(recovered_hand, dtype=np.float64).reshape(20)
    result = base.copy()
    start = 4 * int(finger_index)
    if start < 0 or start + 4 > result.size:
        return result
    blend = float(np.clip(float(fraction), 0.0, 1.0))
    joint_slice = slice(start, start + 4)
    result[joint_slice] = (
        (1.0 - blend) * base[joint_slice]
        + blend * recovered[joint_slice]
    )
    return result


def _recovered_fingertip_hand_from_contact(
    contact_hand: np.ndarray,
    finger_index: int,
    extension_delta: float = RIGHT_RECOVERED_FINGER_EXTENSION_DELTA,
    abduction_delta_rad: float = 0.0,
    late_extension_delta: float = 0.0,
    late_proximal_scale: float = 1.0,
    late_middle_scale: float = 1.0,
    late_distal_tip_scale: float = 1.0,
) -> np.ndarray:
    """Add the small hold increment to the measured contact posture."""

    result = np.asarray(contact_hand, dtype=np.float64).reshape(20).copy()
    index = int(finger_index)
    start = 4 * index
    if index < 0 or index >= len(_FINGERTIP_EXTENSION):
        return result
    result[start : start + 4] += (
        float(extension_delta) * _FINGERTIP_EXTENSION[index]
    )
    late_profile = _FINGERTIP_EXTENSION[index].copy()
    late_profile[0] *= float(late_proximal_scale)
    late_profile[2] *= float(late_middle_scale)
    late_profile[3] *= float(late_distal_tip_scale)
    result[start : start + 4] += float(late_extension_delta) * late_profile
    if index > 0:
        result[start + 1] += float(abduction_delta_rad)
    return result


def _effective_recovered_finger_extension_delta(
    blade_progress: float,
    base_extension_delta: float,
    late_extension_delta: float = 0.0,
    start_progress: float = RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS,
) -> float:
    """Ramp an optional recovered-finger increment near task completion."""

    fraction = float(
        np.clip(
            (
                float(blade_progress)
                - float(start_progress)
            )
            / (
                SUCCESS_PROGRESS
                - float(start_progress)
            ),
            0.0,
            1.0,
        )
    )
    return float(base_extension_delta) + fraction * float(
        late_extension_delta
    )


def _effective_recovered_contact_in_plane_preload_m(
    blade_progress: float,
    late_preload_m: float,
    start_progress: float = RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS,
) -> float:
    """Increase recovered-target centering only near task completion."""

    fraction = float(
        np.clip(
            (float(blade_progress) - float(start_progress))
            / (SUCCESS_PROGRESS - float(start_progress)),
            0.0,
            1.0,
        )
    )
    return RIGHT_RECOVERED_CONTACT_IN_PLANE_PRELOAD_M + fraction * float(
        late_preload_m
    )


def _effective_recovered_late_normal_preload_m(
    blade_progress: float,
    late_preload_m: float,
    start_progress: float = RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS,
) -> float:
    """Ramp optional recovered-target normal preload near completion."""

    fraction = float(
        np.clip(
            (float(blade_progress) - float(start_progress))
            / (SUCCESS_PROGRESS - float(start_progress)),
            0.0,
            1.0,
        )
    )
    return fraction * float(late_preload_m)


def _effective_recovered_late_finger_abduction_delta(
    blade_progress: float,
    late_abduction_delta_rad: float,
    start_progress: float = RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS,
) -> float:
    """Ramp optional recovered-finger abduction near completion."""

    fraction = float(
        np.clip(
            (float(blade_progress) - float(start_progress))
            / (SUCCESS_PROGRESS - float(start_progress)),
            0.0,
            1.0,
        )
    )
    return fraction * float(late_abduction_delta_rad)


def _slide_fingertip_hand_from_contact(
    contact_hand: np.ndarray,
    finger_index: int,
    fraction: float,
    extension_delta: float,
    abduction_delta_rad: float = 0.0,
) -> np.ndarray:
    """Progressively extend only the contacting finger during the slide."""

    result = np.asarray(contact_hand, dtype=np.float64).reshape(20).copy()
    index = int(finger_index)
    start = 4 * index
    if index < 0 or index >= len(_FINGERTIP_EXTENSION):
        return result
    result[start : start + 4] += (
        float(np.clip(float(fraction), 0.0, 1.0))
        * float(extension_delta)
        * _FINGERTIP_EXTENSION[index]
    )
    if index > 0:
        result[start + 1] += (
            float(np.clip(float(fraction), 0.0, 1.0))
            * float(abduction_delta_rad)
        )
    return result


def _slide_finger_extension_fraction(
    slide_control_frames: int,
    total_control_frames: int,
    palm_travel_m: float,
) -> float:
    """Preserve the proven finger ramp while extending palm travel."""

    palm_fraction = float(
        np.clip(
            max(0, int(slide_control_frames))
            / float(max(2, int(total_control_frames)) - 1),
            0.0,
            1.0,
        )
    )
    commanded_travel_m = palm_fraction * max(0.0, float(palm_travel_m))
    return float(
        np.clip(
            commanded_travel_m / RIGHT_SLIDE_PALM_TRAVEL_M,
            0.0,
            1.0,
        )
    )


def _slide_late_travel_fraction(
    slide_control_frames: int,
    total_control_frames: int,
    palm_travel_m: float,
) -> float:
    """Ramp optional contact compensation only beyond proven travel."""

    configured_travel_m = max(0.0, float(palm_travel_m))
    extra_travel_m = configured_travel_m - RIGHT_SLIDE_PALM_TRAVEL_M
    if extra_travel_m <= 1.0e-12:
        return 0.0
    palm_fraction = float(
        np.clip(
            max(0, int(slide_control_frames))
            / float(max(2, int(total_control_frames)) - 1),
            0.0,
            1.0,
        )
    )
    commanded_travel_m = palm_fraction * configured_travel_m
    return float(
        np.clip(
            (commanded_travel_m - RIGHT_SLIDE_PALM_TRAVEL_M)
            / extra_travel_m,
            0.0,
            1.0,
        )
    )


def _late_slide_roll_relief_fraction(
    blade_progress: float,
    start_progress: float = RIGHT_SLIDE_ROLL_RELIEF_START_PROGRESS,
    end_progress: float = RIGHT_SLIDE_ROLL_RELIEF_END_PROGRESS,
) -> float:
    """Ramp wrist relief only after the proven early-contact trajectory."""

    start = float(start_progress)
    end = float(end_progress)
    progress = float(blade_progress)
    if not np.isfinite([progress, start, end]).all() or end <= start:
        return 0.0
    return float(np.clip((progress - start) / (end - start), 0.0, 1.0))


def _effective_slide_finger_extension_delta(
    slide_control_frames: int,
    total_control_frames: int,
    palm_travel_m: float,
    base_extension_delta: float,
    late_extension_delta: float = 0.0,
) -> float:
    """Add optional fingertip compensation only beyond proven travel."""

    return float(base_extension_delta) + (
        _slide_late_travel_fraction(
            slide_control_frames,
            total_control_frames,
            palm_travel_m,
        )
        * float(late_extension_delta)
    )


def _advance_right_recontact_grace_frames(
    current: int,
    recontact_lifted: bool,
    current_slider_contact_force: float,
    grace_control_frames: int = RIGHT_RECONTACT_CONTACT_GRACE_CONTROL_FRAMES,
) -> int:
    """Hold a recovered contact target briefly across sensor dropouts."""

    if not recontact_lifted:
        return 0
    if float(current_slider_contact_force) > CONTACT_FORCE_THRESHOLD_N:
        return max(1, int(grace_control_frames))
    return max(0, int(current) - 1)


def _advance_initial_contact_dropout_grace_frames(
    current: int,
    ordered_contact_seen: bool,
    current_left_stable_contact: bool,
    current_slider_contact_force: float,
    grace_control_frames: int,
) -> int:
    """Count a bounded first-contact dropout without changing the hand pose."""

    limit = max(0, int(grace_control_frames))
    if (
        limit == 0
        or not ordered_contact_seen
        or not current_left_stable_contact
        or float(current_slider_contact_force) > CONTACT_FORCE_THRESHOLD_N
    ):
        return 0
    return min(limit + 1, max(0, int(current)) + 1)


def _updated_last_proven_right_contact_local(
    previous_local: np.ndarray | None,
    current_local: np.ndarray,
    current_slider_contact_force: float,
    recontact_lifted: bool,
) -> np.ndarray | None:
    """Track the latest physical slider contact before any recovery lift."""

    if (
        float(current_slider_contact_force) > CONTACT_FORCE_THRESHOLD_N
        and not recontact_lifted
    ):
        return np.asarray(current_local, dtype=np.float64).reshape(3).copy()
    if previous_local is None:
        return None
    return np.asarray(previous_local, dtype=np.float64).reshape(3).copy()


def _advance_late_left_stability_grace_frames(
    current: int,
    current_left_stable_contact: bool,
    blade_progress: float,
    start_progress: float,
    grace_control_frames: int = LEFT_STABLE_CONTACT_CONTROL_GRACE_FRAMES,
) -> int:
    """Pause across bounded late-slide stabilization jitter."""

    start = float(start_progress)
    progress = float(blade_progress)
    if (
        not np.isfinite(start)
        or not np.isfinite(progress)
        or start >= SUCCESS_PROGRESS
        or progress < start
    ):
        return 0
    if bool(current_left_stable_contact):
        return max(1, int(grace_control_frames))
    return max(0, int(current) - 1)


def _advance_recovered_target_blend(
    current: float,
    recontact_lifted: bool,
    grace_active: bool,
    contact_settle_frames: int,
) -> float:
    """Blend toward advected tracking only after contact has settled."""

    if not recontact_lifted or not grace_active:
        return 0.0
    value = float(
        np.clip(
            float(current),
            0.0,
            RIGHT_RECOVERED_TARGET_MAX_BLEND,
        )
    )
    if (
        value <= 0.0
        and int(contact_settle_frames)
        < RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES
    ):
        return 0.0
    return float(
        np.clip(
            value
            + 1.0 / float(RIGHT_RECOVERED_TARGET_BLEND_CONTROL_FRAMES),
            0.0,
            RIGHT_RECOVERED_TARGET_MAX_BLEND,
        )
    )


def _settling_right_recontact_target_local(
    computed_target_local: np.ndarray,
    recovered_target_local: np.ndarray | None,
    grace_active: bool,
    advection_blend: float,
) -> np.ndarray:
    """Smoothly leave a recovered physical contact for advected tracking."""

    computed = np.asarray(
        computed_target_local,
        dtype=np.float64,
    ).reshape(3)
    if (
        recovered_target_local is not None
        and grace_active
    ):
        recovered = np.asarray(
            recovered_target_local,
            dtype=np.float64,
        ).reshape(3)
        blend = float(np.clip(float(advection_blend), 0.0, 1.0))
        return (1.0 - blend) * recovered + blend * computed
    return computed.copy()


def _advance_slide_control_frames(
    current: int,
    contact_ready: bool,
    total_control_frames: int = RIGHT_SLIDE_CONTROL_FRAMES,
) -> int:
    """Advance scripted slide time only while both physical contacts hold."""

    value = max(0, int(current))
    maximum = max(2, int(total_control_frames)) - 1
    if not contact_ready:
        return min(maximum, value)
    return min(maximum, value + 1)


def _latest_approach_timeout_frame(release_control_frames: int) -> int:
    """Reserve complete release plus 20 control frames before frame 1800."""

    return (
        1800
        - RIGHT_RELEASE_STABLE_RESERVE_CONTROL_FRAMES
        - max(1, int(release_control_frames))
    )


def _release_ready_after_success_hold(
    slide_complete_frame: int | None,
    frame_index: int,
    hold_control_frames: int,
) -> bool:
    """Delay release briefly after physical success without latching success."""

    return bool(
        slide_complete_frame is not None
        and int(frame_index) - int(slide_complete_frame)
        >= max(0, int(hold_control_frames))
    )


def _release_fraction_after_dwell(
    frame_index: int,
    release_frame: int,
    release_control_frames: int,
    dwell_control_frames: int,
) -> float:
    """Hold the captured contact pose briefly before lifting both hands."""

    elapsed = max(
        0,
        int(frame_index)
        - int(release_frame)
        - max(0, int(dwell_control_frames)),
    )
    return min(
        1.0,
        elapsed / float(max(1, int(release_control_frames) - 1)),
    )


def _post_success_overtravel_pose(
    pose: np.ndarray,
    joint_world_axis: np.ndarray,
    overtravel_m: float,
) -> np.ndarray:
    """Advance the captured right-hand release pose along the slide axis."""

    result = np.asarray(pose, dtype=np.float64).reshape(7).copy()
    axis = np.asarray(joint_world_axis, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if (
        not np.isfinite(result).all()
        or not np.isfinite(axis).all()
        or axis_norm <= 1.0e-12
    ):
        return result
    result[:3] += max(0.0, float(overtravel_m)) * axis / axis_norm
    return result


def _passive_drive_latch_motion_ready(
    right_slide_command: float,
    slide_complete_seen: bool,
) -> bool:
    """Allow optional causal forward assist from proven operator intent.

    The passive position target follows measured joint state every physics
    substep and deliberately does not use this gate.  Otherwise a transient
    contact/command dropout leaves the target at an older position and turns
    the utility-knife slider into a spring returning toward reset.
    """

    return bool(
        float(right_slide_command) > 1.0e-5 or slide_complete_seen
    )


def _right_slide_reanchor_needed(
    configured_progress: float,
    reanchor_origin_progress: float | None,
    ordered_contact_seen: bool,
    current_left_stable_contact: bool,
    current_slider_contact_force: float,
    current_progress: float,
) -> bool:
    """Request one causal second slide segment from a proven live contact."""

    threshold = float(configured_progress)
    progress = float(current_progress)
    return bool(
        threshold > 0.0
        and reanchor_origin_progress is None
        and ordered_contact_seen
        and current_left_stable_contact
        and float(current_slider_contact_force) > CONTACT_FORCE_THRESHOLD_N
        and np.isfinite(progress)
        and progress >= threshold
    )


def _right_slide_reanchor_extra_lead(
    configured_extra_lead_m: float,
    reanchor_frame: int | None,
    current_frame: int,
    ramp_control_frames: int = (
        RIGHT_SLIDE_REANCHOR_EXTRA_LEAD_RAMP_CONTROL_FRAMES
    ),
) -> float:
    """Ramp a bounded second-segment lead without a trigger-frame jump."""

    if reanchor_frame is None:
        return 0.0
    elapsed = max(0, int(current_frame) - int(reanchor_frame))
    fraction = float(
        np.clip(
            elapsed / float(max(1, int(ramp_control_frames))),
            0.0,
            1.0,
        )
    )
    return fraction * max(0.0, float(configured_extra_lead_m))


def _right_slide_pose_with_servo_correction(
    slide_pose: np.ndarray,
    servo_correction_world: np.ndarray | None,
    correction_scale: float,
    correction_axis_world: np.ndarray | None = None,
) -> np.ndarray:
    """Add bounded live fingertip feedback to the physical slide command.

    When an axis is supplied, compensate only tangential tracking lag.  The
    authored slide pose remains solely responsible for the contact-normal and
    cross-edge components, so live feedback cannot unload or roll the finger
    off the slider button.
    """

    pose = np.asarray(slide_pose, dtype=np.float64).reshape(7).copy()
    scale = float(correction_scale)
    if servo_correction_world is None or scale <= 0.0:
        return pose
    correction = np.asarray(
        servo_correction_world,
        dtype=np.float64,
    ).reshape(3)
    if not np.isfinite(pose).all() or not np.isfinite(correction).all():
        return pose
    if correction_axis_world is not None:
        axis = np.asarray(
            correction_axis_world,
            dtype=np.float64,
        ).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if not np.isfinite(axis).all() or axis_norm <= 1.0e-12:
            return pose
        unit_axis = axis / axis_norm
        correction = float(np.dot(correction, unit_axis)) * unit_axis
    pose[:3] += float(np.clip(scale, 0.0, 1.0)) * correction
    return pose


def _right_slide_pose_with_cross_edge_servo(
    slide_pose: np.ndarray,
    current_fingertip_local: np.ndarray,
    slider_target_local: np.ndarray,
    blade_world_cross_edge: np.ndarray,
    correction_scale: float,
    max_correction_m: float,
) -> np.ndarray:
    """Center the measured fingertip across the authored slider button.

    This feedback translates only the commanded hand pose.  Its error comes
    from the live fingertip position in blade-local x, is projected onto the
    matching world-space blade axis, and shares the existing cross-edge bound.
    """

    pose = np.asarray(slide_pose, dtype=np.float64).reshape(7).copy()
    scale = float(correction_scale)
    if scale <= 0.0:
        return pose
    current = np.asarray(current_fingertip_local, dtype=np.float64).reshape(3)
    target = np.asarray(slider_target_local, dtype=np.float64).reshape(3)
    axis = np.asarray(blade_world_cross_edge, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    limit = max(0.0, float(max_correction_m))
    if (
        not np.isfinite(pose).all()
        or not np.isfinite(current).all()
        or not np.isfinite(target).all()
        or not np.isfinite(axis).all()
        or axis_norm <= 1.0e-12
        or limit <= 0.0
    ):
        return pose
    local_error_m = float(np.clip(target[0] - current[0], -limit, limit))
    pose[:3] += (
        float(np.clip(scale, 0.0, 1.0))
        * local_error_m
        * axis
        / axis_norm
    )
    return pose


def _effective_slide_servo_correction_scale(
    slide_control_frames: int,
    total_control_frames: int,
    palm_travel_m: float,
    base_correction_scale: float,
    late_correction_scale: float = 0.0,
) -> float:
    """Ramp optional live feedback only beyond proven palm travel."""

    return float(base_correction_scale) + (
        _slide_late_travel_fraction(
            slide_control_frames,
            total_control_frames,
            palm_travel_m,
        )
        * float(late_correction_scale)
    )


def _left_counter_slide_pose(
    contact_pose: np.ndarray,
    hold_pose: np.ndarray,
    actual_pose: np.ndarray,
    joint_world_axis: np.ndarray,
    slide_control_frames: int,
    counter_slide_distance_m: float,
    total_control_frames: int = LEFT_COUNTER_SLIDE_CONTROL_FRAMES,
) -> np.ndarray:
    """Track a bounded anchor trajectory plus the live hold correction."""

    anchor = np.asarray(contact_pose, dtype=np.float64).reshape(7)
    hold = np.asarray(hold_pose, dtype=np.float64).reshape(7)
    actual = np.asarray(actual_pose, dtype=np.float64).reshape(7)
    pose = hold.copy()
    axis = np.asarray(joint_world_axis, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    frame = max(0, int(slide_control_frames))
    total = max(2, int(total_control_frames))
    if (
        not np.isfinite(axis).all()
        or axis_norm <= 1.0e-12
        or frame <= 0
        or not np.isfinite(anchor).all()
        or not np.isfinite(hold).all()
        or not np.isfinite(actual).all()
    ):
        return pose
    fraction = min(1.0, frame / float(total - 1))
    live_hold_correction = hold[:3] - actual[:3]
    pose[:3] = (
        anchor[:3]
        - fraction * max(0.0, float(counter_slide_distance_m))
        * axis
        / axis_norm
        + live_hold_correction
    )
    return pose


def _advance_right_contact_settle_frames(
    current: int,
    contact_ready: bool,
) -> int:
    """Hold a refreshed physical contact pose before starting the slide."""

    if not contact_ready:
        return 0
    return min(
        RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES,
        max(0, int(current)) + 1,
    )


def _right_slide_contact_pose_refresh_needed(
    *,
    contact_pose_ready: bool,
    contact_pose_missing: bool,
    slide_control_frames: int,
    recontact_lifted: bool,
    contact_settle_frames: int,
) -> bool:
    """Capture only the initial slide anchor; preserve motion on recontact."""

    del recontact_lifted, contact_settle_frames
    return bool(
        contact_pose_ready
        and (contact_pose_missing or int(slide_control_frames) == 0)
    )


def _maximum_right_slide_tangential_distance(
    current_progress: float,
    contact_progress: float,
    joint_range_m: float,
    maximum_lead_m: float,
    palm_travel_m: float,
    full_travel_fallback: bool = False,
) -> float:
    """Bound the palm trajectory while preserving a reachable fallback."""

    travel = max(0.0, float(palm_travel_m))
    if full_travel_fallback:
        return travel
    observed_forward_m = max(
        0.0,
        float(current_progress) - float(contact_progress),
    ) * max(0.0, float(joint_range_m))
    return min(
        travel,
        observed_forward_m + max(0.0, float(maximum_lead_m)),
    )


def _right_slide_servo_step_pose(
    contact_pose: np.ndarray,
    joint_world_axis: np.ndarray,
    blade_world_normal: np.ndarray,
    slide_control_frames: int,
    maximum_tangential_distance_m: float | None = None,
    blade_world_cross_edge: np.ndarray | None = None,
    cross_edge_correction_m: float = 0.0,
    normal_preload_m: float = RIGHT_SLIDE_NORMAL_PRELOAD_M,
    cross_edge_max_correction_m: float = (
        RIGHT_SLIDE_CROSS_EDGE_MAX_CORRECTION_M
    ),
    slide_yaw_relief_deg: float = 0.0,
    slide_pitch_relief_deg: float = 0.0,
    slide_roll_relief_deg: float = 0.0,
    slide_roll_relief_fraction: float | None = None,
    slide_roll_pivot_offset_world: np.ndarray | None = None,
    total_control_frames: int = RIGHT_SLIDE_CONTROL_FRAMES,
    palm_travel_m: float = RIGHT_SLIDE_PALM_TRAVEL_M,
    late_normal_preload_m: float = 0.0,
) -> np.ndarray:
    """Return an accumulated, normally preloaded physical slide pose."""

    pose = np.asarray(contact_pose, dtype=np.float64).reshape(7).copy()
    axis = np.asarray(joint_world_axis, dtype=np.float64).reshape(3)
    normal = np.asarray(blade_world_normal, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    normal_norm = float(np.linalg.norm(normal))
    if (
        not np.isfinite(axis).all()
        or not np.isfinite(normal).all()
        or axis_norm <= 1.0e-12
        or normal_norm <= 1.0e-12
    ):
        return pose
    fraction = float(
        np.clip(
            max(0, int(slide_control_frames))
            / float(max(2, int(total_control_frames)) - 1),
            0.0,
            1.0,
        )
    )
    tangential_distance_m = max(0.0, float(palm_travel_m)) * fraction
    if maximum_tangential_distance_m is not None:
        tangential_distance_m = min(
            tangential_distance_m,
            max(0.0, float(maximum_tangential_distance_m)),
        )
    pose[:3] += tangential_distance_m * axis / axis_norm
    effective_normal_preload_m = max(0.0, float(normal_preload_m)) + (
        _slide_late_travel_fraction(
            slide_control_frames,
            total_control_frames,
            palm_travel_m,
        )
        * max(0.0, float(late_normal_preload_m))
    )
    pose[:3] -= effective_normal_preload_m * normal / normal_norm
    unit_cross_edge: np.ndarray | None = None
    if blade_world_cross_edge is not None:
        cross_edge = np.asarray(
            blade_world_cross_edge,
            dtype=np.float64,
        ).reshape(3)
        cross_edge_norm = float(np.linalg.norm(cross_edge))
        if np.isfinite(cross_edge).all() and cross_edge_norm > 1.0e-12:
            unit_cross_edge = cross_edge / cross_edge_norm
            cross_edge_fraction = float(
                np.clip(
                    max(0, int(slide_control_frames))
                    / float(RIGHT_SLIDE_CROSS_EDGE_BLEND_CONTROL_FRAMES),
                    0.0,
                    1.0,
                )
            )
            pose[:3] += (
                cross_edge_fraction
                * float(
                    np.clip(
                        float(cross_edge_correction_m),
                        -max(0.0, float(cross_edge_max_correction_m)),
                        max(0.0, float(cross_edge_max_correction_m)),
                    )
                )
                * cross_edge
                / cross_edge_norm
            )
    relief_angle = fraction * math.radians(float(slide_yaw_relief_deg))
    if abs(relief_angle) > 1.0e-12:
        unit_normal = normal / normal_norm
        cross_matrix = np.asarray(
            [
                [0.0, -unit_normal[2], unit_normal[1]],
                [unit_normal[2], 0.0, -unit_normal[0]],
                [-unit_normal[1], unit_normal[0], 0.0],
            ],
            dtype=np.float64,
        )
        relief_rotation = (
            np.eye(3, dtype=np.float64)
            + math.sin(relief_angle) * cross_matrix
            + (1.0 - math.cos(relief_angle))
            * (cross_matrix @ cross_matrix)
        )
        pose_matrix = _pose_vector_to_matrix(pose)
        pose_matrix[:3, :3] = relief_rotation @ pose_matrix[:3, :3]
        pose[3:7] = _matrix_to_quat_wxyz(pose_matrix[:3, :3])
    pitch_relief_angle = fraction * math.radians(
        float(slide_pitch_relief_deg)
    )
    if abs(pitch_relief_angle) > 1.0e-12:
        unit_axis = axis / axis_norm
        axis_matrix = np.asarray(
            [
                [0.0, -unit_axis[2], unit_axis[1]],
                [unit_axis[2], 0.0, -unit_axis[0]],
                [-unit_axis[1], unit_axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        relief_rotation = (
            np.eye(3, dtype=np.float64)
            + math.sin(pitch_relief_angle) * axis_matrix
            + (1.0 - math.cos(pitch_relief_angle))
            * (axis_matrix @ axis_matrix)
        )
        pose_matrix = _pose_vector_to_matrix(pose)
        pose_matrix[:3, :3] = relief_rotation @ pose_matrix[:3, :3]
        pose[3:7] = _matrix_to_quat_wxyz(pose_matrix[:3, :3])
    roll_fraction = (
        fraction
        if slide_roll_relief_fraction is None
        else float(np.clip(float(slide_roll_relief_fraction), 0.0, 1.0))
    )
    roll_relief_angle = roll_fraction * math.radians(
        float(slide_roll_relief_deg)
    )
    if unit_cross_edge is not None and abs(roll_relief_angle) > 1.0e-12:
        pivot_offset: np.ndarray | None = None
        if slide_roll_pivot_offset_world is not None:
            candidate_offset = np.asarray(
                slide_roll_pivot_offset_world,
                dtype=np.float64,
            ).reshape(3)
            if np.isfinite(candidate_offset).all():
                pivot_offset = candidate_offset
        cross_matrix = np.asarray(
            [
                [0.0, -unit_cross_edge[2], unit_cross_edge[1]],
                [unit_cross_edge[2], 0.0, -unit_cross_edge[0]],
                [-unit_cross_edge[1], unit_cross_edge[0], 0.0],
            ],
            dtype=np.float64,
        )
        relief_rotation = (
            np.eye(3, dtype=np.float64)
            + math.sin(roll_relief_angle) * cross_matrix
            + (1.0 - math.cos(roll_relief_angle))
            * (cross_matrix @ cross_matrix)
        )
        if pivot_offset is not None:
            pose[:3] += pivot_offset - relief_rotation @ pivot_offset
        pose_matrix = _pose_vector_to_matrix(pose)
        pose_matrix[:3, :3] = relief_rotation @ pose_matrix[:3, :3]
        pose[3:7] = _matrix_to_quat_wxyz(pose_matrix[:3, :3])
    return pose


def _reanchored_right_contact_pose_from_command(
    previous_command_pose: np.ndarray,
    joint_world_axis: np.ndarray,
    blade_world_normal: np.ndarray,
    slide_control_frames: int,
    maximum_tangential_distance_m: float,
    blade_world_cross_edge: np.ndarray | None = None,
    cross_edge_correction_m: float = 0.0,
    normal_preload_m: float = RIGHT_SLIDE_NORMAL_PRELOAD_M,
    cross_edge_max_correction_m: float = (
        RIGHT_SLIDE_CROSS_EDGE_MAX_CORRECTION_M
    ),
    total_control_frames: int = RIGHT_SLIDE_CONTROL_FRAMES,
    palm_travel_m: float = RIGHT_SLIDE_PALM_TRAVEL_M,
    late_normal_preload_m: float = 0.0,
) -> np.ndarray:
    """Invert the zero-relief slide offsets to keep a reanchor continuous."""

    pose = np.asarray(previous_command_pose, dtype=np.float64).reshape(7).copy()
    axis = np.asarray(joint_world_axis, dtype=np.float64).reshape(3)
    normal = np.asarray(blade_world_normal, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    normal_norm = float(np.linalg.norm(normal))
    if (
        not np.isfinite(pose).all()
        or not np.isfinite(axis).all()
        or not np.isfinite(normal).all()
        or axis_norm <= 1.0e-12
        or normal_norm <= 1.0e-12
    ):
        return pose
    fraction = float(
        np.clip(
            max(0, int(slide_control_frames))
            / float(max(2, int(total_control_frames)) - 1),
            0.0,
            1.0,
        )
    )
    tangential_distance_m = min(
        max(0.0, float(palm_travel_m)) * fraction,
        max(0.0, float(maximum_tangential_distance_m)),
    )
    pose[:3] -= tangential_distance_m * axis / axis_norm
    effective_normal_preload_m = max(0.0, float(normal_preload_m)) + (
        _slide_late_travel_fraction(
            slide_control_frames,
            total_control_frames,
            palm_travel_m,
        )
        * max(0.0, float(late_normal_preload_m))
    )
    pose[:3] += effective_normal_preload_m * normal / normal_norm
    if blade_world_cross_edge is not None:
        cross_edge = np.asarray(
            blade_world_cross_edge,
            dtype=np.float64,
        ).reshape(3)
        cross_edge_norm = float(np.linalg.norm(cross_edge))
        if np.isfinite(cross_edge).all() and cross_edge_norm > 1.0e-12:
            cross_edge_fraction = float(
                np.clip(
                    max(0, int(slide_control_frames))
                    / float(RIGHT_SLIDE_CROSS_EDGE_BLEND_CONTROL_FRAMES),
                    0.0,
                    1.0,
                )
            )
            pose[:3] -= (
                cross_edge_fraction
                * float(
                    np.clip(
                        float(cross_edge_correction_m),
                        -max(0.0, float(cross_edge_max_correction_m)),
                        max(0.0, float(cross_edge_max_correction_m)),
                    )
                )
                * cross_edge
                / cross_edge_norm
            )
    return pose


def _right_contact_servo_only_needed(
    recontact_active: bool,
    preload_settling_active: bool,
    slide_motion_ready: bool,
    initial_contact_dropout_descent_active: bool = False,
) -> bool:
    """Hold slide motion while contact is being established or recovered."""

    return bool(
        recontact_active
        or initial_contact_dropout_descent_active
        or (preload_settling_active and not slide_motion_ready)
    )


def _right_contact_phase_flags(
    slider_contact_seen: bool,
    current_left_stable_contact: bool,
    current_slider_contact_force: float,
    contact_settle_frames: int,
) -> tuple[bool, bool]:
    """Enter preload/slide servo only on current dual-contact stability."""

    ready = bool(
        slider_contact_seen
        and current_left_stable_contact
        and float(current_slider_contact_force) > CONTACT_FORCE_THRESHOLD_N
    )
    settle_frames = int(contact_settle_frames)
    return (
        bool(ready and settle_frames < RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES),
        bool(ready and settle_frames >= RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES),
    )


def _late_slide_direct_recontact_needed(
    recontact_active: bool,
    current_left_stable_contact: bool,
    current_slider_contact_force: float,
    slide_control_frames: int,
) -> bool:
    """Keep the contacting finger closed during a late slide dropout."""

    return bool(
        recontact_active
        and current_left_stable_contact
        and float(current_slider_contact_force) <= CONTACT_FORCE_THRESHOLD_N
        and int(slide_control_frames) > 0
    )


def _capture_right_recontact_target_local(
    contact_center_local: np.ndarray,
) -> np.ndarray:
    """Keep the complete proven fingertip-center pose for recontact."""

    return np.asarray(contact_center_local, dtype=np.float64).reshape(3).copy()


def _preload_right_recontact_target_local(
    contact_center_local: np.ndarray,
    slider_target_local: np.ndarray,
) -> np.ndarray:
    """Move a proven fingertip center into the slider in-plane."""

    target = _capture_right_recontact_target_local(contact_center_local)
    slider = np.asarray(slider_target_local, dtype=np.float64).reshape(3)
    inward_xy = slider[:2] - target[:2]
    distance = float(np.linalg.norm(inward_xy))
    if distance > 1.0e-12:
        target[:2] += (
            RIGHT_RECONTACT_CONTACT_PRELOAD_M * inward_xy / distance
        )
    return target


def _latest_right_recontact_target_local(
    initial_target_local: np.ndarray,
    last_proven_contact_local: np.ndarray | None,
    _slider_target_local: np.ndarray,
    recontact_active: bool,
) -> np.ndarray:
    """Start a lost-contact search from the latest physical contact.

    The blade can move several centimetres after the initial contact pose is
    captured.  Returning to that stale pose after the short dropout grace
    would therefore search behind the moving slider.  Keep the original
    target before a loss, but reuse the exact latest pre-lift physical contact
    while recontact is active.  The downstream recovered-contact path applies
    its own bounded in-plane preload after the lift has begun.
    """

    initial = np.asarray(initial_target_local, dtype=np.float64).reshape(3)
    if not recontact_active or last_proven_contact_local is None:
        return initial.copy()
    return np.asarray(
        last_proven_contact_local,
        dtype=np.float64,
    ).reshape(3).copy()


def _recontact_right_fingertip_target_local(
    current_local: np.ndarray,
    recontact_target_local: np.ndarray,
    already_lifted: bool,
    current_slider_contact_force: float = 0.0,
    allow_descent: bool = True,
) -> tuple[np.ndarray, str, bool]:
    """Lift from a valid edge contact, then retrace a vertical recontact."""

    current = np.asarray(current_local, dtype=np.float64).reshape(3)
    recontact_target = np.asarray(
        recontact_target_local,
        dtype=np.float64,
    ).reshape(3)
    lifted = bool(
        already_lifted
        or (
            current[2] >= RIGHT_RECONTACT_CLEAR_HEIGHT_M
            and float(current_slider_contact_force)
            <= CONTACT_FORCE_THRESHOLD_N
        )
    )
    if lifted:
        if not allow_descent:
            target = recontact_target.copy()
            target[2] = RIGHT_RECONTACT_WAIT_HEIGHT_M
            return target, "recontact_wait", True
        return recontact_target.copy(), "contact_descent", True
    target = recontact_target.copy()
    target[2] = RIGHT_RECONTACT_HEIGHT_M
    return target, "recontact_lift", False


# Native Wuji joint ordering: thumb, index, middle, ring, pinky.
_POWER_GRIP = np.asarray(
    [
        0.60,
        0.10,
        0.84,
        0.70,
        0.68,
        0.00,
        0.90,
        0.78,
        0.70,
        0.00,
        0.92,
        0.80,
        0.68,
        0.00,
        0.90,
        0.78,
        0.64,
        0.00,
        0.86,
        0.74,
    ],
    dtype=np.float64,
)
# Small, range-safe extension of one selected Wuji fingertip.  This is kept
# separate from the power-grip posture because closing the whole hand moves
# the nearest fingertip away from the top-mounted blade slider.  The explicit
# diagnostic fraction defaults to zero until the live direction check has
# established that this posture moves the named fingertip toward the slider.
_FINGERTIP_EXTENSION = np.asarray(
    [
        [0.05, 0.0, -0.20, -0.20],
        [-0.08, 0.0, -0.20, -0.20],
        [-0.08, 0.0, -0.20, -0.20],
        [-0.08, 0.0, -0.20, -0.20],
        [-0.08, 0.0, -0.20, -0.20],
    ],
    dtype=np.float64,
)


def _right_touch_hand_posture(
    control_finger_number: int | None,
    extension_fraction: float,
    abduction_rad: float,
) -> np.ndarray:
    """Curl non-control fingers while exposing one fingertip to the slider."""

    if control_finger_number is None:
        return np.zeros(20, dtype=np.float64)
    finger_index = int(control_finger_number) - 1
    if finger_index < 0 or finger_index >= len(_FINGERTIP_EXTENSION):
        return np.zeros(20, dtype=np.float64)
    hand = RIGHT_NONCONTROL_GRIP_FRACTION * _POWER_GRIP
    joint_slice = slice(4 * finger_index, 4 * finger_index + 4)
    hand[joint_slice] = (
        float(extension_fraction) * _FINGERTIP_EXTENSION[finger_index]
    )
    if finger_index > 0:
        hand[joint_slice.start + 1] = float(abduction_rad)
    return hand


def _left_hold_hand_with_retracted_unsafe_finger(
    contact_hand: np.ndarray,
    fraction: float,
    *,
    distal_only: bool = False,
) -> np.ndarray:
    """Retract only the blade-path finger while preserving the housing hold."""

    base = np.asarray(contact_hand, dtype=np.float64).reshape(20)
    result = base.copy()
    finger_index = LEFT_UNSAFE_BLADE_FINGER_NUMBER - 1
    joint_start = 4 * finger_index
    joint_slice = (
        slice(joint_start + 2, joint_start + 4)
        if distal_only
        else slice(joint_start, joint_start + 4)
    )
    blend = float(np.clip(float(fraction), 0.0, 1.0))
    target = (
        base[joint_slice] + _FINGERTIP_EXTENSION[finger_index, 2:4]
        if distal_only
        else _POWER_GRIP[joint_slice]
    )
    result[joint_slice] = (
        (1.0 - blend) * base[joint_slice]
        + blend * target
    )
    return result


def _left_hold_hand_with_reinforced_safe_finger(
    contact_hand: np.ndarray,
    fraction: float,
    finger_number: int = LEFT_SAFE_STABILIZER_FINGER_NUMBER,
) -> np.ndarray:
    """Curl only the housing stabilizer finger toward the power grip."""

    base = np.asarray(contact_hand, dtype=np.float64).reshape(20)
    result = base.copy()
    finger_index = int(finger_number) - 1
    if not 0 <= finger_index < 5:
        raise ValueError("AS013 stabilizer finger number must be in [1, 5]")
    joint_start = 4 * finger_index
    joint_slice = slice(joint_start, joint_start + 4)
    blend = float(np.clip(float(fraction), 0.0, 1.0))
    result[joint_slice] = (
        (1.0 - blend) * base[joint_slice]
        + blend * _POWER_GRIP[joint_slice]
    )
    return result


def _as013_target_endpoint(direction: int) -> str:
    """Return the public endpoint label for a signed task direction."""

    return (
        TARGET_ENDPOINT_EXTENDED
        if int(direction) == TARGET_DIRECTION_EXTEND
        else TARGET_ENDPOINT_RETRACTED
    )


def _as013_target_progress(direction: int) -> float:
    """Return the normalized physical endpoint requested this episode."""

    return 1.0 if int(direction) == TARGET_DIRECTION_EXTEND else 0.0


def _as013_target_endpoint_reached(progress: float, direction: int) -> bool:
    """Check the appropriate authored end of the bidirectional joint."""

    if not np.isfinite(progress):
        return False
    if int(direction) == TARGET_DIRECTION_EXTEND:
        return bool(float(progress) >= EXTEND_SUCCESS_PROGRESS_MIN)
    return bool(float(progress) <= RETRACT_SUCCESS_PROGRESS_MAX)


def _as013_target_motion_delta(
    progress: float,
    anchor_progress: float,
    direction: int,
) -> float:
    """Return positive normalized travel toward the sampled endpoint."""

    if not np.isfinite([progress, anchor_progress]).all():
        return -math.inf
    sign = (
        TARGET_DIRECTION_EXTEND
        if int(direction) == TARGET_DIRECTION_EXTEND
        else TARGET_DIRECTION_RETRACT
    )
    return float(sign) * (float(progress) - float(anchor_progress))


def sample_reset_contract(
    random_state: np.random.RandomState,
    *,
    randomize: bool,
    center_xy: np.ndarray,
    hand_points: tuple[np.ndarray, ...] = (),
) -> tuple[np.ndarray, float, float, int]:
    """Return a bounded, seeded knife pose, slider start, and target."""

    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    if not randomize:
        # The default/non-random contract is the user's primary task: start
        # extended and move the slider to the bottom/inside endpoint.
        return (
            np.zeros(2, dtype=np.float64),
            0.0,
            0.80,
            TARGET_DIRECTION_RETRACT,
        )

    target_direction = (
        TARGET_DIRECTION_EXTEND
        if int(random_state.randint(0, 2))
        else TARGET_DIRECTION_RETRACT
    )
    reset_progress_range = (
        RESET_EXTEND_PROGRESS_RANGE
        if target_direction == TARGET_DIRECTION_EXTEND
        else RESET_RETRACT_PROGRESS_RANGE
    )

    for _ in range(128):
        offset = random_state.uniform(
            -RESET_XY_HALF_RANGE_M,
            RESET_XY_HALF_RANGE_M,
            size=2,
        ).astype(np.float64)
        candidate = center + offset
        if not (-0.52 <= float(candidate[0]) <= 0.16):
            continue
        if not (-0.48 <= float(candidate[1]) <= 0.48):
            continue
        if any(
            float(np.linalg.norm(candidate - np.asarray(point)[:2]))
            < RESET_HAND_CLEARANCE_M
            for point in hand_points
        ):
            continue
        yaw = float(
            random_state.uniform(
                -RESET_YAW_HALF_RANGE_DEG,
                RESET_YAW_HALF_RANGE_DEG,
            )
        )
        progress = float(random_state.uniform(*reset_progress_range))
        return offset, yaw, progress, target_direction
    # The certified center pose is the deterministic bounded fallback.  Joint
    # progress is still sampled from the task RNG so replay consumes the same
    # reset stream even when conservative hand screening rejects every offset.
    return (
        np.zeros(2, dtype=np.float64),
        0.0,
        float(random_state.uniform(*reset_progress_range)),
        target_direction,
    )


def _role_metric(
    metrics: dict[str, Any],
    canonical_name: str,
    legacy_name: str,
    default: Any,
) -> Any:
    """Read a canonical role metric while accepting a legacy alias."""

    if canonical_name in metrics:
        return metrics[canonical_name]
    return metrics.get(legacy_name, default)


def _legacy_failure_reason(reason: str) -> str:
    """Translate canonical hand-role reasons for old recording consumers."""

    return {
        "right_thumb_slider_contact_before_pickup": (
            "left_slider_contact_before_right_stabilization"
        ),
        "blade_progress_before_pickup": (
            "blade_progress_before_right_stabilization"
        ),
        "unsafe_right_hand_non_thumb_or_off_slider_contact": (
            "unsafe_left_hand_blade_contact"
        ),
        "left_hand_touched_knife": "unsafe_left_hand_blade_contact",
        "knife_dropped_after_pickup": "housing_displaced_or_dropped",
        "left_slider_contact_before_right_stabilization": (
            "right_slider_contact_before_left_stabilization"
        ),
        "blade_progress_before_right_stabilization": (
            "blade_progress_before_left_stabilization"
        ),
        "unsafe_right_hand_moving_component_contact": (
            "unsafe_left_hand_blade_contact"
        ),
        "unsafe_right_hand_blade_contact": "unsafe_left_hand_blade_contact",
    }.get(str(reason), str(reason))


def _canonical_contact_force_tuple(
    values: tuple[float, ...] | list[float],
    *,
    legacy_left_housing: float = 0.0,
    legacy_left_moving: float = 0.0,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """Normalize v3 forces while retaining the old five-value test/replay ABI."""

    forces = tuple(float(value) for value in values)
    if len(forces) == 10:
        return forces  # type: ignore[return-value]
    if len(forces) != 5:
        raise ValueError(f"AS013 expected 5 or 10 contact forces, got {len(forces)}")
    right_grip, thumb_moving, thumb_slider, right_unsafe, table_support = forces
    # A v2 aggregate did not distinguish palm from the other stabilizing
    # fingers or the two table-contacting knife bodies.  Mirroring those
    # aggregates is an explicit compatibility approximation, never the live
    # v3 runtime path.
    return (
        right_grip,
        right_grip,
        right_grip,
        thumb_moving,
        thumb_slider,
        right_unsafe,
        float(legacy_left_housing),
        float(legacy_left_moving),
        table_support,
        table_support,
    )


def _advance_as013_table_contact_event(
    *,
    current_contact: bool,
    contact_substeps: int,
    event_active: bool,
    event_count: int,
    armed: bool,
    required_substeps: int = DROP_TABLE_CONTACT_CONSECUTIVE_SUBSTEPS,
) -> tuple[int, bool, int, bool]:
    """Debounce the first table re-contact after a proven pickup."""

    bounded_required = max(1, int(required_substeps))
    bounded_count = max(0, int(event_count))
    if not armed:
        return 0, False, bounded_count, bool(bounded_count > 0)
    if not current_contact:
        # A contact event is edge-triggered: clear the active latch on the
        # first separated sample so a later re-contact can count as another
        # distinct table event.
        return 0, False, bounded_count, bool(bounded_count > 0)
    streak = min(bounded_required, max(0, int(contact_substeps)) + 1)
    active = bool(event_active)
    if streak >= bounded_required and not active:
        bounded_count += 1
        active = True
    return streak, active, bounded_count, bool(bounded_count > 0)


def _advance_as013_slider_contact_observation(
    *,
    current_contact: bool,
    contact_substeps: int,
    dropout_substeps: int,
    required_substeps: int = RIGHT_SLIDER_CONTACT_CONSECUTIVE_SUBSTEPS,
    dropout_grace_substeps: int = SLIDER_CONTACT_DROPOUT_GRACE_SUBSTEPS,
) -> tuple[bool, int, int]:
    """Tolerate short valid PhysX slider-row dropouts while acquiring touch."""

    bounded_required = max(1, int(required_substeps))
    bounded_grace = max(0, int(dropout_grace_substeps))
    streak = max(0, int(contact_substeps))
    dropout = max(0, int(dropout_substeps))
    if current_contact:
        return True, min(bounded_required, streak + 1), 0
    if streak > 0 and dropout < bounded_grace:
        return True, streak, dropout + 1
    return False, 0, 0


def success_from_metrics(metrics: dict[str, Any]) -> bool:
    role_schema = int(metrics.get("as013_role_schema_version", 1))
    if role_schema >= 5:
        return _v5_completion_ready(metrics)
    if role_schema >= 4:
        return bool(
            _v4_completion_ready_without_final_hold(metrics)
            and _v4_effective_final_hold_from_metrics(metrics)
        )
    if role_schema >= 3:
        return bool(
            float(metrics.get("blade_progress", -math.inf))
            >= SUCCESS_PROGRESS
            and bool(metrics.get("right_grip_seen", False))
            and bool(metrics.get("pickup_seen", False))
            and bool(metrics.get("right_thumb_slider_contact_seen", False))
            and bool(metrics.get("slide_progress_seen", False))
            and bool(metrics.get("ordered_slide_completion_seen", False))
            and bool(metrics.get("right_grip_contact_now", False))
            and bool(metrics.get("knife_lifted_now", False))
            and float(metrics.get("right_thumb_slider_contact_force", 0.0))
            > CONTACT_FORCE_THRESHOLD_N
            and float(
                metrics.get("left_forbidden_contact_force", math.inf)
            )
            <= CONTACT_FORCE_THRESHOLD_N
            and bool(metrics.get("contact_report_valid", False))
            and bool(metrics.get("held_stable_now", False))
            and not bool(metrics.get("left_hand_participation_failure", False))
            and not bool(metrics.get("failed", True))
        )
    ordered_completion = (
        bool(metrics["ordered_slide_completion_seen"])
        if "ordered_slide_completion_seen" in metrics
        else bool(role_schema < ORDERED_CONTACT_SCHEMA_VERSION)
        and bool(metrics.get("slide_progress_seen", False))
    )
    return bool(
        float(metrics.get("blade_progress", -math.inf)) >= SUCCESS_PROGRESS
        and bool(
            _role_metric(
                metrics,
                "right_stabilized_seen",
                "left_stabilized_seen",
                False,
            )
        )
        and bool(
            _role_metric(
                metrics,
                "left_slider_contact_seen",
                "right_slider_contact_seen",
                False,
            )
        )
        and bool(metrics.get("slide_progress_seen", False))
        and ordered_completion
        and bool(metrics.get("hands_released", False))
        and bool(metrics.get("table_support_contact", False))
        and bool(metrics.get("stable", False))
        and not bool(metrics.get("failed", True))
    )


def _v5_completion_ready(metrics: dict[str, Any]) -> bool:
    """Return the complete schema-v5 event-latched task contract.

    A valid pickup already proves a physical right-hand multi-point grasp and
    a 20 mm lift.  The left touch is independently debounced and latched after
    pickup.  Completion is then the sampled endpoint, without requiring either
    hand to remain in contact or the tool to wait through a stability window.
    """

    return bool(
        int(metrics.get("as013_role_schema_version", 1)) >= 5
        and bool(metrics.get("pickup_seen", False))
        and bool(metrics.get("left_slider_contact_seen", False))
        and _as013_target_endpoint_reached(
            float(metrics.get("blade_progress", math.nan)),
            int(
                metrics.get(
                    "target_direction",
                    TARGET_DIRECTION_RETRACT,
                )
            ),
        )
        and bool(metrics.get("target_endpoint_reached", False))
        and bool(metrics.get("ordered_slide_completion_seen", False))
        and not bool(metrics.get("failed", True))
    )


def _as013_stability_diagnostics(
    body_velocity: Any | None,
    blade_velocity: Any | None,
    *,
    contact_report_valid: bool = True,
    right_grip_contact_now: bool = True,
    knife_lifted_now: bool = True,
    left_slider_contact_force: float = math.inf,
    require_left_slider_contact: bool = False,
    linear_limit_mps: float = STABLE_LINEAR_SPEED_MPS,
    angular_limit_radps: float = STABLE_ANGULAR_SPEED_RADPS,
) -> dict[str, Any]:
    """Return component-level, human-readable AS013 stability diagnostics.

    ``stable`` intentionally remains a velocity-only gate over both rigid
    bodies.  This helper does not loosen that physical predicate; it exposes
    each measured component and the exact threshold/reason that blocked it.
    The optional contact/grip fields describe the separate final-hold gate so
    a failed interaction can be diagnosed without parsing a giant raw metrics
    dictionary or guessing whether the issue was speed or contact.
    """

    linear_limit = float(linear_limit_mps)
    angular_limit = float(angular_limit_radps)

    def inspect(label: str, velocity: Any | None) -> dict[str, Any]:
        prefix = f"{label}_"
        if velocity is None:
            return {
                f"{prefix}velocity_available": False,
                f"{prefix}velocity_finite": False,
                f"{prefix}linear_speed_mps": math.nan,
                f"{prefix}angular_speed_radps": math.nan,
                f"{prefix}linear_speed_ok": False,
                f"{prefix}angular_speed_ok": False,
                f"{prefix}velocity_components": None,
            }
        try:
            values = np.asarray(velocity, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            values = np.zeros(0, dtype=np.float64)
        available = bool(values.size >= 6)
        finite = bool(available and np.isfinite(values[:6]).all())
        linear_speed = (
            float(np.linalg.norm(values[:3])) if available and finite else math.nan
        )
        angular_speed = (
            # Body velocity truth is a 6-vector ``[vx, vy, vz, wx, wy, wz]``.
            # Ignore any accidental trailing tensor fields rather than letting
            # an unrelated value make the physical angular-speed gate fail.
            float(np.linalg.norm(values[3:6]))
            if available and finite
            else math.nan
        )
        linear_ok = bool(finite and linear_speed <= linear_limit)
        angular_ok = bool(finite and angular_speed <= angular_limit)
        return {
            f"{prefix}velocity_available": available,
            f"{prefix}velocity_finite": finite,
            f"{prefix}linear_speed_mps": linear_speed,
            f"{prefix}angular_speed_radps": angular_speed,
            f"{prefix}linear_speed_ok": linear_ok,
            f"{prefix}angular_speed_ok": angular_ok,
            f"{prefix}velocity_components": (
                [float(value) for value in values[:6]] if available else None
            ),
        }

    housing = inspect("housing", body_velocity)
    blade = inspect("blade", blade_velocity)
    reasons: list[str] = []
    for label, record in (("housing", housing), ("blade", blade)):
        if not record[f"{label}_velocity_available"]:
            reasons.append(f"{label} velocity unavailable (need 6 components)")
        elif not record[f"{label}_velocity_finite"]:
            reasons.append(f"{label} velocity contains NaN/Inf")
        else:
            linear_speed = record[f"{label}_linear_speed_mps"]
            angular_speed = record[f"{label}_angular_speed_radps"]
            if not record[f"{label}_linear_speed_ok"]:
                reasons.append(
                    f"{label} linear speed {linear_speed:.6f} m/s "
                    f"> {linear_limit:.6f} m/s limit"
                )
            if not record[f"{label}_angular_speed_ok"]:
                reasons.append(
                    f"{label} angular speed {angular_speed:.6f} rad/s "
                    f"> {angular_limit:.6f} rad/s limit"
                )
    speed_stable = bool(
        housing["housing_velocity_available"]
        and blade["blade_velocity_available"]
        and housing["housing_velocity_finite"]
        and blade["blade_velocity_finite"]
        and housing["housing_linear_speed_ok"]
        and housing["housing_angular_speed_ok"]
        and blade["blade_linear_speed_ok"]
        and blade["blade_angular_speed_ok"]
    )

    hold_reasons = list(reasons)
    if not bool(contact_report_valid):
        hold_reasons.append("contact report invalid/unavailable")
    if not bool(right_grip_contact_now):
        hold_reasons.append("right housing grip contact missing")
    if not bool(knife_lifted_now):
        hold_reasons.append("knife is not lifted clear of the table")
    if require_left_slider_contact:
        try:
            slider_force = float(left_slider_contact_force)
        except (TypeError, ValueError, OverflowError):
            slider_force = math.nan
        if not np.isfinite(slider_force) or slider_force <= CONTACT_FORCE_THRESHOLD_N:
            hold_reasons.append(
                "left slider contact force "
                f"{slider_force:.6f} N <= {CONTACT_FORCE_THRESHOLD_N:.6f} N threshold"
            )
    else:
        # Keep malformed diagnostic input fail-closed and serializable even
        # when the optional contact gate is disabled.  The runtime normally
        # supplies a finite float, but replay/debug callers may pass None or
        # a non-numeric value.
        try:
            slider_force = float(left_slider_contact_force)
        except (TypeError, ValueError):
            slider_force = math.nan

    result: dict[str, Any] = {
        **housing,
        **blade,
        "stable_speed_gate": speed_stable,
        "stable": speed_stable,
        "stable_linear_speed_limit_mps": linear_limit,
        "stable_angular_speed_limit_radps": angular_limit,
        "stable_failure_reasons": reasons,
        "stable_speed_failure_reasons": reasons,
        "held_stability_failure_reasons": hold_reasons,
        "contact_report_valid": bool(contact_report_valid),
        "right_grip_contact_now": bool(right_grip_contact_now),
        "knife_lifted_now": bool(knife_lifted_now),
        "left_slider_contact_force": slider_force,
        "left_slider_contact_required": bool(require_left_slider_contact),
        "left_slider_contact_threshold_n": float(CONTACT_FORCE_THRESHOLD_N),
    }
    return result


def _v4_completion_ready_without_final_hold(metrics: dict[str, Any]) -> bool:
    """Return the v4 prerequisites that the final-hold grace cannot mask."""

    return bool(
        int(metrics.get("as013_role_schema_version", 1)) >= 4
        and float(metrics.get("blade_progress", -math.inf))
        >= SUCCESS_PROGRESS
        and bool(metrics.get("right_grip_seen", False))
        and bool(metrics.get("pickup_seen", False))
        and bool(metrics.get("left_slider_contact_seen", False))
        and bool(metrics.get("slide_progress_seen", False))
        and bool(metrics.get("ordered_slide_completion_seen", False))
        and bool(metrics.get("contact_report_valid", False))
        and not bool(metrics.get("failed", True))
    )


def _v4_raw_final_hold_from_metrics(metrics: dict[str, Any]) -> bool:
    """Return the ungraced fourth success condition for role schema v4."""

    return bool(
        bool(metrics.get("right_grip_contact_now", False))
        and bool(metrics.get("knife_lifted_now", False))
        and float(metrics.get("left_slider_contact_force", 0.0))
        > CONTACT_FORCE_THRESHOLD_N
        and bool(metrics.get("held_stable_now", False))
        and not bool(metrics.get("failed", True))
    )


def _v4_effective_final_hold_from_metrics(metrics: dict[str, Any]) -> bool:
    """Return raw final-hold truth or an explicitly active runtime grace."""

    return bool(
        _v4_raw_final_hold_from_metrics(metrics)
        or (
            bool(metrics.get("final_hold_grace_active", False))
            and bool(metrics.get("final_hold_with_grace_now", False))
        )
    )


def _advance_final_hold_dropout_grace_frames(
    *,
    completion_ready: bool,
    raw_final_hold_now: bool,
    valid_hold_seen: bool,
    dropout_control_frames: int,
    grace_control_frames: int = FINAL_HOLD_DROPOUT_GRACE_CONTROL_FRAMES,
) -> tuple[bool, int, bool, bool]:
    """Advance the final-hold dropout grace state by one control frame.

    Returns ``(hold_window_retained_now, dropout_frames, grace_active,
    valid_hold_seen)``.  Grace starts only after a real valid hold frame and is
    cleared immediately when completion prerequisites fail.
    """

    if not completion_ready:
        return False, 0, False, False
    if raw_final_hold_now:
        return True, 0, False, True
    bounded_grace = max(0, int(grace_control_frames))
    bounded_dropout = max(0, int(dropout_control_frames))
    if valid_hold_seen and bounded_dropout < bounded_grace:
        return True, bounded_dropout + 1, True, True
    return False, 0, False, False


def _slide_complete_from_metrics(metrics: dict[str, Any]) -> bool:
    """Require the current schema's ordered operation contacts to be live."""

    role_schema = int(metrics.get("as013_role_schema_version", 1))
    if role_schema >= 5:
        return _v5_completion_ready(metrics)
    if role_schema >= 4:
        return bool(
            float(metrics.get("blade_progress", -math.inf))
            >= SUCCESS_PROGRESS
            and bool(metrics.get("pickup_seen", False))
            and bool(metrics.get("slide_progress_seen", False))
            and bool(metrics.get("ordered_slide_completion_seen", False))
            and bool(metrics.get("right_grip_contact_now", False))
            and bool(metrics.get("knife_lifted_now", False))
            and float(metrics.get("left_slider_contact_force", 0.0))
            > CONTACT_FORCE_THRESHOLD_N
        )
    if role_schema >= 3:
        return bool(
            float(metrics.get("blade_progress", -math.inf))
            >= SUCCESS_PROGRESS
            and bool(metrics.get("pickup_seen", False))
            and bool(metrics.get("slide_progress_seen", False))
            and bool(metrics.get("ordered_slide_completion_seen", False))
            and bool(metrics.get("right_grip_contact_now", False))
            and bool(metrics.get("knife_lifted_now", False))
            and float(metrics.get("right_thumb_slider_contact_force", 0.0))
            > CONTACT_FORCE_THRESHOLD_N
            and float(
                metrics.get("left_forbidden_contact_force", math.inf)
            )
            <= CONTACT_FORCE_THRESHOLD_N
            and bool(metrics.get("contact_report_valid", False))
            and not bool(metrics.get("failed", True))
        )
    completion_evidence = (
        bool(metrics.get("ordered_slide_completion_seen", False))
        if role_schema >= ORDERED_CONTACT_SCHEMA_VERSION
        else bool(metrics.get("slide_progress_seen", False))
    )
    return bool(
        float(metrics.get("blade_progress", -math.inf)) >= SUCCESS_PROGRESS
        and bool(metrics.get("slide_progress_seen", False))
        and completion_evidence
        and bool(
            _role_metric(
                metrics,
                "right_stable_contact_now",
                "left_stable_contact_now",
                False,
            )
        )
        and float(
            _role_metric(
                metrics,
                "left_slider_contact_force",
                "right_slider_contact_force",
                0.0,
            )
        )
        > CONTACT_FORCE_THRESHOLD_N
    )


class HandsOnAS013IsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Full Isaac 5.1 PhysX collection runtime for B320-AS013."""

    recorded_drive_target_replay_supported = True

    def __init__(self, *args: Any, **kwargs: Any):
        physics_backend = str(kwargs.get("physics_backend", "physx")).strip().lower()
        if physics_backend != "physx":
            raise ValueError(f"{TASK_CODE} supports PhysX only; Newton is forbidden")
        self._as013_joint_limits = (0.0, 1.0)
        self._as013_joint_prim_path = ""
        self._as013_joint_axis_body_local = JOINT_AXIS_BODY_LOCAL.copy()
        self._as013_reset_progress = 0.80
        self._as013_target_direction = TARGET_DIRECTION_RETRACT
        self._as013_target_endpoint = TARGET_ENDPOINT_RETRACTED
        self._as013_target_progress = 0.0
        self._as013_target_endpoint_reached_now = False
        self._as013_drive_target = 0.0
        self._as013_drive_assist_active = False
        self._as013_blade_drive_stiffness_scale = float(
            os.environ.get(BLADE_DRIVE_STIFFNESS_SCALE_ENV, 1.0)
        )
        if not (
            np.isfinite(self._as013_blade_drive_stiffness_scale)
            and 0.10 <= self._as013_blade_drive_stiffness_scale <= 2.0
        ):
            raise ValueError(
                "AS013 blade drive stiffness scale must be finite and in "
                "[0.10, 2.0]"
            )
        self._as013_blade_drive_stiffness_scaled = False
        self._as013_blade_drive_damping_scale = float(
            os.environ.get(BLADE_DRIVE_DAMPING_SCALE_ENV, 1.0)
        )
        if not (
            np.isfinite(self._as013_blade_drive_damping_scale)
            and 0.10 <= self._as013_blade_drive_damping_scale <= 2.0
        ):
            raise ValueError(
                "AS013 blade drive damping scale must be finite and in "
                "[0.10, 2.0]"
            )
        self._as013_blade_drive_damping_scaled = False
        self._as013_blade_joint_friction_scale = float(
            os.environ.get(BLADE_JOINT_FRICTION_SCALE_ENV, 1.0)
        )
        if not (
            np.isfinite(self._as013_blade_joint_friction_scale)
            and 0.10 <= self._as013_blade_joint_friction_scale <= 2.0
        ):
            raise ValueError(
                "AS013 blade joint friction scale must be finite and in "
                "[0.10, 2.0]"
            )
        self._as013_blade_joint_friction_scaled = False
        # The environment variable name is retained for compatibility with
        # sealed diagnostic manifests.  Schema v3 applies the setting to the
        # physical right arm that owns both grasp and thumb operation.
        self._as013_right_ik_dls_lambda = float(
            os.environ.get(
                RIGHT_IK_DLS_LAMBDA_ENV,
                RIGHT_IK_DLS_LAMBDA_DEFAULT,
            )
        )
        if not (
            np.isfinite(self._as013_right_ik_dls_lambda)
            and 0.005 <= self._as013_right_ik_dls_lambda <= 0.20
        ):
            raise ValueError(
                "AS013 right-arm DLS lambda must be finite and in "
                "[0.005, 0.20]"
            )
        self._as013_right_recontact_grace_control_frames = int(
            os.environ.get(
                SCRIPTED_RIGHT_RECONTACT_GRACE_FRAMES_ENV,
                RIGHT_RECONTACT_CONTACT_GRACE_CONTROL_FRAMES,
            )
        )
        if not 1 <= self._as013_right_recontact_grace_control_frames <= 60:
            raise ValueError(
                "AS013 right recontact grace frames must be in [1, 60]"
            )
        self._as013_initial_contact_dropout_grace_control_frames = int(
            os.environ.get(
                SCRIPTED_RIGHT_INITIAL_CONTACT_DROPOUT_GRACE_FRAMES_ENV,
                RIGHT_INITIAL_CONTACT_DROPOUT_GRACE_CONTROL_FRAMES_DEFAULT,
            )
        )
        if not 0 <= self._as013_initial_contact_dropout_grace_control_frames <= 30:
            raise ValueError(
                "AS013 initial contact dropout grace frames must be in [0, 30]"
            )
        self._as013_late_left_stability_grace_start_progress = float(
            os.environ.get(
                SCRIPTED_LATE_LEFT_STABILITY_GRACE_START_PROGRESS_ENV,
                LATE_LEFT_STABILITY_GRACE_START_PROGRESS_DEFAULT,
            )
        )
        if not (
            np.isfinite(
                self._as013_late_left_stability_grace_start_progress
            )
            and LATE_LEFT_STABILITY_GRACE_START_PROGRESS_RANGE[0]
            <= self._as013_late_left_stability_grace_start_progress
            <= LATE_LEFT_STABILITY_GRACE_START_PROGRESS_RANGE[1]
        ):
            raise ValueError(
                "AS013 late left-stability grace start progress must be "
                "finite and in [0.75, 0.80]"
            )
        self._as013_causal_drive_enabled = os.environ.get(
            CAUSAL_FINGERTIP_DRIVE_ENV,
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._as013_causal_drive_contact_finger_number: int | None = None
        self._as013_causal_drive_contact_joint_position: float | None = None
        self._as013_causal_drive_contact_fingertip_axis_coordinate: (
            float | None
        ) = None
        self._as013_causal_drive_contact_command_axis_coordinate: (
            float | None
        ) = None
        self._as013_causal_drive_forward_displacement_m = 0.0
        self._as013_causal_drive_fingertip_forward_m = 0.0
        self._as013_causal_drive_command_forward_m = 0.0
        self._as013_right_grip_contact_now = False
        self._as013_right_grip_contact_owner_count = 0
        self._as013_right_grip_seen = False
        self._as013_pickup_candidate_substeps = 0
        self._as013_pickup_seen = False
        self._as013_pickup_lift_m = 0.0
        self._as013_pickup_peak_lift_m = 0.0
        self._as013_knife_lifted_now = False
        self._as013_drop_substeps = 0
        self._as013_drop_failure = False
        self._as013_drop_near_table_now = False
        self._as013_drop_peak_descent_now = False
        self._as013_drop_falling_now = False
        self._as013_drop_table_impact_now = False
        self._as013_drop_evidence_now = False
        self._as013_table_contact_substeps = 0
        self._as013_table_contact_event_active = False
        self._as013_table_contact_event_count = 0
        self._as013_second_table_contact_seen = False
        self._as013_slider_contact_observation_substeps = 0
        self._as013_slider_contact_dropout_substeps = 0
        self._as013_slider_contact_effective_now = False
        self._as013_body_pose_report_valid = False
        self._as013_body_velocity_report_valid = False
        self._as013_body_vertical_velocity_mps = math.nan
        self._as013_left_hand_participation_substeps = 0
        self._as013_left_hand_participation_failure = False
        self._as013_reset_body_height_m = math.nan
        self._as013_recording_origin_accepted = False
        self._as013_left_stabilization_substeps = 0
        self._as013_left_stable_contact_now = False
        self._as013_left_stabilized_seen = False
        self._as013_right_slider_contact_substeps = 0
        self._as013_right_slider_contact_seen = False
        self._as013_preorder_slider_contact_substeps = 0
        self._as013_prestabilization_progress_substeps = 0
        self._as013_unsafe_blade_contact_substeps = 0
        self._as013_ordered_contact_progress: float | None = None
        self._as013_slide_progress_substeps = 0
        self._as013_slide_progress_now = False
        self._as013_slide_progress_seen = False
        self._as013_previous_blade_progress = math.nan
        self._as013_completion_forward_progress_now = False
        self._as013_completion_target_progress_now = False
        self._as013_ordered_slide_completion_seen = False
        self._as013_wrong_order_failure = False
        self._as013_unsafe_blade_contact_failure = False
        self._as013_nonfinite_failure = False
        self._as013_displacement_failure = False
        self._as013_failure_reason = ""
        self._as013_failure_reported = False
        self._as013_left_body_contact_view = None
        self._as013_left_contact_filter_paths: tuple[str, ...] = ()
        self._as013_right_blade_contact_view = None
        self._as013_right_contact_filter_paths: tuple[str, ...] = ()
        self._as013_left_blade_contact_view = None
        self._as013_body_table_contact_view = None
        self._as013_right_body_contact_view = None
        self._as013_right_stabilizer_filter_paths: tuple[str, ...] = ()
        self._as013_left_slider_blade_contact_view = None
        self._as013_left_slider_filter_paths: tuple[str, ...] = ()
        self._as013_right_unsafe_blade_contact_view = None
        self._as013_left_housing_release_contact_view = None
        self._as013_left_moving_release_contact_view = None
        self._as013_left_release_filter_paths: tuple[str, ...] = ()
        self._as013_left_housing_release_contact_force = math.inf
        self._as013_left_moving_release_contact_force = math.inf
        self._as013_left_release_contact_report_valid = False
        self._as013_right_grip_housing_contact_view = None
        self._as013_right_grip_filter_paths: tuple[str, ...] = ()
        self._as013_right_palm_housing_contact_view = None
        self._as013_right_non_thumb_housing_contact_view = None
        self._as013_right_thumb_moving_contact_view = None
        self._as013_right_thumb_filter_paths: tuple[str, ...] = ()
        self._as013_left_slider_moving_contact_view = None
        self._as013_left_slider_fingertip_filter_paths: tuple[str, ...] = ()
        self._as013_right_unsafe_moving_contact_view = None
        self._as013_left_housing_contact_view = None
        self._as013_left_moving_contact_view = None
        self._as013_blade_table_contact_view = None
        self._as013_contact_view_contracts: dict[str, dict[str, Any]] = {}
        self._as013_housing_table_contact_force = 0.0
        self._as013_moving_table_contact_force = 0.0
        self._as013_contact_views_ready = False
        self._as013_contact_report_valid = False
        self._as013_right_contact_rows_valid = False
        self._as013_right_contact_row_count = 0
        self._as013_right_slider_contact_row_count = 0
        self._as013_right_outside_slider_contact_row_count = 0
        self._as013_right_thumb_net_contact_force = 0.0
        self._as013_right_thumb_outside_slider_contact_force = 0.0
        self._as013_left_contact_rows_valid = False
        self._as013_left_blade_contact_row_count = 0
        self._as013_left_slider_contact_row_count = 0
        self._as013_start_body_pose: np.ndarray | None = None
        self._as013_previous_right_command: np.ndarray | None = None
        self._as013_previous_slider_command: np.ndarray | None = None
        self._as013_right_slide_command = 0.0
        self._as013_scripted_left_hold_pose: np.ndarray | None = None
        self._as013_scripted_left_nominal_world_rotation: (
            np.ndarray | None
        ) = None
        self._as013_scripted_left_hold_hand: np.ndarray | None = None
        self._as013_scripted_left_hand_preload_start_frame: int | None = None
        self._as013_scripted_left_counter_slide_anchor_pose: (
            np.ndarray | None
        ) = None
        self._as013_scripted_left_hold_fingertip_index: int | None = None
        self._as013_scripted_left_stabilizer_finger_number = (
            LEFT_SAFE_STABILIZER_FINGER_NUMBER
        )
        self._as013_scripted_left_surface_target_local: np.ndarray | None = None
        self._as013_scripted_left_surface_preload_steps = 0
        self._as013_scripted_left_surface_last_preload_frame: int | None = None
        self._as013_reset_yaw_degrees = 0.0
        self._as013_scripted_left_contact_standoff_m: float | None = None
        self._as013_scripted_left_recovery_body_relative: (
            np.ndarray | None
        ) = None
        self._as013_scripted_left_recovery_fingertip_index: int | None = None
        self._as013_scripted_left_recovery_fingertip_body_local: (
            np.ndarray | None
        ) = None
        self._as013_scripted_left_recovery_fingertip_target_world: (
            np.ndarray | None
        ) = None
        self._as013_scripted_left_fixed_anchor_offset_world = np.zeros(
            3,
            dtype=np.float64,
        )
        self._as013_scripted_left_fixed_anchor_slide_preloaded = False
        self._as013_scripted_left_fallback_grip_active = False
        self._as013_scripted_right_contact_pose: np.ndarray | None = None
        self._as013_scripted_right_contact_hand: np.ndarray | None = None
        self._as013_scripted_right_contact_frame: int | None = None
        self._as013_scripted_right_finger_number: int | None = None
        self._as013_scripted_right_approach_blade_relative: (
            np.ndarray | None
        ) = None
        self._as013_scripted_right_alignment_rotation_blade_relative: (
            np.ndarray | None
        ) = None
        self._as013_scripted_right_servo_stage = "vertical_entry"
        self._as013_scripted_right_recontact_lifted = False
        self._as013_scripted_right_recontact_target_blade_local: (
            np.ndarray | None
        ) = None
        self._as013_scripted_left_recovery_ready_frames = 0
        self._as013_scripted_right_target_blade_local: np.ndarray | None = None
        self._as013_scripted_timeout_right_pose: np.ndarray | None = None
        self._as013_scripted_timeout_left_pose: np.ndarray | None = None
        self._as013_scripted_timeout_right_hand: np.ndarray | None = None
        self._as013_scripted_timeout_left_hand: np.ndarray | None = None
        self._as013_scripted_slide_control_frames = 0
        self._as013_scripted_right_slide_reanchor_origin_progress: (
            float | None
        ) = None
        self._as013_scripted_right_slide_reanchor_frame: int | None = None
        self._as013_scripted_right_slide_cross_edge_correction_m = 0.0
        self._as013_scripted_right_contact_settle_frames = 0
        self._as013_scripted_right_recontact_grace_frames = 0
        self._as013_scripted_right_initial_contact_dropout_frames = 0
        self._as013_scripted_right_last_proven_contact_blade_local: (
            np.ndarray | None
        ) = None
        self._as013_scripted_late_left_stability_grace_frames = 0
        self._as013_scripted_right_recovered_target_blade_local: (
            np.ndarray | None
        ) = None
        self._as013_scripted_right_recovered_hand_blend = 0.0
        self._as013_scripted_right_late_continuous_hand_blend = 0.0
        self._as013_scripted_right_recovered_target_blend = 0.0
        self._as013_scripted_slide_complete_frame: int | None = None
        self._as013_scripted_release_frame: int | None = None
        self._as013_scripted_single_right_home: np.ndarray | None = None
        self._as013_scripted_single_left_home: np.ndarray | None = None
        self._as013_scripted_single_grip_pose: np.ndarray | None = None
        self._as013_scripted_single_grip_frame: int | None = None
        self._as013_scripted_single_pickup_pose: np.ndarray | None = None
        self._as013_scripted_single_pickup_frame: int | None = None
        self._as013_scripted_single_slide_hand: np.ndarray | None = None
        self._as013_scripted_bimanual_left_start_pose: np.ndarray | None = None
        self._as013_scripted_bimanual_left_target_pose: np.ndarray | None = None
        self._as013_scripted_bimanual_left_start_frame: int | None = None
        self._as013_final_hold_valid_seen = False
        self._as013_final_hold_dropout_control_frames = 0
        self._as013_final_hold_grace_active = False
        self._as013_final_hold_with_grace_now = False
        self._as013_last_metrics: dict[str, Any] = {}
        self._as013_spawn_contract: dict[str, Any] = {}
        kwargs.pop("task_name", None)
        super().__init__(TASK_NAME, *args, **kwargs)
        if (
            self._launch_isaac
            and self._as013_right_ik_dls_lambda != 0.04
        ):
            slider_robot = self._control_robot(SLIDER_HAND_SIDE)
            self._ik_controllers[SLIDER_HAND_SIDE] = self._ik_controller_cls(
                self._ik_controller_cfg_cls(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                    ik_params={
                        "lambda_val": self._as013_right_ik_dls_lambda
                    },
                ),
                num_envs=1,
                device=slider_robot.device,
            )
            print(
                "[HandsOn/AS013] slider-hand DLS lambda="
                f"{self._as013_right_ik_dls_lambda:.6g}",
                flush=True,
            )

    def _build_observation_space(self) -> spaces.Dict:
        observation_space = super()._build_observation_space()
        observation_spaces = dict(observation_space.spaces)
        state_spaces = dict(observation_spaces["state"].spaces)
        state_spaces.update(
            {
                "knife_body_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "blade_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "blade_joint_state": spaces.Box(-np.inf, np.inf, shape=(4,)),
                "knife_interaction_state": spaces.Box(-np.inf, np.inf, shape=(16,)),
            }
        )
        observation_spaces["state"] = spaces.Dict(state_spaces)
        return spaces.Dict(observation_spaces)

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        if self._stage is None:
            raise RuntimeError("AS013 stage is unavailable before asset composition")
        if not ASSET_PATH.is_file():
            raise FileNotFoundError(f"AS013 exact USD bundle is missing: {ASSET_PATH}")

        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        source_stage = Usd.Stage.Open(str(ASSET_PATH), load=Usd.Stage.LoadAll)
        if source_stage is None or not source_stage.GetDefaultPrim().IsValid():
            raise RuntimeError("AS013 utility-knife USD has no valid default prim")
        fresh_selected_asset = _is_fresh_selected_asset(source_stage)
        if (
            ASSET_VARIANT.lower() in {part.lower() for part in ASSET_PATH.parts}
            and not _ASSET_METADATA_LOADED
        ):
            raise RuntimeError(
                "AS013 selected asset metadata sidecar is required for fresh provenance"
            )
        # The replacement USD is authored with convex hulls and deliberately
        # has no inherited SDF schema.  Validate that declaration when the
        # sidecar is available; legacy DexVerse assets retain the historical
        # one-SDF-removal requirement below.
        metadata_sdf_value = _metadata_lookup(
            _ASSET_METADATA,
            "inherited_sdf_schema",
            "has_inherited_sdf_schema",
        )
        if fresh_selected_asset:
            if metadata_sdf_value is None:
                raise RuntimeError(
                    "AS013 selected asset metadata is missing "
                    "collision_summary.inherited_sdf_schema"
                )
            normalized_sdf_value = str(metadata_sdf_value).strip().lower()
            if normalized_sdf_value not in {"false", "0", "no", "none"}:
                raise RuntimeError(
                    "AS013 selected asset metadata must declare "
                    "inherited_sdf_schema=false"
                )
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
                UsdGeom.Tokens.guide,
            ],
            useExtentsHint=True,
        )
        local_range = bbox_cache.ComputeWorldBound(
            source_stage.GetDefaultPrim()
        ).ComputeAlignedRange()
        local_min = np.asarray(local_range.GetMin(), dtype=np.float64)

        knife = UsdGeom.Xform.Define(self._stage, KNIFE_ROOT_PATH).GetPrim()
        knife.GetReferences().ClearReferences()
        knife.GetReferences().AddReference(str(ASSET_PATH))
        position = np.asarray(
            [
                -0.12,
                0.0,
                _TABLE_TOP_Z
                + TARGET_TABLE_CLEARANCE_M
                - ASSET_SCALE * float(local_min[2]),
            ],
            dtype=np.float64,
        )
        xform = UsdGeom.Xformable(knife)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(*position))
        xform.AddRotateYOp(UsdGeom.XformOp.PrecisionDouble).Set(
            ASSET_REST_ROLL_DEG
        )
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(ASSET_SCALE, ASSET_SCALE, ASSET_SCALE)
        )
        sim_utils.update_stage()

        # Compose the selected source at its authored meter scale and derive
        # the table height from the final transformed bounds.  The replacement
        # asset is intentionally kept at zero rest roll (its source-derived
        # convex support is already coplanar); the historical asset's
        # -1.9415-degree support correction remains encoded only in its legacy
        # constants/evidence.  This is authored before the first PhysX start
        # and does not clamp live object state.
        target_min_z = _TABLE_TOP_Z + TARGET_TABLE_CLEARANCE_M
        composed_bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
                UsdGeom.Tokens.guide,
            ],
            useExtentsHint=True,
        )
        composed_min_z = float(
            composed_bbox_cache.ComputeWorldBound(knife)
            .ComputeAlignedRange()
            .GetMin()[2]
        )
        if not np.isfinite(composed_min_z):
            raise RuntimeError("AS013 rotated utility-knife bound is non-finite")
        position[2] += target_min_z - composed_min_z
        translate_op.Set(Gf.Vec3d(*position))
        sim_utils.update_stage()

        joint_prim = next(
            (
                prim
                for prim in Usd.PrimRange(
                    knife,
                    Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
                )
                if prim.IsA(UsdPhysics.PrismaticJoint)
                and prim.GetName() == JOINT_NAME
            ),
            None,
        )
        if joint_prim is None:
            raise RuntimeError(f"AS013 composed blade joint is missing: {JOINT_NAME}")
        joint = UsdPhysics.PrismaticJoint(joint_prim)
        body0_targets = list(joint.GetBody0Rel().GetTargets())
        if len(body0_targets) != 1 or body0_targets[0].name != "E_body_1":
            raise RuntimeError(
                "AS013 blade joint body0 must be the utility-knife housing"
            )
        axis_token = str(joint.GetAxisAttr().Get())
        axis_local = {
            "X": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            "Y": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            "Z": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        }.get(axis_token)
        local_rotation = joint.GetLocalRot0Attr().Get()
        if axis_local is None or local_rotation is None:
            raise RuntimeError("AS013 blade joint axis frame is unavailable")
        imaginary = local_rotation.GetImaginary()
        joint_frame_pose = np.asarray(
            [
                0.0,
                0.0,
                0.0,
                float(local_rotation.GetReal()),
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
            ],
            dtype=np.float64,
        )
        axis_body = (
            _pose_vector_to_matrix(joint_frame_pose)[:3, :3] @ axis_local
        )
        axis_norm = float(np.linalg.norm(axis_body))
        if not np.isfinite(axis_body).all() or axis_norm <= 1.0e-12:
            raise RuntimeError("AS013 blade joint axis frame is invalid")
        self._as013_joint_axis_body_local = axis_body / axis_norm
        lower = float(joint.GetLowerLimitAttr().Get())
        upper = float(joint.GetUpperLimitAttr().Get())
        if not np.isfinite([lower, upper]).all() or upper <= lower:
            raise RuntimeError(f"AS013 blade limits are invalid: {(lower, upper)}")
        self._as013_joint_limits = (lower, upper)
        self._as013_joint_prim_path = str(joint_prim.GetPath())
        reset_position = lower + 0.20 * (upper - lower)
        self._as013_drive_target = reset_position

        drive = UsdPhysics.DriveAPI.Get(joint_prim, "linear")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(BLADE_JOINT_DRIVE_STIFFNESS)
        drive.CreateDampingAttr(BLADE_JOINT_DRIVE_DAMPING)
        drive.CreateMaxForceAttr(BLADE_JOINT_DRIVE_MAX_FORCE)
        drive.CreateTargetPositionAttr(reset_position)
        for name, value in (
            ("physxJoint:jointFriction", BLADE_JOINT_FRICTION),
            (
                "physxJoint:dynamicFriction",
                BLADE_JOINT_DYNAMIC_FRICTION,
            ),
        ):
            attribute = joint_prim.GetAttribute(name)
            if not attribute.IsValid():
                attribute = joint_prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
            attribute.Set(value)
        # Isaac Sim 5.1 does not expose the legacy PhysX joint-state schema
        # API.  The authored drive target gives the newly composed articulation
        # a finite target, while ``_apply_task_pose_randomization`` writes the
        # exact seeded DOF position/velocity through the live articulation
        # tensor view at the reset boundary, before action frame zero.  Do not
        # make scene startup depend on a schema from other PhysX/USD builds.

        preview_material = UsdShade.Material.Define(
            self._stage,
            f"{KNIFE_ROOT_PATH}/AS013Materials/PreviewFallback",
        )
        shader = UsdShade.Shader.Define(
            self._stage,
            f"{preview_material.GetPath()}/PreviewSurface",
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.24, 0.28, 0.33)
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.35)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.42)
        preview_material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI.Apply(knife).Bind(
            preview_material,
            bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        )

        if self._randomize_dynamics:
            housing_contact_static_friction = float(
                self._random.uniform(*HOUSING_CONTACT_STATIC_FRICTION_RANGE)
            )
            housing_contact_dynamic_friction = (
                housing_contact_static_friction
                - (
                    HOUSING_CONTACT_STATIC_FRICTION_DEFAULT
                    - HOUSING_CONTACT_DYNAMIC_FRICTION_DEFAULT
                )
            )
        else:
            housing_contact_static_friction = (
                HOUSING_CONTACT_STATIC_FRICTION_DEFAULT
            )
            housing_contact_dynamic_friction = (
                HOUSING_CONTACT_DYNAMIC_FRICTION_DEFAULT
            )
        physics_material = UsdShade.Material.Define(
            self._stage,
            f"{KNIFE_ROOT_PATH}/AS013Materials/Physics",
        )
        material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
        material_api.CreateStaticFrictionAttr(
            housing_contact_static_friction
        )
        material_api.CreateDynamicFrictionAttr(
            housing_contact_dynamic_friction
        )
        material_api.CreateRestitutionAttr(0.0)
        # The default average combine mode would dilute the housing-side grip
        # against lower-friction Wuji fingertip materials.  Max combine keeps
        # the requested housing traction at the actual grasp contact.
        housing_physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(
            physics_material.GetPrim()
        )
        housing_physx_material_api.CreateFrictionCombineModeAttr("max")
        blade_contact_friction_default = (
            float(self._random.uniform(1.70, 2.30))
            if self._randomize_dynamics
            else BLADE_CONTACT_FRICTION_DEFAULT
        )
        blade_contact_friction = _configured_blade_contact_friction(
            os.environ.get(BLADE_CONTACT_FRICTION_ENV),
            blade_contact_friction_default,
        )
        blade_physics_material = UsdShade.Material.Define(
            self._stage,
            f"{KNIFE_ROOT_PATH}/AS013Materials/BladePhysics",
        )
        blade_material_api = UsdPhysics.MaterialAPI.Apply(
            blade_physics_material.GetPrim()
        )
        blade_material_api.CreateStaticFrictionAttr(blade_contact_friction)
        blade_material_api.CreateDynamicFrictionAttr(
            max(0.75, blade_contact_friction - 0.10)
        )
        blade_material_api.CreateRestitutionAttr(0.0)
        # PhysX's default average combine mode would dilute the deliberately
        # high slider-side traction against the robot-hand material.  Max
        # combine preserves the slider coefficient so a fingertip can move it
        # without excessive normal preload.
        blade_physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(
            blade_physics_material.GetPrim()
        )
        blade_physx_material_api.CreateFrictionCombineModeAttr("max")

        rigid_paths: list[str] = []
        collision_paths: list[str] = []
        gravity_enabled_bodies = 0
        articulation_sleep_threshold_count = 0
        housing_colliders_tuned = 0
        blade_colliders_bound = 0
        housing_sdf_schemas_removed = 0
        inertia_scaled_bodies = 0
        mass_scale = (
            float(self._random.uniform(0.8, 1.2))
            if self._randomize_dynamics
            else 1.0
        )
        for prim in Usd.PrimRange(
            knife,
            Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
        ):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_api = PhysxSchema.PhysxArticulationAPI(prim)
                if not articulation_api:
                    articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                sleep_threshold = articulation_api.GetSleepThresholdAttr()
                if not sleep_threshold.IsValid():
                    sleep_threshold = articulation_api.CreateSleepThresholdAttr()
                sleep_threshold.Set(RESET_SLEEP_THRESHOLD)
                articulation_sleep_threshold_count += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_paths.append(str(prim.GetPath()))
                rigid_owner = prim.GetParent()
                while rigid_owner.IsValid() and not rigid_owner.HasAPI(
                    UsdPhysics.RigidBodyAPI
                ):
                    if rigid_owner == knife:
                        break
                    rigid_owner = rigid_owner.GetParent()
                collider_material = physics_material
                if (
                    rigid_owner.IsValid()
                    and rigid_owner.GetName() == "E_link_blade_2"
                ):
                    collider_material = blade_physics_material
                    blade_colliders_bound += 1
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    collider_material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
                if (
                    rigid_owner.IsValid()
                    and rigid_owner.GetName() == "E_body_1"
                    and prim.IsA(UsdGeom.Mesh)
                ):
                    if prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
                        prim.RemoveAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
                        housing_sdf_schemas_removed += 1
                    mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
                    if not mesh_collision:
                        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                    approximation = mesh_collision.GetApproximationAttr()
                    if not approximation.IsValid():
                        approximation = mesh_collision.CreateApproximationAttr()
                    approximation.Set(HOUSING_COLLISION_APPROXIMATION)
                    if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                        physx_collision = PhysxSchema.PhysxCollisionAPI(prim)
                    else:
                        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    contact_offset = physx_collision.GetContactOffsetAttr()
                    if not contact_offset.IsValid():
                        contact_offset = physx_collision.CreateContactOffsetAttr()
                    contact_offset.Set(HOUSING_CONTACT_OFFSET_M)
                    rest_offset = physx_collision.GetRestOffsetAttr()
                    if not rest_offset.IsValid():
                        rest_offset = physx_collision.CreateRestOffsetAttr()
                    rest_offset.Set(HOUSING_REST_OFFSET_M)
                    housing_colliders_tuned += 1
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            rigid_paths.append(str(prim.GetPath()))
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            else:
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            disable_gravity = rigid_api.GetDisableGravityAttr()
            if not disable_gravity.IsValid():
                disable_gravity = rigid_api.CreateDisableGravityAttr()
            disable_gravity.Set(False)
            for getter_name, creator_name, value in (
                (
                    "GetLinearDampingAttr",
                    "CreateLinearDampingAttr",
                    RESET_LINEAR_DAMPING,
                ),
                (
                    "GetAngularDampingAttr",
                    "CreateAngularDampingAttr",
                    RESET_ANGULAR_DAMPING,
                ),
                (
                    "GetSleepThresholdAttr",
                    "CreateSleepThresholdAttr",
                    RESET_SLEEP_THRESHOLD,
                ),
                (
                    "GetSolverPositionIterationCountAttr",
                    "CreateSolverPositionIterationCountAttr",
                    RESET_SOLVER_POSITION_ITERATIONS,
                ),
                (
                    "GetSolverVelocityIterationCountAttr",
                    "CreateSolverVelocityIterationCountAttr",
                    RESET_SOLVER_VELOCITY_ITERATIONS,
                ),
            ):
                attribute = getattr(rigid_api, getter_name)()
                if not attribute.IsValid():
                    attribute = getattr(rigid_api, creator_name)()
                attribute.Set(value)
            gravity_enabled_bodies += 1
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                report_api = PhysxSchema.PhysxContactReportAPI.Get(
                    self._stage, prim.GetPath()
                )
            else:
                report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report_api.CreateThresholdAttr().Set(0.0)
            mass_api = UsdPhysics.MassAPI(prim)
            mass_attr = mass_api.GetMassAttr() if mass_api else None
            if (
                mass_scale != 1.0
                and mass_attr is not None
                and mass_attr.IsValid()
                and mass_attr.HasAuthoredValueOpinion()
            ):
                old_mass = float(mass_attr.Get())
                if old_mass > 0.0:
                    mass_attr.Set(old_mass * mass_scale)
                    # The generated asset authors a dimension-derived
                    # diagonal inertia.  Keep the density/shape relationship
                    # intact when dynamics randomization scales mass; leaving
                    # inertia unchanged would silently change the object's
                    # rotational response for the same geometry.
                    inertia_attr = mass_api.GetDiagonalInertiaAttr()
                    if (
                        inertia_attr.IsValid()
                        and inertia_attr.HasAuthoredValueOpinion()
                    ):
                        inertia = inertia_attr.Get()
                        inertia_attr.Set(
                            Gf.Vec3f(
                                float(inertia[0]) * mass_scale,
                                float(inertia[1]) * mass_scale,
                                float(inertia[2]) * mass_scale,
                            )
                        )
                        inertia_scaled_bodies += 1

        if len(rigid_paths) < 2 or len(collision_paths) < 2:
            raise RuntimeError(
                "AS013 exact asset must expose housing/blade rigid bodies and colliders"
            )
        if housing_colliders_tuned != 1:
            raise RuntimeError(
                "AS013 exact asset must expose one housing mesh collider; "
                f"tuned {housing_colliders_tuned}"
            )
        if fresh_selected_asset:
            # The selected generated colliders are source-derived convex
            # hulls and metadata explicitly declares no inherited SDF API.
            # Treat any discovered SDF as stale provenance rather than
            # silently accepting/removing it.
            if housing_sdf_schemas_removed != 0:
                raise RuntimeError(
                    "AS013 fresh housing collider unexpectedly carries "
                    f"SDF schemas; removed {housing_sdf_schemas_removed}"
                )
        elif housing_sdf_schemas_removed != 1:
            raise RuntimeError(
                "AS013 legacy housing collider must remove one inherited SDF "
                f"schema; removed {housing_sdf_schemas_removed}"
            )
        if blade_colliders_bound < 1:
            raise RuntimeError(
                "AS013 exact asset must expose at least one "
                "E_link_blade_2 collider"
            )
        self._as013_spawn_contract = {
            "asset_path": str(ASSET_PATH),
            "asset_variant": ASSET_VARIANT,
            "role_schema_version": ROLE_SCHEMA_VERSION,
            "operation_semantics": (
                "right_hand_pickup_left_hand_touch_bidirectional_endpoint"
            ),
            "operator_hand": "bimanual",
            "inactive_hand": INACTIVE_HAND_SIDE,
            "stabilizer_hand": STABILIZER_HAND_SIDE,
            "slider_hand": SLIDER_HAND_SIDE,
            "left_slider_fingertip_rigid_bodies": list(
                LEFT_SLIDER_FINGERTIP_RIGID_BODY_NAMES
            ),
            "right_thumb_distal_rigid_body": (
                RIGHT_THUMB_DISTAL_RIGID_BODY_NAME
            ),
            "right_thumb_tip_collider": RIGHT_THUMB_TIP_COLLIDER_NAME,
            "pickup_lift_height_m": PICKUP_LIFT_HEIGHT_M,
            "drop_detection": (
                "debounced_post_pickup_drop_with_downward_evidence"
            ),
            "drop_table_contact_force_threshold_n": (
                DROP_TABLE_CONTACT_FORCE_THRESHOLD_N
            ),
            "drop_table_contact_required_substeps": (
                DROP_TABLE_CONTACT_CONSECUTIVE_SUBSTEPS
            ),
            "slider_contact_region_tolerance_m": (
                SLIDER_CONTACT_REGION_TOLERANCE_M
            ),
            "slider_contact_dropout_grace_substeps": (
                SLIDER_CONTACT_DROPOUT_GRACE_SUBSTEPS
            ),
            "drop_return_lift_height_m": DROP_RETURN_LIFT_HEIGHT_M,
            "drop_descent_from_peak_m": DROP_DESCENT_FROM_PEAK_M,
            "drop_falling_vertical_speed_mps": (
                DROP_FALLING_VERTICAL_SPEED_MPS
            ),
            "drop_table_impact_force_threshold_n": (
                DROP_TABLE_IMPACT_FORCE_THRESHOLD_N
            ),
            "right_grip_required_contact_owners": (
                RIGHT_GRIP_REQUIRED_CONTACT_OWNERS
            ),
            "right_grip_contact_region": "full_right_hand_on_housing",
            "right_palm_contact_required": False,
            "target_endpoints": [
                TARGET_ENDPOINT_RETRACTED,
                TARGET_ENDPOINT_EXTENDED,
            ],
            "retract_success_progress_max": (
                RETRACT_SUCCESS_PROGRESS_MAX
            ),
            "extend_success_progress_min": EXTEND_SUCCESS_PROGRESS_MIN,
            "reset_retract_progress_range": list(
                RESET_RETRACT_PROGRESS_RANGE
            ),
            "reset_extend_progress_range": list(
                RESET_EXTEND_PROGRESS_RANGE
            ),
            "reset_xy_half_range_m": RESET_XY_HALF_RANGE_M,
            "reset_yaw_half_range_deg": RESET_YAW_HALF_RANGE_DEG,
            "fresh_selected_asset": fresh_selected_asset,
            "asset_metadata_path": str(ASSET_METADATA_PATH),
            "asset_metadata_loaded": _ASSET_METADATA_LOADED,
            "asset_source_archive_sha256": ASSET_SOURCE_ARCHIVE_SHA256,
            "composition_root": KNIFE_ROOT_PATH,
            "joint_path": self._as013_joint_prim_path,
            "joint_limits": [lower, upper],
            "joint_axis_body_local": (
                self._as013_joint_axis_body_local.tolist()
            ),
            "slider_button_local_bounds_m": {
                "min": SLIDER_BUTTON_LOCAL_MIN_M.tolist(),
                "max": SLIDER_BUTTON_LOCAL_MAX_M.tolist(),
            },
            "slider_button_center_body0_m": BUTTON_REGION_CENTER_BODY0_M.tolist(),
            "slider_button_center_body1_m": BUTTON_REGION_CENTER_LOCAL_M.tolist(),
            "left_slider_fingertip_target_local_xy_m": (
                LEFT_SLIDER_FINGERTIP_TARGET_LOCAL_XY_M.tolist()
            ),
            # Retain the historical key/value for sealed manifests, but mark
            # it unambiguously as a role-name alias rather than a physical
            # right-hand target.
            "right_fingertip_target_local_xy_m": (
                LEFT_SLIDER_FINGERTIP_TARGET_LOCAL_XY_M.tolist()
            ),
            "right_fingertip_target_local_xy_m_legacy": (
                LEFT_SLIDER_FINGERTIP_TARGET_LOCAL_XY_M.tolist()
            ),
            "right_fingertip_target_local_xy_m_is_legacy_role_alias": True,
            "left_slider_orientation_live_certified": False,
            "left_housing_surface_target_local_m": (
                LEFT_HOUSING_SURFACE_TARGET_LOCAL_M.tolist()
            ),
            "left_housing_surface_target_local_x_range_m": list(
                LEFT_HOUSING_SURFACE_TARGET_LOCAL_X_RANGE_M
            ),
            "left_housing_contact_trace_target_local_yz_m": list(
                LEFT_HOUSING_CONTACT_TRACE_TARGET_LOCAL_YZ_M
            ),
            "right_housing_surface_target_local_m": (
                RIGHT_HOUSING_SURFACE_TARGET_LOCAL_M.tolist()
            ),
            "right_housing_surface_target_local_x_range_m": list(
                RIGHT_HOUSING_SURFACE_TARGET_LOCAL_X_RANGE_M
            ),
            "right_housing_contact_trace_target_local_yz_m": list(
                RIGHT_HOUSING_CONTACT_TRACE_TARGET_LOCAL_YZ_M
            ),
            "rigid_body_count": len(rigid_paths),
            "collision_shape_count": len(collision_paths),
            "gravity_enabled_body_count": gravity_enabled_bodies,
            "articulation_sleep_threshold_count": (
                articulation_sleep_threshold_count
            ),
            "uniform_scale": ASSET_SCALE,
            "rest_roll_degrees": ASSET_REST_ROLL_DEG,
            "table_clearance_m": TARGET_TABLE_CLEARANCE_M,
            "housing_colliders_tuned": housing_colliders_tuned,
            "housing_sdf_schemas_removed": housing_sdf_schemas_removed,
            "housing_sdf_guard": (
                "fresh_provenance_no_inherited_sdf"
                if fresh_selected_asset
                else "legacy_exactly_one_inherited_sdf"
            ),
            "housing_collision_approximation": (
                HOUSING_COLLISION_APPROXIMATION
            ),
            "housing_contact_offset_m": HOUSING_CONTACT_OFFSET_M,
            "housing_rest_offset_m": HOUSING_REST_OFFSET_M,
            # Preserve the historical generic key for sealed consumers while
            # exposing explicit housing-side material values for new audits.
            "friction": housing_contact_static_friction,
            "housing_contact_static_friction": (
                housing_contact_static_friction
            ),
            "housing_contact_dynamic_friction": (
                housing_contact_dynamic_friction
            ),
            "housing_friction_combine_mode": "max",
            "housing_grip_material_scope": (
                "single_housing_collider_including_side_faces"
            ),
            "blade_contact_friction": blade_contact_friction,
            "blade_friction_combine_mode": "max",
            "blade_colliders_bound": blade_colliders_bound,
            "mass_scale": mass_scale,
            "inertia_scaled_bodies": inertia_scaled_bodies,
            "linear_damping": RESET_LINEAR_DAMPING,
            "angular_damping": RESET_ANGULAR_DAMPING,
            "sleep_threshold": RESET_SLEEP_THRESHOLD,
            "solver_position_iterations": RESET_SOLVER_POSITION_ITERATIONS,
            "solver_velocity_iterations": RESET_SOLVER_VELOCITY_ITERATIONS,
            "reset_position": reset_position,
            "blade_joint_drive_damping": BLADE_JOINT_DRIVE_DAMPING,
            "blade_joint_static_friction": BLADE_JOINT_FRICTION,
            "blade_joint_dynamic_friction": BLADE_JOINT_DYNAMIC_FRICTION,
            "initial_joint_state_owner": "reset_boundary_articulation_tensor",
        }

    def _set_blade_drive_stiffness_scale(self, scale: float) -> None:
        """Set the passive hold stiffness without writing joint position."""

        from pxr import UsdPhysics

        if not self._as013_joint_prim_path:
            raise RuntimeError("AS013 blade joint path is unavailable")
        joint_prim = self._stage.GetPrimAtPath(self._as013_joint_prim_path)
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "linear")
        if not drive:
            raise RuntimeError("AS013 blade linear drive is unavailable")
        drive.GetStiffnessAttr().Set(
            BLADE_JOINT_DRIVE_STIFFNESS * float(scale)
        )

    def _set_blade_drive_damping_scale(self, scale: float) -> None:
        """Set passive blade-drive damping without writing joint position."""

        from pxr import UsdPhysics

        if not self._as013_joint_prim_path:
            raise RuntimeError("AS013 blade joint path is unavailable")
        joint_prim = self._stage.GetPrimAtPath(self._as013_joint_prim_path)
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "linear")
        if not drive:
            raise RuntimeError("AS013 blade linear drive is unavailable")
        drive.GetDampingAttr().Set(BLADE_JOINT_DRIVE_DAMPING * float(scale))

    def _set_blade_joint_friction_scale(self, scale: float) -> None:
        """Set authored passive joint friction without writing joint state."""

        if not self._as013_joint_prim_path:
            raise RuntimeError("AS013 blade joint path is unavailable")
        joint_prim = self._stage.GetPrimAtPath(self._as013_joint_prim_path)
        for name, value in (
            ("physxJoint:jointFriction", BLADE_JOINT_FRICTION),
            ("physxJoint:dynamicFriction", BLADE_JOINT_DYNAMIC_FRICTION),
        ):
            attribute = joint_prim.GetAttribute(name)
            if not attribute.IsValid():
                raise RuntimeError(f"AS013 blade joint attribute is missing: {name}")
            attribute.Set(float(value) * float(scale))

    def _write_blade_drive_target(self, target: float) -> float | None:
        """Write one finite, joint-limit-bounded blade position target.

        Callers use the measured joint position as the passive baseline.  An
        optional causal target may lead that baseline only when separately
        gated by ordered physical contact and observed operator motion.  A
        bounded no-op is skipped so a stationary articulation can sleep.
        """

        if not np.isfinite(target):
            return None
        lower, upper = self._as013_joint_limits
        bounded_target = float(np.clip(float(target), lower, upper))
        entry = self._task_dof_views.get(JOINT_NAME)
        if entry is None:
            return None
        previous_target = float(getattr(self, "_as013_drive_target", math.nan))
        if (
            np.isfinite(previous_target)
            and abs(bounded_target - previous_target)
            <= BLADE_DRIVE_TARGET_WRITE_DEADBAND_M
        ):
            return previous_target
        view, index, _initial = entry
        try:
            targets = view.get_dof_positions().clone()
            targets[:, int(index)] = bounded_target
            indices = self._torch.arange(
                int(view.count),
                dtype=self._torch.long,
                device=self._isaac_device,
            )
            view.set_dof_position_targets(targets, indices)
        except Exception as exc:  # noqa: BLE001 - simulator detail.
            self._add_task_state_warning(
                f"AS013 blade-target write failed: {exc}"
            )
            return None
        self._as013_drive_target = bounded_target
        return bounded_target

    def _task_extra_pose_names(self) -> set[str]:
        return {"blade_2"}

    def _hand_body_paths(self, side: str, *, fingertips_only: bool) -> list[str]:
        if self._stage is None:
            return []
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        paths: list[str] = []
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            name = str(prim.GetName())
            if fingertips_only:
                selected = name.startswith(f"{side}_finger") and name.endswith(
                    ("_tip_link", "_link4")
                )
            else:
                selected = name == f"{side}_palm_link" or name.startswith(
                    f"{side}_finger"
                )
            if selected:
                paths.append(str(prim.GetPath()))
        return paths

    def _hand_body_paths_for_roles(
        self,
        side: str,
        roles: set[str] | frozenset[str],
    ) -> list[str]:
        """Resolve exact Wuji rigid bodies for canonical AS013 roles."""

        if self._stage is None:
            return []
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        selected_roles = frozenset(str(role) for role in roles)
        paths: list[str] = []
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if _wuji_hand_body_role(str(prim.GetName()), side) in selected_roles:
                paths.append(str(prim.GetPath()))
        return paths

    def _as013_fingertip_points(self, side: str) -> tuple[np.ndarray, ...]:
        """Return live centers for the same tip links used by contact views."""

        robot = self._robots.get(side)
        if robot is None:
            return ()
        points: list[np.ndarray] = []
        try:
            body_poses = robot.data.body_pose_w[0]
            for index, name in enumerate(robot.body_names):
                normalized = str(name).lower()
                if not (
                    normalized.startswith(f"{side}_finger")
                    and normalized.endswith(("_tip_link", "_link4"))
                ):
                    continue
                point = body_poses[int(index), :3]
                if hasattr(point, "detach"):
                    point = point.detach().cpu().numpy()
                point_np = np.asarray(point, dtype=np.float64).reshape(3)
                if np.isfinite(point_np).all():
                    points.append(point_np)
        except Exception:
            return ()
        return tuple(points)

    def _as013_named_hand_body_points(
        self,
        side: str,
    ) -> dict[str, np.ndarray]:
        """Return live palm/finger rigid-body centers for diagnostics."""

        robot = self._robots.get(side)
        if robot is None:
            return {}
        points: dict[str, np.ndarray] = {}
        try:
            body_poses = robot.data.body_pose_w[0]
            for index, name in enumerate(robot.body_names):
                normalized = str(name).lower()
                if not (
                    normalized == f"{side}_palm_link"
                    or normalized.startswith(f"{side}_finger")
                ):
                    continue
                point = body_poses[int(index), :3]
                if hasattr(point, "detach"):
                    point = point.detach().cpu().numpy()
                point_np = np.asarray(point, dtype=np.float64).reshape(3)
                if np.isfinite(point_np).all():
                    points[str(name)] = point_np
        except Exception:
            return {}
        return points

    def _as013_contact_filter_counts(
        self,
        view: Any | None,
        filter_paths: list[str] | tuple[str, ...],
        *,
        with_validity: bool = False,
    ) -> dict[str, int] | tuple[dict[str, int], bool]:
        """Read current contact-row counts for diagnostic filter owners."""

        def result(
            counts_by_path: dict[str, int],
            valid: bool,
        ) -> dict[str, int] | tuple[dict[str, int], bool]:
            return (
                (counts_by_path, bool(valid))
                if with_validity
                else counts_by_path
            )

        if view is None:
            return result({}, False)
        try:
            contact_data = view.get_contact_data(self._sim_dt)
            counts = contact_data[4]
            starts = contact_data[5]
            if hasattr(counts, "detach"):
                counts = counts.detach().cpu().numpy()
            if hasattr(starts, "detach"):
                starts = starts.detach().cpu().numpy()
            sensor_count = int(view.sensor_count)
            filter_count = int(view.filter_count)
            if sensor_count < 1 or filter_count != len(filter_paths):
                return result({}, False)
            expected_size = sensor_count * filter_count
            count_values = np.asarray(counts, dtype=np.float64).reshape(-1)
            start_values = np.asarray(starts, dtype=np.float64).reshape(-1)
            capacity = int(view.max_contact_data_count)
            if (
                count_values.size != expected_size
                or start_values.size != expected_size
                or capacity < 1
                or not np.isfinite(count_values).all()
                or not np.isfinite(start_values).all()
                or np.any(count_values < 0.0)
                or np.any(start_values < 0.0)
                or not np.array_equal(count_values, np.floor(count_values))
                or not np.array_equal(start_values, np.floor(start_values))
                or np.any(count_values > capacity)
                or np.any(start_values > capacity)
                or np.any(start_values + count_values > capacity)
                or float(np.sum(count_values)) >= float(capacity)
            ):
                return result({}, False)
            mapped = _contact_filter_counts_by_path(
                count_values.astype(np.uint32),
                sensor_count,
                filter_paths,
            )
            return result(mapped, len(mapped) == filter_count)
        except Exception:
            return result({}, False)

    def _as013_fingertip_point(
        self,
        side: str,
        finger_number: int,
    ) -> np.ndarray | None:
        """Return one named live fingertip center without finger switching."""

        robot = self._robots.get(side)
        if robot is None:
            return None
        expected = f"{side}_finger{int(finger_number)}_link4"
        try:
            body_poses = robot.data.body_pose_w[0]
            for index, name in enumerate(robot.body_names):
                if str(name).lower() != expected:
                    continue
                point = body_poses[int(index), :3]
                if hasattr(point, "detach"):
                    point = point.detach().cpu().numpy()
                point_np = np.asarray(point, dtype=np.float64).reshape(3)
                return point_np if np.isfinite(point_np).all() else None
        except Exception:
            return None
        return None

    def _as013_select_right_slider_fingertip(
        self,
    ) -> tuple[int | None, np.ndarray | None]:
        """Return the live canonical slider fingertip for causal diagnostics.

        The sealed helper name is retained for replay compatibility.  Schema
        v4 selects the physical-left distal owner with the most detailed
        contact rows (falling back to finger1); the optional causal drive is
        disabled by default and never changes which contact rows count.
        """

        finger_number = 1
        if ROLE_SCHEMA_VERSION >= 4:
            counts = self._as013_contact_filter_counts(
                self._as013_left_slider_moving_contact_view,
                self._as013_left_slider_fingertip_filter_paths,
            )
            contacted = [
                (int(count), int(path.rsplit("finger", 1)[1].split("_", 1)[0]))
                for path, count in counts.items()
                if int(count) > 0 and "finger" in path
            ]
            if contacted:
                _count, finger_number = max(contacted)
        fingertip = self._as013_fingertip_point(
            SLIDER_HAND_SIDE, finger_number
        )
        return (
            (finger_number, fingertip.copy())
            if fingertip is not None
            else (None, None)
        )

    def _as013_try_anchor_causal_drive(self, position: float) -> None:
        """Anchor the optional drive to ordered live contact exactly once."""

        if (
            not getattr(self, "_as013_causal_drive_enabled", False)
            or self._as013_causal_drive_contact_joint_position is not None
            or not np.isfinite(position)
        ):
            return
        body = self._physics_body_pose("E_body_1")
        finger_number, fingertip = self._as013_select_right_slider_fingertip()
        command_pose = (
            self._left_tcp_pose
            if SLIDER_HAND_SIDE == "left"
            else self._right_tcp_pose
        )
        command = np.asarray(command_pose, dtype=np.float64).reshape(7)
        if body is None or finger_number is None or fingertip is None:
            return
        fingertip_coordinate = _body_relative_axis_coordinate(
            fingertip,
            body,
            self._as013_joint_axis_body_local,
        )
        command_coordinate = _body_relative_axis_coordinate(
            command[:3],
            body,
            self._as013_joint_axis_body_local,
        )
        if not np.isfinite([fingertip_coordinate, command_coordinate]).all():
            return
        self._as013_causal_drive_contact_finger_number = int(finger_number)
        self._as013_causal_drive_contact_joint_position = float(position)
        self._as013_causal_drive_contact_fingertip_axis_coordinate = float(
            fingertip_coordinate
        )
        self._as013_causal_drive_contact_command_axis_coordinate = float(
            command_coordinate
        )

    def _as013_causal_drive_target_now(
        self,
        measured_joint_position: float,
    ) -> tuple[float, float, float, float]:
        """Evaluate the optional target from current physical/task state."""

        finger_number = self._as013_causal_drive_contact_finger_number
        body = self._physics_body_pose("E_body_1")
        fingertip = (
            self._as013_fingertip_point(SLIDER_HAND_SIDE, finger_number)
            if finger_number is not None
            else None
        )
        if body is None or fingertip is None:
            return math.nan, math.nan, math.nan, math.nan
        fingertip_coordinate = _body_relative_axis_coordinate(
            fingertip,
            body,
            self._as013_joint_axis_body_local,
        )
        command_pose = (
            self._left_tcp_pose
            if SLIDER_HAND_SIDE == "left"
            else self._right_tcp_pose
        )
        command_coordinate = _body_relative_axis_coordinate(
            np.asarray(command_pose, dtype=np.float64).reshape(7)[:3],
            body,
            self._as013_joint_axis_body_local,
        )
        return _causal_fingertip_drive_target(
            contact_joint_position=float(
                self._as013_causal_drive_contact_joint_position
            ),
            measured_joint_position=float(measured_joint_position),
            contact_fingertip_axis_coordinate=float(
                self._as013_causal_drive_contact_fingertip_axis_coordinate
            ),
            current_fingertip_axis_coordinate=fingertip_coordinate,
            contact_command_axis_coordinate=float(
                self._as013_causal_drive_contact_command_axis_coordinate
            ),
            current_command_axis_coordinate=command_coordinate,
            previous_forward_displacement_m=(
                self._as013_causal_drive_forward_displacement_m
            ),
            joint_limits=self._as013_joint_limits,
        )

    def _init_task_specific_runtime(self) -> None:
        if not self._robots:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        body = self._stage.GetPrimAtPath(f"{KNIFE_ROOT_PATH}/E_body_1")
        blade = self._stage.GetPrimAtPath(
            f"{KNIFE_ROOT_PATH}/E_link_blade_2"
        )
        if not body.IsValid():
            body = None
        if not blade.IsValid():
            blade = None
        table = self._find_rigid_body_prim("table")
        if (
            table is not None
            and (
                str(table.GetName()) != "table"
            )
        ):
            table = None
        right_palm_paths = self._hand_body_paths_for_roles(
            "right", {"palm"}
        )
        right_non_thumb_paths = self._hand_body_paths_for_roles(
            "right", {"non_thumb"}
        )
        right_thumb_paths = self._hand_body_paths_for_roles(
            "right", {"thumb_distal_tip"}
        )
        right_hand_paths = self._hand_body_paths_for_roles(
            "right",
            {"palm", "thumb_proximal", "thumb_distal_tip", "non_thumb"},
        )
        # The current Wuji tip collider is a child of link4, not a separate
        # rigid body.  Fail closed if a future import omits that actual owner.
        right_thumb_paths = [
            path
            for path in right_thumb_paths
            if path.rsplit("/", 1)[-1].lower()
            == RIGHT_THUMB_DISTAL_RIGID_BODY_NAME
        ]
        right_unsafe_paths = self._hand_body_paths_for_roles(
            "right", {"palm", "non_thumb", "thumb_proximal"}
        )
        left_paths = self._hand_body_paths_for_roles(
            "left",
            {"palm", "thumb_proximal", "thumb_distal_tip", "non_thumb"},
        )
        left_slider_fingertip_paths = [
            path
            for path in left_paths
            if path.rsplit("/", 1)[-1]
            in LEFT_SLIDER_FINGERTIP_RIGID_BODY_NAMES
        ]
        right_grip_paths = list(dict.fromkeys(right_hand_paths))
        if (
            physics_sim_view is None
            or body is None
            or blade is None
            or table is None
            or not right_palm_paths
            or not right_non_thumb_paths
            or not right_thumb_paths
            or not right_hand_paths
            or not right_unsafe_paths
            or not left_slider_fingertip_paths
            or not left_paths
        ):
            message = "AS013 physical contact owners could not all be resolved"
            self._add_task_state_warning(message)
            raise RuntimeError(message)

        def require_exact_hand_owners(
            label: str,
            side: str,
            roles: set[str] | frozenset[str],
            paths: list[str],
        ) -> None:
            expected_names = _expected_wuji_hand_body_names(side, roles)
            actual_paths = tuple(str(path) for path in paths)
            actual_names = tuple(
                path.rsplit("/", 1)[-1] for path in actual_paths
            )
            if (
                len(actual_paths) != len(set(actual_paths))
                or len(actual_names) != len(set(actual_names))
                or frozenset(actual_names) != expected_names
            ):
                raise RuntimeError(
                    "AS013 Wuji contact-owner set mismatch: "
                    f"label={label}, expected={sorted(expected_names)}, "
                    f"actual={sorted(actual_names)}"
                )

        require_exact_hand_owners(
            "right_palm", "right", {"palm"}, right_palm_paths
        )
        require_exact_hand_owners(
            "right_non_thumb",
            "right",
            {"non_thumb"},
            right_non_thumb_paths,
        )
        require_exact_hand_owners(
            "right_thumb_distal",
            "right",
            {"thumb_distal_tip"},
            right_thumb_paths,
        )
        require_exact_hand_owners(
            "right_full_hand",
            "right",
            {"palm", "thumb_proximal", "thumb_distal_tip", "non_thumb"},
            right_hand_paths,
        )
        require_exact_hand_owners(
            "right_unsafe",
            "right",
            {"palm", "non_thumb", "thumb_proximal"},
            right_unsafe_paths,
        )
        require_exact_hand_owners(
            "left_full_hand",
            "left",
            {"palm", "thumb_proximal", "thumb_distal_tip", "non_thumb"},
            left_paths,
        )
        left_slider_names = tuple(
            path.rsplit("/", 1)[-1] for path in left_slider_fingertip_paths
        )
        if (
            len(left_slider_names) != len(set(left_slider_names))
            or frozenset(left_slider_names)
            != frozenset(LEFT_SLIDER_FINGERTIP_RIGID_BODY_NAMES)
        ):
            raise RuntimeError(
                "AS013 left slider fingertip-owner set mismatch: "
                f"expected={sorted(LEFT_SLIDER_FINGERTIP_RIGID_BODY_NAMES)}, "
                f"actual={sorted(left_slider_names)}"
            )

        self._as013_contact_view_contracts = {}

        def contact_view(
            name: str,
            sensor: Any,
            filters: list[str],
        ) -> Any:
            expected_sensor_paths = (str(sensor.GetPath()),)
            expected_filter_paths = tuple(str(path) for path in filters)
            view = physics_sim_view.create_rigid_contact_view(
                expected_sensor_paths[0],
                filter_patterns=filters,
                max_contact_data_count=max(64, len(filters) * 8),
            )
            if (
                view is None
                or int(view.sensor_count) != 1
                or int(view.filter_count) != len(expected_filter_paths)
            ):
                raise RuntimeError(
                    "AS013 contact view layout mismatch: "
                    f"name={name}, sensors={getattr(view, 'sensor_count', 0)}, "
                    f"filters={getattr(view, 'filter_count', 0)}, "
                    f"expected_sensors=1, "
                    f"expected_filters={len(expected_filter_paths)}"
                )
            missing = object()
            raw_sensor_paths = getattr(view, "sensor_paths", missing)
            if raw_sensor_paths is missing:
                raise RuntimeError(
                    "AS013 contact view does not expose exact sensor_paths: "
                    f"name={name}"
                )
            elif isinstance(raw_sensor_paths, (str, bytes)):
                actual_sensor_paths = (str(raw_sensor_paths),)
                sensor_paths_exposed = True
            else:
                sensor_paths_exposed = True
                raw_sensors = list(raw_sensor_paths)
                if raw_sensors and isinstance(
                    raw_sensors[0], (list, tuple)
                ):
                    per_sensor_paths = tuple(
                        tuple(str(path) for path in paths)
                        for paths in raw_sensors
                    )
                    if len(per_sensor_paths) != 1:
                        raise RuntimeError(
                            "AS013 contact view returned a nested sensor "
                            f"layout for {len(per_sensor_paths)} sensors: "
                            f"name={name}"
                        )
                    actual_sensor_paths = per_sensor_paths[0]
                else:
                    actual_sensor_paths = tuple(
                        str(path) for path in raw_sensors
                    )
            raw_filter_paths = getattr(view, "filter_paths", missing)
            if raw_filter_paths is missing:
                raise RuntimeError(
                    "AS013 contact view does not expose exact filter_paths: "
                    f"name={name}"
                )
            else:
                filter_paths_exposed = True
                raw_filters = (
                    [raw_filter_paths]
                    if isinstance(raw_filter_paths, (str, bytes))
                    else list(raw_filter_paths)
                )
                if raw_filters and isinstance(
                    raw_filters[0], (list, tuple)
                ):
                    per_sensor_filters = tuple(
                        tuple(str(path) for path in paths)
                        for paths in raw_filters
                    )
                    if len(per_sensor_filters) != 1:
                        raise RuntimeError(
                            "AS013 contact view returned a nested filter "
                            f"layout for {len(per_sensor_filters)} sensors: "
                            f"name={name}"
                        )
                    actual_filter_paths = per_sensor_filters[0]
                else:
                    actual_filter_paths = tuple(
                        str(path) for path in raw_filters
                    )
            if (
                actual_sensor_paths != expected_sensor_paths
                or actual_filter_paths != expected_filter_paths
            ):
                raise RuntimeError(
                    "AS013 contact view returned different exact paths: "
                    f"name={name}, expected_sensors={expected_sensor_paths}, "
                    f"actual_sensors={actual_sensor_paths}, "
                    f"expected_filters={expected_filter_paths}, "
                    f"actual_filters={actual_filter_paths}"
                )
            checker = getattr(view, "check", None)
            check_exposed = callable(checker)
            if not check_exposed:
                raise RuntimeError(
                    "AS013 contact view does not expose callable check(): "
                    f"name={name}"
                )
            check_passed = bool(checker())
            if not check_passed:
                raise RuntimeError(
                    f"AS013 contact view check() failed: name={name}"
                )
            self._as013_contact_view_contracts[name] = {
                "sensor_count": int(view.sensor_count),
                "filter_count": int(view.filter_count),
                "sensor_paths": list(actual_sensor_paths),
                "filter_paths": list(actual_filter_paths),
                "sensor_paths_exposed": sensor_paths_exposed,
                "filter_paths_exposed": filter_paths_exposed,
                "check_exposed": check_exposed,
                "check_passed": check_passed,
            }
            return view

        try:
            self._as013_right_grip_housing_contact_view = contact_view(
                "right_grip_housing", body, right_grip_paths
            )
            self._as013_right_grip_filter_paths = tuple(right_grip_paths)
            self._as013_right_palm_housing_contact_view = contact_view(
                "right_palm_housing", body, right_palm_paths
            )
            self._as013_right_palm_filter_paths = tuple(right_palm_paths)
            self._as013_right_non_thumb_housing_contact_view = contact_view(
                "right_non_thumb_housing", body, right_non_thumb_paths
            )
            self._as013_right_non_thumb_filter_paths = tuple(
                right_non_thumb_paths
            )
            self._as013_left_slider_moving_contact_view = contact_view(
                "left_slider_moving", blade, left_slider_fingertip_paths
            )
            self._as013_left_slider_fingertip_filter_paths = tuple(
                left_slider_fingertip_paths
            )
            self._as013_right_unsafe_moving_contact_view = contact_view(
                "right_unsafe_moving", blade, right_unsafe_paths
            )
            self._as013_right_unsafe_filter_paths = tuple(right_unsafe_paths)
            self._as013_left_housing_contact_view = contact_view(
                "left_housing", body, left_paths
            )
            self._as013_left_moving_contact_view = contact_view(
                "left_moving", blade, left_paths
            )
            self._as013_left_release_filter_paths = tuple(left_paths)
            self._as013_body_table_contact_view = contact_view(
                "housing_table", body, [str(table.GetPath())]
            )
            self._as013_blade_table_contact_view = contact_view(
                "moving_table", blade, [str(table.GetPath())]
            )

            # Stable legacy aliases remain readable, but all canonical logic
            # above is bound to explicit physical owners.
            self._as013_left_body_contact_view = (
                self._as013_right_grip_housing_contact_view
            )
            self._as013_left_contact_filter_paths = (
                self._as013_right_grip_filter_paths
            )
            self._as013_right_blade_contact_view = (
                self._as013_left_slider_moving_contact_view
            )
            self._as013_right_contact_filter_paths = (
                self._as013_left_slider_fingertip_filter_paths
            )
            self._as013_left_blade_contact_view = (
                self._as013_right_unsafe_moving_contact_view
            )
            self._as013_left_housing_release_contact_view = (
                self._as013_left_housing_contact_view
            )
            self._as013_left_moving_release_contact_view = (
                self._as013_left_moving_contact_view
            )
            self._as013_right_body_contact_view = (
                self._as013_right_grip_housing_contact_view
            )
            self._as013_right_stabilizer_filter_paths = (
                self._as013_right_grip_filter_paths
            )
            self._as013_left_slider_blade_contact_view = (
                self._as013_left_slider_moving_contact_view
            )
            self._as013_left_slider_filter_paths = (
                self._as013_left_slider_fingertip_filter_paths
            )
            self._as013_right_unsafe_blade_contact_view = (
                self._as013_right_unsafe_moving_contact_view
            )
            if set(self._as013_contact_view_contracts) != _AS013_CONTACT_VIEW_NAMES:
                raise RuntimeError(
                    "AS013 exact contact-view contract names differ: "
                    f"expected={sorted(_AS013_CONTACT_VIEW_NAMES)}, "
                    f"actual={sorted(self._as013_contact_view_contracts)}"
                )
            if hasattr(self, "_as013_spawn_contract"):
                self._as013_spawn_contract[
                    "resolved_contact_views"
                ] = {
                    key: dict(value)
                    for key, value in self._as013_contact_view_contracts.items()
                }
            self._as013_contact_views_ready = True
            print(
                "[HandsOn/AS013] Nine exact contact views ready: "
                + json.dumps(
                    self._as013_contact_view_contracts,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - PhysX tensor APIs vary by release.
            self._as013_contact_views_ready = False
            message = f"AS013 physical contact views are unavailable: {exc}"
            self._add_task_state_warning(message)
            raise RuntimeError(message) from exc

    def _validate_task_initial_state_before_ready(
        self,
    ) -> Mapping[str, Any] | None:
        """Fail closed before AS013 can advertise collection readiness."""

        parent_validation = super()._validate_task_initial_state_before_ready()
        contracts = getattr(self, "_as013_contact_view_contracts", {})
        if (
            not getattr(self, "_as013_contact_views_ready", False)
            or set(contracts) != _AS013_CONTACT_VIEW_NAMES
        ):
            raise RuntimeError(
                "AS013 cannot become collection-ready without all nine exact "
                "physical contact views"
            )
        invalid_contracts = {
            name: dict(contract)
            for name, contract in contracts.items()
            if int(contract.get("sensor_count", 0)) != 1
            or int(contract.get("filter_count", 0))
            != _AS013_CONTACT_VIEW_FILTER_COUNTS[name]
            or len(contract.get("sensor_paths", ())) != 1
            or len(contract.get("filter_paths", ()))
            != _AS013_CONTACT_VIEW_FILTER_COUNTS[name]
            or not bool(contract.get("sensor_paths_exposed", False))
            or not bool(contract.get("filter_paths_exposed", False))
            or not bool(contract.get("check_exposed", False))
            or not bool(contract.get("check_passed", False))
        }
        if invalid_contracts:
            raise RuntimeError(
                "AS013 cannot become collection-ready with invalid contact-view "
                f"contracts: {invalid_contracts}"
            )
        self._as013_contact_forces()
        if not getattr(self, "_as013_contact_report_valid", False):
            raise RuntimeError(
                "AS013 cannot become collection-ready without a valid physical "
                "contact report readback"
            )
        position, progress, velocity = self._as013_joint_state()
        lower, upper = self._as013_joint_limits

        def finite_vector(value: Any, minimum_size: int) -> bool:
            try:
                values = np.asarray(value, dtype=np.float64).reshape(-1)
            except Exception:  # noqa: BLE001 - malformed truth fails closed.
                return False
            return bool(
                values.size >= minimum_size
                and np.isfinite(values[:minimum_size]).all()
            )

        housing_pose = self._physics_body_pose("E_body_1")
        moving_pose = self._physics_body_pose("blade_2")
        housing_velocity = self._physics_body_velocity("E_body_1")
        moving_velocity = self._physics_body_velocity("blade_2")
        joint_truth_valid = bool(
            np.isfinite([position, progress, velocity]).all()
            and lower - 1.0e-4 <= position <= upper + 1.0e-4
            and -1.0e-6 <= progress <= 1.0 + 1.0e-6
        )
        physical_truth_valid = bool(
            joint_truth_valid
            and finite_vector(housing_pose, 7)
            and finite_vector(moving_pose, 7)
            and finite_vector(housing_velocity, 6)
            and finite_vector(moving_velocity, 6)
        )
        if not physical_truth_valid:
            raise RuntimeError(
                "AS013 cannot become collection-ready without finite joint, "
                "housing, and moving-component pose/velocity truth: "
                f"joint_position={position}, joint_progress={progress}, "
                f"joint_velocity={velocity}, joint_truth_valid={joint_truth_valid}"
            )
        payload = (
            dict(parent_validation)
            if isinstance(parent_validation, Mapping)
            else {}
        )
        payload.update(
            {
                "as013_contact_views_ready": True,
                "as013_contact_view_count": len(contracts),
                "as013_contact_report_valid": True,
                "as013_joint_truth_valid": True,
                "as013_task_body_truth_valid": True,
            }
        )
        return payload

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self._as013_reset_progress = 0.80
        self._as013_target_direction = TARGET_DIRECTION_RETRACT
        self._as013_target_endpoint = TARGET_ENDPOINT_RETRACTED
        self._as013_target_progress = 0.0
        self._as013_target_endpoint_reached_now = False
        self._as013_drive_assist_active = False
        if (
            self._as013_joint_prim_path
            and self._as013_blade_drive_stiffness_scaled
        ):
            self._set_blade_drive_stiffness_scale(1.0)
        if (
            self._as013_joint_prim_path
            and self._as013_blade_drive_damping_scaled
        ):
            self._set_blade_drive_damping_scale(1.0)
        if (
            self._as013_joint_prim_path
            and self._as013_blade_joint_friction_scaled
        ):
            self._set_blade_joint_friction_scale(1.0)
        self._as013_blade_drive_stiffness_scaled = False
        self._as013_blade_drive_damping_scaled = False
        self._as013_blade_joint_friction_scaled = False
        self._as013_causal_drive_contact_finger_number = None
        self._as013_causal_drive_contact_joint_position = None
        self._as013_causal_drive_contact_fingertip_axis_coordinate = None
        self._as013_causal_drive_contact_command_axis_coordinate = None
        self._as013_causal_drive_forward_displacement_m = 0.0
        self._as013_causal_drive_fingertip_forward_m = 0.0
        self._as013_causal_drive_command_forward_m = 0.0
        self._as013_right_grip_contact_now = False
        self._as013_right_grip_contact_owner_count = 0
        self._as013_right_grip_seen = False
        self._as013_pickup_candidate_substeps = 0
        self._as013_pickup_seen = False
        self._as013_pickup_lift_m = 0.0
        self._as013_pickup_peak_lift_m = 0.0
        self._as013_knife_lifted_now = False
        self._as013_drop_substeps = 0
        self._as013_drop_failure = False
        self._as013_drop_near_table_now = False
        self._as013_drop_peak_descent_now = False
        self._as013_drop_falling_now = False
        self._as013_drop_table_impact_now = False
        self._as013_drop_evidence_now = False
        self._as013_table_contact_substeps = 0
        self._as013_table_contact_event_active = False
        self._as013_table_contact_event_count = 0
        self._as013_second_table_contact_seen = False
        self._as013_slider_contact_observation_substeps = 0
        self._as013_slider_contact_dropout_substeps = 0
        self._as013_slider_contact_effective_now = False
        self._as013_body_pose_report_valid = False
        self._as013_body_velocity_report_valid = False
        self._as013_body_vertical_velocity_mps = math.nan
        self._as013_left_hand_participation_substeps = 0
        self._as013_left_hand_participation_failure = False
        self._as013_reset_body_height_m = math.nan
        self._as013_recording_origin_accepted = False
        self._as013_left_stabilization_substeps = 0
        self._as013_left_stable_contact_now = False
        self._as013_left_stabilized_seen = False
        self._as013_right_slider_contact_substeps = 0
        self._as013_right_slider_contact_seen = False
        self._as013_preorder_slider_contact_substeps = 0
        self._as013_prestabilization_progress_substeps = 0
        self._as013_unsafe_blade_contact_substeps = 0
        self._as013_ordered_contact_progress = None
        self._as013_slide_progress_substeps = 0
        self._as013_slide_progress_now = False
        self._as013_slide_progress_seen = False
        self._as013_previous_blade_progress = math.nan
        self._as013_completion_forward_progress_now = False
        self._as013_completion_target_progress_now = False
        self._as013_ordered_slide_completion_seen = False
        self._as013_wrong_order_failure = False
        self._as013_unsafe_blade_contact_failure = False
        self._as013_nonfinite_failure = False
        self._as013_displacement_failure = False
        self._as013_failure_reason = ""
        self._as013_failure_reported = False
        self._as013_contact_report_valid = False
        self._as013_right_contact_rows_valid = False
        self._as013_right_contact_row_count = 0
        self._as013_right_slider_contact_row_count = 0
        self._as013_right_outside_slider_contact_row_count = 0
        self._as013_right_thumb_net_contact_force = 0.0
        self._as013_right_thumb_outside_slider_contact_force = 0.0
        self._as013_left_contact_rows_valid = False
        self._as013_left_blade_contact_row_count = 0
        self._as013_left_slider_contact_row_count = 0
        self._as013_left_housing_release_contact_force = math.inf
        self._as013_left_moving_release_contact_force = math.inf
        self._as013_left_release_contact_report_valid = False
        self._as013_start_body_pose = None
        self._as013_previous_right_command = self._right_tcp_pose.copy()
        self._as013_previous_slider_command = (
            self._left_tcp_pose.copy()
            if SLIDER_HAND_SIDE == "left"
            else self._right_tcp_pose.copy()
        )
        self._as013_right_slide_command = 0.0
        self._as013_scripted_left_hold_pose = None
        self._as013_scripted_left_nominal_world_rotation = None
        self._as013_scripted_left_hold_hand = None
        self._as013_scripted_left_hand_preload_start_frame = None
        self._as013_scripted_left_counter_slide_anchor_pose = None
        self._as013_scripted_left_hold_fingertip_index = None
        self._as013_scripted_left_stabilizer_finger_number = (
            LEFT_SAFE_STABILIZER_FINGER_NUMBER
        )
        self._as013_scripted_left_surface_target_local = None
        self._as013_scripted_left_surface_preload_steps = 0
        self._as013_scripted_left_surface_last_preload_frame = None
        self._as013_reset_yaw_degrees = 0.0
        self._as013_scripted_left_contact_standoff_m = None
        self._as013_scripted_left_recovery_body_relative = None
        self._as013_scripted_left_recovery_fingertip_index = None
        self._as013_scripted_left_recovery_fingertip_body_local = None
        self._as013_scripted_left_recovery_fingertip_target_world = None
        self._as013_scripted_left_fixed_anchor_offset_world = np.zeros(
            3,
            dtype=np.float64,
        )
        self._as013_scripted_left_fixed_anchor_slide_preloaded = False
        self._as013_scripted_left_fallback_grip_active = False
        self._as013_scripted_right_contact_pose = None
        self._as013_scripted_right_contact_hand = None
        self._as013_scripted_right_contact_frame = None
        self._as013_scripted_right_finger_number = None
        self._as013_scripted_right_approach_blade_relative = None
        self._as013_scripted_right_alignment_rotation_blade_relative = None
        self._as013_scripted_right_servo_stage = "vertical_entry"
        self._as013_scripted_right_recontact_lifted = False
        self._as013_scripted_right_recontact_target_blade_local = None
        self._as013_scripted_left_recovery_ready_frames = 0
        self._as013_scripted_right_target_blade_local = None
        self._as013_scripted_timeout_right_pose = None
        self._as013_scripted_timeout_left_pose = None
        self._as013_scripted_timeout_right_hand = None
        self._as013_scripted_timeout_left_hand = None
        self._as013_scripted_slide_control_frames = 0
        self._as013_scripted_right_slide_reanchor_origin_progress = None
        self._as013_scripted_right_slide_reanchor_frame = None
        self._as013_scripted_right_slide_cross_edge_correction_m = 0.0
        self._as013_scripted_right_contact_settle_frames = 0
        self._as013_scripted_right_recontact_grace_frames = 0
        self._as013_scripted_right_initial_contact_dropout_frames = 0
        self._as013_scripted_right_last_proven_contact_blade_local = None
        self._as013_scripted_late_left_stability_grace_frames = 0
        self._as013_scripted_right_recovered_target_blade_local = None
        self._as013_scripted_right_recovered_hand_blend = 0.0
        self._as013_scripted_right_late_continuous_hand_blend = 0.0
        self._as013_scripted_right_recovered_target_blend = 0.0
        self._as013_scripted_slide_complete_frame = None
        self._as013_scripted_release_frame = None
        self._as013_scripted_single_right_home = None
        self._as013_scripted_single_left_home = None
        self._as013_scripted_single_grip_pose = None
        self._as013_scripted_single_grip_frame = None
        self._as013_scripted_single_pickup_pose = None
        self._as013_scripted_single_pickup_frame = None
        self._as013_scripted_single_slide_hand = None
        self._as013_scripted_bimanual_left_start_pose = None
        self._as013_scripted_bimanual_left_target_pose = None
        self._as013_scripted_bimanual_left_start_frame = None
        self._as013_final_hold_valid_seen = False
        self._as013_final_hold_dropout_control_frames = 0
        self._as013_final_hold_grace_active = False
        self._as013_final_hold_with_grace_now = False
        self._as013_last_metrics = {}

    def _clear_as013_v3_episode_semantics(self) -> None:
        """Clear schema-v3+ history without touching physics/view readiness."""

        self._as013_drive_assist_active = False
        self._as013_causal_drive_contact_finger_number = None
        self._as013_causal_drive_contact_joint_position = None
        self._as013_causal_drive_contact_fingertip_axis_coordinate = None
        self._as013_causal_drive_contact_command_axis_coordinate = None
        self._as013_causal_drive_forward_displacement_m = 0.0
        self._as013_causal_drive_fingertip_forward_m = 0.0
        self._as013_causal_drive_command_forward_m = 0.0
        self._as013_right_grip_contact_now = False
        self._as013_right_grip_contact_owner_count = 0
        self._as013_right_grip_seen = False
        self._as013_pickup_candidate_substeps = 0
        self._as013_pickup_seen = False
        self._as013_pickup_lift_m = 0.0
        self._as013_pickup_peak_lift_m = 0.0
        self._as013_knife_lifted_now = False
        self._as013_drop_substeps = 0
        self._as013_drop_failure = False
        self._as013_drop_near_table_now = False
        self._as013_drop_peak_descent_now = False
        self._as013_drop_falling_now = False
        self._as013_drop_table_impact_now = False
        self._as013_drop_evidence_now = False
        self._as013_table_contact_substeps = 0
        self._as013_table_contact_event_active = False
        self._as013_table_contact_event_count = 0
        self._as013_second_table_contact_seen = False
        self._as013_slider_contact_observation_substeps = 0
        self._as013_slider_contact_dropout_substeps = 0
        self._as013_slider_contact_effective_now = False
        self._as013_body_pose_report_valid = False
        self._as013_body_velocity_report_valid = False
        self._as013_body_vertical_velocity_mps = math.nan
        self._as013_left_hand_participation_substeps = 0
        self._as013_left_hand_participation_failure = False
        self._as013_left_stabilization_substeps = 0
        self._as013_left_stable_contact_now = False
        self._as013_left_stabilized_seen = False
        self._as013_right_slider_contact_substeps = 0
        self._as013_right_slider_contact_seen = False
        self._as013_preorder_slider_contact_substeps = 0
        self._as013_prestabilization_progress_substeps = 0
        self._as013_unsafe_blade_contact_substeps = 0
        self._as013_ordered_contact_progress = None
        self._as013_slide_progress_substeps = 0
        self._as013_slide_progress_now = False
        self._as013_slide_progress_seen = False
        self._as013_completion_forward_progress_now = False
        self._as013_completion_target_progress_now = False
        self._as013_target_endpoint_reached_now = False
        self._as013_ordered_slide_completion_seen = False
        self._as013_wrong_order_failure = False
        self._as013_unsafe_blade_contact_failure = False
        self._as013_nonfinite_failure = False
        self._as013_displacement_failure = False
        self._as013_failure_reason = ""
        self._as013_failure_reported = False
        self._as013_right_contact_rows_valid = False
        self._as013_right_contact_row_count = 0
        self._as013_right_slider_contact_row_count = 0
        self._as013_right_outside_slider_contact_row_count = 0
        self._as013_right_thumb_net_contact_force = 0.0
        self._as013_right_thumb_outside_slider_contact_force = 0.0
        self._as013_left_contact_rows_valid = False
        self._as013_left_blade_contact_row_count = 0
        self._as013_left_slider_contact_row_count = 0
        self._as013_right_slide_command = 0.0
        self._as013_previous_slider_command = np.asarray(
            self._left_tcp_pose
            if SLIDER_HAND_SIDE == "left"
            else self._right_tcp_pose,
            dtype=np.float64,
        ).reshape(7).copy()
        self._as013_scripted_single_right_home = None
        self._as013_scripted_single_left_home = None
        self._as013_scripted_single_grip_pose = None
        self._as013_scripted_single_grip_frame = None
        self._as013_scripted_single_pickup_pose = None
        self._as013_scripted_single_pickup_frame = None
        self._as013_scripted_single_slide_hand = None
        self._as013_scripted_bimanual_left_start_pose = None
        self._as013_scripted_bimanual_left_target_pose = None
        self._as013_scripted_bimanual_left_start_frame = None
        self._as013_final_hold_valid_seen = False
        self._as013_final_hold_dropout_control_frames = 0
        self._as013_final_hold_grace_active = False
        self._as013_final_hold_with_grace_now = False
        self._success_counter = 0
        self._as013_last_metrics = {}

    def accept_stable_recording_origin(
        self,
        report: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Rebase current semantics to the accepted write-free physical origin."""

        from .recording_origin import RecordingOriginStabilityError

        if int(self.env_step) != 0:
            raise RecordingOriginStabilityError(
                "AS013 recording-origin acceptance ran after action zero.",
                diagnostics={
                    "failure_phase": "as013_origin_acceptance",
                    "env_step": int(self.env_step),
                },
            )
        position, progress, _velocity = self._as013_joint_state()
        body_pose = self._physics_body_pose("E_body_1")
        try:
            body_values = np.asarray(body_pose, dtype=np.float64).reshape(-1)
        except Exception:  # noqa: BLE001 - malformed physics readback.
            body_values = np.zeros(0, dtype=np.float64)
        joint_valid = bool(
            np.isfinite([position, progress]).all()
            and self._as013_joint_limits[0] - 1.0e-4
            <= position
            <= self._as013_joint_limits[1] + 1.0e-4
            and -1.0e-6 <= progress <= 1.0 + 1.0e-6
        )
        pose_valid = bool(
            body_values.size >= 7 and np.isfinite(body_values[:7]).all()
        )
        if not joint_valid or not pose_valid:
            raise RecordingOriginStabilityError(
                "AS013 accepted origin lacks finite settled joint/body truth.",
                diagnostics={
                    "failure_phase": "as013_origin_truth",
                    "joint_position": float(position),
                    "joint_progress": float(progress),
                    "joint_truth_valid": joint_valid,
                    "body_pose_truth_valid": pose_valid,
                },
            )
        if not self._as013_contact_views_ready:
            raise RecordingOriginStabilityError(
                "AS013 accepted origin lacks its physical contact views.",
                diagnostics={
                    "failure_phase": "as013_origin_contacts",
                    "contact_views_ready": False,
                },
            )
        # Read-only contact sampling keeps the post-accept observation honest;
        # it does not advance any task latch or write a physics target.
        self._as013_contact_forces()
        if not self._as013_contact_report_valid:
            raise RecordingOriginStabilityError(
                "AS013 accepted origin has an invalid contact report.",
                diagnostics={
                    "failure_phase": "as013_origin_contacts",
                    "contact_views_ready": True,
                    "contact_report_valid": False,
                },
            )

        parent_acceptance = super().accept_stable_recording_origin(report)
        payload = (
            dict(parent_acceptance)
            if isinstance(parent_acceptance, Mapping)
            else {}
        )
        self._clear_as013_v3_episode_semantics()
        settled_body_pose = body_values[:7].copy()
        self._as013_reset_progress = float(progress)
        self._as013_previous_blade_progress = float(progress)
        self._as013_drive_target = float(position)
        self._as013_reset_body_height_m = float(settled_body_pose[2])
        self._as013_start_body_pose = settled_body_pose.copy()
        self._as013_body_pose_report_valid = True
        self._as013_previous_right_command = np.asarray(
            self._right_tcp_pose, dtype=np.float64
        ).reshape(7).copy()
        self._as013_recording_origin_accepted = True
        payload.update(
            {
                "accepted": True,
                "kind": "as013_v3_settled_physical_boundary",
                "physics_state_writes": False,
                "joint_position": float(position),
                "joint_progress": float(progress),
                "target_direction": int(self._as013_target_direction),
                "target_endpoint": str(self._as013_target_endpoint),
                "target_progress": float(self._as013_target_progress),
                "baseline_body_pose_wxyz": settled_body_pose.tolist(),
                "contact_views_ready": True,
                "contact_report_valid": True,
                "episode_semantics_cleared": True,
                "resolved_contact_views": {
                    key: dict(value)
                    for key, value in self._as013_contact_view_contracts.items()
                },
            }
        )
        return payload

    def _apply_task_pose_randomization(self) -> None:
        dof_entry = self._task_dof_views.get(JOINT_NAME)
        if dof_entry is None:
            self._add_task_state_warning(
                f"AS013 live articulation did not expose exact DOF {JOINT_NAME}"
            )
            return
        dof_view, dof_index, _captured_initial = dof_entry
        articulation_state = next(
            (
                state
                for state in self._task_reset_articulation_states
                if state[1] is dof_view
            ),
            None,
        )
        if articulation_state is None:
            self._add_task_state_warning(
                "AS013 blade DOF is not owned by a captured live articulation root"
            )
            return

        (
            root_path,
            view,
            indices,
            root_pose,
            root_velocity,
            joint_positions,
            joint_velocities,
        ) = articulation_state
        hand_points = (
            *self._reset_robot_body_points("right"),
            *self._reset_robot_body_points("left"),
        )
        center_xy = root_pose[0, :2].detach().cpu().numpy()
        offset, yaw_degrees, progress, target_direction = sample_reset_contract(
            self._random,
            randomize=bool(self.randomize),
            center_xy=center_xy,
            hand_points=hand_points,
        )
        self._as013_reset_yaw_degrees = float(yaw_degrees)
        self._as013_target_direction = int(target_direction)
        self._as013_target_endpoint = _as013_target_endpoint(target_direction)
        self._as013_target_progress = _as013_target_progress(target_direction)
        self._as013_target_endpoint_reached_now = False
        randomized_pose_np = _randomized_physx_pose(
            root_pose.detach().cpu().numpy(),
            xy_offset=offset,
            z_offset=0.0,
            yaw_degrees=yaw_degrees,
        )
        randomized_pose = self._torch.as_tensor(
            randomized_pose_np,
            dtype=root_pose.dtype,
            device=root_pose.device,
        )
        reset_joint_positions = joint_positions.clone()
        reset_joint_velocities = self._torch.zeros_like(joint_velocities)
        lower, upper = self._as013_joint_limits
        reset_position = lower + progress * (upper - lower)
        reset_joint_positions[:, int(dof_index)] = reset_position
        view.set_root_transforms(randomized_pose, indices)
        view.set_root_velocities(self._torch.zeros_like(root_velocity), indices)
        view.set_dof_positions(reset_joint_positions, indices)
        view.set_dof_velocities(reset_joint_velocities, indices)
        view.set_dof_position_targets(reset_joint_positions, indices)
        self._record_authored_task_reset_write(
            root_path,
            transform_xyzw=randomized_pose,
            velocity=self._torch.zeros_like(root_velocity),
            dof_position=reset_joint_positions,
            dof_velocity=reset_joint_velocities,
        )
        self._task_dof_views[JOINT_NAME] = (
            dof_view,
            int(dof_index),
            float(reset_position),
        )
        self._as013_reset_progress = float(progress)
        self._as013_drive_target = float(reset_position)
        self._as013_reset_body_height_m = float(randomized_pose_np[0, 2])

    def _postprocess_hand_action(self) -> None:
        # Historical state names remain aliases; schema v5 sources slider
        # intent from the physical left operator hand while the right hand
        # remains the stabilizer/pickup owner.
        command_pose = (
            self._left_tcp_pose
            if SLIDER_HAND_SIDE == "left"
            else self._right_tcp_pose
        )
        current = np.asarray(command_pose, dtype=np.float64).reshape(7)
        previous = self._as013_previous_slider_command
        if previous is None:
            self._as013_right_slide_command = 0.0
        else:
            delta = current[:3] - np.asarray(previous, dtype=np.float64)[:3]
            self._as013_right_slide_command = float(
                np.dot(delta, self._as013_joint_world_axis())
            )
        self._as013_previous_slider_command = current.copy()
        self._as013_previous_right_command = np.asarray(
            self._right_tcp_pose, dtype=np.float64
        ).reshape(7).copy()

    def _as013_latch_failure(self, reason: str) -> None:
        """Keep the first physical task failure until the next reset."""

        normalized = str(reason).strip()
        if normalized and not self._as013_failure_reason:
            self._as013_failure_reason = normalized

    def _as013_joint_world_axis(self) -> np.ndarray:
        body = self._physics_body_pose("E_body_1")
        if body is None:
            body = getattr(self, "_poses", {}).get("E_body_1")
        axis_body = np.asarray(
            self._as013_joint_axis_body_local, dtype=np.float64
        ).reshape(3)
        axis = axis_body
        if body is not None:
            body_pose = np.asarray(body, dtype=np.float64).reshape(-1)
            if body_pose.size >= 7 and np.isfinite(body_pose[:7]).all():
                axis = (
                    _pose_vector_to_matrix(body_pose[:7])[:3, :3]
                    @ axis_body
                )
        norm = float(np.linalg.norm(axis))
        if np.isfinite(axis).all() and norm > 1.0e-6:
            return axis / norm
        return JOINT_AXIS_BODY_LOCAL.copy()

    def _as013_joint_state(self) -> tuple[float, float, float]:
        reading = self._physics_dof_position(JOINT_NAME)
        if reading is None:
            return math.nan, math.nan, math.nan
        position = float(reading[0])
        lower, upper = self._as013_joint_limits
        progress = (position - lower) / max(upper - lower, 1.0e-12)
        velocity = math.nan
        entry = self._task_dof_views.get(JOINT_NAME)
        if entry is not None:
            view, index, _initial = entry
            try:
                value = view.get_dof_velocities()[0, int(index)]
                if hasattr(value, "detach"):
                    value = value.detach().cpu().item()
                velocity = float(value)
            except Exception:  # noqa: BLE001,S110 - unavailable readback stays NaN.
                pass
        return position, float(progress), velocity

    def _as013_contact_force_sample(self, view: Any | None) -> tuple[float, bool]:
        """Return force and live-report validity without treating errors as 0 N."""

        if view is None:
            return 0.0, False
        try:
            matrix = view.get_contact_force_matrix(self._sim_dt)
            if hasattr(matrix, "detach"):
                matrix = matrix.detach().cpu().numpy()
            forces = np.asarray(matrix, dtype=np.float64)
        except Exception:  # noqa: BLE001 - invalid readback is a task failure.
            return 0.0, False
        expected_shape = (
            int(getattr(view, "sensor_count", 0)),
            int(getattr(view, "filter_count", 0)),
            3,
        )
        if forces.shape != expected_shape or not np.isfinite(forces).all():
            return 0.0, False
        return float(np.linalg.norm(forces, axis=-1).sum()), True

    def _as013_right_slider_contact_sample(
        self, view: Any | None
    ) -> tuple[float, float, float, bool]:
        """Return net, inside-button, and outside-button operator forces.

        ``get_contact_force_matrix`` reports a net force vector, while the
        detailed contact rows report scalar normal forces.  Those quantities
        cannot be subtracted: simultaneous contacts can cancel in the net
        vector.  Both region forces therefore come from the same detailed rows
        and use complementary masks in the blade-local button AABB.
        """

        self._as013_right_contact_row_count = 0
        self._as013_right_slider_contact_row_count = 0
        self._as013_right_outside_slider_contact_row_count = 0
        self._as013_left_blade_contact_row_count = 0
        self._as013_left_slider_contact_row_count = 0
        aggregate_force, matrix_valid = self._as013_contact_force_sample(view)
        if not matrix_valid or view is None:
            return aggregate_force, 0.0, 0.0, False
        try:
            data = view.get_contact_data(self._sim_dt)
            forces, points, normals, separations, counts, starts = data

            def to_numpy(value: Any, dtype: Any) -> np.ndarray:
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                return np.asarray(value, dtype=dtype)

            forces_np = to_numpy(forces, np.float64).reshape(-1)
            points_np = to_numpy(points, np.float64).reshape(-1, 3)
            normals_np = to_numpy(normals, np.float64).reshape(-1, 3)
            separations_np = to_numpy(separations, np.float64).reshape(-1)
            counts_values = to_numpy(counts, np.float64)
            starts_values = to_numpy(starts, np.float64)
            if not (
                np.isfinite(counts_values).all()
                and np.isfinite(starts_values).all()
                and np.all(counts_values >= 0.0)
                and np.all(starts_values >= 0.0)
                and np.array_equal(counts_values, np.floor(counts_values))
                and np.array_equal(starts_values, np.floor(starts_values))
            ):
                return aggregate_force, 0.0, 0.0, False
            counts_np = counts_values.astype(np.uint64).reshape(
                int(view.sensor_count), int(view.filter_count)
            )
            starts_np = starts_values.astype(np.uint64).reshape(
                counts_np.shape
            )
        except Exception:  # noqa: BLE001 - malformed rows fail closed.
            return aggregate_force, 0.0, 0.0, False

        total = int(np.sum(counts_np, dtype=np.uint64))
        capacity = int(view.max_contact_data_count)
        # Equality means PhysX filled the entire row buffer, so truncation
        # cannot be excluded.  Treat saturation and nonsensical capacity as
        # invalid instead of silently classifying an incomplete manifold.
        if capacity < 1 or total >= capacity:
            return aggregate_force, 0.0, 0.0, False

        selected_forces: list[float] = []
        selected_points: list[np.ndarray] = []
        selected_normals: list[np.ndarray] = []
        selected_separations: list[float] = []
        for count, start in zip(
            counts_np.reshape(-1), starts_np.reshape(-1)
        ):
            first = int(start)
            stop = first + int(count)
            if (
                stop > len(forces_np)
                or stop > len(points_np)
                or stop > len(normals_np)
                or stop > len(separations_np)
            ):
                return aggregate_force, 0.0, 0.0, False
            selected_forces.extend(forces_np[first:stop])
            selected_points.extend(points_np[first:stop])
            selected_normals.extend(normals_np[first:stop])
            selected_separations.extend(separations_np[first:stop])

        if not selected_points:
            return (
                aggregate_force,
                0.0,
                0.0,
                aggregate_force == 0.0,
            )
        if not all(
            np.isfinite(value).all()
            for value in (
                np.asarray(selected_forces),
                np.asarray(selected_points),
                np.asarray(selected_normals),
                np.asarray(selected_separations),
            )
        ):
            return aggregate_force, 0.0, 0.0, False
        self._as013_right_contact_row_count = len(selected_points)
        self._as013_left_blade_contact_row_count = len(selected_points)
        blade_pose = self._physics_body_pose("blade_2")
        if blade_pose is None or not np.isfinite(blade_pose).all():
            return aggregate_force, 0.0, 0.0, False
        points_world = np.asarray(selected_points, dtype=np.float64).reshape(-1, 3)
        homogeneous = np.column_stack(
            [points_world, np.ones(len(points_world), dtype=np.float64)]
        )
        blade_inverse = _invert_pose_matrix(_pose_vector_to_matrix(blade_pose))
        points_local = (blade_inverse @ homogeneous.T).T[:, :3]
        slider_region_min = (
            SLIDER_BUTTON_LOCAL_MIN_M - SLIDER_CONTACT_REGION_TOLERANCE_M
        )
        slider_region_max = (
            SLIDER_BUTTON_LOCAL_MAX_M + SLIDER_CONTACT_REGION_TOLERANCE_M
        )
        in_slider_region = np.logical_and(
            points_local >= slider_region_min,
            points_local <= slider_region_max,
        ).all(axis=1)
        self._as013_right_slider_contact_row_count = int(
            np.count_nonzero(in_slider_region)
        )
        self._as013_right_outside_slider_contact_row_count = int(
            np.count_nonzero(~in_slider_region)
        )
        self._as013_left_slider_contact_row_count = (
            self._as013_right_slider_contact_row_count
        )
        absolute_forces = np.abs(
            np.asarray(selected_forces, dtype=np.float64)
        )
        slider_force = float(absolute_forces[in_slider_region].sum())
        outside_force = float(absolute_forces[~in_slider_region].sum())
        return aggregate_force, slider_force, outside_force, True

    def _as013_contact_forces(
        self,
    ) -> tuple[float, ...]:
        """Return canonical v3/v4 physical-contact forces.

        Order is full-right-hand grip aggregate, right palm housing, right
        non-thumb housing, slider-hand detailed-row total, slider-button region,
        unsafe right moving contact, left housing, left moving, housing-table,
        and moving-component-table.  The complementary slider-hand outside-
        button force is stored explicitly because the 10-value tuple is a
        stable replay ABI; it is never inferred from the net force matrix.
        """

        right_grip = self._as013_contact_force_sample(
            self._as013_right_grip_housing_contact_view
        )
        right_palm = self._as013_contact_force_sample(
            self._as013_right_palm_housing_contact_view
        )
        right_non_thumb = self._as013_contact_force_sample(
            self._as013_right_non_thumb_housing_contact_view
        )
        slider_hand = self._as013_right_slider_contact_sample(
            self._as013_left_slider_moving_contact_view
            if ROLE_SCHEMA_VERSION >= 4
            else self._as013_right_thumb_moving_contact_view
        )
        right_unsafe = self._as013_contact_force_sample(
            self._as013_right_unsafe_moving_contact_view
        )
        left_housing = self._as013_contact_force_sample(
            self._as013_left_housing_contact_view
        )
        left_moving = self._as013_contact_force_sample(
            self._as013_left_moving_contact_view
        )
        housing_table = self._as013_contact_force_sample(
            self._as013_body_table_contact_view
        )
        moving_table = self._as013_contact_force_sample(
            self._as013_blade_table_contact_view
        )
        grip_counts, grip_counts_valid = self._as013_contact_filter_counts(
            self._as013_right_grip_housing_contact_view,
            self._as013_right_grip_filter_paths,
            with_validity=True,
        )
        self._as013_right_grip_contact_owner_count = sum(
            int(count) > 0 for count in grip_counts.values()
        )
        self._as013_left_housing_release_contact_force = float(
            left_housing[0]
        )
        self._as013_left_moving_release_contact_force = float(left_moving[0])
        self._as013_left_release_contact_report_valid = bool(
            left_housing[1] and left_moving[1]
        )
        self._as013_housing_table_contact_force = float(housing_table[0])
        self._as013_moving_table_contact_force = float(moving_table[0])
        self._as013_right_thumb_net_contact_force = float(slider_hand[0])
        self._as013_right_thumb_outside_slider_contact_force = float(
            slider_hand[2]
        )
        samples = (
            right_grip,
            right_palm,
            right_non_thumb,
            right_unsafe,
            left_housing,
            left_moving,
            housing_table,
            moving_table,
        )
        self._as013_right_contact_rows_valid = bool(slider_hand[3])
        self._as013_left_contact_rows_valid = bool(slider_hand[3])
        self._as013_contact_report_valid = bool(
            all(valid for _force, valid in samples)
            and slider_hand[3]
            and grip_counts_valid
        )
        return (
            right_grip[0],
            right_palm[0],
            right_non_thumb[0],
            slider_hand[1] + slider_hand[2],
            slider_hand[1],
            right_unsafe[0],
            left_housing[0],
            left_moving[0],
            housing_table[0],
            moving_table[0],
        )

    @staticmethod
    def _as013_housing_velocity_is_stable(velocity: Any | None) -> bool:
        if velocity is None:
            return False
        values = np.asarray(velocity, dtype=np.float64).reshape(-1)
        return bool(
            values.size >= 6
            and np.isfinite(values[:6]).all()
            and float(np.linalg.norm(values[:3])) <= STABLE_LINEAR_SPEED_MPS
            and float(np.linalg.norm(values[3:6]))
            <= STABLE_ANGULAR_SPEED_RADPS
        )

    def _after_task_physics_substep(self) -> None:
        if ROLE_SCHEMA_VERSION >= 3:
            self._after_task_physics_substep_v3()
            return
        (
            right_body,
            left_blade,
            left_slider,
            right_blade,
            table_support,
        ) = (
            self._as013_contact_forces()
        )
        position, progress, velocity = self._as013_joint_state()
        previous_progress = float(
            getattr(self, "_as013_previous_blade_progress", math.nan)
        )
        self._as013_completion_forward_progress_now = bool(
            np.isfinite([previous_progress, progress]).all()
            and progress
            > previous_progress + COMPLETION_FORWARD_PROGRESS_EPS
        )
        self._as013_completion_target_progress_now = bool(
            _as013_target_motion_delta(
                progress,
                previous_progress,
                getattr(
                    self,
                    "_as013_target_direction",
                    TARGET_DIRECTION_RETRACT,
                ),
            )
            > COMPLETION_FORWARD_PROGRESS_EPS
        )
        self._as013_target_endpoint_reached_now = (
            _as013_target_endpoint_reached(
                progress,
                getattr(
                    self,
                    "_as013_target_direction",
                    TARGET_DIRECTION_RETRACT,
                ),
            )
        )
        body_velocity = self._physics_body_velocity("E_body_1")
        right_stabilized_before_substep = self._as013_left_stabilized_seen
        if not self._as013_contact_views_ready:
            self._as013_latch_failure("contact_views_unavailable")
        if not self._as013_contact_report_valid:
            self._as013_latch_failure("contact_report_invalid")
        self._as013_left_stable_contact_now = bool(
            self._as013_contact_report_valid
            and right_body > CONTACT_FORCE_THRESHOLD_N
            and self._as013_housing_velocity_is_stable(body_velocity)
        )
        if self._as013_left_stable_contact_now:
            self._as013_left_stabilization_substeps = min(
                LEFT_STABILIZATION_CONSECUTIVE_SUBSTEPS,
                self._as013_left_stabilization_substeps + 1,
            )
        else:
            self._as013_left_stabilization_substeps = 0
        if (
            self._as013_left_stabilization_substeps
            >= LEFT_STABILIZATION_CONSECUTIVE_SUBSTEPS
        ):
            self._as013_left_stabilized_seen = True
        # Consume the current substep's stabilizer count before judging order.
        # This makes the fourth stabilizing substep a valid boundary when the
        # left fingertip first reaches the slider in that same physics sample.
        right_stabilized_now = bool(self._as013_left_stabilized_seen)
        left_slider_contact_now = bool(
            self._as013_contact_report_valid
            and left_slider > CONTACT_FORCE_THRESHOLD_N
        )
        if left_slider_contact_now and not right_stabilized_before_substep:
            self._as013_preorder_slider_contact_substeps = min(
                ORDER_VIOLATION_CONSECUTIVE_SUBSTEPS,
                getattr(
                    self,
                    "_as013_preorder_slider_contact_substeps",
                    0,
                )
                + 1,
            )
        else:
            self._as013_preorder_slider_contact_substeps = 0
        if self._as013_preorder_slider_contact_substeps >= (
            ORDER_VIOLATION_CONSECUTIVE_SUBSTEPS
        ):
            self._as013_wrong_order_failure = True
            self._as013_latch_failure(
                "left_slider_contact_before_right_stabilization"
            )
        elif right_stabilized_now:
            # A first slider touch on the exact fourth stabilizing substep is
            # ordered and must not leak a pre-order count into the next frame.
            self._as013_preorder_slider_contact_substeps = 0
        if (
            right_stabilized_now
            and self._as013_left_stable_contact_now
            and left_slider_contact_now
            and not self._as013_wrong_order_failure
        ):
            self._as013_right_slider_contact_substeps = min(
                RIGHT_SLIDER_CONTACT_CONSECUTIVE_SUBSTEPS,
                self._as013_right_slider_contact_substeps + 1,
            )
        else:
            self._as013_right_slider_contact_substeps = 0
        if (
            self._as013_right_slider_contact_substeps
            >= RIGHT_SLIDER_CONTACT_CONSECUTIVE_SUBSTEPS
        ):
            if not self._as013_right_slider_contact_seen:
                self._as013_ordered_contact_progress = (
                    float(progress) if np.isfinite(progress) else None
                )
            self._as013_right_slider_contact_seen = True
            self._as013_try_anchor_causal_drive(position)
            if (
                not getattr(
                    self,
                    "_as013_blade_drive_stiffness_scaled",
                    False,
                )
                and getattr(
                    self,
                    "_as013_blade_drive_stiffness_scale",
                    1.0,
                )
                != 1.0
            ):
                self._set_blade_drive_stiffness_scale(
                    self._as013_blade_drive_stiffness_scale
                )
                self._as013_blade_drive_stiffness_scaled = True
            if (
                not getattr(
                    self, "_as013_blade_drive_damping_scaled", False
                )
                and getattr(self, "_as013_blade_drive_damping_scale", 1.0)
                != 1.0
            ):
                self._set_blade_drive_damping_scale(
                    self._as013_blade_drive_damping_scale
                )
                self._as013_blade_drive_damping_scaled = True
            if (
                not getattr(
                    self, "_as013_blade_joint_friction_scaled", False
                )
                and getattr(self, "_as013_blade_joint_friction_scale", 1.0)
                != 1.0
            ):
                self._set_blade_joint_friction_scale(
                    self._as013_blade_joint_friction_scale
                )
                self._as013_blade_joint_friction_scaled = True
        progress_before_stabilization = bool(
            np.isfinite(progress)
            and progress > self._as013_reset_progress + 0.03
            and not right_stabilized_before_substep
        )
        if progress_before_stabilization:
            self._as013_prestabilization_progress_substeps = min(
                PRESTABILIZATION_PROGRESS_CONSECUTIVE_SUBSTEPS,
                getattr(
                    self,
                    "_as013_prestabilization_progress_substeps",
                    0,
                )
                + 1,
            )
        else:
            self._as013_prestabilization_progress_substeps = 0
        if self._as013_prestabilization_progress_substeps >= (
            PRESTABILIZATION_PROGRESS_CONSECUTIVE_SUBSTEPS
        ):
            self._as013_wrong_order_failure = True
            self._as013_latch_failure(
                "blade_progress_before_right_stabilization"
            )
        elif right_stabilized_now:
            self._as013_prestabilization_progress_substeps = 0
        self._as013_slide_progress_now = bool(
            self._as013_ordered_contact_progress is not None
            and self._as013_left_stable_contact_now
            and left_slider_contact_now
            and np.isfinite(progress)
            and progress
            > self._as013_ordered_contact_progress + SLIDE_PROGRESS_DELTA
        )
        if self._as013_slide_progress_now:
            self._as013_slide_progress_substeps = min(
                SLIDE_PROGRESS_CONSECUTIVE_SUBSTEPS,
                self._as013_slide_progress_substeps + 1,
            )
        else:
            self._as013_slide_progress_substeps = 0
        if (
            self._as013_slide_progress_substeps
            >= SLIDE_PROGRESS_CONSECUTIVE_SUBSTEPS
        ):
            self._as013_slide_progress_seen = True
        if (
            np.isfinite(progress)
            and progress >= SUCCESS_PROGRESS
            and self._as013_left_stable_contact_now
            and left_slider_contact_now
            and self._as013_right_slider_contact_seen
            and self._as013_slide_progress_seen
            and self._as013_completion_forward_progress_now
            and not self._as013_wrong_order_failure
            and not self._as013_unsafe_blade_contact_failure
        ):
            # Latch physical completion before the required release.  Merely
            # touching both sides early, finishing one-handed, and later
            # restoring a static contact must never satisfy the bimanual task
            # contract: forward travel at or above the threshold must occur
            # while both task-defining contacts are live.
            self._as013_ordered_slide_completion_seen = True
        if (
            self._as013_contact_report_valid
            and right_blade > CONTACT_FORCE_THRESHOLD_N
        ):
            self._as013_unsafe_blade_contact_substeps = min(
                UNSAFE_BLADE_CONTACT_CONSECUTIVE_SUBSTEPS,
                getattr(
                    self,
                    "_as013_unsafe_blade_contact_substeps",
                    0,
                )
                + 1,
            )
        else:
            self._as013_unsafe_blade_contact_substeps = 0
        if self._as013_unsafe_blade_contact_substeps >= (
            UNSAFE_BLADE_CONTACT_CONSECUTIVE_SUBSTEPS
        ):
            self._as013_unsafe_blade_contact_failure = True
            self._as013_latch_failure(
                "unsafe_right_hand_moving_component_contact"
            )

        causal_advance_ready = bool(
            self._as013_left_stabilized_seen
            and self._as013_left_stable_contact_now
            and left_slider_contact_now
            and self._as013_right_slider_contact_seen
            and _passive_drive_latch_motion_ready(
                self._as013_right_slide_command,
                self._as013_slide_progress_seen
                and progress >= SUCCESS_PROGRESS,
            )
            and np.isfinite(position)
            and not self._as013_wrong_order_failure
            and not self._as013_unsafe_blade_contact_failure
        )
        self._as013_drive_assist_active = False
        target = float(position)
        if (
            causal_advance_ready
            and getattr(self, "_as013_causal_drive_enabled", False)
            and self._as013_causal_drive_contact_joint_position is not None
        ):
            (
                causal_target,
                common_forward,
                fingertip_forward,
                command_forward,
            ) = self._as013_causal_drive_target_now(position)
            if np.isfinite(
                [
                    causal_target,
                    common_forward,
                    fingertip_forward,
                    command_forward,
                ]
            ).all():
                target = float(causal_target)
                self._as013_causal_drive_forward_displacement_m = float(
                    common_forward
                )
                self._as013_causal_drive_fingertip_forward_m = float(
                    fingertip_forward
                )
                self._as013_causal_drive_command_forward_m = float(
                    command_forward
                )
        written_target = None
        if getattr(self, "_trajectory_action_phase_active", False):
            written_target = self._write_blade_drive_target(target)
        if written_target is not None:
            self._as013_drive_assist_active = bool(
                written_target > float(position) + 1.0e-6
            )
        # Stable-origin settling calls this hook outside the audited action
        # phase and must remain write-free.  During action, the passive baseline
        # follows measured q with a 0.1 mm write deadband: it can resist velocity
        # through damping/friction but cannot accumulate meaningful displacement
        # error toward reset.  The optional causal branch remains bounded by
        # live fingertip and net hand-command motion from ordered contact, never
        # by elapsed time or a pre-authored blade trajectory.

        finite = np.isfinite([position, progress, velocity]).all()
        blade_velocity = self._physics_body_velocity("blade_2")
        for value in (body_velocity, blade_velocity):
            if value is not None:
                finite = bool(finite and np.isfinite(value).all())
        if not finite:
            self._as013_nonfinite_failure = True
            self._as013_latch_failure("nonfinite_physics_state")
        body_pose = self._physics_body_pose("E_body_1")
        if body_pose is not None:
            if self._as013_start_body_pose is None:
                self._as013_start_body_pose = body_pose.copy()
            displacement = float(
                np.linalg.norm(body_pose[:2] - self._as013_start_body_pose[:2])
            )
            dropped = bool(body_pose[2] < _TABLE_TOP_Z - 0.02)
            if displacement > MAX_HOUSING_XY_DISPLACEMENT_M or dropped:
                self._as013_displacement_failure = True
                self._as013_latch_failure("housing_displaced_or_dropped")
        lower, upper = self._as013_joint_limits
        if np.isfinite(position) and not (lower - 1.0e-4 <= position <= upper + 1.0e-4):
            self._as013_displacement_failure = True
            self._as013_latch_failure("blade_joint_out_of_limits")
        self._as013_previous_blade_progress = (
            float(progress) if np.isfinite(progress) else math.nan
        )
        self._as013_last_metrics = self._as013_metrics(
            contact_forces=(
                right_body,
                left_blade,
                left_slider,
                right_blade,
                table_support,
            )
        )

    def _after_task_physics_substep_v3(self) -> None:
        """Latch handheld pickup plus the current schema's slider sequence.

        Schema v5 keeps the proven physical-right grip/pickup path, assigns the
        slider to physical-left fingertips, and accepts either sampled joint
        endpoint.  The method name is retained so sealed v3/v4 recordings keep
        a stable Python entry point.
        """

        # Stable-origin settling is an exact write-free, semantics-free phase.
        # Its physics can contain startup contacts and joint drift that must
        # become the accepted baseline, never episode progress or failure.
        if not getattr(self, "_trajectory_action_phase_active", False):
            return

        raw_forces = tuple(self._as013_contact_forces())
        legacy_force_tuple = len(raw_forces) == 5
        (
            right_grip,
            right_palm,
            right_non_thumb,
            right_thumb_moving,
            right_thumb_slider,
            right_unsafe_moving,
            left_housing,
            left_moving,
            housing_table,
            moving_table,
        ) = _canonical_contact_force_tuple(
            raw_forces,
            legacy_left_housing=getattr(
                self, "_as013_left_housing_release_contact_force", 0.0
            ),
            legacy_left_moving=getattr(
                self, "_as013_left_moving_release_contact_force", 0.0
            ),
        )
        position, progress, velocity = self._as013_joint_state()
        previous_progress = float(
            getattr(self, "_as013_previous_blade_progress", math.nan)
        )
        self._as013_completion_forward_progress_now = bool(
            np.isfinite([previous_progress, progress]).all()
            and progress
            > previous_progress + COMPLETION_FORWARD_PROGRESS_EPS
        )
        self._as013_completion_target_progress_now = bool(
            _as013_target_motion_delta(
                progress,
                previous_progress,
                getattr(
                    self,
                    "_as013_target_direction",
                    TARGET_DIRECTION_RETRACT,
                ),
            )
            > COMPLETION_FORWARD_PROGRESS_EPS
        )
        self._as013_target_endpoint_reached_now = (
            _as013_target_endpoint_reached(
                progress,
                getattr(
                    self,
                    "_as013_target_direction",
                    TARGET_DIRECTION_RETRACT,
                ),
            )
        )
        body_pose = self._physics_body_pose("E_body_1")
        body_velocity = self._physics_body_velocity("E_body_1")
        blade_velocity = self._physics_body_velocity("blade_2")
        try:
            body_velocity_values = np.asarray(
                body_velocity, dtype=np.float64
            ).reshape(-1)
        except Exception:  # noqa: BLE001 - malformed physics readback.
            body_velocity_values = np.zeros(0, dtype=np.float64)
        self._as013_body_velocity_report_valid = bool(
            body_velocity_values.size >= 6
            and np.isfinite(body_velocity_values[:6]).all()
        )
        self._as013_body_vertical_velocity_mps = (
            float(body_velocity_values[2])
            if self._as013_body_velocity_report_valid
            else math.nan
        )
        if not getattr(self, "_as013_contact_views_ready", False):
            self._as013_latch_failure("contact_views_unavailable")
        if not getattr(self, "_as013_contact_report_valid", False):
            self._as013_latch_failure("contact_report_invalid")

        grip_counts = getattr(
            self, "_as013_right_grip_contact_owner_count", 0
        )
        if legacy_force_tuple and right_grip > CONTACT_FORCE_THRESHOLD_N:
            grip_counts = RIGHT_GRIP_REQUIRED_CONTACT_OWNERS
        self._as013_right_grip_contact_owner_count = int(grip_counts)
        self._as013_right_grip_contact_now = bool(
            self._as013_contact_report_valid
            and right_grip > CONTACT_FORCE_THRESHOLD_N
            and grip_counts >= RIGHT_GRIP_REQUIRED_CONTACT_OWNERS
        )
        # Live grip requires force from at least two distinct full-right-hand
        # rigid bodies on the housing.  Palm contact is diagnostic only.  Tool
        # velocity is deliberately not a gate here because lifting must not
        # clear its own grasp evidence. Schema-v5 success consumes the pickup
        # latch and does not add a terminal stability window.
        self._as013_left_stable_contact_now = (
            self._as013_right_grip_contact_now
        )
        if self._as013_right_grip_contact_now:
            self._as013_left_stabilization_substeps = min(
                LEFT_STABILIZATION_CONSECUTIVE_SUBSTEPS,
                getattr(self, "_as013_left_stabilization_substeps", 0) + 1,
            )
        else:
            self._as013_left_stabilization_substeps = 0
        if self._as013_left_stabilization_substeps >= (
            LEFT_STABILIZATION_CONSECUTIVE_SUBSTEPS
        ):
            self._as013_right_grip_seen = True
            self._as013_left_stabilized_seen = True

        try:
            body_values = np.asarray(body_pose, dtype=np.float64).reshape(-1)
        except Exception:  # noqa: BLE001 - malformed physics readback.
            body_values = np.zeros(0, dtype=np.float64)
        self._as013_body_pose_report_valid = bool(
            body_values.size >= 7 and np.isfinite(body_values[:7]).all()
        )
        if not self._as013_body_pose_report_valid:
            # Never reuse the preceding substep's lift height.  A stale lifted
            # value could otherwise satisfy pickup/success after pose readback
            # disappeared or became non-finite.
            self._as013_pickup_lift_m = math.nan
            self._as013_knife_lifted_now = False
            self._as013_pickup_candidate_substeps = 0
            self._as013_nonfinite_failure = True
            self._as013_latch_failure("nonfinite_physics_state")
        else:
            if self._as013_start_body_pose is None:
                self._as013_start_body_pose = body_values[:7].copy()
            reset_height = float(
                getattr(self, "_as013_reset_body_height_m", math.nan)
            )
            if not np.isfinite(reset_height):
                start_values = np.asarray(
                    self._as013_start_body_pose, dtype=np.float64
                ).reshape(-1)
                reset_height = (
                    float(start_values[2])
                    if start_values.size >= 3
                    and np.isfinite(start_values[2])
                    else float(body_values[2])
                )
                self._as013_reset_body_height_m = reset_height
            self._as013_pickup_lift_m = float(
                body_values[2] - reset_height
            )
            inside_workspace = bool(
                HANDHELD_WORKSPACE_X_RANGE_M[0]
                <= body_values[0]
                <= HANDHELD_WORKSPACE_X_RANGE_M[1]
                and HANDHELD_WORKSPACE_Y_RANGE_M[0]
                <= body_values[1]
                <= HANDHELD_WORKSPACE_Y_RANGE_M[1]
                and HANDHELD_WORKSPACE_Z_RANGE_M[0]
                <= body_values[2]
                <= HANDHELD_WORKSPACE_Z_RANGE_M[1]
            )
            if not inside_workspace:
                self._as013_displacement_failure = True
                self._as013_latch_failure("knife_escaped_handheld_workspace")

        both_parts_off_table = bool(
            self._as013_contact_report_valid
            and housing_table <= CONTACT_FORCE_THRESHOLD_N
            and moving_table <= CONTACT_FORCE_THRESHOLD_N
        )
        self._as013_knife_lifted_now = bool(
            self._as013_body_pose_report_valid
            and np.isfinite(self._as013_pickup_lift_m)
            and self._as013_pickup_lift_m >= PICKUP_LIFT_HEIGHT_M
            and both_parts_off_table
        )
        pickup_candidate = bool(
            self._as013_right_grip_seen
            and self._as013_right_grip_contact_now
            and self._as013_knife_lifted_now
        )
        if pickup_candidate:
            self._as013_pickup_candidate_substeps = min(
                PICKUP_CONSECUTIVE_SUBSTEPS,
                getattr(self, "_as013_pickup_candidate_substeps", 0) + 1,
            )
        else:
            self._as013_pickup_candidate_substeps = 0
        if self._as013_pickup_candidate_substeps >= PICKUP_CONSECUTIVE_SUBSTEPS:
            self._as013_pickup_seen = True
        if self._as013_pickup_seen:
            self._as013_pickup_peak_lift_m = max(
                float(getattr(self, "_as013_pickup_peak_lift_m", 0.0)),
                float(self._as013_pickup_lift_m),
            )

        slider_contact_raw_now = bool(
            self._as013_contact_report_valid
            and right_thumb_slider > CONTACT_FORCE_THRESHOLD_N
        )
        # Preserve a partially debounced left-fingertip touch across a short
        # valid PhysX contact-manifold dropout.  Invalid reports still fail
        # closed above; this only bridges zero-row samples between valid rows.
        if (
            self._as013_pickup_seen
            and not self._as013_right_slider_contact_seen
        ):
            (
                slider_contact_now,
                self._as013_slider_contact_observation_substeps,
                self._as013_slider_contact_dropout_substeps,
            ) = _advance_as013_slider_contact_observation(
                current_contact=slider_contact_raw_now,
                contact_substeps=self._as013_slider_contact_observation_substeps,
                dropout_substeps=self._as013_slider_contact_dropout_substeps,
            )
        else:
            slider_contact_now = slider_contact_raw_now
            self._as013_slider_contact_observation_substeps = 0
            self._as013_slider_contact_dropout_substeps = 0
        self._as013_slider_contact_effective_now = bool(slider_contact_now)
        # A light button touch while acquiring/lifting is natural and is not a
        # terminal ordering error.  It simply cannot latch operation until the
        # physical pickup has completed.  In v4 these are physical-left
        # fingertip rows; the legacy local variable names preserve tuple ABI.
        if slider_contact_now and not self._as013_pickup_seen:
            self._as013_preorder_slider_contact_substeps = min(
                ORDER_VIOLATION_CONSECUTIVE_SUBSTEPS,
                getattr(self, "_as013_preorder_slider_contact_substeps", 0) + 1,
            )
        else:
            self._as013_preorder_slider_contact_substeps = 0

        # v3 tuple index 3 is the detailed-row total and index 4 is the
        # complementary inside-button partition.  The tuple is therefore
        # self-contained; the object attribute is diagnostic only.  Legacy
        # five-value recordings retain their historical approximation.
        thumb_outside_slider = max(
            0.0,
            float(right_thumb_moving) - float(right_thumb_slider),
        )
        # Incidental contact with the moving assembly is diagnostic, not a
        # terminal event.  Only detailed slider-hand rows inside the authored
        # button region count toward ordered operation.
        unsafe_right_force = float(right_unsafe_moving) + thumb_outside_slider
        if (
            self._as013_contact_report_valid
            and unsafe_right_force > CONTACT_FORCE_THRESHOLD_N
        ):
            self._as013_unsafe_blade_contact_substeps = min(
                UNSAFE_BLADE_CONTACT_CONSECUTIVE_SUBSTEPS,
                getattr(self, "_as013_unsafe_blade_contact_substeps", 0) + 1,
            )
        else:
            self._as013_unsafe_blade_contact_substeps = 0
        self._as013_unsafe_blade_contact_failure = False

        left_operator_contact_force = float(left_housing) + float(left_moving)
        if (
            self._as013_contact_report_valid
            and left_operator_contact_force > CONTACT_FORCE_THRESHOLD_N
        ):
            self._as013_left_hand_participation_substeps = min(
                LEFT_HAND_PARTICIPATION_CONSECUTIVE_SUBSTEPS,
                getattr(self, "_as013_left_hand_participation_substeps", 0) + 1,
            )
        else:
            self._as013_left_hand_participation_substeps = 0
        # The physical left hand is the v4 slider operator.  Its housing or
        # off-button moving contacts remain recoverable diagnostics; expected
        # contact must never reset the episode.
        self._as013_left_hand_participation_failure = False

        progress_before_pickup = bool(
            np.isfinite(progress)
            and abs(progress - self._as013_reset_progress) > 0.03
            and not self._as013_pickup_seen
        )
        if progress_before_pickup:
            self._as013_prestabilization_progress_substeps = min(
                PRESTABILIZATION_PROGRESS_CONSECUTIVE_SUBSTEPS,
                getattr(
                    self, "_as013_prestabilization_progress_substeps", 0
                )
                + 1,
            )
        else:
            self._as013_prestabilization_progress_substeps = 0
        # Slider travel before pickup is recoverable.  It is deliberately not
        # credited as ordered task progress: the operation anchor is created
        # only after pickup and a live thumb-button contact.  Natural contact
        # while closing the same hand around the knife must not hard-reset the
        # episode.
        self._as013_wrong_order_failure = False

        ordered_operation_contact_now = bool(
            self._as013_pickup_seen
            and slider_contact_now
            and (
                ROLE_SCHEMA_VERSION >= 5
                or (
                    self._as013_right_grip_contact_now
                    and self._as013_knife_lifted_now
                )
            )
        )
        if ordered_operation_contact_now:
            self._as013_right_slider_contact_substeps = min(
                RIGHT_SLIDER_CONTACT_CONSECUTIVE_SUBSTEPS,
                getattr(self, "_as013_right_slider_contact_substeps", 0) + 1,
            )
        else:
            self._as013_right_slider_contact_substeps = 0
        if self._as013_right_slider_contact_substeps >= (
            RIGHT_SLIDER_CONTACT_CONSECUTIVE_SUBSTEPS
        ):
            if not self._as013_right_slider_contact_seen:
                self._as013_ordered_contact_progress = (
                    float(progress) if np.isfinite(progress) else None
                )
            self._as013_right_slider_contact_seen = True
            self._as013_try_anchor_causal_drive(position)

        self._as013_slide_progress_now = bool(
            self._as013_ordered_contact_progress is not None
            and ordered_operation_contact_now
            and np.isfinite(progress)
            and _as013_target_motion_delta(
                progress,
                self._as013_ordered_contact_progress,
                self._as013_target_direction,
            )
            > SLIDE_PROGRESS_DELTA
        )
        if self._as013_slide_progress_now:
            self._as013_slide_progress_substeps = min(
                SLIDE_PROGRESS_CONSECUTIVE_SUBSTEPS,
                getattr(self, "_as013_slide_progress_substeps", 0) + 1,
            )
        else:
            self._as013_slide_progress_substeps = 0
        if self._as013_slide_progress_substeps >= (
            SLIDE_PROGRESS_CONSECUTIVE_SUBSTEPS
        ):
            self._as013_slide_progress_seen = True
        if (
            self._as013_pickup_seen
            and self._as013_right_slider_contact_seen
            and self._as013_target_endpoint_reached_now
        ):
            self._as013_ordered_slide_completion_seen = True

        # Preserve debounced table-contact events for diagnostics, but require
        # physical return/descent evidence for terminal drop truth.  A local
        # touch of either knife part can therefore be recovered from.
        table_contact_now = bool(
            self._as013_contact_report_valid
            and max(float(housing_table), float(moving_table))
            > DROP_TABLE_CONTACT_FORCE_THRESHOLD_N
        )
        (
            self._as013_table_contact_substeps,
            self._as013_table_contact_event_active,
            self._as013_table_contact_event_count,
            self._as013_second_table_contact_seen,
        ) = _advance_as013_table_contact_event(
            current_contact=table_contact_now,
            contact_substeps=self._as013_table_contact_substeps,
            event_active=self._as013_table_contact_event_active,
            event_count=self._as013_table_contact_event_count,
            armed=bool(self._as013_pickup_seen),
        )
        pickup_lift = float(
            getattr(self, "_as013_pickup_lift_m", math.nan)
        )
        pickup_peak_lift = float(
            getattr(self, "_as013_pickup_peak_lift_m", math.nan)
        )
        self._as013_drop_near_table_now = bool(
            self._as013_pickup_seen
            and self._as013_body_pose_report_valid
            and np.isfinite(pickup_lift)
            and pickup_lift <= DROP_RETURN_LIFT_HEIGHT_M
        )
        self._as013_drop_peak_descent_now = bool(
            self._as013_pickup_seen
            and self._as013_body_pose_report_valid
            and np.isfinite(pickup_lift)
            and np.isfinite(pickup_peak_lift)
            and pickup_peak_lift - pickup_lift >= DROP_DESCENT_FROM_PEAK_M
        )
        self._as013_drop_falling_now = bool(
            self._as013_body_velocity_report_valid
            and np.isfinite(self._as013_body_vertical_velocity_mps)
            and self._as013_body_vertical_velocity_mps
            <= DROP_FALLING_VERTICAL_SPEED_MPS
        )
        self._as013_drop_table_impact_now = bool(
            self._as013_contact_report_valid
            and max(float(housing_table), float(moving_table))
            > DROP_TABLE_IMPACT_FORCE_THRESHOLD_N
        )
        downward_drop_evidence = bool(
            self._as013_drop_peak_descent_now
            and self._as013_body_velocity_report_valid
            and (
                self._as013_drop_falling_now
                or self._as013_drop_table_impact_now
            )
        )
        self._as013_drop_evidence_now = bool(
            self._as013_pickup_seen
            and self._as013_contact_report_valid
            and not self._as013_right_grip_contact_now
            and (self._as013_drop_near_table_now or downward_drop_evidence)
        )
        if self._as013_drop_evidence_now:
            self._as013_drop_substeps = min(
                DROP_CONSECUTIVE_SUBSTEPS,
                int(getattr(self, "_as013_drop_substeps", 0)) + 1,
            )
        else:
            self._as013_drop_substeps = 0
        if self._as013_drop_substeps >= DROP_CONSECUTIVE_SUBSTEPS:
            self._as013_drop_failure = True
            self._as013_latch_failure("knife_dropped_after_pickup")

        causal_advance_ready = bool(
            self._as013_target_direction == TARGET_DIRECTION_EXTEND
            and
            ordered_operation_contact_now
            and self._as013_right_slider_contact_seen
            and _passive_drive_latch_motion_ready(
                self._as013_right_slide_command,
                self._as013_slide_progress_seen
                and progress >= SUCCESS_PROGRESS,
            )
            and np.isfinite(position)
        )
        self._as013_drive_assist_active = False
        target = float(position)
        if (
            causal_advance_ready
            and getattr(self, "_as013_causal_drive_enabled", False)
            and self._as013_causal_drive_contact_joint_position is not None
        ):
            (
                causal_target,
                common_forward,
                fingertip_forward,
                command_forward,
            ) = self._as013_causal_drive_target_now(position)
            if np.isfinite(
                [causal_target, common_forward, fingertip_forward, command_forward]
            ).all():
                target = float(causal_target)
                self._as013_causal_drive_forward_displacement_m = float(
                    common_forward
                )
                self._as013_causal_drive_fingertip_forward_m = float(
                    fingertip_forward
                )
                self._as013_causal_drive_command_forward_m = float(
                    command_forward
                )
        written_target = None
        if getattr(self, "_trajectory_action_phase_active", False):
            written_target = self._write_blade_drive_target(target)
        if written_target is not None:
            self._as013_drive_assist_active = bool(
                written_target > float(position) + 1.0e-6
            )

        finite = np.isfinite([position, progress, velocity]).all()
        for value in (body_velocity, blade_velocity):
            if value is not None:
                finite = bool(finite and np.isfinite(value).all())
        if not finite:
            self._as013_nonfinite_failure = True
            self._as013_latch_failure("nonfinite_physics_state")
        lower, upper = self._as013_joint_limits
        if np.isfinite(position) and not (
            lower - 1.0e-4 <= position <= upper + 1.0e-4
        ):
            self._as013_displacement_failure = True
            self._as013_latch_failure("blade_joint_out_of_limits")
        self._as013_previous_blade_progress = (
            float(progress) if np.isfinite(progress) else math.nan
        )
        self._as013_last_metrics = self._as013_metrics(
            contact_forces=raw_forces
        )

    def _as013_metrics(
        self,
        *,
        contact_forces: tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        raw_contact_forces = tuple(
            contact_forces
            if contact_forces is not None
            else self._as013_contact_forces()
        )
        legacy_force_tuple = len(raw_contact_forces) == 5
        (
            right_grip,
            right_palm,
            right_non_thumb,
            right_thumb_moving,
            right_thumb_slider,
            right_unsafe_moving,
            left_housing,
            left_moving,
            housing_table,
            moving_table,
        ) = _canonical_contact_force_tuple(
            raw_contact_forces,
            legacy_left_housing=getattr(
                self, "_as013_left_housing_release_contact_force", 0.0
            ),
            legacy_left_moving=getattr(
                self, "_as013_left_moving_release_contact_force", 0.0
            ),
        )
        table_support = float(housing_table) + float(moving_table)
        right_thumb_outside_slider = max(
            0.0,
            float(right_thumb_moving) - float(right_thumb_slider),
        )
        right_thumb_outside_slider_diagnostic = float(
            getattr(
                self,
                "_as013_right_thumb_outside_slider_contact_force",
                math.nan,
            )
        )
        right_thumb_net_force = (
            float(right_thumb_moving)
            if legacy_force_tuple
            else float(
                getattr(
                    self,
                    "_as013_right_thumb_net_contact_force",
                    right_thumb_moving,
                )
            )
        )
        right_unsafe_total = float(right_unsafe_moving) + (
            right_thumb_outside_slider if ROLE_SCHEMA_VERSION == 3 else 0.0
        )
        left_forbidden_force = float(left_housing) + float(left_moving)
        contact_force_values_valid = bool(
            np.isfinite(
                [
                    right_grip,
                    right_palm,
                    right_non_thumb,
                    right_thumb_moving,
                    right_thumb_slider,
                    right_thumb_outside_slider,
                    right_unsafe_moving,
                    left_housing,
                    left_moving,
                    housing_table,
                    moving_table,
                ]
            ).all()
            and min(
                right_grip,
                right_palm,
                right_non_thumb,
                right_thumb_moving,
                right_thumb_slider,
                right_thumb_outside_slider,
                right_unsafe_moving,
                left_housing,
                left_moving,
                housing_table,
                moving_table,
            )
            >= 0.0
        )
        position, progress, velocity = self._as013_joint_state()
        body_velocity = self._physics_body_velocity("E_body_1")
        blade_velocity = self._physics_body_velocity("blade_2")
        stability_diagnostics = _as013_stability_diagnostics(
            body_velocity,
            blade_velocity,
            contact_report_valid=self._as013_contact_report_valid,
            right_grip_contact_now=bool(
                getattr(self, "_as013_right_grip_contact_now", False)
            ),
            knife_lifted_now=bool(
                getattr(self, "_as013_knife_lifted_now", False)
            ),
            left_slider_contact_force=float(right_thumb_slider),
            # The speed-only ``stable`` flag is useful during pickup and slide
            # diagnostics.  Final v4 hold truth separately requires current
            # slider contact in ``_v4_raw_final_hold_from_metrics``.
            require_left_slider_contact=True,
        )
        stable = bool(stability_diagnostics["stable"])
        body_pose_report_valid = bool(
            getattr(self, "_as013_body_pose_report_valid", False)
        )
        if (
            not getattr(self, "_as013_recording_origin_accepted", False)
            and not getattr(self, "_trajectory_action_phase_active", False)
        ):
            # reset() and stable-origin settling intentionally do not advance
            # AS013 semantics, so the substep latch above remains cleared.  Do
            # a read-only truth check for reset info/observations; never copy it
            # into episode state.  Once the origin is accepted, action-phase
            # substeps own this validity flag and missing pose truth stays
            # fail-closed.
            try:
                current_body_pose = np.asarray(
                    self._physics_body_pose("E_body_1"),
                    dtype=np.float64,
                ).reshape(-1)
            except Exception:  # noqa: BLE001 - malformed physics readback.
                current_body_pose = np.zeros(0, dtype=np.float64)
            body_pose_report_valid = bool(
                current_body_pose.size >= 7
                and np.isfinite(current_body_pose[:7]).all()
            )
        failed = bool(
            self._as013_failure_reason
            or self._as013_wrong_order_failure
            or self._as013_unsafe_blade_contact_failure
            or self._as013_nonfinite_failure
            or self._as013_displacement_failure
            or getattr(self, "_as013_drop_failure", False)
            or (
                ROLE_SCHEMA_VERSION < 4
                and getattr(
                    self, "_as013_left_hand_participation_failure", False
                )
            )
            or not body_pose_report_valid
            or not contact_force_values_valid
            or not self._as013_contact_views_ready
            or not self._as013_contact_report_valid
        )
        metrics = {
            "as013_role_schema_version": ROLE_SCHEMA_VERSION,
            "slider_hand": SLIDER_HAND_SIDE,
            "stabilizer_hand": STABILIZER_HAND_SIDE,
            "operator_hand": "bimanual",
            "inactive_hand": INACTIVE_HAND_SIDE,
            "operation_semantics": (
                "right_hand_pickup_left_hand_touch_bidirectional_endpoint"
            ),
            "blade_position": float(position),
            "blade_progress": float(progress),
            "blade_velocity": float(velocity),
            "blade_reset_progress": float(self._as013_reset_progress),
            "target_direction": int(
                getattr(
                    self,
                    "_as013_target_direction",
                    TARGET_DIRECTION_RETRACT,
                )
            ),
            "target_endpoint": str(
                getattr(
                    self,
                    "_as013_target_endpoint",
                    TARGET_ENDPOINT_RETRACTED,
                )
            ),
            "target_progress": float(
                getattr(self, "_as013_target_progress", 0.0)
            ),
            "target_endpoint_reached": bool(
                getattr(
                    self,
                    "_as013_target_endpoint_reached_now",
                    False,
                )
            ),
            "retract_success_progress_max": float(
                RETRACT_SUCCESS_PROGRESS_MAX
            ),
            "extend_success_progress_min": float(
                EXTEND_SUCCESS_PROGRESS_MIN
            ),
            "right_grip_housing_contact_force": float(right_grip),
            "right_palm_housing_contact_force": float(right_palm),
            "right_non_thumb_housing_contact_force": float(right_non_thumb),
            "right_grip_contact_owner_count": int(
                getattr(self, "_as013_right_grip_contact_owner_count", 0)
            ),
            "right_grip_required_contact_owners": int(
                RIGHT_GRIP_REQUIRED_CONTACT_OWNERS
            ),
            "right_grip_contact_now": bool(
                getattr(self, "_as013_right_grip_contact_now", False)
            ),
            "right_grip_seen": bool(
                getattr(self, "_as013_right_grip_seen", False)
            ),
            "pickup_lift_m": float(
                getattr(self, "_as013_pickup_lift_m", 0.0)
            ),
            "pickup_lift_required_m": float(PICKUP_LIFT_HEIGHT_M),
            "pickup_candidate_substeps": int(
                getattr(self, "_as013_pickup_candidate_substeps", 0)
            ),
            "pickup_required_substeps": int(PICKUP_CONSECUTIVE_SUBSTEPS),
            "pickup_seen": bool(getattr(self, "_as013_pickup_seen", False)),
            "knife_lifted_now": bool(
                getattr(self, "_as013_knife_lifted_now", False)
            ),
            "pickup_peak_lift_m": float(
                getattr(self, "_as013_pickup_peak_lift_m", 0.0)
            ),
            "right_thumb_moving_contact_force": float(
                right_thumb_moving if ROLE_SCHEMA_VERSION == 3 else 0.0
            ),
            "right_thumb_net_contact_force": float(
                right_thumb_net_force if ROLE_SCHEMA_VERSION == 3 else 0.0
            ),
            "right_thumb_slider_contact_force": float(
                right_thumb_slider if ROLE_SCHEMA_VERSION == 3 else 0.0
            ),
            "right_thumb_outside_slider_contact_force": float(
                right_thumb_outside_slider if ROLE_SCHEMA_VERSION == 3 else 0.0
            ),
            "right_thumb_outside_slider_diagnostic_force": float(
                right_thumb_outside_slider_diagnostic
                if ROLE_SCHEMA_VERSION == 3
                else 0.0
            ),
            "right_thumb_region_partition_consistent": bool(
                ROLE_SCHEMA_VERSION >= 4
                or legacy_force_tuple
                or (
                    np.isfinite(right_thumb_outside_slider_diagnostic)
                    and math.isclose(
                        right_thumb_outside_slider,
                        right_thumb_outside_slider_diagnostic,
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-9,
                    )
                )
            ),
            "right_thumb_outside_slider_contact_row_count": int(
                getattr(
                    self,
                    "_as013_right_outside_slider_contact_row_count",
                    0,
                )
                if ROLE_SCHEMA_VERSION == 3
                else 0
            ),
            "right_thumb_slider_contact_now": bool(
                ROLE_SCHEMA_VERSION == 3
                and
                self._as013_contact_report_valid
                and right_thumb_slider > CONTACT_FORCE_THRESHOLD_N
            ),
            "right_thumb_slider_contact_seen": bool(
                ROLE_SCHEMA_VERSION == 3
                and self._as013_right_slider_contact_seen
            ),
            "right_unsafe_moving_contact_force": float(right_unsafe_total),
            "right_incidental_moving_contact_force": float(
                right_unsafe_total
            ),
            "right_incidental_moving_contact_now": bool(
                self._as013_contact_report_valid
                and right_unsafe_total > CONTACT_FORCE_THRESHOLD_N
            ),
            "right_incidental_moving_contact_terminal": False,
            "left_housing_contact_force": float(left_housing),
            "left_moving_contact_force": float(left_moving),
            "left_forbidden_contact_force": float(left_forbidden_force),
            "left_slider_moving_contact_force": float(right_thumb_moving),
            "left_slider_net_contact_force": float(right_thumb_net_force),
            "left_slider_contact_force": float(right_thumb_slider),
            "left_slider_outside_button_contact_force": float(
                right_thumb_outside_slider
            ),
            "left_slider_outside_button_diagnostic_force": float(
                right_thumb_outside_slider_diagnostic
            ),
            "left_slider_outside_button_contact_row_count": int(
                getattr(
                    self,
                    "_as013_right_outside_slider_contact_row_count",
                    0,
                )
            ),
            "left_slider_region_partition_consistent": bool(
                legacy_force_tuple
                or (
                    np.isfinite(right_thumb_outside_slider_diagnostic)
                    and math.isclose(
                        right_thumb_outside_slider,
                        right_thumb_outside_slider_diagnostic,
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-9,
                    )
                )
            ),
            "left_slider_contact_now": bool(
                getattr(
                    self,
                    "_as013_slider_contact_effective_now",
                    self._as013_contact_report_valid
                    and right_thumb_slider > CONTACT_FORCE_THRESHOLD_N,
                )
            ),
            "left_slider_contact_raw_now": bool(
                self._as013_contact_report_valid
                and right_thumb_slider > CONTACT_FORCE_THRESHOLD_N
            ),
            "left_slider_contact_observation_substeps": int(
                getattr(
                    self,
                    "_as013_slider_contact_observation_substeps",
                    0,
                )
            ),
            "left_slider_contact_dropout_substeps": int(
                getattr(
                    self,
                    "_as013_slider_contact_dropout_substeps",
                    0,
                )
            ),
            "left_slider_contact_dropout_grace_substeps": int(
                SLIDER_CONTACT_DROPOUT_GRACE_SUBSTEPS
            ),
            "left_hand_participation_substeps": int(
                getattr(self, "_as013_left_hand_participation_substeps", 0)
            ),
            "left_hand_participation_required_substeps": int(
                LEFT_HAND_PARTICIPATION_CONSECUTIVE_SUBSTEPS
            ),
            "left_hand_participation_failure": bool(
                getattr(
                    self, "_as013_left_hand_participation_failure", False
                )
            ),
            "drop_substeps": int(getattr(self, "_as013_drop_substeps", 0)),
            "drop_required_substeps": int(DROP_CONSECUTIVE_SUBSTEPS),
            "drop_failure": bool(getattr(self, "_as013_drop_failure", False)),
            "drop_near_table_now": bool(
                getattr(self, "_as013_drop_near_table_now", False)
            ),
            "drop_peak_descent_now": bool(
                getattr(self, "_as013_drop_peak_descent_now", False)
            ),
            "drop_falling_now": bool(
                getattr(self, "_as013_drop_falling_now", False)
            ),
            "drop_table_impact_now": bool(
                getattr(self, "_as013_drop_table_impact_now", False)
            ),
            "drop_evidence_now": bool(
                getattr(self, "_as013_drop_evidence_now", False)
            ),
            "drop_detection": (
                "debounced_post_pickup_drop_with_downward_evidence"
            ),
            "drop_table_contact_now": bool(
                getattr(self, "_as013_table_contact_substeps", 0) > 0
            ),
            "drop_table_contact_substeps": int(
                getattr(self, "_as013_table_contact_substeps", 0)
            ),
            "drop_table_contact_event_count": int(
                getattr(self, "_as013_table_contact_event_count", 0)
            ),
            "drop_second_table_contact_seen": bool(
                getattr(self, "_as013_second_table_contact_seen", False)
            ),
            "drop_table_contact_force_threshold_n": float(
                DROP_TABLE_CONTACT_FORCE_THRESHOLD_N
            ),
            "drop_table_contact_required_substeps": int(
                DROP_TABLE_CONTACT_CONSECUTIVE_SUBSTEPS
            ),
            "right_grip_contact_region": "full_right_hand_on_housing",
            "right_palm_contact_required": False,
            "drop_return_lift_height_m": float(DROP_RETURN_LIFT_HEIGHT_M),
            "drop_descent_from_peak_required_m": float(
                DROP_DESCENT_FROM_PEAK_M
            ),
            "drop_falling_vertical_speed_threshold_mps": float(
                DROP_FALLING_VERTICAL_SPEED_MPS
            ),
            "drop_table_impact_force_threshold_n": float(
                DROP_TABLE_IMPACT_FORCE_THRESHOLD_N
            ),
            "body_pose_report_valid": bool(
                body_pose_report_valid
            ),
            "body_velocity_report_valid": bool(
                getattr(self, "_as013_body_velocity_report_valid", False)
            ),
            "body_vertical_velocity_mps": float(
                getattr(self, "_as013_body_vertical_velocity_mps", math.nan)
            ),
            "housing_table_contact_force": float(housing_table),
            "moving_table_contact_force": float(moving_table),
            "both_parts_off_table": bool(
                self._as013_contact_report_valid
                and housing_table <= CONTACT_FORCE_THRESHOLD_N
                and moving_table <= CONTACT_FORCE_THRESHOLD_N
            ),
            # Explicit legacy aliases retained for v1/v2 recordings.
            "right_body_contact_force": float(right_grip),
            "left_blade_contact_force": float(right_thumb_net_force),
            "left_slider_contact_rows_valid": bool(
                getattr(
                    self,
                    "_as013_left_contact_rows_valid",
                    getattr(self, "_as013_right_contact_rows_valid", False),
                )
            ),
            "left_blade_contact_row_count": int(
                getattr(
                    self,
                    "_as013_left_blade_contact_row_count",
                    getattr(self, "_as013_right_contact_row_count", 0),
                )
            ),
            "left_slider_contact_row_count": int(
                getattr(
                    self,
                    "_as013_left_slider_contact_row_count",
                    getattr(
                        self,
                        "_as013_right_slider_contact_row_count",
                        0,
                    ),
                )
            ),
            "unsafe_right_moving_component_contact_force": float(
                right_unsafe_total
            ),
            "left_housing_release_contact_force": float(
                left_housing
            ),
            "left_moving_release_contact_force": float(
                left_moving
            ),
            "left_release_contact_report_valid": bool(
                getattr(
                    self,
                    "_as013_left_release_contact_report_valid",
                    False,
                )
            ),
            "table_support_contact_force": float(table_support),
            "right_stable_contact_now": bool(
                self._as013_left_stable_contact_now
            ),
            "right_stabilization_substeps": int(
                self._as013_left_stabilization_substeps
            ),
            "right_stabilization_required_substeps": int(
                LEFT_STABILIZATION_CONSECUTIVE_SUBSTEPS
            ),
            "right_stabilized_seen": bool(self._as013_left_stabilized_seen),
            "left_slider_contact_substeps": int(
                self._as013_right_slider_contact_substeps
            ),
            "left_slider_contact_required_substeps": int(
                RIGHT_SLIDER_CONTACT_CONSECUTIVE_SUBSTEPS
            ),
            "left_slider_contact_seen": bool(
                self._as013_right_slider_contact_seen
            ),
            "preorder_slider_contact_substeps": int(
                getattr(self, "_as013_preorder_slider_contact_substeps", 0)
            ),
            "preorder_slider_contact_required_substeps": int(
                ORDER_VIOLATION_CONSECUTIVE_SUBSTEPS
            ),
            "prestabilization_progress_substeps": int(
                getattr(
                    self,
                    "_as013_prestabilization_progress_substeps",
                    0,
                )
            ),
            "prestabilization_progress_required_substeps": int(
                PRESTABILIZATION_PROGRESS_CONSECUTIVE_SUBSTEPS
            ),
            "prepickup_slider_progress_guard_active": bool(
                getattr(
                    self,
                    "_as013_prestabilization_progress_substeps",
                    0,
                )
                > 0
                and not getattr(self, "_as013_pickup_seen", False)
            ),
            "prepickup_slider_progress_terminal": False,
            "unsafe_moving_component_contact_substeps": int(
                getattr(self, "_as013_unsafe_blade_contact_substeps", 0)
            ),
            "unsafe_moving_component_contact_required_substeps": int(
                UNSAFE_BLADE_CONTACT_CONSECUTIVE_SUBSTEPS
            ),
            "ordered_contact_progress": (
                float(self._as013_ordered_contact_progress)
                if self._as013_ordered_contact_progress is not None
                else math.nan
            ),
            "slide_progress_now": bool(self._as013_slide_progress_now),
            "slide_progress_substeps": int(
                self._as013_slide_progress_substeps
            ),
            "slide_progress_required_substeps": int(
                SLIDE_PROGRESS_CONSECUTIVE_SUBSTEPS
            ),
            "slide_progress_seen": bool(self._as013_slide_progress_seen),
            "completion_forward_progress_now": bool(
                getattr(
                    self,
                    "_as013_completion_forward_progress_now",
                    False,
                )
            ),
            "completion_target_progress_now": bool(
                getattr(
                    self,
                    "_as013_completion_target_progress_now",
                    False,
                )
            ),
            "ordered_slide_completion_seen": bool(
                getattr(
                    self,
                    "_as013_ordered_slide_completion_seen",
                    False,
                )
            ),
            "hands_released": bool(
                self._as013_contact_report_valid
                and right_grip <= CONTACT_FORCE_THRESHOLD_N
                and right_thumb_moving <= CONTACT_FORCE_THRESHOLD_N
                and right_unsafe_total <= CONTACT_FORCE_THRESHOLD_N
                and left_housing <= CONTACT_FORCE_THRESHOLD_N
                and left_moving <= CONTACT_FORCE_THRESHOLD_N
            ),
            "table_support_contact": bool(
                self._as013_contact_report_valid
                and table_support > CONTACT_FORCE_THRESHOLD_N
            ),
            "stable": stable,
            "held_stable_now": bool(
                stable
                and self._as013_contact_report_valid
                and getattr(self, "_as013_right_grip_contact_now", False)
                and getattr(self, "_as013_knife_lifted_now", False)
                and (
                    ROLE_SCHEMA_VERSION >= 4
                    or left_forbidden_force <= CONTACT_FORCE_THRESHOLD_N
                )
            ),
            "wrong_order_failure": bool(self._as013_wrong_order_failure),
            "unsafe_moving_component_contact_failure": bool(
                self._as013_unsafe_blade_contact_failure
            ),
            "nonfinite_failure": bool(self._as013_nonfinite_failure),
            "displacement_failure": bool(self._as013_displacement_failure),
            "contact_views_ready": bool(self._as013_contact_views_ready),
            "contact_report_valid": bool(self._as013_contact_report_valid),
            "drive_assist_active": bool(self._as013_drive_assist_active),
            "drive_target": float(self._as013_drive_target),
            "blade_drive_stiffness": float(BLADE_JOINT_DRIVE_STIFFNESS),
            "blade_drive_damping": float(BLADE_JOINT_DRIVE_DAMPING),
            "blade_drive_max_force": float(BLADE_JOINT_DRIVE_MAX_FORCE),
            "blade_joint_static_friction": float(BLADE_JOINT_FRICTION),
            "blade_joint_dynamic_friction": float(
                BLADE_JOINT_DYNAMIC_FRICTION
            ),
            "blade_drive_stiffness_scale": float(
                getattr(self, "_as013_blade_drive_stiffness_scale", 1.0)
            ),
            "blade_drive_stiffness_scaled": bool(
                getattr(self, "_as013_blade_drive_stiffness_scaled", False)
            ),
            "blade_drive_damping_scale": float(
                getattr(self, "_as013_blade_drive_damping_scale", 1.0)
            ),
            "blade_drive_damping_scaled": bool(
                getattr(self, "_as013_blade_drive_damping_scaled", False)
            ),
            "blade_joint_friction_scale": float(
                getattr(self, "_as013_blade_joint_friction_scale", 1.0)
            ),
            "blade_joint_friction_scaled": bool(
                getattr(self, "_as013_blade_joint_friction_scaled", False)
            ),
            "slider_ik_dls_lambda": float(
                getattr(self, "_as013_right_ik_dls_lambda", 0.04)
            ),
            "causal_drive_enabled": bool(
                getattr(self, "_as013_causal_drive_enabled", False)
            ),
            "causal_drive_contact_finger_number": int(
                getattr(
                    self,
                    "_as013_causal_drive_contact_finger_number",
                    -1,
                )
                if getattr(
                    self,
                    "_as013_causal_drive_contact_finger_number",
                    None,
                )
                is not None
                else -1
            ),
            "causal_drive_contact_joint_position": float(
                getattr(
                    self,
                    "_as013_causal_drive_contact_joint_position",
                    math.nan,
                )
                if getattr(
                    self,
                    "_as013_causal_drive_contact_joint_position",
                    None,
                )
                is not None
                else math.nan
            ),
            "causal_drive_forward_displacement_m": float(
                getattr(
                    self,
                    "_as013_causal_drive_forward_displacement_m",
                    0.0,
                )
            ),
            "causal_drive_fingertip_forward_m": float(
                getattr(
                    self,
                    "_as013_causal_drive_fingertip_forward_m",
                    0.0,
                )
            ),
            "causal_drive_command_forward_m": float(
                getattr(
                    self,
                    "_as013_causal_drive_command_forward_m",
                    0.0,
                )
            ),
            "right_thumb_slide_command": float(
                self._as013_right_slide_command
            ),
            "left_slide_command": float(self._as013_right_slide_command),
            "hands_released_legacy_only": True,
            "table_support_contact_legacy_only": True,
            "legacy_role_aliases": {
                "left_body_contact_force": "right_grip_housing_contact_force",
                "right_body_contact_force": "right_grip_housing_contact_force",
                "right_thumb_slider_contact_force": (
                    "sealed_role_schema_v3_only"
                ),
                "right_slider_contact_force": "left_slider_contact_force",
                "left_stabilized_seen": "right_grip_seen",
                "right_stabilized_seen": "right_grip_seen",
                "right_slider_contact_seen": "left_slider_contact_seen",
            },
            "as013_failure_reason": str(self._as013_failure_reason),
            # Shared B320 recorder/status extraction reads this family-level
            # key before falling back to generic terminal text.
            "b320_failure_reason": str(self._as013_failure_reason),
            "as013_failure_reason_legacy": _legacy_failure_reason(
                self._as013_failure_reason
            ),
            "failed": failed,
        }
        # Preserve the existing public predicates while adding an explicit
        # speed/contact diagnostic block.  The helper's duplicate canonical
        # fields are reasserted below so legacy aliases and replay records
        # remain byte/semantic compatible.
        metrics.update(stability_diagnostics)
        metrics.update(
            {
                "stable": stable,
                "contact_report_valid": bool(
                    self._as013_contact_report_valid
                ),
                "right_grip_contact_now": bool(
                    getattr(self, "_as013_right_grip_contact_now", False)
                ),
                "knife_lifted_now": bool(
                    getattr(self, "_as013_knife_lifted_now", False)
                ),
                "left_slider_contact_force": float(right_thumb_slider),
            }
        )
        final_hold_completion_ready = (
            _v4_completion_ready_without_final_hold(metrics)
        )
        final_hold_raw_now = _v4_raw_final_hold_from_metrics(metrics)
        final_hold_with_grace_now = bool(
            final_hold_completion_ready
            and (
                final_hold_raw_now
                or getattr(self, "_as013_final_hold_with_grace_now", False)
            )
        )
        metrics.update(
            {
                "final_hold_completion_ready": final_hold_completion_ready,
                "final_hold_raw_now": final_hold_raw_now,
                "final_hold_with_grace_now": final_hold_with_grace_now,
                "final_hold_grace_active": bool(
                    final_hold_completion_ready
                    and not final_hold_raw_now
                    and getattr(
                        self,
                        "_as013_final_hold_grace_active",
                        False,
                    )
                ),
                "final_hold_dropout_control_frames": int(
                    getattr(
                        self,
                        "_as013_final_hold_dropout_control_frames",
                        0,
                    )
                    if final_hold_completion_ready
                    else 0
                ),
                "final_hold_dropout_grace_control_frames": int(
                    FINAL_HOLD_DROPOUT_GRACE_CONTROL_FRAMES
                ),
            }
        )
        # Stable aliases preserve v1/v2 recordings and callers.  Each value
        # mirrors canonical v3 physical evidence; no old predicate is used by
        # v3 success.
        metrics.update(
            {
                "left_body_contact_force": metrics[
                    "right_body_contact_force"
                ],
                "right_blade_contact_force": metrics[
                    "left_blade_contact_force"
                ],
                "right_slider_contact_force": metrics[
                    "left_slider_contact_force"
                ],
                "right_slider_contact_rows_valid": metrics[
                    "left_slider_contact_rows_valid"
                ],
                "right_blade_contact_row_count": metrics[
                    "left_blade_contact_row_count"
                ],
                "right_slider_contact_row_count": metrics[
                    "left_slider_contact_row_count"
                ],
                "unsafe_right_blade_contact_force": metrics[
                    "unsafe_right_moving_component_contact_force"
                ],
                "unsafe_left_blade_contact_force": metrics[
                    "unsafe_right_moving_component_contact_force"
                ],
                "unsafe_blade_contact_substeps": metrics[
                    "unsafe_moving_component_contact_substeps"
                ],
                "unsafe_blade_contact_required_substeps": metrics[
                    "unsafe_moving_component_contact_required_substeps"
                ],
                "unsafe_blade_contact_failure": metrics[
                    "unsafe_moving_component_contact_failure"
                ],
                "left_stable_contact_now": metrics[
                    "right_stable_contact_now"
                ],
                "left_stabilization_substeps": metrics[
                    "right_stabilization_substeps"
                ],
                "left_stabilization_required_substeps": metrics[
                    "right_stabilization_required_substeps"
                ],
                "left_stabilized_seen": metrics[
                    "right_stabilized_seen"
                ],
                "right_slider_contact_substeps": metrics[
                    "left_slider_contact_substeps"
                ],
                "right_slider_contact_required_substeps": metrics[
                    "left_slider_contact_required_substeps"
                ],
                "right_slider_contact_seen": metrics[
                    "left_slider_contact_seen"
                ],
                "right_ik_dls_lambda": metrics["slider_ik_dls_lambda"],
                "right_slide_command": metrics["right_thumb_slide_command"],
            }
        )
        return metrics

    def _compute_failure(self) -> bool:
        metrics = self._as013_last_metrics or self._as013_metrics()
        failed = bool(metrics["failed"] or super()._compute_failure())
        if failed and not self._as013_failure_reported:
            self._as013_failure_reported = True
            unsafe_right_contact_filter_row_counts = (
                self._as013_contact_filter_counts(
                    self._as013_right_unsafe_moving_contact_view,
                    getattr(self, "_as013_right_unsafe_filter_paths", ()),
                )
            )
            left_forbidden_contact_filter_row_counts = {
                "housing": self._as013_contact_filter_counts(
                    self._as013_left_housing_contact_view,
                    self._as013_left_release_filter_paths,
                ),
                "moving": self._as013_contact_filter_counts(
                    self._as013_left_moving_contact_view,
                    self._as013_left_release_filter_paths,
                ),
            }
            print(
                "[HandsOn/AS013] first physical failure: "
                f"{metrics}; unsafe_right_contact_filter_row_counts="
                f"{unsafe_right_contact_filter_row_counts}; "
                "left_forbidden_contact_filter_row_counts="
                f"{left_forbidden_contact_filter_row_counts}; "
                "stability_diagnosis="
                f"{'; '.join(metrics.get('held_stability_failure_reasons', ())) or 'none'}",
                flush=True,
            )
        return failed

    def _compute_success_now(self) -> bool:
        metrics = self._as013_metrics()
        if int(metrics.get("as013_role_schema_version", 1)) >= 5:
            self._as013_final_hold_valid_seen = False
            self._as013_final_hold_dropout_control_frames = 0
            self._as013_final_hold_grace_active = False
            self._as013_final_hold_with_grace_now = False
            metrics.update(
                {
                    "final_hold_completion_ready": _v5_completion_ready(
                        metrics
                    ),
                    "final_hold_raw_now": False,
                    "final_hold_with_grace_now": False,
                    "final_hold_grace_active": False,
                    "final_hold_dropout_control_frames": 0,
                    "final_hold_dropout_grace_control_frames": 0,
                }
            )
            self._as013_last_metrics = metrics
            return _v5_completion_ready(metrics)
        if int(metrics.get("as013_role_schema_version", 1)) >= 4:
            completion_ready = _v4_completion_ready_without_final_hold(
                metrics
            )
            raw_final_hold_now = _v4_raw_final_hold_from_metrics(metrics)
            (
                hold_window_retained_now,
                dropout_control_frames,
                grace_active,
                valid_hold_seen,
            ) = _advance_final_hold_dropout_grace_frames(
                completion_ready=completion_ready,
                raw_final_hold_now=raw_final_hold_now,
                valid_hold_seen=bool(
                    getattr(self, "_as013_final_hold_valid_seen", False)
                ),
                dropout_control_frames=int(
                    getattr(
                        self,
                        "_as013_final_hold_dropout_control_frames",
                        0,
                    )
                ),
            )
            self._as013_final_hold_valid_seen = valid_hold_seen
            self._as013_final_hold_dropout_control_frames = (
                dropout_control_frames
            )
            self._as013_final_hold_grace_active = grace_active
            self._as013_final_hold_with_grace_now = (
                hold_window_retained_now
            )
            metrics.update(
                {
                    "final_hold_completion_ready": completion_ready,
                    "final_hold_raw_now": raw_final_hold_now,
                    "final_hold_with_grace_now": (
                        hold_window_retained_now
                    ),
                    "final_hold_grace_active": grace_active,
                    "final_hold_dropout_control_frames": (
                        dropout_control_frames
                    ),
                    "final_hold_dropout_grace_control_frames": int(
                        FINAL_HOLD_DROPOUT_GRACE_CONTROL_FRAMES
                    ),
                }
            )
            self._as013_last_metrics = metrics
            return hold_window_retained_now
        self._as013_final_hold_valid_seen = False
        self._as013_final_hold_dropout_control_frames = 0
        self._as013_final_hold_grace_active = False
        self._as013_final_hold_with_grace_now = False
        self._as013_last_metrics = metrics
        return success_from_metrics(metrics)

    def _compute_success(self) -> bool:
        """Apply immediate v5 completion or the sealed legacy hold window."""

        final_hold_retained_now = self._compute_success_now()
        active_schema = int(
            self._as013_last_metrics.get(
                "as013_role_schema_version",
                ROLE_SCHEMA_VERSION,
            )
        )
        if active_schema >= 5:
            self._success_counter = int(final_hold_retained_now)
            self._update_success_condition_overlay()
            return bool(final_hold_retained_now)
        required_steps = self._success_steps_required()
        if self._as013_final_hold_grace_active:
            success = False
        elif final_hold_retained_now:
            self._success_counter += 1
            success = self._success_counter >= required_steps
        else:
            self._success_counter = 0
            success = False
        self._update_success_condition_overlay()
        return success

    def _success_steps_required(self) -> int:
        """Schema v5 terminates as soon as all three event predicates hold."""

        if ROLE_SCHEMA_VERSION >= 5:
            return 1
        return super()._success_steps_required()

    def _task_info(self) -> dict[str, Any]:
        info = dict(self._as013_last_metrics or self._as013_metrics())
        info.update(
            {
                "as013_task_code": TASK_CODE,
                "as013_joint_name": JOINT_NAME,
                "as013_role_schema_version": ROLE_SCHEMA_VERSION,
                "as013_slider_hand": SLIDER_HAND_SIDE,
                "as013_stabilizer_hand": STABILIZER_HAND_SIDE,
                "as013_operator_hand": "bimanual",
                "as013_inactive_hand": INACTIVE_HAND_SIDE,
                "as013_left_slider_fingertip_rigid_bodies": list(
                    LEFT_SLIDER_FINGERTIP_RIGID_BODY_NAMES
                ),
                "as013_right_thumb_rigid_body": (
                    RIGHT_THUMB_DISTAL_RIGID_BODY_NAME
                ),
                "as013_right_thumb_tip_collider": (
                    RIGHT_THUMB_TIP_COLLIDER_NAME
                ),
                "as013_knife_interaction_state_v4_layout": [
                    "right_grip_housing_contact_force",
                    "left_slider_moving_contact_force",
                    "left_slider_contact_force",
                    "table_support_contact_force",
                    "right_grip_seen",
                    "pickup_seen",
                    "left_slider_contact_seen",
                    "slide_progress_seen",
                    "held_stable_now",
                    "failed",
                    "drive_assist_active",
                    "drive_target",
                    "ordered_slide_completion_seen",
                ],
                "as013_knife_interaction_state_v5_layout": [
                    "right_grip_housing_contact_force",
                    "left_slider_moving_contact_force",
                    "left_slider_contact_force",
                    "table_support_contact_force",
                    "right_grip_seen",
                    "pickup_seen",
                    "left_slider_contact_seen",
                    "slide_progress_seen",
                    "held_stable_now",
                    "failed",
                    "drive_assist_active",
                    "drive_target",
                    "ordered_slide_completion_seen",
                    "target_direction",
                    "target_progress",
                    "target_endpoint_reached",
                ],
                "as013_articulation_owner_discovered": bool(
                    JOINT_NAME in self._task_dof_views
                ),
                "as013_recording_origin_accepted": bool(
                    self._as013_recording_origin_accepted
                ),
                "as013_success_counter": int(self._success_counter),
                "as013_success_steps_required": int(
                    self._success_steps_required()
                ),
                "as013_spawn_contract": dict(self._as013_spawn_contract),
            }
        )
        return info

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        metrics = self._as013_last_metrics or self._as013_metrics()
        state = observation["state"]
        state["knife_body_pose"] = self._poses.get(
            "E_body_1", self._fallback_pose("E_body_1")
        ).astype(np.float64, copy=True)
        state["blade_pose"] = self._poses.get(
            "blade_2", self._fallback_pose("blade_2")
        ).astype(np.float64, copy=True)
        state["blade_joint_state"] = np.asarray(
            [
                metrics["blade_position"],
                metrics["blade_progress"],
                metrics["blade_velocity"],
                metrics["blade_reset_progress"],
            ],
            dtype=np.float64,
        )
        state["knife_interaction_state"] = np.asarray(
            [
                metrics["right_grip_housing_contact_force"],
                metrics["left_slider_moving_contact_force"],
                metrics["left_slider_contact_force"],
                metrics["table_support_contact_force"],
                float(metrics["right_grip_seen"]),
                float(metrics["pickup_seen"]),
                float(metrics["left_slider_contact_seen"]),
                float(metrics["slide_progress_seen"]),
                float(metrics["held_stable_now"]),
                float(metrics["failed"]),
                float(metrics["drive_assist_active"]),
                metrics["drive_target"],
                float(metrics["ordered_slide_completion_seen"]),
                float(metrics["target_direction"]),
                metrics["target_progress"],
                float(metrics["target_endpoint_reached"]),
            ],
            dtype=np.float64,
        )
        return observation

    def _task_status_text(self) -> str:
        metrics = self._as013_last_metrics or self._as013_metrics()
        failure = " | Failure latched" if metrics["failed"] else ""
        target = (
            "EXTEND to top"
            if metrics.get(
                "target_direction", TARGET_DIRECTION_RETRACT
            )
            == TARGET_DIRECTION_EXTEND
            else "RETRACT to bottom"
        )
        return (
            f"Target: {target} | Slider {100.0 * metrics['blade_progress']:.1f}%"
            f"{failure}"
        )

    @staticmethod
    def _success_condition_display_label(row: dict[str, Any]) -> str:
        """Keep AS013's omni.ui panel human-readable and font-safe."""

        key = str(row.get("key") or "")
        if key.startswith("as013_") and row.get("label"):
            return str(row["label"])
        return HandsOnABCIsaacLabEnv._success_condition_display_label(row)

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._as013_last_metrics or self._as013_metrics()
        target_label = (
            "Slider reaches top / blade fully extended"
            if metrics.get(
                "target_direction", TARGET_DIRECTION_RETRACT
            )
            == TARGET_DIRECTION_EXTEND
            else "Slider reaches bottom / blade fully retracted"
        )
        rows = [
            self._condition_row(
                "as013_pickup",
                "Right hand picks up the knife by at least 20 mm",
                metrics["pickup_seen"],
            ),
            self._condition_row(
                "as013_left_slider_touch",
                "Left fingertips have touched the slider after pickup",
                metrics.get("left_slider_contact_seen", False),
            ),
            self._condition_row(
                "as013_target_endpoint",
                target_label,
                metrics["ordered_slide_completion_seen"],
            ),
        ]
        return rows

    @staticmethod
    def _blend_pose(start: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
        result = np.asarray(start, dtype=np.float64).reshape(7).copy()
        result[:3] = (1.0 - fraction) * result[:3] + fraction * target[:3]
        quaternion = (1.0 - fraction) * result[3:7] + fraction * target[3:7]
        norm = float(np.linalg.norm(quaternion))
        result[3:7] = quaternion / max(norm, 1.0e-12)
        return result

    def _scripted_timeout_release_action(
        self,
        *,
        right_pose: np.ndarray,
        left_pose: np.ndarray,
        right_hand: np.ndarray,
        left_hand: np.ndarray,
        fraction: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Release from one latched fallback pose without cumulative drift."""

        if self._as013_scripted_timeout_right_pose is None:
            self._as013_scripted_timeout_right_pose = np.asarray(
                right_pose,
                dtype=np.float64,
            ).reshape(7).copy()
            self._as013_scripted_timeout_left_pose = np.asarray(
                left_pose,
                dtype=np.float64,
            ).reshape(7).copy()
            self._as013_scripted_timeout_right_hand = np.asarray(
                right_hand,
                dtype=np.float64,
            ).reshape(20).copy()
            self._as013_scripted_timeout_left_hand = np.asarray(
                left_hand,
                dtype=np.float64,
            ).reshape(20).copy()
        right_start = self._as013_scripted_timeout_right_pose.copy()
        left_start = self._as013_scripted_timeout_left_pose.copy()
        right_release = right_start.copy()
        left_release = left_start.copy()
        right_release[2] += 0.10
        left_release[2] += 0.10
        return (
            self._blend_pose(right_start, right_release, fraction),
            self._blend_pose(left_start, left_release, fraction),
            (1.0 - fraction) * self._as013_scripted_timeout_right_hand,
            (1.0 - fraction) * self._as013_scripted_timeout_left_hand,
        )

    def _scripted_single_hand_success_action(
        self, frame_index: int
    ) -> np.ndarray:
        """Use physical right-hand actions for grip, pickup, and thumb slide.

        This diagnostic policy never writes knife or articulation state.  The
        left arm remains at its captured safe home, the right palm/fingers
        acquire the housing, the commanded wrist lifts it, and only the four
        right-thumb joints change during the slide phase.  On completion the
        right hand continues holding the lifted knife; there is no release.
        """

        frame = max(0, int(frame_index))
        if self._as013_scripted_single_right_home is None:
            self._as013_scripted_single_right_home = np.asarray(
                self._right_tcp_pose, dtype=np.float64
            ).reshape(7).copy()
        if self._as013_scripted_single_left_home is None:
            self._as013_scripted_single_left_home = np.asarray(
                self._left_tcp_pose, dtype=np.float64
            ).reshape(7).copy()
        right_home = self._as013_scripted_single_right_home.copy()
        left_home = self._as013_scripted_single_left_home.copy()
        left_hand = np.zeros(20, dtype=np.float64)
        if frame == 0:
            action = np.concatenate(
                [right_home, left_home, np.zeros(20), left_hand]
            ).astype(np.float32)
            return action

        body = self._poses.get("E_body_1", self._fallback_pose("E_body_1"))
        axis = self._as013_joint_world_axis()
        grip_target = right_home.copy()
        grip_target[:3] = np.asarray(body[:3], dtype=np.float64) + 0.012 * axis
        grip_target[2] += LEFT_PALM_WORK_HEIGHT_M
        grip_hover = grip_target.copy()
        grip_hover[2] += 0.090
        thumb_ready = _right_touch_hand_posture(
            1, 0.35, 0.0
        )
        thumb_push = thumb_ready.copy()
        # Wuji hand joint order is thumb/index/middle/ring/pinky.  Keep the
        # non-thumb power grip fixed and move only the canonical thumb slice.
        thumb_push[:4] = _POWER_GRIP[:4]

        if not getattr(self, "_as013_right_grip_seen", False):
            if frame < 60:
                fraction = frame / 59.0
                right_pose = self._blend_pose(
                    right_home, grip_hover, fraction
                )
                right_hand = np.zeros(20, dtype=np.float64)
            else:
                fraction = min(1.0, (frame - 60) / 89.0)
                right_pose = self._blend_pose(
                    grip_hover, grip_target, fraction
                )
                right_hand = fraction * thumb_ready
        elif not getattr(self, "_as013_pickup_seen", False):
            if self._as013_scripted_single_grip_frame is None:
                self._as013_scripted_single_grip_frame = frame
                self._as013_scripted_single_grip_pose = np.asarray(
                    self._right_tcp_pose, dtype=np.float64
                ).reshape(7).copy()
            grip_pose = self._as013_scripted_single_grip_pose.copy()
            pickup_target = grip_pose.copy()
            pickup_target[2] += PICKUP_LIFT_HEIGHT_M + 0.015
            fraction = min(
                1.0,
                (frame - self._as013_scripted_single_grip_frame) / 59.0,
            )
            right_pose = self._blend_pose(
                grip_pose, pickup_target, fraction
            )
            right_hand = thumb_ready.copy()
        else:
            if self._as013_scripted_single_pickup_frame is None:
                self._as013_scripted_single_pickup_frame = frame
                self._as013_scripted_single_pickup_pose = np.asarray(
                    self._right_tcp_pose, dtype=np.float64
                ).reshape(7).copy()
                actual_hand = (
                    self._actual_hand_joint_pose("right")
                    if self._robots.get("right") is not None
                    and self._hand_joint_ids.get("right")
                    else thumb_ready
                )
                self._as013_scripted_single_slide_hand = np.asarray(
                    actual_hand, dtype=np.float64
                ).reshape(20).copy()
            right_pose = self._as013_scripted_single_pickup_pose.copy()
            slide_start = self._as013_scripted_single_slide_hand.copy()
            elapsed = frame - self._as013_scripted_single_pickup_frame
            # Give the physical thumb one second to settle on the button before
            # its joint-space push; the other 16 joint commands remain fixed.
            fraction = float(
                np.clip(
                    (elapsed - 30) / float(RIGHT_SLIDE_CONTROL_FRAMES),
                    0.0,
                    1.0,
                )
            )
            right_hand = slide_start.copy()
            right_hand[:4] = (
                (1.0 - fraction) * slide_start[:4]
                + fraction * thumb_push[:4]
            )

        action = np.concatenate(
            [right_pose, left_home, right_hand, left_hand]
        ).astype(np.float32)
        if action.shape != (54,) or not np.isfinite(action).all():
            raise RuntimeError(
                f"AS013 scripted single-hand action is invalid: {action.shape}"
            )
        return action

    def _scripted_bimanual_handheld_success_action(
        self, frame_index: int
    ) -> np.ndarray:
        """Diagnostic v5 policy: right pickup, left touch, sampled endpoint.

        The policy preserves the 54-value physical action ABI and never writes
        knife state.  It reuses the right-hand grip/pickup approach until the
        physical pickup latch is true, then aligns the left index fingertip to
        the live button center and follows the slider axis.  Fresh live
        calibration is still required before this helper can certify a demo.
        """

        frame = max(0, int(frame_index))
        if not getattr(self, "_as013_pickup_seen", False):
            return self._scripted_single_hand_success_action(frame)

        if self._as013_scripted_single_right_home is None:
            self._as013_scripted_single_right_home = np.asarray(
                self._right_tcp_pose, dtype=np.float64
            ).reshape(7).copy()
        if self._as013_scripted_single_left_home is None:
            self._as013_scripted_single_left_home = np.asarray(
                self._left_tcp_pose, dtype=np.float64
            ).reshape(7).copy()
        if self._as013_scripted_single_pickup_pose is None:
            self._as013_scripted_single_pickup_pose = np.asarray(
                self._right_tcp_pose, dtype=np.float64
            ).reshape(7).copy()

        right_pose = self._as013_scripted_single_pickup_pose.copy()
        right_hand = _right_touch_hand_posture(1, 0.35, 0.0)
        left_hand = _right_touch_hand_posture(2, 0.35, 0.0)

        if self._as013_scripted_bimanual_left_start_frame is None:
            self._as013_scripted_bimanual_left_start_frame = frame
            self._as013_scripted_bimanual_left_start_pose = np.asarray(
                self._left_tcp_pose, dtype=np.float64
            ).reshape(7).copy()
            blade = self._poses.get(
                "blade_2", self._fallback_pose("blade_2")
            )
            blade_matrix = _pose_vector_to_matrix(blade)
            button_world = (
                blade_matrix[:3, :3] @ BUTTON_REGION_CENTER_LOCAL_M
                + blade_matrix[:3, 3]
            )
            button_normal = blade_matrix[:3, :3] @ np.asarray(
                [0.0, 0.0, 1.0], dtype=np.float64
            )
            left_tip = self._as013_fingertip_point("left", 2)
            tip_offset = (
                np.asarray(left_tip, dtype=np.float64)
                - self._as013_scripted_bimanual_left_start_pose[:3]
                if left_tip is not None
                else np.asarray([0.0, 0.0, -0.08], dtype=np.float64)
            )
            target = self._as013_scripted_bimanual_left_start_pose.copy()
            target[:3] = button_world + 0.002 * button_normal - tip_offset
            self._as013_scripted_bimanual_left_target_pose = target

        elapsed = frame - self._as013_scripted_bimanual_left_start_frame
        approach_fraction = float(np.clip(elapsed / 89.0, 0.0, 1.0))
        left_pose = self._blend_pose(
            self._as013_scripted_bimanual_left_start_pose,
            self._as013_scripted_bimanual_left_target_pose,
            approach_fraction,
        )
        if elapsed > 90:
            slide_fraction = float(
                np.clip(
                    (elapsed - 90) / float(RIGHT_SLIDE_CONTROL_FRAMES),
                    0.0,
                    1.0,
                )
            )
            target_direction = float(
                getattr(
                    self,
                    "_as013_target_direction",
                    TARGET_DIRECTION_RETRACT,
                )
            )
            left_pose[:3] += (
                slide_fraction
                * 0.080
                * target_direction
                * self._as013_joint_world_axis()
            )

        action = np.concatenate(
            [right_pose, left_pose, right_hand, left_hand]
        ).astype(np.float32)
        if action.shape != (54,) or not np.isfinite(action).all():
            raise RuntimeError(
                f"AS013 scripted bimanual action is invalid: {action.shape}"
            )
        return action

    def _scripted_success_action_v2_legacy_unreachable(
        self, frame_index: int
    ) -> np.ndarray:
        """Retain sealed v2 diagnostics without exposing them to v3 runtime."""

        if ROLE_SCHEMA_VERSION >= 3:
            raise RuntimeError(
                "AS013 v2 bimanual scripted policy is forbidden by role schema v3"
            )

        frame = max(0, int(frame_index))
        body = self._poses.get("E_body_1", self._fallback_pose("E_body_1"))
        blade = self._poses.get("blade_2", self._fallback_pose("blade_2"))
        axis = self._as013_joint_world_axis()
        left_work_height = float(
            os.environ.get(
                SCRIPTED_LEFT_HEIGHT_ENV,
                LEFT_PALM_WORK_HEIGHT_M,
            )
        )
        left_counter_slide_m = float(
            os.environ.get(
                SCRIPTED_LEFT_COUNTER_SLIDE_ENV,
                0.0,
            )
        )
        left_hand_preload_fraction = float(
            os.environ.get(
                SCRIPTED_LEFT_HAND_PRELOAD_ENV,
                LEFT_HOLD_HAND_PRELOAD_FRACTION,
            )
        )
        left_safe_finger_preload_fraction = float(
            os.environ.get(
                SCRIPTED_LEFT_SAFE_FINGER_PRELOAD_ENV,
                LEFT_SAFE_FINGER_PRELOAD_FRACTION,
            )
        )
        left_unsafe_finger_retraction_target_fraction = float(
            os.environ.get(
                SCRIPTED_LEFT_UNSAFE_FINGER_RETRACTION_ENV,
                LEFT_UNSAFE_FINGER_RETRACTION_FRACTION,
            )
        )
        left_unsafe_finger_distal_only = (
            os.environ.get(
                SCRIPTED_LEFT_UNSAFE_FINGER_DISTAL_ONLY_ENV,
                "1" if LEFT_UNSAFE_FINGER_DISTAL_ONLY else "0",
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
        left_approach_grip_fraction = float(
            os.environ.get(
                SCRIPTED_LEFT_APPROACH_GRIP_ENV,
                LEFT_APPROACH_GRIP_FRACTION,
            )
        )
        self._as013_scripted_left_fallback_grip_active = (
            _left_fallback_grip_should_activate(
                frame,
                self._as013_left_stabilized_seen,
                self._as013_scripted_left_fallback_grip_active,
                SCRIPTED_LEFT_APPROACH_GRIP_ENV in os.environ,
            )
        )
        if self._as013_scripted_left_fallback_grip_active:
            left_approach_grip_fraction = (
                LEFT_FALLBACK_APPROACH_GRIP_FRACTION
            )
        right_work_height = float(
            os.environ.get(
                SCRIPTED_RIGHT_HEIGHT_ENV,
                RIGHT_PALM_WORK_HEIGHT_M,
            )
        )
        right_finger_extension_fraction = float(
            os.environ.get(
                SCRIPTED_RIGHT_FINGER_EXTENSION_ENV,
                RIGHT_FINGER_EXTENSION_FRACTION,
            )
        )
        right_finger_abduction_rad = float(
            os.environ.get(
                SCRIPTED_RIGHT_FINGER_ABDUCTION_ENV,
                RIGHT_FINGER_ABDUCTION_RAD,
            )
        )
        requested_right_finger_number = int(
            os.environ.get(
                SCRIPTED_RIGHT_FINGER_NUMBER_ENV,
                str(RIGHT_FINGER_NUMBER),
            )
        )
        right_vertical_gain = float(
            os.environ.get(
                SCRIPTED_RIGHT_VERTICAL_GAIN_ENV,
                RIGHT_VERTICAL_GAIN,
            )
        )
        right_lateral_gain = float(
            os.environ.get(
                SCRIPTED_RIGHT_LATERAL_GAIN_ENV,
                RIGHT_LATERAL_GAIN,
            )
        )
        right_palm_yaw_deg = float(
            os.environ.get(
                SCRIPTED_RIGHT_PALM_YAW_ENV,
                RIGHT_PALM_YAW_DEG,
            )
        )
        right_palm_roll_deg = float(
            os.environ.get(
                SCRIPTED_RIGHT_PALM_ROLL_ENV,
                RIGHT_PALM_ROLL_DEG,
            )
        )
        right_alignment_reference_blend = float(
            os.environ.get(
                SCRIPTED_RIGHT_ALIGNMENT_REFERENCE_BLEND_ENV,
                SLIDER_ALIGNMENT_REFERENCE_BLEND_DEFAULT,
            )
        )
        right_fallback_target_adapt_frame = int(
            os.environ.get(
                SCRIPTED_RIGHT_FALLBACK_TARGET_ADAPT_FRAME_ENV,
                0,
            )
        )
        right_fallback_alignment_lock_frame = int(
            os.environ.get(
                SCRIPTED_RIGHT_FALLBACK_ALIGNMENT_LOCK_FRAME_ENV,
                0,
            )
        )
        right_slide_palm_travel_m = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_PALM_TRAVEL_ENV,
                RIGHT_SLIDE_PALM_TRAVEL_DEFAULT_M,
            )
        )
        right_slide_max_lead_m = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_MAX_LEAD_ENV,
                RIGHT_SLIDE_MAX_LEAD_DEFAULT_M,
            )
        )
        right_slide_normal_preload_m = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_NORMAL_PRELOAD_ENV,
                RIGHT_SLIDE_NORMAL_PRELOAD_M,
            )
        )
        right_slide_late_normal_preload_m = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_LATE_NORMAL_PRELOAD_ENV,
                RIGHT_SLIDE_LATE_NORMAL_PRELOAD_M,
            )
        )
        right_surface_slow_radius_m = float(
            os.environ.get(
                SCRIPTED_RIGHT_SURFACE_SLOW_RADIUS_ENV,
                RIGHT_SURFACE_SLOW_RADIUS_M,
            )
        )
        right_slide_cross_edge_max_correction_m = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_CROSS_EDGE_MAX_CORRECTION_ENV,
                RIGHT_SLIDE_CROSS_EDGE_MAX_CORRECTION_M,
            )
        )
        right_slide_cross_edge_servo_scale = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_CROSS_EDGE_SERVO_SCALE_ENV,
                0.0,
            )
        )
        right_slide_yaw_relief_deg = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_YAW_RELIEF_ENV,
                0.0,
            )
        )
        right_slide_pitch_relief_deg = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_PITCH_RELIEF_ENV,
                0.0,
            )
        )
        right_slide_roll_relief_deg = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_ROLL_RELIEF_ENV,
                RIGHT_SLIDE_ROLL_RELIEF_DEFAULT_DEG,
            )
        )
        right_slide_finger_extension_delta = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_FINGER_EXTENSION_DELTA_ENV,
                RIGHT_SLIDE_FINGER_EXTENSION_DELTA_DEFAULT,
            )
        )
        right_slide_late_finger_extension_delta = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_LATE_FINGER_EXTENSION_DELTA_ENV,
                0.0,
            )
        )
        right_slide_finger_abduction_delta_rad = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_FINGER_ABDUCTION_DELTA_ENV,
                RIGHT_SLIDE_FINGER_ABDUCTION_DELTA_RAD_DEFAULT,
            )
        )
        right_slide_control_frames = int(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_CONTROL_FRAMES_ENV,
                RIGHT_SLIDE_CONTROL_FRAMES_DEFAULT,
            )
        )
        right_slide_reanchor_progress = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_REANCHOR_PROGRESS_ENV,
                0.0,
            )
        )
        right_slide_reanchor_extra_lead_m = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_REANCHOR_EXTRA_LEAD_ENV,
                0.0,
            )
        )
        right_slide_servo_correction_scale = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_SERVO_CORRECTION_SCALE_ENV,
                0.0,
            )
        )
        right_slide_late_servo_correction_scale = float(
            os.environ.get(
                SCRIPTED_RIGHT_SLIDE_LATE_SERVO_CORRECTION_SCALE_ENV,
                0.0,
            )
        )
        right_recovered_finger_extension_delta = float(
            os.environ.get(
                SCRIPTED_RIGHT_RECOVERED_FINGER_EXTENSION_DELTA_ENV,
                RIGHT_RECOVERED_FINGER_EXTENSION_DELTA_DEFAULT,
            )
        )
        right_recovered_late_finger_extension_delta = float(
            os.environ.get(
                SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_EXTENSION_DELTA_ENV,
                RIGHT_RECOVERED_LATE_FINGER_EXTENSION_DELTA_DEFAULT,
            )
        )
        right_recovered_late_finger_extension_start_progress = float(
            os.environ.get(
                SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS_ENV,
                RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS_DEFAULT,
            )
        )
        right_recovered_late_proximal_extension_scale = float(
            os.environ.get(
                SCRIPTED_RIGHT_RECOVERED_LATE_PROXIMAL_EXTENSION_SCALE_ENV,
                1.0,
            )
        )
        right_recovered_late_middle_extension_scale = float(
            os.environ.get(
                SCRIPTED_RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE_ENV,
                RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE_DEFAULT,
            )
        )
        right_recovered_late_distal_tip_extension_scale = float(
            os.environ.get(
                SCRIPTED_RIGHT_RECOVERED_LATE_DISTAL_TIP_EXTENSION_SCALE_ENV,
                1.0,
            )
        )
        right_recovered_late_in_plane_preload_m = float(
            os.environ.get(
                SCRIPTED_RIGHT_RECOVERED_LATE_IN_PLANE_PRELOAD_ENV,
                0.0,
            )
        )
        right_recovered_late_normal_preload_m = float(
            os.environ.get(
                SCRIPTED_RIGHT_RECOVERED_LATE_NORMAL_PRELOAD_ENV,
                0.0,
            )
        )
        right_recovered_late_finger_abduction_delta_rad = float(
            os.environ.get(
                SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_DELTA_ENV,
                RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_DELTA_RAD_DEFAULT,
            )
        )
        right_late_continuous_hand_blend = float(
            os.environ.get(
                SCRIPTED_RIGHT_LATE_CONTINUOUS_HAND_BLEND_ENV,
                0.0,
            )
        )
        right_finger_reentry_control_frames = int(
            os.environ.get(
                SCRIPTED_RIGHT_FINGER_REENTRY_CONTROL_FRAMES_ENV,
                RIGHT_FINGER_REENTRY_CONTROL_FRAMES,
            )
        )
        right_approach_timeout_frame = int(
            os.environ.get(
                SCRIPTED_RIGHT_APPROACH_TIMEOUT_FRAME_ENV,
                RIGHT_APPROACH_TIMEOUT_FRAME_DEFAULT,
            )
        )
        release_control_frames = int(
            os.environ.get(
                SCRIPTED_RELEASE_CONTROL_FRAMES_ENV,
                RIGHT_RELEASE_CONTROL_FRAMES_DEFAULT,
            )
        )
        post_success_hold_control_frames = int(
            os.environ.get(
                SCRIPTED_POST_SUCCESS_HOLD_CONTROL_FRAMES_ENV,
                RIGHT_POST_SUCCESS_HOLD_CONTROL_FRAMES,
            )
        )
        release_dwell_control_frames = int(
            os.environ.get(
                SCRIPTED_RELEASE_DWELL_CONTROL_FRAMES_ENV,
                RIGHT_RELEASE_DWELL_CONTROL_FRAMES_DEFAULT,
            )
        )
        post_success_overtravel_m = float(
            os.environ.get(
                SCRIPTED_POST_SUCCESS_OVERTRAVEL_ENV,
                RIGHT_POST_SUCCESS_OVERTRAVEL_DEFAULT_M,
            )
        )
        right_recontact_servo_step_m = float(
            os.environ.get(
                SCRIPTED_RIGHT_RECONTACT_SERVO_STEP_ENV,
                RIGHT_SURFACE_APPROACH_SERVO_STEP_M,
            )
        )
        if not (
            np.isfinite([left_work_height, right_work_height]).all()
            and 0.04 <= left_work_height <= 0.16
            and 0.04 <= right_work_height <= 0.16
            and np.isfinite(left_counter_slide_m)
            and 0.0 <= left_counter_slide_m <= 0.03
            and np.isfinite(left_hand_preload_fraction)
            and 0.0 <= left_hand_preload_fraction <= 0.50
            and np.isfinite(left_safe_finger_preload_fraction)
            and 0.0 <= left_safe_finger_preload_fraction <= 0.50
            and np.isfinite(left_unsafe_finger_retraction_target_fraction)
            and 0.0 <= left_unsafe_finger_retraction_target_fraction <= 1.0
            and np.isfinite(left_approach_grip_fraction)
            and 0.25 <= left_approach_grip_fraction <= 0.75
            and np.isfinite(right_finger_extension_fraction)
            and 0.0 <= right_finger_extension_fraction <= 1.0
            and np.isfinite(right_finger_abduction_rad)
            and -0.30 <= right_finger_abduction_rad <= 0.30
            and 0 <= requested_right_finger_number <= 5
            and np.isfinite(right_vertical_gain)
            and 1.0 <= right_vertical_gain <= 2.5
            and np.isfinite(right_lateral_gain)
            and 1.0 <= right_lateral_gain <= 3.0
            and np.isfinite(right_palm_yaw_deg)
            and -45.0 <= right_palm_yaw_deg <= 45.0
            and np.isfinite(right_palm_roll_deg)
            and -20.0 <= right_palm_roll_deg <= 20.0
            and np.isfinite(right_alignment_reference_blend)
            and 0.0 <= right_alignment_reference_blend <= 1.0
            and (
                right_fallback_target_adapt_frame == 0
                or 360 <= right_fallback_target_adapt_frame <= 900
            )
            and (
                right_fallback_alignment_lock_frame == 0
                or 360 <= right_fallback_alignment_lock_frame <= 600
            )
            and np.isfinite(right_slide_palm_travel_m)
            and 0.04
            <= right_slide_palm_travel_m
            <= RIGHT_SLIDE_PALM_TRAVEL_CONFIG_MAX_M
            and np.isfinite(right_slide_max_lead_m)
            and 0.0 <= right_slide_max_lead_m <= 0.03
            and np.isfinite(right_slide_normal_preload_m)
            and 0.0 <= right_slide_normal_preload_m <= 0.003
            and np.isfinite(right_slide_late_normal_preload_m)
            and 0.0 <= right_slide_late_normal_preload_m <= 0.0015
            and np.isfinite(right_surface_slow_radius_m)
            and 0.005 <= right_surface_slow_radius_m <= 0.03
            and np.isfinite(right_slide_cross_edge_max_correction_m)
            and 0.0
            <= right_slide_cross_edge_max_correction_m
            <= RIGHT_SLIDE_CROSS_EDGE_CONFIG_MAX_M
            and np.isfinite(right_slide_cross_edge_servo_scale)
            and 0.0 <= right_slide_cross_edge_servo_scale <= 1.0
            and np.isfinite(right_slide_yaw_relief_deg)
            and -10.0 <= right_slide_yaw_relief_deg <= 10.0
            and np.isfinite(right_slide_pitch_relief_deg)
            and -20.0 <= right_slide_pitch_relief_deg <= 20.0
            and np.isfinite(right_slide_roll_relief_deg)
            and -10.0 <= right_slide_roll_relief_deg <= 10.0
            and np.isfinite(right_slide_finger_extension_delta)
            and -0.20 <= right_slide_finger_extension_delta <= 0.20
            and np.isfinite(right_slide_late_finger_extension_delta)
            and -0.05 <= right_slide_late_finger_extension_delta <= 0.05
            and -0.20
            <= (
                right_slide_finger_extension_delta
                + right_slide_late_finger_extension_delta
            )
            <= 0.20
            and np.isfinite(right_slide_finger_abduction_delta_rad)
            and -0.10 <= right_slide_finger_abduction_delta_rad <= 0.10
            and 60
            <= right_slide_control_frames
            <= RIGHT_SLIDE_CONTROL_FRAMES_CONFIG_MAX
            and (
                right_slide_reanchor_progress == 0.0
                or (
                    0.35 <= right_slide_reanchor_progress <= 0.75
                    and abs(right_slide_yaw_relief_deg) <= 1.0e-12
                    and abs(right_slide_pitch_relief_deg) <= 1.0e-12
                    and abs(right_slide_roll_relief_deg) <= 1.0e-12
                )
            )
            and np.isfinite(right_slide_reanchor_extra_lead_m)
            and 0.0 <= right_slide_reanchor_extra_lead_m <= 0.018
            and (
                right_slide_reanchor_extra_lead_m == 0.0
                or right_slide_reanchor_progress > 0.0
            )
            and (
                right_slide_max_lead_m
                + right_slide_reanchor_extra_lead_m
                <= 0.03
            )
            and np.isfinite(right_slide_servo_correction_scale)
            and 0.0 <= right_slide_servo_correction_scale <= 1.0
            and np.isfinite(right_slide_late_servo_correction_scale)
            and 0.0 <= right_slide_late_servo_correction_scale <= 1.0
            and (
                right_slide_servo_correction_scale
                + right_slide_late_servo_correction_scale
                <= 1.0
            )
            and np.isfinite(right_recovered_finger_extension_delta)
            and 0.0
            <= right_recovered_finger_extension_delta
            <= RIGHT_RECOVERED_FINGER_EXTENSION_CONFIG_MAX
            and np.isfinite(right_recovered_late_finger_extension_delta)
            and -RIGHT_RECOVERED_LATE_FINGER_EXTENSION_CONFIG_ABS_MAX
            <= right_recovered_late_finger_extension_delta
            <= RIGHT_RECOVERED_LATE_FINGER_EXTENSION_CONFIG_ABS_MAX
            and np.isfinite(
                right_recovered_late_finger_extension_start_progress
            )
            and RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS_RANGE[0]
            <= right_recovered_late_finger_extension_start_progress
            <= RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS_RANGE[1]
            and np.isfinite(right_recovered_late_proximal_extension_scale)
            and RIGHT_RECOVERED_LATE_PROXIMAL_EXTENSION_SCALE_RANGE[0]
            <= right_recovered_late_proximal_extension_scale
            <= RIGHT_RECOVERED_LATE_PROXIMAL_EXTENSION_SCALE_RANGE[1]
            and np.isfinite(right_recovered_late_middle_extension_scale)
            and RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE_RANGE[0]
            <= right_recovered_late_middle_extension_scale
            <= RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE_RANGE[1]
            and np.isfinite(right_recovered_late_distal_tip_extension_scale)
            and RIGHT_RECOVERED_LATE_DISTAL_TIP_EXTENSION_SCALE_RANGE[0]
            <= right_recovered_late_distal_tip_extension_scale
            <= RIGHT_RECOVERED_LATE_DISTAL_TIP_EXTENSION_SCALE_RANGE[1]
            and np.isfinite(right_recovered_late_in_plane_preload_m)
            and 0.0
            <= right_recovered_late_in_plane_preload_m
            <= RIGHT_RECOVERED_LATE_IN_PLANE_PRELOAD_CONFIG_MAX_M
            and np.isfinite(right_recovered_late_normal_preload_m)
            and 0.0
            <= right_recovered_late_normal_preload_m
            <= RIGHT_RECOVERED_LATE_NORMAL_PRELOAD_CONFIG_MAX_M
            and np.isfinite(
                right_recovered_late_finger_abduction_delta_rad
            )
            and -RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_CONFIG_ABS_MAX_RAD
            <= right_recovered_late_finger_abduction_delta_rad
            <= RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_CONFIG_ABS_MAX_RAD
            and np.isfinite(right_late_continuous_hand_blend)
            and 0.0 <= right_late_continuous_hand_blend <= 1.0
            and 0.0
            <= (
                right_recovered_finger_extension_delta
                + right_recovered_late_finger_extension_delta
            )
            <= RIGHT_RECOVERED_FINGER_EXTENSION_CONFIG_MAX
            and 10 <= right_finger_reentry_control_frames <= 60
            and RIGHT_RELEASE_CONTROL_FRAMES_MIN
            <= release_control_frames
            <= RIGHT_RELEASE_CONTROL_FRAMES
            and 0
            <= post_success_hold_control_frames
            <= RIGHT_POST_SUCCESS_HOLD_CONTROL_FRAMES_MAX
            and 0
            <= release_dwell_control_frames
            <= RIGHT_RELEASE_DWELL_CONTROL_FRAMES_MAX
            and np.isfinite(post_success_overtravel_m)
            and 0.0
            <= post_success_overtravel_m
            <= RIGHT_POST_SUCCESS_OVERTRAVEL_MAX_M
            and 1600
            <= right_approach_timeout_frame
            <= _latest_approach_timeout_frame(release_control_frames)
            and (
                RIGHT_SURFACE_APPROACH_SERVO_STEP_M
                <= right_recontact_servo_step_m
                <= RIGHT_CONTACT_PRELOAD_SERVO_STEP_M
            )
        ):
            raise ValueError(
                "AS013 scripted palm heights must be finite values in "
                "[0.04, 0.16] m and the fingertip extension fraction must "
                "be in [0, 1]; left counter slide must be in [0, 0.03] m "
                "and left hand preload fraction in [0, 0.50], "
                "left approach grip fraction in [0.25, 0.75], "
                "fingertip abduction in [-0.30, 0.30] rad, "
                "right finger number in [0, 5] (zero selects automatically), "
                "right vertical gain in [1.0, 2.5], right lateral gain in "
                "[1.0, 3.0], and right palm yaw in [-45, 45] degrees with "
                "roll in [-20, 20] degrees; the bounded physical slide lead "
                "uses a fallback target adaptation frame of zero (disabled) "
                "or [360, 900], and a fallback alignment lock frame of zero "
                "(disabled) or [360, 600]; "
                "uses palm travel in [0.04, 0.12] m, and the slide lead "
                "must be in [0, 0.03] m and normal preload in [0, 0.003] m"
                "; late normal preload must be in [0, 0.0015] m"
                "; the free-space surface slow radius must be in "
                "[0.005, 0.03] m"
                " and cross-edge correction in [0, 0.012] m"
                "; cross-edge servo scale must be in [0, 1]"
                "; slide yaw relief must be in [-10, 10] degrees"
                " and slide roll relief in [-10, 10] degrees"
                " and slide finger extension delta in [-0.20, 0.20]"
                "; slide finger abduction delta must be in [-0.10, 0.10] rad"
                "; slide control frames must be in [60, 720]"
                "; slide reanchor progress must be zero (disabled) or in "
                "[0.35, 0.75] with zero slide yaw/roll relief"
                "; slide reanchor extra lead must be in [0, 0.018] m, "
                "requires reanchor, and keeps total lead at most 0.03 m"
                "; slide servo correction scale must be in [0, 1]"
                "; recovered finger extension delta must be in [0, 0.20]"
                "; late continuous hand blend must be in [0, 1]"
                "; finger reentry control frames must be in [10, 60]"
                "; release control frames must be in [15, 60], and the "
                "post-success hold must be in [0, 5] control frames; the "
                "release dwell must be in [0, 3] control frames; the "
                "post-success overtravel must be in [0, 0.0015] m; the "
                "approach timeout must leave the full release plus 20 frames "
                "before frame 1800"
                "; recontact servo step must be in [0.001, 0.006] m"
            )
        # Local variable names retain the legacy diagnostic roles: ``right``
        # means slider and ``left`` means stabilizer.  Bind them explicitly to
        # the v2 physical sides, then swap back to the base action order below.
        right_home = np.asarray(self._left_tcp_pose, dtype=np.float64).copy()
        left_home = np.asarray(self._right_tcp_pose, dtype=np.float64).copy()
        left_physical_contact_now = bool(
            self._as013_last_metrics.get("contact_report_valid", False)
            and float(
                self._as013_last_metrics.get("left_body_contact_force", 0.0)
            )
            > CONTACT_FORCE_THRESHOLD_N
        )
        left_hold = left_home.copy()
        left_hold[:3] = body[:3] + 0.012 * axis
        left_hold[2] += left_work_height
        if self._as013_scripted_left_nominal_world_rotation is None:
            self._as013_scripted_left_nominal_world_rotation = (
                self._native_end_effector_pose_matrix(
                    STABILIZER_HAND_SIDE,
                    left_home,
                )[:3, :3].copy()
            )
        left_yaw_aligned_rotation = _yaw_rotated_world_rotation(
            self._as013_scripted_left_nominal_world_rotation,
            self._as013_reset_yaw_degrees,
        )
        left_hold[3:7] = _matrix_to_quat_wxyz(left_yaw_aligned_rotation)
        left_fingertips = self._as013_fingertip_points(
            STABILIZER_HAND_SIDE
        )
        left_servo_pose: np.ndarray | None = None
        body_matrix = _pose_vector_to_matrix(body)
        nominal_left_target_world = (
            body_matrix[:3, :3] @ RIGHT_HOUSING_SURFACE_TARGET_LOCAL_M
            + body_matrix[:3, 3]
        )
        nearest_left_tip, left_hold_fingertip_index = (
            _select_left_hold_fingertip(
                left_fingertips,
                nominal_left_target_world,
                self._as013_scripted_left_hold_fingertip_index,
            )
        )
        nearest_left_tip_body_local = (
            (
                _invert_pose_matrix(body_matrix)
                @ np.append(nearest_left_tip, 1.0)
            )[:3]
            if nearest_left_tip is not None
            else None
        )
        self._as013_scripted_left_fallback_grip_active = (
            _left_fallback_grip_should_activate(
                frame,
                self._as013_left_stabilized_seen,
                self._as013_scripted_left_fallback_grip_active,
                SCRIPTED_LEFT_APPROACH_GRIP_ENV in os.environ,
                (
                    float(
                        np.linalg.norm(
                            nearest_left_tip - nominal_left_target_world
                        )
                    )
                    if nearest_left_tip is not None
                    else None
                ),
            )
        )
        if self._as013_scripted_left_fallback_grip_active:
            left_approach_grip_fraction = (
                LEFT_FALLBACK_APPROACH_GRIP_FRACTION
            )
            named_left_contact_tip = self._as013_fingertip_point(
                STABILIZER_HAND_SIDE,
                LEFT_HOUSING_CONTACT_FINGER_NUMBER,
            )
            if named_left_contact_tip is not None:
                nearest_left_tip = named_left_contact_tip.copy()
                if left_fingertips:
                    left_hold_fingertip_index = int(
                        np.argmin(
                            [
                                float(
                                    np.linalg.norm(
                                        point - named_left_contact_tip
                                    )
                                )
                                for point in left_fingertips
                            ]
                        )
                    )
                nearest_left_tip_body_local = (
                    _invert_pose_matrix(body_matrix)
                    @ np.append(nearest_left_tip, 1.0)
                )[:3]
        if (
            self._as013_scripted_left_fallback_grip_active
            and frame >= LEFT_HOUSING_SURFACE_PROJECTION_START_FRAME
            and not left_physical_contact_now
            and self._as013_scripted_left_recovery_body_relative is None
            and self._as013_scripted_left_surface_target_local is None
        ):
            self._as013_scripted_left_surface_target_local = (
                _project_left_housing_surface_target_local(
                    RIGHT_HOUSING_FALLBACK_FEATURE_TARGET_LOCAL_M,
                    nearest_left_tip_body_local,
                    x_range_m=(
                        RIGHT_HOUSING_SURFACE_TARGET_LOCAL_X_RANGE_M
                    ),
                    trace_yz_m=(
                        RIGHT_HOUSING_CONTACT_TRACE_TARGET_LOCAL_YZ_M
                    ),
                )
            )
            if left_hold_fingertip_index is not None:
                self._as013_scripted_left_hold_fingertip_index = int(
                    left_hold_fingertip_index
                )
        if (
            self._as013_scripted_left_fallback_grip_active
            and frame >= LEFT_HOUSING_FALLBACK_PRELOAD_START_FRAME
            and self._as013_scripted_left_surface_target_local is not None
            and not left_physical_contact_now
            and self._as013_scripted_left_recovery_body_relative is None
            and self._as013_scripted_left_surface_preload_steps
            < LEFT_HOUSING_FALLBACK_MAX_PRELOAD_STEPS
            and (
                self._as013_scripted_left_surface_last_preload_frame is None
                or frame - self._as013_scripted_left_surface_last_preload_frame
                >= LEFT_HOUSING_FALLBACK_PRELOAD_INTERVAL_FRAMES
            )
            and nearest_left_tip_body_local is not None
            and float(
                np.linalg.norm(
                    nearest_left_tip_body_local
                    - self._as013_scripted_left_surface_target_local
                )
            )
            <= LEFT_HOUSING_FALLBACK_READY_DISTANCE_M
        ):
            self._as013_scripted_left_surface_target_local = (
                _preload_left_housing_surface_target_local(
                    self._as013_scripted_left_surface_target_local,
                    RIGHT_HOUSING_FALLBACK_INSIDE_TARGET_LOCAL_M,
                )
            )
            self._as013_scripted_left_surface_preload_steps += 1
            self._as013_scripted_left_surface_last_preload_frame = frame
        left_target_local = (
            self._as013_scripted_left_surface_target_local.copy()
            if self._as013_scripted_left_surface_target_local is not None
            else (
                RIGHT_HOUSING_FALLBACK_FEATURE_TARGET_LOCAL_M.copy()
                if self._as013_scripted_left_fallback_grip_active
                else RIGHT_HOUSING_SURFACE_TARGET_LOCAL_M.copy()
            )
        )
        left_target_world = (
            body_matrix[:3, :3] @ left_target_local
            + body_matrix[:3, 3]
        )
        left_contact_filter_row_counts = self._as013_contact_filter_counts(
            self._as013_left_body_contact_view,
            self._as013_left_contact_filter_paths,
        )
        if (
            self._as013_scripted_left_fallback_grip_active
            and self._as013_left_stabilized_seen
            and left_fingertips
        ):
            if (
                left_physical_contact_now
                and self._as013_scripted_left_recovery_fingertip_target_world
                is None
            ):
                self._as013_scripted_left_stabilizer_finger_number = (
                    _left_stabilizer_finger_from_contact_counts(
                        left_contact_filter_row_counts,
                        hand_side=STABILIZER_HAND_SIDE,
                    )
                )
            safe_stabilizer_tip = self._as013_fingertip_point(
                STABILIZER_HAND_SIDE,
                self._as013_scripted_left_stabilizer_finger_number,
            )
            if safe_stabilizer_tip is not None:
                self._as013_scripted_left_hold_fingertip_index = int(
                    np.argmin(
                        [
                            float(
                                np.linalg.norm(
                                    point - safe_stabilizer_tip
                                )
                            )
                            for point in left_fingertips
                        ]
                    )
                )
        left_recovery_feature_error: np.ndarray | None = None
        left_recovery_feature_target_world: np.ndarray | None = None
        left_recovery_feature_current_world: np.ndarray | None = None
        left_fixed_world_anchor_active = False
        left_hold_preload_correction: np.ndarray | None = None
        left_hold_effective_standoff_m: float | None = None
        left_hold_preload_m = LEFT_HOLD_CONTACT_PRELOAD_M
        fixed_anchor_index = (
            self._as013_scripted_left_recovery_fingertip_index
        )
        fixed_anchor_target = (
            self._as013_scripted_left_recovery_fingertip_target_world
        )
        if (
            self._as013_scripted_left_fallback_grip_active
            and self._as013_right_slider_contact_seen
            and fixed_anchor_target is not None
            and not self._as013_scripted_left_fixed_anchor_slide_preloaded
        ):
            fixed_anchor_target = _left_fixed_anchor_slide_preload_target(
                fixed_anchor_target,
                left_target_world,
            )
            self._as013_scripted_left_recovery_fingertip_target_world = (
                fixed_anchor_target.copy()
            )
            self._as013_scripted_left_fixed_anchor_slide_preloaded = True
        left_fixed_world_anchor_ready = bool(
            self._as013_scripted_left_fallback_grip_active
            and self._as013_left_stabilized_seen
            and fixed_anchor_index is not None
            and fixed_anchor_target is not None
            and 0 <= fixed_anchor_index < len(left_fingertips)
        )
        if (
            frame >= 60
            and frame < right_approach_timeout_frame
            and (
                not left_physical_contact_now
                or left_fixed_world_anchor_ready
            )
            and self._as013_scripted_release_frame is None
            and (
                nearest_left_tip is not None
                or self._as013_scripted_left_recovery_body_relative is not None
            )
        ):
            left_actual_matrix = self._native_end_effector_pose_matrix(
                STABILIZER_HAND_SIDE,
                left_home,
            )
            if left_fixed_world_anchor_ready:
                left_fixed_world_anchor_active = True
                left_recovery_feature_target_world = (
                    fixed_anchor_target.copy()
                )
                left_recovery_feature_current_world = left_fingertips[
                    fixed_anchor_index
                ].copy()
                left_recovery_feature_error = (
                    left_recovery_feature_target_world
                    - left_recovery_feature_current_world
                )
                self._as013_scripted_left_fixed_anchor_offset_world = (
                    _advance_left_fixed_anchor_offset(
                        self._as013_scripted_left_fixed_anchor_offset_world,
                        left_recovery_feature_target_world,
                        left_recovery_feature_current_world,
                    )
                )
                left_correction = (
                    self._as013_scripted_left_fixed_anchor_offset_world.copy()
                )
                if self._as013_scripted_left_hold_pose is not None:
                    left_actual_matrix = _pose_vector_to_matrix(
                        self._as013_scripted_left_hold_pose
                    )
                recovery_rotation = left_actual_matrix[:3, :3]
            elif self._as013_scripted_left_recovery_body_relative is not None:
                left_recovery_target_matrix = (
                    body_matrix
                    @ self._as013_scripted_left_recovery_body_relative
                )
                recovery_index = (
                    self._as013_scripted_left_recovery_fingertip_index
                )
                recovery_local = (
                    self._as013_scripted_left_recovery_fingertip_body_local
                )
                if (
                    recovery_index is not None
                    and recovery_local is not None
                    and 0 <= recovery_index < len(left_fingertips)
                ):
                    recovery_feature_contact_world = (
                        body_matrix
                        @ np.append(recovery_local, 1.0)
                    )[:3]
                    left_recovery_feature_target_world = (
                        _left_recovery_preload_target(
                            recovery_feature_contact_world,
                            left_target_world,
                        )
                    )
                    left_recovery_feature_current_world = left_fingertips[
                        recovery_index
                    ].copy()
                    left_recovery_feature_error = (
                        left_recovery_feature_target_world
                        - left_recovery_feature_current_world
                    )
                    left_recovery_target_matrix[:3, 3] += (
                        left_recovery_feature_error
                    )
                left_correction = (
                    left_recovery_target_matrix[:3, 3]
                    - left_actual_matrix[:3, 3]
                )
                recovery_rotation = left_recovery_target_matrix[:3, :3]
            else:
                left_correction = left_target_world - nearest_left_tip
                recovery_rotation = left_yaw_aligned_rotation
            left_correction_norm = float(np.linalg.norm(left_correction))
            left_servo_step = (
                LEFT_FIXED_ANCHOR_MAX_OFFSET_M
                if left_fixed_world_anchor_active
                else (
                    LEFT_HOLD_RECOVERY_STEP_M
                    if self._as013_left_stabilized_seen
                    else LEFT_CARTESIAN_SERVO_STEP_M
                )
            )
            if left_correction_norm > left_servo_step:
                left_correction *= (
                    left_servo_step
                    / left_correction_norm
                )
            left_actual_matrix[:3, 3] += left_correction
            left_servo_pose = np.concatenate(
                [
                    left_actual_matrix[:3, 3],
                    _matrix_to_quat_wxyz(recovery_rotation),
                ]
            ).astype(np.float64)
        left_hold_pose_should_refresh = (
            _left_hold_pose_should_refresh(
                left_physical_contact_now,
                self._as013_left_stable_contact_now,
                self._as013_left_stabilized_seen,
                self._as013_scripted_left_hold_pose is not None,
            )
            if self._as013_scripted_left_fallback_grip_active
            else bool(
                left_physical_contact_now
                or self._as013_left_stable_contact_now
                or (
                    self._as013_left_stabilized_seen
                    and self._as013_scripted_left_hold_pose is None
                )
            )
        )
        if left_hold_pose_should_refresh:
            left_actual_matrix = self._native_end_effector_pose_matrix(
                STABILIZER_HAND_SIDE,
                left_home,
            )
            left_contact_matrix = left_actual_matrix.copy()
            if left_physical_contact_now and nearest_left_tip is not None:
                left_error = left_target_world - nearest_left_tip
                left_error_norm = float(np.linalg.norm(left_error))
                if self._as013_scripted_left_contact_standoff_m is None:
                    self._as013_scripted_left_contact_standoff_m = max(
                        0.0,
                        left_error_norm - LEFT_HOLD_CONTACT_PRELOAD_M,
                    )
                slide_hold_active = bool(
                    self._as013_right_slider_contact_seen
                    and float(
                        self._as013_last_metrics.get(
                            "right_slider_contact_force",
                            0.0,
                        )
                    )
                    > CONTACT_FORCE_THRESHOLD_N
                )
                left_hold_preload_m = (
                    LEFT_SLIDE_HOLD_CONTACT_PRELOAD_M
                    if slide_hold_active
                    else LEFT_HOLD_CONTACT_PRELOAD_M
                )
                left_hold_effective_standoff_m = (
                    _left_hold_effective_standoff(
                        self._as013_scripted_left_contact_standoff_m,
                        left_hold_preload_m,
                    )
                )
                left_hold_preload_correction = _left_hold_preload_correction(
                    left_error,
                    left_hold_effective_standoff_m,
                    left_hold_preload_m,
                )
                left_actual_matrix[:3, 3] += left_hold_preload_correction
            self._as013_scripted_left_hold_pose = np.concatenate(
                [
                    left_actual_matrix[:3, 3],
                    _matrix_to_quat_wxyz(left_actual_matrix[:3, :3]),
                ]
            ).astype(np.float64)
            if float(
                self._as013_last_metrics.get(
                    "left_body_contact_force",
                    0.0,
                )
            ) >= LEFT_RECOVERY_ORIENTATION_MIN_FORCE_N:
                if (
                    self._as013_scripted_left_hold_fingertip_index is None
                    and left_hold_fingertip_index is not None
                ):
                    self._as013_scripted_left_hold_fingertip_index = (
                        left_hold_fingertip_index
                    )
                self._as013_scripted_left_recovery_body_relative = (
                    _invert_pose_matrix(body_matrix)
                    @ left_contact_matrix
                )
                if left_fingertips:
                    recovery_index = (
                        self._as013_scripted_left_hold_fingertip_index
                    )
                    if recovery_index is None:
                        recovery_index = int(
                            np.argmin(
                                [
                                    float(np.linalg.norm(point - body[:3]))
                                    for point in left_fingertips
                                ]
                            )
                        )
                    self._as013_scripted_left_recovery_fingertip_index = (
                        recovery_index
                    )
                    self._as013_scripted_left_recovery_fingertip_body_local = (
                        _invert_pose_matrix(body_matrix)
                        @ np.append(left_fingertips[recovery_index], 1.0)
                    )[:3]
                    if self._as013_scripted_left_fallback_grip_active:
                        self._as013_scripted_left_recovery_fingertip_target_world = (
                            _left_recovery_preload_target(
                                left_fingertips[recovery_index],
                                left_target_world,
                            )
                        )
        if (
            self._as013_scripted_left_fallback_grip_active
            and self._as013_left_stabilized_seen
            and self._as013_scripted_left_recovery_fingertip_target_world
            is None
            and left_fingertips
        ):
            recovery_index = self._as013_scripted_left_hold_fingertip_index
            if recovery_index is None:
                recovery_index = int(
                    np.argmin(
                        [
                            float(np.linalg.norm(point - body[:3]))
                            for point in left_fingertips
                        ]
                    )
                )
            self._as013_scripted_left_recovery_fingertip_index = recovery_index
            self._as013_scripted_left_recovery_fingertip_target_world = (
                _left_recovery_preload_target(
                    left_fingertips[recovery_index],
                    left_target_world,
                )
            )
        if (
            (left_physical_contact_now or self._as013_left_stabilized_seen)
            and self._as013_scripted_left_hold_hand is None
        ):
            left_actual_hand = (
                self._actual_hand_joint_pose(STABILIZER_HAND_SIDE)
                if self._robots.get(STABILIZER_HAND_SIDE) is not None
                and self._hand_joint_ids.get(STABILIZER_HAND_SIDE)
                else self._right_hand
            )
            self._as013_scripted_left_hold_hand = np.asarray(
                left_actual_hand,
                dtype=np.float64,
            ).reshape(20).copy()
        if (
            self._as013_left_stabilized_seen
            and self._as013_scripted_left_hand_preload_start_frame is None
        ):
            self._as013_scripted_left_hand_preload_start_frame = frame
        if self._as013_scripted_left_hold_pose is not None:
            left_hold = self._as013_scripted_left_hold_pose.copy()
        left_recovery_servo_active = (
            _left_recovery_servo_should_apply(
                self._as013_scripted_left_hold_pose is not None,
                left_physical_contact_now,
                left_servo_pose is not None,
                self._as013_left_stabilized_seen,
                self._as013_scripted_left_recovery_fingertip_target_world
                is not None,
            )
            if self._as013_scripted_left_fallback_grip_active
            else bool(
                self._as013_scripted_left_hold_pose is not None
                and not left_physical_contact_now
                and left_servo_pose is not None
            )
        )
        if left_recovery_servo_active:
            left_hold = left_servo_pose.copy()
        left_grip = (
            self._as013_scripted_left_hold_hand.copy()
            if self._as013_scripted_left_hold_hand is not None
            else _POWER_GRIP.copy()
        )
        left_hand_preload_effective_fraction = 0.0
        left_safe_finger_preload_effective_fraction = 0.0
        if (
            self._as013_scripted_left_hold_hand is not None
            and self._as013_scripted_left_hand_preload_start_frame is not None
        ):
            left_hand_preload_effective_fraction = (
                _effective_left_hand_preload_fraction(
                    frame,
                    self._as013_scripted_left_hand_preload_start_frame,
                    left_hand_preload_fraction,
                )
            )
            left_grip += left_hand_preload_effective_fraction * (
                _POWER_GRIP - left_grip
            )
            left_safe_finger_preload_effective_fraction = (
                _effective_left_hand_preload_fraction(
                    frame,
                    self._as013_scripted_left_hand_preload_start_frame,
                    left_safe_finger_preload_fraction,
                )
            )
        left_grip = _left_hold_hand_with_reinforced_safe_finger(
            left_grip,
            left_safe_finger_preload_effective_fraction,
            self._as013_scripted_left_stabilizer_finger_number,
        )
        left_unsafe_finger_retraction_fraction = (
            _effective_left_hand_preload_fraction(
                frame,
                (
                    LEFT_UNSAFE_FINGER_RETRACTION_START_FRAME
                    if (
                        not left_physical_contact_now
                        and not self._as013_left_stabilized_seen
                    )
                    else None
                ),
                left_unsafe_finger_retraction_target_fraction,
                control_frames=(
                    LEFT_UNSAFE_FINGER_RETRACTION_CONTROL_FRAMES
                ),
            )
        )
        left_grip = _left_hold_hand_with_retracted_unsafe_finger(
            left_grip,
            left_unsafe_finger_retraction_fraction,
            distal_only=left_unsafe_finger_distal_only,
        )
        left_hover = left_hold.copy()
        left_hover[2] += 0.090
        blade_matrix = _pose_vector_to_matrix(blade)
        right_touch = right_home.copy()
        right_touch[:3] = blade[:3] - 0.018 * axis
        right_touch[2] += right_work_height
        yaw_half = math.radians(right_palm_yaw_deg) / 2.0
        roll_half = math.radians(right_palm_roll_deg) / 2.0
        cos_yaw = math.cos(yaw_half)
        sin_yaw = math.sin(yaw_half)
        cos_roll = math.cos(roll_half)
        sin_roll = math.sin(roll_half)
        legacy_slider_orientation_override = bool(
            SCRIPTED_RIGHT_PALM_YAW_ENV in os.environ
            or SCRIPTED_RIGHT_PALM_ROLL_ENV in os.environ
        )
        if legacy_slider_orientation_override:
            right_touch[3:7] = np.asarray(
                [
                    -sin_roll * cos_yaw,
                    cos_roll * cos_yaw,
                    cos_roll * sin_yaw,
                    sin_roll * sin_yaw,
                ],
                dtype=np.float64,
            )
        else:
            # No cross-hand pose has been live-certified yet.  Default to the
            # physical left arm's current reachable palm orientation; the
            # closed-loop fingertip servo can then adapt from live geometry.
            right_touch[3:7] = right_home[3:7]
        right_hover = right_touch.copy()
        right_hover[2] += 0.090
        right_slide = right_touch.copy()
        right_slide[:3] += right_slide_palm_travel_m * axis
        slider_center_local = 0.5 * (
            SLIDER_BUTTON_LOCAL_MIN_M + SLIDER_BUTTON_LOCAL_MAX_M
        )
        slider_target_local = slider_center_local.copy()
        slider_target_local[2] += RIGHT_FINGERTIP_TARGET_CLEARANCE_M
        if self._as013_scripted_left_fallback_grip_active:
            slider_target_local = _fixed_right_fingertip_target_local(
                slider_center_local
            )
        slider_center_world = (
            blade_matrix[:3, :3] @ slider_center_local
            + blade_matrix[:3, 3]
        )
        slider_target_world = (
            blade_matrix[:3, :3] @ slider_target_local
            + blade_matrix[:3, 3]
        )
        right_fingertips = self._as013_fingertip_points(SLIDER_HAND_SIDE)
        right_named_fingertips = {
            finger_number: point
            for finger_number in range(1, 6)
            if (
                point := self._as013_fingertip_point(
                    SLIDER_HAND_SIDE,
                    finger_number,
                )
            )
            is not None
        }
        if (
            frame >= 180
            and self._as013_scripted_right_finger_number is None
            and right_named_fingertips
        ):
            if requested_right_finger_number in right_named_fingertips:
                self._as013_scripted_right_finger_number = (
                    requested_right_finger_number
                )
            else:
                self._as013_scripted_right_finger_number = min(
                    right_named_fingertips,
                    key=lambda finger_number: float(
                        np.linalg.norm(
                            right_named_fingertips[finger_number]
                            - slider_target_world
                        )
                    ),
                )
        right_control_fingertip = right_named_fingertips.get(
            self._as013_scripted_right_finger_number
        )
        if (
            right_control_fingertip is not None
            and self._as013_scripted_right_target_blade_local is None
        ):
            if not self._as013_scripted_left_fallback_grip_active:
                control_local = (
                    _invert_pose_matrix(blade_matrix)
                    @ np.append(right_control_fingertip, 1.0)
                )[:3]
                target_lower = SLIDER_BUTTON_LOCAL_MIN_M.copy()
                target_upper = SLIDER_BUTTON_LOCAL_MAX_M.copy()
                target_lower[:2] += RIGHT_FINGERTIP_TARGET_INSET_M
                target_upper[:2] -= RIGHT_FINGERTIP_TARGET_INSET_M
                slider_target_local[:2] = np.clip(
                    control_local[:2],
                    target_lower[:2],
                    target_upper[:2],
                )
            self._as013_scripted_right_target_blade_local = (
                slider_target_local.copy()
            )
        if self._as013_scripted_right_target_blade_local is not None:
            slider_target_local = (
                self._as013_scripted_right_target_blade_local.copy()
            )
            slider_target_world = (
                blade_matrix[:3, :3] @ slider_target_local
                + blade_matrix[:3, 3]
            )
        right_touch_hand = _right_touch_hand_posture(
            self._as013_scripted_right_finger_number,
            right_finger_extension_fraction,
            right_finger_abduction_rad,
        )
        right_servo_pose: np.ndarray | None = None
        right_servo_correction: np.ndarray | None = None
        right_servo_stage = "inactive"
        right_servo_target_local: np.ndarray | None = None
        current_right_slider_contact_force = float(
            self._as013_last_metrics.get(
                "right_slider_contact_force",
                0.0,
            )
        )
        current_left_stable_contact = bool(
            self._as013_last_metrics.get(
                "left_stable_contact_now",
                False,
            )
        )
        current_blade_progress = float(
            self._as013_last_metrics.get(
                "blade_progress",
                self._as013_reset_progress,
            )
        )
        self._as013_scripted_late_left_stability_grace_frames = (
            _advance_late_left_stability_grace_frames(
                self._as013_scripted_late_left_stability_grace_frames,
                current_left_stable_contact,
                current_blade_progress,
                self._as013_late_left_stability_grace_start_progress,
            )
        )
        current_left_stable_for_control = bool(
            current_left_stable_contact
            or self._as013_scripted_late_left_stability_grace_frames > 0
        )
        self._as013_scripted_right_initial_contact_dropout_frames = (
            _advance_initial_contact_dropout_grace_frames(
                self._as013_scripted_right_initial_contact_dropout_frames,
                (
                    self._as013_right_slider_contact_seen
                    and not self._as013_scripted_right_recontact_lifted
                ),
                current_left_stable_for_control,
                current_right_slider_contact_force,
                self._as013_initial_contact_dropout_grace_control_frames,
            )
        )
        right_initial_contact_dropout_grace_active = bool(
            0
            < self._as013_scripted_right_initial_contact_dropout_frames
            <= self._as013_initial_contact_dropout_grace_control_frames
        )
        right_recontact_active = _right_recontact_needed(
            self._as013_right_slider_contact_seen,
            current_right_slider_contact_force,
            current_left_stable_for_control,
        ) and not right_initial_contact_dropout_grace_active
        right_late_slide_direct_recontact_active = (
            _late_slide_direct_recontact_needed(
                right_recontact_active,
                current_left_stable_for_control,
                current_right_slider_contact_force,
                self._as013_scripted_slide_control_frames,
            )
        )
        self._as013_scripted_right_recontact_grace_frames = (
            _advance_right_recontact_grace_frames(
                self._as013_scripted_right_recontact_grace_frames,
                self._as013_scripted_right_recontact_lifted,
                current_right_slider_contact_force,
                self._as013_right_recontact_grace_control_frames,
            )
        )
        right_recontact_grace_active = bool(
            self._as013_scripted_right_recontact_lifted
            and current_left_stable_for_control
            and self._as013_scripted_right_recontact_grace_frames > 0
        )
        self._as013_scripted_right_recovered_hand_blend = (
            _advance_recovered_right_hand_blend(
                self._as013_scripted_right_recovered_hand_blend,
                self._as013_scripted_right_recontact_lifted,
                right_recontact_grace_active,
            )
        )
        self._as013_scripted_right_late_continuous_hand_blend = (
            _advance_late_continuous_right_hand_blend(
                self._as013_scripted_right_late_continuous_hand_blend,
                right_late_continuous_hand_blend,
                current_blade_progress,
                right_recovered_late_finger_extension_start_progress,
                bool(
                    self._as013_right_slider_contact_seen
                    and not self._as013_scripted_right_recontact_lifted
                    and current_left_stable_for_control
                    and current_right_slider_contact_force
                    > CONTACT_FORCE_THRESHOLD_N
                ),
            )
        )
        self._as013_scripted_right_recovered_target_blend = (
            _advance_recovered_target_blend(
                self._as013_scripted_right_recovered_target_blend,
                self._as013_scripted_right_recontact_lifted,
                right_recontact_grace_active,
                self._as013_scripted_right_contact_settle_frames,
            )
        )
        if (
            not right_recontact_grace_active
            and current_right_slider_contact_force
            <= CONTACT_FORCE_THRESHOLD_N
        ):
            self._as013_scripted_right_recovered_target_blade_local = None
        if self._as013_right_slider_contact_seen:
            self._as013_scripted_left_recovery_ready_frames = (
                _advance_left_recovery_ready_frames(
                    self._as013_scripted_left_recovery_ready_frames,
                    current_left_stable_for_control,
                    current_right_slider_contact_force,
                    right_finger_reentry_control_frames,
                )
            )
        else:
            self._as013_scripted_left_recovery_ready_frames = 0
        right_finger_reentry_fraction = _right_finger_reentry_fraction(
            self._as013_right_slider_contact_seen,
            current_left_stable_for_control,
            current_right_slider_contact_force,
            self._as013_scripted_left_recovery_ready_frames,
            right_finger_reentry_control_frames,
        )
        if (
            right_recontact_grace_active
            or right_initial_contact_dropout_grace_active
        ):
            right_finger_reentry_fraction = 1.0
        if right_late_slide_direct_recontact_active:
            right_finger_reentry_fraction = 1.0
        right_finger_withdrawal_active = _right_finger_withdrawal_needed(
            self._as013_right_slider_contact_seen,
            current_left_stable_for_control,
            current_right_slider_contact_force,
            self._as013_scripted_left_recovery_ready_frames,
        ) and not (
            right_recontact_grace_active
            or right_initial_contact_dropout_grace_active
            or right_late_slide_direct_recontact_active
        )
        right_finger_reentry_active = bool(
            self._as013_right_slider_contact_seen
            and current_left_stable_for_control
            and current_right_slider_contact_force
            <= CONTACT_FORCE_THRESHOLD_N
            and self._as013_scripted_left_recovery_ready_frames
            >= LEFT_RECOVERY_READY_CONTROL_FRAMES
            and self._as013_scripted_left_recovery_ready_frames
            < (
                LEFT_RECOVERY_READY_CONTROL_FRAMES
                + right_finger_reentry_control_frames
            )
            and not right_recontact_grace_active
            and not right_initial_contact_dropout_grace_active
            and not right_late_slide_direct_recontact_active
        )
        (
            right_preload_settling_active,
            right_slide_servo_active,
        ) = _right_contact_phase_flags(
            self._as013_right_slider_contact_seen,
            current_left_stable_contact,
            current_right_slider_contact_force,
            self._as013_scripted_right_contact_settle_frames,
        )
        right_effective_recontact_target_local: np.ndarray | None = None
        if frame >= 180 and right_control_fingertip is not None:
            right_actual_matrix = self._native_end_effector_pose_matrix(
                SLIDER_HAND_SIDE,
                right_home,
            )
            right_control_local = (
                _invert_pose_matrix(blade_matrix)
                @ np.append(right_control_fingertip, 1.0)
            )[:3]
            self._as013_scripted_right_last_proven_contact_blade_local = (
                _updated_last_proven_right_contact_local(
                    self._as013_scripted_right_last_proven_contact_blade_local,
                    right_control_local,
                    current_right_slider_contact_force,
                    self._as013_scripted_right_recontact_lifted,
                )
            )
            if (
                self._as013_scripted_left_fallback_grip_active
                and not self._as013_right_slider_contact_seen
                and frame == right_fallback_target_adapt_frame
                and self._as013_scripted_right_target_blade_local is not None
            ):
                self._as013_scripted_right_target_blade_local = (
                    _adapt_fallback_right_target_cross_edge(
                        right_control_local,
                        self._as013_scripted_right_target_blade_local,
                    )
                )
                slider_target_local = (
                    self._as013_scripted_right_target_blade_local.copy()
                )
                slider_target_world = (
                    blade_matrix[:3, :3] @ slider_target_local
                    + blade_matrix[:3, 3]
                )
            right_servo_target_local, right_servo_stage = (
                _staged_right_fingertip_target_local(
                    right_control_local,
                    slider_target_local,
                    self._as013_scripted_right_servo_stage,
                )
            )
            if (
                self._as013_right_slider_contact_seen
                and current_right_slider_contact_force
                > CONTACT_FORCE_THRESHOLD_N
                and self._as013_scripted_right_recontact_target_blade_local
                is None
            ):
                self._as013_scripted_right_recontact_target_blade_local = (
                    _preload_right_recontact_target_local(
                        right_control_local,
                        slider_target_local,
                    )
                )
            if (
                right_recontact_active
                or right_preload_settling_active
                or right_slide_servo_active
                or right_initial_contact_dropout_grace_active
            ):
                if (
                    self._as013_scripted_right_recontact_target_blade_local
                    is None
                ):
                    self._as013_scripted_right_recontact_target_blade_local = (
                        _preload_right_recontact_target_local(
                            right_control_local,
                            slider_target_local,
                        )
                    )
                recovered_preload_active = bool(
                    self._as013_scripted_right_recontact_lifted
                    and (
                        right_preload_settling_active
                        or right_slide_servo_active
                        or right_recontact_grace_active
                    )
                )
                right_contact_target_local = (
                    _latest_right_recontact_target_local(
                        self._as013_scripted_right_recontact_target_blade_local,
                        self._as013_scripted_right_last_proven_contact_blade_local,
                        slider_target_local,
                        right_recontact_active,
                    )
                )
                right_effective_recontact_target_local = (
                    _preload_recovered_right_contact_in_plane(
                        _preload_lost_right_contact_normal(
                            right_contact_target_local,
                            self._as013_scripted_right_recontact_lifted,
                        ),
                        slider_target_local,
                        recovered_preload_active,
                    )
                )
                if (
                    self._as013_scripted_right_recontact_lifted
                    and current_right_slider_contact_force
                    > CONTACT_FORCE_THRESHOLD_N
                    and self._as013_scripted_right_recovered_target_blade_local
                    is None
                ):
                    self._as013_scripted_right_recovered_target_blade_local = (
                        _preload_recovered_right_contact_normal(
                            _preload_recovered_capture_cross_edge(
                                _preload_recovered_capture_in_plane(
                                    _capture_right_recontact_target_local(
                                        right_control_local,
                                    ),
                                    slider_target_local,
                                ),
                                slider_target_local,
                            )
                        )
                    )
                if (
                    self._as013_scripted_right_recovered_target_blade_local
                    is not None
                    and right_recontact_active
                ):
                    # Search from the last proven physical contact while the
                    # slider is lost.  Once contact returns, the fixed authored
                    # button point and bounded tangential step resume.
                    recovered_target = (
                        self._as013_scripted_right_recovered_target_blade_local
                    )
                    right_effective_recontact_target_local = (
                        recovered_target.copy()
                    )
                elif (
                    self._as013_scripted_right_recovered_target_blade_local
                    is not None
                ):
                    right_effective_recontact_target_local = (
                        _settling_right_recontact_target_local(
                            right_effective_recontact_target_local,
                            self._as013_scripted_right_recovered_target_blade_local,
                            right_recontact_grace_active,
                            self._as013_scripted_right_recovered_target_blend,
                        )
                    )
                late_recovered_in_plane_preload_m = max(
                    0.0,
                    _effective_recovered_contact_in_plane_preload_m(
                        current_blade_progress,
                        right_recovered_late_in_plane_preload_m,
                        right_recovered_late_finger_extension_start_progress,
                    )
                    - RIGHT_RECOVERED_CONTACT_IN_PLANE_PRELOAD_M,
                )
                right_effective_recontact_target_local = (
                    _preload_recovered_right_contact_in_plane(
                        right_effective_recontact_target_local,
                        slider_target_local,
                        recovered_preload_active,
                        late_recovered_in_plane_preload_m,
                    )
                )
                if recovered_preload_active:
                    right_effective_recontact_target_local[2] -= (
                        _effective_recovered_late_normal_preload_m(
                            current_blade_progress,
                            right_recovered_late_normal_preload_m,
                            right_recovered_late_finger_extension_start_progress,
                        )
                    )
                if right_initial_contact_dropout_grace_active:
                    # A brief threshold crossing should not open or lift the
                    # finger.  Preserve the latest physically proven in-plane
                    # position and add only the bounded blade-normal preload.
                    if (
                        self._as013_scripted_right_last_proven_contact_blade_local
                        is not None
                    ):
                        right_effective_recontact_target_local = (
                            _preload_lost_right_contact_normal(
                                (
                                    self._as013_scripted_right_last_proven_contact_blade_local
                                ),
                                True,
                            )
                        )
                    right_servo_target_local = (
                        right_effective_recontact_target_local.copy()
                    )
                    right_servo_stage = "contact_descent"
                elif right_preload_settling_active or right_slide_servo_active:
                    right_servo_target_local = (
                        right_effective_recontact_target_local.copy()
                    )
                    right_servo_stage = "contact_descent"
                elif (
                    right_finger_withdrawal_active
                    or right_finger_reentry_active
                ):
                    right_servo_target_local = right_control_local.copy()
                    right_servo_stage = (
                        "finger_withdrawal"
                        if right_finger_withdrawal_active
                        else "finger_reentry"
                    )
                    if (
                        current_right_slider_contact_force
                        <= CONTACT_FORCE_THRESHOLD_N
                    ):
                        self._as013_scripted_right_recontact_lifted = True
                else:
                    (
                        right_servo_target_local,
                        right_servo_stage,
                        self._as013_scripted_right_recontact_lifted,
                    ) = _recontact_right_fingertip_target_local(
                        right_control_local,
                        right_effective_recontact_target_local,
                        self._as013_scripted_right_recontact_lifted,
                        current_right_slider_contact_force,
                        current_left_stable_for_control,
                    )
            if (
                self._as013_left_stabilized_seen
                and right_servo_stage
                in {"vertical_entry", "planar_alignment", "contact_descent"}
            ):
                self._as013_scripted_right_servo_stage = right_servo_stage
            right_servo_target_world = (
                blade_matrix[:3, :3] @ right_servo_target_local
                + blade_matrix[:3, 3]
            )
            correction = _bounded_right_servo_correction(
                right_servo_target_world - right_control_fingertip,
                lateral_gain=right_lateral_gain,
                vertical_gain=right_vertical_gain,
                max_step_m=_right_servo_state_max_step(
                    recontact_active=right_recontact_active,
                    preload_settling_active=(
                        right_preload_settling_active
                        or right_initial_contact_dropout_grace_active
                    ),
                    slide_servo_active=right_slide_servo_active,
                    stage=right_servo_stage,
                    current_local=right_control_local,
                    final_target_local=slider_target_local,
                    surface_slow_radius_m=right_surface_slow_radius_m,
                    recontact_step_m=right_recontact_servo_step_m,
                ),
            )
            right_servo_correction = correction.copy()
            if self._as013_left_stabilized_seen:
                right_actual_matrix[:3, 3] += correction
                if (
                    self._as013_scripted_right_alignment_rotation_blade_relative
                    is None
                    and (
                        right_servo_stage != "vertical_entry"
                        or (
                            self._as013_scripted_left_fallback_grip_active
                            and right_fallback_alignment_lock_frame > 0
                            and frame >= right_fallback_alignment_lock_frame
                        )
                    )
                ):
                    self._as013_scripted_right_alignment_rotation_blade_relative = (
                        _invert_pose_matrix(blade_matrix)[:3, :3]
                        @ _blended_right_alignment_world_rotation(
                            right_actual_matrix[:3, :3],
                            self._as013_reset_yaw_degrees,
                            right_alignment_reference_blend,
                        )
                    )
                if (
                    self._as013_scripted_right_alignment_rotation_blade_relative
                    is not None
                ):
                    right_actual_matrix[:3, :3] = (
                        blade_matrix[:3, :3]
                        @ self._as013_scripted_right_alignment_rotation_blade_relative
                    )
                self._as013_scripted_right_approach_blade_relative = (
                    _invert_pose_matrix(blade_matrix)
                    @ right_actual_matrix
                )
            if (
                self._as013_scripted_right_approach_blade_relative
                is not None
            ):
                right_actual_matrix = (
                    blade_matrix
                    @ self._as013_scripted_right_approach_blade_relative
                )
            right_servo_pose = np.concatenate(
                [
                    right_actual_matrix[:3, 3],
                    _matrix_to_quat_wxyz(right_actual_matrix[:3, :3]),
                ]
            ).astype(np.float64)
        contact_pose_ready = _right_contact_pose_ready(
                ordered_contact_seen=self._as013_right_slider_contact_seen,
                current_left_stable_contact=current_left_stable_for_control,
                current_slider_contact_force=float(
                    self._as013_last_metrics.get(
                        "right_slider_contact_force",
                        0.0,
                    )
                ),
                servo_stage=right_servo_stage,
            )
        if _right_slide_contact_pose_refresh_needed(
            contact_pose_ready=contact_pose_ready,
            contact_pose_missing=(
                self._as013_scripted_right_contact_pose is None
            ),
            slide_control_frames=(
                self._as013_scripted_slide_control_frames
            ),
            recontact_lifted=(
                self._as013_scripted_right_recontact_lifted
            ),
            contact_settle_frames=(
                self._as013_scripted_right_contact_settle_frames
            ),
        ):
            initial_contact_pose = (
                self._as013_scripted_right_contact_pose is None
            )
            self._as013_scripted_right_contact_pose = (
                _reanchored_right_contact_pose_from_command(
                    self._as013_previous_right_command,
                    axis,
                    blade_matrix[:3, 2],
                    self._as013_scripted_slide_control_frames,
                    right_slide_max_lead_m,
                    blade_world_cross_edge=blade_matrix[:3, 0],
                    cross_edge_correction_m=(
                        self._as013_scripted_right_slide_cross_edge_correction_m
                    ),
                    normal_preload_m=right_slide_normal_preload_m,
                    cross_edge_max_correction_m=(
                        right_slide_cross_edge_max_correction_m
                    ),
                    total_control_frames=right_slide_control_frames,
                    palm_travel_m=right_slide_palm_travel_m,
                    late_normal_preload_m=(
                        right_slide_late_normal_preload_m
                    ),
                )
            )
            self._as013_scripted_right_contact_hand = np.asarray(
                self._actual_hand_joint_pose(SLIDER_HAND_SIDE),
                dtype=np.float64,
            ).reshape(20).copy()
            if initial_contact_pose:
                self._as013_scripted_right_contact_frame = frame
                self._as013_scripted_slide_control_frames = 0
                left_actual_matrix = self._native_end_effector_pose_matrix(
                    STABILIZER_HAND_SIDE,
                    left_home,
                )
                self._as013_scripted_left_counter_slide_anchor_pose = (
                    np.concatenate(
                        [
                            left_actual_matrix[:3, 3],
                            _matrix_to_quat_wxyz(
                                left_actual_matrix[:3, :3]
                            ),
                        ]
                    ).astype(np.float64)
                )
                self._as013_scripted_right_slide_cross_edge_correction_m = (
                    float(
                        np.clip(
                            slider_target_local[0] - right_control_local[0],
                            -right_slide_cross_edge_max_correction_m,
                            right_slide_cross_edge_max_correction_m,
                        )
                    )
                )
        if _right_slide_reanchor_needed(
            right_slide_reanchor_progress,
            self._as013_scripted_right_slide_reanchor_origin_progress,
            self._as013_right_slider_contact_seen,
            current_left_stable_for_control,
            current_right_slider_contact_force,
            current_blade_progress,
        ):
            right_actual_matrix = self._native_end_effector_pose_matrix(
                SLIDER_HAND_SIDE,
                right_home,
            )
            self._as013_scripted_right_contact_pose = np.concatenate(
                [
                    right_actual_matrix[:3, 3],
                    _matrix_to_quat_wxyz(right_actual_matrix[:3, :3]),
                ]
            ).astype(np.float64)
            self._as013_scripted_right_contact_hand = np.asarray(
                self._actual_hand_joint_pose(SLIDER_HAND_SIDE),
                dtype=np.float64,
            ).reshape(20).copy()
            self._as013_scripted_right_slide_reanchor_origin_progress = float(
                current_blade_progress
            )
            self._as013_scripted_right_slide_reanchor_frame = frame
        if self._as013_scripted_right_contact_pose is not None:
            right_touch = self._as013_scripted_right_contact_pose.copy()
            right_slide = right_touch.copy()
            right_slide[:3] += right_slide_palm_travel_m * axis

        if (
            os.environ.get(SCRIPTED_DIAGNOSTIC_ENV, "").strip().lower()
            in {"1", "true", "yes", "on"}
            and frame
            in {
                60,
                120,
                180,
                240,
                300,
                360,
                420,
                600,
                720,
                900,
                1080,
                1200,
                1320,
                1350,
                1500,
                1520,
                1560,
                1620,
                1700,
            }
        ):
            right_actual = self._native_end_effector_pose_matrix(
                SLIDER_HAND_SIDE,
                right_home,
            )[:3, 3]
            right_arm_diagnostic: dict[str, Any] = {}
            right_robot = self._robots.get(SLIDER_HAND_SIDE)
            right_arm_ids = self._arm_joint_ids.get(SLIDER_HAND_SIDE, [])
            if right_robot is not None and right_arm_ids:
                actual_q = (
                    _torch_view(right_robot.data.joint_pos)[:, right_arm_ids]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                target_state = self._joint_targets.get(SLIDER_HAND_SIDE)
                target_q = (
                    actual_q.copy()
                    if target_state is None
                    else (
                        _torch_view(target_state)[:, right_arm_ids]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                )
                soft_limits = _torch_view(
                    right_robot.data.soft_joint_pos_limits
                )[:, right_arm_ids]
                right_arm_diagnostic = {
                    "joint_names": [
                        str(right_robot.joint_names[int(joint_id)])
                        for joint_id in right_arm_ids
                    ],
                    "actual_q": actual_q.tolist(),
                    "target_q": target_q.tolist(),
                    "soft_limits": (
                        soft_limits.detach()
                        .cpu()
                        .numpy()
                        .reshape(-1, 2)
                        .tolist()
                    ),
                }
            left_actual = self._native_end_effector_pose_matrix(
                STABILIZER_HAND_SIDE,
                left_home,
            )[:3, 3]
            left_points = left_fingertips
            left_named_body_points = self._as013_named_hand_body_points(
                STABILIZER_HAND_SIDE
            )
            right_points = right_fingertips
            left_distances = [
                float(np.linalg.norm(point - body[:3]))
                for point in left_points
            ]
            nearest_left_index = (
                int(np.argmin(left_distances)) if left_distances else None
            )
            nearest_left_world = (
                left_points[nearest_left_index]
                if nearest_left_index is not None
                else None
            )
            nearest_left_local = (
                (
                    _invert_pose_matrix(body_matrix)
                    @ np.append(nearest_left_world, 1.0)
                )[:3]
                if nearest_left_world is not None
                else None
            )
            right_distances = [
                float(np.linalg.norm(point - slider_center_world))
                for point in right_points
            ]
            nearest_right_index = (
                int(np.argmin(right_distances)) if right_distances else None
            )
            nearest_right_world = (
                right_points[nearest_right_index]
                if nearest_right_index is not None
                else None
            )
            nearest_right_local = (
                (
                    _invert_pose_matrix(blade_matrix)
                    @ np.append(nearest_right_world, 1.0)
                )[:3]
                if nearest_right_world is not None
                else None
            )
            diagnostic = {
                "role_schema_version": ROLE_SCHEMA_VERSION,
                "slider_hand": SLIDER_HAND_SIDE,
                "stabilizer_hand": STABILIZER_HAND_SIDE,
                "legacy_role_name_binding": {
                    "right_*": "left_slider",
                    "left_*": "right_stabilizer",
                },
                "metrics": self._as013_last_metrics,
                "left_work_height_m": left_work_height,
                "reset_yaw_degrees": self._as013_reset_yaw_degrees,
                "left_surface_fallback_preloaded": (
                    self._as013_scripted_left_surface_preload_steps > 0
                ),
                "left_surface_fallback_preload_steps": (
                    self._as013_scripted_left_surface_preload_steps
                ),
                "left_surface_fallback_last_preload_frame": (
                    self._as013_scripted_left_surface_last_preload_frame
                ),
                "left_surface_target_local_xyz": (
                    left_target_local.tolist()
                ),
                "left_counter_slide_m": left_counter_slide_m,
                "left_hand_preload_fraction": left_hand_preload_fraction,
                "left_hand_preload_effective_fraction": (
                    left_hand_preload_effective_fraction
                ),
                "left_safe_finger_preload_fraction": (
                    left_safe_finger_preload_fraction
                ),
                "left_safe_finger_preload_effective_fraction": (
                    left_safe_finger_preload_effective_fraction
                ),
                "left_unsafe_finger_retraction_fraction": (
                    left_unsafe_finger_retraction_fraction
                ),
                "left_safe_stabilizer_finger_number": (
                    self._as013_scripted_left_stabilizer_finger_number
                ),
                "left_unsafe_blade_finger_number": (
                    LEFT_UNSAFE_BLADE_FINGER_NUMBER
                ),
                "left_unsafe_finger_retraction_target_fraction": (
                    left_unsafe_finger_retraction_target_fraction
                ),
                "left_unsafe_finger_distal_only": (
                    left_unsafe_finger_distal_only
                ),
                "left_hand_preload_start_frame": (
                    self._as013_scripted_left_hand_preload_start_frame
                ),
                "left_approach_grip_fraction": left_approach_grip_fraction,
                "left_fallback_grip_active": (
                    self._as013_scripted_left_fallback_grip_active
                ),
                "right_work_height_m": right_work_height,
                "body_xyz": body[:3].tolist(),
                "blade_xyz": blade[:3].tolist(),
                "left_target_xyz": left_hold[:3].tolist(),
                "right_target_xyz": right_touch[:3].tolist(),
                "left_actual_xyz": left_actual.tolist(),
                "right_actual_xyz": right_actual.tolist(),
                "right_previous_command_xyz": np.asarray(
                    self._left_tcp_pose[:3],
                    dtype=np.float64,
                ).tolist(),
                "right_servo_target_xyz": (
                    right_servo_pose[:3].tolist()
                    if right_servo_pose is not None
                    else None
                ),
                "right_servo_correction_world_xyz": (
                    right_servo_correction.tolist()
                    if right_servo_correction is not None
                    else None
                ),
                "right_servo_stage": right_servo_stage,
                "right_finger_withdrawal_active": (
                    right_finger_withdrawal_active
                ),
                "right_finger_reentry_active": right_finger_reentry_active,
                "right_preload_settling_active": (
                    right_preload_settling_active
                ),
                "right_slide_servo_active": right_slide_servo_active,
                "right_finger_reentry_fraction": (
                    right_finger_reentry_fraction
                ),
                "left_recovery_ready_control_frames": (
                    self._as013_scripted_left_recovery_ready_frames
                ),
                "right_contact_settle_control_frames": (
                    self._as013_scripted_right_contact_settle_frames
                ),
                "right_alignment_orientation_locked": (
                    self._as013_scripted_right_alignment_rotation_blade_relative
                    is not None
                ),
                "right_servo_target_blade_local_xyz": (
                    right_servo_target_local.tolist()
                    if right_servo_target_local is not None
                    else None
                ),
                "left_tcp_error_m": float(
                    np.linalg.norm(left_actual - left_hold[:3])
                ),
                "right_tcp_error_m": float(
                    np.linalg.norm(right_actual - right_touch[:3])
                ),
                "left_hand_point_min_body_origin_distance_m": min(
                    left_distances,
                    default=math.inf,
                ),
                "left_fingertip_target_world_xyz": (
                    left_target_world.tolist()
                ),
                "left_fingertip_min_target_distance_m": min(
                    (
                        float(np.linalg.norm(point - left_target_world))
                        for point in left_points
                    ),
                    default=math.inf,
                ),
                "left_recovery_reference_available": (
                    self._as013_scripted_left_recovery_body_relative
                    is not None
                ),
                "left_recovery_feature_error_m": (
                    float(np.linalg.norm(left_recovery_feature_error))
                    if left_recovery_feature_error is not None
                    else None
                ),
                "left_recovery_feature_target_world_xyz": (
                    left_recovery_feature_target_world.tolist()
                    if left_recovery_feature_target_world is not None
                    else None
                ),
                "left_fixed_world_anchor_active": (
                    left_fixed_world_anchor_active
                ),
                "left_fixed_world_anchor_target_xyz": (
                    self._as013_scripted_left_recovery_fingertip_target_world
                    .tolist()
                    if (
                        self._as013_scripted_left_recovery_fingertip_target_world
                        is not None
                    )
                    else None
                ),
                "left_fixed_world_anchor_offset_xyz": (
                    self._as013_scripted_left_fixed_anchor_offset_world.tolist()
                ),
                "left_fixed_world_anchor_slide_preloaded": (
                    self._as013_scripted_left_fixed_anchor_slide_preloaded
                ),
                "left_recovery_feature_current_world_xyz": (
                    left_recovery_feature_current_world.tolist()
                    if left_recovery_feature_current_world is not None
                    else None
                ),
                "nearest_left_fingertip_world_xyz": (
                    nearest_left_world.tolist()
                    if nearest_left_world is not None
                    else None
                ),
                "nearest_left_fingertip_body_local_xyz": (
                    nearest_left_local.tolist()
                    if nearest_left_local is not None
                    else None
                ),
                "left_selected_fingertip_index": (
                    int(left_hold_fingertip_index)
                    if left_hold_fingertip_index is not None
                    else None
                ),
                "left_selected_fingertip_body_local_xyz": (
                    nearest_left_tip_body_local.tolist()
                    if nearest_left_tip_body_local is not None
                    else None
                ),
                "left_contact_filter_row_counts": (
                    left_contact_filter_row_counts
                ),
                "right_contact_filter_row_counts": (
                    self._as013_contact_filter_counts(
                        self._as013_right_blade_contact_view,
                        self._as013_right_contact_filter_paths,
                    )
                ),
                "left_named_body_local_xyz": {
                    name: (
                        _invert_pose_matrix(body_matrix)
                        @ np.append(point, 1.0)
                    )[:3].tolist()
                    for name, point in left_named_body_points.items()
                },
                "slider_center_world_xyz": slider_center_world.tolist(),
                "right_hand_point_min_slider_center_distance_m": min(
                    right_distances,
                    default=math.inf,
                ),
                "right_control_finger_number": (
                    self._as013_scripted_right_finger_number
                ),
                "right_requested_finger_number": (
                    requested_right_finger_number
                ),
                "right_finger_extension_fraction": (
                    right_finger_extension_fraction
                ),
                "right_finger_abduction_rad": right_finger_abduction_rad,
                "right_vertical_gain": right_vertical_gain,
                "right_lateral_gain": right_lateral_gain,
                "right_palm_yaw_deg": right_palm_yaw_deg,
                "right_palm_roll_deg": right_palm_roll_deg,
                "legacy_slider_orientation_override": (
                    legacy_slider_orientation_override
                ),
                "slider_orientation_live_certified": False,
                "right_alignment_reference_blend": (
                    right_alignment_reference_blend
                ),
                "right_fallback_target_adapt_frame": (
                    right_fallback_target_adapt_frame
                ),
                "right_fallback_alignment_lock_frame": (
                    right_fallback_alignment_lock_frame
                ),
                "right_slide_palm_travel_m": right_slide_palm_travel_m,
                "right_slide_max_lead_m": right_slide_max_lead_m,
                "right_slide_normal_preload_m": (
                    right_slide_normal_preload_m
                ),
                "right_slide_late_normal_preload_m": (
                    right_slide_late_normal_preload_m
                ),
                "right_slide_late_travel_fraction": (
                    _slide_late_travel_fraction(
                        self._as013_scripted_slide_control_frames,
                        right_slide_control_frames,
                        right_slide_palm_travel_m,
                    )
                ),
                "right_surface_slow_radius_m": right_surface_slow_radius_m,
                "right_slide_cross_edge_max_correction_m": (
                    right_slide_cross_edge_max_correction_m
                ),
                "right_slide_cross_edge_servo_scale": (
                    right_slide_cross_edge_servo_scale
                ),
                "right_slide_yaw_relief_deg": right_slide_yaw_relief_deg,
                "right_slide_pitch_relief_deg": right_slide_pitch_relief_deg,
                "right_slide_roll_relief_deg": right_slide_roll_relief_deg,
                "right_slide_roll_relief_fraction": (
                    _late_slide_roll_relief_fraction(
                        current_blade_progress
                    )
                ),
                "right_slide_finger_extension_delta": (
                    right_slide_finger_extension_delta
                ),
                "right_slide_late_finger_extension_delta": (
                    right_slide_late_finger_extension_delta
                ),
                "right_slide_effective_finger_extension_delta": (
                    _effective_slide_finger_extension_delta(
                        self._as013_scripted_slide_control_frames,
                        right_slide_control_frames,
                        right_slide_palm_travel_m,
                        right_slide_finger_extension_delta,
                        right_slide_late_finger_extension_delta,
                    )
                ),
                "right_slide_finger_extension_fraction": (
                    _slide_finger_extension_fraction(
                        self._as013_scripted_slide_control_frames,
                        right_slide_control_frames,
                        right_slide_palm_travel_m,
                    )
                ),
                "right_slide_finger_abduction_delta_rad": (
                    right_slide_finger_abduction_delta_rad
                ),
                "right_slide_control_frames": right_slide_control_frames,
                "right_slide_reanchor_progress": right_slide_reanchor_progress,
                "right_slide_reanchor_origin_progress": (
                    self._as013_scripted_right_slide_reanchor_origin_progress
                ),
                "right_slide_reanchor_extra_lead_m": (
                    right_slide_reanchor_extra_lead_m
                ),
                "right_slide_reanchor_applied_extra_lead_m": (
                    _right_slide_reanchor_extra_lead(
                        right_slide_reanchor_extra_lead_m,
                        self._as013_scripted_right_slide_reanchor_frame,
                        frame,
                    )
                ),
                "right_slide_servo_correction_scale": (
                    right_slide_servo_correction_scale
                ),
                "right_slide_late_servo_correction_scale": (
                    right_slide_late_servo_correction_scale
                ),
                "right_slide_effective_servo_correction_scale": (
                    _effective_slide_servo_correction_scale(
                        self._as013_scripted_slide_control_frames,
                        right_slide_control_frames,
                        right_slide_palm_travel_m,
                        right_slide_servo_correction_scale,
                        right_slide_late_servo_correction_scale,
                    )
                ),
                "right_recovered_finger_extension_delta": (
                    right_recovered_finger_extension_delta
                ),
                "right_recovered_late_finger_extension_delta": (
                    right_recovered_late_finger_extension_delta
                ),
                "right_recovered_effective_finger_extension_delta": (
                    _effective_recovered_finger_extension_delta(
                        current_blade_progress,
                        right_recovered_finger_extension_delta,
                        right_recovered_late_finger_extension_delta,
                        right_recovered_late_finger_extension_start_progress,
                    )
                ),
                "right_recovered_late_finger_extension_start_progress": (
                    right_recovered_late_finger_extension_start_progress
                ),
                "right_recovered_late_in_plane_preload_m": (
                    right_recovered_late_in_plane_preload_m
                ),
                "right_recovered_late_proximal_extension_scale": (
                    right_recovered_late_proximal_extension_scale
                ),
                "right_recovered_effective_in_plane_preload_m": (
                    _effective_recovered_contact_in_plane_preload_m(
                        current_blade_progress,
                        right_recovered_late_in_plane_preload_m,
                        right_recovered_late_finger_extension_start_progress,
                    )
                ),
                "right_recovered_late_normal_preload_m": (
                    right_recovered_late_normal_preload_m
                ),
                "right_recovered_effective_late_normal_preload_m": (
                    _effective_recovered_late_normal_preload_m(
                        current_blade_progress,
                        right_recovered_late_normal_preload_m,
                        right_recovered_late_finger_extension_start_progress,
                    )
                ),
                "right_recovered_late_finger_abduction_delta_rad": (
                    right_recovered_late_finger_abduction_delta_rad
                ),
                "right_late_continuous_hand_blend_maximum": (
                    right_late_continuous_hand_blend
                ),
                "right_late_continuous_hand_blend": (
                    self._as013_scripted_right_late_continuous_hand_blend
                ),
                "right_recovered_effective_late_finger_abduction_delta_rad": (
                    _effective_recovered_late_finger_abduction_delta(
                        current_blade_progress,
                        right_recovered_late_finger_abduction_delta_rad,
                        right_recovered_late_finger_extension_start_progress,
                    )
                ),
                "right_finger_reentry_control_frames": (
                    right_finger_reentry_control_frames
                ),
                "right_approach_timeout_frame": right_approach_timeout_frame,
                "release_control_frames": release_control_frames,
                "post_success_hold_control_frames": (
                    post_success_hold_control_frames
                ),
                "release_dwell_control_frames": (
                    release_dwell_control_frames
                ),
                "post_success_overtravel_m": post_success_overtravel_m,
                "right_recontact_servo_step_m": right_recontact_servo_step_m,
                "right_recontact_grace_control_frames": (
                    self._as013_right_recontact_grace_control_frames
                ),
                "right_initial_contact_dropout_grace_control_frames": (
                    self._as013_initial_contact_dropout_grace_control_frames
                ),
                "right_initial_contact_dropout_frames": (
                    self._as013_scripted_right_initial_contact_dropout_frames
                ),
                "right_initial_contact_dropout_grace_active": (
                    right_initial_contact_dropout_grace_active
                ),
                "right_late_slide_direct_recontact_active": (
                    right_late_slide_direct_recontact_active
                ),
                "right_last_proven_contact_blade_local_xyz": (
                    self._as013_scripted_right_last_proven_contact_blade_local.tolist()
                    if (
                        self._as013_scripted_right_last_proven_contact_blade_local
                        is not None
                    )
                    else None
                ),
                "late_left_stability_grace_start_progress": (
                    self._as013_late_left_stability_grace_start_progress
                ),
                "right_named_fingertips_blade_local_xyz": {
                    finger_number: (
                        _invert_pose_matrix(blade_matrix)
                        @ np.append(point, 1.0)
                    )[:3].tolist()
                    for finger_number, point in right_named_fingertips.items()
                },
                "right_actual_quat_wxyz": _matrix_to_quat_wxyz(
                    self._native_end_effector_pose_matrix(
                        SLIDER_HAND_SIDE,
                        right_home,
                    )[:3, :3]
                ).tolist(),
                "right_servo_quat_wxyz": (
                    right_servo_pose[3:7].tolist()
                    if right_servo_pose is not None
                    else None
                ),
                "right_arm_joint_diagnostic": right_arm_diagnostic,
                "right_touch_hand_target": right_touch_hand.tolist(),
                "right_control_fingertip_world_xyz": (
                    right_control_fingertip.tolist()
                    if right_control_fingertip is not None
                    else None
                ),
                "right_control_fingertip_blade_local_xyz": (
                    (
                        _invert_pose_matrix(blade_matrix)
                        @ np.append(right_control_fingertip, 1.0)
                    )[:3].tolist()
                    if right_control_fingertip is not None
                    else None
                ),
                "nearest_right_hand_point_world_xyz": (
                    nearest_right_world.tolist()
                    if nearest_right_world is not None
                    else None
                ),
                "nearest_right_hand_point_blade_local_xyz": (
                    nearest_right_local.tolist()
                    if nearest_right_local is not None
                    else None
                ),
                "housing_velocity": (
                    self._physics_body_velocity("E_body_1").tolist()
                    if self._physics_body_velocity("E_body_1") is not None
                    else None
                ),
            }
            print(
                f"[HandsOn/AS013] scripted checkpoint frame={frame}: "
                f"{diagnostic}",
                flush=True,
            )

        if (
            frame >= 60
            and left_servo_pose is not None
            and not self._as013_left_stabilized_seen
        ):
            right_pose = right_home
            left_pose = left_servo_pose
            right_hand = np.zeros(20, dtype=np.float64)
            left_hand = (
                left_grip.copy()
                if self._as013_scripted_left_hold_hand is not None
                else _left_hold_hand_with_retracted_unsafe_finger(
                    (
                        min(1.0, max(0, frame - 60) / 59.0)
                        * left_approach_grip_fraction
                        * _POWER_GRIP
                    ),
                    left_unsafe_finger_retraction_fraction,
                    distal_only=left_unsafe_finger_distal_only,
                )
            )
        elif self._as013_scripted_right_contact_pose is not None:
            right_grip = np.asarray(
                self._as013_scripted_right_contact_hand,
                dtype=np.float64,
            ).reshape(20)
            right_recovered_contact_hand = (
                _recovered_fingertip_hand_from_contact(
                    right_grip,
                    self._as013_scripted_right_finger_number - 1,
                    extension_delta=right_recovered_finger_extension_delta,
                    abduction_delta_rad=(
                        _effective_recovered_late_finger_abduction_delta(
                            current_blade_progress,
                            right_recovered_late_finger_abduction_delta_rad,
                            right_recovered_late_finger_extension_start_progress,
                        )
                    ),
                    late_extension_delta=(
                        _effective_recovered_finger_extension_delta(
                            current_blade_progress,
                            right_recovered_finger_extension_delta,
                            right_recovered_late_finger_extension_delta,
                            right_recovered_late_finger_extension_start_progress,
                        )
                        - right_recovered_finger_extension_delta
                    ),
                    late_proximal_scale=(
                        right_recovered_late_proximal_extension_scale
                    ),
                    late_middle_scale=(
                        right_recovered_late_middle_extension_scale
                    ),
                    late_distal_tip_scale=(
                        right_recovered_late_distal_tip_extension_scale
                    ),
                )
                if self._as013_scripted_right_finger_number is not None
                else right_grip.copy()
            )
            right_slide_contact_hand = (
                _slide_fingertip_hand_from_contact(
                    right_grip,
                    self._as013_scripted_right_finger_number - 1,
                    _slide_finger_extension_fraction(
                        self._as013_scripted_slide_control_frames,
                        right_slide_control_frames,
                        right_slide_palm_travel_m,
                    ),
                    _effective_slide_finger_extension_delta(
                        self._as013_scripted_slide_control_frames,
                        right_slide_control_frames,
                        right_slide_palm_travel_m,
                        right_slide_finger_extension_delta,
                        right_slide_late_finger_extension_delta,
                    ),
                    right_slide_finger_abduction_delta_rad,
                )
                if self._as013_scripted_right_finger_number is not None
                else right_grip.copy()
            )
            right_late_continuous_hand = (
                _blend_recovered_fingertip_hand(
                    right_slide_contact_hand,
                    right_recovered_contact_hand,
                    self._as013_scripted_right_finger_number - 1,
                    self._as013_scripted_right_late_continuous_hand_blend,
                )
                if self._as013_scripted_right_finger_number is not None
                else right_slide_contact_hand
            )
            blade_progress = float(
                self._as013_last_metrics.get(
                    "blade_progress",
                    self._as013_reset_progress,
                )
            )
            contact_ready_now = bool(
                self._as013_last_metrics.get(
                    "left_stable_contact_now",
                    False,
                )
                and float(
                    self._as013_last_metrics.get(
                        "right_slider_contact_force",
                        0.0,
                    )
                )
                > CONTACT_FORCE_THRESHOLD_N
            )
            slide_complete = _slide_complete_from_metrics(
                self._as013_last_metrics
            )
            if (
                slide_complete
                and self._as013_scripted_slide_complete_frame is None
            ):
                self._as013_scripted_slide_complete_frame = frame
            release_ready = _release_ready_after_success_hold(
                self._as013_scripted_slide_complete_frame,
                frame,
                post_success_hold_control_frames,
            )
            right_contact_ready_now = bool(
                float(
                    self._as013_last_metrics.get(
                        "right_slider_contact_force",
                        0.0,
                    )
                )
                > CONTACT_FORCE_THRESHOLD_N
            )
            self._as013_scripted_right_contact_settle_frames = (
                _advance_right_contact_settle_frames(
                    self._as013_scripted_right_contact_settle_frames,
                    contact_ready_now,
                )
            )
            if (
                os.environ.get(SCRIPTED_DIAGNOSTIC_ENV, "")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
                and right_contact_ready_now
            ):
                print(
                    "[HandsOn/AS013] right-contact frame="
                    f"{frame}: "
                    "left_force="
                    f"{float(self._as013_last_metrics.get('left_body_contact_force', 0.0)):.6f} "
                    "right_force="
                    f"{float(self._as013_last_metrics.get('right_slider_contact_force', 0.0)):.6f} "
                    f"left_stable={contact_ready_now} "
                    "left_stable_for_control="
                    f"{current_left_stable_for_control} "
                    "late_left_stability_grace_frames="
                    f"{self._as013_scripted_late_left_stability_grace_frames} "
                    f"settle_frames={self._as013_scripted_right_contact_settle_frames} "
                    f"preload_settling={right_preload_settling_active} "
                    f"slide_servo={right_slide_servo_active} "
                    f"stage={right_servo_stage} "
                    "control_local="
                    f"{right_control_local.tolist() if right_control_fingertip is not None else None} "
                    "recontact_target_local="
                    f"{right_effective_recontact_target_local.tolist() if right_effective_recontact_target_local is not None else None} "
                    "servo_correction="
                    f"{right_servo_correction.tolist() if right_servo_correction is not None else None} "
                    "left_target_xyz="
                    f"{left_hold[:3].tolist()} "
                    "left_actual_xyz="
                    f"{self._native_end_effector_pose_matrix(STABILIZER_HAND_SIDE, left_home)[:3, 3].tolist()} "
                    "left_surface_target_xyz="
                    f"{left_target_world.tolist()} "
                    "left_nearest_tip_xyz="
                    f"{nearest_left_tip.tolist() if nearest_left_tip is not None else None} "
                    "left_hold_fingertip_index="
                    f"{self._as013_scripted_left_hold_fingertip_index} "
                    "left_contact_standoff="
                    f"{self._as013_scripted_left_contact_standoff_m} "
                    "left_effective_standoff="
                    f"{left_hold_effective_standoff_m} "
                    "left_preload="
                    f"{left_hold_preload_m} "
                    "left_preload_correction="
                    f"{left_hold_preload_correction.tolist() if left_hold_preload_correction is not None else None} "
                    "left_hand_actual_mean="
                    f"{float(np.mean(self._actual_hand_joint_pose(STABILIZER_HAND_SIDE))):.6f} "
                    "left_hand_target_mean="
                    f"{float(np.mean(left_grip)):.6f} "
                    "right_recontact_grace_frames="
                    f"{self._as013_scripted_right_recontact_grace_frames} "
                    "right_initial_contact_dropout_frames="
                    f"{self._as013_scripted_right_initial_contact_dropout_frames} "
                    "right_initial_contact_dropout_grace_active="
                    f"{right_initial_contact_dropout_grace_active} "
                    "right_recovered_target_local="
                    f"{self._as013_scripted_right_recovered_target_blade_local.tolist() if self._as013_scripted_right_recovered_target_blade_local is not None else None} "
                    "right_recovered_hand_blend="
                    f"{self._as013_scripted_right_recovered_hand_blend:.3f} "
                    "right_late_continuous_hand_blend="
                    f"{self._as013_scripted_right_late_continuous_hand_blend:.3f} "
                    "right_recovered_target_blend="
                    f"{self._as013_scripted_right_recovered_target_blend:.3f} "
                    "blade_progress="
                    f"{float(self._as013_last_metrics.get('blade_progress', 0.0)):.6f}",
                    flush=True,
                )
            slide_motion_ready_now = bool(
                contact_ready_now
                and self._as013_scripted_right_contact_settle_frames
                >= RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES
            )
            left_actual_matrix = self._native_end_effector_pose_matrix(
                STABILIZER_HAND_SIDE,
                left_home,
            )
            left_actual_pose = np.concatenate(
                [
                    left_actual_matrix[:3, 3],
                    _matrix_to_quat_wxyz(left_actual_matrix[:3, :3]),
                ]
            ).astype(np.float64)
            left_counter_pose = (
                _left_counter_slide_pose(
                    self._as013_scripted_left_counter_slide_anchor_pose,
                    left_hold,
                    left_actual_pose,
                    axis,
                    self._as013_scripted_slide_control_frames,
                    left_counter_slide_m,
                )
                if (
                    left_counter_slide_m > 0.0
                    and self._as013_scripted_left_counter_slide_anchor_pose
                    is not None
                )
                else left_hold
            )
            if (
                os.environ.get(SCRIPTED_DIAGNOSTIC_ENV, "")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
                and (
                    slide_complete
                    or self._as013_scripted_slide_complete_frame is not None
                    or self._as013_scripted_release_frame is not None
                    or frame >= right_approach_timeout_frame
                )
            ):
                print(
                    "[HandsOn/AS013] release frame="
                    f"{frame}: "
                    f"slide_complete={slide_complete} "
                    f"release_ready={release_ready} "
                    "slide_complete_frame="
                    f"{self._as013_scripted_slide_complete_frame} "
                    "release_start_frame="
                    f"{self._as013_scripted_release_frame} "
                    "blade_progress="
                    f"{float(self._as013_last_metrics.get('blade_progress', 0.0)):.6f} "
                    "blade_velocity="
                    f"{float(self._as013_last_metrics.get('blade_velocity', 0.0)):.6f} "
                    "left_force="
                    f"{float(self._as013_last_metrics.get('left_body_contact_force', 0.0)):.6f} "
                    "right_blade_force="
                    f"{float(self._as013_last_metrics.get('right_blade_contact_force', 0.0)):.6f} "
                    "hands_released="
                    f"{bool(self._as013_last_metrics.get('hands_released', False))} "
                    "table_support="
                    f"{bool(self._as013_last_metrics.get('table_support_contact', False))} "
                    f"stable={bool(self._as013_last_metrics.get('stable', False))} "
                    f"failed={bool(self._as013_last_metrics.get('failed', True))} "
                    "success_now="
                    f"{success_from_metrics(self._as013_last_metrics)}",
                    flush=True,
                )
            if release_ready:
                if self._as013_scripted_release_frame is None:
                    self._as013_scripted_release_frame = frame
                fraction = _release_fraction_after_dwell(
                    frame,
                    self._as013_scripted_release_frame,
                    release_control_frames,
                    release_dwell_control_frames,
                )
                right_pose, left_pose, right_hand, left_hand = (
                    self._scripted_timeout_release_action(
                        right_pose=_post_success_overtravel_pose(
                            np.asarray(
                                self._left_tcp_pose,
                                dtype=np.float64,
                            ).reshape(7),
                            axis,
                            post_success_overtravel_m,
                        ),
                        left_pose=np.asarray(
                            self._right_tcp_pose,
                            dtype=np.float64,
                        ).reshape(7),
                        right_hand=np.asarray(
                            self._left_hand,
                            dtype=np.float64,
                        ).reshape(20),
                        left_hand=np.asarray(
                            self._right_hand,
                            dtype=np.float64,
                        ).reshape(20),
                        fraction=fraction,
                    )
                )
            elif frame >= right_approach_timeout_frame:
                fraction = min(
                    1.0,
                    (frame - right_approach_timeout_frame)
                    / float(max(1, release_control_frames - 1)),
                )
                right_pose, left_pose, right_hand, left_hand = (
                    self._scripted_timeout_release_action(
                        right_pose=self._as013_scripted_right_contact_pose,
                        left_pose=left_hold,
                        right_hand=right_grip,
                        left_hand=left_grip,
                        fraction=fraction,
                    )
                )
            elif _right_contact_servo_only_needed(
                right_recontact_active,
                right_preload_settling_active,
                slide_motion_ready_now,
                right_initial_contact_dropout_grace_active,
            ) and right_servo_pose is not None:
                right_pose = right_servo_pose
                left_pose = left_counter_pose
                right_hand = (
                    _blend_recovered_fingertip_hand(
                        right_grip,
                        right_recovered_contact_hand,
                        self._as013_scripted_right_finger_number - 1,
                        self._as013_scripted_right_recovered_hand_blend,
                    )
                    if (
                        right_recontact_grace_active
                        and self._as013_scripted_right_finger_number
                        is not None
                    )
                    else (
                        right_late_continuous_hand
                        if right_initial_contact_dropout_grace_active
                        else (
                            right_slide_contact_hand
                            if right_late_slide_direct_recontact_active
                            else right_finger_reentry_fraction * right_grip
                        )
                    )
                )
                left_hand = left_grip.copy()
            else:
                self._as013_scripted_slide_control_frames = (
                    _advance_slide_control_frames(
                        self._as013_scripted_slide_control_frames,
                        slide_motion_ready_now,
                        right_slide_control_frames,
                    )
                )
                if right_servo_pose is not None:
                    right_pose = _right_slide_servo_step_pose(
                        right_touch,
                        axis,
                        blade_matrix[:3, 2],
                        self._as013_scripted_slide_control_frames,
                        maximum_tangential_distance_m=(
                            _maximum_right_slide_tangential_distance(
                                blade_progress,
                                float(
                                    self._as013_scripted_right_slide_reanchor_origin_progress
                                    if self._as013_scripted_right_slide_reanchor_origin_progress
                                    is not None
                                    else (
                                        self._as013_ordered_contact_progress
                                        if self._as013_ordered_contact_progress
                                        is not None
                                        else blade_progress
                                    )
                                ),
                                float(
                                    self._as013_joint_limits[1]
                                    - self._as013_joint_limits[0]
                                ),
                                right_slide_max_lead_m
                                + _right_slide_reanchor_extra_lead(
                                    right_slide_reanchor_extra_lead_m,
                                    self._as013_scripted_right_slide_reanchor_frame,
                                    frame,
                                ),
                                right_slide_palm_travel_m,
                                self._as013_scripted_left_fallback_grip_active,
                            )
                        ),
                        blade_world_cross_edge=blade_matrix[:3, 0],
                        cross_edge_correction_m=(
                            self._as013_scripted_right_slide_cross_edge_correction_m
                        ),
                        normal_preload_m=right_slide_normal_preload_m,
                        cross_edge_max_correction_m=(
                            right_slide_cross_edge_max_correction_m
                        ),
                        slide_yaw_relief_deg=right_slide_yaw_relief_deg,
                        slide_pitch_relief_deg=right_slide_pitch_relief_deg,
                        slide_roll_relief_deg=right_slide_roll_relief_deg,
                        slide_roll_relief_fraction=(
                            _late_slide_roll_relief_fraction(
                                current_blade_progress
                            )
                        ),
                        slide_roll_pivot_offset_world=(
                            right_control_fingertip
                            - self._native_end_effector_pose_matrix(
                                SLIDER_HAND_SIDE,
                                right_home,
                            )[:3, 3]
                        ),
                        total_control_frames=right_slide_control_frames,
                        palm_travel_m=right_slide_palm_travel_m,
                        late_normal_preload_m=(
                            right_slide_late_normal_preload_m
                        ),
                    )
                    right_pose = _right_slide_pose_with_servo_correction(
                        right_pose,
                        right_servo_correction,
                        _effective_slide_servo_correction_scale(
                            self._as013_scripted_slide_control_frames,
                            right_slide_control_frames,
                            right_slide_palm_travel_m,
                            right_slide_servo_correction_scale,
                            right_slide_late_servo_correction_scale,
                        ),
                        correction_axis_world=axis,
                    )
                    right_pose = _right_slide_pose_with_cross_edge_servo(
                        right_pose,
                        right_control_local,
                        slider_target_local,
                        blade_matrix[:3, 0],
                        right_slide_cross_edge_servo_scale,
                        right_slide_cross_edge_max_correction_m,
                    )
                else:
                    fraction = (
                        self._as013_scripted_slide_control_frames
                        / float(right_slide_control_frames - 1)
                    )
                    right_pose = self._blend_pose(
                        right_touch,
                        right_slide,
                        fraction,
                    )
                left_pose = (
                    _left_counter_slide_pose(
                        self._as013_scripted_left_counter_slide_anchor_pose,
                        left_hold,
                        left_actual_pose,
                        axis,
                        self._as013_scripted_slide_control_frames,
                        left_counter_slide_m,
                    )
                    if (
                        left_counter_slide_m > 0.0
                        and self._as013_scripted_left_counter_slide_anchor_pose
                        is not None
                    )
                    else left_counter_pose
                )
                right_hand = (
                    _blend_recovered_fingertip_hand(
                        right_grip,
                        right_recovered_contact_hand,
                        self._as013_scripted_right_finger_number - 1,
                        self._as013_scripted_right_recovered_hand_blend,
                    )
                    if (
                        right_recontact_grace_active
                        and self._as013_scripted_right_finger_number
                        is not None
                    )
                    else right_late_continuous_hand
                )
                left_hand = left_grip.copy()
        elif frame < 60:
            fraction = frame / 59.0
            right_pose = right_home
            left_pose = self._blend_pose(left_home, left_hover, fraction)
            right_hand = np.zeros(20, dtype=np.float64)
            left_hand = np.zeros(20, dtype=np.float64)
        elif frame < 120:
            right_pose = right_home
            right_hand = np.zeros(20, dtype=np.float64)
            if self._as013_scripted_left_hold_pose is not None:
                left_pose = left_hold
                left_hand = left_grip
            else:
                fraction = (frame - 60) / 59.0
                left_pose = self._blend_pose(left_hover, left_hold, fraction)
                left_hand = fraction * _POWER_GRIP
        elif frame < 180:
            fraction = (frame - 120) / 59.0
            right_pose = self._blend_pose(right_home, right_hover, fraction)
            left_pose = left_hold
            right_hand = np.zeros(20, dtype=np.float64)
            left_hand = left_grip.copy()
        elif frame < 240:
            fraction = (frame - 180) / 59.0
            right_pose = (
                right_servo_pose
                if right_servo_pose is not None
                else self._blend_pose(right_hover, right_touch, fraction)
            )
            left_pose = left_hold
            if not right_fingertips:
                right_hand = fraction * _POWER_GRIP
            elif self._as013_left_stabilized_seen:
                right_hand = fraction * right_touch_hand
            else:
                right_hand = self._actual_hand_joint_pose(SLIDER_HAND_SIDE)
            left_hand = left_grip.copy()
        elif frame < right_approach_timeout_frame and right_fingertips:
            right_pose = (
                right_servo_pose
                if right_servo_pose is not None
                else right_home
            )
            left_pose = left_hold
            right_hand = (
                right_finger_reentry_fraction * right_touch_hand
                if self._as013_left_stabilized_seen
                else (
                    self._actual_hand_joint_pose(SLIDER_HAND_SIDE)
                )
            )
            left_hand = left_grip.copy()
        elif right_fingertips:
            fraction = min(
                1.0,
                (frame - right_approach_timeout_frame)
                / float(max(1, release_control_frames - 1)),
            )
            right_pose, left_pose, right_hand, left_hand = (
                self._scripted_timeout_release_action(
                    right_pose=(
                        right_servo_pose
                        if right_servo_pose is not None
                        else right_home
                    ),
                    left_pose=left_hold,
                    right_hand=self._actual_hand_joint_pose(SLIDER_HAND_SIDE),
                    left_hand=left_grip,
                    fraction=fraction,
                )
            )
        elif frame < 360:
            fraction = (frame - 240) / 119.0
            right_pose = self._blend_pose(right_touch, right_slide, fraction)
            left_pose = left_hold
            right_hand = _POWER_GRIP.copy()
            left_hand = left_grip.copy()
        else:
            fraction = min(1.0, (frame - 360) / 59.0)
            right_release = right_slide.copy()
            right_release[2] += 0.10
            left_release = left_hold.copy()
            left_release[2] += 0.10
            right_pose = self._blend_pose(right_slide, right_release, fraction)
            left_pose = self._blend_pose(left_hold, left_release, fraction)
            right_hand = (1.0 - fraction) * _POWER_GRIP
            left_hand = (1.0 - fraction) * left_grip
        action = np.concatenate(
            # Base dual-arm action order is always physical right, physical
            # left, physical-right hand, physical-left hand.  The local names
            # above intentionally retain their legacy role meaning.
            [left_pose, right_pose, left_hand, right_hand]
        ).astype(np.float32)
        if action.shape != (54,) or not np.isfinite(action).all():
            raise RuntimeError(f"AS013 scripted policy action is invalid: {action.shape}")
        return action

    def scripted_success_action(self, frame_index: int) -> np.ndarray:
        """Return the canonical v5 right-pickup/left-slide action."""

        return self._scripted_bimanual_handheld_success_action(frame_index)


__all__ = [
    "ASSET_PATH",
    "ASSET_METADATA_PATH",
    "ASSET_VARIANT",
    "ASSET_SCALE",
    "ASSET_REST_ROLL_DEG",
    "LEGACY_ASSET_PATH",
    "LEGACY_ASSET_SCALE",
    "LEGACY_ASSET_REST_ROLL_DEG",
    "LEGACY_LEFT_HOUSING_SURFACE_TARGET_LOCAL_M",
    "LEGACY_LEFT_HOUSING_FALLBACK_FEATURE_TARGET_LOCAL_M",
    "LEGACY_LEFT_HOUSING_CONTACT_TRACE_TARGET_LOCAL_YZ_M",
    "LEGACY_RIGHT_FINGERTIP_TARGET_LOCAL_XY_M",
    "ASSET_SOURCE_ARCHIVE_SHA256",
    "ASSET_METADATA_SHA256",
    "ASSET_USDA_SHA256",
    "BUTTON_REGION_CENTER_BODY0_M",
    "BUTTON_REGION_CENTER_LOCAL_M",
    "BLADE_DRIVE_DAMPING_SCALE_ENV",
    "BLADE_DRIVE_STIFFNESS_SCALE_ENV",
    "BLADE_CONTACT_FRICTION_DEFAULT",
    "BLADE_CONTACT_FRICTION_RANGE",
    "HOUSING_CONTACT_STATIC_FRICTION_DEFAULT",
    "HOUSING_CONTACT_DYNAMIC_FRICTION_DEFAULT",
    "HOUSING_CONTACT_STATIC_FRICTION_RANGE",
    "BLADE_JOINT_FRICTION_SCALE_ENV",
    "CAUSAL_FINGERTIP_DRIVE_ENV",
    "JOINT_AXIS_BODY_LOCAL",
    "JOINT_NAME",
    "ROLE_SCHEMA_VERSION",
    "TARGET_DIRECTION_EXTEND",
    "TARGET_DIRECTION_RETRACT",
    "TARGET_ENDPOINT_EXTENDED",
    "TARGET_ENDPOINT_RETRACTED",
    "RESET_EXTEND_PROGRESS_RANGE",
    "RESET_RETRACT_PROGRESS_RANGE",
    "RETRACT_SUCCESS_PROGRESS_MAX",
    "EXTEND_SUCCESS_PROGRESS_MIN",
    "SLIDER_HAND_SIDE",
    "STABILIZER_HAND_SIDE",
    "CONTACT_EVENT_DEBOUNCE_SUBSTEPS",
    "COMPLETION_FORWARD_PROGRESS_EPS",
    "ORDER_VIOLATION_CONSECUTIVE_SUBSTEPS",
    "UNSAFE_BLADE_CONTACT_CONSECUTIVE_SUBSTEPS",
    "RIGHT_HOUSING_SURFACE_TARGET_LOCAL_M",
    "RIGHT_HOUSING_SURFACE_TARGET_LOCAL_X_RANGE_M",
    "RIGHT_HOUSING_CONTACT_TRACE_TARGET_LOCAL_YZ_M",
    "LEFT_COUNTER_SLIDE_CONTROL_FRAMES",
    "LEFT_HOLD_HAND_PRELOAD_CONTROL_FRAMES",
    "LEFT_SLIDER_FINGERTIP_TARGET_LOCAL_XY_M",
    "LEFT_PALM_WORK_HEIGHT_M",
    "LEFT_STABILIZATION_CONSECUTIVE_SUBSTEPS",
    "LEFT_STABLE_CONTACT_CONTROL_GRACE_FRAMES",
    "FINAL_HOLD_DROPOUT_GRACE_CONTROL_FRAMES",
    "STABLE_LINEAR_SPEED_MPS",
    "STABLE_ANGULAR_SPEED_RADPS",
    "LATE_LEFT_STABILITY_GRACE_START_PROGRESS_DEFAULT",
    "LATE_LEFT_STABILITY_GRACE_START_PROGRESS_RANGE",
    "RIGHT_PALM_WORK_HEIGHT_M",
    "RIGHT_IK_DLS_LAMBDA_DEFAULT",
    "RIGHT_IK_DLS_LAMBDA_ENV",
    "RIGHT_SLIDER_CONTACT_CONSECUTIVE_SUBSTEPS",
    "SLIDE_PROGRESS_CONSECUTIVE_SUBSTEPS",
    "SLIDE_PROGRESS_DELTA",
    "SLIDER_BUTTON_LOCAL_MAX_M",
    "SLIDER_BUTTON_LOCAL_MIN_M",
    "SLIDER_CONTACT_REGION_TOLERANCE_M",
    "DROP_TABLE_CONTACT_FORCE_THRESHOLD_N",
    "DROP_TABLE_CONTACT_CONSECUTIVE_SUBSTEPS",
    "SLIDER_CONTACT_DROPOUT_GRACE_SUBSTEPS",
    "SLIDER_ALIGNMENT_REFERENCE_BLEND_DEFAULT",
    "SCRIPTED_RIGHT_SLIDE_CONTROL_FRAMES_ENV",
    "SCRIPTED_RIGHT_SLIDE_CROSS_EDGE_SERVO_SCALE_ENV",
    "SCRIPTED_RIGHT_SLIDE_LATE_NORMAL_PRELOAD_ENV",
    "SCRIPTED_RIGHT_SLIDE_PALM_TRAVEL_ENV",
    "SCRIPTED_RIGHT_SLIDE_REANCHOR_PROGRESS_ENV",
    "SCRIPTED_RIGHT_SLIDE_REANCHOR_EXTRA_LEAD_ENV",
    "SCRIPTED_RIGHT_RECONTACT_GRACE_FRAMES_ENV",
    "SCRIPTED_RIGHT_INITIAL_CONTACT_DROPOUT_GRACE_FRAMES_ENV",
    "SCRIPTED_LATE_LEFT_STABILITY_GRACE_START_PROGRESS_ENV",
    "SCRIPTED_RIGHT_FINGER_REENTRY_CONTROL_FRAMES_ENV",
    "SCRIPTED_POST_SUCCESS_HOLD_CONTROL_FRAMES_ENV",
    "SCRIPTED_RELEASE_DWELL_CONTROL_FRAMES_ENV",
    "SCRIPTED_POST_SUCCESS_OVERTRAVEL_ENV",
    "SCRIPTED_RIGHT_RECONTACT_SERVO_STEP_ENV",
    "SCRIPTED_RIGHT_SLIDE_PITCH_RELIEF_ENV",
    "SCRIPTED_RIGHT_SLIDE_ROLL_RELIEF_ENV",
    "RIGHT_SLIDE_ROLL_RELIEF_DEFAULT_DEG",
    "RIGHT_SLIDE_ROLL_RELIEF_START_PROGRESS",
    "RIGHT_SLIDE_ROLL_RELIEF_END_PROGRESS",
    "SCRIPTED_LEFT_COUNTER_SLIDE_ENV",
    "SCRIPTED_LEFT_APPROACH_GRIP_ENV",
    "SCRIPTED_LEFT_HAND_PRELOAD_ENV",
    "SCRIPTED_LEFT_SAFE_FINGER_PRELOAD_ENV",
    "SCRIPTED_LEFT_UNSAFE_FINGER_RETRACTION_ENV",
    "SCRIPTED_LEFT_UNSAFE_FINGER_DISTAL_ONLY_ENV",
    "TASK_CODE",
    "TASK_NAME",
    "HandsOnAS013IsaacLabEnv",
    "_body_relative_axis_coordinate",
    "_causal_fingertip_drive_target",
    "_advance_late_left_stability_grace_frames",
    "_advance_initial_contact_dropout_grace_frames",
    "_advance_final_hold_dropout_grace_frames",
    "_updated_last_proven_right_contact_local",
    "_latest_right_recontact_target_local",
    "_late_slide_direct_recontact_needed",
    "_right_contact_phase_flags",
    "_release_ready_after_success_hold",
    "_release_fraction_after_dwell",
    "_post_success_overtravel_pose",
    "_slide_complete_from_metrics",
    "_advance_as013_table_contact_event",
    "_advance_as013_slider_contact_observation",
    "_as013_stability_diagnostics",
    "_as013_target_endpoint_reached",
    "_as013_target_motion_delta",
    "_passive_drive_latch_motion_ready",
    "_effective_left_hand_preload_fraction",
    "_left_hold_pose_should_refresh",
    "_left_recovery_servo_should_apply",
    "_left_stabilizer_finger_from_contact_counts",
    "_left_fixed_anchor_recovery_correction",
    "_left_counter_slide_pose",
    "_maximum_right_slide_tangential_distance",
    "_right_slide_pose_with_cross_edge_servo",
    "_late_slide_roll_relief_fraction",
    "sample_reset_contract",
    "success_from_metrics",
]
