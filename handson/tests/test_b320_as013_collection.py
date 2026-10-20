from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml
from handson.runtime.collection_support import ISAACSIM_COLLECTION_TASKS
from handson.sim.isaaclab.abc_collection_env import (
    _C_TOOL_CLEAR_START_TASKS,
    supported_abc_collection_tasks,
    supported_native_collection_tasks,
)
from handson.sim.isaaclab.as013_collection_env import (
    ASSET_METADATA_PATH,
    ASSET_PATH,
    ASSET_SCALE,
    ASSET_SOURCE_ARCHIVE_SHA256,
    ASSET_METADATA_SHA256,
    ASSET_USDA_SHA256,
    ASSET_VARIANT,
    ASSET_REST_ROLL_DEG,
    BLADE_DRIVE_DAMPING_SCALE_ENV,
    BLADE_DRIVE_STIFFNESS_SCALE_ENV,
    BLADE_CONTACT_FRICTION_ENV,
    BLADE_CONTACT_FRICTION_RANGE,
    BLADE_JOINT_DRIVE_DAMPING,
    BLADE_JOINT_DRIVE_MAX_FORCE,
    BLADE_JOINT_DRIVE_STIFFNESS,
    BLADE_JOINT_DYNAMIC_FRICTION,
    BLADE_JOINT_FRICTION,
    BLADE_JOINT_FRICTION_SCALE_ENV,
    CAUSAL_FINGERTIP_DRIVE_ENV,
    CONTACT_FORCE_THRESHOLD_N,
    HOUSING_COLLISION_APPROXIMATION,
    HOUSING_CONTACT_OFFSET_M,
    HOUSING_REST_OFFSET_M,
    JOINT_AXIS_BODY_LOCAL,
    JOINT_NAME,
    LEFT_COUNTER_SLIDE_CONTROL_FRAMES,
    LEFT_HOLD_HAND_PRELOAD_CONTROL_FRAMES,
    LEFT_CARTESIAN_SERVO_STEP_M,
    LEFT_HOLD_HAND_PRELOAD_FRACTION,
    LEFT_HOLD_CONTACT_PRELOAD_M,
    LEFT_SLIDE_HOLD_CONTACT_PRELOAD_M,
    LEFT_HOLD_RECOVERY_STEP_M,
    LEFT_FIXED_ANCHOR_RECOVERY_STEP_M,
    LEFT_FIXED_ANCHOR_MAX_OFFSET_M,
    LEFT_APPROACH_GRIP_FRACTION,
    LEFT_FALLBACK_APPROACH_GRIP_FRACTION,
    LEFT_FALLBACK_GRIP_ACTIVATION_FRAME,
    LEFT_FALLBACK_GRIP_DISTANCE_M,
    LEFT_SAFE_STABILIZER_FINGER_NUMBER,
    LEFT_SAFE_FINGER_PRELOAD_FRACTION,
    LEFT_UNSAFE_BLADE_FINGER_NUMBER,
    LEFT_UNSAFE_FINGER_DISTAL_ONLY,
    LEFT_UNSAFE_FINGER_RETRACTION_CONTROL_FRAMES,
    LEFT_UNSAFE_FINGER_RETRACTION_FRACTION,
    LEFT_UNSAFE_FINGER_RETRACTION_START_FRAME,
    LEFT_HOUSING_CONTACT_TRACE_TARGET_LOCAL_YZ_M,
    LEFT_HOUSING_CONTACT_FINGER_NUMBER,
    LEFT_HOUSING_FALLBACK_FEATURE_TARGET_LOCAL_M,
    LEFT_HOUSING_FALLBACK_INSIDE_TARGET_LOCAL_M,
    LEFT_HOUSING_FALLBACK_INWARD_PRELOAD_M,
    LEFT_HOUSING_FALLBACK_MAX_PRELOAD_STEPS,
    LEFT_HOUSING_FALLBACK_PRELOAD_INTERVAL_FRAMES,
    LEFT_HOUSING_FALLBACK_PRELOAD_START_FRAME,
    LEFT_HOUSING_FALLBACK_READY_DISTANCE_M,
    LEFT_HOUSING_SURFACE_TARGET_LOCAL_M,
    LEFT_HOUSING_SURFACE_TARGET_LOCAL_X_RANGE_M,
    LEFT_HOUSING_SURFACE_PROJECTION_START_FRAME,
    LEFT_RECOVERY_CONTACT_PRELOAD_M,
    LEFT_RECOVERY_ORIENTATION_MIN_FORCE_N,
    LEFT_RECOVERY_READY_CONTROL_FRAMES,
    LEFT_PALM_WORK_HEIGHT_M,
    LEFT_STABILIZATION_CONSECUTIVE_SUBSTEPS,
    LEFT_STABLE_CONTACT_CONTROL_GRACE_FRAMES,
    LATE_LEFT_STABILITY_GRACE_START_PROGRESS_DEFAULT,
    LATE_LEFT_STABILITY_GRACE_START_PROGRESS_RANGE,
    RIGHT_FINGER_EXTENSION_FRACTION,
    RIGHT_FINGER_ABDUCTION_RAD,
    RIGHT_FINGER_NUMBER,
    RIGHT_NONCONTROL_GRIP_FRACTION,
    RIGHT_VERTICAL_GAIN,
    RIGHT_LATERAL_GAIN,
    RIGHT_PALM_YAW_DEG,
    RIGHT_PALM_ROLL_DEG,
    RIGHT_PALM_WORK_HEIGHT_M,
    RIGHT_IK_DLS_LAMBDA_DEFAULT,
    RIGHT_IK_DLS_LAMBDA_ENV,
    RIGHT_SURFACE_APPROACH_SERVO_STEP_M,
    RIGHT_SURFACE_SLOW_RADIUS_M,
    RIGHT_PLANAR_ALIGNMENT_TOL_M,
    RIGHT_PLANAR_ENTRY_HEIGHT_M,
    RIGHT_VERTICAL_ENTRY_TOL_M,
    RIGHT_RECONTACT_CLEAR_HEIGHT_M,
    RIGHT_RECONTACT_HEIGHT_M,
    RIGHT_RECONTACT_LIFT_SERVO_STEP_M,
    RIGHT_RECONTACT_CONTACT_PRELOAD_M,
    RIGHT_RECONTACT_NORMAL_PRELOAD_M,
    RIGHT_RECOVERED_CONTACT_NORMAL_PRELOAD_M,
    RIGHT_RECOVERED_CAPTURE_IN_PLANE_PRELOAD_M,
    RIGHT_RECOVERED_CAPTURE_CROSS_EDGE_PRELOAD_M,
    RIGHT_RECOVERED_CONTACT_IN_PLANE_PRELOAD_M,
    RIGHT_RECOVERED_LATE_IN_PLANE_PRELOAD_CONFIG_MAX_M,
    RIGHT_RECOVERED_LATE_NORMAL_PRELOAD_CONFIG_MAX_M,
    RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_CONFIG_ABS_MAX_RAD,
    RIGHT_RECOVERED_FINGER_EXTENSION_DELTA,
    RIGHT_RECOVERED_FINGER_EXTENSION_CONFIG_MAX,
    RIGHT_RECOVERED_FINGER_EXTENSION_DELTA_DEFAULT,
    RIGHT_RECOVERED_LATE_FINGER_EXTENSION_DELTA_DEFAULT,
    RIGHT_RECOVERED_LATE_FINGER_EXTENSION_CONFIG_ABS_MAX,
    RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS,
    RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS_DEFAULT,
    RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE_DEFAULT,
    RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_DELTA_RAD_DEFAULT,
    RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS_RANGE,
    RIGHT_RECOVERED_LATE_PROXIMAL_EXTENSION_SCALE_RANGE,
    RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE_RANGE,
    RIGHT_RECOVERED_LATE_DISTAL_TIP_EXTENSION_SCALE_RANGE,
    RIGHT_RECOVERED_HAND_BLEND_CONTROL_FRAMES,
    RIGHT_LATE_CONTINUOUS_HAND_BLEND_CONTROL_FRAMES,
    RIGHT_RECONTACT_CONTACT_GRACE_CONTROL_FRAMES,
    RIGHT_RECONTACT_ADVECTION_FRACTION,
    RIGHT_RECOVERED_TARGET_BLEND_CONTROL_FRAMES,
    RIGHT_RECOVERED_TARGET_MAX_BLEND,
    RIGHT_RECONTACT_WAIT_HEIGHT_M,
    RIGHT_FINGER_REENTRY_CONTROL_FRAMES,
    RIGHT_SLIDE_CONTROL_FRAMES,
    RIGHT_SLIDE_REANCHOR_EXTRA_LEAD_RAMP_CONTROL_FRAMES,
    RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES,
    RIGHT_SLIDE_CROSS_EDGE_BLEND_CONTROL_FRAMES,
    RIGHT_SLIDE_CROSS_EDGE_CONFIG_MAX_M,
    RIGHT_SLIDE_CROSS_EDGE_MAX_CORRECTION_M,
    RIGHT_SLIDE_MAX_LEAD_M,
    RIGHT_SLIDE_NORMAL_PRELOAD_M,
    RIGHT_SLIDE_LATE_NORMAL_PRELOAD_M,
    RIGHT_SLIDE_PALM_TRAVEL_M,
    RIGHT_SLIDE_PALM_TRAVEL_DEFAULT_M,
    RIGHT_SLIDE_PALM_TRAVEL_CONFIG_MAX_M,
    RIGHT_SLIDE_MAX_LEAD_DEFAULT_M,
    RIGHT_SLIDE_ROLL_RELIEF_DEFAULT_DEG,
    RIGHT_SLIDE_ROLL_RELIEF_START_PROGRESS,
    RIGHT_SLIDE_ROLL_RELIEF_END_PROGRESS,
    RIGHT_SLIDE_FINGER_EXTENSION_DELTA_DEFAULT,
    RIGHT_SLIDE_FINGER_ABDUCTION_DELTA_RAD_DEFAULT,
    RIGHT_SLIDE_CONTROL_FRAMES_DEFAULT,
    RIGHT_SLIDE_CONTROL_FRAMES_CONFIG_MAX,
    SCRIPTED_RIGHT_SLIDE_MAX_LEAD_ENV,
    SCRIPTED_RIGHT_ALIGNMENT_REFERENCE_BLEND_ENV,
    SCRIPTED_RIGHT_FALLBACK_TARGET_ADAPT_FRAME_ENV,
    SCRIPTED_RIGHT_FALLBACK_ALIGNMENT_LOCK_FRAME_ENV,
    SCRIPTED_RIGHT_SLIDE_PALM_TRAVEL_ENV,
    SCRIPTED_RIGHT_SLIDE_NORMAL_PRELOAD_ENV,
    SCRIPTED_RIGHT_SLIDE_LATE_NORMAL_PRELOAD_ENV,
    SCRIPTED_RIGHT_SURFACE_SLOW_RADIUS_ENV,
    SCRIPTED_RIGHT_SLIDE_CROSS_EDGE_MAX_CORRECTION_ENV,
    SCRIPTED_RIGHT_SLIDE_CROSS_EDGE_SERVO_SCALE_ENV,
    SCRIPTED_RIGHT_SLIDE_YAW_RELIEF_ENV,
    SCRIPTED_RIGHT_SLIDE_PITCH_RELIEF_ENV,
    SCRIPTED_RIGHT_SLIDE_ROLL_RELIEF_ENV,
    SCRIPTED_RIGHT_SLIDE_FINGER_EXTENSION_DELTA_ENV,
    SCRIPTED_RIGHT_SLIDE_LATE_FINGER_EXTENSION_DELTA_ENV,
    SCRIPTED_RIGHT_SLIDE_FINGER_ABDUCTION_DELTA_ENV,
    SCRIPTED_RIGHT_SLIDE_CONTROL_FRAMES_ENV,
    SCRIPTED_RIGHT_SLIDE_REANCHOR_PROGRESS_ENV,
    SCRIPTED_RIGHT_SLIDE_REANCHOR_EXTRA_LEAD_ENV,
    SCRIPTED_RIGHT_SLIDE_SERVO_CORRECTION_SCALE_ENV,
    SCRIPTED_RIGHT_SLIDE_LATE_SERVO_CORRECTION_SCALE_ENV,
    SCRIPTED_RIGHT_RECONTACT_GRACE_FRAMES_ENV,
    SCRIPTED_RIGHT_INITIAL_CONTACT_DROPOUT_GRACE_FRAMES_ENV,
    SCRIPTED_LATE_LEFT_STABILITY_GRACE_START_PROGRESS_ENV,
    SCRIPTED_RIGHT_FINGER_REENTRY_CONTROL_FRAMES_ENV,
    SCRIPTED_RIGHT_APPROACH_TIMEOUT_FRAME_ENV,
    SCRIPTED_RELEASE_CONTROL_FRAMES_ENV,
    SCRIPTED_RIGHT_RECOVERED_FINGER_EXTENSION_DELTA_ENV,
    SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_EXTENSION_DELTA_ENV,
    SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS_ENV,
    SCRIPTED_RIGHT_RECOVERED_LATE_PROXIMAL_EXTENSION_SCALE_ENV,
    SCRIPTED_RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE_ENV,
    SCRIPTED_RIGHT_RECOVERED_LATE_DISTAL_TIP_EXTENSION_SCALE_ENV,
    SCRIPTED_RIGHT_RECOVERED_LATE_IN_PLANE_PRELOAD_ENV,
    SCRIPTED_RIGHT_RECOVERED_LATE_NORMAL_PRELOAD_ENV,
    SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_DELTA_ENV,
    SCRIPTED_RIGHT_LATE_CONTINUOUS_HAND_BLEND_ENV,
    SCRIPTED_POST_SUCCESS_HOLD_CONTROL_FRAMES_ENV,
    SCRIPTED_RELEASE_DWELL_CONTROL_FRAMES_ENV,
    SCRIPTED_POST_SUCCESS_OVERTRAVEL_ENV,
    SCRIPTED_RIGHT_RECONTACT_SERVO_STEP_ENV,
    SCRIPTED_LEFT_COUNTER_SLIDE_ENV,
    SCRIPTED_LEFT_APPROACH_GRIP_ENV,
    SCRIPTED_LEFT_HAND_PRELOAD_ENV,
    SCRIPTED_LEFT_SAFE_FINGER_PRELOAD_ENV,
    SCRIPTED_LEFT_UNSAFE_FINGER_RETRACTION_ENV,
    SCRIPTED_LEFT_UNSAFE_FINGER_DISTAL_ONLY_ENV,
    RIGHT_CARTESIAN_SERVO_STEP_M,
    RIGHT_CONTACT_PRELOAD_SERVO_STEP_M,
    RIGHT_APPROACH_TIMEOUT_FRAME,
    RIGHT_APPROACH_TIMEOUT_FRAME_DEFAULT,
    RIGHT_RELEASE_CONTROL_FRAMES,
    RIGHT_RELEASE_CONTROL_FRAMES_DEFAULT,
    RIGHT_RELEASE_CONTROL_FRAMES_MIN,
    RIGHT_INITIAL_CONTACT_DROPOUT_GRACE_CONTROL_FRAMES_DEFAULT,
    RIGHT_POST_SUCCESS_HOLD_CONTROL_FRAMES,
    RIGHT_RELEASE_DWELL_CONTROL_FRAMES_DEFAULT,
    RIGHT_POST_SUCCESS_OVERTRAVEL_DEFAULT_M,
    RIGHT_RELEASE_STABLE_RESERVE_CONTROL_FRAMES,
    RIGHT_SLIDER_CONTACT_CONSECUTIVE_SUBSTEPS,
    RIGHT_FINGERTIP_TARGET_CLEARANCE_M,
    RIGHT_FINGERTIP_TARGET_INSET_M,
    RIGHT_FINGERTIP_TARGET_LOCAL_XY_M,
    RIGHT_CERTIFIED_ALIGNMENT_QUAT_WXYZ,
    RIGHT_ALIGNMENT_REFERENCE_BLEND_DEFAULT,
    RESET_ANGULAR_DAMPING,
    RESET_LINEAR_DAMPING,
    RESET_SLEEP_THRESHOLD,
    SLIDER_BUTTON_LOCAL_MAX_M,
    SLIDER_BUTTON_LOCAL_MIN_M,
    SLIDE_PROGRESS_CONSECUTIVE_SUBSTEPS,
    SLIDE_PROGRESS_DELTA,
    TASK_NAME,
    TARGET_TABLE_CLEARANCE_M,
    HandsOnAS013IsaacLabEnv,
    _FINGERTIP_EXTENSION,
    _POWER_GRIP,
    _advance_slide_control_frames,
    _advance_left_recovery_ready_frames,
    _advance_late_left_stability_grace_frames,
    _release_ready_after_success_hold,
    _release_fraction_after_dwell,
    _post_success_overtravel_pose,
    _slide_complete_from_metrics,
    _passive_drive_latch_motion_ready,
    _advance_right_contact_settle_frames,
    _advance_right_recontact_grace_frames,
    _advance_initial_contact_dropout_grace_frames,
    _updated_last_proven_right_contact_local,
    _latest_right_recontact_target_local,
    _late_slide_direct_recontact_needed,
    _advance_recovered_right_hand_blend,
    _advance_late_continuous_right_hand_blend,
    _advance_recovered_target_blend,
    _advect_right_recontact_target_local,
    _bounded_right_servo_correction,
    _contact_filter_counts_by_path,
    _body_relative_axis_coordinate,
    _configured_blade_contact_friction,
    _adapt_fallback_right_target_cross_edge,
    _causal_fingertip_drive_target,
    _effective_left_hand_preload_fraction,
    _left_hold_pose_should_refresh,
    _left_fallback_grip_should_activate,
    _left_recovery_servo_should_apply,
    _left_stabilizer_finger_from_contact_counts,
    _left_counter_slide_pose,
    _project_left_housing_surface_target_local,
    _preload_left_housing_surface_target_local,
    _latest_approach_timeout_frame,
    _capture_right_recontact_target_local,
    _left_hold_preload_correction,
    _left_hold_effective_standoff,
    _left_fixed_anchor_recovery_correction,
    _left_hold_hand_with_retracted_unsafe_finger,
    _left_hold_hand_with_reinforced_safe_finger,
    _advance_left_fixed_anchor_offset,
    _left_fixed_anchor_slide_preload_target,
    _left_recovery_preload_target,
    _select_left_hold_fingertip,
    _settling_right_recontact_target_local,
    _preload_lost_right_contact_normal,
    _preload_recovered_right_contact_normal,
    _preload_recovered_capture_in_plane,
    _preload_recovered_capture_cross_edge,
    _preload_recovered_right_contact_in_plane,
    _recovered_fingertip_hand_from_contact,
    _effective_recovered_finger_extension_delta,
    _effective_recovered_contact_in_plane_preload_m,
    _effective_recovered_late_normal_preload_m,
    _effective_recovered_late_finger_abduction_delta,
    _blend_recovered_fingertip_hand,
    _fixed_right_fingertip_target_local,
    _yaw_rotated_world_rotation,
    _certified_right_alignment_world_rotation,
    _blended_right_alignment_world_rotation,
    _preload_right_recontact_target_local,
    _right_contact_pose_ready,
    _right_contact_servo_only_needed,
    _right_contact_phase_flags,
    _right_finger_withdrawal_needed,
    _right_finger_reentry_fraction,
    _right_recontact_needed,
    _recontact_right_fingertip_target_local,
    _right_servo_max_step,
    _right_servo_state_max_step,
    _right_slide_reanchor_needed,
    _right_slide_reanchor_extra_lead,
    _right_slide_pose_with_servo_correction,
    _right_slide_pose_with_cross_edge_servo,
    _right_touch_hand_posture,
    _right_slide_contact_pose_refresh_needed,
    _maximum_right_slide_tangential_distance,
    _effective_slide_servo_correction_scale,
    _right_slide_servo_step_pose,
    _reanchored_right_contact_pose_from_command,
    _slide_fingertip_hand_from_contact,
    _effective_slide_finger_extension_delta,
    _slide_finger_extension_fraction,
    _slide_late_travel_fraction,
    _late_slide_roll_relief_fraction,
    _staged_right_fingertip_target_local,
    sample_reset_contract,
    success_from_metrics,
)
from handson.sim.isaaclab.recording_origin import (
    stable_recording_origin_kwargs_for_task,
)
from handson.tasks.mappings import CONFIG_MAPPING

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = (
    REPOSITORY_ROOT
    / "taskdev"
    / "tasks"
    / "bench320_as013_utility_knife_body_stabilized_blade_slide"
    / "assets"
    / ASSET_VARIANT
)
EXPECTED_ASSET_HASHES = {
    "source/utility_knife.zip": "85fe8c49f43383aa15e5ea9803dfca52fec196d3dc00424ea8eaaca326aa648b",
    "source/sketchfab_utility_knife/license.txt": "5f89a19cb535af5dd35016f8315ff3c4ea1cc07bd58da421f5eb7a05642eddd1",
    "source/sketchfab_utility_knife/scene.bin": "5b41b20a8a2a0c523fcb06dcca2da09301dbebbb7ea09cb37a14becef81755ba",
    "source/sketchfab_utility_knife/scene.gltf": "7e155ea959934f7d37380e1d0eb5f35d0d36076dd71d063f68b66acc5b8100ee",
    "source/sketchfab_utility_knife/textures/Material.001_baseColor.png": "fea7c19eb3a7898bdf6edf231eefe110ebd2a0c09dc519a40f4791d2b0ae507b",
    "source/sketchfab_utility_knife/textures/Material.001_metallicRoughness.png": "a2e686f398053f2ae43aa98eaab9e2c33f4f695a8b94cbb43a8a237abf8066ca",
    "source/sketchfab_utility_knife/textures/Material.001_normal.png": "40f7b47ae84a175ec7bb83256e2f6bee03d9e663ec2a598c645ee59d468b04d7",
    "runtime/utility_knife_ahtusik.usda": "332f1b717677caf73ccbff6fd48debd3c862fb729960b373909692d12f68dfa0",
    "runtime/utility_knife_ahtusik_metadata.json": "f67a448283852e90a4349947a0e342245f04a65f3533eaa78621947218f725f6",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_as013_identity_and_collection_registration() -> None:
    assert TASK_NAME in CONFIG_MAPPING
    assert CONFIG_MAPPING[TASK_NAME].task_id == "40"
    assert TASK_NAME in ISAACSIM_COLLECTION_TASKS
    assert TASK_NAME in CONFIG_MAPPING
    assert TASK_NAME in supported_native_collection_tasks()
    assert TASK_NAME not in _C_TOOL_CLEAR_START_TASKS
    assert BLADE_JOINT_DRIVE_STIFFNESS == pytest.approx(80.0)
    assert BLADE_JOINT_DRIVE_DAMPING == pytest.approx(25.0)
    assert BLADE_JOINT_DRIVE_MAX_FORCE == pytest.approx(12.0)
    assert BLADE_JOINT_FRICTION == pytest.approx(3.0)
    assert BLADE_JOINT_DYNAMIC_FRICTION == pytest.approx(2.4)
    assert (
        BLADE_DRIVE_STIFFNESS_SCALE_ENV
        == "HANDSON_AS013_BLADE_DRIVE_STIFFNESS_SCALE"
    )
    assert (
        BLADE_DRIVE_DAMPING_SCALE_ENV
        == "HANDSON_AS013_BLADE_DRIVE_DAMPING_SCALE"
    )
    assert (
        BLADE_JOINT_FRICTION_SCALE_ENV
        == "HANDSON_AS013_BLADE_JOINT_FRICTION_SCALE"
    )
    assert LEFT_COUNTER_SLIDE_CONTROL_FRAMES == 120
    assert LEFT_HOLD_HAND_PRELOAD_CONTROL_FRAMES == 30
    assert (
        SCRIPTED_LEFT_COUNTER_SLIDE_ENV
        == "HANDSON_AS013_SCRIPTED_LEFT_COUNTER_SLIDE_M"
    )
    assert (
        SCRIPTED_LEFT_HAND_PRELOAD_ENV
        == "HANDSON_AS013_SCRIPTED_LEFT_HAND_PRELOAD_FRACTION"
    )
    assert SCRIPTED_LEFT_SAFE_FINGER_PRELOAD_ENV == (
        "HANDSON_AS013_SCRIPTED_LEFT_SAFE_FINGER_PRELOAD_FRACTION"
    )
    assert SCRIPTED_LEFT_UNSAFE_FINGER_RETRACTION_ENV == (
        "HANDSON_AS013_SCRIPTED_LEFT_UNSAFE_FINGER_RETRACTION_FRACTION"
    )
    assert SCRIPTED_LEFT_UNSAFE_FINGER_DISTAL_ONLY_ENV == (
        "HANDSON_AS013_SCRIPTED_LEFT_UNSAFE_FINGER_DISTAL_ONLY"
    )
    assert (
        CAUSAL_FINGERTIP_DRIVE_ENV
        == "HANDSON_AS013_CAUSAL_FINGERTIP_DRIVE"
    )
    assert RIGHT_IK_DLS_LAMBDA_ENV == "HANDSON_AS013_RIGHT_IK_DLS_LAMBDA"
    assert RIGHT_IK_DLS_LAMBDA_DEFAULT == pytest.approx(0.02)
    assert (
        SCRIPTED_RIGHT_RECONTACT_GRACE_FRAMES_ENV
        == "HANDSON_AS013_SCRIPTED_RIGHT_RECONTACT_GRACE_FRAMES"
    )
    assert SCRIPTED_RIGHT_INITIAL_CONTACT_DROPOUT_GRACE_FRAMES_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_INITIAL_CONTACT_DROPOUT_GRACE_FRAMES"
    )
    assert SCRIPTED_LATE_LEFT_STABILITY_GRACE_START_PROGRESS_ENV == (
        "HANDSON_AS013_SCRIPTED_LATE_LEFT_STABILITY_GRACE_START_PROGRESS"
    )
    assert SCRIPTED_RIGHT_FINGER_REENTRY_CONTROL_FRAMES_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_FINGER_REENTRY_CONTROL_FRAMES"
    )
    assert LEFT_CARTESIAN_SERVO_STEP_M == pytest.approx(0.005)
    assert LEFT_HOLD_HAND_PRELOAD_FRACTION == pytest.approx(0.0)
    assert LEFT_HOLD_CONTACT_PRELOAD_M == pytest.approx(0.001)
    assert LEFT_SLIDE_HOLD_CONTACT_PRELOAD_M == pytest.approx(0.002)
    assert LEFT_HOLD_RECOVERY_STEP_M == pytest.approx(0.005)
    assert LEFT_FIXED_ANCHOR_RECOVERY_STEP_M == pytest.approx(0.001)
    assert LEFT_RECOVERY_CONTACT_PRELOAD_M == pytest.approx(0.0005)
    assert LEFT_RECOVERY_ORIENTATION_MIN_FORCE_N == pytest.approx(1.0)
    assert LEFT_RECOVERY_READY_CONTROL_FRAMES == 15
    assert LEFT_STABLE_CONTACT_CONTROL_GRACE_FRAMES == 15
    assert LATE_LEFT_STABILITY_GRACE_START_PROGRESS_DEFAULT == pytest.approx(
        0.75
    )
    assert LATE_LEFT_STABILITY_GRACE_START_PROGRESS_RANGE == pytest.approx(
        (0.75, 0.80)
    )
    assert RIGHT_FINGER_REENTRY_CONTROL_FRAMES == 30
    assert RIGHT_FINGER_EXTENSION_FRACTION == pytest.approx(0.67)
    assert RIGHT_FINGER_ABDUCTION_RAD == pytest.approx(-0.20)
    assert RIGHT_FINGER_NUMBER == 1
    assert RIGHT_VERTICAL_GAIN == pytest.approx(2.5)
    assert RIGHT_LATERAL_GAIN == pytest.approx(2.0)
    assert RIGHT_PALM_YAW_DEG == pytest.approx(25.0)
    assert RIGHT_PALM_ROLL_DEG == pytest.approx(15.0)
    assert RIGHT_CARTESIAN_SERVO_STEP_M == pytest.approx(0.015)
    assert RIGHT_RECONTACT_WAIT_HEIGHT_M == pytest.approx(0.030)
    assert RIGHT_RECONTACT_CONTACT_PRELOAD_M == pytest.approx(0.003)
    assert RIGHT_RECONTACT_NORMAL_PRELOAD_M == pytest.approx(0.0004)
    assert RIGHT_RECOVERED_CONTACT_NORMAL_PRELOAD_M == pytest.approx(0.0002)
    assert RIGHT_RECOVERED_CAPTURE_IN_PLANE_PRELOAD_M == pytest.approx(0.0002)
    assert RIGHT_RECOVERED_CAPTURE_CROSS_EDGE_PRELOAD_M == pytest.approx(
        0.0
    )
    assert RIGHT_RECOVERED_CONTACT_IN_PLANE_PRELOAD_M == pytest.approx(0.001)
    assert RIGHT_RECOVERED_LATE_IN_PLANE_PRELOAD_CONFIG_MAX_M == (
        pytest.approx(0.003)
    )
    assert RIGHT_RECOVERED_LATE_NORMAL_PRELOAD_CONFIG_MAX_M == (
        pytest.approx(0.0005)
    )
    assert RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_CONFIG_ABS_MAX_RAD == (
        pytest.approx(0.05)
    )
    assert RIGHT_RECOVERED_FINGER_EXTENSION_DELTA == pytest.approx(0.12)
    assert RIGHT_RECOVERED_FINGER_EXTENSION_CONFIG_MAX == pytest.approx(0.20)
    assert RIGHT_RECOVERED_FINGER_EXTENSION_DELTA_DEFAULT == pytest.approx(0.16)
    assert RIGHT_RECOVERED_LATE_FINGER_EXTENSION_DELTA_DEFAULT == pytest.approx(
        0.024
    )
    assert RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS == (
        pytest.approx(0.70)
    )
    assert RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS_DEFAULT == (
        pytest.approx(0.75)
    )
    assert RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE_DEFAULT == pytest.approx(
        2.23
    )
    assert RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_DELTA_RAD_DEFAULT == (
        pytest.approx(-0.0125)
    )
    assert RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS_RANGE == (
        pytest.approx((0.65, 0.79))
    )
    assert RIGHT_RECOVERED_LATE_FINGER_EXTENSION_CONFIG_ABS_MAX == (
        pytest.approx(0.05)
    )
    assert RIGHT_RECOVERED_LATE_PROXIMAL_EXTENSION_SCALE_RANGE == (
        pytest.approx((0.0, 2.0))
    )
    assert RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE_RANGE == (
        pytest.approx((0.0, 4.0))
    )
    assert RIGHT_RECOVERED_LATE_DISTAL_TIP_EXTENSION_SCALE_RANGE == (
        pytest.approx((0.0, 2.0))
    )
    assert SCRIPTED_RIGHT_RECOVERED_FINGER_EXTENSION_DELTA_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_FINGER_EXTENSION_DELTA"
    )
    assert SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_EXTENSION_DELTA_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_EXTENSION_DELTA"
    )
    assert (
        SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS_ENV
        == "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_EXTENSION_START_PROGRESS"
    )
    assert SCRIPTED_RIGHT_RECOVERED_LATE_PROXIMAL_EXTENSION_SCALE_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_PROXIMAL_EXTENSION_SCALE"
    )
    assert SCRIPTED_RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_MIDDLE_EXTENSION_SCALE"
    )
    assert SCRIPTED_RIGHT_RECOVERED_LATE_DISTAL_TIP_EXTENSION_SCALE_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_DISTAL_TIP_EXTENSION_SCALE"
    )
    assert SCRIPTED_RIGHT_RECOVERED_LATE_IN_PLANE_PRELOAD_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_IN_PLANE_PRELOAD_M"
    )
    assert SCRIPTED_RIGHT_RECOVERED_LATE_NORMAL_PRELOAD_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_NORMAL_PRELOAD_M"
    )
    assert SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_DELTA_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_RECOVERED_LATE_FINGER_ABDUCTION_DELTA_RAD"
    )
    assert RIGHT_RECOVERED_HAND_BLEND_CONTROL_FRAMES == 10
    assert RIGHT_RECONTACT_CONTACT_GRACE_CONTROL_FRAMES == 30
    assert RIGHT_RECONTACT_ADVECTION_FRACTION == pytest.approx(1.0)
    assert RIGHT_RECOVERED_TARGET_BLEND_CONTROL_FRAMES == 10
    assert RIGHT_RECOVERED_TARGET_MAX_BLEND == pytest.approx(1.0)
    assert RIGHT_CONTACT_PRELOAD_SERVO_STEP_M == pytest.approx(0.006)
    assert RIGHT_SLIDE_PALM_TRAVEL_M == pytest.approx(0.055)
    assert RIGHT_SLIDE_PALM_TRAVEL_DEFAULT_M == pytest.approx(0.107)
    assert RIGHT_SLIDE_PALM_TRAVEL_CONFIG_MAX_M == pytest.approx(0.120)
    assert SCRIPTED_RIGHT_SLIDE_PALM_TRAVEL_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_PALM_TRAVEL_M"
    )
    assert RIGHT_SLIDE_NORMAL_PRELOAD_M == pytest.approx(0.0012)
    assert RIGHT_SLIDE_LATE_NORMAL_PRELOAD_M == pytest.approx(0.00056)
    assert RIGHT_SLIDE_MAX_LEAD_M == pytest.approx(0.012)
    assert RIGHT_SLIDE_MAX_LEAD_DEFAULT_M == pytest.approx(0.016)
    assert RIGHT_SLIDE_FINGER_EXTENSION_DELTA_DEFAULT == pytest.approx(-0.03)
    assert RIGHT_SLIDE_FINGER_ABDUCTION_DELTA_RAD_DEFAULT == pytest.approx(
        0.01
    )
    assert (
        SCRIPTED_RIGHT_SLIDE_MAX_LEAD_ENV
        == "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_MAX_LEAD_M"
    )
    assert (
        SCRIPTED_RIGHT_SLIDE_NORMAL_PRELOAD_ENV
        == "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_NORMAL_PRELOAD_M"
    )
    assert SCRIPTED_RIGHT_SLIDE_LATE_NORMAL_PRELOAD_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_LATE_NORMAL_PRELOAD_M"
    )
    assert (
        SCRIPTED_RIGHT_SURFACE_SLOW_RADIUS_ENV
        == "HANDSON_AS013_SCRIPTED_RIGHT_SURFACE_SLOW_RADIUS_M"
    )
    assert RIGHT_SURFACE_SLOW_RADIUS_M == pytest.approx(0.010)
    assert (
        SCRIPTED_RIGHT_SLIDE_CROSS_EDGE_MAX_CORRECTION_ENV
        == "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_CROSS_EDGE_MAX_CORRECTION_M"
    )
    assert SCRIPTED_RIGHT_SLIDE_CROSS_EDGE_SERVO_SCALE_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_CROSS_EDGE_SERVO_SCALE"
    )
    assert (
        SCRIPTED_RIGHT_SLIDE_YAW_RELIEF_ENV
        == "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_YAW_RELIEF_DEG"
    )
    assert (
        SCRIPTED_RIGHT_SLIDE_PITCH_RELIEF_ENV
        == "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_PITCH_RELIEF_DEG"
    )
    assert (
        SCRIPTED_RIGHT_SLIDE_ROLL_RELIEF_ENV
        == "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_ROLL_RELIEF_DEG"
    )
    assert (
        SCRIPTED_RIGHT_SLIDE_FINGER_EXTENSION_DELTA_ENV
        == "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_FINGER_EXTENSION_DELTA"
    )
    assert SCRIPTED_RIGHT_SLIDE_LATE_FINGER_EXTENSION_DELTA_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_LATE_FINGER_EXTENSION_DELTA"
    )
    assert (
        SCRIPTED_RIGHT_SLIDE_FINGER_ABDUCTION_DELTA_ENV
        == "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_FINGER_ABDUCTION_DELTA_RAD"
    )
    assert (
        SCRIPTED_RIGHT_SLIDE_CONTROL_FRAMES_ENV
        == "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_CONTROL_FRAMES"
    )
    assert SCRIPTED_RIGHT_SLIDE_SERVO_CORRECTION_SCALE_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_SERVO_CORRECTION_SCALE"
    )
    assert SCRIPTED_RIGHT_SLIDE_LATE_SERVO_CORRECTION_SCALE_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_LATE_SERVO_CORRECTION_SCALE"
    )
    assert RIGHT_SLIDE_CROSS_EDGE_MAX_CORRECTION_M == pytest.approx(0.006)
    assert RIGHT_SLIDE_CROSS_EDGE_CONFIG_MAX_M == pytest.approx(0.012)
    assert RIGHT_SLIDE_CROSS_EDGE_BLEND_CONTROL_FRAMES == 30
    assert RIGHT_SLIDE_CONTROL_FRAMES == 360
    assert RIGHT_SLIDE_CONTROL_FRAMES_DEFAULT == 630
    assert RIGHT_SLIDE_CONTROL_FRAMES_CONFIG_MAX == 720
    assert RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES == 1
    assert RIGHT_APPROACH_TIMEOUT_FRAME == 1600
    assert RIGHT_APPROACH_TIMEOUT_FRAME_DEFAULT == 1765
    assert SCRIPTED_RIGHT_APPROACH_TIMEOUT_FRAME_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_APPROACH_TIMEOUT_FRAME"
    )
    assert SCRIPTED_RELEASE_CONTROL_FRAMES_ENV == (
        "HANDSON_AS013_SCRIPTED_RELEASE_CONTROL_FRAMES"
    )
    assert RIGHT_RELEASE_CONTROL_FRAMES == 60
    assert RIGHT_RELEASE_CONTROL_FRAMES_DEFAULT == 15
    assert RIGHT_RELEASE_CONTROL_FRAMES_MIN == 15
    assert RIGHT_INITIAL_CONTACT_DROPOUT_GRACE_CONTROL_FRAMES_DEFAULT == 15
    assert RIGHT_POST_SUCCESS_HOLD_CONTROL_FRAMES == 5
    assert RIGHT_RELEASE_DWELL_CONTROL_FRAMES_DEFAULT == 3
    assert RIGHT_POST_SUCCESS_OVERTRAVEL_DEFAULT_M == pytest.approx(0.0015)
    assert RIGHT_RELEASE_STABLE_RESERVE_CONTROL_FRAMES == 20
    assert _latest_approach_timeout_frame(60) == 1720
    assert _latest_approach_timeout_frame(30) == 1750
    assert _latest_approach_timeout_frame(15) == 1765
    preview_source = (
        REPOSITORY_ROOT
        / "handson"
        / "handson"
        / "sim"
        / "isaaclab"
        / "task_preview.py"
    ).read_text(encoding="utf-8")
    assert (
        f'"{TASK_NAME}": "arena_arm_hand_c_tool_interactions_rokae_wuji1.xml"'
        in preview_source
    )
    assert f'"{TASK_NAME}": set()' in preview_source
    for label in (
        "Right hand picks up the utility knife by at least 2 cm",
        "Left fingertips touch the slider after pickup",
        "Per-episode target is randomized: retract to bottom or extend to top",
        "Slider starts away from its target at a randomized position",
    ):
        assert label in preview_source


def test_as013_exact_asset_bundle_hashes() -> None:
    assert ASSET_PATH.is_file()
    assert ASSET_METADATA_PATH.is_file()
    assert _sha256(ASSET_PATH) == EXPECTED_ASSET_HASHES["runtime/utility_knife_ahtusik.usda"]
    assert _sha256(ASSET_METADATA_PATH) == EXPECTED_ASSET_HASHES["runtime/utility_knife_ahtusik_metadata.json"]
    assert ASSET_SOURCE_ARCHIVE_SHA256 == EXPECTED_ASSET_HASHES["source/utility_knife.zip"]
    assert ASSET_METADATA_SHA256 == EXPECTED_ASSET_HASHES["runtime/utility_knife_ahtusik_metadata.json"]
    assert ASSET_USDA_SHA256 == EXPECTED_ASSET_HASHES["runtime/utility_knife_ahtusik.usda"]


def test_as013_config_is_physx_only() -> None:
    config = CONFIG_MAPPING[TASK_NAME]()
    with pytest.raises(ValueError, match="isaacsim"):
        config.get_environment(
            sim_backend="mujoco",
            launch_isaac=False,
            image_obs=False,
        )
    with pytest.raises(ValueError, match="Newton"):
        config.get_environment(
            sim_backend="isaacsim",
            physics_backend="newton",
            launch_isaac=False,
            image_obs=False,
        )


def test_as013_config_enforces_native_640x480_ego_frames() -> None:
    config = CONFIG_MAPPING[TASK_NAME]()
    env = config.get_environment(
        sim_backend="isaacsim",
        launch_isaac=False,
        image_obs=False,
        render_mode="rgb_array",
        randomize=False,
    )
    try:
        raw = env.unwrapped
        assert raw._render_width == 640
        assert raw._render_height == 480
    finally:
        env.close()
    with pytest.raises(ValueError, match="native 640x480 ego frames"):
        config.get_environment(
            sim_backend="isaacsim",
            launch_isaac=False,
            image_obs=False,
            render_resolution=(640, 640),
        )


def test_as013_reset_only_restores_properties_that_were_scaled() -> None:
    source = inspect.getsource(HandsOnAS013IsaacLabEnv._sample_reset_state)
    assert "and self._as013_blade_drive_stiffness_scaled" in source
    assert "and self._as013_blade_drive_damping_scaled" in source
    assert "and self._as013_blade_joint_friction_scaled" in source


def test_as013_interface_policy_contract() -> None:
    config = CONFIG_MAPPING[TASK_NAME]()
    env = config.get_environment(
        policy_mode=True,
        sim_backend="isaacsim",
        launch_isaac=False,
        image_obs=False,
        render_mode="rgb_array",
        randomize=False,
    )
    try:
        assert env.action_space.shape == (52,)
        raw = env.unwrapped
        assert isinstance(raw, HandsOnAS013IsaacLabEnv)
        state_spaces = raw.observation_space.spaces["state"].spaces
        assert state_spaces["knife_body_pose"].shape == (7,)
        assert state_spaces["blade_pose"].shape == (7,)
        assert state_spaces["blade_joint_state"].shape == (4,)
        assert state_spaces["knife_interaction_state"].shape == (16,)
        first_action = raw.scripted_success_action(0)
        np.testing.assert_allclose(first_action[:7], raw._right_tcp_pose)
        np.testing.assert_allclose(first_action[7:14], raw._left_tcp_pose)
        np.testing.assert_array_equal(
            first_action[14:], np.zeros(40, dtype=np.float32)
        )
        for frame in (0, 59, 60, 119, 120, 179, 180, 239, 240, 359, 360, 420):
            action = raw.scripted_success_action(frame)
            assert action.shape == (54,)
            assert action.dtype == np.float32
            assert np.isfinite(action).all()
    finally:
        env.close()


def test_as013_seeded_reset_sampler_is_bounded_and_reproducible() -> None:
    kwargs = {
        "randomize": True,
        "center_xy": np.asarray([-0.12, 0.0]),
        "hand_points": (
            np.asarray([0.30, -0.40, 1.15]),
            np.asarray([0.30, 0.40, 1.15]),
        ),
    }
    first = sample_reset_contract(np.random.RandomState(431), **kwargs)
    second = sample_reset_contract(np.random.RandomState(431), **kwargs)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert np.max(np.abs(first[0])) <= 0.03
    assert -180.0 <= first[1] <= 180.0
    assert (0.15 <= first[2] <= 0.45) if first[3] == 1 else (0.55 <= first[2] <= 0.85)
    deterministic = sample_reset_contract(
        np.random.RandomState(999),
        randomize=False,
        center_xy=np.asarray([-0.12, 0.0]),
    )
    np.testing.assert_array_equal(deterministic[0], np.zeros(2))
    assert deterministic[1:] == (0.0, 0.80, -1)


def test_as013_right_servo_weights_then_bounds_each_closed_loop_step() -> None:
    correction = _bounded_right_servo_correction(
        np.asarray([0.02, -0.04, -0.06], dtype=np.float64),
        lateral_gain=3.0,
        vertical_gain=2.5,
    )
    assert np.linalg.norm(correction) == pytest.approx(
        RIGHT_CARTESIAN_SERVO_STEP_M
    )
    np.testing.assert_allclose(
        correction / correction[0],
        np.asarray([1.0, -6.0, -7.5]),
    )

    contact_correction = _bounded_right_servo_correction(
        np.asarray([0.0, 0.0, -0.02], dtype=np.float64),
        lateral_gain=1.0,
        vertical_gain=2.5,
        max_step_m=RIGHT_SURFACE_APPROACH_SERVO_STEP_M,
    )
    assert np.linalg.norm(contact_correction) == pytest.approx(
        RIGHT_SURFACE_APPROACH_SERVO_STEP_M
    )


def test_as013_contact_filter_diagnostics_preserve_body_ownership() -> None:
    paths = ("/left/finger1", "/left/finger2", "/left/palm")
    counts = np.asarray([[1, 2, 0], [3, 0, 4]], dtype=np.uint32)
    assert _contact_filter_counts_by_path(counts, 2, paths) == {
        "/left/finger1": 4,
        "/left/finger2": 2,
        "/left/palm": 4,
    }
    assert _contact_filter_counts_by_path(counts, 1, paths) == {}
    assert _contact_filter_counts_by_path(np.zeros(0), 0, paths) == {}


def test_as013_left_stabilizer_locks_the_live_safe_contact_owner() -> None:
    assert _left_stabilizer_finger_from_contact_counts({}) == 5
    assert _left_stabilizer_finger_from_contact_counts(
        {
            "/World/right_finger1_link4": 2,
            "/World/right_finger2_link4": 9,
            "/World/right_palm_link": 4,
        }
    ) == 2
    assert _left_stabilizer_finger_from_contact_counts(
        {
            "/World/right_finger1_link4": 1,
            "/World/right_finger5_link4": 1,
        }
    ) == 5


def test_as013_left_hold_preload_is_bounded_and_not_cumulative() -> None:
    error = np.asarray([0.0, 0.0, -0.015], dtype=np.float64)
    correction = _left_hold_preload_correction(error, 0.014)
    np.testing.assert_allclose(
        correction,
        [0.0, 0.0, -LEFT_HOLD_CONTACT_PRELOAD_M],
    )
    assert np.linalg.norm(correction) == pytest.approx(
        LEFT_HOLD_CONTACT_PRELOAD_M
    )
    slide_correction = _left_hold_preload_correction(
        error,
        0.013,
        LEFT_SLIDE_HOLD_CONTACT_PRELOAD_M,
    )
    assert np.linalg.norm(slide_correction) == pytest.approx(
        LEFT_SLIDE_HOLD_CONTACT_PRELOAD_M
    )
    assert _left_hold_effective_standoff(
        0.014,
        LEFT_HOLD_CONTACT_PRELOAD_M,
    ) == pytest.approx(0.014)
    assert _left_hold_effective_standoff(
        0.014,
        LEFT_SLIDE_HOLD_CONTACT_PRELOAD_M,
    ) == pytest.approx(0.013)
    np.testing.assert_array_equal(
        _left_hold_preload_correction(
            np.asarray([0.0, 0.0, -0.013]),
            0.014,
        ),
        np.zeros(3),
    )


def test_as013_left_hand_preload_waits_for_stable_latch_and_ramps() -> None:
    assert _effective_left_hand_preload_fraction(120, None, 0.10) == 0.0
    assert _effective_left_hand_preload_fraction(119, 120, 0.10) == 0.0
    assert _effective_left_hand_preload_fraction(120, 120, 0.10) == 0.0
    assert _effective_left_hand_preload_fraction(135, 120, 0.10) == pytest.approx(
        0.05
    )
    assert _effective_left_hand_preload_fraction(150, 120, 0.10) == pytest.approx(
        0.10
    )
    assert _effective_left_hand_preload_fraction(180, 120, 0.10) == pytest.approx(
        0.10
    )


def test_as013_left_palm_pose_freezes_after_stable_contact_latch() -> None:
    assert _left_hold_pose_should_refresh(True, True, False, True)
    assert _left_hold_pose_should_refresh(True, True, True, False)
    assert not _left_hold_pose_should_refresh(True, True, True, True)
    assert not _left_hold_pose_should_refresh(False, False, False, False)

    assert _left_recovery_servo_should_apply(True, False, True, False)
    assert not _left_recovery_servo_should_apply(True, False, True, True)
    assert _left_recovery_servo_should_apply(
        True,
        False,
        True,
        True,
        fixed_world_anchor_available=True,
    )
    assert not _left_recovery_servo_should_apply(True, True, True, False)
    assert not _left_recovery_servo_should_apply(
        True,
        True,
        True,
        True,
        fixed_world_anchor_available=True,
    )


def test_as013_left_hold_keeps_the_proven_fingertip_locked() -> None:
    fingertips = [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.020, 0.0]),
    ]
    first, first_index = _select_left_hold_fingertip(
        fingertips,
        np.asarray([0.0, 0.001, 0.0]),
        None,
    )
    np.testing.assert_array_equal(first, fingertips[0])
    assert first_index == 0

    locked, locked_index = _select_left_hold_fingertip(
        fingertips,
        np.asarray([0.0, 0.019, 0.0]),
        first_index,
    )
    np.testing.assert_array_equal(locked, fingertips[0])
    assert locked_index == first_index

    missing, missing_index = _select_left_hold_fingertip(
        [],
        np.zeros(3),
        first_index,
    )
    assert missing is None
    assert missing_index is None


def test_as013_left_recovery_adds_bounded_inward_contact_preload() -> None:
    contact = np.asarray([0.0, 0.0, 0.0])
    surface = np.asarray([0.0, 0.0, -0.02])
    target = _left_recovery_preload_target(contact, surface)
    np.testing.assert_allclose(
        target,
        [0.0, 0.0, -LEFT_RECOVERY_CONTACT_PRELOAD_M],
    )
    assert np.linalg.norm(target - contact) == pytest.approx(
        LEFT_RECOVERY_CONTACT_PRELOAD_M
    )
    np.testing.assert_allclose(
        _left_recovery_preload_target(contact, contact),
        contact,
    )


def test_as013_left_fixed_anchor_recovery_is_world_fixed_and_bounded() -> None:
    assert LEFT_FIXED_ANCHOR_MAX_OFFSET_M == pytest.approx(0.003)
    target = np.asarray([0.01, -0.02, 0.03])
    current = np.asarray([0.0, 0.0, 0.0])
    correction = _left_fixed_anchor_recovery_correction(target, current)
    assert np.linalg.norm(correction) == pytest.approx(
        LEFT_FIXED_ANCHOR_RECOVERY_STEP_M
    )
    np.testing.assert_allclose(
        correction / correction[0],
        target / target[0],
    )
    np.testing.assert_array_equal(
        _left_fixed_anchor_recovery_correction(target, target),
        np.zeros(3),
    )

    offset = np.zeros(3)
    for _ in range(3):
        offset = _advance_left_fixed_anchor_offset(offset, target, current)
    assert np.linalg.norm(offset) == pytest.approx(
        3.0 * LEFT_FIXED_ANCHOR_RECOVERY_STEP_M
    )
    for _ in range(20):
        offset = _advance_left_fixed_anchor_offset(offset, target, current)
    assert np.linalg.norm(offset) == pytest.approx(
        LEFT_FIXED_ANCHOR_MAX_OFFSET_M
    )

    slide_target = _left_fixed_anchor_slide_preload_target(
        np.zeros(3),
        np.asarray([0.0, 0.0, -1.0]),
    )
    np.testing.assert_allclose(
        slide_target,
        [
            0.0,
            0.0,
            -(
                LEFT_SLIDE_HOLD_CONTACT_PRELOAD_M
                - LEFT_RECOVERY_CONTACT_PRELOAD_M
            ),
        ],
    )
    np.testing.assert_array_equal(
        _left_fixed_anchor_slide_preload_target(target, target),
        target,
    )


def test_as013_timeout_release_latches_start_and_is_not_cumulative() -> None:
    env = object.__new__(HandsOnAS013IsaacLabEnv)
    env._as013_scripted_timeout_right_pose = None
    env._as013_scripted_timeout_left_pose = None
    env._as013_scripted_timeout_right_hand = None
    env._as013_scripted_timeout_left_hand = None
    right_start = np.asarray([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])
    left_start = np.asarray([-0.1, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0])
    right_hand = np.full(20, 0.4)
    left_hand = np.full(20, 0.6)

    env._scripted_timeout_release_action(
        right_pose=right_start,
        left_pose=left_start,
        right_hand=right_hand,
        left_hand=left_hand,
        fraction=0.0,
    )
    right, left, released_right_hand, released_left_hand = (
        env._scripted_timeout_release_action(
            right_pose=right_start + np.asarray(
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            ),
            left_pose=left_start + np.asarray(
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            ),
            right_hand=np.ones(20),
            left_hand=np.ones(20),
            fraction=0.5,
        )
    )

    np.testing.assert_allclose(right[:3], right_start[:3] + [0.0, 0.0, 0.05])
    np.testing.assert_allclose(left[:3], left_start[:3] + [0.0, 0.0, 0.05])
    np.testing.assert_allclose(released_right_hand, 0.5 * right_hand)
    np.testing.assert_allclose(released_left_hand, 0.5 * left_hand)


def test_as013_right_servo_slows_before_crossing_contact_region() -> None:
    target = np.zeros(3, dtype=np.float64)
    assert _right_servo_max_step(
        "vertical_entry", np.asarray([0.06, 0.0, 0.03]), target
    ) == pytest.approx(RIGHT_CARTESIAN_SERVO_STEP_M)
    assert _right_servo_max_step(
        "planar_alignment", np.asarray([0.06, 0.0, 0.025]), target
    ) == pytest.approx(RIGHT_CARTESIAN_SERVO_STEP_M)
    assert _right_servo_max_step(
        "planar_alignment",
        np.asarray([RIGHT_SURFACE_SLOW_RADIUS_M + 0.005, 0.0, 0.025]),
        target,
    ) == pytest.approx(0.005)
    assert _right_servo_max_step(
        "planar_alignment",
        np.asarray([RIGHT_SURFACE_SLOW_RADIUS_M - 0.005, 0.0, 0.025]),
        target,
    ) == pytest.approx(RIGHT_SURFACE_APPROACH_SERVO_STEP_M)
    assert _right_servo_max_step(
        "planar_alignment",
        np.asarray([0.026, 0.0, 0.025]),
        target,
        surface_slow_radius_m=0.010,
    ) == pytest.approx(RIGHT_CARTESIAN_SERVO_STEP_M)
    assert _right_servo_max_step(
        "contact_descent", np.asarray([0.0, 0.0, 0.02]), target
    ) == pytest.approx(RIGHT_SURFACE_APPROACH_SERVO_STEP_M)
    assert _right_servo_max_step(
        "recontact_lift", np.asarray([0.0, 0.0, 0.02]), target
    ) == pytest.approx(RIGHT_RECONTACT_LIFT_SERVO_STEP_M)
    assert _right_servo_max_step(
        "recontact_wait", np.asarray([0.0, 0.0, 0.028]), target
    ) == pytest.approx(RIGHT_SURFACE_APPROACH_SERVO_STEP_M)


def test_as013_recontact_servo_uses_bounded_contact_step() -> None:
    current = np.asarray([0.0, 0.0, 0.03], dtype=np.float64)
    target = np.asarray([0.0, 0.0, 0.02], dtype=np.float64)

    assert _right_servo_state_max_step(
        recontact_active=True,
        preload_settling_active=False,
        slide_servo_active=False,
        stage="contact_descent",
        current_local=current,
        final_target_local=target,
        recontact_step_m=RIGHT_CONTACT_PRELOAD_SERVO_STEP_M,
    ) == pytest.approx(RIGHT_CONTACT_PRELOAD_SERVO_STEP_M)
    assert _right_servo_state_max_step(
        recontact_active=True,
        preload_settling_active=False,
        slide_servo_active=False,
        stage="contact_descent",
        current_local=current,
        final_target_local=target,
    ) == pytest.approx(RIGHT_SURFACE_APPROACH_SERVO_STEP_M)
    assert _right_servo_state_max_step(
        recontact_active=False,
        preload_settling_active=False,
        slide_servo_active=False,
        stage="contact_descent",
        current_local=current,
        final_target_local=target,
    ) == pytest.approx(RIGHT_SURFACE_APPROACH_SERVO_STEP_M)


def test_as013_slide_reanchor_requires_one_proven_live_contact() -> None:
    assert _right_slide_reanchor_needed(0.50, None, True, True, 1.0, 0.50)
    assert not _right_slide_reanchor_needed(0.0, None, True, True, 1.0, 0.50)
    assert not _right_slide_reanchor_needed(0.50, 0.50, True, True, 1.0, 0.70)
    assert not _right_slide_reanchor_needed(0.50, None, False, True, 1.0, 0.70)
    assert not _right_slide_reanchor_needed(0.50, None, True, False, 1.0, 0.70)
    assert not _right_slide_reanchor_needed(0.50, None, True, True, 0.0, 0.70)
    assert not _right_slide_reanchor_needed(0.50, None, True, True, 1.0, 0.49)


def test_as013_right_target_is_fixed_in_the_slider_asset_frame() -> None:
    center = 0.5 * (
        SLIDER_BUTTON_LOCAL_MIN_M + SLIDER_BUTTON_LOCAL_MAX_M
    )
    first = _fixed_right_fingertip_target_local(center)
    second = _fixed_right_fingertip_target_local(center)

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first[:2], RIGHT_FINGERTIP_TARGET_LOCAL_XY_M)
    assert first[2] == pytest.approx(
        center[2] + RIGHT_FINGERTIP_TARGET_CLEARANCE_M
    )
    assert np.all(
        first[:2]
        >= SLIDER_BUTTON_LOCAL_MIN_M[:2] + RIGHT_FINGERTIP_TARGET_INSET_M
    )
    assert np.all(
        first[:2]
        <= SLIDER_BUTTON_LOCAL_MAX_M[:2] - RIGHT_FINGERTIP_TARGET_INSET_M
    )


def test_as013_certified_right_alignment_rotates_only_with_root_yaw() -> None:
    nominal = _certified_right_alignment_world_rotation(0.0)
    yaw_degrees = 3.0
    randomized = _certified_right_alignment_world_rotation(yaw_degrees)
    yaw = np.deg2rad(yaw_degrees)
    yaw_rotation = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert np.linalg.norm(RIGHT_CERTIFIED_ALIGNMENT_QUAT_WXYZ) == pytest.approx(
        1.0,
        abs=1.0e-5,
    )
    np.testing.assert_allclose(randomized, yaw_rotation @ nominal, atol=1.0e-12)
    np.testing.assert_allclose(randomized.T @ randomized, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(randomized) == pytest.approx(1.0, abs=1.0e-12)

    reference = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(
        _yaw_rotated_world_rotation(reference, yaw_degrees),
        yaw_rotation @ reference,
        atol=1.0e-12,
    )


def test_as013_right_alignment_blend_preserves_reachable_endpoints() -> None:
    actual = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(
        _blended_right_alignment_world_rotation(actual, 2.0, 0.0),
        actual,
    )
    np.testing.assert_allclose(
        _blended_right_alignment_world_rotation(actual, 2.0, 1.0),
        _certified_right_alignment_world_rotation(2.0),
        atol=1.0e-12,
    )
    halfway = _blended_right_alignment_world_rotation(actual, 2.0, 0.5)
    np.testing.assert_allclose(halfway.T @ halfway, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(halfway) == pytest.approx(1.0, abs=1.0e-12)
    assert SCRIPTED_RIGHT_ALIGNMENT_REFERENCE_BLEND_ENV.endswith(
        "RIGHT_ALIGNMENT_REFERENCE_BLEND"
    )
    assert RIGHT_ALIGNMENT_REFERENCE_BLEND_DEFAULT == pytest.approx(0.60)


def test_as013_slide_reanchor_environment_name_is_task_scoped() -> None:
    assert SCRIPTED_RIGHT_SLIDE_REANCHOR_PROGRESS_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_REANCHOR_PROGRESS"
    )
    assert SCRIPTED_RIGHT_RECONTACT_SERVO_STEP_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_RECONTACT_SERVO_STEP_M"
    )
    assert SCRIPTED_RIGHT_SLIDE_REANCHOR_EXTRA_LEAD_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_SLIDE_REANCHOR_EXTRA_LEAD_M"
    )


def test_as013_slide_reanchor_extra_lead_is_continuous_and_bounded() -> None:
    assert _right_slide_reanchor_extra_lead(0.018, None, 100) == 0.0
    assert _right_slide_reanchor_extra_lead(0.018, 100, 100) == 0.0
    assert _right_slide_reanchor_extra_lead(0.018, 100, 130) == pytest.approx(
        0.009
    )
    assert _right_slide_reanchor_extra_lead(0.018, 100, 500) == pytest.approx(
        0.018
    )


def test_as013_slide_servo_correction_is_scaled_and_non_cumulative() -> None:
    slide_pose = np.asarray(
        [0.10, -0.20, 0.90, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    correction = np.asarray([0.002, 0.006, -0.001], dtype=np.float64)
    disabled = _right_slide_pose_with_servo_correction(
        slide_pose,
        correction,
        0.0,
    )
    half = _right_slide_pose_with_servo_correction(
        slide_pose,
        correction,
        0.5,
    )
    full = _right_slide_pose_with_servo_correction(
        slide_pose,
        correction,
        1.0,
    )
    np.testing.assert_array_equal(disabled, slide_pose)
    np.testing.assert_allclose(half[:3], slide_pose[:3] + 0.5 * correction)
    np.testing.assert_allclose(full[:3], slide_pose[:3] + correction)
    np.testing.assert_array_equal(full[3:7], slide_pose[3:7])
    np.testing.assert_array_equal(slide_pose[:3], [0.10, -0.20, 0.90])
    assert RIGHT_SLIDE_REANCHOR_EXTRA_LEAD_RAMP_CONTROL_FRAMES == 60


def test_as013_slide_servo_can_compensate_only_joint_axis_lag() -> None:
    slide_pose = np.asarray(
        [0.10, -0.20, 0.90, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    correction = np.asarray([0.003, 0.006, -0.004], dtype=np.float64)
    axis = np.asarray([0.0, 2.0, 0.0], dtype=np.float64)
    projected = _right_slide_pose_with_servo_correction(
        slide_pose,
        correction,
        0.5,
        correction_axis_world=axis,
    )
    np.testing.assert_allclose(
        projected[:3],
        slide_pose[:3] + np.asarray([0.0, 0.003, 0.0]),
    )
    np.testing.assert_array_equal(projected[3:7], slide_pose[3:7])


def test_as013_slide_servo_rejects_invalid_projection_axis() -> None:
    slide_pose = np.asarray(
        [0.10, -0.20, 0.90, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    projected = _right_slide_pose_with_servo_correction(
        slide_pose,
        np.asarray([0.003, 0.006, -0.004], dtype=np.float64),
        1.0,
        correction_axis_world=np.zeros(3, dtype=np.float64),
    )
    np.testing.assert_array_equal(projected, slide_pose)


def test_as013_cross_edge_servo_is_measured_scaled_and_bounded() -> None:
    slide_pose = np.asarray(
        [0.10, -0.20, 0.90, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    current_local = np.asarray([-0.010, -0.020, 0.033], dtype=np.float64)
    target_local = np.asarray([-0.006, -0.021, 0.033], dtype=np.float64)
    cross_edge = np.asarray([2.0, 0.0, 0.0], dtype=np.float64)
    disabled = _right_slide_pose_with_cross_edge_servo(
        slide_pose,
        current_local,
        target_local,
        cross_edge,
        0.0,
        0.003,
    )
    half = _right_slide_pose_with_cross_edge_servo(
        slide_pose,
        current_local,
        target_local,
        cross_edge,
        0.5,
        0.003,
    )
    np.testing.assert_array_equal(disabled, slide_pose)
    np.testing.assert_allclose(
        half[:3],
        slide_pose[:3] + np.asarray([0.0015, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(half[3:7], slide_pose[3:7])
    np.testing.assert_array_equal(slide_pose[:3], [0.10, -0.20, 0.90])


def test_as013_cross_edge_servo_rejects_invalid_world_axis() -> None:
    slide_pose = np.asarray(
        [0.10, -0.20, 0.90, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    corrected = _right_slide_pose_with_cross_edge_servo(
        slide_pose,
        np.asarray([-0.010, -0.020, 0.033], dtype=np.float64),
        np.asarray([-0.006, -0.021, 0.033], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        1.0,
        0.003,
    )
    np.testing.assert_array_equal(corrected, slide_pose)


def test_as013_blade_contact_friction_is_independent_and_bounded() -> None:
    assert BLADE_CONTACT_FRICTION_ENV == (
        "HANDSON_AS013_BLADE_CONTACT_FRICTION"
    )
    assert _configured_blade_contact_friction(None, 0.65) == pytest.approx(
        0.65
    )
    assert _configured_blade_contact_friction("1.20", 0.65) == pytest.approx(
        1.20
    )
    for invalid in ("not-a-number", "nan", "0.39", "3.01"):
        with pytest.raises(RuntimeError):
            _configured_blade_contact_friction(invalid, 0.65)
    assert BLADE_CONTACT_FRICTION_RANGE == (0.40, 3.00)


def test_as013_blade_material_binds_only_blade_owned_colliders() -> None:
    source = inspect.getsource(
        HandsOnAS013IsaacLabEnv._spawn_task_specific_assets
    )
    assert 'rigid_owner.GetName() == "E_link_blade_2"' in source
    assert "collider_material = blade_physics_material" in source
    assert "blade_colliders_bound < 1" in source
    assert '"blade_contact_friction": blade_contact_friction' in source


def test_as013_fallback_target_adapts_only_across_button_width() -> None:
    fixed = np.asarray([-0.00575, -0.02114, 0.025], dtype=np.float64)
    positive_edge = _adapt_fallback_right_target_cross_edge(
        np.asarray([0.030, -0.035, 0.024]),
        fixed,
    )
    negative_edge = _adapt_fallback_right_target_cross_edge(
        np.asarray([-0.030, 0.010, 0.040]),
        fixed,
    )
    assert positive_edge[0] == pytest.approx(
        SLIDER_BUTTON_LOCAL_MAX_M[0] - RIGHT_FINGERTIP_TARGET_INSET_M
    )
    assert negative_edge[0] == pytest.approx(
        SLIDER_BUTTON_LOCAL_MIN_M[0] + RIGHT_FINGERTIP_TARGET_INSET_M
    )
    np.testing.assert_array_equal(positive_edge[1:], fixed[1:])
    np.testing.assert_array_equal(negative_edge[1:], fixed[1:])
    np.testing.assert_array_equal(fixed, [-0.00575, -0.02114, 0.025])


def test_as013_late_slide_feedback_starts_after_proven_travel() -> None:
    assert _effective_slide_servo_correction_scale(
        358,
        392,
        0.060,
        0.0,
        0.10,
    ) == 0.0
    beginning = _effective_slide_servo_correction_scale(
        359,
        392,
        0.060,
        0.0,
        0.10,
    )
    assert 0.0 < beginning < 0.005
    assert _effective_slide_servo_correction_scale(
        391,
        392,
        0.060,
        0.0,
        0.10,
    ) == pytest.approx(0.10)


def test_as013_slide_reanchor_preserves_the_previous_command() -> None:
    command = np.asarray(
        [0.11, -0.07, 0.94, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    axis = np.asarray([0.2, 0.97, 0.1], dtype=np.float64)
    normal = np.asarray([0.1, -0.05, 0.99], dtype=np.float64)
    cross = np.cross(axis, normal)
    base = _reanchored_right_contact_pose_from_command(
        command,
        axis,
        normal,
        RIGHT_SLIDE_CONTROL_FRAMES - 1,
        0.012,
        blade_world_cross_edge=cross,
        cross_edge_correction_m=0.003,
        normal_preload_m=RIGHT_SLIDE_NORMAL_PRELOAD_M,
    )
    reproduced = _right_slide_servo_step_pose(
        base,
        axis,
        normal,
        RIGHT_SLIDE_CONTROL_FRAMES - 1,
        maximum_tangential_distance_m=0.012,
        blade_world_cross_edge=cross,
        cross_edge_correction_m=0.003,
        normal_preload_m=RIGHT_SLIDE_NORMAL_PRELOAD_M,
    )

    np.testing.assert_allclose(reproduced, command, atol=1.0e-12)


def test_as013_slide_pose_requires_simultaneous_current_contacts() -> None:
    assert _right_contact_pose_ready(
        ordered_contact_seen=True,
        current_left_stable_contact=True,
        current_slider_contact_force=1.0,
        servo_stage="planar_alignment",
    )
    assert not _right_contact_pose_ready(
        ordered_contact_seen=True,
        current_left_stable_contact=True,
        current_slider_contact_force=0.0,
        servo_stage="contact_descent",
    )
    assert not _right_contact_pose_ready(
        ordered_contact_seen=True,
        current_left_stable_contact=False,
        current_slider_contact_force=1.0,
        servo_stage="contact_descent",
    )
    assert _right_contact_pose_ready(
        ordered_contact_seen=True,
        current_left_stable_contact=True,
        current_slider_contact_force=1.0,
        servo_stage="contact_descent",
    )


def test_as013_slide_time_advances_only_during_simultaneous_contact() -> None:
    assert _advance_slide_control_frames(12, False) == 12
    assert _advance_slide_control_frames(12, True) == 13
    assert (
        _advance_slide_control_frames(RIGHT_SLIDE_CONTROL_FRAMES - 1, True)
        == RIGHT_SLIDE_CONTROL_FRAMES - 1
    )
    assert _advance_slide_control_frames(118, True, 120) == 119
    assert _advance_slide_control_frames(119, True, 120) == 119


def test_as013_slide_waits_for_a_settled_contact_pose() -> None:
    settled = 0
    for expected in range(1, RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES + 1):
        settled = _advance_right_contact_settle_frames(settled, True)
        assert settled == expected
    assert (
        _advance_right_contact_settle_frames(settled, True)
        == RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES
    )
    assert _advance_right_contact_settle_frames(settled, False) == 0


def test_as013_slide_anchor_preserves_the_authored_slide_after_recontact() -> None:
    common = {
        "contact_pose_ready": True,
        "contact_pose_missing": False,
        "slide_control_frames": 87,
        "recontact_lifted": True,
    }
    assert not _right_slide_contact_pose_refresh_needed(
        **common,
        contact_settle_frames=0,
    )
    assert not _right_slide_contact_pose_refresh_needed(
        **common,
        contact_settle_frames=RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES,
    )
    assert not _right_slide_contact_pose_refresh_needed(
        **{
            **common,
            "contact_pose_ready": False,
        },
        contact_settle_frames=0,
    )
    assert _right_slide_contact_pose_refresh_needed(
        **{
            **common,
            "contact_pose_missing": True,
        },
        contact_settle_frames=0,
    )
    assert _right_slide_contact_pose_refresh_needed(
        **{
            **common,
            "slide_control_frames": 0,
        },
        contact_settle_frames=0,
    )


def test_as013_contact_servo_waits_for_recovery_or_settle() -> None:
    assert _right_contact_servo_only_needed(True, False, False)
    assert _right_contact_servo_only_needed(False, True, False)
    assert not _right_contact_servo_only_needed(False, True, True)
    assert _right_contact_servo_only_needed(
        False,
        False,
        False,
        initial_contact_dropout_descent_active=True,
    )


def test_as013_seen_right_contact_reenters_closed_loop_when_lost() -> None:
    assert not _right_recontact_needed(False, 0.0)
    assert not _right_recontact_needed(True, 1.0)
    assert _right_recontact_needed(True, 0.0)
    assert _right_recontact_needed(
        True,
        1.0,
        current_left_stable_contact=False,
    )


def test_as013_touching_finger_withdraws_until_left_hold_returns() -> None:
    assert not _right_finger_withdrawal_needed(False, False)
    assert not _right_finger_withdrawal_needed(
        True,
        True,
        current_slider_contact_force=1.0,
    )
    assert _right_finger_withdrawal_needed(True, False)
    assert _right_finger_withdrawal_needed(
        True,
        True,
        recovery_ready_frames=LEFT_RECOVERY_READY_CONTROL_FRAMES - 1,
    )
    assert not _right_finger_withdrawal_needed(
        True,
        True,
        recovery_ready_frames=LEFT_RECOVERY_READY_CONTROL_FRAMES,
    )


def test_as013_slide_completion_requires_live_dual_contact() -> None:
    valid = {
        "blade_progress": 0.81,
        "slide_progress_seen": True,
        "left_stable_contact_now": True,
        "right_slider_contact_force": CONTACT_FORCE_THRESHOLD_N + 0.1,
    }
    assert _slide_complete_from_metrics(valid)
    for key, value in {
        "blade_progress": 0.79,
        "slide_progress_seen": False,
        "left_stable_contact_now": False,
        "right_slider_contact_force": CONTACT_FORCE_THRESHOLD_N,
    }.items():
        assert not _slide_complete_from_metrics({**valid, key: value}), key


def test_as013_passive_drive_latch_follows_only_motion_or_proven_success() -> None:
    assert not _passive_drive_latch_motion_ready(0.0, False)
    assert not _passive_drive_latch_motion_ready(-0.001, False)
    assert _passive_drive_latch_motion_ready(0.001, False)
    assert _passive_drive_latch_motion_ready(0.0, True)
    source = inspect.getsource(
        HandsOnAS013IsaacLabEnv._after_task_physics_substep
    )
    assert "target = float(position)" in source
    assert "self._as013_slide_progress_seen" in source
    assert "progress >= SUCCESS_PROGRESS" in source
    assert "_as013_scripted_slide_complete_frame" not in source


def test_as013_left_recovery_waits_then_reenters_gradually() -> None:
    ready = 0
    for _ in range(LEFT_RECOVERY_READY_CONTROL_FRAMES):
        ready = _advance_left_recovery_ready_frames(ready, True, 0.0)
    assert ready == LEFT_RECOVERY_READY_CONTROL_FRAMES
    assert _right_finger_reentry_fraction(True, True, 0.0, ready) == 0.0
    for _ in range(RIGHT_FINGER_REENTRY_CONTROL_FRAMES):
        ready = _advance_left_recovery_ready_frames(ready, True, 0.0)
    assert _right_finger_reentry_fraction(True, True, 0.0, ready) == 1.0
    assert _advance_left_recovery_ready_frames(ready, False, 0.0) == 0
    assert _advance_left_recovery_ready_frames(ready, True, 1.0) == 0
    assert _right_finger_reentry_fraction(False, False, 0.0, 0) == 1.0


def test_as013_left_hold_retracts_only_the_unsafe_finger() -> None:
    contact = np.linspace(-0.3, 0.7, 20, dtype=np.float64)
    unchanged = _left_hold_hand_with_retracted_unsafe_finger(contact, 0.0)
    np.testing.assert_allclose(unchanged, contact)

    retracted = _left_hold_hand_with_retracted_unsafe_finger(contact, 1.0)
    unsafe_start = 4 * (LEFT_UNSAFE_BLADE_FINGER_NUMBER - 1)
    unsafe_slice = slice(unsafe_start, unsafe_start + 4)
    np.testing.assert_allclose(retracted[unsafe_slice], _POWER_GRIP[unsafe_slice])
    np.testing.assert_allclose(retracted[:unsafe_start], contact[:unsafe_start])
    np.testing.assert_allclose(
        retracted[unsafe_start + 4 :], contact[unsafe_start + 4 :]
    )

    distal_only = _left_hold_hand_with_retracted_unsafe_finger(
        contact,
        1.0,
        distal_only=True,
    )
    np.testing.assert_allclose(
        distal_only[unsafe_start : unsafe_start + 2],
        contact[unsafe_start : unsafe_start + 2],
    )
    np.testing.assert_allclose(
        distal_only[unsafe_start + 2 : unsafe_start + 4],
        contact[unsafe_start + 2 : unsafe_start + 4]
        + _FINGERTIP_EXTENSION[
            LEFT_UNSAFE_BLADE_FINGER_NUMBER - 1,
            2:4,
        ],
    )
    np.testing.assert_allclose(
        distal_only[:unsafe_start],
        contact[:unsafe_start],
    )
    np.testing.assert_allclose(
        distal_only[unsafe_start + 4 :], contact[unsafe_start + 4 :]
    )


def test_as013_left_hold_reinforces_only_the_safe_finger() -> None:
    contact = np.linspace(-0.3, 0.7, 20, dtype=np.float64)
    unchanged = _left_hold_hand_with_reinforced_safe_finger(contact, 0.0)
    np.testing.assert_allclose(unchanged, contact)

    reinforced = _left_hold_hand_with_reinforced_safe_finger(contact, 1.0)
    safe_start = 4 * (LEFT_SAFE_STABILIZER_FINGER_NUMBER - 1)
    safe_slice = slice(safe_start, safe_start + 4)
    np.testing.assert_allclose(reinforced[safe_slice], _POWER_GRIP[safe_slice])
    np.testing.assert_allclose(reinforced[:safe_start], contact[:safe_start])
    np.testing.assert_allclose(reinforced[safe_start + 4 :], contact[safe_start + 4 :])

    finger1 = _left_hold_hand_with_reinforced_safe_finger(contact, 1.0, 1)
    np.testing.assert_allclose(finger1[:4], _POWER_GRIP[:4])
    np.testing.assert_allclose(finger1[4:], contact[4:])


def test_as013_left_recovery_supports_a_bounded_faster_reentry() -> None:
    reentry_frames = 15
    ready = 0
    for _ in range(LEFT_RECOVERY_READY_CONTROL_FRAMES + reentry_frames):
        ready = _advance_left_recovery_ready_frames(
            ready,
            True,
            0.0,
            reentry_frames,
        )
    assert ready == LEFT_RECOVERY_READY_CONTROL_FRAMES + reentry_frames
    assert _right_finger_reentry_fraction(
        True,
        True,
        0.0,
        ready,
        reentry_frames,
    ) == 1.0


def test_as013_contact_phase_requires_current_left_stability() -> None:
    assert _right_contact_phase_flags(True, False, 2.0, 0) == (
        False,
        False,
    )
    assert _right_contact_phase_flags(True, True, 2.0, 0) == (
        True,
        False,
    )
    assert _right_contact_phase_flags(
        True,
        True,
        2.0,
        RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES,
    ) == (False, True)
    assert _right_contact_phase_flags(True, True, 0.0, 0) == (
        False,
        False,
    )


def test_as013_confirmed_recontact_moves_in_plane_toward_button() -> None:
    target = np.asarray([0.008, -0.037, 0.0253], dtype=np.float64)
    slider = np.asarray([-0.001, 0.0, 0.005], dtype=np.float64)
    inactive = _preload_recovered_right_contact_in_plane(
        target,
        slider,
        False,
    )
    active = _preload_recovered_right_contact_in_plane(
        target,
        slider,
        True,
    )
    np.testing.assert_allclose(inactive, target)
    assert np.linalg.norm(active[:2] - target[:2]) == pytest.approx(
        RIGHT_RECOVERED_CONTACT_IN_PLANE_PRELOAD_M
    )
    assert np.linalg.norm(active[:2] - slider[:2]) < np.linalg.norm(
        target[:2] - slider[:2]
    )
    assert active[2] == pytest.approx(target[2])


def test_as013_recovered_late_normal_preload_ramps_near_success() -> None:
    assert _effective_recovered_late_normal_preload_m(
        blade_progress=0.74,
        late_preload_m=0.0002,
        start_progress=0.75,
    ) == pytest.approx(0.0)
    assert _effective_recovered_late_normal_preload_m(
        blade_progress=0.775,
        late_preload_m=0.0002,
        start_progress=0.75,
    ) == pytest.approx(0.0001)
    assert _effective_recovered_late_normal_preload_m(
        blade_progress=0.80,
        late_preload_m=0.0002,
        start_progress=0.75,
    ) == pytest.approx(0.0002)


def test_as013_recovered_late_abduction_ramps_near_success() -> None:
    assert _effective_recovered_late_finger_abduction_delta(
        blade_progress=0.74,
        late_abduction_delta_rad=0.01,
        start_progress=0.75,
    ) == pytest.approx(0.0)
    assert _effective_recovered_late_finger_abduction_delta(
        blade_progress=0.775,
        late_abduction_delta_rad=0.01,
        start_progress=0.75,
    ) == pytest.approx(0.005)
    assert _effective_recovered_late_finger_abduction_delta(
        blade_progress=0.80,
        late_abduction_delta_rad=-0.01,
        start_progress=0.75,
    ) == pytest.approx(-0.01)


def test_as013_confirmed_recontact_adds_normal_hold() -> None:
    target = np.asarray([0.008, -0.037, 0.0259], dtype=np.float64)
    held = _preload_recovered_right_contact_normal(target)
    np.testing.assert_allclose(held[:2], target[:2])
    assert held[2] == pytest.approx(
        target[2] - RIGHT_RECOVERED_CONTACT_NORMAL_PRELOAD_M
    )


def test_as013_confirmed_recontact_nudges_in_plane_from_edge() -> None:
    target = np.asarray([0.0084, -0.0382, 0.0259], dtype=np.float64)
    slider = np.asarray([-0.001, 0.0, 0.005], dtype=np.float64)
    held = _preload_recovered_capture_in_plane(target, slider)
    assert np.linalg.norm(held[:2] - target[:2]) == pytest.approx(
        RIGHT_RECOVERED_CAPTURE_IN_PLANE_PRELOAD_M
    )
    assert np.linalg.norm(held[:2] - slider[:2]) < np.linalg.norm(
        target[:2] - slider[:2]
    )
    assert held[2] == pytest.approx(target[2])


def test_as013_confirmed_recontact_moves_across_button_edge() -> None:
    target = np.asarray([0.0103, -0.0262, 0.0263], dtype=np.float64)
    slider = np.asarray([-0.001, 0.0, 0.005], dtype=np.float64)
    held = _preload_recovered_capture_cross_edge(target, slider)
    np.testing.assert_allclose(held, target)


def test_as013_recovered_contact_adds_relative_finger_extension() -> None:
    contact = np.linspace(-0.2, 0.2, 20)
    recovered = _recovered_fingertip_hand_from_contact(contact, 4)
    np.testing.assert_array_equal(recovered[:16], contact[:16])
    np.testing.assert_allclose(
        recovered[16:20] - contact[16:20],
        RIGHT_RECOVERED_FINGER_EXTENSION_DELTA
        * np.asarray([-0.08, 0.0, -0.20, -0.20]),
    )


def test_as013_recovered_contact_extension_can_be_disabled() -> None:
    contact = np.linspace(-0.2, 0.2, 20)
    recovered = _recovered_fingertip_hand_from_contact(
        contact,
        finger_index=4,
        extension_delta=0.0,
    )
    np.testing.assert_array_equal(recovered, contact)


def test_as013_recovered_contact_abduction_changes_only_selected_finger() -> None:
    contact = np.linspace(-0.2, 0.2, 20)
    recovered = _recovered_fingertip_hand_from_contact(
        contact,
        finger_index=4,
        extension_delta=0.0,
        abduction_delta_rad=0.01,
    )
    expected = contact.copy()
    expected[17] += 0.01
    np.testing.assert_allclose(recovered, expected)


def test_as013_recovered_late_extension_can_be_distal_only() -> None:
    contact = np.linspace(-0.2, 0.2, 20)
    distal_only = _recovered_fingertip_hand_from_contact(
        contact,
        finger_index=4,
        extension_delta=0.0,
        late_extension_delta=0.02,
        late_proximal_scale=0.0,
    )
    expected_delta = np.zeros(20, dtype=np.float64)
    expected_delta[18:20] = -0.004
    np.testing.assert_allclose(distal_only - contact, expected_delta)

    middle_only = _recovered_fingertip_hand_from_contact(
        contact,
        finger_index=4,
        extension_delta=0.0,
        late_extension_delta=0.02,
        late_proximal_scale=0.0,
        late_distal_tip_scale=0.0,
    )
    expected_delta[19] = 0.0
    np.testing.assert_allclose(middle_only - contact, expected_delta)

    tip_heavy = _recovered_fingertip_hand_from_contact(
        contact,
        finger_index=4,
        extension_delta=0.0,
        late_extension_delta=0.02,
        late_proximal_scale=0.0,
        late_distal_tip_scale=2.0,
    )
    expected_delta[19] = -0.008
    np.testing.assert_allclose(tip_heavy - contact, expected_delta)

    tip_only = _recovered_fingertip_hand_from_contact(
        contact,
        finger_index=4,
        extension_delta=0.0,
        late_extension_delta=0.02,
        late_proximal_scale=0.0,
        late_middle_scale=0.0,
    )
    expected_delta[18] = 0.0
    expected_delta[19] = -0.004
    np.testing.assert_allclose(tip_only - contact, expected_delta)

    middle_heavy = _recovered_fingertip_hand_from_contact(
        contact,
        finger_index=4,
        extension_delta=0.0,
        late_extension_delta=0.02,
        late_proximal_scale=0.0,
        late_middle_scale=4.0,
        late_distal_tip_scale=0.0,
    )
    expected_delta[18] = -0.016
    expected_delta[19] = 0.0
    np.testing.assert_allclose(middle_heavy - contact, expected_delta)

    default_shape = _recovered_fingertip_hand_from_contact(
        contact,
        finger_index=4,
        extension_delta=0.16,
        late_extension_delta=0.02,
    )
    combined = _recovered_fingertip_hand_from_contact(
        contact,
        finger_index=4,
        extension_delta=0.18,
    )
    np.testing.assert_allclose(default_shape, combined)


def test_as013_recovered_late_extension_ramps_near_success() -> None:
    assert _effective_recovered_finger_extension_delta(
        blade_progress=0.69,
        base_extension_delta=0.16,
        late_extension_delta=0.04,
    ) == pytest.approx(0.16)
    assert _effective_recovered_finger_extension_delta(
        blade_progress=0.75,
        base_extension_delta=0.16,
        late_extension_delta=0.04,
    ) == pytest.approx(0.18)
    assert _effective_recovered_finger_extension_delta(
        blade_progress=0.80,
        base_extension_delta=0.16,
        late_extension_delta=0.04,
    ) == pytest.approx(0.20)
    assert _effective_recovered_finger_extension_delta(
        blade_progress=0.75,
        base_extension_delta=0.16,
        late_extension_delta=0.04,
        start_progress=0.75,
    ) == pytest.approx(0.16)
    assert _effective_recovered_finger_extension_delta(
        blade_progress=0.775,
        base_extension_delta=0.16,
        late_extension_delta=0.04,
        start_progress=0.75,
    ) == pytest.approx(0.18)


def test_as013_slide_extension_changes_only_selected_finger() -> None:
    contact = np.linspace(-0.2, 0.2, 20)
    extended = _slide_fingertip_hand_from_contact(
        contact,
        finger_index=4,
        fraction=0.5,
        extension_delta=0.10,
    )
    np.testing.assert_array_equal(extended[:16], contact[:16])
    np.testing.assert_allclose(
        extended[16:20] - contact[16:20],
        0.05 * np.asarray([-0.08, 0.0, -0.20, -0.20]),
    )
    abducted = _slide_fingertip_hand_from_contact(
        contact,
        finger_index=4,
        fraction=0.5,
        extension_delta=0.0,
        abduction_delta_rad=0.04,
    )
    expected_delta = np.zeros(20, dtype=np.float64)
    expected_delta[17] = 0.02
    np.testing.assert_allclose(abducted - contact, expected_delta)


def test_as013_touch_posture_retracts_noncontrol_fingertips() -> None:
    assert RIGHT_NONCONTROL_GRIP_FRACTION == pytest.approx(1.0)
    hand = _right_touch_hand_posture(5, 0.67, -0.20)
    np.testing.assert_allclose(
        hand[:16],
        [
            0.60, 0.10, 0.84, 0.70,
            0.68, 0.00, 0.90, 0.78,
            0.70, 0.00, 0.92, 0.80,
            0.68, 0.00, 0.90, 0.78,
        ],
    )
    np.testing.assert_allclose(
        hand[16:20],
        0.67 * np.asarray([-0.08, 0.0, -0.20, -0.20])
        + np.asarray([0.0, -0.20, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        _right_touch_hand_posture(None, 0.67, -0.20),
        np.zeros(20),
    )


def test_as013_extended_palm_travel_preserves_the_proven_finger_ramp() -> None:
    baseline_first = _slide_finger_extension_fraction(
        1,
        RIGHT_SLIDE_CONTROL_FRAMES,
        RIGHT_SLIDE_PALM_TRAVEL_M,
    )
    extended_first = _slide_finger_extension_fraction(1, 425, 0.065)
    assert extended_first == pytest.approx(baseline_first, rel=0.002)
    assert _slide_finger_extension_fraction(0, 425, 0.065) == 0.0
    assert _slide_finger_extension_fraction(359, 425, 0.065) == 1.0
    assert _slide_finger_extension_fraction(424, 425, 0.065) == 1.0


def test_as013_late_travel_fraction_starts_after_proven_palm_travel() -> None:
    assert _slide_late_travel_fraction(359, 360, 0.055) == 0.0
    assert _slide_late_travel_fraction(358, 392, 0.060) == 0.0
    assert 0.0 < _slide_late_travel_fraction(359, 392, 0.060) < 0.05
    assert _slide_late_travel_fraction(391, 392, 0.060) == 1.0


def test_as013_late_finger_extension_starts_after_proven_travel() -> None:
    base_delta = 0.06
    late_delta = 0.01
    assert _effective_slide_finger_extension_delta(
        358,
        392,
        0.060,
        base_delta,
        late_delta,
    ) == pytest.approx(base_delta)
    beginning = _effective_slide_finger_extension_delta(
        359,
        392,
        0.060,
        base_delta,
        late_delta,
    )
    assert base_delta < beginning < base_delta + 0.0005
    assert _effective_slide_finger_extension_delta(
        391,
        392,
        0.060,
        base_delta,
        late_delta,
    ) == pytest.approx(base_delta + late_delta)


def test_as013_recovered_hand_blends_only_selected_finger() -> None:
    blend = 0.0
    for expected in (0.1, 0.2, 0.3):
        blend = _advance_recovered_right_hand_blend(blend, True, True)
        assert blend == pytest.approx(expected)
    assert _advance_recovered_right_hand_blend(blend, True, False) == 0.0
    assert _advance_recovered_right_hand_blend(blend, False, True) == 0.0

    contact = np.linspace(-0.2, 0.2, 20)
    recovered = np.zeros(20, dtype=np.float64)
    recovered[16:20] = np.asarray([-0.04, -0.30, -0.12, -0.12])
    result = _blend_recovered_fingertip_hand(
        contact,
        recovered,
        finger_index=4,
        fraction=0.25,
    )
    np.testing.assert_allclose(result[:16], contact[:16])
    np.testing.assert_allclose(
        result[16:20],
        0.75 * contact[16:20] + 0.25 * recovered[16:20],
    )


def test_as013_late_continuous_hand_blend_is_bounded_and_contact_gated() -> None:
    assert RIGHT_LATE_CONTINUOUS_HAND_BLEND_CONTROL_FRAMES == 30
    assert SCRIPTED_RIGHT_LATE_CONTINUOUS_HAND_BLEND_ENV == (
        "HANDSON_AS013_SCRIPTED_RIGHT_LATE_CONTINUOUS_HAND_BLEND"
    )
    advance = _advance_late_continuous_right_hand_blend
    assert advance(0.0, 0.0, 0.76, 0.75, True) == 0.0
    assert advance(0.0, 1.0, 0.74, 0.75, True) == 0.0
    blend = advance(0.0, 0.6, 0.75, 0.75, True)
    assert blend == pytest.approx(0.02)
    assert advance(blend, 0.6, 0.76, 0.75, False) == pytest.approx(blend)
    for _ in range(100):
        blend = advance(blend, 0.6, 0.76, 0.75, True)
    assert blend == pytest.approx(0.6)
    assert advance(blend, 1.0, 0.70, 0.75, True) == pytest.approx(blend)


def test_as013_recovered_contact_has_bounded_servo_grace() -> None:
    assert _advance_right_recontact_grace_frames(0, False, 1.0) == 0
    grace = _advance_right_recontact_grace_frames(0, True, 1.0)
    assert grace == RIGHT_RECONTACT_CONTACT_GRACE_CONTROL_FRAMES
    assert _advance_right_recontact_grace_frames(0, True, 1.0, 5) == 5
    for expected in range(grace - 1, -1, -1):
        grace = _advance_right_recontact_grace_frames(grace, True, 0.0)
        assert grace == expected
    assert _advance_right_recontact_grace_frames(grace, True, 0.0) == 0


def test_as013_initial_contact_dropout_grace_is_bounded_and_resets() -> None:
    advance = _advance_initial_contact_dropout_grace_frames
    assert advance(0, False, True, 0.0, 3) == 0
    assert advance(0, True, False, 0.0, 3) == 0
    assert advance(0, True, True, 1.0, 3) == 0
    assert advance(0, True, True, 0.0, 0) == 0
    frames = 0
    for expected in (1, 2, 3, 4, 4):
        frames = advance(frames, True, True, 0.0, 3)
        assert frames == expected
    assert advance(frames, True, True, 1.0, 3) == 0


def test_as013_last_proven_contact_tracks_only_pre_lift_physical_contact() -> None:
    update = _updated_last_proven_right_contact_local
    first = np.asarray([-0.0080, -0.0207, 0.0326])
    assert update(None, first, 0.0, False) is None
    tracked = update(None, first, 0.02, False)
    assert tracked is not None
    np.testing.assert_allclose(tracked, first)
    tracked[0] = 9.0
    np.testing.assert_allclose(first, [-0.0080, -0.0207, 0.0326])
    previous = np.asarray([-0.0082, -0.0205, 0.0326])
    np.testing.assert_allclose(update(previous, first, 0.0, False), previous)
    np.testing.assert_allclose(update(previous, first, 0.02, True), previous)


def test_as013_first_lost_contact_search_uses_latest_physical_anchor() -> None:
    initial = np.asarray([-0.0081, -0.0203, 0.0326])
    latest = np.asarray([-0.0073, -0.0158, 0.0340])
    slider = np.asarray([-0.00575, -0.02114, 0.0040])

    inactive = _latest_right_recontact_target_local(
        initial,
        latest,
        slider,
        False,
    )
    np.testing.assert_allclose(inactive, initial)
    inactive[0] = 9.0
    assert initial[0] == pytest.approx(-0.0081)

    missing = _latest_right_recontact_target_local(
        initial,
        None,
        slider,
        True,
    )
    np.testing.assert_allclose(missing, initial)

    active = _latest_right_recontact_target_local(
        initial,
        latest,
        slider,
        True,
    )
    np.testing.assert_allclose(active, latest)
    active[0] = 9.0
    assert latest[0] == pytest.approx(-0.0073)
    assert active[2] == pytest.approx(latest[2])


def test_as013_recovered_target_blends_after_contact_settles() -> None:
    computed = np.asarray([0.008, -0.040, 0.0255])
    recovered = np.asarray([0.008, -0.037, 0.0253])
    np.testing.assert_allclose(
        _settling_right_recontact_target_local(
            computed,
            recovered,
            True,
            0.0,
        ),
        recovered,
    )
    blend = 0.0
    assert (
        _advance_recovered_target_blend(
            blend,
            True,
            True,
            RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES - 1,
        )
        == 0.0
    )
    blend = _advance_recovered_target_blend(
        blend,
        True,
        True,
        RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES,
    )
    assert blend == pytest.approx(
        1.0 / RIGHT_RECOVERED_TARGET_BLEND_CONTROL_FRAMES
    )
    np.testing.assert_allclose(
        _settling_right_recontact_target_local(
            computed,
            recovered,
            True,
            blend,
        ),
        (1.0 - blend) * recovered + blend * computed,
    )
    for _ in range(30):
        blend = _advance_recovered_target_blend(
            blend,
            True,
            True,
            RIGHT_SLIDE_CONTACT_SETTLE_CONTROL_FRAMES,
        )
    assert blend == pytest.approx(RIGHT_RECOVERED_TARGET_MAX_BLEND)
    np.testing.assert_allclose(
        _settling_right_recontact_target_local(
            computed,
            recovered,
            True,
            blend,
        ),
        (1.0 - blend) * recovered + blend * computed,
    )
    np.testing.assert_allclose(
        _settling_right_recontact_target_local(computed, recovered, False, 0),
        computed,
    )
    assert _advance_recovered_target_blend(blend, True, False, 5) == 0.0


def test_as013_recontact_captures_complete_proven_contact_center() -> None:
    contact_center = np.asarray([0.010, -0.020, 0.026])
    captured = _capture_right_recontact_target_local(contact_center)
    np.testing.assert_array_equal(captured, contact_center)
    captured[2] = 0.005
    assert contact_center[2] == pytest.approx(0.026)


def test_as013_recontact_adds_bounded_in_plane_contact_preload() -> None:
    contact_center = np.asarray([0.010, -0.020, 0.026])
    slider_target = np.asarray([-0.006, -0.021, 0.005])
    target = _preload_right_recontact_target_local(
        contact_center,
        slider_target,
    )
    assert np.linalg.norm(target[:2] - contact_center[:2]) == pytest.approx(
        RIGHT_RECONTACT_CONTACT_PRELOAD_M
    )
    assert target[2] == pytest.approx(contact_center[2])
    assert np.linalg.norm(target[:2] - slider_target[:2]) < np.linalg.norm(
        contact_center[:2] - slider_target[:2]
    )


def test_as013_left_counter_slide_is_slow_bounded_physical_motion() -> None:
    anchor = np.asarray([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])
    actual = anchor.copy()
    hold = anchor.copy()
    hold[:3] += np.asarray([0.001, 0.002, -0.001])
    unchanged = _left_counter_slide_pose(
        anchor,
        hold,
        actual,
        np.asarray([0.0, 2.0, 0.0]),
        0,
        0.03,
    )
    np.testing.assert_array_equal(unchanged, hold)
    first_step = _left_counter_slide_pose(
        anchor,
        hold,
        actual,
        np.asarray([0.0, 2.0, 0.0]),
        1,
        0.03,
    )
    step_m = 0.03 / float(LEFT_COUNTER_SLIDE_CONTROL_FRAMES - 1)
    assert first_step[1] - hold[1] == pytest.approx(-step_m)
    np.testing.assert_array_equal(first_step[[0, 2]], hold[[0, 2]])
    np.testing.assert_array_equal(first_step[3:7], hold[3:7])
    complete = _left_counter_slide_pose(
        anchor,
        hold,
        actual,
        np.asarray([0.0, 2.0, 0.0]),
        LEFT_COUNTER_SLIDE_CONTROL_FRAMES - 1,
        0.03,
    )
    assert complete[1] - hold[1] == pytest.approx(-0.03)
    stopped = _left_counter_slide_pose(
        anchor,
        hold,
        actual,
        np.asarray([0.0, 2.0, 0.0]),
        LEFT_COUNTER_SLIDE_CONTROL_FRAMES,
        0.03,
    )
    np.testing.assert_array_equal(stopped, complete)
    assert (
        step_m * 30.0
        < 0.025
    )


def test_as013_slide_step_is_bounded_and_preserves_orientation() -> None:
    pose = np.asarray([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])
    stepped = _right_slide_servo_step_pose(
        pose,
        np.asarray([2.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 3.0]),
        1,
    )
    assert stepped[0] - pose[0] == pytest.approx(
        RIGHT_SLIDE_PALM_TRAVEL_M / (RIGHT_SLIDE_CONTROL_FRAMES - 1)
    )
    faster = _right_slide_servo_step_pose(
        pose,
        np.asarray([2.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 3.0]),
        1,
        total_control_frames=120,
    )
    assert faster[0] - pose[0] == pytest.approx(
        RIGHT_SLIDE_PALM_TRAVEL_M / 119.0
    )
    assert stepped[2] - pose[2] == pytest.approx(
        -RIGHT_SLIDE_NORMAL_PRELOAD_M
    )
    np.testing.assert_allclose(stepped[1:2], pose[1:2])
    np.testing.assert_allclose(stepped[3:], pose[3:])
    np.testing.assert_allclose(
        _right_slide_servo_step_pose(
            pose,
            np.zeros(3),
            np.asarray([0.0, 0.0, 1.0]),
            1,
        ),
        pose,
    )


def test_as013_slide_pose_accumulates_from_proven_contact_anchor() -> None:
    pose = np.asarray([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])
    quarter = _right_slide_servo_step_pose(
        pose,
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        (RIGHT_SLIDE_CONTROL_FRAMES - 1) // 4,
    )
    complete = _right_slide_servo_step_pose(
        pose,
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        RIGHT_SLIDE_CONTROL_FRAMES - 1,
    )
    assert quarter[0] > pose[0]
    assert complete[0] - pose[0] == pytest.approx(RIGHT_SLIDE_PALM_TRAVEL_M)
    assert complete[2] - pose[2] == pytest.approx(
        -RIGHT_SLIDE_NORMAL_PRELOAD_M
    )
    np.testing.assert_allclose(complete[1:2], pose[1:2])
    np.testing.assert_allclose(complete[3:], pose[3:])


def test_as013_slide_pose_supports_longer_travel_at_the_same_step_size() -> None:
    pose = np.asarray([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])
    extended_travel_m = 0.065
    extended_control_frames = 425
    first = _right_slide_servo_step_pose(
        pose,
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        1,
        total_control_frames=extended_control_frames,
        palm_travel_m=extended_travel_m,
    )
    complete = _right_slide_servo_step_pose(
        pose,
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        extended_control_frames - 1,
        total_control_frames=extended_control_frames,
        palm_travel_m=extended_travel_m,
    )
    assert first[0] - pose[0] == pytest.approx(
        extended_travel_m / (extended_control_frames - 1)
    )
    assert first[0] - pose[0] == pytest.approx(
        RIGHT_SLIDE_PALM_TRAVEL_M / (RIGHT_SLIDE_CONTROL_FRAMES - 1),
        rel=0.002,
    )
    assert complete[0] - pose[0] == pytest.approx(extended_travel_m)


def test_as013_slide_pose_never_outpaces_observed_blade_motion() -> None:
    pose = np.asarray([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])
    maximum_distance = 0.012
    capped = _right_slide_servo_step_pose(
        pose,
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        RIGHT_SLIDE_CONTROL_FRAMES - 1,
        maximum_tangential_distance_m=maximum_distance,
    )
    assert capped[0] - pose[0] == pytest.approx(maximum_distance)
    assert maximum_distance < RIGHT_SLIDE_PALM_TRAVEL_M


def test_as013_fallback_slide_can_use_the_authored_full_palm_travel() -> None:
    nominal = _maximum_right_slide_tangential_distance(
        current_progress=0.20,
        contact_progress=0.20,
        joint_range_m=0.08,
        maximum_lead_m=0.016,
        palm_travel_m=0.060,
    )
    following = _maximum_right_slide_tangential_distance(
        current_progress=0.50,
        contact_progress=0.20,
        joint_range_m=0.08,
        maximum_lead_m=0.016,
        palm_travel_m=0.060,
    )
    fallback = _maximum_right_slide_tangential_distance(
        current_progress=0.20,
        contact_progress=0.20,
        joint_range_m=0.08,
        maximum_lead_m=0.016,
        palm_travel_m=0.060,
        full_travel_fallback=True,
    )
    assert nominal == pytest.approx(0.016)
    assert following == pytest.approx(0.040)
    assert fallback == pytest.approx(0.060)


def test_as013_slide_pose_applies_requested_normal_preload() -> None:
    pose = np.asarray([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])
    requested_preload = 0.001
    preloaded = _right_slide_servo_step_pose(
        pose,
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        0,
        normal_preload_m=requested_preload,
    )
    assert preloaded[2] - pose[2] == pytest.approx(-requested_preload)


def test_as013_slide_pose_ramps_late_normal_preload_after_55mm() -> None:
    pose = np.asarray([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])
    requested_late_preload = 0.0003
    before_late_travel = _right_slide_servo_step_pose(
        pose,
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        358,
        total_control_frames=392,
        palm_travel_m=0.060,
        late_normal_preload_m=requested_late_preload,
    )
    complete = _right_slide_servo_step_pose(
        pose,
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        391,
        total_control_frames=392,
        palm_travel_m=0.060,
        late_normal_preload_m=requested_late_preload,
    )
    assert before_late_travel[2] - pose[2] == pytest.approx(
        -RIGHT_SLIDE_NORMAL_PRELOAD_M
    )
    assert complete[2] - pose[2] == pytest.approx(
        -(RIGHT_SLIDE_NORMAL_PRELOAD_M + requested_late_preload)
    )


def test_as013_slide_pose_gradually_applies_requested_yaw_relief() -> None:
    pose = np.asarray([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])
    relieved = _right_slide_servo_step_pose(
        pose,
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        RIGHT_SLIDE_CONTROL_FRAMES - 1,
        slide_yaw_relief_deg=6.0,
    )
    half_angle = np.deg2rad(3.0)
    np.testing.assert_allclose(
        relieved[3:7],
        [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)],
        atol=1.0e-8,
    )


def test_as013_slide_pose_gradually_applies_requested_roll_relief() -> None:
    pose = np.asarray([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])
    relieved = _right_slide_servo_step_pose(
        pose,
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        RIGHT_SLIDE_CONTROL_FRAMES - 1,
        blade_world_cross_edge=np.asarray([1.0, 0.0, 0.0]),
        slide_roll_relief_deg=6.0,
    )
    half_angle = np.deg2rad(3.0)
    np.testing.assert_allclose(
        relieved[3:7],
        [np.cos(half_angle), np.sin(half_angle), 0.0, 0.0],
        atol=1.0e-8,
    )


def test_as013_slide_pose_gradually_applies_requested_pitch_relief() -> None:
    pose = np.asarray([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])
    relieved = _right_slide_servo_step_pose(
        pose,
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        RIGHT_SLIDE_CONTROL_FRAMES - 1,
        slide_pitch_relief_deg=6.0,
    )
    half_angle = np.deg2rad(3.0)
    np.testing.assert_allclose(
        relieved[3:7],
        [np.cos(half_angle), 0.0, np.sin(half_angle), 0.0],
        atol=1.0e-8,
    )


def test_as013_slide_pose_blends_inside_authored_button_cross_edge() -> None:
    pose = np.asarray([0.1, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])
    correction = -RIGHT_SLIDE_CROSS_EDGE_MAX_CORRECTION_M
    halfway = _right_slide_servo_step_pose(
        pose,
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        RIGHT_SLIDE_CROSS_EDGE_BLEND_CONTROL_FRAMES // 2,
        blade_world_cross_edge=np.asarray([1.0, 0.0, 0.0]),
        cross_edge_correction_m=correction,
    )
    disabled = _right_slide_servo_step_pose(
        pose,
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        RIGHT_SLIDE_CROSS_EDGE_BLEND_CONTROL_FRAMES,
        blade_world_cross_edge=np.asarray([1.0, 0.0, 0.0]),
        cross_edge_correction_m=correction,
        cross_edge_max_correction_m=0.0,
    )
    complete = _right_slide_servo_step_pose(
        pose,
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        RIGHT_SLIDE_CROSS_EDGE_BLEND_CONTROL_FRAMES,
        blade_world_cross_edge=np.asarray([1.0, 0.0, 0.0]),
        cross_edge_correction_m=correction,
    )
    assert halfway[0] - pose[0] == pytest.approx(0.5 * correction)
    assert complete[0] - pose[0] == pytest.approx(correction)
    assert disabled[0] - pose[0] == pytest.approx(0.0)


def test_as013_edge_contact_lifts_then_retraces_vertical_recontact() -> None:
    recontact_target = np.asarray([0.020, -0.010, 0.008])
    lift, stage, ready = _recontact_right_fingertip_target_local(
        np.asarray([0.020, -0.010, 0.025]),
        recontact_target,
        False,
    )
    assert stage == "recontact_lift"
    assert not ready
    np.testing.assert_allclose(
        lift,
        [0.020, -0.010, RIGHT_RECONTACT_HEIGHT_M],
    )

    descent, stage, ready = _recontact_right_fingertip_target_local(
        np.asarray(
            [
                0.020,
                -0.010,
                RIGHT_RECONTACT_CLEAR_HEIGHT_M,
            ]
        ),
        recontact_target,
        False,
    )
    assert stage == "contact_descent"
    assert ready
    np.testing.assert_allclose(descent, recontact_target)

    vertical_descent, stage, ready = _recontact_right_fingertip_target_local(
        np.asarray(
            [
                recontact_target[0] + 0.010,
                recontact_target[1],
                RIGHT_RECONTACT_CLEAR_HEIGHT_M,
            ]
        ),
        recontact_target,
        True,
    )
    assert stage == "contact_descent"
    assert ready
    np.testing.assert_allclose(
        vertical_descent,
        recontact_target,
    )

    blocked, stage, ready = _recontact_right_fingertip_target_local(
        np.asarray([0.020, -0.010, RIGHT_RECONTACT_CLEAR_HEIGHT_M]),
        recontact_target,
        False,
        current_slider_contact_force=1.0,
    )
    assert stage == "recontact_lift"
    assert not ready
    np.testing.assert_allclose(
        blocked,
        [0.020, -0.010, RIGHT_RECONTACT_HEIGHT_M],
    )

    wait_for_left, stage, ready = _recontact_right_fingertip_target_local(
        np.asarray([0.030, -0.015, RIGHT_RECONTACT_CLEAR_HEIGHT_M]),
        recontact_target,
        True,
        current_slider_contact_force=0.0,
        allow_descent=False,
    )
    assert stage == "recontact_wait"
    assert ready
    np.testing.assert_allclose(
        wait_for_left,
        [0.020, -0.010, RIGHT_RECONTACT_WAIT_HEIGHT_M],
    )


def test_as013_right_servo_stages_vertical_planar_then_contact() -> None:
    final_target = np.asarray([0.004, -0.010, 0.008])
    vertical, stage = _staged_right_fingertip_target_local(
        np.asarray([0.040, -0.060, 0.050]),
        final_target,
    )
    assert stage == "vertical_entry"
    np.testing.assert_allclose(
        vertical,
        [0.040, -0.060, RIGHT_PLANAR_ENTRY_HEIGHT_M],
    )

    planar, stage = _staged_right_fingertip_target_local(
        np.asarray([0.040, -0.060, 0.024]),
        final_target,
    )
    assert stage == "planar_alignment"
    np.testing.assert_allclose(
        planar,
        [0.004, -0.010, RIGHT_PLANAR_ENTRY_HEIGHT_M],
    )

    planar_with_ik_residual, stage = _staged_right_fingertip_target_local(
        np.asarray(
            [
                0.040,
                -0.060,
                RIGHT_PLANAR_ENTRY_HEIGHT_M
                + 0.5 * RIGHT_VERTICAL_ENTRY_TOL_M,
            ]
        ),
        final_target,
    )
    assert stage == "planar_alignment"
    np.testing.assert_allclose(
        planar_with_ik_residual,
        [0.004, -0.010, RIGHT_PLANAR_ENTRY_HEIGHT_M],
    )

    planar_after_height_rebound, stage = _staged_right_fingertip_target_local(
        np.asarray([0.030, -0.030, 0.026]),
        final_target,
        "planar_alignment",
    )
    assert stage == "planar_alignment"
    np.testing.assert_allclose(
        planar_after_height_rebound,
        [0.004, -0.010, RIGHT_PLANAR_ENTRY_HEIGHT_M],
    )

    contact, stage = _staged_right_fingertip_target_local(
        np.asarray(
            [
                0.004 + 0.5 * RIGHT_PLANAR_ALIGNMENT_TOL_M,
                -0.010,
                RIGHT_PLANAR_ENTRY_HEIGHT_M,
            ]
        ),
        final_target,
    )
    assert stage == "contact_descent"
    np.testing.assert_allclose(contact, final_target)


def test_as013_causal_drive_combines_passive_and_observed_motion() -> None:
    coordinate = _body_relative_axis_coordinate(
        np.asarray([4.0, 2.25, -7.0]),
        np.asarray([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]),
        JOINT_AXIS_BODY_LOCAL,
    )
    assert coordinate == pytest.approx(0.25)

    target, common, fingertip, command = _causal_fingertip_drive_target(
        contact_joint_position=0.20,
        measured_joint_position=0.20,
        contact_fingertip_axis_coordinate=0.50,
        current_fingertip_axis_coordinate=0.55,
        contact_command_axis_coordinate=-0.10,
        current_command_axis_coordinate=-0.07,
        previous_forward_displacement_m=0.0,
        joint_limits=(0.0, 1.0),
    )
    assert fingertip == pytest.approx(0.05)
    assert command == pytest.approx(0.03)
    assert common == pytest.approx(0.03)
    assert target == pytest.approx(0.23)

    # Backward/noisy motion cannot invent progress.  A displacement already
    # reached by both live signals is retained for a stable hand release.
    target, common, fingertip, command = _causal_fingertip_drive_target(
        contact_joint_position=0.20,
        measured_joint_position=0.20,
        contact_fingertip_axis_coordinate=0.50,
        current_fingertip_axis_coordinate=0.49,
        contact_command_axis_coordinate=-0.10,
        current_command_axis_coordinate=-0.11,
        previous_forward_displacement_m=0.04,
        joint_limits=(0.0, 1.0),
    )
    assert fingertip == pytest.approx(0.0)
    assert command == pytest.approx(0.0)
    assert common == pytest.approx(0.04)
    assert target == pytest.approx(0.24)

    target, common, _fingertip, _command = _causal_fingertip_drive_target(
        contact_joint_position=0.20,
        measured_joint_position=0.20,
        contact_fingertip_axis_coordinate=0.0,
        current_fingertip_axis_coordinate=0.20,
        contact_command_axis_coordinate=0.0,
        current_command_axis_coordinate=0.20,
        previous_forward_displacement_m=0.0,
        joint_limits=(0.0, 0.25),
    )
    assert common == pytest.approx(0.20)
    assert target == pytest.approx(0.25)

    # If contact physics has already moved the blade farther than the causal
    # target, keep the passive latch at measured q instead of pulling back.
    target, common, _fingertip, _command = _causal_fingertip_drive_target(
        contact_joint_position=0.20,
        measured_joint_position=0.40,
        contact_fingertip_axis_coordinate=0.50,
        current_fingertip_axis_coordinate=0.55,
        contact_command_axis_coordinate=-0.10,
        current_command_axis_coordinate=-0.07,
        previous_forward_displacement_m=0.0,
        joint_limits=(0.0, 1.0),
    )
    assert common == pytest.approx(0.03)
    assert target == pytest.approx(0.40)


@pytest.mark.parametrize("direction,progress,near_miss", [(1, 0.95, 0.949), (-1, 0.05, 0.051)])
def test_as013_success_requires_pickup_then_left_slider_and_sampled_endpoint(direction, progress, near_miss):
    valid = dict(as013_role_schema_version=5, pickup_seen=True,
                 left_slider_contact_seen=True, target_direction=direction,
                 blade_progress=progress, target_endpoint_reached=True,
                 ordered_slide_completion_seen=True, failed=False)
    assert success_from_metrics(valid)
    for key in ("pickup_seen", "left_slider_contact_seen", "target_endpoint_reached", "ordered_slide_completion_seen"):
        assert not success_from_metrics(dict(valid, **{key: False})), key
    assert not success_from_metrics(dict(valid, failed=True))
    assert not success_from_metrics(dict(valid, blade_progress=near_miss))
    assert not success_from_metrics(dict(valid, blade_progress=float("nan")))


def test_as013_pre_action_settling_cannot_advance_success_or_read_contact():
    env = object.__new__(HandsOnAS013IsaacLabEnv)
    env._trajectory_action_phase_active = False
    env._after_task_physics_substep()
    assert not hasattr(env, "_as013_last_metrics")


def test_as013_three_camera_yaml_and_recorder_policy_hook() -> None:
    task_yaml = yaml.safe_load(
        (
            REPOSITORY_ROOT
            / "configs"
            / "multi_task"
            / f"{TASK_NAME}.yaml"
        ).read_text(encoding="utf-8")
    )
    assert task_yaml["camera_mapping"] == {
        "base": "ego",
        "wrist_left": "wrist_left",
        "wrist_right": "wrist_right",
    }
    recorder = (REPOSITORY_ROOT / "scripts" / "record_demos_zarr.py").read_text(
        encoding="utf-8"
    )
    assert '"scripted_as013_success"' in recorder
    assert "scripted_success_action" in recorder
    policy_keywords = [
        keyword
        for node in ast.walk(ast.parse(recorder))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "policy_mode"
    ]
    assert len(policy_keywords) == 1
    assert ast.unparse(policy_keywords[0].value) == (
        "bool(FLAGS.scripted_policy or scripted_as013_success or "
        "policy_actions is not None)"
    )


def test_as013_uses_strict_full_frame_static_pose_fallback() -> None:
    profile = stable_recording_origin_kwargs_for_task(TASK_NAME)
    assert profile["stability_metric"] == "velocity_or_static_pose"
    assert profile["stability_sample_seconds"] == 0.0
    assert profile["pose_translation_limit"] == pytest.approx(1.0e-6)
    assert profile["pose_rotation_limit"] == pytest.approx(1.0e-5)
    assert profile["post_settle_velocity_zeroing"] is True
    assert profile["post_settle_sleeping"] is False
    assert profile["stabilization_velocity_clamp"] is False
    assert profile["reset_translation_drift_limit"] == pytest.approx(0.005)
    assert profile["reset_rotation_drift_limit"] == pytest.approx(
        np.deg2rad(3.0)
    )
    assert profile["require_valid_contact_report"] is True
    assert profile["contact_penetration_limit"] == pytest.approx(0.0005)
