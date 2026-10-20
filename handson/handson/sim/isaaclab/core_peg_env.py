from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from handson.runtime.task_families import ABC_PREFIX_TASKS, ABCD_PREFIX_TASKS

from ..isaacsim.runtime import IsaacSimUnavailable
from .core_peg_base import (
    BimanualAssemblyIsaacLabEnv,
    _DEFAULT_RENDER_SIZE,
    _LEFT_HOME,
    _RIGHT_HOME,
    _blend_quat_wxyz,
)
from .physics_config import (
    SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT,
    SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT,
)
from .physx_ground_truth import configure_physx_contact_reports_for_recording
from .runtime_display import enforce_isaac_viewport_only, isaac_kit_args
from .recording_cameras import (
    HEAD_RECORDING_CAMERA_EYE,
    HEAD_RECORDING_CAMERA_FOVY_DEG,
    HEAD_RECORDING_CAMERA_QUAT_WXYZ_OPENGL,
    RECORDING_CAMERA_VERTICAL_APERTURE,
    RECORDING_OBJECT_XY_BOUNDS,
    WRIST_RECORDING_CAMERA_HFOV_DEG,
    WRIST_RECORDING_CAMERA_LOCAL_POSES,
    WRIST_RECORDING_CAMERA_MOUNT_POSES,
    WRIST_RECORDING_CAMERA_NEAR_M,
    WRIST_RECORDING_CAMERA_OUTPUT_SIZE,
    constrain_recording_object_xy,
    focal_length_for_horizontal_fov,
    focal_length_for_vertical_fov,
    recording_camera_profile,
    set_wrist_obstructions_hidden,
    wrist_obstruction_visibility_records,
)


_IDENTITY_QUAT_WXYZ = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_TABLE_TOP_Z = 0.89
_TASK_ROOT = "/World/Task"
_WORLD_UP = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
_DEFERRED_ROBOT_VISUAL_ROOTS = (
    "/World/HandsOnRobots/RightRobot",
    "/World/HandsOnRobots/LeftRobot",
    "/World/HandsOnRobots/CenterRig",
)
_DEFERRED_VISIBLE_HAND_PATH_MARKERS = (
    "/right_palm_link",
    "/left_palm_link",
    "/right_finger",
    "/left_finger",
)
_C04_TRIGGER_PULL_RAD = 0.30
_C04_TRIGGER_RELEASE_RAD = 0.25
_C04_TRIGGER_STIFFNESS = 0.20
_C04_TRIGGER_DAMPING = 0.10
_C04_RESET_LINEAR_DAMPING = 0.40
_C04_RESET_ANGULAR_DAMPING = 2.00
_C04_RESET_SLEEP_THRESHOLD = 0.020
_C04_TRIGGER_FRICTION = 0.015
_C04_TRIGGER_MAX_FORCE = 0.08
_C04_TRIGGER_CONTACT_FORCE = 0.10
_C04_SPRAY_GRASP_FORCE = 0.30
_C04_TRIGGER_CONTACT_GRACE_STEPS = 2
_C04_GRASP_CONTACT_GRACE_STEPS = 5
_C04_TRIGGER_ARM_STEPS = 18
_C04_SUCCESS_STEPS = 12
_C04_NOZZLE_LOCAL = np.asarray([0.0, 0.076, 0.072], dtype=np.float64)
_C04_PLANT_TARGET_OFFSET = np.asarray([0.0, 0.0, 0.08], dtype=np.float64)
_C04_AIM_MAX_ANGLE_COS = float(np.cos(np.deg2rad(18.0)))
_C04_AIM_RAY_RADIUS = 0.085
_C04_AIM_MIN_DISTANCE = 0.04
_C04_AIM_MAX_DISTANCE = 0.70
_C04_WATER_DOWNWARD_Z = -0.15
_C04_SOIL_TARGET_RADIUS = 0.135
_C04_PLANT_RAY_RADIUS = 0.120
# cone.obj is 0.10 m long with a 0.025 m radius, so its visible half-angle
# corresponds to this 0.25 slope.  Treat any sampled stream that reaches the
# plant or soil as a hit; requiring a majority made visible edge hits fail.
_C04_WATER_CONE_SPREAD = 0.025 / 0.10
_C04_WATER_HIT_FRACTION_THRESHOLD = 1.0 / 9.0
_C04_SPRAY_INIT_XY = np.asarray([-0.30, -0.22], dtype=np.float64)
_C04_SPRAY_XY_BOUNDS = np.asarray(
    [[-0.38, -0.30], [-0.18, -0.08]],
    dtype=np.float64,
)
_C04_PLANT_INIT_XY = np.asarray([-0.30, 0.14], dtype=np.float64)
_C04_PLANT_XY_BOUNDS = np.asarray(
    [[-0.38, 0.06], [-0.18, 0.24]],
    dtype=np.float64,
)
_C04_SPRAY_PLANT_MIN_DISTANCE = 0.22
_C04_SPRAY_PLANT_MAX_DISTANCE = 0.58
_C04_SPRAY_YAW_BOUNDS_DEG = (-180.0, 180.0)
_C04_SPRAY_RIGHT_ARM_MIN_XY_DISTANCE = 0.10
_C04_PLANT_RESET_HAND_MIN_XY_DISTANCE = 0.10
_C04_SPRAY_RESET_MAX_ATTEMPTS = 256
_C04_RESET_CANDIDATES_PER_OBJECT = 96
_C04_RESET_MIN_CLEARANCE_SLACK = 0.005
_C04_WATER_SAMPLE_OFFSETS = np.asarray(
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.70710678, 0.70710678),
        (0.70710678, -0.70710678),
        (-0.70710678, 0.70710678),
        (-0.70710678, -0.70710678),
    ],
    dtype=np.float64,
)
_C05_MOUSE_XY_BOUNDS = np.asarray(
    [[-0.40, -0.30], [-0.30, 0.04]],
    dtype=np.float64,
)
_C05_MOUSEPAD_XY_BOUNDS = np.asarray(
    [[-0.18, -0.30], [-0.12, 0.12]],
    dtype=np.float64,
)
_C05_DEFAULT_MOUSE_XY = np.asarray([-0.38, -0.24], dtype=np.float64)
_C05_DEFAULT_MOUSEPAD_XY = np.asarray([-0.14, -0.24], dtype=np.float64)
_C05_MOUSEPAD_MIN_OFFSET = 0.22
_C05_MOUSEPAD_RADIUS = 0.15
_C05_MOUSEPAD_Z_TOLERANCE = 0.05
_C05_MIN_MOUSE_MOVE_DISTANCE = 0.08
_C05_MOUSE_YAW_BOUNDS_DEG = (-10.0, 10.0)
_C05_MOUSE_HAND_CLEARANCE = 0.080
_C05_MOUSEPAD_HAND_CLEARANCE = 0.080
_C05_RESET_MAX_ATTEMPTS = 64
_C05_RESET_CANDIDATES_PER_OBJECT = 96
_C05_RESET_MIN_CLEARANCE_SLACK = 0.005
_C05_BUTTON_PRESS_THRESHOLD = 0.001
_C05_BUTTON_STIFFNESS = 1000.0
_C05_BUTTON_DAMPING = 100.0
_C05_BUTTON_FRICTION = 0.01
_C05_BUTTON_MAX_FORCE = 8.0
_C05_SUCCESS_STEPS = 10
_C06_CAMERA_XY_BOUNDS = np.asarray(
    [[-0.36, -0.12], [-0.18, 0.18]],
    dtype=np.float64,
)
_C06_DEFAULT_CAMERA_XY = np.asarray([-0.25, 0.0], dtype=np.float64)
_C06_CAMERA_HAND_CLEARANCE = 0.100
_C06_RESET_MAX_ATTEMPTS = 256
_C06_RESET_MIN_CLEARANCE_SLACK = 0.005
_C06_SHUTTER_JOINT = "camera_shutter_joint"
_C06_SHUTTER_PRESS_TRAVEL = 0.0005
_C06_SHUTTER_RELEASE_TRAVEL = 0.0002
_C06_SHUTTER_STIFFNESS = 80.0
_C06_SHUTTER_DAMPING = 0.5
_C06_SHUTTER_MAX_FORCE = 1.0
_C06_SHUTTER_FRICTION = 0.01
_C06_SHUTTER_RIGHT_CONTACT_FORCE = 0.02
_C06_SHUTTER_CONTACT_GRACE_STEPS = 3
_C06_CAMERA_VIEW_LOCAL = np.asarray([0.03, 0.0, 0.12], dtype=np.float64)
_C06_LOGO_LOCAL = np.asarray([0.29, 0.18, 1.1006], dtype=np.float64)
_C06_AIM_MAX_ANGLE_DEG = 30.0
_C06_AIM_MAX_RAY_ERROR = 0.12
_C07_PARTICLE_NAMES = tuple(f"c07_particle_{index}" for index in range(8))
_C07_REQUIRED_INSIDE = 7
_C07_MAX_SPILLED = 1
_C07_PARTICLE_RADIUS = 0.0065
_C07_CUP_INNER_RADIUS = 0.0235
_C07_CUP_LOCAL_Z_RANGE = (-0.032, 0.032)
_C07_PITCHER_INNER_RADIUS = 0.046
_C07_PITCHER_LOCAL_Z_RANGE = (-0.108, 0.098)
_C07_PITCHER_TILT_ALIGNMENT = float(np.cos(np.deg2rad(60.0)))
_C07_PITCHER_UPRIGHT_ALIGNMENT = float(np.cos(np.deg2rad(25.0)))
_C07_RETURN_XY_TOLERANCE = 0.025
_C07_PITCHER_HALF_HEIGHT = 0.118
_C07_RETURN_Z_TOLERANCE = 0.016
_C07_STABLE_LINEAR_SPEED = 0.020
_C07_STABLE_ANGULAR_SPEED = 0.30
_C07_HAND_RELEASE_CLEARANCE = 0.018
_C07_RELEASE_HOLD_SECONDS = 0.50
_C07_TASK_CONTACT_OFFSET = 0.00075
_C07_FINGER_CONTACT_OFFSET = 0.00050
_C07_PITCHER_BODY_RADIUS = 0.058
_C07_PITCHER_BOTTOM_Z = -0.118
_C07_PITCHER_TOP_Z = 0.108
# Invisible numerical frame fitted to the three visible parts of the original
# YCB 019 handle.  It is intentionally never authored as MJCF/USD geometry:
# transparent helper geoms can be rendered opaque by some Isaac import paths.
_C07_HANDLE_FRAME_RADIUS = 0.013
_C07_HANDLE_FRAME_SEGMENTS = np.asarray(
    [
        [[0.0, -0.0402, -0.0396], [0.0, -0.0857, 0.0056]],
        [[0.0, -0.0818, 0.0071], [0.0, -0.0818, 0.0904]],
        [[0.0, -0.0850, 0.0895], [0.0, -0.0476, 0.1072]],
    ],
    dtype=np.float64,
)
_C07_HANDLE_CONTACT_MARGIN = 0.014
_C07_HANDLE_NEAR_CONTACT_MARGIN = 0.009
_C07_HANDLE_GRIP_CENTER_MAX_DISTANCE = 0.030
_C07_HANDLE_PINCH_MAX_DISTANCE = 0.095
_C07_HANDLE_MIN_PINCH_SPAN = 0.012
_C07_HANDLE_MAX_RADIAL_DOT = -0.20
_C07_HANDLE_PINCH_LINE_MAX_DISTANCE = 0.020
_C07_HANDLE_PROJECTION_MAX_SEPARATION = 0.040
_C07_HANDLE_GRASP_PALM_LOCAL = np.asarray([0.0, 0.0, 0.075], dtype=np.float64)
_C07_GRASP_ATTACH_HAND_COMMAND = 0.82
_C07_SPOUT_OUTLET_LOCAL = np.asarray([0.0, 0.089, 0.097], dtype=np.float64)
_C07_POUR_XY_CAPTURE_RADIUS = 0.065
_C07_POUR_MIN_HEIGHT_ABOVE_CUP = 0.060
_C07_POUR_MAX_HEIGHT_ABOVE_CUP = 0.320
_C07_POUR_TARGET_SPOUT_HEIGHT_ABOVE_CUP = 0.140
_C07_POUR_EMPTY_HOLD_SECONDS = 0.0
# The opening normal is the pitcher's local +Z axis.  Flow begins only after
# it crosses horizontal and points ten degrees down.
_C07_PARTICLE_RELEASE_TOTAL_TILT_DEG = 100.0
# The YCB pitcher has a narrow visible spout and its Isaac-only hollow
# collider deliberately favours a stable body over detailed fluid dynamics.
# Move pinned particles continuously through its interior and visible spout;
# never teleport them from the saved bottom positions to the outlet.
_C07_POUR_MAX_TOTAL_TILT_DEG = 150.0
# From upright, 150 degrees leaves the opening normal 60 degrees below the
# world horizontal plane.  A six-centimetre upward glove stroke requests the
# full arc; the applied hand/pitcher pair advances together at a safe rate.
_C07_POUR_FULL_TILT_INPUT_M = 0.060
_C07_POUR_TILT_RATE_DEG_PER_S = 90.0
_C07_POUR_ALIGN_DURATION_SECONDS = 0.20
_C07_POUR_ALIGN_TRANSLATION_RATE_MPS = 0.25
_C07_POUR_ALIGN_ROTATION_RATE_DEG_PER_S = 90.0
_C07_POUR_TRACK_POSITION_TOLERANCE = 0.015
_C07_POUR_TRACK_ANGLE_TOLERANCE = float(np.deg2rad(8.0))
_C07_PARTICLE_RELEASE_STABLE_STEPS = 4
_C07_PARTICLE_FIRST_RELEASE_DELAY_SECONDS = 0.10
_C07_PARTICLE_RELEASE_INTERVAL_SECONDS = 0.10
_C07_PARTICLE_RELEASE_EXIT_HOLD_STEPS = 2
_C07_PARTICLE_STAGE_SPEED_MPS = 0.60
_C07_PARTICLE_STAGE_REACHED_EPS_M = 0.0005
_C07_PARTICLE_STAGE_CLEAR_Z_LOCAL = 0.109
_C07_PARTICLE_MOUTH_SURFACE_CLEARANCE = 0.002
# Particle staging uses a clearance line above the visible YCB spout lip and
# exits along local +Y.  The geometric lock pivot remains the lower centre of
# the outlet; using it as the ball centre made the sphere cross the rendered
# mesh even after the invisible collision channel had been opened.
_C07_PARTICLE_MOUTH_CENTER_LOCAL = np.asarray(
    [0.0, 0.0915, _C07_PARTICLE_STAGE_CLEAR_Z_LOCAL],
    dtype=np.float64,
)
_C07_PARTICLE_MOUTH_CENTER_DISTANCE = (
    _C07_PARTICLE_RADIUS + _C07_PARTICLE_MOUTH_SURFACE_CLEARANCE
)
_C07_PARTICLE_EXIT_LOCAL = (
    _C07_PARTICLE_MOUTH_CENTER_LOCAL
    + np.asarray(
        [0.0, _C07_PARTICLE_MOUTH_CENTER_DISTANCE, 0.0],
        dtype=np.float64,
    )
)
_C07_PARTICLE_EXIT_X_OFFSETS = np.asarray(
    (-0.0025, 0.0025, 0.0, -0.0015, 0.0015, -0.0020, 0.0020, 0.0),
    dtype=np.float64,
)
_C07_PARTICLE_STABLE_POSITION_TOLERANCE = 0.00075
_C07_PARTICLE_STABLE_ANGLE_TOLERANCE = float(np.deg2rad(0.75))
_C07_PARTICLE_CAPTURE_RADIUS = 0.032
_C07_PARTICLE_CAPTURE_Z_RANGE = (-0.040, 0.055)
_C07_RETURN_ASSIST_RADIUS = 0.080
_C07_RETURN_ASSIST_MAX_Z_ERROR = 0.090
_C07_RETURN_ASSIST_UPRIGHT_ALIGNMENT = float(np.cos(np.deg2rad(35.0)))
_C07_RETURN_PIN_RADIUS = 0.060
_C07_RETURN_PIN_MAX_Z_ERROR = 0.045
_C07_RETURN_PIN_UPRIGHT_ALIGNMENT = float(np.cos(np.deg2rad(35.0)))
_C07_NOMINAL_PITCHER_XY = np.asarray([-0.38, -0.24], dtype=np.float64)
_C07_NOMINAL_CUP_XY = np.asarray([-0.20, -0.11], dtype=np.float64)
_C07_PITCHER_XY_BOUNDS = np.asarray(
    [[-0.40, -0.30], [-0.30, -0.12]],
    dtype=np.float64,
)
_C07_CUP_XY_BOUNDS = np.asarray(
    [[-0.25, -0.18], [-0.12, 0.06]],
    dtype=np.float64,
)
_C07_PITCHER_YAW_BOUNDS_DEG = (-35.0, 35.0)
_C07_CUP_YAW_BOUNDS_DEG = (-25.0, 25.0)
_C07_MIN_PITCHER_CUP_DISTANCE = 0.20
_C07_MAX_PITCHER_CUP_DISTANCE = 0.34
_C07_HAND_CLEARANCE = 0.045
_C07_PITCHER_FOOTPRINT_RADIUS = 0.095
_C07_HANDLE_LOCAL_XY = np.asarray([0.0, -0.086], dtype=np.float64)
_C07_HANDLE_FOOTPRINT_RADIUS = 0.035
_C07_CUP_FOOTPRINT_RADIUS = 0.060
_C07_RESET_MAX_ATTEMPTS = 256
_C07_PARTICLE_PILE_OFFSET = np.asarray([0.000, 0.000], dtype=np.float64)
_C07_PARTICLE_CUP_CLEARANCE = 0.095
_C07_CAMERA_XY_BOUNDS = np.asarray(
    [[-0.40, -0.34], [-0.12, 0.30]],
    dtype=np.float64,
)
_C07_PARTICLE_XY_OFFSETS = np.asarray(
    [
        (0.000, 0.000),
        (0.018, 0.000),
        (-0.018, 0.000),
        (0.000, 0.018),
        (0.000, -0.018),
        (0.010, 0.010),
        (-0.010, 0.010),
        (0.000, -0.010),
    ],
    dtype=np.float64,
)
_C08_ICE_NAMES = tuple(f"c08_ice_{index}" for index in range(4))
_C08_REQUIRED_ICE = 4
_C08_ICE_SIDE_LENGTH = 0.026
_C08_ICE_HALF_EXTENT = 0.5 * _C08_ICE_SIDE_LENGTH
_C08_RIGHT_HAND_XY_BOUNDS = RECORDING_OBJECT_XY_BOUNDS.copy()
_C08_RESET_XY_BOUNDS = {
    "c08_tongs": np.asarray([[-0.27, -0.27], [-0.14, -0.13]], dtype=np.float64),
    "c08_glass_cup": np.asarray([[-0.39, -0.02], [-0.18, 0.25]], dtype=np.float64),
    **{
        name: np.asarray([[-0.39, -0.10], [-0.18, 0.12]], dtype=np.float64)
        for name in _C08_ICE_NAMES
    },
}
_C08_RESET_XY_FALLBACK = {
    "c08_tongs": np.asarray([-0.15, -0.19], dtype=np.float64),
    "c08_glass_cup": np.asarray([-0.36, 0.205], dtype=np.float64),
    "c08_ice_0": np.asarray([-0.385, -0.025], dtype=np.float64),
    "c08_ice_1": np.asarray([-0.335, -0.025], dtype=np.float64),
    "c08_ice_2": np.asarray([-0.385, 0.025], dtype=np.float64),
    "c08_ice_3": np.asarray([-0.335, 0.025], dtype=np.float64),
}
_C08_RESET_FOOTPRINT_RADII = {
    "c08_glass_cup": 0.034,
    **{
        name: float(np.sqrt(2.0)) * _C08_ICE_HALF_EXTENT
        for name in _C08_ICE_NAMES
    },
}
_C08_TONGS_RESET_LOCAL_SEGMENT = np.asarray(
    [[-0.145, 0.0], [0.290, 0.0]],
    dtype=np.float64,
)
_C08_TONGS_RESET_FOOTPRINT_RADIUS = 0.035
_C08_TONGS_PICKUP_MAX_DISTANCE = 0.32
_C08_OBJECT_HAND_CLEARANCE_MARGIN = 0.050
_C08_TONGS_HAND_CLEARANCE_MARGIN = 0.030
_C08_OBJECT_CLEARANCE_MARGIN = 0.010
_C08_CUP_OBJECT_CLEARANCE_MARGIN = 0.015
_C08_RESET_MIN_CLEARANCE_SLACK = 0.005
_C08_RESET_CANDIDATES_PER_OBJECT = 128
_C08_RESET_MAX_ATTEMPTS = 96
_C08_OBJECT_YAW_BOUNDS_DEG = (-180.0, 180.0)
_C08_CUP_INNER_RADIUS = 0.023
_C08_CUP_LOCAL_Z_RANGE = (-0.025, 0.040)
_C08_TONGS_PAD_RADIUS = 0.014
_C08_TONGS_FIXED_PAD_SEGMENT = np.asarray(
    [[0.102, -0.018, 0.012], [0.151, -0.018, 0.012]],
    dtype=np.float64,
)
_C08_TONGS_MOVING_PAD_SEGMENT = np.asarray(
    [[0.222, 0.0, 0.0], [0.271, 0.0, 0.0]],
    dtype=np.float64,
)
_C08_TONGS_FIXED_COLLISION_CAPSULES = (
    (
        np.asarray(
            [[-0.128, -0.018, 0.012], [-0.128, 0.018, 0.012]],
            dtype=np.float64,
        ),
        0.011,
    ),
    (
        np.asarray(
            [[-0.120, -0.018, 0.012], [0.112, -0.018, 0.012]],
            dtype=np.float64,
        ),
        0.006,
    ),
    (_C08_TONGS_FIXED_PAD_SEGMENT.copy(), _C08_TONGS_PAD_RADIUS),
)
_C08_TONGS_MOVING_ARM_ROOT = np.asarray(
    [-0.120, 0.018, 0.012],
    dtype=np.float64,
)
_C08_TONGS_MOVING_COLLISION_CAPSULES = (
    (
        np.asarray([[0.0, 0.0, 0.0], [0.232, 0.0, 0.0]], dtype=np.float64),
        0.006,
    ),
    (_C08_TONGS_MOVING_PAD_SEGMENT.copy(), _C08_TONGS_PAD_RADIUS),
)
_C08_TONGS_COLLISION_SAMPLE_STEP = 0.004
_C08_TONGS_SWEEP_MAX_POINT_STEP = 0.004
_C08_TONGS_MAX_ALLOWED_PENETRATION = 0.001
_C08_TONGS_COLLISION_BISECTION_STEPS = 12
_C08_TONGS_SUPPORT_ESCAPE_CLEARANCE = 0.010
_C08_TONGS_SUPPORT_ESCAPE_MIN_LIFT = 0.002
_C08_TONGS_SUPPORT_ESCAPE_STEP = 0.030
_C08_TONGS_MAX_INNER_GAP = 1.5 * _C08_ICE_SIDE_LENGTH
_C08_TONGS_OPEN_RAD = 0.121765
_C08_TONGS_CLOSED_RAD = 0.070
_C08_TONGS_REST_RAD = _C08_TONGS_OPEN_RAD
_C08_TONGS_SPRING_STIFFNESS = 1.00
_C08_TONGS_SPRING_DAMPING = 0.12
_C08_TONGS_SPRING_MAX_FORCE = 0.50
_C08_TONGS_JOINT_FRICTION = 0.012
_C08_GLASS_UPRIGHT_ALIGNMENT = float(np.cos(np.deg2rad(20.0)))
_C08_TONGS_FIXED_GRIP_SEGMENT = np.asarray(
    [[-0.105, -0.018, 0.012], [-0.025, -0.018, 0.012]],
    dtype=np.float64,
)
_C08_TONGS_MOVING_GRIP_SEGMENT = np.asarray(
    [[0.015, 0.0, 0.0], [0.095, 0.0, 0.0]],
    dtype=np.float64,
)
_C08_TONGS_GRIP_RADIUS = 0.009
_C08_TONGS_GRIP_CONTACT_MARGIN = 0.011
_C08_TONGS_PINCH_MAX_DISTANCE = 0.095
_C08_TONGS_CONTROL_OPEN_HAND = 0.75
_C08_TONGS_CONTROL_CLOSED_HAND = 1.35
_C08_TONGS_JOINT_MAX_RATE = 0.8
_C08_ICE_CAPTURE_RADIUS = 0.035
_C08_ICE_CAPTURE_RAD = 0.075
_C08_ICE_RELEASE_RAD = _C08_TONGS_OPEN_RAD - 0.004
_C08_ICE_BETWEEN_JAWS_MARGIN = 0.004
_C08_ICE_MAX_JAW_AXIS_FRACTION = 0.88
_C08_ICE_MIN_JAW_AXIS_FRACTION = 0.12
_C08_ICE_MAX_PERP_DISTANCE = 0.030
_C08_ICE_ONE_JAW_CONTACT_TOLERANCE = 0.005
_C08_ICE_BOTH_JAW_CONTACT_TOLERANCE = 0.002
_C08_ICE_INNER_NORMAL_ALIGNMENT = 0.80
_C08_ICE_MAX_INNER_TRANSVERSE_ERROR = 0.018
_C08_ICE_AUTO_CLOSE_MAX_DISTANCE = 0.045
_C08_POUR_XY_CAPTURE_RADIUS = 0.090
_C08_POUR_MIN_HEIGHT_ABOVE_CUP = 0.035
_C08_POUR_MAX_HEIGHT_ABOVE_CUP = 0.350
_C08_HELD_ICE_CUP_CAPTURE_RADIUS = 0.060
_C08_HELD_ICE_CUP_MIN_Z = 0.020
_C08_HELD_ICE_CUP_MAX_Z = 0.180
_C08_DROP_ICE_LOCAL_Z = 0.058
_C08_RELEASE_ALIGN_POSITION_TOLERANCE = 0.012
_C08_RELEASE_ALIGN_ANGLE_TOLERANCE = float(np.deg2rad(10.0))
_C08_RELEASE_ALIGN_MAX_LINEAR_SPEED = 0.20
_C08_RELEASE_CENTERED_CUP_RADIUS = 0.004
_C08_RELEASE_CENTERED_Z_TOLERANCE = 0.006
_C08_RELEASE_RIM_TRANSIT_CLEARANCE = 0.008
_C08_RELEASE_ALIGN_TIMEOUT_SECONDS = 4.00
_C08_CUP_POSE_MISSING_TIMEOUT_SECONDS = 0.10
_C08_RELEASE_OPEN_TIMEOUT_SECONDS = 0.10
_C08_RELEASE_OPEN_FAILURE_TIMEOUT_SECONDS = 0.20
_C08_RELEASE_OPEN_COMMAND_SECONDS = (
    max(0.0, _C08_ICE_RELEASE_RAD - _C08_TONGS_CLOSED_RAD)
    / _C08_TONGS_JOINT_MAX_RATE
    + 0.015
)
_C08_DROP_TIMEOUT_SECONDS = 1.00
_C08_DROP_SWEEP_CAPTURE_RADIUS = 0.012
_C08_CUP_RIM_LOCAL_Z = 0.031
_C08_DROP_CLEARANCE_ABOVE_SLOT = 0.040
_C08_STACK_CAPTURE_HEIGHT_TOLERANCE = 0.018
_C08_DELIVERED_ICE_LOCAL_SLOTS = np.asarray(
    [
        [-0.003, -0.003, -0.010],
        [0.003, 0.003, 0.0165],
        [-0.003, 0.003, 0.043],
        [0.003, -0.003, 0.0695],
    ],
    dtype=np.float64,
)
_C09_PAPER_NAMES = ("c09_paper_0", "c09_paper_1")
_C09_TOP_OPEN_RAD = 0.35
_C09_TOP_PRESSED_RAD = 0.08
_C09_PAPER_HALF_EXTENTS_XY = np.asarray([0.075, 0.050], dtype=np.float64)
_C09_GRIP_TAB_CENTER_LOCAL = np.asarray(
    [-0.060, -0.038, 0.006],
    dtype=np.float64,
)
_C09_GRIP_TAB_CENTERS_LOCAL = {
    "c09_paper_0": _C09_GRIP_TAB_CENTER_LOCAL,
    "c09_paper_1": np.asarray(
        [-0.060, 0.038, 0.006],
        dtype=np.float64,
    ),
}
_C09_GRIP_TAB_HALF_EXTENTS = np.asarray(
    [0.012, 0.010, 0.004],
    dtype=np.float64,
)
_C09_GRIP_TAB_CONTACT_MARGIN = 0.011
_C09_GRIP_TAB_NEAR_MARGIN = 0.0085
_C09_GRIP_TAB_PINCH_MIN_DISTANCE = 0.008
_C09_GRIP_TAB_PINCH_MAX_DISTANCE = 0.060
_C09_GRASP_ATTACH_HAND_COMMAND = 0.82
_C09_GRASP_RELEASE_HAND_COMMAND = 0.70
_C09_MIN_PAPER_OVERLAP_RATIO = 0.90
_C09_PAPER_Z_TOLERANCE = 0.012
_C09_PAPER_YAW_ALIGNMENT = float(np.cos(np.deg2rad(15.0)))
_C09_WORK_ZONE_HALF_EXTENTS = np.asarray([0.080, 0.055, 0.030], dtype=np.float64)
_C09_STACK_CAPTURE_OVERLAP_RATIO = 0.80
_C09_STACK_CAPTURE_Z_TOLERANCE = 0.015
_C09_STACK_CAPTURE_ALIGNMENT = float(np.cos(np.deg2rad(20.0)))
_C09_STACK_SEPARATION = 0.004
_C09_STACK_ALIGN_MAX_LINEAR_SPEED = 0.15
_C09_STACK_ALIGN_MAX_ANGULAR_SPEED = float(np.deg2rad(90.0))
_C09_STAPLE_MARK_LOCAL = np.asarray([0.060, 0.038, 0.0025], dtype=np.float64)
_C09_STAPLE_ZONE_LOCAL = np.asarray([0.080, 0.0, 0.020], dtype=np.float64)
_C09_STAPLER_CAPTURE_XY = 0.035
_C09_STAPLER_CAPTURE_Z = 0.025
_C09_STAPLER_CAPTURE_ALIGNMENT = float(np.cos(np.deg2rad(20.0)))
_C09_FINAL_MARK_XY_TOLERANCE = 0.008
_C09_FINAL_MARK_Z_TOLERANCE = 0.006
_C09_HAND_TRACK_POSITION_TOLERANCE = 0.012
_C09_HAND_TRACK_ANGLE_TOLERANCE = float(np.deg2rad(8.0))
_C09_INSERT_MAX_LINEAR_SPEED = 0.15
_C09_INSERT_MAX_ANGULAR_SPEED = float(np.deg2rad(75.0))
_C09_GUIDE_STABLE_STEPS = 3
_C10_PLUG_LATERAL_TOLERANCE = 0.008
_C10_PLUG_MIN_INSERTION_DEPTH = 0.032
_C10_PLUG_SHOULDER_TOLERANCE = 0.003
_C10_CABLE_LATERAL_TOLERANCE = 0.005
_C10_CABLE_MIN_INSERTION_DEPTH = 0.014
_C10_CABLE_SHOULDER_TOLERANCE = 0.0025
_C10_PLUG_AXIS_ALIGNMENT = float(np.cos(np.deg2rad(15.0)))
_C10_CAPTURE_AXIS_ALIGNMENT = float(np.cos(np.deg2rad(30.0)))
_C10_IN_CHANNEL_SECONDARY_ALIGNMENT = float(np.cos(np.deg2rad(10.0)))
_C10_PLUG_CAPTURE_LATERAL = 0.018
_C10_CABLE_CAPTURE_LATERAL = 0.012
_C10_CAPTURE_MIN_DEPTH = -0.020
_C10_CAPTURE_PULL_IN_DEPTH = -0.010
_C10_GRASP_ATTACH_HAND_COMMAND = 0.82
_C10_GRASP_RELEASE_HAND_COMMAND = 0.70
_C10_PLUG_GRASP_BOX_CENTER = np.asarray([-0.028, 0.0, 0.0], dtype=np.float64)
_C10_PLUG_GRASP_BOX_HALF_EXTENTS = np.asarray(
    [0.022, 0.028, 0.022],
    dtype=np.float64,
)
_C10_CABLE_GRASP_BOX_CENTER = np.zeros(3, dtype=np.float64)
_C10_CABLE_GRASP_BOX_HALF_EXTENTS = np.asarray(
    [0.025, 0.014, 0.010],
    dtype=np.float64,
)
_C10_GRASP_CONTACT_MARGIN = 0.011
_C10_GRASP_NEAR_MARGIN = 0.0085
_C10_PLUG_PINCH_DISTANCE_RANGE = (0.015, 0.085)
_C10_CABLE_PINCH_DISTANCE_RANGE = (0.010, 0.055)
_C10_HAND_TRACK_POSITION_TOLERANCE = 0.015
_C10_HAND_TRACK_ANGLE_TOLERANCE = float(np.deg2rad(8.0))
_C10_ALIGN_MAX_LINEAR_SPEED = 0.22
_C10_ALIGN_MAX_ANGULAR_SPEED = float(np.deg2rad(90.0))
_C10_INSERT_MAX_LINEAR_SPEED = 0.12
_C10_APPROACH_PLUG_DEPTH = -0.004
_C10_APPROACH_CABLE_DEPTH = -0.003
_C10_PLUG_IDEAL_DEPTH = 0.034
_C10_CABLE_IDEAL_DEPTH = 0.016
_C10_LATCH_STABLE_STEPS = 3
_C10_PLUG_SEATED_RELATIVE = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.032],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_C10_CABLE_SEATED_RELATIVE = np.asarray(
    [
        [1.0, 0.0, 0.0, -0.108],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_C_TOOL_GRASP_ATTACH_HAND_COMMAND = 1.00
_C_TOOL_GRASP_RELEASE_HAND_COMMAND = 0.75
_C_TOOL_ASSIST_MAX_LINEAR_SPEED = 0.80
_C_TOOL_ASSIST_MAX_ANGULAR_SPEED = 4.0
_C_TOOL_LOCK_POSITION_TOLERANCE = 0.006
_C_TOOL_LOCK_ANGLE_TOLERANCE = float(np.deg2rad(5.0))
_C_TOOL_WUJI_POWER_GRIP = np.asarray(
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
_C08_TONGS_WUJI_OPEN_GRIP = np.asarray(
    [
        0.54,
        0.10,
        0.68,
        0.55,
        0.61,
        0.00,
        0.72,
        0.59,
        0.63,
        0.00,
        0.74,
        0.61,
        0.61,
        0.00,
        0.72,
        0.59,
        0.57,
        0.00,
        0.69,
        0.56,
    ],
    dtype=np.float64,
)
_C08_TONGS_WUJI_CLOSED_GRIP = _C_TOOL_WUJI_POWER_GRIP.copy()
_C11_KEY_NAMES = ("c11_key_red", "c11_key_blue", "c11_key_yellow", "c11_key_green")
_C11_DEFAULT_SEQUENCE = _C11_KEY_NAMES
_C11_MIN_SEQUENCE_LENGTH = 3
_C11_MAX_SEQUENCE_LENGTH = 5
_C11_KEY_COLOR_LABELS = {
    "c11_key_red": "红",
    "c11_key_blue": "蓝",
    "c11_key_yellow": "黄",
    "c11_key_green": "绿",
}
_C11_MALLET_HEAD_LOCAL = np.asarray([0.0, 0.105, 0.0], dtype=np.float64)
_C11_MALLET_HANDLE_SEGMENT = np.asarray(
    [[0.0, -0.085, 0.0], [0.0, 0.075, 0.0]],
    dtype=np.float64,
)
_C11_MALLET_HANDLE_RADIUS = 0.006
_C11_MALLET_CONTACT_MARGIN = 0.0115
_C11_MALLET_NEAR_CONTACT_MARGIN = 0.0085
_C11_MALLET_PINCH_MIN_DISTANCE = 0.012
_C11_MALLET_PINCH_MAX_DISTANCE = 0.055
_C11_MALLET_MAX_AXIAL_SEPARATION = 0.035
_C11_GRASP_ATTACH_HAND_COMMAND = 0.82
_C11_GRASP_RELEASE_HAND_COMMAND = 0.70
_C11_KEY_HALF_EXTENTS = {
    "c11_key_red": np.asarray([0.047, 0.022, 0.006], dtype=np.float64),
    "c11_key_yellow": np.asarray([0.051, 0.022, 0.006], dtype=np.float64),
    "c11_key_blue": np.asarray([0.055, 0.022, 0.006], dtype=np.float64),
    "c11_key_green": np.asarray([0.059, 0.022, 0.006], dtype=np.float64),
}
_C11_MALLET_HEAD_RADIUS = 0.016
_C11_KEY_CONTACT_HEAD_Z = 0.022
_C11_KEY_CONTACT_Z_TOLERANCE = 0.005
_C11_KEY_HIT_XY_MARGIN = 0.006
_C11_KEY_REARM_LOCAL_Z = 0.050
_C11_KEY_MIN_DOWNWARD_STEP = 0.0005
_C11_GUIDE_CAPTURE_XY_MARGIN = 0.020
_C11_GUIDE_LOCAL_Z_RANGE = (0.030, 0.100)
_C11_GUIDE_MAX_LINEAR_SPEED = 0.15
_C11_GUIDE_MAX_ANGULAR_SPEED = float(np.deg2rad(60.0))
_C11_HAND_TRACK_POSITION_TOLERANCE = 0.015
_C11_HAND_TRACK_ANGLE_TOLERANCE = float(np.deg2rad(8.0))
_C11_RETURN_XY_TOLERANCE = 0.075
_C11_RETURN_GUIDE_CAPTURE_RADIUS = 0.090
_C11_RETURN_CENTER_HEIGHT = 0.020
_C11_RETURN_Z_TOLERANCE = 0.012
_C11_RETURN_MAX_LINEAR_SPEED = 0.08
_C11_RETURN_MAX_ANGULAR_SPEED = float(np.deg2rad(25.0))
_C12_BLOCK_NAMES = tuple(f"c12_block_{index}" for index in range(3))
_C12_BLOCK_HALF_EXTENTS = np.asarray([0.0215, 0.050, 0.0215], dtype=np.float64)
_C12_LINE_DISTANCE_TOLERANCE = 0.020
_C12_ORIENTATION_ALIGNMENT = float(np.cos(np.deg2rad(10.0)))
_C12_SPACING_RANGE = (0.055, 0.140)
_C12_TOOL_MIN_MOVE_DISTANCE = 0.060
_C12_HAND_BLOCK_CLEARANCE = 0.015
_C12_RESET_BLOCK_CLEARANCE = 0.065
_C12_RESET_HAND_CLEARANCE = 0.080
_C12_RESET_MAX_ATTEMPTS = 64
_C12_RESET_CANDIDATES_PER_BLOCK = 96
_C12_RESET_MIN_CLEARANCE_SLACK = 0.005
_C12_DIRECT_TOUCH_DISTANCE = 0.006
_C12_BLOCK_LOCAL_X_CENTERS = np.asarray([-0.11, 0.0, 0.11], dtype=np.float64)
_C12_TARGET_HALF_EXTENTS_XY = np.asarray([0.050, 0.022], dtype=np.float64)
_C12_BLOCK_FOOTPRINT_RADIUS = float(np.linalg.norm(_C12_BLOCK_HALF_EXTENTS[:2]))
_C12_SET_SQUARE_LOCAL_XY = np.asarray([-0.14, -0.27], dtype=np.float64)
_C12_MIN_TARGET_OVERLAP_RATIO = 0.90
_C12_BLOCK_LOCAL_X_JITTER = 0.022
_C12_BLOCK_LOCAL_RIGHT_Y_BOUNDS = (-0.265, -0.075)
_C12_BLOCK_YAW_BOUNDS_DEG = (-65.0, 65.0)
_C12_TOOL_SEGMENTS_LOCAL = (
    np.asarray([[-0.009, 0.0, 0.0], [0.141, 0.0, 0.0]], dtype=np.float64),
    np.asarray([[0.0, -0.009, 0.0], [0.0, 0.121, 0.0]], dtype=np.float64),
    np.asarray([[0.006, 0.115, 0.0], [0.135, 0.006, 0.0]], dtype=np.float64),
)
_C12_TOOL_SEGMENT_RADII = (0.0135, 0.0135, 0.0095)
_C12_TOOL_GRASP_CONTACT_MARGIN = 0.010
_C12_TOOL_GRASP_NEAR_MARGIN = 0.0075
_C12_TOOL_PINCH_MIN_DISTANCE = 0.014
_C12_TOOL_PINCH_MAX_DISTANCE = 0.070
_C12_TOOL_GRASP_ATTACH_HAND_COMMAND = 0.82
_C12_TOOL_GRASP_RELEASE_HAND_COMMAND = 0.70
_C12_TOOL_BLOCK_CONTACT_TOLERANCE = 0.006
_C12_TOOL_BLOCK_CONTACT_DEBOUNCE_STEPS = 2
_C12_PUSH_DIRECTION_ALIGNMENT = float(np.cos(np.deg2rad(45.0)))
_C12_PUSH_MIN_FORWARD_STEP = 0.00015
_C12_PUSH_MAX_LINEAR_SPEED = 0.18
_C12_PUSH_MAX_ANGULAR_SPEED = float(np.deg2rad(60.0))
_C12_HAND_TRACK_POSITION_TOLERANCE = 0.015
_C12_HAND_TRACK_ANGLE_TOLERANCE = float(np.deg2rad(8.0))
_C12_SETTLE_HOLD_STEPS = 4
_C01_HAMMER_INIT_XY = np.asarray([-0.34, 0.0], dtype=np.float64)
_C01_WOOD_INIT_XY = np.asarray([-0.16, 0.0], dtype=np.float64)
_C01_HAMMER_XY_BOUNDS = np.asarray(
    [[-0.44, -0.15], [0.02, 0.15]],
    dtype=np.float64,
)
_C01_WOOD_XY_BOUNDS = np.asarray(
    [[-0.30, -0.16], [-0.08, 0.16]],
    dtype=np.float64,
)
_C01_HAMMER_WOOD_MIN_DISTANCE = 0.10
_C01_HAMMER_WOOD_SCREEN_CLEARANCE = 0.08
_C01_HAMMER_YAW_BOUNDS_DEG = (-180.0, 180.0)
_C01_HAMMER_HAND_SCREEN_CLEARANCE = 0.22
_C01_WOOD_HAND_SCREEN_CLEARANCE = 0.10
_C01_WOOD_HAND_MIN_XY_DISTANCE = 0.10
_C01_OBJECT_RESET_MAX_ATTEMPTS = 1024
_C01_HAMMER_RESET_Z_OFFSET = 0.020
_C01_HAMMER_SCREEN_Z = 0.96
_C01_WOOD_SCREEN_Z = 0.94
_C01_HAMMER_SCREEN_X_LIMIT = 0.90
_C01_HAMMER_SCREEN_Y_BOUNDS = (-0.94, 0.70)
_C01_HAMMER_FALLBACK_XY = np.asarray([0.02, 0.0], dtype=np.float64)
_C01_HAMMER_FALLBACK_YAW_DEG = 60.0
# The imported hammer's 31.8 cm major axis points 30 degrees from world X at
# zero randomized yaw. Couple the front X bound to yaw so the full hammer stays
# visible, while the rear bound leaves room for every orientation above the
# reset hands.
_C01_HAMMER_MAJOR_HALF_LENGTH = 0.16
_C01_HAMMER_MAJOR_AXIS_OFFSET_DEG = 30.0
_C01_HAMMER_MIN_FOOTPRINT_X = -0.44
_TASK_OBJECT_COMMON_XY_JITTER = 0.025
_TASK_OBJECT_RELATIVE_XY_JITTER = 0.012
_TASK_OBJECT_YAW_BOUNDS_DEG = (-35.0, 35.0)
_TASK_OBJECT_RESET_MAX_ATTEMPTS = 128
_TASK_OBJECT_RESET_CANDIDATES_PER_OBJECT = 96
_TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK = 0.005
_TASK_OBJECT_HAND_CLEARANCE = 0.085
_C_TOOL_GROUP_RANDOMIZED_TASKS = frozenset(
    {
        "c08_tongs_ice_cubes",
        "c09_staple_paper_sheets",
        "t02_connect_charger_and_cable",
        "c11_xylophone_sequence",
        "t11_line_up_blocks",
    }
)
_C_TOOL_CLEAR_START_TASKS = frozenset(
    {
        "c07_pitcher_pour_cup",
        "c08_tongs_ice_cubes",
        "c09_staple_paper_sheets",
        "t02_connect_charger_and_cable",
        "c11_xylophone_sequence",
        "t11_line_up_blocks",
    }
)
_C_TOOL_SPREAD_ARM_HOME = {
    "right": np.asarray(
        [
            -1.5707963,
            1.5707963,
            2.0207963,
            1.5707963,
            -1.5707963,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    ),
    "left": np.asarray(
        [
            1.5707963,
            1.5707963,
            -2.0207963,
            1.5707963,
            1.5707963,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    ),
}
_C_TOOL_SPREAD_RIGHT_START_TCP = _RIGHT_HOME.copy()
_C_TOOL_SPREAD_RIGHT_START_TCP[1] = -0.36
_C_TOOL_SPREAD_LEFT_START_TCP = _LEFT_HOME.copy()
_C_TOOL_SPREAD_LEFT_START_TCP[1] = 0.36
_C_TOOL_GROUP_PIVOTS = {
    "c08_tongs_ice_cubes": np.asarray([-0.36, 0.00], dtype=np.float64),
    "c09_staple_paper_sheets": np.asarray([-0.30, 0.00], dtype=np.float64),
    "t02_connect_charger_and_cable": np.asarray([-0.32, 0.00], dtype=np.float64),
    "c11_xylophone_sequence": np.asarray([-0.31, 0.02], dtype=np.float64),
    "t11_line_up_blocks": np.asarray([-0.28, 0.00], dtype=np.float64),
}
_C_TOOL_GROUP_YAW_BOUNDS_DEG = (-18.0, 18.0)
_TASK_TABLE_HEIGHT_BOUNDS = {
    # On B03, measured resets at >= 0.0385 m made the tabletop contact the
    # right-hand home pose and physically push joint7.  A 0.0272 m sample was
    # successful, but cap at 0.025 m to keep a collision-clear safety margin.
    "t09_insert_peg": (0.0, 0.025),
    "t44_stack_blocks": (0.0, 0.02),
    "t08_hammer_nail": (0.0, 0.03),
    "t50_water_plant": (0.0, 0.02),
    # The C08 tongs rest on a fixed cradle.  Keep their shared tabletop height
    # unchanged while randomizing the cup and cubes in XY/yaw.
    "c08_tongs_ice_cubes": (0.0, 0.0),
}
_A01_RIGHT_HAND_XY_BOUNDS = np.asarray(
    [(-0.40, -0.30), (-0.12, 0.30)],
    dtype=np.float64,
)
_A01_RESET_XY_BOUNDS = {
    "bucket": RECORDING_OBJECT_XY_BOUNDS.copy(),
    "011_banana": RECORDING_OBJECT_XY_BOUNDS.copy(),
    "apple": RECORDING_OBJECT_XY_BOUNDS.copy(),
    "orange": RECORDING_OBJECT_XY_BOUNDS.copy(),
    "pear": RECORDING_OBJECT_XY_BOUNDS.copy(),
}
_A01_RESET_XY_FALLBACK = {
    "bucket": np.asarray([-0.15, 0.27], dtype=np.float64),
    "011_banana": np.asarray([-0.39, -0.29], dtype=np.float64),
    "apple": np.asarray([-0.17, -0.27], dtype=np.float64),
    "orange": np.asarray([-0.39, -0.06], dtype=np.float64),
    "pear": np.asarray([-0.13, 0.04], dtype=np.float64),
}
_A01_BUCKET_HAND_CLEARANCE = 0.16
_A01_FRUIT_HAND_CLEARANCE = 0.11
_A01_BUCKET_OBJECT_CLEARANCE = 0.13
_A01_FRUIT_OBJECT_CLEARANCE = 0.06
_A01_OBJECT_FOOTPRINT_RADII = {
    # Conservative XY footprint radii in meters. Banana is long and can rotate
    # through 360 deg, so center-distance clearance must account for its length.
    "bucket": 0.120,
    "011_banana": 0.105,
    "apple": 0.045,
    "orange": 0.045,
    "pear": 0.045,
}
_A01_OBJECT_CLEARANCE_MARGIN = 0.015
_A01_BUCKET_OBJECT_CLEARANCE_MARGIN = 0.020
_A01_OBJECT_HAND_CLEARANCE_MARGIN = 0.045
_A01_RESET_MIN_CLEARANCE_SLACK = 0.005
_A01_RESET_SAMPLE_ORDER = ("bucket", "011_banana", "apple", "orange", "pear")
_A01_RESET_CANDIDATES_PER_OBJECT = 96
_A01_RESET_MAX_ATTEMPTS = 64
_A01_YAW_BOUNDS_DEG = (-180.0, 180.0)
_A02_RESET_XY_BOUNDS = {
    "tray": RECORDING_OBJECT_XY_BOUNDS.copy(),
    "025_mug": RECORDING_OBJECT_XY_BOUNDS.copy(),
}
_A02_RESET_XY_FALLBACK = {
    "tray": np.asarray([-0.15, 0.26], dtype=np.float64),
    "025_mug": np.asarray([-0.39, -0.28], dtype=np.float64),
}
_A02_OBJECT_FOOTPRINT_RADII = {
    "tray": 0.135,
    "025_mug": 0.060,
}
_A02_OBJECT_CLEARANCE_MARGIN = 0.015
_A02_OBJECT_HAND_CLEARANCE_MARGIN = 0.045
_A02_RESET_MIN_CLEARANCE_SLACK = 0.005
_A02_RESET_SAMPLE_ORDER = ("tray", "025_mug")
_A02_RESET_CANDIDATES_PER_OBJECT = 96
_A02_RESET_MAX_ATTEMPTS = 64
_A02_YAW_BOUNDS_DEG = (-180.0, 180.0)
_A03_RESET_XY_BOUNDS = {
    "024_bowl": RECORDING_OBJECT_XY_BOUNDS.copy(),
    "tennis_ball": RECORDING_OBJECT_XY_BOUNDS.copy(),
    "golf_ball": RECORDING_OBJECT_XY_BOUNDS.copy(),
    "mini_soccer_ball": RECORDING_OBJECT_XY_BOUNDS.copy(),
}
_A03_RESET_XY_FALLBACK = {
    "024_bowl": np.asarray([-0.30, 0.24], dtype=np.float64),
    "tennis_ball": np.asarray([-0.38, -0.24], dtype=np.float64),
    "golf_ball": np.asarray([-0.22, -0.06], dtype=np.float64),
    "mini_soccer_ball": np.asarray([-0.38, 0.03], dtype=np.float64),
}
_A03_OBJECT_FOOTPRINT_RADII = {
    "024_bowl": 0.105,
    "tennis_ball": 0.034,
    "golf_ball": 0.022,
    "mini_soccer_ball": 0.038,
}
_A03_BOWL_BALL_CLEARANCE = 0.180
_A03_OBJECT_CLEARANCE_MARGIN = 0.025
_A03_OBJECT_HAND_CLEARANCE_MARGIN = 0.055
_A03_RESET_MIN_CLEARANCE_SLACK = 0.005
_A03_RESET_SAMPLE_ORDER = (
    "024_bowl",
    "tennis_ball",
    "golf_ball",
    "mini_soccer_ball",
)
_A03_RESET_CANDIDATES_PER_OBJECT = 128
_A03_RESET_MAX_ATTEMPTS = 96
_A03_YAW_BOUNDS_DEG = (-180.0, 180.0)
_B01_RESET_XY_BOUNDS = {
    "019_pitcher_base": RECORDING_OBJECT_XY_BOUNDS.copy(),
    "pitcher_lid": RECORDING_OBJECT_XY_BOUNDS.copy(),
}
_B01_RESET_XY_FALLBACK = {
    "019_pitcher_base": np.asarray([-0.15, 0.26], dtype=np.float64),
    "pitcher_lid": np.asarray([-0.39, -0.28], dtype=np.float64),
}
_B01_OBJECT_FOOTPRINT_RADII = {
    "019_pitcher_base": 0.115,
    "pitcher_lid": 0.090,
}
_B01_OBJECT_CLEARANCE_MARGIN = 0.015
_B01_OBJECT_HAND_CLEARANCE_MARGIN = 0.045
_B01_RESET_MIN_CLEARANCE_SLACK = 0.005
_B01_RESET_SAMPLE_ORDER = ("019_pitcher_base", "pitcher_lid")
_B01_RESET_CANDIDATES_PER_OBJECT = 96
_B01_RESET_MAX_ATTEMPTS = 64
_B01_YAW_BOUNDS_DEG = (-180.0, 180.0)
_TASK_SPECIFIC_RESET_XY_BOUNDS = {
    "t25_place_cup_on_plate": {
        "tray": _A02_RESET_XY_BOUNDS["tray"],
        "025_mug": _A02_RESET_XY_BOUNDS["025_mug"],
    },
    "t24_place_ball_in_bowl": {
        "024_bowl": _A03_RESET_XY_BOUNDS["024_bowl"],
        "tennis_ball": _A03_RESET_XY_BOUNDS["tennis_ball"],
        "golf_ball": _A03_RESET_XY_BOUNDS["golf_ball"],
        "mini_soccer_ball": _A03_RESET_XY_BOUNDS["mini_soccer_ball"],
    },
    "t19_orient_cup_upright": {
        "a_cup": np.asarray([[-0.38, -0.24], [-0.16, 0.18]], dtype=np.float64),
    },
    "t04_cover_pitcher": {
        "019_pitcher_base": _B01_RESET_XY_BOUNDS["019_pitcher_base"],
        "pitcher_lid": _B01_RESET_XY_BOUNDS["pitcher_lid"],
    },
    "b02_nut_bolt_assembly": {
        "bolt": np.asarray([[-0.34, 0.02], [-0.16, 0.24]], dtype=np.float64),
        "nut": np.asarray([[-0.40, -0.30], [-0.22, -0.08]], dtype=np.float64),
    },
    "t44_stack_blocks": {
        "036_wood_block": np.asarray([[-0.36, 0.04], [-0.18, 0.24]], dtype=np.float64),
        "BuildingBlock017": np.asarray([[-0.40, -0.30], [-0.24, -0.08]], dtype=np.float64),
        "BuildingBlock018": np.asarray([[-0.36, -0.30], [-0.18, -0.06]], dtype=np.float64),
        "BuildingBlock019": np.asarray([[-0.32, -0.26], [-0.14, 0.02]], dtype=np.float64),
    },
    "t03_cover_box": {
        "Clear_Box": np.asarray([[-0.30, 0.02], [-0.14, 0.24]], dtype=np.float64),
        "Box_lid": np.asarray([[-0.40, 0.02], [-0.26, 0.24]], dtype=np.float64),
    },
    "t36_scoop_marbles": {
        "024_bowl": np.asarray([[-0.34, 0.08], [-0.16, 0.26]], dtype=np.float64),
        "spoon": np.asarray([[-0.40, -0.26], [-0.24, -0.02]], dtype=np.float64),
        "golf_ball": np.asarray([[-0.40, -0.08], [-0.24, 0.12]], dtype=np.float64),
    },
}
_TASK_SPECIFIC_RESET_DEFAULT_PAIR_DISTANCE = 0.085
_TASK_SPECIFIC_RESET_PAIR_DISTANCE = {
    ("t25_place_cup_on_plate", frozenset(("025_mug", "tray"))): 0.20,
    ("t24_place_ball_in_bowl", frozenset(("024_bowl", "tennis_ball"))): 0.18,
    ("t24_place_ball_in_bowl", frozenset(("024_bowl", "golf_ball"))): 0.18,
    ("t24_place_ball_in_bowl", frozenset(("024_bowl", "mini_soccer_ball"))): 0.18,
    ("t24_place_ball_in_bowl", frozenset(("tennis_ball", "golf_ball"))): 0.09,
    ("t24_place_ball_in_bowl", frozenset(("tennis_ball", "mini_soccer_ball"))): 0.105,
    ("t24_place_ball_in_bowl", frozenset(("golf_ball", "mini_soccer_ball"))): 0.095,
    ("t04_cover_pitcher", frozenset(("019_pitcher_base", "pitcher_lid"))): 0.22,
    ("b02_nut_bolt_assembly", frozenset(("bolt", "nut"))): 0.18,
    ("t03_cover_box", frozenset(("Clear_Box", "Box_lid"))): 0.15,
    ("t36_scoop_marbles", frozenset(("024_bowl", "spoon"))): 0.18,
    ("t36_scoop_marbles", frozenset(("024_bowl", "golf_ball"))): 0.14,
}
_NO_RANDOMIZED_YAW_OBJECTS = {
    "a_marbles",
    "spoon",
    "tennis_ball",
    "golf_ball",
    "mini_soccer_ball",
    "Clear_Box",
    "Box_lid",
    "c08_tongs",
}
_RECORDING_CAMERA_NAMES = (
    "head_camera",
    "wrist_left_camera",
    "wrist_right_camera",
)
_RECORDING_CAMERA_PRIM_PATHS = {
    name: f"/World/RecordCameras/{name}"
    for name in _RECORDING_CAMERA_NAMES
}
_HEAD_RECORDING_CAMERA_EYE = HEAD_RECORDING_CAMERA_EYE
_HEAD_RECORDING_CAMERA_QUAT_WXYZ = HEAD_RECORDING_CAMERA_QUAT_WXYZ_OPENGL
_HEAD_RECORDING_CAMERA_FOVY = HEAD_RECORDING_CAMERA_FOVY_DEG
_WRIST_RECORDING_CAMERA_HFOV = WRIST_RECORDING_CAMERA_HFOV_DEG
_WRIST_RECORDING_CAMERA_FOVY = math.degrees(
    2.0
    * math.atan(
        math.tan(math.radians(_WRIST_RECORDING_CAMERA_HFOV) / 2.0)
        * WRIST_RECORDING_CAMERA_OUTPUT_SIZE[1]
        / WRIST_RECORDING_CAMERA_OUTPUT_SIZE[0]
    )
)
_RECORDING_CAMERA_VERTICAL_APERTURE = RECORDING_CAMERA_VERTICAL_APERTURE
_WRIST_RECORDING_CAMERA_LOCAL_POSES = WRIST_RECORDING_CAMERA_LOCAL_POSES
_WRIST_RECORDING_CAMERA_MOUNT_POSES = WRIST_RECORDING_CAMERA_MOUNT_POSES
_LEGACY_WRIST_RECORDING_CAMERA_MOUNT_POSES = {
    "left": np.asarray(
        [
            0.0,
            0.0100032829,
            0.0282831105,
            0.0524277738,
            0.3054651071,
            0.9370572242,
            -0.1608295853,
        ],
        dtype=np.float64,
    ),
    "right": np.asarray(
        [
            0.0,
            0.0100032829,
            0.0282831105,
            0.0524423546,
            0.3055500602,
            -0.9370295266,
            0.1608248315,
        ],
        dtype=np.float64,
    ),
}
_WRIST_RECORDING_CAMERA_MOUNT_CLEARANCE = float(
    np.linalg.norm(_WRIST_RECORDING_CAMERA_MOUNT_POSES["left"][:3])
)
_CAMERA_DEBUG_PRIM_NAMES = {
    "head_main_optical_site",
    "left_wrist_fisheye_optical_site",
    "right_wrist_fisheye_optical_site",
}
_OBJECT_HALF_HEIGHTS = {
    "011_banana": 0.020,
    "apple": 0.038,
    "orange": 0.038,
    "pear": 0.052,
    "025_mug": 0.043,
    "a_cup": 0.031,
    "pitcher_lid": 0.024,
    "Box_lid": 0.010,
    "036_wood_block": 0.042689,
    "BuildingBlock017": 0.014925,
    "BuildingBlock018": 0.015,
    "BuildingBlock019": 0.015,
}
_A06_CUP_OPENING_UP_MAX_ANGLE_DEG = 25.0
_A06_CUP_TABLE_HEIGHT_TOLERANCE = 0.040
_A06_HAND_CONTACT_DISTANCE = 0.020
_A06_RELEASE_HOLD_SECONDS = 0.50
_STACK_BLOCK_HALF_EXTENTS = {
    "036_wood_block": np.asarray([0.0433705, 0.100369, 0.042689], dtype=np.float64),
    "BuildingBlock017": np.asarray([0.014925, 0.014925, 0.014925], dtype=np.float64),
    "BuildingBlock018": np.asarray([0.015000, 0.015000, 0.015000], dtype=np.float64),
    "BuildingBlock019": np.asarray([0.015000, 0.015000, 0.015000], dtype=np.float64),
}
_STACK_ORIENTATION_FREE_BLOCKS = frozenset(
    {"BuildingBlock017", "BuildingBlock018", "BuildingBlock019"}
)
_STACK_XY_ALIGNMENT_TOLERANCE = 0.060
_STACK_Z_GAP_TOLERANCE = 0.045
_STACK_HEIGHT_TOLERANCE = 0.050
_STACK_BOTTOM_BLOCK_NAME = "036_wood_block"
_STACK_MIN_LAYER_GAP = 0.020
_STACK_LAYER_GAP_FRACTION = 0.50
_STACK_UPRIGHT_ALIGNMENT_THRESHOLD = float(np.cos(np.deg2rad(55.0)))
_STACK_SUPPORT_Z_TOLERANCE = 0.012
_STACK_SUPPORT_MIN_XY_OVERLAP = 0.003
_STACK_RELEASE_STABLE_SECONDS = 0.50
_STACK_STABLE_LINEAR_SPEED = 0.025
_STACK_STABLE_ANGULAR_SPEED = 0.35
_STACK_VELOCITY_FILTER_WINDOW = 5
_STACK_STABLE_EXIT_LINEAR_SPEED = 0.040
_STACK_STABLE_EXIT_ANGULAR_SPEED = 0.55
_STACK_HARD_MOTION_LINEAR_SPEED = 0.10
_STACK_HARD_MOTION_ANGULAR_SPEED = 1.40
_STACK_HAND_CONTACT_DISTANCE = 0.015
_STACK_HAND_CONTACT_FORCE_THRESHOLD = 1.0e-3
_STACK_BASE_STATIC_FRICTION = 1.4
_STACK_BASE_DYNAMIC_FRICTION = 1.4
_STACK_SMALL_STATIC_FRICTION = 4.0
_STACK_SMALL_DYNAMIC_FRICTION = 4.0
_STACK_LINEAR_DAMPING = 0.25
_STACK_ANGULAR_DAMPING = 1.00
_STACK_CONTACT_OFFSET = 0.001
_STACK_REST_OFFSET = 0.0
_STACK_SLEEP_THRESHOLD = 0.010
_C12_RESET_LINEAR_DAMPING = 0.40
_C12_RESET_ANGULAR_DAMPING = 2.00
_C12_RESET_SLEEP_THRESHOLD = 0.020
# A01 begins with rounded primitives plus compound banana/bucket hulls in light
# table contact.  Match the stable A06 damping envelope and give every compound
# collider the same narrow contact shell so neighboring hull pieces cannot
# alternate between different separation distances after a reset.
_A01_CONTACT_OFFSET = 0.001
_A01_REST_OFFSET = 0.0
_A01_RESET_LINEAR_DAMPING = 0.40
_A01_RESET_ANGULAR_DAMPING = 2.00
_A01_RESET_SLEEP_THRESHOLD = 0.010
_A01_RESET_STABILITY_BODIES = ("apple", "011_banana", "orange", "pear", "bucket")
# A03 uses analytic source sphere colliders.  Modest physical damping models
# rolling resistance so PhysX can put the balls to sleep naturally; no state is
# clamped during stabilization or collection.
_A03_RESET_LINEAR_DAMPING = 0.12
_A03_RESET_ANGULAR_DAMPING = 0.80
_A03_RESET_SLEEP_THRESHOLD = 0.005
_A03_RESET_STABILITY_BODIES = (
    "tennis_ball",
    "golf_ball",
    "mini_soccer_ball",
)
# A06's thin cup collider otherwise keeps a small PhysX contact spin after it
# is placed on the table.  Damping represents table-contact rolling resistance
# and remains ordinary wakeable rigid-body physics during manipulation.
_A06_SOURCE_MASS = 0.0599999994
_A06_SOURCE_CENTER_OF_MASS = (0.00013567, 0.00011084, -0.01130477)
_A06_SOURCE_DIAGONAL_INERTIA = (
    3.32982077e-5,
    3.28791113e-5,
    2.36141326e-5,
)
_A06_SOURCE_PRINCIPAL_AXES_WXYZ = (
    0.920333713,
    -0.000063751,
    0.000525214,
    0.391133707,
)
_A06_CONTACT_OFFSET = 0.001
_A06_REST_OFFSET = 0.0
_A06_RESET_LINEAR_DAMPING = 0.40
_A06_RESET_ANGULAR_DAMPING = 2.00
_A06_RESET_SLEEP_THRESHOLD = 0.010
_A06_RESET_STABILITY_BODIES = ("a_cup",)
_STACK_GEOMETRY_METRIC_KEYS = (
    "stack_four_layers",
    "stack_large_block_on_bottom",
    "stack_support_graph_valid",
    "stack_upright",
    "stack_bottom_on_table",
)
_B04_SUCCESS_CONDITIONS = (
    ("stack_four_layers", "四个积木形成四层"),
    ("stack_large_block_on_bottom", "大木块在最底层"),
    ("stack_support_graph_valid", "每层积木都由下一层支撑"),
    ("stack_upright", "积木没有明显歪倒"),
    ("stack_bottom_on_table", "底层积木在桌面上"),
    ("stack_hand_released", "右手已与所有积木脱离接触"),
    ("stack_velocity_stable", "积木堆已经稳定"),
)
_LID_GOAL_LOCAL_Z = {
    "t03_cover_box": 0.0845,
}


@dataclass(frozen=True)
class _LidGraspPhysicsCfg:
    mass: float
    static_friction: float
    dynamic_friction: float
    friction_combine_mode: Literal["average", "min", "multiply", "max"]
    contact_offset: float | None = None
    rest_offset: float | None = None


_LID_GRASP_PHYSICS = {
    # The Wuji fingertips use high-friction rubber.  Capping the pair at the
    # lid's friction and restoring the source MJCF's 60 g mass lets B01 remain
    # graspable without following an open hand after it is placed.
    "t04_cover_pitcher": _LidGraspPhysicsCfg(
        mass=0.060,
        static_friction=1.2,
        dynamic_friction=0.8,
        friction_combine_mode="min",
        contact_offset=0.001,
        rest_offset=0.0,
    ),
    # B05 has a different thin, flat grasp geometry and still needs the
    # stronger material that was tuned for that task.
    "t03_cover_box": _LidGraspPhysicsCfg(
        mass=0.040,
        static_friction=2.5,
        dynamic_friction=2.0,
        friction_combine_mode="max",
    ),
}


@dataclass(frozen=True)
class _B02NutGraspPhysicsCfg:
    mass: float
    static_friction: float
    dynamic_friction: float
    hand_static_friction: float
    hand_dynamic_friction: float
    friction_combine_mode: Literal["average", "min", "multiply", "max"]
    contact_offset: float
    rest_offset: float
    disable_inner_finger_guard: bool


# The MJCF importer inserts a 1e-6 kg bound-mass inertial on nut_root_body.
# That overrides the six collider densities, while the native MuJoCo task
# explicitly raises the completed nut body to 40 g.  Keep B02's Isaac material
# values source-matched for this debug path: the nut collision mesh has sliding
# friction 0.42 and the Wuji fingertip meshes have sliding friction 3.0.
# PhysX has static/dynamic sliding coefficients rather than MuJoCo's
# sliding/torsional/rolling tuple, so both PhysX coefficients use the source
# sliding value.  The source-only inner guard relies on MuJoCo collision bits
# that do not target the separately spawned PhysX hand, so it remains disabled
# here instead of becoming a solid plug inside the nut.
_B02_NUT_GRASP_PHYSICS = _B02NutGraspPhysicsCfg(
    mass=0.040,
    static_friction=0.42,
    dynamic_friction=0.42,
    hand_static_friction=3.0,
    hand_dynamic_friction=3.0,
    friction_combine_mode="max",
    contact_offset=0.0010,
    rest_offset=0.0,
    disable_inner_finger_guard=True,
)
_B02_RESET_LINEAR_DAMPING = 0.25
_B02_RESET_ANGULAR_DAMPING = 2.00
_B02_RESET_SLEEP_THRESHOLD = 0.010

_B02_NUT_OUTER_DIAMETER = 0.017542
_B02_NUT_HALF_HEIGHT = 0.004074
_B02_WUJI_FINGERTIP_RADIUS = 0.0085
_B02_GRASP_ATTACH_MIN_FINGER_DISTANCE = 0.55 * _B02_NUT_OUTER_DIAMETER
_B02_GRASP_ATTACH_MAX_FINGER_DISTANCE = (
    _B02_NUT_OUTER_DIAMETER + 2.0 * _B02_WUJI_FINGERTIP_RADIUS + 0.002
)
_B02_GRASP_RELEASE_FINGER_DISTANCE = (
    _B02_NUT_OUTER_DIAMETER + 2.0 * _B02_WUJI_FINGERTIP_RADIUS + 0.006
)
_B02_GRASP_ATTACH_HORIZONTAL_RADIUS = 0.026
_B02_GRASP_ATTACH_PINCH_CENTER_RADIUS = 0.014
_B02_GRASP_ATTACH_Z_ABOVE_CENTER = _B02_NUT_HALF_HEIGHT + 0.004
_B02_GRASP_ATTACH_Z_BELOW_CENTER = 0.025
_B02_GRASP_ASSIST_MAX_LINEAR_SPEED = 1.0
_B02_WUJI_FINGERTIP_LOCAL_OFFSETS = {
    "right_finger1_tip": (
        "right_finger1_link4",
        np.asarray([-0.00105, -0.0002, 0.0313], dtype=np.float64),
    ),
    "right_finger2_tip": (
        "right_finger2_link4",
        np.asarray([-0.00105, 0.0002, 0.0267], dtype=np.float64),
    ),
    "right_finger3_tip": (
        "right_finger3_link4",
        np.asarray([-0.00104909, 0.00020104, 0.0267], dtype=np.float64),
    ),
    "right_finger4_tip": (
        "right_finger4_link4",
        np.asarray([-0.00105961, 0.000190451, 0.0267], dtype=np.float64),
    ),
    "right_finger5_tip": (
        "right_finger5_link4",
        np.asarray([-0.00104464, 0.00021131, 0.0267], dtype=np.float64),
    ),
}
_B02_WUJI_LEFT_FINGERTIP_LOCAL_OFFSETS = {
    "left_finger1_tip": (
        "left_finger1_link4",
        np.asarray([-0.00105, 0.0002, 0.0313], dtype=np.float64),
    ),
    "left_finger2_tip": (
        "left_finger2_link4",
        np.asarray([-0.0010521, 0.000210869, 0.0267], dtype=np.float64),
    ),
    "left_finger3_tip": (
        "left_finger3_link4",
        np.asarray([-0.0010504, 0.000202672, 0.0267], dtype=np.float64),
    ),
    "left_finger4_tip": (
        "left_finger4_link4",
        np.asarray([-0.00104998, 0.000202557, 0.0267], dtype=np.float64),
    ),
    "left_finger5_tip": (
        "left_finger5_link4",
        np.asarray([-0.00105041, 0.000201047, 0.0267], dtype=np.float64),
    ),
}
_B02_WUJI_THUMB_TIP = "right_finger1_tip"
_B02_WUJI_OPPOSING_TIPS = (
    "right_finger2_tip",
    "right_finger3_tip",
    "right_finger4_tip",
    "right_finger5_tip",
)


@dataclass(frozen=True)
class _RigidObjectStabilityCfg:
    mass: float
    static_friction: float
    dynamic_friction: float
    linear_damping: float
    angular_damping: float
    contact_offset: float | None = None
    rest_offset: float | None = None


# The source MJCF assigns the empty pitcher base about 180 g.  The converted
# USD loses that explicit mass and leaves PhysX to infer it from the collision
# meshes.  Use a modest weighted-base equivalent so the single active hand can
# seat the lid without the tall receptacle skittering or rocking excessively.
_B01_PITCHER_BASE_PHYSICS = _RigidObjectStabilityCfg(
    mass=0.350,
    static_friction=1.4,
    dynamic_friction=1.0,
    linear_damping=0.2,
    angular_damping=1.0,
    contact_offset=0.001,
    rest_offset=0.0,
)
_C06_CAMERA_GRASP_PHYSICS = _RigidObjectStabilityCfg(
    mass=0.140,
    static_friction=1.6,
    dynamic_friction=1.2,
    linear_damping=0.40,
    angular_damping=2.00,
    contact_offset=0.001,
    rest_offset=0.0,
)
_B01_PITCHER_LID_SEAT_LOCAL_Z = 0.1176775
_B01_PITCHER_MOUTH_INNER_RADIUS = 0.052
_B01_LID_INSERT_RADIUS = 0.043
_B01_LID_INSERT_HALF_HEIGHT = 0.010
_B01_LID_INSERT_CENTER_LOCAL_Z = -0.0135045
_B01_LID_INSERT_TOP_LOCAL_Z = (
    _B01_LID_INSERT_CENTER_LOCAL_Z + _B01_LID_INSERT_HALF_HEIGHT
)
_B01_LID_INSERT_BOTTOM_LOCAL_Z = (
    _B01_LID_INSERT_CENTER_LOCAL_Z - _B01_LID_INSERT_HALF_HEIGHT
)
_B01_LID_INSERTED_GOAL_LOCAL_Z = (
    _B01_PITCHER_LID_SEAT_LOCAL_Z - _B01_LID_INSERT_TOP_LOCAL_Z
)
_B01_LID_INSERT_XY_TOLERANCE = 0.002
_B01_LID_INSERT_Z_BELOW_TOLERANCE = 0.012
_B01_LID_INSERT_Z_ABOVE_TOLERANCE = 0.006
_B01_LID_SUCCESS_TOP_ANGLE_DEG = 12.0
_B01_HAND_LID_CLEAR_HOLD_SECONDS = 0.50
_B01_HAND_LID_CONTACT_DISTANCE = 0.018
_B01_HAND_LID_CLEARANCE_RADIUS = 0.070
_B01_HAND_LID_CLEARANCE_HALF_HEIGHT = 0.030
_B01_RELEASE_ZONE_XY_TOLERANCE = 0.045
_B01_RELEASE_ZONE_Z_BELOW_TOLERANCE = 0.030
_B01_RELEASE_ZONE_Z_ABOVE_TOLERANCE = 0.050
_B01_RELEASE_ZONE_TOP_ANGLE_DEG = 25.0
_B01_OPEN_HAND_MAX_JOINT = 0.85
_B01_CLOSED_HAND_MIN_JOINT = 1.00
_B01_RIGHT_ROBOT_ARTICULATION_PATH = (
    "/World/HandsOnRobots/RightRobot/base_right/base_right"
)
_LID_MIN_MOVE_DISTANCE = 0.050
_LID_MIN_LIFT_HEIGHT = 0.015
_LID_STABLE_LINEAR_SPEED = 0.025
_LID_STABLE_ANGULAR_SPEED = 0.30
_B02_TARGET_DEPTH = 0.032
_B02_DEPTH_TOLERANCE = 0.014
_B02_MAX_RADIAL_ERROR = 0.0065
_B02_MIN_AXIS_ALIGNMENT = float(np.cos(np.deg2rad(18.0)))
_B02_MIN_TWIST_PROGRESS_DEG = 40.0
_B02_MIN_AXIAL_TRAVEL = 0.007
_B02_THREAD_ENGAGE_RADIAL_ERROR = 0.0040
_B02_THREAD_ENGAGE_AXIS_ALIGNMENT = float(np.cos(np.deg2rad(15.0)))
_B02_THREAD_ACTIVE_RADIAL_ERROR = 0.010
_B02_THREAD_ACTIVE_AXIS_ALIGNMENT = float(np.cos(np.deg2rad(28.0)))
_B02_STABLE_LINEAR_SPEED = 0.040
_B02_STABLE_ANGULAR_SPEED = 0.65
_B02_DEFAULT_PHYSICS_HZ = 120.0
_B02_DEFAULT_PHYSICS_DT = 1.0 / _B02_DEFAULT_PHYSICS_HZ
_B02_COLLECTION_SAFE_PHYSICS_HZ = 250.0
_B02_COLLECTION_SAFE_PHYSICS_DT = 1.0 / _B02_COLLECTION_SAFE_PHYSICS_HZ
_B02_MUJOCO_PHYSICS_HZ = 500.0
_B02_MUJOCO_PHYSICS_DT = 1.0 / _B02_MUJOCO_PHYSICS_HZ
_B02_MUJOCO_HAND_JOINT_DRIVES = {
    "finger1_joint1": (0.40844710645048043, 0.020882010257063675, 0.8904),
    "finger1_joint2": (0.6858601063643346, 0.030610939996373314, 0.8518),
    "finger1_joint3": (0.2391112196099482, 0.010181475050560962, 0.3776),
    "finger1_joint4": (0.20736128319711747, 0.00909698844675045, 0.2936),
    "finger2_joint1": (0.37352218155860073, 0.01882274330029718, 1.2376),
    "finger2_joint2": (0.45592448909027794, 0.019798167597016643, 0.3644),
    "finger2_joint3": (0.24368366649522863, 0.010477031953162727, 0.4502),
    "finger2_joint4": (0.18026971340925335, 0.008240212147903584, 0.4340),
    "finger3_joint1": (0.3687093483646485, 0.01848622487024593, 1.2988),
    "finger3_joint2": (0.4164253443634641, 0.018032947229953678, 0.3654),
    "finger3_joint3": (0.22218607502059182, 0.009592200014076666, 0.4156),
    "finger3_joint4": (0.19427606072023446, 0.009152994605972402, 0.4036),
    "finger4_joint1": (0.35718151495111794, 0.018376606800780005, 1.2778),
    "finger4_joint2": (0.42977315313086895, 0.01867700966212433, 0.3664),
    "finger4_joint3": (0.24930151196247122, 0.01059512121059555, 0.4498),
    "finger4_joint4": (0.2285032688178066, 0.009917602877441107, 0.4088),
    "finger5_joint1": (0.3655325975433942, 0.018616960278988272, 1.2882),
    "finger5_joint2": (0.41393113081120425, 0.018732177029667153, 0.3596),
    "finger5_joint3": (0.22729367621965954, 0.00951005441616486, 0.4768),
    "finger5_joint4": (0.1964723550341816, 0.009017295241756363, 0.3732),
}
_B03_PEG_COLLISION_LOCAL_POS = np.asarray([0.0, 0.0, 0.081], dtype=np.float64)
_B03_PEG_SHAFT_HALF_LENGTH = 0.0675
_B03_SOCKET_SITE_LOCAL_POS = np.asarray([0.0, 0.0, 0.027], dtype=np.float64)
_B03_INSERTION_RADIAL_TOLERANCE = 0.032
_B03_INSERTION_MIN_DEPTH = 0.002
_B03_AXIS_ALIGNMENT_THRESHOLD = float(np.cos(np.deg2rad(45.0)))
# B02 permits 0.040 m/s and 0.65 rad/s because the nut may still be turning
# while its thread seats.  B03 has no such intended motion at completion, so
# require both free bodies to remain below roughly half those limits for a
# full half-second before accepting an insertion.
_B03_STABLE_LINEAR_SPEED = 0.020
_B03_STABLE_ANGULAR_SPEED = 0.30
_B03_STABLE_HOLD_SECONDS = 0.50
_B03_RESET_PEG_MIN_LOCAL_Z = 0.0135
_B03_RESET_PEG_MAX_LOCAL_Z = 0.2385
_B03_RESET_SOCKET_SEGMENT_CLEARANCE = 0.150
_B03_RESET_RIGHT_HAND_CLEARANCE = 0.155
_B03_RESET_SOCKET_HAND_CLEARANCE = 0.235
_B03_RESET_MIN_CLEARANCE_SLACK = 0.005
_B03_RESET_CANDIDATES_PER_OBJECT = 128
_B03_RESET_MAX_ATTEMPTS = 64
_B03_RESET_RIGHT_HAND_XY = np.asarray(_RIGHT_HOME[:2], dtype=np.float64)
_B03_RESET_LEFT_HAND_XY = np.asarray(_LEFT_HOME[:2], dtype=np.float64)
_B03_HAND_MIDPOINT_XY = 0.5 * (
    np.asarray(_RIGHT_HOME[:2], dtype=np.float64)
    + np.asarray(_LEFT_HOME[:2], dtype=np.float64)
)
_B03_RESET_RIGHT_HAND_AVOID_XY = np.asarray(
    [
        _RIGHT_HOME[:2],
        np.asarray(_RIGHT_HOME[:2], dtype=np.float64) + np.asarray([-0.060, -0.020]),
        np.asarray(_RIGHT_HOME[:2], dtype=np.float64) + np.asarray([-0.035, -0.060]),
        np.asarray(_RIGHT_HOME[:2], dtype=np.float64) + np.asarray([0.030, -0.045]),
    ],
    dtype=np.float64,
)
_B03_RESET_LEFT_HAND_AVOID_XY = np.asarray(
    [
        _LEFT_HOME[:2],
        np.asarray(_LEFT_HOME[:2], dtype=np.float64) + np.asarray([-0.060, 0.020]),
        np.asarray(_LEFT_HOME[:2], dtype=np.float64) + np.asarray([-0.035, 0.060]),
        np.asarray(_LEFT_HOME[:2], dtype=np.float64) + np.asarray([0.030, 0.045]),
    ],
    dtype=np.float64,
)
_B03_RESET_SOCKET_HAND_AVOID_XY = np.concatenate(
    (_B03_RESET_RIGHT_HAND_AVOID_XY,),
    axis=0,
)
_B03_RESET_FALLBACK_DIRECTION = np.asarray([-0.55, -0.83], dtype=np.float64)
_B03_ROKAE_PEG_XY_BOUNDS = RECORDING_OBJECT_XY_BOUNDS.copy()
_B03_ROKAE_SOCKET_XY_BOUNDS = RECORDING_OBJECT_XY_BOUNDS.copy()
_B03_DEFAULT_RESET_XY = {
    "peg": np.asarray([-0.30, -0.42], dtype=np.float64),
    "socket": _B03_HAND_MIDPOINT_XY.copy(),
}
_B03_PEG_YAW_BOUNDS_DEG = (-10.0, 10.0)
_B03_SOCKET_YAW_BOUNDS_DEG = (-15.0, 15.0)
_SUCCESS_CONDITION_PANEL_TITLE = "Success Conditions"
_SUCCESS_CONDITION_OK_TEXT = "OK"
_SUCCESS_CONDITION_DISPLAY_LABELS = {
    "a01_target_inside_bucket_xy": "Target inside bucket footprint",
    "a01_target_on_bucket_bottom": "Target resting at bucket bottom",
    "a02_target_is_mug": "Target object is the brown mug",
    "a02_centered_on_tray": "Mug centered on tray",
    "a02_height_on_tray": "Mug bottom at tray height",
    "a02_bottom_down": "Mug upright",
    "a02_not_dropped": "Mug not dropped below table",
    "a02_stable": "Mug stable",
    "a03_ball_in_bowl_xy": "Ball inside bowl opening",
    "a03_ball_in_bowl_z": "Ball height inside bowl",
    "a06_opening_up": "Cup opening faces up",
    "a06_on_table": "Cup at table height",
    "a06_hand_clear": "Hand clear of cup",
    "b01_insert_inside": "Lid plug fully inside cup mouth",
    "b01_z_close": "Lid height fully seated",
    "b01_top_aligned": "Lid top is level",
    "b01_moved": "Lid moved onto cup",
    "b01_stable": "Lid stable",
    "b01_hand_clear": "Hand clear of lid",
    "b01_hand_clear_held": "Hand clear for 0.5 sec",
    "b02_radial_aligned": "Nut centered on bolt",
    "b02_depth_ok": "Nut near target depth",
    "b02_axis_aligned": "Nut axis aligned to bolt",
    "b02_thread_seen": "Thread engagement seen",
    "b02_twisted": "Nut has clear twist progress",
    "b02_axial_travel": "Nut traveled down bolt",
    "b02_stable": "Nut stable",
    "b03_radial_aligned": "Peg shaft aligned at socket mouth",
    "b03_inserted": "Peg inserted into socket",
    "b03_axis_aligned": "Peg axis aligned to socket",
    "b03_tip_end_correct": "Narrow peg tip points into socket",
    "b03_peg_stable": "Peg motion is stable",
    "b03_socket_stable": "Socket motion is stable",
    "b03_stable_hold": "Insertion and stability held for 0.5 sec",
    "b04_four_layers": "Four block layers formed",
    "b04_large_bottom": "Large block at bottom",
    "b04_xy_aligned": "Block layers aligned in XY",
    "b04_z_gaps_ok": "Layer heights are valid",
    "b04_upright": "Blocks are upright",
    "b04_bottom_on_table": "Bottom block on table",
    "b04_hand_released": "Hand clear of blocks",
    "b04_velocity_stable": "Block stack stable",
    "b05_centered": "Box lid centered on box",
    "b05_height": "Box lid near seated height",
    "b05_top_aligned": "Box lid top is level",
    "b05_yaw_aligned": "Box lid yaw aligned",
    "b05_moved": "Box lid moved onto box",
    "b05_stable": "Box lid stable",
    "c01_valid_hit": "Hammer has hit nail",
    "c01_depth": "Nail driven deep enough",
    "c01_wood_lift": "Wood lifted by nail",
    "c02_spoon_used": "Spoon picked up",
    "c02_spoon_loaded": "Spoon loaded marbles",
    "c02_over_bowl": "Loaded spoon over bowl",
    "c02_marble_in_bowl": "Marble dropped into bowl",
    "c04_spray_grasped": "Spray bottle held by hand",
    "c04_trigger_contact": "Trigger pressed by hand",
    "c04_water_emitting": "Trigger held and water spraying",
    "c04_water_hits_target": "Water hits plant or soil",
    "c05_mouse_moved": "Mouse moved from start",
    "c05_mouse_on_pad": "Mouse is on mouse pad",
    "c05_no_early_click": "No button pressed while moving",
    "c05_left_clicked": "Left button clicked on pad",
    "c06_camera_aimed": "Camera aimed at logo",
    "c06_right_hand_contact": "Right hand touching shutter",
    "c06_shutter_pressed": "Shutter button pressed",
    "c07_enough_inside": "At least 80% poured into cup",
    "c07_spill_limited": "Spill count within limit",
    "c07_tilt_seen": "Pouring mug was tilted to pour",
    "c07_pitcher_centered_on_pad": "Pouring mug fully centered on return pad",
    "c07_pitcher_on_pad_height": "Pouring mug bottom resting on return pad",
    "c07_pitcher_stable": "Pouring mug stopped moving",
    "c07_hand_released": "Hand fully clear of pouring mug",
    "c07_pitcher_returned": "Pouring mug upright and stably returned",
    "c07_release_hold": "Returned and released for 0.5 sec",
    "c08_all_ice_inside": "All ice cubes in cup",
    "c08_tongs_cycles": "Tongs closed four times",
    "c08_cup_upright": "Glass remains upright",
    "c09_papers_overlap": "Paper sheets aligned",
    "c09_papers_in_zone": "Papers inserted into stapler zone",
    "c09_stapler_opened": "Stapler top opened",
    "c09_stapler_pressed": "Stapler pressed closed",
    "c10_first_connected": "Charger plugged into strip first",
    "c10_second_connected": "Cable plugged into charger",
    "c10_order_valid": "Two-stage order respected",
    "c11_sequence_complete": "Xylophone sequence complete",
    "c11_wrong_hits_ok": "Wrong hit resets sequence",
    "c11_mallet_returned": "Mallet returned",
    "c12_blocks_on_line": "Blocks on baseline",
    "c12_blocks_aligned": "Blocks aligned with baseline",
    "c12_spacing_ok": "Block spacing valid",
    "c12_tool_used": "Only tool push was used",
    "c12_no_direct_grasp": "No direct block grasp",
    "event_success_unavailable": "Task uses event trigger",
    "condition_panel_error": "Condition panel error",
    "success_hold": "Hold all conditions stable",
}
_RAISED_ARM_HOME = {
    # IK-derived reset homes.  Compared with the shared tabletop home, the wrist
    # is about 55 cm higher in the ROKAE kinematic world, so tall objects cannot
    # be depenetrated by either hand on the first PhysX frame.
    "right": np.asarray(
        [0.441741, 0.721729, 0.239304, 1.982058, -0.346778, -0.925025, 0.925025],
        dtype=np.float64,
    ),
    "left": np.asarray(
        [-0.441741, 0.721729, -0.239304, 1.982058, 0.346778, -0.925025, -0.925025],
        dtype=np.float64,
    ),
}
_RAISED_LEFT_START_TCP = _LEFT_HOME.copy()
_RAISED_LEFT_START_TCP[2] += 0.55
_RAISED_INACTIVE_LEFT_TASKS = frozenset(
    {
        "t26_place_fruit_in_bucket",
        "t25_place_cup_on_plate",
        "t04_cover_pitcher",
        "t09_insert_peg",
        "t44_stack_blocks",
        "t03_cover_box",
        "t50_water_plant",
    }
)
_C01_SUCCESS_DEPTH = 0.006
_C01_MIN_DEPTH_INCREMENT = 0.00025
_C01_MIN_HAMMER_SPEED = 0.20
_C01_HAMMER_NAIL_CONTACT_FORCE = 0.10
_C01_MAX_FACE_NAIL_DISTANCE = 0.018
_C01_MAX_FACE_NAIL_RADIAL_ERROR = 0.012
_C01_MIN_FACE_NAIL_AXIAL_OFFSET = -0.003
_C01_MAX_FACE_NAIL_AXIAL_OFFSET = 0.008
_C01_MIN_FACE_DOWN_ALIGNMENT = float(np.cos(np.deg2rad(25.0)))
_C01_MIN_MOTION_DOWN_ALIGNMENT = float(np.cos(np.deg2rad(30.0)))
_C01_HAMMER_FACE_LOCAL_TIP = np.asarray(
    [0.059062, 0.0, 0.1730],
    dtype=np.float64,
)
_C01_NAIL_HEAD_LOCAL_TOP = np.asarray(
    [0.0, 0.0, 0.0084],
    dtype=np.float64,
)
# The nail is an impact-driven prismatic joint.  PhysX joint friction is a
# dimensionless coefficient rather than a force in newtons, so it cannot be
# used as a calibrated gravity hold.  Keep exactly one passive velocity brake:
# its 0.20 N force limit is only slightly above the 0.015 kg nail's 0.147 N
# weight.  Zero position stiffness means it never pulls an inserted nail out.
_C01_HAMMER_MASS = 0.295
_C01_NAIL_MASS = 0.015
_C01_NAIL_JOINT_FRICTION = 0.0
_C01_NAIL_JOINT_STIFFNESS = 0.0
_C01_NAIL_JOINT_DAMPING = 100000.0
_C01_NAIL_JOINT_MAX_FORCE = 0.20
_C01_ENABLE_CCD = True
_C01_ENABLE_SPECULATIVE_CCD = False
_C01_HAMMER_CONTACT_OFFSET = 0.00075
_C01_NAIL_CONTACT_OFFSET = 0.00075
_C01_CONTACT_REST_OFFSET = 0.0
_C01_SOLVER_POSITION_ITERATIONS = 32
_C01_SOLVER_VELOCITY_ITERATIONS = 4
_C01_MAX_DEPENETRATION_VELOCITY = 1.0
_C01_WOOD_HALF_HEIGHT = 0.0225
_C01_WOOD_MASS = 0.085
_C01_WOOD_LIFT_CLEARANCE = 0.030
_C01_HAMMER_GUIDE_MAX_RAY_LENGTH = 0.35
_C01_HAMMER_GUIDE_DEFAULT_RAY_LENGTH = 0.18
_C01_GRASP_ATTACH_HAND_COMMAND = 0.82
_C01_GRASP_RELEASE_HAND_COMMAND = 0.70
_C01_HAMMER_HANDLE_SEGMENT = np.asarray(
    [[0.0018, 0.0005, -0.125], [0.0030, 0.0000, 0.120]],
    dtype=np.float64,
)
_C01_HAMMER_GRIP_SEGMENT = np.asarray(
    [[0.0018, 0.0005, -0.110], [0.0026, 0.00015, 0.055]],
    dtype=np.float64,
)
_C01_HAMMER_HANDLE_RADIUS = 0.0175
_C01_HAMMER_HANDLE_CONTACT_MARGIN = 0.012
_C01_HAMMER_HANDLE_NEAR_MARGIN = 0.009
_C01_HAMMER_HANDLE_PINCH_MIN_DISTANCE = 0.014
_C01_HAMMER_HANDLE_PINCH_MAX_DISTANCE = 0.080
_C01_HAMMER_HANDLE_MAX_AXIAL_SEPARATION = 0.060
_C01_HAMMER_GRASP_MIN_CONTACT_COUNT = 3
_C01_GRASP_ATTACH_DEBOUNCE_SECONDS = 0.08
_C01_WOOD_GRASP_BOX_CENTER = np.zeros(3, dtype=np.float64)
_C01_WOOD_GRASP_BOX_HALF_EXTENTS = np.asarray(
    [0.050, 0.035, _C01_WOOD_HALF_HEIGHT],
    dtype=np.float64,
)
_C01_WOOD_GRASP_CONTACT_MARGIN = 0.012
_C01_WOOD_GRASP_NEAR_MARGIN = 0.009
_C01_WOOD_GRASP_PINCH_DISTANCE_RANGE = (0.020, 0.130)
_C01_STRIKE_ALIGN_MIN_HAND_COMMAND = 0.55
_C01_STRIKE_ALIGN_MIN_RAW_DOWNWARD_SPEED = 0.035
_C01_STRIKE_ALIGN_CAPTURE_RADIAL_ERROR = 0.040
_C01_STRIKE_ALIGN_DEADBAND = 0.003
_C01_STRIKE_ALIGN_MIN_AXIAL_OFFSET = -0.010
_C01_STRIKE_ALIGN_MAX_AXIAL_OFFSET = 0.080
_C01_STRIKE_ALIGN_MIN_FACE_DOWN_ALIGNMENT = float(np.cos(np.deg2rad(50.0)))
_C01_STRIKE_ALIGN_MAX_HAND_HAMMER_DISTANCE = 0.28
_C01_STRIKE_ALIGN_MAX_LINEAR_SPEED = 0.18

# Logical task names do not always match the body names in the source MJCF.
# Keep the task-facing names stable while resolving physics state from the
# actual imported rigid body.
_TASK_SCENE_BODY_NAMES: dict[str, dict[str, str]] = {
    "t25_place_cup_on_plate": {"tray": "plate"},
    "b02_nut_bolt_assembly": {
        "nut": "nut_root_body",
        "round_nut": "nut_root_body",
        "square_nut": "nut_root_body",
        "bolt": "bolt_root_body",
    },
    "t09_insert_peg": {
        "peg": "industreal_round_peg_8mm",
        "round_peg": "industreal_round_peg_8mm",
        "socket": "industreal_tray_insert_round_peg_8mm",
    },
    "t08_hammer_nail": {"hammer": "hammer_body"},
    "t50_water_plant": {
        "spray": "link_2",
        "spray_bottle": "link_2",
    },
    "t14_move_and_click_mouse": {
        "mouse": "link_3",
    },
    "t23_photograph_logo": {
        "camera": "link_19",
        "photograph_camera": "link_19",
    },
    "c07_pitcher_pour_cup": {
        "c07_pitcher": "019_pitcher_base",
        "c07_cup": "025_mug",
    },
}


@dataclass(frozen=True)
class ABCIsaacTaskSpec:
    task_name: str
    bimanual: bool
    target_objects: tuple[str, ...]
    receptacle: str
    aliases: dict[str, str]
    success_distance: float = 0.08
    success_steps: int = 15
    head_eye: tuple[float, float, float] = (-1.35, 0.0, 1.58)
    head_target: tuple[float, float, float] = (-0.12, 0.0, 1.02)


_TASK_SPECS: dict[str, ABCIsaacTaskSpec] = {
    "t26_place_fruit_in_bucket": ABCIsaacTaskSpec(
        "t26_place_fruit_in_bucket",
        False,
        ("apple", "011_banana", "orange", "pear"),
        "bucket",
        {"target_object_pose": "target", "receptacle_pose": "receptacle"},
        success_distance=0.12,
        success_steps=3,
    ),
    "t25_place_cup_on_plate": ABCIsaacTaskSpec(
        "t25_place_cup_on_plate",
        False,
        ("025_mug",),
        "tray",
        {"target_object_pose": "target", "plate_pose": "receptacle"},
        success_distance=0.10,
        success_steps=25,
    ),
    "t24_place_ball_in_bowl": ABCIsaacTaskSpec(
        "t24_place_ball_in_bowl",
        False,
        ("tennis_ball", "golf_ball", "mini_soccer_ball"),
        "024_bowl",
        {"target_ball_pose": "target", "bowl_pose": "receptacle"},
        success_distance=0.10,
        success_steps=6,
    ),
    "t19_orient_cup_upright": ABCIsaacTaskSpec(
        "t19_orient_cup_upright",
        False,
        ("a_cup",),
        "table",
        {"target_object_pose": "target", "receptacle_pose": "receptacle"},
        success_distance=0.06,
        success_steps=8,
    ),
    "t04_cover_pitcher": ABCIsaacTaskSpec(
        "t04_cover_pitcher",
        False,
        ("pitcher_lid",),
        "019_pitcher_base",
        {"target_object_pose": "target", "receptacle_pose": "receptacle"},
        success_distance=0.09,
        success_steps=15,
    ),
    "b02_nut_bolt_assembly": ABCIsaacTaskSpec(
        "b02_nut_bolt_assembly",
        False,
        ("nut", "round_nut", "square_nut"),
        "bolt",
        {"nut_pose": "target", "bolt_pose": "receptacle"},
        success_distance=0.045,
        success_steps=12,
    ),
    "t09_insert_peg": ABCIsaacTaskSpec(
        "t09_insert_peg",
        False,
        ("peg", "round_peg", "industreal_round_peg_8mm"),
        "socket",
        {
            "peg_pose": "target",
            "peg_ori_pose": "target",
            "socket_pose": "receptacle",
            "socket_ori_pose": "receptacle",
        },
        success_distance=0.035,
        success_steps=10,
    ),
    "t44_stack_blocks": ABCIsaacTaskSpec(
        "t44_stack_blocks",
        False,
        ("036_wood_block", "BuildingBlock017", "BuildingBlock018", "BuildingBlock019"),
        "036_wood_block",
        {"target_object_pose": "target", "receptacle_pose": "receptacle"},
        success_distance=0.075,
        success_steps=15,
    ),
    "t03_cover_box": ABCIsaacTaskSpec(
        "t03_cover_box",
        False,
        ("Box_lid",),
        "Clear_Box",
        {"target_object_pose": "target", "receptacle_pose": "receptacle"},
        success_distance=0.12,
        success_steps=15,
    ),
    "t08_hammer_nail": ABCIsaacTaskSpec(
        "t08_hammer_nail",
        True,
        ("hammer",),
        "nail",
        {"hammer_ori_pose": "target", "nail_ori_pose": "receptacle", "wood_ori_pose": "wood"},
        success_distance=0.07,
        success_steps=3,
    ),
    "t36_scoop_marbles": ABCIsaacTaskSpec(
        "t36_scoop_marbles",
        False,
        ("golf_ball",),
        "024_bowl",
        {"target_object_pose": "target", "receptacle_pose": "receptacle"},
        success_distance=0.10,
        success_steps=8,
    ),
    "t50_water_plant": ABCIsaacTaskSpec(
        "t50_water_plant",
        False,
        ("spray", "spray_bottle"),
        "plant",
        {"spray_ori_pose": "target", "plant_ori_pose": "receptacle"},
        success_distance=0.18,
        success_steps=_C04_SUCCESS_STEPS,
    ),
    "t14_move_and_click_mouse": ABCIsaacTaskSpec(
        "t14_move_and_click_mouse",
        False,
        ("link_3", "mouse"),
        "mousepad",
        {"mouse_ori_pose": "target", "mousepad_ori_pose": "receptacle"},
        success_distance=0.12,
        success_steps=_C05_SUCCESS_STEPS,
    ),
    "t23_photograph_logo": ABCIsaacTaskSpec(
        "t23_photograph_logo",
        True,
        ("camera", "photograph_camera"),
        "logo",
        {"camera_ori_pose": "target", "logo_ori_pose": "receptacle"},
        success_distance=0.18,
        success_steps=1,
    ),
    "c07_pitcher_pour_cup": ABCIsaacTaskSpec(
        "c07_pitcher_pour_cup",
        False,
        ("c07_pitcher", *_C07_PARTICLE_NAMES),
        "c07_cup",
        {"pitcher_pose": "c07_pitcher", "cup_pose": "receptacle"},
        success_distance=0.12,
        success_steps=8,
    ),
    "c08_tongs_ice_cubes": ABCIsaacTaskSpec(
        "c08_tongs_ice_cubes",
        False,
        ("c08_tongs", *_C08_ICE_NAMES),
        "c08_glass_cup",
        {"tongs_pose": "c08_tongs", "cup_pose": "receptacle"},
        success_distance=0.12,
        success_steps=8,
    ),
    "c09_staple_paper_sheets": ABCIsaacTaskSpec(
        "c09_staple_paper_sheets",
        True,
        _C09_PAPER_NAMES,
        "c09_work_zone",
        {"paper_pose": "c09_paper_0", "stapler_pose": "c09_stapler"},
        success_distance=0.10,
        success_steps=3,
    ),
    "t02_connect_charger_and_cable": ABCIsaacTaskSpec(
        "t02_connect_charger_and_cable",
        True,
        ("c10_charger_plug", "c10_charging_cable"),
        "c10_power_strip",
        {"plug_pose": "c10_charger_plug", "strip_pose": "receptacle"},
        success_distance=0.08,
        success_steps=3,
    ),
    "c11_xylophone_sequence": ABCIsaacTaskSpec(
        "c11_xylophone_sequence",
        False,
        ("c11_mallet",),
        "c11_return_zone",
        {"mallet_pose": "target", "xylophone_pose": "c11_xylophone"},
        success_distance=0.08,
        success_steps=2,
    ),
    "t11_line_up_blocks": ABCIsaacTaskSpec(
        "t11_line_up_blocks",
        False,
        ("c12_set_square", *_C12_BLOCK_NAMES),
        "c12_baseline_pad",
        {"tool_pose": "c12_set_square", "baseline_pose": "receptacle"},
        success_distance=0.12,
        success_steps=10,
    ),
    # D-series tasks use ``HandsOnDIsaacLabEnv`` for their task logic.  The
    # specs live here because the shared scene/camera/reset runtime resolves
    # task bodies through this table before the D subclass is constructed.
    "d01_bimanual_nut_bolt_assembly": ABCIsaacTaskSpec(
        "d01_bimanual_nut_bolt_assembly",
        True,
        ("nut_root_body",),
        "bolt_root_body",
        {"nut_pose": "target", "bolt_pose": "receptacle"},
        success_distance=0.045,
        success_steps=12,
    ),
    "t12_load_and_start_microwave": ABCIsaacTaskSpec(
        "t12_load_and_start_microwave",
        True,
        ("d02_plate", "003_cracker_box", "008_pudding_box"),
        "microwave_object",
        {
            "plate_pose": "d02_plate",
            "food_pose": "003_cracker_box",
            "microwave_pose": "receptacle",
        },
        success_distance=0.30,
        success_steps=1,
    ),
    "t21_pass_block": ABCIsaacTaskSpec(
        "t21_pass_block",
        True,
        (
            "d03_target_root",
            "d03_flat_screwdriver",
            "d03_phillips_screwdriver",
            "d03_spoon",
            "d03_plastic_bolt",
            "d03_a_marbles",
            "d03_b_marble_1",
            "d03_b_marble_2",
            "d03_hammer",
            "d03_nails",
            "d03_bowl",
        ),
        "d03_box",
        {"target_pose": "d03_target_root", "box_pose": "receptacle"},
        success_distance=0.20,
        success_steps=8,
    ),
    "d04_clear_hanoi_obstacles": ABCIsaacTaskSpec(
        "d04_clear_hanoi_obstacles",
        True,
        (
            "hanoi_disk_large",
            "hanoi_disk_medium",
            "hanoi_disk_small",
            "d04_block_0",
            "d04_block_1",
            "d04_block_2",
            "d04_block_3",
            "d04_block_4",
            "d04_tennis_ball",
            "d04_golf_ball",
        ),
        "hanoi_base",
        {"hanoi_base_ori_pos": "receptacle"},
        success_distance=0.07,
        success_steps=25,
    ),
    "t43_squeeze_scissors": ABCIsaacTaskSpec(
        "t43_squeeze_scissors",
        True,
        ("d05_scissors_root_body",),
        "table",
        {"scissors_pose": "target"},
        success_distance=0.08,
        success_steps=1,
    ),
    "t42_sort_objects": ABCIsaacTaskSpec(
        "t42_sort_objects",
        True,
        (
            "d06_cracker_box",
            "d06_sugar_box",
            "d06_tomato_soup_can",
            "d06_tuna_fish_can",
            "d06_bowl",
            "d06_mug",
        ),
        "d06_basket_box",
        {
            "target_pose": "d06_cracker_box",
            "box_basket_pose": "receptacle",
        },
        success_distance=0.18,
        success_steps=8,
    ),
    "d07_organize_office_desk": ABCIsaacTaskSpec(
        "d07_organize_office_desk",
        True,
        (
            "d07_keyboard",
            "d07_mouse",
            "d07_figurine",
            "d07_alarm_clock",
            "d07_misc_0",
            "d07_misc_1",
            "d07_misc_2",
        ),
        "d07_drawer",
        {
            "mouse_pose": "d07_mouse",
            "keyboard_pose": "d07_keyboard",
            "drawer_pose": "receptacle",
        },
        success_distance=0.14,
        success_steps=8,
    ),
    "d08_toast_bread": ABCIsaacTaskSpec(
        "d08_toast_bread",
        True,
        ("d08_bread_0", "d08_bread_1"),
        "d08_plate",
        {
            "bread_pose": "d08_bread_0",
            "plate_pose": "receptacle",
        },
        success_distance=0.14,
        success_steps=8,
    ),
    "t31_prepare_coffee": ABCIsaacTaskSpec(
        "t31_prepare_coffee",
        True,
        ("d09_mug",),
        "d09_tray",
        {
            "mug_pose": "target",
            "tray_pose": "receptacle",
        },
        success_distance=0.12,
        success_steps=8,
    ),
    "d10_load_dishwasher": ABCIsaacTaskSpec(
        "d10_load_dishwasher",
        True,
        ("d10_mug", "d10_bowl", "d10_plate"),
        "d10_dishwasher",
        {
            "mug_pose": "d10_mug",
            "dishwasher_pose": "receptacle",
        },
        success_distance=0.25,
        success_steps=8,
    ),
    "d11_presoak_pan": ABCIsaacTaskSpec(
        "d11_presoak_pan",
        True,
        ("d11_pan", "d11_sponge"),
        "d11_sink",
        {
            "pan_pose": "target",
            "sink_pose": "receptacle",
        },
        success_distance=0.25,
        success_steps=8,
    ),
    "d12_prepare_cutting_station": ABCIsaacTaskSpec(
        "d12_prepare_cutting_station",
        True,
        ("d12_knife", "d12_cutting_board", "d12_meat"),
        "d12_board_goal",
        {
            "knife_pose": "d12_knife",
            "board_pose": "d12_cutting_board",
            "goal_pose": "receptacle",
        },
        success_distance=0.18,
        success_steps=8,
    ),
    # E-series tasks keep the same robot, controller, camera, reset, and
    # recorder contract as D-series tasks.  Their non-rigid state is owned by
    # the E subclass, so the logical garment name intentionally has no rigid
    # body view in the imported MJCF scene.
    "e01_fold_garment": ABCIsaacTaskSpec(
        "e01_fold_garment",
        True,
        ("garment",),
        "table",
        {"garment_pose": "target"},
        success_distance=0.12,
        success_steps=15,
        head_eye=(-1.40, 0.0, 1.72),
        head_target=(-0.15, 0.0, 0.98),
    ),
    "e02_tie_rope_knot": ABCIsaacTaskSpec(
        "e02_tie_rope_knot",
        True,
        ("e02_rope",),
        "table",
        {"rope_pose": "target"},
        success_distance=0.10,
        success_steps=15,
        head_eye=(-1.42, 0.0, 1.74),
        head_target=(-0.16, 0.0, 0.96),
    ),
    "e03_untie_random_rope_knot": ABCIsaacTaskSpec(
        "e03_untie_random_rope_knot",
        True,
        ("e03_rope",),
        "table",
        {"rope_pose": "target"},
        success_distance=0.10,
        success_steps=15,
        head_eye=(-1.42, 0.0, 1.74),
        head_target=(-0.16, 0.0, 0.96),
    ),
    "e04_cats_cradle_cycle": ABCIsaacTaskSpec(
        "e04_cats_cradle_cycle",
        True,
        ("e04_string_loop",),
        "table",
        {"string_pose": "target"},
        success_distance=0.10,
        success_steps=15,
        head_eye=(-1.42, 0.0, 1.74),
        head_target=(-0.16, 0.0, 0.97),
    ),
    "e05_fold_cloth_wipe_table": ABCIsaacTaskSpec(
        "e05_fold_cloth_wipe_table",
        True,
        ("e05_cloth",),
        "table",
        {"cloth_pose": "target"},
        success_distance=0.10,
        success_steps=15,
        head_eye=(-1.42, 0.0, 1.78),
        head_target=(-0.18, 0.0, 0.94),
    ),
}


def supported_abc_collection_tasks() -> frozenset[str]:
    return frozenset(name for name in _TASK_SPECS if name in ABC_PREFIX_TASKS or name[:1] in {"a", "b", "c"})


def supported_native_collection_tasks() -> frozenset[str]:
    """Return every task backed by the shared native IsaacLab runtime."""
    return frozenset(_TASK_SPECS)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        parsed = float(value)
    except ValueError:
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return parsed


def _b02_physics_dt(default_physics_dt: float) -> float:
    if _bool_env("HANDSON_DISABLE_B02_MUJOCO_TIMESTEP"):
        return float(default_physics_dt)
    # 250/500 Hz improves tiny-object contacts but is too heavy for GUI
    # collection on some machines.  Keep the original 120 Hz default and allow
    # explicit runtime overrides for A/B testing.
    target_hz = _float_env("HANDSON_B02_PHYSICS_HZ", _B02_DEFAULT_PHYSICS_HZ)
    target_hz = float(np.clip(target_hz, 120.0, _B02_MUJOCO_PHYSICS_HZ))
    target_dt = 1.0 / target_hz
    return min(float(default_physics_dt), target_dt)


def _b02_mujoco_hand_drive_maps(
    side: Literal["left", "right"],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    stiffness: dict[str, float] = {}
    damping: dict[str, float] = {}
    effort: dict[str, float] = {}
    for joint_suffix, (kp, kv, force_limit) in _B02_MUJOCO_HAND_JOINT_DRIVES.items():
        joint_name = f"{side}_{joint_suffix}"
        stiffness[joint_name] = float(kp)
        damping[joint_name] = float(kv)
        effort[joint_name] = float(force_limit)
    return stiffness, damping, effort


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        q = _IDENTITY_QUAT_WXYZ
    else:
        q = q / norm
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quat))
    return _IDENTITY_QUAT_WXYZ.copy() if norm < 1e-9 else quat / norm


def _quat_mul_wxyz(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = np.asarray(q0, dtype=np.float64).reshape(4)
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64).reshape(4)
    result = np.asarray(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(result))
    return _IDENTITY_QUAT_WXYZ.copy() if norm < 1e-9 else result / norm


def _deferred_keeps_hand_visual_path(path: str) -> bool:
    normalized = str(path).lower()
    return any(marker in normalized for marker in _DEFERRED_VISIBLE_HAND_PATH_MARKERS)


def _randomized_physx_pose(
    pose_xyzw: np.ndarray,
    *,
    xy_offset: np.ndarray,
    z_offset: float,
    yaw_degrees: float,
) -> np.ndarray:
    """Apply a world-frame reset perturbation to PhysX ``pos + xyzw`` poses."""
    result = np.asarray(pose_xyzw, dtype=np.float64).copy().reshape(-1, 7)
    result[:, :2] += np.asarray(xy_offset, dtype=np.float64).reshape(1, 2)
    result[:, 2] += float(z_offset)
    if abs(float(yaw_degrees)) > 1e-12:
        yaw = math.radians(float(yaw_degrees))
        yaw_quat = np.asarray(
            [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
            dtype=np.float64,
        )
        for index in range(result.shape[0]):
            quat_xyzw = result[index, 3:7]
            quat_wxyz = np.asarray(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                dtype=np.float64,
            )
            randomized = _quat_mul_wxyz(yaw_quat, quat_wxyz)
            result[index, 3:7] = [
                randomized[1],
                randomized[2],
                randomized[3],
                randomized[0],
            ]
    return result


def _constrain_task_xy_to_workspace(task_name: str, xy: np.ndarray) -> np.ndarray:
    """Keep randomized centers visible and inside the shared arm workspace."""
    result = constrain_recording_object_xy(xy)
    if task_name == "t26_place_fruit_in_bucket":
        result = np.clip(
            result,
            _A01_RIGHT_HAND_XY_BOUNDS[0],
            _A01_RIGHT_HAND_XY_BOUNDS[1],
        )
    return result


def _pose_vector_to_matrix(pose: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_wxyz_to_matrix(np.asarray(pose, dtype=np.float64)[3:7])
    matrix[:3, 3] = np.asarray(pose, dtype=np.float64)[:3]
    return matrix


def _usd_transform_to_pose_matrix(transform) -> np.ndarray:
    translation = transform.ExtractTranslation()
    rotation = transform.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    quat = np.asarray(
        [
            float(rotation.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_wxyz_to_matrix(quat)
    matrix[:3, 3] = [float(translation[0]), float(translation[1]), float(translation[2])]
    return matrix


def _invert_pose_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    inv_pose = np.eye(4, dtype=np.float64)
    inv_pose[:3, :3] = pose[:3, :3].T
    inv_pose[:3, 3] = -(inv_pose[:3, :3] @ pose[:3, 3])
    return inv_pose


def _focal_length_for_vertical_fov(
    fovy_degrees: float,
    vertical_aperture: float = _RECORDING_CAMERA_VERTICAL_APERTURE,
) -> float:
    return focal_length_for_vertical_fov(fovy_degrees, vertical_aperture)


def _focal_length_for_horizontal_fov(
    hfov_degrees: float,
    horizontal_aperture: float,
) -> float:
    return focal_length_for_horizontal_fov(hfov_degrees, horizontal_aperture)


class HandsOnABCIsaacLabEnv(BimanualAssemblyIsaacLabEnv):
    """Task-specific scene layer over the bimanual assembly robot runtime."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        task_name: str,
        seed: int | None = None,
        control_dt: float | None = None,
        physics_dt: float = 1.0 / 120.0,
        time_limit: float = 120.0,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
        randomize: bool = False,
        randomize_dynamics: bool = False,
        hand_type: str = "wuji1",
        arm_type: str = "rokae",
        launch_isaac: bool | None = None,
        headless: bool | None = None,
        render_resolution: tuple[int, int] = _DEFAULT_RENDER_SIZE,
        hz: int = 30,
        isaac_device: str | None = None,
        target_object: str | None = None,
        existing_simulation_app: Any | None = None,
        existing_app_launcher: Any | None = None,
        **_: Any,
    ):
        if task_name not in _TASK_SPECS:
            supported = ", ".join(sorted(_TASK_SPECS))
            raise ValueError(f"Unsupported IsaacLab ABC task {task_name!r}; expected one of: {supported}")
        self._spec = _TASK_SPECS[task_name]
        self.task_name = task_name
        self._deferred_arm_render = bool(
            _bool_env("HANDSON_ISAACLAB_DEFER_ARM_RENDER")
            and (task_name in ABCD_PREFIX_TASKS or task_name[0].lower() in {"a", "b", "c", "d"})
        )
        if self._deferred_arm_render:
            self._fast_shutdown_on_close = True
        if control_dt is None:
            control_dt = 0.02
        if task_name == "b02_nut_bolt_assembly":
            physics_dt = _b02_physics_dt(float(physics_dt))
        if launch_isaac is None:
            launch_isaac = not _bool_env("HANDSON_ISAACSIM_INTERFACE_ONLY")
        launch_task_scene = bool(launch_isaac)
        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_mode=render_mode,
            image_obs=image_obs,
            randomize=randomize,
            randomize_dynamics=randomize_dynamics,
            hand_type=hand_type,
            arm_type=arm_type,
            launch_isaac=False,
            headless=headless,
            render_resolution=render_resolution,
            hz=hz,
            isaac_device=isaac_device,
            existing_simulation_app=existing_simulation_app,
            existing_app_launcher=existing_app_launcher,
        )

        self._launch_isaac = launch_task_scene
        self._randomize_dynamics = bool(randomize_dynamics)
        self._stage = None
        self._cameras: dict[str, Any] = {}
        self._camera_aliases: dict[str, str] = {}
        self._dynamic_cameras: set[str] = set()
        self._legacy_wrist_cameras = False
        self._recording_camera_body_indices: dict[str, int] = {}
        self._wrist_obstruction_visibility: tuple[tuple[str, Any], ...] | None = None
        self._deferred_robot_body_visibility: tuple[tuple[str, Any], ...] | None = None
        self.recording_camera_profile = recording_camera_profile()
        self._robots: dict[str, Any] = {}
        self._prim_cache: dict[str, Any] = {}
        self._task_body_views: dict[str, Any] = {}
        self._task_articulation_views: dict[str, tuple[Any, Any]] = {}
        self._task_reset_body_states: list[tuple[str, Any, Any, Any, Any]] = []
        self._task_kinematic_body_paths: set[str] = set()
        self._task_reset_articulation_states: list[tuple[str, Any, Any, Any, Any, Any, Any]] = []
        self._task_dof_views: dict[str, tuple[Any, int, float]] = {}
        self._randomize_task_scene_visuals = None
        self._target_object_name = self._resolve_target_object(target_object)
        self._poses: dict[str, np.ndarray] = {}
        self._lid_start_pose: np.ndarray | None = None
        self._lid_was_moved = False
        self._b01_hand_was_closed = False
        self._b01_release_assist_active = False
        self._b01_hand_lid_clear_steps = 0
        self._last_b01_lid_success_metrics: dict[str, Any] = {}
        self._b02_prev_relative_yaw: float | None = None
        self._b02_signed_twist_deg = 0.0
        self._b02_thread_entry_depth: float | None = None
        self._b02_min_thread_depth: float | None = None
        self._b02_thread_engaged_seen = False
        self._b02_grasp_assist: dict[str, Any] | None = None
        self._b02_grasp_assist_metrics: dict[str, Any] = {}
        self._b02_grasp_body_indices: dict[str, int] = {}
        self._c01_prev_nail_depth: float | None = None
        self._c01_prev_hammer_pose: np.ndarray | None = None
        self._c01_prev_nail_pose: np.ndarray | None = None
        self._c01_had_valid_hit = False
        self._c01_max_valid_hit_depth = 0.0
        self._c01_candidate_strike_depth_start: float | None = None
        self._c01_candidate_strike_max_depth = 0.0
        self._c01_last_valid_hit_depth_increment = 0.0
        self._c01_last_candidate_hit_depth_increment = 0.0
        self._t08_hammer_nail_contact_view: Any | None = None
        self._t08_hammer_nail_contact_views: list[Any] = []
        self._c01_hit_contact_available = False
        self._c01_last_hit_contact_force = 0.0
        self._c01_last_face_nail_distance = math.inf
        self._c01_last_face_nail_radial_error = math.inf
        self._c01_last_face_nail_axial_offset = math.inf
        self._c01_last_face_down_alignment = -1.0
        self._c01_last_hammer_downward_speed = 0.0
        self._c01_last_hammer_motion_down_alignment = -1.0
        self._c01_grasp_assists: dict[str, dict[str, Any]] = {}
        self._c01_grasp_candidates: dict[str, dict[str, Any]] = {}
        self._c01_strike_align_prev_raw_tcp: np.ndarray | None = None
        self._c01_strike_align_assist_metrics: dict[str, Any] = {}
        self._last_a06_success_metrics: dict[str, Any] = {}
        self._last_b04_stack_metrics: dict[str, Any] = {}
        self._b04_velocity_samples: list[tuple[float, float]] = []
        self._b04_velocity_stable_latched = False
        self._b04_block_contact_views: dict[str, Any] = {}
        self._c04_spray_visible = False
        self._c04_trigger_contact_view: Any | None = None
        self._c04_grasp_contact_views: list[Any] = []
        self._c04_trigger_phase = "idle"
        self._c04_trigger_cycle_latched = False
        self._c04_trigger_contact_seen = False
        self._c04_grasp_seen = False
        self._c04_trigger_contact_grace = 0
        self._c04_grasp_contact_grace = 0
        self._c04_grasp_ready_steps = 0
        self._c04_trigger_armed = False
        self._c04_trigger_pulled = False
        self._c04_water_emitting = False
        self._c04_failed_trigger_outside = False
        self._last_c04_success_metrics: dict[str, Any] = {}
        self._c05_capture_start_pending = False
        self._c05_mouse_start_xy: np.ndarray | None = None
        self._c05_button_initial_positions = {
            "mouse_joint_0": 0.0,
            "mouse_joint_1": 0.0,
        }
        self._c05_failed_early_click = False
        self._last_c05_success_metrics: dict[str, Any] = {}
        self._c06_shutter_was_down = False
        self._c06_shutter_contact_view: Any | None = None
        self._c06_right_contact_grace = 0
        self._last_c06_success_metrics: dict[str, Any] = {}
        self._last_c07_success_metrics: dict[str, Any] = {}
        self._c07_tilt_seen = False
        self._assist_robot_body_indices: dict[str, int] = {}
        self._c_assist_raw_right_hand = self._right_hand.copy()
        self._c_assist_raw_right_tcp_pose = self._right_tcp_pose.copy()
        self._c_assist_raw_left_hand = self._left_hand.copy()
        self._c_assist_raw_left_tcp_pose = self._left_tcp_pose.copy()
        self._c_assist_control_frame = 0
        self._c_assist_free_hand_handoffs: dict[
            str,
            dict[str, np.ndarray],
        ] = {}
        self._c07_pitcher_grasp: dict[str, Any] | None = None
        self._c07_pitcher_pinned = False
        self._c07_pitcher_pinned_pose: np.ndarray | None = None
        self._c07_delivered_particles: set[str] = set()
        self._c07_previous_particle_cup_local: dict[str, np.ndarray] = {}
        self._last_c08_success_metrics: dict[str, Any] = {}
        self._c08_tongs_close_count = 0
        self._c08_tongs_was_open = False
        self._c08_tongs_grasp: dict[str, Any] | None = None
        self._c08_tongs_control_armed = False
        self._c08_held_ice: dict[str, Any] | None = None
        self._c08_delivered_ice: set[str] = set()
        self._c08_delivered_ice_local: dict[str, np.ndarray] = {}
        self._last_c09_success_metrics: dict[str, Any] = {}
        self._c09_top_opened_seen = False
        self._c09_paper_grasps: dict[str, dict[str, Any]] = {}
        self._c09_papers_stacked = False
        self._c09_stack_state: dict[str, Any] | None = None
        self._c09_stapler_anchor: np.ndarray | None = None
        self._last_c10_success_metrics: dict[str, Any] = {}
        self._c10_assist_metrics: dict[str, Any] = {}
        self._c10_connector_grasps: dict[str, dict[str, Any]] = {}
        self._c10_first_connected_seen = False
        self._c10_order_violation = False
        self._c10_plug_latched = False
        self._c10_cable_latched = False
        self._c10_latch_stable_steps = {
            "c10_charger_plug": 0,
            "c10_charging_cable": 0,
        }
        self._c10_plug_latch_relative: np.ndarray | None = None
        self._c10_cable_latch_relative: np.ndarray | None = None
        self._last_c11_success_metrics: dict[str, Any] = {}
        self._c11_target_sequence = _C11_DEFAULT_SEQUENCE
        self._c11_sequence_progress = 0
        self._c11_wrong_hits = 0
        self._c11_last_hit_key: str | None = None
        self._c11_current_hit_key: str | None = None
        self._c11_mallet_grasp: dict[str, Any] | None = None
        self._c11_previous_head_world: np.ndarray | None = None
        self._c11_pending_crossings: dict[str, int] = {}
        self._c11_hit_armed = True
        self._c11_key_contact_views: dict[str, Any] = {}
        self._last_c12_success_metrics: dict[str, Any] = {}
        self._c12_tool_start_xy: np.ndarray | None = None
        self._c12_tool_used = False
        self._c12_direct_hand_contact = False
        self._c12_block_contact_views: dict[str, Any] = {}
        self._c12_tool_block_contact_views: dict[str, Any] = {}
        self._c12_tool_grasp: dict[str, Any] | None = None
        self._c12_active_push: dict[str, Any] | None = None
        self._c12_push_candidate: dict[str, Any] | None = None
        self._c12_slot_by_block: dict[str, int] = {}
        self._c12_settled_blocks: set[str] = set()
        self._c12_settled_targets: dict[str, np.ndarray] = {}
        self._success_condition_window: Any | None = None
        self._success_condition_frame: Any | None = None
        self._success_condition_labels: dict[str, Any] = {}
        self._success_condition_text_labels: dict[str, Any] = {}
        self._task_status_label: Any | None = None
        self._success_condition_overlay_failed = False

        if self._spec.bimanual:
            action_shape = (14 + 2 * self._hand_action_dim,)
        else:
            action_shape = (7 + self._hand_action_dim,)
        self.action_space = spaces.Box(
            low=np.full(action_shape, -1.0, dtype=np.float32),
            high=np.full(action_shape, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        if self._launch_isaac:
            try:
                self._setup_isaaclab()
            except BaseException:
                import traceback
                traceback.print_exc()
                self.close()
                raise
        self._initialize_physx_ground_truth_recording()

    def _in_process_scene_rebuild_kwargs(self, seed: int) -> dict[str, Any]:
        kwargs = super()._in_process_scene_rebuild_kwargs(seed)
        kwargs.update(
            {
                "task_name": self.task_name,
                "randomize_dynamics": bool(self._randomize_dynamics),
                "target_object": self._target_object_name,
            }
        )
        return kwargs

    def _resolve_target_object(self, target_object: str | None) -> str:
        if target_object:
            value = str(target_object).strip()
            if value in self._spec.target_objects:
                return value
        return self._spec.target_objects[0]

    def _build_observation_space(self) -> spaces.Dict:
        state_spaces: dict[str, spaces.Space] = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14 if self._spec.bimanual else 7,)),
            "gripper_pose": spaces.Box(-np.inf, np.inf, shape=(32 if self._spec.bimanual else self._hand_action_dim,)),
            "target_object_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "receptacle_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "object_poses": spaces.Box(-np.inf, np.inf, shape=(max(1, len(self._spec.target_objects)), 7)),
            "target_selector": spaces.Box(-np.inf, np.inf, shape=(max(1, len(self._spec.target_objects)),)),
            "target_id": spaces.Box(-np.inf, np.inf, shape=(1,)),
            "target_name": spaces.Box(-np.inf, np.inf, shape=(max(1, len(self._spec.target_objects)),)),
            "plate_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "distractor_poses": spaces.Box(-np.inf, np.inf, shape=(max(0, len(self._spec.target_objects) - 1), 7)),
            "target_ball_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "tennis_ball_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "golf_ball_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "mini_soccer_ball_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "bowl_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "nut_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "bolt_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "nut_to_bolt": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "assembly_metrics": spaces.Box(-np.inf, np.inf, shape=(6,)),
            "peg_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "socket_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "peg_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "socket_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "hammer_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "nail_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "wood_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "nail_depth": spaces.Box(-np.inf, np.inf, shape=(1,)),
            "spray_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "plant_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "mouse_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "mousepad_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "camera_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "logo_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "pitcher_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "cup_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "particle_count": spaces.Box(-np.inf, np.inf, shape=(1,)),
            "spill_count": spaces.Box(-np.inf, np.inf, shape=(1,)),
            "return_zone_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "tongs_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "tongs_open_angle": spaces.Box(-np.inf, np.inf, shape=(1,)),
            "ice_cube_poses": spaces.Box(-np.inf, np.inf, shape=(len(_C08_ICE_NAMES), 7)),
            "stapler_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "stapler_hinge": spaces.Box(-np.inf, np.inf, shape=(1,)),
            "paper_poses": spaces.Box(-np.inf, np.inf, shape=(len(_C09_PAPER_NAMES), 7)),
            "staple_zone_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "charger_plug_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "power_strip_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "charging_cable_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "connection_state": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "mallet_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "xylophone_key_poses": spaces.Box(-np.inf, np.inf, shape=(len(_C11_KEY_NAMES), 7)),
            "sequence_target": spaces.Box(
                -np.inf,
                np.inf,
                shape=(_C11_MAX_SEQUENCE_LENGTH,),
            ),
            "sequence_progress": spaces.Box(-np.inf, np.inf, shape=(1,)),
            "wrong_hit_count": spaces.Box(-np.inf, np.inf, shape=(1,)),
            "tool_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "block_poses": spaces.Box(-np.inf, np.inf, shape=(len(_C12_BLOCK_NAMES), 7)),
            "baseline_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
            "alignment_error": spaces.Box(-np.inf, np.inf, shape=(6,)),
            "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,)),
        }
        observation_spaces: dict[str, Any] = {"state": spaces.Dict(state_spaces)}
        if self.image_obs:
            wrist_width, wrist_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
            wrist_image_shape = (wrist_height, wrist_width, 3)
            observation_spaces["images"] = spaces.Dict(
                {
                    "wrist_left": spaces.Box(
                        0,
                        255,
                        shape=wrist_image_shape,
                        dtype=np.uint8,
                    ),
                    "wrist_right": spaces.Box(
                        0,
                        255,
                        shape=wrist_image_shape,
                        dtype=np.uint8,
                    ),
                    "random_camera" if self.randomize else "ego": spaces.Box(
                        0,
                        255,
                        shape=self._image_shape,
                        dtype=np.uint8,
                    ),
                }
            )
        return spaces.Dict(observation_spaces)

    def _setup_isaaclab(self) -> None:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        try:
            from isaaclab.app import AppLauncher
        except Exception as exc:
            raise IsaacSimUnavailable("Isaac Lab AppLauncher is not available in this Python environment.") from exc

        launcher_args = {
            "headless": self._headless,
            "enable_cameras": self.image_obs,
            "width": self._render_width,
            "height": self._render_height,
            "device": self._isaac_device,
        }
        kit_args = isaac_kit_args(
            existing=self._task_specific_isaac_kit_args(),
            headless=self._headless,
        )
        if kit_args:
            launcher_args["kit_args"] = kit_args
        if self._simulation_app is None:
            from .compat import create_app_launcher
            self._app_launcher = create_app_launcher(AppLauncher, launcher_args)
            self._simulation_app = self._app_launcher.app
        enforce_isaac_viewport_only(self._simulation_app, headless=self._headless)
        self._setup_timeline_pause_guard()
        self._setup_keyboard_reset()
        try:
            from isaacsim.core.utils.extensions import enable_extension

            enable_extension("isaacsim.asset.importer.mjcf")
            self._simulation_app.update()
        except Exception as exc:
            self._isaac_warnings.append(f"Could not enable MJCF importer: {exc}")

        try:
            import torch
            import isaaclab.sim as sim_utils
            from isaaclab.assets import Articulation
            from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
            from isaaclab.sim import SimulationContext
            from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms
        except Exception as exc:
            raise IsaacSimUnavailable("Isaac Lab APIs are not available in this Python environment.") from exc

        self._ik_controller_cls = DifferentialIKController
        self._ik_controller_cfg_cls = DifferentialIKControllerCfg
        self._math_utils = {
            "matrix_from_quat": matrix_from_quat,
            "quat_inv": quat_inv,
            "subtract_frame_transforms": subtract_frame_transforms,
        }

        self._torch = torch
        sim_cfg = sim_utils.SimulationCfg(
            device=self._isaac_device,
            dt=self._physics_dt,
            render_interval=max(1, int(round(self._control_dt / self._physics_dt))),
            gravity=(0.0, 0.0, -9.81),
            physx=sim_utils.PhysxCfg(
                solver_type=1,
                enable_enhanced_determinism=True,
                enable_ccd=(
                    (self.task_name == "t08_hammer_nail" and _C01_ENABLE_CCD)
                    or self.task_name == "c07_pitcher_pour_cup"
                ),
                max_position_iteration_count=64,
                max_velocity_iteration_count=4,
                bounce_threshold_velocity=0.2,
                friction_offset_threshold=0.01,
                friction_correlation_distance=0.00625,
                gpu_max_rigid_contact_count=SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT,
                gpu_max_rigid_patch_count=SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
        )
        self._sim = SimulationContext(sim_cfg)
        self._sim.set_camera_view(eye=self._spec.head_eye, target=self._spec.head_target)

        from .bimanual_assembly_scene import (
            _repair_imported_visual_materials,
            _robot_cfg,
            fix_robot_base_paths,
            randomize_task_scene_visuals,
            spawn_center_rig,
        )
        from .task_preview import spawn_task_scene

        print(f"[HandsOn/IsaacLab] Constructing ABC collection scene: {self.task_name}", flush=True)
        task_mjcf, _ = spawn_task_scene(self.task_name, _TASK_ROOT, preview_appearance=False)
        import omni.usd

        self._stage = omni.usd.get_context().get_stage()
        self._randomize_task_scene_visuals = randomize_task_scene_visuals
        table_fixed_joints_disabled, table_kinematic = (
            self._configure_randomizable_table_physics()
        )
        self._configure_c05_mousepad_physics()
        fixed_task_fixtures = self._configure_c_tool_fixture_physics()
        disabled_robot_prims = self._disable_imported_task_robots()
        self._spawn_task_specific_assets(sim_utils)
        sim_utils.update_stage()

        center_rig_path = "/World/HandsOnRobots/CenterRig"
        spawned_center_rig = False
        try:
            if not self._stage.GetPrimAtPath(center_rig_path).IsValid():
                spawn_center_rig(center_rig_path)
                spawned_center_rig = True
        except Exception as exc:
            self._add_task_state_warning(f"Could not spawn native center rig: {exc}")

        robot_root_paths = []
        b02_soft_hand_robot_configs = 0
        for side in ("right", "left"):
            robot_cfg = _robot_cfg(side)
            robot_cfg.prim_path = f"/World/HandsOnRobots/{side.capitalize()}Robot"
            if self._apply_b02_mujoco_soft_hand_actuators(robot_cfg, side):
                b02_soft_hand_robot_configs += 1
            self._apply_raised_hand_home(robot_cfg, side)
            self._robots[side] = Articulation(robot_cfg)
            robot_root_paths.append(
                f"{robot_cfg.prim_path}/base_{side}/base_{side}"
            )
        if robot_root_paths:
            fix_robot_base_paths(tuple(robot_root_paths))
        sim_utils.update_stage()
        repaired_robot_materials = 0
        hidden_robot_colliders = 0
        for root_path in (
            "/World/HandsOnRobots/RightRobot",
            "/World/HandsOnRobots/LeftRobot",
            center_rig_path,
        ):
            repaired, hidden = _repair_imported_visual_materials(
                self._stage,
                root_path,
            )
            repaired_robot_materials += repaired
            hidden_robot_colliders += hidden
        if repaired_robot_materials or hidden_robot_colliders:
            sim_utils.update_stage()
        lid_material_bindings, lid_mass_tuned = self._configure_lid_grasp_physics(
            sim_utils
        )
        nut_material_bindings, nut_mass_tuned = (
            self._configure_b02_nut_grasp_physics(sim_utils)
        )
        b02_reset_bodies_tuned = self._configure_b02_reset_stability(sim_utils)
        b02_hand_material_bindings = self._configure_b02_hand_grasp_physics(
            sim_utils
        )
        pitcher_base_material_bindings, pitcher_base_tuned = (
            self._configure_b01_pitcher_base_physics(sim_utils)
        )
        a01_reset_bodies_tuned = self._configure_a01_reset_stability(sim_utils)
        a03_reset_bodies_tuned = self._configure_a03_reset_stability(sim_utils)
        a06_reset_bodies_tuned = self._configure_a06_reset_stability(sim_utils)
        b04_stack_material_bindings, b04_stack_bodies_tuned = (
            self._configure_b04_stack_physics(sim_utils)
        )
        c01_wood_mass_tuned = self._configure_c01_wood_physics()
        t08_hammer_nail_physics_tuned = (
            self._configure_t08_hammer_nail_physics(sim_utils)
        )
        c04_trigger_physics_tuned = self._configure_c04_trigger_physics()
        c05_mouse_buttons_tuned = self._configure_c05_mouse_button_physics()
        c06_shutter_physics_tuned = self._configure_c06_shutter_physics()
        c08_tongs_spring_tuned = self._configure_c08_tongs_spring_physics()
        c06_camera_material_bindings, c06_camera_mass_tuned = (
            self._configure_c06_camera_grasp_physics(sim_utils)
        )
        c12_reset_bodies_tuned = self._configure_c12_reset_stability(sim_utils)
        c11_contact_reporters = self._configure_c11_contact_reporting()
        c12_contact_reporters = self._configure_c12_contact_reporting()
        task_collision_bodies, task_collision_shapes = (
            self._configure_task_collision_safety(sim_utils)
        )
        if (
            lid_material_bindings
            or lid_mass_tuned
            or nut_material_bindings
            or nut_mass_tuned
            or b02_reset_bodies_tuned
            or b02_hand_material_bindings
            or pitcher_base_material_bindings
            or pitcher_base_tuned
            or a01_reset_bodies_tuned
            or a03_reset_bodies_tuned
            or a06_reset_bodies_tuned
            or b04_stack_material_bindings
            or b04_stack_bodies_tuned
            or c01_wood_mass_tuned
            or t08_hammer_nail_physics_tuned
            or c04_trigger_physics_tuned
            or c05_mouse_buttons_tuned
            or c06_shutter_physics_tuned
            or c08_tongs_spring_tuned
            or c06_camera_material_bindings
            or c06_camera_mass_tuned
            or c12_reset_bodies_tuned
            or c11_contact_reporters
            or c12_contact_reporters
            or task_collision_bodies
            or task_collision_shapes
        ):
            sim_utils.update_stage()
        hidden_camera_helpers = self._hide_camera_debug_helpers()
        if hidden_camera_helpers:
            sim_utils.update_stage()
        deferred_robot_visuals_hidden = 0
        if self._deferred_arm_render:
            deferred_robot_visuals_hidden = self._set_deferred_robot_body_hidden(True)
            if deferred_robot_visuals_hidden:
                sim_utils.update_stage()
        runtime_label = (
            "complete robots with arm/body visuals hidden until replay"
            if self._deferred_arm_render
            else "native bimanual assembly robots"
        )
        print(
            f"[HandsOn/IsaacLab] ABC runtime uses {runtime_label} "
            f"({disabled_robot_prims} imported robot prims disabled, "
            f"center_rig_spawned={spawned_center_rig}, "
            f"table_kinematic={table_kinematic}, "
            f"{table_fixed_joints_disabled} table fixed joints disabled, "
            f"{fixed_task_fixtures} C-tool fixtures made kinematic, "
            f"{repaired_robot_materials} robot meshes rebound, "
            f"{hidden_robot_colliders} robot collider meshes hidden, "
            f"{lid_material_bindings} lid colliders friction-tuned, "
            f"lid_mass_tuned={lid_mass_tuned}, "
            f"{nut_material_bindings} B02 nut colliders friction-tuned, "
            f"nut_mass_tuned={nut_mass_tuned}, "
            f"b02_reset_bodies_tuned={b02_reset_bodies_tuned}, "
            f"{b02_hand_material_bindings} B02 hand prims friction-tuned, "
            f"{b02_soft_hand_robot_configs} B02 robot hand drives source-matched, "
            f"{pitcher_base_material_bindings} pitcher-base colliders stability-tuned, "
            f"pitcher_base_tuned={pitcher_base_tuned}, "
            f"a01_reset_bodies_tuned={a01_reset_bodies_tuned}, "
            f"a03_reset_bodies_tuned={a03_reset_bodies_tuned}, "
            f"a06_reset_bodies_tuned={a06_reset_bodies_tuned}, "
            f"{b04_stack_material_bindings} B04 colliders stability-tuned, "
            f"{b04_stack_bodies_tuned} B04 rigid bodies damping-tuned, "
            f"c01_wood_mass_tuned={c01_wood_mass_tuned}, "
            f"t08_hammer_nail_physics_tuned={t08_hammer_nail_physics_tuned}, "
            f"c04_trigger_physics_tuned={c04_trigger_physics_tuned}, "
            f"c05_mouse_buttons_tuned={c05_mouse_buttons_tuned}, "
            f"c06_shutter_physics_tuned={c06_shutter_physics_tuned}, "
            f"c08_tongs_spring_tuned={c08_tongs_spring_tuned}, "
            f"{c06_camera_material_bindings} C06 camera colliders grasp-tuned, "
            f"c06_camera_mass_tuned={c06_camera_mass_tuned}, "
            f"c12_reset_bodies_tuned={c12_reset_bodies_tuned}, "
            f"{c11_contact_reporters} C11 mallet/key contact reporters, "
            f"{c12_contact_reporters} C12 block contact reporters, "
            f"{task_collision_bodies} task bodies anti-tunneling-tuned, "
            f"{task_collision_shapes} task/finger collision shapes shell-tuned, "
            f"{hidden_camera_helpers} camera debug helpers hidden, "
            f"{deferred_robot_visuals_hidden} deferred robot body visuals hidden)",
            flush=True,
        )
        self._init_recording_cameras()
        if self._physx_ground_truth_enabled:
            self._configure_ground_truth_contact_reports(task_mjcf)
            sim_utils.update_stage()
        from .default_appearance import install_environment
        self._reference_robot_visual_roots = (
            "/World/HandsOnRobots/RightRobot",
            "/World/HandsOnRobots/LeftRobot",
            center_rig_path,
        )
        install_environment(self)
        self._sim.reset()
        self._sim_dt = float(self._sim.get_physics_dt())
        self._init_task_specific_runtime()
        self._capture_task_scene_state(task_mjcf)
        self._initialize_c08_tongs_open_state()
        self._init_b04_contact_views()
        self._init_t08_hammer_nail_contact_view()
        self._init_c04_contact_views()
        self._init_c06_shutter_contact_view()
        self._init_c11_key_contact_views()
        self._init_c12_block_contact_views()
        self._cache_joint_targets()
        self._init_haorui_rokae_ik()
        self._sample_reset_state(clear_manual_request=True)
        self._reset_scene_state()
        self._apply_reset_visual_randomization()
        self._update_recording_cameras()
        for _ in range(5):
            self._step_isaaclab()
        self._refresh_object_poses()
        self._init_success_condition_overlay()
        self._update_success_condition_overlay()
        print(
            f"[HandsOn/IsaacLab] Native ABC collection ready: {self.task_name} "
            f"(simulated_arms=2, online_ik=True, "
            f"deferred_arm_render={self._deferred_arm_render})",
            flush=True,
        )

    def _task_specific_isaac_kit_args(self) -> str | None:
        """Return optional Kit settings that must be present before launch."""
        return None

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        """Author extra task assets before the first PhysX reset."""
        return None

    def _init_task_specific_runtime(self) -> None:
        """Create task tensor views after the first PhysX reset."""
        return None

    def _configure_task_collision_safety(
        self, sim_utils: Any
    ) -> tuple[int, int]:
        """Configure C07 anti-tunneling and narrow-handle contact shells."""
        if self._stage is None or self.task_name != "c07_pitcher_pour_cup":
            return 0, 0

        from pxr import PhysxSchema, Usd, UsdPhysics

        body_names = (
            "019_pitcher_base",
            "025_mug",
            *_C07_PARTICLE_NAMES,
        )
        body_paths: dict[str, str] = {}
        for name in body_names:
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                continue
            body_paths[name] = str(prim.GetPath())
        robot_root_paths = tuple(
            path
            for path in _DEFERRED_ROBOT_VISUAL_ROOTS[:2]
            if self._stage.GetPrimAtPath(path).IsValid()
        )
        for path in sorted({*body_paths.values(), *robot_root_paths}):
            sim_utils.make_uninstanceable(path, stage=self._stage)
        if body_paths or robot_root_paths:
            sim_utils.update_stage()
            for name in body_names:
                self._prim_cache.pop(name, None)

        def set_collision_shell(prim: Any, contact_offset: float) -> None:
            collision_api = PhysxSchema.PhysxCollisionAPI(prim)
            if not collision_api:
                collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            contact_attr = collision_api.GetContactOffsetAttr()
            if not contact_attr or not contact_attr.IsValid():
                contact_attr = collision_api.CreateContactOffsetAttr()
            contact_attr.Set(contact_offset)
            rest_attr = collision_api.GetRestOffsetAttr()
            if not rest_attr or not rest_attr.IsValid():
                rest_attr = collision_api.CreateRestOffsetAttr()
            rest_attr.Set(0.0)

        tuned_bodies = 0
        tuned_colliders = 0
        for name in body_names:
            body_prim = self._find_rigid_body_prim(name)
            if body_prim is None:
                continue
            rigid_api = PhysxSchema.PhysxRigidBodyAPI(body_prim)
            if not rigid_api:
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
            particle_ccd = name in _C07_PARTICLE_NAMES
            for getter_name, creator_name, value in (
                ("GetEnableCCDAttr", "CreateEnableCCDAttr", particle_ccd),
                (
                    "GetEnableSpeculativeCCDAttr",
                    "CreateEnableSpeculativeCCDAttr",
                    False,
                ),
                ("GetSolverPositionIterationCountAttr", "CreateSolverPositionIterationCountAttr", 32),
                ("GetSolverVelocityIterationCountAttr", "CreateSolverVelocityIterationCountAttr", 4),
                ("GetMaxDepenetrationVelocityAttr", "CreateMaxDepenetrationVelocityAttr", 1.5),
            ):
                attr = getattr(rigid_api, getter_name)()
                if not attr or not attr.IsValid():
                    attr = getattr(rigid_api, creator_name)()
                attr.Set(value)
            tuned_bodies += 1

            for prim in Usd.PrimRange(body_prim):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                set_collision_shell(prim, _C07_TASK_CONTACT_OFFSET)
                tuned_colliders += 1

        tuned_finger_colliders = 0
        for root_path in _DEFERRED_ROBOT_VISUAL_ROOTS[:2]:
            robot_root = self._stage.GetPrimAtPath(root_path)
            if not robot_root.IsValid():
                continue
            for prim in Usd.PrimRange(robot_root):
                if "finger" not in str(prim.GetPath()).lower():
                    continue
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                set_collision_shell(prim, _C07_FINGER_CONTACT_OFFSET)
                tuned_finger_colliders += 1

        if (
            tuned_bodies != len(body_names)
            or tuned_colliders == 0
            or tuned_finger_colliders == 0
        ):
            self._add_task_state_warning(
                "C07 anti-tunneling physics was only partially applied "
                f"({tuned_bodies}/{len(body_names)} bodies, "
                f"{tuned_colliders} task colliders, "
                f"{tuned_finger_colliders} finger colliders)"
            )
        return tuned_bodies, tuned_colliders + tuned_finger_colliders

    def _b02_mujoco_soft_hand_enabled(self) -> bool:
        return bool(
            self.task_name == "b02_nut_bolt_assembly"
            and not _bool_env("HANDSON_DISABLE_B02_MUJOCO_SOFT_HAND")
        )

    def _apply_b02_mujoco_soft_hand_actuators(self, robot_cfg: Any, side: str) -> bool:
        if not self._b02_mujoco_soft_hand_enabled() or side not in ("left", "right"):
            return False
        hand_actuator = getattr(robot_cfg, "actuators", {}).get("hand")
        if hand_actuator is None:
            return False

        side_name: Literal["left", "right"] = "left" if side == "left" else "right"
        stiffness, damping, effort = _b02_mujoco_hand_drive_maps(side_name)
        hand_actuator.stiffness = stiffness
        hand_actuator.damping = damping
        hand_actuator.effort_limit_sim = effort
        hand_actuator.friction = 0.0
        return True

    def _configure_lid_grasp_physics(self, sim_utils) -> tuple[int, bool]:
        """Tune each thin lid so it can be grasped and released reliably."""
        if self._stage is None or self.task_name not in _LID_GRASP_PHYSICS:
            return 0, False

        physics_cfg = _LID_GRASP_PHYSICS[self.task_name]

        lid_name = self._spec.target_objects[0]
        lid_prim = self._find_rigid_body_prim(lid_name)
        if lid_prim is None:
            self._add_task_state_warning(
                f"Could not tune {lid_name} grasp physics: rigid body is unavailable"
            )
            return 0, False
        lid_path = str(lid_prim.GetPath())
        # YCB and imported MJCF meshes commonly retain instanced collision
        # children.  Physics material bindings cannot be authored on instance
        # proxies, so make this one small grasp target editable first.
        sim_utils.make_uninstanceable(lid_path, stage=self._stage)
        sim_utils.update_stage()
        lid_prim = self._stage.GetPrimAtPath(lid_path)

        material_path = "/World/PhysicsMaterials/HandsOnLidGrasp"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=physics_cfg.static_friction,
            dynamic_friction=physics_cfg.dynamic_friction,
            restitution=0.0,
            friction_combine_mode=physics_cfg.friction_combine_mode,
            restitution_combine_mode="min",
        )
        material_cfg.func(material_path, material_cfg)
        sim_utils.bind_physics_material(
            lid_path,
            material_path,
            stage=self._stage,
            stronger_than_descendants=True,
        )

        from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

        bound_colliders = 0
        for prim in Usd.PrimRange(lid_prim):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if physics_cfg.contact_offset is not None:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
                if not physx_collision_api:
                    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                contact_attr = physx_collision_api.GetContactOffsetAttr()
                if not contact_attr or not contact_attr.IsValid():
                    contact_attr = physx_collision_api.CreateContactOffsetAttr()
                contact_attr.Set(float(physics_cfg.contact_offset))
                if physics_cfg.rest_offset is not None:
                    rest_attr = physx_collision_api.GetRestOffsetAttr()
                    if not rest_attr or not rest_attr.IsValid():
                        rest_attr = physx_collision_api.CreateRestOffsetAttr()
                    rest_attr.Set(float(physics_cfg.rest_offset))
            targets = (
                UsdShade.MaterialBindingAPI(prim)
                .GetDirectBindingRel("physics")
                .GetTargets()
            )
            if any(str(target) == material_path for target in targets):
                bound_colliders += 1

        mass_api = UsdPhysics.MassAPI(lid_prim)
        if not mass_api:
            mass_api = UsdPhysics.MassAPI.Apply(lid_prim)
        mass_attr = mass_api.GetMassAttr()
        if not mass_attr or not mass_attr.IsValid():
            mass_attr = mass_api.CreateMassAttr()
        mass_attr.Set(float(physics_cfg.mass))
        if bound_colliders == 0:
            self._add_task_state_warning(
                f"Could not tune {lid_name} grasp friction: no collider accepted the material"
            )
        return bound_colliders, True

    def _configure_b02_nut_grasp_physics(self, sim_utils) -> tuple[int, bool]:
        """Restore B02's source mass and usable outer grasp surface."""
        if self._stage is None or self.task_name != "b02_nut_bolt_assembly":
            return 0, False

        nut_prim = self._find_rigid_body_prim("nut")
        if nut_prim is None:
            self._add_task_state_warning(
                "Could not tune B02 nut grasp physics: rigid body is unavailable"
            )
            return 0, False

        nut_collision_root = self._find_prim("nut_object")
        if nut_collision_root is None:
            self._add_task_state_warning(
                "Could not tune B02 nut grasp physics: collision body is unavailable"
            )
            return 0, False

        nut_path = str(nut_prim.GetPath())
        nut_collision_root_path = str(nut_collision_root.GetPath())
        sim_utils.make_uninstanceable(nut_collision_root_path, stage=self._stage)
        sim_utils.update_stage()
        nut_prim = self._stage.GetPrimAtPath(nut_path)
        nut_collision_root = self._stage.GetPrimAtPath(nut_collision_root_path)

        physics_cfg = _B02_NUT_GRASP_PHYSICS
        material_path = "/World/PhysicsMaterials/HandsOnB02NutGrasp"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=physics_cfg.static_friction,
            dynamic_friction=physics_cfg.dynamic_friction,
            restitution=0.0,
            friction_combine_mode=physics_cfg.friction_combine_mode,
            restitution_combine_mode="min",
        )
        material_cfg.func(material_path, material_cfg)

        from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

        bound_colliders = 0
        disabled_inner_guards = 0
        for prim in Usd.PrimRange(nut_collision_root):
            prim_path = str(prim.GetPath())
            if (
                physics_cfg.disable_inner_finger_guard
                and "b02_nut_finger_guard" in prim_path
                and prim.HasAPI(UsdPhysics.CollisionAPI)
            ):
                collision_api = UsdPhysics.CollisionAPI(prim)
                collision_api.GetCollisionEnabledAttr().Set(False)
                disabled_inner_guards += 1
                continue
            if (
                not prim.HasAPI(UsdPhysics.CollisionAPI)
                or "b02_nut_collision_mesh_" not in prim_path
            ):
                continue
            sim_utils.bind_physics_material(
                prim_path,
                material_path,
                stage=self._stage,
                stronger_than_descendants=True,
            )
            physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
            if not physx_collision_api:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            contact_attr = physx_collision_api.GetContactOffsetAttr()
            if not contact_attr or not contact_attr.IsValid():
                contact_attr = physx_collision_api.CreateContactOffsetAttr()
            contact_attr.Set(float(physics_cfg.contact_offset))
            rest_attr = physx_collision_api.GetRestOffsetAttr()
            if not rest_attr or not rest_attr.IsValid():
                rest_attr = physx_collision_api.CreateRestOffsetAttr()
            rest_attr.Set(float(physics_cfg.rest_offset))
            targets = (
                UsdShade.MaterialBindingAPI(prim)
                .GetDirectBindingRel("physics")
                .GetTargets()
            )
            if any(str(target) == material_path for target in targets):
                bound_colliders += 1

        mass_api = UsdPhysics.MassAPI(nut_prim)
        if not mass_api:
            mass_api = UsdPhysics.MassAPI.Apply(nut_prim)
        mass_attr = mass_api.GetMassAttr()
        if not mass_attr or not mass_attr.IsValid():
            mass_attr = mass_api.CreateMassAttr()
        mass_attr.Set(float(physics_cfg.mass))

        if bound_colliders != 6:
            self._add_task_state_warning(
                "Expected 6 B02 nut grasp colliders, "
                f"but bound the PhysX material to {bound_colliders}"
            )
        if physics_cfg.disable_inner_finger_guard and disabled_inner_guards != 1:
            self._add_task_state_warning(
                "Expected to disable 1 B02 inner finger guard, "
                f"but disabled {disabled_inner_guards}"
            )
        return bound_colliders, True

    def _configure_b02_hand_grasp_physics(self, sim_utils) -> int:
        """Match the source Wuji hand friction for B02's small, flat nut."""
        if self._stage is None or self.task_name != "b02_nut_bolt_assembly":
            return 0

        physics_cfg = _B02_NUT_GRASP_PHYSICS
        material_path = "/World/PhysicsMaterials/HandsOnB02HandGrasp"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=physics_cfg.hand_static_friction,
            dynamic_friction=physics_cfg.hand_dynamic_friction,
            restitution=0.0,
            friction_combine_mode=physics_cfg.friction_combine_mode,
            restitution_combine_mode="min",
        )
        material_cfg.func(material_path, material_cfg)

        from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

        hand_material = UsdShade.Material(
            self._stage.GetPrimAtPath(material_path)
        )

        bound_links = 0
        bound_colliders = 0
        for side in ("Right", "Left"):
            root_path = f"/World/HandsOnRobots/{side}Robot"
            root_prim = self._stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                continue
            for prim in Usd.PrimRange(root_prim):
                prim_name = prim.GetName().lower()
                prim_path = str(prim.GetPath())
                normalized_path = prim_path.lower()
                is_hand_prim = (
                    "finger" in prim_name
                    or "palm" in prim_name
                    or "finger" in normalized_path
                    or "palm" in normalized_path
                )
                if not is_hand_prim:
                    continue

                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                    binding_api.Bind(
                        hand_material,
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                        materialPurpose="physics",
                    )
                    targets = binding_api.GetDirectBindingRel(
                        "physics"
                    ).GetTargets()
                    if any(str(target) == material_path for target in targets):
                        bound_links += 1

                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                    binding_api.Bind(
                        hand_material,
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                        materialPurpose="physics",
                    )
                    physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
                    if not physx_collision_api:
                        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    contact_attr = physx_collision_api.GetContactOffsetAttr()
                    if not contact_attr or not contact_attr.IsValid():
                        contact_attr = physx_collision_api.CreateContactOffsetAttr()
                    contact_attr.Set(float(physics_cfg.contact_offset))
                    rest_attr = physx_collision_api.GetRestOffsetAttr()
                    if not rest_attr or not rest_attr.IsValid():
                        rest_attr = physx_collision_api.CreateRestOffsetAttr()
                    rest_attr.Set(float(physics_cfg.rest_offset))
                    targets = binding_api.GetDirectBindingRel(
                        "physics"
                    ).GetTargets()
                    if any(str(target) == material_path for target in targets):
                        bound_colliders += 1

        if bound_links != 42:
            self._add_task_state_warning(
                "Expected to tune 42 B02 hand links, "
                f"but bound the PhysX material to {bound_links}"
            )
        if bound_colliders == 0:
            self._add_task_state_warning(
                "Could not bind B02 hand grasp material directly to any hand colliders"
            )
        return bound_links + bound_colliders

    def _configure_named_reset_stability(
        self,
        sim_utils,
        names: tuple[str, ...],
        *,
        linear_damping: float,
        angular_damping: float,
        sleep_threshold: float,
        warning_label: str,
    ) -> int:
        """Apply wakeable PhysX damping to independent task rigid bodies."""
        from pxr import PhysxSchema

        paths: list[str] = []
        for name in names:
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                self._add_task_state_warning(
                    f"Could not tune {warning_label} reset stability for {name}"
                )
                continue
            path = str(prim.GetPath())
            paths.append(path)
            sim_utils.make_uninstanceable(path, stage=self._stage)
        if not paths:
            return 0
        sim_utils.update_stage()

        tuned = 0
        for path in paths:
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not rigid_api:
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            for getter_name, creator_name, value in (
                ("GetLinearDampingAttr", "CreateLinearDampingAttr", linear_damping),
                ("GetAngularDampingAttr", "CreateAngularDampingAttr", angular_damping),
                ("GetSleepThresholdAttr", "CreateSleepThresholdAttr", sleep_threshold),
            ):
                attribute = getattr(rigid_api, getter_name)()
                if not attribute or not attribute.IsValid():
                    attribute = getattr(rigid_api, creator_name)()
                attribute.Set(value)
            tuned += 1
        return tuned

    def _configure_b02_reset_stability(self, sim_utils) -> int:
        if self._stage is None or self.task_name != "b02_nut_bolt_assembly":
            return 0
        return self._configure_named_reset_stability(
            sim_utils,
            ("nut",),
            linear_damping=_B02_RESET_LINEAR_DAMPING,
            angular_damping=_B02_RESET_ANGULAR_DAMPING,
            sleep_threshold=_B02_RESET_SLEEP_THRESHOLD,
            warning_label="B02",
        )

    def _configure_c12_reset_stability(self, sim_utils) -> int:
        if self._stage is None or self.task_name != "t11_line_up_blocks":
            return 0
        return self._configure_named_reset_stability(
            sim_utils,
            ("c12_set_square", *_C12_BLOCK_NAMES),
            linear_damping=_C12_RESET_LINEAR_DAMPING,
            angular_damping=_C12_RESET_ANGULAR_DAMPING,
            sleep_threshold=_C12_RESET_SLEEP_THRESHOLD,
            warning_label="C12",
        )

    def _configure_b01_pitcher_base_physics(self, sim_utils) -> tuple[int, bool]:
        """Keep B01's free pitcher base physical but stable for one-hand seating."""
        if self._stage is None or self.task_name not in {
            "t04_cover_pitcher",
            "c07_pitcher_pour_cup",
        }:
            return 0, False

        base_prim = self._find_rigid_body_prim("019_pitcher_base")
        if base_prim is None:
            self._add_task_state_warning(
                "Could not tune 019_pitcher_base stability: rigid body is unavailable"
            )
            return 0, False
        base_path = str(base_prim.GetPath())
        sim_utils.make_uninstanceable(base_path, stage=self._stage)
        sim_utils.update_stage()
        base_prim = self._stage.GetPrimAtPath(base_path)

        physics_cfg = _B01_PITCHER_BASE_PHYSICS
        material_path = "/World/PhysicsMaterials/HandsOnPitcherBaseStability"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=physics_cfg.static_friction,
            dynamic_friction=physics_cfg.dynamic_friction,
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="min",
        )
        material_cfg.func(material_path, material_cfg)
        sim_utils.bind_physics_material(
            base_path,
            material_path,
            stage=self._stage,
            stronger_than_descendants=True,
        )

        from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

        bound_colliders = 0
        for prim in Usd.PrimRange(base_prim):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if physics_cfg.contact_offset is not None:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
                if not physx_collision_api:
                    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                contact_attr = physx_collision_api.GetContactOffsetAttr()
                if not contact_attr or not contact_attr.IsValid():
                    contact_attr = physx_collision_api.CreateContactOffsetAttr()
                contact_attr.Set(float(physics_cfg.contact_offset))
                if physics_cfg.rest_offset is not None:
                    rest_attr = physx_collision_api.GetRestOffsetAttr()
                    if not rest_attr or not rest_attr.IsValid():
                        rest_attr = physx_collision_api.CreateRestOffsetAttr()
                    rest_attr.Set(float(physics_cfg.rest_offset))
            targets = (
                UsdShade.MaterialBindingAPI(prim)
                .GetDirectBindingRel("physics")
                .GetTargets()
            )
            if any(str(target) == material_path for target in targets):
                bound_colliders += 1

        mass_api = UsdPhysics.MassAPI(base_prim)
        if not mass_api:
            mass_api = UsdPhysics.MassAPI.Apply(base_prim)
        mass_attr = mass_api.GetMassAttr()
        if not mass_attr or not mass_attr.IsValid():
            mass_attr = mass_api.CreateMassAttr()
        mass_attr.Set(float(physics_cfg.mass))

        rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(base_prim)
        if not rigid_body_api:
            rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(base_prim)
        linear_damping_attr = rigid_body_api.GetLinearDampingAttr()
        if not linear_damping_attr or not linear_damping_attr.IsValid():
            linear_damping_attr = rigid_body_api.CreateLinearDampingAttr()
        linear_damping_attr.Set(float(physics_cfg.linear_damping))
        angular_damping_attr = rigid_body_api.GetAngularDampingAttr()
        if not angular_damping_attr or not angular_damping_attr.IsValid():
            angular_damping_attr = rigid_body_api.CreateAngularDampingAttr()
        angular_damping_attr.Set(float(physics_cfg.angular_damping))

        if bound_colliders == 0:
            self._add_task_state_warning(
                "Could not tune 019_pitcher_base friction: no collider accepted the material"
            )
        return bound_colliders, True

    def _configure_a01_reset_stability(self, sim_utils) -> int:
        """Settle A01's reset contacts without changing any other task."""
        if self._stage is None or self.task_name != "t26_place_fruit_in_bucket":
            return 0

        from pxr import PhysxSchema, Usd, UsdPhysics

        tuned = 0
        for name in _A01_RESET_STABILITY_BODIES:
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                self._add_task_state_warning(
                    f"Could not tune A01 reset stability: {name} rigid body is unavailable"
                )
                continue
            path = str(prim.GetPath())
            sim_utils.make_uninstanceable(path, stage=self._stage)
            sim_utils.update_stage()
            prim = self._stage.GetPrimAtPath(path)
            for collision_prim in Usd.PrimRange(prim):
                if not collision_prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                collision_api = PhysxSchema.PhysxCollisionAPI(collision_prim)
                if not collision_api:
                    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(
                        collision_prim
                    )
                collision_api.CreateContactOffsetAttr().Set(
                    float(_A01_CONTACT_OFFSET)
                )
                collision_api.CreateRestOffsetAttr().Set(
                    float(_A01_REST_OFFSET)
                )
            rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not rigid_api:
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            for getter_name, creator_name, value in (
                ("GetLinearDampingAttr", "CreateLinearDampingAttr", _A01_RESET_LINEAR_DAMPING),
                ("GetAngularDampingAttr", "CreateAngularDampingAttr", _A01_RESET_ANGULAR_DAMPING),
                ("GetSleepThresholdAttr", "CreateSleepThresholdAttr", _A01_RESET_SLEEP_THRESHOLD),
            ):
                attribute = getattr(rigid_api, getter_name)()
                if not attribute or not attribute.IsValid():
                    attribute = getattr(rigid_api, creator_name)()
                attribute.Set(value)
            tuned += 1
        return tuned

    def _configure_a03_reset_stability(self, sim_utils) -> int:
        """Model A03 rolling resistance so PhysX reaches its sleep state."""
        if self._stage is None or self.task_name != "t24_place_ball_in_bowl":
            return 0

        from pxr import PhysxSchema

        tuned = 0
        for name in _A03_RESET_STABILITY_BODIES:
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                self._add_task_state_warning(
                    f"Could not tune A03 reset stability: {name} rigid body is unavailable"
                )
                continue
            path = str(prim.GetPath())
            sim_utils.make_uninstanceable(path, stage=self._stage)
            rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not rigid_api:
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            for getter_name, creator_name, value in (
                (
                    "GetLinearDampingAttr",
                    "CreateLinearDampingAttr",
                    _A03_RESET_LINEAR_DAMPING,
                ),
                (
                    "GetAngularDampingAttr",
                    "CreateAngularDampingAttr",
                    _A03_RESET_ANGULAR_DAMPING,
                ),
                (
                    "GetSleepThresholdAttr",
                    "CreateSleepThresholdAttr",
                    _A03_RESET_SLEEP_THRESHOLD,
                ),
            ):
                attribute = getattr(rigid_api, getter_name)()
                if not attribute or not attribute.IsValid():
                    attribute = getattr(rigid_api, creator_name)()
                attribute.Set(value)
            tuned += 1
        return tuned

    def _configure_a06_reset_stability(self, sim_utils) -> int:
        """Restore A06's source rigid body and deterministic contact shell."""
        if self._stage is None or self.task_name != "t19_orient_cup_upright":
            return 0

        from pxr import Gf, PhysxSchema, Usd, UsdPhysics

        tuned = 0
        for name in _A06_RESET_STABILITY_BODIES:
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                self._add_task_state_warning(
                    f"Could not tune A06 reset stability: {name} rigid body is unavailable"
                )
                continue
            path = str(prim.GetPath())
            sim_utils.make_uninstanceable(path, stage=self._stage)
            sim_utils.update_stage()
            prim = self._stage.GetPrimAtPath(path)

            # The MJCF importer drops the per-geom masses on this 27-piece
            # compound mesh.  Restore the compiled MuJoCo body inertial so
            # every reset seed has the same center of mass and angular
            # response as the source task.
            mass_api = UsdPhysics.MassAPI(prim)
            if not mass_api:
                mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr().Set(float(_A06_SOURCE_MASS))
            mass_api.CreateCenterOfMassAttr().Set(
                Gf.Vec3f(*_A06_SOURCE_CENTER_OF_MASS)
            )
            mass_api.CreateDiagonalInertiaAttr().Set(
                Gf.Vec3f(*_A06_SOURCE_DIAGONAL_INERTIA)
            )
            principal_w, principal_x, principal_y, principal_z = (
                _A06_SOURCE_PRINCIPAL_AXES_WXYZ
            )
            mass_api.CreatePrincipalAxesAttr().Set(
                Gf.Quatf(
                    principal_w,
                    Gf.Vec3f(principal_x, principal_y, principal_z),
                )
            )

            # PhysX otherwise derives a different default contact offset for
            # each convex piece from its individual bounds.  One shared shell
            # prevents the three-point table support from switching between
            # collider-specific separation distances.
            for collision_prim in Usd.PrimRange(prim):
                if not collision_prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                collision_api = PhysxSchema.PhysxCollisionAPI(collision_prim)
                if not collision_api:
                    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(
                        collision_prim
                    )
                collision_api.CreateContactOffsetAttr().Set(
                    float(_A06_CONTACT_OFFSET)
                )
                collision_api.CreateRestOffsetAttr().Set(float(_A06_REST_OFFSET))

            rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not rigid_api:
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            for getter_name, creator_name, value in (
                (
                    "GetLinearDampingAttr",
                    "CreateLinearDampingAttr",
                    _A06_RESET_LINEAR_DAMPING,
                ),
                (
                    "GetAngularDampingAttr",
                    "CreateAngularDampingAttr",
                    _A06_RESET_ANGULAR_DAMPING,
                ),
                (
                    "GetSleepThresholdAttr",
                    "CreateSleepThresholdAttr",
                    _A06_RESET_SLEEP_THRESHOLD,
                ),
            ):
                attribute = getattr(rigid_api, getter_name)()
                if not attribute or not attribute.IsValid():
                    attribute = getattr(rigid_api, creator_name)()
                attribute.Set(value)
            tuned += 1
        return tuned

    def _configure_b04_stack_physics(self, sim_utils) -> tuple[int, int]:
        """Match B04's source friction and damping so a released tower settles."""
        if self._stage is None or self.task_name != "t44_stack_blocks":
            return 0, 0

        block_paths: dict[str, str] = {}
        for name in _STACK_BLOCK_HALF_EXTENTS:
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                self._add_task_state_warning(
                    f"Could not tune B04 block physics: {name} rigid body is unavailable"
                )
                continue
            path = str(prim.GetPath())
            block_paths[name] = path
            sim_utils.make_uninstanceable(path, stage=self._stage)
        if not block_paths:
            return 0, 0
        sim_utils.update_stage()

        material_cfgs = {
            "base": sim_utils.RigidBodyMaterialCfg(
                static_friction=_STACK_BASE_STATIC_FRICTION,
                dynamic_friction=_STACK_BASE_DYNAMIC_FRICTION,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
            "small": sim_utils.RigidBodyMaterialCfg(
                static_friction=_STACK_SMALL_STATIC_FRICTION,
                dynamic_friction=_STACK_SMALL_DYNAMIC_FRICTION,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
        }
        material_paths = {
            "base": "/World/PhysicsMaterials/HandsOnB04BaseStability",
            "small": "/World/PhysicsMaterials/HandsOnB04SmallBlockStability",
        }
        for key, cfg in material_cfgs.items():
            cfg.func(material_paths[key], cfg)

        from pxr import PhysxSchema, Usd, UsdPhysics

        tuned_colliders = 0
        tuned_bodies = 0
        for name, path in block_paths.items():
            prim = self._stage.GetPrimAtPath(path)
            material_key = "base" if name == _STACK_BOTTOM_BLOCK_NAME else "small"
            sim_utils.bind_physics_material(
                path,
                material_paths[material_key],
                stage=self._stage,
                stronger_than_descendants=True,
            )
            for descendant in Usd.PrimRange(prim):
                if not descendant.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                collision_api = PhysxSchema.PhysxCollisionAPI(descendant)
                if not collision_api:
                    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(descendant)
                contact_attr = collision_api.GetContactOffsetAttr()
                if not contact_attr or not contact_attr.IsValid():
                    contact_attr = collision_api.CreateContactOffsetAttr()
                contact_attr.Set(_STACK_CONTACT_OFFSET)
                rest_attr = collision_api.GetRestOffsetAttr()
                if not rest_attr or not rest_attr.IsValid():
                    rest_attr = collision_api.CreateRestOffsetAttr()
                rest_attr.Set(_STACK_REST_OFFSET)
                tuned_colliders += 1

            rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not rigid_api:
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            linear_attr = rigid_api.GetLinearDampingAttr()
            if not linear_attr or not linear_attr.IsValid():
                linear_attr = rigid_api.CreateLinearDampingAttr()
            linear_attr.Set(_STACK_LINEAR_DAMPING)
            angular_attr = rigid_api.GetAngularDampingAttr()
            if not angular_attr or not angular_attr.IsValid():
                angular_attr = rigid_api.CreateAngularDampingAttr()
            angular_attr.Set(_STACK_ANGULAR_DAMPING)
            sleep_attr = rigid_api.GetSleepThresholdAttr()
            if not sleep_attr or not sleep_attr.IsValid():
                sleep_attr = rigid_api.CreateSleepThresholdAttr()
            sleep_attr.Set(_STACK_SLEEP_THRESHOLD)
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                contact_report_api = PhysxSchema.PhysxContactReportAPI.Get(
                    self._stage,
                    prim.GetPath(),
                )
            else:
                contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            contact_report_api.CreateThresholdAttr().Set(0.0)
            tuned_bodies += 1

        if tuned_colliders == 0:
            self._add_task_state_warning(
                "Could not tune B04 block friction: no colliders were found"
            )
        return tuned_colliders, tuned_bodies

    def _configure_c01_wood_physics(self) -> bool:
        """Match the compact C01 block's mass to the graspable MuJoCo task."""
        if self._stage is None or self.task_name != "t08_hammer_nail":
            return False
        wood_prim = self._find_rigid_body_prim("wood")
        if wood_prim is None:
            self._add_task_state_warning(
                "Could not tune C01 wood mass: rigid body is unavailable"
            )
            return False

        from pxr import UsdPhysics

        mass_api = UsdPhysics.MassAPI(wood_prim)
        if not mass_api:
            mass_api = UsdPhysics.MassAPI.Apply(wood_prim)
        mass_attr = mass_api.GetMassAttr()
        if not mass_attr or not mass_attr.IsValid():
            mass_attr = mass_api.CreateMassAttr()
        mass_attr.Set(_C01_WOOD_MASS)
        return True

    def _configure_t08_hammer_nail_physics(self, sim_utils: Any) -> bool:
        """Make C01 impacts physical, insertable, and resistant to tunneling."""
        if self._stage is None or self.task_name != "t08_hammer_nail":
            return False

        from pxr import PhysxSchema, Usd, UsdPhysics

        # MJCF collision groups are commonly imported as instance proxies.
        # Deinstance only the hammer and wood/nail roots before authoring CCD
        # and per-shape contact shells.
        task_paths: dict[str, str] = {}
        for object_name in ("hammer", "nail"):
            prim = self._find_prim(object_name)
            if prim is None or not prim.IsValid():
                continue
            top_prim = prim
            while top_prim.GetParent().IsValid() and str(
                top_prim.GetParent().GetPath()
            ) != _TASK_ROOT:
                top_prim = top_prim.GetParent()
            top_path = str(top_prim.GetPath())
            if top_path.startswith(f"{_TASK_ROOT}/"):
                task_paths[object_name] = top_path

        for path in sorted(set(task_paths.values())):
            sim_utils.make_uninstanceable(path, stage=self._stage)
        if task_paths:
            sim_utils.update_stage()
            # Cached prim handles may refer to the old instance proxies.
            for object_name in ("hammer", "nail"):
                self._prim_cache.pop(object_name, None)

        task_root = self._stage.GetPrimAtPath(_TASK_ROOT)
        nail_joint = next(
            (
                prim
                for prim in Usd.PrimRange(task_root)
                if prim.GetName() == "nail_slide_joint"
                and prim.IsA(UsdPhysics.PrismaticJoint)
            ),
            None,
        )
        if nail_joint is None:
            self._add_task_state_warning(
                "Could not tune C01 nail slide joint: joint is unavailable"
            )
            return False

        drive = UsdPhysics.DriveAPI.Get(nail_joint, "linear")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(nail_joint, "linear")
        drive.GetTargetPositionAttr().Set(0.0)
        drive.GetTargetVelocityAttr().Set(0.0)
        drive.GetStiffnessAttr().Set(_C01_NAIL_JOINT_STIFFNESS)
        drive.GetDampingAttr().Set(_C01_NAIL_JOINT_DAMPING)
        drive.GetMaxForceAttr().Set(_C01_NAIL_JOINT_MAX_FORCE)
        nail_joint.GetAttribute("physxJoint:jointFriction").Set(
            _C01_NAIL_JOINT_FRICTION
        )

        if nail_joint.HasAPI(PhysxSchema.JointStateAPI, "linear"):
            joint_state = PhysxSchema.JointStateAPI.Get(nail_joint, "linear")
        else:
            joint_state = PhysxSchema.JointStateAPI.Apply(nail_joint, "linear")
        joint_state.GetPositionAttr().Set(0.0)
        joint_state.GetVelocityAttr().Set(0.0)

        hammer_prim = self._find_rigid_body_prim("hammer")
        nail_prim = self._find_rigid_body_prim("nail")
        if hammer_prim is None or nail_prim is None:
            self._add_task_state_warning(
                "Could not tune C01 hammer/nail bodies: a rigid body is unavailable"
            )
            return False
        for prim, mass in (
            (hammer_prim, _C01_HAMMER_MASS),
            (nail_prim, _C01_NAIL_MASS),
        ):
            mass_api = UsdPhysics.MassAPI(prim)
            if not mass_api:
                mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_attr = mass_api.GetMassAttr()
            if not mass_attr or not mass_attr.IsValid():
                mass_attr = mass_api.CreateMassAttr()
            mass_attr.Set(float(mass))

        tuned_bodies = 0
        tuned_colliders = 0
        for object_name, root_path in task_paths.items():
            root_prim = self._stage.GetPrimAtPath(root_path)
            contact_offset = (
                _C01_HAMMER_CONTACT_OFFSET
                if object_name == "hammer"
                else _C01_NAIL_CONTACT_OFFSET
            )
            for prim in Usd.PrimRange(root_prim):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
                    if not rigid_api:
                        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    for getter_name, creator_name, value in (
                        (
                            "GetEnableCCDAttr",
                            "CreateEnableCCDAttr",
                            _C01_ENABLE_CCD,
                        ),
                        (
                            "GetEnableSpeculativeCCDAttr",
                            "CreateEnableSpeculativeCCDAttr",
                            _C01_ENABLE_SPECULATIVE_CCD,
                        ),
                        (
                            "GetSolverPositionIterationCountAttr",
                            "CreateSolverPositionIterationCountAttr",
                            _C01_SOLVER_POSITION_ITERATIONS,
                        ),
                        (
                            "GetSolverVelocityIterationCountAttr",
                            "CreateSolverVelocityIterationCountAttr",
                            _C01_SOLVER_VELOCITY_ITERATIONS,
                        ),
                        (
                            "GetMaxDepenetrationVelocityAttr",
                            "CreateMaxDepenetrationVelocityAttr",
                            _C01_MAX_DEPENETRATION_VELOCITY,
                        ),
                    ):
                        attr = getattr(rigid_api, getter_name)()
                        if not attr or not attr.IsValid():
                            attr = getattr(rigid_api, creator_name)()
                        attr.Set(value)
                    tuned_bodies += 1

                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                collision_api = PhysxSchema.PhysxCollisionAPI(prim)
                if not collision_api:
                    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                contact_attr = collision_api.GetContactOffsetAttr()
                if not contact_attr or not contact_attr.IsValid():
                    contact_attr = collision_api.CreateContactOffsetAttr()
                contact_attr.Set(contact_offset)
                rest_attr = collision_api.GetRestOffsetAttr()
                if not rest_attr or not rest_attr.IsValid():
                    rest_attr = collision_api.CreateRestOffsetAttr()
                rest_attr.Set(_C01_CONTACT_REST_OFFSET)
                tuned_colliders += 1

        if tuned_bodies < 2 or tuned_colliders == 0:
            self._add_task_state_warning(
                "C01 hammer/nail anti-tunneling physics was only partially applied "
                f"({tuned_bodies} bodies, {tuned_colliders} colliders)"
            )
        # Tensor contact reporting has differed by sensor/filter orientation
        # across Isaac Sim releases.  Author both bodies as report sources so
        # the runtime can query the pair in both directions.
        for report_prim in (hammer_prim, nail_prim):
            if report_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                report_api = PhysxSchema.PhysxContactReportAPI.Get(
                    self._stage,
                    report_prim.GetPath(),
                )
            else:
                report_api = PhysxSchema.PhysxContactReportAPI.Apply(report_prim)
            report_api.CreateThresholdAttr().Set(0.0)
        return True

    def _configure_c06_camera_grasp_physics(self, sim_utils) -> tuple[int, bool]:
        """Make the articulated C06 camera light and grippy enough to lift."""
        if self._stage is None or self.task_name != "t23_photograph_logo":
            return 0, False

        camera_prim = self._find_rigid_body_prim("camera")
        if camera_prim is None:
            self._add_task_state_warning(
                "Could not tune C06 camera grasp physics: rigid body is unavailable"
            )
            return 0, False

        physics_cfg = _C06_CAMERA_GRASP_PHYSICS
        camera_path = str(camera_prim.GetPath())
        collisions_path = f"{camera_path}/collisions"
        collisions_prim = self._stage.GetPrimAtPath(collisions_path)
        if not collisions_prim.IsValid():
            self._add_task_state_warning(
                "Could not tune C06 camera grasp physics: collision root is unavailable"
            )
            return 0, False

        # MJCF collision groups are imported as instance proxies.  Deinstance
        # only this four-shape subtree so material and helper-collision edits
        # remain narrow and the large camera visual asset stays instanced.
        sim_utils.make_uninstanceable(collisions_path, stage=self._stage)
        sim_utils.update_stage()
        camera_prim = self._stage.GetPrimAtPath(camera_path)
        collisions_prim = self._stage.GetPrimAtPath(collisions_path)

        material_path = "/World/PhysicsMaterials/HandsOnC06CameraGrasp"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=physics_cfg.static_friction,
            dynamic_friction=physics_cfg.dynamic_friction,
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="min",
        )
        material_cfg.func(material_path, material_cfg)

        from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

        bound_colliders = 0
        disabled_helpers = 0
        for prim in Usd.PrimRange(collisions_prim):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            prim_name = prim.GetName()
            if prim_name in {"camera_body_top_cap", "camera_shutter_touch_zone"}:
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                disabled_helpers += 1
                continue
            if prim_name not in {"camera_body_collision", "camera_lens_collision"}:
                continue

            sim_utils.bind_physics_material(
                str(prim.GetPath()),
                material_path,
                stage=self._stage,
                stronger_than_descendants=True,
            )
            physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
            if not physx_collision_api:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            contact_attr = physx_collision_api.GetContactOffsetAttr()
            if not contact_attr or not contact_attr.IsValid():
                contact_attr = physx_collision_api.CreateContactOffsetAttr()
            contact_attr.Set(float(physics_cfg.contact_offset))
            rest_attr = physx_collision_api.GetRestOffsetAttr()
            if not rest_attr or not rest_attr.IsValid():
                rest_attr = physx_collision_api.CreateRestOffsetAttr()
            rest_attr.Set(float(physics_cfg.rest_offset))
            targets = (
                UsdShade.MaterialBindingAPI(prim)
                .GetDirectBindingRel("physics")
                .GetTargets()
            )
            if any(str(target) == material_path for target in targets):
                bound_colliders += 1

        mass_api = UsdPhysics.MassAPI(camera_prim)
        if not mass_api:
            mass_api = UsdPhysics.MassAPI.Apply(camera_prim)
        mass_attr = mass_api.GetMassAttr()
        if not mass_attr or not mass_attr.IsValid():
            mass_attr = mass_api.CreateMassAttr()
        mass_attr.Set(float(physics_cfg.mass))

        rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(camera_prim)
        if not rigid_body_api:
            rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(camera_prim)
        linear_damping_attr = rigid_body_api.GetLinearDampingAttr()
        if not linear_damping_attr or not linear_damping_attr.IsValid():
            linear_damping_attr = rigid_body_api.CreateLinearDampingAttr()
        linear_damping_attr.Set(float(physics_cfg.linear_damping))
        angular_damping_attr = rigid_body_api.GetAngularDampingAttr()
        if not angular_damping_attr or not angular_damping_attr.IsValid():
            angular_damping_attr = rigid_body_api.CreateAngularDampingAttr()
        angular_damping_attr.Set(float(physics_cfg.angular_damping))

        if bound_colliders == 0:
            self._add_task_state_warning(
                "Could not tune C06 camera friction: no collider accepted the material"
            )
        if disabled_helpers != 2:
            self._add_task_state_warning(
                "Expected to disable 2 C06 transparent helper colliders, "
                f"but disabled {disabled_helpers}"
            )
        return bound_colliders, True

    def _disable_imported_task_robots(self) -> int:
        if self._stage is None:
            return 0
        from pxr import Usd, UsdPhysics

        imported_roots = tuple(f"{_TASK_ROOT}/base_{side}" for side in ("right", "left"))
        task_root = self._stage.GetPrimAtPath(_TASK_ROOT)
        disabled = 0
        for prim in list(Usd.PrimRange(task_root)):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            targets = (
                list(joint.GetBody0Rel().GetTargets())
                + list(joint.GetBody1Rel().GetTargets())
            )
            if any(str(target).startswith(imported_roots) for target in targets):
                prim.SetActive(False)
                disabled += 1
        for root_path in imported_roots:
            imported_robot = self._stage.GetPrimAtPath(root_path)
            if imported_robot.IsValid() and imported_robot.IsActive():
                imported_robot.SetActive(False)
                disabled += 1
        return disabled

    def _hide_camera_debug_helpers(self) -> int:
        if self._stage is None:
            return 0
        from pxr import Usd, UsdGeom

        world = self._stage.GetPrimAtPath("/World")
        hidden = 0
        for prim in Usd.PrimRange(world):
            name = prim.GetName()
            if (
                name not in _CAMERA_DEBUG_PRIM_NAMES
                and not name.startswith("head_main_fov_")
            ):
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
                hidden += 1
        return hidden

    def _configure_c04_trigger_physics(self) -> bool:
        if self.task_name != "t50_water_plant" or self._stage is None:
            return False

        from pxr import PhysxSchema, UsdPhysics

        joint = self._stage.GetPrimAtPath(f"{_TASK_ROOT}/joints/joint_0")
        if not joint.IsValid():
            self._add_task_state_warning("Could not find the C04 trigger joint")
            return False

        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.GetStiffnessAttr().Set(_C04_TRIGGER_STIFFNESS)
        drive.GetDampingAttr().Set(_C04_TRIGGER_DAMPING)
        drive.GetMaxForceAttr().Set(_C04_TRIGGER_MAX_FORCE)
        drive.GetTargetPositionAttr().Set(0.0)

        joint.GetAttribute("physxJoint:jointFriction").Set(_C04_TRIGGER_FRICTION)
        joint.GetAttribute("physxLimit:X:stiffness").Set(_C04_TRIGGER_STIFFNESS)
        joint.GetAttribute("physxLimit:X:damping").Set(_C04_TRIGGER_DAMPING)

        for path in (
            f"{_TASK_ROOT}/link_2/link_0",
            f"{_TASK_ROOT}/link_2/link_1",
            f"{_TASK_ROOT}/link_2/link_2",
        ):
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not rigid_api:
                rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            for getter_name, creator_name, value in (
                (
                    "GetLinearDampingAttr",
                    "CreateLinearDampingAttr",
                    _C04_RESET_LINEAR_DAMPING,
                ),
                (
                    "GetAngularDampingAttr",
                    "CreateAngularDampingAttr",
                    _C04_RESET_ANGULAR_DAMPING,
                ),
                (
                    "GetSleepThresholdAttr",
                    "CreateSleepThresholdAttr",
                    _C04_RESET_SLEEP_THRESHOLD,
                ),
            ):
                attribute = getattr(rigid_api, getter_name)()
                if not attribute or not attribute.IsValid():
                    attribute = getattr(rigid_api, creator_name)()
                attribute.Set(value)
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                report_api = PhysxSchema.PhysxContactReportAPI.Get(
                    self._stage,
                    prim.GetPath(),
                )
            else:
                report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report_api.CreateThresholdAttr().Set(0.0)
        return True

    def _configure_c05_mouse_button_physics(self) -> int:
        if self.task_name != "t14_move_and_click_mouse" or self._stage is None:
            return 0

        from pxr import PhysxSchema, UsdPhysics

        configured = 0
        for name in ("mouse_joint_0", "mouse_joint_1"):
            joint = self._stage.GetPrimAtPath(f"{_TASK_ROOT}/joints/{name}")
            if not joint.IsValid():
                self._add_task_state_warning(f"Could not find C05 button joint {name}")
                continue

            drive = UsdPhysics.DriveAPI.Get(joint, "linear")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(joint, "linear")
            drive.GetTargetPositionAttr().Set(0.0)
            drive.GetStiffnessAttr().Set(_C05_BUTTON_STIFFNESS)
            drive.GetDampingAttr().Set(_C05_BUTTON_DAMPING)
            drive.GetMaxForceAttr().Set(_C05_BUTTON_MAX_FORCE)
            joint.GetAttribute("physxJoint:jointFriction").Set(
                _C05_BUTTON_FRICTION
            )
            joint.GetAttribute("physxLimit:X:stiffness").Set(
                _C05_BUTTON_STIFFNESS
            )
            joint.GetAttribute("physxLimit:X:damping").Set(
                _C05_BUTTON_DAMPING
            )

            if joint.HasAPI(PhysxSchema.JointStateAPI, "linear"):
                state = PhysxSchema.JointStateAPI.Get(joint, "linear")
            else:
                state = PhysxSchema.JointStateAPI.Apply(joint, "linear")
            state.GetPositionAttr().Set(0.0)
            state.GetVelocityAttr().Set(0.0)
            configured += 1
        return configured

    def _configure_c06_shutter_physics(self) -> bool:
        if self.task_name != "t23_photograph_logo" or self._stage is None:
            return False

        from pxr import PhysxSchema, UsdPhysics

        joint = self._stage.GetPrimAtPath(
            f"{_TASK_ROOT}/joints/{_C06_SHUTTER_JOINT}"
        )
        if not joint.IsValid():
            self._add_task_state_warning("Could not find the C06 shutter joint")
            return False

        drive = UsdPhysics.DriveAPI.Get(joint, "linear")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint, "linear")
        drive.GetTargetPositionAttr().Set(0.0)
        drive.GetTargetVelocityAttr().Set(0.0)
        drive.GetStiffnessAttr().Set(_C06_SHUTTER_STIFFNESS)
        drive.GetDampingAttr().Set(_C06_SHUTTER_DAMPING)
        drive.GetMaxForceAttr().Set(_C06_SHUTTER_MAX_FORCE)
        joint.GetAttribute("physxJoint:jointFriction").Set(
            _C06_SHUTTER_FRICTION
        )
        joint.GetAttribute("physxLimit:X:stiffness").Set(
            _C06_SHUTTER_STIFFNESS
        )
        joint.GetAttribute("physxLimit:X:damping").Set(
            _C06_SHUTTER_DAMPING
        )

        if joint.HasAPI(PhysxSchema.JointStateAPI, "linear"):
            state = PhysxSchema.JointStateAPI.Get(joint, "linear")
        else:
            state = PhysxSchema.JointStateAPI.Apply(joint, "linear")
        state.GetPositionAttr().Set(0.0)
        state.GetVelocityAttr().Set(0.0)

        shutter_body = self._stage.GetPrimAtPath(f"{_TASK_ROOT}/link_19/link_7")
        if not shutter_body.IsValid():
            self._add_task_state_warning("Could not find the C06 shutter body")
            return False
        if shutter_body.HasAPI(PhysxSchema.PhysxContactReportAPI):
            report_api = PhysxSchema.PhysxContactReportAPI.Get(
                self._stage,
                shutter_body.GetPath(),
            )
        else:
            report_api = PhysxSchema.PhysxContactReportAPI.Apply(shutter_body)
        report_api.CreateThresholdAttr().Set(0.0)
        return True

    def _configure_c08_tongs_spring_physics(self) -> bool:
        """Make the food tongs rest open and spring back after release."""
        if self.task_name != "c08_tongs_ice_cubes" or self._stage is None:
            return False

        from pxr import PhysxSchema, Usd, UsdPhysics

        task_root = self._stage.GetPrimAtPath(_TASK_ROOT)
        joint = next(
            (
                prim
                for prim in Usd.PrimRange(task_root)
                if prim.GetName() == "c08_tongs_joint"
                and prim.IsA(UsdPhysics.RevoluteJoint)
            ),
            None,
        )
        if joint is None:
            self._add_task_state_warning(
                "Could not configure C08 tongs spring: hinge joint is unavailable"
            )
            return False

        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        rest_degrees = math.degrees(_C08_TONGS_REST_RAD)
        drive.GetTargetPositionAttr().Set(rest_degrees)
        drive.GetTargetVelocityAttr().Set(0.0)
        drive.GetStiffnessAttr().Set(_C08_TONGS_SPRING_STIFFNESS)
        drive.GetDampingAttr().Set(_C08_TONGS_SPRING_DAMPING)
        drive.GetMaxForceAttr().Set(_C08_TONGS_SPRING_MAX_FORCE)
        joint.GetAttribute("physxJoint:jointFriction").Set(
            _C08_TONGS_JOINT_FRICTION
        )

        if joint.HasAPI(PhysxSchema.JointStateAPI, "angular"):
            state = PhysxSchema.JointStateAPI.Get(joint, "angular")
        else:
            state = PhysxSchema.JointStateAPI.Apply(joint, "angular")
        state.GetPositionAttr().Set(rest_degrees)
        state.GetVelocityAttr().Set(0.0)
        return True

    def _initialize_c08_tongs_open_state(self) -> bool:
        """Force the live and cached reset states to the spring rest angle."""
        if self.task_name != "c08_tongs_ice_cubes" or self._torch is None:
            return False
        dof_state = self._task_dof_views.get("c08_tongs_joint")
        if dof_state is None:
            self._add_task_state_warning(
                "Could not initialize C08 tongs open: DOF view is unavailable"
            )
            return False

        view, dof_index, _initial_position = dof_state
        try:
            positions = view.get_dof_positions().clone()
            velocities = view.get_dof_velocities().clone()
            positions[:, dof_index] = _C08_TONGS_REST_RAD
            velocities[:, dof_index] = 0.0
            indices = self._torch.arange(
                int(view.count),
                dtype=self._torch.long,
                device=self._isaac_device,
            )
            view.set_dof_positions(positions, indices)
            view.set_dof_velocities(velocities, indices)
            for (
                _path,
                cached_view,
                _indices,
                _root_pose,
                _root_velocity,
                cached_positions,
                cached_velocities,
            ) in self._task_reset_articulation_states:
                if cached_view is not view:
                    continue
                cached_positions[:, dof_index] = _C08_TONGS_REST_RAD
                cached_velocities[:, dof_index] = 0.0
            self._task_dof_views["c08_tongs_joint"] = (
                view,
                dof_index,
                _C08_TONGS_REST_RAD,
            )
            print(
                "[HandsOn/IsaacLab] C08 tongs spring initialized open "
                f"at {_C08_TONGS_REST_RAD:.3f} rad",
                flush=True,
            )
            return True
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not initialize C08 tongs open state: {exc}"
            )
            return False

    def _configure_c11_contact_reporting(self) -> int:
        if self.task_name != "c11_xylophone_sequence" or self._stage is None:
            return 0

        from pxr import PhysxSchema

        configured = 0
        for name in ("c11_mallet", *_C11_KEY_NAMES):
            body_prim = self._find_rigid_body_prim(name)
            if body_prim is None:
                self._add_task_state_warning(
                    f"Could not configure C11 contact reporting for {name}"
                )
                continue
            if body_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                report_api = PhysxSchema.PhysxContactReportAPI.Get(
                    self._stage,
                    body_prim.GetPath(),
                )
            else:
                report_api = PhysxSchema.PhysxContactReportAPI.Apply(body_prim)
            report_api.CreateThresholdAttr().Set(0.0)
            configured += 1
        return configured

    def _configure_c12_contact_reporting(self) -> int:
        if self.task_name != "t11_line_up_blocks" or self._stage is None:
            return 0

        from pxr import PhysxSchema

        configured = 0
        for name in _C12_BLOCK_NAMES:
            block_prim = self._find_rigid_body_prim(name)
            if block_prim is None:
                self._add_task_state_warning(
                    f"Could not configure C12 contact reporting for {name}"
                )
                continue
            if block_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                report_api = PhysxSchema.PhysxContactReportAPI.Get(
                    self._stage,
                    block_prim.GetPath(),
                )
            else:
                report_api = PhysxSchema.PhysxContactReportAPI.Apply(block_prim)
            report_api.CreateThresholdAttr().Set(0.0)
            configured += 1
        return configured

    def _c04_right_hand_body_paths(self) -> list[str]:
        if self._stage is None:
            return []

        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            "/World/HandsOnRobots/RightRobot"
        )
        paths: list[str] = []
        for prim in Usd.PrimRange(root):
            name = prim.GetName()
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if name == "right_palm_link" or name.startswith("right_finger"):
                paths.append(str(prim.GetPath()))
        return paths

    def _init_b04_contact_views(self) -> None:
        """Report actual right-hand contacts for every B04 block."""
        self._b04_block_contact_views = {}
        if self.task_name != "t44_stack_blocks" or not self._robots:
            return
        physics_sim_view = self._robots["right"]._physics_sim_view
        hand_paths = self._c04_right_hand_body_paths()
        if physics_sim_view is None or not hand_paths:
            self._add_task_state_warning(
                "B04 right-hand contact reporting is unavailable"
            )
            return

        max_contacts = max(64, len(hand_paths) * 8)
        try:
            for name in _STACK_BLOCK_HALF_EXTENTS:
                block_prim = self._find_rigid_body_prim(name)
                if block_prim is None:
                    continue
                view = physics_sim_view.create_rigid_contact_view(
                    str(block_prim.GetPath()),
                    filter_patterns=hand_paths,
                    max_contact_data_count=max_contacts,
                )
                if (
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                ):
                    self._b04_block_contact_views[name] = view
        except Exception as exc:
            self._b04_block_contact_views = {}
            self._add_task_state_warning(
                f"Could not initialize B04 right-hand contact reporting: {exc}"
            )
            return

        if len(self._b04_block_contact_views) != len(_STACK_BLOCK_HALF_EXTENTS):
            self._b04_block_contact_views = {}
            self._add_task_state_warning(
                "B04 contact reporting did not resolve all four block bodies"
            )

    def _init_t08_hammer_nail_contact_view(self) -> None:
        """Report physical hammer-body contacts against the articulated nail."""
        self._t08_hammer_nail_contact_view = None
        self._t08_hammer_nail_contact_views = []
        if self.task_name != "t08_hammer_nail" or not self._robots:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        hammer_prim = self._find_rigid_body_prim("hammer")
        nail_prim = self._find_rigid_body_prim("nail")
        if physics_sim_view is None or hammer_prim is None or nail_prim is None:
            self._add_task_state_warning(
                "C01 hammer/nail contact reporting is unavailable"
            )
            return
        errors: list[str] = []
        for sensor_prim, filter_prim in (
            (hammer_prim, nail_prim),
            (nail_prim, hammer_prim),
        ):
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    str(sensor_prim.GetPath()),
                    filter_patterns=[str(filter_prim.GetPath())],
                    max_contact_data_count=64,
                )
                if (
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                ):
                    self._t08_hammer_nail_contact_views.append(view)
                else:
                    errors.append(
                        f"{sensor_prim.GetName()} sensor has no sensors or filters"
                    )
            except Exception as exc:
                errors.append(f"{sensor_prim.GetName()} sensor: {exc}")

        if self._t08_hammer_nail_contact_views:
            # Keep the original field as a compatibility alias for callers
            # that inspect the first contact view.
            self._t08_hammer_nail_contact_view = (
                self._t08_hammer_nail_contact_views[0]
            )
        else:
            detail = "; ".join(errors) if errors else "no contact views resolved"
            self._add_task_state_warning(
                f"Could not initialize C01 hammer/nail contact reporting: {detail}"
            )

    def _init_c04_contact_views(self) -> None:
        self._c04_trigger_contact_view = None
        self._c04_grasp_contact_views = []
        if self.task_name != "t50_water_plant" or not self._robots:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        hand_paths = self._c04_right_hand_body_paths()
        if physics_sim_view is None or not hand_paths:
            self._add_task_state_warning("C04 hand contact reporting is unavailable")
            return

        max_contacts = max(64, len(hand_paths) * 8)
        try:
            self._c04_trigger_contact_view = physics_sim_view.create_rigid_contact_view(
                f"{_TASK_ROOT}/link_2/link_0",
                filter_patterns=hand_paths,
                max_contact_data_count=max_contacts,
            )
            for path in (
                f"{_TASK_ROOT}/link_2/link_1",
                f"{_TASK_ROOT}/link_2/link_2",
            ):
                view = physics_sim_view.create_rigid_contact_view(
                    path,
                    filter_patterns=hand_paths,
                    max_contact_data_count=max_contacts,
                )
                if int(view.sensor_count) > 0 and int(view.filter_count) > 0:
                    self._c04_grasp_contact_views.append(view)
        except Exception as exc:
            self._c04_trigger_contact_view = None
            self._c04_grasp_contact_views = []
            self._add_task_state_warning(
                f"Could not initialize C04 hand contact reporting: {exc}"
            )

    def _init_c06_shutter_contact_view(self) -> None:
        self._c06_shutter_contact_view = None
        if self.task_name != "t23_photograph_logo" or not self._robots:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        hand_paths = self._c04_right_hand_body_paths()
        if physics_sim_view is None or not hand_paths:
            self._add_task_state_warning(
                "C06 right-hand shutter contact reporting is unavailable"
            )
            return
        try:
            view = physics_sim_view.create_rigid_contact_view(
                f"{_TASK_ROOT}/link_19/link_7",
                filter_patterns=hand_paths,
                max_contact_data_count=max(64, len(hand_paths) * 8),
            )
            if int(view.sensor_count) > 0 and int(view.filter_count) > 0:
                self._c06_shutter_contact_view = view
            else:
                self._add_task_state_warning(
                    "C06 right-hand shutter contact view has no sensors or filters"
                )
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not initialize C06 right-hand shutter contact reporting: {exc}"
            )

    def _init_c11_key_contact_views(self) -> None:
        self._c11_key_contact_views = {}
        if self.task_name != "c11_xylophone_sequence" or not self._robots:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        mallet_prim = self._find_rigid_body_prim("c11_mallet")
        if physics_sim_view is None or mallet_prim is None:
            self._add_task_state_warning(
                "C11 mallet/key contact reporting is unavailable"
            )
            return
        errors: list[str] = []
        for key_name in _C11_KEY_NAMES:
            key_prim = self._find_rigid_body_prim(key_name)
            if key_prim is None:
                errors.append(f"{key_name}: rigid body unavailable")
                continue
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    str(key_prim.GetPath()),
                    filter_patterns=[str(mallet_prim.GetPath())],
                    max_contact_data_count=64,
                )
                if (
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                ):
                    self._c11_key_contact_views[key_name] = view
                else:
                    errors.append(f"{key_name}: empty contact view")
            except Exception as exc:
                errors.append(f"{key_name}: {exc}")
        if errors:
            self._add_task_state_warning(
                "C11 mallet/key contact reporting is incomplete: "
                + "; ".join(errors)
            )

    def _init_c12_block_contact_views(self) -> None:
        self._c12_block_contact_views = {}
        self._c12_tool_block_contact_views = {}
        if self.task_name != "t11_line_up_blocks" or not self._robots:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        hand_paths = self._c04_right_hand_body_paths()
        if physics_sim_view is None or not hand_paths:
            self._add_task_state_warning(
                "C12 right-hand/block contact reporting is unavailable"
            )
            return
        errors: list[str] = []
        for name in _C12_BLOCK_NAMES:
            block_prim = self._find_rigid_body_prim(name)
            if block_prim is None:
                errors.append(f"{name}: rigid body unavailable")
                continue
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    str(block_prim.GetPath()),
                    filter_patterns=hand_paths,
                    max_contact_data_count=max(64, len(hand_paths) * 8),
                )
                if (
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                ):
                    self._c12_block_contact_views[name] = view
                else:
                    errors.append(f"{name}: empty contact view")
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if errors:
            self._add_task_state_warning(
                "C12 block contact reporting is incomplete: " + "; ".join(errors)
            )
        tool_prim = self._find_rigid_body_prim("c12_set_square")
        if tool_prim is None:
            self._add_task_state_warning(
                "C12 set-square/block contact reporting is unavailable"
            )
            return
        tool_errors: list[str] = []
        for name in _C12_BLOCK_NAMES:
            block_prim = self._find_rigid_body_prim(name)
            if block_prim is None:
                continue
            try:
                view = physics_sim_view.create_rigid_contact_view(
                    str(block_prim.GetPath()),
                    filter_patterns=[str(tool_prim.GetPath())],
                    max_contact_data_count=64,
                )
                if (
                    view is not None
                    and int(view.sensor_count) > 0
                    and int(view.filter_count) > 0
                ):
                    self._c12_tool_block_contact_views[name] = view
                else:
                    tool_errors.append(f"{name}: empty contact view")
            except Exception as exc:
                tool_errors.append(f"{name}: {exc}")
        if tool_errors:
            self._add_task_state_warning(
                "C12 set-square/block contact reporting is incomplete: "
                + "; ".join(tool_errors)
            )

    def _c04_contact_force(self, view: Any | None) -> float:
        if view is None:
            return 0.0
        try:
            matrix = view.get_contact_force_matrix(self._sim_dt)
            if hasattr(matrix, "detach"):
                matrix = matrix.detach().cpu().numpy()
            forces = np.asarray(matrix, dtype=np.float64)
            if forces.size == 0:
                return 0.0
            return float(np.linalg.norm(forces, axis=-1).sum())
        except Exception:
            return 0.0

    def _c04_trigger_contact_force(self) -> float:
        return self._c04_contact_force(self._c04_trigger_contact_view)

    def _t08_hammer_nail_contact_force(self) -> tuple[float, bool]:
        views = list(self._t08_hammer_nail_contact_views)
        if not views and self._t08_hammer_nail_contact_view is not None:
            views.append(self._t08_hammer_nail_contact_view)
        if not views:
            return 0.0, False
        # Both orientations describe the same contact pair, so use the larger
        # report instead of double-counting it.
        return max(self._c04_contact_force(view) for view in views), True

    def _c04_grasp_contact_force(self) -> float:
        return float(
            sum(self._c04_contact_force(view) for view in self._c04_grasp_contact_views)
        )

    def _b04_right_hand_contact_force(self) -> tuple[float, bool]:
        available = len(self._b04_block_contact_views) == len(
            _STACK_BLOCK_HALF_EXTENTS
        )
        if not available:
            return 0.0, False
        return (
            float(
                sum(
                    self._c04_contact_force(view)
                    for view in self._b04_block_contact_views.values()
                )
            ),
            True,
        )

    def _c06_right_hand_shutter_contact_force(self) -> float:
        return self._c04_contact_force(self._c06_shutter_contact_view)

    def _c12_right_hand_block_contact_force(self) -> float:
        return float(
            sum(
                self._c04_contact_force(view)
                for view in self._c12_block_contact_views.values()
            )
        )

    def _reset_c04_trigger_cycle(self) -> None:
        self._c04_trigger_phase = "idle"
        self._c04_trigger_cycle_latched = False
        self._c04_trigger_contact_seen = False
        self._c04_grasp_seen = False
        self._c04_trigger_contact_grace = 0
        self._c04_grasp_contact_grace = 0
        self._c04_grasp_ready_steps = 0
        self._c04_trigger_armed = False
        self._c04_trigger_pulled = False
        self._c04_water_emitting = False
        self._c04_failed_trigger_outside = False
        self._last_c04_success_metrics = {}

    def _c04_hand_gripping_spray(self) -> bool:
        force = self._c04_grasp_contact_force()
        gripping_now = force >= _C04_SPRAY_GRASP_FORCE
        touching = force > 1e-6
        return bool(
            gripping_now
            or (
                self._c04_grasp_seen
                and (touching or self._c04_grasp_contact_grace > 0)
            )
        )

    def _update_c04_trigger_state(self) -> None:
        if self.task_name != "t50_water_plant":
            return
        reading = self._physics_dof_position("joint_0")
        if reading is None:
            self._c04_trigger_pulled = False
            self._c04_water_emitting = False
            return

        trigger_position, _ = reading
        trigger_force = self._c04_trigger_contact_force()
        grasp_force = self._c04_grasp_contact_force()
        trigger_touching = trigger_force > 1e-6
        touching_grasp_surface = grasp_force > 1e-6
        gripping_now = grasp_force >= _C04_SPRAY_GRASP_FORCE

        self._c04_trigger_contact_seen = bool(
            self._c04_trigger_contact_seen or trigger_touching
        )
        self._c04_trigger_contact_grace = (
            _C04_TRIGGER_CONTACT_GRACE_STEPS
            if trigger_touching
            else max(0, self._c04_trigger_contact_grace - 1)
        )
        if gripping_now:
            self._c04_grasp_seen = True
            self._c04_grasp_contact_grace = _C04_GRASP_CONTACT_GRACE_STEPS
        elif self._c04_grasp_seen and touching_grasp_surface:
            self._c04_grasp_contact_grace = _C04_GRASP_CONTACT_GRACE_STEPS
        elif self._c04_grasp_seen:
            self._c04_grasp_contact_grace = max(
                0,
                self._c04_grasp_contact_grace - 1,
            )
            if self._c04_grasp_contact_grace == 0:
                self._c04_grasp_seen = False

        grasp_ok = bool(
            gripping_now
            or (
                self._c04_grasp_seen
                and (
                    touching_grasp_surface
                    or self._c04_grasp_contact_grace > 0
                )
            )
        )

        if trigger_position <= _C04_TRIGGER_RELEASE_RAD:
            self._c04_grasp_ready_steps = (
                min(_C04_TRIGGER_ARM_STEPS, self._c04_grasp_ready_steps + 1)
                if grasp_ok
                else 0
            )
            self._c04_trigger_armed = bool(
                self._c04_grasp_ready_steps >= _C04_TRIGGER_ARM_STEPS
            )
            self._c04_trigger_phase = (
                "armed" if self._c04_trigger_armed else "idle"
            )
            self._c04_trigger_cycle_latched = False
            self._c04_trigger_pulled = False
            self._c04_water_emitting = False
            return

        # Picking the bottle up while a finger is already depressing the
        # trigger must not start water.  The operator has to establish a grasp
        # with the trigger released, then perform a separate press.
        if not self._c04_trigger_armed:
            self._c04_trigger_phase = "unarmed"
            self._c04_trigger_cycle_latched = False
            self._c04_trigger_pulled = False
            self._c04_water_emitting = False
            return

        contacted_pull = bool(
            (trigger_touching or self._c04_trigger_contact_grace > 0)
            and grasp_ok
            and (
                touching_grasp_surface
                or self._c04_grasp_contact_grace > 0
            )
        )

        if self._c04_trigger_cycle_latched:
            self._c04_trigger_phase = "pressed"
            self._c04_trigger_pulled = True
        elif trigger_position >= _C04_TRIGGER_PULL_RAD and contacted_pull:
            self._c04_trigger_phase = "pressed"
            self._c04_trigger_cycle_latched = True
            self._c04_trigger_pulled = True
        else:
            self._c04_trigger_phase = "pressing"
            self._c04_trigger_pulled = False
        self._c04_water_emitting = bool(
            self._c04_trigger_cycle_latched and self._c04_trigger_pulled
        )

    @staticmethod
    def _c04_spray_should_be_visible(
        trigger_position: float,
        currently_visible: bool,
    ) -> bool:
        position = float(trigger_position)
        if currently_visible:
            return bool(position > _C04_TRIGGER_RELEASE_RAD)
        return bool(position >= _C04_TRIGGER_PULL_RAD)

    def _set_c04_spray_visible(self, visible: bool) -> bool:
        visible = bool(visible)
        if self.task_name != "t50_water_plant" or self._stage is None:
            self._c04_spray_visible = False
            return False
        if visible == self._c04_spray_visible:
            return False

        from pxr import UsdGeom

        cone_prim = self._stage.GetPrimAtPath(f"{_TASK_ROOT}/link_2/cone_vis")
        if not cone_prim.IsValid():
            self._add_task_state_warning("Could not find the C04 spray visual")
            return False
        imageable = UsdGeom.Imageable(cone_prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
        self._c04_spray_visible = visible
        return True

    def _update_c04_spray_visual(self) -> None:
        if self.task_name != "t50_water_plant":
            return
        self._set_c04_spray_visible(self._c04_water_emitting)

    def _set_task_visual_names_visible(
        self,
        names: set[str],
        visible: bool,
    ) -> int:
        if self._stage is None or not names:
            return 0

        from pxr import Usd, UsdGeom

        task_root = self._stage.GetPrimAtPath(_TASK_ROOT)
        if not task_root.IsValid():
            return 0
        changed = 0
        for prim in Usd.PrimRange(task_root):
            if prim.GetName() not in names:
                continue
            imageable = UsdGeom.Imageable(prim)
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
            changed += 1
        return changed

    def _set_task_visual_colors(
        self,
        colors: dict[str, tuple[float, float, float]],
        *,
        opacity: float = 1.0,
    ) -> int:
        if self._stage is None or not colors:
            return 0

        from pxr import Gf, Usd, UsdGeom, UsdShade

        task_root = self._stage.GetPrimAtPath(_TASK_ROOT)
        if not task_root.IsValid():
            return 0
        changed = 0
        for prim in Usd.PrimRange(task_root):
            color = colors.get(prim.GetName())
            if color is None:
                continue
            candidates = [prim, *Usd.PrimRange(prim)]
            seen_paths: set[str] = set()
            for candidate in candidates:
                path = str(candidate.GetPath())
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                gprim = UsdGeom.Gprim(candidate)
                if not gprim:
                    continue
                UsdShade.MaterialBindingAPI(candidate).UnbindAllBindings()
                gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
                gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
                changed += 1
        return changed

    def _reset_c_tool_visual_feedback(self) -> None:
        hidden_by_task = {
            "c07_pitcher_pour_cup": {
                f"c07_cup_goal_rim_{index}" for index in range(8)
            },
            "c08_tongs_ice_cubes": {
                "c08_cup_goal_rim_xp",
                "c08_cup_goal_rim_xn",
                "c08_cup_goal_rim_yp",
                "c08_cup_goal_rim_yn",
            },
            "c09_staple_paper_sheets": {"c09_staple_done"},
            "t02_connect_charger_and_cable": {
                "c10_strip_indicator",
                "c10_charge_indicator",
            },
            "c11_xylophone_sequence": {
                f"{key_name}_glow" for key_name in _C11_KEY_NAMES
            },
        }
        self._set_task_visual_names_visible(
            hidden_by_task.get(self.task_name, set()),
            False,
        )
        if self.task_name == "t11_line_up_blocks":
            self._set_task_visual_colors(
                {
                    f"c12_block_{index}_visual": (0.58, 0.38, 0.18)
                    for index in range(3)
                }
            )

    def _update_c11_prompt_visuals(self) -> None:
        if self.task_name != "c11_xylophone_sequence":
            return
        key_colors = {
            "c11_key_red": (0.86, 0.10, 0.08),
            "c11_key_blue": (0.16, 0.35, 0.90),
            "c11_key_yellow": (0.98, 0.78, 0.18),
            "c11_key_green": (0.30, 0.92, 0.20),
        }
        prompt_colors = {
            f"c11_prompt_slot_{slot}": (0.48, 0.50, 0.52)
            for slot in range(_C11_MAX_SEQUENCE_LENGTH)
        }
        prompt_colors.update(
            {
                f"c11_prompt_slot_{slot}": key_colors[key_name]
                for slot, key_name in enumerate(self._c11_target_sequence)
            }
        )
        self._set_task_visual_colors(prompt_colors)

    def _control_robot(self, side: str):
        try:
            return self._robots[side]
        except KeyError as exc:
            raise RuntimeError(f"IsaacLab {side} robot is not initialized") from exc

    @staticmethod
    def _tensor_view(value: Any) -> Any:
        return value.torch if hasattr(value, "torch") else value

    def _cache_joint_targets(self) -> None:
        super()._cache_joint_targets()

    def _reset_robot_control_state(self) -> None:
        super()._reset_robot_control_state()

    def _apply_joint_targets(self) -> None:
        super()._apply_joint_targets()

    def _apply_raised_hand_home(self, robot_cfg: Any, side: str) -> bool:
        if self.task_name in _C_TOOL_CLEAR_START_TASKS:
            arm_home = _C_TOOL_SPREAD_ARM_HOME.get(side)
        elif self.task_name in _RAISED_INACTIVE_LEFT_TASKS and side == "left":
            arm_home = _RAISED_ARM_HOME.get(side)
        else:
            return False
        if arm_home is None:
            return False
        init_state = getattr(robot_cfg, "init_state", None)
        joint_pos = getattr(init_state, "joint_pos", None)
        if not isinstance(joint_pos, dict):
            return False
        for index, value in enumerate(arm_home, start=1):
            joint_pos[f"joint{index}_{side}(_1)?"] = float(value)
        return True

    def _configure_randomizable_table_physics(self) -> tuple[int, bool]:
        """Turn the imported fixed table into a movable kinematic fixture."""
        if self._stage is None:
            return 0, False

        from pxr import PhysxSchema, Usd, UsdPhysics

        table_prim = self._find_rigid_body_prim("table")
        if table_prim is None:
            self._add_task_state_warning(
                "Could not configure table-height randomization: table rigid body is unavailable"
            )
            return 0, False

        table_path = table_prim.GetPath()
        task_root = self._stage.GetPrimAtPath(_TASK_ROOT)
        disabled_joints = 0
        for prim in Usd.PrimRange(task_root):
            if not prim.IsActive() or not prim.IsA(UsdPhysics.FixedJoint):
                continue
            joint = UsdPhysics.Joint(prim)
            targets = set(joint.GetBody0Rel().GetTargets()) | set(
                joint.GetBody1Rel().GetTargets()
            )
            if table_path not in targets:
                continue
            prim.SetActive(False)
            disabled_joints += 1

        # The MJCF importer treats each fixed world body as a one-body
        # articulation.  A kinematic fixture must be a plain rigid body.
        if table_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            table_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if table_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            table_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

        rigid_api = UsdPhysics.RigidBodyAPI(table_prim)
        if not rigid_api:
            rigid_api = UsdPhysics.RigidBodyAPI.Apply(table_prim)
        kinematic_attr = rigid_api.GetKinematicEnabledAttr()
        if not kinematic_attr.IsValid():
            kinematic_attr = rigid_api.CreateKinematicEnabledAttr()
        kinematic_attr.Set(True)
        return disabled_joints, True

    def _configure_c05_mousepad_physics(self) -> bool:
        """Keep the visual C05 mousepad above a randomized tabletop."""
        if self._stage is None or self.task_name != "t14_move_and_click_mouse":
            return False

        from pxr import PhysxSchema, Usd, UsdPhysics

        mousepad_prim = self._find_rigid_body_prim("mousepad")
        if mousepad_prim is None:
            self._add_task_state_warning(
                "Could not configure the C05 mousepad visual"
            )
            return False

        mousepad_path = mousepad_prim.GetPath()
        task_root = self._stage.GetPrimAtPath(_TASK_ROOT)
        for prim in Usd.PrimRange(task_root):
            if not prim.IsActive() or not prim.IsA(UsdPhysics.FixedJoint):
                continue
            joint = UsdPhysics.Joint(prim)
            targets = set(joint.GetBody0Rel().GetTargets()) | set(
                joint.GetBody1Rel().GetTargets()
            )
            if mousepad_path in targets:
                prim.SetActive(False)

        if mousepad_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            mousepad_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if mousepad_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            mousepad_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

        rigid_api = UsdPhysics.RigidBodyAPI(mousepad_prim)
        if not rigid_api:
            rigid_api = UsdPhysics.RigidBodyAPI.Apply(mousepad_prim)
        kinematic_attr = rigid_api.GetKinematicEnabledAttr()
        if not kinematic_attr.IsValid():
            kinematic_attr = rigid_api.CreateKinematicEnabledAttr()
        kinematic_attr.Set(True)
        return True

    @staticmethod
    def _c_tool_fixture_names(task_name: str) -> tuple[str, ...]:
        if task_name == "b02_nut_bolt_assembly":
            # The source bolt has no free joint; it is a fixed assembly post.
            return ("bolt",)
        if task_name == "t14_move_and_click_mouse":
            return ("display_link_1",)
        if task_name == "c07_pitcher_pour_cup":
            return ("c07_cup", "c07_return_zone")
        if task_name == "t02_connect_charger_and_cable":
            return ("c10_power_strip",)
        if task_name == "c11_xylophone_sequence":
            return (
                "c11_xylophone",
                "c11_xylophone_frame",
                *_C11_KEY_NAMES,
                "c11_return_zone",
            )
        if task_name == "t11_line_up_blocks":
            return ("c12_baseline_pad",)
        return ()

    def _configure_c_tool_fixture_physics(self) -> int:
        """Keep task fixtures fixed while preserving reset randomization."""
        fixture_names = self._c_tool_fixture_names(self.task_name)
        if not fixture_names:
            return 0
        if self._stage is None:
            return 0

        from pxr import PhysxSchema, Usd, UsdPhysics

        task_root = self._stage.GetPrimAtPath(_TASK_ROOT)
        configured = 0
        for name in fixture_names:
            fixture_prim = self._find_rigid_body_prim(name)
            if fixture_prim is None:
                self._add_task_state_warning(
                    f"Could not fix task fixture {name}: rigid body is unavailable"
                )
                continue
            fixture_path = fixture_prim.GetPath()
            for prim in Usd.PrimRange(task_root):
                if not prim.IsActive() or not prim.IsA(UsdPhysics.FixedJoint):
                    continue
                joint = UsdPhysics.Joint(prim)
                targets = set(joint.GetBody0Rel().GetTargets()) | set(
                    joint.GetBody1Rel().GetTargets()
                )
                if fixture_path in targets:
                    prim.SetActive(False)

            if fixture_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                fixture_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if fixture_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                fixture_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

            rigid_api = UsdPhysics.RigidBodyAPI(fixture_prim)
            if not rigid_api:
                rigid_api = UsdPhysics.RigidBodyAPI.Apply(fixture_prim)
            kinematic_attr = rigid_api.GetKinematicEnabledAttr()
            if not kinematic_attr.IsValid():
                kinematic_attr = rigid_api.CreateKinematicEnabledAttr()
            kinematic_attr.Set(True)
            configured += 1
        return configured

    def _reset_scene_state(self) -> None:
        self._set_b01_release_assist(False)
        self._clear_b02_nut_grasp_assist("reset")
        if not self._robots:
            # Interface-only smoke tests intentionally skip AppLauncher and
            # therefore have no PhysX robot or task views to restore.
            return
        # Put the robots at their reset homes before sampling task-object poses.
        # Otherwise the clearance sampler sees the hand position from the end of
        # the previous episode, then the reset teleports the hand into an object.
        self._reset_robot_control_state()
        for robot in self._robots.values():
            robot.write_data_to_sim()
        if self._sim is not None:
            self._sim.forward()
        for robot in self._robots.values():
            robot.update(0.0)
        if self.task_name == "t02_connect_charger_and_cable":
            for name in (
                "c10_charger_plug",
                "c10_charging_cable",
            ):
                if name not in self._task_articulation_views:
                    self._set_c10_connector_kinematic(name, False)
        elif self.task_name == "c07_pitcher_pour_cup":
            self._set_task_body_kinematic("c07_pitcher", False)
        elif self.task_name == "t11_line_up_blocks":
            for name in _C12_BLOCK_NAMES:
                if name not in self._task_articulation_views:
                    self._set_task_body_kinematic(
                        name,
                        False,
                        warning_label="C12 block",
                    )
        self._restore_task_scene_state()
        self._apply_task_pose_randomization()
        if self.task_name == "t14_move_and_click_mouse":
            self._c05_capture_start_pending = True

    def _reset_robot_body_points(self, side: str) -> tuple[np.ndarray, ...]:
        """Return live reset-home arm/hand body origins for clearance sampling."""
        points: list[np.ndarray] = []
        robot = self._robots.get(side)
        if robot is not None:
            try:
                body_names = list(robot.body_names)
                body_poses = robot.data.body_pose_w[0]
                for index, name in enumerate(body_names):
                    normalized = str(name).lower()
                    if normalized in {f"base_{side}", f"base_{side}/base_{side}"}:
                        continue
                    point = body_poses[int(index), :3]
                    if hasattr(point, "detach"):
                        point = point.detach().cpu().numpy()
                    point_np = np.asarray(point, dtype=np.float64).reshape(3)
                    if np.all(np.isfinite(point_np)):
                        points.append(point_np)
            except Exception:
                points = []
        if points:
            return tuple(points)

        fallback = self._left_tcp_pose if side == "left" else self._right_tcp_pose
        return (np.asarray(fallback[:3], dtype=np.float64).copy(),)

    def _c05_reset_xy_positions(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.randomize:
            return _C05_DEFAULT_MOUSE_XY.copy(), _C05_DEFAULT_MOUSEPAD_XY.copy()

        hand_points = (
            *self._reset_robot_body_points("right"),
            *self._reset_robot_body_points("left"),
        )
        best: tuple[np.ndarray, np.ndarray] | None = None
        best_margin = -math.inf

        for _layout_attempt in range(_C05_RESET_MAX_ATTEMPTS):
            mouse_xy: np.ndarray | None = None
            mouse_margin = -math.inf
            for _candidate_index in range(_C05_RESET_CANDIDATES_PER_OBJECT):
                candidate = self._random.uniform(
                    _C05_MOUSE_XY_BOUNDS[0],
                    _C05_MOUSE_XY_BOUNDS[1],
                ).astype(np.float64)
                clearance = min(
                    (
                        float(np.linalg.norm(candidate - point[:2]))
                        for point in hand_points
                    ),
                    default=math.inf,
                )
                margin = clearance - _C05_MOUSE_HAND_CLEARANCE
                if margin > mouse_margin:
                    mouse_xy = candidate.copy()
                    mouse_margin = margin
                if margin >= _C05_RESET_MIN_CLEARANCE_SLACK:
                    mouse_xy = candidate
                    mouse_margin = margin
                    break
            if mouse_xy is None or mouse_margin < _C05_RESET_MIN_CLEARANCE_SLACK:
                continue

            mousepad_xy: np.ndarray | None = None
            mousepad_margin = -math.inf
            for _candidate_index in range(_C05_RESET_CANDIDATES_PER_OBJECT):
                candidate = self._random.uniform(
                    _C05_MOUSEPAD_XY_BOUNDS[0],
                    _C05_MOUSEPAD_XY_BOUNDS[1],
                ).astype(np.float64)
                margin = self._c05_reset_layout_margin(
                    mouse_xy,
                    candidate,
                    hand_points,
                )
                if margin > mousepad_margin:
                    mousepad_xy = candidate.copy()
                    mousepad_margin = margin
                if margin >= _C05_RESET_MIN_CLEARANCE_SLACK:
                    mousepad_xy = candidate
                    mousepad_margin = margin
                    break
            if (
                mousepad_xy is None
                or mousepad_margin < _C05_RESET_MIN_CLEARANCE_SLACK
            ):
                continue

            margin = self._c05_reset_layout_margin(
                mouse_xy,
                mousepad_xy,
                hand_points,
            )
            if margin > best_margin:
                best = (mouse_xy.copy(), mousepad_xy.copy())
                best_margin = margin
            if margin >= _C05_RESET_MIN_CLEARANCE_SLACK:
                return mouse_xy, mousepad_xy

        if best is not None and best_margin >= _C05_RESET_MIN_CLEARANCE_SLACK:
            return best
        return _C05_DEFAULT_MOUSE_XY.copy(), _C05_DEFAULT_MOUSEPAD_XY.copy()

    @staticmethod
    def _c05_reset_layout_margin(
        mouse_xy: np.ndarray,
        mousepad_xy: np.ndarray,
        hand_points: tuple[np.ndarray, ...],
    ) -> float:
        mouse = np.asarray(mouse_xy, dtype=np.float64)
        mousepad = np.asarray(mousepad_xy, dtype=np.float64)
        mouse_clearance = min(
            (
                float(np.linalg.norm(mouse - point[:2]))
                for point in hand_points
            ),
            default=math.inf,
        )
        mousepad_clearance = min(
            (
                float(np.linalg.norm(mousepad - point[:2]))
                for point in hand_points
            ),
            default=math.inf,
        )
        mousepad_distance = float(np.linalg.norm(mousepad - mouse))
        return float(
            min(
                mouse_clearance - _C05_MOUSE_HAND_CLEARANCE,
                mousepad_clearance - _C05_MOUSEPAD_HAND_CLEARANCE,
                mousepad_distance - _C05_MOUSEPAD_MIN_OFFSET,
            )
        )

    @staticmethod
    def _c01_hammer_xy_bounds(yaw_degrees: float) -> np.ndarray:
        axis_x = abs(
            math.cos(
                math.radians(
                    float(yaw_degrees) + _C01_HAMMER_MAJOR_AXIS_OFFSET_DEG
                )
            )
        )
        lower_x = (
            _C01_HAMMER_MIN_FOOTPRINT_X
            + _C01_HAMMER_MAJOR_HALF_LENGTH * axis_x
        )
        return np.asarray(
            [
                [lower_x, _C01_HAMMER_XY_BOUNDS[0, 1]],
                [
                    _C01_HAMMER_XY_BOUNDS[1, 0],
                    _C01_HAMMER_XY_BOUNDS[1, 1],
                ],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _c01_hammer_major_segment(
        center_xy: np.ndarray,
        yaw_degrees: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        axis_angle = math.radians(
            float(yaw_degrees) + _C01_HAMMER_MAJOR_AXIS_OFFSET_DEG
        )
        offset = _C01_HAMMER_MAJOR_HALF_LENGTH * np.asarray(
            [math.cos(axis_angle), math.sin(axis_angle)],
            dtype=np.float64,
        )
        center = np.asarray(center_xy, dtype=np.float64)
        return center - offset, center + offset

    @staticmethod
    def _point_segment_distance_xy(
        point_xy: np.ndarray,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
    ) -> float:
        start = np.asarray(start_xy, dtype=np.float64)
        delta = np.asarray(end_xy, dtype=np.float64) - start
        denominator = float(np.dot(delta, delta))
        if denominator <= 1e-12:
            return float(np.linalg.norm(np.asarray(point_xy) - start))
        fraction = float(
            np.clip(
                np.dot(np.asarray(point_xy, dtype=np.float64) - start, delta)
                / denominator,
                0.0,
                1.0,
            )
        )
        return float(
            np.linalg.norm(
                np.asarray(point_xy, dtype=np.float64)
                - (start + fraction * delta)
            )
        )

    @staticmethod
    def _c01_head_ndc(world_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(world_xyz, dtype=np.float64).reshape(-1, 3)
        rotation = _quat_wxyz_to_matrix(
            np.asarray(_HEAD_RECORDING_CAMERA_QUAT_WXYZ, dtype=np.float64)
        )
        eye = np.asarray(HEAD_RECORDING_CAMERA_EYE, dtype=np.float64)
        camera_points = (rotation.T @ (points - eye).T).T
        depth = -camera_points[:, 2]
        tan_half_fovy = math.tan(
            math.radians(_HEAD_RECORDING_CAMERA_FOVY) / 2.0
        )
        return camera_points[:, :2] / (
            np.maximum(depth, 1e-9)[:, None] * tan_half_fovy
        )

    def _c01_reset_pose(
        self,
    ) -> tuple[float, dict[str, np.ndarray]]:
        if not self.randomize:
            return (
                0.0,
                {
                    "hammer": _C01_HAMMER_INIT_XY.copy(),
                    "wood": _C01_WOOD_INIT_XY.copy(),
                },
            )

        right_points = self._reset_robot_body_points("right")
        left_points = self._reset_robot_body_points("left")
        hand_points = (*right_points, *left_points)
        hand_ndc = self._c01_head_ndc(
            np.asarray(hand_points, dtype=np.float64)
        )
        best_wood_xy = _C01_WOOD_INIT_XY.copy()
        best_wood_clearance = -math.inf
        for _ in range(_C01_OBJECT_RESET_MAX_ATTEMPTS):
            candidate = self._random.uniform(*_C01_WOOD_XY_BOUNDS)
            physical_clearance = min(
                (
                    float(np.linalg.norm(candidate - point[:2]))
                    for point in hand_points
                ),
                default=math.inf,
            )
            wood_ndc = self._c01_head_ndc(
                [[candidate[0], candidate[1], _C01_WOOD_SCREEN_Z]]
            )[0]
            screen_clearance = min(
                (
                    float(np.linalg.norm(wood_ndc - point_ndc))
                    for point_ndc in hand_ndc
                ),
                default=math.inf,
            )
            clearance = min(
                physical_clearance / _C01_WOOD_HAND_MIN_XY_DISTANCE,
                screen_clearance / _C01_WOOD_HAND_SCREEN_CLEARANCE,
            )
            if clearance > best_wood_clearance:
                best_wood_xy = candidate.copy()
                best_wood_clearance = clearance
            if (
                physical_clearance >= _C01_WOOD_HAND_MIN_XY_DISTANCE
                and screen_clearance >= _C01_WOOD_HAND_SCREEN_CLEARANCE
            ):
                best_wood_xy = candidate
                break
        wood_xy = best_wood_xy
        wood_ndc = self._c01_head_ndc(
            [[wood_xy[0], wood_xy[1], _C01_WOOD_SCREEN_Z]]
        )[0]
        for _ in range(_C01_OBJECT_RESET_MAX_ATTEMPTS):
            hammer_yaw_degrees = self._c01_reset_hammer_yaw_degrees()
            hammer_bounds = self._c01_hammer_xy_bounds(hammer_yaw_degrees)
            candidate = self._random.uniform(*hammer_bounds)
            wood_clearance = float(np.linalg.norm(candidate - wood_xy))
            segment_start, segment_end = self._c01_hammer_major_segment(
                candidate,
                hammer_yaw_degrees,
            )
            segment_ndc = self._c01_head_ndc(
                [
                    [segment_start[0], segment_start[1], _C01_HAMMER_SCREEN_Z],
                    [segment_end[0], segment_end[1], _C01_HAMMER_SCREEN_Z],
                ]
            )
            hand_clearance = min(
                (
                    self._point_segment_distance_xy(
                        point_ndc,
                        segment_ndc[0],
                        segment_ndc[1],
                    )
                    for point_ndc in hand_ndc
                ),
                default=math.inf,
            )
            wood_screen_clearance = self._point_segment_distance_xy(
                wood_ndc,
                segment_ndc[0],
                segment_ndc[1],
            )
            fully_visible = bool(
                np.all(
                    np.abs(segment_ndc[:, 0])
                    <= _C01_HAMMER_SCREEN_X_LIMIT
                )
                and np.all(
                    segment_ndc[:, 1]
                    >= _C01_HAMMER_SCREEN_Y_BOUNDS[0]
                )
                and np.all(
                    segment_ndc[:, 1]
                    <= _C01_HAMMER_SCREEN_Y_BOUNDS[1]
                )
            )
            if (
                wood_clearance >= _C01_HAMMER_WOOD_MIN_DISTANCE
                and hand_clearance >= _C01_HAMMER_HAND_SCREEN_CLEARANCE
                and wood_screen_clearance
                >= _C01_HAMMER_WOOD_SCREEN_CLEARANCE
                and fully_visible
            ):
                return (
                    float(hammer_yaw_degrees),
                    {
                        "hammer": candidate.astype(np.float64),
                        "wood": wood_xy.astype(np.float64),
                    },
                )

        # The reset pose is fixed, so this conservative rear-center placement
        # is a deterministic last resort if random rejection sampling exhausts.
        return (
            _C01_HAMMER_FALLBACK_YAW_DEG,
            {
                "hammer": _C01_HAMMER_FALLBACK_XY.copy(),
                "wood": wood_xy.astype(np.float64),
            },
        )

    def _c01_reset_hammer_yaw_degrees(self) -> float:
        if not self.randomize:
            return 0.0
        return float(self._random.uniform(*_C01_HAMMER_YAW_BOUNDS_DEG))

    @staticmethod
    def _c04_reset_layout_margin(
        xy_by_name: dict[str, np.ndarray],
        right_points: tuple[np.ndarray, ...],
        all_points: tuple[np.ndarray, ...],
    ) -> float:
        spray_xy = np.asarray(xy_by_name["spray"], dtype=np.float64)
        plant_xy = np.asarray(xy_by_name["plant"], dtype=np.float64)
        spray_clearance = min(
            (
                float(np.linalg.norm(spray_xy - point[:2]))
                for point in right_points
            ),
            default=math.inf,
        )
        plant_clearance = min(
            (
                float(np.linalg.norm(plant_xy - point[:2]))
                for point in all_points
            ),
            default=math.inf,
        )
        spray_plant_distance = float(np.linalg.norm(spray_xy - plant_xy))
        return float(
            min(
                spray_clearance - _C04_SPRAY_RIGHT_ARM_MIN_XY_DISTANCE,
                plant_clearance - _C04_PLANT_RESET_HAND_MIN_XY_DISTANCE,
                spray_plant_distance - _C04_SPRAY_PLANT_MIN_DISTANCE,
                _C04_SPRAY_PLANT_MAX_DISTANCE - spray_plant_distance,
            )
        )

    def _c04_reset_layout(self) -> tuple[dict[str, np.ndarray], float]:
        if not self.randomize:
            return (
                {
                    "spray": _C04_SPRAY_INIT_XY.copy(),
                    "plant": _C04_PLANT_INIT_XY.copy(),
                },
                float(np.mean(_C04_SPRAY_YAW_BOUNDS_DEG)),
            )

        right_points = self._reset_robot_body_points("right")
        all_points = (*right_points, *self._reset_robot_body_points("left"))
        fallback_xy = {
            "spray": _C04_SPRAY_INIT_XY.copy(),
            "plant": _C04_PLANT_INIT_XY.copy(),
        }
        best_xy: dict[str, np.ndarray] | None = None
        best_margin = -math.inf
        for _ in range(_C04_SPRAY_RESET_MAX_ATTEMPTS):
            spray_xy = None
            spray_margin = -math.inf
            for _candidate_index in range(_C04_RESET_CANDIDATES_PER_OBJECT):
                candidate = self._random.uniform(*_C04_SPRAY_XY_BOUNDS)
                clearance = min(
                    (
                        float(np.linalg.norm(candidate - point[:2]))
                        for point in right_points
                    ),
                    default=math.inf,
                )
                margin = clearance - _C04_SPRAY_RIGHT_ARM_MIN_XY_DISTANCE
                if margin > spray_margin:
                    spray_xy = candidate.copy()
                    spray_margin = margin
                if margin >= _C04_RESET_MIN_CLEARANCE_SLACK:
                    spray_xy = candidate
                    spray_margin = margin
                    break
            if spray_xy is None or spray_margin < _C04_RESET_MIN_CLEARANCE_SLACK:
                continue

            plant_xy = None
            plant_margin = -math.inf
            for _candidate_index in range(_C04_RESET_CANDIDATES_PER_OBJECT):
                candidate = self._random.uniform(*_C04_PLANT_XY_BOUNDS)
                xy_by_name = {"spray": spray_xy, "plant": candidate}
                margin = self._c04_reset_layout_margin(
                    xy_by_name,
                    right_points,
                    all_points,
                )
                if margin > plant_margin:
                    plant_xy = candidate.copy()
                    plant_margin = margin
                if margin >= _C04_RESET_MIN_CLEARANCE_SLACK:
                    plant_xy = candidate
                    plant_margin = margin
                    break
            if plant_xy is None or plant_margin < _C04_RESET_MIN_CLEARANCE_SLACK:
                continue

            candidate_xy = {"spray": spray_xy, "plant": plant_xy}
            margin = self._c04_reset_layout_margin(
                candidate_xy,
                right_points,
                all_points,
            )
            if margin > best_margin:
                best_xy = {
                    "spray": spray_xy.copy(),
                    "plant": plant_xy.copy(),
                }
                best_margin = margin
            if margin >= _C04_RESET_MIN_CLEARANCE_SLACK:
                break
        if best_xy is None or best_margin < _C04_RESET_MIN_CLEARANCE_SLACK:
            best_xy = fallback_xy
        yaw_degrees = float(self._random.uniform(*_C04_SPRAY_YAW_BOUNDS_DEG))
        return (
            {name: xy.astype(np.float64) for name, xy in best_xy.items()},
            yaw_degrees,
        )

    def _c04_reset_spray_pose(self) -> tuple[np.ndarray, float]:
        xy_by_name, yaw_degrees = self._c04_reset_layout()
        return xy_by_name["spray"].copy(), yaw_degrees

    @staticmethod
    def _reset_transform_state_for_path(
        path: str,
        rigid_states: Mapping[str, tuple[Any, Any, Any, Any]],
        articulation_states: Mapping[str, tuple[Any, Any, Any, Any]],
    ) -> tuple[str, Any, Any, Any, Any] | None:
        """Prefer an articulation root writer over a child rigid-body writer."""
        state = articulation_states.get(path)
        if state is not None:
            return ("articulation", *state)
        state = rigid_states.get(path)
        if state is not None:
            return ("rigid", *state)
        return None

    def _write_rigid_reset_state(
        self,
        path: str,
        view: Any,
        indices: Any,
        transforms: Any,
        velocities: Any,
    ) -> None:
        """Write a rigid reset pose without assigning velocity to kinematics."""
        view.set_transforms(transforms, indices)
        if path not in self._task_kinematic_body_paths:
            view.set_velocities(velocities, indices)

    def _c06_reset_xy_position(self) -> np.ndarray:
        if not self.randomize:
            return _C06_DEFAULT_CAMERA_XY.copy()
        hand_points = (
            *self._reset_robot_body_points("right"),
            *self._reset_robot_body_points("left"),
        )
        best = _C06_DEFAULT_CAMERA_XY.copy()
        best_margin = -math.inf
        for _ in range(_C06_RESET_MAX_ATTEMPTS):
            candidate = self._random.uniform(
                _C06_CAMERA_XY_BOUNDS[0],
                _C06_CAMERA_XY_BOUNDS[1],
            ).astype(np.float64)
            margin = self._c06_reset_layout_margin(candidate, hand_points)
            if margin > best_margin:
                best = candidate.copy()
                best_margin = margin
            if margin >= _C06_RESET_MIN_CLEARANCE_SLACK:
                return candidate
        if best_margin >= _C06_RESET_MIN_CLEARANCE_SLACK:
            return best
        return _C06_DEFAULT_CAMERA_XY.copy()

    @staticmethod
    def _c06_reset_layout_margin(
        camera_xy: np.ndarray,
        hand_points: tuple[np.ndarray, ...],
    ) -> float:
        camera = np.asarray(camera_xy, dtype=np.float64)
        hand_clearance = min(
            (
                float(np.linalg.norm(camera - point[:2]))
                for point in hand_points
            ),
            default=math.inf,
        )
        return float(hand_clearance - _C06_CAMERA_HAND_CLEARANCE)

    @staticmethod
    def _c07_visible_and_clear_xy(
        xy: np.ndarray,
        footprint_radius: float = 0.0,
        camera_bounds: np.ndarray = RECORDING_OBJECT_XY_BOUNDS,
        hand_points: tuple[np.ndarray, ...] | None = None,
    ) -> bool:
        center = np.asarray(xy, dtype=np.float64).reshape(2)
        if not (
            np.all(center >= camera_bounds[0])
            and np.all(center <= camera_bounds[1])
        ):
            return False
        clearance_points = hand_points or (
            np.asarray(_RIGHT_HOME[:3], dtype=np.float64),
            np.asarray(_LEFT_HOME[:3], dtype=np.float64),
        )
        required_clearance = _C07_HAND_CLEARANCE + float(footprint_radius)
        return bool(
            all(
                float(np.linalg.norm(center - np.asarray(point, dtype=np.float64)[:2]))
                >= required_clearance
                for point in clearance_points
            )
        )

    @staticmethod
    def _c07_particle_xy_for_pitcher(
        pitcher_xy: np.ndarray,
        pitcher_yaw_degrees: float = 0.0,
    ) -> np.ndarray:
        center = np.asarray(pitcher_xy, dtype=np.float64).reshape(1, 2)
        offsets = _C07_PARTICLE_XY_OFFSETS + _C07_PARTICLE_PILE_OFFSET.reshape(1, 2)
        yaw = math.radians(float(pitcher_yaw_degrees))
        rotation = np.asarray(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
            dtype=np.float64,
        )
        return offsets @ rotation.T + center

    @staticmethod
    def _c07_handle_xy_for_pitcher(
        pitcher_xy: np.ndarray,
        pitcher_yaw_degrees: float,
    ) -> np.ndarray:
        yaw = math.radians(float(pitcher_yaw_degrees))
        rotation = np.asarray(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
            dtype=np.float64,
        )
        return (
            np.asarray(pitcher_xy, dtype=np.float64).reshape(2)
            + rotation @ _C07_HANDLE_LOCAL_XY
        )

    def _c07_reset_layout(self) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        hand_points = (
            *self._a06_hand_points("right"),
            *self._a06_hand_points("left"),
        )
        if self.randomize:
            pitcher_yaw = 0.0
            cup_yaw = 0.0
            pitcher_xy = _C07_NOMINAL_PITCHER_XY.copy()
            cup_xy = _C07_NOMINAL_CUP_XY.copy()
            for _ in range(_C07_RESET_MAX_ATTEMPTS):
                candidate_pitcher_yaw = float(
                    self._random.uniform(*_C07_PITCHER_YAW_BOUNDS_DEG)
                )
                candidate_cup_yaw = float(
                    self._random.uniform(*_C07_CUP_YAW_BOUNDS_DEG)
                )
                candidate_pitcher = self._random.uniform(
                    _C07_PITCHER_XY_BOUNDS[0],
                    _C07_PITCHER_XY_BOUNDS[1],
                ).astype(np.float64)
                candidate_cup = self._random.uniform(
                    _C07_CUP_XY_BOUNDS[0],
                    _C07_CUP_XY_BOUNDS[1],
                ).astype(np.float64)
                distance = float(np.linalg.norm(candidate_pitcher - candidate_cup))
                if not (
                    _C07_MIN_PITCHER_CUP_DISTANCE
                    <= distance
                    <= _C07_MAX_PITCHER_CUP_DISTANCE
                ):
                    continue
                if not (
                    self._c07_visible_and_clear_xy(
                        candidate_pitcher,
                        _C07_PITCHER_FOOTPRINT_RADIUS,
                        _C07_CAMERA_XY_BOUNDS,
                        hand_points,
                    )
                    and self._c07_visible_and_clear_xy(
                        candidate_cup,
                        _C07_CUP_FOOTPRINT_RADIUS,
                        hand_points=hand_points,
                    )
                ):
                    continue
                candidate_particles = self._c07_particle_xy_for_pitcher(
                    candidate_pitcher,
                    candidate_pitcher_yaw,
                )
                candidate_handle = self._c07_handle_xy_for_pitcher(
                    candidate_pitcher,
                    candidate_pitcher_yaw,
                )
                if not self._c07_visible_and_clear_xy(
                    candidate_handle,
                    _C07_HANDLE_FOOTPRINT_RADIUS,
                    _C07_CAMERA_XY_BOUNDS,
                    hand_points,
                ):
                    continue
                if not all(
                    self._c07_visible_and_clear_xy(
                        candidate_particle,
                        camera_bounds=_C07_CAMERA_XY_BOUNDS,
                        hand_points=hand_points,
                    )
                    for candidate_particle in candidate_particles
                ):
                    continue
                if any(
                    float(np.linalg.norm(candidate_particle - candidate_cup))
                    < _C07_PARTICLE_CUP_CLEARANCE
                    for candidate_particle in candidate_particles
                ):
                    continue
                pitcher_xy = candidate_pitcher
                cup_xy = candidate_cup
                pitcher_yaw = candidate_pitcher_yaw
                cup_yaw = candidate_cup_yaw
                break
        else:
            pitcher_yaw = 0.0
            cup_yaw = 0.0
            pitcher_xy = _C07_NOMINAL_PITCHER_XY.copy()
            cup_xy = _C07_NOMINAL_CUP_XY.copy()

        particle_xy = self._c07_particle_xy_for_pitcher(pitcher_xy, pitcher_yaw)
        constrain_c07_xy = lambda value: np.clip(
            np.asarray(value, dtype=np.float64),
            _C07_CAMERA_XY_BOUNDS[0],
            _C07_CAMERA_XY_BOUNDS[1],
        )
        xy_by_name: dict[str, np.ndarray] = {
            "c07_pitcher": constrain_c07_xy(pitcher_xy),
            "c07_cup": constrain_recording_object_xy(cup_xy),
            "c07_return_zone": constrain_c07_xy(pitcher_xy),
        }
        xy_by_name.update(
            {
                name: constrain_c07_xy(particle_xy[index])
                for index, name in enumerate(_C07_PARTICLE_NAMES)
            }
        )
        yaw_by_name: dict[str, float] = {
            "c07_pitcher": pitcher_yaw,
            "c07_cup": cup_yaw,
            "c07_return_zone": pitcher_yaw,
        }
        yaw_by_name.update({name: 0.0 for name in _C07_PARTICLE_NAMES})
        return xy_by_name, yaw_by_name

    def _c12_randomized_block_xy(
        self,
        group_rotation: np.ndarray,
        group_pivot: np.ndarray,
        common_xy: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Scatter all C12 blocks on the right-hand side of the baseline."""
        group_rotation_np = np.asarray(group_rotation, dtype=np.float64)
        group_pivot_np = np.asarray(group_pivot, dtype=np.float64)
        common_xy_np = np.asarray(common_xy, dtype=np.float64)
        hand_points = (
            *self._a06_hand_points("right"),
            *self._a06_hand_points("left"),
        )
        best: dict[str, np.ndarray] | None = None
        best_margin = -math.inf
        set_square_xy = (
            group_pivot_np
            + group_rotation_np @ _C12_SET_SQUARE_LOCAL_XY
            + common_xy_np
        )
        tool_segments = tuple(
            set_square_xy.reshape(1, 2)
            + segment[:, :2] @ group_rotation_np.T
            for segment in _C12_TOOL_SEGMENTS_LOCAL
        )

        for _layout_attempt in range(_C12_RESET_MAX_ATTEMPTS):
            local_x = _C12_BLOCK_LOCAL_X_CENTERS + self._random.uniform(
                -_C12_BLOCK_LOCAL_X_JITTER,
                _C12_BLOCK_LOCAL_X_JITTER,
                size=len(_C12_BLOCK_NAMES),
            )
            self._random.shuffle(local_x)
            layout: dict[str, np.ndarray] = {}
            failed = False
            for index, name in enumerate(_C12_BLOCK_NAMES):
                accepted_xy: np.ndarray | None = None
                accepted_margin = -math.inf
                for _candidate_index in range(_C12_RESET_CANDIDATES_PER_BLOCK):
                    local_y = float(
                        self._random.uniform(*_C12_BLOCK_LOCAL_RIGHT_Y_BOUNDS)
                    )
                    candidate = (
                        group_pivot_np
                        + group_rotation_np
                        @ np.asarray([local_x[index], local_y], dtype=np.float64)
                        + common_xy_np
                    )
                    partial_layout = dict(layout)
                    partial_layout[name] = candidate
                    margin = self._c12_reset_layout_margin(
                        partial_layout,
                        hand_points,
                        tool_segments,
                    )
                    if margin > accepted_margin:
                        accepted_xy = candidate.copy()
                        accepted_margin = margin
                    if margin >= _C12_RESET_MIN_CLEARANCE_SLACK:
                        accepted_xy = candidate
                        accepted_margin = margin
                        break
                if (
                    accepted_xy is None
                    or accepted_margin < _C12_RESET_MIN_CLEARANCE_SLACK
                ):
                    failed = True
                    break
                layout[name] = accepted_xy

            if failed or set(layout) != set(_C12_BLOCK_NAMES):
                continue

            margin = self._c12_reset_layout_margin(
                layout,
                hand_points,
                tool_segments,
            )
            if margin > best_margin:
                best = {name: xy.copy() for name, xy in layout.items()}
                best_margin = margin
            if margin >= _C12_RESET_MIN_CLEARANCE_SLACK:
                return layout

        if best is not None and best_margin >= _C12_RESET_MIN_CLEARANCE_SLACK:
            return best
        return {
            name: (
                group_pivot_np
                + group_rotation_np
                @ np.asarray(
                    [
                        _C12_BLOCK_LOCAL_X_CENTERS[index],
                        0.5
                        * (
                            _C12_BLOCK_LOCAL_RIGHT_Y_BOUNDS[0]
                            + _C12_BLOCK_LOCAL_RIGHT_Y_BOUNDS[1]
                        ),
                    ],
                    dtype=np.float64,
                )
                + common_xy_np
            )
            for index, name in enumerate(_C12_BLOCK_NAMES)
        }

    @staticmethod
    def _c12_reset_layout_margin(
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...],
        tool_segments: tuple[np.ndarray, ...] = (),
    ) -> float:
        margin = math.inf
        for xy in xy_by_name.values():
            hand_clearance = min(
                (
                    float(np.linalg.norm(np.asarray(xy, dtype=np.float64) - point[:2]))
                    for point in hand_points
                ),
                default=math.inf,
            )
            margin = min(margin, hand_clearance - _C12_RESET_HAND_CLEARANCE)
            for segment, radius in zip(
                tool_segments,
                _C12_TOOL_SEGMENT_RADII,
                strict=False,
            ):
                start = np.asarray(segment[0], dtype=np.float64)
                end = np.asarray(segment[1], dtype=np.float64)
                delta = end - start
                length_sq = float(np.dot(delta, delta))
                if length_sq <= 1e-12:
                    distance = float(np.linalg.norm(xy - start))
                else:
                    fraction = float(
                        np.clip(
                            np.dot(xy - start, delta) / length_sq,
                            0.0,
                            1.0,
                        )
                    )
                    distance = float(
                        np.linalg.norm(xy - (start + fraction * delta))
                    )
                margin = min(
                    margin,
                    distance - _C12_BLOCK_FOOTPRINT_RADIUS - float(radius),
                )
        names = sorted(xy_by_name)
        for index, name_a in enumerate(names):
            for name_b in names[index + 1 :]:
                distance = float(
                    np.linalg.norm(xy_by_name[name_a] - xy_by_name[name_b])
                )
                margin = min(margin, distance - _C12_RESET_BLOCK_CLEARANCE)
        return float(margin)

    @staticmethod
    def _c08_point_segment_distance(
        point_xy: np.ndarray,
        segment_start: np.ndarray,
        segment_end: np.ndarray,
    ) -> float:
        point = np.asarray(point_xy, dtype=np.float64).reshape(2)
        start = np.asarray(segment_start, dtype=np.float64).reshape(2)
        end = np.asarray(segment_end, dtype=np.float64).reshape(2)
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            return float(np.linalg.norm(point - start))
        fraction = float(
            np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0)
        )
        return float(np.linalg.norm(point - (start + fraction * segment)))

    @staticmethod
    def _c08_reset_tongs_segment(
        tongs_xy: np.ndarray,
        tongs_yaw_degrees: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        yaw = math.radians(float(tongs_yaw_degrees))
        rotation = np.asarray(
            [
                [math.cos(yaw), -math.sin(yaw)],
                [math.sin(yaw), math.cos(yaw)],
            ],
            dtype=np.float64,
        )
        points = (
            np.asarray(tongs_xy, dtype=np.float64).reshape(1, 2)
            + _C08_TONGS_RESET_LOCAL_SEGMENT @ rotation.T
        )
        return points[0], points[1]

    @staticmethod
    def _c08_reset_pair_clearance(name_a: str, name_b: str) -> float:
        radius_a = float(_C08_RESET_FOOTPRINT_RADII[name_a])
        radius_b = float(_C08_RESET_FOOTPRINT_RADII[name_b])
        margin = (
            _C08_CUP_OBJECT_CLEARANCE_MARGIN
            if "c08_glass_cup" in {name_a, name_b}
            else _C08_OBJECT_CLEARANCE_MARGIN
        )
        return radius_a + radius_b + margin

    @classmethod
    def _c08_reset_layout_margin(
        cls,
        xy_by_name: dict[str, np.ndarray],
        tongs_yaw_degrees: float,
        hand_points: tuple[np.ndarray, ...] | None = None,
    ) -> float:
        """Return the smallest C08 reset clearance in the right-hand workspace."""
        hands = tuple(
            np.asarray(point[:2], dtype=np.float64)
            for point in (
                hand_points
                if hand_points is not None
                else (
                    np.asarray(
                        _C_TOOL_SPREAD_RIGHT_START_TCP[:3],
                        dtype=np.float64,
                    ),
                )
            )
        )
        margins: list[float] = []
        for name, xy_value in xy_by_name.items():
            xy = np.asarray(xy_value, dtype=np.float64).reshape(2)
            margins.extend(
                [
                    float(xy[0] - _C08_RIGHT_HAND_XY_BOUNDS[0, 0]),
                    float(_C08_RIGHT_HAND_XY_BOUNDS[1, 0] - xy[0]),
                    float(xy[1] - _C08_RIGHT_HAND_XY_BOUNDS[0, 1]),
                    float(_C08_RIGHT_HAND_XY_BOUNDS[1, 1] - xy[1]),
                ]
            )
            if name == "c08_tongs":
                continue
            required = (
                float(_C08_RESET_FOOTPRINT_RADII[name])
                + _C08_OBJECT_HAND_CLEARANCE_MARGIN
            )
            margins.extend(
                float(np.linalg.norm(xy - hand_xy)) - required
                for hand_xy in hands
            )

        tongs_xy = xy_by_name.get("c08_tongs")
        if tongs_xy is not None:
            segment_start, segment_end = cls._c08_reset_tongs_segment(
                tongs_xy,
                tongs_yaw_degrees,
            )
            tongs_hand_clearance = (
                _C08_TONGS_RESET_FOOTPRINT_RADIUS
                + _C08_TONGS_HAND_CLEARANCE_MARGIN
            )
            margins.extend(
                cls._c08_point_segment_distance(
                    hand_xy,
                    segment_start,
                    segment_end,
                )
                - tongs_hand_clearance
                for hand_xy in hands
            )
            pickup_distance = float(
                np.linalg.norm(
                    np.asarray(tongs_xy, dtype=np.float64)
                    - _C_TOOL_SPREAD_RIGHT_START_TCP[:2]
                )
            )
            margins.append(_C08_TONGS_PICKUP_MAX_DISTANCE - pickup_distance)
            for name, xy_value in xy_by_name.items():
                if name == "c08_tongs":
                    continue
                required = (
                    _C08_TONGS_RESET_FOOTPRINT_RADIUS
                    + float(_C08_RESET_FOOTPRINT_RADII[name])
                    + _C08_OBJECT_CLEARANCE_MARGIN
                )
                margins.append(
                    cls._c08_point_segment_distance(
                        xy_value,
                        segment_start,
                        segment_end,
                    )
                    - required
                )

        object_names = sorted(
            name for name in xy_by_name if name != "c08_tongs"
        )
        for index, name_a in enumerate(object_names):
            for name_b in object_names[index + 1 :]:
                margins.append(
                    float(
                        np.linalg.norm(
                            np.asarray(xy_by_name[name_a], dtype=np.float64)
                            - np.asarray(xy_by_name[name_b], dtype=np.float64)
                        )
                    )
                    - cls._c08_reset_pair_clearance(name_a, name_b)
                )
        return min(margins) if margins else float("inf")

    def _c08_reset_layout(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]] | None:
        """Seeded A01-style sampler around the fixed, reachable pickup tongs."""
        if self.task_name != "c08_tongs_ice_cubes":
            return None
        if not self.randomize:
            return (
                {
                    name: xy.copy()
                    for name, xy in _C08_RESET_XY_FALLBACK.items()
                },
                {name: 0.0 for name in _C08_RESET_XY_FALLBACK},
            )

        hand_points = self._reset_robot_body_points("right")
        best_layout: dict[str, np.ndarray] | None = None
        best_yaws: dict[str, float] | None = None
        best_margin = -math.inf
        for _ in range(_C08_RESET_MAX_ATTEMPTS):
            # Keep the tongs on their matching cradle.  The cup and every ice
            # cube randomize around this known-reachable pickup fixture.
            tongs_xy = _C08_RESET_XY_FALLBACK["c08_tongs"].copy()
            tongs_yaw = 0.0
            if (
                self._c08_reset_layout_margin(
                    {"c08_tongs": tongs_xy},
                    tongs_yaw,
                    hand_points,
                )
                < _C08_RESET_MIN_CLEARANCE_SLACK
            ):
                continue

            xy_by_name = {"c08_tongs": tongs_xy}
            yaw_by_name = {"c08_tongs": tongs_yaw}
            sample_order = ["c08_glass_cup", *_C08_ICE_NAMES]
            self._random.shuffle(sample_order)
            for name in sample_order:
                bounds = _C08_RESET_XY_BOUNDS[name]
                accepted = False
                for _candidate_index in range(_C08_RESET_CANDIDATES_PER_OBJECT):
                    candidate_xy = constrain_recording_object_xy(
                        self._random.uniform(bounds[0], bounds[1])
                    )
                    candidate_layout = dict(xy_by_name)
                    candidate_layout[name] = candidate_xy
                    if (
                        self._c08_reset_layout_margin(
                            candidate_layout,
                            tongs_yaw,
                            hand_points,
                        )
                        < _C08_RESET_MIN_CLEARANCE_SLACK
                    ):
                        continue
                    xy_by_name[name] = candidate_xy
                    yaw_by_name[name] = float(
                        self._random.uniform(*_C08_OBJECT_YAW_BOUNDS_DEG)
                    )
                    accepted = True
                    break
                if not accepted:
                    break

            if set(xy_by_name) != set(_C08_RESET_XY_BOUNDS):
                continue
            margin = self._c08_reset_layout_margin(
                xy_by_name,
                tongs_yaw,
                hand_points,
            )
            if margin > best_margin:
                best_margin = margin
                best_layout = {
                    name: xy.copy() for name, xy in xy_by_name.items()
                }
                best_yaws = dict(yaw_by_name)
            if margin >= _C08_RESET_MIN_CLEARANCE_SLACK:
                return xy_by_name, yaw_by_name

        if (
            best_layout is not None
            and best_yaws is not None
            and best_margin >= _C08_RESET_MIN_CLEARANCE_SLACK
        ):
            return best_layout, best_yaws

        fallback = {
            name: constrain_recording_object_xy(xy)
            for name, xy in _C08_RESET_XY_FALLBACK.items()
        }
        if (
            self._c08_reset_layout_margin(
                fallback,
                0.0,
                hand_points,
            )
            >= 0.0
        ):
            return fallback, {name: 0.0 for name in fallback}
        raise RuntimeError("c08_tongs_ice_cubes 无法生成无穿模可达初始布局")

    @staticmethod
    def _a01_object_hand_clearance(name: str) -> float:
        baseline = (
            _A01_BUCKET_HAND_CLEARANCE
            if name == "bucket"
            else _A01_FRUIT_HAND_CLEARANCE
        )
        radius = float(_A01_OBJECT_FOOTPRINT_RADII.get(name, 0.045))
        return max(float(baseline), radius + _A01_OBJECT_HAND_CLEARANCE_MARGIN)

    @staticmethod
    def _a01_pair_clearance(name_a: str, name_b: str) -> float:
        baseline = (
            _A01_BUCKET_OBJECT_CLEARANCE
            if "bucket" in {name_a, name_b}
            else _A01_FRUIT_OBJECT_CLEARANCE
        )
        radius_a = float(_A01_OBJECT_FOOTPRINT_RADII.get(name_a, 0.045))
        radius_b = float(_A01_OBJECT_FOOTPRINT_RADII.get(name_b, 0.045))
        margin = (
            _A01_BUCKET_OBJECT_CLEARANCE_MARGIN
            if "bucket" in {name_a, name_b}
            else _A01_OBJECT_CLEARANCE_MARGIN
        )
        return max(float(baseline), radius_a + radius_b + margin)

    @staticmethod
    def _a01_reset_layout_margin(
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...] | None = None,
    ) -> float:
        # A01 is collected with the right hand only; the left arm starts raised
        # and should not collapse the reset sampling volume.
        hand_xy = tuple(
            np.asarray(point[:2], dtype=np.float64)
            for point in (
                hand_points
                if hand_points is not None
                else (np.asarray(_RIGHT_HOME[:3], dtype=np.float64),)
            )
        )
        margins: list[float] = []
        for name, xy in xy_by_name.items():
            required = HandsOnABCIsaacLabEnv._a01_object_hand_clearance(name)
            margins.extend(
                float(np.linalg.norm(xy - point)) - required for point in hand_xy
            )

        names = sorted(xy_by_name)
        for index, name_a in enumerate(names):
            for name_b in names[index + 1 :]:
                required = HandsOnABCIsaacLabEnv._a01_pair_clearance(name_a, name_b)
                margins.append(
                    float(np.linalg.norm(xy_by_name[name_a] - xy_by_name[name_b]))
                    - required
                )
        return min(margins) if margins else float("inf")

    @classmethod
    def _a01_reset_layout_is_clear(
        cls,
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...] | None = None,
    ) -> bool:
        return cls._a01_reset_layout_margin(xy_by_name, hand_points) >= 0.0

    def _a01_reset_xy_positions(self) -> dict[str, np.ndarray] | None:
        if self.task_name != "t26_place_fruit_in_bucket":
            return None
        if not self.randomize:
            return {
                name: constrain_recording_object_xy(xy)
                for name, xy in _A01_RESET_XY_FALLBACK.items()
            }

        hand_points = self._reset_robot_body_points("right")
        best_layout: dict[str, np.ndarray] | None = None
        best_margin = -float("inf")
        for _ in range(_A01_RESET_MAX_ATTEMPTS):
            xy_by_name: dict[str, np.ndarray] = {}
            sample_order = list(_A01_RESET_SAMPLE_ORDER)
            self._random.shuffle(sample_order)
            for name in sample_order:
                bounds = _A01_RESET_XY_BOUNDS[name]
                accepted = False
                for _candidate_index in range(_A01_RESET_CANDIDATES_PER_OBJECT):
                    candidate_xy = constrain_recording_object_xy(
                        self._random.uniform(bounds[0], bounds[1])
                    )
                    candidate_layout = dict(xy_by_name)
                    candidate_layout[name] = candidate_xy
                    margin = self._a01_reset_layout_margin(
                        candidate_layout,
                        hand_points,
                    )
                    if margin >= _A01_RESET_MIN_CLEARANCE_SLACK:
                        xy_by_name[name] = candidate_xy
                        accepted = True
                        break
                if not accepted:
                    break

            if set(xy_by_name) != set(_A01_RESET_XY_BOUNDS):
                continue
            margin = self._a01_reset_layout_margin(xy_by_name, hand_points)
            if margin > best_margin:
                best_margin = margin
                best_layout = {
                    name: xy.copy() for name, xy in xy_by_name.items()
                }
            if margin >= _A01_RESET_MIN_CLEARANCE_SLACK:
                return {
                    name: xy_by_name[name]
                    for name in _A01_RESET_XY_BOUNDS
                }

        if best_layout is not None and best_margin >= _A01_RESET_MIN_CLEARANCE_SLACK:
            return {
                name: best_layout[name]
                for name in _A01_RESET_XY_BOUNDS
            }
        return {
            name: constrain_recording_object_xy(
                _A01_RESET_XY_FALLBACK.get(
                    name,
                    _A01_RESET_XY_BOUNDS[name].mean(axis=0),
                )
            )
            for name in _A01_RESET_XY_BOUNDS
        }

    @staticmethod
    def _a02_object_hand_clearance(name: str) -> float:
        radius = float(_A02_OBJECT_FOOTPRINT_RADII.get(name, 0.060))
        return radius + _A02_OBJECT_HAND_CLEARANCE_MARGIN

    @staticmethod
    def _a02_pair_clearance(name_a: str, name_b: str) -> float:
        radius_a = float(_A02_OBJECT_FOOTPRINT_RADII.get(name_a, 0.060))
        radius_b = float(_A02_OBJECT_FOOTPRINT_RADII.get(name_b, 0.060))
        return radius_a + radius_b + _A02_OBJECT_CLEARANCE_MARGIN

    @staticmethod
    def _a02_reset_layout_margin(
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...] | None = None,
    ) -> float:
        # A02 is a right-hand collection task; the inactive left arm is raised.
        hand_xy = tuple(
            np.asarray(point[:2], dtype=np.float64)
            for point in (
                hand_points
                if hand_points is not None
                else (np.asarray(_RIGHT_HOME[:3], dtype=np.float64),)
            )
        )
        margins: list[float] = []
        for name, xy in xy_by_name.items():
            required = HandsOnABCIsaacLabEnv._a02_object_hand_clearance(name)
            margins.extend(
                float(np.linalg.norm(xy - point)) - required for point in hand_xy
            )

        names = sorted(xy_by_name)
        for index, name_a in enumerate(names):
            for name_b in names[index + 1 :]:
                required = HandsOnABCIsaacLabEnv._a02_pair_clearance(name_a, name_b)
                margins.append(
                    float(np.linalg.norm(xy_by_name[name_a] - xy_by_name[name_b]))
                    - required
                )
        return min(margins) if margins else float("inf")

    @classmethod
    def _a02_reset_layout_is_clear(
        cls,
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...] | None = None,
    ) -> bool:
        return cls._a02_reset_layout_margin(xy_by_name, hand_points) >= 0.0

    def _a02_reset_xy_positions(self) -> dict[str, np.ndarray] | None:
        if self.task_name != "t25_place_cup_on_plate":
            return None
        if not self.randomize:
            return {
                name: constrain_recording_object_xy(xy)
                for name, xy in _A02_RESET_XY_FALLBACK.items()
            }

        hand_points = self._reset_robot_body_points("right")
        best_layout: dict[str, np.ndarray] | None = None
        best_margin = -float("inf")
        for _ in range(_A02_RESET_MAX_ATTEMPTS):
            xy_by_name: dict[str, np.ndarray] = {}
            sample_order = list(_A02_RESET_SAMPLE_ORDER)
            self._random.shuffle(sample_order)
            for name in sample_order:
                bounds = _A02_RESET_XY_BOUNDS[name]
                accepted = False
                for _candidate_index in range(_A02_RESET_CANDIDATES_PER_OBJECT):
                    candidate_xy = constrain_recording_object_xy(
                        self._random.uniform(bounds[0], bounds[1])
                    )
                    candidate_layout = dict(xy_by_name)
                    candidate_layout[name] = candidate_xy
                    margin = self._a02_reset_layout_margin(
                        candidate_layout,
                        hand_points,
                    )
                    if margin >= _A02_RESET_MIN_CLEARANCE_SLACK:
                        xy_by_name[name] = candidate_xy
                        accepted = True
                        break
                if not accepted:
                    break

            if set(xy_by_name) != set(_A02_RESET_XY_BOUNDS):
                continue
            margin = self._a02_reset_layout_margin(xy_by_name, hand_points)
            if margin > best_margin:
                best_margin = margin
                best_layout = {
                    name: xy.copy() for name, xy in xy_by_name.items()
                }
            if margin >= _A02_RESET_MIN_CLEARANCE_SLACK:
                return {
                    name: xy_by_name[name]
                    for name in _A02_RESET_XY_BOUNDS
                }

        if best_layout is not None and best_margin >= _A02_RESET_MIN_CLEARANCE_SLACK:
            return {
                name: best_layout[name]
                for name in _A02_RESET_XY_BOUNDS
            }
        return {
            name: constrain_recording_object_xy(
                _A02_RESET_XY_FALLBACK.get(
                    name,
                    _A02_RESET_XY_BOUNDS[name].mean(axis=0),
                )
            )
            for name in _A02_RESET_XY_BOUNDS
        }

    @staticmethod
    def _a03_object_hand_clearance(name: str) -> float:
        radius = float(_A03_OBJECT_FOOTPRINT_RADII.get(name, 0.045))
        return radius + _A03_OBJECT_HAND_CLEARANCE_MARGIN

    @staticmethod
    def _a03_pair_clearance(name_a: str, name_b: str) -> float:
        radius_a = float(_A03_OBJECT_FOOTPRINT_RADII.get(name_a, 0.045))
        radius_b = float(_A03_OBJECT_FOOTPRINT_RADII.get(name_b, 0.045))
        clearance = radius_a + radius_b + _A03_OBJECT_CLEARANCE_MARGIN
        if "024_bowl" in {name_a, name_b}:
            clearance = max(clearance, _A03_BOWL_BALL_CLEARANCE)
        return float(clearance)

    @staticmethod
    def _a03_reset_layout_margin(
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...] | None = None,
    ) -> float:
        hand_xy = tuple(
            np.asarray(point[:2], dtype=np.float64)
            for point in (
                hand_points
                if hand_points is not None
                else (
                    np.asarray(_RIGHT_HOME[:3], dtype=np.float64),
                    np.asarray(_LEFT_HOME[:3], dtype=np.float64),
                )
            )
        )
        margins: list[float] = []
        for name, xy in xy_by_name.items():
            required = HandsOnABCIsaacLabEnv._a03_object_hand_clearance(name)
            margins.extend(
                float(np.linalg.norm(xy - point)) - required for point in hand_xy
            )

        names = sorted(xy_by_name)
        for index, name_a in enumerate(names):
            for name_b in names[index + 1 :]:
                required = HandsOnABCIsaacLabEnv._a03_pair_clearance(name_a, name_b)
                margins.append(
                    float(np.linalg.norm(xy_by_name[name_a] - xy_by_name[name_b]))
                    - required
                )
        return min(margins) if margins else float("inf")

    @classmethod
    def _a03_reset_layout_is_clear(
        cls,
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...] | None = None,
    ) -> bool:
        return cls._a03_reset_layout_margin(xy_by_name, hand_points) >= 0.0

    def _a03_reset_xy_positions(self) -> dict[str, np.ndarray] | None:
        if self.task_name != "t24_place_ball_in_bowl" or not self.randomize:
            return None

        hand_points = (
            *self._reset_robot_body_points("right"),
            *self._reset_robot_body_points("left"),
        )
        best_layout: dict[str, np.ndarray] | None = None
        best_margin = -float("inf")
        for _ in range(_A03_RESET_MAX_ATTEMPTS):
            xy_by_name: dict[str, np.ndarray] = {}
            sample_order = list(_A03_RESET_SAMPLE_ORDER)
            self._random.shuffle(sample_order)
            for name in sample_order:
                bounds = _A03_RESET_XY_BOUNDS[name]
                accepted = False
                for _candidate_index in range(_A03_RESET_CANDIDATES_PER_OBJECT):
                    candidate_xy = constrain_recording_object_xy(
                        self._random.uniform(bounds[0], bounds[1])
                    )
                    candidate_layout = dict(xy_by_name)
                    candidate_layout[name] = candidate_xy
                    margin = self._a03_reset_layout_margin(
                        candidate_layout,
                        hand_points,
                    )
                    if margin >= _A03_RESET_MIN_CLEARANCE_SLACK:
                        xy_by_name[name] = candidate_xy
                        accepted = True
                        break
                if not accepted:
                    break

            if set(xy_by_name) != set(_A03_RESET_XY_BOUNDS):
                continue
            margin = self._a03_reset_layout_margin(xy_by_name, hand_points)
            if margin > best_margin:
                best_margin = margin
                best_layout = {
                    name: xy.copy() for name, xy in xy_by_name.items()
                }
            if margin >= _A03_RESET_MIN_CLEARANCE_SLACK:
                return {
                    name: xy_by_name[name]
                    for name in _A03_RESET_XY_BOUNDS
                }

        if best_layout is not None and best_margin >= _A03_RESET_MIN_CLEARANCE_SLACK:
            return {
                name: best_layout[name]
                for name in _A03_RESET_XY_BOUNDS
            }

        fallback = {
            name: constrain_recording_object_xy(
                _A03_RESET_XY_FALLBACK.get(
                    name,
                    _A03_RESET_XY_BOUNDS[name].mean(axis=0),
                )
            )
            for name in _A03_RESET_XY_BOUNDS
        }
        if self._a03_reset_layout_margin(fallback, hand_points) >= 0.0:
            return fallback
        raise RuntimeError("t24_place_ball_in_bowl 无法生成无穿模初始布局")

    @staticmethod
    def _b01_object_hand_clearance(name: str) -> float:
        radius = float(_B01_OBJECT_FOOTPRINT_RADII.get(name, 0.090))
        return radius + _B01_OBJECT_HAND_CLEARANCE_MARGIN

    @staticmethod
    def _b01_pair_clearance(name_a: str, name_b: str) -> float:
        radius_a = float(_B01_OBJECT_FOOTPRINT_RADII.get(name_a, 0.090))
        radius_b = float(_B01_OBJECT_FOOTPRINT_RADII.get(name_b, 0.090))
        return radius_a + radius_b + _B01_OBJECT_CLEARANCE_MARGIN

    @staticmethod
    def _b01_reset_layout_margin(
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...] | None = None,
    ) -> float:
        # B01 is a right-hand collection task; the inactive left arm is raised.
        hand_xy = tuple(
            np.asarray(point[:2], dtype=np.float64)
            for point in (
                hand_points
                if hand_points is not None
                else (np.asarray(_RIGHT_HOME[:3], dtype=np.float64),)
            )
        )
        margins: list[float] = []
        for name, xy in xy_by_name.items():
            required = HandsOnABCIsaacLabEnv._b01_object_hand_clearance(name)
            margins.extend(
                float(np.linalg.norm(xy - point)) - required for point in hand_xy
            )

        names = sorted(xy_by_name)
        for index, name_a in enumerate(names):
            for name_b in names[index + 1 :]:
                required = HandsOnABCIsaacLabEnv._b01_pair_clearance(name_a, name_b)
                margins.append(
                    float(np.linalg.norm(xy_by_name[name_a] - xy_by_name[name_b]))
                    - required
                )
        return min(margins) if margins else float("inf")

    @classmethod
    def _b01_reset_layout_is_clear(
        cls,
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...] | None = None,
    ) -> bool:
        return cls._b01_reset_layout_margin(xy_by_name, hand_points) >= 0.0

    def _b01_reset_xy_positions(self) -> dict[str, np.ndarray] | None:
        if self.task_name != "t04_cover_pitcher":
            return None
        if not self.randomize:
            return {
                name: constrain_recording_object_xy(xy)
                for name, xy in _B01_RESET_XY_FALLBACK.items()
            }

        hand_points = self._reset_robot_body_points("right")
        best_layout: dict[str, np.ndarray] | None = None
        best_margin = -float("inf")
        for _ in range(_B01_RESET_MAX_ATTEMPTS):
            xy_by_name: dict[str, np.ndarray] = {}
            sample_order = list(_B01_RESET_SAMPLE_ORDER)
            self._random.shuffle(sample_order)
            for name in sample_order:
                bounds = _B01_RESET_XY_BOUNDS[name]
                accepted = False
                for _candidate_index in range(_B01_RESET_CANDIDATES_PER_OBJECT):
                    candidate_xy = constrain_recording_object_xy(
                        self._random.uniform(bounds[0], bounds[1])
                    )
                    candidate_layout = dict(xy_by_name)
                    candidate_layout[name] = candidate_xy
                    margin = self._b01_reset_layout_margin(
                        candidate_layout,
                        hand_points,
                    )
                    if margin >= _B01_RESET_MIN_CLEARANCE_SLACK:
                        xy_by_name[name] = candidate_xy
                        accepted = True
                        break
                if not accepted:
                    break

            if set(xy_by_name) != set(_B01_RESET_XY_BOUNDS):
                continue
            margin = self._b01_reset_layout_margin(xy_by_name, hand_points)
            if margin > best_margin:
                best_margin = margin
                best_layout = {
                    name: xy.copy() for name, xy in xy_by_name.items()
                }
            if margin >= _B01_RESET_MIN_CLEARANCE_SLACK:
                return {
                    name: xy_by_name[name]
                    for name in _B01_RESET_XY_BOUNDS
                }

        if best_layout is not None and best_margin >= _B01_RESET_MIN_CLEARANCE_SLACK:
            return {
                name: best_layout[name]
                for name in _B01_RESET_XY_BOUNDS
            }
        return {
            name: constrain_recording_object_xy(
                _B01_RESET_XY_FALLBACK.get(
                    name,
                    _B01_RESET_XY_BOUNDS[name].mean(axis=0),
                )
            )
            for name in _B01_RESET_XY_BOUNDS
        }

    @staticmethod
    def _task_specific_pair_distance(task_name: str, name_a: str, name_b: str) -> float:
        return float(
            _TASK_SPECIFIC_RESET_PAIR_DISTANCE.get(
                (task_name, frozenset((name_a, name_b))),
                _TASK_SPECIFIC_RESET_DEFAULT_PAIR_DISTANCE,
            )
        )

    def _task_specific_reset_layout_margin(
        self,
        xy_by_name: dict[str, np.ndarray],
        hand_points: tuple[np.ndarray, ...],
    ) -> float:
        margin = math.inf
        for xy in xy_by_name.values():
            hand_clearance = min(
                (
                    float(np.linalg.norm(xy - point[:2]))
                    for point in hand_points
                ),
                default=math.inf,
            )
            margin = min(margin, hand_clearance - _TASK_OBJECT_HAND_CLEARANCE)

        names = sorted(xy_by_name)
        for index, name_a in enumerate(names):
            for name_b in names[index + 1 :]:
                distance = float(np.linalg.norm(xy_by_name[name_a] - xy_by_name[name_b]))
                margin = min(
                    margin,
                    distance
                    - self._task_specific_pair_distance(
                        self.task_name,
                        name_a,
                        name_b,
                    ),
                )
        return float(margin)

    def _task_specific_reset_xy_positions(self) -> dict[str, np.ndarray] | None:
        if self.task_name == "t25_place_cup_on_plate":
            return self._a02_reset_xy_positions()
        if self.task_name == "t24_place_ball_in_bowl":
            return self._a03_reset_xy_positions()
        if self.task_name == "t04_cover_pitcher":
            return self._b01_reset_xy_positions()
        bounds_by_name = _TASK_SPECIFIC_RESET_XY_BOUNDS.get(self.task_name)
        if bounds_by_name is None or not self.randomize:
            return None

        hand_points = (
            *self._reset_robot_body_points("right"),
            *self._reset_robot_body_points("left"),
        )
        best: dict[str, np.ndarray] | None = None
        best_margin = -math.inf
        for _ in range(_TASK_OBJECT_RESET_MAX_ATTEMPTS):
            sample_order = list(bounds_by_name)
            self._random.shuffle(sample_order)
            candidate: dict[str, np.ndarray] = {}
            failed = False

            for name in sample_order:
                bounds = bounds_by_name[name]
                accepted_xy: np.ndarray | None = None
                accepted_margin = -math.inf

                for _candidate_index in range(_TASK_OBJECT_RESET_CANDIDATES_PER_OBJECT):
                    xy = constrain_recording_object_xy(
                        self._random.uniform(bounds[0], bounds[1])
                    )
                    partial_candidate = dict(candidate)
                    partial_candidate[name] = xy
                    margin = self._task_specific_reset_layout_margin(
                        partial_candidate,
                        hand_points,
                    )
                    if margin > accepted_margin:
                        accepted_xy = xy
                        accepted_margin = margin
                    if margin >= _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK:
                        accepted_xy = xy
                        accepted_margin = margin
                        break

                if (
                    accepted_xy is None
                    or accepted_margin < _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK
                ):
                    failed = True
                    break

                candidate[name] = accepted_xy

            if failed or set(candidate) != set(bounds_by_name):
                continue

            margin = self._task_specific_reset_layout_margin(candidate, hand_points)
            if margin > best_margin:
                best = {name: xy.copy() for name, xy in candidate.items()}
                best_margin = margin
            if margin >= _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK:
                return candidate

        if best is not None and best_margin >= _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK:
            return best
        return None

    def _apply_task_pose_randomization(self) -> None:
        if not self._task_reset_body_states and not self._task_reset_articulation_states:
            return
        if not self.randomize and self.task_name not in {
            "t09_insert_peg",
            "t08_hammer_nail",
            "t50_water_plant",
            "t14_move_and_click_mouse",
            "t23_photograph_logo",
            "c07_pitcher_pour_cup",
        }:
            return

        c01_reset_pose = (
            self._c01_reset_pose()
            if self.task_name == "t08_hammer_nail"
            else None
        )
        c01_hammer_yaw = (
            c01_reset_pose[0] if c01_reset_pose is not None else None
        )
        c01_reset_xy = (
            c01_reset_pose[1] if c01_reset_pose is not None else None
        )
        c04_reset_layout = (
            self._c04_reset_layout()
            if self.task_name == "t50_water_plant"
            else None
        )
        c04_reset_xy = (
            c04_reset_layout[0] if c04_reset_layout is not None else None
        )
        c04_reset_yaw = (
            c04_reset_layout[1] if c04_reset_layout is not None else None
        )
        c05_reset_xy = (
            self._c05_reset_xy_positions()
            if self.task_name == "t14_move_and_click_mouse"
            else None
        )
        c06_reset_xy = (
            self._c06_reset_xy_position()
            if self.task_name == "t23_photograph_logo"
            else None
        )
        c07_reset_layout = (
            self._c07_reset_layout()
            if self.task_name == "c07_pitcher_pour_cup"
            else None
        )
        c07_reset_xy = c07_reset_layout[0] if c07_reset_layout is not None else None
        c07_reset_yaw = c07_reset_layout[1] if c07_reset_layout is not None else None
        c08_reset_layout = (
            self._c08_reset_layout()
            if self.task_name == "c08_tongs_ice_cubes"
            else None
        )
        c08_reset_xy = c08_reset_layout[0] if c08_reset_layout is not None else None
        c08_reset_yaw = c08_reset_layout[1] if c08_reset_layout is not None else None
        a01_reset_xy = self._a01_reset_xy_positions()
        task_reset_xy = self._task_specific_reset_xy_positions()
        c_tool_grouped = self.task_name in _C_TOOL_GROUP_RANDOMIZED_TASKS
        c_tool_group_yaw = (
            float(self._random.uniform(*_C_TOOL_GROUP_YAW_BOUNDS_DEG))
            if c_tool_grouped
            and self.randomize
            and self.task_name != "c08_tongs_ice_cubes"
            else 0.0
        )
        c_tool_group_pivot = _C_TOOL_GROUP_PIVOTS.get(self.task_name)
        c_tool_group_yaw_rad = math.radians(c_tool_group_yaw)
        c_tool_group_rotation = np.asarray(
            [
                [math.cos(c_tool_group_yaw_rad), -math.sin(c_tool_group_yaw_rad)],
                [math.sin(c_tool_group_yaw_rad), math.cos(c_tool_group_yaw_rad)],
            ],
            dtype=np.float64,
        )

        reset_state_by_path = {
            path: (view, indices, pose, velocity)
            for path, view, indices, pose, velocity in self._task_reset_body_states
        }
        reset_articulation_by_path = {
            path: (view, indices, root_pose, root_velocity)
            for (
                path,
                view,
                indices,
                root_pose,
                root_velocity,
                _joint_pos,
                _joint_vel,
            ) in self._task_reset_articulation_states
        }

        table_state = None
        table_prim = self._find_rigid_body_prim("table")
        if table_prim is not None:
            table_state = reset_state_by_path.get(str(table_prim.GetPath()))
        applied_height = float(self.delta_h[0]) if table_state is not None else 0.0
        if table_state is None and abs(float(self.delta_h[0])) > 1e-12:
            self.delta_h[:] = 0.0
            self._add_task_state_warning(
                "Table rigid body is unavailable; table-height randomization was skipped"
            )
        elif table_state is not None:
            view, indices, root_pose, root_velocity = table_state
            randomized_pose = _randomized_physx_pose(
                root_pose.detach().cpu().numpy(),
                xy_offset=np.zeros(2, dtype=np.float64),
                z_offset=applied_height,
                yaw_degrees=0.0,
            )
            pose_tensor = self._torch.as_tensor(
                randomized_pose,
                dtype=root_pose.dtype,
                device=root_pose.device,
            )
            self._write_rigid_reset_state(
                str(table_prim.GetPath()),
                view,
                indices,
                pose_tensor,
                self._torch.zeros_like(root_velocity),
            )

        common_xy = (
            self._random.uniform(
                -_TASK_OBJECT_COMMON_XY_JITTER,
                _TASK_OBJECT_COMMON_XY_JITTER,
                size=2,
            )
            if self.randomize
            and self.task_name
            not in {
                "t08_hammer_nail",
                "t50_water_plant",
                "t14_move_and_click_mouse",
                "t23_photograph_logo",
                "c07_pitcher_pour_cup",
                "c08_tongs_ice_cubes",
            }
            else np.zeros(2, dtype=np.float64)
        )
        c12_reset_xy = (
            self._c12_randomized_block_xy(
                c_tool_group_rotation,
                c_tool_group_pivot,
                common_xy,
            )
            if self.task_name == "t11_line_up_blocks"
            and self.randomize
            and c_tool_group_pivot is not None
            else None
        )
        if self.task_name == "t08_hammer_nail":
            # The nail is a prismatic child of the free wood articulation.
            # Move the articulation root so the nail stays seated in the wood.
            logical_names = ["hammer", "wood"]
        elif self.task_name == "t23_photograph_logo":
            # The logo is fixed to the table and already follows its height.
            logical_names = list(self._spec.target_objects)
        elif self.task_name == "c07_pitcher_pour_cup":
            logical_names = list(self._spec.target_objects) + [
                self._spec.receptacle,
                "c07_return_zone",
            ]
        else:
            logical_names = list(self._spec.target_objects)
            if self._spec.receptacle != "table":
                logical_names.append(self._spec.receptacle)
        if c_tool_grouped:
            logical_names.extend(sorted(self._task_extra_pose_names()))
            logical_names = list(dict.fromkeys(logical_names))
        if self.task_name == "t36_scoop_marbles":
            logical_names.append("spoon")

        randomized_paths: set[str] = set()
        randomized_names: list[str] = []
        b03_peg_randomized_pose: np.ndarray | None = None
        for name in logical_names:
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                continue
            path = str(prim.GetPath())
            if path in randomized_paths:
                continue
            state = self._reset_transform_state_for_path(
                path,
                reset_state_by_path,
                reset_articulation_by_path,
            )
            if state is None:
                continue
            state_kind, view, indices, root_pose, root_velocity = state
            relative_xy = (
                self._random.uniform(
                    -_TASK_OBJECT_RELATIVE_XY_JITTER,
                    _TASK_OBJECT_RELATIVE_XY_JITTER,
                    size=2,
                )
                if self.randomize
                and self.task_name
                not in {
                    "t08_hammer_nail",
                    "t50_water_plant",
                    "c07_pitcher_pour_cup",
                    "c08_tongs_ice_cubes",
                }
                else np.zeros(2, dtype=np.float64)
            )
            if c_tool_grouped and name not in self._spec.target_objects:
                relative_xy = np.zeros(2, dtype=np.float64)
            if self.task_name == "t09_insert_peg":
                yaw_degrees = self._b03_reset_yaw_degrees(name)
            elif self.task_name == "t08_hammer_nail":
                yaw_degrees = (
                    float(c01_hammer_yaw)
                    if name == "hammer"
                    else 0.0
                )
            elif self.task_name == "t50_water_plant":
                yaw_degrees = (
                    float(c04_reset_yaw)
                    if c04_reset_yaw is not None
                    and name in self._spec.target_objects
                    else 0.0
                )
            elif self.task_name == "t23_photograph_logo":
                yaw_degrees = 0.0
            elif c07_reset_yaw is not None:
                yaw_degrees = float(c07_reset_yaw.get(name, 0.0))
            elif c08_reset_yaw is not None:
                yaw_degrees = float(c08_reset_yaw.get(name, 0.0))
            elif (
                self.task_name == "t11_line_up_blocks"
                and name in _C12_BLOCK_NAMES
            ):
                yaw_degrees = c_tool_group_yaw + (
                    float(self._random.uniform(*_C12_BLOCK_YAW_BOUNDS_DEG))
                    if self.randomize
                    else 0.0
                )
            elif c_tool_grouped:
                local_yaw = (
                    float(self._random.uniform(-12.0, 12.0))
                    if self.randomize and name in self._spec.target_objects
                    else 0.0
                )
                yaw_degrees = c_tool_group_yaw + local_yaw
            elif self.task_name == "t26_place_fruit_in_bucket":
                yaw_degrees = (
                    0.0
                    if name in _NO_RANDOMIZED_YAW_OBJECTS or not self.randomize
                    else float(self._random.uniform(*_A01_YAW_BOUNDS_DEG))
                )
            elif self.task_name == "t25_place_cup_on_plate":
                yaw_degrees = (
                    0.0
                    if name in _NO_RANDOMIZED_YAW_OBJECTS or not self.randomize
                    else float(self._random.uniform(*_A02_YAW_BOUNDS_DEG))
                )
            elif self.task_name == "t24_place_ball_in_bowl":
                yaw_degrees = (
                    float(self._random.uniform(*_A03_YAW_BOUNDS_DEG))
                    if self.randomize
                    else 0.0
                )
            elif self.task_name == "t04_cover_pitcher":
                yaw_degrees = (
                    0.0
                    if name in _NO_RANDOMIZED_YAW_OBJECTS or not self.randomize
                    else float(self._random.uniform(*_B01_YAW_BOUNDS_DEG))
                )
            elif self.task_name == "t14_move_and_click_mouse":
                yaw_degrees = (
                    float(
                        self._random.uniform(
                            _C05_MOUSE_YAW_BOUNDS_DEG[0],
                            _C05_MOUSE_YAW_BOUNDS_DEG[1],
                        )
                    )
                    if self.randomize and name in self._spec.target_objects
                    else 0.0
                )
            else:
                yaw_degrees = (
                    0.0
                    if name in _NO_RANDOMIZED_YAW_OBJECTS or not self.randomize
                    else float(
                        self._random.uniform(
                            _TASK_OBJECT_YAW_BOUNDS_DEG[0],
                            _TASK_OBJECT_YAW_BOUNDS_DEG[1],
                        )
                    )
                )
            xy_offset = common_xy + relative_xy
            if c_tool_grouped and c_tool_group_pivot is not None:
                root_pose_np = root_pose.detach().cpu().numpy()
                base_xy = np.asarray(root_pose_np[0, :2], dtype=np.float64)
                grouped_xy = (
                    c_tool_group_pivot
                    + c_tool_group_rotation @ (base_xy - c_tool_group_pivot)
                    + common_xy
                    + relative_xy
                )
                xy_offset = grouped_xy - base_xy
            randomized_pose = _randomized_physx_pose(
                root_pose.detach().cpu().numpy(),
                xy_offset=xy_offset,
                z_offset=applied_height,
                yaw_degrees=yaw_degrees,
            )
            randomized_pose[:, :2] = _constrain_task_xy_to_workspace(
                self.task_name,
                randomized_pose[:, :2],
            )
            if a01_reset_xy is not None and name in a01_reset_xy:
                randomized_pose[:, :2] = a01_reset_xy[name]
            if task_reset_xy is not None and name in task_reset_xy:
                randomized_pose[:, :2] = task_reset_xy[name]
            if c01_reset_xy is not None:
                randomized_pose[:, :2] = c01_reset_xy[name]
                if name == "hammer":
                    # The source pose leaves the lowest hammer-head collider
                    # only millimetres above the imported tabletop.  Give the
                    # first PhysX frame enough room to settle without choosing
                    # a depenetration direction through the table.
                    randomized_pose[:, 2] += _C01_HAMMER_RESET_Z_OFFSET
            if c04_reset_xy is not None:
                if name in self._spec.target_objects and "spray" in c04_reset_xy:
                    randomized_pose[:, :2] = c04_reset_xy["spray"]
                elif name == self._spec.receptacle and name in c04_reset_xy:
                    randomized_pose[:, :2] = c04_reset_xy[name]
            if c05_reset_xy is not None:
                reset_xy = (
                    c05_reset_xy[0]
                    if name in self._spec.target_objects
                    else c05_reset_xy[1]
                )
                randomized_pose[:, :2] = reset_xy
            if c06_reset_xy is not None and name in self._spec.target_objects:
                randomized_pose[:, :2] = c06_reset_xy
            if c07_reset_xy is not None and name in c07_reset_xy:
                randomized_pose[:, :2] = c07_reset_xy[name]
            if c08_reset_xy is not None and name in c08_reset_xy:
                randomized_pose[:, :2] = c08_reset_xy[name]
            if c12_reset_xy is not None and name in c12_reset_xy:
                randomized_pose[:, :2] = c12_reset_xy[name]
            if (
                self.task_name == "t09_insert_peg"
                and (name in self._spec.target_objects or name == self._spec.receptacle)
            ):
                randomized_pose = self._b03_reset_pose_for_name(name, randomized_pose)
            if (
                self.task_name == "t09_insert_peg"
                and name in self._spec.target_objects
                and b03_peg_randomized_pose is None
            ):
                randomized_pose = self._b03_sample_clear_peg_reset_pose(
                    randomized_pose,
                )
                b03_peg_randomized_pose = randomized_pose.copy()
            if (
                self.task_name == "t09_insert_peg"
                and name == self._spec.receptacle
                and b03_peg_randomized_pose is not None
            ):
                randomized_pose = self._b03_sample_clear_socket_reset_pose(
                    b03_peg_randomized_pose,
                    randomized_pose,
                )
            pose_tensor = self._torch.as_tensor(
                randomized_pose,
                dtype=root_pose.dtype,
                device=root_pose.device,
            )
            if state_kind == "articulation":
                view.set_root_transforms(pose_tensor, indices)
                view.set_root_velocities(
                    self._torch.zeros_like(root_velocity),
                    indices,
                )
            else:
                self._write_rigid_reset_state(
                    path,
                    view,
                    indices,
                    pose_tensor,
                    self._torch.zeros_like(root_velocity),
                )
            randomized_paths.add(path)
            randomized_names.append(name)

        print(
            "[HandsOn/IsaacLab] Randomized ABC object poses: "
            f"table_delta_h={applied_height:.4f}m, "
                f"common_xy=({common_xy[0]:+.4f}, {common_xy[1]:+.4f})m, "
                f"c01_hammer_yaw={c01_hammer_yaw if c01_hammer_yaw is not None else 'n/a'}, "
                f"c04_spray_xy={c04_reset_xy['spray'].round(4).tolist() if c04_reset_xy is not None else 'n/a'}, "
                f"c04_plant_xy={c04_reset_xy['plant'].round(4).tolist() if c04_reset_xy is not None else 'n/a'}, "
                f"c04_spray_yaw={c04_reset_yaw if c04_reset_yaw is not None else 'n/a'}, "
                f"c07_pitcher_xy={c07_reset_xy['c07_pitcher'].round(4).tolist() if c07_reset_xy is not None else 'n/a'}, "
                f"c07_cup_xy={c07_reset_xy['c07_cup'].round(4).tolist() if c07_reset_xy is not None else 'n/a'}, "
                f"c_tool_group_yaw={c_tool_group_yaw if c_tool_grouped else 'n/a'}, "
                f"objects={','.join(randomized_names) or 'none'}",
                flush=True,
            )

    @staticmethod
    def _free_body_names(task_mjcf: Path) -> tuple[str, ...]:
        root = ET.parse(task_mjcf).getroot()
        names: list[str] = []
        for body in root.iter("body"):
            has_free_joint = any(
                child.tag == "freejoint"
                or (child.tag == "joint" and child.attrib.get("type") == "free")
                for child in body
            )
            name = body.attrib.get("name")
            if has_free_joint and name:
                names.append(name)
        return tuple(names)

    def _task_state_candidate_names(self, task_mjcf: Path) -> set[str]:
        names = set(self._spec.target_objects) | {
            self._spec.receptacle,
            "wood",
            "table",
        }
        names.update(self._task_extra_pose_names())
        names.update(self._free_body_names(task_mjcf))
        return names

    def _configure_ground_truth_contact_reports(self, task_mjcf: Path) -> None:
        """Enable task-body reports before the first simulation reset."""

        if self._stage is None:
            raise RuntimeError(
                "PhysX contact recording requires the constructed USD stage"
            )
        prim_paths = []
        for name in sorted(self._task_state_candidate_names(task_mjcf)):
            prim = self._find_rigid_body_prim(name)
            if prim is not None:
                prim_paths.append(str(prim.GetPath()))
        self._physx_contact_sensor_paths_preconfigured = tuple(
            dict.fromkeys(prim_paths)
        )
        self._physx_contact_report_paths_preconfigured = (
            configure_physx_contact_reports_for_recording(
                self._stage,
                prim_paths,
            )
        )

    def _capture_task_scene_state(self, task_mjcf: Path) -> None:
        """Cache live PhysX state so reset does not need to reload the stage."""
        self._task_kinematic_body_paths.clear()
        self._task_articulation_views.clear()
        if self._stage is None or not self._robots:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_sim_view is None:
            self._add_task_state_warning("PhysX simulation view is unavailable")
            return

        from pxr import Usd, UsdPhysics

        task_root = self._stage.GetPrimAtPath(_TASK_ROOT)
        for prim in Usd.PrimRange(task_root):
            if not prim.IsActive() or not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                continue
            path = str(prim.GetPath())
            # The MJCF importer marks its structural worldBody container as an
            # articulation root even though PhysX does not create an
            # articulation for it.  Querying it only produces tensor errors;
            # the real task articulations are its descendants and are cached
            # independently below.
            if path == f"{_TASK_ROOT}/worldBody":
                continue
            if path.startswith(f"{_TASK_ROOT}/base_right") or path.startswith(f"{_TASK_ROOT}/base_left"):
                continue
            try:
                view = physics_sim_view.create_articulation_view(path)
                if int(view.count) < 1:
                    continue
                indices = self._torch.arange(
                    int(view.count),
                    dtype=self._torch.long,
                    device=self._isaac_device,
                )
                joint_pos = view.get_dof_positions().clone()
                # The MJCF importer also tags free rigid bodies (including the
                # B03 peg and socket) as articulation roots with zero DOFs.
                # They are the same PhysX actors exposed by the rigid-body
                # views built below, so keeping both would restore each root
                # twice.  Only cache genuine reduced-coordinate articulations
                # here; zero-DOF free bodies use the rigid-body path.
                if int(joint_pos.shape[-1]) == 0:
                    continue
                self._task_reset_articulation_states.append(
                    (
                        path,
                        view,
                        indices,
                        view.get_root_transforms().clone(),
                        view.get_root_velocities().clone(),
                        joint_pos,
                        view.get_dof_velocities().clone(),
                    )
                )
                metatype = view.get_metatype(0)
                for dof_name, dof_index in metatype.dof_indices.items():
                    index = int(dof_index)
                    self._task_dof_views.setdefault(
                        str(dof_name),
                        (view, index, float(joint_pos[0, index].item())),
                    )
            except Exception as exc:
                self._add_task_state_warning(f"Could not snapshot articulation {path}: {exc}")

        articulation_root_paths = {
            path for path, *_rest in self._task_reset_articulation_states
        }
        articulation_view_by_path = {
            path: (view, indices)
            for path, view, indices, *_rest in self._task_reset_articulation_states
        }
        articulation_container_paths = {
            path.rsplit("/", 1)[0] for path in articulation_root_paths
        }

        logical_names = set(self._spec.target_objects) | {
            self._spec.receptacle,
            "wood",
            "table",
        }
        logical_names.update(self._task_extra_pose_names())
        candidate_names = self._task_state_candidate_names(task_mjcf)
        reset_view_by_path: dict[str, Any] = {}
        for name in sorted(candidate_names):
            prim = self._find_rigid_body_prim(name)
            if prim is None:
                continue
            path = str(prim.GetPath())
            matching_articulation_paths = [
                root_path
                for root_path in articulation_root_paths
                if path == root_path or path.startswith(f"{root_path}/")
            ]
            if matching_articulation_paths:
                root_path = max(
                    matching_articulation_paths,
                    key=len,
                )
                self._task_articulation_views[name] = (
                    articulation_view_by_path[root_path]
                )
            view = reset_view_by_path.get(path)
            if view is None:
                try:
                    view = physics_sim_view.create_rigid_body_view(path)
                    if int(view.count) < 1:
                        continue
                    rigid_api = UsdPhysics.RigidBodyAPI(prim)
                    kinematic_attr = rigid_api.GetKinematicEnabledAttr()
                    if kinematic_attr.IsValid() and bool(kinematic_attr.Get()):
                        self._task_kinematic_body_paths.add(path)
                    indices = self._torch.arange(
                        int(view.count),
                        dtype=self._torch.long,
                        device=self._isaac_device,
                    )
                    reset_view_by_path[path] = view
                    parent_path = path.rsplit("/", 1)[0]
                    is_non_root_articulation_link = bool(
                        path not in articulation_root_paths
                        and parent_path in articulation_container_paths
                    )
                    if not is_non_root_articulation_link:
                        self._task_reset_body_states.append(
                            (
                                path,
                                view,
                                indices,
                                view.get_transforms().clone(),
                                view.get_velocities().clone(),
                            )
                        )
                except Exception as exc:
                    self._add_task_state_warning(f"Could not snapshot rigid body {path}: {exc}")
                    continue
            self._task_body_views[name] = view

        # Bind logical aliases that resolve to an already snapshotted body.
        for name in sorted(logical_names):
            if name in self._task_body_views:
                continue
            prim = self._find_rigid_body_prim(name)
            if prim is not None:
                view = reset_view_by_path.get(str(prim.GetPath()))
                if view is not None:
                    self._task_body_views[name] = view

        print(
            "[HandsOn/IsaacLab] Cached task reset state: "
            f"{len(self._task_reset_body_states)} rigid bodies, "
            f"{len(self._task_reset_articulation_states)} articulations",
            flush=True,
        )

    def _restore_task_scene_state(self) -> None:
        for (
            path,
            view,
            indices,
            root_pose,
            root_velocity,
            joint_pos,
            joint_vel,
        ) in self._task_reset_articulation_states:
            try:
                view.set_root_transforms(root_pose, indices)
                view.set_root_velocities(root_velocity, indices)
                if joint_pos.shape[-1] > 0:
                    view.set_dof_positions(joint_pos, indices)
                    view.set_dof_velocities(joint_vel, indices)
            except Exception as exc:
                self._add_task_state_warning(f"Could not restore articulation {path}: {exc}")
        for path, view, indices, pose, velocity in self._task_reset_body_states:
            try:
                self._write_rigid_reset_state(
                    path,
                    view,
                    indices,
                    pose,
                    velocity,
                )
            except Exception as exc:
                self._add_task_state_warning(f"Could not restore rigid body {path}: {exc}")

    def _add_task_state_warning(self, detail: str) -> None:
        message = f"Task reset state warning: {detail}"
        if message not in self._isaac_warnings:
            self._isaac_warnings.append(message)
            print(f"[HandsOn/IsaacLab] {message}", flush=True)

    def _find_rigid_body_prim(self, name: str):
        if self._stage is None:
            return None
        from pxr import UsdPhysics

        prim = self._find_prim(name)
        while prim is not None and prim.IsValid() and str(prim.GetPath()).startswith(_TASK_ROOT):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim
            prim = prim.GetParent()
        return None

    def _init_recording_cameras(self) -> None:
        if not self.image_obs:
            return
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.sensors import Camera, CameraCfg

            legacy_wrist_cameras = _bool_env(
                "HANDSON_ISAACLAB_LEGACY_WRIST_CAMERAS",
                False,
            )
            self._legacy_wrist_cameras = legacy_wrist_cameras
            head_horizontal_aperture = (
                _RECORDING_CAMERA_VERTICAL_APERTURE
                * float(self._render_width)
                / float(self._render_height)
            )
            wrist_width, wrist_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
            wrist_horizontal_aperture = (
                _RECORDING_CAMERA_VERTICAL_APERTURE
                * float(wrist_width)
                / float(wrist_height)
            )
            camera_sizes = {
                "head_camera": (self._render_width, self._render_height),
                "wrist_left_camera": WRIST_RECORDING_CAMERA_OUTPUT_SIZE,
                "wrist_right_camera": WRIST_RECORDING_CAMERA_OUTPUT_SIZE,
            }
            camera_lenses = {
                "head_camera": {
                    "focal_length": _focal_length_for_vertical_fov(
                        _HEAD_RECORDING_CAMERA_FOVY
                    ),
                    "focus_distance": 1.2,
                    "horizontal_aperture": head_horizontal_aperture,
                    "vertical_aperture": _RECORDING_CAMERA_VERTICAL_APERTURE,
                    "clipping_range": (0.0008, 10.0),
                },
                "wrist_left_camera": {
                    "focal_length": (
                        _focal_length_for_vertical_fov(60.0)
                        if legacy_wrist_cameras
                        else _focal_length_for_horizontal_fov(
                            _WRIST_RECORDING_CAMERA_HFOV,
                            wrist_horizontal_aperture,
                        )
                    ),
                    "focus_distance": 0.55,
                    "horizontal_aperture": (
                        head_horizontal_aperture
                        if legacy_wrist_cameras
                        else wrist_horizontal_aperture
                    ),
                    "vertical_aperture": _RECORDING_CAMERA_VERTICAL_APERTURE,
                    "clipping_range": (
                        (0.0008, 5.0)
                        if legacy_wrist_cameras
                        else (WRIST_RECORDING_CAMERA_NEAR_M, 5.0)
                    ),
                },
                "wrist_right_camera": {
                    "focal_length": (
                        _focal_length_for_vertical_fov(60.0)
                        if legacy_wrist_cameras
                        else _focal_length_for_horizontal_fov(
                            _WRIST_RECORDING_CAMERA_HFOV,
                            wrist_horizontal_aperture,
                        )
                    ),
                    "focus_distance": 0.55,
                    "horizontal_aperture": (
                        head_horizontal_aperture
                        if legacy_wrist_cameras
                        else wrist_horizontal_aperture
                    ),
                    "vertical_aperture": _RECORDING_CAMERA_VERTICAL_APERTURE,
                    "clipping_range": (
                        (0.0008, 5.0)
                        if legacy_wrist_cameras
                        else (WRIST_RECORDING_CAMERA_NEAR_M, 5.0)
                    ),
                },
            }
            self._camera_aliases.clear()
            self._dynamic_cameras = set(_RECORDING_CAMERA_NAMES)
            legacy_wrist_source_desc = ""

            camera_sources: dict[str, str | None] = {
                name: _RECORDING_CAMERA_PRIM_PATHS[name]
                for name in _RECORDING_CAMERA_NAMES
            }
            if legacy_wrist_cameras:
                right_wrist_prim = self._native_camera_prim_path(
                    "right",
                    ("wrist_fisheye_right", "handcam_rgb_right", "handcam_rgb"),
                )
                left_wrist_prim = self._native_camera_prim_path(
                    "left",
                    ("wrist_fisheye_left", "handcam_rgb_left"),
                )
                camera_sources = {
                    "head_camera": _RECORDING_CAMERA_PRIM_PATHS["head_camera"],
                    "wrist_right_camera": right_wrist_prim,
                    "wrist_left_camera": (
                        left_wrist_prim
                        if self._spec.bimanual
                        else right_wrist_prim
                    ),
                }
                self._dynamic_cameras = {"head_camera"}
                if not self._spec.bimanual:
                    self._camera_aliases["wrist_left_camera"] = "wrist_right_camera"
                right_source = "native" if right_wrist_prim is not None else "dynamic"
                if "wrist_left_camera" in self._camera_aliases:
                    left_source = "alias:right"
                else:
                    left_source = "native" if left_wrist_prim is not None else "dynamic"
                legacy_wrist_source_desc = (
                    f", wrist_sources=left:{left_source},right:{right_source}"
                )

            for name in _RECORDING_CAMERA_NAMES:
                if name in self._camera_aliases:
                    continue
                camera_width, camera_height = camera_sizes[name]
                prim_path = camera_sources[name]
                spawn = sim_utils.PinholeCameraCfg(**camera_lenses[name])
                if legacy_wrist_cameras and name.startswith("wrist_"):
                    if prim_path is None:
                        prim_path = _RECORDING_CAMERA_PRIM_PATHS[name]
                        self._dynamic_cameras.add(name)
                    else:
                        spawn = None
                cfg = CameraCfg(
                    prim_path=prim_path,
                    update_period=0.0,
                    height=camera_height,
                    width=camera_width,
                    data_types=["rgb"],
                    update_latest_camera_pose=True,
                    spawn=spawn,
                )
                self._cameras[name] = Camera(cfg)

            if legacy_wrist_cameras:
                print(
                    "[HandsOn/IsaacLab] Legacy recording cameras enabled: "
                    f"head={self._render_width}x{self._render_height}, "
                    f"wrists={wrist_width}x{wrist_height} 60deg VFOV, "
                    "native MJCF wrist prim binding, "
                    f"wrist_left_alias={self._camera_aliases.get('wrist_left_camera', '')}"
                    f"{legacy_wrist_source_desc}",
                    flush=True,
                )
            else:
                print(
                    "[HandsOn/IsaacLab] V6 recording cameras enabled: "
                    f"head={self._render_width}x{self._render_height}, "
                    f"wrists={wrist_width}x{wrist_height} "
                    f"{_WRIST_RECORDING_CAMERA_HFOV:g}deg HFOV, "
                    "independent circular-mount tracking",
                    flush=True,
                )
        except Exception as exc:
            raise RuntimeError("Native recording camera construction failed") from exc

    def _native_camera_prim_path(
        self,
        side: str,
        names: tuple[str, ...],
    ) -> str | None:
        if self._stage is None:
            return None
        from pxr import Usd, UsdGeom

        root_path = f"/World/HandsOnRobots/{side.capitalize()}Robot"
        root = self._stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return None
        cameras = {
            prim.GetName(): str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.IsA(UsdGeom.Camera)
        }
        for name in names:
            if name in cameras:
                return cameras[name]
        return None

    def reset(self, seed: int | None = None, **kwargs: Any):
        target_label = kwargs.get("target_label") or kwargs.get("target_object")
        if target_label:
            self._target_object_name = self._resolve_target_object(str(target_label))
        return super().reset(seed=seed)

    def restore_recorded_initial_state(
        self,
        state_parts: Mapping[str, np.ndarray],
    ) -> dict[str, Any]:
        """Restore recorded table and task-object poses before action replay."""
        selector = np.asarray(
            state_parts.get("target_selector", np.zeros(0)),
            dtype=np.float64,
        ).ravel()
        if selector.size:
            target_index = int(
                np.clip(
                    np.argmax(selector),
                    0,
                    len(self._spec.target_objects) - 1,
                )
            )
            self._target_object_name = self._spec.target_objects[target_index]

        delta = np.asarray(
            state_parts.get("table_delta_height", self.delta_h),
            dtype=np.float64,
        ).ravel()
        if delta.size:
            self.delta_h[:] = float(delta[0])
            self._restore_recorded_table_height(float(delta[0]))

        recorded_poses: dict[str, np.ndarray] = {}
        object_poses = np.asarray(
            state_parts.get("object_poses", np.zeros((0, 7))),
            dtype=np.float64,
        ).reshape(-1, 7)
        for name, pose in zip(self._spec.target_objects, object_poses):
            recorded_poses[name] = pose

        if not selector.size and object_poses.size and "target_object_pose" in state_parts:
            target_pose = np.asarray(
                state_parts.get("target_object_pose", np.zeros(0)),
                dtype=np.float64,
            ).ravel()
            if target_pose.size >= 3:
                candidates = list(zip(self._spec.target_objects, object_poses))
                if candidates:
                    distances = np.asarray(
                        [
                            np.linalg.norm(np.asarray(pose[:3], dtype=np.float64) - target_pose[:3])
                            for _name, pose in candidates
                        ],
                        dtype=np.float64,
                    )
                    finite = np.isfinite(distances)
                    if np.any(finite):
                        target_index = int(np.argmin(np.where(finite, distances, np.inf)))
                        self._target_object_name = candidates[target_index][0]

        for state_key, role in self._spec.aliases.items():
            if state_key not in state_parts:
                continue
            if role == "target":
                name = self._target_object_name
            elif role == "receptacle":
                name = self._spec.receptacle
            else:
                name = role
            pose = np.asarray(state_parts[state_key], dtype=np.float64).ravel()
            if pose.size >= 7:
                recorded_poses[name] = pose[:7]

        restored_views: set[int] = set()
        for name, pose in recorded_poses.items():
            view = self._task_body_views.get(name)
            if view is None or id(view) in restored_views:
                continue
            if self._set_task_body_world_pose(view, pose):
                restored_views.add(id(view))

        if self._sim is not None:
            self._sim.forward()
        self._refresh_object_poses()
        return self._compute_observation()

    def restore_recorded_visual_state(
        self,
        state_parts: Mapping[str, np.ndarray],
        *,
        robot_iterations: int = 8,
    ) -> dict[str, Any]:
        """Restore recorded task objects and drive robot visuals toward state.

        Historical ABC zarr episodes only store TCP and hand command state, not
        the complete PhysX articulation qpos.  This method is therefore a
        best-effort visual reconstruction path: it restores task objects exactly
        and repeatedly solves/writes the robot joint targets from the recorded
        TCP + gripper state before rendering.
        """
        self.restore_recorded_initial_state(state_parts)
        self._restore_recorded_robot_command_state(state_parts)
        self._snap_robot_to_command_state(robot_iterations)
        if self._sim is not None:
            self._sim.forward()
        if self._scene is not None:
            try:
                self._scene.update(0.0)
            except Exception:
                pass
        self._refresh_object_poses()
        return self._compute_observation()

    def evaluate_restored_physx_replay_state(self) -> dict[str, Any]:
        """Evaluate a trusted recorded snapshot without advancing PhysX.

        Contact impulses are not part of the ground-truth state contract.  C04
        therefore reconstructs its spray latch from the recorded trigger DOF;
        object aiming and every other success condition are still recomputed
        from the restored simulator poses.
        """

        self._refresh_object_poses()
        if self.task_name == "t50_water_plant":
            reading = self._physics_dof_position("joint_0")
            if reading is not None:
                trigger_position, _initial_position = reading
                emitting = self._c04_spray_should_be_visible(
                    trigger_position,
                    self._c04_water_emitting,
                )
                if trigger_position <= _C04_TRIGGER_RELEASE_RAD:
                    self._c04_trigger_cycle_latched = False
                elif emitting:
                    self._c04_trigger_cycle_latched = True
                self._c04_trigger_pulled = bool(emitting)
                self._c04_water_emitting = bool(emitting)
                self._c04_trigger_phase = (
                    "pressed" if emitting else "pressing"
                )
                self._update_c04_spray_visual()

        success = bool(self._compute_success())
        failed = bool(self._compute_failure())
        info: dict[str, Any] = {
            "env_succeed": success,
            "failed": failed,
        }
        info.update(self._task_info())
        return info

    def _restore_recorded_robot_command_state(
        self,
        state_parts: Mapping[str, np.ndarray],
    ) -> None:
        tcp_pose = np.asarray(
            state_parts.get("tcp_pose", np.zeros(0)),
            dtype=np.float64,
        ).ravel()
        if tcp_pose.size >= 7:
            right_pose = tcp_pose[:7].astype(np.float64, copy=True)
            if np.all(np.isfinite(right_pose)) and not np.allclose(right_pose[:7], 0.0):
                quat_norm = float(np.linalg.norm(right_pose[3:7]))
                if quat_norm > 1e-8:
                    right_pose[3:7] /= quat_norm
                    self._right_tcp_pose = right_pose
                    self._right_pose_commanded = True
                    self._arm_pose_filters["right"] = None
        if tcp_pose.size >= 14:
            left_pose = tcp_pose[7:14].astype(np.float64, copy=True)
            if np.all(np.isfinite(left_pose)) and not np.allclose(left_pose[:7], 0.0):
                quat_norm = float(np.linalg.norm(left_pose[3:7]))
                if quat_norm > 1e-8:
                    left_pose[3:7] /= quat_norm
                    self._left_tcp_pose = left_pose
                    self._left_pose_commanded = True
                    self._arm_pose_filters["left"] = None

        gripper_pose = np.asarray(
            state_parts.get("gripper_pose", np.zeros(0)),
            dtype=np.float64,
        ).ravel()
        if not gripper_pose.size:
            return

        if self._spec.bimanual and gripper_pose.size >= 32:
            right_size = min(16, self._right_hand.size)
            left_size = min(16, self._left_hand.size)
            if right_size:
                self._right_hand[:right_size] = gripper_pose[:right_size]
            if left_size:
                self._left_hand[:left_size] = gripper_pose[16 : 16 + left_size]
            return

        right_size = min(self._right_hand.size, gripper_pose.size)
        if right_size:
            self._right_hand[:right_size] = gripper_pose[:right_size]

    def _snap_robot_to_command_state(self, robot_iterations: int) -> None:
        if not self._robots or self._torch is None:
            return
        iterations = max(1, int(robot_iterations))
        for _ in range(iterations):
            self._apply_joint_targets()
            self._write_joint_targets_as_state()
            if self._sim is not None:
                self._sim.forward()
            if self._scene is not None:
                try:
                    self._scene.update(0.0)
                except Exception:
                    pass
            for robot in self._robots.values():
                try:
                    robot.update(0.0)
                except Exception:
                    pass

    def _write_joint_targets_as_state(self) -> None:
        for side, robot in self._robots.items():
            target = self._joint_targets.get(side)
            if target is None:
                continue
            try:
                joint_pos = target.clone()
                joint_vel = self._torch.zeros_like(joint_pos)
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.set_joint_position_target(joint_pos)
                robot.write_data_to_sim()
            except Exception as exc:
                message = f"{side} robot visual state restore failed: {exc}"
                if message not in self._isaac_warnings:
                    self._isaac_warnings.append(message)
                    print(f"[HandsOn/IsaacLab] {message}", flush=True)

    def _restore_recorded_table_height(self, delta_height: float) -> bool:
        table_prim = self._find_rigid_body_prim("table")
        if table_prim is None:
            return False
        table_path = str(table_prim.GetPath())
        for path, view, indices, root_pose, root_velocity in self._task_reset_body_states:
            if path != table_path:
                continue
            restored_pose = _randomized_physx_pose(
                root_pose.detach().cpu().numpy(),
                xy_offset=np.zeros(2, dtype=np.float64),
                z_offset=float(delta_height),
                yaw_degrees=0.0,
            )
            pose_tensor = self._torch.as_tensor(
                restored_pose,
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
            return True
        return False

    def _set_task_body_world_pose(self, view: Any, pose_wxyz: np.ndarray) -> bool:
        if self._torch is None:
            return False
        try:
            transforms = view.get_transforms().clone()
            velocities = view.get_velocities().clone()
            original_transforms = transforms.clone()
            original_velocities = velocities.clone()
            pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
            transforms[:, :3] = self._torch.as_tensor(
                pose[:3],
                dtype=transforms.dtype,
                device=transforms.device,
            )
            transforms[:, 3:7] = self._torch.as_tensor(
                [pose[4], pose[5], pose[6], pose[3]],
                dtype=transforms.dtype,
                device=transforms.device,
            )
            indices = self._torch.arange(
                int(view.count),
                dtype=self._torch.long,
                device=self._isaac_device,
            )
            try:
                # A velocity failure must not leave a transform-only teleport.
                view.set_velocities(
                    self._torch.zeros_like(velocities),
                    indices,
                )
                view.set_transforms(transforms, indices)
            except Exception:
                try:
                    view.set_velocities(original_velocities, indices)
                    view.set_transforms(original_transforms, indices)
                except Exception:
                    pass
                raise
            return True
        except Exception as exc:
            self._add_task_state_warning(f"Could not restore recorded object pose: {exc}")
            return False

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        table_height_bounds = _TASK_TABLE_HEIGHT_BOUNDS.get(
            self.task_name,
            (0.0, 0.05),
        )
        self.delta_h = np.asarray(
            [self._random.uniform(*table_height_bounds) if self.randomize else 0.0],
            dtype=np.float64,
        )
        self._right_tcp_pose = _RIGHT_HOME.copy()
        self._left_tcp_pose = _LEFT_HOME.copy()
        if self.task_name in _C_TOOL_CLEAR_START_TASKS:
            self._right_tcp_pose = _C_TOOL_SPREAD_RIGHT_START_TCP.copy()
            self._left_tcp_pose = _C_TOOL_SPREAD_LEFT_START_TCP.copy()
        elif self.task_name in _RAISED_INACTIVE_LEFT_TASKS:
            self._left_tcp_pose = _RAISED_LEFT_START_TCP.copy()
        self._right_hand.fill(0.0)
        self._left_hand.fill(0.0)
        self._right_pose_commanded = False
        self._left_pose_commanded = False
        self.env_step = 0
        self._success_counter = 0
        self._c02_spoon_used = False
        self._c02_spoon_loaded_seen = False
        self._c02_spoon_over_bowl_seen = False
        self._c02_spoon_start_pos = None
        self._lid_start_pose = None
        self._lid_was_moved = False
        self._b01_hand_was_closed = False
        self._b01_release_assist_active = False
        self._b01_hand_lid_clear_steps = 0
        self._last_b01_lid_success_metrics = {}
        self._b02_prev_relative_yaw = None
        self._b02_signed_twist_deg = 0.0
        self._b02_thread_entry_depth = None
        self._b02_min_thread_depth = None
        self._b02_thread_engaged_seen = False
        self._clear_b02_nut_grasp_assist("reset")
        self._c01_prev_nail_depth = None
        self._c01_prev_hammer_pose = None
        self._c01_prev_nail_pose = None
        self._c01_had_valid_hit = False
        self._c01_max_valid_hit_depth = 0.0
        self._c01_candidate_strike_depth_start = None
        self._c01_candidate_strike_max_depth = 0.0
        self._c01_last_valid_hit_depth_increment = 0.0
        self._c01_last_candidate_hit_depth_increment = 0.0
        self._c01_hit_contact_available = False
        self._c01_last_hit_contact_force = 0.0
        self._c01_last_face_nail_distance = math.inf
        self._c01_last_face_nail_radial_error = math.inf
        self._c01_last_face_nail_axial_offset = math.inf
        self._c01_last_face_down_alignment = -1.0
        self._c01_last_hammer_downward_speed = 0.0
        self._c01_last_hammer_motion_down_alignment = -1.0
        self._c01_grasp_assists = {}
        self._c01_grasp_candidates = {}
        self._c01_strike_align_prev_raw_tcp = None
        self._c01_strike_align_assist_metrics = {}
        self._last_a06_success_metrics = {}
        self._last_b04_stack_metrics = {}
        self._b04_velocity_samples = []
        self._b04_velocity_stable_latched = False
        self._reset_c04_trigger_cycle()
        self._c05_mouse_start_xy = None
        self._c05_button_initial_positions = {
            "mouse_joint_0": 0.0,
            "mouse_joint_1": 0.0,
        }
        self._c05_failed_early_click = False
        self._last_c05_success_metrics = {}
        self._c06_shutter_was_down = False
        self._c06_right_contact_grace = 0
        self._last_c06_success_metrics = {}
        self._last_c07_success_metrics = {}
        self._c07_tilt_seen = False
        self._c_assist_raw_right_hand = self._right_hand.copy()
        self._c_assist_raw_right_tcp_pose = self._right_tcp_pose.copy()
        self._c_assist_raw_left_hand = self._left_hand.copy()
        self._c_assist_raw_left_tcp_pose = self._left_tcp_pose.copy()
        self._c_assist_control_frame = 0
        self._c_assist_free_hand_handoffs = {}
        self._c07_pitcher_grasp = None
        self._c07_pitcher_pinned = False
        self._c07_pitcher_pinned_pose = None
        self._c07_delivered_particles = set()
        self._c07_previous_particle_cup_local = {}
        self._last_c08_success_metrics = {}
        self._c08_tongs_close_count = 0
        self._c08_tongs_was_open = False
        self._c08_tongs_grasp = None
        self._c08_tongs_control_armed = False
        self._c08_held_ice = None
        self._c08_delivered_ice = set()
        self._c08_delivered_ice_local = {}
        self._last_c09_success_metrics = {}
        self._c09_top_opened_seen = False
        self._c09_paper_grasps = {}
        self._c09_papers_stacked = False
        self._c09_stack_state = None
        self._c09_stapler_anchor = None
        self._last_c10_success_metrics = {}
        self._c10_assist_metrics = {}
        self._c10_connector_grasps = {}
        self._c10_first_connected_seen = False
        self._c10_order_violation = False
        self._c10_plug_latched = False
        self._c10_cable_latched = False
        self._c10_latch_stable_steps = {
            "c10_charger_plug": 0,
            "c10_charging_cable": 0,
        }
        self._c10_plug_latch_relative = None
        self._c10_cable_latch_relative = None
        self._last_c11_success_metrics = {}
        if self.task_name == "c11_xylophone_sequence":
            sequence_length = int(
                self._random.choice(
                    np.arange(
                        _C11_MIN_SEQUENCE_LENGTH,
                        _C11_MAX_SEQUENCE_LENGTH + 1,
                    )
                )
            )
            self._c11_target_sequence = tuple(
                str(self._random.choice(_C11_KEY_NAMES))
                for _ in range(sequence_length)
            )
        else:
            self._c11_target_sequence = _C11_DEFAULT_SEQUENCE
        self._c11_sequence_progress = 0
        self._c11_wrong_hits = 0
        self._c11_last_hit_key = None
        self._c11_current_hit_key = None
        self._c11_mallet_grasp = None
        self._c11_previous_head_world = None
        self._c11_pending_crossings = {}
        self._c11_hit_armed = True
        self._last_c12_success_metrics = {}
        self._c12_tool_start_xy = None
        self._c12_tool_used = False
        self._c12_direct_hand_contact = False
        self._c12_tool_grasp = None
        self._c12_active_push = None
        self._c12_push_candidate = None
        self._c12_slot_by_block = {}
        self._c12_settled_blocks = set()
        self._c12_settled_targets = {}
        self._c04_spray_visible = True
        self._set_c04_spray_visible(False)
        self._reset_c_tool_visual_feedback()
        self._update_c11_prompt_visuals()
        if clear_manual_request:
            self._manual_reset_requested = False
        self._arm_pose_filters = {"right": None, "left": None}
        self._refresh_object_poses()
        self._update_success_condition_overlay()

    def _apply_reset_visual_randomization(self) -> None:
        from .default_appearance import refresh_environment
        refresh_environment(self)

    def _c07_pour_hand_tracking_ready(
        self,
        state: dict[str, Any],
    ) -> bool:
        """Advance the shared hand/pitcher path only after the palm catches up."""
        target_value = state.get("pour_hand_target")
        if target_value is None:
            state["pour_hand_tracking_ready"] = True
            return True
        target = np.asarray(
            target_value,
            dtype=np.float64,
        ).reshape(4, 4)
        if not bool(state.get("pour_hand_target_installed")):
            palm = self._robot_link_pose_matrix("right")
            if palm is None:
                state["pour_hand_tracking_ready"] = False
                return False
            position_error = float(
                np.linalg.norm(palm[:3, 3] - target[:3, 3])
            )
            angle_error = float(
                np.linalg.norm(
                    self._rotation_error_vector(
                        target[:3, :3],
                        palm[:3, :3],
                    )
                )
            )
            ready = bool(
                position_error <= _C07_POUR_TRACK_POSITION_TOLERANCE
                and angle_error <= _C07_POUR_TRACK_ANGLE_TOLERANCE
            )
            state["pour_hand_tracking_position_error"] = position_error
            state["pour_hand_tracking_angle_error"] = angle_error
            state["pour_hand_tracking_ready"] = ready
            if ready:
                state["pour_hand_target_installed"] = True
                state["pour_hand_target_installed_matrix"] = target.copy()
                state["pour_hand_target_installed_frame"] = int(
                    self._c_assist_control_frame
                )
            return ready
        installed_value = state.get("pour_hand_target_installed_matrix")
        if installed_value is not None:
            installed = np.asarray(
                installed_value,
                dtype=np.float64,
            ).reshape(4, 4)
            if not np.allclose(installed, target, rtol=0.0, atol=1e-10):
                state["pour_hand_target_installed"] = False
                state["pour_hand_tracking_ready"] = False
                return False
        palm = self._robot_link_pose_matrix("right")
        if target_value is None or palm is None:
            # Missing readback must not corrupt the path state.  Isaac normally
            # provides the palm matrix; the fallback keeps non-simulator reuse
            # and shutdown paths from deadlocking.
            state["pour_hand_tracking_ready"] = True
            return True
        position_error = float(
            np.linalg.norm(palm[:3, 3] - target[:3, 3])
        )
        angle_error = float(
            np.linalg.norm(
                self._rotation_error_vector(
                    target[:3, :3],
                    palm[:3, :3],
                )
            )
        )
        ready = bool(
            position_error <= _C07_POUR_TRACK_POSITION_TOLERANCE
            and angle_error <= _C07_POUR_TRACK_ANGLE_TOLERANCE
        )
        state["pour_hand_tracking_position_error"] = position_error
        state["pour_hand_tracking_angle_error"] = angle_error
        state["pour_hand_tracking_ready"] = ready
        return ready

    def _postprocess_hand_action(self) -> None:
        """Preserve visual intent while task assists hold a stable tool grip."""
        if self.task_name not in {
            "t08_hammer_nail",
            "c07_pitcher_pour_cup",
            "c08_tongs_ice_cubes",
            "c09_staple_paper_sheets",
            "t02_connect_charger_and_cable",
            "c11_xylophone_sequence",
            "t11_line_up_blocks",
        }:
            return

        self._c_assist_control_frame += 1
        self._c_assist_raw_right_hand = self._right_hand.copy()
        self._c_assist_raw_right_tcp_pose = self._right_tcp_pose.copy()
        self._c_assist_raw_left_hand = self._left_hand.copy()
        self._c_assist_raw_left_tcp_pose = self._left_tcp_pose.copy()
        if self.task_name == "t08_hammer_nail":
            self._apply_c01_strike_align_assist()
            for side, assist_state in self._c01_grasp_assists.items():
                visual_hand = assist_state.get("visual_hand")
                if visual_hand is None:
                    continue
                visual = np.asarray(visual_hand, dtype=np.float64).reshape(-1)
                if side == "right" and visual.size == self._right_hand.size:
                    self._right_hand = visual.copy()
                elif side == "left" and visual.size == self._left_hand.size:
                    self._left_hand = visual.copy()
            return
        if self.task_name not in {
            "c07_pitcher_pour_cup",
            "c08_tongs_ice_cubes",
        }:
            # A guide may end many centimetres from the absolute tracker
            # mapping that entered it.  Keep the guide's final raw-to-hand
            # frame after releasing the object so the hand itself remains
            # continuous instead of snapping back on the next action.
            for side in ("right", "left"):
                handoff = self._c_assist_free_hand_handoffs.get(side)
                if handoff is None:
                    continue
                raw_pose = (
                    self._c_assist_raw_right_tcp_pose
                    if side == "right"
                    else self._c_assist_raw_left_tcp_pose
                )
                adjusted = self._map_raw_hand_handoff(
                    _pose_vector_to_matrix(raw_pose),
                    np.asarray(
                        handoff["raw_anchor"],
                        dtype=np.float64,
                    ).reshape(4, 4),
                    np.asarray(
                        handoff["hand_anchor"],
                        dtype=np.float64,
                    ).reshape(4, 4),
                )
                adjusted_pose = self._pose_matrix_to_vector(adjusted)
                if side == "right":
                    self._right_tcp_pose = adjusted_pose
                else:
                    self._left_tcp_pose = adjusted_pose
            if self.task_name == "c09_staple_paper_sheets":
                states = self._c09_paper_grasps
            elif self.task_name == "t02_connect_charger_and_cable":
                states = self._c10_connector_grasps
            elif self.task_name == "c11_xylophone_sequence":
                states = (
                    {}
                    if self._c11_mallet_grasp is None
                    else {"right": self._c11_mallet_grasp}
                )
            else:
                states = (
                    {}
                    if self._c12_tool_grasp is None
                    else {"right": self._c12_tool_grasp}
                )
            for side, assist_state in states.items():
                target_value = assist_state.get("hand_target_matrix")
                if target_value is not None:
                    target_matrix = np.asarray(
                        target_value,
                        dtype=np.float64,
                    ).reshape(4, 4)
                    target_pose = self._pose_matrix_to_vector(target_matrix)
                    if side == "right":
                        self._right_tcp_pose = target_pose
                    else:
                        self._left_tcp_pose = target_pose
                    assist_state["hand_target_installed"] = True
                    assist_state["hand_target_installed_matrix"] = (
                        target_matrix.copy()
                    )
                    assist_state["hand_target_installed_frame"] = int(
                        self._c_assist_control_frame
                    )
                else:
                    assist_state["hand_target_installed"] = False
                    assist_state.pop(
                        "hand_target_installed_matrix",
                        None,
                    )
                    assist_state.pop(
                        "hand_target_installed_frame",
                        None,
                    )
                visual_hand = assist_state.get("visual_hand")
                if visual_hand is None:
                    continue
                visual = np.asarray(visual_hand, dtype=np.float64).reshape(-1)
                if side == "right" and visual.size == self._right_hand.size:
                    self._right_hand = visual.copy()
                elif side == "left" and visual.size == self._left_hand.size:
                    self._left_hand = visual.copy()
            return

        state = (
            self._c07_pitcher_grasp
            if self.task_name == "c07_pitcher_pour_cup"
            else self._c08_tongs_grasp
        )
        if state is None:
            return
        if (
            self.task_name == "c07_pitcher_pour_cup"
            and bool(state.get("pour_xy_lock"))
        ):
            if state.get("phase") == "pour_aligning":
                alignment_from = state.get("pour_alignment_from_object")
                object_anchor = state.get("pour_object_anchor")
                hand_to_object = state.get("hand_to_object")
                if (
                    alignment_from is not None
                    and object_anchor is not None
                    and hand_to_object is not None
                ):
                    duration = max(
                        _C07_POUR_ALIGN_DURATION_SECONDS,
                        float(
                            state.get(
                                "pour_alignment_duration_s",
                                _C07_POUR_ALIGN_DURATION_SECONDS,
                            )
                        ),
                    )
                    committed_fraction = float(
                        np.clip(
                            state.get("pour_alignment_fraction", 0.0),
                            0.0,
                            1.0,
                        )
                    )
                    tracking_ready = self._c07_pour_hand_tracking_ready(
                        state
                    )
                    fraction = committed_fraction
                    if tracking_ready:
                        fraction = min(
                            1.0,
                            committed_fraction
                            + max(float(self._control_dt), 0.0)
                            / max(duration, 1e-9),
                        )
                    object_target = self._c07_interpolate_pose(
                        alignment_from,
                        object_anchor,
                        fraction,
                    )
                    hand_matrix = object_target @ _invert_pose_matrix(
                        np.asarray(
                            hand_to_object,
                            dtype=np.float64,
                        ).reshape(4, 4)
                    )
                    hand_pose = np.concatenate(
                        [
                            hand_matrix[:3, 3],
                            _matrix_to_quat_wxyz(
                                hand_matrix[:3, :3]
                            ),
                        ]
                    )
                    self._right_tcp_pose = hand_pose
                    state["pour_alignment_candidate_fraction"] = fraction
                    state["pour_hand_target"] = hand_matrix
                    state["pour_object_target"] = object_target
                    state["pour_target_ready"] = True
                    state["pour_hand_target_installed"] = True
                    state["pour_hand_target_installed_matrix"] = (
                        hand_matrix.copy()
                    )
                    state["pour_hand_target_installed_frame"] = int(
                        self._c_assist_control_frame
                    )
            elif state.get("phase") == "pour_locked":
                _requested_hand, requested_pitcher = (
                    self._c07_tilt_path_targets(
                        self._c_assist_raw_right_tcp_pose,
                        state,
                    )
                )
                if requested_pitcher is not None:
                    requested_tilt = (
                        self._c07_particle_release_tilt_degrees(
                            state,
                            requested_pitcher,
                        )
                    )
                    applied_tilt = float(
                        state.get("pour_applied_tilt_deg", 0.0)
                    )
                    tracking_ready = self._c07_pour_hand_tracking_ready(
                        state
                    )
                    max_delta = (
                        _C07_POUR_TILT_RATE_DEG_PER_S
                        * max(float(self._control_dt), 0.0)
                    )
                    candidate_tilt = applied_tilt
                    if tracking_ready:
                        candidate_tilt = applied_tilt + float(
                            np.clip(
                                requested_tilt - applied_tilt,
                                -max_delta,
                                max_delta,
                            )
                        )
                    candidate_hand, candidate_pitcher = (
                        self._c07_tilt_path_targets(
                            self._c_assist_raw_right_tcp_pose,
                            state,
                            tilt_degrees=candidate_tilt,
                        )
                    )
                    state["pour_requested_tilt_deg"] = float(
                        requested_tilt
                    )
                    if candidate_hand is not None:
                        self._right_tcp_pose = candidate_hand
                        candidate_hand_matrix = _pose_vector_to_matrix(
                            candidate_hand
                        )
                        state["pour_hand_target"] = candidate_hand_matrix
                        state["pour_hand_target_installed"] = True
                        state["pour_hand_target_installed_matrix"] = (
                            candidate_hand_matrix.copy()
                        )
                        state["pour_hand_target_installed_frame"] = int(
                            self._c_assist_control_frame
                        )
                    if candidate_pitcher is not None:
                        state["pour_object_target"] = candidate_pitcher
                        state["pour_candidate_tilt_deg"] = float(
                            candidate_tilt
                        )
                        state["pour_target_ready"] = True
        elif (
            self.task_name == "c07_pitcher_pour_cup"
            and bool(state.get("pour_complete"))
        ):
            rebased_hand = self._c07_post_pour_hand_target(
                self._c_assist_raw_right_tcp_pose,
                state,
            )
            if rebased_hand is not None:
                self._right_tcp_pose = rebased_hand
        elif self.task_name == "c08_tongs_ice_cubes":
            hand_target_value = state.get("hand_target_matrix")
            hand_to_object_value = state.get("hand_to_object")
            if (
                hand_target_value is None
                and not bool(state.get("pour_xy_lock"))
                and hand_to_object_value is not None
            ):
                tongs = self._current_body_matrix("c08_tongs")
                reading = self._physics_dof_position("c08_tongs_joint")
                if tongs is not None:
                    joint_angle = (
                        _C08_TONGS_REST_RAD
                        if reading is None
                        else float(reading[0])
                    )
                    hand_to_object = np.asarray(
                        hand_to_object_value,
                        dtype=np.float64,
                    ).reshape(4, 4)
                    requested_hand = _pose_vector_to_matrix(
                        self._right_tcp_pose
                    )
                    requested_tongs = requested_hand @ hand_to_object
                    (
                        safe_tongs,
                        command_blocked,
                        command_clearance,
                    ) = self._c08_collision_safe_tongs_target(
                        tongs,
                        requested_tongs,
                        joint_angle,
                    )
                    state["collision_guard_command_blocked"] = bool(
                        command_blocked
                    )
                    state["collision_guard_command_clearance"] = float(
                        command_clearance
                    )
                    if command_blocked:
                        safe_hand = safe_tongs @ _invert_pose_matrix(
                            hand_to_object
                        )
                        self._right_tcp_pose = self._pose_matrix_to_vector(
                            safe_hand
                        )
            if hand_target_value is not None:
                hand_target = np.asarray(
                    hand_target_value,
                    dtype=np.float64,
                ).reshape(4, 4)
                self._right_tcp_pose = self._pose_matrix_to_vector(
                    hand_target
                )
                state["drop_hand_pose"] = self._right_tcp_pose.copy()
                state["hand_target_installed"] = True
                state["hand_target_installed_matrix"] = hand_target.copy()
                state["hand_target_installed_frame"] = int(
                    self._c_assist_control_frame
                )
                state["pour_hand_target_installed"] = True
            elif bool(state.get("pour_xy_lock")) and state.get(
                "drop_hand_pose"
            ) is not None:
                self._right_tcp_pose = np.asarray(
                    state["drop_hand_pose"],
                    dtype=np.float64,
                ).reshape(7).copy()
                hand_target = _pose_vector_to_matrix(self._right_tcp_pose)
                state["hand_target_installed"] = True
                state["hand_target_installed_matrix"] = hand_target.copy()
                state["hand_target_installed_frame"] = int(
                    self._c_assist_control_frame
                )
                state["pour_hand_target_installed"] = True
            else:
                state["hand_target_installed"] = False
                state.pop("hand_target_installed_matrix", None)
                state.pop("hand_target_installed_frame", None)
                state["pour_hand_target_installed"] = False
        visual_hand = state.get("visual_hand")
        if visual_hand is not None:
            self._right_hand = np.asarray(
                visual_hand,
                dtype=np.float64,
            ).reshape(self._right_hand.shape).copy()

    def _step_isaaclab(self) -> None:
        if self._sim is None:
            return
        self._resume_timeline_if_paused(source="Timeline pause")
        self._update_b01_release_assist()
        for robot in self._robots.values():
            robot.write_data_to_sim()
        # Defer rendering until render(), after every physics substep has
        # finished and the wrist-mount body tensors contain the latest pose.
        self._sim.step(render=False)
        if self.task_name == "t08_hammer_nail":
            # A hammer impact can begin and end inside one 30 Hz control
            # interval. Sample it at the PhysX rate so a real strike is not
            # missed while still requiring the contact tensor below.
            self._refresh_object_poses()
            self._update_c01_hammer_hit_state()
        self._update_c04_trigger_state()
        self._update_c04_spray_visual()
        for robot in self._robots.values():
            robot.update(self._sim_dt)
        self._after_task_physics_substep()

    def _after_task_physics_substep(self) -> None:
        """Task-only high-frequency event hook; robot stepping stays shared."""
        if self.task_name == "t08_hammer_nail":
            self._update_c01_grasp_assists()
        elif self.task_name == "c07_pitcher_pour_cup":
            self._update_c07_pitcher_grasp_assist()
        elif self.task_name == "c08_tongs_ice_cubes":
            self._update_c08_tongs_assist()
        elif self.task_name == "c09_staple_paper_sheets":
            self._update_c09_paper_assist()
        elif self.task_name == "t02_connect_charger_and_cable":
            self._update_c10_connector_assist()
        elif self.task_name == "c11_xylophone_sequence":
            self._update_c11_mallet_assist()
            self._update_c11_hit_events()
        elif self.task_name == "t11_line_up_blocks":
            if (
                not self._c12_direct_hand_contact
                and self._c12_direct_hand_touch_now()
            ):
                self._c12_direct_hand_contact = True
                self._manual_reset_requested = True
                print(
                    "[HandsOn/IsaacLab] C12 direct hand/block contact; resetting episode",
                    flush=True,
                )
            elif not self._c12_direct_hand_contact:
                self._update_c12_tool_assist()

    def _hand_command_max(self, side: str) -> float:
        values = self._right_hand if side == "right" else self._left_hand
        if values.size == 0:
            return 0.0
        return float(np.max(np.abs(values)))

    def _raw_right_hand_command_max(self) -> float:
        values = np.asarray(
            getattr(self, "_c_assist_raw_right_hand", self._right_hand),
            dtype=np.float64,
        )
        if values.size == 0:
            return 0.0
        return float(np.max(np.abs(values)))

    def _actual_hand_joint_pose(self, side: str) -> np.ndarray:
        """Read a visible hand pose at contact, falling back to the raw glove."""
        raw_name = f"_c_assist_raw_{side}_hand"
        commanded = self._right_hand if side == "right" else self._left_hand
        fallback = np.asarray(
            getattr(self, raw_name, commanded),
            dtype=np.float64,
        ).reshape(-1).copy()
        robot = self._robots.get(side)
        joint_ids = self._hand_joint_ids.get(side, [])
        if robot is None or not joint_ids:
            return fallback
        try:
            values = robot.data.joint_pos[0, joint_ids]
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            actual = np.asarray(values, dtype=np.float64).reshape(-1)
        except Exception:
            return fallback
        if actual.size != fallback.size or not np.all(np.isfinite(actual)):
            return fallback
        return actual.copy()

    def _actual_right_hand_joint_pose(self) -> np.ndarray:
        """Compatibility wrapper for the established C07/C08 right hand."""
        return self._actual_hand_joint_pose("right")

    def _raw_hand_command_max(self, side: str) -> float:
        values = np.asarray(
            getattr(
                self,
                f"_c_assist_raw_{side}_hand",
                self._right_hand if side == "right" else self._left_hand,
            ),
            dtype=np.float64,
        )
        if values.size == 0:
            return 0.0
        return float(np.max(np.abs(values)))

    def _raw_tcp_pose(self, side: str) -> np.ndarray:
        fallback = self._right_tcp_pose if side == "right" else self._left_tcp_pose
        return np.asarray(
            getattr(self, f"_c_assist_raw_{side}_tcp_pose", fallback),
            dtype=np.float64,
        ).reshape(7).copy()

    @staticmethod
    def _tool_power_grip_pose(hand: np.ndarray) -> np.ndarray:
        values = np.asarray(hand, dtype=np.float64).reshape(-1)
        if values.size == _C_TOOL_WUJI_POWER_GRIP.size:
            return _C_TOOL_WUJI_POWER_GRIP.copy()
        result = values.copy()
        # The production C07/C08 path uses the 20-DoF Wuji hand.  Keep a
        # conservative grouped fallback for interface-only tests or another
        # four-joint-per-finger hand representation.
        for start in range(0, result.size - 3, 4):
            result[start] = max(float(result[start]), 0.60)
            result[start + 2] = max(float(result[start + 2]), 0.82)
            result[start + 3] = max(float(result[start + 3]), 0.72)
        return result

    def _current_body_pose(self, name: str) -> np.ndarray | None:
        pose = self._physics_body_pose(name)
        if pose is None:
            pose = self._poses.get(name)
        if pose is None:
            return None
        pose_np = np.asarray(pose, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(pose_np)):
            return None
        return pose_np.copy()

    def _current_body_matrix(self, name: str) -> np.ndarray | None:
        pose = self._current_body_pose(name)
        return None if pose is None else _pose_vector_to_matrix(pose)

    def _robot_link_pose_matrix(
        self,
        side: str,
        body_fragment: str = "palm_link",
    ) -> np.ndarray | None:
        robot = self._robots.get(side)
        if robot is None:
            return None
        cache_key = f"{side}:{body_fragment}"
        body_index = self._assist_robot_body_indices.get(cache_key)
        if body_index is None:
            try:
                body_index = next(
                    index
                    for index, name in enumerate(robot.body_names)
                    if body_fragment in str(name)
                )
            except (StopIteration, TypeError):
                return None
            self._assist_robot_body_indices[cache_key] = int(body_index)
        try:
            pose = robot.data.body_pose_w[0, int(body_index), :7]
            if hasattr(pose, "detach"):
                pose = pose.detach().cpu().numpy()
            pose_np = np.asarray(pose, dtype=np.float64).reshape(7)
        except Exception:
            return None
        if not np.all(np.isfinite(pose_np)):
            return None
        return _pose_vector_to_matrix(pose_np)

    @staticmethod
    def _rotation_error_vector(
        target_rotation: np.ndarray,
        current_rotation: np.ndarray,
    ) -> np.ndarray:
        error = (
            np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
            @ np.asarray(current_rotation, dtype=np.float64).reshape(3, 3).T
        )
        vector = 0.5 * np.asarray(
            [
                error[2, 1] - error[1, 2],
                error[0, 2] - error[2, 0],
                error[1, 0] - error[0, 1],
            ],
            dtype=np.float64,
        )
        sin_angle = float(np.linalg.norm(vector))
        cos_angle = float(np.clip((np.trace(error) - 1.0) * 0.5, -1.0, 1.0))
        angle = float(math.atan2(sin_angle, cos_angle))
        if sin_angle <= 1e-8 or angle <= 1e-8:
            return np.zeros(3, dtype=np.float64)
        return vector * (angle / sin_angle)

    @staticmethod
    def _pose_matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
        pose = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        return np.concatenate(
            [pose[:3, 3], _matrix_to_quat_wxyz(pose[:3, :3])]
        )

    @staticmethod
    def _wuji_tip_names(side: str) -> tuple[str, tuple[str, ...]]:
        return (
            f"{side}_finger1_tip",
            tuple(f"{side}_finger{index}_tip" for index in range(2, 6)),
        )

    @staticmethod
    def _point_box_surface_metrics(
        point: np.ndarray,
        center: np.ndarray,
        half_extents: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return signed distance and outward normal for an axis-aligned box."""
        local = np.asarray(point, dtype=np.float64).reshape(3) - np.asarray(
            center,
            dtype=np.float64,
        ).reshape(3)
        half = np.asarray(half_extents, dtype=np.float64).reshape(3)
        outside = np.maximum(np.abs(local) - half, 0.0)
        outside_distance = float(np.linalg.norm(outside))
        if outside_distance > 1e-12:
            closest = np.clip(local, -half, half)
            delta = local - closest
            return outside_distance, delta / max(float(np.linalg.norm(delta)), 1e-12)

        face_clearance = half - np.abs(local)
        axis = int(np.argmin(face_clearance))
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0 if float(local[axis]) >= 0.0 else -1.0
        return -float(face_clearance[axis]), normal

    @staticmethod
    def _segment_intersects_aabb(
        start: np.ndarray,
        end: np.ndarray,
        center: np.ndarray,
        half_extents: np.ndarray,
        *,
        margin: float = 0.0,
    ) -> bool:
        first = np.asarray(start, dtype=np.float64).reshape(3) - np.asarray(
            center,
            dtype=np.float64,
        ).reshape(3)
        second = np.asarray(end, dtype=np.float64).reshape(3) - np.asarray(
            center,
            dtype=np.float64,
        ).reshape(3)
        half = np.asarray(half_extents, dtype=np.float64).reshape(3) + max(
            float(margin),
            0.0,
        )
        direction = second - first
        lower = 0.0
        upper = 1.0
        for axis in range(3):
            if abs(float(direction[axis])) <= 1e-12:
                if abs(float(first[axis])) > float(half[axis]):
                    return False
                continue
            inverse = 1.0 / float(direction[axis])
            near = (-float(half[axis]) - float(first[axis])) * inverse
            far = (float(half[axis]) - float(first[axis])) * inverse
            if near > far:
                near, far = far, near
            lower = max(lower, near)
            upper = min(upper, far)
            if lower > upper:
                return False
        return True

    @classmethod
    def _visible_box_pinch_contact(
        cls,
        object_matrix: np.ndarray,
        tip_positions: dict[str, np.ndarray],
        *,
        side: str,
        center_local: np.ndarray,
        half_extents: np.ndarray,
        contact_margin: float,
        near_margin: float,
        pinch_distance_range: tuple[float, float],
    ) -> tuple[bool, float, int]:
        """Require opposing visible fingertips on the same authored box."""
        thumb_name, opposing_names = cls._wuji_tip_names(side)
        thumb_world = tip_positions.get(thumb_name)
        if thumb_world is None:
            return False, math.inf, 0
        inverse = _invert_pose_matrix(
            np.asarray(object_matrix, dtype=np.float64).reshape(4, 4)
        )

        def local_point(world: np.ndarray) -> np.ndarray:
            return (
                inverse
                @ np.asarray(
                    [*np.asarray(world, dtype=np.float64).reshape(3), 1.0],
                    dtype=np.float64,
                )
            )[:3]

        thumb_local = local_point(thumb_world)
        thumb_signed, thumb_normal = cls._point_box_surface_metrics(
            thumb_local,
            center_local,
            half_extents,
        )
        if (
            thumb_signed > float(contact_margin)
            or thumb_signed < -0.003
        ):
            return False, abs(float(thumb_signed)), 0

        valid: list[float] = []
        for name in opposing_names:
            world = tip_positions.get(name)
            if world is None:
                continue
            local = local_point(world)
            signed, normal = cls._point_box_surface_metrics(
                local,
                center_local,
                half_extents,
            )
            if signed > float(contact_margin) or signed < -0.003:
                continue
            span = float(np.linalg.norm(local - thumb_local))
            if not (
                float(pinch_distance_range[0])
                <= span
                <= float(pinch_distance_range[1])
            ):
                continue
            if float(np.dot(thumb_normal, normal)) > -0.20:
                continue
            if min(abs(float(thumb_signed)), abs(float(signed))) > float(
                near_margin
            ):
                continue
            if not cls._segment_intersects_aabb(
                thumb_local,
                local,
                center_local,
                half_extents,
                margin=0.002,
            ):
                continue
            valid.append(max(abs(float(thumb_signed)), abs(float(signed))))
        if not valid:
            return False, abs(float(thumb_signed)), 1
        return True, max(abs(float(thumb_signed)), min(valid)), 1 + len(valid)

    @classmethod
    def _visible_segment_pinch_contact(
        cls,
        object_matrix: np.ndarray,
        tip_positions: dict[str, np.ndarray],
        *,
        side: str,
        segments_local: tuple[np.ndarray, ...],
        segment_radii: tuple[float, ...],
        contact_margin: float,
        near_margin: float,
        pinch_distance_range: tuple[float, float],
        max_axial_separation: float | None = None,
    ) -> tuple[bool, float, int]:
        """Require a thumb/finger pinch across one visible rail or handle."""
        thumb_name, opposing_names = cls._wuji_tip_names(side)
        thumb_world = tip_positions.get(thumb_name)
        if thumb_world is None:
            return False, math.inf, 0
        inverse = _invert_pose_matrix(
            np.asarray(object_matrix, dtype=np.float64).reshape(4, 4)
        )

        def local_point(world: np.ndarray) -> np.ndarray:
            return (
                inverse
                @ np.asarray(
                    [*np.asarray(world, dtype=np.float64).reshape(3), 1.0],
                    dtype=np.float64,
                )
            )[:3]

        def projection(
            point: np.ndarray,
        ) -> tuple[float, np.ndarray, int, float]:
            best = (math.inf, np.zeros(3, dtype=np.float64), -1, 0.0)
            for index, segment_value in enumerate(segments_local):
                segment = np.asarray(segment_value, dtype=np.float64).reshape(2, 3)
                delta = segment[1] - segment[0]
                denominator = float(np.dot(delta, delta))
                fraction = (
                    0.0
                    if denominator <= 1e-12
                    else float(
                        np.clip(
                            np.dot(point - segment[0], delta) / denominator,
                            0.0,
                            1.0,
                        )
                    )
                )
                closest = segment[0] + fraction * delta
                center_distance = float(np.linalg.norm(point - closest))
                surface_distance = abs(
                    center_distance - float(segment_radii[index])
                )
                if surface_distance < best[0]:
                    best = (surface_distance, closest, int(index), fraction)
            return best

        thumb_local = local_point(thumb_world)
        (
            thumb_surface,
            thumb_axis_point,
            thumb_segment,
            thumb_fraction,
        ) = projection(thumb_local)
        if thumb_surface > float(contact_margin):
            return False, float(thumb_surface), 0
        thumb_radial = thumb_local - thumb_axis_point
        thumb_norm = float(np.linalg.norm(thumb_radial))
        valid: list[float] = []
        for name in opposing_names:
            world = tip_positions.get(name)
            if world is None:
                continue
            local = local_point(world)
            surface, axis_point, segment, fraction = projection(local)
            if segment != thumb_segment or surface > float(contact_margin):
                continue
            span = float(np.linalg.norm(local - thumb_local))
            if not (
                float(pinch_distance_range[0])
                <= span
                <= float(pinch_distance_range[1])
            ):
                continue
            if (
                max_axial_separation is not None
                and float(np.linalg.norm(axis_point - thumb_axis_point))
                > float(max_axial_separation)
            ):
                continue
            radial = local - axis_point
            radial_norm = float(np.linalg.norm(radial))
            if thumb_norm <= 1e-6 or radial_norm <= 1e-6:
                continue
            if float(np.dot(thumb_radial, radial) / (thumb_norm * radial_norm)) > -0.20:
                continue
            midpoint = 0.5 * (thumb_local + local)
            segment = np.asarray(
                segments_local[thumb_segment],
                dtype=np.float64,
            ).reshape(2, 3)
            if (
                cls._point_segment_distance_3d(
                    midpoint,
                    segment[0],
                    segment[1],
                )
                > float(segment_radii[thumb_segment]) + 0.004
            ):
                continue
            if min(float(thumb_surface), float(surface)) > float(near_margin):
                continue
            valid.append(max(float(thumb_surface), float(surface)))
        if not valid:
            return False, float(thumb_surface), 1
        return True, max(float(thumb_surface), min(valid)), 1 + len(valid)

    def _assist_hand_tracking_ready(
        self,
        side: str,
        state: dict[str, Any],
        *,
        position_tolerance: float,
        angle_tolerance: float,
    ) -> bool:
        target_value = state.get("hand_target_matrix")
        if target_value is None:
            state["hand_tracking_ready"] = True
            return True
        if not bool(state.get("hand_target_installed")):
            # The target is only copied into the robot command by
            # _postprocess_hand_action at the next control frame.  Advancing
            # the object during the intervening PhysX substeps would make it
            # visibly move before the hand has even received that command.
            state["hand_tracking_ready"] = False
            return False
        installed_value = state.get("hand_target_installed_matrix")
        if installed_value is not None:
            installed = np.asarray(
                installed_value,
                dtype=np.float64,
            ).reshape(4, 4)
            requested = np.asarray(
                target_value,
                dtype=np.float64,
            ).reshape(4, 4)
            if not np.allclose(
                installed,
                requested,
                rtol=0.0,
                atol=1e-10,
            ):
                # A PhysX substep has staged a newer guide target than the one
                # copied into the robot command at the control-frame boundary.
                # The object must wait for that exact target to be installed.
                state["hand_target_installed"] = False
                state["hand_tracking_ready"] = False
                return False
        palm = self._robot_link_pose_matrix(side)
        if palm is None:
            state["hand_tracking_ready"] = False
            return False
        target = np.asarray(
            installed_value
            if installed_value is not None
            else target_value,
            dtype=np.float64,
        ).reshape(4, 4)
        position_error = float(np.linalg.norm(palm[:3, 3] - target[:3, 3]))
        angle_error = float(
            np.linalg.norm(
                self._rotation_error_vector(
                    target[:3, :3],
                    palm[:3, :3],
                )
            )
        )
        ready = bool(
            position_error <= float(position_tolerance)
            and angle_error <= float(angle_tolerance)
        )
        state["hand_tracking_position_error"] = position_error
        state["hand_tracking_angle_error"] = angle_error
        state["hand_tracking_ready"] = ready
        return ready

    def _start_raw_hand_handoff(
        self,
        side: str,
        state: dict[str, Any],
        hand_target: np.ndarray,
    ) -> None:
        """Keep a guided hand continuous while returning motion to the glove."""
        target = np.asarray(
            hand_target,
            dtype=np.float64,
        ).reshape(4, 4)
        raw = _pose_vector_to_matrix(self._raw_tcp_pose(side))
        state["raw_handoff_raw_anchor"] = raw.copy()
        state["raw_handoff_hand_anchor"] = target.copy()
        # Keep the legacy matrix readable for old recordings/tests, but use
        # the decoupled anchors below so wrist rotation cannot swing a large
        # translational offset around an artificial lever arm.
        state["raw_to_hand_offset"] = _invert_pose_matrix(raw) @ target
        state["hand_target_matrix"] = target.copy()
        state["hand_target_installed"] = False
        state.pop("hand_target_installed_matrix", None)
        state.pop("hand_target_installed_frame", None)
        self._c_assist_free_hand_handoffs[side] = {
            "raw_anchor": raw.copy(),
            "hand_anchor": target.copy(),
        }

    @staticmethod
    def _map_raw_hand_handoff(
        raw: np.ndarray,
        raw_anchor: np.ndarray,
        hand_anchor: np.ndarray,
    ) -> np.ndarray:
        """Apply raw translation/rotation deltas without a rotating lever arm."""
        current = np.asarray(raw, dtype=np.float64).reshape(4, 4)
        source = np.asarray(raw_anchor, dtype=np.float64).reshape(4, 4)
        destination = np.asarray(
            hand_anchor,
            dtype=np.float64,
        ).reshape(4, 4)
        target = destination.copy()
        target[:3, 3] = (
            destination[:3, 3]
            + current[:3, 3]
            - source[:3, 3]
        )
        target[:3, :3] = (
            current[:3, :3]
            @ source[:3, :3].T
            @ destination[:3, :3]
        )
        return target

    def _preserve_released_hand_handoff(
        self,
        side: str,
        state: dict[str, Any],
        palm: np.ndarray | None,
    ) -> None:
        """Detach the object without returning the hand to an old raw frame."""
        del palm
        value = state.get("hand_target_installed_matrix")
        if value is None:
            commanded = (
                self._right_tcp_pose
                if side == "right"
                else self._left_tcp_pose
            )
            value = _pose_vector_to_matrix(commanded)
        hand = np.asarray(
            value,
            dtype=np.float64,
        ).reshape(4, 4)
        if hand is None:
            return
        raw = _pose_vector_to_matrix(self._raw_tcp_pose(side))
        self._c_assist_free_hand_handoffs[side] = {
            "raw_anchor": raw.copy(),
            "hand_anchor": hand.copy(),
        }

    def _raw_hand_handoff_target(
        self,
        side: str,
        state: dict[str, Any],
    ) -> np.ndarray | None:
        raw_anchor = state.get("raw_handoff_raw_anchor")
        hand_anchor = state.get("raw_handoff_hand_anchor")
        if raw_anchor is None or hand_anchor is None:
            offset = state.get("raw_to_hand_offset")
            if offset is None:
                return None
            return (
                _pose_vector_to_matrix(self._raw_tcp_pose(side))
                @ np.asarray(offset, dtype=np.float64).reshape(4, 4)
            )
        return self._map_raw_hand_handoff(
            _pose_vector_to_matrix(self._raw_tcp_pose(side)),
            np.asarray(raw_anchor, dtype=np.float64).reshape(4, 4),
            np.asarray(hand_anchor, dtype=np.float64).reshape(4, 4),
        )

    @classmethod
    def _bounded_pose_step(
        cls,
        current: np.ndarray,
        goal: np.ndarray,
        *,
        dt: float,
        max_linear_speed: float,
        max_angular_speed: float,
    ) -> np.ndarray:
        start = np.asarray(current, dtype=np.float64).reshape(4, 4)
        end = np.asarray(goal, dtype=np.float64).reshape(4, 4)
        distance = float(np.linalg.norm(end[:3, 3] - start[:3, 3]))
        angle = float(
            np.linalg.norm(
                cls._rotation_error_vector(
                    end[:3, :3],
                    start[:3, :3],
                )
            )
        )
        alpha = 1.0
        step_dt = max(float(dt), 0.0)
        if distance > 1e-9:
            alpha = min(
                alpha,
                float(max_linear_speed) * step_dt / distance,
            )
        if angle > 1e-9:
            alpha = min(
                alpha,
                float(max_angular_speed) * step_dt / angle,
            )
        return cls._c07_interpolate_pose(start, end, alpha)

    def _tool_lock_target_reached(
        self,
        carried_target: np.ndarray,
        lock_target: np.ndarray,
    ) -> bool:
        carried = np.asarray(
            carried_target,
            dtype=np.float64,
        ).reshape(4, 4)
        locked = np.asarray(
            lock_target,
            dtype=np.float64,
        ).reshape(4, 4)
        position_error = float(
            np.linalg.norm(carried[:3, 3] - locked[:3, 3])
        )
        angle_error = float(
            np.linalg.norm(
                self._rotation_error_vector(
                    locked[:3, :3],
                    carried[:3, :3],
                )
            )
        )
        return bool(
            position_error <= _C_TOOL_LOCK_POSITION_TOLERANCE
            and angle_error <= _C_TOOL_LOCK_ANGLE_TOLERANCE
        )

    def _set_body_soft_target(
        self,
        name: str,
        target_matrix: np.ndarray,
        *,
        max_linear_speed: float = _C_TOOL_ASSIST_MAX_LINEAR_SPEED,
        max_angular_speed: float = _C_TOOL_ASSIST_MAX_ANGULAR_SPEED,
    ) -> tuple[bool, float, float]:
        """Velocity-follow a nearby visual affordance without teleporting it."""
        view = self._task_body_views.get(name)
        current_pose = self._physics_body_pose(name)
        if view is None or current_pose is None or self._torch is None:
            return False, math.inf, math.inf
        current_matrix = _pose_vector_to_matrix(current_pose)
        target = np.asarray(target_matrix, dtype=np.float64).reshape(4, 4)
        position_error = target[:3, 3] - current_matrix[:3, 3]
        rotation_error = self._rotation_error_vector(
            target[:3, :3],
            current_matrix[:3, :3],
        )
        linear_error = float(np.linalg.norm(position_error))
        angular_error = float(np.linalg.norm(rotation_error))
        try:
            velocities = view.get_velocities().clone()
            linear_velocity = position_error / max(float(self._sim_dt), 1e-6)
            linear_speed = float(np.linalg.norm(linear_velocity))
            if linear_speed > max_linear_speed:
                linear_velocity *= max_linear_speed / linear_speed
            angular_velocity = rotation_error / max(float(self._sim_dt), 1e-6)
            angular_speed = float(np.linalg.norm(angular_velocity))
            if angular_speed > max_angular_speed:
                angular_velocity *= max_angular_speed / angular_speed
            velocities[0, :3] = self._torch.as_tensor(
                linear_velocity,
                dtype=velocities.dtype,
                device=velocities.device,
            )
            velocities[0, 3:6] = self._torch.as_tensor(
                angular_velocity,
                dtype=velocities.dtype,
                device=velocities.device,
            )
            indices = self._torch.arange(
                int(view.count),
                dtype=self._torch.long,
                device=velocities.device,
            )
            view.set_velocities(velocities, indices)
            return True, linear_error, angular_error
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not apply {name} visual-interaction assist: {exc}"
            )
            return False, linear_error, angular_error

    def _set_body_rigid_grasp_target(
        self,
        name: str,
        target_matrix: np.ndarray,
        *,
        previous_target: np.ndarray | None = None,
    ) -> tuple[bool, float, float]:
        """Set an attached object's exact root pose for this physics substep.

        The attachment stores the hand/object transform at the user's actual
        contact point.  Writing the articulation root directly preserves that
        transform without a velocity-following lag, while the target delta is
        retained as root velocity for physically sensible carried contents.
        """
        target = np.asarray(target_matrix, dtype=np.float64).reshape(4, 4)
        current_pose = self._physics_body_pose(name)
        if current_pose is None:
            current_pose = self._poses.get(name)
        if current_pose is None:
            linear_error = math.inf
            angular_error = math.inf
        else:
            current = _pose_vector_to_matrix(
                np.asarray(current_pose, dtype=np.float64).reshape(7)
            )
            linear_error = float(
                np.linalg.norm(target[:3, 3] - current[:3, 3])
            )
            angular_error = float(
                np.linalg.norm(
                    self._rotation_error_vector(
                        target[:3, :3],
                        current[:3, :3],
                    )
                )
            )

        if self._torch is None:
            return False, linear_error, angular_error

        dt = max(float(self._sim_dt), 1e-6)
        root_velocity = np.zeros(6, dtype=np.float64)
        if previous_target is not None:
            previous = np.asarray(
                previous_target,
                dtype=np.float64,
            ).reshape(4, 4)
            root_velocity[:3] = (
                target[:3, 3] - previous[:3, 3]
            ) / dt
            root_velocity[3:] = self._rotation_error_vector(
                target[:3, :3],
                previous[:3, :3],
            ) / dt

        pose = np.concatenate(
            [
                target[:3, 3],
                _matrix_to_quat_wxyz(target[:3, :3]),
            ]
        )
        articulation = self._task_articulation_views.get(name)
        rigid_view = self._task_body_views.get(name)
        if articulation is None and rigid_view is None:
            return False, linear_error, angular_error

        try:
            if articulation is not None:
                view, indices = articulation
                transforms = view.get_root_transforms().clone()
                velocities = view.get_root_velocities().clone()
                original_transforms = transforms.clone()
                original_velocities = velocities.clone()
                transforms[:, :3] = self._torch.as_tensor(
                    pose[:3],
                    dtype=transforms.dtype,
                    device=transforms.device,
                )
                transforms[:, 3:7] = self._torch.as_tensor(
                    [pose[4], pose[5], pose[6], pose[3]],
                    dtype=transforms.dtype,
                    device=transforms.device,
                )
                velocities[:, :6] = self._torch.as_tensor(
                    root_velocity,
                    dtype=velocities.dtype,
                    device=velocities.device,
                )
                try:
                    # Velocity first: a failed velocity writer can no longer
                    # leave the object teleported without a matching state.
                    view.set_root_velocities(velocities, indices)
                    view.set_root_transforms(transforms, indices)
                except Exception:
                    try:
                        view.set_root_velocities(
                            original_velocities,
                            indices,
                        )
                        view.set_root_transforms(
                            original_transforms,
                            indices,
                        )
                    except Exception:
                        pass
                    raise
            else:
                view = rigid_view
                transforms = view.get_transforms().clone()
                velocities = view.get_velocities().clone()
                original_transforms = transforms.clone()
                original_velocities = velocities.clone()
                transforms[:, :3] = self._torch.as_tensor(
                    pose[:3],
                    dtype=transforms.dtype,
                    device=transforms.device,
                )
                transforms[:, 3:7] = self._torch.as_tensor(
                    [pose[4], pose[5], pose[6], pose[3]],
                    dtype=transforms.dtype,
                    device=transforms.device,
                )
                velocities[:, :6] = self._torch.as_tensor(
                    root_velocity,
                    dtype=velocities.dtype,
                    device=velocities.device,
                )
                indices = self._torch.arange(
                    int(view.count),
                    dtype=self._torch.long,
                    device=transforms.device,
                )
                try:
                    view.set_velocities(velocities, indices)
                    view.set_transforms(transforms, indices)
                except Exception:
                    try:
                        view.set_velocities(original_velocities, indices)
                        view.set_transforms(original_transforms, indices)
                    except Exception:
                        pass
                    raise
            self._poses[name] = pose.copy()
            return True, linear_error, angular_error
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not rigidly attach {name} at the contacted pose: {exc}"
            )
            return False, linear_error, angular_error

    def _c07_finger_grip_center(
        self,
        palm: np.ndarray,
    ) -> np.ndarray:
        """Return the visible finger cluster center used to meet the handle."""
        tip_positions = self._b02_wuji_fingertip_positions()
        if len(tip_positions) >= 3:
            points = np.stack(tuple(tip_positions.values()), axis=0)
            center = np.mean(points, axis=0)
            if np.all(np.isfinite(center)):
                return center.astype(np.float64)
        palm_matrix = np.asarray(palm, dtype=np.float64).reshape(4, 4)
        return (
            palm_matrix
            @ np.asarray(
                [*_C07_HANDLE_GRASP_PALM_LOCAL, 1.0],
                dtype=np.float64,
            )
        )[:3]

    @classmethod
    def _c07_handle_sensor_contains(
        cls,
        pitcher: np.ndarray,
        palm: np.ndarray,
        grip_center_world: np.ndarray | None = None,
    ) -> tuple[bool, np.ndarray]:
        pitcher_matrix = np.asarray(pitcher, dtype=np.float64).reshape(4, 4)
        palm_matrix = np.asarray(palm, dtype=np.float64).reshape(4, 4)
        finger_probe_world = (
            (
                palm_matrix
                @ np.asarray(
                    [*_C07_HANDLE_GRASP_PALM_LOCAL, 1.0],
                    dtype=np.float64,
                )
            )[:3]
            if grip_center_world is None
            else np.asarray(grip_center_world, dtype=np.float64).reshape(3)
        )
        finger_probe_local = (
            _invert_pose_matrix(pitcher_matrix)
            @ np.asarray([*finger_probe_world, 1.0], dtype=np.float64)
        )[:3]
        frame_distance = min(
            cls._point_segment_distance_3d(
                finger_probe_local,
                segment[0],
                segment[1],
            )
            for segment in _C07_HANDLE_FRAME_SEGMENTS
        )
        return bool(
            frame_distance
            <= _C07_HANDLE_FRAME_RADIUS + _C07_HANDLE_CONTACT_MARGIN
        ), finger_probe_local

    @classmethod
    def _c07_handle_grasp_contact(
        cls,
        pitcher: np.ndarray,
        tip_positions: dict[str, np.ndarray],
    ) -> tuple[bool, float, int]:
        """Require an actual thumb/finger pinch on the original YCB handle."""
        thumb_world = tip_positions.get(_B02_WUJI_THUMB_TIP)
        if thumb_world is None:
            return False, math.inf, 0
        pitcher_inverse = _invert_pose_matrix(
            np.asarray(pitcher, dtype=np.float64).reshape(4, 4)
        )

        def local_point(world: np.ndarray) -> np.ndarray:
            return (
                pitcher_inverse
                @ np.asarray(
                    [*np.asarray(world, dtype=np.float64).reshape(3), 1.0],
                    dtype=np.float64,
                )
            )[:3]

        def frame_projection(
            local: np.ndarray,
        ) -> tuple[float, np.ndarray, int]:
            best_distance = math.inf
            best_point = np.zeros(3, dtype=np.float64)
            best_index = -1
            for index, segment in enumerate(_C07_HANDLE_FRAME_SEGMENTS):
                start = segment[0]
                delta = segment[1] - start
                denominator = float(np.dot(delta, delta))
                fraction = (
                    0.0
                    if denominator <= 1e-12
                    else float(
                        np.clip(
                            np.dot(local - start, delta) / denominator,
                            0.0,
                            1.0,
                        )
                    )
                )
                closest = start + fraction * delta
                distance = float(np.linalg.norm(local - closest))
                if distance < best_distance:
                    best_distance = distance
                    best_point = closest
                    best_index = int(index)
            return best_distance, best_point, best_index

        def frame_distance(local: np.ndarray) -> float:
            return frame_projection(local)[0]

        loose_contact_limit = (
            _C07_HANDLE_FRAME_RADIUS + _C07_HANDLE_CONTACT_MARGIN
        )
        near_contact_limit = (
            _C07_HANDLE_FRAME_RADIUS + _C07_HANDLE_NEAR_CONTACT_MARGIN
        )
        thumb_local = local_point(thumb_world)
        (
            thumb_distance,
            thumb_frame_point,
            thumb_segment_index,
        ) = frame_projection(thumb_local)
        if thumb_distance > loose_contact_limit:
            return False, thumb_distance, 0
        thumb_radial = thumb_local - thumb_frame_point
        thumb_radial_norm = float(np.linalg.norm(thumb_radial))

        valid_opponents: list[tuple[float, np.ndarray]] = []
        for name in _B02_WUJI_OPPOSING_TIPS:
            world = tip_positions.get(name)
            if world is None:
                continue
            world_np = np.asarray(world, dtype=np.float64).reshape(3)
            if (
                float(
                    np.linalg.norm(
                        world_np
                        - np.asarray(thumb_world, dtype=np.float64).reshape(3)
                    )
                )
                > _C07_HANDLE_PINCH_MAX_DISTANCE
            ):
                continue
            local = local_point(world_np)
            (
                distance,
                opponent_frame_point,
                opponent_segment_index,
            ) = frame_projection(local)
            if distance > loose_contact_limit:
                continue
            # One fingertip must be genuinely close to the visible handle.
            # The other gets a little tracker/collision tolerance, which makes
            # pickup forgiving without restoring the old empty-space sensor.
            if min(thumb_distance, distance) > near_contact_limit:
                continue
            pinch_span = float(np.linalg.norm(thumb_local - local))
            if pinch_span < _C07_HANDLE_MIN_PINCH_SPAN:
                continue
            opponent_radial = local - opponent_frame_point
            opponent_radial_norm = float(np.linalg.norm(opponent_radial))
            if thumb_radial_norm <= 1e-6 or opponent_radial_norm <= 1e-6:
                continue
            radial_dot = float(
                np.dot(thumb_radial, opponent_radial)
                / (thumb_radial_norm * opponent_radial_norm)
            )
            if radial_dot > _C07_HANDLE_MAX_RADIAL_DOT:
                continue
            pinch_center = 0.5 * (thumb_local + local)
            if (
                frame_distance(pinch_center)
                > _C07_HANDLE_GRIP_CENTER_MAX_DISTANCE
            ):
                continue
            if (
                abs(thumb_segment_index - opponent_segment_index) > 1
                or float(
                    np.linalg.norm(
                        thumb_frame_point - opponent_frame_point
                    )
                )
                > _C07_HANDLE_PROJECTION_MAX_SEPARATION
            ):
                continue
            frame_between_fingers = 0.5 * (
                thumb_frame_point + opponent_frame_point
            )
            if (
                cls._point_segment_distance_3d(
                    frame_between_fingers,
                    thumb_local,
                    local,
                )
                > _C07_HANDLE_PINCH_LINE_MAX_DISTANCE
            ):
                continue
            valid_opponents.append((distance, local))

        if not valid_opponents:
            return False, thumb_distance, 1
        best_opponent_distance = min(item[0] for item in valid_opponents)
        return (
            True,
            max(float(thumb_distance), float(best_opponent_distance)),
            1 + len(valid_opponents),
        )

    @staticmethod
    def _c07_spout_outlet_world(pitcher: np.ndarray) -> np.ndarray:
        pitcher_matrix = np.asarray(pitcher, dtype=np.float64).reshape(4, 4)
        return (
            pitcher_matrix
            @ np.asarray([*_C07_SPOUT_OUTLET_LOCAL, 1.0], dtype=np.float64)
        )[:3]

    @classmethod
    def _c07_pitcher_over_cup(
        cls,
        pitcher: np.ndarray,
        cup: np.ndarray,
    ) -> bool:
        outlet = cls._c07_spout_outlet_world(pitcher)
        cup_matrix = np.asarray(cup, dtype=np.float64).reshape(4, 4)
        xy_distance = float(np.linalg.norm(outlet[:2] - cup_matrix[:2, 3]))
        height = float(outlet[2] - cup_matrix[2, 3])
        return bool(
            xy_distance <= _C07_POUR_XY_CAPTURE_RADIUS
            and _C07_POUR_MIN_HEIGHT_ABOVE_CUP
            <= height
            <= _C07_POUR_MAX_HEIGHT_ABOVE_CUP
        )

    @classmethod
    def _c07_lock_pitcher_target_xy(
        cls,
        target: np.ndarray,
        cup_xy: np.ndarray,
    ) -> np.ndarray:
        locked = np.asarray(target, dtype=np.float64).reshape(4, 4).copy()
        outlet = cls._c07_spout_outlet_world(locked)
        locked[:2, 3] += (
            np.asarray(cup_xy, dtype=np.float64).reshape(2)
            - outlet[:2]
        )
        return locked

    def _start_c07_tilt_path(
        self,
        state: dict[str, Any],
        pitcher: np.ndarray,
        cup: np.ndarray,
    ) -> None:
        """Capture a smooth upright/centred alignment and fixed-spout arc."""
        pitcher_matrix = np.asarray(
            pitcher,
            dtype=np.float64,
        ).reshape(4, 4)
        cup_matrix = np.asarray(cup, dtype=np.float64).reshape(4, 4)
        object_anchor = self._c07_lock_pitcher_target_xy(
            pitcher_matrix,
            cup_matrix[:2, 3],
        )
        # Normalize to world-up so the same glove Z always means the same pour
        # angle.  Preserve table yaw, then place the visible outlet at a fixed
        # safe point over the receiving cup.  Alignment to this pose is
        # interpolated as a hand/object pair; it is never applied in the
        # detection substep where the hand target has not yet been installed.
        x_axis_xy = object_anchor[:2, 0]
        if float(np.linalg.norm(x_axis_xy)) > 1e-8:
            yaw = math.atan2(
                float(x_axis_xy[1]),
                float(x_axis_xy[0]),
            )
        else:
            y_axis_xy = object_anchor[:2, 1]
            yaw = math.atan2(
                -float(y_axis_xy[0]),
                float(y_axis_xy[1]),
            )
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        upright_rotation = np.asarray(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        spout_anchor = np.asarray(
            [
                cup_matrix[0, 3],
                cup_matrix[1, 3],
                cup_matrix[2, 3]
                + _C07_POUR_TARGET_SPOUT_HEIGHT_ABOVE_CUP,
            ],
            dtype=np.float64,
        )
        object_anchor[:3, :3] = upright_rotation
        object_anchor[:3, 3] = (
            spout_anchor
            - upright_rotation @ _C07_SPOUT_OUTLET_LOCAL
        )
        input_anchor = _pose_vector_to_matrix(
            np.asarray(
                self._c_assist_raw_right_tcp_pose,
                dtype=np.float64,
            ).reshape(7)
        )
        hand_to_object = np.asarray(
            state["hand_to_object"],
            dtype=np.float64,
        ).reshape(4, 4)
        hand_anchor = object_anchor @ _invert_pose_matrix(hand_to_object)
        alignment_translation = float(
            np.linalg.norm(
                object_anchor[:3, 3] - pitcher_matrix[:3, 3]
            )
        )
        alignment_rotation = float(
            np.linalg.norm(
                self._rotation_error_vector(
                    object_anchor[:3, :3],
                    pitcher_matrix[:3, :3],
                )
            )
        )
        alignment_duration = max(
            _C07_POUR_ALIGN_DURATION_SECONDS,
            alignment_translation
            / max(_C07_POUR_ALIGN_TRANSLATION_RATE_MPS, 1e-9),
            math.degrees(alignment_rotation)
            / max(_C07_POUR_ALIGN_ROTATION_RATE_DEG_PER_S, 1e-9),
        )
        state.update(
            {
                "pour_input_anchor": input_anchor,
                "pour_object_anchor": object_anchor,
                "pour_pivot_world": self._c07_spout_outlet_world(
                    object_anchor
                ),
                "pour_alignment_from_object": pitcher_matrix.copy(),
                "pour_alignment_elapsed_s": 0.0,
                "pour_alignment_fraction": 0.0,
                "pour_alignment_candidate_fraction": 0.0,
                "pour_alignment_duration_s": float(alignment_duration),
                "pour_target_ready": False,
                "pour_object_target": pitcher_matrix.copy(),
                "pour_hand_anchor": hand_anchor,
                "pour_hand_target": (
                    pitcher_matrix @ _invert_pose_matrix(hand_to_object)
                ),
                "pour_requested_tilt_deg": 0.0,
                "pour_candidate_tilt_deg": 0.0,
                "pour_applied_tilt_deg": 0.0,
            }
        )

    @staticmethod
    def _c07_interpolate_pose(
        start: np.ndarray,
        end: np.ndarray,
        fraction: float,
    ) -> np.ndarray:
        alpha = float(np.clip(fraction, 0.0, 1.0))
        start_matrix = np.asarray(start, dtype=np.float64).reshape(4, 4)
        end_matrix = np.asarray(end, dtype=np.float64).reshape(4, 4)
        target = np.eye(4, dtype=np.float64)
        target[:3, 3] = (
            (1.0 - alpha) * start_matrix[:3, 3]
            + alpha * end_matrix[:3, 3]
        )
        start_quat = _matrix_to_quat_wxyz(start_matrix[:3, :3])
        end_quat = _matrix_to_quat_wxyz(end_matrix[:3, :3])
        target[:3, :3] = _quat_wxyz_to_matrix(
            _blend_quat_wxyz(start_quat, end_quat, alpha)
        )
        return target

    @classmethod
    def _c07_tilt_path_targets(
        cls,
        raw_hand_pose: np.ndarray,
        state: dict[str, Any],
        *,
        tilt_degrees: float | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Map upward glove travel to the fixed-spout pour arc."""
        required = (
            "pour_input_anchor",
            "pour_object_anchor",
            "pour_pivot_world",
            "hand_to_object",
        )
        if any(state.get(key) is None for key in required):
            return None, None
        raw_hand = _pose_vector_to_matrix(
            np.asarray(raw_hand_pose, dtype=np.float64).reshape(7)
        )
        input_anchor = np.asarray(
            state["pour_input_anchor"],
            dtype=np.float64,
        ).reshape(4, 4)
        object_anchor = np.asarray(
            state["pour_object_anchor"],
            dtype=np.float64,
        ).reshape(4, 4)
        pivot = np.asarray(
            state["pour_pivot_world"],
            dtype=np.float64,
        ).reshape(3)
        hand_to_object = np.asarray(
            state["hand_to_object"],
            dtype=np.float64,
        ).reshape(4, 4)

        def object_target_for_tilt(tilt: float) -> np.ndarray:
            cosine = math.cos(-tilt)
            sine = math.sin(-tilt)
            local_pour_rotation = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, cosine, -sine],
                    [0.0, sine, cosine],
                ],
                dtype=np.float64,
            )
            target = object_anchor.copy()
            target[:3, :3] = (
                object_anchor[:3, :3] @ local_pour_rotation
            )
            target[:3, 3] = (
                pivot
                - target[:3, :3] @ _C07_SPOUT_OUTLET_LOCAL
            )
            return target

        # During the lock, XY and wrist rotation are deliberately ignored.
        # Zero to six centimetres of world-Z input requests zero to 150
        # degrees.  At the endpoint the opening normal is sixty degrees below
        # horizontal; extra upward travel moves neither hand nor pitcher.
        max_tilt = math.radians(_C07_POUR_MAX_TOTAL_TILT_DEG)
        if tilt_degrees is None:
            input_lift = max(
                0.0,
                float(raw_hand[2, 3] - input_anchor[2, 3]),
            )
            input_fraction = float(
                np.clip(
                    input_lift / max(_C07_POUR_FULL_TILT_INPUT_M, 1e-9),
                    0.0,
                    1.0,
                )
            )
            tilt = input_fraction * max_tilt
        else:
            tilt = math.radians(
                float(
                    np.clip(
                        tilt_degrees,
                        0.0,
                        _C07_POUR_MAX_TOTAL_TILT_DEG,
                    )
                )
            )

        object_target = object_target_for_tilt(tilt)
        hand_target = object_target @ _invert_pose_matrix(hand_to_object)
        hand_pose = np.concatenate(
            [
                hand_target[:3, 3],
                _matrix_to_quat_wxyz(hand_target[:3, :3]),
            ]
        )
        return hand_pose, object_target

    @staticmethod
    def _c07_post_pour_hand_target(
        raw_hand_pose: np.ndarray,
        state: dict[str, Any],
    ) -> np.ndarray | None:
        """Rebase free carrying at the hand pose where pouring unlocked."""
        input_anchor_value = state.get("carry_input_anchor")
        hand_anchor_value = state.get("carry_hand_anchor")
        if input_anchor_value is None or hand_anchor_value is None:
            return None
        raw_hand = _pose_vector_to_matrix(
            np.asarray(raw_hand_pose, dtype=np.float64).reshape(7)
        )
        input_anchor = np.asarray(
            input_anchor_value,
            dtype=np.float64,
        ).reshape(4, 4)
        hand_anchor = np.asarray(
            hand_anchor_value,
            dtype=np.float64,
        ).reshape(4, 4)
        target = hand_anchor.copy()
        target[:3, 3] += raw_hand[:3, 3] - input_anchor[:3, 3]
        world_rotation_delta = (
            raw_hand[:3, :3] @ input_anchor[:3, :3].T
        )
        target[:3, :3] = (
            world_rotation_delta @ hand_anchor[:3, :3]
        )
        return np.concatenate(
            [
                target[:3, 3],
                _matrix_to_quat_wxyz(target[:3, :3]),
            ]
        )

    @staticmethod
    def _c07_particle_capture_candidate(
        cup_local: np.ndarray,
        previous_cup_local: np.ndarray | None,
    ) -> bool:
        local = np.asarray(cup_local, dtype=np.float64).reshape(3)
        radial = float(np.linalg.norm(local[:2]))
        within_capture_volume = bool(
            radial <= _C07_PARTICLE_CAPTURE_RADIUS
            and _C07_PARTICLE_CAPTURE_Z_RANGE[0]
            <= float(local[2])
            <= _C07_PARTICLE_CAPTURE_Z_RANGE[1]
        )
        if within_capture_volume:
            return True
        if previous_cup_local is None:
            return False
        previous = np.asarray(previous_cup_local, dtype=np.float64).reshape(3)
        rim_z = _C07_CUP_LOCAL_Z_RANGE[1]
        crossed_rim_downward = bool(
            float(previous[2]) > rim_z
            and float(local[2]) <= rim_z
            and radial <= _C07_PARTICLE_CAPTURE_RADIUS
        )
        return crossed_rim_downward

    @staticmethod
    def _c07_particle_settle_local(index: int) -> np.ndarray:
        slots = np.asarray(
            [
                (0.000, 0.000, -0.022),
                (0.010, 0.000, -0.010),
                (-0.010, 0.000, -0.010),
                (0.000, 0.010, -0.010),
                (0.000, -0.010, -0.010),
                (0.008, 0.008, 0.004),
                (-0.008, 0.008, 0.004),
                (0.000, -0.008, 0.004),
            ],
            dtype=np.float64,
        )
        return slots[int(index) % len(slots)].copy()

    def _c07_capture_particle_pins(
        self,
        pitcher: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Capture each ball exactly where it is inside the carried pitcher."""
        pitcher_matrix = np.asarray(
            pitcher,
            dtype=np.float64,
        ).reshape(4, 4)
        pitcher_inverse = _invert_pose_matrix(pitcher_matrix)
        local_transforms: dict[str, np.ndarray] = {}
        last_targets: dict[str, np.ndarray] = {}
        for name in _C07_PARTICLE_NAMES:
            if name in self._c07_delivered_particles:
                continue
            pose = self._current_body_pose(name)
            if pose is None:
                continue
            local_position = (
                pitcher_inverse
                @ np.asarray([*pose[:3], 1.0], dtype=np.float64)
            )[:3]
            inside_capture_shell = bool(
                float(np.linalg.norm(local_position[:2]))
                <= _C07_PITCHER_BODY_RADIUS + _C07_PARTICLE_RADIUS
                and _C07_PITCHER_BOTTOM_Z - _C07_PARTICLE_RADIUS
                <= float(local_position[2])
                <= _C07_PITCHER_TOP_Z + _C07_PARTICLE_RADIUS
            )
            if (
                not self._c07_particle_is_in_pitcher(
                    pose[:3],
                    pitcher_inverse,
                )
                and not inside_capture_shell
            ):
                continue
            particle = _pose_vector_to_matrix(pose)
            local_transforms[name] = pitcher_inverse @ particle
            last_targets[name] = particle.copy()
        return local_transforms, last_targets

    @staticmethod
    def _c07_particle_release_tilt_degrees(
        state: dict[str, Any],
        pitcher_target: np.ndarray,
    ) -> float:
        del state
        target_matrix = np.asarray(
            pitcher_target,
            dtype=np.float64,
        ).reshape(4, 4)
        # atan2 preserves the full 0..180-degree arc.  asin would fold a
        # 150-degree pour back to 30 degrees and disable the drain again.
        total_tilt = math.degrees(
            math.atan2(
                float(-target_matrix[2, 1]),
                float(target_matrix[2, 2]),
            )
        )
        return float(np.clip(total_tilt, 0.0, 180.0))

    def _update_c07_particle_pins(
        self,
        state: dict[str, Any],
        pitcher_target: np.ndarray,
        *,
        advance_release: bool = True,
    ) -> None:
        """Move one pinned ball continuously through the real mouth at a time."""
        local_transforms = state.get("particle_local_transforms")
        if not isinstance(local_transforms, dict) or not local_transforms:
            return
        last_targets = state.setdefault("particle_last_targets", {})
        target_matrix = np.asarray(
            pitcher_target,
            dtype=np.float64,
        ).reshape(4, 4)
        tilt_degrees = self._c07_particle_release_tilt_degrees(
            state,
            target_matrix,
        )
        state["particle_release_tilt_deg"] = float(tilt_degrees)
        drain_armed = bool(
            bool(state.get("pour_xy_lock"))
            and not bool(state.get("pour_lock_pending"))
            and tilt_degrees >= _C07_PARTICLE_RELEASE_TOTAL_TILT_DEG
        )
        state["particle_drain_armed"] = drain_armed

        stability_reference = state.get("particle_stability_target")
        if not drain_armed:
            state["particle_target_stable_steps"] = 0
        elif stability_reference is None:
            state["particle_target_stable_steps"] = 0
        else:
            previous_pitcher = np.asarray(
                stability_reference,
                dtype=np.float64,
            ).reshape(4, 4)
            position_delta = float(
                np.linalg.norm(
                    target_matrix[:3, 3] - previous_pitcher[:3, 3]
                )
            )
            angle_delta = float(
                np.linalg.norm(
                    self._rotation_error_vector(
                        target_matrix[:3, :3],
                        previous_pitcher[:3, :3],
                    )
                )
            )
            if (
                position_delta
                <= _C07_PARTICLE_STABLE_POSITION_TOLERANCE
                and angle_delta
                <= _C07_PARTICLE_STABLE_ANGLE_TOLERANCE
            ):
                state["particle_target_stable_steps"] = (
                    int(state.get("particle_target_stable_steps", 0)) + 1
                )
            else:
                state["particle_target_stable_steps"] = 0
        state["particle_stability_target"] = target_matrix.copy()

        if (
            drain_armed
            and int(state.get("particle_target_stable_steps", 0))
            >= _C07_PARTICLE_RELEASE_STABLE_STEPS
        ):
            state["particle_drain_started"] = True
        if (
            advance_release
            and drain_armed
            and bool(state.get("particle_drain_started"))
        ):
            state["particle_drain_elapsed_s"] = float(
                state.get("particle_drain_elapsed_s", 0.0)
            ) + max(float(self._sim_dt), 0.0)

        released_value = state.get("particle_released_names", set())
        released_names = (
            released_value
            if isinstance(released_value, set)
            else set(released_value)
        )
        state["particle_released_names"] = released_names

        cooldown = max(
            0.0,
            float(state.get("particle_stage_release_cooldown_s", 0.0)),
        )
        if advance_release and drain_armed and cooldown > 0.0:
            cooldown = max(0.0, cooldown - max(float(self._sim_dt), 0.0))
            state["particle_stage_release_cooldown_s"] = cooldown

        stage_order_value = state.get("particle_stage_order")
        if not isinstance(stage_order_value, (tuple, list)):
            stage_order_value = sorted(
                local_transforms,
                key=lambda name: (
                    -float(
                        np.asarray(
                            local_transforms[name],
                            dtype=np.float64,
                        ).reshape(4, 4)[2, 3]
                    ),
                    name,
                ),
            )
            state["particle_stage_order"] = list(stage_order_value)

        active = state.get("particle_stage_active")
        if not isinstance(active, dict):
            active = None

        previous_release_clear = True
        previous_name = str(
            state.get("particle_stage_last_release_name", "")
        )
        previous_outside = state.get(
            "particle_stage_last_outside_world"
        )
        if (
            previous_name
            and previous_name not in self._c07_delivered_particles
            and previous_outside is not None
        ):
            previous_pose = self._current_body_pose(previous_name)
            if previous_pose is not None:
                previous_release_clear = bool(
                    float(
                        np.linalg.norm(
                            np.asarray(
                                previous_pose[:3],
                                dtype=np.float64,
                            )
                            - np.asarray(
                                previous_outside,
                                dtype=np.float64,
                            ).reshape(3)
                        )
                    )
                    >= 0.025
                )

        if (
            active is None
            and advance_release
            and drain_armed
            and bool(state.get("particle_drain_started"))
            and float(state.get("particle_drain_elapsed_s", 0.0))
            >= _C07_PARTICLE_FIRST_RELEASE_DELAY_SECONDS
            and cooldown <= 0.0
            and previous_release_clear
        ):
            next_name = next(
                (
                    str(name)
                    for name in stage_order_value
                    if name in local_transforms
                    and name not in released_names
                    and name not in self._c07_delivered_particles
                ),
                "",
            )
            if next_name:
                start_local = np.asarray(
                    local_transforms[next_name],
                    dtype=np.float64,
                ).reshape(4, 4)[:3, 3].copy()
                active = {
                    "name": next_name,
                    "current_local": start_local,
                    "waypoint_index": 0,
                    "outside_hold_successes": 0,
                }
                state["particle_stage_active"] = active

        active_name = "" if active is None else str(active.get("name", ""))
        stage_can_advance = bool(advance_release and drain_armed)

        def stage_waypoints(
            name: str,
            start_local: np.ndarray,
        ) -> tuple[np.ndarray, ...]:
            try:
                index = _C07_PARTICLE_NAMES.index(name)
            except ValueError:
                index = 0
            lane_x = float(
                _C07_PARTICLE_EXIT_X_OFFSETS[
                    index % len(_C07_PARTICLE_EXIT_X_OFFSETS)
                ]
            )
            clear_z = max(
                float(start_local[2]),
                _C07_PARTICLE_STAGE_CLEAR_Z_LOCAL,
            )
            return (
                np.asarray(
                    [start_local[0], start_local[1], clear_z],
                    dtype=np.float64,
                ),
                np.asarray([lane_x, 0.0, clear_z], dtype=np.float64),
                np.asarray([lane_x, 0.040, clear_z], dtype=np.float64),
                np.asarray(
                    [
                        lane_x,
                        _C07_PARTICLE_MOUTH_CENTER_LOCAL[1],
                        clear_z,
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    [
                        lane_x,
                        _C07_PARTICLE_EXIT_LOCAL[1],
                        clear_z,
                    ],
                    dtype=np.float64,
                ),
            )

        for name, local_transform in local_transforms.items():
            if (
                name in released_names
                or name in self._c07_delivered_particles
            ):
                continue
            local_target = np.asarray(
                local_transform,
                dtype=np.float64,
            ).reshape(4, 4).copy()
            proposed_active_local: np.ndarray | None = None
            reached_before_write = False
            active_waypoint_index = 0
            active_waypoints: tuple[np.ndarray, ...] = ()
            if name == active_name and active is not None:
                current_local = np.asarray(
                    active.get(
                        "current_local",
                        local_target[:3, 3],
                    ),
                    dtype=np.float64,
                ).reshape(3)
                active_waypoints = stage_waypoints(
                    name,
                    np.asarray(
                        local_transform,
                        dtype=np.float64,
                    ).reshape(4, 4)[:3, 3],
                )
                active_waypoint_index = int(
                    np.clip(
                        int(active.get("waypoint_index", 0)),
                        0,
                        len(active_waypoints) - 1,
                    )
                )
                goal = active_waypoints[active_waypoint_index]
                delta = goal - current_local
                distance = float(np.linalg.norm(delta))
                reached_before_write = bool(
                    distance <= _C07_PARTICLE_STAGE_REACHED_EPS_M
                )
                if not stage_can_advance:
                    proposed_active_local = current_local.copy()
                    reached_before_write = False
                elif reached_before_write:
                    proposed_active_local = goal.copy()
                else:
                    max_step = (
                        _C07_PARTICLE_STAGE_SPEED_MPS
                        * max(float(self._sim_dt), 0.0)
                    )
                    proposed_active_local = (
                        current_local
                        + delta
                        * min(
                            1.0,
                            max_step / max(distance, 1e-12),
                        )
                    )
                local_target[:3, 3] = proposed_active_local

            particle_target = target_matrix @ local_target
            applied, _linear_error, _angular_error = (
                self._set_body_rigid_grasp_target(
                    name,
                    particle_target,
                    # Pinned particles are pose-controlled; carrying a large
                    # synthetic root velocity into the next PhysX step is the
                    # mechanism that previously produced tunnelling.
                    previous_target=particle_target,
                )
            )
            if applied:
                last_targets[name] = particle_target.copy()
                if (
                    name == active_name
                    and active is not None
                    and proposed_active_local is not None
                ):
                    active["current_local"] = (
                        proposed_active_local.copy()
                    )
                    reached_after_write = bool(
                        float(
                            np.linalg.norm(
                                active_waypoints[
                                    active_waypoint_index
                                ]
                                - proposed_active_local
                            )
                        )
                        <= _C07_PARTICLE_STAGE_REACHED_EPS_M
                    )
                    final_waypoint = bool(
                        active_waypoint_index
                        == len(active_waypoints) - 1
                    )
                    if reached_after_write and not final_waypoint:
                        active["waypoint_index"] = (
                            active_waypoint_index + 1
                        )
                    elif final_waypoint and reached_before_write:
                        hold_successes = int(
                            active.get(
                                "outside_hold_successes",
                                0,
                            )
                        ) + 1
                        active["outside_hold_successes"] = hold_successes
                        if (
                            hold_successes
                            >= _C07_PARTICLE_RELEASE_EXIT_HOLD_STEPS
                        ):
                            released_names.add(name)
                            state["particle_stage_last_release_name"] = name
                            state[
                                "particle_stage_last_outside_world"
                            ] = particle_target[:3, 3].copy()
                            state[
                                "particle_stage_release_cooldown_s"
                            ] = _C07_PARTICLE_RELEASE_INTERVAL_SECONDS
                            state["particle_stage_active"] = None
                            active = None
                            active_name = ""

        tracked_names = set(local_transforms)
        state["particles_released"] = bool(
            tracked_names
            and tracked_names.issubset(
                released_names | self._c07_delivered_particles
            )
        )

    def _update_c07_particle_delivery_events(self) -> None:
        state = self._c07_pitcher_grasp
        cup = self._current_body_matrix("c07_cup")
        if state is None or cup is None:
            return
        cup_inverse = _invert_pose_matrix(cup)
        capture_enabled = bool(
            state.get("pour_xy_lock") or state.get("pour_lock_seen")
        )
        for index, name in enumerate(_C07_PARTICLE_NAMES):
            pose = self._current_body_pose(name)
            if pose is None:
                continue
            particle = _pose_vector_to_matrix(pose)
            local = (cup_inverse @ np.asarray([*pose[:3], 1.0]))[:3]
            previous = self._c07_previous_particle_cup_local.get(name)
            if (
                capture_enabled
                and name not in self._c07_delivered_particles
                and self._c07_particle_capture_candidate(local, previous)
            ):
                self._c07_delivered_particles.add(name)
            self._c07_previous_particle_cup_local[name] = local.copy()

            if name not in self._c07_delivered_particles:
                continue
            target = particle.copy()
            settle_local = self._c07_particle_settle_local(index)
            target[:3, 3] = (
                cup
                @ np.asarray([*settle_local, 1.0], dtype=np.float64)
            )[:3]
            self._set_body_soft_target(
                name,
                target,
                max_linear_speed=0.45,
                max_angular_speed=4.0,
            )

    @staticmethod
    def _c07_particle_is_in_pitcher(
        particle_world: np.ndarray,
        pitcher_inverse: np.ndarray,
    ) -> bool:
        local = (
            np.asarray(pitcher_inverse, dtype=np.float64).reshape(4, 4)
            @ np.asarray(
                [*np.asarray(particle_world, dtype=np.float64).reshape(3), 1.0],
                dtype=np.float64,
            )
        )[:3]
        return bool(
            float(np.linalg.norm(local[:2])) <= _C07_PITCHER_INNER_RADIUS
            and _C07_PITCHER_LOCAL_Z_RANGE[0]
            <= float(local[2])
            <= _C07_PITCHER_LOCAL_Z_RANGE[1]
        )

    def _c07_particles_remaining_in_pitcher(self) -> int:
        pitcher = self._current_body_matrix("c07_pitcher")
        if pitcher is None:
            return len(_C07_PARTICLE_NAMES)
        pitcher_inverse = _invert_pose_matrix(pitcher)
        pending_names = tuple(
            name
            for name in _C07_PARTICLE_NAMES
            if name not in self._c07_delivered_particles
        )
        if not pending_names:
            return 0
        remaining = 0
        observed = 0
        for name in pending_names:
            pose = self._current_body_pose(name)
            if pose is None:
                continue
            observed += 1
            if self._c07_particle_is_in_pitcher(pose[:3], pitcher_inverse):
                remaining += 1
        if observed == 0:
            remaining = len(pending_names)

        # A staged ball stops satisfying the coarse cylindrical "inside"
        # predicate while it is travelling through the neck and mouth.  It is
        # still kinematically pinned at that point and must not make the last
        # ball unlock the pitcher early.  Count every captured ball until its
        # outside-mouth hold has completed (or cup delivery has latched).
        state = self._c07_pitcher_grasp
        if state is not None:
            captured_value = state.get("particle_local_transforms", {})
            released_value = state.get("particle_released_names", set())
            if isinstance(captured_value, dict):
                released_names = (
                    released_value
                    if isinstance(released_value, set)
                    else set(released_value)
                )
                staged_remaining = (
                    set(captured_value)
                    - released_names
                    - self._c07_delivered_particles
                )
                remaining = max(remaining, len(staged_remaining))
        return remaining

    def _c07_empty_hold_steps_required(self) -> int:
        return max(
            1,
            int(
                math.ceil(
                    _C07_POUR_EMPTY_HOLD_SECONDS
                    / max(float(self._sim_dt), 1e-6)
                )
            ),
        )

    def _c07_ready_for_return_release(self, pitcher: np.ndarray) -> bool:
        return_zone = self._current_body_matrix("c07_return_zone")
        if return_zone is None:
            return False
        xy_distance = float(
            np.linalg.norm(pitcher[:2, 3] - return_zone[:2, 3])
        )
        expected_z = self._current_table_top_z() + _C07_PITCHER_HALF_HEIGHT
        z_error = abs(float(pitcher[2, 3]) - expected_z)
        pose = np.concatenate(
            [
                pitcher[:3, 3],
                _matrix_to_quat_wxyz(pitcher[:3, :3]),
            ]
        )
        upright = self._z_axis_alignment(pose)
        return bool(
            xy_distance <= _C07_RETURN_ASSIST_RADIUS
            and z_error <= _C07_RETURN_ASSIST_MAX_Z_ERROR
            and upright >= _C07_RETURN_ASSIST_UPRIGHT_ALIGNMENT
        )

    def _c07_ready_to_pin_on_pad(self, pitcher: np.ndarray) -> bool:
        return_zone = self._current_body_matrix("c07_return_zone")
        if return_zone is None:
            return False
        pitcher_matrix = np.asarray(pitcher, dtype=np.float64).reshape(4, 4)
        xy_distance = float(
            np.linalg.norm(
                pitcher_matrix[:2, 3] - return_zone[:2, 3]
            )
        )
        expected_z = self._current_table_top_z() + _C07_PITCHER_HALF_HEIGHT
        z_error = abs(float(pitcher_matrix[2, 3]) - expected_z)
        pose = np.concatenate(
            [
                pitcher_matrix[:3, 3],
                _matrix_to_quat_wxyz(pitcher_matrix[:3, :3]),
            ]
        )
        return bool(
            xy_distance <= _C07_RETURN_PIN_RADIUS
            and z_error <= _C07_RETURN_PIN_MAX_Z_ERROR
            and self._z_axis_alignment(pose)
            >= _C07_RETURN_PIN_UPRIGHT_ALIGNMENT
        )

    def _c07_return_pin_target(self, pitcher: np.ndarray) -> np.ndarray | None:
        return_zone = self._current_body_matrix("c07_return_zone")
        if return_zone is None:
            return None
        pitcher_matrix = np.asarray(pitcher, dtype=np.float64).reshape(4, 4)
        yaw = math.atan2(
            float(pitcher_matrix[1, 0]),
            float(pitcher_matrix[0, 0]),
        )
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        target = np.eye(4, dtype=np.float64)
        target[:3, :3] = np.asarray(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        target[:2, 3] = return_zone[:2, 3]
        target[2, 3] = (
            self._current_table_top_z() + _C07_PITCHER_HALF_HEIGHT
        )
        return target

    def _pin_c07_pitcher_on_return_pad(self, pitcher: np.ndarray) -> bool:
        target = self._c07_return_pin_target(pitcher)
        if target is None:
            return False
        pose = np.concatenate(
            [
                target[:3, 3],
                _matrix_to_quat_wxyz(target[:3, :3]),
            ]
        )
        self._set_body_rigid_grasp_target(
            "c07_pitcher",
            target,
            previous_target=target,
        )
        self._poses["c07_pitcher"] = pose.copy()
        self._c07_pitcher_pinned_pose = pose.copy()
        self._c07_pitcher_pinned = True
        self._c07_pitcher_grasp = None
        return True

    def _update_c07_pitcher_grasp_assist(self) -> None:
        if self._c07_pitcher_pinned:
            if self._c07_pitcher_pinned_pose is not None:
                pinned = _pose_vector_to_matrix(
                    self._c07_pitcher_pinned_pose
                )
                self._set_body_rigid_grasp_target(
                    "c07_pitcher",
                    pinned,
                    previous_target=pinned,
                )
            return
        palm = self._robot_link_pose_matrix("right")
        pitcher = self._current_body_matrix("c07_pitcher")
        if palm is None or pitcher is None:
            return
        raw_hand_command = self._raw_right_hand_command_max()
        state = self._c07_pitcher_grasp
        if state is None:
            tip_positions = self._b02_wuji_fingertip_positions()
            (
                handle_contact,
                handle_distance,
                handle_contact_count,
            ) = self._c07_handle_grasp_contact(
                pitcher,
                tip_positions,
            )
            if (
                raw_hand_command < _C07_GRASP_ATTACH_HAND_COMMAND
                or not handle_contact
            ):
                return
            grip_center = self._c07_finger_grip_center(palm)
            (
                particle_local_transforms,
                particle_last_targets,
            ) = self._c07_capture_particle_pins(pitcher)
            particle_stage_order = sorted(
                particle_local_transforms,
                key=lambda name: (
                    -float(
                        np.asarray(
                            particle_local_transforms[name],
                            dtype=np.float64,
                        ).reshape(4, 4)[2, 3]
                    ),
                    name,
                ),
            )
            state = {
                "phase": "carry",
                # Preserve the exact user-selected contact point.  The first
                # target equals the current pitcher pose, so attachment can
                # never make the object fly to a predefined grasp location.
                "hand_to_object": _invert_pose_matrix(palm) @ pitcher,
                "visual_hand": self._actual_right_hand_joint_pose(),
                "handle_distance": float(handle_distance),
                "handle_contact_count": int(handle_contact_count),
                "pour_xy_lock": False,
                "pour_lock_pending": False,
                "pour_lock_pending_elapsed_s": 0.0,
                "pour_hand_target_installed": False,
                "pour_lock_seen": False,
                "pour_complete": False,
                "pour_target_ready": False,
                "pour_requested_tilt_deg": 0.0,
                "pour_applied_tilt_deg": 0.0,
                "empty_steps": 0,
                "particles_remaining": len(_C07_PARTICLE_NAMES),
                "applied": False,
                "linear_error": 0.0,
                "angular_error": 0.0,
                "grip_center_world": grip_center.copy(),
                "last_target": pitcher.copy(),
                "particle_local_transforms": particle_local_transforms,
                "particle_last_targets": particle_last_targets,
                "particle_released_names": set(),
                "particles_released": False,
                "particle_release_tilt_deg": 0.0,
                "particle_drain_armed": False,
                "particle_drain_started": False,
                "particle_drain_elapsed_s": 0.0,
                "particle_target_stable_steps": 0,
                "particle_stability_target": pitcher.copy(),
                "particle_stage_order": particle_stage_order,
                "particle_stage_active": None,
                "particle_stage_release_cooldown_s": 0.0,
                "particle_stage_last_release_name": "",
                "particle_stage_last_outside_world": None,
            }
            self._c07_pitcher_grasp = state

        cup = self._current_body_matrix("c07_cup")
        if (
            cup is not None
            and not bool(state.get("pour_complete"))
            and (
                bool(state.get("pour_xy_lock"))
                or self._c07_pitcher_over_cup(pitcher, cup)
            )
        ):
            newly_locked = not bool(state.get("pour_xy_lock"))
            state["pour_xy_lock"] = True
            state["pour_lock_seen"] = True
            if newly_locked or state.get("pour_object_target") is None:
                state["phase"] = "pour_aligning"
                state["pour_lock_pending"] = True
                state["pour_lock_pending_elapsed_s"] = 0.0
                state["pour_hand_target_installed"] = False
                self._start_c07_tilt_path(state, pitcher, cup)

        self._update_c07_particle_delivery_events()
        remaining = self._c07_particles_remaining_in_pitcher()
        state["particles_remaining"] = int(remaining)
        if bool(state.get("pour_lock_seen")) and remaining == 0:
            state["empty_steps"] = int(state.get("empty_steps", 0)) + 1
        else:
            state["empty_steps"] = 0
        if (
            int(state["empty_steps"])
            >= self._c07_empty_hold_steps_required()
            and not bool(state.get("pour_complete"))
        ):
            # Do not wait for stale real-palm readback after the last ball
            # leaves.  Rebase free carrying from the final synchronized
            # commanded hand/object pair, so the unlock frame is continuous.
            raw_input_anchor = _pose_vector_to_matrix(
                np.asarray(
                    self._c_assist_raw_right_tcp_pose,
                    dtype=np.float64,
                ).reshape(7)
            )
            hand_to_object = np.asarray(
                state["hand_to_object"],
                dtype=np.float64,
            ).reshape(4, 4)
            controlled_hand_value = state.get("pour_hand_target")
            controlled_hand = (
                palm.copy()
                if controlled_hand_value is None
                else np.asarray(
                    controlled_hand_value,
                    dtype=np.float64,
                ).reshape(4, 4).copy()
            )
            controlled_object = controlled_hand @ hand_to_object
            state["carry_input_anchor"] = raw_input_anchor
            state["carry_hand_anchor"] = controlled_hand
            state["last_target"] = controlled_object.copy()
            state["phase"] = "return"
            state["pour_complete"] = True
            state["pour_xy_lock"] = False
            state["pour_lock_pending"] = False
            state["pour_hand_target_installed"] = False
            state["pour_target_ready"] = False
            state["pour_object_target"] = None

        if (
            bool(state.get("pour_complete"))
            and self._c07_ready_to_pin_on_pad(pitcher)
        ):
            self._pin_c07_pitcher_on_return_pad(pitcher)
            return

        target = palm @ np.asarray(
            state["hand_to_object"],
            dtype=np.float64,
        )
        commit_guided_target = True
        if bool(state.get("pour_complete")):
            rebased_hand_pose = self._c07_post_pour_hand_target(
                self._c_assist_raw_right_tcp_pose,
                state,
            )
            if rebased_hand_pose is not None:
                target = _pose_vector_to_matrix(
                    rebased_hand_pose
                ) @ np.asarray(
                    state["hand_to_object"],
                    dtype=np.float64,
                ).reshape(4, 4)
        elif (
            bool(state.get("pour_xy_lock"))
            and bool(state.get("pour_target_ready"))
            and state.get("pour_object_target") is not None
        ):
            guided_target = np.asarray(
                state["pour_object_target"],
                dtype=np.float64,
            ).reshape(4, 4)
            hand_tracking_ready = self._c07_pour_hand_tracking_ready(state)
            state["pour_write_hand_tracking_ready"] = hand_tracking_ready
            if hand_tracking_ready:
                target = guided_target
            else:
                fallback_target = state.get("last_target")
                target = (
                    pitcher.copy()
                    if fallback_target is None
                    else np.asarray(
                        fallback_target,
                        dtype=np.float64,
                    ).reshape(4, 4).copy()
                )
                commit_guided_target = False
            hand_target_value = state.get("pour_hand_target")
            if hand_target_value is not None:
                hand_target = np.asarray(
                    hand_target_value,
                    dtype=np.float64,
                ).reshape(4, 4)
                state["pour_hand_tracking_position_error"] = float(
                    np.linalg.norm(
                        palm[:3, 3] - hand_target[:3, 3]
                    )
                )
                state["pour_hand_tracking_angle_error"] = float(
                    np.linalg.norm(
                        self._rotation_error_vector(
                            hand_target[:3, :3],
                            palm[:3, :3],
                        )
                    )
                )

        # Write the pitcher first and give it zero residual root velocity.  A
        # successful write is the transaction boundary for advancing particle
        # staging; balls can never move to a future outlet pose before the
        # source pitcher reaches that same pose.
        phase_before_write = str(state.get("phase", ""))
        applied, linear_error, angular_error = (
            self._set_body_rigid_grasp_target(
                "c07_pitcher",
                target,
                previous_target=target,
            )
        )
        if applied:
            if phase_before_write == "pour_aligning" and commit_guided_target:
                fraction = float(
                    np.clip(
                        state.get(
                            "pour_alignment_candidate_fraction",
                            state.get("pour_alignment_fraction", 0.0),
                        ),
                        0.0,
                        1.0,
                    )
                )
                duration = max(
                    _C07_POUR_ALIGN_DURATION_SECONDS,
                    float(
                        state.get(
                            "pour_alignment_duration_s",
                            _C07_POUR_ALIGN_DURATION_SECONDS,
                        )
                    ),
                )
                state["pour_alignment_fraction"] = fraction
                state["pour_alignment_elapsed_s"] = fraction * duration
                if fraction >= 1.0:
                    # Commit the phase transition only after the pitcher root
                    # accepted the matching terminal hand/object pair.
                    state["pour_input_anchor"] = _pose_vector_to_matrix(
                        self._c_assist_raw_right_tcp_pose
                    )
                    hand_target_value = state.get("pour_hand_target")
                    if hand_target_value is not None:
                        state["pour_hand_anchor"] = np.asarray(
                            hand_target_value,
                            dtype=np.float64,
                        ).reshape(4, 4).copy()
                    state["phase"] = "pour_locked"
                    state["pour_lock_pending"] = False
                    state["pour_requested_tilt_deg"] = 0.0
                    state["pour_candidate_tilt_deg"] = 0.0
                    state["pour_applied_tilt_deg"] = 0.0
            elif phase_before_write == "pour_locked" and commit_guided_target:
                state["pour_applied_tilt_deg"] = float(
                    np.clip(
                        state.get(
                            "pour_candidate_tilt_deg",
                            state.get("pour_applied_tilt_deg", 0.0),
                        ),
                        0.0,
                        _C07_POUR_MAX_TOTAL_TILT_DEG,
                    )
                )
            self._update_c07_particle_pins(
                state,
                target,
                advance_release=commit_guided_target,
            )
        state.update(
            {
                "applied": bool(applied),
                "linear_error": float(linear_error),
                "angular_error": float(angular_error),
            }
        )
        if applied:
            state["last_target"] = np.asarray(
                target,
                dtype=np.float64,
            ).reshape(4, 4).copy()

    def _set_c08_tongs_joint_target(self, target_angle: float) -> bool:
        entry = self._task_dof_views.get("c08_tongs_joint")
        if entry is None or self._torch is None:
            return False
        view, dof_index, _initial = entry
        try:
            positions = view.get_dof_positions().clone()
            velocities = view.get_dof_velocities().clone()
            current = float(positions[0, int(dof_index)].item())
            max_step = _C08_TONGS_JOINT_MAX_RATE * max(
                float(self._sim_dt),
                1e-6,
            )
            next_value = current + float(
                np.clip(float(target_angle) - current, -max_step, max_step)
            )
            positions[:, int(dof_index)] = next_value
            velocities[:, int(dof_index)] = 0.0
            indices = self._torch.arange(
                int(view.count),
                dtype=self._torch.long,
                device=positions.device,
            )
            view.set_dof_positions(positions, indices)
            view.set_dof_velocities(velocities, indices)
            return True
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not control C08 tongs aperture: {exc}"
            )
            return False

    @staticmethod
    def _c08_tongs_tip_positions(
        root_matrix: np.ndarray,
        joint_angle: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fixed_local = np.asarray([0.135, -0.018, 0.012], dtype=np.float64)
        moving_root = np.asarray([-0.120, 0.018, 0.012], dtype=np.float64)
        moving_arm = np.asarray([0.255, 0.0, 0.0], dtype=np.float64)
        cosine = math.cos(float(joint_angle))
        sine = math.sin(float(joint_angle))
        moving_rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        moving_local = moving_root + moving_rotation @ moving_arm
        root = np.asarray(root_matrix, dtype=np.float64).reshape(4, 4)
        fixed = root[:3, 3] + root[:3, :3] @ fixed_local
        moving = root[:3, 3] + root[:3, :3] @ moving_local
        return fixed, moving, 0.5 * (fixed + moving)

    @staticmethod
    def _c08_tongs_pad_segments(
        root_matrix: np.ndarray,
        joint_angle: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the two real pad-capsule centerlines in world coordinates."""
        cosine = math.cos(float(joint_angle))
        sine = math.sin(float(joint_angle))
        moving_rotation = np.asarray(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        moving_root = np.asarray(
            [-0.120, 0.018, 0.012],
            dtype=np.float64,
        )
        moving_local = np.stack(
            [
                moving_root + moving_rotation @ point
                for point in _C08_TONGS_MOVING_PAD_SEGMENT
            ],
            axis=0,
        )
        root = np.asarray(root_matrix, dtype=np.float64).reshape(4, 4)

        def to_world(points: np.ndarray) -> np.ndarray:
            return (
                root[:3, 3].reshape(1, 3)
                + np.asarray(points, dtype=np.float64) @ root[:3, :3].T
            )

        return (
            to_world(_C08_TONGS_FIXED_PAD_SEGMENT),
            to_world(moving_local),
        )

    @staticmethod
    def _c08_tongs_collision_capsules(
        root_matrix: np.ndarray,
        joint_angle: float,
    ) -> tuple[tuple[np.ndarray, float], ...]:
        """Return every authored C08 tongs collider as a world-space capsule."""
        root = np.asarray(root_matrix, dtype=np.float64).reshape(4, 4)
        cosine = math.cos(float(joint_angle))
        sine = math.sin(float(joint_angle))
        moving_rotation = np.asarray(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        def to_world(points: np.ndarray) -> np.ndarray:
            local = np.asarray(points, dtype=np.float64).reshape(2, 3)
            return root[:3, 3] + local @ root[:3, :3].T

        capsules = [
            (to_world(points), float(radius))
            for points, radius in _C08_TONGS_FIXED_COLLISION_CAPSULES
        ]
        for points, radius in _C08_TONGS_MOVING_COLLISION_CAPSULES:
            local = (
                _C08_TONGS_MOVING_ARM_ROOT
                + np.asarray(points, dtype=np.float64) @ moving_rotation.T
            )
            capsules.append((to_world(local), float(radius)))
        return tuple(capsules)

    @staticmethod
    def _c08_capsule_centerline_samples(segment: np.ndarray) -> np.ndarray:
        centerline = np.asarray(segment, dtype=np.float64).reshape(2, 3)
        length = float(np.linalg.norm(centerline[1] - centerline[0]))
        sample_count = max(
            2,
            int(
                math.ceil(
                    length / max(_C08_TONGS_COLLISION_SAMPLE_STEP, 1e-6)
                )
            )
            + 1,
        )
        alpha = np.linspace(0.0, 1.0, sample_count).reshape(-1, 1)
        return centerline[0] + alpha * (centerline[1] - centerline[0])

    def _c08_tongs_scene_clearance(
        self,
        root_matrix: np.ndarray,
        joint_angle: float,
    ) -> float:
        """Return the minimum signed clearance from the full tongs to the table."""
        # The tongs must intentionally contact ice and may pass close to the
        # cup during pickup/drop. Only the table remains a blocking surface.
        minimum_clearance = math.inf
        table_top = float(self._current_table_top_z())

        for segment, radius in self._c08_tongs_collision_capsules(
            root_matrix,
            joint_angle,
        ):
            points = self._c08_capsule_centerline_samples(segment)
            minimum_clearance = min(
                minimum_clearance,
                float(np.min(points[:, 2] - radius - table_top)),
            )
        return float(minimum_clearance)

    def _c08_collision_safe_tongs_target(
        self,
        current_matrix: np.ndarray,
        requested_matrix: np.ndarray,
        joint_angle: float,
    ) -> tuple[np.ndarray, bool, float]:
        """Stop a rigidly attached tongs pose at the first swept collision."""
        current = np.asarray(current_matrix, dtype=np.float64).reshape(4, 4)
        requested = np.asarray(
            requested_matrix,
            dtype=np.float64,
        ).reshape(4, 4)
        start_clearance = self._c08_tongs_scene_clearance(
            current,
            joint_angle,
        )
        support_escape_active = bool(
            start_clearance <= _C08_TONGS_SUPPORT_ESCAPE_CLEARANCE
            and float(requested[2, 3] - current[2, 3])
            >= _C08_TONGS_SUPPORT_ESCAPE_MIN_LIFT
        )
        if support_escape_active:
            # A combined lift/translation/rotation can initially sweep a
            # resting collider deeper through the table or cradle and get
            # clipped to microscopic steps.  Leave the support vertically
            # first; the full requested motion resumes as soon as it is clear.
            support_escape_target = current.copy()
            support_escape_target[2, 3] = min(
                float(requested[2, 3]),
                float(current[2, 3]) + _C08_TONGS_SUPPORT_ESCAPE_STEP,
            )
            requested = support_escape_target
        distance = float(
            np.linalg.norm(requested[:3, 3] - current[:3, 3])
        )
        angle = float(
            np.linalg.norm(
                self._rotation_error_vector(
                    requested[:3, :3],
                    current[:3, :3],
                )
            )
        )
        tongs_radius = max(
            float(
                np.linalg.norm(
                    np.asarray(segment, dtype=np.float64).reshape(2, 3),
                    axis=1,
                ).max()
                + radius
            )
            for segment, radius in (
                *_C08_TONGS_FIXED_COLLISION_CAPSULES,
                *_C08_TONGS_MOVING_COLLISION_CAPSULES,
            )
        )
        maximum_point_motion = distance + tongs_radius * angle
        sweep_steps = max(
            1,
            int(
                math.ceil(
                    maximum_point_motion
                    / max(_C08_TONGS_SWEEP_MAX_POINT_STEP, 1e-6)
                )
            ),
        )
        normal_limit = -_C08_TONGS_MAX_ALLOWED_PENETRATION
        # If an imported/reset pose begins in shallow contact, allow motion
        # that does not make the existing overlap worse so it can move out.
        collision_limit = min(
            normal_limit,
            start_clearance - 1e-6,
        )
        safe_alpha = 0.0
        safe_target = current.copy()
        safe_clearance = start_clearance

        for step in range(1, sweep_steps + 1):
            alpha = float(step) / float(sweep_steps)
            candidate = self._c07_interpolate_pose(
                current,
                requested,
                alpha,
            )
            clearance = self._c08_tongs_scene_clearance(
                candidate,
                joint_angle,
            )
            if clearance >= collision_limit:
                safe_alpha = alpha
                safe_target = candidate
                safe_clearance = clearance
                if clearance >= normal_limit:
                    collision_limit = normal_limit
                continue

            lower = safe_alpha
            upper = alpha
            for _ in range(_C08_TONGS_COLLISION_BISECTION_STEPS):
                midpoint = 0.5 * (lower + upper)
                candidate = self._c07_interpolate_pose(
                    current,
                    requested,
                    midpoint,
                )
                clearance = self._c08_tongs_scene_clearance(
                    candidate,
                    joint_angle,
                )
                if clearance >= collision_limit:
                    lower = midpoint
                    safe_target = candidate
                    safe_clearance = clearance
                else:
                    upper = midpoint
            return safe_target, True, float(safe_clearance)

        return (
            requested.copy(),
            bool(support_escape_active),
            float(safe_clearance),
        )

    @classmethod
    def _c08_tongs_inner_gap(cls, joint_angle: float) -> float:
        fixed, moving, _midpoint = cls._c08_tongs_tip_positions(
            np.eye(4, dtype=np.float64),
            joint_angle,
        )
        center_distance = float(np.linalg.norm(moving - fixed))
        return max(0.0, center_distance - 2.0 * _C08_TONGS_PAD_RADIUS)

    @staticmethod
    def _c08_tongs_grip_segments_local(
        joint_angle: float,
    ) -> np.ndarray:
        cosine = math.cos(float(joint_angle))
        sine = math.sin(float(joint_angle))
        moving_rotation = np.asarray(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        moving_root = np.asarray(
            [-0.120, 0.018, 0.012],
            dtype=np.float64,
        )
        moving = np.stack(
            [
                moving_root + moving_rotation @ point
                for point in _C08_TONGS_MOVING_GRIP_SEGMENT
            ],
            axis=0,
        )
        return np.stack(
            [_C08_TONGS_FIXED_GRIP_SEGMENT, moving],
            axis=0,
        )

    @classmethod
    def _c08_tongs_grasp_contact(
        cls,
        tongs: np.ndarray,
        joint_angle: float,
        tip_positions: dict[str, np.ndarray],
    ) -> tuple[bool, float, int]:
        """Require fingers on opposite visible tongs grip rails."""
        thumb_world = tip_positions.get(_B02_WUJI_THUMB_TIP)
        if thumb_world is None:
            return False, math.inf, 0
        tongs_inverse = _invert_pose_matrix(
            np.asarray(tongs, dtype=np.float64).reshape(4, 4)
        )
        segments = cls._c08_tongs_grip_segments_local(joint_angle)

        def local_point(world: np.ndarray) -> np.ndarray:
            return (
                tongs_inverse
                @ np.asarray(
                    [*np.asarray(world, dtype=np.float64).reshape(3), 1.0],
                    dtype=np.float64,
                )
            )[:3]

        def rail_distances(local: np.ndarray) -> np.ndarray:
            return np.asarray(
                [
                    cls._point_segment_distance_3d(
                        local,
                        segment[0],
                        segment[1],
                    )
                    for segment in segments
                ],
                dtype=np.float64,
            )

        contact_limit = (
            _C08_TONGS_GRIP_RADIUS + _C08_TONGS_GRIP_CONTACT_MARGIN
        )
        thumb_local = local_point(thumb_world)
        thumb_distances = rail_distances(thumb_local)
        thumb_rail = int(np.argmin(thumb_distances))
        thumb_distance = float(thumb_distances[thumb_rail])
        if thumb_distance > contact_limit:
            return False, thumb_distance, 0

        valid_distances: list[float] = []
        thumb_world_np = np.asarray(
            thumb_world,
            dtype=np.float64,
        ).reshape(3)
        for name in _B02_WUJI_OPPOSING_TIPS:
            world = tip_positions.get(name)
            if world is None:
                continue
            world_np = np.asarray(world, dtype=np.float64).reshape(3)
            if (
                float(np.linalg.norm(world_np - thumb_world_np))
                > _C08_TONGS_PINCH_MAX_DISTANCE
            ):
                continue
            distances = rail_distances(local_point(world_np))
            rail = int(np.argmin(distances))
            distance = float(distances[rail])
            if rail == thumb_rail or distance > contact_limit:
                continue
            valid_distances.append(distance)

        if not valid_distances:
            return False, thumb_distance, 1
        return (
            True,
            max(thumb_distance, min(valid_distances)),
            1 + len(valid_distances),
        )

    @classmethod
    def _c08_tongs_visual_grip(
        cls,
        hand: np.ndarray,
        joint_angle: float,
    ) -> np.ndarray:
        values = np.asarray(hand, dtype=np.float64).reshape(-1)
        closed_pose = cls._tool_power_grip_pose(values)
        if values.size == _C08_TONGS_WUJI_OPEN_GRIP.size:
            # Anchor the visual fingers at the user's real contact pose.  Only
            # add the calibrated open-to-closed flexion as the tongs closes;
            # never replace the grasp with a hard-coded hand pose.
            open_pose = values.copy()
            closed_pose = (
                open_pose
                + _C08_TONGS_WUJI_CLOSED_GRIP
                - _C08_TONGS_WUJI_OPEN_GRIP
            )
        else:
            open_pose = closed_pose.copy()
            for start in range(0, open_pose.size - 3, 4):
                open_pose[start] *= 0.90
                open_pose[start + 2] *= 0.78
                open_pose[start + 3] *= 0.76
        denominator = max(
            _C08_TONGS_OPEN_RAD - _C08_TONGS_CLOSED_RAD,
            1e-6,
        )
        close_fraction = float(
            np.clip(
                (_C08_TONGS_OPEN_RAD - float(joint_angle)) / denominator,
                0.0,
                1.0,
            )
        )
        return (
            (1.0 - close_fraction) * np.asarray(open_pose, dtype=np.float64)
            + close_fraction * np.asarray(closed_pose, dtype=np.float64)
        )

    @staticmethod
    def _c08_ice_between_jaws(
        fixed_tip: np.ndarray,
        moving_tip: np.ndarray,
        ice_position: np.ndarray,
    ) -> tuple[bool, float, float]:
        fixed = np.asarray(fixed_tip, dtype=np.float64).reshape(3)
        moving = np.asarray(moving_tip, dtype=np.float64).reshape(3)
        ice = np.asarray(ice_position, dtype=np.float64).reshape(3)
        jaw = moving - fixed
        jaw_length = float(np.linalg.norm(jaw))
        if jaw_length <= 1e-8:
            return False, math.inf, math.inf
        jaw_axis = jaw / jaw_length
        projection = float(np.dot(ice - fixed, jaw_axis))
        fraction = projection / jaw_length
        closest = fixed + projection * jaw_axis
        perpendicular = float(np.linalg.norm(ice - closest))
        endpoint_clearance = min(projection, jaw_length - projection)
        between = bool(
            _C08_ICE_MIN_JAW_AXIS_FRACTION
            <= fraction
            <= _C08_ICE_MAX_JAW_AXIS_FRACTION
            and endpoint_clearance >= _C08_ICE_BETWEEN_JAWS_MARGIN
            and perpendicular <= _C08_ICE_MAX_PERP_DISTANCE
        )
        return between, perpendicular, fraction

    @staticmethod
    def _c08_pad_inner_surface_metrics(
        segment: np.ndarray,
        other_segment: np.ndarray,
        ice_position: np.ndarray,
        ice_rotation: np.ndarray,
    ) -> dict[str, float | bool]:
        """Measure a cube only against the inward face of one pad capsule."""
        pad = np.asarray(segment, dtype=np.float64).reshape(2, 3)
        other = np.asarray(
            other_segment,
            dtype=np.float64,
        ).reshape(2, 3)
        ice = np.asarray(ice_position, dtype=np.float64).reshape(3)
        rotation = np.asarray(
            ice_rotation,
            dtype=np.float64,
        ).reshape(3, 3)
        direction = pad[1] - pad[0]
        length_squared = float(np.dot(direction, direction))
        if length_squared <= 1e-12:
            return {
                "segment_valid": False,
                "segment_fraction": math.inf,
                "signed_inward": -math.inf,
                "alignment": -1.0,
                "contact_error": math.inf,
                "transverse_error": math.inf,
            }
        segment_fraction = float(
            np.dot(ice - pad[0], direction) / length_squared
        )
        closest = (
            pad[0]
            + np.clip(segment_fraction, 0.0, 1.0) * direction
        )
        segment_axis = direction / math.sqrt(length_squared)
        inward = np.mean(other, axis=0) - np.mean(pad, axis=0)
        inward -= float(np.dot(inward, segment_axis)) * segment_axis
        inward_norm = float(np.linalg.norm(inward))
        if inward_norm <= 1e-8:
            return {
                "segment_valid": False,
                "segment_fraction": segment_fraction,
                "signed_inward": -math.inf,
                "alignment": -1.0,
                "contact_error": math.inf,
                "transverse_error": math.inf,
            }
        inward /= inward_norm
        delta = ice - closest
        radial_distance = float(np.linalg.norm(delta))
        signed_inward = float(np.dot(delta, inward))
        alignment = (
            signed_inward / radial_distance
            if radial_distance > 1e-8
            else -1.0
        )
        transverse_error = float(
            np.linalg.norm(delta - signed_inward * inward)
        )
        # A rotated cube's support distance along the contact normal is the
        # sum of its three oriented half-extents projected onto that normal.
        cube_support = float(
            _C08_ICE_HALF_EXTENT
            * np.sum(np.abs(rotation.T @ inward))
        )
        contact_error = abs(
            radial_distance
            - (_C08_TONGS_PAD_RADIUS + cube_support)
        )
        return {
            "segment_valid": bool(0.0 <= segment_fraction <= 1.0),
            "segment_fraction": segment_fraction,
            "signed_inward": signed_inward,
            "alignment": alignment,
            "contact_error": contact_error,
            "transverse_error": transverse_error,
        }

    @classmethod
    def _c08_inner_pad_metrics(
        cls,
        tongs_matrix: np.ndarray,
        joint_angle: float,
        ice_position: np.ndarray,
        ice_rotation: np.ndarray | None = None,
    ) -> tuple[dict[str, float | bool], dict[str, float | bool]]:
        fixed_segment, moving_segment = cls._c08_tongs_pad_segments(
            tongs_matrix,
            joint_angle,
        )
        rotation = (
            np.eye(3, dtype=np.float64)
            if ice_rotation is None
            else np.asarray(ice_rotation, dtype=np.float64).reshape(3, 3)
        )
        return (
            cls._c08_pad_inner_surface_metrics(
                fixed_segment,
                moving_segment,
                ice_position,
                rotation,
            ),
            cls._c08_pad_inner_surface_metrics(
                moving_segment,
                fixed_segment,
                ice_position,
                rotation,
            ),
        )

    @classmethod
    def _c08_one_jaw_inner_contact(
        cls,
        tongs_matrix: np.ndarray,
        joint_angle: float,
        ice_position: np.ndarray,
        ice_rotation: np.ndarray | None = None,
    ) -> tuple[bool, str, float, float]:
        """Accept contact from a pad's inner face, never its outer/end face."""
        fixed, moving = cls._c08_inner_pad_metrics(
            tongs_matrix,
            joint_angle,
            ice_position,
            ice_rotation,
        )

        def contact(
            own: dict[str, float | bool],
            opposite: dict[str, float | bool],
        ) -> bool:
            return bool(
                own["segment_valid"]
                and opposite["segment_valid"]
                and float(own["signed_inward"]) > 0.0
                and float(opposite["signed_inward"]) > 0.0
                and float(own["alignment"])
                >= _C08_ICE_INNER_NORMAL_ALIGNMENT
                and float(own["transverse_error"])
                <= _C08_ICE_MAX_INNER_TRANSVERSE_ERROR
                and float(own["contact_error"])
                <= _C08_ICE_ONE_JAW_CONTACT_TOLERANCE
            )

        fixed_contact = contact(fixed, moving)
        moving_contact = contact(moving, fixed)
        if not fixed_contact and not moving_contact:
            return False, "", min(
                float(fixed["contact_error"]),
                float(moving["contact_error"]),
            ), min(
                float(fixed["transverse_error"]),
                float(moving["transverse_error"]),
            )
        side = (
            "fixed"
            if fixed_contact
            and (
                not moving_contact
                or float(fixed["contact_error"])
                <= float(moving["contact_error"])
            )
            else "moving"
        )
        selected = fixed if side == "fixed" else moving
        return (
            True,
            side,
            float(selected["contact_error"]),
            float(selected["transverse_error"]),
        )

    @classmethod
    def _c08_ice_between_inner_pads(
        cls,
        tongs_matrix: np.ndarray,
        joint_angle: float,
        ice_position: np.ndarray,
        ice_rotation: np.ndarray | None = None,
    ) -> tuple[bool, float, float]:
        """Require simultaneous contact with both inward pad surfaces."""
        fixed, moving = cls._c08_inner_pad_metrics(
            tongs_matrix,
            joint_angle,
            ice_position,
            ice_rotation,
        )
        common_inside = bool(
            fixed["segment_valid"]
            and moving["segment_valid"]
            and float(fixed["signed_inward"]) > 0.0
            and float(moving["signed_inward"]) > 0.0
            and float(fixed["alignment"])
            >= _C08_ICE_INNER_NORMAL_ALIGNMENT
            and float(moving["alignment"])
            >= _C08_ICE_INNER_NORMAL_ALIGNMENT
            and float(fixed["transverse_error"])
            <= _C08_ICE_MAX_INNER_TRANSVERSE_ERROR
            and float(moving["transverse_error"])
            <= _C08_ICE_MAX_INNER_TRANSVERSE_ERROR
        )
        max_contact_error = max(
            float(fixed["contact_error"]),
            float(moving["contact_error"]),
        )
        centered_fraction = 0.5 * (
            float(fixed["segment_fraction"])
            + float(moving["segment_fraction"])
        )
        return (
            bool(
                common_inside
                and max_contact_error
                <= _C08_ICE_BOTH_JAW_CONTACT_TOLERANCE
            ),
            max_contact_error,
            centered_fraction,
        )

    def _c08_ice_one_jaw_contact_candidate(
        self,
        tongs_matrix: np.ndarray,
        joint_angle: float,
    ) -> tuple[str, np.ndarray, str, float, float] | None:
        cup = self._current_body_matrix("c08_glass_cup")
        cup_inverse = None if cup is None else _invert_pose_matrix(cup)
        candidates: list[
            tuple[float, str, np.ndarray, str, float, float]
        ] = []
        for name in _C08_ICE_NAMES:
            if name in self._c08_delivered_ice:
                continue
            pose = self._current_body_pose(name)
            if pose is None:
                continue
            if cup_inverse is not None:
                cup_local = (
                    cup_inverse
                    @ np.asarray([*pose[:3], 1.0], dtype=np.float64)
                )[:3]
                if (
                    float(np.linalg.norm(cup_local[:2]))
                    <= _C08_CUP_INNER_RADIUS
                    and _C08_CUP_LOCAL_Z_RANGE[0]
                    <= float(cup_local[2])
                    <= _C08_CUP_LOCAL_Z_RANGE[1]
                ):
                    continue
            contact, side, contact_error, perpendicular = (
                self._c08_one_jaw_inner_contact(
                    tongs_matrix,
                    joint_angle,
                    pose[:3],
                    _pose_vector_to_matrix(pose)[:3, :3],
                )
            )
            if not contact:
                continue
            candidates.append(
                (
                    contact_error + 0.25 * perpendicular,
                    name,
                    pose,
                    side,
                    contact_error,
                    perpendicular,
                )
            )
        if not candidates:
            return None
        (
            _score,
            name,
            pose,
            side,
            contact_error,
            perpendicular,
        ) = min(candidates, key=lambda item: item[0])
        return name, pose, side, contact_error, perpendicular

    def _c08_ice_capture_candidate(
        self,
        tongs_matrix: np.ndarray,
        joint_angle: float,
    ) -> tuple[str, np.ndarray, float, float] | None:
        _fixed, _moving, midpoint = self._c08_tongs_tip_positions(
            tongs_matrix,
            joint_angle,
        )
        cup = self._current_body_matrix("c08_glass_cup")
        cup_inverse = None if cup is None else _invert_pose_matrix(cup)
        candidates: list[tuple[float, str, np.ndarray, float, float]] = []
        for name in _C08_ICE_NAMES:
            if name in self._c08_delivered_ice:
                continue
            pose = self._current_body_pose(name)
            if pose is None:
                continue
            if cup_inverse is not None:
                cup_local = (
                    cup_inverse
                    @ np.asarray([*pose[:3], 1.0], dtype=np.float64)
                )[:3]
                if (
                    float(np.linalg.norm(cup_local[:2]))
                    <= _C08_CUP_INNER_RADIUS
                    and _C08_CUP_LOCAL_Z_RANGE[0]
                    <= float(cup_local[2])
                    <= _C08_CUP_LOCAL_Z_RANGE[1]
                ):
                    continue
            between, contact_error, fraction = (
                self._c08_ice_between_inner_pads(
                    tongs_matrix,
                    joint_angle,
                    pose[:3],
                    _pose_vector_to_matrix(pose)[:3, :3],
                )
            )
            midpoint_distance = float(
                np.linalg.norm(pose[:3] - midpoint)
            )
            if (
                not between
                or midpoint_distance > _C08_ICE_AUTO_CLOSE_MAX_DISTANCE
            ):
                continue
            candidates.append(
                (
                    contact_error + 0.25 * midpoint_distance,
                    name,
                    pose,
                    contact_error,
                    fraction,
                )
            )
        if not candidates:
            return None
        _score, name, pose, perpendicular, fraction = min(
            candidates,
            key=lambda item: item[0],
        )
        return name, pose, perpendicular, fraction

    @classmethod
    def _c08_lock_tongs_target_xy(
        cls,
        target: np.ndarray,
        joint_angle: float,
        cup_xy: np.ndarray,
    ) -> np.ndarray:
        locked = np.asarray(target, dtype=np.float64).reshape(4, 4).copy()
        _fixed, _moving, midpoint = cls._c08_tongs_tip_positions(
            locked,
            joint_angle,
        )
        locked[:2, 3] += (
            np.asarray(cup_xy, dtype=np.float64).reshape(2)
            - midpoint[:2]
        )
        return locked

    def _c08_held_ice_relative_transform(
        self,
        tongs_matrix: np.ndarray,
        held: dict[str, Any],
        joint_angle: float,
    ) -> np.ndarray:
        relative = held.get("tongs_to_ice")
        if relative is not None:
            return np.asarray(relative, dtype=np.float64).reshape(4, 4)

        name = str(held.get("name", ""))
        pose = self._current_body_pose(name) if name else None
        if pose is not None:
            relative = (
                _invert_pose_matrix(
                    np.asarray(tongs_matrix, dtype=np.float64).reshape(4, 4)
                )
                @ _pose_vector_to_matrix(pose)
            )
            held["tongs_to_ice"] = relative.copy()
            return relative

        # Compatibility for old/injected states without a captured cube pose:
        # treating the jaw midpoint as the held center is still deterministic.
        _fixed, _moving, midpoint = self._c08_tongs_tip_positions(
            tongs_matrix,
            joint_angle,
        )
        relative = np.eye(4, dtype=np.float64)
        relative[:3, 3] = (
            np.asarray(tongs_matrix, dtype=np.float64).reshape(4, 4)[:3, :3].T
            @ (
                midpoint
                - np.asarray(tongs_matrix, dtype=np.float64).reshape(4, 4)[
                    :3, 3
                ]
            )
        )
        held["tongs_to_ice"] = relative.copy()
        return relative

    def _c08_held_ice_over_cup(
        self,
        tongs_matrix: np.ndarray,
        joint_angle: float,
        cup_matrix: np.ndarray,
        held: dict[str, Any],
    ) -> bool:
        # The release zone belongs to the actual cube, not the jaw midpoint or
        # a predicted tool-relative pose.  This avoids locking the hand while
        # offset ice is still beside the narrow glass opening.
        del tongs_matrix, joint_angle
        name = str(held.get("name", ""))
        pose = self._current_body_pose(name) if name else None
        if pose is None:
            return False
        ice_local = (
            _invert_pose_matrix(
                np.asarray(cup_matrix, dtype=np.float64).reshape(4, 4)
            )
            @ np.asarray([*pose[:3], 1.0], dtype=np.float64)
        )[:3]
        held["live_ice_cup_local"] = ice_local.copy()
        # The cup mouth is an infinite local-Z extrusion: entering its XY
        # aperture at any height begins the automatic centering/release flow.
        return bool(
            float(np.linalg.norm(ice_local[:2]))
            <= _C08_HELD_ICE_CUP_CAPTURE_RADIUS
        )

    def _c08_held_ice_cup_local(
        self,
        tongs_matrix: np.ndarray,
        joint_angle: float,
        cup_matrix: np.ndarray,
        held: dict[str, Any],
    ) -> np.ndarray:
        relative = self._c08_held_ice_relative_transform(
            tongs_matrix,
            held,
            joint_angle,
        )
        ice_world = (
            np.asarray(tongs_matrix, dtype=np.float64).reshape(4, 4)
            @ relative
        )
        ice_local = (
            _invert_pose_matrix(
                np.asarray(cup_matrix, dtype=np.float64).reshape(4, 4)
            )
            @ np.asarray([*ice_world[:3, 3], 1.0], dtype=np.float64)
        )[:3]
        return ice_local

    def _c08_next_delivery_slot(self) -> np.ndarray:
        index = min(
            len(self._c08_delivered_ice_local),
            len(_C08_DELIVERED_ICE_LOCAL_SLOTS) - 1,
        )
        return _C08_DELIVERED_ICE_LOCAL_SLOTS[index].copy()

    def _c08_release_local_z(self) -> float:
        next_slot = self._c08_next_delivery_slot()
        return max(
            _C08_DROP_ICE_LOCAL_Z,
            float(next_slot[2]) + _C08_DROP_CLEARANCE_ABOVE_SLOT,
        )

    def _c08_release_local_is_centered(
        self,
        ice_local: np.ndarray,
    ) -> bool:
        local = np.asarray(ice_local, dtype=np.float64).reshape(3)
        # Release follows the cup mouth's Z-axis extrusion as requested.  The
        # ice does not need to be at a fixed height once it is XY-centered.
        return bool(
            float(np.linalg.norm(local[:2]))
            <= _C08_RELEASE_CENTERED_CUP_RADIUS
        )

    def _c08_lock_held_ice_over_cup(
        self,
        target: np.ndarray,
        joint_angle: float,
        cup_matrix: np.ndarray,
        held: dict[str, Any],
    ) -> np.ndarray:
        """Lift past the rim, center the actual cube, then lower it."""
        locked = np.asarray(target, dtype=np.float64).reshape(4, 4).copy()
        relative = self._c08_held_ice_relative_transform(
            locked,
            held,
            joint_angle,
        )
        predicted_ice = locked @ relative
        cup = np.asarray(cup_matrix, dtype=np.float64).reshape(4, 4)
        cup_inverse = _invert_pose_matrix(cup)
        current_local = (
            cup_inverse
            @ np.asarray([*predicted_ice[:3, 3], 1.0], dtype=np.float64)
        )[:3]
        release_local_z = self._c08_release_local_z()
        rim_transit_local_z = max(
            release_local_z,
            _C08_CUP_RIM_LOCAL_Z
            + float(np.sqrt(2.0)) * _C08_ICE_HALF_EXTENT
            + _C08_RELEASE_RIM_TRANSIT_CLEARANCE,
        )
        radial_error = float(np.linalg.norm(current_local[:2]))
        if (
            radial_error > _C08_RELEASE_CENTERED_CUP_RADIUS
            and float(current_local[2]) < rim_transit_local_z
        ):
            # Never pull a low, offset cube diagonally through the glass rim.
            # Lift in place until the cube's full rotated footprint clears it.
            desired_local = current_local.copy()
            desired_local[2] = rim_transit_local_z
            stage = "lift_clear"
        elif radial_error > _C08_RELEASE_CENTERED_CUP_RADIUS:
            # Once clear of the rim, translate over the live cup center while
            # preserving the operator-selected tool orientation and height.
            desired_local = np.asarray(
                [
                    0.0,
                    0.0,
                    max(float(current_local[2]), rim_transit_local_z),
                ],
                dtype=np.float64,
            )
            stage = "center_above"
        else:
            desired_local = np.asarray(
                [0.0, 0.0, release_local_z],
                dtype=np.float64,
            )
            stage = "lower_centered"
        desired_center = (
            cup
            @ np.asarray(
                [*desired_local, 1.0],
                dtype=np.float64,
            )
        )[:3]
        locked[:3, 3] += desired_center - predicted_ice[:3, 3]
        held["release_path_stage"] = stage
        held["release_path_current_local"] = current_local.copy()
        held["release_path_goal_local"] = desired_local.copy()
        return locked

    @staticmethod
    def _abort_c08_release(grasp: dict[str, Any]) -> None:
        """Give control back without detaching the held cube."""
        grasp.update(
            {
                "phase": "holding_ice",
                "pour_xy_lock": False,
                "pour_lock_pending": False,
                "pour_hand_target_installed": False,
                "hand_target_matrix": None,
                "hand_target_installed": False,
                "release_armed": False,
                "release_retry_blocked": True,
                "release_suspended": False,
                "reanchor_after_release_abort": True,
                "drop_hand_pose": None,
                "drop_tongs_pose": None,
                "drop_goal_pose": None,
                "align_elapsed_s": 0.0,
                "cup_missing_elapsed_s": 0.0,
                "release_elapsed_s": 0.0,
                "open_command_applied": False,
                "open_command_success_s": 0.0,
            }
        )
        grasp.pop("hand_target_installed_matrix", None)
        grasp.pop("hand_target_installed_frame", None)

    def _c08_settle_delivered_ice(
        self,
        name: str,
        cup_matrix: np.ndarray,
    ) -> bool:
        """Pin a delivered cube in a stable cup-relative slot for the episode."""
        pose = self._current_body_pose(name)
        if pose is None:
            return False
        cup = np.asarray(cup_matrix, dtype=np.float64).reshape(4, 4)
        local = _invert_pose_matrix(cup) @ _pose_vector_to_matrix(pose)
        # Fill from the bottom in actual delivery order.  Object identity is
        # randomized and must not decide whether the first cube floats above
        # the rim in a high-numbered slot.
        slot_index = min(
            len(self._c08_delivered_ice_local),
            len(_C08_DELIVERED_ICE_LOCAL_SLOTS) - 1,
        )
        local[:3, :3] = np.eye(3, dtype=np.float64)
        local[:3, 3] = _C08_DELIVERED_ICE_LOCAL_SLOTS[
            slot_index % len(_C08_DELIVERED_ICE_LOCAL_SLOTS)
        ]
        target = cup @ local
        applied, _linear_error, _angular_error = (
            self._set_body_rigid_grasp_target(
                name,
                target,
                previous_target=target,
            )
        )
        if not applied:
            return False
        self._c08_delivered_ice.add(name)
        self._c08_delivered_ice_local[name] = local.copy()
        return True

    def _maintain_c08_delivered_ice(
        self,
        cup_matrix: np.ndarray | None,
    ) -> None:
        """Keep successful cubes visually and logically inside the live cup."""
        if cup_matrix is None:
            return
        cup = np.asarray(cup_matrix, dtype=np.float64).reshape(4, 4)
        for name in tuple(self._c08_delivered_ice):
            local = self._c08_delivered_ice_local.get(name)
            if local is None:
                # Never retain an unbacked success latch.
                self._c08_delivered_ice.discard(name)
                continue
            target = cup @ np.asarray(local, dtype=np.float64).reshape(4, 4)
            applied, _linear_error, _angular_error = (
                self._set_body_rigid_grasp_target(
                    name,
                    target,
                    previous_target=target,
                )
            )
            if applied:
                continue
            pose = self._current_body_pose(name)
            if pose is None:
                continue
            current_local = (
                _invert_pose_matrix(cup)
                @ np.asarray([*pose[:3], 1.0], dtype=np.float64)
            )[:3]
            if not (
                float(np.linalg.norm(current_local[:2]))
                <= _C08_CUP_INNER_RADIUS
                and _C08_CUP_LOCAL_Z_RANGE[0]
                <= float(current_local[2])
                <= _C08_CUP_LOCAL_Z_RANGE[1]
            ):
                self._c08_delivered_ice.discard(name)
                self._c08_delivered_ice_local.pop(name, None)

    def _c08_tongs_over_cup(
        self,
        tongs_matrix: np.ndarray,
        joint_angle: float,
        cup_matrix: np.ndarray,
    ) -> bool:
        _fixed, _moving, midpoint = self._c08_tongs_tip_positions(
            tongs_matrix,
            joint_angle,
        )
        xy_distance = float(
            np.linalg.norm(midpoint[:2] - cup_matrix[:2, 3])
        )
        height = float(midpoint[2] - cup_matrix[2, 3])
        return bool(
            xy_distance <= _C08_POUR_XY_CAPTURE_RADIUS
            and _C08_POUR_MIN_HEIGHT_ABOVE_CUP
            <= height
            <= _C08_POUR_MAX_HEIGHT_ABOVE_CUP
        )

    def _c08_ice_is_inside_cup(
        self,
        name: str,
        cup_matrix: np.ndarray,
    ) -> bool:
        pose = self._current_body_pose(name)
        if pose is None:
            return False
        local = (
            _invert_pose_matrix(cup_matrix)
            @ np.asarray([*pose[:3], 1.0], dtype=np.float64)
        )[:3]
        return bool(
            float(np.linalg.norm(local[:2])) <= _C08_CUP_INNER_RADIUS
            and _C08_CUP_LOCAL_Z_RANGE[0]
            <= float(local[2])
            <= _C08_CUP_LOCAL_Z_RANGE[1]
        )

    def _update_c08_ice_assist(
        self,
        tongs_matrix: np.ndarray,
        joint_angle: float,
    ) -> None:
        _fixed, _moving, midpoint = self._c08_tongs_tip_positions(
            tongs_matrix,
            joint_angle,
        )
        grasp = self._c08_tongs_grasp
        if grasp is None:
            return
        phase = str(grasp.get("phase", "carry_open"))
        if phase == "dropping":
            name = str(grasp.get("dropping_ice_name", ""))
            cup = self._current_body_matrix("c08_glass_cup")
            next_slot = self._c08_next_delivery_slot()
            delivery_ceiling_z = max(
                _C08_CUP_LOCAL_Z_RANGE[1],
                float(next_slot[2])
                + _C08_STACK_CAPTURE_HEIGHT_TOLERANCE,
            )
            grasp["drop_elapsed_s"] = float(
                grasp.get("drop_elapsed_s", 0.0)
            ) + max(float(self._sim_dt), 0.0)
            delivered = False
            swept_entry = False
            current_local: np.ndarray | None = None
            if name and cup is not None:
                pose = self._current_body_pose(name)
                if pose is not None:
                    current_local = (
                        _invert_pose_matrix(cup)
                        @ np.asarray([*pose[:3], 1.0], dtype=np.float64)
                    )[:3]
                    previous_local = grasp.get("drop_previous_cup_local")
                    if previous_local is not None:
                        previous = np.asarray(
                            previous_local,
                            dtype=np.float64,
                        ).reshape(3)
                        rim_z = _C08_CUP_RIM_LOCAL_Z
                        z_delta = float(previous[2] - current_local[2])
                        if (
                            float(previous[2]) > rim_z
                            and float(current_local[2]) <= rim_z
                            and z_delta > 1e-9
                        ):
                            fraction = float(
                                np.clip(
                                    (float(previous[2]) - rim_z) / z_delta,
                                    0.0,
                                    1.0,
                                )
                            )
                            crossing_xy = (
                                previous[:2]
                                + fraction
                                * (current_local[:2] - previous[:2])
                            )
                            swept_entry = bool(
                                float(np.linalg.norm(crossing_xy))
                                <= _C08_DROP_SWEEP_CAPTURE_RADIUS
                            )
                    delivered = bool(
                        float(np.linalg.norm(current_local[:2]))
                        <= _C08_CUP_INNER_RADIUS
                        and _C08_CUP_LOCAL_Z_RANGE[0]
                        <= float(current_local[2])
                        <= delivery_ceiling_z
                    )
                    grasp["drop_previous_cup_local"] = (
                        current_local.copy()
                    )

            settled = bool(
                (delivered or swept_entry)
                and cup is not None
                and self._c08_settle_delivered_ice(name, cup)
            )
            if settled:
                grasp.update(
                    {
                        "phase": "carry_open",
                        "pour_xy_lock": False,
                        "pour_lock_pending": False,
                        "pour_hand_target_installed": False,
                        "hand_target_matrix": None,
                        "hand_target_installed": False,
                        "release_armed": False,
                        "capture_armed": True,
                        "candidate_name": "",
                        "candidate_contact_side": "",
                        "dropping_ice_name": "",
                        "drop_hand_pose": None,
                        "drop_tongs_pose": None,
                        "drop_goal_pose": None,
                        "align_elapsed_s": 0.0,
                        "cup_missing_elapsed_s": 0.0,
                        "release_elapsed_s": 0.0,
                        "open_command_applied": False,
                        "open_command_success_s": 0.0,
                        "release_retry_blocked": False,
                        "release_suspended": False,
                        "drop_elapsed_s": 0.0,
                        "drop_previous_cup_local": None,
                    }
                )
            elif (
                float(grasp.get("drop_elapsed_s", 0.0))
                >= _C08_DROP_TIMEOUT_SECONDS
            ):
                # Never leave the operator's hand frozen forever after a
                # bounced/missed cube.  A timeout unlocks the tool but does not
                # count the cube as delivered, so it can be retried.
                grasp.update(
                    {
                        "phase": "carry_open",
                        "pour_xy_lock": False,
                        "pour_lock_pending": False,
                        "pour_hand_target_installed": False,
                        "hand_target_matrix": None,
                        "hand_target_installed": False,
                        "release_armed": False,
                        "capture_armed": True,
                        "candidate_name": "",
                        "candidate_contact_side": "",
                        "dropping_ice_name": "",
                        "drop_hand_pose": None,
                        "drop_tongs_pose": None,
                        "drop_goal_pose": None,
                        "align_elapsed_s": 0.0,
                        "cup_missing_elapsed_s": 0.0,
                        "release_elapsed_s": 0.0,
                        "open_command_applied": False,
                        "open_command_success_s": 0.0,
                        "release_retry_blocked": False,
                        "release_suspended": False,
                        "drop_elapsed_s": 0.0,
                        "drop_previous_cup_local": None,
                    }
                )
            return

        held = self._c08_held_ice
        if held is not None:
            tongs_to_ice = self._c08_held_ice_relative_transform(
                tongs_matrix,
                held,
                joint_angle,
            )
            target = (
                np.asarray(tongs_matrix, dtype=np.float64).reshape(4, 4)
                @ tongs_to_ice
            )
            release_suspended = bool(grasp.get("release_suspended"))
            if phase == "releasing" and not release_suspended:
                grasp["release_elapsed_s"] = float(
                    grasp.get("release_elapsed_s", 0.0)
                ) + max(float(self._sim_dt), 0.0)
            measured_open = bool(
                bool(grasp.get("joint_readback_valid", True))
                and abs(joint_angle) >= _C08_ICE_RELEASE_RAD
            )
            commanded_open = bool(
                bool(grasp.get("open_command_applied"))
                and float(grasp.get("open_command_success_s", 0.0))
                >= max(
                    _C08_RELEASE_OPEN_TIMEOUT_SECONDS,
                    _C08_RELEASE_OPEN_COMMAND_SECONDS,
                )
            )
            cup = (
                self._current_body_matrix("c08_glass_cup")
                if phase == "releasing"
                else None
            )
            release_ready = bool(
                phase == "releasing"
                and not release_suspended
                and cup is not None
                and (measured_open or commanded_open)
            )
            if (
                phase == "releasing"
                and not release_suspended
                and not release_ready
                and float(grasp.get("release_elapsed_s", 0.0))
                >= _C08_RELEASE_OPEN_FAILURE_TIMEOUT_SECONDS
            ):
                # An absent/failed DOF writer must never detach a cube through
                # visibly closed pads.  Keep the cube, restore user control,
                # and require leaving/re-entering the cup zone before retry.
                self._abort_c08_release(grasp)
                return
            applied, linear_error, angular_error = (
                self._set_body_rigid_grasp_target(
                    str(held["name"]),
                    target,
                    # On the release frame, explicitly clear any carried root
                    # velocity after centering the actual cube over the glass.
                    previous_target=(
                        target if release_ready else held.get("last_target")
                    ),
                )
            )
            held.update(
                {
                    "applied": bool(applied),
                    "linear_error": float(linear_error),
                    "angular_error": float(angular_error),
                }
            )
            if applied:
                held["last_target"] = target.copy()
            if release_ready and applied:
                name = str(held["name"])
                self._c08_held_ice = None
                previous_cup_local = None
                if cup is not None:
                    previous_cup_local = (
                        _invert_pose_matrix(cup)
                        @ np.asarray(
                            [*target[:3, 3], 1.0],
                            dtype=np.float64,
                        )
                    )[:3]
                grasp.update(
                    {
                        "phase": "dropping",
                        "release_armed": False,
                        "capture_armed": False,
                        "candidate_name": "",
                        "candidate_contact_side": "",
                        "dropping_ice_name": name,
                        "drop_elapsed_s": 0.0,
                        "drop_previous_cup_local": previous_cup_local,
                        "release_suspended": False,
                        "open_command_applied": False,
                        "open_command_success_s": 0.0,
                    }
                )
                return
            return

        if phase == "auto_closing":
            name = str(grasp.get("candidate_name", ""))
            pose = self._current_body_pose(name) if name else None
            if pose is None:
                grasp.update(
                    {
                        "phase": "carry_open",
                        "candidate_name": "",
                        "candidate_contact_side": "",
                    }
                )
                return
            midpoint_distance = float(np.linalg.norm(pose[:3] - midpoint))
            if midpoint_distance > _C08_ICE_AUTO_CLOSE_MAX_DISTANCE:
                grasp.update(
                    {
                        "phase": "carry_open",
                        "candidate_name": "",
                        "candidate_contact_side": "",
                    }
                )
                return
            capture = self._c08_ice_capture_candidate(
                tongs_matrix,
                joint_angle,
            )
            if capture is None or str(capture[0]) != name:
                one_jaw, _side, _contact_error, _transverse = (
                    self._c08_one_jaw_inner_contact(
                        tongs_matrix,
                        joint_angle,
                        pose[:3],
                        _pose_vector_to_matrix(pose)[:3, :3],
                    )
                )
                if (
                    one_jaw
                    and abs(joint_angle)
                    > _C08_TONGS_CLOSED_RAD + 1e-4
                ):
                    # Keep closing only while the originally contacted inner
                    # pad still touches the cube.  The both-pad test above is
                    # evaluated every substep, so a rotated cube is captured
                    # at its real projected width instead of being crushed to
                    # a fixed 0.075-rad aperture first.
                    return
                # The touched cube moved outside the inner faces, or the
                # commanded minimum aperture was reached without a valid
                # two-sided contact.  Never suspend it from a stale event.
                grasp.update(
                    {
                        "phase": "carry_open",
                        "candidate_name": "",
                        "candidate_contact_side": "",
                        "candidate_contact_error": math.inf,
                        "candidate_perpendicular_error": math.inf,
                    }
                )
                return
            pose = np.asarray(capture[1], dtype=np.float64).reshape(7)
            ice_matrix = _pose_vector_to_matrix(pose)
            self._c08_held_ice = {
                "name": name,
                "tongs_to_ice": (
                    _invert_pose_matrix(tongs_matrix) @ ice_matrix
                ),
                "rotation_local": (
                    tongs_matrix[:3, :3].T @ ice_matrix[:3, :3]
                ),
                "applied": False,
                "linear_error": 0.0,
                "angular_error": 0.0,
                # Hold the real two-pad contact aperture.  Continuing toward
                # a fixed "closed" angle after latching would drive the pads
                # through wide or rotated cubes.
                "grip_joint_angle": float(joint_angle),
                "last_target": ice_matrix.copy(),
            }
            self._c08_tongs_close_count += 1
            self._c08_tongs_was_open = False
            grasp.update(
                {
                    "phase": "holding_ice",
                    "release_armed": False,
                    "capture_armed": False,
                }
            )
            return

        if phase != "carry_open":
            return
        candidate = self._c08_ice_one_jaw_contact_candidate(
            tongs_matrix,
            joint_angle,
        )
        if candidate is None:
            grasp.update(
                {
                    "candidate_name": "",
                    "candidate_contact_side": "",
                    "candidate_contact_error": math.inf,
                    "candidate_perpendicular_error": math.inf,
                }
            )
            return
        name, _pose, side, contact_error, perpendicular = candidate
        grasp.update(
            {
                "candidate_name": name,
                "candidate_contact_side": side,
                "candidate_contact_error": float(contact_error),
                "candidate_perpendicular_error": float(perpendicular),
                "phase": "auto_closing",
                "capture_armed": True,
            }
        )

    def _update_c08_tongs_assist(self) -> None:
        cup = self._current_body_matrix("c08_glass_cup")
        # Delivered cubes remain cup-relative for the rest of the episode.
        # Run this even if a robot tensor is briefly unavailable.
        self._maintain_c08_delivered_ice(cup)

        palm = self._robot_link_pose_matrix("right")
        tongs = self._current_body_matrix("c08_tongs")
        if palm is None or tongs is None:
            return
        raw_hand_command = self._raw_right_hand_command_max()
        reading = self._physics_dof_position("c08_tongs_joint")
        joint_angle = (
            _C08_TONGS_REST_RAD
            if reading is None
            else float(reading[0])
        )
        if self._c08_tongs_grasp is None:
            (
                handle_contact,
                handle_distance,
                handle_contact_count,
            ) = self._c08_tongs_grasp_contact(
                tongs,
                joint_angle,
                self._b02_wuji_fingertip_positions(),
            )
            if (
                raw_hand_command >= _C_TOOL_GRASP_ATTACH_HAND_COMMAND
                and handle_contact
            ):
                reference_hand = self._actual_right_hand_joint_pose()
                self._c08_tongs_grasp = {
                    "phase": "carry_open",
                    # Preserve the exact contact transform selected by the
                    # user; no fixed grip point and no pickup teleport.
                    "hand_to_object": _invert_pose_matrix(palm) @ tongs,
                    "visual_hand": reference_hand.copy(),
                    "grip_reference_hand": reference_hand,
                    "handle_distance": float(handle_distance),
                    "handle_contact_count": int(handle_contact_count),
                    "pour_xy_lock": False,
                    "pour_lock_pending": False,
                    "pour_hand_target_installed": False,
                    "hand_target_matrix": None,
                    "hand_target_installed": False,
                    "pour_lock_seen": False,
                    "release_armed": False,
                    "capture_armed": True,
                    "candidate_name": "",
                    "candidate_contact_side": "",
                    "dropping_ice_name": "",
                    "drop_hand_pose": None,
                    "drop_tongs_pose": None,
                    "drop_goal_pose": None,
                    "align_elapsed_s": 0.0,
                    "cup_missing_elapsed_s": 0.0,
                    "release_elapsed_s": 0.0,
                    "open_command_applied": False,
                    "open_command_success_s": 0.0,
                    "release_retry_blocked": False,
                    "release_suspended": False,
                    "reanchor_after_release_abort": False,
                    "drop_elapsed_s": 0.0,
                    "drop_previous_cup_local": None,
                    "last_target": tongs.copy(),
                }
        grasp = self._c08_tongs_grasp
        if grasp is None:
            return
        self._c08_tongs_control_armed = True

        if bool(grasp.get("reanchor_after_release_abort")):
            # Resume from the current real contact transform after a safe
            # abort; returning user control must not snap the tongs.
            grasp["hand_to_object"] = _invert_pose_matrix(palm) @ tongs
            grasp["last_target"] = tongs.copy()
            grasp["hand_target_matrix"] = None
            grasp["hand_target_installed"] = False
            grasp.pop("hand_target_installed_matrix", None)
            grasp.pop("hand_target_installed_frame", None)
            grasp["reanchor_after_release_abort"] = False

        held = self._c08_held_ice
        held_over_cup = False
        if held is not None and cup is not None:
            held_over_cup = self._c08_held_ice_over_cup(
                tongs,
                joint_angle,
                cup,
                held,
            )
            if (
                bool(grasp.get("release_retry_blocked"))
                and not held_over_cup
            ):
                grasp["release_retry_blocked"] = False

        if (
            held is not None
            and str(grasp.get("phase")) == "holding_ice"
            and cup is not None
            and held_over_cup
            and not bool(grasp.get("release_retry_blocked"))
        ):
            live_ice = self._current_body_matrix(str(held.get("name", "")))
            if live_ice is not None:
                # Re-anchor from the live cube at the instant it enters the
                # release zone.  Any small PhysX/control lag is therefore
                # removed before computing the centered drop target.
                held["tongs_to_ice"] = _invert_pose_matrix(tongs) @ live_ice
                held["last_target"] = live_ice.copy()
            drop_goal = self._c08_lock_held_ice_over_cup(
                tongs,
                joint_angle,
                cup,
                held,
            )
            current_hand = tongs @ _invert_pose_matrix(
                np.asarray(
                    grasp["hand_to_object"],
                    dtype=np.float64,
                ).reshape(4, 4)
            )
            grasp.update(
                {
                    "phase": "aligning_release",
                    "pour_xy_lock": True,
                    "pour_lock_pending": True,
                    "pour_hand_target_installed": False,
                    "hand_target_matrix": current_hand.copy(),
                    "hand_target_installed": False,
                    "pour_lock_seen": True,
                    "release_armed": False,
                    "drop_tongs_pose": tongs.copy(),
                    "drop_goal_pose": drop_goal.copy(),
                    "align_elapsed_s": 0.0,
                    "cup_missing_elapsed_s": 0.0,
                    "release_elapsed_s": 0.0,
                    "open_command_applied": False,
                    "open_command_success_s": 0.0,
                    "release_suspended": False,
                    "drop_elapsed_s": 0.0,
                    "drop_previous_cup_local": None,
                    "drop_hand_pose": np.concatenate(
                        [
                            current_hand[:3, 3],
                            _matrix_to_quat_wxyz(
                                current_hand[:3, :3]
                            ),
                        ]
                    ),
                }
            )

        carried_target = palm @ np.asarray(
            grasp["hand_to_object"],
            dtype=np.float64,
        )
        target = carried_target.copy()
        release_transition_ready = False
        phase = str(grasp.get("phase", "carry_open"))
        if (
            bool(grasp.get("pour_xy_lock"))
            and phase in {"aligning_release", "releasing"}
            and held is not None
        ):
            if cup is None:
                grasp["cup_missing_elapsed_s"] = float(
                    grasp.get("cup_missing_elapsed_s", 0.0)
                ) + max(float(self._sim_dt), 0.0)
                grasp["release_suspended"] = True
                if phase == "releasing":
                    grasp["release_elapsed_s"] = 0.0
                target = tongs.copy()
                if (
                    float(grasp["cup_missing_elapsed_s"])
                    >= _C08_CUP_POSE_MISSING_TIMEOUT_SECONDS
                ):
                    self._abort_c08_release(grasp)
                    grasp["hand_to_object"] = _invert_pose_matrix(palm) @ tongs
                    grasp["last_target"] = tongs.copy()
                    grasp["reanchor_after_release_abort"] = False
                    target = tongs.copy()
            else:
                grasp["cup_missing_elapsed_s"] = 0.0
                grasp["release_suspended"] = False
                hand_to_object_inverse = _invert_pose_matrix(
                    np.asarray(
                        grasp["hand_to_object"],
                        dtype=np.float64,
                    ).reshape(4, 4)
                )
                if phase == "aligning_release":
                    drop_goal = self._c08_lock_held_ice_over_cup(
                        tongs,
                        joint_angle,
                        cup,
                        held,
                    )
                    grasp["drop_goal_pose"] = drop_goal.copy()
                    grasp["align_elapsed_s"] = float(
                        grasp.get("align_elapsed_s", 0.0)
                    ) + max(float(self._sim_dt), 0.0)
                    if grasp.get("hand_target_matrix") is None:
                        grasp["hand_target_matrix"] = (
                            carried_target @ hand_to_object_inverse
                        )
                        grasp["hand_target_installed"] = False
                        grasp.pop("hand_target_installed_matrix", None)
                        grasp.pop("hand_target_installed_frame", None)
                        grasp["pour_hand_target_installed"] = False
                    self._assist_hand_tracking_ready(
                        "right",
                        grasp,
                        position_tolerance=(
                            _C08_RELEASE_ALIGN_POSITION_TOLERANCE
                        ),
                        angle_tolerance=_C08_RELEASE_ALIGN_ANGLE_TOLERANCE,
                    )
                    release_centered_over_cup = False
                    live_ice_pose = self._current_body_pose(
                        str(held.get("name", ""))
                    )
                    if live_ice_pose is not None:
                        ice_local = (
                            _invert_pose_matrix(cup)
                            @ np.asarray(
                                [*live_ice_pose[:3], 1.0],
                                dtype=np.float64,
                            )
                        )[:3]
                        held["live_release_ice_cup_local"] = ice_local.copy()
                        release_centered_over_cup = (
                            self._c08_release_local_is_centered(
                                ice_local,
                            )
                        )
                        if release_centered_over_cup:
                            live_ice_matrix = _pose_vector_to_matrix(
                                live_ice_pose
                            )
                            # Freeze the exact real cube/tool relation at the
                            # valid release pose.  Small physics lag must not
                            # make the following collision check use an older
                            # predicted side position.
                            held["tongs_to_ice"] = (
                                _invert_pose_matrix(tongs)
                                @ live_ice_matrix
                            )
                            held["last_target"] = live_ice_matrix.copy()
                    release_transition_ready = bool(
                        release_centered_over_cup
                    )
                    grasp["release_centered_over_cup"] = bool(
                        release_centered_over_cup
                    )
                    grasp["release_transition_reason"] = (
                        "auto_centered"
                        if release_transition_ready
                        else ""
                    )
                    if release_transition_ready:
                        # The real cube is already at the valid release pose.
                        # Freeze that exact tool pose before opening.
                        target = tongs.copy()
                    else:
                        # Guide the robot hand along the automatic path while
                        # keeping the tongs rigidly attached to the live palm.
                        # The tool must never move ahead and leave the hand
                        # visibly grasping empty space.
                        guided_target = self._bounded_pose_step(
                            tongs,
                            drop_goal,
                            dt=self._control_dt,
                            max_linear_speed=(
                                _C08_RELEASE_ALIGN_MAX_LINEAR_SPEED
                            ),
                            max_angular_speed=(
                                _C_TOOL_ASSIST_MAX_ANGULAR_SPEED
                            ),
                        )
                        safe_guided_target, _, _ = (
                            self._c08_collision_safe_tongs_target(
                                tongs,
                                guided_target,
                                joint_angle,
                            )
                        )
                        hand_target = (
                            safe_guided_target @ hand_to_object_inverse
                        )
                        grasp["hand_target_matrix"] = hand_target.copy()
                        grasp["hand_target_installed"] = False
                        grasp.pop("hand_target_installed_matrix", None)
                        grasp.pop("hand_target_installed_frame", None)
                        grasp["pour_hand_target_installed"] = False
                        grasp["drop_hand_pose"] = (
                            self._pose_matrix_to_vector(hand_target)
                        )
                        target = carried_target.copy()
                    if (
                        not release_transition_ready
                        and float(grasp.get("align_elapsed_s", 0.0))
                        >= _C08_RELEASE_ALIGN_TIMEOUT_SECONDS
                    ):
                        self._abort_c08_release(grasp)
                        grasp["hand_to_object"] = (
                            _invert_pose_matrix(palm) @ tongs
                        )
                        grasp["last_target"] = tongs.copy()
                        grasp["reanchor_after_release_abort"] = False
                        target = tongs.copy()
                        release_transition_ready = False
                else:
                    target = np.asarray(
                        grasp.get("drop_tongs_pose", tongs),
                        dtype=np.float64,
                    ).reshape(4, 4).copy()
        elif (
            bool(grasp.get("pour_xy_lock"))
            and phase == "dropping"
            and grasp.get("drop_tongs_pose") is not None
        ):
            target = np.asarray(
                grasp["drop_tongs_pose"],
                dtype=np.float64,
            ).reshape(4, 4).copy()

        requested_target = target.copy()
        target, collision_blocked, collision_clearance = (
            self._c08_collision_safe_tongs_target(
                tongs,
                requested_target,
                joint_angle,
            )
        )
        grasp.update(
            {
                "collision_guard_blocked": bool(collision_blocked),
                "collision_guard_clearance": float(collision_clearance),
                "collision_guard_requested_target": requested_target.copy(),
            }
        )
        if (
            collision_blocked
            and release_transition_ready
            and cup is not None
            and held is not None
        ):
            # The table guard must block an unsafe tool target, but it must not
            # deadlock a cube whose live and current safe poses are centered.
            # The live cube is already valid.  Hold the current collision-safe
            # tongs pose while opening rather than applying another clipped
            # motion that could drag the cube back toward the rim.
            target = tongs.copy()
            safe_ice_local = self._c08_held_ice_cup_local(
                tongs,
                joint_angle,
                cup,
                held,
            )
            safe_write_centered = self._c08_release_local_is_centered(
                safe_ice_local,
            )
            grasp["release_safe_write_centered"] = bool(
                safe_write_centered
            )
            if not safe_write_centered:
                release_transition_ready = False
                grasp["release_centered_over_cup"] = False
                grasp["release_transition_reason"] = ""
        applied, linear_error, angular_error = (
            self._set_body_rigid_grasp_target(
                "c08_tongs",
                target,
                previous_target=grasp.get("last_target"),
            )
        )
        grasp.update(
            {
                "applied": bool(applied),
                "linear_error": float(linear_error),
                "angular_error": float(angular_error),
            }
        )
        if applied:
            grasp["last_target"] = np.asarray(
                target,
                dtype=np.float64,
            ).reshape(4, 4).copy()
            if bool(grasp.get("pour_xy_lock")) and phase in {
                "aligning_release",
                "releasing",
            }:
                grasp["drop_tongs_pose"] = target.copy()
                hand_target_value = grasp.get("hand_target_matrix")
                drop_hand = (
                    target
                    @ _invert_pose_matrix(
                        np.asarray(
                            grasp["hand_to_object"],
                            dtype=np.float64,
                        ).reshape(4, 4)
                    )
                    if hand_target_value is None
                    else np.asarray(
                        hand_target_value,
                        dtype=np.float64,
                    ).reshape(4, 4)
                )
                grasp["drop_hand_pose"] = self._pose_matrix_to_vector(
                    drop_hand
                )
        if release_transition_ready:
            # A verified XY-centered cube can already be at the desired tool
            # pose, in which case the rigid-body writer reports no change.
            # That must not suppress the automatic release transition.
            release_tongs = target if applied else tongs
            grasp["hand_target_matrix"] = (
                release_tongs
                @ _invert_pose_matrix(
                    np.asarray(
                        grasp["hand_to_object"],
                        dtype=np.float64,
                    ).reshape(4, 4)
                )
            )
            grasp["drop_hand_pose"] = self._pose_matrix_to_vector(
                np.asarray(
                    grasp["hand_target_matrix"],
                    dtype=np.float64,
                ).reshape(4, 4)
            )
            grasp.update(
                {
                    "phase": "releasing",
                    "pour_lock_pending": False,
                    "release_armed": True,
                    "release_elapsed_s": 0.0,
                    "open_command_applied": False,
                    "open_command_success_s": 0.0,
                }
            )

        phase = str(grasp.get("phase", "carry_open"))
        if (
            phase == "releasing"
            and not bool(grasp.get("release_suspended"))
        ) or phase == "dropping":
            target_angle = _C08_TONGS_REST_RAD
        elif phase == "auto_closing":
            target_angle = _C08_TONGS_CLOSED_RAD
        elif self._c08_held_ice is not None:
            target_angle = float(
                self._c08_held_ice.get(
                    "grip_joint_angle",
                    _C08_TONGS_CLOSED_RAD,
                )
            )
        else:
            # Pickup cannot crush the tongs.  They remain at the calibrated
            # 39 mm maximum inner gap until one inner jaw touches an ice cube.
            target_angle = _C08_TONGS_REST_RAD
        joint_write_applied = self._set_c08_tongs_joint_target(target_angle)
        if (
            phase == "releasing"
            and not bool(grasp.get("release_suspended"))
            and joint_write_applied
        ):
            grasp["open_command_applied"] = True
            grasp["open_command_success_s"] = float(
                grasp.get("open_command_success_s", 0.0)
            ) + max(float(self._sim_dt), 0.0)
        elif phase == "releasing":
            # The stale-readback fallback represents one continuous, known-
            # successful opening command.  A failed write or cup-pose
            # suspension commands CLOSED and must restart that evidence.
            grasp["open_command_applied"] = False
            grasp["open_command_success_s"] = 0.0
        post_reading = self._physics_dof_position("c08_tongs_joint")
        grasp["joint_readback_valid"] = post_reading is not None
        if post_reading is not None:
            joint_angle = float(post_reading[0])
        effective_tongs_target = (
            np.asarray(target, dtype=np.float64).reshape(4, 4)
            if applied
            else tongs
        )
        self._update_c08_ice_assist(
            effective_tongs_target,
            joint_angle,
        )
        grasp["visual_hand"] = self._c08_tongs_visual_grip(
            np.asarray(
                grasp.get(
                    "grip_reference_hand",
                    self._c_assist_raw_right_hand,
                ),
                dtype=np.float64,
            ),
            joint_angle,
        )
        grasp["jaw_inner_gap"] = self._c08_tongs_inner_gap(joint_angle)

    def _c09_paper_grasp_candidate(
        self,
        side: str,
        palm: np.ndarray,
        tip_positions: dict[str, np.ndarray] | None = None,
    ) -> tuple[str, np.ndarray, float, int] | None:
        """Select only a paper whose visible raised grip tab is pinched."""
        occupied = {
            str(state["name"])
            for held_side, state in self._c09_paper_grasps.items()
            if held_side != side
        }
        tips = (
            self._wuji_fingertip_positions(side)
            if tip_positions is None
            else tip_positions
        )
        candidates: list[tuple[float, str, np.ndarray, int]] = []
        for name in _C09_PAPER_NAMES:
            if name in occupied:
                continue
            paper = self._current_body_matrix(name)
            if paper is None:
                continue
            contact, error, count = self._visible_box_pinch_contact(
                paper,
                tips,
                side=side,
                center_local=_C09_GRIP_TAB_CENTERS_LOCAL[name],
                half_extents=_C09_GRIP_TAB_HALF_EXTENTS,
                contact_margin=_C09_GRIP_TAB_CONTACT_MARGIN,
                near_margin=_C09_GRIP_TAB_NEAR_MARGIN,
                pinch_distance_range=(
                    _C09_GRIP_TAB_PINCH_MIN_DISTANCE,
                    _C09_GRIP_TAB_PINCH_MAX_DISTANCE,
                ),
            )
            if not contact:
                continue
            candidates.append((float(error), name, paper, int(count)))
        if not candidates:
            return None
        score, name, paper, count = min(candidates, key=lambda item: item[0])
        return name, paper, score, count

    @staticmethod
    def _c09_pair_capture_geometry(
        paper0: np.ndarray,
        paper1: np.ndarray,
    ) -> tuple[float, float, float]:
        matrix0 = _pose_vector_to_matrix(paper0)
        matrix1 = _pose_vector_to_matrix(paper1)
        relative = _invert_pose_matrix(matrix0) @ matrix1
        full_extents = _C09_PAPER_HALF_EXTENTS_XY * 2.0
        overlap_extents = np.maximum(
            0.0,
            full_extents - np.abs(relative[:2, 3]),
        )
        overlap = float(np.prod(overlap_extents) / np.prod(full_extents))
        z_error = abs(float(relative[2, 3]))
        # A 180-degree flip puts the authored colored staple corners at
        # opposite ends.  It is not an aligned stack even though abs(dot)=1.
        x_alignment = float(np.dot(matrix0[:3, 0], matrix1[:3, 0]))
        normal_alignment = float(
            np.dot(matrix0[:3, 2], matrix1[:3, 2])
        )
        return overlap, z_error, min(x_alignment, normal_alignment)

    @staticmethod
    def _c09_other_paper(name: str) -> str:
        return "c09_paper_1" if name == "c09_paper_0" else "c09_paper_0"

    def _c09_grasp_side_for_paper(self, name: str) -> str | None:
        for side, state in self._c09_paper_grasps.items():
            if str(state.get("name")) == name:
                return side
        return None

    @staticmethod
    def _c09_paired_target(
        anchor_target: np.ndarray,
        *,
        anchor_name: str,
        paired_name: str,
    ) -> np.ndarray:
        target = np.asarray(anchor_target, dtype=np.float64).reshape(4, 4).copy()
        sign = (
            1.0
            if anchor_name == "c09_paper_0" and paired_name == "c09_paper_1"
            else -1.0
        )
        target[:3, 3] += (
            sign * _C09_STACK_SEPARATION * target[:3, 2]
        )
        return target

    def _c09_install_hand_target(
        self,
        side: str | None,
        object_target: np.ndarray,
    ) -> None:
        if side is None:
            return
        state = self._c09_paper_grasps.get(side)
        if state is None:
            return
        state["hand_target_matrix"] = (
            np.asarray(object_target, dtype=np.float64).reshape(4, 4)
            @ _invert_pose_matrix(
                np.asarray(
                    state["hand_to_object"],
                    dtype=np.float64,
                ).reshape(4, 4)
            )
        )

    def _c09_clear_hand_targets(self) -> None:
        for state in self._c09_paper_grasps.values():
            state["hand_target_matrix"] = None
            state["hand_target_installed"] = False

    def _c09_hand_tracking_ready(
        self,
        side: str | None,
    ) -> bool:
        if side is None:
            return True
        state = self._c09_paper_grasps.get(side)
        if state is None:
            return False
        return self._assist_hand_tracking_ready(
            side,
            state,
            position_tolerance=_C09_HAND_TRACK_POSITION_TOLERANCE,
            angle_tolerance=_C09_HAND_TRACK_ANGLE_TOLERANCE,
        )

    def _c09_stack_mark_world(
        self,
        paper_matrix: np.ndarray,
    ) -> np.ndarray:
        return (
            np.asarray(paper_matrix, dtype=np.float64).reshape(4, 4)
            @ np.asarray([*_C09_STAPLE_MARK_LOCAL, 1.0], dtype=np.float64)
        )[:3]

    def _c09_stack_insertion_candidate(
        self,
        paper_matrix: np.ndarray,
        stapler: np.ndarray,
    ) -> bool:
        reading = self._physics_dof_position("c09_stapler_top_joint")
        top_angle = 0.0 if reading is None else float(reading[0] - reading[1])
        if top_angle < _C09_TOP_OPEN_RAD:
            return False
        paper = np.asarray(paper_matrix, dtype=np.float64).reshape(4, 4)
        stapler_matrix = np.asarray(stapler, dtype=np.float64).reshape(4, 4)
        mark_local = (
            _invert_pose_matrix(stapler_matrix)
            @ np.asarray(
                [*self._c09_stack_mark_world(paper), 1.0],
                dtype=np.float64,
            )
        )[:3]
        x_alignment = float(
            np.dot(paper[:3, 0], stapler_matrix[:3, 0])
        )
        normal_alignment = float(
            np.dot(paper[:3, 2], stapler_matrix[:3, 2])
        )
        return bool(
            float(
                np.linalg.norm(
                    mark_local[:2] - _C09_STAPLE_ZONE_LOCAL[:2]
                )
            )
            <= _C09_STAPLER_CAPTURE_XY
            and abs(float(mark_local[2] - _C09_STAPLE_ZONE_LOCAL[2]))
            <= _C09_STAPLER_CAPTURE_Z
            and x_alignment >= _C09_STAPLER_CAPTURE_ALIGNMENT
            and normal_alignment >= _C09_STAPLER_CAPTURE_ALIGNMENT
        )

    @staticmethod
    def _c09_insertion_goal(
        stapler: np.ndarray,
    ) -> np.ndarray:
        stapler_matrix = np.asarray(
            stapler,
            dtype=np.float64,
        ).reshape(4, 4)
        target = stapler_matrix.copy()
        zone_world = (
            stapler_matrix
            @ np.asarray([*_C09_STAPLE_ZONE_LOCAL, 1.0], dtype=np.float64)
        )[:3]
        target[:3, 3] = (
            zone_world
            - target[:3, :3] @ _C09_STAPLE_MARK_LOCAL
        )
        return target

    def _c09_write_stack_targets(
        self,
        targets: dict[str, np.ndarray],
        previous: dict[str, np.ndarray],
    ) -> bool:
        rollback_targets: dict[str, np.ndarray] = {}
        for name in _C09_PAPER_NAMES:
            current = self._current_body_matrix(name)
            if current is None:
                current = previous.get(name)
            if current is not None:
                rollback_targets[name] = np.asarray(
                    current,
                    dtype=np.float64,
                ).reshape(4, 4).copy()

        applied_names: list[str] = []
        for name in _C09_PAPER_NAMES:
            applied, _linear, _angular = self._set_body_rigid_grasp_target(
                name,
                targets[name],
                previous_target=previous.get(name),
            )
            if applied:
                applied_names.append(name)
                continue
            # The two sheets form one visible stack.  If either root writer
            # fails, put every already-written sheet back so a failed frame
            # cannot leave one sheet teleported away from the other.
            for applied_name in reversed(applied_names):
                rollback = rollback_targets.get(applied_name)
                if rollback is None:
                    continue
                restored, _linear, _angular = (
                    self._set_body_rigid_grasp_target(
                    applied_name,
                    rollback,
                    previous_target=rollback,
                )
                )
                if not restored:
                    # The sheets are now in an unknown split state.  Reset the
                    # episode instead of continuing collection with an
                    # invisible/partial attachment.
                    self._manual_reset_requested = True
            return False
        return True

    def _c09_pin_stapler_root(self) -> None:
        stapler = self._current_body_matrix("c09_stapler")
        if stapler is None:
            return
        if self._c09_stapler_anchor is None:
            self._c09_stapler_anchor = stapler.copy()
        anchor = np.asarray(
            self._c09_stapler_anchor,
            dtype=np.float64,
        ).reshape(4, 4)
        self._set_body_rigid_grasp_target(
            "c09_stapler",
            anchor,
            previous_target=anchor,
        )

    def _update_c09_paper_assist(self) -> None:
        # The base must not fly away when the free hand opens or presses the
        # top hinge.  Only the root is held; the authored hinge stays dynamic.
        self._c09_pin_stapler_root()
        palms = {
            side: self._robot_link_pose_matrix(side)
            for side in ("right", "left")
        }
        for side in tuple(self._c09_paper_grasps):
            if (
                palms.get(side) is None
                or self._raw_hand_command_max(side)
                <= _C09_GRASP_RELEASE_HAND_COMMAND
            ):
                self._preserve_released_hand_handoff(
                    side,
                    self._c09_paper_grasps[side],
                    palms.get(side),
                )
                self._c09_paper_grasps.pop(side, None)
        for side, palm in palms.items():
            if (
                palm is None
                or side in self._c09_paper_grasps
                or (
                    self._c09_stack_state is not None
                    and bool(self._c09_paper_grasps)
                )
                or self._raw_hand_command_max(side)
                < _C09_GRASP_ATTACH_HAND_COMMAND
            ):
                continue
            candidate = self._c09_paper_grasp_candidate(
                side,
                palm,
                self._wuji_fingertip_positions(side),
            )
            if candidate is None:
                continue
            name, paper, score, contact_count = candidate
            visual_hand = self._actual_hand_joint_pose(side)
            self._c09_paper_grasps[side] = {
                "name": name,
                "hand_to_object": _invert_pose_matrix(palm) @ paper,
                "last_target": paper.copy(),
                "hand_target_matrix": None,
                "hand_target_installed": False,
                "visual_hand": visual_hand.copy(),
                "score": score,
                "contact_count": int(contact_count),
            }

        current = {
            name: self._current_body_matrix(name)
            for name in _C09_PAPER_NAMES
        }
        if any(matrix is None for matrix in current.values()):
            return
        current_matrices = {
            name: np.asarray(matrix, dtype=np.float64).reshape(4, 4)
            for name, matrix in current.items()
        }

        # Normal carry is exact at the contact transform.  No velocity follower
        # is used, so a still hand cannot let a sheet drift or spin.
        if self._c09_stack_state is None:
            self._c09_clear_hand_targets()
            effective = dict(current_matrices)
            for side, state in self._c09_paper_grasps.items():
                palm = palms.get(side)
                if palm is None:
                    continue
                name = str(state["name"])
                target = palm @ np.asarray(
                    state["hand_to_object"],
                    dtype=np.float64,
                ).reshape(4, 4)
                applied, _linear, _angular = self._set_body_rigid_grasp_target(
                    name,
                    target,
                    previous_target=state.get("last_target"),
                )
                if applied:
                    state["last_target"] = target.copy()
                    effective[name] = target.copy()

            if not self._c09_paper_grasps:
                self._c09_papers_stacked = False
                return
            poses = {
                name: self._pose_matrix_to_vector(matrix)
                for name, matrix in effective.items()
            }
            overlap, z_error, alignment = self._c09_pair_capture_geometry(
                poses["c09_paper_0"],
                poses["c09_paper_1"],
            )
            if (
                overlap < _C09_STACK_CAPTURE_OVERLAP_RATIO
                or z_error > _C09_STACK_CAPTURE_Z_TOLERANCE
                or alignment < _C09_STACK_CAPTURE_ALIGNMENT
            ):
                return

            held = {
                str(state["name"]): side
                for side, state in self._c09_paper_grasps.items()
            }
            if len(held) == 1:
                guided_name, guided_side = next(iter(held.items()))
                anchor_name = self._c09_other_paper(guided_name)
            else:
                anchor_side = "right" if "right" in self._c09_paper_grasps else next(
                    iter(self._c09_paper_grasps)
                )
                anchor_name = str(self._c09_paper_grasps[anchor_side]["name"])
                guided_name = self._c09_other_paper(anchor_name)
                guided_side = held.get(guided_name)
            if guided_side is None:
                return
            self._c09_stack_state = {
                "phase": "pair_aligning",
                "anchor_name": anchor_name,
                "guided_name": guided_name,
                "guided_side": guided_side,
                "leader_name": guided_name,
                "leader_side": guided_side,
                "last_targets": {
                    name: matrix.copy()
                    for name, matrix in effective.items()
                },
                "stable_steps": 0,
            }
            # The first target is exactly the current contacted pose.
            self._c09_install_hand_target(
                guided_side,
                effective[guided_name],
            )
            return

        stack = self._c09_stack_state
        phase = str(stack.get("phase", "pair_aligning"))
        last_targets = {
            name: np.asarray(
                stack.get("last_targets", current_matrices).get(
                    name,
                    current_matrices[name],
                ),
                dtype=np.float64,
            ).reshape(4, 4).copy()
            for name in _C09_PAPER_NAMES
        }

        if phase == "pair_aligning":
            anchor_name = str(stack["anchor_name"])
            guided_name = str(stack["guided_name"])
            guided_side = str(stack["guided_side"])
            guided_state = self._c09_paper_grasps.get(guided_side)
            if (
                guided_state is None
                or str(guided_state.get("name")) != guided_name
            ):
                # The guide may only move a sheet while the same visible
                # pinch remains held.  Releasing it cancels the guide instead
                # of leaving a paper invisibly pinned in mid-air.
                self._c09_stack_state = None
                self._c09_papers_stacked = False
                self._c09_clear_hand_targets()
                return
            anchor_side = self._c09_grasp_side_for_paper(anchor_name)
            anchor_target = current_matrices[anchor_name].copy()
            if anchor_side is not None and palms.get(anchor_side) is not None:
                anchor_state = self._c09_paper_grasps[anchor_side]
                anchor_target = palms[anchor_side] @ np.asarray(
                    anchor_state["hand_to_object"],
                    dtype=np.float64,
                ).reshape(4, 4)
            goal = self._c09_paired_target(
                anchor_target,
                anchor_name=anchor_name,
                paired_name=guided_name,
            )
            guided_palm = palms.get(guided_side)
            if guided_palm is None:
                return
            guided_target = guided_palm @ np.asarray(
                guided_state["hand_to_object"],
                dtype=np.float64,
            ).reshape(4, 4)
            if self._c09_hand_tracking_ready(guided_side):
                command_target = self._bounded_pose_step(
                    guided_target,
                    goal,
                    dt=self._control_dt,
                    max_linear_speed=_C09_STACK_ALIGN_MAX_LINEAR_SPEED,
                    max_angular_speed=_C09_STACK_ALIGN_MAX_ANGULAR_SPEED,
                )
                self._c09_install_hand_target(
                    guided_side,
                    command_target,
                )
            targets = {
                anchor_name: anchor_target,
                guided_name: guided_target,
            }
            applied = self._c09_write_stack_targets(targets, last_targets)
            if not applied:
                return
            stack["last_targets"] = {
                name: targets[name].copy() for name in _C09_PAPER_NAMES
            }
            reached = self._tool_lock_target_reached(guided_target, goal)
            stack["stable_steps"] = (
                int(stack.get("stable_steps", 0)) + 1 if reached else 0
            )
            if int(stack["stable_steps"]) >= _C09_GUIDE_STABLE_STEPS:
                still_held = {
                    side: grasp
                    for side, grasp in self._c09_paper_grasps.items()
                    if str(grasp.get("name")) in _C09_PAPER_NAMES
                }
                stack.update(
                    {
                        "phase": (
                            "stack_wait_release"
                            if len(still_held) >= 2
                            else "stack_carry"
                        ),
                        "leader_name": guided_name,
                        "leader_side": guided_side,
                        "stable_steps": 0,
                    }
                )
                self._c09_papers_stacked = True
                if len(still_held) >= 2:
                    # Two closed hands cannot rigidly drive one stack along
                    # different trajectories.  Hold the just-aligned stack
                    # and both visible hands at their contacted poses until
                    # the operator opens either hand; the remaining hand then
                    # becomes the sole continuous leader.
                    for held_side, held_state in still_held.items():
                        held_name = str(held_state["name"])
                        self._c09_install_hand_target(
                            held_side,
                            targets[held_name],
                        )
                    return
                leader_state = self._c09_paper_grasps.get(guided_side)
                if leader_state is not None:
                    leader_hand_target = (
                        guided_target
                        @ _invert_pose_matrix(
                            np.asarray(
                                leader_state["hand_to_object"],
                                dtype=np.float64,
                            ).reshape(4, 4)
                        )
                    )
                    self._start_raw_hand_handoff(
                        guided_side,
                        leader_state,
                        leader_hand_target,
                    )
            return

        if phase == "stack_wait_release":
            still_held = {
                side: grasp
                for side, grasp in self._c09_paper_grasps.items()
                if str(grasp.get("name")) in _C09_PAPER_NAMES
            }
            if len(still_held) >= 2:
                self._c09_write_stack_targets(last_targets, last_targets)
                for held_side, held_state in still_held.items():
                    self._c09_install_hand_target(
                        held_side,
                        last_targets[str(held_state["name"])],
                    )
                return
            if not still_held:
                self._c09_stack_state = None
                self._c09_papers_stacked = False
                self._c09_clear_hand_targets()
                return
            leader_side, leader_state = next(iter(still_held.items()))
            leader_name = str(leader_state["name"])
            leader_palm = palms.get(leader_side)
            if leader_palm is None:
                return
            leader_state["hand_to_object"] = (
                _invert_pose_matrix(leader_palm)
                @ current_matrices[leader_name]
            )
            leader_state["last_target"] = current_matrices[
                leader_name
            ].copy()
            stack.update(
                {
                    "phase": "stack_carry",
                    "leader_name": leader_name,
                    "leader_side": leader_side,
                    "last_targets": {
                        name: current_matrices[name].copy()
                        for name in _C09_PAPER_NAMES
                    },
                }
            )
            hand_target = (
                current_matrices[leader_name]
                @ _invert_pose_matrix(
                    np.asarray(
                        leader_state["hand_to_object"],
                        dtype=np.float64,
                    ).reshape(4, 4)
                )
            )
            self._start_raw_hand_handoff(
                leader_side,
                leader_state,
                hand_target,
            )
            phase = "stack_carry"
            last_targets = {
                name: current_matrices[name].copy()
                for name in _C09_PAPER_NAMES
            }

        leader_name = str(stack.get("leader_name", "c09_paper_0"))
        leader_side = self._c09_grasp_side_for_paper(leader_name)
        if leader_side is None:
            # Deliberate transfer is rebased at the current physical pose; it
            # never switches through a hard-coded 4 mm offset.
            if self._c09_paper_grasps:
                leader_side, leader_grasp = next(
                    iter(self._c09_paper_grasps.items())
                )
                leader_name = str(leader_grasp["name"])
                leader_grasp["hand_to_object"] = (
                    _invert_pose_matrix(palms[leader_side])
                    @ current_matrices[leader_name]
                )
                leader_grasp["last_target"] = current_matrices[leader_name].copy()
                transfer_hand_target = (
                    current_matrices[leader_name]
                    @ _invert_pose_matrix(
                        np.asarray(
                            leader_grasp["hand_to_object"],
                            dtype=np.float64,
                        ).reshape(4, 4)
                    )
                )
                self._start_raw_hand_handoff(
                    leader_side,
                    leader_grasp,
                    transfer_hand_target,
                )
                stack["leader_side"] = leader_side
                stack["leader_name"] = leader_name
                last_targets = dict(current_matrices)
            elif phase != "inserted":
                # Never pin a released stack in mid-air.
                self._c09_stack_state = None
                self._c09_papers_stacked = False
                self._c09_clear_hand_targets()
                return
        follower_name = self._c09_other_paper(leader_name)

        if phase == "inserted":
            targets = last_targets
            self._c09_write_stack_targets(targets, last_targets)
            for name in _C09_PAPER_NAMES:
                side = self._c09_grasp_side_for_paper(name)
                self._c09_install_hand_target(side, targets[name])
            return

        if leader_side is None or palms.get(leader_side) is None:
            return
        leader_state = self._c09_paper_grasps[leader_side]
        carried_leader = palms[leader_side] @ np.asarray(
            leader_state["hand_to_object"],
            dtype=np.float64,
        ).reshape(4, 4)
        leader_target = carried_leader.copy()
        if phase == "stack_carry":
            handoff_target = self._raw_hand_handoff_target(
                leader_side,
                leader_state,
            )
            if handoff_target is not None:
                leader_state["hand_target_matrix"] = handoff_target
        stapler = self._current_body_matrix("c09_stapler")
        if phase == "stack_carry" and stapler is not None:
            if self._c09_stack_insertion_candidate(
                carried_leader,
                stapler,
            ):
                stack.update(
                    {
                        "phase": "insert_aligning",
                        "insert_goal": self._c09_insertion_goal(stapler),
                        "stable_steps": 0,
                    }
                )
                phase = "insert_aligning"
                # First guide command remains exactly where the user made the
                # valid visible approach.
                leader_target = last_targets[leader_name].copy()
        if phase == "insert_aligning":
            goal = np.asarray(
                stack["insert_goal"],
                dtype=np.float64,
            ).reshape(4, 4)
            leader_target = carried_leader.copy()
            if self._c09_hand_tracking_ready(leader_side):
                command_target = self._bounded_pose_step(
                    carried_leader,
                    goal,
                    dt=self._control_dt,
                    max_linear_speed=_C09_INSERT_MAX_LINEAR_SPEED,
                    max_angular_speed=_C09_INSERT_MAX_ANGULAR_SPEED,
                )
                self._c09_install_hand_target(
                    leader_side,
                    command_target,
                )

        follower_target = self._c09_paired_target(
            leader_target,
            anchor_name=leader_name,
            paired_name=follower_name,
        )
        targets = {
            leader_name: leader_target,
            follower_name: follower_target,
        }
        follower_side = self._c09_grasp_side_for_paper(follower_name)
        self._c09_install_hand_target(follower_side, follower_target)
        applied = self._c09_write_stack_targets(targets, last_targets)
        if not applied:
            return
        stack["last_targets"] = {
            name: targets[name].copy() for name in _C09_PAPER_NAMES
        }
        leader_state["last_target"] = leader_target.copy()
        if phase == "insert_aligning":
            reached = self._tool_lock_target_reached(
                leader_target,
                np.asarray(stack["insert_goal"], dtype=np.float64),
            )
            stack["stable_steps"] = (
                int(stack.get("stable_steps", 0)) + 1 if reached else 0
            )
            if int(stack["stable_steps"]) >= _C09_GUIDE_STABLE_STEPS:
                stack["phase"] = "inserted"

    def _b01_hand_command_max(self) -> float:
        return self._hand_command_max("right")

    def _b01_lid_in_release_zone(self) -> bool:
        if self.task_name != "t04_cover_pitcher":
            return False
        lid = self._pose_for_role("target")
        base = self._pose_for_role("receptacle")
        local, _target, _receptacle = self._target_local_to_receptacle()
        z_error = float(local[2] - _B01_LID_INSERTED_GOAL_LOCAL_Z)
        return bool(
            float(np.linalg.norm(local[:2]))
            <= _B01_RELEASE_ZONE_XY_TOLERANCE
            and -_B01_RELEASE_ZONE_Z_BELOW_TOLERANCE
            <= z_error
            <= _B01_RELEASE_ZONE_Z_ABOVE_TOLERANCE
            and self._z_axis_alignment(lid, base)
            >= float(np.cos(np.deg2rad(_B01_RELEASE_ZONE_TOP_ANGLE_DEG)))
        )

    def _set_b01_release_assist(self, enabled: bool) -> bool:
        """Filter lid/hand contacts only during an intentional near-goal release."""
        if self.task_name != "t04_cover_pitcher" or self._stage is None:
            self._b01_release_assist_active = False
            return False

        from pxr import UsdPhysics

        lid_prim = self._find_rigid_body_prim("pitcher_lid")
        robot_prim = self._stage.GetPrimAtPath(
            _B01_RIGHT_ROBOT_ARTICULATION_PATH
        )
        if lid_prim is None or not robot_prim.IsValid():
            if enabled:
                self._add_task_state_warning(
                    "Could not enable B01 release assist: lid or right-hand articulation is unavailable"
                )
            self._b01_release_assist_active = False
            return False

        filtered_pairs_api = UsdPhysics.FilteredPairsAPI(lid_prim)
        if enabled:
            if not filtered_pairs_api:
                filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(lid_prim)
            filtered_pairs_api.CreateFilteredPairsRel().AddTarget(
                robot_prim.GetPath()
            )
            if not self._b01_release_assist_active:
                print(
                    "[HandsOn/IsaacLab] B01 release assist enabled: "
                    "lid/right-hand contacts filtered until re-grasp or reset",
                    flush=True,
                )
            self._b01_release_assist_active = True
            return True

        if filtered_pairs_api:
            filtered_pairs_api.GetFilteredPairsRel().RemoveTarget(
                robot_prim.GetPath()
            )
        self._b01_release_assist_active = False
        return True

    def _update_b01_release_assist(self) -> None:
        if self.task_name != "t04_cover_pitcher":
            return
        hand_max = self._b01_hand_command_max()
        if self._b01_release_assist_active:
            if hand_max >= _B01_CLOSED_HAND_MIN_JOINT:
                self._set_b01_release_assist(False)
                self._b01_hand_was_closed = True
            return
        if hand_max >= _B01_CLOSED_HAND_MIN_JOINT:
            self._b01_hand_was_closed = True
            return
        if (
            not self._b01_hand_was_closed
            or hand_max > _B01_OPEN_HAND_MAX_JOINT
            or not self._b01_lid_in_release_zone()
        ):
            return
        self._set_b01_release_assist(True)

    def _b02_nut_grasp_assist_enabled(self) -> bool:
        # Disabled while testing source-matched B02 MuJoCo parameters.  Keeping
        # the helper state readable in ``info`` makes it obvious during
        # collection that no velocity-based nut attachment is active.
        return False

    def _clear_b02_nut_grasp_assist(self, reason: str = "") -> None:
        self._b02_grasp_assist = None
        self._b02_grasp_assist_metrics = {
            "b02_grasp_assist_enabled": bool(self._b02_nut_grasp_assist_enabled()),
            "b02_grasp_assist_active": False,
            "b02_grasp_assist_pair": "",
            "b02_grasp_assist_finger_distance": -1.0,
            "b02_grasp_assist_release_distance": float(_B02_GRASP_RELEASE_FINGER_DISTANCE),
            "b02_grasp_assist_reason": str(reason),
        }

    def _wuji_fingertip_positions(self, side: str) -> dict[str, np.ndarray]:
        """Return actual visible Wuji fingertip centers for either hand."""
        robot = self._robots.get(side)
        if robot is None:
            return {}

        offsets = (
            _B02_WUJI_FINGERTIP_LOCAL_OFFSETS
            if side == "right"
            else _B02_WUJI_LEFT_FINGERTIP_LOCAL_OFFSETS
        )
        positions: dict[str, np.ndarray] = {}
        for tip_name, (body_name, local_offset) in offsets.items():
            cache_key = f"{side}:{body_name}"
            body_index = self._b02_grasp_body_indices.get(cache_key)
            if body_index is None:
                try:
                    body_index = next(
                        index
                        for index, name in enumerate(robot.body_names)
                        if body_name in str(name)
                    )
                except (StopIteration, TypeError):
                    continue
                self._b02_grasp_body_indices[cache_key] = int(body_index)

            try:
                pose = robot.data.body_pose_w[0, int(body_index), :7]
                if hasattr(pose, "detach"):
                    pose = pose.detach().cpu().numpy()
                pose_np = np.asarray(pose, dtype=np.float64).reshape(7)
            except Exception:
                continue
            if not np.all(np.isfinite(pose_np)):
                continue

            matrix = _pose_vector_to_matrix(pose_np)
            tip_position = matrix[:3, 3] + matrix[:3, :3] @ local_offset
            if np.all(np.isfinite(tip_position)):
                positions[tip_name] = tip_position.astype(np.float64)
        return positions

    def _b02_wuji_fingertip_positions(self) -> dict[str, np.ndarray]:
        """Compatibility wrapper used by the existing right-hand tasks."""
        return self._wuji_fingertip_positions("right")

    @staticmethod
    def _b02_tip_touches_nut_underside(
        tip_position: np.ndarray,
        nut_position: np.ndarray,
    ) -> tuple[bool, float, float]:
        delta = np.asarray(tip_position, dtype=np.float64) - np.asarray(
            nut_position,
            dtype=np.float64,
        )
        horizontal_distance = float(np.linalg.norm(delta[:2]))
        vertical_delta = float(delta[2])
        touches = bool(
            horizontal_distance <= _B02_GRASP_ATTACH_HORIZONTAL_RADIUS
            and -_B02_GRASP_ATTACH_Z_BELOW_CENTER
            <= vertical_delta
            <= _B02_GRASP_ATTACH_Z_ABOVE_CENTER
        )
        return touches, horizontal_distance, vertical_delta

    def _b02_grasp_assist_candidate(
        self,
        nut_position: np.ndarray,
        tip_positions: dict[str, np.ndarray],
    ) -> dict[str, Any] | None:
        thumb_position = tip_positions.get(_B02_WUJI_THUMB_TIP)
        if thumb_position is None:
            return None

        candidates: list[tuple[float, dict[str, Any]]] = []
        for opposing_tip in _B02_WUJI_OPPOSING_TIPS:
            opposing_position = tip_positions.get(opposing_tip)
            if opposing_position is None:
                continue
            thumb_touches, thumb_horizontal, _thumb_vertical = (
                self._b02_tip_touches_nut_underside(
                    thumb_position,
                    nut_position,
                )
            )
            opposing_touches, opposing_horizontal, _opposing_vertical = (
                self._b02_tip_touches_nut_underside(
                    opposing_position,
                    nut_position,
                )
            )
            if not (thumb_touches and opposing_touches):
                continue

            finger_distance = float(
                np.linalg.norm(thumb_position - opposing_position)
            )
            if (
                finger_distance < _B02_GRASP_ATTACH_MIN_FINGER_DISTANCE
                or finger_distance > _B02_GRASP_ATTACH_MAX_FINGER_DISTANCE
            ):
                continue

            pinch_center = 0.5 * (thumb_position + opposing_position)
            pinch_center_error = float(
                np.linalg.norm(pinch_center[:2] - nut_position[:2])
            )
            if pinch_center_error > _B02_GRASP_ATTACH_PINCH_CENTER_RADIUS:
                continue

            target_distance = (
                _B02_NUT_OUTER_DIAMETER + 2.0 * _B02_WUJI_FINGERTIP_RADIUS
            )
            score = (
                abs(finger_distance - target_distance)
                + 0.5 * (thumb_horizontal + opposing_horizontal)
                + pinch_center_error
            )
            candidates.append(
                (
                    score,
                    {
                        "pair": (_B02_WUJI_THUMB_TIP, opposing_tip),
                        "finger_distance": finger_distance,
                        "pinch_center": pinch_center,
                        "pinch_center_error": pinch_center_error,
                    },
                )
            )

        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _set_b02_nut_grasp_velocity(self, target_position: np.ndarray) -> bool:
        view = self._task_body_views.get(self._target_object_name)
        if view is None:
            view = self._task_body_views.get("nut")
        if view is None or self._torch is None:
            return False

        try:
            transforms = view.get_transforms()
            velocities = view.get_velocities().clone()
            current_position = transforms[0, :3]
            target = self._torch.as_tensor(
                np.asarray(target_position, dtype=np.float64).reshape(3),
                dtype=current_position.dtype,
                device=current_position.device,
            )
            linear_velocity = (target - current_position) / max(
                float(self._sim_dt),
                1e-6,
            )
            speed = self._torch.linalg.vector_norm(linear_velocity)
            if float(speed.item()) > _B02_GRASP_ASSIST_MAX_LINEAR_SPEED:
                linear_velocity = (
                    linear_velocity
                    * (_B02_GRASP_ASSIST_MAX_LINEAR_SPEED / speed)
                )
            velocities[0, :3] = linear_velocity
            indices = self._torch.arange(
                int(view.count),
                dtype=self._torch.long,
                device=velocities.device,
            )
            view.set_velocities(velocities, indices)
            return True
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not apply B02 nut grasp assist velocity: {exc}"
            )
            return False

    def _update_b02_nut_grasp_assist(self) -> None:
        if not self._b02_nut_grasp_assist_enabled():
            if self._b02_grasp_assist is not None:
                self._clear_b02_nut_grasp_assist("disabled")
            return

        nut_pose = self._physics_body_pose(self._target_object_name)
        if nut_pose is None:
            nut_pose = self._physics_body_pose("nut")
        if nut_pose is None:
            nut_pose = self._pose_for_role("target")
        nut_position = np.asarray(nut_pose[:3], dtype=np.float64)
        tip_positions = self._b02_wuji_fingertip_positions()

        state = self._b02_grasp_assist
        if state is not None:
            pair = tuple(state.get("pair", ()))
            if len(pair) != 2 or pair[0] not in tip_positions or pair[1] not in tip_positions:
                self._clear_b02_nut_grasp_assist("tip_missing")
                return

            first = tip_positions[str(pair[0])]
            second = tip_positions[str(pair[1])]
            finger_distance = float(np.linalg.norm(first - second))
            if finger_distance > _B02_GRASP_RELEASE_FINGER_DISTANCE:
                self._clear_b02_nut_grasp_assist("finger_open")
                self._b02_grasp_assist_metrics["b02_grasp_assist_finger_distance"] = finger_distance
                return

            pinch_center = 0.5 * (first + second)
            offset = np.asarray(state["offset"], dtype=np.float64).reshape(3)
            target_position = pinch_center + offset
            velocity_applied = self._set_b02_nut_grasp_velocity(target_position)
            self._b02_grasp_assist_metrics = {
                "b02_grasp_assist_enabled": True,
                "b02_grasp_assist_active": True,
                "b02_grasp_assist_pair": f"{pair[0]}/{pair[1]}",
                "b02_grasp_assist_finger_distance": finger_distance,
                "b02_grasp_assist_release_distance": float(_B02_GRASP_RELEASE_FINGER_DISTANCE),
                "b02_grasp_assist_target_error": float(
                    np.linalg.norm(target_position - nut_position)
                ),
                "b02_grasp_assist_velocity_applied": bool(velocity_applied),
                "b02_grasp_assist_reason": "active",
            }
            return

        candidate = self._b02_grasp_assist_candidate(nut_position, tip_positions)
        if candidate is None:
            self._b02_grasp_assist_metrics = {
                "b02_grasp_assist_enabled": True,
                "b02_grasp_assist_active": False,
                "b02_grasp_assist_pair": "",
                "b02_grasp_assist_finger_distance": -1.0,
                "b02_grasp_assist_release_distance": float(_B02_GRASP_RELEASE_FINGER_DISTANCE),
                "b02_grasp_assist_reason": "waiting_for_two_underside_tips",
            }
            return

        pinch_center = np.asarray(candidate["pinch_center"], dtype=np.float64)
        offset = nut_position - pinch_center
        self._b02_grasp_assist = {
            "pair": tuple(candidate["pair"]),
            "offset": offset,
        }
        self._b02_grasp_assist_metrics = {
            "b02_grasp_assist_enabled": True,
            "b02_grasp_assist_active": True,
            "b02_grasp_assist_pair": "/".join(candidate["pair"]),
            "b02_grasp_assist_finger_distance": float(
                candidate["finger_distance"]
            ),
            "b02_grasp_assist_release_distance": float(_B02_GRASP_RELEASE_FINGER_DISTANCE),
            "b02_grasp_assist_target_error": 0.0,
            "b02_grasp_assist_velocity_applied": False,
            "b02_grasp_assist_reason": "attached",
        }

    def _update_recording_cameras(self) -> None:
        if not self._cameras or self._torch is None:
            return
        try:
            if "head_camera" in self._cameras:
                head_position = self._torch.tensor(
                    [[*_HEAD_RECORDING_CAMERA_EYE]],
                    dtype=self._torch.float32,
                    device=self._isaac_device,
                )
                head_orientation = self._torch.tensor(
                    [[*_HEAD_RECORDING_CAMERA_QUAT_WXYZ]],
                    dtype=self._torch.float32,
                    device=self._isaac_device,
                )
                self._cameras["head_camera"].set_world_poses(
                    head_position,
                    head_orientation,
                    convention="opengl",
                )

            wrist_sources = (
                ("wrist_left_camera", "left"),
                ("wrist_right_camera", "right"),
            )
            for name, source_side in wrist_sources:
                if name not in self._dynamic_cameras or name not in self._cameras:
                    continue
                mount_poses = (
                    _LEGACY_WRIST_RECORDING_CAMERA_MOUNT_POSES
                    if self._legacy_wrist_cameras
                    else _WRIST_RECORDING_CAMERA_MOUNT_POSES
                )
                camera_pose = None
                if self._legacy_wrist_cameras:
                    camera_pose = self._native_robot_site_pose_matrix(
                        source_side,
                        f"{source_side}_wrist_fisheye_optical_site",
                    )
                fallback = (
                    self._left_tcp_pose
                    if source_side == "left"
                    else self._right_tcp_pose
                )
                if camera_pose is None:
                    mount_pose = self._native_robot_body_pose_matrix(
                        source_side,
                        f"{source_side}_wrist_fisheye_mount",
                    )
                    if mount_pose is not None:
                        camera_pose = (
                            mount_pose
                            @ _pose_vector_to_matrix(
                                mount_poses[source_side]
                            )
                        )
                    else:
                        palm_pose = self._native_end_effector_pose_matrix(
                            source_side,
                            fallback,
                        )
                        camera_pose = (
                            palm_pose
                            @ _pose_vector_to_matrix(
                                _WRIST_RECORDING_CAMERA_LOCAL_POSES[source_side]
                            )
                        )
                position = self._torch.tensor(
                    [[*camera_pose[:3, 3]]],
                    dtype=self._torch.float32,
                    device=self._isaac_device,
                )
                orientation = self._torch.tensor(
                    [[*_matrix_to_quat_wxyz(camera_pose[:3, :3])]],
                    dtype=self._torch.float32,
                    device=self._isaac_device,
                )
                self._cameras[name].set_world_poses(
                    position,
                    orientation,
                    convention="opengl",
                )
        except Exception as exc:
            message = f"Recording camera pose update failed: {exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)

    def _set_wrist_obstructions_hidden(self, hidden: bool) -> None:
        if self._stage is None:
            return
        if self._wrist_obstruction_visibility is None:
            self._wrist_obstruction_visibility = (
                wrist_obstruction_visibility_records(self._stage)
            )
            print(
                "[HandsOn/IsaacLab] Wrist-only camera obstruction filter: "
                f"{len(self._wrist_obstruction_visibility)} visual prims",
                flush=True,
            )
        set_wrist_obstructions_hidden(
            self._stage,
            self._wrist_obstruction_visibility,
            hidden,
        )

    def _deferred_robot_body_visibility_records(self) -> tuple[tuple[str, Any], ...]:
        if self._stage is None:
            return ()
        from pxr import Usd, UsdGeom

        records: list[tuple[str, Any]] = []
        seen_paths: set[str] = set()
        for root_path in _DEFERRED_ROBOT_VISUAL_ROOTS:
            root = self._stage.GetPrimAtPath(root_path)
            if not root.IsValid():
                continue
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                path = str(prim.GetPath())
                if _deferred_keeps_hand_visual_path(path):
                    continue
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                target = prim
                while target.IsInstanceProxy():
                    target = target.GetParent()
                target_path = str(target.GetPath())
                imageable = UsdGeom.Imageable(target)
                if not imageable:
                    continue
                visibility = imageable.GetVisibilityAttr().Get()
                if target_path not in seen_paths:
                    records.append((target_path, visibility or UsdGeom.Tokens.inherited))
                    seen_paths.add(target_path)
        return tuple(records)

    def _set_deferred_robot_body_hidden(self, hidden: bool) -> int:
        if self._stage is None:
            return 0
        from pxr import UsdGeom

        if self._deferred_robot_body_visibility is None:
            self._deferred_robot_body_visibility = (
                self._deferred_robot_body_visibility_records()
            )
            print(
                "[HandsOn/IsaacLab] Deferred arm render body filter: "
                f"{len(self._deferred_robot_body_visibility)} robot visual prims",
                flush=True,
            )
        for path, original in self._deferred_robot_body_visibility:
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            visibility = UsdGeom.Imageable(prim).GetVisibilityAttr()
            visibility.Set(UsdGeom.Tokens.invisible if hidden else original)
        return len(self._deferred_robot_body_visibility)

    def _task_extra_pose_names(self) -> set[str]:
        extras: set[str] = set()
        if self.task_name == "t36_scoop_marbles":
            extras.add("spoon")
        elif self.task_name == "c07_pitcher_pour_cup":
            extras.add("c07_return_zone")
        elif self.task_name == "c09_staple_paper_sheets":
            extras.add("c09_stapler")
        elif self.task_name == "c11_xylophone_sequence":
            extras.update(
                {
                    "c11_xylophone",
                    "c11_xylophone_frame",
                    "c11_return_zone",
                    *_C11_KEY_NAMES,
                }
            )
        return extras

    def _refresh_object_poses(self) -> None:
        extras = {"wood", "table"}
        extras.update(self._task_extra_pose_names())
        for name in set(self._spec.target_objects) | {self._spec.receptacle} | extras:
            self._poses[name] = self._prim_pose(name, self._fallback_pose(name))
        if self.task_name == "t14_move_and_click_mouse" and self._c05_capture_start_pending:
            self._c05_mouse_start_xy = self._pose_for_role("target")[:2].copy()
            for name in ("mouse_joint_0", "mouse_joint_1"):
                reading = self._physics_dof_position(name)
                self._c05_button_initial_positions[name] = (
                    float(reading[0]) if reading is not None else 0.0
                )
            self._c05_capture_start_pending = False

    def _fallback_pose(self, name: str) -> np.ndarray:
        if name == "table":
            return np.asarray([-0.15, 0.0, 0.0 + float(self.delta_h[0]), 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        x = -0.22 + 0.08 * (abs(hash(name)) % 5)
        y = -0.20 + 0.08 * (abs(hash(name[::-1])) % 5)
        return np.asarray([x, y, _TABLE_TOP_Z + 0.04 + float(self.delta_h[0]), 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _pose_for_role(self, role: str) -> np.ndarray:
        if role == "target":
            return self._poses.get(self._target_object_name, self._fallback_pose(self._target_object_name))
        if role == "receptacle":
            return self._poses.get(self._spec.receptacle, self._fallback_pose(self._spec.receptacle))
        return self._poses.get(role, self._fallback_pose(role))

    def _prim_pose(self, name: str, fallback: np.ndarray) -> np.ndarray:
        physics_pose = self._physics_body_pose(name)
        if physics_pose is not None:
            return physics_pose
        matrix = self._prim_pose_matrix(name, None)
        if matrix is None:
            return np.asarray(fallback, dtype=np.float64).reshape(7).copy()
        return np.concatenate([matrix[:3, 3], _matrix_to_quat_wxyz(matrix[:3, :3])]).astype(np.float64)

    def _physics_body_pose(self, name: str) -> np.ndarray | None:
        view = self._task_body_views.get(name)
        if view is None:
            return None
        try:
            transform = view.get_transforms()[0]
            if hasattr(transform, "detach"):
                transform = transform.detach().cpu().numpy()
            transform = np.asarray(transform, dtype=np.float64).reshape(7)
            # PhysX tensor transforms use xyzw quaternion ordering.
            return np.concatenate(
                [transform[:3], transform[6:7], transform[3:6]]
            ).astype(np.float64)
        except Exception:
            return None

    def _current_table_top_z(self) -> float:
        table_pose = self._physics_body_pose("table")
        if table_pose is None:
            table_pose = self._poses.get("table")
        if table_pose is not None and np.isfinite(table_pose[2]):
            return float(table_pose[2]) + _TABLE_TOP_Z
        return _TABLE_TOP_Z + float(self.delta_h[0])

    def _physics_body_velocity(self, name: str) -> np.ndarray | None:
        view = self._task_body_views.get(name)
        if view is None:
            return None
        try:
            velocity = view.get_velocities()[0]
            if hasattr(velocity, "detach"):
                velocity = velocity.detach().cpu().numpy()
            return np.asarray(velocity, dtype=np.float64).reshape(6)
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
            positions = view.get_dof_positions()
            value = positions[0, int(index)]
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            return float(value), float(initial)
        except Exception:
            return None

    def _prim_pose_matrix(self, name: str, fallback: np.ndarray | None) -> np.ndarray | None:
        if self._stage is None:
            return fallback
        try:
            from pxr import Usd, UsdGeom

            prim = self._find_prim(name)
            if prim is None or not prim.IsValid():
                return fallback
            transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            return _usd_transform_to_pose_matrix(transform)
        except Exception:
            return fallback

    def _find_prim(self, name: str):
        if name in self._prim_cache:
            return self._prim_cache[name]
        if self._stage is None:
            return None
        from pxr import Usd, UsdGeom, UsdPhysics

        root = self._stage.GetPrimAtPath(_TASK_ROOT)
        scene_name = _TASK_SCENE_BODY_NAMES.get(self.task_name, {}).get(name, name)
        normalized = scene_name.lower()
        exact_xform = None
        fuzzy_rigid = None
        fuzzy_xform = None
        for prim in Usd.PrimRange(root):
            prim_name = prim.GetName()
            if not prim.IsA(UsdGeom.Xformable):
                continue
            is_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            if prim_name == scene_name and is_rigid:
                self._prim_cache[name] = prim
                return prim
            if prim_name == scene_name and exact_xform is None:
                exact_xform = prim
            if normalized in prim_name.lower():
                if is_rigid and fuzzy_rigid is None:
                    fuzzy_rigid = prim
                elif fuzzy_xform is None:
                    fuzzy_xform = prim
        best = fuzzy_rigid or exact_xform or fuzzy_xform
        self._prim_cache[name] = best
        return best

    def _compute_success(self) -> bool:
        success_now = self._compute_success_now()
        if success_now:
            self._success_counter += 1
        else:
            self._success_counter = 0
        success = self._success_counter >= self._success_steps_required()
        self._update_success_condition_overlay()
        return success

    def _compute_failure(self) -> bool:
        return bool(
            (
                self.task_name == "t50_water_plant"
                and self._c04_failed_trigger_outside
            )
            or (
                self.task_name == "t14_move_and_click_mouse"
                and self._c05_failed_early_click
            )
        )

    @staticmethod
    def _c04_spray_sample_directions(forward: np.ndarray) -> list[np.ndarray]:
        forward = np.asarray(forward, dtype=np.float64).reshape(3)
        world_up = _WORLD_UP.copy()
        if abs(float(np.dot(forward, world_up))) > 0.90:
            world_up = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1e-8:
            return [forward.copy()]
        right /= right_norm
        up = np.cross(right, forward)
        up_norm = float(np.linalg.norm(up))
        if up_norm < 1e-8:
            return [forward.copy()]
        up /= up_norm

        directions: list[np.ndarray] = []
        for offset_x, offset_y in _C04_WATER_SAMPLE_OFFSETS:
            direction = forward + _C04_WATER_CONE_SPREAD * (
                offset_x * right + offset_y * up
            )
            norm = float(np.linalg.norm(direction))
            if norm >= 1e-8:
                directions.append(direction / norm)
        return directions or [forward.copy()]

    @staticmethod
    def _c04_spray_ray_hit_metrics(
        nozzle_pos: np.ndarray,
        direction: np.ndarray,
        plant_body_pos: np.ndarray,
    ) -> tuple[bool, float, float]:
        nozzle_pos = np.asarray(nozzle_pos, dtype=np.float64).reshape(3)
        direction = np.asarray(direction, dtype=np.float64).reshape(3)
        plant_body_pos = np.asarray(plant_body_pos, dtype=np.float64).reshape(3)

        soil_z = float(plant_body_pos[2] + 0.02)
        soil_hit_radius = math.inf
        soil_hit = False
        if float(direction[2]) < -1e-6:
            distance_to_soil = (soil_z - float(nozzle_pos[2])) / float(direction[2])
            if distance_to_soil > 0.0:
                soil_hit_pos = nozzle_pos + distance_to_soil * direction
                soil_hit_radius = float(
                    np.linalg.norm(soil_hit_pos[:2] - plant_body_pos[:2])
                )
                soil_hit = soil_hit_radius <= _C04_SOIL_TARGET_RADIUS

        plant_projection = float(np.dot(plant_body_pos - nozzle_pos, direction))
        plant_closest = nozzle_pos + plant_projection * direction
        plant_ray_error = float(np.linalg.norm(plant_body_pos - plant_closest))
        plant_hit = bool(
            plant_projection > 0.0
            and plant_ray_error <= _C04_PLANT_RAY_RADIUS
        )
        return bool(soil_hit or plant_hit), soil_hit_radius, plant_ray_error

    def _c04_spray_aim_metrics(self) -> dict[str, float | bool | int]:
        spray_pose = self._pose_for_role("target")
        plant_pose = self._pose_for_role("receptacle")
        spray_matrix = _pose_vector_to_matrix(spray_pose)
        nozzle_pos = (
            spray_matrix[:3, 3]
            + spray_matrix[:3, :3] @ _C04_NOZZLE_LOCAL
        )
        forward = spray_matrix[:3, :3] @ np.asarray(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 1e-8:
            forward = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            forward /= forward_norm

        plant_body_pos = np.asarray(plant_pose[:3], dtype=np.float64)
        plant_target = plant_body_pos + _C04_PLANT_TARGET_OFFSET
        to_target = plant_target - nozzle_pos
        distance = float(np.linalg.norm(to_target))
        target_direction = forward if distance < 1e-8 else to_target / distance
        projection = float(np.dot(to_target, forward))
        closest = nozzle_pos + projection * forward
        ray_error = float(np.linalg.norm(plant_target - closest))
        aim_cos = float(np.clip(np.dot(forward, target_direction), -1.0, 1.0))
        geometric_aimed = bool(
            _C04_AIM_MIN_DISTANCE <= distance <= _C04_AIM_MAX_DISTANCE
            and projection > 0.0
            and ray_error <= _C04_AIM_RAY_RADIUS
            and aim_cos >= _C04_AIM_MAX_ANGLE_COS
        )

        hit_count = 0
        soil_hit_radius = math.inf
        plant_ray_error = math.inf
        directions = self._c04_spray_sample_directions(forward)
        for direction in directions:
            hit, sample_soil_radius, sample_plant_error = (
                self._c04_spray_ray_hit_metrics(
                    nozzle_pos,
                    direction,
                    plant_body_pos,
                )
            )
            if hit:
                hit_count += 1
            soil_hit_radius = min(soil_hit_radius, sample_soil_radius)
            plant_ray_error = min(plant_ray_error, sample_plant_error)
        sample_count = len(directions)
        hit_fraction = float(hit_count / max(1, sample_count))
        target_hit = hit_fraction >= _C04_WATER_HIT_FRACTION_THRESHOLD
        return {
            "aimed": bool(target_hit),
            "geometric_aimed": geometric_aimed,
            "spray_downward": bool(float(forward[2]) <= _C04_WATER_DOWNWARD_Z),
            "spray_hits_soil_or_plant": bool(target_hit),
            "water_hit_fraction": hit_fraction,
            "water_hit_count": int(hit_count),
            "water_sample_count": int(sample_count),
            "spray_forward_z": float(forward[2]),
            "spray_distance": distance,
            "spray_projection": projection,
            "spray_ray_error": ray_error,
            "spray_aim_cos": aim_cos,
            "soil_hit_radius": soil_hit_radius,
            "plant_ray_error": plant_ray_error,
        }

    def _t50_water_plant_success_now(self) -> bool:
        aim_metrics = self._c04_spray_aim_metrics()
        aimed = bool(aim_metrics["aimed"])
        trigger_force = self._c04_trigger_contact_force()
        grasp_force = self._c04_grasp_contact_force()
        spraying = bool(
            self._c04_trigger_pulled
            and self._c04_trigger_cycle_latched
            and self._c04_water_emitting
        )
        valid_spray = bool(spraying and aimed)
        self._c04_failed_trigger_outside = bool(spraying and not aimed)
        self._last_c04_success_metrics = {
            **aim_metrics,
            "spray_grasp_ok": bool(self._c04_hand_gripping_spray()),
            "spray_trigger_contact_ok": bool(
                trigger_force >= _C04_TRIGGER_CONTACT_FORCE
                or self._c04_trigger_contact_grace > 0
            ),
            "spray_state_ok": spraying,
            "spray_valid": valid_spray,
            "c04_water_emitting": spraying,
            "c04_trigger_phase": self._c04_trigger_phase,
            "c04_trigger_position": float(
                (self._physics_dof_position("joint_0") or (0.0, 0.0))[0]
            ),
            "c04_trigger_contact_force": trigger_force,
            "c04_grasp_contact_force": grasp_force,
            "c04_trigger_cycle_latched": bool(self._c04_trigger_cycle_latched),
            "c04_trigger_armed": bool(self._c04_trigger_armed),
            "c04_grasp_ready_steps": int(self._c04_grasp_ready_steps),
            "c04_trigger_contact_seen": bool(self._c04_trigger_contact_seen),
            "c04_spray_grasp_seen": bool(self._c04_grasp_seen),
            "c04_failed_trigger_outside": bool(self._c04_failed_trigger_outside),
        }
        return valid_spray

    def _c05_button_travel(self, name: str) -> float:
        reading = self._physics_dof_position(name)
        if reading is None:
            return 0.0
        baseline = float(self._c05_button_initial_positions.get(name, 0.0))
        return max(0.0, float(reading[0]) - baseline)

    def _c05_mouse_click_metrics(self, *, update_failure: bool) -> dict[str, Any]:
        mouse = self._pose_for_role("target")
        mousepad = self._pose_for_role("receptacle")
        pad_distance = float(np.linalg.norm(mouse[:2] - mousepad[:2]))
        pad_height_error = abs(float(mouse[2] - mousepad[2]))
        mouse_on_pad = bool(
            pad_distance <= _C05_MOUSEPAD_RADIUS
            and pad_height_error <= _C05_MOUSEPAD_Z_TOLERANCE
        )
        start_xy = (
            mouse[:2]
            if self._c05_mouse_start_xy is None
            else self._c05_mouse_start_xy
        )
        move_distance = float(np.linalg.norm(mouse[:2] - start_xy))
        mouse_moved = move_distance >= _C05_MIN_MOUSE_MOVE_DISTANCE
        left_travel = self._c05_button_travel("mouse_joint_0")
        right_travel = self._c05_button_travel("mouse_joint_1")
        left_pressed = left_travel >= _C05_BUTTON_PRESS_THRESHOLD
        right_pressed = right_travel >= _C05_BUTTON_PRESS_THRESHOLD
        ready_for_click = bool(mouse_moved and mouse_on_pad)
        early_click_now = bool(
            (left_pressed or right_pressed) and not ready_for_click
        )
        if update_failure and early_click_now:
            self._c05_failed_early_click = True
        valid_left_click = bool(
            ready_for_click
            and left_pressed
            and not right_pressed
            and not self._c05_failed_early_click
        )
        return {
            "c05_mouse_moved": mouse_moved,
            "c05_mouse_on_pad": mouse_on_pad,
            "c05_no_early_click": not self._c05_failed_early_click,
            "c05_left_clicked": valid_left_click,
            "c05_ready_for_click": ready_for_click,
            "c05_early_click_now": early_click_now,
            "c05_failed_early_click": bool(self._c05_failed_early_click),
            "c05_mouse_move_distance": move_distance,
            "c05_mousepad_distance": pad_distance,
            "c05_mousepad_height_error": pad_height_error,
            "c05_left_button_travel": left_travel,
            "c05_right_button_travel": right_travel,
        }

    def _t14_move_and_click_mouse_success_now(self) -> bool:
        self._last_c05_success_metrics = self._c05_mouse_click_metrics(
            update_failure=True
        )
        return bool(self._last_c05_success_metrics["c05_left_clicked"])

    def _c06_photograph_metrics(self) -> dict[str, Any]:
        reading = self._physics_dof_position(_C06_SHUTTER_JOINT)
        if reading is None:
            shutter_travel = 0.0
            shutter_available = False
        else:
            current, initial = reading
            shutter_travel = max(0.0, float(current - initial))
            shutter_available = True

        was_down = bool(self._c06_shutter_was_down)
        if was_down:
            shutter_down = shutter_travel > _C06_SHUTTER_RELEASE_TRAVEL
        else:
            shutter_down = shutter_travel >= _C06_SHUTTER_PRESS_TRAVEL
        shutter_press_edge = bool(shutter_down and not was_down)
        self._c06_shutter_was_down = shutter_down

        right_contact_force = self._c06_right_hand_shutter_contact_force()
        right_contact_now = bool(
            right_contact_force >= _C06_SHUTTER_RIGHT_CONTACT_FORCE
        )
        if right_contact_now:
            self._c06_right_contact_grace = _C06_SHUTTER_CONTACT_GRACE_STEPS
        else:
            self._c06_right_contact_grace = max(
                0,
                self._c06_right_contact_grace - 1,
            )
        right_hand_contact = bool(
            right_contact_now or self._c06_right_contact_grace > 0
        )

        camera_pose = self._pose_for_role("target")
        table_pose = self._poses.get("table", self._fallback_pose("table"))
        camera_matrix = _pose_vector_to_matrix(camera_pose)
        table_matrix = _pose_vector_to_matrix(table_pose)
        camera_position = camera_matrix[:3, 3]
        logo_position = (
            table_matrix
            @ np.asarray([*_C06_LOGO_LOCAL, 1.0], dtype=np.float64)
        )[:3]

        view_direction = camera_matrix[:3, :3] @ _C06_CAMERA_VIEW_LOCAL
        view_norm = float(np.linalg.norm(view_direction))
        if view_norm > 1e-8:
            view_direction /= view_norm
        to_logo = logo_position - camera_position
        logo_distance = float(np.linalg.norm(to_logo))
        if view_norm <= 1e-8 or logo_distance <= 1e-8:
            view_angle_deg = 180.0
            view_projection = -1.0
            view_ray_error = math.inf
        else:
            to_logo_direction = to_logo / logo_distance
            view_cos = float(
                np.clip(np.dot(view_direction, to_logo_direction), -1.0, 1.0)
            )
            view_angle_deg = float(np.degrees(np.arccos(view_cos)))
            view_projection = float(np.dot(to_logo, view_direction))
            closest = camera_position + view_projection * view_direction
            view_ray_error = float(np.linalg.norm(logo_position - closest))

        aimed = bool(
            view_projection > 0.0
            and view_angle_deg <= _C06_AIM_MAX_ANGLE_DEG
            and view_ray_error <= _C06_AIM_MAX_RAY_ERROR
        )
        return {
            "c06_camera_aimed": aimed,
            "c06_right_hand_contact": right_hand_contact,
            "c06_shutter_pressed": shutter_down,
            "c06_shutter_press_edge": shutter_press_edge,
            "c06_shutter_available": shutter_available,
            "c06_shutter_travel": shutter_travel,
            "c06_right_hand_contact_force": right_contact_force,
            "c06_right_contact_grace_steps": int(self._c06_right_contact_grace),
            "c06_view_angle_deg": view_angle_deg,
            "c06_view_projection": view_projection,
            "c06_view_ray_error": view_ray_error,
            "c06_logo_distance": logo_distance,
        }

    def _t23_photograph_logo_success_now(self) -> bool:
        self._last_c06_success_metrics = self._c06_photograph_metrics()
        return bool(
            self._last_c06_success_metrics["c06_camera_aimed"]
            and self._last_c06_success_metrics["c06_right_hand_contact"]
            and self._last_c06_success_metrics["c06_shutter_pressed"]
        )

    def _local_point_in_body_frame(
        self,
        body_name: str,
        point_world: np.ndarray,
    ) -> np.ndarray:
        body_matrix = self._pose_matrix_for_name(body_name)
        point = np.asarray([*np.asarray(point_world, dtype=np.float64).reshape(3), 1.0])
        return (_invert_pose_matrix(body_matrix) @ point)[:3]

    def _body_local_point_world(self, body_name: str, local_point: np.ndarray) -> np.ndarray:
        matrix = self._pose_matrix_for_name(body_name)
        point = np.asarray([*np.asarray(local_point, dtype=np.float64).reshape(3), 1.0])
        return (matrix @ point)[:3]

    @staticmethod
    def _axis_alignment(pose: np.ndarray, axis_index: int, reference: np.ndarray) -> float:
        rot = _quat_wxyz_to_matrix(np.asarray(pose, dtype=np.float64)[3:7])
        axis = rot[:, int(axis_index)]
        reference_np = np.asarray(reference, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(reference_np))
        if norm < 1e-8:
            return 0.0
        return float(np.dot(axis, reference_np / norm))

    @staticmethod
    def _point_segment_distance_3d(
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> float:
        point_np = np.asarray(point, dtype=np.float64).reshape(3)
        start_np = np.asarray(start, dtype=np.float64).reshape(3)
        segment = np.asarray(end, dtype=np.float64).reshape(3) - start_np
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            return float(np.linalg.norm(point_np - start_np))
        fraction = float(
            np.clip(np.dot(point_np - start_np, segment) / length_sq, 0.0, 1.0)
        )
        return float(np.linalg.norm(point_np - (start_np + fraction * segment)))

    def _c07_hand_clearance(self) -> tuple[float, bool]:
        min_surface_distance = math.inf
        for point_world in self._a06_hand_points("right"):
            local = self._local_point_in_body_frame("c07_pitcher", point_world)
            radial = float(np.linalg.norm(local[:2]))
            side_z = float(
                np.clip(
                    local[2],
                    _C07_PITCHER_BOTTOM_Z,
                    _C07_PITCHER_TOP_Z,
                )
            )
            side_distance = float(
                math.hypot(
                    abs(radial - _C07_PITCHER_BODY_RADIUS),
                    float(local[2]) - side_z,
                )
            )
            bottom_radial_delta = max(
                0.0,
                radial - (_C07_PITCHER_BODY_RADIUS - 0.004),
            )
            bottom_distance = float(
                math.hypot(
                    bottom_radial_delta,
                    float(local[2]) - _C07_PITCHER_BOTTOM_Z,
                )
            )
            min_surface_distance = min(
                min_surface_distance,
                side_distance,
                bottom_distance,
            )
            for segment in _C07_HANDLE_FRAME_SEGMENTS:
                handle_distance = max(
                    0.0,
                    self._point_segment_distance_3d(
                        local,
                        segment[0],
                        segment[1],
                    )
                    - _C07_HANDLE_FRAME_RADIUS,
                )
                min_surface_distance = min(
                    min_surface_distance,
                    handle_distance,
                )
        return min_surface_distance, bool(
            min_surface_distance >= _C07_HAND_RELEASE_CLEARANCE
        )

    def _c07_pitcher_pour_metrics(self) -> dict[str, Any]:
        cup = self._poses.get("c07_cup", self._fallback_pose("c07_cup"))
        pitcher = self._poses.get("c07_pitcher", self._fallback_pose("c07_pitcher"))
        delivered_particles = getattr(self, "_c07_delivered_particles", set())
        inside_count = 0
        pitcher_count = 0
        spilled_count = 0
        for name in _C07_PARTICLE_NAMES:
            particle = self._poses.get(name, self._fallback_pose(name))
            local = self._local_point_in_body_frame("c07_cup", particle[:3])
            inside = bool(
                float(np.linalg.norm(local[:2])) <= _C07_CUP_INNER_RADIUS
                and _C07_CUP_LOCAL_Z_RANGE[0] <= float(local[2]) <= _C07_CUP_LOCAL_Z_RANGE[1]
            )
            if (
                inside
                and self._c07_pitcher_grasp is not None
                and bool(self._c07_pitcher_grasp.get("pour_lock_seen"))
            ):
                delivered_particles.add(name)
            pitcher_local = self._local_point_in_body_frame(
                "c07_pitcher",
                particle[:3],
            )
            inside_pitcher = bool(
                float(np.linalg.norm(pitcher_local[:2]))
                <= _C07_PITCHER_INNER_RADIUS
                and _C07_PITCHER_LOCAL_Z_RANGE[0]
                <= float(pitcher_local[2])
                <= _C07_PITCHER_LOCAL_Z_RANGE[1]
            )
            if name in delivered_particles or inside:
                inside_count += 1
            elif inside_pitcher:
                pitcher_count += 1
            elif float(particle[2]) <= self._current_table_top_z() + 0.018:
                spilled_count += 1
        upright_alignment = self._z_axis_alignment(pitcher)
        if upright_alignment <= _C07_PITCHER_TILT_ALIGNMENT:
            self._c07_tilt_seen = True
        return_local = self._local_point_in_body_frame(
            "c07_return_zone",
            pitcher[:3],
        )
        return_distance = float(np.linalg.norm(return_local[:2]))
        centered_on_pad = return_distance <= _C07_RETURN_XY_TOLERANCE
        expected_center_z = self._current_table_top_z() + _C07_PITCHER_HALF_HEIGHT
        center_z_error = float(pitcher[2] - expected_center_z)
        on_pad_height = abs(center_z_error) <= _C07_RETURN_Z_TOLERANCE
        velocity = self._physics_body_velocity("c07_pitcher")
        stable = velocity is None or (
            float(np.linalg.norm(velocity[:3])) <= _C07_STABLE_LINEAR_SPEED
            and float(np.linalg.norm(velocity[3:])) <= _C07_STABLE_ANGULAR_SPEED
        )
        hand_min_distance, hand_clear = self._c07_hand_clearance()
        returned = bool(
            centered_on_pad
            and upright_alignment >= _C07_PITCHER_UPRIGHT_ALIGNMENT
            and on_pad_height
            and stable
            and hand_clear
        )
        enough_inside = inside_count >= _C07_REQUIRED_INSIDE
        spill_limited = spilled_count <= _C07_MAX_SPILLED
        success_now = bool(enough_inside and spill_limited and self._c07_tilt_seen and returned)
        return {
            "c07_enough_inside": bool(enough_inside),
            "c07_spill_limited": bool(spill_limited),
            "c07_tilt_seen": bool(self._c07_tilt_seen),
            "c07_pitcher_returned": bool(returned),
            "c07_pitcher_centered_on_pad": bool(centered_on_pad),
            "c07_pitcher_on_pad_height": bool(on_pad_height),
            "c07_pitcher_stable": bool(stable),
            "c07_hand_released": bool(hand_clear),
            "c07_success_now": bool(success_now),
            "c07_particles_inside": int(inside_count),
            "c07_particles_delivered_latched": int(len(delivered_particles)),
            "c07_particles_in_pitcher": int(pitcher_count),
            "c07_particles_required": int(_C07_REQUIRED_INSIDE),
            "c07_spilled_count": int(spilled_count),
            "c07_max_spilled": int(_C07_MAX_SPILLED),
            "c07_pitcher_upright_alignment": float(upright_alignment),
            "c07_return_distance": float(return_distance),
            "c07_pitcher_center_z_error": float(center_z_error),
            "c07_pitcher_expected_center_z": float(expected_center_z),
            "c07_pitcher_linear_speed": (
                float(np.linalg.norm(velocity[:3])) if velocity is not None else 0.0
            ),
            "c07_pitcher_angular_speed": (
                float(np.linalg.norm(velocity[3:])) if velocity is not None else 0.0
            ),
            "c07_hand_min_surface_distance": float(hand_min_distance),
            "c07_hand_release_clearance": float(_C07_HAND_RELEASE_CLEARANCE),
            "c07_release_hold_seconds": float(_C07_RELEASE_HOLD_SECONDS),
        }

    def _c07_pitcher_pour_cup_success_now(self) -> bool:
        self._last_c07_success_metrics = self._c07_pitcher_pour_metrics()
        self._set_task_visual_names_visible(
            {f"c07_cup_goal_rim_{index}" for index in range(8)},
            bool(self._last_c07_success_metrics["c07_enough_inside"]),
        )
        return bool(self._last_c07_success_metrics["c07_success_now"])

    def _c08_tongs_ice_metrics(self) -> dict[str, Any]:
        cup = self._poses.get("c08_glass_cup", self._fallback_pose("c08_glass_cup"))
        delivered_ice = getattr(self, "_c08_delivered_ice", set())
        ice_inside = 0
        for name in _C08_ICE_NAMES:
            ice = self._poses.get(name, self._fallback_pose(name))
            local = self._local_point_in_body_frame("c08_glass_cup", ice[:3])
            inside = bool(
                float(np.linalg.norm(local[:2])) <= _C08_CUP_INNER_RADIUS
                and _C08_CUP_LOCAL_Z_RANGE[0] <= float(local[2]) <= _C08_CUP_LOCAL_Z_RANGE[1]
            )
            if name in delivered_ice or inside:
                ice_inside += 1
        reading = self._physics_dof_position("c08_tongs_joint")
        joint_angle = _C08_TONGS_REST_RAD if reading is None else float(reading[0])
        opened = abs(joint_angle) >= _C08_TONGS_OPEN_RAD
        closed = abs(joint_angle) <= _C08_TONGS_CLOSED_RAD
        if self._c08_tongs_grasp is None:
            if opened:
                self._c08_tongs_was_open = True
            elif closed and self._c08_tongs_was_open:
                self._c08_tongs_close_count += 1
                self._c08_tongs_was_open = False
        cup_upright = self._z_axis_alignment(cup) >= _C08_GLASS_UPRIGHT_ALIGNMENT
        all_inside = ice_inside >= _C08_REQUIRED_ICE
        enough_closes = self._c08_tongs_close_count >= _C08_REQUIRED_ICE
        success_now = bool(all_inside and enough_closes and cup_upright)
        return {
            "c08_all_ice_inside": bool(all_inside),
            "c08_tongs_cycles": bool(enough_closes),
            "c08_cup_upright": bool(cup_upright),
            "c08_success_now": bool(success_now),
            "c08_ice_inside": int(ice_inside),
            "c08_ice_delivered_latched": int(len(delivered_ice)),
            "c08_required_ice": int(_C08_REQUIRED_ICE),
            "c08_tongs_close_count": int(self._c08_tongs_close_count),
            "c08_tongs_joint_angle": float(joint_angle),
            "c08_tongs_open_latched": bool(self._c08_tongs_was_open),
        }

    def _c08_tongs_ice_cubes_success_now(self) -> bool:
        self._last_c08_success_metrics = self._c08_tongs_ice_metrics()
        self._set_task_visual_names_visible(
            {
                "c08_cup_goal_rim_xp",
                "c08_cup_goal_rim_xn",
                "c08_cup_goal_rim_yp",
                "c08_cup_goal_rim_yn",
            },
            bool(self._last_c08_success_metrics["c08_all_ice_inside"]),
        )
        return bool(self._last_c08_success_metrics["c08_success_now"])

    def _c09_staple_metrics(self) -> dict[str, Any]:
        paper0 = self._poses.get("c09_paper_0", self._fallback_pose("c09_paper_0"))
        paper1 = self._poses.get("c09_paper_1", self._fallback_pose("c09_paper_1"))
        stapler = self._poses.get(
            "c09_stapler",
            self._fallback_pose("c09_stapler"),
        )
        staple_zone_local = _C09_STAPLE_ZONE_LOCAL
        staple_zone_world = self._body_local_point_world(
            "c09_stapler",
            staple_zone_local,
        )
        reading = self._physics_dof_position("c09_stapler_top_joint")
        top_angle = 0.0 if reading is None else float(reading[0] - reading[1])
        if top_angle >= _C09_TOP_OPEN_RAD:
            self._c09_top_opened_seen = True
        top_pressed = bool(self._c09_top_opened_seen and top_angle <= _C09_TOP_PRESSED_RAD)
        paper1_in_paper0 = self._local_point_in_body_frame(
            "c09_paper_0",
            paper1[:3],
        )
        paper_xy_error = float(np.linalg.norm(paper1_in_paper0[:2]))
        paper_z_error = abs(float(paper1_in_paper0[2]))
        paper0_matrix = _pose_vector_to_matrix(paper0)
        paper1_matrix = _pose_vector_to_matrix(paper1)
        yaw_aligned = bool(
            float(np.dot(paper0_matrix[:3, 0], paper1_matrix[:3, 0]))
            >= _C09_PAPER_YAW_ALIGNMENT
            and float(np.dot(paper0_matrix[:3, 1], paper1_matrix[:3, 1]))
            >= _C09_PAPER_YAW_ALIGNMENT
        )
        full_extents = _C09_PAPER_HALF_EXTENTS_XY * 2.0
        overlap_extents = np.maximum(
            0.0,
            full_extents - np.abs(paper1_in_paper0[:2]),
        )
        paper_overlap_ratio = float(
            np.prod(overlap_extents) / np.prod(full_extents)
        )
        papers_overlap = bool(
            paper_overlap_ratio >= _C09_MIN_PAPER_OVERLAP_RATIO
            and paper_z_error <= _C09_PAPER_Z_TOLERANCE
            and yaw_aligned
        )
        stapler_matrix = _pose_vector_to_matrix(stapler)
        stapler_inverse = _invert_pose_matrix(stapler_matrix)
        in_zone_flags = []
        mark_xy_errors = []
        mark_z_errors = []
        mark_alignment_values = []
        for paper_matrix in (paper0_matrix, paper1_matrix):
            mark_world = (
                paper_matrix
                @ np.asarray([*_C09_STAPLE_MARK_LOCAL, 1.0], dtype=np.float64)
            )[:3]
            mark_local = (
                stapler_inverse
                @ np.asarray([*mark_world, 1.0], dtype=np.float64)
            )[:3]
            mark_xy_error = float(
                np.linalg.norm(mark_local[:2] - staple_zone_local[:2])
            )
            mark_z_error = abs(
                float(mark_local[2] - staple_zone_local[2])
            )
            mark_alignment = float(
                np.dot(paper_matrix[:3, 0], stapler_matrix[:3, 0])
            )
            mark_normal_alignment = float(
                np.dot(paper_matrix[:3, 2], stapler_matrix[:3, 2])
            )
            mark_xy_errors.append(mark_xy_error)
            mark_z_errors.append(mark_z_error)
            mark_alignment_values.append(
                min(mark_alignment, mark_normal_alignment)
            )
            in_zone_flags.append(
                bool(
                    mark_xy_error <= _C09_FINAL_MARK_XY_TOLERANCE
                    and mark_z_error <= _C09_FINAL_MARK_Z_TOLERANCE
                    and mark_alignment >= _C09_STAPLER_CAPTURE_ALIGNMENT
                    and mark_normal_alignment
                    >= _C09_STAPLER_CAPTURE_ALIGNMENT
                )
            )
        papers_in_zone = bool(all(in_zone_flags))
        success_now = bool(papers_overlap and papers_in_zone and self._c09_top_opened_seen and top_pressed)
        return {
            "c09_papers_overlap": bool(papers_overlap),
            "c09_papers_in_zone": bool(papers_in_zone),
            "c09_stapler_opened": bool(self._c09_top_opened_seen),
            "c09_stapler_pressed": bool(top_pressed),
            "c09_success_now": bool(success_now),
            "c09_paper_xy_error": paper_xy_error,
            "c09_paper_overlap_ratio": paper_overlap_ratio,
            "c09_paper_z_error": paper_z_error,
            "c09_paper_yaw_aligned": bool(yaw_aligned),
            "c09_staple_mark_xy_errors": mark_xy_errors,
            "c09_staple_mark_z_errors": mark_z_errors,
            "c09_staple_mark_alignments": mark_alignment_values,
            "c09_top_joint_angle": float(top_angle),
            "c09_work_zone_center_distance": float(
                np.linalg.norm(
                    (
                        (
                            self._c09_stack_mark_world(paper0_matrix)
                            + self._c09_stack_mark_world(paper1_matrix)
                        )[:2]
                        * 0.5
                    )
                    - staple_zone_world[:2]
                )
            ),
        }

    def _c09_staple_paper_sheets_success_now(self) -> bool:
        self._last_c09_success_metrics = self._c09_staple_metrics()
        self._set_task_visual_names_visible(
            {"c09_staple_done"},
            bool(self._last_c09_success_metrics["c09_success_now"]),
        )
        return bool(self._last_c09_success_metrics["c09_success_now"])

    @staticmethod
    def _c10_connection_geometry(
        plug: np.ndarray,
        cable: np.ndarray,
        strip: np.ndarray,
    ) -> dict[str, Any]:
        plug_matrix = _pose_vector_to_matrix(plug)
        cable_matrix = _pose_vector_to_matrix(cable)
        strip_matrix = _pose_vector_to_matrix(strip)
        strip_socket_entrance = (
            strip_matrix @ np.asarray([0.0, 0.0, 0.038, 1.0], dtype=np.float64)
        )[:3]
        plug_prong_tip = (
            plug_matrix @ np.asarray([0.028, 0.0, 0.0, 1.0], dtype=np.float64)
        )[:3]
        plug_shoulder = (
            plug_matrix @ np.asarray([-0.006, 0.0, 0.0, 1.0], dtype=np.float64)
        )[:3]
        plug_socket = (
            plug_matrix @ np.asarray([-0.083, 0.0, 0.0, 1.0], dtype=np.float64)
        )[:3]
        cable_tip = (
            cable_matrix @ np.asarray([0.041, 0.0, 0.0, 1.0], dtype=np.float64)
        )[:3]
        cable_shoulder = (
            cable_matrix @ np.asarray([0.025, 0.0, 0.0, 1.0], dtype=np.float64)
        )[:3]
        strip_normal = strip_matrix[:3, 2]
        plug_axis = plug_matrix[:3, 0]
        cable_axis = cable_matrix[:3, 0]

        plug_strip_delta = plug_prong_tip - strip_socket_entrance
        plug_insertion_depth = -float(np.dot(plug_strip_delta, strip_normal))
        plug_lateral_delta = (
            plug_strip_delta + plug_insertion_depth * strip_normal
        )
        plug_shoulder_gap = float(
            np.dot(plug_shoulder - strip_socket_entrance, strip_normal)
        )
        plug_alignment = float(np.dot(plug_axis, -strip_normal))
        ideal_plug_matrix = strip_matrix @ _C10_PLUG_SEATED_RELATIVE
        plug_secondary_alignment = float(
            np.dot(
                plug_matrix[:3, 1],
                ideal_plug_matrix[:3, 1],
            )
        )
        first_connected = bool(
            float(np.linalg.norm(plug_lateral_delta))
            <= _C10_PLUG_LATERAL_TOLERANCE
            and plug_insertion_depth >= _C10_PLUG_MIN_INSERTION_DEPTH
            and abs(plug_shoulder_gap) <= _C10_PLUG_SHOULDER_TOLERANCE
            and plug_alignment >= _C10_PLUG_AXIS_ALIGNMENT
            and plug_secondary_alignment >= _C10_PLUG_AXIS_ALIGNMENT
        )

        cable_delta = cable_tip - plug_socket
        cable_insertion_depth = float(np.dot(cable_delta, plug_axis))
        cable_lateral_delta = cable_delta - cable_insertion_depth * plug_axis
        cable_shoulder_gap = float(
            np.dot(cable_shoulder - plug_socket, plug_axis)
        )
        cable_alignment = float(np.dot(cable_axis, plug_axis))
        ideal_cable_matrix = (
            plug_matrix @ _C10_CABLE_SEATED_RELATIVE
        )
        cable_secondary_alignment = float(
            np.dot(
                cable_matrix[:3, 1],
                ideal_cable_matrix[:3, 1],
            )
        )
        second_connected = bool(
            float(np.linalg.norm(cable_lateral_delta))
            <= _C10_CABLE_LATERAL_TOLERANCE
            and cable_insertion_depth >= _C10_CABLE_MIN_INSERTION_DEPTH
            and abs(cable_shoulder_gap) <= _C10_CABLE_SHOULDER_TOLERANCE
            and cable_alignment >= _C10_PLUG_AXIS_ALIGNMENT
            and cable_secondary_alignment >= _C10_PLUG_AXIS_ALIGNMENT
        )
        return {
            "first_connected_now": first_connected,
            "second_connected_now": second_connected,
            "plug_strip_delta": plug_strip_delta,
            "plug_lateral_delta": plug_lateral_delta,
            "cable_plug_delta": cable_delta,
            "cable_lateral_delta": cable_lateral_delta,
            "plug_lateral_error": float(np.linalg.norm(plug_lateral_delta)),
            "plug_insertion_depth": plug_insertion_depth,
            "plug_shoulder_gap": plug_shoulder_gap,
            "plug_axis_alignment": plug_alignment,
            "plug_secondary_alignment": plug_secondary_alignment,
            "cable_lateral_error": float(np.linalg.norm(cable_lateral_delta)),
            "cable_insertion_depth": cable_insertion_depth,
            "cable_shoulder_gap": cable_shoulder_gap,
            "cable_axis_alignment": cable_alignment,
            "cable_secondary_alignment": cable_secondary_alignment,
            "plug_matrix": plug_matrix,
            "cable_matrix": cable_matrix,
            "strip_matrix": strip_matrix,
        }

    def _apply_c10_progressive_capture(
        self,
        poses: dict[str, np.ndarray],
        geometry: dict[str, Any],
    ) -> None:
        """Report capture state without magnetizing a connector held by nobody.

        The previous implementation drove any free rigid body inside the broad
        funnel to a canonical pose.  C10 guidance now lives in the connector's
        real hand-grasp state, so this compatibility hook is deliberately
        read-only.
        """
        self._c10_assist_metrics = {
            "c10_plug_capture_active": any(
                str(state.get("name")) == "c10_charger_plug"
                and str(state.get("phase")) != "carry"
                for state in self._c10_connector_grasps.values()
            ),
            "c10_cable_capture_active": any(
                str(state.get("name")) == "c10_charging_cable"
                and str(state.get("phase")) != "carry"
                for state in self._c10_connector_grasps.values()
            ),
        }

    @staticmethod
    def _c10_grasp_box(
        name: str,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        if name == "c10_charger_plug":
            return (
                _C10_PLUG_GRASP_BOX_CENTER,
                _C10_PLUG_GRASP_BOX_HALF_EXTENTS,
                _C10_PLUG_PINCH_DISTANCE_RANGE,
            )
        return (
            _C10_CABLE_GRASP_BOX_CENTER,
            _C10_CABLE_GRASP_BOX_HALF_EXTENTS,
            _C10_CABLE_PINCH_DISTANCE_RANGE,
        )

    def _c10_grasp_candidate(
        self,
        side: str,
        tip_positions: dict[str, np.ndarray] | None = None,
    ) -> tuple[str, np.ndarray, float, int] | None:
        occupied = {
            str(state.get("name"))
            for held_side, state in self._c10_connector_grasps.items()
            if held_side != side
        }
        tips = (
            self._wuji_fingertip_positions(side)
            if tip_positions is None
            else tip_positions
        )
        candidates: list[tuple[float, str, np.ndarray, int]] = []
        for name in ("c10_charger_plug", "c10_charging_cable"):
            if name in occupied:
                continue
            if (
                name == "c10_charger_plug"
                and self._c10_plug_latched
            ) or (
                name == "c10_charging_cable"
                and self._c10_cable_latched
            ):
                continue
            matrix = self._current_body_matrix(name)
            if matrix is None:
                continue
            center, half_extents, pinch_range = self._c10_grasp_box(name)
            contact, error, count = self._visible_box_pinch_contact(
                matrix,
                tips,
                side=side,
                center_local=center,
                half_extents=half_extents,
                contact_margin=_C10_GRASP_CONTACT_MARGIN,
                near_margin=_C10_GRASP_NEAR_MARGIN,
                pinch_distance_range=pinch_range,
            )
            if contact:
                candidates.append(
                    (float(error), name, matrix.copy(), int(count))
                )
        if not candidates:
            return None
        error, name, matrix, count = min(candidates, key=lambda item: item[0])
        return name, matrix, error, count

    @staticmethod
    def _c10_guidance_targets(
        name: str,
        geometry: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        strip = np.asarray(geometry["strip_matrix"], dtype=np.float64)
        plug = np.asarray(geometry["plug_matrix"], dtype=np.float64)
        ideal_plug = strip @ _C10_PLUG_SEATED_RELATIVE
        if name == "c10_charger_plug":
            approach = ideal_plug.copy()
            approach[:3, 3] += (
                _C10_PLUG_IDEAL_DEPTH - _C10_APPROACH_PLUG_DEPTH
            ) * strip[:3, 2]
            return approach, ideal_plug
        # Once the first plug is latched, the cable must follow that actual
        # visible socket pose rather than a separate canonical plug pose.
        seated = plug @ _C10_CABLE_SEATED_RELATIVE
        approach = seated.copy()
        approach[:3, 3] -= (
            _C10_CABLE_IDEAL_DEPTH - _C10_APPROACH_CABLE_DEPTH
        ) * plug[:3, 0]
        return approach, seated

    def _c10_guidance_trigger(
        self,
        name: str,
        geometry: dict[str, Any],
    ) -> bool:
        if name == "c10_charger_plug":
            lateral = float(geometry["plug_lateral_error"])
            alignment = float(geometry["plug_axis_alignment"])
            secondary_alignment = float(
                geometry.get("plug_secondary_alignment", 1.0)
            )
            depth = float(geometry["plug_insertion_depth"])
            if self._c10_plug_latched:
                return False
            broad = bool(
                lateral <= _C10_PLUG_CAPTURE_LATERAL
                and alignment >= _C10_CAPTURE_AXIS_ALIGNMENT
                and _C10_CAPTURE_MIN_DEPTH
                <= depth
                <= _C10_PLUG_MIN_INSERTION_DEPTH + 0.020
            )
        else:
            lateral = float(geometry["cable_lateral_error"])
            alignment = float(geometry["cable_axis_alignment"])
            secondary_alignment = float(
                geometry.get("cable_secondary_alignment", 1.0)
            )
            depth = float(geometry["cable_insertion_depth"])
            if (
                not self._c10_plug_latched
                or self._c10_cable_latched
                or self._c10_plug_latch_handoff_pending()
            ):
                return False
            broad = bool(
                lateral <= _C10_CABLE_CAPTURE_LATERAL
                and alignment >= _C10_CAPTURE_AXIS_ALIGNMENT
                and _C10_CAPTURE_MIN_DEPTH
                <= depth
                <= _C10_CABLE_MIN_INSERTION_DEPTH + 0.020
            )
        if not broad:
            return False
        # Never rotate a visibly wedged connector inside a tight channel.
        if depth > 0.0 and (
            lateral > 0.004
            or alignment < float(np.cos(np.deg2rad(10.0)))
            or secondary_alignment
            < _C10_IN_CHANNEL_SECONDARY_ALIGNMENT
        ):
            return False
        return True

    def _c10_install_hand_target(
        self,
        side: str,
        state: dict[str, Any],
        object_target: np.ndarray,
    ) -> None:
        state["hand_target_matrix"] = (
            np.asarray(object_target, dtype=np.float64).reshape(4, 4)
            @ _invert_pose_matrix(
                np.asarray(
                    state["hand_to_object"],
                    dtype=np.float64,
                ).reshape(4, 4)
            )
        )

    def _c10_hand_tracking_ready(
        self,
        side: str,
        state: dict[str, Any],
    ) -> bool:
        return self._assist_hand_tracking_ready(
            side,
            state,
            position_tolerance=_C10_HAND_TRACK_POSITION_TOLERANCE,
            angle_tolerance=_C10_HAND_TRACK_ANGLE_TOLERANCE,
        )

    def _c10_plug_latch_handoff_pending(self) -> bool:
        return any(
            str(state.get("name")) == "c10_charger_plug"
            and str(state.get("phase")) == "latched_handoff"
            and not bool(state.get("hand_target_installed"))
            for state in self._c10_connector_grasps.values()
        )

    def _pin_c10_latched_connectors(self) -> bool:
        """Hold latched freejoint imports through their articulation root."""
        strip = self._current_body_matrix("c10_power_strip")
        if strip is None:
            return True
        handoff_pending = {
            str(state.get("name"))
            for state in self._c10_connector_grasps.values()
            if (
                str(state.get("phase")) == "latched_handoff"
                and not bool(state.get("hand_target_installed"))
            )
        }
        plug_relative = (
            _C10_PLUG_SEATED_RELATIVE
            if self._c10_plug_latch_relative is None
            else np.asarray(
                self._c10_plug_latch_relative,
                dtype=np.float64,
            ).reshape(4, 4)
        )
        plug_target = strip @ plug_relative
        if (
            self._c10_plug_latched
            and "c10_charger_plug" not in handoff_pending
        ):
            applied, _linear, _angular = self._set_body_rigid_grasp_target(
                "c10_charger_plug",
                plug_target,
                previous_target=plug_target,
            )
            if not applied:
                self._manual_reset_requested = True
                return False
        if (
            self._c10_cable_latched
            and "c10_charging_cable" not in handoff_pending
        ):
            cable_relative = (
                _C10_CABLE_SEATED_RELATIVE
                if self._c10_cable_latch_relative is None
                else np.asarray(
                    self._c10_cable_latch_relative,
                    dtype=np.float64,
                ).reshape(4, 4)
            )
            cable_target = plug_target @ cable_relative
            applied, _linear, _angular = self._set_body_rigid_grasp_target(
                "c10_charging_cable",
                cable_target,
                previous_target=cable_target,
            )
            if not applied:
                self._manual_reset_requested = True
                return False
        return True

    def _update_c10_connector_assist(self) -> None:
        if not self._pin_c10_latched_connectors():
            return
        palms = {
            side: self._robot_link_pose_matrix(side)
            for side in ("right", "left")
        }
        for side in tuple(self._c10_connector_grasps):
            state = self._c10_connector_grasps[side]
            name = str(state.get("name"))
            already_latched = (
                self._c10_plug_latched
                if name == "c10_charger_plug"
                else self._c10_cable_latched
            )
            if (
                palms.get(side) is None
                or self._raw_hand_command_max(side)
                <= _C10_GRASP_RELEASE_HAND_COMMAND
                or (
                    already_latched
                    and str(state.get("phase")) != "latched_handoff"
                )
            ):
                self._preserve_released_hand_handoff(
                    side,
                    state,
                    palms.get(side),
                )
                self._c10_connector_grasps.pop(side, None)

        for side, palm in palms.items():
            if (
                palm is None
                or side in self._c10_connector_grasps
                or self._raw_hand_command_max(side)
                < _C10_GRASP_ATTACH_HAND_COMMAND
            ):
                continue
            candidate = self._c10_grasp_candidate(
                side,
                self._wuji_fingertip_positions(side),
            )
            if candidate is None:
                continue
            name, matrix, error, count = candidate
            self._c10_connector_grasps[side] = {
                "name": name,
                "phase": "carry",
                "hand_to_object": _invert_pose_matrix(palm) @ matrix,
                "last_target": matrix.copy(),
                "hand_target_matrix": None,
                "hand_target_installed": False,
                "visual_hand": self._actual_hand_joint_pose(side),
                "contact_error": float(error),
                "contact_count": int(count),
                "stable_steps": 0,
            }

        poses: dict[str, np.ndarray] = {}
        for name in (
            "c10_charger_plug",
            "c10_charging_cable",
            "c10_power_strip",
        ):
            current = self._current_body_pose(name)
            if current is None:
                self._update_c10_connection_latches()
                return
            poses[name] = current
        geometry = self._c10_connection_geometry(
            poses["c10_charger_plug"],
            poses["c10_charging_cable"],
            poses["c10_power_strip"],
        )

        for side, state in tuple(self._c10_connector_grasps.items()):
            palm = palms.get(side)
            if palm is None:
                continue
            name = str(state["name"])
            current = self._current_body_matrix(name)
            if current is None:
                continue
            phase = str(state.get("phase", "carry"))
            if phase == "latched_handoff":
                if not bool(state.get("hand_target_installed")):
                    # During the remainder of the commit control frame the arm
                    # may still be following the previously installed guide.
                    # Keep the connector at the exact actual palm transform and
                    # continuously stage that same hand pose.  The next control
                    # frame installs it; only then does the latch become a
                    # fixed strip/plug-relative pin.
                    carried = palm @ np.asarray(
                        state["hand_to_object"],
                        dtype=np.float64,
                    ).reshape(4, 4)
                    applied, _linear, _angular = (
                        self._set_body_rigid_grasp_target(
                            name,
                            carried,
                            previous_target=state.get("last_target"),
                        )
                    )
                    if applied:
                        state["last_target"] = carried.copy()
                        state["hand_target_matrix"] = palm.copy()
                        self._poses[name] = self._pose_matrix_to_vector(
                            carried
                        )
                        if name == "c10_charger_plug":
                            strip = self._current_body_matrix(
                                "c10_power_strip"
                            )
                            if strip is not None:
                                self._c10_plug_latch_relative = (
                                    _invert_pose_matrix(strip) @ carried
                                )
                        else:
                            plug = self._current_body_matrix(
                                "c10_charger_plug"
                            )
                            if plug is not None:
                                self._c10_cable_latch_relative = (
                                    _invert_pose_matrix(plug) @ carried
                                )
                continue
            carried = palm @ np.asarray(
                state["hand_to_object"],
                dtype=np.float64,
            ).reshape(4, 4)
            target = carried.copy()
            if phase == "carry":
                state["hand_target_matrix"] = None
                if self._c10_guidance_trigger(name, geometry):
                    approach, seated = self._c10_guidance_targets(
                        name,
                        geometry,
                    )
                    state.update(
                        {
                            "phase": "funnel_aligning",
                            "approach_target": approach,
                            "seated_target": seated,
                            "last_target": current.copy(),
                            "stable_steps": 0,
                        }
                    )
                    phase = "funnel_aligning"
                    target = current.copy()
                    self._c10_install_hand_target(side, state, target)
            if phase in {"funnel_aligning", "inserting"}:
                goal = np.asarray(
                    state[
                        "approach_target"
                        if phase == "funnel_aligning"
                        else "seated_target"
                    ],
                    dtype=np.float64,
                ).reshape(4, 4)
                # The connector itself stays at the exact transform selected
                # by the user's physical palm.  Guidance only stages the next
                # hand command; it never moves the connector ahead of a hand
                # that has not reached that command yet.
                target = carried.copy()
                if self._c10_hand_tracking_ready(side, state):
                    command_target = self._bounded_pose_step(
                        carried,
                        goal,
                        dt=self._control_dt,
                        max_linear_speed=(
                            _C10_ALIGN_MAX_LINEAR_SPEED
                            if phase == "funnel_aligning"
                            else _C10_INSERT_MAX_LINEAR_SPEED
                        ),
                        max_angular_speed=_C10_ALIGN_MAX_ANGULAR_SPEED,
                    )
                    self._c10_install_hand_target(
                        side,
                        state,
                        command_target,
                    )

            applied, linear_error, angular_error = (
                self._set_body_rigid_grasp_target(
                    name,
                    target,
                    previous_target=state.get("last_target"),
                )
            )
            state.update(
                {
                    "applied": bool(applied),
                    "linear_error": float(linear_error),
                    "angular_error": float(angular_error),
                }
            )
            if not applied:
                continue
            state["last_target"] = target.copy()
            if phase in {"funnel_aligning", "inserting"}:
                goal = np.asarray(
                    state[
                        "approach_target"
                        if phase == "funnel_aligning"
                        else "seated_target"
                    ],
                    dtype=np.float64,
                ).reshape(4, 4)
                reached = self._tool_lock_target_reached(target, goal)
                state["stable_steps"] = (
                    int(state.get("stable_steps", 0)) + 1 if reached else 0
                )
                if (
                    phase == "funnel_aligning"
                    and int(state["stable_steps"]) >= 2
                ):
                    state["phase"] = "inserting"
                    state["stable_steps"] = 0

        refreshed = {
            name: self._current_body_pose(name)
            for name in (
                "c10_charger_plug",
                "c10_charging_cable",
                "c10_power_strip",
            )
        }
        if all(value is not None for value in refreshed.values()):
            refreshed_geometry = self._c10_connection_geometry(
                refreshed["c10_charger_plug"],
                refreshed["c10_charging_cable"],
                refreshed["c10_power_strip"],
            )
            self._apply_c10_progressive_capture(
                {
                    name: np.asarray(value, dtype=np.float64)
                    for name, value in refreshed.items()
                },
                refreshed_geometry,
            )
        self._update_c10_connection_latches()

    def _update_c10_connection_latches(self) -> None:
        poses: dict[str, np.ndarray] = {}
        for name in (
            "c10_charger_plug",
            "c10_charging_cable",
            "c10_power_strip",
        ):
            pose = self._physics_body_pose(name)
            if pose is None:
                pose = self._poses.get(name, self._fallback_pose(name))
            poses[name] = np.asarray(pose, dtype=np.float64)
            self._poses[name] = poses[name]

        geometry = self._c10_connection_geometry(
            poses["c10_charger_plug"],
            poses["c10_charging_cable"],
            poses["c10_power_strip"],
        )
        first_connected_now = bool(geometry["first_connected_now"])
        second_connected_now = bool(geometry["second_connected_now"])
        self._c10_latch_stable_steps["c10_charger_plug"] = (
            int(self._c10_latch_stable_steps["c10_charger_plug"]) + 1
            if first_connected_now
            else 0
        )
        if second_connected_now and not self._c10_first_connected_seen:
            self._c10_order_violation = True
        if (
            first_connected_now
            and int(
                self._c10_latch_stable_steps["c10_charger_plug"]
            )
            >= _C10_LATCH_STABLE_STEPS
        ):
            if self._c10_plug_latched:
                self._c10_first_connected_seen = True
            else:
                committed = self._commit_c10_connector_latch(
                    "c10_charger_plug",
                    np.asarray(geometry["strip_matrix"], dtype=np.float64)
                    @ _C10_PLUG_SEATED_RELATIVE,
                )
                if committed:
                    self._c10_first_connected_seen = True
        second_stable_eligible = bool(
            second_connected_now
            and self._c10_plug_latched
            and not self._c10_plug_latch_handoff_pending()
            and not self._c10_order_violation
        )
        self._c10_latch_stable_steps["c10_charging_cable"] = (
            int(self._c10_latch_stable_steps["c10_charging_cable"]) + 1
            if second_stable_eligible
            else 0
        )
        if (
            second_stable_eligible
            and int(
                self._c10_latch_stable_steps["c10_charging_cable"]
            )
            >= _C10_LATCH_STABLE_STEPS
            and not self._c10_cable_latched
        ):
            plug_matrix = np.asarray(
                geometry["plug_matrix"],
                dtype=np.float64,
            )
            self._commit_c10_connector_latch(
                "c10_charging_cable",
                plug_matrix @ _C10_CABLE_SEATED_RELATIVE,
            )

    def _commit_c10_connector_latch(
        self,
        name: str,
        target_matrix: np.ndarray,
    ) -> bool:
        """Commit pose, kinematic state, and logical latch transactionally."""
        target = np.asarray(target_matrix, dtype=np.float64).reshape(4, 4)
        previous_pose = self._current_body_pose(name)
        current_matrix = (
            None
            if previous_pose is None
            else _pose_vector_to_matrix(previous_pose)
        )
        if (
            current_matrix is not None
            and not self._tool_lock_target_reached(
                current_matrix,
                target,
            )
        ):
            # Connection geometry is intentionally a little forgiving for
            # detection.  The irreversible latch itself is not: let the hand
            # finish the visible guide instead of snapping a merely-near plug
            # to the canonical seated pose.
            return False
        latch_target = (
            target.copy()
            if current_matrix is None
            else current_matrix.copy()
        )
        pose = self._pose_matrix_to_vector(latch_target)
        uses_articulation_pin = name in self._task_articulation_views
        if uses_articulation_pin:
            pose_applied, _linear_error, _angular_error = (
                self._set_body_rigid_grasp_target(
                    name,
                    latch_target,
                    previous_target=latch_target,
                )
            )
        else:
            view = self._task_body_views.get(name)
            if view is None:
                pose_applied = self._stage is None
            else:
                pose_applied = self._set_task_body_world_pose(view, pose)
        if not pose_applied:
            return False
        if not uses_articulation_pin:
            kinematic_applied = self._set_c10_connector_kinematic(name, True)
            if not kinematic_applied and self._stage is not None:
                # The connector is dynamic before latching.  Restore both its
                # physical pose and dynamic state if the kinematic commit
                # fails, instead of leaving a visually seated but unlatched
                # connector behind.
                view = self._task_body_views.get(name)
                if view is not None and previous_pose is not None:
                    self._set_task_body_world_pose(view, previous_pose)
                self._set_c10_connector_kinematic(name, False)
                return False
        self._poses[name] = pose.copy()
        for side, state in tuple(self._c10_connector_grasps.items()):
            if str(state.get("name")) == name:
                hand_target = (
                    latch_target
                    @ _invert_pose_matrix(
                        np.asarray(
                            state["hand_to_object"],
                            dtype=np.float64,
                        ).reshape(4, 4)
                    )
                )
                state.update(
                    {
                        "phase": "latched_handoff",
                        "last_target": latch_target.copy(),
                        "hand_target_matrix": hand_target,
                    }
                )
                self._start_raw_hand_handoff(
                    side,
                    state,
                    hand_target,
                )
        if name == "c10_charger_plug":
            strip = self._current_body_matrix("c10_power_strip")
            self._c10_plug_latch_relative = (
                _C10_PLUG_SEATED_RELATIVE.copy()
                if strip is None
                else _invert_pose_matrix(strip) @ latch_target
            )
            self._c10_plug_latched = True
            self._c10_first_connected_seen = True
        else:
            plug = self._current_body_matrix("c10_charger_plug")
            self._c10_cable_latch_relative = (
                _C10_CABLE_SEATED_RELATIVE.copy()
                if plug is None
                else _invert_pose_matrix(plug) @ latch_target
            )
            self._c10_cable_latched = True
        return True

    def _set_c10_connector_kinematic(
        self,
        name: str,
        enabled: bool,
    ) -> bool:
        return self._set_task_body_kinematic(
            name,
            enabled,
            warning_label="C10 connector",
        )

    def _set_task_body_kinematic(
        self,
        name: str,
        enabled: bool,
        *,
        warning_label: str = "task body",
    ) -> bool:
        if self._stage is None:
            return False
        try:
            from pxr import UsdPhysics

            prim = self._find_rigid_body_prim(name)
            if prim is None:
                return False
            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            if not rigid_api:
                rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            attribute = rigid_api.GetKinematicEnabledAttr()
            if not attribute.IsValid():
                attribute = rigid_api.CreateKinematicEnabledAttr()
            attribute.Set(bool(enabled))
            return True
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not set {warning_label} kinematic state for {name}: {exc}"
            )
            return False

    def _c10_connection_metrics(self) -> dict[str, Any]:
        plug = self._poses.get("c10_charger_plug", self._fallback_pose("c10_charger_plug"))
        cable = self._poses.get("c10_charging_cable", self._fallback_pose("c10_charging_cable"))
        strip = self._poses.get("c10_power_strip", self._fallback_pose("c10_power_strip"))
        geometry = self._c10_connection_geometry(plug, cable, strip)
        first_connected_now = bool(geometry["first_connected_now"])
        second_connected_now = bool(geometry["second_connected_now"])
        order_valid = bool(self._c10_first_connected_seen and not self._c10_order_violation)
        first_connected = bool(self._c10_plug_latched)
        second_connected = bool(self._c10_cable_latched)
        workspace_ok = bool(
            float(np.linalg.norm(strip[:2])) <= 1.0
            and plug[2] >= self._current_table_top_z() - 0.03
            and cable[2] >= self._current_table_top_z() - 0.03
        )
        success_now = bool(order_valid and second_connected and workspace_ok)
        return {
            "c10_first_connected": first_connected,
            "c10_second_connected": bool(second_connected),
            "c10_order_valid": bool(order_valid),
            "c10_workspace_ok": bool(workspace_ok),
            "c10_success_now": bool(success_now),
            "c10_first_connected_now": bool(first_connected_now),
            "c10_order_violation": bool(self._c10_order_violation),
            "c10_plug_latched": bool(self._c10_plug_latched),
            "c10_cable_latched": bool(self._c10_cable_latched),
            "c10_plug_latch_stable_steps": int(
                self._c10_latch_stable_steps["c10_charger_plug"]
            ),
            "c10_cable_latch_stable_steps": int(
                self._c10_latch_stable_steps["c10_charging_cable"]
            ),
            "c10_plug_strip_xy_error": float(geometry["plug_lateral_error"]),
            "c10_plug_strip_z_error": abs(float(geometry["plug_shoulder_gap"])),
            "c10_plug_insertion_depth": float(geometry["plug_insertion_depth"]),
            "c10_plug_axis_alignment": float(geometry["plug_axis_alignment"]),
            "c10_plug_secondary_alignment": float(
                geometry["plug_secondary_alignment"]
            ),
            "c10_cable_plug_xy_error": float(geometry["cable_lateral_error"]),
            "c10_cable_plug_z_error": abs(float(geometry["cable_shoulder_gap"])),
            "c10_cable_insertion_depth": float(geometry["cable_insertion_depth"]),
            "c10_cable_axis_alignment": float(geometry["cable_axis_alignment"]),
            "c10_cable_secondary_alignment": float(
                geometry["cable_secondary_alignment"]
            ),
        }

    def _t02_connect_charger_and_cable_success_now(self) -> bool:
        self._last_c10_success_metrics = self._c10_connection_metrics()
        self._set_task_visual_names_visible(
            {"c10_strip_indicator"},
            bool(self._last_c10_success_metrics["c10_first_connected"]),
        )
        self._set_task_visual_names_visible(
            {"c10_charge_indicator"},
            bool(self._last_c10_success_metrics["c10_second_connected"]),
        )
        return bool(self._last_c10_success_metrics["c10_success_now"])

    def _c11_mallet_grasp_contact(
        self,
        mallet: np.ndarray,
        tip_positions: dict[str, np.ndarray] | None = None,
    ) -> tuple[bool, float, int]:
        tips = (
            self._b02_wuji_fingertip_positions()
            if tip_positions is None
            else tip_positions
        )
        return self._visible_segment_pinch_contact(
            mallet,
            tips,
            side="right",
            segments_local=(_C11_MALLET_HANDLE_SEGMENT,),
            segment_radii=(_C11_MALLET_HANDLE_RADIUS,),
            contact_margin=_C11_MALLET_CONTACT_MARGIN,
            near_margin=_C11_MALLET_NEAR_CONTACT_MARGIN,
            pinch_distance_range=(
                _C11_MALLET_PINCH_MIN_DISTANCE,
                _C11_MALLET_PINCH_MAX_DISTANCE,
            ),
            max_axial_separation=_C11_MALLET_MAX_AXIAL_SEPARATION,
        )

    @staticmethod
    def _c11_head_world(mallet: np.ndarray) -> np.ndarray:
        return (
            np.asarray(mallet, dtype=np.float64).reshape(4, 4)
            @ np.asarray([*_C11_MALLET_HEAD_LOCAL, 1.0], dtype=np.float64)
        )[:3]

    def _c11_expected_key(self) -> str | None:
        if self._c11_sequence_progress >= len(self._c11_target_sequence):
            return None
        return str(self._c11_target_sequence[self._c11_sequence_progress])

    def _c11_head_local_to_key(
        self,
        key_name: str,
        head_world: np.ndarray,
    ) -> np.ndarray | None:
        key = self._current_body_matrix(key_name)
        if key is None:
            return None
        return (
            _invert_pose_matrix(key)
            @ np.asarray(
                [*np.asarray(head_world, dtype=np.float64).reshape(3), 1.0],
                dtype=np.float64,
            )
        )[:3]

    def _c11_strike_guide_candidate(
        self,
        mallet: np.ndarray,
        key_name: str,
    ) -> bool:
        head_local = self._c11_head_local_to_key(
            key_name,
            self._c11_head_world(mallet),
        )
        if head_local is None:
            return False
        half = _C11_KEY_HALF_EXTENTS[key_name]
        return bool(
            abs(float(head_local[0]))
            <= float(half[0]) + _C11_GUIDE_CAPTURE_XY_MARGIN
            and abs(float(head_local[1]))
            <= float(half[1]) + _C11_GUIDE_CAPTURE_XY_MARGIN
            and _C11_GUIDE_LOCAL_Z_RANGE[0]
            <= float(head_local[2])
            <= _C11_GUIDE_LOCAL_Z_RANGE[1]
        )

    def _c11_xy_guided_target(
        self,
        carried: np.ndarray,
        key_name: str,
    ) -> np.ndarray:
        target = np.asarray(carried, dtype=np.float64).reshape(4, 4).copy()
        key = self._current_body_matrix(key_name)
        if key is None:
            return target
        head_world = self._c11_head_world(target)
        local = (
            _invert_pose_matrix(key)
            @ np.asarray([*head_world, 1.0], dtype=np.float64)
        )[:3]
        correction_world = key[:3, :3] @ np.asarray(
            [-float(local[0]), -float(local[1]), 0.0],
            dtype=np.float64,
        )
        target[:3, 3] += correction_world
        return target

    def _c11_return_guided_target(
        self,
        carried: np.ndarray,
    ) -> np.ndarray:
        target = np.asarray(carried, dtype=np.float64).reshape(4, 4).copy()
        zone = self._current_body_matrix("c11_return_zone")
        if zone is None:
            return target
        target[:2, 3] += zone[:2, 3] - target[:2, 3]
        return target

    def _update_c11_mallet_assist(self) -> None:
        palm = self._robot_link_pose_matrix("right")
        mallet = self._current_body_matrix("c11_mallet")
        if palm is None or mallet is None:
            return
        raw_tcp = _pose_vector_to_matrix(self._raw_tcp_pose("right"))
        raw_hand = self._raw_right_hand_command_max()
        state = self._c11_mallet_grasp
        if state is not None and raw_hand <= _C11_GRASP_RELEASE_HAND_COMMAND:
            # Opening in mid-air really releases the mallet.  Never keep an
            # invisible attachment or pin it above the table.
            self._preserve_released_hand_handoff(
                "right",
                state,
                palm,
            )
            self._c11_mallet_grasp = None
            return
        if state is None:
            contact, error, count = self._c11_mallet_grasp_contact(
                mallet,
                self._b02_wuji_fingertip_positions(),
            )
            if (
                raw_hand >= _C11_GRASP_ATTACH_HAND_COMMAND
                and contact
            ):
                state = {
                    "phase": "carry",
                    "hand_to_object": _invert_pose_matrix(palm) @ mallet,
                    "last_target": mallet.copy(),
                    "hand_target_matrix": None,
                    "hand_target_installed": False,
                    "visual_hand": self._actual_right_hand_joint_pose(),
                    "contact_error": float(error),
                    "contact_count": int(count),
                    "guide_key": "",
                }
                self._c11_mallet_grasp = state
            else:
                return

        carried = palm @ np.asarray(
            state["hand_to_object"],
            dtype=np.float64,
        ).reshape(4, 4)
        target = carried.copy()
        phase = str(state.get("phase", "carry"))
        expected = self._c11_expected_key()
        if phase == "carry_offset":
            handoff_target = self._raw_hand_handoff_target("right", state)
            if handoff_target is not None:
                state["hand_target_matrix"] = handoff_target
        if (
            phase in {"carry", "carry_offset"}
            and expected is not None
            and self._c11_strike_guide_candidate(carried, expected)
        ):
            state.update(
                {
                    "phase": "strike_guided",
                    "guide_key": expected,
                    "last_target": mallet.copy(),
                    "hand_target_matrix": mallet
                    @ _invert_pose_matrix(
                        np.asarray(
                            state["hand_to_object"],
                            dtype=np.float64,
                        ).reshape(4, 4)
                    ),
                    "guide_raw_start_position": raw_tcp[:3, 3].copy(),
                    "guide_object_start": mallet.copy(),
                }
            )
            state.pop("raw_to_hand_offset", None)
            state.pop("raw_handoff_raw_anchor", None)
            state.pop("raw_handoff_hand_anchor", None)
            phase = "strike_guided"
            target = mallet.copy()
        elif (
            phase in {"carry", "carry_offset"}
            and expected is None
        ):
            zone = self._current_body_matrix("c11_return_zone")
            if (
                zone is not None
                and float(np.linalg.norm(mallet[:2, 3] - zone[:2, 3]))
                <= _C11_RETURN_GUIDE_CAPTURE_RADIUS
            ):
                state.update(
                    {
                        "phase": "return_guided",
                        "last_target": mallet.copy(),
                        "hand_target_matrix": mallet
                        @ _invert_pose_matrix(
                            np.asarray(
                                state["hand_to_object"],
                                dtype=np.float64,
                            ).reshape(4, 4)
                        ),
                        "guide_raw_start_position": raw_tcp[:3, 3].copy(),
                        "guide_object_start": mallet.copy(),
                    }
                )
                state.pop("raw_to_hand_offset", None)
                state.pop("raw_handoff_raw_anchor", None)
                state.pop("raw_handoff_hand_anchor", None)
                phase = "return_guided"
                target = mallet.copy()

        if phase in {"strike_guided", "return_guided"}:
            guide_object_start = np.asarray(
                state.get("guide_object_start", mallet),
                dtype=np.float64,
            ).reshape(4, 4)
            guide_raw_start = np.asarray(
                state.get(
                    "guide_raw_start_position",
                    raw_tcp[:3, 3],
                ),
                dtype=np.float64,
            ).reshape(3)
            raw_driven = guide_object_start.copy()
            raw_driven[:3, 3] += raw_tcp[:3, 3] - guide_raw_start
            if phase == "strike_guided":
                guide_key = str(state.get("guide_key", ""))
                goal = (
                    self._c11_xy_guided_target(raw_driven, guide_key)
                    if guide_key in _C11_KEY_HALF_EXTENTS
                    else raw_driven
                )
            else:
                goal = self._c11_return_guided_target(raw_driven)
            # Keep the mallet rigidly attached at the user's actual palm.
            # The projected guide pose is only the next arm command and is
            # not committed to the mallet until the hand physically follows.
            target = carried.copy()
            if self._assist_hand_tracking_ready(
                "right",
                state,
                position_tolerance=_C11_HAND_TRACK_POSITION_TOLERANCE,
                angle_tolerance=_C11_HAND_TRACK_ANGLE_TOLERANCE,
            ):
                command_target = self._bounded_pose_step(
                    carried,
                    goal,
                    dt=self._control_dt,
                    max_linear_speed=_C11_GUIDE_MAX_LINEAR_SPEED,
                    max_angular_speed=_C11_GUIDE_MAX_ANGULAR_SPEED,
                )
                state["hand_target_matrix"] = (
                    command_target
                    @ _invert_pose_matrix(
                        np.asarray(
                            state["hand_to_object"],
                            dtype=np.float64,
                        ).reshape(4, 4)
                    )
                )
        elif phase == "carry":
            state["hand_target_matrix"] = None

        applied, linear_error, angular_error = self._set_body_rigid_grasp_target(
            "c11_mallet",
            target,
            previous_target=state.get("last_target"),
        )
        state.update(
            {
                "applied": bool(applied),
                "linear_error": float(linear_error),
                "angular_error": float(angular_error),
            }
        )
        if applied:
            state["last_target"] = target.copy()
        if (
            phase == "strike_guided"
            and self._c11_hit_armed
            and str(state.get("guide_key", "")) != self._c11_expected_key()
        ):
            # The previous impact completed (or reset) progress and the mallet
            # has lifted enough to re-arm.  Rebase at the visible current pose.
            state["phase"] = "carry_offset"
            state["guide_key"] = ""
            state.pop("guide_raw_start_position", None)
            state.pop("guide_object_start", None)
            hand_target = (
                target
                @ _invert_pose_matrix(
                    np.asarray(
                        state["hand_to_object"],
                        dtype=np.float64,
                    ).reshape(4, 4)
                )
            )
            self._start_raw_hand_handoff(
                "right",
                state,
                hand_target,
            )

    def _c11_key_contact_force(self, key_name: str) -> float:
        return self._c04_contact_force(
            self._c11_key_contact_views.get(key_name)
        )

    def _update_c11_hit_events(self) -> None:
        """Advance the score only on a downward physical contact-layer event."""
        mallet = self._current_body_matrix("c11_mallet")
        if mallet is None:
            return
        head = self._c11_head_world(mallet)
        previous = self._c11_previous_head_world
        self._c11_previous_head_world = head.copy()
        if previous is None:
            return
        self._c11_pending_crossings = {
            key: int(steps) - 1
            for key, steps in self._c11_pending_crossings.items()
            if int(steps) > 1
        }

        local_now: dict[str, np.ndarray] = {}
        local_previous: dict[str, np.ndarray] = {}
        for key_name in _C11_KEY_NAMES:
            now = self._c11_head_local_to_key(key_name, head)
            before = self._c11_head_local_to_key(key_name, previous)
            if now is not None and before is not None:
                local_now[key_name] = now
                local_previous[key_name] = before

        if not self._c11_hit_armed:
            last = self._c11_last_hit_key
            local = local_now.get(str(last)) if last is not None else None
            if local is None:
                self._c11_hit_armed = True
                self._c11_current_hit_key = None
                self._c11_last_hit_key = None
                self._c11_pending_crossings = {}
            else:
                half = _C11_KEY_HALF_EXTENTS[str(last)]
                outside = bool(
                    abs(float(local[0])) > float(half[0]) + 0.015
                    or abs(float(local[1])) > float(half[1]) + 0.015
                )
                if float(local[2]) >= _C11_KEY_REARM_LOCAL_Z or outside:
                    self._c11_hit_armed = True
                    self._c11_current_hit_key = None
                    self._c11_last_hit_key = None
                    self._c11_pending_crossings = {}
            return

        candidates: list[tuple[float, str]] = []
        for key_name, local in local_now.items():
            before = local_previous[key_name]
            half = _C11_KEY_HALF_EXTENTS[key_name]
            in_xy = bool(
                abs(float(local[0]))
                <= float(half[0]) + _C11_KEY_HIT_XY_MARGIN
                and abs(float(local[1]))
                <= float(half[1]) + _C11_KEY_HIT_XY_MARGIN
            )
            downward = bool(
                float(before[2] - local[2])
                >= _C11_KEY_MIN_DOWNWARD_STEP
            )
            crossed = bool(
                float(before[2])
                > _C11_KEY_CONTACT_HEAD_Z + _C11_KEY_CONTACT_Z_TOLERANCE
                and float(local[2])
                <= _C11_KEY_CONTACT_HEAD_Z + _C11_KEY_CONTACT_Z_TOLERANCE
                and float(local[2])
                >= _C11_KEY_CONTACT_HEAD_Z
                - 2.0 * _C11_KEY_CONTACT_Z_TOLERANCE
            )
            if in_xy and downward and crossed:
                # Contact tensors describe the PhysX step that just finished,
                # while the assisted mallet root is written immediately
                # afterwards.  Retain a short geometric down-crossing only as
                # a prerequisite; live scoring still waits for real force.
                self._c11_pending_crossings[key_name] = 3
            physical_contact = self._c11_key_contact_force(key_name) > 1e-6
            contact_available = key_name in self._c11_key_contact_views
            head_in_contact_layer = bool(
                _C11_KEY_CONTACT_HEAD_Z
                - 2.0 * _C11_KEY_CONTACT_Z_TOLERANCE
                <= float(local[2])
                <= _C11_KEY_CONTACT_HEAD_Z
                + _C11_KEY_CONTACT_Z_TOLERANCE
            )
            if (
                in_xy
                and (
                    (
                        physical_contact
                        and head_in_contact_layer
                        and (
                            downward
                            or key_name in self._c11_pending_crossings
                        )
                    )
                    or (
                        not self._launch_isaac
                        and not contact_available
                        and downward
                        and crossed
                    )
                )
            ):
                normalized = float(
                    np.linalg.norm(
                        local[:2] / np.maximum(half[:2], 1e-9)
                    )
                )
                candidates.append((normalized, key_name))
        if not candidates:
            return
        _score, hit_key = min(candidates, key=lambda item: item[0])
        expected = self._c11_expected_key()
        if hit_key == expected:
            self._c11_sequence_progress += 1
        else:
            self._c11_wrong_hits += 1
            self._c11_sequence_progress = 0
        self._c11_last_hit_key = hit_key
        self._c11_current_hit_key = hit_key
        self._c11_hit_armed = False
        self._c11_pending_crossings = {}

    def _c11_xylophone_metrics(self) -> dict[str, Any]:
        mallet = self._poses.get("c11_mallet", self._fallback_pose("c11_mallet"))
        head = self._body_local_point_world("c11_mallet", _C11_MALLET_HEAD_LOCAL)
        best_error = math.inf
        for key_name in _C11_KEY_NAMES:
            local = self._local_point_in_body_frame(key_name, head)
            xy_error = float(np.linalg.norm(local[:2]))
            best_error = min(best_error, xy_error)
        return_zone = self._poses.get("c11_return_zone", self._fallback_pose("c11_return_zone"))
        expected_return_z = (
            self._current_table_top_z() + _C11_RETURN_CENTER_HEIGHT
        )
        velocity = self._physics_body_velocity("c11_mallet")
        return_stable = bool(
            velocity is None
            or (
                float(np.linalg.norm(velocity[:3]))
                <= _C11_RETURN_MAX_LINEAR_SPEED
                and float(np.linalg.norm(velocity[3:]))
                <= _C11_RETURN_MAX_ANGULAR_SPEED
            )
        )
        returned = bool(
            float(np.linalg.norm(mallet[:2] - return_zone[:2])) <= _C11_RETURN_XY_TOLERANCE
            and abs(float(mallet[2] - expected_return_z))
            <= _C11_RETURN_Z_TOLERANCE
            and return_stable
            and self._c11_mallet_grasp is None
        )
        sequence_complete = (
            self._c11_sequence_progress >= len(self._c11_target_sequence)
        )
        success_now = bool(sequence_complete and returned)
        return {
            "c11_sequence_complete": bool(sequence_complete),
            "c11_wrong_hits_ok": True,
            "c11_mallet_returned": bool(returned),
            "c11_success_now": bool(success_now),
            "c11_sequence_progress": int(self._c11_sequence_progress),
            "c11_sequence_required": int(len(self._c11_target_sequence)),
            "c11_wrong_hits": int(self._c11_wrong_hits),
            "c11_sequence_reset_on_error": True,
            "c11_target_sequence": tuple(self._c11_target_sequence),
            "c11_current_hit_key": self._c11_current_hit_key or "",
            "c11_best_hit_error": float(best_error),
            "c11_mallet_return_distance": float(np.linalg.norm(mallet[:2] - return_zone[:2])),
            "c11_mallet_return_z_error": abs(
                float(mallet[2] - expected_return_z)
            ),
            "c11_mallet_return_stable": bool(return_stable),
            "c11_mallet_return_released": self._c11_mallet_grasp is None,
        }

    def _c11_xylophone_sequence_success_now(self) -> bool:
        self._last_c11_success_metrics = self._c11_xylophone_metrics()
        completed = set(
            self._c11_target_sequence[: self._c11_sequence_progress]
        )
        for key_name in _C11_KEY_NAMES:
            self._set_task_visual_names_visible(
                {f"{key_name}_glow"},
                key_name in completed,
            )
        return bool(self._last_c11_success_metrics["c11_success_now"])

    def _c12_tool_grasp_contact(
        self,
        tool: np.ndarray,
        tip_positions: dict[str, np.ndarray] | None = None,
    ) -> tuple[bool, float, int]:
        tips = (
            self._b02_wuji_fingertip_positions()
            if tip_positions is None
            else tip_positions
        )
        return self._visible_segment_pinch_contact(
            tool,
            tips,
            side="right",
            segments_local=_C12_TOOL_SEGMENTS_LOCAL,
            segment_radii=_C12_TOOL_SEGMENT_RADII,
            contact_margin=_C12_TOOL_GRASP_CONTACT_MARGIN,
            near_margin=_C12_TOOL_GRASP_NEAR_MARGIN,
            pinch_distance_range=(
                _C12_TOOL_PINCH_MIN_DISTANCE,
                _C12_TOOL_PINCH_MAX_DISTANCE,
            ),
            max_axial_separation=0.050,
        )

    def _c12_ensure_slot_assignment(self) -> None:
        if len(self._c12_slot_by_block) == len(_C12_BLOCK_NAMES):
            return
        baseline = self._current_body_matrix("c12_baseline_pad")
        if baseline is None:
            return
        inverse = _invert_pose_matrix(baseline)
        sortable: list[tuple[float, str]] = []
        for name in _C12_BLOCK_NAMES:
            block = self._current_body_matrix(name)
            if block is None:
                return
            local = (
                inverse
                @ np.asarray([*block[:3, 3], 1.0], dtype=np.float64)
            )[:3]
            sortable.append((float(local[0]), name))
        sortable.sort(key=lambda item: item[0])
        self._c12_slot_by_block = {
            name: slot for slot, (_x, name) in enumerate(sortable)
        }

    def _c12_block_goal_matrix(
        self,
        name: str,
        current: np.ndarray,
    ) -> np.ndarray | None:
        self._c12_ensure_slot_assignment()
        slot = self._c12_slot_by_block.get(name)
        baseline = self._current_body_matrix("c12_baseline_pad")
        if slot is None or baseline is None:
            return None
        target = np.eye(4, dtype=np.float64)
        target[:3, 3] = (
            baseline
            @ np.asarray(
                [
                    float(_C12_BLOCK_LOCAL_X_CENTERS[int(slot)]),
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float64,
            )
        )[:3]
        target[2, 3] = (
            self._current_table_top_z() + _C12_BLOCK_HALF_EXTENTS[2]
        )
        z_axis = baseline[:3, 2].copy()
        z_axis /= max(float(np.linalg.norm(z_axis)), 1e-12)
        baseline_x = baseline[:3, 0].copy()
        baseline_x -= float(np.dot(baseline_x, z_axis)) * z_axis
        baseline_x /= max(float(np.linalg.norm(baseline_x)), 1e-12)
        rotations: list[np.ndarray] = []
        for sign in (1.0, -1.0):
            y_axis = sign * baseline_x
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
            rotation = np.column_stack([x_axis, y_axis, z_axis])
            rotations.append(rotation)
        current_matrix = np.asarray(current, dtype=np.float64).reshape(4, 4)
        target[:3, :3] = min(
            rotations,
            key=lambda rotation: float(
                np.linalg.norm(
                    self._rotation_error_vector(
                        rotation,
                        current_matrix[:3, :3],
                    )
                )
            ),
        )
        return target

    def _c12_tool_block_contact_metrics(
        self,
        tool: np.ndarray,
        block_pose: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        tool_matrix = np.asarray(tool, dtype=np.float64).reshape(4, 4)
        pose = np.asarray(block_pose, dtype=np.float64).reshape(7)
        best_gap = math.inf
        best_point = tool_matrix[:3, 3].copy()
        for segment, radius in zip(
            _C12_TOOL_SEGMENTS_LOCAL,
            _C12_TOOL_SEGMENT_RADII,
            strict=True,
        ):
            local_segment = np.asarray(segment, dtype=np.float64).reshape(2, 3)
            for fraction in np.linspace(0.0, 1.0, 13):
                local = (
                    (1.0 - float(fraction)) * local_segment[0]
                    + float(fraction) * local_segment[1]
                )
                world = (
                    tool_matrix
                    @ np.asarray([*local, 1.0], dtype=np.float64)
                )[:3]
                centerline_distance = self._point_to_block_surface_distance(
                    world,
                    pose,
                    _C12_BLOCK_HALF_EXTENTS,
                )
                gap = abs(float(centerline_distance) - float(radius))
                if gap < best_gap:
                    best_gap = gap
                    best_point = world.copy()
        return float(best_gap), best_point

    def _c12_tool_block_contact_force(self, name: str) -> tuple[float, bool]:
        view = self._c12_tool_block_contact_views.get(name)
        if view is None:
            return 0.0, False
        return self._c04_contact_force(view), True

    def _c12_push_contact_candidate(
        self,
        tool: np.ndarray,
        previous_tool: np.ndarray,
        *,
        movement_delta: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        valid: list[tuple[float, dict[str, Any]]] = []
        tool_matrix = np.asarray(tool, dtype=np.float64).reshape(4, 4)
        previous = np.asarray(
            previous_tool,
            dtype=np.float64,
        ).reshape(4, 4)
        tool_delta = (
            tool_matrix[:3, 3] - previous[:3, 3]
            if movement_delta is None
            else np.asarray(
                movement_delta,
                dtype=np.float64,
            ).reshape(3)
        )
        unsettled_names = tuple(
            name
            for name in _C12_BLOCK_NAMES
            if name not in self._c12_settled_blocks
        )
        contact_readings = {
            name: self._c12_tool_block_contact_force(name)
            for name in unsettled_names
        }
        if self._launch_isaac and any(
            not available
            for _force, available in contact_readings.values()
        ):
            return None
        if any(
            available
            for _force, available in contact_readings.values()
        ):
            physically_touched = {
                name
                for name, (force, available) in contact_readings.items()
                if available and force > 1e-6
            }
            # Reject simultaneous contacts before direction filtering.  A
            # second touched block cannot be hidden merely because its own
            # goal happens to point in another direction.
            if len(physically_touched) != 1:
                return None
        for name in _C12_BLOCK_NAMES:
            if name in self._c12_settled_blocks:
                continue
            block = self._current_body_matrix(name)
            if block is None:
                continue
            goal = self._c12_block_goal_matrix(name, block)
            if goal is None:
                continue
            direction = goal[:3, 3] - block[:3, 3]
            direction[2] = 0.0
            distance = float(np.linalg.norm(direction))
            if distance <= 1e-6:
                continue
            unit = direction / distance
            forward_step = float(np.dot(tool_delta, unit))
            if forward_step < _C12_PUSH_MIN_FORWARD_STEP:
                continue
            gap, contact_point = self._c12_tool_block_contact_metrics(
                tool_matrix,
                self._pose_matrix_to_vector(block),
            )
            if gap > _C12_TOOL_BLOCK_CONTACT_TOLERANCE:
                continue
            force, force_available = contact_readings.get(
                name,
                (0.0, False),
            )
            if self._launch_isaac and not force_available:
                # In a live collection run, geometric proximity alone can be
                # separated by a small gap or an unseen collider.  Fail closed
                # instead of recreating C08's old "ice floats beside the jaws"
                # failure mode.  Interface-only tests retain the fallback.
                continue
            if force_available and force <= 1e-6:
                continue
            behind = block[:3, 3] - contact_point
            behind[2] = 0.0
            behind_norm = float(np.linalg.norm(behind))
            if behind_norm <= 1e-6:
                continue
            alignment = float(np.dot(behind / behind_norm, unit))
            if alignment < _C12_PUSH_DIRECTION_ALIGNMENT:
                continue
            valid.append(
                (
                    gap,
                    {
                        "name": name,
                        "block": block.copy(),
                        "goal": goal.copy(),
                        "direction": unit.copy(),
                        "distance": distance,
                        "contact_gap": float(gap),
                        "contact_alignment": alignment,
                        "forward_step": forward_step,
                    },
                )
            )
        # A simultaneous two-block contact is intentionally rejected.  This is
        # the C12 equivalent of C08's "inner jaw only" rule.
        if len(valid) != 1:
            return None
        return valid[0][1]

    def _c12_set_block_kinematic(self, name: str, enabled: bool) -> bool:
        applied = self._set_task_body_kinematic(
            name,
            enabled,
            warning_label="C12 block",
        )
        return bool(applied or self._stage is None)

    def _c12_pin_settled_blocks(self) -> bool:
        for name, target_value in tuple(self._c12_settled_targets.items()):
            target = np.asarray(
                target_value,
                dtype=np.float64,
            ).reshape(4, 4)
            applied, _linear, _angular = self._set_body_rigid_grasp_target(
                name,
                target,
                previous_target=target,
            )
            if not applied:
                self._manual_reset_requested = True
                return False
        return True

    def _update_c12_tool_assist(self) -> None:
        if not self._c12_pin_settled_blocks():
            return
        palm = self._robot_link_pose_matrix("right")
        tool = self._current_body_matrix("c12_set_square")
        if palm is None or tool is None:
            return
        self._c12_ensure_slot_assignment()
        raw_hand = self._raw_right_hand_command_max()
        raw_tcp = _pose_vector_to_matrix(self._raw_tcp_pose("right"))
        state = self._c12_tool_grasp
        if state is not None and raw_hand <= _C12_TOOL_GRASP_RELEASE_HAND_COMMAND:
            self._preserve_released_hand_handoff(
                "right",
                state,
                palm,
            )
            self._c12_tool_grasp = None
            self._c12_active_push = None
            self._c12_push_candidate = None
            return
        if state is None:
            contact, error, count = self._c12_tool_grasp_contact(
                tool,
                self._b02_wuji_fingertip_positions(),
            )
            if (
                raw_hand >= _C12_TOOL_GRASP_ATTACH_HAND_COMMAND
                and contact
            ):
                state = {
                    "phase": "carry",
                    "hand_to_object": _invert_pose_matrix(palm) @ tool,
                    "last_target": tool.copy(),
                    "previous_carried": tool.copy(),
                    "previous_raw_tcp": raw_tcp.copy(),
                    "hand_target_matrix": None,
                    "hand_target_installed": False,
                    "visual_hand": self._actual_right_hand_joint_pose(),
                    "contact_error": float(error),
                    "contact_count": int(count),
                }
                self._c12_tool_grasp = state
            else:
                return

        carried = palm @ np.asarray(
            state["hand_to_object"],
            dtype=np.float64,
        ).reshape(4, 4)
        previous_carried = np.asarray(
            state.get("previous_carried", tool),
            dtype=np.float64,
        ).reshape(4, 4)
        control_frame = int(self._c_assist_control_frame)
        if int(state.get("raw_intent_frame", -1)) != control_frame:
            previous_raw_tcp = np.asarray(
                state.get("previous_raw_tcp", raw_tcp),
                dtype=np.float64,
            ).reshape(4, 4)
            state["raw_intent_delta"] = (
                raw_tcp[:3, 3] - previous_raw_tcp[:3, 3]
            )
            state["raw_intent_frame"] = control_frame
        raw_intent_delta = np.asarray(
            state.get("raw_intent_delta", np.zeros(3, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(3)
        active = self._c12_active_push
        if active is None:
            if str(state.get("phase")) == "carry_offset":
                handoff_target = self._raw_hand_handoff_target(
                    "right",
                    state,
                )
                if handoff_target is not None:
                    state["hand_target_matrix"] = handoff_target
            else:
                state["phase"] = "carry"
                state["hand_target_matrix"] = None
            candidate = self._c12_push_contact_candidate(
                tool,
                previous_carried,
                movement_delta=raw_intent_delta,
            )
            if candidate is None:
                self._c12_push_candidate = None
            else:
                pending = self._c12_push_candidate
                if (
                    pending is not None
                    and str(pending.get("name")) == str(candidate["name"])
                ):
                    candidate["steps"] = int(pending.get("steps", 1)) + 1
                else:
                    candidate["steps"] = 1
                self._c12_push_candidate = candidate
                if (
                    int(candidate["steps"])
                    >= _C12_TOOL_BLOCK_CONTACT_DEBOUNCE_STEPS
                ):
                    name = str(candidate["name"])
                    block = np.asarray(
                        candidate["block"],
                        dtype=np.float64,
                    ).reshape(4, 4)
                    self._c12_active_push = {
                        "name": name,
                        "block_start": block.copy(),
                        "goal": np.asarray(
                            candidate["goal"],
                            dtype=np.float64,
                        ).reshape(4, 4).copy(),
                        "direction": np.asarray(
                            candidate["direction"],
                            dtype=np.float64,
                        ).reshape(3).copy(),
                        "distance": float(candidate["distance"]),
                        "tool_to_block": _invert_pose_matrix(tool) @ block,
                        "progress": 0.0,
                        "command_progress": 0.0,
                        "raw_start_position": raw_tcp[:3, 3].copy(),
                        "last_tool_target": tool.copy(),
                        "last_block_target": block.copy(),
                        "stable_steps": 0,
                        "contact_gap": float(candidate["contact_gap"]),
                        "contact_alignment": float(
                            candidate["contact_alignment"]
                        ),
                    }
                    active = self._c12_active_push
                    state["phase"] = "guided_push"
                    state.pop("raw_to_hand_offset", None)
                    state.pop("raw_handoff_raw_anchor", None)
                    state.pop("raw_handoff_hand_anchor", None)
                    state["hand_target_matrix"] = (
                        tool
                        @ _invert_pose_matrix(
                            np.asarray(
                                state["hand_to_object"],
                                dtype=np.float64,
                            ).reshape(4, 4)
                        )
                    )
                    self._c12_push_candidate = None

            if active is None:
                applied, linear_error, angular_error = (
                    self._set_body_rigid_grasp_target(
                        "c12_set_square",
                        carried,
                        previous_target=state.get("last_target"),
                    )
                )
                state.update(
                    {
                        "applied": bool(applied),
                        "linear_error": float(linear_error),
                        "angular_error": float(angular_error),
                    }
                )
                if applied:
                    state["last_target"] = carried.copy()
                    state["previous_carried"] = carried.copy()
                    state["previous_raw_tcp"] = raw_tcp.copy()
                return

        active = self._c12_active_push
        if active is None:
            return
        state["phase"] = "guided_push"
        block_name = str(active["name"])
        block_current = self._current_body_matrix(block_name)
        if block_current is None:
            return
        committed_progress = float(active.get("progress", 0.0))
        command_progress = float(
            active.get("command_progress", committed_progress)
        )
        raw_start_value = active.get("raw_start_position")
        if raw_start_value is None:
            legacy_raw_tool = active.get("last_raw_tool")
            raw_start_value = (
                raw_tcp[:3, 3]
                if legacy_raw_tool is None
                else np.asarray(
                    legacy_raw_tool,
                    dtype=np.float64,
                ).reshape(4, 4)[:3, 3]
            )
            active["raw_start_position"] = np.asarray(
                raw_start_value,
                dtype=np.float64,
            ).reshape(3).copy()
        tracking_ready = self._assist_hand_tracking_ready(
            "right",
            state,
            position_tolerance=_C12_HAND_TRACK_POSITION_TOLERANCE,
            angle_tolerance=_C12_HAND_TRACK_ANGLE_TOLERANCE,
        )
        next_command_progress = command_progress
        if tracking_ready:
            raw_displacement = raw_tcp[:3, 3] - np.asarray(
                raw_start_value,
                dtype=np.float64,
            ).reshape(3)
            requested_distance = max(
                0.0,
                float(
                    np.dot(
                        raw_displacement,
                        np.asarray(active["direction"], dtype=np.float64),
                    )
                ),
            )
            requested_progress = min(
                1.0,
                requested_distance / max(float(active["distance"]), 1e-9),
            )
            max_delta = (
                _C12_PUSH_MAX_LINEAR_SPEED
                * max(float(self._control_dt), 0.0)
                / max(float(active["distance"]), 1e-9)
            )
            total_rotation = float(
                np.linalg.norm(
                    self._rotation_error_vector(
                        np.asarray(
                            active["goal"],
                            dtype=np.float64,
                        ).reshape(4, 4)[:3, :3],
                        np.asarray(
                            active["block_start"],
                            dtype=np.float64,
                        ).reshape(4, 4)[:3, :3],
                    )
                )
            )
            if total_rotation > 1e-9:
                max_delta = min(
                    max_delta,
                    _C12_PUSH_MAX_ANGULAR_SPEED
                    * max(float(self._control_dt), 0.0)
                    / total_rotation,
                )
            next_command_progress = min(
                1.0,
                max(
                    command_progress,
                    min(
                        requested_progress,
                        command_progress + max_delta,
                    ),
                ),
            )
        tool_to_block = np.asarray(
            active["tool_to_block"],
            dtype=np.float64,
        ).reshape(4, 4)
        # The square and block follow the actual palm together throughout the
        # guided push.  The next interpolated pose is only queued as an arm
        # command, so neither rigid body can move ahead of a stationary hand.
        tool_target = carried.copy()
        block_target = tool_target @ tool_to_block
        hand_to_object_inverse = _invert_pose_matrix(
            np.asarray(
                state["hand_to_object"],
                dtype=np.float64,
            ).reshape(4, 4)
        )
        current_hand_target = state.get("hand_target_matrix")
        last_tool_target = np.asarray(
            active["last_tool_target"],
            dtype=np.float64,
        ).reshape(4, 4)
        tool_applied, tool_linear, tool_angular = (
            self._set_body_rigid_grasp_target(
                "c12_set_square",
                tool_target,
                previous_target=last_tool_target,
            )
        )
        if not tool_applied:
            state.update(
                {
                    "applied": False,
                    "linear_error": float(tool_linear),
                    "angular_error": float(tool_angular),
                }
            )
            active.update(
                {
                    "block_applied": False,
                    "block_linear_error": math.inf,
                    "block_angular_error": math.inf,
                }
            )
            return
        block_applied, block_linear, block_angular = (
            self._set_body_rigid_grasp_target(
                block_name,
                block_target,
                previous_target=active.get("last_block_target"),
            )
        )
        if not block_applied:
            # Tool and block are one guided interaction while pushing.  Roll
            # the tool back when the block writer fails so the hand, square,
            # and block never split across two physical states.
            restored, _rollback_linear, _rollback_angular = (
                self._set_body_rigid_grasp_target(
                "c12_set_square",
                last_tool_target,
                previous_target=last_tool_target,
            )
            )
            if not restored:
                self._manual_reset_requested = True
                self._c12_active_push = None
                self._c12_push_candidate = None
            state.update(
                {
                    "applied": False,
                    "linear_error": float(tool_linear),
                    "angular_error": float(tool_angular),
                }
            )
            active.update(
                {
                    "block_applied": False,
                    "block_linear_error": float(block_linear),
                    "block_angular_error": float(block_angular),
                }
            )
            return
        state.update(
            {
                "applied": True,
                "linear_error": float(tool_linear),
                "angular_error": float(tool_angular),
            }
        )
        active.update(
            {
                "block_applied": True,
                "block_linear_error": float(block_linear),
                "block_angular_error": float(block_angular),
            }
        )
        position_delta = (
            block_target[:3, 3]
            - np.asarray(
                active["block_start"],
                dtype=np.float64,
            ).reshape(4, 4)[:3, 3]
        )
        achieved_progress = float(
            np.clip(
                np.dot(
                    position_delta,
                    np.asarray(active["direction"], dtype=np.float64),
                )
                / max(float(active["distance"]), 1e-9),
                0.0,
                max(command_progress, committed_progress),
            )
        )
        active["progress"] = achieved_progress
        active["last_tool_target"] = tool_target.copy()
        active["last_block_target"] = block_target.copy()
        state["last_target"] = tool_target.copy()
        state["previous_carried"] = carried.copy()
        if tracking_ready:
            command_block_target = self._c07_interpolate_pose(
                active["block_start"],
                active["goal"],
                next_command_progress,
            )
            command_tool_target = (
                command_block_target @ _invert_pose_matrix(tool_to_block)
            )
            state["hand_target_matrix"] = (
                command_tool_target @ hand_to_object_inverse
            )
            active["command_progress"] = next_command_progress
        elif current_hand_target is not None:
            state["hand_target_matrix"] = current_hand_target

        final_reached = bool(
            command_progress >= 1.0 - 1e-9
            and tracking_ready
            and self._tool_lock_target_reached(
                block_target,
                np.asarray(active["goal"], dtype=np.float64),
            )
        )
        if final_reached:
            active["stable_steps"] = int(active.get("stable_steps", 0)) + 1
        else:
            active["stable_steps"] = 0
        if int(active["stable_steps"]) < _C12_SETTLE_HOLD_STEPS:
            return
        if block_name in self._task_articulation_views:
            self._c12_settled_targets[block_name] = block_target.copy()
        elif not self._c12_set_block_kinematic(block_name, True):
            return
        self._c12_settled_blocks.add(block_name)
        # Continue from the final synchronized pose without jumping back to the
        # raw tracker frame used before the assisted path.
        state["hand_to_object"] = _invert_pose_matrix(palm) @ tool_target
        state["last_target"] = tool_target.copy()
        state["previous_carried"] = tool_target.copy()
        state["previous_raw_tcp"] = raw_tcp.copy()
        # Do not let the just-consumed glove delta survive into the remaining
        # PhysX substeps of this same control frame and arm a neighbouring
        # block without a new operator motion.
        state["raw_intent_delta"] = np.zeros(3, dtype=np.float64)
        state["raw_intent_frame"] = int(self._c_assist_control_frame)
        self._c12_push_candidate = None
        hand_target = (
            tool_target
            @ _invert_pose_matrix(
                np.asarray(
                    state["hand_to_object"],
                    dtype=np.float64,
                ).reshape(4, 4)
            )
        )
        self._start_raw_hand_handoff(
            "right",
            state,
            hand_target,
        )
        state["phase"] = "carry_offset"
        self._c12_active_push = None

    def _c12_direct_hand_touch_now(self) -> bool:
        if self._c12_right_hand_block_contact_force() > 1e-6:
            return True
        block_poses = []
        for name in _C12_BLOCK_NAMES:
            pose = self._physics_body_pose(name)
            if pose is None:
                pose = self._poses.get(name, self._fallback_pose(name))
            block_poses.append(np.asarray(pose, dtype=np.float64))
        return any(
            self._point_to_block_surface_distance(
                point,
                pose,
                _C12_BLOCK_HALF_EXTENTS,
            )
            <= _C12_DIRECT_TOUCH_DISTANCE
            for point in self._a06_hand_points("right")
            for pose in block_poses
        )

    @staticmethod
    def _clip_polygon_to_axis_aligned_box(
        polygon: np.ndarray,
        half_extents: np.ndarray,
    ) -> np.ndarray:
        clipped = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        for axis in range(2):
            for bound, keep_less in (
                (float(half_extents[axis]), True),
                (-float(half_extents[axis]), False),
            ):
                if len(clipped) == 0:
                    return clipped
                output: list[np.ndarray] = []
                previous = clipped[-1]
                previous_inside = (
                    float(previous[axis]) <= bound
                    if keep_less
                    else float(previous[axis]) >= bound
                )
                for current in clipped:
                    current_inside = (
                        float(current[axis]) <= bound
                        if keep_less
                        else float(current[axis]) >= bound
                    )
                    if current_inside != previous_inside:
                        denominator = float(current[axis] - previous[axis])
                        if abs(denominator) > 1e-12:
                            fraction = (bound - float(previous[axis])) / denominator
                            output.append(
                                previous + fraction * (current - previous)
                            )
                    if current_inside:
                        output.append(current)
                    previous = current
                    previous_inside = current_inside
                clipped = np.asarray(output, dtype=np.float64).reshape(-1, 2)
        return clipped

    @staticmethod
    def _polygon_area(polygon: np.ndarray) -> float:
        vertices = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        if len(vertices) < 3:
            return 0.0
        return 0.5 * abs(
            float(
                np.dot(vertices[:, 0], np.roll(vertices[:, 1], -1))
                - np.dot(vertices[:, 1], np.roll(vertices[:, 0], -1))
            )
        )

    @classmethod
    def _c12_block_target_overlap_ratio(
        cls,
        block_pose: np.ndarray,
        baseline_pose: np.ndarray,
        target_local_x: float,
    ) -> float:
        block_matrix = _pose_vector_to_matrix(block_pose)
        baseline_inverse = _invert_pose_matrix(
            _pose_vector_to_matrix(baseline_pose)
        )
        half_x, half_y = _C12_BLOCK_HALF_EXTENTS[:2]
        local_corners = np.asarray(
            [
                [-half_x, -half_y, 0.0, 1.0],
                [half_x, -half_y, 0.0, 1.0],
                [half_x, half_y, 0.0, 1.0],
                [-half_x, half_y, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        corners_world = (block_matrix @ local_corners.T).T
        corners_baseline = (baseline_inverse @ corners_world.T).T[:, :2]
        target_centered = corners_baseline - np.asarray(
            [target_local_x, 0.0],
            dtype=np.float64,
        )
        intersection = cls._clip_polygon_to_axis_aligned_box(
            target_centered,
            _C12_TARGET_HALF_EXTENTS_XY,
        )
        block_area = 4.0 * float(half_x) * float(half_y)
        return float(np.clip(cls._polygon_area(intersection) / block_area, 0.0, 1.0))

    def _c12_push_line_metrics(self) -> dict[str, Any]:
        baseline = self._poses.get("c12_baseline_pad", self._fallback_pose("c12_baseline_pad"))
        tool = self._poses.get("c12_set_square", self._fallback_pose("c12_set_square"))
        self._c12_ensure_slot_assignment()
        if self._c12_tool_start_xy is None:
            self._c12_tool_start_xy = tool[:2].copy()
        tool_move = float(np.linalg.norm(tool[:2] - self._c12_tool_start_xy))
        if tool_move >= _C12_TOOL_MIN_MOVE_DISTANCE:
            self._c12_tool_used = True
        block_poses = [self._poses.get(name, self._fallback_pose(name)) for name in _C12_BLOCK_NAMES]
        baseline_matrix = _pose_vector_to_matrix(baseline)
        baseline_inverse = _invert_pose_matrix(baseline_matrix)
        block_local_positions = [
            (baseline_inverse @ np.asarray([*pose[:3], 1.0], dtype=np.float64))[:3]
            for pose in block_poses
        ]
        line_errors = [abs(float(local[1])) for local in block_local_positions]
        on_line = all(error <= _C12_LINE_DISTANCE_TOLERANCE for error in line_errors)
        baseline_axis = baseline_matrix[:3, 0]
        alignments = [
            abs(self._axis_alignment(pose, 1, baseline_axis))
            for pose in block_poses
        ]
        blocks_aligned = all(value >= _C12_ORIENTATION_ALIGNMENT for value in alignments)
        xs = sorted(float(local[0]) for local in block_local_positions)
        gaps = [xs[index + 1] - xs[index] for index in range(len(xs) - 1)]
        spacing_ok = all(_C12_SPACING_RANGE[0] <= gap <= _C12_SPACING_RANGE[1] for gap in gaps)
        overlap_ratios = [0.0] * len(_C12_BLOCK_NAMES)
        slot_mapping_complete = bool(
            set(self._c12_slot_by_block) == set(_C12_BLOCK_NAMES)
            and set(self._c12_slot_by_block.values())
            == set(range(len(_C12_BLOCK_NAMES)))
        )
        for block_index, name in enumerate(_C12_BLOCK_NAMES):
            slot = self._c12_slot_by_block.get(name)
            if slot is None:
                continue
            overlap_ratios[block_index] = self._c12_block_target_overlap_ratio(
                block_poses[block_index],
                baseline,
                float(_C12_BLOCK_LOCAL_X_CENTERS[int(slot)]),
            )
        targets_overlap = all(
            ratio > _C12_MIN_TARGET_OVERLAP_RATIO
            for ratio in overlap_ratios
        ) and slot_mapping_complete
        min_hand_distance = math.inf
        for side in ("right",):
            for point in self._a06_hand_points(side):
                for pose in block_poses:
                    min_hand_distance = min(
                        min_hand_distance,
                        self._point_to_block_surface_distance(point, pose, _C12_BLOCK_HALF_EXTENTS),
                    )
        no_direct_grasp = bool(
            not self._c12_direct_hand_contact
            and min_hand_distance >= _C12_HAND_BLOCK_CLEARANCE
        )
        success_now = bool(
            targets_overlap
            and on_line
            and blocks_aligned
            and spacing_ok
            and self._c12_tool_used
            and no_direct_grasp
        )
        return {
            "c12_targets_overlap": bool(targets_overlap),
            "c12_blocks_on_line": bool(on_line),
            "c12_blocks_aligned": bool(blocks_aligned),
            "c12_spacing_ok": bool(spacing_ok),
            "c12_tool_used": bool(self._c12_tool_used),
            "c12_no_direct_grasp": bool(no_direct_grasp),
            "c12_direct_hand_contact": bool(self._c12_direct_hand_contact),
            "c12_success_now": bool(success_now),
            "c12_line_errors": [float(value) for value in line_errors],
            "c12_target_overlap_ratios": [
                float(value) for value in overlap_ratios
            ],
            "c12_slot_by_block": dict(self._c12_slot_by_block),
            "c12_orientation_alignments": [float(value) for value in alignments],
            "c12_spacing_gaps": [float(value) for value in gaps],
            "c12_tool_move_distance": float(tool_move),
            "c12_min_hand_block_distance": float(min_hand_distance),
        }

    def _t11_line_up_blocks_success_now(self) -> bool:
        self._last_c12_success_metrics = self._c12_push_line_metrics()
        overlap_ratios = self._last_c12_success_metrics[
            "c12_target_overlap_ratios"
        ]
        for index, overlap_ratio in enumerate(overlap_ratios):
            aligned = bool(overlap_ratio > _C12_MIN_TARGET_OVERLAP_RATIO)
            self._set_task_visual_colors(
                {
                    f"c12_block_{index}_visual": (
                        (0.08, 0.85, 0.25) if aligned else (0.58, 0.38, 0.18)
                    )
                }
            )
        return bool(self._last_c12_success_metrics["c12_success_now"])

    def _success_steps_required(self) -> int:
        required_steps = max(1, int(self._spec.success_steps))
        if self.task_name == "t04_cover_pitcher":
            return 1
        if self.task_name == "t19_orient_cup_upright":
            hold_steps = int(math.ceil(_A06_RELEASE_HOLD_SECONDS / max(self._control_dt, 1e-6)))
            required_steps = max(required_steps, hold_steps)
        if self.task_name == "t09_insert_peg":
            hold_steps = int(
                math.ceil(
                    _B03_STABLE_HOLD_SECONDS
                    / max(float(self._control_dt), 1e-6)
                )
            )
            required_steps = max(required_steps, hold_steps)
        if self.task_name == "t44_stack_blocks":
            hold_steps = int(math.ceil(_STACK_RELEASE_STABLE_SECONDS / max(self._control_dt, 1e-6)))
            required_steps = max(required_steps, hold_steps)
        if self.task_name == "c07_pitcher_pour_cup":
            hold_steps = int(
                math.ceil(_C07_RELEASE_HOLD_SECONDS / max(self._control_dt, 1e-6))
            )
            required_steps = max(required_steps, hold_steps)
        return required_steps

    def _compute_success_now(self) -> bool:
        if self.task_name == "t26_place_fruit_in_bucket":
            return self._target_on_bucket_bottom()
        if self.task_name == "t25_place_cup_on_plate":
            return self._target_on_tray_center()
        if self.task_name == "t24_place_ball_in_bowl":
            return self._target_inside_bowl_precise()
        if self.task_name == "t19_orient_cup_upright":
            return self._cup_bottom_down()
        if self.task_name in {"t04_cover_pitcher", "t03_cover_box"}:
            return self._lid_fit_success()
        if self.task_name == "b02_nut_bolt_assembly":
            return self._nut_bolt_success()
        if self.task_name == "t09_insert_peg":
            return self._peg_socket_success()
        if self.task_name == "t44_stack_blocks":
            return self._block_stack_stable()
        if self.task_name == "t08_hammer_nail":
            return self._t08_hammer_nail_success()
        if self.task_name == "t36_scoop_marbles":
            return self._spoon_marbles_in_bowl()
        if self.task_name == "t50_water_plant":
            return self._t50_water_plant_success_now()
        if self.task_name == "t14_move_and_click_mouse":
            return self._t14_move_and_click_mouse_success_now()
        if self.task_name == "t23_photograph_logo":
            return self._t23_photograph_logo_success_now()
        if self.task_name == "c07_pitcher_pour_cup":
            return self._c07_pitcher_pour_cup_success_now()
        if self.task_name == "c08_tongs_ice_cubes":
            return self._c08_tongs_ice_cubes_success_now()
        if self.task_name == "c09_staple_paper_sheets":
            return self._c09_staple_paper_sheets_success_now()
        if self.task_name == "t02_connect_charger_and_cable":
            return self._t02_connect_charger_and_cable_success_now()
        if self.task_name == "c11_xylophone_sequence":
            return self._c11_xylophone_sequence_success_now()
        if self.task_name == "t11_line_up_blocks":
            return self._t11_line_up_blocks_success_now()

        return False

    def _pose_matrix_for_name(self, name: str) -> np.ndarray:
        pose = self._poses.get(name, self._fallback_pose(name))
        return _pose_vector_to_matrix(pose)

    def _target_local_to_receptacle(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target = self._pose_for_role("target")
        receptacle = self._pose_for_role("receptacle")
        receptacle_matrix = _pose_vector_to_matrix(receptacle)
        target_matrix = _pose_vector_to_matrix(target)
        local = _invert_pose_matrix(receptacle_matrix) @ target_matrix
        return local[:3, 3], target, receptacle

    def _z_axis_alignment(self, pose: np.ndarray, reference_pose: np.ndarray | None = None) -> float:
        rot = _quat_wxyz_to_matrix(np.asarray(pose, dtype=np.float64)[3:7])
        if reference_pose is None:
            reference_axis = _WORLD_UP
        else:
            reference_axis = _quat_wxyz_to_matrix(np.asarray(reference_pose, dtype=np.float64)[3:7])[:, 2]
        return float(np.dot(rot[:, 2], reference_axis))

    def _x_axis_abs_alignment(self, pose: np.ndarray, reference_pose: np.ndarray) -> float:
        rot = _quat_wxyz_to_matrix(np.asarray(pose, dtype=np.float64)[3:7])
        ref_rot = _quat_wxyz_to_matrix(np.asarray(reference_pose, dtype=np.float64)[3:7])
        return abs(float(np.dot(rot[:, 0], ref_rot[:, 0])))

    def _b01_hand_lid_clear_steps_required(self) -> int:
        return max(
            1,
            int(
                math.ceil(
                    _B01_HAND_LID_CLEAR_HOLD_SECONDS
                    / max(float(self._control_dt), 1e-6)
                )
            ),
        )

    def _b01_hand_lid_clearance(self, lid: np.ndarray) -> tuple[float, bool]:
        lid_matrix = _pose_vector_to_matrix(lid)
        lid_inv = _invert_pose_matrix(lid_matrix)
        min_distance = math.inf
        for side in ("right", "left") if self._spec.bimanual else ("right",):
            for point in self._a06_hand_points(side):
                point_local = lid_inv @ np.asarray(
                    [point[0], point[1], point[2], 1.0],
                    dtype=np.float64,
                )
                radial_outside = max(
                    0.0,
                    float(np.linalg.norm(point_local[:2]))
                    - _B01_HAND_LID_CLEARANCE_RADIUS,
                )
                z_outside = max(
                    0.0,
                    abs(float(point_local[2])) - _B01_HAND_LID_CLEARANCE_HALF_HEIGHT,
                )
                distance = float(np.linalg.norm([radial_outside, z_outside]))
                min_distance = min(min_distance, distance)
        hand_clear = bool(min_distance >= _B01_HAND_LID_CONTACT_DISTANCE)
        return min_distance, hand_clear

    def _b01_lid_insert_metrics(
        self,
        lid: np.ndarray,
        base: np.ndarray,
        *,
        update_hand_clear_hold: bool,
    ) -> dict[str, Any]:
        base_matrix = _pose_vector_to_matrix(base)
        lid_matrix = _pose_vector_to_matrix(lid)
        base_inv = _invert_pose_matrix(base_matrix)

        local_points: dict[str, np.ndarray] = {}
        for label, local_z in (
            ("bottom", _B01_LID_INSERT_BOTTOM_LOCAL_Z),
            ("center", _B01_LID_INSERT_CENTER_LOCAL_Z),
            ("top", _B01_LID_INSERT_TOP_LOCAL_Z),
        ):
            point_world = lid_matrix @ np.asarray(
                [0.0, 0.0, local_z, 1.0],
                dtype=np.float64,
            )
            local_points[label] = base_inv @ point_world

        max_axis_xy_error = max(
            float(np.linalg.norm(point[:2])) for point in local_points.values()
        )
        radial_capacity = max(
            0.0,
            _B01_PITCHER_MOUTH_INNER_RADIUS - _B01_LID_INSERT_RADIUS,
        )
        radial_tolerance = radial_capacity + _B01_LID_INSERT_XY_TOLERANCE
        xy_close = max_axis_xy_error <= radial_tolerance
        top_z_error = float(local_points["top"][2] - _B01_PITCHER_LID_SEAT_LOCAL_Z)
        insert_inside = bool(xy_close and top_z_error <= _B01_LID_INSERT_Z_ABOVE_TOLERANCE)
        z_close = (
            -_B01_LID_INSERT_Z_BELOW_TOLERANCE
            <= top_z_error
            <= _B01_LID_INSERT_Z_ABOVE_TOLERANCE
        )
        top_aligned = self._z_axis_alignment(lid, base) >= float(
            np.cos(np.deg2rad(_B01_LID_SUCCESS_TOP_ANGLE_DEG))
        )

        if self._lid_start_pose is None:
            self._lid_start_pose = lid.copy()
        move_distance = float(np.linalg.norm(lid[:3] - self._lid_start_pose[:3]))
        lift_height = float(lid[2] - self._lid_start_pose[2])
        if (
            move_distance >= _LID_MIN_MOVE_DISTANCE
            or lift_height >= _LID_MIN_LIFT_HEIGHT
        ):
            self._lid_was_moved = True

        velocity = self._physics_body_velocity(self._target_object_name)
        stable = velocity is None or (
            float(np.linalg.norm(velocity[:3])) <= _LID_STABLE_LINEAR_SPEED
            and float(np.linalg.norm(velocity[3:])) <= _LID_STABLE_ANGULAR_SPEED
        )
        hand_min_distance, hand_clear = self._b01_hand_lid_clearance(lid)
        ready_for_release_hold = bool(
            self._lid_was_moved
            and insert_inside
            and z_close
            and top_aligned
            and stable
        )
        if update_hand_clear_hold:
            if ready_for_release_hold and hand_clear:
                self._b01_hand_lid_clear_steps += 1
            else:
                self._b01_hand_lid_clear_steps = 0
        hand_clear_steps_required = self._b01_hand_lid_clear_steps_required()
        hand_clear_held = bool(
            self._b01_hand_lid_clear_steps >= hand_clear_steps_required
        )
        success_now = bool(ready_for_release_hold and hand_clear_held)
        return {
            "b01_lid_insert_inside": bool(insert_inside),
            "b01_lid_insert_xy_close": bool(xy_close),
            "b01_lid_z_close": bool(z_close),
            "b01_lid_top_aligned": bool(top_aligned),
            "b01_lid_moved": bool(self._lid_was_moved),
            "b01_lid_stable": bool(stable),
            "b01_hand_clear": bool(hand_clear),
            "b01_hand_clear_held": bool(hand_clear_held),
            "b01_success_now": bool(success_now),
            "b01_insert_max_axis_xy_error": float(max_axis_xy_error),
            "b01_insert_radial_tolerance": float(radial_tolerance),
            "b01_insert_top_z_error": float(top_z_error),
            "b01_inserted_goal_local_z": float(_B01_LID_INSERTED_GOAL_LOCAL_Z),
            "b01_hand_min_distance": float(hand_min_distance),
            "b01_hand_contact_distance": float(_B01_HAND_LID_CONTACT_DISTANCE),
            "b01_hand_clear_steps": int(self._b01_hand_lid_clear_steps),
            "b01_hand_clear_steps_required": int(hand_clear_steps_required),
            "b01_hand_clear_hold_seconds": float(_B01_HAND_LID_CLEAR_HOLD_SECONDS),
        }

    def _target_on_bucket_bottom(self) -> bool:
        local, _target, _bucket = self._target_local_to_receptacle()
        # The bucket asset is authored on its side: local X/Z span the bottom
        # and local Y is its vertical axis. Match the eight MuJoCo bucket_ref
        # sites instead of assuming local Z is up.
        horizontal_margin = 0.020
        horizontal_ok = bool(
            -0.18 + horizontal_margin <= float(local[0]) <= 0.16 - horizontal_margin
            and -0.13 + horizontal_margin <= float(local[2]) <= 0.12 - horizontal_margin
        )
        object_half_height = _OBJECT_HALF_HEIGHTS.get(self._target_object_name, 0.04)
        bottom_clearance = float(local[1] - (-0.08 + object_half_height))
        return bool(horizontal_ok and -0.020 <= bottom_clearance <= 0.055)

    def _target_inside_bowl_precise(self) -> bool:
        local, _target, _bowl = self._target_local_to_receptacle()
        radial = float(np.linalg.norm(local[:2]))
        return bool(radial <= 0.095 and 0.010 <= float(local[2]) <= 0.115)

    def _target_on_tray_center(self) -> bool:
        target = self._pose_for_role("target")
        tray = self._pose_for_role("receptacle")
        target_is_brown_cup = self._target_object_name == "025_mug"
        xy_error = float(np.linalg.norm(target[:2] - tray[:2]))
        z_above = float(target[2] - tray[2])
        bottom_down = self._z_axis_alignment(target) >= float(np.cos(np.deg2rad(25.0)))
        not_dropped = target[2] >= self._current_table_top_z() - 0.02
        velocity = self._physics_body_velocity(self._target_object_name)
        stable = velocity is None or (
            float(np.linalg.norm(velocity[:3])) <= 0.025
            and float(np.linalg.norm(velocity[3:])) <= 0.35
        )
        return bool(
            target_is_brown_cup
            and xy_error <= 0.065
            and 0.015 <= z_above <= 0.10
            and bottom_down
            and not_dropped
            and stable
        )

    def _cup_bottom_down(self) -> bool:
        metrics = self._a06_cup_success_metrics()
        self._last_a06_success_metrics = metrics
        return bool(metrics["a06_success_now"])

    def _a06_cup_success_metrics(self) -> dict[str, Any]:
        cup = self._pose_for_role("target")
        # The a_cup mesh opening points along local +Z, so a positive alignment
        # is required.  Taking the absolute value also accepts an upside-down
        # cup, which contradicts the A06 prompt and records false successes.
        opening_alignment = self._z_axis_alignment(cup)
        opening_angle_deg = float(np.rad2deg(np.arccos(np.clip(opening_alignment, -1.0, 1.0))))
        opening_up = opening_alignment >= float(
            np.cos(np.deg2rad(_A06_CUP_OPENING_UP_MAX_ANGLE_DEG))
        )
        expected_z = self._current_table_top_z() + _OBJECT_HALF_HEIGHTS.get(
            "a_cup", 0.031
        )
        center_z_error = float(cup[2] - expected_z)
        on_table = abs(center_z_error) <= _A06_CUP_TABLE_HEIGHT_TOLERANCE
        hand_min_distance, hand_contact_free = self._a06_hand_clearance(cup)
        success_now = bool(opening_up and on_table and hand_contact_free)
        return {
            "a06_success_now": success_now,
            "a06_pose_upright": bool(opening_up),
            "a06_opening_up": bool(opening_up),
            "a06_on_table": bool(on_table),
            "a06_hand_clear": bool(hand_contact_free),
            "a06_hand_contact_free": bool(hand_contact_free),
            "a06_opening_alignment": float(opening_alignment),
            "a06_opening_angle_deg": opening_angle_deg,
            "a06_center_z": float(cup[2]),
            "a06_expected_center_z": float(expected_z),
            "a06_center_z_error": center_z_error,
            "a06_hand_min_distance": float(hand_min_distance),
            "a06_hand_contact_distance": float(_A06_HAND_CONTACT_DISTANCE),
            "a06_hand_clearance_distance": float(_A06_HAND_CONTACT_DISTANCE),
            "a06_opening_up_max_angle_deg": float(_A06_CUP_OPENING_UP_MAX_ANGLE_DEG),
            "a06_table_height_tolerance": float(_A06_CUP_TABLE_HEIGHT_TOLERANCE),
            "a06_release_hold_seconds": float(_A06_RELEASE_HOLD_SECONDS),
        }

    def _a06_hand_clearance(self, cup_pose: np.ndarray) -> tuple[float, bool]:
        cup_position = np.asarray(cup_pose[:3], dtype=np.float64)
        min_distance = math.inf
        sides = ["right"]
        if self._spec.bimanual or "left" in self._robots:
            sides.append("left")
        for side in sides:
            for point in self._a06_hand_points(side):
                distance = float(np.linalg.norm(point - cup_position))
                min_distance = min(min_distance, distance)
        return min_distance, bool(min_distance >= _A06_HAND_CONTACT_DISTANCE)

    def _a06_hand_points(self, side: str) -> tuple[np.ndarray, ...]:
        points: list[np.ndarray] = []
        robot = self._robots.get(side)
        if robot is not None:
            try:
                body_names = list(robot.body_names)
                body_poses = robot.data.body_pose_w[0]
                for index, name in enumerate(body_names):
                    normalized = str(name).lower()
                    if "palm" not in normalized and "finger" not in normalized:
                        continue
                    point = body_poses[int(index), :3]
                    if hasattr(point, "detach"):
                        point = point.detach().cpu().numpy()
                    point_np = np.asarray(point, dtype=np.float64).reshape(3)
                    if np.all(np.isfinite(point_np)):
                        points.append(point_np)
            except Exception:
                points = []
        if points:
            return tuple(points)

        fallback = self._left_tcp_pose if side == "left" else self._right_tcp_pose
        matrix = self._native_end_effector_pose_matrix(side, fallback)
        if matrix is None:
            return ()
        point_np = np.asarray(matrix[:3, 3], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(point_np)):
            return ()
        return (point_np,)

    def _task_info(self) -> dict[str, Any]:
        if self.task_name == "t08_hammer_nail":
            info = {
                "c01_had_valid_hit": bool(self._c01_had_valid_hit),
                "c01_max_valid_hit_depth": float(
                    self._c01_max_valid_hit_depth
                ),
                "c01_hit_contact_available": bool(
                    self._c01_hit_contact_available
                ),
                "t08_hammer_nail_contact_force": float(
                    self._c01_last_hit_contact_force
                ),
                "t08_hammer_nail_contact_force_threshold": float(
                    _C01_HAMMER_NAIL_CONTACT_FORCE
                ),
                "c01_face_nail_distance": float(
                    self._c01_last_face_nail_distance
                ),
                "c01_face_nail_distance_threshold": float(
                    _C01_MAX_FACE_NAIL_DISTANCE
                ),
                "c01_face_nail_radial_error": float(
                    self._c01_last_face_nail_radial_error
                ),
                "c01_face_nail_radial_error_threshold": float(
                    _C01_MAX_FACE_NAIL_RADIAL_ERROR
                ),
                "c01_face_nail_axial_offset": float(
                    self._c01_last_face_nail_axial_offset
                ),
                "c01_face_nail_axial_offset_min": float(
                    _C01_MIN_FACE_NAIL_AXIAL_OFFSET
                ),
                "c01_face_nail_axial_offset_max": float(
                    _C01_MAX_FACE_NAIL_AXIAL_OFFSET
                ),
                "c01_face_down_alignment": float(
                    self._c01_last_face_down_alignment
                ),
                "c01_face_down_alignment_threshold": float(
                    _C01_MIN_FACE_DOWN_ALIGNMENT
                ),
                "c01_hammer_downward_speed": float(
                    self._c01_last_hammer_downward_speed
                ),
                "c01_hammer_downward_speed_threshold": float(
                    _C01_MIN_HAMMER_SPEED
                ),
                "c01_hammer_motion_down_alignment": float(
                    self._c01_last_hammer_motion_down_alignment
                ),
                "c01_hammer_motion_down_alignment_threshold": float(
                    _C01_MIN_MOTION_DOWN_ALIGNMENT
                ),
                "c01_valid_hit_depth_increment": float(
                    self._c01_last_valid_hit_depth_increment
                ),
                "c01_candidate_hit_depth_increment": float(
                    self._c01_last_candidate_hit_depth_increment
                ),
                "c01_success_counter": int(self._success_counter),
                "c01_success_steps_required": int(
                    self._success_steps_required()
                ),
            }
            info.update(self._c01_strike_align_assist_metrics)
            info.update(self._c01_grasp_assist_info())
            return info
        if self.task_name == "t50_water_plant":
            info = dict(self._last_c04_success_metrics)
            info["c04_success_counter"] = int(self._success_counter)
            info["c04_success_steps_required"] = int(
                self._success_steps_required()
            )
            info["c04_trigger_pull_threshold"] = float(_C04_TRIGGER_PULL_RAD)
            info["c04_trigger_release_threshold"] = float(
                _C04_TRIGGER_RELEASE_RAD
            )
            info["c04_trigger_stiffness"] = float(_C04_TRIGGER_STIFFNESS)
            info["c04_trigger_damping"] = float(_C04_TRIGGER_DAMPING)
            info["c04_trigger_friction"] = float(_C04_TRIGGER_FRICTION)
            info["c04_trigger_max_force"] = float(_C04_TRIGGER_MAX_FORCE)
            info["c04_trigger_arm_steps"] = int(_C04_TRIGGER_ARM_STEPS)
            info["c04_trigger_armed"] = bool(self._c04_trigger_armed)
            return info
        if self.task_name == "t14_move_and_click_mouse":
            info = dict(
                self._last_c05_success_metrics
                or self._c05_mouse_click_metrics(update_failure=False)
            )
            info["c05_success_counter"] = int(self._success_counter)
            info["c05_success_steps_required"] = int(
                self._success_steps_required()
            )
            info["c05_min_mouse_move_distance"] = float(
                _C05_MIN_MOUSE_MOVE_DISTANCE
            )
            info["c05_button_press_threshold"] = float(
                _C05_BUTTON_PRESS_THRESHOLD
            )
            return info
        if self.task_name == "t23_photograph_logo":
            info = dict(
                self._last_c06_success_metrics or self._c06_photograph_metrics()
            )
            info["c06_success_counter"] = int(self._success_counter)
            info["c06_success_steps_required"] = int(
                self._success_steps_required()
            )
            info["c06_shutter_press_threshold"] = float(
                _C06_SHUTTER_PRESS_TRAVEL
            )
            info["c06_right_hand_contact_force_threshold"] = float(
                _C06_SHUTTER_RIGHT_CONTACT_FORCE
            )
            info["c06_aim_max_angle_deg"] = float(_C06_AIM_MAX_ANGLE_DEG)
            info["c06_aim_max_ray_error"] = float(_C06_AIM_MAX_RAY_ERROR)
            return info
        if self.task_name == "c07_pitcher_pour_cup":
            info = dict(self._last_c07_success_metrics or self._c07_pitcher_pour_metrics())
            info["c07_success_counter"] = int(self._success_counter)
            info["c07_success_steps_required"] = int(self._success_steps_required())
            grasp = self._c07_pitcher_grasp or {}
            info["c07_grasp_assist_active"] = bool(self._c07_pitcher_grasp)
            info["c07_grasp_assist_linear_error"] = float(
                grasp.get("linear_error", 0.0)
            )
            info["c07_grasp_assist_angular_error"] = float(
                grasp.get("angular_error", 0.0)
            )
            info["c07_assist_phase"] = str(grasp.get("phase", "idle"))
            info["c07_pour_xy_locked"] = bool(
                grasp.get("pour_xy_lock", False)
            )
            info["c07_pour_complete"] = bool(
                grasp.get("pour_complete", False)
            )
            info["c07_particles_remaining_assist"] = int(
                grasp.get(
                    "particles_remaining",
                    len(_C07_PARTICLE_NAMES),
                )
            )
            info["c07_visual_hand_locked"] = bool(self._c07_pitcher_grasp)
            info["c07_pitcher_pinned_on_pad"] = bool(
                self._c07_pitcher_pinned
            )
            return info
        if self.task_name == "c08_tongs_ice_cubes":
            info = dict(self._last_c08_success_metrics or self._c08_tongs_ice_metrics())
            info["c08_success_counter"] = int(self._success_counter)
            info["c08_success_steps_required"] = int(self._success_steps_required())
            info["c08_tongs_grasp_assist_active"] = bool(
                self._c08_tongs_grasp
            )
            info["c08_tongs_control_armed"] = bool(
                self._c08_tongs_control_armed
            )
            info["c08_ice_grasp_assist_active"] = bool(self._c08_held_ice)
            info["c08_ice_grasp_assist_name"] = str(
                (self._c08_held_ice or {}).get("name", "")
            )
            grasp = self._c08_tongs_grasp or {}
            info["c08_assist_phase"] = str(grasp.get("phase", "idle"))
            info["c08_pour_xy_locked"] = bool(
                grasp.get("pour_xy_lock", False)
            )
            info["c08_release_armed"] = bool(
                grasp.get("release_armed", False)
            )
            info["c08_release_align_elapsed_s"] = float(
                grasp.get("align_elapsed_s", 0.0)
            )
            info["c08_hand_tracking_ready"] = bool(
                grasp.get("hand_tracking_ready", False)
            )
            info["c08_release_centered_over_cup"] = bool(
                grasp.get("release_centered_over_cup", False)
            )
            info["c08_release_transition_reason"] = str(
                grasp.get("release_transition_reason", "")
            )
            info["c08_candidate_ice"] = str(
                grasp.get("candidate_name", "")
            )
            info["c08_candidate_contact_side"] = str(
                grasp.get("candidate_contact_side", "")
            )
            info["c08_jaw_inner_gap_m"] = float(
                grasp.get(
                    "jaw_inner_gap",
                    self._c08_tongs_inner_gap(_C08_TONGS_REST_RAD),
                )
            )
            info["c08_dropping_ice"] = str(
                grasp.get("dropping_ice_name", "")
            )
            info["c08_tongs_collision_blocked"] = bool(
                grasp.get("collision_guard_blocked", False)
                or grasp.get("collision_guard_command_blocked", False)
            )
            collision_clearance = min(
                grasp.get("collision_guard_clearance", math.inf),
                grasp.get(
                    "collision_guard_command_clearance",
                    math.inf,
                ),
            )
            info["c08_tongs_collision_clearance_m"] = (
                float(collision_clearance)
                if np.isfinite(collision_clearance)
                else 0.0
            )
            info["c08_visual_hand_locked"] = bool(self._c08_tongs_grasp)
            return info
        if self.task_name == "c09_staple_paper_sheets":
            info = dict(self._last_c09_success_metrics or self._c09_staple_metrics())
            info["c09_success_counter"] = int(self._success_counter)
            info["c09_success_steps_required"] = int(self._success_steps_required())
            info["c09_paper_grasp_assist_active"] = bool(
                self._c09_paper_grasps
            )
            info["c09_papers_stacked_assist"] = bool(
                self._c09_papers_stacked
            )
            return info
        if self.task_name == "t02_connect_charger_and_cable":
            info = dict(self._last_c10_success_metrics or self._c10_connection_metrics())
            info.update(self._c10_assist_metrics)
            info["c10_success_counter"] = int(self._success_counter)
            info["c10_success_steps_required"] = int(self._success_steps_required())
            return info
        if self.task_name == "c11_xylophone_sequence":
            info = dict(self._last_c11_success_metrics or self._c11_xylophone_metrics())
            info["c11_success_counter"] = int(self._success_counter)
            info["c11_success_steps_required"] = int(self._success_steps_required())
            return info
        if self.task_name == "t11_line_up_blocks":
            info = dict(self._last_c12_success_metrics or self._c12_push_line_metrics())
            info["c12_success_counter"] = int(self._success_counter)
            info["c12_success_steps_required"] = int(self._success_steps_required())
            return info
        if self.task_name == "t04_cover_pitcher":
            info = dict(self._last_b01_lid_success_metrics)
            info["b01_success_counter"] = int(self._success_counter)
            info["b01_success_steps_required"] = int(self._success_steps_required())
            return info
        if self.task_name == "t19_orient_cup_upright":
            info = dict(self._last_a06_success_metrics)
            info["a06_success_counter"] = int(self._success_counter)
            info["a06_success_steps_required"] = int(self._success_steps_required())
            return info
        if self.task_name == "b02_nut_bolt_assembly":
            info = dict(self._b02_grasp_assist_metrics)
            info.setdefault(
                "b02_grasp_assist_enabled",
                bool(self._b02_nut_grasp_assist_enabled()),
            )
            info.setdefault(
                "b02_grasp_assist_active",
                bool(self._b02_grasp_assist is not None),
            )
            info.setdefault("b02_grasp_assist_pair", "")
            info.setdefault("b02_grasp_assist_finger_distance", -1.0)
            info.setdefault(
                "b02_grasp_assist_release_distance",
                float(_B02_GRASP_RELEASE_FINGER_DISTANCE),
            )
            info["b02_physics_dt"] = float(self._physics_dt)
            info["b02_physics_hz"] = float(1.0 / self._physics_dt)
            info["b02_physics_substeps"] = int(self._n_substeps)
            info["b02_mujoco_soft_hand"] = bool(
                self._b02_mujoco_soft_hand_enabled()
            )
            return info
        if self.task_name == "t09_insert_peg":
            metrics = self._b03_peg_socket_geometry()
            info = {
                "b03_peg_radial_error": float(metrics["radial_error"]),
                "b03_peg_center_radial_error": float(
                    metrics["center_radial_error"]
                ),
                "b03_peg_tip_radial_error": float(
                    metrics["inserted_tip_radial_error"]
                ),
                "b03_peg_insertion_depth": float(metrics["insertion_depth"]),
                "b03_peg_handle_end_depth": float(
                    metrics["handle_end_depth"]
                ),
                "b03_peg_axis_alignment": float(metrics["axis_alignment"]),
                "b03_peg_signed_axis_alignment": float(
                    metrics["signed_axis_alignment"]
                ),
                "b03_peg_tip_end_correct": bool(
                    metrics["tip_end_correct"]
                ),
                "b03_success_counter": int(self._success_counter),
                "b03_success_steps_required": int(self._success_steps_required()),
                "b03_stable_seconds": float(
                    self._success_counter * self._control_dt
                ),
            }
            info.update(self._b03_stability_metrics())
            return info
        if self.task_name == "t44_stack_blocks":
            info = dict(self._last_b04_stack_metrics or self._block_stack_metrics())
            info["stack_success_counter"] = int(self._success_counter)
            info["stack_success_steps_required"] = int(self._success_steps_required())
            info["stable_steps"] = int(self._success_counter)
            info["stable_seconds"] = float(self._success_counter * self._control_dt)
            info["required_stable_seconds"] = float(_STACK_RELEASE_STABLE_SECONDS)
            return info
        return {}

    def _init_success_condition_overlay(self) -> None:
        if (
            self._headless
            or not self._launch_isaac
            or self._success_condition_window is not None
            or self._success_condition_frame is not None
        ):
            return
        try:
            import omni.ui as ui

            viewport_window = None
            try:
                from omni.kit.viewport.utility import get_active_viewport_window

                viewport_window = get_active_viewport_window()
            except Exception:
                viewport_window = None

            if viewport_window is not None:
                self._success_condition_frame = viewport_window.get_frame(
                    "handson.success_condition_overlay"
                )
                self._success_condition_frame.visible = True
                self._build_success_condition_overlay(ui, self._success_condition_frame)
            else:
                self._success_condition_window = ui.Window(
                    "HandsOn",
                    width=420,
                    height=320,
                    visible=True,
                )
                for attr, value in (
                    ("position_x", max(20, int(self._render_width) - 440)),
                    ("position_y", 30),
                ):
                    try:
                        setattr(self._success_condition_window, attr, value)
                    except Exception:
                        pass
                self._build_success_condition_overlay(
                    ui,
                    self._success_condition_window.frame,
                )
        except Exception as exc:
            if not self._success_condition_overlay_failed:
                self._success_condition_overlay_failed = True
                self._add_task_state_warning(
                    f"Could not create Isaac success-condition overlay: {exc}"
                )

    def _build_success_condition_overlay(self, ui: Any, root: Any) -> None:
        if root is None:
            return
        self._success_condition_labels.clear()
        self._success_condition_text_labels.clear()
        self._task_status_label = None
        try:
            with root:
                with ui.VStack():
                    ui.Spacer(height=30)
                    with ui.HStack():
                        ui.Spacer()
                        with ui.ZStack(width=420, height=320):
                            ui.Rectangle(
                                style={
                                    "background_color": 0xB0202020,
                                    "border_color": 0x70FFFFFF,
                                    "border_width": 1,
                                    "border_radius": 6,
                                }
                            )
                            with ui.VStack(spacing=6, style={"margin": 12}):
                                ui.Label(_SUCCESS_CONDITION_PANEL_TITLE, height=24)
                                self._task_status_label = ui.Label(
                                    self._task_status_text(),
                                    height=28,
                                )
                                for row in self._success_condition_rows():
                                    with ui.HStack(height=22):
                                        condition_label = ui.Label(
                                            self._success_condition_display_label(row),
                                            width=350,
                                        )
                                        mark = ui.Label("", width=42)
                                        self._success_condition_text_labels[
                                            row["key"]
                                        ] = condition_label
                                        self._success_condition_labels[row["key"]] = mark
                        ui.Spacer(width=20)
                    ui.Spacer()
        except Exception as exc:
            if not self._success_condition_overlay_failed:
                self._success_condition_overlay_failed = True
                self._add_task_state_warning(
                    f"Could not build Isaac success-condition overlay: {exc}"
                )

    def _update_success_condition_overlay(self) -> None:
        if not self._success_condition_labels:
            return
        try:
            if self._success_condition_frame is not None:
                self._success_condition_frame.visible = True
            if self._success_condition_window is not None:
                self._success_condition_window.visible = True
            rows = self._success_condition_rows()
            if self._task_status_label is not None:
                self._task_status_label.text = self._task_status_text()
            for row in rows:
                condition_label = self._success_condition_text_labels.get(
                    row["key"]
                )
                if condition_label is not None:
                    condition_label.text = (
                        self._success_condition_display_label(row)
                    )
                label = self._success_condition_labels.get(row["key"])
                if label is not None:
                    label.text = _SUCCESS_CONDITION_OK_TEXT if row["ok"] else ""
        except Exception as exc:
            if not self._success_condition_overlay_failed:
                self._success_condition_overlay_failed = True
                self._add_task_state_warning(
                    f"Could not update Isaac success-condition overlay: {exc}"
                )

    def _task_status_text(self) -> str:
        if self.task_name == "c07_pitcher_pour_cup":
            metrics = self._last_c07_success_metrics
            return (
                f"杯内 {int(metrics.get('c07_particles_inside', 0))}/8"
                f"  |  洒落 {int(metrics.get('c07_spilled_count', 0))}"
            )
        if self.task_name == "c08_tongs_ice_cubes":
            metrics = self._last_c08_success_metrics
            return (
                f"杯内冰块 {int(metrics.get('c08_ice_inside', 0))}/4"
                f"  |  有效闭合 {self._c08_tongs_close_count}/4"
            )
        if self.task_name == "t09_insert_peg":
            held_seconds = min(
                float(_B03_STABLE_HOLD_SECONDS),
                float(self._success_counter * self._control_dt),
            )
            return (
                f"稳定保持 {held_seconds:.2f}/"
                f"{_B03_STABLE_HOLD_SECONDS:.2f} 秒"
            )
        if self.task_name == "c09_staple_paper_sheets":
            metrics = self._last_c09_success_metrics
            return (
                f"上盖角度 {float(metrics.get('c09_top_joint_angle', 0.0)):.2f} rad"
                f"  |  已打开 {'是' if self._c09_top_opened_seen else '否'}"
            )
        if self.task_name == "t02_connect_charger_and_cable":
            metrics = self._last_c10_success_metrics
            stage = int(bool(metrics.get("c10_plug_latched", False))) + int(
                bool(metrics.get("c10_cable_latched", False))
            )
            suffix = "  |  顺序错误" if self._c10_order_violation else ""
            return f"锁定阶段 {stage}/2{suffix}"
        if self.task_name == "c11_xylophone_sequence":
            sequence = " > ".join(
                _C11_KEY_COLOR_LABELS[name]
                for name in self._c11_target_sequence
            )
            return (
                f"顺序 {sequence}"
                f"  |  进度 {self._c11_sequence_progress}/{len(self._c11_target_sequence)}"
                f"  |  错误 {self._c11_wrong_hits}"
            )
        if self.task_name == "t11_line_up_blocks":
            metrics = self._last_c12_success_metrics
            overlap_ratios = metrics.get("c12_target_overlap_ratios", ())
            min_overlap = (
                min(float(value) for value in overlap_ratios) * 100.0
                if overlap_ratios
                else 0.0
            )
            suffix = "  |  手碰木块，正在重置" if self._c12_direct_hand_contact else ""
            return f"最低轮廓重合率 {min_overlap:.1f}%{suffix}"
        return ""

    @staticmethod
    def _condition_row(key: str, label: str, ok: bool) -> dict[str, Any]:
        return {"key": key, "label": label, "ok": bool(ok)}

    @staticmethod
    def _success_condition_display_label(row: dict[str, Any]) -> str:
        key = str(row.get("key") or "")
        if key == "c11_sequence_complete" and row.get("label"):
            return str(row["label"])
        if key in _SUCCESS_CONDITION_DISPLAY_LABELS:
            return _SUCCESS_CONDITION_DISPLAY_LABELS[key]
        return key.replace("_", " ") or "Condition"

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        try:
            if self.task_name == "t26_place_fruit_in_bucket":
                rows = self._a01_success_condition_rows()
            elif self.task_name == "t25_place_cup_on_plate":
                rows = self._a02_success_condition_rows()
            elif self.task_name == "t24_place_ball_in_bowl":
                rows = self._a03_success_condition_rows()
            elif self.task_name == "t19_orient_cup_upright":
                rows = self._a06_success_condition_rows()
            elif self.task_name == "t04_cover_pitcher":
                rows = self._b01_success_condition_rows()
            elif self.task_name == "b02_nut_bolt_assembly":
                rows = self._b02_success_condition_rows()
            elif self.task_name == "t09_insert_peg":
                rows = self._b03_success_condition_rows()
            elif self.task_name == "t44_stack_blocks":
                rows = self._b04_success_condition_rows()
            elif self.task_name == "t03_cover_box":
                rows = self._b05_success_condition_rows()
            elif self.task_name == "t08_hammer_nail":
                rows = self._c01_success_condition_rows()
            elif self.task_name == "t36_scoop_marbles":
                rows = self._c02_success_condition_rows()
            elif self.task_name == "t50_water_plant":
                rows = self._c04_success_condition_rows()
            elif self.task_name == "t14_move_and_click_mouse":
                rows = self._c05_success_condition_rows()
            elif self.task_name == "t23_photograph_logo":
                rows = self._c06_success_condition_rows()
            elif self.task_name == "c07_pitcher_pour_cup":
                rows = self._c07_success_condition_rows()
            elif self.task_name == "c08_tongs_ice_cubes":
                rows = self._c08_success_condition_rows()
            elif self.task_name == "c09_staple_paper_sheets":
                rows = self._c09_success_condition_rows()
            elif self.task_name == "t02_connect_charger_and_cable":
                rows = self._c10_success_condition_rows()
            elif self.task_name == "c11_xylophone_sequence":
                rows = self._c11_success_condition_rows()
            elif self.task_name == "t11_line_up_blocks":
                rows = self._c12_success_condition_rows()
            else:
                rows = [
                    self._condition_row(
                        "event_success_unavailable",
                        "该任务需要按钮/触发器事件，Isaac 当前未接入事件判定",
                        False,
                    )
                ]
        except Exception as exc:
            rows = [
                self._condition_row(
                    "condition_panel_error",
                    f"判定条件读取失败：{exc}",
                    False,
                )
            ]
        return self._append_success_hold_row(rows)

    def _c07_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c07_success_metrics or self._c07_pitcher_pour_metrics()
        return [
            self._condition_row("c07_enough_inside", "至少 80% 代理物进入目标杯", bool(metrics["c07_enough_inside"])),
            self._condition_row("c07_spill_limited", "桌面洒落数量不超过 1 个", bool(metrics["c07_spill_limited"])),
            self._condition_row("c07_tilt_seen", "大把手杯已经发生有效倾倒", bool(metrics["c07_tilt_seen"])),
            self._condition_row("c07_pitcher_centered_on_pad", "大把手杯完整放入回收垫中心", bool(metrics["c07_pitcher_centered_on_pad"])),
            self._condition_row("c07_pitcher_on_pad_height", "大把手杯底已经落到回收垫上", bool(metrics["c07_pitcher_on_pad_height"])),
            self._condition_row("c07_pitcher_stable", "大把手杯已停止移动", bool(metrics["c07_pitcher_stable"])),
            self._condition_row("c07_hand_released", "手已完全离开大把手杯", bool(metrics["c07_hand_released"])),
            self._condition_row("c07_pitcher_returned", "大把手杯回正并稳定放回", bool(metrics["c07_pitcher_returned"])),
        ]

    def _c08_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c08_success_metrics or self._c08_tongs_ice_metrics()
        return [
            self._condition_row("c08_all_ice_inside", "4 个冰块均进入杯内", bool(metrics["c08_all_ice_inside"])),
            self._condition_row("c08_tongs_cycles", "食品夹完成至少 4 次闭合", bool(metrics["c08_tongs_cycles"])),
            self._condition_row("c08_cup_upright", "玻璃杯未倾倒", bool(metrics["c08_cup_upright"])),
        ]

    def _c09_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c09_success_metrics or self._c09_staple_metrics()
        return [
            self._condition_row("c09_stapler_opened", "订书机上盖已打开", bool(metrics["c09_stapler_opened"])),
            self._condition_row("c09_papers_overlap", "两张纸重叠率达到要求", bool(metrics["c09_papers_overlap"])),
            self._condition_row("c09_papers_in_zone", "纸张进入有效装订区", bool(metrics["c09_papers_in_zone"])),
            self._condition_row("c09_stapler_pressed", "订书机压合越过阈值", bool(metrics["c09_stapler_pressed"])),
        ]

    def _c10_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c10_success_metrics or self._c10_connection_metrics()
        return [
            self._condition_row("c10_first_connected", "充电头先插入排插", bool(metrics["c10_first_connected"])),
            self._condition_row("c10_second_connected", "充电线接头插入充电头", bool(metrics["c10_second_connected"])),
            self._condition_row("c10_order_valid", "插接顺序正确", bool(metrics["c10_order_valid"])),
        ]

    def _c11_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c11_success_metrics or self._c11_xylophone_metrics()
        sequence = " → ".join(
            _C11_KEY_COLOR_LABELS[name]
            for name in self._c11_target_sequence
        )
        return [
            self._condition_row(
                "c11_sequence_complete",
                (
                    f"敲击顺序：{sequence}"
                    f"（{self._c11_sequence_progress}/{len(self._c11_target_sequence)}）"
                ),
                bool(metrics["c11_sequence_complete"]),
            ),
            self._condition_row("c11_wrong_hits_ok", "敲错后已从序列开头重置", bool(metrics["c11_wrong_hits_ok"])),
            self._condition_row("c11_mallet_returned", "小槌最终位于放回区", bool(metrics["c11_mallet_returned"])),
        ]

    def _c12_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c12_success_metrics or self._c12_push_line_metrics()
        return [
            self._condition_row(
                "c12_targets_overlap",
                "三个木块与目标轮廓重合率均大于 90%",
                bool(metrics["c12_targets_overlap"]),
            ),
            self._condition_row("c12_blocks_on_line", "三个木块中心贴近基准线", bool(metrics["c12_blocks_on_line"])),
            self._condition_row("c12_blocks_aligned", "三个木块朝向与基准线对齐", bool(metrics["c12_blocks_aligned"])),
            self._condition_row("c12_spacing_ok", "相邻木块间距在设定范围内", bool(metrics["c12_spacing_ok"])),
            self._condition_row("c12_tool_used", "已使用直角尺/刮板推动", bool(metrics["c12_tool_used"])),
            self._condition_row("c12_no_direct_grasp", "手未直接触碰木块", bool(metrics["c12_no_direct_grasp"])),
        ]

    def _append_success_hold_row(
        self,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        required_steps = self._success_steps_required()
        if required_steps > 1:
            hold_label = f"以上条件连续保持 {required_steps} 帧"
            hold_key = "success_hold"
            if self.task_name == "c07_pitcher_pour_cup":
                hold_label = "放稳并脱手连续保持 0.5 秒"
                hold_key = "c07_release_hold"
            elif self.task_name == "t09_insert_peg":
                hold_label = "插入且插销/插座稳定连续保持 0.5 秒"
                hold_key = "b03_stable_hold"
            rows.append(
                self._condition_row(
                    hold_key,
                    hold_label,
                    self._success_counter >= required_steps,
                )
            )
        return rows

    def _a01_success_condition_rows(self) -> list[dict[str, Any]]:
        local, _target, _bucket = self._target_local_to_receptacle()
        horizontal_margin = 0.020
        horizontal_ok = bool(
            -0.18 + horizontal_margin <= float(local[0]) <= 0.16 - horizontal_margin
            and -0.13 + horizontal_margin <= float(local[2]) <= 0.12 - horizontal_margin
        )
        object_half_height = _OBJECT_HALF_HEIGHTS.get(self._target_object_name, 0.04)
        bottom_clearance = float(local[1] - (-0.08 + object_half_height))
        bottom_ok = -0.020 <= bottom_clearance <= 0.055
        return [
            self._condition_row("a01_target_inside_bucket_xy", "目标物体在桶底范围内", horizontal_ok),
            self._condition_row("a01_target_on_bucket_bottom", "目标物体已经落到桶底高度", bottom_ok),
        ]

    def _a02_success_condition_rows(self) -> list[dict[str, Any]]:
        target = self._pose_for_role("target")
        tray = self._pose_for_role("receptacle")
        velocity = self._physics_body_velocity(self._target_object_name)
        stable = velocity is None or (
            float(np.linalg.norm(velocity[:3])) <= 0.025
            and float(np.linalg.norm(velocity[3:])) <= 0.35
        )
        return [
            self._condition_row("a02_target_is_mug", "目标是棕色杯子", self._target_object_name == "025_mug"),
            self._condition_row("a02_centered_on_tray", "杯子放在托盘中心附近", float(np.linalg.norm(target[:2] - tray[:2])) <= 0.065),
            self._condition_row("a02_height_on_tray", "杯底高度在托盘上方", 0.015 <= float(target[2] - tray[2]) <= 0.10),
            self._condition_row("a02_bottom_down", "杯子底朝下放正", self._z_axis_alignment(target) >= float(np.cos(np.deg2rad(25.0)))),
            self._condition_row(
                "a02_not_dropped",
                "杯子没有掉到桌面下方",
                target[2] >= self._current_table_top_z() - 0.02,
            ),
            self._condition_row("a02_stable", "杯子已经稳定不滑动", stable),
        ]

    def _a03_success_condition_rows(self) -> list[dict[str, Any]]:
        local, _target, _bowl = self._target_local_to_receptacle()
        return [
            self._condition_row("a03_ball_in_bowl_xy", "球在碗口范围内", float(np.linalg.norm(local[:2])) <= 0.095),
            self._condition_row("a03_ball_in_bowl_z", "球的高度在碗里面", 0.010 <= float(local[2]) <= 0.115),
        ]

    def _a06_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_a06_success_metrics or self._a06_cup_success_metrics()
        return [
            self._condition_row("a06_opening_up", "杯口朝上", bool(metrics["a06_opening_up"])),
            self._condition_row("a06_on_table", "杯子放在桌面高度", bool(metrics["a06_on_table"])),
            self._condition_row("a06_hand_clear", "手已经离开杯子", bool(metrics["a06_hand_clear"])),
        ]

    def _b01_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_b01_lid_success_metrics
        if not metrics:
            metrics = self._b01_lid_insert_metrics(
                self._pose_for_role("target"),
                self._pose_for_role("receptacle"),
                update_hand_clear_hold=False,
            )
        return [
            self._condition_row("b01_insert_inside", "杯盖下方凸起整体进入杯口", bool(metrics["b01_lid_insert_inside"])),
            self._condition_row("b01_z_close", "杯盖高度接近完全扣合", bool(metrics["b01_lid_z_close"])),
            self._condition_row("b01_top_aligned", "杯盖基本放平", bool(metrics["b01_lid_top_aligned"])),
            self._condition_row("b01_moved", "杯盖已被移动到杯子上", bool(metrics["b01_lid_moved"])),
            self._condition_row("b01_stable", "杯盖已经稳定不滑动", bool(metrics["b01_lid_stable"])),
            self._condition_row("b01_hand_clear", "手已经离开杯盖", bool(metrics["b01_hand_clear"])),
            self._condition_row("b01_hand_clear_held", "手离开杯盖保持 0.5 秒", bool(metrics["b01_hand_clear_held"])),
        ]

    def _b02_success_condition_rows(self) -> list[dict[str, Any]]:
        nut = self._pose_for_role("target")
        bolt = self._pose_for_role("receptacle")
        local, _target, _receptacle = self._target_local_to_receptacle()
        radial_error = float(np.linalg.norm(local[:2]))
        depth = float(local[2])
        axis_alignment = abs(self._z_axis_alignment(nut, bolt))
        axial_travel = 0.0
        if self._b02_thread_entry_depth is not None and self._b02_min_thread_depth is not None:
            axial_travel = max(
                0.0,
                float(self._b02_thread_entry_depth) - float(self._b02_min_thread_depth),
            )
        velocity = self._physics_body_velocity(self._target_object_name)
        stable = velocity is None or (
            float(np.linalg.norm(velocity[:3])) <= _B02_STABLE_LINEAR_SPEED
            and float(np.linalg.norm(velocity[3:])) <= _B02_STABLE_ANGULAR_SPEED
        )
        return [
            self._condition_row("b02_radial_aligned", "螺母中心对准螺杆", radial_error <= _B02_MAX_RADIAL_ERROR),
            self._condition_row("b02_depth_ok", "螺母旋入到目标深度附近", abs(depth - _B02_TARGET_DEPTH) <= _B02_DEPTH_TOLERANCE and 0.008 <= depth <= 0.055),
            self._condition_row("b02_axis_aligned", "螺母轴线和螺杆对齐", axis_alignment >= _B02_MIN_AXIS_ALIGNMENT),
            self._condition_row("b02_thread_seen", "螺纹已经进入啮合区", self._b02_thread_engaged_seen),
            self._condition_row("b02_twisted", "螺母已经有明显旋转", abs(self._b02_signed_twist_deg) >= _B02_MIN_TWIST_PROGRESS_DEG),
            self._condition_row("b02_axial_travel", "螺母沿螺杆向下走过一段距离", axial_travel >= _B02_MIN_AXIAL_TRAVEL),
            self._condition_row("b02_stable", "螺母已经稳定不滑动", stable),
        ]

    def _b03_success_condition_rows(self) -> list[dict[str, Any]]:
        geometry = self._b03_peg_socket_geometry()
        stability = self._b03_stability_metrics()
        return [
            self._condition_row(
                "b03_radial_aligned",
                "插销入孔处对准孔位",
                float(geometry["radial_error"])
                <= _B03_INSERTION_RADIAL_TOLERANCE,
            ),
            self._condition_row("b03_inserted", "插销已经插入孔中", float(geometry["insertion_depth"]) >= _B03_INSERTION_MIN_DEPTH),
            self._condition_row("b03_axis_aligned", "插销方向和孔方向对齐", float(geometry["axis_alignment"]) >= _B03_AXIS_ALIGNMENT_THRESHOLD),
            self._condition_row(
                "b03_tip_end_correct",
                "插销细端朝向孔内",
                bool(geometry["tip_end_correct"]),
            ),
            self._condition_row(
                "b03_peg_stable",
                "插销线速度和角速度已经稳定",
                bool(stability["b03_peg_stable"]),
            ),
            self._condition_row(
                "b03_socket_stable",
                "插座线速度和角速度已经稳定",
                bool(stability["b03_socket_stable"]),
            ),
        ]

    def _b04_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_b04_stack_metrics or self._block_stack_metrics()
        return [
            self._condition_row(
                f"b04_{metric_key.removeprefix('stack_')}",
                label,
                bool(metrics[metric_key]),
            )
            for metric_key, label in _B04_SUCCESS_CONDITIONS
        ]

    def _b05_success_condition_rows(self) -> list[dict[str, Any]]:
        lid = self._pose_for_role("target")
        base = self._pose_for_role("receptacle")
        local, _target, _receptacle = self._target_local_to_receptacle()
        z_error = float(local[2] - _LID_GOAL_LOCAL_Z[self.task_name])
        top_aligned = self._z_axis_alignment(lid, base) >= float(np.cos(np.deg2rad(20.0)))
        yaw_aligned = self._x_axis_abs_alignment(lid, base) >= float(np.cos(np.deg2rad(25.0)))
        moved = bool(self._lid_was_moved)
        if self._lid_start_pose is not None:
            moved = moved or float(np.linalg.norm(lid[:3] - self._lid_start_pose[:3])) >= _LID_MIN_MOVE_DISTANCE
        velocity = self._physics_body_velocity(self._target_object_name)
        stable = velocity is None or (
            float(np.linalg.norm(velocity[:3])) <= _LID_STABLE_LINEAR_SPEED
            and float(np.linalg.norm(velocity[3:])) <= _LID_STABLE_ANGULAR_SPEED
        )
        return [
            self._condition_row("b05_centered", "盒盖中心对准盒口", float(np.linalg.norm(local[:2])) <= 0.055),
            self._condition_row("b05_height", "盒盖高度接近盖好位置", -0.020 <= z_error <= 0.025),
            self._condition_row("b05_top_aligned", "盒盖基本放平", top_aligned),
            self._condition_row("b05_yaw_aligned", "盒盖方向和盒子对齐", yaw_aligned),
            self._condition_row("b05_moved", "盒盖已被移动到盒子上", moved),
            self._condition_row("b05_stable", "盒盖已经稳定不滑动", stable),
        ]

    def _c01_success_condition_rows(self) -> list[dict[str, Any]]:
        depth = self._c01_nail_depth()
        effective_depth = max(depth, self._c01_max_valid_hit_depth)
        wood = self._pose_for_role("wood")
        table_z = self._current_table_top_z()
        wood_clearance = float(wood[2] - _C01_WOOD_HALF_HEIGHT - table_z)
        return [
            self._condition_row("c01_valid_hit", "锤子确实敲中过钉子", self._c01_had_valid_hit),
            self._condition_row("c01_depth", "钉子已经被敲入足够深", effective_depth >= _C01_SUCCESS_DEPTH),
            self._condition_row("c01_wood_lift", "木板被钉子带起离开桌面", wood_clearance >= _C01_WOOD_LIFT_CLEARANCE),
        ]

    def _c02_success_condition_rows(self) -> list[dict[str, Any]]:
        marble = self._pose_for_role("target")
        bowl = self._pose_for_role("receptacle")
        marble_rel_bowl = _invert_pose_matrix(_pose_vector_to_matrix(bowl)) @ _pose_vector_to_matrix(marble)
        bowl_rel = marble_rel_bowl[:3, 3]
        marble_inside_bowl = (
            float(np.linalg.norm(bowl_rel[:2])) <= 0.095
            and 0.010 <= float(bowl_rel[2]) <= 0.130
        )
        return [
            self._condition_row("c02_spoon_used", "勺子已经被拿起使用", self._c02_spoon_used),
            self._condition_row("c02_spoon_loaded", "勺子里曾经装到弹珠", self._c02_spoon_loaded_seen),
            self._condition_row("c02_over_bowl", "装着弹珠的勺子到过碗上方", self._c02_spoon_over_bowl_seen),
            self._condition_row("c02_marble_in_bowl", "弹珠已经落在碗里", marble_inside_bowl),
        ]

    def _c04_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c04_success_metrics
        if not metrics:
            aim_metrics = self._c04_spray_aim_metrics()
            metrics = {
                **aim_metrics,
                "spray_grasp_ok": self._c04_hand_gripping_spray(),
                "spray_trigger_contact_ok": False,
                "c04_water_emitting": self._c04_water_emitting,
            }
        return [
            self._condition_row(
                "c04_spray_grasped",
                "手已经抓住喷壶",
                bool(metrics.get("spray_grasp_ok", False)),
            ),
            self._condition_row(
                "c04_trigger_contact",
                "手指确实按到扳机",
                bool(metrics.get("spray_trigger_contact_ok", False)),
            ),
            self._condition_row(
                "c04_water_emitting",
                "扳机保持按下并正在喷水",
                bool(metrics.get("c04_water_emitting", False)),
            ),
            self._condition_row(
                "c04_water_hits_target",
                "水流命中植株或盆土",
                bool(metrics.get("spray_hits_soil_or_plant", False)),
            ),
        ]

    def _c05_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c05_success_metrics or self._c05_mouse_click_metrics(
            update_failure=False
        )
        return [
            self._condition_row(
                "c05_mouse_moved",
                "鼠标已经离开初始位置并移动足够距离",
                bool(metrics["c05_mouse_moved"]),
            ),
            self._condition_row(
                "c05_mouse_on_pad",
                "鼠标已经移动到鼠标垫范围内",
                bool(metrics["c05_mouse_on_pad"]),
            ),
            self._condition_row(
                "c05_no_early_click",
                "移动完成前没有压到左右按键",
                bool(metrics["c05_no_early_click"]),
            ),
            self._condition_row(
                "c05_left_clicked",
                "鼠标到位后按下左键",
                bool(metrics["c05_left_clicked"]),
            ),
        ]

    def _c06_success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._last_c06_success_metrics or self._c06_photograph_metrics()
        return [
            self._condition_row(
                "c06_camera_aimed",
                "相机镜头对准标志",
                bool(metrics["c06_camera_aimed"]),
            ),
            self._condition_row(
                "c06_right_hand_contact",
                "右手手指接触快门按钮",
                bool(metrics["c06_right_hand_contact"]),
            ),
            self._condition_row(
                "c06_shutter_pressed",
                "快门按钮已经压下",
                bool(metrics["c06_shutter_pressed"]),
            ),
        ]

    def _lid_fit_success(self) -> bool:
        lid = self._pose_for_role("target")
        base = self._pose_for_role("receptacle")
        local, _target, _receptacle = self._target_local_to_receptacle()
        if self.task_name == "t04_cover_pitcher":
            metrics = self._b01_lid_insert_metrics(
                lid,
                base,
                update_hand_clear_hold=True,
            )
            self._last_b01_lid_success_metrics = metrics
            return bool(metrics["b01_success_now"])
        else:
            xy_tolerance = 0.055
            z_below_tolerance = 0.020
            z_above_tolerance = 0.025
            top_threshold = float(np.cos(np.deg2rad(20.0)))
            yaw_threshold = float(np.cos(np.deg2rad(25.0)))
            xy_close = float(np.linalg.norm(local[:2])) <= xy_tolerance
            goal_z = _LID_GOAL_LOCAL_Z[self.task_name]
            z_error = float(local[2] - goal_z)
            z_close = -z_below_tolerance <= z_error <= z_above_tolerance
        top_aligned = self._z_axis_alignment(lid, base) >= top_threshold
        yaw_aligned = self._x_axis_abs_alignment(lid, base) >= yaw_threshold

        if self._lid_start_pose is None:
            self._lid_start_pose = lid.copy()
        move_distance = float(np.linalg.norm(lid[:3] - self._lid_start_pose[:3]))
        lift_height = float(lid[2] - self._lid_start_pose[2])
        if (
            move_distance >= _LID_MIN_MOVE_DISTANCE
            or lift_height >= _LID_MIN_LIFT_HEIGHT
        ):
            self._lid_was_moved = True
        velocity = self._physics_body_velocity(self._target_object_name)
        stable = velocity is None or (
            float(np.linalg.norm(velocity[:3])) <= _LID_STABLE_LINEAR_SPEED
            and float(np.linalg.norm(velocity[3:])) <= _LID_STABLE_ANGULAR_SPEED
        )
        return bool(
            self._lid_was_moved
            and xy_close
            and z_close
            and top_aligned
            and yaw_aligned
            and stable
        )

    def _nut_bolt_success(self) -> bool:
        nut = self._pose_for_role("target")
        bolt = self._pose_for_role("receptacle")
        local, _target, _receptacle = self._target_local_to_receptacle()
        radial_error = float(np.linalg.norm(local[:2]))
        depth = float(local[2])
        axis_alignment = abs(self._z_axis_alignment(nut, bolt))
        relative_rotation = (
            _quat_wxyz_to_matrix(bolt[3:7]).T
            @ _quat_wxyz_to_matrix(nut[3:7])
        )
        relative_yaw = float(
            math.atan2(relative_rotation[1, 0], relative_rotation[0, 0])
        )
        thread_active = bool(
            radial_error <= _B02_THREAD_ACTIVE_RADIAL_ERROR
            and axis_alignment >= _B02_THREAD_ACTIVE_AXIS_ALIGNMENT
            and 0.008 <= depth <= 0.064
        )
        thread_engaged_now = bool(
            radial_error <= _B02_THREAD_ENGAGE_RADIAL_ERROR
            and axis_alignment >= _B02_THREAD_ENGAGE_AXIS_ALIGNMENT
            and 0.010 <= depth <= 0.064
        )
        if thread_engaged_now and not self._b02_thread_engaged_seen:
            self._b02_thread_engaged_seen = True
            self._b02_thread_entry_depth = depth
            self._b02_min_thread_depth = depth
            self._b02_prev_relative_yaw = relative_yaw
        elif (
            self._b02_thread_engaged_seen
            and thread_active
            and self._b02_prev_relative_yaw is not None
        ):
            yaw_delta = math.atan2(
                math.sin(relative_yaw - self._b02_prev_relative_yaw),
                math.cos(relative_yaw - self._b02_prev_relative_yaw),
            )
            yaw_delta_degrees = math.degrees(yaw_delta)
            if abs(yaw_delta_degrees) <= 25.0:
                self._b02_signed_twist_deg += yaw_delta_degrees
            self._b02_min_thread_depth = min(
                float(self._b02_min_thread_depth), depth
            )
            self._b02_prev_relative_yaw = relative_yaw
        elif self._b02_thread_engaged_seen and thread_active:
            self._b02_min_thread_depth = min(
                float(self._b02_min_thread_depth), depth
            )
            self._b02_prev_relative_yaw = relative_yaw
        elif not thread_active:
            self._b02_prev_relative_yaw = None

        axial_travel = 0.0
        if (
            self._b02_thread_entry_depth is not None
            and self._b02_min_thread_depth is not None
        ):
            axial_travel = max(
                0.0,
                float(self._b02_thread_entry_depth)
                - float(self._b02_min_thread_depth),
            )
        velocity = self._physics_body_velocity(self._target_object_name)
        stable = velocity is None or (
            float(np.linalg.norm(velocity[:3])) <= _B02_STABLE_LINEAR_SPEED
            and float(np.linalg.norm(velocity[3:])) <= _B02_STABLE_ANGULAR_SPEED
        )
        return bool(
            radial_error <= _B02_MAX_RADIAL_ERROR
            and abs(depth - _B02_TARGET_DEPTH) <= _B02_DEPTH_TOLERANCE
            and 0.008 <= depth <= 0.055
            and axis_alignment >= _B02_MIN_AXIS_ALIGNMENT
            and self._b02_thread_engaged_seen
            and abs(self._b02_signed_twist_deg)
            >= _B02_MIN_TWIST_PROGRESS_DEG
            and axial_travel >= _B02_MIN_AXIAL_TRAVEL
            and stable
        )

    def _b03_stability_metrics(self) -> dict[str, float | bool | None]:
        metrics: dict[str, float | bool | None] = {
            "b03_stable_linear_speed_limit": float(
                _B03_STABLE_LINEAR_SPEED
            ),
            "b03_stable_angular_speed_limit": float(
                _B03_STABLE_ANGULAR_SPEED
            ),
            "b03_required_stable_seconds": float(
                _B03_STABLE_HOLD_SECONDS
            ),
        }
        for role, name in (
            ("peg", self._target_object_name),
            ("socket", self._spec.receptacle),
        ):
            velocity = self._physics_body_velocity(name)
            available = bool(
                velocity is not None
                and np.asarray(velocity).shape == (6,)
                and np.all(np.isfinite(velocity))
            )
            linear_speed = (
                float(np.linalg.norm(np.asarray(velocity)[:3]))
                if available
                else None
            )
            angular_speed = (
                float(np.linalg.norm(np.asarray(velocity)[3:]))
                if available
                else None
            )
            stable = bool(
                available
                and linear_speed is not None
                and angular_speed is not None
                and linear_speed <= _B03_STABLE_LINEAR_SPEED
                and angular_speed <= _B03_STABLE_ANGULAR_SPEED
            )
            metrics[f"b03_{role}_velocity_available"] = available
            metrics[f"b03_{role}_linear_speed"] = linear_speed
            metrics[f"b03_{role}_angular_speed"] = angular_speed
            metrics[f"b03_{role}_stable"] = stable
        return metrics

    def _peg_socket_success(self) -> bool:
        geometry = self._b03_peg_socket_geometry()
        stability = self._b03_stability_metrics()
        radial_error = float(geometry["radial_error"])
        insertion_depth = float(geometry["insertion_depth"])
        axis_alignment = float(geometry["axis_alignment"])
        return bool(
            radial_error <= _B03_INSERTION_RADIAL_TOLERANCE
            and insertion_depth >= _B03_INSERTION_MIN_DEPTH
            and axis_alignment >= _B03_AXIS_ALIGNMENT_THRESHOLD
            and bool(geometry["tip_end_correct"])
            and bool(stability["b03_peg_stable"])
            and bool(stability["b03_socket_stable"])
        )

    def _b03_peg_socket_geometry(
        self,
    ) -> dict[str, np.ndarray | float | bool]:
        peg = self._pose_for_role("target")
        socket = self._pose_for_role("receptacle")
        peg_rot = _quat_wxyz_to_matrix(np.asarray(peg, dtype=np.float64)[3:7])
        socket_rot = _quat_wxyz_to_matrix(np.asarray(socket, dtype=np.float64)[3:7])

        socket_site = (
            np.asarray(socket, dtype=np.float64)[:3]
            + socket_rot @ _B03_SOCKET_SITE_LOCAL_POS
        )
        socket_axis = socket_rot[:, 2].copy()
        socket_axis_norm = float(np.linalg.norm(socket_axis))
        if socket_axis_norm < 1e-9:
            socket_axis = _WORLD_UP.copy()
        else:
            socket_axis = socket_axis / socket_axis_norm

        peg_center = (
            np.asarray(peg, dtype=np.float64)[:3]
            + peg_rot @ _B03_PEG_COLLISION_LOCAL_POS
        )
        peg_axis = peg_rot[:, 2].copy()
        peg_axis_norm = float(np.linalg.norm(peg_axis))
        if peg_axis_norm < 1e-9:
            peg_axis = _WORLD_UP.copy()
        else:
            peg_axis = peg_axis / peg_axis_norm

        center_delta = peg_center - socket_site
        axial_offset = float(np.dot(center_delta, socket_axis))
        center_radial_vec = center_delta - axial_offset * socket_axis
        # The peg is asymmetric: local -Z is the narrow insertion tip while
        # local +Z meets the enlarged handle.  Never choose whichever endpoint
        # happens to be deeper, because that makes a handle-first pose look
        # equivalent to a valid insertion.
        inserted_tip = peg_center - peg_axis * _B03_PEG_SHAFT_HALF_LENGTH
        handle_end = peg_center + peg_axis * _B03_PEG_SHAFT_HALF_LENGTH
        insertion_depth = float(
            np.dot(socket_site - inserted_tip, socket_axis)
        )
        handle_end_depth = float(
            np.dot(socket_site - handle_end, socket_axis)
        )
        inserted_tip_delta = inserted_tip - socket_site
        inserted_tip_axial_offset = float(
            np.dot(inserted_tip_delta, socket_axis)
        )
        inserted_tip_radial_vec = (
            inserted_tip_delta
            - inserted_tip_axial_offset * socket_axis
        )

        # Judge centering where the shaft crosses the socket-mouth plane.
        # Using the long peg's collision center makes a genuine, slightly
        # tilted insertion look far off-axis even though its tip and shaft
        # have entered the hole.
        peg_socket_axis_dot = float(np.dot(peg_axis, socket_axis))
        if abs(peg_socket_axis_dot) > 1e-9:
            mouth_axis_offset = float(
                np.clip(
                    -axial_offset / peg_socket_axis_dot,
                    -_B03_PEG_SHAFT_HALF_LENGTH,
                    _B03_PEG_SHAFT_HALF_LENGTH,
                )
            )
            mouth_point = peg_center + mouth_axis_offset * peg_axis
        else:
            mouth_point = inserted_tip.copy()
        mouth_delta = mouth_point - socket_site
        mouth_axial_offset = float(np.dot(mouth_delta, socket_axis))
        radial_vec = mouth_delta - mouth_axial_offset * socket_axis
        signed_axis_alignment = float(np.dot(peg_axis, socket_axis))
        axis_alignment = float(abs(signed_axis_alignment))
        return {
            "socket_site": socket_site,
            "socket_axis": socket_axis,
            "peg_center": peg_center,
            "peg_axis": peg_axis,
            "inserted_tip": inserted_tip,
            "mouth_point": mouth_point,
            "radial_vec": radial_vec,
            "radial_error": float(np.linalg.norm(radial_vec)),
            "center_radial_error": float(np.linalg.norm(center_radial_vec)),
            "inserted_tip_radial_error": float(
                np.linalg.norm(inserted_tip_radial_vec)
            ),
            "insertion_depth": insertion_depth,
            "handle_end_depth": handle_end_depth,
            "axis_alignment": axis_alignment,
            "signed_axis_alignment": signed_axis_alignment,
            "tip_end_correct": signed_axis_alignment > 0.0,
        }

    def _b03_reset_yaw_degrees(self, name: str) -> float:
        if not self.randomize:
            return 0.0
        bounds = (
            _B03_PEG_YAW_BOUNDS_DEG
            if name in self._spec.target_objects
            else _B03_SOCKET_YAW_BOUNDS_DEG
        )
        return float(self._random.uniform(bounds[0], bounds[1]))

    def _b03_reset_pose_for_name(
        self,
        name: str,
        pose_xyzw: np.ndarray,
    ) -> np.ndarray:
        pose = np.asarray(pose_xyzw, dtype=np.float64).copy().reshape(-1, 7)
        if pose.shape[0] == 0:
            return pose
        role = "peg" if name in self._spec.target_objects else "socket"
        if self.randomize:
            return pose
        else:
            pose[:, :2] = _B03_DEFAULT_RESET_XY[role]
        return pose

    @classmethod
    def _b03_peg_reset_layout_margin(cls, peg_pose_xyzw: np.ndarray) -> float:
        segment_start, segment_end = cls._b03_peg_segment_xy(peg_pose_xyzw)
        margin = math.inf
        for hand_xy in _B03_RESET_RIGHT_HAND_AVOID_XY:
            distance, _closest = cls._point_to_segment_distance_xy(
                hand_xy,
                segment_start,
                segment_end,
            )
            margin = min(margin, distance - _B03_RESET_RIGHT_HAND_CLEARANCE)
        return float(margin)

    @classmethod
    def _b03_reset_layout_margin(
        cls,
        peg_pose_xyzw: np.ndarray,
        socket_pose_xyzw: np.ndarray,
    ) -> float:
        segment_start, segment_end = cls._b03_peg_segment_xy(peg_pose_xyzw)
        socket_xy = np.asarray(socket_pose_xyzw, dtype=np.float64).reshape(7)[:2]
        margin = cls._b03_peg_reset_layout_margin(peg_pose_xyzw)
        for hand_xy in _B03_RESET_SOCKET_HAND_AVOID_XY:
            margin = min(
                margin,
                float(np.linalg.norm(socket_xy - hand_xy))
                - _B03_RESET_SOCKET_HAND_CLEARANCE,
            )
        socket_segment_distance, _closest = cls._point_to_segment_distance_xy(
            socket_xy,
            segment_start,
            segment_end,
        )
        margin = min(
            margin,
            socket_segment_distance - _B03_RESET_SOCKET_SEGMENT_CLEARANCE,
        )
        return float(margin)

    @staticmethod
    def _b03_pose_with_sampled_xy(
        pose_xyzw: np.ndarray,
        xy: np.ndarray,
    ) -> np.ndarray:
        pose = np.asarray(pose_xyzw, dtype=np.float64).copy().reshape(-1, 7)
        if pose.shape[0] > 0:
            pose[:, :2] = constrain_recording_object_xy(xy)
        return pose

    def _b03_sample_clear_peg_reset_pose(self, pose_xyzw: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose_xyzw, dtype=np.float64).copy().reshape(-1, 7)
        if pose.shape[0] == 0:
            return pose
        if not self.randomize:
            return self._b03_clear_reset_pose_from_right_hand(
                pose,
                use_peg_segment=True,
            )

        best: np.ndarray | None = None
        best_margin = -math.inf
        for _layout_attempt in range(_B03_RESET_MAX_ATTEMPTS):
            for _candidate_index in range(_B03_RESET_CANDIDATES_PER_OBJECT):
                candidate = pose.copy()
                candidate[:, :2] = self._random.uniform(
                    _B03_ROKAE_PEG_XY_BOUNDS[0],
                    _B03_ROKAE_PEG_XY_BOUNDS[1],
                    size=(candidate.shape[0], 2),
                )
                margin = self._b03_peg_reset_layout_margin(candidate[0])
                if margin > best_margin:
                    best = candidate.copy()
                    best_margin = margin
                if margin >= _B03_RESET_MIN_CLEARANCE_SLACK:
                    return candidate

        if best is not None and best_margin >= _B03_RESET_MIN_CLEARANCE_SLACK:
            return best
        fallback = self._b03_pose_with_sampled_xy(
            pose,
            np.asarray([-0.34, -0.28], dtype=np.float64),
        )
        return self._b03_clear_reset_pose_from_right_hand(
            fallback,
            use_peg_segment=True,
        )

    def _b03_sample_clear_socket_reset_pose(
        self,
        peg_pose_xyzw: np.ndarray,
        socket_pose_xyzw: np.ndarray,
    ) -> np.ndarray:
        peg_pose = np.asarray(peg_pose_xyzw, dtype=np.float64).copy().reshape(-1, 7)
        socket_pose = np.asarray(socket_pose_xyzw, dtype=np.float64).copy().reshape(-1, 7)
        if peg_pose.shape[0] == 0 or socket_pose.shape[0] == 0:
            return socket_pose
        if not self.randomize:
            socket_pose = self._b03_clear_socket_reset_pose(peg_pose, socket_pose)
            socket_pose = self._b03_clear_socket_reset_pose_from_hands(socket_pose)
            return self._b03_clear_socket_reset_pose(peg_pose, socket_pose)

        best: np.ndarray | None = None
        best_margin = -math.inf
        for _layout_attempt in range(_B03_RESET_MAX_ATTEMPTS):
            for _candidate_index in range(_B03_RESET_CANDIDATES_PER_OBJECT):
                candidate = socket_pose.copy()
                candidate[:, :2] = self._random.uniform(
                    _B03_ROKAE_SOCKET_XY_BOUNDS[0],
                    _B03_ROKAE_SOCKET_XY_BOUNDS[1],
                    size=(candidate.shape[0], 2),
                )
                margin = self._b03_reset_layout_margin(peg_pose[0], candidate[0])
                if margin > best_margin:
                    best = candidate.copy()
                    best_margin = margin
                if margin >= _B03_RESET_MIN_CLEARANCE_SLACK:
                    return candidate

        if best is not None and best_margin >= _B03_RESET_MIN_CLEARANCE_SLACK:
            return best
        fallback = self._b03_pose_with_sampled_xy(
            socket_pose,
            np.asarray([-0.18, 0.26], dtype=np.float64),
        )
        fallback = self._b03_clear_socket_reset_pose(peg_pose, fallback)
        fallback = self._b03_clear_socket_reset_pose_from_hands(fallback)
        return self._b03_clear_socket_reset_pose(peg_pose, fallback)

    @staticmethod
    def _b03_peg_segment_xy(peg_pose_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        peg_pose = np.asarray(peg_pose_xyzw, dtype=np.float64).reshape(7)
        peg_quat_xyzw = peg_pose[3:7]
        peg_rot = _quat_wxyz_to_matrix(
            np.asarray(
                [
                    peg_quat_xyzw[3],
                    peg_quat_xyzw[0],
                    peg_quat_xyzw[1],
                    peg_quat_xyzw[2],
                ],
                dtype=np.float64,
            )
        )
        peg_axis_xy = peg_rot[:2, 2].astype(np.float64, copy=True)
        axis_norm = float(np.linalg.norm(peg_axis_xy))
        if axis_norm < 1e-9:
            peg_axis_xy = np.asarray([1.0, 0.0], dtype=np.float64)
        else:
            peg_axis_xy /= axis_norm
        return (
            peg_pose[:2] + peg_axis_xy * _B03_RESET_PEG_MIN_LOCAL_Z,
            peg_pose[:2] + peg_axis_xy * _B03_RESET_PEG_MAX_LOCAL_Z,
        )

    @staticmethod
    def _point_to_segment_distance_xy(
        point: np.ndarray,
        segment_start: np.ndarray,
        segment_end: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        point_np = np.asarray(point, dtype=np.float64).reshape(2)
        start = np.asarray(segment_start, dtype=np.float64).reshape(2)
        end = np.asarray(segment_end, dtype=np.float64).reshape(2)
        segment = end - start
        segment_len_sq = float(np.dot(segment, segment))
        if segment_len_sq < 1e-12:
            closest = start
        else:
            t = float(np.clip(np.dot(point_np - start, segment) / segment_len_sq, 0.0, 1.0))
            closest = start + t * segment
        return float(np.linalg.norm(point_np - closest)), closest

    @classmethod
    def _b03_clear_reset_pose_from_right_hand(
        cls,
        pose_xyzw: np.ndarray,
        *,
        use_peg_segment: bool,
        avoid_xy: np.ndarray = _B03_RESET_RIGHT_HAND_AVOID_XY,
        clearance: float = _B03_RESET_RIGHT_HAND_CLEARANCE,
    ) -> np.ndarray:
        pose = np.asarray(pose_xyzw, dtype=np.float64).copy().reshape(-1, 7)
        if pose.shape[0] == 0:
            return pose
        fallback = _B03_RESET_FALLBACK_DIRECTION.copy()
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm < 1e-9:
            fallback = np.asarray([-1.0, 0.0], dtype=np.float64)
        else:
            fallback /= fallback_norm
        for index in range(pose.shape[0]):
            for _attempt in range(6):
                nearest_distance = math.inf
                nearest_delta = fallback
                for hand_xy in avoid_xy:
                    if use_peg_segment:
                        segment_start, segment_end = cls._b03_peg_segment_xy(pose[index])
                        distance, closest = cls._point_to_segment_distance_xy(
                            hand_xy,
                            segment_start,
                            segment_end,
                        )
                        delta = closest - hand_xy
                    else:
                        delta = pose[index, :2] - hand_xy
                        distance = float(np.linalg.norm(delta))
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_delta = delta
                if nearest_distance >= clearance:
                    break
                if nearest_distance < 1e-9:
                    direction = fallback
                else:
                    direction = nearest_delta / nearest_distance
                pose[index, :2] += direction * (
                    clearance - nearest_distance
                )
        return pose

    @classmethod
    def _b03_clear_socket_reset_pose_from_hands(
        cls,
        socket_pose_xyzw: np.ndarray,
    ) -> np.ndarray:
        return cls._b03_clear_reset_pose_from_right_hand(
            socket_pose_xyzw,
            use_peg_segment=False,
            avoid_xy=_B03_RESET_SOCKET_HAND_AVOID_XY,
            clearance=_B03_RESET_SOCKET_HAND_CLEARANCE,
        )

    @staticmethod
    def _b03_clear_socket_reset_pose(
        peg_pose_xyzw: np.ndarray,
        socket_pose_xyzw: np.ndarray,
    ) -> np.ndarray:
        """Keep the B03 socket tray clear of the horizontal peg during reset."""
        peg_pose = np.asarray(peg_pose_xyzw, dtype=np.float64).copy().reshape(-1, 7)
        socket_pose = np.asarray(socket_pose_xyzw, dtype=np.float64).copy().reshape(-1, 7)
        if peg_pose.shape[0] == 0 or socket_pose.shape[0] == 0:
            return socket_pose

        segment_start, segment_end = HandsOnABCIsaacLabEnv._b03_peg_segment_xy(peg_pose[0])
        segment = segment_end - segment_start
        segment_len_sq = float(np.dot(segment, segment))
        if segment_len_sq < 1e-12:
            peg_axis_xy = np.asarray([-1.0, 0.0], dtype=np.float64)
        else:
            peg_axis_xy = segment / math.sqrt(segment_len_sq)

        for index in range(socket_pose.shape[0]):
            socket_xy = socket_pose[index, :2]
            if segment_len_sq < 1e-12:
                closest = segment_start
            else:
                t = float(
                    np.clip(
                        np.dot(socket_xy - segment_start, segment) / segment_len_sq,
                        0.0,
                        1.0,
                    )
                )
                closest = segment_start + t * segment
            delta = socket_xy - closest
            distance = float(np.linalg.norm(delta))
            if distance >= _B03_RESET_SOCKET_SEGMENT_CLEARANCE:
                continue
            if distance < 1e-9:
                delta = np.asarray([-peg_axis_xy[1], peg_axis_xy[0]], dtype=np.float64)
                distance = float(np.linalg.norm(delta))
            direction = delta / max(distance, 1e-9)
            socket_pose[index, :2] += direction * (
                _B03_RESET_SOCKET_SEGMENT_CLEARANCE - distance
            )
        return socket_pose

    def _c01_nail_depth(self) -> float:
        state = self._physics_dof_position("nail_slide_joint")
        if state is None:
            return 0.0
        current, initial = state
        return float(np.clip(initial - current, 0.0, 0.030))

    @staticmethod
    def _c01_nail_head_geometry(
        nail_pose: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the nail-head center and its outward/upward axis."""
        nail_matrix = _pose_vector_to_matrix(nail_pose)
        nail_axis = nail_matrix[:3, :3] @ _WORLD_UP
        nail_axis /= max(float(np.linalg.norm(nail_axis)), 1e-9)
        nail_top = (
            nail_matrix[:3, 3]
            + nail_matrix[:3, :3] @ _C01_NAIL_HEAD_LOCAL_TOP
        )
        return nail_top, nail_axis

    @staticmethod
    def _c01_hammer_face_hit_geometry(
        hammer_pose: np.ndarray,
        nail_pose: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Return face-tip position, nail distance, and nail-axis alignment."""
        hammer_matrix = _pose_vector_to_matrix(hammer_pose)
        face_tip = (
            hammer_matrix[:3, 3]
            + hammer_matrix[:3, :3] @ _C01_HAMMER_FACE_LOCAL_TIP
        )
        nail_top, nail_axis = HandsOnABCIsaacLabEnv._c01_nail_head_geometry(
            nail_pose
        )
        face_normal = hammer_matrix[:3, :3] @ np.asarray(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(face_tip - nail_top))
        downward_alignment = float(np.dot(face_normal, -nail_axis))
        return face_tip, distance, downward_alignment

    def _c01_hammer_guide_geometry(self) -> dict[str, Any] | None:
        """Return the C01 hammer-face ray and predicted wood impact point."""
        if self.task_name != "t08_hammer_nail":
            return None

        hammer_pose = self._pose_for_role("target")
        nail_pose = self._pose_for_role("receptacle")
        wood_pose = self._pose_for_role("wood")
        hammer_matrix = _pose_vector_to_matrix(hammer_pose)
        wood_matrix = _pose_vector_to_matrix(wood_pose)
        face_tip = (
            hammer_matrix[:3, 3]
            + hammer_matrix[:3, :3] @ _C01_HAMMER_FACE_LOCAL_TIP
        )
        face_direction = hammer_matrix[:3, :3] @ np.asarray(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        )
        face_direction /= max(float(np.linalg.norm(face_direction)), 1e-9)
        wood_axis = wood_matrix[:3, :3] @ _WORLD_UP
        wood_axis /= max(float(np.linalg.norm(wood_axis)), 1e-9)
        wood_top_center = wood_matrix[:3, 3] + wood_axis * _C01_WOOD_HALF_HEIGHT

        ray_length = _C01_HAMMER_GUIDE_DEFAULT_RAY_LENGTH
        impact_world = face_tip + face_direction * ray_length
        intersects_wood_plane = False
        denominator = float(np.dot(face_direction, wood_axis))
        if abs(denominator) > 1e-6:
            candidate_length = float(
                np.dot(wood_top_center - face_tip, wood_axis) / denominator
            )
            if 0.0 <= candidate_length <= _C01_HAMMER_GUIDE_MAX_RAY_LENGTH:
                ray_length = candidate_length
                impact_world = face_tip + face_direction * ray_length
                intersects_wood_plane = True

        wood_local = wood_matrix[:3, :3].T @ (impact_world - wood_matrix[:3, 3])
        on_wood_top = bool(
            intersects_wood_plane
            and abs(float(wood_local[0])) <= 0.050
            and abs(float(wood_local[1])) <= 0.035
        )
        nail_top, nail_axis = self._c01_nail_head_geometry(nail_pose)
        impact_offset = impact_world - nail_top
        impact_axial_offset = float(np.dot(impact_offset, nail_axis))
        impact_radial_error = float(
            np.linalg.norm(
                impact_offset - impact_axial_offset * nail_axis
            )
        )
        nail_plane_denominator = float(np.dot(nail_axis, wood_axis))
        if abs(nail_plane_denominator) > 1e-6:
            nail_target_world = nail_top + nail_axis * (
                np.dot(wood_top_center - nail_top, wood_axis)
                / nail_plane_denominator
            )
        else:
            nail_target_world = nail_top

        return {
            "ray_start_world": face_tip.astype(np.float64, copy=True),
            "ray_end_world": impact_world.astype(np.float64, copy=True),
            "impact_world": impact_world.astype(np.float64, copy=True),
            "nail_target_world": nail_target_world.astype(np.float64, copy=True),
            "intersects_wood_plane": bool(intersects_wood_plane),
            "on_wood_top": bool(on_wood_top),
            "impact_radial_error": float(impact_radial_error),
            "aligned": bool(
                on_wood_top
                and impact_radial_error <= _C01_MAX_FACE_NAIL_RADIAL_ERROR
            ),
        }

    def c01_hammer_guide_pixels(
        self,
        camera_key: str,
        image_shape: tuple[int, ...],
    ) -> dict[str, Any] | None:
        """Project the C01 hammer guide into a recording camera frame."""
        if not _bool_env("HANDSON_C01_HAMMER_GUIDE", True):
            return None
        guide = self._c01_hammer_guide_geometry()
        if guide is None:
            return None
        camera_pose = self._recording_camera_pose_matrix_for_key(camera_key)
        if camera_pose is None:
            return None
        height = int(image_shape[0])
        width = int(image_shape[1])
        if height <= 0 or width <= 0:
            return None
        projected = self._project_opengl_camera_points(
            camera_pose,
            np.stack(
                [
                    guide["ray_start_world"],
                    guide["ray_end_world"],
                    guide["impact_world"],
                    guide["nail_target_world"],
                ],
                axis=0,
            ),
            image_width=width,
            image_height=height,
            camera_key=camera_key,
        )
        if projected is None:
            return None
        return {
            "line_start": projected[0],
            "line_end": projected[1],
            "impact": projected[2],
            "nail_target": projected[3],
            "aligned": bool(guide["aligned"]),
            "on_wood_top": bool(guide["on_wood_top"]),
            "impact_radial_error": float(guide["impact_radial_error"]),
        }

    def _c01_hammer_grasp_contact(
        self,
        hammer: np.ndarray,
        tip_positions: dict[str, np.ndarray],
    ) -> tuple[bool, float, int]:
        contact, error, count = self._visible_segment_pinch_contact(
            hammer,
            tip_positions,
            side="right",
            segments_local=(_C01_HAMMER_GRIP_SEGMENT,),
            segment_radii=(_C01_HAMMER_HANDLE_RADIUS,),
            contact_margin=_C01_HAMMER_HANDLE_CONTACT_MARGIN,
            near_margin=_C01_HAMMER_HANDLE_NEAR_MARGIN,
            pinch_distance_range=(
                _C01_HAMMER_HANDLE_PINCH_MIN_DISTANCE,
                _C01_HAMMER_HANDLE_PINCH_MAX_DISTANCE,
            ),
            max_axial_separation=_C01_HAMMER_HANDLE_MAX_AXIAL_SEPARATION,
        )
        if contact and count < _C01_HAMMER_GRASP_MIN_CONTACT_COUNT:
            return False, float(error), int(count)
        return bool(contact), float(error), int(count)

    def _c01_wood_grasp_contact(
        self,
        wood: np.ndarray,
        tip_positions: dict[str, np.ndarray],
    ) -> tuple[bool, float, int]:
        return self._visible_box_pinch_contact(
            wood,
            tip_positions,
            side="left",
            center_local=_C01_WOOD_GRASP_BOX_CENTER,
            half_extents=_C01_WOOD_GRASP_BOX_HALF_EXTENTS,
            contact_margin=_C01_WOOD_GRASP_CONTACT_MARGIN,
            near_margin=_C01_WOOD_GRASP_NEAR_MARGIN,
            pinch_distance_range=_C01_WOOD_GRASP_PINCH_DISTANCE_RANGE,
        )

    def _c01_grasp_assist_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "c01_grasp_assist_enabled": bool(
                _bool_env("HANDSON_C01_GRASP_ASSIST", True)
            ),
        }
        for side, object_name in (("right", "hammer"), ("left", "wood")):
            state = self._c01_grasp_assists.get(side, {})
            candidate = self._c01_grasp_candidates.get(side, {})
            prefix = (
                "c01_hammer_grasp_assist"
                if object_name == "hammer"
                else "c01_wood_grasp_assist"
            )
            active = str(state.get("object_name", "")) == object_name
            pending = str(candidate.get("object_name", "")) == object_name
            info.update(
                {
                    f"{prefix}_active": bool(active),
                    f"{prefix}_pending": bool(pending),
                    f"{prefix}_candidate_elapsed_s": float(
                        candidate.get("elapsed_s", 0.0)
                    )
                    if pending
                    else 0.0,
                    f"{prefix}_attach_debounce_s": float(
                        _C01_GRASP_ATTACH_DEBOUNCE_SECONDS
                    ),
                    f"{prefix}_side": side if active else "",
                    f"{prefix}_contact_error": float(
                        state.get(
                            "contact_error",
                            candidate.get("contact_error", math.inf),
                        )
                    ),
                    f"{prefix}_contact_count": int(
                        state.get(
                            "contact_count",
                            candidate.get("contact_count", 0),
                        )
                    ),
                    f"{prefix}_applied": bool(state.get("applied", False)),
                    f"{prefix}_linear_error": float(
                        state.get("linear_error", math.inf)
                    ),
                    f"{prefix}_angular_error": float(
                        state.get("angular_error", math.inf)
                    ),
                }
            )
        return info

    def _update_c01_single_grasp_assist(
        self,
        side: str,
        object_name: str,
    ) -> None:
        palm = self._robot_link_pose_matrix(side)
        obj = self._current_body_matrix(object_name)
        state = self._c01_grasp_assists.get(side)
        raw_hand = self._raw_hand_command_max(side)

        if (
            state is not None
            and str(state.get("object_name", "")) == object_name
            and raw_hand <= _C01_GRASP_RELEASE_HAND_COMMAND
        ):
            self._c01_grasp_assists.pop(side, None)
            self._c01_grasp_candidates.pop(side, None)
            return
        if palm is None or obj is None:
            self._c01_grasp_candidates.pop(side, None)
            return

        if state is None or str(state.get("object_name", "")) != object_name:
            if raw_hand < _C01_GRASP_ATTACH_HAND_COMMAND:
                self._c01_grasp_candidates.pop(side, None)
                return
            tip_positions = self._wuji_fingertip_positions(side)
            if object_name == "hammer":
                contact, error, count = self._c01_hammer_grasp_contact(
                    obj,
                    tip_positions,
                )
            else:
                contact, error, count = self._c01_wood_grasp_contact(
                    obj,
                    tip_positions,
                )
            if not contact:
                self._c01_grasp_candidates.pop(side, None)
                return

            candidate = self._c01_grasp_candidates.get(side)
            elapsed = max(float(self._sim_dt), 0.0)
            if candidate is not None and str(candidate.get("object_name")) == object_name:
                elapsed += float(candidate.get("elapsed_s", 0.0))
            candidate = {
                "object_name": object_name,
                "elapsed_s": float(elapsed),
                "contact_error": float(error),
                "contact_count": int(count),
            }
            self._c01_grasp_candidates[side] = candidate
            if elapsed < _C01_GRASP_ATTACH_DEBOUNCE_SECONDS:
                return

            state = {
                "object_name": object_name,
                "hand_to_object": _invert_pose_matrix(palm) @ obj,
                "last_target": obj.copy(),
                "visual_hand": self._actual_hand_joint_pose(side),
                "contact_error": float(error),
                "contact_count": int(count),
                "applied": False,
                "linear_error": 0.0,
                "angular_error": 0.0,
                "attach_elapsed_s": float(elapsed),
            }
            self._c01_grasp_assists[side] = state
            self._c01_grasp_candidates.pop(side, None)

        target = palm @ np.asarray(
            state["hand_to_object"],
            dtype=np.float64,
        ).reshape(4, 4)
        applied, linear_error, angular_error = self._set_body_rigid_grasp_target(
            object_name,
            target,
            previous_target=state.get("last_target"),
        )
        state.update(
            {
                "applied": bool(applied),
                "linear_error": float(linear_error),
                "angular_error": float(angular_error),
            }
        )
        if applied:
            state["last_target"] = target.copy()

    def _update_c01_grasp_assists(self) -> None:
        if not _bool_env("HANDSON_C01_GRASP_ASSIST", True):
            self._c01_grasp_assists.clear()
            self._c01_grasp_candidates.clear()
            return
        self._update_c01_single_grasp_assist("right", "hammer")
        self._update_c01_single_grasp_assist("left", "wood")

    def _set_c01_strike_align_metrics(
        self,
        *,
        enabled: bool,
        active: bool,
        reason: str,
        radial_error: float = math.inf,
        axial_offset: float = math.inf,
        raw_downward_speed: float = 0.0,
        face_down_alignment: float = -1.0,
        hand_hammer_distance: float = math.inf,
        correction_step: float = 0.0,
    ) -> None:
        self._c01_strike_align_assist_metrics = {
            "c01_strike_align_assist_enabled": bool(enabled),
            "c01_strike_align_assist_active": bool(active),
            "c01_strike_align_assist_reason": str(reason),
            "c01_strike_align_radial_error": float(radial_error),
            "c01_strike_align_capture_radial_error": float(
                _C01_STRIKE_ALIGN_CAPTURE_RADIAL_ERROR
            ),
            "c01_strike_align_axial_offset": float(axial_offset),
            "c01_strike_align_raw_downward_speed": float(
                raw_downward_speed
            ),
            "c01_strike_align_min_raw_downward_speed": float(
                _C01_STRIKE_ALIGN_MIN_RAW_DOWNWARD_SPEED
            ),
            "c01_strike_align_face_down_alignment": float(
                face_down_alignment
            ),
            "c01_strike_align_hand_hammer_distance": float(
                hand_hammer_distance
            ),
            "c01_strike_align_correction_step": float(correction_step),
        }

    def _apply_c01_strike_align_assist(self) -> None:
        """Nudge a real downward hammer command laterally toward the nail head."""
        enabled = _bool_env("HANDSON_C01_STRIKE_ALIGN_ASSIST", True)
        try:
            raw_tcp = _pose_vector_to_matrix(self._c_assist_raw_right_tcp_pose)
            command_tcp = _pose_vector_to_matrix(self._right_tcp_pose)
        except Exception:
            self._set_c01_strike_align_metrics(
                enabled=enabled,
                active=False,
                reason="invalid_tcp",
            )
            return

        previous_raw = self._c01_strike_align_prev_raw_tcp
        self._c01_strike_align_prev_raw_tcp = raw_tcp.copy()
        if not enabled:
            self._set_c01_strike_align_metrics(
                enabled=False,
                active=False,
                reason="disabled",
            )
            return
        if previous_raw is None:
            self._set_c01_strike_align_metrics(
                enabled=True,
                active=False,
                reason="waiting_for_previous_raw_tcp",
            )
            return
        if self._raw_right_hand_command_max() < _C01_STRIKE_ALIGN_MIN_HAND_COMMAND:
            self._set_c01_strike_align_metrics(
                enabled=True,
                active=False,
                reason="hand_not_closed",
            )
            return

        hammer = self._current_body_matrix("hammer")
        nail = self._current_body_matrix("nail")
        if hammer is None or nail is None:
            self._set_c01_strike_align_metrics(
                enabled=True,
                active=False,
                reason="missing_hammer_or_nail_pose",
            )
            return

        nail_axis = nail[:3, :3] @ _WORLD_UP
        nail_axis_norm = float(np.linalg.norm(nail_axis))
        if nail_axis_norm <= 1e-9:
            self._set_c01_strike_align_metrics(
                enabled=True,
                active=False,
                reason="invalid_nail_axis",
            )
            return
        nail_axis = nail_axis / nail_axis_norm
        raw_delta = raw_tcp[:3, 3] - previous_raw[:3, 3]
        raw_downward_speed = max(
            0.0,
            -float(np.dot(raw_delta, nail_axis)) / max(float(self._control_dt), 1e-6),
        )

        face_tip = hammer[:3, 3] + hammer[:3, :3] @ _C01_HAMMER_FACE_LOCAL_TIP
        nail_top = nail[:3, 3] + nail[:3, :3] @ _C01_NAIL_HEAD_LOCAL_TOP
        face_normal = hammer[:3, :3] @ np.asarray(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        )
        face_down_alignment = float(np.dot(face_normal, -nail_axis))
        face_nail_offset = face_tip - nail_top
        axial_offset = float(np.dot(face_nail_offset, nail_axis))
        radial_offset = face_nail_offset - axial_offset * nail_axis
        radial_error = float(np.linalg.norm(radial_offset))

        palm = self._robot_link_pose_matrix("right")
        hand_reference = raw_tcp if palm is None else palm
        hand_hammer_distance = float(
            np.linalg.norm(hand_reference[:3, 3] - hammer[:3, 3])
        )

        common_metrics = {
            "radial_error": radial_error,
            "axial_offset": axial_offset,
            "raw_downward_speed": raw_downward_speed,
            "face_down_alignment": face_down_alignment,
            "hand_hammer_distance": hand_hammer_distance,
        }
        if raw_downward_speed < _C01_STRIKE_ALIGN_MIN_RAW_DOWNWARD_SPEED:
            self._set_c01_strike_align_metrics(
                enabled=True,
                active=False,
                reason="not_downward_strike",
                **common_metrics,
            )
            return
        if hand_hammer_distance > _C01_STRIKE_ALIGN_MAX_HAND_HAMMER_DISTANCE:
            self._set_c01_strike_align_metrics(
                enabled=True,
                active=False,
                reason="hand_far_from_hammer",
                **common_metrics,
            )
            return
        if face_down_alignment < _C01_STRIKE_ALIGN_MIN_FACE_DOWN_ALIGNMENT:
            self._set_c01_strike_align_metrics(
                enabled=True,
                active=False,
                reason="hammer_face_not_down",
                **common_metrics,
            )
            return
        if not (
            _C01_STRIKE_ALIGN_MIN_AXIAL_OFFSET
            <= axial_offset
            <= _C01_STRIKE_ALIGN_MAX_AXIAL_OFFSET
        ):
            self._set_c01_strike_align_metrics(
                enabled=True,
                active=False,
                reason="axial_offset_out_of_range",
                **common_metrics,
            )
            return
        if radial_error > _C01_STRIKE_ALIGN_CAPTURE_RADIAL_ERROR:
            self._set_c01_strike_align_metrics(
                enabled=True,
                active=False,
                reason="radial_error_out_of_range",
                **common_metrics,
            )
            return
        if radial_error <= _C01_STRIKE_ALIGN_DEADBAND:
            self._set_c01_strike_align_metrics(
                enabled=True,
                active=False,
                reason="already_aligned",
                **common_metrics,
            )
            return

        max_step = (
            _C01_STRIKE_ALIGN_MAX_LINEAR_SPEED
            * max(float(self._control_dt), 0.0)
        )
        correction_step = min(radial_error, max_step)
        if correction_step <= 1e-9:
            self._set_c01_strike_align_metrics(
                enabled=True,
                active=False,
                reason="zero_correction_budget",
                **common_metrics,
            )
            return

        correction = -radial_offset * (correction_step / radial_error)
        command_tcp[:3, 3] += correction
        self._right_tcp_pose = self._pose_matrix_to_vector(command_tcp)
        self._set_c01_strike_align_metrics(
            enabled=True,
            active=True,
            reason="active",
            correction_step=correction_step,
            **common_metrics,
        )

    def _recording_camera_pose_matrix_for_key(
        self,
        camera_key: str,
    ) -> np.ndarray | None:
        key = str(camera_key)
        if key in {
            "front",
            "main",
            "random_camera",
            "ego",
            "head_camera",
        }:
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = _quat_wxyz_to_matrix(
                np.asarray(_HEAD_RECORDING_CAMERA_QUAT_WXYZ, dtype=np.float64)
            )
            matrix[:3, 3] = np.asarray(_HEAD_RECORDING_CAMERA_EYE, dtype=np.float64)
            return matrix

        side_by_key = {
            "wrist_left": "left",
            "wrist_left_camera": "left",
            "left_wrist": "left",
            "wrist_right": "right",
            "wrist_right_camera": "right",
            "right_wrist": "right",
            "wrist": "right",
        }
        side = side_by_key.get(key)
        if side is None:
            return None
        if (
            self._legacy_wrist_cameras
            and not self._spec.bimanual
            and side == "left"
        ):
            side = "right"
        mount_poses = (
            _LEGACY_WRIST_RECORDING_CAMERA_MOUNT_POSES
            if self._legacy_wrist_cameras
            else _WRIST_RECORDING_CAMERA_MOUNT_POSES
        )
        if self._legacy_wrist_cameras:
            site_pose = self._native_robot_site_pose_matrix(
                side,
                f"{side}_wrist_fisheye_optical_site",
            )
            if site_pose is not None:
                return site_pose
        mount_pose = self._native_robot_body_pose_matrix(
            side,
            f"{side}_wrist_fisheye_mount",
        )
        if mount_pose is not None:
            return (
                mount_pose
                @ _pose_vector_to_matrix(mount_poses[side])
            )
        fallback = self._left_tcp_pose if side == "left" else self._right_tcp_pose
        palm_pose = self._native_end_effector_pose_matrix(side, fallback)
        if palm_pose is None:
            return None
        return (
            palm_pose
            @ _pose_vector_to_matrix(_WRIST_RECORDING_CAMERA_LOCAL_POSES[side])
        )

    @staticmethod
    def _project_opengl_camera_points(
        camera_pose: np.ndarray,
        world_points: np.ndarray,
        *,
        image_width: int,
        image_height: int,
        camera_key: str,
    ) -> list[tuple[int, int]] | None:
        pose = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)
        points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
        camera_points = (pose[:3, :3].T @ (points - pose[:3, 3]).T).T
        depth = -camera_points[:, 2]
        if not np.all(np.isfinite(camera_points)) or np.any(depth <= 1e-5):
            return None

        key = str(camera_key)
        if key.startswith("wrist") or key.endswith("_wrist"):
            fx = float(image_width) / (
                2.0 * math.tan(math.radians(_WRIST_RECORDING_CAMERA_HFOV) / 2.0)
            )
            fy = fx
        else:
            fy = float(image_height) / (
                2.0 * math.tan(math.radians(_HEAD_RECORDING_CAMERA_FOVY) / 2.0)
            )
            fx = fy
        cx = 0.5 * float(image_width - 1)
        cy = 0.5 * float(image_height - 1)
        pixels = np.empty((points.shape[0], 2), dtype=np.float64)
        pixels[:, 0] = cx + fx * camera_points[:, 0] / depth
        pixels[:, 1] = cy - fy * camera_points[:, 1] / depth
        if not np.all(np.isfinite(pixels)):
            return None
        max_abs = float(max(image_width, image_height) * 4)
        pixels = np.clip(pixels, -max_abs, max_abs)
        return [
            (int(round(float(pixel[0]))), int(round(float(pixel[1]))))
            for pixel in pixels
        ]

    def _update_c01_hammer_hit_state(self) -> float:
        """Sample one PhysX frame and latch only a real hammer-face strike."""
        depth = self._c01_nail_depth()
        hammer = self._pose_for_role("target")
        nail = self._pose_for_role("receptacle")
        face_tip, face_nail_distance, face_down_alignment = (
            self._c01_hammer_face_hit_geometry(hammer, nail)
        )
        nail_top, nail_axis = self._c01_nail_head_geometry(nail)
        face_nail_offset = face_tip - nail_top
        face_nail_axial_offset = float(np.dot(face_nail_offset, nail_axis))
        face_nail_radial_error = float(
            np.linalg.norm(
                face_nail_offset - face_nail_axial_offset * nail_axis
            )
        )
        downward_speed = 0.0
        motion_down_alignment = -1.0
        if (
            self._c01_prev_hammer_pose is not None
            and self._c01_prev_nail_pose is not None
        ):
            previous_face_tip, _distance, _alignment = (
                self._c01_hammer_face_hit_geometry(
                    self._c01_prev_hammer_pose,
                    self._c01_prev_nail_pose,
                )
            )
            previous_nail_top, _previous_nail_axis = (
                self._c01_nail_head_geometry(self._c01_prev_nail_pose)
            )
            relative_face_motion = (
                face_tip - nail_top
            ) - (previous_face_tip - previous_nail_top)
            relative_motion_distance = float(
                np.linalg.norm(relative_face_motion)
            )
            if relative_motion_distance > 1e-9:
                downward_distance = max(
                    0.0,
                    -float(np.dot(relative_face_motion, nail_axis)),
                )
                downward_speed = downward_distance / max(self._sim_dt, 1e-6)
                motion_down_alignment = (
                    downward_distance / relative_motion_distance
                )

        contact_force, contact_available = self._t08_hammer_nail_contact_force()
        self._c01_hit_contact_available = bool(contact_available)
        self._c01_last_hit_contact_force = float(contact_force)
        self._c01_last_face_nail_distance = float(face_nail_distance)
        self._c01_last_face_nail_radial_error = float(
            face_nail_radial_error
        )
        self._c01_last_face_nail_axial_offset = float(
            face_nail_axial_offset
        )
        self._c01_last_face_down_alignment = float(face_down_alignment)
        self._c01_last_hammer_downward_speed = float(downward_speed)
        self._c01_last_hammer_motion_down_alignment = float(
            motion_down_alignment
        )

        if self._c01_prev_nail_depth is not None:
            depth_increment = max(0.0, depth - self._c01_prev_nail_depth)
            strike_kinematics_ok = (
                face_nail_distance <= _C01_MAX_FACE_NAIL_DISTANCE
                and face_nail_radial_error
                <= _C01_MAX_FACE_NAIL_RADIAL_ERROR
                and face_nail_axial_offset
                >= _C01_MIN_FACE_NAIL_AXIAL_OFFSET
                and face_nail_axial_offset
                <= _C01_MAX_FACE_NAIL_AXIAL_OFFSET
                and face_down_alignment >= _C01_MIN_FACE_DOWN_ALIGNMENT
                and downward_speed >= _C01_MIN_HAMMER_SPEED
                and motion_down_alignment
                >= _C01_MIN_MOTION_DOWN_ALIGNMENT
            )
            if strike_kinematics_ok:
                self._c01_had_valid_hit = True
                if self._c01_candidate_strike_depth_start is None:
                    self._c01_candidate_strike_depth_start = float(
                        self._c01_prev_nail_depth
                    )
                    self._c01_candidate_strike_max_depth = float(depth)
                else:
                    self._c01_candidate_strike_max_depth = max(
                        self._c01_candidate_strike_max_depth,
                        float(depth),
                    )
                candidate_increment = max(
                    0.0,
                    self._c01_candidate_strike_max_depth
                    - float(self._c01_candidate_strike_depth_start),
                )
            else:
                self._c01_candidate_strike_depth_start = None
                self._c01_candidate_strike_max_depth = float(depth)
                candidate_increment = 0.0

            self._c01_last_valid_hit_depth_increment = float(depth_increment)
            self._c01_last_candidate_hit_depth_increment = float(
                candidate_increment
            )
            # Some Isaac Sim backends resolve this articulated contact while
            # returning an all-zero filtered contact tensor.  Actual positive
            # joint travel during a correctly aligned downward face strike is
            # itself strict physical evidence: it is never synthesized here
            # and no task code advances the nail joint.  Accumulate the travel
            # across a continuous strike window so a real impact split over
            # several PhysX substeps is not missed.
            if candidate_increment >= _C01_MIN_DEPTH_INCREMENT:
                self._c01_max_valid_hit_depth = max(
                    self._c01_max_valid_hit_depth,
                    depth,
                )
        self._c01_prev_nail_depth = depth
        self._c01_prev_hammer_pose = hammer.copy()
        self._c01_prev_nail_pose = nail.copy()
        return depth

    def _t08_hammer_nail_success(self) -> bool:
        depth = self._update_c01_hammer_hit_state()
        effective_depth = max(depth, self._c01_max_valid_hit_depth)
        wood = self._pose_for_role("wood")
        table_z = self._current_table_top_z()
        wood_clearance = float(wood[2] - _C01_WOOD_HALF_HEIGHT - table_z)
        return bool(
            self._c01_had_valid_hit
            and effective_depth >= _C01_SUCCESS_DEPTH
            and wood_clearance >= _C01_WOOD_LIFT_CLEARANCE
        )

    def _block_stack_stable(self) -> bool:
        metrics = self._block_stack_metrics()
        self._apply_b04_velocity_stability_filter(metrics)
        self._last_b04_stack_metrics = metrics
        return all(
            bool(metrics[metric_key])
            for metric_key, _label in _B04_SUCCESS_CONDITIONS
        )

    def _apply_b04_velocity_stability_filter(
        self,
        metrics: dict[str, Any],
    ) -> None:
        """Debounce contact-solver velocity spikes without hiding real motion."""
        raw_linear = float(metrics["stack_max_linear_speed"])
        raw_angular = float(metrics["stack_max_angular_speed"])
        available = bool(metrics["stack_velocity_available"])
        instant_stable = bool(metrics["stack_velocity_stable"])
        metrics["stack_velocity_instant_stable"] = instant_stable

        if not available or not np.isfinite(raw_linear + raw_angular):
            self._b04_velocity_samples = []
            self._b04_velocity_stable_latched = False
            metrics.update(
                stack_velocity_stable=False,
                stack_filtered_linear_speed=math.inf,
                stack_filtered_angular_speed=math.inf,
                stack_velocity_filter_samples=0,
                stack_velocity_stable_latched=False,
            )
            return

        self._b04_velocity_samples.append((raw_linear, raw_angular))
        del self._b04_velocity_samples[:-_STACK_VELOCITY_FILTER_WINDOW]
        samples = np.asarray(self._b04_velocity_samples, dtype=np.float64)
        filtered_linear, filtered_angular = np.median(samples, axis=0)

        hard_motion = bool(
            raw_linear > _STACK_HARD_MOTION_LINEAR_SPEED
            or raw_angular > _STACK_HARD_MOTION_ANGULAR_SPEED
        )
        if hard_motion:
            self._b04_velocity_stable_latched = False
        elif self._b04_velocity_stable_latched:
            if (
                filtered_linear > _STACK_STABLE_EXIT_LINEAR_SPEED
                or filtered_angular > _STACK_STABLE_EXIT_ANGULAR_SPEED
            ):
                self._b04_velocity_stable_latched = False
        elif instant_stable:
            self._b04_velocity_stable_latched = True
            # Start a fresh window when the tower has genuinely re-entered
            # the strict thresholds; stale motion samples must not unlatch it
            # again on the following frame.
            self._b04_velocity_samples = [(raw_linear, raw_angular)]
            filtered_linear = raw_linear
            filtered_angular = raw_angular

        metrics.update(
            stack_velocity_stable=bool(self._b04_velocity_stable_latched),
            stack_filtered_linear_speed=float(filtered_linear),
            stack_filtered_angular_speed=float(filtered_angular),
            stack_velocity_filter_samples=len(self._b04_velocity_samples),
            stack_velocity_stable_latched=bool(
                self._b04_velocity_stable_latched
            ),
            stack_stable_exit_linear_speed=float(
                _STACK_STABLE_EXIT_LINEAR_SPEED
            ),
            stack_stable_exit_angular_speed=float(
                _STACK_STABLE_EXIT_ANGULAR_SPEED
            ),
        )

    def _block_stack_metrics(self) -> dict[str, Any]:
        block_names = [
            name
            for name in self._spec.target_objects
            if name in _STACK_BLOCK_HALF_EXTENTS
        ]
        if len(block_names) < 4:
            return {
                "stack_geometry_valid": False,
                "stack_four_layers": False,
                "stack_large_block_on_bottom": False,
                "stack_support_graph_valid": False,
                "stack_bottom_block": "",
                "stack_min_layer_gap": 0.0,
                "stack_min_required_layer_gap": float(_STACK_MIN_LAYER_GAP),
                "stack_xy_aligned": False,
                "stack_z_gaps_ok": False,
                "stack_upright": False,
                "stack_bottom_on_table": False,
                "stack_height_ok": False,
                "stack_hand_released": False,
                "stack_velocity_stable": False,
                "stack_velocity_available": False,
                "stack_max_qvel": math.inf,
                "stack_max_linear_speed": math.inf,
                "stack_max_angular_speed": math.inf,
                "stack_stable_linear_speed": float(_STACK_STABLE_LINEAR_SPEED),
                "stack_stable_angular_speed": float(_STACK_STABLE_ANGULAR_SPEED),
                "stack_max_xy_error": math.inf,
                "stack_max_z_gap_error": math.inf,
                "stack_max_support_z_error": math.inf,
                "stack_min_support_xy_overlap": -math.inf,
                "stack_min_upright_alignment": -1.0,
                "stack_bottom_clearance": math.inf,
                "tower_height": 0.0,
                "target_height": 0.0,
                "stack_hand_min_distance": math.inf,
                "stack_hand_contact_distance": float(_STACK_HAND_CONTACT_DISTANCE),
                "stack_hand_contact_force": math.inf,
                "stack_hand_contact_force_threshold": float(
                    _STACK_HAND_CONTACT_FORCE_THRESHOLD
                ),
                "stack_hand_contact_available": False,
                "stack_hand_release_source": "unavailable",
                "stack_order": "",
            }

        poses = {
            name: self._poses.get(name, self._fallback_pose(name))
            for name in block_names
        }
        vertical_half_extents = {
            name: self._stack_vertical_half_extent(name, poses[name])
            for name in block_names
        }
        sorted_names = sorted(block_names, key=lambda name: float(poses[name][2]))
        bottom_xy = poses[sorted_names[0]][:2]
        table_z = self._current_table_top_z()

        max_xy_error = max(
            float(np.linalg.norm(poses[name][:2] - bottom_xy))
            for name in sorted_names
        )
        min_upright = min(
            self._stack_upright_alignment(name, poses[name])
            for name in sorted_names
        )
        xy_aligned = max_xy_error <= _STACK_XY_ALIGNMENT_TOLERANCE
        upright = min_upright >= _STACK_UPRIGHT_ALIGNMENT_THRESHOLD

        max_z_gap_error = 0.0
        for lower, upper in zip(sorted_names, sorted_names[1:]):
            expected_gap = float(
                vertical_half_extents[lower]
                + vertical_half_extents[upper]
            )
            actual_gap = float(poses[upper][2] - poses[lower][2])
            max_z_gap_error = max(
                max_z_gap_error,
                abs(actual_gap - expected_gap),
            )
        z_gaps_ok = max_z_gap_error <= _STACK_Z_GAP_TOLERANCE

        bottom_clearance = float(
            poses[sorted_names[0]][2]
            - table_z
            - vertical_half_extents[sorted_names[0]]
        )
        bottom_on_table = abs(bottom_clearance) <= _STACK_SUPPORT_Z_TOLERANCE
        tower_height = self._stack_tower_height(
            poses,
            sorted_names,
            vertical_half_extents,
        )
        target_height = self._stack_target_height(sorted_names, vertical_half_extents)
        height_ok = tower_height >= target_height - _STACK_HEIGHT_TOLERANCE

        hand_metrics = self._block_stack_hand_metrics(poses)
        max_linear_speed, max_angular_speed, velocity_available = (
            self._block_stack_max_speeds(sorted_names)
        )
        velocity_stable = bool(
            velocity_available
            and max_linear_speed <= _STACK_STABLE_LINEAR_SPEED
            and max_angular_speed <= _STACK_STABLE_ANGULAR_SPEED
        )
        layer_metrics = self._block_stack_layer_metrics(
            poses,
            sorted_names,
            vertical_half_extents,
        )
        support_metrics = self._block_stack_support_metrics(
            poses,
            sorted_names,
            vertical_half_extents,
        )
        geometry_metrics = {
            **layer_metrics,
            **support_metrics,
            "stack_upright": bool(upright),
            "stack_bottom_on_table": bool(bottom_on_table),
        }
        stack_geometry_valid = all(
            bool(geometry_metrics[key]) for key in _STACK_GEOMETRY_METRIC_KEYS
        )
        return {
            "stack_geometry_valid": stack_geometry_valid,
            **layer_metrics,
            **support_metrics,
            "stack_xy_aligned": bool(xy_aligned),
            "stack_z_gaps_ok": bool(z_gaps_ok),
            "stack_upright": bool(upright),
            "stack_bottom_on_table": bool(bottom_on_table),
            "stack_height_ok": bool(height_ok),
            "stack_hand_released": bool(hand_metrics["released"]),
            "stack_velocity_stable": bool(velocity_stable),
            "stack_velocity_available": bool(velocity_available),
            "stack_max_qvel": float(max(max_linear_speed, max_angular_speed)),
            "stack_max_linear_speed": float(max_linear_speed),
            "stack_max_angular_speed": float(max_angular_speed),
            "stack_stable_linear_speed": float(_STACK_STABLE_LINEAR_SPEED),
            "stack_stable_angular_speed": float(_STACK_STABLE_ANGULAR_SPEED),
            "stack_max_xy_error": float(max_xy_error),
            "stack_max_z_gap_error": float(max_z_gap_error),
            "stack_min_upright_alignment": float(min_upright),
            "stack_bottom_clearance": float(bottom_clearance),
            "tower_height": float(tower_height),
            "target_height": float(target_height),
            "stack_hand_min_distance": float(hand_metrics["min_distance"]),
            "stack_hand_contact_distance": float(_STACK_HAND_CONTACT_DISTANCE),
            "stack_hand_contact_force": float(hand_metrics["contact_force"]),
            "stack_hand_contact_force_threshold": float(
                _STACK_HAND_CONTACT_FORCE_THRESHOLD
            ),
            "stack_hand_contact_available": bool(
                hand_metrics["contact_available"]
            ),
            "stack_hand_release_source": str(hand_metrics["source"]),
            "stack_order": ",".join(sorted_names),
        }

    @staticmethod
    def _stack_vertical_half_extent(name: str, pose: np.ndarray) -> float:
        extents = _STACK_BLOCK_HALF_EXTENTS[name]
        rot = _quat_wxyz_to_matrix(np.asarray(pose, dtype=np.float64)[3:7])
        return float(np.dot(np.abs(rot[2, :]), extents))

    @staticmethod
    def _stack_upright_alignment(name: str, pose: np.ndarray) -> float:
        rot = _quat_wxyz_to_matrix(np.asarray(pose, dtype=np.float64)[3:7])
        if name in _STACK_ORIENTATION_FREE_BLOCKS:
            return float(np.max(np.abs(rot[2, :])))
        return float(rot[2, 2])

    def _stack_target_height(
        self,
        names: list[str] | tuple[str, ...],
        vertical_half_extents: dict[str, float] | None = None,
    ) -> float:
        extents = vertical_half_extents or {
            name: self._stack_vertical_half_extent(name, self._poses.get(name, self._fallback_pose(name)))
            for name in names
        }
        return float(
            sum(2.0 * float(extents[name]) for name in names)
        )

    def _stack_tower_height(
        self,
        poses: dict[str, np.ndarray],
        names: list[str] | tuple[str, ...],
        vertical_half_extents: dict[str, float] | None = None,
    ) -> float:
        if not names:
            return 0.0
        extents = vertical_half_extents or {
            name: self._stack_vertical_half_extent(name, poses[name])
            for name in names
        }
        table_z = self._current_table_top_z()
        top_z = max(
            float(poses[name][2]) + float(extents[name])
            for name in names
        )
        return float(top_z - table_z)

    def _block_stack_layer_metrics(
        self,
        poses: dict[str, np.ndarray],
        sorted_names: list[str] | tuple[str, ...],
        vertical_half_extents: dict[str, float] | None = None,
    ) -> dict[str, float | bool | str]:
        extents = vertical_half_extents or {
            name: self._stack_vertical_half_extent(name, poses[name])
            for name in sorted_names
        }
        layer_gaps: list[float] = []
        required_gaps: list[float] = []
        for lower, upper in zip(sorted_names, sorted_names[1:]):
            gap = float(poses[upper][2] - poses[lower][2])
            required = max(
                _STACK_MIN_LAYER_GAP,
                _STACK_LAYER_GAP_FRACTION
                * float(extents[lower] + extents[upper]),
            )
            layer_gaps.append(gap)
            required_gaps.append(required)

        four_layers = bool(
            len(sorted_names) == 4
            and len(layer_gaps) == 3
            and all(gap >= required for gap, required in zip(layer_gaps, required_gaps))
        )
        bottom_block = sorted_names[0] if sorted_names else ""
        return {
            "stack_four_layers": four_layers,
            "stack_large_block_on_bottom": bottom_block == _STACK_BOTTOM_BLOCK_NAME,
            "stack_bottom_block": bottom_block,
            "stack_min_layer_gap": float(min(layer_gaps) if layer_gaps else 0.0),
            "stack_min_required_layer_gap": float(
                min(required_gaps) if required_gaps else _STACK_MIN_LAYER_GAP
            ),
        }

    def _block_stack_support_metrics(
        self,
        poses: dict[str, np.ndarray],
        sorted_names: list[str] | tuple[str, ...],
        vertical_half_extents: dict[str, float] | None = None,
    ) -> dict[str, float | bool]:
        extents_z = vertical_half_extents or {
            name: self._stack_vertical_half_extent(name, poses[name])
            for name in sorted_names
        }
        z_errors: list[float] = []
        min_xy_overlaps: list[float] = []
        supported_edges: list[bool] = []
        for lower, upper in zip(sorted_names, sorted_names[1:]):
            lower_pose = np.asarray(poses[lower], dtype=np.float64)
            upper_pose = np.asarray(poses[upper], dtype=np.float64)
            lower_top = float(lower_pose[2] + extents_z[lower])
            upper_bottom = float(upper_pose[2] - extents_z[upper])
            z_error = abs(upper_bottom - lower_top)

            lower_xy = self._stack_xy_half_extents(lower, lower_pose)
            upper_xy = self._stack_xy_half_extents(upper, upper_pose)
            xy_overlap = (
                lower_xy
                + upper_xy
                - np.abs(upper_pose[:2] - lower_pose[:2])
            )
            min_overlap = float(np.min(xy_overlap))
            z_errors.append(z_error)
            min_xy_overlaps.append(min_overlap)
            supported_edges.append(
                z_error <= _STACK_SUPPORT_Z_TOLERANCE
                and min_overlap >= _STACK_SUPPORT_MIN_XY_OVERLAP
            )

        return {
            "stack_support_graph_valid": bool(
                len(sorted_names) == 4
                and len(supported_edges) == 3
                and all(supported_edges)
            ),
            "stack_max_support_z_error": float(
                max(z_errors) if z_errors else math.inf
            ),
            "stack_min_support_xy_overlap": float(
                min(min_xy_overlaps) if min_xy_overlaps else -math.inf
            ),
            "stack_support_z_tolerance": float(_STACK_SUPPORT_Z_TOLERANCE),
            "stack_support_min_xy_overlap": float(
                _STACK_SUPPORT_MIN_XY_OVERLAP
            ),
        }

    @staticmethod
    def _stack_xy_half_extents(name: str, pose: np.ndarray) -> np.ndarray:
        rot = _quat_wxyz_to_matrix(np.asarray(pose, dtype=np.float64)[3:7])
        return np.abs(rot[:2, :]) @ _STACK_BLOCK_HALF_EXTENTS[name]

    def _block_stack_max_speeds(
        self,
        names: list[str] | tuple[str, ...],
    ) -> tuple[float, float, bool]:
        max_linear_speed = 0.0
        max_angular_speed = 0.0
        available_count = 0
        for name in names:
            velocity = self._physics_body_velocity(name)
            if velocity is None:
                continue
            available_count += 1
            max_linear_speed = max(
                max_linear_speed,
                float(np.linalg.norm(velocity[:3])),
            )
            max_angular_speed = max(
                max_angular_speed,
                float(np.linalg.norm(velocity[3:])),
            )
        return (
            max_linear_speed,
            max_angular_speed,
            available_count == len(names),
        )

    def _block_stack_hand_metrics(
        self,
        poses: dict[str, np.ndarray],
    ) -> dict[str, float | bool | str]:
        contact_force, contact_available = self._b04_right_hand_contact_force()
        min_distance = math.inf
        # B04 is a right-arm task.  The parked left robot is present in the
        # shared scene but must not participate in release detection.
        for point in self._a06_hand_points("right"):
            for name, pose in poses.items():
                distance = self._point_to_block_surface_distance(
                    point,
                    pose,
                    _STACK_BLOCK_HALF_EXTENTS[name],
                )
                min_distance = min(min_distance, distance)

        if contact_available:
            released = contact_force <= _STACK_HAND_CONTACT_FORCE_THRESHOLD
            source = "physx_contact"
        elif self._launch_isaac:
            # A collection run must never silently fall back to body-origin
            # distances, because fingertip colliders are offset from those
            # origins.  Missing contact reporting therefore fails closed.
            released = False
            source = "contact_unavailable"
        else:
            # Interface-only tests do not launch PhysX.  Keep the geometric
            # fallback there, but real collection uses the contact tensor.
            released = min_distance >= _STACK_HAND_CONTACT_DISTANCE
            source = "distance_fallback"
        return {
            "min_distance": float(min_distance),
            "released": bool(released),
            "contact_force": float(contact_force),
            "contact_available": bool(contact_available),
            "source": source,
        }

    def _block_stack_hand_clearance(
        self,
        poses: dict[str, np.ndarray],
    ) -> tuple[float, bool]:
        """Compatibility wrapper for callers that only need distance/release."""
        metrics = self._block_stack_hand_metrics(poses)
        return float(metrics["min_distance"]), bool(metrics["released"])

    @staticmethod
    def _point_to_block_surface_distance(
        point: np.ndarray,
        block_pose: np.ndarray,
        half_extents: np.ndarray,
    ) -> float:
        point_np = np.asarray(point, dtype=np.float64).reshape(3)
        pose_np = np.asarray(block_pose, dtype=np.float64).reshape(7)
        extents = np.asarray(half_extents, dtype=np.float64).reshape(3)
        rot = _quat_wxyz_to_matrix(pose_np[3:7])
        local = rot.T @ (point_np - pose_np[:3])
        outside = np.maximum(np.abs(local) - extents, 0.0)
        return float(np.linalg.norm(outside))

    def _spoon_marbles_in_bowl(self) -> bool:
        marble = self._pose_for_role("target")
        bowl = self._pose_for_role("receptacle")
        spoon = self._poses.get("spoon")
        if spoon is None:
            return False
        if self._c02_spoon_start_pos is None:
            self._c02_spoon_start_pos = spoon[:3].copy()
            return False
        spoon_delta = spoon[:3] - self._c02_spoon_start_pos
        if float(np.linalg.norm(spoon_delta[:2])) >= 0.050 or float(spoon_delta[2]) >= 0.025:
            self._c02_spoon_used = True

        spoon_matrix = _pose_vector_to_matrix(spoon)
        marble_local = _invert_pose_matrix(spoon_matrix) @ _pose_vector_to_matrix(marble)
        rel = marble_local[:3, 3]
        spoon_face_up = self._z_axis_alignment(spoon)
        marble_in_spoon = (
            spoon_face_up >= float(np.cos(np.deg2rad(45.0)))
            and abs(float(rel[0])) <= 0.025
            and 0.030 <= float(rel[1]) <= 0.102
            and -0.008 <= float(rel[2]) <= 0.060
        )
        if marble_in_spoon:
            self._c02_spoon_loaded_seen = True

        spoon_bowl_center = spoon_matrix[:3, 3] + spoon_matrix[:3, :3] @ np.asarray(
            [0.0, 0.073, 0.018],
            dtype=np.float64,
        )
        radial_spoon_bowl = float(np.linalg.norm(spoon_bowl_center[:2] - bowl[:2]))
        spoon_z_rel = float(spoon_bowl_center[2] - bowl[2])
        if self._c02_spoon_loaded_seen and radial_spoon_bowl <= 0.105 and 0.050 <= spoon_z_rel <= 0.260:
            self._c02_spoon_over_bowl_seen = True

        marble_rel_bowl = _invert_pose_matrix(_pose_vector_to_matrix(bowl)) @ _pose_vector_to_matrix(marble)
        bowl_rel = marble_rel_bowl[:3, 3]
        marble_inside_bowl = float(np.linalg.norm(bowl_rel[:2])) <= 0.095 and 0.010 <= float(bowl_rel[2]) <= 0.130
        return bool(
            self._c02_spoon_used
            and self._c02_spoon_loaded_seen
            and self._c02_spoon_over_bowl_seen
            and marble_inside_bowl
        )

    def render(self):
        frames: dict[str, np.ndarray] = {}
        if self._sim is not None and self._cameras:
            self._update_recording_cameras()
            if self._deferred_arm_render:
                self._sim.render()
                for name, fallback_key in (
                    ("head_camera", "ego"),
                    ("wrist_left_camera", "wrist_left"),
                    ("wrist_right_camera", "wrist_right"),
                ):
                    camera = self._cameras.get(name)
                    if camera is None:
                        continue
                    camera.update(self._control_dt)
                    frames[name] = self._camera_rgb_frame(name, fallback_key)
            else:
                # A stable visibility set is shared by the viewport and all
                # recording cameras. Per-camera visibility switching polluted
                # RTX/DLSS temporal history and required four render passes.
                self._set_wrist_obstructions_hidden(True)
                self._sim.render()
                for name, fallback_key in (
                    ("head_camera", "ego"),
                    ("wrist_left_camera", "wrist_left"),
                    ("wrist_right_camera", "wrist_right"),
                ):
                    camera = self._cameras.get(name)
                    if camera is None:
                        continue
                    camera.update(self._control_dt)
                    frames[name] = self._camera_rgb_frame(name, fallback_key)
        for name, fallback_key in (
            ("head_camera", "ego"),
            ("wrist_left_camera", "wrist_left"),
            ("wrist_right_camera", "wrist_right"),
        ):
            if name not in frames:
                frames[name] = self._camera_rgb_frame(name, fallback_key)
        return [
            frames["head_camera"],
            frames["wrist_left_camera"],
            frames["wrist_right_camera"],
        ]

    def _camera_rgb_frame(self, sensor_name: str, fallback_key: str) -> np.ndarray:
        is_wrist = sensor_name.startswith("wrist_")
        if is_wrist:
            target_width, target_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
        else:
            target_height, target_width = self._render_height, self._render_width
        sensor_name = self._camera_aliases.get(sensor_name, sensor_name)
        camera = self._cameras.get(sensor_name)
        if camera is not None:
            try:
                frame = camera.data.output["rgb"][0]
                if hasattr(frame, "detach"):
                    frame = frame.detach().cpu().numpy()
                frame = np.asarray(frame)
                if frame.ndim == 3 and frame.shape[-1] >= 3:
                    frame = frame[..., :3]
                    if frame.dtype != np.uint8:
                        if frame.size and float(np.nanmax(frame)) <= 1.0:
                            frame = frame * 255.0
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    if frame.shape[:2] == (target_height, target_width):
                        return frame.copy()
                    return self._resize_rgb_nearest(frame, target_height, target_width)
            except Exception as exc:
                message = f"{sensor_name} RGB unavailable; using fallback frame: {exc}"
                if message not in self._isaac_warnings:
                    self._isaac_warnings.append(message)
                    print(f"[HandsOn/IsaacLab] {message}", flush=True)
        return self._fallback_camera_frame(
            fallback_key,
            height=target_height,
            width=target_width,
        )

    @staticmethod
    def _resize_rgb_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
        src_h, src_w = frame.shape[:2]
        if src_h <= 0 or src_w <= 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        y_idx = np.clip((np.arange(height) * src_h / height).astype(np.int64), 0, src_h - 1)
        x_idx = np.clip((np.arange(width) * src_w / width).astype(np.int64), 0, src_w - 1)
        return frame[y_idx][:, x_idx, :3].astype(np.uint8, copy=True)

    def _fallback_camera_frame(
        self,
        key: str,
        *,
        height: int | None = None,
        width: int | None = None,
    ) -> np.ndarray:
        frame = np.zeros(
            (
                self._render_height if height is None else int(height),
                self._render_width if width is None else int(width),
                3,
            ),
            dtype=np.uint8,
        )
        colors = {"ego": (58, 66, 70), "wrist_left": (64, 74, 92), "wrist_right": (92, 72, 62)}
        frame[:, :] = colors.get(key, (48, 48, 48))
        h, w, _ = frame.shape
        frame[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3, :] = (154, 154, 145)
        return frame

    def _compute_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        if self.image_obs:
            ego, wrist_left, wrist_right = self.render()
            obs["images"] = {
                "random_camera" if self.randomize else "ego": ego,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }
        right_tcp_pose = self._right_tcp_pose
        left_tcp_pose = self._left_tcp_pose
        right_hand = self._right_hand
        left_hand = self._left_hand
        gripper_pose = np.zeros(32 if self._spec.bimanual else self._hand_action_dim, dtype=np.float64)
        if self._spec.bimanual:
            gripper_pose[: min(16, right_hand.size)] = right_hand[: min(16, right_hand.size)]
            gripper_pose[16 : 16 + min(16, left_hand.size)] = left_hand[: min(16, left_hand.size)]
            tcp_pose = np.concatenate([right_tcp_pose, left_tcp_pose]).astype(np.float64)
        else:
            gripper_size = min(gripper_pose.size, right_hand.size)
            gripper_pose[:gripper_size] = right_hand[:gripper_size]
            tcp_pose = right_tcp_pose.astype(np.float64, copy=True)

        target_pose = self._pose_for_role("target")
        receptacle_pose = self._pose_for_role("receptacle")
        object_poses = np.stack(
            [
                self._poses.get(name, self._fallback_pose(name))
                for name in self._spec.target_objects
            ],
            axis=0,
        )
        target_index = max(
            0,
            self._spec.target_objects.index(self._target_object_name)
            if self._target_object_name in self._spec.target_objects
            else 0,
        )
        selector = np.zeros(len(self._spec.target_objects), dtype=np.float64)
        selector[target_index] = 1.0
        distractors = np.stack(
            [
                self._poses.get(name, self._fallback_pose(name))
                for name in self._spec.target_objects
                if name != self._target_object_name
            ],
            axis=0,
        ) if len(self._spec.target_objects) > 1 else np.zeros((0, 7), dtype=np.float64)
        nut_to_bolt = target_pose - receptacle_pose
        c07_metrics = self._last_c07_success_metrics
        c08_tongs_reading = self._physics_dof_position("c08_tongs_joint")
        c08_tongs_angle = (
            _C08_TONGS_REST_RAD
            if c08_tongs_reading is None
            else float(c08_tongs_reading[0])
        )
        c09_hinge_reading = self._physics_dof_position("c09_stapler_top_joint")
        c09_hinge_angle = 0.0 if c09_hinge_reading is None else float(c09_hinge_reading[0] - c09_hinge_reading[1])
        c10_metrics = self._last_c10_success_metrics
        c12_block_poses = np.stack(
            [self._poses.get(name, self._fallback_pose(name)) for name in _C12_BLOCK_NAMES],
            axis=0,
        )
        c12_baseline_pose = self._poses.get("c12_baseline_pad", self._fallback_pose("c12_baseline_pad"))
        c12_baseline_inverse = _invert_pose_matrix(
            _pose_vector_to_matrix(c12_baseline_pose)
        )
        c12_block_local_positions = [
            (
                c12_baseline_inverse
                @ np.asarray([*pose[:3], 1.0], dtype=np.float64)
            )[:3]
            for pose in c12_block_poses
        ]
        c12_line_errors = np.asarray(
            [abs(float(local[1])) for local in c12_block_local_positions],
            dtype=np.float64,
        )
        c12_xs = sorted(float(local[0]) for local in c12_block_local_positions)
        c12_gaps = np.asarray(
            [c12_xs[index + 1] - c12_xs[index] for index in range(len(c12_xs) - 1)],
            dtype=np.float64,
        )
        c12_alignment_error = np.asarray(
            [
                float(np.max(c12_line_errors)) if c12_line_errors.size else 0.0,
                float(c12_line_errors[0]) if c12_line_errors.size > 0 else 0.0,
                float(c12_line_errors[1]) if c12_line_errors.size > 1 else 0.0,
                float(c12_line_errors[2]) if c12_line_errors.size > 2 else 0.0,
                float(c12_gaps[0]) if c12_gaps.size > 0 else 0.0,
                float(c12_gaps[1]) if c12_gaps.size > 1 else 0.0,
            ],
            dtype=np.float64,
        )
        state = {
            "tcp_pose": tcp_pose,
            "gripper_pose": gripper_pose,
            "target_object_pose": target_pose,
            "receptacle_pose": receptacle_pose,
            "object_poses": object_poses,
            "target_selector": selector,
            "target_id": np.asarray([target_index], dtype=np.float64),
            "target_name": selector.copy(),
            "plate_pose": receptacle_pose,
            "distractor_poses": distractors,
            "target_ball_pose": target_pose,
            "tennis_ball_pose": self._poses.get("tennis_ball", self._fallback_pose("tennis_ball")),
            "golf_ball_pose": self._poses.get("golf_ball", self._fallback_pose("golf_ball")),
            "mini_soccer_ball_pose": self._poses.get("mini_soccer_ball", self._fallback_pose("mini_soccer_ball")),
            "bowl_pose": self._poses.get("024_bowl", receptacle_pose),
            "nut_pose": target_pose,
            "bolt_pose": receptacle_pose,
            "nut_to_bolt": nut_to_bolt,
            "assembly_metrics": np.asarray(
                [
                    np.linalg.norm(nut_to_bolt[:3]),
                    *nut_to_bolt[:3],
                    self._success_counter,
                    0.0,
                ],
                dtype=np.float64,
            ),
            "peg_pose": target_pose,
            "socket_pose": receptacle_pose,
            "peg_ori_pose": target_pose,
            "socket_ori_pose": receptacle_pose,
            "hammer_ori_pose": target_pose,
            "nail_ori_pose": receptacle_pose,
            "wood_ori_pose": self._pose_for_role("wood"),
            "nail_depth": np.asarray([self._c01_nail_depth()], dtype=np.float64),
            "spray_ori_pose": target_pose,
            "plant_ori_pose": receptacle_pose,
            "mouse_ori_pose": target_pose,
            "mousepad_ori_pose": receptacle_pose,
            "camera_ori_pose": target_pose,
            "logo_ori_pose": receptacle_pose,
            "pitcher_pose": self._poses.get("c07_pitcher", target_pose),
            "cup_pose": (
                self._poses.get("c07_cup", receptacle_pose)
                if self.task_name == "c07_pitcher_pour_cup"
                else self._poses.get("c08_glass_cup", receptacle_pose)
            ),
            "particle_count": np.asarray(
                [float(c07_metrics.get("c07_particles_inside", 0.0))],
                dtype=np.float64,
            ),
            "spill_count": np.asarray(
                [float(c07_metrics.get("c07_spilled_count", 0.0))],
                dtype=np.float64,
            ),
            "return_zone_pose": self._poses.get("c07_return_zone", self._fallback_pose("c07_return_zone")),
            "tongs_pose": self._poses.get("c08_tongs", target_pose),
            "tongs_open_angle": np.asarray([c08_tongs_angle], dtype=np.float64),
            "ice_cube_poses": np.stack(
                [self._poses.get(name, self._fallback_pose(name)) for name in _C08_ICE_NAMES],
                axis=0,
            ),
            "stapler_pose": self._poses.get("c09_stapler", self._fallback_pose("c09_stapler")),
            "stapler_hinge": np.asarray([c09_hinge_angle], dtype=np.float64),
            "paper_poses": np.stack(
                [self._poses.get(name, self._fallback_pose(name)) for name in _C09_PAPER_NAMES],
                axis=0,
            ),
            "staple_zone_pose": self._poses.get("c09_work_zone", self._fallback_pose("c09_work_zone")),
            "charger_plug_pose": self._poses.get("c10_charger_plug", self._fallback_pose("c10_charger_plug")),
            "power_strip_pose": self._poses.get("c10_power_strip", self._fallback_pose("c10_power_strip")),
            "charging_cable_pose": self._poses.get("c10_charging_cable", self._fallback_pose("c10_charging_cable")),
            "connection_state": np.asarray(
                [
                    float(bool(c10_metrics.get("c10_first_connected", False))),
                    float(bool(c10_metrics.get("c10_second_connected", False))),
                    float(bool(c10_metrics.get("c10_order_valid", False))),
                ],
                dtype=np.float64,
            ),
            "mallet_pose": self._poses.get("c11_mallet", target_pose),
            "xylophone_key_poses": np.stack(
                [self._poses.get(name, self._fallback_pose(name)) for name in _C11_KEY_NAMES],
                axis=0,
            ),
            "sequence_target": np.pad(
                np.asarray(
                    [
                        float(_C11_KEY_NAMES.index(name))
                        for name in self._c11_target_sequence
                    ],
                    dtype=np.float64,
                ),
                (
                    0,
                    _C11_MAX_SEQUENCE_LENGTH
                    - len(self._c11_target_sequence),
                ),
                constant_values=-1.0,
            ),
            "sequence_progress": np.asarray([float(self._c11_sequence_progress)], dtype=np.float64),
            "wrong_hit_count": np.asarray([float(self._c11_wrong_hits)], dtype=np.float64),
            "tool_pose": self._poses.get("c12_set_square", target_pose),
            "block_poses": c12_block_poses,
            "baseline_pose": c12_baseline_pose,
            "alignment_error": c12_alignment_error,
            "table_delta_height": self.delta_h.astype(np.float64, copy=True),
        }
        obs["state"] = state
        return obs

    def get_end_effector_pose_matrix(self):
        right = self._native_end_effector_pose_matrix("right", self._right_tcp_pose)
        if not self._spec.bimanual:
            return right
        left = self._native_end_effector_pose_matrix("left", self._left_tcp_pose)
        return right, left

    def _native_end_effector_pose_matrix(self, side: str, fallback_pose: np.ndarray) -> np.ndarray:
        robot = self._robots.get(side)
        body_index = self._ee_body_indices.get(side)
        if robot is None or body_index is None:
            return self._prim_pose_matrix(
                f"{side}_palm_link",
                _pose_vector_to_matrix(fallback_pose),
            )
        pose = robot.data.body_pose_w[0, int(body_index), :7]
        pose_np = pose.detach().cpu().numpy().astype(np.float64)
        if not np.all(np.isfinite(pose_np)):
            return _pose_vector_to_matrix(fallback_pose)
        return _pose_vector_to_matrix(pose_np)

    def _native_robot_body_pose_matrix(
        self,
        side: str,
        body_name: str,
    ) -> np.ndarray | None:
        robot = self._robots.get(side)
        if robot is None:
            return None
        cache_key = f"{side}:{body_name}"
        body_index = self._recording_camera_body_indices.get(cache_key)
        if body_index is None:
            try:
                body_index = next(
                    index
                    for index, name in enumerate(robot.body_names)
                    if body_name in name
                )
            except (StopIteration, TypeError):
                return None
            self._recording_camera_body_indices[cache_key] = int(body_index)
            print(
                "[HandsOn/IsaacLab] Wrist camera follows circular mount body "
                f"{robot.body_names[int(body_index)]}",
                flush=True,
            )
        try:
            pose = robot.data.body_pose_w[0, int(body_index), :7]
            pose_np = pose.detach().cpu().numpy().astype(np.float64)
        except Exception:
            return None
        if not np.all(np.isfinite(pose_np)):
            return None
        return _pose_vector_to_matrix(pose_np)

    def _native_robot_site_pose_matrix(
        self,
        side: str,
        site_name: str,
    ) -> np.ndarray | None:
        if self._stage is None:
            return None
        try:
            from pxr import Usd, UsdGeom

            side_title = str(side).capitalize()
            exact_path = (
                f"/World/HandsOnRobots/{side_title}Robot/"
                f"base_{side}/attachment_{side}/sites/{site_name}"
            )
            prim = self._stage.GetPrimAtPath(exact_path)
            if not prim.IsValid():
                root = self._stage.GetPrimAtPath(
                    f"/World/HandsOnRobots/{side_title}Robot"
                )
                if not root.IsValid():
                    return None
                prim = next(
                    (
                        candidate
                        for candidate in Usd.PrimRange(root)
                        if candidate.GetName() == site_name
                        and candidate.IsA(UsdGeom.Xformable)
                    ),
                    None,
                )
            if prim is None or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
                return None
            cache_key = f"{side}:site:{site_name}"
            if cache_key not in self._prim_cache:
                self._prim_cache[cache_key] = prim
                print(
                    "[HandsOn/IsaacLab] Wrist camera follows optical site "
                    f"{prim.GetPath()}",
                    flush=True,
                )
            transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            return _usd_transform_to_pose_matrix(transform)
        except Exception:
            return None

    def close(self) -> None:
        if self._success_condition_frame is not None:
            frame = self._success_condition_frame
            self._success_condition_frame = None
            self._success_condition_labels.clear()
            self._success_condition_text_labels.clear()
            try:
                frame.visible = False
            except Exception:
                pass
        if self._success_condition_window is not None:
            window = self._success_condition_window
            self._success_condition_window = None
            self._success_condition_labels.clear()
            self._success_condition_text_labels.clear()
            try:
                window.visible = False
                destroy = getattr(window, "destroy", None)
                if destroy is not None:
                    destroy()
            except Exception:
                pass
        if self._sim is not None:
            try:
                timeline = getattr(self._sim, "_timeline", None)
                if timeline is not None:
                    timeline.stop()
                clear_callbacks = getattr(self._sim, "clear_all_callbacks", None)
                if clear_callbacks is not None:
                    clear_callbacks()
            except Exception:
                pass
        self._cameras.clear()
        if getattr(self, "_fast_shutdown_on_close", False):
            # SimulationContext.stop() can block indefinitely after RTX camera
            # capture. The app's skip-cleanup path releases this worker
            # directly so the separate full-arm replay process can start.
            self._sim = None
        super().close()
