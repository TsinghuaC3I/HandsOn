from __future__ import annotations

import inspect
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from handson.sim.isaaclab import bimanual_assembly_env as assembly_env
from handson.sim.isaaclab import abc_collection_env as abc_env
from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
from handson.sim.isaaclab.abc_collection_env import _physx_gpu_buffer_kwargs
from handson.sim.isaaclab.abc_collection_env import _A01_OBJECT_FOOTPRINT_RADII
from handson.sim.isaaclab.abc_collection_env import _A01_RESET_MIN_CLEARANCE_SLACK
from handson.sim.isaaclab.abc_collection_env import _A01_RESET_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _A01_RIGHT_HAND_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _A01_SUPPORT_LOCAL_MIN_Z
from handson.sim.isaaclab.abc_collection_env import _A01_YAW_BOUNDS_DEG
from handson.sim.isaaclab.abc_collection_env import (
    _A06_SIDE_REST_SUPPORT_LOCAL_MIN_Z,
)
from handson.sim.isaaclab.abc_collection_env import _A02_RESET_MIN_CLEARANCE_SLACK
from handson.sim.isaaclab.abc_collection_env import _A02_RESET_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _A02_YAW_BOUNDS_DEG
from handson.sim.isaaclab.abc_collection_env import _A03_RESET_MIN_CLEARANCE_SLACK
from handson.sim.isaaclab.abc_collection_env import _A03_RESET_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _A03_YAW_BOUNDS_DEG
from handson.sim.isaaclab.abc_collection_env import _B01_LID_INSERTED_GOAL_LOCAL_Z
from handson.sim.isaaclab.abc_collection_env import _B01_PITCHER_BASE_PHYSICS
from handson.sim.isaaclab.abc_collection_env import _B01_RESET_MIN_CLEARANCE_SLACK
from handson.sim.isaaclab.abc_collection_env import _B01_SUPPORT_LOCAL_MIN_Z
from handson.sim.isaaclab.abc_collection_env import _B01_RESET_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _B01_YAW_BOUNDS_DEG
from handson.sim.isaaclab.abc_collection_env import _B02_COLLECTION_SAFE_PHYSICS_DT
from handson.sim.isaaclab.abc_collection_env import _B02_DEFAULT_PHYSICS_DT
from handson.sim.isaaclab.abc_collection_env import _B02_MUJOCO_PHYSICS_DT
from handson.sim.isaaclab.abc_collection_env import _B02_NUT_GRASP_PHYSICS
from handson.sim.isaaclab.abc_collection_env import _B03_RESET_PEG_MAX_LOCAL_Z
from handson.sim.isaaclab.abc_collection_env import _B03_RESET_PEG_MIN_LOCAL_Z
from handson.sim.isaaclab.abc_collection_env import _B03_RESET_MIN_CLEARANCE_SLACK
from handson.sim.isaaclab.abc_collection_env import _B03_RESET_SOCKET_HAND_AVOID_XY
from handson.sim.isaaclab.abc_collection_env import _B03_RESET_SOCKET_HAND_CLEARANCE
from handson.sim.isaaclab.abc_collection_env import _B03_RESET_SOCKET_SEGMENT_CLEARANCE
from handson.sim.isaaclab.abc_collection_env import _B03_ROKAE_PEG_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _B03_ROKAE_SOCKET_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _B03_STABLE_ANGULAR_SPEED
from handson.sim.isaaclab.abc_collection_env import _B03_STABLE_HOLD_SECONDS
from handson.sim.isaaclab.abc_collection_env import _B03_STABLE_LINEAR_SPEED
from handson.sim.isaaclab.abc_collection_env import _B04_SUPPORT_LOCAL_MIN_Z
from handson.sim.isaaclab.abc_collection_env import _B04_RESET_FOOTPRINT_RADII
from handson.sim.isaaclab.abc_collection_env import (
    _B04_RESET_HAND_CLEARANCE_MARGIN,
)
from handson.sim.isaaclab.abc_collection_env import (
    _B04_RESET_OBJECT_CLEARANCE_MARGIN,
)
from handson.sim.isaaclab.abc_collection_env import _B05_FOOTPRINT_HALF_EXTENTS
from handson.sim.isaaclab.abc_collection_env import _B05_MIN_COVERAGE_RATIO
from handson.sim.isaaclab.abc_collection_env import _B05_CLEAR_BOX_CENTER_OF_MASS
from handson.sim.isaaclab.abc_collection_env import _B05_CLEAR_BOX_COLLIDER_COUNT
from handson.sim.isaaclab.abc_collection_env import _B05_CLEAR_BOX_DIAGONAL_INERTIA
from handson.sim.isaaclab.abc_collection_env import _B05_CLEAR_BOX_MASS
from handson.sim.isaaclab.abc_collection_env import (
    _B05_CLEAR_BOX_PRINCIPAL_AXES_WXYZ,
)
from handson.sim.isaaclab.abc_collection_env import _B05_CONTACT_OFFSET
from handson.sim.isaaclab.abc_collection_env import _B05_REST_OFFSET
from handson.sim.isaaclab.abc_collection_env import _B05_RESET_SUPPORT_CLEARANCE
from handson.sim.isaaclab.abc_collection_env import _B05_SUPPORT_LOCAL_MIN_Z
from handson.sim.isaaclab.abc_collection_env import _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK
from handson.sim.isaaclab.abc_collection_env import _TASK_SPECIFIC_RESET_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _STACK_BLOCK_HALF_EXTENTS
from handson.sim.isaaclab.abc_collection_env import _C04_SUCCESS_STEPS
from handson.sim.isaaclab.abc_collection_env import _C04_PLANT_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _C04_PLANT_HAND_MAX_Z
from handson.sim.isaaclab.abc_collection_env import _C04_SPRAY_INIT_XY
from handson.sim.isaaclab.abc_collection_env import _C04_RESET_MIN_CLEARANCE_SLACK
from handson.sim.isaaclab.abc_collection_env import _C04_SPRAY_RIGHT_ARM_MIN_XY_DISTANCE
from handson.sim.isaaclab.abc_collection_env import _C04_SPRAY_SUPPORT_LOCAL_MIN_Z
from handson.sim.isaaclab.abc_collection_env import _C04_SPRAY_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _C04_SPRAY_YAW_BOUNDS_DEG
from handson.sim.isaaclab.abc_collection_env import _C04_TRIGGER_ARM_STEPS
from handson.sim.isaaclab.abc_collection_env import _C04_TRIGGER_DAMPING
from handson.sim.isaaclab.abc_collection_env import _C04_TRIGGER_FRICTION
from handson.sim.isaaclab.abc_collection_env import _C04_TRIGGER_MAX_FORCE
from handson.sim.isaaclab.abc_collection_env import _C04_TRIGGER_PULL_RAD
from handson.sim.isaaclab.abc_collection_env import _C04_TRIGGER_RELEASE_RAD
from handson.sim.isaaclab.abc_collection_env import _C04_TRIGGER_STIFFNESS
from handson.sim.isaaclab.abc_collection_env import _C04_WATER_CONE_SPREAD
from handson.sim.isaaclab.abc_collection_env import _C04_WATER_HIT_FRACTION_THRESHOLD
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_FACE_LOCAL_TIP
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_FLAT_RESET_QUAT_XYZW
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_HAND_SCREEN_CLEARANCE
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_ANGULAR_DAMPING
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_LINEAR_DAMPING
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_CONTACT_OFFSET
from handson.sim.isaaclab.abc_collection_env import (
    _C01_HAMMER_FOOTPRINT_VERTICES_LOCAL_XY,
)
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_GRIP_SEGMENT
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_HANDLE_RADIUS
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_MIN_FOOTPRINT_X
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_SUPPORT_LOCAL_MIN_Z
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_SCREEN_X_LIMIT
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_SCREEN_Y_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_SLEEP_THRESHOLD
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_SCREEN_Z
from handson.sim.isaaclab.abc_collection_env import (
    _C01_HAMMER_WOOD_COLLISION_CLEARANCE,
)
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_WOOD_MIN_DISTANCE
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_WOOD_SCREEN_CLEARANCE
from handson.sim.isaaclab.abc_collection_env import _C01_HAMMER_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _C01_ENABLE_CCD
from handson.sim.isaaclab.abc_collection_env import _C01_ENABLE_SPECULATIVE_CCD
from handson.sim.isaaclab.abc_collection_env import _C01_MAX_DEPENETRATION_VELOCITY
from handson.sim.isaaclab.abc_collection_env import _C01_MIN_HAMMER_SPEED
from handson.sim.isaaclab.abc_collection_env import _C01_NAIL_CONTACT_OFFSET
from handson.sim.isaaclab.abc_collection_env import _C01_NAIL_HEAD_LOCAL_TOP
from handson.sim.isaaclab.abc_collection_env import _C01_NAIL_JOINT_DAMPING
from handson.sim.isaaclab.abc_collection_env import _C01_NAIL_JOINT_FRICTION
from handson.sim.isaaclab.abc_collection_env import _C01_NAIL_JOINT_MAX_FORCE
from handson.sim.isaaclab.abc_collection_env import _C01_NAIL_JOINT_STIFFNESS
from handson.sim.isaaclab.abc_collection_env import _C01_NAIL_MASS
from handson.sim.isaaclab.abc_collection_env import _C01_SOLVER_POSITION_ITERATIONS
from handson.sim.isaaclab.abc_collection_env import _C01_SOLVER_VELOCITY_ITERATIONS
from handson.sim.isaaclab.abc_collection_env import _C01_GRASP_ATTACH_DEBOUNCE_SECONDS
from handson.sim.isaaclab.abc_collection_env import _C01_GRASP_ATTACH_HAND_COMMAND
from handson.sim.isaaclab.abc_collection_env import _C01_GRASP_RELEASE_HAND_COMMAND
from handson.sim.isaaclab.abc_collection_env import _C01_STRIKE_ALIGN_CAPTURE_RADIAL_ERROR
from handson.sim.isaaclab.abc_collection_env import _C01_STRIKE_ALIGN_MAX_LINEAR_SPEED
from handson.sim.isaaclab.abc_collection_env import _C01_STRIKE_ALIGN_MIN_RAW_DOWNWARD_SPEED
from handson.sim.isaaclab.abc_collection_env import _C01_WOOD_ANGULAR_DAMPING
from handson.sim.isaaclab.abc_collection_env import _C01_WOOD_CENTER_OF_MASS
from handson.sim.isaaclab.abc_collection_env import _C01_WOOD_HALF_HEIGHT
from handson.sim.isaaclab.abc_collection_env import _C01_WOOD_CENTER_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _C01_WOOD_DIAGONAL_INERTIA
from handson.sim.isaaclab.abc_collection_env import (
    _C01_WOOD_FOOTPRINT_HALF_EXTENTS,
)
from handson.sim.isaaclab.abc_collection_env import _C01_WOOD_SUPPORT_LOCAL_MIN_Z
from handson.sim.isaaclab.abc_collection_env import _C01_WOOD_GRASP_BOX_HALF_EXTENTS
from handson.sim.isaaclab.abc_collection_env import _C01_WOOD_HAND_MIN_XY_DISTANCE
from handson.sim.isaaclab.abc_collection_env import _C01_WOOD_LINEAR_DAMPING
from handson.sim.isaaclab.abc_collection_env import _C01_WOOD_MASS
from handson.sim.isaaclab.abc_collection_env import _C01_WOOD_SCREEN_Z
from handson.sim.isaaclab.abc_collection_env import _C01_WOOD_SLEEP_THRESHOLD
from handson.sim.isaaclab.abc_collection_env import _C01_WOOD_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _C02_SUPPORT_LOCAL_MIN_Z
from handson.sim.isaaclab.abc_collection_env import _C02_RESET_SUPPORT_CLEARANCE
from handson.sim.isaaclab.abc_collection_env import _C02_SPOON_REST_ANGLE_RAD
from handson.sim.isaaclab.abc_collection_env import _C02_SPOON_REST_QUAT_XYZW
from handson.sim.isaaclab.abc_collection_env import _C05_BUTTON_PRESS_THRESHOLD
from handson.sim.isaaclab.abc_collection_env import _C05_DEFAULT_MOUSEPAD_XY
from handson.sim.isaaclab.abc_collection_env import _C05_DEFAULT_MOUSE_XY
from handson.sim.isaaclab.abc_collection_env import _C05_MOUSE_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _C05_RESET_MIN_CLEARANCE_SLACK
from handson.sim.isaaclab.abc_collection_env import _C05_MOUSEPAD_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _C05_MOUSEPAD_MIN_OFFSET
from handson.sim.isaaclab.abc_collection_env import _C05_SUCCESS_STEPS
from handson.sim.isaaclab.abc_collection_env import _C06_CAMERA_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _C06_CAMERA_HAND_CLEARANCE
from handson.sim.isaaclab.abc_collection_env import _C06_CAMERA_YAW_BOUNDS_DEG
from handson.sim.isaaclab.abc_collection_env import _C06_RESET_MIN_CLEARANCE_SLACK
from handson.sim.isaaclab.abc_collection_env import _C06_DEFAULT_CAMERA_XY
from handson.sim.isaaclab.abc_collection_env import _C06_CAMERA_GRASP_PHYSICS
from handson.sim.isaaclab.abc_collection_env import _C06_SHUTTER_PRESS_TRAVEL
from handson.sim.isaaclab.abc_collection_env import _C07_CAMERA_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _C07_CUP_INNER_RADIUS
from handson.sim.isaaclab.abc_collection_env import _C07_CUP_FOOTPRINT_RADIUS
from handson.sim.isaaclab.abc_collection_env import _C07_CUP_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _C07_HAND_CLEARANCE
from handson.sim.isaaclab.abc_collection_env import _C07_HANDLE_FOOTPRINT_RADIUS
from handson.sim.isaaclab.abc_collection_env import _C07_NOMINAL_CUP_XY
from handson.sim.isaaclab.abc_collection_env import _C07_NOMINAL_PITCHER_XY
from handson.sim.isaaclab.abc_collection_env import _C07_MIN_PITCHER_CUP_DISTANCE
from handson.sim.isaaclab.abc_collection_env import _C07_MAX_PITCHER_CUP_DISTANCE
from handson.sim.isaaclab.abc_collection_env import (
    _C07_GRASP_ATTACH_HAND_COMMAND,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C07_GRASP_RELEASE_HAND_COMMAND,
)
from handson.sim.isaaclab.abc_collection_env import _C07_PARTICLE_EXIT_LOCAL
from handson.sim.isaaclab.abc_collection_env import (
    _C07_PARTICLE_MOUTH_CENTER_DISTANCE,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C07_PARTICLE_MOUTH_CENTER_LOCAL,
)
from handson.sim.isaaclab.abc_collection_env import _C07_PARTICLE_EXIT_X_OFFSETS
from handson.sim.isaaclab.abc_collection_env import (
    _C07_PARTICLE_FIRST_RELEASE_DELAY_SECONDS,
)
from handson.sim.isaaclab.abc_collection_env import _C07_PARTICLE_RADIUS
from handson.sim.isaaclab.abc_collection_env import _C07_PARTICLE_NAMES
from handson.sim.isaaclab.abc_collection_env import _C07_PARTICLE_RESET_ANGULAR_DAMPING
from handson.sim.isaaclab.abc_collection_env import _C07_RESET_LINEAR_DAMPING
from handson.sim.isaaclab.abc_collection_env import _C07_RESET_SLEEP_THRESHOLD
from handson.sim.isaaclab.abc_collection_env import _C07_RESET_STABILITY_BODIES
from handson.sim.isaaclab.abc_collection_env import _C07_VESSEL_RESET_ANGULAR_DAMPING
from handson.sim.isaaclab.abc_collection_env import _C07_PARTICLE_SUPPORT_ROOT_Z
from handson.sim.isaaclab.abc_collection_env import _C07_PARTICLE_XY_OFFSETS
from handson.sim.isaaclab.abc_collection_env import _C07_CUP_SUPPORT_LOCAL_MIN_Z
from handson.sim.isaaclab.abc_collection_env import (
    _C07_PITCHER_SUPPORT_LOCAL_MIN_Z,
)
from handson.sim.isaaclab.abc_collection_env import _C07_PITCHER_FOOTPRINT_RADIUS
from handson.sim.isaaclab.abc_collection_env import (
    _C07_PARTICLE_RELEASE_INTERVAL_SECONDS,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C07_PARTICLE_RELEASE_EXIT_HOLD_STEPS,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C07_PARTICLE_RELEASE_TOTAL_TILT_DEG,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C07_PARTICLE_RELEASE_STABLE_STEPS,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C07_PARTICLE_STAGE_SPEED_MPS,
)
from handson.sim.isaaclab.abc_collection_env import _C07_PITCHER_HALF_HEIGHT
from handson.sim.isaaclab.abc_collection_env import (
    _C07_POUR_ALIGN_DURATION_SECONDS,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C07_POUR_ALIGN_ROTATION_RATE_DEG_PER_S,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C07_POUR_ALIGN_TRANSLATION_RATE_MPS,
)
from handson.sim.isaaclab.abc_collection_env import _C07_POUR_FULL_TILT_INPUT_M
from handson.sim.isaaclab.abc_collection_env import _C07_POUR_FULL_TILT_WRIST_DEG
from handson.sim.isaaclab.abc_collection_env import (
    _C07_POUR_MAX_TOTAL_TILT_DEG,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C07_POUR_TILT_RATE_DEG_PER_S,
)
from handson.sim.isaaclab.abc_collection_env import _C07_RELEASE_HOLD_SECONDS
from handson.sim.isaaclab.abc_collection_env import _C07_SPOUT_OUTLET_LOCAL
from handson.sim.isaaclab.abc_collection_env import _C07_TASK_CONTACT_OFFSET
from handson.sim.isaaclab.abc_collection_env import _C07_WRENCH_MAX_FORCE
from handson.sim.isaaclab.abc_collection_env import _C07_WRENCH_MAX_TORQUE
from handson.sim.isaaclab.abc_collection_env import _C08_DROP_ICE_LOCAL_Z
from handson.sim.isaaclab.abc_collection_env import (
    _C08_DELIVERED_ICE_LOCAL_SLOTS,
)
from handson.sim.isaaclab.abc_collection_env import _C08_ICE_HALF_EXTENT
from handson.sim.isaaclab.abc_collection_env import _C08_ICE_NAMES
from handson.sim.isaaclab.abc_collection_env import (
    _C08_ICE_PAIR_CLEARANCE_MARGIN,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C08_OBJECT_YAW_BOUNDS_DEG,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C08_RESET_MIN_CLEARANCE_SLACK,
)
from handson.sim.isaaclab.abc_collection_env import _C08_RESET_XY_BOUNDS
from handson.sim.isaaclab.abc_collection_env import (
    _C08_RIGHT_HAND_XY_BOUNDS,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C08_CUP_POSE_MISSING_TIMEOUT_SECONDS,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C08_RELEASE_CENTERED_CUP_RADIUS,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C08_RELEASE_OPEN_FAILURE_TIMEOUT_SECONDS,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C08_RELEASE_OPEN_TIMEOUT_SECONDS,
)
from handson.sim.isaaclab.abc_collection_env import _C08_TONGS_SPRING_DAMPING
from handson.sim.isaaclab.abc_collection_env import _C08_TONGS_SPRING_MAX_FORCE
from handson.sim.isaaclab.abc_collection_env import _C08_TONGS_SPRING_STIFFNESS
from handson.sim.isaaclab.abc_collection_env import _C08_TONGS_CLOSED_RAD
from handson.sim.isaaclab.abc_collection_env import (
    _C08_TONGS_GRIP_JOINT_MAX_RATE,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C08_TONGS_MAX_ALLOWED_PENETRATION,
)
from handson.sim.isaaclab.abc_collection_env import _C08_TONGS_MAX_INNER_GAP
from handson.sim.isaaclab.abc_collection_env import _C08_TONGS_OPEN_RAD
from handson.sim.isaaclab.abc_collection_env import _C08_TONGS_PAD_RADIUS
from handson.sim.isaaclab.abc_collection_env import (
    _C08_TONGS_PICKUP_LATCH_CLEARANCE,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C08_TONGS_PICKUP_CONTACT_HOLD_SECONDS,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C08_TONGS_FOLLOW_FAILURE_TIMEOUT_SECONDS,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C08_TONGS_PICKUP_MIN_ROOT_LIFT,
)
from handson.sim.isaaclab.abc_collection_env import (
    _C08_TONGS_SUPPORT_ESCAPE_STEP,
)
from handson.sim.isaaclab.abc_collection_env import _C10_CABLE_SEATED_RELATIVE
from handson.sim.isaaclab.abc_collection_env import _C10_PLUG_SEATED_RELATIVE
from handson.sim.isaaclab.abc_collection_env import _C11_KEY_NAMES
from handson.sim.isaaclab.abc_collection_env import _C12_BLOCK_LOCAL_RIGHT_Y_BOUNDS
from handson.sim.isaaclab.abc_collection_env import _C12_BLOCK_NAMES
from handson.sim.isaaclab.abc_collection_env import _C12_RESET_MIN_CLEARANCE_SLACK
from handson.sim.isaaclab.abc_collection_env import (
    _C12_RESET_BLOCK_CLEARANCE_MARGIN,
)
from handson.sim.isaaclab.abc_collection_env import _C12_BLOCK_FOOTPRINT_RADIUS
from handson.sim.isaaclab.abc_collection_env import _C12_SUPPORT_LOCAL_MIN_Z
from handson.sim.isaaclab.abc_collection_env import _C12_SET_SQUARE_LOCAL_XY
from handson.sim.isaaclab.abc_collection_env import _C12_TOOL_SEGMENTS_LOCAL
from handson.sim.isaaclab.abc_collection_env import _C_TOOL_SPREAD_LEFT_START_TCP
from handson.sim.isaaclab.abc_collection_env import _C_TOOL_SPREAD_RIGHT_START_TCP
from handson.sim.isaaclab.abc_collection_env import _HEAD_RECORDING_CAMERA_FOVY
from handson.sim.isaaclab.abc_collection_env import _HEAD_RECORDING_CAMERA_QUAT_WXYZ
from handson.sim.isaaclab.abc_collection_env import _LID_GRASP_PHYSICS
from handson.sim.isaaclab.abc_collection_env import _A01_CLEAR_ARM_HOME
from handson.sim.isaaclab.abc_collection_env import _A01_CLEAR_RIGHT_START_TCP
from handson.sim.isaaclab.abc_collection_env import _RAISED_ARM_HOME
from handson.sim.isaaclab.abc_collection_env import _RAISED_LEFT_START_TCP
from handson.sim.isaaclab.abc_collection_env import _RAISED_RIGHT_START_TCP
from handson.sim.isaaclab.abc_collection_env import _RECORDING_CAMERA_NAMES
from handson.sim.isaaclab.abc_collection_env import _RECORDING_CAMERA_PRIM_PATHS
from handson.sim.isaaclab.abc_collection_env import _WRIST_RECORDING_CAMERA_FOVY
from handson.sim.isaaclab.abc_collection_env import _WRIST_RECORDING_CAMERA_HFOV
from handson.sim.isaaclab.abc_collection_env import _WRIST_RECORDING_CAMERA_LOCAL_POSES
from handson.sim.isaaclab.abc_collection_env import _WRIST_RECORDING_CAMERA_MOUNT_CLEARANCE
from handson.sim.isaaclab.abc_collection_env import _WRIST_RECORDING_CAMERA_MOUNT_POSES
from handson.sim.isaaclab.abc_collection_env import WRIST_RECORDING_CAMERA_OUTPUT_SIZE
from handson.sim.isaaclab.abc_collection_env import _b02_mujoco_hand_drive_maps
from handson.sim.isaaclab.abc_collection_env import _constrain_task_xy_to_workspace
from handson.sim.isaaclab.abc_collection_env import _focal_length_for_vertical_fov
from handson.sim.isaaclab.abc_collection_env import _focal_length_for_horizontal_fov
from handson.sim.isaaclab.abc_collection_env import _invert_pose_matrix
from handson.sim.isaaclab.abc_collection_env import _pose_vector_to_matrix
from handson.sim.isaaclab.abc_collection_env import _randomized_physx_pose
from handson.sim.isaaclab.abc_collection_env import _deferred_keeps_hand_visual_path
from handson.sim.isaaclab.bimanual_assembly_env import BimanualAssemblyIsaacLabEnv
from handson.sim.isaaclab.bimanual_assembly_env import _LEFT_HOME
from handson.sim.isaaclab.bimanual_assembly_env import _RIGHT_HOME
from handson.sim.isaaclab.recording_cameras import RECORDING_OBJECT_XY_BOUNDS


_ISAACLAB_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "handson" / "sim" / "isaaclab"


SHARED_ROBOT_METHODS = (
    "step",
    "_apply_action",
    "_compute_arm_joint_target",
    "_compute_arm_joint_target_haorui",
    "_limit_arm_pose_target",
    "_limit_arm_joint_target",
    "_init_haorui_rokae_ik",
    "_setup_timeline_pause_guard",
    "_on_timeline_pause",
    "_resume_timeline_if_paused",
    "_setup_keyboard_reset",
    "_on_keyboard_event",
    "_apply_keyboard_reset_immediately",
)


class _FakeTimeline:
    def __init__(self, *, playing: bool = False, stopped: bool = False):
        self.playing = playing
        self.stopped = stopped
        self.play_calls = 0

    def is_playing(self) -> bool:
        return self.playing

    def is_stopped(self) -> bool:
        return self.stopped

    def play(self) -> None:
        self.play_calls += 1
        self.playing = True


class _FakeKeyboardEventType:
    KEY_PRESS = object()


class _FakeKeyboardInput:
    R = object()
    V = object()
    Z = object()
    SPACE = object()


class _FakeCarbInput:
    KeyboardEventType = _FakeKeyboardEventType
    KeyboardInput = _FakeKeyboardInput


def test_abc_env_reuses_bimanual_assembly_robot_runtime() -> None:
    assert issubclass(HandsOnABCIsaacLabEnv, BimanualAssemblyIsaacLabEnv)
    for method_name in SHARED_ROBOT_METHODS:
        assert getattr(HandsOnABCIsaacLabEnv, method_name) is getattr(
            BimanualAssemblyIsaacLabEnv,
            method_name,
        )


def test_abc_deferred_arm_runtime_is_explicit_and_does_not_change_actions(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_ARM_RENDER", "1")
    env = HandsOnABCIsaacLabEnv(
        "t08_hammer_nail",
        launch_isaac=False,
        image_obs=False,
    )
    assert env._deferred_arm_render is True
    assert env._control_dt == pytest.approx(0.02)
    assert env._n_substeps == 2

    right_pose = np.asarray(
        [-0.20, -0.12, 1.05, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    left_pose = np.asarray(
        [-0.18, 0.15, 1.08, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    right_hand = np.linspace(0.0, 0.19, env._hand_action_dim)
    left_hand = np.linspace(0.2, 0.39, env._hand_action_dim)
    env._apply_action(
        {
            "right": np.concatenate([right_pose, right_hand]),
            "left": np.concatenate([left_pose, left_hand]),
        }
    )

    np.testing.assert_allclose(env._right_tcp_pose, right_pose)
    np.testing.assert_allclose(env._left_tcp_pose, left_pose)
    np.testing.assert_allclose(env._right_hand, right_hand)
    np.testing.assert_allclose(env._left_hand, left_hand)


@pytest.mark.parametrize(
    "task_name",
    [
        "t21_pass_block",
        "t10_lift_carton",
    ],
)
def test_d08_aligned_physx_tasks_can_defer_all_recording_cameras(
    monkeypatch,
    task_name: str,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "1")

    env = HandsOnABCIsaacLabEnv(
        task_name,
        launch_isaac=False,
        image_obs=False,
        render_mode="human",
        physics_backend="physx",
    )

    assert env._deferred_camera_render is True

    class ViewportOnlySimulation:
        render_calls = 0

        def render(self) -> None:
            self.render_calls += 1

    simulation = ViewportOnlySimulation()
    env._sim = simulation
    frames = env.render()

    assert simulation.render_calls == 1
    assert len(frames) == 3
    assert env._cameras == {}


def test_deferred_arm_runtime_preserves_explicit_control_period(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_ARM_RENDER", "1")
    env = HandsOnABCIsaacLabEnv(
        "t08_hammer_nail",
        launch_isaac=False,
        image_obs=False,
        control_dt=0.02,
        hz=30,
    )

    assert env._control_dt == pytest.approx(0.02)
    assert env._n_substeps == 2


def test_ka005_2_cannot_reenable_unqualified_deferred_arm_render(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_ARM_RENDER", "1")
    env = HandsOnABCIsaacLabEnv(
        "t38_set_dual_knobs",
        launch_isaac=False,
        image_obs=False,
    )
    assert env._deferred_arm_render is False
    assert "t38_set_dual_knobs" in (
        abc_env._PHYSX_TENSOR_ONLY_CONTACT_REPORT_TASKS
    )


def test_ka005_2_ground_truth_uses_post_reset_tensor_contact_capture(
    monkeypatch,
) -> None:
    class _Prim:
        def __init__(self, path: str) -> None:
            self._path = path

        def GetPath(self) -> str:
            return self._path

    env = HandsOnABCIsaacLabEnv(
        "t38_set_dual_knobs",
        launch_isaac=False,
        image_obs=False,
    )
    env._stage = object()
    monkeypatch.setattr(
        env,
        "_task_state_candidate_names",
        lambda _path: {"knob_right", "panel_base", "knob_left"},
    )
    paths = {
        "panel_base": "/World/Task/B320Benchmark/panel_base",
        "knob_left": "/World/Task/B320Benchmark/panel_base/knob_left",
        "knob_right": "/World/Task/B320Benchmark/panel_base/knob_right",
    }
    monkeypatch.setattr(
        env,
        "_find_rigid_body_prim",
        lambda name: _Prim(paths[name]),
    )

    def _forbidden_reporter_apply(*_args, **_kwargs):
        raise AssertionError("KA005-2 must not apply pre-reset reporter APIs")

    monkeypatch.setattr(
        abc_env,
        "configure_physx_contact_reports_for_recording",
        _forbidden_reporter_apply,
    )

    env._configure_ground_truth_contact_reports(Path("unused.xml"))

    assert env._physx_contact_sensor_paths_preconfigured == tuple(
        paths[name] for name in sorted(paths)
    )
    assert env._physx_contact_report_paths_preconfigured == ()


@pytest.mark.parametrize(
    ("task_name", "expected_right", "expected_left"),
    [
        ("t08_hammer_nail", _RIGHT_HOME, _LEFT_HOME),
        (
            "t26_place_fruit_in_bucket",
            _A01_CLEAR_RIGHT_START_TCP,
            _RAISED_LEFT_START_TCP,
        ),
        ("t25_place_cup_on_plate", _RIGHT_HOME, _RAISED_LEFT_START_TCP),
        ("t04_cover_pitcher", _RIGHT_HOME, _RAISED_LEFT_START_TCP),
        ("t09_insert_peg", _RIGHT_HOME, _RAISED_LEFT_START_TCP),
        ("t44_stack_blocks", _RIGHT_HOME, _RAISED_LEFT_START_TCP),
        (
            "t03_cover_box",
            _RIGHT_HOME,
            _RAISED_LEFT_START_TCP,
        ),

        *[
            (
                task_name,
                abc_env._D06_LOWERED_RIGHT_START_TCP,
                abc_env._D06_LOWERED_LEFT_START_TCP,
            )
            for task_name in sorted(
                abc_env._D06_ALIGNED_C_TOOL_START_TASKS & set(abc_env._TASK_SPECS) - {"eval_flex03_pack_pillow"}
            )
        ],
        (
            "t11_line_up_blocks",
            abc_env._C12_LOWERED_RIGHT_START_TCP,
            abc_env._C12_LOWERED_LEFT_START_TCP,
        ),
        *[
            (task_name, _RAISED_RIGHT_START_TCP, _RAISED_LEFT_START_TCP)
            for task_name in sorted(
                abc_env._C_TOOL_PHYSICS_ONLY_COLLECTION_TASKS
                - abc_env._D06_ALIGNED_C_TOOL_START_TASKS
                - {"t11_line_up_blocks", "eval_flex03_pack_pillow"}
                & set(abc_env._TASK_SPECS)
            )
        ],
    ],
)
def test_deferred_arm_collection_starts_at_live_robot_home(
    monkeypatch,
    task_name: str,
    expected_right: np.ndarray,
    expected_left: np.ndarray,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_ARM_RENDER", "1")
    env = HandsOnABCIsaacLabEnv(
        task_name,
        launch_isaac=False,
        image_obs=False,
    )

    env._sample_reset_state(clear_manual_request=True)

    np.testing.assert_allclose(env._right_tcp_pose, expected_right)
    np.testing.assert_allclose(env._left_tcp_pose, expected_left)


@pytest.mark.parametrize(
    (
        "task_name",
        "reference_right_tcp",
        "reference_left_tcp",
        "reference_home",
        "lowered_home",
        "lowering_meters",
    ),
    (
        *(
            (
                task_name,
                abc_env._D06_LOWERED_RIGHT_START_TCP,
                abc_env._D06_LOWERED_LEFT_START_TCP,
                abc_env._D06_LOWERED_ARM_HOME,
                abc_env._D06_LOWERED_ARM_HOME,
                0.0,
            )
            for task_name in sorted(
                abc_env._D06_ALIGNED_C_TOOL_START_TASKS & set(abc_env._TASK_SPECS) - {"eval_flex03_pack_pillow"}
            )
        ),
        (
            "t11_line_up_blocks",
            abc_env._D06_LOWERED_RIGHT_START_TCP,
            abc_env._D06_LOWERED_LEFT_START_TCP,
            abc_env._D06_LOWERED_ARM_HOME,
            abc_env._C12_LOWERED_ARM_HOME,
            abc_env._C12_LOWERED_FROM_D06_METERS,
        ),
    ),
)
@pytest.mark.usefixtures("rokae_assets")
def test_lowered_c_tool_starts_match_reference_home(
    task_name: str,
    reference_right_tcp: np.ndarray,
    reference_left_tcp: np.ndarray,
    reference_home: dict[str, np.ndarray],
    lowered_home: dict[str, np.ndarray],
    lowering_meters: float,
) -> None:
    env = HandsOnABCIsaacLabEnv(
        task_name,
        launch_isaac=False,
        image_obs=False,
    )
    env._sample_reset_state(clear_manual_request=True)

    np.testing.assert_allclose(
        env._right_tcp_pose[:2],
        reference_right_tcp[:2],
    )
    np.testing.assert_allclose(
        env._left_tcp_pose[:2],
        reference_left_tcp[:2],
    )
    assert env._right_tcp_pose[2] == pytest.approx(
        reference_right_tcp[2] - lowering_meters
    )
    assert env._left_tcp_pose[2] == pytest.approx(
        reference_left_tcp[2] - lowering_meters
    )

    for side in ("right", "left"):
        cfg = types.SimpleNamespace(
            init_state=types.SimpleNamespace(
                joint_pos={
                    f"joint{index}_{side}(_1)?": -99.0
                    for index in range(1, 8)
                }
            )
        )
        assert env._apply_raised_hand_home(cfg, side) is True
        actual = [
            cfg.init_state.joint_pos[f"joint{index}_{side}(_1)?"]
            for index in range(1, 8)
        ]
        np.testing.assert_allclose(
            actual,
            lowered_home[side],
        )

    xml_path = (
        Path(__file__).resolve().parents[1]
        / "handson"
        / "sim"
        / "envs"
        / "xmls"
        / "rokae_wuji1_bimanual.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    def attachment_pose(
        home: dict[str, np.ndarray], side: str
    ) -> tuple[np.ndarray, np.ndarray]:
        data = mujoco.MjData(model)
        data.qpos[:] = 0.10
        for index, value in enumerate(home[side], start=1):
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"joint{index}_{side}",
            )
            data.qpos[model.jnt_qposadr[joint_id]] = float(value)
        mujoco.mj_forward(model, data)
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"attachment_site_{side}",
        )
        return (
            data.site_xpos[site_id].copy(),
            data.site_xmat[site_id].copy(),
        )

    for side in ("right", "left"):
        reference_position, reference_rotation = attachment_pose(
            reference_home, side
        )
        lowered_position, lowered_rotation = attachment_pose(lowered_home, side)
        np.testing.assert_allclose(
            lowered_position[:2], reference_position[:2], atol=2e-9
        )
        assert lowered_position[2] == pytest.approx(
            reference_position[2] - lowering_meters,
            abs=2e-9,
        )
        np.testing.assert_allclose(
            lowered_rotation, reference_rotation, atol=2e-8
        )

    data = mujoco.MjData(model)
    data.qpos[:] = 0.10
    for side in ("right", "left"):
        for index, value in enumerate(lowered_home[side], start=1):
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"joint{index}_{side}",
            )
            data.qpos[model.jnt_qposadr[joint_id]] = float(value)
    mujoco.mj_forward(model, data)
    lowest_hand_origin = min(
        float(data.xpos[body_id, 2])
        for body_id in range(model.nbody)
        if any(
            marker
            in (
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_id,
                )
                or ""
            ).lower()
            for marker in ("palm", "finger")
        )
    )
    assert lowest_hand_origin > abc_env._TABLE_TOP_Z + 0.20


def test_deferred_robot_body_filter_keeps_only_hand_visual_paths() -> None:
    assert _deferred_keeps_hand_visual_path(
        "/World/HandsOnRobots/RightRobot/base_right/right_palm_link/visuals"
    )
    assert _deferred_keeps_hand_visual_path(
        "/World/HandsOnRobots/LeftRobot/base_left/left_finger5_link4/visuals"
    )
    assert not _deferred_keeps_hand_visual_path(
        "/World/HandsOnRobots/RightRobot/base_right/link7_right/visuals"
    )
    assert not _deferred_keeps_hand_visual_path(
        "/World/HandsOnRobots/RightRobot/base_right/right_wrist_fisheye_mount/visuals"
    )
    assert not _deferred_keeps_hand_visual_path(
        "/World/HandsOnRobots/CenterRig/rokae_center_head_visual"
    )


def test_b03_newton_removes_decorative_center_rig_from_rtx_cameras() -> None:
    class FakeCenterRig:
        def __init__(self) -> None:
            self.active = True

        def IsValid(self) -> bool:
            return True

        def SetActive(self, active: bool) -> None:
            self.active = bool(active)

    center_rig = FakeCenterRig()

    class FakeStage:
        def GetPrimAtPath(self, path: str) -> FakeCenterRig:
            assert path == "/World/HandsOnRobots/CenterRig"
            return center_rig

    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        launch_isaac=False,
        image_obs=False,
        physics_backend="newton",
    )
    env._stage = FakeStage()

    assert env._hide_b03_newton_center_rig_visuals() == 1
    assert center_rig.active is False


def test_b03_operator_view_is_from_robot_toward_table() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        launch_isaac=False,
        image_obs=False,
    )

    np.testing.assert_allclose(env._spec.head_eye, [-1.35, 0.0, 1.58])
    np.testing.assert_allclose(env._spec.head_target, [-0.12, 0.0, 1.02])
    assert env._spec.head_target[0] > env._spec.head_eye[0]


def test_isaaclab_tasks_default_to_native_differential_ik(monkeypatch) -> None:
    monkeypatch.delenv("HANDSON_ISAACLAB_ARM_IK_BACKEND", raising=False)
    monkeypatch.setenv("HANDSON_ROKAE_IK_BACKEND", "haorui_new")

    bimanual = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
    )
    abc = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        launch_isaac=False,
        image_obs=False,
    )

    assert bimanual._arm_ik_backend == "dls"
    assert abc._arm_ik_backend == "dls"


def test_isaaclab_arm_ik_requires_explicit_isaaclab_override(monkeypatch) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_ARM_IK_BACKEND", "haorui_new")

    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
    )

    assert env._arm_ik_backend == "haorui"


def test_position_multistart_targets_use_existing_joint_step_limit() -> None:
    torch = pytest.importorskip("torch")

    class FakeRobot:
        def __init__(self) -> None:
            self.data = types.SimpleNamespace(
                default_joint_pos=torch.zeros((1, 7), dtype=torch.float64),
                joint_pos=torch.zeros((1, 7), dtype=torch.float64),
                soft_joint_pos_limits=None,
            )
            self.target = None

        def set_joint_position_target(self, target) -> None:
            self.target = target.clone()

    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
    )
    robots = {"right": FakeRobot(), "left": FakeRobot()}
    # Mirror the native ABC/D control topology: these environments own their
    # imported articulations in ``_robots`` instead of an InteractiveScene.
    # Without that marker the interface-only guard intentionally skips live
    # target writes before this unit test reaches the limiter under test.
    env._robots = robots
    env._torch = torch
    env._physics_backend = "physx"
    env._arm_ik_backend = "dls"
    env._control_robot = lambda side: robots[side]
    env._joint_targets = {
        side: torch.zeros((1, 7), dtype=torch.float64) for side in robots
    }
    env._arm_joint_ids = {side: list(range(7)) for side in robots}
    env._hand_joint_ids = {side: [] for side in robots}
    env._right_hand = np.zeros(0, dtype=np.float64)
    env._left_hand = np.zeros(0, dtype=np.float64)
    env._right_tcp_pose = np.zeros(7, dtype=np.float64)
    env._left_tcp_pose = np.zeros(7, dtype=np.float64)
    env._right_pose_commanded = True
    env._left_pose_commanded = True
    env._compute_arm_joint_target = lambda *_args: (
        torch.ones((1, 7), dtype=torch.float64),
        "test",
    )
    env._use_position_only_multistart_ik = lambda side: side == "right"
    env._last_control_log_time = float("inf")

    env._apply_joint_targets()

    np.testing.assert_allclose(robots["right"].target.numpy(), 0.03)
    np.testing.assert_allclose(robots["left"].target.numpy(), 1.0)


def test_haorui_tool_transform_uses_measured_arm_configuration() -> None:
    torch = pytest.importorskip("torch")
    calls = {}

    class FakeModel:
        def fk(self, q):
            calls["fk_q"] = np.asarray(q).copy()
            return np.eye(4, dtype=np.float64)

    class FakeSolver:
        def solve_multi_start(
            self,
            target,
            seeds,
            *,
            first_success,
            reference_q,
        ):
            calls["target"] = np.asarray(target).copy()
            calls["seeds"] = [np.asarray(seed).copy() for seed in seeds]
            calls["first_success"] = first_success
            calls["reference_q"] = np.asarray(reference_q).copy()
            return types.SimpleNamespace(success=True, q=np.zeros(7))

    class FakeRokae:
        def make_seeds(self, side, q):
            calls["seed_side"] = side
            calls["seed_q"] = np.asarray(q).copy()
            return [np.asarray(q).copy()]

    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
    )
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_pos=torch.ones((1, 7), dtype=torch.float64)
        )
    )
    env._torch = torch
    env._arm_joint_ids = {"right": list(range(7))}
    env._joint_targets = {
        "right": torch.full((1, 7), 2.0, dtype=torch.float64)
    }
    env._ee_body_indices = {"right": 0}
    env._robot_base_pose_matrix = lambda *_args: np.eye(4, dtype=np.float64)
    env._robot_body_pose_matrix = lambda *_args: np.eye(4, dtype=np.float64)
    env._haorui_ik_models = {"right": FakeModel()}
    env._haorui_ik_solvers = {"right": FakeSolver()}
    env._haorui_position_ik_solvers = {"right": FakeSolver()}
    env._haorui_rokae = FakeRokae()

    result = env._compute_arm_joint_target_haorui(
        "right",
        robot,
        np.asarray([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]),
        position_only=True,
    )

    assert result is not None
    np.testing.assert_allclose(calls["fk_q"], 1.0)
    np.testing.assert_allclose(calls["seed_q"], 2.0)
    np.testing.assert_allclose(calls["reference_q"], 2.0)
    assert calls["seed_side"] == "right"
    assert calls["first_success"] is True


def test_space_key_is_consumed_and_resumes_a_paused_timeline() -> None:
    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
    )
    env._timeline = _FakeTimeline()
    event = types.SimpleNamespace(
        type=_FakeKeyboardEventType.KEY_PRESS,
        input=_FakeKeyboardInput.SPACE,
    )

    propagate = env._on_keyboard_event(_FakeCarbInput, event)

    assert propagate is False
    assert env._timeline.play_calls == 1
    assert env._timeline.is_playing()
    assert env._manual_reset_requested is False


def test_r_key_is_consumed_and_queues_one_recorder_reset() -> None:
    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
    )
    immediate_reset_calls: list[bool] = []
    env._apply_keyboard_reset_immediately = lambda: immediate_reset_calls.append(True)
    frozen_camera = {
        "camera_path": "/OmniverseKit_Persp",
        "eye": (-1.0, 0.2, 1.5),
        "target": (0.0, 0.0, 1.0),
    }
    env._capture_viewport_camera_state = lambda: frozen_camera
    event = types.SimpleNamespace(
        type=_FakeKeyboardEventType.KEY_PRESS,
        input=_FakeKeyboardInput.R,
    )

    propagate = env._on_keyboard_event(_FakeCarbInput, event)

    assert propagate is False
    assert env._manual_reset_requested is True
    assert env._manual_reset_viewport_camera_state is frozen_camera
    assert immediate_reset_calls == []


def test_v_key_is_consumed_and_restores_the_task_view() -> None:
    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
    )
    camera_reset_calls: list[bool] = []
    env._reset_viewport_camera_to_default = (
        lambda **_kwargs: camera_reset_calls.append(True) or True
    )
    event = types.SimpleNamespace(
        type=_FakeKeyboardEventType.KEY_PRESS,
        input=_FakeKeyboardInput.V,
    )

    propagate = env._on_keyboard_event(_FakeCarbInput, event)

    assert propagate is False
    assert camera_reset_calls == [True]


def test_z_key_queues_exactly_one_no_jump_tracker_rebase() -> None:
    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
    )
    event = types.SimpleNamespace(
        type=_FakeKeyboardEventType.KEY_PRESS,
        input=_FakeKeyboardInput.Z,
    )

    propagate = env._on_keyboard_event(_FakeCarbInput, event)

    assert propagate is False
    assert env.consume_manual_tracker_rebase_request() is True
    assert env.consume_manual_tracker_rebase_request() is False


def test_timeline_pause_callback_resumes_but_does_not_restart_a_stopped_timeline() -> None:
    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
    )
    paused = _FakeTimeline()
    env._timeline = paused
    env._on_timeline_pause()
    assert paused.play_calls == 1

    stopped = _FakeTimeline(stopped=True)
    env._timeline = stopped
    env._on_timeline_pause()
    assert stopped.play_calls == 0


def test_step_resumes_timeline_before_entering_blocking_simulation_step() -> None:
    events: list[str] = []

    class FakeScene:
        def write_data_to_sim(self) -> None:
            events.append("scene.write")

        def update(self, dt: float) -> None:
            events.append(f"scene.update:{dt}")

    class FakeSimulation:
        def step(self, *, render=True) -> None:
            assert env._timeline.is_playing()
            events.append(f"sim.step:{render}")

    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
    )
    env._timeline = _FakeTimeline()
    env._scene = FakeScene()
    env._sim = FakeSimulation()
    env._sim_dt = 0.01

    env._step_isaaclab()

    assert env._timeline.play_calls == 1
    assert events == ["scene.write", "sim.step:False", "scene.update:0.01"]


def test_isaaclab_control_rate_can_be_aligned_to_sixty_hz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_CONTROL_HZ", "60")
    monkeypatch.setenv("HANDSON_ISAACLAB_CONTROL_DT", str(1.0 / 60.0))

    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        hz=30,
        control_dt=0.02,
        physics_dt=1.0 / 120.0,
    )

    assert env.hz == 60
    assert env._control_dt == pytest.approx(1.0 / 60.0)
    assert env._n_substeps == 2


def test_control_period_accounts_for_render_work_before_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        hz=10,
        control_dt=0.02,
        physics_dt=0.01,
    )
    clock = [0.0]
    sleeps: list[float] = []

    monkeypatch.setattr(assembly_env.time, "monotonic", lambda: clock[0])

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(assembly_env.time, "sleep", fake_sleep)
    env._control_perf_window_start = 0.0
    env._step_isaaclab = lambda: None
    env._apply_action = lambda _action: None
    env._apply_joint_targets = lambda: None
    env._refresh_object_poses = lambda: None
    env._compute_success = lambda: False
    env._compute_failure = lambda: False

    def render_observation() -> dict[str, bool]:
        clock[0] += 0.06
        return {"rendered": True}

    env._compute_observation = render_observation

    observation, _, _, _, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float64)
    )

    assert observation == {"rendered": True}
    assert sleeps == pytest.approx([0.04])
    assert info["control_timing"]["render_ms"] == pytest.approx(60.0)
    assert info["control_timing"]["actual_hz"] == pytest.approx(10.0)


def test_imported_abc_robots_are_a_live_control_scene_without_interactive_scene() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
    )
    env._launch_isaac = True
    env._scene = None
    env._robots = {"right": object(), "left": object()}

    class ReachedImportedRobot(RuntimeError):
        pass

    def imported_robot(_side: str):
        raise ReachedImportedRobot

    env._control_robot = imported_robot

    # Passing the base-class live-scene guard is the regression contract.  A
    # real runtime continues here with the imported Articulation wrappers.
    with pytest.raises(ReachedImportedRobot):
        env._apply_joint_targets()


def test_manual_keyboard_reset_waits_for_the_control_step_boundary() -> None:
    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        control_dt=0.05,
        physics_dt=0.01,
        hz=30,
    )
    calls: list[str] = []

    env._apply_action = lambda _action: calls.append("action")
    env._apply_joint_targets = lambda: calls.append("targets")

    physics_calls = 0

    def request_reset_during_first_substep() -> None:
        nonlocal physics_calls
        calls.append("physics")
        physics_calls += 1
        if physics_calls == 1:
            env._manual_reset_requested = True

    env._step_isaaclab = request_reset_during_first_substep
    env._refresh_object_poses = lambda: calls.append("refresh")
    env._compute_observation = lambda: {"reset": True}
    env._compute_success = lambda: False
    env._compute_failure = lambda: False
    env._task_info = lambda: {}

    observation, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float64)
    )

    assert calls == [
        "action",
        "targets",
        "physics",
        "physics",
        "physics",
        "physics",
        "physics",
        "refresh",
    ]
    assert observation == {"reset": True}
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["manual_reset"] is True
    assert "manual_reset_applied" not in info
    assert env.env_step == 1


def test_repeated_keyboard_resets_leave_the_next_step_controllable() -> None:
    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        hz=0,
    )
    env._refresh_object_poses = lambda: None
    env._compute_observation = lambda: {"reset": True}
    env._sample_reset_state = lambda **_kwargs: None
    env._reset_scene_state = lambda: None
    env._apply_reset_visual_randomization = lambda: None
    env._step_isaaclab = lambda: None

    for _ in range(6):
        env._manual_reset_requested = True
        env._manual_reset_already_applied = True
        env._manual_reset_viewport_camera_state = {"eye": (1.0, 2.0, 3.0)}
        _, _, _, _, step_info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float64)
        )
        assert step_info["manual_reset"] is True

        _, reset_info = env.reset()
        assert reset_info["manual_reset_applied"] is True
        assert env._manual_reset_requested is False
        assert env._manual_reset_already_applied is False
        assert env._manual_reset_viewport_camera_state is None


def test_abc_env_keeps_task_specific_action_shapes() -> None:
    single = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
    )
    dual = HandsOnABCIsaacLabEnv(
        "t08_hammer_nail",
        launch_isaac=False,
        image_obs=False,
    )

    assert single.action_space.shape == (27,)
    assert dual.action_space.shape == (54,)
    assert single.observation_space["state"]["tcp_pose"].shape == (7,)
    assert dual.observation_space["state"]["tcp_pose"].shape == (14,)


def test_c04_spray_visual_uses_trigger_hysteresis() -> None:
    decide = HandsOnABCIsaacLabEnv._c04_spray_should_be_visible

    assert not decide(0.10, False)
    assert not decide(0.29, False)
    assert decide(0.30, False)
    assert decide(0.31, False)
    assert decide(0.26, True)
    assert not decide(0.24, True)


def _c04_aimed_poses() -> dict[str, np.ndarray]:
    plant = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    # A 180-degree X rotation points the Water Plant cone's local +Z axis down.
    # Place the root so its ref_point nozzle is 30 cm above the pot center.
    spray = np.asarray([0.0, 0.076, 0.372, 0.0, 1.0, 0.0, 0.0])
    return {"spray": spray, "plant": plant}


def test_c04_isaac_uses_water_plant_ray_hit_geometry() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t50_water_plant",
        launch_isaac=False,
        image_obs=False,
    )
    env._poses = _c04_aimed_poses()

    aimed = env._c04_spray_aim_metrics()

    assert aimed["aimed"]
    assert aimed["spray_downward"]
    assert aimed["water_hit_count"] >= 5
    assert aimed["water_sample_count"] == 9

    env._poses["spray"][3:7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    missed = env._c04_spray_aim_metrics()
    assert not missed["aimed"]


def test_c04_isaac_hit_volume_matches_visible_water_cone() -> None:
    assert np.isclose(_C04_WATER_CONE_SPREAD, 0.025 / 0.10)
    assert np.isclose(_C04_WATER_HIT_FRACTION_THRESHOLD, 1.0 / 9.0)
    assert _C04_SUCCESS_STEPS == 12


def test_c04_isaac_requires_contacted_grasp_before_spraying() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t50_water_plant",
        launch_isaac=False,
        image_obs=False,
    )
    trigger_position = _C04_TRIGGER_PULL_RAD + 0.01
    env._physics_dof_position = lambda _name: (trigger_position, 0.0)
    env._c04_trigger_contact_force = lambda: 0.0
    env._c04_grasp_contact_force = lambda: 0.0

    env._update_c04_trigger_state()
    assert not env._c04_water_emitting

    env._c04_trigger_contact_force = lambda: 0.20
    env._c04_grasp_contact_force = lambda: 0.40
    env._update_c04_trigger_state()
    assert not env._c04_trigger_cycle_latched
    assert not env._c04_water_emitting

    trigger_position = _C04_TRIGGER_RELEASE_RAD - 0.01
    for _ in range(_C04_TRIGGER_ARM_STEPS):
        env._update_c04_trigger_state()
    assert env._c04_trigger_armed

    trigger_position = _C04_TRIGGER_PULL_RAD + 0.01
    env._update_c04_trigger_state()
    assert env._c04_trigger_cycle_latched
    assert env._c04_water_emitting

    trigger_position = _C04_TRIGGER_RELEASE_RAD - 0.01
    env._update_c04_trigger_state()
    assert not env._c04_trigger_cycle_latched
    assert not env._c04_water_emitting


def test_c04_isaac_prepressed_trigger_requires_release_before_arming() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t50_water_plant",
        launch_isaac=False,
        image_obs=False,
    )
    trigger_position = _C04_TRIGGER_PULL_RAD + 0.05
    env._physics_dof_position = lambda _name: (trigger_position, 0.0)
    env._c04_trigger_contact_force = lambda: 0.20
    env._c04_grasp_contact_force = lambda: 0.40

    for _ in range(_C04_TRIGGER_ARM_STEPS * 2):
        env._update_c04_trigger_state()

    assert not env._c04_trigger_armed
    assert not env._c04_water_emitting
    assert env._c04_trigger_phase == "unarmed"

    trigger_position = _C04_TRIGGER_RELEASE_RAD - 0.01
    for _ in range(_C04_TRIGGER_ARM_STEPS):
        env._update_c04_trigger_state()
    assert env._c04_trigger_armed

    trigger_position = _C04_TRIGGER_PULL_RAD + 0.05
    env._update_c04_trigger_state()
    assert env._c04_water_emitting


def test_c04_isaac_water_plant_success_and_off_target_failure() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t50_water_plant",
        launch_isaac=False,
        image_obs=False,
    )
    env._c04_trigger_pulled = True
    env._c04_trigger_cycle_latched = True
    env._c04_water_emitting = True
    env._c04_trigger_contact_force = lambda: 0.20
    env._c04_grasp_contact_force = lambda: 0.40
    env._physics_dof_position = lambda _name: (0.35, 0.0)
    env._c04_spray_aim_metrics = lambda: {
        "aimed": True,
        "spray_hits_soil_or_plant": True,
    }

    for _ in range(_C04_SUCCESS_STEPS - 1):
        assert not env._compute_success()
    assert env._compute_success()
    assert not env._compute_failure()

    env._c04_spray_aim_metrics = lambda: {
        "aimed": False,
        "spray_hits_soil_or_plant": False,
    }
    assert not env._compute_success()
    assert env._compute_failure()
    assert env._last_c04_success_metrics["c04_failed_trigger_outside"]


def test_c04_isaac_off_target_spray_terminates_with_failed_info() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t50_water_plant",
        launch_isaac=False,
        image_obs=False,
        hz=0,
    )
    env._n_substeps = 1
    env._c04_trigger_pulled = True
    env._c04_trigger_cycle_latched = True
    env._c04_water_emitting = True
    env._c04_spray_aim_metrics = lambda: {
        "aimed": False,
        "spray_hits_soil_or_plant": False,
    }
    env._c04_trigger_contact_force = lambda: 0.20
    env._c04_grasp_contact_force = lambda: 0.40
    env._physics_dof_position = lambda _name: (0.35, 0.0)
    env._apply_action = lambda _action: None
    env._apply_joint_targets = lambda: None
    env._step_isaaclab = lambda: None
    env._refresh_object_poses = lambda: None
    env._compute_observation = lambda: {}

    _, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    assert reward == 0.0
    assert terminated
    assert not truncated
    assert not info["succeed"]
    assert info["failed"]


def test_c04_isaac_condition_panel_exposes_water_plant_checks() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t50_water_plant",
        launch_isaac=False,
        image_obs=False,
    )

    keys = [row["key"] for row in env._success_condition_rows()]

    assert keys == [
        "c04_spray_grasped",
        "c04_trigger_contact",
        "c04_water_emitting",
        "c04_water_hits_target",
        "success_hold",
    ]


def test_c04_isaac_reports_water_plant_trigger_parameters() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t50_water_plant",
        launch_isaac=False,
        image_obs=False,
    )

    info = env._task_info()

    assert info["c04_success_steps_required"] == _C04_SUCCESS_STEPS
    assert info["c04_trigger_pull_threshold"] == 0.30
    assert info["c04_trigger_release_threshold"] == 0.25
    assert info["c04_trigger_stiffness"] == _C04_TRIGGER_STIFFNESS == 0.20
    assert info["c04_trigger_damping"] == _C04_TRIGGER_DAMPING == 0.10
    assert info["c04_trigger_friction"] == _C04_TRIGGER_FRICTION == 0.015
    assert info["c04_trigger_max_force"] == _C04_TRIGGER_MAX_FORCE == 0.08
    assert info["c04_trigger_arm_steps"] == _C04_TRIGGER_ARM_STEPS == 18
    assert not info["c04_trigger_armed"]


def test_b01_lid_physics_releases_without_weakening_b05_grasp() -> None:
    b01 = _LID_GRASP_PHYSICS["t04_cover_pitcher"]
    b05 = _LID_GRASP_PHYSICS["t03_cover_box"]

    assert b01.mass == 0.060
    assert b01.static_friction == 1.2
    assert b01.dynamic_friction < b01.static_friction
    assert b01.friction_combine_mode == "min"
    assert b01.contact_offset == 0.001
    assert b01.rest_offset == 0.0

    assert b05.mass == 0.040
    assert b05.friction_combine_mode == "max"
    assert b05.static_friction == 2.5


def test_b01_pitcher_base_is_weighted_and_damped_for_one_hand_seating() -> None:
    cfg = _B01_PITCHER_BASE_PHYSICS

    assert cfg.mass == 0.350
    assert cfg.static_friction > cfg.dynamic_friction
    assert cfg.linear_damping == 0.2
    assert cfg.angular_damping == 1.0
    assert cfg.contact_offset == 0.001
    assert cfg.rest_offset == 0.0


def test_a02_success_uses_the_real_plate_pose_and_stability_window() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t25_place_cup_on_plate",
        launch_isaac=False,
        image_obs=False,
    )
    plate = np.asarray(
        [
            -0.10,
            -0.36,
            env._current_table_top_z() + 0.012,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    mug = plate.copy()
    mug[2] += 0.055
    env._poses = {"025_mug": mug, "tray": plate}
    env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)
    env._a06_hand_points = lambda _side: (
        np.asarray([0.70, 0.70, 1.20], dtype=np.float64),
    )

    assert env._target_on_tray_center()
    env._success_counter = env._success_steps_required() - 1
    assert env._compute_success()

    env._physics_body_velocity = lambda _name: None
    assert not env._target_on_tray_center()
    assert not env._last_a02_success_metrics["a02_target_stable"]

    env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)
    env._a06_hand_points = lambda _side: (mug[:3].copy(),)
    assert not env._target_on_tray_center()
    assert not env._last_a02_success_metrics["a02_hand_released"]

    env._a06_hand_points = lambda _side: (
        np.asarray([0.70, 0.70, 1.20], dtype=np.float64),
    )
    env._poses["tray"] = plate.copy()
    env._poses["tray"][2] += 0.08
    assert not env._target_on_tray_center()
    assert not env._last_a02_success_metrics["a02_tray_on_table"]


def test_a03_success_requires_stable_supported_bowl_and_released_hand() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t24_place_ball_in_bowl",
        launch_isaac=False,
        image_obs=False,
    )
    bowl = np.asarray(
        [-0.18, 0.20, env._current_table_top_z(), 1.0, 0.0, 0.0, 0.0]
    )
    ball = bowl.copy()
    ball[2] += 0.050
    env._poses = {env._target_object_name: ball, "024_bowl": bowl}
    env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)
    env._a06_hand_points = lambda _side: (
        np.asarray([0.70, 0.70, 1.20], dtype=np.float64),
    )

    assert env._target_inside_bowl_precise()
    assert env._success_steps_required() >= int(np.ceil(0.5 / env._control_dt))
    env._success_counter = env._success_steps_required() - 1
    assert env._compute_success()

    env._physics_body_velocity = lambda name: (
        None if name == "024_bowl" else np.zeros(6, dtype=np.float64)
    )
    assert not env._target_inside_bowl_precise()
    assert not env._last_a03_success_metrics["a03_bowl_stable"]

    env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)
    env._a06_hand_points = lambda _side: (ball[:3].copy(),)
    assert not env._target_inside_bowl_precise()
    assert not env._last_a03_success_metrics["a03_hand_released"]

    env._a06_hand_points = lambda _side: (
        np.asarray([0.70, 0.70, 1.20], dtype=np.float64),
    )
    env._poses["024_bowl"] = bowl.copy()
    env._poses["024_bowl"][3:7] = np.asarray(
        [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
    )
    assert not env._target_inside_bowl_precise()
    assert not env._last_a03_success_metrics["a03_bowl_upright"]


def test_a01_bucket_geometry_uses_bucket_local_y_as_vertical_axis() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
        target_object="apple",
    )
    bucket = np.asarray([-0.06, 0.02, 1.03, 0.0, 0.0, -0.7071, -0.7071])
    target_local = np.asarray([0.0, -0.08 + 0.038, 0.0, 1.0])
    target_world = _pose_vector_to_matrix(bucket) @ target_local
    apple = np.asarray([*target_world[:3], 1.0, 0.0, 0.0, 0.0])
    env._poses = {"apple": apple, "bucket": bucket}

    assert env._target_on_bucket_bottom()


def test_a01_newton_contact_pairs_only_accept_right_hand_against_bucket() -> None:
    hand_label = (
        "/World/HandsOnRobots/RightRobot/Geometry/base_right/"
        "right_palm_link/right_finger3_link1/right_finger3_link1_col2"
    )
    left_hand_label = (
        "/World/HandsOnRobots/LeftRobot/Geometry/base_left/"
        "left_palm_link/left_finger3_link1/left_finger3_link1_col2"
    )
    bucket_label = "/World/Task/Geometry/bucket/bucket_bottom"
    apple_label = "/World/Task/Geometry/apple/apple_collision"

    assert HandsOnABCIsaacLabEnv._a01_newton_right_hand_shape_label(
        hand_label
    )
    assert not HandsOnABCIsaacLabEnv._a01_newton_right_hand_shape_label(
        left_hand_label
    )
    assert HandsOnABCIsaacLabEnv._a01_newton_bucket_shape_label(
        bucket_label
    )
    assert not HandsOnABCIsaacLabEnv._a01_newton_bucket_shape_label(
        apple_label
    )

    hand_indices = frozenset({3})
    bucket_indices = frozenset({7})
    assert not HandsOnABCIsaacLabEnv._a01_contact_pairs_include_hand_bucket(
        np.asarray([3]),
        np.asarray([9]),
        hand_indices,
        bucket_indices,
    )
    assert HandsOnABCIsaacLabEnv._a01_contact_pairs_include_hand_bucket(
        np.asarray([7]),
        np.asarray([3]),
        hand_indices,
        bucket_indices,
    )


def test_a01_bucket_contact_reporter_is_authored_before_physx_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class FakeAttribute:
        def Set(self, value: float) -> None:
            events.append(("threshold", value))

    class FakeReport:
        def CreateThresholdAttr(self) -> FakeAttribute:
            return FakeAttribute()

    class FakeContactReportAPI:
        @staticmethod
        def Get(_stage, path):
            events.append(("get", path))
            return FakeReport()

        @staticmethod
        def Apply(prim):
            events.append(("apply", prim))
            return FakeReport()

    fake_physx_schema = types.SimpleNamespace(
        PhysxContactReportAPI=FakeContactReportAPI,
    )
    fake_pxr = types.ModuleType("pxr")
    fake_pxr.PhysxSchema = fake_physx_schema
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)

    class FakeBucket:
        def HasAPI(self, _api) -> bool:
            return False

        def GetPath(self) -> str:
            return "/World/Task/bucket/bucket"

    bucket = FakeBucket()
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
    )
    env._stage = object()
    env._find_rigid_body_prim = lambda name: bucket if name == "bucket" else None

    assert env._configure_a01_contact_reporting() == 1
    assert events == [("apply", bucket), ("threshold", 0.0)]


def test_a01_success_requires_stable_fruit_in_upright_bucket_and_release() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
        target_object="apple",
    )
    bucket = np.asarray(
        [-0.25, 0.20, 0.97, 0.0, 0.0, -0.70710678, -0.70710678]
    )
    target_local = np.asarray([0.0, -0.08 + 0.038, 0.0, 1.0])
    target_world = _pose_vector_to_matrix(bucket) @ target_local
    env._poses = {
        "apple": np.asarray([*target_world[:3], 1.0, 0.0, 0.0, 0.0]),
        "bucket": bucket,
    }
    env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)
    env._a06_hand_points = lambda _side: (
        np.asarray([0.25, -0.25, 1.20], dtype=np.float64),
    )
    env._a01_hand_touches_bucket = lambda: True

    required = env._success_steps_required()
    assert required == int(np.ceil(0.5 / env._control_dt))
    for _ in range(required - 1):
        assert not env._compute_success()
    assert env._compute_success()
    assert [row["key"] for row in env._success_condition_rows()] == [
        "a01_target_in_bucket",
        "a01_target_stable",
        "a01_bucket_upright",
        "a01_bucket_on_table",
        "a01_bucket_stable",
        "a01_hand_released",
        "success_hold",
    ]
    assert env._task_info()["a01_success_steps_required"] == required

    env._success_counter = 0
    env._poses["apple"][0] += 0.40
    assert not env._compute_success()


def test_b03_success_uses_peg_tip_geometry_not_body_origin() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        launch_isaac=False,
        image_obs=False,
    )
    env._physics_body_velocity = lambda name: np.zeros(6, dtype=np.float64)

    socket = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    # The narrow local -Z tip is inside the socket while the body origin is
    # still above the socket body origin.  A body-origin depth check would
    # incorrectly return a negative insertion depth here.
    peg = np.asarray([0.0, 0.0, 0.0035, 1.0, 0.0, 0.0, 0.0])
    env._poses = {"peg": peg, "socket": socket}

    geometry = env._b03_peg_socket_geometry()

    assert -float(peg[2] - socket[2]) < 0.0
    assert geometry["radial_error"] <= 1e-9
    assert geometry["insertion_depth"] >= 0.002
    assert geometry["handle_end_depth"] < 0.0
    assert geometry["tip_end_correct"]
    assert env._peg_socket_success()
    assert env._task_info()["b03_peg_insertion_depth"] >= 0.002


def test_b03_success_rejects_handle_first_insertion() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        launch_isaac=False,
        image_obs=False,
    )
    env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)

    socket = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    # With local +Z pointing down, the shaft endpoint beside the enlarged
    # handle crosses the socket mouth but the actual local -Z tip does not.
    peg = np.asarray([0.0, 0.0, 0.17, 0.0, 1.0, 0.0, 0.0])
    env._poses = {"peg": peg, "socket": socket}

    geometry = env._b03_peg_socket_geometry()

    assert geometry["handle_end_depth"] >= 0.002
    assert geometry["insertion_depth"] < 0.0
    assert geometry["axis_alignment"] == pytest.approx(1.0)
    assert geometry["signed_axis_alignment"] == pytest.approx(-1.0)
    assert not geometry["tip_end_correct"]
    assert not env._peg_socket_success()


def test_b03_success_rejects_high_speed_transient_insertion() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        launch_isaac=False,
        image_obs=False,
    )
    env._physics_body_velocity = lambda _name: np.asarray(
        [0.08, 0.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    env._poses = {
        "peg": np.asarray([0.0, 0.0, 0.0035, 1.0, 0.0, 0.0, 0.0]),
        "socket": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    }

    assert not env._peg_socket_success()
    rows = {
        row["key"]: row["ok"]
        for row in env._b03_success_condition_rows()
    }
    assert not rows["b03_peg_stable"]
    assert not rows["b03_socket_stable"]


def test_b03_success_requires_both_free_bodies_stable_for_half_second() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        launch_isaac=False,
        image_obs=False,
        control_dt=0.02,
    )
    velocities = {
        "peg": np.asarray([0.019, 0.0, 0.0, 0.0, 0.0, 0.29]),
        "socket": np.asarray([0.0, 0.019, 0.0, 0.29, 0.0, 0.0]),
    }
    env._physics_body_velocity = lambda name: velocities.get(name)
    env._poses = {
        "peg": np.asarray([0.0, 0.0, 0.0035, 1.0, 0.0, 0.0, 0.0]),
        "socket": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    }

    required_steps = env._success_steps_required()
    assert required_steps == 25
    assert required_steps * env._control_dt >= _B03_STABLE_HOLD_SECONDS
    assert env._peg_socket_success()
    env._success_counter = required_steps - 1
    assert env._compute_success()

    info = env._task_info()
    assert info["b03_peg_stable"]
    assert info["b03_socket_stable"]
    assert info["b03_stable_linear_speed_limit"] == pytest.approx(
        _B03_STABLE_LINEAR_SPEED
    )
    assert info["b03_stable_angular_speed_limit"] == pytest.approx(
        _B03_STABLE_ANGULAR_SPEED
    )
    assert info["b03_required_stable_seconds"] == pytest.approx(0.5)
    assert [
        row["key"] for row in env._success_condition_rows()
    ] == [
        "b03_radial_aligned",
        "b03_inserted",
        "b03_axis_aligned",
        "b03_tip_end_correct",
        "b03_peg_stable",
        "b03_socket_stable",
        "b03_stable_hold",
    ]

    velocities["socket"] = np.asarray(
        [0.0, 0.0, 0.0, 0.0, 0.0, _B03_STABLE_ANGULAR_SPEED + 0.01]
    )
    assert not env._peg_socket_success()
    env._success_counter = required_steps - 1
    assert not env._compute_success()
    assert env._success_counter == 0


def test_b03_tilted_insertion_is_centered_at_socket_mouth() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        launch_isaac=False,
        image_obs=False,
    )
    env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)

    socket = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    tilt = np.deg2rad(30.0)
    peg_axis = np.asarray([np.sin(tilt), 0.0, np.cos(tilt)])
    inserted_tip = np.asarray([0.0, 0.0, 0.017])
    collision_center = inserted_tip + peg_axis * 0.0675
    body_origin = collision_center - peg_axis * 0.081
    peg = np.asarray(
        [
            *body_origin,
            np.cos(tilt / 2.0),
            0.0,
            np.sin(tilt / 2.0),
            0.0,
        ],
        dtype=np.float64,
    )
    env._poses = {"peg": peg, "socket": socket}

    geometry = env._b03_peg_socket_geometry()

    assert geometry["center_radial_error"] > 0.032
    assert geometry["radial_error"] < 0.010
    assert geometry["insertion_depth"] == pytest.approx(0.010, abs=1e-9)
    assert env._peg_socket_success()


def test_b03_reset_keeps_socket_clear_of_horizontal_peg() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        launch_isaac=False,
        image_obs=False,
    )
    peg_pose = np.asarray(
        [[-0.24, -0.38, 0.94, -0.1228, 0.6964, 0.1228, 0.6964]],
        dtype=np.float64,
    )
    overlapping_socket_pose = np.asarray(
        [[-0.12, -0.26, 0.92, 0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    adjusted = env._b03_clear_socket_reset_pose(
        peg_pose,
        overlapping_socket_pose,
    )

    assert not np.allclose(adjusted[:, :2], overlapping_socket_pose[:, :2])
    peg_rot = _pose_vector_to_matrix(
        np.asarray(
            [
                *peg_pose[0, :3],
                peg_pose[0, 6],
                peg_pose[0, 3],
                peg_pose[0, 4],
                peg_pose[0, 5],
            ],
            dtype=np.float64,
        )
    )[:3, :3]
    peg_axis_xy = peg_rot[:2, 2]
    peg_axis_xy = peg_axis_xy / np.linalg.norm(peg_axis_xy)
    segment_start = peg_pose[0, :2] + peg_axis_xy * _B03_RESET_PEG_MIN_LOCAL_Z
    segment_end = peg_pose[0, :2] + peg_axis_xy * _B03_RESET_PEG_MAX_LOCAL_Z
    segment = segment_end - segment_start
    t = np.clip(
        np.dot(adjusted[0, :2] - segment_start, segment)
        / np.dot(segment, segment),
        0.0,
        1.0,
    )
    closest = segment_start + t * segment

    assert (
        np.linalg.norm(adjusted[0, :2] - closest)
        >= _B03_RESET_SOCKET_SEGMENT_CLEARANCE - 1e-9
    )


def test_b03_nonrandom_reset_preserves_authored_isaac_poses() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    source_peg = np.asarray(
        [[-0.10, -0.16, 0.94, -0.1228, 0.6964, 0.1228, 0.6964]],
        dtype=np.float64,
    )
    source_socket = np.asarray(
        [[-0.10, -0.16, 0.92, 0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    peg = env._b03_reset_pose_for_name("peg", source_peg)
    socket = env._b03_reset_pose_for_name("socket", source_socket)

    np.testing.assert_allclose(peg, source_peg)
    np.testing.assert_allclose(socket, source_socket)
    np.testing.assert_allclose(
        env._b03_sample_clear_peg_reset_pose(source_peg),
        source_peg,
    )
    np.testing.assert_allclose(
        env._b03_sample_clear_socket_reset_pose(source_peg, source_socket),
        source_socket,
    )


def test_b03_randomized_reset_samples_clear_peg_then_socket() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=67,
    )
    base_peg_pose = np.asarray(
        [[-0.24, -0.38, 0.94, -0.1228, 0.6964, 0.1228, 0.6964]],
        dtype=np.float64,
    )
    base_socket_pose = np.asarray(
        [[-0.12, -0.26, 0.92, 0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    peg_poses = []
    socket_poses = []
    for _ in range(96):
        peg = env._b03_sample_clear_peg_reset_pose(base_peg_pose)
        socket = env._b03_sample_clear_socket_reset_pose(peg, base_socket_pose)
        peg_poses.append(peg[0, :2].copy())
        socket_poses.append(socket[0, :2].copy())

        assert env._b03_peg_reset_layout_margin(peg[0]) >= _B03_RESET_MIN_CLEARANCE_SLACK
        assert env._b03_reset_layout_margin(
            peg[0],
            socket[0],
        ) >= _B03_RESET_MIN_CLEARANCE_SLACK
        socket_hand_distances = [
            float(np.linalg.norm(socket[0, :2] - hand_xy))
            for hand_xy in _B03_RESET_SOCKET_HAND_AVOID_XY
        ]
        assert (
            min(socket_hand_distances)
            >= _B03_RESET_SOCKET_HAND_CLEARANCE
            + _B03_RESET_MIN_CLEARANCE_SLACK
            - 1e-9
        )
        assert np.all(peg[0, :2] >= _B03_ROKAE_PEG_XY_BOUNDS[0])
        assert np.all(peg[0, :2] <= _B03_ROKAE_PEG_XY_BOUNDS[1])
        assert np.all(socket[0, :2] >= _B03_ROKAE_SOCKET_XY_BOUNDS[0])
        assert np.all(socket[0, :2] <= _B03_ROKAE_SOCKET_XY_BOUNDS[1])

    peg_spread = np.ptp(np.asarray(peg_poses), axis=0)
    socket_spread = np.ptp(np.asarray(socket_poses), axis=0)
    assert peg_spread[0] > 0.20
    assert peg_spread[1] > 0.34
    assert socket_spread[0] > 0.20
    assert socket_spread[1] > 0.40


def test_robot_and_center_rig_visuals_keep_supported_material_paths() -> None:
    source = (_ISAACLAB_SOURCE_ROOT / "bimanual_assembly_scene.py").read_text(encoding="utf-8")

    assert '"/rightrobot/"' in source
    assert '"/leftrobot/"' in source
    assert '"/centerrig/"' in source
    assert '"/rokae_center"' in source
    assert "_TABLE_OCCLUDER_OPACITY_DEFAULT" not in source
    assert "apply_table_occluder_transparency" not in source
    assert "_transparent_table_occluder_material" not in source
    assert "UsdPreviewSurface" in source


def test_success_condition_overlay_display_labels_are_ascii() -> None:
    rows = [
        {"key": "b03_radial_aligned", "label": "插销中心对准孔位"},
        {"key": "unknown_condition", "label": "未知条件"},
    ]
    for row in rows:
        label = HandsOnABCIsaacLabEnv._success_condition_display_label(row)
        label.encode("ascii")
    assert (
        HandsOnABCIsaacLabEnv._success_condition_display_label(rows[0])
        == "Peg shaft aligned at socket mouth"
    )
    assert HandsOnABCIsaacLabEnv._success_condition_display_label(rows[1]) == "unknown condition"


def test_inactive_left_arm_tasks_use_raised_joint_home() -> None:
    class InitState:
        def __init__(self):
            self.joint_pos = {
                **{f"joint{index}_right(_1)?": 0.0 for index in range(1, 8)},
                **{f"joint{index}_left(_1)?": 0.0 for index in range(1, 8)},
                "right_finger1_joint1": 0.10,
                "left_finger1_joint1": 0.10,
            }

    class RobotCfg:
        def __init__(self):
            self.init_state = InitState()

    a01 = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
    )
    a01._sample_reset_state(clear_manual_request=True)
    np.testing.assert_allclose(a01._right_tcp_pose, _A01_CLEAR_RIGHT_START_TCP)
    np.testing.assert_allclose(a01._left_tcp_pose, _RAISED_LEFT_START_TCP)
    a01_right_cfg = RobotCfg()
    a01_left_cfg = RobotCfg()
    assert a01._apply_raised_hand_home(a01_right_cfg, "right")
    assert a01._apply_raised_hand_home(a01_left_cfg, "left")
    for side, cfg in (("right", a01_right_cfg), ("left", a01_left_cfg)):
        np.testing.assert_allclose(
            [
                cfg.init_state.joint_pos[f"joint{index}_{side}(_1)?"]
                for index in range(1, 8)
            ],
            _A01_CLEAR_ARM_HOME[side],
        )

    for task_name in (
        "t25_place_cup_on_plate",
        "t04_cover_pitcher",
        "t09_insert_peg",
        "t44_stack_blocks",
        "t50_water_plant",
    ):
        env = HandsOnABCIsaacLabEnv(
            task_name,
            launch_isaac=False,
            image_obs=False,
        )
        env._sample_reset_state(clear_manual_request=True)
        np.testing.assert_allclose(env._right_tcp_pose, _RIGHT_HOME)
        np.testing.assert_allclose(env._left_tcp_pose, _RAISED_LEFT_START_TCP)

        right_cfg = RobotCfg()
        left_cfg = RobotCfg()
        assert not env._apply_raised_hand_home(right_cfg, "right")
        assert env._apply_raised_hand_home(left_cfg, "left")

        np.testing.assert_allclose(
            [
                right_cfg.init_state.joint_pos[f"joint{index}_right(_1)?"]
                for index in range(1, 8)
            ],
            np.zeros(7, dtype=np.float64),
        )
        np.testing.assert_allclose(
            [
                left_cfg.init_state.joint_pos[f"joint{index}_left(_1)?"]
                for index in range(1, 8)
            ],
            _RAISED_ARM_HOME["left"],
        )
        assert right_cfg.init_state.joint_pos["right_finger1_joint1"] == 0.10
        assert left_cfg.init_state.joint_pos["left_finger1_joint1"] == 0.10

    b05 = HandsOnABCIsaacLabEnv(
        "t03_cover_box",
        launch_isaac=False,
        image_obs=False,
    )
    b05._sample_reset_state(clear_manual_request=True)
    np.testing.assert_allclose(b05._right_tcp_pose, _RIGHT_HOME)
    np.testing.assert_allclose(b05._left_tcp_pose, _RAISED_LEFT_START_TCP)
    b05_right_cfg = RobotCfg()
    b05_left_cfg = RobotCfg()
    assert not b05._apply_raised_hand_home(b05_right_cfg, "right")
    assert b05._apply_raised_hand_home(b05_left_cfg, "left")
    np.testing.assert_allclose(
        [
            b05_right_cfg.init_state.joint_pos[f"joint{index}_right(_1)?"]
            for index in range(1, 8)
        ],
        np.zeros(7, dtype=np.float64),
    )
    np.testing.assert_allclose(
        [
            b05_left_cfg.init_state.joint_pos[f"joint{index}_left(_1)?"]
            for index in range(1, 8)
        ],
        _RAISED_ARM_HOME["left"],
    )

    c05 = HandsOnABCIsaacLabEnv("t14_move_and_click_mouse", launch_isaac=False, image_obs=False)
    c05._sample_reset_state(clear_manual_request=True)
    np.testing.assert_allclose(c05._right_tcp_pose, _RIGHT_HOME)
    np.testing.assert_allclose(c05._left_tcp_pose, _LEFT_HOME)

    c05_right_cfg = RobotCfg()
    c05_left_cfg = RobotCfg()
    assert not c05._apply_raised_hand_home(c05_right_cfg, "right")
    assert not c05._apply_raised_hand_home(c05_left_cfg, "left")


def test_c05_reset_keeps_mouse_outside_mousepad() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t14_move_and_click_mouse",
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )

    mouse_xy, mousepad_xy = env._c05_reset_xy_positions()

    np.testing.assert_allclose(mouse_xy, _C05_DEFAULT_MOUSE_XY)
    np.testing.assert_allclose(mousepad_xy, _C05_DEFAULT_MOUSEPAD_XY)
    assert np.linalg.norm(mousepad_xy - mouse_xy) >= _C05_MOUSEPAD_MIN_OFFSET

    env.randomize = True
    hand_points = (
        *env._reset_robot_body_points("right"),
        *env._reset_robot_body_points("left"),
    )
    mouse_positions = []
    mousepad_positions = []
    for _ in range(256):
        mouse_xy, mousepad_xy = env._c05_reset_xy_positions()
        mouse_positions.append(mouse_xy)
        mousepad_positions.append(mousepad_xy)
        assert np.all(mouse_xy >= _C05_MOUSE_XY_BOUNDS[0])
        assert np.all(mouse_xy <= _C05_MOUSE_XY_BOUNDS[1])
        assert np.all(mousepad_xy >= _C05_MOUSEPAD_XY_BOUNDS[0])
        assert np.all(mousepad_xy <= _C05_MOUSEPAD_XY_BOUNDS[1])
        assert np.linalg.norm(mousepad_xy - mouse_xy) >= _C05_MOUSEPAD_MIN_OFFSET - 1e-9
        assert (
            env._c05_reset_layout_margin(mouse_xy, mousepad_xy, hand_points)
            >= _C05_RESET_MIN_CLEARANCE_SLACK
        )

    mouse_span = np.ptp(np.asarray(mouse_positions), axis=0)
    mousepad_span = np.ptp(np.asarray(mousepad_positions), axis=0)
    assert mouse_span[0] > 0.03
    assert mouse_span[1] > 0.08
    assert mousepad_span[0] > 0.03
    assert mousepad_span[1] > 0.10


def test_c06_camera_reset_stays_in_reachable_workspace() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t23_photograph_logo",
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )

    np.testing.assert_allclose(
        env._c06_reset_xy_position(),
        _C06_DEFAULT_CAMERA_XY,
    )

    env.randomize = True
    hand_points = (
        *env._reset_robot_body_points("right"),
        *env._reset_robot_body_points("left"),
    )
    for _ in range(100):
        camera_xy, yaw_degrees = env._c06_reset_pose()
        assert np.all(camera_xy >= _C06_CAMERA_XY_BOUNDS[0])
        assert np.all(camera_xy <= _C06_CAMERA_XY_BOUNDS[1])
        assert (
            _C06_CAMERA_YAW_BOUNDS_DEG[0]
            <= yaw_degrees
            <= _C06_CAMERA_YAW_BOUNDS_DEG[1]
        )
        assert (
            env._c06_reset_layout_margin(
                camera_xy,
                hand_points,
                yaw_degrees,
            )
            >= _C06_RESET_MIN_CLEARANCE_SLACK
        )


def test_c06_camera_reset_rejects_yawed_footprint_touching_hand() -> None:
    hand_points = (
        np.asarray([-0.10, -0.16, 1.07], dtype=np.float64),
        np.asarray([-0.10, 0.16, 1.07], dtype=np.float64),
    )
    camera_xy = np.asarray([-0.20, -0.12], dtype=np.float64)

    center_only_margin = min(
        float(np.linalg.norm(camera_xy - point[:2]))
        for point in hand_points
    ) - _C06_CAMERA_HAND_CLEARANCE
    oriented_margin = HandsOnABCIsaacLabEnv._c06_reset_layout_margin(
        camera_xy,
        hand_points,
        10.0,
    )

    assert center_only_margin >= _C06_RESET_MIN_CLEARANCE_SLACK
    assert oriented_margin < _C06_RESET_MIN_CLEARANCE_SLACK


def test_c06_camera_grasp_physics_is_light_and_high_friction() -> None:
    physics = _C06_CAMERA_GRASP_PHYSICS

    assert physics.mass == 0.140
    assert physics.static_friction == 1.6
    assert physics.dynamic_friction == 1.2
    assert physics.linear_damping == 0.40
    assert physics.angular_damping == 2.00


def test_c06_isaac_requires_aimed_camera_and_pressed_shutter() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t23_photograph_logo",
        launch_isaac=False,
        image_obs=False,
    )
    env._poses = {
        "camera": np.asarray(
            [0.0, 0.18, 0.0, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        "table": np.asarray(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    }
    shutter_position = _C06_SHUTTER_PRESS_TRAVEL + 0.0001
    env._physics_dof_position = lambda _name: (shutter_position, 0.0)
    env._c06_right_hand_shutter_contact_force = lambda: 0.10

    metrics = env._c06_photograph_metrics()

    assert metrics["c06_camera_aimed"]
    assert metrics["c06_right_hand_contact"]
    assert metrics["c06_shutter_pressed"]
    assert env._23_photograph_logo_success_now()
    assert env._success_steps_required() == 1
    assert [row["key"] for row in env._success_condition_rows()] == [
        "c06_camera_aimed",
        "c06_right_hand_contact",
        "c06_shutter_pressed",
    ]

    env._c06_right_contact_grace = 0
    env._c06_right_hand_shutter_contact_force = lambda: 0.0
    assert not env._23_photograph_logo_success_now()
    env._c06_right_hand_shutter_contact_force = lambda: 0.10

    env._poses["camera"][3:7] = np.asarray([0.0, 1.0, 0.0, 0.0])
    assert not env._23_photograph_logo_success_now()

    env._poses["camera"][3:7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    shutter_position = 0.0
    assert not env._23_photograph_logo_success_now()


def test_c06_shutter_proximity_requires_a_visible_right_fingertip() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t23_photograph_logo",
        launch_isaac=False,
        image_obs=False,
    )
    env._poses = {
        "camera": np.asarray(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    }
    shutter_position = np.asarray([-0.105, 0.062, -0.031], dtype=np.float64)
    tips = {"right_finger2_tip": shutter_position + [0.020, 0.0, 0.0]}
    env._wuji_fingertip_positions = lambda side: tips if side == "right" else {}

    near, distance = env._c06_right_hand_shutter_proximity()

    assert near
    assert distance == pytest.approx(0.020)

    tips["right_finger2_tip"] = shutter_position + [0.040, 0.0, 0.0]
    near, distance = env._c06_right_hand_shutter_proximity()
    assert not near
    assert distance == pytest.approx(0.040)


def _c05_logic_env() -> HandsOnABCIsaacLabEnv:
    env = HandsOnABCIsaacLabEnv(
        "t14_move_and_click_mouse",
        launch_isaac=False,
        image_obs=False,
    )
    env._c05_mouse_start_xy = _C05_DEFAULT_MOUSE_XY.copy()
    env._c05_button_initial_positions = {
        "mouse_joint_0": 0.0,
        "mouse_joint_1": 0.0,
    }
    env._poses = {
        "link_3": np.asarray([-0.36, -0.34, 0.91, 1.0, 0.0, 0.0, 0.0]),
        "mouse": np.asarray([-0.36, -0.34, 0.91, 1.0, 0.0, 0.0, 0.0]),
        "mousepad": np.asarray([-0.08, -0.34, 0.89, 1.0, 0.0, 0.0, 0.0]),
    }
    return env


def test_c05_button_press_while_moving_fails_immediately() -> None:
    env = _c05_logic_env()
    button_positions = {
        "mouse_joint_0": _C05_BUTTON_PRESS_THRESHOLD + 0.0002,
        "mouse_joint_1": 0.0,
    }
    env._physics_dof_position = lambda name: (button_positions[name], 0.0)

    assert not env._compute_success()
    assert env._compute_failure()
    assert env._last_c05_success_metrics["c05_failed_early_click"]


def test_c05_requires_move_to_pad_then_left_click() -> None:
    env = _c05_logic_env()
    env._poses["link_3"][:2] = env._poses["mousepad"][:2]
    env._poses["mouse"][:2] = env._poses["mousepad"][:2]
    button_positions = {
        "mouse_joint_0": _C05_BUTTON_PRESS_THRESHOLD + 0.0002,
        "mouse_joint_1": 0.0,
    }
    env._physics_dof_position = lambda name: (button_positions[name], 0.0)

    for _ in range(_C05_SUCCESS_STEPS - 1):
        assert not env._compute_success()
    assert env._compute_success()
    assert not env._compute_failure()

    keys = [row["key"] for row in env._success_condition_rows()]
    assert keys == [
        "c05_mouse_moved",
        "c05_mouse_on_pad",
        "c05_no_early_click",
        "c05_left_clicked",
        "success_hold",
    ]


def test_a01_reset_centers_are_clamped_to_the_right_hand_workspace() -> None:
    constrained = _constrain_task_xy_to_workspace(
        "t26_place_fruit_in_bucket",
        np.asarray([[-0.80, -0.70], [0.20, 0.40], [-0.30, -0.20]]),
    )

    np.testing.assert_allclose(
        constrained,
        [
            _A01_RIGHT_HAND_XY_BOUNDS[0],
            _A01_RIGHT_HAND_XY_BOUNDS[1],
            [-0.30, -0.20],
        ],
    )


def test_a01_reset_sampler_spreads_objects_clear_of_reset_hands() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=7,
    )

    layouts = [env._a01_reset_xy_positions() for _ in range(96)]

    assert _A01_YAW_BOUNDS_DEG == (-180.0, 180.0)
    for bounds in _A01_RESET_XY_BOUNDS.values():
        np.testing.assert_allclose(bounds, RECORDING_OBJECT_XY_BOUNDS)
    positions_by_name = {
        name: np.stack([layout[name] for layout in layouts if layout is not None])
        for name in _A01_RESET_XY_BOUNDS
    }
    bucket_spread = np.ptp(positions_by_name["bucket"], axis=0)
    assert bucket_spread[0] > 0.24
    assert bucket_spread[1] > 0.50
    fruit_spreads = [
        np.ptp(positions_by_name[name], axis=0)
        for name in positions_by_name
        if name != "bucket"
    ]
    assert min(spread[0] for spread in fruit_spreads) > 0.24
    assert min(spread[1] for spread in fruit_spreads) > 0.50
    for layout in layouts:
        assert layout is not None
        assert env._a01_reset_layout_is_clear(layout)
        assert env._a01_reset_layout_margin(layout) >= _A01_RESET_MIN_CLEARANCE_SLACK
        for name, xy in layout.items():
            effective_bounds = np.stack(
                (
                    np.maximum(
                        _A01_RESET_XY_BOUNDS[name][0],
                        RECORDING_OBJECT_XY_BOUNDS[0],
                    ),
                    np.minimum(
                        _A01_RESET_XY_BOUNDS[name][1],
                        RECORDING_OBJECT_XY_BOUNDS[1],
                    ),
                )
            )
            assert np.all(xy >= effective_bounds[0])
            assert np.all(xy <= effective_bounds[1])
            assert np.all(xy >= RECORDING_OBJECT_XY_BOUNDS[0])
            assert np.all(xy <= RECORDING_OBJECT_XY_BOUNDS[1])


def test_a01_reset_fallback_keeps_footprints_clear() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )

    layout = env._a01_reset_xy_positions()

    assert layout is not None
    assert env._a01_reset_layout_is_clear(layout)
    assert env._a01_reset_layout_margin(layout) >= _A01_RESET_MIN_CLEARANCE_SLACK
    names = sorted(layout)
    for index, name_a in enumerate(names):
        for name_b in names[index + 1 :]:
            distance = float(np.linalg.norm(layout[name_a] - layout[name_b]))
            assert distance >= env._a01_pair_clearance(name_a, name_b)


def test_a01_reset_sampler_avoids_right_hand_body_points() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=31,
    )
    right_body_points = (
        np.asarray([-0.30, -0.11, 1.02], dtype=np.float64),
        np.asarray([-0.22, 0.10, 1.02], dtype=np.float64),
    )
    env._reset_robot_body_points = lambda side: right_body_points if side == "right" else ()

    layouts = [env._a01_reset_xy_positions() for _ in range(32)]

    for layout in layouts:
        assert layout is not None
        assert env._a01_reset_layout_margin(
            layout,
            right_body_points,
        ) >= _A01_RESET_MIN_CLEARANCE_SLACK
        for name, xy in layout.items():
            hand_clearance = min(
                float(np.linalg.norm(xy - point[:2]))
                for point in right_body_points
            )
            assert hand_clearance >= env._a01_object_hand_clearance(name)


def test_a01_live_robot_reset_cloud_keeps_random_sampler_bounded() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=38,
    )
    right_body_points = tuple(
        np.asarray(point, dtype=np.float64)
        for point in (
            (-0.4615, -0.2395, 0.9975),
            (-0.3715, -0.2395, 0.9975),
            (-0.3616, -0.2813, 0.9557),
            (-0.3426, -0.2189, 0.9891),
            (-0.3125, -0.1455, 0.9790),
            (-0.2796, -0.2101, 1.0034),
            (-0.2326, -0.2037, 1.0080),
            (-0.2033, -0.2002, 1.0094),
            (-0.2221, -0.2898, 1.0068),
        )
    )
    env._reset_robot_body_points = (
        lambda side: right_body_points if side == "right" else ()
    )

    layout = env._a01_reset_xy_positions()

    assert layout is not None
    assert layout["bucket"][1] >= _A01_RESET_XY_BOUNDS["bucket"][1, 1] - 0.04
    assert (
        env._a01_reset_layout_margin(layout, right_body_points)
        >= _A01_RESET_MIN_CLEARANCE_SLACK
    )


def test_a01_complete_footprint_sampler_is_clear_for_one_hundred_seeds() -> None:
    for seed in range(100):
        env = HandsOnABCIsaacLabEnv(
            "t26_place_fruit_in_bucket",
            launch_isaac=False,
            image_obs=False,
            randomize=True,
            seed=seed,
        )
        layout = env._a01_reset_xy_positions()
        assert layout is not None
        hand_points = env._reset_robot_body_points("right")
        assert (
            env._a01_reset_layout_margin(layout, hand_points)
            >= _A01_RESET_MIN_CLEARANCE_SLACK
        )

    # This separation was present in a rejected physical event.  The former
    # 12 cm bucket radius accepted it even though the apple crossed a wall.
    assert (
        HandsOnABCIsaacLabEnv._a01_bucket_point_margin(
            np.asarray([-0.15 + 0.215, 0.27]),
            np.asarray([-0.15, 0.27]),
        )
        - 0.038
        - 0.020
        < 0.0
    )


def test_a01_a06_b04_support_roots_match_real_rokae_table() -> None:
    assert _A01_OBJECT_FOOTPRINT_RADII["pear"] == pytest.approx(
        np.hypot(0.028, 0.028)
    )
    assert 0.89 - _A01_SUPPORT_LOCAL_MIN_Z["bucket"] == pytest.approx(0.970)
    assert 0.89 - _A01_SUPPORT_LOCAL_MIN_Z["011_banana"] == pytest.approx(
        0.909722
    )
    assert 0.89 - _A01_SUPPORT_LOCAL_MIN_Z["apple"] == pytest.approx(0.928)
    assert 0.89 - _A01_SUPPORT_LOCAL_MIN_Z["orange"] == pytest.approx(0.927)
    assert 0.89 - _A01_SUPPORT_LOCAL_MIN_Z["pear"] == pytest.approx(0.942)
    assert 0.89 - _A06_SIDE_REST_SUPPORT_LOCAL_MIN_Z == pytest.approx(
        0.914622404
    )
    assert 0.89 - _B04_SUPPORT_LOCAL_MIN_Z["036_wood_block"] == pytest.approx(
        0.932689
    )
    assert 0.89 - _B04_SUPPORT_LOCAL_MIN_Z["BuildingBlock017"] == pytest.approx(
        0.904925
    )
    assert 0.89 - _B04_SUPPORT_LOCAL_MIN_Z["BuildingBlock018"] == pytest.approx(
        0.905
    )
    assert 0.89 - _B04_SUPPORT_LOCAL_MIN_Z["BuildingBlock019"] == pytest.approx(
        0.905
    )


def test_a01_runtime_authors_complete_source_inertials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_inertials = {
        "011_banana": {
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "mass": "0.07",
            "diaginertia": "0.00023 0.000035 0.000247",
        },
        "apple": {
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "mass": "0.10",
            "diaginertia": "0.00005776 0.00005776 0.00005776",
        },
        "orange": {
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "mass": "0.11",
            "diaginertia": "0.000060236 0.000060236 0.000060236",
        },
        "pear": {
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "mass": "0.11",
            "diaginertia": "0.000128 0.000128 0.0000575",
        },
        "bucket": {
            "pos": "-0.00918 -0.00267 -0.00466",
            "quat": "0.501 0.5007 0.4993 0.4990",
            "mass": "1.62832",
            "diaginertia": "0.04345 0.02946 0.02595",
        },
    }

    class FakeAttribute:
        def __init__(self) -> None:
            self.value = None

        def IsValid(self) -> bool:
            return True

        def Set(self, value) -> None:
            self.value = value

    class FakePrim:
        def __init__(self, name: str, *, collision: bool = False) -> None:
            self.name = name
            self.path = f"/World/Task/{name}"
            self.collision = collision
            self.mass_api = None
            self.rigid_api = None
            self.collision_api = None
            self.descendants = []

        def GetPath(self) -> str:
            return self.path

        def HasAPI(self, api) -> bool:
            return api is FakeCollisionSchema and self.collision

    class FakeMassAPI:
        def __init__(self, prim: FakePrim) -> None:
            self.attributes = {}
            prim.mass_api = self

        @classmethod
        def Apply(cls, prim: FakePrim):
            return cls(prim)

        def _attribute(self, name: str) -> FakeAttribute:
            return self.attributes.setdefault(name, FakeAttribute())

        def CreateMassAttr(self) -> FakeAttribute:
            return self._attribute("mass")

        def CreateCenterOfMassAttr(self) -> FakeAttribute:
            return self._attribute("center_of_mass")

        def CreateDiagonalInertiaAttr(self) -> FakeAttribute:
            return self._attribute("diagonal_inertia")

        def CreatePrincipalAxesAttr(self) -> FakeAttribute:
            return self._attribute("principal_axes")

    class FakeRigidAPI:
        def __init__(self, prim: FakePrim) -> None:
            self.attributes = {}
            prim.rigid_api = self

        @classmethod
        def Apply(cls, prim: FakePrim):
            return cls(prim)

        def _attribute(self, name: str) -> FakeAttribute:
            return self.attributes.setdefault(name, FakeAttribute())

        def GetLinearDampingAttr(self) -> FakeAttribute:
            return self._attribute("linear_damping")

        CreateLinearDampingAttr = GetLinearDampingAttr

        def GetAngularDampingAttr(self) -> FakeAttribute:
            return self._attribute("angular_damping")

        CreateAngularDampingAttr = GetAngularDampingAttr

        def GetSleepThresholdAttr(self) -> FakeAttribute:
            return self._attribute("sleep_threshold")

        CreateSleepThresholdAttr = GetSleepThresholdAttr

    class FakeCollisionAPI:
        def __init__(self, prim: FakePrim) -> None:
            self.attributes = {}
            prim.collision_api = self

        @classmethod
        def Apply(cls, prim: FakePrim):
            return cls(prim)

        def CreateContactOffsetAttr(self) -> FakeAttribute:
            return self.attributes.setdefault("contact_offset", FakeAttribute())

        def CreateRestOffsetAttr(self) -> FakeAttribute:
            return self.attributes.setdefault("rest_offset", FakeAttribute())

    class FakeCollisionSchema:
        def __init__(self, prim):
            self.prim = prim
        def GetCollisionEnabledAttr(self):
            return types.SimpleNamespace(Get=lambda: self.prim.collision)

    prims = {
        name: FakePrim(name)
        for name in source_inertials
    }
    for name, prim in prims.items():
        collider = FakePrim(f"{name}_collision", collision=True)
        prim.descendants = [collider]

    class FakeStage:
        def GetPrimAtPath(self, path: str) -> FakePrim:
            return next(prim for prim in prims.values() if prim.path == path)

    pxr = types.ModuleType("pxr")
    pxr.Gf = types.SimpleNamespace(
        Vec3f=lambda *values: tuple(values),
        Quatf=lambda real, imaginary: (real, *imaginary),
    )
    pxr.PhysxSchema = types.SimpleNamespace(
        PhysxCollisionAPI=FakeCollisionAPI,
        PhysxRigidBodyAPI=FakeRigidAPI,
    )
    pxr.Usd = types.SimpleNamespace(PrimRange=lambda prim: prim.descendants)
    pxr.UsdPhysics = types.SimpleNamespace(
        CollisionAPI=FakeCollisionSchema,
        MassAPI=FakeMassAPI,
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    task_preview = types.ModuleType("handson.sim.isaaclab.task_preview")
    task_preview._A01_ANALYTIC_FRUIT_INERTIALS = source_inertials
    monkeypatch.setitem(
        sys.modules,
        "handson.sim.isaaclab.task_preview",
        task_preview,
    )

    class FakeSimUtils:
        def __init__(self) -> None:
            self.uninstanced = []
            self.stage_updates = 0

        def make_uninstanceable(self, path: str, *, stage) -> None:
            assert stage is env._stage
            self.uninstanced.append(path)

        def update_stage(self) -> None:
            self.stage_updates += 1

    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
    )
    env._stage = FakeStage()
    env._find_rigid_body_prim = prims.get
    sim_utils = FakeSimUtils()

    assert env._configure_a01_reset_stability(sim_utils) == 5
    assert sim_utils.stage_updates == 5
    assert set(sim_utils.uninstanced) == {
        prim.path for prim in prims.values()
    }
    for name, source in source_inertials.items():
        attributes = prims[name].mass_api.attributes
        assert attributes["mass"].value == float(source["mass"])
        assert attributes["center_of_mass"].value == tuple(
            float(value) for value in source["pos"].split()
        )
        assert attributes["diagonal_inertia"].value == tuple(
            float(value) for value in source["diaginertia"].split()
        )
        assert attributes["principal_axes"].value == tuple(
            float(value) for value in source["quat"].split()
        )

    env.task_name = "t25_place_cup_on_plate"
    assert env._configure_a01_reset_stability(sim_utils) == 0
    env.task_name = "t26_place_fruit_in_bucket"

    bucket_colliders = prims["bucket"].descendants
    prims["bucket"].descendants = []
    with pytest.raises(RuntimeError, match="bucket rigid body has no colliders"):
        env._configure_a01_reset_stability(sim_utils)
    assert any(
        "bucket rigid body has no colliders" in warning
        for warning in env._isaac_warnings
    )
    prims["bucket"].descendants = bucket_colliders

    env._find_rigid_body_prim = (
        lambda name: None if name == "pear" else prims.get(name)
    )
    with pytest.raises(RuntimeError, match="pear rigid body is unavailable"):
        env._configure_a01_reset_stability(sim_utils)
    assert any(
        "pear rigid body is unavailable" in warning
        for warning in env._isaac_warnings
    )


def test_a02_reset_sampler_spreads_tray_and_mug_without_overlap() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t25_place_cup_on_plate",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=23,
    )

    layouts = [env._a02_reset_xy_positions() for _ in range(96)]

    assert _A02_YAW_BOUNDS_DEG == (-180.0, 180.0)
    for bounds in _A02_RESET_XY_BOUNDS.values():
        np.testing.assert_allclose(bounds, RECORDING_OBJECT_XY_BOUNDS)
    positions_by_name = {
        name: np.stack([layout[name] for layout in layouts if layout is not None])
        for name in _A02_RESET_XY_BOUNDS
    }
    tray_spread = np.ptp(positions_by_name["tray"], axis=0)
    mug_spread = np.ptp(positions_by_name["025_mug"], axis=0)
    assert tray_spread[0] > 0.24
    assert tray_spread[1] > 0.50
    assert mug_spread[0] > 0.24
    assert mug_spread[1] > 0.50
    for layout in layouts:
        assert layout is not None
        assert env._a02_reset_layout_is_clear(layout)
        assert env._a02_reset_layout_margin(layout) >= _A02_RESET_MIN_CLEARANCE_SLACK
        for name, xy in layout.items():
            effective_bounds = np.stack(
                (
                    np.maximum(
                        _A02_RESET_XY_BOUNDS[name][0],
                        RECORDING_OBJECT_XY_BOUNDS[0],
                    ),
                    np.minimum(
                        _A02_RESET_XY_BOUNDS[name][1],
                        RECORDING_OBJECT_XY_BOUNDS[1],
                    ),
                )
            )
            assert np.all(xy >= effective_bounds[0])
            assert np.all(xy <= effective_bounds[1])
            assert np.all(xy >= RECORDING_OBJECT_XY_BOUNDS[0])
            assert np.all(xy <= RECORDING_OBJECT_XY_BOUNDS[1])


def test_a02_reset_fallback_keeps_tray_and_mug_clear() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t25_place_cup_on_plate",
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )

    layout = env._a02_reset_xy_positions()

    assert layout is not None
    assert env._a02_reset_layout_is_clear(layout)
    assert env._a02_reset_layout_margin(layout) >= _A02_RESET_MIN_CLEARANCE_SLACK
    distance = float(np.linalg.norm(layout["025_mug"] - layout["tray"]))
    assert distance >= env._a02_pair_clearance("025_mug", "tray")


def test_a02_reset_sampler_avoids_right_hand_body_points() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t25_place_cup_on_plate",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=37,
    )
    right_body_points = (
        np.asarray([-0.30, -0.10, 1.02], dtype=np.float64),
        np.asarray([-0.20, 0.16, 1.02], dtype=np.float64),
    )
    env._reset_robot_body_points = lambda side: right_body_points if side == "right" else ()

    layouts = [env._a02_reset_xy_positions() for _ in range(32)]

    for layout in layouts:
        assert layout is not None
        assert env._a02_reset_layout_margin(
            layout,
            right_body_points,
        ) >= _A02_RESET_MIN_CLEARANCE_SLACK
        for name, xy in layout.items():
            hand_clearance = min(
                float(np.linalg.norm(xy - point[:2]))
                for point in right_body_points
            )
            assert hand_clearance >= env._a02_object_hand_clearance(name)


def test_a03_reset_sampler_spreads_balls_and_bowl_without_overlap() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t24_place_ball_in_bowl",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=43,
    )

    layouts = [env._a03_reset_xy_positions() for _ in range(96)]

    assert _A03_YAW_BOUNDS_DEG == (-180.0, 180.0)
    for bounds in _A03_RESET_XY_BOUNDS.values():
        np.testing.assert_allclose(bounds, RECORDING_OBJECT_XY_BOUNDS)
    positions_by_name = {
        name: np.stack([layout[name] for layout in layouts if layout is not None])
        for name in _A03_RESET_XY_BOUNDS
    }
    for positions in positions_by_name.values():
        spread = np.ptp(positions, axis=0)
        assert spread[0] > 0.16
        assert spread[1] > 0.34
    for layout in layouts:
        assert layout is not None
        assert env._a03_reset_layout_is_clear(layout)
        assert env._a03_reset_layout_margin(layout) >= _A03_RESET_MIN_CLEARANCE_SLACK
        for name, xy in layout.items():
            effective_bounds = np.stack(
                (
                    np.maximum(
                        _A03_RESET_XY_BOUNDS[name][0],
                        RECORDING_OBJECT_XY_BOUNDS[0],
                    ),
                    np.minimum(
                        _A03_RESET_XY_BOUNDS[name][1],
                        RECORDING_OBJECT_XY_BOUNDS[1],
                    ),
                )
            )
            assert np.all(xy >= effective_bounds[0])
            assert np.all(xy <= effective_bounds[1])


def test_a03_reset_sampler_avoids_live_hand_body_points() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t24_place_ball_in_bowl",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=47,
    )
    right_body_points = (
        np.asarray([-0.30, -0.10, 1.02], dtype=np.float64),
        np.asarray([-0.20, 0.16, 1.02], dtype=np.float64),
    )
    left_body_points = (
        np.asarray([-0.32, 0.18, 1.02], dtype=np.float64),
    )
    all_body_points = right_body_points + left_body_points
    env._reset_robot_body_points = (
        lambda side: right_body_points if side == "right" else left_body_points
    )

    layouts = [env._a03_reset_xy_positions() for _ in range(32)]

    for layout in layouts:
        assert layout is not None
        assert env._a03_reset_layout_margin(
            layout,
            all_body_points,
        ) >= _A03_RESET_MIN_CLEARANCE_SLACK
        for name, xy in layout.items():
            hand_clearance = min(
                float(np.linalg.norm(xy - point[:2]))
                for point in all_body_points
            )
            assert hand_clearance >= env._a03_object_hand_clearance(name)


def test_b01_reset_sampler_spreads_pitcher_and_lid_without_overlap() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t04_cover_pitcher",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=41,
    )

    layouts = [env._b01_reset_xy_positions() for _ in range(96)]

    assert _B01_YAW_BOUNDS_DEG == (-180.0, 180.0)
    for bounds in _B01_RESET_XY_BOUNDS.values():
        np.testing.assert_allclose(bounds, RECORDING_OBJECT_XY_BOUNDS)
    assert 0.89 - _B01_SUPPORT_LOCAL_MIN_Z["019_pitcher_base"] == pytest.approx(
        1.0081775
    )
    assert 0.89 - _B01_SUPPORT_LOCAL_MIN_Z["pitcher_lid"] == pytest.approx(
        0.9140045
    )
    positions_by_name = {
        name: np.stack([layout[name] for layout in layouts if layout is not None])
        for name in _B01_RESET_XY_BOUNDS
    }
    base_spread = np.ptp(positions_by_name["019_pitcher_base"], axis=0)
    lid_spread = np.ptp(positions_by_name["pitcher_lid"], axis=0)
    assert base_spread[0] > 0.24
    assert base_spread[1] > 0.50
    assert lid_spread[0] > 0.24
    assert lid_spread[1] > 0.50
    for layout in layouts:
        assert layout is not None
        assert env._b01_reset_layout_is_clear(layout)
        assert env._b01_reset_layout_margin(layout) >= _B01_RESET_MIN_CLEARANCE_SLACK
        for name, xy in layout.items():
            effective_bounds = np.stack(
                (
                    np.maximum(
                        _B01_RESET_XY_BOUNDS[name][0],
                        RECORDING_OBJECT_XY_BOUNDS[0],
                    ),
                    np.minimum(
                        _B01_RESET_XY_BOUNDS[name][1],
                        RECORDING_OBJECT_XY_BOUNDS[1],
                    ),
                )
            )
            assert np.all(xy >= effective_bounds[0])
            assert np.all(xy <= effective_bounds[1])
            assert np.all(xy >= RECORDING_OBJECT_XY_BOUNDS[0])
            assert np.all(xy <= RECORDING_OBJECT_XY_BOUNDS[1])


def test_b01_reset_fallback_keeps_pitcher_and_lid_clear() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t04_cover_pitcher",
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )

    layout = env._b01_reset_xy_positions()

    assert layout is not None
    assert env._b01_reset_layout_is_clear(layout)
    assert env._b01_reset_layout_margin(layout) >= _B01_RESET_MIN_CLEARANCE_SLACK
    distance = float(np.linalg.norm(layout["019_pitcher_base"] - layout["pitcher_lid"]))
    assert distance >= env._b01_pair_clearance("019_pitcher_base", "pitcher_lid")


def test_b01_reset_sampler_avoids_right_hand_body_points() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t04_cover_pitcher",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=43,
    )
    right_body_points = (
        np.asarray([-0.30, -0.10, 1.02], dtype=np.float64),
        np.asarray([-0.20, 0.16, 1.02], dtype=np.float64),
    )
    env._reset_robot_body_points = (
        lambda side: right_body_points if side == "right" else ()
    )

    layouts = [env._b01_reset_xy_positions() for _ in range(32)]

    for layout in layouts:
        assert layout is not None
        assert env._b01_reset_layout_margin(
            layout,
            right_body_points,
        ) >= _B01_RESET_MIN_CLEARANCE_SLACK
        for name, xy in layout.items():
            hand_clearance = min(
                float(np.linalg.norm(xy - point[:2]))
                for point in right_body_points
            )
            assert hand_clearance >= env._b01_object_hand_clearance(name)


def test_default_isaaclab_seed_uses_time_when_not_provided(monkeypatch) -> None:
    monkeypatch.setattr(assembly_env.time, "time_ns", lambda: 123_456_789)
    monkeypatch.setattr(assembly_env.os, "getpid", lambda: 321)

    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
    )

    expected_seed = (123_456_789 ^ (321 << 16)) % (2**32 - 1)
    assert env._seed == expected_seed


@pytest.mark.parametrize(
    "task_name",
    [
        "t19_orient_cup_upright",

        "t44_stack_blocks",
        "t03_cover_box",
        "t36_scoop_marbles",
    ],
)
def test_generic_task_reset_sampler_places_objects_without_overlap(task_name: str) -> None:
    env = HandsOnABCIsaacLabEnv(
        task_name,
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=53,
    )

    layouts = [env._task_specific_reset_xy_positions() for _ in range(32)]
    hand_points = (
        *env._reset_robot_body_points("right"),
        *env._reset_robot_body_points("left"),
    )

    for layout in layouts:
        assert layout is not None
        assert set(layout) == set(_TASK_SPECIFIC_RESET_XY_BOUNDS[task_name])
        assert (
            env._task_specific_reset_layout_margin(layout, hand_points)
            >= _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK
        )


def test_b04_complete_rotating_footprints_are_clear_for_one_hundred_seeds() -> None:
    for seed in range(100):
        env = HandsOnABCIsaacLabEnv(
            "t44_stack_blocks",
            launch_isaac=False,
            image_obs=False,
            randomize=True,
            seed=seed,
        )
        layout = env._task_specific_reset_xy_positions()
        assert layout is not None
        hand_points = (
            *env._reset_robot_body_points("right"),
            *env._reset_robot_body_points("left"),
        )
        assert (
            env._task_specific_reset_layout_margin(layout, hand_points)
            >= _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK
        )
        names = sorted(layout)
        for index, name_a in enumerate(names):
            for name_b in names[index + 1 :]:
                required = (
                    _B04_RESET_FOOTPRINT_RADII[name_a]
                    + _B04_RESET_FOOTPRINT_RADII[name_b]
                    + _B04_RESET_OBJECT_CLEARANCE_MARGIN
                    + _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK
                )
                assert np.linalg.norm(layout[name_a] - layout[name_b]) >= required


def test_b04_hand_clearance_uses_the_swept_cylinder_height() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t44_stack_blocks",
        launch_isaac=False,
        image_obs=False,
    )
    body_name = "036_wood_block"
    body_xy = np.asarray([-0.27, 0.14], dtype=np.float64)
    half_height = float(_STACK_BLOCK_HALF_EXTENTS[body_name][2])
    body_center_z = 0.89 + float(env.delta_h[0]) + half_height
    clearance_slack = _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK + 0.001

    high_same_xy = np.asarray(
        [
            body_xy[0],
            body_xy[1],
            body_center_z
            + half_height
            + _B04_RESET_HAND_CLEARANCE_MARGIN
            + clearance_slack,
        ],
        dtype=np.float64,
    )
    assert env._b04_point_footprint_margin(
        high_same_xy,
        body_xy,
        body_name,
    ) == pytest.approx(clearance_slack)

    low_same_xy = np.asarray(
        [body_xy[0], body_xy[1], body_center_z],
        dtype=np.float64,
    )
    assert env._b04_point_footprint_margin(
        low_same_xy,
        body_xy,
        body_name,
    ) < 0.0


def test_b04_production_reset_origins_find_clear_layout_for_one_hundred_seeds() -> None:
    # Representative body origins from the production ROKAE/Wuji reset.  The
    # right wrist stays near the tabletop while the inactive left wrist/hand
    # is raised to z=1.51..1.57 m.  The old XY-only gate projected those high
    # left-hand points onto the table and rejected every large-block lane.
    production_points = {
        "right": (
            np.asarray([-0.461500, -0.239500, 0.997500]),
            np.asarray([-0.361620, -0.281347, 0.955653]),
            np.asarray([-0.371500, -0.239500, 0.997500]),
            np.asarray([-0.312533, -0.145520, 0.979042]),
            np.asarray([-0.203290, -0.200211, 1.009372]),
            np.asarray([-0.222123, -0.289761, 1.006832]),
        ),
        "left": (
            np.asarray([-0.372430, 0.230380, 1.547035]),
            np.asarray([-0.357752, 0.276228, 1.511224]),
            np.asarray([-0.322929, 0.134213, 1.516184]),
            np.asarray([-0.208218, 0.203299, 1.565828]),
        ),
    }

    for seed in range(100):
        env = HandsOnABCIsaacLabEnv(
            "t44_stack_blocks",
            launch_isaac=False,
            image_obs=False,
            randomize=True,
            seed=seed,
        )
        env.delta_h[:] = 0.02 * seed / 99.0
        env._reset_robot_body_points = lambda side: production_points[side]

        layout = env._task_specific_reset_xy_positions()
        hand_points = production_points["right"] + production_points["left"]
        assert layout is not None
        assert (
            env._task_specific_reset_layout_margin(layout, hand_points)
            >= _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK
        )


def test_b04_dense_table_height_obstacles_still_fail_closed() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t44_stack_blocks",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=0,
    )
    blocked_points = tuple(
        np.asarray([x, y, 0.90], dtype=np.float64)
        for x in np.arange(-0.42, -0.10, 0.04)
        for y in np.arange(-0.32, 0.29, 0.04)
    )
    env._reset_robot_body_points = (
        lambda side: blocked_points if side == "right" else ()
    )

    with pytest.raises(RuntimeError, match="完整旋转包络无穿模初始布局"):
        env._task_specific_reset_xy_positions()


def test_b05_reset_uses_true_footprints_and_opposite_staging_lanes() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t03_cover_box",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=505,
    )
    hand_points = (
        *env._reset_robot_body_points("right"),
        *env._reset_robot_body_points("left"),
    )

    for _ in range(64):
        layout = env._task_specific_reset_xy_positions()
        assert layout is not None
        assert layout["Clear_Box"][1] > 0.0
        assert layout["Box_lid"][1] < 0.0
        assert (
            env._b05_pair_footprint_margin(
                layout["Clear_Box"],
                "Clear_Box",
                layout["Box_lid"],
                "Box_lid",
            )
            >= _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK
        )
        assert (
            env._task_specific_reset_layout_margin(layout, hand_points)
            >= _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK
        )

    # The previous 15 cm centre-distance rule accepted this layout even
    # though the shipped 29.2 x 42.9 cm collision footprints overlap deeply.
    assert (
        env._b05_pair_footprint_margin(
            np.asarray([-0.30, 0.10]),
            "Clear_Box",
            np.asarray([-0.15, 0.10]),
            "Box_lid",
        )
        < 0.0
    )
    assert _B05_FOOTPRINT_HALF_EXTENTS["Clear_Box"][1] == pytest.approx(0.2145)
    assert 0.89 - _B05_SUPPORT_LOCAL_MIN_Z["Clear_Box"] == pytest.approx(0.9645)
    assert 0.89 - _B05_SUPPORT_LOCAL_MIN_Z["Box_lid"] == pytest.approx(0.9000)
    assert _B05_RESET_SUPPORT_CLEARANCE == pytest.approx(_B05_CONTACT_OFFSET)
    lid_xy = np.asarray([-0.33, -0.26])
    # The production right-hand home has low link origins around z=0.956 m.
    # They are safely above the 0.91 m lid top and must not force the sampler
    # back to the overlapping authored layout.  A point inside the actual
    # lid volume remains a hard rejection.
    lid_center_z = (
        0.89
        + float(env.delta_h[0])
        + _B05_RESET_SUPPORT_CLEARANCE
        + 0.010
    )
    assert env._b05_point_footprint_margin(
        np.asarray([-0.33, -0.26, lid_center_z + 0.041]),
        lid_xy,
        "Box_lid",
    ) >= _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK
    assert env._b05_point_footprint_margin(
        np.asarray([-0.33, -0.26, lid_center_z]),
        lid_xy,
        "Box_lid",
    ) < 0.0

    deterministic_env = HandsOnABCIsaacLabEnv(
        "t03_cover_box",
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    deterministic_layout = deterministic_env._task_specific_reset_xy_positions()
    assert deterministic_layout is not None
    assert (
        deterministic_env._b05_pair_footprint_margin(
            deterministic_layout["Clear_Box"],
            "Clear_Box",
            deterministic_layout["Box_lid"],
            "Box_lid",
        )
        >= _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK
    )
    deterministic_hand_points = (
        *deterministic_env._reset_robot_body_points("right"),
        *deterministic_env._reset_robot_body_points("left"),
    )
    assert (
        deterministic_env._task_specific_reset_layout_margin(
            deterministic_layout,
            deterministic_hand_points,
        )
        >= _TASK_OBJECT_RESET_MIN_CLEARANCE_SLACK
    )


def _prime_b01_lid_release_hold(env: HandsOnABCIsaacLabEnv) -> None:
    env._b01_hand_lid_clear_steps = env._b01_hand_lid_clear_steps_required()


def test_transfer_coverage_helper_requires_ninety_percent_and_release() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t03_cover_box",
        launch_isaac=False,
        image_obs=False,
    )
    base_z = (
        env._current_table_top_z()
        - _B05_SUPPORT_LOCAL_MIN_Z["Clear_Box"]
        + _B05_RESET_SUPPORT_CLEARANCE
    )
    base = np.asarray([0.0, 0.0, base_z, 1.0, 0.0, 0.0, 0.0])
    lid = np.asarray([0.0, 0.0, base_z + 0.0845, 1.0, 0.0, 0.0, 0.0])
    env._lid_start_pose = np.asarray(
        [-0.40, -0.20, 0.90, 1.0, 0.0, 0.0, 0.0]
    )
    env._lid_was_moved = True
    env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)
    env._a06_hand_points = lambda _side: (
        np.asarray([0.70, 0.70, 1.20], dtype=np.float64),
    )
    env._poses = {"Box_lid": lid.copy(), "Clear_Box": base}

    def success():
        env._last_b05_lid_success_metrics = env._b05_lid_fit_metrics(
            env._poses["Box_lid"], base, start_pose=env._lid_start_pose,
            was_moved=True, lid_name="Box_lid")
        return env._last_b05_lid_success_metrics["b05_success_now"]

    metrics = env._b05_full_coverage_metrics(lid, base)
    assert metrics["coverage_ratio"] == pytest.approx(1.0)
    assert metrics["coverage_threshold"] == _B05_MIN_COVERAGE_RATIO
    assert success()
    assert env._success_steps_required() >= int(np.ceil(0.5 / env._control_dt))

    env._physics_body_velocity = lambda _name: None
    assert not success()
    env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)

    env._a06_hand_points = lambda _side: (lid[:3].copy(),)
    assert not success()
    assert not env._last_b05_lid_success_metrics["b05_hand_released"]
    env._a06_hand_points = lambda _side: (
        np.asarray([0.70, 0.70, 1.20], dtype=np.float64),
    )

    above_ninety = lid.copy()
    above_ninety[0] += 0.028
    env._poses["Box_lid"] = above_ninety
    metrics = env._b05_full_coverage_metrics(above_ninety, base)
    assert metrics["coverage_ratio"] > 0.90
    assert metrics["full_coverage"]
    assert success()

    below_ninety = lid.copy()
    below_ninety[0] += 0.030
    env._poses["Box_lid"] = below_ninety
    metrics = env._b05_full_coverage_metrics(below_ninety, base)
    assert metrics["coverage_ratio"] < 0.90
    assert not metrics["full_coverage"]
    assert not success()


def test_b01_lid_success_requires_inserted_height_and_stability() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t04_cover_pitcher",
        launch_isaac=False,
        image_obs=False,
    )
    base = np.asarray([-0.38, 0.06, 0.90, 1.0, 0.0, 0.0, 0.0])
    start = np.asarray([-0.45, -0.23, 0.914, 1.0, 0.0, 0.0, 0.0])
    seated = np.asarray(
        [
            base[0],
            base[1],
            base[2] + _B01_LID_INSERTED_GOAL_LOCAL_Z,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    env._lid_start_pose = start
    env._physics_body_velocity = lambda name: np.zeros(6, dtype=np.float64)
    _prime_b01_lid_release_hold(env)

    env._poses = {"pitcher_lid": seated.copy(), "019_pitcher_base": base}
    assert env._lid_fit_success()

    env._poses["pitcher_lid"][2] += 0.09
    assert not env._lid_fit_success()

    env._poses["pitcher_lid"] = seated.copy()
    env._physics_body_velocity = lambda name: np.asarray(
        [0.10, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    _prime_b01_lid_release_hold(env)
    assert not env._lid_fit_success()


def test_b01_success_requires_lid_insert_inside_and_hand_clear_hold() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t04_cover_pitcher",
        launch_isaac=False,
        image_obs=False,
    )
    base = np.asarray([-0.38, 0.06, 0.90, 1.0, 0.0, 0.0, 0.0])
    seated = np.asarray(
        [
            base[0],
            base[1],
            base[2] + _B01_LID_INSERTED_GOAL_LOCAL_Z,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    env._lid_start_pose = np.asarray(
        [-0.45, -0.23, 0.914, 1.0, 0.0, 0.0, 0.0]
    )
    env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)

    env._poses = {"pitcher_lid": seated.copy(), "019_pitcher_base": base}
    _prime_b01_lid_release_hold(env)
    assert env._lid_fit_success()

    env._physics_body_velocity = lambda _name: None
    _prime_b01_lid_release_hold(env)
    assert not env._lid_fit_success()
    env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)

    env._a06_hand_points = lambda _side: ()
    _prime_b01_lid_release_hold(env)
    assert not env._lid_fit_success()
    env._a06_hand_points = lambda _side: (
        np.asarray([2.0, 2.0, 2.0]),
    )

    env._b01_release_assist_ever_applied = True
    _prime_b01_lid_release_hold(env)
    assert not env._lid_fit_success()
    assert not env._last_b01_lid_success_metrics["b01_collection_eligible"]
    env._b01_release_assist_ever_applied = False

    env._poses["pitcher_lid"] = seated.copy()
    env._poses["pitcher_lid"][2] += 0.030
    _prime_b01_lid_release_hold(env)
    assert not env._lid_fit_success()

    env._poses["pitcher_lid"] = seated.copy()
    env._poses["pitcher_lid"][0] += 0.006
    _prime_b01_lid_release_hold(env)
    assert env._lid_fit_success()

    env._poses["pitcher_lid"] = seated.copy()
    env._poses["pitcher_lid"][0] += 0.012
    _prime_b01_lid_release_hold(env)
    assert not env._lid_fit_success()

    env._poses["pitcher_lid"] = seated.copy()
    env._right_hand[:] = 1.2
    _prime_b01_lid_release_hold(env)
    assert env._lid_fit_success()

    env._poses["pitcher_lid"] = seated.copy()
    env._poses["pitcher_lid"][2] += 0.008
    _prime_b01_lid_release_hold(env)
    assert not env._lid_fit_success()

    env._poses["pitcher_lid"] = seated.copy()
    env._a06_hand_points = lambda _side: (seated[:3],)
    _prime_b01_lid_release_hold(env)
    assert not env._lid_fit_success()

    env._a06_hand_points = lambda _side: (np.asarray([2.0, 2.0, 2.0]),)
    env._b01_hand_lid_clear_steps = env._b01_hand_lid_clear_steps_required() - 1
    assert env._lid_fit_success()


def test_b01_release_assist_activates_on_open_and_reverses_on_regrasp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSON_B01_RELEASE_ASSIST", "1")
    env = HandsOnABCIsaacLabEnv(
        "t04_cover_pitcher",
        launch_isaac=False,
        image_obs=False,
    )
    base = np.asarray([-0.38, 0.06, 0.90, 1.0, 0.0, 0.0, 0.0])
    env._poses = {
        "pitcher_lid": np.asarray(
            [
                base[0],
                base[1],
                base[2] + _B01_LID_INSERTED_GOAL_LOCAL_Z,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        "019_pitcher_base": base,
    }
    filter_events: list[bool] = []

    def set_release_filter(enabled: bool) -> bool:
        filter_events.append(enabled)
        env._b01_release_assist_active = enabled
        return True

    env._set_b01_release_assist = set_release_filter

    env._right_hand[:] = 1.2
    env._update_b01_release_assist()
    assert env._b01_hand_was_closed
    assert filter_events == []

    env._right_hand[:] = 0.5
    env._update_b01_release_assist()
    assert env._b01_release_assist_active
    assert filter_events == [True]

    env._right_hand[:] = 1.2
    env._update_b01_release_assist()
    assert not env._b01_release_assist_active
    assert filter_events == [True, False]


def test_b01_release_assist_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HANDSON_B01_RELEASE_ASSIST", raising=False)
    env = HandsOnABCIsaacLabEnv(
        "t04_cover_pitcher",
        launch_isaac=False,
        image_obs=False,
    )
    events: list[bool] = []
    env._set_b01_release_assist = lambda enabled: events.append(enabled) or True
    env._right_hand[:] = 1.2

    env._update_b01_release_assist()

    assert not env._b01_release_assist_enabled()
    assert events == []


def _c01_interface_hit_env(
    contact_force: float,
    *,
    depth_samples: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (-0.007, 0.0),
    ),
) -> HandsOnABCIsaacLabEnv:
    env = HandsOnABCIsaacLabEnv(
        "t08_hammer_nail",
        launch_isaac=False,
        image_obs=False,
    )
    depths = iter(depth_samples)
    env._physics_dof_position = lambda name: next(depths)
    env._08_hammer_nail_contact_force = lambda: (contact_force, True)

    nail_pose = np.asarray(
        [-0.35, 0.20, 1.04, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    hammer_quat = np.asarray(
        [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0],
        dtype=np.float64,
    )
    hammer_rotation = _pose_vector_to_matrix(
        np.concatenate([np.zeros(3), hammer_quat])
    )[:3, :3]
    nail_top = nail_pose[:3] + _C01_NAIL_HEAD_LOCAL_TOP
    hammer_position = (
        nail_top
        - hammer_rotation @ _C01_HAMMER_FACE_LOCAL_TIP
        + np.asarray([0.0, 0.0, 0.004])
    )
    env._poses = {
        "hammer": np.concatenate([hammer_position, hammer_quat]),
        "nail": nail_pose,
        "wood": np.asarray([-0.35, 0.20, 0.95, 1.0, 0.0, 0.0, 0.0]),
    }
    env._current_table_top_z = lambda: 0.89
    return env


def _move_c01_hammer_down_at_hit_speed(
    env: HandsOnABCIsaacLabEnv,
    speed_multiplier: float = 1.2,
) -> None:
    env._poses["hammer"][2] -= (
        speed_multiplier * _C01_MIN_HAMMER_SPEED * env._sim_dt
    )


def test_c01_recognizes_a_contact_backed_hammer_face_strike() -> None:
    env = _c01_interface_hit_env(contact_force=0.5)

    assert not env._08_hammer_nail_success()
    _move_c01_hammer_down_at_hit_speed(env)
    assert env._08_hammer_nail_success()
    assert env._c01_had_valid_hit
    assert env._c01_hit_contact_available
    assert env._c01_max_valid_hit_depth >= 0.007


def test_c01_rejects_success_after_any_state_assist_was_applied() -> None:
    env = _c01_interface_hit_env(contact_force=0.5)
    env._c01_assist_ever_applied = True

    assert not env._08_hammer_nail_success()
    _move_c01_hammer_down_at_hit_speed(env)

    assert not env._08_hammer_nail_success()
    assert env._c01_had_valid_hit
    assert env._task_info()["c01_collection_eligible"] is False


def test_c01_uses_real_nail_travel_when_contact_tensor_is_silent() -> None:
    env = _c01_interface_hit_env(contact_force=0.0)

    assert not env._08_hammer_nail_success()
    _move_c01_hammer_down_at_hit_speed(env)
    assert env._08_hammer_nail_success()
    assert env._c01_had_valid_hit
    assert env._c01_hit_contact_available


def test_c01_accumulates_valid_strike_depth_across_substeps() -> None:
    env = _c01_interface_hit_env(
        contact_force=0.0,
        depth_samples=(
            (0.0, 0.0),
            (-0.00010, 0.0),
            (-0.00020, 0.0),
            (-0.00030, 0.0),
        ),
    )

    assert not env._08_hammer_nail_success()
    _move_c01_hammer_down_at_hit_speed(env)
    assert not env._08_hammer_nail_success()
    assert env._c01_had_valid_hit
    _move_c01_hammer_down_at_hit_speed(env)
    assert not env._08_hammer_nail_success()
    assert env._c01_had_valid_hit

    _move_c01_hammer_down_at_hit_speed(env)
    assert not env._08_hammer_nail_success()
    assert env._c01_had_valid_hit
    info = env._task_info()
    assert info["c01_candidate_hit_depth_increment"] >= 0.00025


def test_c01_latches_face_strike_without_actual_nail_travel() -> None:
    env = _c01_interface_hit_env(
        contact_force=0.5,
        depth_samples=((0.0, 0.0), (0.0, 0.0)),
    )

    assert not env._08_hammer_nail_success()
    _move_c01_hammer_down_at_hit_speed(env)
    assert not env._08_hammer_nail_success()
    assert env._c01_had_valid_hit
    info = env._task_info()
    assert info["c01_valid_hit_depth_increment"] == 0.0


def test_c01_rejects_nail_travel_without_a_downward_hammer_strike() -> None:
    env = _c01_interface_hit_env(contact_force=0.0)

    assert not env._08_hammer_nail_success()
    assert not env._08_hammer_nail_success()
    assert not env._c01_had_valid_hit


def test_c01_rejects_passive_nail_travel_without_hammer_motion() -> None:
    env = _c01_interface_hit_env(
        contact_force=0.0,
        depth_samples=(
            (0.0, 0.0),
            (-0.00015, 0.0),
            (-0.00030, 0.0),
        ),
    )

    assert not env._08_hammer_nail_success()
    assert not env._08_hammer_nail_success()
    assert not env._c01_had_valid_hit
    assert not env._08_hammer_nail_success()
    assert not env._c01_had_valid_hit
    info = env._task_info()
    assert info["c01_candidate_hit_depth_increment"] == 0.0


def test_c01_rejects_a_slow_downward_touch() -> None:
    env = _c01_interface_hit_env(contact_force=0.5)

    assert not env._08_hammer_nail_success()
    env._poses["hammer"][2] -= (
        0.5 * _C01_MIN_HAMMER_SPEED * env._sim_dt
    )

    assert not env._08_hammer_nail_success()
    assert not env._c01_had_valid_hit
    assert env._c01_last_hammer_downward_speed < _C01_MIN_HAMMER_SPEED


def test_c01_rejects_non_face_hammer_contact() -> None:
    env = _c01_interface_hit_env(contact_force=0.5)
    env._poses["hammer"][0] += 0.10

    assert not env._08_hammer_nail_success()
    env._poses["hammer"][2] -= 0.004
    assert not env._08_hammer_nail_success()
    assert not env._c01_had_valid_hit


def test_c01_rejects_left_and_right_sweeps_across_nail_head() -> None:
    for lateral_direction in (-1.0, 1.0):
        env = _c01_interface_hit_env(contact_force=0.5)

        assert not env._08_hammer_nail_success()
        env._poses["hammer"][0] += lateral_direction * 0.004
        env._poses["hammer"][2] -= 0.001

        assert not env._08_hammer_nail_success()
        assert not env._c01_had_valid_hit
        info = env._task_info()
        assert (
            info["c01_hammer_motion_down_alignment"]
            < info["c01_hammer_motion_down_alignment_threshold"]
        )


def test_c01_rejects_vertical_motion_that_misses_nail_head_center() -> None:
    for lateral_direction in (-1.0, 1.0):
        env = _c01_interface_hit_env(contact_force=0.5)
        env._poses["hammer"][0] += lateral_direction * 0.014

        assert not env._08_hammer_nail_success()
        env._poses["hammer"][2] -= 0.004

        assert not env._08_hammer_nail_success()
        assert not env._c01_had_valid_hit
        info = env._task_info()
        assert (
            info["c01_face_nail_radial_error"]
            > info["c01_face_nail_radial_error_threshold"]
        )


def _run_c01_strike_align_action(
    env: HandsOnABCIsaacLabEnv,
    raw_pose: np.ndarray,
    previous_pose: np.ndarray,
    *,
    hand_command: float = 0.90,
) -> None:
    env._c01_strike_align_prev_raw_tcp = _pose_vector_to_matrix(previous_pose)
    hand = np.full(env._hand_action_dim, hand_command, dtype=np.float64)
    env._apply_action({"right": np.concatenate([raw_pose, hand])})


def test_c01_strike_align_assist_nudges_downward_hit_toward_nail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSON_C01_STRIKE_ALIGN_ASSIST", "1")
    env = _c01_interface_hit_env(contact_force=0.0)
    radial_offset = 0.024
    env._poses["hammer"][0] += radial_offset
    raw_pose = env._poses["hammer"].copy()
    raw_pose[2] += 0.030
    previous_pose = raw_pose.copy()
    previous_pose[2] += (
        1.5 * _C01_STRIKE_ALIGN_MIN_RAW_DOWNWARD_SPEED * env._control_dt
    )

    _run_c01_strike_align_action(env, raw_pose, previous_pose)

    info = env._task_info()
    assert info["c01_strike_align_assist_active"] is True
    assert info["c01_strike_align_assist_reason"] == "active"
    assert info["c01_strike_align_radial_error"] == pytest.approx(radial_offset)
    assert env._right_tcp_pose[0] < raw_pose[0]
    assert raw_pose[0] - env._right_tcp_pose[0] <= (
        _C01_STRIKE_ALIGN_MAX_LINEAR_SPEED * env._control_dt + 1e-10
    )
    np.testing.assert_allclose(env._right_tcp_pose[1:], raw_pose[1:])
    assert info["c01_assist_ever_applied"] is True
    assert info["c01_collection_eligible"] is False


def test_c01_strike_align_assist_ignores_upward_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSON_C01_STRIKE_ALIGN_ASSIST", "1")
    env = _c01_interface_hit_env(contact_force=0.0)
    env._poses["hammer"][0] += 0.020
    raw_pose = env._poses["hammer"].copy()
    raw_pose[2] += 0.030
    previous_pose = raw_pose.copy()
    previous_pose[2] -= (
        1.5 * _C01_STRIKE_ALIGN_MIN_RAW_DOWNWARD_SPEED * env._control_dt
    )

    _run_c01_strike_align_action(env, raw_pose, previous_pose)

    info = env._task_info()
    assert info["c01_strike_align_assist_active"] is False
    assert info["c01_strike_align_assist_reason"] == "not_downward_strike"
    np.testing.assert_allclose(env._right_tcp_pose, raw_pose)


def test_c01_strike_align_assist_does_not_capture_far_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSON_C01_STRIKE_ALIGN_ASSIST", "1")
    env = _c01_interface_hit_env(contact_force=0.0)
    env._poses["hammer"][0] += _C01_STRIKE_ALIGN_CAPTURE_RADIAL_ERROR + 0.010
    raw_pose = env._poses["hammer"].copy()
    raw_pose[2] += 0.030
    previous_pose = raw_pose.copy()
    previous_pose[2] += (
        1.5 * _C01_STRIKE_ALIGN_MIN_RAW_DOWNWARD_SPEED * env._control_dt
    )

    _run_c01_strike_align_action(env, raw_pose, previous_pose)

    info = env._task_info()
    assert info["c01_strike_align_assist_active"] is False
    assert info["c01_strike_align_assist_reason"] == "radial_error_out_of_range"
    np.testing.assert_allclose(env._right_tcp_pose, raw_pose)


def test_c01_state_assists_are_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HANDSON_C01_GRASP_ASSIST", raising=False)
    monkeypatch.delenv("HANDSON_C01_STRIKE_ALIGN_ASSIST", raising=False)
    env = _c01_interface_hit_env(contact_force=0.0)
    raw_pose = env._poses["hammer"].copy()
    previous_pose = raw_pose.copy()
    previous_pose[2] += (
        1.5 * _C01_STRIKE_ALIGN_MIN_RAW_DOWNWARD_SPEED * env._control_dt
    )

    _run_c01_strike_align_action(env, raw_pose, previous_pose)
    env._update_c01_grasp_assists()

    info = env._task_info()
    assert info["c01_strike_align_assist_enabled"] is False
    assert info["c01_strike_align_assist_reason"] == "disabled"
    assert info["c01_grasp_assist_enabled"] is False
    assert info["c01_assist_ever_applied"] is False
    assert info["c01_collection_eligible"] is True
    assert env._c01_grasp_assists == {}
    np.testing.assert_allclose(env._right_tcp_pose, raw_pose)


def _world_point(matrix: np.ndarray, local: np.ndarray) -> np.ndarray:
    return (
        np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        @ np.asarray([*np.asarray(local, dtype=np.float64).reshape(3), 1.0])
    )[:3]


def _install_c01_grasp_test_hooks(
    env: HandsOnABCIsaacLabEnv,
    palm_by_side: dict[str, np.ndarray],
    tips_by_side: dict[str, dict[str, np.ndarray]],
) -> list[tuple[str, np.ndarray]]:
    writes: list[tuple[str, np.ndarray]] = []

    def link_pose(side: str, body_fragment: str = "palm_link") -> np.ndarray | None:
        del body_fragment
        return palm_by_side.get(side)

    def set_target(
        name: str,
        target: np.ndarray,
        *,
        previous_target: np.ndarray | None = None,
    ) -> tuple[bool, float, float]:
        del previous_target
        matrix = np.asarray(target, dtype=np.float64).reshape(4, 4).copy()
        writes.append((name, matrix))
        env._poses[name] = env._pose_matrix_to_vector(matrix)
        return True, 0.0, 0.0

    env._robot_link_pose_matrix = link_pose
    env._wuji_fingertip_positions = lambda side: tips_by_side.get(side, {})
    env._set_body_rigid_grasp_target = set_target
    return writes


def _advance_c01_grasp_debounce(
    env: HandsOnABCIsaacLabEnv,
    steps_extra: int = 2,
) -> None:
    steps = int(
        np.ceil(
            _C01_GRASP_ATTACH_DEBOUNCE_SECONDS
            / max(float(env._sim_dt), 1e-6)
        )
    ) + int(steps_extra)
    for _ in range(steps):
        env._update_c01_grasp_assists()


def test_c01_hammer_grasp_assist_attaches_and_carries_closed_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSON_C01_GRASP_ASSIST", "1")
    env = _c01_interface_hit_env(contact_force=0.0)
    hammer = _pose_vector_to_matrix(env._poses["hammer"])
    palm_by_side = {"right": hammer.copy()}
    handle_center = 0.5 * (
        _C01_HAMMER_GRIP_SEGMENT[0] + _C01_HAMMER_GRIP_SEGMENT[1]
    )
    tips_by_side = {
        "right": {
            "right_finger1_tip": _world_point(
                hammer,
                handle_center + np.asarray([0.0, _C01_HAMMER_HANDLE_RADIUS, 0.0]),
            ),
            "right_finger2_tip": _world_point(
                hammer,
                handle_center - np.asarray([0.0, _C01_HAMMER_HANDLE_RADIUS, 0.0]),
            ),
            "right_finger3_tip": _world_point(
                hammer,
                handle_center
                - np.asarray([0.0, _C01_HAMMER_HANDLE_RADIUS * 0.95, 0.012]),
            ),
        }
    }
    writes = _install_c01_grasp_test_hooks(env, palm_by_side, tips_by_side)

    env._c_assist_raw_right_hand = np.full(
        env._hand_action_dim,
        _C01_GRASP_ATTACH_HAND_COMMAND - 0.01,
    )
    env._update_c01_grasp_assists()
    assert "right" not in env._c01_grasp_assists

    env._c_assist_raw_right_hand = np.full(
        env._hand_action_dim,
        _C01_GRASP_ATTACH_HAND_COMMAND + 0.01,
    )
    env._update_c01_grasp_assists()
    assert "right" not in env._c01_grasp_assists
    info = env._task_info()
    assert info["c01_hammer_grasp_assist_pending"] is True
    assert info["c01_hammer_grasp_assist_candidate_elapsed_s"] > 0.0

    _advance_c01_grasp_debounce(env)
    assert env._c01_grasp_assists["right"]["object_name"] == "hammer"
    assert writes[-1][0] == "hammer"

    moved_palm = hammer.copy()
    moved_palm[:3, 3] += np.asarray([0.020, -0.010, 0.015])
    palm_by_side["right"] = moved_palm
    env._update_c01_grasp_assists()
    expected = moved_palm @ env._c01_grasp_assists["right"]["hand_to_object"]
    np.testing.assert_allclose(writes[-1][1], expected)
    info = env._task_info()
    assert info["c01_hammer_grasp_assist_active"] is True
    assert info["c01_hammer_grasp_assist_applied"] is True

    env._c_assist_raw_right_hand = np.full(
        env._hand_action_dim,
        _C01_GRASP_RELEASE_HAND_COMMAND - 0.01,
    )
    env._update_c01_grasp_assists()
    assert "right" not in env._c01_grasp_assists


def test_c01_hammer_grasp_assist_rejects_adjustment_contacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSON_C01_GRASP_ASSIST", "1")
    env = _c01_interface_hit_env(contact_force=0.0)
    hammer = _pose_vector_to_matrix(env._poses["hammer"])
    palm_by_side = {"right": hammer.copy()}
    handle_center = 0.5 * (
        _C01_HAMMER_GRIP_SEGMENT[0] + _C01_HAMMER_GRIP_SEGMENT[1]
    )
    tips_by_side = {
        "right": {
            "right_finger1_tip": _world_point(
                hammer,
                handle_center + np.asarray([0.0, _C01_HAMMER_HANDLE_RADIUS, 0.0]),
            ),
            "right_finger2_tip": _world_point(
                hammer,
                handle_center - np.asarray([0.0, _C01_HAMMER_HANDLE_RADIUS, 0.0]),
            ),
        }
    }
    _install_c01_grasp_test_hooks(env, palm_by_side, tips_by_side)
    env._c_assist_raw_right_hand = np.full(
        env._hand_action_dim,
        _C01_GRASP_ATTACH_HAND_COMMAND + 0.01,
    )

    _advance_c01_grasp_debounce(env)

    assert "right" not in env._c01_grasp_assists
    assert "right" not in env._c01_grasp_candidates

    upper_handle = np.asarray([0.0028, 0.0, 0.105], dtype=np.float64)
    tips_by_side["right"] = {
        "right_finger1_tip": _world_point(
            hammer,
            upper_handle + np.asarray([0.0, _C01_HAMMER_HANDLE_RADIUS, 0.0]),
        ),
        "right_finger2_tip": _world_point(
            hammer,
            upper_handle - np.asarray([0.0, _C01_HAMMER_HANDLE_RADIUS, 0.0]),
        ),
        "right_finger3_tip": _world_point(
            hammer,
            upper_handle
            - np.asarray([0.0, _C01_HAMMER_HANDLE_RADIUS * 0.95, 0.012]),
        ),
    }

    _advance_c01_grasp_debounce(env)

    assert "right" not in env._c01_grasp_assists
    assert "right" not in env._c01_grasp_candidates


def test_c01_wood_grasp_assist_attaches_left_hand_to_wood_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSON_C01_GRASP_ASSIST", "1")
    env = _c01_interface_hit_env(contact_force=0.0)
    wood = _pose_vector_to_matrix(env._poses["wood"])
    palm_by_side = {"left": wood.copy()}
    y_half = float(_C01_WOOD_GRASP_BOX_HALF_EXTENTS[1])
    tips_by_side = {
        "left": {
            "left_finger1_tip": _world_point(
                wood,
                np.asarray([0.0, y_half, 0.0]),
            ),
            "left_finger2_tip": _world_point(
                wood,
                np.asarray([0.0, -y_half, 0.0]),
            ),
        }
    }
    writes = _install_c01_grasp_test_hooks(env, palm_by_side, tips_by_side)
    env._c_assist_raw_left_hand = np.full(
        env._hand_action_dim,
        _C01_GRASP_ATTACH_HAND_COMMAND + 0.01,
    )

    _advance_c01_grasp_debounce(env)

    assert env._c01_grasp_assists["left"]["object_name"] == "wood"
    assert writes[-1][0] == "wood"
    moved_palm = wood.copy()
    moved_palm[:3, 3] += np.asarray([-0.012, 0.018, 0.020])
    palm_by_side["left"] = moved_palm
    env._update_c01_grasp_assists()
    expected = moved_palm @ env._c01_grasp_assists["left"]["hand_to_object"]
    np.testing.assert_allclose(writes[-1][1], expected)
    info = env._task_info()
    assert info["c01_wood_grasp_assist_active"] is True
    assert info["c01_wood_grasp_assist_applied"] is True


def test_c01_reset_colliders_start_exactly_on_the_table_support() -> None:
    assert 0.89 - _C01_HAMMER_SUPPORT_LOCAL_MIN_Z == pytest.approx(
        0.9036390064
    )
    np.testing.assert_allclose(
        _C01_HAMMER_FLAT_RESET_QUAT_XYZW,
        np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
    )
    assert 0.89 - _C01_WOOD_SUPPORT_LOCAL_MIN_Z == pytest.approx(0.9125)
    expected_inertia = (_C01_WOOD_MASS / 3.0) * np.asarray(
        [
            0.035**2 + 0.0225**2,
            0.050**2 + 0.0225**2,
            0.050**2 + 0.035**2,
        ]
    )
    assert _C01_WOOD_CENTER_OF_MASS == (0.0, 0.0, 0.0)
    assert np.asarray(_C01_WOOD_DIAGONAL_INERTIA) == pytest.approx(
        expected_inertia
    )
    assert _C01_WOOD_LINEAR_DAMPING == pytest.approx(0.12)
    assert _C01_WOOD_ANGULAR_DAMPING == pytest.approx(0.80)
    assert _C01_WOOD_SLEEP_THRESHOLD == pytest.approx(0.005)
    assert _C01_HAMMER_LINEAR_DAMPING == pytest.approx(0.12)
    assert _C01_HAMMER_ANGULAR_DAMPING == pytest.approx(0.80)
    assert _C01_HAMMER_SLEEP_THRESHOLD == pytest.approx(0.005)


def test_c02_support_heights_and_golf_ball_collider_are_exact() -> None:
    assert 0.89 - _C02_SUPPORT_LOCAL_MIN_Z["golf_ball"] == pytest.approx(
        0.911659
    )
    assert 0.89 - _C02_SUPPORT_LOCAL_MIN_Z["024_bowl"] == pytest.approx(0.891)
    assert (
        0.89
        - _C02_SUPPORT_LOCAL_MIN_Z["spoon"]
        + _C02_RESET_SUPPORT_CLEARANCE
    ) == pytest.approx(0.9036368806287686)
    assert _C02_SPOON_REST_ANGLE_RAD == pytest.approx(0.06481273405400481)
    assert _C02_SPOON_REST_QUAT_XYZW == pytest.approx(
        [
            np.sin(_C02_SPOON_REST_ANGLE_RAD / 2.0),
            0.0,
            0.0,
            np.cos(_C02_SPOON_REST_ANGLE_RAD / 2.0),
        ]
    )

    xml_path = (
        _ISAACLAB_SOURCE_ROOT.parent
        / "envs"
        / "xmls"
        / "a_series_ycb_task_objects.xml"
    )
    root = ET.parse(xml_path).getroot()
    collider = root.find(".//geom[@name='golf_ball_collision_1']")
    assert collider is not None
    assert collider.attrib["type"] == "sphere"
    assert float(collider.attrib["size"]) == pytest.approx(0.021659)
    assert "mesh" not in collider.attrib


def test_b05_lid_uses_one_visual_hull_support_collider() -> None:
    xml_path = (
        _ISAACLAB_SOURCE_ROOT.parent
        / "envs"
        / "xmls"
        / "a_series_ycb_task_objects.xml"
    )
    root = ET.parse(xml_path).getroot()
    lid = root.find(".//body[@name='Box_lid']")
    assert lid is not None

    colliders = [
        geom
        for geom in lid.findall("geom")
        if geom.attrib.get("class") == "a_series_ycb_collision"
    ]
    assert [geom.attrib.get("name") for geom in colliders] == [
        "Box_lid_collision_1"
    ]
    collider = colliders[0]
    visual = lid.find("geom[@name='Box_lid_visual']")
    assert visual is not None
    assert collider.attrib["mesh"] == visual.attrib["mesh"] == "Box_lid_vis"
    assert "type" not in collider.attrib
    assert "size" not in collider.attrib
    assert float(collider.attrib["mass"]) == pytest.approx(
        _LID_GRASP_PHYSICS["t03_cover_box"].mass
    )
    visual_mesh_path = (
        xml_path.parent
        / "lightwheel_ycb"
        / "Box_lid"
        / "visuals"
        / "Box_lid.obj"
    )
    visual_vertices = np.asarray(
        [
            [float(value) for value in line.split()[1:4]]
            for line in visual_mesh_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("v ")
        ],
        dtype=np.float64,
    )
    assert visual_vertices.shape[1] == 3
    assert float(visual_vertices[:, 2].min()) == pytest.approx(
        _B05_SUPPORT_LOCAL_MIN_Z["Box_lid"]
    )
    assert lid.find("geom[@name='Box_lid_table_guard']") is None
    assert "quat" not in lid.attrib


def test_b05_clear_box_uses_exact_analytic_bottom_and_keeps_walls() -> None:
    xml_path = (
        _ISAACLAB_SOURCE_ROOT.parent
        / "envs"
        / "xmls"
        / "a_series_ycb_task_objects.xml"
    )
    root = ET.parse(xml_path).getroot()
    clear_box = root.find(".//body[@name='Clear_Box']")
    assert clear_box is not None

    colliders = {
        geom.attrib["name"]: geom
        for geom in clear_box.findall("geom")
        if geom.attrib.get("class") == "a_series_ycb_collision"
    }
    assert set(colliders) == {
        f"Clear_Box_collision_{index}" for index in range(1, 10)
    }
    bottom = colliders["Clear_Box_collision_2"]
    assert bottom.attrib["type"] == "box"
    assert "mesh" not in bottom.attrib
    center = np.fromstring(bottom.attrib["pos"], sep=" ")
    half_extents = np.fromstring(bottom.attrib["size"], sep=" ")
    np.testing.assert_allclose(
        center - half_extents,
        [-0.100795, -0.152635, -0.074500],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        center + half_extents,
        [0.100795, 0.152667, -0.071314],
        rtol=0.0,
        atol=1e-12,
    )
    assert float(bottom.attrib["mass"]) == pytest.approx(0.01)

    wall_colliders = {
        name: collider
        for name, collider in colliders.items()
        if name != "Clear_Box_collision_2"
    }
    assert len(wall_colliders) == 8
    assert all("mesh" in collider.attrib for collider in wall_colliders.values())
    assert sum(float(collider.attrib["mass"]) for collider in colliders.values()) == (
        pytest.approx(0.09)
    )


def test_b05_runtime_authors_one_contact_shell_on_all_clear_box_colliders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAttribute:
        def __init__(self) -> None:
            self.value = None

        def Set(self, value) -> None:
            self.value = value

    class FakeCollisionSchema:
        def __init__(self, prim):
            self.prim = prim
        def GetCollisionEnabledAttr(self):
            return types.SimpleNamespace(Get=lambda: self.prim.collision)

    class FakePrim:
        def __init__(self, name: str, *, collision: bool = False) -> None:
            self.name = name
            self.path = f"/World/Task/{name}"
            self.collision = collision
            self.contact_api = None
            self.mass_api = None
            self.descendants = []

        def GetPath(self) -> str:
            return self.path

        def HasAPI(self, api) -> bool:
            return api is FakeCollisionSchema and self.collision

    class FakeCollisionAPI:
        def __init__(self, prim: FakePrim) -> None:
            self.contact_offset = FakeAttribute()
            self.rest_offset = FakeAttribute()
            prim.contact_api = self

        @classmethod
        def Apply(cls, prim: FakePrim):
            return cls(prim)

        def CreateContactOffsetAttr(self) -> FakeAttribute:
            return self.contact_offset

        def CreateRestOffsetAttr(self) -> FakeAttribute:
            return self.rest_offset

    class FakeMassAPI:
        def __init__(self, prim: FakePrim) -> None:
            self.attributes = {}
            prim.mass_api = self

        @classmethod
        def Apply(cls, prim: FakePrim):
            return cls(prim)

        def _attribute(self, name: str) -> FakeAttribute:
            return self.attributes.setdefault(name, FakeAttribute())

        def CreateMassAttr(self) -> FakeAttribute:
            return self._attribute("mass")

        def CreateCenterOfMassAttr(self) -> FakeAttribute:
            return self._attribute("center_of_mass")

        def CreatePrincipalAxesAttr(self) -> FakeAttribute:
            return self._attribute("principal_axes")

        def CreateDiagonalInertiaAttr(self) -> FakeAttribute:
            return self._attribute("diagonal_inertia")

    clear_box = FakePrim("Clear_Box")
    colliders = [
        FakePrim(f"Clear_Box_collision_{index}", collision=True)
        for index in range(1, 10)
    ]
    visual = FakePrim("Clear_Box_visual")
    clear_box.descendants = [visual, *colliders]

    class FakeStage:
        def GetPrimAtPath(self, path: str) -> FakePrim:
            assert path == clear_box.path
            return clear_box

    pxr = types.ModuleType("pxr")
    pxr.Gf = types.SimpleNamespace(
        Vec3f=lambda *values: tuple(values),
        Quatf=lambda real, imaginary: (real, *imaginary),
    )
    pxr.PhysxSchema = types.SimpleNamespace(
        PhysxCollisionAPI=FakeCollisionAPI,
    )
    pxr.Usd = types.SimpleNamespace(PrimRange=lambda prim: prim.descendants)
    pxr.UsdPhysics = types.SimpleNamespace(
        CollisionAPI=FakeCollisionSchema,
        MassAPI=FakeMassAPI,
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    class FakeSimUtils:
        def __init__(self) -> None:
            self.uninstanced = []
            self.stage_updates = 0

        def make_uninstanceable(self, path: str, *, stage) -> None:
            assert stage is env._stage
            self.uninstanced.append(path)

        def update_stage(self) -> None:
            self.stage_updates += 1

    env = HandsOnABCIsaacLabEnv(
        "t03_cover_box",
        launch_isaac=False,
        image_obs=False,
    )
    env._stage = FakeStage()
    env._find_rigid_body_prim = lambda name: clear_box if name == "Clear_Box" else None
    sim_utils = FakeSimUtils()

    assert env._configure_b05_clear_box_contacts(sim_utils) == 9
    assert sim_utils.uninstanced == [clear_box.path]
    assert sim_utils.stage_updates == 1
    assert visual.contact_api is None
    mass_attributes = clear_box.mass_api.attributes
    assert mass_attributes["mass"].value == pytest.approx(_B05_CLEAR_BOX_MASS)
    np.testing.assert_allclose(
        mass_attributes["center_of_mass"].value,
        _B05_CLEAR_BOX_CENTER_OF_MASS,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        mass_attributes["principal_axes"].value,
        _B05_CLEAR_BOX_PRINCIPAL_AXES_WXYZ,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        mass_attributes["diagonal_inertia"].value,
        _B05_CLEAR_BOX_DIAGONAL_INERTIA,
        rtol=0.0,
        atol=0.0,
    )
    for collider in colliders:
        assert collider.contact_api.contact_offset.value == pytest.approx(
            _B05_CONTACT_OFFSET
        )
        assert collider.contact_api.rest_offset.value == pytest.approx(
            _B05_REST_OFFSET
        )

    env.task_name = "b04_build_house"
    assert env._configure_b05_clear_box_contacts(sim_utils) == 0
    env.task_name = "t03_cover_box"

    clear_box.descendants = [visual, *colliders[:-1]]
    with pytest.raises(RuntimeError, match="expected exactly 9 colliders, found 8"):
        env._configure_b05_clear_box_contacts(sim_utils)
    assert any(
        "expected exactly 9 colliders, found 8" in warning
        for warning in env._isaac_warnings
    )


def test_c01_nail_has_one_weight_sized_passive_hold() -> None:
    # A 15 g nail weighs about 0.147 N.  PhysX joint friction is dimensionless,
    # so the sole hold is a zero-target velocity brake capped just above weight.
    nail_weight = _C01_NAIL_MASS * 9.81

    assert _C01_NAIL_MASS == 0.015
    assert _C01_NAIL_JOINT_STIFFNESS == 0.0
    assert _C01_NAIL_JOINT_FRICTION == 0.0
    assert _C01_NAIL_JOINT_DAMPING >= 50000.0
    assert nail_weight <= _C01_NAIL_JOINT_MAX_FORCE <= 1.5 * nail_weight


def test_c01_impacts_use_true_ccd_and_small_contact_shells() -> None:
    assert _C01_ENABLE_CCD
    assert not _C01_ENABLE_SPECULATIVE_CCD

    assert 0.0005 <= _C01_HAMMER_CONTACT_OFFSET <= 0.001
    assert 0.0005 <= _C01_NAIL_CONTACT_OFFSET <= 0.001
    assert _C01_HAMMER_CONTACT_OFFSET + _C01_NAIL_CONTACT_OFFSET <= 0.002

    assert 16 <= _C01_SOLVER_POSITION_ITERATIONS <= 32
    assert 2 <= _C01_SOLVER_VELOCITY_ITERATIONS <= 4
    assert _C01_MAX_DEPENETRATION_VELOCITY <= 1.0


def test_c01_hammer_face_reference_matches_ycb_positive_x_face() -> None:
    np.testing.assert_allclose(
        _C01_HAMMER_FACE_LOCAL_TIP,
        np.asarray([0.059062, 0.0, 0.1730]),
        atol=1e-7,
    )


def test_c01_hammer_guide_projects_face_ray_to_wood_top() -> None:
    env = _c01_interface_hit_env(contact_force=0.5)

    guide = env._c01_hammer_guide_geometry()

    assert guide is not None
    assert guide["intersects_wood_plane"]
    assert guide["on_wood_top"]
    assert guide["aligned"]
    np.testing.assert_allclose(
        guide["impact_world"],
        np.asarray(
            [
                env._poses["wood"][0],
                env._poses["wood"][1],
                env._poses["wood"][2] + _C01_WOOD_HALF_HEIGHT,
            ],
            dtype=np.float64,
        ),
        atol=1e-7,
    )
    assert guide["impact_radial_error"] == pytest.approx(0.0)


def test_c01_hammer_guide_projects_into_head_camera_pixels() -> None:
    env = _c01_interface_hit_env(contact_force=0.5)

    guide = env.c01_hammer_guide_pixels("random_camera", (480, 848, 3))

    assert guide is not None
    assert guide["aligned"]
    for key in ("line_start", "line_end", "impact", "nail_target"):
        pixel = guide[key]
        assert len(pixel) == 2
        assert all(isinstance(value, int) for value in pixel)


def test_c01_contact_pair_uses_strongest_tensor_orientation() -> None:
    class FakeContactView:
        def __init__(self, force: float):
            self.force = force

        def get_contact_force_matrix(self, _dt: float) -> np.ndarray:
            return np.asarray([[[self.force, 0.0, 0.0]]], dtype=np.float64)

    env = HandsOnABCIsaacLabEnv(
        "t08_hammer_nail",
        launch_isaac=False,
        image_obs=False,
    )
    env._08_hammer_nail_contact_views = [
        FakeContactView(0.2),
        FakeContactView(0.5),
    ]

    force, available = env._08_hammer_nail_contact_force()

    assert available
    assert force == 0.5


def test_c01_vectorized_point_polygon_distances_preserve_signed_gate() -> None:
    polygon = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=np.float64,
    )
    points = np.asarray(
        [
            [0.0, 0.0],
            [1.5, 0.0],
            [1.5, 1.5],
            [1.0, 0.25],
        ],
        dtype=np.float64,
    )

    distances = HandsOnABCIsaacLabEnv._c01_points_polygon_distance(
        points,
        polygon,
    )

    np.testing.assert_allclose(
        distances,
        [-1.0, 0.5, np.sqrt(0.5), 0.0],
        atol=1e-12,
    )


def test_c01_reset_backtracks_after_locally_valid_wood_dead_end() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t08_hammer_nail",
        seed=4,
        launch_isaac=False,
        image_obs=False,
        randomize=True,
    )
    sampled_wood: list[tuple[float, float]] = []
    env._c01_wood_candidate_is_valid = lambda *_args: True

    def layout_is_valid(
        _hammer_xy,
        _hammer_yaw,
        wood_xy,
        _hand_points,
        _hand_ndc,
    ) -> bool:
        sampled_wood.append(tuple(np.asarray(wood_xy, dtype=np.float64)))
        return len(set(sampled_wood)) >= 2

    env._c01_reset_layout_is_valid = layout_is_valid

    _yaw, positions = env._c01_reset_pose()

    assert len(set(sampled_wood)) >= 2
    np.testing.assert_allclose(positions["wood"], sampled_wood[-1])


def test_c01_reset_randomizes_visible_hammer_pose_between_reset_hands() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t08_hammer_nail",
        seed=17,
        launch_isaac=False,
        image_obs=False,
        randomize=True,
    )

    hammer_positions = []
    hammer_yaws = []
    hand_points = (
        *env._reset_robot_body_points("right"),
        *env._reset_robot_body_points("left"),
    )
    hand_ndc = env._c01_head_ndc(np.asarray(hand_points))
    for _ in range(100):
        hammer_yaw, positions = env._c01_reset_pose()
        hammer_xy = positions["hammer"]
        wood_xy = positions["wood"]
        hammer_bounds = env._c01_hammer_xy_bounds(hammer_yaw)
        hammer_positions.append(hammer_xy)
        hammer_yaws.append(hammer_yaw)

        assert np.all(hammer_xy >= hammer_bounds[0])
        assert np.all(hammer_xy <= hammer_bounds[1])
        assert -0.44 <= hammer_xy[0] <= 0.02
        assert -0.15 <= hammer_xy[1] <= 0.15

        wood_footprint = env._c01_wood_footprint(wood_xy)
        assert np.all(wood_footprint >= _C01_WOOD_XY_BOUNDS[0])
        assert np.all(wood_footprint <= _C01_WOOD_XY_BOUNDS[1])
        assert np.all(wood_footprint >= RECORDING_OBJECT_XY_BOUNDS[0])
        assert np.all(wood_footprint <= RECORDING_OBJECT_XY_BOUNDS[1])

        hammer_footprint = env._c01_hammer_footprint(hammer_xy, hammer_yaw)
        assert np.min(hammer_footprint[:, 0]) >= _C01_HAMMER_MIN_FOOTPRINT_X
        assert (
            float(np.linalg.norm(hammer_xy - wood_xy))
            >= _C01_HAMMER_WOOD_MIN_DISTANCE
        )
        assert (
            env._c01_convex_polygon_axis_margin(
                hammer_footprint,
                wood_footprint,
            )
            >= _C01_HAMMER_WOOD_COLLISION_CLEARANCE
        )

        hammer_footprint_ndc = env._c01_head_ndc(
            np.column_stack(
                [
                    hammer_footprint,
                    np.full(len(hammer_footprint), _C01_HAMMER_SCREEN_Z),
                ]
            )
        )
        wood_footprint_ndc = env._c01_head_ndc(
            np.column_stack(
                [
                    wood_footprint,
                    np.full(len(wood_footprint), _C01_WOOD_SCREEN_Z),
                ]
            )
        )
        assert (
            env._c01_convex_polygon_axis_margin(
                hammer_footprint_ndc,
                wood_footprint_ndc,
            )
            >= _C01_HAMMER_WOOD_SCREEN_CLEARANCE
        )
        assert np.all(
            np.abs(hammer_footprint_ndc[:, 0]) <= _C01_HAMMER_SCREEN_X_LIMIT
        )
        assert np.all(
            hammer_footprint_ndc[:, 1] >= _C01_HAMMER_SCREEN_Y_BOUNDS[0]
        )
        assert np.all(
            hammer_footprint_ndc[:, 1] <= _C01_HAMMER_SCREEN_Y_BOUNDS[1]
        )
        for point, point_ndc in zip(hand_points, hand_ndc, strict=True):
            assert (
                env._c01_point_polygon_distance(
                    point_ndc,
                    hammer_footprint_ndc,
                )
                >= _C01_HAMMER_HAND_SCREEN_CLEARANCE
            )
            assert (
                env._c01_point_polygon_distance(point[:2], wood_footprint)
                >= _C01_WOOD_HAND_MIN_XY_DISTANCE
            )
        assert env._c01_reset_layout_is_valid(
            hammer_xy,
            hammer_yaw,
            wood_xy,
            hand_points,
            hand_ndc,
        )

    hammer_span = np.ptp(np.asarray(hammer_positions), axis=0)
    assert hammer_span[0] > 0.12
    assert hammer_span[1] > 0.10
    assert min(hammer_yaws) >= -180.0
    assert max(hammer_yaws) <= 180.0
    yaw_quadrants = (
        np.floor((np.asarray(hammer_yaws) + 180.0) / 90.0).astype(int) % 4
    )
    assert set(yaw_quadrants) == {0, 1, 2, 3}


def test_c01_collision_envelopes_and_center_bounds_are_exact() -> None:
    np.testing.assert_allclose(
        _C01_WOOD_FOOTPRINT_HALF_EXTENTS,
        np.asarray([0.050, 0.035]),
    )
    np.testing.assert_allclose(
        _C01_WOOD_CENTER_XY_BOUNDS,
        np.asarray([[-0.25, -0.125], [-0.17, 0.125]]),
    )
    assert _C01_HAMMER_FOOTPRINT_VERTICES_LOCAL_XY.shape == (9, 2)
    np.testing.assert_allclose(
        np.min(_C01_HAMMER_FOOTPRINT_VERTICES_LOCAL_XY, axis=0),
        np.asarray([-0.059062, -0.1903415]),
    )
    np.testing.assert_allclose(
        np.max(_C01_HAMMER_FOOTPRINT_VERTICES_LOCAL_XY, axis=0),
        np.asarray([0.059062, 0.1287415]),
    )

    collision_dir = (
        _ISAACLAB_SOURCE_ROOT.parent
        / "envs"
        / "xmls"
        / "lightwheel_ycb"
        / "hammer"
        / "collision"
    )
    collision_paths = sorted(collision_dir.glob("hammer_Collider*_m.obj"))
    assert len(collision_paths) == 12
    for collision_path in collision_paths:
        for line in collision_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("v "):
                continue
            object_vertex = np.fromstring(line[2:], sep=" ")
            # The MJCF +90-degree X geom rotation followed by the broad-side
            # +90-degree X root reset maps OBJ XY to [x, -y - 0.0308].
            footprint_vertex = np.asarray(
                [object_vertex[0], -object_vertex[1] - 0.0308],
                dtype=np.float64,
            )
            edge_cross_products = []
            for index, start in enumerate(
                _C01_HAMMER_FOOTPRINT_VERTICES_LOCAL_XY
            ):
                end = _C01_HAMMER_FOOTPRINT_VERTICES_LOCAL_XY[
                    (index + 1)
                    % len(_C01_HAMMER_FOOTPRINT_VERTICES_LOCAL_XY)
                ]
                edge = end - start
                offset = footprint_vertex - start
                edge_cross_products.append(
                    edge[0] * offset[1] - edge[1] * offset[0]
                )
            assert min(edge_cross_products) >= -1e-12


def test_c01_deterministic_layout_passes_and_unsafe_fallback_is_rejected() -> None:
    deterministic_env = HandsOnABCIsaacLabEnv(
        "t08_hammer_nail",
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    hand_points = (
        *deterministic_env._reset_robot_body_points("right"),
        *deterministic_env._reset_robot_body_points("left"),
    )
    hand_ndc = deterministic_env._c01_head_ndc(np.asarray(hand_points))
    yaw, positions = deterministic_env._c01_reset_pose()
    assert deterministic_env._c01_reset_layout_is_valid(
        positions["hammer"],
        yaw,
        positions["wood"],
        hand_points,
        hand_ndc,
    )

    # The old emergency poses are outside the published camera/hand envelope.
    with pytest.raises(RuntimeError, match="fallback.*完整碰撞包络"):
        deterministic_env._c01_fallback_reset_pose(hand_points, hand_ndc)


def test_c01_fallback_fails_closed_when_its_envelope_gate_rejects() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t08_hammer_nail",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
    )
    env._c01_reset_layout_is_valid = lambda *args: False
    with pytest.raises(RuntimeError, match="fallback.*完整碰撞包络"):
        env._c01_fallback_reset_pose((), np.empty((0, 2)))


def test_c10_physical_connection_tracking_is_read_only_and_requires_current_fit() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t02_connect_charger_and_cable",
        launch_isaac=False,
        image_obs=False,
    )
    strip_matrix = np.eye(4, dtype=np.float64)
    strip_matrix[:3, 3] = (-0.39, 0.0, 0.893)
    plug_matrix = strip_matrix @ abc_env._C10_PLUG_SEATED_RELATIVE
    cable_matrix = plug_matrix @ abc_env._C10_CABLE_SEATED_RELATIVE
    cable_seated_pose = env._pose_matrix_to_vector(cable_matrix)
    cable_staged_pose = cable_seated_pose.copy()
    cable_staged_pose[0] += 0.20
    poses = {
        "table": np.asarray(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        "c10_power_strip": env._pose_matrix_to_vector(strip_matrix),
        "c10_charger_plug": env._pose_matrix_to_vector(plug_matrix),
        "c10_charging_cable": cable_staged_pose,
    }
    env._physics_body_pose = lambda name: poses[name].copy()
    env._set_body_rigid_grasp_target = lambda *_args, **_kwargs: pytest.fail(
        "physical tracking must not write a rigid root"
    )
    env._set_task_body_world_pose = lambda *_args, **_kwargs: pytest.fail(
        "physical tracking must not teleport a connector"
    )
    env._set_c10_connector_kinematic = lambda *_args, **_kwargs: pytest.fail(
        "physical tracking must not make a connector kinematic"
    )

    for _ in range(abc_env._C10_LATCH_STABLE_STEPS):
        env._update_c10_physical_connection_state()
    poses["c10_charging_cable"] = cable_seated_pose
    for _ in range(abc_env._C10_LATCH_STABLE_STEPS):
        env._update_c10_physical_connection_state()

    metrics = env._c10_connection_metrics()
    assert env._c10_first_connected_seen is True
    assert env._c10_plug_latched is True
    assert env._c10_cable_latched is True
    assert metrics["c10_success_now"] is True

    poses["c10_charging_cable"] = poses["c10_charging_cable"].copy()
    poses["c10_charging_cable"][0] += 0.20
    env._poses["c10_charging_cable"] = poses["c10_charging_cable"].copy()
    assert env._c10_connection_metrics()["c10_success_now"] is False


def test_c04_c07_supports_remove_large_reset_free_fall() -> None:
    assert 0.89 - _C04_SPRAY_SUPPORT_LOCAL_MIN_Z == pytest.approx(0.914)
    assert 0.89 - _C07_PITCHER_SUPPORT_LOCAL_MIN_Z == pytest.approx(1.008)
    assert 0.89 - _C07_CUP_SUPPORT_LOCAL_MIN_Z == pytest.approx(0.931)
    assert _C07_PARTICLE_SUPPORT_ROOT_Z == pytest.approx(0.9025)

    pair_distances = []
    for index, offset_a in enumerate(_C07_PARTICLE_XY_OFFSETS):
        assert np.linalg.norm(offset_a) <= 0.021
        for offset_b in _C07_PARTICLE_XY_OFFSETS[index + 1 :]:
            pair_distances.append(float(np.linalg.norm(offset_a - offset_b)))
    assert min(pair_distances) > 2.0 * _C07_PARTICLE_RADIUS


def _c07_particle_test_target(total_tilt_degrees: float) -> np.ndarray:
    angle = np.deg2rad(total_tilt_degrees)
    target = np.eye(4, dtype=np.float64)
    target[:3, :3] = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(-angle), -np.sin(-angle)],
            [0.0, np.sin(-angle), np.cos(-angle)],
        ],
        dtype=np.float64,
    )
    return target


def _c07_prearmed_particle_state(
    local_transforms: dict[str, np.ndarray],
    target: np.ndarray,
) -> dict:
    return {
        "pour_xy_lock": True,
        "pour_lock_pending": False,
        "particle_local_transforms": local_transforms,
        "particle_last_targets": {
            name: target @ local
            for name, local in local_transforms.items()
        },
        "particle_released_names": set(),
        "particles_released": False,
        "particle_drain_started": True,
        "particle_drain_elapsed_s": (
            _C07_PARTICLE_FIRST_RELEASE_DELAY_SECONDS
        ),
        "particle_stability_target": target.copy(),
        "particle_target_stable_steps": (
            _C07_PARTICLE_RELEASE_STABLE_STEPS
        ),
        "particle_stage_release_cooldown_s": 0.0,
    }


def test_rigid_grasp_writer_prefers_articulation_root_and_keeps_motion() -> None:
    torch = pytest.importorskip("torch")

    class FakeArticulationView:
        def __init__(self) -> None:
            self.transforms = torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=torch.float64,
            )
            self.velocities = torch.zeros((1, 6), dtype=torch.float64)
            self.transform_writes = 0
            self.velocity_writes = 0

        def get_root_transforms(self):
            return self.transforms

        def get_root_velocities(self):
            return self.velocities

        def set_root_transforms(self, values, _indices) -> None:
            self.transforms = values.clone()
            self.transform_writes += 1

        def set_root_velocities(self, values, _indices) -> None:
            self.velocities = values.clone()
            self.velocity_writes += 1

    class ForbiddenRigidView:
        def __getattr__(self, name):
            raise AssertionError(f"rigid writer should not be used: {name}")

    env = HandsOnABCIsaacLabEnv(
        "c08_tongs_ice_cubes",
        launch_isaac=False,
        image_obs=False,
    )
    env._torch = torch
    env._sim_dt = 0.02
    env._physics_body_pose = lambda _name: np.asarray(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    )
    view = FakeArticulationView()
    env._task_articulation_views = {
        "c08_tongs": (view, torch.arange(1, dtype=torch.long))
    }
    env._task_body_views = {"c08_tongs": ForbiddenRigidView()}
    env._physical_recording_origin = {
        "trajectory_per_frame_state_writes": False,
        "per_frame_state_writes": False,
    }
    env._trajectory_action_phase_active = True
    previous = np.eye(4, dtype=np.float64)
    target = np.eye(4, dtype=np.float64)
    angle = np.deg2rad(30.0)
    target[:3, :3] = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target[:3, 3] = [0.02, -0.04, 0.06]

    applied, _linear_error, _angular_error = (
        env._set_body_rigid_grasp_target(
            "c08_tongs",
            target,
            previous_target=previous,
        )
    )

    assert applied is True
    assert env._physical_recording_origin[
        "trajectory_per_frame_state_writes"
    ] is True
    assert env._physical_recording_origin["per_frame_state_writes"] is True
    assert env._physical_recording_origin["trajectory_state_write_count"] == 1
    assert env._physical_recording_origin["trajectory_state_write_journal"] == [
        {
            "write_index": 1,
            "control_step": 0,
            "physics_substep": -1,
            "writer": "_set_body_rigid_grasp_target",
            "subject": "c08_tongs",
        }
    ]
    assert view.transform_writes == 1
    assert view.velocity_writes == 1
    np.testing.assert_allclose(
        view.transforms[0, :3].numpy(),
        target[:3, 3],
    )
    np.testing.assert_allclose(
        view.velocities[0, :3].numpy(),
        target[:3, 3] / env._sim_dt,
    )
    np.testing.assert_allclose(
        view.velocities[0, 3:].numpy(),
        [0.0, 0.0, angle / env._sim_dt],
        atol=1e-10,
    )


def test_action_phase_state_write_journal_is_bounded_and_stable_setup_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "0")
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
    )
    origin = {
        "trajectory_per_frame_state_writes": False,
        "per_frame_state_writes": False,
    }
    env._physical_recording_origin = origin

    # Stable-origin settling bypasses step_with_current_joint_targets and
    # therefore cannot turn a pre-action setup write into a trajectory write.
    env._record_trajectory_state_write("stable_setup")
    assert origin == {
        "trajectory_per_frame_state_writes": False,
        "per_frame_state_writes": False,
    }

    journal_limit = assembly_env._TRAJECTORY_STATE_WRITE_JOURNAL_LIMIT

    def fake_action_interval(*, start_time=None):
        assert start_time == pytest.approx(12.5)
        assert env._trajectory_action_phase_active is True
        for index in range(journal_limit + 3):
            env._trajectory_action_phase_substep_index = index
            env._record_trajectory_state_write(
                "synthetic_writer",
                subject=f"body_{index}",
            )
        return "advanced"

    env._step_with_current_joint_targets_impl = fake_action_interval

    assert env.step_with_current_joint_targets(start_time=12.5) == "advanced"
    assert env._trajectory_action_phase_active is False
    assert env._trajectory_action_phase_substep_index == -1
    assert origin["trajectory_per_frame_state_writes"] is True
    assert origin["per_frame_state_writes"] is True
    assert origin["trajectory_state_write_count"] == journal_limit + 3
    assert len(origin["trajectory_state_write_journal"]) == journal_limit
    assert origin["trajectory_state_write_journal_dropped"] == 3
    assert origin["trajectory_state_write_journal"][0]["physics_substep"] == 0
    assert origin["trajectory_state_write_journal"][-1][
        "physics_substep"
    ] == journal_limit - 1


def test_successful_abc_state_writers_increment_action_phase_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "0")

    class FakeRigidView:
        count = 1

        def __init__(self) -> None:
            self.transforms = torch.tensor(
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=torch.float64,
            )
            self.velocities = torch.zeros((1, 6), dtype=torch.float64)

        def get_transforms(self):
            return self.transforms

        def get_velocities(self):
            return self.velocities

        def set_transforms(self, values, _indices) -> None:
            self.transforms = values.clone()

        def set_velocities(self, values, _indices) -> None:
            self.velocities = values.clone()

    class FakeDofView:
        count = 1

        def __init__(self) -> None:
            self.positions = torch.zeros((1, 1), dtype=torch.float64)
            self.velocities = torch.zeros((1, 1), dtype=torch.float64)

        def get_dof_positions(self):
            return self.positions

        def get_dof_velocities(self):
            return self.velocities

        def set_dof_positions(self, values, _indices) -> None:
            self.positions = values.clone()

        def set_dof_velocities(self, values, _indices) -> None:
            self.velocities = values.clone()

    env = HandsOnABCIsaacLabEnv(
        "c08_tongs_ice_cubes",
        launch_isaac=False,
        image_obs=False,
        isaac_device="cpu",
    )
    env._torch = torch
    env._sim_dt = 0.02
    rigid_view = FakeRigidView()
    dof_view = FakeDofView()
    env._task_body_views = {"nut": rigid_view}
    env._task_dof_views = {"c08_tongs_joint": (dof_view, 0, 0.0)}
    env._target_object_name = "nut"
    env._physics_body_pose = lambda _name: np.asarray(
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    env._physical_recording_origin = {
        "trajectory_per_frame_state_writes": False,
        "per_frame_state_writes": False,
    }
    env._trajectory_action_phase_active = True
    identity_target = np.eye(4, dtype=np.float64)
    identity_target[2, 3] = 1.0

    assert env._set_task_body_world_pose(
        rigid_view,
        np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    assert env._set_body_soft_target("nut", identity_target)[0]
    assert env._set_c08_tongs_joint_target(0.1)
    assert env._set_b02_nut_grasp_velocity(
        np.asarray([0.0, 0.0, 1.0])
    )

    origin = env._physical_recording_origin
    assert origin["trajectory_state_write_count"] == 4
    assert [
        entry["writer"]
        for entry in origin["trajectory_state_write_journal"]
    ] == [
        "_set_task_body_world_pose",
        "_set_body_soft_target",
        "_set_c08_tongs_joint_target",
        "_set_b02_nut_grasp_velocity",
    ]


def test_c05_display_is_a_fixed_runtime_fixture() -> None:
    assert HandsOnABCIsaacLabEnv._c_tool_fixture_names(
        "t14_move_and_click_mouse"
    ) == ("display_link_1",)


def test_c10_two_stage_connections_latch_in_order() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t02_connect_charger_and_cable",
        seed=31,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    strip = np.asarray([0.0, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0])
    plug_quat = np.asarray(
        [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0],
        dtype=np.float64,
    )
    env._poses["c10_power_strip"] = strip.copy()
    # Prong tips only touch the socket entrance here; this must not latch.
    env._poses["c10_charger_plug"] = np.asarray(
        [0.0, 0.0, 0.966, *plug_quat]
    )
    env._poses["c10_charging_cable"] = np.asarray(
        [0.30, 0.30, 0.930, 1.0, 0.0, 0.0, 0.0]
    )

    env._update_c10_connection_latches()
    assert env._c10_plug_latched is False

    # The charger face is now seated at the socket plate and the prongs are
    # 34 mm into the strip.
    env._poses["c10_charger_plug"] = np.asarray(
        [0.0, 0.0, 0.932, *plug_quat]
    )
    for _ in range(3):
        env._update_c10_connection_latches()
    assert env._c10_plug_latched is True
    assert env._c10_cable_latched is False
    assert env._c10_order_violation is False
    np.testing.assert_allclose(
        _invert_pose_matrix(_pose_vector_to_matrix(strip))
        @ _pose_vector_to_matrix(env._poses["c10_charger_plug"]),
        _C10_PLUG_SEATED_RELATIVE,
        atol=1.0e-7,
    )

    plug = env._poses["c10_charger_plug"]
    plug_matrix = _pose_vector_to_matrix(plug)
    cable_root = (plug_matrix @ [-0.108, 0.0, 0.0, 1.0])[:3]
    env._poses["c10_charging_cable"] = np.asarray(
        [*cable_root, *plug_quat]
    )
    for _ in range(3):
        env._update_c10_connection_latches()

    assert env._c10_cable_latched is True
    metrics = env._c10_connection_metrics()
    assert metrics["c10_first_connected"] is True
    assert metrics["c10_second_connected"] is True
    assert metrics["c10_order_valid"] is True
    np.testing.assert_allclose(
        _invert_pose_matrix(_pose_vector_to_matrix(env._poses["c10_charger_plug"]))
        @ _pose_vector_to_matrix(env._poses["c10_charging_cable"]),
        _C10_CABLE_SEATED_RELATIVE,
        atol=1.0e-7,
    )


def test_c10_connectors_require_near_full_insertion_before_latching() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t02_connect_charger_and_cable",
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    strip = np.asarray([0.0, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0])
    plug_quat = np.asarray(
        [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0],
        dtype=np.float64,
    )
    env._poses["c10_power_strip"] = strip
    env._poses["c10_charging_cable"] = np.asarray(
        [0.30, 0.30, 0.930, 1.0, 0.0, 0.0, 0.0]
    )

    # The prongs are 30 mm inside, but the charger shoulder is still 4 mm
    # above the socket.  This must not be accepted as a completed insertion.
    env._poses["c10_charger_plug"] = np.asarray(
        [0.0, 0.0, 0.936, *plug_quat]
    )
    env._update_c10_connection_latches()
    assert env._c10_plug_latched is False

    env._poses["c10_charger_plug"] = np.asarray(
        [0.0, 0.0, 0.932, *plug_quat]
    )
    for _ in range(3):
        env._update_c10_connection_latches()
    assert env._c10_plug_latched is True

    plug_matrix = _pose_vector_to_matrix(env._poses["c10_charger_plug"])
    shallow_cable_root = (
        plug_matrix @ np.asarray([-0.105, 0.0, 0.0, 1.0])
    )[:3]
    env._poses["c10_charging_cable"] = np.asarray(
        [*shallow_cable_root, *plug_quat]
    )
    env._update_c10_connection_latches()
    assert env._c10_cable_latched is False

    seated_cable_root = (
        plug_matrix @ np.asarray([-0.108, 0.0, 0.0, 1.0])
    )[:3]
    env._poses["c10_charging_cable"] = np.asarray(
        [*seated_cable_root, *plug_quat]
    )
    for _ in range(3):
        env._update_c10_connection_latches()
    assert env._c10_cable_latched is True


def test_c10_first_stage_latch_never_rewrites_loose_cable() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t02_connect_charger_and_cable",
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    strip = np.asarray([0.0, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0])
    plug_quat = np.asarray(
        [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0],
        dtype=np.float64,
    )
    env._poses = {
        "c10_power_strip": strip,
        "c10_charger_plug": np.asarray(
            [0.0, 0.0, 0.932, *plug_quat]
        ),
        "c10_charging_cable": np.asarray(
            [0.30, 0.30, 0.930, 1.0, 0.0, 0.0, 0.0]
        ),
    }
    plug_view = object()
    cable_view = object()
    env._task_body_views = {
        "c10_charger_plug": plug_view,
        "c10_charging_cable": cable_view,
    }
    pose_writes: list[tuple[object, np.ndarray]] = []
    kinematic_writes: list[tuple[str, bool]] = []
    env._physics_body_pose = lambda name: env._poses[name].copy()
    env._set_task_body_world_pose = lambda view, pose: (
        pose_writes.append((view, np.asarray(pose).copy())) or True
    )
    env._set_c10_connector_kinematic = lambda name, enabled: (
        kinematic_writes.append((name, enabled)) or True
    )

    for _ in range(3):
        env._update_c10_connection_latches()
    assert env._c10_plug_latched is True
    assert env._c10_cable_latched is False
    assert [view for view, _pose in pose_writes] == [plug_view]
    assert kinematic_writes == [("c10_charger_plug", True)]

    moved_cable = np.asarray(
        [0.34, 0.22, 1.02, 1.0, 0.0, 0.0, 0.0]
    )
    env._poses["c10_charging_cable"] = moved_cable.copy()
    env._update_c10_connection_latches()

    assert [view for view, _pose in pose_writes] == [plug_view]
    assert kinematic_writes == [("c10_charger_plug", True)]
    np.testing.assert_allclose(
        env._poses["c10_charging_cable"],
        moved_cable,
    )


def test_c12_randomized_blocks_stay_on_right_side_of_rotated_baseline() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t11_line_up_blocks",
        seed=41,
        launch_isaac=False,
        image_obs=False,
        randomize=True,
    )
    yaw = np.deg2rad(11.0)
    rotation = np.asarray(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=np.float64,
    )
    pivot = np.asarray([-0.28, 0.0], dtype=np.float64)
    hand_points = (
        *env._a06_hand_points("right"),
        *env._a06_hand_points("left"),
    )
    set_square_xy = pivot + rotation @ _C12_SET_SQUARE_LOCAL_XY
    tool_segments = tuple(
        set_square_xy.reshape(1, 2) + segment[:, :2] @ rotation.T
        for segment in _C12_TOOL_SEGMENTS_LOCAL
    )
    layouts = []
    for _ in range(32):
        xy_by_name = env._c12_randomized_block_xy(
            rotation,
            pivot,
            np.zeros(2, dtype=np.float64),
        )
        local_points = [
            rotation.T @ (xy_by_name[name] - pivot)
            for name in _C12_BLOCK_NAMES
        ]
        layouts.append(np.stack(local_points))
        assert all(
            _C12_BLOCK_LOCAL_RIGHT_Y_BOUNDS[0]
            <= float(local[1])
            <= _C12_BLOCK_LOCAL_RIGHT_Y_BOUNDS[1]
            for local in local_points
        )
        assert (
            env._c12_reset_layout_margin(
                xy_by_name,
                hand_points,
                tool_segments,
            )
            >= _C12_RESET_MIN_CLEARANCE_SLACK
        )
    assert np.ptp(np.stack(layouts), axis=0).min() > 0.01


def test_c12_complete_rotating_footprints_are_clear_for_one_hundred_seeds() -> None:
    pivot = np.asarray([-0.28, 0.0], dtype=np.float64)
    required_pair_distance = (
        2.0 * _C12_BLOCK_FOOTPRINT_RADIUS
        + _C12_RESET_BLOCK_CLEARANCE_MARGIN
    )
    for seed in range(100):
        env = HandsOnABCIsaacLabEnv(
            "t11_line_up_blocks",
            seed=seed,
            launch_isaac=False,
            image_obs=False,
            randomize=True,
        )
        yaw = np.deg2rad(float(env._random.uniform(-18.0, 18.0)))
        rotation = np.asarray(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
            dtype=np.float64,
        )
        common_xy = env._random.uniform(-0.025, 0.025, size=2)
        common_xy = env._c12_workspace_safe_common_xy(
            rotation,
            pivot,
            common_xy,
        )
        layout = env._c12_randomized_block_xy(
            rotation,
            pivot,
            common_xy,
        )
        hand_points = (
            *env._a06_hand_points("right"),
            *env._a06_hand_points("left"),
        )
        set_square_xy = pivot + rotation @ _C12_SET_SQUARE_LOCAL_XY + common_xy
        tool_segments = tuple(
            set_square_xy.reshape(1, 2) + segment[:, :2] @ rotation.T
            for segment in _C12_TOOL_SEGMENTS_LOCAL
        )
        assert (
            env._c12_reset_layout_margin(layout, hand_points, tool_segments)
            >= _C12_RESET_MIN_CLEARANCE_SLACK
        )
        names = sorted(layout)
        for index, name_a in enumerate(names):
            for name_b in names[index + 1 :]:
                assert (
                    np.linalg.norm(layout[name_a] - layout[name_b])
                    >= required_pair_distance + _C12_RESET_MIN_CLEARANCE_SLACK
                )

    assert 0.89 - _C12_SUPPORT_LOCAL_MIN_Z["c12_set_square"] == pytest.approx(
        0.900
    )
    for name in _C12_BLOCK_NAMES:
        assert 0.89 - _C12_SUPPORT_LOCAL_MIN_Z[name] == pytest.approx(0.9115)


def test_c12_reset_clearance_uses_the_block_vertical_envelope() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t11_line_up_blocks",
        launch_isaac=False,
        image_obs=False,
    )
    layout = {"c12_block_0": np.asarray([-0.28, -0.16])}
    same_xy_high = np.asarray([-0.28, -0.16, 1.50])
    same_xy_at_block = np.asarray([-0.28, -0.16, 0.912])

    assert env._c12_reset_layout_margin(layout, (same_xy_high,)) > 0.40
    assert env._c12_reset_layout_margin(layout, (same_xy_at_block,)) < 0.0


def test_c12_target_overlap_requires_more_than_eighty_percent() -> None:
    assert abc_env._C12_MIN_TARGET_OVERLAP_RATIO == pytest.approx(0.80)
    baseline = np.asarray(
        [0.0, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    block_quat = np.asarray(
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        dtype=np.float64,
    )
    centered = np.asarray(
        [-0.11, 0.0, 0.92, *block_quat],
        dtype=np.float64,
    )
    assert (
        HandsOnABCIsaacLabEnv._c12_block_target_overlap_ratio(
            centered,
            baseline,
            -0.11,
        )
        == pytest.approx(1.0)
    )

    above_eighty = centered.copy()
    above_eighty[1] = 0.007
    below_eighty = centered.copy()
    below_eighty[1] = 0.010
    assert (
        HandsOnABCIsaacLabEnv._c12_block_target_overlap_ratio(
            above_eighty,
            baseline,
            -0.11,
        )
        > abc_env._C12_MIN_TARGET_OVERLAP_RATIO
    )
    assert (
        HandsOnABCIsaacLabEnv._c12_block_target_overlap_ratio(
            below_eighty,
            baseline,
            -0.11,
        )
        < abc_env._C12_MIN_TARGET_OVERLAP_RATIO
    )


def test_c12_direct_hand_touch_requests_episode_reset_after_debounce() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t11_line_up_blocks",
        launch_isaac=False,
        image_obs=False,
    )
    env._c12_direct_hand_touch_now = lambda: True
    env._after_task_physics_substep()
    assert env._c12_direct_hand_contact is False
    assert env._manual_reset_requested is False

    env._after_task_physics_substep()
    assert env._c12_direct_hand_contact is True
    assert env._manual_reset_requested is True


def test_c12_direct_hand_touch_debounce_requires_consecutive_substeps() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t11_line_up_blocks",
        launch_isaac=False,
        image_obs=False,
    )
    touch_samples = iter((True, False, True, True))
    env._c12_direct_hand_touch_now = lambda: next(touch_samples)

    for expected_steps in (1, 0, 1):
        env._after_task_physics_substep()
        assert env._c12_direct_hand_contact is False
        assert env._manual_reset_requested is False
        assert env._c12_direct_hand_touch_steps == expected_steps

    env._after_task_physics_substep()
    assert env._c12_direct_hand_contact is True
    assert env._manual_reset_requested is True


def test_c12_complete_contact_reporting_disables_geometric_air_gap() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t11_line_up_blocks",
        launch_isaac=False,
        image_obs=False,
    )
    env._c12_block_contact_views = {
        name: object() for name in abc_env._C12_BLOCK_NAMES
    }
    env._c12_right_hand_block_contact_force = lambda: 0.0
    env._physics_body_pose = lambda _name: np.asarray(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    env._a06_hand_points = lambda _side: (np.zeros(3, dtype=np.float64),)

    assert env._c12_direct_hand_touch_now() is False

    env._c12_right_hand_block_contact_force = lambda: (
        abc_env._C12_DIRECT_TOUCH_FORCE
    )
    assert env._c12_direct_hand_touch_now() is True


def test_c12_geometric_touch_is_tight_fallback_for_unmonitored_block() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t11_line_up_blocks",
        launch_isaac=False,
        image_obs=False,
    )
    monitored = abc_env._C12_BLOCK_NAMES[:2]
    missing = abc_env._C12_BLOCK_NAMES[2]
    env._c12_block_contact_views = {name: object() for name in monitored}
    env._c12_right_hand_block_contact_force = lambda: 0.0
    poses = {
        monitored[0]: np.asarray(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        ),
        monitored[1]: np.asarray(
            [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        ),
        missing: np.asarray(
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        ),
    }
    env._physics_body_pose = lambda name: poses[name]

    # Even an origin inside a monitored block must not duplicate its healthy
    # contact tensor with the coarse geometric fallback.
    env._a06_hand_points = lambda _side: (np.zeros(3, dtype=np.float64),)
    assert env._c12_direct_hand_touch_now() is False

    half_x = float(abc_env._C12_BLOCK_HALF_EXTENTS[0])
    for gap_mm, expected in ((1.9, True), (2.1, False)):
        point = np.asarray(
            [1.0 + half_x + gap_mm / 1000.0, 0.0, 0.0],
            dtype=np.float64,
        )
        env._a06_hand_points = lambda _side, point=point: (point,)
        assert env._c12_direct_hand_touch_now() is expected


def test_c04_reset_randomizes_full_table_yaw_near_right_hand() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t50_water_plant",
        seed=29,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )

    xy, yaw = env._c04_reset_spray_pose()
    np.testing.assert_allclose(xy, _C04_SPRAY_INIT_XY)
    assert yaw == -105.0

    avoid_points = (
        np.asarray(_RIGHT_HOME[:3], dtype=np.float64),
    )
    env.randomize = True
    env._reset_robot_body_points = lambda _side: avoid_points
    positions = []
    yaws = []
    for _ in range(160):
        layout, yaw = env._c04_reset_layout()
        xy = layout["spray"]
        plant_xy = layout["plant"]
        positions.append(xy)
        yaws.append(yaw)
        assert np.all(xy >= _C04_SPRAY_XY_BOUNDS[0])
        assert np.all(xy <= _C04_SPRAY_XY_BOUNDS[1])
        assert np.all(plant_xy >= _C04_PLANT_XY_BOUNDS[0])
        assert np.all(plant_xy <= _C04_PLANT_XY_BOUNDS[1])
        assert min(
            float(np.linalg.norm(xy - point[:2])) for point in avoid_points
        ) >= _C04_SPRAY_RIGHT_ARM_MIN_XY_DISTANCE
        assert (
            env._c04_reset_layout_margin(
                layout,
                avoid_points,
                avoid_points + avoid_points,
                yaw,
            )
            >= _C04_RESET_MIN_CLEARANCE_SLACK
        )
        assert _C04_SPRAY_YAW_BOUNDS_DEG[0] <= yaw <= _C04_SPRAY_YAW_BOUNDS_DEG[1]

    assert np.ptp(np.asarray(positions), axis=0).min() > 0.01
    assert min(yaws) >= -180.0
    assert max(yaws) <= 180.0
    yaw_quadrants = (
        np.floor((np.asarray(yaws) + 180.0) / 90.0).astype(int) % 4
    )
    assert set(yaw_quadrants) == {0, 1, 2, 3}


def test_c04_reset_rejects_rotated_spray_footprint_touching_hand() -> None:
    layout = {
        "spray": np.asarray([-0.2211, -0.0809], dtype=np.float64),
        "plant": np.asarray([-0.3651, 0.1322], dtype=np.float64),
    }
    right_points = (
        np.asarray([-0.10, -0.16, 1.00], dtype=np.float64),
    )

    unrotated_margin = HandsOnABCIsaacLabEnv._c04_reset_layout_margin(
        layout,
        right_points,
        right_points,
        0.0,
    )
    rotated_margin = HandsOnABCIsaacLabEnv._c04_reset_layout_margin(
        layout,
        right_points,
        right_points,
        68.2583,
    )

    assert unrotated_margin >= _C04_RESET_MIN_CLEARANCE_SLACK
    assert rotated_margin < _C04_RESET_MIN_CLEARANCE_SLACK


def test_c04_layout_ignores_robot_bodies_above_the_plant_contact_envelope() -> None:
    layout = {
        "spray": np.asarray([-0.18, -0.08], dtype=np.float64),
        "plant": np.asarray([-0.30, 0.14], dtype=np.float64),
    }
    low_right = (np.asarray([-0.36, -0.24, 1.00], dtype=np.float64),)
    high_left = (
        np.asarray(
            [-0.30, 0.14, _C04_PLANT_HAND_MAX_Z + 0.20],
            dtype=np.float64,
        ),
    )

    margin = HandsOnABCIsaacLabEnv._c04_reset_layout_margin(
        layout,
        low_right,
        low_right + high_left,
    )

    assert margin >= _C04_RESET_MIN_CLEARANCE_SLACK


def test_c04_fallback_fails_closed_when_its_envelope_gate_rejects() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t50_water_plant",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
    )
    blocked_point = np.asarray(
        [_C04_SPRAY_INIT_XY[0], _C04_SPRAY_INIT_XY[1], 1.0],
        dtype=np.float64,
    )
    env._reset_robot_body_points = lambda _side: (blocked_point,)
    def blocked_uniform(low, high):
        if np.asarray(low).ndim == 0:
            return 0.0
        return np.asarray(_C04_SPRAY_XY_BOUNDS[0], dtype=np.float64)

    env._random = types.SimpleNamespace(uniform=blocked_uniform)

    with pytest.raises(RuntimeError, match="t50_water_plant fallback.*初态门禁"):
        env._c04_reset_layout()


def test_task_reset_prefers_articulation_root_transform_writer() -> None:
    rigid_state = (object(), object(), object(), object())
    articulation_state = (object(), object(), object(), object())

    selected = HandsOnABCIsaacLabEnv._reset_transform_state_for_path(
        "/World/Task/link_2",
        {"/World/Task/link_2": rigid_state},
        {"/World/Task/link_2": articulation_state},
    )

    assert selected == ("articulation", *articulation_state)


def test_b04_c01_and_c04_use_source_table_height_ranges() -> None:
    for task_name, maximum in (
        ("t44_stack_blocks", 0.02),
        ("t08_hammer_nail", 0.03),
        ("t50_water_plant", 0.02),
    ):
        env = HandsOnABCIsaacLabEnv(
            task_name,
            seed=23,
            launch_isaac=False,
            image_obs=False,
            randomize=True,
        )
        samples = []
        for _ in range(40):
            env._sample_reset_state(clear_manual_request=True)
            samples.append(float(env.delta_h[0]))
        assert min(samples) >= 0.0
        assert max(samples) <= maximum
        assert max(samples) > 0.8 * maximum


def test_b03_table_height_stays_below_right_hand_contact_limit() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        seed=23,
        launch_isaac=False,
        image_obs=False,
        randomize=True,
    )

    samples = []
    for _ in range(80):
        env._sample_reset_state(clear_manual_request=True)
        samples.append(float(env.delta_h[0]))

    assert min(samples) >= 0.0
    assert max(samples) <= 0.010
    assert max(samples) > 0.008


@pytest.mark.parametrize(
    "task_name",
    ["t36_scoop_marbles", "t03_cover_box"],
)
def test_contact_sensitive_tasks_keep_randomized_safe_table_height(
    task_name: str,
) -> None:
    env = HandsOnABCIsaacLabEnv(
        task_name,
        seed=29,
        launch_isaac=False,
        image_obs=False,
        randomize=True,
    )

    samples = []
    for _ in range(80):
        env._sample_reset_state(clear_manual_request=True)
        samples.append(float(env.delta_h[0]))

    assert min(samples) >= 0.0
    assert max(samples) <= 0.010
    assert max(samples) > 0.008


def test_a06_accepts_visually_upright_cups_and_rejects_upside_down_cups() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t19_orient_cup_upright",
        launch_isaac=False,
        image_obs=False,
    )

    def x_tilt_quat(degrees: float) -> np.ndarray:
        radians = np.deg2rad(degrees)
        return np.asarray([np.cos(radians / 2.0), np.sin(radians / 2.0), 0.0, 0.0])

    env._poses = {
        "a_cup": np.asarray([-0.43, -0.06, 0.921, 1.0, 0.0, 0.0, 0.0])
    }
    required_steps = env._success_steps_required()
    assert required_steps >= 25
    assert env._cup_bottom_down()
    for _ in range(required_steps - 1):
        assert not env._compute_success()
    assert env._compute_success()
    assert env._task_info()["a06_success_counter"] == required_steps

    # Isaac can report the imported rigid-body origin at the table surface
    # while the rendered cup is visibly standing upright.  Do not reject that
    # one-millimetre-over-the-old-boundary pose.
    env._success_counter = 0
    env._poses["a_cup"] = np.asarray([-0.43, -0.06, 0.890, 1.0, 0.0, 0.0, 0.0])
    assert env._cup_bottom_down()
    assert env._task_info()["a06_on_table"] is True

    env._poses["a_cup"][:3] = np.asarray([-0.43, -0.06, 0.921])
    env._poses["a_cup"][3:7] = x_tilt_quat(20.0)
    assert env._cup_bottom_down()
    assert env._task_info()["a06_opening_angle_deg"] < 25.0

    env._poses["a_cup"][3:7] = x_tilt_quat(30.0)
    assert not env._cup_bottom_down()

    env._success_counter = 0
    env._poses["a_cup"][3:7] = np.asarray([0.0, 1.0, 0.0, 0.0])
    assert not env._cup_bottom_down()
    assert not env._compute_success()
    assert env._success_counter == 0
    assert env._task_info()["a06_opening_up"] is False

    env._poses["a_cup"][3:7] = x_tilt_quat(45.0)
    assert not env._cup_bottom_down()


def test_a06_requires_the_hand_to_leave_for_half_a_second() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t19_orient_cup_upright",
        launch_isaac=False,
        image_obs=False,
        control_dt=0.02,
    )
    env._poses = {
        "a_cup": np.asarray([-0.43, -0.06, 0.921, 1.0, 0.0, 0.0, 0.0])
    }

    env._right_tcp_pose = np.asarray([-0.42, -0.06, 0.921, 1.0, 0.0, 0.0, 0.0])
    assert not env._cup_bottom_down()
    info = env._task_info()
    assert info["a06_pose_upright"] is True
    assert info["a06_hand_clear"] is False
    assert info["a06_hand_contact_free"] is False
    assert info["a06_hand_min_distance"] < info["a06_hand_clearance_distance"]

    env._success_counter = env._success_steps_required() - 1
    assert not env._compute_success()
    assert env._success_counter == 0

    env._right_tcp_pose = np.asarray([-0.10, -0.16, 1.07, 1.0, 0.0, 0.0, 0.0])
    required_steps = env._success_steps_required()
    assert env._task_info()["a06_success_steps_required"] == required_steps
    for _ in range(required_steps - 1):
        assert not env._compute_success()
    assert env._compute_success()
    info = env._task_info()
    assert info["a06_hand_clear"] is True
    assert info["a06_release_hold_seconds"] == 0.5


class _FakeContactView:
    sensor_count = 1
    filter_count = 1

    def __init__(self, force: float):
        self.force = force

    def get_contact_force_matrix(self, _dt: float) -> np.ndarray:
        return np.asarray([[[self.force, 0.0, 0.0]]], dtype=np.float64)


def _b04_supported_poses(table_z: float = 0.89) -> dict[str, np.ndarray]:
    half_heights = {
        "036_wood_block": 0.042689,
        "BuildingBlock017": 0.014925,
        "BuildingBlock018": 0.015000,
        "BuildingBlock019": 0.015000,
    }
    names = tuple(half_heights)
    poses: dict[str, np.ndarray] = {}
    center_z = table_z
    for index, name in enumerate(names):
        center_z += half_heights[name]
        # The small tower may sit off-center on the wide base, but every
        # adjacent pair must still have a real support overlap.
        xy = np.asarray([-0.30 if index == 0 else -0.28, -0.10])
        poses[name] = np.asarray([*xy, center_z, 1.0, 0.0, 0.0, 0.0])
        center_z += half_heights[name]
    return poses


class _FakeRigidBodyPoseView:
    def __init__(self, z: float):
        self._transform = np.asarray(
            [[-0.15, 0.0, z, 0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def get_transforms(self) -> np.ndarray:
        return self._transform


def _set_b04_contact_force(
    env: HandsOnABCIsaacLabEnv,
    force: float,
) -> None:
    env._sim_dt = 1.0 / 120.0
    env._b04_block_contact_views = {
        name: _FakeContactView(force if index == 0 else 0.0)
        for index, name in enumerate(_b04_supported_poses())
    }


def test_b04_requires_supported_stable_stack_and_real_hand_release() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t44_stack_blocks",
        launch_isaac=False,
        image_obs=False,
        control_dt=0.02,
    )
    poses = _b04_supported_poses()
    env._poses = {name: pose.copy() for name, pose in poses.items()}
    env._physics_body_velocity = lambda name: np.asarray(
        [0.0, 0.0, 0.0, 0.20, 0.0, 0.0],
        dtype=np.float64,
    )

    env._right_tcp_pose = np.asarray([0.30, 0.30, 1.20, 1.0, 0.0, 0.0, 0.0])
    _set_b04_contact_force(env, 0.02)
    assert not env._block_stack_stable()
    info = env._task_info()
    assert info["stack_geometry_valid"] is True
    assert info["stack_four_layers"] is True
    assert info["stack_large_block_on_bottom"] is True
    assert info["stack_support_graph_valid"] is True
    assert info["stack_bottom_block"] == "036_wood_block"
    assert info["stack_hand_released"] is False
    assert info["stack_hand_contact_available"] is True
    assert info["stack_hand_release_source"] == "physx_contact"
    assert info["stack_velocity_stable"] is True
    assert info["stack_max_angular_speed"] == 0.20

    _set_b04_contact_force(env, 0.0)
    assert env._block_stack_stable()
    assert env._task_info()["stack_hand_released"] is True

    env._physics_body_velocity = lambda name: np.asarray(
        [0.026, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    assert env._block_stack_stable()
    info = env._task_info()
    assert info["stack_velocity_instant_stable"] is False
    assert info["stack_velocity_stable"] is True

    env._physics_body_velocity = lambda name: np.asarray(
        [0.045, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    for _ in range(5):
        stable = env._block_stack_stable()
    assert not stable
    info = env._task_info()
    assert np.isclose(info["stack_filtered_linear_speed"], 0.045)
    assert info["stack_velocity_stable"] is False

    invalid_poses = {name: pose.copy() for name, pose in poses.items()}
    invalid_poses["036_wood_block"] = poses["BuildingBlock018"].copy()
    invalid_poses["BuildingBlock018"] = poses["036_wood_block"].copy()
    env._poses = invalid_poses
    assert not env._block_stack_stable()
    assert env._task_info()["stack_large_block_on_bottom"] is False

    env._poses = {name: pose.copy() for name, pose in poses.items()}
    env._physics_body_velocity = lambda name: np.zeros(6, dtype=np.float64)
    env._right_tcp_pose = np.asarray([0.30, 0.30, 1.20, 1.0, 0.0, 0.0, 0.0])
    _set_b04_contact_force(env, 0.0)
    required_steps = env._success_steps_required()
    assert required_steps >= 25
    for _ in range(required_steps - 1):
        assert not env._compute_success()
    assert env._compute_success()
    info = env._task_info()
    assert info["stack_success_counter"] == required_steps
    assert info["stack_success_steps_required"] == required_steps
    assert info["stable_steps"] == required_steps
    assert info["stable_seconds"] >= 0.5
    assert info["required_stable_seconds"] == 0.5


def test_b04_velocity_filter_ignores_one_frame_spike_but_rejects_motion() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t44_stack_blocks",
        launch_isaac=False,
        image_obs=False,
    )
    env._poses = _b04_supported_poses()
    env._right_tcp_pose = np.asarray(
        [0.30, 0.30, 1.20, 1.0, 0.0, 0.0, 0.0]
    )
    _set_b04_contact_force(env, 0.0)

    velocity = np.zeros(6, dtype=np.float64)
    env._physics_body_velocity = lambda _name: velocity.copy()
    assert env._block_stack_stable()

    velocity[0] = 0.035
    velocity[3] = 0.48
    assert env._block_stack_stable()
    info = env._task_info()
    assert info["stack_velocity_instant_stable"] is False
    assert info["stack_velocity_stable"] is True

    velocity[0] = 0.11
    velocity[3] = 0.0
    assert not env._block_stack_stable()
    assert env._task_info()["stack_velocity_stable_latched"] is False

    velocity.fill(0.0)
    assert env._block_stack_stable()


def test_b04_bottom_on_table_uses_live_physx_table_height() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t44_stack_blocks",
        launch_isaac=False,
        image_obs=False,
    )
    physical_delta_h = 0.018
    env.delta_h[:] = 0.05
    env._task_body_views["table"] = _FakeRigidBodyPoseView(physical_delta_h)
    env._poses = _b04_supported_poses(0.89 + physical_delta_h)
    env._physics_body_velocity = lambda name: np.zeros(6, dtype=np.float64)
    env._right_tcp_pose = np.asarray(
        [0.30, 0.30, 1.20, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    _set_b04_contact_force(env, 0.0)

    metrics = env._block_stack_metrics()

    assert metrics["stack_bottom_on_table"] is True
    assert abs(float(metrics["stack_bottom_clearance"])) < 1e-9
    assert metrics["stack_height_ok"] is True


def test_b04_rejects_floating_or_laterally_disconnected_layers() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t44_stack_blocks",
        launch_isaac=False,
        image_obs=False,
    )
    poses = _b04_supported_poses()
    env._physics_body_velocity = lambda name: np.zeros(6, dtype=np.float64)
    env._right_tcp_pose = np.asarray([0.30, 0.30, 1.20, 1.0, 0.0, 0.0, 0.0])
    _set_b04_contact_force(env, 0.0)

    floating = {name: pose.copy() for name, pose in poses.items()}
    floating["BuildingBlock019"][2] += 0.12
    env._poses = floating
    assert not env._block_stack_stable()
    info = env._task_info()
    assert info["stack_four_layers"] is True
    assert info["stack_support_graph_valid"] is False
    assert info["stack_geometry_valid"] is False

    disconnected = {name: pose.copy() for name, pose in poses.items()}
    disconnected["BuildingBlock019"][0] += 0.08
    env._poses = disconnected
    assert not env._block_stack_stable()
    assert env._task_info()["stack_support_graph_valid"] is False


def test_b04_single_arm_release_ignores_parked_left_hand() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t44_stack_blocks",
        launch_isaac=False,
        image_obs=False,
    )
    poses = _b04_supported_poses()
    calls: list[str] = []

    def hand_points(side: str) -> tuple[np.ndarray, ...]:
        calls.append(side)
        if side == "left":
            return (poses["036_wood_block"][:3].copy(),)
        return (np.asarray([0.30, 0.30, 1.20]),)

    env._robots = {"left": object(), "right": object()}
    env._b04_block_contact_views = {}
    env._a06_hand_points = hand_points
    metrics = env._block_stack_hand_metrics(poses)
    assert metrics["released"] is True
    assert metrics["source"] == "distance_fallback"
    assert calls == ["right"]

    env._launch_isaac = True
    metrics = env._block_stack_hand_metrics(poses)
    assert metrics["released"] is False
    assert metrics["source"] == "contact_unavailable"


def test_b04_overlay_rows_are_exactly_the_success_gates() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t44_stack_blocks",
        launch_isaac=False,
        image_obs=False,
    )
    env._poses = _b04_supported_poses()
    env._physics_body_velocity = lambda name: np.zeros(6, dtype=np.float64)
    env._right_tcp_pose = np.asarray([0.30, 0.30, 1.20, 1.0, 0.0, 0.0, 0.0])
    _set_b04_contact_force(env, 0.0)
    rows = env._b04_success_condition_rows()
    assert [row["key"] for row in rows] == [
        "b04_four_layers",
        "b04_large_block_on_bottom",
        "b04_support_graph_valid",
        "b04_upright",
        "b04_bottom_on_table",
        "b04_hand_released",
        "b04_velocity_stable",
    ]
    assert all(row["ok"] for row in rows)


def test_task_free_body_names_drive_reset_snapshot(tmp_path) -> None:
    task_mjcf = tmp_path / "task.xml"
    task_mjcf.write_text(
        "<mujoco><worldbody>"
        '<body name="cup"><freejoint name="cup_joint"/></body>'
        '<body name="hammer"><joint name="hammer_joint" type="free"/></body>'
        '<body name="table"><geom type="box" size="1 1 1"/></body>'
        "</worldbody></mujoco>",
        encoding="utf-8",
    )

    assert HandsOnABCIsaacLabEnv._free_body_names(task_mjcf) == ("cup", "hammer")


def test_abc_reset_pose_randomization_applies_xyz_and_world_yaw() -> None:
    initial = np.asarray([[0.10, -0.20, 0.90, 0.0, 0.0, 0.0, 1.0]])

    randomized = _randomized_physx_pose(
        initial,
        xy_offset=np.asarray([0.012, -0.006]),
        z_offset=0.035,
        yaw_degrees=90.0,
    )

    np.testing.assert_allclose(randomized[0, :3], [0.112, -0.206, 0.935])
    np.testing.assert_allclose(
        randomized[0, 3:7],
        [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
        atol=1e-7,
    )


def test_isaaclab_collection_defaults_to_cuda(monkeypatch) -> None:
    monkeypatch.setenv("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "0")
    monkeypatch.delenv("HANDSON_ISAACLAB_DEVICE", raising=False)
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
    )

    assert env._isaac_device == "cuda:0"


@pytest.mark.parametrize(
    "device",
    (
        "",
        "cpu",
        "cuda",
        "cuda:-1",
        "cuda:01",
        "CUDA:0",
        " cuda:0",
        "cuda:0 ",
        "cuda:\N{FULLWIDTH DIGIT ZERO}",
    ),
)
def test_physx_ground_truth_rejects_noncanonical_device_before_kit_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    monkeypatch.setenv("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "1")
    kit_launch_attempted = False

    def unexpected_setup(_self) -> None:
        nonlocal kit_launch_attempted
        kit_launch_attempted = True

    monkeypatch.setattr(
        assembly_env.BimanualAssemblyIsaacLabEnv,
        "_setup_isaaclab",
        unexpected_setup,
    )

    with pytest.raises(
        RuntimeError,
        match="explicit canonical CUDA device",
    ):
        assembly_env.BimanualAssemblyIsaacLabEnv(
            launch_isaac=True,
            image_obs=False,
            isaac_device=device,
        )

    assert kit_launch_attempted is False


@pytest.mark.parametrize("device", ("cuda:0", "cuda:1", "cuda:12"))
def test_physx_ground_truth_accepts_canonical_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    monkeypatch.setenv("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "1")
    env = assembly_env.BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        isaac_device=device,
    )

    assert env._isaac_device == device


def test_non_ground_truth_interface_mode_still_allows_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "0")
    env = assembly_env.BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        isaac_device="cpu",
    )

    assert env._isaac_device == "cpu"


def test_abc_recording_cameras_are_spawned_outside_the_imported_task_robot(
    monkeypatch,
) -> None:
    created = []

    class FakePinholeCameraCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCameraCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeCamera:
        def __init__(self, cfg):
            self.cfg = cfg
            created.append(self)

    isaaclab = types.ModuleType("isaaclab")
    isaaclab_sim = types.ModuleType("isaaclab.sim")
    isaaclab_sim.PinholeCameraCfg = FakePinholeCameraCfg
    isaaclab_sensors = types.ModuleType("isaaclab.sensors")
    isaaclab_sensors.Camera = FakeCamera
    isaaclab_sensors.CameraCfg = FakeCameraCfg
    isaaclab.sim = isaaclab_sim
    isaaclab.sensors = isaaclab_sensors
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", isaaclab_sim)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", isaaclab_sensors)

    env = HandsOnABCIsaacLabEnv(
        "t24_place_ball_in_bowl",
        launch_isaac=False,
        image_obs=True,
    )
    env._init_recording_cameras()

    assert tuple(env._cameras) == _RECORDING_CAMERA_NAMES
    assert env._dynamic_cameras == set(_RECORDING_CAMERA_NAMES)
    assert env._camera_aliases == {}
    assert [camera.cfg.prim_path for camera in created] == [
        _RECORDING_CAMERA_PRIM_PATHS[name]
        for name in _RECORDING_CAMERA_NAMES
    ]
    assert created[0].cfg.spawn.kwargs["focal_length"] == _focal_length_for_vertical_fov(
        _HEAD_RECORDING_CAMERA_FOVY
    )
    assert created[0].cfg.spawn.kwargs["vertical_aperture"] == 24.0
    from handson.sim.isaaclab.recording_cameras import WRIST_REAL_V1_INTRINSICS_BY_SIDE
    for side, camera in zip(("left", "right"), created[1:]):
        assert (camera.cfg.width, camera.cfg.height) == WRIST_RECORDING_CAMERA_OUTPUT_SIZE
        aperture = camera.cfg.spawn.kwargs["horizontal_aperture"]
        k = WRIST_REAL_V1_INTRINSICS_BY_SIDE[side]["K_new_848x480"]
        assert camera.cfg.spawn.kwargs["focal_length"] == pytest.approx(k[0][0] * aperture / 848)
        assert camera.cfg.spawn.kwargs["clipping_range"] == (0.03, 5.0)


def test_abc_recording_camera_uses_dynamic_world_prim_when_mjcf_camera_is_missing(
    monkeypatch,
) -> None:
    created = []

    class FakePinholeCameraCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCameraCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeCamera:
        def __init__(self, cfg):
            self.cfg = cfg
            created.append(self)

    isaaclab = types.ModuleType("isaaclab")
    isaaclab_sim = types.ModuleType("isaaclab.sim")
    isaaclab_sim.PinholeCameraCfg = FakePinholeCameraCfg
    isaaclab_sensors = types.ModuleType("isaaclab.sensors")
    isaaclab_sensors.Camera = FakeCamera
    isaaclab_sensors.CameraCfg = FakeCameraCfg
    isaaclab.sim = isaaclab_sim
    isaaclab.sensors = isaaclab_sensors
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", isaaclab_sim)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", isaaclab_sensors)

    env = HandsOnABCIsaacLabEnv(
        "t24_place_ball_in_bowl",
        launch_isaac=False,
        image_obs=True,
    )
    env._init_recording_cameras()

    assert tuple(env._cameras) == _RECORDING_CAMERA_NAMES
    assert env._dynamic_cameras == set(_RECORDING_CAMERA_NAMES)
    assert env._camera_aliases == {}
    assert [camera.cfg.prim_path for camera in created] == [
        _RECORDING_CAMERA_PRIM_PATHS[name]
        for name in _RECORDING_CAMERA_NAMES
    ]
    assert all(camera.cfg.spawn is not None for camera in created)


def test_b03_newton_recording_cameras_use_newton_warp_renderer(
    monkeypatch,
) -> None:
    created = []

    class FakePinholeCameraCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCameraCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeCamera:
        def __init__(self, cfg):
            self.cfg = cfg
            created.append(self)

    class FakeNewtonWarpRendererCfg:
        pass

    isaaclab = types.ModuleType("isaaclab")
    isaaclab_sim = types.ModuleType("isaaclab.sim")
    isaaclab_sim.PinholeCameraCfg = FakePinholeCameraCfg
    isaaclab_sensors = types.ModuleType("isaaclab.sensors")
    isaaclab_sensors.Camera = FakeCamera
    isaaclab_sensors.CameraCfg = FakeCameraCfg
    isaaclab.sim = isaaclab_sim
    isaaclab.sensors = isaaclab_sensors
    isaaclab_newton = types.ModuleType("isaaclab_newton")
    isaaclab_newton_renderers = types.ModuleType("isaaclab_newton.renderers")
    isaaclab_newton_renderers.NewtonWarpRendererCfg = (
        FakeNewtonWarpRendererCfg
    )
    isaaclab_newton.renderers = isaaclab_newton_renderers
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", isaaclab_sim)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", isaaclab_sensors)
    monkeypatch.setitem(sys.modules, "isaaclab_newton", isaaclab_newton)
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_newton.renderers",
        isaaclab_newton_renderers,
    )

    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        launch_isaac=False,
        image_obs=True,
        physics_backend="newton",
    )
    env._init_recording_cameras()

    assert len(created) == len(_RECORDING_CAMERA_NAMES)
    assert all(
        isinstance(camera.cfg.renderer_cfg, FakeNewtonWarpRendererCfg)
        for camera in created
    )


def test_newton_recording_camera_pose_is_resynced_after_cubric_handoff() -> None:
    class FakeTorch:
        float32 = np.float32

        @staticmethod
        def tensor(value, *, dtype, device):
            del device
            return np.asarray(value, dtype=dtype)

    class FakeCamera:
        def __init__(self) -> None:
            self.calls = []

        def set_world_poses(self, positions, orientations, *, convention) -> None:
            self.calls.append((positions, orientations, convention))

    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        launch_isaac=False,
        image_obs=False,
        physics_backend="newton",
    )
    env._torch = FakeTorch()
    env._isaac_device = "cpu"
    env._isaaclab_uses_xyzw = True
    env._cameras = {"head_camera": FakeCamera()}

    env._update_recording_cameras()
    env._update_recording_cameras()

    assert len(env._cameras["head_camera"].calls) == 2
    _, orientation, convention = env._cameras["head_camera"].calls[-1]
    np.testing.assert_allclose(
        orientation,
        [
            [
                _HEAD_RECORDING_CAMERA_QUAT_WXYZ[1],
                _HEAD_RECORDING_CAMERA_QUAT_WXYZ[2],
                _HEAD_RECORDING_CAMERA_QUAT_WXYZ[3],
                _HEAD_RECORDING_CAMERA_QUAT_WXYZ[0],
            ]
        ],
    )
    assert convention == "opengl"


@pytest.mark.parametrize("passes", [1, 2, 3])
def test_render_captures_each_camera_after_exposure_and_accumulation(monkeypatch, passes):
    from handson.sim.isaaclab import default_appearance
    events = []
    monkeypatch.setattr(default_appearance, "configure_exposure", lambda name=None: events.append(("exposure", name)))
    class FakeSimulation:
        def render(self):
            events.append(("render",))
    class FakeCamera:
        def __init__(self, name):
            self.name = name
        def update(self, dt, *, force_recompute):
            assert force_recompute is True
            assert dt == pytest.approx(1 / 30)
            events.append(("update", self.name))
    env = HandsOnABCIsaacLabEnv("t24_place_ball_in_bowl", launch_isaac=False, image_obs=False)
    env._sim = FakeSimulation()  # Deliberately no step: rendering must not advance physics.
    env._control_dt = 1 / 30
    env._render_accumulation_passes = passes
    env._cameras = {name: FakeCamera(name) for name in _RECORDING_CAMERA_NAMES}
    env._update_recording_cameras = lambda: events.append(("pose",))
    env._set_wrist_obstructions_hidden = lambda hidden: events.append(("hidden", hidden))
    def read(name, fallback):
        events.append(("read", name))
        return np.full((2, 2, 3), _RECORDING_CAMERA_NAMES.index(name) + 1, np.uint8)
    env._camera_rgb_frame = read
    frames = env.render()
    expected = [("pose",), ("hidden", False)]
    for i, name in enumerate(_RECORDING_CAMERA_NAMES):
        if i == 1:
            expected.append(("hidden", True))
        expected.append(("exposure", name))
        expected.extend([("render",), ("update", name)] * passes)
        expected.append(("read", name))
    expected.extend([("hidden", False), ("exposure", None)])
    assert events == expected
    for i, frame in enumerate(frames):
        np.testing.assert_array_equal(frame, np.full((2, 2, 3), i + 1, np.uint8))
