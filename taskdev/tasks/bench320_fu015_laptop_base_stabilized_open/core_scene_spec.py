"""Immutable core-scene contract for B320-FU015.

This module is deliberately safe to import without Isaac Sim. The live runner
consumes the same constants that the CPU tests and handoff manifest audit.
"""

from __future__ import annotations

import math
from pathlib import Path


TASK_ID = "B320-FU015"
RUNTIME_TASK_NAME = "t16_open_laptop"
FOUNDATION_SCHEMA = "fu015_isaac51_core_scene_v1"

TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]
LAPTOP_RELATIVE = Path(
    "taskdev/tasks/t16_open_laptop/"
    "assets/dv_laptop/model_computer_9.usd"
)
ROBOT_RELATIVE = Path("handson/handson/sim/envs/xmls/rokae_wuji1_bimanual.xml")

SOURCE_SHA256 = "31ad6079c08b7f2f80a082ec1d5d67309eb98d4bffb3fc24688c583170847aac"
ROBOT_SHA256 = "a425da3c71e547fc5f427fc172d0a18de1c51668b22d87ac6628ddba38bf6a63"

SCENE_ROOT = "/World/FU015Core"
LAPTOP_ROOT = f"{SCENE_ROOT}/Laptop"
BASE_PATH = f"{LAPTOP_ROOT}/E_body_1"
DISPLAY_PATH = f"{LAPTOP_ROOT}/E_displayer_5"
HINGE_PATH = f"{DISPLAY_PATH}/RevoluteJoint_computer_9_up"
HEAD_CAMERA_PATH = "/World/RecordCameras/head_camera"

# Source-local measured AABB in meters. The root is rotated -90 degrees so
# local -Y (keyboard front edge) faces the head camera on world -X.
SOURCE_AABB_M = (
    (-0.16404410622272492, -0.1428278000949949, -0.0004359643514315377),
    (0.16419598521804815, 0.08717669743961787, 0.017044617270144728),
)
SCALE_XYZ = (1.0, 1.0, 1.0)
PLACEMENT_XY_M = (0.02, 0.0)
PLACEMENT_YAW_DEG = -90.0
TABLE_SUPPORT_GAP_M = 0.001

# Authoritative reset allows 10%-30% of the 110-degree travel. -22 degrees is
# 20%, visibly separates lid/base colliders, and is not already successful.
HINGE_LOWER_DEG = -110.0
HINGE_UPPER_DEG = 0.0
INITIAL_HINGE_DEG = -22.0
INITIAL_HINGE_RAD = math.radians(INITIAL_HINGE_DEG)
TARGET_OPEN_DEG = -105.0
TARGET_TOLERANCE_DEG = 7.0

# A light positional drive represents passive hinge holding friction during
# the zero-action check. It is not a task action or motion controller.
HINGE_DRIVE_STIFFNESS = 3.0
HINGE_DRIVE_DAMPING = 0.8
HINGE_DRIVE_MAX_FORCE = 2.0

PHYSICS_HZ = 120
CONTROL_HZ = 30
# This is a bounded reset smoke, not the task-level 10 s completion gate in
# task.yaml. Thirty physics steps expose reset explosions, collider overlap,
# non-finite state, and an unheld hinge while remaining schedulable when the
# cluster's GPU pipeline falls back to software.
ZERO_ACTION_SECONDS = 0.25
ZERO_ACTION_STEPS = int(PHYSICS_HZ * ZERO_ACTION_SECONDS)
HEAD_SHAPE_HWC = (480, 640, 3)

EXPECTED_COLLISION_PATH_SUFFIXES = (
    "E_body_1/E_Group_2/P_1e5d57b88817687f",
    "E_body_1/E_Group_2/P_14642a3a360c107f",
    "E_body_1/E_Group_2/P_5ed99b1c4c47687f",
    "E_body_1/E_Group_2/touch_pad/touch_pad",
    "E_body_1/E_keyboard_3/P_8dde13f8c6c4547f",
    "E_body_1/E_keyboard_3/P_7a7cc6daf657687f",
    "E_body_1/E_keyboard_3/P_32dc8237d0f0687f",
    "E_body_1/E_foot_mat_4/P_c97dfd96654f687f",
    "E_displayer_5/screen/P_eeb48f6c1dbce87f",
    "E_displayer_5/lid/P_74e9628836e5b87f",
    "E_displayer_5/lid/P_1b5d16155170687f",
    "E_displayer_5/lid/P_a3edd5af057687f",
    "E_displayer_5/lid/P_d0d77947399b37bf",
)

SOURCE_TASK_PROPS_TO_DEACTIVATE = (
    "c10_power_strip",
    "c10_charger_plug",
    "c10_charging_cable",
)


def initial_fraction_of_travel() -> float:
    """Return the magnitude fraction of the authored hinge travel."""

    return abs(INITIAL_HINGE_DEG) / abs(HINGE_LOWER_DEG)
