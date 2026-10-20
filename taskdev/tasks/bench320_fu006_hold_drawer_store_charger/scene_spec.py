"""Pure-Python scene contract for B320-FU006.

This module deliberately imports no Isaac modules so asset provenance, scale,
pose, joint intent, and initial AABB separation can be tested on CPU.
"""

from __future__ import annotations

import hashlib
import math
from itertools import product
from pathlib import Path
from typing import Iterable


TASK_ID = "t46_store_charger"
TASK_TITLE = "B320-FU006 HoldDrawerStoreCharger"
TASK_COMPLETE = False
SCENE_SCHEMA = "b320_fu006_core_scene_v1"
PHYSICS_HZ = 120
CONTROL_HZ = 30
STABILITY_STEPS = 240
HEAD_SHAPE = (480, 640, 3)
TABLE_TOP_Z_M = 0.89

TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]

ROBOT_RELATIVE = Path("handson/handson/sim/envs/xmls/rokae_wuji1_bimanual.xml")
DRAWER_RELATIVE = Path(
    "taskdev/tasks/t46_store_charger/"
    "assets/dv_drawer_45575/45575.usd"
)
CHARGER_RELATIVE = Path(
    "taskdev/tasks/t46_store_charger/"
    "assets/dexverse_charger/model_charger.usd"
)

EXPECTED_SHA256 = {
    str(DRAWER_RELATIVE): "0ded92a0ac6e5aafc0b3b9dbe3cd2d50f295ab8de21bf46bc97ed39f2c3a5157",
    str(CHARGER_RELATIVE): "9f996dc4d29fcc9d9062f40fbb994f5e7daec12027b32f93c027be830bb41219",
    str(ROBOT_RELATIVE): "a425da3c71e547fc5f427fc172d0a18de1c51668b22d87ac6628ddba38bf6a63",
}

DRAWER_DEFAULT_PRIM = "/partnet_26dcf7a76c89d5aa4da5feafe6f1c8fc"
DRAWER_LOCAL_AABB_M = (
    (-0.6276660643383529, -0.38127199665462974, -0.8518780171924022),
    (0.5594600347489106, 0.4288990366499816, 0.7009500653388878),
)
DRAWER_SCALE = (0.35, 0.35, 0.35)
# Keep the validated PartNet articulation as the physical mechanism while the
# downloaded low cabinet supplies the visible shell.  Center it laterally and
# shift it 10 cm away from the robot, then lower the mechanism so the modern
# cabinet rests at the original lower-cabinet support height.
ORIGINAL_DRAWER_TRANSLATION_M = (-0.07, 0.0, 1.1931573060173408)
DRAWER_BACK_SHIFT_M = 0.10
ORIGINAL_LOWER_CABINET_BOTTOM_SOURCE_Z = -0.8518780171924022
DRAWER_SUPPORT_Z_M = (
    ORIGINAL_DRAWER_TRANSLATION_M[2]
    + DRAWER_SCALE[2] * ORIGINAL_LOWER_CABINET_BOTTOM_SOURCE_Z
)
DRAWER_TRANSLATION_M = (
    ORIGINAL_DRAWER_TRANSLATION_M[0] + DRAWER_BACK_SHIFT_M,
    ORIGINAL_DRAWER_TRANSLATION_M[1],
    DRAWER_SUPPORT_Z_M - DRAWER_SCALE[2] * 0.32,
)
DRAWER_YAW_DEG = 0.0
DRAWER_COLLISION_COUNT = 29
DRAWER_RIGID_BODY_COUNT = 4
DRAWER_MAIN_JOINT = "joints/joint_1"
DRAWER_MAIN_JOINT_TYPE = "PhysicsPrismaticJoint"
DRAWER_MAIN_JOINT_AXIS = "Z"
DRAWER_MAIN_JOINT_LIMITS_SOURCE_M = (0.0, 0.7760000228881836)
DRAWER_MAIN_JOINT_INITIAL_SOURCE_M = 0.28
DRAWER_MAIN_JOINT_INITIAL_WORLD_M = (
    DRAWER_MAIN_JOINT_INITIAL_SOURCE_M * DRAWER_SCALE[0]
)

CHARGER_LOCAL_AABB_M = (
    (-0.020983299240469933, -0.015187909826636314, -0.013504070229828358),
    (0.021016700193285942, 0.039905499666929245, 0.013500000350177288),
)
CHARGER_SCALE = (1.0, 1.0, 1.0)
CHARGER_TRANSLATION_M = (-0.37, -0.22, 0.9075040702298284)
CHARGER_YAW_DEG = -12.0

DRAWER_ROOT_PATH = "/World/FU006Core/Drawer"
CHARGER_ROOT_PATH = "/World/FU006Core/Charger"
HEAD_CAMERA_PATH = "/World/RecordCameras/head_camera"
LEGACY_TASK_BODY_NAMES = frozenset(
    {"c10_power_strip", "c10_charger_plug", "c10_charging_cable"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corners(aabb: tuple[tuple[float, ...], tuple[float, ...]]) -> Iterable[tuple[float, ...]]:
    low, high = aabb
    return product(*zip(low, high))


def transformed_aabb(
    aabb: tuple[tuple[float, ...], tuple[float, ...]],
    scale: tuple[float, float, float],
    translation: tuple[float, float, float],
    yaw_deg: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    yaw = math.radians(yaw_deg)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    points = []
    for x, y, z in _corners(aabb):
        x, y, z = x * scale[0], y * scale[1], z * scale[2]
        points.append(
            (
                cosine * x - sine * y + translation[0],
                sine * x + cosine * y + translation[1],
                z + translation[2],
            )
        )
    return tuple(min(point[i] for point in points) for i in range(3)), tuple(
        max(point[i] for point in points) for i in range(3)
    )


def aabb_separation_m(
    first: tuple[tuple[float, ...], tuple[float, ...]],
    second: tuple[tuple[float, ...], tuple[float, ...]],
) -> float:
    gaps = [
        max(second[0][axis] - first[1][axis], first[0][axis] - second[1][axis])
        for axis in range(3)
    ]
    return max(0.0, max(gaps))


def layout_contract() -> dict[str, object]:
    drawer = transformed_aabb(
        DRAWER_LOCAL_AABB_M, DRAWER_SCALE, DRAWER_TRANSLATION_M, DRAWER_YAW_DEG
    )
    charger = transformed_aabb(
        CHARGER_LOCAL_AABB_M,
        CHARGER_SCALE,
        CHARGER_TRANSLATION_M,
        CHARGER_YAW_DEG,
    )
    return {
        "drawer_world_aabb_m": drawer,
        "charger_world_aabb_m": charger,
        "drawer_table_clearance_m": drawer[0][2] - TABLE_TOP_Z_M,
        "charger_table_clearance_m": charger[0][2] - TABLE_TOP_Z_M,
        "drawer_charger_separation_m": aabb_separation_m(drawer, charger),
        "drawer_scaled_extent_m": tuple(drawer[1][i] - drawer[0][i] for i in range(3)),
        "drawer_opening_travel_m": DRAWER_MAIN_JOINT_INITIAL_WORLD_M,
    }


def validate_static_contract(repository_root: Path = REPOSITORY_ROOT) -> list[str]:
    errors: list[str] = []
    for relative, expected in EXPECTED_SHA256.items():
        path = repository_root / relative
        if not path.is_file():
            errors.append(f"missing asset: {relative}")
        elif sha256_file(path) != expected:
            errors.append(f"asset hash mismatch: {relative}")
    layout = layout_contract()
    if float(layout["drawer_table_clearance_m"]) < 0.0049:
        errors.append("drawer must start at least 4.9 mm above the tabletop")
    if float(layout["charger_table_clearance_m"]) < 0.0039:
        errors.append("charger must start at least 3.9 mm above the tabletop")
    if float(layout["drawer_charger_separation_m"]) < 0.03:
        errors.append("drawer and charger initial AABBs need 30 mm separation")
    if not (
        DRAWER_MAIN_JOINT_LIMITS_SOURCE_M[0]
        < DRAWER_MAIN_JOINT_INITIAL_SOURCE_M
        < DRAWER_MAIN_JOINT_LIMITS_SOURCE_M[1]
    ):
        errors.append("drawer initial position must be strictly inside its limits")
    return errors
