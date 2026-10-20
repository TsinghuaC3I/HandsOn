"""Immutable core-scene contract for B320-BC012.

The coordinates are reset-scene authoring data, not motion targets. No path,
grasp, controller, or success-action parameters live in this module.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Final


TASK_ID: Final = "B320-BC012"
RUNTIME_TASK_NAME: Final = "t13_load_cans"
AUTHORITATIVE_SOURCE_CATALOG: Final = (
    "/mnt/pfs/luotianwei/handson-taskdev-orchestration/"
    "b320_32_scene_wave_20260817/source_rows.json"
)
AUTHORITATIVE_SOURCE_CATALOG_SHA256: Final = (
    "c1bf0990e8ec0ae651ae6d904c6b61e45796c8e11a670f6da782f4b23052e987"
)
AUTHORITATIVE_EXCEL_ROW: Final = 173
TASK_DIR: Final = Path(__file__).resolve().parent
REPOSITORY_ROOT: Final = TASK_DIR.parents[2]
ASSET_DIR: Final = TASK_DIR / "assets"
CAN_USD: Final = ASSET_DIR / "tomato_soup_can" / "textured.usd"
CAN_MESH_USD: Final = ASSET_DIR / "tomato_soup_can" / "Props" / "instanceable_meshes.usd"
CAN_TEXTURE: Final = ASSET_DIR / "tomato_soup_can" / "textures" / "texture_map.png"
OPEN_BIN_GLB: Final = ASSET_DIR / "open_bin" / "dexverse_authored_open_bin.glb"
ROBOT_MJCF: Final = REPOSITORY_ROOT / "handson/handson/sim/envs/xmls/rokae_wuji1_bimanual.xml"

EXPECTED_SHA256: Final = {
    CAN_USD: "c54c63ab31dc0e369ccc27d368f0fa60d1e83c2d358ae207022d5a1317d30f26",
    CAN_MESH_USD: "1b8eb20b782d4907076b4cb927e5f88ea238c9229ad375ab2a616e564f2f4ed4",
    CAN_TEXTURE: "33183a894eb75ae04ae5188a3ae689fe8d06ec6b9f9624da1bf853c2a25e70d2",
    OPEN_BIN_GLB: "eabe5ab04f5c3ec2f80cd81e9deba260eeb35a671b0a31a931e8a7a62440f1e6",
    ROBOT_MJCF: "a425da3c71e547fc5f427fc172d0a18de1c51668b22d87ac6628ddba38bf6a63",
}

SCENE_ROOT: Final = "/World/BC012CoreScene"
HEAD_CAMERA_PRIM: Final = "/World/RecordCameras/head_camera"
TABLE_TOP_Z_M: Final = 0.89
CAN_ROOT_POSITIONS_M: Final = {
    "left": (-0.43, 0.28, TABLE_TOP_Z_M + 0.002),
    "right": (-0.43, -0.28, TABLE_TOP_Z_M + 0.002),
}
CAN_SOURCE_SCALE_XYZ: Final = (1.0, 1.0, 1.0)
CAN_SOURCE_ORIENTATION_XYZW: Final = (0.0, 0.0, 0.0, 1.0)
OPEN_BIN_ROOT_POSITION_M: Final = (-0.40, 0.0, TABLE_TOP_Z_M + 0.002)
OPEN_BIN_SOURCE_SCALE_XYZ: Final = (1.0, 1.0, 1.0)
OPEN_BIN_SOURCE_ORIENTATION_XYZW: Final = (
    math.sin(math.pi / 4.0),
    0.0,
    0.0,
    math.cos(math.pi / 4.0),
)
OPEN_BIN_SOURCE_BOUNDS_M: Final = (
    (-0.1899999976, 0.0, -0.1400000006),
    (0.1899999976, 0.1500000060, 0.1400000006),
)

# Collision panels after the exact GLB's Y-up -> Z-up +90-degree X rotation.
# Each value is (center_xyz_m, size_xyz_m) in the authored bin-root frame.
OPEN_BIN_COLLISION_PANELS: Final = {
    "bottom": ((0.0, 0.0, 0.0125), (0.38, 0.28, 0.025)),
    "front": ((0.0, -0.1275, 0.075), (0.38, 0.025, 0.15)),
    "back": ((0.0, 0.1275, 0.075), (0.38, 0.025, 0.15)),
    "left": ((-0.1775, 0.0, 0.075), (0.025, 0.23, 0.15)),
    "right": ((0.1775, 0.0, 0.075), (0.025, 0.23, 0.15)),
}
PARTITION_DIVIDER: Final = ((0.0, 0.0, 0.075), (0.36, 0.012, 0.14))
GOAL_ZONES: Final = {
    "left": ((0.0, 0.065, 0.035), (0.33, 0.095, 0.012)),
    "right": ((0.0, -0.065, 0.035), (0.33, 0.095, 0.012)),
}

PHYSICS_HZ: Final = 120
ZERO_ACTION_STEPS: Final = 3 * PHYSICS_HZ
HEAD_RESOLUTION_HW: Final = (480, 640)


def quaternion_norm(values: tuple[float, float, float, float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def usd_prim_identifier(value: str) -> str:
    """Return a conservative ASCII USD prim identifier."""

    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", str(value)) or "mesh"
    return sanitized if re.match(r"^[A-Za-z_]", sanitized) else f"_{sanitized}"


def validate_static_contract() -> list[str]:
    errors: list[str] = []
    for path in EXPECTED_SHA256:
        if not path.is_file():
            errors.append(f"required asset missing: {path.relative_to(REPOSITORY_ROOT)}")
    for name, quaternion in (
        ("can", CAN_SOURCE_ORIENTATION_XYZW),
        ("open_bin", OPEN_BIN_SOURCE_ORIENTATION_XYZW),
    ):
        if not math.isclose(quaternion_norm(quaternion), 1.0, abs_tol=1.0e-9):
            errors.append(f"{name} quaternion is not normalized")
    if CAN_ROOT_POSITIONS_M["left"] == CAN_ROOT_POSITIONS_M["right"]:
        errors.append("the two cans occupy the same reset pose")
    if len(OPEN_BIN_COLLISION_PANELS) != 5:
        errors.append("open bin must retain its five authored panels")
    if ZERO_ACTION_STEPS / PHYSICS_HZ < 3.0:
        errors.append("zero-action smoke is shorter than three simulated seconds")
    return errors
