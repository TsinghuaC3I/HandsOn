"""Immutable core-scene contract for B320-AS019.

This module is intentionally simulator independent.  Isaac-specific imports
live in :mod:`scene_runtime` and occur only after ``AppLauncher`` starts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TASK_ID = "B320-AS019"
TASK_NAME = "bench320_as019_screwdriver_safe_orientation_handover"
TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]

AUTHORITATIVE_SOURCE_ROWS_SHA256 = (
    "c1bf0990e8ec0ae651ae6d904c6b61e45796c8e11a670f6da782f4b23052e987"
)
AUTHORITATIVE_EXCEL_ROW = 260
AUTHORITATIVE_OBJECTIVE_ZH = (
    "将螺丝刀从右手交给左手，保持刀尖朝外且不触碰接收手掌。"
)
AUTHORITATIVE_LEFT_ROLE_ZH = "从刀柄中部接收并放入工具区"
AUTHORITATIVE_RIGHT_ROLE_ZH = "从刀柄尾部交付后完全释放"

ASSET_DIR_RELATIVE = Path(
    "taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/"
    "assets/polyhaven_flathead_screwdriver"
)
SOURCE_ARCHIVE_RELATIVE = (
    ASSET_DIR_RELATIVE / "source/flathead_screwdriver_4k.gltf.zip"
)
ASSET_MANIFEST_RELATIVE = ASSET_DIR_RELATIVE / "asset_manifest.json"
RUNTIME_USD_RELATIVE = ASSET_DIR_RELATIVE / "runtime/flathead_screwdriver.usd"
RUNTIME_TEXTURE_RELATIVE_BY_SOURCE_MEMBER = {
    f"textures/{name}": ASSET_DIR_RELATIVE / f"runtime/textures/{name}"
    for name in (
        "flathead_screwdriver_arm_4k.jpg",
        "flathead_screwdriver_diff_4k.jpg",
        "flathead_screwdriver_nor_gl_4k.jpg",
    )
}
RUNTIME_RIGID_OWNER = "/FlatheadScrewdriver"
RUNTIME_VISUAL_MESH_PATHS = (
    "/FlatheadScrewdriver/Visual/SourceMesh",
)
RUNTIME_PHYSICS_MATERIAL_PATH = "/FlatheadScrewdriver/Looks/PhysicsMaterial"
RUNTIME_COLLIDER_PATHS = tuple(
    f"/FlatheadScrewdriver/Colliders/{name}"
    for name in ("TailMain", "HandleNeck", "Ferrule", "Stem", "FlatBlade")
)
OVERLAY_RELATIVE = Path(
    "taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/"
    "scene/as019_core_scene.usda"
)
PREVIEW_RELATIVE = Path(
    "taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/"
    "previews/reset_head.png"
)
SUMMARY_RELATIVE = Path(
    "taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/"
    "evidence/core_scene_summary.json"
)
ZERO_ACTION_RELATIVE = Path(
    "taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/"
    "evidence/zero_action.json"
)
ROBOT_RELATIVE = Path("handson/handson/sim/envs/xmls/rokae_wuji1_bimanual.xml")

SOURCE_ARCHIVE_SHA256 = (
    "0162ab02ff2357585ebf702fc3844e4bf6313cf043239eaf763180ab12c26dbb"
)
SOURCE_ARCHIVE_BYTES = 27238057
SOURCE_MEMBER_SHA256 = {
    "flathead_screwdriver_4k.gltf": (
        "657eb5aab940361252ecd071ed3b8e5a1f159fd27b74417829792c43419ff0c1"
    ),
    "flathead_screwdriver.bin": (
        "d6cf744aa352c7ed92532c2aaa98e84baf2936361aa07025a6665a8d9ca0b9b0"
    ),
    "textures/flathead_screwdriver_arm_4k.jpg": (
        "97043e2b562a0acb7835d9da26259ad17abbbe820649b97c6458af1bb5998439"
    ),
    "textures/flathead_screwdriver_diff_4k.jpg": (
        "d3df707481cd6a14bbe0a075fb06abc9684316d0933a816682f7e935b586f4b1"
    ),
    "textures/flathead_screwdriver_nor_gl_4k.jpg": (
        "b85b2ca09f710358b5a8b037654b7b676db58b8e4f4d4538b30466583676c730"
    ),
}
# The reviewed Poly Haven mesh is uniformly normalized by 0.9950483267 and
# rotated from source +Y (tip) onto task-local +X.  That source-authored USD
# remains byte-identical; AS019 applies the declared 1.75x task transform below,
# producing a 455 mm tool while keeping visual and collision lineage explicit
# and source-derived.
SOURCE_TO_TASK_UNIFORM_SCALE = 0.9950483266936156
SOURCE_TO_TASK_TRANSLATION_M = (-0.0778213224721204, 0.025, 0.0)
SOURCE_LOCAL_BOUNDS_M = (
    (-0.125, 0.010059480492583918, -0.014940518580705114),
    (
        0.13500000163912773,
        0.039940519507416085,
        0.014940518580705114,
    ),
)
SOURCE_EXTENTS_M = (
    0.26000000163912773,
    0.029881039014832167,
    0.029881037161410228,
)

TABLE_CENTER_M = (-0.15, 0.0, 0.89)
TABLE_HALF_EXTENTS_M = (0.40, 0.80, 0.03)
TABLE_TOP_Z_M = 0.92

# Keep AS019 on the same mirrored home used by the native bimanual-assembly
# family and the ordinary dual-arm collection tasks.  A task-local raised pose
# made reset visually inconsistent and caused tracker commands to begin from a
# different frame than neighbouring tasks.
AS019_TABLETOP_ARM_HOME_RAD = {
    "right": (
        -1.5707963,
        1.5707963,
        1.5707963,
        1.5707963,
        -1.5707963,
        0.0,
        0.0,
    ),
    "left": (
        1.5707963,
        1.5707963,
        -1.5707963,
        1.5707963,
        1.5707963,
        0.0,
        0.0,
    ),
}
AS019_TABLETOP_TCP_HOME_WXYZ = {
    "right": (-0.10, -0.16, 1.07, 0.0, 1.0, 0.0, 0.0),
    "left": (-0.10, 0.16, 1.07, 0.0, 1.0, 0.0, 0.0),
}

# The authored handle occupies local -X and the shaft/tip local +X.  The head
# camera looks along world +X, so a +90-degree world-Z yaw presents the tool
# horizontally: the blunt tail maps to world -Y (screen right) and the sharp
# tip maps to world +Y (screen left).
SCREWDRIVER_UNIFORM_SCALE = 1.75
SCREWDRIVER_SCALE_XYZ = (SCREWDRIVER_UNIFORM_SCALE,) * 3
# The normalized handle has a real flat support face.  Author its first PhysX
# pose 0.2 mm above the table: close enough for the 1 mm tool contact offset
# (and the table's 2 mm offset) to support the sleeping object, without
# preserving the old asset's visible
# 12 mm hover or introducing initial penetration.
SCREWDRIVER_POSITION_M = (-0.145, -0.025, 0.9463459075162339)
SCREWDRIVER_ORIENTATION_WXYZ = (
    math.sqrt(0.5),
    0.0,
    0.0,
    math.sqrt(0.5),
)
SCREWDRIVER_SAFE_AXIS_WORLD = (0.0, 1.0, 0.0)
SCREWDRIVER_MASS_KG = 0.18
SCREWDRIVER_DEFAULT_FRICTION = 0.68
SCREWDRIVER_CONTACT_OFFSET_M = 0.001
SCREWDRIVER_REST_OFFSET_M = 0.0
# Source-derived convex hull volumes are weighted with 700 kg/m^3 wood and
# 7850 kg/m^3 steel, then uniformly normalized to the task's 0.18 kg nominal
# mass.  The committed asset manifest records the full per-part derivation.
SCREWDRIVER_CENTER_OF_MASS_M = (
    -0.03348301709027515,
    0.02500000005176284,
    0.0,
)
SCREWDRIVER_LINEAR_DAMPING = 0.08
SCREWDRIVER_ANGULAR_DAMPING = 2.0
# The production recorder validates a stable physical origin before any
# operator or automatic action is admitted.  Spawn the untouched tabletop
# tool asleep so convex-facet contact jitter cannot keep that gate open;
# normal hand collisions still wake the dynamic rigid body.
SCREWDRIVER_STARTS_ASLEEP = True
SCREWDRIVER_SLEEP_THRESHOLD = 0.005
SCREWDRIVER_STABILIZATION_THRESHOLD = 0.0025
SCREWDRIVER_SOLVER_POSITION_ITERATIONS = 8
SCREWDRIVER_SOLVER_VELOCITY_ITERATIONS = 1
SCREWDRIVER_COLLIDER_APPROXIMATION = "convexHull"
SCREWDRIVER_EXPECTED_COLLIDER_COUNT = 5

# Authoritative SceneSmith reset-envelope contract.  The single live core-scene
# audit uses the zero-offset canonical pose; randomized reset trials belong to
# the later motion/validation owner and must remain inside these bounds.
SCREWDRIVER_RANDOMIZATION_XY_M = 0.035
SCREWDRIVER_RANDOMIZATION_YAW_RAD = math.radians(12.0)
SCREWDRIVER_FRICTION_RANGE = (0.45, 0.85)
SCREWDRIVER_MASS_MULTIPLIER_RANGE = (0.8, 1.2)
MARKER_RANDOMIZATION_XY_M = 0.025

# A bright, non-physical strip communicates the safe longitudinal axis.
# It is parallel to the horizontal screwdriver and offset toward the robots in
# world -X, leaving a visible gap from the reset tool.  It must never
# contribute contacts or support the dynamic screwdriver.
MARKER_POSITION_M = (-0.300, -0.020, 0.924)
MARKER_ORIENTATION_WXYZ = SCREWDRIVER_ORIENTATION_WXYZ
MARKER_SIZE_M = (0.52, 0.14, 0.0015)
MARKER_SCALE_XYZ = (1.0, 1.0, 1.0)
MARKER_COLLISION_ENABLED = False

HEAD_CAMERA_PRIM_PATH = "/World/envs/env_0/HeadRecordCamera"
HEAD_CAMERA_RESOLUTION_HW = (480, 640)
HEAD_CAMERA_EYE_M = (-0.675, 0.0, 1.5885)
HEAD_CAMERA_TARGET_M = (-0.12, 0.0, 0.95)

PHYSICS_HZ = 120
CONTROL_HZ = 30
SETTLE_STEPS = 240
ZERO_ACTION_STEPS = 480
ZERO_ACTION_SECONDS = ZERO_ACTION_STEPS / PHYSICS_HZ
MAX_POSITION_DRIFT_M = 0.005
MAX_ORIENTATION_DRIFT_RAD = math.radians(2.0)
MAX_FINAL_LINEAR_SPEED_MPS = 0.02
MAX_FINAL_ANGULAR_SPEED_RADPS = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quaternion_rotation_matrix(
    orientation: tuple[float, float, float, float],
) -> np.ndarray:
    quaternion = np.asarray(orientation, dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def world_bounds(
    local_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = SOURCE_LOCAL_BOUNDS_M,
    position: tuple[float, float, float] = SCREWDRIVER_POSITION_M,
    orientation: tuple[float, float, float, float] = SCREWDRIVER_ORIENTATION_WXYZ,
    scale: tuple[float, float, float] = SCREWDRIVER_SCALE_XYZ,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the rotated world AABB for the authored reset pose."""

    low = np.asarray(local_bounds[0], dtype=np.float64)
    high = np.asarray(local_bounds[1], dtype=np.float64)
    translation = np.asarray(position, dtype=np.float64)
    corners = np.asarray(
        [
            (x, y, z)
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=np.float64,
    )
    corners *= np.asarray(scale, dtype=np.float64)
    world = corners @ _quaternion_rotation_matrix(orientation).T + translation
    return tuple(map(tuple, (world.min(axis=0), world.max(axis=0))))  # type: ignore[return-value]


def contract_errors(repository_root: Path = REPOSITORY_ROOT) -> tuple[str, ...]:
    errors: list[str] = []
    source = repository_root / SOURCE_ARCHIVE_RELATIVE
    if not source.is_file():
        errors.append(f"missing source archive: {SOURCE_ARCHIVE_RELATIVE}")
    else:
        if source.stat().st_size != SOURCE_ARCHIVE_BYTES:
            errors.append("source archive byte size mismatch")
        if sha256_file(source) != SOURCE_ARCHIVE_SHA256:
            errors.append("source archive SHA-256 mismatch")
    asset_manifest = repository_root / ASSET_MANIFEST_RELATIVE
    runtime_asset = repository_root / RUNTIME_USD_RELATIVE
    if not asset_manifest.is_file():
        errors.append(f"missing asset manifest: {ASSET_MANIFEST_RELATIVE}")
    if not runtime_asset.is_file():
        errors.append(f"missing runtime USD: {RUNTIME_USD_RELATIVE}")
    if SCREWDRIVER_SCALE_XYZ != (1.75, 1.75, 1.75):
        errors.append("screwdriver must use the requested 1.75x uniform task scale")
    if MARKER_SCALE_XYZ != (1.0, 1.0, 1.0):
        errors.append("marker must retain identity scale")
    if MARKER_COLLISION_ENABLED:
        errors.append("safe-orientation marker must be visual-only")
    quaternion = np.asarray(SCREWDRIVER_ORIENTATION_WXYZ, dtype=np.float64)
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1.0e-9):
        errors.append("screwdriver reset quaternion must be normalized")
    axis = np.asarray(SCREWDRIVER_SAFE_AXIS_WORLD, dtype=np.float64)
    reset_axis = _quaternion_rotation_matrix(SCREWDRIVER_ORIENTATION_WXYZ)[:, 0]
    if not np.allclose(axis, reset_axis, atol=1.0e-9):
        errors.append("safe tip axis must match the rotated local +X axis")
    if not np.allclose(-reset_axis, (0.0, -1.0, 0.0), atol=1.0e-9):
        errors.append("blunt handle tail must face screen-right world -Y")
    marker_quaternion = np.asarray(MARKER_ORIENTATION_WXYZ, dtype=np.float64)
    if not np.isclose(np.linalg.norm(marker_quaternion), 1.0, atol=1.0e-9):
        errors.append("marker reset quaternion must be normalized")
    marker_axis = _quaternion_rotation_matrix(MARKER_ORIENTATION_WXYZ)[:, 0]
    if not np.allclose(marker_axis, reset_axis, atol=1.0e-9):
        errors.append("tool-zone marker must remain parallel to the screwdriver")
    mirror_signs = np.asarray((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0))
    right_arm_home = np.asarray(AS019_TABLETOP_ARM_HOME_RAD["right"])
    left_arm_home = np.asarray(AS019_TABLETOP_ARM_HOME_RAD["left"])
    if (
        right_arm_home.shape != (7,)
        or left_arm_home.shape != (7,)
        or not np.all(np.isfinite(right_arm_home))
        or not np.all(np.isfinite(left_arm_home))
    ):
        errors.append("AS019 tabletop arm homes must be finite mirrored 7D poses")
    elif not np.allclose(
        left_arm_home,
        mirror_signs * right_arm_home,
        atol=1.0e-12,
        rtol=0.0,
    ):
        errors.append("AS019 tabletop arm homes must remain exactly mirrored")
    for side, expected_y_sign in (("right", -1.0), ("left", 1.0)):
        palm_home = np.asarray(
            AS019_TABLETOP_TCP_HOME_WXYZ[side], dtype=np.float64
        )
        if palm_home.shape != (7,) or not np.all(np.isfinite(palm_home)):
            errors.append(f"AS019 {side} tabletop TCP home must be finite 7D")
            continue
        if not np.isclose(np.linalg.norm(palm_home[3:7]), 1.0, atol=1.0e-9):
            errors.append(f"AS019 {side} tabletop TCP quaternion must be normalized")
        palm_rotation = _quaternion_rotation_matrix(palm_home[3:7])
        if not np.allclose(palm_rotation[:, 0], (1.0, 0.0, 0.0), atol=1.0e-9):
            errors.append(f"AS019 {side} shared-home palm axis must face world +X")
        if not np.allclose(palm_rotation[:, 2], (0.0, 0.0, -1.0), atol=1.0e-9):
            errors.append(f"AS019 {side} shared-home tool axis must face downward")
        if expected_y_sign * palm_home[1] <= 0.0 or palm_home[2] <= TABLE_TOP_Z_M:
            errors.append(f"AS019 {side} tabletop TCP home is outside its safe side")
    low, high = map(np.asarray, world_bounds())
    if low[2] <= TABLE_TOP_Z_M:
        errors.append("screwdriver reset penetrates the table")
    elif low[2] - TABLE_TOP_Z_M > 0.001:
        errors.append("screwdriver reset is visibly unsupported above the table")
    if not (
        TABLE_CENTER_M[0] - TABLE_HALF_EXTENTS_M[0] <= low[0] <= high[0] <= TABLE_CENTER_M[0] + TABLE_HALF_EXTENTS_M[0]
        and TABLE_CENTER_M[1] - TABLE_HALF_EXTENTS_M[1] <= low[1] <= high[1] <= TABLE_CENTER_M[1] + TABLE_HALF_EXTENTS_M[1]
    ):
        errors.append("screwdriver reset bounds exceed tabletop support")
    marker_low = np.asarray(MARKER_POSITION_M) - 0.5 * np.asarray(MARKER_SIZE_M)
    if marker_low[2] < TABLE_TOP_Z_M - 1.0e-6:
        errors.append("orientation marker penetrates the tabletop")
    if ZERO_ACTION_SECONDS < 3.0:
        errors.append("zero-action duration must be at least three simulated seconds")
    if AUTHORITATIVE_EXCEL_ROW != 260:
        errors.append("authoritative source-row binding changed")
    return tuple(errors)


def frame_metrics(frame: np.ndarray) -> dict[str, Any]:
    array = np.asarray(frame)
    shape_ok = tuple(array.shape) == (*HEAD_CAMERA_RESOLUTION_HW, 3)
    dtype_ok = array.dtype == np.uint8
    finite = bool(np.all(np.isfinite(array)))
    if not shape_ok or not dtype_ok or not finite:
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": finite,
            "mean_rgb": 0.0,
            "std_rgb": 0.0,
            "black_fraction": 1.0,
            "unique_rgb_count": 0,
            "pass": False,
        }
    numeric = array.astype(np.float32, copy=False)
    unique = int(np.unique(array.reshape(-1, 3), axis=0).shape[0])
    metrics = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": finite,
        "mean_rgb": float(np.mean(numeric)),
        "std_rgb": float(np.std(numeric)),
        "black_fraction": float(np.mean(np.all(array <= 10, axis=2))),
        "unique_rgb_count": unique,
    }
    metrics["pass"] = bool(
        metrics["mean_rgb"] > 30.0
        and metrics["std_rgb"] > 15.0
        and metrics["black_fraction"] < 0.90
        and unique >= 128
    )
    return metrics


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value
